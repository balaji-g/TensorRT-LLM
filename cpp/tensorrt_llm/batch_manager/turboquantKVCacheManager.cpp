/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/batch_manager/turboquantKVCacheManager.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

namespace
{
// Bytes per element for the underlying TRT data type. The persistent
// pool is allocated as `mDataType`-typed buffer of length blockSize, so
// the physical byte budget per (pool, block, layer, K-or-V) is
// `blockSize * sizeOfDtype(mDataType)`. We size blockSize to fit our
// packed bytes + inline fp32 norms in that physical budget.
SizeType32 sizeOfDtype(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    default:
        TLLM_THROW("TurboquantKVCacheManager: unsupported pool dtype %d", static_cast<int>(dtype));
    }
}
} // namespace

TurboquantKVCacheManager::TurboquantKVCacheManager(SizeType32 turboquantBits,
    std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
    SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, CudaStreamPtr stream, std::optional<SizeType32> maxSequenceLength,
    bool enableBlockReuse, bool onboardBlocks, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse)
    : KVCacheManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksInPrimaryPool, blocksInSecondaryPool,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, tempAttentionWindowInputs, dtype, sinkTokenLength,
        std::move(stream), maxSequenceLength, enableBlockReuse, onboardBlocks, cacheType, secondaryOffloadMinPriority,
        std::move(eventManager), enableHashKey, enablePartialReuse, copyOnPartialReuse)
    , mTurboquantBits(turboquantBits)
{
    TLLM_CHECK_WITH_INFO(turboquantBits == 4 || turboquantBits == 8,
        "TurboquantKVCacheManager: turboquantBits must be 4 or 8, got %d", turboquantBits);
    TLLM_LOG_INFO("TurboquantKVCacheManager constructed (bits=%d). Pool sizing override active in allocatePools.",
        mTurboquantBits);
}

void TurboquantKVCacheManager::allocatePools(bool useUvm)
{
    // Phase B sizing override: shrink each pool's blockSize so the
    // allocation in WindowBlockManager::allocatePools (which uses the
    // pool's stored blockSize as the element count) drops to packed
    // bytes + an inline fp32 norm per (token, head) slot.
    //
    // For a pool with numKvHeads, sizePerHead, tokensPerBlock:
    //   raw_packed_bytes = numKvHeads * sizePerHead * tokensPerBlock * bits / 8
    //   norms_bytes      = tokensPerBlock * numKvHeads * sizeof(float)
    //   total_bytes      = raw_packed_bytes + norms_bytes
    //   blockSize_elems  = ceil(total_bytes / sizeOfDtype(mDataType))
    //
    // The parent's allocatePools then allocates a tensor of shape
    // {numPrimaryBlocks, numLayers, kvFactor, blockSize_elems} typed
    // as mDataType — same total byte budget, but compressed.
    //
    // The TurboquantAttentionPlugin (M12.4 B.2) writes packed bytes +
    // norms into this layout via K3 and reads them back via K6.

    auto& blockManager = getMutableBlockManager();
    auto const numPools = blockManager.getNumPools(/*includeBlockScalePools=*/true);
    for (SizeType32 i = 0; i < numPools; ++i)
    {
        auto& pool = blockManager.getMutablePool(i);
        if (pool.containsBlockScales)
        {
            // FP4 block-scale pools are not on our path; leave unchanged.
            continue;
        }
        // mDataType lives on the WindowBlockManager; the pool itself
        // doesn't carry it, but the parent's allocatePools uses the
        // owning WindowBlockManager's mDataType. We use the pool's own
        // {numKvHeads, sizePerHead, tokensPerBlock} since those don't
        // vary.
        // Use the per-pool numLayers/kvFactor only for byte budget
        // sanity; the block-level layout is per (block, layer, K-or-V)
        // slot which is `numKvHeads * sizePerHead * tokensPerBlock`
        // raw elements.
        SizeType32 const numKvHeads = pool.numKvHeads;
        SizeType32 const sizePerHead = pool.sizePerHead;
        SizeType32 const tokensPerBlock = pool.tokensPerBlock;

        SizeType32 const rawElements = numKvHeads * sizePerHead * tokensPerBlock;
        SizeType32 const packedBytes = (rawElements * mTurboquantBits + 7) / 8;
        SizeType32 const normsBytes = tokensPerBlock * numKvHeads * static_cast<SizeType32>(sizeof(float));
        SizeType32 const totalBytes = packedBytes + normsBytes;

        // The owning WindowBlockManager's dtype determines pool dtype.
        // Look it up via the const-getPool path on the same window;
        // dtype is window-uniform.
        auto const dtype = blockManager.getDataType();
        SizeType32 const dtypeBytes = sizeOfDtype(dtype);

        SizeType32 const oldBlockSize = pool.blockSize;
        SizeType32 const newBlockSize = (totalBytes + dtypeBytes - 1) / dtypeBytes;

        TLLM_LOG_INFO(
            "TurboquantKVCacheManager: pool[%d] resize bits=%d raw=%d packed=%dB norms=%dB total=%dB "
            "blockSize %d->%d (%.2fx compression vs fp16-shaped)",
            i, mTurboquantBits, rawElements, packedBytes, normsBytes, totalBytes, oldBlockSize, newBlockSize,
            static_cast<float>(oldBlockSize) / static_cast<float>(newBlockSize));

        pool.blockSize = newBlockSize;
    }

    KVCacheManager::allocatePools(useUvm);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
