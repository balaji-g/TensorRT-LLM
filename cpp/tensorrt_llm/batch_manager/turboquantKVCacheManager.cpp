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
    TLLM_LOG_INFO("TurboquantKVCacheManager constructed (bits=%d). Pool sizing override is not yet active in this build "
                  "(B.1 stub); allocation falls through to the parent.",
        mTurboquantBits);
}

void TurboquantKVCacheManager::allocatePools(bool useUvm)
{
    // B.1 stub: identity override. The full implementation that shrinks
    // the primary pool to packed-bits + norms layout requires the
    // parent to expose a mutable BlockManager accessor (or a per-pool
    // quantSize plumb-through) — that small upstream-PR-sized change
    // lands in the follow-up commit. For now we forward so the class
    // is linkable and the runtime selection plumbing (B.3) can be
    // wired without depending on the sizing override being complete.
    TLLM_LOG_INFO("TurboquantKVCacheManager::allocatePools (bits=%d, useUvm=%d) — forwarding to parent",
        mTurboquantBits, static_cast<int>(useUvm));
    KVCacheManager::allocatePools(useUvm);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
