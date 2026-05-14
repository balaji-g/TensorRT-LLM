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
#pragma once

#include "tensorrt_llm/batch_manager/kvCacheManager.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

// TurboquantKVCacheManager: KVCacheManager subclass that allocates a
// **compressed** persistent KV pool sized for TurboQuant's packed bits
// (4 or 8) plus an inline fp32 norm per (token, head) slot.
//
// Compression factor vs the fp16 baseline pool (Llama-3 reference shape:
// d_head=128, tokensPerBlock=16, numKvHeads=8):
//   bits=8 : 1.94x smaller  (16896 / 32768 bytes per block / layer / K)
//   bits=4 : 3.76x smaller  ( 8704 / 32768 bytes per block / layer / K)
//
// Pairs with TurboquantAttentionPlugin (cpp/tensorrt_llm/plugins/
// turboquantAttentionPlugin/): the plugin's enqueue calls K3 (paged
// quantize) on the cache write path so the bytes that hit this pool
// are the packed indices + norms layout this manager allocates for.
//
// Customer-facing selection: enable via `PluginConfig
// .turboquant_attention_plugin = "float16"` + `turboquant_bits ∈ {4,
// 8}` at engine-build time; the runtime instantiates this subclass
// instead of plain KVCacheManager when the engine declares the
// turboquant plugin.
class TurboquantKVCacheManager : public KVCacheManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    // Constructor mirrors the primary KVCacheManager constructor exactly,
    // with one extra leading parameter for the turboquant bit-depth.
    // turboquantBits must be 4 or 8.
    TurboquantKVCacheManager(SizeType32 turboquantBits, std::vector<SizeType32> const& numKvHeadsPerLayer,
        SizeType32 sizePerHead, SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool,
        SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec,
        std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
        SizeType32 sinkTokenLength, CudaStreamPtr stream, std::optional<SizeType32> maxSequenceLength,
        bool enableBlockReuse = false, bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enableHashKey = false,
        bool enablePartialReuse = true, bool copyOnPartialReuse = true);

    ~TurboquantKVCacheManager() override = default;

    // Override allocatePools to allocate a smaller, packed-layout primary
    // pool. The fp16-shaped sizing inherited from KVCacheBlockPool is
    // replaced by `(packed_bytes_per_slot + 4) * num_slots_per_block`
    // where packed_bytes_per_slot = sizePerHead * turboquantBits / 8.
    //
    // Phase B v1 (this commit): identity override — calls the parent's
    // allocatePools so the class compiles and links cleanly. The real
    // sizing logic lands in the follow-up commit once the protected
    // accessor for mBlockManager + mPools is added to the parent.
    void allocatePools(bool useUvm = false) override;

    [[nodiscard]] SizeType32 getTurboquantBits() const noexcept
    {
        return mTurboquantBits;
    }

private:
    SizeType32 mTurboquantBits;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
