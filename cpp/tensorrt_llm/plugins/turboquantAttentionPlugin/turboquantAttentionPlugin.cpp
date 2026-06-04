/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "turboquantAttentionPlugin.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/turboquant/turboquantConstants.h"
#include "tensorrt_llm/kernels/turboquant/turboquantStreaming.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/plugins/common/plugin.h"

#include <atomic>
#include <cstring>
#include <cuda_runtime.h>

namespace tensorrt_llm::plugins
{

namespace
{
constexpr char const* TURBOQUANT_ATTENTION_PLUGIN_NAME = "TurboquantAttention";
constexpr char const* TURBOQUANT_ATTENTION_PLUGIN_VERSION = "1";

// d_head is locked to 128 for the v1 demo (Llama-3 / DeepSeek-7B /
// Mistral all use 128). Other head sizes require a different
// Hadamard butterfly and different codebook constants — out of
// scope for B.2.1b.
constexpr int kDHead = 128;
constexpr int kDtypeFP16 = 0; // matches tq_dtype::TQ_DTYPE_FP16

void validateBits(int bits)
{
    TLLM_CHECK_WITH_INFO(bits == 4 || bits == 8,
        "TurboquantAttentionPlugin: turboquantBits must be 4 or 8, got %d", bits);
}

// Device-resident quantizer state. Lazy initialized on first enqueue.
// Shared across all plugin instances since the math is deterministic
// for (d_head=128, bits, seed=0). Process-wide singleton; freed at
// atexit / process teardown.
struct DeviceQuantState
{
    int8_t* signs{nullptr};
    float* centroids{nullptr};
    float* thresholds{nullptr};
    int bits{0};

    bool initOnce(int b)
    {
        if (signs != nullptr && bits == b)
        {
            return true;
        }
        if (signs != nullptr && bits != b)
        {
            // Re-init for a different bit-depth. Free old.
            if (signs)
            {
                cudaFree(signs);
                signs = nullptr;
            }
            if (centroids)
            {
                cudaFree(centroids);
                centroids = nullptr;
            }
            if (thresholds)
            {
                cudaFree(thresholds);
                thresholds = nullptr;
            }
        }
        int8_t const* hostSigns = nullptr;
        float const* hostCentroids = nullptr;
        float const* hostThresholds = nullptr;
        int nCentroids = 0;
        int nThresholds = 0;
        if (b == 8)
        {
            hostSigns = turboquant::constants::b8_d128::kSigns;
            hostCentroids = turboquant::constants::b8_d128::kCentroids;
            hostThresholds = turboquant::constants::b8_d128::kThresholds;
            nCentroids = 256;
            nThresholds = 255;
        }
        else
        {
            hostSigns = turboquant::constants::b4_d128::kSigns;
            hostCentroids = turboquant::constants::b4_d128::kCentroids;
            hostThresholds = turboquant::constants::b4_d128::kThresholds;
            nCentroids = 16;
            nThresholds = 15;
        }
        if (cudaMalloc(&signs, kDHead) != cudaSuccess)
            return false;
        if (cudaMalloc(&centroids, nCentroids * sizeof(float)) != cudaSuccess)
            return false;
        if (cudaMalloc(&thresholds, nThresholds * sizeof(float)) != cudaSuccess)
            return false;
        cudaMemcpy(signs, hostSigns, kDHead, cudaMemcpyHostToDevice);
        cudaMemcpy(centroids, hostCentroids, nCentroids * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(thresholds, hostThresholds, nThresholds * sizeof(float), cudaMemcpyHostToDevice);
        bits = b;
        TLLM_LOG_INFO("TurboquantAttentionPlugin: DeviceQuantState initialised (d_head=%d, bits=%d)", kDHead, bits);
        return true;
    }
};

DeviceQuantState gQuantState;

} // namespace

void TurboquantEnqueueWorkspace::free()
{
    if (packed)
    {
        cudaFree(packed);
        packed = nullptr;
    }
    if (norms)
    {
        cudaFree(norms);
        norms = nullptr;
    }
    if (kv_out)
    {
        cudaFree(kv_out);
        kv_out = nullptr;
    }
    packed_capacity = 0;
    norms_capacity = 0;
    kv_out_capacity = 0;
}

namespace
{
bool ensureWorkspace(TurboquantEnqueueWorkspace& ws, int nTokens, int nHeads, int dHead, int bits)
{
    std::size_t packedBytes = static_cast<std::size_t>(nTokens) * nHeads * dHead * bits / 8;
    std::size_t normsBytes = static_cast<std::size_t>(nTokens) * nHeads * sizeof(float);
    std::size_t kvBytes = static_cast<std::size_t>(nTokens) * nHeads * dHead * sizeof(std::uint16_t); // fp16

    if (packedBytes > ws.packed_capacity)
    {
        if (ws.packed)
            cudaFree(ws.packed);
        if (cudaMalloc(&ws.packed, packedBytes) != cudaSuccess)
            return false;
        ws.packed_capacity = packedBytes;
    }
    if (normsBytes > ws.norms_capacity)
    {
        if (ws.norms)
            cudaFree(ws.norms);
        if (cudaMalloc(reinterpret_cast<void**>(&ws.norms), normsBytes) != cudaSuccess)
            return false;
        ws.norms_capacity = normsBytes;
    }
    if (kvBytes > ws.kv_out_capacity)
    {
        if (ws.kv_out)
            cudaFree(ws.kv_out);
        if (cudaMalloc(&ws.kv_out, kvBytes) != cudaSuccess)
            return false;
        ws.kv_out_capacity = kvBytes;
    }
    return true;
}
} // namespace

TurboquantAttentionPlugin::TurboquantAttentionPlugin(int layer_idx, int num_heads, int vision_start, int vision_length,
    int num_kv_heads, int num_kv_heads_origin, int head_size, int unidirectional, float q_scaling,
    float attn_logit_softcapping_scale, tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
    int rotary_embedding_dim, float rotary_embedding_base,
    tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type, float rotary_embedding_scale,
    float rotary_embedding_short_m_scale, float rotary_embedding_long_m_scale, int rotary_embedding_max_positions,
    int rotary_embedding_original_max_positions, int tp_size, int tp_rank, bool unfuse_qkv_gemm, bool use_logn_scaling,
    tensorrt_llm::kernels::ContextFMHAType context_fmha_type, int kv_cache_quant_mode, bool remove_input_padding,
    tensorrt_llm::kernels::AttentionMaskType mask_type, tensorrt_llm::kernels::BlockSparseParams block_sparse_params,
    bool paged_kv_cache, int tokens_per_block, nvinfer1::DataType type, int32_t max_context_length,
    bool qkv_bias_enabled, bool cross_attention, int max_distance, bool pos_shift_enabled, bool dense_context_fmha,
    bool use_paged_context_fmha, bool use_fp8_context_fmha, bool has_full_attention_mask, bool use_cache,
    bool is_spec_decoding_enabled, bool spec_decoding_is_generation_length_variable,
    int32_t spec_decoding_max_generation_length, bool is_mla_enabled, int q_lora_rank, int kv_lora_rank,
    int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim, bool fuse_fp4_quant, bool skip_attn, int cp_size,
    int cp_rank, std::set<int32_t> cp_group, int turboquantBits)
    : GPTAttentionPlugin(layer_idx, num_heads, vision_start, vision_length, num_kv_heads, num_kv_heads_origin,
        head_size, unidirectional, q_scaling, attn_logit_softcapping_scale, position_embedding_type,
        rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale_type, rotary_embedding_scale,
        rotary_embedding_short_m_scale, rotary_embedding_long_m_scale, rotary_embedding_max_positions,
        rotary_embedding_original_max_positions, tp_size, tp_rank, unfuse_qkv_gemm, use_logn_scaling, context_fmha_type,
        kv_cache_quant_mode, remove_input_padding, mask_type, block_sparse_params, paged_kv_cache, tokens_per_block,
        type, max_context_length, qkv_bias_enabled, cross_attention, max_distance, pos_shift_enabled,
        dense_context_fmha, use_paged_context_fmha, use_fp8_context_fmha, has_full_attention_mask, use_cache,
        is_spec_decoding_enabled, spec_decoding_is_generation_length_variable, spec_decoding_max_generation_length,
        is_mla_enabled, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, fuse_fp4_quant,
        skip_attn, cp_size, cp_rank, cp_group)
    , mTurboquantBits(turboquantBits)
{
    validateBits(turboquantBits);
    TLLM_LOG_INFO(
        "TurboquantAttentionPlugin constructed (layer=%d, bits=%d). K1+K2 streaming round-trip is wired on "
        "enqueue; persistent cache stays fp16-shaped (B.3b adds packed pool selection).",
        layer_idx, mTurboquantBits);
}

TurboquantAttentionPlugin::TurboquantAttentionPlugin(void const* data, size_t length)
    : GPTAttentionPlugin(data, length - sizeof(int))
    , mTurboquantBits(0)
{
    auto const* tail = static_cast<char const*>(data) + length - sizeof(int);
    std::memcpy(&mTurboquantBits, tail, sizeof(int));
    validateBits(mTurboquantBits);
}

int TurboquantAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // K1 + K2 streaming round-trip on the fused QKV tensor (inputs[0]).
    // Treats the whole QKV as [n_tokens, n_total_heads, d_head] — Q heads
    // get quantized too, which adds bounded drift but doesn't change
    // attention math correctness. B.2.1c carves Q out.
    //
    // Persistent KV cache (the inner GPTAttentionPlugin's write into the
    // KVCacheManager-allocated pool) sees the round-tripped K/V — so the
    // cache holds post-compression numerics, exactly like Phase A
    // (M12.3) but in-tree.
    auto const& qkvDesc = inputDesc[0];
    if (qkvDesc.dims.nbDims < 2 || qkvDesc.type != nvinfer1::DataType::kHALF)
    {
        // Not fp16 / not a 2D-or-higher tensor — bail to parent.
        return GPTAttentionPlugin::enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }

    int qkvDim = qkvDesc.dims.d[qkvDesc.dims.nbDims - 1];
    int nTokens = 1;
    for (int i = 0; i < qkvDesc.dims.nbDims - 1; ++i)
    {
        nTokens *= qkvDesc.dims.d[i];
    }
    if (qkvDim % kDHead != 0)
    {
        return GPTAttentionPlugin::enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    int nTotalHeads = qkvDim / kDHead;

    if (!gQuantState.initOnce(mTurboquantBits))
    {
        TLLM_LOG_ERROR("TurboquantAttentionPlugin::enqueue: DeviceQuantState init failed (bits=%d)", mTurboquantBits);
        return -1;
    }

    if (!ensureWorkspace(mWorkspace, nTokens, nTotalHeads, kDHead, mTurboquantBits))
    {
        TLLM_LOG_ERROR("TurboquantAttentionPlugin::enqueue: workspace alloc failed (nTokens=%d, nHeads=%d)", nTokens,
            nTotalHeads);
        return -1;
    }

    int err = tq_kv_quantize_streaming(inputs[0], gQuantState.signs, gQuantState.centroids, gQuantState.thresholds,
        mWorkspace.packed, mWorkspace.norms, mTurboquantBits, kDtypeFP16, nTokens, nTotalHeads, kDHead, stream);
    if (err != 0)
    {
        TLLM_LOG_ERROR("TurboquantAttentionPlugin: tq_kv_quantize_streaming returned %d", err);
        return -1;
    }
    err = tq_kv_dequantize_streaming(mWorkspace.packed, mWorkspace.norms, gQuantState.signs, gQuantState.centroids,
        mWorkspace.kv_out, mTurboquantBits, kDtypeFP16, nTokens, nTotalHeads, kDHead, stream);
    if (err != 0)
    {
        TLLM_LOG_ERROR("TurboquantAttentionPlugin: tq_kv_dequantize_streaming returned %d", err);
        return -1;
    }

    static std::atomic<bool> sLogged{false};
    bool expected = false;
    if (sLogged.compare_exchange_strong(expected, true))
    {
        TLLM_LOG_INFO(
            "TurboquantAttentionPlugin::enqueue#1 — K1+K2 round-trip applied: bits=%d nTokens=%d nTotalHeads=%d "
            "d_head=%d",
            mTurboquantBits, nTokens, nTotalHeads, kDHead);

        // B.2.2 sanity ping — K3 (paged write) + K6 (paged read) with
        // TQ_LAYOUT_VLLM_BLOCKED_INLINE_NORMS. Allocates a small
        // shadow pool sized for nTokens slots (heuristic block_size=32),
        // packs the QKV via K3 then reads back via K6. Just verifies
        // the kernel API is wired and the inline-norms layout doesn't
        // segfault. Byte-equality check vs K1+K2 is the next-session
        // milestone. No HBM savings here — shadow pool is freed at the
        // end of this once-only block.
        constexpr int kTokensPerBlock = 32;
        int const nBlocks = (nTokens + kTokensPerBlock - 1) / kTokensPerBlock;
        int const packedBytesPerHead = kDHead * mTurboquantBits / 8;
        int const packedSectionBytes = nTotalHeads * kTokensPerBlock * packedBytesPerHead;
        int const normsSectionBytes = nTotalHeads * kTokensPerBlock * static_cast<int>(sizeof(float));
        int const totalBlockBytes = packedSectionBytes + normsSectionBytes;
        std::size_t const poolBytes = static_cast<std::size_t>(nBlocks) * totalBlockBytes;

        void* shadowPool = nullptr;
        int32_t* slotMapping = nullptr;
        int32_t* physBlocks = nullptr;
        void* k6Scratch = nullptr;
        std::size_t const k6ScratchBytes = static_cast<std::size_t>(nBlocks) * nTotalHeads * kTokensPerBlock * kDHead
            * sizeof(std::uint16_t);
        bool sanityOk = true;
        if (cudaMalloc(&shadowPool, poolBytes) != cudaSuccess)
            sanityOk = false;
        if (sanityOk && cudaMalloc(&slotMapping, nTokens * sizeof(int32_t)) != cudaSuccess)
            sanityOk = false;
        if (sanityOk && cudaMalloc(&physBlocks, nBlocks * sizeof(int32_t)) != cudaSuccess)
            sanityOk = false;
        if (sanityOk && cudaMalloc(&k6Scratch, k6ScratchBytes) != cudaSuccess)
            sanityOk = false;

        if (sanityOk)
        {
            cudaMemsetAsync(shadowPool, 0, poolBytes, stream);
            // Identity slot mapping: token i → slot i in block i/32.
            std::vector<int32_t> hSlot(nTokens);
            for (int i = 0; i < nTokens; ++i)
                hSlot[i] = i;
            cudaMemcpyAsync(slotMapping, hSlot.data(), nTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            std::vector<int32_t> hPhys(nBlocks);
            for (int i = 0; i < nBlocks; ++i)
                hPhys[i] = i;
            cudaMemcpyAsync(physBlocks, hPhys.data(), nBlocks * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

            // Layered launcher with n_layers_per_pool=1, kv_factor=1
            // degenerates to single-layer inline-norms. Strides match
            // what TQ_LAYOUT_VLLM_BLOCKED_INLINE_NORMS produces, so this
            // is equivalent to the prior sanity ping but exercises the
            // layered code path that the real-pool integration will use.
            int rc3 = tq_kv_quantize_paged_trtllm_layered(inputs[0], slotMapping, shadowPool, gQuantState.signs,
                gQuantState.centroids, gQuantState.thresholds, mTurboquantBits, kDtypeFP16, nTokens, nTotalHeads, kDHead,
                kTokensPerBlock, /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0, /*k_or_v=*/0, stream);
            int rc6 = tq_kv_dequantize_paged_trtllm_layered(shadowPool, physBlocks, gQuantState.signs,
                gQuantState.centroids, k6Scratch, mTurboquantBits, kDtypeFP16, nBlocks, nTotalHeads, kDHead,
                kTokensPerBlock, /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0, /*k_or_v=*/0, stream);
            cudaError_t syncErr = cudaStreamSynchronize(stream);
            TLLM_LOG_INFO(
                "[B.2.2] K3+K6 layered sanity (nLayers=1, kvFactor=1): rc3=%d rc6=%d cudaSync=%d "
                "(nBlocks=%d, poolBytes=%zu, k6Scratch=%zu)",
                rc3, rc6, static_cast<int>(syncErr), nBlocks, poolBytes, k6ScratchBytes);
        }
        else
        {
            TLLM_LOG_ERROR("[B.2.2] K3+K6 sanity: cudaMalloc failed");
        }
        if (shadowPool)
            cudaFree(shadowPool);
        if (slotMapping)
            cudaFree(slotMapping);
        if (physBlocks)
            cudaFree(physBlocks);
        if (k6Scratch)
            cudaFree(k6Scratch);

        // B.2.2 step 7 — call K3 layered against the REAL persistent
        // KV pool (HOST_KV_CACHE_POOL_POINTERS) for this layer's K
        // slice of the new tokens. The K3 write is ephemeral: the
        // inner GPTAttention's enqueue overwrites this slot's bytes
        // with its fp16 K immediately after we forward. No K6 read,
        // no pool-ptr patch yet — just validates the production-shape
        // K3 invocation succeeds. K1+K2 streaming stays the active
        // math hook so generation continues unchanged.
        //
        // Assumes Llama-3-style prefill of a single sequence at past
        // length 0 (matches the smoke). General multi-seq + generation
        // is the follow-up commit.
        if (isEntryUsed(IdxEntry::HOST_KV_CACHE_POOL_POINTERS)
            && isEntryUsed(IdxEntry::HOST_KV_CACHE_POOL_MAPPING)
            && isEntryUsed(IdxEntry::HOST_KV_CACHE_BLOCK_OFFSETS))
        {
            auto mappingIdx = getIdx(IdxEntry::HOST_KV_CACHE_POOL_MAPPING);
            auto poolPtrsIdx = getIdx(IdxEntry::HOST_KV_CACHE_POOL_POINTERS);
            auto hBlockOffsetsIdx = getIdx(IdxEntry::HOST_KV_CACHE_BLOCK_OFFSETS);
            int32_t const* hMapping = static_cast<int32_t const*>(inputs[mappingIdx]);
            int64_t const* hPoolPtrs = static_cast<int64_t const*>(inputs[poolPtrsIdx]);
            int32_t const* hBlockOffsets = static_cast<int32_t const*>(inputs[hBlockOffsetsIdx]);
            int nLayersPerPool = inputDesc[mappingIdx].dims.d[0];
            int poolIdxForLayer = hMapping[this->mLayerIdx * 2 + 0];
            int relativeLayer = hMapping[this->mLayerIdx * 2 + 1];
            int nPools = inputDesc[poolPtrsIdx].dims.d[0];
            int kvFactor = 2;
            // pool_ptrs shape [n_pools, 2]; primary at index 0.
            std::uintptr_t poolRaw
                = static_cast<std::uintptr_t>(hPoolPtrs[poolIdxForLayer * 2 + 0]);
            void* poolBase = reinterpret_cast<void*>(poolRaw);

            int nKvHeadsLayer = this->mNumKVHeads; // Llama-3-8B: 8
            int dHeadLayer = this->mHeadSize;       // 128
            int nQHeads = this->mNumHeads;          // 32

            // Compute slot mapping for K from host block_offsets.
            // Layout [batch, beam, kvFactor=2, max_blocks_per_seq];
            // K is k_or_v=0.
            auto const& boDesc = inputDesc[hBlockOffsetsIdx];
            int batchSize = boDesc.dims.d[0];
            int beamWidth = boDesc.dims.d[1];
            int maxBlocksPerSeq = boDesc.dims.d[3];
            (void)batchSize;
            (void)beamWidth;
            constexpr int kTokensPerBlock_b22 = 32;
            std::vector<int32_t> hSlotK(nTokens);
            for (int t = 0; t < nTokens; ++t)
            {
                int blockInSeq = t / kTokensPerBlock_b22;
                int offInBlock = t - blockInSeq * kTokensPerBlock_b22;
                // [0, 0, 0, blockInSeq] index, k_or_v=0:
                int physBlock
                    = hBlockOffsets[((0 * beamWidth + 0) * kvFactor + 0) * maxBlocksPerSeq + blockInSeq];
                hSlotK[t] = physBlock * kTokensPerBlock_b22 + offInBlock;
            }

            // Slot mapping for V — uses block_offsets with k_or_v=1.
            std::vector<int32_t> hSlotV(nTokens);
            for (int t = 0; t < nTokens; ++t)
            {
                int blockInSeq = t / kTokensPerBlock_b22;
                int offInBlock = t - blockInSeq * kTokensPerBlock_b22;
                int physBlock
                    = hBlockOffsets[((0 * beamWidth + 0) * kvFactor + 1) * maxBlocksPerSeq + blockInSeq];
                hSlotV[t] = physBlock * kTokensPerBlock_b22 + offInBlock;
            }

            // For K6 read-back, gather active blocks for this batch
            // (just block 0 in the smoke). For now reconstruct the
            // physical-block list from hSlotK (one block in the smoke).
            int const nActiveBlocks = (nTokens + kTokensPerBlock_b22 - 1) / kTokensPerBlock_b22;
            std::vector<int32_t> hPhysBlocks(nActiveBlocks);
            for (int b = 0; b < nActiveBlocks; ++b)
            {
                hPhysBlocks[b]
                    = hBlockOffsets[((0 * beamWidth + 0) * kvFactor + 0) * maxBlocksPerSeq + b];
            }

            int32_t* dSlotK = nullptr;
            int32_t* dSlotV = nullptr;
            int32_t* dPhysBlocks = nullptr;
            void* dKContig = nullptr;
            void* dVContig = nullptr;
            void* dK6ScratchK = nullptr;
            void* dK6ScratchV = nullptr;
            std::size_t const kContigBytes
                = static_cast<std::size_t>(nTokens) * nKvHeadsLayer * dHeadLayer * sizeof(std::uint16_t);
            std::size_t const k6ScratchBytes = static_cast<std::size_t>(nActiveBlocks) * nKvHeadsLayer
                * kTokensPerBlock_b22 * dHeadLayer * sizeof(std::uint16_t);
            bool realOk = (cudaMalloc(&dSlotK, nTokens * sizeof(int32_t)) == cudaSuccess)
                && (cudaMalloc(&dSlotV, nTokens * sizeof(int32_t)) == cudaSuccess)
                && (cudaMalloc(&dPhysBlocks, nActiveBlocks * sizeof(int32_t)) == cudaSuccess)
                && (cudaMalloc(&dKContig, kContigBytes) == cudaSuccess)
                && (cudaMalloc(&dVContig, kContigBytes) == cudaSuccess)
                && (cudaMalloc(&dK6ScratchK, k6ScratchBytes) == cudaSuccess)
                && (cudaMalloc(&dK6ScratchV, k6ScratchBytes) == cudaSuccess);
            if (realOk)
            {
                cudaMemcpyAsync(dSlotK, hSlotK.data(), nTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(dSlotV, hSlotV.data(), nTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(
                    dPhysBlocks, hPhysBlocks.data(), nActiveBlocks * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

                // Slice K and V out of fused QKV [nTokens, (nQ + 2*nKv)*dHead].
                std::size_t const qkvStrideBytes
                    = static_cast<std::size_t>(nQHeads + 2 * nKvHeadsLayer) * dHeadLayer * sizeof(std::uint16_t);
                std::size_t const kSrcOffsetBytes
                    = static_cast<std::size_t>(nQHeads) * dHeadLayer * sizeof(std::uint16_t);
                std::size_t const vSrcOffsetBytes
                    = static_cast<std::size_t>(nQHeads + nKvHeadsLayer) * dHeadLayer * sizeof(std::uint16_t);
                std::size_t const kvBytesPerToken
                    = static_cast<std::size_t>(nKvHeadsLayer) * dHeadLayer * sizeof(std::uint16_t);
                cudaMemcpy2DAsync(dKContig, kvBytesPerToken,
                    static_cast<std::uint8_t const*>(inputs[0]) + kSrcOffsetBytes, qkvStrideBytes, kvBytesPerToken,
                    nTokens, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpy2DAsync(dVContig, kvBytesPerToken,
                    static_cast<std::uint8_t const*>(inputs[0]) + vSrcOffsetBytes, qkvStrideBytes, kvBytesPerToken,
                    nTokens, cudaMemcpyDeviceToDevice, stream);

                int rcK = tq_kv_quantize_paged_trtllm_layered(dKContig, dSlotK, poolBase, gQuantState.signs,
                    gQuantState.centroids, gQuantState.thresholds, mTurboquantBits, kDtypeFP16, nTokens, nKvHeadsLayer,
                    dHeadLayer, kTokensPerBlock_b22, nLayersPerPool, kvFactor, relativeLayer, /*k_or_v=*/0, stream);
                int rcV = tq_kv_quantize_paged_trtllm_layered(dVContig, dSlotV, poolBase, gQuantState.signs,
                    gQuantState.centroids, gQuantState.thresholds, mTurboquantBits, kDtypeFP16, nTokens, nKvHeadsLayer,
                    dHeadLayer, kTokensPerBlock_b22, nLayersPerPool, kvFactor, relativeLayer, /*k_or_v=*/1, stream);
                int rcK6 = tq_kv_dequantize_paged_trtllm_layered(poolBase, dPhysBlocks, gQuantState.signs,
                    gQuantState.centroids, dK6ScratchK, mTurboquantBits, kDtypeFP16, nActiveBlocks, nKvHeadsLayer,
                    dHeadLayer, kTokensPerBlock_b22, nLayersPerPool, kvFactor, relativeLayer, /*k_or_v=*/0, stream);
                int rcV6 = tq_kv_dequantize_paged_trtllm_layered(poolBase, dPhysBlocks, gQuantState.signs,
                    gQuantState.centroids, dK6ScratchV, mTurboquantBits, kDtypeFP16, nActiveBlocks, nKvHeadsLayer,
                    dHeadLayer, kTokensPerBlock_b22, nLayersPerPool, kvFactor, relativeLayer, /*k_or_v=*/1, stream);
                cudaError_t syncErr2 = cudaStreamSynchronize(stream);
                TLLM_LOG_INFO(
                    "[B.2.2] K3+K6 real-pool (layer=%d): rcK=%d rcV=%d rcK6=%d rcV6=%d cudaSync=%d "
                    "(slotK[0..last]=%d..%d slotV[0..last]=%d..%d active_blocks=%d k6Scratch=%zuB each)",
                    this->mLayerIdx, rcK, rcV, rcK6, rcV6, static_cast<int>(syncErr2), hSlotK.front(), hSlotK.back(),
                    hSlotV.front(), hSlotV.back(), nActiveBlocks, k6ScratchBytes);
            }
            else
            {
                TLLM_LOG_ERROR("[B.2.2] K3+K6 real-pool: cudaMalloc failed");
            }
            if (dSlotK)
                cudaFree(dSlotK);
            if (dSlotV)
                cudaFree(dSlotV);
            if (dPhysBlocks)
                cudaFree(dPhysBlocks);
            if (dKContig)
                cudaFree(dKContig);
            if (dVContig)
                cudaFree(dVContig);
            if (dK6ScratchK)
                cudaFree(dK6ScratchK);
            if (dK6ScratchV)
                cudaFree(dK6ScratchV);
        }

        // B.2.2 reconnaissance — log the input tensor descriptors we'll
        // need for K3/K6 paged write/read. Fires once globally; cheap.
        auto logTensor = [&](char const* label, IdxEntry e) {
            if (!isEntryUsed(e))
            {
                TLLM_LOG_INFO("[B.2.2] %s: <not used>", label);
                return;
            }
            auto idx = getIdx(e);
            auto const& d = inputDesc[idx];
            int totalElems = 1;
            char dimsStr[128];
            int n = 0;
            for (int i = 0; i < d.dims.nbDims; ++i)
            {
                int dim = d.dims.d[i];
                totalElems *= dim;
                n += snprintf(dimsStr + n, sizeof(dimsStr) - n, "%s%d", (i == 0 ? "" : "x"), dim);
                if (n >= static_cast<int>(sizeof(dimsStr)))
                    break;
            }
            TLLM_LOG_INFO("[B.2.2] %s idx=%d shape=%s dtype=%d totalElems=%d", label, idx, dimsStr,
                static_cast<int>(d.type), totalElems);
        };
        logTensor("QKV_TENSOR", IdxEntry::QKV_TENSOR);
        logTensor("SEQUENCE_LENGTH", IdxEntry::SEQUENCE_LENGTH);
        logTensor("HOST_PAST_KEY_VALUE_LENGTHS", IdxEntry::HOST_PAST_KEY_VALUE_LENGTHS);
        logTensor("CONTEXT_LENGTHS", IdxEntry::CONTEXT_LENGTHS);
        logTensor("REQUEST_TYPES", IdxEntry::REQUEST_TYPES);
        logTensor("KV_CACHE_BLOCK_OFFSETS", IdxEntry::KV_CACHE_BLOCK_OFFSETS);
        logTensor("HOST_KV_CACHE_BLOCK_OFFSETS", IdxEntry::HOST_KV_CACHE_BLOCK_OFFSETS);
        logTensor("HOST_KV_CACHE_POOL_POINTERS", IdxEntry::HOST_KV_CACHE_POOL_POINTERS);
        logTensor("HOST_KV_CACHE_POOL_MAPPING", IdxEntry::HOST_KV_CACHE_POOL_MAPPING);
    }

    // Patch inputs[0] to our round-tripped buffer, forward to parent.
    constexpr int kMaxInputs = 64;
    void const* patchedInputs[kMaxInputs] = {nullptr};
    for (int i = 0; i < kMaxInputs; ++i)
    {
        patchedInputs[i] = inputs[i];
    }
    patchedInputs[0] = mWorkspace.kv_out;

    // B.2.2 step 9a — pool-pointer patch dry run. Copies
    // HOST_KV_CACHE_POOL_POINTERS into a stack-local host buffer with
    // identical values and patches the inputs array to point at the
    // copy. Generation must continue byte-for-byte unchanged since the
    // pointer values are the same; this just validates the patch
    // machinery so the next session can replace the copy with a
    // scratch ptr that holds K6-dequantized data.
    //
    // Pool pointers tensor shape [n_pools, 2] of int64; n_pools is
    // typically 1 for Llama-3. Bounded array size accommodates up to
    // 8 pools.
    std::int64_t hPoolPtrsCopy[16] = {0};
    if (isEntryUsed(IdxEntry::HOST_KV_CACHE_POOL_POINTERS))
    {
        auto poolPtrsIdx = getIdx(IdxEntry::HOST_KV_CACHE_POOL_POINTERS);
        auto const& d = inputDesc[poolPtrsIdx];
        int totalElems = 1;
        for (int i = 0; i < d.dims.nbDims; ++i)
        {
            totalElems *= d.dims.d[i];
        }
        if (totalElems > 0 && totalElems <= static_cast<int>(sizeof(hPoolPtrsCopy) / sizeof(std::int64_t)))
        {
            std::memcpy(hPoolPtrsCopy, inputs[poolPtrsIdx], totalElems * sizeof(std::int64_t));
            patchedInputs[poolPtrsIdx] = hPoolPtrsCopy;
        }
    }
    return GPTAttentionPlugin::enqueue(inputDesc, outputDesc, patchedInputs, outputs, workspace, stream);
}

char const* TurboquantAttentionPlugin::getPluginType() const noexcept
{
    return TURBOQUANT_ATTENTION_PLUGIN_NAME;
}

char const* TurboquantAttentionPlugin::getPluginVersion() const noexcept
{
    return TURBOQUANT_ATTENTION_PLUGIN_VERSION;
}

TurboquantAttentionPlugin* TurboquantAttentionPlugin::clone() const noexcept
{
    // The parent's clone() uses cloneImpl<T>() which calls T's copy
    // constructor and then plugin->initialize(). Without initialize(),
    // GPTAttentionPluginCommon's runtime state stays uninitialized and
    // getWorkspaceSize segfaults (verified 2026-06-03 backtrace into
    // AttentionOp::getWorkspaceSizeForContext).
    //
    // We can't use the compiler-default copy ctor blindly because our
    // mWorkspace owns GPU buffers; the clone must start with an empty
    // workspace (lazy-allocated on first enqueue). Round-trip through
    // serialize/deserialize (which gives a fresh mWorkspace) and then
    // explicitly call initialize() to match the parent's invariant.
    auto const size = getSerializationSize();
    std::vector<char> buffer(size);
    serialize(buffer.data());
    auto* cloned = new TurboquantAttentionPlugin(buffer.data(), size);
    cloned->setPluginNamespace(getPluginNamespace());
    cloned->initialize();
    return cloned;
}

size_t TurboquantAttentionPlugin::getSerializationSize() const noexcept
{
    return GPTAttentionPlugin::getSerializationSize() + sizeof(int);
}

void TurboquantAttentionPlugin::serialize(void* buffer) const noexcept
{
    GPTAttentionPlugin::serialize(buffer);
    auto* tail = static_cast<char*>(buffer) + GPTAttentionPlugin::getSerializationSize();
    std::memcpy(tail, &mTurboquantBits, sizeof(int));
}

void TurboquantAttentionPlugin::destroy() noexcept
{
    mWorkspace.free();
    GPTAttentionPlugin::destroy();
}

//
// Creator
//

TurboquantAttentionPluginCreator::TurboquantAttentionPluginCreator()
    : GPTAttentionPluginCreator()
{
    auto const* parentFC = GPTAttentionPluginCreator::getFieldNames();
    if (parentFC != nullptr)
    {
        mPluginAttributes.reserve(parentFC->nbFields + 1);
        for (int i = 0; i < parentFC->nbFields; ++i)
        {
            mPluginAttributes.push_back(parentFC->fields[i]);
        }
    }
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField{"turboquant_bits", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* TurboquantAttentionPluginCreator::getPluginName() const noexcept
{
    return TURBOQUANT_ATTENTION_PLUGIN_NAME;
}

char const* TurboquantAttentionPluginCreator::getPluginVersion() const noexcept
{
    return TURBOQUANT_ATTENTION_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* TurboquantAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* TurboquantAttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    // Parse PluginField directly (same field set as GPTAttentionPluginCreator
    // plus our turboquant_bits). The earlier delegate-and-deserialize
    // approach segfaulted on getWorkspaceSize because the
    // serialize→destroy→deserialize round-trip didn't fully restore
    // GPTAttentionPluginCommon's internal state (M12.4 B.2.1b
    // diagnosis 2026-06-03).
    PluginFieldParser p{fc->nbFields, fc->fields};
    try
    {
        int turboquantBits = p.getScalar<int32_t>("turboquant_bits").value();
        validateBits(turboquantBits);
        auto* obj = new TurboquantAttentionPlugin(p.getScalar<int32_t>("layer_idx").value(),
            p.getScalar<int32_t>("num_heads").value(), p.getScalar<int32_t>("vision_start").value(),
            p.getScalar<int32_t>("vision_length").value(), p.getScalar<int32_t>("num_kv_heads").value(),
            p.getScalar<int32_t>("num_kv_heads_origin").value(), p.getScalar<int32_t>("head_size").value(),
            p.getScalar<int32_t>("unidirectional").value(), p.getScalar<float>("q_scaling").value(),
            p.getScalar<float>("attn_logit_softcapping_scale").value(),
            static_cast<tensorrt_llm::kernels::PositionEmbeddingType>(
                p.getScalar<int8_t>("position_embedding_type").value()),
            p.getScalar<int32_t>("rotary_embedding_dim").value(), p.getScalar<float>("rotary_embedding_base").value(),
            static_cast<tensorrt_llm::kernels::RotaryScalingType>(
                p.getScalar<int8_t>("rotary_embedding_scale_type").value()),
            p.getScalar<float>("rotary_embedding_scale").value(),
            p.getScalar<float>("rotary_embedding_short_m_scale").value(),
            p.getScalar<float>("rotary_embedding_long_m_scale").value(),
            p.getScalar<int32_t>("rotary_embedding_max_positions").value(),
            p.getScalar<int32_t>("rotary_embedding_original_max_positions").value(),
            static_cast<int32_t>(p.getScalar<int32_t>("tp_size").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("tp_rank").value()),
            static_cast<bool>(p.getScalar<int8_t>("unfuse_qkv_gemm").value()),
            static_cast<bool>(p.getScalar<int8_t>("use_logn_scaling").value()),
            static_cast<tensorrt_llm::kernels::ContextFMHAType>(p.getScalar<int8_t>("context_fmha_type").value()),
            p.getScalar<int32_t>("kv_cache_quant_mode").value(),
            static_cast<bool>(p.getScalar<int8_t>("remove_input_padding").value()),
            static_cast<tensorrt_llm::kernels::AttentionMaskType>(p.getScalar<int32_t>("mask_type").value()),
            tensorrt_llm::kernels::BlockSparseParams{p.getScalar<int32_t>("block_sparse_block_size").value(),
                static_cast<bool>(p.getScalar<int8_t>("block_sparse_homo_head_pattern").value()),
                p.getScalar<int32_t>("block_sparse_num_local_blocks").value(),
                p.getScalar<int32_t>("block_sparse_vertical_stride").value()},
            static_cast<bool>(p.getScalar<int32_t>("paged_kv_cache").value()),
            p.getScalar<int32_t>("tokens_per_block").value(),
            static_cast<nvinfer1::DataType>(p.getScalar<int32_t>("type_id").value()),
            p.getScalar<int32_t>("max_context_length").value(),
            static_cast<bool>(p.getScalar<int8_t>("qkv_bias_enabled").value()),
            static_cast<bool>(p.getScalar<int8_t>("do_cross_attention").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("max_distance").value()),
            static_cast<bool>(p.getScalar<int8_t>("pos_shift_enabled").value()),
            static_cast<bool>(p.getScalar<int8_t>("dense_context_fmha").value()),
            static_cast<bool>(p.getScalar<int8_t>("use_paged_context_fmha").value()),
            static_cast<bool>(p.getScalar<int8_t>("use_fp8_context_fmha").value()),
            static_cast<bool>(p.getScalar<int8_t>("has_full_attention_mask").value()),
            static_cast<bool>(p.getScalar<int32_t>("use_cache").value()),
            static_cast<bool>(p.getScalar<int8_t>("is_spec_decoding_enabled").value()),
            static_cast<bool>(p.getScalar<int8_t>("spec_decoding_is_generation_length_variable").value()),
            p.getScalar<int32_t>("spec_decoding_max_generation_length").value(),
            static_cast<int8_t>(p.getScalar<int8_t>("is_mla_enabled").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("q_lora_rank").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("kv_lora_rank").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("qk_nope_head_dim").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("qk_rope_head_dim").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("v_head_dim").value()),
            static_cast<bool>(p.getScalar<int8_t>("fuse_fp4_quant").value()),
            static_cast<bool>(p.getScalar<int8_t>("skip_attn").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("cp_size").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("cp_rank").value()),
            static_cast<std::set<int32_t>>(p.getSet<int32_t>("cp_group").value()), turboquantBits);
        obj->setPluginNamespace(getPluginNamespace());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2* TurboquantAttentionPluginCreator::deserializePlugin(
    char const* /*name*/, void const* serialData, size_t serialLength) noexcept
{
    auto* obj = new TurboquantAttentionPlugin(serialData, serialLength);
    obj->setPluginNamespace(getPluginNamespace());
    return obj;
}

} // namespace tensorrt_llm::plugins
