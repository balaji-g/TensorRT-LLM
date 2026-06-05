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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <set>
#include <unordered_map>
#include <vector>

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

// Optional separate K-side quantizer state, loaded from a binary file
// pointed to by env TQ_K_CODEBOOK_PATH. File layout:
//   [n_levels * float32 centroids][(n_levels - 1) * float32 thresholds]
// where n_levels = 256 for bits=8, 16 for bits=4. Shares the same signs
// as gQuantState (signs are random ±1; only the codebook differs).
// Loaded lazily on first stage-2 enqueue when env is set; falls back
// to gQuantState's centroids/thresholds when env is unset or load
// fails. Used for K3+K6 of K only (V keeps gQuantState).
struct DeviceQuantStateExt
{
    int8_t* signs{nullptr};       // shared with gQuantState (alias)
    float* centroids{nullptr};
    float* thresholds{nullptr};
    int bits{0};
    bool active{false};
    bool init_attempted{false};

    bool initOnce(int b, int8_t* sharedSigns)
    {
        if (init_attempted)
            return active;
        init_attempted = true;
        char const* path = std::getenv("TQ_K_CODEBOOK_PATH");
        if (path == nullptr)
        {
            TLLM_LOG_INFO("[B.2.2 stage 3] TQ_K_CODEBOOK_PATH unset -- K uses baked-in codebook");
            return false;
        }
        int const nLevels = 1 << b;
        int const nThresholds = nLevels - 1;
        std::size_t const expectedBytes
            = static_cast<std::size_t>(nLevels + nThresholds) * sizeof(float);
        FILE* fp = std::fopen(path, "rb");
        if (fp == nullptr)
        {
            TLLM_LOG_WARNING("[B.2.2 stage 3] TQ_K_CODEBOOK_PATH=%s could not be opened", path);
            return false;
        }
        std::vector<float> hostCentroids(nLevels), hostThresholds(nThresholds);
        std::size_t readC = std::fread(hostCentroids.data(), sizeof(float), nLevels, fp);
        std::size_t readT = std::fread(hostThresholds.data(), sizeof(float), nThresholds, fp);
        std::fclose(fp);
        if (readC != static_cast<std::size_t>(nLevels) || readT != static_cast<std::size_t>(nThresholds))
        {
            TLLM_LOG_WARNING("[B.2.2 stage 3] TQ_K_CODEBOOK_PATH=%s short read: got %zu centroids + %zu thresholds, "
                             "expected %d + %d (%zu bytes)",
                path, readC, readT, nLevels, nThresholds, expectedBytes);
            return false;
        }
        if (cudaMalloc(&centroids, nLevels * sizeof(float)) != cudaSuccess
            || cudaMalloc(&thresholds, nThresholds * sizeof(float)) != cudaSuccess)
        {
            TLLM_LOG_ERROR("[B.2.2 stage 3] alt K codebook cudaMalloc failed");
            return false;
        }
        cudaMemcpy(centroids, hostCentroids.data(), nLevels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(thresholds, hostThresholds.data(), nThresholds * sizeof(float), cudaMemcpyHostToDevice);
        signs = sharedSigns;
        bits = b;
        active = true;
        TLLM_LOG_INFO("[B.2.2 stage 3] alt K codebook loaded from %s "
                      "(centroid range %+.4f .. %+.4f)",
            path, hostCentroids.front(), hostCentroids.back());
        return true;
    }
};

DeviceQuantStateExt gQuantStateK;

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
// Lazy-allocate/grow a device buffer. Returns false on cudaMalloc failure.
bool ensureBuf(void** buf, std::size_t* cap, std::size_t needBytes)
{
    if (needBytes <= *cap)
        return true;
    if (*buf)
    {
        cudaFree(*buf);
        *buf = nullptr;
        *cap = 0;
    }
    if (cudaMalloc(buf, needBytes) != cudaSuccess)
        return false;
    *cap = needBytes;
    return true;
}
template <typename T>
bool ensureBufT(T** buf, std::size_t* cap, std::size_t needBytes)
{
    return ensureBuf(reinterpret_cast<void**>(buf), cap, needBytes);
}

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

    // TQ_DISABLE_K1K2=1 skips the K1+K2 streaming round-trip and copies
    // inputs[0] straight to mWorkspace.kv_out. Useful for isolating
    // K3+K6 quality (stage 2): with K1+K2 disabled and stage 9b enabled,
    // K experiences ONE quantization round-trip (K3+K6 on post-RoPE K)
    // instead of TWO (K1+K2 pre-RoPE plus K3+K6 post-RoPE).
    static bool const sDisableK1K2 = []() {
        char const* v = std::getenv("TQ_DISABLE_K1K2");
        bool d = (v != nullptr && v[0] == '1');
        if (d) TLLM_LOG_INFO("[B.2.2 stage 3] TQ_DISABLE_K1K2=1 (K1+K2 streaming round-trip skipped, raw inputs[0] passed through)");
        return d;
    }();
    if (sDisableK1K2)
    {
        // Just copy inputs[0] -> mWorkspace.kv_out unchanged.
        std::size_t bytes = static_cast<std::size_t>(nTokens) * nTotalHeads * kDHead * sizeof(std::uint16_t);
        cudaMemcpyAsync(mWorkspace.kv_out, inputs[0], bytes, cudaMemcpyDeviceToDevice, stream);
    }
    else
    {
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
    }

    // TQ_QUANT_Q=1 (stage 4 fix candidate (i)): apply an extra K1+K2
    // round-trip to Q's slice of mWorkspace.kv_out. Closes the Q-K
    // asymmetric-error gap that stage 2 opens: after this, Q has 2x
    // K1+K2 round-trip error per layer (matching K-history's 1x
    // K1+K2 + 1x K3+K6 ~= 2x). If the Q.K asymmetric-error
    // hypothesis is right, stage 2 with this switch on should
    // produce coherent text near baseline.
    static bool const sQuantQ = []() {
        char const* v = std::getenv("TQ_QUANT_Q");
        bool q = (v != nullptr && v[0] == '1');
        if (q) TLLM_LOG_INFO("[B.2.2 stage 4] TQ_QUANT_Q=1 (extra K1+K2 on Q slice to match K-history error level)");
        return q;
    }();
    if (sQuantQ)
    {
        int const nQHeads = this->mNumHeads;
        std::size_t const qBytes
            = static_cast<std::size_t>(nTokens) * nQHeads * kDHead * sizeof(std::uint16_t);
        std::size_t const tokenStrideKvOut
            = static_cast<std::size_t>(nTotalHeads) * kDHead * sizeof(std::uint16_t);
        std::size_t const tokenStrideQ
            = static_cast<std::size_t>(nQHeads) * kDHead * sizeof(std::uint16_t);
        void* qContig = nullptr;
        if (cudaMalloc(&qContig, qBytes) == cudaSuccess)
        {
            // Gather Q slice: per-token first nQHeads*dHead*2 bytes.
            cudaMemcpy2DAsync(qContig, tokenStrideQ, mWorkspace.kv_out, tokenStrideKvOut,
                tokenStrideQ, nTokens, cudaMemcpyDeviceToDevice, stream);
            // Round-trip via streaming K1+K2 in place.
            tq_kv_quantize_streaming(qContig, gQuantState.signs, gQuantState.centroids,
                gQuantState.thresholds, mWorkspace.packed, mWorkspace.norms, mTurboquantBits,
                kDtypeFP16, nTokens, nQHeads, kDHead, stream);
            tq_kv_dequantize_streaming(mWorkspace.packed, mWorkspace.norms, gQuantState.signs,
                gQuantState.centroids, qContig, mTurboquantBits, kDtypeFP16, nTokens, nQHeads,
                kDHead, stream);
            // Scatter back into mWorkspace.kv_out's Q region.
            cudaMemcpy2DAsync(mWorkspace.kv_out, tokenStrideKvOut, qContig, tokenStrideQ,
                tokenStrideQ, nTokens, cudaMemcpyDeviceToDevice, stream);
            // Diagnostic-grade: sync free. Promote to member buffer if
            // we keep this in production.
            cudaStreamSynchronize(stream);
            cudaFree(qContig);
        }
        else
        {
            TLLM_LOG_WARNING("[B.2.2 stage 4] TQ_QUANT_Q cudaMalloc(%zu) failed; skipping Q re-quantization",
                qBytes);
        }
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
                kTokensPerBlock, /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0, /*k_or_v=*/0,
                /*slot_inner_bytes=*/-1, stream);
            int rc6 = tq_kv_dequantize_paged_trtllm_layered(shadowPool, physBlocks, gQuantState.signs,
                gQuantState.centroids, k6Scratch, mTurboquantBits, kDtypeFP16, nBlocks, nTotalHeads, kDHead,
                kTokensPerBlock, /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0, /*k_or_v=*/0,
                /*slot_inner_bytes=*/-1, stream);
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

        // B.2.2 step 7 (re-enabled after slot_inner_bytes API fix) —
        // K3 layered + K6 layered against the REAL persistent KV
        // pool. Now passes slot_inner_bytes = n_kv_heads *
        // tokens_per_block * d_head * sizeof(fp16) to match the
        // manager's actual stride (no B.3b yet, pool is full fp16).
        // The K3 write is still ephemeral (inner overwrites this
        // slot's bytes during its own enqueue). Validates the full
        // production-shape K3 + K6 invocation on the real pool.
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
            (void)nPools;
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

                // Manager (no B.3b) allocates slot_inner = n_kv_heads
                // * tokens_per_block * d_head * sizeof(fp16) per
                // (block, layer, K-or-V). Pass that as slot_inner_bytes
                // so the layered launcher's stride math matches the
                // actual pool layout.
                int const slotInnerBytes = nKvHeadsLayer * kTokensPerBlock_b22 * dHeadLayer
                    * static_cast<int>(sizeof(std::uint16_t));
                int rcK = tq_kv_quantize_paged_trtllm_layered(dKContig, dSlotK, poolBase, gQuantState.signs,
                    gQuantState.centroids, gQuantState.thresholds, mTurboquantBits, kDtypeFP16, nTokens, nKvHeadsLayer,
                    dHeadLayer, kTokensPerBlock_b22, nLayersPerPool, kvFactor, relativeLayer, /*k_or_v=*/0,
                    slotInnerBytes, stream);
                int rcV = tq_kv_quantize_paged_trtllm_layered(dVContig, dSlotV, poolBase, gQuantState.signs,
                    gQuantState.centroids, gQuantState.thresholds, mTurboquantBits, kDtypeFP16, nTokens, nKvHeadsLayer,
                    dHeadLayer, kTokensPerBlock_b22, nLayersPerPool, kvFactor, relativeLayer, /*k_or_v=*/1,
                    slotInnerBytes, stream);
                int rcK6 = tq_kv_dequantize_paged_trtllm_layered(poolBase, dPhysBlocks, gQuantState.signs,
                    gQuantState.centroids, dK6ScratchK, mTurboquantBits, kDtypeFP16, nActiveBlocks, nKvHeadsLayer,
                    dHeadLayer, kTokensPerBlock_b22, nLayersPerPool, kvFactor, relativeLayer, /*k_or_v=*/0,
                    slotInnerBytes, stream);
                int rcV6 = tq_kv_dequantize_paged_trtllm_layered(poolBase, dPhysBlocks, gQuantState.signs,
                    gQuantState.centroids, dK6ScratchV, mTurboquantBits, kDtypeFP16, nActiveBlocks, nKvHeadsLayer,
                    dHeadLayer, kTokensPerBlock_b22, nLayersPerPool, kvFactor, relativeLayer, /*k_or_v=*/1,
                    slotInnerBytes, stream);
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

    // B.2.2 step 9b — K3 + K6 + scratch + pool-ptr swap.
    // Real persistent path: write new K/V into the real pool via K3
    // (round-tripped via the layered launcher), dequantize active
    // blocks from the pool via K6 into a pool-layout scratch buffer
    // that the inner GPTAttention then reads (via the patched pool
    // pointer) instead of the real pool. Without B.3b the real pool
    // is still fp16-shaped, so K3 writes packed bytes into the first
    // portion of each slot and leaves the rest unused — wasteful but
    // semantically correct. Once B.3b shrinks the pool, the same K3
    // call lands at exactly slot_inner = packed + norms.
    //
    // K1+K2 streaming round-trip on inputs[0] (mWorkspace.kv_out)
    // stays the source of K/V we slice into K3 — that keeps the
    // semantics consistent: scratch[N] (written by inner from
    // patchedInputs[0] = mWorkspace.kv_out) matches K6's dequant of
    // K3's persisted value (both round-tripped from the same fresh
    // K/V), so FMHA sees fully round-tripped tokens 0..N.
    bool stepB22Done = false;
    // Stage-2 post-enqueue context — populated below if step 9b ran
    // its pre-enqueue half successfully. Drives the post-enqueue K3
    // that captures parent's post-RoPE K from mScratchPool into the
    // persistent real pool.
    struct Stage2Ctx
    {
        bool active{false};
        void* poolBaseShifted{nullptr};
        int slotInnerBytes{0};
        int nKvHeads{0};
        int dHead{0};
        int blockInSeq{0};    // single-block assumption (TODO multi-block prefill)
        int offInBlockFirst{0};
        int kFlat{0};
        int vFlat{0};
        int turboquantBits{0};
        int nKBlocks{0};
        int nVBlocks{0};
    } stage2;

    // step 9b is opt-in via TQ_ENABLE_STEP9B=1. Stage 2 (default in
    // this build): K6 -> scratchPool BEFORE parent's enqueue (load
    // history), patch pool ptr, run parent (which writes post-RoPE K
    // + V into scratchPool); AFTER parent returns, gather the new
    // tokens from scratchPool and K3 quantize into the persistent
    // real pool. Closes the persistence loop with post-RoPE K so
    // the next decode's K6 dequant matches FMHA's expectation.
    static bool const sStep9bEnabled = []() {
        char const* v = std::getenv("TQ_ENABLE_STEP9B");
        bool enabled = (v != nullptr && v[0] == '1');
        TLLM_LOG_INFO("[B.2.2] step 9b %s (TQ_ENABLE_STEP9B=%s)",
                      enabled ? "ENABLED (stage 2 — post-enqueue K3)" : "DISABLED (K1+K2 only)",
                      v ? v : "<unset>");
        return enabled;
    }();
    if (sStep9bEnabled
        && isEntryUsed(IdxEntry::HOST_KV_CACHE_POOL_POINTERS)
        && isEntryUsed(IdxEntry::HOST_KV_CACHE_POOL_MAPPING)
        && isEntryUsed(IdxEntry::HOST_KV_CACHE_BLOCK_OFFSETS)
        && isEntryUsed(IdxEntry::HOST_PAST_KEY_VALUE_LENGTHS))
    {
        auto poolPtrsIdx = getIdx(IdxEntry::HOST_KV_CACHE_POOL_POINTERS);
        auto mappingIdx = getIdx(IdxEntry::HOST_KV_CACHE_POOL_MAPPING);
        auto hBOIdx = getIdx(IdxEntry::HOST_KV_CACHE_BLOCK_OFFSETS);
        auto pastIdx = getIdx(IdxEntry::HOST_PAST_KEY_VALUE_LENGTHS);

        std::int64_t const* hPoolPtrs = static_cast<std::int64_t const*>(inputs[poolPtrsIdx]);
        std::int32_t const* hMapping = static_cast<std::int32_t const*>(inputs[mappingIdx]);
        std::int32_t const* hBlockOffsets = static_cast<std::int32_t const*>(inputs[hBOIdx]);
        std::int32_t const* hPastKv = static_cast<std::int32_t const*>(inputs[pastIdx]);

        int const nLayersPerPool = inputDesc[mappingIdx].dims.d[0];
        (void)nLayersPerPool;
        int const nPools = inputDesc[poolPtrsIdx].dims.d[0];
        constexpr int kvFactor = 2;
        constexpr int kTokensPerBlockB22 = 32;

        int const poolIdxForLayer = hMapping[this->mLayerIdx * 2 + 0];
        int const relativeLayer = hMapping[this->mLayerIdx * 2 + 1];
        void* const poolBase
            = reinterpret_cast<void*>(static_cast<std::uintptr_t>(hPoolPtrs[poolIdxForLayer * 2 + 0]));

        int const nKvHeadsL = this->mNumKVHeads;
        int const nQHeadsL = this->mNumHeads;
        (void)nQHeadsL;
        int const dHeadL = this->mHeadSize;
        int const slotInnerBytes
            = nKvHeadsL * kTokensPerBlockB22 * dHeadL * static_cast<int>(sizeof(std::uint16_t));

        auto const& boDesc = inputDesc[hBOIdx];
        int const batchSize = boDesc.dims.d[0];
        int const beamWidth = boDesc.dims.d[1];
        int const maxBlocksPerSeq = boDesc.dims.d[3];

        // Compute per-token slot mappings + active flat-slot ID sets.
        // Single-sequence prefill / single-token decode assumption
        // (the smoke). Multi-seq batches: TODO follow-up.
        //
        // Pool layout (kvCacheManager.cpp:694) is
        //   pool[nBlocks][nLayersPerPool][kvFactor][slot_inner].
        // setOffsets (kvCacheManager.cpp:823) populates block_offsets
        // with FLAT slot indices: block_offset_value = memIdx *
        // (nLayersPerPool * kvFactor) + xIdx, where xIdx is 0 for K
        // and 1 for V, and layerIdx is hardcoded to 0 — the parent
        // adds layerOffset = L * kvFactor * slot_inner separately
        // (gptAttentionPlugin.cpp:896-898). So K_block_offset and
        // V_block_offset for the same logical block differ by 1
        // (memIdx=0 → K=0, V=1; memIdx=1 → K=64, V=65 at kvFactor=2).
        //
        // We therefore call K3/K6 with n_layers_per_pool=1, kv_factor=1,
        // relative_layer=0, k_or_v=0, and shift pool_base by
        // L*kvFactor*slot_inner so each (block_offset_value)-indexed
        // 65 KB slot lands at its true pool address. slot_mapping[t]
        // = block_offset_value * tokens_per_block + offInBlock.
        std::vector<std::int32_t> hSlotK(nTokens), hSlotV(nTokens);
        std::set<std::int32_t> kBlocksSet, vBlocksSet;
        int const s = 0;
        // TRT-LLM batch manager sets HOST_PAST_KEY_VALUE_LENGTHS to
        // (beginCompute + inputLength) — the *after-this-step* total
        // for the sequence (runtimeBuffers.cpp:549).
        int const seqTotalAfter = (batchSize > 0) ? hPastKv[s] : nTokens;
        int const seqPastBefore = seqTotalAfter - nTokens;
        int const nFullBlocks = (seqTotalAfter + kTokensPerBlockB22 - 1) / kTokensPerBlockB22;
        for (int t = 0; t < nTokens; ++t)
        {
            int absPos = seqPastBefore + t;
            int blockInSeq = absPos / kTokensPerBlockB22;
            int offInBlock = absPos - blockInSeq * kTokensPerBlockB22;
            int kFlat
                = hBlockOffsets[((s * beamWidth + 0) * kvFactor + 0) * maxBlocksPerSeq + blockInSeq];
            int vFlat
                = hBlockOffsets[((s * beamWidth + 0) * kvFactor + 1) * maxBlocksPerSeq + blockInSeq];
            hSlotK[t] = kFlat * kTokensPerBlockB22 + offInBlock;
            hSlotV[t] = vFlat * kTokensPerBlockB22 + offInBlock;
        }
        for (int b = 0; b < nFullBlocks; ++b)
        {
            int kFlat = hBlockOffsets[((s * beamWidth + 0) * kvFactor + 0) * maxBlocksPerSeq + b];
            int vFlat = hBlockOffsets[((s * beamWidth + 0) * kvFactor + 1) * maxBlocksPerSeq + b];
            kBlocksSet.insert(kFlat);
            vBlocksSet.insert(vFlat);
        }
        std::vector<std::int32_t> kBlockList(kBlocksSet.begin(), kBlocksSet.end());
        std::vector<std::int32_t> vBlockList(vBlocksSet.begin(), vBlocksSet.end());
        int const maxFlatSlot = std::max(kBlocksSet.empty() ? 0 : *kBlocksSet.rbegin(),
                                          vBlocksSet.empty() ? 0 : *vBlocksSet.rbegin());

        // Parallel sequence-block IDs for the (sorted) flat-slot lists, used
        // by the RoPE-forward kernel to derive absolute token positions
        // from K6's scratch_block_idx (= position-in-kBlockList).
        std::vector<std::int32_t> hKSeqBlocks(kBlockList.size());
        std::vector<std::int32_t> hVSeqBlocks(vBlockList.size());
        {
            std::unordered_map<std::int32_t, std::int32_t> kF2B, vF2B;
            for (int b = 0; b < nFullBlocks; ++b)
            {
                int kF = hBlockOffsets[((s * beamWidth + 0) * kvFactor + 0) * maxBlocksPerSeq + b];
                int vF = hBlockOffsets[((s * beamWidth + 0) * kvFactor + 1) * maxBlocksPerSeq + b];
                kF2B[kF] = b;
                vF2B[vF] = b;
            }
            for (std::size_t i = 0; i < kBlockList.size(); ++i)
                hKSeqBlocks[i] = kF2B[kBlockList[i]];
            for (std::size_t i = 0; i < vBlockList.size(); ++i)
                hVSeqBlocks[i] = vF2B[vBlockList[i]];
        }

        // Lazy-grow per-instance buffers.
        std::size_t const kvBytesPerToken
            = static_cast<std::size_t>(nKvHeadsL) * dHeadL * sizeof(std::uint16_t);
        std::size_t const kContigBytes = static_cast<std::size_t>(nTokens) * kvBytesPerToken;
        std::size_t const scratchK6KBytes
            = static_cast<std::size_t>(kBlockList.size()) * slotInnerBytes;
        std::size_t const scratchK6VBytes
            = static_cast<std::size_t>(vBlockList.size()) * slotInnerBytes;
        // scratchPool just needs to span [0 .. (maxFlatSlot + 1) *
        // slot_inner). We patch HOST_KV_CACHE_POOL_POINTERS to
        // mScratchPool - L * kvFactor * slot_inner so the parent's
        // layerOffset addition wraps back to mScratchPool base, then
        // its block_offset_value * slot_inner indexing lands in our
        // populated region. Pointer "negative-offset" arithmetic is
        // a 64-bit integer manipulation on GPU; the negative range
        // is never dereferenced.
        std::size_t const scratchPoolBytes
            = static_cast<std::size_t>(maxFlatSlot + 1) * slotInnerBytes;

        bool ok = ensureBuf(&mKContig, &mKContigCap, kContigBytes)
            && ensureBuf(&mVContig, &mVContigCap, kContigBytes)
            && ensureBufT(&mSlotMappingK, &mSlotMappingKCap, nTokens * sizeof(std::int32_t))
            && ensureBufT(&mSlotMappingV, &mSlotMappingVCap, nTokens * sizeof(std::int32_t))
            && ensureBufT(&mPhysBlocksK, &mPhysBlocksKCap, kBlockList.size() * sizeof(std::int32_t))
            && ensureBufT(&mPhysBlocksV, &mPhysBlocksVCap, vBlockList.size() * sizeof(std::int32_t))
            && ensureBuf(&mScratchK6K, &mScratchK6KCap, scratchK6KBytes)
            && ensureBuf(&mScratchK6V, &mScratchK6VCap, scratchK6VBytes)
            && ensureBuf(&mScratchPool, &mScratchPoolCap, scratchPoolBytes)
            && ensureBufT(&mSeqBlockIdsK, &mSeqBlockIdsKCap, hKSeqBlocks.size() * sizeof(std::int32_t))
            && ensureBufT(&mSeqBlockIdsV, &mSeqBlockIdsVCap, hVSeqBlocks.size() * sizeof(std::int32_t));

        if (ok)
        {
            // Copy host metadata to device.
            cudaMemcpyAsync(
                mSlotMappingK, hSlotK.data(), nTokens * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                mSlotMappingV, hSlotV.data(), nTokens * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(mPhysBlocksK, kBlockList.data(), kBlockList.size() * sizeof(std::int32_t),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(mPhysBlocksV, vBlockList.data(), vBlockList.size() * sizeof(std::int32_t),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(mSeqBlockIdsK, hKSeqBlocks.data(),
                hKSeqBlocks.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(mSeqBlockIdsV, hVSeqBlocks.data(),
                hVSeqBlocks.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream);

            // Shift pool_base by L*kvFactor*slot_inner so each
            // (block_offset_value)-indexed slot lines up with the
            // address the parent would compute as
            //   parent_addr = pool_base + L*kvFactor*slot_inner
            //                            + block_offset_value*slot_inner.
            std::size_t const layerStrideBytes
                = static_cast<std::size_t>(relativeLayer) * kvFactor * slotInnerBytes;
            void* const poolBaseShifted
                = static_cast<std::uint8_t*>(poolBase) + layerStrideBytes;

            // Lazily init the optional K-side codebook from
            // TQ_K_CODEBOOK_PATH. If active, use it for K's K3+K6
            // (V stays on the baked-in gQuantState since V is RoPE-
            // free and the baked codebook works for K1+K2 of V).
            gQuantStateK.initOnce(mTurboquantBits, gQuantState.signs);
            float const* kCentroids = gQuantStateK.active ? gQuantStateK.centroids : gQuantState.centroids;
            float const* kThresholds = gQuantStateK.active ? gQuantStateK.thresholds : gQuantState.thresholds;

            // K6 dequant active K and V slots from the real pool into
            // contiguous scratch_k6 buffers. For the very first enqueue
            // the real pool is uninitialized; we accept K6's output as
            // the history view regardless — on T=0 prefill there is no
            // history yet (pastBefore=0) so what K6 dequants is unused.
            // From T=1 onward, post-enqueue K3 (below) keeps the real
            // pool populated with parent's RoPE-applied K and V.
            int rcK6 = tq_kv_dequantize_paged_trtllm_layered(poolBaseShifted, mPhysBlocksK,
                gQuantState.signs, kCentroids, mScratchK6K, mTurboquantBits, kDtypeFP16,
                static_cast<int>(kBlockList.size()), nKvHeadsL, dHeadL, kTokensPerBlockB22,
                /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0,
                /*k_or_v=*/0, slotInnerBytes, stream);
            int rcV6 = tq_kv_dequantize_paged_trtllm_layered(poolBaseShifted, mPhysBlocksV,
                gQuantState.signs, gQuantState.centroids, mScratchK6V, mTurboquantBits, kDtypeFP16,
                static_cast<int>(vBlockList.size()), nKvHeadsL, dHeadL, kTokensPerBlockB22,
                /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0,
                /*k_or_v=*/0, slotInnerBytes, stream);

            if (rcK6 == 0 && rcV6 == 0)
            {
                // Stage 5 (option α) — K-cache stores K1+K2-roundtripped
                // *pre*-RoPE K (post-enqueue K3 below sources from
                // mWorkspace.kv_out's K slice, not from scratchPool's
                // post-RoPE K). To feed the parent's FMHA correctly, we
                // apply RoPE forward in-place on mScratchK6K AFTER the
                // K6 dequant so scratchPool ends up with post-RoPE
                // K-history. V is RoPE-free; we don't touch mScratchK6V.
                if (this->mRotaryEmbeddingDim > 0)
                {
                    int rcRope = tq_apply_rope_forward_inplace(mScratchK6K,
                        static_cast<int>(kBlockList.size()), nKvHeadsL, kTokensPerBlockB22, dHeadL,
                        this->mRotaryEmbeddingDim, this->mRotaryEmbeddingBase,
                        mSeqBlockIdsK, kDtypeFP16, stream);
                    if (rcRope != 0)
                    {
                        TLLM_LOG_WARNING("[B.2.2 stage 5] RoPE forward returned %d", rcRope);
                    }
                }

                cudaMemsetAsync(mScratchPool, 0, scratchPoolBytes, stream);
                // Place K6 fp16 outputs at the flat slot offsets so
                // the parent's block_offset_value * slot_inner indexing
                // lands on our data.
                for (std::size_t i = 0; i < kBlockList.size(); ++i)
                {
                    std::size_t dstOff
                        = static_cast<std::size_t>(kBlockList[i]) * slotInnerBytes;
                    cudaMemcpyAsync(static_cast<std::uint8_t*>(mScratchPool) + dstOff,
                        static_cast<std::uint8_t const*>(mScratchK6K) + i * slotInnerBytes, slotInnerBytes,
                        cudaMemcpyDeviceToDevice, stream);
                }
                for (std::size_t i = 0; i < vBlockList.size(); ++i)
                {
                    std::size_t dstOff
                        = static_cast<std::size_t>(vBlockList[i]) * slotInnerBytes;
                    cudaMemcpyAsync(static_cast<std::uint8_t*>(mScratchPool) + dstOff,
                        static_cast<std::uint8_t const*>(mScratchK6V) + i * slotInnerBytes, slotInnerBytes,
                        cudaMemcpyDeviceToDevice, stream);
                }

                // Patch pool ptr = mScratchPool - L*kvFactor*slot_inner,
                // so parent's layerOffset addition wraps back to base.
                int const totalPoolEntries = nPools * 2;
                int const safeEntries = std::min(totalPoolEntries,
                    static_cast<int>(sizeof(mPatchedPoolPtrs) / sizeof(std::int64_t)));
                std::memcpy(mPatchedPoolPtrs, hPoolPtrs, safeEntries * sizeof(std::int64_t));
                std::uintptr_t const scratchBiased
                    = reinterpret_cast<std::uintptr_t>(mScratchPool) - layerStrideBytes;
                mPatchedPoolPtrs[poolIdxForLayer * 2 + 0]
                    = static_cast<std::int64_t>(scratchBiased);
                patchedInputs[poolPtrsIdx] = mPatchedPoolPtrs;
                stepB22Done = true;

                // Single-block assumption (smoke: ≤ 32 new tokens in
                // one logical block — covers decode + short prefill).
                // For longer prefill that crosses block boundaries,
                // the post-enqueue gather + K3 needs to run per
                // contiguous (blockInSeq) segment; TODO follow-up.
                int const absPosFirst = seqPastBefore;
                int const absPosLast = seqTotalAfter - 1;
                int const blockFirst = absPosFirst / kTokensPerBlockB22;
                int const blockLast = absPosLast / kTokensPerBlockB22;
                if (blockFirst == blockLast)
                {
                    stage2.active = true;
                    stage2.poolBaseShifted = poolBaseShifted;
                    stage2.slotInnerBytes = slotInnerBytes;
                    stage2.nKvHeads = nKvHeadsL;
                    stage2.dHead = dHeadL;
                    stage2.blockInSeq = blockFirst;
                    stage2.offInBlockFirst = absPosFirst - blockFirst * kTokensPerBlockB22;
                    stage2.kFlat
                        = hBlockOffsets[((s * beamWidth + 0) * kvFactor + 0) * maxBlocksPerSeq + blockFirst];
                    stage2.vFlat
                        = hBlockOffsets[((s * beamWidth + 0) * kvFactor + 1) * maxBlocksPerSeq + blockFirst];
                    stage2.turboquantBits = mTurboquantBits;
                    stage2.nKBlocks = static_cast<int>(kBlockList.size());
                    stage2.nVBlocks = static_cast<int>(vBlockList.size());
                }

                static std::atomic<bool> sB22Logged{false};
                bool expected = false;
                if (sB22Logged.compare_exchange_strong(expected, true))
                {
                    TLLM_LOG_INFO(
                        "[B.2.2] step 9b stage 2 (layer=%d): K6 pre-enqueue + K3 post-enqueue. "
                        "nTokens=%d pastBefore=%d totalAfter=%d nFullBlocks=%d kFlat=%zu vFlat=%zu "
                        "maxFlatSlot=%d scratchPoolBytes=%zu layerStrideBytes=%zu stage2.active=%d",
                        this->mLayerIdx, nTokens, seqPastBefore, seqTotalAfter, nFullBlocks, kBlockList.size(),
                        vBlockList.size(), maxFlatSlot, scratchPoolBytes, layerStrideBytes,
                        stage2.active ? 1 : 0);
                }
            }
        }
    }

    // Fallback: if step 9b didn't run (multi-pool, unexpected layout, etc.),
    // keep the legacy K1+K2-only behavior with no pool-ptr patch — math hook
    // continues to round-trip via patchedInputs[0] = mWorkspace.kv_out.
    static std::atomic<bool> sPatchDiagLogged{false};
    if (!stepB22Done && isEntryUsed(IdxEntry::HOST_KV_CACHE_POOL_POINTERS))
    {
        auto poolPtrsIdx = getIdx(IdxEntry::HOST_KV_CACHE_POOL_POINTERS);
        auto const& d = inputDesc[poolPtrsIdx];
        int totalElems = 1;
        for (int i = 0; i < d.dims.nbDims; ++i)
        {
            totalElems *= d.dims.d[i];
        }
        if (totalElems > 0 && totalElems <= static_cast<int>(sizeof(mPatchedPoolPtrs) / sizeof(std::int64_t)))
        {
            std::memcpy(mPatchedPoolPtrs, inputs[poolPtrsIdx], totalElems * sizeof(std::int64_t));
            patchedInputs[poolPtrsIdx] = mPatchedPoolPtrs;
        }
        bool expected = false;
        if (sPatchDiagLogged.compare_exchange_strong(expected, true) && totalElems > 0)
        {
            std::int64_t const* orig = static_cast<std::int64_t const*>(inputs[poolPtrsIdx]);
            TLLM_LOG_INFO("[B.2.2] pool-ptr no-op patch layer=%d totalElems=%d orig=[0x%lx 0x%lx ...] "
                          "copy=[0x%lx 0x%lx ...]",
                this->mLayerIdx, totalElems, totalElems > 0 ? orig[0] : 0, totalElems > 1 ? orig[1] : 0,
                mPatchedPoolPtrs[0], mPatchedPoolPtrs[1]);
        }
    }
    int const parentRc
        = GPTAttentionPlugin::enqueue(inputDesc, outputDesc, patchedInputs, outputs, workspace, stream);

    // Stage 2 post-enqueue: gather parent's RoPE-applied K + V from
    // mScratchPool (where parent just wrote them) and K3 quantize
    // into the persistent real pool so the NEXT enqueue's K6 dequant
    // returns post-RoPE K matching FMHA's expectation.
    if (parentRc == 0 && stage2.active)
    {
        constexpr int kTokensPerBlock_s2 = 32;
        int const nKvHeadsL = stage2.nKvHeads;
        int const dHeadL = stage2.dHead;
        std::size_t const headRowBytesInSlot
            = static_cast<std::size_t>(kTokensPerBlock_s2) * dHeadL * sizeof(std::uint16_t); // 8 KB
        std::size_t const perTokenBytesDst
            = static_cast<std::size_t>(nKvHeadsL) * dHeadL * sizeof(std::uint16_t); // 2 KB
        std::size_t const tokenBytes
            = static_cast<std::size_t>(dHeadL) * sizeof(std::uint16_t); // 256 B for fp16 d_head=128

        // Stage 5 (option α) — gather K and V from mWorkspace.kv_out
        // (the K1+K2-roundtripped fused QKV, *pre-RoPE*). K3 stores
        // pre-RoPE K so the next enqueue's K6 returns pre-RoPE K;
        // we apply RoPE on K6's output via the kernel above. The
        // noise from K3+K6 now lives in the same pre-RoPE basis as
        // Q's K1+K2 noise and cancels in the Q·K dot product.
        //
        // V is RoPE-free; gathering V from mWorkspace.kv_out is
        // equivalent to gathering from scratchPool (parent's V
        // write is just K1+K2-roundtripped V passed through).
        //
        // mWorkspace.kv_out layout: [n_tokens, n_total_heads, d_head]
        // fp16. K at heads [nQHeads .. nQHeads + nKvHeads); V at
        // heads [nQHeads + nKvHeads .. nQHeads + 2*nKvHeads).
        int const nQHeadsL = this->mNumHeads;
        std::size_t const qkvStrideBytes
            = static_cast<std::size_t>(nQHeadsL + 2 * nKvHeadsL) * dHeadL * sizeof(std::uint16_t);
        std::size_t const kSrcOffsetBytes
            = static_cast<std::size_t>(nQHeadsL) * dHeadL * sizeof(std::uint16_t);
        std::size_t const vSrcOffsetBytes
            = static_cast<std::size_t>(nQHeadsL + nKvHeadsL) * dHeadL * sizeof(std::uint16_t);
        // K slice: nTokens consecutive tokens, each contributes
        // nKvHeads*dHead*sizeof(fp16) = perTokenBytesDst contiguous
        // bytes. src_pitch = qkvStrideBytes, dst_pitch =
        // perTokenBytesDst.
        cudaMemcpy2DAsync(mKContig, perTokenBytesDst,
            static_cast<std::uint8_t const*>(mWorkspace.kv_out) + kSrcOffsetBytes, qkvStrideBytes,
            perTokenBytesDst, nTokens, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(mVContig, perTokenBytesDst,
            static_cast<std::uint8_t const*>(mWorkspace.kv_out) + vSrcOffsetBytes, qkvStrideBytes,
            perTokenBytesDst, nTokens, cudaMemcpyDeviceToDevice, stream);
        // (kSlotBase / vSlotBase / offBytesInRow / headRowBytesInSlot
        // are no longer needed for α; left out vs the stage-2 path.)
        (void)headRowBytesInSlot;
        (void)tokenBytes;

        // Stage 3 bisect switches:
        //   TQ_STAGE2_SKIP_K=1: skip post-enqueue K3 of K (leave K
        //     stale in real pool). If output improves, K3-of-K is the
        //     drift source.
        //   TQ_STAGE2_SKIP_V=1: skip post-enqueue K3 of V.
        //   TQ_STAGE2_DUMP=1: once-only host-side dump of mScratchPool
        //     (post-parent-write) vs mKContig (post-gather) for layer
        //     0 first enqueue, to verify gather lands the bytes we
        //     think it does.
        static bool const sSkipK = []() {
            char const* v = std::getenv("TQ_STAGE2_SKIP_K");
            bool s = (v != nullptr && v[0] == '1');
            if (s) TLLM_LOG_INFO("[B.2.2 stage 3] TQ_STAGE2_SKIP_K=1 (post-enqueue K3 of K disabled)");
            return s;
        }();
        static bool const sSkipV = []() {
            char const* v = std::getenv("TQ_STAGE2_SKIP_V");
            bool s = (v != nullptr && v[0] == '1');
            if (s) TLLM_LOG_INFO("[B.2.2 stage 3] TQ_STAGE2_SKIP_V=1 (post-enqueue K3 of V disabled)");
            return s;
        }();
        // (stage 3 TQ_STAGE2_DUMP removed in stage 5; sources differ now.)

        // K3 quantize gathered K / V into the persistent real pool
        // at the new-token slot positions. Same flat-slot addressing
        // as the pre-enqueue K6: n_layers_per_pool=1, kv_factor=1,
        // relative_layer=0, k_or_v=0; pool_base already shifted.
        int rcK3K = 0;
        if (!sSkipK)
        {
            // K uses the alt codebook from TQ_K_CODEBOOK_PATH if loaded.
            float const* kCentroidsPost = gQuantStateK.active ? gQuantStateK.centroids : gQuantState.centroids;
            float const* kThresholdsPost = gQuantStateK.active ? gQuantStateK.thresholds : gQuantState.thresholds;
            rcK3K = tq_kv_quantize_paged_trtllm_layered(mKContig, mSlotMappingK,
                stage2.poolBaseShifted, gQuantState.signs, kCentroidsPost, kThresholdsPost,
                stage2.turboquantBits, kDtypeFP16, nTokens, nKvHeadsL, dHeadL, kTokensPerBlock_s2,
                /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0,
                /*k_or_v=*/0, stage2.slotInnerBytes, stream);
        }
        int rcK3V = 0;
        if (!sSkipV)
        {
            rcK3V = tq_kv_quantize_paged_trtllm_layered(mVContig, mSlotMappingV,
                stage2.poolBaseShifted, gQuantState.signs, gQuantState.centroids, gQuantState.thresholds,
                stage2.turboquantBits, kDtypeFP16, nTokens, nKvHeadsL, dHeadL, kTokensPerBlock_s2,
                /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0,
                /*k_or_v=*/0, stage2.slotInnerBytes, stream);
        }

        static std::atomic<bool> sStage2Logged{false};
        bool expected = false;
        if (sStage2Logged.compare_exchange_strong(expected, true))
        {
            TLLM_LOG_INFO(
                "[B.2.2] step 9b stage 2 post-enqueue (layer=%d): K3 K rc=%d V rc=%d "
                "(gather: blockInSeq=%d offFirst=%d kFlat=%d vFlat=%d nTokens=%d)",
                this->mLayerIdx, rcK3K, rcK3V, stage2.blockInSeq, stage2.offInBlockFirst,
                stage2.kFlat, stage2.vFlat, nTokens);
        }

        // TQ_RT_SELFCHECK=1: round-trip K6-dequant the K slot we just K3'd
        // back into a fresh temp buffer and compare element-wise to
        // mKContig (the K3 input). This isolates whether the K3+K6
        // layered launcher pair is numerically near-lossless at bits=8,
        // separately from the stage-2 plumbing. Fires once for layer 0,
        // first enqueue.
        static bool const sRtSelfCheck = []() {
            char const* v = std::getenv("TQ_RT_SELFCHECK");
            return v != nullptr && v[0] == '1';
        }();
        static std::atomic<bool> sRtFired{false};
        if (sRtSelfCheck && this->mLayerIdx == 0)
        {
            bool exp2 = false;
            if (sRtFired.compare_exchange_strong(exp2, true))
            {
                std::size_t const rtBytes = static_cast<std::size_t>(stage2.nKBlocks) * stage2.slotInnerBytes;
                void* rtK = nullptr;
                if (cudaMalloc(&rtK, rtBytes) == cudaSuccess)
                {
                    tq_kv_dequantize_paged_trtllm_layered(stage2.poolBaseShifted, mPhysBlocksK,
                        gQuantState.signs, gQuantStateK.active ? gQuantStateK.centroids : gQuantState.centroids,
                        rtK, stage2.turboquantBits, kDtypeFP16,
                        stage2.nKBlocks, nKvHeadsL, dHeadL, kTokensPerBlock_s2,
                        /*n_layers_per_pool=*/1, /*kv_factor=*/1, /*relative_layer=*/0,
                        /*k_or_v=*/0, stage2.slotInnerBytes, stream);
                    cudaStreamSynchronize(stream);
                    // Copy first 8 fp16 of rtK[head=0, slot=offFirst] vs mKContig[t=0, head=0].
                    std::size_t const slotInBytes
                        = static_cast<std::size_t>(stage2.offInBlockFirst) * dHeadL * sizeof(std::uint16_t);
                    std::uint16_t rtVals[8] = {0}, srcVals[8] = {0};
                    cudaMemcpy(rtVals,
                        static_cast<std::uint8_t const*>(rtK) + slotInBytes,
                        sizeof(rtVals), cudaMemcpyDeviceToHost);
                    cudaMemcpy(srcVals, mKContig, sizeof(srcVals), cudaMemcpyDeviceToHost);
                    TLLM_LOG_INFO("[B.2.2 stage 4 self-check] K3+K6 round-trip layer=0 head=0 token=%d first 8 fp16: "
                                  "src=[%04x %04x %04x %04x %04x %04x %04x %04x] "
                                  "rt =[%04x %04x %04x %04x %04x %04x %04x %04x]",
                        stage2.offInBlockFirst,
                        srcVals[0], srcVals[1], srcVals[2], srcVals[3],
                        srcVals[4], srcVals[5], srcVals[6], srcVals[7],
                        rtVals[0], rtVals[1], rtVals[2], rtVals[3],
                        rtVals[4], rtVals[5], rtVals[6], rtVals[7]);
                    cudaFree(rtK);
                }
            }
        }
    }
    return parentRc;
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
    auto freeDev = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    freeDev(mKContig);     mKContigCap = 0;
    freeDev(mVContig);     mVContigCap = 0;
    freeDev(reinterpret_cast<void*&>(mSlotMappingK)); mSlotMappingKCap = 0;
    freeDev(reinterpret_cast<void*&>(mSlotMappingV)); mSlotMappingVCap = 0;
    freeDev(reinterpret_cast<void*&>(mPhysBlocksK));  mPhysBlocksKCap = 0;
    freeDev(reinterpret_cast<void*&>(mPhysBlocksV));  mPhysBlocksVCap = 0;
    freeDev(mScratchK6K);  mScratchK6KCap = 0;
    freeDev(mScratchK6V);  mScratchK6VCap = 0;
    freeDev(mScratchPool); mScratchPoolCap = 0;
    freeDev(reinterpret_cast<void*&>(mSeqBlockIdsK)); mSeqBlockIdsKCap = 0;
    freeDev(reinterpret_cast<void*&>(mSeqBlockIdsV)); mSeqBlockIdsVCap = 0;
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
