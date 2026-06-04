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
#pragma once

#include "tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h"

#include <cstddef>
#include <cstdint>

namespace tensorrt_llm::plugins
{

// Per-plugin lazy-allocated GPU workspace for the K1/K2 streaming
// round-trip. Grown on demand, freed in destroy(). Inline here so
// the .cpp can include it without exposing CUDA types in the public
// header.
struct TurboquantEnqueueWorkspace
{
    void* packed{nullptr};
    float* norms{nullptr};
    void* kv_out{nullptr};
    std::size_t packed_capacity{0};
    std::size_t norms_capacity{0};
    std::size_t kv_out_capacity{0};

    // Frees all three buffers. Safe to call multiple times. Defined
    // in the .cpp to keep CUDA includes out of the header.
    void free();
};

// TurboquantAttentionPlugin: in-tree successor to the Phase A drop-in
// adapter at engines/trtllm/cpp/. Subclasses GPTAttentionPlugin so we
// inherit its full enqueue / configurePlugin / clone / supportsFormat
// surface — same TRT input contract, same FMHA dispatch, same RoPE
// wiring. We override only:
//   - getPluginType()/Version()/getPluginName() to register under a
//     distinct name so the engine builder picks our plugin via
//     PluginConfig.turboquant_attention_plugin instead of the
//     hardcoded "GPTAttention" lookup.
//   - clone() to return our type.
//   - serialize/getSerializationSize to add the `bits ∈ {4, 8}` field
//     after the common payload so engines roundtrip.
//
// The enqueue() override (the actual K3/K6 hook) is the next commit;
// this scaffold proves the in-tree plugin registration plumbing
// builds + loads alongside the existing GPTAttentionPlugin.
class TurboquantAttentionPlugin : public GPTAttentionPlugin
{
public:
    // Constructor mirrors GPTAttentionPlugin exactly, with one extra
    // trailing parameter: `turboquantBits ∈ {4, 8}`.
    TurboquantAttentionPlugin(int layer_idx, int num_heads, int vision_start, int vision_length, int num_kv_heads,
        int num_kv_heads_origin, int head_size, int unidirectional, float q_scaling, float attn_logit_softcapping_scale,
        tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type, int rotary_embedding_dim,
        float rotary_embedding_base, tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type,
        float rotary_embedding_scale, float rotary_embedding_short_m_scale, float rotary_embedding_long_m_scale,
        int rotary_embedding_max_positions, int rotary_embedding_original_max_positions, int tp_size, int tp_rank,
        bool unfuse_qkv_gemm, bool use_logn_scaling, tensorrt_llm::kernels::ContextFMHAType context_fmha_type,
        int kv_cache_quant_mode, bool remove_input_padding, tensorrt_llm::kernels::AttentionMaskType mask_type,
        tensorrt_llm::kernels::BlockSparseParams block_sparse_params, bool paged_kv_cache, int tokens_per_block,
        nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled, bool cross_attention,
        int max_distance, bool pos_shift_enabled, bool dense_context_fmha, bool use_paged_context_fmha,
        bool use_fp8_context_fmha, bool has_full_attention_mask, bool use_cache, bool is_spec_decoding_enabled,
        bool spec_decoding_is_generation_length_variable, int32_t spec_decoding_max_generation_length,
        bool is_mla_enabled, int q_lora_rank, int kv_lora_rank, int qk_nope_head_dim, int qk_rope_head_dim,
        int v_head_dim, bool fuse_fp4_quant, bool skip_attn, int cp_size, int cp_rank, std::set<int32_t> cp_group,
        int turboquantBits);

    TurboquantAttentionPlugin(void const* data, size_t length);

    ~TurboquantAttentionPlugin() override = default;

    // IPluginV2 — override identity + clone.
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    TurboquantAttentionPlugin* clone() const noexcept override;

    // IPluginV2DynamicExt — telemetry-wrapped forward. v1 logs the
    // first invocation (shape + bits) and forwards to GPTAttentionPlugin
    // unchanged. The K1/K2 streaming round-trip math hook lands in
    // B.2.1b once the kernel sources are vendored under
    // cpp/tensorrt_llm/kernels/turboquant/.
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // Serialization: append turboquantBits after parent's common payload.
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    // IPluginV2 — release workspace + parent state.
    void destroy() noexcept override;

    [[nodiscard]] int getTurboquantBits() const noexcept
    {
        return mTurboquantBits;
    }

private:
    int mTurboquantBits;
    TurboquantEnqueueWorkspace mWorkspace{};
    // Heap-stable host buffer for the HOST_KV_CACHE_POOL_POINTERS patch.
    // TRT-LLM's gpt_attention reads this input lazily (after our enqueue
    // returns), so a stack-local buffer races with the read. Lives as
    // long as the plugin instance.
    std::int64_t mPatchedPoolPtrs[16] = {0};
};

class TurboquantAttentionPluginCreator : public GPTAttentionPluginCreator
{
public:
    TurboquantAttentionPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC{};
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
