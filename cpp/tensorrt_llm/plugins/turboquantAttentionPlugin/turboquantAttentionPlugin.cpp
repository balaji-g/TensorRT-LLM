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
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/plugins/common/plugin.h"

#include <atomic>
#include <cstring>

namespace tensorrt_llm::plugins
{

namespace
{
constexpr char const* TURBOQUANT_ATTENTION_PLUGIN_NAME = "TurboquantAttention";
constexpr char const* TURBOQUANT_ATTENTION_PLUGIN_VERSION = "1";

void validateBits(int bits)
{
    TLLM_CHECK_WITH_INFO(bits == 4 || bits == 8,
        "TurboquantAttentionPlugin: turboquantBits must be 4 or 8, got %d", bits);
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
        "TurboquantAttentionPlugin constructed (layer=%d, bits=%d). enqueue forwards to GPTAttentionPlugin until "
        "the K3/K6 hook lands.",
        layer_idx, mTurboquantBits);
}

TurboquantAttentionPlugin::TurboquantAttentionPlugin(void const* data, size_t length)
    : GPTAttentionPlugin(data, length - sizeof(int))
    , mTurboquantBits(0)
{
    // Trailing int after the parent's serialized payload.
    auto const* tail = static_cast<char const*>(data) + length - sizeof(int);
    std::memcpy(&mTurboquantBits, tail, sizeof(int));
    validateBits(mTurboquantBits);
}

int TurboquantAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // v1: telemetry only — log the first invocation and forward to parent.
    // This proves PluginConfig.turboquant_attention_plugin actually routes
    // through our class at inference time. The K1/K2 round-trip math hook
    // lands in B.2.1b after the kernel sources are vendored.
    static std::atomic<bool> sLogged{false};
    bool expected = false;
    if (sLogged.compare_exchange_strong(expected, true))
    {
        int nbDims = inputDesc[0].dims.nbDims;
        int qkvDim = nbDims > 0 ? inputDesc[0].dims.d[nbDims - 1] : 0;
        int nTokens = 1;
        for (int i = 0; i < nbDims - 1; ++i)
        {
            nTokens *= inputDesc[0].dims.d[i];
        }
        TLLM_LOG_INFO(
            "TurboquantAttentionPlugin::enqueue#1 — bits=%d nTokens=%d qkvDim=%d dtype=%d. "
            "Math hook is OFF in this build (B.2.1b will add K1/K2 round-trip).",
            mTurboquantBits, nTokens, qkvDim, static_cast<int>(inputDesc[0].type));
    }
    return GPTAttentionPlugin::enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream);
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
    // Round-trip via serialize/deserialize so every parent member
    // (including private state in GPTAttentionPluginCommon) is preserved.
    auto const size = getSerializationSize();
    std::vector<char> buffer(size);
    serialize(buffer.data());
    auto* cloned = new TurboquantAttentionPlugin(buffer.data(), size);
    cloned->setPluginNamespace(getPluginNamespace());
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

//
// Creator
//

TurboquantAttentionPluginCreator::TurboquantAttentionPluginCreator()
    : GPTAttentionPluginCreator()
{
    // Start from the parent's field set and append our `bits` knob.
    auto const* parentFC = GPTAttentionPluginCreator::getFieldNames();
    if (parentFC != nullptr)
    {
        mPluginAttributes.reserve(parentFC->nbFields + 1);
        for (int i = 0; i < parentFC->nbFields; ++i)
        {
            mPluginAttributes.push_back(parentFC->fields[i]);
        }
    }
    mPluginAttributes.emplace_back(nvinfer1::PluginField{"turboquant_bits", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
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
    // Delegate the bulk of the parsing to GPTAttentionPluginCreator by
    // constructing a parent and then deserializing it through our
    // own ctor. Simpler than duplicating the ~50-field PluginFieldParser
    // wall here, and bit-identical to what the parent would do.
    auto* parent = static_cast<GPTAttentionPlugin*>(GPTAttentionPluginCreator::createPlugin(name, fc));
    if (parent == nullptr)
    {
        return nullptr;
    }

    int turboquantBits = 8;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        auto const& f = fc->fields[i];
        if (f.name != nullptr && std::strcmp(f.name, "turboquant_bits") == 0
            && f.type == nvinfer1::PluginFieldType::kINT32 && f.data != nullptr)
        {
            turboquantBits = *static_cast<int32_t const*>(f.data);
            break;
        }
    }
    validateBits(turboquantBits);

    auto const parentSize = parent->getSerializationSize();
    std::vector<char> buf(parentSize + sizeof(int));
    parent->serialize(buf.data());
    std::memcpy(buf.data() + parentSize, &turboquantBits, sizeof(int));
    parent->destroy();

    auto* obj = new TurboquantAttentionPlugin(buf.data(), buf.size());
    obj->setPluginNamespace(getPluginNamespace());
    return obj;
}

nvinfer1::IPluginV2* TurboquantAttentionPluginCreator::deserializePlugin(
    char const* /*name*/, void const* serialData, size_t serialLength) noexcept
{
    auto* obj = new TurboquantAttentionPlugin(serialData, serialLength);
    obj->setPluginNamespace(getPluginNamespace());
    return obj;
}

} // namespace tensorrt_llm::plugins
