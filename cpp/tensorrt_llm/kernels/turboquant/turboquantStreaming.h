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

#include <cstdint>

// C ABI for the TurboQuant streaming K1 / K2 kernels.
// Definitions live in turboquantStreaming.cu (vendored from
// cuda/turboquant_kernel.cu in the turboquant-rs repo).
//
//   dtype: 0 = fp16, 1 = bf16
//   bits:  4 or 8
//   stream: cudaStream_t cast to void*

extern "C" {

// K1 — streaming quantize.
// Input: d_kv [n_tokens, n_heads, d_head] in `dtype`.
// Output: d_packed [n_tokens, n_heads, d_head * bits / 8] uint8;
//         d_norms  [n_tokens, n_heads] float32.
int tq_kv_quantize_streaming(void const* d_kv, int8_t const* d_signs, float const* d_centroids,
    float const* d_thresholds, void* d_packed, void* d_norms, int bits, int dtype, int n_tokens, int n_heads,
    int d_head, void* stream);

// K2 — streaming dequantize.
// Input: d_packed [n_tokens, n_heads, d_head * bits / 8] uint8;
//        d_norms  [n_tokens, n_heads] float32.
// Output: d_kv_out [n_tokens, n_heads, d_head] in `dtype`.
int tq_kv_dequantize_streaming(void const* d_packed, float const* d_norms, int8_t const* d_signs,
    float const* d_centroids, void* d_kv_out, int bits, int dtype, int n_tokens, int n_heads, int d_head,
    void* stream);

// `layout` argument for the paged kernels:
//   0 = TQ_LAYOUT_VLLM_BLOCKED          — separate packed/norms buffers.
//   2 = TQ_LAYOUT_VLLM_BLOCKED_INLINE_NORMS  — single buffer per block,
//       [ packed_section ][ norms_section ]; caller passes
//       d_packed_cache = pool_base and d_norms_cache =
//       pool_base + packed_section_bytes.
//       packed_section_bytes = n_kv_heads * block_size * d_head * bits / 8
//       This matches TurboquantKVCacheManager (M12.4 B.1.1).

// K3 — paged quantize. Writes one K (or one V) tensor through
// slot_mapping. `entry < 0` in slot_mapping skips that token.
int tq_kv_quantize_paged(void const* d_kv, int32_t const* d_slot_mapping, void* d_packed_cache,
    float* d_norms_cache, int8_t const* d_signs, float const* d_centroids, float const* d_thresholds, int bits,
    int dtype, int layout, int n_tokens, int n_kv_heads, int d_head, int n_blocks, int block_size, void* stream);

// K6 — paged dequantize. Gathers blocks listed in d_physical_block_ids
// into a contiguous scratch tensor
// `[n_scratch_blocks, n_kv_heads, block_size, d_head]`.
int tq_kv_dequantize_paged(void const* d_packed_cache, float const* d_norms_cache, int32_t const* d_physical_block_ids,
    int8_t const* d_signs, float const* d_centroids, void* d_scratch_out, int bits, int dtype, int layout,
    int n_scratch_blocks, int n_kv_heads, int d_head, int block_size, void* stream);

// K3 / K6 — TRT-LLM layered-pool variants. Take pool_base for a
// [nBlocks, nLayers, kvFactor=2, slotInner] cache and the
// (relative_layer, k_or_v) selector; internally compute strides for
// the inline-norms layout TurboquantKVCacheManager allocates.
int tq_kv_quantize_paged_trtllm_layered(void const* d_kv, int32_t const* d_slot_mapping, void* pool_base,
    int8_t const* d_signs, float const* d_centroids, float const* d_thresholds, int bits, int dtype, int n_tokens,
    int n_kv_heads, int d_head, int tokens_per_block, int n_layers_per_pool, int kv_factor, int relative_layer,
    int k_or_v, void* stream);

int tq_kv_dequantize_paged_trtllm_layered(void const* pool_base, int32_t const* d_physical_block_ids,
    int8_t const* d_signs, float const* d_centroids, void* d_scratch_out, int bits, int dtype, int n_scratch_blocks,
    int n_kv_heads, int d_head, int tokens_per_block, int n_layers_per_pool, int kv_factor, int relative_layer,
    int k_or_v, void* stream);

} // extern "C"
