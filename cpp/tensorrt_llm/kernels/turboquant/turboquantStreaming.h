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

} // extern "C"
