// turboquant_kernel.cu — production scoring kernel for the
// turboquant-rs SearchIndex. Distinct from int4_dot_bench.cu, which
// is a microbench with no codebook lookup, no rotation, and no norm.
//
// Algorithm:
//   For each base vector i:
//     score[i] = (norms[i] / sqrt(dim))
//              * sum_k {codebook[idx_k] * rotated_query[k]}
//   where idx_k = unpack<BITS>(packed_indices[i], k) and rotated_query
//   is the Hadamard rotation of the user's query, computed on host.
//
// Bit widths supported: 1, 2, 4, 8 (powers of two, no bit-straddling).
// 3, 5, 6, 7 bit are intentionally not supported here — their packed
// layouts straddle byte boundaries which complicates the kernel; add
// them later if there's a real demand.

#include <cub/device/device_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <new>
#include <stdint.h>

namespace {

// Dtype enum mirroring docs/KERNELS.md §3 — used by the C ABI to
// pick between fp16 and bf16 input/output without overloading the
// pointer type. Kept in sync with the `tq_dtype` Rust enum in
// python/src/lib.rs and the `DTYPE_*` constants in callers.
enum tq_dtype : int {
    TQ_DTYPE_FP16 = 0,
    TQ_DTYPE_BF16 = 1,
};

// fp16 / bf16 → float32 widening, dispatched at compile time. The
// streaming kernels do all interior math in float32 (for the
// Hadamard butterfly + Lloyd-Max threshold compare) and only touch
// the half precision type at the I/O boundary; specialising the
// intrinsics here keeps the kernel body type-agnostic.
template <typename T> __device__ __forceinline__ float widen_to_float(T v);
template <> __device__ __forceinline__ float widen_to_float<__half>(__half v) {
    return __half2float(v);
}
template <> __device__ __forceinline__ float widen_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T narrow_from_float(float v);
template <> __device__ __forceinline__ __half narrow_from_float<__half>(float v) {
    return __float2half(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 narrow_from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <int BITS>
__device__ __forceinline__ int decode_index(uint8_t p, int k) {
    constexpr int MASK = (1 << BITS) - 1;
    return (p >> (k * BITS)) & MASK;
}

template <int BITS>
__global__ void score_kernel(
    const uint8_t* __restrict__ packed_indices,
    const float*   __restrict__ norms,
    const float*   __restrict__ codebook,        // [1 << BITS]
    const float*   __restrict__ rotated_query,   // [dim]
    float*         __restrict__ scores,
    int            dim_bytes,
    float          inv_sqrt_dim
) {
    constexpr int VPB = 8 / BITS;  // values per byte

    extern __shared__ float partial[];
    int vector_id = blockIdx.x;
    int tid       = threadIdx.x;
    const uint8_t* row = packed_indices + (size_t)vector_id * dim_bytes;

    float sum = 0.0f;
    for (int byte = tid; byte < dim_bytes; byte += blockDim.x) {
        uint8_t p     = row[byte];
        int coord_base = byte * VPB;
        #pragma unroll
        for (int k = 0; k < VPB; ++k) {
            int idx = decode_index<BITS>(p, k);
            sum += codebook[idx] * rotated_query[coord_base + k];
        }
    }

    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        scores[vector_id] = partial[0] * inv_sqrt_dim * norms[vector_id];
    }
}

cudaError_t launch_score(
    int             bits,
    int             n_vectors,
    int             dim_bytes,
    int             threads,
    size_t          shared_bytes,
    const uint8_t*  d_packed,
    const float*    d_norms,
    const float*    d_codebook,
    const float*    d_query,
    float*          d_scores,
    float           inv_sqrt_dim
) {
    switch (bits) {
        case 1:
            score_kernel<1><<<n_vectors, threads, shared_bytes>>>(
                d_packed, d_norms, d_codebook, d_query, d_scores,
                dim_bytes, inv_sqrt_dim);
            break;
        case 2:
            score_kernel<2><<<n_vectors, threads, shared_bytes>>>(
                d_packed, d_norms, d_codebook, d_query, d_scores,
                dim_bytes, inv_sqrt_dim);
            break;
        case 4:
            score_kernel<4><<<n_vectors, threads, shared_bytes>>>(
                d_packed, d_norms, d_codebook, d_query, d_scores,
                dim_bytes, inv_sqrt_dim);
            break;
        case 8:
            score_kernel<8><<<n_vectors, threads, shared_bytes>>>(
                d_packed, d_norms, d_codebook, d_query, d_scores,
                dim_bytes, inv_sqrt_dim);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

bool valid_bits_dim(int bits, int dim) {
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) return false;
    if (dim <= 0) return false;
    int vpb = 8 / bits;
    return (dim % vpb) == 0;
}

// Fill an int array with [0, 1, ..., n-1]. Used once at index creation
// to seed the immutable index-side input to cub::DeviceRadixSort::
// SortPairsDescending, which never mutates its inputs so we only need
// to set this up once and reuse across queries.
__global__ void iota_kernel(int* __restrict__ indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        indices[tid] = tid;
    }
}

// --------------------------------------------------------------------
// Streaming K/V cache kernels (M6.3)
//
// One block per (token, head). Each block:
//   1. Load `d_head` fp16 elements into shared memory as fp32.
//   2. Compute L2 norm (warp+shared reduction).
//   3. Normalise + apply Hadamard signs.
//   4. In-place Hadamard butterfly over the head (d_head must be a
//      power of two).
//   5. For each rotated value, find the codebook bin via partition-point
//      on the threshold array.
//   6. Pack indices into bytes (BITS in {1, 2, 4, 8}; only bits=8 in
//      this MVP — bits<8 follows once we settle the index-streaming
//      contract with the vLLM plugin).
// --------------------------------------------------------------------

template <int BITS>
__device__ __forceinline__ uint8_t encode_index(float v, const float* thresh) {
    int idx = 0;
    constexpr int N_THRESH = (1 << BITS) - 1;
    #pragma unroll
    for (int t = 0; t < N_THRESH; ++t) {
        if (v > thresh[t]) idx = t + 1;
    }
    return static_cast<uint8_t>(idx);
}

template <int BITS, typename T>
__global__ void kv_quantize_per_head_kernel(
    const T*       __restrict__ kv,                  // [n_tokens * n_heads * d_head]
    const int8_t*  __restrict__ rotation_signs,      // [d_head]
    const float*   __restrict__ centroids,           // [1 << BITS]
    const float*   __restrict__ thresholds,          // [(1 << BITS) - 1]
    uint8_t*       __restrict__ packed_out,          // [n_tokens * n_heads * dim_bytes]
    float*         __restrict__ norms_out,           // [n_tokens * n_heads]
    int            n_heads,
    int            d_head
) {
    constexpr int VPB = 8 / BITS;
    int token = blockIdx.x;
    int head  = blockIdx.y;
    int tid   = threadIdx.x;

    extern __shared__ float smem[];
    float* head_data = smem;                  // [d_head]
    float* reduce_buf = smem + d_head;        // [blockDim.x]

    const T* head_in = kv + ((size_t)token * n_heads + head) * d_head;
    uint8_t* head_out =
        packed_out + ((size_t)token * n_heads + head) * (d_head / VPB);

    // 1. fp16/bf16 -> fp32 + partial sum-of-squares. The widening
    //    happens via widen_to_float<T> which compile-time dispatches
    //    to __half2float or __bfloat162float.
    float local_ss = 0.0f;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float v = widen_to_float<T>(head_in[i]);
        head_data[i] = v;
        local_ss += v * v;
    }
    reduce_buf[tid] = local_ss;
    __syncthreads();

    // 2. block-wide reduce of sum-of-squares.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float norm = sqrtf(reduce_buf[0]);

    if (norm == 0.0f) {
        // Zero head: write all-zero packed + zero norm and bail.
        for (int b = tid; b < d_head / VPB; b += blockDim.x) {
            head_out[b] = 0;
        }
        if (tid == 0) {
            norms_out[(size_t)token * n_heads + head] = 0.0f;
        }
        return;
    }

    // 3. Normalise and apply Hadamard signs (in shared mem).
    for (int i = tid; i < d_head; i += blockDim.x) {
        head_data[i] = (head_data[i] / norm) * (float)rotation_signs[i];
    }
    __syncthreads();

    // 4. In-place Hadamard butterfly. d_head must be power of two.
    for (int step = 1; step < d_head; step <<= 1) {
        // Each "pair index" in [0, d_head/2) handles one butterfly slot.
        for (int p = tid; p < d_head / 2; p += blockDim.x) {
            int group = p / step;
            int off = p % step;
            int i = group * step * 2 + off;
            int j = i + step;
            float a = head_data[i];
            float b = head_data[j];
            head_data[i] = a + b;
            head_data[j] = a - b;
        }
        __syncthreads();
    }
    // After Hadamard: head_data is the un-normalised rotation; the
    // 1/sqrt(d) scale that the CPU `RandomHadamard::rotate` applies is
    // exactly cancelled by the *sqrt(d)* the encoder multiplies by, so
    // we leave the values as-is and feed them straight into the encoder
    // — this matches `MseQuantizer::quantize` byte-for-byte.

    // Suppress unused-parameter warning when BITS == 8.
    (void)centroids;

    // 5+6. Encode + pack VPB consecutive coords into one byte.
    for (int byte = tid; byte < d_head / VPB; byte += blockDim.x) {
        uint8_t packed = 0;
        #pragma unroll
        for (int k = 0; k < VPB; ++k) {
            int coord = byte * VPB + k;
            uint8_t idx = encode_index<BITS>(head_data[coord], thresholds);
            packed |= (uint8_t)(idx << (k * BITS));
        }
        head_out[byte] = packed;
    }

    // 7. Write per-head norm.
    if (tid == 0) {
        norms_out[(size_t)token * n_heads + head] = norm;
    }
}

template <int BITS, typename T>
__global__ void kv_dequantize_per_head_kernel(
    const uint8_t* __restrict__ packed,              // [n_tokens * n_heads * dim_bytes]
    const float*   __restrict__ norms,               // [n_tokens * n_heads]
    const int8_t*  __restrict__ rotation_signs,      // [d_head]
    const float*   __restrict__ centroids,           // [1 << BITS]
    T*             __restrict__ kv_out,              // [n_tokens * n_heads * d_head]
    int            n_heads,
    int            d_head
) {
    constexpr int VPB = 8 / BITS;
    int token = blockIdx.x;
    int head  = blockIdx.y;
    int tid   = threadIdx.x;

    extern __shared__ float smem[];
    float* head_data = smem;  // [d_head]

    const uint8_t* head_in =
        packed + ((size_t)token * n_heads + head) * (d_head / VPB);
    T* head_out = kv_out + ((size_t)token * n_heads + head) * d_head;
    float head_norm = norms[(size_t)token * n_heads + head];

    if (head_norm == 0.0f) {
        for (int i = tid; i < d_head; i += blockDim.x) {
            head_out[i] = narrow_from_float<T>(0.0f);
        }
        return;
    }

    // 1. Decode codebook bins into rotated-space float values.
    for (int byte = tid; byte < d_head / VPB; byte += blockDim.x) {
        uint8_t p = head_in[byte];
        int coord_base = byte * VPB;
        #pragma unroll
        for (int k = 0; k < VPB; ++k) {
            int idx = decode_index<BITS>(p, k);
            head_data[coord_base + k] = centroids[idx];
        }
    }
    __syncthreads();

    // 2. Inverse Hadamard butterfly. The forward pass above wrote
    //    `H · (signs · (x / norm))`. The inverse (since Hadamard is
    //    self-inverse up to scale 1/d) gives back `signs · (x / norm)`
    //    after a final 1/d scale. We then apply signs and norm.
    for (int step = 1; step < d_head; step <<= 1) {
        for (int p = tid; p < d_head / 2; p += blockDim.x) {
            int group = p / step;
            int off = p % step;
            int i = group * step * 2 + off;
            int j = i + step;
            float a = head_data[i];
            float b = head_data[j];
            head_data[i] = a + b;
            head_data[j] = a - b;
        }
        __syncthreads();
    }

    // 3. Final scale (1/d_head from the inverse Hadamard, signs, norm)
    //    and write fp16/bf16 output via the compile-time-dispatched
    //    narrowing helper.
    float inv_d = 1.0f / (float)d_head;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float v = head_data[i] * inv_d * (float)rotation_signs[i] * head_norm;
        head_out[i] = narrow_from_float<T>(v);
    }
}

// Launch dispatch macros: one BITS × dtype combination expands per
// macro invocation. Keeps the if/switch ladder readable and ensures
// the only thing varying between fp16 and bf16 paths is the
// templated kernel instantiation.
#define TQ_LAUNCH_QUANTIZE(BITS_VAL, T)                                        \
    kv_quantize_per_head_kernel<BITS_VAL, T><<<grid, threads, smem, stream>>>( \
        static_cast<const T*>(d_kv),                                           \
        d_rotation_signs, d_centroids, d_thresholds,                           \
        static_cast<uint8_t*>(const_cast<void*>(d_packed)),                    \
        static_cast<float*>(const_cast<void*>(d_norms)),                       \
        n_heads, d_head)

#define TQ_LAUNCH_DEQUANTIZE(BITS_VAL, T)                                      \
    kv_dequantize_per_head_kernel<BITS_VAL, T><<<grid, threads, smem, stream>>>( \
        static_cast<const uint8_t*>(d_packed),                                 \
        static_cast<const float*>(d_norms),                                    \
        d_rotation_signs, d_centroids,                                         \
        static_cast<T*>(d_out_kv_or_packed),                                   \
        n_heads, d_head)

cudaError_t launch_kv_streaming(
    bool        quantize,
    int         bits,
    int         dtype,                 // tq_dtype enum: TQ_DTYPE_FP16 / TQ_DTYPE_BF16
    int         n_tokens,
    int         n_heads,
    int         d_head,
    const void* d_kv,                  // for quantize: const __half/__nv_bfloat16*; for dequant: unused
    const void* d_packed,              // const uint8_t*
    const void* d_norms,               // const float*
    const int8_t* d_rotation_signs,
    const float*  d_centroids,
    const float*  d_thresholds,
    void*       d_out_kv_or_packed,    // for dequant: __half/__nv_bfloat16* output; for quant: unused
    cudaStream_t stream
) {
    // d_head must be a power of two for Hadamard.
    if ((d_head & (d_head - 1)) != 0 || d_head <= 0 || d_head > 1024) {
        return cudaErrorInvalidValue;
    }
    if (dtype != TQ_DTYPE_FP16 && dtype != TQ_DTYPE_BF16) {
        return cudaErrorInvalidValue;
    }

    int threads = 64;
    if (d_head >= 128) threads = 128;
    if (d_head >= 256) threads = 256;

    dim3 grid(n_tokens, n_heads);
    size_t smem = (d_head + threads) * sizeof(float);

    if (quantize) {
        if (dtype == TQ_DTYPE_FP16) {
            switch (bits) {
                case 8: TQ_LAUNCH_QUANTIZE(8, __half); break;
                case 4: TQ_LAUNCH_QUANTIZE(4, __half); break;
                default: return cudaErrorInvalidValue;
            }
        } else {  // TQ_DTYPE_BF16
            switch (bits) {
                case 8: TQ_LAUNCH_QUANTIZE(8, __nv_bfloat16); break;
                case 4: TQ_LAUNCH_QUANTIZE(4, __nv_bfloat16); break;
                default: return cudaErrorInvalidValue;
            }
        }
    } else {
        if (dtype == TQ_DTYPE_FP16) {
            switch (bits) {
                case 8: TQ_LAUNCH_DEQUANTIZE(8, __half); break;
                case 4: TQ_LAUNCH_DEQUANTIZE(4, __half); break;
                default: return cudaErrorInvalidValue;
            }
        } else {  // TQ_DTYPE_BF16
            switch (bits) {
                case 8: TQ_LAUNCH_DEQUANTIZE(8, __nv_bfloat16); break;
                case 4: TQ_LAUNCH_DEQUANTIZE(4, __nv_bfloat16); break;
                default: return cudaErrorInvalidValue;
            }
        }
    }
    return cudaGetLastError();
}

#undef TQ_LAUNCH_QUANTIZE
#undef TQ_LAUNCH_DEQUANTIZE

// =====================================================================
// K3 — paged write kernel (docs/KERNELS.md §3.3).
//
// Compresses incoming K/V tokens and writes them into a paged
// per-(block, head, slot) cache. Same Hadamard + Lloyd-Max math as
// K1; the only thing that differs is slot addressing — each token
// is steered to a destination slot via `slot_mapping[token]`, and a
// negative entry skips the token (vLLM padding-token convention).
//
// One CUDA block per (token, head) pair. Caller supplies the
// packed/norm cache strides as elements (computed by
// `launch_paged_quantize` from a layout enum), so the kernel itself
// is layout-agnostic.
// =====================================================================

enum tq_paged_layout : int {
    // vLLM-style separate packed and norms cache buffers:
    //   packed_cache: [n_blocks, n_kv_heads, block_size, packed_bytes_per_head] uint8
    //   norms_cache:  [n_blocks, n_kv_heads, block_size]                        fp32
    // Recommended layout per docs/KERNELS.md §3.3. Used by vLLM's
    // BlockManager and (with a thin remap) by SGLang.
    TQ_LAYOUT_VLLM_BLOCKED   = 0,
    // TRT-LLM pagedKVCache layout deferred to M11 alongside the C++
    // adapter that consumes it. Adding it without a caller would be
    // dead code; adding it as a stub would violate the "no cheap
    // tricks" rule the kernels-v1 contract was written under.
    // TQ_LAYOUT_TRTLLM_PAGED = 1,
    // M12.4 B.2.2 — Option A inline-norms layout, used by the
    // TurboquantKVCacheManager subclass in the TRT-LLM fork (see
    // docs/trtllm-cpp-design.md §4.2). Each per-(block, layer, K-or-V)
    // slot stores the packed bytes for all heads, then the fp32 norm
    // for all heads contiguously:
    //   block_layout = [ n_kv_heads * block_size * packed_bytes_per_head bytes
    //                  | n_kv_heads * block_size * sizeof(float)    bytes ]
    // Callers pass d_packed_cache = pool_base and d_norms_cache =
    // pool_base + packed_section_bytes; this case computes strides
    // that walk the FULL per-block stride (packed + norms) on the
    // block-step axis.
    TQ_LAYOUT_VLLM_BLOCKED_INLINE_NORMS = 2,
};

template <int BITS, typename T>
__global__ void paged_quantize_per_head_kernel(
    const T*       __restrict__ kv,                  // [n_tokens, n_kv_heads, d_head]
    const int32_t* __restrict__ slot_mapping,        // [n_tokens]
    const int8_t*  __restrict__ rotation_signs,      // [d_head]
    const float*   __restrict__ centroids,           // [1 << BITS]
    const float*   __restrict__ thresholds,          // [(1 << BITS) - 1]
    uint8_t*       __restrict__ packed_cache,        // strided per `packed_*_stride`
    float*         __restrict__ norms_cache,         // strided per `norms_*_stride`
    int            n_kv_heads,
    int            d_head,
    int            block_size,
    int            packed_block_stride,              // elements (bytes)
    int            packed_head_stride,
    int            packed_slot_stride,
    int            norms_block_stride,               // elements (fp32)
    int            norms_head_stride,
    int            norms_slot_stride
) {
    constexpr int VPB = 8 / BITS;
    int token = blockIdx.x;
    int head  = blockIdx.y;
    int tid   = threadIdx.x;

    // Slot < 0 → padding token, skip. vLLM emits this for batched
    // prefill alignment; SGLang does too.
    int32_t slot = slot_mapping[token];
    if (slot < 0) return;

    int block_idx       = slot / block_size;
    int offset_in_block = slot - block_idx * block_size;

    extern __shared__ float smem[];
    float* head_data  = smem;                  // [d_head]
    float* reduce_buf = smem + d_head;         // [blockDim.x]

    const T* head_in = kv + ((size_t)token * n_kv_heads + head) * d_head;
    uint8_t* packed_out = packed_cache
        + (size_t)block_idx       * (size_t)packed_block_stride
        + (size_t)head            * (size_t)packed_head_stride
        + (size_t)offset_in_block * (size_t)packed_slot_stride;
    float* norm_out = norms_cache
        + (size_t)block_idx       * (size_t)norms_block_stride
        + (size_t)head            * (size_t)norms_head_stride
        + (size_t)offset_in_block * (size_t)norms_slot_stride;

    // 1. fp16/bf16 → fp32 + partial sum-of-squares.
    float local_ss = 0.0f;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float v = widen_to_float<T>(head_in[i]);
        head_data[i] = v;
        local_ss += v * v;
    }
    reduce_buf[tid] = local_ss;
    __syncthreads();

    // 2. Block-wide reduce of sum-of-squares.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float norm = sqrtf(reduce_buf[0]);

    if (norm == 0.0f) {
        // All-zero head: packed bytes zero, norm zero, bail.
        for (int b = tid; b < d_head / VPB; b += blockDim.x) {
            packed_out[b] = 0;
        }
        if (tid == 0) *norm_out = 0.0f;
        return;
    }

    // 3. Normalise and apply Hadamard signs (in shared mem).
    for (int i = tid; i < d_head; i += blockDim.x) {
        head_data[i] = (head_data[i] / norm) * (float)rotation_signs[i];
    }
    __syncthreads();

    // 4. In-place Hadamard butterfly. d_head must be power of two.
    for (int step = 1; step < d_head; step <<= 1) {
        for (int p = tid; p < d_head / 2; p += blockDim.x) {
            int group = p / step;
            int off   = p % step;
            int i     = group * step * 2 + off;
            int j     = i + step;
            float a = head_data[i];
            float b = head_data[j];
            head_data[i] = a + b;
            head_data[j] = a - b;
        }
        __syncthreads();
    }
    // 1/sqrt(d) and sqrt(d) cancel against the encoder's scale, same
    // as in the streaming kernel — matches MseQuantizer::quantize on
    // CPU byte-for-byte.
    (void)centroids;

    // 5+6. Encode + pack VPB consecutive coords per byte.
    for (int byte = tid; byte < d_head / VPB; byte += blockDim.x) {
        uint8_t packed = 0;
        #pragma unroll
        for (int k = 0; k < VPB; ++k) {
            int coord = byte * VPB + k;
            uint8_t idx = encode_index<BITS>(head_data[coord], thresholds);
            packed |= (uint8_t)(idx << (k * BITS));
        }
        packed_out[byte] = packed;
    }

    // 7. Write per-head norm.
    if (tid == 0) *norm_out = norm;
}

#define TQ_LAUNCH_PAGED_QUANTIZE(BITS_VAL, T)                                         \
    paged_quantize_per_head_kernel<BITS_VAL, T><<<grid, threads, smem, stream>>>(     \
        static_cast<const T*>(d_kv),                                                  \
        d_slot_mapping,                                                               \
        d_signs, d_centroids, d_thresholds,                                           \
        d_packed_cache, d_norms_cache,                                                \
        n_kv_heads, d_head, block_size,                                               \
        packed_block_stride, packed_head_stride, packed_slot_stride,                  \
        norms_block_stride, norms_head_stride, norms_slot_stride)

cudaError_t launch_paged_quantize(
    int            dtype,
    int            bits,
    int            layout,
    int            n_tokens,
    int            n_kv_heads,
    int            d_head,
    int            n_blocks,
    int            block_size,
    const void*    d_kv,
    const int32_t* d_slot_mapping,
    void*          d_packed_cache_v,    // uint8*
    float*         d_norms_cache,
    const int8_t*  d_signs,
    const float*   d_centroids,
    const float*   d_thresholds,
    cudaStream_t   stream
) {
    if ((d_head & (d_head - 1)) != 0 || d_head <= 0 || d_head > 1024) {
        return cudaErrorInvalidValue;
    }
    if (dtype != TQ_DTYPE_FP16 && dtype != TQ_DTYPE_BF16) {
        return cudaErrorInvalidValue;
    }
    if (n_tokens <= 0 || n_kv_heads <= 0 || n_blocks <= 0 || block_size <= 0) {
        return cudaErrorInvalidValue;
    }
    if (bits != 4 && bits != 8) {
        return cudaErrorInvalidValue;
    }

    const int packed_bytes_per_head = d_head * bits / 8;

    int packed_block_stride = 0;
    int packed_head_stride  = 0;
    int packed_slot_stride  = 0;
    int norms_block_stride  = 0;
    int norms_head_stride   = 0;
    int norms_slot_stride   = 0;

    switch (layout) {
        case TQ_LAYOUT_VLLM_BLOCKED:
            // packed_cache: [n_blocks, n_kv_heads, block_size, packed_bytes_per_head] uint8
            packed_block_stride = n_kv_heads * block_size * packed_bytes_per_head;
            packed_head_stride  = block_size * packed_bytes_per_head;
            packed_slot_stride  = packed_bytes_per_head;
            // norms_cache: [n_blocks, n_kv_heads, block_size] fp32
            norms_block_stride  = n_kv_heads * block_size;
            norms_head_stride   = block_size;
            norms_slot_stride   = 1;
            break;
        case TQ_LAYOUT_VLLM_BLOCKED_INLINE_NORMS: {
            // Per-(block, K-or-V) layout in a single buffer:
            //   [ packed_section ][ norms_section ]
            //   packed_section: n_kv_heads * block_size * packed_bytes_per_head bytes
            //   norms_section:  n_kv_heads * block_size * sizeof(float)         bytes
            // Caller passes:
            //   d_packed_cache = pool_base
            //   d_norms_cache  = pool_base + packed_section_bytes
            // Strides walk the FULL block on the block axis so block-N's
            // packed + norms regions land at the right address.
            int const packed_section_bytes = n_kv_heads * block_size * packed_bytes_per_head;
            int const norms_section_bytes  = n_kv_heads * block_size * (int)sizeof(float);
            int const total_block_bytes    = packed_section_bytes + norms_section_bytes;
            packed_block_stride = total_block_bytes;
            packed_head_stride  = block_size * packed_bytes_per_head;
            packed_slot_stride  = packed_bytes_per_head;
            norms_block_stride  = total_block_bytes / (int)sizeof(float);
            norms_head_stride   = block_size;
            norms_slot_stride   = 1;
            break;
        }
        default:
            return cudaErrorInvalidValue;
    }

    uint8_t* d_packed_cache = static_cast<uint8_t*>(d_packed_cache_v);

    int threads = 64;
    if (d_head >= 128) threads = 128;
    if (d_head >= 256) threads = 256;

    dim3 grid(n_tokens, n_kv_heads);
    size_t smem = (d_head + threads) * sizeof(float);

    if (dtype == TQ_DTYPE_FP16) {
        switch (bits) {
            case 8: TQ_LAUNCH_PAGED_QUANTIZE(8, __half); break;
            case 4: TQ_LAUNCH_PAGED_QUANTIZE(4, __half); break;
            default: return cudaErrorInvalidValue;
        }
    } else {  // TQ_DTYPE_BF16
        switch (bits) {
            case 8: TQ_LAUNCH_PAGED_QUANTIZE(8, __nv_bfloat16); break;
            case 4: TQ_LAUNCH_PAGED_QUANTIZE(4, __nv_bfloat16); break;
            default: return cudaErrorInvalidValue;
        }
    }
    return cudaGetLastError();
}

#undef TQ_LAUNCH_PAGED_QUANTIZE

// =====================================================================
// K6 — paged read (gather + dequantize blocks → fp16/bf16 scratch).
//
// The read-path counterpart of K3 for the v1 vLLM/SGLang integration:
// when an attention forward step needs the K/V history, the engine
// passes a list of `physical_block_ids` from its block_table, and we
// dequantize each block's slots into a contiguous scratch tensor
// shaped `[n_scratch_blocks, n_kv_heads, block_size, d_head]`. The
// engine then hands the scratch to FlashAttention (or any standard
// fp16/bf16 attention kernel) which reads it as if it were the
// uncompressed cache, with a remapped block_table pointing at scratch
// indices.
//
// This deliberately keeps the persistent cache compressed (real
// memory savings) while letting attention compute on fp16/bf16 — the
// "decompress-to-scratch then call FA" v1 design from M6.5. The
// scratch is bounded by the active block_table size per forward step
// and freed afterwards.
//
// One CUDA block per (scratch_token, kv_head). Same Hadamard + Lloyd-
// Max dequantize as K2; layout addressing is the only thing different.
// =====================================================================

template <int BITS, typename T>
__global__ void paged_dequantize_per_token_kernel(
    const uint8_t* __restrict__ packed_cache,         // vllm_blocked
    const float*   __restrict__ norms_cache,
    const int32_t* __restrict__ physical_block_ids,   // [n_scratch_blocks]
    const int8_t*  __restrict__ rotation_signs,       // [d_head]
    const float*   __restrict__ centroids,            // [1 << BITS]
    T*             __restrict__ scratch_out,          // [n_scratch_blocks, n_kv_heads, block_size, d_head]
    int            n_kv_heads,
    int            d_head,
    int            block_size,
    int            packed_block_stride,
    int            packed_kv_head_stride,
    int            packed_slot_stride,
    int            norms_block_stride,
    int            norms_kv_head_stride,
    int            norms_slot_stride
) {
    constexpr int VPB = 8 / BITS;
    int scratch_token_idx = blockIdx.x;
    int kv_head           = blockIdx.y;
    int tid               = threadIdx.x;

    int scratch_block_idx = scratch_token_idx / block_size;
    int slot              = scratch_token_idx - scratch_block_idx * block_size;
    int phys_block        = physical_block_ids[scratch_block_idx];

    const uint8_t* packed_head = packed_cache
        + (size_t)phys_block * (size_t)packed_block_stride
        + (size_t)kv_head    * (size_t)packed_kv_head_stride
        + (size_t)slot       * (size_t)packed_slot_stride;
    float head_norm = norms_cache[
        (size_t)phys_block * (size_t)norms_block_stride
      + (size_t)kv_head    * (size_t)norms_kv_head_stride
      + (size_t)slot       * (size_t)norms_slot_stride
    ];

    // Output address: scratch[scratch_block_idx, kv_head, slot, :]
    T* out = scratch_out
        + (((size_t)scratch_block_idx * n_kv_heads + kv_head) * block_size + slot)
          * d_head;

    extern __shared__ float smem[];
    float* head_data = smem;

    if (head_norm == 0.0f) {
        for (int i = tid; i < d_head; i += blockDim.x) {
            out[i] = narrow_from_float<T>(0.0f);
        }
        return;
    }

    // 1. Decode bytes → centroids.
    for (int byte = tid; byte < d_head / VPB; byte += blockDim.x) {
        uint8_t p = packed_head[byte];
        int coord_base = byte * VPB;
        #pragma unroll
        for (int k = 0; k < VPB; ++k) {
            int idx = decode_index<BITS>(p, k);
            head_data[coord_base + k] = centroids[idx];
        }
    }
    __syncthreads();

    // 2. Inverse Hadamard butterfly.
    for (int step = 1; step < d_head; step <<= 1) {
        for (int p = tid; p < d_head / 2; p += blockDim.x) {
            int group = p / step;
            int off   = p - group * step;
            int i     = group * step * 2 + off;
            int j     = i + step;
            float a = head_data[i];
            float b = head_data[j];
            head_data[i] = a + b;
            head_data[j] = a - b;
        }
        __syncthreads();
    }

    // 3. Scale and write.
    float inv_d = 1.0f / (float)d_head;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float v = head_data[i] * inv_d * (float)rotation_signs[i] * head_norm;
        out[i] = narrow_from_float<T>(v);
    }
}

#define TQ_LAUNCH_PAGED_DEQUANT(BITS_VAL, T)                                              \
    paged_dequantize_per_token_kernel<BITS_VAL, T><<<grid, threads, smem, stream>>>(      \
        d_packed_cache, d_norms_cache, d_physical_block_ids,                              \
        d_signs, d_centroids,                                                             \
        static_cast<T*>(d_scratch_out),                                                   \
        n_kv_heads, d_head, block_size,                                                   \
        packed_block_stride, packed_kv_head_stride, packed_slot_stride,                   \
        norms_block_stride, norms_kv_head_stride, norms_slot_stride)

cudaError_t launch_paged_dequantize(
    int            dtype,
    int            bits,
    int            layout,
    int            n_scratch_blocks,
    int            n_kv_heads,
    int            d_head,
    int            block_size,
    const uint8_t* d_packed_cache,
    const float*   d_norms_cache,
    const int32_t* d_physical_block_ids,
    const int8_t*  d_signs,
    const float*   d_centroids,
    void*          d_scratch_out,
    cudaStream_t   stream
) {
    if ((d_head & (d_head - 1)) != 0 || d_head <= 0 || d_head > 1024) {
        return cudaErrorInvalidValue;
    }
    if (dtype != TQ_DTYPE_FP16 && dtype != TQ_DTYPE_BF16) {
        return cudaErrorInvalidValue;
    }
    if (bits != 4 && bits != 8) {
        return cudaErrorInvalidValue;
    }
    if (n_scratch_blocks <= 0 || n_kv_heads <= 0 || block_size <= 0) {
        return cudaErrorInvalidValue;
    }

    const int packed_bytes_per_head = d_head * bits / 8;
    int packed_block_stride   = 0;
    int packed_kv_head_stride = 0;
    int packed_slot_stride    = 0;
    int norms_block_stride    = 0;
    int norms_kv_head_stride  = 0;
    int norms_slot_stride     = 0;

    switch (layout) {
        case TQ_LAYOUT_VLLM_BLOCKED:
            packed_block_stride   = n_kv_heads * block_size * packed_bytes_per_head;
            packed_kv_head_stride = block_size * packed_bytes_per_head;
            packed_slot_stride    = packed_bytes_per_head;
            norms_block_stride    = n_kv_heads * block_size;
            norms_kv_head_stride  = block_size;
            norms_slot_stride     = 1;
            break;
        case TQ_LAYOUT_VLLM_BLOCKED_INLINE_NORMS: {
            // Mirror of launch_paged_quantize's inline-norms case. See
            // that comment for layout. Caller passes d_packed_cache =
            // pool_base, d_norms_cache = pool_base + packed_section_bytes.
            int const packed_section_bytes = n_kv_heads * block_size * packed_bytes_per_head;
            int const norms_section_bytes  = n_kv_heads * block_size * (int)sizeof(float);
            int const total_block_bytes    = packed_section_bytes + norms_section_bytes;
            packed_block_stride   = total_block_bytes;
            packed_kv_head_stride = block_size * packed_bytes_per_head;
            packed_slot_stride    = packed_bytes_per_head;
            norms_block_stride    = total_block_bytes / (int)sizeof(float);
            norms_kv_head_stride  = block_size;
            norms_slot_stride     = 1;
            break;
        }
        default:
            return cudaErrorInvalidValue;
    }

    int threads = 64;
    if (d_head >= 128) threads = 128;
    if (d_head >= 256) threads = 256;

    dim3 grid(n_scratch_blocks * block_size, n_kv_heads);
    size_t smem = (size_t)d_head * sizeof(float);

    if (dtype == TQ_DTYPE_FP16) {
        switch (bits) {
            case 8: TQ_LAUNCH_PAGED_DEQUANT(8, __half); break;
            case 4: TQ_LAUNCH_PAGED_DEQUANT(4, __half); break;
            default: return cudaErrorInvalidValue;
        }
    } else {
        switch (bits) {
            case 8: TQ_LAUNCH_PAGED_DEQUANT(8, __nv_bfloat16); break;
            case 4: TQ_LAUNCH_PAGED_DEQUANT(4, __nv_bfloat16); break;
            default: return cudaErrorInvalidValue;
        }
    }
    return cudaGetLastError();
}

#undef TQ_LAUNCH_PAGED_DEQUANT

// =====================================================================
// K5 — paged attention decode (docs/KERNELS.md §3.4.3).
//
// One CUDA block per (seq, q_head). For each sequence:
//   1. Load Q for this q_head into shared memory.
//   2. Pass 1: walk the sequence's block_table; for each (block, slot)
//      dequantize the K head into shared scratch, compute the dot
//      product with Q, and write `scores[token_idx]`.
//   3. Pass 2: softmax(scores) (single-pass, max+exp+normalise).
//      Then walk the same blocks again, dequantize V per slot,
//      accumulate `o[tid] += probs[token_idx] * V[token_idx, tid]`
//      cooperatively across the d_head threads.
//   4. Write `out[seq, q_head, tid] = o[tid]`.
//
// Two-pass deliberately — online-softmax tiling is the perf
// optimisation; for v1 correctness we keep `scores[seq_len]` in
// shared memory and walk the cache twice. Shared memory for the
// Llama-3 reference workload (d=128, seq=2048): ~9.5 KB / block,
// fits comfortably in H100's 100 KB/SM budget.
//
// GQA: callers may pass n_q_heads ≥ n_kv_heads. The kernel maps
// `kv_head = q_head / (n_q_heads / n_kv_heads)`. Q heads in the
// same group share K/V cache.
// =====================================================================

// M10.8 — block-parallel dequantize. Same Hadamard + Lloyd-Max
// math as `dequantize_one_head_to_shared` below, but processes
// `block_size` slots concurrently, sharing each butterfly step's
// __syncthreads() across all slots in the block. Brings K5 decode
// from O(block_size * log d) syncs per K block down to O(log d) —
// ~16× fewer block-wide syncs at block_size=16. The cost is one
// extra `block_size * d_head * 4` bytes of shared memory per CUDA
// block.
template <int BITS>
__device__ __forceinline__ void dequantize_block_to_shared(
    const uint8_t* __restrict__ packed_block_base,   // base of K (or V) block
    const float*   __restrict__ norms_block_base,    // base of norms for this block
    int                         valid_slots,         // <= block_size
    int                         block_size,
    int                         packed_slot_stride,  // bytes between adjacent slots
    int                         norms_slot_stride,   // fp32 elements between adjacent slots
    const int8_t*  __restrict__ rotation_signs,      // [d_head]
    const float*   __restrict__ centroids,           // [1 << BITS]
    int                         d_head,
    float*                      block_data           // [block_size, d_head] in shared
) {
    constexpr int VPB = 8 / BITS;
    int tid = threadIdx.x;

    // 1. Decode bytes for all slots in the block in parallel.
    int total_bytes = block_size * (d_head / VPB);
    for (int idx = tid; idx < total_bytes; idx += blockDim.x) {
        int slot = idx / (d_head / VPB);
        int byte = idx - slot * (d_head / VPB);
        if (slot >= valid_slots) {
            // Pad slots past `valid_slots` with zeros so they don't
            // contribute to attention if anything reads them. The
            // attention kernel should never reach those indices, but
            // defensive zero-fill is cheap.
            int coord_base = byte * VPB;
            #pragma unroll
            for (int k = 0; k < VPB; ++k) {
                block_data[(size_t)slot * d_head + coord_base + k] = 0.0f;
            }
            continue;
        }
        const uint8_t* slot_packed = packed_block_base
            + (size_t)slot * (size_t)packed_slot_stride;
        uint8_t p = slot_packed[byte];
        int coord_base = byte * VPB;
        #pragma unroll
        for (int k = 0; k < VPB; ++k) {
            int cidx = decode_index<BITS>(p, k);
            block_data[(size_t)slot * d_head + coord_base + k] = centroids[cidx];
        }
    }
    __syncthreads();

    // 2. Inverse Hadamard butterfly across all slots in parallel.
    //    Each slot has its own d_head-dim butterfly; one set of
    //    log2(d_head) syncs covers the whole block.
    int total_pairs = block_size * (d_head / 2);
    for (int step = 1; step < d_head; step <<= 1) {
        for (int idx = tid; idx < total_pairs; idx += blockDim.x) {
            int slot = idx / (d_head / 2);
            int p    = idx - slot * (d_head / 2);
            int group = p / step;
            int off   = p - group * step;
            int i     = group * step * 2 + off;
            int j     = i + step;
            float* row = block_data + (size_t)slot * d_head;
            float a = row[i];
            float b = row[j];
            row[i] = a + b;
            row[j] = a - b;
        }
        __syncthreads();
    }

    // 3. Final scale: per (slot, dim), multiply by signs[dim] / d_head
    //    * norms[slot]. norm == 0 zeroes the slot via multiplication.
    int total_elems = block_size * d_head;
    float inv_d = 1.0f / (float)d_head;
    for (int idx = tid; idx < total_elems; idx += blockDim.x) {
        int slot = idx / d_head;
        int dim  = idx - slot * d_head;
        if (slot >= valid_slots) {
            block_data[idx] = 0.0f;
            continue;
        }
        float n = norms_block_base[(size_t)slot * norms_slot_stride];
        block_data[idx] = block_data[idx] * inv_d * (float)rotation_signs[dim] * n;
    }
    __syncthreads();
}

template <int BITS>
__device__ __forceinline__ void dequantize_one_head_to_shared(
    const uint8_t* __restrict__ packed_head,         // [packed_bytes_per_head] uint8
    float                       head_norm,
    const int8_t*  __restrict__ rotation_signs,      // [d_head]
    const float*   __restrict__ centroids,           // [1 << BITS]
    int                         d_head,
    float*                      head_data            // [d_head] in shared mem (work buffer + result)
) {
    constexpr int VPB = 8 / BITS;
    int tid = threadIdx.x;

    if (head_norm == 0.0f) {
        for (int i = tid; i < d_head; i += blockDim.x) {
            head_data[i] = 0.0f;
        }
        __syncthreads();
        return;
    }

    // 1. Decode codebook bins into rotated-space float values.
    for (int byte = tid; byte < d_head / VPB; byte += blockDim.x) {
        uint8_t p = packed_head[byte];
        int coord_base = byte * VPB;
        #pragma unroll
        for (int k = 0; k < VPB; ++k) {
            int idx = decode_index<BITS>(p, k);
            head_data[coord_base + k] = centroids[idx];
        }
    }
    __syncthreads();

    // 2. Inverse Hadamard butterfly (self-inverse up to scale 1/d).
    for (int step = 1; step < d_head; step <<= 1) {
        for (int p = tid; p < d_head / 2; p += blockDim.x) {
            int group = p / step;
            int off   = p % step;
            int i     = group * step * 2 + off;
            int j     = i + step;
            float a = head_data[i];
            float b = head_data[j];
            head_data[i] = a + b;
            head_data[j] = a - b;
        }
        __syncthreads();
    }

    // 3. Final scale: signs, norm, 1/d.
    float inv_d = 1.0f / (float)d_head;
    for (int i = tid; i < d_head; i += blockDim.x) {
        head_data[i] = head_data[i] * inv_d * (float)rotation_signs[i] * head_norm;
    }
    __syncthreads();
}

template <int BITS, typename T>
__global__ void paged_attention_decode_kernel(
    const T*       __restrict__ q,                    // [n_seqs, n_q_heads, d_head]
    const uint8_t* __restrict__ packed_k_cache,       // vllm_blocked layout
    const float*   __restrict__ norms_k_cache,
    const uint8_t* __restrict__ packed_v_cache,
    const float*   __restrict__ norms_v_cache,
    const int8_t*  __restrict__ rotation_signs,       // [d_head]
    const float*   __restrict__ centroids,            // [1 << BITS]
    const int32_t* __restrict__ block_table,          // [n_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,             // [n_seqs]
    T*             __restrict__ out,                  // [n_seqs, n_q_heads, d_head]
    int n_q_heads,
    int n_kv_heads,
    int d_head,
    int block_size,
    int max_blocks_per_seq,
    int packed_bytes_per_head,
    // packed_*_cache strides in elements (bytes for packed, fp32 for norms)
    int packed_block_stride,
    int packed_kv_head_stride,
    int packed_slot_stride,
    int norms_block_stride,
    int norms_kv_head_stride,
    int norms_slot_stride
) {
    int seq_idx = blockIdx.x;
    int q_head  = blockIdx.y;
    int tid     = threadIdx.x;

    // GQA: map this Q head to its KV head group.
    int gqa_group = n_q_heads / n_kv_heads;
    int kv_head   = q_head / gqa_group;

    int seq_len = seq_lens[seq_idx];
    int n_blocks = (seq_len + block_size - 1) / block_size;

    // M10.8 shared layout:
    //   q_vec[d_head]                       — fp32 query
    //   block_dequant[block_size, d_head]   — fp32 dequantized K (or V) block
    //   scores[seq_len]                     — fp32 attention scores
    //   reduce_buf[blockDim.x]              — fp32 partial sums
    extern __shared__ float smem[];
    float* q_vec        = smem;
    float* block_dequant= q_vec + d_head;
    float* scores       = block_dequant + (size_t)block_size * (size_t)d_head;
    float* reduce_buf   = scores + seq_len;

    // Load Q for this (seq, q_head) into shared memory as fp32.
    {
        const T* q_ptr = q + ((size_t)seq_idx * n_q_heads + q_head) * d_head;
        for (int i = tid; i < d_head; i += blockDim.x) {
            q_vec[i] = widen_to_float<T>(q_ptr[i]);
        }
        __syncthreads();
    }

    const float scale = 1.0f / sqrtf((float)d_head);

    // -------- Pass 1: compute scores[token_idx] = (Q · K_token) / √d --------
    // After dequantize_block_to_shared loads block_dequant[slot, dim],
    // we reduce all `slots_in_block` rows to scalars in parallel:
    //
    //   1. Element-wise multiply: block_dequant[s, i] *= q_vec[i] for
    //      every (s, i). In-place — we won't need the K values after.
    //   2. Parallel tree-reduce across the d_head dimension for all
    //      slots simultaneously: log2(d_head) sync rounds total
    //      instead of `slots_in_block` × log2(d_head).
    //   3. Write block_dequant[s, 0] * scale to scores[token_idx + s].
    int token_idx = 0;
    for (int b = 0; b < n_blocks; ++b) {
        int block_id = block_table[(size_t)seq_idx * max_blocks_per_seq + b];
        int slots_in_block = min(block_size, seq_len - b * block_size);

        const uint8_t* k_block_base = packed_k_cache
            + (size_t)block_id * (size_t)packed_block_stride
            + (size_t)kv_head  * (size_t)packed_kv_head_stride;
        const float* k_norms_base = norms_k_cache
            + (size_t)block_id * (size_t)norms_block_stride
            + (size_t)kv_head  * (size_t)norms_kv_head_stride;

        dequantize_block_to_shared<BITS>(
            k_block_base, k_norms_base,
            slots_in_block, block_size,
            packed_slot_stride, norms_slot_stride,
            rotation_signs, centroids, d_head, block_dequant
        );

        // Step 1 — multiply by Q in place across all (slot, dim).
        int total_elems = block_size * d_head;
        for (int idx = tid; idx < total_elems; idx += blockDim.x) {
            int slot = idx / d_head;
            int dim  = idx - slot * d_head;
            if (slot < slots_in_block) {
                block_dequant[idx] *= q_vec[dim];
            } else {
                block_dequant[idx] = 0.0f;  // pad slots out of attention range
            }
        }
        __syncthreads();

        // Step 2 — parallel tree reduction across `d_head` for all slots.
        for (int stride = d_head / 2; stride > 0; stride >>= 1) {
            int total_pairs = block_size * stride;
            for (int idx = tid; idx < total_pairs; idx += blockDim.x) {
                int slot = idx / stride;
                int pos  = idx - slot * stride;
                block_dequant[(size_t)slot * d_head + pos] +=
                    block_dequant[(size_t)slot * d_head + pos + stride];
            }
            __syncthreads();
        }

        // Step 3 — write scores. block_dequant[s * d_head + 0] now holds
        // the dot product for slot s.
        if (tid < slots_in_block) {
            scores[token_idx + tid] =
                block_dequant[(size_t)tid * d_head] * scale;
        }
        __syncthreads();
        token_idx += slots_in_block;
    }

    // -------- Single-pass softmax over scores[0..seq_len) --------
    // Step 1: find max.
    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + r]);
        __syncthreads();
    }
    float row_max = reduce_buf[0];

    // Step 2: exp(score - max) and sum.
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float e = expf(scores[i] - row_max);
        scores[i] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) reduce_buf[tid] += reduce_buf[tid + r];
        __syncthreads();
    }
    float row_sum = reduce_buf[0];

    // Step 3: normalise. (Done lazily — fold the divide into pass 2's
    // accumulation: o /= row_sum at the end.)

    // -------- Pass 2: o[tid] = sum_t probs[t] * V[t, tid] --------
    // Same block-parallel pattern: dequantize the V block once, then
    // iterate its slots accumulating into o_local. Reuses
    // `block_dequant` shared buffer (no longer holds K at this point).
    float o_local = 0.0f;
    token_idx = 0;
    for (int b = 0; b < n_blocks; ++b) {
        int block_id = block_table[(size_t)seq_idx * max_blocks_per_seq + b];
        int slots_in_block = min(block_size, seq_len - b * block_size);

        const uint8_t* v_block_base = packed_v_cache
            + (size_t)block_id * (size_t)packed_block_stride
            + (size_t)kv_head  * (size_t)packed_kv_head_stride;
        const float* v_norms_base = norms_v_cache
            + (size_t)block_id * (size_t)norms_block_stride
            + (size_t)kv_head  * (size_t)norms_kv_head_stride;

        dequantize_block_to_shared<BITS>(
            v_block_base, v_norms_base,
            slots_in_block, block_size,
            packed_slot_stride, norms_slot_stride,
            rotation_signs, centroids, d_head, block_dequant
        );

        for (int s = 0; s < slots_in_block; ++s) {
            float p = scores[token_idx];
            if (tid < d_head) {
                o_local += p * block_dequant[(size_t)s * d_head + tid];
            }
            ++token_idx;
        }
        __syncthreads();   // block_dequant reused next iter
    }

    // Final write: out[seq, q_head, tid] = o_local / row_sum.
    if (tid < d_head) {
        T* out_ptr = out + ((size_t)seq_idx * n_q_heads + q_head) * d_head;
        out_ptr[tid] = narrow_from_float<T>(o_local / row_sum);
    }
}

#define TQ_LAUNCH_PAGED_DECODE(BITS_VAL, T)                                                 \
    paged_attention_decode_kernel<BITS_VAL, T><<<grid, threads, smem, stream>>>(            \
        static_cast<const T*>(d_q),                                                         \
        d_packed_k, d_norms_k, d_packed_v, d_norms_v,                                       \
        d_signs, d_centroids,                                                               \
        d_block_table, d_seq_lens,                                                          \
        static_cast<T*>(d_out),                                                             \
        n_q_heads, n_kv_heads, d_head, block_size, max_blocks_per_seq,                      \
        packed_bytes_per_head,                                                              \
        packed_block_stride, packed_kv_head_stride, packed_slot_stride,                     \
        norms_block_stride, norms_kv_head_stride, norms_slot_stride)

cudaError_t launch_paged_attention_decode(
    int            dtype,
    int            bits,
    int            layout,
    int            n_seqs,
    int            n_q_heads,
    int            n_kv_heads,
    int            d_head,
    int            n_blocks,
    int            block_size,
    int            max_blocks_per_seq,
    int            max_seq_len,
    const void*    d_q,
    const void*    d_packed_k_v,                    // uint8*
    const float*   d_norms_k,
    const void*    d_packed_v_v,                    // uint8*
    const float*   d_norms_v,
    const int8_t*  d_signs,
    const float*   d_centroids,
    const int32_t* d_block_table,
    const int32_t* d_seq_lens,
    void*          d_out,
    cudaStream_t   stream
) {
    if ((d_head & (d_head - 1)) != 0 || d_head <= 0 || d_head > 1024) {
        return cudaErrorInvalidValue;
    }
    if (dtype != TQ_DTYPE_FP16 && dtype != TQ_DTYPE_BF16) {
        return cudaErrorInvalidValue;
    }
    if (bits != 4 && bits != 8) {
        return cudaErrorInvalidValue;
    }
    if (n_seqs <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 || n_blocks <= 0
        || block_size <= 0 || max_blocks_per_seq <= 0 || max_seq_len <= 0) {
        return cudaErrorInvalidValue;
    }
    if (n_q_heads % n_kv_heads != 0) {
        return cudaErrorInvalidValue;  // GQA group ratio must divide cleanly.
    }

    const int packed_bytes_per_head = d_head * bits / 8;
    int packed_block_stride   = 0;
    int packed_kv_head_stride = 0;
    int packed_slot_stride    = 0;
    int norms_block_stride    = 0;
    int norms_kv_head_stride  = 0;
    int norms_slot_stride     = 0;

    switch (layout) {
        case TQ_LAYOUT_VLLM_BLOCKED:
            packed_block_stride   = n_kv_heads * block_size * packed_bytes_per_head;
            packed_kv_head_stride = block_size * packed_bytes_per_head;
            packed_slot_stride    = packed_bytes_per_head;
            norms_block_stride    = n_kv_heads * block_size;
            norms_kv_head_stride  = block_size;
            norms_slot_stride     = 1;
            break;
        default:
            return cudaErrorInvalidValue;
    }

    const uint8_t* d_packed_k = static_cast<const uint8_t*>(d_packed_k_v);
    const uint8_t* d_packed_v = static_cast<const uint8_t*>(d_packed_v_v);

    int threads = d_head;
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;

    dim3 grid(n_seqs, n_q_heads);
    // Shared memory (M10.8 layout):
    //   q_vec[d_head]                       fp32
    //   block_dequant[block_size, d_head]   fp32  — block-parallel dequant
    //   scores[max_seq_len]                 fp32
    //   reduce_buf[threads]                 fp32
    size_t smem = ((size_t)d_head
                 + (size_t)block_size * (size_t)d_head
                 + (size_t)max_seq_len
                 + (size_t)threads) * sizeof(float);

    if (dtype == TQ_DTYPE_FP16) {
        switch (bits) {
            case 8: TQ_LAUNCH_PAGED_DECODE(8, __half); break;
            case 4: TQ_LAUNCH_PAGED_DECODE(4, __half); break;
            default: return cudaErrorInvalidValue;
        }
    } else {
        switch (bits) {
            case 8: TQ_LAUNCH_PAGED_DECODE(8, __nv_bfloat16); break;
            case 4: TQ_LAUNCH_PAGED_DECODE(4, __nv_bfloat16); break;
            default: return cudaErrorInvalidValue;
        }
    }
    return cudaGetLastError();
}

#undef TQ_LAUNCH_PAGED_DECODE

// =====================================================================
// K4 — paged attention prefill (docs/KERNELS.md §3.4.2).
//
// Per-token causal attention against the cache built so far. Unlike
// K5, q has multiple tokens per sequence; each token i attends only
// to cache slots [0, i]. v1 ships single-sequence (n_seqs == 1) —
// multi-sequence batched prefill is additive, not breaking.
//
// Kernel layout: one CUDA block per (q_token_idx, q_head). Each
// block runs the same two-pass attention as K5, with the visible
// cache length truncated to q_token_idx + 1 (causal). Across q_len ×
// n_q_heads blocks, prefill parallelises trivially — perfect for
// H100 SM occupancy on Llama-3 prefill (2048 tokens × 32 q_heads =
// 65536 blocks).
// =====================================================================

template <int BITS, typename T>
__global__ void paged_attention_prefill_kernel(
    const T*       __restrict__ q,                    // [q_len, n_q_heads, d_head]
    const uint8_t* __restrict__ packed_k_cache,
    const float*   __restrict__ norms_k_cache,
    const uint8_t* __restrict__ packed_v_cache,
    const float*   __restrict__ norms_v_cache,
    const int8_t*  __restrict__ rotation_signs,
    const float*   __restrict__ centroids,
    const int32_t* __restrict__ block_table,          // [1, max_blocks_per_seq]
    int                         q_len,                // total q tokens (= visible cache len)
    T*             __restrict__ out,                  // [q_len, n_q_heads, d_head]
    int n_q_heads,
    int n_kv_heads,
    int d_head,
    int block_size,
    int max_blocks_per_seq,
    int packed_block_stride,
    int packed_kv_head_stride,
    int packed_slot_stride,
    int norms_block_stride,
    int norms_kv_head_stride,
    int norms_slot_stride
) {
    int q_token_idx = blockIdx.x;
    int q_head      = blockIdx.y;
    int tid         = threadIdx.x;

    int gqa_group = n_q_heads / n_kv_heads;
    int kv_head   = q_head / gqa_group;

    // Causal: this Q token sees cache slots [0, q_token_idx].
    int visible_len = q_token_idx + 1;
    int n_blocks_visible = (visible_len + block_size - 1) / block_size;

    extern __shared__ float smem[];
    float* q_vec      = smem;
    float* scratch    = q_vec + d_head;
    float* scores     = scratch + d_head;
    float* reduce_buf = scores + visible_len;
    // Note: caller sizes shared mem for the worst-case visible_len
    // (== q_len). Smaller q_token_idx blocks waste a few KB.

    // Load Q[q_token_idx, q_head].
    {
        const T* q_ptr = q + ((size_t)q_token_idx * n_q_heads + q_head) * d_head;
        for (int i = tid; i < d_head; i += blockDim.x) {
            q_vec[i] = widen_to_float<T>(q_ptr[i]);
        }
        __syncthreads();
    }

    const float scale = 1.0f / sqrtf((float)d_head);

    // Pass 1: scores
    int token_idx = 0;
    for (int b = 0; b < n_blocks_visible; ++b) {
        int block_id = block_table[b];   // single-sequence: row 0
        int slots_in_block = min(block_size, visible_len - b * block_size);
        for (int s = 0; s < slots_in_block; ++s) {
            const uint8_t* k_packed_head = packed_k_cache
                + (size_t)block_id * (size_t)packed_block_stride
                + (size_t)kv_head  * (size_t)packed_kv_head_stride
                + (size_t)s        * (size_t)packed_slot_stride;
            float k_norm = norms_k_cache[
                (size_t)block_id * (size_t)norms_block_stride
              + (size_t)kv_head  * (size_t)norms_kv_head_stride
              + (size_t)s        * (size_t)norms_slot_stride
            ];

            dequantize_one_head_to_shared<BITS>(
                k_packed_head, k_norm, rotation_signs, centroids,
                d_head, scratch
            );

            float partial = 0.0f;
            for (int i = tid; i < d_head; i += blockDim.x) {
                partial += q_vec[i] * scratch[i];
            }
            reduce_buf[tid] = partial;
            __syncthreads();
            for (int r = blockDim.x / 2; r > 0; r >>= 1) {
                if (tid < r) reduce_buf[tid] += reduce_buf[tid + r];
                __syncthreads();
            }
            if (tid == 0) scores[token_idx] = reduce_buf[0] * scale;
            __syncthreads();
            ++token_idx;
        }
    }

    // Softmax over scores[0..visible_len)
    float local_max = -INFINITY;
    for (int i = tid; i < visible_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + r]);
        __syncthreads();
    }
    float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int i = tid; i < visible_len; i += blockDim.x) {
        float e = expf(scores[i] - row_max);
        scores[i] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) reduce_buf[tid] += reduce_buf[tid + r];
        __syncthreads();
    }
    float row_sum = reduce_buf[0];

    // Pass 2: o = sum_t probs[t] * V[t]
    float o_local = 0.0f;
    token_idx = 0;
    for (int b = 0; b < n_blocks_visible; ++b) {
        int block_id = block_table[b];
        int slots_in_block = min(block_size, visible_len - b * block_size);
        for (int s = 0; s < slots_in_block; ++s) {
            const uint8_t* v_packed_head = packed_v_cache
                + (size_t)block_id * (size_t)packed_block_stride
                + (size_t)kv_head  * (size_t)packed_kv_head_stride
                + (size_t)s        * (size_t)packed_slot_stride;
            float v_norm = norms_v_cache[
                (size_t)block_id * (size_t)norms_block_stride
              + (size_t)kv_head  * (size_t)norms_kv_head_stride
              + (size_t)s        * (size_t)norms_slot_stride
            ];

            dequantize_one_head_to_shared<BITS>(
                v_packed_head, v_norm, rotation_signs, centroids,
                d_head, scratch
            );

            float p = scores[token_idx];
            if (tid < d_head) {
                o_local += p * scratch[tid];
            }
            __syncthreads();
            ++token_idx;
        }
    }

    if (tid < d_head) {
        T* out_ptr = out + ((size_t)q_token_idx * n_q_heads + q_head) * d_head;
        out_ptr[tid] = narrow_from_float<T>(o_local / row_sum);
    }
}

#define TQ_LAUNCH_PAGED_PREFILL(BITS_VAL, T)                                                  \
    paged_attention_prefill_kernel<BITS_VAL, T><<<grid, threads, smem, stream>>>(             \
        static_cast<const T*>(d_q),                                                           \
        d_packed_k, d_norms_k, d_packed_v, d_norms_v,                                         \
        d_signs, d_centroids,                                                                 \
        d_block_table, q_len,                                                                 \
        static_cast<T*>(d_out),                                                               \
        n_q_heads, n_kv_heads, d_head, block_size, max_blocks_per_seq,                        \
        packed_block_stride, packed_kv_head_stride, packed_slot_stride,                       \
        norms_block_stride, norms_kv_head_stride, norms_slot_stride)

cudaError_t launch_paged_attention_prefill(
    int            dtype,
    int            bits,
    int            layout,
    int            q_len,
    int            n_q_heads,
    int            n_kv_heads,
    int            d_head,
    int            n_blocks,
    int            block_size,
    int            max_blocks_per_seq,
    const void*    d_q,
    const void*    d_packed_k_v,
    const float*   d_norms_k,
    const void*    d_packed_v_v,
    const float*   d_norms_v,
    const int8_t*  d_signs,
    const float*   d_centroids,
    const int32_t* d_block_table,
    void*          d_out,
    cudaStream_t   stream
) {
    if ((d_head & (d_head - 1)) != 0 || d_head <= 0 || d_head > 1024) {
        return cudaErrorInvalidValue;
    }
    if (dtype != TQ_DTYPE_FP16 && dtype != TQ_DTYPE_BF16) {
        return cudaErrorInvalidValue;
    }
    if (bits != 4 && bits != 8) {
        return cudaErrorInvalidValue;
    }
    if (q_len <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 || n_blocks <= 0
        || block_size <= 0 || max_blocks_per_seq <= 0) {
        return cudaErrorInvalidValue;
    }
    if (n_q_heads % n_kv_heads != 0) {
        return cudaErrorInvalidValue;
    }

    const int packed_bytes_per_head = d_head * bits / 8;
    int packed_block_stride   = 0;
    int packed_kv_head_stride = 0;
    int packed_slot_stride    = 0;
    int norms_block_stride    = 0;
    int norms_kv_head_stride  = 0;
    int norms_slot_stride     = 0;

    switch (layout) {
        case TQ_LAYOUT_VLLM_BLOCKED:
            packed_block_stride   = n_kv_heads * block_size * packed_bytes_per_head;
            packed_kv_head_stride = block_size * packed_bytes_per_head;
            packed_slot_stride    = packed_bytes_per_head;
            norms_block_stride    = n_kv_heads * block_size;
            norms_kv_head_stride  = block_size;
            norms_slot_stride     = 1;
            break;
        default:
            return cudaErrorInvalidValue;
    }

    const uint8_t* d_packed_k = static_cast<const uint8_t*>(d_packed_k_v);
    const uint8_t* d_packed_v = static_cast<const uint8_t*>(d_packed_v_v);

    int threads = d_head;
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;

    dim3 grid(q_len, n_q_heads);
    // Worst-case shared mem: q_len visible (last token sees the whole seq).
    size_t smem = ((size_t)d_head + (size_t)d_head + (size_t)q_len + (size_t)threads)
                  * sizeof(float);

    if (dtype == TQ_DTYPE_FP16) {
        switch (bits) {
            case 8: TQ_LAUNCH_PAGED_PREFILL(8, __half); break;
            case 4: TQ_LAUNCH_PAGED_PREFILL(4, __half); break;
            default: return cudaErrorInvalidValue;
        }
    } else {
        switch (bits) {
            case 8: TQ_LAUNCH_PAGED_PREFILL(8, __nv_bfloat16); break;
            case 4: TQ_LAUNCH_PAGED_PREFILL(4, __nv_bfloat16); break;
            default: return cudaErrorInvalidValue;
        }
    }
    return cudaGetLastError();
}

#undef TQ_LAUNCH_PAGED_PREFILL

}  // namespace

// GPU-resident index handle. Owns the packed indices, norms, and
// codebook on the device, plus the workspace cub::DeviceRadixSort
// needs for on-GPU top-K. The top-K buffers are sized so even a
// memory-tight cluster never re-allocs across queries.
struct TqIndex {
    uint8_t* d_packed;
    float*   d_norms;
    float*   d_codebook;
    float*   d_query;
    float*   d_scores;        // [n_vectors] — written by score kernel per query
    int*     d_indices_in;    // [n_vectors] — preset to iota, reused across queries
    int*     d_indices_out;   // [n_vectors] — sorted indices output
    float*   d_scores_sorted; // [n_vectors] — sorted scores output
    void*    d_topk_temp;     // CUB temp workspace
    size_t   topk_temp_bytes;
    int      n_vectors;
    int      dim;
    int      bits;
};

extern "C" {

// Allocate GPU memory for an index, populate with host data, return an
// opaque handle in *handle_out. Returns cudaError_t (== int). Caller
// must release with tq_index_destroy(). On failure, *handle_out is NULL
// and any partially-allocated GPU memory is freed before return.
//
// `bits` must be one of {1, 2, 4, 8}. `h_codebook` must have 2^bits
// floats. `h_packed` must be `n_vectors * dim * bits / 8` bytes
// (i.e. dim / (8/bits) bytes per vector).
int tq_index_create(
    const uint8_t* h_packed,
    const float*   h_norms,
    const float*   h_codebook,
    int            bits,
    int            n_vectors,
    int            dim,
    void**         handle_out
) {
    if (handle_out == nullptr) return cudaErrorInvalidValue;
    *handle_out = nullptr;
    if (n_vectors <= 0 || !valid_bits_dim(bits, dim)) {
        return cudaErrorInvalidValue;
    }

    TqIndex* idx = new (std::nothrow) TqIndex{};
    if (idx == nullptr) return cudaErrorMemoryAllocation;
    idx->n_vectors = n_vectors;
    idx->dim       = dim;
    idx->bits      = bits;

    const int    vpb          = 8 / bits;
    const int    dim_bytes    = dim / vpb;
    const size_t packed_bytes = (size_t)n_vectors * (size_t)dim_bytes;
    const size_t norms_bytes  = (size_t)n_vectors * sizeof(float);
    const size_t scores_bytes = (size_t)n_vectors * sizeof(float);
    const size_t book_bytes   = (size_t)(1 << bits) * sizeof(float);
    const size_t query_bytes  = (size_t)dim * sizeof(float);

    const size_t indices_bytes = (size_t)n_vectors * sizeof(int);

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&idx->d_packed,        packed_bytes);  if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&idx->d_norms,         norms_bytes);   if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&idx->d_codebook,      book_bytes);    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&idx->d_query,         query_bytes);   if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&idx->d_scores,        scores_bytes);  if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&idx->d_indices_in,    indices_bytes); if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&idx->d_indices_out,   indices_bytes); if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&idx->d_scores_sorted, scores_bytes);  if (err != cudaSuccess) goto fail;

    err = cudaMemcpy(idx->d_packed,   h_packed,   packed_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(idx->d_norms,    h_norms,    norms_bytes,  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(idx->d_codebook, h_codebook, book_bytes,   cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int threads = 256;
        const int blocks  = (n_vectors + threads - 1) / threads;
        iota_kernel<<<blocks, threads>>>(idx->d_indices_in, n_vectors);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
    }

    {
        size_t needed = 0;
        err = cub::DeviceRadixSort::SortPairsDescending(
            nullptr, needed,
            idx->d_scores, idx->d_scores_sorted,
            idx->d_indices_in, idx->d_indices_out,
            n_vectors);
        if (err != cudaSuccess) goto fail;
        idx->topk_temp_bytes = needed;
        err = cudaMalloc(&idx->d_topk_temp, needed);
        if (err != cudaSuccess) goto fail;
    }

    *handle_out = idx;
    return cudaSuccess;

fail:
    if (idx->d_packed)        cudaFree(idx->d_packed);
    if (idx->d_norms)         cudaFree(idx->d_norms);
    if (idx->d_codebook)      cudaFree(idx->d_codebook);
    if (idx->d_query)         cudaFree(idx->d_query);
    if (idx->d_scores)        cudaFree(idx->d_scores);
    if (idx->d_indices_in)    cudaFree(idx->d_indices_in);
    if (idx->d_indices_out)   cudaFree(idx->d_indices_out);
    if (idx->d_scores_sorted) cudaFree(idx->d_scores_sorted);
    if (idx->d_topk_temp)     cudaFree(idx->d_topk_temp);
    delete idx;
    return (int)err;
}

void tq_index_destroy(void* handle) {
    TqIndex* idx = static_cast<TqIndex*>(handle);
    if (idx == nullptr) return;
    if (idx->d_packed)        cudaFree(idx->d_packed);
    if (idx->d_norms)         cudaFree(idx->d_norms);
    if (idx->d_codebook)      cudaFree(idx->d_codebook);
    if (idx->d_query)         cudaFree(idx->d_query);
    if (idx->d_scores)        cudaFree(idx->d_scores);
    if (idx->d_indices_in)    cudaFree(idx->d_indices_in);
    if (idx->d_indices_out)   cudaFree(idx->d_indices_out);
    if (idx->d_scores_sorted) cudaFree(idx->d_scores_sorted);
    if (idx->d_topk_temp)     cudaFree(idx->d_topk_temp);
    delete idx;
}

// Score the current query against all vectors in `handle`, write
// `n_vectors` scores into `h_scores`. `h_rotated_query` must be the
// caller's query already passed through the same RandomHadamard
// rotation that produced the index's codebook indices.
int tq_index_score(
    void*        handle,
    const float* h_rotated_query,
    float*       h_scores
) {
    TqIndex* idx = static_cast<TqIndex*>(handle);
    if (idx == nullptr) return cudaErrorInvalidValue;

    const int    vpb          = 8 / idx->bits;
    const int    dim_bytes    = idx->dim / vpb;
    const size_t query_bytes  = (size_t)idx->dim       * sizeof(float);
    const size_t scores_bytes = (size_t)idx->n_vectors * sizeof(float);

    cudaError_t err = cudaMemcpy(idx->d_query, h_rotated_query, query_bytes,
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;

    const int    threads      = 128;
    const size_t shared_bytes = threads * sizeof(float);
    const float  inv_sqrt_dim = 1.0f / sqrtf((float)idx->dim);

    err = launch_score(idx->bits, idx->n_vectors, dim_bytes, threads,
                       shared_bytes, idx->d_packed, idx->d_norms,
                       idx->d_codebook, idx->d_query, idx->d_scores,
                       inv_sqrt_dim);
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;

    return (int)cudaMemcpy(h_scores, idx->d_scores, scores_bytes,
                           cudaMemcpyDeviceToHost);
}

// Score the query and return only the top-K (index, score) pairs,
// sorted by descending score. Top-K selection happens entirely on GPU
// via cub::DeviceRadixSort::SortPairsDescending. This is the
// production-latency path (M2.2) — the score-only `tq_index_score`
// remains available for tests and debugging.
//
// h_topk_indices and h_topk_scores must each have room for `k` entries.
// `k` is clamped at `n_vectors` (the index size). Returns
// cudaError_t (== int).
int tq_index_search_top_k(
    void*        handle,
    const float* h_rotated_query,  // [dim]
    int          k,
    int*         h_topk_indices,   // [k] OUT
    float*       h_topk_scores     // [k] OUT
) {
    TqIndex* idx = static_cast<TqIndex*>(handle);
    if (idx == nullptr || k <= 0) return cudaErrorInvalidValue;
    int take = k < idx->n_vectors ? k : idx->n_vectors;

    const int    vpb         = 8 / idx->bits;
    const int    dim_bytes   = idx->dim / vpb;
    const size_t query_bytes = (size_t)idx->dim * sizeof(float);

    cudaError_t err = cudaMemcpy(idx->d_query, h_rotated_query, query_bytes,
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;

    {
        const int    threads      = 128;
        const size_t shared_bytes = threads * sizeof(float);
        const float  inv_sqrt_dim = 1.0f / sqrtf((float)idx->dim);
        err = launch_score(idx->bits, idx->n_vectors, dim_bytes, threads,
                           shared_bytes, idx->d_packed, idx->d_norms,
                           idx->d_codebook, idx->d_query, idx->d_scores,
                           inv_sqrt_dim);
        if (err != cudaSuccess) return (int)err;
    }

    err = cub::DeviceRadixSort::SortPairsDescending(
        idx->d_topk_temp, idx->topk_temp_bytes,
        idx->d_scores, idx->d_scores_sorted,
        idx->d_indices_in, idx->d_indices_out,
        idx->n_vectors);
    if (err != cudaSuccess) return (int)err;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;

    err = cudaMemcpy(h_topk_indices, idx->d_indices_out,
                     (size_t)take * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(h_topk_scores, idx->d_scores_sorted,
                     (size_t)take * sizeof(float), cudaMemcpyDeviceToHost);
    return (int)err;
}

// One-shot host-pointer FFI for the M2.1 kernel-equivalence test:
// allocate on GPU, copy in, run scoring at `bits`, copy out, free.
// Returns cudaError_t (== int). 0 = success.
int tq_score_oneshot(
    const uint8_t* h_packed,
    const float*   h_norms,
    const float*   h_codebook,
    const float*   h_rotated_query,
    float*         h_scores,
    int            bits,
    int            n_vectors,
    int            dim
) {
    if (n_vectors <= 0 || !valid_bits_dim(bits, dim)) {
        return cudaErrorInvalidValue;
    }
    const int    vpb           = 8 / bits;
    const int    dim_bytes     = dim / vpb;
    const size_t packed_bytes  = (size_t)n_vectors * (size_t)dim_bytes;
    const size_t norms_bytes   = (size_t)n_vectors * sizeof(float);
    const size_t scores_bytes  = (size_t)n_vectors * sizeof(float);
    const size_t book_bytes    = (size_t)(1 << bits) * sizeof(float);
    const size_t query_bytes   = (size_t)dim * sizeof(float);

    uint8_t* d_packed   = nullptr;
    float*   d_norms    = nullptr;
    float*   d_codebook = nullptr;
    float*   d_query    = nullptr;
    float*   d_scores   = nullptr;
    cudaError_t err     = cudaSuccess;

    err = cudaMalloc(&d_packed,   packed_bytes);  if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_norms,    norms_bytes);   if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_codebook, book_bytes);    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_query,    query_bytes);   if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_scores,   scores_bytes);  if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_packed,   h_packed,        packed_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_norms,    h_norms,         norms_bytes,  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_codebook, h_codebook,      book_bytes,   cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_query,    h_rotated_query, query_bytes,  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    {
        const int    threads      = 128;
        const size_t shared_bytes = threads * sizeof(float);
        const float  inv_sqrt_dim = 1.0f / sqrtf((float)dim);
        err = launch_score(bits, n_vectors, dim_bytes, threads, shared_bytes,
                           d_packed, d_norms, d_codebook, d_query, d_scores,
                           inv_sqrt_dim);
        if (err != cudaSuccess) goto cleanup;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto cleanup;
    }

    err = cudaMemcpy(h_scores, d_scores, scores_bytes, cudaMemcpyDeviceToHost);

cleanup:
    if (d_packed)   cudaFree(d_packed);
    if (d_norms)    cudaFree(d_norms);
    if (d_codebook) cudaFree(d_codebook);
    if (d_query)    cudaFree(d_query);
    if (d_scores)   cudaFree(d_scores);
    return (int)err;
}

// --------------------------------------------------------------------
// Streaming K/V quantize / dequantize kernels — K1 / K2 in the
// docs/KERNELS.md kernels-v1 contract. All buffers are device
// pointers; called from Python through PyO3 with
// `torch.Tensor.data_ptr()` values and the user's CUDA stream.
//
// Layout assumptions:
//   d_kv           : fp16/bf16, [n_tokens, n_heads, d_head]
//   d_packed       : uint8,     [n_tokens, n_heads, d_head * bits / 8]
//   d_norms        : fp32,      [n_tokens, n_heads]
//   d_signs        : int8,      [d_head]            (Hadamard signs ±1)
//   d_centroids    : fp32,      [1 << bits]         (codebook centroids)
//   d_thresholds   : fp32,      [(1 << bits) - 1]   (encoder thresholds)
//
// `d_head` must be a power of two, ≤ 1024. `bits` ∈ {4, 8}.
// `dtype`  ∈ {0=fp16, 1=bf16} per `tq_dtype` enum.
// `stream` is `cudaStream_t` cast to `void*`. The kernel always
// launches on this exact handle and never reads a thread-local
// current-stream of its own. Passing 0 (CUDA default/legacy stream)
// is legitimate — PyTorch sometimes returns 0 for
// `current_stream().cuda_stream` — and is still treated as "caller
// chose explicitly" per the kernels-v1 stream contract.
// --------------------------------------------------------------------

int tq_kv_quantize_streaming(
    const void*   d_kv,
    const int8_t* d_signs,
    const float*  d_centroids,
    const float*  d_thresholds,
    void*         d_packed,
    void*         d_norms,
    int           bits,
    int           dtype,
    int           n_tokens,
    int           n_heads,
    int           d_head,
    void*         stream
) {
    if (n_tokens <= 0 || n_heads <= 0 || d_head <= 0) {
        return cudaErrorInvalidValue;
    }
    return (int)launch_kv_streaming(
        /*quantize=*/true, bits, dtype, n_tokens, n_heads, d_head,
        d_kv, d_packed, d_norms, d_signs, d_centroids, d_thresholds,
        /*d_out_kv_or_packed=*/nullptr,
        static_cast<cudaStream_t>(stream));
}

int tq_kv_dequantize_streaming(
    const void*   d_packed,
    const float*  d_norms,
    const int8_t* d_signs,
    const float*  d_centroids,
    void*         d_kv_out,
    int           bits,
    int           dtype,
    int           n_tokens,
    int           n_heads,
    int           d_head,
    void*         stream
) {
    if (n_tokens <= 0 || n_heads <= 0 || d_head <= 0) {
        return cudaErrorInvalidValue;
    }
    return (int)launch_kv_streaming(
        /*quantize=*/false, bits, dtype, n_tokens, n_heads, d_head,
        /*d_kv=*/nullptr, d_packed, d_norms, d_signs, d_centroids,
        /*d_thresholds=*/nullptr, d_kv_out,
        static_cast<cudaStream_t>(stream));
}

// --------------------------------------------------------------------
// K3 paged write — kernels-v1 contract docs/KERNELS.md §3.3.
//
//   d_kv             : fp16/bf16, [n_tokens, n_kv_heads, d_head]
//   d_slot_mapping   : int32,     [n_tokens]; entry < 0 → skip token
//   d_packed_cache   : uint8,     paged buffer (layout-dependent)
//   d_norms_cache    : fp32,      paged buffer (layout-dependent)
//
// `bits`   ∈ {4, 8}; `dtype` ∈ {0=fp16, 1=bf16}.
// `layout` ∈ tq_paged_layout — currently only TQ_LAYOUT_VLLM_BLOCKED
//          (see launch_paged_quantize for the strides it implies).
// `stream` is the caller's `cudaStream_t` cast to `void*`; same
// stream contract as the streaming kernels.
// --------------------------------------------------------------------

int tq_kv_quantize_paged(
    const void*    d_kv,
    const int32_t* d_slot_mapping,
    void*          d_packed_cache,
    float*         d_norms_cache,
    const int8_t*  d_signs,
    const float*   d_centroids,
    const float*   d_thresholds,
    int            bits,
    int            dtype,
    int            layout,
    int            n_tokens,
    int            n_kv_heads,
    int            d_head,
    int            n_blocks,
    int            block_size,
    void*          stream
) {
    return (int)launch_paged_quantize(
        dtype, bits, layout,
        n_tokens, n_kv_heads, d_head,
        n_blocks, block_size,
        d_kv, d_slot_mapping,
        d_packed_cache, d_norms_cache,
        d_signs, d_centroids, d_thresholds,
        static_cast<cudaStream_t>(stream));
}

// --------------------------------------------------------------------
// K5 paged attention decode — kernels-v1 docs/KERNELS.md §3.4.3.
//
// Single-token-per-sequence Q against the full cache history. One
// CUDA block per (seq, q_head); shared memory holds the query, a
// per-token dequant scratch, and the row of attention scores. Two-
// pass over the cache (compute scores, then softmax-weighted V sum).
//
//   d_q              : fp16/bf16, [n_seqs, n_q_heads, d_head]
//   d_packed_k_cache : uint8,     vllm_blocked layout
//   d_norms_k_cache  : fp32,      vllm_blocked layout
//   d_packed_v_cache : uint8,     same shape
//   d_norms_v_cache  : fp32,      same shape
//   d_block_table    : int32,     [n_seqs, max_blocks_per_seq]
//   d_seq_lens       : int32,     [n_seqs]
//   d_out            : fp16/bf16, [n_seqs, n_q_heads, d_head]
//
// `bits` ∈ {4, 8}; `dtype` ∈ {0=fp16, 1=bf16}; `layout` ∈
// `tq_paged_layout`. GQA via `n_q_heads >= n_kv_heads` with
// `n_q_heads % n_kv_heads == 0`.
// --------------------------------------------------------------------

int tq_paged_attention_decode(
    const void*    d_q,
    const void*    d_packed_k_cache,
    const float*   d_norms_k_cache,
    const void*    d_packed_v_cache,
    const float*   d_norms_v_cache,
    const int8_t*  d_signs,
    const float*   d_centroids,
    const int32_t* d_block_table,
    const int32_t* d_seq_lens,
    void*          d_out,
    int            bits,
    int            dtype,
    int            layout,
    int            n_seqs,
    int            n_q_heads,
    int            n_kv_heads,
    int            d_head,
    int            n_blocks,
    int            block_size,
    int            max_blocks_per_seq,
    int            max_seq_len,
    void*          stream
) {
    return (int)launch_paged_attention_decode(
        dtype, bits, layout,
        n_seqs, n_q_heads, n_kv_heads, d_head,
        n_blocks, block_size, max_blocks_per_seq, max_seq_len,
        d_q, d_packed_k_cache, d_norms_k_cache,
        d_packed_v_cache, d_norms_v_cache,
        d_signs, d_centroids,
        d_block_table, d_seq_lens, d_out,
        static_cast<cudaStream_t>(stream));
}

// --------------------------------------------------------------------
// K4 paged attention prefill — kernels-v1 docs/KERNELS.md §3.4.2.
//
// Causal-masked per-token attention. Single-sequence in v1; multi-
// sequence batched prefill is additive. Token i in q attends to
// cache slots [0, i] inclusive.
//
//   d_q              : fp16/bf16, [q_len, n_q_heads, d_head]
//   d_packed_k_cache : uint8,     vllm_blocked layout
//   d_norms_k_cache  : fp32,      vllm_blocked layout
//   d_packed_v_cache : uint8,     same
//   d_norms_v_cache  : fp32,      same
//   d_block_table    : int32,     [1, max_blocks_per_seq] (single seq)
//   d_out            : fp16/bf16, [q_len, n_q_heads, d_head]
// --------------------------------------------------------------------

int tq_paged_attention_prefill(
    const void*    d_q,
    const void*    d_packed_k_cache,
    const float*   d_norms_k_cache,
    const void*    d_packed_v_cache,
    const float*   d_norms_v_cache,
    const int8_t*  d_signs,
    const float*   d_centroids,
    const int32_t* d_block_table,
    void*          d_out,
    int            bits,
    int            dtype,
    int            layout,
    int            q_len,
    int            n_q_heads,
    int            n_kv_heads,
    int            d_head,
    int            n_blocks,
    int            block_size,
    int            max_blocks_per_seq,
    void*          stream
) {
    return (int)launch_paged_attention_prefill(
        dtype, bits, layout,
        q_len, n_q_heads, n_kv_heads, d_head,
        n_blocks, block_size, max_blocks_per_seq,
        d_q, d_packed_k_cache, d_norms_k_cache,
        d_packed_v_cache, d_norms_v_cache,
        d_signs, d_centroids, d_block_table,
        d_out,
        static_cast<cudaStream_t>(stream));
}

// --------------------------------------------------------------------
// K6 paged read — kernels-v1 docs/KERNELS.md §3.5 (added M10.9 prep).
//
// Gather + dequantize a list of physical blocks from the compressed
// cache into a contiguous fp16/bf16 scratch tensor. The vLLM/SGLang
// integration pattern: persistent cache stays compressed, scratch
// lives only for one attention forward step; engine remaps its
// block_table to point at scratch indices and delegates to
// FlashAttention.
//
//   d_packed_cache       : uint8,  vllm_blocked layout
//   d_norms_cache        : fp32,   vllm_blocked layout
//   d_physical_block_ids : int32,  [n_scratch_blocks]
//   d_scratch_out        : fp16/bf16, [n_scratch_blocks, n_kv_heads, block_size, d_head]
// --------------------------------------------------------------------

int tq_kv_dequantize_paged(
    const void*    d_packed_cache,
    const float*   d_norms_cache,
    const int32_t* d_physical_block_ids,
    const int8_t*  d_signs,
    const float*   d_centroids,
    void*          d_scratch_out,
    int            bits,
    int            dtype,
    int            layout,
    int            n_scratch_blocks,
    int            n_kv_heads,
    int            d_head,
    int            block_size,
    void*          stream
) {
    return (int)launch_paged_dequantize(
        dtype, bits, layout,
        n_scratch_blocks, n_kv_heads, d_head, block_size,
        static_cast<const uint8_t*>(d_packed_cache),
        d_norms_cache,
        d_physical_block_ids,
        d_signs, d_centroids,
        d_scratch_out,
        static_cast<cudaStream_t>(stream));
}

}  // extern "C"
