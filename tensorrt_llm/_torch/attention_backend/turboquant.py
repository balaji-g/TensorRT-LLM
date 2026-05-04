"""TurboQuant attention backend for TRT-LLM.

This is the v1 (correctness-first) integration: we subclass
VanillaAttention and intercept the per-request KV-cache update so each
new K/V is round-tripped through the TurboQuant streaming
quantize/dequantize kernels before it's written to TRT-LLM's standard
fp16 cache buffer. The cache *storage* stays fp16 in this v1 (no
memory savings yet); what we prove is that the *math* through the
compressed representation matches the original fp16 at acceptable
drift — same approach M6.4 took for transformers and which produced
0.00% perplexity drift at bits=8 / 0.55% at bits=4 on Llama-3-8B.

The v2 (memory-efficient) variant will land separately: a custom
KVCacheManager subclass that allocates a packed compressed buffer and
calls our paged-attention kernels (already shipped in
turboquant/cuda/turboquant_paged_attention.cu).

Selecting this backend:

    PYTORCHBACKEND_ATTN_BACKEND=TURBOQUANT
    # or pass `backend="TURBOQUANT"` through the Python API.

Configuration:
    Pass `tq_bits=4` or `tq_bits=8` through `quant_config.kv_cache_quant_algo`
    or as a kwarg to the backend constructor.

Math reference:
- Hadamard rotation + per-coordinate Lloyd-Max scalar quantization on
  K and V independently (see turboquant-rs paper / MseQuantizer).
- bits ∈ {4, 8}: 256-byte head_size=128 fp16 → packed 64-byte (4-bit)
  or 128-byte (8-bit) plus a 4-byte fp32 norm per head.
- The streaming quantize+dequantize kernels we use here are validated
  on H100 sm_90 and 5070 Ti sm_120 with cosine consistency 1.000000
  vs the CPU reference (see python/examples/streaming_kv_demo.py).
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm.models.modeling_utils import QuantConfig

from .interface import (AttentionBackend, AttentionMask, AttentionMetadata,
                        PredefinedAttentionMask)
from .vanilla import VanillaAttention, VanillaAttentionMetadata, repeat_kv

_VALID_BITS = (4, 8)


def _resolve_bits(quant_config: Optional[QuantConfig], bits_kwarg: Optional[int]) -> int:
    """Pick the bit-width for KV quantization.

    Order of precedence: explicit `tq_bits` constructor kwarg → a
    `kv_cache_quant_algo` value carrying our scheme → default 8.
    Keep the constructor surface small for v1 — most callers will
    just pass `tq_bits=8`.
    """
    if bits_kwarg is not None:
        if bits_kwarg not in _VALID_BITS:
            raise ValueError(
                f"TurboQuant backend supports bits ∈ {_VALID_BITS}; got {bits_kwarg}"
            )
        return bits_kwarg
    if quant_config is not None:
        algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if algo is not None:
            algo_str = str(algo)
            if algo_str.endswith("INT4") or algo_str.endswith("int4"):
                return 4
            if algo_str.endswith("INT8") or algo_str.endswith("int8"):
                return 8
    return 8


class TurboquantAttentionMetadata(VanillaAttentionMetadata):
    """Metadata for TurboQuant attention.

    Identical layout to VanillaAttentionMetadata in v1; subclassed so
    that backend-selection / type-dispatch stays clean and so that
    the v2 variant can extend this with paged-cache fields without
    touching VanillaAttentionMetadata.
    """
    pass


class TurboquantAttention(VanillaAttention):
    """v1 attention backend: vanilla attention with K/V round-tripped
    through TurboQuant streaming quantize+dequantize on every cache
    write.

    The cache buffer that TRT-LLM's KVCacheManager allocates is the
    same standard fp16 layout VanillaAttention uses
    (`[num_pool_seqs, 2, max_seq_len, num_kv_heads, head_size]`). We
    write the *post-compression* fp16 there. Reads, attention math,
    and the rest of the forward path are unchanged from
    VanillaAttention.

    Net effect: model sees the same numerical degradation it would see
    with full compressed-paged-cache storage, but the storage layout
    stays vanilla. Useful for validating drift before committing to
    the v2 paged-storage variant. After v1 prints 0.00% / <2% drift
    matching the M6.4 transformers number, we ship v2 for memory
    savings.
    """

    Metadata = TurboquantAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        tq_bits: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            layer_idx, num_heads, head_dim, num_kv_heads, quant_config, **kwargs
        )
        self.bits = _resolve_bits(quant_config, tq_bits)

        # Lazy import — turboquant streaming kernels live in our
        # native wheel. Importing inside __init__ keeps tests of the
        # rest of TRT-LLM working when the wheel isn't installed.
        try:
            from turboquant._native import (
                kv_streaming_dequantize_cuda,
                kv_streaming_quantize_cuda,
                quantizer_state,
            )
        except ImportError as e:
            raise ImportError(
                "TurboQuant backend requires the `turboquant` wheel built "
                "with the `cuda` feature. From the turboquant-rs repo:\n"
                "    cd python && maturin build --release --features cuda\n"
                "    pip install target/wheels/turboquant-*.whl"
            ) from e
        self._kv_quant = kv_streaming_quantize_cuda
        self._kv_dequant = kv_streaming_dequantize_cuda
        self._quantizer_state_fn = quantizer_state

        # Quantizer state is a function of (head_dim, bits, seed); cached
        # on first device-side use.
        self._signs = None
        self._centroids = None
        self._thresholds = None

    def _ensure_state(self, device: torch.device) -> None:
        if self._signs is not None:
            return
        signs_np, centroids_np, thresholds_np = self._quantizer_state_fn(
            self.head_dim, self.bits, 0
        )
        self._signs = torch.from_numpy(signs_np).to(device).contiguous()
        self._centroids = torch.from_numpy(centroids_np).to(device).contiguous()
        self._thresholds = torch.from_numpy(thresholds_np).to(device).contiguous()

    def _round_trip(self, kv: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize K (or V) via the streaming kernels.

        Input shape: `[1, kv_len, num_kv_heads, head_size]` fp16,
        contiguous. Output: same shape and dtype, with each
        (token, head) row passed through Hadamard + Lloyd-Max +
        inverse pipeline, recovering the post-compression value.

        We allocate the packed/norms buffers freshly per call. They're
        small (head_size * bits / 8 + 4 bytes per (token, head)) and
        the kernel call is the dominant cost; v2 will reuse buffers.
        """
        assert kv.dtype is torch.float16, (
            f"TurboQuant backend currently requires fp16; got {kv.dtype}"
        )
        assert kv.is_contiguous() and kv.dim() == 4, (
            f"Expected [1, kv_len, num_kv_heads, head_size] contiguous; "
            f"got shape {tuple(kv.shape)}"
        )
        n_tokens = kv.shape[1] * kv.shape[0]
        num_kv_heads = kv.shape[2]
        head_size = kv.shape[3]
        assert head_size == self.head_dim
        self._ensure_state(kv.device)

        packed_bytes = (head_size * self.bits) // 8
        packed = torch.empty(
            (n_tokens, num_kv_heads, packed_bytes),
            dtype=torch.uint8,
            device=kv.device,
        )
        norms = torch.empty(
            (n_tokens, num_kv_heads), dtype=torch.float32, device=kv.device
        )
        out = torch.empty_like(kv)

        kv_view = kv.reshape(n_tokens, num_kv_heads, head_size)

        self._kv_quant(
            int(kv_view.data_ptr()),
            int(self._signs.data_ptr()),
            int(self._centroids.data_ptr()),
            int(self._thresholds.data_ptr()),
            int(packed.data_ptr()),
            int(norms.data_ptr()),
            int(self.bits),
            int(n_tokens),
            int(num_kv_heads),
            int(head_size),
        )
        self._kv_dequant(
            int(packed.data_ptr()),
            int(norms.data_ptr()),
            int(self._signs.data_ptr()),
            int(self._centroids.data_ptr()),
            int(out.data_ptr()),
            int(self.bits),
            int(n_tokens),
            int(num_kv_heads),
            int(head_size),
        )
        return out.reshape_as(kv)

    def _single_request_update_kv_cache(
        self, k, v, kv_cache_tensor, seq_len, cache_idx, cache_position
    ):
        """Identical control flow to VanillaAttention's version but
        K and V are first round-tripped through compression."""
        if k is not None and v is not None:
            k = self._round_trip(k)
            v = self._round_trip(v)
        return super()._single_request_update_kv_cache(
            k, v, kv_cache_tensor, seq_len, cache_idx, cache_position
        )
