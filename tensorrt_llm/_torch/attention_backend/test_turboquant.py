"""Standalone test for the TurboQuant attention backend.

Run with:
    python -m pytest tensorrt_llm/_torch/attention_backend/test_turboquant.py

Or directly:
    python tensorrt_llm/_torch/attention_backend/test_turboquant.py

Tests are CUDA-required (the streaming kernels run on device). They
exercise the `_round_trip` method (the core of v1) on synthetic K/V
of Llama-3 shape and assert cosine-similarity bounds matching the
M6.4 transformers PoC numbers (drift p50 ≥ 0.99 at bits=8 / ≥ 0.93
at bits=4).
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def _need_cuda():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping.")
        sys.exit(0)


def _need_turboquant():
    try:
        import turboquant._native  # noqa: F401
    except ImportError as e:
        print(f"turboquant wheel not installed — skipping. ({e})")
        sys.exit(0)


def _per_head_cosine(a: torch.Tensor, b: torch.Tensor, head_dim: int) -> np.ndarray:
    flat_a = a.float().reshape(-1, head_dim)
    flat_b = b.float().reshape(-1, head_dim)
    return torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=1).cpu().numpy()


def run_one(bits: int, n_tokens: int, num_kv_heads: int, head_dim: int, seed: int = 0):
    print(
        f"\n--- bits={bits} n_tokens={n_tokens} "
        f"num_kv_heads={num_kv_heads} head_dim={head_dim} ---"
    )
    _need_cuda()
    _need_turboquant()

    from tensorrt_llm._torch.attention_backend.turboquant import (
        TurboquantAttention,
    )

    backend = TurboquantAttention(
        layer_idx=0,
        num_heads=num_kv_heads,  # not used by _round_trip
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        quant_config=None,
        tq_bits=bits,
    )

    torch.manual_seed(seed)
    # Shape matches what VanillaAttention's _single_request_update_kv_cache
    # passes to _round_trip: [1, kv_len, num_kv_heads, head_dim] fp16.
    k = (
        torch.randn(1, n_tokens, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        * 1.5
    ).contiguous()
    v = (
        torch.randn(1, n_tokens, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        * 1.5
    ).contiguous()

    k_rt = backend._round_trip(k)
    v_rt = backend._round_trip(v)

    assert k_rt.shape == k.shape
    assert v_rt.shape == v.shape
    assert k_rt.dtype == torch.float16
    assert v_rt.dtype == torch.float16
    assert torch.isfinite(k_rt).all()
    assert torch.isfinite(v_rt).all()

    cos_k = _per_head_cosine(k, k_rt, head_dim)
    cos_v = _per_head_cosine(v, v_rt, head_dim)

    p50_k = float(np.median(cos_k))
    p50_v = float(np.median(cos_v))
    min_k = float(np.min(cos_k))
    min_v = float(np.min(cos_v))

    print(
        f"  K cosine vs original  p50={p50_k:.4f}  min={min_k:.4f}\n"
        f"  V cosine vs original  p50={p50_v:.4f}  min={min_v:.4f}"
    )

    # Same envelope as M6.5 paged_kv_smoke.py and M6.4 transformers PoC.
    expected_p50 = 0.99 if bits == 8 else 0.93
    drift_p50 = min(p50_k, p50_v)
    if drift_p50 < expected_p50:
        raise SystemExit(
            f"FAIL: bits={bits} drift p50 {drift_p50:.4f} < {expected_p50}"
        )
    print(f"  PASS bits={bits}")


def main() -> None:
    print("TurboquantAttention standalone round-trip test")
    print(
        f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}"
    )
    _need_cuda()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Llama-3 shape × n_tokens × bits
    run_one(bits=8, n_tokens=64, num_kv_heads=8, head_dim=128)
    run_one(bits=4, n_tokens=64, num_kv_heads=8, head_dim=128)
    run_one(bits=8, n_tokens=2048, num_kv_heads=8, head_dim=128)
    run_one(bits=4, n_tokens=2048, num_kv_heads=8, head_dim=128)
    print("\nAll passed.")


if __name__ == "__main__":
    main()
