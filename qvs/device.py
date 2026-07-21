"""Device abstraction — the single place that differs between Apple-silicon MPS
(local) and CUDA (Hugging Face ZeroGPU). Everything else in the app is written
against this module so the CUDA path is never a rewrite.

Verified locally: ``from_pretrained(repo, device_map="mps", dtype=bfloat16,
attn_implementation="sdpa")`` loads all three checkpoints with no monkeypatching,
provided ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set before torch runs the STFT/mel
path used for reference audio.
"""
from __future__ import annotations

import functools
import os


def setup_runtime() -> None:
    """Set process-wide env before torch is used. Idempotent; call at import."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # STFT/mel fallback on MPS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def on_zerogpu() -> bool:
    """True when running inside a Hugging Face ZeroGPU Space."""
    return bool(os.environ.get("SPACES_ZERO_GPU") or os.environ.get("SPACES_ZERO_GPU_V2"))


def get_device() -> str:
    """Best available device string, cuda > mps > cpu.

    On ZeroGPU the GPU is only attached inside an ``@spaces.GPU`` fork, so at
    import time CUDA is *not* visible; callers that must know the eventual device
    should treat ``on_zerogpu()`` as "cuda".
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def target_device() -> str:
    """The device inference will actually run on (accounts for ZeroGPU forks)."""
    return "cuda" if on_zerogpu() else get_device()


def get_dtype():
    """bfloat16 everywhere — verified good on both MPS and CUDA (Blackwell)."""
    import torch

    return torch.bfloat16


@functools.lru_cache(maxsize=1)
def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def get_attn_impl(device: str | None = None) -> str:
    """flash_attention_2 on CUDA when the wheel is present, else sdpa.

    flash-attn is CUDA-only, so MPS always uses sdpa. sdpa is a safe default on
    CUDA too (just a little slower than flash-attn).
    """
    dev = device or target_device()
    if dev == "cuda" and _flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


# ---- optional ZeroGPU decorator ----------------------------------------------
try:  # `spaces` is only present on the Space; keep the app importable locally.
    import spaces  # type: ignore

    _HAS_SPACES = True
except Exception:  # pragma: no cover - local path
    spaces = None  # type: ignore
    _HAS_SPACES = False


def gpu(duration: int = 60):
    """Decorator: run under ``@spaces.GPU`` on ZeroGPU, passthrough elsewhere.

    Usage: ``@gpu(duration=120)`` on the request-scoped inference function.
    """

    def decorator(fn):
        if _HAS_SPACES and on_zerogpu():
            return spaces.GPU(duration=duration)(fn)
        return fn

    return decorator


setup_runtime()
