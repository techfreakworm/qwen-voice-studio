"""Inference engine — a thin, device-agnostic layer over ``qwen_tts``.

The same functions run on MPS and CUDA; the only device knowledge lives in
``qvs.device``. Adds what the raw wrapper lacks for a studio: explicit seeding,
long-form sentence chunking, and a uniform return of ``(waveform, sample_rate)``.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from . import audio
from .config import GEN_DEFAULTS, LONGFORM_CHAR_THRESHOLD, OUTPUT_SAMPLE_RATE
from .device import get_attn_impl, get_dtype, target_device


# ---- generation parameters ----------------------------------------------------
@dataclass
class GenParams:
    temperature: float = GEN_DEFAULTS.temperature
    top_p: float = GEN_DEFAULTS.top_p
    top_k: int = GEN_DEFAULTS.top_k
    repetition_penalty: float = GEN_DEFAULTS.repetition_penalty
    subtalker_temperature: float = GEN_DEFAULTS.subtalker_temperature
    subtalker_top_p: float = GEN_DEFAULTS.subtalker_top_p
    subtalker_top_k: int = GEN_DEFAULTS.subtalker_top_k
    max_new_tokens: int = GEN_DEFAULTS.max_new_tokens
    seed: int = GEN_DEFAULTS.seed

    def to_kwargs(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("seed", None)
        d["do_sample"] = self.temperature is not None and self.temperature > 0
        return d


# ---- model loading / placement ------------------------------------------------
def load_model(repo: str, device: Optional[str] = None, load_on_cpu: bool = False):
    """Load a checkpoint. ``load_on_cpu=True`` builds on CPU (ZeroGPU: move to
    cuda inside the ``@spaces.GPU`` fork afterwards)."""
    from qwen_tts import Qwen3TTSModel

    device = device or target_device()
    device_map = "cpu" if load_on_cpu else device
    return Qwen3TTSModel.from_pretrained(
        repo,
        device_map=device_map,
        dtype=get_dtype(),
        attn_implementation=get_attn_impl(device),
    )


def move_model(model, device: str):
    """Relocate a loaded model (used to move CPU-built models onto the GPU)."""
    import torch

    model.model.to(device)
    model.device = torch.device(device)
    return model


# ---- seeding ------------------------------------------------------------------
def apply_seed(seed: int) -> None:
    if seed is None or seed < 0:
        return
    import torch

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- long-form chunking -------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？…])\s+")


def split_sentences(text: str) -> list[str]:
    return [p for p in _SENT_SPLIT.split(text.strip()) if p.strip()]


def chunk_text(text: str, threshold: int = LONGFORM_CHAR_THRESHOLD) -> list[str]:
    """Group sentences into chunks no longer than ``threshold`` characters."""
    if len(text) <= threshold:
        return [text]
    chunks: list[str] = []
    cur = ""
    for s in split_sentences(text):
        if cur and len(cur) + len(s) + 1 > threshold:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip() if cur else s
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _run(model, method: str, texts: list[str], params: GenParams, **fixed) -> tuple[np.ndarray, int]:
    """Call a ``generate_*`` method once per chunk and concatenate."""
    apply_seed(params.seed)
    fn = getattr(model, method)
    wavs_out: list[np.ndarray] = []
    sr = OUTPUT_SAMPLE_RATE
    for t in texts:
        wavs, sr = fn(text=t, **fixed, **params.to_kwargs())
        wavs_out.append(np.asarray(wavs[0], dtype=np.float32))
    return audio.concat(wavs_out, sr), sr


# ---- the three modes ----------------------------------------------------------
def synth_custom_voice(model, text: str, speaker: str, instruct: Optional[str], language: str,
                       params: GenParams, longform: bool = True) -> tuple[np.ndarray, int]:
    texts = chunk_text(text) if longform else [text]
    return _run(model, "generate_custom_voice", texts, params,
                speaker=speaker, instruct=(instruct or None), language=language)


def synth_voice_design(model, text: str, instruct: str, language: str,
                       params: GenParams, longform: bool = True) -> tuple[np.ndarray, int]:
    texts = chunk_text(text) if longform else [text]
    return _run(model, "generate_voice_design", texts, params,
                instruct=instruct, language=language)


def synth_clone(model, text: str, language: str, params: GenParams,
                ref_audio=None, ref_text: Optional[str] = None,
                x_vector_only: bool = False, voice_clone_prompt=None,
                longform: bool = True) -> tuple[np.ndarray, int]:
    """Clone. Provide either (ref_audio[, ref_text]) or a prebuilt
    ``voice_clone_prompt`` (from the voice library)."""
    texts = chunk_text(text) if longform else [text]
    apply_seed(params.seed)
    wavs_out: list[np.ndarray] = []
    sr = OUTPUT_SAMPLE_RATE
    # Build the reusable prompt once so features aren't re-extracted per chunk.
    if voice_clone_prompt is None and ref_audio is not None:
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only
        )
    for t in texts:
        wavs, sr = model.generate_voice_clone(
            text=t, language=language, voice_clone_prompt=voice_clone_prompt, **params.to_kwargs()
        )
        wavs_out.append(np.asarray(wavs[0], dtype=np.float32))
    return audio.concat(wavs_out, sr), sr
