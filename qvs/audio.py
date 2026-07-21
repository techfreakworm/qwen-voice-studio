"""Audio I/O helpers shared by the engine and the UI."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

from .config import OUTPUT_SAMPLE_RATE


def to_gradio(wav: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """Shape a waveform for ``gr.Audio(type="numpy")`` -> ``(sr, float32[])``."""
    return int(sr), np.asarray(wav, dtype=np.float32)


def ref_from_gradio(audio) -> Optional[tuple[np.ndarray, int]]:
    """Normalise a ``gr.Audio`` value into the ``(waveform, sr)`` tuple that
    ``Qwen3TTSModel`` accepts as ``ref_audio``. Accepts numpy ``(sr, data)``,
    a filepath string, or ``None``.
    """
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], (int, np.integer)):
        sr, data = audio
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=-1)
        m = float(np.max(np.abs(data))) if data.size else 0.0
        if m > 1.0:
            data = data / m
        return data.astype(np.float32), int(sr)
    if isinstance(audio, str) and os.path.exists(audio):
        return audio  # the wrapper loads paths itself
    return None


def _edge_fade(w: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    """Linear fade-in/out on the chunk edges so joins into the silence gap don't
    click (a hard cut from a non-zero sample is a step discontinuity)."""
    n = min(int(sr * ms / 1000.0), len(w) // 2)
    if n <= 0:
        return w
    w = w.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    w[:n] *= ramp
    w[-n:] *= ramp[::-1]
    return w


def concat(wavs: list[np.ndarray], sr: int, gap_s: float = 0.12, fade_ms: float = 8.0) -> np.ndarray:
    """Join chunk waveforms with a short silence between them (long-form),
    edge-fading each chunk so the joins are click-free."""
    if not wavs:
        return np.zeros(0, dtype=np.float32)
    if len(wavs) == 1:
        return np.asarray(wavs[0], dtype=np.float32)
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, w in enumerate(wavs):
        out.append(_edge_fade(np.asarray(w, dtype=np.float32), sr, fade_ms))
        if i != len(wavs) - 1:
            out.append(gap)
    return np.concatenate(out)


def save_wav(wav: np.ndarray, sr: int = OUTPUT_SAMPLE_RATE, path: Optional[str] = None) -> str:
    """Write a waveform to disk (temp file if no path); returns the path."""
    import soundfile as sf

    if path is None:
        fd, path = tempfile.mkstemp(prefix="qvs_", suffix=".wav")
        os.close(fd)
    sf.write(path, np.asarray(wav, dtype=np.float32), int(sr))
    return path


def is_silent(wav: np.ndarray, thresh: float = 1e-3) -> bool:
    w = np.asarray(wav)
    return not (w.size and float(np.abs(w).max()) > thresh)
