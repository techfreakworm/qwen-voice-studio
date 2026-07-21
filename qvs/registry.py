"""Model registry — owns the three checkpoints and where they live.

Residency (env ``QVS_RESIDENCY``):
  - ``auto`` (default): keep models resident until committed memory crosses
    ``QVS_EVICT_CEILING`` (default 60 GB), then evict the others before loading a
    new one. On an unloaded machine this behaves like ``all``; on the operator's
    loaded workstation (~58 GB baseline) it degrades to single-resident so a
    generation spike never breaches the 76 GB abort. This is the DESIGN §6
    adaptive-degrade, driven by live memory rather than only startup baseline.
  - ``all``: never evict (fastest switching; needs headroom).
  - ``single``: never hold more than the current model.
  - ``fork_move``: ZeroGPU fallback — build on CPU, move to cuda inside the fork.

Local (MPS/CUDA): ``device_map`` load straight onto the device. ZeroGPU:
built on CPU at startup then ``.to('cuda')`` at module level (CUDA emulation),
unless ``fork_move`` moves inside the ``@spaces.GPU`` fork.
"""
from __future__ import annotations

import gc
import os
from typing import Optional

from .config import MODEL_REPOS
from .device import on_zerogpu, target_device
from .engine import free_cache, load_model, move_model
from .memory import committed_gb


class ModelRegistry:
    def __init__(self, residency: Optional[str] = None):
        # ZeroGPU: models live resident on CPU in the parent process and are moved
        # to CUDA inside each @spaces.GPU fork (fork_move). This is the pattern that
        # actually persists across requests — a fork that *loads* the model loses it
        # when it exits, so lazy-load-in-fork reloads ~4.5 GB every call. With the
        # parent holding them (warmed in a background thread at startup), each request
        # is just a fast CPU→GPU move + generate.
        # Local: adaptive by default (evict when RAM is tight).
        default = "fork_move" if on_zerogpu() else "auto"
        self.residency = residency or os.environ.get("QVS_RESIDENCY", default)
        self.fork_move = self.residency == "fork_move"
        self.on_cpu = on_zerogpu()
        self.device = "cuda" if on_zerogpu() else target_device()
        self.evict_ceiling = float(os.environ.get("QVS_EVICT_CEILING", "60"))
        self._models: dict[str, object] = {}
        self._codec_ids: dict[str, int] = {}
        self._device_logged: set[str] = set()
        self.load_log: list[tuple[str, float, float]] = []

    # ---- eviction ------------------------------------------------------------
    def _evict_others(self, keep: str) -> None:
        for mode in [m for m in self._models if m != keep]:
            del self._models[mode]
            self._codec_ids.pop(mode, None)
            print(f"[registry] evicted {mode} to free memory", flush=True)
        gc.collect()
        free_cache()

    def _should_evict(self) -> bool:
        if self.residency == "all" or self.fork_move:
            return False
        if self.residency == "single":
            return True
        return committed_gb() > self.evict_ceiling  # auto

    # ---- loading -------------------------------------------------------------
    def _load_one(self, mode: str):
        if self.on_cpu and self.fork_move:
            return load_model(MODEL_REPOS[mode], device=self.device, load_on_cpu=True)
        if self.on_cpu:  # ZeroGPU non-fork: build on CPU, move at module level
            m = load_model(MODEL_REPOS[mode], device=self.device, load_on_cpu=True)
            return move_model(m, "cuda")
        return load_model(MODEL_REPOS[mode], device=self.device)  # local device_map

    def get(self, mode: str):
        if mode not in MODEL_REPOS:
            raise KeyError(f"unknown mode {mode!r}; expected one of {list(MODEL_REPOS)}")
        if mode not in self._models:
            if self._should_evict():
                self._evict_others(mode)
            before = committed_gb()
            model = self._load_one(mode)
            after = committed_gb()
            self._models[mode] = model
            try:
                self._codec_ids[mode] = id(model.model.speech_tokenizer)
            except Exception:
                pass
            self.load_log.append((mode, after - before, after))
            print(f"[registry] loaded {mode}: +{after - before:.1f} GB -> {after:.1f} GB committed "
                  f"(resident: {list(self._models)})", flush=True)
        return self._models[mode]

    def to_device(self, mode: str):
        """Ensure the model is on the compute device (moves in-fork if fork_move)."""
        model = self.get(mode)
        if self.fork_move:
            move_model(model, "cuda")
        if mode not in self._device_logged:
            try:
                print(f"[device] {mode} generating on {next(model.model.parameters()).device}", flush=True)
                self._device_logged.add(mode)
            except Exception:
                pass
        return model

    def preload_all(self) -> None:
        for mode in MODEL_REPOS:
            self.get(mode)

    def reload(self, mode: str):
        self._models.pop(mode, None)
        self._codec_ids.pop(mode, None)
        gc.collect()
        free_cache()
        return self.get(mode)

    def codec_shared(self) -> Optional[bool]:
        """True if loaded models share one codec instance; None if <2 loaded."""
        ids = list(self._codec_ids.values())
        if len(ids) < 2:
            return None
        return len(set(ids)) == 1

    @property
    def loaded(self) -> list[str]:
        return list(self._models)
