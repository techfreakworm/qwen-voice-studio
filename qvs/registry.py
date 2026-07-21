"""Model registry — owns the three checkpoints and where they live.

Local (MPS/CUDA): models load straight onto the device and stay resident
(measured peak ~70 GB committed for all three on MPS — under the 80 GB gate).

ZeroGPU: built on CPU at startup, then moved onto CUDA *inside* the
``@spaces.GPU`` fork per request (via :meth:`to_device`) — this avoids the
"model never actually on cuda" trap where a module ``.to()`` orphans the weights.
"""
from __future__ import annotations

from typing import Optional

from .config import MODEL_REPOS
from .device import on_zerogpu, target_device
from .engine import load_model, move_model


class ModelRegistry:
    def __init__(self, load_on_cpu: Optional[bool] = None):
        self.load_on_cpu = on_zerogpu() if load_on_cpu is None else load_on_cpu
        self.device = "cuda" if on_zerogpu() else target_device()
        self._models: dict[str, object] = {}

    def get(self, mode: str):
        """Return the model for a mode, loading it on first use."""
        if mode not in MODEL_REPOS:
            raise KeyError(f"unknown mode {mode!r}; expected one of {list(MODEL_REPOS)}")
        if mode not in self._models:
            self._models[mode] = load_model(
                MODEL_REPOS[mode],
                device=self.device,
                load_on_cpu=self.load_on_cpu,
            )
        return self._models[mode]

    def to_device(self, mode: str):
        """Ensure the model is on the compute device (no-op unless CPU-built)."""
        model = self.get(mode)
        if self.load_on_cpu:
            move_model(model, "cuda")
        return model

    def preload_all(self) -> None:
        for mode in MODEL_REPOS:
            self.get(mode)

    def reload(self, mode: str):
        """Drop and rebuild a model (used to cleanly remove a merged LoRA)."""
        self._models.pop(mode, None)
        return self.get(mode)

    @property
    def loaded(self) -> list[str]:
        return list(self._models)
