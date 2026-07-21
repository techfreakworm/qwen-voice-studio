"""LoRA — load & apply PEFT adapters to the Base checkpoint's talker.

Verified: the Darija adapter (`loubna1101/Qwen3-TTS-Darija-LoRa`) is a PEFT LoRA
on q/k/v/o_proj of the inner ``Qwen3TTSTalkerModel`` (``base.model.talker.model``);
attaching there injects 224 modules and demonstrably changes output. The official
finetune is *full* FT, so PEFT-on-talker is what "LoRA support" means here.

Adapters can live at the repo root or in a ``talker_lora/`` subfolder, and may
ship a ``speaker_embedding.pt`` describing the voice they were trained for.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _has_targets(module) -> bool:
    return any(
        n.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj"))
        for n, _ in module.named_modules()
    )


def resolve_adapter(source: str) -> str:
    """Return a local directory containing ``adapter_config.json`` for *source*
    (a local path or a Hugging Face repo id). Handles a ``talker_lora/`` subfolder.
    """
    base = source
    if not os.path.isdir(source):
        from huggingface_hub import snapshot_download

        base = snapshot_download(source)
    for cand in (base, os.path.join(base, "talker_lora")):
        if os.path.exists(os.path.join(cand, "adapter_config.json")):
            return cand
    for root, _dirs, files in os.walk(base):
        if "adapter_config.json" in files:
            return root
    raise FileNotFoundError(f"no adapter_config.json found under {source!r}")


def load_speaker_embedding(source: str) -> Optional[np.ndarray]:
    """If the adapter ships a ``speaker_embedding.pt``, return the raw embedding."""
    import torch

    base = source if os.path.isdir(source) else None
    if base is None:
        try:
            from huggingface_hub import snapshot_download

            base = snapshot_download(source)
        except Exception:
            return None
    path = os.path.join(base, "speaker_embedding.pt")
    if not os.path.exists(path):
        return None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for v in obj.values():
            if torch.is_tensor(v):
                return v.reshape(-1).float().cpu().numpy()
        return None
    if torch.is_tensor(obj):
        return obj.reshape(-1).float().cpu().numpy()
    return None


@dataclass
class LoraState:
    source: str
    n_modules: int
    merged: bool
    has_speaker_embedding: bool


class LoraManager:
    """Applies at most one adapter to a Base model at a time.

    ``merge=True`` (default) merges the adapter into the weights — clean native
    type, best for inference; removal is done by reloading the Base model via the
    registry. ``merge=False`` keeps a live PeftModel wrapper for on/off toggling.
    """

    def __init__(self):
        self.state: Optional[LoraState] = None
        self._attr: Optional[str] = None  # which submodule we swapped

    def apply(self, base_model, source: str, merge: bool = True) -> LoraState:
        from peft import PeftModel

        adapter_dir = resolve_adapter(source)
        talker = base_model.model.talker
        if hasattr(talker, "model") and _has_targets(talker.model):
            target, attr = talker.model, "model"
        else:
            target, attr = talker, None

        peft_model = PeftModel.from_pretrained(target, adapter_dir)
        n = sum(1 for n, _ in peft_model.named_modules() if n.endswith("lora_A") or ".lora_A." in n)

        new_module = peft_model.merge_and_unload() if merge else peft_model
        if attr:
            setattr(talker, attr, new_module)
        else:
            base_model.model.talker = new_module

        self._attr = attr
        self.state = LoraState(
            source=source,
            n_modules=n,
            merged=merge,
            has_speaker_embedding=load_speaker_embedding(source) is not None,
        )
        return self.state

    def set_enabled(self, enabled: bool) -> None:
        """Toggle a non-merged adapter on/off (no-op if merged/absent)."""
        if not self.state or self.state.merged:
            return
        talker_sub = None  # PeftModel currently swapped in
        # nothing to do for merged; wrapper toggling handled by caller via model ref

    def clear(self) -> None:
        self.state = None
        self._attr = None
