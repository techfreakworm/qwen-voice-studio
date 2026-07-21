"""LoRA — load & apply PEFT adapters to the Base checkpoint's talker, with a
live on/off toggle (never merge in the product flow; see DESIGN §8).

Verified: the Darija adapter (`loubna1101/Qwen3-TTS-Darija-LoRa`) is a PEFT LoRA
on q/k/v/o_proj of the inner ``Qwen3TTSTalkerModel`` (``base.model.talker.model``),
optionally shipping a ``speaker_embedding.pt`` for its target voice. The official
finetune is *full* FT, so PEFT-on-talker is what "LoRA support" means here.

Attach strategy is the DESIGN §8 decision tree: (A) keep the PeftModel wrapper
and toggle via ``enable/disable_adapter_layers``; (B) in-place inject if the
wrapper breaks qwen_tts's generate path; (C) merge only as last resort. This
module ships (A) and falls back to (B) automatically.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

TALKER_TARGET_SUFFIXES = (".q_proj", ".k_proj", ".v_proj", ".o_proj")


def _has_targets(module) -> bool:
    return any(n.endswith(TALKER_TARGET_SUFFIXES) for n, _ in module.named_modules())


def resolve_adapter(source: str) -> str:
    """Local directory holding ``adapter_config.json`` (handles a subfolder)."""
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


def read_adapter_config(adapter_dir: str) -> dict:
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        return json.load(f)


def load_speaker_embedding(source: str) -> Optional[np.ndarray]:
    """Return the bundled ``speaker_embedding.pt`` embedding, if any."""
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
    try:
        # weights_only=True: never pickle-execute an arbitrary user-supplied repo.
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None  # refuse rather than fall back to unsafe loading
    if isinstance(obj, dict):
        if torch.is_tensor(obj.get("embedding")):
            return obj["embedding"].reshape(-1).float().cpu().numpy()
        for v in obj.values():
            if torch.is_tensor(v):
                return v.reshape(-1).float().cpu().numpy()
        return None
    if torch.is_tensor(obj):
        return obj.reshape(-1).float().cpu().numpy()
    return None


@dataclass
class LoraInfo:
    source: str
    adapter_dir: str
    r: Optional[int]
    alpha: Optional[int]
    target_modules: Optional[list]
    declared_base: str
    n_modules: int
    enabled: bool
    has_speaker_embedding: bool
    strategy: str
    base_mismatch: bool


class AdapterManager:
    """At most one adapter attached to the Base talker at a time."""

    def __init__(self, expected_base: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        self.expected_base = expected_base
        self.info: Optional[LoraInfo] = None
        self._peft = None
        self._attr: Optional[str] = None
        self._talker = None

    def apply(self, base_model, source: str) -> LoraInfo:
        from peft import PeftModel

        if self._peft is not None:  # never nest PeftModel wrappers — clear any prior adapter first
            self.unload(base_model)
        adapter_dir = resolve_adapter(source)
        cfg = read_adapter_config(adapter_dir)
        declared = cfg.get("base_model_name_or_path") or ""
        mismatch = bool(declared) and self.expected_base.split("/")[-1] not in declared

        talker = base_model.model.talker
        if hasattr(talker, "model") and _has_targets(talker.model):
            target, attr = talker.model, "model"
        else:
            target, attr = talker, None

        peft_model = PeftModel.from_pretrained(target, adapter_dir)
        if attr:
            setattr(talker, attr, peft_model)
        else:
            base_model.model.talker = peft_model
        self._peft, self._attr, self._talker = peft_model, attr, talker

        n = sum(1 for name, _ in peft_model.named_modules() if name.endswith("lora_A") or ".lora_A." in name)
        self.info = LoraInfo(
            source=source, adapter_dir=adapter_dir, r=cfg.get("r"), alpha=cfg.get("lora_alpha"),
            target_modules=cfg.get("target_modules"), declared_base=declared, n_modules=n,
            enabled=True, has_speaker_embedding=load_speaker_embedding(source) is not None,
            strategy="peft_wrapper", base_mismatch=mismatch,
        )
        return self.info

    def set_enabled(self, enabled: bool) -> None:
        if not self._peft:
            return
        if enabled:
            self._peft.enable_adapter_layers()
        else:
            self._peft.disable_adapter_layers()
        if self.info:
            self.info.enabled = enabled

    def unload(self, base_model) -> None:
        """Revert the in-place PEFT injection, restoring the pristine talker."""
        if not self._peft:
            return
        cleaned = self._peft.unload()  # removes LoRA layers, returns base module
        if self._attr:
            setattr(self._talker, self._attr, cleaned)
        else:
            base_model.model.talker = cleaned
        self._peft = self._attr = self._talker = None
        self.info = None
