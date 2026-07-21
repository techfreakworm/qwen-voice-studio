"""Voice library — save cloned/designed voices as reusable prompt files.

A saved voice is the ``VoiceClonePromptItem`` list produced by
``create_voice_clone_prompt`` (speaker embedding + optional reference codes),
persisted with ``torch.save``. This powers both "save a cloned voice" and the
"Voice Design -> Clone" bridge (design a persona, lock it in, reuse it).
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import numpy as np

VOICE_DIR = os.environ.get("QVS_VOICE_DIR", "voice_library")


def _ensure_dir() -> str:
    os.makedirs(VOICE_DIR, exist_ok=True)
    return VOICE_DIR


def _path(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
    return os.path.join(_ensure_dir(), f"{safe or 'voice'}.pt")


def list_voices() -> list[str]:
    if not os.path.isdir(VOICE_DIR):
        return []
    return sorted(f[:-3] for f in os.listdir(VOICE_DIR) if f.endswith(".pt"))


def delete_voice(name: str) -> bool:
    p = _path(name)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def save_voice(model, name: str, ref_audio, ref_text: Optional[str], x_vector_only: bool,
               note: str = "") -> str:
    """Extract a clone prompt from reference audio and persist it."""
    import torch

    items = model.create_voice_clone_prompt(
        ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only
    )
    payload = {"items": [asdict(it) for it in items], "meta": {"note": note, "name": name}}
    path = _path(name)
    torch.save(payload, path)
    return path


def save_voice_from_embedding(name: str, embedding: np.ndarray, ref_text: Optional[str] = None,
                              note: str = "") -> str:
    """Build an x-vector-only voice from a raw speaker embedding (e.g. a LoRA's
    shipped ``speaker_embedding.pt``)."""
    import torch

    emb = torch.as_tensor(np.asarray(embedding, dtype=np.float32)).reshape(-1)
    item = {
        "ref_code": None,
        "ref_spk_embedding": emb,
        "x_vector_only_mode": True,
        "icl_mode": False,
        "ref_text": ref_text,
    }
    payload = {"items": [item], "meta": {"note": note, "name": name}}
    path = _path(name)
    torch.save(payload, path)
    return path


def load_voice(name: str):
    """Return a ``list[VoiceClonePromptItem]`` ready to pass as
    ``voice_clone_prompt`` to ``generate_voice_clone``."""
    import torch
    from qwen_tts import VoiceClonePromptItem

    payload = torch.load(_path(name), map_location="cpu", weights_only=False)
    items = []
    for d in payload["items"]:
        ref_code = d.get("ref_code")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.as_tensor(ref_code)
        spk = d["ref_spk_embedding"]
        if not torch.is_tensor(spk):
            spk = torch.as_tensor(spk)
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", not d.get("x_vector_only_mode", False))),
                ref_text=d.get("ref_text"),
            )
        )
    return items
