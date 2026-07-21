"""Static configuration: model repos, speakers, languages, generation defaults.

Runtime truth (supported speakers/languages) is always cross-checked against the
loaded model via ``get_supported_speakers()`` / ``get_supported_languages()``;
the tables here carry the human-facing metadata the model does not expose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---- model checkpoints (the whole 1.7B family) --------------------------------
MODEL_REPOS: dict[str, str] = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",            # voice cloning
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",  # 9 preset voices + instruct
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",  # text -> new voice
}
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

MODE_LABELS = {
    "base": "Voice Clone",
    "custom_voice": "Preset Voices",
    "voice_design": "Voice Design",
}

OUTPUT_SAMPLE_RATE = 24000  # all checkpoints emit 24 kHz


# ---- preset speakers (CustomVoice) --------------------------------------------
@dataclass(frozen=True)
class Speaker:
    key: str          # id passed to the model (case-insensitive)
    display: str      # UI label
    description: str
    language: str     # native language / dialect
    gender: str

SPEAKERS: list[Speaker] = [
    Speaker("Vivian",   "Vivian",   "Bright, slightly edgy young female voice.",        "Chinese",           "female"),
    Speaker("Serena",   "Serena",   "Warm, gentle young female voice.",                 "Chinese",           "female"),
    Speaker("Uncle_Fu", "Uncle Fu", "Seasoned male voice with a low, mellow timbre.",   "Chinese",           "male"),
    Speaker("Dylan",    "Dylan",    "Youthful Beijing male voice, clear natural timbre.", "Chinese (Beijing)", "male"),
    Speaker("Eric",     "Eric",     "Lively Chengdu male voice, slightly husky.",       "Chinese (Sichuan)", "male"),
    Speaker("Ryan",     "Ryan",     "Dynamic male voice with strong rhythmic drive.",   "English",           "male"),
    Speaker("Aiden",    "Aiden",    "Sunny American male voice, clear midrange.",       "English",           "male"),
    Speaker("Ono_Anna", "Ono Anna", "Playful Japanese female voice, light and nimble.", "Japanese",          "female"),
    Speaker("Sohee",    "Sohee",    "Warm Korean female voice with rich emotion.",      "Korean",            "female"),
]
SPEAKER_KEYS = [s.key for s in SPEAKERS]


# ---- languages ----------------------------------------------------------------
# Display -> value passed to the model. "Auto" lets the model detect.
LANGUAGES: dict[str, str] = {
    "Auto (detect)": "Auto",
    "Chinese": "Chinese",
    "English": "English",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "German": "German",
    "French": "French",
    "Russian": "Russian",
    "Portuguese": "Portuguese",
    "Spanish": "Spanish",
    "Italian": "Italian",
}


# ---- generation defaults (match qwen_tts hard defaults) ------------------------
@dataclass
class GenDefaults:
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: int = 50
    max_new_tokens: int = 2048
    seed: int = -1  # -1 => random

GEN_DEFAULTS = GenDefaults()

# Long-form: split text longer than this many characters into sentence chunks.
LONGFORM_CHAR_THRESHOLD = 400


# ---- example prompts (seed the UI so it never opens empty) --------------------
EMOTION_PRESETS: list[str] = [
    "Very happy and upbeat.",
    "Calm and reassuring.",
    "Whisper softly.",
    "Angry and forceful.",
    "Sad and wistful.",
    "Excited, fast-paced.",
    "Warm, like a bedtime story.",
    "Sarcastic and dry.",
]

VOICE_DESIGN_EXAMPLES: list[str] = [
    "A soft, wondrous elderly woman narrating a nature documentary.",
    "A gravelly noir detective, tired but sharp, speaking slowly.",
    "An energetic young sports announcer at a stadium.",
    "A gentle ASMR voice, breathy and close.",
    "A confident female CEO delivering a keynote.",
]
