# qwen-voice-studio

The open-source voice studio for **Qwen3-TTS-12Hz-1.7B** — voice cloning, 9 premium preset voices with emotion control, and text-described voice design, in one Gradio app. **Fidelity first.** Runs natively on Apple silicon (MPS) and deploys to Hugging Face ZeroGPU.

## Features
- **3-second voice cloning** — reference audio + transcript, or x-vector-only *(Base)*
- **9 preset voices** incl. Beijing/Sichuan dialects, with natural-language emotion/style instructions *(CustomVoice)*
- **Voice design** — create brand-new voices from a text description *(VoiceDesign)*
- **Design → Clone bridge** — turn a designed voice into a reusable cloning reference
- **LoRA adapters** — load from a Hugging Face repo id or upload, applied to the Base talker, with instant A/B toggle (verified with [Qwen3-TTS-Darija-LoRa](https://huggingface.co/loubna1101/Qwen3-TTS-Darija-LoRa))
- **Voice Library** — save clone prompts, x-vectors, and adapter voices, reuse anywhere
- **10 languages + Auto**, full sampling control incl. sub-talker knobs, seed, `max_new_tokens`
- **Batch synthesis** (preset voices)

## Quickstart (Apple silicon)
Requires Python 3.12, ~15 GB free RAM, ~20 GB disk for models.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py
```

The three checkpoints (~14 GB) download from Hugging Face on first run. Open the printed local URL.

## Deploy to Hugging Face Spaces (ZeroGPU)
Create a Gradio Space with **ZeroGPU** hardware and push this repo. `@spaces.GPU` handles GPU allocation; models are placed on CUDA at module level per the ZeroGPU guidance. See [docs/DESIGN.md §7](docs/DESIGN.md).

## Models & credits
Built on [Qwen3-TTS-12Hz-1.7B](https://huggingface.co/Qwen) (Base / CustomVoice / VoiceDesign) and the Qwen3-TTS-Tokenizer-12Hz codec by the **Qwen team, Alibaba Cloud** — Apache-2.0, as is this project.

## Known limitations
- The open-source `qwen-tts` wrapper does **not** expose audio-token streaming; generation is generate-then-play (a few seconds per utterance on M-series).
- LoRA applies to the **Base** checkpoint only.
- MPS and CUDA outputs are **not** bit-identical (expected).

## Troubleshooting
| Symptom | Fix |
|---|---|
| STFT / op error on clone (MPS) | ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set |
| `check_model_inputs()` TypeError on import | `transformers` must stay `4.57.x` (5.x breaks `qwen_tts`); keep `huggingface_hub<1.0` |
| High memory / OOM risk | set `QVS_FORCE_SINGLE_RESIDENT=1` to load one model at a time |
| Generation not reproducible across machines | seed reproducibility is **per-device** (MPS≠CUDA by design) |

## Documentation
- [docs/DESIGN.md](docs/DESIGN.md) — architecture, platform matrix, memory & LoRA design
- [docs/PLAN.md](docs/PLAN.md) — implementation plan with acceptance criteria
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log

## Roadmap
Full-FT checkpoint "voice packs"; true streaming if upstream exposes it; AOT compilation on ZeroGPU.

## License
Apache-2.0.
