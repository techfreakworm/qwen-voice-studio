# qwen-voice-studio — Design

> Authored by the project brain (`qwen-voice-brain`) and committed by the lead. Decisions logged in [DECISIONS.md](DECISIONS.md); implementation tasks in [PLAN.md](PLAN.md).

## 1. Purpose & principles
Open-source voice studio on the Qwen3-TTS-12Hz-1.7B family. Priorities, in order: (1) **fidelity** — never trade audio quality for latency or convenience; (2) **feature-completeness** — every capability the three checkpoints expose; (3) **one codebase, two platforms** — Apple-silicon MPS locally, HF ZeroGPU (CUDA, RTX Pro 6000 Blackwell) deployed, with platform variance isolated in one policy object.

## 2. Models & capabilities
| Checkpoint | Mode | API |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base | Voice cloning (ref audio+text, or x-vector-only); LoRA target | `generate_voice_clone`, `create_voice_clone_prompt` |
| …-CustomVoice | 9 preset voices + natural-language `instruct`; batch | `generate_custom_voice` |
| …-VoiceDesign | Voice from text description | `generate_voice_design` |

Shared: 10 languages + Auto (via `get_supported_languages()`), sampling knobs incl. `subtalker_*`, `max_new_tokens` (default 2048), 24 kHz output (always use the API-returned `sr`; never hardcode). Speakers via `get_supported_speakers()` at runtime. Codec `Qwen3-TTS-Tokenizer-12Hz` auto-downloaded.

## 3. Platform matrix
| | Local (M5 Max, 128 GB) | HF ZeroGPU Space (private) |
|---|---|---|
| Device | mps | cuda (half RTX Pro 6000, 48 GB) |
| Load | `from_pretrained(device_map="mps", dtype=bf16, attn_implementation="sdpa")` | `from_pretrained(dtype=bf16, attn="sdpa")` → module-level `.to("cuda")` (CUDA emulation; per ZeroGPU docs, never lazy-move inside `@spaces.GPU`) |
| Env | `PYTORCH_ENABLE_MPS_FALLBACK=1` | `@spaces.GPU(duration=callable)`, size default `large` |
| Python / torch | 3.12 / 2.13 (dev-validated) | 3.12.12 / torch pin ladder: 2.13.* → fallback 2.11.0 + local re-smoke |
| attn / compile | sdpa / no torch.compile | sdpa / no torch.compile (AOT-inductor = post-v1 only) |
| Residency | all-3-resident (adaptive degrade, §6) | all-3 on cuda at startup; `QVS_RESIDENCY=fork_move` fallback flag |

`spaces` is an optional import: `try: import spaces except ImportError: <no-op decorator shim>`. The decorator is effect-free off-ZeroGPU — one `app.py`, no entrypoint branching.

## 4. Module architecture
```
qvs/
  config.py    # Settings; env flags QVS_RESIDENCY, QVS_FORCE_SINGLE_RESIDENT,
               # QVS_DTYPE_OVERRIDE; pinned version constants; paths (data/voices, data/adapters)
  device.py    # DevicePolicy {platform, device, dtype, attn_impl, load_strategy};
               # detect ZeroGPU (spaces import / env) vs mps vs cpu; seed_all(seed);
               # memory gauge: macOS = vm_stat committed (wired+active+compressed) + pageout delta;
               # cuda = torch.cuda.memory_allocated; Watchdog(warn=72GB, abort=76GB, sample during gen)
  registry.py  # ModelRegistry: sequential load of 3 checkpoints per DevicePolicy;
               # per-load committed-delta logging; residency modes all_resident | single_on_demand;
               # adaptive degrade (§6); get(mode) accessor; codec-copy count reported at startup
  engine.py    # THE seam: synthesize(SynthesisRequest) -> AudioResult{wav, sr, meta}.
               # Request covers all 3 modes + voice source (ref_pair | prompt | xvector) + adapter flag
               # + all gen params incl. subtalker_*; per-request seeding via device.seed_all;
               # duration_estimate(request) used by UI and @spaces.GPU dynamic duration
  lora.py      # AdapterManager: resolve HF repo-id or upload → snapshot; locate adapter_config.json
               # by subfolder scan (Darija: talker_lora/); validate declared base model (warn+override);
               # attach at base.model.talker.model; enable/disable/unload; one active adapter;
               # surface bundled speaker_embedding.pt to VoiceLibrary; introspection (r, alpha, targets)
  voices.py    # VoiceLibrary: VoicePrompt {kind: ref_pair | full_prompt(.pt) | xvector(2048-d [+speaker_id]),
               # name, meta.json sidecar} stored under data/voices (gitignored);
               # sources: user save, create_voice_clone_prompt cache, adapter-bundled, Design→Clone bridge
  audio.py     # save/load wav (returned sr), duration/RMS/silence utils — shared by app AND tests
ui/            # clone.py, presets.py, design.py, lora_lab.py, library.py, shared.py (Advanced accordion
               # builder: sampling + subtalker + max_new_tokens + seed + language; status strip)
app.py         # wires tabs; gr.queue(default_concurrency_limit=1); handlers decorated @spaces.GPU
```

## 5. UI decomposition (5 tabs)
1. **Clone** — ref audio + ref text | `x_vector_only_mode` toggle | active-voice picker (from Library) | adapter dropdown (state owned by LoRA Lab) | Advanced | Generate.
2. **Preset Voices** — speaker dropdown (runtime introspection), `instruct` textbox, Advanced; batch: multiline one-per-line → zip (built last).
3. **Voice Design** — `instruct` (required), text, Advanced; **"Send to Library / use as Clone ref"** bridge (canonical pattern for making a designed voice reusable).
4. **LoRA Lab** — repo-id fetch / file upload, subfolder autodetect, config display (r/α/targets/base), attach + on/off toggle, unload, "save bundled speaker to Library", one canned-sentence quick-test routed through `synthesize()`. No generation UI duplication.
5. **Voice Library** — list/rename/delete prompts of all three kinds; "set active for Clone".

Global status strip: device, dtype, attn, per-model residency, committed memory, active adapter.

## 6. Memory policy (local hard gate)
Gauge = vm_stat committed ("Memory Used"), NOT process RSS (undercounts on MPS). Budget: baseline ≈55 GB (operator workload) + all-3 ≈14.6 GB → peak ≈69.6 GB measured. Watchdog: **warn ≥72 GB, hard-abort ≥76 GB**, sampled every few seconds during loads AND generation; pageout/swap delta growth = early warning, log loudly. **Adaptive startup:** baseline >65 GB before loading → degrade to single_on_demand + loud log. **fp32 contingency:** if MPS-bf16 ever fails quality checks, fp32 all-3 (~+28 GB) breaches the gate → fp32 forces single_on_demand. **Hard rule: exactly one model-holding process, ever.** Playwright drives the running app; no parallel pytest workers loading models.

## 7. ZeroGPU deployment design
Module-level: build models, `.to("cuda")` at import (emulation mode). Handlers: `@spaces.GPU(duration=estimate)` where `estimate = clamp(15 + k·expected_new_tokens, 30, 120)`; calibrate k from first Space runs; log GPU-seconds per request. First acceptance on Space: two consecutive generations with `next(model.parameters()).device` logged inside the fork both times → proves no orphan; if orphaned, flip `QVS_RESIDENCY=fork_move` (build CPU, idempotent move inside fork) — flag flip, not redesign. Quota: PRO 40 min/day; overage = pre-paid credits = **operator approval required**.

## 8. LoRA design + acceptance
Mechanism: unmerged PEFT attach at `base.model.talker.model` (q/k/v/o_proj, per Darija r8/α16), toggle = enable/disable_adapter_layers; removal = unload(); **never merge_and_unload in product flow**. Attach implementation decision tree: **(A)** keep PeftModel wrapper (`talker.model = PeftModel(...)`), toggle via `enable_adapters()/disable_adapters()` — test with generate smoke; if it works, ship A. **(B)** if A breaks the generate path → `peft.inject_adapter_in_model` (model identity & module tree stay native), toggle by iterating LoraLayers. **(C)** last resort → `merge_and_unload` with documented one-way semantics ("toggle off" = reload Base talker; no repeated merge/reload cycles). Acceptance (fixed seed, adapter-on vs -off): (i) `generate_voice_clone` works post-attach; (ii) toggle OFF→ON changes output audibly, <1 s; (iii) after unload, same-seed generation matches pre-attach baseline; (iv) Darija adapter + bundled `speaker_embedding.pt` {(2048,), speaker_id 3000} through the x-vector path yields the Darija speaker's voice, audibly distinct from adapter-off. Uploads: accept zip/safetensors+config; store under data/adapters.

## 9. Testing strategy
Layered: (0) env/version print + memory gauge sanity; (1) headless engine smokes per mode (wav exists, duration>0.5 s, RMS above silence floor, sr==API sr, no NaN); (2) MPS silent-corruption check ONCE: mel/STFT path (modeling_qwen3_tts.py:447/459) MPS-vs-CPU cosine >0.999 IF stft executes natively (`FALLBACK=1` only rescues *unimplemented* ops — implemented-but-buggy fails silently) + clone-resemblance ear check; (3) Playwright full UI walkthrough, every tab + param, **programmatic audio asserts** (screenshots prove UI state, never audio); (4) same suite against the private Space. Determinism policy: same-seed repeatability asserted same-device only; MPS vs CUDA never bit-identical — assert properties, not waveforms.

## 10. Pinned versions
py 3.12; torch 2.13 (dev) with Space ladder →2.11.0; transformers==4.57.3 (**5.x breaks qwen_tts**); qwen-tts==0.1.1; peft==0.19.1; accelerate==1.12.0; huggingface_hub<1.0; gradio 6.17.3; spaces (Space only). `requirements.txt` = Space runtime; `requirements-dev.txt` = + playwright, pytest, psutil.

## 11. Risks register (accepted)
bf16-on-MPS numerics (mitigated: A/B smoke + fp32 contingency); torch 2.13 Space rejection (mitigated: pin ladder + re-smoke trigger); module-level cuda orphan regression (mitigated: fork_move flag + device-log acceptance); adapter key-prefix mismatch (mitigated: inspect-first + remap); Space cold-boot re-downloads ~14 GB (accepted: private personal Space; persistent storage = money = operator call); streaming absent (accepted: wrapper limitation, documented).
