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
| Python / torch | 3.12 / 2.11.0 (2.13 = scratch venv) | 3.12.12 / torch==2.11.0 (ZeroGPU patch-exact) |
| Load (as shipped) | `from_pretrained(device_map="mps", dtype=bf16, attn="sdpa")` | CPU load → `.to("cuda")` **lazily in the first `@spaces.GPU` request per mode** (§13, D9) |
| attn / compile | sdpa / no torch.compile | sdpa / no torch.compile (AOT-inductor = post-v1 only) |
| Residency | all-3-resident; request-time adaptive evict >60 GB (M2, darwin-only) | lazy-load in fork (module-level preload overran Space startup; D9) |

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
Empirically (D9): module-level preload exceeded the Space startup window and parent-side runtime CUDA ops can fail under emulation; models **lazy-load on first use inside the fork**, and all state-carrying features are designed **stateless per-request** (D8). Handlers are `@spaces.GPU`-decorated (generation, save-voice, design-bridge, lora quick-test); state/inspect handlers stay undecorated (parent). Quota: PRO 40 min/day included; **overage = pre-paid credits = operator approval required** (verified staying within the included minutes — no credits used).

## 8. LoRA design + acceptance
LoRA is per-request (D8): the generation request optionally names an adapter; it is attached to the Base talker (`base.model.talker.model`, q/k/v/o_proj — Darija r8/α16) inside the GPU context for that request and fully unloaded afterwards (`unload()` is verified bit-exact-restorative). A/B comparison = fixed seed, adapter field set vs empty. **No persistent adapter state exists on either platform.** The Clone tab exposes an optional adapter field; LoRA Lab = inspect config + a self-contained quick-test (attach → generate with the bundled `speaker_embedding.pt` x-vector → unload). Adapters resolve from an HF repo-id or local path (subfolder-scanned for `adapter_config.json`; Darija's is under `talker_lora/`); `speaker_embedding.pt` is loaded with `weights_only=True` — never pickle-execute an untrusted payload.

Acceptance (verified): post-attach `generate_voice_clone` works; `unload()` restores exact pre-attach output (fixed-seed diff 0.0); Darija adapter + bundled embedding {(2048,), speaker_id 3000} yields the Darija voice, audibly distinct from adapter-off; on the Space, clone+adapter and quick-test both produce real audio.

## 9. Testing strategy
Layered: (0) env/version print + memory gauge sanity; (1) headless engine smokes per mode (wav exists, duration>0.5 s, RMS above silence floor, sr==API sr, no NaN); (2) MPS silent-corruption check ONCE: mel/STFT path (modeling_qwen3_tts.py:447/459) MPS-vs-CPU cosine >0.999 IF stft executes natively (`FALLBACK=1` only rescues *unimplemented* ops — implemented-but-buggy fails silently) + clone-resemblance ear check; (3) Playwright full UI walkthrough, every tab + param, **programmatic audio asserts** (screenshots prove UI state, never audio); (4) same suite against the private Space. Determinism policy: same-seed repeatability asserted same-device only; MPS vs CUDA never bit-identical — assert properties, not waveforms.

## 10. Pinned versions
py 3.12; **torch==2.11.0 both platforms** (ZeroGPU patch-exact supported set {2.8.0, 2.9.1, 2.10.0, 2.11.0}; 2.13 kept only as a local scratch venv); transformers==4.57.3 (**5.x breaks qwen_tts**); qwen-tts==0.1.1; peft==0.19.1; accelerate==1.12.0; huggingface_hub<1.0; gradio 6.17.3; spaces (Space only). attn = **sdpa everywhere** (no flash-attn). `requirements.txt` = Space runtime; `requirements-dev.txt` = + playwright, pytest, psutil.

## 11. Risks register (accepted)
bf16-on-MPS numerics (mitigated: A/B smoke + fp32 contingency); torch 2.13 Space rejection (mitigated: pin ladder + re-smoke trigger); module-level cuda orphan regression (mitigated: fork_move flag + device-log acceptance); adapter key-prefix mismatch (mitigated: inspect-first + remap); Space cold-boot re-downloads ~14 GB (accepted: private personal Space; persistent storage = money = operator call); streaming absent (accepted: wrapper limitation, documented).

## 12. Long-form chunking
Text over `LONGFORM_CHAR_THRESHOLD` (400 chars) is split into sentence chunks, synthesized per chunk, and concatenated. Acceptance criteria:
- **(a) Voice consistency** — Clone reuses the SAME voice prompt/x-vector for every chunk; Preset the same speaker+instruct; Design the same instruct (Design regenerating per-chunk can drift → the honest mitigation is the Design→Clone bridge: design once, clone for long-form).
- **(b) Click-free joins** — each chunk gets a short (~8 ms) edge fade before the silence gap so joins have no step discontinuity (`audio._edge_fade`).
- **(c) Seed semantics** — one seed for the whole request (applied once); documented as full-request reproducible on the same device. (Per-chunk `seed+idx` derivation is a future nicety for chunk-level regeneration.)
- **(d) Sentence boundaries** — splits only after terminators `.!?。！？…`, using `\s*` so CJK (no post-terminator whitespace) chunks correctly; never mid-word/mid-clause.

## 13. Deployment reality (as shipped)
Verified deltas from the pre-build design: ZeroGPU models **lazy-load in-fork** on first request per mode (module-level preload overran the Space startup window → RUNTIME_ERROR; **D9**); Gradio launched with `server_name="0.0.0.0"`, `ssr_mode=False`; RAM watchdog disabled on ZeroGPU (container psutil misreports); `packages.txt` = sox/ffmpeg; binaries via git-LFS; **LoRA per-request (D8)**. Space verified end-to-end (5/5 modes, real audio) within PRO's included 40 min/day ZeroGPU quota — **no paid credits used**.
