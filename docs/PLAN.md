# Implementation Plan

> Authored by the project brain. Ordered tasks; each has binding acceptance criteria (AC). Cross-cutting rules at the bottom apply to every task. (The lead ran ahead of this plan during design; a working core + app already exists, so several tasks land as reconciliation against the rulings rather than greenfield.)

**T0 — Scaffold (do first, push immediately).** Repo tree per DESIGN §4; `LICENSE` = Apache-2.0; `.gitignore` (data/, *.wav except docs/samples/, __pycache__, .venv, .DS_Store); `requirements.txt` (Space runtime, pinned exact) + `requirements-dev.txt` (adds pytest, playwright, psutil); README; docs/{DESIGN,PLAN,DECISIONS}.md.
*AC:* `git push` green to public GitHub; sole-operator authorship (no co-author lines, no "Generated with"); no venv/test artifacts in repo (they live under ~/Projects/tests).

**T1 — `config.py` + `device.py`.** DevicePolicy detection (ZeroGPU via spaces-import/env → cuda; else mps; else cpu), `seed_all`, memory gauge (vm_stat committed on macOS; torch.cuda stats on cuda), Watchdog (warn 72 / abort 76 GB, pageout-delta tracking, background sampling thread).
*AC:* policy script prints `mps/bf16/sdpa/all_resident` locally; gauge within ±2 GB of Activity Monitor "Memory Used"; watchdog abort path proven with an artificially low threshold; `QVS_*` env overrides all work.

**T2 — `registry.py`.** Sequential 3-checkpoint load per policy; per-load committed-delta logging; `all_resident | single_on_demand`; adaptive degrade (baseline >65 GB); codec-instance count reported.
*AC:* all-3 resident, peak <72 GB, per-model deltas logged; `QVS_FORCE_SINGLE_RESIDENT=1` exercises degrade path; report whether codec is shared or 3×.

**T3 — `audio.py` + `engine.py`.** `synthesize()` for all three modes headless; per-request seeding; `duration_estimate()`; watchdog sampling active during generation.
*AC:* three wavs — non-silent (RMS > floor), duration >0.5 s, sr == API-returned; same seed+device ⇒ repeatable; different seed ⇒ differs; estimator sane for short/long text.

**T4 — `voices.py`.** VoicePrompt kinds ref_pair / full_prompt / xvector; save/load with meta sidecars under data/voices.
*AC:* roundtrip all three kinds; each usable as clone voice source; Darija `speaker_embedding.pt` imports as xvector{(2048,), speaker_id 3000} and generates.

**T5 — `lora.py`.** AdapterManager per DESIGN §8.
*AC:* DESIGN §8 (i)–(iv) all pass with the Darija adapter, including <1 s toggle and post-unload same-seed baseline match.

**T6 — `ui/` + `app.py`.** Five tabs per DESIGN §5, shared Advanced builder, status strip, `gr.queue(default_concurrency_limit=1)`, spaces no-op shim, `@spaces.GPU(duration=estimate)` on handlers. Use the frontend-design plugin for the visual pass (operator preference).
*AC:* Playwright walkthrough of EVERY tab and param group with **programmatic audio asserts** (file, duration, RMS, sr) + screenshots for UI state; Design→Clone bridge produces a working clone ref; memory stays <72 GB for the entire suite; one app process only.

**T7 — Batch (Preset tab) + polish.** Multiline → zip. *AC:* 3 lines → zip of 3 valid wavs; UI errors are readable (no raw tracebacks).

**T8 — Pre-deploy gate.** Finalize README; Space `requirements.txt` (torch pin ladder per D7); push GitHub.
*AC:* fresh-clone + README-only install reproduces the app locally (doc-follow test); repo public, Space repo private confirmed.

**T9 — Space deploy + verify.** Create private ZeroGPU Space `techfreakworm/qwen-voice-studio`; deploy; watch first boot logs; **acceptance #1: two consecutive generations logging `next(model.parameters()).device` inside the fork = cuda both times** (orphan check → else flip `QVS_RESIDENCY=fork_move`); if torch 2.13 rejected → pin 2.11.0 AND rerun local smoke suite on a 2.11.0 venv BEFORE redeploying; then full Playwright suite against the Space; review GPU-seconds vs 40 min/day quota; calibrate duration-estimate k.
*AC:* all modes + LoRA green on the Space; GitHub synced at the deploy commit; quota log reviewed; **no credit top-up without operator approval.**

## Cross-cutting rules
Commit at least per-task, push regularly; long runs = background shells with monitoring; watchdog active in every model-touching run; never a second model-holding process; on the 2nd failed fix of any bug — stop patching, bring it to the brain for first-principles review; MPS↔CUDA outputs are never bit-compared.

## Post-v1 (non-blocking; deferred by design — mind the 40 min/day ZeroGPU quota, E2)
Both hard gates are met and the Space is verified 5/5. Deferred, in rough priority:

**Feature / polish gaps:**
1. **`subtalker_dosample` UI exposure** — the one known gap against the "every feature" mandate; smallest possible scope; first in line for any follow-up session.
2. Dynamic-`duration` callable (queue-priority polish; does NOT affect quota — quota = effective runtime × size multiplier, not declared duration).
3. int16 mic-input scaling consistency (scale by dtype, don't peak-normalize).
4. Residency hysteresis (re-expand < 55 GB) + status-strip eviction visibility.

**Verification checks (do cheaply, ≤ 1 GPU-min each):**
5. **D8 A/B regression** — fixed-seed adapter-set vs adapter-empty pair in the Space `gradio_client` suite; assert they differ.
6. **Worker-persistence / quota math** — 3-consecutive-same-mode burst; if #2/#3 aren't faster than #1, models reload per call → re-cost GPU-seconds/request vs the 40 min/day budget; state the true behavior in DESIGN once measured.
7. **In-fork device log** — emit `next(model.parameters()).device` inside a `@spaces.GPU` handler; confirm `cuda:0`.
