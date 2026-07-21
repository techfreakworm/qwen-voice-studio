# Decision Log

> Authored by the project brain (`qwen-voice-brain`) acting as operator proxy. Appended over time.

| ID | Decision | Rationale (one line) | Date |
|---|---|---|---|
| D1 | All-3-resident both platforms; ZeroGPU module-level cuda; `fork_move` fallback flag | Official ZeroGPU guidance + 69.6 GB measured local peak | 2026-07-21 |
| D2 | Generate-then-play; no DIY streaming | Wrapper exposes none; chunked decode = fidelity risk | 2026-07-21 |
| D3 | LoRA: Base talker only, unmerged PEFT, toggle not merge | Instant A/B; merge kills toggle; operator pre-scoped Base | 2026-07-21 |
| D4 | 5 tabs; LoRA Lab = management-only; Design→Clone bridge | No duplicated generation UI; library unifies voice assets | 2026-07-21 |
| D5 | Local `device_map="mps"`+FALLBACK; Space CPU→module `.to(cuda)`; DevicePolicy seam | Empirically proven local; docs-mandated Space path | 2026-07-21 |
| D6 | Python 3.12 | ZeroGPU parity (3.12.12); wheel maturity (onnxruntime) | 2026-07-21 |
| D7 | ZeroGPU large slice; **sdpa everywhere (no flash-attn auto-select)**; no compile; **torch==2.11.0 both platforms** | Patch-exact ZeroGPU list (2.9.0 excluded but 2.9.1 in ⇒ version-specific patches); unlisted torch = silent GPU-attach breakage, not a clean rejection | 2026-07-21 |
| M1 | Memory: vm_stat gauge; warn 72 / abort 76; adaptive degrade; fp32 ⇒ single-resident | 55 GB operator baseline; 80 GB OS-crash gate | 2026-07-21 |
| M2 | Request-time adaptive residency (`QVS_RESIDENCY=auto`: auto-evict >60 GB, **darwin-only**, adapter-pinned) | Operator baseline fluctuates 45–58 GB; startup-only check insufficient (hit 77.3 GB abort) | 2026-07-21 |
| D8 | LoRA is per-request: adapter applied inside each generation fork, unloaded after; Clone gets an optional adapter field; Lab = inspect + self-contained quick-test. Supersedes D3's persistent toggle in the shipped app | ZeroGPU forks don't persist model mutations; parent-side attach fails ("Low-level CUDA init" under emulation); stateless = recycle-proof + kills global mutable state; ~1–2 s cost negligible | 2026-07-21 |
| D9 | ZeroGPU: models warmed **resident in the PARENT** (background thread at startup) + **fork_move** (CPU→GPU per request) — lazy-load-in-fork reloaded ~4.5 GB every call (~38 s, GPU idle). No module-level preload *before* launch() (startup overrun). RAM watchdog darwin-only (psutil misreports in container); bind 0.0.0.0 + ssr_mode=False; torch==2.11.0; packages.txt sox/ffmpeg; LFS binaries; deprecation warnings filtered | Empirical — each item hit during T9; burst confirmed cuda:0 @ ~6 s/req | 2026-07-21 |
| E1 | Escalation: only real-money items (ZeroGPU credit top-ups, persistent storage) go to operator; **stay within PRO's included 40 min/day ZeroGPU quota** | Everything else delegated to brain-proxy | 2026-07-21 |
| E2 | Stay within PRO's included 40 min/day ZeroGPU quota; paid credits forbidden without operator approval; Space verification runs batched + minimal; NO automated/scheduled Space tests (silent quota burn) | Operator hard-gate 2026-07-21; observed full-suite cost ≈ small single-digit GPU-minutes | 2026-07-21 |
