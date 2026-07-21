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
| D7 | ZeroGPU large slice; sdpa; no compile; dynamic duration 30–120 s; torch ladder 2.13→2.11.0 | 48 GB ≫ 16 GB need; wheel risk zero; quota priority | 2026-07-21 |
| M1 | Memory: vm_stat gauge; warn 72 / abort 76; adaptive degrade >65; fp32 ⇒ single-resident | 55 GB operator baseline; 80 GB OS-crash gate | 2026-07-21 |
| E1 | Escalation: only real-money items (ZeroGPU credits, persistent storage) go to operator | Everything else delegated to brain-proxy | 2026-07-21 |
