# RecoveryCall QA — Dataset Cost Estimate

**Date:** 2026-10-21  
**Status:** Pre-run estimate. Confirm available credits cover the totals below before starting the T3 pipeline run.

---

## Assumptions

| Parameter | Value | Source |
|---|---|---|
| TTS model | Timbre v2.5 | `data/scripts_dev.json` |
| Average turns per script | 6.5 | Counted across 40 dev scripts (range 5–7) |
| Average TTS seconds per turn | ~6 s | Spike: scenario turns averaged 4–9 s |
| VAD segment length cap | 12 s | `src/vad.py`; product pipeline |
| Average segments per call | ~3 | Spike: 24–29 s clean audio → 2–3 segments |
| STT recovery overhead | +35% | `outputs/experiment/report.md`: 36 recovery / 104 base |
| Held-out scripts (written after rule freeze) | ~20 | PLAN.md T1 |

---

## Dev set (40 scripts)

| Item | Count | Basis |
|---|---|---|
| Scripts | 40 | `data/scripts_dev.json` |
| Total turns | ~260 | 40 × 6.5 avg |
| TTS calls (first run; cached thereafter) | **~260** | 1 per turn |
| Estimated audio duration | ~28 min | 260 turns × 6.5 s |
| VAD segments (base) | ~120 | 40 calls × ~3 segments |
| STT calls (base) | **~120** | 1 per segment (clean, verbatim) |
| STT recovery calls (+35%) | **~42** | 120 × 0.35 |
| **Total STT (dev)** | **~162** | base + recovery |

---

## Held-out set (~20 scripts, written after rule freeze)

| Item | Count | Basis |
|---|---|---|
| Scripts | ~20 | PLAN.md |
| Total turns | ~130 | 20 × 6.5 avg |
| TTS calls | **~130** | 1 per turn |
| VAD segments (base) | ~60 | 20 × ~3 |
| STT calls (base) | **~60** | 1 per segment |
| STT recovery calls (+35%) | **~21** | 60 × 0.35 |
| **Total STT (held-out)** | **~81** | base + recovery |

---

## Spike scenarios already cached (do not re-synthesize)

The 3 original spike scenarios (A, B, C = 21 turns) are TTS-cached in `cache/tts/`.  
The experiment's 140 STT calls are also cached in `outputs/stt_cache/`.  
These will not be re-billed.

---

## Full dataset totals

| | TTS calls | STT calls |
|---|---|---|
| Dev set | ~260 | ~162 |
| Held-out set | ~130 | ~81 |
| **Total** | **~390** | **~243** |

The PLAN.md estimate of ~420 TTS and ~200–250 STT is consistent; the slight difference is rounding.

---

## ⚠ Credit gate

**Before running `run_eval.py` (T3 pipeline), confirm:**

1. Log in to the Gnani dashboard and check available TTS and STT credit balance.
2. Confirm the balance covers at least **400 TTS calls** and **250 STT calls** with a comfortable margin.
3. Record the gate decision in `outputs/eval/run_log.md`:
   - Date checked
   - Remaining balance (TTS and STT units or rupees)
   - Decision: proceed / reduce dataset / abort

If the balance is insufficient:
- Reduce to 20 dev + 10 held-out scripts (see PLAN.md "What to cut first").
- Re-estimate: ~130 TTS + ~81 STT for the reduced set.

---

## Notes

- TTS calls are one-time per turn (cached by text+voice+language+model hash). Re-running the pipeline does not re-bill them.
- STT calls are cached per audio SHA-256 + language code. A re-run retries only failed/non-cached segments.
- The experiment's noise variants (SNR sweep) are **not** included in this estimate; evaluation runs clean audio only unless explicitly enabled.
- Noise variant evaluation would add ~4× the base STT count; add this only if time and credits allow.
