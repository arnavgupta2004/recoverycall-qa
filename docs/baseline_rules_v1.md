# Baseline: rules v1 on held-out v1 (spent)

**Rules v1 frozen 2026-10-21: conduct recall 5/17, 0 compliant false flags.**

| | value |
|---|---|
| Rules | `src/concern_rules.py` frozen at commit `12cb3b1` (2026-10-21) |
| Evaluated at | commit `fdbcf2b`, text only (no TTS/STT), run once |
| Held-out v1 | 20 externally written synthetic scripts (Hindi 7, Kannada 6, Hinglish 7); 10 compliant (all hard negatives), 10 concerning |
| Labels | fixed by `prepare_labels.py` before the classifier was run (log in `data/dataset_manifest.json`, `label_changes`) |
| Full report | `reports/classifier_text_held_out_2026-10-21.md` |

## Results (call-level)

| category | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| intimidation_or_threats | 3 | 0 | 5 | 1.00 | 0.38 |
| abusive_language | 2 | 0 | 3 | 1.00 | 0.40 |
| third_party_contact_threat | 0 | 1 | 4 | 0.00 | 0.00 |
| **conduct total** | **5** | **1** | **12** | 0.83 | **0.29 (5/17)** |
| authority_or_legal_action_claim (verify flag) | 4 | 0 | 2 | 1.00 | 0.67 |

- False flags on compliant calls (conduct categories): **0/10**; on hard negatives: 0/10.
- Authority misses were both lawful legal-notice mentions in compliant calls (no "legal notice" phrase in the rules).
- Conduct misses were phrasings absent from the dev-built rules (employer/HR letters, family phone calls, "cheater", house visits in Hinglish, most Kannada concerns).

## Status

Held-out v1 is **spent**: it was merged into the dev set as `split=dev_v2` on 2026-10-21 and is now used for development. Any later held-out number must come from a new, externally written held-out v2. These figures are a smoke test on 20 synthetic scripts, not a benchmark.
