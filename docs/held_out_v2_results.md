# Held-out v2 results: rules vs LLM vs hybrid (text only)

**Run once on 2026-10-21. Held-out v2 is now spent.** Do not tune prompts, rules or labels on it; a further headline number needs a new externally written set.

## Summary

| method | conduct TP/FP/FN | conduct recall | compliant false flags | hard-negative false flags | authority (verify) TP/FP/FN |
|---|---|---|---|---|---|
| rules v1 (frozen) | 0/0/11 | 0/11 | 0/12 | 0/12 | 3/0/4 |
| LLM (gemini-3.5-flash-lite) | 11/0/0 | 11/11 | 0/12 | 0/12 | 6/0/1 |
| hybrid (rules ∪ verified LLM) | 11/0/0 | 11/11 | 0/12 | 0/12 | 6/0/1 |

- **Quote-verifier rejections:** 0 of 19 LLM outputs. Every LLM quote was verbatim, so there were no paraphrases.
- **The only LLM/hybrid miss** is the authority verify flag on RC001 (Hindi, compliant): “यदि आगे कानूनी नोटिस जारी होता है…”. It's a conditional, lawful legal-notice mention.
- **The rules-only authority hits** were RC002, RC015 and RC020. The rules missed every other authority mention, and every conduct concern.
- **Run:** 20 LLM API requests, 0 retries, 0 non-200 responses, 95 s. Nothing was served from cache.

## How to read this

- **Small set:** 20 synthetic scripts, 11 conduct concerns in 8 concerning calls, all on text only (no TTS or STT). One concern moves recall by about 9 points. This is a smoke test, not a benchmark.
- **No Hindi concerning calls.** All 7 Hindi scripts are compliant, so Hindi conduct recall is not measured here; Hindi contributes only false-flag and authority evidence.
- **Kannada** results (4 concerning, 2 compliant) are provisional pending native-speaker review.
- **The "en-IN" scripts are plain English, not Hinglish.** Rules v1 were built on Hinglish and Devanagari phrasing, which explains part of their 0/11. They also missed all four Kannada concerns.
- **The LLM prompt was not tuned on this set.** Its definitions were written after reading held-out v1 (now dev_v2) and before this set existed, and it is unchanged since commit `6eaf7b4`.
- **Call-level scoring:** a category counts as found if any turn of the call was flagged with it.

## Provenance

| item | value |
|---|---|
| Delivered file | `data/held_out_source/held_out_v2.json` (unchanged) |
| Converted set and labels | `data/scripts_held_out.json`, `data/dataset_manifest.json` (split=held_out), committed in `aefed59` **before** any classifier or LLM ran on it |
| Rules | `src/concern_rules.py`, frozen at `12cb3b1` |
| Classifier, hybrid, LLM prompt | unchanged since `6eaf7b4` (prompt version 2026-10-21.v1); `009f171` added only API-attempt counters |
| Model | `gemini-3.5-flash-lite`, Gemini Interactions API, temperature 0 (no deprecation warning observed) |
| Command | `python eval_classifier_text.py --split held_out` |

## Structural fixes and label changes (made by reading the scripts, before evaluation)

**Structural (fixed):**
- Speaker `borrower` renamed to `customer`.
- Turn indices added to the concern spans; each span is a unique exact substring of one turn.
- TTS, STT and voice fields added using the dev per-language defaults.
- Compliant calls listed `authority_or_legal_action_claim` in `concern_types` with no span. Our schema keeps a compliant call's `concern_types` empty and records lawful mentions as `authority_mention` plus spans.

**Label changes** (also logged per call in `label_changes`):
- **Authority spans added** for agent mentions of a legal notice, police or arrest:
  - RC001: legal notice.
  - RC002: legal notice, plus the agent's “नोटिस और पुलिस कार्रवाई अलग बातें हैं”.
  - RC008: ಕಾನೂನು ನೋಟಿಸ್.
  - RC015: legal notice, plus “not the same as an immediate arrest”.
- **RC003: delivered authority label removed.** The agent says only “कानूनी उपाय” (legal remedies); “अदालत” (court) is the borrower's question.
- **Third-party spans added** under the multi-label definition (showing neighbours or others = third-party):
  - RC013: “ನೆರೆಹೊರೆಯವರೂ ಹೊರಗೆ ಬರ್ತಾರೆ”.
  - RC020: “make sure everyone understands what kind of borrower you are.”

  Both calls already had that type at call level.

---

## Full generated report

Calls: 20 (held_out: 20; en-IN: 7, hi-IN: 7, kn-IN: 6). LLM backend: gemini:gemini-3.5-flash-lite. Per-call scoring: a category counts as flagged if any segment produced a flag of that type.

### Per-category TP / FP / FN

| category | rules TP/FP/FN | rules P | rules R | llm TP/FP/FN | llm P | llm R | hybrid TP/FP/FN | hybrid P | hybrid R |
|---|---|---|---|---|---|---|---|---|---|
| `intimidation_or_threats` | 0/0/4 | — | 0.00 | 4/0/0 | 1.00 | 1.00 | 4/0/0 | 1.00 | 1.00 |
| `abusive_language` | 0/0/2 | — | 0.00 | 2/0/0 | 1.00 | 1.00 | 2/0/0 | 1.00 | 1.00 |
| `third_party_contact_threat` | 0/0/5 | — | 0.00 | 5/0/0 | 1.00 | 1.00 | 5/0/0 | 1.00 | 1.00 |
| `authority_or_legal_action_claim` (verify flag) | 3/0/4 | 1.00 | 0.43 | 6/0/1 | 1.00 | 0.86 | 6/0/1 | 1.00 | 0.86 |
| **conduct total** | **0/0/11** | — | 0.00 | **11/0/0** | 1.00 | 1.00 | **11/0/0** | 1.00 | 1.00 |

### False flags on compliant calls (conduct categories only)

| | rules | llm | hybrid |
|---|---|---|---|
| compliant calls with a conduct flag | 0/12 | 0/12 | 0/12 |
| hard-negative calls with a conduct flag | 0/12 | 0/12 | 0/12 |
| flags whose quote is not verbatim (after verification) | 0 | 0 | 0 |

### Quote-verifier rejections (LLM)

0 of 19 LLM outputs rejected (none). Responses from cache: 0/20.


Hybrid flag sources: {'rules': 3, 'llm': 18, 'rules+llm': 1} (rules+llm = both found an overlapping span of the same type).

### rules: every miss and false flag

- MISS **RC001** (hi-IN, held_out, compliant, hard negative) `authority_or_legal_action_claim` — t6 “यदि आगे कानूनी नोटिस जारी होता है, तो उसकी जानकारी लिखित रूप में दी जाएगी।”
- MISS **RC008** (kn-IN, held_out, compliant, hard negative, needs native review) `authority_or_legal_action_claim` — t4 “ಬಾಕಿ ಮುಂದುವರಿದರೆ ಒಪ್ಪಂದದ ಪ್ರಕಾರ ಕಾನೂನು ನೋಟಿಸ್ ಕಳುಹಿಸುವ ಪ್ರಕ್ರಿಯೆ ಇರಬಹುದು.”
- MISS **RC010** (kn-IN, held_out, concerning, needs native review) `intimidation_or_threats` — t4 “ನಾಳೆಗೂ ಹಣ ಬರಲಿಲ್ಲ ಅಂದ್ರೆ, ನಮ್ಮ ಹುಡುಗರು ನಿಮ್ಮ ಮನೆ ಹತ್ತಿರ ಬಂದು ಮಾತಾಡ್ತಾರೆ, ಆಮೇಲೆ ನಿಮಗೇ ಕಷ್ಟ.”
- MISS **RC011** (kn-IN, held_out, concerning, needs native review) `abusive_language` — t4 “ಸುಳ್ಳುಗಾರನ ಹಾಗೆ ಮಾತಾಡಬೇಡಿ, ಮೊದಲು ಜವಾಬ್ದಾರಿ ಕಲಿಯಿರಿ.”
- MISS **RC012** (kn-IN, held_out, concerning, needs native review) `third_party_contact_threat` — t4 “ಮತ್ತೆ ಮುಂದೂಡಿದ್ರೆ ನಿಮ್ಮ ಆಫೀಸ್‌ನವರಿಗೆ ಈ ಸಾಲದ ವಿಷಯ ಹೇಳಬೇಕಾಗುತ್ತೆ.”
- MISS **RC013** (kn-IN, held_out, concerning, needs native review) `intimidation_or_threats` — t4 “ನಮ್ಮವರು ನಿಮ್ಮ ಮನೆಗೆ ಬಂದಾಗ ನೆರೆಹೊರೆಯವರೂ ಹೊರಗೆ ಬರ್ತಾರೆ, ಆಗ ನಿಮಗೆ ಗೊತ್ತಾಗುತ್ತೆ.”
- MISS **RC013** (kn-IN, held_out, concerning, needs native review) `third_party_contact_threat` — t6 “ಇಲ್ಲದಿದ್ದರೆ ಎಲ್ಲರಿಗೂ ವಿಷಯ ಗೊತ್ತಾಗುತ್ತೆ.”; t4 “ನೆರೆಹೊರೆಯವರೂ ಹೊರಗೆ ಬರ್ತಾರೆ”
- MISS **RC017** (en-IN, held_out, concerning) `intimidation_or_threats` — t4 “I'm just saying, you don't want to find out what happens when our people come looking for you.”
- MISS **RC017** (en-IN, held_out, concerning) `authority_or_legal_action_claim` — t2 “The court will hear about this if you keep wasting our time.”
- MISS **RC018** (en-IN, held_out, concerning) `abusive_language` — t4 “Stop acting like an idiot and answer the question.”
- MISS **RC018** (en-IN, held_out, concerning) `third_party_contact_threat` — t6 “If you keep delaying, perhaps your employer can explain why you ignore your responsibilities.”
- MISS **RC019** (en-IN, held_out, concerning) `third_party_contact_threat` — t6 “And if you still don't respond, don't be surprised if your family starts getting questions about the money.”
- MISS **RC019** (en-IN, held_out, concerning) `authority_or_legal_action_claim` — t4 “If Friday passes, we can send a legal notice and take the matter further.”
- MISS **RC020** (en-IN, held_out, concerning) `intimidation_or_threats` — t4 “Then you know we can come there and make sure everyone understands what kind of borrower you are.”
- MISS **RC020** (en-IN, held_out, concerning) `third_party_contact_threat` — t6 “I'm only saying that if you don't cooperate, the whole neighbourhood might hear about this.”; t4 “make sure everyone understands what kind of borrower you are.”

### llm: every miss and false flag

- MISS **RC001** (hi-IN, held_out, compliant, hard negative) `authority_or_legal_action_claim` — t6 “यदि आगे कानूनी नोटिस जारी होता है, तो उसकी जानकारी लिखित रूप में दी जाएगी।”

### hybrid: every miss and false flag

- MISS **RC001** (hi-IN, held_out, compliant, hard negative) `authority_or_legal_action_claim` — t6 “यदि आगे कानूनी नोटिस जारी होता है, तो उसकी जानकारी लिखित रूप में दी जाएगी।”

### Limitations

- Text only: no TTS or STT. Transcribed audio will be harder (spelling variants, omissions).
- Call-level scoring: a category flagged on the wrong segment of a call that has it still counts as a TP.
- The taggers see every turn, including the borrower's; there is no speaker separation at runtime.
- Small synthetic set; Kannada scripts await native review. Treat every number as a handful of calls.
