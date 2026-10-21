# RecoveryCall QA — Product Plan
**Submission deadline:** 9 November 2026  
**Builder:** solo  
**Schedule constraint:** light work until 30 Oct (classifier + scripts, no API); full build 31 Oct – 8 Nov (UI first); 9–10 Nov buffer.

---

## Top-Level Overview

The API spike (complete) confirmed that Timbre TTS and Prisma STT work for Hinglish (en-IN), Hindi (hi-IN), and Kannada (kn-IN); that VAD + coverage-check + recovery reduces silent omissions; and that short-turn drops are the hardest remaining problem. The product turns that validated pipeline into a single-page audit tool:

1. Upload a fictional recovery call audio file.
2. The user selects the call language (hi-IN, kn-IN, or en-IN for code-mixed Hinglish); this drives STT and the Timbre summary. No auto-detection.
3. The backend runs the VAD → Prisma STT → coverage check → flag/re-split low-density segments pipeline (VAD segments ≤ 12 s throughout); untranscribed speech regions are surfaced rather than hidden.
4. A rules-based classifier scans the transcript for conduct concerns (intimidation, abusive language, third-party contact threats, authority/legal action claims). Every flag is tied to a quoted transcript span and the audio segment that contains it.
5. A single-page UI shows the transcript with highlighted flagged spans, lets the reviewer click to play each flagged segment's audio, and displays coverage warnings.
6. Timbre generates a short spoken summary using a per-language template filled from the flags.
7. An audit report can be downloaded.

**No LLM verdict, no risk score, no legal conclusion.** Every concern flag reads "potential conduct concern — human review required". Every regulatory reference is marked "to verify against the RBI circular".

**Public demo safety:** bundled sample calls run by default (labelled "cached example" in the UI); live uploads are rate-limited, capped at 2 minutes, and guarded by an optional passcode. A notice reads "fictional or test audio only; uploads are not stored". The API key is server-side only; nothing is persisted. The recorded 60-second demo uses a live Prisma STT run and a live Timbre summary, not cached results. The Timbre summary for bundled samples is cached per sample; live summary generation is rate-limited.

---

## Sub-Tasks

---

### T1 — Expanded dataset: fictional call scripts

**Status:** `[ ] pending`

**Intent**  
The spike has 3 calls (one per language). The evaluation needs ~40 dev scripts (used to build and tune the classifier rules) plus ~20 held-out scripts (written after the rules are frozen, used as the headline evaluation). This split prevents overfitting the rules to the test set. More calls per language matters more than more noise conditions. Scripts are LLM-drafted and human-reviewed; Kannada scripts are flagged for native speaker review before use.

**Expected Outcomes**
- `data/scripts_dev.json` — ~40 dev scripts: ~20 compliant (including hard negatives) + ~20 concerning, across Hindi, Kannada, Hinglish.
- `data/scripts_held_out.json` — ~20 held-out scripts: ~10 compliant + ~10 concerning, written after rules are frozen in T2.
- Each concerning call includes at least one of the four concern categories (see T2); each compliant call may include hard negatives (firm but lawful reminders).
- Each script is 5–10 turns, amounts written as spoken words, same schema as the existing `data/scripts.json`.
- `data/dataset_manifest.json` records per-call: id, split (dev/held-out), language, compliance label, concern types present, hard-negative flag where relevant, Kannada-native-review status.
- `data/cost_estimate.md` — estimated TTS and STT call counts and credit cost for the full dataset (~420 TTS, ~200–250 STT calls estimated); includes a gate instruction: do not proceed to T3 pipeline run without confirming available credits cover this estimate.
- Kannada scripts are marked `"needs_native_review": true` in the manifest until reviewed.

**Todo List**
1. Read the existing `data/scripts.json` schema and the spike's `data/` conventions.
2. LLM-draft ~20 dev compliant scripts (polite reminders, payment plans, call closings); include ≥ 5 hard negatives per language (firm tone, legal but no threats). Distribute across Hindi, Kannada, Hinglish. Human-review all; flag Kannada for native review.
3. LLM-draft ~20 dev concerning scripts, covering each of the four categories (T2) at least 3 times each, spread across the three languages. Human-review all; flag Kannada.
4. Write `data/scripts_dev.json` and `data/dataset_manifest.json` (dev entries only).
5. Estimate TTS cost: count turns per script → TTS calls (~420 total). Estimate STT cost: estimate seconds of audio per script → STT calls at ≤ 12 s VAD segments (~200–250 total, plus ~35% recovery overhead). Write `data/cost_estimate.md` with a **credit gate**: state the estimated cost and instruct: confirm credits are sufficient before running the T3 pipeline.
6. Run `python run_spike.py synthesize` on a 3-script sample to validate the schema is accepted without errors. (No STT calls.)
7. After T2 rules are frozen: LLM-draft + human-review the ~20 held-out scripts; write `data/scripts_held_out.json`; add held-out entries to `data/dataset_manifest.json`.

**Relevant Context**
- [`data/scripts.json`](data/scripts.json) — existing schema: `scenarios[].id`, `tts_language`, `stt_language_codes`, `primary_stt_language_code`, `voices`, `turns[].speaker`, `turns[].text`, `turns[].critical_spans`.
- [`src/gnani_client.py`](src/gnani_client.py) — voices per model, language codes; the client rejects unknown language codes.
- README §Data and §Expected call volume — cost basis.
- Spike result: Timbre TTS is cached by turn hash; adding new scripts only synthesizes new turns.
- **Hard negative definition:** agent uses phrases like "we will need to escalate" or "this will affect your CIBIL score" without threatening physical contact, contacting third parties, or making false legal claims.

---

### T2 — Rules-based conduct-concern classifier

**Status:** `[ ] pending`

**Intent**  
A lightweight, deterministic classifier that scans a verbatim transcript for four concern categories. Rules-first means: keyword/phrase lists with script-aware matching; no LLM, no model weights, no probability score. The output is a list of flagged spans, each with a category, a quoted transcript excerpt, and a segment reference. Rules are built against the dev set (T1 dev scripts) and then frozen before the held-out scripts are written.

**Expected Outcomes**
- `src/classifier.py` exports a `classify(transcript_segments, language)` function.
- Input: list of `{"segment_id", "text", "start_s", "end_s"}` dicts (from the pipeline).
- Output: list of `{"concern_type", "quote", "segment_id", "start_s", "end_s", "label", "regulatory_note"}` dicts.
- Four concern types:
  - `intimidation_or_threats` — physical threats, agents visiting, sending men to the house.
  - `abusive_language` — slurs, profanity, derogatory terms (script-specific word lists, scoped to clear-cut cases).
  - `third_party_contact_threat` — threatening to contact family, employer, neighbours, or any named third party.
  - `authority_or_legal_action_claim` — claiming police involvement, arrest warrant, court summons, or any official legal process. Label for this type is exactly `"claim of police, arrest or court action — verify"` (the transcript cannot show whether such a process exists).
- Every flag's `label` for the first three types is exactly `"potential conduct concern — human review required"`.
- Every flag's `regulatory_note` is exactly `"to verify against the RBI circular"`.
- Out-of-hours and repeated-call checks are **not** implemented here; they are metadata checks added in T4.
- False-flag rate on compliant and hard-negative scripts is reported honestly in T3; there is no pass/fail gate on that rate at this stage.
- A unit test in `tests/` checks: (a) each concern type fires on a synthetic one-turn transcript; (b) at least three hard-negative phrases do not fire; (c) every flag's quote is a substring of the input text; (d) empty transcript returns empty list.

**Todo List**
1. Define keyword/phrase lists for each of the four categories in three scripts: Devanagari (hi-IN), Kannada (kn-IN), Latin (en-IN / Hinglish). Store in `src/concern_rules.py` or a sibling JSON file, not inline in `classifier.py`. Scope `abusive_language` to clear-cut slurs only (a short list), not broad profanity.
2. Implement `classify()` in `src/classifier.py`. For each segment's text: apply script detection (reuse the logic already in `src/metrics.py`), select the matching rule set, scan for keyword matches. Return a deduplicated list of flags with the narrowest matching substring as `quote`.
3. Apply the same Unicode normalization used in `src/metrics.py` before matching, so nukta/ZWNJ variants and half-formed characters don't break rules.
4. Iterate rules against the dev set until confident, then mark rules as frozen (a comment in `concern_rules.py`: `# RULES FROZEN — do not edit after <date>`).
5. Write unit tests covering the four checks in Expected Outcomes above.
6. Run tests: `python -m pytest tests/ -q`.
7. After tests pass and rules are frozen: write `data/scripts_held_out.json` (T1 step 7).

**Relevant Context**
- [`src/metrics.py`](src/metrics.py) — Unicode NFC, Cf removal, script detection logic; reuse directly.
- [`src/vad.py`](src/vad.py) — segment dict shape the classifier will consume.
- Spike concern examples from `data/scripts.json`: scenario B turn 4 (physical visit threat), turn 6 (shaming threat); scenario C turn 4 (visit threat), turn 5 (family contact).
- **No legal verdict:** classifier must never output "illegal", "violation", or "criminal".
- **Short-turn limitation:** turns of 2–3 words (e.g., "जी, बताइए।") will not contain concern language; this is fine and need not be flagged or warned.

---

### T3 — Full dataset pipeline run + evaluation

**Status:** `[ ] pending`

**Intent**  
Run the complete pipeline (TTS → VAD ≤ 12 s segments → STT → coverage check → classifier) over all dev + held-out scripts and evaluate against `data/dataset_manifest.json` ground truth. The **headline numbers are held-out only**. Dev results are reported separately as context. State the short-turn limitation explicitly. A credit check must pass before the full run begins.

**Expected Outcomes**
- **Credit gate:** before any API calls, check `data/cost_estimate.md` and confirm available credits cover ~420 TTS + ~200–250 STT + ~35% recovery overhead. Document the gate outcome in `outputs/eval/run_log.md`.
- `outputs/eval/results.json` — per-call, per-segment, per-flag record (dev + held-out, labelled by split).
- `outputs/eval/report.md` — evaluation report with:
  - **Headline section (held-out only):** per-concern precision and recall (tp/fp/fn counts); false-flag rate on compliant calls; false-flag rate on hard-negative calls specifically.
  - Dev section: same metrics, clearly marked "dev set — rules were tuned on this data; do not use as headline".
  - Quote-existence check: % of flags whose `quote` is a substring of the segment transcript (both splits).
  - Omission/recovery stats (detection recall, recovery rate, silent omissions) across all calls.
  - A clearly labelled section "Limitations and known gaps": (a) short-turn limitation (2–3 word turns are undetectable by density check); (b) synthetic TTS audio only; (c) no real-noise evaluation; (d) classifier covers only the four listed categories; (e) Kannada scripts pending native review.
  - Total TTS and STT calls made and estimated credit cost.

**Todo List**
1. Check `data/cost_estimate.md`; confirm credits before proceeding. Log the decision in `outputs/eval/run_log.md`.
2. Extend `run_spike.py` or write a new `run_eval.py` that iterates over all scripts in `data/scripts_dev.json` and `data/scripts_held_out.json`, runs synthesize → VAD (≤ 12 s segments) → transcribe → coverage check → classify, and writes `outputs/eval/results.json`.
3. Write an evaluation scorer that reads `results.json` + `dataset_manifest.json` and computes the metrics above, split by dev/held-out.
4. Add the quote-existence assertion as an automated check (warn in the report if any flag's quote is not found in the segment transcript; do not silently pass).
5. Write `outputs/eval/report.md` with the structure above. Include the "Limitations and known gaps" section.
6. Run the pipeline and verify the report generates without assertion failures.

**Relevant Context**
- [`run_spike.py`](run_spike.py) — existing pipeline runner; extend or clone.
- [`experiment.py`](experiment.py) — omission/recovery evaluation pattern to reuse.
- [`src/metrics.py`](src/metrics.py) — span matching logic; reuse for quote-existence check.
- [`src/vad.py`](src/vad.py) — VAD segments ≤ 12 s; this is the only segmentation path in the product (the 28 s manifest-chunk path is spike-only).
- Known short-turn issue: `C_hindi_family_threat` turn 1 ("जी, बताइए।") is undetectable by density; document, don't fix.
- Recovery adds ~35% extra STT calls (from experiment); already factored into `cost_estimate.md`.

---

### T4 — Single-page audit UI + public deployment

**Status:** `[ ] pending`

**Intent**  
A minimal single-page web app that runs the full pipeline server-side and renders the result. Deployed free (Hugging Face Spaces / Gradio) with bundled sample calls as the default experience, and live uploads guarded for public safety. Must be live by 2 Nov so the demo can be recorded 3–4 Nov.

**Expected Outcomes**
- **Bundled sample calls:** the app ships with 4 selectable examples: spike scenarios A (Hinglish compliant), B (Kannada intimidation), C (Hindi family threat), and D (Hindi 10 dB noisy, from the experiment — scenario C at SNR 10 dB seed 0, where the coverage warning fires). Bundled samples use cached transcripts and pre-run classifier output; the UI labels them "(cached example)". The Timbre spoken summary for bundled samples is cached per sample. Live summary generation is rate-limited.
- **Language selector:** user chooses hi-IN, kn-IN, or en-IN (code-mixed) before running; this drives STT language code and the Timbre summary voice. No auto-detection.
- **Upload path (rate-limited):** accepts WAV or MP3, max 2 minutes audio; rate-limited (e.g., 5 uploads per hour per IP); guarded by an optional passcode (env var `UPLOAD_PASSCODE`; if unset, upload is disabled). A banner reads: "fictional or test audio only; uploads are not stored". Nothing is written to disk after the response is sent; the API key is never exposed client-side.
- **Transcript panel:** full transcript with flagged spans highlighted inline. Each highlighted span shows the concern type and label on hover/click.
- **Audio playback:** clicking a flagged span seeks the full-call `<audio>` element to the segment's start time (no server-side clip trimming needed).
- **Coverage warnings:** untranscribed speech segments show a labelled warning block with the time range and "untranscribed speech — human review required".
- **Timbre spoken summary:** a "Generate spoken summary" button calls Timbre TTS using a template filled from the flags (one sentence per concern found, plus an opening and closing line), in the user-selected language. Result plays inline; the text is also shown.
- **Metadata check (out-of-hours):** if the uploaded filename matches `YYYYMMDD_HHMMSS`, check whether the time is outside 08:00–19:00 IST and show a warning "call outside permitted hours — to verify against the RBI circular". If no timestamp in filename, skip silently.
- **Download:** exports `audit_<filename>_<timestamp>.json` with all flags, quotes, segment times, and the transcript.
- All UI labels follow the wording constraints; no legal verdict language.
- Hosted on a free tier; URL is functional by 2 Nov 2026.

**Todo List**
1. Confirm Hugging Face Spaces (Gradio) supports: audio upload + playback, custom HTML blocks, server-side env vars, rate-limiting middleware.
2. Write `app.py` wiring: language selector → upload or sample selection → pipeline → classify → return transcript JSON + flagged spans + audio path.
3. Build the UI: transcript panel with inline highlighted spans; audio element with seek-on-click; coverage warning blocks; download button.
4. Implement upload safety: 2-minute cap (reject longer files at the server), rate limiter, passcode check, no-persist guarantee, banner text.
5. Implement the Timbre spoken summary: per-language template (3 languages × ~5 template sentences); call `POST /api/v1/tts/inference`; play inline.
6. Implement the out-of-hours metadata check (filename pattern only).
7. Deploy to Hugging Face Spaces; verify end-to-end with each of the 3 bundled sample calls. Confirm the URL is live by 2 Nov.

**Relevant Context**
- [`src/gnani_client.py`](src/gnani_client.py) — TTS and STT client; reuse directly.
- [`src/vad.py`](src/vad.py) — segmentation ≤ 12 s; reuse.
- Experiment report: untranscribed speech records must be surfaced, not silenced.
- Timbre voices confirmed working: Poorvi (hi-en for Hinglish), Hemraj (hi-IN), Kavin (kn-IN). Summary voice matches user-selected language.
- **Cut first if behind:** (1) audio seek on click → fall back to showing segment timestamps only; (2) Timbre summary → text-only; (3) out-of-hours check → omit and note in report.

---

### T5 — Final report

**Status:** `[ ] pending`

**Intent**  
Produce the final written report for submission. It lives in `docs/` (not `outputs/`, which is git-ignored) so it is versioned and accessible. The headline numbers come from the held-out evaluation. The omission chart from the experiment is copied into `docs/`.

**Expected Outcomes**
- `docs/final_report.md` containing:
  - Pipeline overview (one paragraph).
  - **Headline:** held-out per-concern precision/recall table with tp/fp/fn.
  - Held-out false-flag rate on compliant calls and on hard-negative calls.
  - Dev results table, clearly labelled "dev set — rules were tuned on these scripts".
  - Quote-existence pass rate (both splits; explain any failures).
  - Omission/recovery summary table.
  - `docs/omission_vs_snr.png` — copied from `outputs/experiment/omission_vs_snr.png`.
  - Stated short-turn limitation: "turns of ≤ 3 words are below the minimum detectable length for the density check; this affects e.g. single-word acknowledgements (जी, बताइए।)."
  - Stated data limitation: "all audio is TTS-synthesized; absolute WER and concern-detection rates will be optimistic vs real calls."
  - Kannada caveat: "Kannada scripts are flagged for native speaker review; results should be treated as provisional until reviewed."
  - Regulatory disclaimer: "all regulatory references are marked 'to verify against the RBI circular'; this tool does not constitute legal or compliance advice."
  - UI demo link.
  - Dataset cost summary (TTS + STT calls, estimated credit cost).

**Todo List**
1. Create `docs/` directory.
2. Copy `outputs/experiment/omission_vs_snr.png` → `docs/omission_vs_snr.png`.
3. Run T3 evaluation to completion; pull tables into `docs/final_report.md`.
4. Write the contextualising prose sections.
5. Add all limitations and disclaimers exactly as specified above.
6. Add the UI link once T4 is deployed.
7. Proofread: confirm no flag in any example output uses legal verdict language ("illegal", "violation", "criminal").

**Relevant Context**
- `outputs/eval/report.md` (T3) — source of tables; `outputs/` is git-ignored so the final report must be in `docs/`.
- T4 deployment URL — needed for the demo link.
- `outputs/report.md` (spike) and `outputs/experiment/report.md` — short-turn limitation and known issues documented here; carry forward.

---

### T6 — 60-second demo storyboard and build-in-public posts

**Status:** `[ ] pending`

**Intent**  
A shot-by-shot storyboard for the 60-second screen-capture demo and three LinkedIn/X posts. The nominated public demo post goes live by 4 Nov, giving ~6 days of engagement before the 10 Nov cutoff. Post 1 (this week, omission-vs-SNR chart) is build-in-public. The evaluation results post is a follow-up, not the nominated demo post.

**Expected Outcomes**
- `docs/demo_storyboard.md` — shot-by-shot 60-second script (10–12 shots, ~5 s each):
  1. Landing page with bundled sample call selected (language: hi-IN).
  2. Click "Analyse" — progress indicator.
  3. Transcript appears; flagged span highlighted in amber.
  4. Click highlighted span → audio seeks and plays from that moment.
  5. Coverage warning shown for an untranscribed segment.
  6. "Generate spoken summary" → Timbre audio plays; summary text shown.
  7. Download audit report JSON shown in browser.
- `docs/posts.md` — 3 posts with dates and both X (≤ 280 chars) and LinkedIn versions:
  - **Post 1 (this week, ~22 Oct):** build-in-public — share the omission-vs-SNR chart from the spike experiment; explain what it shows (Prisma STT silent omissions in noisy Hindi calls, detected and partially recovered by VAD density check). No product claim yet.
  - **Post 2 / nominated demo post (3–4 Nov):** "Built RecoveryCall QA — a multilingual audit tool for debt-recovery calls in India." Share the demo URL and a screenshot of a flagged transcript. Note: rules-based, not an LLM, not a legal verdict, human review required.
  - **Post 3 (follow-up, ~8 Nov):** evaluation results — share one headline precision/recall number from the held-out set; link to the final report; note the Kannada caveat.
- Demo screen-capture recorded after T4 is live (3–4 Nov target). The demo uses a live Prisma STT run and a live Timbre summary (not cached).

**Todo List**
1. Write `docs/demo_storyboard.md` with the shot list above; annotate each shot with on-screen content and caption/voiceover.
2. Write `docs/posts.md` with all three posts. Keep wording factual; no legal verdict language; include the demo URL placeholder for post 2.
3. Publish post 1 this week using `outputs/experiment/omission_vs_snr.png`.
4. Record the 60-second screen-capture demo using the bundled sample calls (3–4 Nov, after T4 is live).
5. Publish post 2 (nominated demo post) by 4 Nov.
6. Publish post 3 after T5 is complete (~8 Nov).

**Relevant Context**
- T4 must be live by 2 Nov for the demo to be recordable on 3–4 Nov.
- `outputs/experiment/omission_vs_snr.png` — used in post 1; also copied to `docs/` in T5.
- Spike limitation: "all calls are fictional; no real recordings, no real personal data."

---

## Day-by-Day Schedule

### Light-work phase (21–30 Oct) — other deadlines are primary

| Date | Task |
|---|---|
| 21–22 Oct | T6: write and publish Post 1 (omission-vs-SNR chart, build-in-public) |
| 21–23 Oct | T1: draft dev compliant scripts (Hindi + Hinglish), review schema |
| 24–25 Oct | T1: draft dev compliant scripts (Kannada, flag for native review), hard negatives |
| 26–28 Oct | T1: draft dev concerning scripts (all four categories, all three languages) |
| 29–30 Oct | T1: write `dataset_manifest.json` (dev entries), `cost_estimate.md`; validate schema on 3-script sample |
| 28–30 Oct | T2: write `concern_rules.py` + implement `classify()`, unit tests pass (text-only, no API) |

All Oct work is scripts, rules, and planning. No API calls needed (schema validation uses `--mock`).

### Full-build phase (31 Oct – 8 Nov)

| Date | Task | Cut first if behind |
|---|---|---|
| 31 Oct | T2: iterate rules against dev set; freeze rules; write held-out scripts (T1 step 7) | — |
| 31 Oct – 2 Nov | T4: `app.py` + Gradio UI; bundled sample calls (A, B, C, D) working; deploy to HF Spaces | — |
| 3–4 Nov | T4: upload path + safety guards + Timbre summary + download; record live demo; publish Post 2 | Cut upload path; demo bundled calls only |
| 5 Nov | T3: credit gate check; run TTS synthesis on dev + held-out scripts | Reduce dataset if cost gate fails |
| 6 Nov | T3: run STT + coverage check + classify on full dataset; write `results.json` | Clean audio only; no noise variants |
| 7 Nov | T3: run evaluation scorer; write `outputs/eval/report.md` | — |
| 8 Nov | T5: write `docs/final_report.md`; T6: `docs/posts.md`; publish Post 3 | — |

### Buffer (9–10 Nov)

- 9 Nov: submission day. Buffer for deployment fixes, formatting, final proofread.
- 10 Nov: hard stop.

### What to cut first if behind

1. Upload path → demo bundled sample calls only; note "upload coming soon".
2. Audio seek on click → show segment timestamps only.
3. Timbre summary → text-only summary in UI.
4. Noise variants in T3 → evaluate on clean audio only.
5. Dataset size → 20 dev + 10 held-out if cost gate fails; record the reduction in the report.
6. Out-of-hours metadata check → omit entirely; note in report.

---

## Key Constraints and Conventions

- **No LLM, no risk score, no legal verdict.** Classifier is rules-based only.
- **All regulatory references** are marked "to verify against the RBI circular".
- **Flag labels:** first three concern types: "potential conduct concern — human review required"; `authority_or_legal_action_claim`: "claim of police, arrest or court action — verify".
- **Out-of-hours and repeated calls** are metadata checks, not transcript checks. Implement only if time allows (T4); cut without affecting headline results.
- **No code.** This is a planning document.
- **Free hosting only** for the UI.
- **All calls are fictional.** No real recordings, no real personal data, no voice cloning.
- **Short-turn limitation** must be stated in both T3 and T5 reports; do not suppress or work around it.
- **en-IN STT** is the correct code for code-mixed Hinglish (hi-IN returns Devanagari for Roman-script input, confirmed in spike).
- **VAD segments ≤ 12 s** throughout the product pipeline. The 28 s manifest-chunk path is spike-only and is not used here.
- **Timbre TTS** numbers must be written as spoken words in script inputs.
- **Dev/held-out split:** rules are frozen before held-out scripts are written. Headline results are held-out only.
- **False-flag rate** is reported honestly; there is no pass/fail gate on it.
- **Kannada scripts** are flagged for native speaker review; `docs/final_report.md` notes this caveat.
- **Credit gate:** do not start the T3 full pipeline run without confirming credits against `data/cost_estimate.md`.
- **Public demo safety:** bundled calls by default; uploads rate-limited + passcode-gated + max 2 min; nothing persisted; API key server-side only.
- **Final report location:** `docs/final_report.md` (versioned); `outputs/` is git-ignored.
- **Demo post timing:** Post 2 (nominated) live by 4 Nov; ~6 days engagement before 10 Nov cutoff.
