# RecoveryCall QA: Gnani API spike

A minimal Python spike for the Gnani AI Internship Challenge. It tests four things:

1. Are the Gnani APIs reachable and behaving as documented?
2. Can Timbre (TTS, `timbre-v2.5`) synthesize three fictional debt-recovery calls?
3. How well does Prisma (STT v2.5) transcribe them, especially the critical spans: amounts, negations, threats and family mentions?
4. How much does phone-quality audio (8 kHz, G.711 mu-law, noise) hurt that?

There is no web app, no LLM classifier, no risk score and no legal verdict. All calls, names, companies and amounts are fictional. There is no voice cloning and no real recording.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps + pytest, matplotlib
cp .env.example .env        # then put your key in .env as GNANI_API_KEY=...
```

You also need `ffmpeg` on PATH (macOS: `brew install ffmpeg`). Every audio step checks for it and exits with a clear message if it is missing.

The key is read from the `GNANI_API_KEY` environment variable, or from `.env` if the variable is unset. It is only ever placed in the `X-API-Key-ID` request header. It is never printed or logged, and request headers never appear in error output.

## Commands

```bash
python -m pytest -q                      # unit tests (metrics, normalization, client retry/errors)
python run_spike.py all --mock           # whole pipeline offline, no credits; writes outputs/mock/
python run_spike.py synthesize           # Timbre TTS per turn (cached) -> normalize -> concat + manifest
python run_spike.py degrade              # phone / phone_noisy / mu-law variants + <=28 s upload chunks
python run_spike.py transcribe           # Prisma STT for every chunk x variant x language code
python run_spike.py report               # re-score results.json and write report.md
python run_spike.py all                  # the four steps above in order
```

`--force` makes `transcribe` ignore its result cache. The TTS cache is always honoured.

`--max-chunk-s N` (default 28, max 30) sets the longest upload chunk; chunks are always split at turn boundaries. `degrade` re-plans the chunks from the manifest, so changing it needs no new TTS calls: `python run_spike.py degrade --max-chunk-s 12 && python run_spike.py transcribe && python run_spike.py report`. In the first real run, Prisma dropped the last two words of a 24.8 s Kannada chunk that it transcribed fully when the turn was sent alone, so shorter chunks are worth comparing.

### Credit protection

- **TTS:** each turn's raw API audio is cached at `cache/tts/<scenario>/turnNN_<hash>.wav`. The hash covers text, voice, language, model and speed. A rerun skips any turn that already exists, and editing one turn's text re-synthesizes only that turn. `cache/` is git-ignored, and deleting `outputs/` does not delete it.
- **STT:** successful responses, and definitive 400/403 errors, are cached in `outputs/stt_cache/`, keyed by the audio's SHA-256, language code and format. Transient failures (429/5xx/transport) are not cached, so a rerun retries only those.

### Expected call volume for a full real run

- **TTS: 21 calls**, one per turn (3 scenarios × 7 turns). That is a one-time cost, because the turns are cached.
- **STT:** depends on how many ≤28 s chunks each call needs, which depends on the real TTS durations:
  - per chunk: 3 variants × number of language codes, plus 1 ITN run on the clean audio
  - per scenario: 1 mu-law container run
  - Scenario A (2 codes): 8 calls for 1 chunk, 15 calls for 2 chunks.
  - Scenarios B and C (1 code each): 5 calls for 1 chunk, 9 calls for 2 chunks.
  - **Total: 18 to 33 STT calls**, plus any retries (at most 3 extra per request). The mock audio produced 2 / 2 / 1 chunks, which gives 29 calls. `transcribe` prints the planned count before it sends anything.

## Silent-omission experiment (`experiment.py`)

The first real runs showed Prisma sometimes drops whole turns without any error, especially in noise. `experiment.py` tests a way to detect and recover those omissions **using only the audio and the STT output**. The reference script is used only for evaluation, and to place noise at a measured SNR in the simulation.

```bash
python experiment.py prepare     # noise sweep audio + VAD segments; no API calls
python experiment.py plan        # exact base STT count + worst-case recovery count; no API calls
python experiment.py run         # STT + coverage check + recovery; stops on 403 or on 429/5xx that survive retries
python experiment.py report      # evaluation vs reference -> outputs/experiment/report.md (+ omission_vs_snr.png)
python run_spike.py report       # embeds the experiment report in outputs/report.md
```

All four commands accept `--mock`.

1. **Segmentation (product path).** `src/vad.py` is an energy VAD with no torch and no extra dependencies.
   - **Speech detection:** a 30 ms frame counts as speech when it is 3 dB above a noise floor, taken as the 5th percentile of frame energy. Gaps under 300 ms are bridged.
   - **Partitioning:** the whole timeline is split into segments of at most 12 s. Cuts go at pause midpoints, choosing the furthest pause that doesn't leave a tail shorter than 1.5 s. Segments shorter than 1.5 s are merged with a neighbour when the result still fits, and each segment is padded with 300 ms of silence.
   - **No audio is discarded,** so a VAD miss can only distort the speech-seconds estimate.
   - The manifest-based chunking in `run_spike.py` is unchanged and stays available for comparison.
2. **Coverage check.**
   - Density = normalized transcript characters (spaces excluded) per VAD speech-second.
   - The per-language baseline comes from the clean verbatim chunks of the archived runs in `outputs/runs/`; it is computed from transcripts, not the reference.
   - A segment is flagged when its density is below `--flag-fraction` (default 0.5) × baseline.
   - Segments with less than 0.5 s of detected speech are not judged.
3. **Recovery.**
   - A flagged segment is re-split at shorter pauses (≥120 ms; pieces ≤6 s, merged up to ≥0.8 s) and the pieces are transcribed.
   - The pieces' joined transcript is kept only if density improves.
   - Anything still below threshold is recorded as **untranscribed speech** with start and end times. So is a segment whose STT request failed.
4. **Evaluation (reference used here only).**
   - A turn is **missing** if its text matches the transcript of the segments it overlaps with ratio < 0.5.
   - The report gives detection recall, recovery rate, false flags on clean audio, silent omissions after recovery (missing and not covered by an untranscribed record), and extra STT calls.
5. **Noise sweep.**
   - Conditions: clean, plus 20, 15, 10 and 5 dB SNR with 3 seeds each. That's 13 conditions per call, all built from cached TTS audio.
   - Seed 0 at 10 dB is byte-identical to `run_spike`'s `phone_noisy` audio, so the report can compare the 12 s manifest chunks with the VAD segments on the same audio, at no extra cost.

## Pipeline

| step | what happens |
|---|---|
| synthesize | Each turn in `data/scripts.json` goes to `POST /api/v1/tts/inference` with `model=timbre-v2.5`, `audio_config` 16 kHz / mono / 16-bit / `linear_pcm` / `wav`. ffmpeg normalizes each turn to 16 kHz mono s16le. Turns are joined with 350 ms of silence into `outputs/audio/<scenario>/clean.wav`. `manifest.json` records per-turn start/end offsets in seconds and the chunk plan. |
| degrade | **phone:** highpass 300 Hz + lowpass 3400 Hz, resample to 8 kHz mono, encode G.711 mu-law (`phone_ulaw.wav`), decode back to 8 kHz s16le (`phone.wav`), which is what gets uploaded. **phone_noisy:** same chain, but band-limited pink noise is mixed in before the codec step, at a target of 10 dB SNR; the measured SNR is recorded. **phone_ulaw_container:** the first chunk of each scenario is re-encoded as a mu-law WAV and uploaded as-is, once per scenario, to see whether the API accepts it. Every variant is cut into the same ≤28 s chunks at turn boundaries. |
| transcribe | `POST /stt/v3` (multipart) is called for every chunk × {clean, phone, phone_noisy} × each listed language code with `format=verbatim`. There is also one `format=transcribe` (ITN) run per clean chunk, and one mu-law container run per scenario. Timeouts are 10 s connect and 60 s read. 429/500/503 and transport errors are retried with exponential backoff and jitter, up to 4 tries in total, honouring `Retry-After`. 400 and 403 are never retried. |
| report | All results are re-scored with the current `src/metrics.py`, `outputs/results.json` is rewritten, and `outputs/report.md` is written. |

The ITN and mu-law container runs use the scenario's `primary_stt_language_code`, which defaults to the first listed code. For the Hinglish scenario it is set to `en-IN`, because the docs say en-IN accepts mixed English–Hindi audio and writes Latin script, which matches the Roman-script reference. Both codes are still run on every verbatim variant.

### SNR measurement

- Signal power is measured only over the turn segments from the manifest. The inter-turn silence is digital zero and would otherwise deflate the signal power.
- The measurement is made on the 8 kHz band-limited signal, before the codec.
- Noise is pink (1/f), band-limited to 300–3400 Hz and seeded, so runs are reproducible.
- The recorded `measured_snr_db` is recomputed from the final int16 mix, so it includes quantization and any clipping. The clipped-sample count is recorded too.
- The mu-law codec adds its own quantization noise afterwards; that is not included in the number.

## Metrics (`src/metrics.py`)

**Normalization.** The same steps are applied to reference and hypothesis:

1. Unicode NFC.
2. Remove every format character (category `Cf`), including ZWJ U+200D and ZWNJ U+200C. Kannada scenario B deliberately contains ZWNJ.
3. Lowercase.
4. Replace every punctuation character (category `P*`, plus `।` and `॥`) with a space.
5. Collapse whitespace.

**Not done:** no number or word conversion (`500` ≠ `paanch sau`), no transliteration, no nukta folding (`हज़ार` ≠ `हजार`), and no spelling normalization of Hinglish romanizations. These differences count as errors on purpose, because they are exactly what a downstream QA rule would trip over.

**WER and CER** come from a Levenshtein alignment with backtrace, which also gives the substitution, deletion and insertion counts. WER works on whitespace tokens. CER works on Unicode code points of the normalized string, spaces included. Code points are not grapheme clusters, so an Indic vowel sign counts as its own character. Aggregates are micro-averaged: total edits divided by total reference words.

**Script check.**
- Each letter (category `L*`) is tagged with its Unicode script name: LATIN, DEVANAGARI, KANNADA and so on.
- If fewer than half the hypothesis letters are in the reference's dominant script, the run gets `script_mismatch=true` and is excluded from WER aggregates. The report lists it separately.
- The typical case is `hi-IN` returning Devanagari for the Roman-script Hinglish reference.
- Combining vowel signs (`Mn`/`Mc`) are not letters, so they are not counted.

**Critical spans.**
- Each span is fuzzy-matched against the transcript of the chunk that contains its turn.
- Matching slides word windows of length n−1, n and n+1 (n = span words) over the normalized transcript and scores each with `difflib.SequenceMatcher` ratio.
- Classes:
  - `exact`: the normalized span appears verbatim as a token sequence.
  - `near`: best ratio ≥ 0.8.
  - `garbled`: anything lower.
- The best-matching transcript text is always reported.
- Negation spans carry `negation_markers`. If none of them appears in the best match, `negation_missing=true` and the report flags **NEGATION LOST**. A dropped "not" flips the meaning, so this is the most dangerous error type.
- Kannada negation is a verb suffix (ಸಾಧ್ಯವಿಲ್ಲ, ಮಾಡಬೇಡಿ). Its markers (`ಲ್ಲ`, `ಬೇಡ`) are therefore substrings, not whole words.

## Reading the report (`outputs/report.md`)

1. **Aggregates by variant:** clean vs phone vs phone_noisy (and the mu-law container), reported separately. These rows pool scenarios and language codes, so use the **by scenario / lang / variant** table for like-for-like comparisons. For the Hinglish scenario, that table is where hi-IN vs en-IN is decided.
2. **Script mismatches:** runs that were excluded, and why.
3. **Per-run results:** one row per request, with HTTP status, attempts, latency, WER/CER, S/D/I, script mismatch, span counts (Exact/Near/Garbled) and request_id.
4. **Critical errors:** every non-exact amount, negation, threat and family span, with the text the transcript had instead.
5. **ITN:** how `format=transcribe` renders the amounts.
6. **Unsupported or failed features:** every failed TTS/STT request with its HTTP status and error body exactly as returned, every retry, and whether the mu-law container upload was accepted.

`outputs/results.json` has one record per request, with every field above plus the reference, the transcript and the per-span details.

A mock run writes to `outputs/mock/` and is labelled **MOCK RUN** at the top of the report. Its numbers come from a fake that corrupts the reference text, so they say nothing about Gnani's models.

## Docs check (re-read 21 Oct 2026)

These pages were re-read before coding:
- [speech-to-text](https://docs.gnani.ai/api/STT/speech-to-text.md)
- [tts-inference](https://docs.gnani.ai/api/TTS/tts-inference.md)
- [available-voices](https://docs.gnani.ai/api/TTS/available-voices.md)

Endpoints, the auth header, field names, language codes, `verbatim`/`transcribe`, the error shape, the speed range 0.85–1.15 and the raw-audio TTS response all match the brief. Differences and extra findings:

- **Max STT duration:** most of the page says 60 s (ideal ≤ 30 s), but the OpenAPI path description says 30 s. This spike caps chunks at 28 s.
- **`request_id`** appears in the success example but not in the OpenAPI `STTResponse` schema. It is recorded when present and `null` otherwise.
- **en-IN STT** is documented as accepting mixed English–Hindi audio and returning English in Latin script. There is still no Hinglish STT code, and none is invented here. The client rejects any code outside the documented ten.
- **403 FORBIDDEN** appears only in the OpenAPI spec, not in the prose error table.
- **No STT rate-limit numbers** are published.
- **Extra STT fields** exist that this spike does not use: `itn_native_numerals`, `enable_substitution` / `substitution_map`, `bias_list` / `bias_score`.
- **TTS `voice` and `language`** are "required" in the prose but not in the OpenAPI `required` list. Both are always sent here.
- **Voices** are per model; sending a v2.5 voice to v3.0 (or the reverse) returns 400. The v2.5 voices used here, with their preferred languages:

  | scenario | agent | customer |
  |---|---|---|
  | A (Hinglish) | Poorvi (hi-en) | Pranav (en-IN, run with `language=hi-en`) |
  | B (Kannada) | Kavin (kn-IN) | Saanvi (kn-IN) |
  | C (Hindi) | Hemraj (hi-IN) | Chitra (hi-IN) |

  Poorvi is the only v2.5 voice whose preferred language is hi-en. The docs say a language/voice mismatch "may reduce quality", but it is not rejected.
- **Numbers** must be written as spoken words in TTS input, which `scripts.json` does.

## Data

`data/scripts.json` holds three fictional calls. Each turn's `text` is both the TTS input and the STT reference. Each scenario defines:
- `tts_language`
- `voices` per speaker (a turn can override `voice` or `tts_language`)
- `stt_language_codes`
- optionally `primary_stt_language_code`

Each turn may list `critical_spans` of type `amount`, `negation`, `threat` or `family`. A unit test checks that every span occurs verbatim in its turn.

Note: no `data/scripts.json` existed when this spike was built, so the file was written from the brief (A compliant Hinglish reminder, B Kannada intimidation, C Hindi family threat). Replace it with your own if you have one; the pipeline only depends on the schema above.

## Limits

- **Synthetic audio only.** TTS speech is cleaner and more regular than real callers, so absolute WER will be optimistic. Use the results to compare variants, not as a production estimate.
- **The reference is the TTS input text.** If Timbre mispronounces something, that error is charged to STT.
- **Only one sample per cell:** three short calls and one noise seed. Differences of a few WER points are noise.
- **Romanized Hinglish has no canonical spelling** ("nahi" / "nahin", "hazaar" / "hazar"), so en-IN WER on scenario A is inflated by spelling variants. Read its span matches, not just its WER.
- **The mu-law container test** covers one chunk per scenario, with the primary language code only.
- **The REST STT response has no word timestamps**, so spans are located by text matching, not by time.

## Concern tagging: rules, LLM and hybrid

```bash
python prepare_labels.py                                   # rebuild labels: merge held-out v1 into dev as dev_v2 (no classifier)
python eval_classifier_text.py --split dev --llm-mock      # rules vs LLM vs hybrid, offline mock LLM
python eval_classifier_text.py --split dev                 # same with the real LLM (needs GEMINI_API_KEY; cached)
python eval_classifier_text.py --split dev --no-llm        # rules only
```

- **Rules v1** (`src/concern_rules.py`) are frozen at `12cb3b1`. Their one held-out result is the baseline: conduct recall 5/17, 0 compliant false flags (`docs/baseline_rules_v1.md`).
- **Held-out v1 is spent.** It is now `split=dev_v2` in `data/scripts_dev.json` and the manifest; the delivered file is kept in `data/held_out_source/`. Every label change is in `label_changes` and in `prepare_labels.py`.
  - Third-party contact threat: telling or showing family, employer, neighbours or contacts.
  - Intimidation: visiting, harming or shaming in person.
  - One passage can carry both labels.
- **Held-out v2** will be written externally to `data/scripts_held_out.json`. `--split held_out` evaluates it once it exists.
- **Quotes:** a rules flag quotes the matched phrase, mapped back to the original text, not the whole turn.
- **`src/llm_tagger.py`** calls the Gemini API free tier (default `gemini-3.5-flash-lite`, overridable with `GEMINI_MODEL`).
  - The model gets the four category definitions and returns JSON `{segment_id, type, quote}` at temperature 0.
  - Every quote must be a verbatim substring of its segment (`verify_quote`); anything else is dropped and reported.
  - Responses are cached in `outputs/llm_cache/` (git-ignored; `--llm-cache DIR` overrides it), and `--llm-mock` runs offline. A cache entry holds only the model's JSON text, backend name, prompt version and language: no interaction ids, thought signatures, headers or keys.
  - The evaluation prints LLM API requests (including retries), cache hits, every non-200 attempt and the run time.
  - The key is read from `GEMINI_API_KEY` (environment or `.env`) and sent only in the request header.
  - The request format follows the 2026-10-21 Interactions API docs and was confirmed with a live call on 2026-10-21. No temperature deprecation warning came back.
- **Hybrid** (`src/hybrid.py`) is rules ∪ verified LLM flags. Same-type flags whose quotes overlap in the same segment are merged, keeping the rules quote. Labels and wording are unchanged.
- **Scoring:**
  - Conduct categories are scored per call.
  - `authority_or_legal_action_claim` is a verify flag with separate precision and recall.
  - The compliant and hard-negative false-flag rates count conduct flags only.
  - Dev results are context only.

## Web app: RecoveryCall QA single page

The app is a **FastAPI** backend (`app.py`) that serves a static single-page frontend (`static/`: plain JavaScript, no build step). The frontend uses wavesurfer.js 7.12.12, pinned from jsDelivr, and the fonts Inter, Noto Sans Devanagari and Noto Sans Kannada. The backend wraps `src/pipeline.analyse(audio_path, language)`, which runs in this order:

1. ffmpeg decode.
2. Energy VAD into segments of at most 12 s.
3. Prisma STT, using `en-IN` for Hinglish.
4. Coverage check and re-split recovery.
5. Frozen rules together with verified Gemini flags (20 s cap, with a rules-only fallback notice).
6. Timbre spoken summary.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
RCQA_MOCK=1 UPLOAD_PASSCODE=local-test .venv/bin/uvicorn app:app --port 7860   # live runs use mock STT/LLM/TTS
.venv/bin/uvicorn app:app --port 7860                                         # real APIs for live runs (keys from .env)
```

| endpoint | what it returns |
|---|---|
| `GET /api/samples` | the four bundled calls, with flag and coverage-gap positions for the card sketches |
| `GET /api/samples/{id}` | the cached result, plus `/audio` and `/summary` URLs. These views make 0 API calls and log `cached example served … api_calls=0`. |
| `POST /api/analyse` | multipart `language`, `passcode`, and either `audio` (WAV or MP3) or `sample_id`. The response is a server-sent event stream: `stage` events (transcribing → coverage → concerns → summary), then `done` with the audit result and the spoken summary, or `error`. |

Stage progress comes from wrapping the services `analyse()` already receives: segment STT calls, re-split piece STT calls, the LLM call and the TTS call. `src/pipeline.py` is unchanged.

**Safety rules:**
- **Passcode:** live analysis needs `UPLOAD_PASSCODE`, and is disabled when it's unset.
- **Rate limit:** per client IP, taken from the **rightmost** `X-Forwarded-For` hop. The left part of that header can be forged; `XFF_TRUSTED_HOPS` handles extra trusted proxies.
- **Limits:** 20 MB and 2 minutes per upload.
- **No persistence:** uploads are written only to a per-request temporary directory, which is deleted before the stream ends. The player uses the browser's local copy of the file, and the audit JSON is built in the browser.
- **Logs:** counts and timings only.
- **Wording:** unchanged — potential conduct concern, verify claim, "to verify against the RBI circular", no score.

**Page:**
- **Start screen:** four sample cards, each labelled "cached example", plus "Analyse your own audio" behind a passcode dialog. The dialog can also run a sample live.
- **Timeline:** a waveform with segment boundaries, amber and violet flag pins, and hatched low-coverage stretches. Clicking anywhere seeks.
- **Transcript:** segment bubbles with timestamps and underlined, clickable quotes. Speakers aren't separated, because the pipeline has no diarization.
- **Findings:** a card per flag (detector Rules, AI or Both), rejected AI flags shown struck through, plus coverage and call-time cards.
- **Also:** the spoken summary, the audit JSON download, a Details panel with stage timings and API calls, and a step indicator for live runs.
- **Layout:** light and dark mode. On phones it stacks timeline, findings, then transcript.

Screenshots are in `docs/screenshots/`.

### Deploy to Google Cloud Run

Hugging Face no longer runs Gradio Spaces on the free CPU tier without PRO, so the app is deployed to **Cloud Run** in `asia-south1` (Mumbai, a Tier 1 pricing region) with its own GCP project (`recoverycall-qa-1021-308c`).

The deployment is set up like this:
- **Image:** the `Dockerfile` (Python 3.13 slim + ffmpeg + `requirements.txt`) holds only `app.py`, `src/` and `samples/`. `.dockerignore` and `.gcloudignore` keep `.env`, caches and outputs out of the build upload.
- **Service:** public (`--allow-unauthenticated`), scale to zero (`--min-instances 0`), `--max-instances 1` so the in-memory rate limiter is exact and cost is bounded. It has 1 vCPU, 1 GiB of memory and a 300 s request timeout.
- **Secrets:** `GNANI_API_KEY`, `GEMINI_API_KEY` and `UPLOAD_PASSCODE` live in **Secret Manager** and are exposed to the container as environment variables. Only the `rcqa-run` service account can read them.
  - To read the passcode: `gcloud secrets versions access latest --secret=UPLOAD_PASSCODE --project recoverycall-qa-1021-308c`
  - To rotate a secret: `printf %s "$NEW" | gcloud secrets versions add NAME --data-file=- --project …`, then redeploy.
- **Budget:** ₹100 a month on this project, with alerts at 50%, 90% and 100% of actual spend and 100% of forecast, emailed to billing admins.
- **Deploy:** `PROJECT=recoverycall-qa-1021-308c ./deploy_cloudrun.sh` builds with Cloud Build and deploys.
- **Live URL:** https://recoverycall-qa-711436060077.asia-south1.run.app (revision `recoverycall-qa-00004-rg8`, FastAPI frontend, 2026-10-21).
- **Measured on 2026-10-21 (revision 00004):**
  - **Cold start:** after 20 minutes idle, the first page load took 2.6 s to first byte; Gradio had taken 11.4 s. The instance started at 18:27:12 UTC and the app was ready 2.4 s later. After that, a warm request takes 0.1 s.
  - **Cached samples:** all four load with their waveform, flags and summary audio and 0 API calls, logged as `cached example served … api_calls=0`.
  - **Log audit:** 88 entries with no API key, passcode or transcript text.
  - **Live run, sample D (Gradio revision 00003, same pipeline):** 8.3 s total — STT 0.65 s, recovery 0.44 s, Gemini 5.0 s, Timbre 1.96 s — with 7 API calls (5 STT, 1 Gemini, 1 Timbre).
- **Client IP:** Google doesn't document how Cloud Run fills `X-Forwarded-For` for direct `*.run.app` traffic. The app uses the rightmost entry, which is the real client whether Google's front end appends the address it saw or replaces the header. With an external load balancer in front, which appends `<client-ip>,<lb-ip>`, set `XFF_TRUSTED_HOPS=1`.
- **Logs:** counts and timings only.
- **Wording:** unchanged — potential conduct concern, verify claim, "to verify against the RBI circular", no score.

**Page:**
- **Start screen:** four sample cards, each labelled "cached example", plus "Analyse your own audio" behind a passcode dialog. The dialog can also run a sample live.
- **Timeline:** a waveform with segment boundaries, amber and violet flag pins, and hatched low-coverage stretches. Clicking anywhere seeks.
- **Transcript:** segment bubbles with timestamps and underlined, clickable quotes. Speakers aren't separated, because the pipeline has no diarization.
- **Findings:** a card per flag (detector Rules, AI or Both), rejected AI flags shown struck through, plus coverage and call-time cards.
- **Also:** the spoken summary, the audit JSON download, a Details panel with stage timings and API calls, and a step indicator for live runs.
- **Layout:** light and dark mode. On phones it stacks timeline, findings, then transcript.

Screenshots are in `docs/screenshots/`.

### Deploy to Google Cloud Run

Hugging Face no longer runs Gradio Spaces on the free CPU tier without PRO, so the app is deployed to **Cloud Run** in `asia-south1` (Mumbai, a Tier 1 pricing region) with its own GCP project (`recoverycall-qa-1021-308c`).

The deployment is set up like this:
- **Image:** the `Dockerfile` (Python 3.13 slim + ffmpeg + `requirements.txt`) holds only `app.py`, `src/` and `samples/`. `.dockerignore` and `.gcloudignore` keep `.env`, caches and outputs out of the build upload.
- **Service:** public (`--allow-unauthenticated`), scale to zero (`--min-instances 0`), `--max-instances 1` so the in-memory rate limiter is exact and cost is bounded. It has 1 vCPU, 1 GiB of memory and a 300 s request timeout.
- **Secrets:** `GNANI_API_KEY`, `GEMINI_API_KEY` and `UPLOAD_PASSCODE` live in **Secret Manager** and are exposed to the container as environment variables. Only the `rcqa-run` service account can read them.
  - To read the passcode: `gcloud secrets versions access latest --secret=UPLOAD_PASSCODE --project recoverycall-qa-1021-308c`
  - To rotate a secret: `printf %s "$NEW" | gcloud secrets versions add NAME --data-file=- --project …`, then redeploy.
- **Budget:** ₹100 a month on this project, with alerts at 50%, 90% and 100% of actual spend and 100% of forecast, emailed to billing admins.
- **Deploy:** `PROJECT=recoverycall-qa-1021-308c ./deploy_cloudrun.sh` builds with Cloud Build and deploys.
- **Live URL:** https://recoverycall-qa-711436060077.asia-south1.run.app (revision `recoverycall-qa-00002-wqj`, 2026-10-21).
- **Measured on 2026-10-21:**
  - A cold start (first request after about 3.5 h idle) took 11.4 s to first byte; a warm request took 0.1 s.
  - All four cached samples loaded with 0 API calls (logged as `cached example served … api_calls=0`).
  - A log audit of 203 entries found no API key, passcode or transcript text.
- **Logs:** `gcloud run services logs read recoverycall-qa --region asia-south1 --project recoverycall-qa-1021-308c`. They contain counts and timings only: `analysis done: … | calls … | latency_ms …`, `live request accepted` or `live request refused: reason=…`, and `cached example served: sample=X api_calls=0`.
