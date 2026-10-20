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
pip install -r requirements.txt
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
