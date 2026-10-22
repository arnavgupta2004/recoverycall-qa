# Submission — ready to paste

## Project title

RecoveryCall QA — evidence-first review of debt-recovery calls in Hindi, Kannada and Hinglish

## Problem

Lenders and collection agencies make large numbers of recovery calls in Indian languages, and the RBI's guidance on recovery agents rules out intimidation, abusive language and threats to contact a borrower's family, employer or neighbours. Compliance teams can listen to only a small sample. Speech-to-text alone isn't enough, because on noisy phone lines STT can drop whole turns without any error. A reviewer then never sees the words that matter.

## Users

QA and compliance reviewers at banks, NBFCs and collection agencies who audit recovery calls, and team leads who coach agents. The tool points a reviewer at the passages worth listening to. It does not decide anything.

## Gnani models and APIs used, and how

- **Prisma STT (REST, `POST /stt/v3`, `format=verbatim`):** transcribes each call in segments of at most 12 s, cut at pauses by an energy VAD. Language codes: `hi-IN` (Hindi), `kn-IN` (Kannada) and `en-IN` for code-mixed Hinglish, which returns Latin script. A coverage check compares transcript characters per second of detected speech with a per-language baseline. A low-coverage segment is re-split at shorter pauses and sent to Prisma again. Speech that still isn't transcribed is shown to the reviewer as an "untranscribed speech" warning, never dropped silently.
- **Timbre TTS v2.5 (`POST /api/v1/tts/inference`):**
  - It synthesized every fictional test call, per turn, using Hemraj and Chitra (Hindi), Kavin and Saanvi (Kannada), and Poorvi and Pranav (Hinglish).
  - It speaks a short review summary in the call's own language (Hemraj, Kavin or Pranav). The summary is built from a fixed template, with numbers written as words.

## Other tools

- **Google Gemini** (`gemini-3.5-flash-lite`, Interactions API, free tier): a second tagger. Every quote it returns must appear word for word in the transcript; otherwise the flag is thrown out and shown as rejected.
- **The app:** a frozen rule set alongside the model, FastAPI, wavesurfer.js and ffmpeg, deployed on Google Cloud Run (Mumbai) with secrets in Secret Manager.
- **Built with** Claude Code and IBM Bob as coding assistants.

## My contribution

Solo project: problem framing, test-call scripts and labels policy, the STT/TTS spike and the omission experiment, the pipeline, the evaluation, the web app, deployment and the demo video.

## Written explanation (250 words)

RecoveryCall QA helps compliance reviewers audit debt-recovery calls in Hindi, Kannada and Hinglish. A reviewer uploads a call or opens an example. Gnani Prisma transcribes it in short segments, and the app flags passages that may breach recovery-conduct rules: intimidation, abusive language, threats to contact family or employers, and police or court claims that need checking.

Every flag quotes the exact words and links to that moment in the audio, so the reviewer listens before judging. Flags come from frozen rules and from Gemini. An AI flag is kept only if its quote appears verbatim in the transcript, which blocks invented evidence. There is no risk score and no verdict.

Early tests found a real hazard: on noisy phone audio, STT can drop a whole turn without any error. RecoveryCall measures transcript density per second of speech, re-transcribes thin segments in smaller pieces, and labels anything still missing as "untranscribed speech" instead of hiding it.

Gnani Timbre then reads a short summary aloud in the call's language.

All test calls are fictional and synthesized with Timbre; no real recordings or personal data were used. On an externally written held-out set of 18 calls, run end to end on phone-quality and noisy audio, the hybrid found all 15 labelled concerns with no conduct false flags on 9 lawful-but-firm calls. One transcription error caused one false "verify" flag. These are small synthetic sets, and the Kannada scripts still need native-speaker review.

## Links

- Live demo: https://recoverycall-qa-711436060077.asia-south1.run.app
- Code: https://github.com/arnavgupta2004/recoverycall-qa
- Demo video: `docs/demo.mp4` in the repo; post link: <ADD LINK>
- X post: <ADD LINK>
- LinkedIn post: <ADD LINK>
