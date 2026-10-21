#!/usr/bin/env python3
"""Bundle the four sample calls and build their cached results for the app.

    python build_samples.py prepare                 # copy sample audio + reference turns into samples/
    python build_samples.py build --mock            # cached results with mock APIs -> samples/cache_mock/ (git-ignored)
    python build_samples.py build --only D          # live result for sample D -> samples/cache/ (committed)
    python build_samples.py build --only D --no-stt-cache   # force fresh STT calls (latency measurement)

Samples: A (Hinglish reminder, compliant), B (Kannada intimidation), C (Hindi family threat),
D (C at 10 dB SNR, seed 0: the experiment condition where the coverage check fires). All audio is
synthetic Timbre speech of fictional scripts. Live builds cache STT/LLM responses under outputs/
(samples only; uploads are never cached).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.pipeline import analyse, audit_json, live_services, mock_services

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
SAMPLE_DEFS = {
    "A": {"title": "A — Hinglish payment reminder (compliant)", "language": "en-IN",
          "source": "outputs/audio/A_hinglish_reminder/clean.wav", "turns_from": "A_hinglish_reminder"},
    "B": {"title": "B — Kannada call with intimidation", "language": "kn-IN",
          "source": "outputs/audio/B_kannada_intimidation/clean.wav", "turns_from": "B_kannada_intimidation"},
    "C": {"title": "C — Hindi call with a family threat", "language": "hi-IN",
          "source": "outputs/audio/C_hindi_family_threat/clean.wav", "turns_from": "C_hindi_family_threat"},
    "D": {"title": "D — Hindi call C at 10 dB noise (coverage check fires)", "language": "hi-IN",
          "source": "outputs/audio/C_hindi_family_threat/phone_noisy.wav", "turns_from": "C_hindi_family_threat"},
}

# Mock only: turns the real STT dropped from D's first 11 s segment at 10 dB (experiment, all seeds).
MOCK_OMISSIONS = {"D": {1, 2, 3}}


def load_samples() -> dict:
    return json.loads((SAMPLES / "samples.json").read_text(encoding="utf-8"))


def cmd_prepare() -> None:
    (SAMPLES / "audio").mkdir(parents=True, exist_ok=True)
    out = {}
    for sid, d in SAMPLE_DEFS.items():
        src = ROOT / d["source"]
        if not src.exists():
            raise SystemExit(f"{d['source']} missing: run `python run_spike.py synthesize && python run_spike.py degrade`")
        dst = SAMPLES / "audio" / f"{sid}.wav"
        shutil.copyfile(src, dst)
        m = json.loads((ROOT / "outputs" / "audio" / d["turns_from"] / "manifest.json").read_text(encoding="utf-8"))
        out[sid] = {"title": d["title"], "language": d["language"], "audio": f"samples/audio/{sid}.wav",
                    "reference_turns": [{"start_s": t["start_s"], "end_s": t["end_s"], "speaker": t["speaker"],
                                         "text": t["text"],
                                         **({"mock_omit_in_long_requests": True}
                                            if sid in MOCK_OMISSIONS and t["turn"] in MOCK_OMISSIONS[sid] else {})}
                                        for t in m["turns"]]}
    (SAMPLES / "samples.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote samples/samples.json and {len(out)} audio files")


def cache_dir(mock: bool) -> Path:
    return SAMPLES / ("cache_mock" if mock else "cache")


def cmd_build(args) -> None:
    samples = load_samples()
    out_dir = cache_dir(args.mock)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in (args.only or list(samples)):
        s = samples[sid]
        if args.mock:
            services = mock_services(reference=s["reference_turns"])
        else:
            services = live_services(ROOT / ".env",
                                     stt_cache_dir=None if args.no_stt_cache else ROOT / "outputs" / "stt_cache",
                                     llm_cache_dir=ROOT / "outputs" / "llm_cache_app")
        result = analyse(ROOT / s["audio"], s["language"], services, filename=f"{sid}.wav")
        (out_dir / f"{sid}.json").write_text(audit_json(result), encoding="utf-8")
        if result["summary"]["audio_wav"]:
            (out_dir / f"{sid}_summary.wav").write_bytes(result["summary"]["audio_wav"])
        lat = result["latency_ms"]
        print(f"{sid}: {len(result['flags'])} flags, {len(result['coverage_warnings'])} coverage warning(s), "
              f"LLM {result['detectors']['llm_status']}, summary {result['summary']['status']}; "
              f"API calls {result['api_calls']}; latency ms {lat}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "build"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", nargs="*", choices=list(SAMPLE_DEFS))
    ap.add_argument("--no-stt-cache", action="store_true", help="send fresh STT requests (for latency measurement)")
    args = ap.parse_args()
    cmd_prepare() if args.command == "prepare" else cmd_build(args)


if __name__ == "__main__":
    main()
