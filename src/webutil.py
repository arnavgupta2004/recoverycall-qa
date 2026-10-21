"""Helpers for the web API: a numbers-only log line per analysis and inline MP3 encoding of WAV clips."""

from __future__ import annotations

import base64
import subprocess

from src.audio import require_ffmpeg


def wav_data_uri(wav: bytes) -> str:
    """Inline a WAV clip (e.g. the spoken summary) as MP3 when ffmpeg can encode it, else as WAV."""
    proc = subprocess.run([require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ac", "1",
                           "-c:a", "libmp3lame", "-b:a", "48k", "-f", "mp3", "pipe:1"], input=wav, capture_output=True)
    if proc.returncode == 0 and proc.stdout:
        return "data:audio/mpeg;base64," + base64.b64encode(proc.stdout).decode("ascii")
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


def analysis_log_line(result: dict) -> str:
    """One log line per analysis: language, counts, LLM status class, API calls, per-stage ms. Numbers only, no text
    from the call (no transcript, quotes, file name or error messages)."""
    status = str(result["detectors"]["llm_status"]).split(":")[0].split(" ")[0]   # ok / timeout / error / disabled
    calls = " ".join(f"{k}={int(v)}" for k, v in result["api_calls"].items())
    stages = " ".join(f"{k}={int(round(v))}" for k, v in result["latency_ms"].items())
    return (f"analysis done: lang={result['language']} duration_s={result['duration_s']:.1f} "
            f"segments={len(result['segments'])} flags={len(result['flags'])} "
            f"coverage_warnings={len(result['coverage_warnings'])} llm={status} | calls {calls} | latency_ms {stages}")
