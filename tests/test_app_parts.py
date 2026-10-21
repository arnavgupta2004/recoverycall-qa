"""Web guards and the numbers-only analysis log line."""

import json
from pathlib import Path

import pytest

from src.pipeline import analyse, mock_services
from src.webguard import RateLimiter, check_upload, client_ip, passcode_ok

ROOT = Path(__file__).resolve().parents[1]


def test_passcode_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("UPLOAD_PASSCODE", raising=False)
    assert passcode_ok("") is False and passcode_ok("anything") is False
    monkeypatch.setenv("UPLOAD_PASSCODE", "s3cret")
    assert passcode_ok("s3cret") and not passcode_ok("s3cre") and not passcode_ok(None)


def test_rate_limiter_per_key():
    now = [0.0]
    rl = RateLimiter(limit=2, window_s=60, clock=lambda: now[0])
    assert rl.try_acquire("a")[0] and rl.try_acquire("a")[0]
    ok, wait = rl.try_acquire("a")
    assert not ok and 0 < wait <= 61
    assert rl.try_acquire("b")[0]          # other IPs unaffected
    now[0] = 61
    assert rl.try_acquire("a")[0]          # window slid


def test_client_ip_uses_rightmost_forwarded_hop():
    class R:
        headers = {"x-forwarded-for": "1.2.3.4, 203.0.113.9"}        # client-forged 1.2.3.4, proxy-appended last
        client = type("C", (), {"host": "169.254.1.1"})()
    assert client_ip(R()) == "203.0.113.9"
    assert client_ip(R(), trusted_hops=1) == "1.2.3.4"
    class NoHeader:
        headers = {}
        client = type("C", (), {"host": "127.0.0.1"})()
    assert client_ip(NoHeader()) == "127.0.0.1"
    assert client_ip(None) == "unknown"


def test_check_upload(tmp_path):
    assert check_upload(None)
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    assert "WAV or MP3" in check_upload(txt)
    wav = tmp_path / "a.WAV"
    wav.write_bytes(b"RIFF")
    assert check_upload(wav) is None


@pytest.fixture(scope="module")
def result():
    s = json.loads((ROOT / "samples" / "samples.json").read_text(encoding="utf-8"))["D"]
    return analyse(ROOT / s["audio"], "hi-IN", mock_services(reference=s["reference_turns"]), make_summary_audio=False)


def test_analysis_log_line_is_numbers_only(result):
    import re
    from src.webutil import analysis_log_line
    r = dict(result)
    r["detectors"] = {**result["detectors"], "llm_status": "error: HTTP 400 quote “घर आएँगे” rejected"}
    line = analysis_log_line(r)
    assert re.fullmatch(r"[A-Za-z0-9_=.:| -]+", line), line          # ASCII tokens only: no transcript or quotes
    assert "llm=error" in line and "घर" not in line
    for stage in ("decode", "vad", "stt", "coverage_recovery", "rules", "llm", "summary_tts", "total"):
        assert re.search(rf"latency_ms .*\b{stage}=\d+", line), stage
    assert re.search(r"calls stt=\d+ stt_cache_hits=\d+ llm=\d+ tts=\d+", line)
