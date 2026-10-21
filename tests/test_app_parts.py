"""Web guards and HTML rendering (no Gradio server needed)."""

import json
from pathlib import Path

import pytest

from src.pipeline import analyse, mock_services
from src.render import render_result
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


def test_client_ip_prefers_forwarded_for():
    class R:
        headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
        client = type("C", (), {"host": "10.0.0.1"})()
    assert client_ip(R()) == "203.0.113.9"
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


def test_render_highlights_and_seek_targets(result):
    page = render_result(result, "data:audio/mpeg;base64,AAAA", None, cached=True)
    assert "(cached example)" in page
    assert page.count("<mark") >= 1 and 'data-seek="' in page
    assert "untranscribed speech at" in page
    assert "illegal" not in page.lower() and "violation" not in page.lower() and "risk score" not in page.lower()


def test_render_escapes_transcript_html(result):
    r = json.loads(json.dumps({k: v for k, v in result.items() if k != "summary"}))
    r["summary"] = {"text": "<b>x</b>", "status": "ok"}
    r["transcript"][0]["text"] = '<script>alert(1)</script> घर आएँगे'
    r["flags"] = []
    page = render_result(r, None, None, cached=False)
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "<b>x</b>" not in page


def test_details_show_verifier_rejections_and_cached_note():
    from src.render import details_markdown
    r = json.loads((ROOT / "samples" / "cache" / "B.json").read_text(encoding="utf-8"))
    r["detectors"]["llm_quotes_rejected"] = [
        {"type": "abusive_language", "quote": "a | b *x*", "reason": "quote not verbatim in segment"},
        {"type": "risk_score", "quote": "q", "reason": "unknown category"}]
    md = details_markdown(r, cached=True)
    assert "1 AI flag rejected: quote not found in transcript" in md
    assert "0 API calls" in md and "a \\| b \\*x\\*" in md and "other rejection (unknown category)" in md
    r["detectors"]["llm_quotes_rejected"] = []
    assert "0 AI flags rejected: quote not found in transcript" in details_markdown(r)


def test_analysis_log_line_is_numbers_only(result):
    import re
    from src.render import analysis_log_line
    r = dict(result)
    r["detectors"] = {**result["detectors"], "llm_status": "error: HTTP 400 quote “घर आएँगे” rejected"}
    line = analysis_log_line(r)
    assert re.fullmatch(r"[A-Za-z0-9_=.:| -]+", line), line          # ASCII tokens only: no transcript or quotes
    assert "llm=error" in line and "घर" not in line
    for stage in ("decode", "vad", "stt", "coverage_recovery", "rules", "llm", "summary_tts", "total"):
        assert re.search(rf"latency_ms .*\b{stage}=\d+", line), stage
    assert re.search(r"calls stt=\d+ stt_cache_hits=\d+ llm=\d+ tts=\d+", line)
