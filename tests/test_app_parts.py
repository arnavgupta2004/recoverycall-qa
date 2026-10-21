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
