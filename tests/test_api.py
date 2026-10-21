"""FastAPI endpoints: samples, live analysis over SSE, and every safety rule (mock services, no network)."""

import importlib
import json
import logging
import re
import shutil
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"illegal|violation|risk score|unlawful", re.I)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("RCQA_MOCK", "1")
    monkeypatch.setenv("UPLOAD_PASSCODE", "test-pass")
    import app as app_module
    app_module = importlib.reload(app_module)
    return app_module, TestClient(app_module.app)


def sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        ev = re.search(r"^event: (.+)$", block, re.M)
        data = re.search(r"^data: (.+)$", block, re.M)
        if ev and data:
            out.append((ev.group(1), json.loads(data.group(1))))
    return out


def test_page_and_static_assets(api):
    _, c = api
    page = c.get("/")
    assert page.status_code == 200
    assert "Fictional or test audio only. Uploads are not stored. Live analysis sends the transcript to Google Gemini." in page.text
    assert "wavesurfer.js@7.12.12" in c.get("/static/app.js").text
    for f in ("index.html", "app.js", "app.css"):
        assert not FORBIDDEN.search((ROOT / "static" / f).read_text(encoding="utf-8")), f


def test_config_reflects_passcode(api, monkeypatch):
    _, c = api
    assert c.get("/api/config").json()["live_enabled"] is True
    monkeypatch.delenv("UPLOAD_PASSCODE")
    assert c.get("/api/config").json()["live_enabled"] is False


def test_samples_list_and_detail(api, caplog):
    _, c = api
    lst = c.get("/api/samples").json()
    assert [s["id"] for s in lst] == ["D", "B", "C", "A"]
    assert [s["card"] for s in lst] == ["Hindi · noisy phone line", "Kannada", "Hindi", "Hinglish · compliant"]
    d = lst[0]
    assert d["coverage_warnings"] == 1 and d["flags"] == 2 and d["pins"] and d["gaps"]
    with caplog.at_level(logging.INFO, logger="rcqa"):
        detail = c.get("/api/samples/D").json()
    assert detail["cached"] is True and detail["result"]["flags"]
    assert "cached example served: sample=D api_calls=0" in caplog.text
    audio = c.get(detail["audio_url"])
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert c.get(detail["summary_audio_url"]).status_code == 200
    assert c.get("/api/samples/Z").status_code == 404
    assert c.get("/api/samples/Z/audio").status_code == 404


def _post(c, **form):
    files = form.pop("files", None)
    return c.post("/api/analyse", data=form, files=files)


def test_live_refusals(api, monkeypatch):
    mod, c = api
    assert _post(c, language="ta-IN", passcode="test-pass", sample_id="D").status_code == 400
    r = _post(c, language="hi-IN", passcode="wrong", sample_id="D")
    assert r.status_code == 403 and "passcode" in r.json()["error"].lower()
    assert _post(c, language="hi-IN", passcode="test-pass").status_code == 400                  # no source
    txt = ("a.txt", b"hello", "text/plain")
    assert _post(c, language="hi-IN", passcode="test-pass", files={"audio": txt}).status_code == 400
    monkeypatch.delenv("UPLOAD_PASSCODE")
    r = _post(c, language="hi-IN", passcode="test-pass", sample_id="D")
    assert r.status_code == 403 and "disabled" in r.json()["error"]


def test_rate_limit_keys_on_rightmost_forwarded_hop(api):
    mod, c = api
    from src.webguard import RateLimiter
    mod.LIVE_LIMIT = RateLimiter(limit=1, window_s=3600)
    h = {"x-forwarded-for": "9.9.9.9, 198.51.100.7"}
    assert c.post("/api/analyse", data={"language": "hi-IN", "passcode": "test-pass", "sample_id": "C"}, headers=h).status_code == 200
    # Forging a different leftmost hop does not get a fresh allowance.
    h2 = {"x-forwarded-for": "1.1.1.1, 198.51.100.7"}
    r = c.post("/api/analyse", data={"language": "hi-IN", "passcode": "test-pass", "sample_id": "C"}, headers=h2)
    assert r.status_code == 429
    h3 = {"x-forwarded-for": "198.51.100.8"}
    assert c.post("/api/analyse", data={"language": "hi-IN", "passcode": "test-pass", "sample_id": "C"}, headers=h3).status_code == 200


def test_live_sample_streams_stages_then_result(api, caplog):
    _, c = api
    with caplog.at_level(logging.INFO, logger="rcqa"):
        r = _post(c, language="hi-IN", passcode="test-pass", sample_id="D")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    stages = [(d["stage"], d["status"]) for e, d in events if e == "stage"]
    assert stages[0] == ("transcribing", "active")
    first_active = [s for s, st in stages if st == "active"]
    assert first_active == sorted(first_active, key=["transcribing", "coverage", "concerns", "summary"].index)
    assert "coverage" in first_active                     # D's mock omission triggers re-split recovery
    done = [d for e, d in events if e == "done"]
    assert len(done) == 1
    res = done[0]["result"]
    assert res["coverage_warnings"] and res["flags"] and "audio_wav" not in res["summary"]
    assert done[0]["summary_audio"].startswith("data:audio/")
    assert not FORBIDDEN.search(json.dumps(res, ensure_ascii=False))
    # logs: counts and timings only, no transcript text
    assert "live request accepted" in caplog.text and "analysis done:" in caplog.text
    assert not re.search(r"[ऀ-ॿಀ-೿]", caplog.text)


def test_upload_is_analysed_then_deleted(api, caplog):
    _, c = api
    before = {p.name for p in Path(tempfile.gettempdir()).glob("rcqa_up_*")}
    src = ROOT / "samples" / "audio" / "A.wav"
    with caplog.at_level(logging.INFO, logger="rcqa"):
        r = _post(c, language="en-IN", passcode="test-pass",
                  files={"audio": ("20261021_213000_call.wav", src.read_bytes(), "audio/wav")})
    events = sse_events(r.text)
    res = [d for e, d in events if e == "done"][0]["result"]
    assert res["out_of_hours"]["outside_window"] is True
    after = {p.name for p in Path(tempfile.gettempdir()).glob("rcqa_up_*")}
    assert after == before                                   # the upload's temp directory is gone
    assert "20261021_213000_call" not in caplog.text          # no file names in logs


def test_too_long_upload_streams_an_error(api, tmp_path):
    _, c = api
    f = tmp_path / "long.wav"
    with wave.open(str(f), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000)
        w.writeframes((np.random.default_rng(0).normal(0, 2000, 8000 * 125)).astype("<i2").tobytes())
    r = _post(c, language="hi-IN", passcode="test-pass", files={"audio": ("long.wav", f.read_bytes(), "audio/wav")})
    errs = [d for e, d in sse_events(r.text) if e == "error"]
    assert errs and "limit" in errs[0]["message"]


def test_llm_failure_shows_fallback_notice(api, monkeypatch):
    mod, c = api
    from src.llm_tagger import LLMError
    from src.pipeline import LLM_UNAVAILABLE_NOTICE, mock_services

    class Failing:
        name = "failing"

        def complete(self, system, user):
            raise LLMError("HTTP 503")

    samples = json.loads((ROOT / "samples" / "samples.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(mod, "_live_services",
                        lambda sid: mock_services(reference=samples["C"]["reference_turns"], llm_backend=Failing()))
    r = _post(c, language="hi-IN", passcode="test-pass", sample_id="C")
    res = [d for e, d in sse_events(r.text) if e == "done"][0]["result"]
    assert res["notices"] == [LLM_UNAVAILABLE_NOTICE]
    assert all(f["detector"] == "rules" for f in res["flags"])
