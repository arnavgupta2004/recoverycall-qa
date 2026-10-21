"""pipeline.analyse with mock services (no network, no credits)."""

import json
import re
import shutil
import time
from pathlib import Path

import pytest

from src.classifier import verify_quote
from src.pipeline import (LLM_UNAVAILABLE_NOTICE, AnalysisError, analyse, audit_json, mock_services, out_of_hours,
                          summary_text)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = json.loads((ROOT / "samples" / "samples.json").read_text(encoding="utf-8"))
FORBIDDEN = re.compile(r"illegal|violation|risk score|unlawful|गैरकानूनी|उल्लंघन|ಉಲ್ಲಂಘನೆ|ಅಕ್ರಮ", re.I)


def run_sample(sid, **kw):
    s = SAMPLES[sid]
    return analyse(ROOT / s["audio"], s["language"], mock_services(reference=s["reference_turns"], **kw),
                   filename=f"{sid}.wav")


@pytest.fixture(scope="module")
def result_d():
    return run_sample("D")


def test_result_shape_and_json(result_d):
    r = result_d
    for key in ("segments", "transcript", "flags", "coverage_warnings", "summary", "latency_ms", "api_calls",
                "detectors", "out_of_hours", "notices"):
        assert key in r
    assert set(r["latency_ms"]) >= {"decode", "vad", "stt", "coverage_recovery", "rules", "llm", "summary_tts", "total"}
    assert all(s["end_s"] - s["start_s"] <= 12.0 + 1e-6 for s in r["segments"])
    doc = json.loads(audit_json(r))
    assert "audio_wav" not in doc["summary"] and doc["summary"]["text"] == r["summary"]["text"]
    assert r["temp_files_removed"] is True


def test_every_flag_quote_is_verbatim_and_labelled(result_d):
    rows = {x["segment_id"]: x["text"] for x in result_d["transcript"]}
    assert result_d["flags"]
    for f in result_d["flags"]:
        assert verify_quote(f["quote"], rows[f["segment_id"]])
        assert f["regulatory_note"] == "to verify against the RBI circular"
        assert f["label"] in ("potential conduct concern — human review required",
                              "claim of police, arrest or court action — verify")
        assert f["detector"] in ("rules", "LLM", "both")


def test_sample_d_coverage_check_fires_and_recovers(result_d):
    texts = [w["text"] for w in result_d["coverage_warnings"]]
    assert any("re-transcribed in shorter pieces — listen and review" in t for t in texts)
    assert any(re.match(r"untranscribed speech at \d+:\d\d–\d+:\d\d — listen and review", t) for t in texts)
    joined = " ".join(x["text"] for x in result_d["transcript"])
    assert "बयालीस हज़ार" in joined   # the amount turn the long segment dropped comes back after re-splitting


def test_clean_sample_has_no_coverage_warning():
    assert run_sample("A")["coverage_warnings"] == []


def test_no_verdict_or_score_wording_anywhere(result_d):
    assert not FORBIDDEN.search(audit_json(result_d))
    for lang in ("hi-IN", "kn-IN", "en-IN"):
        flags = [{"concern_type": t} for t in ("intimidation_or_threats", "abusive_language",
                                               "third_party_contact_threat", "authority_or_legal_action_claim")]
        text = summary_text(flags, [{"x": 1}], {"outside_window": True}, lang)
        assert not FORBIDDEN.search(text) and not re.search(r"\d", text)   # numbers are spoken words


class FailingBackend:
    name = "failing"

    def complete(self, system, user):
        from src.llm_tagger import LLMError
        raise LLMError("HTTP 503 service_unavailable")


class SlowBackend:
    name = "slow"

    def complete(self, system, user):
        time.sleep(3)
        return '{"flags": []}'


@pytest.mark.parametrize("backend, status", [(FailingBackend(), "error"), (SlowBackend(), "timeout"), (None, "disabled")])
def test_llm_failure_falls_back_to_rules_with_notice(backend, status):
    s = SAMPLES["C"]
    t0 = time.perf_counter()
    r = analyse(ROOT / s["audio"], "hi-IN", mock_services(reference=s["reference_turns"], llm_backend=backend),
                llm_timeout_s=0.5, make_summary_audio=False)
    assert time.perf_counter() - t0 < 5                       # never hangs on the advanced check
    assert r["detectors"]["llm_status"].startswith(status)
    assert r["notices"] == [LLM_UNAVAILABLE_NOTICE]
    assert all(f["detector"] == "rules" for f in r["flags"])
    assert r["flags"]                                          # rules still find the family threat in C


def test_too_long_and_unreadable_audio_rejected(tmp_path):
    s = SAMPLES["A"]
    with pytest.raises(AnalysisError, match="limit"):
        analyse(ROOT / s["audio"], "en-IN", mock_services(), max_duration_s=10)
    bad = tmp_path / "x.mp3"
    bad.write_bytes(b"not audio")
    with pytest.raises(AnalysisError, match="could not read"):
        analyse(bad, "en-IN", mock_services())
    with pytest.raises(AnalysisError, match="unsupported language"):
        analyse(ROOT / s["audio"], "ta-IN", mock_services())


def test_upload_without_reference_and_out_of_hours(tmp_path):
    up = tmp_path / "20261021_213000_call.wav"
    shutil.copyfile(ROOT / SAMPLES["A"]["audio"], up)
    r = analyse(up, "en-IN", mock_services(), filename=up.name)
    assert r["out_of_hours"]["checked"] and r["out_of_hours"]["outside_window"]
    assert "outside 08:00–19:00" in r["out_of_hours"]["note"]
    assert r["filename"] == up.name


def test_out_of_hours_parsing():
    assert out_of_hours("call.wav") == {"checked": False, "reason": "no YYYYMMDD_HHMMSS timestamp in the file name"}
    assert out_of_hours("20261021_101500.wav")["outside_window"] is False
    assert out_of_hours("x_20261021_190000.mp3")["outside_window"] is True       # 19:00 is outside
    assert out_of_hours("20261021_075959.wav")["outside_window"] is True
    assert out_of_hours("20261399_101500.wav")["checked"] is False               # invalid date
