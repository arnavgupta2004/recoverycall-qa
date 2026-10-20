"""STT result cache: the real key function and cmd_transcribe's caching policy, driven through the mock."""

import json
from argparse import Namespace

import numpy as np
import pytest

import run_spike
from src import audio
from src.gnani_client import STT_URL, GnaniClient, MockSession, _MockResponse

SCENARIO = "S_test"


def _write_wav(path, seed):
    audio.write_wav(path, np.random.default_rng(seed).integers(-2000, 2000, 8000, dtype=np.int16), 8000)


# ---------------------------------------------------------------- stt_cache_path

def test_cache_key_changes_when_bytes_change_at_same_path(tmp_path):
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, seed=1)
    first, digest1 = run_spike.stt_cache_path(tmp_path, wav, "hi-IN", "verbatim")
    _write_wav(wav, seed=2)  # regenerated audio, same path
    second, digest2 = run_spike.stt_cache_path(tmp_path, wav, "hi-IN", "verbatim")
    assert digest1 != digest2
    assert first != second


def test_cache_key_ignores_path_and_separates_language_and_format(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "sub_b.wav"
    _write_wav(a, seed=3)
    b.write_bytes(a.read_bytes())
    key_a, _ = run_spike.stt_cache_path(tmp_path, a, "hi-IN", "verbatim")
    assert run_spike.stt_cache_path(tmp_path, b, "hi-IN", "verbatim")[0] == key_a
    assert run_spike.stt_cache_path(tmp_path, a, "en-IN", "verbatim")[0] != key_a
    assert run_spike.stt_cache_path(tmp_path, a, "hi-IN", "transcribe")[0] != key_a


# ---------------------------------------------------------------- cmd_transcribe policy

class FixedStatusSession(MockSession):
    """MockSession whose STT endpoint always answers with one HTTP error."""

    def __init__(self, status, error_type):
        super().__init__()
        self.status, self.error_type = status, error_type

    def post(self, url, **kwargs):
        if url != STT_URL:
            return super().post(url, **kwargs)
        self.calls["stt"] += 1
        body = {"success": False, "error": {"type": self.error_type, "message": "test"}}
        return _MockResponse(self.status, json.dumps(body).encode(), "application/json")


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A one-scenario, one-chunk project under tmp_path with every chunk file cmd_transcribe needs."""
    monkeypatch.setattr(run_spike, "ROOT", tmp_path)
    monkeypatch.setattr(run_spike, "load_scripts", lambda: {"scenarios": [{"id": SCENARIO}]})
    monkeypatch.setattr(run_spike, "log", lambda msg: None)
    paths = run_spike.Paths(mock=True)
    sdir = paths.scenario_dir(SCENARIO)
    files = {}
    for i, v in enumerate(run_spike.VARIANTS + ["phone_ulaw_container"]):
        f = sdir / "chunks" / f"{v}_c0.wav"
        f.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(f, seed=10 + i)
        files[v] = str(f.relative_to(tmp_path))
    manifest = {
        "scenario": SCENARIO, "stt_language_codes": ["hi-IN"], "primary_stt_language_code": "hi-IN",
        "max_chunk_s": 28.0,
        "turns": [{"turn": 0, "text": "नमस्ते जी", "critical_spans": []}],
        "chunks": [{"chunk": 0, "turns": [0], "start_s": 0.0, "end_s": 1.0, "files": files}],
        "variants": {v: {"snr": None} for v in run_spike.VARIANTS + ["phone_ulaw_container"]},
    }
    (sdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return paths


def _client(session):
    c = GnaniClient(api_key="test-key", session=session, backoff_base_s=0)
    c.sleep = lambda s: None
    return c


def _results(paths):
    return json.loads(paths.results.read_text(encoding="utf-8"))


def test_403_is_not_cached_and_is_retried_next_run(project):
    session = FixedStatusSession(403, "FORBIDDEN")
    run_spike.cmd_transcribe(Namespace(force=False), project, _client(session))
    results = _results(project)
    assert results and all(r["http_status"] == 403 for r in results)
    assert list(project.stt_cache.glob("*.json")) == []
    first_calls = session.calls["stt"]
    assert first_calls == len(results)  # 403 is not retried within a run

    run_spike.cmd_transcribe(Namespace(force=False), project, _client(session))
    assert session.calls["stt"] == 2 * first_calls  # nothing came from cache
    assert not any(r["from_cache"] for r in _results(project))


def test_400_is_cached_and_reported_from_cache(project):
    session = FixedStatusSession(400, "INVALID_REQUEST_ERROR")
    run_spike.cmd_transcribe(Namespace(force=False), project, _client(session))
    n = len(_results(project))
    assert len(list(project.stt_cache.glob("*.json"))) == n

    run_spike.cmd_transcribe(Namespace(force=False), project, _client(session))
    assert session.calls["stt"] == n  # second run made no calls
    assert all(r["from_cache"] and r["http_status"] == 400 for r in _results(project))


def test_success_is_cached_and_regenerated_audio_misses_cache(project):
    session = MockSession()
    client = _client(session)
    run_spike.cmd_transcribe(Namespace(force=False), project, client)
    n = len(_results(project))
    assert all(r["http_status"] == 200 for r in _results(project))
    calls_after_first = session.calls["stt"]  # includes the mock's one scripted 429 retry

    run_spike.cmd_transcribe(Namespace(force=False), project, client)
    assert session.calls["stt"] == calls_after_first
    assert all(r["from_cache"] for r in _results(project))

    # Regenerate one chunk at the same path: only that request goes back to the API.
    clean = run_spike.ROOT / json.loads(
        (project.scenario_dir(SCENARIO) / "manifest.json").read_text(encoding="utf-8"))["chunks"][0]["files"]["clean"]
    _write_wav(clean, seed=99)
    run_spike.cmd_transcribe(Namespace(force=False), project, client)
    fresh = [r for r in _results(project) if not r["from_cache"]]
    assert {r["variant"] for r in fresh} == {"clean"}
    assert len(fresh) == 2  # clean verbatim + clean ITN
    assert session.calls["stt"] == calls_after_first + 2
    assert n == len(_results(project))
