import json

import pytest

from src.gnani_client import ConfigError, GnaniClient, _MockResponse, load_api_key

SECRET = "sk-test-SECRET-1234"


class ScriptedSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, url, **kwargs):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _err(status, typ):
    body = json.dumps({"success": False, "error": {"type": typ, "message": "m"}}).encode()
    return _MockResponse(status, body, "application/json")


def _client(responses):
    c = GnaniClient(api_key=SECRET, session=ScriptedSession(responses), backoff_base_s=0)
    c.sleep = lambda s: None
    return c


def _wav(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"RIFF0000WAVE")
    return p


def test_retries_on_429_then_succeeds(tmp_path):
    ok = _MockResponse(200, json.dumps({"success": True, "request_id": "r1", "transcript": "hi"}).encode(),
                       "application/json")
    c = _client([_err(429, "RATE_LIMIT_ERROR"), _err(503, "SERVICE_UNAVAILABLE"), ok])
    res = c.transcribe(_wav(tmp_path), "hi-IN")
    assert res.ok and res.attempts == 3 and res.request_id == "r1"


def test_no_retry_on_400_or_403(tmp_path):
    for status in (400, 403):
        c = _client([_err(status, "X")])
        res = c.transcribe(_wav(tmp_path), "hi-IN")
        assert not res.ok and res.attempts == 1 and res.http_status == status
        assert res.error == {"type": "X", "message": "m"}


def test_gives_up_after_four_tries(tmp_path):
    c = _client([_err(500, "API_ERROR")] * 4)
    res = c.transcribe(_wav(tmp_path), "kn-IN")
    assert not res.ok and res.attempts == 4 and c.session.calls == 4


def test_tts_returns_raw_bytes_and_rejects_json_200():
    c = _client([_MockResponse(200, b"RIFFdata", "audio/wav")])
    assert c.synthesize("namaste", "Poorvi", "hi-en").audio == b"RIFFdata"
    c = _client([_MockResponse(200, b'{"oops":1}', "application/json")])
    assert not c.synthesize("namaste", "Poorvi", "hi-en").ok


def test_key_never_in_repr_or_errors(tmp_path):
    import requests
    c = _client([requests.ConnectionError("boom")] * 4)
    res = c.transcribe(_wav(tmp_path), "hi-IN")
    assert SECRET not in repr(c)
    assert SECRET not in json.dumps(res.to_dict())


def test_rejects_invented_language_code(tmp_path):
    with pytest.raises(ValueError):
        _client([]).transcribe(_wav(tmp_path), "hi-en")


def test_load_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nGNANI_API_KEY="abc123"\n')
    assert load_api_key(env) == "abc123"
    with pytest.raises(ConfigError):
        load_api_key(tmp_path / "missing.env")
