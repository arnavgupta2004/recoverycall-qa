import json

import pytest

from src.concern_rules import ABUSIVE, AUTHORITY, CATEGORY_LABELS, INTIMIDATION, REGULATORY_NOTE, THIRD_PARTY
from src.gnani_client import _MockResponse
from src.hybrid import merge_flags
from src.llm_tagger import GeminiBackend, LLMError, MockBackend, extract_text, tag

SEGS = [
    {"segment_id": "turn00", "text": "Namaste, main Apex Recovery se hoon.", "start_s": 0.0, "end_s": 6.0},
    {"segment_id": "turn01", "text": "Hum aapke employer ko letter bhejenge, aur police complaint karenge.", "start_s": 6.0, "end_s": 12.0},
]


class FixedBackend:
    name = "fixed"

    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def complete(self, system, user):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


def test_verifier_keeps_verbatim_and_drops_everything_else(tmp_path):
    backend = FixedBackend({"flags": [
        {"segment_id": "turn01", "type": THIRD_PARTY, "quote": "employer ko letter bhejenge"},   # verbatim
        {"segment_id": "turn01", "type": AUTHORITY, "quote": "police complaint karegi"},        # respelled
        {"segment_id": "turn00", "type": THIRD_PARTY, "quote": "employer ko letter bhejenge"},  # wrong segment
        {"segment_id": "turn09", "type": AUTHORITY, "quote": "police"},                         # unknown segment
        {"segment_id": "turn01", "type": "risk_score", "quote": "police"},                      # unknown category
    ]})
    res = tag(SEGS, "en-IN", backend, tmp_path)
    assert [(f["concern_type"], f["quote"]) for f in res["flags"]] == [(THIRD_PARTY, "employer ko letter bhejenge")]
    f = res["flags"][0]
    assert f["label"] == CATEGORY_LABELS[THIRD_PARTY] and f["regulatory_note"] == REGULATORY_NOTE
    assert f["start_s"] == 6.0 and f["end_s"] == 12.0
    reasons = sorted(r["reason"] for r in res["rejected"])
    assert reasons == ["quote not verbatim in segment", "quote not verbatim in segment", "unknown category",
                       "unknown segment_id"]


def test_cache_hit_makes_no_call(tmp_path):
    backend = FixedBackend({"flags": [{"segment_id": "turn01", "type": AUTHORITY, "quote": "police complaint"}]})
    first = tag(SEGS, "en-IN", backend, tmp_path)
    second = tag(SEGS, "en-IN", backend, tmp_path)
    assert backend.calls == 1 and not first["from_cache"] and second["from_cache"]
    assert first["flags"] == second["flags"]


def test_malformed_response_is_an_error_not_a_crash(tmp_path):
    class Bad:
        name = "bad"

        def complete(self, s, u):
            return "not json"
    res = tag(SEGS, "en-IN", Bad(), None)
    assert res["flags"] == [] and "not JSON" in res["error"]


def test_mock_backend_quotes_are_mostly_verbatim(tmp_path):
    res = tag(SEGS, "en-IN", MockBackend(), tmp_path)
    assert {f["concern_type"] for f in res["flags"]} >= {THIRD_PARTY, AUTHORITY}


@pytest.mark.parametrize("resp", [
    {"output_text": '{"flags": []}'},
    {"steps": [{"type": "thought", "text": "thinking..."}, {"type": "model_output", "content": [{"type": "text", "text": '{"flags": []}'}]}]},
    {"candidates": [{"content": {"parts": [{"text": '{"flags": []}'}]}}]},
])
def test_extract_text_shapes(resp):
    assert json.loads(extract_text(resp)) == {"flags": []}


class ScriptedSession:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def post(self, url, headers=None, data=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "body": json.loads(data)})
        return self.responses.pop(0)


SECRET = "gm-test-SECRET-9876"


def _backend(responses):
    b = GeminiBackend(api_key=SECRET, session=ScriptedSession(responses), min_interval_s=0)
    b.sleep = lambda s: None
    return b


def test_gemini_backend_retries_429_and_sends_key_only_in_header():
    ok = _MockResponse(200, json.dumps({"output_text": '{"flags": []}'}).encode(), "application/json")
    limited = _MockResponse(429, b'{"error": {"code": "rate_limit_exceeded", "message": "slow down"}}', "application/json")
    b = _backend([limited, ok])
    assert b.complete("sys", "user") == '{"flags": []}'
    reqs = b.session.requests
    assert len(reqs) == 2
    assert reqs[0]["headers"]["x-goog-api-key"] == SECRET and SECRET not in reqs[0]["url"]
    assert reqs[0]["body"]["generation_config"]["temperature"] == 0
    assert SECRET not in json.dumps(reqs[0]["body"]) and SECRET not in repr(b)


def test_gemini_backend_does_not_retry_400_and_error_has_no_key():
    bad = _MockResponse(400, b'{"error": {"code": "invalid_request", "message": "bad schema"}}', "application/json")
    b = _backend([bad])
    with pytest.raises(LLMError) as exc:
        b.complete("sys", "user")
    assert "400" in str(exc.value) and SECRET not in str(exc.value)
    assert len(b.session.requests) == 1


# ---------------------------------------------------------------- hybrid merge

def _f(typ, seg, quote, source=None):
    d = {"concern_type": typ, "segment_id": seg, "quote": quote, "start_s": 0.0, "end_s": 6.0,
         "label": CATEGORY_LABELS[typ], "regulatory_note": REGULATORY_NOTE}
    if source:
        d["source"] = source
    return d


def test_hybrid_dedupes_overlapping_same_type_and_keeps_rules_quote():
    rules = [_f(AUTHORITY, "turn01", "police")]
    llm = [_f(AUTHORITY, "turn01", "police complaint karenge", "llm")]
    merged = merge_flags(rules, llm, SEGS)
    assert len(merged) == 1 and merged[0]["quote"] == "police" and merged[0]["source"] == "rules+llm"


def test_hybrid_keeps_different_types_on_same_text_and_non_overlapping_spans():
    rules = [_f(THIRD_PARTY, "turn01", "employer ko letter")]
    llm = [_f(INTIMIDATION, "turn01", "employer ko letter", "llm"),
           _f(THIRD_PARTY, "turn01", "police complaint", "llm"),
           _f(ABUSIVE, "turn00", "Apex Recovery", "llm")]
    merged = merge_flags(rules, llm, SEGS)
    assert len(merged) == 4
    assert {m["label"] for m in merged} <= set(CATEGORY_LABELS.values())
