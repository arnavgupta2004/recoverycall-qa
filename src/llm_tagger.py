"""LLM conduct-concern tagger (Gemini API free tier) with a verbatim-quote gate, cache and mock.

    from src.llm_tagger import make_backend, tag
    backend = make_backend(mock=False, env_file=Path(".env"))
    result = tag(segments, "hi-IN", backend, cache_dir=Path("outputs/llm_cache"))
    result["flags"]      # verified flags, same fields and labels as src.classifier flags (+ "source": "llm")
    result["rejected"]   # model outputs dropped by the quote verifier, with the reason

The model sees the four category definitions and the call's segments (numbered by segment_id)
and must return JSON {"flags": [{"segment_id", "type", "quote"}]} at temperature 0. Every
returned quote is passed through src.classifier.verify_quote: a quote that is not a verbatim
substring of the named segment is dropped, never repaired. The LLM output is a tag, never a
verdict: flags carry the same labels and wording as the rules classifier.

API notes: the request uses the Gemini Interactions API as documented on 2026-10-21
(POST /v1beta/interactions, x-goog-api-key header, response_format JSON schema). The key
travels only in that header, never in a URL, log line or error message. Response parsing
is defensive (output_text, steps/outputs text parts, or generateContent candidates). The
live path was written without a key to test against; check the first real response.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from src.classifier import verify_quote
from src.concern_rules import ABUSIVE, AUTHORITY, CATEGORY_LABELS, INTIMIDATION, REGULATORY_NOTE, THIRD_PARTY
from src.gnani_client import load_api_key

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "2026-10-21.v1"
CONNECT_TIMEOUT_S, READ_TIMEOUT_S = 10, 60
MAX_TRIES = 4
RETRY_STATUSES = {429, 500, 503}

CATEGORIES = [INTIMIDATION, ABUSIVE, THIRD_PARTY, AUTHORITY]
DEFINITIONS = {
    INTIMIDATION: "The agent threatens to visit, harm or shame the borrower in person: sending people to the "
                  "borrower's home or door, sitting or standing outside it, physical threats.",
    ABUSIVE: "The agent uses slurs, insults, or derogatory or shaming words directed at the borrower "
             "(e.g. calling them a cheat, fraud, shameless, stupid).",
    THIRD_PARTY: "The agent threatens to tell or show the borrower's situation to family, employer, colleagues, "
                 "neighbours, a child's school, or phone or WhatsApp contacts.",
    AUTHORITY: "The agent mentions police, arrest, FIR, court, a case or a legal notice, whether or not the claim "
               "is lawful. This is a claim to verify, not a finding.",
}
NOT_CONCERNS = ("Firm but lawful statements are not conduct concerns: payment reminders and deadlines, late fees, "
                "credit score or CIBIL impact, internal escalation or review, contacting a named guarantor or "
                "co-borrower under the loan agreement, offers made with the borrower's consent. Only tag "
                "authority_or_legal_action_claim for those if they mention police, arrest, FIR, court, a case or a "
                "legal notice.")

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"flags": {"type": "array", "items": {
        "type": "object",
        "properties": {"segment_id": {"type": "string"}, "type": {"type": "string", "enum": CATEGORIES},
                       "quote": {"type": "string"}},
        "required": ["segment_id", "type", "quote"]}}},
    "required": ["flags"],
}


def build_prompt(segments: list[dict], language: str) -> tuple[str, str]:
    defs = "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    system = (
        "You tag passages in a debt-collection call transcript for human review. You do not judge legality, "
        "give verdicts or scores.\n\nCategories:\n" + defs + "\n\n" + NOT_CONCERNS + "\n\n"
        "Rules: tag only what the transcript says. One passage may carry more than one category. Each quote must "
        "be copied character for character from ONE segment's text (same script, spelling and punctuation; do not "
        "translate, transliterate, fix or join segments) and be as short as possible while containing the concern. "
        "Return JSON {\"flags\": [{\"segment_id\": ..., \"type\": ..., \"quote\": ...}]}; return {\"flags\": []} if "
        "nothing applies.")
    lines = [f"[{s['segment_id']}] {s['text']}" for s in segments]
    user = f"Language code: {language}\nTranscript segments:\n" + "\n".join(lines)
    return system, user


def extract_text(resp: Any) -> str | None:
    """Find the model's JSON text in an Interactions or generateContent response."""
    if isinstance(resp, dict):
        if isinstance(resp.get("output_text"), str):
            return resp["output_text"]
        for c in resp.get("candidates", []) or []:
            for part in ((c.get("content") or {}).get("parts") or []):
                if isinstance(part.get("text"), str):
                    return part["text"]
    texts: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if x.get("type") in (None, "text", "output_text") and isinstance(x.get("text"), str):
                texts.append(x["text"])
            for k, v in x.items():
                if k not in ("text",) and k not in ("thought", "thoughts", "thinking", "thought_summary"):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(resp)
    for t in reversed(texts):  # the answer comes after any thought summaries
        try:
            json.loads(t)
            return t
        except ValueError:
            continue
    return texts[-1] if texts else None


class LLMError(RuntimeError):
    pass


@dataclass
class GeminiBackend:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    min_interval_s: float = 4.5          # stay under free-tier per-minute limits
    read_timeout_s: float = READ_TIMEOUT_S
    max_tries: int = MAX_TRIES
    session: Any = field(default_factory=requests.Session, repr=False)
    sleep = staticmethod(time.sleep)
    _last: float = 0.0
    events: list = field(default_factory=list)   # non-200 outcomes per attempt, e.g. "429 rate_limit_exceeded"
    calls: int = 0

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "system_instruction": system, "input": user,
                "generation_config": {"temperature": 0, "thinking_level": "minimal", "max_output_tokens": 2048},
                "response_format": {"type": "text", "mime_type": "application/json", "schema": RESPONSE_SCHEMA}}
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        for attempt in range(1, self.max_tries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                resp = self.session.post(API_URL, headers=headers, data=json.dumps(body, ensure_ascii=False).encode(),
                                         timeout=(CONNECT_TIMEOUT_S, self.read_timeout_s))
            except requests.RequestException as exc:
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
                status = None
            else:
                if resp.status_code == 200:
                    text = extract_text(resp.json())
                    if text is None:
                        raise LLMError(f"no text in 200 response: {json.dumps(resp.json())[:300]}")
                    return text
                status = resp.status_code
                try:
                    e = resp.json().get("error", {})
                    err = f"HTTP {status} {e.get('code', '')} {str(e.get('message', ''))[:300]}"
                except ValueError:
                    err = f"HTTP {status} {resp.text[:300]}"
            self.events.append(err[:120])
            if (status is not None and status not in RETRY_STATUSES) or attempt == self.max_tries:
                raise LLMError(err)
            self.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)) * (1 + random.random() * 0.25))
        raise LLMError("unreachable")


# Mock cues: (category, cue substrings). The mock quotes a short verbatim window around the
# first cue and deliberately respells every fifth quote so the verifier path is exercised.
_MOCK_CUES = [
    (AUTHORITY, ["पुलिस", "एफ़आईआर", "FIR", "कोर्ट", "नोटिस", "police", "court", "arrest", "ಪೊಲೀಸ್", "ಅರೆಸ್ಟ್",
                 "ಕೋರ್ಟ್", "ನೋಟಿಸ್"]),
    (THIRD_PARTY, ["पड़ोस", "ऑफ़िस", "ऑफिस", "दफ़्तर", "मैनेजर", "भाई", "HR", "wife", "family", "neighbours",
                   "employer", "boss", "WhatsApp", "ಆಫೀಸ", "ಹೆಂಡತಿ", "ನೆರೆ", "ಶಾಲೆ"]),
    (INTIMIDATION, ["घर आ", "दरवाज़े", "घर के बाहर", "ghar aa", "ghar ke bahar", "ಮನೆಗೆ ಬರ", "ಮನೆ ಬಾಗಿಲ"]),
    (ABUSIVE, ["बेशर्म", "बेईमान", "bewakoof", "cheater", "napake", "ಮೋಸಗಾರ", "ನಾಚಿಕೆ", "ಮೋಸದ"]),
]


@dataclass
class MockBackend:
    name: str = "mock"
    calls: int = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        flags, n = [], 0
        for line in user.splitlines():
            if not line.startswith("["):
                continue
            sid, _, text = line[1:].partition("] ")
            for cat, cues in _MOCK_CUES:
                for cue in cues:
                    i = text.find(cue)
                    if i < 0:
                        continue
                    quote = text[max(0, i - 8): i + len(cue) + 12].strip()
                    n += 1
                    if n % 5 == 0:
                        quote = quote.replace(cue, cue.lower() if cue.lower() != cue else cue + "्")  # respelled
                    flags.append({"segment_id": sid, "type": cat, "quote": quote})
                    break
        return json.dumps({"flags": flags}, ensure_ascii=False)


def make_backend(mock: bool, env_file: Path | None = None, **overrides):
    """Mock or Gemini backend; overrides (e.g. read_timeout_s, max_tries, min_interval_s) apply to Gemini."""
    if mock:
        return MockBackend()
    return GeminiBackend(api_key=load_api_key(env_file, "GEMINI_API_KEY"),
                         model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL), **overrides)


def _cache_key(backend_name: str, language: str, segments: list[dict]) -> str:
    payload = json.dumps({"backend": backend_name, "prompt": PROMPT_VERSION, "language": language,
                          "segments": [[s["segment_id"], s["text"]] for s in segments]}, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def tag(segments: list[dict], language: str, backend, cache_dir: Path | None = None) -> dict:
    """Tag one call. Returns {"flags", "rejected", "from_cache", "error"}; errors never raise."""
    if not segments:
        return {"flags": [], "rejected": [], "from_cache": False, "error": None}
    cache_file = cache_dir / f"{_cache_key(backend.name, language, segments)}.json" if cache_dir else None
    from_cache, error, text = False, None, None
    if cache_file and cache_file.exists():
        text = json.loads(cache_file.read_text(encoding="utf-8"))["response_text"]
        from_cache = True
    else:
        system, user = build_prompt(segments, language)
        try:
            text = backend.complete(system, user)
        except LLMError as exc:
            error = str(exc)
        if text is not None and cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"backend": backend.name, "prompt_version": PROMPT_VERSION,
                                              "language": language, "response_text": text}, ensure_ascii=False,
                                             indent=2), encoding="utf-8")
    flags, rejected = [], []
    if text is not None:
        try:
            items = json.loads(text).get("flags", [])
        except (ValueError, AttributeError):
            items, error = [], f"response is not JSON with a flags list: {text[:200]!r}"
        by_id = {s["segment_id"]: s for s in segments}
        for it in items if isinstance(items, list) else []:
            sid, typ, quote = it.get("segment_id"), it.get("type"), it.get("quote")
            seg = by_id.get(sid)
            reason = ("unknown category" if typ not in CATEGORIES else
                      "unknown segment_id" if seg is None else
                      "quote not verbatim in segment" if not verify_quote(quote, seg["text"]) else None)
            if reason:
                rejected.append({"segment_id": sid, "type": typ, "quote": quote, "reason": reason})
                continue
            flags.append({"concern_type": typ, "quote": quote, "segment_id": sid, "start_s": seg["start_s"],
                          "end_s": seg["end_s"], "label": CATEGORY_LABELS[typ], "regulatory_note": REGULATORY_NOTE,
                          "source": "llm"})
    return {"flags": flags, "rejected": rejected, "from_cache": from_cache, "error": error}
