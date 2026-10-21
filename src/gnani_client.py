"""Thin client for Gnani Timbre (TTS) and Prisma (STT) REST APIs, plus an offline mock.

The API key is held privately and only ever placed in the request header. Nothing in
this module logs, prints, or returns request headers; errors carry status, error type
and error message from the response body only.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import struct
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import requests

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"

STT_LANGUAGE_CODES = {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"}
STT_FORMATS = {"verbatim", "transcribe"}
TTS_SPEED_RANGE = (0.85, 1.15)

CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 60
MAX_TRIES = 4
RETRY_STATUSES = {429, 500, 503}

DEFAULT_AUDIO_CONFIG = {
    "sample_rate": 16000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav",
}


class ConfigError(RuntimeError):
    pass


def load_api_key(env_file: Path | None = None, var: str = "GNANI_API_KEY") -> str:
    """Return the key named `var` from the environment, falling back to a .env file. Never logged."""
    key = os.environ.get(var, "").strip()
    if not key and env_file is not None and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ").strip() == var:
                key = value.strip().strip('"').strip("'")
                break
    if not key:
        raise ConfigError(
            f"{var} is not set. Export it or put it in .env (see .env.example), "
            "or run with --mock to test offline."
        )
    return key


@dataclass
class ApiResult:
    ok: bool
    http_status: int | None
    latency_ms: float | None
    attempts: int
    request_id: str | None = None
    transcript: str | None = None
    audio: bytes | None = None
    content_type: str | None = None
    error: dict | None = None  # {"type": ..., "message": ...}

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "audio"}
        return d


def _parse_error(resp) -> dict:
    try:
        body = resp.json()
    except ValueError:
        text = (resp.text or "")[:300]
        return {"type": f"HTTP_{resp.status_code}", "message": text or "non-JSON error body"}
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return {"type": str(err.get("type")), "message": str(err.get("message"))}
    # Unexpected shape (e.g. FastAPI "detail"); keep the body, it never contains headers.
    return {"type": f"HTTP_{resp.status_code}", "message": json.dumps(body, ensure_ascii=False)[:300]}


@dataclass
class GnaniClient:
    api_key: str = field(repr=False)
    session: object = field(default_factory=requests.Session, repr=False)
    backoff_base_s: float = 1.0
    sleep = staticmethod(time.sleep)

    def _post(self, url: str, **kwargs) -> tuple[object | None, float | None, int, dict | None]:
        """POST with retries. Returns (response, latency_ms of last attempt, attempts, transport_error)."""
        headers = {"X-API-Key-ID": self.api_key, **kwargs.pop("headers", {})}
        files_factory = kwargs.pop("files_factory", None)
        resp, latency, transport_error = None, None, None
        for attempt in range(1, MAX_TRIES + 1):
            if files_factory is not None:
                kwargs["files"] = files_factory()
            t0 = time.perf_counter()
            try:
                resp = self.session.post(
                    url, headers=headers, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S), **kwargs
                )
                transport_error = None
            except requests.RequestException as exc:
                # The exception type and message are enough; never include the request object.
                resp, transport_error = None, {"type": type(exc).__name__, "message": str(exc)[:300]}
            finally:
                latency = (time.perf_counter() - t0) * 1000.0
            retryable = transport_error is not None or (resp is not None and resp.status_code in RETRY_STATUSES)
            if not retryable or attempt == MAX_TRIES:
                return resp, latency, attempt, transport_error
            delay = self.backoff_base_s * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            if resp is not None:
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, min(float(retry_after), 30.0))
            self.sleep(delay)
        return resp, latency, MAX_TRIES, transport_error  # unreachable

    def synthesize(self, text: str, voice: str, language: str, model: str = "timbre-v2.5",
                   speed: float = 1.0, audio_config: dict | None = None) -> ApiResult:
        if not TTS_SPEED_RANGE[0] <= speed <= TTS_SPEED_RANGE[1]:
            raise ValueError(f"speed {speed} outside {TTS_SPEED_RANGE}")
        body = {
            "text": text,
            "voice": voice,
            "model": model,
            "language": language,
            "speed": speed,
            "audio_config": audio_config or DEFAULT_AUDIO_CONFIG,
        }
        resp, latency, attempts, terr = self._post(
            TTS_URL, headers={"Content-Type": "application/json"},
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        if resp is None:
            return ApiResult(False, None, latency, attempts, error=terr)
        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and "json" not in ctype.lower() and resp.content:
            return ApiResult(True, 200, latency, attempts, audio=resp.content, content_type=ctype,
                             request_id=resp.headers.get("X-Request-ID"))
        if resp.status_code == 200:
            return ApiResult(False, 200, latency, attempts, content_type=ctype,
                             error={"type": "UNEXPECTED_200_BODY",
                                    "message": f"expected raw audio, got Content-Type={ctype!r}, {len(resp.content)} bytes"})
        return ApiResult(False, resp.status_code, latency, attempts, content_type=ctype, error=_parse_error(resp))

    def transcribe(self, audio_path: Path, language_code: str, fmt: str = "verbatim") -> ApiResult:
        if language_code not in STT_LANGUAGE_CODES:
            raise ValueError(f"unsupported STT language_code {language_code!r}")
        if fmt not in STT_FORMATS:
            raise ValueError(f"unsupported STT format {fmt!r}")
        audio_bytes = Path(audio_path).read_bytes()

        def files_factory():
            return {"audio_file": (Path(audio_path).name, io.BytesIO(audio_bytes), "audio/wav")}

        resp, latency, attempts, terr = self._post(
            STT_URL, files_factory=files_factory, data={"language_code": language_code, "format": fmt}
        )
        if resp is None:
            return ApiResult(False, None, latency, attempts, error=terr)
        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                return ApiResult(False, 200, latency, attempts,
                                 error={"type": "NON_JSON_200", "message": (resp.text or "")[:300]})
            if body.get("success") is True:
                # transcript may be None (silence) or "" (empty audio); treat both as ok.
                transcript = body.get("transcript") or ""
                return ApiResult(True, 200, latency, attempts, request_id=body.get("request_id"),
                                 transcript=transcript)
            return ApiResult(False, 200, latency, attempts, request_id=body.get("request_id"),
                             error={"type": "UNEXPECTED_200_BODY",
                                    "message": json.dumps(body, ensure_ascii=False)[:300]})
        return ApiResult(False, resp.status_code, latency, attempts, error=_parse_error(resp))


# --------------------------------------------------------------------------- mock


class _MockResponse:
    def __init__(self, status_code: int, content: bytes, content_type: str, headers: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type, **(headers or {})}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.content.decode("utf-8"))


def _tone_wav(seconds: float, sample_rate: int, freq: float) -> bytes:
    n = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            # Amplitude-modulated tone so silence detection / SNR maths sees "speech-like" energy.
            env = 0.6 + 0.4 * math.sin(2 * math.pi * 3.0 * i / sample_rate)
            frames += struct.pack("<h", int(8000 * env * math.sin(2 * math.pi * freq * i / sample_rate)))
        w.writeframes(bytes(frames))
    return buf.getvalue()


_DEVANAGARI_FILLER = ["नमस्ते", "जी", "पैसे", "हाँ", "ठीक", "है"]


class MockSession:
    """Stands in for requests.Session. Fakes both endpoints at the HTTP level so the real
    client code (retries, error parsing, raw-bytes handling) is exercised offline.

    STT needs to know what the audio "says": the runner registers each upload's reference
    text by file content hash. The fake transcript is the reference, lightly corrupted
    (more for noisier variants). A Roman-script reference sent with hi-IN comes back in
    Devanagari so the script-mismatch path is exercised. The first STT call returns 429
    once to exercise retry.
    """

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.references: dict[str, tuple[str, float]] = {}
        self.calls = {"tts": 0, "stt": 0}
        self._first_stt = True

    def register_reference(self, audio_path: Path, text: str, corruption: float) -> None:
        digest = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
        self.references[digest] = (text, corruption)

    def post(self, url, headers=None, timeout=None, data=None, files=None, **_):
        if "X-API-Key-ID" not in (headers or {}):
            return _MockResponse(403, b'{"success":false,"error":{"type":"FORBIDDEN","message":"missing key"}}',
                                 "application/json")
        if url == TTS_URL:
            self.calls["tts"] += 1
            body = json.loads(data.decode("utf-8"))
            sr = body["audio_config"]["sample_rate"]
            seconds = min(12.0, 0.6 + 0.065 * len(body["text"]))
            freq = 180.0 + (sum(map(ord, body["voice"])) % 120)
            return _MockResponse(200, _tone_wav(seconds, sr, freq), "audio/wav")
        if url == STT_URL:
            self.calls["stt"] += 1
            if self._first_stt:
                self._first_stt = False
                return _MockResponse(429, b'{"success":false,"error":{"type":"RATE_LIMIT_ERROR",'
                                          b'"message":"Rate limit exceeded. Please try again later."}}',
                                     "application/json")
            _, fobj, _ = files["audio_file"]
            raw = fobj.read()
            digest = hashlib.sha256(raw).hexdigest()
            if digest not in self.references:
                return _MockResponse(400, b'{"success":false,"error":{"type":"INVALID_REQUEST_ERROR",'
                                          b'"message":"mock: unknown audio"}}', "application/json")
            ref, corruption = self.references[digest]
            hyp = self._corrupt(ref, corruption, data["language_code"], digest)
            out = {"success": True, "request_id": "mock_" + digest[:12],
                   "timestamp": time.strftime("%Y%m%d_%H%M%S"), "transcript": hyp}
            return _MockResponse(200, json.dumps(out, ensure_ascii=False).encode("utf-8"), "application/json")
        return _MockResponse(404, b'{"success":false,"error":{"type":"NOT_FOUND","message":"mock"}}',
                             "application/json")

    def _corrupt(self, ref: str, corruption: float, lang: str, digest: str) -> str:
        rng = random.Random(f"{self.seed}:{digest}:{lang}")
        words = ref.split()
        latin = sum(c.isascii() and c.isalpha() for c in ref) > len(ref) / 3
        if lang == "hi-IN" and latin:
            return " ".join(rng.choice(_DEVANAGARI_FILLER) for _ in words)
        out = []
        for w in words:
            r = rng.random()
            if r < corruption * 0.4:
                continue  # deletion
            if r < corruption * 0.8:
                out.append(w[:-1] if len(w) > 2 else w + w)  # substitution
            else:
                out.append(w)
            if rng.random() < corruption * 0.1:
                out.append(w)  # insertion
        return " ".join(out)


def make_client(mock: bool, env_file: Path | None) -> GnaniClient:
    if mock:
        client = GnaniClient(api_key="mock-key", session=MockSession(), backoff_base_s=0.01)
        return client
    return GnaniClient(api_key=load_api_key(env_file))
