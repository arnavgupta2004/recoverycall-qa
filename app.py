"""RecoveryCall QA — web API + static single-page frontend (FastAPI, for Google Cloud Run).

    uvicorn app:app --port 8080                       # real APIs for live runs (keys from env / .env)
    RCQA_MOCK=1 UPLOAD_PASSCODE=local-test uvicorn app:app --port 8080   # live runs use mock STT/LLM/TTS

Endpoints
    GET  /                        the page (static/index.html)
    GET  /api/config              live analysis enabled?, limits, languages
    GET  /api/samples             the four bundled samples (cached results; no API calls)
    GET  /api/samples/{id}        cached result + audio URLs
    GET  /api/samples/{id}/audio  sample audio (WAV);  /summary: cached spoken summary (WAV)
    POST /api/analyse             multipart: language, passcode, audio file OR sample_id.
                                  Server-sent events: stage (transcribing, coverage, concerns,
                                  summary), then done (audit result) or error.

Safety: live analysis needs UPLOAD_PASSCODE (disabled when unset) and is rate-limited per client
IP (rightmost X-Forwarded-For hop); uploads are capped at 20 MB and 2 minutes, written only to a
per-request temporary directory that is deleted before the stream ends, and never cached. Logs
carry counts and timings only — no keys, transcript text, quotes or file names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import (LANGUAGES, LLM_UNAVAILABLE_NOTICE, AnalysisError, analyse, audit_json, live_services,
                          mock_services)
from src.webguard import (ALLOWED_EXTENSIONS, MAX_AUDIO_S, MAX_UPLOAD_BYTES, RateLimiter, client_ip,
                          passcode_configured, passcode_ok)
from src.webutil import analysis_log_line, wav_data_uri

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MOCK = os.environ.get("RCQA_MOCK") == "1"
SAMPLES = json.loads((ROOT / "samples" / "samples.json").read_text(encoding="utf-8"))
CACHE_DIR = ROOT / "samples" / "cache"          # real cached results; serving them makes no API calls
STAGES = ["transcribing", "coverage", "concerns", "summary"]
SAMPLE_CARDS = {"D": "Hindi · noisy phone line", "B": "Kannada", "C": "Hindi", "A": "Hinglish · compliant"}
LIVE_LIMIT = RateLimiter(limit=int(os.environ.get("LIVE_RUNS_PER_HOUR", "5")), window_s=3600)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rcqa")
log.info("cached samples available in samples/cache: %s",
         sorted(sid for sid in SAMPLES if (CACHE_DIR / f"{sid}.json").exists()) or "none")

app = FastAPI(title="RecoveryCall QA", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def config() -> dict:
    return {"live_enabled": passcode_configured(), "max_audio_s": MAX_AUDIO_S,
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "languages": [{"code": c, "name": n} for c, n in LANGUAGES.items()],
            "samples_for_live": [{"id": sid, "card": card} for sid, card in SAMPLE_CARDS.items()],
            "llm_unavailable_notice": LLM_UNAVAILABLE_NOTICE}


def _cached(sample_id: str) -> dict | None:
    f = CACHE_DIR / f"{sample_id}.json"
    return json.loads(f.read_text(encoding="utf-8")) if sample_id in SAMPLES and f.exists() else None


@app.get("/api/samples")
def samples() -> list[dict]:
    out = []
    for sid in SAMPLE_CARDS:
        r = _cached(sid)
        if r is None:
            continue
        out.append({"id": sid, "card": SAMPLE_CARDS[sid], "title": SAMPLES[sid]["title"], "language": r["language"],
                    "language_name": r["language_name"], "duration_s": r["duration_s"], "flags": len(r["flags"]),
                    "coverage_warnings": len(r["coverage_warnings"]),
                    # positions (fraction of the call) for the card sketch: flag pins and coverage gaps
                    "pins": [{"at": round(f["start_s"] / r["duration_s"], 3),
                              "tone": "violet" if f["concern_type"] == "authority_or_legal_action_claim" else "amber"}
                             for f in r["flags"]],
                    "gaps": [[round(w["start_s"] / r["duration_s"], 3), round(w["end_s"] / r["duration_s"], 3)]
                             for w in r["coverage_warnings"]]})
    return out


@app.get("/api/samples/{sample_id}")
def sample(sample_id: str):
    r = _cached(sample_id)
    if r is None:
        return _error(404, "unknown sample")
    log.info("cached example served: sample=%s api_calls=0", sample_id)
    return {"id": sample_id, "card": SAMPLE_CARDS.get(sample_id), "title": SAMPLES[sample_id]["title"],
            "cached": True, "result": r, "audio_url": f"/api/samples/{sample_id}/audio",
            "summary_audio_url": (f"/api/samples/{sample_id}/summary"
                                  if (CACHE_DIR / f"{sample_id}_summary.wav").exists() else None)}


@app.get("/api/samples/{sample_id}/audio")
def sample_audio(sample_id: str):
    if sample_id not in SAMPLES:
        return _error(404, "unknown sample")
    return FileResponse(ROOT / SAMPLES[sample_id]["audio"], media_type="audio/wav")


@app.get("/api/samples/{sample_id}/summary")
def sample_summary(sample_id: str):
    f = CACHE_DIR / f"{sample_id}_summary.wav"
    if sample_id not in SAMPLES or not f.exists():
        return _error(404, "no cached summary")
    return FileResponse(f, media_type="audio/wav")


# --------------------------------------------------------------------------- live analysis


class _StageBackend:
    """Wraps the LLM backend so its call marks the 'concerns' stage; behaviour is otherwise unchanged."""

    def __init__(self, inner, mark):
        self.inner, self.mark, self.name = inner, mark, inner.name

    def complete(self, system: str, user: str) -> str:
        self.mark("concerns")
        return self.inner.complete(system, user)


def _instrument(services, emit) -> None:
    """Report pipeline progress by observing the services analyse() calls (src/pipeline.py is unchanged):
    segment STT = transcribing, re-split piece STT = coverage, LLM = concerns, TTS = summary.
    The stream itself announces 'transcribing' first, so tracking starts there."""
    reached = {"i": 0}

    def mark(stage: str) -> None:
        i = STAGES.index(stage)
        if i > reached["i"]:
            for s in STAGES[reached["i"] + 1: i]:
                emit("stage", {"stage": s, "status": "done"})
            emit("stage", {"stage": STAGES[reached["i"]], "status": "done"})
            reached["i"] = i
            emit("stage", {"stage": stage, "status": "active"})

    transcribe, synthesize = services.transcribe, services.synthesize

    def t(path, language, start_s, end_s):
        mark("coverage" if "_piece" in Path(path).name else "transcribing")
        return transcribe(path, language, start_s, end_s)

    services.transcribe = t
    if services.llm_backend is not None:
        services.llm_backend = _StageBackend(services.llm_backend, mark)
    if synthesize is not None:
        def s(text, voice, language):
            mark("summary")
            return synthesize(text, voice, language)
        services.synthesize = s


def _live_services(sample_id: str | None):
    if MOCK:
        return mock_services(reference=SAMPLES[sample_id]["reference_turns"] if sample_id else None)
    return live_services(ROOT / ".env")


async def _save_upload(upload: UploadFile, dst: Path) -> str | None:
    """Stream the upload to dst, enforcing the size cap. Returns an error message or None."""
    size = 0
    with dst.open("wb") as f:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                return f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            f.write(chunk)
    return None if size else "The uploaded file is empty."


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/analyse")
async def analyse_endpoint(request: Request, language: str = Form(...), passcode: str = Form(""),
                           sample_id: str | None = Form(None), audio: UploadFile | None = File(None)):
    if language not in LANGUAGES:
        return _error(400, "Choose Hindi, Kannada or Hinglish / English.")
    if not passcode_configured():
        log.info("live request refused: reason=disabled")
        return _error(403, "Live analysis is disabled on this deployment. The cached examples still work.")
    if not passcode_ok(passcode):
        log.info("live request refused: reason=passcode")
        return _error(403, "Wrong or missing passcode.")
    if (audio is None) == (sample_id is None):
        return _error(400, "Send either an audio file or a sample id.")
    if sample_id is not None and sample_id not in SAMPLES:
        return _error(404, "unknown sample")
    if audio is not None and Path(audio.filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        return _error(400, "Only WAV or MP3 files are accepted.")
    ok, wait = LIVE_LIMIT.try_acquire(client_ip(request))
    if not ok:
        log.info("live request refused: reason=rate_limit")
        return _error(429, f"Live analysis limit reached. Try again in about {max(1, wait // 60)} min.")

    tmp = tempfile.TemporaryDirectory(prefix="rcqa_up_")
    if audio is not None:
        name = Path(audio.filename).name
        path = Path(tmp.name) / f"upload{Path(name).suffix.lower()}"
        problem = await _save_upload(audio, path)
        if problem:
            tmp.cleanup()
            return _error(400, problem)
    else:
        name, path = f"{sample_id}.wav", ROOT / SAMPLES[sample_id]["audio"]
    log.info("live request accepted")

    events: queue.Queue = queue.Queue()
    services = _live_services(sample_id)
    _instrument(services, lambda e, d: events.put((e, d)))

    def work() -> None:
        try:
            result = analyse(path, language, services, filename=name, max_duration_s=MAX_AUDIO_S)
            for s in STAGES:
                events.put(("stage", {"stage": s, "status": "done"}))
            log.info("%s", analysis_log_line(result))
            wav = result["summary"]["audio_wav"]
            events.put(("done", {"result": json.loads(audit_json(result)), "cached": False,
                                 "summary_audio": wav_data_uri(wav) if wav else None,
                                 "audio_url": f"/api/samples/{sample_id}/audio" if sample_id else None}))
        except AnalysisError as exc:
            events.put(("error", {"message": str(exc)}))
        except Exception as exc:  # never show internals (or anything near a key) to the client or the log
            log.error("analysis failed: %s", type(exc).__name__)
            events.put(("error", {"message": "The analysis failed. Please try again later."}))
        finally:
            tmp.cleanup()          # the upload and every temp file are gone before the final event is sent
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        yield _sse("stage", {"stage": STAGES[0], "status": "active"})
        while True:
            try:
                item = await asyncio.to_thread(events.get, True, 10)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield _sse(*item)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
