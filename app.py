"""RecoveryCall QA — single-page audit app (Gradio, for Hugging Face Spaces).

    python app.py                 # real APIs for live runs (keys from env / .env / HF secrets)
    RCQA_MOCK=1 python app.py     # everything offline: mock STT, LLM and TTS

Bundled samples are served from cached results (labelled "(cached example)"). Live runs (a sample
or an upload) need UPLOAD_PASSCODE (live features are disabled when it is unset) and are rate-limited
per IP. Uploads are deleted right after analysis and never cached; nothing is written to disk except
the audit JSON offered for download, which Gradio's cache deletes after at most 10 minutes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # no telemetry; must be set before importing gradio

import gradio as gr  # noqa: E402

from src.pipeline import LANGUAGES, AnalysisError, analyse, audit_json, live_services, mock_services
from src.render import CSS, JS_ON_LOAD, audio_data_uri, details_markdown, render_result, wav_data_uri
from src.webguard import (MAX_AUDIO_S, RateLimiter, check_upload, client_ip, passcode_configured, passcode_ok)

ROOT = Path(__file__).resolve().parent
MOCK = os.environ.get("RCQA_MOCK") == "1"
SAMPLES = json.loads((ROOT / "samples" / "samples.json").read_text(encoding="utf-8"))
CACHE_DIR = ROOT / "samples" / ("cache_mock" if MOCK else "cache")
LIVE_LIMIT = RateLimiter(limit=int(os.environ.get("LIVE_RUNS_PER_HOUR", "5")), window_s=3600)
DOWNLOAD_DIR = Path(tempfile.mkdtemp(prefix="rcqa_dl_"))

BANNER = ("**Fictional or test audio only. Uploads are not stored. Live analysis sends the transcript to "
          "Google Gemini.**  \nFlags are potential conduct concerns for human review; claims of police, arrest "
          "or court action are marked to verify. This tool gives no score and no legal conclusion.")
LANG_CHOICES = [(f"{name} ({code})", code) for code, name in
                (("hi-IN", "Hindi"), ("kn-IN", "Kannada"), ("en-IN", "Hinglish / English"))]
SAMPLE_CHOICES = [(s["title"], sid) for sid, s in SAMPLES.items()]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rcqa")


def _download_file(result: dict, stem: str) -> str:
    """Write the audit JSON for download; files older than 10 minutes are swept on every call."""
    now = time.time()
    for old in DOWNLOAD_DIR.glob("*.json"):
        if now - old.stat().st_mtime > 600:
            old.unlink(missing_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = DOWNLOAD_DIR / f"audit_{stem}_{stamp}.json"
    path.write_text(audit_json(result), encoding="utf-8")
    return str(path)


def _outputs(result: dict, audio_path: Path, cached: bool, stem: str, summary_wav: bytes | None):
    html_out = render_result(result, audio_data_uri(audio_path), wav_data_uri(summary_wav) if summary_wav else None,
                             cached)
    return (html_out, gr.update(value=_download_file(result, stem), visible=True), details_markdown(result, cached),
            "")


def _error(msg: str):
    return "", gr.update(value=None, visible=False), "", f"⚠️ {msg}"


def show_cached(sample_id: str):
    s = SAMPLES[sample_id]
    f = CACHE_DIR / f"{sample_id}.json"
    if not f.exists():
        return _error(f"The cached result for sample {sample_id} has not been built yet "
                      f"(python build_samples.py build --only {sample_id}).")
    result = json.loads(f.read_text(encoding="utf-8"))
    log.info("cached example served: sample=%s api_calls=0", sample_id)
    wav = CACHE_DIR / f"{sample_id}_summary.wav"
    return _outputs(result, ROOT / s["audio"], True, f"sample{sample_id}", wav.read_bytes() if wav.exists() else None)


def _live_gate(passcode: str, request: gr.Request) -> str | None:
    if not passcode_configured():
        return "Live analysis is disabled on this deployment (no UPLOAD_PASSCODE set). Cached examples still work."
    if not passcode_ok(passcode):
        return "Wrong or missing passcode for live analysis."
    ok, wait = LIVE_LIMIT.try_acquire(client_ip(request))
    if not ok:
        return f"Live analysis limit reached for your connection. Try again in about {max(1, wait // 60)} min."
    return None


def _services():
    return mock_services() if MOCK else live_services(ROOT / ".env")


def _run(path: Path, language: str, filename: str, stem: str, services):
    try:
        result = analyse(path, language, services, filename=filename, max_duration_s=MAX_AUDIO_S)
    except AnalysisError as exc:
        return _error(str(exc))
    except Exception as exc:  # never show internals (or anything near a key) to the user
        log.error("analysis failed: %s", type(exc).__name__)
        return _error("The analysis failed. Please try again later.")
    log.info("analysis done: lang=%s duration=%ss flags=%d llm=%s calls=%s total_ms=%s", language,
             result["duration_s"], len(result["flags"]), result["detectors"]["llm_status"], result["api_calls"],
             result["latency_ms"]["total"])
    return _outputs(result, path, False, stem, result["summary"]["audio_wav"])


def run_live_sample(sample_id: str, language: str, passcode: str, request: gr.Request):
    denied = _live_gate(passcode, request)
    if denied:
        return _error(denied)
    s = SAMPLES[sample_id]
    services = mock_services(reference=s["reference_turns"]) if MOCK else _services()
    return _run(ROOT / s["audio"], language, f"{sample_id}.wav", f"sample{sample_id}_live", services)


def run_upload(file_path: str | None, language: str, passcode: str, request: gr.Request):
    try:
        problem = check_upload(file_path)
        if problem:
            return _error(problem)
        denied = _live_gate(passcode, request)
        if denied:
            return _error(denied)
        name = Path(file_path).name
        return _run(Path(file_path), language, name, Path(name).stem[:40], _services())
    finally:
        if file_path:  # uploads are not stored: remove Gradio's temp copy as soon as we are done
            Path(file_path).unlink(missing_ok=True)


def on_sample_change(sample_id: str):
    return SAMPLES[sample_id]["language"]


with gr.Blocks(title="RecoveryCall QA", delete_cache=(300, 600), analytics_enabled=False) as demo:
    gr.Markdown("# RecoveryCall QA\nAudit a debt-recovery call: transcript, flagged passages, coverage warnings.")
    gr.Markdown(BANNER)
    language = gr.Radio(LANG_CHOICES, value="hi-IN", label="Call language (no auto-detect)")
    with gr.Group():
        sample = gr.Dropdown(SAMPLE_CHOICES, value="D", label="Sample calls")
        with gr.Row():
            btn_cached = gr.Button("Show cached example", variant="primary")
            btn_live = gr.Button("Run live")
    with gr.Accordion("Upload your own audio (live)", open=False):
        upload = gr.File(label="WAV or MP3, max 2 minutes", file_types=[".wav", ".mp3"], type="filepath")
        btn_upload = gr.Button("Analyse upload")
    passcode = gr.Textbox(label="Passcode for live analysis", type="password",
                          placeholder="required for Run live and uploads")
    status = gr.Markdown()
    result_html = gr.HTML(js_on_load=JS_ON_LOAD, apply_default_css=True)
    download = gr.File(label="Download audit JSON", visible=False, interactive=False)
    with gr.Accordion("Details", open=False):
        details = gr.Markdown()

    outputs = [result_html, download, details, status]
    sample.change(on_sample_change, sample, language)
    btn_cached.click(show_cached, sample, outputs)
    btn_live.click(run_live_sample, [sample, language, passcode], outputs, concurrency_limit=2)
    btn_upload.click(run_upload, [upload, language, passcode], outputs, concurrency_limit=2)
    demo.load(show_cached, sample, outputs)


if __name__ == "__main__":
    # Host/port come from GRADIO_SERVER_NAME / GRADIO_SERVER_PORT (set by Spaces); default 127.0.0.1:7860.
    demo.queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft(), css=CSS)
