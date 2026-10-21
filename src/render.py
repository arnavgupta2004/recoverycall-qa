"""HTML rendering of an analysis result for the web app (escaped; no scores, no verdict wording)."""

from __future__ import annotations

import base64
import html
import subprocess
from pathlib import Path

from src.audio import require_ffmpeg
from src.concern_rules import ABUSIVE, AUTHORITY, INTIMIDATION, THIRD_PARTY
from src.pipeline import fmt_time

CATEGORY_NAMES = {
    INTIMIDATION: "Intimidation or threats",
    ABUSIVE: "Abusive language",
    THIRD_PARTY: "Third-party contact threat",
    AUTHORITY: "Police, arrest or court claim",
}
CATEGORY_CLASS = {INTIMIDATION: "c-int", ABUSIVE: "c-abu", THIRD_PARTY: "c-tp", AUTHORITY: "c-auth"}

CSS = """
.rcqa{--int:#d03b3b;--abu:#8a4fd6;--tp:#d97a00;--auth:#2a78d6;font-size:15px;line-height:1.55}
.rcqa .player{width:100%;margin:4px 0 12px}
.rcqa .notice{border-left:4px solid var(--tp);background:rgba(217,122,0,.12);padding:8px 12px;border-radius:6px;margin:8px 0}
.rcqa .warn{border-left:4px solid var(--int);background:rgba(208,59,59,.10);padding:8px 12px;border-radius:6px;margin:6px 0;cursor:pointer}
.rcqa .info{border-left:4px solid var(--auth);background:rgba(42,120,214,.10);padding:8px 12px;border-radius:6px;margin:6px 0}
.rcqa h3{margin:18px 0 6px;font-size:1.05em}
.rcqa .row{display:flex;gap:10px;align-items:baseline;padding:6px 0;border-bottom:1px solid rgba(127,127,127,.2)}
.rcqa .ts{flex:0 0 auto;font-variant-numeric:tabular-nums;font-size:.85em;padding:1px 8px;border-radius:999px;
  border:1px solid rgba(127,127,127,.45);background:transparent;color:inherit;cursor:pointer}
.rcqa .txt{flex:1 1 auto;overflow-wrap:anywhere}
.rcqa .empty{opacity:.6;font-style:italic}
.rcqa mark{color:inherit;border-radius:3px;padding:0 2px;cursor:pointer;border-bottom:2px solid}
.rcqa mark.c-int{background:rgba(208,59,59,.18);border-color:var(--int)}
.rcqa mark.c-abu{background:rgba(138,79,214,.18);border-color:var(--abu)}
.rcqa mark.c-tp{background:rgba(217,122,0,.20);border-color:var(--tp)}
.rcqa mark.c-auth{background:rgba(42,120,214,.18);border-color:var(--auth)}
.rcqa .flag{border:1px solid rgba(127,127,127,.3);border-left-width:5px;border-radius:8px;padding:8px 12px;margin:8px 0;cursor:pointer}
.rcqa .flag.c-int{border-left-color:var(--int)} .rcqa .flag.c-abu{border-left-color:var(--abu)}
.rcqa .flag.c-tp{border-left-color:var(--tp)} .rcqa .flag.c-auth{border-left-color:var(--auth)}
.rcqa .flag .cat{font-weight:600}
.rcqa .flag .meta{font-size:.85em;opacity:.8}
.rcqa .legend span{display:inline-block;margin:2px 10px 2px 0;font-size:.85em}
.rcqa .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
"""

JS_ON_LOAD = """
element.addEventListener('click', (e) => {
  const t = e.target.closest('[data-seek]');
  if (!t) return;
  const a = element.querySelector('audio.rcqa-call');
  if (!a) return;
  a.currentTime = parseFloat(t.dataset.seek) || 0;
  a.play().catch(() => {});
  a.scrollIntoView({block: 'nearest', behavior: 'smooth'});
});
"""


def audio_data_uri(path: str | Path, mp3: bool = True) -> str:
    """Encode audio for an inline <audio> element (MP3 64 kbps mono keeps it small)."""
    if mp3:
        proc = subprocess.run([require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-ac", "1",
                               "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1"], capture_output=True)
        if proc.returncode == 0 and proc.stdout:
            return "data:audio/mpeg;base64," + base64.b64encode(proc.stdout).decode("ascii")
    return "data:audio/wav;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def wav_data_uri(wav: bytes) -> str:
    """Inline a WAV clip (e.g. the spoken summary) as MP3 when ffmpeg can encode it, else as WAV."""
    proc = subprocess.run([require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ac", "1",
                           "-c:a", "libmp3lame", "-b:a", "48k", "-f", "mp3", "pipe:1"], input=wav, capture_output=True)
    if proc.returncode == 0 and proc.stdout:
        return "data:audio/mpeg;base64," + base64.b64encode(proc.stdout).decode("ascii")
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


def _highlighted(text: str, flags: list[dict]) -> str:
    """Escape text and wrap flagged quotes in <mark>; overlapping flags share a mark (first category colours it)."""
    cats_at: list[list[dict]] = [[] for _ in text]
    for f in flags:
        if f.get("char_start") is None:
            continue
        for i in range(f["char_start"], min(f["char_end"], len(text))):
            cats_at[i].append(f)
    out, i = [], 0
    while i < len(text):
        j = i
        while j < len(text) and [id(f) for f in cats_at[j]] == [id(f) for f in cats_at[i]]:
            j += 1
        chunk = html.escape(text[i:j])
        if cats_at[i]:
            fs = cats_at[i]
            title = "; ".join(f"{CATEGORY_NAMES[f['concern_type']]}: {f['label']}" for f in fs)
            out.append(f'<mark class="{CATEGORY_CLASS[fs[0]["concern_type"]]}" data-seek="{fs[0]["start_s"]:.2f}" '
                       f'title="{html.escape(title)}">{chunk}</mark>')
        else:
            out.append(chunk)
        i = j
    return "".join(out)


def render_result(result: dict, audio_uri: str | None, summary_uri: str | None, cached: bool) -> str:
    e = html.escape
    H = ['<div class="rcqa">']
    origin = " (cached example)" if cached else ""
    H.append(f'<div class="info">{e(result["language_name"])} · {result["duration_s"]:.0f} s'
             f'{e(origin)} · {len(result["flags"])} flag(s) for human review</div>')
    for n in result.get("notices", []):
        H.append(f'<div class="notice">{e(n)}</div>')
    if audio_uri:
        H.append(f'<audio class="rcqa-call player" controls preload="metadata" src="{audio_uri}"></audio>')

    if result["coverage_warnings"]:
        H.append("<h3>Coverage warnings</h3>")
        for w in result["coverage_warnings"]:
            H.append(f'<div class="warn" data-seek="{w["start_s"]:.2f}">{e(w["text"])}</div>')
    hours = result.get("out_of_hours", {})
    if hours.get("outside_window"):
        H.append(f'<div class="warn">{e(hours["note"])}</div>')

    H.append("<h3>Flags</h3>")
    if not result["flags"]:
        H.append('<div class="empty">No potential conduct concerns were flagged. Coverage warnings, if any, still '
                 'need review.</div>')
    for f in result["flags"]:
        H.append(f'<div class="flag {CATEGORY_CLASS[f["concern_type"]]}" data-seek="{f["start_s"]:.2f}">'
                 f'<div class="cat">{e(CATEGORY_NAMES[f["concern_type"]])} · {e(f["time"])}</div>'
                 f'<div>“{e(f["quote"])}”</div>'
                 f'<div class="meta">{e(f["label"])} · {e(f["regulatory_note"])} · found by: {e(f["detector"])}</div>'
                 "</div>")

    H.append("<h3>Transcript</h3>")
    H.append('<div class="legend">' + "".join(
        f'<span><i style="background:var(--{c})"></i>{e(n)}</span>'
        for c, n in (("int", CATEGORY_NAMES[INTIMIDATION]), ("abu", CATEGORY_NAMES[ABUSIVE]),
                     ("tp", CATEGORY_NAMES[THIRD_PARTY]), ("auth", CATEGORY_NAMES[AUTHORITY]))) + "</div>")
    by_row: dict[str, list] = {}
    for f in result["flags"]:
        by_row.setdefault(f["segment_id"], []).append(f)
    for r in result["transcript"]:
        ts = fmt_time(r["start_s"]) + "–" + fmt_time(r["end_s"])
        body = _highlighted(r["text"], by_row.get(r["segment_id"], [])) if r["text"] else \
            '<span class="empty">(no transcript returned)</span>'
        H.append(f'<div class="row"><button class="ts" data-seek="{r["start_s"]:.2f}">{ts}</button>'
                 f'<div class="txt">{body}</div></div>')

    s = result["summary"]
    H.append("<h3>Spoken summary</h3>")
    if summary_uri:
        H.append(f'<audio class="player" controls preload="none" src="{summary_uri}"></audio>')
    else:
        H.append(f'<div class="empty">Audio summary {e(s.get("status", "not available"))}.</div>')
    H.append(f"<div>{e(s['text'])}</div>")
    H.append("</div>")
    return "\n".join(H)


def _md(text: str) -> str:
    """Escape Markdown-significant characters in transcript text shown in the Details panel."""
    return html.escape(text).replace("|", "\\|").replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")


def details_markdown(result: dict, cached: bool = False) -> str:
    """Details panel: stage latency, API calls, detector status, verifier rejections (Markdown blocks)."""
    lat, calls, det = result["latency_ms"], result["api_calls"], result["detectors"]
    rejected = det.get("llm_quotes_rejected") or []
    bad_quotes = [r for r in rejected if "quote" in str(r.get("reason", ""))]
    other = [r for r in rejected if r not in bad_quotes]
    blocks = []
    if cached:
        blocks.append("**This view: cached example — 0 API calls.** The timings and call counts below are from when "
                      "the example was built.")
    blocks.append("\n".join(["| stage | ms |", "|---|---|"] + [f"| {k} | {v:.0f} |" for k, v in lat.items()]))
    blocks.append(f"**API calls:** STT {calls.get('stt', 0)} (cache hits {calls.get('stt_cache_hits', 0)}), "
                  f"Gemini {calls.get('llm', 0)}, Timbre {calls.get('tts', 0)}")
    blocks.append(f"**Advanced check (LLM):** {det['llm_status']} · model {det['llm'] or 'none'} · prompt "
                  f"{det['llm_prompt_version']}")
    rej = [f"**{len(bad_quotes)} AI flag{'' if len(bad_quotes) == 1 else 's'} rejected: quote not found in transcript**"]
    rej += [f"- {_md(str(r.get('type')))}: “{_md(str(r.get('quote')))}”" for r in bad_quotes]
    rej += [f"- other rejection ({_md(str(r.get('reason')))}): {_md(str(r.get('type')))}" for r in other]
    blocks.append("\n".join(rej))
    blocks.append(f"**Rules:** {det['rules']}")
    blocks.append(f"**Segments:** {len(result['segments'])} (≤ 12 s each); recovered: "
                  f"{sum(s['recovered'] for s in result['segments'])}")
    hours = result["out_of_hours"]
    blocks.append("**Out-of-hours check:** " + (f"{hours['call_time']} IST" if hours["checked"]
                                                else f"skipped — {hours['reason']}"))
    return "\n\n".join(blocks)



def analysis_log_line(result: dict) -> str:
    """One log line per analysis: language, counts, LLM status class, API calls, per-stage ms. Numbers only, no text
    from the call (no transcript, quotes, file name or error messages)."""
    status = str(result["detectors"]["llm_status"]).split(":")[0].split(" ")[0]   # ok / timeout / error / disabled
    calls = " ".join(f"{k}={int(v)}" for k, v in result["api_calls"].items())
    stages = " ".join(f"{k}={int(round(v))}" for k, v in result["latency_ms"].items())
    return (f"analysis done: lang={result['language']} duration_s={result['duration_s']:.1f} "
            f"segments={len(result['segments'])} flags={len(result['flags'])} "
            f"coverage_warnings={len(result['coverage_warnings'])} llm={status} | calls {calls} | latency_ms {stages}")
