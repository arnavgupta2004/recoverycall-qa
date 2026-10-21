// RecoveryCall QA — single-page frontend (no build step). wavesurfer.js pinned to 7.12.12.
import WaveSurfer from "https://cdn.jsdelivr.net/npm/wavesurfer.js@7.12.12/dist/wavesurfer.esm.js";
import RegionsPlugin from "https://cdn.jsdelivr.net/npm/wavesurfer.js@7.12.12/dist/plugins/regions.esm.js";

const $ = (s, el = document) => el.querySelector(s);
const CONDUCT = {
  intimidation_or_threats: "Intimidation or threats",
  abusive_language: "Abusive language",
  third_party_contact_threat: "Third-party contact threat",
};
const AUTHORITY = "authority_or_legal_action_claim";
const CATEGORY = { ...CONDUCT, [AUTHORITY]: "Police, arrest or court claim" };
const tone = (type) => (type === AUTHORITY ? "violet" : "amber");
const DETECTOR = { rules: "Rules", LLM: "AI", both: "Both" };
const LANG_ATTR = { "hi-IN": "hi", "kn-IN": "kn", "en-IN": "en" };
const STAGES = ["transcribing", "coverage", "concerns", "summary", "done"];

const state = { config: null, samples: [], ws: null, regions: null, current: null, objectUrl: null };

function fmt(sec) {
  const s = Math.max(0, Math.round(sec || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

/* ---------------------------------------------------------------- views */

function show(view) {
  for (const v of ["start", "live", "result"]) $(`#view-${v}`).hidden = v !== view;
  window.scrollTo({ top: 0 });
}

function goHome() {
  stopAudio();
  history.replaceState(null, "", "/");
  show("start");
}

/* ---------------------------------------------------------------- start screen */

function sampleBars(s) {
  // A small sketch of the call: a bar where each flag is (amber/violet), hatched bars over coverage gaps.
  const n = 32, bars = [];
  for (let i = 0; i < n; i++) {
    const x = (i + 0.5) / n;
    const h = 28 + 62 * Math.abs(Math.sin(i * 1.7 + s.id.charCodeAt(0)));
    const pin = (s.pins || []).find((p) => Math.abs(p.at - x) < 0.5 / n + 1e-9);
    const gap = (s.gaps || []).some(([a, b]) => x >= a && x <= b);
    bars.push(el("span", { class: pin ? `f-${pin.tone}` : gap ? "f-gap" : "", style: `height:${pin ? 100 : h}%` }));
  }
  return el("div", { class: "card__bars", "aria-hidden": "true" }, bars);
}

function renderCards() {
  const wrap = $("#cards");
  wrap.replaceChildren();
  for (const s of state.samples) {
    const parts = [`${Math.round(s.duration_s)} s`, `${s.flags} flag${s.flags === 1 ? "" : "s"}`];
    if (s.coverage_warnings) parts.push(`${s.coverage_warnings} coverage gap${s.coverage_warnings === 1 ? "" : "s"}`);
    wrap.append(el("button", { class: "card", type: "button", onclick: () => openSample(s.id) },
      el("span", { class: "tag" }, "cached example"),
      el("span", { class: "card__lang" }, s.language),
      el("span", { class: "card__title" }, s.card),
      sampleBars(s),
      el("span", { class: "card__meta" }, parts.map((p) => el("span", {}, p)))));
  }
  wrap.append(el("button", { class: "card card--own", type: "button", onclick: openDialog },
    el("span", { class: "card__title" }, "Analyse your own audio"),
    el("p", {}, state.config?.live_enabled ? "WAV or MP3 up to 2 minutes. Needs the passcode."
      : "Live analysis is turned off on this deployment.")));
}

async function openSample(id) {
  const res = await fetch(`/api/samples/${id}`);
  if (!res.ok) return;
  const data = await res.json();
  history.replaceState(null, "", `#sample-${id}`);
  renderResult({ result: data.result, cached: true, title: data.card, audioUrl: data.audio_url,
                 summaryUrl: data.summary_audio_url });
}

/* ---------------------------------------------------------------- result */

function rowById(result) { return Object.fromEntries(result.transcript.map((r) => [r.segment_id, r])); }

function seek(t, play = true) {
  if (!state.ws) return;
  const d = state.ws.getDuration() || 1;
  state.ws.setTime(Math.min(Math.max(0, t), d - 0.01));
  if (play) state.ws.play();
}

function tsButton(t, label) {
  return el("button", { class: "ts", type: "button", onclick: () => seek(t), "aria-label": `Play from ${fmt(t)}` },
    label || fmt(t));
}

function renderResult({ result, cached, title, audioUrl, summaryUrl, summaryDataUri }) {
  state.current = { result, title };
  const langName = result.language_name;
  $("#r-eyebrow").textContent = cached ? "Cached example" : "Live analysis";
  $("#r-title").textContent = title || "Your call";
  const meta = $("#r-meta");
  meta.replaceChildren(
    el("span", {}, `${langName} (${result.language})`),
    el("span", {}, `${fmt(result.duration_s)} long`),
    el("span", {}, `${result.segments.length} segments`));

  const notices = $("#r-notices");
  notices.replaceChildren(...(result.notices || []).map((n) => el("div", { class: "banner", role: "note" }, n)));

  renderFindings(result);
  renderTranscript(result);
  renderSummary(result, summaryUrl || summaryDataUri);
  renderDetails(result, cached);
  show("result");
  setupWave(audioUrl, result);
}

function renderFindings(result) {
  const rows = rowById(result);
  const list = $("#f-list");
  list.replaceChildren();
  const flags = result.flags || [];
  const conduct = flags.filter((f) => f.concern_type !== AUTHORITY).length;
  const verify = flags.length - conduct;
  const gaps = result.coverage_warnings || [];
  $("#f-count").textContent = [
    `${conduct} conduct concern${conduct === 1 ? "" : "s"}`,
    `${verify} claim${verify === 1 ? "" : "s"} to verify`,
    `${gaps.length} coverage gap${gaps.length === 1 ? "" : "s"}`].join(" · ");

  if (!flags.length) list.append(el("p", { class: "empty" }, "No passages were flagged. Coverage gaps, if any, still need a listen."));
  for (const f of flags) {
    const t = tone(f.concern_type);
    list.append(el("article", { class: "fcard" },
      el("div", { class: "fcard__top" },
        el("span", { class: `chip chip--${t}` }, CATEGORY[f.concern_type]),
        el("span", { class: "det", title: "Which detector found it" }, DETECTOR[f.detector] || f.detector),
        tsButton(f.start_s)),
      el("p", { class: `fcard__quote fcard__quote--${t}` }, `“${f.quote}”`),
      el("p", { class: "fcard__meta" }, el("span", {}, f.label), el("span", {}, f.regulatory_note))));
  }

  const rejected = (result.detectors?.llm_quotes_rejected || []).filter((r) => String(r.reason || "").includes("quote"));
  if (rejected.length) {
    list.append(el("p", { class: "subhead" }, `AI flags rejected (${rejected.length})`));
    for (const r of rejected) {
      const row = rows[r.segment_id];
      list.append(el("article", { class: "fcard fcard--rejected" },
        el("div", { class: "fcard__top" },
          el("span", { class: "chip chip--muted" }, CATEGORY[r.type] || r.type),
          el("span", { class: "det" }, "AI"),
          row ? tsButton(row.start_s) : null),
        el("p", { class: "fcard__quote" }, `“${r.quote}”`),
        el("p", { class: "fcard__meta" }, "quote not found in transcript — not counted")));
    }
  }

  const extra = [];
  for (const w of gaps) {
    extra.push(el("article", { class: "fcard fcard--gap" },
      el("div", { class: "fcard__top" },
        el("span", { class: "chip chip--gap" }, w.kind === "untranscribed" ? "Untranscribed speech" : "Coverage gap"),
        tsButton(w.start_s, `${fmt(w.start_s)}–${fmt(w.end_s)}`)),
      el("p", {}, w.text)));
  }
  const hours = result.out_of_hours || {};
  if (hours.outside_window) {
    extra.push(el("article", { class: "fcard" },
      el("div", { class: "fcard__top" }, el("span", { class: "chip chip--amber" }, "Call time")),
      el("p", {}, hours.note)));
  }
  if (extra.length) list.append(el("p", { class: "subhead" }, "Coverage and call checks"), ...extra);
}

function quoteSpans(text, flags) {
  // Underline each flagged quote in place; overlapping flags share one span (first flag colours it).
  const marks = Array.from(text, () => []);
  for (const f of flags) {
    if (f.char_start == null) continue;
    for (let i = f.char_start; i < Math.min(f.char_end, text.length); i++) marks[i].push(f);
  }
  const out = [];
  let i = 0;
  while (i < text.length) {
    let j = i;
    const key = marks[i].map((f) => f.concern_type + f.quote).join("|");
    while (j < text.length && marks[j].map((f) => f.concern_type + f.quote).join("|") === key) j++;
    const chunk = text.slice(i, j);
    if (marks[i].length) {
      const f = marks[i][0];
      const title = marks[i].map((x) => `${CATEGORY[x.concern_type]} — ${x.label}`).join("; ");
      out.push(el("span", { class: `q q--${tone(f.concern_type)}`, title, role: "button", tabindex: "0",
        onclick: () => seek(f.start_s), onkeydown: (e) => { if (e.key === "Enter") seek(f.start_s); } }, chunk));
    } else out.push(chunk);
    i = j;
  }
  return out;
}

function renderTranscript(result) {
  const list = $("#t-list");
  list.replaceChildren();
  list.setAttribute("lang", LANG_ATTR[result.language] || "en");
  const gapRanges = (result.coverage_warnings || []).map((w) => [w.start_s, w.end_s]);
  const byRow = {};
  for (const f of result.flags || []) (byRow[f.segment_id] ||= []).push(f);
  for (const r of result.transcript) {
    const inGap = gapRanges.some(([a, b]) => r.start_s < b && r.end_s > a);
    const body = r.text
      ? quoteSpans(r.text, byRow[r.segment_id] || [])
      : [el("span", { class: "bubble__empty" }, "(no transcript returned for this stretch)")];
    list.append(el("div", { class: "msg", "data-start": r.start_s, "data-end": r.end_s },
      el("div", { class: "msg__time" }, tsButton(r.start_s)),
      el("div", { class: `bubble${inGap ? " bubble--gap" : ""}` },
        r.origin === "re-split piece" ? el("span", { class: "bubble__tag" }, "re-transcribed in a shorter piece") : null,
        body)));
  }
}

function renderSummary(result, src) {
  const audio = $("#summary-audio");
  if (src) { audio.src = src; audio.hidden = false; } else { audio.removeAttribute("src"); audio.hidden = true; }
  const p = $("#summary-text");
  p.textContent = result.summary?.text || "";
  p.setAttribute("lang", LANG_ATTR[result.language] || "en");
}

function renderDetails(result, cached) {
  const lat = result.latency_ms || {};
  const calls = result.api_calls || {};
  const det = result.detectors || {};
  const rejected = (det.llm_quotes_rejected || []).filter((r) => String(r.reason || "").includes("quote")).length;
  const table = el("table", {}, el("tbody", {}, Object.entries(lat).map(([k, v]) =>
    el("tr", {}, el("td", {}, k.replace(/_/g, " ")), el("td", {}, `${Math.round(v)} ms`)))));
  $("#details-body").replaceChildren(
    cached ? el("p", {}, el("strong", {}, "This view: cached example — 0 API calls."), " Timings and calls below are from when it was built.") : null,
    table,
    el("p", {}, `API calls: STT ${calls.stt ?? 0} (cache hits ${calls.stt_cache_hits ?? 0}), Gemini ${calls.llm ?? 0}, Timbre ${calls.tts ?? 0}`),
    el("p", {}, `Advanced check: ${det.llm_status} · ${det.llm || "no model"} · prompt ${det.llm_prompt_version}`),
    el("p", {}, `${rejected} AI flag${rejected === 1 ? "" : "s"} rejected: quote not found in transcript`),
    el("p", {}, `Rules: ${det.rules}`),
    el("p", {}, `Out-of-hours check: ${result.out_of_hours?.checked ? result.out_of_hours.call_time + " IST" : "skipped — " + (result.out_of_hours?.reason || "")}`));
}

/* ---------------------------------------------------------------- waveform */

function stopAudio() {
  if (state.ws) { state.ws.destroy(); state.ws = null; }
  if (state.objectUrl) { URL.revokeObjectURL(state.objectUrl); state.objectUrl = null; }
  $("#play").classList.remove("is-playing");
}

function setupWave(url, result) {
  stopAudio();
  const regions = RegionsPlugin.create();
  const ws = WaveSurfer.create({
    container: "#wave", url, height: 88, barWidth: 2, barGap: 1, barRadius: 2, normalize: true,
    waveColor: cssVar("--wave"), progressColor: cssVar("--wave-progress"), cursorColor: cssVar("--text"),
    cursorWidth: 1, dragToSeek: true, plugins: [regions],
  });
  state.ws = ws;
  state.regions = regions;
  const play = $("#play");
  play.onclick = () => ws.playPause();
  ws.on("play", () => { play.classList.add("is-playing"); play.setAttribute("aria-label", "Pause"); });
  ws.on("pause", () => { play.classList.remove("is-playing"); play.setAttribute("aria-label", "Play"); });
  const timeEl = $("#time");
  ws.on("timeupdate", (t) => {
    timeEl.textContent = `${fmt(t)} / ${fmt(ws.getDuration())}`;
    for (const m of document.querySelectorAll("#t-list .msg")) {
      m.classList.toggle("is-now", t >= +m.dataset.start && t < +m.dataset.end);
    }
  });
  ws.on("decode", (d) => {
    timeEl.textContent = `0:00 / ${fmt(d)}`;
    for (const w of result.coverage_warnings || []) {
      const r = regions.addRegion({ start: w.start_s, end: Math.max(w.end_s, w.start_s + 0.2), drag: false, resize: false,
        color: "transparent" });
      r.element.style.background = `repeating-linear-gradient(135deg, ${cssVar("--hatch")} 0 2px, transparent 2px 7px)`;
      r.element.style.opacity = "0.55";
      r.element.title = w.text;
    }
    for (const row of result.transcript.slice(1)) {
      const b = regions.addRegion({ start: row.start_s, drag: false, resize: false, color: cssVar("--border-strong") });
      b.element.title = `segment boundary ${fmt(row.start_s)}`;
    }
    const seen = {};
    for (const f of result.flags || []) {
      const k = `${f.start_s}`;
      const offset = (seen[k] = (seen[k] || 0) + 1) - 1;
      const pin = el("span", { style: `display:block;width:12px;height:12px;border-radius:50%;margin:${2 + offset * 15}px 0 0 -5px;` +
        `background:${cssVar(tone(f.concern_type) === "violet" ? "--violet-fill" : "--amber-fill")};` +
        `box-shadow:0 0 0 2px ${cssVar("--surface")}` });
      const m = regions.addRegion({ start: f.start_s, drag: false, resize: false, content: pin,
        color: cssVar(tone(f.concern_type) === "violet" ? "--violet-fill" : "--amber-fill") });
      m.element.title = `${CATEGORY[f.concern_type]} · ${fmt(f.start_s)}`;
    }
  });
  regions.on("region-clicked", (region, e) => { e.stopPropagation(); seek(region.start); });
}

/* ---------------------------------------------------------------- live analysis */

function openDialog() {
  const dlg = $("#dlg");
  const live = !!state.config?.live_enabled;
  $("#dlg-err").textContent = live ? "" : "Live analysis is turned off on this deployment (no passcode is configured).";
  $("#dlg-go").disabled = !live;
  dlg.showModal();
}

function setupDialog() {
  const langs = $("#dlg-lang");
  for (const l of state.config.languages) {
    langs.append(el("label", {}, el("input", { type: "radio", name: "lang", value: l.code, checked: l.code === "hi-IN" }),
      ` ${l.name}`));
  }
  const sel = $("#dlg-sample");
  for (const s of state.config.samples_for_live) sel.append(el("option", { value: s.id }, s.card));
  for (const r of document.querySelectorAll('input[name="src"]')) {
    r.addEventListener("change", () => {
      const up = $('input[name="src"]:checked').value === "upload";
      $("#f-upload").hidden = !up;
      $("#f-sample").hidden = up;
    });
  }
  sel.addEventListener("change", () => {
    const s = state.samples.find((x) => x.id === sel.value);
    if (s) for (const r of document.querySelectorAll('input[name="lang"]')) r.checked = r.value === s.language;
  });
  $("#dlg-form").addEventListener("submit", (e) => {
    if (e.submitter?.value !== "go") return;
    e.preventDefault();
    startLive();
  });
}

async function startLive() {
  const err = $("#dlg-err");
  err.textContent = "";
  const language = $('input[name="lang"]:checked')?.value;
  const passcode = $("#dlg-pass").value;
  const useUpload = $('input[name="src"]:checked').value === "upload";
  const file = $("#dlg-file").files[0];
  if (useUpload && !file) { err.textContent = "Choose a WAV or MP3 file."; return; }
  if (useUpload && file.size > (state.config.max_upload_mb * 1024 * 1024)) { err.textContent = `The file is larger than ${state.config.max_upload_mb} MB.`; return; }
  const form = new FormData();
  form.append("language", language);
  form.append("passcode", passcode);
  let title, audioUrl;
  if (useUpload) {
    form.append("audio", file, file.name);
    stopAudio();
    state.objectUrl = URL.createObjectURL(file);   // the player uses the local file; nothing is fetched back
    audioUrl = state.objectUrl;
    title = file.name;
  } else {
    const id = $("#dlg-sample").value;
    form.append("sample_id", id);
    title = state.config.samples_for_live.find((s) => s.id === id)?.card;
    audioUrl = `/api/samples/${id}/audio`;
  }
  $("#dlg-go").disabled = true;
  let res;
  try {
    res = await fetch("/api/analyse", { method: "POST", body: form });
  } catch {
    err.textContent = "Could not reach the server.";
    $("#dlg-go").disabled = false;
    return;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    err.textContent = body.error || `Request failed (${res.status}).`;
    $("#dlg-go").disabled = false;
    return;
  }
  $("#dlg").close();
  $("#dlg-go").disabled = false;
  $("#dlg-pass").value = "";
  runSteps(title);
  await readStream(res, title, audioUrl);
}

function runSteps(title) {
  $("#live-title").textContent = title ? `Analysing ${title}` : "Analysing call";
  $("#live-msg").textContent = "";
  $("#live-msg").classList.remove("is-error");
  $("#live-back").hidden = true;
  for (const li of document.querySelectorAll("#steps li")) li.className = "";
  show("live");
}

function markStage(stage, status) {
  const li = document.querySelector(`#steps li[data-stage="${stage}"]`);
  if (!li) return;
  li.classList.toggle("is-active", status === "active");
  li.classList.toggle("is-done", status === "done");
}

async function readStream(res, title, audioUrl) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let event = "message", data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const payload = JSON.parse(data);
      if (event === "stage") markStage(payload.stage, payload.status);
      else if (event === "error") {
        $("#live-msg").textContent = payload.message;
        $("#live-msg").classList.add("is-error");
        $("#live-back").hidden = false;
        document.querySelector("#steps li.is-active")?.classList.remove("is-active");
        return;
      } else if (event === "done") {
        markStage("done", "done");
        renderResult({ result: payload.result, cached: false, title, audioUrl: payload.audio_url || audioUrl,
                       summaryDataUri: payload.summary_audio });
        return;
      }
    }
  }
}

/* ---------------------------------------------------------------- misc */

function download() {
  if (!state.current) return;
  const { result, title } = state.current;
  const stem = (title || "call").replace(/[^\w.-]+/g, "_").slice(0, 40);
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace("T", "_").slice(0, 15);
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `audit_${stem}_${stamp}.json` });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

function setupTheme() {
  const btn = $("#theme");
  const saved = (() => { try { return localStorage.getItem("rcqa-theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  btn.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("rcqa-theme", next); } catch { /* private mode */ }
    if (state.ws) state.ws.setOptions({ waveColor: cssVar("--wave"), progressColor: cssVar("--wave-progress"), cursorColor: cssVar("--text") });
  });
}

async function init() {
  setupTheme();
  $("#home").addEventListener("click", goHome);
  $("#r-back").addEventListener("click", goHome);
  $("#live-back").addEventListener("click", goHome);
  $("#r-download").addEventListener("click", download);
  const [config, samples] = await Promise.all([fetch("/api/config").then((r) => r.json()),
                                               fetch("/api/samples").then((r) => r.json())]);
  state.config = config;
  state.samples = samples;
  setupDialog();
  renderCards();
  const m = location.hash.match(/^#sample-([A-Z])$/);
  if (m) openSample(m[1]);
}

init();
