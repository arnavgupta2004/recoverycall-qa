"""Coverage check and recovery for STT of VAD segments (shared by experiment.py and src/pipeline.py).

A segment's density is normalized transcript characters (spaces excluded) per VAD speech-second.
A segment is flagged when its density falls below FLAG_FRACTION x the language baseline; it is
then re-split at shorter pauses and the pieces are transcribed. The pieces' transcript is kept
only if density improves. Anything still below threshold, or a segment whose STT failed, is
reported as untranscribed speech with start/end times: never dropped silently.

Transcription is passed in as a callback so callers control caching, mocking and file handling.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src import vad
from src.metrics import normalize

FLAG_FRACTION = 0.5
MIN_JUDGE_SPEECH_S = 0.5   # segments/pieces with less detected speech are not judged

# Chars per VAD speech-second on clean audio, measured by experiment.py (cmd_baseline) on the clean
# verbatim chunks of the 28 s and 12 s spike runs, 2026-10-21 (see docs/ and the experiment report).
BASELINE_CHARS_PER_SPEECH_S = {"en-IN": 14.866, "hi-IN": 14.211, "kn-IN": 14.367}

# transcribe(piece_index, piece, padded_samples) -> (transcript or None, extra fields for the record)
PieceTranscriber = Callable[[int, dict, np.ndarray], tuple[str | None, dict]]


def char_count(text: str) -> int:
    return len(normalize(text).replace(" ", ""))


def density(text: str | None, speech_s: float) -> float | None:
    if text is None or speech_s < MIN_JUDGE_SPEECH_S:
        return None
    return char_count(text) / speech_s


def threshold_for(language: str, fraction: float = FLAG_FRACTION, baseline: dict | None = None) -> float:
    return fraction * (baseline or BASELINE_CHARS_PER_SPEECH_S)[language]


def fine_pieces(x: np.ndarray, sr: int, seg: dict, threshold_db: float) -> list[dict]:
    """Re-split one coarse segment at shorter pauses; times are absolute (seconds in the call)."""
    a, b = seg["start_s"], seg["end_s"]
    sub = x[int(round(a * sr)): int(round(b * sr))]
    pieces, _ = vad.segment(sub, sr, vad.FINE, threshold_db=threshold_db)
    for p in pieces:
        p["start_s"], p["end_s"] = round(a + p["start_s"], 3), round(a + p["end_s"], 3)
        p["regions"] = [(round(a + s, 3), round(a + e, 3)) for s, e in p["regions"]]
    return pieces


def recover(x: np.ndarray, sr: int, seg: dict, threshold_db: float, thr: float, d0: float | None,
            transcribe: PieceTranscriber) -> dict:
    """Re-split a flagged segment at shorter pauses and transcribe the pieces; keep only if density improves."""
    texts, out = [], []
    for j, p in enumerate(fine_pieces(x, sr, seg, threshold_db)):
        samples = vad.cut_padded(x, sr, p["start_s"], p["end_s"], vad.FINE.pad_s)
        text, extra = transcribe(j, p, samples)
        t = text or ""
        texts.append(t)
        d = density(t, p["speech_s"])
        out.append({"piece": j, "start_s": p["start_s"], "end_s": p["end_s"], "speech_s": p["speech_s"], **extra,
                    "transcript": t, "density": d, "below_threshold": d is not None and d < thr})
    joined = " ".join(t for t in texts if t)
    d_new = density(joined, seg["speech_s"])
    return {"pieces": out, "transcript": joined, "density": d_new,
            "kept": d_new is not None and (d0 is None or d_new > d0)}


def check_segment(x: np.ndarray, sr: int, seg: dict, first: str | None, thr: float, threshold_db: float,
                  transcribe_piece: PieceTranscriber, stt_error: str | None = None) -> dict:
    """Coverage check (and recovery if flagged) for one segment whose first transcript is `first`."""
    d0 = density(first, seg["speech_s"])
    rec = {"transcript_first": first, "density_first": d0, "threshold": round(thr, 3),
           "flagged": d0 is not None and d0 < thr, "recovery": None,
           "transcript_final": first, "density_final": d0, "untranscribed": []}
    if first is None and seg["speech_s"] >= MIN_JUDGE_SPEECH_S:
        rec["untranscribed"] = [{"start_s": seg["start_s"], "end_s": seg["end_s"], "speech_s": seg["speech_s"],
                                 "reason": stt_error or "STT error"}]
    if rec["flagged"]:
        rec["recovery"] = recover(x, sr, seg, threshold_db, thr, d0, transcribe_piece)
        if rec["recovery"]["kept"]:
            rec["transcript_final"] = rec["recovery"]["transcript"]
            rec["density_final"] = rec["recovery"]["density"]
        low = [p for p in rec["recovery"]["pieces"] if p["below_threshold"]] if rec["recovery"]["kept"] else []
        if rec["density_final"] is None or rec["density_final"] < thr:
            # Never drop silently: name the speech that is still not transcribed.
            rec["untranscribed"] = ([{"start_s": p["start_s"], "end_s": p["end_s"], "speech_s": p["speech_s"]}
                                     for p in low] or
                                    [{"start_s": seg["start_s"], "end_s": seg["end_s"], "speech_s": seg["speech_s"]}])
        else:
            rec["untranscribed"] = [{"start_s": p["start_s"], "end_s": p["end_s"], "speech_s": p["speech_s"]}
                                    for p in low]
    return rec
