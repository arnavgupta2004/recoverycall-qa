import numpy as np
import pytest

from src import vad

SR = 8000


def _call(parts, sr=SR, noise_db=None, seed=0):
    """Build audio from (kind, seconds) parts: 'speech' = AM tone, 'pause' = silence (or noise)."""
    rng = np.random.default_rng(seed)
    chunks, t0 = [], 0.0
    for kind, sec in parts:
        n = int(sec * sr)
        t = np.arange(n) / sr
        if kind == "speech":
            x = 6000 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
        else:
            x = np.zeros(n)
        chunks.append(x)
    x = np.concatenate(chunks)
    if noise_db is not None:
        x = x + rng.normal(scale=6000 / np.sqrt(2) / 10 ** (noise_db / 20), size=len(x))
    return np.clip(x, -32768, 32767).astype(np.int16)


def test_segments_partition_whole_timeline_and_respect_max():
    parts = [("speech", 3.0), ("pause", 0.5)] * 8  # 28 s, a pause every 3.5 s
    x = _call(parts, noise_db=15)
    segs, _ = vad.segment(x, SR, vad.COARSE)
    assert segs[0]["start_s"] == 0.0
    assert segs[-1]["end_s"] == pytest.approx(len(x) / SR, abs=1e-3)
    for a, b in zip(segs, segs[1:]):
        assert a["end_s"] == b["start_s"]  # no gaps, nothing discarded
    assert all(s["end_s"] - s["start_s"] <= vad.COARSE.max_segment_s + 1e-6 for s in segs)
    assert len(segs) >= 3


def test_cuts_fall_inside_pauses():
    parts = [("speech", 3.0), ("pause", 0.5)] * 8
    x = _call(parts, noise_db=15)
    segs, _ = vad.segment(x, SR, vad.COARSE)
    pauses = [(3.0 + 3.5 * i, 3.5 + 3.5 * i) for i in range(8)]
    for s in segs[:-1]:
        assert any(p0 <= s["end_s"] <= p1 for p0, p1 in pauses), s["end_s"]


def test_short_tail_avoided_by_earlier_pause():
    # Cutting at the last pause would leave a 0.9 s tail; an earlier pause gives two full segments.
    x = _call([("speech", 5.0), ("pause", 0.6), ("speech", 6.0), ("pause", 0.6), ("speech", 0.6), ("pause", 0.2)])
    segs, _ = vad.segment(x, SR, vad.COARSE)
    assert len(segs) == 2
    assert all(vad.COARSE.min_segment_s <= s["end_s"] - s["start_s"] <= vad.COARSE.max_segment_s for s in segs)


def test_short_segment_kept_when_merge_would_exceed_max():
    # Only one pause; merging the 0.6 s blip would exceed 12 s, so it stays (and is still uploaded).
    x = _call([("speech", 11.0), ("pause", 0.6), ("speech", 0.6), ("pause", 0.2)])
    segs, _ = vad.segment(x, SR, vad.COARSE)
    assert len(segs) == 2 and segs[1]["speech_s"] > 0


def test_hard_cut_when_no_pause():
    x = _call([("speech", 26.0)])
    segs, _ = vad.segment(x, SR, vad.COARSE)
    assert len(segs) == 3 and all(s["end_s"] - s["start_s"] <= 12.0 + 1e-6 for s in segs)


def test_speech_seconds_tracks_speech_not_pauses():
    x = _call([("pause", 1.0), ("speech", 4.0), ("pause", 1.0)], noise_db=20)
    segs, _ = vad.segment(x, SR, vad.COARSE)
    assert sum(s["speech_s"] for s in segs) == pytest.approx(4.0, abs=0.3)


def test_cut_padded_adds_silence():
    x = _call([("speech", 2.0)])
    y = vad.cut_padded(x, SR, 0.5, 1.5, 0.3)
    pad = int(0.3 * SR)
    assert len(y) == int(1.0 * SR) + 2 * pad
    assert not y[:pad].any() and not y[-pad:].any()
