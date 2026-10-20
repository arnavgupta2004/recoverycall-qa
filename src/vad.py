"""Energy-based voice activity detection and pause-aligned segmentation (numpy only).

Uses the audio alone: no reference text, no manifest offsets. A frame is speech when its
energy is more than margin_db above an adaptive noise floor (a low percentile of all frame
energies). Short gaps are bridged (they are not pauses) and short blips are dropped.

Segmentation partitions the WHOLE timeline: cuts are placed at the middle of detected
pauses, so a VAD miss can only distort the speech-seconds estimate, never discard audio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VadConfig:
    frame_s: float = 0.030
    hop_s: float = 0.010
    floor_percentile: float = 5.0
    margin_db: float = 3.0
    min_pause_s: float = 0.30      # gaps shorter than this are bridged
    min_speech_s: float = 0.10     # speech blips shorter than this are dropped
    max_segment_s: float = 12.0
    min_segment_s: float = 1.5     # shorter segments are merged with a neighbour
    pad_s: float = 0.30            # silence added on both sides of each uploaded segment

    def to_dict(self) -> dict:
        return asdict(self)


COARSE = VadConfig()
# Recovery re-split: shorter pauses count, pieces may be short.
FINE = VadConfig(min_pause_s=0.12, max_segment_s=6.0, min_segment_s=0.8)


def frame_energy_db(x: np.ndarray, sr: int, cfg: VadConfig) -> np.ndarray:
    """RMS energy (dBFS) of overlapping frames; one value per hop."""
    x = x.astype(np.float64) / 32768.0
    frame, hop = int(cfg.frame_s * sr), int(cfg.hop_s * sr)
    if len(x) < frame:
        return np.array([-120.0])
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20 * np.log10(rms + 1e-9)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def speech_regions(x: np.ndarray, sr: int, cfg: VadConfig = COARSE,
                   threshold_db: float | None = None) -> tuple[list[tuple[float, float]], dict]:
    """Speech regions in seconds, plus diagnostics (noise floor, threshold).

    threshold_db overrides the adaptive threshold (used when re-splitting a short slice,
    whose own percentile floor would be unreliable).
    """
    e = frame_energy_db(x, sr, cfg)
    floor = float(np.percentile(e, cfg.floor_percentile))
    # Digital silence (as in the clean concatenation) gives a floor near -180 dB; keep the
    # threshold at a level real speech always exceeds.
    threshold = threshold_db if threshold_db is not None else max(floor + cfg.margin_db, -60.0)
    mask = e > threshold
    hop = cfg.hop_s
    # Bridge gaps shorter than min_pause_s.
    for s, t in _runs(~mask):
        if 0 < s and t < len(mask) and (t - s) * hop < cfg.min_pause_s:
            mask[s:t] = True
    regions = []
    for s, t in _runs(mask):
        start, end = s * hop, (t - 1) * hop + cfg.frame_s
        if end - start >= cfg.min_speech_s:
            regions.append((round(float(start), 3), round(float(min(end, len(x) / sr)), 3)))
    return regions, {"noise_floor_db": round(floor, 1), "threshold_db": round(threshold, 1)}


def segment(x: np.ndarray, sr: int, cfg: VadConfig = COARSE,
            threshold_db: float | None = None) -> tuple[list[dict], dict]:
    """Partition the whole timeline into segments <= max_segment_s, cut at pause midpoints.

    Each cut takes the furthest pause that keeps the segment within max_segment_s (a hard
    cut only if there is no pause). Segments shorter than min_segment_s, or with no detected
    speech, are merged into the neighbour across the shorter pause when the result still fits.
    Each segment: {"start_s", "end_s", "speech_s", "regions"}.
    """
    total = len(x) / sr
    regions, diag = speech_regions(x, sr, cfg, threshold_db)
    pause_cuts = [round(float(a[1] + b[0]) / 2, 3) for a, b in zip(regions, regions[1:])]
    bounds, start = [0.0], 0.0
    while total - start > cfg.max_segment_s:
        fits = [c for c in pause_cuts if start < c <= start + cfg.max_segment_s]
        # Prefer the furthest pause that does not leave a final remainder shorter than min_segment_s.
        ok = [c for c in fits if total - c >= cfg.min_segment_s]
        start = (ok or fits)[-1] if fits else round(start + cfg.max_segment_s, 3)
        bounds.append(start)
    bounds.append(round(total, 3))
    spans = list(zip(bounds, bounds[1:]))

    def speech_in(a: float, b: float) -> float:
        return sum(max(0.0, min(b, re) - max(a, rs)) for rs, re in regions)

    def small(span) -> bool:
        return span[1] - span[0] < cfg.min_segment_s or speech_in(*span) == 0

    merged = True
    while merged and len(spans) > 1:
        merged = False
        for i, sp in enumerate(spans):
            if not small(sp):
                continue
            # Prefer the neighbour that keeps the merged segment shortest.
            for j in sorted((k for k in (i - 1, i + 1) if 0 <= k < len(spans)),
                            key=lambda k: max(spans[k][1], sp[1]) - min(spans[k][0], sp[0])):
                a, b = min(spans[j][0], sp[0]), max(spans[j][1], sp[1])
                if b - a <= cfg.max_segment_s:
                    spans[min(i, j)] = (a, b)
                    del spans[max(i, j)]
                    merged = True
                    break
            if merged:
                break
    segs = []
    for a, b in spans:
        inside = [(max(a, rs), min(b, re)) for rs, re in regions if re > a and rs < b]
        segs.append({"start_s": float(a), "end_s": float(b), "speech_s": round(float(speech_in(a, b)), 3),
                     "regions": [(float(s), float(e)) for s, e in inside]})
    diag["pause_cuts_available"] = len(pause_cuts)
    return segs, diag


def cut_padded(x: np.ndarray, sr: int, start_s: float, end_s: float, pad_s: float) -> np.ndarray:
    """Audio from start_s to end_s with pad_s of digital silence on both sides."""
    pad = np.zeros(int(round(pad_s * sr)), dtype=np.int16)
    return np.concatenate([pad, x[int(round(start_s * sr)): int(round(end_s * sr))], pad])
