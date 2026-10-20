"""ffmpeg + numpy audio helpers: normalize, concatenate, phone-channel simulation, chunking."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

CLEAN_SR = 16000
PHONE_SR = 8000
GAP_S = 0.35
PHONE_BAND = (300, 3400)
MAX_CHUNK_S = 28.0  # docs disagree (60 s vs 30 s); stay under 30 s with margin
CHUNK_PAD_S = 0.15


class FFmpegMissing(RuntimeError):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegMissing(
            "ffmpeg not found on PATH. Install it (macOS: `brew install ffmpeg`, "
            "Debian/Ubuntu: `sudo apt install ffmpeg`) and retry."
        )
    return path


def _ffmpeg(*args: str) -> None:
    cmd = [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(args[:4])} ...): {proc.stderr.strip()[:500]}")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM mono WAV into int16 samples."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono s16le, got {w.getnchannels()}ch/{w.getsampwidth() * 8}bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy(), w.getframerate()


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.astype("<i2").tobytes())


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def normalize_turn(src: Path, dst: Path) -> None:
    """Any input -> 16 kHz mono s16le WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-i", str(src), "-ar", str(CLEAN_SR), "-ac", "1", "-c:a", "pcm_s16le", str(dst))


def concatenate(turn_paths: list[Path], dst: Path, gap_s: float = GAP_S) -> list[tuple[float, float]]:
    """Concatenate normalized turns with gap_s of digital silence. Returns per-turn (start, end) seconds."""
    gap = np.zeros(int(round(gap_s * CLEAN_SR)), dtype=np.int16)
    parts, offsets, cursor = [], [], 0
    for i, p in enumerate(turn_paths):
        samples, sr = read_wav(p)
        if sr != CLEAN_SR:
            raise ValueError(f"{p}: expected {CLEAN_SR} Hz after normalization, got {sr}")
        if i:
            parts.append(gap)
            cursor += len(gap)
        offsets.append((cursor / CLEAN_SR, (cursor + len(samples)) / CLEAN_SR))
        parts.append(samples)
        cursor += len(samples)
    write_wav(dst, np.concatenate(parts), CLEAN_SR)
    return offsets


def bandlimit_8k(src: Path, dst: Path) -> None:
    """16 kHz clean -> 8 kHz PCM, highpass 300 Hz + lowpass 3400 Hz."""
    _ffmpeg("-i", str(src), "-af", f"highpass=f={PHONE_BAND[0]},lowpass=f={PHONE_BAND[1]}",
            "-ar", str(PHONE_SR), "-ac", "1", "-c:a", "pcm_s16le", str(dst))


def encode_ulaw(src: Path, dst: Path) -> None:
    """PCM -> G.711 mu-law in a WAV container (8 kHz)."""
    _ffmpeg("-i", str(src), "-ar", str(PHONE_SR), "-ac", "1", "-c:a", "pcm_mulaw", str(dst))


def decode_to_pcm(src: Path, dst: Path) -> None:
    """mu-law WAV -> 8 kHz s16le PCM WAV."""
    _ffmpeg("-i", str(src), "-ar", str(PHONE_SR), "-ac", "1", "-c:a", "pcm_s16le", str(dst))


def pink_noise(n: int, sr: int, seed: int, band: tuple[float, float] | None = PHONE_BAND) -> np.ndarray:
    """Pink (1/f power) noise via spectral shaping, optionally band-limited, unit RMS, float64."""
    rng = np.random.default_rng(seed)
    spectrum = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    shape = np.zeros_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    if band is not None:
        shape[(freqs < band[0]) | (freqs > band[1])] = 0.0
    noise = np.fft.irfft(spectrum * shape, n=n)
    return noise / np.sqrt(np.mean(noise ** 2))


def speech_mask(n: int, sr: int, offsets: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in offsets:
        mask[int(start * sr): min(n, int(round(end * sr)))] = True
    return mask


def mix_at_snr(signal: np.ndarray, sr: int, offsets: list[tuple[float, float]], target_snr_db: float,
               seed: int) -> tuple[np.ndarray, dict]:
    """Add pink noise scaled so speech-segment SNR is target_snr_db.

    Signal power is measured over the turn segments only (the inter-turn silence is
    digital zero and would otherwise drag the "signal" power down). Noise is added
    everywhere. The returned SNR is re-measured on the final int16 mix, so it includes
    quantization and any clipping.
    """
    sig = signal.astype(np.float64)
    mask = speech_mask(len(sig), sr, offsets)
    p_sig = float(np.mean(sig[mask] ** 2))
    noise = pink_noise(len(sig), sr, seed)
    noise *= np.sqrt(p_sig / (10 ** (target_snr_db / 10)))
    mixed = sig + noise
    clipped = int(np.sum(np.abs(mixed) > 32767))
    mixed_i16 = np.clip(np.round(mixed), -32768, 32767).astype(np.int16)
    actual_noise = mixed_i16.astype(np.float64) - sig
    p_noise = float(np.mean(actual_noise[mask] ** 2))
    measured = 10 * np.log10(p_sig / p_noise)
    return mixed_i16, {
        "target_snr_db": target_snr_db,
        "measured_snr_db": round(float(measured), 2),
        "snr_measured_over": "speech segments (turn offsets), pre-codec, at 8 kHz after band-limiting",
        "noise_seed": seed,
        "clipped_samples": clipped,
    }


def plan_chunks(offsets: list[tuple[float, float]], total_s: float, max_s: float = MAX_CHUNK_S,
                pad_s: float = CHUNK_PAD_S) -> list[dict]:
    """Greedily group consecutive turns into chunks no longer than max_s, splitting only at turn boundaries."""
    chunks: list[dict] = []
    current: list[int] = []

    def bounds(idxs: list[int]) -> tuple[float, float]:
        return max(0.0, offsets[idxs[0]][0] - pad_s), min(total_s, offsets[idxs[-1]][1] + pad_s)

    for i, (start, end) in enumerate(offsets):
        if end - start + 2 * pad_s > max_s:
            raise ValueError(f"turn {i} is {end - start:.1f}s, longer than the {max_s}s upload limit; shorten it")
        if current:
            s, e = bounds(current + [i])
            if e - s > max_s:
                chunks.append(current)
                current = []
        current.append(i)
    if current:
        chunks.append(current)
    out = []
    for k, idxs in enumerate(chunks):
        s, e = bounds(idxs)
        out.append({"chunk": k, "turns": idxs, "start_s": round(s, 4), "end_s": round(e, 4)})
    return out


def cut(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    samples, sr = read_wav(src)
    write_wav(dst, samples[int(round(start_s * sr)): int(round(end_s * sr))], sr)
