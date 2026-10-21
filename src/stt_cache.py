"""Content-addressed cache key for STT results (shared by run_spike, experiment and the app pipeline)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stt_cache_path(cache_dir: Path, audio_path: Path, language_code: str, fmt: str) -> tuple[Path, str]:
    """Cache file for one STT request, keyed on the audio's content (not its path), language and format.

    Returns (cache file, full SHA-256 hex digest of the audio bytes).
    """
    digest = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
    return cache_dir / f"{digest[:16]}_{language_code}_{fmt}.json", digest
