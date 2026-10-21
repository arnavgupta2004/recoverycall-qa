"""Public-demo guards for the web app: passcode, per-IP rate limit, upload checks."""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

ALLOWED_EXTENSIONS = {".wav", ".mp3"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_AUDIO_S = 120.0


def passcode_configured() -> bool:
    return bool(os.environ.get("UPLOAD_PASSCODE", "").strip())


def passcode_ok(given: str | None) -> bool:
    """Live features are disabled when UPLOAD_PASSCODE is unset; otherwise compare in constant time."""
    expected = os.environ.get("UPLOAD_PASSCODE", "").strip()
    if not expected:
        return False
    return hmac.compare_digest((given or "").strip().encode("utf-8"), expected.encode("utf-8"))


class RateLimiter:
    """At most `limit` events per `window_s` seconds per key (client IP). Thread-safe, in memory."""

    def __init__(self, limit: int, window_s: float, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def try_acquire(self, key: str) -> tuple[bool, int]:
        """Record an event if allowed. Returns (allowed, seconds until the next slot frees up)."""
        now = self.clock()
        with self._lock:
            q = self._events[key]
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False, int(self.window_s - (now - q[0])) + 1
            q.append(now)
            return True, 0


def client_ip(request, trusted_hops: int | None = None) -> str:
    """Client IP for rate limiting, from X-Forwarded-For.

    Proxies append the address they received the request from, so the rightmost entries are
    written by infrastructure and anything to their left can be forged by the client. Cloud
    Run's front end appends the client address it saw, so the default takes the LAST entry;
    set XFF_TRUSTED_HOPS=n when n further trusted proxies (e.g. a load balancer) append after it.
    Falls back to the socket peer when the header is absent (local runs).
    """
    if request is None:
        return "unknown"
    if trusted_hops is None:
        trusted_hops = int(os.environ.get("XFF_TRUSTED_HOPS", "0"))
    headers = getattr(request, "headers", None) or {}
    fwd = headers.get("x-forwarded-for") if hasattr(headers, "get") else None
    hops = [h.strip() for h in fwd.split(",") if h.strip()] if fwd else []
    if hops:
        return hops[max(0, len(hops) - 1 - trusted_hops)]
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


def check_upload(path: str | Path | None) -> str | None:
    """Return an error message for an unacceptable upload, or None if it may be analysed."""
    if not path:
        return "Choose a WAV or MP3 file first."
    p = Path(path)
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return "Only WAV or MP3 files are accepted."
    if p.stat().st_size > MAX_UPLOAD_BYTES:
        return "The file is larger than 20 MB."
    return None
