"""A polite JSON client: on-disk caching, a request interval and exponential backoff."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from wildlens.config import USER_AGENT

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS
    return isinstance(error, httpx.TransportError)


def cache_key(url: str, params: Mapping[str, Any] | None) -> str:
    query = urlencode(sorted((params or {}).items()), doseq=True)
    return hashlib.sha256(f"{url}?{query}".encode()).hexdigest()


class Throttle:
    """Spaces calls at least `interval` seconds apart, across threads."""

    def __init__(
        self,
        interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = interval
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            self._sleep(start - now)


class PoliteClient:
    """GET JSON with caching so reruns never repeat a successful request."""

    def __init__(
        self,
        cache_dir: Path,
        interval: float = 1.1,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.throttle = Throttle(interval)
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=60,
            follow_redirects=True,
            transport=transport,
        )
        self.network_requests = 0

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        path = self.cache_dir / f"{cache_key(url, params)}.json"
        if path.exists():
            return json.loads(path.read_text())
        data = self._fetch(url, params)
        # Writing through a temporary file means an interrupted run cannot leave a corrupt entry.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
        return data

    @retry(
        retry=retry_if_exception(is_retryable),
        wait=wait_exponential_jitter(multiplier=2, max=120),
        stop=stop_after_attempt(7),
        reraise=True,
    )
    def _fetch(self, url: str, params: Mapping[str, Any] | None) -> Any:
        self.throttle.wait()
        self.network_requests += 1
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
