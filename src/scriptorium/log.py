"""Console and JSONL logging."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class Logger:
    def __init__(
        self,
        logfile: Path | None = None,
        *,
        verbose: bool = False,
        sink: Callable[[str], None] | None = None,
    ):
        self.logfile = logfile
        self.verbose = verbose
        self.sink = sink
        self._lock = threading.Lock()
        if logfile is not None:
            logfile.parent.mkdir(parents=True, exist_ok=True)

    def attach(self, logfile: Path) -> None:
        """Start writing JSONL events to ``logfile``."""
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self.logfile = logfile

    # -- internals ----------------------------------------------------------- #

    def event(self, name: str, **fields: Any) -> None:
        if self.logfile is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": name,
            **fields,
        }
        with self._lock:
            with self.logfile.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _print(self, prefix: str, message: str, stream: Any = None) -> None:
        line = f"{prefix} {message}"
        if self.sink is not None:
            self.sink(line)
            return
        stream = stream or sys.stdout
        print(line, file=stream, flush=True)

    # -- public -------------------------------------------------------------- #

    def info(self, message: str, **fields: Any) -> None:
        self._print("::", message)
        self.event("info", message=message, **fields)

    def step(self, message: str, **fields: Any) -> None:
        self._print(">>", message)
        self.event("step", message=message, **fields)

    def success(self, message: str, **fields: Any) -> None:
        self._print("[ok]", message)
        self.event("success", message=message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self._print("[warn]", message)
        self.event("warning", message=message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._print("[error]", message, sys.stderr)
        self.event("error", message=message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        if self.verbose:
            self._print("[debug]", message)
        self.event("debug", message=message, **fields)


class Timer:
    def __init__(self) -> None:
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def human(self) -> str:
        seconds = int(self.elapsed)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h{minutes:02d}m{seconds:02d}s"
        if minutes:
            return f"{minutes}m{seconds:02d}s"
        return f"{seconds}s"
