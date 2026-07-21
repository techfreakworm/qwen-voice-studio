"""Memory accounting that is honest on macOS.

Why this exists: process RSS *undercounts* MPS/IOKit allocations (a 4.5 GB model
showed 0.9 GB RSS), while ``total - available`` *overcounts* (it charges the huge
reclaimable file cache from the HF cache). The gauge that matches Activity
Monitor's "Memory Used" — and that actually tracks MPS demand — is
``wired + active + compressed`` from ``vm_stat``.

Used to keep local test/inference under the operator's 80 GB ceiling.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

_PAGE = 16384  # Apple-silicon page size (bytes)


def committed_gb() -> float:
    """Real committed memory in GB (Activity-Monitor "Memory Used").

    macOS: parses ``vm_stat`` (wired + active + compressed). Other platforms:
    falls back to psutil ``used``.
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout

            def pages(pattern: str) -> int:
                m = re.search(pattern + r":\s+(\d+)", out)
                return int(m.group(1)) if m else 0

            wired = pages(r"Pages wired down")
            active = pages(r"Pages active")
            comp = pages(r"Pages occupied by compressor")
            return (wired + active + comp) * _PAGE / 2**30
        except Exception:
            pass
    try:
        import psutil

        return psutil.virtual_memory().used / 2**30
    except Exception:
        return 0.0


def total_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:
        return 0.0


@dataclass
class MemorySnapshot:
    committed: float
    total: float

    def __str__(self) -> str:
        return f"{self.committed:.1f}/{self.total:.0f} GB committed"


def snapshot() -> MemorySnapshot:
    return MemorySnapshot(committed_gb(), total_gb())


class MemoryGuard:
    """Daemon thread that hard-aborts the process if committed memory crosses a
    ceiling — a backstop against OOM-ing the OS during local testing.

    ``on_trip`` defaults to ``os._exit`` because a soft exception cannot reliably
    unwind a runaway allocation.
    """

    def __init__(self, hard_gb: float = 76.0, soft_gb: float | None = 72.0, interval: float = 0.5):
        self.hard_gb = hard_gb
        self.soft_gb = soft_gb
        self.interval = interval
        self._warned = False
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while True:
            c = committed_gb()
            if c > self.hard_gb:
                print(f"[MemoryGuard] committed {c:.1f} GB > {self.hard_gb} GB -> HARD ABORT", flush=True)
                os._exit(97)
            if self.soft_gb and c > self.soft_gb and not self._warned:
                print(f"[MemoryGuard] WARN committed {c:.1f} GB > {self.soft_gb} GB (ceiling {self.hard_gb})", flush=True)
                self._warned = True
            time.sleep(self.interval)

    def start(self) -> "MemoryGuard":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self
