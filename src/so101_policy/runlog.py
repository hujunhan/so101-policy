"""Copy a run's terminal output to a log file in the git-ignored logs/ directory, so Claude can read it.

Serial numbers are masked in the file (LeRobot prints its whole config, ports included). Progress bars that
redraw a line with carriage returns are kept only in their final state.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "logs"


def mask_serials(text: str) -> str:
    """Hide adapter and camera serial numbers (the repo is public; logs get pasted and quoted)."""
    text = re.sub(r"(USB_Single_Serial_)[^-\s\"']+", r"\1<SERIAL>", text)
    return re.sub(r"(Camera_)[0-9A-Za-z]{6,}", r"\1<SERIAL>", text)


class _Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
        self.pending = ""

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.pending += s
        *lines, self.pending = self.pending.split("\n")
        for line in lines:
            self.file.write(mask_serials(line.rsplit("\r", 1)[-1]) + "\n")
        self.file.flush()
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start(name: str, argv: list[str] | None = None) -> Path:
    """Tee stdout and stderr into logs/<name>_<timestamp>.log from now on. Returns the log path."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    f = path.open("w", buffering=1)
    f.write(f"# {name} {' '.join(argv if argv is not None else sys.argv[1:])}\n")
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    return path
