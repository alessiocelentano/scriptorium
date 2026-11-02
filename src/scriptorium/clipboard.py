"""Copy text to the system clipboard without adding a dependency.

The TUI runs inside a terminal, so the most portable mechanism is the OSC 52
escape sequence.  Native helpers (``pbcopy`` on macOS, ``wl-copy``/``xclip``/
``xsel`` on Linux, ``clip`` on Windows) are preferred when they are installed
because they do not depend on terminal support.  Nothing here logs the copied
text.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence

_COPY_TIMEOUT = 5.0


def _candidate_commands() -> list[list[str]]:
    """Clipboard commands to try, most specific first."""
    if sys.platform == "darwin":
        return [["pbcopy"]]
    if os.name == "nt":
        return [["clip"]]
    return [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["clip.exe"],
    ]


def _run_command(command: Sequence[str], text: str) -> bool:
    try:
        completed = subprocess.run(
            list(command),
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_COPY_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _osc52_sequence(text: str) -> bytes:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"\x1b]52;c;{encoded}\x07".encode("ascii")


def _write_osc52(text: str) -> bool:
    """Write the OSC 52 sequence to the controlling terminal.

    ``sys.stdout`` is wrapped by the TUI log stream, so writing the escape
    there would show up as a log line instead of reaching the terminal.  A
    direct ``/dev/tty`` handle bypasses that wrapper and is also what makes
    the sequence work when stdout is redirected.
    """
    sequence = _osc52_sequence(text)
    try:
        descriptor = os.open("/dev/tty", os.O_WRONLY | getattr(os, "O_NOCTTY", 0))
    except OSError:
        return False
    try:
        os.write(descriptor, sequence)
    except OSError:
        return False
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    return True


def copy_to_clipboard(text: str) -> bool:
    """Copy ``text`` and report whether a mechanism accepted it."""
    for command in _candidate_commands():
        if shutil.which(command[0]) is None:
            continue
        if _run_command(command, text):
            return True
    return _write_osc52(text)
