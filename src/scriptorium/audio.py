"""Video to MP3 conversion with ffmpeg, and media probing."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import AudioError


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool
    format_name: str = ""


def require_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise AudioError(
            "missing required tool(s): "
            + ", ".join(missing)
            + ". Install ffmpeg (for example: brew install ffmpeg)."
        )


def probe(path: Path) -> MediaInfo:
    require_tools()
    if not path.is_file():
        raise AudioError(f"media file not found: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed for {path.name}: {_tail(result.stderr)}")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AudioError(f"ffprobe returned invalid JSON for {path.name}") from exc

    streams = payload.get("streams") or []
    kinds = {stream.get("codec_type") for stream in streams}
    duration = _float((payload.get("format") or {}).get("duration"))
    if duration <= 0:
        for stream in streams:
            duration = max(duration, _float(stream.get("duration")))
    return MediaInfo(
        duration=duration,
        has_audio="audio" in kinds,
        has_video="video" in kinds,
        format_name=str((payload.get("format") or {}).get("format_name") or ""),
    )


def to_mp3(video: Path, destination: Path, *, bitrate: str = "64k") -> MediaInfo:
    """Convert any ffmpeg-readable media to mono 16 kHz MP3."""
    require_tools()
    source = probe(video)
    if not source.has_audio:
        raise AudioError(f"no audio stream found in {video.name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        bitrate,
        "-map_metadata",
        "-1",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.is_file():
        raise AudioError(f"ffmpeg conversion failed: {_tail(result.stderr)}")

    output = probe(destination)
    if output.duration <= 0:
        raise AudioError(f"converted audio has no duration: {destination}")
    tolerance = max(2.0, source.duration * 0.01)
    if source.duration > 0 and abs(output.duration - source.duration) > tolerance:
        raise AudioError(
            "converted audio duration differs from the source "
            f"({output.duration:.1f}s vs {source.duration:.1f}s)"
        )
    return output


def delete_media(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AudioError(f"cannot delete {path}: {exc}") from exc


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text
