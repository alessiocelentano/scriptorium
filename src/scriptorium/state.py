"""Resumable job state stored as JSON on disk."""

from __future__ import annotations

import os
import re
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .errors import StageError
from .sources import MEDIA_SUFFIXES


class Stage(StrEnum):
    PREPARE = "prepare"
    SOURCES = "sources"
    AUDIO = "audio"
    TRANSCRIBE = "transcribe"
    EXTRACT = "extract"
    GROUP = "group"
    OUTLINE = "outline"
    COMPOSE = "compose"
    APPLY = "apply"
    FLASHCARDS = "flashcards"
    REPORT = "report"
    CLEANUP = "cleanup"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.SOURCES,
    Stage.PREPARE,
    Stage.AUDIO,
    Stage.TRANSCRIBE,
    Stage.EXTRACT,
    Stage.GROUP,
    Stage.OUTLINE,
    Stage.COMPOSE,
    Stage.APPLY,
    Stage.FLASHCARDS,
    Stage.CLEANUP,
    Stage.REPORT,
)

Status = str  # pending | running | done | failed | skipped

STAGE_LABELS: dict[Stage, str] = {
    Stage.PREPARE: "Resolve course and vault context",
    Stage.SOURCES: "Read source material",
    Stage.AUDIO: "Convert media to audio",
    Stage.TRANSCRIBE: "Transcribe with the local speech-to-text model",
    Stage.EXTRACT: "Extract knowledge units",
    Stage.GROUP: "Split sources into topic groups",
    Stage.OUTLINE: "Plan vault placement",
    Stage.COMPOSE: "Compose notes and cards",
    Stage.APPLY: "Apply note writes",
    Stage.FLASHCARDS: "Write draft flashcards",
    Stage.REPORT: "Write report",
    Stage.CLEANUP: "Clean intermediate artifacts",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


GROUP_STAGES: tuple[Stage, ...] = (
    Stage.OUTLINE,
    Stage.COMPOSE,
    Stage.APPLY,
    Stage.FLASHCARDS,
)

GLOBAL_STAGES: tuple[Stage, ...] = (
    Stage.SOURCES,
    Stage.PREPARE,
    Stage.AUDIO,
    Stage.TRANSCRIBE,
    Stage.EXTRACT,
    Stage.GROUP,
)


class GroupState(BaseModel):
    """One topic group: a subset of sources processed as a unit."""

    model_config = ConfigDict(extra="ignore")

    id: str
    topic: str = ""
    source_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    order: int = 0
    stages: dict[str, str] = Field(default_factory=dict)

    def status(self, stage: Stage | str) -> Status:
        key = stage.value if isinstance(stage, Stage) else str(stage)
        return self.stages.get(key, "pending")

    def set_status(self, stage: Stage | str, status: Status) -> None:
        key = stage.value if isinstance(stage, Stage) else str(stage)
        self.stages[key] = status


class JobState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    video: str
    course: str
    moc: str | None = None
    sources: list[str] = Field(default_factory=list)
    groups: list[GroupState] = Field(default_factory=list)
    stages: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    warnings: list[dict[str, str]] = Field(default_factory=list)
    options: dict[str, object] = Field(default_factory=dict)

    def status(self, stage: Stage | str) -> Status:
        key = stage.value if isinstance(stage, Stage) else str(stage)
        return self.stages.get(key, "pending")

    def set_status(self, stage: Stage | str, status: Status) -> None:
        key = stage.value if isinstance(stage, Stage) else str(stage)
        self.stages[key] = status
        self.updated_at = _now()

    def next_stage(self) -> Stage | None:
        for stage in STAGE_ORDER:
            if self.status(stage) != "done":
                return stage
        return None

    def completed(self) -> bool:
        if any(self.status(stage) != "done" for stage in GLOBAL_STAGES):
            return False
        if not self.groups:
            return False
        return all(
            group.status(stage) in {"done", "skipped"}
            for group in self.groups
            for stage in GROUP_STAGES
        )


def slugify(value: str, limit: int = 40) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return (value or "lecture")[:limit].strip("-") or "lecture"


class JobStore:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.jobs_dir = self.home / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------- #

    def job_dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", job_id):
            raise StageError(f"invalid job id: {job_id!r}")
        return self.jobs_dir / job_id

    def artifact(self, job_id: str, name: str) -> Path:
        return self.job_dir(job_id) / name

    # -- lifecycle ----------------------------------------------------------- #

    def create(self, sources: list[Path], course: str, moc: str | None = None) -> JobState:
        if not sources:
            raise StageError("a job needs at least one source")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{slugify(sources[0].stem, 36)}"
        job_id = base
        counter = 2
        while self.job_dir(job_id).exists():
            job_id = f"{base}-{counter}"
            counter += 1
        first_media = next(
            (path for path in sources if path.suffix.lower() in MEDIA_SUFFIXES), None
        )
        state = JobState(
            job_id=job_id,
            video=str(first_media or ""),
            course=course,
            moc=moc,
            sources=[str(Path(path).expanduser()) for path in sources],
        )
        directory = self.job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "logs").mkdir()
        self.save(state)
        return state

    def save(self, state: JobState) -> None:
        directory = self.job_dir(state.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        state.updated_at = _now()
        _atomic_write(directory / "state.json", state.model_dump_json(indent=2) + "\n")

    def load(self, job_id: str) -> JobState:
        path = self.job_dir(job_id) / "state.json"
        if not path.is_file():
            raise StageError(f"job not found: {job_id}")
        return JobState.model_validate_json(path.read_text(encoding="utf-8"))

    def recent(self, limit: int = 20) -> list[JobState]:
        states: list[JobState] = []
        for path in sorted(self.jobs_dir.glob("*/state.json"), reverse=True)[:limit]:
            try:
                states.append(JobState.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return states

    # -- single-run lock ----------------------------------------------------- #

    def lock(self, job_id: str):
        return _JobLock(self.job_dir(job_id) / "lock")


class _JobLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self) -> _JobLock:
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
            except (OSError, ValueError):
                pid = -1
            if pid > 0 and _pid_alive(pid):
                raise StageError(f"job is already running (pid {pid})")
        self.path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        self.acquired = True
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def job_age_seconds(state: JobState) -> float:
    try:
        created = datetime.fromisoformat(state.created_at)
    except ValueError:
        return 0.0
    return max(0.0, time.time() - created.timestamp())
