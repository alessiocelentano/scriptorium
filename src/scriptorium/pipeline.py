"""Stage orchestration for one lecture job."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .apply import ApplyResult, apply_plan
from .asr import ApiTranscriber, AsrSettings, LocalTranscriber, build_bias_prompt
from .audio import to_mp3
from .config import LANGUAGE_ASK, SCANNED_PDF_VISION, Config, asr_api_key
from .context import ContextBundle, build_context, load_glossary
from .errors import (
    ConfigurationError,
    FlashcardError,
    LLMError,
    PlanValidationError,
    ScriptoriumError,
    StageError,
)
from .flashcards import CardWriteResult, write_drafts
from .history import record_run, state_status
from .llm.client import DeepSeekClient
from .log import Logger, Timer
from .models import ExtractResult, Grouping, Outline, Plan, SourceGroup, TodoItem, Transcript
from .pdf_images import ExtractedImage, extract_pdf_images, publish_images
from .planning import (
    build_groups,
    build_outline,
    compose_notes,
    extract_sources,
    units_for_sources,
)
from .sandbox import SandboxVault
from .scanned import ScannedPdfReader
from .sources import SourceFile, TextSource, clean_course_name, list_sources, read_source
from .state import (
    GLOBAL_STAGES,
    GROUP_STAGES,
    GroupState,
    JobState,
    JobStore,
    Stage,
    slugify,
)
from .ui import MOC_CONFIDENCE, ConsoleUI, moc_title
from .vault import ChangeRecord, Vault
from .warnings import RunWarning, WarningCode, WarningCollector


@dataclass
class RunOptions:
    course: str = ""
    moc: str | None = None
    dry_run: bool | None = None
    accept_merges: bool | None = None
    no_moc: bool | None = None
    keep_audio: bool | None = None
    keep_srt: bool | None = None


class Pipeline:
    def __init__(
        self,
        config: Config,
        ui: ConsoleUI,
        logger: Logger,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config
        self.ui = ui
        self.logger = logger
        self.on_event = on_event
        self.store = JobStore(config.artifacts_dir)
        self.vault = Vault(config.notes_root)
        self._job_language = config.language
        self._cache: dict[str, Any] = {}
        self.warnings = WarningCollector()
        self.sandbox: SandboxVault | None = None

    # -- entry points -------------------------------------------------------- #

    def run(self, sources: Sequence[Path], options: RunOptions) -> JobState:
        resolved = [Path(source).expanduser() for source in sources]
        if not resolved:
            raise ConfigurationError("at least one source path is required")
        missing = [source for source in resolved if not source.is_file()]
        if missing:
            raise ConfigurationError(f"source not found: {missing[0]}")
        course = options.course.strip() or clean_course_name(resolved[0].stem)

        state = self.store.create(resolved, course, options.moc)
        state.options = {key: value for key, value in asdict(options).items() if value is not None}
        state.options.pop("course", None)
        self.store.save(state)
        self._write_job_snapshot(state)
        return self._execute(state)

    def resume(self, job_id: str, options: RunOptions) -> JobState:
        state = self.store.load(job_id)
        missing = [Path(source) for source in state.sources if not Path(source).is_file()]
        if missing:
            raise ConfigurationError(f"source not found: {missing[0]}")
        for key, value in asdict(options).items():
            if key == "course" or value is None:
                continue
            state.options[key] = value
        if options.dry_run is None:
            state.options["dry_run"] = False
        self.store.save(state)
        return self._execute(state)

    # -- execution ----------------------------------------------------------- #

    def _execute(self, state: JobState) -> JobState:
        logfile = self.store.artifact(state.job_id, "logs/run.jsonl")
        self.logger.attach(logfile)
        self._cache = {}
        self._install_state(state)
        self._job_language = str(state.options.get("language") or self.config.language)
        self.warnings = WarningCollector(disabled=self.config.disabled_warnings)
        self.warnings.extend(RunWarning.from_dict(item) for item in state.warnings)
        self.vault = Vault(self.config.notes_root)
        self.sandbox = None
        if self._flag(state, "dry_run"):
            self.sandbox = SandboxVault.create(self.vault)
            self.vault = self.sandbox
            self.logger.info(f"dry run: simulating against {self.sandbox.directory}")
        self._emit(
            "job_started",
            job_id=state.job_id,
            course=state.course,
            sources=list(state.sources),
        )

        with self.store.lock(state.job_id):
            try:
                self._record_history(state, status="running")
                self._execute_stages(state)
                status, error = state_status(state)
                self._record_history(state, status=status, error=error)
            except BaseException:
                self._record_history(
                    state,
                    status="failed",
                    error=state.error or "interrupted",
                )
                raise
            finally:
                if self.sandbox is not None:
                    self.sandbox.discard()
                    self.logger.debug("dry run sandbox discarded")
                    self.sandbox = None
                    self.vault = Vault(self.config.notes_root)

        self._emit("job_done", job_id=state.job_id)
        return state

    def _execute_stages(self, state: JobState) -> None:
        """Run every stage the state has not finished yet."""
        for stage in GLOBAL_STAGES:
            if state.status(stage) == "done":
                self.logger.debug(f"skipping completed stage: {stage.value}")
                continue
            self._run_one(state, stage)

        self._ensure_groups(state)
        for index, group in enumerate(state.groups, start=1):
            self._emit(
                "group_start",
                group_id=group.id,
                topic=group.topic,
                index=index,
                total=len(state.groups),
            )
            for stage in GROUP_STAGES:
                if group.status(stage) == "done":
                    continue
                if self._flag(state, "dry_run") and stage is Stage.FLASHCARDS:
                    group.set_status(stage, "skipped")
                    self.store.save(state)
                    continue
                self._run_one(state, stage, group)
                if self._flag(state, "dry_run") and stage is Stage.APPLY:
                    # The simulated apply must not count as the real one.
                    group.set_status(stage, "skipped")
                    self.store.save(state)

        for stage in (Stage.CLEANUP, Stage.REPORT):
            if state.status(stage) == "done":
                continue
            self._run_one(state, stage)

        if self._flag(state, "dry_run"):
            self.logger.info(
                f"dry run complete; apply with: scriptorium resume {state.job_id} --yes"
            )

    def _run_one(self, state: JobState, stage: Stage, group: GroupState | None = None) -> None:
        if group is not None:
            group.set_status(stage, "running")
        else:
            state.set_status(stage, "running")
        state.error = None
        self.store.save(state)
        where = f" [{group.topic or group.id}]" if group is not None else ""
        self.logger.step(f"{stage.value}{where}: {stage_label(stage)}")
        self._emit(
            "stage_start",
            stage=stage.value,
            group=group.id if group is not None else None,
            group_topic=group.topic if group is not None else "",
            total_groups=len(state.groups),
        )
        timer = Timer()
        try:
            self._run_stage(stage, state, group)
        except ScriptoriumError as exc:
            self._fail(state, stage, exc, group)
            raise
        except Exception as exc:  # unexpected: keep the job resumable
            self._fail(state, stage, exc, group)
            raise StageError(f"{stage.value}: {exc}") from exc
        if group is not None:
            group.set_status(stage, "done")
        else:
            state.set_status(stage, "done")
            if stage is Stage.GROUP:
                self._sync_groups(state)
        self._save_warnings(state)
        self.store.save(state)
        self.logger.success(f"{stage.value} completed in {timer.human()}")
        self._emit(
            "stage_done",
            stage=stage.value,
            seconds=round(timer.elapsed, 1),
            group=group.id if group is not None else None,
        )

    def _fail(
        self,
        state: JobState,
        stage: Stage,
        exc: Exception,
        group: GroupState | None = None,
    ) -> None:
        if group is not None:
            group.set_status(stage, "failed")
        else:
            state.set_status(stage, "failed")
        state.error = f"{stage.value}: {exc}"
        self._save_warnings(state)
        self.store.save(state)
        self.logger.error(state.error)
        self._emit(
            "stage_failed",
            stage=stage.value,
            error=str(exc),
            group=group.id if group is not None else None,
        )
        self._write_report(state, failure=exc)

    def _run_stage(self, stage: Stage, state: JobState, group: GroupState | None = None) -> None:
        if stage is Stage.SOURCES:
            self._sources(state)
        elif stage is Stage.PREPARE:
            self._prepare(state)
        elif stage is Stage.AUDIO:
            self._audio(state)
        elif stage is Stage.TRANSCRIBE:
            self._transcribe(state)
        elif stage is Stage.EXTRACT:
            self._extract(state)
        elif stage is Stage.GROUP:
            self._group(state)
        elif stage is Stage.OUTLINE:
            self._outline(state, self._group_or_error(group))
        elif stage is Stage.COMPOSE:
            self._compose(state, self._group_or_error(group))
        elif stage is Stage.APPLY:
            self._apply(state, self._group_or_error(group))
        elif stage is Stage.FLASHCARDS:
            self._flashcards(state, self._group_or_error(group))
        elif stage is Stage.CLEANUP:
            self._cleanup(state)
        elif stage is Stage.REPORT:
            self._report(state)
        else:
            raise StageError(f"unknown stage: {stage}")

    @staticmethod
    def _group_or_error(group: GroupState | None) -> GroupState:
        if group is None:
            raise StageError("internal error: this stage needs a topic group")
        return group

    def _ensure_groups(self, state: JobState) -> None:
        if state.groups:
            return
        path = self.store.artifact(state.job_id, "04.groups.json")
        if path.is_file():
            payload = self._read_json_artifact(path, "groups artifact")
            state.groups = [GroupState(**item) for item in payload.get("groups", [])]
            self.store.save(state)
        if not state.groups:
            raise StageError("no topic groups were produced for this job")

    def _sync_groups(self, state: JobState) -> None:
        path = self.store.artifact(state.job_id, "04.groups.json")
        if not path.is_file():
            return
        payload = self._read_json_artifact(path, "groups artifact")
        proposed = [GroupState(**item) for item in payload.get("groups", [])]
        if proposed:
            state.groups = proposed
            self.store.save(state)

    def _group(self, state: JobState) -> None:
        texts = self._text_sources(state)
        extract = self._require("extract")
        grouping: Grouping | None = None
        try:
            grouping = build_groups(
                texts, extract, self._client(), self.logger, course=state.course
            )
        except (LLMError, PlanValidationError) as exc:
            self._warn(
                RunWarning(
                    code=WarningCode.GROUPING_FALLBACK,
                    message=(
                        f"Topic grouping failed ({exc}); all sources are processed as one group."
                    ),
                )
            )
        if grouping is None or not grouping.groups:
            grouping = Grouping(
                groups=[
                    SourceGroup(
                        id="g1",
                        topic=state.course,
                        source_ids=[
                            source.id for source in self._source_files(state) if source.kind
                        ],
                        rationale="fallback group with every source",
                    )
                ]
            )
        state.groups = [
            GroupState(
                id=group.id,
                topic=group.topic,
                source_ids=list(group.source_ids),
                rationale=group.rationale,
                order=index,
            )
            for index, group in enumerate(grouping.groups, start=1)
        ]
        self._cache["grouping_todos"] = list(grouping.todos)
        self._write_json(
            self.store.artifact(state.job_id, "04.groups.json"),
            json.dumps(
                {
                    "groups": [group.model_dump() for group in grouping.groups],
                    "todos": [todo.model_dump() for todo in grouping.todos],
                },
                indent=2,
            ),
        )
        for group in state.groups:
            self.logger.info(f"group {group.id}: {group.topic} ({', '.join(group.source_ids)})")

    # -- stages -------------------------------------------------------------- #

    def _sources(self, state: JobState) -> None:
        """Read every document source; media text arrives with transcription."""
        manifest: list[dict[str, object]] = []
        for source in self._source_files(state):
            entry: dict[str, object] = {
                "id": source.id,
                "name": source.name,
                "kind": source.kind,
                "path": str(source.path),
            }
            if source.kind == "document":
                result = read_source(source, page_reader=self._page_reader(state, source))
                for warning in result.warnings:
                    self._warn(warning)
                self._write_text(
                    self.store.artifact(state.job_id, f"sources/{source.id}.txt"), result.text
                )
                entry["chars"] = len(result.text)
                self.logger.info(f"source {source.id}: {source.name} ({len(result.text)} chars)")
                if source.path.suffix.lower() == ".pdf":
                    images = self._harvest_images(state, source)
                    entry["images"] = len(images)
            manifest.append(entry)
        self._write_json(
            self.store.artifact(state.job_id, "sources.json"),
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )

    def _prepare(self, state: JobState) -> None:
        moc = state.moc
        if moc and Path(moc).is_absolute():
            try:
                moc = self.vault.relative(Path(moc))
            except ValueError as exc:
                raise ConfigurationError(f"MOC is outside the vault: {moc}") from exc
            state.moc = moc
        if moc:
            if not self.vault.exists(moc):
                raise ConfigurationError(f"MOC not found in the vault: {moc}")
            self.logger.info(f"MOC from the input: {moc}")
        else:
            matches = self.vault.match_moc(self._moc_query(state), roots=self.config.search_paths)
            top = matches[0] if matches else None
            if top is None or top.score < MOC_CONFIDENCE:
                self.logger.info("no MOC matched the source material confidently")
                self._warn(
                    RunWarning(
                        code=WarningCode.MOC_NOT_FOUND,
                        message=(
                            "No existing MOC matched this material confidently; "
                            "a new MOC can be created for the course."
                        ),
                        detail=", ".join(f"{match.path} ({match.score:.2f})" for match in matches),
                    )
                )
            elif len(matches) > 1 and top.score - matches[1].score < 0.05:
                self._warn(
                    RunWarning(
                        code=WarningCode.MOC_AMBIGUOUS,
                        message=(
                            f"Two MOCs match this material almost equally: "
                            f"{top.path} and {matches[1].path}."
                        ),
                        detail=f"scores {top.score:.2f} and {matches[1].score:.2f}",
                    )
                )

            decision = self.ui.choose_moc(
                state.course, matches, default_title=f"{state.course} MOC"
            )
            if decision.action == "abort" or not (decision.path or decision.title):
                raise ConfigurationError("aborted: no MOC selected")
            if decision.action == "create":
                moc = self._create_moc(state, decision.title)
                self._warn(
                    RunWarning(
                        code=WarningCode.MOC_CREATED,
                        message=f"Created a new MOC: {moc}",
                        severity="info",
                        source=moc,
                    )
                )
            else:
                moc = decision.path or ""
                if not moc:
                    raise ConfigurationError("no MOC selected")
            state.moc = moc

        if self.config.language == LANGUAGE_ASK and not state.options.get("language"):
            answer = self.ui.ask_language("en")
            state.options["language"] = "" if answer in {"detect", "auto"} else answer
            self._job_language = str(state.options["language"])
            self.store.save(state)

        self.logger.info(f"course: {state.course}")
        self.logger.info(f"MOC: {moc}")
        state.artifacts["moc"] = moc
        self.store.save(state)
        self._emit("moc_selected", moc=moc)

    def _moc_query(self, state: JobState) -> str:
        """Course name, source names and readable document text, for MOC matching.

        The MOC is chosen before transcription, so video content only contributes
        its file name; documents are already readable at this point.
        """
        parts = [state.course]
        for source in self._source_files(state):
            parts.append(source.path.stem)
        for text in self._text_sources(state):
            if text.kind == "document" and text.text.strip():
                parts.append(text.text[:4000])
        return "\n".join(parts)

    def _moc_directory(self) -> str:
        """Where a new MOC is created: a configured root, else University/, else root."""
        vault_root = Path(self.config.notes_root).resolve()
        for root in self.config.search_paths:
            resolved = Path(root).expanduser()
            resolved = (
                resolved.resolve() if resolved.is_absolute() else (vault_root / resolved).resolve()
            )
            if resolved == vault_root or not resolved.is_relative_to(vault_root):
                continue
            if resolved.is_dir():
                return resolved.relative_to(vault_root).as_posix()
        university = vault_root / "University"
        if university.is_dir():
            return "University"
        return "."

    def _create_moc(self, state: JobState, title: str) -> str:
        title = moc_title(title, f"{state.course} MOC")
        directory = self._moc_directory()
        relative = f"{directory}/{title}.md" if directory != "." else f"{title}.md"
        if self.vault.exists(relative):
            self.logger.info(f"MOC already exists, using it: {relative}")
            return relative
        self.vault.write_new(relative, f"# {title}\n\n## Table of Contents\n")
        self.vault.refresh()
        self.logger.success(f"created MOC {relative}")
        return relative

    def _audio(self, state: JobState) -> None:
        media = [source for source in self._source_files(state) if source.kind == "media"]
        if not media:
            self.logger.info("no media sources; this run only reads documents")
            return
        raw_durations = state.options.get("audio_durations")
        durations: dict[str, float] = {}
        if isinstance(raw_durations, dict):
            durations = {str(key): float(value) for key, value in raw_durations.items()}
        total = 0.0
        for source in media:
            destination = self.store.artifact(state.job_id, f"sources/{source.id}/audio.mp3")
            if destination.is_file() and source.id in durations:
                duration = float(durations[source.id])
                self.logger.info(f"{source.name}: reusing audio ({duration:.0f}s)")
            else:
                if not source.path.is_file():
                    raise StageError(f"source media missing: {source.path}")
                info = to_mp3(source.path, destination)
                duration = info.duration
                durations[source.id] = duration
                self.logger.info(f"{source.name}: {duration:.0f}s mono 16 kHz")
            total += duration
        state.options["audio_durations"] = durations
        state.options["audio_duration"] = total
        self.store.save(state)

    def _transcribe(self, state: JobState) -> None:
        media = [source for source in self._source_files(state) if source.kind == "media"]
        if not media:
            self.logger.info("no media sources; nothing to transcribe")
            return

        transcripts: dict[str, Transcript] = {}
        pending: list[SourceFile] = []
        for source in media:
            transcript_path = self.store.artifact(
                state.job_id, f"sources/{source.id}/transcript.json"
            )
            if transcript_path.is_file():
                transcripts[source.id] = Transcript.model_validate_json(
                    transcript_path.read_text(encoding="utf-8")
                )
                self.logger.info(f"{source.name}: reusing transcript artifact")
                continue
            audio_path = self.store.artifact(state.job_id, f"sources/{source.id}/audio.mp3")
            if not audio_path.is_file():
                raise StageError(f"audio artifact missing for {source.name}")
            pending.append(source)

        if pending:
            self._transcribe_pending(state, pending, transcripts)

        combined = "\n\n".join(
            f"# {source.name}\n{transcripts[source.id].timed_text()}"
            for source in media
            if source.id in transcripts
        )
        self._write_text(self.store.artifact(state.job_id, "02.transcript.txt"), combined + "\n")
        state.artifacts["transcript"] = "02.transcript.txt"
        if len(media) == 1 and media[0].id in transcripts:
            transcript = transcripts[media[0].id]
            self._write_json(
                self.store.artifact(state.job_id, "02.transcript.json"),
                transcript.model_dump_json(indent=2),
            )
            self._write_text(
                self.store.artifact(state.job_id, "02.transcript.srt"), transcript.to_srt()
            )
            state.artifacts["transcript_json"] = "02.transcript.json"
            state.artifacts["srt"] = "02.transcript.srt"
        self.store.save(state)

    def _transcribe_pending(
        self, state: JobState, pending: list[SourceFile], transcripts: dict[str, Transcript]
    ) -> None:
        """Transcribe the sources without an artifact; write what succeeds.

        The transcriber owns the request budget and the workers never touch the
        job: they return transcripts and the artifacts are written here, in
        source order, so a concurrent backend cannot interleave two files.
        """
        finished: set[str] = set()
        paths = {source.id: str(source.path) for source in pending}

        def on_progress(source_id: str, *, state: str, done: int, total: int) -> None:
            if state in {"done", "failed"}:
                finished.add(source_id)
            self._emit(
                "source_progress",
                source_id=source_id,
                path=paths.get(source_id, ""),
                state=state,
                done=done,
                total=total,
                sources_done=len(finished),
                sources_total=len(pending),
            )

        batch = self._transcriber().transcribe_many(
            [
                (source.id, self.store.artifact(state.job_id, f"sources/{source.id}/audio.mp3"))
                for source in pending
            ],
            prompt=self._bias_prompt(state),
            on_progress=on_progress,
        )

        for source in pending:
            transcript = batch.transcripts.get(source.id)
            if transcript is None:
                continue
            transcripts[source.id] = transcript
            self._write_json(
                self.store.artifact(state.job_id, f"sources/{source.id}/transcript.json"),
                transcript.model_dump_json(indent=2),
            )
            self._write_text(
                self.store.artifact(state.job_id, f"sources/{source.id}.txt"),
                transcript.timed_text() + "\n",
            )

        if batch.failures:
            names = {source.id: source.name for source in pending}
            detail = "; ".join(
                f"{names.get(source_id, source_id)} ({reason})"
                for source_id, reason in batch.failures.items()
            )
            raise StageError(
                f"transcription failed for {len(batch.failures)} source(s): {detail}. "
                "The other transcripts are saved; `scriptorium resume` retries only "
                "the missing ones"
            )

    def _extract(self, state: JobState) -> None:
        texts = self._text_sources(state)
        self._cache["corpus"] = "\n\n".join(
            f"# {source.name}\n{source.text}" for source in texts if source.text
        )

        images = [image for text in texts for image in text.images]
        if images:
            self._check_image_capability(images)

        extract = extract_sources(
            texts,
            self._client(),
            self.logger,
            course=state.course,
        )
        self._cache["extract"] = extract
        self._write_json(
            self.store.artifact(state.job_id, "03.extract.json"),
            extract.model_dump_json(indent=2),
        )

    def _context_for(self, state: JobState, group: GroupState) -> ContextBundle:
        contexts = self._cache.setdefault("contexts", {})
        context = contexts.get(group.id)
        if context is None:
            context = build_context(
                self.vault,
                state.course,
                state.moc or "",
                str(self._cache.get("corpus") or ""),
                self.config,
            )
            contexts[group.id] = context
            self._write_text(
                self.store.artifact(state.job_id, f"03.context.{group.id}.md"),
                _context_markdown(context),
            )
        return context

    def _outline(self, state: JobState, group: GroupState) -> None:
        extract = self._require("extract")
        names = {
            source.name
            for source in self._source_files(state)
            if source.id in set(group.source_ids)
        }
        units = units_for_sources(extract, names)
        if units:
            context = self._context_for(state, group)
            subset = ExtractResult(units=units, asides=extract.asides, topics=extract.topics)
            outline = build_outline(
                context,
                subset,
                self._client(),
                self.logger,
                self.vault,
                config=self.config,
                chooser=self.ui.choose_target,
            )
        else:
            outline = Outline(
                notes=[],
                todos=[
                    TodoItem(
                        kind="review",
                        text=(
                            f"No knowledge units were extracted for group "
                            f"'{group.topic or group.id}'."
                        ),
                    )
                ],
            )
        if group.order == 1:
            outline.todos.extend(self._cache.get("grouping_todos") or [])
        self._cache.setdefault("outlines", {})[group.id] = outline
        self._write_json(
            self.store.artifact(state.job_id, f"04.outline.{group.id}.json"),
            outline.model_dump_json(indent=2),
        )

    def _outline_for(self, state: JobState, group: GroupState) -> Outline:
        outlines = self._cache.setdefault("outlines", {})
        outline = outlines.get(group.id)
        if outline is None:
            outline = Outline.model_validate_json(
                self.store.artifact(state.job_id, f"04.outline.{group.id}.json").read_text(
                    encoding="utf-8"
                )
            )
            outlines[group.id] = outline
        return outline

    def _compose(self, state: JobState, group: GroupState) -> None:
        outline = self._outline_for(state, group)
        if outline.notes:
            plan = compose_notes(
                outline,
                self._require("extract"),
                self._context_for(state, group),
                self._client(),
                self.logger,
                self.vault,
                self.config,
            )
        else:
            plan = Plan(
                course=state.course,
                moc_path=state.moc,
                notes=[],
                todos=list(outline.todos),
            )
        self._cache.setdefault("plans", {})[group.id] = plan
        self._save_plan(state, plan, group)

    def _plan_for(self, state: JobState, group: GroupState) -> Plan:
        plans = self._cache.setdefault("plans", {})
        plan = plans.get(group.id)
        if plan is None:
            plan = Plan.model_validate_json(
                self.store.artifact(state.job_id, f"05.plan.{group.id}.json").read_text(
                    encoding="utf-8"
                )
            )
            plans[group.id] = plan
        return plan

    def _apply(self, state: JobState, group: GroupState) -> None:
        plan = self._plan_for(state, group)
        result = apply_plan(
            plan,
            self.vault,
            self.ui,
            self.logger,
            accept_merges=self._flag(state, "accept_merges") or self._flag(state, "dry_run"),
            update_moc=not self._flag(state, "no_moc"),
            conflict_dir=self.store.job_dir(state.job_id) / "merges",
            merge_policy=self.config.merge_policy,
        )
        self._publish_images(state, plan, result)
        self._cache.setdefault("applies", {})[group.id] = result
        self._write_json(
            self.store.artifact(state.job_id, f"apply.{group.id}.json"),
            json.dumps(_apply_dict(result), indent=2),
        )
        self._write_merged_apply(state)
        for record in [*result.created, *result.merged, *result.moc]:
            self._emit(
                "file_changed",
                path=record.path,
                action=record.action,
                added_lines=record.added_lines,
                removed_lines=record.removed_lines,
                diff=record.diff,
            )
        for path in result.skipped_merges:
            self._emit("file_skipped", path=path, reason="merge needs review")
        for path in result.conflicts:
            self._emit("merge_conflict", path=path, reason="manual merge requested")
        for path in result.refused:
            self._emit("file_skipped", path=path, reason="write refused")

    def _apply_for(self, state: JobState, group: GroupState) -> ApplyResult:
        applies = self._cache.setdefault("applies", {})
        result = applies.get(group.id)
        if result is None:
            result = self._load_apply(group.id)
            applies[group.id] = result
        return result

    def _write_merged_apply(self, state: JobState) -> None:
        applies = self._cache.get("applies") or {}
        merged = ApplyResult()
        for group in state.groups:
            result = applies.get(group.id)
            if result is None:
                continue
            merged.created.extend(result.created)
            merged.merged.extend(result.merged)
            merged.moc.extend(result.moc)
            merged.applied_paths.extend(result.applied_paths)
            merged.skipped_merges.extend(result.skipped_merges)
            merged.conflicts.extend(result.conflicts)
            merged.refused.extend(result.refused)
            merged.todos.extend(result.todos)
        self._cache["apply_all"] = merged
        self._write_json(
            self.store.artifact(state.job_id, "apply.json"),
            json.dumps(_apply_dict(merged), indent=2),
        )

    def _flashcards(self, state: JobState, group: GroupState) -> None:
        plan = self._plan_for(state, group)
        applied = self._apply_for(state, group)
        results: list[CardWriteResult] = []

        for path in applied.applied_paths:
            note = plan.note(path)
            if note is None or not note.flashcards:
                continue
            try:
                results.append(
                    write_drafts(
                        self.vault,
                        path,
                        note.flashcards,
                        self.config,
                        self.logger,
                        extra_tags=note.tags,
                    )
                )
            except FlashcardError as exc:
                self.logger.warn(str(exc))

        self._cache.setdefault("cards", {})[group.id] = results
        if any(result.written == 0 and result.requested for result in results):
            for result in results:
                if result.written == 0 and result.requested:
                    plan.todos.append(
                        TodoItem(
                            kind="tag",
                            text=(
                                f"No draft cards were written to {result.path}; check the "
                                "flashcard tag and card validity."
                            ),
                            note_path=result.path,
                        )
                    )
            self._save_plan(state, plan, group)
        all_results = [item for items in self._cache["cards"].values() for item in items]
        state.artifacts["cards"] = str(sum(result.written for result in all_results))
        self.store.save(state)
        self._write_json(
            self.store.artifact(state.job_id, f"cards.{group.id}.json"),
            json.dumps(
                [
                    {
                        "path": result.path,
                        "requested": result.requested,
                        "written": result.written,
                        "skipped": result.skipped,
                        "writer": result.writer,
                    }
                    for result in results
                ],
                indent=2,
            ),
        )
        self._write_json(
            self.store.artifact(state.job_id, "cards.json"),
            json.dumps(
                [
                    {
                        "path": result.path,
                        "requested": result.requested,
                        "written": result.written,
                        "skipped": result.skipped,
                        "writer": result.writer,
                    }
                    for result in all_results
                ],
                indent=2,
            ),
        )
        self._emit("cards_ready", job_id=state.job_id)

    def _report(self, state: JobState) -> None:
        self._write_report(state)

    def _cleanup(self, state: JobState) -> None:
        job_dir = self.store.job_dir(state.job_id)
        for pattern in (
            "03.extract.json",
            "04.groups.json",
            "03.context.*.md",
            "04.outline.*.json",
            "cards.*.json",
        ):
            for path in job_dir.glob(pattern):
                if path.is_file():
                    path.unlink()
                    self.logger.debug(f"removed intermediate artifact: {path.name}")

        if not self._flag(state, "keep_srt"):
            for path in (job_dir / "sources").rglob("*.srt"):
                path.unlink(missing_ok=True)
            (job_dir / "02.transcript.srt").unlink(missing_ok=True)
        if not self._flag(state, "keep_audio"):
            for path in (job_dir / "sources").rglob("*.mp3"):
                path.unlink(missing_ok=True)
            (job_dir / "01.audio.mp3").unlink(missing_ok=True)
        else:
            self.logger.info("kept audio artifacts as requested")

        for directory in (job_dir / "sources").glob("pdf-*"):
            if directory.is_dir():
                try:
                    shutil.rmtree(directory, ignore_errors=True)
                except OSError as exc:
                    self.logger.debug(f"could not remove {directory}: {exc}")

        for key in ("audio", "srt"):
            state.artifacts.pop(key, None)
        self.store.save(state)

    # -- helpers ------------------------------------------------------------- #

    def _source_files(self, state: JobState) -> list[SourceFile]:
        paths = [Path(str(path)) for path in state.sources]
        if not paths and state.video:
            # Jobs created before multi-source input stored a single video.
            paths = [Path(state.video)]
        return list_sources(paths)

    def _text_sources(self, state: JobState) -> list[TextSource]:
        texts: list[TextSource] = []
        for source in self._source_files(state):
            path = self.store.artifact(state.job_id, f"sources/{source.id}.txt")
            if path.is_file():
                text = path.read_text(encoding="utf-8")
            elif source.kind == "document":
                result = read_source(source, page_reader=self._page_reader(state, source))
                for warning in result.warnings:
                    self._warn(warning)
                text = result.text
            else:
                text = ""
            texts.append(
                TextSource(
                    id=source.id,
                    name=source.name,
                    kind=source.kind,
                    path=source.path,
                    text=text,
                    images=tuple(self._load_images(state, source.id)),
                )
            )
        return texts

    def _page_reader(self, state: JobState, source: SourceFile) -> ScannedPdfReader | None:
        """Reader for scanned PDF pages, or None for non-PDF sources."""
        if source.path.suffix.lower() != ".pdf":
            return None
        factory = self._client if self.config.scanned_pdf == SCANNED_PDF_VISION else None
        return ScannedPdfReader(
            self.config,
            self.logger,
            cache_dir=self.store.job_dir(state.job_id) / "sources" / source.id / "pages",
            client_factory=factory,
        )

    def _harvest_images(self, state: JobState, source: SourceFile) -> list[ExtractedImage]:
        harvest = extract_pdf_images(
            source.path,
            workdir=self.store.job_dir(state.job_id) / "sources",
            vault_root=Path(self.config.notes_root).resolve(),
            attachment_root=Path(self.config.attachments_root),
            source_id=source.id,
            slug=slugify(source.path.stem),
        )
        for warning in harvest.warnings:
            self._warn(warning)
        self._write_json(
            self.store.artifact(state.job_id, f"sources/{source.id}/images.json"),
            json.dumps([image.to_dict() for image in harvest.images], indent=2),
        )
        if harvest.images:
            self.logger.info(f"{source.name}: {len(harvest.images)} figure(s) extracted")
        return harvest.images

    def _load_images(self, state: JobState, source_id: str) -> list[ExtractedImage]:
        path = self.store.artifact(state.job_id, f"sources/{source_id}/images.json")
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return [
            ExtractedImage.from_dict(item)
            for item in payload
            if isinstance(item, dict) and item.get("name")
        ]

    def _all_images(self, state: JobState) -> list[ExtractedImage]:
        images: list[ExtractedImage] = []
        for source in self._source_files(state):
            images.extend(self._load_images(state, source.id))
        return images

    def _publish_images(self, state: JobState, plan: Plan, applied: ApplyResult) -> None:
        """Copy figures referenced by applied notes; nothing else reaches the vault."""
        used: set[str] = set()
        for path in applied.applied_paths:
            note = plan.note(path)
            if note is None:
                continue
            for target in re.findall(r"!\[\[([^\]|#]+)", note.body):
                used.add(target.strip())
                used.add(target.rsplit("/", 1)[-1].strip())
        images = [
            image
            for image in self._all_images(state)
            if image.name in used or image.vault_path in used
        ]
        if not images:
            return
        published = publish_images(images, self._vault_write_root())
        self.logger.info(f"published {len(published)} figure(s) into the vault")
        self._emit("images_published", count=len(published))

    def _vault_write_root(self) -> Path:
        if self.sandbox is not None:
            return self.sandbox.directory
        return Path(self.config.notes_root).resolve()

    def _check_image_capability(self, images: list[ExtractedImage]) -> None:
        supported = self._client().image_input_supported()
        if supported is False:
            self._warn(
                RunWarning(
                    code=WarningCode.IMAGE_INPUT_UNSUPPORTED,
                    message=(
                        f"{len(images)} figure(s) were extracted, but the configured model "
                        "does not read images. They are embedded using captions and page "
                        "positions only; check their placement."
                    ),
                    detail="image_input=disabled or rejected by the provider",
                )
            )
        elif supported is None:
            self._warn(
                RunWarning(
                    code=WarningCode.IMAGE_INPUT_UNKNOWN,
                    message=(
                        f"{len(images)} figure(s) were extracted, but image support of the "
                        "configured model could not be determined. Placement may be wrong."
                    ),
                )
            )

    def _transcriber(self) -> LocalTranscriber | ApiTranscriber:
        remote = bool(self.config.asr_api_base_url)
        settings = AsrSettings(
            model=self.config.asr_model,
            language=self._job_language,
            device=self.config.device,
            batch_size=self.config.batch_size,
            backend="api" if remote else "local",
            api_model=self.config.asr_api_model,
            api_base_url=self.config.asr_api_base_url,
            api_protocol=self.config.asr_api_protocol,
            api_timeout=self.config.asr_api_timeout,
            api_key=(asr_api_key(self.config.asr_api_key) if remote else ""),
        )
        if remote:
            return ApiTranscriber(settings, self.logger)
        return LocalTranscriber(settings, self.logger)

    def _warn(self, warning: RunWarning) -> None:
        added = self.warnings.add(
            warning.code,
            warning.message,
            severity=warning.severity,
            source=warning.source,
            detail=warning.detail,
        )
        if added is None:
            return
        self.logger.warn(added.message)
        self._emit(
            "warning",
            code=added.code,
            message=added.message,
            severity=added.severity,
            source=added.source,
        )

    def _save_warnings(self, state: JobState) -> None:
        state.warnings = self.warnings.to_dicts()

    def _client(self) -> DeepSeekClient:
        client = self._cache.get("client")
        if client is None:
            client = DeepSeekClient(self.config, self.logger)
            self._cache["client"] = client
        return client

    def _require(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        loaders = {
            "extract": lambda: ExtractResult.model_validate_json(
                self.store.artifact(self._state_id(), "03.extract.json").read_text()
            ),
        }
        if name not in loaders:
            raise StageError(f"internal error: no loader for {name}")
        value = loaders[name]()
        if value is None:
            raise StageError(f"required artifact is missing: {name}")
        self._cache[name] = value
        return value

    def _state_id(self) -> str:
        state = self._cache.get("_state")
        if state is None:
            raise StageError("internal error: job state not initialized")
        return state.job_id

    def _load_apply(self, group_id: str | None = None) -> ApplyResult:
        name = f"apply.{group_id}.json" if group_id else "apply.json"
        path = self.store.artifact(self._state_id(), name)
        payload = self._read_json_artifact(path, name)
        result = ApplyResult()
        result.created = [ChangeRecord(**item) for item in payload.get("created", [])]
        result.merged = [ChangeRecord(**item) for item in payload.get("merged", [])]
        result.moc = [ChangeRecord(**item) for item in payload.get("moc", [])]
        result.applied_paths = payload.get("applied_paths", [])
        result.skipped_merges = payload.get("skipped_merges", [])
        result.conflicts = payload.get("conflicts", [])
        result.refused = payload.get("refused", [])
        result.todos = [TodoItem.model_validate(item) for item in payload.get("todos", [])]
        return result

    def _load_merged_plan(self) -> Plan:
        path = self.store.artifact(self._state_id(), "05.plan.json")
        return Plan.model_validate_json(path.read_text(encoding="utf-8"))

    def _load_grouping_todos(self, state: JobState) -> list[TodoItem]:
        path = self.store.artifact(state.job_id, "04.groups.json")
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [TodoItem.model_validate(item) for item in payload.get("todos", [])]

    def _load_cards(self) -> list[CardWriteResult]:
        path = self.store.artifact(self._state_id(), "cards.json")
        if not path.is_file():
            return []
        payload = self._read_json_artifact(path, "cards.json")
        if not isinstance(payload, list):
            raise StageError("cards.json must contain a JSON array")
        return [CardWriteResult(**item) for item in payload]

    def _bias_prompt(self, state: JobState) -> str:
        moc_text = self.vault.read(state.moc) if state.moc else ""
        glossary = load_glossary(self.vault, state.course)
        return build_bias_prompt(state.course, moc_text, glossary)

    def _save_plan(self, state: JobState, plan: Plan, group: GroupState) -> None:
        client = self._cache.get("client")
        if client is not None:
            plan.stats.update(
                {
                    "llm_calls": client.usage.calls,
                    "prompt_tokens": client.usage.prompt_tokens,
                    "completion_tokens": client.usage.completion_tokens,
                }
            )
        self._cache.setdefault("plans", {})[group.id] = plan
        self._write_json(
            self.store.artifact(state.job_id, f"05.plan.{group.id}.json"),
            plan.model_dump_json(indent=2),
        )
        self._write_merged_plan(state)

    def _write_merged_plan(self, state: JobState) -> None:
        plans = self._cache.get("plans") or {}
        notes: list[Any] = []
        todos: list[TodoItem] = []
        for group in state.groups:
            plan = plans.get(group.id)
            if plan is None:
                continue
            notes.extend(plan.notes)
            todos.extend(plan.todos)
        try:
            asides = list(self._require("extract").asides)
        except (StageError, OSError, ValueError):
            asides = []
        merged = Plan(
            course=state.course,
            moc_path=state.moc,
            notes=notes,
            asides=asides,
            todos=todos,
        )
        self._cache["plan_all"] = merged
        self._write_json(
            self.store.artifact(state.job_id, "05.plan.json"),
            merged.model_dump_json(indent=2),
        )

    def _write_report(self, state: JobState, failure: Exception | None = None) -> None:
        try:
            markdown = self._report_markdown(state, failure)
        except Exception as exc:  # never fail a job because reporting failed
            self.logger.error(f"could not build report: {exc}")
            return
        self._write_text(self.store.artifact(state.job_id, "report.md"), markdown)
        state.artifacts["report"] = "report.md"
        self.store.save(state)
        self.ui.print("")
        self.ui.print(markdown)

    def _report_markdown(self, state: JobState, failure: Exception | None) -> str:
        plan: Plan | None = self._cache.get("plan_all")
        if plan is None:
            try:
                plan = self._load_merged_plan()
            except (StageError, OSError, ValueError):
                plan = None
        applied: ApplyResult | None = self._cache.get("apply_all")
        if applied is None:
            try:
                applied = self._load_apply()
            except (StageError, OSError, ValueError):
                applied = None
        texts = self._text_sources(state)
        cached_cards = self._cache.get("cards")
        if isinstance(cached_cards, dict):
            cards = [item for items in cached_cards.values() for item in items]
        elif isinstance(cached_cards, list):
            cards = cached_cards
        else:
            try:
                cards = self._load_cards()
            except (OSError, ValueError):
                cards = []
        client: DeepSeekClient | None = self._cache.get("client")

        created = [record.path for record in applied.created] if applied else []
        merged = [record.path for record in applied.merged] if applied else []
        moc_updates = [record.path for record in applied.moc] if applied else []
        todos = list(plan.todos) if plan else []
        grouping_todos = self._cache.get("grouping_todos") or self._load_grouping_todos(state)
        todos.extend(grouping_todos)
        if applied:
            todos.extend(applied.todos)

        status = (
            "failed" if failure else ("planned" if self._flag(state, "dry_run") else "complete")
        )
        lines = [
            f"# Lecture import: {state.course}",
            "",
            f"- Job: `{state.job_id}`",
            f"- Status: {status}",
            "- Sources: " + (", ".join(f"`{source}`" for source in state.sources) or "-"),
            f"- MOC: `{state.moc or '-'}`",
        ]
        duration = state.options.get("audio_duration")
        if duration:
            lines.append(f"- Duration: {float(str(duration)):.0f}s")
        readable = [text for text in texts if text.text.strip()]
        if readable:
            words = sum(len(text.text.split()) for text in readable)
            lines.append(f"- Readable sources: {len(readable)} ({words} words)")
        if self._flag(state, "dry_run"):
            lines.append(
                "- Dry run: the change set was simulated in a temporary copy of the vault "
                "and discarded."
            )
        if failure:
            lines.append(f"- Failure: `{failure}`")
        lines.append("")

        if state.groups:
            by_id = {source.id: source.name for source in self._source_files(state)}
            lines.append("## Topic groups")
            for group in state.groups:
                names = [by_id.get(source_id, source_id) for source_id in group.source_ids]
                lines.append(
                    f"- {group.id}: {group.topic or '-'} ({', '.join(names)})"
                    + (f" - {group.rationale}" if group.rationale else "")
                )
            lines.append("")

        lines.append("## Notes")
        if created:
            lines.append("Created:")
            lines.extend(f"- `{path}`" for path in created)
        if merged:
            lines.append("Merged:")
            lines.extend(f"- `{path}`" for path in merged)
        if applied and applied.conflicts:
            lines.append("Manual merges (conflict files, note untouched):")
            lines.extend(f"- `{path}`" for path in applied.conflicts)
        if applied and applied.skipped_merges:
            lines.append("Skipped merges (needs review):")
            lines.extend(f"- `{path}`" for path in applied.skipped_merges)
        if applied and applied.refused:
            lines.append("Refused:")
            lines.extend(f"- `{path}`" for path in applied.refused)
        if plan and not applied:
            lines.append("Planned:")
            lines.extend(
                f"- `{note.path}` ({note.action}, {len(note.flashcards)} cards)"
                for note in plan.notes
            )
        if not (created or merged or plan):
            lines.append("- none")
        lines.append("")

        if moc_updates:
            lines.append("## MOC updates")
            lines.extend(f"- `{path}`" for path in moc_updates)
            lines.append("")

        lines.append("## Flashcards")
        if cards:
            for result in cards:
                lines.append(f"- {result.summary}")
        else:
            lines.append("- none")
        lines.append("")

        if client is not None:
            usage = client.usage
            lines.append("## LLM usage")
            lines.append(
                f"- {usage.calls} calls, {usage.prompt_tokens} prompt tokens, "
                f"{usage.completion_tokens} completion tokens"
            )
            lines.append("")
        elif plan and plan.stats:
            lines.append("## LLM usage")
            lines.append(
                f"- {plan.stats.get('llm_calls', 0)} calls, "
                f"{plan.stats.get('prompt_tokens', 0)} prompt tokens, "
                f"{plan.stats.get('completion_tokens', 0)} completion tokens"
            )
            lines.append("")

        lines.append("## Warnings")
        if self.warnings:
            for warning in self.warnings:
                where = f" (`{warning.source}`)" if warning.source else ""
                lines.append(f"- [{warning.severity}] {warning.code}: {warning.message}{where}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## TODOs")
        if todos:
            for todo in todos:
                suffix = f" (`{todo.note_path}`)" if todo.note_path else ""
                lines.append(f"- [{todo.kind}] {todo.text}{suffix}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Artifacts")
        directory = self.store.job_dir(state.job_id)
        for name in sorted(path.name for path in directory.iterdir() if path.is_file()):
            if name in {"state.json", "job.json", "lock"}:
                continue
            lines.append(f"- `{name}`")
        lines.append("")

        if failure:
            lines.append("## Next step")
            lines.append(f"- Resume after fixing the error: `scriptorium resume {state.job_id}`")
            lines.append("- Or resume the latest unfinished job: `scriptorium resume`")
        elif self._flag(state, "dry_run"):
            lines.append("## Next step")
            lines.append(f"- Apply the plan: `scriptorium resume {state.job_id} --yes`")
        lines.append("")
        return "\n".join(lines)

    def _flag(self, state: JobState, name: str) -> bool:
        return bool(state.options.get(name, False))

    def _option(self, state: JobState, name: str) -> Any:
        return state.options.get(name)

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event({"type": kind, **fields})
            except Exception:
                pass

    def _record_history(
        self,
        state: JobState,
        *,
        status: str = "completed",
        error: str = "",
    ) -> None:
        """Save the run's generated text for the TUI history."""
        try:
            record_run(
                self.config.history_dir,
                self.store.job_dir(state.job_id),
                job_id=state.job_id,
                course=state.course,
                sources=state.sources,
                created_at=state.created_at,
                finished_at=state.updated_at,
                status=status,
                error=error,
            )
        except (OSError, ValueError) as exc:
            # A history snapshot is a convenience, never a reason to fail a run.
            self.logger.warn(f"could not save run history: {exc}")

    def _write_job_snapshot(self, state: JobState) -> None:
        snapshot = {
            "job_id": state.job_id,
            "created_at": state.created_at,
            "course": state.course,
            "sources": list(state.sources),
            "video": state.video,
            "notes_root": str(self.config.notes_root),
            "llm_model": self.config.llm_model,
            "llm_base_url": self.config.llm_base_url,
            "asr_model": self.config.asr_model,
            "asr_api_model": self.config.asr_api_model,
            "asr_api_base_url": self.config.asr_api_base_url,
            "asr_api_protocol": self.config.asr_api_protocol,
            "language": self.config.language,
            "device": self.config.device,
        }
        self._write_json(
            self.store.artifact(state.job_id, "job.json"),
            json.dumps(snapshot, indent=2, ensure_ascii=False),
        )

    def _read_json_artifact(self, path: Path, label: str) -> Any:
        """Read a JSON artifact, turning I/O and parse errors into stage errors."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StageError(f"cannot read {label} at {path}: {exc}") from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise StageError(f"{label} at {path} is not valid JSON: {exc}") from exc

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)

    def _write_json(self, path: Path, payload: str) -> None:
        self._write_text(path, payload.rstrip("\n") + "\n")

    # Keep the job state in the cache so artifact loaders know the job id.
    def _install_state(self, state: JobState) -> None:
        self._cache["_state"] = state


def stage_label(stage: Stage) -> str:
    from .state import STAGE_LABELS

    return STAGE_LABELS[stage]


def _context_markdown(context: ContextBundle) -> str:
    lines = [
        f"# Context: {context.course}",
        "",
        f"- MOC: `{context.moc_path}`",
        f"- Tags: {', '.join(context.tags) or '-'}",
        f"- Keywords: {', '.join(context.keywords[:40])}",
        "",
        "## Inventory",
        "",
        context.inventory_block() or "-",
        "",
        "## Related notes",
        "",
        context.related_block() or "-",
        "",
    ]
    return "\n".join(lines)


def _apply_dict(result: ApplyResult) -> dict[str, Any]:
    return {
        "created": [asdict(record) for record in result.created],
        "merged": [asdict(record) for record in result.merged],
        "moc": [asdict(record) for record in result.moc],
        "applied_paths": result.applied_paths,
        "skipped_merges": result.skipped_merges,
        "conflicts": result.conflicts,
        "refused": result.refused,
        "todos": [todo.model_dump() for todo in result.todos],
    }
