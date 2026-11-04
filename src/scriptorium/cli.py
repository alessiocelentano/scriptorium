"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config, is_first_run, llm_api_key, load_config
from .errors import ConfigurationError, ScriptoriumError, StageError
from .history import state_status, status_text
from .log import Logger
from .pipeline import Pipeline, RunOptions
from .state import JobState
from .ui import ConsoleUI

VERSION = "0.2.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scriptorium",
        description="Turn video lectures into linked Obsidian notes and draft flashcards.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--verbose", action="store_true", help="Verbose console logging")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Process one or more sources from the command line")
    run.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="Lecture videos and/or documents (slides, notes, PDFs)",
    )
    run.add_argument(
        "--course", help="Course name used to find the MOC (default: first source name)"
    )
    _add_pipeline_flags(run)

    resume = subparsers.add_parser("resume", help="Continue a job stopped by an error or a dry run")
    resume.add_argument(
        "job_id",
        nargs="?",
        help="Job id (default: the most recent unfinished job)",
    )
    _add_pipeline_flags(resume)

    return parser


def _add_pipeline_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--moc", help="Explicit MOC path inside notes_root")
    parser.add_argument("--language", help="Spoken language, or 'detect' (default: en)")
    parser.add_argument("--asr-model", help="Local speech-to-text model name or local path")
    parser.add_argument("--asr-api-model", help="Model id for the transcription API")
    parser.add_argument("--asr-api-base-url", help="Base URL of the transcription API")
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], help="Torch device for ASR"
    )
    parser.add_argument("--model", help="LLM model id")
    parser.add_argument("--llm-base-url", help="Base URL of an OpenAI-compatible LLM endpoint")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Stop after the plan")
    parser.add_argument(
        "--accept-merges",
        action="store_true",
        default=None,
        help="Apply merges into existing notes without asking",
    )
    parser.add_argument(
        "--no-moc", action="store_true", default=None, help="Do not update the MOC index"
    )
    parser.add_argument(
        "--keep-audio", action="store_true", default=None, help="Keep the MP3 after the run"
    )
    parser.add_argument(
        "--keep-srt", action="store_true", default=None, help="Keep the SRT after the run"
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Accept default answers")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; create notes, skip merges unless --accept-merges",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command is None:
            return run_tui(args)
        if args.command == "resume":
            return run_resume(args)
        return run_pipeline(args)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except ScriptoriumError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def run_tui(args: argparse.Namespace) -> int:
    first_run = is_first_run()
    # The first run may start from a copied config.toml.example whose notes_root
    # does not exist yet; the wizard is where the user creates or fixes it.
    config = load_config(require_notes_root=not first_run)
    from .tui import run_tui as start_tui

    return start_tui(config, verbose=args.verbose, show_setup=first_run)


def run_pipeline(args: argparse.Namespace) -> int:
    config, ui, logger = _pipeline_context(args)
    llm_api_key(config)
    pipeline = Pipeline(config, ui, logger)
    state = pipeline.run(args.sources, _run_options(args))
    _report_finished(config, logger, state)
    return 0


def run_resume(args: argparse.Namespace) -> int:
    config, ui, logger = _pipeline_context(args)
    pipeline = Pipeline(config, ui, logger)
    if args.job_id:
        state = pipeline.store.resolve(args.job_id)
    else:
        candidates = pipeline.store.resumable(limit=1)
        if not candidates:
            raise StageError(
                "no unfinished job to resume. Start one with `scriptorium run <source>`"
            )
        state = candidates[0]
    status, error = state_status(state)
    if status == "completed":
        print(f"job {state.job_id} is already complete. Nothing to resume")
        return 0
    llm_api_key(config)
    print(f"resuming {state.job_id} · {state.course} · {status_text(status, error)}")
    resumed = pipeline.resume(state.job_id, _run_options(args))
    _report_finished(config, logger, resumed)
    return 0


def _pipeline_context(args: argparse.Namespace) -> tuple[Config, ConsoleUI, Logger]:
    if is_first_run():
        print(
            "first run: start `scriptorium` to run the setup wizard, or edit config.toml",
            file=sys.stderr,
        )
    config = load_config(
        language=args.language,
        asr_model=args.asr_model,
        asr_api_model=args.asr_api_model,
        asr_api_base_url=args.asr_api_base_url,
        device=args.device,
        llm_model=args.model,
        llm_base_url=args.llm_base_url,
    )
    logger = Logger(None, verbose=args.verbose)
    ui = ConsoleUI(
        yes=args.yes,
        accept_merges=bool(args.accept_merges),
        interactive=False if args.non_interactive else None,
    )
    return config, ui, logger


def _run_options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        course=getattr(args, "course", "") or "",
        moc=args.moc,
        dry_run=args.dry_run,
        accept_merges=args.accept_merges,
        no_moc=args.no_moc,
        keep_audio=args.keep_audio,
        keep_srt=args.keep_srt,
    )


def _report_finished(config: Config, logger: Logger, state: JobState) -> None:
    report = state.artifacts.get("report")
    if report:
        path = config.artifacts_dir / "jobs" / state.job_id / report
        logger.success(f"job {state.job_id} finished; report: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
