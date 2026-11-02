"""Configuration resolution: defaults, TOML file, environment variables."""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import cast

from .errors import ConfigurationError
from .tui_layout import DEFAULT_LAYOUT, LayoutNode, parse_layout

# The language model is any OpenAI-compatible chat-completions endpoint: the
# base URL and the model id are the only two values that select it.  DeepSeek is
# the default, not a special case.
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_TIMEOUT = 240.0

# How the model is asked for JSON.  "object" sends
# ``response_format: {"type": "json_object"}``; "none" omits the field for
# endpoints that reject it (the parser tolerates prose and fences anyway).
LLM_JSON_MODES = ("object", "none")
DEFAULT_LLM_JSON_MODE = "object"

DEFAULT_ASR_MODEL = "openai/whisper-large-v3"

# How speech recognition runs.  There is no provider list and no backend switch:
# an empty API base URL runs the local transformers pipeline, any base URL sends
# the audio to that remote endpoint.
DEFAULT_ASR_API_TIMEOUT = 600.0

# Request style of the transcription endpoint.  OpenAI, Groq and most
# compatible services accept multipart /audio/transcriptions; the Hugging Face
# router and Inference Endpoints take a raw audio body at models/<model>.
ASR_PROTOCOL_OPENAI = "openai"
ASR_PROTOCOL_HUGGINGFACE = "huggingface"
ASR_PROTOCOLS = (ASR_PROTOCOL_OPENAI, ASR_PROTOCOL_HUGGINGFACE)

# Environment variables tried for a transcription key after
# SCRIPTORIUM_ASR_API_KEY and the `asr_api_key` setting.  A convenience
# fallback, not a provider registry: the portable setting is `asr_api_key`.
_ASR_KEY_ENVS = (
    "ASR_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)

# Settings are machine-local: the source checkout keeps them in a gitignored
# ``config.toml`` (see ``config.toml.example``), installed copies fall back to
# ``~/.config/scriptorium/config.toml``.  Never hard-code personal paths in
# tracked files.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CONFIG_NAME = "config.toml"

# Attachments are resolved against ``notes_root`` when the value is relative;
# an absolute directory is allowed too.
DEFAULT_ATTACHMENT_DIR = "Attachments"

# The built-in flashcard skill ships with the package; a user can point
# ``flashcard_skill`` at their own copy.
DEFAULT_FLASHCARD_SKILL = (
    Path(__file__).resolve().parent / "skills" / "manage-flashcards" / "manage_flashcards.py"
)

_SIZE_KEYS_REMOVED = ()

IMAGE_INPUT_AUTO = "auto"
IMAGE_INPUT_ENABLED = "enabled"
IMAGE_INPUT_DISABLED = "disabled"
IMAGE_INPUT_MODES = (IMAGE_INPUT_AUTO, IMAGE_INPUT_ENABLED, IMAGE_INPUT_DISABLED)

# How a PDF page without a usable text layer is read: one vision call per page
# or local Tesseract.
SCANNED_PDF_VISION = "vision"
SCANNED_PDF_OCR = "ocr"
SCANNED_PDF_MODES = (SCANNED_PDF_VISION, SCANNED_PDF_OCR)
DEFAULT_SCANNED_PDF = SCANNED_PDF_VISION
DEFAULT_MAX_SCANNED_PAGES = 60

# How a merge into an existing note is decided.  "ask" shows the diff and lets
# the user apply, skip or keep a conflict file; "apply" accepts every reviewed
# merge; "manual" always writes a conflict file and never touches the vault.
MERGE_POLICIES = ("ask", "apply", "manual")
DEFAULT_MERGE_POLICY = "ask"

# How the target note of a planned note is chosen.  "ask" shows the candidates,
# "auto" picks one only above a score and with a margin, "create" never merges.
TARGET_POLICIES = ("ask", "auto", "create")
DEFAULT_TARGET_POLICY = "ask"

# Which movement model the TUI uses.  "vim" keeps the modal input line and
# hjkl navigation; "standard" is a modeless editor driven by the arrow keys.
MOVEMENT_VIM = "vim"
MOVEMENT_STANDARD = "standard"
MOVEMENT_STYLES = (MOVEMENT_VIM, MOVEMENT_STANDARD)
DEFAULT_MOVEMENT = MOVEMENT_VIM

# Spoken language sentinels.  "detect" is the empty language; "ask" prompts once
# per run, after the MOC has been chosen.
LANGUAGE_DETECT = "detect"
LANGUAGE_ASK = "ask"


def _default_notes_root() -> Path:
    candidate = Path.home() / "Obsidian" / "Second Brain"
    return candidate if candidate.is_dir() else Path.cwd()


def _default_artifacts_dir() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "scriptorium"


def _default_search_path() -> Path:
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.home()


def _project_config_path() -> Path | None:
    """Return the repo-local settings file when running from a source checkout."""
    if (PROJECT_ROOT / "pyproject.toml").is_file():
        return PROJECT_ROOT / PROJECT_CONFIG_NAME
    return None


def _user_config_path() -> Path:
    return Path.home() / ".config" / "scriptorium" / "config.toml"


# Marker recorded by the first-run TUI wizard once it has been completed.
SETUP_MARKER_NAME = "setup_complete"


def setup_marker_path() -> Path:
    """Where the "setup completed" marker lives.

    The marker sits in the *default* artifacts directory, not in
    ``Config.artifacts_dir``: it must be computable before a configuration file
    is known to be valid, because a first run may start from a copied
    ``config.toml`` whose ``notes_root`` does not exist yet.
    """
    return _default_artifacts_dir() / SETUP_MARKER_NAME


def is_first_run() -> bool:
    """True until the setup wizard has been completed."""
    return not setup_marker_path().exists()


def mark_setup_complete() -> Path:
    """Record that the setup wizard ran; idempotent."""
    path = setup_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _config_path() -> Path:
    """Return the settings file to read.

    A repo-local file is used when it exists so a checkout can be configured
    without touching the user home; otherwise ``~/.config`` is used.
    """
    project = _project_config_path()
    if project is not None and project.is_file():
        return project
    return _user_config_path()


def _config_write_path() -> Path:
    """Return the settings file to write: repo-local in a checkout, else XDG."""
    project = _project_config_path()
    return project if project is not None else _user_config_path()


@dataclass(frozen=True)
class Config:
    notes_root: Path
    artifacts_dir: Path
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_timeout: float = DEFAULT_LLM_TIMEOUT
    llm_api_key: str = ""
    llm_json_mode: str = DEFAULT_LLM_JSON_MODE
    asr_model: str = DEFAULT_ASR_MODEL
    asr_api_model: str = ""
    asr_api_base_url: str = ""
    asr_api_protocol: str = ""
    asr_api_timeout: float = DEFAULT_ASR_API_TIMEOUT
    asr_api_key: str = ""
    language: str = "en"
    device: str = "auto"
    batch_size: int = 8
    flashcard_skill: Path = DEFAULT_FLASHCARD_SKILL
    flashcard_max_per_note: int = 60
    # Maximum characters kept from each related-note excerpt in the prompt
    # context. This is unrelated to the Extras pane.
    extra_context_chars: int = 2600
    search_paths: tuple[Path, ...] = ()
    # Paths the user actually configured.  ``search_paths`` adds the default
    # source root and the notes root implicitly; the wizard and the settings
    # modal show and write only this explicit list.
    user_search_paths: tuple[Path, ...] | None = None
    attachment_dir: Path = Path(DEFAULT_ATTACHMENT_DIR)
    image_input: str = IMAGE_INPUT_AUTO
    scanned_pdf: str = DEFAULT_SCANNED_PDF
    ocr_languages: str = ""
    max_scanned_pages: int = DEFAULT_MAX_SCANNED_PAGES
    merge_policy: str = DEFAULT_MERGE_POLICY
    target_policy: str = DEFAULT_TARGET_POLICY
    movement: str = DEFAULT_MOVEMENT
    target_auto_above: float = 0.85
    target_min_gap: float = 0.15
    disabled_warnings: tuple[str, ...] = ()
    tui_layout: LayoutNode = DEFAULT_LAYOUT
    tui_pane_keys: dict[str, str] = field(default_factory=dict)
    tui_panes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.llm_json_mode not in LLM_JSON_MODES:
            raise ConfigurationError(
                "llm_json_mode must be one of "
                f"{', '.join(LLM_JSON_MODES)}, got {self.llm_json_mode!r}"
            )
        # "detect" (or "auto") leaves the language empty, which both ASR
        # backends treat as "detect it from the audio".
        if str(self.language).strip().lower() in {"detect", "auto"}:
            object.__setattr__(self, "language", "")
        object.__setattr__(
            self, "llm_base_url", (self.llm_base_url or DEFAULT_LLM_BASE_URL).rstrip("/")
        )
        # Speech recognition: an empty base URL is the only "local" switch, and
        # a base URL is the only "remote" switch.
        if self.asr_api_base_url:
            object.__setattr__(self, "asr_api_base_url", self.asr_api_base_url.rstrip("/"))
            if not self.asr_api_model:
                object.__setattr__(self, "asr_api_model", self.asr_model)
        if not self.asr_api_protocol:
            object.__setattr__(self, "asr_api_protocol", ASR_PROTOCOL_OPENAI)
        if self.asr_api_protocol not in ASR_PROTOCOLS:
            raise ConfigurationError(
                "asr_api_protocol must be one of "
                f"{', '.join(ASR_PROTOCOLS)}, got {self.asr_api_protocol!r}"
            )
        if self.asr_api_timeout <= 0:
            raise ConfigurationError(
                f"asr_api_timeout must be positive, got {self.asr_api_timeout}"
            )
        if self.image_input not in IMAGE_INPUT_MODES:
            raise ConfigurationError(
                "image_input must be one of "
                f"{', '.join(IMAGE_INPUT_MODES)}, got {self.image_input!r}"
            )
        if self.scanned_pdf not in SCANNED_PDF_MODES:
            raise ConfigurationError(
                "scanned_pdf must be one of "
                f"{', '.join(SCANNED_PDF_MODES)}, got {self.scanned_pdf!r}"
            )
        if self.max_scanned_pages < 0:
            raise ConfigurationError(
                f"max_scanned_pages must be zero or more, got {self.max_scanned_pages}"
            )
        if self.merge_policy not in MERGE_POLICIES:
            raise ConfigurationError(
                "merge_policy must be one of "
                f"{', '.join(MERGE_POLICIES)}, got {self.merge_policy!r}"
            )
        if self.target_policy not in TARGET_POLICIES:
            raise ConfigurationError(
                "target_policy must be one of "
                f"{', '.join(TARGET_POLICIES)}, got {self.target_policy!r}"
            )
        if self.movement not in MOVEMENT_STYLES:
            raise ConfigurationError(
                f"movement must be one of {', '.join(MOVEMENT_STYLES)}, got {self.movement!r}"
            )
        for name, value in (
            ("target_auto_above", self.target_auto_above),
            ("target_min_gap", self.target_min_gap),
        ):
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be between 0 and 1, got {value}")
        # Tesseract expects "eng+ita"; accept commas, spaces and plus signs.
        languages: list[str] = []
        for item in re.split(r"[+,;\s]+", self.ocr_languages):
            value = item.strip()
            if value and value not in languages:
                languages.append(value)
        object.__setattr__(self, "ocr_languages", "+".join(languages))
        # Normalize here as well as in load_config so a Config built by a test
        # or an embedder cannot hold an unvalidated layout.  Pane names come
        # from the registry, so a layout may reference any configured box.
        # Imported locally: the registry pulls in the pane modules, and only the
        # TUI needs them.
        from .pane_registry import placeable_names

        panes: dict[str, str] = {}
        for raw_name, raw_target in (self.tui_panes or {}).items():
            pane = str(raw_name).strip().lower()
            if not pane:
                raise ConfigurationError("tui_panes cannot contain an empty pane name")
            if not isinstance(raw_target, str) or not raw_target.strip():
                raise ConfigurationError(
                    f"tui_panes.{pane} must be an import target such as 'pkg.module:MyPane'"
                )
            panes[pane] = raw_target.strip()
        object.__setattr__(self, "tui_panes", panes)
        keys: dict[str, str] = {}
        for raw_name, raw_key in (self.tui_pane_keys or {}).items():
            pane = str(raw_name).strip().lower()
            keys[pane] = "" if raw_key is None else str(raw_key).strip()
        object.__setattr__(self, "tui_pane_keys", keys)
        object.__setattr__(
            self, "tui_layout", parse_layout(self.tui_layout, placeable_names(panes))
        )
        configured = self.user_search_paths
        if configured is None:
            configured = self.search_paths
        roots: list[Path] = []
        for candidate in configured or ():
            path = Path(candidate).expanduser()
            if path not in roots:
                roots.append(path)
        object.__setattr__(self, "user_search_paths", tuple(roots))
        # The effective roots add a default source location only when the user
        # configured none, then always include the notes root.
        if not roots:
            roots.append(_default_search_path())
        notes_root = Path(self.notes_root).expanduser()
        if notes_root not in roots:
            roots.append(notes_root)
        object.__setattr__(self, "search_paths", tuple(roots))
        object.__setattr__(self, "attachment_dir", Path(self.attachment_dir).expanduser())
        object.__setattr__(self, "flashcard_skill", Path(self.flashcard_skill).expanduser())
        codes: list[str] = []
        for code in self.disabled_warnings:
            value = str(code).strip()
            if value and value not in codes:
                codes.append(value)
        object.__setattr__(self, "disabled_warnings", tuple(codes))

    @property
    def jobs_dir(self) -> Path:
        return self.artifacts_dir / "jobs"

    @property
    def history_dir(self) -> Path:
        """Completed-run snapshots shown by the TUI history window."""
        return self.artifacts_dir / "history"

    @property
    def attachments_root(self) -> Path:
        """Absolute attachment directory; relative values resolve inside notes_root."""
        directory = Path(self.attachment_dir).expanduser()
        root = Path(self.notes_root).expanduser().resolve()
        if directory.is_absolute():
            return directory.resolve()
        return (root / directory).resolve()

    def with_overrides(self, **overrides: object) -> Config:
        clean = {key: value for key, value in overrides.items() if value is not None}
        # A caller that replaces ``search_paths`` means the explicit list, not
        # the effective one, so keep the two fields in sync.
        if "search_paths" in clean and "user_search_paths" not in clean:
            clean["user_search_paths"] = clean["search_paths"]
        return replace(self, **clean)


def _read_toml(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"cannot read config file {path}: {exc}") from exc
    return data


def load_config(*, require_notes_root: bool = True, **overrides: object) -> Config:
    """Resolve configuration in order: defaults < TOML < environment < arguments.

    ``require_notes_root=False`` skips the "notes_root must exist" check.  The
    first-run wizard needs it: a freshly copied ``config.toml.example`` may
    point at a vault that does not exist yet, and the wizard is where the user
    creates or corrects it.
    """
    explicit_notes_root = overrides.pop("notes_root", None)
    explicit_artifacts_dir = overrides.pop("artifacts_dir", None)

    data = _read_toml(_config_path())

    notes_root = (
        explicit_notes_root or os.environ.get("SCRIPTORIUM_NOTES_ROOT") or data.get("notes_root")
    )
    artifacts_dir = (
        explicit_artifacts_dir
        or os.environ.get("SCRIPTORIUM_ARTIFACTS_DIR")
        or data.get("artifacts_dir")
    )

    def _path(value: object) -> Path | None:
        if value in (None, ""):
            return None
        return Path(str(value)).expanduser()

    pane_targets, pane_keys = _pane_tables(data.get("tui_panes"))

    config = Config(
        notes_root=_path(notes_root) or _default_notes_root(),
        artifacts_dir=_path(artifacts_dir) or _default_artifacts_dir(),
        llm_model=str(
            os.environ.get("SCRIPTORIUM_LLM_MODEL") or data.get("llm_model") or DEFAULT_LLM_MODEL
        ),
        llm_base_url=str(
            os.environ.get("SCRIPTORIUM_LLM_BASE_URL")
            or data.get("llm_base_url")
            or DEFAULT_LLM_BASE_URL
        ).rstrip("/"),
        llm_timeout=float(
            str(
                os.environ.get("SCRIPTORIUM_LLM_TIMEOUT")
                or data.get("llm_timeout")
                or DEFAULT_LLM_TIMEOUT
            )
        ),
        llm_api_key=str(
            os.environ.get("SCRIPTORIUM_LLM_API_KEY") or data.get("llm_api_key") or ""
        ).strip(),
        llm_json_mode=str(
            os.environ.get("SCRIPTORIUM_LLM_JSON_MODE")
            or data.get("llm_json_mode")
            or DEFAULT_LLM_JSON_MODE
        ).strip(),
        asr_model=str(
            os.environ.get("SCRIPTORIUM_ASR_MODEL") or data.get("asr_model") or DEFAULT_ASR_MODEL
        ),
        asr_api_model=str(
            os.environ.get("SCRIPTORIUM_ASR_API_MODEL") or data.get("asr_api_model") or ""
        ).strip(),
        asr_api_base_url=str(
            os.environ.get("SCRIPTORIUM_ASR_API_BASE_URL") or data.get("asr_api_base_url") or ""
        ).strip(),
        asr_api_protocol=str(
            os.environ.get("SCRIPTORIUM_ASR_API_PROTOCOL") or data.get("asr_api_protocol") or ""
        ).strip(),
        asr_api_timeout=float(
            str(
                os.environ.get("SCRIPTORIUM_ASR_API_TIMEOUT")
                or data.get("asr_api_timeout")
                or DEFAULT_ASR_API_TIMEOUT
            )
        ),
        asr_api_key=str(
            os.environ.get("SCRIPTORIUM_ASR_API_KEY") or data.get("asr_api_key") or ""
        ).strip(),
        language=str(os.environ.get("SCRIPTORIUM_LANGUAGE") or data.get("language") or "en"),
        device=str(os.environ.get("SCRIPTORIUM_DEVICE") or data.get("device") or "auto"),
        batch_size=int(str(os.environ.get("SCRIPTORIUM_ASR_BATCH") or data.get("batch_size") or 8)),
        flashcard_skill=_path(
            os.environ.get("SCRIPTORIUM_FLASHCARD_SKILL")
            or data.get("flashcard_skill")
            or DEFAULT_FLASHCARD_SKILL
        )
        or DEFAULT_FLASHCARD_SKILL,
        flashcard_max_per_note=int(str(data.get("flashcard_max_per_note") or 60)),
        extra_context_chars=int(str(data.get("extra_context_chars") or 2600)),
        search_paths=tuple(_search_paths(data)),
        attachment_dir=_path(
            os.environ.get("SCRIPTORIUM_ATTACHMENT_DIR") or data.get("attachment_dir")
        )
        or Path(DEFAULT_ATTACHMENT_DIR),
        image_input=str(
            os.environ.get("SCRIPTORIUM_IMAGE_INPUT") or data.get("image_input") or IMAGE_INPUT_AUTO
        ).strip(),
        scanned_pdf=str(
            os.environ.get("SCRIPTORIUM_SCANNED_PDF")
            or data.get("scanned_pdf")
            or DEFAULT_SCANNED_PDF
        ).strip(),
        ocr_languages=str(
            os.environ.get("SCRIPTORIUM_OCR_LANGUAGES") or data.get("ocr_languages") or ""
        ).strip(),
        max_scanned_pages=int(
            str(
                os.environ.get("SCRIPTORIUM_MAX_SCANNED_PAGES")
                or data.get("max_scanned_pages")
                or DEFAULT_MAX_SCANNED_PAGES
            )
        ),
        merge_policy=str(
            os.environ.get("SCRIPTORIUM_MERGE_POLICY")
            or data.get("merge_policy")
            or DEFAULT_MERGE_POLICY
        ).strip(),
        target_policy=str(
            os.environ.get("SCRIPTORIUM_TARGET_POLICY")
            or data.get("target_policy")
            or DEFAULT_TARGET_POLICY
        ).strip(),
        movement=str(
            os.environ.get("SCRIPTORIUM_MOVEMENT") or data.get("movement") or DEFAULT_MOVEMENT
        ).strip(),
        target_auto_above=float(
            str(
                os.environ.get("SCRIPTORIUM_TARGET_AUTO_ABOVE")
                or data.get("target_auto_above")
                or 0.85
            )
        ),
        target_min_gap=float(
            str(os.environ.get("SCRIPTORIUM_TARGET_MIN_GAP") or data.get("target_min_gap") or 0.15)
        ),
        disabled_warnings=tuple(
            _string_list(os.environ.get("SCRIPTORIUM_DISABLED_WARNINGS"))
            + _string_list(data.get("disabled_warnings"))
        ),
        # The layout is validated in ``Config.__post_init__``, where the pane
        # registry is available: a layout may name a box that the
        # ``[tui_panes]`` table registers, so parsing it here would reject every
        # custom pane.
        tui_layout=cast(LayoutNode, data.get("tui_layout")),
        tui_pane_keys=pane_keys,
        tui_panes=pane_targets,
    )

    config = config.with_overrides(**overrides)
    if require_notes_root and not config.notes_root.is_dir():
        raise ConfigurationError(f"notes_root directory does not exist: {config.notes_root}")
    return config


def llm_api_key(config: Config | None = None) -> str:
    """Return the API key for the configured chat-completions endpoint.

    Order: ``SCRIPTORIUM_LLM_API_KEY``, then ``llm_api_key`` from the settings
    file, then the generic ``LLM_API_KEY`` and ``OPENAI_API_KEY``.  A custom
    base URL without a key is accepted (local servers such as Ollama need none),
    so the error only fires for the default endpoint.
    """
    explicit = os.environ.get("SCRIPTORIUM_LLM_API_KEY", "").strip()
    if explicit:
        return explicit
    if config is not None and config.llm_api_key.strip():
        return config.llm_api_key.strip()
    default_endpoint = config is None or config.llm_base_url == DEFAULT_LLM_BASE_URL
    names = (
        ("LLM_API_KEY",)
        if default_endpoint
        else (
            "LLM_API_KEY",
            "OPENAI_API_KEY",
        )
    )
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if default_endpoint:
        raise ConfigurationError(
            "no LLM API key: set SCRIPTORIUM_LLM_API_KEY or llm_api_key in config.toml "
            "(LLM_API_KEY also works)"
        )
    return ""


def asr_api_key(configured: str = "") -> str:
    """Return the API key for a remote transcription endpoint.

    ``SCRIPTORIUM_ASR_API_KEY`` always wins so one machine can override the
    credential, then the ``asr_api_key`` setting from the file.  The remaining
    environment names are a convenience fallback; ``asr_api_key`` is the
    portable setting and no provider list is involved.
    """
    explicit = os.environ.get("SCRIPTORIUM_ASR_API_KEY", "").strip()
    if explicit:
        return explicit
    if configured.strip():
        return configured.strip()
    for name in _ASR_KEY_ENVS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    names = ("asr_api_key", "SCRIPTORIUM_ASR_API_KEY", *_ASR_KEY_ENVS)
    raise ConfigurationError(
        f"a remote transcription endpoint needs one of {', '.join(names)} to be set"
    )


def _search_paths(data: dict[str, object]) -> list[Path]:
    """Resolve the ordered search paths from the environment and the TOML file."""
    explicit = os.environ.get("SCRIPTORIUM_SEARCH_PATHS")
    values: list[object] = []
    if explicit:
        values.extend(part for part in explicit.split(os.pathsep) if part.strip())
    else:
        raw_paths = data.get("search_paths")
        if isinstance(raw_paths, list):
            values.extend(raw_paths)
    roots: list[Path] = []
    for value in values:
        if value in (None, ""):
            continue
        path = Path(str(value)).expanduser()
        if path not in roots:
            roots.append(path)
    return roots


def _pane_tables(
    raw: object,
) -> tuple[dict[str, str], dict[str, str]]:
    """Read ``[tui_panes]``: import targets and spawn keys.

    Each entry is either a plain import target string (the short form) or a
    table with an optional ``target`` and an optional ``key``.  Size keys are
    not part of a pane table: proportions live in ``tui_layout``.
    """
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            "tui_panes must be a table such as [tui_panes] with name = 'module:Class' entries"
        )
    targets: dict[str, str] = {}
    keys: dict[str, str] = {}
    for name, value in raw.items():
        pane = str(name).strip().lower()
        if not pane:
            raise ConfigurationError("tui_panes cannot contain an empty pane name")
        if isinstance(value, str):
            targets[pane] = value.strip()
            continue
        if not isinstance(value, dict):
            raise ConfigurationError(
                f"tui_panes.{pane} must be an import target or a table with target/key"
            )
        unknown = set(value) - {"target", "key"}
        if unknown:
            raise ConfigurationError(
                f"tui_panes.{pane} has unknown keys: {', '.join(sorted(unknown))}"
            )
        target = value.get("target")
        if target is not None:
            targets[pane] = str(target).strip()
        if "key" in value:
            keys[pane] = "" if value["key"] is None else str(value["key"]).strip()
    return targets, keys


def _string_list(value: object) -> list[str]:
    """Accept a TOML list, a comma/semicolon separated string, or nothing."""
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[,;]", str(value)) if part.strip()]


def config_path() -> Path:
    """Return the path of the optional user configuration file."""
    return _config_path()


def write_config_value(key: str, value: object) -> Path:
    """Persist one flat configuration value, preserving the other keys."""
    return write_config_values({key: value})


def write_config_values(values: dict[str, object]) -> Path:
    """Persist configuration values, preserving the other keys.

    Nested tables such as ``tui_panes`` keep their TOML table syntax; a flat
    ``key = value`` rewrite would turn them into an unreadable string.
    """
    path = _config_write_path()
    data = _read_toml(path)
    data.update(values)
    lines: list[str] = []
    _render_table(lines, "", data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _is_table(item: object) -> bool:
    return isinstance(item, Mapping) or is_dataclass(item)


def _as_table(item: object) -> Mapping[str, object]:
    if isinstance(item, Mapping):
        return item
    return {key: value for key, value in vars(item).items() if not key.startswith("_")}


def _toml_scalar(item: object) -> str:
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, int):
        return str(item)
    if isinstance(item, float):
        return repr(item)
    return json.dumps(str(item), ensure_ascii=False)


def _toml_value(item: object) -> str:
    if isinstance(item, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(entry) for entry in item) + "]"
    return _toml_scalar(item)


def _render_table(lines: list[str], prefix: str, table: Mapping[str, object]) -> None:
    scalars = {
        str(name): item for name, item in table.items() if item is not None and not _is_table(item)
    }
    tables = {str(name): item for name, item in table.items() if _is_table(item)}
    for name in sorted(scalars):
        lines.append(f"{name} = {_toml_value(scalars[name])}")
    for name in sorted(tables):
        header = f"{prefix}.{name}" if prefix else name
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(f"[{header}]")
        _render_table(lines, header, _as_table(tables[name]))
