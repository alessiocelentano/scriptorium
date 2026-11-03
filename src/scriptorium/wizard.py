"""First-run configuration wizard.

The wizard is the small, opinionated subset of ``Config`` a new user must set
before the first ``/run``.  It is deliberately not a second settings screen:
``settings.py`` keeps the full set of editable fields, while this module only
describes the essential pages and how they map back onto real configuration
keys.  ``tui.py`` renders the pages and reuses the settings editor and field
conversion; this module stays free of curses so the mapping is unit-testable.

Two values are wizard-only and never reach ``config.toml``:

- ``llm_provider`` chooses between the DeepSeek default, a local Ollama server
  and any other OpenAI-compatible endpoint, so the user picks a provider
  instead of typing a URL.
- ``asr_mode`` chooses between the local transformers pipeline and a remote
  OpenAI-compatible transcription endpoint.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import (
    _ASR_KEY_ENVS,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    PROJECT_ROOT,
    Config,
)
from .settings import (
    KIND_CHOICE,
    KIND_LIST,
    KIND_PASSWORD,
    KIND_PATH,
    KIND_TEXT,
    PATH_DIR,
    Choice,
    SettingField,
)

# Wizard-only values.
LLM_DEEPSEEK = "deepseek"
LLM_OLLAMA = "ollama"
LLM_CUSTOM = "custom"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1"

ASR_LOCAL = "local"
ASR_REMOTE = "remote"

# Page identifiers.
PAGE_VAULT = "vault"
PAGE_MODEL = "model"
PAGE_SPEECH = "speech"
PAGE_REVIEW = "review"

_PROVIDER_CHOICES = (
    Choice(LLM_DEEPSEEK, "DeepSeek (api.deepseek.com)"),
    Choice(LLM_OLLAMA, "Local (Ollama)"),
    Choice(LLM_CUSTOM, "Other OpenAI-compatible"),
)
_ASR_MODE_CHOICES = (
    Choice(ASR_LOCAL, "Local (download a model)"),
    Choice(ASR_REMOTE, "Remote endpoint"),
)


@dataclass(frozen=True)
class WizardPage:
    key: str
    label: str
    fields: tuple[SettingField, ...]


PAGES: tuple[WizardPage, ...] = (
    WizardPage(
        PAGE_VAULT,
        "Vault",
        (
            SettingField(
                "notes_root",
                "Notes root",
                KIND_PATH,
                "Where notes are written. Created on save if missing",
                target=PATH_DIR,
            ),
            SettingField(
                "search_paths",
                "Search paths",
                KIND_LIST,
                "Folders searched for sources (completion, MOC lookup, Extras export). "
                "Notes root will be automatically added",
                target=PATH_DIR,
                must_exist=True,
            ),
        ),
    ),
    WizardPage(
        PAGE_MODEL,
        "Model",
        (
            SettingField(
                "llm_provider",
                "Provider",
                KIND_CHOICE,
                "Who runs the language model",
                choices=_PROVIDER_CHOICES,
            ),
            SettingField(
                "llm_model",
                "Model",
                KIND_TEXT,
                "Model id. Provider default if missing",
            ),
            SettingField(
                "llm_base_url",
                "Base URL",
                KIND_TEXT,
                "OpenAI-compatible endpoint",
            ),
            SettingField(
                "llm_api_key",
                "API key",
                KIND_PASSWORD,
                "Credential for the endpoint. SCRIPTORIUM_LLM_API_KEY / LLM_API_KEY also work",
            ),
        ),
    ),
    WizardPage(
        PAGE_SPEECH,
        "Speech",
        (
            SettingField(
                "language",
                "Language",
                KIND_TEXT,
                'A code (such as "en" or "it"), "detect" or "ask"',
                empty_value="detect",
            ),
            SettingField(
                "asr_mode",
                "Transcription",
                KIND_CHOICE,
                '"Local" downloads openai/whisper-large-v3 once and runs it here. '
                '"Remote" calls an OpenAI-compatible /audio/transcriptions endpoint',
                choices=_ASR_MODE_CHOICES,
            ),
            SettingField(
                "asr_api_base_url",
                "ASR base URL",
                KIND_TEXT,
                "Endpoint base URL, e.g. https://api.groq.com/openai/v1",
            ),
            SettingField(
                "asr_api_model",
                "ASR model",
                KIND_TEXT,
                "Model id on that server, e.g. whisper-large-v3. "
                "It reuses the local ASR model id if missing",
            ),
            SettingField(
                "asr_api_key",
                "ASR API key",
                KIND_PASSWORD,
                "Credential for the endpoint. SCRIPTORIUM_ASR_API_KEY, ASR_API_KEY, "
                "OPENAI_API_KEY, GROQ_API_KEY, HF_TOKEN also work",
            ),
        ),
    ),
    WizardPage(PAGE_REVIEW, "Review", ()),
)

PAGE_KEYS: tuple[str, ...] = tuple(page.key for page in PAGES)


def page_index(key: str) -> int:
    try:
        return PAGE_KEYS.index(key)
    except ValueError:
        return 0


def page_by_key(key: str) -> WizardPage:
    return PAGES[page_index(key)]


def visible_fields(key: str, values: Mapping[str, object]) -> tuple[SettingField, ...]:
    """The fields of one page, honouring the provider and transcription mode."""
    page = page_by_key(key)
    provider = str(values.get("llm_provider") or LLM_DEEPSEEK)
    asr_mode = str(values.get("asr_mode") or ASR_LOCAL)
    visible: list[SettingField] = []
    for field in page.fields:
        if field.key == "llm_base_url" and provider == LLM_DEEPSEEK:
            # The DeepSeek endpoint is implied by the provider choice.
            continue
        if field.key == "llm_api_key" and provider == LLM_OLLAMA:
            # A local Ollama server needs no credential.
            continue
        if field.key in {"asr_api_base_url", "asr_api_model", "asr_api_key"}:
            if asr_mode == ASR_LOCAL:
                continue
        visible.append(field)
    return tuple(visible)


def _provider_from_config(config: Config) -> str:
    base = config.llm_base_url.rstrip("/")
    if base == DEFAULT_LLM_BASE_URL.rstrip("/"):
        return LLM_DEEPSEEK
    if base in {
        OLLAMA_BASE_URL.rstrip("/"),
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://127.0.0.1:11434/v1",
    }:
        return LLM_OLLAMA
    return LLM_CUSTOM


def default_values(config: Config) -> dict[str, object]:
    """Seed the wizard from the current configuration."""
    return {
        "notes_root": str(config.notes_root),
        "search_paths": [str(path) for path in config.user_search_paths or ()],
        "llm_provider": _provider_from_config(config),
        "llm_model": config.llm_model,
        "llm_base_url": config.llm_base_url,
        "llm_api_key": config.llm_api_key,
        "language": config.language or "detect",
        "asr_mode": ASR_REMOTE if config.asr_api_base_url else ASR_LOCAL,
        "asr_api_base_url": config.asr_api_base_url,
        "asr_api_model": config.asr_api_model,
        "asr_api_key": config.asr_api_key,
    }


def _string_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def wizard_to_config(
    values: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Map wizard values onto ``config.toml`` keys and ``Config`` overrides.

    Raises ``ValueError`` with a user-facing message when a required value is
    missing, so the caller can show it in the wizard instead of writing a
    half-configured file.
    """
    notes_root = str(values.get("notes_root", "")).strip()
    if not notes_root:
        # An empty value is not fatal: notes go to a well-known folder next to
        # the app, and the review page marks this as a warning.
        notes_root = str(PROJECT_ROOT / "notes")
    # Only the user-configured paths are written; Config adds the default
    # source root and the notes root implicitly.
    search_paths = _string_list(values.get("search_paths"))

    provider = str(values.get("llm_provider") or LLM_DEEPSEEK)
    model = str(values.get("llm_model", "")).strip()
    base_url = str(values.get("llm_base_url", "")).strip()
    api_key = str(values.get("llm_api_key", "")).strip()
    if provider == LLM_DEEPSEEK:
        base_url = DEFAULT_LLM_BASE_URL
        model = model or DEFAULT_LLM_MODEL
    elif provider == LLM_OLLAMA:
        base_url = base_url or OLLAMA_BASE_URL
        model = model or OLLAMA_MODEL
        api_key = ""
    elif not base_url:
        raise ValueError("Set a base URL for the custom endpoint")
    elif not model:
        raise ValueError("Set a model id for the custom endpoint")

    asr_mode = str(values.get("asr_mode") or ASR_LOCAL)
    if asr_mode == ASR_REMOTE:
        asr_base_url = str(values.get("asr_api_base_url", "")).strip()
        if not asr_base_url:
            raise ValueError("Set the remote transcription base URL")
        asr_model = str(values.get("asr_api_model", "")).strip()
        asr_key = str(values.get("asr_api_key", "")).strip()
    else:
        asr_base_url = ""
        asr_model = ""
        asr_key = ""

    language = str(values.get("language", "")).strip() or "detect"

    file_values: dict[str, object] = {
        "notes_root": notes_root,
        "search_paths": search_paths,
        "llm_base_url": base_url,
        "llm_model": model,
        "llm_api_key": api_key,
        "language": language,
        "asr_api_base_url": asr_base_url,
        "asr_api_model": asr_model,
        "asr_api_key": asr_key,
    }
    overrides = dict(file_values)
    overrides["notes_root"] = Path(notes_root).expanduser()
    return file_values, overrides


def _provider_label(value: str) -> str:
    for choice in _PROVIDER_CHOICES:
        if choice.value == value:
            return choice.label
    return value


def _asr_label(value: str) -> str:
    for choice in _ASR_MODE_CHOICES:
        if choice.value == value:
            return choice.label
    return value


# Review statuses, drawn as a coloured glyph on the review page.
REVIEW_OK = "ok"
REVIEW_WARN = "warn"
REVIEW_ERROR = "error"

REVIEW_GLYPHS = {REVIEW_OK: "✓", REVIEW_WARN: "⚠", REVIEW_ERROR: "✗"}

# Header shown on the review page as soon as a red row is present.
REVIEW_WARNING = "Some settings are missing or invalid. Scriptorium may not work properly"


def _llm_key_source(values: Mapping[str, object]) -> str:
    """Where the LLM key comes from: ``file``, ``env`` or an empty string."""
    if os.environ.get("SCRIPTORIUM_LLM_API_KEY", "").strip():
        return "env"
    if str(values.get("llm_api_key", "")).strip():
        return "file"
    provider = str(values.get("llm_provider") or LLM_DEEPSEEK)
    names = ("LLM_API_KEY",) if provider == LLM_DEEPSEEK else ("LLM_API_KEY", "OPENAI_API_KEY")
    for name in names:
        if os.environ.get(name, "").strip():
            return "env"
    return ""


def _asr_key_source(values: Mapping[str, object]) -> str:
    """Where the transcription key comes from: ``file``, ``env`` or empty."""
    if os.environ.get("SCRIPTORIUM_ASR_API_KEY", "").strip():
        return "env"
    if str(values.get("asr_api_key", "")).strip():
        return "file"
    for name in _ASR_KEY_ENVS:
        if os.environ.get(name, "").strip():
            return "env"
    return ""


def _local_asr_available() -> bool:
    """True when the local speech-to-text dependencies can be imported."""
    for name in ("torch", "transformers"):
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _local_asr_detail() -> str:
    return "Local transcription needs torch and transformers: pip install -e '.[asr]'"


@dataclass(frozen=True)
class ReviewRow:
    """One line of the review page, with a status and an optional explanation."""

    label: str
    value: str
    status: str = REVIEW_OK
    detail: str = ""


def _path_exists(value: object) -> bool:
    return Path(str(value)).expanduser().is_dir()


def review(values: Mapping[str, object]) -> list[ReviewRow]:
    """Classify every visible wizard value for the review page (never raises).

    ``REVIEW_OK`` is a value that is set and valid, or a choice with a concrete
    selection; ``REVIEW_WARN`` is empty but still works thanks to a default or
    fallback; ``REVIEW_ERROR`` is empty and needed, or invalid, so the run can
    fail.  The caller shows the errors and asks before saving.
    """
    rows: list[ReviewRow] = []
    provider = str(values.get("llm_provider") or LLM_DEEPSEEK)
    asr_mode = str(values.get("asr_mode") or ASR_LOCAL)

    # Vault.
    notes_root = str(values.get("notes_root", "")).strip()
    if notes_root:
        rows.append(ReviewRow("Notes root", notes_root))
    else:
        rows.append(
            ReviewRow(
                "Notes root",
                "(not set)",
                REVIEW_WARN,
                'Empty: a "notes/" folder will be created next to the app',
            )
        )
    search_paths = _string_list(values.get("search_paths"))
    if not search_paths:
        rows.append(
            ReviewRow(
                "Search paths",
                "(notes root only)",
                REVIEW_WARN,
                "Empty: only the notes root is searched for sources",
            )
        )
    else:
        missing = [path for path in search_paths if not _path_exists(path)]
        if missing:
            rows.append(
                ReviewRow(
                    "Search paths",
                    ", ".join(search_paths),
                    REVIEW_ERROR,
                    f"Folder not found: {missing[0]}",
                )
            )
        else:
            rows.append(ReviewRow("Search paths", ", ".join(search_paths)))

    # Model.
    rows.append(ReviewRow("Provider", _provider_label(provider)))
    model = str(values.get("llm_model", "")).strip()
    base_url = str(values.get("llm_base_url", "")).strip()
    if model:
        rows.append(ReviewRow("Model id", model))
    elif provider == LLM_CUSTOM:
        rows.append(
            ReviewRow(
                "Model id",
                "(not set)",
                REVIEW_ERROR,
                "A custom endpoint needs a model id",
            )
        )
    elif provider == LLM_OLLAMA:
        rows.append(
            ReviewRow(
                "Model id",
                f"{OLLAMA_MODEL} (default)",
                REVIEW_WARN,
                "Empty: uses the Ollama default model",
            )
        )
    else:
        rows.append(
            ReviewRow(
                "Model id",
                f"{DEFAULT_LLM_MODEL} (default)",
                REVIEW_WARN,
                "Empty: uses the DeepSeek default model",
            )
        )
    if provider != LLM_DEEPSEEK:
        if base_url:
            rows.append(ReviewRow("Base URL", base_url))
        elif provider == LLM_OLLAMA:
            rows.append(
                ReviewRow(
                    "Base URL",
                    f"{OLLAMA_BASE_URL} (default)",
                    REVIEW_WARN,
                    "Empty: defaults to a local Ollama server",
                )
            )
        else:
            rows.append(
                ReviewRow(
                    "Base URL",
                    "(not set)",
                    REVIEW_ERROR,
                    "A custom endpoint needs a base URL",
                )
            )
    if provider != LLM_OLLAMA:
        key_source = _llm_key_source(values)
        if key_source == "file":
            rows.append(ReviewRow("API key", "set"))
        elif key_source == "env":
            rows.append(ReviewRow("API key", "(from environment)"))
        elif provider == LLM_CUSTOM:
            rows.append(
                ReviewRow(
                    "API key",
                    "(not set)",
                    REVIEW_WARN,
                    "Optional: local servers usually need no key",
                )
            )
        else:
            rows.append(
                ReviewRow(
                    "API key",
                    "(not set)",
                    REVIEW_ERROR,
                    "DeepSeek needs an API key",
                )
            )

    # Speech.
    language = str(values.get("language", "")).strip()
    if language:
        rows.append(ReviewRow("Language", language))
    else:
        rows.append(
            ReviewRow(
                "Language",
                "detect (default)",
                REVIEW_WARN,
                "Empty: the spoken language is detected from the audio",
            )
        )
    if asr_mode == ASR_REMOTE:
        rows.append(ReviewRow("Transcription", _asr_label(asr_mode)))
    elif _local_asr_available():
        rows.append(ReviewRow("Transcription", _asr_label(asr_mode)))
    else:
        rows.append(
            ReviewRow(
                "Transcription",
                _asr_label(asr_mode),
                REVIEW_WARN,
                _local_asr_detail(),
            )
        )
    if asr_mode == ASR_REMOTE:
        asr_base_url = str(values.get("asr_api_base_url", "")).strip()
        if asr_base_url:
            rows.append(ReviewRow("ASR endpoint", asr_base_url))
        else:
            rows.append(
                ReviewRow(
                    "ASR endpoint",
                    "(not set)",
                    REVIEW_ERROR,
                    "A remote endpoint needs a base URL",
                )
            )
        asr_model = str(values.get("asr_api_model", "")).strip()
        if asr_model:
            rows.append(ReviewRow("ASR model", asr_model))
        else:
            rows.append(
                ReviewRow(
                    "ASR model",
                    "openai/whisper-large-v3 (default)",
                    REVIEW_WARN,
                    "Empty: uses the local ASR model id",
                )
            )
        asr_key_source = _asr_key_source(values)
        if asr_key_source == "file":
            rows.append(ReviewRow("ASR API key", "set"))
        elif asr_key_source == "env":
            rows.append(ReviewRow("ASR API key", "(from environment)"))
        else:
            rows.append(
                ReviewRow(
                    "ASR API key",
                    "(not set)",
                    REVIEW_WARN,
                    "Most endpoints need a key: set SCRIPTORIUM_ASR_API_KEY or "
                    "ASR_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, HF_TOKEN",
                )
            )
    return rows


def review_errors(values: Mapping[str, object]) -> list[ReviewRow]:
    """The review rows that would make the run fail (red)."""
    return [row for row in review(values) if row.status == REVIEW_ERROR]
