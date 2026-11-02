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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, Config
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
                "The directory where your notes live in",
                target=PATH_DIR,
            ),
            SettingField(
                "search_paths",
                "Search paths",
                KIND_LIST,
                "Where you save your sources",
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
                "Model id",
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
                "Your API key. Environment Variables override it",
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
                "Where speech-to-text runs",
                choices=_ASR_MODE_CHOICES,
            ),
            SettingField(
                "asr_api_base_url",
                "ASR base URL",
                KIND_TEXT,
                "OpenAI-compatible transcription endpoint",
            ),
            SettingField(
                "asr_api_model",
                "ASR model",
                KIND_TEXT,
                "Remote model id. Defaults to the local ASR model",
            ),
            SettingField(
                "asr_api_key",
                "ASR API key",
                KIND_PASSWORD,
                "ASR API Key. Only needed by the remote endpoint",
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
        raise ValueError("Notes root cannot be empty")
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


def summary_rows(values: Mapping[str, object]) -> list[tuple[str, str]]:
    """Human-readable rows for the review page (never raises)."""
    notes_root = str(values.get("notes_root", "")).strip() or "(not set)"
    search_paths = _string_list(values.get("search_paths"))
    provider = str(values.get("llm_provider") or LLM_DEEPSEEK)
    model = str(values.get("llm_model", "")).strip() or "(default)"
    base_url = str(values.get("llm_base_url", "")).strip()
    api_key = str(values.get("llm_api_key", "")).strip()
    language = str(values.get("language", "")).strip() or "detect"
    asr_mode = str(values.get("asr_mode") or ASR_LOCAL)
    asr_base_url = str(values.get("asr_api_base_url", "")).strip()

    rows: list[tuple[str, str]] = [
        ("Notes root", notes_root),
        ("Search paths", ", ".join(search_paths) if search_paths else "(notes root)"),
        ("Model", _provider_label(provider)),
        ("Model id", model),
    ]
    if provider != LLM_OLLAMA:
        rows.append(("API key", "set" if api_key else "(unset; env vars also work)"))
    if provider != LLM_DEEPSEEK and base_url:
        rows.append(("Base URL", base_url))
    rows.append(("Language", language))
    rows.append(("Transcription", _asr_label(asr_mode)))
    if asr_mode == ASR_REMOTE:
        rows.append(("ASR endpoint", asr_base_url or "(not set)"))
    return rows
