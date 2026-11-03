"""In-app settings schema.

A declarative description of every ``Config`` field the TUI settings modal can
edit.  ``tui.py`` renders one section in a left column and one field per row in
the right column, then maps each field back to a typed value when the user
saves.  ``config.py`` stays the single source of truth for resolution order and
validation; this module only describes *how* each value is presented and edited
inside the TUI.

The two ``Config`` tables that register custom panes (``tui_panes`` and
``tui_pane_keys``) are deliberately not exposed here: they are TOML tables with
a short-form/long-form syntax that does not map onto a single-line editor, and
a mistake there removes a box from the layout.  Edit those in ``config.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .warnings import known_codes

# Field kinds.
KIND_TEXT = "text"  # free text, one line
KIND_PASSWORD = "password"  # free text, masked while editing
KIND_INT = "int"  # integer, optional minimum/maximum
KIND_FLOAT = "float"  # float, optional minimum/maximum
KIND_CHOICE = "choice"  # cycles through ``choices``
KIND_PATH = "path"  # filesystem path; ``target`` picks the completion flavour
KIND_LIST = "list"  # ordered list of strings (one row per entry)
KIND_FLAGS = "flags"  # multi-select over ``choices``

# Path completion and existence-check targets.
PATH_DIR = "dir"
PATH_FILE = "file"


@dataclass(frozen=True)
class Choice:
    """One selectable value for a ``choice`` or ``flags`` field."""

    value: str
    label: str = ""

    @property
    def display(self) -> str:
        return self.label or self.value


def _choices(*pairs: tuple[str, str]) -> tuple[Choice, ...]:
    return tuple(Choice(value=value, label=label) for value, label in pairs)


@dataclass(frozen=True)
class SettingField:
    """How one ``Config`` field is presented in the settings modal."""

    key: str
    label: str
    kind: str
    help: str = ""
    choices: tuple[Choice, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    exclusive_min: bool = False
    target: str = PATH_DIR
    must_exist: bool = False
    empty_value: str = ""

    def display_value(self, value: object) -> str:
        """Render a stored value for the field row."""
        if self.kind == KIND_CHOICE:
            for choice in self.choices:
                if choice.value == value:
                    return choice.display
        return str(value)


@dataclass(frozen=True)
class SettingSection:
    key: str
    label: str
    fields: tuple[SettingField, ...]


@dataclass(frozen=True)
class SettingsRow:
    """One selectable row in the right column of the settings modal.

    A scalar field produces one row (``index == -1``); a ``list`` field produces
    one row per entry and a ``flags`` field one row per choice, both carrying
    the entry/choice ``index``.
    """

    field: SettingField
    index: int = -1


SECTIONS: tuple[SettingSection, ...] = (
    SettingSection(
        "paths",
        "Paths",
        (
            SettingField(
                "notes_root",
                "Notes root",
                KIND_PATH,
                "Every note write is confined here. Created on save if missing",
                target=PATH_DIR,
                must_exist=True,
            ),
            SettingField(
                "artifacts_dir",
                "Artifacts dir",
                KIND_PATH,
                "Jobs, history and the capability cache",
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
            SettingField(
                "attachment_dir",
                "Attachments dir",
                KIND_PATH,
                "Images extracted from PDFs; relative to the notes root",
                target=PATH_DIR,
            ),
            SettingField(
                "flashcard_skill",
                "Flashcard skill",
                KIND_PATH,
                "Your own copy of manage_flashcards.py",
                target=PATH_FILE,
            ),
        ),
    ),
    SettingSection(
        "model",
        "Model",
        (
            SettingField(
                "llm_model",
                "LLM model",
                KIND_TEXT,
                "Model id. Provider default if missing",
            ),
            SettingField(
                "llm_base_url",
                "LLM base URL",
                KIND_TEXT,
                "Any OpenAI-compatible endpoint",
            ),
            SettingField(
                "llm_api_key",
                "LLM API key",
                KIND_PASSWORD,
                "Credential for the endpoint. SCRIPTORIUM_LLM_API_KEY / LLM_API_KEY also work",
            ),
            SettingField(
                "llm_timeout",
                "LLM timeout",
                KIND_FLOAT,
                "Seconds",
                minimum=0.0,
                exclusive_min=True,
            ),
            SettingField(
                "llm_json_mode",
                "JSON mode",
                KIND_CHOICE,
                "How JSON output is requested",
                choices=_choices(
                    ("object", "object mode"),
                    ("none", "plain text"),
                ),
            ),
        ),
    ),
    SettingSection(
        "asr",
        "Speech-to-text",
        (
            SettingField(
                "asr_model",
                "ASR model",
                KIND_TEXT,
                "HF model id or local directory",
            ),
            SettingField(
                "asr_api_base_url",
                "ASR base URL",
                KIND_TEXT,
                "Endpoint base URL, e.g. https://api.groq.com/openai/v1. "
                "Empty runs the local pipeline",
            ),
            SettingField(
                "asr_api_model",
                "ASR API model",
                KIND_TEXT,
                "Model id on that server, e.g. whisper-large-v3. "
                "It reuses the local ASR model id if missing",
            ),
            SettingField(
                "asr_api_protocol",
                "ASR protocol",
                KIND_CHOICE,
                "Request style of the transcription endpoint",
                choices=_choices(
                    ("openai", "openai"),
                    ("huggingface", "huggingface"),
                ),
            ),
            SettingField(
                "asr_api_key",
                "ASR API key",
                KIND_PASSWORD,
                "Credential for the endpoint. SCRIPTORIUM_ASR_API_KEY, ASR_API_KEY, "
                "OPENAI_API_KEY, GROQ_API_KEY, HF_TOKEN also work",
            ),
            SettingField(
                "asr_api_timeout",
                "ASR timeout",
                KIND_FLOAT,
                "Seconds",
                minimum=0.0,
                exclusive_min=True,
            ),
            SettingField(
                "language",
                "Language",
                KIND_TEXT,
                "A code, or detect / ask",
                empty_value="detect",
            ),
            SettingField(
                "device",
                "Device",
                KIND_CHOICE,
                "Where the local speech-to-text model runs",
                choices=_choices(
                    ("auto", "auto (detect)"),
                    ("cpu", "cpu"),
                    ("cuda", "cuda"),
                    ("mps", "mps"),
                ),
            ),
            SettingField(
                "batch_size",
                "Batch size",
                KIND_INT,
                "Audio chunks transcribed together",
                minimum=1,
            ),
        ),
    ),
    SettingSection(
        "reading",
        "Reading",
        (
            SettingField(
                "image_input",
                "Image input",
                KIND_CHOICE,
                "Whether the model reads images",
                choices=_choices(
                    ("auto", "auto (probe once)"),
                    ("enabled", "enabled"),
                    ("disabled", "disabled"),
                ),
            ),
            SettingField(
                "scanned_pdf",
                "Scanned PDFs",
                KIND_CHOICE,
                "How a page without a text layer is read",
                choices=_choices(
                    ("vision", "vision (model)"),
                    ("ocr", "ocr (Tesseract)"),
                ),
            ),
            SettingField(
                "ocr_languages",
                "OCR languages",
                KIND_TEXT,
                "Tesseract languages joined with +, e.g. eng+ita",
            ),
            SettingField(
                "max_scanned_pages",
                "Max scanned pages",
                KIND_INT,
                "Pages read per PDF by the scanned-page fallback",
                minimum=0,
            ),
            SettingField(
                "extra_context_chars",
                "Context chars",
                KIND_INT,
                "Characters kept from each related-note excerpt",
                minimum=0,
            ),
        ),
    ),
    SettingSection(
        "notes",
        "Notes & cards",
        (
            SettingField(
                "merge_policy",
                "Merge policy",
                KIND_CHOICE,
                "How a merge into an existing note is decided",
                choices=_choices(
                    ("ask", "ask (show diff)"),
                    ("apply", "apply every merge"),
                    ("manual", "manual (conflict files)"),
                ),
            ),
            SettingField(
                "target_policy",
                "Target policy",
                KIND_CHOICE,
                "How the target note of a planned note is chosen",
                choices=_choices(
                    ("ask", "ask (ranked candidates)"),
                    ("auto", "auto (above thresholds)"),
                    ("create", "create (always new note)"),
                ),
            ),
            SettingField(
                "target_auto_above",
                "Auto target above",
                KIND_FLOAT,
                "Minimum score for target_policy = auto",
                minimum=0.0,
                maximum=1.0,
            ),
            SettingField(
                "target_min_gap",
                "Auto target gap",
                KIND_FLOAT,
                "Lead the runner-up must have for target_policy = auto",
                minimum=0.0,
                maximum=1.0,
            ),
            SettingField(
                "flashcard_max_per_note",
                "Max cards per note",
                KIND_INT,
                "Draft cards generated from one note",
                minimum=1,
            ),
        ),
    ),
    SettingSection(
        "interface",
        "Interface",
        (
            SettingField(
                "movement",
                "Movement",
                KIND_CHOICE,
                "Vim modal editing or modeless arrow keys",
                choices=_choices(
                    ("vim", "vim"),
                    ("standard", "standard"),
                ),
            ),
            SettingField(
                "tui_layout",
                "Layout",
                KIND_TEXT,
                "Pane topology: a | b, a / b, ( … ), name:weight",
            ),
        ),
    ),
    SettingSection(
        "warnings",
        "Warnings",
        (
            SettingField(
                "disabled_warnings",
                "Warnings to keep",
                KIND_FLAGS,
                "Unchecked codes are never recorded",
                choices=tuple(Choice(value=code, label=code) for code in known_codes()),
            ),
        ),
    ),
)

SECTION_BY_KEY: dict[str, SettingSection] = {section.key: section for section in SECTIONS}
ALL_FIELDS: tuple[SettingField, ...] = tuple(
    field for section in SECTIONS for field in section.fields
)
FIELD_BY_KEY: dict[str, SettingField] = {field.key: field for field in ALL_FIELDS}


def section_by_key(key: str) -> SettingSection:
    return SECTION_BY_KEY[key]


def cycle_choice(field: SettingField, current: str) -> str:
    """Advance a ``choice`` field to its next allowed value."""
    if not field.choices:
        return current
    for index, choice in enumerate(field.choices):
        if choice.value == current:
            return field.choices[(index + 1) % len(field.choices)].value
    return field.choices[0].value
