"""Run warnings collected outside any user interface.

A warning is data, never presentation: stages add warnings to a
:class:`WarningCollector`, the pipeline persists them in the job state and in
the report, and the TUI or the CLI decide later where (and whether) to display
them.  Which codes are kept is user-configurable through
``Config.disabled_warnings``, so the settings modal can expose them without this
module knowing about curses, prompts or terminals.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class WarningCode(StrEnum):
    """Stable identifiers; they are also the values used in the settings file."""

    SOURCE_UNREADABLE = "source_unreadable"
    SOURCE_TEXT_EMPTY = "source_text_empty"
    SOURCE_UNSUPPORTED = "source_unsupported"
    IMAGE_EXTRACTION_FAILED = "image_extraction_failed"
    IMAGE_INPUT_UNSUPPORTED = "image_input_unsupported"
    IMAGE_INPUT_UNKNOWN = "image_input_unknown"
    SCANNED_PDF_PAGE = "scanned_pdf_page"
    SCANNED_PDF_FALLBACK_FAILED = "scanned_pdf_fallback_failed"
    SCANNED_PDF_CAPPED = "scanned_pdf_capped"
    SCANNED_PDF_UNAVAILABLE = "scanned_pdf_unavailable"
    MOC_NOT_FOUND = "moc_not_found"
    MOC_AMBIGUOUS = "moc_ambiguous"
    MOC_CREATED = "moc_created"
    GROUPING_FALLBACK = "grouping_fallback"
    EXTRACTION_CAPPED = "extraction_capped"
    NOTE_CAP_REACHED = "note_cap_reached"
    MERGE_SKIPPED = "merge_skipped"
    WRITE_REFUSED = "write_refused"
    CARDS_EMPTY = "cards_empty"


SEVERITIES = ("info", "warning", "error")


@dataclass(frozen=True)
class RunWarning:
    """One warning, addressable by code and attributed to a source or note."""

    code: str
    message: str
    severity: str = "warning"
    source: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown warning severity: {self.severity!r}")

    @property
    def title(self) -> str:
        return self.code.replace("_", " ")

    def to_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message, "severity": self.severity}
        if self.source:
            payload["source"] = self.source
        if self.detail:
            payload["detail"] = self.detail
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunWarning:
        return cls(
            code=str(payload.get("code") or "unknown"),
            message=str(payload.get("message") or ""),
            severity=str(payload.get("severity") or "warning"),
            source=str(payload["source"]) if payload.get("source") else None,
            detail=str(payload.get("detail") or ""),
        )


@dataclass
class WarningCollector:
    """Ordered, de-duplicated warning list filtered by disabled codes."""

    disabled: tuple[str, ...] = ()
    _items: list[RunWarning] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.disabled = tuple(str(code) for code in self.disabled)

    # -- writing ------------------------------------------------------------- #

    def add(
        self,
        code: str | WarningCode,
        message: str,
        *,
        severity: str = "warning",
        source: str | None = None,
        detail: str = "",
    ) -> RunWarning | None:
        code = str(code)
        if self.is_disabled(code):
            return None
        warning = RunWarning(
            code=code,
            message=message,
            severity=severity,
            source=source,
            detail=detail,
        )
        if warning in self._items:
            return None
        self._items.append(warning)
        return warning

    def extend(self, warnings: Iterable[RunWarning]) -> None:
        for warning in warnings:
            if self.is_disabled(warning.code):
                continue
            if warning not in self._items:
                self._items.append(warning)

    # -- reading ------------------------------------------------------------- #

    def is_disabled(self, code: str) -> bool:
        return code in self.disabled

    def items(self) -> tuple[RunWarning, ...]:
        return tuple(self._items)

    def by_severity(self, severity: str) -> list[RunWarning]:
        return [warning for warning in self._items if warning.severity == severity]

    def to_dicts(self) -> list[dict[str, str]]:
        return [warning.to_dict() for warning in self._items]

    def summary(self) -> str:
        if not self._items:
            return "No warnings."
        counts = {severity: 0 for severity in SEVERITIES}
        for warning in self._items:
            counts[warning.severity] = counts.get(warning.severity, 0) + 1
        parts = [f"{count} {severity}" for severity, count in counts.items() if count]
        return f"{len(self._items)} warning(s): " + ", ".join(parts)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[RunWarning]:
        return iter(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    @classmethod
    def from_dicts(
        cls, payloads: Sequence[Mapping[str, object]], *, disabled: Sequence[str] = ()
    ) -> WarningCollector:
        collector = cls(disabled=tuple(disabled))
        collector.extend(RunWarning.from_dict(payload) for payload in payloads)
        return collector


def known_codes() -> tuple[str, ...]:
    """Every code the application can emit, for settings and documentation."""
    return tuple(code.value for code in WarningCode)
