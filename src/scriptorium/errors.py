"""Typed errors with actionable messages."""

from __future__ import annotations


class ScriptoriumError(Exception):
    """Base class for expected failures."""


class ConfigurationError(ScriptoriumError):
    pass


class DependencyError(ScriptoriumError):
    pass


class StageError(ScriptoriumError):
    pass


class AudioError(StageError):
    pass


class TranscribeError(StageError):
    pass


class LLMError(StageError):
    pass


class PlanValidationError(StageError):
    pass


class VaultError(StageError):
    pass


class FlashcardError(StageError):
    pass
