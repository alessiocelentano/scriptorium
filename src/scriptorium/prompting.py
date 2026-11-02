"""Prompt templates: load a Markdown file from ``prompts/`` and fill placeholders.

Prompts stay in files, never inline in Python; this module is the only place
that knows how to read them and how ``{{placeholder}}`` substitution works.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Any

PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")


def load_prompt(name: str) -> str:
    return resources.files("scriptorium").joinpath("prompts", name).read_text(encoding="utf-8")


def render(template: str, values: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"missing prompt value: {key}")
        return str(values[key])

    return PLACEHOLDER_RE.sub(replace, template)
