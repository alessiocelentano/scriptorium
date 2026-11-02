"""Pane registry: resolve box names to modules declared in ``config.toml``.

Each box is one module exposing one :class:`pane_api.Pane` subclass.  The
registry maps a pane name to an import target written as ``module`` or
``module:Class``, so adding a box never requires editing ``tui.py``::

    [tui_panes]
    sources = "my_panes.sources:SourcesPane"

Built-in panes live under ``scriptorium/panes/`` and are the defaults; a config
entry only overrides or extends them.  Resolution is validated eagerly, so a
typo in a module path, a class that is not a pane, or a name mismatch is a
configuration error reported at start-up instead of a half-working interface.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from functools import cache

from .errors import ConfigurationError
from .pane_api import Pane

#: Built-in boxes: pane name -> import target.
BUILTIN_PANES: dict[str, str] = {
    "sources": "scriptorium.panes.sources:SourcesPane",
    "changes": "scriptorium.panes.changes:ChangesPane",
    "flashcards": "scriptorium.panes.flashcards:FlashcardsPane",
    "input": "scriptorium.panes.input_pane:InputPane",
    "extras": "scriptorium.panes.extras:ExtrasPane",
    "logs": "scriptorium.panes.logs:LogsPane",
}


def parse_target(name: str, target: object) -> tuple[str, str]:
    """Split ``"pkg.module:Class"`` into ``(module, attribute)``.

    A bare ``"pkg.module"`` yields an empty attribute, which means "the module
    itself exposes the pane class as ``Pane``".
    """
    if not isinstance(target, str) or not target.strip():
        raise ConfigurationError(
            f"tui_panes.{name} must be an import target such as 'pkg.module:MyPane'"
        )
    text = target.strip()
    module, _, attribute = text.partition(":")
    module = module.strip()
    attribute = attribute.strip()
    if not module or any(part in module for part in ("..", " ", "\t")):
        raise ConfigurationError(f"tui_panes.{name} is not a valid module path: {target!r}")
    return module, attribute


@cache
def _import_target(name: str, module_name: str, attribute: str) -> type[Pane]:
    """Import one pane class, with caching so redraws never re-import."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigurationError(f"tui_panes.{name}: cannot import {module_name!r}: {exc}") from exc
    wanted = attribute or "Pane"
    try:
        candidate = getattr(module, wanted)
    except AttributeError as exc:
        raise ConfigurationError(
            f"tui_panes.{name}: {module_name!r} has no attribute {wanted!r}"
        ) from exc
    if not (isinstance(candidate, type) and issubclass(candidate, Pane)):
        raise ConfigurationError(
            f"tui_panes.{name}: {module_name}.{wanted} must subclass pane_api.Pane"
        )
    pane_name = getattr(candidate, "name", "")
    if pane_name != name:
        raise ConfigurationError(
            f"tui_panes.{name}: {module_name}.{wanted}.name is {pane_name!r}; it must match "
            f"the configured pane name {name!r}"
        )
    return candidate


def resolve_pane(name: str, target: object) -> type[Pane]:
    """Return the pane class registered under ``name``."""
    module_name, attribute = parse_target(name, target)
    return _import_target(name, module_name, attribute)


def pane_targets(overrides: Mapping[str, object] | None = None) -> dict[str, str]:
    """Merge the built-in panes with the ``[tui_panes]`` overrides."""
    targets = dict(BUILTIN_PANES)
    for raw_name, target in (overrides or {}).items():
        name = str(raw_name).strip().lower()
        if not name:
            raise ConfigurationError("tui_panes cannot contain an empty pane name")
        module_name, attribute = parse_target(name, target)
        targets[name] = f"{module_name}:{attribute}" if attribute else module_name
    return targets


def placeable_names(overrides: Mapping[str, object] | None = None) -> tuple[str, ...]:
    """Pane names the layout may contain, in registration order."""
    return tuple(
        name
        for name, target in pane_targets(overrides).items()
        if resolve_pane(name, target).placeable
    )


def known_names(overrides: Mapping[str, object] | None = None) -> tuple[str, ...]:
    """Every registered pane name, placeable or carved from a parent."""
    return tuple(pane_targets(overrides))


class PaneRegistry:
    """One instance per box, plus the geometry metadata they declare."""

    def __init__(self, panes: dict[str, Pane]):
        self._panes = panes

    def __contains__(self, name: object) -> bool:
        return name in self._panes

    def __iter__(self) -> Iterator[str]:
        return iter(self._panes)

    def __len__(self) -> int:
        return len(self._panes)

    def get(self, name: str) -> Pane | None:
        return self._panes.get(name)

    def __getitem__(self, name: str) -> Pane:
        return self._panes[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._panes)

    @property
    def placeable(self) -> tuple[str, ...]:
        return tuple(name for name in self._panes if self._panes[name].placeable)

    def carved(self) -> tuple[str, ...]:
        """Panes that are cut out of another pane's rectangle, in draw order."""
        return tuple(name for name in self._panes if not self._panes[name].placeable)

    def children_of(self, parent: str) -> tuple[str, ...]:
        return tuple(name for name in self.carved() if self._panes[name].parent == parent)

    def reset(self, host: object) -> None:
        """Reset every pane's view state (used by ``/new``)."""
        for pane in self._panes.values():
            pane.reset(host)  # type: ignore[arg-type]


def build_registry(overrides: Mapping[str, object] | None = None) -> PaneRegistry:
    """Instantiate every registered pane.

    Raises :class:`ConfigurationError` when a declared import target cannot be
    resolved, so ``load_config`` fails before the screen is opened.
    """
    panes: dict[str, Pane] = {}
    for name, target in pane_targets(overrides).items():
        pane_class = resolve_pane(name, target)
        try:
            instantiated = pane_class()
        except TypeError as exc:
            raise ConfigurationError(
                f"tui_panes.{name}: {pane_class.__name__} must be constructible "
                f"without arguments: {exc}"
            ) from exc
        panes[name] = instantiated
    return PaneRegistry(panes)
