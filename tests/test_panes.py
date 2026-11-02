"""Composability tests: every box is one module, selected by configuration.

These tests protect the extension contract rather than the rendering details:

* a pane module never imports the curses application, so it can be developed
  and shipped on its own;
* ``[tui_panes]`` can add or replace a box without touching ``tui.py``;
* the layout and the size table are validated against the registry, so a typo
  fails at start-up instead of producing a half-working interface.
"""

from __future__ import annotations

import ast
import importlib
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from scriptorium.config import Config, _pane_tables
from scriptorium.errors import ConfigurationError
from scriptorium.pane_api import Pane, PaneHost
from scriptorium.pane_api import __all__ as PANE_API_ALL
from scriptorium.pane_registry import (
    BUILTIN_PANES,
    build_registry,
    known_names,
    pane_targets,
    parse_target,
    placeable_names,
    resolve_pane,
)
from scriptorium.panes.changes import ChangesPane
from scriptorium.panes.input_pane import InputPane
from scriptorium.panes.logs import LogsPane
from scriptorium.tui import TuiApp
from scriptorium.tui_layout import layout_panes, parse_layout

PANE_MODULES = (
    "scriptorium.panes.sources",
    "scriptorium.panes.changes",
    "scriptorium.panes.flashcards",
    "scriptorium.panes.extras",
    "scriptorium.panes.logs",
    "scriptorium.panes.input_pane",
)

CUSTOM_PANE_SOURCE = '''
"""A test-only box declared through the configuration file."""

from scriptorium.pane_api import Pane


class WeatherPane(Pane):
    name = "weather"

    def title(self, host):
        return "Weather · test"

    def rows(self, host):
        return [("sunny", "normal")], -1

    def has_selection(self):
        return False
'''


class FakeScreen:
    """Minimal curses stand-in: records the text written to each cell."""

    def __init__(self, rows: int = 30, cols: int = 120):
        self.rows = rows
        self.cols = cols
        self.cells: dict[tuple[int, int], str] = {}

    def getmaxyx(self):
        return self.rows, self.cols

    def erase(self):
        self.cells.clear()

    def addnstr(self, y, x, text, n, attr=0):
        for offset, char in enumerate(str(text)[:n]):
            if 0 <= y < self.rows and 0 <= x + offset < self.cols:
                self.cells[(y, x + offset)] = char

    def move(self, y, x):
        self.last_move = (y, x)

    def refresh(self):
        pass

    def text(self) -> str:
        return "\n".join(
            "".join(self.cells.get((y, x), " ") for x in range(self.cols)) for y in range(self.rows)
        )


class PaneModuleIndependenceTests(unittest.TestCase):
    """A box must not depend on the host module to be importable."""

    def test_every_pane_module_imports_without_the_curses_app(self):
        for name in PANE_MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertTrue(module.__doc__, f"{name} needs a module docstring")

    def test_no_pane_module_imports_the_host(self):
        root = Path(__file__).resolve().parents[1] / "src" / "scriptorium" / "panes"
        for path in sorted(root.glob("*.py")):
            if path.name == "__init__.py":
                continue
            with self.subTest(module=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        imported = node.module or ""
                        self.assertNotIn(
                            "tui",
                            imported.split("."),
                            f"{path.name} must not import {imported}",
                        )
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotEqual(
                                alias.name, "scriptorium.tui", f"{path.name} imports the host"
                            )

    def test_each_module_exposes_exactly_one_registered_pane(self):
        for name, target in BUILTIN_PANES.items():
            with self.subTest(pane=name):
                pane_class = resolve_pane(name, target)
                self.assertTrue(issubclass(pane_class, Pane))
                self.assertEqual(pane_class.name, name)

    def test_every_builtin_box_is_placeable(self):
        self.assertEqual(
            placeable_names(),
            ("sources", "changes", "flashcards", "input", "extras", "logs"),
        )
        carved = [name for name in known_names() if name not in placeable_names()]
        self.assertEqual(carved, [])


class RegistryResolutionTests(unittest.TestCase):
    def test_parse_target_accepts_module_and_module_class_forms(self):
        self.assertEqual(parse_target("x", "pkg.mod"), ("pkg.mod", ""))
        self.assertEqual(parse_target("x", "pkg.mod:Class"), ("pkg.mod", "Class"))

    def test_parse_target_rejects_empty_and_relative_paths(self):
        for bad in ("", "   ", ":Class", "../evil:Class", 42):
            with self.subTest(target=bad):
                with self.assertRaises(ConfigurationError):
                    parse_target("x", bad)

    def test_an_unimportable_module_is_a_configuration_error(self):
        with self.assertRaises(ConfigurationError) as raised:
            resolve_pane("ghost", "scriptorium.panes.does_not_exist:Pane")
        self.assertIn("cannot import", str(raised.exception))

    def test_a_missing_class_is_a_configuration_error(self):
        with self.assertRaises(ConfigurationError) as raised:
            resolve_pane("ghost", "scriptorium.panes.sources:NotAPane")
        self.assertIn("has no attribute", str(raised.exception))

    def test_the_class_must_subclass_pane(self):
        with self.assertRaises(ConfigurationError):
            resolve_pane("ghost", "pathlib:Path")

    def test_the_pane_name_must_match_its_configuration_key(self):
        with self.assertRaises(ConfigurationError) as raised:
            resolve_pane("wrong", "scriptorium.panes.sources:SourcesPane")
        self.assertIn("must match", str(raised.exception))

    def test_overrides_extend_and_replace_the_builtins(self):
        targets = pane_targets({"changes": "scriptorium.panes.flashcards:FlashcardsPane"})
        self.assertEqual(targets["changes"], "scriptorium.panes.flashcards:FlashcardsPane")
        self.assertIn("sources", targets)

    def test_registry_instantiates_one_pane_per_name(self):
        registry = build_registry()
        self.assertEqual(registry.names, tuple(BUILTIN_PANES))
        self.assertIsInstance(registry["sources"], Pane)


class ConfigSelectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.package = self.root / "test_panes_pkg"
        self.package.mkdir()
        (self.package / "__init__.py").write_text("", encoding="utf-8")
        (self.package / "weather.py").write_text(
            textwrap.dedent(CUSTOM_PANE_SOURCE), encoding="utf-8"
        )
        sys.path.insert(0, str(self.root))
        sys.modules.pop("test_panes_pkg", None)
        sys.modules.pop("test_panes_pkg.weather", None)
        importlib.invalidate_caches()

    def tearDown(self):
        sys.path.remove(str(self.root))
        sys.modules.pop("test_panes_pkg", None)
        sys.modules.pop("test_panes_pkg.weather", None)
        self._tmp.cleanup()

    def config(self, **kwargs) -> Config:
        return Config(notes_root=self.vault, artifacts_dir=self.root / "home", **kwargs)

    def test_a_configured_pane_can_be_placed_in_the_layout(self):
        app = TuiApp(
            config=self.config(
                tui_panes={"weather": "test_panes_pkg.weather:WeatherPane"},
                tui_layout=parse_layout(
                    ["sources | changes | weather", "input"],
                    names=("sources", "changes", "flashcards", "input", "weather"),
                ),
            )
        )
        app.scr = FakeScreen()
        app._draw()
        self.assertIn("Weather · test", app.scr.text())
        self.assertIn("weather", app.layout_rects())

    def test_a_configured_pane_is_registered_even_when_the_layout_omits_it(self):
        app = TuiApp(
            config=self.config(tui_panes={"weather": "test_panes_pkg.weather:WeatherPane"})
        )
        app.scr = FakeScreen()
        self.assertIn("weather", app.panes)
        self.assertNotIn("weather", app.layout_rects())

    def test_a_broken_pane_target_fails_while_loading_the_config(self):
        with self.assertRaises(ConfigurationError):
            self.config(tui_panes={"weather": "test_panes_pkg.missing:WeatherPane"})

    def test_the_layout_cannot_name_an_unregistered_pane(self):
        with self.assertRaises(ConfigurationError):
            self.config(tui_layout=(("sources", "changes", "weather"), ("input",)))

    def test_a_registered_pane_can_be_weighted_in_the_layout(self):
        config = self.config(
            tui_panes={"weather": "test_panes_pkg.weather:WeatherPane"},
            tui_layout=parse_layout(
                ["sources | changes | weather:2", "input"],
                names=("sources", "changes", "flashcards", "input", "weather"),
            ),
        )
        self.assertIn("weather", layout_panes(config.tui_layout))

    def test_extras_and_logs_can_be_placed_directly(self):
        config = self.config(
            tui_layout=parse_layout(["sources | changes | extras | logs", "input"])
        )
        self.assertIn("extras", layout_panes(config.tui_layout))
        self.assertIn("logs", layout_panes(config.tui_layout))

    def test_pane_tables_are_normalized_from_toml(self):
        targets, keys = _pane_tables({"Weather ": {"target": " pkg.mod:Class ", "key": "C-w"}})
        self.assertEqual(targets, {"weather": "pkg.mod:Class"})
        self.assertEqual(keys, {"weather": "C-w"})

    def test_pane_tables_reject_a_non_table(self):
        with self.assertRaises(ConfigurationError):
            _pane_tables(["pkg.mod:Class"])

    def test_pane_tables_reject_size_keys(self):
        with self.assertRaises(ConfigurationError):
            _pane_tables({"weather": {"target": "pkg.mod:Class", "min_width": 20}})


class PaneStateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def app(self) -> TuiApp:
        app = TuiApp(config=Config(notes_root=self.vault, artifacts_dir=self.root / "home"))
        app.scr = FakeScreen()
        return app

    def test_new_session_resets_every_pane(self):
        app = self.app()
        app.source_paths = [self.root / "a.mp4"]
        app.source_index = 0
        app.log_lines = ["one", "two"]
        app.log_scroll = 1
        app._new_session()
        self.assertEqual(app.source_index, 0)
        self.assertEqual(app.source_scroll, 0)
        self.assertEqual(app.log_scroll, 0)

    def test_sources_selection_is_clamped_by_the_pane(self):
        pane = self.app().panes["sources"]
        app = self.app()
        app.source_paths = [self.root / "a.mp4", self.root / "b.mp4"]
        pane.set_selection(app, 99)
        self.assertEqual(app.source_index, 1)
        pane.set_selection(app, -5)
        self.assertEqual(app.source_index, 0)

    def test_the_extras_scroll_is_clamped_to_its_content(self):
        app = self.app()
        app.asides = []
        app.extras_scroll = 999
        app.panes["extras"].set_scroll(app, 999)
        self.assertLessEqual(app.extras_scroll, app.panes["extras"].scroll_limit(app))

    def test_extras_has_its_own_layout_cell(self):
        app = self.app()
        rects = app.layout_rects()
        self.assertIn("extras", rects)
        self.assertNotIn("logs", rects)

    def test_the_host_renders_every_registered_pane_it_has_a_rect_for(self):
        app = self.app()
        app._draw()
        text = app.scr.text()
        for title in ("Sources ·", "Extras ·", "Changes", "Flashcards", "Input"):
            self.assertIn(title, text)

    def test_a_pane_that_is_not_in_the_layout_is_never_drawn(self):
        app = TuiApp(
            config=Config(
                notes_root=self.vault,
                artifacts_dir=self.root / "home",
                tui_layout=parse_layout(
                    ["sources | changes | flashcards", "input"],
                    names=("sources", "changes", "flashcards", "input"),
                ),
            )
        )
        app.scr = FakeScreen()
        app._draw()
        self.assertNotIn("Logs ·", app.scr.text())


class HostCapabilityTests(unittest.TestCase):
    """The host is a faithful PaneHost and never branches on pane names."""

    def test_the_host_satisfies_the_pane_host_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vault = root / "vault"
            vault.mkdir()
            app = TuiApp(config=Config(notes_root=vault, artifacts_dir=root / "home"))
            self.assertIsInstance(app, PaneHost)

    def test_builtin_panes_declare_their_capabilities(self):
        self.assertTrue(InputPane.is_editor)
        self.assertTrue(ChangesPane.owns_merge_review)
        self.assertEqual(LogsPane.spawn_key, "C-l")

    def test_pane_api_exports_the_public_surface(self):
        for name in (
            "Pane",
            "PaneHost",
            "PaneRect",
            "Row",
            "MODE_INSERT",
            "MODE_NORMAL",
            "STYLE_DIM",
            "STYLE_SELECTED",
            "wrap",
            "clip",
            "diff_style",
            "one_line",
        ):
            self.assertIn(name, PANE_API_ALL)

    def test_the_host_has_no_hard_coded_pane_name_branches(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "scriptorium" / "tui.py").read_text(
            encoding="utf-8"
        )
        for name in (
            "PANE_SOURCES",
            "PANE_CHANGES",
            "PANE_DIFFS",
            "PANE_FLASHCARDS",
            "PANE_EXTRAS",
            "PANE_LOGS",
        ):
            self.assertNotIn(name, source, f"tui.py must not branch on {name}")


if __name__ == "__main__":
    unittest.main()
