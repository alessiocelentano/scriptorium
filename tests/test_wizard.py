"""Tests for the first-run setup wizard (schema and TUI integration)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scriptorium import wizard
from scriptorium.config import Config
from scriptorium.tui import TuiApp


class FakeScreen:
    """Minimal curses screen: record cells instead of painting a terminal."""

    def __init__(self, rows: int = 30, cols: int = 120):
        self.rows = rows
        self.cols = cols
        self.cells: dict[tuple[int, int], str] = {}
        self.last_move: tuple[int, int] | None = None

    def getmaxyx(self):
        return self.rows, self.cols

    def erase(self):
        self.cells.clear()

    def addnstr(self, y, x, text, n, attr=0):
        written = str(text)[:n]
        for offset, char in enumerate(written):
            if 0 <= y < self.rows and 0 <= x + offset < self.cols:
                self.cells[(y, x + offset)] = char
        if written:
            # Real curses advances the terminal cursor after every write.
            self.last_move = (y, x + len(written))

    def move(self, y, x):
        self.last_move = (y, x)

    def refresh(self):
        return None


def _values(**overrides):
    values: dict[str, object] = {
        "notes_root": "/tmp/vault",
        "search_paths": ["/tmp/vault"],
        "llm_provider": wizard.LLM_DEEPSEEK,
        "llm_model": "",
        "llm_base_url": "https://api.deepseek.com",
        "llm_api_key": "secret",
        "language": "it",
        "asr_mode": wizard.ASR_LOCAL,
        "asr_api_base_url": "",
        "asr_api_model": "",
        "asr_api_key": "",
    }
    values.update(overrides)
    return values


class WizardSchemaTests(unittest.TestCase):
    def test_pages_are_ordered_and_end_with_review(self):
        self.assertEqual(
            wizard.PAGE_KEYS,
            (wizard.PAGE_VAULT, wizard.PAGE_MODEL, wizard.PAGE_SPEECH, wizard.PAGE_REVIEW),
        )
        self.assertEqual(wizard.PAGE_KEYS[-1], wizard.PAGE_REVIEW)
        self.assertEqual(wizard.page_by_key(wizard.PAGE_MODEL).label, "Model")

    def test_deepseek_hides_the_base_url_but_keeps_the_key(self):
        keys = {field.key for field in wizard.visible_fields(wizard.PAGE_MODEL, _values())}
        self.assertNotIn("llm_base_url", keys)
        self.assertIn("llm_api_key", keys)

    def test_ollama_hides_the_api_key_and_shows_the_base_url(self):
        fields = wizard.visible_fields(wizard.PAGE_MODEL, _values(llm_provider=wizard.LLM_OLLAMA))
        keys = {field.key for field in fields}
        self.assertIn("llm_base_url", keys)
        self.assertNotIn("llm_api_key", keys)

    def test_local_transcription_hides_every_remote_field(self):
        keys = {field.key for field in wizard.visible_fields(wizard.PAGE_SPEECH, _values())}
        self.assertIn("language", keys)
        self.assertIn("asr_mode", keys)
        self.assertNotIn("asr_api_base_url", keys)
        self.assertNotIn("asr_api_key", keys)

    def test_remote_transcription_shows_the_remote_fields(self):
        keys = {
            field.key
            for field in wizard.visible_fields(
                wizard.PAGE_SPEECH, _values(asr_mode=wizard.ASR_REMOTE)
            )
        }
        self.assertIn("asr_api_base_url", keys)
        self.assertIn("asr_api_model", keys)
        self.assertIn("asr_api_key", keys)

    def test_deepseek_maps_to_the_default_endpoint(self):
        file_values, overrides = wizard.wizard_to_config(
            _values(llm_model="", llm_base_url="ignored")
        )
        self.assertEqual(file_values["llm_base_url"], "https://api.deepseek.com")
        self.assertEqual(file_values["llm_model"], "deepseek-chat")
        self.assertEqual(file_values["llm_api_key"], "secret")
        self.assertIsInstance(overrides["notes_root"], Path)
        self.assertNotIn("llm_provider", file_values)

    def test_ollama_defaults_the_endpoint_and_drops_the_key(self):
        file_values, _ = wizard.wizard_to_config(
            _values(llm_provider=wizard.LLM_OLLAMA, llm_base_url="", llm_model="")
        )
        self.assertEqual(file_values["llm_base_url"], wizard.OLLAMA_BASE_URL)
        self.assertEqual(file_values["llm_model"], wizard.OLLAMA_MODEL)
        self.assertEqual(file_values["llm_api_key"], "")

    def test_custom_provider_requires_a_base_url(self):
        with self.assertRaises(ValueError):
            wizard.wizard_to_config(
                _values(llm_provider=wizard.LLM_CUSTOM, llm_base_url="", llm_model="m")
            )
        file_values, _ = wizard.wizard_to_config(
            _values(
                llm_provider=wizard.LLM_CUSTOM,
                llm_base_url="http://host/v1",
                llm_model="m",
            )
        )
        self.assertEqual(file_values["llm_base_url"], "http://host/v1")

    def test_a_remote_transcription_endpoint_is_required(self):
        with self.assertRaises(ValueError):
            wizard.wizard_to_config(_values(asr_mode=wizard.ASR_REMOTE))
        file_values, _ = wizard.wizard_to_config(
            _values(
                asr_mode=wizard.ASR_REMOTE,
                asr_api_base_url="https://asr/v1",
                asr_api_model="whisper",
                asr_api_key="k",
            )
        )
        self.assertEqual(file_values["asr_api_base_url"], "https://asr/v1")
        self.assertEqual(file_values["asr_api_key"], "k")

    def test_local_transcription_clears_the_remote_values(self):
        file_values, _ = wizard.wizard_to_config(
            _values(
                asr_mode=wizard.ASR_LOCAL,
                asr_api_base_url="https://asr/v1",
                asr_api_model="whisper",
                asr_api_key="k",
            )
        )
        self.assertEqual(file_values["asr_api_base_url"], "")
        self.assertEqual(file_values["asr_api_model"], "")
        self.assertEqual(file_values["asr_api_key"], "")

    def test_empty_language_becomes_detect(self):
        file_values, _ = wizard.wizard_to_config(_values(language=""))
        self.assertEqual(file_values["language"], "detect")

    def test_summary_never_raises_on_empty_values(self):
        rows = wizard.summary_rows({})
        self.assertTrue(rows)
        self.assertTrue(any(label == "Notes root" for label, _ in rows))

    def test_default_values_leave_search_paths_empty(self):
        config = Config(
            notes_root=Path("/tmp/vault"),
            artifacts_dir=Path("/tmp/home"),
        )
        self.assertEqual(wizard.default_values(config)["search_paths"], [])

    def test_only_configured_search_paths_are_seeded_and_written(self):
        config = Config(
            notes_root=Path("/tmp/vault"),
            artifacts_dir=Path("/tmp/home"),
            search_paths=(Path("/tmp/courses"),),
        )
        seeded = wizard.default_values(config)
        self.assertEqual(seeded["search_paths"], ["/tmp/courses"])
        file_values, overrides = wizard.wizard_to_config(seeded)
        self.assertEqual(file_values["search_paths"], ["/tmp/courses"])
        self.assertEqual(overrides["search_paths"], ["/tmp/courses"])

    def test_default_values_recognise_a_local_endpoint(self):
        config = Config(
            notes_root=Path("/tmp/vault"),
            artifacts_dir=Path("/tmp/home"),
            llm_base_url=wizard.OLLAMA_BASE_URL,
        )
        self.assertEqual(wizard.default_values(config)["llm_provider"], wizard.LLM_OLLAMA)

    def test_default_values_recognise_a_custom_endpoint(self):
        config = Config(
            notes_root=Path("/tmp/vault"),
            artifacts_dir=Path("/tmp/home"),
            llm_base_url="http://example.test/v1",
        )
        self.assertEqual(wizard.default_values(config)["llm_provider"], wizard.LLM_CUSTOM)


class WizardTuiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.config = Config(
            notes_root=self.vault,
            artifacts_dir=self.root / "home",
            search_paths=(self.root,),
        )
        self.written: dict[str, object] = {}
        self._write = mock.patch(
            "scriptorium.tui.write_config_values",
            side_effect=lambda values: self.written.update(values),
        )
        self._mark = mock.patch("scriptorium.tui.mark_setup_complete")
        self._write.start()
        self.mark_mock = self._mark.start()

    def tearDown(self):
        self._mark.stop()
        self._write.stop()
        self._tmp.cleanup()

    def _app(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        return app

    def test_open_setup_wizard_seeds_values_and_first_page(self):
        app = self._app()
        app.open_setup_wizard()
        state = app.wizard
        assert state is not None
        self.assertEqual(state.page, wizard.PAGE_VAULT)
        self.assertEqual(state.values["notes_root"], str(self.vault))
        self.assertEqual(state.values["llm_provider"], wizard.LLM_DEEPSEEK)

    def test_pages_advance_and_wrap(self):
        app = self._app()
        app.open_setup_wizard()
        app._handle_wizard_key("\t")
        assert app.wizard is not None
        self.assertEqual(app.wizard.page, wizard.PAGE_MODEL)
        app._handle_wizard_key("b")
        assert app.wizard is not None
        self.assertEqual(app.wizard.page, wizard.PAGE_VAULT)
        app._handle_wizard_key("b")
        assert app.wizard is not None
        self.assertEqual(app.wizard.page, wizard.PAGE_REVIEW)

    def test_choice_cycles_the_provider_and_updates_visibility(self):
        app = self._app()
        app.open_setup_wizard()
        app._handle_wizard_key("\t")  # Model page, provider is the first row
        state = app.wizard
        assert state is not None
        self.assertEqual(state.values["llm_provider"], wizard.LLM_DEEPSEEK)
        app._handle_wizard_key(" ")
        self.assertEqual(state.values["llm_provider"], wizard.LLM_OLLAMA)
        keys = {field.key for field in wizard.visible_fields(state.page, state.values)}
        self.assertNotIn("llm_api_key", keys)

    def test_list_search_path_can_be_added(self):
        app = self._app()
        app.open_setup_wizard()
        state = app.wizard
        assert state is not None
        app._handle_wizard_key("j")  # search paths row
        app._handle_wizard_key("a")
        assert state.editor is not None
        state.editor.set_text(str(self.vault), 0)
        app._handle_wizard_key("\n")
        self.assertIsNone(state.editor)
        paths = state.values["search_paths"]
        assert isinstance(paths, list)
        self.assertIn(self.vault.resolve(), [Path(path) for path in paths])

    def test_review_page_enter_saves_and_creates_the_notes_root(self):
        app = self._app()
        app.open_setup_wizard()
        state = app.wizard
        assert state is not None
        new_root = self.root / "new-vault"
        state.values["notes_root"] = str(new_root)
        for _ in range(3):
            app._handle_wizard_key("\t")
        self.assertEqual(state.page, wizard.PAGE_REVIEW)
        app._handle_wizard_key("\n")
        self.assertIsNone(app.wizard)
        self.assertTrue(new_root.is_dir())
        self.assertEqual(app.config.notes_root, new_root)
        self.assertEqual(self.written["notes_root"], str(new_root))
        self.assertEqual(self.written["llm_base_url"], "https://api.deepseek.com")
        self.mark_mock.assert_called_once()

    def test_skip_key_no_longer_closes_the_wizard(self):
        app = self._app()
        app.open_setup_wizard()
        app._handle_wizard_key("s")
        self.assertIsNotNone(app.wizard)
        self.assertEqual(self.written, {})
        self.mark_mock.assert_not_called()

    def test_escape_keeps_the_wizard_open_and_explains(self):
        app = self._app()
        app.open_setup_wizard()
        state = app.wizard
        assert state is not None
        app._handle_wizard_key("\x1b")
        self.assertIsNotNone(app.wizard)
        self.assertIn("Enter saves", state.message)
        self.assertEqual(self.written, {})
        self.mark_mock.assert_not_called()

    def test_draw_offers_no_skip_key(self):
        app = self._app()
        app.open_setup_wizard()
        app._draw()
        text = "".join(app.scr.cells.values())
        self.assertNotIn("skip", text.lower())

    def _draw_lines(self, app) -> list[str]:
        rows: dict[int, list[tuple[int, str]]] = {}
        for (row, column), char in app.scr.cells.items():
            rows.setdefault(row, []).append((column, char))
        return ["".join(char for _, char in sorted(cells)) for _, cells in sorted(rows.items())]

    def test_draw_renders_the_wizard_page(self):
        app = self._app()
        app.open_setup_wizard()
        app._draw()
        self.assertTrue(app.scr.cells)
        lines = self._draw_lines(app)
        text = "".join(lines)
        self.assertIn("Welcome", text)
        self.assertIn("First-run setup", text)
        self.assertNotIn("everything else lives", text)
        self.assertIn("Vault", text)

    def test_draw_shows_each_description_under_its_option_with_a_gap(self):
        app = self._app()
        app.open_setup_wizard()
        app._draw()
        lines = self._draw_lines(app)
        notes_index = next(index for index, line in enumerate(lines) if "Notes root" in line)
        self.assertIn("The directory where your notes live in", lines[notes_index + 1])
        self.assertEqual(lines[notes_index + 2].strip("║ "), "")
        self.assertIn("Search paths", lines[notes_index + 3])

    def test_search_paths_render_as_one_list_with_a_single_description(self):
        app = self._app()
        app.open_setup_wizard()
        app._draw()
        text = "\n".join(self._draw_lines(app))
        self.assertEqual(text.count("Where you save your sources"), 1)
        self.assertIn("+ add another search path", text)

    def test_the_add_row_opens_the_editor_for_a_new_search_path(self):
        app = self._app()
        app.open_setup_wizard()
        state = app.wizard
        assert state is not None
        rows = app._wizard_rows(state)
        add_index = next(
            index
            for index, row in enumerate(rows)
            if row.field.key == "search_paths" and row.index < 0
        )
        for _ in range(add_index):
            app._handle_wizard_key("j")
        app._handle_wizard_key("\n")
        self.assertIsNotNone(state.editor)
        assert state.editor is not None
        self.assertTrue(state.adding)
        self.assertEqual(state.message, "")
        new_path = self.root / "new-sources"
        new_path.mkdir()
        state.editor.set_text(str(new_path), 0)
        app._handle_wizard_key("\n")
        paths = state.values["search_paths"]
        assert isinstance(paths, list)
        self.assertIn(new_path.resolve(), [Path(path) for path in paths])

    def test_opening_the_editor_does_not_move_the_description_to_the_footer(self):
        app = self._app()
        app.open_setup_wizard()
        state = app.wizard
        assert state is not None
        app._handle_wizard_key("\n")
        self.assertIsNotNone(state.editor)
        self.assertEqual(state.message, "")

    def test_editing_a_list_entry_keeps_the_cursor_in_the_edit_line(self):
        app = self._app()
        app.open_setup_wizard()
        state = app.wizard
        assert state is not None
        # Move to the existing search path row and open it for editing.
        rows = app._wizard_rows(state)
        entry = next(
            index
            for index, row in enumerate(rows)
            if row.field.key == "search_paths" and row.index >= 0
        )
        for _ in range(entry):
            app._handle_wizard_key("j")
        app._handle_wizard_key("\n")
        assert state.editor is not None
        self.assertEqual(state.message, "")
        # A message drawn after the edit line must not steal the cursor.
        state.message = "a footer message"
        app._draw()
        edit_row = app.scr.last_move[0]
        edit_line = "".join(
            char for (row, _), char in sorted(app.scr.cells.items()) if row == edit_row
        )
        self.assertIn("edit: ", edit_line)
        self.assertNotIn("a footer message", edit_line)
        # The cursor column lands after the "edit: " label, inside the text.
        self.assertGreater(app.scr.last_move[1], edit_line.index("edit: "))

    def test_invalid_custom_endpoint_keeps_the_wizard_open(self):
        app = self._app()
        app.open_setup_wizard()
        state = app.wizard
        assert state is not None
        state.values["llm_provider"] = wizard.LLM_CUSTOM
        state.values["llm_base_url"] = ""
        state.page = wizard.PAGE_REVIEW
        app._handle_wizard_key("\n")
        self.assertIsNotNone(app.wizard)
        self.assertIn("base URL", state.message)
        self.assertEqual(self.written, {})


if __name__ == "__main__":
    unittest.main()
