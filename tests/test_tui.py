import curses
import json
import os
import tempfile
import threading
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scriptorium.config import (
    Config,
    load_config,
    write_config_value,
    write_config_values,
)
from scriptorium.context import build_context
from scriptorium.errors import ConfigurationError
from scriptorium.history import HistoryEntry, HistoryStore
from scriptorium.models import Aside, Flashcard, Plan, PlannedNote
from scriptorium.settings import SECTIONS
from scriptorium.state import GLOBAL_STAGES, GROUP_STAGES, GroupState, JobStore, Stage
from scriptorium.targeting import NoteTarget
from scriptorium.tui import (
    STYLE_ACTIVE_BORDER,
    STYLE_FOCUS,
    PasteEvent,
    PromptModal,
    TuiApp,
)
from scriptorium.tui_completion import (
    complete_files,
    complete_note_paths,
    complete_paths,
    replace_token,
    token_at,
    token_prefix,
    token_spans,
)
from scriptorium.tui_data import (
    CardGroup,
    DiffEntry,
    load_asides,
    load_card_groups,
    load_diff_entries,
)
from scriptorium.tui_input import (
    clean_course_name,
    missing_paths,
    parse_input,
    parse_pasted_input,
)
from scriptorium.tui_layout import (
    DEFAULT_LAYOUT,
    Leaf,
    Split,
    clamp_point,
    content_area,
    edge_point,
    layout_panes,
    layout_text,
    neighbor,
    neighbor_rects,
    pane_rects,
    parse_layout,
    prune_layout,
)
from scriptorium.ui import MocDecision
from scriptorium.vault import MocMatch, Vault
from scriptorium.vimline import MODE_INSERT, MODE_NORMAL


def _tab_to(app, section: str):
    """Cycle the settings modal's sections until ``section`` is active."""
    for _ in range(len(SECTIONS) + 1):
        state = app.settings
        if state is not None and state.section == section:
            return state
        app._handle_settings_key("\t")
    raise AssertionError(f"settings section {section!r} not reached")


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "videos").mkdir()
        (self.root / "videos" / "lecture-01.mp4").write_bytes(b"x")
        (self.root / "archive").mkdir()
        (self.root / "archive" / "lecture-old.mp4").write_bytes(b"x")
        (self.root / "slides").mkdir()
        (self.root / "slides" / "lecture-slides.pdf").write_bytes(b"x")
        (self.root / "notes.md").write_text("notes", encoding="utf-8")
        (self.root / "lecture-folder").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_completion_is_recursive_and_never_returns_directories(self):
        matches = complete_files(self.root, "lecture")
        names = {match.relative for match in matches}
        self.assertEqual(
            names,
            {
                "videos/lecture-01.mp4",
                "archive/lecture-old.mp4",
                "slides/lecture-slides.pdf",
            },
        )
        self.assertNotIn("lecture-folder", names)

    def test_completion_matches_basename_prefix_case_insensitively(self):
        matches = complete_files(self.root, "LECT")
        self.assertEqual(len(matches), 3)
        self.assertTrue(all(match.path.is_file() for match in matches))

    def test_empty_prefix_lists_files_only(self):
        matches = complete_files(self.root, "", limit=100)
        self.assertIn("notes.md", {match.relative for match in matches})
        self.assertNotIn("lecture-folder", {match.relative for match in matches})

    def test_directory_matches_follow_direct_matches_newest_first(self):
        downloads = self.root / "Downloads"
        (downloads / "nested").mkdir(parents=True)
        old = downloads / "old.pdf"
        new = downloads / "new.pdf"
        deep = downloads / "nested" / "deep.pdf"
        for path, modified in ((old, 1_000), (new, 3_000), (deep, 2_000)):
            path.write_bytes(b"x")
            os.utime(path, (modified, modified))
        direct = self.root / "Downloads.txt"
        direct.write_text("x", encoding="utf-8")
        os.utime(direct, (500, 500))

        matches = complete_files(self.root, "Downloads")

        self.assertEqual(
            [match.relative for match in matches],
            [
                "Downloads.txt",
                "Downloads/new.pdf",
                "Downloads/nested/deep.pdf",
                "Downloads/old.pdf",
            ],
        )

    def test_token_helpers(self):
        text = "video.mp4 'University/Course MOC.md' slides.pdf"
        spans = token_spans(text)
        self.assertEqual(len(spans), 3)
        start, end, raw = token_at(text, 15)
        self.assertEqual(text[start:end], "'University/Course MOC.md'")
        self.assertEqual(token_prefix(raw), "Course MOC.md")
        new_text, cursor = replace_token(text, start, end, "/tmp/new.md")
        self.assertIn("/tmp/new.md", new_text)
        self.assertEqual(cursor, new_text.index("/tmp/new.md") + len("/tmp/new.md"))

    def test_unterminated_quote_keeps_the_last_component(self):
        self.assertEqual(token_prefix("'downloads/lec"), "lec")

    def test_note_paths_complete_on_the_bare_stem_or_the_relative_path(self):
        paths = [
            "University/Foundations of AI MOC.md",
            "University/Foundations of ML MOC.md",
            "Other/Logistics MOC.md",
        ]
        names = [match.relative for match in complete_note_paths(paths, "Foundations")]
        self.assertEqual(
            names,
            [
                "University/Foundations of AI MOC.md",
                "University/Foundations of ML MOC.md",
            ],
        )
        relative = [match.relative for match in complete_note_paths(paths, "University/Log")]
        self.assertEqual(relative, [])
        prefixed = [match.relative for match in complete_note_paths(paths, "Other/")]
        self.assertEqual(prefixed, ["Other/Logistics MOC.md"])

    def test_path_completion_offers_directories_and_files(self):
        matches = complete_paths(str(self.root) + "/")
        displays = {match.display for match in matches}
        self.assertTrue(any(display.endswith("videos/") for display in displays))
        self.assertTrue(any(display.endswith("notes.md") for display in displays))

    def test_cursor_after_a_space_starts_a_new_token(self):
        start, end, raw = token_at("video.mp4 ", 10)
        self.assertEqual((start, end, raw), (10, 10, ""))


class MultiRootCompletionTests(unittest.TestCase):
    def setUp(self):
        self._first = tempfile.TemporaryDirectory()
        self._second = tempfile.TemporaryDirectory()
        self.first = Path(self._first.name)
        self.second = Path(self._second.name)
        (self.first / "lecture-one.mp4").write_bytes(b"x")
        (self.second / "lecture-two.mp4").write_bytes(b"x")
        (self.second / "other.txt").write_text("x", encoding="utf-8")

    def tearDown(self):
        self._first.cleanup()
        self._second.cleanup()

    def test_completion_searches_every_root(self):
        matches = complete_files([self.first, self.second], "lecture")
        names = {match.path.name for match in matches}
        self.assertEqual(names, {"lecture-one.mp4", "lecture-two.mp4"})
        self.assertEqual(len({match.group for match in matches}), 2)

    def test_results_are_grouped_by_root_name(self):
        matches = complete_files([self.first, self.second], "lecture")
        displays = {match.display for match in matches}
        self.assertIn(f"{self.first.name}/lecture-one.mp4", displays)
        self.assertIn(f"{self.second.name}/lecture-two.mp4", displays)

    def test_single_root_keeps_the_plain_relative_display(self):
        matches = complete_files(self.first, "lecture")
        self.assertEqual(matches[0].display, "lecture-one.mp4")

    def test_parse_input_resolves_relative_paths_against_every_root(self):
        plan = parse_input("lecture-two.mp4", [self.first, self.second])
        self.assertEqual(plan.sources, [self.second / "lecture-two.mp4"])


class InputTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_parse_input_collects_sources_and_moc_hint(self):
        plan = parse_input(
            "videos/lecture.mp4 'University/Foundations of AI MOC.md' slides.pdf notes.md",
            self.root,
        )
        self.assertFalse(plan.error)
        self.assertEqual(
            plan.sources,
            [
                self.root / "videos" / "lecture.mp4",
                self.root / "slides.pdf",
                self.root / "notes.md",
            ],
        )
        self.assertEqual(plan.moc, self.root / "University" / "Foundations of AI MOC.md")
        self.assertEqual(plan.course, "Foundations of AI")

    def test_parse_input_allows_several_media_files(self):
        plan = parse_input("a.mp4 b.mkv notes.md", self.root)
        self.assertFalse(plan.error)
        self.assertEqual(len(plan.media), 2)
        self.assertEqual(plan.documents, [self.root / "notes.md"])

    def test_parse_input_allows_documents_only(self):
        plan = parse_input("slides.pdf notes.md", self.root)
        self.assertFalse(plan.error)
        self.assertEqual(plan.media, [])
        self.assertEqual(plan.course, "slides")

    def test_parse_input_without_moc_uses_the_first_source_name(self):
        plan = parse_input("lecture-07.mp4", self.root)
        self.assertEqual(plan.course, "lecture 07")

    def test_parse_input_requires_a_source(self):
        plan = parse_input("'University/Foundations of AI MOC.md'", self.root)
        self.assertIn("at least one source", plan.error)
        self.assertTrue(plan.moc)

    def test_parse_input_deduplicates_paths(self):
        plan = parse_input("a.mp4 a.mp4", self.root)
        self.assertEqual(len(plan.sources), 1)

    def test_missing_paths(self):
        plan = parse_input("missing.mp4", self.root)
        self.assertEqual(missing_paths(plan), [self.root / "missing.mp4"])

    def test_parse_pasted_input_keeps_one_path_per_line_with_spaces(self):
        plan = parse_pasted_input(
            "videos/my lecture.mp4\nslides/Week 01.pdf\n",
            self.root,
        )
        self.assertEqual(
            plan.sources,
            [
                self.root / "videos" / "my lecture.mp4",
                self.root / "slides" / "Week 01.pdf",
            ],
        )

    def test_parse_pasted_input_treats_a_single_line_as_one_existing_path(self):
        path = self.root / "a lecture with spaces.mp4"
        path.write_bytes(b"x")
        plan = parse_pasted_input(str(path), self.root)
        self.assertEqual(plan.sources, [path])

    def test_parse_pasted_input_splits_a_single_line_list(self):
        plan = parse_pasted_input("a.mp4 b.mkv", self.root)
        self.assertEqual(plan.sources, [self.root / "a.mp4", self.root / "b.mkv"])

    def test_parse_pasted_input_survives_an_over_long_single_line(self):
        text = " ".join(["lecture.mp4"] * 100)
        plan = parse_pasted_input(text, self.root)
        self.assertEqual(plan.sources, [self.root / "lecture.mp4"])

    def test_parse_pasted_input_reports_an_over_long_path_as_missing(self):
        path = self.root / ("x" * 300)
        plan = parse_pasted_input(str(path), self.root)
        self.assertEqual(missing_paths(plan), [path])

    def test_clean_course_name(self):
        self.assertEqual(clean_course_name("Foundations_of_AI-MOC"), "Foundations of AI")


class TuiDataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_diff_entries(self):
        path = self.root / "apply.json"
        path.write_text(
            """{
              "created": [{"path": "New.md", "action": "create", "added_lines": 3,
                           "removed_lines": 0, "notes": "", "diff": "+line"}],
              "merged": [{"path": "Old.md", "action": "merge", "added_lines": 1,
                          "removed_lines": 0, "notes": "", "diff": "@@ -1 +1 @@\\n+a"}],
              "moc": [], "applied_paths": [], "skipped_merges": [], "refused": [], "todos": []
            }""",
            encoding="utf-8",
        )
        entries = load_diff_entries(path)
        self.assertEqual([entry.path for entry in entries], ["New.md", "Old.md"])
        self.assertEqual(entries[0].group, "created")
        self.assertTrue(entries[0].diff_lines)

    def test_load_card_groups(self):
        plan = Plan(
            course="Course",
            notes=[
                PlannedNote(
                    action="create",
                    path="New.md",
                    title="New",
                    flashcards=[Flashcard(front="What is X?", back="Y.")],
                )
            ],
        )
        plan_path = self.root / "05.plan.json"
        plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
        cards_path = self.root / "cards.json"
        cards_path.write_text(
            '[{"path": "New.md", "requested": 1, "written": 1, '
            '"skipped": 0, "writer": "internal"}]',
            encoding="utf-8",
        )
        groups = load_card_groups(plan_path, cards_path)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].written, 1)
        self.assertEqual(groups[0].cards[0].front, "What is X?")

    def test_load_asides(self):
        plan = Plan(
            course="Course",
            asides=[Aside(kind="exam", text="The exam is oral.", source="lecture.mp4 00:00:10")],
        )
        plan_path = self.root / "05.plan.json"
        plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
        asides = load_asides(plan_path)
        self.assertEqual(len(asides), 1)
        self.assertEqual(asides[0].kind, "exam")
        self.assertEqual(asides[0].source, "lecture.mp4 00:00:10")

        extract_path = self.root / "03.extract.json"
        extract_path.write_text(
            '{"units": [], "asides": [{"kind": "link", "text": "See the course page.", '
            '"source": "lecture.mp4 00:01:00"}]}',
            encoding="utf-8",
        )
        asides = load_asides(plan_path, extract_path)
        self.assertEqual(asides[0].kind, "link")
        self.assertEqual(asides[0].source, "lecture.mp4 00:01:00")


class ConfigWriteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "scriptorium" / "config.toml"
        self.env = mock.patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")})
        self.env.start()
        self._path_patch = mock.patch("scriptorium.config._config_path", return_value=self.path)
        self._write_patch = mock.patch(
            "scriptorium.config._config_write_path", return_value=self.path
        )
        self._path_patch.start()
        self._write_patch.start()

    def tearDown(self):
        self._write_patch.stop()
        self._path_patch.stop()
        self.env.stop()
        self._tmp.cleanup()

    def test_write_config_value_preserves_existing_values(self):
        notes_root = self.root / "vault"
        notes_root.mkdir()
        path = write_config_value("notes_root", str(notes_root))
        write_config_value("search_paths", [str(self.root / "videos")])
        config = load_config()
        self.assertEqual(config.notes_root, notes_root)
        self.assertEqual(config.search_paths[0], self.root / "videos")
        self.assertIn("notes_root =", path.read_text(encoding="utf-8"))

    def test_search_paths_are_loaded_and_deduplicated(self):
        notes_root = self.root / "vault"
        notes_root.mkdir()
        first = self.root / "videos"
        second = self.root / "downloads"
        first.mkdir()
        second.mkdir()
        write_config_values(
            {
                "notes_root": str(notes_root),
                "search_paths": [str(first), str(first), str(second)],
            }
        )
        config = load_config()
        self.assertEqual(config.search_paths, (first, second, notes_root))

    def test_write_config_values_preserves_nested_tables(self):
        notes_root = self.root / "vault"
        notes_root.mkdir()
        path = self.root / "scriptorium" / "config.toml"
        path.parent.mkdir()
        path.write_text(
            f"""notes_root = "{notes_root}"
[tui_panes.logs]
key = "C-o"
""",
            encoding="utf-8",
        )
        write_config_value("image_input", "enabled")
        text = path.read_text(encoding="utf-8")
        self.assertIn("[tui_panes.logs]", text)
        config = load_config()
        self.assertEqual(config.tui_pane_keys["logs"], "C-o")
        self.assertEqual(config.image_input, "enabled")

    def test_tui_layout_is_loaded_from_the_config_file(self):
        notes_root = self.root / "vault"
        notes_root.mkdir()
        write_config_values(
            {
                "notes_root": str(notes_root),
                "tui_layout": ["input", "sources | changes | flashcards"],
            }
        )
        config = load_config()
        self.assertEqual(
            layout_text(config.tui_layout),
            "input / (sources | changes | flashcards)",
        )
        self.assertEqual(
            layout_panes(config.tui_layout),
            ("input", "sources", "changes", "flashcards"),
        )

    def test_tui_pane_tables_carry_spawn_keys(self):
        notes_root = self.root / "vault"
        notes_root.mkdir()
        path = self.root / "scriptorium" / "config.toml"
        path.parent.mkdir()
        path.write_text(
            f"""notes_root = "{notes_root}"
[tui_panes.logs]
target = "scriptorium.panes.logs:LogsPane"
key = "C-o"
""",
            encoding="utf-8",
        )
        config = load_config()
        self.assertEqual(config.tui_pane_keys["logs"], "C-o")

    def test_tui_pane_tables_reject_size_keys(self):
        notes_root = self.root / "vault"
        notes_root.mkdir()
        path = self.root / "scriptorium" / "config.toml"
        path.parent.mkdir()
        path.write_text(
            f"""notes_root = "{notes_root}"
[tui_panes.sources]
min_width = 20
""",
            encoding="utf-8",
        )
        with self.assertRaises(ConfigurationError):
            load_config()

    def test_settings_modal_adds_a_root_and_saves_it(self):
        vault = self.root / "vault"
        vault.mkdir()
        first = self.root / "videos"
        second = self.root / "downloads"
        first.mkdir()
        second.mkdir()
        config = Config(notes_root=vault, artifacts_dir=self.root / "home", search_paths=(first,))
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        state = app.settings
        assert state is not None
        # Paths section: notes root, artifacts dir, then the search-path rows.
        app._handle_settings_key("j")
        app._handle_settings_key("j")
        self.assertEqual(state.values["search_paths"], [str(first)])
        app._handle_settings_key("a")
        editor = state.editor
        assert editor is not None
        editor.set_text(str(second), 0)
        app._handle_settings_key("\n")
        self.assertEqual(
            state.values["search_paths"],
            [str(first), str(second.resolve())],
        )
        app._handle_settings_key("s")
        self.assertIsNone(app.settings)
        # After saving, the implicit notes root is appended by Config.
        self.assertEqual(app.search_roots, [first, second.resolve(), vault])
        saved = load_config()
        self.assertEqual(
            tuple(root.resolve() for root in saved.search_paths),
            (first.resolve(), second.resolve(), vault.resolve()),
        )
        # The notes root stays implicit: it is not written back to the file.
        data = tomllib.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(
            [Path(path).resolve() for path in data["search_paths"]],
            [first.resolve(), second.resolve()],
        )

    def test_settings_path_editor_completes_directories_with_tab(self):
        vault = self.root / "vault"
        vault.mkdir()
        parent = self.root / "roots"
        parent.mkdir()
        (parent / "alpha").mkdir()
        (parent / "beta").mkdir()
        config = Config(
            notes_root=vault,
            artifacts_dir=self.root / "home",
            search_paths=(self.root,),
        )
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        state = app.settings
        assert state is not None
        app._handle_settings_key("j")
        app._handle_settings_key("j")
        app._handle_settings_key("a")
        assert state.editor is not None
        state.editor.set_text(str(parent) + "/", 0)
        app._handle_settings_key("\t")
        self.assertEqual(state.editor.text, str(parent / "alpha") + "/")
        self.assertIsNotNone(state.completion)
        app._handle_settings_key("\t")
        self.assertEqual(state.editor.text, str(parent / "beta") + "/")
        app._handle_settings_key("\n")
        self.assertIsNone(state.completion)
        app._handle_settings_key("\n")
        paths = state.values["search_paths"]
        assert isinstance(paths, list)
        self.assertIn((parent / "beta").resolve(), [Path(path) for path in paths])

    def test_settings_modal_saves_sources_paths_and_warnings(self):
        vault = self.root / "vault"
        vault.mkdir()
        courses = self.root / "courses"
        courses.mkdir()
        config = Config(
            notes_root=vault,
            artifacts_dir=self.root / "home",
            search_paths=(self.root,),
        )
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        state = app.settings
        assert state is not None

        # attachment_dir follows notes_root, artifacts_dir and the search root.
        for _ in range(3):
            app._handle_settings_key("j")
        app._handle_settings_key("e")
        assert state.editor is not None
        state.editor.set_text("Attachments/scriptorium", 0)
        app._handle_settings_key("\n")
        self.assertEqual(state.values["attachment_dir"], "Attachments/scriptorium")

        _tab_to(app, "warnings")
        app._handle_settings_key(" ")
        disabled = state.values["disabled_warnings"]
        assert isinstance(disabled, list)
        self.assertEqual(len(disabled), 1)

        app._handle_settings_key("s")
        self.assertIsNone(app.settings)
        saved = load_config()
        self.assertEqual(saved.image_input, "auto")
        self.assertEqual(saved.attachment_dir, Path("Attachments/scriptorium"))
        self.assertEqual(len(saved.disabled_warnings), 1)
        self.assertIn(self.root.resolve(), tuple(root.resolve() for root in saved.search_paths))

    def test_settings_accepts_an_absolute_attachment_dir(self):
        vault = self.root / "vault"
        vault.mkdir()
        config = Config(
            notes_root=vault,
            artifacts_dir=self.root / "home",
            search_paths=(self.root,),
        )
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        state = app.settings
        assert state is not None
        for _ in range(3):
            app._handle_settings_key("j")
        app._handle_settings_key("e")
        assert state.editor is not None
        state.editor.set_text(str(self.root / "elsewhere"), 0)
        app._handle_settings_key("\n")
        self.assertIsNone(state.editor)
        self.assertEqual(
            state.values["attachment_dir"],
            str((self.root / "elsewhere").resolve()),
        )

    def test_settings_modal_rejects_a_missing_directory(self):
        vault = self.root / "vault"
        vault.mkdir()
        config = Config(
            notes_root=vault,
            artifacts_dir=self.root / "home",
            search_paths=(self.root,),
        )
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        state = app.settings
        assert state is not None
        app._handle_settings_key("e")
        editor = state.editor
        assert editor is not None
        editor.set_text(str(self.root / "missing"), 0)
        app._handle_settings_key("\n")
        self.assertIsNotNone(state.editor)
        self.assertIn("not a directory", state.message)

    def test_settings_choice_cycles_and_saves(self):
        vault = self.root / "vault"
        vault.mkdir()
        config = Config(notes_root=vault, artifacts_dir=self.root / "home")
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        _tab_to(app, "interface")
        state = app.settings
        assert state is not None
        app._handle_settings_key("e")  # movement is the first field
        self.assertEqual(state.values["movement"], "standard")
        app._handle_settings_key("s")
        self.assertIsNone(app.settings)
        saved = load_config()
        self.assertEqual(saved.movement, "standard")
        self.assertFalse(app.vim_movements)
        self.assertEqual(app.mode, MODE_INSERT)

    def test_settings_device_is_a_choice_not_free_text(self):
        vault = self.root / "vault"
        vault.mkdir()
        config = Config(notes_root=vault, artifacts_dir=self.root / "home")
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        _tab_to(app, "asr")
        state = app.settings
        assert state is not None
        # device is the eighth field: after the ASR block and language.
        for _ in range(7):
            app._handle_settings_key("j")
        app._handle_settings_key("e")
        self.assertIsNone(state.editor)
        self.assertEqual(state.values["device"], "cpu")
        app._handle_settings_key(" ")
        self.assertEqual(state.values["device"], "cuda")

    def test_settings_int_field_rejects_out_of_range(self):
        vault = self.root / "vault"
        vault.mkdir()
        config = Config(notes_root=vault, artifacts_dir=self.root / "home")
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        _tab_to(app, "reading")
        state = app.settings
        assert state is not None
        # max_scanned_pages is the fourth field; its minimum is 0.
        for _ in range(3):
            app._handle_settings_key("j")
        app._handle_settings_key("e")
        assert state.editor is not None
        state.editor.set_text("-1", 0)
        app._handle_settings_key("\n")
        self.assertIsNotNone(state.editor)
        self.assertIn("at least 0", state.message)

    def test_settings_password_is_masked_when_drawn(self):
        vault = self.root / "vault"
        vault.mkdir()
        config = Config(notes_root=vault, artifacts_dir=self.root / "home")
        app = TuiApp(config=config)
        app.scr = FakeScreen(rows=30, cols=100)
        app._open_settings()
        _tab_to(app, "model")
        state = app.settings
        assert state is not None
        # llm_api_key is the third field.
        app._handle_settings_key("j")
        app._handle_settings_key("j")
        app._handle_settings_key("e")
        assert state.editor is not None
        state.editor.set_text("sekrit", 0)
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("••••••", text)
        self.assertNotIn("sekrit", text)
        app._handle_settings_key("\x1b")
        app._handle_settings_key("\x1b")
        self.assertIsNone(app.settings)

    def test_settings_saves_a_text_field(self):
        vault = self.root / "vault"
        vault.mkdir()
        config = Config(notes_root=vault, artifacts_dir=self.root / "home")
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        _tab_to(app, "model")
        state = app.settings
        assert state is not None
        app._handle_settings_key("e")  # llm_model is the first field
        assert state.editor is not None
        state.editor.set_text("llama3.1", 0)
        app._handle_settings_key("\n")
        app._handle_settings_key("s")
        saved = load_config()
        self.assertEqual(saved.llm_model, "llama3.1")

    def test_settings_layout_is_validated_on_save(self):
        vault = self.root / "vault"
        vault.mkdir()
        config = Config(notes_root=vault, artifacts_dir=self.root / "home")
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app._open_settings()
        _tab_to(app, "interface")
        state = app.settings
        assert state is not None
        app._handle_settings_key("j")  # tui_layout is the second field
        app._handle_settings_key("e")
        assert state.editor is not None
        state.editor.set_text("not a layout /", 0)
        app._handle_settings_key("\n")
        app._handle_settings_key("s")
        self.assertIsNotNone(app.settings)
        self.assertIn("invalid settings", state.message)


class ExtrasExportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.exports = self.root / "exports"
        self.exports.mkdir()
        self.config = Config(
            notes_root=self.vault,
            artifacts_dir=self.root / "home",
            search_paths=(self.exports,),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _app_with_asides(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.asides = [Aside(kind="exam", text="The exam is oral.", source="lecture.mp4 00:00:10")]
        app.focus = "extras"
        app.active = "extras"
        return app

    def test_c_key_copies_every_extra_to_the_clipboard(self):
        app = self._app_with_asides()
        with mock.patch("scriptorium.tui.copy_to_clipboard", return_value=True) as copy:
            app._handle_active_key("c")
        copy.assert_called_once()
        self.assertIn("[exam]", copy.call_args.args[0])
        self.assertIn("The exam is oral.", copy.call_args.args[0])
        self.assertIn("copied 1 extras", app.status)

    def test_c_key_without_extras_does_not_touch_the_clipboard(self):
        app = TuiApp(config=self.config)
        app.focus = "extras"
        app.active = "extras"
        with mock.patch("scriptorium.tui.copy_to_clipboard", return_value=True) as copy:
            app._handle_active_key("c")
        copy.assert_not_called()
        self.assertEqual(app.status, "no extras to copy")

    def test_w_key_writes_to_a_configured_root_and_appends_next_time(self):
        app = self._app_with_asides()
        app._handle_active_key("w")
        self.assertIsNotNone(app.modal)
        assert app.modal is not None
        self.assertEqual(app.modal.kind, "extras_export")
        app._handle_modal_key("\n")
        target = self.exports / "extras.md"
        self.assertTrue(target.is_file())
        self.assertIsNone(app.modal)
        self.assertIn("The exam is oral.", target.read_text(encoding="utf-8"))

        app.asides = [Aside(kind="link", text="See the course page.")]
        app._handle_active_key("w")
        app._handle_modal_key("\n")
        text = target.read_text(encoding="utf-8")
        self.assertIn("The exam is oral.", text)
        self.assertIn("See the course page.", text)
        self.assertLess(text.index("The exam is oral."), text.index("See the course page."))

    def test_new_path_completes_files_and_writes_them(self):
        app = self._app_with_asides()
        folder = self.root / "output"
        folder.mkdir()
        (folder / "alpha").mkdir()
        (folder / "notes.md").write_text("", encoding="utf-8")
        app._handle_active_key("w")
        assert app.modal is not None
        # Two roots are offered (the configured one and the implicit notes
        # root); the last entry writes to a new path.
        app._handle_modal_key("j")
        app._handle_modal_key("j")
        self.assertEqual(app.modal.selected, 2)
        app._handle_modal_key("\n")
        modal = app.modal
        assert modal is not None and modal.editor is not None
        modal.editor.set_text(str(folder) + "/", 0)
        app._handle_modal_key("\t")
        self.assertEqual(modal.editor.text, str(folder / "alpha") + "/")
        app._handle_modal_key("\t")
        self.assertEqual(modal.editor.text, str(folder / "notes.md"))
        app._handle_modal_key("\n")
        self.assertIsNone(modal.completion)
        app._handle_modal_key("\n")
        self.assertIsNone(app.modal)
        self.assertIn("The exam is oral.", (folder / "notes.md").read_text(encoding="utf-8"))

    def test_the_export_modal_renders_its_configured_paths(self):
        app = self._app_with_asides()
        app.scr = FakeScreen(rows=30, cols=80)
        app._handle_active_key("w")
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("Write Extras to a file", text)
        self.assertIn("write to a new path", text)


class WrappingTests(unittest.TestCase):
    """Long content must wrap instead of being clipped at the right edge."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.config = Config(notes_root=self.vault, artifacts_dir=self.root / "home")

    def tearDown(self):
        self._tmp.cleanup()

    def _screen_text(self, app):
        return "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )

    def test_a_long_target_option_wraps_and_is_not_ellipsized(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen(rows=30, cols=60)
        anchor = "A very long anchor heading that ends with ZZZFINISH"
        option = NoteTarget(
            kind="merge",
            path=str(self.vault / "Section" / "Another Section" / "Note.md"),
            title="Target note",
            anchor=anchor,
            score=0.9,
        )
        app.modal = PromptModal(
            kind="target",
            payload={"options": [option]},
            box={},
            event=threading.Event(),
        )
        app._draw()
        text = self._screen_text(app)
        self.assertIn("ZZZFINISH", text)

    def test_a_long_source_path_wraps_inside_its_pane(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen(rows=30, cols=60)
        folder = self.root / ("deeply nested " * 4 + "ZZZPARENT")
        app.source_paths = [folder / "lecture.pdf"]
        app._draw()
        self.assertIn("ZZZPARENT", self._screen_text(app))

    def test_a_long_settings_path_wraps_inside_the_window(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen(rows=30, cols=60)
        app._open_settings()
        state = app.settings
        assert state is not None
        state.values["search_paths"] = [str(self.root / ("a" * 20 + " ZZZROOT") / "subdirectory")]
        app._draw()
        self.assertIn("ZZZROOT", self._screen_text(app))

    def test_settings_lists_sections_in_a_separate_column(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen(rows=30, cols=80)
        app._open_settings()
        app._draw()
        text = self._screen_text(app)
        self.assertIn("Sections", text)
        self.assertIn("Search paths", text)
        self.assertIn("│", text)

    def test_a_long_merge_preview_line_wraps(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen(rows=30, cols=60)
        app.merge_review = PromptModal(
            kind="merge",
            payload={
                "note": SimpleNamespace(
                    path=str(self.vault / "A very long note path ZZZNOTE.md"),
                    anchor="Section heading",
                ),
                "preview": "+ " + "word " * 20 + "ZZZPREVIEW",
            },
            box={},
            event=threading.Event(),
        )
        app._draw()
        text = self._screen_text(app)
        self.assertIn("ZZZPREVIEW", text)

    def test_extras_wrap_and_scroll_without_ellipsis(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen(rows=30, cols=60)
        app.asides = [
            Aside(
                kind="university",
                text=("A long remark about the exam and the room " * 4) + "ZZZEND",
                source="lecture.mp4 00:00:10",
            )
            for _ in range(8)
        ]
        app.focus = "extras"
        app._draw()
        text = self._screen_text(app)
        self.assertIn("Extras · 8", text)
        app._handle_normal_key("\n")
        app._handle_normal_key("G")
        app._draw()
        self.assertIn("ZZZEND", self._screen_text(app))
        self.assertGreater(app.extras_scroll, 0)


class LayoutTests(unittest.TestCase):
    def test_default_layout_places_changes_left_cards_right_input_below(self):
        rects = pane_rects(DEFAULT_LAYOUT, 30, 120)
        changes = rects["changes"]
        cards = rects["flashcards"]
        input_rect = rects["input"]
        self.assertLess(changes.x, cards.x)
        self.assertEqual(changes.y, cards.y)
        self.assertGreater(input_rect.y, changes.y)
        self.assertEqual(input_rect.h, 3)

    def test_extras_and_logs_take_half_of_their_column(self):
        rects = pane_rects(DEFAULT_LAYOUT, 30, 120)
        sources = rects["sources"]
        extras = rects["extras"]
        flashcards = rects["flashcards"]
        logs = rects["logs"]
        self.assertEqual((sources.x, sources.w), (extras.x, extras.w))
        self.assertEqual((flashcards.x, flashcards.w), (logs.x, logs.w))
        self.assertEqual(sources.h, extras.h)
        self.assertEqual(flashcards.h, logs.h)
        self.assertEqual(extras.y, sources.bottom)
        self.assertEqual(logs.y, flashcards.bottom)
        self.assertEqual(sources.h + extras.h, rects["changes"].h)
        self.assertEqual(extras.bottom, rects["input"].y)
        self.assertEqual(rects["input"].h, 3)
        self.assertEqual(rects["input"].bottom, 30)

    def test_weights_are_relative_to_the_enclosing_split(self):
        layout = parse_layout(["(sources / extras) | changes:2 | (flashcards / logs:3)", "input"])
        rects = pane_rects(layout, 30, 120)
        self.assertEqual(rects["sources"].w, 29)
        self.assertEqual(rects["changes"].w, 58)
        self.assertEqual(rects["flashcards"].w, 29)
        self.assertEqual(rects["sources"].h, rects["extras"].h)
        self.assertGreater(rects["logs"].h, rects["flashcards"].h)

    def test_a_hidden_pane_redistributes_its_share(self):
        layout = parse_layout(["(sources / extras) | changes | (flashcards / logs)", "input"])
        shown = pane_rects(layout, 30, 120)
        hidden = pane_rects(layout, 30, 120, frozenset({"logs"}))
        self.assertNotIn("logs", hidden)
        self.assertEqual(hidden["flashcards"].h, shown["flashcards"].h + shown["logs"].h)
        self.assertEqual(hidden["sources"].h, hidden["extras"].h)

    def test_a_split_left_with_one_child_collapses(self):
        layout = parse_layout(["sources | changes | (flashcards / logs)", "input"])
        rects = pane_rects(layout, 30, 120, frozenset({"logs"}))
        self.assertNotIn("logs", rects)
        self.assertEqual(rects["flashcards"].h, rects["changes"].h)

    def test_a_nested_split_is_recursive(self):
        layout = parse_layout(["sources | (changes / (extras | logs))", "input"])
        rects = pane_rects(layout, 30, 120)
        self.assertEqual(rects["changes"].w, rects["extras"].w + 1 + rects["logs"].w)
        self.assertEqual(rects["changes"].h, rects["extras"].h)
        self.assertEqual(rects["extras"].h, rects["logs"].h)
        self.assertEqual(rects["sources"].h, rects["changes"].h + rects["extras"].h)

    def test_the_input_keeps_three_rows_at_every_size(self):
        for rows in (10, 12, 15, 24, 30, 40, 60, 80):
            rects = pane_rects(DEFAULT_LAYOUT, rows, 120)
            self.assertEqual(rects["input"].h, 3, f"terminal of {rows} rows")
            self.assertEqual(rects["input"].bottom, rows)
            self.assertGreaterEqual(rects["extras"].h, 1, f"terminal of {rows} rows")
            self.assertGreaterEqual(rects["logs"].h, 1, f"terminal of {rows} rows")

    def test_a_short_terminal_shrinks_the_other_rows_before_the_input_line(self):
        rects = pane_rects(DEFAULT_LAYOUT, 8, 120)
        self.assertEqual(rects["input"].h, 3)
        self.assertEqual(rects["sources"].h, 2)

    def test_layout_weights_and_nesting_round_trip(self):
        layout = parse_layout(["(sources:2 / extras) | changes:3 | (flashcards / logs)", "input"])
        self.assertEqual(
            layout_text(layout),
            "((sources:2 / extras) | changes:3 | (flashcards / logs)) / input",
        )
        self.assertEqual(
            layout_panes(layout),
            ("sources", "extras", "changes", "flashcards", "logs", "input"),
        )

    def test_layout_can_be_one_expression(self):
        layout = parse_layout("((sources / extras) | changes | (flashcards / logs)) / input")
        self.assertEqual(
            layout_panes(layout),
            ("sources", "extras", "changes", "flashcards", "logs", "input"),
        )

    def test_layout_rejects_mixed_operators_without_parentheses(self):
        with self.assertRaises(ConfigurationError):
            parse_layout(["sources | changes / flashcards", "input"])

    def test_layout_weights_must_be_positive(self):
        for bad in ("0", "-1", "abc"):
            with self.assertRaises(ConfigurationError):
                parse_layout([f"sources:{bad} | changes | flashcards", "input"])

    def test_layout_rejects_unclosed_parentheses(self):
        with self.assertRaises(ConfigurationError):
            parse_layout(["((sources / extras) | changes", "input"])

    def test_prune_layout_drops_hidden_leaves(self):
        layout = parse_layout(["sources | (flashcards / logs)", "input"])
        pruned = prune_layout(layout, frozenset({"logs"}))
        self.assertIsNotNone(pruned)
        assert pruned is not None
        self.assertNotIn("logs", layout_panes(pruned))
        self.assertIn("flashcards", layout_panes(pruned))

    def test_layout_tree_nodes_are_structural(self):
        layout = parse_layout(["sources | changes", "input"])
        self.assertIsInstance(layout, Split)
        assert isinstance(layout, Split)
        self.assertEqual(layout.axis, "rows")
        first, _weight = layout.children[0]
        self.assertIsInstance(first, Split)
        assert isinstance(first, Split)
        self.assertEqual(first.axis, "cols")
        self.assertEqual(first.children[0][0], Leaf("sources"))

    def test_default_navigation_is_spatial(self):
        self.assertEqual(neighbor(DEFAULT_LAYOUT, "changes", "l", 30, 120), "flashcards")
        self.assertEqual(neighbor(DEFAULT_LAYOUT, "flashcards", "h", 30, 120), "changes")
        self.assertEqual(neighbor(DEFAULT_LAYOUT, "changes", "j", 30, 120), "input")
        self.assertEqual(neighbor(DEFAULT_LAYOUT, "flashcards", "j", 30, 120), "logs")
        self.assertEqual(neighbor(DEFAULT_LAYOUT, "input", "k", 30, 120), "changes")
        self.assertIsNone(neighbor(DEFAULT_LAYOUT, "input", "h", 30, 120))

    def test_custom_layout_changes_navigation_directions(self):
        layout = parse_layout(["input", "sources | changes | flashcards"])
        self.assertEqual(neighbor(layout, "input", "j", 30, 120), "changes")
        self.assertEqual(neighbor(layout, "changes", "l", 30, 120), "flashcards")
        self.assertEqual(neighbor(layout, "changes", "k", 30, 120), "input")
        self.assertIsNone(neighbor(layout, "changes", "j", 30, 120))

    def test_the_anchor_decides_which_box_is_straight_up(self):
        rects = pane_rects(DEFAULT_LAYOUT, 30, 120)
        # The input line spans the whole width, so the remembered horizontal
        # position decides which top box is above it.
        self.assertEqual(neighbor_rects(rects, "input", "k", point=(2.0, 28.0)), "extras")
        self.assertEqual(neighbor_rects(rects, "input", "k", point=(118.0, 28.0)), "logs")
        # Leaving the right edge and turning up lands on the right box.
        input_rect = rects["input"]
        anchor = edge_point(input_rect, "l", (input_rect.center_x, input_rect.center_y))
        self.assertEqual(neighbor_rects(rects, "input", "k", point=anchor), "logs")

    def test_right_then_up_uses_the_anchor_in_a_custom_layout(self):
        layout = parse_layout(["sources | flashcards", "input | changes"])
        rects = pane_rects(layout, 30, 120)
        self.assertEqual(neighbor_rects(rects, "input", "l"), "changes")
        input_rect = rects["input"]
        anchor = edge_point(input_rect, "l", (input_rect.center_x, input_rect.center_y))
        self.assertEqual(neighbor_rects(rects, "changes", "k", point=anchor), "sources")

    def test_clamp_point_keeps_the_anchor_inside_a_pane(self):
        rect = pane_rects(DEFAULT_LAYOUT, 30, 120)["flashcards"]
        self.assertEqual(clamp_point((0.0, 0.0), rect), (float(rect.x), float(rect.y)))
        self.assertEqual(
            clamp_point((10_000.0, 10_000.0), rect),
            (float(rect.right - 1), float(rect.bottom - 1)),
        )

    def test_layout_rejects_unknown_duplicate_and_non_placeable_panes(self):
        with self.assertRaises(ConfigurationError):
            parse_layout(["changes | mystery", "input"])
        with self.assertRaises(ConfigurationError):
            parse_layout(["changes | changes", "input"])
        # Extras and logs are ordinary placeable panes.
        self.assertEqual(
            layout_panes(parse_layout(["changes | extras | logs", "input"])),
            ("changes", "extras", "logs", "input"),
        )

    def test_layout_must_keep_the_input_line(self):
        with self.assertRaises(ConfigurationError):
            parse_layout(["sources | changes | flashcards"])

    def test_a_registered_pane_may_be_left_out_of_the_layout(self):
        layout = parse_layout(["sources | changes", "input"])
        self.assertEqual(layout_panes(layout), ("sources", "changes", "input"))

    def test_tui_focus_follows_a_custom_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vault = root / "vault"
            vault.mkdir()
            config = Config(
                notes_root=vault,
                artifacts_dir=root / "home",
                tui_layout=parse_layout(["input", "sources | changes | flashcards"]),
            )
            app = TuiApp(config=config)
            app.scr = FakeScreen()
            app.focus = "input"
            app._handle_normal_key("j")
            self.assertEqual(app.focus, "changes")
            app._handle_normal_key("l")
            self.assertEqual(app.focus, "flashcards")
            app._handle_normal_key("k")
            self.assertEqual(app.focus, "input")


class ContextSourceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault_dir = self.root / "vault"
        self.vault_dir.mkdir()
        self.config = Config(notes_root=self.vault_dir, artifacts_dir=self.root / "home")

    def tearDown(self):
        self._tmp.cleanup()

    def test_source_text_drives_the_related_notes(self):
        note = self.vault_dir / "Heuristics.md"
        note.write_text("## Overview\n\nHeuristics estimate cost.\n", encoding="utf-8")
        context = build_context(
            Vault(self.vault_dir),
            "Course",
            "",
            "Heuristics estimate cost in search.",
            self.config,
        )
        self.assertIn("heuristics", context.keywords)
        self.assertIn("Heuristics.md", context.related_block())


class FakeScreen:
    def __init__(self, rows: int = 30, cols: int = 120):
        self.rows = rows
        self.cols = cols
        self.cells: dict[tuple[int, int], str] = {}
        self.attrs: dict[tuple[int, int], int] = {}
        self.last_move: tuple[int, int] | None = None

    def getmaxyx(self):
        return self.rows, self.cols

    def erase(self):
        self.cells.clear()
        self.attrs.clear()

    def addnstr(self, y, x, text, n, attr=0):
        for offset, char in enumerate(str(text)[:n]):
            if 0 <= y < self.rows and 0 <= x + offset < self.cols:
                self.cells[(y, x + offset)] = char
                self.attrs[(y, x + offset)] = attr

    def move(self, y, x):
        self.last_move = (y, x)
        return None

    def refresh(self):
        return None


class FakeKeyScreen(FakeScreen):
    """A screen whose key reader serves a fixed script of characters."""

    def __init__(self, keys: list):
        super().__init__()
        self.keys = list(keys)
        self.timeout_ms = 0

    def get_wch(self):
        if not self.keys:
            raise curses.error("no more keys")
        return self.keys.pop(0)

    def timeout(self, milliseconds):
        self.timeout_ms = milliseconds


class AppRenderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        (self.root / "nested").mkdir()
        (self.root / "nested" / "slides.pdf").write_bytes(b"x")
        (self.root / "nested" / "slides-2.pdf").write_bytes(b"x")
        self.config = Config(
            notes_root=self.vault,
            artifacts_dir=self.root / "home",
            search_paths=(self.root,),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_render_with_changes_cards_and_completion(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "sli"
        app.cursor = 3
        app.diff_entries = [
            DiffEntry(
                path="New.md",
                action="create",
                added=1,
                diff="--- a/New.md\n+++ b/New.md\n@@ -0,0 +1 @@\n+hello",
                group="created",
                expanded=True,
            )
        ]
        app.card_groups = [
            CardGroup(
                path="New.md",
                cards=[Flashcard(front="What is X?", back="Y.")],
                requested=1,
                written=1,
                writer="internal",
                expanded=True,
            )
        ]
        app.current_stage = "compose"
        app.running = True
        app._draw()
        app._start_completion()
        self.assertIsNotNone(app.completion)
        app._draw()
        self.assertTrue(
            any(
                "Changes" in "".join(app.scr.cells.get((y, x), "") for x in range(app.scr.cols))
                for y in range(app.scr.rows)
            )
        )
        app._handle_normal_key("j")
        app._handle_normal_key("l")
        app._handle_normal_key("j")
        app._draw()
        app._open_settings()
        app._draw()
        app._handle_settings_key("\x1b")
        self.assertIsNone(app.settings)

    def test_input_cursor_uses_the_prompt_offset(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "abc"
        app.cursor = 1
        app.mode = "insert"
        app.focus = "input"
        app._sync_editor_from_fields()
        app._place_cursor(10, 1, 40, 3)
        self.assertEqual(app.scr.last_move, (11, 1 + 4 + 1))

    def test_normal_mode_draws_a_block_cursor_on_the_character(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "abc"
        app.cursor = 1
        app.mode = MODE_NORMAL
        app.focus = "input"
        app._sync_editor_from_fields()
        app._draw()
        self.assertEqual(app.scr.cells[(app.scr.rows - 2, 1 + 4 + 1)], "b")

    def test_settings_editor_renders_and_commits(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._open_settings()
        app._handle_settings_key("e")
        state = app.settings
        assert state is not None
        self.assertIsNotNone(state.editor)
        app._draw()
        app._handle_settings_key("\x1b")
        self.assertIsNone(state.editor)
        app._handle_settings_key("\x1b")
        self.assertIsNone(app.settings)

    def test_bracketed_paste_markers_become_one_paste_event(self):
        app = TuiApp(config=self.config)
        app.scr = FakeKeyScreen(list("\x1b[200~nested/slides.pdf\nnested/slides-2.pdf\x1b[201~"))
        event = app._read_key()
        self.assertIsInstance(event, PasteEvent)
        app._handle_key(event)
        self.assertEqual(
            app.source_paths,
            [self.root / "nested" / "slides.pdf", self.root / "nested" / "slides-2.pdf"],
        )

    def test_a_lone_escape_is_not_swallowed_by_paste_detection(self):
        app = TuiApp(config=self.config)
        app.scr = FakeKeyScreen(["\x1b"])
        self.assertEqual(app._read_key(), "\x1b")

    def test_paste_adds_every_existing_path_without_enter(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        first = self.root / "nested" / "slides.pdf"
        second = self.root / "nested" / "slides-2.pdf"
        app._handle_key(PasteEvent(text=f"{first}\n{second}\n"))
        self.assertEqual(app.source_paths, [first, second])
        self.assertEqual(app.input_text, "")
        self.assertIsNone(app.completion)
        self.assertIn("added 2 source(s)", app.status)
        self.assertNotIn("/run", app.status)

    def test_paste_works_when_the_input_box_is_not_focused(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "sources"
        app.active = None
        source = self.root / "nested" / "slides.pdf"
        app._handle_key(PasteEvent(text=str(source)))
        self.assertEqual(app.source_paths, [source])

    def test_paste_of_a_long_single_line_list_does_not_crash(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        source = self.root / "nested" / "slides.pdf"
        app._handle_key(PasteEvent(text=" ".join([str(source)] * 100)))
        self.assertEqual(app.source_paths, [source])
        self.assertEqual(app.status, "added 1 source(s)")

    def test_paste_reports_missing_files_in_the_header_and_log(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        found = self.root / "nested" / "slides.pdf"
        missing_a = self.root / "missing-a.pdf"
        missing_b = self.root / "missing-b.pdf"
        app._handle_key(PasteEvent(text=f"{found}\n{missing_a}\n{missing_b}"))
        self.assertEqual(app.source_paths, [found])
        self.assertEqual(
            app.status,
            "2 source(s) not found (open logs for details)",
        )
        self.assertIn(f"[warn] file not found: {missing_a}", app.log_lines)
        self.assertIn(f"[warn] file not found: {missing_b}", app.log_lines)

    def test_paste_does_not_trigger_completion(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "sli"
        app.cursor = 3
        app._sync_editor_from_fields()
        source = self.root / "nested" / "slides.pdf"
        with mock.patch.object(app, "refresh_completion") as refresh:
            app._handle_key(PasteEvent(text=str(source)))
        refresh.assert_not_called()
        self.assertEqual(app.input_text, "sli")

    def test_paste_is_ignored_while_a_modal_is_open(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.modal = PromptModal(kind="commands", payload={}, box={}, event=threading.Event())
        source = self.root / "nested" / "slides.pdf"
        app._handle_key(PasteEvent(text=str(source)))
        self.assertEqual(app.source_paths, [])

    def test_input_line_starts_in_normal_mode(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        self.assertEqual(app.mode, MODE_NORMAL)
        self.assertEqual(app.editor.mode, MODE_NORMAL)
        app.focus = "input"
        app._handle_input_key("Z")
        self.assertEqual(app.input_text, "")
        app._handle_input_key("i")
        app._handle_input_key("Z")
        self.assertEqual(app.input_text, "Z")

    def test_input_keys_drive_the_vim_editor(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "abc"
        app.cursor = 0
        app.mode = MODE_NORMAL
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key("x")
        self.assertEqual(app.input_text, "bc")
        self.assertEqual(app.cursor, 0)
        app._handle_input_key("i")
        app._handle_input_key("Z")
        self.assertEqual(app.input_text, "Zbc")
        app._handle_input_key("\x1b")
        self.assertEqual(app.mode, MODE_NORMAL)

    def test_enter_adds_one_source_to_the_sidebar_and_clears_the_input(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        source = self.root / "nested" / "slides.pdf"
        app.input_text = str(source)
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key("\n")
        self.assertEqual(app.source_paths, [source])
        self.assertEqual(app.input_text, "")
        self.assertEqual(app.cursor, 0)
        self.assertIn("added slides.pdf", app.status)
        self.assertIn("/run", app.status)

    def test_typing_a_source_prefix_opens_the_popup_without_tab(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app.active = "input"
        app.mode = MODE_INSERT
        app._sync_editor_from_fields()
        app._handle_input_key("s")
        app._handle_input_key("l")
        app._handle_input_key("i")
        popup = app.completion
        assert popup is not None
        self.assertFalse(popup.is_command)
        self.assertEqual(app.input_text, "sli")
        self.assertEqual(
            {Path(match.path).name for match in popup.matches},
            {"slides.pdf", "slides-2.pdf"},
        )

    def test_enter_on_a_source_match_adds_it_and_clears_the_line(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app.active = "input"
        app.mode = MODE_INSERT
        app._sync_editor_from_fields()
        for char in "sli":
            app._handle_input_key(char)
        popup = app.completion
        assert popup is not None
        selected = Path(popup.matches[popup.index].path)
        app._handle_input_key("\n")
        self.assertEqual(app.source_paths, [selected])
        self.assertEqual(app.input_text, "")
        self.assertIsNone(app.completion)

    def test_completion_index_is_reused_until_the_input_is_cleared(self):
        app = TuiApp(config=self.config)
        app.input_text = "s"
        app.cursor = 1
        app.refresh_completion()
        index = app._completion_index
        self.assertIsNotNone(index)

        app.input_text = "sl"
        app.cursor = 2
        app.refresh_completion()
        self.assertIs(app._completion_index, index)

        app.input_text = ""
        app.cursor = 0
        app.refresh_completion()
        self.assertIsNone(app._completion_index)

    def test_completion_index_is_rebuilt_when_the_roots_change(self):
        app = TuiApp(config=self.config)
        app.input_text = "s"
        app.cursor = 1
        app.refresh_completion()
        index = app._completion_index
        self.assertIsNotNone(index)

        app.search_roots = [self.vault]
        app.refresh_completion()

        self.assertIsNot(app._completion_index, index)

    def test_c_opens_the_list_of_available_commands(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._handle_key("c")
        modal = app.modal
        assert modal is not None
        self.assertEqual(modal.kind, "commands")
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("/run", text)
        self.assertIn("start the pipeline", text)
        app._handle_modal_key("\x1b")
        self.assertIsNone(app.modal)

    def test_the_header_lists_c_commands_before_settings(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.status = ""
        app._draw()
        header = "".join(app.scr.cells.get((0, x), " ") for x in range(app.scr.cols))
        self.assertIn("c commands", header)
        self.assertLess(header.index("c commands"), header.index("s settings"))
        self.assertLess(header.index("Ctrl-L logs"), header.index("f full"))

    def test_input_requires_enter_before_accepting_text(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app._handle_key("a")
        self.assertEqual(app.input_text, "")
        self.assertIsNone(app.active)
        app._handle_key("\n")
        self.assertEqual(app.active, "input")
        self.assertEqual(app.mode, MODE_INSERT)
        app._handle_key("a")
        self.assertEqual(app.input_text, "a")

    def test_submitting_a_source_returns_the_input_to_normal_mode(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        source = self.root / "nested" / "slides.pdf"
        app._handle_key("\n")
        app.input_text = str(source)
        app.cursor = len(app.input_text)
        app._sync_editor_from_fields()
        app._handle_key("\n")
        self.assertEqual(app.source_paths, [source])
        self.assertEqual(app.input_text, "")
        self.assertEqual(app.mode, MODE_NORMAL)
        self.assertEqual(app.editor.mode, MODE_NORMAL)
        self.assertIsNone(app.active)

    def test_escape_from_insert_keeps_the_input_box_selected(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app._handle_key("\n")
        self.assertEqual(app.active, "input")
        self.assertEqual(app.mode, MODE_INSERT)
        app._handle_key("\x1b")
        self.assertEqual(app.active, "input")
        self.assertEqual(app.mode, MODE_NORMAL)
        self.assertEqual(app.editor.mode, MODE_NORMAL)

    def test_normal_mode_keeps_editing_the_selected_input_line(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app._handle_key("\n")
        app.editor.set_text("abc", cursor=3)
        app._pull_editor()
        app._handle_key("\x1b")
        self.assertEqual(app.active, "input")
        app._handle_key("x")
        self.assertEqual(app.input_text, "ab")
        self.assertEqual(app.mode, MODE_NORMAL)

    def test_second_escape_leaves_the_selected_input_box(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app._handle_key("\n")
        app._handle_key("\x1b")
        self.assertEqual(app.active, "input")
        app._handle_key("\x1b")
        self.assertIsNone(app.active)
        self.assertEqual(app.focus, "input")
        self.assertEqual(app.mode, MODE_NORMAL)

    def test_i_from_another_selected_box_selects_the_input_in_insert_mode(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "sources"
        app.active = "sources"
        app._handle_key("i")
        self.assertEqual(app.focus, "input")
        self.assertEqual(app.active, "input")
        self.assertEqual(app.mode, MODE_INSERT)
        app._handle_key("a")
        self.assertEqual(app.input_text, "a")

    def test_enter_rejects_several_sources_at_once(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "nested/slides.pdf nested/slides-2.pdf"
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key("\n")
        self.assertEqual(app.source_paths, [])
        self.assertIn("one source at a time", app.status)
        self.assertEqual(app.input_text, "nested/slides.pdf nested/slides-2.pdf")

    def test_slash_run_starts_the_job_with_the_collected_sources(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        source = self.root / "nested" / "slides.pdf"
        app.source_paths = [source]
        app.input_text = "/run"
        app.focus = "input"
        app._sync_editor_from_fields()
        with mock.patch.object(app, "_start_job") as start:
            app._handle_input_key("\n")
        start.assert_called_once()
        self.assertEqual(app.input_text, "")

    def test_slash_new_clears_the_current_session(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        source = self.root / "nested" / "slides.pdf"
        app.source_paths = [source]
        app.moc_hint = source
        app.diff_entries = [DiffEntry(path="New.md", action="create")]
        app.card_groups = [CardGroup(path="New.md")]
        app.asides = [Aside(text="exam on Friday")]
        app.current_job_id = "job-1"
        app.current_stage = "compose"
        app.current_group = "Topic"
        app.group_position = "1/2"
        app.stage_status = {"compose": "done"}
        app.log_lines = ["old log line"]
        app.log_scroll = 3
        app.input_text = "/new"
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key("\n")
        self.assertEqual(app.source_paths, [])
        self.assertIsNone(app.moc_hint)
        self.assertEqual(app.diff_entries, [])
        self.assertEqual(app.card_groups, [])
        self.assertEqual(app.asides, [])
        self.assertIsNone(app.current_job_id)
        self.assertEqual(app.current_stage, "")
        self.assertEqual(app.current_group, "")
        self.assertEqual(app.stage_status, {})
        self.assertEqual(app.log_lines, [])
        self.assertEqual(app.log_scroll, 0)
        self.assertEqual(app.input_text, "")
        self.assertIn("new session", app.status)

    def test_slash_new_is_refused_while_a_job_runs(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        source = self.root / "nested" / "slides.pdf"
        app.source_paths = [source]
        app.current_job_id = "job-1"
        app.running = True
        app.input_text = "/new"
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key("\n")
        self.assertEqual(app.source_paths, [source])
        self.assertEqual(app.current_job_id, "job-1")
        self.assertIn("already running", app.status)

    def test_unknown_slash_command_is_reported(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "/quit"
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key("\n")
        self.assertIn("unknown command", app.status)
        self.assertEqual(app.input_text, "/quit")

    def test_d_removes_the_selected_source(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        first = self.root / "nested" / "slides.pdf"
        second = self.root / "nested" / "slides-2.pdf"
        app.source_paths = [first, second]
        app.source_index = 1
        app.focus = "sources"
        app._handle_normal_key("d")
        self.assertEqual(app.source_paths, [first])
        self.assertEqual(app.source_index, 0)
        self.assertIn("removed slides-2.pdf", app.status)

    def test_sources_panel_lists_the_collected_paths(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        source = self.root / "nested" / "slides.pdf"
        app.source_paths = [source]
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("Sources · 1", text)
        self.assertIn("slides.pdf", text)

    def test_sources_group_files_under_their_parent_directory(self):
        app = TuiApp(config=self.config)
        first = self.root / "nested" / "slides.pdf"
        second = self.root / "nested" / "slides-2.pdf"
        third = self.root / "downloads" / "lecture.mp4"
        app.source_paths = [first, second, third]
        app.source_index = 1
        rows, selected = app.panes["sources"].rows(app)
        self.assertEqual(rows[0][0], str(first.parent))
        self.assertIn("slides.pdf", rows[1][0])
        self.assertIn("slides-2.pdf", rows[2][0])
        self.assertEqual(rows[3][0], str(third.parent))
        self.assertIn("lecture.mp4", rows[4][0])
        self.assertEqual(selected, 2)

    def test_the_empty_input_shows_a_dim_placeholder(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("add sources and type /run", text)

    def test_the_header_always_shows_the_product_name(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.status = ""
        app._draw()
        header = "".join(app.scr.cells.get((0, x), " ") for x in range(app.scr.cols))
        self.assertIn("scriptorium", header)

    def test_escape_clears_transient_status(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.status = "no pane to the left"
        app.input_text = "abc"
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_key("\x1b")
        self.assertEqual(app.status, "")
        self.assertEqual(app.mode, MODE_NORMAL)

    def test_vim_mode_ignores_arrow_movements(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "abc"
        app.cursor = 1
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key(curses.KEY_LEFT)
        self.assertEqual(app.cursor, 1)
        app.diff_entries = [
            DiffEntry(path="A.md", action="create", added=1, group="created"),
            DiffEntry(path="B.md", action="create", added=1, group="created"),
        ]
        app.focus = "changes"
        app._handle_normal_key(curses.KEY_DOWN)
        self.assertEqual(app.diff_index, 0)

    def test_standard_mode_types_without_modes_and_uses_arrows(self):
        config = Config(
            notes_root=self.vault,
            artifacts_dir=self.root / "home",
            movement="standard",
        )
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        self.assertEqual(app.mode, MODE_INSERT)
        app.focus = "input"
        app._handle_input_key("h")
        self.assertEqual(app.input_text, "h")
        app._handle_input_key(curses.KEY_LEFT)
        self.assertEqual(app.cursor, 0)
        app._handle_input_key("\x1b")
        self.assertEqual(app.mode, MODE_INSERT)
        app._handle_input_key(curses.KEY_UP)
        self.assertNotEqual(app.focus, "input")
        self.assertEqual(app.mode, MODE_INSERT)

    def test_enter_selects_a_box_and_esc_returns_to_focus_navigation(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.diff_entries = [
            DiffEntry(path="A.md", action="create", added=1, group="created"),
            DiffEntry(path="B.md", action="create", added=1, group="created"),
        ]
        app.focus = "changes"
        app._handle_normal_key("\n")
        self.assertEqual(app.active, "changes")
        app._handle_normal_key("j")
        self.assertEqual(app.diff_index, 1)
        app._handle_key("\x1b")
        self.assertIsNone(app.active)
        self.assertEqual(app.diff_index, 1)

    def test_enter_inside_the_selected_box_toggles_the_entry(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.diff_entries = [DiffEntry(path="A.md", action="create", added=1, group="created")]
        app.focus = "changes"
        app._handle_normal_key("\n")
        self.assertFalse(app.diff_entries[0].expanded)
        app._handle_normal_key("\n")
        self.assertTrue(app.diff_entries[0].expanded)

    def test_merge_review_is_rendered_in_the_changes_box(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        review = PromptModal(
            kind="merge",
            payload={
                "note": PlannedNote(action="merge", path="Old.md", title="Old"),
                "preview": "+new",
            },
            box={},
            event=threading.Event(),
        )
        app.prompt_requests.put(review)
        app._maybe_accept_prompt()
        self.assertIsNone(app.modal)
        self.assertIs(app.merge_review, review)
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("Review a merge", text)
        self.assertIn("y apply", text)
        app._handle_key("y")
        self.assertEqual(review.box["response"], "apply")
        self.assertIsNone(app.merge_review)

    def test_footer_hints_are_gone_and_add_precedes_remove(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.log_lines = ["line"]
        app.set_pane_shown("logs", True)
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertNotIn("Enter/Space expand", text)
        self.assertNotIn("j/k panes", text)
        self.assertIn("i add · d remove", text)
        self.assertLess(text.index("i add"), text.index("d remove"))
        self.assertEqual(text.count("f full"), 1)

    def test_vim_mode_badge_shows_on_input_box(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.mode = MODE_NORMAL
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        lines = text.splitlines()
        self.assertNotIn("NORMAL", lines[0])
        self.assertTrue(any("Input" in line and "NORMAL" in line for line in lines))
        normal_line = next(line for line in lines if "NORMAL" in line and "Input" in line)
        self.assertNotIn("[", normal_line)
        self.assertNotIn("]", normal_line)
        self.assertLess(normal_line.index("Input"), normal_line.index("NORMAL"))
        app.mode = MODE_INSERT
        app._draw()
        text = "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        lines = text.splitlines()
        self.assertNotIn("INSERT", lines[0])
        self.assertTrue(any("Input" in line and "INSERT" in line for line in lines))
        insert_line = next(line for line in lines if "INSERT" in line and "Input" in line)
        self.assertNotIn("[", insert_line)
        self.assertNotIn("]", insert_line)
        self.assertLess(insert_line.index("Input"), insert_line.index("INSERT"))

    def test_vim_mode_badge_has_padding_on_both_sides(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.mode = MODE_NORMAL
        app._draw()
        row = app._layout_rects()["input"].y
        line = "".join(app.scr.cells.get((row, x), " ") for x in range(app.scr.cols))
        # One blank cell inside each end of the filled chip, painted with the
        # chip background so the label never touches the chip edge.
        self.assertIn(" NORMAL ", line)
        start = line.index("NORMAL")
        chip_attr = app.scr.attrs[(row, start)]
        self.assertEqual(app.scr.cells[(row, start - 1)], " ")
        self.assertEqual(app.scr.cells[(row, start + len("NORMAL"))], " ")
        self.assertEqual(app.scr.attrs[(row, start - 1)], chip_attr)
        self.assertEqual(app.scr.attrs[(row, start + len("NORMAL"))], chip_attr)
        app.mode = MODE_INSERT
        app._draw()
        line = "".join(app.scr.cells.get((row, x), " ") for x in range(app.scr.cols))
        self.assertIn(" INSERT ", line)

    def test_input_box_border_uses_focus_and_active_colours(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._colors[STYLE_FOCUS] = 0x100
        app._colors[STYLE_ACTIVE_BORDER] = 0x200
        rect = app._layout_rects()["input"]
        app.focus = "input"
        app.active = None
        app._draw()
        self.assertEqual(app.scr.attrs[(rect.y, rect.x)], 0x100 | curses.A_BOLD)
        app.active = "input"
        app._draw()
        self.assertEqual(app.scr.attrs[(rect.y, rect.x)], 0x200)


class LogWindowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.config = Config(notes_root=self.vault, artifacts_dir=self.root / "home")

    def tearDown(self):
        self._tmp.cleanup()

    def _screen_text(self, app):
        return "\n".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )

    def test_logs_stay_hidden_until_the_window_is_opened(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._on_log("[warn] transformers loading a model")
        app._draw()
        self.assertNotIn("transformers loading", self._screen_text(app))
        app._handle_normal_key("\x0c")
        app._draw()
        self.assertIn("Logs", self._screen_text(app))
        self.assertIn("transformers loading", self._screen_text(app))
        app._handle_normal_key("\x0c")
        self.assertFalse(app.pane_shown("logs"))

    def test_opening_logs_keeps_focus_and_movement_selects_them(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "flashcards"
        app._handle_normal_key("\x0c")
        self.assertTrue(app.pane_shown("logs"))
        self.assertEqual(app.focus, "flashcards")
        app._handle_normal_key("j")
        self.assertEqual(app.focus, "logs")
        app._handle_normal_key("\x0c")
        self.assertFalse(app.pane_shown("logs"))
        self.assertEqual(app.focus, "flashcards")

    def test_the_open_log_window_can_be_left_upwards_into_the_cards(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "flashcards"
        app._handle_normal_key("\x0c")
        app._handle_normal_key("j")
        self.assertEqual(app.focus, "logs")
        app._handle_normal_key("k")
        self.assertEqual(app.focus, "flashcards")
        app._handle_normal_key("j")
        self.assertEqual(app.focus, "logs")
        app._handle_normal_key("j")
        self.assertEqual(app.focus, "input")

    def test_the_log_viewer_scrolls_with_the_arrow_keys_in_vim_mode(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.log_lines = [f"line {index}" for index in range(200)]
        app.set_pane_shown("logs", True)
        app.focus = "logs"
        app.log_scroll = 10
        app._handle_normal_key(curses.KEY_DOWN)
        self.assertEqual(app.log_scroll, 11)
        app._handle_normal_key(curses.KEY_UP)
        self.assertEqual(app.log_scroll, 10)
        app._handle_normal_key(curses.KEY_NPAGE)
        self.assertGreater(app.log_scroll, 10)
        app._handle_normal_key(curses.KEY_PPAGE)
        self.assertEqual(app.log_scroll, 10)

    def test_a_direction_without_a_pane_still_moves_the_anchor(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app._move_focus("l")
        self.assertEqual(app.focus, "input")
        self.assertEqual(app.status, "ready")
        app._move_focus("k")
        self.assertEqual(app.focus, "flashcards")

    def test_standard_mode_arrows_leave_the_log_viewer(self):
        config = Config(
            notes_root=self.vault,
            artifacts_dir=self.root / "home",
            movement="standard",
        )
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        app.log_lines = [f"line {index}" for index in range(200)]
        app.set_pane_shown("logs", True)
        app.focus = "logs"
        app.log_scroll = 10
        app._handle_normal_key(curses.KEY_UP)
        self.assertEqual(app.focus, "flashcards")
        self.assertEqual(app.log_scroll, 10)

    def test_ctrl_l_also_toggles_logs_from_insert_mode(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app.editor.enter_insert(0)
        app._handle_key("\x0c")
        self.assertTrue(app.pane_shown("logs"))
        self.assertEqual(app.focus, "input")

    def test_status_is_shown_in_the_header_not_in_the_log_window(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.status = "ready"
        app._draw()
        self.assertIn("ready", self._screen_text(app).splitlines()[0])
        app._on_log("transformers is downloading a model")
        app._handle_normal_key("\x0c")
        app._draw()
        lines = self._screen_text(app).splitlines()
        logs_row = next(index for index, line in enumerate(lines) if "Logs ·" in line)
        self.assertIn("ready", lines[0])
        self.assertNotIn("ready", "\n".join(lines[logs_row:]))

    def test_the_open_log_window_sits_below_cards_and_above_the_input(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._on_log("transformers is downloading a model")
        app._draw()
        before = self._screen_text(app).splitlines()
        cards_row = next(index for index, line in enumerate(before) if "Flashcards" in line)
        app._handle_normal_key("\x0c")
        app._draw()
        after = self._screen_text(app).splitlines()
        logs_row = next(index for index, line in enumerate(after) if "Logs ·" in line)
        input_row = next(index for index, line in enumerate(after) if "›" in line)
        self.assertGreater(logs_row, cards_row)
        self.assertLess(logs_row, input_row)
        extras_row = next(index for index, line in enumerate(after) if "Extras ·" in line)
        # Extras takes the lower half of the left column, beside the log window.
        self.assertLessEqual(extras_row, logs_row)
        self.assertLess(extras_row, input_row)

    def test_the_settings_window_is_opaque_and_restores_what_it_covers(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.diff_entries = [
            DiffEntry(
                path="X" * 60,
                action="create",
                added=1,
                group="created",
            )
            for _ in range(15)
        ]
        app._draw()
        before = {cell: char for cell, char in app.scr.cells.items() if cell[0] > 0}
        app._open_settings()
        app._draw()
        text = self._screen_text(app)
        settings_row = next(
            index for index, line in enumerate(text.splitlines()) if "Settings" in line
        )
        settings_col = text.splitlines()[settings_row].index("Settings") - 2
        # The underlying long diff title no longer shows through the window.
        covered = (settings_row + 2, settings_col + 5)
        self.assertNotEqual(app.scr.cells[covered], before.get(covered, " "))
        app._handle_settings_key("\x1b")
        app._draw()
        after = {cell: char for cell, char in app.scr.cells.items() if cell[0] > 0}
        self.assertEqual(after, before)

    def test_a_long_log_line_soft_wraps_instead_of_clipping(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen(rows=30, cols=60)
        app._on_log("alpha " * 12 + "OMEGA")
        app._handle_normal_key("\x0c")
        app._draw()
        # Clipped output would drop the tail of the line off the right edge.
        self.assertIn("OMEGA", self._screen_text(app))

    def test_f_expands_the_focused_pane_and_keeps_the_input(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "changes"
        before = app._layout_rects()
        app._handle_normal_key("f")
        self.assertEqual(app.fullscreen, "changes")
        rects = app._layout_rects()
        self.assertEqual(rects["changes"], content_area(before))
        self.assertEqual(rects["input"], before["input"])
        self.assertNotIn("sources", rects)
        app._draw()
        text = self._screen_text(app).splitlines()
        self.assertTrue(any("Changes" in line for line in text))
        self.assertTrue(any("›" in line for line in text))
        app._handle_normal_key("f")
        self.assertIsNone(app.fullscreen)
        self.assertEqual(app._layout_rects(), before)

    def test_f_restores_the_layout_when_the_focus_reaches_the_input(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "sources"
        app._handle_normal_key("f")
        self.assertEqual(app.fullscreen, "sources")
        app.focus = "input"
        app._handle_normal_key("f")
        self.assertIsNone(app.fullscreen)
        app._handle_normal_key("f")
        self.assertIsNone(app.fullscreen)

    def test_leaving_the_expanded_pane_restores_the_layout(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "changes"
        before = app._layout_rects()
        app._handle_normal_key("f")
        app._move_focus("j")
        self.assertEqual(app.focus, "input")
        self.assertIsNone(app.fullscreen)
        self.assertEqual(app._layout_rects(), before)

    def test_f_expands_the_extras_pane_too(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "extras"
        before = app._layout_rects()
        app._handle_normal_key("f")
        self.assertEqual(app.fullscreen, "extras")
        rects = app._layout_rects()
        self.assertEqual(rects["extras"], content_area(before))
        self.assertEqual(rects["input"], before["input"])
        app._draw()
        self.assertTrue(any("Extras ·" in line for line in self._screen_text(app).splitlines()))
        app._handle_normal_key("f")
        self.assertIsNone(app.fullscreen)

    def test_f_expands_the_log_viewer_over_every_other_box(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "flashcards"
        app._handle_normal_key("\x0c")
        app._handle_normal_key("j")
        self.assertEqual(app.focus, "logs")
        full = content_area(app._layout_rects())
        app._handle_normal_key("f")
        self.assertEqual(app.fullscreen, "logs")
        rects = app._layout_rects()
        self.assertEqual(rects["logs"], full)
        self.assertNotIn("flashcards", rects)
        self.assertNotIn("changes", rects)
        app._handle_normal_key("f")
        self.assertIsNone(app.fullscreen)

    def test_log_stream_queues_complete_lines(self):
        from scriptorium.tui import _LogStream

        app = TuiApp(config=self.config)
        stream = _LogStream(app, mock.Mock())
        stream.write("first line\nsecond")
        self.assertEqual(app.events.get_nowait(), ("log", "first line"))
        stream.write(" line\n")
        self.assertEqual(app.events.get_nowait(), ("log", "second line"))

    def test_a_running_job_shows_a_spinner_in_the_header(self):
        from scriptorium.tui import SPINNER_FRAMES

        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.running = True
        app.status = "compose: Compose"
        app._draw()
        header = "".join(app.scr.cells.get((0, x), " ") for x in range(app.scr.cols))
        self.assertIn("scriptorium · ", header)
        self.assertTrue(any(frame in header for frame in SPINNER_FRAMES))
        self.assertIn("compose: Compose", header)

    def test_an_idle_job_has_no_spinner(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.status = "ready"
        app._draw()
        self.assertEqual(app.scr.cells[(0, 1)], "s")
        header = "".join(app.scr.cells.get((0, x), " ") for x in range(app.scr.cols))
        self.assertIn("scriptorium · ready", header)

    def test_a_failure_goes_to_the_log_and_leaves_the_status_readable(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.running = True
        app.events.put(("failed", "first line\nsecond line"))
        app._drain_events()
        self.assertFalse(app.running)
        self.assertNotIn("first line", app.status)
        self.assertIn("Ctrl-L", app.status)
        self.assertIn("[error] first line", app.log_lines)
        self.assertIn("[error] second line", app.log_lines)
        app._draw()
        header = "".join(app.scr.cells.get((0, x), " ") for x in range(app.scr.cols))
        self.assertNotIn("first line", header)

    def test_header_shows_the_ctrl_l_log_hint_and_settings(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._draw()
        header = "".join(app.scr.cells.get((0, x), " ") for x in range(app.scr.cols))
        self.assertIn("Ctrl-L logs", header)
        self.assertIn("s settings", header)
        self.assertIn("scriptorium", header)
        self.assertNotIn("focus:", header)

    def test_plain_l_no_longer_opens_the_log_window(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._handle_normal_key("l")
        self.assertFalse(app.pane_shown("logs"))

    def test_input_box_only_shows_input_and_no_footer_hints(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app._draw()
        text = self._screen_text(app)
        self.assertIn("Input", text)
        self.assertNotIn("Esc normal", text)
        self.assertNotIn("Enter run", text)
        self.assertNotIn("s settings", text.split("Input", 1)[1])


class MocModalTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = Config(
            notes_root=self.root,
            artifacts_dir=self.root / "home",
            search_paths=(self.root,),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def modal(self, matches=None) -> PromptModal:
        return PromptModal(
            kind="moc",
            payload={
                "course": "Test Course",
                "matches": matches or [],
                "default_title": "Test Course MOC",
            },
            box={},
            event=threading.Event(),
        )

    def test_enter_on_the_create_row_opens_the_name_editor(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("\n")
        self.assertEqual(modal.kind, "moc_name")
        self.assertIsNotNone(modal.editor)
        assert modal.editor is not None
        self.assertEqual(modal.editor.text, "Test Course MOC")

    def test_typed_name_is_normalized_and_returned(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("c")
        assert modal.editor is not None
        modal.editor.text = "New Course"
        app._handle_modal_key("\n")
        self.assertEqual(modal.box["response"], MocDecision.create("New Course MOC"))

    def test_choosing_an_existing_moc_returns_its_path(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal(
            [
                MocMatch(path="University/A MOC.md", score=0.9),
                MocMatch(path="University/B MOC.md", score=0.8),
            ]
        )
        app.modal = modal
        app._handle_modal_key("j")
        app._handle_modal_key("\n")
        self.assertEqual(modal.box["response"], MocDecision.use("University/B MOC.md"))

    def test_standard_mode_uses_arrows_in_the_moc_modal(self):
        config = Config(notes_root=self.root, artifacts_dir=self.root / "home", movement="standard")
        app = TuiApp(config=config)
        app.scr = FakeScreen()
        modal = self.modal(
            [
                MocMatch(path="University/A MOC.md", score=0.9),
                MocMatch(path="University/B MOC.md", score=0.8),
            ]
        )
        app.modal = modal
        app._handle_modal_key(curses.KEY_DOWN)
        app._handle_modal_key("\n")
        self.assertEqual(modal.box["response"], MocDecision.use("University/B MOC.md"))

    def test_vim_mode_ignores_arrows_in_the_moc_modal(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal(
            [
                MocMatch(path="University/A MOC.md", score=0.9),
                MocMatch(path="University/B MOC.md", score=0.8),
            ]
        )
        app.modal = modal
        app._handle_modal_key(curses.KEY_DOWN)
        app._handle_modal_key("\n")
        self.assertEqual(modal.box["response"], MocDecision.use("University/A MOC.md"))

    def test_escape_aborts(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("q")
        self.assertEqual(modal.box["response"], MocDecision.abort())

    def test_m_opens_the_manual_moc_editor(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("m")
        self.assertEqual(modal.kind, "moc_manual")
        self.assertIsNotNone(modal.editor)

    def test_enter_on_the_manual_row_opens_the_manual_editor(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal([MocMatch(path="University/A MOC.md", score=0.9)])
        app.modal = modal
        app._handle_modal_key("j")
        app._handle_modal_key("j")
        app._handle_modal_key("\n")
        self.assertEqual(modal.kind, "moc_manual")

    def test_manual_path_selects_an_existing_moc(self):
        (self.root / "University").mkdir()
        (self.root / "University" / "Course MOC.md").write_text("# Course", encoding="utf-8")
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("m")
        assert modal.editor is not None
        modal.editor.set_text("University/Course MOC.md")
        app._handle_modal_key("\n")
        self.assertEqual(modal.box["response"], MocDecision.use("University/Course MOC.md"))

    def test_manual_title_selects_the_unique_moc(self):
        (self.root / "University").mkdir()
        (self.root / "University" / "Course MOC.md").write_text("# Course", encoding="utf-8")
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("m")
        assert modal.editor is not None
        modal.editor.set_text("Course MOC")
        app._handle_modal_key("\n")
        self.assertEqual(modal.box["response"], MocDecision.use("University/Course MOC.md"))

    def test_manual_unknown_name_reports_an_error(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("m")
        assert modal.editor is not None
        modal.editor.set_text("Missing MOC")
        app._handle_modal_key("\n")
        self.assertIsNone(modal.box.get("response"))
        self.assertIn("no such MOC", modal.message)

    def test_manual_tab_completion_fills_a_moc_path(self):
        (self.root / "University").mkdir()
        (self.root / "University" / "Course MOC.md").write_text("# Course", encoding="utf-8")
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("m")
        assert modal.editor is not None
        modal.editor.set_text("Cou")
        app._handle_modal_key("\t")
        self.assertEqual(modal.editor.text, "University/Course MOC.md")

    def test_manual_modal_renders(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        modal = self.modal()
        app.modal = modal
        app._handle_modal_key("m")
        app._draw()
        text = "".join(
            "".join(app.scr.cells.get((y, x), " ") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("Select an existing MOC", text)


class HistoryAndCommandsTests(unittest.TestCase):
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

    def tearDown(self):
        self._tmp.cleanup()

    def test_typing_a_slash_opens_command_suggestions_that_filter(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app.active = "input"
        app.mode = MODE_INSERT
        app._sync_editor_from_fields()

        app._handle_input_key("/")
        popup = app.completion
        assert popup is not None
        self.assertTrue(popup.is_command)
        self.assertEqual(
            {match.path for match in popup.matches},
            {"/run", "/new", "/history", "/resume", "/wizard"},
        )

        app._handle_input_key("h")
        popup = app.completion
        assert popup is not None
        self.assertTrue(popup.is_command)
        self.assertEqual([match.path for match in popup.matches], ["/history"])
        app._draw()
        text = "".join(
            "".join(app.scr.cells.get((y, x), "") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("Commands", text)
        self.assertIn("browse every run", text)

    def test_enter_on_a_command_suggestion_runs_it(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.focus = "input"
        app.active = "input"
        app.mode = MODE_INSERT
        app._sync_editor_from_fields()
        app._handle_input_key("/")
        app._handle_input_key("h")
        app._handle_input_key("\n")
        self.assertEqual(app.input_text, "")
        self.assertIsNotNone(app.modal)
        assert app.modal is not None
        self.assertEqual(app.modal.kind, "history")

    def test_wizard_command_reopens_the_setup_wizard(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "/wizard"
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key("\n")
        self.assertEqual(app.input_text, "")
        assert app.wizard is not None
        self.assertIsNone(app.settings)
        self.assertIsNone(app.modal)
        self.assertEqual(app.wizard.page, "vault")
        self.assertEqual(app.wizard.values["notes_root"], str(self.config.notes_root))

    def test_history_window_restores_a_previous_run(self):
        source = self.root / "lecture.mp4"
        source.write_bytes(b"x")
        entry = HistoryEntry(
            id="job-1",
            job_id="job-1",
            course="Test Course",
            created_at="2026-01-01T10:00:00+00:00",
            finished_at="2026-01-01T10:05:00+00:00",
            sources=[str(source)],
            changes=[
                DiffEntry(
                    path="New.md",
                    action="create",
                    added=1,
                    diff="--- /dev/null\n+++ b/New.md\n@@ -0,0 +1 @@\n+hello",
                    group="created",
                )
            ],
            card_groups=[
                CardGroup(
                    path="New.md",
                    cards=[Flashcard(front="What is X?", back="Y.")],
                    requested=1,
                    written=1,
                    writer="internal",
                )
            ],
            extras=[Aside(kind="exam", text="The exam is oral.")],
        )
        HistoryStore(self.config.history_dir).save(entry)

        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.input_text = "/history"
        app.focus = "input"
        app._sync_editor_from_fields()
        app._handle_input_key("\n")
        self.assertIsNotNone(app.modal)
        app._draw()
        text = "".join(
            "".join(app.scr.cells.get((y, x), "") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("Run history", text)
        self.assertIn("Test Course", text)

        app._handle_modal_key("\n")
        self.assertIsNone(app.modal)
        self.assertEqual(app.source_paths, [source])
        self.assertEqual(app.diff_entries[0].path, "New.md")
        self.assertEqual(app.card_groups[0].cards[0].front, "What is X?")
        self.assertEqual(app.asides[0].text, "The exam is oral.")
        self.assertEqual(app.current_job_id, "job-1")
        self.assertIn("history · Test Course", app.status)
        app._draw()
        text = "".join(
            "".join(app.scr.cells.get((y, x), "") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("New.md", text)

    def test_history_is_refused_while_a_job_runs(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.running = True
        app.open_history()
        self.assertIsNone(app.modal)
        self.assertIn("already running", app.status)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.source = self.root / "lecture.mp4"
        self.source.write_bytes(b"x")
        self.config = Config(
            notes_root=self.vault,
            artifacts_dir=self.root / "home",
            search_paths=(self.root,),
        )
        self.store = JobStore(self.config.artifacts_dir)
        self._key = mock.patch.dict(os.environ, {"SCRIPTORIUM_LLM_API_KEY": "test-key"})
        self._key.start()

    def tearDown(self):
        self._key.stop()
        self._tmp.cleanup()

    def _job(self, *, finished=False, error=None):
        state = self.store.create([self.source], "Test Course")
        state.set_status(Stage.SOURCES, "done")
        state.error = error
        if finished:
            for stage in GLOBAL_STAGES:
                state.set_status(stage, "done")
            group = GroupState(id="g1", topic="Topic")
            for stage in GROUP_STAGES:
                group.set_status(stage, "done")
            state.groups = [group]
        self.store.save(state)
        return state

    def _app_with_loaded_job(self, state):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.current_job_id = state.job_id
        return app

    def test_resume_without_a_loaded_job_points_to_history(self):
        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.start_resume()
        self.assertEqual(app.status, "no job loaded, pick one with /history")
        self.assertFalse(app.running)

    def test_resume_starts_the_loaded_job_without_asking_for_an_id(self):
        state = self._job(error="outline: boom")
        app = self._app_with_loaded_job(state)
        started = []

        class FakeThread:
            def __init__(self, target=None, args=(), **_kwargs):
                self.target = target
                self.args = args

            def start(self):
                started.append(self.args[0])

        with mock.patch("scriptorium.tui.threading.Thread", FakeThread):
            app.start_resume()

        self.assertTrue(app.running)
        self.assertEqual(len(started), 1)
        request = started[0]
        self.assertEqual(request.job_id, state.job_id)
        self.assertEqual(request.sources, [self.source])
        self.assertEqual(app.current_job_id, state.job_id)
        self.assertIn("resuming", app.status)

    def test_resume_refuses_a_completed_job(self):
        state = self._job(finished=True)
        app = self._app_with_loaded_job(state)
        app.start_resume()
        self.assertEqual(app.status, f"job {state.job_id} is already complete")
        self.assertFalse(app.running)

    def test_resume_refuses_a_missing_source(self):
        state = self._job(error="outline: boom")
        self.source.unlink()
        app = self._app_with_loaded_job(state)
        app.start_resume()
        self.assertEqual(app.status, f"source not found: {self.source}")
        self.assertFalse(app.running)

    def test_history_lists_failed_runs_and_resume_continues_the_loaded_one(self):
        state = self._job(error="compose: boom")
        entry = HistoryEntry(
            id=state.job_id,
            job_id=state.job_id,
            course="Test Course",
            created_at="2026-01-01T10:00:00+00:00",
            finished_at="2026-01-01T10:01:00+00:00",
            status="failed",
            error=state.error,
            sources=[str(self.source)],
        )
        HistoryStore(self.config.history_dir).save(entry)

        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.open_history()
        app._draw()
        text = "".join(
            "".join(app.scr.cells.get((y, x), "") for x in range(app.scr.cols))
            for y in range(app.scr.rows)
        )
        self.assertIn("failed at", text)

        app._handle_modal_key("\n")
        self.assertEqual(app.current_job_id, state.job_id)
        self.assertIn("/resume to continue", app.status)

    def test_loading_a_resumable_run_reads_the_latest_artifacts(self):
        state = self._job(error="apply: boom")
        job_dir = self.store.job_dir(state.job_id)
        (job_dir / "apply.json").write_text(
            json.dumps(
                {
                    "created": [
                        {
                            "path": "New.md",
                            "action": "create",
                            "added_lines": 1,
                            "removed_lines": 0,
                            "diff": "--- /dev/null\n+++ b/New.md\n@@ -0,0 +1 @@\n+one",
                        }
                    ],
                    "merged": [],
                    "moc": [],
                }
            ),
            encoding="utf-8",
        )
        HistoryStore(self.config.history_dir).save(
            HistoryEntry(
                id=state.job_id,
                job_id=state.job_id,
                course="Test Course",
                created_at="2026-01-01T10:00:00+00:00",
                finished_at="2026-01-01T10:01:00+00:00",
                status="failed",
                error=state.error,
                sources=[str(self.source)],
            )
        )

        app = TuiApp(config=self.config)
        app.scr = FakeScreen()
        app.open_history()
        app._handle_modal_key("\n")

        self.assertEqual([entry.path for entry in app.diff_entries], ["New.md"])


if __name__ == "__main__":
    unittest.main()
