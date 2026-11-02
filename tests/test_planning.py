import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scriptorium.config import Config
from scriptorium.context import ContextBundle
from scriptorium.log import Logger
from scriptorium.models import (
    ExtractResult,
    NoteOutline,
    Outline,
    Segment,
    TodoItem,
    Transcript,
    Unit,
)
from scriptorium.planning import (
    compose_notes,
    extract_units,
    normalize_outline,
    split_transcript,
)
from scriptorium.vault import Vault


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.usage = SimpleNamespace(calls=0, prompt_tokens=0, completion_tokens=0)
        self.prompts = []

    def chat_json(self, system, user, *, phase="llm", temperature=0.2, max_tokens=8192):
        self.prompts.append(user)
        return self.responses.pop(0)


def make_transcript(count=6):
    return Transcript(
        segments=[
            Segment(start=index * 10, end=index * 10 + 10, text=f"Segment number {index} text")
            for index in range(count)
        ]
    )


class SplitTests(unittest.TestCase):
    def test_split_respects_size_and_overlap(self):
        transcript = make_transcript(20)
        chunks = split_transcript(transcript, max_chars=120, overlap_chars=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))

    def test_empty_transcript(self):
        self.assertEqual(split_transcript(Transcript()), [])


class ExtractTests(unittest.TestCase):
    def test_extract_dedupes_and_assigns_ids(self):
        client = FakeClient(
            [
                {
                    "units": [
                        {
                            "kind": "definition",
                            "title": "Heuristic Function",
                            "content": "Estimates the cost to the goal.",
                            "source": "00:00:10",
                            "confidence": 0.9,
                        },
                        {
                            "kind": "definition",
                            "title": "heuristic function",
                            "content": "Better content.",
                            "source": "00:00:20",
                            "confidence": 0.95,
                        },
                        {
                            "kind": "unknown-kind",
                            "title": "Frontier",
                            "content": "The set of nodes to expand.",
                        },
                    ],
                    "topics": ["Search"],
                }
            ]
        )
        result = extract_units(make_transcript(), client, Logger(), course="AI")
        self.assertEqual(len(result.units), 2)
        self.assertEqual(result.units[0].id, "u1")
        self.assertEqual(result.units[0].content, "Better content.")
        self.assertEqual(result.units[1].kind, "concept")

    def test_extract_keeps_off_topic_asides_separate_and_deduped(self):
        client = FakeClient(
            [
                {
                    "units": [
                        {
                            "kind": "definition",
                            "title": "A concept",
                            "content": "Some lecture content.",
                        }
                    ],
                    "asides": [
                        {"kind": "exam", "text": "The exam is oral.", "source": "00:00:10"},
                        {"kind": "exam", "text": "The exam is oral.", "source": "00:00:12"},
                        {"kind": "mystery", "text": "The room changed.", "source": "00:00:20"},
                    ],
                    "topics": [],
                }
            ]
        )
        result = extract_units(make_transcript(), client, Logger(), course="AI")
        self.assertEqual(len(result.asides), 2)
        self.assertEqual(result.asides[0].text, "The exam is oral.")
        self.assertIn("lecture transcript", result.asides[0].source)
        self.assertEqual(result.asides[1].kind, "remark")


class OutlineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = Vault(self.root)
        (self.root / "Existing.md").write_text(
            "---\ntags:\n  - cs/ai\n---\n\n## Section\n\ntext\n", encoding="utf-8"
        )
        self.context = ContextBundle(
            course="Course",
            moc_path="University/Course MOC.md",
            moc_text="",
            tags=["cs/ai"],
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_converts_to_append_when_note_exists(self):
        outline = Outline(
            notes=[
                NoteOutline(action="create", path="Existing.md", title="Existing", unit_ids=["u1"])
            ]
        )
        normalized = normalize_outline(outline, self.vault, self.context, Logger())
        self.assertEqual(normalized.notes[0].action, "append")

    def test_targeting_merges_into_an_existing_heading(self):
        config = Config(
            notes_root=self.root,
            artifacts_dir=self.root / "home",
            target_policy="auto",
            target_auto_above=0.35,
            target_min_gap=0.0,
        )
        outline = Outline(
            notes=[
                NoteOutline(action="create", path="Section.md", title="Section", unit_ids=["u1"])
            ]
        )
        normalized = normalize_outline(
            outline, self.vault, self.context, Logger(), config=config
        )
        note = normalized.notes[0]
        self.assertEqual(note.action, "merge")
        self.assertEqual(note.path, "Existing.md")
        self.assertEqual(note.anchor, "Section")

    def test_targeting_ask_without_answer_creates_and_lists_candidates(self):
        config = Config(notes_root=self.root, artifacts_dir=self.root / "home", target_policy="ask")
        outline = Outline(
            notes=[
                NoteOutline(action="create", path="Section.md", title="Section", unit_ids=["u1"])
            ]
        )
        normalized = normalize_outline(
            outline,
            self.vault,
            self.context,
            Logger(),
            config=config,
            chooser=lambda _options: None,
        )
        note = normalized.notes[0]
        self.assertEqual(note.action, "create")
        self.assertTrue(any("candidates were" in todo.text for todo in normalized.todos))

    def test_targeting_ask_applies_the_chosen_option(self):
        config = Config(notes_root=self.root, artifacts_dir=self.root / "home", target_policy="ask")
        outline = Outline(
            notes=[
                NoteOutline(action="create", path="Section.md", title="Section", unit_ids=["u1"])
            ]
        )
        normalized = normalize_outline(
            outline,
            self.vault,
            self.context,
            Logger(),
            config=config,
            chooser=lambda options: options[0],
        )
        self.assertEqual(normalized.notes[0].anchor, "Section")
        self.assertEqual(normalized.notes[0].path, "Existing.md")

    def test_missing_append_becomes_create_and_missing_anchor_is_reported(self):
        outline = Outline(
            notes=[
                NoteOutline(action="append", path="Missing.md", title="Missing", unit_ids=["u1"]),
                NoteOutline(
                    action="append",
                    path="Existing.md",
                    title="Existing",
                    anchor="Nope",
                    unit_ids=["u2"],
                ),
            ]
        )
        normalized = normalize_outline(outline, self.vault, self.context, Logger())
        self.assertEqual(normalized.notes[0].action, "create")
        self.assertEqual(normalized.notes[0].tags, ["cs/ai"])
        self.assertIsNone(normalized.notes[1].anchor)
        self.assertTrue(any(todo.kind == "restructure" for todo in normalized.todos))


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = Vault(self.root)
        self.config = Config(notes_root=self.root, artifacts_dir=self.root / "home")
        self.context = ContextBundle(
            course="Course",
            moc_path="University/Course MOC.md",
            moc_text="",
            tags=["cs/ai"],
            titles=["Existing"],
        )
        self.extract = ExtractResult(
            units=[Unit(id="u1", kind="definition", title="Concept", content="Definition.")]
        )
        self.outline = Outline(
            notes=[
                NoteOutline(
                    action="create",
                    path="New.md",
                    title="New",
                    tags=["cs/ai"],
                    unit_ids=["u1"],
                )
            ],
            todos=[TodoItem(kind="unclear", text="Check the formula.")],
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_compose_flattens_missing_links_and_filters_cards(self):
        client = FakeClient(
            [
                {
                    "body": (
                        "---\ntags:\n  - nope\n---\n\n# New\n\n## Section\n\n"
                        "See [[Existing]] and [[Missing|missing note]].\n"
                    ),
                    "flashcards": [
                        {"front": "What is X?", "back": "Y.", "kind": "single"},
                        {"front": "Not a question", "back": "Z.", "kind": "single"},
                        {"front": "What is X?", "back": "Duplicate.", "kind": "single"},
                    ],
                }
            ]
        )
        plan = compose_notes(
            self.outline,
            self.extract,
            self.context,
            client,
            Logger(),
            self.vault,
            self.config,
        )
        note = plan.notes[0]
        self.assertIn("[[Existing]]", note.body)
        self.assertNotIn("[[Missing", note.body)
        self.assertIn("missing note", note.body)
        self.assertTrue(note.body.startswith("## Section"))
        self.assertEqual(len(note.flashcards), 1)
        self.assertTrue(any(todo.kind == "link" for todo in plan.todos))


if __name__ == "__main__":
    unittest.main()


class GroupingTests(unittest.TestCase):
    def setUp(self):
        from scriptorium.sources import TextSource

        self.sources = [
            TextSource(
                id="s01",
                name="slides.pdf",
                kind="document",
                path=Path("slides.pdf"),
                text="Slides about heuristics and search.",
            ),
            TextSource(
                id="s02",
                name="lecture.mp4",
                kind="media",
                path=Path("lecture.mp4"),
                text="Lecture about heuristic search.",
            ),
        ]
        self.extract = ExtractResult(
            units=[
                Unit(
                    id="u1",
                    kind="concept",
                    title="Heuristic",
                    content="An estimate.",
                    source="slides.pdf [page 1]",
                ),
                Unit(
                    id="u2",
                    kind="concept",
                    title="Search",
                    content="Exploring states.",
                    source="lecture.mp4 00:01:00",
                ),
            ]
        )

    def test_single_source_needs_no_llm_call(self):
        from scriptorium.planning import build_groups

        client = FakeClient([])
        grouping = build_groups(self.sources[:1], self.extract, client, Logger(), course="AI")
        self.assertEqual(len(grouping.groups), 1)
        self.assertEqual(grouping.groups[0].source_ids, ["s01"])
        self.assertEqual(client.prompts, [])

    def test_model_groups_are_ordered_and_keep_every_source(self):
        from scriptorium.planning import build_groups

        client = FakeClient(
            [
                {
                    "groups": [
                        {"id": "g1", "topic": "Search", "source_ids": ["s02"]},
                        {"id": "g2", "topic": "Heuristics", "source_ids": ["s01"]},
                    ]
                }
            ]
        )
        grouping = build_groups(self.sources, self.extract, client, Logger(), course="AI")
        self.assertEqual([group.topic for group in grouping.groups], ["Search", "Heuristics"])
        self.assertEqual([group.id for group in grouping.groups], ["g1", "g2"])
        self.assertIn("slides.pdf", client.prompts[0])

    def test_unknown_and_missing_sources_are_repaired(self):
        from scriptorium.planning import build_groups

        client = FakeClient(
            [{"groups": [{"id": "g1", "topic": "Only", "source_ids": ["s01", "s99"]}]}]
        )
        grouping = build_groups(self.sources, self.extract, client, Logger(), course="AI")
        self.assertEqual(len(grouping.groups), 1)
        self.assertEqual(grouping.groups[0].source_ids, ["s01", "s02"])
        self.assertTrue(grouping.todos, "an unassigned source must be reported")

    def test_units_are_selected_by_source_provenance(self):
        from scriptorium.planning import units_for_sources

        self.assertEqual(
            [unit.id for unit in units_for_sources(self.extract, {"slides.pdf"})],
            ["u1"],
        )
        self.assertEqual(
            [unit.id for unit in units_for_sources(self.extract, {"lecture.mp4"})],
            ["u2"],
        )
