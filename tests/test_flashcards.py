import tempfile
import unittest
from pathlib import Path

from scriptorium.config import Config
from scriptorium.flashcards import (
    render_cards,
    validate_flashcard,
    write_drafts,
)
from scriptorium.log import Logger
from scriptorium.models import Flashcard
from scriptorium.vault import Vault


class ValidationTests(unittest.TestCase):
    def test_valid_card(self):
        card = Flashcard(front="What is a framebuffer?", back="A region of memory.")
        self.assertEqual(validate_flashcard(card), [])

    def test_front_must_be_a_question(self):
        card = Flashcard(front="framebuffer", back="A region of memory.")
        self.assertTrue(validate_flashcard(card))

    def test_rejects_em_dash(self):
        card = Flashcard(front="What is X?", back="A \u2014 B")
        self.assertTrue(validate_flashcard(card))

    def test_rejects_helper_breaking_suffix(self):
        card = Flashcard(front="What is X?", back="A --> B")
        self.assertTrue(validate_flashcard(card))

    def test_rejects_separator_inside_card(self):
        card = Flashcard(front="What is X?", back="A ;;; B")
        self.assertTrue(validate_flashcard(card))

    def test_cloze_requires_deletion(self):
        self.assertTrue(validate_flashcard(Flashcard(front="A sentence", kind="cloze")))
        self.assertEqual(
            validate_flashcard(Flashcard(front="A {{c1::sentence}}", kind="cloze")),
            [],
        )

    def test_multi_rejects_standalone_separator(self):
        card = Flashcard(front="Question?", back="A\n???\nB", kind="multi")
        self.assertTrue(validate_flashcard(card))


class RenderTests(unittest.TestCase):
    def test_render_kinds(self):
        cards = [
            Flashcard(front="Q1?", back="A1", kind="single"),
            Flashcard(front="Q2?", back="A2", kind="reversed"),
            Flashcard(front="Q3?", back="A3", kind="multi"),
            Flashcard(front="A {{c1::cloze}}", back="extra", kind="cloze"),
        ]
        rendered = render_cards(cards)
        self.assertIn("Q1? ;;; A1", rendered)
        self.assertIn("Q2? ;;;; A2", rendered)
        self.assertIn("Q3?\n???\nA3", rendered)
        self.assertIn("A {{c1::cloze}}", rendered)
        self.assertNotIn("A {{c1::cloze}}\n\nextra", rendered)
        self.assertTrue(rendered.endswith("\n"))


class WriterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = Vault(self.root)
        self.config = Config(notes_root=self.root, artifacts_dir=self.root / "home")
        self.logger = Logger(verbose=False)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def test_writes_drafts_without_touching_existing_lines(self):
        original = "---\ntags:\n  - cs/ai\n---\n\n## Section\n\nBody.\n"
        self.write("N.md", original)
        result = write_drafts(
            self.vault,
            "N.md",
            [Flashcard(front="What is X?", back="Y.")],
            self.config,
            self.logger,
        )
        self.assertEqual(result.written, 1)
        text = self.read("N.md")
        self.assertTrue(text.startswith(original))
        self.assertIn("## Flashcards", text)
        self.assertIn("<!--DRAFT: What is X? ;;; Y. -->", text)

    def test_skips_untagged_note(self):
        self.write("N.md", "# N\n\nbody\n")
        result = write_drafts(
            self.vault,
            "N.md",
            [Flashcard(front="What is X?", back="Y.")],
            self.config,
            self.logger,
        )
        self.assertEqual(result.written, 0)
        self.assertEqual(result.skipped, 1)
        self.assertNotIn("DRAFT", self.read("N.md"))

    def test_skill_without_add_drafts_falls_back_to_internal_insertion(self):
        self.write("N.md", "---\ntags:\n  - cs/ai\n---\n\n## Flashcards\n")
        skill = self.root / "skill.py"
        skill.write_text(
            "import sys\nprint(\"invalid choice: 'add-drafts'\", file=sys.stderr)\nsys.exit(2)\n",
            encoding="utf-8",
        )
        config = self.config.with_overrides(flashcard_skill=skill)
        result = write_drafts(
            self.vault,
            "N.md",
            [Flashcard(front="What is X?", back="Y.")],
            config,
            self.logger,
        )
        self.assertEqual(result.written, 1)
        self.assertEqual(result.writer, "internal")
        self.assertIn("<!--DRAFT: What is X? ;;; Y. -->", self.read("N.md"))

    def test_any_skill_failure_falls_back_and_restores_partial_writes(self):
        original = "---\ntags:\n  - cs/ai\n---\n\n## Flashcards\n"
        self.write("N.md", original)
        skill = self.root / "skill.py"
        skill.write_text(
            "import pathlib\n"
            "import sys\n"
            "path = pathlib.Path(sys.argv[sys.argv.index('--file') + 1])\n"
            "path.write_text(path.read_text() + '<!--DRAFT: partial -->\\n')\n"
            "print('boom', file=sys.stderr)\n"
            "sys.exit(3)\n",
            encoding="utf-8",
        )
        config = self.config.with_overrides(flashcard_skill=skill)
        result = write_drafts(
            self.vault,
            "N.md",
            [Flashcard(front="What is X?", back="Y.")],
            config,
            self.logger,
        )
        self.assertEqual(result.written, 1)
        self.assertEqual(result.writer, "internal")
        text = self.read("N.md")
        self.assertNotIn("partial", text)
        self.assertEqual(text.count("<!--DRAFT: What is X? ;;; Y. -->"), 1)

    def test_skill_writing_active_cards_is_reverted_and_replaced_with_drafts(self):
        self.write("N.md", "---\ntags:\n  - cs/ai\n---\n\n## Flashcards\n")
        skill = self.root / "skill.py"
        skill.write_text(
            "import pathlib\n"
            "import sys\n"
            "path = pathlib.Path(sys.argv[sys.argv.index('--file') + 1])\n"
            "path.write_text(path.read_text().rstrip('\\n') + '\\n\\nWhat is X? ;;; Y.\\n')\n",
            encoding="utf-8",
        )
        config = self.config.with_overrides(flashcard_skill=skill)
        result = write_drafts(
            self.vault,
            "N.md",
            [Flashcard(front="What is X?", back="Y.")],
            config,
            self.logger,
        )
        self.assertEqual(result.written, 1)
        self.assertEqual(result.writer, "internal")
        lines = self.read("N.md").splitlines()
        self.assertNotIn("What is X? ;;; Y.", lines)
        self.assertIn("<!--DRAFT: What is X? ;;; Y. -->", lines)

    def test_multiline_card_is_commented_line_by_line(self):
        self.write("N.md", "---\ntags:\n  - cs/ai\n---\n\n## Flashcards\n")
        card = Flashcard(front="Question?", back="Line one\nLine two", kind="multi")
        write_drafts(self.vault, "N.md", [card], self.config, self.logger)
        text = self.read("N.md")
        self.assertIn("<!--DRAFT: Line one -->", text)
        self.assertIn("<!--DRAFT: Line two -->", text)
        self.assertNotIn("<!--DRAFT: Question?\n???", text)


if __name__ == "__main__":
    unittest.main()
