import io
import unittest

from scriptorium.ui import MOC_CONFIDENCE, ConsoleUI, MocDecision, moc_title
from scriptorium.vault import MocMatch


class MocTitleTests(unittest.TestCase):
    def test_suffix_is_appended(self):
        self.assertEqual(moc_title("Foundations"), "Foundations MOC")

    def test_existing_suffix_is_kept_and_normalized(self):
        self.assertEqual(moc_title("Foundations MOC"), "Foundations MOC")
        self.assertEqual(moc_title("foundations moc"), "foundations MOC")

    def test_empty_value_falls_back(self):
        self.assertEqual(moc_title("   ", "Course MOC"), "Course MOC")
        self.assertEqual(moc_title(""), "Course MOC")

    def test_path_separators_are_replaced(self):
        self.assertEqual(moc_title("AI/ML: Basics"), "AI ML Basics MOC")


class ChooseMocTests(unittest.TestCase):
    def ui(self, answers: str, *, interactive: bool = True, yes: bool = False):
        stdout = io.StringIO()
        return (
            ConsoleUI(
                yes=yes,
                interactive=interactive,
                stdin=io.StringIO(answers),
                stdout=stdout,
            ),
            stdout,
        )

    def match(self, path: str, score: float) -> MocMatch:
        return MocMatch(path=path, score=score)

    def test_non_interactive_uses_a_confident_match(self):
        ui, _ = self.ui("", interactive=False)
        decision = ui.choose_moc("Course", [self.match("A MOC.md", 0.9)])
        self.assertEqual(decision, MocDecision.use("A MOC.md"))

    def test_non_interactive_creates_when_nothing_matches(self):
        ui, _ = self.ui("", interactive=False)
        decision = ui.choose_moc("Course", [], default_title="Course MOC")
        self.assertEqual(decision, MocDecision.create("Course MOC"))

    def test_non_interactive_creates_below_the_threshold(self):
        ui, _ = self.ui("", interactive=False)
        decision = ui.choose_moc("Course", [self.match("Weak MOC.md", MOC_CONFIDENCE - 0.1)])
        self.assertEqual(decision.action, "create")

    def test_interactive_selection(self):
        ui, _ = self.ui("2\n")
        decision = ui.choose_moc(
            "Course", [self.match("A MOC.md", 0.9), self.match("B MOC.md", 0.8)]
        )
        self.assertEqual(decision, MocDecision.use("B MOC.md"))

    def test_interactive_creation_shows_the_final_name(self):
        ui, out = self.ui("c\nNew Course\ny\n")
        decision = ui.choose_moc("Course", [], default_title="Course MOC")
        self.assertEqual(decision, MocDecision.create("New Course MOC"))
        self.assertIn("New MOC: New Course MOC", out.getvalue())

    def test_interactive_abort(self):
        ui, _ = self.ui("q\n")
        decision = ui.choose_moc("Course", [self.match("A MOC.md", 0.9)])
        self.assertEqual(decision.action, "abort")

    def test_blank_answer_without_candidates_creates_the_default(self):
        ui, _ = self.ui("\n\n")
        decision = ui.choose_moc("Course", [], default_title="Course MOC")
        self.assertEqual(decision, MocDecision.create("Course MOC"))


if __name__ == "__main__":
    unittest.main()
