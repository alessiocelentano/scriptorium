import tempfile
import unittest
from pathlib import Path

from scriptorium.config import Config
from scriptorium.targeting import (
    decide,
    merge_candidates,
    resolve_targets,
)
from scriptorium.vault import Vault


class TargetResolutionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = Vault(self.root)
        (self.root / "Derivate.md").write_text(
            "# Derivate\n\n## Derivate parziali\n\nOld text.\n\n## Gradiente\n\nMore.\n",
            encoding="utf-8",
        )
        (self.root / "Integrali.md").write_text(
            "# Integrali\n\n## Integrali definiti\n\nText.\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def config(self, **overrides) -> Config:
        values: dict[str, object] = {"notes_root": self.root, "artifacts_dir": self.root / "home"}
        values.update(overrides)
        return Config(**values)  # type: ignore[arg-type]

    def test_close_title_and_heading_produce_a_merge_candidate(self):
        options = resolve_targets(self.vault, "Derivate parziali")
        best = options[0]
        self.assertEqual(best.kind, "merge")
        self.assertEqual(best.path, "Derivate.md")
        self.assertEqual(best.anchor, "Derivate parziali")
        self.assertTrue(best.score > 0.5)
        self.assertEqual(options[-1].kind, "create")

    def test_unrelated_title_only_offers_the_new_note(self):
        options = resolve_targets(self.vault, "Trasformata di Laplace")
        self.assertEqual([option.kind for option in options], ["create"])

    def test_merge_candidates_exclude_create(self):
        options = resolve_targets(self.vault, "Derivate parziali")
        self.assertTrue(all(item.kind != "create" for item in merge_candidates(options)))

    def test_policy_create_never_merges(self):
        options = resolve_targets(self.vault, "Derivate parziali")
        choice = decide(options, self.config(target_policy="create"))
        self.assertEqual(choice.target.kind, "create")
        self.assertEqual(choice.origin, "policy")

    def test_policy_auto_merges_above_the_threshold(self):
        options = resolve_targets(self.vault, "Derivate parziali")
        choice = decide(
            options,
            self.config(target_policy="auto", target_auto_above=0.5, target_min_gap=0.0),
        )
        self.assertEqual(choice.target.kind, "merge")
        self.assertEqual(choice.origin, "auto")

    def test_policy_auto_falls_back_to_a_new_note_when_unsure(self):
        options = resolve_targets(self.vault, "Derivate parziali")
        choice = decide(options, self.config(target_policy="auto", target_auto_above=0.99))
        self.assertEqual(choice.target.kind, "create")
        self.assertEqual(choice.origin, "fallback")

    def test_policy_ask_uses_the_chooser(self):
        options = resolve_targets(self.vault, "Derivate parziali")
        choice = decide(
            options,
            self.config(target_policy="ask"),
            chooser=lambda choices: choices[1] if len(choices) > 1 else None,
        )
        self.assertEqual(choice.target, options[1])
        self.assertEqual(choice.origin, "policy")

    def test_policy_ask_without_an_answer_creates_a_note(self):
        options = resolve_targets(self.vault, "Derivate parziali")
        choice = decide(options, self.config(target_policy="ask"), chooser=lambda _choices: None)
        self.assertEqual(choice.target.kind, "create")
        self.assertEqual(choice.origin, "fallback")

    def test_empty_vault_only_offers_a_new_note(self):
        empty = self.root / "empty"
        empty.mkdir()
        options = resolve_targets(Vault(empty), "Anything")
        self.assertEqual([option.kind for option in options], ["create"])


if __name__ == "__main__":
    unittest.main()
