import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scriptorium.config import (
    Config,
    ConfigurationError,
    asr_api_key,
    is_first_run,
    load_config,
    mark_setup_complete,
    setup_marker_path,
    write_config_values,
)
from scriptorium.warnings import known_codes


class ConfigSettingsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.home = self.root / "home"

    def tearDown(self):
        self._tmp.cleanup()

    def config(self, **overrides: object) -> Config:
        values: dict[str, object] = {"notes_root": self.vault, "artifacts_dir": self.home}
        values.update(overrides)
        return Config(**values)  # type: ignore[arg-type]

    def test_search_paths_are_exactly_what_was_configured(self):
        config = self.config(search_paths=(self.vault,))
        self.assertEqual(config.search_paths, (self.vault,))

    def test_notes_root_is_always_a_search_path(self):
        other = self.root / "courses"
        config = self.config(search_paths=(other,))
        self.assertEqual(config.search_paths, (other, self.vault))
        self.assertEqual(config.user_search_paths, (other,))
        # Listing it explicitly does not duplicate it.
        config = self.config(search_paths=(other, self.vault))
        self.assertEqual(config.search_paths, (other, self.vault))
        self.assertEqual(config.user_search_paths, (other, self.vault))

    def test_user_search_paths_is_empty_when_nothing_was_configured(self):
        config = self.config()
        self.assertEqual(config.user_search_paths, ())
        # Nothing is added implicitly apart from the notes root.
        self.assertEqual(config.search_paths, (self.vault,))

    def test_language_detect_leaves_the_language_empty(self):
        self.assertEqual(self.config(language="detect").language, "")
        self.assertEqual(self.config(language="auto").language, "")
        self.assertEqual(self.config(language="it").language, "it")
        self.assertEqual(self.config(language="ask").language, "ask")

    def test_attachment_dir_resolves_inside_the_vault(self):
        config = self.config(attachment_dir=Path("Attachments/scriptorium"))
        self.assertEqual(
            config.attachments_root.resolve(),
            (self.vault / "Attachments" / "scriptorium").resolve(),
        )

    def test_absolute_attachment_dir_may_live_outside_notes_root(self):
        config = self.config(attachment_dir=self.root / "elsewhere")
        self.assertEqual(config.attachments_root, (self.root / "elsewhere").resolve())

    def test_unknown_image_input_mode_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(image_input="maybe")

    def test_unknown_merge_policy_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(merge_policy="always")

    def test_unknown_target_policy_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(target_policy="guess")

    def test_target_thresholds_must_be_probabilities(self):
        with self.assertRaises(ConfigurationError):
            self.config(target_auto_above=1.5)
        with self.assertRaises(ConfigurationError):
            self.config(target_min_gap=-0.1)

    def test_unknown_scanned_pdf_mode_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(scanned_pdf="magic")

    def test_unknown_movement_style_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(movement="modal")

    def test_asr_defaults_to_the_local_endpoint(self):
        config = self.config()
        self.assertEqual(config.asr_api_model, "")
        self.assertEqual(config.asr_api_base_url, "")

    def test_asr_api_key_reads_the_provider_variable(self):
        env = {
            "SCRIPTORIUM_ASR_API_KEY": "",
            "ASR_API_KEY": "",
            "OPENAI_API_KEY": "",
            "GROQ_API_KEY": "groq-key",
            "HF_TOKEN": "",
            "HUGGING_FACE_HUB_TOKEN": "",
        }
        with mock.patch.dict(os.environ, env):
            self.assertEqual(asr_api_key(), "groq-key")

    def test_asr_api_key_prefers_the_explicit_override(self):
        env = {
            "SCRIPTORIUM_ASR_API_KEY": "override",
            "GROQ_API_KEY": "groq-key",
        }
        with mock.patch.dict(os.environ, env):
            self.assertEqual(asr_api_key(), "override")

    def test_asr_api_key_uses_the_configured_value(self):
        env = {"SCRIPTORIUM_ASR_API_KEY": "", "GROQ_API_KEY": ""}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(asr_api_key("file-key"), "file-key")

    def test_asr_api_key_prefers_the_environment_over_the_configured_value(self):
        env = {"SCRIPTORIUM_ASR_API_KEY": "override"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(asr_api_key("file-key"), "override")

    def test_a_base_url_alone_switches_to_the_api(self):
        config = self.config(asr_api_base_url="https://example.invalid/v1")
        # No explicit api model: the transcription model is used.
        self.assertEqual(config.asr_api_model, config.asr_model)

    def test_an_explicit_api_model_is_kept(self):
        config = self.config(
            asr_api_base_url="https://example.invalid/v1",
            asr_api_model="whisper-1",
        )
        self.assertEqual(config.asr_api_model, "whisper-1")

    def test_unknown_asr_api_protocol_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(
                asr_api_base_url="https://api.openai.com/v1",
                asr_api_model="whisper-1",
                asr_api_protocol="grpc",
            )

    def test_asr_api_base_url_is_normalized(self):
        config = self.config(
            asr_api_base_url="https://api.openai.com/v1/",
            asr_api_model="whisper-1",
        )
        self.assertEqual(config.asr_api_base_url, "https://api.openai.com/v1")

    def test_overriding_the_endpoint_replaces_the_base_url(self):
        local = self.config()
        remote = local.with_overrides(asr_api_base_url="https://example.invalid/v1")
        self.assertEqual(remote.asr_api_base_url, "https://example.invalid/v1")
        back = remote.with_overrides(asr_api_base_url="")
        self.assertEqual(back.asr_api_base_url, "")

    def test_unknown_llm_json_mode_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(llm_json_mode="strict")

    def test_llm_base_url_is_normalized(self):
        config = self.config(llm_base_url="https://example.invalid/v1/")
        self.assertTrue(config.llm_base_url.endswith("/v1"))

    def test_asr_api_timeout_must_be_positive(self):
        with self.assertRaises(ConfigurationError):
            self.config(asr_api_timeout=0)

    def test_asr_api_rpm_defaults_to_no_throttling(self):
        self.assertEqual(self.config().asr_api_rpm, 0)

    def test_a_negative_asr_api_rpm_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(asr_api_rpm=-1)

    def test_ocr_languages_are_normalized_for_tesseract(self):
        config = self.config(ocr_languages="eng, ita, eng")
        self.assertEqual(config.ocr_languages, "eng+ita")

    def test_negative_page_cap_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            self.config(max_scanned_pages=-1)

    def test_disabled_warnings_are_deduplicated_and_trimmed(self):
        config = self.config(disabled_warnings=(" cards_empty ", "", "cards_empty"))
        self.assertEqual(config.disabled_warnings, ("cards_empty",))


class LoadConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.config_path = self.root / "config.toml"
        self.config_path.write_text(
            "\n".join(
                [
                    f'notes_root = "{self.vault}"',
                    f'artifacts_dir = "{self.root / "home"}"',
                    f'search_paths = ["{self.root / "courses"}", "{self.root / "courses"}"]',
                    'attachment_dir = "Media"',
                    'image_input = "enabled"',
                    'scanned_pdf = "ocr"',
                    'ocr_languages = "eng+ita"',
                    "max_scanned_pages = 12",
                    'merge_policy = "manual"',
                    'target_policy = "auto"',
                    'movement = "standard"',
                    "target_auto_above = 0.7",
                    "target_min_gap = 0.2",
                    'asr_api_key = "toml-key"',
                    'disabled_warnings = ["cards_empty", "merge_skipped"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self.env = {
            "PATH": os.environ.get("PATH", ""),
        }
        self._path_patch = mock.patch(
            "scriptorium.config._config_path", return_value=self.config_path
        )
        self._write_patch = mock.patch(
            "scriptorium.config._config_write_path", return_value=self.config_path
        )
        self._path_patch.start()
        self._write_patch.start()

    def tearDown(self):
        self._write_patch.stop()
        self._path_patch.stop()
        self._tmp.cleanup()

    def test_toml_settings_are_resolved(self):
        with mock.patch.dict(os.environ, self.env, clear=True):
            config = load_config()
        self.assertEqual(config.search_paths, (self.root / "courses", self.vault))
        self.assertEqual(config.attachments_root.resolve(), (self.vault / "Media").resolve())
        self.assertEqual(config.image_input, "enabled")
        self.assertEqual(config.scanned_pdf, "ocr")
        self.assertEqual(config.ocr_languages, "eng+ita")
        self.assertEqual(config.max_scanned_pages, 12)
        self.assertEqual(config.merge_policy, "manual")
        self.assertEqual(config.target_policy, "auto")
        self.assertEqual(config.movement, "standard")
        self.assertEqual(config.target_auto_above, 0.7)
        self.assertEqual(config.target_min_gap, 0.2)
        self.assertEqual(config.asr_api_key, "toml-key")
        self.assertEqual(config.disabled_warnings, ("cards_empty", "merge_skipped"))

    def test_environment_overrides_search_paths(self):
        env = dict(self.env)
        env["SCRIPTORIUM_SEARCH_PATHS"] = os.pathsep.join(
            [str(self.root / "a"), str(self.root / "b")]
        )
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_config()
        self.assertEqual(config.search_paths, (self.root / "a", self.root / "b", self.vault))

    def test_warning_codes_are_serialized_as_a_list(self):
        with mock.patch.dict(os.environ, self.env, clear=True):
            write_config_values({"disabled_warnings": list(known_codes()[:2])})
            config = load_config()
        self.assertEqual(config.disabled_warnings, known_codes()[:2])

    def test_asr_api_rpm_comes_from_the_environment(self):
        env = dict(self.env)
        env["SCRIPTORIUM_ASR_API_RPM"] = "180"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(load_config().asr_api_rpm, 180)

    def test_asr_api_rpm_defaults_to_zero_when_unset(self):
        with mock.patch.dict(os.environ, self.env, clear=True):
            self.assertEqual(load_config().asr_api_rpm, 0)


class WarningCollectorTests(unittest.TestCase):
    def setUp(self):
        from scriptorium.warnings import WarningCollector

        self.collector = WarningCollector(disabled=("cards_empty",))

    def test_disabled_codes_are_dropped(self):
        self.assertIsNone(self.collector.add("cards_empty", "no cards"))
        self.assertEqual(len(self.collector), 0)

    def test_duplicate_warnings_are_collapsed(self):
        self.collector.add("source_unreadable", "cannot read", source="a.pdf")
        self.collector.add("source_unreadable", "cannot read", source="a.pdf")
        self.assertEqual(len(self.collector), 1)

    def test_round_trip_preserves_every_field(self):
        from scriptorium.warnings import WarningCollector

        self.collector.add(
            "image_input_unknown",
            "capability unknown",
            severity="info",
            source="slides.pdf",
            detail="probe timed out",
        )
        restored = WarningCollector.from_dicts(self.collector.to_dicts())
        self.assertEqual(restored.items(), self.collector.items())

    def test_unknown_severity_is_rejected(self):
        from scriptorium.warnings import RunWarning

        with self.assertRaises(ValueError):
            RunWarning(code="x", message="y", severity="fatal")

    def test_summary_counts_by_severity(self):
        self.collector.add("a", "a", severity="error")
        self.collector.add("b", "b", severity="warning")
        self.assertIn("2 warning(s)", self.collector.summary())


class SetupMarkerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(self.root)})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_first_run_is_reported_until_the_marker_exists(self):
        self.assertTrue(is_first_run())
        marker = mark_setup_complete()
        self.assertEqual(marker, setup_marker_path())
        self.assertTrue(marker.is_file())
        self.assertFalse(is_first_run())

    def test_marking_twice_is_idempotent(self):
        mark_setup_complete()
        mark_setup_complete()
        self.assertFalse(is_first_run())

    def test_marker_lives_in_the_default_artifacts_dir(self):
        self.assertEqual(
            setup_marker_path(),
            self.root / "scriptorium" / "setup_complete",
        )


class FirstRunNotesRootTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.missing = self.root / "not-created-yet"
        self._path_patch = mock.patch(
            "scriptorium.config._config_path", return_value=self.root / "config.toml"
        )
        self._path_patch.start()

    def tearDown(self):
        self._path_patch.stop()
        self._tmp.cleanup()

    def test_missing_notes_root_is_rejected_by_default(self):
        with self.assertRaises(ConfigurationError):
            load_config(notes_root=self.missing)

    def test_the_first_run_wizard_can_load_without_a_notes_root(self):
        config = load_config(notes_root=self.missing, require_notes_root=False)
        self.assertEqual(config.notes_root, self.missing)


if __name__ == "__main__":
    unittest.main()
