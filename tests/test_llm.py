import unittest
from unittest import mock

from scriptorium.errors import LLMError
from scriptorium.llm.client import parse_json_object


class JsonParsingTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prose_wrapped(self):
        self.assertEqual(
            parse_json_object('Here is the result: {"a": {"b": 2}} done.'),
            {"a": {"b": 2}},
        )

    def test_braces_inside_strings(self):
        self.assertEqual(
            parse_json_object('{"a": "text { with } braces"}'), {"a": "text { with } braces"}
        )

    def test_rejects_non_object(self):
        with self.assertRaises(LLMError):
            parse_json_object("[1, 2]")

    def test_rejects_garbage(self):
        with self.assertRaises(LLMError):
            parse_json_object("no json here")


if __name__ == "__main__":
    unittest.main()


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        from scriptorium.config import Config
        from scriptorium.log import Logger

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = Config(
            notes_root=self.root,
            artifacts_dir=self.root / "home",
            image_input="auto",
        )
        self.logger = Logger()

    def tearDown(self):
        self._tmp.cleanup()

    def client(self, **overrides):
        from scriptorium.llm.client import DeepSeekClient

        config = self.config.with_overrides(**overrides)
        return DeepSeekClient(config, self.logger, api_key="test-key")

    def test_settings_override_the_probe(self):
        self.assertIs(self.client(image_input="enabled").image_input_supported(), True)
        self.assertIs(self.client(image_input="disabled").image_input_supported(), False)

    def test_cached_answer_skips_the_probe(self):
        config = self.config.with_overrides(image_input="auto")
        self.config = config
        cache_dir = self.root / "home" / "cache"
        cache_dir.mkdir(parents=True)
        key = f"{config.llm_base_url}|{config.llm_model}"
        (cache_dir / "capabilities.json").write_text(f'{{"{key}": false}}', encoding="utf-8")
        self.assertIs(self.client().image_input_supported(), False)

    def test_probe_result_is_cached(self):
        client = self.client()
        with mock.patch.object(client, "_probe_image_support", return_value=True) as probe:
            self.assertIs(client.image_input_supported(), True)
            self.assertIs(client.image_input_supported(), True)
        probe.assert_called_once()
        self.assertTrue((self.root / "home" / "cache" / "capabilities.json").is_file())

    def test_inconclusive_probe_is_not_cached(self):
        client = self.client()
        with mock.patch.object(client, "_probe_image_support", return_value=None):
            self.assertIsNone(client.image_input_supported())
        self.assertFalse((self.root / "home" / "cache" / "capabilities.json").exists())

    def test_json_mode_sends_response_format(self):
        client = self.client()
        response = {"choices": [{"message": {"content": "{}"}}], "usage": {}}
        with mock.patch.object(client, "_post", return_value=response) as post:
            client.chat_json("system", "user")
        payload = post.call_args.args[0]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["model"], client.config.llm_model)

    def test_json_mode_none_omits_response_format(self):
        client = self.client(llm_json_mode="none")
        response = {"choices": [{"message": {"content": "{}"}}], "usage": {}}
        with mock.patch.object(client, "_post", return_value=response) as post:
            client.chat_json("system", "user")
        self.assertNotIn("response_format", post.call_args.args[0])
