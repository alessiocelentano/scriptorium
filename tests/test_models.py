import unittest

from scriptorium.models import (
    Flashcard,
    NoteOutline,
    Segment,
    Transcript,
    format_timestamp,
    heading_level_and_text,
)


class TimestampTests(unittest.TestCase):
    def test_human_timestamp(self):
        self.assertEqual(format_timestamp(0), "00:00:00")
        self.assertEqual(format_timestamp(65.4), "00:01:05")
        self.assertEqual(format_timestamp(3725), "01:02:05")

    def test_srt_timestamp(self):
        self.assertEqual(format_timestamp(65.4, srt=True), "00:01:05,400")
        self.assertEqual(format_timestamp(0.9999, srt=True), "00:00:01,000")


class TranscriptTests(unittest.TestCase):
    def test_text_and_srt(self):
        transcript = Transcript(
            segments=[
                Segment(start=0, end=2.5, text="Hello  world"),
                Segment(start=2.5, end=5, text="Second line"),
            ]
        )
        self.assertEqual(transcript.text, "Hello world\nSecond line")
        srt = transcript.to_srt()
        self.assertIn("1\n00:00:00,000 --> 00:00:02,500\nHello world", srt)
        self.assertIn("2\n00:00:02,500 --> 00:00:05,000\nSecond line", srt)

    def test_timed_text(self):
        transcript = Transcript(
            segments=[Segment(start=61, end=62, text="Minute two")]
        )
        self.assertEqual(transcript.timed_text(), "[00:01:01] Minute two")


class ModelTests(unittest.TestCase):
    def test_note_path_normalization(self):
        note = NoteOutline(path="./sub/Note", title="Note")
        self.assertEqual(note.path, "sub/Note.md")
        self.assertEqual(note.title, "Note")

    def test_note_path_rejects_traversal(self):
        with self.assertRaises(ValueError):
            NoteOutline(path="../escape.md", title="x")

    def test_tags_are_normalized(self):
        note = NoteOutline(path="N.md", tags=["#cs/ai", "cs/ai", "language/en"])
        self.assertEqual(note.tags, ["cs/ai", "language/en"])

    def test_heading_parser(self):
        self.assertEqual(heading_level_and_text("## A\\* Search"), (2, "A* Search"))
        self.assertIsNone(heading_level_and_text("plain text"))

    def test_flashcard_kind_fallback(self):
        card = Flashcard(front="Q?", back="A", kind="unknown")  # type: ignore[arg-type]
        self.assertEqual(card.kind, "single")


if __name__ == "__main__":
    unittest.main()
