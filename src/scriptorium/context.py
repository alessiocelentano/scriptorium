"""Build the compact vault context used by the planning prompts."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import Config
from .state import slugify
from .vault import NoteInfo, Vault

STOPWORDS = {
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "may", "me", "might", "more", "most", "must", "my", "no", "not", "now", "of", "on",
    "one", "only", "or", "other", "our", "out", "over", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "under", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "why", "will", "with", "would",
    "you", "your",
}

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")


@dataclass
class ContextBundle:
    course: str
    moc_path: str
    moc_text: str
    tags: list[str] = field(default_factory=list)
    inventory_lines: list[str] = field(default_factory=list)
    related: list[tuple[str, str]] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    linked_paths: list[str] = field(default_factory=list)

    def inventory_block(self) -> str:
        return "\n".join(self.inventory_lines)

    def related_block(self) -> str:
        chunks: list[str] = []
        for path, excerpt in self.related:
            chunks.append(f"### {path}\n{excerpt}")
        return "\n\n".join(chunks)


def load_glossary(vault: Vault, course: str) -> str:
    slug = slugify(course)
    candidates = [
        vault.root / "Meta" / "glossaries" / f"{slug}.md",
        vault.root / "Meta" / "glossaries" / f"{course.strip()}.md",
        vault.root / "Meta" / "Glossaries" / f"{slug}.md",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                continue
    return ""


def moc_linked_titles(moc_text: str) -> list[str]:
    titles: list[str] = []
    for match in _WIKILINK_RE.findall(moc_text):
        title = match.strip()
        if title and not title.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".pdf")):
            if title not in titles:
                titles.append(title)
    return titles


def infer_course_tags(
    vault: Vault, linked_titles: list[str], course: str
) -> list[str]:
    by_title = {note.title.casefold(): note for note in vault.inventory()}
    counts: Counter[str] = Counter()
    for title in linked_titles:
        note = by_title.get(title.casefold())
        if note is None:
            continue
        for tag in note.tags:
            if _is_deck_tag(tag):
                counts[tag] += 1

    if not counts:
        course_tokens = _tokens(course)
        for note in vault.inventory():
            if not note.has_flashcard_tag:
                continue
            if course_tokens & _tokens(note.title):
                for tag in note.tags:
                    if _is_deck_tag(tag):
                        counts[tag] += 1

    return [tag for tag, _ in counts.most_common(4)]


def _is_deck_tag(tag: str) -> bool:
    return (
        tag == "cs"
        or tag.startswith("cs/")
        or tag == "language"
        or tag.startswith("language/")
    )


def build_context(
    vault: Vault,
    course: str,
    moc_path: str,
    text: str,
    config: Config,
    *,
    max_inventory: int = 90,
) -> ContextBundle:
    notes = vault.inventory()
    moc_text = vault.read(moc_path) if moc_path else ""
    linked_titles = moc_linked_titles(moc_text)
    linked_set = {title.casefold() for title in linked_titles}

    query_tokens = _top_tokens(text, limit=180)
    query_set = set(query_tokens)
    idf = _idf(notes)

    def score(note: NoteInfo) -> float:
        title_tokens = _tokens(note.title)
        heading_tokens = _tokens(" ".join(note.headings))
        value = sum(idf.get(token, 1.0) for token in query_set & title_tokens) * 2.5
        value += sum(idf.get(token, 1.0) for token in query_set & heading_tokens)
        if note.title.casefold() in linked_set:
            value += 6.0
        return value

    ranked = sorted(notes, key=score, reverse=True)
    tags = infer_course_tags(vault, linked_titles, course)

    selected: list[NoteInfo] = []
    seen: set[str] = set()

    def add(note: NoteInfo) -> None:
        if note.path not in seen and len(selected) < max_inventory:
            seen.add(note.path)
            selected.append(note)

    for note in notes:
        if note.title.casefold() in linked_set:
            add(note)
    for note in ranked[:40]:
        add(note)
    for note in notes:
        if tags and any(tag in note.tags for tag in tags):
            add(note)

    inventory_lines = [_inventory_line(note) for note in selected]

    by_path = {note.path: note for note in notes}
    related: list[tuple[str, str]] = []
    for note in ranked[:8]:
        try:
            text = vault.read(note.path)
        except OSError:
            continue
        excerpt = _excerpt(text, config.extra_context_chars)
        if excerpt:
            related.append((note.path, excerpt))

    return ContextBundle(
        course=course,
        moc_path=moc_path,
        moc_text=moc_text,
        tags=tags,
        inventory_lines=inventory_lines,
        related=related,
        titles=sorted(by_path[note.path].title for note in selected),
        keywords=query_tokens[:60],
        linked_paths=[note.path for note in notes if note.title.casefold() in linked_set],
    )


def _inventory_line(note: NoteInfo) -> str:
    tags = ", ".join(note.tags) if note.tags else "-"
    headings = "; ".join(note.headings[:8]) if note.headings else "-"
    return f"{note.path} | {note.title} | {tags} | {headings}"


def _excerpt(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0].strip() + "\n[...]"


def _top_tokens(text: str, limit: int) -> list[str]:
    counts = Counter(_tokens_with_counts(text))
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [token for token, _ in ordered[:limit]]


def _tokens_with_counts(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in STOPWORDS
    ]


def _tokens(text: str) -> set[str]:
    return set(_tokens_with_counts(text))


def _idf(notes: list[NoteInfo]) -> dict[str, float]:
    total = max(1, len(notes))
    document_frequency: Counter[str] = Counter()
    for note in notes:
        for token in _tokens(note.title + " " + " ".join(note.headings)):
            document_frequency[token] += 1
    return {
        token: math.log((1 + total) / (1 + count)) + 1.0
        for token, count in document_frequency.items()
    }
