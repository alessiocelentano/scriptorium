# Manage flashcards

Bundled flashcard skill used by `scriptorium`. It writes draft cards into an
Obsidian note and never activates or deletes a card.

## Contract

The pipeline invokes the skill as:

```
<script interpreter> manage_flashcards.py add-drafts --file /path/to/note.md
```

It reads the rendered cards from standard input and inserts them as commented
draft lines under the note's `## Flashcards` heading:

```
<!--DRAFT: front ;;; back -->
```

The note is changed by insertion only. Existing lines, including any Obsidian
Spaced Repetition `<!--SR:...-->` comment, are preserved byte for byte.

## Customizing

`flashcard_skill` in `config.toml` can point at this script or at your own copy.
A custom skill should accept the same `add-drafts --file` command and stdin
contract, and must not modify existing lines. If it does not support that
command, `scriptorium` inserts the draft cards itself.
