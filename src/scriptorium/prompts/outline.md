You plan how the knowledge units extracted from a run's sources map onto an
Obsidian vault.
Return a single JSON object. No markdown fences, no commentary.

Course: {{course}}
Target MOC: {{moc_path}}
Flashcard tags for this course: {{course_tags}}

Existing notes (path | title | tags | headings):
{{inventory}}

Related note excerpts:
{{related}}

Knowledge units (id | kind | title | source):
{{units}}

Rules:
- Assign every unit id to exactly one note. Unit ids must be copied exactly.
- Prefer action "append" when the unit belongs under a heading that already exists.
  "anchor" must be an existing heading text in that note, without the leading #.
- Use action "create" only for a coherent concept that deserves its own note and is not
  already covered by an existing note. Use a short Title Case path ending in .md.
  Put new lecture notes at the vault root, matching where course notes already live.
- Never create a note in University/; that folder is reserved for MOCs.
- Do not create two notes for the same concept. Note titles must be unique.
- For new notes, reuse the course flashcard tags listed above. Do not invent deck tags.
- Anything uncertain, contradictory, or unverifiable belongs in "todos", not in a note.
- A typical lecture produces 1 to 4 notes. Never exceed 8.

JSON shape:
{"notes": [{"action": "create", "path": "Note Title.md", "title": "Note Title", "anchor": null, "tags": ["cs/example"], "unit_ids": ["u1", "u2"], "confidence": 0.9}], "todos": [{"kind": "unclear", "text": "What needs checking", "note_path": null}]}
