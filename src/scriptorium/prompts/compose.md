You write one section of an Obsidian note for a university course vault. The
knowledge units may come from a lecture recording, slides, notes or a textbook.
Return a single JSON object. No markdown fences, no commentary outside the JSON.

Course: {{course}}
Note path: {{note_path}}
Action: {{action}}
Anchor heading: {{anchor}}
Note tags: {{tags}}
Allowed wikilink targets: {{allowed_links}}

Existing section content (empty for a new note):
{{existing}}

Figures available for these units (exact names):
{{images}}

Knowledge units assigned to this note:
{{units}}

Body rules:
- Return Markdown only in "body". No YAML frontmatter and no H1 repeating the note title.
- Use ## for top-level sections and ### for nesting. Title Case. No numbering.
- Use callouts for structure: > [!tip] Definition, > [!example] Example: short title,
  > [!info] title, > [!note], > [!warning], > [!question] Exercise. Put a blank ">"
  line between paragraphs inside a callout.
- Math is LaTeX only: $...$ inline and $$...$$ display on their own lines. Never Unicode
  math symbols.
- Wikilinks: use [[Target]] only when Target appears in the allowed list. Never link to a
  note that might not exist. Link each concept on first mention.
- Use tabs for nested list indentation.
- Embed every figure listed above exactly once, on its own line, as ![[name]] with the
  exact name. Place it right after the paragraph or callout it illustrates. Never write
  an embed for a name that is not in the list and never add an "image missing" box.
- Stay faithful to the units. Do not add facts that are not supported. If a clarification
  is genuinely needed, put it in "> [!info]+ Clarification (added)".
- For an append action, continue the anchored section from where it stops. Do not repeat
  its existing text and do not restate the heading.
- For a merge action, "Existing section content" is the whole current section, heading
  included. Return the complete replacement of that section, starting with its heading
  line at the same level. You may edit, reorder or delete existing lines when the units
  supersede them, but keep every fact whose meaning is unchanged and never delete a line
  you cannot justify from the units. Do not add a heading for a section that exists.
- Avoid em dashes and semicolons if they are not strictly necessary.

Flashcard rules:
- Every front is a real question ending with "?". Never a bare topic or a completion stem.
- Answers are atomic and concise. Add a reason clause only when the reason is the point.
- Never compare two things in one card. Split a comparison into one card per side.
- Keep enumerations on one line separated by semicolons unless items are long.
- Formula cards ask about the property in words; use symbols only when unavoidable.
- Do not generate numeric compute cards.
- kind must be "single" (front ;;; back), "reversed" (only for a central named term),
  "multi" (only for code, long lists, or multi-step reasoning), or "cloze" (front contains
  {{c1::...}}).
- Never use "-->", ";;;", or "???" inside front or back text.
- Cover the assigned units exhaustively but without padding.
- When the source marks something as uncertain, keep [REVIEW: <marker>] at the end of the
  answer.

JSON shape:
{"body": "## Heading\n...", "flashcards": [{"front": "What is X?", "back": "Y.", "kind": "single"}]}
