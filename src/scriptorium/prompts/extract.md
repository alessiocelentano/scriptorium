You extract structured knowledge from one chunk of source material: either a
lecture transcript or an excerpt of a document (slides, notes, textbook).
Return a single JSON object. No markdown fences, no commentary.

Course: {{course}}
Source: {{source_name}} ({{source_kind}})
Topics already covered: {{topics}}

Source chunk:

{{transcript}}

Figures extracted from this chunk's pages (use these exact file names):
{{images}}

Rules:
- Emit one unit per coherent idea, not one per sentence. Merge repetitions.
- "content" must be self-contained and faithful to the source. Keep concrete
  examples, numbers, formulas, and distinctions. Preserve LaTeX as $...$ or $$...$$.
- "source" is the location marker of the first relevant line: a transcript
  timestamp such as 00:12:30, or a document marker such as [page 4] or
  [slide 12]. Copy an existing marker. Never invent one and never put the file
  name there; the application prefixes the file name itself.
- "title" is a short Title Case noun phrase; it must be unique enough to become a note
  heading.
- "kind" is one of: definition, example, formula, exercise, remark, term, procedure,
  comparison, concept.
- "terms" lists the key terms worth highlighting or carding.
- "images" lists the file names of figures that belong to this unit, copied exactly
  from the figure list above. Leave it empty rather than inventing a name. An image
  belongs to exactly one unit: the one whose explanation it illustrates.
- "confidence" is between 0 and 1. Use a low value when the audio is unclear or the
  document is fragmentary; never invent content to fill a gap.
- Emit at most {{max_units}} units. Prefer the most substantive ones.
- Keep material that does not belong in a study note out of "units". That includes
  comments about exams, grading, the course, the university, deadlines, logistics,
  recommended books, useful links, personal anecdotes, jokes, and curiosities. Put
  each such item in "asides" instead.
- "asides" is a list of short, self-contained items. "kind" is one of exam,
  university, lesson, link, curiosity, advice, logistics, remark. "text" preserves
  the useful detail (a URL, a date, a room, a warning) without adding commentary.
  "source" follows the same location-marker rule as a unit. Never move an aside into
  a unit and never drop it because it is off-topic.

JSON shape:
{"units": [{"kind": "definition", "title": "Title", "content": "Text", "source": "00:00:00", "terms": ["term"], "images": [], "confidence": 0.9}], "asides": [{"kind": "exam", "text": "The exam is oral and covers chapters 1-4.", "source": "00:12:30"}], "topics": ["Topic"]}
