You organize the sources of one study run into ordered topic groups.
Return a single JSON object. No markdown fences, no commentary.

Course: {{course}}

Sources (id | file name | kind | beginning of the text):
{{sources}}

Knowledge units already extracted, per source:
{{units}}

Rules:
- Every source id appears in exactly one group. Copy the ids exactly. Never invent an id
  and never drop a source.
- A group is the set of sources that cover one macro-topic and should be studied
  together. Sources that overlap belong to the same group even when one of them also has
  material the others do not.
- Sources on unrelated topics belong to different groups, also when they are part of the
  same course.
- Order the groups so a reader can follow them: prerequisites and foundations first, then
  the material that builds on them. When the order does not matter, keep the input order.
- Use a short Title Case phrase as "topic". Put the ordering reason in "rationale".
- Prefer few groups: use 1 group when the sources share one topic. Never create a group
  without sources and never exceed 6 groups.
- Anything you are unsure about belongs in "todos", not in a new group.

JSON shape:
{"groups": [{"id": "g1", "topic": "Basics", "source_ids": ["s01", "s02"], "rationale": "Foundational definitions used by every later topic.", "confidence": 0.9}], "todos": [{"kind": "unclear", "text": "What needs checking", "note_path": null}]}
