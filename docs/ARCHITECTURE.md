# Architecture

## Goals

- Produce notes that follow the vault's existing Obsidian conventions without ever
  reformatting what is already there.
- Make every stage inspectable, resumable, and testable in isolation.
- Keep the model on a short leash: JSON contracts in, deterministic writes out.

## Layers

```
cli.py / tui.py
        |
   pipeline.py            orchestration, stage state, review prompts
        |
 planning.py  apply.py  flashcards.py
        |         |          |
 llm/client.py  vault.py   vault.py
        |         |
  prompts/*.md  state.py, log.py
        |
 audio.py, asr.py, context.py
```

`pipeline.py` owns sequencing only. It never contains formatting or card logic.
The TUI runs `Pipeline` in a worker thread and consumes `on_event` messages; the
same `ConsoleUI` interface is adapted to modal prompts, so merge review keeps the
same guarantees as the CLI.

When a non-dry run completes, `history.py` reads the job's `apply.json`,
`05.plan.json` and `cards.json` into one snapshot under
`$SCRIPTORIUM_ARTIFACTS_DIR/history/<job-id>.json`: the generated text mapped to the box
that displayed it (changes and diffs, flashcards, Extras) plus the collected
sources. The snapshot is presentation data, not a resume artifact; the TUI
`/history` window restores it into the panes without replaying the pipeline.

## Data contracts

All contracts are pydantic models in `models.py`:

- `Transcript` / `Segment`: ASR output with timestamps.
- `Unit` / `ExtractResult`: map-phase knowledge units plus `Aside` items for
  off-topic material that must not reach the notes.
- `Outline` / `NoteOutline` / `TodoItem`: reduce-phase placement plan.
- `ComposedNote`: body and flashcards for one planned note.
- `Plan`: the persisted, reviewable artifact in `plan.json`.
- `Flashcard`: one validated card.

The LLM only ever produces the JSON shapes mirroring `ExtractResult`, `Outline`, and
`ComposedNote`. `Plan` is assembled locally and validated before any write.

## Phase design

**Ingest.** Every document source is converted to text by `sources.py` (PDF through
poppler, DOCX/PPTX/XLSX through `zipfile` plus XML, HTML through the standard parser);
an unreadable source becomes a warning, never a failed stage. A PDF is read page by page:
a page with a usable text layer keeps it, including a page that also holds figures, while
a page whose text layer is empty or almost empty is handed to the fallback in
`scanned.py`, which renders it with `pdftoppm` and reads it with the configured model
(`scanned_pdf = "vision"`) or with local Tesseract (`"ocr"`). Page text is cached per
job, so a later run never pays for the same page twice; a missing binary, a rejected
image or a cap on the number of pages is a warning. Every media source is converted to
MP3 and transcribed independently, so a run may contain several recordings or none at
all. `asr.py` exposes one transcription interface with two backends: `LocalTranscriber`
runs the local transformers pipeline for any speech-to-text model, while
`ApiTranscriber` covers hosted providers.
Named backends (`openai`, `groq`, `huggingface`) resolve their endpoint, model and
request style from presets in `config.py`; `custom` takes an explicit base URL and model.
The OpenAI/Groq style posts multipart to `/audio/transcriptions` and reads
`verbose_json` segments (falling back to a single segment when the provider only
returns text); the Hugging Face style posts the raw MP3 to `models/<model>` or to a
dedicated endpoint and reads the `text`/`chunks` response. Files above the provider
upload limit are split with ffmpeg and the chunk timestamps are shifted back onto the
original timeline. Provider credentials come from `GROQ_API_KEY`, `HF_TOKEN` or
`OPENAI_API_KEY`, with `SCRIPTORIUM_ASR_API_KEY` as the override.

**MOC selection.** The MOC is resolved before transcription, so video content only
contributes its file name; document text is already available. `Vault.find_mocs` walks every
entry of `Config.search_paths` recursively and ignores roots outside `notes_root`,
because a MOC must be inside the notes root to be updated. The notes root is always
part of `search_paths` (implicitly, even when `config.toml` does not list it), so the
whole vault is searched. `Vault.match_moc`
scores a candidate from its title similarity and from the overlap between its own body
and the query (course name plus source names plus document text). A weak best score is
reported as `moc_not_found`; two near-equal candidates as `moc_ambiguous`. The decision
in the modal is `use`, `create` or `abort`. When the wanted MOC is not among the
candidates, `m` opens a text box that tab-completes the MOC notes found in the
configured roots; the typed value is resolved as a vault path first, then as a unique
MOC title. Creating is deterministic code, never the
model: the user types the name, `ui.moc_title` guarantees the trailing `MOC`, the file
is created in the first configured root inside the vault (else `University/`, else the
vault root) with a `## Table of Contents` heading, and `moc_created` is recorded.

**Figures.** For PDF sources, `pdf_images.py` runs `pdftohtml -xml -hidden`, keeps the
image bounding boxes and captions and drops the duplicate files poppler emits for soft
masks (deduplicated by content hash). Images stay in the job directory until `apply`,
which copies only the figures referenced by notes that were actually written into
`Config.attachments_root`; a dry run therefore leaves nothing behind. Placement is
decided by the extraction phase (which unit owns a figure) and the compose phase (where
inside that unit to embed it). The model only ever sees captions, page numbers and
nearby text, so a wrong placement is possible and is reported as a warning instead of a
broken `> [!missing]` box.

**Map (extract).** Each source's text is cut at line boundaries into chunks of roughly
6k tokens with a small overlap. Every chunk is sent with its source name and kind and
returns typed units whose `source` carries the file name plus the transcript timestamp
or the document page/slide marker. Units are deduplicated by `(kind, title)`, keeping
the highest-confidence copy. The same call also returns `asides`: comments about
exams, courses, university logistics, useful links, anecdotes and curiosities. They
are deduplicated by text, never become knowledge units, and are persisted in
`03.extract.json` and then in the merged `05.plan.json` so the TUI can show them in the
Extras pane as soon as extraction finishes. They are never written to the vault.

**Group (topics).** Once every source has units, `planning.build_groups` asks the model
to partition the sources into ordered topic groups. The answer is normalized
deterministically: unknown ids are dropped, duplicate ids are resolved to the first
group, sources the model forgot are appended to the last group with a TODO, and a
failure or an empty answer falls back to one group containing everything
(`grouping_fallback`). Only one source needs no model call. The proposed order is kept:
it is the reading order.

**Per-group reduce/compose/apply.** `outline`, `compose`, `apply` and `flashcards` run
once per group, sequentially. The extraction result is filtered to the group's sources
by the `sources` provenance on each unit. The vault context is rebuilt before each
group's outline, so a later group's inventory contains the notes the earlier groups just
wrote. Artifacts are per group (`04.outline.<gid>.json`, `05.plan.<gid>.json`,
`apply.<gid>.json`, `cards.<gid>.json`) and merged into the single `05.plan.json`,
`apply.json` and `cards.json` that the report and the TUI read.

**Reduce (outline).** The union of units plus a compact vault inventory and the course
MOC are sent once. The result assigns every unit to either a new note or an existing
note anchor. Anything doubtful becomes a `TodoItem` instead of a write. The model's
title is then resolved against the vault by `targeting.py`: existing notes are ranked by
title similarity and by heading match, and the chosen candidate decides the note
`action` (`append` without a heading, `merge` with one). `Config.target_policy` says who
chooses: the user (`ask`, a modal listing the candidates plus "create a new note"),
the score (`auto`, only above `target_auto_above` with a `target_min_gap` lead), or
never (`create`). A declined or missing answer creates a note and records the candidates
as a TODO, so nothing is lost and nothing is merged by surprise.

**Compose.** One call per planned note produces the Obsidian body and its flashcards.
The prompt embeds the vault's formatting rules (highlights, callouts, LaTeX, links) and
the card style rules (question fronts, atomic answers, one blank line between cards).

Splitting compose per note keeps outputs under the provider's token limit and limits
blast radius: a malformed note does not invalidate the whole run.

## Write safety

`vault.apply_change(path, new_text, expect_only_additions=True)`:

1. Reads the original bytes.
2. Writes the new bytes atomically.
3. Computes a unified diff against the original.
4. If any line is removed or changed, restores the original and raises.

An **append** keeps `expect_only_additions=True`: the insertion helpers find a heading by
normalized text, compute the end of its section (next heading of the same or higher
level), and insert there, preserving all other bytes including newline style and trailing
newline. A **merge** (`action="merge"`) replaces the whole anchored section, heading
included, so it is written with `expect_only_additions=False`; the composer may edit or
remove existing lines, and the unified diff is what the user approves. The decision is
`Config.merge_policy`: `ask` (default) shows the diff and accepts `y` / `n` / `m`, where
`m` writes a git-style conflict file (`<<<<<<< existing`, `=======`, `>>>>>>> scriptorium`)
into the job directory and leaves the note untouched; `apply` accepts composed merges
without prompting; `manual` always writes the conflict file. A rejected merge is reported
as `skipped_merges`, a manual one as `conflicts`, and neither changes the vault.

New files are created only when the plan says `action="create"` and the path does not
exist. Paths are resolved and must stay inside the vault root.

## Context and retrieval

`context.py` builds:

- course identity and MOC content,
- a note inventory (path, title, tags, headings),
- the top related notes by a lightweight TF-IDF score against the transcript,
- the course tag inferred from notes linked in the MOC.

No embedding model is required. The inventory is capped so the outline prompt stays
within budget.

## State machine

`state.py` stores `state.json` per job with one status per global `Stage`
(`sources`, `prepare`, `audio`, `transcribe`, `extract`, `group`) plus a `GroupState` per
topic group holding the status of the four per-group stages. `completed()` requires every
global stage and every per-group stage to be `done` or `skipped`. Re-running an interrupted
job continues from the first non-done stage. Artifacts are written before the stage is marked done, so
a crash mid-write re-executes the stage rather than trusting partial output.

`--dry-run` runs against `SandboxVault` (`sandbox.py`): reads fall through to the real
vault, writes go to a temporary directory under the system temp dir, and the directory is
deleted when the run ends. The pipeline therefore produces the real diffs, applies the
real insert-only checks and lets group N see group N-1's simulated notes, while the vault
stays byte-identical. A simulated apply is recorded as `skipped` so the next real run
performs the write.

## Cleanup policy

Completed jobs keep `job.json`, `state.json`, `sources.json`, `sources/sNN.txt`,
`sources/sNN/` per-media artifacts, `02.transcript.txt`, `02.transcript.json`,
`05.plan.json`, `apply.json`, `cards.json`, `report.md`, and logs. Audio, SRT, and
intermediate extract/outline files are removed. **Sources are never deleted**; a
recording, slide deck or PDF always stays where the user put it, so a later run can
reuse it.

## TUI input, panes and settings

The input line adds one source at a time.  Select the box with `Enter` (or with
`i` from another box) before typing (its border turns green).  In Vim mode
`Esc` leaves insert mode but keeps the box selected, so normal commands keep
editing the line; a second `Esc` leaves the box.  `Enter` then appends the resolved path to the
Sources pane, clears the editor and returns the box to normal mode, while
`/run` starts the pipeline with the collected paths, `/new` clears the
current session (sources, changes, cards, extras and logs) without deleting the
job directories under `$SCRIPTORIUM_ARTIFACTS_DIR`, and `/wizard` reopens the
setup wizard.  Sources are data owned by `tui.py`,
never parsed from the input line at run time; `d` removes the selected entry.  A
MOC-looking Markdown file is kept as an optional MOC hint instead of a source,
exactly as before.

`vimline.py` is a curses-free, single-line Vim editor ported from
[pivim](https://github.com/alessiocelentano/pivim).  `tui.py` forwards input-line
keys to it and only renders the resulting `text`, `cursor` and `mode`: a block
cursor on the character in normal mode, and the hardware bar cursor at the
insertion point in insert mode.  Keeping the editor pure makes the Vim behaviour
(modal commands, operators, counts, undo/redo, repeat) unit-testable without a
terminal.  `Config.movement` selects the movement model: `vim` keeps modal
editing and disables the arrow keys, `standard` hides the modes and drives the
cursor and focus with the arrow keys.  Editors built by `tui.py` forward the
setting as `arrow_movements`, so the same rendering code serves both styles.

File completion searches an ordered list of roots, `Config.search_paths`; the first
entry resolves relative paths typed in the input line. The notes root is always
appended as an implicit entry, while the first-run wizard and the settings modal
show and write only `Config.user_search_paths` (the paths the user configured).  The TUI settings modal
opens with `s` and edits every `Config` field except the two custom-pane tables
(`tui_panes` and `tui_pane_keys`, which stay TOML-only).  `src/scriptorium/settings.py`
is a declarative schema of sections and fields; `tui.py` renders one section in
a left column and one field per row in the right column, and maps edits back to
typed values when saving.  Fields come in kinds -- free text, masked password,
integer/float with range checks, choice (space/Enter cycles), filesystem path
with directory/file completion, ordered list (search paths), flags (warnings),
and the layout expression -- and saving goes through `write_config_values` into
the machine-local `config.toml` and `Config.with_overrides` for the in-memory
config, so an invalid value (for example a bad `tui_layout`) is rejected with a
message instead of being written.  Settings path fields use the same popup and
Tab-cycling completion as the input line.  Tracked files must never contain
machine paths; `config.toml.example` is the template agents update whenever
configuration grows.

On the very first launch `scriptorium` (the bare TUI entry point) opens a short
setup wizard instead of the normal interface.  `config.py` records a
`setup_complete` marker in the *default* artifacts directory; `cli.py` checks it
before loading the configuration and passes `require_notes_root=False`, because
a freshly copied `config.toml.example` may point at a vault that does not exist
yet.  The wizard (`wizard.py` + the `WizardState` handling in `tui.py`) has four
pages -- vault, model, speech, review -- and deliberately covers only what the
first run needs; every other option stays in the full settings modal (`s`).
Two wizard-only choices map onto real keys: `llm_provider` selects the DeepSeek
default, a local Ollama endpoint or a custom base URL, and `asr_mode` selects
the local pipeline or a remote transcription endpoint.  `Enter` on the review
page creates the notes root if needed, goes through `write_config_values` and
only then records the marker: there is no skip key, so the first run always
writes a working `config.toml`.  A CLI `scriptorium run` never opens curses: on
a first run it prints a one-line hint to start the TUI instead.  Delete the
marker to see the wizard again at startup, or type `/wizard` in the input line
to reopen it from a running session; saving rewrites `config.toml` and re-touches
the marker.

`tui_layout.py` owns the pane topology and geometry.  `Config.tui_layout` is a
tree of `Leaf` and `Split` nodes parsed from `tui_layout` in `config.toml`:
`a | b` is a `cols` split, `a / b` is a `rows` split, parentheses nest, and
`name:N` sets the weight inside the enclosing split (default 1).  A list of row
strings is shorthand for a `rows` split of `cols` rows.  `parse_layout` rejects
names that are unknown, used twice or not placeable, mixed operators at one
level without parentheses, non-positive weights, and a layout without the input
line, which is the only way to start a job.  A registered pane left out of the
layout is simply not shown, so the layout decides which boxes exist.  Weights
are relative to the enclosing split, so `sources:2 / extras:1` gives Sources
twice the height of Extras on the terminal size of the moment.

A box table may declare a spawn key: `Config.tui_pane_keys` overrides the
class's `spawn_key`, and while hidden the pane is pruned from the tree, its
weight goes to its siblings, and a split left with one child collapses into it
(`prune_layout`).  The input box always keeps one content row plus its two
borders (three terminal rows), whatever its weight.  `pane_rects` walks the
pruned tree, allocating each split's cells by weight with the largest-remainder
method and a floor of three rows or four columns per box.  `split_rect` carves a
child pane out of one edge of a parent (`bottom`/`top`/`left`/`right`, with a
collapse rule for tiny boxes) for panes that declare a `parent`, and `neighbor`
maps `h`/`j`/`k`/`l` to the adjacent pane using those rectangles.  Navigation
follows the tiling-manager rule: only panes on the requested side are candidates,
the closest facing edge
wins, and ties are broken by an anchor point carried across moves
(`clamp_point`, `edge_point`).  A direction with no pane still moves the anchor
to that edge, so right then up reaches the upper-right box.  `tui.py` keeps that
anchor while focus changes and resets it on `Tab` or any non-spatial focus
change.  Navigation therefore adapts to any configured layout instead of
following a hard-coded cyclic order.

## Panes: one box, one module

Every box is a self-contained extension.  A pane module imports
`pane_api.py` (the contract), `tui_render.py` (the drawing primitives) and the
data it displays -- never `tui.py` -- so a box can be written, unit-tested and
shipped in one file.  The built-in boxes are `panes/sources.py`,
`panes/changes.py`, `panes/flashcards.py`, `panes/extras.py`, `panes/logs.py`
and `panes/input_pane.py`; `tests/test_panes.py` asserts that no pane module
imports the application.

`pane_api.Pane` defines the box: metadata (`name`, `placeable`, `parent`,
`side`, `transient`, `collapse`, `splits_fullscreen`, `spawn_key`, `is_editor`,
`owns_merge_review`) plus the behaviour hooks `title`, `footer`, `rows`, `draw`,
`handle_key`, `visible`, `split_ratio`, `reset`, the lifecycle hooks
(`on_shown`, `on_log`, `on_asides`) and the selection/scroll hooks
(`item_count`, `selection`, `set_selection`, `toggle_item`, `scroll`,
`set_scroll`, `scroll_limit`).  Only `title` and `rows` are required: the base
class draws the bordered box, maps a selected logical row into the viewport,
and implements the list keys (`j`/`k`, `Enter`, `g`/`G`, `Home`/`End`), so a
read-only box is a handful of lines.  `PaneHost` is a runtime-checkable
`Protocol` documenting the services a pane may use: the shared session state
the pipeline fills in, the drawing surface, and the generic cross-box actions
(`set_focus`, `start_job`, `toggle_pane`, `copy_text`, `open_settings`,
`answer_merge`, ...).  A pane never reaches past that surface, and the host
never branches on a pane name: it finds the editor and the merge-review box by
their capability flags.  The full recipe lives in `docs/CUSTOM_PANES.md`.

`pane_registry.py` resolves a pane name to an import target.  `BUILTIN_PANES`
holds the defaults (`scriptorium.panes.sources:SourcesPane`, ...); the
`[tui_panes]` table in `config.toml` extends or overrides them, so adding a box
never requires editing `tui.py`.  Resolution is eager and cached: an
unimportable module, a class that is not a `Pane`, or a class whose `name` does
not match its configuration key is a `ConfigurationError` raised while the
configuration loads.  `parse_layout` validates every name against the
registry, so a typo cannot produce a half-working interface.

At run time `TuiApp` holds one `PaneRegistry`; `layout_rects` places the
placeable panes from `Config.tui_layout`, dropping the panes whose spawn key has
not been pressed, then labels any remaining carved pane as a child of its parent.
Extras and logs are ordinary placeable panes in the default layout; the logs box
declares `spawn_key = "C-l"`, so it starts hidden.  `_draw` walks the resulting
rectangles and calls `pane.draw(self, rect)`, `Tab` builds the focus ring from
the layout plus any carved panes, and key routing
asks the focused pane first and falls back to the host for focus movement,
overlays and global commands.  Adding a box therefore touches configuration and
one new file -- `tui.py` itself stays unchanged.
