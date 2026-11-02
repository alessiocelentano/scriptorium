# AGENTS.md

Instructions for coding agents working in this repository.

## What this project is

`scriptorium` is a local pipeline: video lecture -> MP3 -> local speech-to-text transcript ->
DeepSeek plan -> Obsidian notes -> draft flashcards. Read `README.md` and
`docs/ARCHITECTURE.md` before changing behavior.

## Hard invariants

1. **The LLM never writes files.** Models return JSON validated by `models.py` and applied
   by `apply.py`. Never add code that executes model-provided paths or shell commands.
2. **Appends are insert-only; merges are diff-reviewed.** `vault.apply_change()` refuses
   any append whose unified diff removes or changes a pre-existing line, and restores the
   original bytes. A merge (`action="merge"`) may rewrite the anchored section only
   through `merge_policy`: `ask` (default) shows the unified diff and accepts the write,
   skips it, or writes a git-style conflict file into the job directory for the user to
   reconcile by hand; `manual` never writes the vault; `apply` is the explicit opt-in.
   Never write a merge outside that review path.
3. **New notes are automatic; merges require review.** `merge_policy` and
   `--accept-merges` decide; `--accept-merges` applies pending merges without an
   interactive answer.
4. **Flashcards are always drafts.** Never emit an active card and never delete a
   `<!--SR:...-->` scheduling comment. A skill that lacks `add-drafts` (or fails for
   any reason) falls back to internal draft insertion; a skill that writes a non-draft
   line is reverted and those cards are written internally instead.
5. **Sources are never deleted.** Recordings, slides, notes and PDFs stay where the user
   put them; only intermediate artifacts (the MP3, the SRT) are cleaned up. Never add
   code that removes a source path.
6. **Keep the vault clean.** Job artifacts stay in `$SCRIPTORIUM_HOME`, never inside the
   vault, except the notes the user asked for and the image attachments those notes
   embed.
7. **Settings are machine-local.** Personal paths, the vault location, completion roots,
   model names and similar values live in the gitignored `config.toml` (repo-local in a
   source checkout) or `$XDG_CONFIG_HOME/scriptorium/config.toml`; never hard-code them
   in tracked code, tests, docs or prompts. `config.toml.example` is the tracked template
   and must be updated whenever `config.py` gains a setting. A new setting needs: a
   `Config` field, a resolver in `load_config`, a line in `config.toml.example`, and a
   mention in `README.md`.

## Layout

```
src/scriptorium/
  cli.py          argparse entry point and command wiring
  config.py       TOML + environment configuration (machine-local settings)
  tui.py          curses application (layout, keys, worker events)
  tui_layout.py   configurable pane topology, geometry and spatial navigation
  tui_render.py   drawing primitives (styles, text helpers, the box painter)
  pane_api.py     the pane contract: one box is one Pane subclass
  pane_registry.py  pane name -> import target, resolved from [tui_panes]
  panes/          the built-in boxes, one self-contained module each
  vimline.py      single-line Vim editor used by the input line and settings
  settings.py     declarative schema for the full in-app settings modal
  wizard.py       first-run setup pages and their mapping onto Config keys
  tui_completion.py  recursive filename tab completion over one or more roots
  tui_input.py    input line -> source list + optional MOC hint
  tui_data.py     apply.json and plan/artifact display models
  slash_commands.py  the input line's slash commands and descriptions
  history.py      completed-run snapshots restored by /history
  tui_prompt.py   ConsoleUI adapter that routes prompts into the TUI loop
  models.py       pydantic contracts shared by LLM phases and the applier
  sources.py      document readers (PDF, DOCX, PPTX, XLSX, HTML, text)
  scanned.py      PDF pages without a text layer (vision model or Tesseract)
  targeting.py    ranked merge/create candidates and the target policy
  prompting.py    prompt file loading and {{placeholder}} rendering
  pdf_images.py   PDF figure extraction, captions, vault publishing
  sandbox.py      copy-on-write vault overlay used by --dry-run
  warnings.py     run warnings as data, filterable and UI-independent
  state.py        job store and resumable stage state
  log.py          console + JSONL logging
  ui.py           interactive console prompts, shared with the TUI adapter
  audio.py        ffmpeg/ffprobe wrapper
  asr.py          lazy transformers speech-to-text pipeline
  vault.py        path safety, inventory, MOC matching, insert-only writes
  context.py      retrieval context for prompts (built from all source text)
  llm/client.py   DeepSeek chat-completions client with JSON mode
  planning.py     extract -> outline -> compose phases
  apply.py        deterministic plan applier
  flashcards.py   card validation, rendering, draft insertion
  pipeline.py     stage orchestration
  prompts/        prompts as Markdown with {{placeholders}}
```

## Commands

```bash
./bin/scriptorium                 # opens the TUI (default)
./bin/scriptorium run <video> --course "<course>" --dry-run
bash bin/scriptorium              # when the executable bit is not set yet
python3 -m unittest discover -s tests -v
```

After cloning, run `chmod +x bin/scriptorium` once; git records the mode.
Copy `config.toml.example` to `config.toml` and edit the machine-local values
(vault path, completion roots, model names). `config.toml` is gitignored.

## Conventions

- Python 3.11+, standard library first. `pydantic` is the only runtime dependency;
  `torch`/`transformers` are imported lazily inside `asr.py`.
- Dataclasses or pydantic models for data crossing module boundaries. No raw dicts in
  business logic.
- Pure functions take paths and strings; file I/O stays in `vault.py`, `state.py`, and
  `flashcards.py`.
- Every stage must be idempotent: re-running a completed job skips finished stages.
- Add a unit test for every change to validation, insertion, or card rendering.
- Do not add a dependency if the standard library can do the job.
- Keep prompts in `prompts/*.md`, never inline long prompts in Python.

## Extending

- New LLM phase: add a prompt file, a pydantic model in `models.py`, a function in
  `planning.py`, a stage in `state.Stage`, and wire it in `pipeline.py`.
- New output backend (for example Anki export): implement it behind a function that takes
  validated cards; do not fork the card model.
- TUI: subscribe to `pipeline.Pipeline` events through the `on_event` callback; the CLI and
  the TUI must call the same pipeline, not duplicate stage logic.
- TUI editing: route input-line keys through `vimline.VimLineEditor`; do not add ad-hoc
  cursor or text-editing code to `tui.py`. The editor is curses-free and unit-tested, and
  `tui.py` only draws the block/insert cursor at the offset returned by the editor.
- TUI settings: file-completion roots are an ordered list (`completion_root` plus
  `completion_roots`). Read them through `Config.search_roots` and persist them with
  `config.write_config_values`.
- New TUI box: write one module under `panes/` exposing a `pane_api.Pane` subclass, then
  register it in `config.toml` under `[tui_panes]` and place it in `tui_layout`. Do not add
  a branch on a pane name in `tui.py`: a pane draws its own rectangle and handles its own
  keys through the `PaneHost` services. The full contract lives in
  `docs/CUSTOM_PANES.md` (the public import surface, the hook table, the `PaneHost`
  service catalogue, capabilities and lifecycle hooks) — read it before writing a pane.
  `tests/test_panes.py` enforces that a pane module never imports `tui.py` and that
  `tui.py` never branches on a hard-coded pane name.

## Testing notes

- Tests use `unittest`, temporary directories, and never call the network, ffmpeg, or
  transformers.
- Mock the LLM client with `FakeLLM` (see `tests/test_planning.py`).
- When testing insertion, assert both the new content and byte equality of untouched
  regions.
