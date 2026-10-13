# rutts

Local Russian narration previews on Apple Silicon. Engines: `qwen-mlx` (Qwen3-TTS, Apache-2.0), `fish-mlx` (Fish Audio S2 Pro 8-bit community conversion, non-commercial research license) and `voxcpm2` (VoxCPM2 MLX port, Apache-2.0; the official PyTorch MPS path is not used).

Runbook for one book, offline, from a clean checkout, with the acceptance record: [docs/runbook.md](docs/runbook.md).

## Setup

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[mlx,dev]"
```

## Commands

Create a narrator profile from a clean Russian reference recording (3-60 s) and its transcript:

```sh
.venv/bin/rutts narrator create --name "Narrator" --audio ref.wav \
  --transcript-file ref.txt --provenance "Recorded by me, 2026" --out profiles/narrator
```

Render previews (use `--suite` for the bundled evaluation excerpts, or `--text`, `--text-file`, `--excerpts`):

```sh
.venv/bin/rutts preview --engine qwen-mlx --precision bf16 --narrator profiles/narrator \
  --out previews/run1 --suite --retries 1 --offline
```

Results are written to `results.json` and `results.md` in the output directory. Each attempt is kept as `<excerpt_id>/attempt-N.wav`.

Other engines use the same command, for example `--engine fish-mlx` (precision must be `8bit`) or `--engine voxcpm2 --precision bf16|8bit`. Precision is validated per engine; an unsupported value exits with code 2.

Before a full suite, run the one-sentence memory smoke test for a single engine:

```sh
.venv/bin/rutts smoke --engine fish-mlx --narrator profiles/narrator --out smokes/fish --offline
```

It loads one model, renders one fixed Russian sentence, and writes `smoke.json` (status, `feasible`, error, timings, resource measurements) plus `smoke.wav` on success. A failed load or render still writes `smoke.json` and exits with code 1. No other engine or network service is used as a fallback.

## Book workflow

A book project is a self-contained directory: `source/original.txt` (byte-identical input), `source/source.json` (chapters and segments with line locations), `narration/narration.json` (narration text linked to source segment ids), `narrator/` (copied profile), `chunks/<chapter>/<target>/` (plan, manifest and lossless chunk WAVs) and `chapters/<chapter>.wav` with `chapters/<chapter>.json` (assembly and coverage report).

```sh
.venv/bin/rutts book import novel.txt --out books/novel --title "Novel" --author "Author"
.venv/bin/rutts book show books/novel
.venv/bin/rutts book show books/novel --chapter c002
.venv/bin/rutts book split books/novel --at c001-s0004 --title "Evening"
.venv/bin/rutts book exclude books/novel --segment c002-s0003 --reason "footnote"
.venv/bin/rutts book narrator books/novel --profile profiles/narrator
.venv/bin/rutts book plan books/novel --chapter c001 --target short
.venv/bin/rutts book render books/novel --chapter c001 --engine qwen-mlx --precision bf16 --target short --offline
.venv/bin/rutts book status books/novel --chapter c001 --target short
.venv/bin/rutts book assemble books/novel --chapter c001 --target short
.venv/bin/rutts book compare-targets books/novel --chapter c001
```

- Chapter markers are lines starting with `Глава` or `Часть` (also `ГЛАВА` and `ЧАСТЬ`) followed by a number or title, markdown `# Title` lines, and lines matching `--chapter-regex`. Without markers the whole file is one chapter.
- `plan` durations are estimates at 14 characters per second. `compare-targets` reports durations measured from the rendered audio.
- `render` resumes: each chunk is saved atomically, and a rerun reuses only chunks whose text, engine, narrator, audio hash and sample rate still match. Invalid chunks are regenerated and their reason is kept in the record's `history` (`missing_audio`, `corrupt_audio`, `text_changed`, `config_changed`). Ctrl-C leaves saved chunks intact and exits with code 130.
- A changed engine or narrator exits with code 2 unless `--accept-config-change` is given; the flag regenerates the chapter and moves old audio to `chunks/<chapter>/<target>/superseded/`. `--force-new-run` archives the whole chapter to `chunks/<chapter>.prev-<n>/` instead.
- `render` retries a failed chunk up to `--retries` times (default 2). Failed attempts keep their error, exception type, timing and resources; invalid audio is kept as `<chunk>.attempt-<n>.failed.wav`. A chunk that still fails is flagged `needs_review`, the other chunks continue, and the run exits with code 1. Use `--retry-failed` on a later run to try flagged chunks again.
- `render` exits with code 1 on a fatal runtime error (for example out of memory), leaving the project resumable. `assemble` writes `<chapter>.incomplete.wav` and exits with code 1 when chunks are missing or failed, printing each one.
- `book status` reads progress from disk: completed, reused, remaining and failed chunks, elapsed and cumulative time, recent failures and chunks needing review. Run state is in `chunks/<chapter>/<target>/run.json`.

## Full-book generation

`book generate` renders every included chapter in source order with one engine instance, loaded once for the whole job and unloaded at the end. Its state lives in `job/` inside the project.

```sh
.venv/bin/rutts book generate books/novel --engine qwen-mlx --precision bf16 --target short --offline --assemble
.venv/bin/rutts book generate books/novel --engine qwen-mlx --chapters c002,c003
.venv/bin/rutts book generate books/novel --engine qwen-mlx --from c002 --retry-failed
.venv/bin/rutts book job-status books/novel
.venv/bin/rutts book check-portable books/novel
```

- The first run freezes the engine config, narrator hash, target, chunk limits, retry budget and resource thresholds in `job/job.json`. A later run with a different engine, precision, narrator or target exits with code 2 and names each differing field; `--accept-config-change` supersedes the old configuration (kept in `superseded_configs`) and regenerates the affected chunks.
- Chapters that are already valid are verified by fingerprint and not re-rendered; a fully valid book never loads the engine. Excluded chapters are skipped and listed with their reasons.
- `job/state.json` (rewritten after every chunk) holds the status (`running`, `completed`, `incomplete`, `interrupted`, `stopped_resource_pressure`, `failed`), per-chapter counts and failed chunk ids, cumulative elapsed time, the last 10 failures, observed throughput and the resource summary. A time-left estimate appears only after three chunks have been rendered in the job, computed from measured render seconds per character.
- `job/coverage.json` is rebuilt at the end of every invocation: each source segment with its inclusion or exclusion reason, covering chunk ids, their status and audio hash, totals, the uncovered segments and `complete`.
- Each chunk is preceded and followed by a resource sample (`job/resources.jsonl`). Swap growth since the job started above `--max-swap-gb` (default 4), free memory below `--min-free-percent` (default 5) on two consecutive samples, or peak RSS above `--max-rss-gb` (default 13) stops the job before the next chunk with exit code 3 and writes `job/diagnostics/<timestamp>.json`. Metrics that cannot be read are logged as `unavailable` and never stop the job. An engine memory error also stops with exit code 3.
- Ctrl-C leaves the project resumable with state `interrupted` and exit code 130. Re-running the same command continues from the first chunk that is not valid, across chapter boundaries.
- Exit codes: 0 complete, 1 incomplete (failed chunks are listed; the others continue), 2 refused, 3 stopped for resource pressure, 130 interrupted.
- `book check-portable` verifies that no project file stores an absolute path or a path outside the project. Model weights are referenced by repository id and revision only, so a moved project resumes without re-rendering.

## EPUB import

`rutts book import book.epub --out book/ --footnotes omit|chapter-end|inline` creates a book project from an EPUB.
Use `rutts book preview-import book.epub --footnotes P` first: it prints the spine order, each document's handling
(narrated, or excluded with a reason such as `cover`, `navigation`, `front_matter`, `non_linear_spine_item`,
`unreadable_document`) and the counts of paragraphs, scene breaks, illustrations and footnotes. It writes nothing.

- Reading order follows the spine; titles come from the EPUB 3 nav document, else the EPUB 2 NCX, else the first heading.
- A document whose nav entries point at several anchors is split into one chapter per anchor.
- Illustrations, tables, scripts and media are kept as excluded `auxiliary` segments with their reason and alt text.
- Footnote policies: `omit` (default) excludes note bodies (`footnote_omitted`); `chapter-end` and `inline` read each
  note once, prefixed `Примечание N.`, and exclude the original (`footnotes_relocated`). The reference marker is never
  spoken. Change the policy later with `rutts book footnotes book/ --policy P`; this re-derives the source from
  `source/original.epub`, so boundary corrections made before the change must be re-applied.
- `source/original.epub` is kept byte-identical. `source/import-report.json` records the spine, footnote relocations
  and every exclusion. Exit code 1 means a document could not be read or contained unsupported content (the rest is imported);
  exit code 2 means the input is not an EPUB, is DRM-protected, lacks its package document, or has no Russian prose.
- Only the Python standard library is used for parsing; nothing is fetched from the network.

## Pronunciation corrections

Narration text is derived: source text plus ordered corrections. `source/source.json` and `source/original.txt` are never edited. Corrections live in `narration/dictionary.json` (per-book patterns), `narration/overrides.json` (one exact span of one source segment, guarded by its original text), `narration/markup.json` (per-engine stress-hint convention) and `narration/history.json` (append-only).

```sh
.venv/bin/rutts book dict add books/novel --pattern "замок" --replacement "з+амок" --kind stress --yes
.venv/bin/rutts book override add books/novel --segment c001-s0002 --find "замок" --occurrence 2 --replacement "зам+ок" --yes
.venv/bin/rutts book diff books/novel --chapter c001 --pending-file changes.json
.venv/bin/rutts book history books/novel
.venv/bin/rutts book revert books/novel 2
.venv/bin/rutts book markup-test books/novel --engine qwen-mlx --narrator profiles/narrator --out review/markup
.venv/bin/rutts book markup-set books/novel --engine qwen-mlx --convention plus --verified-by-listening
.venv/bin/rutts book regenerate books/novel --chapter c001 --engine qwen-mlx --target short
```

- `dict add` and `override add` print the source-to-narration diff and the chunks it would invalidate, then require `--yes` (or an interactive yes). `--pending-file` previews a JSON list of proposed entries without applying them.
- A stress hint is a `+` directly before the stressed vowel (`з+амок`). It is accepted only with the engine's convention: `none` strips hints (and warns), `plus` keeps `+`, `acute` writes U+0301 after the vowel. Only conventions enabled with `markup-set --verified-by-listening` are used; all engines start at `none`. Unsupported markup (angle brackets, several stress marks in a word, a `+` without a following vowel) is rejected when the correction is added.
- Each chunk is fingerprinted on the text the engine receives, so a correction re-renders only the chunks whose spoken text changed. `status` reports a chapter master as STALE until `assemble` rebuilds it.
- `regenerate` renders only invalidated chunks and writes `review/joins/<chunk>.wav`: previous chunk, pause, regenerated chunk, pause, next chunk.
- An override whose source text no longer matches is reported as stale and not applied. Splits and merges keep overrides attached to their source segment.

## Stress and ё suggestions

An optional local assistant (RUAccent, https://github.com/Den4ikAI/ruaccent) proposes stress and ё changes. Nothing is applied until you accept it.

Install (not part of the default install; the adapter is written against RUAccent's documented API and has not been verified against the real package):

```bash
.venv/bin/pip install ruaccent
```

```bash
rutts book suggest book/ --chapter c001 [--stress-scope homographs|all] [--yo on|off]
rutts book suggestions book/ --chapter c001 [--state proposed|accepted|rejected|all] [--kind yo|stress]
rutts book suggestion listen book/ c001-g001 --engine qwen-mlx   # A/B clips in narration/suggestions/listen/
rutts book suggestion accept book/ c001-g001 --yes               # or --all-yo --yes
rutts book suggestion reject book/ c001-g002
rutts book suggestion revert book/ c001-g001
```

- Suggestions are stored in `narration/suggestions/<chapter>.json` with the assistant's provenance. Original text is never changed.
- `--stress-scope homographs` (default) only suggests stress for words in `src/rutts/data/homographs_ru.json`.
- Words covered by a dictionary entry or an occurrence override are never suggested.
- Accepting creates an ordinary override (visible in `rutts book history` and `rutts book diff`); reverting removes it.
- Stress hints are stored as `+` and are removed before synthesis for any engine without a verified convention. Check with `rutts book markup-test` and enable with `rutts book markup-set` only after listening. ё suggestions take effect immediately.
- The assistant is loaded only by `rutts book suggest` and unloaded afterwards; ordinary render and verify never import it.

## Verification

Verification flags possible narration errors in rendered chunks. It runs in its own process, loads only the Whisper transcriber (`mlx-community/whisper-large-v3-turbo`, revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`, language `ru`) and unloads it when done.

```sh
.venv/bin/rutts book verify books/novel --chapter c001 --target short --offline
.venv/bin/rutts book findings books/novel --chapter c001 --target short --chunk c001-k0002
```

- Findings are written to `review/findings/<chapter>/<target>/<chunk>.json` (expected text, raw transcript with word times, normalised comparison, findings) and summarised in `report.md`. Each result is tied to the chunk's `audio_sha256` and fingerprint, so a chunk is re-checked only after its audio changes or with `--recheck`.
- Finding types: `omission`, `addition`, `substitution`, `repetition`, `uncertain_number`, `missing_audio`, `transcription_failed`, plus audio flags for internal silence over 2 s, leading or trailing silence over 1.5 s, speaking rate outside 8 to 25 characters per second, and clipping.
- Comparison ignores case, punctuation, stress marks and the ё/е difference, and treats spelled-out and digit numbers as equivalent. ASR spelling noise on long words is reported as `minor`.
- Limits: ASR checks what words were heard, not how they were said. It cannot judge lexical stress, е/ё pronunciation, prosody or listening comfort; every chunk file and the report say so, and a chunk without findings is labelled `no_automatic_findings`, not verified. Listen to the chapter before accepting it.

## Review queue

`rutts book review` connects flagged passages to their source text, narration, audio and generation history. Items are derived from disk state on every call; only decisions (`review/decisions.json`, append-only log in `review/decisions-log.json`) and manual findings (`review/manual.json`) are stored.

```sh
.venv/bin/rutts book review list books/novel --chapter c001
.venv/bin/rutts book review show books/novel <item_id>        # prints the exact override/regenerate/recheck commands
.venv/bin/rutts book review play books/novel <item_id> --join  # writes review/clips/<item_id>[-join].wav and plays it with afplay
.venv/bin/rutts book review flag books/novel --chunk c001-k0003 --kind stress --word открыла --note "stress on verb"
.venv/bin/rutts book review resolve books/novel <item_id> --reason "checked by ear"
.venv/bin/rutts book review recheck books/novel --chapter c001 --target short
.venv/bin/rutts book review status books/novel --target short
```

- Failed and missing chunks, and open content failures (omission, addition, substitution, repetition, missing audio, transcription failure), are blocking for `reviewed` export regardless of decisions. Other automatic flags are listed under `draft`.
- A resolution or dismissal is valid only while the chunk's audio hash, fingerprint and verifier configuration are unchanged; otherwise the item is open again as `stale`, with the old decision shown.
- Manual findings (stress, pronunciation, prosody, join, voice, other) stay open until resolved or dismissed. If the chunk audio changes they are marked `needs_relisten`.
- `status` prints two states per chapter: `reviewed` (no blocking reasons) and `draft` (audio exists; unresolved items are listed). Corrections and regeneration stay with `rutts book dict|override|regenerate`.

## Export

`rutts book export <book>` builds `exports/audiobook/audiobook.m4b`: an AAC (96 kb/s mono by default, `--bitrate`) M4B with title, author and one chapter marker per included chapter. `--format mp3` adds one tagged MP3 per chapter plus an `.m3u8` playlist. A reviewed export is refused, listing every reason, unless all included chapters have complete, current masters, every chunk is present and verified, and no content finding is open. `--draft` exports anyway as `audiobook.draft.m4b` with " (ЧЕРНОВИК)" in the title and every gap recorded.

Loudness uses one book-wide gain (gated RMS, not LUFS; `--no-loudness` disables it). Chunk and chapter master files are never modified. Each export records its settings, validation results and an input signature in `exports/<name>/export.json`; `rutts book export-status`, `export-check` and `export-open` inspect it. An export is reported stale when audio, chapter order or titles, book metadata or processing settings change.

## Blind listening evaluation

Run the preview (or `book render` for chapter trials) for each candidate first, then:

```sh
rutts eval prepare --candidates runs/qwen runs/fish --out evals/june --seed 20261013 --attempt first
rutts eval score evals/june                      # interactive; resumable; afplay plays each sample
rutts eval import-scores evals/june --file scores.csv   # scores written down elsewhere
rutts eval chapter-trial book/ --chapter c001 --target short --label qwen --out evals/june
rutts eval chapter-notes evals/june --label qwen --stress-errors 0 --omissions 0 --distracting-joins 1 --notes "..."
rutts eval report evals/june                     # un-blinds; writes report.json and report.md
rutts eval decide evals/june [--approve-target-revision "reason"]
```

- `prepare` writes anonymous samples under `samples/<excerpt>/<label>.wav` and `blind.json`. The identity key is in `key.json`; keep it out of the listening session. `--attempt final` compares the last successful attempt within the retry budget.
- Loudness is matched with one constant gain per file. Speed, length and sample rate are unchanged. If raising the level would clip, the common target is lowered for the whole set and `key.json` records it.
- Scores are only what the listener enters. Nothing is defaulted or estimated; `report` marks the report `incomplete` and lists every unscored sample or preference.
- `decide` applies the PRD gates and exits with code 1, writing `decision.json` with `status: "no_decision"` and the gaps, when scores are missing, fewer than two candidates are feasible, the top two lack a completed chapter trial, or none passes. A target revision needs `--approve-target-revision` and its reason is recorded verbatim.
