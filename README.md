# rutts

Local Russian narration previews on Apple Silicon. Engines: `qwen-mlx` (Qwen3-TTS, Apache-2.0), `fish-mlx` (Fish Audio S2 Pro 8-bit community conversion, non-commercial research license) and `voxcpm2` (VoxCPM2 MLX port, Apache-2.0; the official PyTorch MPS path is not used).

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
.venv/bin/rutts book assemble books/novel --chapter c001 --target short
.venv/bin/rutts book compare-targets books/novel --chapter c001
```

- Chapter markers are lines starting with `Глава` or `Часть` (also `ГЛАВА` and `ЧАСТЬ`) followed by a number or title, markdown `# Title` lines, and lines matching `--chapter-regex`. Without markers the whole file is one chapter.
- `plan` durations are estimates at 14 characters per second. `compare-targets` reports durations measured from the rendered audio.
- `render` refuses a chapter already rendered with a different engine configuration or narrator unless `--force-new-run` is given, which moves the old `chunks/<chapter>/` to `chunks/<chapter>.prev-<n>/`.
- `render` exits with code 1 when any chunk fails. `assemble` writes `<chapter>.incomplete.wav` and exits with code 1 when chunks are missing or failed, printing each one.

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
