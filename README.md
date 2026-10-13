# rutts

Local Russian narration previews with Qwen3-TTS (MLX) on Apple Silicon.

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
