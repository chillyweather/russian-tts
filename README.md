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
