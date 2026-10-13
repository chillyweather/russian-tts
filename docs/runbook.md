# Runbook: one book, offline, from a clean checkout

This runbook takes one book from text or EPUB to a chaptered M4B, then records the acceptance evidence for issue #14. Every command below exists in `.venv/bin/rutts --help`. Replace `books/novel`, `profiles/narrator` and the other paths with your own.

Measured throughput is not yet measured on this machine. Do not treat any number in this repository as a measurement.

## 1. Setup (networked, once)

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[mlx,dev]"
.venv/bin/rutts --help
```

Model weights are not shipped. They are fetched on first load. There is no separate download command:

- Run one smoke test with the network on, for the engine you will use. It loads the synthesis model and caches it: `rutts smoke --engine qwen-mlx --precision bf16 --narrator profiles/narrator --out smokes/qwen` (without `--offline`).
- Run one `rutts book verify` without `--offline` to fetch the Whisper transcriber (`mlx-community/whisper-large-v3-turbo`, pinned revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`). If the cached revision differs from the pin, verification refuses to run.

Once cached, every later command can run with `--offline`, which sets `HF_HUB_OFFLINE=1`.

## 2. Narrator profile

Use a clean Russian reference recording (3 to 60 s) and its exact transcript:

```sh
.venv/bin/rutts narrator create --name "Narrator" --audio ref.wav \
  --transcript-file ref.txt --provenance "Recorded by me, 2026" --out profiles/narrator
```

The provenance is required and is frozen into the job configuration. Acceptance fails if it is empty.

## 3. Previews and smoke test

```sh
.venv/bin/rutts preview --engine qwen-mlx --precision bf16 --narrator profiles/narrator \
  --out previews/run1 --suite --retries 1 --offline
.venv/bin/rutts smoke --engine qwen-mlx --precision bf16 --narrator profiles/narrator \
  --out smokes/qwen --offline
```

Listen to the previews. Do not start a full book with an engine you have not heard. A failed smoke test writes `smoke.json` and exits 1.

## 4. Import and boundary review

```sh
.venv/bin/rutts book import novel.txt --out books/novel --title "Novel" --author "Author"
# EPUB instead:
.venv/bin/rutts book preview-import novel.epub --footnotes omit
.venv/bin/rutts book import novel.epub --out books/novel --footnotes omit
```

Review the chapter boundaries and exclusions, and correct them before generating:

```sh
.venv/bin/rutts book show books/novel
.venv/bin/rutts book show books/novel --chapter c002
.venv/bin/rutts book split books/novel --at c001-s0004 --title "Evening"
.venv/bin/rutts book merge books/novel --chapter c003
.venv/bin/rutts book rename books/novel --chapter c002 --title "Дорога"
.venv/bin/rutts book exclude books/novel --segment c002-s0003 --reason "footnote"
.venv/bin/rutts book include books/novel --segment c002-s0003
.venv/bin/rutts book narrator books/novel --profile profiles/narrator
```

Every exclusion needs a reason. The acceptance coverage section lists them all.

## 5. Corrections and optional suggestions

Pronunciation and text corrections are applied to the narration, never to the source. Each command shows the diff and the chunks it would invalidate, and needs `--yes` to apply:

```sh
.venv/bin/rutts book dict add books/novel --pattern "замок" --replacement "з+амок" --kind stress --yes
.venv/bin/rutts book override add books/novel --segment c001-s0002 --find "замок" --occurrence 2 --replacement "зам+ок" --yes
.venv/bin/rutts book diff books/novel --chapter c001
.venv/bin/rutts book history books/novel
.venv/bin/rutts book revert books/novel 2
```

Stress hints are stripped unless an engine has a convention you have verified by ear. Test first, and enable only after listening:

```sh
.venv/bin/rutts book markup-test books/novel --engine qwen-mlx --narrator profiles/narrator --out review/markup --offline
.venv/bin/rutts book markup-set books/novel --engine qwen-mlx --convention plus --verified-by-listening
```

The optional assistant (RUAccent, not part of the default install and not verified against the real package) proposes stress and ё changes. Nothing is applied until you accept it:

```sh
.venv/bin/rutts book suggest books/novel --chapter c001
.venv/bin/rutts book suggestions books/novel --chapter c001
.venv/bin/rutts book suggestion accept books/novel c001-g001 --yes
.venv/bin/rutts book suggestion reject books/novel c001-g002
```

## 6. Full-book generation and resume

Run with networking off after step 1. The first run freezes the configuration in `job/job.json`:

```sh
.venv/bin/rutts book generate books/novel --engine qwen-mlx --precision bf16 --target short --offline --assemble
.venv/bin/rutts book job-status books/novel
```

- Ctrl-C stops the job, keeps every valid chunk, and exits 130. Run the same command again to resume.
- Exit 3 means a resource threshold stopped the job; diagnostics are in `job/diagnostics/`. Free memory and run the same command again.
- Exit 1 means some chunks failed. Retry them with `--retry-failed`.
- Chapter masters are in `chapters/`. Check progress with `.venv/bin/rutts book status books/novel`.

## 7. Verification and review

Verification uses the local Whisper transcriber. It flags possible errors; it does not judge stress, е/ё, prosody or comfort:

```sh
.venv/bin/rutts book verify books/novel --chapter c001 --transcriber whisper-mlx --offline
.venv/bin/rutts book verify books/novel --chapter c002 --transcriber whisper-mlx --offline
.venv/bin/rutts book verify books/novel --chapter c003 --transcriber whisper-mlx --offline
.venv/bin/rutts book findings books/novel --chapter c001
.venv/bin/rutts book review list books/novel --chapter c001
.venv/bin/rutts book review show books/novel <item_id>
.venv/bin/rutts book review play books/novel <item_id> --join
.venv/bin/rutts book review flag books/novel --chunk c001-k0003 --kind stress --word открыла --note "stress on verb"
.venv/bin/rutts book review resolve books/novel <item_id> --reason "checked by ear"
.venv/bin/rutts book review dismiss books/novel <item_id> --reason "not a problem"
.venv/bin/rutts book review status books/novel
```

After a correction, regenerate only the changed chunks, then re-verify and reassemble the affected chapter:

```sh
.venv/bin/rutts book regenerate books/novel --chapter c002 --engine qwen-mlx --precision bf16 --target short --offline
.venv/bin/rutts book verify books/novel --chapter c002 --transcriber whisper-mlx --offline
.venv/bin/rutts book assemble books/novel --chapter c002 --target short
```

## 8. Export

```sh
.venv/bin/rutts book export books/novel --target short
.venv/bin/rutts book export-status books/novel
.venv/bin/rutts book export-check books/novel
.venv/bin/rutts book export-open books/novel
```

A reviewed export is refused, and every reason is printed, unless every chunk is valid and verified, the masters are complete and current, and no content finding is open. `--draft` produces a clearly labelled draft instead; it is not acceptance evidence.

Check portability at any time:

```sh
.venv/bin/rutts book check-portable books/novel
```

## 9. Acceptance with networking disabled

The acceptance run needs a real model, a real player and your own ears. The record lives in `books/novel/acceptance/`.

1. Turn networking off: switch Wi-Fi off in the menu bar (or System Settings > Wi-Fi), and unplug any Ethernet.
2. Run the workflow stages in order, with `--offline` on the generate, verify and regenerate commands (`book export` has no `--offline` flag). Each `accept step` probes the network when it runs and records the result. Do not run a stage while online:
   ```sh
   .venv/bin/rutts book accept step books/novel import
   .venv/bin/rutts book accept step books/novel preparation
   .venv/bin/rutts book accept step books/novel generation
   .venv/bin/rutts book accept step books/novel verification
   .venv/bin/rutts book accept step books/novel correction
   .venv/bin/rutts book accept step books/novel export
   ```
   Each step records the network state at that moment. A step recorded while online does not count, and a stage whose artifacts are missing or stale is refused.
3. Exercises performed in the same run, so the record can show them. Each is evidenced only from what the commands actually did:
   - A pronunciation correction followed by targeted regeneration (section 5 and section 7 above).
   - An interruption and restart: press Ctrl-C during `book generate`, then run the same command again to completion.
   - An export rebuilt after a change: export, change a correction, regenerate, verify and assemble, then export again.
4. Listen to the acceptance chapter in one uninterrupted pass, through headphones, from the master file `books/novel/chapters/c001.wav` in a player of your choice. Then record your own scores:
   ```sh
   .venv/bin/rutts book accept listen books/novel --chapter c001
   ```
   You type the scores (1 to 5) and answer whether you listened to the whole chapter. Scores are never defaulted. `--scores-file` accepts a JSON file you wrote yourself; its path and hash are recorded.
5. Open the M4B in the player you chose (for example `.venv/bin/rutts book export-open books/novel`, or open `books/novel/exports/audiobook/audiobook.m4b` yourself). Check the title, author, chapter titles and navigation, and that the audio plays without gaps. Then record it:
   ```sh
   .venv/bin/rutts book accept playback books/novel --player "NAME"
   ```
   You type the player version and answer each check. The M4B hash is recorded, so rebuilding the export requires confirming again.
6. Review the record and write the report:
   ```sh
   .venv/bin/rutts book accept status books/novel
   .venv/bin/rutts book accept report books/novel
   ```
   `status` exits 0 only when the overall result is `accepted`. The report is `books/novel/acceptance/acceptance.md`.

Automated ASR results appear in the report, but they never replace the listening section.

## Known limitations

These have not been measured or verified in this repository yet:

- No real synthesis model has been run: memory fit, throughput and listening quality are unmeasured. Throughput: not yet measured.
- The Whisper tokenizer files are missing from the cached MLX repo, so real transcription has not run.
- The RUAccent adapter is unverified against the real package.
- Stress markup conventions are unverified for all engines. Stress hints are stripped unless a convention is verified by listening.
- Chapter headings are not narrated.
- VoxCPM2 uses the MLX port, not the official PyTorch MPS path.
- No real EPUB has been tested.
- Playback in a real player has not been verified.
- The engine revision is recorded as `unresolved` in `job/job.json` until the model has been loaded; acceptance reports configuration as failed until this is fixed.
