# Russian audiobook TTS — Product requirements

**Date:** 13 October 2026  
**Status:** Draft for implementation; model selection awaits local listening tests  
**Target hardware:** Apple M1 Pro, 16 GB unified memory  
**Primary user:** A Russian-speaking reader converting books into audiobooks for personal listening

## 1. Product objective

Build a local system that turns Russian EPUB or plain-text books into natural, accurate, enjoyable audiobooks. Speech and narration quality are the highest priority. Generation speed is secondary, provided the system can reliably finish work on the target Mac.

The product must preserve the author's text, pronounce Russian correctly, maintain a stable narrator, and produce comfortable long-form listening. A good short demonstration is insufficient: the selected configuration must also pass a complete chapter evaluation.

Priority order:

1. Natural, expressive narration and sustained listening comfort.
2. Correct pronunciation, lexical stress, and faithful coverage of the book.
3. Stable voice, pacing, and transitions across passages and chapters.
4. Reliable local operation within the available memory.
5. Simple corrections, resumable generation, and usable audiobook output.
6. Throughput and convenience.

## 2. Confirmed constraints and working assumptions

| Item | Requirement or assumption |
|---|---|
| Language | Russian is the primary narration language. Foreign names and quotations need explicit treatment. |
| Inputs | EPUB is the primary book format; UTF-8 plain text must also work. |
| Hardware | M1 Pro with 16 GB of shared system/GPU memory. |
| Execution | Book preparation, synthesis, and audio verification run locally after dependencies and model files are downloaded. |
| Quality | Select the model and voice by listening tests; published recognition scores alone cannot decide. |
| Speed | Offline batch rendering is acceptable as a working assumption. No real-time requirement. |
| First interface | A command-line workflow with saved configuration and reviewable files is the proposed MVP. A graphical interface is optional later. |
| Narration | One consistent narrator per book is the proposed MVP. Character-specific voices are deferred. |
| Distribution | Initial use is personal. Model and voice-source licenses must be recorded before any later publishing workflow. |

No model has yet been installed or measured in this project. Local memory fit, generation speed, and Russian listening quality remain unverified. All acceptance thresholds below are proposed product targets, not research results.

## 3. User workflow

1. Import an EPUB or text file and create a book project.
2. Review the extracted reading order, chapter boundaries, excluded material, and narration text.
3. Choose a narrator profile and generate several preview passages.
4. Correct recurring names, stress, numbers, and other pronunciation problems in a book dictionary.
5. Generate the selected chapters with visible progress and saved checkpoints.
6. Review flagged passages alongside their text and audio; regenerate individual passages as needed.
7. Export a chaptered audiobook while retaining lossless audio and the project for future corrections.

The user must be able to stop after previews, compare voices, and change the selected configuration before paying the time cost of a full book.

## 4. Model strategy and selection gate

### Initial candidates

| Candidate | Role | Local execution plan | Selection caveat |
|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-Base** | First implementation and quality baseline | MLX; test BF16 with one worker, then 8-bit if memory requires it | Most defensible starting point for this Mac; not an established Russian audiobook quality winner. |
| **Fish Audio S2 Pro** | Required quality challenger, subject to feasibility | Validated community 8-bit MLX implementation; memory smoke test before chapter generation | Official deployment guidance targets larger GPU memory. Mac runtime parity and fit need measurement. |
| **VoxCPM2** | Required alternative audition | Official Apple Silicon MPS path | Reported GPU memory figures do not establish actual total memory on this Mac. |
| **OmniVoice / F5-TTS Russian Base v2** | Second-round alternatives | Validate the exact checkpoint and Apple Silicon runtime | Add if initial models disappoint on expression, pronunciation, or reliability. |

Qwen's Base model should use a clean native-Russian reference recording and accurate transcript. Do not assume that a preset voice or an embedding-only cloning path produces equivalent quality. Reuse the same approved narrator source across a book.

Keep the synthesis engine replaceable. Freeze the exact checkpoint, model revision, runtime version, precision, and synthesis settings in each evaluation and book manifest. A change of model, precision, or voice creates a new configuration that must be evaluated before replacing the selected one.

### License record

The initial research identifies Apache-2.0 weights for Qwen and VoxCPM2, restricted noncommercial/research terms for Fish S2 Pro, and noncommercial pretrained weights for OmniVoice and the Russian F5 candidate. OmniVoice's code license and pretrained-model license differ. Record the license of the exact downloaded artifact and any required attribution; retain provenance for the narrator reference.

### Evaluation gate

Evaluate 12–20 matched Russian excerpts representing ordinary narrative, dialogue, emotional passages, long sentences, questions, names, dates, abbreviations, stress homographs, and е/ё ambiguities. Include representative easy prose as well as difficult examples.

- Use the same text and comparable reference recording wherever supported; document any unavoidable voice-source differences.
- Randomize anonymous sample labels and match playback loudness without changing speaking speed.
- Have the user score naturalness, phrasing, pronunciation, voice stability, and listening comfort from 1 to 5, and state a pairwise preference.
- Preserve first attempts. Separately compare corrected output under an equal, predefined retry budget.
- Compare the preferred two feasible configurations on the same 20–30-minute chapter.
- Measure peak memory, memory pressure/swap, render time, failed chunks, and retries per finished audio hour.

Select the most preferred configuration that passes content-fidelity and hardware gates. Resolve close preferences through longer listening. Speed is a secondary tie-breaker. A candidate that cannot finish a chapter reliably on the target Mac is not eligible as the default.

## 5. Functional requirements

### FR-1: Book ingestion and source fidelity

- Accept EPUB and UTF-8 text. Detect invalid or unsupported input with a clear error.
- For EPUB, retain spine reading order, chapter titles, paragraphs, scene breaks, and available author/title metadata.
- Preview how contents pages, navigation, illustrations, footnotes, and other auxiliary material will be handled. Footnotes must have a selectable policy, such as omit or read at chapter end.
- For plain text, support explicit chapter markers and manual boundary correction.
- Retain the original file and extracted source text unchanged. Store narration text as a separate representation with links back to source locations.
- Record deliberate exclusions. Do not silently summarize, paraphrase, duplicate, or omit prose.

### FR-2: Russian narration preparation

- Preserve punctuation, paragraph structure, and meaningful е/ё distinctions.
- Support reviewable spoken-form overrides for numbers, dates, initials, abbreviations, Roman numerals, names, and foreign phrases.
- Provide a per-book pronunciation dictionary with contextual overrides for ambiguous words. Global replacement alone is insufficient for homographs.
- Allow optional local stress/ё assistance, with changes visible before generation.
- Translate pronunciation hints through each engine's supported input conventions. Test that markup is neither spoken aloud nor harmful to delivery before enabling it.
- Maintain an edit history so a correction can be traced and reverted without altering the source book.

### FR-3: Narrator profiles and previews

- Store the voice reference, its transcript and provenance, language, model configuration, and adjustable settings supported by that engine.
- Generate representative previews before full-book synthesis.
- Keep the narrator reference and selected configuration fixed throughout a generation run.
- Allow chapter- or passage-specific delivery overrides only when supported and explicitly configured.
- Do not chain generated audio as the next reference by default; this requires separate evidence that voice drift remains acceptable.

### FR-4: Segmentation and synthesis

- Split into coherent sentence groups or paragraphs, respecting engine limits and dialogue boundaries.
- Compare initial chunk targets around 20–40 seconds and 45–90 seconds during evaluation. These are experimental ranges, not fixed optimal limits; actual duration is known after synthesis.
- Split exceptionally long sentences at meaningful clause boundaries when necessary.
- Generate one chunk at a time by default on the 16 GB Mac.
- Persist a lossless result and metadata after every successful chunk.
- Support pause, interruption recovery, and resume without regenerating unchanged completed chunks.
- Use content and configuration fingerprints to invalidate affected audio when text, voice, model, or relevant settings change.
- Retry failures under a bounded policy. Preserve failed-attempt diagnostics and require review after the retry limit; never silently skip a failed passage.

### FR-5: Quality verification and correction

- Use local ASR/alignment to flag possible missing, repeated, substituted, or added words and suspicious duration/silence.
- Treat automated detection as a review aid. ASR cannot reliably establish correct lexical stress or pleasant narration.
- Provide a review queue containing the source text, narration text, audio location, reason for the flag, and generation history.
- Let the user correct text or pronunciation and regenerate only affected chunks, followed by rechecking their joins.
- Track unresolved issues. A reviewed export must not silently include missing chunks or unresolved content failures.
- Allow an explicitly labeled draft export for listening before review is complete.

### FR-6: Audio assembly and export

- Assemble chunks in source order, with chapter and scene pauses and natural sentence endings.
- Preserve speech, breaths, and useful silence; avoid automatic trimming that cuts phonemes or changes delivery.
- Apply consistent loudness treatment without clipping. Save the chosen processing settings.
- Retain lossless chunk and chapter masters. Resample only when required and record it.
- Export a chaptered M4B with title, author, and chapter metadata. Provide MP3 as an optional compatibility export.
- Verify chapter order, chapter markers, decodability, and complete chunk coverage before marking an export ready.

### FR-7: Local project and resource management

- Store source files, narration text, dictionaries, voice profiles, model references, chunk manifests, audio, reviews, and exports in a self-contained book project, excluding shared model weights where appropriate.
- Show completed/remaining chunks, current chapter, elapsed time, and recent failures. Estimate completion time only after observing local throughput.
- Run preparation models, synthesis, and ASR sequentially where practical to preserve memory headroom.
- Record resource measurements per configuration. Warn or stop safely when the tested resource envelope is exceeded; preserve resumable work.
- After setup, processing must work with networking disabled. Never require uploading book text, reference audio, or generated audio.

## 6. Acceptance criteria

These initial targets should be calibrated after the first evaluation, with any changes documented.

| Area | Initial acceptance target | Evidence |
|---|---|---|
| Narration | Mean user ratings of at least 4/5 for naturalness and sustained comfort on the selected chapter, with no individual category below 3/5 | Blind sample scores plus uninterrupted chapter listening |
| Pronunciation | No unresolved errors in the reviewed chapter's designated names, homographs, number readings, or stress test cases | Manual listening against the narration script |
| Completeness | Every included source segment accounted for; no unresolved omissions, additions, or repetitions in the reviewed acceptance chapter | Manifest coverage, automated flags, and manual full-chapter review |
| Voice continuity | No unresolved abrupt narrator changes, clipped speech, or distracting joins in the acceptance chapter | Continuous listening and recorded corrections |
| Memory feasibility | Complete the chapter on M1 Pro / 16 GB without an out-of-memory failure or sustained resource pressure that makes the machine impractical to use | Measured peak memory, memory-pressure/swap observations, and render completion |
| Recovery | Interruption and restart preserve completed valid chunks; a text correction regenerates only affected output | Recovery and invalidation checks |
| Local operation | Import, synthesis, verification, and export succeed with networking disabled after setup | Offline end-to-end check |
| Output | M4B opens in the chosen player with correct metadata, chapter navigation, and no missing audio | Export inspection and playback |

Published WER figures are candidate-selection evidence, not acceptance thresholds. Do not optimize an ASR score at the expense of natural delivery or faithful literary text. Full-book correctness requires review; passing a sampled chapter does not prove every later chapter is error-free.

## 7. Delivery stages

### Stage 1: Model and voice evaluation

Create the representative text suite, obtain an appropriate Russian narrator reference, run memory smoke tests, and compare feasible candidates. Deliver anonymous samples, resource measurements, chapter samples, and a documented model/voice decision.

**Exit:** A preferred configuration meets the chapter quality and memory gates, or the limitations are documented and additional candidates are evaluated.

### Stage 2: Plain-text vertical slice

Implement import, narration overrides, segmentation, local synthesis, saved manifests, resume, targeted regeneration, and lossless chapter output. Add local content verification and the review queue.

**Exit:** One text chapter can be generated, corrected, interrupted/resumed, and reviewed end to end.

### Stage 3: EPUB and audiobook release

Add EPUB structure and auxiliary-content policies, full-book jobs, metadata, audio assembly, and chaptered M4B export. Exercise the workflow on a complete representative book.

**Exit:** A complete book can be processed locally with all source segments accounted for, review state preserved, and a playable chaptered export.

### Stage 4: Quality refinement

Use observed failures to improve dictionaries, normalization, chunk boundaries, pauses, and runtime settings. Re-evaluate any model or quantization change. Consider an interface for easier review only after the narration workflow is dependable.

## 8. Deferred scope

- Multi-character casting and automatic emotion assignment.
- Training or fine-tuning a new model.
- Cloud synthesis, hosted APIs, and multi-user services.
- Real-time speech generation or playback while synthesis is in progress.
- Mobile applications and a polished graphical interface.
- OCR for scanned books and broad multilingual narration.

## 9. Risks and open decisions

| Risk or decision | Planned response |
|---|---|
| Best-sounding model may exceed available memory | Measure actual runtime first; evaluate quantization audibly and retain feasible alternatives. |
| Russian stress tools can introduce errors | Keep overrides reviewable and test markup per engine. |
| Short samples can hide drift and fatigue | Require extended chapter listening before model selection. |
| Automatic transcription misses stress/prosody defects | Use manual Russian listening as a release gate. |
| Reference voice dominates results | Choose a clean native-Russian narration source and compare suitable references if needed. |
| Exact narrator preference is unspecified | Decide voice, style, and reference source during Stage 1 previews. |
| Actual books and footnote preferences are unspecified | Choose representative fiction/nonfiction inputs and confirm extraction policies during the first import. |
| Acceptable overnight render duration is unspecified | Report measured throughput before setting a full-book completion target. |
| Model terms may limit later distribution | Keep a versioned license record and revisit requirements if publishing becomes a goal. |

## 10. Research basis

See [Local Russian audiobook narration](reports/Local%20Russian%20audiobook%20narration.md) for the complete research findings, supporting sources, and evidence limitations.

Primary implementation references:

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) and [MLX-Audio](https://github.com/Blaizzy/mlx-audio).
- [Fish Speech](https://github.com/fishaudio/fish-speech) and [community MLX-Speech](https://github.com/appautomaton/mlx-speech).
- [VoxCPM](https://github.com/OpenBMB/VoxCPM).
- [RUAccent](https://github.com/Den4ikAI/ruaccent).
- [Balalaika-Longform evaluation](https://arxiv.org/html/2610.00658v1), whose continuous-generation tests must not be confused with a chunked production pipeline.

The model recommendation is provisional. This PRD defines the experiments and product behavior needed to turn that recommendation into a verified local audiobook system.
