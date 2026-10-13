# Russian narration evaluation and local audiobook pipeline

Research date: 13 October 2026. Research only; no models installed, generated, or listened to. The recommended experiment below is a proposal, not a completed evaluation.

## Which comparative evidence actually supports a Russian narration decision?

### Takeaway
There is credible Russian intelligibility evidence for Fish S2 and Qwen3-TTS, but no retrieved controlled Russian audiobook listening study establishes an overall winner between them and current Chatterbox. An excellent short-utterance WER score is insufficient to claim the best literary narration.

### Cited Findings
- Qwen's own multilingual benchmark reports Russian WER 3.212% and speaker cosine similarity 0.792 for Qwen3-TTS-12Hz-1.7B-Base; 0.6B Base gets 4.458% / 0.781. ElevenLabs Multilingual v2 gets 3.878% / 0.676; MiniMax gets 4.281% / 0.761. The table evaluates content consistency and speaker similarity, not a Russian long-form listening preference. Qwen's advertised long-form experiment uses 100 English/Chinese texts, not Russian. — [Qwen3-TTS technical report, Tables 6 and 10](https://arxiv.org/html/2601.15621v1)
- Fish S2's own report gives Russian WER 2.400% / SIM 0.790 on the MiniMax multilingual set, versus Fish S1 5.250% / 0.764. The repeated ElevenLabs/MiniMax figures match Qwen's report, making the direction informative but still cross-paper rather than a fresh head-to-head run. Fish also reports Russian CV3-Eval WER 2.78%, against CosyVoice3-0.5B+DiffRO 3.79% and 1.5B+DiffRO 4.11%. Its long-audio evaluation is English/Chinese, trimmed at sentence boundaries to fit approximately 185 seconds; it supplies no Russian audiobook listening winner. — [Fish S2 technical report, Tables 2–4](https://arxiv.org/html/2603.08823v1)
- Chatterbox Multilingual v3 launched 10 June 2026; its vendor's evaluation generates 100 held-out prompts per language, transcribes with Whisper-large-v3, then measures CER. Russian falls in the stated 1–5% middle band. The vendor explicitly says this does not measure prosody, speaker similarity, or expression and announces future subjective evaluation. It is CER, on different prompts, so cannot be ranked numerically against the above WERs. — [Resemble evaluation and limitations](https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages)
- Balalaika-Longform evaluates continuous Russian synthesis: 50 fixed text–voice pairs across 22 literary works and 21 held-out voices. Qwen3 Base WER is 3.9% at ~75 words, 5.3% at ~300, and 66.9% at ~1,200. Crucially, external splitting, resets, retries and best-of-N are disabled. A separate earlier CosyVoice sentence-split control completes all examples at 3.84% WER; an earlier F5 comparison finds 3.58% chunked versus 73.35% continuous, on different texts. No human listening study was run. These are evidence against blindly feeding very long passages, not a ranking of practical chunked audiobook systems. — [Balalaika-Longform, sections 3–5](https://arxiv.org/html/2610.00658v1)
- A useful historical Russian study uses roughly 1,000 audiobook utterances / 100 speakers and includes Silero, Vosk, Piper, XTTS2 and commercial engines. It reports low CER for Silero and explicitly notes weak intonation, with a need for better liveliness metrics. Dated 12 July 2024; it cannot settle a 2026 choice between current models. — [Vosk author evaluation and reproducible code](https://alphacephei.com/nsh/2024/07/12/russian-tts.html)

### Inferences
- Fish S2 deserves an audition as a quality candidate; Qwen3 1.7B Base deserves an equally serious audition. The available Russian rows favor Fish for textual fidelity, but Qwen slightly for that particular speaker similarity metric; neither establishes more convincing storytelling. These are author-reported results, with distinct implementations and sampling.
- Long uninterrupted generation is a different problem from a complete novel assembled from checked paragraphs. A quality-first implementation should optimize chunk lengths experimentally and preserve narrator consistency.
- Report each benchmark with model/checkpoint, language, unit length, metric and evaluation owner. Avoid translating “low WER” into “best sound” or comparing CER and WER as interchangeable percentages.

### Gaps
- No retrieved fair, native-Russian, blinded human comparison of current Fish S2, Qwen3 1.7B, Chatterbox v3 and other emerging models on chapters.
- No proof that a community quantized Mac conversion has the same Russian pronunciation or stability as the original precision implementation.
- No direct test on the user's books or preferred voice; we cannot honestly claim we heard these models ourselves.

## What preprocessing and structure matter for quality?

### Takeaway
Use a separate, reviewable narration-text layer with pronunciation overrides, while retaining EPUB structure and original text. Stress tools should assist correction; blindly inserting their markup into every model is not justified.

### Cited Findings
- Russian stress disambiguation includes morphological homographs, contextual homographs such as за́мок/замо́к, and е/ё ambiguity such as все/всё. RUAccent combines a homograph resolver, stress predictor and ё-restoration model. Its COLING evaluation reports 97.2% non-homograph accuracy and approximately 96.37% homograph accuracy for turbo3 on its own test regime; these figures are not an error guarantee on novels. — [RUAccent paper](https://aclanthology.org/2025.coling-main.444.pdf)
- Current RUAccent README demonstrates local CPU/CUDA execution, turbo3.1, a dictionary option and custom dictionary entries using a plus sign before the stressed vowel. It documents a 512 MB minimum for the tiny model. — [RUAccent repository](https://github.com/Den4ikAI/ruaccent)
- An independent implementation's stress benchmark (22 July 2026) reports RUAccent accuracy 0.938 overall / 0.820 homographs, versus Silero 0.914 / 0.381. It also warns that its crowdsourced gold data contain considerable annotation disagreement. Treat it as warning against trusting near-perfect headline accuracy, rather than a clean universal measure. — [stressonnx benchmark protocol and results](https://github.com/TigreGotico/stressonnx/blob/dev/benchmarks/RESULTS.md)
- EPUB's spine determines default reading order; linear=no marks auxiliary content such as notes. The standard allows reading systems to omit such content from aural rendering. Thus retaining EPUB semantics provides useful information beyond raw flattened text. — [W3C EPUB 3.3, spine and itemref](https://www.w3.org/TR/epub-33/#sec-spine-elem)

### Inferences
- Proposed preparation: extract reading order, chapter/scene/paragraph boundaries, dialogue and footnotes; maintain a stable link to original text. Avoid duplicating contents pages, navigation labels or footnotes. Give the user a deliberate policy for notes and foreign quotations.
- Normalize dates, numbers, abbreviations, initials and Roman numerals into intended spoken Russian where synthesis fails. Preserve case, punctuation and intentional ё. Maintain a per-book pronunciation dictionary for surnames, places, archaic words and recurring invented names. Review ambiguous accent/ё changes with surrounding context.
- Do not globally replace letters or paraphrase prose to improve delivery. Keep pronunciation representations separate from source content, so corrections remain traceable.
- RUAccent's `+` convention is an output format, not proof Qwen/Fish accept it. Compare raw text, ё-corrected text and selective model-compatible stress hints experimentally. Reject hints that are ignored, spoken literally or damage cadence.
- Start synthesis with coherent sentence groups / paragraphs, not arbitrary fixed character cuts. A practical initial experiment could compare roughly 20–40-second and 45–90-second chunks; these are proposed test values, not universal optimal limits. Split longer sentences only at meaningful clause boundaries. Use one fixed narrator reference and consistent settings across a chapter; avoid conditioning every new chunk on the preceding generated chunk without testing drift.
- Preserve natural silences and breaths; avoid hard cuts at phonemes or indiscriminate silence trimming. Normalize levels consistently after joining and keep lossless masters before M4B/MP3 encoding. Regenerate faulty chunks individually with a saved manifest of text/model/voice/settings.

### Gaps
- No verified universal stress-marking protocol across Fish, Qwen and Chatterbox; each implementation needs a focused pronunciation test.
- We have not inspected the user's EPUB conventions, fiction types, or whether one narrator versus character voices is preferred.

## How should we choose and run the final system locally?

### Takeaway
Choose the engine using a small paired listening test, then verify an actual chapter. Current local tooling exists, including Apple Silicon implementations, but runtime convenience and model quality are separate decisions.

### Cited Findings
- MLX-Audio documents local Apple Silicon TTS, including Qwen3 Base/CustomVoice/VoiceDesign and current multilingual Chatterbox; its CLI supports joining generated segments. The current model table does not list Fish. — [MLX-Audio](https://github.com/Blaizzy/mlx-audio)
- A separate MLX-Speech implementation explicitly provides Fish S2 Pro with int8 weights, reference voice workflow and Apple Silicon inference. This is community implementation evidence, not a native-Russian quality equivalence benchmark. — [MLX-Speech](https://github.com/appautomaton/mlx-speech)
- ebook2audiobook supports EPUB/TXT and chaptered audiobook outputs, including M4B. Its current supported engines are XTTSv2, Bark, Fairseq, VITS, Tacotron2, Tortoise, GlowTTS and YourTTS. Fish, Qwen3 and several others appear as unchecked planned items near the bottom; do not misread them as implemented integrations. Its broad minimum memory figures do not describe memory needs of all future engines. — [ebook2audiobook README, Features versus future checklist](https://github.com/DrewThomasson/ebook2audiobook)
- Current MLX Qwen 1.7B Base cards display 4.54 GB for BF16 and 3.1 GB for 8-bit. These are published model sizes, not measured peak process memory. — [BF16 card](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16); [8-bit card](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit)
- A community Fish S2 8-bit MLX card displays 6.72 GB. Its card's example accidentally still points to BF16, so implementation instructions require validation. The accompanying API README supports M1/M2/M3, but its 4–16 worker examples and lower file-size estimates must not be copied blindly onto a 16 GB machine. — [Fish conversion card](https://huggingface.co/cs2764/fish-audio-s2-pro-8bit-mlx); [community API implementation](https://github.com/cs2764/mlx-audio-api/blob/main/README.md)

### Inferences
Proposed decision experiment (not yet run):
1. Freeze two or three shortlisted checkpoints/runtimes. Use one clean native-Russian narrator reference with an accurate transcript where each model permits it; use the same voice source and comparable excerpt duration. If a model cannot clone the same voice, report that confound and evaluate its best suitable stock voice separately.
2. Select 12–20 matched excerpts from the user's actual book types: calm exposition, long syntax, emotional dialogue, questions, quotations, names, numerals, archaic prose, stress homographs and е/ё cases. Include unexceptional prose too, so the test does not only reward adversarial edge cases.
3. Generate identical narration text with each model; retain all first attempts and measure first-pass failures. Permit an equal, predefined retry budget in a second production-quality comparison. Keep raw and normalized text cases separate.
4. Blind labels, randomize order, loudness-match without time stretching or denoising, and have the user (ideally additional native Russian listeners) give paired preference and 1–5 ratings for naturalness, lexical stress, phrasing/emotion, narrator stability and comfort over time. Ask which voice they would want to hear for an hour, not merely which clip impresses first.
5. Run independent Russian ASR/alignment to flag omissions, additions, repetitions and mismatches. Manually listen to stress and ё cases: ordinary orthographic transcription may be identical despite the wrong lexical stress. Count audible pronunciation errors and catastrophic failures separately from ASR error rate.
6. Generate one 20–30-minute chapter with the preferred two setups. Check transitions, voice/prosody drift, reading fatigue, retries per audio hour, peak memory and total render time. Select based primarily on preference and correctness; use speed as feasibility, not the ranking objective.

Local deployment suggestion for the now-confirmed M1 Pro with 16 GB unified memory:
- Qwen3 1.7B Base through MLX is the best-supported practical starting point from this research. Try the BF16 conversion with short chunks and one worker for the quality baseline; move to 8-bit if actual peak memory/runtime makes BF16 impractical. A 4.54 GB published model size makes a controlled 16 GB trial plausible, but is not a verified peak-memory guarantee.
- Audition Fish S2 only via a validated quantized community MLX implementation, one model/process at a time. Its 6.72 GB published model size leaves less room for macOS, activations and cache. Treat feasibility as conditional on a short memory test before committing to a chapter, not as an already proven fit. Do not recommend the original full-precision GPU deployment as if it were the 16 GB Mac path.
- Run TTS, ASR QA and accent preparation sequentially where possible; avoid a concurrently resident local LLM or multiple TTS workers. Keep measured peak memory and swap use alongside audio quality results. Offline batch rendering makes slower throughput acceptable, but does not solve memory exhaustion.
- Include current Chatterbox or Higgs v3 as a third contestant only after their actual MLX memory fit is verified by the model-specific researcher. Do not downgrade to a weaker small engine solely to achieve realtime speed: this user prioritizes narration quality and accepts offline batch generation.

### Gaps
- The user confirms M1 Pro / 16 GB unified memory. Exact current free memory, macOS version and actual inference performance remain unmeasured. No hardware purchase recommendation is required.
- No measured local throughput, memory, full-book failure rate or native-Russian listening results exist for this project yet.
- ebook2audiobook may be useful for format/packaging ideas, but needs integration work or a different frontend for the leading engines; don't let its default engine determine the quality ceiling.
