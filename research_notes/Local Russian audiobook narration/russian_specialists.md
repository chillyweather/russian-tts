# Russian specialist and established local TTS — research checked 2026-10-13

## Which specialist deserves an audition, and which is the dependable baseline?

### Takeaway
Audition **Misha24-10/F5-TTS_RUSSIAN, specifically F5TTS_v1_Base_v2**, as the strongest well-documented Russian-specialist candidate found here. Use **Silero v5_5_ru** as an efficient pronunciation/control baseline, without claiming it wins expressive audiobook narration. These are selection judgments, not an established head-to-head quality ranking.

### Cited Findings
- Misha's F5 Russian model card reports 5,000+ training hours spanning Russian and English, explicit `+` before stressed vowels, and recommends RUAccent for automatic stress. It lists original v1 Base, accent_tune (all training sentences accented), and Base_v2 (16 additional epochs, approximately 5% artifact filtering, fully accented text). License is **CC-BY-NC-4.0**. The card does not establish v2 release date or a quantitative naturalness win. — [Model card](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN)
- The developer's comparison page provides audible Russian samples from F5, X-TTS and FishSpeech across several voices and substantial passages. These are selected developer demonstrations; versions/settings are not sufficiently documented for a present-day ranking against Fish S2-Pro or Qwen3. No listening was performed during this research. — [Demo comparison](https://misha24-10.github.io/Misha24-10/)
- F5's upstream project documents installation for Apple Silicon, NVIDIA, AMD and Intel GPUs. The v1 base announcement dates to **2025-03-12**. Code is MIT, while pretrained weights are noncommercial; code licensing must not be substituted for weight licensing. — [Upstream F5](https://github.com/SWivid/F5-TTS)
- A community MLX Russian conversion exists, but its very sparse model card gives inadequate provenance/version validation. Treat as an engineering lead, not a verified drop-in v2 equivalent or benchmark. — [Russian MLX conversion](https://huggingface.co/ink-splatters/f5-tts-russian-mlx)
- Another Russian/English F5 fine-tune, hotstone228/F5-TTS-Russian, states 813k steps and a 100k-hour dataset; these are author claims without an independent quality comparison. Its license is CC-BY-NC-SA-4.0. — [Model card](https://huggingface.co/hotstone228/F5-TTS-Russian)
- Silero's current official model table lists **v5_5_ru** with five voices (`aidar`, `baya`, `kseniya`, `xenia`, `eugene`), automatic stress/homograph handling and question support, with 8/24/48kHz outputs. It advertises CPU and GPU operation. The Russian checkpoints use the main noncommercial repository license; the **CIS base** family is separately MIT and requires explicit stress for Russian. — [Official Silero README](https://github.com/snakers4/silero-models)
- The actual main license text is **CC-BY-NC-SA-4.0**, although the README abbreviates it imprecisely as CC-NC-BY. — [License](https://github.com/snakers4/silero-models/blob/master/LICENSE)
- Silero's official notebook exposes `put_accent`, `put_yo`, `put_stress_homo`, and `put_yo_homo`, so both stress and context-sensitive ё restoration have explicit switches. — [Notebook](https://raw.githubusercontent.com/snakers4/silero-models/master/examples_tts.ipynb)
- Silero SSML allows explicit breaks, paragraph/sentence pauses, rate and pitch, plus manual `+` stresses. This is useful for reproducible audiobook punctuation repair. — [SSML documentation](https://github.com/snakers4/silero-models/wiki/SSML)
- Repository release **v5.6**, displayed as June 4 in current release listing, is a **Windows SAPI5 packaging release**, not a new Russian v5_6_ru model. It packages v5_5_ru, and says question support is not yet present in that SAPI release despite the Python README capability table. Avoid conflating package release numbers, checkpoint IDs, and platform features. — [Releases](https://github.com/snakers4/silero-models/releases)

### Inferences
- A fixed-voice Silero baseline plus a stress-controlled Russian F5 challenger isolates whether richer voice cloning actually improves listening comfort for this user's books. The input and reference voice need matching across tests.
- Do not automatically insert `+` stress marks into every model: explicit support is documented for the Russian F5 and Silero paths, not universally for modern multilingual tokenizers.
- Use original, native Russian reference narration of the desired style for F5. The demo evidence shows why reference voice choice should be considered part of the audition, not held as an invisible variable.

### Gaps
- No independent modern Russian audiobook listening comparison among Silero v5_5, F5 Russian v2, Fish S2-Pro, Qwen3 and VoxCPM2 was found.
- Exact F5 Russian v2 minimum memory and Apple throughput were not established; Apple installation support is not a performance guarantee.
- Silero digit expansion was not verified in primary documentation here. Do not promise arbitrary book numerals, dates, Roman numerals, abbreviations and units will read correctly without preprocessing.

## Do VoxCPM2 and CosyVoice3 materially change the shortlist?

### Takeaway
**VoxCPM2 is a serious practical alternative**, especially for local Apple Silicon and permissive licensing: explicit upstream MPS support, approximately 8GB reported VRAM, and Russian benchmark evidence. CosyVoice3 is worth a secondary audition but its upstream Apple path is less mature.

### Cited Findings
- VoxCPM2 was announced **April 2026**. Its official repository documents a 2B backbone, 48kHz audio, 30 languages including Russian, voice design, controllable cloning, and Apache-2.0 for code and weights. Reported memory is approximately **8GB VRAM**, and RTX 4090 RTF approximately **0.30** standard or **0.13** with Nano-vLLM. These are vendor figures, not measurements on the user's machine. — [VoxCPM repository](https://github.com/OpenBMB/VoxCPM)
- The same repository explicitly supports `cpu`, `mps`, `cuda` and `cuda:N`; auto selects MPS on Apple Silicon. Its reported Minimax-MLS Russian WER is VoxCPM2 **3.634**, Qwen3-TTS **3.212**, FishAudio S2 **2.400**, with similarity **81.1**, **79.2**, **79.0** respectively. Its distinct internal ASR benchmark gives Russian WER **0.90% VoxCPM2 / 0.86% Fish S2-Pro**. These evaluate intelligibility/similarity, not audiobook listening quality; the datasets and Fish labels differ. — [VoxCPM repository](https://github.com/OpenBMB/VoxCPM)
- Fun-CosyVoice3's **0.5B-2512** checkpoint supports Russian among nine languages, multilingual cloning, emotion/speed instructions and text normalization for numerals/symbols. Documented pronunciation inpainting covers Chinese Pinyin and English CMU phonemes, not Russian stress control. — [Upstream CosyVoice](https://github.com/QwenAudio/CosyVoice)
- That released checkpoint is tagged **Apache-2.0** and is downloadable locally. — [Official model card](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- Native Apple MPS support is proposed by upstream PR **#1869**, opened April 5, 2026 and still shown open in the fetched source. Thus official CUDA support should not be represented as fully equivalent to Mac support. — [Upstream PR](https://github.com/QwenAudio/CosyVoice/pull/1869)
- A community CosyVoice3 implementation reports M4/16GB RTF approximately 0.76, approximately 8GB RAM for the worker, and uses a mixture of MLX/MPS/CPU components. This demonstrates a plausible local Mac path but the maintainer's performance claim is unverified here and depends on that specific implementation. — [Community Mac implementation](https://github.com/drmhse/tts-funaudio-cozyvoice3)

### Inferences
- VoxCPM2 belongs alongside Fish and Qwen3 in a quality-first audition, and could become the best operational compromise if its selected Russian voice wins the user's listening test. Strong similarity is not proof of natural expressive narration.
- **User hardware now confirmed: M1 Pro with 16GB unified memory.** VoxCPM2's official MPS support and approximately 8GB reported GPU-memory demand make it a credible first audition on this machine, but actual unified-memory usage must be measured; no M1 Pro performance guarantee is inferred. Offline/overnight generation is acceptable, so speed should not disqualify a model that demonstrably fits and wins on audio. The 2B VoxCPM2 is a more credible unquantized local candidate here than an 8B model whose weights alone can consume the memory budget.
- Approximately 8GB NVIDIA VRAM is not a direct guarantee of 8GB total Apple unified-memory operation. Additional runtime/OS headroom is required; measure on the actual Mac.
- For a first reliable Mac evaluation, prioritize official VoxCPM2 MPS and standard F5 Apple paths before adopting a bespoke CosyVoice fork.
- F5 Russian deserves the specialist comparison on this Mac through upstream PyTorch/MPS support. Its unofficial MLX conversion should be used only after confirming it corresponds to the desired Russian checkpoint; do not assume a sparse older MLX card equals Base_v2 or preserves its stress behavior. Quantization/MLX speed benefits do not constitute evidence of higher narration quality.

### Gaps
- No official VoxCPM2 MPS speed/memory numbers or native Russian stress-control syntax were established.
- No independent paragraph-level Russian listening evidence establishes VoxCPM2 over Fish or Qwen3.

## What do long-form evidence and older contenders actually establish?

### Takeaway
The fresh Russian long-form study strongly supports **chunking and text-fidelity validation**, but cannot rank audiobook naturalness or justify sending an entire chapter through one inference. XTTS-v2 remains a historical/local baseline, not the quality-first default based on this evidence.

### Cited Findings
- **Balalaika-Longform**, arXiv v1 dated **2026-09-30**, studies single-pass continuous generation rather than normal chunked audiobook pipelines. Its Russian corpus contains 189 hours; evaluation uses 50 text–voice pairs from 22 literary works. No listening study was run. In separate earlier recorded-text controls, sentence-split CosyVoice3 completed all items at **3.84% WER**, while F5 Base sentence chunking achieved **3.58%**, versus **73.35%** continuous. These controls use different text sets and do not establish a cross-model ranking. Large continuous F5 error rates must not be used to reject chunked F5. — [Paper](https://arxiv.org/html/2610.00658v1)
- XTTS-v2 supports Russian among 17 languages, 24kHz output, cloning from approximately six seconds, multiple references, and fine-tuning. The official weight license is Coqui Public Model License. — [XTTS-v2 model card](https://huggingface.co/coqui/XTTS-v2)
- The CPML explicitly limits the model **and outputs** to noncommercial use under the provided grant. — [Actual CPML](https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt)
- The maintained `coqui-tts` implementation is now under Idiap's repository; its README states Ubuntu testing and expected Mac/Windows compatibility. This is useful maintenance evidence but does not itself certify XTTS's MPS throughput or maximum Russian quality. — [Maintained code](https://github.com/idiap/coqui-ai-TTS)
- A Russian IPA XTTS fine-tune exists (`omogr/xtts-ru-ipa`) and advertises CC-BY-NC-SA, but its sparse card provides no convincing quality evaluation; a derivative's label does not erase the underlying CPML obligations. — [Fine-tune model card](https://huggingface.co/omogr/xtts-ru-ipa)
- Current official VibeVoice README demonstrates long conversational generation and names English, Chinese and unspecified other languages. Its December 2025 experimental realtime language list does not include Russian. This research did not establish a current official Russian-quality commitment or Russian audiobook benchmark. — [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice)

### Inferences
- Practical recommendation: short paragraph or sentence-group synthesis, fixed reference voice, preserved punctuation, explicit chapter boundaries, automatic transcript matching, regeneration of missed/repeated passages, followed by native-listener spot checks for stress and intonation. Chunk boundaries should be tuned by listening, not chosen solely for speed.
- ASR cannot be the sole acceptance criterion: it can transcribe a word correctly despite wrong stress, and does not capture fatigue, acting quality or unnatural joins.
- A sensible audition includes a literary paragraph, dialogue, questions, long syntactic structures, ё/е ambiguities, stress homographs, names, abbreviations, numerals and foreign words, then at least a 10–15 minute continuous listening sample produced by the actual chunked pipeline.
- The highest-quality final system may be a chosen model plus rigorous normalization and regeneration, rather than the model with the lowest short-sentence WER.

### Gaps
- No sufficiently documented Russian XTTS fine-tune was found that justifies displacing the newer shortlist.
- Russian VibeVoice operation may exist unofficially; lack of verification is not a claim that it cannot produce Russian.
- This research downloaded no weights, synthesized no audio, and did not listen to online samples. Audition judgments remain necessary before declaring a winner.
