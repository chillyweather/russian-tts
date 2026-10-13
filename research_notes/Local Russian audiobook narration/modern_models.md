# Modern multilingual local TTS candidates

## Which model should we audition first for Russian audiobook quality?

### Takeaway
**Hardware update: the user confirmed an M1 Pro with 16GB unified memory.** For that actual machine, my conservative first build is **Qwen3-TTS-12Hz-1.7B-Base through MLX**, using a native Russian narrator reference. Audition **OmniVoice** as the lightweight quality challenger and **Fish Audio S2 Pro 8-bit** as the potentially stronger expressive model whose real memory use must first be checked. With unconstrained memory I would audition Fish first, then Higgs TTS 3 and OmniVoice. No verified Russian audiobook listening tournament establishes a winner.

### Cited Findings
- Fish's March 2026 report gives Russian MiniMax-test WER 2.400%, speaker similarity 0.790; Fish S1 is 5.250%/0.764. It reports Russian CV3 WER 2.78%. Its long-audio test is **English/Chinese only**, about 185 seconds expected maximum after truncation, so it does not establish Russian chapter narration. Its English long-audio WER is 4.38% versus Qwen3-TTS 7.69%. These are vendor measurements. — [Fish S2 technical report, Tables 2–4](https://arxiv.org/html/2603.08823v1)
- Qwen's January report gives Russian WER 3.212%/similarity 0.792 for **12Hz 1.7B Base**, versus 4.458% for 0.6B Base. CustomVoice 1.7B's Russian WER is 4.734%. These concern content accuracy and speaker similarity, not human audiobook preference. — [Qwen technical report, Tables 6 and 9](https://arxiv.org/html/2601.15621v1)
- OmniVoice's April report gives Russian MiniMax-test WER **2.233%**, SIM-o 0.783, and reports 20,338.5 hours Russian training data. Its backbone initializes from Qwen3-0.6B. Training pronunciation controls specifically use Chinese Pinyin and English CMU phonemes: do not assume Russian stress/IPA control from general phoneme claims. — [OmniVoice paper, Table 3, Appendix D, Sections 2.5/3.2](https://arxiv.org/html/2604.00688v1)
- Higgs TTS 3's current model card explicitly lists Russian in its WER/CER-under-5 group. It publishes superior multilingual aggregate WER versus S2 Pro/Qwen/MOSS/OmniVoice, but the language sets differ in coverage and include languages unsupported by some competitors; no Russian-only comparative row was found. It has voice cloning and inline emotion/prosody controls, a 4B backbone, 8192-token training context, and 24kHz output. — [Higgs TTS 3 model card](https://huggingface.co/bosonai/higgs-tts-3-4b)
- MOSS-TTS-v1.5 (May 26, 2026, 8B) and Local-Transformer-v1.5 (June 18, 4B) support Russian among 31 languages. Upstream positions v1.5 for long narration, and reports improved clone stability, punctuation following, explicit pauses, and IPA/Pinyin controls; Local v1.5 adds 48kHz stereo. These are capability claims, not a demonstrated Russian audiobook victory. — [MOSS upstream](https://github.com/OpenMOSS/MOSS-TTS)
- Chatterbox's current multilingual release is **V3, 500M**, Russian included. Turbo and Nano are English models. V3 claims reduced repetition/hallucination and improved voice identity. Official tips recommend a reference clip in the requested language to prevent transferred accent. — [Chatterbox upstream](https://github.com/resemble-ai/chatterbox)

### Inferences
- **Actionable quality priority without hardware constraints:** Fish S2 Pro first; Higgs TTS 3 second for expressive delivery; OmniVoice third as a strong Russian accuracy challenger. This is a testing priority, not an established ranking. Higgs's recency and aggregate wins do not by themselves outweigh Fish's Russian-specific evidence. **On the confirmed 16GB M1 Pro**, shift first implementation to Qwen Base 1.7B, then compare OmniVoice and Fish 8-bit. This practical recommendation is not a claim that Qwen sounds better.
- If the user prioritizes permissive licensing and easier hardware more than absolute expressiveness, OmniVoice and Qwen3-TTS-12Hz-1.7B-Base become particularly attractive. Qwen is a useful known baseline; the existence of newer evidence prevents calling it indisputably best.
- MOSS v1.5 is a meaningful second-round comparator on high-memory hardware if first-round models feel too sentence-like. Do not reject it for being 8B if quality is the only objective; do not promote it without Russian listening evidence either.
- Cross-paper values from nominally the same benchmark help select candidates but are not perfectly controlled head-to-head trials. Small differences (OmniVoice 2.233 versus Fish 2.400) do not establish perceptual superiority. ASR WER can miss stress mistakes, breath artifacts, flat discourse, and listener fatigue.

### Gaps
- No audio listening was performed in this research subtask. No independently administered Russian blind test of these exact releases with long literary passages was found.
- Other assigned researchers cover VoxCPM2, CosyVoice3 and Russian specialist systems, which may change the final three-way shortlist.
- Exact release date of Chatterbox V3 and Higgs TTS 3 was not established from a dated primary release note. Current availability is verified; avoid inventing a day.

## Can they run locally, and what hardware and license constraints matter?

### Takeaway
All six named candidates have downloadable weights. Fish's supported upstream route recommends a 24GB GPU; Apple Silicon is feasible through community MLX ports, including full-precision conversions, but exact memory and speed depend on the user's machine.

### Cited Findings
- Fish's current repository directs users to **S2-Pro**, advertised as 4B; the architecture has a 4B Slow-AR plus 400M Fast-AR and a codec. Weights are at `fishaudio/s2-pro`. — [Fish upstream](https://github.com/fishaudio/fish-speech); [weights](https://huggingface.co/fishaudio/s2-pro)
- Official inference documentation recommends **at least 24GB VRAM**. This is a recommendation for its documented implementation, not proof that every optimized port requires 24GB. — [Fish local inference docs](https://speech.fish.audio/inference/)
- Fish's license, updated March 7, 2026, explicitly permits research and noncommercial use, including personal hobby use/evaluation. Commercial purpose needs a separate written license. It is restricted open weights, not an unrestricted Apache/MIT release. — [Fish Audio Research License](https://github.com/fishaudio/fish-speech/blob/main/LICENSE)
- **S2.1 Pro is different**: Fish's June 23 article announces API access and says it builds on the previously open-weight S2. The currently verified open repository still points to S2 Pro; no S2.1 weights were verified. Do not call the hosted S2.1 demo evidence of local S2 quality. — [S2.1 announcement](https://fish.audio/blog/s2-1-pro-free-api/)
- A BF16 Fish MLX conversion is available for mlx-audio (converted using version 0.4.0); an 8-bit conversion is also available. These are community conversions, not Fish's native official implementation. — [BF16 conversion](https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16); [8-bit conversion](https://huggingface.co/mlx-community/fish-audio-s2-pro-8bit)
- A separate pure-MLX implementation, mlx-speech, supports Fish S2 Pro with int8 weights (reported 4.5GB), cloning and tags. The model-weight size is not peak unified memory. — [mlx-speech model documentation](https://github.com/appautomaton/mlx-speech/blob/main/docs/fish-s2-pro.md)
- Qwen's **12Hz 1.7B Base** weights are Apache-2.0. — [official model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- Qwen's BF16 MLX conversion exists and the model page reports 4.54GB. Runtime overhead still applies; do not mistake artifact bytes for a minimum RAM specification. — [Qwen MLX BF16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16)
- OmniVoice has Apache-2.0 code, downloadable weights, and direct upstream NVIDIA/CUDA and Apple Silicon/MPS instructions. — [OmniVoice repository](https://github.com/k2-fsa/OmniVoice); [weights](https://huggingface.co/k2-fsa/OmniVoice)
- MOSS v1.5 weights are Apache-2.0. — [MOSS model card](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
- Chatterbox weights are MIT. — [official weights](https://huggingface.co/ResembleAI/chatterbox)
- Higgs license allows research/noncommercial use and has a creator grant including monetized audiobooks with Boson attribution; model embedding in a commercial product, APIs and resale require separate licensing. — [Higgs model card license summary](https://huggingface.co/bosonai/higgs-tts-3-4b)
- A community Fish MLX port reports **14.4GB BF16 / 10.5GB int8 / 8.7GB int4 peak memory**, measured on **M4 Max 64GB**, not this user's M1 Pro. The small project has only five commits at inspection, so treat results as preliminary, implementation-specific evidence. Its speed metric uses audio duration / compute time (inverse of conventional RTF). — [port author's benchmark](https://github.com/gafiatulin/fish-speech-mlx)
- A commercial app developer reports Qwen Base 1.7B Q4_K_M plus Q8 codec on exactly **16GB M1 Pro**, 2.0–3.4GB process peak and 1.2–1.5x playback speed. Tests cover English/Spanish/German/Japanese, not Russian, and use that app's GGUF engine rather than MLX. This is firsthand vendor measurement, not independent verification or a recommendation to purchase. — [PocketWebTools measurements](https://pocketweb.tools/mac/models/qwen3-tts)

### Inferences
- On the actual **M1 Pro 16GB**, do **not** choose Fish BF16 as the default. Its codec and activations can consume virtually all memory before macOS. Fish8bit is worth attempting with paragraph chunks and only one loaded model, but no guarantee is justified. Qwen1.7B full precision or8bit and OmniVoice are safer initial candidates. Quantization's effect on Russian stress/timbre has not been demonstrated here.
- On NVIDIA 24GB, Fish's upstream route is the clearest documented high-quality starting configuration. On smaller GPUs, Qwen or OmniVoice offers a more realistic first installation, though optimized Fish variants may still work.

### Gaps
- No reliable official peak-memory minimum or Russian audiobook generation speed was verified for Qwen, MOSS, Higgs or Chatterbox. Parameter counts alone cannot determine working memory, especially with long contexts.
- MLX output parity with upstream for Russian was not independently verified.

## What implementation details affect whether these become good audiobook narrators?

### Takeaway
Voice choice and careful segmentation matter as much as picking a top checkpoint. Evaluate narration using a native Russian reference, paragraph-scale passages, and explicit checks for omissions and stress; do not start by giving any candidate an entire novel.

### Cited Findings
- Qwen's official repository dates open release January 22, 2026. Base clones reference audio with its transcript; embedding-only mode can reduce quality. Cached clone prompts can be reused. CustomVoice's nine preset voices have Chinese/English/Japanese/Korean native languages, **none Russian**, and upstream recommends each voice's native language for best quality. VoiceDesign can create a voice and Base reuse it. — [Qwen upstream](https://github.com/QwenLM/Qwen3-TTS)
- The new Russian Balalaika-Longform study warns about omissions and early stops in continuous long generation and explicitly separates its single-pass regime from chunked synthesis. — [Balalaika-Longform, September 30 2026](https://arxiv.org/html/2610.00658v1)

### Inferences
- For Qwen, audition **Base 1.7B with a clean native Russian narration reference**, not just English-native Ryan reading Russian. Its reference/transcript route is preferable to embedding-only cloning when quality is paramount.
- Use several linked sentences per chunk, retain a fixed narrator reference, normalize numbers/abbreviations/names carefully, and retry failed passages. Test consistent pacing and timbre across boundaries rather than relying on impressive ten-second demos.
- A sensible listening panel: neutral narrative, emotional dialogue, long syntactic sentence, rare names/loanwords, homographs/stress traps, numbers/dates, and at least 10–15 minutes of continuous listening. Content verification and perceived naturalness are separate axes.

### Gaps
- Native stress-mark and ё handling needs empirical testing for each exact checkpoint/runtime. Do not assume a common stress-marker convention works across models.
- No model studied here provides independently verified human-narrator-level full Russian audiobook quality without postprocessing or quality control.
