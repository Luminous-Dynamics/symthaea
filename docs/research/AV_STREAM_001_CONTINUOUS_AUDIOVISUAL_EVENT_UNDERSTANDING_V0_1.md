# AV-STREAM-001 — Continuous Audiovisual Event Understanding

**Status:** Research protocol v0.1 — preregistered design, not implementation or qualification  
**Created:** 2026-11-18  
**Parent issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12837  
**Base reviewed:** Symthaea main at 1ec2699b324bc3133678b1f5162b25b03f522b63

## 1. Executive thesis

Symthaea should not treat perception as a requirement to first convert every input into text. A transcript is one useful projection of speech, but it discards or weakly represents information that matters for continuous perception: non-speech events, overlap, onset and offset, prosody, spatial direction, reverberation, source movement, acoustic texture and uncertainty.

Likewise, a video is not a bag of independent frames. It is a time-evolving observation stream in which objects persist, move, occlude one another, cause changes and reappear.

The proposed research target is an **evidence-grounded audiovisual world model** that maintains native modality state and binds it into explicit, revisable event/entity hypotheses. It should retain the evidence and timing uncertainty behind each claim.

This does not mean one giant HDC vector should replace every encoder. The proposed representation is hybrid:

- native audio and visual signals retain modality-specific semantics;
- compact/local features support fast continuous updates;
- learned dense encoders provide semantic evidence where needed;
- geometry and spatial audio remain explicit;
- HDC supports compositional binding and associative retrieval;
- CfC/LTC supports continuous-time dynamics and prediction;
- a typed event/entity graph preserves the world model, uncertainty and evidence lineage.

The goal is not to claim human-level perception. The goal is measurable, robust, task-useful understanding under real streaming constraints.

## 2. Source audit: what exists and what it does not establish

The following is a source-level inventory, not a physical-device test or full build.

| Subsystem | Repository evidence | Honest status |
|---|---|---|
| Visual manifold | `crates/domains/symthaea-vision-manifold` | Substantial patch-HDC, CfC, attention, tracking and scene-memory foundation |
| Physical camera | Optional V4L2 implementation in the vision-manifold crate | Hardware feature exists; root integration was removed because its hardware path had no active config gates and was untested |
| Android screen | `symthaea-phone-embodiment`, ADB capture and optional scrcpy stream | Concrete screen/video capture paths; not universal desktop or application access |
| Live microphone | `src/perception/audio_stream.rs`, gated by `voice-stt-live` | CPAL + ring-buffer + worker path emits acoustic HDC snapshots |
| Native lexical ASR | `crates/core/symthaea-stt` committed report | Not viable on its documented LibriSpeech evaluation |
| External ASR seam | `symthaea-communication` provider contract and Whisper worker | A batch/provider path exists; live streaming integration is a separate open work item |
| Spatial audio | OWL spatial-acoustic work, including #7926 | Useful geometry and localization research; not evidence that current microphone input provides array-based 3D localization |
| Cross-sensor timing | #621 | Explicit work exists for uncertain clock transforms; do not assume sensors share a clock |
| Cross-modal fusion | #633 | Dependence-aware, modality-specific fusion is already an owner; this document does not create a competing fusion framework |

### Important audio findings to fix before capability claims

The committed native STT evaluation report records 100 utterances, WER 473.4394%, PER 98.90585%, zero correctly transcribed words and RTF 0.07028. The RTF describes the evaluated acoustic processing path, not successful lexical transcription. The crate README explicitly describes its useful niche as acoustic/HV representation, prosody and acoustic-unit research rather than viable word ASR.

Separate open issues already track important source-level correctness risks:

- **#10813:** live microphone capture currently overwrites the device-reported sample rate with the requested rate, does not explicitly downmix interleaved channels, and does not account for all dropped samples.
- **#10894:** the bounded streaming buffer appears to lose its absolute origin when old samples are removed, risking frame-index drift or a post-capacity stall.
- **#10910:** offline trained and live acoustic encoders are not currently the same representation. The live path hashes LTC state with BLAKE3 to produce an HV; a cryptographic digest is suitable for identity, not locality-preserving acoustic similarity.
- **#10890:** some current prosody fields are placeholders or rough proxies, not qualified measurements of pitch, speaking rate or voice quality.
- **#10788:** the live PCM-to-streaming-ASR/full-duplex integration remains separate from the existing latest-wins acoustic-HV channel.

These are separate repair/qualification items. AV-STREAM-001 must consume their exact qualified outputs; it must not silently work around their failure states.

### Important vision distinctions

The vision manifold provides 16,384-dimensional patch-HDC encoding, temporal CfC state, surprise-driven attention, object/scene memory and optional depth/object binding. That is a strong foundation for temporal state, but it does not by itself prove open-vocabulary object recognition, robust video question answering, calibrated depth or general audiovisual reasoning.

The optional V4L2 camera implementation is not equivalent to a verified cross-platform camera product. Android screen capture is not desktop capture. A depth feature populated by a neutral stub value is not a physical depth measurement.

The excluded legacy `symthaea-perception` crate contains older SigLIP/Moondream ONNX work, but its exclusion from the active workspace is explicit. Re-admission should happen only through the provider-neutral vision work already tracked by #3934/#8152—not by assuming the old crate is a deployed, integrated model stack.

## 3. What “naive” real-time understanding should mean

Use precise terms:

1. **Raw-stream access:** consume camera/audio samples as they arrive, without requiring a pre-recorded file.
2. **Native representation:** encode audio and video directly into modality-specific states without requiring ASR text or image captions as the only intermediate.
3. **Semantic perception:** infer events, objects, speech content, relations or source identity from those representations.
4. **Grounded world understanding:** maintain persistent, revisable beliefs about entities and events, tied to time-bounded source observations.
5. **Interactive competence:** use those beliefs to answer, request more evidence or propose actions under independent authority gates.

Symthaea has parts of 1 and 2 for selected paths. The existence of those paths does not establish 3–5.

A direct audio-to-latent pathway can understand sound classes, speech-like structure or spatial cues only to the extent that its features have been trained or designed to preserve those distinctions. A direct HDC mapping is not automatically a semantic recognizer. The same applies to image patches: a high-dimensional encoding is not automatically object identity or scene understanding.

**Answer to the core question:** Symthaea can be designed to ingest and update from video/audio streams without forcing a transcript-first architecture. It cannot currently be described as generally understanding arbitrary live video and audio. To reach that capability, the native stream must be paired with qualified perception models, temporal/spatial grounding, reliable synchronization, persistent belief revision and task-specific evaluation.

## 4. Representation proposal

Do not collapse every modality into one opaque vector. Preserve four complementary representation layers.

### 4.1 Modality-native streams

**Vision**
- frame and region identity;
- image patches and appearance features;
- object/region hypotheses and geometry;
- motion and camera ego-motion;
- occlusion and visibility;
- depth only when sourced from a qualified sensor/model.

**Audio**
- original sample-frame range, rate, channel and transform lineage;
- log-mel or other defined spectral features;
- acoustic-event hypotheses and onset/offset;
- lexical hypotheses from a qualified ASR provider;
- prosody/voice-quality features only when genuinely measured;
- source bearing or location only when microphone geometry and localization evidence support it.

**Digital environment**
- screen/window/application identity and permission scope;
- OCR/UI/accessibility tree where permitted;
- game or application telemetry from authorized interfaces;
- source and capture-session identity.

These are distinct observation types, not interchangeable tensors.

### 4.2 Multi-rate temporal state

Use separate time scales rather than forcing every signal through one update cadence:

| Layer | Candidate time scale | Purpose |
|---|---:|---|
| Signal/event front-end | 5–100 ms windows, as supported by the sensor | Onsets, transients, motion changes, frame/audio feature extraction |
| Local event state | 100 ms–2 s | Sound events, speech turns, object motion, short actions |
| Scene/event state | Seconds | Entity association, interaction, event order and source grounding |
| Episodic memory | Seconds–minutes and longer | What happened, where, when, what changed and what remains uncertain |

These are design ranges to measure, not fixed sensor guarantees. Preserve the original measurement interval and clock uncertainty across all rates.

### 4.3 Evidence-grounded event graph

A scene-level hypothesis might contain:

- entity hypothesis and competing identities;
- event/action hypothesis;
- temporal support interval;
- visual and acoustic observation references;
- spatial relations and coordinate frames;
- model/profile and calibration applicability;
- confidence semantics and uncertainty;
- currentness and contradiction state;
- derivation lineage and exact source roots.

Keep epistemic origin distinct: observed, inferred, remembered, predicted, simulated and counterfactual states must never silently become one another.

### 4.4 HDC/CfC's proper role

HDC remains valuable for:
- compositional binding of entity, attribute, role, location and event concepts;
- associative retrieval and robust memory;
- linking representations across modalities after association is qualified;
- compact temporal summaries and similarity-based hypothesis retrieval.

CfC/LTC remains useful for:
- irregular-time updates;
- continuous state evolution;
- prediction and temporal surprise;
- multi-timescale memory.

Neither should replace:
- raw signal provenance;
- calibrated geometry;
- modality-specific likelihoods;
- explicit unknown/abstention;
- independently evaluated semantic recognition.

The central hypothesis is **native evidence + task-preserving learned features + HDC composition + continuous dynamics + explicit world state**, not “larger hypervectors solve perception.”

## 5. Fair baseline matrix

Compare at least these arms:

| Arm | Architecture | Question |
|---|---|---|
| A0 | Vision-only, audio-only, transcript-only | What can each modality solve independently? |
| A1 | Strong modular pipeline: ASR + visual encoder/detector + audio-event/localization specialist + explicit association | Strong practical modular baseline |
| A2 | Pinned end-to-end audio/video model | How does a unified multimodal model perform when its true input limits are respected? |
| A3 | Existing Symthaea VisionManifold + native acoustic HDC/CfC | What can native continuous representations do today? |
| A4 | Native streams + frozen dense specialists + ASR + event graph | Does a hybrid improve the quality/resource frontier? |
| A5 | A4 plus active observation | Does asking for more evidence selectively improve task outcomes? |

For every model or service, record exact checkpoint/model revision, weights digest, preprocessing, audio and video sampling rates, context window, prompt, decoding, precision, runtime, license, network requirement, session behavior and data-retention configuration.

A hosted API with a low-latency stream is a useful industry baseline, but its access mode, video frame rate, audio packetization and server-side computation must be reported. Do not compare it to an on-device model as if the resource boundary were equal.

### Current speech baseline landscape

The native Symthaea lexical baseline is currently far behind established ASR. It is not rational to claim it can beat Whisper/Qwen/Parakeet/Nemotron on general word recognition before fixing #10813/#10894/#10910 and training a recognizer that actually learns lexical distinctions.

The 2026 ASR landscape includes strong encoder/transducer and speech-language-model systems. NVIDIA's published nine-test-set comparison in its 26.10 performance documentation reports 4.78% average WER for Nemotron Transcribe FP8, 4.94% for Qwen3-ASR-1.7B and 6.66% for Whisper large-v3 under that specific protocol. Those figures are not universal WERs and should not be transplanted to South African languages or Symthaea's own cohort. They are evidence that the lexical baseline is already strong and the comparison must be matched.

For **streaming** recognition, include NVIDIA Nemotron 3.5 ASR Streaming as a separate comparator: NVIDIA describes a 0.6B cache-aware FastConformer-RNNT model with a 40-language/locale profile and automatic language detection. Its language set must be checked against each South African target profile rather than inferred from the “40 languages” label. Model card: https://build.nvidia.com/nvidia/nemotron-asr-streaming/modelcard

For **joint audio/video streaming**, include a current Qwen Omni Realtime API arm where terms, region and session behavior permit. Alibaba's documentation updated 2026-11-12 lists Qwen Omni Realtime models accepting text, streaming audio and visual input and returning text/audio over WebSocket, WebRTC or AOQ. Qwen's published model family also reports real-time audiovisual response capability. Treat latency and SOTA statements as vendor claims until independently reproduced under the same scenario and resource boundary. API docs: https://www.alibabacloud.com/help/en/model-studio/omni-realtime-api ; Qwen3-Omni model card: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking

Industry parity therefore requires two distinct comparisons: (1) streaming ASR quality/latency, and (2) joint audio-video perception/response. A strong ASR model is not by itself a video-understanding baseline, and an end-to-end real-time API is not a fair on-device baseline unless its server-side compute and network boundary are reported.

Sources:
- NVIDIA Speech NIM ASR performance: https://docs.nvidia.com/nim/speech/26.10.0/reference/performances/asr/performance.html
- Open ASR Leaderboard: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard (confirm the live leaderboard revision at run time)
- OpenAI Voice Agents architecture guide: https://developers.openai.com/api/docs/guides/voice-agents

The opportunity is not to claim a generic WER victory based on HDC. It is to improve capabilities that transcript-only systems do not directly preserve, and then separately compete on lexical recognition with a real trained recognizer.

## 6. External benchmark plan

Benchmark suites are complementary. Report their native protocol and do not blend them into a single “multimodal IQ” score.

| Suite | What it measures | How AV-STREAM uses it |
|---|---|---|
| OmniEchoBench / OmniEcho (2026) | Spatial audio-visual perception and sound-guided navigation; 197 real scenes, 2,972 QA pairs and 900 navigation samples in the paper | Primary spatial audiovisual grounding anchor |
| Video-MME-v2 (2026) | Robustness/faithfulness, visual aggregation, temporal modeling and multimodal reasoning | Offline long-video robustness anchor; not online latency evidence |
| StreamingBench | Streaming-video evaluation across 18 tasks, 900 videos and 4,500 QA pairs, with questions introduced at multiple points in each stream | Direct online-causality anchor; re-run current models because the published leaderboard is dated 2024–2025 |\n| EgoSchema | 3-minute egocentric video QA over 5,000+ curated questions | Long-horizon event/memory anchor |
| DCASE 2026 Task 3 | Semantic acoustic imaging and sound-event localization/detection from spatial audio and audiovisual scenes | Audio event/localization and audiovisual comparison |
| LOCATA | Multi-array/moving-source localization with tracking ground truth | Spatial audio localization/temporal tracking |
| Video-MME (CVPR 2025) | Broad short-to-long video analysis with audio/subtitle ablations | Established video baseline |
| PazaBench / AfriVox / NCHLT / Lwazi | African and South African speech recognition, accent/domain variation and code-switching | Lexical speech lane through #10913/#10942 |
| FA-Bench | Word/phone alignment and timestamp quality | ASR timing lane, separate from lexical correctness |

Research references:
- OmniEcho: https://arxiv.org/abs/2609.23407
- Video-MME-v2: https://arxiv.org/abs/2604.05015
- StreamingBench: https://streamingbench.github.io/\n- EgoSchema: https://egoschema.github.io/
- DCASE 2026 Task 3: https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes
- LOCATA: https://www.locata.lms.tf.fau.de/datasets/
- Video-MME: https://video-mme.github.io/
- FA-Bench: https://arxiv.org/abs/2609.32396

Dataset terms, access conditions, source identity, speaker grouping and derived-dataset lineage are part of admission. No restricted corpus is redistributed or downloaded in CI.

## 7. Small deterministic audiovisual crucible

Start with a 40–60 scenario pilot before large external datasets or heavyweight model integration.

Include:
- visible event with matching sound;
- visible event without sound;
- sound event from an off-screen source;
- speech plus overlapping speech and non-verbal vocalization;
- visually similar objects and identity swaps;
- camera motion versus true object motion;
- partial and long occlusion, disappearance/reappearance;
- misleading or delayed subtitles;
- deliberately swapped audio tracks;
- known timestamp skew/drift and uncertain measurement intervals;
- duplicate, stale, reordered, dropped and frozen frames/chunks;
- clipping, silence, missing channels and calibration faults;
- ambiguous/no-signal cases that must remain unknown.

Use paired minimal counterfactuals: change only the audio, only the video, only the clock relationship, or only one event. This tests whether a system actually uses the relevant evidence instead of exploiting a shortcut.

Symtropy retains the canonical hidden state and outcome labels. Symthaea receives only rendered observations and ordinary source metadata. No future frame, transcript reference, object ID, source coordinate ground truth or hidden simulator variable enters the online perception path.

Partition by scene generator seed, source episode, environment, speaker/session and nuisance-factor family. Adjacent clips and repeated sources must not cross confirmatory splits.

## 8. Metrics

### Task quality
- Event precision/recall and event-based F1 with declared onset/offset tolerance.
- Audiovisual association precision/recall, top-k coverage and ambiguity preservation.
- Event-order accuracy and temporal interval overlap.
- Spatial angular error, top-k source inclusion and 3D error only with qualified geometry.
- Object tracking metrics only when the emitted geometry supports the official metric contract; otherwise report the actual tracker output metric.
- WER/CER and code-switch boundary measures for lexical speech, reported separately from acoustic-event and spatial performance.
- Evidence-grounded long-horizon QA and temporal localization.

### Calibration and evidence behavior
- Brier, NLL and ECE only for outputs with defined probabilistic semantics.
- Risk/coverage curves and selective accuracy.
- Unknown/abstention on ambiguous or unobservable scenes.
- False persistence, stale evidence and unsupported claims.
- Correct retention of observation roots, derivation lineage, clock and calibration context.

### Streaming/system
- First detection, first hypothesis, stable hypothesis and finalization latency: p50/p95/p99.
- Throughput, missed deadlines, jitter, queue age and backpressure.
- Dropped samples/frames and recovery behavior.
- Peak RAM/VRAM, load time, CPU/GPU utilization.
- Energy per stream-minute only with calibrated measurement; otherwise NOT_MEASURED.
- Quality/latency/memory/compute/bandwidth Pareto frontiers.

Measure acquisition-to-answer end-to-end. Neural-kernel time alone is not user-visible latency.

## 9. Proposed gates

These are research targets, not achieved results or industry standards.

1. Every deterministic known-answer fixture passes; seeded clock mismatch, future leakage, modality swap, identity swap, stale evidence and corrupted provenance negatives are rejected.
2. The online producer is causal: no future frame, reference transcript, hidden label or unavailable modality input.
3. Any audio/frame loss is accounted and creates an explicit discontinuity; no false temporal continuity.
4. At minimum compare A1, A3 and A4 before claiming a Symthaea advantage; add A2/A5 where resource boundaries can be matched.
5. A quality win target is at least 5 absolute percentage points on at least two preregistered task strata, with paired 95% confidence intervals excluding zero and no more than 2 absolute points of regression on the declared overall primary score. A pilot and power analysis may revise the threshold before confirmatory runs, never after seeing confirmatory labels.
6. A candidate efficiency win is at least 25% lower measured compute or peak memory at a preregistered non-inferiority margin initially set to 1 absolute task-score point; revise only with a documented pilot/power analysis.
7. Report every stratum, including failures, unsupported modalities and NOT_EVALUATED cases. No global SOTA claim from one favorable slice.

## 10. Runner-independent execution order

1. Freeze task definitions, event vocabulary, splits and score denominators.
2. Author deterministic paired audio/video fixtures and negative controls.
3. Build offline replay adapters for A0/A1/A3/A4.
4. Add causal streaming arrival schedules, chunk invariance, latency decomposition and revision accounting.
5. Run rights-reviewed external datasets and exact model baselines.
6. Qualify one physical camera/microphone profile at a time, including measured timing, geometry, capture loss and power.

The first three steps can be implemented and reviewed without remote runners or physical hardware. Synthetic success does not qualify physical capture or real-world semantics.

## 11. Stop conditions and nonclaims

- Do not claim current native Symthaea STT has usable lexical ASR; its current committed evaluation is a known failure.
- Do not claim general live audiovisual understanding from the existence of VisionManifold, microphone capture or an embedding.
- Do not assume HDC dimension implies semantics, calibration or causal grounding.
- Do not claim SOTA from comparisons with different frame/audio rates, model versions, hardware, captions, context or provider budgets.
- Do not enable recording by default or inherit permission across camera, microphone, desktop, app and system audio. Each source needs explicit, revocable scope and visible session state.
- Perception never grants physical action authority.

A negative result is a valid outcome. Keep the simplest architecture that meets the task/resource profile; retain HDC/CfC only where it demonstrates a measurable contribution.

## 12. Cross-workstream dependencies

- Vision Mind: #3934; representation comparison: #8152.
- Live input: #10788, #10813, #10894, #10910, #10890, #10842.
- South African speech: #10913, #10942.
- Spatial timing/fusion: #621, #633, #7926, #7946.
- Speech alignment/interaction: #10871, #5169.
- Integrated benchmark issue: #12837.

This document freezes an evaluation direction only. Implementation, executed benchmark results, physical measurements and independent replication remain separate evidence states.


## 13. November 2026 research update: interaction timing and on-demand event memory

Two 2026 results sharpen the benchmark and architecture priorities. They should be added as explicit evaluation arms and design references, not treated as proof that Symthaea already has the capabilities.

### 13.1 DIVE: full-duplex audiovisual engagement

DIVE (October 20, 2026) evaluates response decisions over continuously arriving audio/video, rather than measuring only clip QA or average response latency. It contains 525 evaluation instances drawn from 413 base scenarios, across six task dimensions:

| DIVE task | Capability to qualify in Symthaea | Existing owner to compose |
|---|---|---|
| VTI — Video Timely Intervention | Identify an emerging visual mistake/risk and intervene inside its valid window | #3934, #12837 |
| SPI — Speech Proactive Intervention | Detect a speech/semantic trigger and decide whether a timely interruption is warranted | #10842, #5169, #10890 |
| CQI — Continuous Query Interaction | Retain visual state while answering successive grounded queries | #3934, #12837 |
| ORA — Overlapping Response Adaptation | Revise or withdraw an in-progress response when new audio/video evidence arrives | #10842, #5169, #10871 |
| AVAT — Audio-Visual Addressee Tracking | Infer whether an utterance is directed to the assistant, another person or an ambiguous group | #10842, #5169, #10840 |
| CAT — Continuous Action Tracking | Count/track recurring actions from a persistent stream | #3934, #12837 |

The DIVE paper reports a best average response-window rate of 53.77% among its evaluated systems (GPT-Realtime-2.1), 41.43% for Qwen3.5-Omni-Plus-Realtime and 40.58% for Gemini-3.1-Flash-Live. The same paper reports sub-second average response latencies for leading proprietary systems but notes that response-window compliance remains weak; for example, the best SPI result in its table is 24.8%. These are results under DIVE's specific setup, not universal vendor rankings.

**Why this changes our strategy:** response latency alone is the wrong objective. The system must decide whether to speak, when to speak, when to hold, when to interrupt, and when to revise an already-started utterance. False intervention and unnecessary interruption must be scored as costs, not ignored.

Add the following DIVE-compatible output contract to the evaluation adapter (not a new turn controller):

- decision: speak / hold / intervene / continue / revise / withdraw;
- triggering event reference and evidence roots;
- target/addressee hypothesis and uncertainty;
- valid response interval and estimated decision time;
- current utterance revision/supersession identity;
- observed response outcome (missed, early, on-time, late) and reason;
- explicit no-trigger controls, with false-intervention cost.

**Proposed research gates (not current results):**

1. At least 70% macro response-window compliance on a frozen DIVE-like confirmatory set, with every task stratum reported. This is an ambitious target above the paper's best reported average; claim superiority only after exact protocol reproduction and paired uncertainty analysis.
2. At least 90% precision for proactive interventions on admitted trigger events, and no more than 5% false intervention on no-trigger controls.
3. p95 decision latency below 500 ms from the first sufficient evidence timestamp, not from packet arrival or end-of-clip.
4. On ORA cases, correctly revise or withdraw an ongoing response when contradictory/new evidence arrives; report adaptation quality, interruption cost and revision latency separately.
5. Maintain addressee uncertainty when the camera/microphone arrangement cannot establish who is being addressed. Do not infer identity from voice alone.

DIVE paper: https://arxiv.org/abs/2610.28405

### 13.2 OASIS: memory should retrieve decisive events, not more tokens

OASIS (CVPR 2026) reframes long streaming-video reasoning as **temporal routing**: when the current window and summary do not decide a question, the system should infer what historical evidence would decide it and retrieve that narrow event region. It uses three temporal tiers—high-fidelity short window, medium-resolution buffer and a multi-resolution event hierarchy—with coarse reasoning first and targeted fine retrieval only when needed.

In the paper's StreamingBench table, the reproduced Qwen3-VL-8B baseline scores 72.83% Real-Time All; Qwen3-VL-8B + OASIS scores 78.22%. Anomaly Context Understanding (ACU) rises from 35.63% to 42.74%, Misleading Context Understanding (MCU) from 35.74% to 49.60%, and Sequential QA from 43.90% to 48.40%. These are paper-specific reproduced comparisons and should not be conflated with the current official leaderboard or an independent Symthaea result.

**Symthaea research hypothesis:** combine a persistent HDC/CfC event memory with evidence-indexed temporal retrieval. HDC similarity can propose candidate memories, but selection must be verified against exact source intervals, event identity, modality and clock uncertainty. Retrieval must not turn a semantically similar event into a claim that the same event occurred.

Candidate memory tiers:

- short: loss-accounted, high-fidelity current sensor windows;
- medium: event summaries with exact source interval references and uncertainty;
- long: hierarchical event/entity relations, including alternatives, contradictions and state changes;
- query path: coarse answer from short/summary state → identify missing decisive evidence → targeted retrieval → evidence-grounded answer or abstention.

A memory node is an index/summary, never a replacement for its source. Preserve immutable observation roots and event lineage; a derived summary does not become independent evidence. The evaluator must detect retrieval of a visually/audio-semantically similar but temporally or causally irrelevant event.

OASIS paper: https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.pdf

### 13.3 Expanded benchmark matrix

| Benchmark | Add to programme for | Important caveat |
|---|---|---|
| DIVE | Full-duplex response timing, proactive intervention, addressee tracking, overlap adaptation | Response-window quality must be separated from response latency |
| OASIS + StreamingBench | Hierarchical event memory, retrieval precision and streaming temporal reasoning | Reproduce the exact baseline/model/frame-rate protocol |
| DCASE 2026 Task 3 | Dense semantic acoustic imaging from low-channel spatial audio, with optional video | Four-channel evaluation input; do not claim 3D localization from mono capture |
| DCASE 2026 Task 4 | Spatial semantic segmentation of sound scenes | Separate audio separation quality from label prediction accuracy |
| DCASE 2026 Tasks 5–6 | Audio-dependent QA and long-audio moment retrieval | Preserve long-range audio timestamps and event localization |
| PMLR 302 African ASR study | Data-scaling and model-selection baseline across 13 African languages and 1–400 hours | Results are language/cohort-specific; do not assume transfer to South African varieties |
| AssemblyAI Research Rivals snapshot (2026-11-14/16) | Current production streaming ASR vendor comparison under a common corpus/scoring pipeline | Vendor-published comparison; disclose endpoint defaults and methodology; add independent South African evaluation |

DCASE 2026 Task 3: https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes
DCASE 2026 overview: https://dcase.community/challenge2026/index
African ASR study: https://proceedings.mlr.press/v302/nahabwe26a.html
Streaming ASR vendor comparison: https://www.assemblyai.com/benchmarks

### 13.4 Priority change

The next major research increment should be **evidence-grounded event memory plus decision timing**, not a larger HDC dimension or another generic multimodal encoder. Keep model arms A1–A5 from this protocol, and add:

- **A6 — OASIS-style temporal memory:** three-tier event hierarchy and query-conditioned retrieval, using the same perception backbone as its no-memory control;
- **A7 — DIVE interaction policy:** existing full-duplex controller (#5169 / #10842) receives event hypotheses and returns a decision; do not fork or bypass its authority semantics.

Required ablations: no memory; flat frame RAG; event hierarchy without semantic query refinement; semantic query refinement without evidence verification; full A6; and the same memory with HDC indexing disabled. This isolates whether HDC contributes beyond ordinary metadata/embedding retrieval.

A6/A7 are benchmark and integration arms, not claims of current implementation. Their first fixtures should be deterministic paired scenarios with hidden truth, misleading context, repeated similar events, clock drift, contradictory new evidence, no-trigger controls and explicit response windows.


## 14. Existing memory and interaction owners — no parallel subsystem

The A6/A7 arms are benchmark compositions. They do not authorize a second episodic database, bitemporal ledger, retrieval engine, proof selector or turn controller.

| Capability needed by A6/A7 | Existing owner | AV-STREAM integration boundary |
|---|---|---|
| Visual persistent belief and visual episodic memory | #3934 (VIS-004/VIS-007) | Consume object/event/scene hypotheses with visual provenance; do not create another vision memory store |
| HDC/dense/hybrid representation comparison | #8152 | Add an HDC-index-disabled control and keep downstream task/memory logic frozen |
| Temporal retrieval with query-anchor preservation | #10644 | Provide query obligations, exact temporal spans and retrieval trace; no free-form rewrite that drops entities/numbers/negation/time anchors |
| Valid-time / known-at history | #9075 | A6 queries carry both valid_at and known_at; late corrections must not leak into historical as-known answers |
| Proof-aware minimal sufficient context | #9177 | Retrieve the least-cost complete evidence packet or return Unknown/BudgetInsufficient; never claim a truncated packet is sufficient |
| Cross-clock mapping | #621 | Keep audio sample coordinates and camera PTS incomparable until a qualified uncertain transform exists |
| Dependence-aware multimodal fusion | #633 | Preserve modality-specific likelihood/evidence; no confidence laundering or double-counting shared capture roots |
| Full-duplex floor/STOP/yield semantics | #5169 | DIVE-style decisions route through the existing controller and retain latching STOP/yield behavior |
| Shared HDC-CfC dialogue research | #10842 | Evaluate shared latent contribution against modular L0; it does not supersede controller authority |

### Evidence packet for temporal memory

An A6 retrieval response should contain:
- original question and immutable query digest;
- preserved anchors and explicit temporal obligations;
- valid_at and known_at query coordinates;
- exact event/revision IDs, source observation roots and half-open source intervals;
- source modality, clock domain, transform/calibration version and uncertainty;
- relation to the claim (supports / contradicts / context-only / excluded);
- why candidate memories were selected or rejected;
- proof completeness and budget status;
- derived-summary identity and lineage (never promoted to an independent observation);
- verifier result or an explicit abstention state.

For audiovisual events, the packet must not imply simultaneity from similar numeric timestamps alone. An unknown clock relation remains unknown, and a source gap remains a gap.

### DIVE decision packet

An A7 request to the existing turn controller should include only qualified event hypotheses and source evidence plus a proposed interaction decision. The controller remains responsible for latching stop/yield, active floor, interrupted-turn identity and renderer stop acknowledgement. Report event-to-controller delay and controller-to-output acknowledgement separately; an early model prediction is not a completed intervention.
