# MEL-BENCH-001 — external capability ledger v1

**Date:** 21 November 2026  
**Parent issue:** [#10417 — reproducible external SOTA and industry comparison ledger](https://github.com/Luminous-Dynamics/symthaea/issues/10417)  
**Machine-readable data:** \`benchmarks/mel/external-capability-ledger-2026-11-21.json\`  
**Schema:** \`benchmarks/mel/schemas/external-capability-ledger-v1.schema.json\`  
**Validator:** \`tools/mel-external-capability-ledger-validator\`

## Objective

Create a source-pinned, machine-readable reference for the systems and benchmarks that Muse should compare itself against. The ledger distinguishes measured results from documented product/API capabilities and from dataset-size facts. It does not manufacture a single cross-industry ranking.

The dated snapshot now has eleven sources and twenty-one observations (21 November 2026), including temporal-control and reward-model benchmarks.

## Snapshot highlights

### Audio preference — separate instrumental and vocal leaderboards

The observed Artificial Analysis AA-Music v1.1 instrumental board lists Suno v6 at 1,148 ±22 Elo. The separate vocal board lists ACE Music v1 at 1,160 ±20 and Suno v6 at 1,145 ±18.

These are blind pairwise preference references for generated audio in their own cohorts. They do not evaluate exact score editing, preservation, reciprocal interaction, authorship, rights, agency or live instrument timing. Instrumental and vocal Elo are separate comparison groups.

Sources:
- https://artificialanalysis.ai/music/leaderboard/instrumental
- https://artificialanalysis.ai/embed/music-generation-leaderboard/leaderboard/vocals

### Symbolic composition — MIREX 2026

The observed MIREX symbolic-generation results list DAMN at 3.68 coherence, 3.39 structure, 3.18 creativity and 3.47 musicality (0–5 ratings); Anticipatory Music Transformer is listed at 3.61 musicality.

The reported study is small (8 prompts, 12 participants and 38 ratings per model) and uses selected outputs from a pool of eight candidates. These numbers are useful reproduction targets, not a universal music-quality metric and not comparable to audio Elo.

Source: https://futuremirex.com/mirex/wiki/2026%3ASymbolic_Music_Generation_Results

### Reward models — CMI-RewardBench (ICML 2026)

CMI-RewardBench evaluates the evaluators: it tests music reward models over musicality, text–music alignment, and compositional instruction alignment when the generation context may include text, lyrics and reference audio. The paper reports 110k pseudo-labelled preference samples and a separate human-annotated CMI-Pref corpus; pseudo-label volume must not be confused with human-label volume.

Selected official-release metrics show why reward models should be compared by task, not by one overall rank:

| Reward model | MusicEval musicality SRCC | Music Arena accuracy | CMI-Pref accuracy |
|---|---:|---:|---:|
| CMI-RM | 0.8266 | 73.43% | 78.20% |
| SongEval-RM | 0.6949 | 73.88% | 72.40% |
| Gemini 3 Pro | 0.6018 | 68.85% | 65.80% |

For compositional alignment, CMI-RM's CMI-Pref accuracy is 67.6% (text), 72.8% (text+lyrics), 76.4% (text+audio), and 79.2% (text+lyrics+audio), versus Gemini 3 Pro at 67.2%, 60.8%, 68.8%, and 64.8%. Gemini 3 Pro still has a higher PAM alignment SRCC (0.5373 vs CMI-RM's 0.4321). The rankings therefore depend on the test set and metric.

**Implication for Muse:** a specialized reward model is a useful independent candidate-evaluation signal, especially for multimodal instruction adherence, but it is neither a music generator nor an edit-authority oracle. Use several separately calibrated signals, keep uncertainty and disagreement visible, and never let a reward score override a formal preservation constraint or artist decision.

Sources:
- [ICML 2026 / PMLR paper](https://proceedings.mlr.press/v306/ma26am.html)
- [Official CMI-RewardBench release and selected result tables](https://github.com/Haiwen-Xia/CMI-RewardBench)

### Temporal controllability — TimingBench

TimingBench (August 2026) measures whether timestamped musical events occur when requested, rather than whether the overall audio merely resembles a prompt. Its 250 30-second instrumental prompts are split into 100 single-event, 100 timed-sequence and 50 global-tempo requests; the corpus defines 399 timed events.

Published benchmark snapshot:

| System | Overall instruction success (95% CI) | Timed events within ±500 ms | Tempo accuracy | FAD |
|---|---:|---:|---:|---:|
| Taan | 96.1% (93.3–99.2) | 91.3% | 92.0% | 4.8 |
| ElevenLabs Music v2 | 28.9% (25.3–32.5) | 12.8% | 87.0% | 3.5 |
| Google Lyria | 23.9% (20.3–27.5) | 8.6% | 89.7% | 2.6 |
| ElevenLabs Music v1 | 25.4% (21.9–29.2) | 8.0% | 93.0% | 3.5 |
| Suno | 22.0% (18.7–25.6) | 6.8% | 86.7% | 3.1 |

The result creates a new, distinct competitive dimension. It also illustrates why timing, tempo and audio fidelity must not be collapsed into one number: Taan leads the reported timing measures, while Google Lyria has the lowest reported FAD (lower is better).

**Evidence caveat:** TimingBench is published by Taan, one of the evaluated systems; independent replication is not established by this snapshot. The public paper reports a 128-event development/calibration validation sample with false credits and false positives, family-specific onset-localization floors, and an unfinished separate 300-clip annotation study at its evaluation cutoff. The ±500 ms metric is not a claim of sub-100-ms live timing or exact MIDI beat-phase control. Use the benchmark's measurement-error profile and reproduce the cohort before claiming parity or superiority.

Source: https://taan.ai/research/rushing-or-dragging/

### Editing — SMEBench

The current paper snapshot describes 11,160 paired source/target examples, a 3,600-example balanced test set, a 416-pair real-audio complement and twelve edit operation types. The benchmark's key methodological separation is source preservation, target agreement and text compatibility. These are distinct evaluation axes.

Source: https://arxiv.org/abs/2610.36715

### Product and research capabilities

- **Suno Studio 2.0:** official documentation describes a browser-based generative production workflow with chat-driven generation/arrangement, audio and MIDI tracks, MIDI editing, automation, effects, synths, custom plugins and stem separation. This is product capability evidence, not independent SOTA performance. https://help.suno.com/en/articles/13670529
- **Google Lyria RealTime:** experimental persistent WebSocket music streaming with weighted-prompt steering and configuration controls including BPM, density, brightness and scale; documented output is 48 kHz stereo 16-bit PCM. It is an audio-stream control comparator, not a note-level symbolic editor. https://ai.google.dev/gemini-api/docs/realtime-music-generation
- **YuE2:** September 2026 score-first full-song planning and realization, with conversational score/arrangement editing described in its research release. Score planning is now a competitive baseline, not a unique Muse claim. https://arxiv.org/abs/2609.33757
- **MAGE:** 2026 multimodal conditional music generation and mixture-grounded target-source editing. This is relevant to audio-mixture editing; it does not establish arbitrary symbolic score lineage or protected-event guarantees. https://arxiv.org/abs/2604.09803
- **JAM_BOT:** NIME 2026 live symbolic improvisation work, including expressive velocity, variable-tempo call-and-response and external MIDI latency compensation. This is a close research comparator for live ensemble behavior. https://nime.org/proc/nime2026_73/index.html

## How to compare without misleading ourselves

Every result should carry:
- source and publication/retrieval date;
- benchmark version, split and license;
- exact model/API/system version;
- modality, task and control surface;
- prompts, decoding, retries and compute budget;
- renderer and audio-chain configuration where relevant;
- scorer version and digest;
- failed/timeout/unparseable outputs and denominator;
- raw predictions/receipts or a precise reason replay is unavailable.

### Comparability rules

1. Compare Elo only inside the same leaderboard cohort and model task.
2. Compare symbolic 0–5 ratings only on the same symbolic benchmark protocol and scorer.
3. Compare edit success on paired source/target tasks with preservation and target agreement reported separately.
4. Compare streaming audio generation separately from MIDI/score control unless both systems expose the same event representation.
5. Compare causal reciprocity only with randomized partner interventions, arm-state isolation, replay/yoked controls and independent musical-relevance evidence.
6. Never claim a product capability is a measured performance result.
7. Never let one composite score conceal a preservation, consent, timing, rights or reliability failure.

## Prospective Muse targets

These are research targets, not achieved results:

| Capability | Benchmark / comparator | Proposed target |
|---|---|---|
| Instrumental audio preference | AA-Music instrumental | First reach statistical non-inferiority on a matched prompt subset; stretch goal above 1,200 Elo with a confidence interval that excludes the dated 1,148 point estimate |
| Vocal audio preference | AA-Music vocal | Separate matched vocal track; do not transfer instrumental scores |
| Symbolic composition | MIREX symbolic | Reproduce published baseline first, then exceed DAMN's reported means on held-out prompts with a larger preregistered musician panel |
| Music editing | SMEBench | ≥95% target-occurrence F1 and ≥95% operation/parameter exact match on musician-adjudicated held-out tasks; 100% preservation on deterministic controls |
| Reciprocal interaction | RMI-Bench | 100% correct discrimination of registered synthetic positive/negative controls; live Muse must beat yoked and one-way controls before any reciprocity claim |
| Live ensemble | JAM_BOT and Lyria RealTime on compatible tasks | Report p50/p95/p99 latency, beat/phase error and missed/false/duplicate entries; zero lookahead and zero backdated events |
| Recovery | MEL-TRG-Music | Exceed reset, local-cue-only and yoked-context controls without degrading preservation or agency |
| Agency/authorship | MEL-CREATIVE-001 | Zero unauthorized committed changes; 100% of commits bound to an approval or explicit autonomy authorization |

Sample sizes, thresholds and confidence intervals must be fixed from pilot variance and preregistered before confirmatory evaluation. Where access or task compatibility is missing, record N/A rather than inventing a comparison.

## Validation

The JSON Schema is provided for external schema tooling. The Rust validator adds strict Serde deserialization and semantic checks for:
- unique source and observation IDs;
- valid source references;
- ISO retrieval/as-of dates and HTTPS source URLs;
- non-empty metric names, units, scopes and finite values;
- non-negative uncertainty and non-zero declared sample sizes;
- restriction of direct audio-Elo comparability to blind-preference audio-generation observations in the named AA-Music cohorts;
- rejection of an Elo metric attached to a non-comparable cohort.

Run locally:

\`\`\`bash
cargo generate-lockfile --manifest-path tools/mel-external-capability-ledger-validator/Cargo.toml
cargo test --locked --manifest-path tools/mel-external-capability-ledger-validator/Cargo.toml
cargo run --locked --manifest-path tools/mel-external-capability-ledger-validator/Cargo.toml
\`\`\`

The validator is an independent standalone workspace and does not load the large root monorepo workspace.

## Known gaps

- API and product feature availability can change; this ledger is a dated snapshot, not an evergreen guarantee.
- Proprietary endpoints may not permit matched prompts, decoding, retries or budgets. Mark those comparisons unavailable when fairness cannot be established.
- Published leaderboard estimates have uncertainty and cohort effects; do not overstate small point differences.
- The ledger records external reference claims. It does not assert that Muse has reproduced or beaten any of them.
