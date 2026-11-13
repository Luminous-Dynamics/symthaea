# MEL-MR-EVAL-030 — SMEBench / MMAE external music-edit benchmark adapter

**Date:** 13 November 2026  
**Status:** first dataset-manifest and comparison-scope contract authored; no benchmark dataset has been downloaded, scored, or rights-cleared by this change.  
**Tracking issue:** [#9284 — Benchmark Muse against paired-target music editing and MMAE](https://github.com/Luminous-Dynamics/symthaea/issues/9284)

## 1. Purpose

Create a reproducible bridge from Muse's typed editing/evidence pipeline to external music-editing benchmarks without confusing:
- a score-level symbolic correctness test with a complete audio-editing benchmark;
- synthetic MIDI-rendered examples with real multitrack recordings;
- a music-only slice with full multimodal audio editing;
- an official 500-example competition test track with the public 2,000-example development set;
- an operation that is represented in a type system with an operation that has actually been qualified end to end.

This first implementation is deliberately metadata-only. It validates dataset/task commitments, source-lineage splits, access/rights declarations, a 12-operation capability matrix and strict comparison scopes. It does not download audio, run a model, calculate a benchmark score, certify rights, or establish SOTA.

## 2. External benchmark protocols (reviewed 13 November 2026)

### SMEBench

SMEBench constructs paired source/instruction/target audio from known MIDI interventions. Its published protocol includes:
- twelve operations covering instrumentation, drums, articulation, dynamics and pitch/tuning;
- a balanced 3,600-example test set;
- a separate 416-pair real multitrack complement derived from 107 tracks;
- independent source-preservation, target-agreement and text-compatibility measures.

The paper reports that metric choice can change system rankings. It also includes human target-validation caveats for difficult operations (including legato, instrument removal and drum replacement); those operation-level caveats must remain visible in Muse reporting. The benchmark is an external anchor, not evidence of Melothaea performance.  
Paper: https://arxiv.org/abs/2610.36715

### ICASSP 2027 MMAE Audio Editing Challenge

The official challenge defines:
- a public development benchmark of 2,000 examples and 17,741 atomic rubrics;
- a separate 500-example Single Model test track;
- a separate 500-example Agent test track;
- independent ranking of the two tracks;
- IFR (instruction following), CR (consistency/preservation) and EMR (all required instruction and preservation rubrics satisfied), with EMR as the primary ranking metric.

The Agent Track explicitly permits an orchestrator to plan, invoke local models and DSP tools, inspect intermediate outputs and revise. The published baseline uses an LLM router, DSP for speed/volume/pitch, SAM-Audio for separation and AuK for generative editing.

The official baseline repository reports:
- Agent, full 2,000-example benchmark: IFR 44.06%, CR 74.63%, EMR 7.45%.
- Single Model, 1,003-example single subset: IFR 42.12%, CR 75.78%, EMR 7.58%.

These are different scopes and must not be compared as if one were a better model result.  
Official challenge: https://audio-editing-challenge.github.io/  
Leaderboard and evaluation protocol: https://audio-editing-challenge.github.io/leaderboard/  
Baseline implementation/results: https://github.com/Audio-Editing-Challenge/Audio-Editing-Challenge-Baseline

The challenge registration deadline was 8 October 2026; the Agent model-version cutoff was 1 November; the hidden test/SDK release was 10 November; and the submission deadline is 25 November (11:59 PM Pacific). The adapter does not imply that Luminous Dynamics is a registered team, has organizer credentials, or is eligible to submit. Do not access or claim hidden-set evaluation without authorized credentials and confirmation of team membership.

## 3. Paradigm: benchmark tasks as lineage-bound transformation contracts

A benchmark row is not merely an input WAV plus a prompt. It must bind:
- immutable dataset release and native track;
- source recording/composition lineage;
- exact source audio digest;
- exact instruction digest;
- expected target audio digest;
- source symbolic artifact when available;
- operation type and target scope;
- target selector (instrument, drum kit, event group or mixture);
- annotation/metadata digest;
- split and dataset stratum;
- data-rights and access evidence.

The source-lineage digest is the grouping key for split integrity. Different edits or paired renditions derived from one underlying composition/recording may live in the same split, but they must never cross train/development/test. This protects against leakage through alternate renders, stems, remixes or near-duplicate source recordings.

The comparison layer requires exact benchmark ID/version, track, stratum, split, protocol digest, scorer digest, native metric set and sample-manifest digest. A music-only subset cannot be labelled directly comparable to the full multimodal benchmark.

## 4. First implementation

### Rust module

Added `crates/domains/symthaea-muse/src/music_edit_benchmark_adapter.rs`, exported by the Muse library.

The module defines:
- the exact 12 SMEBench operation identifiers;
- dataset rows, data strata, native splits, access tiers and rights-use declarations;
- source-lineage grouping and duplicate sample rejection;
- explicit operation qualification states (Qualified, NotYetQualified, Unsupported, AbstainOnly);
- evidence-digest requirements for any Qualified operation;
- strict comparison-scope matching with mismatch reasons.

The validator does not open audio files or verify that their digests match bytes. A later materializer/runner must establish file existence, content hashes, rights, source annotations and scorer behavior before an actual run.

### Current operation qualification matrix

All twelve SMEBench operation families are currently marked **NotYetQualified** in the synthetic contract fixture. This is intentional, not a statement that no related code exists. Related lower-level capabilities do not establish complete operation-level qualification.

| SMEBench operation | Why a related capability is not enough |
|---|---|
| Remove instrument | Single-event removal is not complete part removal |
| Replace instrument | Needs program/bank identity and target-part preservation |
| Extract instrument | Needs qualified part-isolated stem and renderer evidence |
| Double instrument | Needs orchestration identity and leakage-safe stem validation |
| Remove drums | Needs multi-event/percussion-track semantics |
| Replace drums | Needs drum-kit/note identity and target verification |
| Extract drums | Needs part-isolated audio qualification |
| Staccato | Needs exact duration/tick semantics and acoustic articulation validation |
| Legato | Needs exact duration/overlap semantics and acoustic validation |
| Crescendo | Instrument-level and mixture-level targets must be separate |
| Decrescendo | Instrument-level and mixture-level targets must be separate |
| A4=440 tuning correction | Integer MIDI semitones are not cent-level tuning correction |

A Qualified status requires a pinned qualification-evidence digest. The adapter rejects a Qualified status without evidence. Unsupported/unqualified tasks must remain explicit in the run denominator; they may not be silently emitted unchanged and scored as successful.

### Isolated harness and workflow

Added:
- `tools/mel-music-edit-adapter-harness/`: standalone Cargo workspace using serde/serde_json only;
- synthetic dataset and capability-matrix fixtures;
- Draft 2020-12 schemas;
- `.github/workflows/muse-music-edit-benchmark-adapter.yml`: exact-head Rust 1.96 format/test/fixture-check/Clippy workflow.

The workflow stages the exact production module into the standalone harness, so it does not need to resolve every crate in the full Symthaea workspace. The synthetic fixture check validates the serde contract and Rust validator; it does not certify the JSON Schema implementation or an external dataset.

## 5. Benchmark execution protocol

For any eventual SMEBench/MMAE run:

1. Pin the exact dataset revision, source manifest, split manifest, native scorer and task protocol.
2. Verify license/permission and intended use before reading, evaluating, training on, redistributing or publishing assets. An accessible download is not evidence of rights.
3. Deduplicate source-lineage IDs and enforce split isolation before producing candidate outputs.
4. Preserve paired source/target examples and render/source metadata. Do not split a source and its paired target across datasets.
5. Run no-op/identity, exact symbolic oracle, conventional DSP/MIDI, Muse and external baseline systems under the same scope.
6. For synthetic SMEBench examples, keep renderer, SoundFont, sample rate, mastering and level controls matched wherever the native protocol permits.
7. Treat the 416 real multitrack pairs as a separate stratum; do not infer real-audio generalization from MIDI-derived synthetic pairs.
8. Report the twelve operation families separately, plus the published human-validity caveats. Do not let easy operations hide failures on hard ones.
9. Keep instruction following, target agreement, preservation, text compatibility, perceptual preference, abstention and resource use separate.
10. Retain every attempted sample. Failed, timed-out, unparseable, unsupported and abstained cases remain in the denominator.

### Required baselines

- No-op/identity control (must fail any non-no-op requested change).
- Exact symbolic target renderer (oracle ceiling only).
- Conventional DSP/MIDI operator baseline.
- Muse contract-driven path.
- Published SMEBench reference systems where accessible and rights permit.
- Official AuK and MMAE Agent baselines where the exact runtime, weights, dependencies and scorer can be reproduced.
- Unconstrained full regeneration as a collateral-change comparator.

## 6. Prospective targets (not measured results)

### SMEBench

- 100% exact symbolic target correctness on hand-built deterministic fixtures for every operation that becomes Qualified.
- Zero false acceptance on immutable-target mutation, wrong-instrument, wrong-region, wrong-operation and source-lineage-leakage controls.
- Initial held-out target-property success target: ≥95% on supported controlled operations.
- Real multitrack complement reported independently; no synthetic-only result is described as real-audio performance.
- All twelve operation results visible; unsupported operations count as abstentions/failures, not missing rows.

### MMAE

- First reproduce the public 2,000-example Agent baseline on the exact local development scope and scorer.
- Initial stretch target: EMR ≥12% on that same full development scope, with IFR not regressing by more than 2 percentage points and CR improving by at least 2 percentage points relative to the reproduced baseline.
- Require paired, source-lineage-clustered confidence intervals before calling an improvement.
- Report a music-modality subset separately. Do not compare its score to the full multimodal 2,000-example baseline.
- Only claim official hidden Agent Track performance after authorized access, matching model-release cutoff and submission requirements.

These are research targets, not achieved performance or a guarantee of beating the leaderboard.

## 7. What would constitute a real advantage over SOTA

Melothaea should not try to win by adding more tools to an LLM router. The official Agent baseline already composes DSP, source separation and generative editing. The differentiator should be:

- **operation correctness:** prove the exact requested symbolic change happened;
- **lineage and preservation:** show which source events/parts remain unchanged;
- **capability honesty:** reject or abstain on tasks outside verified support;
- **scope-matched evaluation:** compare only identical tracks, splits and scorers;
- **resource efficiency:** report cost and wall time to verified acceptance, not only quality;
- **human-calibrated value:** validate perceived musicality, artifacts and usefulness separately from automatic metrics.

A plausible output is not enough. The stronger result is an output that satisfies the edit, preserves everything protected, carries exact artifact lineage, and survives independent evaluation.

## 8. Validation and claim boundary

- Nine Rust unit tests are authored for duplicate IDs, lineage leakage, rights/access evidence, mutable dataset versions, operation coverage and comparison-scope mismatch.
- The standalone workflow is committed; its current status is pending until a fresh exact-head run reports.
- No dataset was downloaded, no external rights were cleared, and no model/benchmark score was produced by this PR.
- JSON fixtures can be parsed and validated by the Rust harness; no JSON Schema validator result is claimed.
- The capability matrix is a conservative snapshot. A later exact-head, per-operation qualification record must update a status from NotYetQualified to Qualified.
