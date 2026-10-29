# MEL-MR-EVAL-014 — Performed Event Map

**Status:** draft implementation primitive; not yet connected to the live renderer or MIDI exporter. No build, test, perceptual, or SOTA gain is claimed.  
**Tracking:** [MEL-MR-EVAL-013 render receipts](https://github.com/Luminous-Dynamics/symthaea/blob/research/mel-mr-eval-013-render-execution-receipt-20261029/crates/domains/symthaea-muse/MEL-MR-EVAL-013_RENDER_EXECUTION_RECEIPT.md)  
**Research anchors:** [SMEBench (Oct. 27, 2026)](https://arxiv.org/abs/2610.36715), [MuseCPBench](https://arxiv.org/abs/2512.14629), [MMAE / Audio Editing Challenge](https://audio-editing-challenge.github.io/leaderboard/).

## Why this increment exists

The existing performance path has an expressive clock (rubato and swing), note-level articulation/humanization, and operations such as de-overlap. A post-hoc match by approximate onset is not a reliable substitute for explicit event ancestry: nearby repeated notes can be ambiguous, a note can be dropped during performance processing, and extra performed layers can create more than one descendant.

The new performed_event_map module defines a fail-closed receipt format and acceptance verifier. Source event IDs use the same global onset-ordered index as Score::events() and the full-score edit verifier (score_event_index:{index}). Each performed MIDI note identifies a MIDI track and note-on ordinal, channel, pitch, time interval and PCM-frame interval. Every target-score event must have one primary descendant; extra descendants must be explicitly represented with contiguous ordinals. Missing or unresolved lineage is not silently repaired.

## Current implementation boundary

Implemented in this draft:

- Versioned, domain-separated SHA-256 receipt over a canonical event map.
- Exact binding fields for target Score digest, performed MIDI byte digest, sample rate and frame count.
- Fail-closed checks for malformed source IDs, unknown/missing score events, unresolved mappings, duplicate MIDI note-on identities, invalid MIDI channel/pitch, invalid event intervals, frame/time disagreement, duplicate or non-contiguous descendant ordinals, receipt tampering and primary-pitch mismatch.
- A combined public acceptance method that checks receipt/context/coverage before exact primary pitch agreement.
- Eight deterministic unit-test functions covering acceptance and negative cases.

Not implemented yet:

- The MIDI parser/adapter that constructs these rows directly from the exact MIDI bytes.
- Source IDs carried through every transformation in theory_realize.rs (strum, humanization, de-overlap, doubling/color layers).
- Binding the verified map into a real RendererExecutionReceipt.
- Deriving the audio-preservation mask from performed timing and independently checking the actual rendered PCM.
- Renderer attestation. Hashes and local execution records do not defeat a malicious host or prove a backend used its declared input.

Until those pieces land, this module is a verifiable data contract—not evidence that current production renders already have event lineage.

## Integration sequence

1. Add lineage to the performed-note representation at the point source Score events are converted. Carry the identity sidecar through every transform; when an event is dropped or cannot be followed, record unresolved lineage instead of recomputing identity from audio.
2. Parse the serialized MIDI bytes just before renderer invocation. Independently reconstruct track/note-on identities and compare their timing/pitch/lineage against the proposed map. Hash the exact MIDI bytes after serialization.
3. Bind the accepted map, renderer invocation, exact executable/assets/config, WAV bytes and decoded PCM into the execution receipt.
4. Compute edit and protected-region windows from performed event intervals, not only nominal score beats plus a constant latency. A mismatch between symbolic and audio evidence is a hard failure.
5. Keep the external FluidSynth MIDI path and the native Muse render path as distinct adapters with their own execution/config evidence.

## Benchmark plan

Use benchmarks for what they actually establish; do not conflate their scores.

### 1. Controlled lineage and contract tests

| Measurement | Admission target |
|---|---|
| Supported score events with a verified primary descendant | 100%; any unresolved event blocks acceptance |
| Exact target-pitch agreement for supported pitch edits | 100% on deterministic MIDI fixtures |
| Duplicate / missing / unknown lineage detection | 100% in enumerated negative controls |
| False accepts | 0 across at least 384 independent critical-negative cases; report the Wilson interval |
| Reproducibility | Identical canonical receipt for identical Score/MIDI bytes, profile and renderer inputs |

The 384-case zero-accept target gives an approximately sub-1% one-sided 95% Wilson upper bound only for that sampled negative distribution. It is not a universal safety guarantee.

### 2. Music-specific paired benchmark

SMEBench is a particularly relevant new external comparison: its Oct. 27, 2026 paper describes paired source and target audio, explicit edit metadata, 12 instrumentation/articulation/dynamics/pitch operations, a balanced 3,600-example test set, and a separate 416-pair real-audio complement. The paper emphasizes that source preservation, target agreement and text compatibility are different quantities and can rank systems differently.

Run the synthetic and real-audio portions separately. Report operation-conditioned exact edit success, target agreement, non-target preservation and perceptual/music-context outcomes. Compare only systems evaluated on the same split with the same renderer, inputs and metric implementation. Do not call a score win if a system simply returns the unchanged source.

Use MuseCPBench as a complementary preservation evaluation across musical facets such as melody, harmony, rhythm, structure and timbre. Keep facet scores separate; a strong timbre score must not compensate for a failed pitch edit.

### 3. General audio-editing benchmark

The official MMAE challenge evaluates instruction-following (IFR), consistency (CR) and exact-match rate (EMR). The published baseline reports 44.06% IFR / 74.63% CR / 7.45% EMR for its 2,000-example agent run; its single-model row uses a different 1,003-example subset and is not directly comparable. Use the full public set and the same evaluator/version for any numerical comparison.

Proposed stretch target for an agent run on the same full MMAE scope: at least 12% EMR, with a paired 95% confidence interval showing improvement over the reproduced baseline; IFR no more than 2 percentage points below the matched baseline; CR at least 2 percentage points higher. These are targets, not current results. Muse's symbolic edit path should not claim broad MMAE superiority from a narrow music-only score.

### 4. Ablations required before claiming a paradigm shift

- No lineage / onset-nearest-neighbor reconstruction.
- Explicit lineage without a frozen contract.
- Lineage plus frozen contract/profile-bound preservation policy.
- Lineage plus actual renderer execution receipt.
- Full stack plus performed-time audio mask and independent PCM audit.

Hold source examples, edit instructions, renderer assets and seeds fixed. Use paired bootstrap confidence intervals for example-level metrics, report per-operation and real-vs-synthetic strata, include an unchanged-source control and invalid-input controls, and publish hashes for manifests, output audio and evaluator versions. Any model-judge score should be accompanied by deterministic symbolic and signal-level checks where applicable.

## External benchmark references

- [SMEBench paper](https://arxiv.org/abs/2610.36715) — paired music-edit targets and separate evaluation of preservation, target agreement and text compatibility.
- [MuseCPBench paper](https://arxiv.org/abs/2512.14629) — music context preservation.
- [MMAE benchmark and official leaderboard](https://audio-editing-challenge.github.io/leaderboard/) — broad audio-editing rubric protocol and hidden-test timeline.
- [Audio-editing benchmark survey](https://github.com/MM-Speech/AudioEditSurvey) — broader taxonomy and benchmark map.
