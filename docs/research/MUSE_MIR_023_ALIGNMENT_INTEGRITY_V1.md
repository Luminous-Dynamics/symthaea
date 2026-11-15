# MUSE-MIR-023 — Audio–MIDI alignment integrity benchmark (V1)

**Issue:** #12184  
**Parent:** MUSE-MIR-015 dual-fidelity benchmark  
**Research trigger:** Brandão et al., Assessing Alignment in Automatic Piano Transcription Datasets (2026), arXiv:2611.14443  
**Status:** protocol specification; no benchmark results claimed

## 1. Why alignment must become a first-class benchmark target

AMT training and evaluation generally assume paired audio and MIDI are aligned closely enough to be treated as ground truth. That assumption can fail. A transcription model can be penalized for matching the audible performance when the reference MIDI is shifted, incomplete, or structurally inconsistent. Conversely, a model may exploit dataset-specific annotation defects.

The November 2026 alignment-scoring study provides a useful experimental pattern: create known pitch, onset and structural corruptions, then test whether scoring methods detect them. It also evaluates transcription-derived and audio-feature methods, including a soft pitch-constrained inter-onset-interval score (STI), CENS, spectrograms, onset F1 and DTW-based comparisons.

MUSE-MIR-023 turns that pattern into a reproducible, evidence-bound benchmark. It is complementary to notation-vs-playback evaluation: it qualifies the reliability of the audio–symbol pair used to train or score the system.

## 2. Threat model and questions

The benchmark addresses false model regressions, false model progress, metric blindness, metric brittleness, ranking instability and split leakage.

The benchmark asks:
1. Can a scoring method distinguish clean pairs from controlled corruptions?
2. Does its score degrade monotonically as corruption severity increases?
3. Is it invariant to benign representation changes?
4. Do plausible reference uncertainties change downstream system scores or rankings?
5. How do automatic judgments compare with human judgments?

## 3. Deterministic corruption model

Every transformation is a pure, versioned operation over a parsed MIDI event stream. The transformation receipt includes source audio SHA-256, source MIDI SHA-256, parser and canonicalization versions, corruption family, seed, parameters, transformed MIDI SHA-256 and an ordered event-level transformation log.

### 3.1 Families

| Family | Operations | Key controls |
|---|---|---|
| Onset | global/per-note/per-voice/local shift, drift | 5–500 ms; dense 5–50 ms panel |
| Offset | shorten/lengthen, truncation, note-off jitter | absolute and duration-relative |
| Pitch | semitone/octave/quarter-tone, swaps, transposition | preserve or perturb onset/duration |
| Structure | insert/delete/duplicate/split/merge, chord-member change | sparse high-impact and dense low-impact |
| Clock | PPQ rescale, tempo scale/map corruption, drift | exact-equivalent vs genuinely altered |
| Performance | pedal/controller perturbation | only where audio/render mapping is defined |
| Compound | controlled combinations | primitive identities remain separately recorded |

Velocity is excluded from alignment ground truth unless a specific renderer and acoustic mapping make it testable. It must not be treated as a direct note-to-audio correspondence by assumption.

### 3.2 Severity

Use a preregistered severity grid. For onset shifts, include the published study's 5–50 ms fine-grained region and shifts up to 500 ms. The benchmark records the actual per-note shift distribution, not merely a scalar severity label.

For structural corruption, record exact inserted, deleted, duplicated, split and merged event identities. For pitch corruption, preserve the original event identity and record old/new pitch. Do not use random corruption without recording the seed and realized transformation.

### 3.3 Benign metamorphic transformations

These are negative controls: they change representation but not musical timing/content.
- equivalent PPQ rescaling with all ticks and tempo-map ticks scaled consistently;
- equivalent tempo-map encoding where the piecewise time function is unchanged;
- event serialization reordering where MIDI semantics are unchanged;
- permitted track/channel reordering when identity mapping is preserved.

A metric that changes materially under a declared benign transformation is not representation-invariant for that transformation. Do not silently normalize away changes that alter actual performance time.

## 4. Scoring-method adapters

V1 defines an adapter contract; it does not pretend all methods are implemented by the Rust crate.

Methods to pin/reproduce where public implementations and licenses permit:
- note onset/offset/pitch F1 with tolerance sweeps;
- onset-only and frame-level scores;
- CENS/chroma and constrained DTW;
- spectrogram feature alignment with pinned synthesizer, soundfont and gain policy;
- DLNCO/onset-sensitive features;
- transcription-derived scoring;
- STI as specified by the 2026 paper;
- human pairwise/ABX judgments on a stratified subset.

For each method, record exact code commit, model checkpoint digest, preprocessing, renderer, feature parameters, alignment constraints, score direction and runtime. If exact reproduction is unavailable, label it InspiredBy rather than Reproduced.

## 5. Metrics

Do not collapse these into one score:
- clean-vs-corrupt AUROC and AUPRC by corruption family;
- balanced accuracy at frozen thresholds;
- severity/score Spearman and Kendall correlation;
- minimum detectable shift at fixed false-positive rate;
- clean-pair false alarm rate;
- benign-transform invariance error;
- calibration and selective coverage;
- pairwise agreement with human judgments;
- score/rank stability across plausible reference variants;
- corpus-level alignment-quality estimates with work/performance-clustered intervals;
- runtime factor, latency and peak memory.

Report sensitivity and specificity separately. A method that detects onset shifts but ignores structural corruption must not be summarized as a generic alignment verifier.

## 6. Dataset and split controls

- Split by work/composer/performer/recording session, not random excerpts.
- Keep development/tuning and evaluation references disjoint.
- Freeze method parameters and thresholds before opening the evaluation set.
- Preserve source disagreements and annotation revisions; never overwrite a disputed reference silently.
- Keep commercial-compatible, noncommercial-only and external-evaluation-only corpus tracks separate under MIR-020.
- Do not label a source ByteVerified until the exact files are acquired and SHA-256 hashed.
- Use clustered bootstrap confidence intervals at work/performance level; note-level resampling is not independent.

## 7. Typed receipt and replay contract

Proposed Rust types:
- AlignmentCorruptionSpecV1: family, severity, seed and typed parameters;
- MidiEventIdentityV1: stable source event ordinal and track/channel identity;
- AlignmentTransformationV1: old/new exact event coordinates and pitch/structure edits;
- AlignmentCorruptionReceiptV1: source digests, parser/canonicalization identity, spec, transformation log and output digest;
- AlignmentScorerFingerprintV1: implementation/checkpoint/preprocessing/renderer/parameters;
- AlignmentRobustnessReportV1: per-family metrics, severity curves, invariance controls, coverage and resource use.

Replay must reparse the exact source MIDI, regenerate the transformed event stream and compare the transformed-byte digest and canonical event digest. Any mismatch returns invalid receipt; no partial success. All tick arithmetic and PPQ transforms use checked integer/rational operations. Resource budgets are explicit.

A digest proves content consistency, not who authored a reference, who owns it, or whether its license permits use.

## 8. Qualification targets — not achieved results

1. 100% byte-identical replay for deterministic corruption fixtures.
2. 100% clean-control identity for no-op and mathematically equivalent PPQ transforms under the canonical representation.
3. Zero false negatives for high-severity synthetic structural corruption in the synthetic oracle suite.
4. Monotonic score degradation for the method/family combinations that claim sensitivity, with all violations reported.
5. Publish the full 5–50 ms onset sensitivity panel; do not claim perfect 5 ms detectability.
6. Stretch: AUROC >= 0.90 for high-severity onset and structural corruption on a preregistered held-out set.
7. Publish downstream score/ranking uncertainty intervals and whether plausible reference corrections change the winning system.

These are qualification targets, not measured performance or SOTA comparisons.

## 9. Relationship to public results

The 2026 alignment study is a methodological baseline, not an automatic leaderboard comparator. Its corruption operators, 11 severity levels, onset shift range and STI scoring formulation should be reproduced from the paper and public implementation before comparison. Differences in data, renderer, model checkpoint, corruption sampling or scoring parameters must be disclosed.

The 2026 dual-evaluation study supports separating notation similarity from playback similarity and validates playback metrics against human listening judgments. MIR-023 adds a further question: whether the reference pair itself is sufficiently aligned to support either evaluation.

No direct numeric comparison with vendor AMT results is valid unless source corpus, split, alignment status, metric and renderer are matched.

## 10. Acceptance gates

- [ ] Rust-native deterministic corruption engine and typed receipts.
- [ ] Checked arithmetic, bounded event count and fail-closed malformed MIDI behavior.
- [ ] Fixtures for every primitive corruption family.
- [ ] Benign PPQ/tempo metamorphic controls.
- [ ] Receipt replay rejects altered source, seed, parameters, transformations or output.
- [ ] Adapter interface supports external scorers without embedding unpinned Python runtime into production.
- [ ] Protocol pins all method implementations/checkpoints/renderer configurations.
- [ ] Work/composer/performer disjointness validated.
- [ ] Tests authored, executed and exact-head CI distinguished.
- [ ] No SOTA claim without matched public reproduction and independent held-out evaluation.

## References

- Brandão et al. (2026), [Assessing Alignment in Automatic Piano Transcription Datasets](https://arxiv.org/abs/2611.14443).
- Wang et al. (2026), [A Dual Evaluation for Music Transcription](https://arxiv.org/abs/2608.04511).
- [MUSE-MIR-015 dual-fidelity benchmark protocol](https://github.com/Luminous-Dynamics/symthaea/issues/11794).
- [MUSE-MIR-020 rights-aware corpus registry](https://github.com/Luminous-Dynamics/symthaea/issues/11951).
