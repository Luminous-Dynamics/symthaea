# Melothaea task-indexed musical identity quotient v0.1

**Tracking:** [#14151 — MEL-HEAR-010](https://github.com/Luminous-Dynamics/symthaea/issues/14151)  
**Implementation:** standalone research harness in research/melothaea-musical-quotient-v0-1/  
**Date:** 2026-11-28  
**Status:** source and synthetic fixtures authored; exact-head qualification pending

## Executive thesis

A universal musical-similarity score collapses distinct questions. The same chord quality can appear under a different root, inversion, voicing, instrument, performer or tempo. Those transformations preserve different properties. The correct abstraction is a **query-indexed relation**:

    x ~q y

where q is the declared musical property. We may construct a hard quotient only when the relation is a true, task-preserving equivalence on the complete declared domain. A similarity score, sampled invariance, human judgement, or attractive embedding cluster is not by itself permission to discard candidates.

The consumer pipeline is:

    question → target property → action certificate → property map → complete task contract → quotient certificate → representative search → lift → original-task verification

This adapter reuses action admission and quotient compilation from [#11616](https://github.com/Luminous-Dynamics/symthaea/issues/11616), pair/path receipts from [#12253](https://github.com/Luminous-Dynamics/symthaea/issues/12253) and [#14144](https://github.com/Luminous-Dynamics/symthaea/pull/14144), and real-audio task qualification from [#13260](https://github.com/Luminous-Dynamics/symthaea/issues/13260). It does not duplicate those systems.

## 1. Exact symbolic v0 slice

### 1.1 C12 pitch-class transposition

The finite action is:

    T_k(p) = (p + k) mod 12, k in Z_12

The group has twelve elements, identity T_0, composition T_a after T_b = T_(a+b mod 12), and inverse T_(12-k mod 12).

For triad quality and root-relative interval signatures, a global transposition is invariant. For root pitch class, it is equivariant: the output root moves by the same group element. Therefore:

- chord-quality and relative-interval queries may use the exact C12 quotient in the bounded symbolic fixture;
- root-pitch queries must preserve the group action in their answer and may not quotient away the root.

### 1.2 Positive rational constant tempo scaling

A uniform duration transform is:

    d_i' = r d_i, r in Q_positive

For positive rational durations, ratios d_i/d_j remain invariant, while pulse rate and absolute duration transform equivariantly. The implementation stores reduced positive fractions in u64 fields and uses checked composition. The representable subset is not closed under every rational composition because checked arithmetic can overflow. Therefore v0 treats tempo scaling as an exact arithmetic/property-map fixture but always classifies it SoftRelationOnly for quotienting; it is not a complete continuous-group certificate. Rubato is not a constant tempo action: a local time warp does not generally preserve all inter-onset ratios.

### 1.3 Product action and commutators

The exact product action is C12 × Q_positive (rational), acting on independent pitch-root and duration coordinates. The factors commute. This gives a small cross-domain known-answer fixture for the path/commutator work in EVAL-045. It does not imply that all real musical edits commute, nor does it replace the exact-path receipts.

### 1.4 Reversible quotient fixture

The finite fixture enumerates 48 symbolic chords: 12 roots × four triad qualities. It constructs four C12 orbits, one representative per quality, with twelve original members and a transposition lift witness in each orbit. The verifier checks that every original candidate is recovered from its representative and witness.

The factor-of-12 reduction is an exact **candidate-count property of this constructed fixture**, not a runtime speedup measurement, model-performance result or evidence of real-audio equivalence. The benchmark must charge quotient construction, certificate verification, memory and lifting before it reports practical speedup.

## 2. Relation classes and fail-closed policy

| Relation class | Hard quotient? | Permitted use |
|---|---|---|
| Exact finite C12 action, complete declared domain, invariant query | Fixture eligibility only in v0; production requires #11616 verifier | Exact symbolic orbit tests |
| Bounded rational tempo-factor arithmetic | No in v0 | Exact property maps and soft comparisons |
| Equivariant query | No | Transform the answer with the action |
| Restricted partial isomorphism | No in v0 | Domain-local research; never extrapolate |
| Human-calibrated performer/take nuisance | No | Soft retrieval or grouping with uncertainty |
| Sampled/learned symmetry proposal | No | Candidate ordering and compute allocation with unquotiented fallback |
| One-way or lossy transform | No | Explicit change/coverage analysis; possible abstention |

The implementation's FixtureHardQuotientEligible is not production permission. RUNTIME_PRUNING_AUTHORIZED_V0_1 is hard-coded false. The v0 does not authenticate digests or consume a trusted #11616 V2 certificate.

## 3. Negative controls

The standalone tests include:

- all 12 C12 elements, composition, identity and inverse closure;
- exact relative-interval and chord-quality invariance alongside root equivariance;
- positive rational tempo identity, inverse, composition, reduction and overflow rejection;
- pitch/tempo product commutation;
- sampled, human-only, partial-domain and unknown-domain controls;
- rubato falsely declared as constant tempo scaling;
- lossy crop, timbre substitution and unsupported chord inversion;
- duplicate and incomplete orbit rejection;
- complete orbit membership/lift round-trip over all 48 candidates.

The test suite is authored; no Rust test or Clippy pass is claimed before exact-head CI.

## 4. External benchmark landscape (checked 2026-11-28)

The relevant SOTA is multidimensional. We should not compare Melothaea's symbolic identity contract to a general generation Elo and call that a model win.

### Music identity and perceptual robustness

[MusicListenBench](https://arxiv.org/abs/2611.29375), posted 2026-11-22, tests paired FLIP/STAY decisions across melody, harmony, timbre and tempo. Its paper reports 99.0% on held-out clean rendered pairs after GRPO, but 25.7% false flips on sound-only STAY controls. On its real-recording set the trained system called 27 of 42 same-chord two-strum pairs different. This is an especially important comparator because it separates hearing a difference from knowing whether the queried musical property changed.

### Temporal grounding

[TAG-Bench](https://arxiv.org/abs/2609.01542) contains 1,750 human-verified query-recording pairs and 149.5 hours. Its reported best mIoU is 31.2, recall at IoU ≥ 0.7 is 21.5%, and the best maximum-occurrence-count accuracy is 13.2%. The benchmark also retains parsing failures as empty predictions. Temporal grounding, exact occurrence counts and reliable abstention deserve their own Melothaea track.

### Broad music understanding

The [MOSS-Music](https://www.moss-ai.com/models/moss-music) published table reports an 80.38 average over eight public music Q&A tasks, with 89.39 on MuChoMusic. [Music Flamingo](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7f64034009f4a5fa417a57e1a987c5cd-Abstract-Conference.html) reports strong results on a different collection, including 74.58 on MuChoMusic and 65.60 on MMAU-Pro-Music. These are useful baselines, but published tables, prompts, splits and scoring implementations must be reproduced before treating scores as directly comparable.

### Commercial generation and editing

[Artificial Analysis](https://artificialanalysis.ai/methodology/music) evaluates generation using blind pairwise human preferences, Bradley–Terry estimation and confidence intervals. Its vocal leaderboard reports ACE Music v1 at 1161 ± 19 and Suno v6 at 1144 ± 17. It evaluates generation preference, not correctness of musical explanations or edit contracts.

The product frontier also matters. [Suno Studio 2.0](https://help.suno.com/en/articles/13670529), updated 2026-11-23, combines generation with MIDI, automation, effects, synths and custom plugin workflows. [Google Lyria RealTime](https://deepmind.google/models/lyria/lyria-realtime/) supports continuous interactive music generation with direct controls including key, tempo, density and brightness. Melothaea should integrate with such workflows rather than attempt to win by recreating a generative DAW.

## 5. Prospective benchmark targets

All numbers below are proposed targets, not achieved results.

| Track | Baseline to reproduce | Initial target | Stretch target |
|---|---|---:|---:|
| FLIP/STAY property identity | MusicListenBench clean + real recordings | ≥90% clear-case accuracy; ≤5% false-STAY flip and ≤5% FLIP miss | ≥95% / ≤2% / ≤2% |
| Performer/take invariance | #13908 crossed performer × source work | ≥90% accuracy on held-out performers and works; report each nuisance separately | ≥95% with calibrated abstention |
| Temporal grounding | TAG-Bench | mIoU ≥40; Recall@0.7 ≥30%; occurrence count ≥25% | 50 / 40% / 40% |
| Calibration | Per-property reliability and risk-coverage | ECE ≤0.05; nominal 95% intervals 93–97% coverage | ECE ≤0.03 with no subgroup collapse |
| Quotient correctness | Exhaustive C12 and tempo fixtures | 100% C12 closure/inverse/lift checks; zero missed candidates | Same on a held-out exact symbolic family |
| Quotient efficiency | Exhaustive unquotiented search | ≥2× total-cost reduction after construction, verification and lift costs | ≥5× on a larger exact family |
| Safety of hard quotient | Wrong-symmetry and partial-domain controls | Zero hard quotient for unknown, partial, human-only, sampled or one-way relations | Zero on adversarial mutation suite |
| Temporal/output reliability | Parsing and repeated-event controls | ≥99% valid structured outputs; count errors reported separately | ≥99.9% |

For a sampled false-positive claim below 1% with a one-sided 95% confidence bound and zero observed errors, approximately 299 independent negative instances are required. Multiple clips or windows from one recording are not independent source works. Cluster the design by composition, performer, session and renderer as appropriate.

## 6. How to beat the field

The path is not to make one aggregate score look larger. It is to target known failure mechanisms:

1. **Counterfactual pairs, not caption-only scores.** Make the model answer property-specific questions over matched FLIP/STAY variants. Change the target property while holding nuisance factors stable; change the nuisance while the target remains fixed.
2. **Property maps, not global embeddings.** Represent which dimensions are invariant, equivariant, empirical-only or destroyed. A transposition should not erase root identity; a tempo scale should not erase pulse-rate changes.
3. **Intervention algebra, not isolated edits.** Compose exact transformation paths and test identity cycles, inverses and commutators. Catch accumulated drift and non-commuting edits before execution.
4. **Evidence-backed perception.** Separate raw audio, predicted symbolic structure, oracle symbolic structure and no-evidence controls as in #14147. Never leak oracle labels into audio-derived arms.
5. **Selective uncertainty.** An honest abstention on masked, cropped, out-of-domain or ambiguous audio is better than a confident hallucinated chord or timestamp.
6. **Search-space reduction with proof and lift.** Use #11616 only when the complete task contract admits it; preserve original representatives and verify lifted answers against the unquotiented task.

## 7. Execution sequence

- **V0 (this increment):** exact symbolic C12 and bounded rational-tempo arithmetic, property classifier, reversible finite orbit fixture, strict schema, standalone Rust tests.
- **V0.2:** add octave/voice-permutation finite actions with explicit domains; add stabilizer and orbit-size checks; test partial-domain and omitted-element attacks.
- **V0.3:** build an adapter to #11616's complete-task V2 certificate and independent verifier. Until this exists, keep runtime pruning disabled.
- **Audio qualification:** materialize rights-cleared exact audio; render through at least two independently identified renderers; connect EVAL-045 path receipts to #13260 audio verification and #12327 cumulative audible drift.
- **Human confirmatory cohort:** recruit compensated musicians, preserve independent annotations, preregister equivalence margins, and split by performer and source work.
- **External comparison:** reproduce MusicListenBench and TAG-Bench protocols; submit to any credentialed challenge only when registration, rights and artifact requirements are satisfied.

## 8. Claim ceiling

This implementation concerns exact symbolic group actions and finite orbit coverage. It does not establish general music understanding, real-performer invariance, audible equivalence, improved generation quality, or production search acceleration. Those claims require the corresponding independent audio, human, timing, statistical and exact-head evidence.
