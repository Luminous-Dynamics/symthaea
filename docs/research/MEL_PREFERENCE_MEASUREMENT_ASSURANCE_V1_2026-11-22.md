# MEL Preference Measurement Assurance v1

**Date:** 22 November 2026  
**Programme:** Melothaea / MEL-BENCH  
**Implementation:** tools/mel-preference-assay  
**Status:** reference evaluator and synthetic conformance fixture; no human study result.

## Why this increment exists

Melothaea already has a matched-run contract that records a display-order seed and hashes the compared outputs. That is necessary but does not, by itself, establish that each presentation used its assigned order, that a rater saw a pair only once, or that evaluator side preference was measured.

The external field makes this a substantive issue. MIREX 2026's Music Evaluation via CMI-RewardBench reports pairwise preference accuracy on a 249-example core set and vocal/instrumental extensions. Its published table shows CloudEar-overall at 73.9% on the core set, CMI-RM at 65.9%, and an always-choose-B baseline at 59.0%. The table also reports a SYSTASYS submission at 39.8% and a flipped variant at 60.2%. This is an orientation warning, not proof that every evaluator has position bias; the result is enough to justify explicit orientation controls in our own measurement protocol.

References:
- MIREX 2026, [Music Evaluation via CMI-RewardBench results](https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results).
- Ma et al., [CMI-RewardBench, ICML 2026](https://proceedings.mlr.press/v306/ma26am.html), which defines a heterogeneous music reward evaluation spanning musicality, text-music alignment, and compositional instruction alignment. Its resources include a 110k pseudo-labelled corpus and human-annotated preference data.
- Artificial Analysis, [Music Generation Benchmarking Methodology](https://artificialanalysis.ai/methodology/music), which uses blinded pairwise human votes, Bradley-Terry maximum-likelihood ratings, 95% intervals, and normalized audio delivery.

## Paradigm shift: qualify the instrument that measures quality

A preference score is not self-authenticating. The judge, presentation interface, randomization, exposure policy, and analysis pipeline are part of the instrument. MEL-PREF-ASSAY makes those observable and replayable.

The reference evaluator implements five controls:

1. **Committed counterbalancing.** A seed commitment is recorded before collection; after collection, the revealed seed deterministically reconstructs each presentation's A-left/B-left assignment.
2. **Assignment conformance.** The recorded left candidate must equal the deterministic assignment. Any mismatch rejects the assay.
3. **Exposure integrity.** A pseudonymous rater cannot see the same case or the same candidate pair twice in one assay.
4. **Identical-output controls.** Byte-identical candidate pairs measure left/right choice tendency without a genuine content difference.
5. **Order-conditioned outcomes.** Candidate-A selection is reported separately when A appears left and right. Per-case order gaps are descriptive diagnostics; they are not causal estimates unless the study's randomization and sampling assumptions are satisfied.

This is complementary to—not a replacement for—CMI-RewardBench, MIREX, or Artificial Analysis. Those systems measure evaluator/model agreement or human preference on their native cohorts. MEL-PREF-ASSAY audits whether a Melothaea comparison run is sufficiently well-formed to interpret.

## Competitive qualification plan

Targets below are prospective. They are not current Melothaea results.

| Dimension | First real-study gate | Stretch target |
|---|---|---|
| Assignment integrity | 100% assigned orders reproduced; zero mismatches | 100% with an externally timestamped precommit receipt |
| Exposure integrity | Zero repeated rater/case or rater/pair presentations | Same, with independently audited UI telemetry |
| Identical-control position effect | At least 400 decisive duplicate controls per evaluator cohort; report the 95% interval | At least 2,500 controls per cohort for roughly ±2 percentage-point precision near 50% |
| Order invariance | Report per-case A-win rate in both display orientations; minimum 10 judgments per orientation on each eligible case | Mean absolute matched-case gap ≤3 percentage points on at least 50 cases, with uncertainty intervals |
| Preference evaluator | Reproduce the native MIREX/CMI-RewardBench baseline on its official split before claiming comparison | At least 85% pairwise agreement on an independently rights-cleared Melothaea holdout, reported separately by musicality, text adherence, and compositional instruction |
| Calibration and abstention | Retain ties, neither, skips, invalids and failures as explicit outcomes | Publish calibration curves and abstain when evaluator disagreement or uncertainty is high |

The control-count precision objectives assume independent Bernoulli decisions near 50%; clustered raters and repeated tasks require cluster-aware uncertainty estimates and a larger effective sample. A simple Wilson interval is a descriptive first gate, not a replacement for a hierarchical model or a preregistered equivalence test.

## How Melothaea should outperform the field

Do not try to win this workstream by claiming a higher raw preference percentage on a different sample. The defensible advantage is measurement assurance:

- preserve each benchmark's native split and scoring method;
- run exact matched prompts, modality, candidate budgets, normalization, and retry policies;
- bind candidate bytes, evaluator versions, adapters, prompts, and judgment protocols by digest;
- preregister assignment, exclusions, stopping rules and analysis before collection;
- report evaluator accuracy, calibration, side-bias controls, abstentions and subgroup performance separately;
- keep evaluator output advisory: a reward score cannot approve, commit, or publish a creative edit.

Only after this measurement layer passes should Melothaea compare candidate generators or reward models. A preference evaluator must not grade its own authority.

## Claim ceiling

The Rust crate is a deterministic reference validator over supplied JSON. Its fixture is synthetic. It does not authenticate raters, attest UI playback, validate audio rights, prove an external timestamp, reproduce CMI-RewardBench, or establish a live musician preference result. Human-study work remains physical/pending until a rights-cleared cohort has been run and independently analyzed.
