# MEL-MR-EVAL-005 — Rendered-Audio Preservation Under Edit Contracts

**Status:** research protocol, draft v0.1.0  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8814  
**Depends conceptually on:** MEL-MR-EVAL-004 (symbolic edit locality), without replacing or weakening its gates.

## 1. Question and falsifiable hypothesis

Can an editor apply an authorized musical change while preserving protected properties in both the symbolic score and rendered audio?

Hypothesis: a typed edit contract, paired with separate symbolic, acoustic, and human evidence, detects collateral edits that a clean-edit-only symbolic benchmark or a global perceptual score would miss.

This document defines an evaluation protocol. It does **not** claim that MEL-MR-EVAL-004 currently verifies rendered audio, that the audio detectors have been validated, or that Symthaea outperforms a state-of-the-art system.

## 2. Contract semantics

An edit contract is bound to source content and describes an authorization mask over **time × track/source × property**. It must identify:

- authorized regions (frame/tick interval and track/source IDs);
- requested changes and their target values or conditions;
- protected invariants and per-property tolerances;
- renderer and instrument/effect configuration when rendering is controlled;
- verifier, detector, configuration, and schema versions;
- unsupported conditions that require abstention rather than optimistic acceptance.

A hash over content is an integrity commitment; it is not signer identity, a trusted timestamp, or proof of origin. Those claims require a separate authenticated-signature policy.

The initial machine-readable draft is in `mel-mr-eval-005-contract-schema-v0.1.json`; a non-authoritative example is in `mel-mr-eval-005-contract-example-v0.1.json`.

## 3. Four non-compensatory evidence channels

### A. Symbolic contract audit

Reuse the MEL-MR-EVAL-004 verifier for its supported event model. Record requested-property success and each protected-property result separately. Do not infer audio preservation from a passing symbolic receipt.

### B. Acoustic preservation audit

For controlled MIDI-to-audio pairs, pin the renderer, bank, effects, sample rate, channel layout, render seed where supported, and latency-compensation settings. Where these are unavailable, record that limitation.

Measure the authorized mask and protected regions separately. Candidate signals include aligned onset/transient traces, F0/pitch activity where defined, note duration, dynamics/envelope changes, multi-resolution spectral features, and source/track attribution. Calibrate every detector against human annotations and known synthetic mutations. Record alignment uncertainty and detector measurement floors.

Do not use raw waveform equality as the primary criterion: phase and synthesis details can alter samples without altering the musical property under evaluation. Do not let a single learned embedding or global average hide a localized violation. Learned semantic checks may reject uncertain claims but should not independently create a passing verdict.

For effect tails, shared buses, reverberation, and other expected spillover, the contract must define permitted spillover. The verifier must not silently mark the whole mix protected when the edit authorization is wider.

### C. Human perceptual evaluation

Use randomized, blinded A/B comparisons (or A/B/X when source identity must be assessed). Ask separate questions for:
1. audible collateral change in nominally protected regions/properties;
2. musical naturalness / artifacts;
3. overall preference.

Do not collapse these into one score. Preference cannot compensate for a critical protected-property failure. Normalize loudness with a documented method for preference trials, while preserving a separate dynamics-preservation evaluation so normalization does not erase a relevant difference.

### D. Replay and provenance

A replay record should bind the contract, source and output symbolic/audio commitments, renderer/configuration identity, tool and detector versions, per-channel results, and final verdict. Replay recomputes the record and compares the full structured receipt, not only its final Boolean.

Hashes detect accidental or cross-input reuse under the encoded content model; they do not authenticate who produced the record. If signing is later added, define a separate key lifecycle, canonical serialization, rotation, revocation, and verification policy.

## 4. Corpus design

Build in two stages.

**Stage 1 — controllable measurements:** synthesize known MIDI inputs and rendered outputs under pinned settings. Include monophonic and polyphonic material, instruments with sharp and slow attacks, sustain, rests, tempo changes, velocities, and effects. Every positive and negative must include explicit ground truth for requested and protected properties.

**Stage 2 — realistic shifts:** add multi-track stems, DAW exports, mixtures, humanization, reverb, shared buses, overlapping sources, renderers not used in calibration, and edits from capable systems where comparable task access exists.

Required mutation families:
- successful authorized edit with no collateral change;
- under-edit / requested change absent;
- no-op and unsupported request;
- global or off-mask accidental edit;
- one-axis and multi-axis collateral leakage;
- boundary-tolerance cases;
- alignment / source-separation / detector confounds;
- renderer mismatch and stale/mismatched receipt;
- malformed content or missing required evidence.

Hold out **composition/motif families and renderer configurations**, not merely random rows. No family may cross development, calibration, and final test splits. Freeze all thresholds on calibration data before test evaluation. The final set must not be used to tune thresholds, detectors, masks, or exclusion rules.

Pilot planning targets: at least 50 independent composition families per split and 300 human-annotated clip pairs with multiple independent raters. Expand the sample when intervals remain too wide. Publish the annotation guide, adjudication rules, and exclusions.

## 5. Baselines and fairness

Use, at minimum:
1. deterministic transform with known ground truth;
2. no-op / under-edit negative control;
3. injected collateral-mutant controls;
4. simple edit path without a locality guard;
5. at least one capable editing system where access, task controls, and output formats permit a matched comparison.

Use the same source inputs, contracts, masks, rendering configurations, and evaluator. Document unavoidable provider differences. Mark an unsupported capability **not comparable**, not as a win or loss.

Adjacent references inform methodology but are not automatic head-to-head baselines:
- MAGE (2026), multimodal music generation and mixture-grounded editing: https://research.google/pubs/mage-modality-agnostic-music-generation-and-editing/
- TimingBench (2026), timestamped temporal control with detector calibration and rejection gates: https://taan.ai/research/rushing-or-dragging/
- CMI-RewardBench (ICML 2026), compositional multimodal instruction alignment and reward evaluation: https://proceedings.mlr.press/v306/ma26am.html
- Artificial Analysis Music Arena, blind pairwise preference methodology and confidence intervals: https://artificialanalysis.ai/methodology/music

## 6. Metrics and proposed targets

Report by operation, composition family, renderer, instrumentation, polyphony, effects class, and distribution-shift stratum:

| Metric | Reporting rule | Proposed goal (not achieved) |
|---|---|---|
| Requested-change success | Per operation; include false rejection | ≥95% on held-out annotated edits |
| Protected-property violations | Per protected axis and overall | ≤5% on held-out edits |
| Critical-mutant false accepts | Numerator, denominator, one-sided 95% CI | 0 observed in ≥300 adversarial negatives; report nonzero population-rate upper bound |
| Receipt replay | Exact structured recomputation | 100% deterministic replay; reject tampered/cross-input records |
| Abstention | Separate from accept and reject | Publish per condition; do not hide unsupported cases |
| Human agreement | Agreement statistic and annotation design | Report inter-rater reliability and adjudication |
| Preference | Blind pairwise estimate with 95% CI | No superiority claim unless matched comparison supports it |
| Runtime | Rendering and verification separately | p50/p95, realistic event/audio lengths |

Zero observed errors is not proof of zero risk. For a proportion, publish the confidence interval and sample denominator; do not state “zero false accepts” without the tested panel size.

## 7. Non-compensatory decision rule

The top-level result is accepted only if all required gates pass:

```text
accepted =
    contract_valid
    AND symbolic_requested_change_passes
    AND symbolic_critical_invariants_pass
    AND acoustic_requested_change_passes
    AND acoustic_critical_invariants_pass
    AND evidence_replay_passes
```

Human preference is reported alongside the technical verdict, not used to override it. If a required detector is outside its validated domain or measurement uncertainty exceeds the contract tolerance, return `abstain` / `unmeasurable` rather than `pass`.

A future implementation may refine this rule only through a versioned protocol change and adversarial regression evidence; thresholds must not be relaxed merely to improve scores.

## 8. Implementation sequence

1. Review the contract schema and this protocol independently.
2. Build schema validation, corpus integrity checks, and a deterministic independent reference oracle.
3. Define and calibrate the acoustic detector panel on controlled renders; publish measurement floors and sensitivity analysis.
4. Add mutation tests for every critical gate and require negative controls to be rejected.
5. Run the human-annotation pilot with a frozen guide and preregistered analysis.
6. Execute Rust tests, formatting, lint and benchmarks on an exact commit; attach run evidence.
7. Only then evaluate matched external systems and report confidence intervals.

The implementation should remain Rust-first for production. Python/JavaScript may be used as disposable independent research oracles, with source, version, and reproducibility instructions documented; they are not production dependencies.

## 9. Execution honesty

A draft schema, authored test, static fixture audit, or queued/skipped workflow is not evidence that the Rust implementation passes. Every execution claim must include the exact tested commit and a real run result. The follow-up remains a protocol proposal until the schema, acoustic measures, human annotations, and implementation have been independently exercised.
