# Renderer control-effect qualification v1

**Date:** 23 November 2026  
**Implementation:** `crates/domains/symthaea-muse/src/renderer_effect.rs`  
**Compose with:** [#10196](https://github.com/Luminous-Dynamics/symthaea/issues/10196), [#12841](https://github.com/Luminous-Dynamics/symthaea/issues/12841), [#13348](https://github.com/Luminous-Dynamics/symthaea/pull/13348), [#13349](https://github.com/Luminous-Dynamics/symthaea/issues/13349), [#13358](https://github.com/Luminous-Dynamics/symthaea/pull/13358)

## Research question

Does a renderer control request actually change the selected musical output in the intended way, and does the effect survive independent observation, sham controls, reversal and collateral-drift checks?

An API that accepts `bpm=120` has demonstrated request acceptance. It has not demonstrated that the rendered waveform follows 120 BPM, that the change persists, or that the requested musical result is useful.

## Six non-interchangeable evidence layers

| Layer | Evidence | Maximum claim |
|---|---|---|
| Declared | Provider documentation/capability manifest | Provider documents the control |
| Accepted | Request acknowledgement | Provider accepted the request |
| Effective | Adapter/provider configuration receipt | Provider reports a setting became effective |
| Symbolic | Independently checked score/MIDI change | Symbolic output changed |
| Observed output | Independent capture plus qualified observer measurement | Measured output changed |
| Human-perceived | Blinded musician/listener assessment | A person perceived the requested musical change |

No layer may silently promote itself. Accepted and effective control receipts are not acoustic evidence. MIDI output is not audible-onset evidence. A machine-observed tempo/brightness change is not proof that a musician preferred it.

## Four-arm counterfactual probe

Each trial binds one immutable renderer, adapter, non-target configuration and capability-manifest identity across four conditions. Baseline and sham are randomized between themselves before the intervention; active is then applied; reversal must follow active. This constrains the randomized order so the reversal is causally interpretable:

1. **Baseline:** no control request; observe current output.
2. **Sham:** submit a no-op request at the baseline setting; the API may accept it, but the measured output should not materially change.
3. **Active:** submit the target setting; independently measure the intended output change.
4. **Reversal:** restore the baseline setting; independently measure whether the output returns toward baseline.

Each arm has a unique capture ID and a separate complete observation window. Capture-observer and event-observer identities/builds/configurations, event-detector qualification reference, producer identity and clock domain are held fixed. The event manifest declares the exact captured-audio payload digest it analyzes. Request status, effective value and measured metric are recorded separately.

Randomized order slots and a randomization-receipt digest are required. The v1 probe randomizes baseline-versus-sham ordering, then requires active-before-reversal. The evaluator recomputes the receipt over the recorded order and checks the sequence. It does not authenticate the randomization process or prove that a trusted RNG generated it.

## Implemented evaluator contract

`assess_renderer_control_effect_v1` checks:

- exactly one baseline, sham, active and reversal condition;
- a permutation of randomized order slots 0–3 and a trial randomization receipt;
- exact renderer, adapter, non-target configuration and capability-manifest identity across arms (only the declared target control may differ);
- separate request status, requested value, adapter-reported effective setting and measured effect;
- independent capture observer and event detector, each distinct from the producer and separately identified; the event manifest must bind the exact capture-payload digest;
- capture payload, closure, observer manifest, capture-observer and event-observer build/configuration/qualification/receipt digests, and clock-alignment digest shape;
- structurally complete observation windows of the preregistered duration, with distinct capture IDs;
- stable capture-observer and event-observer identity/build/configuration/qualification, producer identity and clock domain across all four conditions;
- conservative active-effect lower bound after measurement uncertainty;
- sham-effect upper bound, reversal-residual upper bound and collateral-drift upper bound.

A trial returns one of three outcomes:

- `structurally_qualified`: complete observation and all preregistered effect rules pass;
- `not_qualified`: complete observation exists, but request/effective-setting, target effect, sham, reversal or collateral rules fail;
- `unobserved`: at least one required window is incomplete, unknown or too short.

Unobserved data are not a zero effect and are not an intervention failure.

These are single-probe structural checks, not population-level causal inference. Production qualification should repeat independently randomized blocks across seeds, musical contexts, devices and sessions, then estimate effect distributions and confidence intervals with listener/session clustering. A single passing synthetic four-arm fixture does not establish that a provider reliably implements the control.

## Statistical interpretation

For target metric (y), baseline (B), sham (S), active (A), reversal (R), and absolute uncertainty (u):

[
L_A = |A-B| - u_A-u_B
]

[
U_S = |S-B| + u_S+u_B
]

[
U_R = |R-B| + u_R+u_B
]

A trial passes only if the active effect has the preregistered direction and (L_A) meets the minimum effect, while (U_S) and (U_R) remain below their limits. For every declared collateral metric, the evaluator computes uncertainty-expanded active-vs-baseline drift and checks it against the collateral budget.

Thresholds are dimension-specific. A useful initial BPM threshold is not a universal brightness, harmonic or groove threshold. Establish instrument-, tempo-, genre- and backend-specific measurement floors before preregistering them.

## Isolated source-linked lint harness

The `tools/mel-renderer-effect-check` package is an explicit, non-default root workspace member. It imports the two changed Rust modules by source path and uses the authoritative root `Cargo.lock`; there is no second independently maintained lockfile.

This keeps the warnings-denied Clippy gate meaningful for the new evaluator without letting unrelated legacy Clippy debt elsewhere in the large `symthaea-muse` package block or silently suppress the new module's lints. The harness runs the same 20 renderer-effect tests plus the evidence-digest unit tests, then Clippy runs across all harness targets with `-D warnings`. The actual `symthaea-muse` package's 20-test run remains a separate required workflow step.

## Negative controls in v1

The 20 synthetic tests cover accepted request without measured effect; null sham and successful reversal; sham effect; sticky reversal; collateral musical drift; incomplete/too-short capture; accepted request with wrong effective setting; renderer identity drift; producer-as-observer; duplicated randomization slot; mismatched randomization receipt; invalid active/reversal order; negative thresholds; non-finite effective setting; observer-build drift; capture/manifest payload mismatch; inconsistent collateral inventory; condition-array reordering; acoustic dimensions rejected from MIDI-only observations; and incomplete capture versus observed silence.

## Industry comparison

- **Lyria RealTime:** compare accepted BPM/key/density/brightness controls with independently measured acoustic effect. The official API guide documents a streaming control surface; community reports of individual experimental-endpoint controls not changing output are signals to test, not proof of a provider-wide defect. Every adapter should have active, sham and reversal probes.
- **Daydream:** its published 81 ms parameter-change-to-changed-sound is a useful responsive-control baseline. Report this separately from musician-input-to-acoustic-onset, and add effect magnitude, uncertainty, reversibility and collateral drift.
- **Taan TimingBench:** timed-event generation is a useful comparator, but ±500 ms is not equivalent to beat-relative phase or live control-effect qualification.
- **JAM_BOT:** hardware-aware MIDI latency compensation motivates measuring symbolic output and captured audio separately.
- **MIREX / Artificial Analysis:** blind finished-track preference remains its own scorecard; it cannot establish whether a live control did what it claimed.

## Prospective targets (not results)

| Dimension | Initial target | Stretch |
|---|---:|---:|
| Controls claiming observed effect with complete observation evidence | 100% | 100% |
| Correct active-effect direction on admitted trials | 100% | 100% |
| Sham false-positive upper 95% bound | ≤0.02 | ≤0.005 |
| Reversal residual | Within preregistered measurement floor | Tighter calibrated floor |
| Collateral drift | Within per-control declared budget | Tighter budget without hiding trade-offs |
| Qualified observation windows | ≥99% per backend/device | ≥99.9% per backend/device |
| Renderer/adapter/config/observer identity stable across arms | 100% | 100% |
| Missing capture interpreted as zero effect | 0 | 0 |

These are prospective design gates, not Melothaea results. If a provider cannot support independent observation, its control must remain best-effort/unqualified or be abstained from—not advertised as a hard guarantee.

## Claim ceiling

The current module validates synthetic fixtures and the structure of supplied observation references. It does not authenticate capture/observer signatures, recompute event-manifest payload digests, independently qualify detectors, verify acoustic calibration or conduct listener studies. Structural qualification is not authenticated physical truth and is not evidence of human-perceived quality.

Compose actual capture through #13349, observation eligibility through #13348, detector validation through #10265, backend capability through #10196, symbolic factorial effects through #12841, and immutable execution identity through #13358/#13254. Do not create duplicate ownership or a second musical time mapper.
