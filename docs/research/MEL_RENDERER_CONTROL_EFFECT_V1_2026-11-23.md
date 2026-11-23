# Renderer control-effect qualification v1

**Date:** 23 November 2026  
**Parent contract:** [#10202 — capability-qualified realization](https://github.com/Luminous-Dynamics/symthaea/pull/10202)  
**Related observation contract:** [#13348 — output-observation closure](https://github.com/Luminous-Dynamics/symthaea/pull/13348)  
**Live capture integration:** [#13349](https://github.com/Luminous-Dynamics/symthaea/issues/13349)  
**Symbolic factorial response surfaces:** [#12841](https://github.com/Luminous-Dynamics/symthaea/issues/12841)

## The gap

A manifest can declare that a backend supports tempo control. A request can be accepted. A receipt can say that tempo was effective. None of those facts establishes that the rendered sound changed in the intended direction.

This increment adds a renderer-level counterfactual protocol:

1. baseline;
2. sham/no-op;
3. active control;
4. reversal to baseline.

Every arm carries the same backend/version/adapter identity, capability-manifest digest, source artifact digest and session seed. Each scored arm links an output artifact, realization receipt, capture payload, capture-closure receipt and event-observer manifest. Capture and event-observer identities must not equal the output producer.

## What is scored

The probe freezes:
- control being tested;
- target observation metric and direction;
- baseline and treatment settings;
- minimum active effect;
- sham equivalence margin;
- reversal-to-baseline margin;
- maximum permitted collateral drift.

The report distinguishes:
- **qualified observed effect:** active effect exceeds the threshold, sham remains equivalent to baseline, reversal returns to baseline, and collateral metrics remain within bounds;
- **control not applied:** the adapter did not report the preregistered effective setting;
- **no observed effect:** the adapter accepted the setting but the measured target did not change enough or changed in the wrong direction;
- **sham confounded:** the no-op condition changed the target beyond the equivalence margin;
- **reversal failed:** the effect persisted after returning to baseline;
- **collateral drift:** the target moved, but a protected/secondary metric moved beyond its preregistered limit;
- **unobserved:** one or more capture windows were incomplete or their independent observation evidence was unavailable.

The module uses integer milli-units to keep comparisons deterministic. It does not invent a universal effect scale: each metric declares its own unit and thresholds.

## Evidence boundary

The evaluator validates identity consistency, digest shape and references to the capture/observer receipts. It does **not** authenticate those receipts or decode the audio itself. The current observation-eligibility field is a declared verifier outcome; the referenced closure/manifest digest still needs an independently trusted verifier and signer policy.

The module also does not establish that a measured acoustic change is musically useful or perceptually salient. Listener/musician evidence is a separate evaluation layer. A control that makes the sound louder or denser may satisfy an acoustic direction while harming the requested musical result.

## Negative controls

The Rust tests cover:
- accepted control with no measured effect;
- sham condition that changes the target (confounded);
- incomplete observation (unobserved, not a negative);
- backend/version change across arms;
- observer identity equal to producer;
- missing arm;
- invalid closure receipt digest;
- failed reversal;
- collateral drift despite a valid target effect.

## How it composes with existing work

- [#12841](https://github.com/Luminous-Dynamics/symthaea/issues/12841) remains the symbolic factorial response-surface owner. Do not clone its 81-condition symbolic design here.
- [#10196](https://github.com/Luminous-Dynamics/symthaea/issues/10196) and [#10202](https://github.com/Luminous-Dynamics/symthaea/pull/10202) own capability manifests, request negotiation and realization receipts.
- [#13348](https://github.com/Luminous-Dynamics/symthaea/pull/13348) owns observation-window structural eligibility and capture-bound event manifests.
- [#13349](https://github.com/Luminous-Dynamics/symthaea/issues/13349) owns connecting independent capture to the live runner/CPAL/MIDI path.
- [#13358](https://github.com/Luminous-Dynamics/symthaea/pull/13358) owns immutable model/adapter/runtime identity across counterfactual RMI arms.
- [#13254](https://github.com/Luminous-Dynamics/symthaea/issues/13254) owns authenticated signer identity and trusted execution provenance.

The intended chain is:

`declared capability → request accepted → effective setting reported → independent capture closed → observed effect measured → human/perceptual effect assessed`

No earlier stage is allowed to imply a later one.

## Prospective targets

These are targets, not measured production performance:

| Gate | Target |
|---|---:|
| Sham/no-op false positives | 0 on preregistered controls |
| Accepted-but-ineffective control correctly classified | 100% in negative controls |
| Incomplete observations incorrectly scored as effects | 0 |
| Backend/version/source/seed mismatch admitted across arms | 0 |
| Reversal and collateral checks applied to every qualified probe | 100% |
| Scored production probes with structurally eligible independent capture | 100% |

A production adapter must run these probes against exact backend/model/adapter versions and report results per control, backend and output modality. No provider is assumed to pass because its API accepts a setting.
