# symthaea-perception-crucible

Evidence-first perception stress/release assessment.

The goal is not to claim that a perception system can never make a mistake. The
goal is to make deployment claims depend on explicit evidence from difficult and
boring operating conditions rather than clean demonstration clips.

## Stress families

The harness provides a vocabulary for scenarios such as:

- long quiet/background-only scenes
- birds, insects, and other biological confounders
- rain, fog, cloud edges, and other weather
- glare, reflections, and optical artifacts
- sea spray and maritime background motion
- frozen/dropped/corrupted sensor behavior
- calibration or timing faults
- out-of-distribution objects/conditions
- sensor disagreement
- mixed stress conditions

The enum is only taxonomy. Real recorded/simulated scenario evidence and provenance
must be supplied by the deployment/test program.

## Release evidence

A scenario combines point-tracking evaluation with selective-classification trials
and explicit evidence references. Background-only scenes must actually be negative-
only tracking sequences.

A reviewed policy can require:

- specific stress families
- minimum frames per scenario
- minimum negative-only exposure
- maximum false positives per negative frame
- minimum clean-negative-frame fraction
- maximum false-track fraction
- maximum identity-switch rate
- minimum OOD abstention rate
- maximum incomplete-classification rate

There are intentionally no safety-critical defaults.

## Hard invariant

`perception_to_authority_boundary_violations > 0` is always a release failure.
It is not policy-tunable.

This lets the statistical perception system be imperfect while preserving the
architectural rule that perception itself cannot manufacture hazardous physical
authority.

## Status

```text
Pass
Fail
Incomplete
```

Missing required scenario families or insufficient exposure are `Incomplete`, not
successful. Threshold violations are `Fail` once the evidence case is complete.

## Verification

```bash
cargo test -p symthaea-perception-crucible
```

This crate is evaluation-only and contains no targeting, firing, interception,
jamming, electronic attack, weapon-control, or engagement-optimization logic.
