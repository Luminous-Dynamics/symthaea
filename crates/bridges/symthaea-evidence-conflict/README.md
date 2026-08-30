# Symthaea Evidence Conflict

Planetary Perception treats disagreement as evidence, not noise to be averaged away.

This crate wraps the provider-neutral `EvidenceConflict` with an explicit contradiction-analysis lifecycle:

1. preserve the original competing evidence;
2. record multiple candidate explanations;
3. state what new evidence would discriminate among them;
4. distinguish `Open` from `ExplainedButUnresolved`;
5. require explicit verification-stage evidence before marking the conflict `Resolved`.

## Candidate cause classes

The first vocabulary includes spatial mismatch, temporal mismatch, calibration mismatch, sensor fault, processing artefact, model failure, sampling bias, real-world heterogeneity, definition mismatch, and unknown cause.

These are explanations to investigate, not automatic diagnoses.

## No destructive consensus

Resolution does not delete the losing observation, model output, or report. The original contradiction stays attached to the assessment so later reviewers can reconstruct why the system disagreed and how the disagreement was resolved.

## Support is not probability

`ConflictExplanation::support` is a bounded relative assessment value. It is not claimed to be a calibrated posterior probability unless a future calibrated inference layer explicitly establishes that property.

## Connection to observation planning

A future adapter can translate `DiscriminatingEvidenceNeed` into candidates for `symthaea-observation-planning`. That adapter should remain separate so conflict diagnosis does not silently task sensors or authorize field activity.
