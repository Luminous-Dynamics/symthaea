# Adaptive Assistance Safety Case

Adaptive assistance is **shadow-only by default**. Learned parameters do not
write directly to a body-coupled actuator path.

## Invariants

1. Candidate gains are bounded to `[0.5, 1.0]`; adaptation cannot amplify the
   independently validated nominal controller.
2. Candidate performance is evaluated as paired baseline/candidate evidence.
3. Any safety-envelope violation rejects the complete candidate.
4. Low-confidence, stale, non-finite, or incomplete evidence rejects the trial.
5. A commit requires a minimum evidence window and a configured relative
   improvement margin.
6. Every successful commit retains the preceding profile as a rollback point.
7. Deployment to a wearer requires separate simulation, replay, HIL, ethics,
   clinical/human-factors, and release approval. The software commit decision is
   necessary evidence, not sufficient authorization.

## Suggested objective

The shadow cost should combine human effort, tracking error, discomfort,
metabolic proxy, actuator effort, and stability margins. Weighting and units
must be versioned as part of the experiment protocol; changing the objective
invalidates comparisons with earlier candidates.

## Prohibited shortcuts

- Updating controller weights from live wearer data without a shadow trial.
- Treating lower actuator effort as improvement when wearer effort increases.
- Averaging away a safety violation across otherwise successful samples.
- Expanding hard torque, velocity, power, passivity, or authority envelopes.
- Committing from synthetic evidence while presenting it as human evidence.
