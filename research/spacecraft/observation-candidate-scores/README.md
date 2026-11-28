# Spacecraft Observation Candidate Scoring

Issue: [#14165 — bounded EVSI planner](https://github.com/Luminous-Dynamics/symthaea/issues/14165)

## Delivered primitive: observational EFE candidate scoring

`ExpectedFreeEnergyComputer::evaluate_candidates_observational` scores a caller-supplied, unique set of abstract action IDs without mutating:
- the source scorer or its action history;
- the belief state;
- the generative model.

Each candidate is scored from the same pre-call action-history snapshot. Output ordering matches the caller's action list. Empty sets, duplicate/out-of-range IDs, malformed model dimensions, invalid belief values and non-finite outputs are rejected.

## What this is not

This is **not** expected value of sample information (EVSI). The existing FEP epistemic component uses predicted hidden-state entropy difference. It does not model a real candidate camera observation, the posterior after possible sensor outcomes, calibration error, downstream mission decision loss, illumination, sensor cost, feasibility, or safety constraints.

It is a deterministic/observational primitive and baseline for future spacecraft candidate ranking. The consumer must provide a validated action-to-observation mapping and a proper observation model before interpreting these scores for real sensing decisions.

## Tests

- Repeated candidate scoring returns identical scores.
- Candidate order is preserved.
- The source scorer/action history, model, and belief remain unchanged.
- Empty, duplicate and out-of-range candidate IDs reject.
- Non-finite belief and model dimension mismatch reject.

The focused workflow runs package rustfmt, candidate tests, the complete FEP crate library tests and Clippy with warnings denied.

No spacecraft observation plan, physical feasibility or actuation authority is implemented here.
