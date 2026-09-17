# Symthaea RSI evidence boundary

This subsystem separates **replay-grounded evidence** from **generated imagination**.

## Invariant

A counterfactual, prediction, interpolation, extrapolation, or adversarially generated world may propose a hypothesis or action prior, but it MUST NOT increase epistemic confidence unless it has been independently validated against recorded evidence.

Exact replay is different: it may only traverse transitions already present in an immutable `ExperienceTree`. Missing transitions fail closed.

## Promotion path

1. Dream/predictive system proposes a candidate.
2. Candidate remains non-empirical.
3. Candidate is checked against exact historical replay and/or a new controlled observation.
4. Validation produces a recorded evidence digest.
5. Only then may the candidate affect confidence/calibration claims.

This boundary is intentionally narrower than Symthaea's broader recursive-improvement ambitions. It exists so later policy evolution and architectural self-modification have a trusted empirical substrate.