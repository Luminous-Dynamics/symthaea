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

## Post-CI-bridge qualification lineage

The continuation line after merge commit `514691f446bc2717247e3c1ad3c930aaa63b9b6c` inherits current-main draft-safe CI without rewriting any earlier RSI subject. Because that merge changes the execution/qualification environment, pre-bridge qualification evidence is not promoted across it automatically.

A descendant exact head must therefore re-run the focused RSI qualification before making compile, lint, unit-test, or protected-measurement-boundary claims for the bridged lineage. This README-only marker intentionally changes no Rust source; it exists to trigger and document that new qualification root.

This boundary is intentionally narrower than Symthaea's broader recursive-improvement ambitions. It exists so later policy evolution and architectural self-modification have a trusted empirical substrate.