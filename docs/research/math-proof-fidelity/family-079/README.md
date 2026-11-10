# Family 079 local-smoothing correspondence crosswalk

This is a **preliminary source-level audit artifact** for OpenAI's 25 October 2026 Euclidean local-smoothing release.

The complete machine-readable crosswalk is in `FAMILY_079_LOCAL_SMOOTHING_CROSSWALK_V0_1.json`.

## Current finding

The paper's main local-smoothing theorem and the Lean `EuclideanLocalSmoothing` proposition were compared at the same pinned `openai/math` commit. The inspected statements match on dimension, p-range, epsilon loss, Bessel-potential order, half-wave sign, time interval, Schwartz data, spacetime measure and norm inequality. No obvious statement mismatch was found in this first comparison.

The comparator target file contains six theorem statements ending in intentional `sorry` placeholders. The corresponding proof module is separate and supplies the six proof terms. The placeholder file must not be misreported as six failed Lean proofs.

## What this does not establish

- It is not a clean Lean rebuild or kernel replay.
- It is not a full dependency-closure axiom audit.
- It is not an independent review of the paper's analysis.
- It is not a complete paper-to-Lean crosswalk for every consequence.
- It does not establish that every advertised corollary is independently verified.

Additional formal results were located in `Main.lean`, `FurtherResults.lean`, and the Kakeya logarithmic-tube module. The compact-paraboloid and Schrödinger consequences still need explicit declaration-level mapping.

## Reproduction

The audit pins repository commit `4eaee87548250b2682affda7d05d947c05554bae` and records blob identities for every inspected source file. Verify each blob against that commit before relying on the mapping. Update this artifact by versioning a new crosswalk rather than silently rewriting the historical audit.

This file is a research crosswalk, not an evidence receipt, theorem prover, or scientific truth score.