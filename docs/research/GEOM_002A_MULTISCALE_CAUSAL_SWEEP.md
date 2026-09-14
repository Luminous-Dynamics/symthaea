# GEOM-002A — Multiscale Causal Sweep Contract

Status: implementation candidate; qualification blocked on GEOM-001 predecessor CI and this tranche's own hosted checks.

## Purpose

GEOM-002A asks a narrow question:

> Does the same causal system admit a declared coarser description whose effective information exceeds that of the finest description under explicit intervention semantics?

A positive result is a mathematical property of the chosen system and coarse-graining. It is **not** by itself evidence of consciousness, downward causation, gravity, or a physical bridge between cognition and spacetime.

## Input authority

The finest scale is supplied as a square transition probability matrix (TPM). It must be finite, non-negative, and row stochastic.

Every later scale is supplied only as a parent-state → macrostate assignment. The coarse TPM is **derived**, not independently supplied.

For macrostate `A`:

1. intervene uniformly over parent states assigned to `A`;
2. propagate each parent intervention through the parent TPM;
3. aggregate destination probability by destination macrostate.

This convention is intentionally explicit. Other intervention semantics belong in separate preregistered experiments.

## Measurements

For each scale GEOM-002A records:

- state count;
- effective information in bits;
- determinism;
- degeneracy;
- EI gain versus the immediately preceding scale;
- EI advantage versus the finest scale.

The sweep also records the scale with maximal EI and evaluates one preregistered threshold:

`peak_advantage_bits >= min_advantage_bits`

with the additional requirement that the peak is not the finest scale.

## Mandatory controls

### Uniform null

A fully uniform TPM has zero effective information. Any valid coarse-graining of that null must remain at zero EI.

### Deterministic identity reference

An `n`-state identity TPM has `log2(n)` bits of effective information. Coarse-graining the four-state identity into two equally sized macrostates yields 2 bits at the fine scale and 1 bit at the macro scale. The implementation must not mislabel that loss as causal emergence.

### Positive causal-advantage construction

A four-state system is included where three fine states are causally degenerate and form one macrostate, while the fourth forms the second macrostate. The fine system has less than one bit of EI; the derived two-state macro system is deterministic with one bit. A threshold of 0.1 bits must pass.

### Invalid coarse-graining controls

Assignments must:

- cover every parent state exactly once;
- use a contiguous macrostate index set beginning at zero;
- leave no macrostate empty;
- strictly reduce the number of states at each step.

Malformed TPMs or coarse-grainings fail closed.

## Claim boundary

Allowed claim:

> Under the preregistered uniform-within-macro intervention semantics, this declared coarse-graining produced X bits more effective information than the finest scale.

Not allowed from GEOM-002A alone:

- consciousness emerged;
- the macrostate has metaphysically independent causal powers;
- causal emergence proves downward causation;
- the result validates IIT;
- the result validates active inference;
- cognitive geometry is equivalent to spacetime geometry;
- gravity and consciousness share a physical mechanism.

## Follow-on tranches

- **GEOM-002B**: repeat the sweep across preregistered discretizations/coarse-grainings and report sensitivity envelopes and peak-scale stability.
- **GEOM-002C**: jointly report the qualified GEOM-001 Fisher-Rao trajectory observables and GEOM-002 causal scale profile without collapsing them into one score.
- **GEOM-003**: lesion/rescue experiments using matched architectures and exact interventions.

The gravity analogue remains downstream of those stages.
