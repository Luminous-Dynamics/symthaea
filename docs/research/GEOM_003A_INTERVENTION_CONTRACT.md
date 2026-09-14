# GEOM-003A — Intact / Lesion / Sham / Rescue Contract

## Status

Implementation sub-tranche for #2974, stacked on GEOM-002C exact head `403ffa09a83f484bede4e09f5676c74e06abbcb8`.

This protocol is theory-neutral. It does not decide which Symthaea subsystem counts as conscious, nor whether any observed effect has a gravitational analogue.

## Experimental quartet

Every experiment has four named roles fixed before measurement:

1. **Intact** — unmodified reference condition.
2. **Lesion** — one declared mechanism disabled or disrupted.
3. **Matched sham** — an intervention with comparable procedural cost that leaves the target mechanism intact.
4. **Rescue** — exact or declared restoration of the lesioned mechanism.

The analysis API fixes those roles in its function signature. Conditions are not selected or relabeled after seeing outcomes.

## Required matching

All four conditions must use the same:

- measurement protocol id;
- random seed;
- observation-step count;
- allocated work budget.

The GEOM-002C bridge additionally requires compatible probability-state dimensionality, causal state-count profiles, and matched presence/absence of sensitivity data.

## Required contrasts

The report exposes all of the following:

- lesion versus intact;
- sham versus intact;
- lesion versus sham;
- rescue versus lesion;
- rescue versus intact.

Each contrast preserves geometry, causal, and sensitivity families separately.

## Rescue recovery

For each scalar observable independently:

`recovery = 1 - |rescue - intact| / |lesion - intact|`

when the lesion produced a non-zero effect.

Interpretation:

- `1.0` = exact return to intact value;
- `0.0` = no closer to intact than the lesion;
- `< 0.0` = rescue worsened the deviation;
- `None` = lesion produced no measurable effect, so recovery is undefined.

Recovery values are not clamped and are never averaged across observables.

## No global verdict

GEOM-003A MUST NOT output:

- a consciousness pass/fail;
- a combined geometry/causality score;
- a single intervention-success score;
- evidence that macro causal advantage is downward causation;
- evidence that gravity and consciousness share a physical mechanism.

## Purpose of the sham

The sham condition estimates effects caused by the intervention procedure itself rather than the target mechanism. `lesion_vs_sham` is therefore reported explicitly rather than inferred from two separately inspected plots.

## Downstream architecture adapters

GEOM-003B may implement adapters for specific Symthaea mechanisms only after this protocol is qualified. Candidate targets include workspace/broadcast, recurrent temporal coupling, memory access, active-inference action selection, and HDC relational binding.

Each adapter must declare:

- the exact capability disabled;
- what remains unchanged;
- the matched sham;
- the rescue operation;
- allocation/activity matching metadata.

GEOM-003C will later add repeated-seed inference, confidence intervals, multiple-comparison handling, and holdout rules. Those statistical choices are intentionally not added after observing architecture-specific outcomes.
