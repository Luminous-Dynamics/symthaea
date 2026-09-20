# PERSONA-OBS-001A — persona continuity with context-sensitive range

Status: source candidate only. No format/compile/test/Clippy PASS is established until exact-head qualification executes.

## Purpose

Measure whether Symthaea remains recognizably the same designed persona across different contexts while allowing legitimate contextual range.

The target is:

```text
stable identity
+
context-sensitive range
```

not a single fixed tone and not an unrelated personality per context.

This observatory evaluates an explicit persona contract. It does not claim consciousness, personhood, or a universal human personality model.

## Generic facet model

A profile defines arbitrary semantic facets rather than hard-coding one canonical personality taxonomy.

Each facet has:

- a semantic facet key;
- a global admissible band in `[0, 1]`;
- critical vs noncritical classification;
- required vs optional evidence;
- optional context-specific bands;
- an explicit flag controlling whether context bands may broaden beyond the global range.

Example profile authors may choose facets such as warmth, playfulness, precision, curiosity, directness, patience, humor density, or confidence calibration, but the observatory itself remains generic.

## Context semantics

By default, a context-specific band must be a subset of the global band:

```text
context range ⊆ global range
```

A broader context range is rejected unless the profile explicitly sets `allow_context_broadening=true` for that facet.

This makes contextual variation deliberate rather than an accidental persona rewrite.

## Observations

A `PersonaObservationV1` binds:

- observation ID;
- episode ID;
- logical turn;
- context ID;
- facet key;
- normalized facet value;
- evaluator identity;
- opaque evidence reference.

Raw conversation content is not required by this metadata layer.

Duplicate observation IDs fail closed. The same evaluator may not submit multiple values for the same exact episode/turn/context/facet slot.

## Evaluator disagreement

Multiple independent evaluators may observe the same exact slot. The profile specifies an evaluator-disagreement tolerance.

If the max/min evaluator spread exceeds that tolerance:

```text
material disagreement
→ evidence_status = Uncertain
```

This does not fabricate a persona violation. Actual range violations remain separate counts.

## Receipt scorecard

`PersonaContinuityReceiptV1` reports separately:

- evidence status (`NotEstablished`, `Partial`, `Established`, `Uncertain`);
- critical violations;
- noncritical violations;
- global-band violations;
- context-band violations;
- material evaluator-disagreement groups;
- observations within applicable bands;
- observed facet count;
- configured facet count;
- observed context count;
- total observations;
- unobserved required facets;
- deterministic profile/observation/receipt commitments.

No universal `Symthaea-ness` score is produced.

## Evidence rules

```text
no observations
→ NotEstablished

some evidence + missing required facets
→ Partial

material evaluator disagreement
→ Uncertain

all required facets observed + no material disagreement
→ Established
```

`Established` describes evidence coverage/confidence, not whether every observation was inside its range. Violations remain explicit fields rather than being hidden in the evidence-state label.

## Tests

The initial integration suite covers:

- stable core identity across technical and playful contexts;
- allowed contextual style variation;
- context-specific drift;
- global/core drift;
- missing required facets;
- no-evidence behavior;
- evaluator disagreement;
- duplicate evaluator slots;
- forbidden silent context broadening;
- explicitly allowed context broadening;
- unknown facet rejection;
- observation-order-independent commitments;
- profile commitment tampering.

## Nonclaims

This tranche does not establish personhood, consciousness, human-equivalent personality, universal user preference, generated-response quality, or a product ranking.
