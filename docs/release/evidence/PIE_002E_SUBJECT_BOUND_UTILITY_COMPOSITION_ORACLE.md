# PIE-002E Subject-Bound Utility Composition Oracle

Status: independent reference semantics; synthetic fixtures only.

## Purpose

`scripts/pie-subject-bound-utility-composition-oracle.py` freezes the narrow composition boundary between a real current `ProcessDefinition`, PIE-002B-style deterministic utility projection, and PIE-002D-style explicit accounting-case binding.

The theorem exists to eliminate one specific production hazard: a detached projection may remain internally valid after the process record that supposedly produced it has changed.

## Core theorem

```text
current process record
+ full process structural validation
+ utility projection recomputed from that exact record
+ explicit external supply/recovery context
-> bound but unevaluated accounting case
```

The preferred subject-bound entry point accepts exactly two inputs:

```text
process
context
```

It does not accept a caller-supplied cached projection.

## Current-subject rule

The composition function validates the exact process argument, derives its utility projection inside the same call, and immediately passes that derived projection to the binding contract.

Therefore a stale projection captured from an earlier version of the same process cannot influence the direct composition result.

This is a semantic current-subject theorem only. It does not introduce a cryptographic process digest, database transaction, distributed currentness oracle, or evidence freshness theorem.

## Preserved lower-layer semantics

PIE-002E does not repair weaker inputs:

- missing electrical energy remains incomplete;
- missing peak power remains incomplete;
- missing process time remains incomplete;
- duplicate peak power remains ambiguous;
- duplicate process time remains ambiguous;
- thermal-energy quantity may remain visible with temperature semantics unresolved;
- cooling-energy quantity may remain visible with rejection semantics unresolved;
- the external recovery/storage/supply context remains fully explicit;
- binding does not become feasibility evaluation.

## Synthetic fixtures

The checked-in self-test covers:

- valid current process -> projection -> binding;
- exact preservation of process identity and explicit context;
- verification that the preferred function signature has no detached projection parameter;
- structural process invalidity rejected before binding;
- captured stale projection followed by utility mutation, proving the direct path recomputes the changed demand;
- incomplete utility basis rejection;
- duplicate peak-power ambiguity rejection;
- duplicate process-time ambiguity rejection;
- thermal/cooling unresolved semantics surviving composition without promotion;
- process identity mutation reflected by the recomputed direct path.

## Boundaries

This oracle does not prove:

- electrical feasibility;
- storage dispatch;
- recovery realizability;
- thermodynamic feasibility;
- thermal-network feasibility;
- equipment capability;
- process performance;
- economics;
- evidence freshness/currentness;
- external-source provenance;
- hardware or execution authority.

Subject-bound composition != feasibility.

Tracks #2782, #2764, #2759, #2724, #1610, #1647, and master #1604.
