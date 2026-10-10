# PROD-EQP-001A — Productive-machine evidence contract

Issue: #5987  
Parent: PROD-EQP-000 #5964  
Frozen corpus: `docs/release/evidence/prod-eqp-001a-productive-machine-reference-v1.json`  
Canonical SHA-256: `2cfcd09a899599e0bec8c7da7424e07c89324608d94365330010a6ac2e996ea7`

## Purpose

Freeze the first reference boundary for productive-equipment evidence before any production adapter or machine-facing implementation exists.

The contract preserves:

`machine/frame design != as-built machine != commissioned machine != observed bounded capability != process-family compatibility != renewable tooling/metrology != G1/G2+ capability-envelope preservation != physical execution authority`.

This tranche is documentation/data only.

## Ownership

PROD-EQP specializes existing owners rather than copying them:

- MFG-PROC #5686 owns generic manufacturing process/capability semantics.
- CIV-BOOT #5774 owns productive-capability closure.
- CIV-BOOT-003 #5782 owns multi-generation capability-envelope preservation.
- ENG-TOL #4883 owns tolerance/fit/metrology semantics.
- ROB-REALIZE #4859 owns physical article/as-built generations.
- FIELD/SE-OBS owns physical observations.
- MFG-LIFE #5705 owns repair/remanufacture/lifecycle projections.
- Mycelix owns actual machine inventory, work, tooling stock, maintenance, custody and production events.

## Evidence planes

The frozen corpus keeps independently representable:

1. design/configuration;
2. as-built generation;
3. installed subsystems;
4. geometry/alignment;
5. bounded machine capability;
6. tooling/workholding;
7. metrology/calibration;
8. process-family compatibility;
9. commissioning;
10. produced-part observation;
11. maintenance/spares;
12. tooling/reference renewal;
13. import dependencies;
14. generation/profile;
15. prior-generation capability-envelope relation.

These planes must not collapse into `machine_available`, `qualified`, `self_reproducing`, or a scalar readiness score.

## Reference dispositions

The synthetic oracle vocabulary is intentionally descriptive:

- `DesignOnly`
- `AsBuiltUncommissioned`
- `PartialProductiveClosure`
- `MetrologyRenewalUnresolved`
- `ToolingRenewalUnresolved`
- `CapabilityUnresolved`
- `ProcessCompatibilityUnresolved`
- `GenerationEnvelopeDegraded`
- `DegradedButSufficientUnderProfile`
- `CapabilityDimensionRestoredPartialClosure`
- `NewConfigurationSubjectRequired`
- `SemanticIdentityUnchanged`
- `InvalidDuplicateReference`
- `HistoryPreserved`
- `NoAsBuiltOrCommissionedState`
- `NoMachineExecutionAuthority`

These are reference-corpus outcomes, not a new generic manufacturing state engine.

## Precision-ratchet theorem

The key multi-generation requirement is:

`same nominal machine class != same productive capability`.

A G1 machine may physically exist yet lose a dimension needed to reproduce the process/tooling/metrology route that produced it. Conversely, a degraded successor may remain useful for a narrower declared repair profile.

No axis silently compensates for another.

## Anti-laundering rules

The corpus freezes these controls:

- friendly labels do not alter semantic identity;
- installed subsystem/as-built changes do;
- duplicate required references reject;
- one good part does not establish a general process capability;
- later success preserves prior negative evidence;
- metrology improvements may restore one dimension only;
- design/optimizer output cannot mint an as-built or commissioned article;
- representation/qualification cannot mint machine execution authority.

## Frozen adversarial corpus

The corpus contains exactly 16 synthetic cases covering:

- design-only and as-built/uncommissioned states;
- local frame with imported critical subsystems;
- metrology-reference and tooling-renewal loss;
- substitute-bearing functional-evidence gaps;
- one-part-success overclaim prevention;
- G1→G2 precision-ratchet degradation;
- bounded usefulness of a degraded successor;
- partial restoration from improved metrology;
- configuration identity changes;
- non-semantic display changes;
- duplicate-ref rejection;
- append-only negative history;
- optimizer/as-built separation;
- zero machine-execution authority.

No case contains machine dimensions, speeds, forces, powers, process parameters, commands, safety settings, or hazardous operating instructions.

## Qualification path

Next child:

1. independent stdlib-only validator over the exact corpus bytes;
2. dedicated path-scoped hosted workflow;
3. only after source qualification, audit whether production code needs a new specialization or can remain a view/profile over existing owners.

A PASS may establish source-contract conformance only.

## Claim ceiling

This tranche establishes no real machine, process capability, dimensional tolerance, tooling life, metrology traceability, commissioning state, safe operation, production capacity, economics, productive closure, or physical execution authority.
