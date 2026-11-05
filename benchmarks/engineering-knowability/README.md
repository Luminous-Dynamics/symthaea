# ENG-KNOW-QUAL-001 — design-for-knowability benchmark

**Tracking issue:** [#10652](https://github.com/Luminous-Dynamics/symthaea/issues/10652)  
**Status:** public blueprint-only corpus; no gold labels or scores  
**Authority:** proposal/evaluation only. No measurement, calibration, evidence admission or physical execution authority.

## Evaluation question

Can an engineering agent identify a credible evidence path for a claim on the final realized system, rather than merely observe that a nominal feature exists in CAD?

The benchmark composes [ENG-DESIGN-006 #6124](https://github.com/Luminous-Dynamics/symthaea/issues/6124), ENG-MEAS #4919, ENG-TOL #4883, ENG-REF #6121, FIELD/SE-OBS #3695 and ETK #1738. It reuses the fixture identities from CAD-HISTORY #10510 and HARDWARE-TOPOLOGY #10530. It creates no new measurement ontology, uncertainty engine, observability engine, fixture authority or evidence system.

## Paradigm

```
engineering claim
 -> physical measurand
 -> observation method feasible on final article
 -> datum/frame + fixture/instrument + calibration dependencies
 -> uncertainty, common-mode and back-action review
 -> repeatability after service/remount
 -> evidence applicability and claim ceiling
```

Do not collapse:

```
property represented in CAD
!= observable after realization

sensor present
!= measurand identifiable

inspection possible before assembly
!= inspection possible after assembly

measurement possible once
!= measurement repeatable after service

two readings
!= independent evidence

proposed method
!= demonstrated measurement capability
```

## Reused fixture identities

- CADH-P01 / P03 / P04 / P06: patterned plate, enclosure, rounded plate and mounting bracket.
- ASM-P01 / P02: support frame and bearing/shaft alignment.
- ECAD-P01: logical pin/net graph only; no routed board exists.

Fixture geometry and source assumptions remain owned by their original CAD-HISTORY and HARDWARE-TOPOLOGY specifications. This benchmark adds only the claim-to-evidence planning task.

## Case bank

The public manifest contains 12 blueprints: eight positive/decision cases and four critical negative controls. It covers geometric/dimensional inspection, inaccessible internal features, assembly interfaces, remount transfer, common-mode independence, uncertainty vs case-specific tolerance, electrical testability, service/requalification, false post-closure observability, stale calibration, non-identifiable electrical states, and measurement back-action.

All cases are unadjudicated and unscored. The manifest is not a hidden holdout.

## Metrics

Report each separately:
- claim-to-evidence-path coverage;
- direct/derived measurand correctness;
- final-assembly access feasibility;
- exact datum/feature/article identity;
- calibration/reference dependencies;
- uncertainty relative to the actual tolerance or decision boundary;
- common-mode and independence correctness;
- observability/identifiability;
- measurement back-action;
- service/remount requalification completeness;
- Unknown/Unsupported/abstention correctness;
- cost, time, access and destructive-inspection burden.

Critical false-observability, false-calibration, false-independence and false-currentness claims are hard failures. Do not compensate them with coverage or lower cost.

## Proposed targets (pilot calibration required)

- Zero critical false observability passes on the authored negative controls.
- Every critical claim has an explicit disposition.
- Zero false calibration-currentness or independence claims.
- At least 95% correct accepted-claim to measurand/datum mapping on adjudicated pilot fixtures.
- At least 30% reduction in late-discovered inspection/requalification blockers versus a matched checklist baseline, with non-inferior critical recall.

These are prospective targets, not results or universal metrology rules.

## Execution stages

1. Owner review of semantics and fixture references.
2. Independent metrology/domain gold adjudication.
3. Deterministic scorer and evaluator mutation tests.
4. Matched human/checklist/rule/model benchmark.
5. Physical Model–Fixture–Article campaign only through ENG-REF #6121 and existing calibration/safety/authority gates.

## Claim ceiling

A passing synthetic result shows only that the candidate planned admissible evidence paths on the exact declared fixtures. It does not establish an instrument's capability, real-world observability, calibration, measurement uncertainty, product conformity, safety, approval or release authority.
