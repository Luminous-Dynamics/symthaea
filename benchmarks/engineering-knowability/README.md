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


## Co-design extension proposal

The next research step extends claim-to-evidence planning into joint design and inspection planning. See [CO_DESIGN_EXTENSION_V0_2.md](CO_DESIGN_EXTENSION_V0_2.md) for a separately versioned proposal with current CAD-agent comparators, eight task-family blueprints, baselines, hard gates and prospective targets. This is not part of the 12-case v0.1 corpus and has no gold labels or scores.

## Standards and metrology references

The manifest pins the following external references to the cases where their scope is relevant:

| Reference | Appropriate use in this benchmark | Important boundary |
|---|---|---|
| [ISO 14253-1:2017](https://www.iso.org/standard/70137.html) | Decision rules for verifying conformity/nonconformity of in-scope quantity-valued characteristics while taking measurement uncertainty into account. | Not an observability standard and not a universal uncertainty-to-tolerance ratio. Apply only to the declared specification/characteristic scope. |
| [ISO 10360-2:2009](https://www.iso.org/standard/40954.html) | Acceptance and reverification tests for Cartesian CMMs using contacting discrete-point probing for linear dimensions. | It does not explicitly apply to optical probing or non-Cartesian CMMs. Do not use it as a generic scanner or all-metrology qualification rule. |
| [JCGM 100:2008/Amd.1:2026](https://www.bipm.org/documents/20126/2071204/JCGM_100_Amd1_2026.pdf/) | Guidance addressing nonlinearity in measurement models. | Does not establish an instrument's calibration or capability. |
| [JCGM GUM-5:2026](https://www.bipm.org/en/web/guest/publications/guides) | Worked examples for uncertainty evaluation and reporting. | Examples are illustrative; they do not certify the benchmark's fixture or method. |
| [JCGM 106:2012](https://www.bipm.org/en/web/guest/publications/guides) | Role of measurement uncertainty in conformity assessment. | Does not replace the governing product specification or declared decision rule. |

The current ISO catalogue records ISO 14253-1:2017 as reviewed and confirmed in 2023 and ISO 10360-2:2009 as reviewed and confirmed in 2026. The BIPM JCGM catalogue lists the 2026 GUM amendment and GUM-5:2026. These are methodological references only; they are not evidence that a particular proposed setup conforms to them.

**No generic 4:1 test-uncertainty ratio is assumed.** The acceptable uncertainty and decision rule are case- and specification-dependent. A method that produces a numeric reading may still be inadequate for the decision boundary.

## Claim-to-measurement contract

The first mapping register is `mapping-matrix.csv`. It has 22 conceptual mappings, each naming a canonical owner, criticality, missing/loss behavior and an independent check. These are semantic contracts, not literal standard-schema element paths.

For every critical claim, the candidate proposal should identify:

1. Exact claim, requirement revision and target article/configuration generation.
2. Physical measurand and whether it is directly observed or derived.
3. Exact feature/datum/reference frame and final assembly state.
4. Method/instrument family and required access, range, resolution and operating conditions.
5. Exact instrument, fixture, calibration and reference-artifact dependencies.
6. Measurement equation, uncertainty contributors, relevant dependence/common-mode effects and decision rule.
7. Whether service, remount, repair or replacement changes the observation path or the evidence applicability.
8. Evidence class and the strongest permissible claim.

The proposal must remain a proposal. Calibration currentness, measurement-system capability, evidence admission and engineering approval remain with their existing owners.

## Baselines and matched comparisons

After independent gold adjudication, compare at least:

- Experienced human metrology/inspection planner.
- Conventional checklist/template workflow.
- Deterministic rule-based claim-to-measurand mapping.
- Frontier-model agent without canonical owner projections.
- Symthaea with canonical owner projections.
- Symthaea ablations (retrieval, causal hypothesis maintenance, and FEP epistemic-action selection) only after the corresponding owners and interfaces are qualified.

Match information access, task revision, available tools, time/resource budget and scoring rules. Split by base geometry, assembly family and measurement-chain family, not randomly by individual row. Use multiple independent runs where agent stochasticity applies; retain full denominators, abstentions, Unknown/Unsupported, failures and resource use.

## Scoring and non-compensatory controls

Keep these outcomes separate:

- Critical claim disposition coverage.
- Correct claim-to-measurand and feature/datum mapping.
- Final-state accessibility/inspection feasibility.
- Instrument/method scope correctness.
- Calibration/reference identity and applicability.
- Uncertainty-model and decision-rule correctness.
- Independence/common-mode correctness.
- Observability/identifiability under the declared model and operating envelope.
- Measurement back-action and disturbance.
- Service/remount/repair requalification completeness.
- Correct Unknown/Unsupported/abstention.
- Time, cost, fixturing burden, access burden and destructive-inspection impact.

Any critical false observability claim, false calibration-currentness claim, false independence claim or stale-evidence false-current disposition is a hard failure. Coverage or lower cost cannot compensate for it.

## Evaluator integrity and staging

Before scoring any candidate:

1. Dual independent metrology/domain assessors label pilot cases without seeing candidate output; disagreements go to a third adjudicator.
2. The gold record binds exact claim, requirement, feature, article, fixture/instrument, calibration and evidence references.
3. Mutation controls must detect omitted claims, a hidden inaccessible feature, stale or substituted calibration, shared common-mode references presented as independent, an unidentifiable measurand, and unmodelled measurement back-action.
4. Public blueprint rows must not be copied into a claimed hidden holdout. Hold out by geometry and measurement-chain family.
5. The scorer must count all tasks and preserve every Unknown, Unsupported, abstention, failure and ungraded case.

Only after those gates should a matched human/rule/model comparison begin. A later physical Model–Fixture–Article campaign must be separately authorized through ENG-REF #6121 and existing article, calibration, safety and evidence authorities.


## Industry benchmark context — captured 2026-11-06

The machine-readable manifest pins four external comparison snapshots. They are **context**, not a merged leaderboard and not scores for this knowability benchmark.

| Comparator | Publicly reported reference | What it measures | What remains outside its score |
|---|---|---|---|
| [Physical-Engineering's Last Exam (PLE), all 85](https://ple-bench.com/leaderboard?set=all) | Claude Opus 5.5 + Claude Code: 0.506, $17.18/task, 63 min median; GPT-6 Astra + Codex: 0.483, $9.15/task, 25 min median | Four professional engineering disciplines; three full runs per model+harness pair; six-hour/task; official rubric judge | The published mean is not a metrology capability, calibration, physical conformity or post-service requalification result |
| [PLE public development, 29 tasks](https://ple-bench.com/leaderboard?set=dev) | Claude Opus 5.5 + Claude Code: 0.551; Claude Opus 5.5 + Codex: 0.549; GPT-6 Astra + Codex: 0.530 | Same public development split and three-run protocol | Not comparable with PLE hidden 56, all-85 or committed-artifact submission mode |
| [Parametric CAD Bench V3](https://cadbench.ai/leaderboard) | Claude Opus 5.5 + Claude Code: 0.6103 (displayed 95% CI ±0.0753), 99/100 scored, $573.94 total | 100 public tasks: 30 text creation, 30 create-and-edit, 40 engineering drawings; separate programmatic FreeCAD validator | Does not score inspection access, calibrated measurement capability, maintenance or physical conformity |
| [BenchCAD 2.0 Core](https://benchcad.com/blog/benchcad-2) | GPT-6 Astra: 0.59, about $11/task, 62 min median; Claude Opus 5.5: 0.57, about $28/task, 222 min median | 100 held-out hardware-engineering cases; geometry/topology, assembly placement and ECAD connectivity | Does not score manufacturability, physical performance, metrology, inspectability or lifecycle requalification; v2.0 is not comparable with v2.1 |

These comparisons show why model-only ranking is insufficient. PLE's live board varies both model and harness; on the 29-task development set, the same Claude Opus 5.5 system has visible harness-dependent score and cost changes. Benchmark records must therefore identify the full model × harness × effort × task-split × evaluator protocol. Never pool the above means or use the highest point estimate as a statistically decisive ranking without considering the stated uncertainty and protocol.

### The opportunity beyond current SOTA

Current external CAD and engineering benchmarks establish useful—but separate—capabilities: a generated CAD artifact can be geometrically similar, a drawing can be reconstructed, a mechanism can be assembled, or a net graph can be recovered. The knowability benchmark asks a different question:

> Does the designed article retain a feasible, identifiable, uncertainty-aware evidence path for every critical claim after assembly, service, remount and repair?

This is intended as an orthogonal capability layer, not a claim that the current 12 blueprints beat or even reproduce an external leaderboard. No PLE, BenchCAD or CAD Bench task is imported into this corpus; their dataset terms and scoring boundaries remain separate.

The immediate extension is [ENG-KNOW-CODESIGN-001 #10702](https://github.com/Luminous-Dynamics/symthaea/issues/10702): test whether integrating inspection, measurement, diagnosis and requalification affordances during design reduces late blockers and lifecycle burden without sacrificing functional feasibility. It composes #10652 with existing SE-OPT #3686, ENG-REF #6121 and canonical metrology/tolerance/lifecycle owners; it creates no new optimizer or measurement authority.


| [CADWorld](https://cad-world.github.io/) | Claude Opus 5.5 extended-run GUI agent: 41.5% on the 200-task FreeCAD cohort; 87.0% expert reference is a single-trial result | Long-horizon native FreeCAD workflows spanning sketching, part design, assembly, CAM, FEM, measurement, mesh, and technical drawing; task-specific executable checks on saved files | Not a population-level human baseline; does not grade complete claim/evidence validity, uncertainty/calibration correctness or lifecycle requalification. The original seven-agent paper snapshot (17.5% strongest result) is separate from the current live extended-run page |


## Standards interoperability dependency

Compose with [MBD-INTEROP-001 #10595](https://github.com/Luminous-Dynamics/symthaea/issues/10595), which owns the bounded semantic round-trip benchmark across STEP AP242, QIF and MTConnect. This benchmark does **not** add a second translator, standards bridge or semantic identity engine.

The knowability task must preserve exact design generation, product/component/assembly identity, feature/datum association, semantic PMI characteristic identity, article identity, inspection-plan/result binding and calibration references when those artifacts cross the declared exchange path. A visually intact annotation that has lost its semantic datum/feature association is a critical failure.

The two directly relevant exchange standards are:

- [ISO 10303-242:2025, AP242 Edition 4](https://www.iso.org/standard/84300.html), published 25 August 2025. The ISO catalogue status was "to be revised" at retrieval (6 November 2026); pin the exact edition and application profile. A bounded subset is not full AP242 conformance.
- [ISO 23952:2020, QIF](https://www.iso.org/standard/77461.html), reviewed and confirmed on 14 June 2026. It defines an integrated model for manufacturing quality information and quality-workflow documents. Schema validity alone does not prove that an inspection result is bound to the correct characteristic, article, instrument or calibration.

The mapping register adds three semantic contracts for profile/edition scope, semantic PMI preservation, and exact inspection-result binding. Full exchange fidelity remains MBD-INTEROP's owner; knowability evaluates whether the preserved information is sufficient to support an admissible evidence path.
