# MFG-FOUNDATION-001 — Physical Technology Improvement Composition

Status: draft research architecture
Original base subject: `458c7b98d81c64b9361e252f85ef9d45132e6682`

## Purpose

Define how Symthaea/Mycelix can systematically improve mature physical technologies by composing existing evidence, systems-engineering, materials, manufacturing, lifecycle and authority infrastructure.

MFG-FOUNDATION is **not** a new universal manufacturing ontology. Its current architectural theorem is stronger:

```text
MFG-FOUNDATION common production-type budget = 0
```

That is the default hypothesis. A future implementation may add a common type only after an adversarial fixture proves that existing owners cannot represent the required semantic losslessly.

## Core separation

```text
observed limitation
!= dominant bottleneck
!= mechanism established
!= validated intervention
!= manufacturable solution
!= qualified process
!= field performance
!= safe/authorized deployment
```

and:

```text
observation
!= evidence-backed claim
!= causal-model assumption
!= recommendation
!= portfolio allocation
!= authorization
!= execution
!= successful outcome
!= certification
```

## Ownership theorem

Before adding any field/type, ask:

```text
who already owns this fact?
```

If an owner exists, use an exact reference or adapter. Do not copy the value, identity, lineage, calibration, hypothesis, currentness, provenance, lifecycle state, planning state or authority.

#5768 is the hard zero-duplication gate.

## Canonical owners discovered by audit

### Materials

The repaired/current MAT line owns material identity, conditioned evidence, campaign/Pareto semantics, negative/OOD SearchMemory, sample/process lineage and materials-discovery proposal/human-review orchestration.

Do not bind new code to stale historical MAT interfaces while MAT-RFMT #5323 is reconstructing the canonical line.

### Physical quantities and observations

- SE-SEM-001 #4861 — physical quantities, units, frames and time bases.
- FIELD-000 #3589 / FIELD-001A #3633 / SE-OBS #3695 — physical observations, source/device/configuration, calibration/traceability, uncertainty and raw-vs-derived evidence.

Therefore MFG-FOUNDATION does not own `Pressure`, `Power`, `Flow`, `Temperature`, `CrackWidth`, `Torque`, local unit strings or copied measurement payloads.

### Evidence independence

ENG-EVID-INDEP-001 #4918 owns evidence dependency/common-mode lineage, repeat-vs-replication distinctions and claim-relative independence.

```text
multiple observations
!= multiple independent observations
```

### Research split integrity

Historical PR #179 already defines the generic `symthaea-research-split` / `ResearchSplitManifest` theorem: Training/Calibration/Evaluation roles, group-disjoint policies, temporal embargo, missing-group rejection, content-addressed manifest identity and separation evidence.

```text
structural split separation
!= statistical / calibration / organizational independence
```

#5756 therefore owns only BIO-CEM / ROT-EQUIP grouping profiles over a future qualified/current split-contract successor. It does not own another partition engine or receipt type.

Historical #179 is not treated as a current qualified dependency: the crate is absent from audited `main`, the old PR remains a draft with non-clean general CI, and no focused exact-head qualifier was found. Its V1 string-heavy identity fields should also be reviewed against newer canonical identities before current use.

### Evidence-bound claims

ENG-EVID-CLAIM-001 #5771 owns the proposed generic claim/evidence graph and semantic non-laundering layer.

```text
claim A supports claim B
!= claim A becomes claim B
```

and:

```text
observation
!= interpretation
!= mechanism support
!= causal-model edge
!= intervention authority
```

#5757 therefore owns only BIO-CEM / ROT-EQUIP evidence-obligation profiles over #5771.

### Hypotheses / failure causes

Hypothesis identity stays with the domain owner, including MAT-HYPOTHESIS #5254, SE-VV #3697 and more specific domain physics/research subjects.

### Computational causality

`symthaea-causal-reasoning` owns SCM/DAG/do-calculus machinery. Temporal/HDC causal inference and knowledge-extracted causal graphs are model/discovery systems, not evidence authority.

#5771 provides the intended firewall:

```text
evidence-bound claim
+ explicit model-admission profile
-> model-relative causal assumption
```

Never:

```text
Granger significance
or text-extracted relation
or HDC similarity
-> scientifically established mechanism
```

### Technology-function / bottleneck planning

MAT-OPPORTUNITY-001 #5196 owns technology function, bottleneck-hypothesis, intervention and multi-axis opportunity semantics.

MAT-OPPORTUNITY-003 #5245 already owns system-performance questions, limiting variables, sensitivity, achievable headroom, co-bottlenecks, bottleneck migration and competing non-material interventions.

Therefore MFG-FOUNDATION does not own a generic `TechnologyConstraintClass`.

Historical MAT-OPPORTUNITY implementation #5212 is not qualified. Qualifier #5214 / run `35526632327` passed source/parent/scope/planning/lock gates and then failed Rust 1.96 `rustfmt`; tests and Clippy did not execute.

- #5769 — reproducible formatter/ancestry reconstruction of historical V1.
- #5770 — separate semantic refit onto canonical MAT/SE/ENG refs.
- MAT-CONVERGE #5263 — decides the reusable successor interface.

Formatter preservation and semantic modernization remain separate evidence subjects.

### Research portfolio planning

MAT-PORTFOLIO-001 #5247 already owns staged evidence-bounded research allocation, resource budgets, exploration/exploitation, option value, shared infrastructure, common-cause dependence, opportunity cost, VOI/VOC bridges, continuation gates and prospective calibration.

#5758 is therefore a cross-domain adapter into existing planning/decision semantics, not another portfolio engine.

### Components, manufacturing and lifecycle

- ENG-CATALOG #5675 — generic component/article identity.
- MFG-PROC — process, process-state transformation, capability, recipe, plan, inspection/hold/rework and capability history.
- MFG-LIFE — lifecycle strategy/design/assessment.
- Mycelix manufacturing/circularity/BIO-MFG — concrete plan instances, resources/sites, execution/service/lifecycle events and biological lot/containment references.

Preserve:

```text
process plan != plan executed
inspection planned != inspection executed != result accepted
historical capability != future capability guarantee
```

### Currentness / applicability / authority

ETK, SE validity semantics and domain authority systems remain owners. MFG-FOUNDATION may reference/project them; it cannot mint currentness, biosafety, maintenance, manufacturing, funding, procurement or deployment authority.

## Removed proposed common primitives

Audit removed or demoted all initially proposed MFG common types:

```text
TechnologyConstraintClassV1
TechnologyConstraintObservationV1
LeakageGroupRefV1
BenchmarkPartitionPolicyV1
BenchmarkPartitionReceiptV1
MechanismHypothesisV1
MechanismAttributionEdgeV1
ApplicabilityProfileV1
```

The corresponding semantics belong to MAT-OPPORTUNITY/system engineering, existing evidence owners, ENG-EVID-INDEP, ResearchSplitManifest, domain hypothesis owners, ENG-EVID-CLAIM and ETK/SE/domain validity owners.

## First proving domain — BIO-CEM

BIO-CEM #5751/#5753/#5767/#5759 remains the first materials/literature consumer.

Independent output planes remain:

```text
VisibleCrackClosure
TransportRecovery
MechanicalRecovery
MineralizationEvidence
BiologicalViability
DurabilityResponse
RepeatedHealingResponse
```

They must not collapse to a `healing_score`.

BIO-CEM may own genuinely domain-specific vocabulary such as biological observation state, healing output-plane classification, control profile, literature extraction state and method/comparability state.

Material identity, measurements, source dependency, split manifests, claim/evidence derivation, hypothesis identity, sample lineage, operational biological lots, lifecycle and authority remain externally owned.

## Second proving domain — ROT-EQUIP

ROT-EQUIP #5760 uses a deliberately different system boundary:

```text
electrical supply
-> drive/starter
-> motor
-> coupling/gearbox
-> pump | fan | compressor
-> piping/ducting/valves/dampers/restrictions
-> process load
-> controls
```

Independent observation planes include electrical input, mechanical state, fluid/air state, useful output/system performance and maintenance/degradation evidence.

Preserve:

```text
component efficiency != system efficiency
anomaly detected != fault identified != root cause established
maintenance performed != restored performance
predicted savings != measured savings
```

ROT-EQUIP may own only genuinely missing domain vocabulary such as driven-system role/topology or operating-context profiles when no existing systems owner supplies them.

## Cross-domain success criterion

The goal is not identical domain types. It is identical integrity discipline:

```text
BIO-CEM
        \
         > exact refs + split discipline + evidence claims + authority boundaries
        /
ROT-EQUIP
```

without:

- `AnyValue` or free-form JSON semantic escape hatches;
- duplicate quantity/unit systems;
- duplicate observation/calibration systems;
- duplicate independence/replication graphs;
- duplicate split engines;
- duplicate claim graphs;
- duplicate hypothesis/bottleneck taxonomies;
- duplicate portfolio engines;
- duplicate process/lifecycle ledgers;
- duplicate currentness/authority systems.

Only after this is demonstrated should new sectors such as industrial thermal systems, building controls, corrosion/coatings, membranes or transformers become additional adapters.

## Child architecture after compression

- #5768 — zero-new-type ownership gate.
- #5769 — historical MAT-OPPORTUNITY formatter reconstruction.
- #5770 — canonical-reference MAT-OPPORTUNITY semantic refit.
- #5771 — generic evidence-bound claim graph and causal-model admission firewall.
- #5755 — prove whether any cross-domain opportunity/evidence binding remains necessary.
- #5756 — BIO-CEM / ROT-EQUIP profiles over ResearchSplitManifest.
- #5757 — BIO-CEM / ROT-EQUIP profiles over ENG-EVID-CLAIM.
- #5758 — adapter into MAT-OPPORTUNITY / MAT-PORTFOLIO / SE decision semantics.
- #5759 / #5767 — BIO-CEM corpus contract + adapter.
- #5760 — ROT-EQUIP second-domain adapter.
- #5761 — public physical-technology evidence corpus over existing planning subjects.
- #5762 — intervention/outcome integration profile.
- #5763 — research-to-manufacturing transfer assessment adapter.
- #5764 — negative-evidence projection.
- #5765 — applicability/currentness projection.
- #5766 — authority non-escalation/composition profile.

## Type-budget gate

Every future implementation PR beneath #5754 must include:

```text
proposed field/type
semantic meaning
existing owners searched
canonical owner/ref used
adversarial fixture proving why a new semantic is still required
```

Convenience, naming preference, serialization convenience or avoiding a real dependency is not justification.

## Execution order

```text
repair / converge required MAT + planning owners
        ↓
resolve a current qualified research-split owner
        ↓
implement/qualify ENG-EVID-CLAIM if generic extraction survives code audit
        ↓
#5768 re-audit against exact current interfaces
        ↓
freeze #5767 BIO-CEM synthetic fixtures
        ↓
prove BIO-CEM adapter with zero common MFG types
        ↓
prove ROT-EQUIP adapter with zero common MFG types
        ↓
only then expand portfolio/domain adapters
```

## Claim ceiling

This document freezes architecture only. It establishes no compile/test/benchmark/experimental/manufacturing/economic/safety/regulatory/field PASS and no claim that any particular mature technology is the best target for research or investment.