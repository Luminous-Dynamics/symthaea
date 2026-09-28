# Critical Platform Architecture Registry v1

Status: architecture registry / preregistration
Claim ceiling: this registry establishes only architectural ownership, dependency, evidence, and qualification-boundary semantics. It establishes no physical capability, safety, service sufficiency, productive closure, regulatory approval, or operational authority.

## 1. Purpose

This registry is the concrete design index for critical Symthaea/CIV infrastructure. It prevents platform programs from independently inventing overlapping state, evidence, identity, lifecycle, closure, or authority systems.

The registry is a composition map, not a universal engine.

## 2. Global architectural invariants

1. Every platform has one explicit semantic owner for each canonical state it creates.
2. Cross-platform designs reference canonical identities rather than copying authoritative records.
3. A generation/currentness change creates a new semantic subject where the changed field can affect evidence validity.
4. Prediction, observation, engineering qualification, operational fact, and authority remain distinct.
5. Synthetic qualification never establishes physical qualification.
6. A service event never creates engineering evidence.
7. An engineering result never creates operational authority.
8. Mycelix provenance/coordination facts never manufacture Symthaea engineering truth.
9. CIV capability/service projections never become hidden physics or optimization engines.
10. Missing, stale, conflicting, or inapplicable evidence narrows claims or produces an explicit unresolved disposition.
11. Historical evidence is immutable; recomputation creates derived dispositions.
12. Every production adapter must identify its predecessor contract and independent qualification prerequisite.

## 3. Common platform contract

Each critical platform should expose, directly or through an adapter, these conceptual fields:

- subject identity and generation;
- declared profile/environment;
- owned state;
- referenced state;
- evidence references;
- dependency references;
- applicability/currentness;
- uncertainty/conflict state;
- lifecycle/repair/requalification state;
- authority disposition;
- derived capability/service disposition;
- replay identity;
- claim ceiling.

No platform may collapse these into one readiness or maturity score.

## 4. Platform registry

| ID | Platform | Primary responsibility | Critical predecessors |
|---|---|---|---|
| CP-01 | Metrology / characterization | measurand, instrument, calibration, observation, uncertainty, traceability | SE-SEM, FIELD, SE-OBS |
| CP-02 | Materials convergence | demand → candidate → process → property → measurement → engineering consequence | MAT-CONVERGE-002A2, materials owners |
| CP-03 | Manufacturing | design/process/as-built/metrology/commissioning/lifecycle evidence | MFG-PROC, ENG-DESIGN, SE-VV |
| CP-04 | Compute | representation → model/runtime → accelerator/deployment evidence | BinaryHV/CfC, compute/HDL owners |
| CP-05 | Sensing | observable → sensor → calibration → state/evidence | SENSE, FIELD, SE-OBS |
| CP-06 | Energy | generation/storage/conversion/distribution/load/degradation/replacement | energy/grid/storage owners |
| CP-07 | Water / fluids | source → characterization → equipment/process → observation → service | WATER-MFG-000, IND-COMP |
| CP-08 | Agriculture / food infrastructure | productive equipment → production-support → processing/storage/cold-chain service | AGRI-MFG-000 |
| CP-09 | Robotics / autonomy | design → embodiment → sensing/state → planning → bounded action → observation | ROB-DESIGN, ROB-REALIZE, ETK/HAL |
| CP-10 | Transport | mission → platform → subsystem → corridor/infrastructure → commissioning → service | TRANSPORT-ENG-001, MOBILITY-MFG-000 |
| CP-11 | Software / formal VV | specification → implementation → proof/test → artifact/deployment identity | SE-VV, formal verification owners |
| CP-12 | CIV service | qualified capability → asset/service instance → availability → continuity → operational projection | CIV-SERVICE |
| CP-13 | CIV productive closure | capability vectors across maintenance/repair/replacement/renewal | CIV-BOOT-000..005 |
| CP-14 | CIV value / industrial ecology | resource → transformation → asset → service → circular return | CIV-VALUE-000, PIE, MFG |
| CP-15 | Symthaea ↔ Mycelix seam | engineering/evidence ↔ identity/provenance/coordination/operations | COS/seam/evidence owners |

## 5. Dependency order

The preferred implementation order is:

Metrology → Materials → Manufacturing → Compute/Sensing/Energy → Robotics/Transport/Utilities → CIV-Service → CIV-BOOT → Federation/Interplanetary.

This is an implementation dependency ordering, not a priority ranking.

## 6. Required qualification pattern

Every platform follows:

architecture contract
→ canonical synthetic corpus
→ independent oracle
→ adversarial mutations
→ deterministic replay
→ qualification receipt
→ hosted provenance
→ production adapter.

A platform PASS must identify its exact source snapshot and claim ceiling.

## 7. Boundary rules

### Symthaea

Owns reasoning/model/evidence semantics, candidate generation, uncertainty, applicability, experimental inference, engineering projections, and scientific workflow where explicitly assigned.

### Mycelix

Owns identity/provenance, coordination, attestation, custody, economic/resource context, organizational boundaries, and operational events where explicitly assigned.

### CIV

Projects qualified capabilities into profile-relative service/productive/civilization state. It does not mint underlying physics, observations, regulatory approvals, or operational authority.

## 8. Required adversarial families

Every platform qualification should test, as applicable:

- equal numeric value from different evidence authorities;
- stale generation/currentness;
- changed profile applicability;
- copied or rewritten historical evidence;
- one specimen presented as population evidence;
- prediction presented as observation;
- operational event presented as engineering qualification;
- correlated evaluators presented as independent;
- missing measurement capability;
- manufacturing dependency hidden by aggregate success;
- lifecycle/renewal dependency hidden by initial operation;
- negative evidence lost during recomputation;
- authority promotion from recommendation;
- synthetic PASS presented as physical qualification.

## 9. Concrete design train

### Phase A — contracts

Freeze one architecture contract per platform and a cross-platform ownership/gap matrix.

### Phase B — synthetic reference systems

Use benign, deterministic fixtures that exercise the complete digital thread without creating physical execution authority.

### Phase C — convergence

Compose qualified contracts through immutable references. Do not create duplicate domain databases merely to make integration easier.

### Phase D — bounded physical evidence

Only after predecessor qualification, run low-consequence physical characterization with explicit metrology, safety, commissioning, and lifecycle boundaries.

### Phase E — operational projection

Project qualified capability into CIV-Service/Mycelix operational state without allowing the operational record to back-propagate authority into engineering evidence.

## 10. Design review gates

A platform is not ready for production integration unless reviewers can answer:

1. What state does it own?
2. What state does it only reference?
3. What exact generation identifies that state?
4. Which evidence can invalidate it?
5. Which changes require a new generation?
6. What negative evidence survives recomputation?
7. What does a PASS actually prove?
8. What does it explicitly not prove?
9. Which existing owner is reused instead of duplicated?
10. Which Mycelix/CIV seam carries operational consequences?
11. Which physical-world boundary remains external?
12. Can the complete result be replayed from immutable references?

## 11. First implementation tranche

The first concrete designs should be:

1. CP-01 Metrology/characterization contract;
2. CP-02 Materials convergence contract, using MAT-CONVERGE-002A2 as the qualification pattern;
3. CP-03 Manufacturing digital-thread contract;
4. CP-15 Symthaea↔Mycelix seam contract;
5. CP-12 CIV-Service projection;
6. CP-13 CIV-BOOT capability-state projection.

Only then should higher-level integrated platform fixtures be promoted.

## 12. Nonclaims

This registry does not claim that Symthaea currently possesses any listed physical manufacturing, energy, water, transport, agricultural, robotic, compute, sensing, metrology, or interplanetary capability. It defines how such claims must be decomposed, evidenced, qualified, and kept within their actual authority boundaries.
