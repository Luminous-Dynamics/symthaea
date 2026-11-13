# ENG-DESIGN-CHANGE-QUAL-001 — Public fixture blueprint bank v0.1

**Status:** DESIGN_ONLY / public task-authoring specification  
**Tracking:** [ENG-DESIGN-CHANGE-QUAL-001 #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543)  
**Semantic change owner:** [ENG-DESIGN-005 #6117](https://github.com/Luminous-Dynamics/symthaea/issues/6117)  
**Lifecycle / authority owners:** [Systems Engineering #3677](https://github.com/Luminous-Dynamics/symthaea/issues/3677), ETK  
**Execution:** no candidate system, external MBSE tool, CAD/CAE runner, production record, or human assessor campaign was executed for this document.

## 1. What this bank is

This document turns the 24 task families in #10543 into named public fixture blueprints. It is **not** a sealed leaderboard or a qualified benchmark. No gold answer key is included in the candidate-visible part of this specification. Exact affected/unaffected claim labels, evidence IDs, and preservation witnesses must be independently authored and adjudicated before a fixture is accepted into a scored development set. A future held-out set must be generated from different design families and kept outside the candidate-visible workspace.

The benchmark measures two different failure directions:
- **Under-impact:** a critical downstream claim or verification obligation is missed and stale evidence is incorrectly treated as applicable.
- **Over-impact:** unrelated evidence is needlessly invalidated, causing avoidable requalification.

The first is a hard correctness concern. The second is an operational-efficiency concern and cannot be optimized by accepting stale evidence.

## 2. Existing-owner composition

| Domain | Authoritative owner | This corpus consumes |
|---|---|---|
| Bounded change/delta semantics | ENG-DESIGN-005 #6117 | exact source/target generation and typed delta |
| Requirements, interfaces, assumptions, verification intent, impact paths | SE-001 / Systems Engineering #3677 | typed dependency graph snapshot and candidate impact paths |
| Evidence applicability/currentness and assurance | ETK #1738/#1730/#1760 | read-only prior evidence/currentness facts and review requests |
| Generic proof/evidence non-amplification | SYM-FV #5794/#6065/#6073 | claim ceiling and no-authority properties |
| Repository/code impact gold set | #10173 | code/repository graph accuracy and edit-to-code/test impact; separate corpus |
| CAD feature and edit intent | CAD-HISTORY-001 #10510 | feature/parameter/dependency subjects |
| Assembly, component and ECAD identity | HARDWARE-TOPOLOGY-001 #10530; ROB-CATALOG #4877 | exact article, variant, pin/net, mating and assembly subjects |
| Tolerance and measurement | ENG-TOL-001 #4883 | dimensional requirement, measurement and calibration applicability |
| As-built realization | ROB-REALIZE #4859 | article/configuration lineage |
| Manufacturing route and production outcome | MFG-VALUE #7608; PCB capability #8223 | machine/process/capability/inspection evidence |

The benchmark does not create a new impact graph, delta model, receipt/authority system, currentness kernel or release path. A proposed impact path is not itself an evidence invalidation decision; the candidate must identify the exact evidence and claim that it affects, and the owner-authorized layer retains authority.

## 3. Public blueprint bank (24 scenarios)

These are authoring blueprints. Each must later be materialized as an exact before/after semantic subject plus an independently adjudicated dependency/evidence fixture.

| ID | Change family | Proposed delta | Key benchmark trap |
|---|---|---|---|
| EDC-A01 | Documentation-only | Edit explanatory prose, no identity-bearing requirement/model/parameter changes | Do not demand physical requalification just because bytes changed |
| EDC-A02 | Unit equivalence | Convert a dimension from mm to in with a qualified exact quantity/frame witness | Preserve physical meaning, but retain new serialization/provenance |
| EDC-A03 | Canonical reserialization | Reorder a semantically identical graph under a declared canonicalization profile | Source-byte digest changes; semantic identity may remain |
| EDC-A04 | Lossy export | Re-export geometry through a lossy/approximate representation without equivalence witness | Visual similarity alone cannot preserve dependent claims |
| EDC-B01 | Cosmetic geometry | Change a non-functional cosmetic radius outside a proven dependency cone | Preserve unrelated structural claims but review surface/finish obligations if affected |
| EDC-B02 | Structural geometry | Reduce a load-bearing cross-section | Reopen structural analysis, tolerance, and manufacturing dependencies |
| EDC-B03 | Datum/frame | Move or rotate a mating datum while leaving other numeric dimensions unchanged | Invalidate frame-dependent assembly and analysis claims |
| EDC-B04 | Feature intent | Preserve the initial solid but alter the feature tree so a required downstream edit no longer behaves | Static geometry equivalence is not semantic/edit equivalence |
| EDC-C01 | Material properties | Substitute same-sized material with different modulus/thermal/environmental properties | Reopen property-dependent analysis and manufacturing claims |
| EDC-C02 | Component variant | Change manufacturer part revision/variant with visually equivalent geometry | Preserve historical article identity; review fit/rating/electrical claims |
| EDC-C03 | Supplier / lot | Change supplier or lot with same nominal part number | Review authenticity, lot condition, inspection and traceability, not unrelated geometry |
| EDC-C04 | Approved equivalence | Replace with approved exact variant plus bounded equivalence witness | Reuse only claims explicitly covered by that witness |
| EDC-D01 | Load / duty | Change load magnitude, direction or duty cycle without changing geometry | Reopen analyses depending on the changed operating envelope |
| EDC-D02 | Environment | Change temperature, humidity, vibration, ingress or EMI requirement | Reopen only material/component/verification obligations that depend on it |
| EDC-D03 | Requirement threshold | Change a hard requirement limit or reference revision | Old verification does not automatically satisfy the new requirement |
| EDC-D04 | Preference-only | Change a non-binding aesthetic or preference objective | Recompute trade-space; do not weaken hard requirements |
| EDC-E01 | Process route | Change machine/process route with nominal CAD fixed | Reopen process capability, tolerance and inspection evidence |
| EDC-E02 | Tolerance semantics | Change datum, tolerance zone or acceptance interpretation | Reopen affected measurement/fit evidence; no silent relaxation |
| EDC-E03 | Metrology | Change instrument, calibration or uncertainty profile | Review affected measurements; old observation remains historical, not automatically current |
| EDC-E04 | Assembly process | Change access/tool/fastener sequence while final geometry is unchanged | Reopen access and assembly-process feasibility |
| EDC-F01 | Software/runtime | Change controller software or runtime configuration | Reopen software/functional verification; preserve mechanical claims only with a valid independence witness |
| EDC-F02 | Sensor/calibration | Change sensor mapping, offset or calibration | Reopen derived measurement, control and physical-validation obligations |
| EDC-F03 | External authority | Change an external standard edition or applicability profile | Review affected mappings; document revision alone is not proof of compliance/noncompliance |
| EDC-F04 | Rollback | Return to an exact prior design generation | Historical evidence may be referenced, but current configuration/incidents/calibration applicability must still be checked |

## 4. Mandatory task record structure

Each materialized public development fixture must bind:
- fixture ID and schema/profile version;
- exact baseline design/configuration digest;
- exact candidate design/configuration digest;
- exact delta/change-set digest and typed change classes;
- semantic graph snapshot identity and owner/source revision;
- prior evidence-manifest identity, including historical timestamps/currentness state;
- candidate-visible references and allowed context;
- evaluator revision and independent adjudication record;
- expected terminal status for each claim/evidence item;
- expected new/reopened verification and validation obligations;
- expected unknown/unsupported states;
- full attempt / timeout / refusal / ungraded accounting.

The candidate must not see the evaluator-only affected/unaffected sets, exact gold preservation witnesses, or mutation controls in scored runs. The 24 names above are public blueprint metadata, not sealed labels.

## 5. Terminal dispositions

Each claim/evidence item receives its own disposition; do not assign one global status to the entire design:
- PreservedByIdentity
- PreservedByQualifiedEquivalence
- PreservedByExactRecalculation
- RequiresApplicabilityReview
- RequiresVerification
- RequiresPhysicalEvidence
- RequiresHumanOrExternalApproval
- NotPreserved
- UnknownImpact
- Unsupported

Historical source evidence is immutable. A new design generation may refer to old evidence only through an explicit applicability/preservation witness accepted by the existing owner.

## 6. Adversarial evaluator controls

At least these mutations must be rejected or cause the appropriate metric violation:

1. Drop one affected critical claim from the gold set.
2. Invent a nonimpact witness that has no exact source/delta/dependency binding.
3. Remove one prior evidence record while keeping the evidence-manifest digest unchanged.
4. Alter the baseline or candidate digest after the gold labels were created.
5. Change an evidence timestamp so evidence appears to predate or postdate the decision incorrectly.
6. Mark a derived estimate as a direct observation.
7. Omit a seeded material/load/calibration/process dependency.
8. Duplicate one task and remove another while preserving the advertised count.
9. Convert an unknown dependency into PreservedByIdentity without an identity-bearing fact.
10. Promote a benchmark receipt into an ETK acceptance/release/authorization claim.

## 7. Scorecard

| Dimension | Measures | Non-compensable guard |
|---|---|---|
| Impact completeness | Recall of affected claims/evidence/obligations; per-class confusion matrix | Zero missed critical impacts in the frozen synthetic corpus |
| Impact precision | Precision / false-positive rate by change family | Do not hide over-requalification behind recall |
| Applicability correctness | Confusion matrix of terminal dispositions; witness validity | Zero stale-evidence false-current dispositions |
| Preservation quality | Valid identity/equivalence/recalculation witnesses | No preservation from visual or textual similarity alone |
| Unknown handling | Correct Unknown/Unsupported/clarification rate | No invented dependency or false certainty |
| Efficiency | Correctly avoided rechecks, review hours, cost per accepted assessment | No efficiency gain may compensate for a critical false negative |
| Integrity | Exact digests, complete case accounting, deterministic replay | Zero omitted cases / evaluator mutations accepted |

Use paired per-task comparisons and cluster uncertainty at the design-family/lineage level. Do not treat several mutations of one base design as independent samples.

## 8. Baselines and ablations

- Full requalification after every change: conservative reference with expected over-requalification cost.
- Byte/field diff only: low-cost baseline that cannot reason about semantic dependencies.
- Static dependency graph traversal without evidence/currentness semantics.
- ENG-DESIGN delta + SE-001 impact-path projection.
- Full candidate assessment with evidence applicability and selective clarification.
- Qualified human systems-engineering reviewers, with disagreement and review time reported separately.
- Commercial MBSE/PLM workflow only when product access, license and task/harness parity permit a same-task comparison.

For Symthaea, ablate SE-001 semantic relationships, ENG-DESIGN delta representation and evidence-applicability reasoning independently. ETK remains the sole authority for accepted currentness/discharge.

## 9. Prospective gates and interpretation

The linked issue proposes 100% recall on seeded critical-impact changes, zero stale-evidence false-current dispositions, at least 95% precision on independently reviewed supported impact relations, at least 90% correct preservation-vs-review classification, and at least 30% fewer unnecessary re-verification obligations than full requalification **without reducing critical-impact recall**.

These are pilot hypotheses, not achieved results, industry standards or universal safety thresholds. The 24-task bank is a contract/pilot set; a confirmatory evaluation requires independent gold adjudication, design-family holdouts, variance/power planning and a sealed evaluator.

## 10. Claim ceiling

This public fixture bank defines task specifications only. It contains no qualified design graph implementation, gold-labeled scored corpus, candidate-agent run, ETK applicability decision, professional approval, physical test, manufacturing release or safety/compliance result. A synthetic benchmark PASS cannot authorize fabrication, operation or release.
