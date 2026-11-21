# Orbital Joint Evidence Value-of-Information Adapter v0.1

**Tracker:** [ORBITAL-JOINT-EVIDENCE-VOI-001 #13111](https://github.com/Luminous-Dynamics/symthaea/issues/13111)  
**Status:** research contract and synthetic scenario catalogue; no selection-performance result claimed.  
**Claim ceiling:** synthetic evidence-action admissibility and decision-value semantics only.

## 1. Purpose

This adapter asks which engineering evaluation should happen next for an exact structural-interface decision: analytical bound, external solver run, mesh/fidelity refinement, material characterization, coupon test, NDE/inspectability study, metrology, repeated mating/wear study or assembled-structure test.

The objective is not maximum entropy reduction, maximum model disagreement, maximum experiment count, or maximum Pareto hypervolume. It is improved engineering decisions under a finite and fully-accounted validation budget, without making a hard acceptance gate compensable.

The decision pipeline is:

`frozen decision context -> unresolved claims -> admissible evidence actions -> predicted decision consequences -> selected action or abstention -> independently recorded result and full cost -> evidence update`

An action is a proposal. It is not an instruction to execute a solver, machine, fixture or physical test.

## 2. Existing selector: useful baseline, not yet the aerospace solution

Symthaea already provides a generic hypothesis-disagreement selector:
- [experiment_selection.rs](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-core/src/hdc/conjecture_engine/experiment_selection.rs) scores the population variance of candidate-hypothesis predictions and selects the most discriminating candidate.
- [isru_material_experiment_selection.rs](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-core/examples/isru_material_experiment_selection.rs) demonstrates hidden synthetic truth after action selection, deterministic baselines, a held-out OOD world and matched candidate cost/noise.

The ISRU example is careful about its scope: it does not claim Bayesian mutual information or a cost-/noise-weighted acquisition function, and it gives every candidate equal cost and measurement noise. That makes it a useful baseline control, but it cannot distinguish (for example) a two-hour calibrated NDE inspection from a multi-day test-cell campaign, nor does prediction disagreement necessarily measure whether the result changes an engineering decision.

This adapter therefore composes the existing selector and does not replace it. It adds aerospace-specific admissibility, hard-claim and cost semantics plus a decision-value evaluation contract.

## 3. Canonical ownership

### Symthaea

- [SE-OPT-BENCH-001 #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803): generic decision-impact / VOI benchmark.
- [AERO-ENGINEERING-LOOP-001 #10902](https://github.com/Luminous-Dynamics/symthaea/issues/10902): adaptive engineering evidence gathering.
- [SPACE-STRUCT-CODESIGN-001 #12319](https://github.com/Luminous-Dynamics/symthaea/issues/12319): design-to-qualified-physics handoff.
- [MAT-UQ-001 #6225](https://github.com/Luminous-Dynamics/symthaea/issues/6225): uncertainty, discrepancy, calibration and common-mode semantics.
- [MFG-FOUNDATION-001I #5763](https://github.com/Luminous-Dynamics/symthaea/issues/5763), [MAT-PSPP-001 #6224](https://github.com/Luminous-Dynamics/symthaea/issues/6224), [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686): material/process state and scale transfer.

### Symtropy

- [Structural interface profile / conformance #6523 and PR #6524](https://github.com/Luminous-Dynamics/symtropy/pull/6524).
- [Qualification test-matrix adequacy #6540](https://github.com/Luminous-Dynamics/symtropy/issues/6540).
- [Interface lifecycle qualification #6484](https://github.com/Luminous-Dynamics/symtropy/issues/6484).
- [Ground test cell #6256](https://github.com/Luminous-Dynamics/symtropy/issues/6256).
- [Aerospace inspectability / NDE #4165](https://github.com/Luminous-Dynamics/symtropy/issues/4165).
- [Connector coupon characterization #5771](https://github.com/Luminous-Dynamics/symtropy/issues/5771).

### Mycelix and Sol-Atlas

- Mycelix [component build packet #6345](https://github.com/Luminous-Dynamics/mycelix/issues/6345) and [AP242/QIF semantic interoperability #6348](https://github.com/Luminous-Dynamics/mycelix/issues/6348) supply canonical article, configuration and evidence joins.
- Sol-Atlas [South African space ecosystem #405](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/405) supplies read-only site/facility/logistics context. Facility context is not proof of operational availability, calibration or process/test qualification.

No second solver, optimizer, material ontology, provenance store, inspection authority, test-matrix evaluator or lifecycle state machine is introduced here.

## 4. Paradigm shift: decision impact and admissibility before acquisition score

A high-uncertainty parameter can be irrelevant to the current decision. A low-uncertainty parameter close to a hard boundary can be decisive. An expensive test can be inadmissible because the exact article, procedure, facility, instrument or calibration scope is wrong.

Each candidate action must first pass a fail-closed admissibility gate. Only then may it be scored.

### 4.1 Admissibility gates

Every action must bind and satisfy, where relevant:

1. exact client requirement, provider profile, serialized article, configuration, ICD and claim identity;
2. the unresolved requirement or decision whose outcome could change;
3. action method, procedure, model, solver, mesh or inspection revision and declared applicability;
4. specimen/material/process lineage and transfer scope;
5. instrument, calibration validity, range, uncertainty, timebase and raw-evidence method;
6. facility asset, permitted activity, operational state and resource availability;
7. the required pre-registered test obligation and critical interaction, if the action is a physical campaign item;
8. required safety review, approvals, human witness and execution authority;
9. a complete resource vector and budget availability;
10. exact outputs expected, their independent verifier and claim ceiling.

A syntactically valid but stale, out-of-domain, uncalibrated, wrong-serial or unsafe action is not “low value”; it is inadmissible. Do not let an acquisition score rank it above an admissible action.

### 4.2 Keep two uncertainty regimes separate

**Probability-bearing regime.** Expected value of sample information (EVSI) or expected decision regret is permitted only if the decision-state probabilities and observation likelihoods are explicitly supplied with provenance and calibration evidence for the exact domain/profile. The expected loss before action is:

`R_0 = min_d sum_s p(s) L(d,s)`

For action `a`, using observation likelihood `p(o|s,a)`, the expected post-action decision loss is:

`R_a = sum_o p(o|a) min_d sum_s p(s|o,a) L(d,s)`

`EVSI(a) = R_0 - R_a`

The action's cost remains separately reported and budget-constrained. A cost penalty may be subtracted only when the decision owner supplies an explicit, auditable conversion from resources to decision utility. Never invent such a conversion.

**Distribution-free regime.** When only hard sets, intervals or constraints are defensible, do not synthesize probabilities. Use a separately defined worst-case regret / robust-feasibility measure over the declared state set and observation partition. The result must retain whether the set is rectangular, correlated, or otherwise structured. An interval is not a probability distribution.

A benchmark instance must choose one regime. It cannot silently move between expected and distribution-free calculations.

### 4.3 Hard constraints remain non-compensatory

A safety/structural hard constraint cannot be offset by mass, cost, schedule, expected information, or an attractive mean score. An action that does not resolve a blocking claim may still be useful, but it cannot make the design eligible by itself.

Unknown, unsupported, missing, out-of-domain, non-converged or invalid evidence stays unresolved. No evidence is imputed into a favorable result.


### Mycelix and Symtropy change-impact owners

Use Mycelix [LISE-ASSURANCE-INVALIDATION-001 #6219](https://github.com/Luminous-Dynamics/mycelix/issues/6219) as the generic owner of event-driven evidence invalidation, dependency closure, historical preservation and targeted requalification. Reuse its existing owners for evidence leases (#181), applicability/subject drift (#1087), evidence epochs and causal ordering (#1207/#2631), bitemporal reconstruction (#6069), source/observer dependencies (#6140) and independent oracle separation (#4835). Do not add another generic graph or currentness service here.

The aviation-specific application is Symtropy [AEROWEFT-MODEL-EVIDENCE-REQUALIFICATION-001 #5371](https://github.com/Luminous-Dynamics/symtropy/issues/5371). Structural lifecycle qualification remains under Symtropy #6484 and as-built structural acceptance under #4263.

The aerospace evidence-action consumer should take the following event-to-claim relationships from those owners, not infer them from names:

| Change event | Potentially affected claims | Non-automatic invalidation boundary |
|---|---|---|
| CAD / ICD / configuration revision | PMI joins, geometry/fit, mating, structural-model applicability, assembly plan, inspection coverage | independent sensor calibration, unrelated material history |
| Serialized unit replacement | unit-specific build, inspection, test, assembly and as-built acceptance | component-family definition and other serial units |
| Material batch / lot / condition | material/process characterization, process capability, allowables and derived structural claims | unchanged design geometry semantics |
| Process route, machine or tooling change | process qualification, as-run build, material-state transfer and affected dimensional/structural claims | unchanged interface definition or unrelated instrument calibration |
| Instrument / reference calibration | measurement claims within that instrument/reference scope | non-measurement design claims |
| NDE method / procedure / probe / coverage | defect-, orientation- and method-specific inspectability/POD/false-call claims | unrelated structural simulation outputs |
| Structural solver / mesh / model / material model | model applicability, correlation and derived model claims | independent physical observations |
| Robot / tool / firmware / task-plan change | grasp, insertion, lock, verification and affected assembly claims | independently qualified material/design facts |
| Repair / joint replacement | article lifecycle and affected joint acceptance/requalification | unaffected claims with verified independent dependencies |

These are candidate mappings requiring owner-reviewed edges, not universal rules. A change can make an old claim historical-only or require requalification without proving the physical article has failed. An incomplete change feed is INDETERMINATE, not evidence that nothing changed.

The selector's own frozen context tuple (profile reference, physical serial, configuration, ICD revision) is a final consumer-side false-join gate; it is not a substitute for #6219 dependency closure. Symthaea only ranks actions after the owner-native invalidation result and affected claim set are supplied. It must not promote a claim or decide which dependencies exist. JEV-016 freezes an incomplete event-frontier receipt and blocks selection; JEV-017 freezes a complete receipt requiring one structural claim to be requalified while an unrelated design claim remains unaffected.


### Change-impact consumer contract v0.1

The current synthetic selector accepts a benchmark-only `ChangeImpact` record with:
- `source_owner = mycelix:6219`;
- a receipt reference and event-frontier reference;
- `completeness = COMPLETE | INCOMPLETE`;
- exact `required_requalification_claims` and `unaffected_claims`.

This is a consumer-side fixture contract, **not** a replacement for the Mycelix #6219 event schema, dependency graph, temporal model, evidence leases or invalidation oracle. References in the synthetic fixture are not authenticated or resolved against a production service.

The selection rule is deliberately fail-closed:
1. If the owner-native event frontier is incomplete, no action is eligible—even when a proposed test looks informative.
2. If the frontier is complete, a requalification action must explicitly cover every required affected claim.
3. A check of an unrelated or unaffected claim cannot satisfy the affected requalification obligation.
4. Historical evidence remains historical; invalidation is not proof of physical failure.
5. The adapter cannot infer dependency edges, decide that a claim is unaffected, or set the change-frontier completeness flag. Those are owner-native inputs.

JEV-016 checks abstention under incomplete event coverage. JEV-017 checks that the target structural claim is requalified while an explicitly unaffected design-identity claim is not needlessly repeated. The production adapter remains future work and requires exact schema mapping plus authenticated, independently reviewed receipts from Mycelix #6219 and domain rules from Symtropy #5371/#6484/#4263.

## 5. Resource accounting

Keep the resource vector explicit:

- engineering / reviewer hours;
- solver or compute core-hours;
- facility / instrument hours;
- specimen count and material consumption;
- calibration / setup / fixture preparation;
- external tool or licensing cost;
- post-processing and independent verification;
- expected rework / repeat campaign reserve, when explicitly supported.

A policy may only select an action whose declared resource vector fits the remaining component-wise budget and prerequisites. Record actual resource use after execution. Failed, aborted, invalid, censored, out-of-domain and incomplete attempts consume their actual resources and remain visible.

Do not compare policies on nominal simulation cost while excluding test preparation, human review, independent replay, failed calls or follow-on physical validation.

## 6. Action and result contract

A future versioned action request should contain:

`EvidenceActionProposalV1`

- `action_id`, policy ID/version and proposal generation;
- exact decision-context, requirement, provider/client profile, article, configuration and ICD references;
- optional owner-native change-impact receipt reference, event/evidence frontier references, completeness state, affected claim set and unaffected claim set; a known change with an incomplete event frontier forces abstention until reconciled;
- exact list of affected claims the proposed requalification action resolves; it must cover the full required requalification set, not only the easiest subset;
- unresolved claim and why its resolution could change the decision;
- action class and exact method/model/procedure revision;
- applicability and admissibility evidence references;
- prerequisite state and required human/safety approvals;
- expected outputs, units, frame, coverage and independent verifier;
- declared probability model/observation likelihood reference **or** uncertainty-set reference, never an ambiguous generic confidence;
- resource vector and remaining-budget snapshot;
- predicted decision-value output with regime, assumptions and limitations;
- claim ceiling and no-actuation marker.

Execution/result is a separate record:

`EvidenceActionOutcomeV1`

- proposal reference and exact input/configuration identity;
- state: planned, admitted, executed, completed, failed, aborted, invalid, censored, out-of-domain, not-run or disputed;
- actual resources and timestamps;
- exact raw output references/digests and evidence lineage;
- model/procedure/instrument/facility identities and currentness;
- measured/modelled outcome and uncertainty semantics;
- independent verifier and disposition;
- discrepancy against prior predictions;
- explicit non-promotion rule (simulation is not physical test; provenance is not truth).

The adapter never starts the action or issues a qualification release.

The machine-readable contract is [action-proposal-outcome.schema.json](../../fixtures/orbital-joint-evidence-voi/v0.1/action-proposal-outcome.schema.json). Rust typed fixtures are [a non-actuating proposal](../../fixtures/orbital-joint-evidence-voi/v0.1/action-proposal.synthetic.json) and [an aborted outcome with actual costs](../../fixtures/orbital-joint-evidence-voi/v0.1/action-outcome-aborted.synthetic.json). A proposal is always `authority_granted=false` and `no_actuation=true`. An attempted outcome must carry actual resource consumption and an external executing-owner/authority reference; an aborted/failed/indeterminate result cannot be promoted to qualification.

## 7. Bounded benchmark v0.1

The first synthetic benchmark should use the 44-case structural-interface compatibility corpus as an immutable decision input. Do not duplicate its evaluator or revise expected compatibility dispositions. Add a separate action-selection layer and independent action-value oracle.

Scenario families for the first tournament:

1. **Decision-irrelevant uncertainty:** one parameter has high predictive variance but does not change feasibility or the preferred robust candidate.
2. **Narrow decision boundary:** one bounded parameter can reverse compatibility; one calibrated measurement has potential decision value.
3. **Cheap-model bias:** a low-cost model is systematically optimistic near a hard feasibility boundary.
4. **Coupled-load interaction:** scalar component measurements cannot resolve an unsupported combined-load interaction model.
5. **No-information control:** every available action is inadmissible or cannot change the declared decision; the correct result is abstention.
6. **Probability/interval confusion:** a policy attempts to score hard intervals as if they were calibrated probability distributions.
7. **Common-cause duplication:** two nominally different tests share the same calibration/reference/analysis dependency and cannot be counted as independent evidence.
8. **Stale configuration:** an attractive action targets an old geometry, material lot, procedure or ICD revision.
9. **Calibration/facility block:** an inspection/test would be informative in principle but exact calibration or facility asset capability is invalid.
10. **Scale-transfer gap:** coupon evidence cannot decide a component/joint/assembled-load-path claim without a supported transfer assessment.
11. **Campaign adequacy:** a high-information action does not cover a mandatory or pre-registered critical interaction from #6540.
12. **Lifecycle gap:** first-mate data cannot resolve wear, repair or requalification questions from #6484.
13. **Cost cliff:** a proposal fits compute budget but exceeds the specimen/facility/reviewer budget, or omits follow-on validation.
14. **Failed/censored action:** action fails or aborts; full incurred cost and missing evidence remain.
15. **Preference/requirement change:** new decision preferences alter selection but historical evidence, full Pareto frontier and raw measurements are retained.

### Baselines

All policies receive identical admissible actions, initial evidence, decision context and total budget:

- fixed canonical sequence;
- one-factor-at-a-time / cycle-count shortcut;
- space-filling / seeded random;
- current Symthaea prediction-disagreement selector (as a baseline, not renamed as EVSI);
- highest predictive uncertainty;
- expected decision-value policy when calibrated probabilities exist;
- distribution-free robust-feasibility / minimax-regret policy;
- human engineer-selected next action, with time and rationale recorded.

A separate oracle grades actions after the policy commits. The policy must not import or query the evaluator-only hidden truth. Use paired seeds and held-out scenario families.

### Scorecard

Report, separately:

- false-feasible and missed-feasible rate;
- action admissibility error and stale/incorrect identity rejection;
- expected decision regret or worst-case regret (never blended);
- decision-change yield and correct abstention;
- mandatory/critical test-obligation coverage;
- common-cause and independence handling;
- total resource use by resource dimension;
- failed/aborted/censored/invalid action accounting;
- probability calibration where applicable;
- held-out scenario performance and policy overhead;
- engineering/reviewer time to a defensible decision.

## 8. Proposed gates, not achieved results

- 100% attempted, rejected, failed, aborted, censored, invalid and abstained action accounting.
- Zero hidden-oracle leakage in the benchmark protocol.
- Zero false-feasible decisions on frozen critical negatives.
- 100% detection of deliberately stale/wrong article, configuration, procedure and calibration cases.
- 100% expected-output agreement with an independently implemented oracle.
- No fabricated benefit on the no-information control.
- Demonstrable decision-regret improvement over random and pure uncertainty/disagreement selection at matched total budget on preregistered held-out decision-relevant cases.
- Prospective target: at least 20% lower total evidence-acquisition cost or review effort at equal or stronger false-feasible rate and critical-coverage performance. This remains a hypothesis until measured on a frozen, blinded benchmark.

No single aggregate score may hide a critical miss, false acceptance, cost overrun or inadmissible action.

## 9. Industry and SOTA positioning

This adapter should not try to replace specialist optimizers, model-based systems engineering, CAE solvers, test management or PLM.

- OpenMDAO and Dymos are useful multidisciplinary/trajectory optimization comparators; compare under matched model fidelity, constraints and resource budgets.
- Bayesian optimization / multi-fidelity methods are relevant only where probability and surrogate assumptions are declared; do not present a hard-bound-only problem as Bayesian.
- Space-filling and DOE methods are strong coverage baselines but do not necessarily target decision regret.
- Highest uncertainty and current prediction-disagreement are meaningful ablations; the test is whether their selected experiment changes the decision enough to justify total cost.
- Human domain engineers remain a necessary comparator and review authority.
- Industry systems (Siemens, Dassault, Ansys, Hexagon) retain their mature specialist workflows; compare cross-tool evidence continuity, action traceability, applicability and review effort rather than unsupported feature superiority.

SOTA claims require the same held-out problems, action catalogue, physics models, evidence availability, policy overhead and total resource budget. Report uncertainty and effect sizes, not just the best single seed.


### 9.1 Standards and product baseline as of 21 November 2026

The design must now be mapped against the published 2026 standards, not just earlier digital-twin concepts:

- **ISO 23247-5:2026** defines digital-thread use to connect manufacturing lifecycle data to digital twins across design, planning, production and testing. It was published in June 2026.
- **ISO 23247-6:2026** defines digital-twin composition across vendors and systems, including integrated, unified and federated composition approaches; it was published in July 2026.
- **CAx-IF** provides AP242 recommended practices, persistent-ID guidance (v1.8, March 2026), and published STEP/QIF trace-information Test Cases 7 and 8. These are important independent reference fixtures.
- **NIST STEP File Analyzer and Viewer 5.41** is a reference analysis tool for STEP AP242 and semantic/graphic PMI; use it and CAx-IF material as external conformance comparators rather than writing a second STEP parser.

Primary references: [ISO 23247-5:2026](https://www.iso.org/standard/87425.html), [ISO 23247-6:2026](https://www.iso.org/standard/87426.html), [CAx-IF recommended practices and sample files](https://www.mbx-if.org/home/cax/recpractices/), [NIST STEP File Analyzer and Viewer](https://www.nist.gov/services-resources/software/step-file-analyzer-and-viewer).

NIST specifically highlights persistent identifiers, semantic PMI, feature associations, hybrid geometry and secure data-asset traceability as areas where digital-thread capabilities still need work. It notes that checking STEP syntax and QIF well-formedness does not, by itself, check the engineering semantics embedded in those files. That supports a sharply scoped Luminous assurance layer: compare semantic assertions and identity/evidence joins across independently generated outputs, while consuming specialist parsers and metrology tools.

### 9.2 Commercial tools: integrate rather than clone

The competitive baseline is already stronger than a simple CAD-to-QIF bridge:

- **Hexagon PC-DMIS 2026.2** publicly describes STEP AP242 semantic-PMI import into QIF-based measurement planning, UUID mapping, AI-assisted edge detection, advanced point-cloud processing, and tighter Teamcenter traceability. That is the immediate shop-floor interoperability target.
- **Siemens Teamcenter Simulation/SPDM** manages simulation and physical-test processes/data, multi-discipline traceability and integrated workflows.
- **Dassault Systèmes 3DEXPERIENCE** is deployed at Airbus across future civil/military aircraft and helicopter programmes, with more than 20,000 users in the announced scope.
- **Ansys/Minerva** positions simulation process/data management and model-based workflows as a digital-engineering ecosystem.

References: [Hexagon PC-DMIS 2026.2 release](https://hexagon.com/about/news-media/media-alerts/2026/pc-dmis), [Siemens Teamcenter SPDM](https://www.siemens.com/en-us/products/teamcenter/solutions/simulation-process-data-management-spdm/), [Airbus–Dassault partnership](https://www.3ds.com/newsroom/press-releases/dassault-systemes-and-airbus-extend-strategic-partnership-use-virtual-twins-next-generation-programs), [Ansys digital engineering](https://www.ansys.com/blog/defense-digital-engineering).

Do not claim these systems lack provenance, workflow control or evidence management. Their feature scope and commercial deployments are substantial. Luminous should operate as a vendor-neutral qualification/evidence and decision layer at the interfaces between them, preserving their native files and authoritative records.

### 9.3 Luminous composition strategy: federated by default

ISO 23247-6's composition framing supports a deliberate architecture choice:

- **Mycelix** remains the authority for article identity, signer/role authority, custody, provenance and correction/revocation state.
- **Symtropy** remains the authority for physics/structural/inspection/test calculations and qualification dispositions.
- **Symthaea** consumes those outputs to propose designs and next evidence actions, with explicit uncertainty/decision regimes.
- **Sol-Atlas** contributes spatial, facility, logistics and infrastructure context without converting a facility listing into an availability or qualification claim.

Start with **federated composition**: each system retains its native schema and authority; exchanged records carry stable references, revisions, subject/configuration scope, evidence digest and claim ceiling. Introduce unified views for decision workflows only after the exact reference joins are verified. Do not create one monolithic universal twin or let one system overwrite another system's source of truth.

This should be assessed against ISO 23247-5/-6 concepts, not described as standards-conformant without a clause-by-clause assessment. The architecture's differentiator is the non-compensatory evidence contract: a semantically valid link is not a physical pass; a signed record is not a verified measurement; a simulation is not a test; a coupon is not a production joint; and an unresolved claim stays unresolved.

### 9.4 Interoperability tournament and measurable targets

Use CAx-IF Test Cases 7/8 and the NIST SFA sample models as external reference inputs. Use PC-DMIS 2026.2 as a commercial workflow comparator where a rights-cleared evaluation environment or exported result is available. Compare at least:
1. AP242 semantic-PMI → QIF plan assertion preservation;
2. persistent UUID and requirement/feature/serial-unit join retention;
3. measurement-result and calibration/instrument trace retention;
4. semantic loss, ambiguity and unsupported-feature detection;
5. stale revision, wrong serial and wrong configuration rejection;
6. content/digest and custody/change history across exports;
7. time and human re-entry required to repair broken links;
8. evidence/action selection quality at matched resource budget.

Proposed benchmark targets, not achieved results:
- zero false joins across frozen wrong-subject/configuration/revision negatives;
- 100% preservation of required semantic assertions supported by the declared test-case subset;
- zero silent drops: every unsupported or lost assertion becomes an explicit loss/Unknown disposition;
- deterministic outputs across input ordering and repeated replay;
- 100% correct invalidation of evidence affected by stale/revoked identity or method revision;
- at least 20% reduction in human reconciliation/review time on held-out cases at equal or higher semantic coverage, measured against a pre-registered manual and vendor-workflow baseline.

Do not claim better semantic PMI conformance than Hexagon or NIST until the same models, assertions, versions and output evidence have been run independently. The stronger near-term claim is a reproducible evidence/decision layer across multiple tools and lifecycle owners, demonstrated with open fixtures.

## 10. Physical test and safety boundary

Any actual connector coupon, NDE, wear or ground test remains with Symtropy #6256/#5771 and the competent test/safety owners. Initial physical work, if approved, is non-flight, uncrewed and limited to benign, appropriately guarded test articles and rated fixtures. The issue supplies no pressure, force, energy or environmental limits. No propellants, cryogens, live spacecraft, flight-critical article or unsupervised machine actuation are introduced by this adapter.

Symthaea proposes and ranks actions only. Competent human/test authorities approve and execute. Mycelix records appropriate provenance/authority references; Sol-Atlas supplies context only. No AI or benchmark outcome issues a test authorization, production release, structural acceptance or airworthiness finding.

## 11. Deliverables

P0: this contract and owner map.  
P1: machine-readable scenario and proposal/outcome schemas, typed synthetic proposal/aborted-outcome examples, and 17 frozen synthetic scenarios.  
P2: executable dry-run admissibility audit and one-step selector that ranks admissible actions by decision-regret/EVSI improvement, then resolves equal-value options only through resource-vector Pareto dominance or requests human review; it also audits claim identity, common-cause labels, mandatory obligations and actual resource consumption including aborted-test overruns. A separately implemented standard-library Python oracle independently checks all 17 known-answer scenarios, including incomplete change-frontier abstention and complete-frontier targeted requalification. No solver or physical execution is invoked.  
P3: matched-budget policy tournament and policy evaluation against the independent oracle.  
P4: adapter to Symtropy compatibility/test-matrix/lifecycle receipts and Mycelix/Sol-Atlas references.  
P5: only after review, authorized human-supervised physical evidence handoff.

**Current milestone:** P0 specification, P1 schema/scenario catalogue and P2 dry-run admissibility/value audit with a deterministic one-step selector are implemented on a public research branch. P3 multi-step/matched-budget policy tournament and external adapter evidence remain future work. No selector-performance win over SOTA, physical test or certification claim is made. The 17-case audit includes owner-native change-frontier completeness and targeted claim requalification coverage.
