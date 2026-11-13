# PLASMA-OED-001 — Decision-focused multifidelity experiment design for durable plasma bonding

**Date:** 13 November 2026  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11907  
**Parents:** PLASMA-SURFACE-001 #11795, PLASMA-VM-001 #11847, PLASMA-ML-GENERALIZATION-001 #11839, MICROPLASMA-001 #11780  
**Generic owners reused:** UAP-OED-001 #11050, UAP-OED-002 #11235, UAP-OED-003 #11237, OLI-EXPERIMENT-001 #11446  
**Status:** synthetic-only application profile and contract evaluator  
**Claim ceiling:** no physical process optimization, no trained model, no measured bond improvement, no laboratory saving, no execution authority.

## 1. Research objective

Select the smallest defensible set of observations that resolves whether one exact material/adhesive/plasma profile meets its downstream product requirements.

The selection objective is **cost to a robust product decision**, not lowest water contact angle, maximum wettability, maximum nominal information gain or generic prediction accuracy.

The possible decisions are:
- `ACCEPT_WINDOW`: all mandatory physical outcomes meet the frozen profile across the required strata;
- `REJECT_WINDOW`: one or more mandatory product floors / preservation constraints fail;
- `ABSTAIN`: evidence, calibration, model adequacy, domain coverage or nuisance/covariance handling is insufficient.

In this S0 fixture all numeric values and observations are synthetic; an evaluator PASS is not a product decision backed by a real specimen.

## 2. Why single-metric optimization is unsafe

Péczi-Kovács & Kókai (2026-10-20), DOI [10.1007/s44245-026-00396-4](https://doi.org/10.1007/s44245-026-00396-4), compare argon and nitrogen across PP, HIPS, PVC, PC and PET under identical processing. The paper reports that the lowest water contact angle does not consistently give the highest bond strength; prolonged nitrogen exposure visibly degraded some polymers, while argon had a wider damage-free range in the tested conditions.

The 2026 LDPE/HDPE study, DOI [10.1002/app.70533](https://doi.org/10.1002/app.70533), reports improved wettability together with increases up to 60% in water-vapour permeability and 45% in oxygen permeability. This is a direct warning that a treatment can improve interfacial wetting while degrading the material's packaging function.

The 2024 review, DOI [10.3390/ma17071494](https://doi.org/10.3390/ma17071494), explains that adhesion may diverge from wettability because of hydrophobic recovery and weakly bonded low-molecular-weight oxidized material.

The June 2026 acrylate-joint paper, DOI [10.1016/j.porgcoat.2026.110101](https://doi.org/10.1016/j.porgcoat.2026.110101), reports that plasma alone was insufficient for long-term water resistance in its studied system, while plasma plus adhesion promoter improved retained strength.

These findings justify a multi-output, time-dependent, material-specific experiment contract. They do not constitute Luminous reproduction.

## 3. Measurement-fidelity ladder

| Level | Observation class | Typical role | Cannot establish alone |
|---|---|---|---|
| F0 | WCA / wetting tension / fast process screen | Cheap triage and process-state evidence | Bond acceptance or durability |
| F1 | XPS / FTIR / AFM / profilometry | Discriminate chemistry, topography and weak-boundary-layer hypotheses | Product bond acceptance |
| F2 | Destructive lap-shear / peel / pull-off | Actual mechanical endpoint and failure locus | Long-term retention |
| F3 | Delayed-bond and aged mechanical test | Bondability-window and environmental retention | Unmeasured substrate functions |
| F4 | WVTR / OTR / laminate residual properties | Preserve application-critical substrate function | Bond strength |

Facility-specific price, elapsed time, sample consumption, resolution and uncertainty must be measured before real use. The fixture's cost and latency units are artificial.

## 4. Frozen rival mechanism families

For each exact material/adhesive profile, the model register should include:

- M1 durable functionalization: stable chemistry contributes to the bond.
- M2 hydrophobic recovery: apparent initial activation decays before bonding.
- M3 weak boundary layer: loosely bonded oxidized fragments create misleading wetting.
- M4 etching/damage trade-off: roughening or fibre exposure improves mechanical interlocking but may damage substrate function.
- M5 adhesive/primer interaction: material and adhesive compatibility changes treatment ranking.
- M6 shared nuisance: preparation, witness location, lot, operator or calibration explains an apparent treatment effect.
- OTHER: the registered family is inadequate; do not force a closed-set winner.

These are rival hypotheses, not universal claims about every polymer.

## 5. Domain contract and ownership

This benchmark is a thin plasma-domain profile:
- UAP-OED-001 #11050 owns cost/fidelity action selection, deterministic ranking and exact small-set oracle.
- UAP-OED-002 #11235 owns target-marginal utility after nuisance parameters.
- UAP-OED-003 #11237 owns selected-set correlated residual covariance and whitening.
- OLI-EXPERIMENT-001 #11446 owns model adequacy, OTHER/abstention and robust decision-value semantics.
- PLASMA-ML-GENERALIZATION-001 #11839 owns group splits and preprocessing lineage.
- PLASMA-VM-001 #11847 owns calibrated telemetry/run/specimen/outcome linkage.
- PLASMA-SURFACE-001 #11795 owns physical acceptance thresholds and the bondability endpoint.

No duplicate generic optimizer, covariance engine, calibration registry, evidence receipt or materials ontology is introduced.

## 6. Decision utility, not information for its own sake

The generic OED engine may report nominal/robust information gain and exact-oracle selection. The plasma adapter must additionally evaluate whether the selected evidence can support the actual product decision.

The decision profile freezes:
- false acceptance loss;
- false rejection loss;
- abstention / additional-evidence cost;
- exact bond and aged-bond floors;
- minimum retention;
- acceptable failure modes;
- optional material-function bounds;
- mandatory measurement endpoints;
- allowed material/adhesive/domain scope.

Decision-focused EVSI is a future integration objective, not a capability claimed by this S0 contract. A generic information score does not override mandatory product evidence or model-adequacy gates.

## 7. Nuisance, covariance and selection bias

Two measurements with different IDs are not necessarily independent. The same sample preparation, witness location, lot, operator, instrument calibration or upstream signal can induce correlated errors.

Any shared preparation/independence group must either:
1. carry a covariance-model reference that is admitted by UAP-OED-003; or
2. remain ineligible for an independence-based information calculation and force ABSTAIN / BLOCKED.

Cheap screens can be used to select specimens for expensive F2/F3 tests. The selection policy and missingness mechanism must be recorded. Otherwise, selected-for-verification samples are not representative of the production population, and apparent calibration/accuracy is biased.

Use whole material-lot, device/configuration and production-day holdouts from PLASMA-ML-GENERALIZATION-001. Do not split rows or adjacent witness spots randomly.

## 8. S0 fixture scenarios

The machine-readable fixture contains eight known-answer cases:
1. proxy-only plan: low WCA but no bond/aging evidence -> ABSTAIN;
2. contact-angle trap: low WCA but bond fails -> REJECT_WINDOW;
3. aging reversal: initial bond passes but aged bond fails -> REJECT_WINDOW;
4. packaging barrier trade-off: bond passes but WVTR/OTR preservation fails -> REJECT_WINDOW;
5. valid multifidelity evidence across independent training holdouts -> ACCEPT_WINDOW (synthetic only);
6. shared sample-preparation signal without admitted covariance model -> ABSTAIN;
7. training/holdout group overlap -> BLOCKED / ABSTAIN;
8. over-budget action portfolio -> REJECTED proposal / ABSTAIN.

The evaluator checks action identity, admissibility, costs, latency, sample budget, measurement provenance, calibration state, holdout disjointness, covariance references, mandatory endpoint coverage and decision gates. It does **not** rank candidate actions or claim an optimizer win; those are owned by UAP-OED.

## 9. Baselines and competitive benchmark

At matched budget and service profile, compare:
1. fixed conventional measurement campaign;
2. random feasible sequence;
3. contact-angle-first heuristic;
4. nominal cost-aware EIG / Fisher design;
5. multifidelity information design;
6. target-marginal design with nuisance parameters;
7. covariance-aware selected-set design;
8. robust decision-value / EVSI design;
9. exact feasible-subset oracle on small action pools.

Report separately:
- false ACCEPT_WINDOW and false REJECT_WINDOW;
- abstention coverage-risk curve;
- expected and worst-group decision regret;
- cost, latency, sample consumption and energy to decision;
- destructive/aging tests per qualified decision;
- prediction/decision calibration by lot, device and day;
- exact-oracle optimality gap;
- sensitivity to covariance and model-family inadequacy.

No single aggregate score may hide escaped defects, material damage or unsupported product decisions.

## 10. Proposed research targets

After a pilot establishes the benchmark difficulty:
- zero false ACCEPT_WINDOW decisions on public known-truth negative controls;
- at least 95% detection of deliberately out-of-domain / inadequate-model challenges at a published in-domain coverage;
- cost-to-decision within 10% of the exact feasible-subset oracle on at least 90% of eligible small cases;
- at least 25% fewer expensive destructive/aging tests than a fixed campaign, without increased false acceptance or relaxed physical evidence requirements;
- 100% of proposals carry cost, latency, sample-use, admissibility, dependency and covariance identity;
- deterministic replay of the decision and selected-action receipt.

These are research targets, not current results or standards.

## 11. Promotion to physical work

A later physical protocol requires exact material and adhesive identity, calibrated/traceable instruments, preregistered randomization and replicate sizing, matched controls, destructive bond ground truth, held-out lots/days/devices, selected aging conditions, preservation testing where relevant, complete utility boundaries and facility/operator safety authorization.

No synthetic outcome, model prediction, contact angle, plasma-state alarm or virtual measurement can release product or authorize HV/RF/gas actuation.

## 12. Claim ceiling

A software PASS validates only the synthetic application contract and its fail-closed dispositions. It does not establish a physical mechanism, process window, stronger bond, longer service life, reduced laboratory cost, industrial savings, model performance or hardware qualification.
