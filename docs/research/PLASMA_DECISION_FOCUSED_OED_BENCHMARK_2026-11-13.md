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

A 2024 PA6 study, DOI [10.1007/s44245-024-00072-5](https://doi.org/10.1007/s44245-024-00072-5), adds a hidden-state warning: dry Durethan B30S PA6 below 0.1 wt% water and material conditioned to 3 wt% water showed no significant difference in macroscopic contact angle/surface energy, yet the saturated state produced roughly half the lap-shear strength of dry material. The water content also changed failure behavior with both PUR and MMA adhesives. The authors therefore warn that contact angle cannot reveal this material-state effect.

A 2024 polyolefin primer study, DOI [10.1007/s11998-024-00938-z](https://doi.org/10.1007/s11998-024-00938-z), is a useful incumbent comparator rather than a universal anti-plasma conclusion: its specific primer retained adhesion after two months in some PP/PE profiles, while plasma-treated PE with additives partially lost adhesion after 30 days. The result depended on additive content and substrate profile. It demonstrates why a plasma process should be compared with the best applicable durable primer/activation baseline, not only untreated substrate.

These findings justify a multi-output, time-dependent, material-state-aware experiment contract. They do not constitute Luminous reproduction.

## 3. Measurement-fidelity ladder

| Level | Observation class | Typical role | Cannot establish alone |
|---|---|---|---|
| F0 | WCA / wetting tension / fast process screen | Cheap triage and process-state evidence | Bond acceptance or durability |
| F1 | XPS / FTIR / AFM / profilometry | Discriminate chemistry, topography and weak-boundary-layer hypotheses | Product bond acceptance |
| F2 | Destructive lap-shear / peel / pull-off | Actual mechanical endpoint and failure locus | Long-term retention |
| F3 | Delayed-bond and aged mechanical test | Bondability-window and environmental retention | Unmeasured substrate functions |
| F4 | WVTR / OTR / laminate residual properties | Preserve application-critical substrate function | Bond strength |

Facility-specific price, elapsed time, sample consumption, resolution and uncertainty must be measured before real use. The fixture's cost and latency units are artificial.

### Time-resolved and material-state measurements

For a material profile susceptible to hydrophobic recovery, the PE aging study gives a stronger initial sampling hypothesis than a sparse 1 h / 24 h / 7 d schedule: it reports that most of the WCA recovery occurred in the first 24 h, with a slower approach to a plateau around 3–6 days over an 18-day observation period. Importantly, that study's aging series follows WCA, not bond strength. Luminous must therefore measure paired WCA **and** destructive bond outcomes on randomized matched specimens at the selected delays.

A useful *initial DOE grid* for a production-buffer profile is: as soon as practicable after treatment, 1 h, 4 h, 24 h, 72 h, 6 d and 18 d. Add 30 d / 60 d where pretreatment-to-assembly storage is a realistic workflow. This is a proposed sampling grid, not a universal aging law; the material/adhesive/storage profile and variance pilot should determine final points and replicate counts.

For hygroscopic substrates such as PA6, the pre-plasma material state must be part of the evidence identity. The application adapter carries `sample_lineage_ref` and `material_interface_ref` that resolve through MAT-013 #4344 and MAT-INTERFACE-001 #6227; it does not define a second sample or interface registry: measured moisture content, conditioning protocol, temperature/RH, measurement method/calibration, sample identity and elapsed time from measurement to plasma. A nominal polymer grade plus contact angle is insufficient. For additive-bearing PE/PP, additive package/grade/lot also belongs in the material-state identity.

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
- MAT-013 #4344 owns physical sample/batch lineage, conditioning and provenance-complete characterization/test links.
- MAT-INTERFACE-001 #6227 owns material-interface identity, surface preparation/state evolution and moisture/chemical exposure semantics.

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

The machine-readable fixture now contains twelve known-answer cases:
1. proxy-only plan: low WCA but no bond/aging evidence -> ABSTAIN;
2. contact-angle trap: low WCA but bond fails -> REJECT_WINDOW;
3. aging reversal: initial bond passes but aged bond fails -> REJECT_WINDOW;
4. packaging barrier trade-off: bond passes but WVTR/OTR preservation fails -> REJECT_WINDOW;
5. valid multifidelity evidence across independent training holdouts -> ACCEPT_WINDOW (synthetic only);
6. shared sample-preparation signal without admitted covariance model -> ABSTAIN;
7. training/holdout group overlap -> REJECTED proposal / ABSTAIN;
8. over-budget action portfolio -> REJECTED proposal / ABSTAIN;
9. inadmissible experiment action -> REJECTED proposal / ABSTAIN;
10. inadequate model family despite otherwise passing outcomes -> BLOCKED / ABSTAIN;
11. PA6 missing pre-treatment moisture state -> BLOCKED / ABSTAIN;
12. PA6 saturated-state, low-WCA / weak-bond challenge with measured moisture state -> REJECT_WINDOW.

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



### Material-state-aware S0 extension

The action catalogue now includes `MOISTURE_CONTENT` (F1) as an optional, profile-required measurement. Profiles that require it freeze a maximum measurement age; the scenario must bind moisture content to a selected action measurement, calibration identity, conditioning protocol, receipt, temperature/RH, timestamp and evidence digest. Missing or stale conditioning evidence causes BLOCKED / ABSTAIN. A measured moisture value must reconcile with the selected measurement's recorded value.

The synthetic PA6 cases use 0.08 wt% and 3.0 wt% as challenge inputs inspired by the dry / saturated states in the cited PA6 study. They are **not** a reproduction of its measured strength values. The missing-state case demonstrates fail-closed behavior. The saturated-state case has a low synthetic WCA but weak initial/aged bond outcomes and therefore REJECT_WINDOW.

The 2024 polyolefin-primer study should become a mandatory comparator family in the eventual physical DOE: untreated control, best applicable incumbent (including primer where appropriate), plasma-only, primer-only, and plasma-plus-promoter where chemically / operationally admissible. The 2026 acrylate paper likewise warns that plasma alone may not achieve long-term water resistance. Luminous must demonstrate value over the best applicable baseline, not only over untreated material.

## 13. S0 implementation and source-side replay — 13 November 2026

Created [PLASMA-OED-001 #11907](https://github.com/Luminous-Dynamics/symthaea/issues/11907), branch `research/plasma-decision-oed-v0-1`, and the following files:
- `docs/research/PLASMA_DECISION_FOCUSED_OED_BENCHMARK_2026-11-13.md`;
- `data/research/plasma_decision_oed_fixture_v0_1.json`;
- `xtask/src/plasma_decision_oed.rs`;
- `xtask/src/main.rs` command: `plasma-decision-oed-audit`.

The nine-action catalogue represents F0–F4 measurement classes, including a material-specific MOISTURE_CONTENT observation, artificial costs/latencies/sample counts, endpoint coverage and admissibility. The synthetic selection receipt points to UAP-OED-001; it is an input to the domain evaluator, **not an executed optimizer result**.

The twelve scenarios freeze:
- 5 audit PASS (four product rejections and one synthetic accept);
- 4 BLOCKED / ABSTAIN cases;
- 3 REJECTED proposal / ABSTAIN cases.

Independent source-side JavaScript replay of the exact fetched JSON matched **12/12** audit dispositions and product decisions. In addition to action catalogue membership, cost/latency/specimen closure, calibration and raw-trace identity, lot/device/day group disjointness, covariance receipts, mandatory endpoints, model adequacy / OTHER, and bond/aging/barrier floors, it checks that a hygroscopic material's measured moisture content is tied to a selected moisture-observation action, calibration receipt and timestamp that is fresh at plasma-run time.

Nineteen Rust unit tests are authored.

The PA6 substrate-condition adapter now requires exact MAT-013 sample-lineage and MAT-INTERFACE subject references in addition to the condition protocol, moisture measurement receipt/calibration, environment, timestamp and digest. Synthetic refs in the fixture are placeholders only; physical promotion must resolve the canonical sample and interface subjects. A missing canonical reference is BLOCKED / ABSTAIN.
 The authoring environment has no Cargo, rustc or rustfmt, so this module has not been compiled, formatted, linted or executed. No hosted CI result is claimed.


The final evaluator hardening also:
- binds every measurement's quantity and unit to its selected action definition;
- requires a calibration-record identity, qualified status, uncertainty and raw-trace digest;
- recomputes bond retention as aged bond strength divided by initial bond strength and rejects a mismatched declared ratio;
- requires an explicit model-adequacy receipt, an ADEQUATE status and registration of an OTHER hypothesis before any ACCEPT/REJECT product decision;
- checks common sample-preparation and independence groups against the selected-set covariance receipt.

The evaluator has 15 authored Rust tests, including mutations for unit relabelling and inconsistent retention. These tests remain uncompiled and unexecuted in this environment.

The fixture is synthetic-only; it does not establish a useful measurement schedule, a real EVSI ranking, physical bond performance, lab savings, a process window or experiment authority. The next integration should consume a real UAP-OED selection result and its exact source/covariance/model-adequacy receipts without copying the generic selector.
