# Plasma decision-focused UAP-OED integration V0.1

**Date:** 13 November 2026  
**Plasma application:** [PLASMA-OED-001 #11907](https://github.com/Luminous-Dynamics/symthaea/issues/11907)  
**Generic selector owners:** UAP-OED-001 #11050, UAP-OED-002 #11235, UAP-OED-003 #11237  
**Implementation dependency:** [UAP Observatory draft PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)  
**Branch:** `research/plasma-uap-oed-integration-v0-1`  
**Status:** synthetic integration example; not compiled or executed

## 1. Purpose and ownership

This example composes the existing `symthaea-uap-observatory` experiment-design API with the PA6 durable-bonding profile. It does not add an optimizer, information-gain engine, covariance engine or exact-subset oracle.

It calls the existing functions directly:
- `greedy_budgeted_design`;
- `exhaustive_budgeted_design`;
- `greedy_correlated_budgeted_design`;
- `exhaustive_correlated_budgeted_design`.

The two paths share the same candidate actions, costs, priors, target dimensions and budget. Their only material difference is whether selected-set residual covariance is modelled.

## 2. Frozen synthetic profile

Target state:
`initial bond strength, aged-bond state, substrate moisture`

Nuisance state:
`shared sample-preparation offset`

Candidate actions:

| Action | Fidelity | Cost units | Product endpoint |
|---|---:|---:|---|
| WCA_SCREEN | F0 | 1 | Wettability proxy |
| FTIR_CHEMISTRY | F1 | 3 | Surface chemistry proxy |
| MOISTURE_CONTENT | F1 | 2 | Hygroscopic substrate state |
| LAP_SHEAR | F2 | 8 | Initial bond strength |
| AGED_LAP_SHEAR | F3 | 14 | Aged bond / retention |

The experiment budget is 26 artificial cost units. Two synthetic strata (dry and conditioned/hygroscopic) have equal scenario weight.

The residual covariance matrix contains:
- WCA / FTIR common preparation correlation: (ho=0.95);
- initial / aged lap-shear correlation: (ho=0.30);
- unit variance on the diagonal.

The covariance is positive definite (eigenvalues 0.05, 0.70, 1.00, 1.30 and 1.95). These are synthetic challenge parameters, not empirically estimated laboratory covariance.

## 3. Why this integration matters

The independent-row greedy path treats every candidate's noise as independent. Under this constructed profile, it spends budget on a second, highly correlated surface proxy and omits the destructive initial lap-shear endpoint.

The selected-set correlated path discounts redundant information from the shared preparation block and selects the initial plus aged mechanical tests alongside the moisture measurement.

This is an application-specific example of a general principle:

> Distinct measurement IDs do not imply independent information. A cheaper proxy can look valuable when common-mode residual covariance is ignored, while an independent destructive endpoint may be the measurement that actually resolves the product decision.

## 4. Expected source-side linear-algebra result

The UAP source implements target-marginal information gain as the difference between posterior and prior target-marginal log determinants after nuisance marginalization. An independent matrix calculation using that same objective yields the following expected known-answer values:

| Selector | Selected actions | Cost | Target EIG (nats) | PA6 mandatory endpoints |
|---|---|---:|---:|---|
| Independent greedy | WCA, MOISTURE, FTIR, AGED_LAP | 20 | 5.279431608 | Missing LAP_SHEAR |
| Independent exhaustive oracle | WCA, MOISTURE, LAP, AGED_LAP | 25 | 5.618701237 | Complete |
| Correlated greedy | WCA, MOISTURE, LAP, AGED_LAP | 25 | 5.705573945 | Complete |
| Correlated exhaustive oracle | WCA, MOISTURE, LAP, AGED_LAP | 25 | 5.705573945 | Complete |

Interpretation:
- independent greedy has an approximately 0.339269630-nat gap to its own small-set oracle;
- correlated greedy matches its small-set oracle for this fixture;
- correlated selection is not permitted to accept a product. The example only checks that its *proposed portfolio* covers the frozen mandatory endpoint set.

The example enforces:
- independent greedy -> `ABSTAIN_MANDATORY_ENDPOINTS_MISSING`;
- correlated greedy -> `PORTFOLIO_READY_FOR_PHYSICAL_REVIEW`.

"Ready for physical review" is not product acceptance, physical execution authorization, or a claim that the selected measurements have been performed.

These numbers are expectations derived from an independent source-side matrix calculation, not Rust program output. The Rust example contains known-answer assertions so the result can be checked when a supported Rust toolchain is available.

### 4.1 Concrete selector composition and exact known-answer replay

The Rust example is registered as `plasma_multifidelity_design` in the UAP Observatory crate manifest. It calls both the independent-row and correlated selected-set UAP APIs plus their exhaustive small-set oracles. It also emits the ownership chain, application-profile ID, policy ID, scenario/candidate/covariance manifest IDs, Git blob IDs for the exact generic API files, standard-family identifiers and an explicit no-authority claim.

Independent dense-matrix replay of the exact hard-coded arrays gives:

| Selector | Selected action IDs | Cost | Target EIG (nats) | Mandatory F2/F3/moisture coverage |
|---|---|---:|---:|---|
| Independent greedy | WCA_SCREEN; MOISTURE_CONTENT; FTIR_CHEMISTRY; AGED_LAP_SHEAR | 20 | 5.279431608 | Missing LAP_SHEAR |
| Independent exhaustive | WCA_SCREEN; MOISTURE_CONTENT; LAP_SHEAR; AGED_LAP_SHEAR | 25 | 5.618701237 | Complete |
| Correlated greedy | WCA_SCREEN; MOISTURE_CONTENT; LAP_SHEAR; AGED_LAP_SHEAR | 25 | 5.705573945 | Complete |
| Correlated exhaustive | WCA_SCREEN; MOISTURE_CONTENT; LAP_SHEAR; AGED_LAP_SHEAR | 25 | 5.705573945 | Complete |

The independently computed covariance eigenvalues are 0.05, 0.70, 1.00, 1.30 and 1.95, so the frozen covariance is positive definite. The high WCA/FTIR preparation correlation (ρ=0.95) makes the two proxies partly redundant. The correlated selector therefore does not spend its remaining budget on FTIR; it selects the missing initial lap-shear endpoint instead.

The independent greedy selector's gap to its own exact oracle is 0.339269630 nats. The correlated greedy selector matches its exact oracle for this specific five-action fixture. This is a *known-answer synthetic composition result*, not an industry benchmark win. A different covariance/prior/scenario can reverse the comparison.

The example asserts these selected IDs, exact-oracle relationships and endpoint-coverage dispositions. It fails loudly if the frozen source code produces a different result when it is eventually run with the parent UAP implementation.

The output IDs identify the exact generic source blobs used:
- budgeted design: `ed3d0286e706eaee9bb670e7e5e62f8ff01894b8`;
- correlated design: `4aebc5e80afe3b2258db04b441cda072c4edde1d`;
- UAP crate API: `aafaffcbc7d9aa1cef462ace453d2d173136b8d8`.

The candidate rows, priors, covariance and costs are embedded in the versioned Rust example. The Git blob ID of the complete example itself is tracked by the branch/PR, not self-reported by the example. No optimizer output, model-adequacy receipt, empirical covariance or physical evidence is fabricated.


## 5. Reproduce when the toolchain is available

From the Symthaea repository root on this branch:

```sh
cargo run --manifest-path crates/domains/symthaea-uap-observatory/Cargo.toml --example plasma_multifidelity_design
```

The UAP Observatory source branch is itself an open draft PR dependency. This integration branch is based on `research/causal-temporal-physics-v0.1`; it is intentionally stacked on that dependency and must not be treated as independently mergeable into main before the parent implementation is reconciled.

## 6. What the current UAP API provides—and what the plasma adapter still needs

The generic `BudgetedDesignReport` already provides:
- selected option IDs and total cost;
- budget and remaining budget;
- expected and worst-case target information gain;
- expected and worst-case joint-state information gain;
- per-scenario scores;
- selection steps and marginal gain per cost.

The current generic report does not, by itself, constitute a full provenance receipt for downstream engineering decisions. A production adapter still needs to bind the exact report to:
- the exact scenario/prior and candidate-option manifest digests;
- the selected-set covariance source and digest;
- action catalogue revision and cost/latency/sample semantics;
- model-adequacy / OTHER disposition from OLI;
- the exact UAP source revision and algorithm configuration;
- the plasma material profile and mandatory endpoint policy.

This should be a thin, versioned evidence envelope around the generic report—not a second optimizer or a competing generic receipt authority. Physical observations, calibration and material-state lineage continue to come from MAT-013 / MAT-INTERFACE and the existing observation/metrology owners.

## 7. Current research frontier and matched baselines

### 7.1 Goal-oriented design is now a direct SOTA comparator

GoBOED (Go et al.), *Goal-driven Bayesian Optimal Experimental Design for Robust Decision-Making Under Model Uncertainty* (arXiv preprint, 25 May 2026; arXiv:2605.26093), couples a Bayesian posterior surrogate to a robust downstream decision layer. Its central lesson is that uncertainty in parameter directions irrelevant to the decision can waste experiment budget. It evaluates source localization, epidemic control and pharmacokinetic control; it is not a plasma/materials benchmark.

A second current comparator is the additive robust Bayesian likelihood planner in [REA PR #10105](https://github.com/Luminous-Dynamics/symthaea/pull/10105) / [REA-RUST-013 #10101](https://github.com/Luminous-Dynamics/symthaea/issues/10101). It supports expected, worst-case and lower-tail CVaR EIG over caller-supplied finite likelihood scenarios, top-1/top-3 ranking stability and explicit unstable-ranking abstention. Its claim ceiling is that the scenarios and likelihoods are caller-supplied and not authenticated/calibrated by the planner.

This is complementary to, not interchangeable with, the UAP Gaussian-Jacobian API:
- use UAP-OED for continuous target/nuisance sensitivity, geometry, target-marginal EIG and selected-set Gaussian residual covariance;
- use the REA robust likelihood API for finite predicted-outcome distributions of competing surface-mechanism hypotheses where such likelihood rows are justified;
- use OLI/CTGP for adequacy, discrepancy and OTHER/open-world challenges;
- use the plasma application adapter for physical endpoint coverage, material state, cost/sample budgets and abstention.

Do not collapse EIG nats from different observation likelihoods into one score or treat the finite REA scenario set as a universal uncertainty guarantee.

The fair plasma comparison is therefore:
- nominal target-marginal EIG using UAP-OED-002;
- selected-set covariance-aware EIG using UAP-OED-003;
- robust scenario and open-world model-adequacy controls using CTGP-ROBUST-001 #11262 / OLI #11446;
- goal/decision-focused EVSI using the reusable decision-aware method family, including REA-RUST-014 #10109, where its utility/likelihood contracts are suitable;
- expert/standard-mandated testing and fixed DOE controls.

Do not claim novelty or superiority for merely replacing total EIG with a decision score. The research contribution must be the *material-specific, correlated, multifidelity and open-world qualification problem*, with matched baselines and outcome-level regret.

### 7.2 Model discrepancy must be an experimental target, not just a final warning

Yang, Dong & Wu, *Bayesian experimental design for model discrepancy calibration: An auto-differentiable ensemble Kalman inversion approach*, Journal of Computational Physics 545 (15 January 2026), DOI 10.1016/j.jcp.2025.114469, treats discrepancy as a core obstacle to trustworthy Bayesian design.

The plasma programme should distinguish:
- parameter uncertainty within a registered mechanism;
- residual/discrepancy uncertainty after conditioning on parameters;
- model-family disagreement;
- domain/lot/adhesive/conditioning shift;
- unknown outcomes outside every registered predictive support.

A candidate that distinguishes M1 from M2 is not valuable if M1 and M2 are both wrong in the held-out adhesive, moisture or storage regime. Reuse CTGP-ROBUST-001 #11262 and the open-world adequacy owner #11181. Keep explicit OTHER and a prospective negative-control panel; do not invent an unknown-model prior or silently renormalize the named mechanisms.

### 7.3 Industry baseline and standards interoperability

The following ASTM catalogue references are useful *method-family anchors* for a laboratory to acquire and apply under its own licensed standards and quality system. This specification does not reproduce normative clauses or claim compliance:

| Standard identifier | Published method family | How the physical campaign should use the reference |
|---|---|---|
| ASTM D6105-04(2019) | Electrical-discharge surface activation of plastics for adhesive bonding | Identify the applicable discharge technology and process classification; use the applicable atmospheric-pressure gas-plasma/DBD section |
| ASTM D3163-01(2023) | Rigid-plastic single-lap shear | Comparative lap-shear path for rigid plastic adherends such as the PA6 fixture family |
| ASTM D5868-01(2023) | FRP lap-shear adhesion | Use a composite-specific route for fiber-reinforced plastic bonding; do not treat rigid-plastic results as composite qualification |
| ASTM D2093-03(2025) | Plastic adherend surface preparation | Freeze conventional preparation/control arms and document the exact preparation history |
| ASTM D1151-00(2022) | Moisture and temperature effects on adhesive bonds | Define applicable continuous humidity/temperature exposure profiles and retained-strength endpoints |
| ASTM D1183-03(2019) | Cyclic laboratory aging of adhesive bonds | Define applicable cyclic environmental-aging comparisons |
| ASTM D4896-01(2024) | Interpretation of single-lap-joint results | Preserve the boundary between comparative coupon strength and structural-joint design allowables |

Source catalogue links: [D6105](https://store.astm.org/d6105-04r19.html), [D3163](https://store.astm.org/d3163-01r23.html), [D5868](https://store.astm.org/d5868-01.html), [D2093](https://store.astm.org/standards/d2093), [D1151](https://store.astm.org/d1151-00.html), [D1183](https://store.astm.org/d1183-03r19.html), [D4896](https://store.astm.org/d4896-01r24.html).

The standard references are not an assertion that one standard covers every polymer, adhesive, plasma source or product geometry. A qualified lab must check the current licensed edition, applicability, specimen geometry, conditioning, test method and safety requirements. In particular, ASTM's catalogue cautions against interpreting a small single-lap coupon result as a design allowable for a different structural joint.

### 7.4 Matched competitive benchmark matrix

| Baseline family | What it represents | Expected strength | Blind spot the benchmark should expose |
|---|---|---|---|
| Fixed industrial recipe / process window | incumbent production control | simple, low-latency repeatability | brittle when lot, moisture, additive or adhesive shifts |
| Conventional factorial DOE / response surface | planned process-variable optimization | interpretable factor interactions | expensive when many material/adhesive/storage strata exist |
| Contact-angle-first heuristic | cheap surface screen | throughput and low specimen use | proxy/product endpoint disagreement |
| Nominal UAP-OED target EIG | current generic independent-row selector | principled local information gain | common-mode over-credit |
| UAP-OED selected-set covariance | existing correlated observation engine | corrects declared residual dependence | still conditional on supplied linear-Gaussian models |
| Goal-driven BOED / EVSI | downstream decision optimization | aligns acquisition with product consequences | depends on trustworthy likelihood, utilities and model coverage |
| Robust / discrepancy-aware design | finite model ambiguity and open-world challenges | guards against false confidence under known shifts | can over-abstain or become conservative if ambiguity is poorly specified |
| Best applicable primer / incumbent | industrial alternative to plasma-only treatment | may offer superior durability in some material profiles | applicability is material-, additive- and adhesive-specific |

All compared policies must use the same candidate pool, service profile, costs, sample availability, material/adhesive identities, holdout splits and compute budget. Report all failed, infeasible and abstained campaigns.

## 8. Paradigm shifts and next benchmark programme

### A. From a recipe to an interface state trajectory

Represent the interface as a time-evolving state tied to an exact material, process run and adhesive:
- bulk moisture/conditioning and additive package;
- initial surface chemistry and topography;
- plasma dose, gas, distance, speed, waveform and process telemetry;
- weak-boundary-layer / damage indicators;
- time since treatment, storage temperature/RH and handling;
- adhesive wetting, application, cure and environmental service history;
- observed failure mode, retained strength and substrate-function loss.

This is an evidence-backed state model, not a claim that all latent variables are already observable. MAT-013 and MAT-INTERFACE remain the canonical lineage/interface authorities.

### B. From in-family parameter fitting to active mechanism falsification

An experiment should sometimes be selected because it can falsify the assumed model family, not because it most improves the posterior within it.

Add sealed worlds where:
- two registered mechanisms predict almost the same WCA but different aged bond outcomes;
- a hidden moisture/additive state changes the outcome;
- a new adhesive changes the ranking;
- the covariance matrix changes by preparation batch or instrument;
- all registered hypotheses have poor held-out predictive coverage.

The selected action must preserve an OTHER/ABSTAIN result when the supplied model family is inadequate. Reuse CTGP-ROBUST-001 #11262 and OLI-EXPERIMENT-001 #11446 rather than adding a parallel discrepancy engine.

### C. From one winner to a robust process region

A useful manufacturing result is a *connected, tolerance-aware operating region*, not a single optimum.

For each material/adhesive/storage stratum, estimate:
- minimum bond and aged-bond constraints;
- maximum damage / substrate-function degradation;
- allowable treatment-to-bond delay;
- sensitivity to moisture, additive and lot variation;
- uncertainty in process boundaries;
- distance to a failure boundary under held-out perturbations.

Use robust feasibility and process-capability measures, not only point-optimal strength. A candidate window is not qualified until it survives the preregistered holdout lots and conditions.

### D. From adaptive exploration to protected confirmation

Separate the campaign into:
1. exploratory design and model challenge;
2. frozen candidate-window nomination;
3. independent confirmatory campaign with new lots/days and blinded sample identity;
4. post-aging / service-profile confirmation;
5. change-control and drift monitoring.

No exploratory adaptive selection may silently reuse the confirmatory set. Randomization, blocking, specimen exclusions and stopping rules must be preregistered.

### E. From proxy correlation to reliability and failure analysis

The primary outcomes should include not just mean lap-shear:
- lower-tail / quantile bond strength;
- failure-mode distribution;
- aged retention with ratio reconciled to initial and aged values;
- survival / time-to-failure under declared service profiles where meaningful;
- creep, fatigue, peel/cleavage or environmental stress outcomes when required by product geometry;
- substrate-function preservation and damage;
- batch-to-batch process capability and decision coverage.

Averages must not hide brittle failure modes or low-performing lots.

## 9. Benchmark scorecard and research targets

Targets below are prospective research goals after a pilot freezes difficulty, not current results, product requirements or universal standards.

| Dimension | Proposed target | Qualification condition |
|---|---|---|
| False acceptance | Zero false ACCEPT on public known-truth negative controls | All mandatory physical endpoint gates active |
| Open-world detection | ≥95% detection of planted out-of-family / discrepancy cases at published in-domain coverage | Sealed family holdout; report false-abstention too |
| Small-set oracle quality | ≤10% decision-regret gap on ≥90% eligible cases | Exact exhaustive oracle; same candidate pool and constraints |
| Destructive-test efficiency | ≥25% fewer expensive tests than fixed campaign | No rise in false acceptance; no mandatory evidence removed |
| Predictive calibration | nominal 90% intervals achieve 85–95% empirical coverage in each adequately powered held-out stratum | Report per lot, adhesive, moisture and day; do not pool away subgroup failure |
| Process-window transfer | no accepted window violates the frozen product floor on the held-out lot/day challenge set | New lots/days, exact article lineage |
| Reproducibility | deterministic replay for identical frozen source, manifest, seeds and policy | Exact source/candidate/covariance SHA binding |
| Safety/authority | zero inadmissible experiment proposals and zero automatic actuation | Separate facility authorization and human release |

Do not combine these into one composite leaderboard number. Plot the quality/cost/latency/sample-use/abstention Pareto frontier.

### Mandatory metrics

Report: false accept, false reject, abstention, false adequacy, false abstention, realized decision regret, worst-group regret, lower-tail bond outcomes, interval coverage, calibration, cost-to-decision, tests/specimens per decision, elapsed time, energy, destructive-test count, number of lots/days, process-window width, exact-oracle gap, and runtime/memory.

## 10. Next executable tranche

1. Run the synthetic integration example against the existing UAP-OED API and compare outputs to the independent dense-matrix reference.
2. Make the selection output bind the action catalogue, scenario/prior manifest, covariance source, selected IDs, budget, UAP code revision and OLI adequacy receipt in a small plasma evidence envelope. Do not create another generic receipt authority.
3. Extend the benchmark with robust scenario/model-discrepancy and decision-loss cases by consuming CTGP-ROBUST #11262, REA decision-aware #10109 and OLI/unknown-model #11181.
4. Add standard-profile identifiers and the selected physical test method to each action; keep licensed normative procedure outside this codebase.
5. Freeze material/adhesive/process/conditioning identities and stratified holdout manifests.
6. Only after source-side synthetic replay and Rust qualification, request a real external-facility protocol review.

## 11. Claim ceiling

This integration is a synthetic UAP selector-composition demonstration. It does not show better real-world selection, a physically improved PA6 bond, reduced laboratory cost, industrial superiority, ASTM/ISO certification or safe plasma actuation. A selection report is a proposed measurement portfolio, not a product release.