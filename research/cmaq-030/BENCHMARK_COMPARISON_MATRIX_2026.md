# CMAQ-030 — Recoverability-Margin Calibration Benchmark Matrix 2026

**Status:** proposed experimental protocol; no comparative result is claimed.  
**Question:** can one-sided calibration of the kth independent recovery-option margin provide a more decision-efficient switch signal than full state-trajectory tubes and conventional barrier-clearance calibration?

## 1. The competitive claim to test

The literature already calibrates safety margins. Chang and Ahmed (2026) calibrate the gap between a planned barrier-function margin and realized safety clearance after trajectory selection. CERT-Replan (CoRL 2026) uses horizon-indexed conformal uncertainty and calibrated barrier risk to trigger replanning.

CMAQ-030 is not claiming the first calibrated safety margin. Its specific hypothesis is that a **kth independent recovery-option margin** is more directly aligned with a multi-mode runtime-assurance decision than:
- a coordinate-wise full-state trajectory envelope;
- a single collision/barrier clearance value;
- a single recovery-controller entry margin.

For a candidate prefix, every qualified recovery profile supplies a margin, but profiles with a shared failure-domain identity collapse to one option. The kth-largest per-domain margin is the limiting margin for retaining k independent options.

We calibrate only dangerous optimism:

`S_c = max over all cells in cluster c max(0, predicted_k_margin - reference_k_margin)`

The finite-sample upper quantile q produces a lower bound `predicted_k_margin - q`. It is useful only within the frozen plant, context, portfolio, certificate and action-selection scope.

## 2. Direct external comparators

| Comparator | Published idea | Fair implementation in this experiment |
|---|---|---|
| Chang & Ahmed (2026), Barrier Function Conformal Safety Clearance Certification with CVaR for Driving Trajectory Selection | Post-selection conformal correction of planned barrier margin versus realized safety clearance on 300 frozen nuPlan sessions | Reproduce the planned/realized clearance score and CVaR policy on the same selected trajectories; do not compare different planner runs |
| CERT-Replan (CoRL 2026) | Horizon-indexed adaptive conformal radii and calibrated barrier risk trigger a change in local planning mode | Use the same action sequence and horizon blocks; record exact replan triggers, risk thresholds, task return and intervention |
| CAFT (ICML 2024) | Simultaneous full-trajectory conformal bands | Apply state-coordinate trajectory band to the same rollouts; derive reachable set and recovery-option margins from the band |
| CMAQ-029 | Cluster-level L-infinity simultaneous trajectory envelope | Use its calibrated reachable tube as the reference state-envelope baseline |
| CMAQ-027 | One-profile recoverability-qualified commitment | k=1 recoverability control |
| CMAQ-028 | k-domain recovery option portfolio without calibrated optimism margin | Same k profiles and same prefix evidence, but raw predicted margins |
| CBF / HJ / viability shield | Hard invariant / reachability filter | Report conservatism, intervention rate, hard-invariant violations and compute |
| Fixed Simplex | High-performance controller with one trusted backup and a fixed switching condition | Matched backup controller and switching logic |

Prior-work links:
- Chang & Ahmed: https://www.max-robotics.com/en/research/publications/cmu42t6js00n9pd42fgp6h78m
- CERT-Replan: https://nail-uh.github.io/corl2026.github.io/
- CAFT: https://proceedings.mlr.press/v235/zhou24l.html
- CMAQ-029: `research/cmaq-029/`

## 3. Environments and tasks

Use staged evidence tiers; do not pool them.

### Stage A — deterministic software witnesses
- Exactly enumerate recovery-profile coverage across candidate prefix blocks.
- Inject optimistic predicted margins and compare against evaluator reference margins.
- Verify failure-domain deduplication and the kth-largest semantics.
- Verify unknown outcome, changed portfolio, changed certificates and action-selection mismatch fail closed.

### Stage B — reproducible simulation
- MuJoCo Hopper/Swimmer and quadrotor reachability tasks from the 2026 Koopman+conformal work.
- Contact-rich manipulation and failure recovery where distinct recovery modes have different sensor, controller or actuator dependencies.
- Driving trajectory selection / nuPlan-style clearance tasks for a direct barrier-clearance comparator.

Each environment uses identical candidate trajectories, recovery profiles, independent failure domains, query/action cells, disturbance scenarios, calibration splits and evaluator reference evidence across all methods.

### Stage C — hardware-in-the-loop / physical
Only after independent verifier, sensor custody, actuator model and recovery controller certificates are available. Simulator, HIL and physical measurements remain separate evidence populations.

## 4. Experimental design

### Predictor × assurance factorial

| Predictor | Assurance method |
|---|---|
| NixWard global action-delta HDC | No calibrated margin; negative/control baseline |
| NixWard contextual HDC | No calibrated margin; negative/control baseline |
| NixWard contextual HDC | CMAQ-029 full state-coordinate cluster envelope |
| NixWard contextual HDC | CMAQ-030 one-sided kth recovery-margin envelope |
| Koopman predictor | State-coordinate conformal reachability |
| Koopman predictor | One-sided recovery-margin calibration |
| Latent ensemble / CDR-LWM-style predictor | One-step conformal backup governor |
| Direct controller / VLA | Fixed Simplex, CBF/HJ and CERT-Replan controls |

The model-fitting, scale-validation, calibration and test mechanism/environment clusters must be disjoint. A candidate action/search selector that searches multiple candidate trajectories must be frozen and represented in the selection-protocol digest. If search budget, query portfolio or candidate selection changes, the old calibrator cannot be reused.

### Recovery portfolio conditions
- k=1: one qualified profile.
- k=2: two declared independent failure domains.
- k=3: three declared independent failure domains, where the benchmark has sufficient diversity.
- common-cause challenge: duplicate controller/sensor/actuator/compute paths with distinct profile IDs but one failure-domain ID.
- injected one-domain failure: remove any one declared failure domain and re-evaluate the remaining k-1 requirement.
- context scope shifts: one or more recovery profiles become ineligible; the candidate must not keep counting them.

### Calibration
- one maximum one-sided optimism score per independent cluster block;
- full pre-registered rollout × action/query cell × horizon-prefix inventory;
- exact candidate-margin block digest bound into the evaluator reference receipt;
- exact recovery portfolio/certificate/failure-domain manifest digest;
- finite-sample split-conformal order statistic;
- fail closed when k-rank exceeds the number of independent calibration clusters.

A 95% non-randomized order statistic requires at least 19 independent calibration clusters to produce a finite radius. This is only the mathematical minimum, not a sufficient scientific sample. Pre-register sample-size and confidence-interval requirements at the independent cluster unit. A practical study should target at least 100 calibration clusters and 200 or more held-out clusters where feasible, then justify the actual sample with coverage interval width and power calculations.

## 5. Primary metrics

### Statistical validity
- cluster-level simultaneous coverage of the kth recovery margin across the whole pre-registered block;
- realized cluster coverage and cluster-level confidence interval;
- family/context-cell coverage only as diagnostic unless a conditional method justifies the claim;
- calibration support and empirical calibration-conditional variability;
- bad-calibration / undercoverage probability where the relevant dependence model supports it.

### Decision / assurance quality
- false-positive portfolio admission (lower bound positive but reference kth margin non-positive);
- false-negative admission (lower bound non-positive but reference margin positive);
- earliest failing prefix and missed-switch count;
- percent of committed prefixes retaining at least k valid domains;
- single-domain fault survival and multi-domain fault survival under declared faults;
- successful recovery rate and time/energy to recovery.

### Conservatism / performance
- calibrated lower margin and frequency of nonpositive lower margins;
- high-performance action utilization / mission return;
- unnecessary switches and late switches;
- reachable set / tube volume, reported separately from the margin statistic;
- action regret and task success at matched action/rollout budget.

### Resource cost
- predictor + calibration + portfolio/evidence index resident bytes;
- calibration and runtime inference latency;
- switch latency, energy, recovery energy and CPU/GPU cost;
- cost per successful task and per recovered failure.

## 6. Falsifiable hypotheses

H1. At matched cluster-level coverage, the kth recovery-margin method yields fewer unnecessary switches than a coordinate-wise L-infinity tube when some state-coordinate errors are irrelevant to all certified recovery entries.

H2. At matched switch/false-admission rate, the calibrated recovery-margin method produces higher high-performance utilization than an invariance-only shield, under the same independently verified recovery profiles.

H3. Under injected common-cause profile duplication, failure-domain deduplication prevents false redundancy and catches cases that raw profile counting admits.

H4. Under optimistic model errors, the one-sided score prevents more false positive recovery-option admissions than uncalibrated point margins.

H5. Under model pessimism, the one-sided score does not penalize the candidate as though conservative margins were dangerous; compare its utility/false-negative rate with symmetric absolute-error calibration.

H6. Under shifts outside the pre-registered observable context and portfolio scope, the evaluator returns no usable calibrated margin instead of claiming coverage.

Each hypothesis needs a matched paired evaluation, cluster-level statistical interval, worst-family/context results and all negative controls.

## 7. Proposed qualification gates — targets, not results

| Gate | Proposed criterion |
|---|---|
| Exact deterministic semantics | Correct kth-largest domain margin and one-sided max score for all known-answer witnesses |
| Finite calibration | Correct exact rank; no finite radius if support is insufficient |
| Evidence binding | Zero acceptance of mismatched candidate/reference block, portfolio, cert manifest, plant/context, selection protocol or cluster split |
| False-positive portfolio admission | Zero on deterministic negative controls; statistically bounded on held-out scenarios |
| Recovery option survival | Preserve k options across each admitted deterministic prefix; survive injected single-domain removal when k=2 requires it |
| Efficiency advantage | At equal coverage/false-admission rate, reduce unnecessary switching or improve task utility versus at least one direct margin baseline |
| Resource advantage | Show lower all-in compute/latency/memory than a matched full-tube method, without coverage/assurance regression |

The gates do not imply physical safety. An external verifier must establish that reference margins derive from valid certificates and authenticated state evidence.

## 8. What would count as beating SOTA

A useful result would be a Pareto improvement over:
- full-state coordinate-wise conformal reachability (less conservatism for equal recovery-option coverage);
- post-selection barrier-clearance conformal (better option survival / common-cause resilience at equal false-admission rate);
- horizon-triggered conformal replan (higher high-performance utilization at equal recovery success);
- uncalibrated portfolio margin (material reduction in optimistic false admissions).

The fair claim is not that Symthaea beats all world models. It is that the same underlying predictor, when paired with the CMAQ evidence/identifiability/recoverability layers, makes better-qualified decisions with a measured resource and utility advantage.

A result must name the predictor/checkpoint, plant, recovery portfolio, failure-domain evidence, calibration population, cluster split, alpha, selection protocol, action budget and independent verifier receipt. No SOTA label should be used until matched external baselines have actually been executed.
