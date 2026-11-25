# SEMI-CAUSAL-DRIFT-001 — falsifiable manufacturing drift diagnosis v0.1

**Issue:** #13673  
**Implementation:** `symthaea-semiconductor::benchmark::causal_drift_diagnosis`  
**Base:** the validated streaming/evidence branch, PR #13478 exact head `9b058ab1c5e4f5c1d20b6fb40f76b306504164cf` (SEMI-BENCH-001 run #1103 passed; CI and PR Governance were skipped on the draft stack).  
**Status:** first typed implementation and adversarial tests authored; this branch's exact-head validation is pending.

## Research basis and honest comparison

M2D2 (Lin, Tseng and Tsai, Applied Sciences 2025, 15(12), 6500; DOI 10.3390/app15126500) brings together multimodal drift detection, SHAP/LIME, active learning, maintenance recovery and drift-aware federated learning. Its paper explicitly identifies deeper graph/knowledge-graph interpretation and heterogeneous adaptation as future work. The authors say their real WBG fab data are not public because of semiconductor-company confidentiality (available on request).

NIST's 2026 TIDAL manufacturing initiative identifies root-cause analysis at the asset level and shared integration-test-bench evaluation across manufacturing applications. ISO/AWI 23247-8 is an approved VVUQ work item under development, not a published standard.

The proposed differentiation is therefore not "we have explainability and others do not." It is a typed boundary between:
- feature attribution;
- observational/matched association;
- randomized intervention consistency;
- unidentifiable or contradictory evidence;
- human-reviewed physical root-cause decisions.

## Implemented v0.1 contract

The input carries an incident, a declared information cutoff, source-inventory completeness, clock/join integrity, policy thresholds and competing causal hypotheses.

Hypotheses include sensor, calibration, tool, chamber, process/recipe, facility common-cause, material/product mix, clock/join, selection bias, model/preprocessing and unknown causes.

Evidence is typed: feature attribution, temporal precedence, maintenance record, reference measurement, clock/join audit, source inventory, matched contrast or controlled intervention. Each record binds:
- hypothesis ID and evidence ID;
- event time and availability time;
- declared clock domain;
- source artifact SHA-256 and authority reference;
- optional attribution score.

A SHAP/LIME-style attribution score is never used as a causal effect estimate. It can only contribute to an association-only assessment.

## Quantitative contrast

The matched contrast accepts four treated/control pre/post means, four sample variances and group counts, maximum covariate standardized mean difference, treated/control pre-trend slopes and protocol/source digests.

The difference-in-differences point estimate is:

`(treated_post - treated_pre) - (control_post - control_pre)`

The initial cell-summary path uses a transparent independent-cell Wald approximation:

`sqrt(var_Tpre/n_Tpre + var_Tpost/n_Tpost + var_Cpre/n_Cpre + var_Cpost/n_Cpost)`

This is intentionally labelled `difference_in_differences_independent_cell_wald_95`. It does **not** account for paired-panel covariance, serial autocorrelation, clustering by tool/chamber/lot or repeated-wafer dependence. It is suitable only for frozen synthetic known-answer tests, not as a production interval.

A separate cluster-summary path is now implemented. Each row represents one declared independent top-level unit (e.g. chamber, tool or lot) and contains its four treated/control pre/post means, per-cell support counts, covariate SMD and pre-trend slopes. The estimator computes one DiD effect per cluster, then reports the equal-weight mean cluster effect. Its standard error is the sample standard deviation of cluster effects divided by the square root of the number of clusters. It uses the conservative 95% Student-t critical value for 9 degrees of freedom and requires at least 10 independent clusters. The method is explicitly labelled `equal_weight_cluster_mean_did_student_t95_conservative_df9_nG`. This estimates an equal-cluster average, not a wafer-weighted average. It still assumes independent clusters; it does not solve higher-level nesting or hidden confounding. A real fab study must predefine the independent unit, avoid overlapping clusters, and consider higher-level cluster/bootstrap or randomization inference where the design requires it.

Balance and pre-trend thresholds are diagnostics, not proof of parallel trends or absence of unmeasured confounding. A matched contrast can only return ASSOCIATION_ONLY or CONTRADICTED, never INTERVENTION_CONSISTENT.

## Intervention gate

INTERVENTION_CONSISTENT is emitted only when the separately declared intervention contains:
- randomized assignment;
- preregistered analysis;
- control group;
- passed manipulation check;
- declared no-interference assumption;
- minimum treated and control support;
- assignment protocol, preregistration and independent-verification digests;
- an effect interval consistent with the predeclared direction;
- no material negative-control violation.

The output still sets `causal_claim_authorized=false`, `source_authenticity_proven=false`, `human_review_required=true`, `physical_action_authorized=false` and `equipment_control_authority=false`. The flags and digests are declarations, not signatures proving that a physical experiment occurred.

## Falsification and abstention

- Observational/feature-attribution evidence => ASSOCIATION_ONLY.
- Matched contrast with failed balance or pre-trend threshold => UNIDENTIFIABLE.
- Material negative-control effect => UNIDENTIFIABLE.
- Missing source inventory, clock/join integrity or evidence available by the analysis cutoff => UNKNOWN.
- Strong confidence interval opposite the declared direction => CONTRADICTED.
- Non-randomized or failed manipulation-check intervention => UNIDENTIFIABLE.
- No evidence rows => UNKNOWN.
- Malformed source digests, duplicate IDs, invalid support or non-finite statistics => reject input.
- Unknown cause type is retained as a candidate; it is not forced into a known class.

Contradictions remain visible. The evaluator does not average contradictory evidence into a single confidence score.

## Authored test coverage

The module currently has 15 unit tests; the integration reference file has 4 tests. They cover:
1. feature attribution never produces a causal claim;
2. difference-in-differences known answer remains association-only;
3. failed covariate balance downgrades identification;
4. negative-control violation blocks promotion;
5. randomized, preregistered intervention can be intervention-consistent but not authorized;
6. non-randomized intervention is unidentifiable;
7. incomplete inventory and failed clock/join integrity return UNKNOWN;
8. evidence after the analysis cutoff or from another clock domain returns UNKNOWN;
9. contradictory effect direction remains visible;
10. input order does not change sealed digest or assessment ordering;
11. malformed digests and duplicate hypothesis IDs are rejected;
12. independent integration oracle verifies the DiD estimate and standard error;
13. independent integration oracle verifies negative-control falsification;
14. independent integration oracle verifies all intervention design gates;
15. cluster-level DiD uses between-cluster variance and refuses fewer than 10 independent clusters;
16. cluster-level known-answer mean/standard error/Student-t interval are independently recomputed.

Tests are authored, not yet claimed passing. Exact-head rustfmt, strict Clippy, semiconductor crate tests, independent binary/reference controls and SEMI-BENCH-001 reference run are pending.

## Competitive evaluation programme

Baselines:
- M2D2 multimodal drift + SHAP/LIME/active learning (paper architecture; private dataset means no direct reproduction claim until data access);
- EWMA, CUSUM, Page-Hinkley and ADWIN;
- PCA/PLS and conventional FDC thresholds;
- Isolation Forest and autoencoder/reconstruction-error anomaly detection;
- SHAP/LIME/attention attribution (association only);
- matched DiD;
- randomized intervention evaluator;
- no-diagnosis/abstention.

Synthetic ground-truth challenge families include sensor-only drift, process-only drift, calibration-only change, chamber-local failure, multi-tool common cause, recipe/product-mix confounding, sampling-policy shift, clock/join error, maintenance coincidence, randomized repair, simultaneous sensor+process shift, post-maintenance recovery, hidden confounding, retroactive calibration revocation, unknown cause and incomplete inventory.

Report top-1/top-3 cause candidate recall separately from causal-claim precision. Also report false intervention-consistency rate, abstention accuracy, common-cause detection, chamber-localization precision, detection delay, false alarms per 1,000 wafers, evidence acquisition cost and experiment-selection regret.

Proposed synthetic acceptance targets (not achieved):
- 100% observational-only cases remain non-causal;
- 100% incomplete clock/source/identity cases return UNKNOWN;
- 0 intervention-consistent outputs when randomization, preregistration, control, manipulation check or independent verification is missing;
- 0 hidden-confounder cases are promoted above association-only without a valid identification design;
- at least 95% top-3 candidate-cause recall on fully observed balanced synthetic cases;
- at least 90% correct abstention on unknown/unidentifiable cases.

## Explicit next steps

1. Add a cluster-aware/block-bootstrap DiD estimator with frozen known-answer calculations.
2. Add a causal graph *hypothesis* layer with explicit parent-child mechanisms but no inferred physical-causal edges.
3. Add experiment candidates ranked by expected information gain per physical-metrology cost, while retaining external authorization gates.
4. Add common-cause versus chamber-local fault challenge suites.
5. Add a human review receipt and evidence closure from the existing K0/K1 label and snapshot chain.
6. Seek an M2D2 author/data collaboration; until then, compare with declared synthetic fault injections, not paper-reported numbers.

## Claim ceiling

This is a typed, falsifiable hypothesis/evidence evaluator over declared summaries. It is not a causal-discovery engine, it does not prove physical source authenticity, and it does not identify a production root cause from SHAP or correlation. No experiment is executed; no maintenance, recipe, APC/R2R, wafer-disposition or equipment-control action is authorized.
