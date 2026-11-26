# MEL temporal-grounding qualification harness

This nested Rust package imports the exact production temporal-grounding scorer and SHA-256 digest module by path. It remains isolated from the main workspace so focused qualification does not inherit unrelated legacy build/lint debt.

Run the scorer checks with:

    cargo +1.96.0 test --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --all-targets -- -D warnings

## Multi-annotator temporal grounding

The temporal_consensus module preserves every independent annotation record and runs the exact production temporal-grounding scorer once per annotator. It reports per-annotator TP/FP/FN, F1, occurrence count accuracy and boundary errors, plus panel summaries for positive grounding, negative claims, occurrence-count spread and boundary errors.

Ambiguous and not-estimable labels are retained in the report but excluded only from their own scored denominators. An absent label remains a scored negative reference. The evaluator does not create a majority-voted/merged consensus interval, rank annotators, compute pairwise inter-annotator agreement, or treat a small panel's score spread as a population confidence interval.

Run the synthetic six-annotator panel with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-multi-annotator-grounding-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-multi-annotator-temporal.synthetic.json

All annotator identities are represented by opaque digests in the fixture/report. Raw personal identifiers must not be placed in public benchmark artifacts.


The strict input envelope is also described by `schemas/mel-multi-annotator-temporal-grounding-v1.schema.json`; cross-record identity, state/interval semantics and digest uniqueness are enforced by the Rust adapter.

Count-gated one-to-many F1 is computed only over Present annotations whose reference contains more than one occurrence. Single-event references remain in ordinary F1 and exact-count accuracy but do not dilute the one-to-many denominator.

The report separates eligible annotation count from model abstention, parse failure and prediction coverage. A present/absent annotation stays eligible even if the model returns Ambiguous, NotEstimable, NotDetected or Unparseable. Coverage is reported overall and separately for Present and Absent references. The panel score spread is human-reference variability; it is not model confidence. The adapter does not compute formal pairwise inter-annotator agreement or population confidence intervals.

Research protocol: `docs/research/MEL-GROUND-006-MULTI-ANNOTATOR-TEMPORAL-GROUNDING-V1.md`.


For each predicted occurrence, the report also derives a **per-event reference-boundary distribution** from the production scorer's existing prediction↔annotation match records. It reports the number and fraction of eligible Present annotators who match that event, plus min/median/max start and end frames and boundary spread. Events that match no eligible Present annotation appear in `unmatched_prediction_indices`. This is not an interval consensus: original annotation records remain intact, and the support denominator is visible.


## Dataset-level inter-annotator reliability

The temporal_reliability module consumes multiple #13560 cases across query items and source works. It reports three different quantities without combining them:

1. Krippendorff's nominal alpha over Present/Absent labels across multiple query items. Ambiguous and NotEstimable labels remain in the source records but are excluded as missing for this nominal-state measure. If expected disagreement is zero, alpha is undefined rather than being reported as 1.
2. Symmetric interval-set agreement for Present/Present annotator pairs. Each pair is scored in both directions using the production scorer; interval F1, matched IoU, exact occurrence-count agreement and count difference remain separate.
3. Boundary agreement in milliseconds, with tolerance rates at the dataset's strictly ascending preregistered millisecond thresholds. These are event boundary comparisons, not model confidence.

Every dataset item binds a source_work_id_sha256. The bootstrap resamples all query items from a source work together using deterministic SplitMix64, and reports separate intervals for alpha, pairwise state agreement, interval F1/IoU, count agreement, start/end boundary MAE, and each preregistered start/end/joint boundary-tolerance rate. Point estimates for pairwise and boundary metrics are source-work macro-averages, matching the bootstrap estimand. A proposed 30-source-work floor is reported but does not by itself authorize population claims; the output always sets population_confidence_claim_permitted to false.

Run the synthetic six-query/three-source-work dataset with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-reliability-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-temporal-reliability-six-items.synthetic.json

The fixture is schema-backed by schemas/mel-multi-annotator-reliability-dataset-v1.schema.json. It is a deterministic implementation control, not a musician-reliability study or SOTA result. Dataset items must share one scoring plan and annotation-policy version; source-work cluster IDs and annotation record IDs must be unique/consistent under the dataset contract.


The reliability report also retains a **separate model-vs-annotator evaluation** for every query item. It includes the exact model prediction, source/audio identity, frozen scorer plan, original pseudonymous annotation records, per-annotator model scores, and the #13560 coverage/F1/count/absence summaries. Dataset model-performance summaries are macro-averaged by source work and keep Present F1, exact count, one-to-many count-gated F1, Absent claim accuracy, coverage and fully-correct rate separate. The inter-annotator alpha and pairwise interval reliability remain independent quantities. This prevents a high/low alpha from being mistaken for model performance. Model confidence calibration is not computed.


## Dual-stream annotation acquisition

The temporal_acquisition module consumes candidate-level outputs from the #13560 per-annotator model scorer and #13617 dataset reliability evaluator. Each candidate binds:
- exact source-work and recording digests;
- the purpose scope and pseudonymous cohort role;
- a rights/consent receipt digest;
- the #13560 model-grounding report digest;
- the #13617 inter-annotator reliability report digest;
- a frozen cohort-assignment manifest digest;
- annotation panel, cost, model-error, disagreement, count, coverage and boundary-dispersion signals.

The planner produces two separate streams:

1. Active refinement: deterministic priority bands for likely model failure on a stable/covered panel, human disagreement, coverage gaps and hidden repeated-event controls. It round-robins across source strata, enforces per-source-work caps, and respects active budget/quotas. An uncertain panel is routed to annotation refinement, not mislabelled as a clean model failure.
2. Probability audit: source works are sampled uniformly within each stratum, then one eligible item is drawn uniformly within each selected source work. The report records source-work, conditional-item, and product inclusion probabilities. Per-stratum sample sizes are fixed before any random draw using worst-case item-cost reservation, so a cheap random item in an earlier stratum cannot change the sample size/probability in later strata.

The active and audit source-work sets are required to be disjoint. Active selections are not a probability sample; probability-audit selections are not adapted to model error. Their results must not be pooled without the proper design weights. A non-selected item is unreviewed, not a negative event.

Rights/consent receipt digests and diagnostic report digests are structurally validated and included in the replayable receipt, but this planner does not authenticate those external records. Its selection receipt is also not an authentication token.

Run the synthetic dual-stream plan with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-annotation-acquisition-plan -- tools/mel-temporal-grounding-qualifier/fixtures/mel-annotation-acquisition.synthetic.json

The committed fixture is 11 candidates across two strata. It is a deterministic planner control only; no human study, unbiased population estimate, or annotation-quality improvement is established.


## Two-stage finite-population design estimator

The `temporal_design_estimator` module replays the source-balanced annotation acquisition planner from the exact candidate frame, configuration and seed. Reviewed runs should supply `expected_selection_receipt_sha256`; a mismatch is rejected. A null expected digest is allowed only for a fresh deterministic demonstration and is explicitly marked unverified.

Only completed outcomes for selected `ProbabilityAudit` items are accepted. Each outcome is bound to the selected item’s source-work digest, scoring-report digest, model identity, evaluator version and scoring-plan digest. The whole run must use one model/evaluator/plan identity. Missing, duplicate, extra, non-audit, wrong-metric, wrong-source and out-of-range outcomes are rejected. These references are still unauthenticated metadata until an independent evidence verifier authenticates them.

The estimator reports two different finite-population estimands; they must not be conflated:

- **Query-item-weighted total and mean:** every eligible query item has equal target weight. The Horvitz–Thompson total is \(\hat T_{item}=\sum_{i\in s}y_i/\pi_i\), and the mean divides by the known number of eligible query items.
- **Source-work-weighted total and mean:** every source work has equal target weight, regardless of its number of eligible query items. Each item contributes \(y_{wi}/N_w\) to the source-work total before Horvitz–Thompson weighting. The estimated total is the sum of work means; the population mean divides by the number of eligible source works.

For a two-stage stratum with \(W\) eligible source works and \(K\) selected works, and work \(w\) containing \(N_w\) eligible items with \(m_w\) sampled, the first-order item inclusion probability is \(\pi_{wi}=(K/W)(m_w/N_w)\). For two distinct items in the same work, \(\pi_{ij}=(K/W)\,m_w(m_w-1)/(N_w(N_w-1))\). For different works, \(\pi_{ij}=K(K-1)/(W(W-1))\,(m_w/N_w)(m_v/N_v)\). The general HT variance estimator uses these first- and second-order probabilities for each estimand separately.

Variance is withheld when the design does not support it (for example, fewer than two sampled works when multiple works exist, or insufficient within-work sampling), or when an inclusion probability is zero/invalid or the computed variance is negative/non-finite. It never fabricates a confidence interval from an unavailable variance. A complete census of a one-work finite population has zero design variance; observing one work out of several does **not** identify between-work variance.

Run the synthetic one-work complete-census example with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-two-stage-design-estimator -- tools/mel-temporal-grounding-qualifier/fixtures/mel-two-stage-design-estimator.synthetic.json

That fixture has one eligible work and two eligible query items, both selected. It returns query-item total 1.0 and mean 0.5; source-work total and mean are both 0.5; both design variances are zero because the entire finite population is observed.

The Rust exhaustive oracle separately enumerates all three possible two-of-three source-work samples for a six-query finite population. It verifies that the average query-item HT total equals the true population total 3.0 and the average estimated query-item variance equals 1.5. It also verifies that the source-work-weighted HT mean is 0.5 and its average estimated total variance equals 0.375. This validates the estimator equations on synthetic finite populations, not a field result or authenticated human/model outcome.

The report flags `observed_outcomes_authenticated=false` and `population_confidence_claim_permitted=false`. A design-based variance estimate is not a license to claim population generalization unless the sampling frame, selection receipts, scoring outputs, rights/consent, and measurement process are independently qualified.


## Design-weighted selective risk/coverage

The `temporal_selective_design` adapter composes the existing #13626 acquisition planner, #13648 selective decision outputs, and #13692 two-stage HT variance estimator. It accepts only one completed selective outcome for each selected ProbabilityAudit item. Active-refinement and unselected items are rejected. Each outcome is joined to the exact planner item/query/source-work and the same model/runtime/decoding/hardware/evaluator/scoring-plan identity tuple.

The threshold list is frozen in a versioned plan with a canonical SHA-256 digest. V1 accepts only the predeclared `greater_or_equal` tie rule. A case is covered when its decision is Answer or ClaimAbsent and its confidence is at least the threshold. AbstainAmbiguous and AbstainNotEstimable remain uncovered. The selective error numerator contains only covered incorrect decisions; false absence claims are also reported separately.

For each threshold, the adapter reports two separate estimands:
- query-item-weighted coverage and selective risk;
- equal-source-work contribution coverage and selective risk.

For risk (R=T_E/T_C), V1 uses ratio linearization with (Z_i=E_i-RC_i), then estimates the design variance of the HT total of (Z_i) using the same first/second-order inclusion probability primitive as #13692. The source-work estimand applies the same calculation to (Z_i/N_w). Coverage variance is the HT variance of the covered total divided by the known population denominator squared. Risk is undefined when the estimated covered total is zero. Unsupported design variance remains unavailable, not zero.

The 95% intervals are normal-approximation, proportion-clipped intervals; they are diagnostics, not guarantees of nominal coverage in small samples. The adapter emits threshold-specific intervals only. It does not compute an AURC confidence interval because covariance across thresholds is deferred. Thresholds must be selected using source-lineage-disjoint calibration data before opening the confirmatory audit.

The synthetic one-work/two-item fixture demonstrates:
- threshold 0: coverage 1.0, selective risk 0.5, false-absence rate 0.5;
- threshold 0.7 or 0.9: coverage 0.5, selective risk 0.0, false-absence rate 0.0;
- always-answer and random-retention baseline risk 0.5.

That fixture is a complete census of one synthetic work, so its design variance is zero. The separate Rust tests exercise two-of-three work sampling and unequal work sizes. Neither fixture nor test establishes a field result or SOTA claim.

Run the synthetic audit with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-selective-design-risk-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-selective-design-risk.synthetic.json

The output hard-codes `selective_outcomes_authenticated=false` and `population_confidence_claim_permitted=false`. A signed execution/scoring receipt and a qualified rights/consent frame remain necessary before a population performance claim.
