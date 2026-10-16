# Prediction-Market Resolution Coverage and Evidence Identifiability v1

**Status:** research proposal; no real-market implementation or result claimed  
**Related work:** [PR #7613](https://github.com/Luminous-Dynamics/symthaea/pull/7613), [benchmark protocol v1](PREDICTION_MARKET_COMPARATIVE_REPLAY_BENCHMARK_V1.md), [corpus/uncertainty issue #7630](https://github.com/Luminous-Dynamics/symthaea/issues/7630)  
**Intended mode:** offline synthetic qualification first; historical shadow replay only after independent data-provenance review

## Executive thesis

A detector cannot rescue an ill-defined estimand. Before calculating an information-timing or informed-flow statistic, the evaluator must establish whether the contract's resolution semantics, event lifecycle, and public-information anchor make the proposed quantity identifiable. An event that cannot be classified from the retained evidence must be reported as ambiguous or out of scope, not coerced into a binary label.

The target is not a universal "insider score." It is a versioned evidence pipeline with separate claims and separate acceptance gates:

1. **Semantic identifiability:** what event did the contract actually resolve, under which rule and source?
2. **Temporal identifiability:** what can be established about public-information availability, and with what timestamp uncertainty?
3. **Forecast quality:** how well did a stated probability forecast the defined outcome at a frozen horizon?
4. **Market-behavior diagnostics:** what price, order-flow, liquidity, or wallet behaviors departed from a category-matched null?
5. **Evidence authenticity and authority:** are retained bytes intact, was the artifact signed by an authenticated principal, and did that principal have the right role under the applicable policy?
6. **Human adjudication:** what limited conclusion is supported by the intersection of the above evidence, and what remains unknown?

These are different estimands. Do not average them into a "trust score," treat a successful hash check as proof of source truth, or convert a statistical anomaly into an allegation.

## Research motivation and limits

The 2026 ForesightFlow population-scale preprint reports examining 12,708 candidate Polymarket markets, with computable scores for 88; it reports 14 of 32 documented FFIC markets as unclassifiable because the resolution criterion was genuinely ambiguous. This is evidence that semantic scope and coverage need to be measured directly. It is a preprint's reported result, not an independently replicated benchmark result.

The initial ForesightFlow methods preprint also reports that a resolution-anchored proxy failed to separate its matched controls as intended and that replacing a proxy timestamp with an article-derived time materially changed a case's score. The methodological implication is that timestamp recovery is a measurement model, not clerical metadata.

Sources:
- Nechepurenko, *Information Leakage at Population Scale: An Evaluation of the Polymarket Insider-Relevant Subpopulation, 2020-2026* (2026 preprint): https://arxiv.org/abs/2605.00459
- Nechepurenko, *ForesightFlow: An Information Leakage Score Framework for Prediction Markets* (2026 preprint): https://arxiv.org/abs/2605.00493
- Nechepurenko, *Empirical Evaluation of Deadline-Resolved Information Leakage on Documented Polymarket Insider Cases* (2026 preprint): https://arxiv.org/abs/2605.02286
- Walker, *The Price is (Almost) Right: A Calibration Analysis of Polymarket Pricing Data* (Princeton thesis, 2026): https://theses-dissertations.princeton.edu/entities/publication/d5708372-5ea3-4252-b669-7f9cee07387c
- Waghmare and Ziegel, *Proper Scoring Rules for Estimation and Forecast Evaluation*, Annual Review of Statistics and Its Application 13 (2026): https://doi.org/10.1146/annurev-statistics-042424-050626

These publications motivate a benchmark design; their published figures are not Mycelix/Symthaea results and are not adopted as a universal prior.

## Semantic eligibility is a reported outcome

For every candidate market, emit exactly one versioned resolution class before computing a scoped metric:

- **Event-resolved:** retained contract language and resolution evidence establish that the terminal state followed a qualifying event occurrence.
- **Deadline-resolved:** retained contract language establishes that the state is determined by a declared deadline or by whether a defined event occurred by that deadline.
- **Ambiguous / unclassifiable:** the contract or retained evidence does not identify one class without discretionary interpretation.
- **Other / unsupported:** cancellation, invalidation, multiple mutually incompatible resolution rules, or another structure outside the registered classifier scope. The exact subtypes must be enumerated in the schema and benchmark corpus.

The exact class names and boundary conditions must be frozen in a versioned taxonomy before scoring. It is acceptable to revise the taxonomy in a new version; silently relabeling old events is not.

For each class, retain: contract ID and revision; canonical contract digest; exact resolution-rule text or immutable reference; source authority claimed by the contract; outcome state; effective event/deadline times; evidence artifact digests; adjudication rule version; classifier decision; exclusion reason; and any human adjudication record. Keep unknown fields unknown. A class label generated by code is a classification output, not evidence that the external-world event actually occurred.

## Temporal evidence should be an interval when time is uncertain

A recovered public-information time can be a point only when the source establishes a defensible point under the registered policy. Otherwise retain an interval or a set of candidate anchors, for example:

- source-published timestamp;
- earliest independently archived availability;
- article publication time;
- official announcement time;
- market-venue ingestion time, if its logs and clock are qualified.

These timestamps answer different questions and must not be substituted for one another. Store source, retrieval time, clock assumptions, timezone normalization, precision, and the reason for selecting an anchor. If evidence only supports `t ∈ [t_min, t_max]`, calculate the diagnostic across that interval or across preregistered anchor choices and report the full sensitivity range. Do not report a point estimate without disclosing the interval's impact.

If the sign, scope condition, or decision changes across supported anchor choices, the correct output is **anchor-sensitive / unresolved**, not the most convenient result. A sensitivity envelope is not a confidence interval; it captures operationalization uncertainty, not sampling variability.

## Separate detector outputs rather than collapsing evidence

The pipeline should emit independently versioned outputs:

| Layer | Core question | Required output | What it must not claim |
|---|---|---|---|
| Semantics | Is the target outcome identifiable under the contract? | class, scope, evidence, exclusion/ambiguity reason | Truth from a schema-valid record |
| Timing | Is the public-information anchor defensible? | anchor candidates/interval, provenance, sensitivity envelope | Intent or insider identity |
| Forecast | Was the probability forecast accurate and calibrated? | horizon/category Brier, log loss, reliability, coverage and uncertainty | Informed trading or manipulation |
| Market behavior | Is observed activity unusual under an appropriate null? | separate flow, price impact, depth, spread, slippage and concentration diagnostics | Guilt from anomaly |
| Integrity | Are bytes and lineage internally consistent? | digest and chain verification outcomes | Publisher authenticity absent signature verification |
| Authority | Is the signer authenticated and authorized for this role? | signature, credential status, role/policy receipt | External-world truth or settlement permission |
| Adjudication | What does the evidence jointly support? | bounded finding, counterevidence, unknowns, reviewer trail | A blended opaque trust number |

A later decision may consume several outputs, but its policy must specify the prerequisites and how missing/ambiguous outputs are handled. Missing data must never be mapped to a normal observation or a reassuring zero.

## Qualification corpus: synthetic first

Extend the existing deterministic corpus with machine-readable, preregistered families. Every family has clean controls and at least one adversarial mutation; train/tuning seeds and held-out evaluation seeds are disjoint.

1. Clear event-resolved contract with correct event evidence.
2. Clear deadline-resolved contract with both deadline outcomes represented.
3. Incomplete or internally contradictory resolution wording.
4. Contract revision after market creation, including unauthorized retrospective changes.
5. Multiple possible trigger events and near-simultaneous announcements.
6. Publication timestamp versus earliest archived availability versus venue-ingestion time.
7. Timestamp precision loss, timezone offsets, clock skew, and timestamp uncertainty intervals.
8. Evidence artifact altered, missing, duplicated, or present but retrieved after the recorded adjudication.
9. Valid content digest but wrong/untrusted signer or unauthorized resolver role (once the Mycelix cryptographic layer exists).
10. Copied accounts, shared model lineage, source-group split/merge, and duplicate source evidence.
11. Low liquidity, quote churn, wash-like turnover, same-owner round trips and depth shocks.
12. Clean high-signal legitimate information releases and category-specific base-rate shifts.

Each scenario record must include scenario ID/version, generating assumptions, seed, expected semantic class, expected eligibility, expected exclusions, truth-known status, and output checks. A synthetic "insider" label means only that the generator injected the specified pattern; it does not calibrate a legal or real-world accusation.

At minimum, test 1,000 deterministic seeds per preregistered family for ordinary property checks. Use at least 2,000 independent coverage replicates per data-generating process when estimating nominal 95% interval coverage, and report the Monte Carlo uncertainty around that coverage. The final replication count is itself preregistered. Synthetic diversity is not a substitute for real-world validation.

## Benchmarks to aim for

All numbers in this section are proposed research targets, not existing results or external standards.

### A. Semantic and temporal identifiability

- 100% agreement with the frozen expected class on the enumerated, machine-readable synthetic cases; every ambiguous/unsupported case must remain explicitly non-actionable.
- 100% of candidate records accounted for in denominator reporting: scored, excluded, ambiguous, unsupported, missing, or malformed. Counts must reconcile exactly.
- 100% of temporal diagnostics must identify their anchor policy and source provenance. When supported timestamp alternatives cross a decision boundary, output must be marked anchor-sensitive.
- Zero future information in historical replay: no feature, resolved label, post-cutoff announcement, later quote or later wallet behavior may alter an earlier feature snapshot.
- Target at least 80% semantic resolvability in a *preregistered, defined evaluation cohort* before making a broad claim about that cohort. This is a programmatic data-quality goal, not a universal benchmark. If the target is not met, report coverage and narrow the scope; never drop ambiguous rows silently to make a detector look better.

### B. Forecast and inference integrity

- Exactly zero post-cutoff forecast selections and zero score changes under post-cutoff-only mutations.
- 100% rejection of the registered structural negative controls while all clean controls remain admitted.
- Report Brier score, log loss, calibration/reliability, eligible coverage, and exclusion counts per horizon and category. Include both all-eligible and common-cohort paired comparisons.
- Verify 90% and 95% interval coverage on known synthetic distributions, varying event count, number of clusters, cluster-size imbalance, serial dependence, outcome prevalence and model misspecification. A nominal 95% method should approach 95% empirical coverage within a preregistered Monte Carlo tolerance; reject the method or narrow its claim if it systematically undercovers. Do not assume cluster bootstrap coverage merely because resampling is clustered.
- Treat parent events/clusters as sampling units where dependence warrants it. Compare at least a cluster bootstrap with a suitable alternative or design-based reference for each simulated regime.
- Adjust or hierarchically control multiple comparisons across horizons, categories, detector variants and anchor choices. Keep a frozen primary estimand.

### C. Informed-flow/anomaly diagnostic

- Under a preregistered synthetic null, target a family-wise false-positive rate no higher than 5% after accounting for the planned comparisons; report per-family and per-category uncertainty, not just a pooled point estimate.
- Report recall separately for each injected mechanism. A single aggregate recall hides regime-specific failure.
- Run sensitivity analysis to liquidity, event type, anchor policy, resolution class and source-lineage assumptions.
- Require prospective, time-forward shadow replication across at least three adequately sampled categories before any real-data superiority claim.
- Do not use the aspirational 5% relative Brier improvement gate as a detection or integrity metric. Predictive accuracy and integrity detection have different estimands.

### D. Evidence chain

- 100% rejection of all registered digest, predecessor, duplicate-ID, missing-artifact and malformed-record mutations.
- Once signature verification exists, 100% rejection of tampered signatures, unsupported algorithms, stale/revoked credentials, wrong keys and correct signatures held by principals lacking the exact resolver role.
- Keep structural consistency, byte integrity, cryptographic authenticity, role authorization and settlement permission as separate boolean/evidence-bearing gates, not weighted components of one score.

### E. Reproducibility and operational cost

- Identical frozen corpus, configuration, executable and seed must reproduce selected-row IDs, exclusions, score rows and interval outputs exactly where determinism is claimed.
- Every artifact binds the corpus digest, generator and version, taxonomy and scoring-policy versions, source/evidence digests, runtime/toolchain, seed, evaluator commit and the complete exclusion ledger.
- Report runtime median/p95, peak memory, corpus size, and replay throughput as separate measurements. Performance targets should be chosen only after obtaining a named-host baseline; no unmeasured latency claim is made here.

## Real-world validation gates

Synthetic qualification is required but insufficient. A real-data phase must remain offline/non-trading and use data collected lawfully with documented venue terms, retention rights, event-definition provenance, timestamps, and artifact digests. Use a frozen retrospective sample for development, then a later time-forward shadow sample for confirmation. Keep related markets and parent events on the same side of train/test boundaries.

Before interpreting results, publish coverage and exclusion-flow tables. If a large fraction of the chosen cohort is ambiguous, treat that as a study result and narrow the estimand. Do not inflate coverage with heuristics whose reliability has not been tested. Have an independent reviewer adjudicate a blinded sample and report inter-rater agreement, disagreements, and the rules used to resolve them. Do not announce or score named individuals as insiders using this research harness.

## Implementation sequence

1. **Corpus contract:** define the versioned resolution taxonomy, timestamp-anchor record, evidence interval, and exclusion vocabulary in a machine-readable fixture.
2. **Deterministic controls:** add clean/mutated contract and timestamp fixtures with expected classifications and invariants.
3. **Coverage-first report:** output a denominator-conserving breakdown by resolution class, category, horizon, timestamp-provenance tier, and exclusion reason.
4. **Uncertainty qualification:** run seeded coverage experiments against known distributions; compare cluster-bootstrap assumptions to declared data-generating processes.
5. **Cross-layer evidence:** integrate Mycelix's contract/digest/lineage and future signature/role outputs without merging their status into the forecasting metric.
6. **Prospective shadow study:** only after local synthetic qualification and independent provenance review; no live trading or settlement.

## What success means

The goal is not to maximize how many markets receive a suspicion label. It is to maximize the number of conclusions that are reproducible, scoped correctly, calibrated for their stated purpose, and reviewable—with a low and measured false-positive rate. Reporting that a method applies to only a narrow subset can be the correct result. A system that can say "not identifiable from this evidence" reliably is more useful than one that always returns a confident score.
