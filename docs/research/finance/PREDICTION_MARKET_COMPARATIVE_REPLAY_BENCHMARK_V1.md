# Prediction-Market Comparative Replay Benchmark v1

**Status:** proposed protocol; offline synthetic phase only  
**Related implementation:** [Symthaea PR #7613](https://github.com/Luminous-Dynamics/symthaea/pull/7613)  
**Work item:** [Symthaea issue #7630](https://github.com/Luminous-Dynamics/symthaea/issues/7630)  
**Evidence ceiling:** no real-market dataset, benchmark results, or production claim is included in this proposal.

## Research premise

A market-implied probability is not a horizon-free claim. Kalshi Research's August 2026 study reports aggregate Brier scores around 0.08–0.09 three months before resolution and roughly 0.02 at close for its resolved-market dataset (excluding Exotics). Those are results on that study's cohort, not portable acceptance thresholds for other venues or categories.

The May 2026 ForesightFlow paper reports that its initial proxy for public-information time reversed the intended separation in a matched-control comparison, and that none of 24 markets in its documented insider-case inventory fit the original scope conditions. The follow-on deadline-resolved study introduces an alternate scope and timestamp model; its reported result is a motivation to test competing operationalizations, not proof that one statistic establishes intent.

Sources:
- Kalshi Research, [Calibration in Prediction Markets — Theory and Evidence](https://kalshi.com/research/publications/calibration) (August 2026).
- Nechepurenko, [ForesightFlow: An Information Leakage Score Framework for Prediction Markets](https://arxiv.org/abs/2605.00493) (May 2026).
- Nechepurenko, [Empirical Evaluation of Deadline-Resolved Information Leakage on Documented Polymarket Insider Cases](https://arxiv.org/abs/2605.02286) (May 2026).
- Waghmare and Ziegel, [Proper Scoring Rules for Estimation and Forecast Evaluation](https://doi.org/10.1146/annurev-statistics-042424-050626) (2026).

## Protocol: what exactly is scored?

1. **Freeze event meaning.** Bind the event definition and resolution-policy version before evaluating forecast rows. A resolution outcome must follow that contract's declared semantics.
2. **Freeze lifecycle timestamps.** Store market creation, target horizon cutoff, relevant public-event time, market close, actual event occurrence (where defined), and official resolution separately. Never substitute one for another without an explicit policy.
3. **Select prequentially.** For each event, method, and horizon, select only the latest forecast observation at or before the frozen cutoff. If none exists, or the latest eligible observation exceeds the staleness bound, record an exclusion; do not select a future quote or fill missing data with zero.

The current synthetic smoke fixture uses actual elapsed-time cutoffs of 90 days, 30 days, 7 days, 1 day, and 1 hour before a synthetic resolution timestamp. The 1-hour horizon is not intended to stand in for a venue's actual market-close timestamp; real replay must preserve close, underlying event occurrence, public-news time and official resolution as separate fields.
4. **Pair comparisons.** Compare methods only on the same event/horizon targets and outcomes. Record every method's eligible coverage and the reasons for exclusions; report common-cohort and all-eligible results separately.
5. **Keep the event as the statistical unit.** Compute per-event/horizon Brier and log loss first. For uncertainty, aggregate repeated horizons within the declared independent unit (event or parent-event cluster) before resampling clusters.
6. **Do not silently tune on the test set.** Freeze the primary horizon, baseline, scoring method, event categories, exclusion logic, bootstrap seed/replicate policy, and comparison direction before viewing outcomes.

The current replay module selects observations and emits per-row Brier/log loss. Its paired cluster bootstrap computes candidate-minus-baseline Brier differences, averages rows within each declared cluster, then resamples clusters with replacement. Negative differences favor the candidate. Its bootstrap share below zero is descriptive and is **not a p-value**. A percentile interval is only as credible as the sampling design and the claim that clusters are appropriate; cluster labels cannot be inferred from market IDs alone.

## Benchmark matrix

| Dimension | Frozen synthetic control | Proposed qualification target |
|---|---|---|
| Horizon leakage | Add a high-quality forecast strictly after cutoff | 100% of post-cutoff forecasts excluded from that cutoff; score unchanged under any post-cutoff-only mutation |
| Staleness | Vary latest quote age around the configured limit | Exact boundary behavior; every stale case explicitly excluded |
| Duplicate records | Duplicate event/method/timestamp or event/horizon keys | Fail closed with stable structured error |
| Pairing | Mutate one method's eligible coverage | No method-to-method paired score unless event, horizon and outcome match |
| Cluster weighting | Give one cluster 10+ snapshots and another one snapshot | Cluster-level estimate equals the equally weighted cluster means, not row-weighted snapshot means |
| Bootstrap replay | Same immutable inputs, seed and replication count | Byte-identical summary across replay; seed and input manifest emitted |
| Source copying | Duplicate accounts within one declared source | Source-capped baseline invariant within (10^{-12}) where mathematically expected |
| Source misclassification | Split/merge source groups and hide common model lineage | Publish sensitivity envelope; never label the true source partition as inferred or authenticated |
| Public-time uncertainty | Shift source publication timestamps by ±1, ±5, ±30 minutes | Publish eligible coverage and per-category false-positive rates for each shift; exclude unsupported scope |
| Resolution contract | Mutate contract fields, digests, predecessor links and evidence times | Every represented mutation rejected; clean controls remain admitted |
| Market activity | Inject same-owner fills, repeated round trips, post-event activity and low-depth quote churn | Report exact synthetic accounting; don't equate gross volume with independent information or net risk transfer |
| Market quality | Paired quote/flow shock simulation | Report spread, depth, slippage, price impact, recovery time and cost separately |

The synthetic qualification design should include at least 12 preregistered threat/regime families and 1,000 deterministic seeds per family, with held-out seeds that were not used to tune detection thresholds. More seeds do not repair a misspecified synthetic generator; each family needs documented assumptions and at least one clean negative control.

## Scoring, inference, and reporting

For binary outcome (y_i \in \{0,1\}) and probability (p_i), use Brier score ((p_i-y_i)^2) and log loss (-[y_i\log(p_i)+(1-y_i)\log(1-p_i)]). Do not silently clamp probabilities for log loss; exact zero probability against an observed outcome yields infinite loss and should remain visible or be handled by an explicit, preregistered forecast policy.

Minimum report:
- Sample counts, resolved/void/missing outcomes, eligible forecast coverage, and exclusion counts by method and horizon.
- Brier score and log loss with event-/parent-event-clustered uncertainty.
- Equal-width reliability bins per method/horizon with counts, mean probabilities, observed frequencies, and explicit empty bins. The current harness emits these bins and a secondary bin-dependent weighted absolute calibration error; calibration uncertainty, intercept/slope, and Brier decomposition remain later-phase work.
- Results by horizon and category, plus all-category summary only as a secondary view.
- Common-cohort paired differences and sensitivity to inclusion/exclusion rules.
- All primary and secondary comparisons declared in advance; account for multiple comparisons when interpreting many horizons/models.
- Runtime median/p95, memory, and exact replay identity for the named environment.

Time-forward splits are mandatory for any trained/calibrated model. Keep related events and parent-event clusters in one split, and never randomly split snapshots of the same event across train/test. A real-data evaluation should proceed as a non-trading shadow run only after the synthetic harness and timestamp/event-definition provenance qualify.

## Provisional acceptance gates

These are targets, not measured results or external guarantees.

- **Synthetic temporal integrity:** zero post-cutoff forecast selections over the frozen corpus.
- **Synthetic structural validator:** 100% of enumerated negative controls rejected and all clean controls accepted.
- **Inference implementation:** deterministic replay for identical seed/manifest; repeated horizons are resampled by the declared cluster unit.
- **Predictive improvement:** do not claim superiority unless the preregistered paired cluster interval excludes zero in the candidate-favorable direction, sufficient coverage is met, and calibration/log-loss guardrails are not materially worse. A proposed aspirational research gate is at least 5% relative Brier improvement, repeated across at least three sufficiently sampled categories; it must not become a success criterion selected after inspecting results.
- **Timing diagnostics:** no allegation or probability of insider trading is inferred from the statistic alone. Report timestamp-source provenance, scope coverage and matched-control false positives.

## Replay manifest

Every run should bind at minimum:
- corpus/version and each source record's digest;
- immutable contract digest and outcome/resolution policy;
- event/parent-cluster IDs and cluster rationale;
- method/version IDs, observations, timezone-normalized timestamps and source provenance;
- frozen horizon definitions and max-staleness limits;
- exclusions and reasons, selected forecast row IDs, per-event scores;
- evaluator commit, runtime/toolchain, configuration, seed, replicate count and bootstrap interval method.

The current smoke fixture prints a minimal metadata manifest containing the synthetic corpus version, event count, parent-cluster rule, actual horizon seconds, staleness limit, bootstrap seed/replicate policy, deterministic generator salts/rules and the exact source commit supplied by the CI workflow. It does not yet emit a content-addressed archive of every generated row, selected observation, exclusion, or source input; those remain acceptance requirements for the next corpus increment.

The manifest proves what this evaluator claims to have processed, not that source data or event outcomes are true. Source authentication, signatures, authority, external-world truth, and any regulated action remain separate assurance problems.
