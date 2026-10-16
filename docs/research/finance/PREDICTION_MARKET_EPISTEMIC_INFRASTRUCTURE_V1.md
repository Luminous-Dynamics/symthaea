# Prediction-Market Epistemic Infrastructure v1

**Status:** proposal and synthetic research agenda  
**Owner boundary:** Symthaea evaluates forecasts and evidence; Mycelix may later bind issuer, provenance, dispute and audit receipts; neither system receives authority to trade or settle a live market from this document.  
**Evidence ceiling:** no real venue dataset is included, no market is connected, and none of the targets below has been measured.

## Thesis

A prediction market should not expose a price as if it were a complete measure of truth. It should expose a typed evidence vector:

- market-implied probability and quote timestamp;
- prospective forecast calibration and score uncertainty;
- source concentration and dependence uncertainty;
- spread, depth, slippage and recovery after shocks;
- gross turnover versus same-owner turnover;
- event-definition and resolution-policy completeness;
- oracle-source, adjudication and appeal provenance;
- known conflicts, missingness and policy limitations.

Do not compress these axes into one “trust score.” A market may be liquid but poorly calibrated; calibrated historically but stale now; active but source-concentrated; well specified but thinly traded. These states require different remedies.

## Why now

Publicly reported concerns about repetitive or potentially self-matched prediction-market activity illustrate why displayed volume is not equivalent to independent information or meaningful risk transfer. The CFTC's 2026 advance notice of proposed rulemaking also emphasizes that event-contract categories, public-interest questions and information—including insider information—need explicit treatment.

References:

- Jane Street, Overview: https://www.janestreet.com/what-we-do/overview/
- Waghmare and Ziegel, Proper Scoring Rules for Estimation and Forecast Evaluation, Annual Review of Statistics and Its Application (2026): https://doi.org/10.1146/annurev-statistics-042424-050626
- Neyman and Roughgarden, From Proper Scoring Rules to Max-Min Optimal Forecast Aggregation: https://arxiv.org/abs/2102.07081
- CFTC, Prediction Markets, ANPRM, Federal Register 91 FR 12516 (March 16, 2026): https://www.cftc.gov/LawRegulation/FederalRegister/proposedrules/2026-05105.html
- CFTC, Understanding Prediction Markets and Event Contracts: https://www.cftc.gov/LearnandProtect/PredictionMarkets
- WIRED, reporting on repetitive trades in election markets (October 15, 2026): https://www.wired.com/story/over-dollar600-million-traded-on-polymarket-election-contracts-after-races-ended
- Reuters, *Inside the booming market for US midterm election bets* (October 14, 2026): https://www.reuters.com/legal/government/inside-booming-market-us-midterm-election-bets-2026-10-14/
- ForesightFlow, *An Information Leakage Score Framework for Prediction Markets* (May 1, 2026): https://arxiv.org/abs/2605.00493
- Kalshi Research, *Calibration In Prediction Markets – Theory and Evidence* (August 2026): https://kalshi.com/research/publications/calibration

These references motivate controls; they do not establish that every venue has the same failure modes.

## Paradigm shifts

### 1. Reward marginal information, not activity

Separate four cash-flow concepts: payment for forecast accuracy, compensation for making liquidity available, trading P&L, and any governance/reputation reward. They are not the same thing.

A forecaster's accuracy rewards should use a declared proper scoring rule (for example Brier or log score for binary outcomes), predeclared horizons, strict outcome-resolution semantics, and prospective evaluation. Score the forecast that existed before the outcome, not a retrospectively edited prediction. Scores are evidence about forecasting skill under that protocol—not a universal reputation for truth.

Do not pay forecast rewards per transaction. That creates an incentive to create trades rather than information. Any market-maker program should be separately assessed for spread, depth, uptime, inventory exposure, toxic flow, and public cost.

### 2. Reward independent evidence, not account multiplication

Expose at least both account-weighted and dependence-adjusted forecast aggregates. Grouping can be based on declared common source, shared model lineage, common data release, copy relationships, or other evidence—but a group label must not self-authenticate. Where lineage is unknown, report the uncertainty rather than inventing independence.

The current synthetic crate deliberately provides only an equal-source-group sensitivity baseline. It does not determine the best aggregator. A later comparison should test linear pooling, logarithmic pooling, robust aggregation, market price and model ensembles on the same prospective outcomes. Choose aggregation by a frozen scoring protocol, not ideology or a preferred model.

### 3. Separate volume from economic risk transfer

Report gross notional, same-beneficial-owner notional, distinct counterparty pairs, open-interest changes, and concentration where qualified data permits. Same-owner fills are a useful exact negative control when ownership attribution is sound. They do not detect every wash-trading pattern; distinct-owner round trips require more history and evidence. A volume-adjusted risk metric must never be labeled verified from raw trades alone.

### 4. Make resolution part of the contract

Before trading starts, freeze the precise question, primary source, source-failure/void rule, deadline, source-revision rule, dispute window, and adjudication/appeal policy. Commit to the exact bytes and version with an independently verifiable receipt.

A complete policy can still be wrong, ambiguous or legally impermissible. The integrity check must not decide whether a real event occurred or give its operator unilateral authority.

### 5. Make uncertainty and conflicts first-class

Represent at least these distinct states: observed, estimated, derived, forecast, disputed and unknown. Make model disagreement visible. Mark insider-sensitive and personally harmful event categories for human/legal review rather than assuming that more markets are always desirable. Use transparent participation and surveillance rules with due process; avoid pretending that a public pseudonym alone proves either a person's legal identity or their informational independence.

### 6. Use conditional and cross-market coherence as an audit

Where contracts have explicit logical relationships, check constraints such as mutually exclusive outcomes summing to one, nested-event probabilities, and conditional/event-probability consistency. A detected inconsistency is a diagnostic—not automatically an arbitrage after fees, limits, settlement timing and liquidity are accounted for. Keep event identity and resolution semantics aligned before comparing prices across venues.

## Symthaea/Mycelix division of responsibility

**Symthaea** owns the theory-neutral forecasting side: candidate model outputs, base-rate and naïve baselines, proper scoring rules, calibration, distributional uncertainty, dependence sensitivity, out-of-sample comparison and model/version provenance. Its existing Futures crates should be reused instead of duplicating forecast types or scoring formulas.

**Mycelix** is a plausible future evidence and coordination layer for: versioned event/contract commitments, source attestations, signer/issuer credentials, dispute events, conflict disclosures, independent review receipts and audit lineage. Identity, issuer authority and source authenticity must be separately verified. A DHT entry, content digest or reputation score alone is not external-world truth.

**Boundary:** neither a high forecast score nor a market price should directly trigger payment, policy, enforcement or trading. Any later action adapter needs separately specified authorization, risk controls and human/legal requirements.

## Synthetic audit crate in this increment

The accompanying pure-std Rust crate implements only:

- probability input validation;
- account-weighted versus declared-source-capped means;
- invariant behavior when the same declared source is copied to many account IDs;
- gross versus same-owner turnover with duplicate-fill rejection;
- resolution-policy completeness with missing-field, deadline and dispute-window controls.

The crate does not ingest platform feeds or prove source/ownership labels. The source-capped mean is one comparison baseline, not a claim of optimal information aggregation.

## Source-partition sensitivity: assumptions should be visible

The new `assess_source_partition_sensitivity` API computes the baseline source-capped mean and reevaluates it under explicitly supplied alternative mappings from accounts to source groups. Every alternative must map exactly the same accounts, contain nonblank group identifiers, and have a unique scenario ID. The result reports the observed minimum, maximum and range across the baseline and alternatives.

This is not a confidence interval, an automated inference of the true source graph, or proof of manipulation. It is an audit of how much the estimate depends on group labels that the crate cannot authenticate. A synthetic test with 100 accounts at probability 0.99 declared as one common source and one other account at 0.20 has a source-capped baseline of 0.595; splitting every account into its own declared group produces an aggregate above 0.98. The large change is the point of the negative control: an unauthenticated grouping label can dominate the statistic.

A real evaluation should preregister plausible grouping alternatives using evidence such as common upstream publication, shared model/data lineage, copied forecast content, or shared release timing. Preserve “unknown” group relationships and publish the sensitivity envelope rather than choosing whichever partition makes a model look best.

## Public-information timing: a diagnostic, not an accusation

The new `assess_pre_public_information_move` function tests a narrow question: given a qualified public-information timestamp and a binary realized outcome, what signed fraction of the terminal outcome-aligned probability move was already present before that timestamp?

The function:
- requires strictly increasing timestamps and validated probabilities;
- requires the public timestamp to lie inside a window with a declared edge-exclusion margin;
- excludes weak/adverse terminal movements rather than manufacturing a score;
- preserves scores below zero or above one instead of clamping away overshoot or reversal;
- returns explicit exclusion reasons and labels no result as insider probability.

This is motivated by ForesightFlow, but it is deliberately not presented as a replication or validated implementation of that paper. ForesightFlow reports that its initial proxy for public-event time reversed the intended separation in a matched-control comparison and that its original scope missed the documented insider cases it reviewed. The engineering lesson is that **timestamp provenance, market/event selection, terminal-move size and category-specific baselines are prerequisites, not details to bolt on later**. See https://arxiv.org/abs/2605.00493.

The new `assess_sampled_timestamp_sensitivity` helper evaluates a strictly increasing, non-empty sample of candidate public timestamps and reports eligible coverage, exclusion counts, and the range of the pre-event-move fraction. It does not replace excluded cases with zero. This is a sample sensitivity summary, not a bound over every possible timestamp in an interval; the candidate grid and timestamp uncertainty must themselves be documented.

A future evaluation should compare timestamp sources and inject controlled timestamp shifts (for example ±1 minute, ±5 minutes, and ±30 minutes) into synthetic fixtures, then measure score instability. For real data, report timestamp uncertainty, edge-exclusion coverage, false-positive rates on matched controls, and results by event-resolution category. The diagnostic must never be marketed as proof of insider trading, manipulation or causality.

## Comparative replay protocol (v2)

The implemented prequential evaluator lives in `crates/domains/symthaea-futures-market-integrity/src/replay.rs`. It selects the latest observation at or before a frozen target cutoff, excludes missing/stale data explicitly, and scores each event/horizon instead of pooling all snapshots.

The same module now includes a deterministic paired cluster bootstrap over candidate-minus-baseline Brier differences. Repeated rows within the same declared event/parent-event cluster are averaged before clusters are resampled. This addresses one common pseudoreplication failure, but it does not repair bad cluster definitions, selection bias, or invalid timestamps. The bootstrap share of samples below zero is not a p-value; score intervals and primary horizons must be preregistered.

The module also emits equal-width reliability bins through `calibration_report`, scoped to one method/horizon and preserving empty bins, sample counts, mean forecast probabilities and observed frequencies. A weighted absolute calibration error is reported only as a secondary bin-dependent statistic; it does not replace the bin table, Brier score, log loss or calibration uncertainty. The synthetic example emits per-horizon bins and a paired common-cohort summary so eligible-set averages are not confused with paired comparison results.

See [PREDICTION_MARKET_COMPARATIVE_REPLAY_BENCHMARK_V1.md](PREDICTION_MARKET_COMPARATIVE_REPLAY_BENCHMARK_V1.md) for the full replay estimand, adversarial corpus design, inference/reporting protocol and provisional gates.

## Benchmark plan

All numeric thresholds below are proposed acceptance targets. They are not measured results, current platform metrics, or evidence of superiority.

| Dimension | Benchmark design | Proposed acceptance target |
|---|---|---|
| Proper scoring | Paired, time-ordered prospective outcomes; compare market close, base-rate, equal-account pool, declared-source-capped pool, and Symthaea ensemble | Publish Brier and log score with uncertainty intervals for every baseline; no superiority claim unless a preregistered paired comparison clears zero after stated costs |
| Calibration | Reliability diagrams, calibration intercept/slope, Brier decomposition, log score, and regime-specific slices | No aggregate-only gate; investigate any well-powered regime with absolute calibration error above 0.03; publish bin counts and intervals |
| Source-copy resistance | Duplicate 1, 10, 100, and 1,000 accounts under the same declared source group | Source-capped baseline is invariant to duplication within numerical tolerance (1e-12 in the synthetic test) |
| Source-partition uncertainty | Compare baseline with alternate account-to-source mappings, including splitting a copied source into 1/10/100/1,000 declared groups | Publish min/max/range and grouping assumptions; missing/extra accounts, blank group labels and duplicate scenario IDs fail closed; do not interpret the range as a confidence interval |
| Dependence misspecification | Shift or falsify declared source groups; hide common model/data lineage in mutation tests | Explicitly demonstrate where the detector fails; never call source independence verified from account counts |
| Same-owner turnover | Known owner-level synthetic fills, reverse ordering, duplicates, and clean controls | 100% exact accounting on the synthetic corpus; duplicate fill IDs fail closed; report this as a lower-bound alert only |
| Resolution completeness | Delete or mutate each required policy field independently | Every missing/blank required field and zero deadline/dispute-window mutation is rejected |
| Cross-market coherence | Freeze synthetic nested, exclusive and conditional event graphs; add inconsistent quote mutations | 100% of injected graph violations identified on the frozen corpus; false positive rate reported on clean controls |
| Manipulation stress | Synthetic self-match, spoof-like quote churn, stale quote, correlated-source herd, oracle delay/conflict, and low-depth shock | Publish precision/recall by threat class; avoid one blended accuracy number; disclose detection blind spots |
| Public-information timing | Compare qualified publication timestamps with alternate/proxy timestamps; include event-category matched controls and source-time shifts | Publish scope coverage, eligible fraction, observed fraction range under ±1/±5/±30-minute shifts, and false-positive rates; no accusation or "insider probability" claim from this score alone |
| Market quality | Spread, depth, slippage, price impact, time to recover after shock, stale-quote rate, and market-maker inventory risk | Paired comparison on identical event/order flow and shock seeds; net improvements only after costs and uncertainty |
| Robustness | At least 12 preregistered regime families, 1,000 deterministic seeds per family for simulation phase; separate held-out seeds and event definitions | Exact replay, seed manifests, frozen baselines, zero leakage; synthetic passes do not count as real-world validation |
| Reproducibility | Bind code revision, config, source hashes, outcome semantics, timestamps, and raw score records | Replay regenerates all aggregate metrics exactly or documents a bounded numerical tolerance; changed inputs change the result identity |
| Runtime | Public reproducible benchmark for 10k, 100k and 1M signal/fill records on named hardware | Record median, p95, peak memory and scaling; set deployment SLO only after first measurement rather than guessing |

A future prospective evaluation should run long enough to include at least one full resolution cycle for each chosen domain. Report selection criteria, exclusions, missing/void outcomes, transaction costs and any researcher's financial conflicts before looking at results. Use time-based splits; do not randomly split rows from the same event into training and test sets.

## Adversarial corpus to add next

1. Source replication: one source posted by many independent-looking accounts.
2. Grouping evasion: the same content is split across declared sources.
3. Common-model dependence: nominally distinct forecasters share one upstream model.
4. Same-owner matched volume with zero owner-level position change.
5. Distinct-owner round trips that defeat the simple same-owner detector.
6. Duplicate/reordered fills and event replay.
7. Thin market with large price impact despite low notional.
8. High volume and little depth, or apparently high depth that disappears under shock.
9. Primary source corrected after settlement, delayed publication, or conflicting official sources.
10. Vague event wording where reasonable readers disagree on resolution.
11. A resolution operator with a material conflict of interest.
12. A manipulable metric where paying for reported liquidity increases quote spam but not executable depth.
13. Shared source/model regime shift causing multiple forecasters to fail together.
14. Conditional and mutually exclusive event contracts with intentionally incoherent prices.
15. Missing, stale or retrospectively backfilled data masquerading as live knowledge.

Every attack family needs positive controls, negative controls, expected output, a frozen fixture identity, and an independent verifier. Unknown should remain unknown rather than coerced to zero or success.

## Benchmarks for competing with elite market-making technology

Jane Street describes itself as a research-driven liquidity provider active across more than 200 electronic venues. Matching such a firm in execution, capital, staffing, operational maturity or latency is not a credible initial target for a small research program.

A differentiated target is possible: an epistemic market assurance tool that detects source concentration, distinguishes turnover from qualified risk transfer, exposes calibration and uncertainty, catches event-semantic inconsistencies, and produces replayable evidence receipts. Compete on an independently demonstrated capability, not on a slogan.

To make stronger claims, measure end-to-end latency, slippage, availability, adverse-selection cost, inventory risk, prospective calibration, incident recovery and regulatory/control compliance against named baselines. There is no defensible basis yet for saying this work is more profitable or generally more capable than Jane Street.

## Stage gates

- **G0 — structural harness:** all unit/adversarial tests execute on the exact commit; APIs and claim ceilings reviewed.
- **G1 — frozen synthetic corpus:** seed replay, mutation corpus, separate reference verifier and benchmark manifest.
- **G2 — historical replay:** use lawfully obtained and versioned public market data; freeze event definitions and time availability, reproduce market prices as they would have been known then, include voided and ambiguous markets.
- **G3 — prospective shadow mode:** publish timestamped forecasts and diagnostics without trading or influencing settlement; preserve all misses.
- **G4 — independent review:** separate reviewer, conflict disclosure, source/identity threat model, and legal review for the target jurisdiction.
- **G5 — limited real-world pilot:** only after operational, security, legal, market-conduct and participant-protection requirements are independently reviewed. Nothing in this PR qualifies that stage.

## Nonclaims

This increment does not establish accurate live probabilities, profitable trading, successful manipulation detection on real venues, identity/source authenticity, independent information, legal compliance, suitability of an event market, a superior market mechanism, or parity with an established trading firm.
