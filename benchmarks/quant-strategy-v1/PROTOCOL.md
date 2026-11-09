# Quantitative Strategy Evaluation Protocol v1

**Status:** research protocol and synthetic manifest foundation only  
**Parent:** [SYM-QUANT-EVAL-001 #11234](https://github.com/Luminous-Dynamics/symthaea/issues/11234)  
**Companion:** [SYM-FIN-EVAL-002 / PR #11233](https://github.com/Luminous-Dynamics/symthaea/pull/11233)
**Shared temporal contract:** [SYM-FIN-TIME-001 #11241](https://github.com/Luminous-Dynamics/symthaea/issues/11241) — use the same bitemporal availability and revision semantics as financial research; do not reimplement a separate timestamp policy.
**Evaluator/reference metrology:** [FIN-EVAL-METROLOGY-001 #11254](https://github.com/Luminous-Dynamics/symthaea/issues/11254) — use only for analyst/judge-scored workflow comparisons; deterministic portfolio accounting oracles remain independently specified.
**Portfolio transition oracle:** [SYM-QUANT-ORACLE-001 #11298](https://github.com/Luminous-Dynamics/symthaea/issues/11298) — deterministic benchmark-only order/fill/position/cash/fee/corporate-action reference calculations; reuse Mycelix exact-money/accounting contracts, do not add a second production ledger or broker authority.

## 1. Research question

Can a Symthaea-assisted research process discover a strategy/policy that improves net, risk-adjusted, capacity-aware outcomes over simple and strong systematic baselines on point-in-time information and genuinely unseen market periods, without overfitting, violating mandates or hiding execution costs?

A forecast metric is not a portfolio result. A high in-sample Sharpe is not an investable edge. The benchmark evaluates the full chain:

`information available at t -> signal/forecast -> constrained portfolio -> orders/fills -> costs -> net exposure/returns -> risk/outcomes`

No live capital, broker writes or autonomous financial authority are in scope.

## 2. Industry and research comparators

- **QuantConnect LEAN** is an established open-source event-driven streaming algorithm engine. Its time frontier emits data only after the relevant bar/time is available, explicitly addressing look-ahead errors. Use it as an execution/backtest comparator or integration target where its licence, data and infrastructure are suitable; do not recreate its general engine without a demonstrated reason.
  - https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/time-modeling/timeslices
  - https://www.quantconnect.com/docs/v1/key-concepts/understanding-time
  - https://github.com/QuantConnect/Lean
- **FinRL Contests 2023–2025** cover stock, crypto, execution and LLM-signal tasks; the 2025 paper emphasizes market non-stationarity, low signal-to-noise, frictions, data quality, survivorship and overfitting as persistent barriers.
  - https://doi.org/10.1049/aie2.12004
  - https://github.com/AI4Finance-Foundation/FinRL
- **Probability of Backtest Overfitting (PBO) / CSCV**: use a multiple-trial overfit diagnostic. Conventional holdout alone can be unreliable after a large number of strategy trials.
  - https://doi.org/10.21314/jcf.2016.322
  - https://escholarship.org/uc/item/4w1110bb
- **Deflated Sharpe Ratio**: interpret selected Sharpe values with multiple testing and non-normality accounted for.
  - https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- General financial research-agent benchmarks in #11233 are complementary. They do not qualify strategy returns, capacity or portfolio risk.

None of these sources establishes one universal trading SOTA. All comparative results must name exact engine/model/data/version, task, cost model and period.

## 3. Paradigm shifts

1. **Forecast-to-fill evaluation.** Evaluate the complete portfolio and execution path, not only direction accuracy or forecast loss.
2. **Availability-time first.** Features, index constituents, filings, macro releases, corporate actions and revisions must be admissible at the exact decision time. Store event/valid time separately from known/available time.
3. **Pay for strategy search.** Count every tried model, feature, universe, seed, window, parameter set and policy. Keep a trial ledger; compute PBO/CSCV and DSR/PSR with that search exposure visible.
4. **Real costs and capacity.** Include spread, fees, slippage, impact, turnover, liquidity, participation, funding, borrow, FX, taxes and settlement where applicable. Unknown cost is not zero.
5. **Held-out regimes.** Separate discovery from confirmation using chronological, purged walk-forward splits and issuer/source/market/period holdouts. Include regime shifts and stress scenarios.
6. **Mechanistic hypotheses.** Explore whether filings, supplier/customer relationships, energy, logistics, macro, geography and physical assets add useful signals. Every link remains an association or hypothesis until appropriately identified; graph proximity is not causality.
7. **Architecture humility.** HDC, CfC/LTC, predictive processing and multi-agent research are candidates to ablate against simpler methods, not presumed advantages. Include controls where linear/rule-based baselines should win.
8. **Abstention is a policy.** Permit hold/cash/reduced risk/no-trade when evidence is stale, signal is weak, market regime is outside support, uncertainty is high or liquidity is inadequate.
9. **Capacity is an outcome.** Report minimum viable and maximum practical capital, participation and liquidity strata; small illiquid-market alpha is not scalable by default.
10. **No universal scalar.** Preserve return, risk, capacity, liquidity, cost, uncertainty, robustness and operational integrity as separate outcomes.

## 3a. Temporal admissibility and evaluation measurement

Use the shared temporal semantics in #11241 for every market observation, corporate-action revision, filing, macro vintage, constituent membership and feature source. The decision frontier is an inclusive upper bound; availability time is actor/entitlement-specific; retrieval time is distinct from source availability.

For historical replay, later retrieval is admissible only when the exact vintage and its historical availability are independently evidenced. For prospective streaming, both availability and local ingestion must occur by the decision frontier. A backdated effective date is not a substitute for known-at time. Preserve revised and superseded versions so the evaluator can reconstruct what was visible at each frontier.

Use deterministic reference oracles for cash, positions, returns, costs, fills and risk metrics wherever exact ground truth is available. For human- or LLM-judged research workflows, qualify references and judges under #11254; do not let an LLM judge replace accounting arithmetic or execution reconciliation.

### Source timestamp precision

Do not coerce date-only vintage metadata (including macro-series vintage dates or filingDate) to midnight for intraday backtests. The shared v0.1 kernel (#11241 / PR #11253) requires exact offset-aware availability instants; coarse or ambiguous timestamps remain indeterminate until the interval-censored extension in [SYM-FIN-TIME-002 #11256](https://github.com/Luminous-Dynamics/symthaea/issues/11256) proves the entire possible-availability interval is before the frontier.

## 4. Cohorts and split design

Keep result cohorts separate:

- **S0 synthetic/reference:** known streams and hand-checkable policies; validate timing, accounting, fills and metric implementations.
- **S1 public development history:** lawful, rights-cleared data; development only.
- **S2 historical confirmation:** frozen issuer/security/time/source-family holdouts, inaccessible during search.
- **S3 regime/market transfer:** untouched time periods, regimes, asset classes, currency or jurisdictions.
- **S4 prospective shadow:** timestamped signals and decisions before outcomes; no live order/capital effect.
- **S5 matched analyst/quant comparison:** same permitted information, risk mandate and time budget.

Use nested chronological walk-forward evaluation. Apply purging/embargo where labels, holding periods or overlapping outcomes require it. Preserve point-in-time index membership, delisted securities, listing changes, corporate actions, filing release/revision times, market calendars and currency conversion. Do not random-split time-series rows. Do not drop failed/delisted strategies or securities without a predeclared reason.

## 5. Baseline tournament

At minimum compare:
- cash / no-signal control;
- appropriate market benchmark and buy-and-hold;
- market-cap weighting and equal weighting;
- declared 60/40 or policy portfolio, where appropriate;
- simple trend, momentum/time-series momentum and cross-sectional momentum;
- source-available value, quality, low-volatility or other factor profiles;
- regularized linear/statistical model;
- tree/boosting ML;
- deep sequence/time-series model where justified;
- FinRL/RL baseline only where task and environment semantics match;
- human quant and human+tools, under matched evidence/time budgets;
- non-deployable hindsight oracle on tiny synthetic worlds only, as an upper bound.

Pin each baseline's universe, data, features, leverage/risk limits, capital, search trials, compute and execution model. Include simple-model-wins fixtures.

## 6. Symthaea experiments and ablations

Progressively test:
1. rule/factor portfolio;
2. statistical forecast + fixed portfolio policy;
3. supervised ML forecast + same fixed policy;
4. Symthaea research/event/fundamental signals + same policy;
5. HDC associative/event representation;
6. CfC/LTC temporal dynamics;
7. fused ensemble or regime-conditioned candidate;
8. constrained portfolio construction;
9. adaptive/RL policy only after S0 execution and accounting controls pass.

Change one factor at a time in ablations. A more complex architecture must demonstrate incremental value after all trial and compute costs. Retain the simpler baseline if the candidate fails.

## 7. Required input and run identity

A run must bind:
- exact universe/constituent profile and its point-in-time membership evidence;
- instrument identifiers, listings/delistings and corporate-action profile;
- feature and source manifests with artifact IDs, digests, licences/rights, source-family, valid/event time, known/availability time, retrieval time and revisions;
- decision timezone/calendar and event-stream time frontier;
- training, normalization, feature-fit and target-label cutoffs;
- strategy code/model, feature set, policy/config and all search-trial identities;
- chronological split/holdout identities, purge and embargo policy;
- mandate, capital, currency, leverage, concentration and liquidity constraints;
- fill, spread, slippage, impact, fee, borrow, funding, FX and tax profiles (unknowns explicit);
- seed, runtime, tool versions, output digest and disposition.

Use provider-neutral Mycelix identities and observations; do not add a competing security, position, order, fill, ledger or settlement ontology here.

## 8. Score vector

### Return and benchmark
Cumulative/geometric and annualized return; net excess versus a declared investable benchmark; gross and net values; factor exposures and any alpha estimate with its stated identification assumptions.

### Risk and tails
Volatility, Sharpe and Sortino with uncertainty; Calmar; maximum drawdown, duration and recovery; VaR and expected shortfall at declared 95%/99%; worst day/week/month; leverage, concentration, gap and liquidity exposures.

### Trading and capacity
Turnover, trade count, holding period, spread, fees, slippage and impact separately; financing/borrow/FX/tax costs; participation and capacity curves over capital; fill/partial-fill/reject behavior under declared models.

### Research validity and efficiency
Point-in-time coverage, leakage count, universe/survivorship controls, PBO/CSCV, DSR/PSR, effective trial count, walk-forward and held-out-regime returns, forecast calibration, signal decay, parameter sensitivity, compute/time/cost, per-stratum dispersion and reproducibility digests.

No candidate can compensate for a hard-gate failure with high returns or Sharpe.

## 9. Non-compensable gates

- Future price, filing, revision, constituent or macro release leaks invalidate the affected run.
- Survivorship or delisting omissions invalidate universe-relative comparisons.
- Corporate-action or unit/currency errors invalidate affected outcomes.
- Gross returns cannot be presented as net.
- Missing friction assumptions cannot be silently set to zero.
- Hidden leverage, undefined downside or breach of the declared risk/mandate profile fails.
- A strategy must not be selected on the final confirmatory cohort.
- Trials, failures, timeouts and candidate exclusions remain in the research ledger.
- A forecast, backtest or simulation cannot create a financial action or authority.
- Read-only/proposal-only Symthaea profiles expose no provider write capability.
- No live account, broker, custody or trading integration is permitted by this benchmark.

## 9a. Portfolio transition reference oracle

Use [SYM-QUANT-ORACLE-001 #11298](https://github.com/Luminous-Dynamics/symthaea/issues/11298) for the deterministic reference path from declared orders and fills to cash, positions, realized/unrealized P&L, costs and portfolio value. This is a benchmark-only reference calculation, not a second production ledger or order-management system.

Compose with Mycelix [FIN-ACC-IMPL-001 #5789](https://github.com/Luminous-Dynamics/mycelix/issues/5789) and [FIN-ACC-BENCH-001 #5782](https://github.com/Luminous-Dynamics/mycelix/issues/5782) for exact money and accounting reference semantics. The quant oracle must not invent another Money type, canonical position ontology, journal, broker adapter or settlement authority.

A qualified net-return result requires explicit fees, spread, slippage, market impact, funding/borrow and other applicable costs. Null/unknown costs are not zero. The oracle must separately identify submitted orders, accepted orders, fills, marks and corporate-action transformations; a price mark cannot create a fill, and a submitted order cannot create a position.

Until #11298 is implemented and independently checked, the current synthetic zero-friction example remains a schema/integrity fixture only—not a historical strategy-performance result.

## 10. Initial vertical slice

1. Freeze a deterministic synthetic market stream with known event/available times, constituents, cash, holdings, fees and corporate actions.
2. Implement a tiny independent portfolio-accounting oracle and test cash/holdings/fees/value conservation.
3. Compare cash, buy-and-hold and equal-weight baselines.
4. Add an event-stream feature time frontier. Negative controls must reject future prices, late filings, revised macro values known only later, and future constituents.
5. Add a simple friction sensitivity envelope: explicit fee, spread, slippage and market-impact profiles; no idealized fill by default.
6. Freeze chronological walk-forward/purge/embargo manifests and the trial-count ledger.
7. Emit per-run vector metrics and a reproducible receipt, then add PBO/CSCV and DSR/PSR.
8. Only after S0 passes, connect one lawful historical dataset adapter or a compatible existing event-driven engine. Real-data permission is a prerequisite, not a post-hoc caveat.

## 10a. Digest semantics for the synthetic reference

For this first fixture, `universe.membership_sha256` is SHA-256 over the exact UTF-8 bytes of `synthetic-universe.json`, including the trailing newline. `data_frontier.source_manifest_sha256` is SHA-256 over the exact UTF-8 bytes of `synthetic-market-data.json`, including the trailing newline. Do not normalize whitespace or line endings before hashing. These are synthetic content digests, not evidence of real-market data quality. Historical source adapters must declare their own content-addressing and normalization profile.

## 11. Provisional hypotheses — not achieved results

After development baselines and confirmatory splits are frozen, test if a candidate can:
- generate positive net excess return with uncertainty intervals against its declared benchmark on confirmatory periods;
- improve a preregistered risk-adjusted objective without worsening max drawdown, expected shortfall, leverage or liquidity beyond declared limits;
- retain value on at least one untouched period and regime;
- demonstrate honest uncertainty and no-trade behavior under drift/low-quality evidence;
- remain persuasive after PBO/DSR and trial-count correction;
- improve capacity-adjusted net performance at more than one capital level.

Do not choose a universal Sharpe/CAGR/PBO cutoff before measuring baseline dispersion, trial count, sample size and statistical power. Freeze primary endpoints and thresholds before opening confirmatory data.

## 12. Luminous responsibility boundaries

- **Symthaea:** hypotheses, features, forecasts, risk analysis, portfolio candidates and uncertainty.
- **Mycelix:** instrument/position/accounting/execution semantics, evidence, rights, authority and any future provider boundary.
- **Symtropy:** bounded economic/physical stress scenarios; simulated outputs remain distinct from observed state.
- **Sol-Atlas:** spatial infrastructure, real assets, supply-chain, energy and regional exposure overlays; no financial state authority.

## Claim ceiling

A PASS applies only to the named strategy, universe, time frontier, data rights, friction model, split, benchmarks, cost and regimes. It does not establish future profitability, durable alpha, investment-adviser competence, regulatory compliance, best execution or superiority to any named industry platform outside the exact matched evaluation.
