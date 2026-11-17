# EGDDM ShiftBench D6.17.2.3-alpha — End-to-end cost receipts and runner-tail measurement

**Date:** 2026-11-17  
**Status:** run-receipt validator prototype; synthetic anti-correlated known-answer fixture; Rust execution pending  
**Related:** [D6.17.2.3 issue #12631](https://github.com/Luminous-Dynamics/symthaea/issues/12631), [D6.17.2.1 exact adaptive-policy PR #12649](https://github.com/Luminous-Dynamics/symthaea/pull/12649), [D6.17.2.2 joint stochastic-kernel PR #12655](https://github.com/Luminous-Dynamics/symthaea/pull/12655).

## Why this matters

An adaptive acquisition policy can look cheaper in a synthetic planner while being more expensive in production because it ignores queue time, retries, warm/cold effects, timeouts, unknown outcomes, provider billing and correlated run overhead.

There is also a statistical error we must avoid: **the sum of marginal per-option p95 values is not generally the p95 of the total trajectory cost**. Quantiles are not additive. Without stronger assumptions about joint dependence, a sum of separate 95th-percentile component bounds only yields the union-bound statement:

\[
P\left(\sum_{i=1}^n X_i\le \sum_{i=1}^n q_{i,0.95}\right)
\ge 1-0.05n,
\]

where \(q_{i,0.95}\) is a marginal upper 95th-percentile threshold for component \(i\). For two attempts, that lower bound is 90%, not 95%. For enough attempts it becomes vacuous. The sum remains a descriptive comparator, but it must not be reported as a full-run p95 guarantee.

D6.17.2.3 records the complete trajectory and calculates full-run cost/time quantiles directly over complete runs. It reports the sum of marginal per-option p95 values in a separate field with an explicit warning.

## Research context and novelty boundary

The field has mature approaches to optimal experimental design and sequential acquisition. Huan, Jagalur and Marzouk's 2024 *Acta Numerica* review surveys Bayesian and decision-theoretic OED, combinatorial observation selection and sequential non-myopic design. Our goal is not to claim we invented adaptive evidence acquisition; it is to make its deployment cost and evidence trail independently measurable. [Optimal experimental design: Formulations and computations](https://doi.org/10.1017/S0962492924000023).

The 2026 review *From benchmarks to deployment* discusses gaps in agentic-AI evaluation around safety, security, cost efficiency and end-to-end workflow reliability. Its survey covers a bounded sample of benchmarks, not every evaluation programme. The practical lesson for EGDDM is to report execution cost and evidence fidelity as primary measures rather than assuming they follow from task accuracy. [Review](https://doi.org/10.1007/s10462-026-11571-0).

NIST's 2026 *Building Evaluation Probes into Agentic AI* programme develops adversarial verifiers and machine-readable evidence trails. D6.17.2.3 complements this direction with an append-only run/attempt receipt, evaluator-approved execution context, replayable hash chain and explicit unknown/censored cost semantics. [NIST project](https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai).

Caprio et al. (UAI 2026) study robust predictive uncertainty under prior and likelihood contamination. D6.17.2.3 complements the model side of D6.17/D6.18 with empirical execution evidence; it does not claim a new robust-Bayes method. [PMLR paper](https://proceedings.mlr.press/v337/caprio26a.html).

## Receipt model

A candidate-visible run receipt contains:
- exact repository, branch, commit and worktree digest;
- model provider/version/config digest, harness/toolset/permissions digests;
- runner class, hardware, image, region and billing currency;
- run ID, task family, paired seed, queue/start/end timestamps;
- ordered attempt events with option ID, attempt number, cold/warm/retry mode, status/result, start/end/elapsed time, CPU/GPU/network measures and price;
- billing record reference and digest where cost is positive;
- evidence digest for pass/fail observations;
- previous-event and event SHA-256 plus receipt SHA-256.

The evaluator supplies an approved context and approved option set separately. Candidate receipts must match this manifest. Event hashes link each event to the preceding event; the receipt hash covers the complete receipt. Hashes detect changes relative to the committed digest but **do not authenticate the producer**. A runner identity and attestation reference/digest are recorded, but v0.1 only checks the digest shape; it does not validate signatures, certificates, provider invoices or a trusted hardware attestation chain.

## Execution outcome semantics

| Attempt status | Execution evidence | Cost treatment | Retry semantics |
|---|---|---|---|
| completed | start/end and pass/fail result | actual price required or run cost marked incomplete | no implicit retry |
| failed | executed, ended, result unknown | actual price required or run cost incomplete | retry may follow a definitive failed event |
| timeout | started; may or may not have a terminal timestamp | known price counted; absent price makes cost incomplete | not retried automatically |
| cancelled | started; may or may not have a terminal timestamp | known price counted; absent price makes cost incomplete | not retried automatically |
| skipped | no execution; explicit zero cost | zero | not a pass |
| queued | not started; unknown cost | full-run cost censored/incomplete | not a pass |
| unknown | unresolved execution result | full-run cost censored/incomplete | no retry until reconciliation |

A failed event is a known execution failure, distinct from an unknown outcome. Retrying after timeout, cancellation or unknown execution state could duplicate a side effect; v0.1 therefore fails closed until the outcome is reconciled. This aligns with the wider EGDDM rule that an unknown execution outcome blocks a new mutation until reconciliation.

## Quantile and tail-cost protocol

The validator uses the nearest-rank quantile definition:

\[
Q_p = x_{(\lceil pn\rceil)}
\]

for sorted complete-run samples \(x_{(1)}\le \ldots \le x_{(n)}\). It reports sample count with p50, p95 and p99. A small sample's p95/p99 is a sample statistic, not a reliable population-tail estimate. The receipt validator never labels these as a rare-event reliability guarantee.

The report distinguishes:
1. full-run total price from all priced attempts on a complete run;
2. full-run wall time from run start to run finish;
3. queue delay from receipt queued time to run start;
4. per-option aggregate cost per complete run (including retries), then p95 across runs;
5. sum of per-option aggregate p95 values, as a descriptive-only comparator.

A run is excluded from the corresponding full-run quantile if its cost or wall time is incomplete. The report separately counts runs incomplete in either dimension. It never treats queued or unknown execution as zero cost. Skipped attempts are explicit and only have zero cost when the receipt records no execution and zero resource use.

## Synthetic anti-correlated example

Two complete runs have two gate costs in micro-USD:

| Run | Gate A | Gate B | Full-run total |
|---|---:|---:|---:|
| 1 | 90 | 10 | 100 |
| 2 | 10 | 90 | 100 |

With nearest-rank quantiles on these two samples:
- Gate A p95 = 90;
- Gate B p95 = 90;
- sum of marginal p95s = 180;
- complete-run total p95 = 100.

The sum overstates the sample full-run p95 by 80% in this deliberately anti-correlated fixture. This is not a production benchmark; it demonstrates why marginal quantiles cannot substitute for whole-trajectory samples. In another joint dependence pattern the sum may behave differently; it still is not a general identity or p95 guarantee.

## Implementation status

The Rust binary run-cost-receipt-validator.rs currently:
- validates an evaluator-approved repository/model/harness/tool/runner context and billing currency;
- validates run IDs, attempt sequence, retry parent, statuses, timestamps and elapsed duration;
- verifies event predecessor hashes, event SHA-256, and receipt SHA-256;
- requires billing records when positive price is reported and evidence digests for pass/fail outputs;
- tracks executed, queued, skipped, failed, timed-out, cancelled, unknown and retry counts separately;
- derives per-run cost and wall-time samples only when complete;
- reports full-run p50/p95/p99, queue-delay quantiles, per-option aggregate-per-run p95s and the sum comparator separately.

The current fixture is entirely synthetic and uses USD micro-units. Its expected report has two complete runs, four attempts, full-run total 100 micro-USD on both runs, per-option aggregate p95s of 90 and 90, and sum comparator 180. A retry negative/positive control confirms that failed-attempt and retry prices are included in both the full-run and per-option totals. It does not contain real runner/billing evidence.

## Benchmark protocol and targets

**Matched baselines**
- full revalidation;
- D6.15 minimum-sufficient set;
- nominal greedy EVSI;
- robust one-step EVSI;
- D6.17.2.1 adaptive policy;
- D6.17.2.2 joint-kernel adaptive policy when available;
- human/operator plan;
- always-act and always-abstain controls.

All methods must use the same model/version, task family, harness, tools, permissions, runner class, concurrency and paired seeds. Preserve all failed attempts and retries. Use exact commit/head identity and independent receipt replay.

**Metrics**
1. End-to-end full-run cost p50/p95/p99 in a named currency.
2. End-to-end wall-time p50/p95/p99.
3. Queue delay p50/p95/p99.
4. Cost per robustly justified action.
5. Cold/warm/retry cost, attempt count and retry rate.
6. Censoring/unknown-outcome rate and reconciliation latency.
7. Receipt completeness, digest-chain failures and context mismatches.
8. Cost-estimate coverage and sharpness by runner/tool/task family.
9. Robust-risk violations, action coverage and policy regret on the same runs.

**Proposed targets (not achieved)**
- 100% run/attempt receipt completeness for reported complete-run quantiles.
- Zero silent retries after unresolved/unknown execution.
- Zero receipt chain/context integrity violations accepted.
- Zero robust-risk or irreversible-gate violations on critical controls.
- At least 95% justified-action coverage on positive controls.
- On matched, independently replayed tasks: target at least 20% lower full-run median cost and 10% lower full-run p95 cost than full revalidation at the same robust-risk envelope.
- Start with a 144-decision validity pilot, then a 400-episode development suite; expand to thousands of independent or cluster-aware runs before rare-tail claims.

## Claim ceiling and next steps

D6.17.2.3-alpha validates receipt integrity, context parity, attempt semantics and statistics over supplied complete runs. It does not authenticate runner honesty, verify real billing providers, estimate population tails from small samples, validate a stochastic policy or establish production reliability.

Next:
1. Add cryptographic signature verification using a trusted runner/provider key registry.
2. Reconcile receipts against external billing and runner attestations.
3. Add censoring-aware estimation for incomplete/unknown trajectories without reclassifying them as zero.
4. Join run receipts to D6.17.2.1 policy nodes and D6.17.2.2 joint outcomes.
5. Add cluster-aware confidence intervals and paired-seed inference for full-run cost savings.

Rust compilation, tests, rustfmt, Clippy and CI remain unexecuted in this environment. Keep the alpha draft until exact-head toolchain validation, independent receipt replay and real runner evidence are available.
