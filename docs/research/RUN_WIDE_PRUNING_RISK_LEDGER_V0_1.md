# Run-Wide Pruning Risk Ledger v0.1

**Status:** Rust source and authored tests added on `research/run-wide-pruning-risk-ledger-v0.1`; compilation, execution and statistical qualification remain pending.

**Tracking:** [#9287](https://github.com/Luminous-Dynamics/symthaea/issues/9287).

**Implementation:** `crates/domains/symthaea-architecture-engineering/src/run_wide_pruning_risk.rs`.

## 1. Purpose

The existing [selection-conditioned audit](https://github.com/Luminous-Dynamics/symthaea/pull/9104) evaluates whether a uniformly sampled subset of discarded candidates would beat the promoted frontier at the same next fidelity. It produces an exact finite-population hypergeometric upper bound and an aggregate disposition.

That guarantee is local to one pruning event. A search policy can make multiple adaptive pruning decisions. Reporting a 95% bound at every rung without controlling the total confidence spent across the run does not establish a 95% run-wide statement.

This increment composes the existing audit across a fixed maximum number of events.

## 2. Confidence budget is not search-quality risk

The configuration declares two independent quantities:

- `delta_run`: the maximum probability budget that one or more event-level audit bounds fail to cover their finite-population false-prune rates, under the audit's assumptions.
- `rho_event`: the largest acceptable upper bound on the fraction of the discarded population that would beat the promoted frontier at that event's audit fidelity.

Neither quantity is the probability that the global optimum or final top-K is retained. Top-K survival remains an empirical outcome unless separately proved.

## 3. Predictable alpha spending

For a predeclared maximum of (T) audit events, event (t) receives:

[
delta_t = rac{delta_{run}}{T}
]

and the finite-population audit is configured at confidence level (1-delta_t).

For any realized number of events (m le T):

[
sum_{t=1}^{m}delta_t = rac{m}{T}delta_{run} le delta_{run}
]

A union bound therefore composes conditional event-level coverage without requiring independent audit outcomes. The schedule is fixed before outcomes are observed and does not depend on previous audit results.

A summable infinite-horizon schedule could be added later if event count must be unbounded. The bounded uniform schedule is preferred for v0.1 because its policy and review contract are simpler.

## 4. Event contract

The ledger accepts:
- unique event and run IDs;
- the promoted cohort's complete target-fidelity scores;
- a private `PruningAuditSample` prepared by the existing sampling module;
- exactly one target-fidelity score per committed sentinel;
- a named fidelity ID.

The ledger itself calls `audit_pruning`, so a caller cannot substitute an arbitrary aggregate report. It verifies the committed sample, recomputes the exact hypergeometric upper bound, and consumes the raw sentinel outcomes. The public event receipt contains aggregate statistics only.

Each receipt binds:
- run and event identity and sequence index;
- named and numeric fidelity;
- canonical promoted-frontier digest;
- discarded-population and sentinel-sample digests;
- seed/sample identity binding digest;
- observed and upper-bound false-prune counts/rates;
- per-event confidence spend, confidence level and cumulative spend;
- pre-registered risk threshold and disposition;
- previous receipt digest and current receipt digest.

The seed binding is a provenance digest. It is not a cryptographic random-number generator, proof of entropy, proof of pre-outcome commitment, or process isolation.

## 5. Fail-closed behavior

Pruning becomes permanently disabled for the run after:
- an invalid audit sample or mismatched fidelity;
- duplicate or malformed event identity;
- exhausted event budget;
- insufficient sentinel sample size;
- an upper confidence bound above the pre-registered false-prune threshold.

The failing audit is recorded when it produces a valid aggregate result. Malformed evidence fails closed without producing a success receipt. Later pruning events are rejected. The controller must fall back to retaining candidates, reactivating dormant candidates, or a separately justified non-pruning policy.

The ledger is an evidence primitive, not a complete execution sandbox. The evaluator must keep raw sentinel identities/scores away from the search policy using an independent process boundary in production.

## 6. Integration

The new module builds on:
- [#9104](https://github.com/Luminous-Dynamics/symthaea/pull/9104): exact finite-population sentinel sampling and budgeted audit allocation;
- [#9172](https://github.com/Luminous-Dynamics/symthaea/pull/9172): metered proxy acquisition and multi-fidelity target-query accounting;
- [#9267](https://github.com/Luminous-Dynamics/symthaea/issues/9267): adaptive value-of-computation scheduling;
- [#9287](https://github.com/Luminous-Dynamics/symthaea/issues/9287): run-wide risk-budget research specification.

The scheduler must treat model-based expected value-of-computation as an acquisition heuristic, and the hypergeometric audit receipt as a distinct statistical certificate. Expected regret reduction is not itself certified by this ledger.

## 7. Authored Rust tests

Ten tests are authored for:
- predictable uniform alpha spending and max-event enforcement;
- a valid census audit and receipt-chain verification;
- cumulative confidence spend;
- observed-risk fail-closed behavior;
- insufficient-sample fail-closed behavior;
- invalid/mismatched-fidelity evidence;
- duplicate event IDs;
- receipt tamper detection;
- order-invariant promoted-frontier digests;
- invalid configuration rejection.

These are test declarations, not passing tests. Compilation and execution are still required.

## 8. Qualification plan

Before making a statistical or performance claim:

1. Compile, format and run the exact branch head.
2. Independently reproduce event-level hypergeometric upper bounds.
3. Simulate repeated adaptive pruning episodes under the declared random-sampling assumptions and compare run-wide failure frequency to `delta_run`.
4. Add negative controls for seed reuse, post-outcome sample selection, biased sentinel selection, changed population after commitment, outcome leakage, receipt edits and interrupted/resumed ledger state.
5. Integrate all sentinel and failed-run costs into the same resource ledger as promoted candidates.
6. Compare raw Successive Halving, calibrated promotion, static guard slots, fixed-schedule audited SH, and adaptive VoC with and without the run-wide ledger at equal total measured cost.
7. Report final-confirmed P@1/P@10/P@100, normalized regret, CVaR-10%, false-pruned fractions, audit-bound coverage, disabled-pruning frequency, anytime quality/cost and worst-task regressions.

## 9. Claim ceiling

This source implements a bounded confidence-spending ledger that recomputes and chains existing finite-population audit results. It does not demonstrate compile correctness, run-wide statistical coverage, improved NAS discovery, best-arm identification, top-K survival, or SOTA performance. Those claims require independent execution and benchmark evidence.

## References

- [Multi-Fidelity Best-Arm Identification, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html).
- [Optimal Multi-Fidelity Best-Arm Identification, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html).
- [HiP-BO, online 16 October 2026](https://www.sciencedirect.com/science/article/pii/S092523122602802X).
- [NAS-Bench-x11 and the Power of Learning Curves, NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/be3159ad04564bfb90db9e32851ebf9c-Abstract.html).
- [NASLib NAS-Bench-201 full learning-curve interface](https://automl.github.io/NASLib/search_spaces/nasbench_201/).
