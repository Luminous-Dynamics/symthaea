# Monte Carlo Calibration and Power of E-Process Betting Policies V1

Status: deterministic Rust Monte Carlo benchmark added to the task-risk research branch. It does not change Symthaea's live control loop. Exact-head qualification is required before calling the Rust output qualified.

## Research question

For a bounded binary loss L_t in {0, 1}, the risk-null is:

    E[L_t | F_(t-1)] <= epsilon

When a nonnegative bet lambda_t is chosen using only observations available before time t and lambda_t * epsilon < 1, the factor

    1 + lambda_t * (L_t - epsilon)

is nonnegative and has conditional expectation at most one under that null. The product is an e-process. Ville's inequality controls the probability that it ever crosses 1/delta; K parallel monitor instances use threshold K/delta with the overall delta allocated by the union bound.

This benchmark compares four policies, all evaluated on the same deterministic Bernoulli event streams:

- **Fixed bet:** fraction 0.50 of 1/epsilon.
- **Three-bet mixture:** fractions [0.25, 0.50, 0.75] of 1/epsilon.
- **Wide seven-bet mixture:** fractions [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95] of 1/epsilon.
- **Predictable adaptive bet:** a plug-in empirical-mean/variance betting rate computed from *past losses only*, with a 20-observation minimum history and lambda_t * epsilon capped at 0.50. The current loss is added to the history only after its factor is evaluated.

The mixtures average nonnegative e-processes and remain e-processes. The adaptive method keeps validity under the same conditional-risk null because each bet is predictable and nonnegative with its bound strictly below 1/epsilon. The adaptive method is a compact plug-in baseline inspired by the general predictable-betting approach; it is not a reproduction of the UAI 2025 paper's GRAPA or PMEB implementations.

## Benchmark design

- Risk bound epsilon = 0.05.
- Overall delta = 0.05 across five parallel monitors. Per-method alarm threshold = K/delta = 100 (log threshold log(100)).
- 300 binary observations per monitor and 1,000 Monte Carlo cohorts per risk level.
- Six true Bernoulli risk levels: 0.01, 0.05, 0.075, 0.10, 0.20 and 0.40.
- Four methods see exactly the same seeded trace for each risk/cohort/monitor tuple.
- Output: 30,000 raw stream records (one per risk/cohort/monitor) containing the full 300-bit event trace, event count and first event, each method's first alarm and final log evidence; 24 summary rows (six risks × four methods).
- Summary reports family-wise alarm rate over the five parallel monitors per cohort, 95% Wilson interval, per-monitor alarm rate, detection delay among alarmed streams, average observed event count and configured bet specification.

Raw traces are retained so the qualification script independently replays all four factor recurrences and first-crossing indices. The Monte Carlo's family-wise statistic asks: for each cohort, did any of its five monitor processes alarm within 300 observations? The 1,000 cohorts estimate this probability; they do not imply that a deployed process can reset its error budget for free.

## Preliminary recurrence-replica results

A deterministic JavaScript replica of the same SplitMix64 seeds and factor recurrences produced these illustrative family-wise alarm rates over 1,000 cohorts per risk level:

| True Bernoulli risk | Fixed bet | Three-bet mix | Seven-bet mix | Predictable adaptive |
|---:|---:|---:|---:|---:|
| 0.010 | 0.2% | 0.2% | 0.2% | 0.0% |
| 0.050 (null boundary) | 2.5% | 3.2% | 3.1% | 0.2% |
| 0.075 | 7.2% | 10.7% | 44.0% | 24.3% |
| 0.100 | 17.9% | 43.6% | 98.9% | 96.8% |
| 0.200 | 97.2% | 100.0% | 100.0% | 100.0% |
| 0.400 | 100.0% | 100.0% | 100.0% | 100.0% |

These are exploratory recurrence-replica measurements, **not qualified Rust benchmark results**. They motivate the controlled test: the wide mixture appears strongest near a 7.5% risk rate in this fixture; the adaptive method beats the three-bet mixture there, while both catch most 10% risk cohorts. At the exact 5% risk boundary, observed false alarms are below 5% for all four methods in this seed panel. Empirical frequencies neither prove nor replace the e-process guarantee.

## Acceptance gates

- Two executions are byte-identical.
- Exact 30,000 raw records and 24 summary rows; no duplicate risk/cohort/monitor tuples.
- Each 300-bit trace matches its event count and first-event index.
- Independent replay matches all four log-evidence trajectories and first-alarm indices.
- Summary false-alarm counts, Wilson intervals, per-monitor rates, mean detection step, average event counts and exact method settings match recomputation from raw traces.
- At risk 0.01, all four family-wise alarm rates remain below 2%; at risk 0.05, their 95% Wilson upper bounds remain below 8% in this 1,000-cohort panel.
- At risk 0.075, wide-mixture family-wise detection exceeds the three-bet mixture by at least 15 percentage points, and adaptive betting exceeds the three-bet mixture by at least 10 points.
- At risk 0.10, wide-mixture detection exceeds the three-bet mix by at least 20 points, and adaptive betting exceeds it by at least 40 points.
- At risk 0.20 each method detects in at least 95% of cohorts; at risk 0.40, at least 99%.

These numbers are regression gates for this deterministic fixture, not universal claims about power. The conditional-risk null is an assumption about the stream's conditional mean; this simulation cannot establish it for a physical system. The declared delta applies to each 300-observation cohort across the five parallel monitors and within each compared method. Combining alerts from multiple method families with OR would require an additional multiplicity allocation.

## Paradigm shift

Instead of tuning one detector against an aggregate score, predeclare policies with different adaptation and bet-spectrum assumptions, then compare their **risk-validity/power frontier**. This reveals which strategy catches risk close to the tolerated bound without choosing the best policy on the test data. The method that is best at one shift size need not be fastest or best at another. A proper report therefore includes family-wise false-alarm probability, Wilson intervals, detection delay and runtime—not only final detection percentages.

## Industry comparison and next steps

The relevant sequential-risk reference is Timans et al., *On Continuous Monitoring of Risk Violations under Unknown Shift* (UAI 2025), which studies testing-by-betting and adaptive risk monitors: [paper](https://proceedings.mlr.press/v286/timans25a.html), [reference code](https://github.com/alextimans/risk-monitor). This benchmark implements simple predeclared fixed/mixture policies and a compact predictable plug-in baseline; it does not reproduce all of their methods or experiments.

For the runtime-control layer, keep the control benchmark comparison distinct from this Bernoulli power test:
- [SODA-MPC, L4DC 2025](https://proceedings.mlr.press/v283/contreras25a.html): a conformalized OOD monitor connected to a reachability fallback.
- [Robust OOD MPC via Conformalized System Level Synthesis, L4DC 2026](https://proceedings.mlr.press/v331/srinivasan26a.html): weighted conformal uncertainty and optimized reachable sets.

A defensible claim against those methods requires a pinned physical/high-fidelity controller benchmark; disjoint model-development, betting/calibration and final disturbance families; matched false-alarm budgets; and results for task reward, constraint violation frequency/severity, false-safe exposure before fallback, no-certified-action rate, detection delay, runtime and uncertainty. Nothing in this synthetic Bernoulli benchmark establishes state-of-the-art performance or physical safety.
