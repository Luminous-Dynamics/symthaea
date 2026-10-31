# NAS anytime confidence receipts v0.1

Tracking: [NAS-ANYTIME-001 / #9443](https://github.com/Luminous-Dynamics/symthaea/issues/9443)

## 1. Why this exists

The randomized-QMC hypervolume estimator reports replicate dispersion and a normal-approximation interval. That remains a useful numerical diagnostic. It is not automatically a time-uniform guarantee after repeatedly looking at estimates while adaptively choosing more evaluations, pruning candidates, or spending hardware resources. Likewise, a nominal 95% interval at each look does not imply 95% coverage for the whole adaptive search.

This increment adds a conservative reference method: a bounded-mean Hoeffding interval with explicit alpha spending over looks and Bonferroni allocation across a fixed family of decision streams. A threshold that is not separated by the interval produces INCONCLUSIVE, never implicit approval.

## 2. Statistical contract

For one stream, let replicate-level observations be independent and lie in a known interval [a,b]. At look t, after n replicate observations, define:

- alpha_stream = alpha_family / family_size;
- alpha_t = alpha_stream / (t (t + 1));
- radius_t = (b-a) sqrt( log(2 / alpha_t) / (2 n) ).

The interval is [mean - radius_t, mean + radius_t], clipped to [a,b]. Since sum over t>=1 of 1/(t(t+1)) is 1, the look-specific error allocations sum to no more than alpha_stream. Bonferroni allocation across the preregistered family of streams bounds the family-wise error by alpha_family. The union bound across looks does not require looks to be independent; the observations within each stream must satisfy the stated independent bounded-replicate assumption.

The result has three outcomes:

- AT_OR_ABOVE_THRESHOLD: lower bound is at least the threshold;
- BELOW_THRESHOLD: upper bound is strictly below the threshold;
- INCONCLUSIVE: the interval overlaps the threshold.

INCONCLUSIVE means gather more independently randomized replicates or abstain. It must not authorize promotion, pruning, or a resource reservation.

## 3. QMC interpretation and limitations

For randomized QMC, the independent units are complete independent randomizations/replicates. Integration points within a replicate are dependent by design. Applying the bound to point-level hits as though they were independent is invalid.

The present QMC implementation derives Cranley-Patterson shifts deterministically from a seed using SplitMix64. This gives reproducibility, not physical randomness. The anytime guarantee is therefore conditional on the random-shift generator behaving as independent uniform randomization; the implementation does not prove PRNG quality or independence. A future production-grade campaign should accept and record separately generated replicate seeds or a reviewed randomization stream, and bind the complete seed manifest to the run receipt.

For paired action comparisons using common random numbers, calculate one difference per independently randomized replicate. If each action's replicate value is in [0, V], the difference is in [-V, V]. The paired observations preserve common-random-number covariance while allowing the same bounded-mean contract. The action identity, posterior, QMC design, and portfolio digests must be bound by the caller's surrounding evidence receipt.

The family size and family/stream identifiers must be fixed before outcomes are inspected. Callers must maintain monotonically increasing look indices, use one stable stream ID per hypothesis, and predeclare all streams in the family. The module recomputes an individual receipt; it does not maintain a distributed append-only ledger or prove that an external caller followed the sequence protocol.

The Hoeffding bound is intentionally conservative and not variance-adaptive. It is a transparent correctness baseline, not a claim to outperform empirical-Bernstein or betting confidence sequences. It also does not establish model calibration, posterior correctness, physical measurement integrity, dataset representativeness, or a SOTA gain.

## 4. Receipt contract and implementation

Source: `crates/domains/symthaea-architecture-engineering/src/anytime_qmc_confidence.rs`.

The receipt binds:

- schema, version, and method identifier;
- preregistered family and stream IDs;
- family alpha, fixed family size, stream alpha, look index, per-look alpha, and cumulative alpha spent;
- sample count, declared support, and decision threshold;
- replicate mean, half-width, lower and upper bounds, and tri-state decision;
- ordered replicate-value SHA-256 and canonical receipt SHA-256.

The verifier recomputes the estimate and all receipt fields from the exact replicate values and configuration. A hash is an integrity binding, not a truth or independence proof.

Specializations are provided for QMC hypervolume replicates in [0, box volume] and paired action differences in [-box volume, +box volume].

## 5. Benchmark programme

Use a fully frozen synthetic campaign first, then the physical NATS-Bench-MCU v2 tabular campaign. Do not use the final evaluation data to tune a scheduler or stopping rule.

### Statistical qualification

1. Exact 2-D hypervolume union fixtures and finite-posterior portfolio-regret/VoC fixtures.
2. Null-boundary simulations at multiple true means near the decision threshold; vary family size, number of looks, replicate count, dimension, candidate count, and portfolio-family size.
3. Measure family-wise false promotion, false pruning, simultaneous coverage across all streams and looks, inconclusive rate, and mean replicates to decision.
4. Compare fixed normal 1.96, Student-t across independent RQMC replicates, this Hoeffding alpha-spending baseline, and an empirical-Bernstein/betting confidence-sequence baseline.
5. Include adversarial negative controls: duplicated shifts disguised as independent replicates, reused stream IDs, look-index rollback, changed support, altered sample order/digest, non-finite observations, and underdeclared family size. A future ledger verifier must reject these rather than treating the per-receipt hash as sufficient.
6. Run at least 10,000 complete simulated decision families per preregistered configuration. Report binomial uncertainty around observed family-wise error; do not interpret a noisy estimate as proof of the nominal bound.

### NAS quality and efficiency

Evaluate every method under equal, explicit vector budgets: accelerator seconds, CPU seconds, target-device occupancy, instrument seconds, energy, monetary cost, storage, and operator minutes. Report each dimension separately and publish the full non-dominated frontier.

Primary outcomes:

- held-out feasible-set hypervolume and portfolio-class hypervolume regret;
- feasible candidate recall and false-feasible rate;
- top-K feasible architecture hit rate;
- Pareto-set precision/recall and rank correlation, with rank metrics clearly separated from decision quality;
- measurements, wall time, device energy, and human/operator time consumed to reach a fixed held-out regret target;
- calibration and coverage of predictive distributions, not only point rank correlation;
- rate of correct abstention when evidence is insufficient.

Baselines should include random search, regularized evolution / NSGA-II, multi-fidelity random and successive-halving methods, predictor/zero-cost NAS, and BoTorch qEHVI/HVKG or an equivalent well-maintained multi-objective Bayesian optimization baseline. For embedded hardware, compare against ConNAS and the hardware-in-the-loop deployment workflow represented by EdgeVolution. The search spaces and tasks are not interchangeable; reproduce each baseline's prescribed protocol rather than comparing headline metrics across different tasks.

### Proposed go/no-go targets (aspirational, not achieved)

- Family-wise false promotion/pruning at or below 5% in synthetic campaigns with nominal 5% family alpha; empirical coverage must be accompanied by uncertainty intervals.
- At least 95% simultaneous coverage for preregistered synthetic test families; investigate every coverage shortfall.
- At least 20% fewer independent hardware measurement replicates than the conservative Hoeffding baseline at the same false-decision rate, after introducing a variance-adaptive method.
- At least 10% lower all-in vector-resource cost to a fixed held-out Pareto-regret target than the strongest matched-budget baseline, without a false-feasible-rate or calibration regression.
- No deployment recommendation where device/firmware/protocol identity is missing, stale, or incompatible.

These are research targets. No real dataset run, Rust test execution, CI pass, qualification, or SOTA improvement is claimed by the authored module.

## 6. Industry context

- Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*, Annals of Statistics (2021): https://doi.org/10.1214/20-AOS1991
- Waudby-Smith and Ramdas, *Estimating means of bounded random variables by betting*, JRSS B (2024 issue; published online 2023): https://doi.org/10.1093/jrsssb/qkad009
- L'Ecuyer et al., *Confidence intervals for randomized quasi-Monte Carlo estimators*, Winter Simulation Conference (2023): https://doi.org/10.1109/WSC60868.2023.10408613
- EdgeVolution, *Communications Engineering* (23 June 2026): https://doi.org/10.1038/s44172-026-00708-2
- ConNAS, ICLR 2026: https://openreview.net/forum?id=7c9f851081499ed5c60c51e312d3235f
- NATS-Bench-MCU v2, Zenodo record 21644470 (28 July 2026): https://doi.org/10.5281/zenodo.21644470

## 7. Next integration gates

1. Replace the resource scheduler's fixed 1.96 decision path only after it consumes and verifies an anytime receipt bound to the same objective action, posterior, QMC design, and resource registry.
2. Add an append-only run-wide stream ledger that rejects duplicate stream IDs, look rollback, undeclared streams, and family-size changes.
3. Add a PRNG/replicate-seed manifest and independent randomization audit.
4. Add empirical-Bernstein or betting confidence sequences, preserving this Hoeffding implementation as a simple reference verifier.
5. Connect the gate to #9388 qualification capabilities and #9329/#9104 run-wide risk budgets. Passing a confidence threshold alone must never authorize operational pruning.
