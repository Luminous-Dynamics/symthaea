# QCOMP-STAT-001: paired inference primitives

Status: **implementation draft; no empirical result claimed**

This addition addresses the first two deliverables in issue [#8438](https://github.com/Luminous-Dynamics/symthaea/issues/8438). It adds dependency-free Rust helpers to the symthaea-quantum-comp crate.

## Implemented

### 1. Order-statistic interval for a population median

The median_order_statistic_interval function sorts finite observations and chooses the narrowest central order-statistic interval that reaches the requested confidence under the binomial coverage calculation. It returns the order statistic and achieved coverage with the bounds, so callers can audit the choice.

A crucial failure mode is handled explicitly: a small sample may not support a finite 95% distribution-free interval. Rather than report a nominal interval with insufficient coverage, the function returns None bounds and None achieved coverage. The documented coverage is exact under independent sampling from a continuous population; ties make this interval conservative.

This is an interval for the population median, not for the mean. It does not address dependence among multiple runs from the same graph.

### 2. Paired sign-flip randomization test for the mean effect

The paired_sign_flip_test function accepts finite paired differences and evaluates the absolute mean difference under sign flips. It enumerates all assignments up to a hard-bounded configurable size of 20 pairs; above that it performs deterministic Monte Carlo sign flips using the recorded seed and a plus-one p-value correction. Results retain the tested pair count, number of evaluated assignments, extreme count, method, seed/draw count, p-value, and approximate Monte Carlo standard error.

The test is exact only when the null makes paired signs exchangeable, such as a suitable paired randomized assignment or a symmetric null model. It is not the existing exact sign test: the sign test uses only wins/losses, whereas sign-flip inference uses effect magnitude. Neither test automatically corrects for multiple comparisons.

Values are rescaled before summation, reducing overflow risk for finite input values. Non-finite values, empty samples, invalid confidence, invalid exact limits and zero Monte Carlo draws fail explicitly.

## Required study practice

- Use independent analysis units. For MIS, a conservative default is one preregistered aggregate per graph instance, with independent graph instances as the paired units. Do not treat shots or repeated seeds on the same graph as independent graph samples.
- Preserve the paired deltas and their graph/cohort identities so inference can be replayed from raw records.
- Freeze direction, statistic, cohort, exclusions and analysis plan before confirmatory evaluation.
- Report the effect size and paired raw outcomes alongside p-values. A p-value alone is not a practical-value measurement.
- Add a documented multiplicity procedure for preregistered families of policy comparisons before using these helpers for a confirmatory leaderboard.
- Add cluster-aware bootstrap and Pareto-frontier stability as follow-up work; these are not implemented by this contribution.
- Keep synthetic, classical solver, ideal simulator, tensor-network simulator, circuit-export-only and physical-QPU evidence separate. These helpers do not attest evidence provenance or verify MIS feasibility.

## Tests authored

The module includes hand-computed fixtures for an exact three-pair sign-flip p-value and a 95% median interval with six observations; an insufficient-sample case; non-finite/invalid-input checks; large finite magnitudes; and deterministic replay for the Monte Carlo path.

**Validation boundary:** these Rust tests are authored but have not been compiled or executed in this authoring environment. The code has not been used to analyze a QOBLIB or physical-QPU cohort. No performance, quantum advantage, SOTA, or statistical calibration claim is made. CI is not treated as passed until a result is actually published.
