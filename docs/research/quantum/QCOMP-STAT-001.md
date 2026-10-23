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

### 4. Holm-Bonferroni family-wise error control

The new `holm_bonferroni_adjust` helper converts a preregistered family of valid p-values into family-wise-error-adjusted p-values, preserving the original input order. It validates that every p-value is finite and in [0, 1], handles ties deterministically by original index, and returns an empty vector for an empty family.

This is the step-down Holm procedure: sort p-values ascending, multiply rank (i) by the number of hypotheses remaining, cap at one, then take a cumulative maximum over the sorted sequence before mapping results back to original order. It controls family-wise error under arbitrary dependence when the individual p-values are valid.

The correction does **not** fix post-hoc selection of which comparisons to include, invalidate-to-valid transformations, violated sign-flip exchangeability, repeated-seed pseudoreplication, or dependence among graph families. Freeze the family of hypotheses before confirmatory analysis and report raw and adjusted p-values side by side. Adjusted p-values do not replace effect sizes, uncertainty intervals, or practical cost-to-target metrics.

### 3. Hierarchical graph-cluster bootstrap

The new `hierarchical_cluster_bootstrap_mean` function accepts paired differences grouped by independent graph instance. It samples graph clusters with replacement in the outer stage, then resamples each selected graph's repeated paired trials within that graph, preserving its repetition count. Each graph is weighted equally in the reported mean, so a graph with many repeated seeds does not become many independent graphs.

The API returns the number of clusters and observations, replicate count, seed, graph-weighted point estimate, and percentile interval. It rejects fewer than two clusters, empty clusters, non-finite differences, invalid confidence levels, and zero bootstrap replicates. A deterministic seed makes the resampling replayable.

**Inference boundary:** this is an approximate percentile bootstrap, not an exact finite-sample interval. It assumes graphs are independent sampling units and that repeated runs within each graph represent the intended within-graph resampling population. It does not correct for family-level dependence, dataset leakage, or multiple policy comparisons. Where many graphs come from a small number of graph-generating families, resampling may need to use the family as the independent outer cluster instead.

The bootstrap is intended for graph-level paired *effects* (for example, one delta per graph/seed, clustered by graph), not raw device shots treated as independent instances. Keep the estimand and aggregation rule preregistered.

## Required study practice

- Use independent analysis units. For MIS, a conservative default is one preregistered aggregate per graph instance, with independent graph instances as the paired units. Do not treat shots or repeated seeds on the same graph as independent graph samples.
- Preserve the paired deltas and their graph/cohort identities so inference can be replayed from raw records.
- Freeze direction, statistic, cohort, exclusions and analysis plan before confirmatory evaluation.
- Report the effect size and paired raw outcomes alongside p-values. A p-value alone is not a practical-value measurement.
- Holm-Bonferroni adjustment is implemented for preregistered families of valid p-values. The comparison family still needs to be frozen before confirmatory evaluation, and raw plus adjusted p-values should both be reported.
- Hierarchical graph-cluster bootstrap is implemented as an approximate percentile interval, and Holm-Bonferroni adjusts preregistered p-value families. Pareto-frontier stability remains follow-up work.
- Keep synthetic, classical solver, ideal simulator, tensor-network simulator, circuit-export-only and physical-QPU evidence separate. These helpers do not attest evidence provenance or verify MIS feasibility.

## Tests authored

The module includes hand-computed fixtures for an exact three-pair sign-flip p-value and a 95% median interval with six observations; a small-sample median-coverage sweep against integer-binomial reference values; non-finite/invalid-input checks; large finite magnitudes; deterministic Monte Carlo replay; and graph-cluster bootstrap fixtures for equal cluster weighting, deterministic replay, and malformed clusters; and hand-computed Holm-adjustment fixtures, including ties, preserved input order, monotonicity, and invalid p-values.

**Validation boundary:** these Rust tests are authored but have not been compiled or executed in this authoring environment. The code has not been used to analyze a QOBLIB or physical-QPU cohort. No performance, quantum advantage, SOTA, or statistical calibration claim is made. CI is not treated as passed until a result is actually published.
