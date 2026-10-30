# QARCH-016 — Cluster-aware inference for hierarchical benchmark suites

**Issue:** [#9125](https://github.com/Luminous-Dynamics/symthaea/issues/9125)  
**Implementation:** crates/domains/symthaea-quantum-comp/src/clustered_advantage.rs  
**Schema:** docs/research/quantum/schemas/QARCH-016-CLUSTERED-ADVANTAGE.schema.json  
**Counterexample:** docs/research/quantum/benchmarks/QARCH-016-PSEUDOREPLICATION-FIXTURE.json  
**Null stress:** docs/research/quantum/benchmarks/QARCH-016-CLUSTER-NULL-STRESS.json  
**Power schema:** docs/research/quantum/schemas/QARCH-016-CLUSTER-POWER.schema.json  
**Power sensitivity:** docs/research/quantum/benchmarks/QARCH-016-CLUSTER-POWER-SENSITIVITY.json

## Research question

QARCH-015 added exact paired sign tests and Holm correction, but it assumes the supplied rows are independent blocks. Quantum benchmark suites frequently expose many instances under a smaller number of workload families, problem classes, generators or shared source lineages. Unique instance IDs do not by themselves make those instances independent.

This is not a quantum-specific statistical issue. Clustered-data literature repeatedly documents type-I error inflation when individual observations are analysed without accounting for their cluster structure, particularly with few clusters and unequal cluster sizes. Small-cluster analyses also face a power/precision tradeoff, so correction is not free. See [the small-cluster analysis comparison](https://academic.oup.com/ije/article/47/1/321/4091562), [the type-I error review and reanalysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC5013635/) and the 2026 [CARE cluster-inference protocol](https://onlinelibrary.wiley.com/doi/10.1002/sim.70610). Quantum benchmarking reviews independently emphasise holistic, reproducible and complementary benchmarks rather than reliance on one metric ([Nature Reviews Physics](https://www.nature.com/articles/s42254-024-00796-z), [benchmark metrics compendium](https://arxiv.org/abs/2502.06717)).

## Hierarchy and estimand

Use the explicit hierarchy:

\[
\text{suite}\rightarrow\text{independent family/lineage}
\rightarrow\text{workload instance}\rightarrow\text{repeated run/shot/retry}
\]

Repeated measurements must first be reduced within an instance by a frozen rule. The current API accepts one already-reduced paired outcome per unique workload instance. It then reduces instances inside each independent cluster using the median paired effect.

For cluster \(j\), the two signed instance effects are:

\[
d^Q_{ij}=Q_{q,ij}-Q_{c,ij}
\]

\[
d^R_{ij}=\log(R_{c,ij})-\log(R_{q,ij})
\]

and the cluster effects are:

\[
\tilde d^Q_j=\operatorname{median}_i(d^Q_{ij}),\qquad
\tilde d^R_j=\operatorname{median}_i(d^R_{ij})
\]

One exact sign test is then performed across the independent clusters for quality non-inferiority, and another for practical resource superiority. The joint claim p-value is the maximum of the two, as in QARCH-015. Holm correction across the preregistered confirmatory claim family remains mandatory.

**The estimand in this implementation is equal-cluster weighting.** A family with 100 instances contributes one cluster effect, just like a family with 5 instances. This prevents large generated families from dominating the inference merely because they contain more rows. It is not automatically the correct deployment-population estimand. If the target population should weight families by prevalence, those weights must be frozen before observing outcomes and a separately justified weighted/cluster-aware method must be implemented; do not silently substitute instance counts for population weights.

## Fail-closed validation

The API checks:
- non-empty and unique instance IDs and workload fingerprints;
- one cluster fingerprint per cluster ID, and no fingerprint reused across cluster IDs;
- one comparison-contract digest across all paired instances;
- positive finite end-to-end resource values and finite quality differences;
- a preregistered minimum number of independent clusters;
- valid margins and tie tolerances.

It reports cluster-size minimum/maximum, per-cluster effects, the equal-cluster median, exact family-level tests, the joint p-value and leave-one-cluster-out ranges. Leave-one-out diagnostics are sensitivity measures, not independent confirmatory tests. A result whose practical margin disappears after removing one family deserves a much weaker interpretation.

The method does not verify that clusters are truly independent. That classification must be evidenced from workload provenance, generator lineage and benchmark design. Cluster IDs are not enough if supposedly different generators share a common ancestor or the same source data.

## Concrete pseudo-replication counterexample

The frozen fixture has 10 independent families and 10 instances per family. Six families have quantum resource 8 versus classical 10; four have quantum resource 11 versus classical 10. Every instance inside one family shares that family's direction. Quality is held non-inferior.

The naive analysis sees 60 favourable instance rows out of 100 and obtains p=0.02844, below 0.05. The cluster analysis sees six favourable families out of ten and obtains:

\[
P\{X\ge6\},\quad X\sim\operatorname{Binomial}(10,0.5)
=386/1024\approx0.37695
\]

The cluster-level analysis does not support the claim. This is a constructed null stress witness, not measured evidence about any hardware or public benchmark suite.

## Reproducible ICC stress simulation

The stress harness is included in the Rust module. It compares an instance-level sign test that incorrectly assumes independent rows with the implemented family-median sign test.

For each independent family, a latent favourable-outcome probability is drawn as \(0.5\pm0.5\sqrt{\rho}\), with each sign selected equally. Conditional instance signs are Bernoulli draws from that probability. This yields marginal win probability 0.5 and within-family Bernoulli ICC \(\rho\). Independent continuous magnitudes are drawn immediately after each instance sign. The cluster effect is the median of these signed magnitudes; for even cluster sizes, the median is the arithmetic mean of the two central signed values. The frozen result fixture uses this exact reduction and random-draw order.

The frozen experiment uses 10 families, 10 instances per family, 10,000 null replicates for each ICC in \(\{0,0.05,0.15,0.30,0.60,1\}\), one-sided alpha=0.05, Mulberry32 seed 0x00516016, and Wilson 95% Monte Carlo intervals.

| ICC | Naive instance-level type-I rate | Cluster-median type-I rate |
|---:|---:|---:|
| 0.00 | 4.73% | 1.19% |
| 0.05 | 8.49% | 1.20% |
| 0.15 | 13.73% | 1.10% |
| 0.30 | 19.13% | 1.03% |
| 0.60 | 25.42% | 1.05% |
| 1.00 | 37.14% | 1.03% |

In this declared model, the naive row-level test becomes substantially anti-conservative as ICC rises. The cluster-level procedure remains below 5%. With only 10 clusters, the exact one-sided sign test is discrete: it rejects at 0.05 only with at least 9 wins (attainable null rate 11/1024, about 1.07%). Its conservative rate is accompanied by reduced power. Do not interpret this one experiment as a universal guarantee for arbitrary dependence structures.

A second exact enumeration varies the number of independent families from 5 to 30 while retaining 10 perfectly dependent instances per family. Under that perfect-dependence null, the naive test's exact false-positive rate is 18.75–37.70% across the displayed designs, whereas the cluster-level sign test stays at or below 4.94%. The machine-readable result file preserves the thresholds and exact enumeration counts.

## Exact power planning: replace sample-size folklore

The Rust API now exposes an exact cluster sign-test power planner. Its inputs are the probabilities that an *independent cluster's median effect* clears the predeclared quality margin and practical resource margin. These are family-level win probabilities, not instance-level win rates.

For one component with \(n\) independent clusters and a family-level win probability \(p\), the exact power is:

\[
\operatorname{Power}(n,p)=P\{\operatorname{Binomial}(n,p)\ge k_\alpha(n)\}
\]

where \(k_\alpha(n)\) is the smallest number of cluster wins whose one-sided null sign-test p-value is no greater than the per-claim alpha.

The joint quality-and-resource claim requires both tests to reject. Without assuming that quality and resource outcomes are independent, the planner uses the valid lower bound:

\[
P(\text{both reject})\ge
\max(0,\operatorname{Power}_Q+\operatorname{Power}_R-1)
\]

The frozen sensitivity design uses alpha=0.05 per claim and a desired joint-power lower bound of 80%. It assumes equal family-level win probabilities for the two components:

| Probability each independent family clears each margin | Independent families required |
|---:|---:|
| 55% | 866 |
| 60% | 213 |
| 65% | 93 |
| 70% | 53 |
| 75% | 33 |

This is why a blanket recommendation of 30–50 independent families is misleading. Under this exact sign-test design, 30 families deliver only about 51% component power when each family clears the margin with probability 65%; even 50 families deliver about 62%. At 75%, 33 families suffice for the specified 80% lower bound on joint power.

These probabilities describe the *true proportion of independent families that exceed the declared practical/quality margins*. They must be estimated from a pilot or justified by prior evidence, not chosen to make the sample size look affordable. Sensitivity tables for fixed counts 5, 10, 20, 30, 50 and 100 are preserved in the machine-readable fixture.

The power planner scans the exact discrete rejection boundary and returns the first feasible cluster count. Exact sign-test power can be non-monotone across adjacent sample sizes because critical win thresholds jump discretely. For multiple confirmatory claims, the alpha supplied to the planner must reflect the preregistered family correction; a worst-case conservative design can use family_alpha / family_size before applying Holm.

The result is an analytic design calculation under independent clusters and the sign-test model, not a guarantee for a particular benchmark suite. It does not account for uncertainty in the pilot estimate of family win probability, selection bias in family construction, or invalid cluster independence. Those require additional design and sensitivity work.

## Industry benchmark implications

- **QOBLIB:** distinguish model-independent problem classes from instances within one class; do not infer broad performance from many near-duplicate instances of one generator.
- **JUNIQ and application suites:** preserve the declared workload and quality contracts, and map each test into the same cluster hierarchy before combining evidence.
- **IonQ application and QUOPS:** retain separate application-utility and system-capacity layers; neither removes the need for independent workload-family inference.
- **Photonic component references:** keep conditional component metrics separate from the workload/application clusters and never treat repeated photon detections as independent application tasks.

The benchmark objective remains complete resource per verified acceptable answer, with strongest applicable classical baselines and matched tuning budgets. QARCH-016 protects the statistical interpretation; it does not itself improve a QPU, reduce optical loss or establish an application crossover.

## What remains open

The first implementation is deliberately narrow. Next required work is:
1. vary family count, unequal family sizes and target family weights;
2. simulate heterogeneous and sign-reversing family effects, missing/timeout outcomes and dependence across nominally different generator lineages;
3. perform power/sensitivity studies for 5, 10, 20 and 30 independent families;
4. compare family-median sign tests with design-based randomization inference when assignment is available, and small-sample corrected cluster methods only where their assumptions are justified;
5. calibrate type-I error using at least 10,000 replicates per preregistered design cell, reporting Monte Carlo uncertainty;
6. freeze target-population weights and a cluster provenance taxonomy before any confirmatory hardware campaign.

The current ICC grid addresses one balanced, exchangeable null model. It does not fulfil the entire broader study. Compilation, Rust tests and CI must still pass on the exact PR head before this is treated as validated implementation evidence.
