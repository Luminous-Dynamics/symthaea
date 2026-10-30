# QARCH-016 — Cluster-aware inference for hierarchical benchmark suites

**Issue:** [#9125](https://github.com/Luminous-Dynamics/symthaea/issues/9125)  
**Implementation:** crates/domains/symthaea-quantum-comp/src/clustered_advantage.rs  
**Result schema:** docs/research/quantum/schemas/QARCH-016-CLUSTERED-ADVANTAGE.schema.json  
**Null simulation:** docs/research/quantum/benchmarks/QARCH-016-CLUSTER-NULL-SIMULATION.json  
**Adversarial fixture:** docs/research/quantum/benchmarks/QARCH-016-PSEUDOREPLICATION-FIXTURE.json

## Paradigm shift: benchmark families, not benchmark rows

A large benchmark table can create the illusion of a large sample. Workload instances often share generators, source data, templates, graph families, parameterization or solver lineage. Multiple instances from one such family are not automatically independent evidence about a broader workload population.

QARCH-016 adds a conservative two-stage estimator:
1. Reduce repeated measurements within each workload instance using a preregistered rule. The input contains one paired instance-level observation only.
2. Reduce instance-level paired effects to one median effect per independent family, then perform exact sign tests across families.

Every family contributes one vote regardless of whether it contains one instance or a thousand. The estimand is the median paired effect across equally weighted independent families. This is not an instance-weighted estimate; it answers a different question.

## Paired effects and margins

For instance \(j\) in family \(k\):

\[
d^Q_{kj}=Q_{q,kj}-Q_{c,kj}
\]

\[
d^R_{kj}=\log(R_{c,kj})-\log(R_{q,kj})
\]

The family summaries are:

\[
D^Q_k=\operatorname{median}_j(d^Q_{kj}),\qquad
D^R_k=\operatorname{median}_j(d^R_{kj})
\]

Quality non-inferiority tests whether the family median exceeds \(-\delta_Q\). Resource superiority tests whether the family median exceeds \(-\log(1-\delta_R)\), where \(\delta_R\in[0,1)\) is the preregistered fractional resource reduction.

Threshold ties remain in the denominator as non-wins. The exact one-sided sign test is applied to the \(K\) independent family summaries, not to all \(N\) workload instances. The joint quality-and-resource p-value is the maximum of the two component p-values. The full confirmatory family of joint claims still requires Holm correction using QARCH-015.

## Fail-closed data contract

The implementation rejects:
- fewer than the preregistered minimum independent families;
- duplicate block IDs or duplicate instance fingerprints;
- empty family/contract identifiers or mismatched comparison-contract digests;
- non-finite quality values or non-positive/non-finite resource values;
- invalid quality margins, tolerances or fractional resource reductions.

family_id is not just a label: it must represent a plausible independent sampling cluster, including common benchmark generator or source lineage. If the family definition is chosen after observing outcomes, the inferential claim is compromised.

The report includes per-family effects and leave-one-family-out ranges. A narrow aggregate result that flips after removing one family is visibly fragile.

## Frozen pseudo-replication negative control

The analytic fixture contains 87 distinct workload instances in eight families. One family contributes 80 resource wins; the other seven families each contribute one resource loss.

If all 87 rows are incorrectly treated as independent, the exact one-sided sign-test p-value is approximately \(4.13\times10^{-17}\). Under the preregistered family-balanced estimand, there is only one winning family out of eight, with p-value \(255/256\approx0.9961\). The family-balanced median resource ratio is 0.8, meaning quantum uses 25% more resource at the family median.

This is a deliberate negative control. It demonstrates how a very large instance-level table can tell the opposite story from the independent-family evidence. It is not a hardware result.

## Deterministic null simulation

The frozen simulation runs 10,000 null replicates in each of 12 cells:
- balanced clusters: eight families of three instances;
- highly imbalanced clusters: one family of 21 instances plus seven singletons;
- within-family intraclass correlation (ICC): 0, 0.1, 0.25, 0.5, 0.75 and 1.

For each family, draw a latent fair sign \(Z\). Each instance sign copies \(Z\) with probability \(\sqrt{\mathrm{ICC}}\); otherwise it draws an independent fair sign. Because \(E[X\mid Z]=\sqrt{\mathrm{ICC}}Z\), the pairwise within-family correlation is the target ICC. Families are independent. Both procedures test the one-sided null at \(\alpha=0.05\). The naive procedure treats every instance as IID; the cluster procedure takes one family-majority sign per family, with exact within-family ties treated as non-wins.

In the imbalanced design, naive instance-level false-positive rate rose from 4.2% at ICC=0 to 50.1% at ICC=0.75. The family-level exact sign test remained between 3.11% and 3.87% across those six ICC cells. In the balanced design, the naive rate rose from 3.13% at ICC=0 to 13.92% at ICC=1, while the family-level method remained between 3.21% and 3.84%.

These are deterministic synthetic simulations under the stated sign-mixture null, not universal guarantees. Wilson intervals in the fixture show Monte Carlo uncertainty. The cluster method's performance depends on correctly identifying independent families and on independent family sampling; it does not correct a wrong cluster definition.

## Power and sample-size reality

The exact power sensitivity table assumes independent families and a common probability \(p\) that a family-level sign exceeds the declared margin. At a one-sided 5% exact sign-test threshold:
- five families have very low power even when \(p=0.8\);
- ten families remain weak for moderate effects;
- 20 families reach about 80.4% power only when \(p=0.8\);
- 30 families reach about 73.0% power at \(p=0.7\) and 97.4% at \(p=0.8\).

This is why we should not claim that 20 or 30 families is universally sufficient. Use the expected family-level effect, practical margin, target family distribution and full Holm family to perform a design-specific power analysis. Exact sign tests are discrete and conservative at small K; do not hide this limitation.

## Target-population boundary

This method estimates an equally weighted family-level median effect. It does not estimate an instance-weighted effect when family sizes differ, nor does it automatically transport results to future workloads. If deployment traffic weights families unequally, preregister those weights and use a separately justified weighted/cluster-aware estimator; do not weight by observed family size after seeing outcomes.

A fixed benchmark census with no sampling frame supports a finite-suite descriptive claim. Population inference requires independently sampled families from a declared target distribution. If family labels conceal common lineage or families are themselves dependent, the method is not valid until the higher-level dependency is modeled.

## Relationship to QARCH-015

- QARCH-013 records comparable workload, platform, resource and provenance results.
- QARCH-015 provides paired instance-level inference and Holm correction.
- QARCH-016 reduces paired instance effects to independent family effects before inference. Feed its single composite p-value per preregistered claim into QARCH-015's Holm procedure. Do not count QARCH-016 instance rows as independent QARCH-015 blocks.

Future extensions should compare this median/sign method against cluster bootstrap and hierarchical models under calibrated null/alternative simulations. Those methods must not be added merely for more power; they must demonstrate calibrated type-I error under their stated assumptions.
