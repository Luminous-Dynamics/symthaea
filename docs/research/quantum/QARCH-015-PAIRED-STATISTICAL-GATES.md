# QARCH-015 — Preregistered paired statistical gates for quantum advantage claims

**Issue:** [#9117](https://github.com/Luminous-Dynamics/symthaea/issues/9117)  
**Implementation:** `crates/domains/symthaea-quantum-comp/src/paired_advantage.rs`  
**Schema:** `docs/research/quantum/schemas/QARCH-015-PAIRED-ADVANTAGE.schema.json`  
**Fixture:** `docs/research/quantum/benchmarks/QARCH-015-PAIRED-ADVANTAGE-FIXTURES.json`

## Why this is the next gap

QARCH-003 already owns end-to-end resource per independently verified acceptable answer. QARCH-013 gives us an interoperable result ledger. Neither one, by itself, protects a benchmark programme from pseudo-replication or false discoveries after trying many workloads, hardware configurations, baselines and resource metrics.

The existing small sign-test helper is suitable for local diagnostics; it is not a suite-level confirmatory protocol. A normal-approximation interval around a mean is also not a substitute for multiplicity control. QARCH-015 adds a narrow, distribution-light gate rather than another score or simulator.

## Estimands and direction

For independent paired block \(i\), let \(Q_{q,i}\) and \(Q_{c,i}\) be quantum and classical quality, with larger values better. Let \(R_{q,i}\) and \(R_{c,i}\) be end-to-end resource per verified acceptable answer, with smaller values better.

Quality non-inferiority requires:

\[
D_{Q,i}=Q_{q,i}-Q_{c,i}>-\delta_Q
\]

Resource superiority requires a predeclared practical reduction from the classical resource baseline, \(\delta_R\in[0,1)\):

\[
D_{R,i}=\log(R_{c,i})-\log(R_{q,i})>-\log(1-\delta_R)
\]

Both conditions are required. A statistically detectable but practically trivial difference does not pass the resource gate. A faster/cheaper system that fails the common output-quality requirement does not pass either.

The API reports the two exact one-sided sign-test p-values and the joint intersection-union p-value:

\[
p_{joint}=\max(p_{quality},p_{resource})
\]

This is the appropriate conservative p-value for the preregistered conjunction “quality non-inferior AND resource reduction exceeds the minimum.” The conjunction must be fixed before outcomes are inspected. The claim is not accepted until this joint p-value is adjusted across the full preregistered family.

## Inferential unit: one independent block

A block is one independently sampled workload unit with one quantum result and one classical result. The paired runs must share the exact workload, objective, output-quality contract, verifier, resource metric/unit and resource scope. The comparison-contract digest binds those semantics.

Shots, retries, repeated seeds, timing samples, compiler repetitions and solver restarts within one workload instance are *nested measurements*, not independent benchmark units. Reduce them using a preregistered rule before passing one row to this API. The API rejects duplicate block IDs and duplicate workload fingerprints. It also rejects records with differing comparison-contract digests.

The exact sign test retains threshold ties in the denominator as non-wins. This is conservative when observations can land exactly on the practical margin; simply dropping many ties can create an anti-conservative conditional win rate.

The binomial sign-test p-value assumes independent blocks and a valid sampling frame for the declared target population. A fixed benchmark census with no defensible sampling model supports a finite-suite descriptive result, not automatic population-level inference. Instances sharing a generator, source lineage or workload family may be clustered; aggregate at the independent cluster level or use an independently justified cluster-aware method. Do not turn 1,000 correlated seeds into 1,000 independent blocks.

## Multiple claims: Holm correction

A programme may test several workloads, backends, resource dimensions, encodings, mitigation settings or baseline choices. Selecting the best raw p-value after the fact is not acceptable.

Pass one joint p-value per preregistered claim to `holm_adjust`. Holm's step-down procedure controls family-wise error without requiring independent p-values. The family must include every confirmatory claim that could be selected for the headline statement. Exploratory analyses must be labelled exploratory and must not be silently promoted into the confirmatory family.

Do not Bonferroni-correct quality and resource components separately when they are deliberately combined into this single intersection-union claim; the joint p-value is their maximum. Apply Holm across the independent claim family.

## Minimal execution protocol

1. Freeze benchmark/version, instance selection, target population, workload and objective fingerprints, verifier, output-quality metric, resource metric/unit/scope, strongest classical baseline, tuning budget and compiler/hardware configuration.
2. Preregister one primary resource metric, the minimum meaningful resource reduction, quality non-inferiority margin, one-sided direction, exclusion rules, block-level reduction, alpha and the full claim family.
3. Interleave or randomize quantum/classical run order. Use equivalent warm-up, compilation, data loading, retry and verification rules. Record the complete pipeline, not just device execution.
4. Preserve per-attempt receipts. Reduce repeated runs within each independent block using the frozen rule. Retain failures and timeouts; do not delete them from denominators.
5. Compute the paired quality/resource tests and joint p-value. Apply Holm to all confirmatory claims.
6. Publish every block, exclusion, raw measurement, uncertainty, contract digest, adjusted p-value and Pareto dimension. Report wall time, energy, cost, quality and tail latency separately; do not collapse regressions into one weighted number.

## Benchmark targets (proposed, not achieved)

- Pilot: at least 20 independent paired blocks to expose protocol flaws and estimate variability.
- Confirmatory design target: 30–50 or more independent blocks when the target population and cost permit. This is not a guarantee of adequate power; perform a sensitivity/power analysis against the predeclared minimum effect and account for family clustering.
- 100% agreement with hand-computed exact sign-test and Holm fixtures.
- Zero false positives across analytic null/negative-control fixtures.
- 100% rejection of deliberate pairing, duplicate-block, contract, denominator and provenance violations.
- Report all primary and secondary metrics, not just the metric/backend that wins.

## Industry comparison and benchmark mapping

- **QOBLIB:** use only an exact, semantically matched optimization class and instance with the same objective, constraints and success criterion. QOBLIB's 2026 release spans ten model-independent problem classes and a large instance collection; its classical solver baselines are a necessary comparison, not an optional reference. The instance collection is not automatically an IID sample: block by problem family/instance lineage before making population claims. [QOBLIB paper](https://doi.org/10.1007/s10994-026-06807-5) · [QOBLIB suite](https://zib-aopt.github.io/QOBLIB/)
- **IonQ application benchmarking:** preserve application-level output quality and end-to-end time-to-solution; do not compare component fidelities to application outcomes. [Framework](https://www.ionq.com/resources/benchmarking-white-paper)
- **QUOPS:** report system capacity/throughput alongside application utility. It is a complementary system-level axis, not a correctness oracle for a given application. [QUOPS overview](https://www.quantinuum.com/blog/introducing-quops-a-new-way-to-measure-the-performance-of-quantum-computers)
- **Photonic references:** keep state-preparation/readout, HOM, fusion and interconnect fidelities conditional on their reported detection/conditioning events. They cannot be used as end-to-end answer success probabilities without the loss and resource chain. [Integrated photonics reference](https://www.nature.com/articles/s41586-025-08737-1)

The strongest classical baseline is workload-dependent. Match implementation quality and tuning budgets, not just the algorithm label. Report Pareto dominance across verified quality, wall time, energy, cost and tail latency. A result that wins only because compilation, data-loading, retries or verification were excluded is not a fair crossover.

## Scope and claim ceiling

QARCH-015 is an inference primitive. It does not establish the independence of a supplied sample, verify measurement receipts, create the comparison contract, correct a weak baseline, or prove that a workload represents future deployment. Those are upstream responsibilities and must be evidenced.

A Holm-adjusted result can support a claim about the preregistered target population under the declared sampling assumptions. It cannot establish universal quantum advantage, prove a complexity-theoretic separation, or generalize to unmatched applications. Synthetic fixtures validate arithmetic only.