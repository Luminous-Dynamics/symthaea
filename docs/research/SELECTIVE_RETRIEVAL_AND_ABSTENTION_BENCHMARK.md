# Selective Retrieval and Abstention Benchmark

**Status:** executable first protocol; results are not yet qualified  
**Implementation:** `crates/core/symthaea-core/examples/platonic_selective_retrieval.rs`  
**Runner:** `.github/workflows/levin-pattern-experiments.yml`

## Research question

A nearest-neighbour retriever always returns a label—even when the query is an unseen pattern, a corrupted signal, or a mixture that has no unique correct answer. Can a simple calibration layer reject unsafe guesses while preserving useful recall?

This deliberately tests a practical systems property rather than treating exact algebraic identities as evidence of intelligence. A successful result would establish a selective-retrieval capability on this synthetic protocol, not general reasoning, biological cognition, or state-of-the-art performance.

## Protocol v1

The executable uses three deterministic seeds, 4,096-dimensional bipolar patterns, and 64 stored prototypes per seed.

| Split | Known-pattern queries | Random out-of-distribution (OOD) queries | Two-prototype mixtures | Total |
|---|---:|---:|---:|---:|
| Threshold selection | 224 | 64 | 64 | 352 |
| Independent certification | 224 | 64 | 64 | 352 |
| Final evaluation | 448 | 128 | 128 | 704 |

Known queries span exact retrieval and bit-flip corruption levels of 0%, 10%, 20%, 30%, 40%, 45%, and 50%. Each split uses independent corruption and OOD-generation seeds. Threshold selection and certification never use final evaluation queries. All splits share the stored prototype memory, because this protocol tests threshold calibration for a fixed memory; a separate memory-held-out generalization test remains a required next phase.

The random OOD samples have no registered target. A bundled pair of stored prototypes is intentionally ambiguous, so a unique-label acceptance counts as an error. This avoids giving the system credit for confidently guessing one component of a genuinely multi-pattern query.

### Compared decision rules

1. **Always top-1:** the uncalibrated nearest-prototype baseline; it never abstains.
2. **Score-only:** accept when the top cosine similarity passes a calibration-selected threshold.
3. **Margin-only:** accept when the best-versus-second-best similarity margin passes a calibration-selected threshold.
4. **Score + margin:** accept only when both calibrated conditions pass.

Thresholds are chosen only on the selection split by maximizing coverage subject to a preliminary one-sided 95% Wilson upper-risk bound of at most 5%, and at least 30 accepted selection queries. This is candidate selection, not a risk guarantee. The selected threshold for each candidate rule is then evaluated on a second, independent certification split. A rule is marked `certified=pass` only when it accepts at least 87 certification queries and its one-sided Wilson upper risk bound is at most 5% using the Bonferroni-adjusted critical value for the three candidate policies compared per seed. The certification split is never used to tune thresholds, and final evaluation is a third held-out split. If certification fails, the rule's held-out metrics are still reported for diagnosis, but it must not be described as risk-certified.

## Reported metrics

- **Top-1 accuracy:** correct unique-label predictions across the full evaluation mix; OOD and ambiguous mixture queries cannot count as correct.
- **Coverage:** fraction of evaluation queries accepted rather than rejected.
- **Selective risk:** fraction of accepted queries that are incorrect. Risk is undefined when the system accepts no queries.
- **95% Wilson interval:** uncertainty interval for errors among accepted queries.
- **Thresholds and counts:** selection, certification, and evaluation sample counts; score/margin thresholds; independent certification pass/fail; accepted count and errors.
- **Runtime:** total benchmark wall time per invocation, recorded separately from per-row decision quality.

The first version reports risk/coverage points rather than claiming a complete area-under-risk-coverage result. A later protocol should add full risk–coverage curves and normalized area under the risk–coverage curve (AURC), as well as calibration cost and query latency distributions.

## What would count as progress?

A rule is not better merely because it abstains more. The primary comparisons are:

- At a fixed evaluation coverage, lower selective risk than score-only and margin-only.
- At a predeclared risk ceiling, higher coverage than each one-dimensional threshold baseline.
- Certification and final evaluation remain independent of threshold selection. Comparisons among three candidate policies use the predeclared Bonferroni correction.
- The direction of the difference persists across at least five independent seeds and multiple memory sizes.
- Final evaluation risk/coverage is reported with intervals; certification on its own does not replace final evaluation.
- Unknown-pattern and mixture rejection remains effective as stored memory grows and distractors become correlated or near-duplicates.

A reasonable next target is an evaluation-set upper 95% Wilson bound on accepted-query risk below 5% while retaining at least 70% coverage on this protocol. This is a **target to test**, not a result or a universal product guarantee. If no rule meets it, preserve that negative result.

## Expansion plan: make it comparable to the field

The HDC/VSA literature spans different algebras and application-specific datasets, which complicates cross-paper performance comparisons. The benchmark should grow from a controlled unit test into a budget-matched suite:

1. **Scaling:** dimensions 1,024 / 4,096 / 16,384 / 65,536; memory sizes 16 / 64 / 256 / 1,024 / 4,096; at least five seeds in smoke mode and substantially more for final estimates.
2. **Adversarial challenge:** correlated distractors, duplicated/near-duplicate prototypes, mixtures of 2/3/5 patterns, distribution shifts, random OOD, and corruption sweeps. Report all defined cases, including catastrophic failures.
3. **Algebra baselines:** MAP/BSC, real-valued HRR, FHRR and TPR where implementations can be independently checked; nearest prototype, sparse distributed memory, and modern Hopfield/attention cleanup as separate algorithmic baselines. Do not label a simplified cleanup implementation as a faithful modern-Hopfield reproduction.
4. **Fair resource budgets:** compare both equal vector dimension and equal bytes, including prototype payload, numeric precision, auxiliary indexes, and model/optimizer state. Also report wall-clock build and query time, peak memory where measurable, throughput, and batch size.
5. **Reproducibility:** pin the toolchain and source SHA, hash raw outputs, use independent seeds, predeclare operating points, and separate calibration from final evaluation. For final claims, target at least 1,000 accepted or attempted queries per reported operating point when feasible, with uncertainty intervals and per-seed breakdowns.
6. **Real workloads:** add the established tabular HDC/VSA benchmark sets only after the synthetic protocol is stable, using published splits and identical preprocessing. Synthetic retrieval results must not be conflated with end-to-end task accuracy.

A useful capacity metric for the expanded suite is **K95**: the largest stored-memory size for which the *lower* 95% confidence bound on held-out recall remains at least 95%, reported separately at fixed byte budgets and noise levels. Capacity must be plotted against both dimensions and total bytes; one easy small-memory point is not evidence of scalable capacity.

## Interpretation guardrails

- The query generator and calibration distribution define what this test measures. An OOD model can fail on realistic unknowns even if it succeeds here.
- The fixed prototype bank is shared across calibration and evaluation; this is threshold generalization to new queries, not new-class generalization.
- Threshold selection itself uses labels and the predeclared synthetic mixture semantics. Report this calibration cost.
- A risk target is a statistical operating objective, not a guarantee for future deployment.
- Beating these baselines would justify a stronger experiment. It would not by itself demonstrate SOTA across HDC/VSA or neural associative-memory research.

## References

- Vergés et al. (2025), “Classification using hyperdimensional computing: a review with comparative analysis,” *Artificial Intelligence Review*. The review highlights that methods often use different benchmarks and provides a shared comparison over established datasets. <https://doi.org/10.1007/s10462-025-11181-2>
- IEEE CIS Task Force on Hyperdimensional Computing and Vector Symbolic Architectures. <https://cis.taskforce.ieee.org/hdcvsa/>
- Kleyko et al. (2023), “A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II: Applications, Cognitive Models, and Challenges,” *ACM Computing Surveys*. <https://doi.org/10.1145/3558000>
- Bhattacharjee and Martin (2025), “Accuracy and capacity of modern Hopfield networks with synaptic noise,” *Physical Review E* 112, 035313. This is a distinct associative-memory family and a later comparison target, not evidence for the current implementation. <https://doi.org/10.1103/nwbb-n1bc>
