# Selective Retrieval and Abstention Benchmark

**Status:** executable first protocol; results are not yet qualified  
**Implementation:** `crates/core/symthaea-core/examples/platonic_selective_retrieval.rs`  
**Runner:** `.github/workflows/levin-pattern-experiments.yml`

## Research question

A nearest-neighbour retriever always returns a label—even when the query is an unseen pattern, a corrupted signal, or a mixture that has no unique correct answer. Can a simple calibration layer reject unsafe guesses while preserving useful recall?

This deliberately tests a practical systems property rather than treating exact algebraic identities as evidence of intelligence. A successful result would establish a selective-retrieval capability on this synthetic protocol, not general reasoning, biological cognition, or state-of-the-art performance.

## Protocol v1

The executable uses five deterministic seeds, 4,096-dimensional bipolar patterns, and 64 stored prototypes per seed.

| Split | Expected known-pattern queries | Expected random OOD queries | Expected two-prototype mixtures | Total draws |
|---|---:|---:|---:|---:|
| Threshold selection | ~1,303 | ~372 | ~372 | 2,048 |
| Independent certification | ~1,303 | ~372 | ~372 | 2,048 |
| Final evaluation | ~2,608 | ~744 | ~744 | 4,096 |

Every query is sampled from the declared 7:2:2 mixture of known/noisy patterns, random OOD patterns, and ambiguous two-prototype mixtures; the observed category counts are random, not forced to these expectations. For known queries, target prototype and one of seven corruption rates (0%, 10%, 20%, 30%, 40%, 45%, 50%) are sampled uniformly. The splits use independently derived seeds. Threshold selection and certification never use final evaluation queries. All splits share the stored prototype memory, because this protocol tests threshold calibration for a fixed memory; a separate memory-held-out generalization test remains a required next phase.

The random OOD samples have no registered target. A bundled pair of stored prototypes is intentionally ambiguous, so a unique-label acceptance counts as an error. This avoids giving the system credit for confidently guessing one component of a genuinely multi-pattern query.

### Identifiability boundary: vector content is not registry identity

The near-prototype challenge generates observations by flipping 5%, 10%, or 20% of the bits in an enrolled prototype, but labels the result as an **unregistered identity**. Those vectors are deliberately outside the *semantic* registry while being close to, and in the 10%/20% cases drawn using corruption rates that also occur for valid registered queries. This challenge is therefore excluded from the certification distribution: it measures an identifiability limit, not the same risk distribution that was certified.

The next addition pairs a known/noisy query and an unregistered query with the **exact same vector**. Their registration labels differ, but the vector, nearest-prototype score, and top-two margin are identical. A vector-only decision rule must make the same accept/reject decision for both. This is a falsifiable negative control for the claim that similarity thresholds alone can guarantee unknown-identity rejection. Solving it requires additional evidence about enrollment/identity (for example, a verified template ID or trustworthy registry lookup), or an explicit abstention policy that acknowledges the ambiguity. It cannot be solved by retuning similarity alone without changing the operating trade-off.

### Compared decision rules

1. **Always top-1:** the uncalibrated nearest-prototype baseline; it never abstains.
2. **Score-only:** accept when the top cosine similarity passes a calibration-selected threshold.
3. **Margin-only:** accept when the best-versus-second-best similarity margin passes a calibration-selected threshold.
4. **Score + margin:** accept only when both calibrated conditions pass.

Thresholds are chosen only on the selection split by maximizing coverage subject to a preliminary one-sided 95% Wilson upper-risk bound of at most 5%, and at least 30 accepted selection queries. This is candidate selection, not a risk guarantee. The selected threshold for each candidate rule is then evaluated on a second, independent certification split. A rule is marked `certified=pass` only when it accepts at least 112 certification queries and its one-sided Clopper–Pearson exact-binomial upper bound is at most 5%, using per-rule `alpha=0.05/15` (Bonferroni correction across three candidate policies and five reported seeds). This gate uses an independent certification split, never the data used to tune thresholds; final evaluation is a third held-out split. The query generator samples each query independently from a fixed mixture, allowing the pooled accepted-query error count to be treated as binomial under the declared synthetic model. It is therefore a scoped benchmark certificate—not a distribution-free guarantee for arbitrary inputs, class-mixture changes, or deployment distribution shift. If certification fails, the rule's held-out metrics are still reported for diagnosis, but it must not be described as risk-certified.

## Reported metrics

- **Top-1 accuracy:** correct unique-label predictions across the full evaluation mix; OOD and ambiguous mixture queries cannot count as correct.
- **Coverage:** fraction of evaluation queries accepted rather than rejected.
- **Selective risk:** fraction of accepted queries that are incorrect. Risk is undefined when the system accepts no queries.
- **95% Wilson interval:** uncertainty interval for errors among accepted queries.
- **Thresholds and counts:** selection, certification, and evaluation sample counts; score/margin thresholds; independent certification pass/fail; accepted certification count, certification errors, exact-binomial upper-risk bound, and held-out accepted count/errors.
- **Stratified diagnostics:** each seed/rule emits an overall row plus separate rows for every corruption level, random OOD queries, and ambiguous mixtures. This prevents pooled risk from hiding failures on a particular query type.
- **Runtime:** total benchmark wall time per invocation, recorded separately from per-row decision quality.

The first version reports risk/coverage points rather than claiming a complete area-under-risk-coverage result. A later protocol should add full risk–coverage curves and normalized area under the risk–coverage curve (AURC), as well as calibration cost and query latency distributions.

## Preliminary result from a completed earlier head

Run [39589249419](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39589249419) completed successfully on source SHA `2757da27bb6f294c60f7b0b9ba7b2045cf648f2b` and archived its [research artifact](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39589249419/artifacts/12341831170). These values are historical, exact-head outputs from that revision; they are not results for later source edits.

| Policy | Mean evaluation coverage | Mean selective risk among accepted | Independent certification |
|---|---:|---:|---:|
| Always top-1 | 100.0% | 45.21% | Not applicable |
| Calibrated score-only | 27.66% | 1.87% | 4/5 seeds pass |
| Calibrated margin-only | 55.90% | 2.09% | 5/5 seeds pass |
| Calibrated score + margin | 55.90% | 2.09% | 5/5 seeds pass |

On the separate near-prototype identity challenge, every one of the 384/384 challenge queries was accepted and incorrect for every tested policy on every seed (20 overall rows: 5 seeds × 4 policies). The Wilson lower bound was 0.9901 for the 100% observed false-accept rate. This is not a deployment-risk estimate: it shows the vector-only rules fail to distinguish the challenge's unregistered identities from noisy enrolled vectors.

Two useful negative results emerge. First, the score-only rule has lower coverage, and it failed certification for one seed. Second, the combined score-and-margin rule selected the same thresholds and gave exactly the same outcomes as margin-only across all five seeds; this added score threshold did not improve the operating point. A more complex rule is not progress unless a preregistered comparison shows better risk–coverage performance or stronger robustness.

## What would count as progress?

A rule is not better merely because it abstains more. The exact-binomial gate is a practical baseline; later work should reproduce stronger published selective-risk methods rather than imply that this threshold search is itself SOTA. The primary comparisons are:

- At a fixed evaluation coverage, lower selective risk than score-only and margin-only.
- At a predeclared risk ceiling, higher coverage than each one-dimensional threshold baseline.
- Certification and final evaluation remain independent of threshold selection. Comparisons among three candidate policies use the predeclared Bonferroni correction for the exact-binomial gate.
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

- Vergés et al. (2025), “Classification using hyperdimensional computing: a review with comparative analysis,” *Artificial Intelligence Review*. The review highlights inconsistent evaluation and reports both seven commonly used HD/VSA datasets and 121 UCI classification tasks. <https://doi.org/10.1007/s10462-025-11181-2>
- Gangrade, Kag, and Saligrama (2021), “Selective Classification via One-Sided Prediction,” AISTATS/PMLR. A published high-accuracy/low-error selective-classification baseline to study and reproduce; it is not implemented in protocol v1. <https://proceedings.mlr.press/v130/gangrade21a.html>
- Xu, Guo, and Wei (2025), “Selective Conformal Risk Control,” preprint. A relevant conformal/selective risk-control comparator. <https://arxiv.org/abs/2512.12844>
- “Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control” (published 2 October 2026). The fixed-sequence threshold-testing idea is a current comparison target for reducing the data inefficiency of selecting thresholds on one split and certifying on a second. It must be implemented or independently reproduced before any superiority claim. <https://www.mdpi.com/2571-905X/9/5/109>
- Yu and Liu (2026), “A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk Control,” preprint. It studies simultaneous certification of risk and acceptance probability under adaptive threshold selection; reported results are on specific image/panoptic settings and are not assumed to transfer to HDC. <https://arxiv.org/abs/2606.08517>
- IEEE CIS Task Force on Hyperdimensional Computing and Vector Symbolic Architectures. <https://cis.taskforce.ieee.org/hdcvsa/>
- Kleyko et al. (2023), “A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II: Applications, Cognitive Models, and Challenges,” *ACM Computing Surveys*. <https://doi.org/10.1145/3558000>
- Bhattacharjee and Martin (2025), “Accuracy and capacity of modern Hopfield networks with synaptic noise,” *Physical Review E* 112, 035313. This is a distinct associative-memory family and a later comparison target, not evidence for the current implementation. <https://doi.org/10.1103/nwbb-n1bc>
