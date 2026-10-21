# Selective Retrieval and Abstention Benchmark

**Status:** executable first protocol; results are not yet qualified  
**Implementation:** `crates/core/symthaea-core/examples/platonic_selective_retrieval.rs`  
**Runner:** `.github/workflows/levin-pattern-experiments.yml`

## Research question

A nearest-neighbour retriever always returns a label—even when the query is an unseen pattern, a corrupted signal, or a mixture that has no unique correct answer. Can calibration reduce incorrect singleton answers, and can set-valued prediction express ambiguity more usefully than a binary answer/abstain decision?

This deliberately tests a practical systems property rather than treating exact algebraic identities as evidence of intelligence. A successful result would establish a selective-retrieval capability on this synthetic protocol, not general reasoning, biological cognition, or state-of-the-art performance.

## Protocol v1

The executable uses five deterministic seeds, 4,096-dimensional bipolar patterns, and 64 stored prototypes per seed.

| Split | Expected known-pattern queries | Expected random OOD queries | Expected two-prototype mixtures | Total draws |
|---|---:|---:|---:|---:|
| Threshold / DCO score selection | ~1,303 | ~372 | ~372 | 2,048 |
| Independent risk certification | ~1,303 | ~372 | ~372 | 2,048 |
| Fresh conformal calibration | ~1,303 | ~372 | ~372 | 2,048 |
| Final evaluation | ~2,608 | ~744 | ~744 | 4,096 |

Every query is sampled from the declared 7:2:2 mixture of known/noisy patterns, random OOD patterns, and ambiguous two-prototype mixtures; the observed category counts are random, not forced to these expectations. For known queries, target prototype and one of seven corruption rates (0%, 10%, 20%, 30%, 40%, 45%, 50%) are sampled uniformly. The threshold-selection, independent risk-certification, fresh conformal-calibration, and final-evaluation streams use independently derived seeds. Neither certification nor conformal calibration is used to select empirical thresholds or the DCO score transform; final evaluation is not used for calibration. All splits share the stored prototype memory, because this protocol tests calibration for a fixed memory; a separate memory-held-out generalization test remains a required next phase.

The random OOD samples have no registered target. A bundled pair of stored prototypes is intentionally ambiguous, so a unique-label acceptance counts as an error. This avoids giving the system credit for confidently guessing one component of a genuinely multi-pattern query.

### Set-valued retrieval: split-conformal prediction sets

The runner includes both a raw-similarity split-conformal baseline and an initial DCO-style score-tuning experiment. The raw baseline calibrates on the **labeled known-query subset only** of the fresh conformal-calibration split, using a(x,y) = 1 - sim(x,p_y). For target miscoverage `alpha=0.05`, it chooses the `ceil((n+1)(1-alpha))`-th ordered calibration score (or q = +infinity if that rank exceeds the sample count) and returns every prototype whose nonconformity is at most q.

The DCO-style variant selects a score transform on the separate selection split from a predeclared gap-penalty grid `lambda ∈ {0, 0.25, 0.5, 1, 2}`. Its candidate-label nonconformity is `a(x,y) = (1 - sim(x,p_y)) + lambda * (max_j sim(x,p_j) - sim(x,p_y))`. The tuning objective minimizes the mean prediction-set size on the selection split subject to its empirical known-query coverage meeting the 95% target. **After the transform is fixed, the final quantile is recomputed from the fresh conformal-calibration split**; the provisional tuning quantile is never used on evaluation data. This is a small DCO-inspired grid experiment, not a reproduction of the full published DCO method, and it may not improve set efficiency on held-out queries.

The runner reports the quantile, calibration count, pooled known-query marginal coverage, mean set size, singleton fraction and empty-set fraction, plus descriptive per-noise strata. The pooled efficiency metrics include a `known_all` row calculated on exactly the known-query population used in the marginal-coverage denominator; `all_diagnostic` is a separately labeled mixture-level set-size diagnostic including OOD and ambiguous inputs. This is a **marginal coverage target over the generator’s known-query distribution**, assuming calibration and future known queries are exchangeable conditional on being known. It is not a class-conditional or per-noise guarantee, and it does not provide semantic coverage for random OOD, ambiguous mixtures, or unregistered identities with no ground-truth prototype in the bank. Set sizes for those probes are diagnostic only.

This changes the interface from “pick one class or abstain” to “return the candidate set justified by calibration.” If set size grows at 40–50% corruption, that is not a free win; it is the visible price of uncertainty. We must compare coverage against set size and singleton coverage, not celebrate marginal coverage alone. The exact-vector identity control remains: no vector-only set rule can distinguish different registry states with identical observed vectors; identity still requires enrollment/provenance evidence.

### Identifiability boundary: vector content is not registry identity

The near-prototype challenge generates observations by flipping 5%, 10%, or 20% of the bits in an enrolled prototype, but labels the result as an **unregistered identity**. Those vectors are deliberately outside the *semantic* registry while being close to, and in the 10%/20% cases drawn using corruption rates that also occur for valid registered queries. This challenge is therefore excluded from the certification distribution: it measures an identifiability limit, not the same risk distribution that was certified.

The executable pairs a known/noisy query and an unregistered query with the **exact same vector** for 10% and 20% corruption. Their registration labels differ, but the vector, nearest-prototype score, top-two margin, and cached prototype similarities are identical. A vector-only decision rule must make the same accept/reject decision for both. This is a negative control for any claim that similarity thresholds alone guarantee unknown-identity rejection. Solving it requires additional evidence about enrollment/identity (for example, a verified template ID or trustworthy registry lookup), or an explicit abstention policy that acknowledges the ambiguity. It cannot be solved by retuning similarity alone without changing the operating trade-off.

### Compared decision rules

1. **Always top-1:** the uncalibrated nearest-prototype baseline; it never abstains.
2. **Score-only:** accept when the top cosine similarity passes a calibration-selected threshold.
3. **Margin-only:** accept when the best-versus-second-best similarity margin passes a calibration-selected threshold.
4. **Score + margin:** accept only when both calibrated conditions pass.

For the empirical-threshold baseline, thresholds are chosen on the selection split by maximizing coverage subject to a preliminary one-sided 95% Wilson upper-risk bound of at most 5%. This is candidate selection, not a risk guarantee. The chosen point is evaluated on an independent certification split. The current protocol marks it `certified=pass` only when at least 125 certification queries are accepted and the one-sided Clopper–Pearson exact-binomial upper bound is at most 5%, with per-policy `alpha=0.05/30` (Bonferroni correction across six threshold strategies and five seeds). Final evaluation is a third held-out split.

A second baseline, fixed-sequence confidence-interval calibration (CIC), uses **one** calibration split rather than a separate threshold-selection and certification split. Its score/margin candidate thresholds are fixed in advance and ordered from strict to permissive. Each point is checked using the same exact binomial upper bound; scan stops at the first failed candidate, and the last point in the certified prefix is returned. Its predeclared threshold sequence controls the threshold-search selection effect; the `0.05/30` correction covers strategy × seed comparisons, not individual thresholds. If no candidate is certified, the method abstains on all queries. Both baselines use the same synthetic query generator. Their guarantee is scoped to that fixed distribution and does not cover arbitrary input, class-mixture, or deployment shifts. Risk limits and minimum counts are certification rules, not proof that a strategy is effective.

## Reported metrics

- **Top-1 accuracy:** correct unique-label predictions across the full evaluation mix; OOD and ambiguous mixture queries cannot count as correct.
- **Coverage:** fraction of evaluation queries accepted rather than rejected.
- **Reason-coded abstention:** each row counts queries rejected by the similarity threshold only, the top-two margin threshold only, or both thresholds. The three rejection categories are mutually exclusive and partition all rejected queries. Interpret them as diagnostics, not perfect semantic labels: a low-score rejection suggests insufficient match evidence; a low-margin rejection suggests competing candidates; both failures indicate weak match evidence and weak separation.
- **Selective risk:** fraction of accepted queries that are incorrect. Risk is undefined when the system accepts no queries.
- **95% Wilson interval:** uncertainty interval for errors among accepted queries.
- **Thresholds and counts:** selection, certification, and evaluation sample counts; score/margin thresholds; independent certification pass/fail; accepted certification count, certification errors, exact-binomial upper-risk bound, held-out accepted count/errors, and all three rejection-reason counts.
- **Stratified diagnostics:** each seed/rule emits an overall row plus separate rows for every corruption level, random OOD queries, and ambiguous mixtures. This prevents pooled risk from hiding failures on a particular query type.
- **Runtime:** total benchmark wall time per invocation, recorded separately from per-row decision quality.
- **Conformal set-valued output:** selected score mode, tuning-split provisional quantile, tuning coverage and mean set size; fresh calibration quantile and known-query calibration count; final marginal target coverage, mean prediction-set size, singleton fraction and empty-set fraction. Per-noise coverage rows are diagnostic only; the guarantee is pooled over known queries, not conditional on a specific corruption rate.

The current executable emits the predeclared 101-point risk–coverage curves and reports both trapezoidal AURC and a right-continuous step-AURC estimate. They are separate integration conventions, not independent evidence; their difference makes discretization/interpolation effects visible. Confidence intervals for AURC, paired same-coverage comparisons, and query-latency distributions remain future work.

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

## Fixed-sequence calibration baseline added

The current executable compares six prespecified policies across five seeds: three empirical-threshold baselines (score, margin, score + margin) and three fixed-sequence CIC baselines. The CIC threshold schedules are **101 predeclared candidates per strategy**, strictly ordered from a high-confidence starting point to permissive thresholds: score-only similarity from 1.0 to -1.0, margin from 0.80 to 0.0, and a combined schedule that relaxes both from similarity 1.0 / margin 0.80 to similarity -1.0 / margin 0.0. Starting with a high-score/high-margin point is intentional: it should admit clean registered patterns before considering noisy or ambiguous examples. The exact-head test includes a deliberately non-monotone-risk fixture to verify that CIC stops at the first failed certificate instead of selecting a later point whose observed risk happens to fall again. These schedules are fixed in source rather than fit to calibration values.

This follows Dong, Hao, and Shinnou (2026), whose *Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control* applies one-sided Hoeffding or exact Clopper–Pearson bounds to a predeclared threshold sequence, stopping at the first failed certificate. The HDC experiment tests whether the fixed-sequence approach transfers to hypervector retrieval; it does not assume an improvement in advance.

The initial DCO-style experiment is now implemented. Its next refinement should compare a broader, preregistered set of candidate score families and tune them on independent folds within the selection budget, while retaining the untouched fresh calibration split for the final quantile. This separation targets set efficiency without allowing score-family selection to invalidate the standard split-conformal marginal-coverage guarantee. It does not certify conditional coverage across noise levels or guarantee performance under distribution shift.

The executable also emits all **held-out risk–coverage points** as `risk_coverage_curve` records and prints a trapezoidal area under the risk–coverage curve (AURC) summary for each seed/strategy. Lower AURC is better, but test-set AURC is descriptive only—it must not be used to pick a threshold. Future work should add AURC uncertainty estimates, direct same-coverage comparisons, calibration-cost comparisons, query-latency distributions, and a faithful reproduction of published one-sided selective-prediction baselines.

**Status:** source and unit tests are committed. The newest CIC plus AURC changes still require an exact-head research-workflow pass before their results can be reported. CIC should only be considered better if held-out risk is controlled and it improves coverage at comparable risk or reduces calibration-data requirements.

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
- Dong, Hao, and Shinnou (2026), “Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control,” *Stats* 9(5), 109. The paper uses a predeclared strict-to-permissive threshold sequence with one-sided Hoeffding or Clopper–Pearson bounds, stopping at the first uncertified point. The HDC port is a benchmark adaptation and must be validated on its exact head before being treated as evidence. <https://doi.org/10.3390/stats9050109>
- Bates, Angelopoulos, Lei, Malik, and Jordan (2021), “Distribution-Free, Risk-Controlling Prediction Sets.” A relevant reference for finite-sample-calibrated set-valued output and risk control; the current HDC implementation is a simpler split-conformal classification adaptation, not a reproduction of every method in that paper. <https://arxiv.org/abs/2101.02703>
- Wu, Niu, Kaski, and Caprio (2026), “Decoupled Conformal Optimisation: Efficient Prediction Sets via Independent Tuning and Calibration.” A current next-step comparator for tuning set efficiency on one split, followed by fresh calibration. This technique is not yet implemented in the HDC benchmark. <https://arxiv.org/abs/2605.18354>
- Yu and Liu (2026), “A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk Control,” preprint. It studies simultaneous certification of risk and acceptance probability under adaptive threshold selection; reported results are on specific image/panoptic settings and are not assumed to transfer to HDC. <https://arxiv.org/abs/2606.08517>
- IEEE CIS Task Force on Hyperdimensional Computing and Vector Symbolic Architectures. <https://cis.taskforce.ieee.org/hdcvsa/>
- Kleyko et al. (2023), “A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II: Applications, Cognitive Models, and Challenges,” *ACM Computing Surveys*. <https://doi.org/10.1145/3558000>
- Bhattacharjee and Martin (2025), “Accuracy and capacity of modern Hopfield networks with synaptic noise,” *Physical Review E* 112, 035313. This is a distinct associative-memory family and a later comparison target, not evidence for the current implementation. <https://doi.org/10.1103/nwbb-n1bc>
