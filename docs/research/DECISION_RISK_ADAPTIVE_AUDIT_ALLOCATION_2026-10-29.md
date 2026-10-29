# Adaptive audit allocation under anytime-valid confidence sequences

**Date:** 2026-10-29  
**Status:** Synthetic prototype; not production certification  
**Question:** Can an audit gate reduce the number of observations by sampling the currently most decision-critical risk cell rather than allocating the same sample count to every cell?

## Summary

This lane compares uniform allocation to a simple adaptive scheduler using the same per-cell beta-mixture confidence sequences, the same random streams, the same predeclared support, and the same stopping categories. After a 128-observation pilot for each of eight cells, uniform allocation adds a batch of 128 observations to every cell. The adaptive heuristic adds its next batch to the cell with the largest current upper bound on absolute model error. It uses per-cell time-uniform confidence sequences with familywise delta = 0.10 divided by eight; selection is based only on past samples from independent cell streams.

Across 200 fixed-seed repeats, adaptive allocation reduced average total observations by **55.5% on the correct model**, while both strategies admitted the model in 200/200 repeats. For a model with a true 0.02 error in the highest-variance cell, uniform admitted 184/200 and adaptive admitted 183/200; the rest were refused unresolved; adaptive allocation used 68.7% fewer observations on average. In the clearly over-budget case, both rejected 61/200 and refused 139/200 unresolved; adaptive allocation used 83.5% fewer observations on average. There were zero observed unsafe admissions in either strategy or scenario.

These outcomes support the hypothesis that sampling where the current certificate is weakest can reduce audit cost. They do **not** establish optimality, industry superiority, or zero real-world risk. The near-threshold corrupted model was refused unresolved, not positively identified as invalid. The allocation heuristic is a prototype and the test uses a small synthetic Bernoulli model.

## Protocol

- Fixed seed 20261029, 200 repeated runs.
- Eight independent Bernoulli risk cells with true means [0.015, 0.035, 0.075, 0.14, 0.27, 0.50, 0.73, 0.94], spanning low through high variance.
- Declared normalized risk range [0, 1]; all risk quantities in this benchmark are normalized risk units, not bits.
- Decision risk budget 0.20 and inherited decision radius 0.05, leaving a maximum cell-error allowance of 0.05.
- Initial pilot: 128 observations per cell; additional audit batch: 128 observations; per-cell cap: 8,192.
- Methods use the same beta-mixture e-process confidence sequence and per-cell delta = 0.10/8. The audit schedule may be adaptive because each cell has an anytime-valid sequence under the stipulated independent Bernoulli stream; familywise validity follows by union bound over the predeclared eight-cell support.
- Both strategies use matched per-cell random streams. The simulator truth is used to generate observations and score the false-admission event, but not to calculate the confidence interval.

### Strategies

**Uniform:** at each look, add 128 observations to all cells and stop when the maximum upper error bound is within budget, any cell's lower error bound exceeds budget, or the sample cap is reached.

**Adaptive:** after the common pilot, add 128 observations to the eligible cell with the largest current upper error bound. If a cell hits the cap and remains ambiguous above the admission threshold, immediately refuse unresolved rather than spend observations on unrelated cells that cannot repair the blocked certificate.

A lower confidence bound exceeding the error allowance triggers a positive rejection. A certificate whose upper error bound cannot be brought within allowance triggers unresolved refusal, not admission. This distinction is material: an unresolved refusal is fail-closed, but it is not evidence that the model is false.

## Results

### Total audit cost and stopping outcomes

| Scenario | Strategy | Mean total observations | 95th percentile | Admit | Reject | Refuse unresolved |
|---|---|---:|---:|---:|---:|---:|
| Correct model (error 0) | Uniform | 21,258 | 30,720 | 200/200 | 0 | 0 |
| Correct model (error 0) | Adaptive | 9,460 | 11,520 | 200/200 | 0 | 0 |
| Safe offset (error 0.02) | Uniform | 36,925 | 65,536 | 184/200 | 0 | 16 |
| Safe offset (error 0.02) | Adaptive | 11,555 | 14,854 | 183/200 | 0 | 17 |
| Near budget violation (error 0.051) | Uniform | 65,536 | 65,536 | 0 | 0 | 200 |
| Near budget violation (error 0.051) | Adaptive | 11,837 | 13,062 | 0 | 0 | 200 |
| Clear violation (error 0.07) | Uniform | 59,515 | 65,536 | 0 | 61 | 139 |
| Clear violation (error 0.07) | Adaptive | 9,827 | 11,520 | 0 | 61 | 139 |

| Scenario | Mean observations saved by adaptive allocation |
|---|---:|
| Correct model | **55.5%** |
| Safe offset | **68.7%** |
| Near-budget violation (unresolved refusal) | **81.9%** |
| Clear violation (same reject/refuse split) | **83.5%** |

False admissions: **0/200 for all four scenarios under both strategies**. With zero observed events in 200 trials, the Wilson 95% upper endpoint is about 1.88%, so these finite results do not establish a zero population error rate.

The strongest positive result is the exact and safe-offset models: the adaptive scheduler maintains the observed admission/refusal split while sharply reducing average audit cost. In over-budget cases it also avoids spending samples uniformly on unrelated, already-easy cells. The near-threshold case is intentionally conservative: the hardest cell reaches its cap and the run refuses unresolved; the method does not claim to have conclusively rejected that model.

## Why the adaptive scheduler is statistically admissible here

The method is not reusing a fixed-time interval. Each cell has an anytime-valid confidence sequence from a beta-mixture e-process. Within a cell, the next audit outcome is an independent Bernoulli draw with that cell's fixed mean. Cell selection depends only on past observations. The sequence remains valid over its own sample count, and the union bound covers the fixed finite family of eight cells.

This argument stops applying when audits share unmodeled common-mode bias, the cell set is incomplete, the risk matrix or policy mapping changes during audit, or the Bernoulli model is inappropriate. The prior audit-assumption preflight and policy/risk manifest checks remain mandatory; adaptive allocation cannot replace them.

## Research and industry comparison

This is an allocation heuristic, not an implementation of the leading adaptive identification algorithms. Relevant literature points to stronger comparators:

- [An Anytime Algorithm for Good Arm Identification (JMLR 2026)](https://www.jmlr.org/papers/v27/24-0680.html) develops an adaptive threshold-identification strategy and reports improved efficiency over uniform sampling in some settings.
- [Optimal Best Arm Identification with Fixed Confidence](https://proceedings.mlr.press/v49/garivier16a.html) develops Track-and-Stop with an instance-dependent lower bound and asymptotically optimal sampling for best-arm identification. The objective differs from certifying every cell, but allocation and stopping ideas are useful comparison points.
- [Balancing Performance and Costs in Best Arm Identification (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/b0dfe8236981ba84b4d98a7ed0426f0b-Abstract-Conference.html) optimizes a risk functional that directly combines observation cost with the penalty for a wrong recommendation.
- [Best Arm Identification with Possibly Biased Offline Data (UAI 2025)](https://proceedings.mlr.press/v286/yang25b.html) studies an important boundary: without a known bound on offline bias, an adaptive algorithm cannot in general identify the best arm with valid fixed-confidence guarantees.

No published allocation implementation has been run on this exact Symthaea protocol. No SOTA or industry-superiority claim is made.

## Paradigm shifts

1. **Allocate evidence by decision-critical uncertainty.** The current heuristic samples whichever cell's upper model-error bound most threatens admission. Next, compare that rule with APGAI, Track-and-Stop-inspired allocation, variance-only allocation, and a budget-optimal decision-aware scheduler.
2. **Optimize audit cost and refusal cost jointly.** A false admission, a correct admission, a conclusive rejection, and an unresolved refusal have different consequences. The next benchmark should score the cost of each outcome rather than treating all stops equally.
3. **Keep a formal stopping contract.** The confidence-sequence family, fixed cell support, alpha allocation, risk matrix digest, cap behavior, and admissible audit selection rules belong in the certificate.
4. **Treat integrity as orthogonal to statistics.** Bias, missing support and changed policy identity must be caught by independent preflight gates. No sampling heuristic can cure invalid evidence.
5. **Benchmark against a sample-complexity lower bound.** Report the gap between observed cost and an oracle/lower-bound diagnostic, clearly labelled as diagnostic only when it uses simulator truth.

## Reproduction

Files:

- validation/decision_risk_adaptive_audit_allocation.py
- validation/test_decision_risk_adaptive_audit_allocation.py
- validation/DECISION_RISK_ADAPTIVE_AUDIT_SUMMARY_2026-10-29.json
- Raw local result: DECISION_RISK_ADAPTIVE_AUDIT_RESULTS_2026-10-29.json
- Test log: DECISION_RISK_ADAPTIVE_AUDIT_TEST_LOG_2026-10-29.txt

Focused tests: **4/4 passed**. Full 200-repeat sweep: 27.289 seconds. Exact source/test local SHA-256 and Git blobs are:

- Source SHA-256 41fee55be24bb536b3ef3e2521030cbefd0ccd5712718d36e3728d0b6ab4c8f2; Git blob 9be0972f6f8746987ba6c732fed2442fcb90e270.
- Test SHA-256 c22df0c3ac469ddc19f04df7458585a48bce043679853e1d3072b3a915c4f54d; Git blob a4f9e8520751e22822b5b3bd6fa9c1d77b4d1c60.
- Raw JSON SHA-256 e647cf4345a507230481890386bf709e7b68356ad65298ee1e893b04446ba3a3.
- Test log SHA-256 961489bf6d5f2e841e14df81c8f8e60f54896e76f64a68bbe2ad11479c347ec8.

## Limitations

This is an eight-cell iid Bernoulli toy. The risk units, error allowance and prior are illustrative; this is not a real-world loss calibration. The allocation rule is heuristic, not a proven optimal policy; the simulator truth is used only for generating/scoring. A 90% familywise confidence target and 200 trials are not enough to establish low real-world false-admission risk. The helper is Python-only, not integrated into the Rust optimizer or dispatch gate; no Rust build, production evaluation, external baseline execution, or remote CI pass is claimed.
