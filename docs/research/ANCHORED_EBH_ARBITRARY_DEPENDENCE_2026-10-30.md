# Anchored e-BH under arbitrary dependence: independent reproduction

**Status:** independent research implementation; 12/12 local unit tests passed; 20,000-run null boundary experiment and 18 paired power settings completed. This is an offline fixed-family benchmark, not a sequential SupFDR result and not a SOTA claim.

## New 2026 research direction

Aaditya Ramdas, *Anchored multiple testing: a transparent use of e-closure to improve FDR procedures*, arXiv:2610.11279 (version dated October 15, 2026), introduces anchoring: augment a baseline FDR procedure with additional valid intersection e-values so that the baseline rejection set remains certified while some unused FDR budget can fund extra discoveries.

Section 7 gives Anchored-eBH under arbitrary dependence. For marginal e-values \(e_1,\ldots,e_m\), each satisfying \(E[e_i]\leq1\) when its null is true, define for every nonempty intersection \(S\):

\[
E_S = \frac{1}{m}\left(\sum_{i\in S}e_i + m-|S|\right).
\]

Under the intersection null \(H_S\), every \(i\in S\) is null, so by linearity \(E[E_S]\leq1\), with no dependence restriction. E-closure selects the largest-cardinality rejection set \(\bar B\) satisfying

\[
\operatorname{FDP}_S(\bar B)\leq\alpha E_S
\quad \text{for every nonempty }S\subseteq[m].
\]

The construction preserves ordinary e-BH's baseline set pointwise and can add discoveries. It is incomparable with mean closed e-BH; it is not a universal replacement for all closure methods.

Reference: https://arxiv.org/abs/2610.11279

## Exact computational reduction

For a candidate rejection set \(R\) of size \(r>0\), the closure condition is equivalent to

\[
\sum_{i=1}^{m}
\left(1+\frac{m}{\alpha r}\mathbf{1}\{i\in R\}-e_i\right)_+
\leq m.
\]

Let \(h_i=(e_i-1)_+\) and \(C=m-\sum_i(1-e_i)_+\). This rearranges to the equation-(40) form

\[
\sum_{i\in R}\left(\frac{m}{\alpha r}-h_i\right)_+\leq C.
\]

For fixed \(r\), the left side is minimized by selecting the \(r\) largest e-values. Sort once, compute prefix sums of \(h_i\), then for each \(r\) use a binary search to find the positive suffix. This yields \(O(m\log m)\) time and \(O(m)\) memory.

## Correctness work

The independent implementation contains ordinary e-BH, scalar and batched Anchored-eBH, a direct all-intersections closure check, and an exhaustive \(2^m-1\) closure verifier for small \(m\).

It reproduces the paper's strict witness \(m=2,\alpha=.05,e=(39.5,1)\): ordinary e-BH returns no discoveries and Anchored-eBH certifies one.

12 regression tests cover randomized differential comparison against exhaustive closure for 640 small families, baseline containment, selected-set maximality, input boundaries, and scalar/batch agreement.

**Local validation:** 12/12 passed.

## Boundary-focused all-null experiment

Generate independent null e-values with

\[
e_i = \begin{cases}
m/\alpha & \text{with probability }\alpha/m,\\
0 & \text{otherwise}.
\end{cases}
\]

Each has expectation exactly one. When at least one spike occurs, ordinary e-BH rejects at least one true null; since all hypotheses are null, FDR equals the probability of any rejection.

At \(m=100,\alpha=.05\), 20,000 repetitions:
- e-BH: 979 false discoveries, empirical FDR 4.895%, Wilson 95% interval 4.605–5.203%.
- Anchored-eBH: 979 false discoveries, empirical FDR 4.895%, same interval.
- Mean additional rejections: zero in this deliberately tight boundary construction.

This is a designed finite-sample stress test; it is not a proof. It checks the expected behavior that anchoring must not spend more budget when no slack is available.

## Paired power benchmark

Generate likelihood-ratio e-values for \(N(0,1)\) versus \(N(\mu,1)\):

\[
e_i=\exp(\mu Z_i-\mu^2/2).
\]

Null coordinates have \(Z_i\sim N(0,1)\), so their marginal expectation is one. Alternatives have \(Z_i\sim N(\mu,1)\). Hypothesis statistics are either independent or positively equicorrelated at \(\rho=.5\); the methods' guarantees require only marginal e-value validity.

Each setting uses \(m=100\), 5,000 paired repetitions, \(\alpha=.05\), and effects \(\mu\in\{2.5,3,4\}\) with 10%, 50% and 90% nonnulls. Complete results, FDR intervals, power intervals and paired power-gain intervals are in the JSON artifact.

Selected effect-\(\mu=3\) results:

| Dependence | Nonnulls | e-BH power | Anchored-eBH power | Paired gain (95% CI) | Anchored FDR |
|---|---:|---:|---:|---:|---:|
| Independent | 10% | 26.58% | 27.21% | +0.64 pp (0.54, 0.73) | 0.327% |
| Independent | 50% | 52.44% | 59.65% | +7.22 pp (7.12, 7.31) | 0.513% |
| Independent | 90% | 61.98% | 72.69% | +10.71 pp (10.64, 10.79) | 0.137% |
| Equicorrelation .5 | 10% | 28.02% | 28.87% | +0.85 pp (0.74, 0.96) | 0.673% |
| Equicorrelation .5 | 50% | 49.24% | 54.77% | +5.54 pp (5.39, 5.69) | 0.998% |
| Equicorrelation .5 | 90% | 58.71% | 67.12% | +8.40 pp (8.23, 8.58) | 0.774% |

Interpretation:
- Gains concentrate in dense or intermediate alternatives, consistent with the anchoring paper's analysis.
- Sparse settings preserve the baseline but provide much smaller gains.
- Estimated FDRs are below alpha in all displayed conditions, but these simulations are conservative; empirical FDR is not the validity proof.
- We do not claim dominance over closed e-BH: the paper explicitly gives pointwise examples in both directions.
- We do not claim superiority over online FDR or dynamic e-closure: those methods solve different arrival/stopping problems.

## Scaling probe

Single-family scalar implementation, one random e-value vector per size:
- \(m=100\): 1.38 ms
- \(m=1,000\): 3.22 ms
- \(m=10,000\): 37.80 ms
- \(m=100,000\): 1.48 s

This is a Python/NumPy microbenchmark, not a head-to-head production benchmark. It confirms the expected growth trend, not a production performance claim. A Rust port needs a separate benchmark and differential parity suite.

## Scope boundaries and next steps

This is an **offline fixed-family FDR procedure** under marginal e-value validity and arbitrary dependence. It does not imply simultaneous SupFDR across repeated looks. Recomputing Anchored-eBH on current e-process values at every time does not inherit a time-uniform guarantee.

Next: differential comparison with eClosure 0.9.7 / mean closed-eBH, recording its GPL-3 license and exact package version; dependence tests with antithetic, common-shock and adversarial e-value constructions; online/anytime benchmark against Xu–Fischer–Ramdas (UAI 2026) and dynamic e-closure; and theorem-led investigation of time-uniform anchoring. No offline formula is informally reused as SupFDR.

Reproduction:
    PYTHONPATH=validation python -m unittest -v test_anchored_ebh.py
    PYTHONPATH=validation python validation/anchored_ebh.py --output /tmp/anchored-ebh-pilot.json
