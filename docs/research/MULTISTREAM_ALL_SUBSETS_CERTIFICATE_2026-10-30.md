# All-subsets certificate for running maxima

**Status:** local synthetic pilot; 11/11 local unit tests passed. This is a research implementation of a SAVI 2026 poster result, not a new theorem or SOTA claim.

## Motivation

A pre-registered group test has to know the group before looking at evidence. The poster's all-subsets inequality offers a different contract: control the chance of falsely rejecting any all-null subset, while allowing the reported subset to be selected after seeing evidence.

For running maxima \(V_k=\max(1,\sup_t E_t^k)\), sort them as \(V_{(1)}\geq\cdots\geq V_{(K)}\), and let \(H_m=\sum_{r=1}^m 1/r\). The poster states:

\[
\Pr\left(\exists\,\emptyset\ne S\subseteq[K]:
\frac{\sum_{k\in S}V_k}{\sqrt{H_{|S|}}}>\lambda\right)
\leq \frac{2K\sqrt{H_K}}{\lambda}.
\]

For each fixed size \(m\), the largest subset sum is the sum of the top \(m\) values. Thus the maximum across all \(2^K-1\) subsets can be computed by sorting once and scanning top-\(m\) prefixes:

\[
T(V)=\max_{1\leq m\leq K}
\frac{\sum_{r=1}^{m}V_{(r)}}{\sqrt{H_m}},
\qquad
Z_{\rm all}= \frac{T(V)}{2K\sqrt{H_K}}.
\]

Under the poster's assumptions, \(\Pr(Z_{\rm all}\geq t)\leq 1/t\). At level \(\alpha\), reject a selected group when \(Z_{\rm all}\geq 1/\alpha\). The implementation is \(O(K\log K)\) time and \(O(K)\) space rather than exponential subset enumeration.

**Scope:** This is a simultaneous group/intersection-null certificate. It is not an individual e-value, not ordinary e-BH, not per-hypothesis FDR-sup, and does not by itself implement full closed testing. A selected group rejection supports the claim that the selected group is not wholly null; it does not tell which members are non-null. Strong familywise interpretation depends on applying the poster theorem to the fixed true-null subset family and using the full-K normalizer as an upper bound.

Poster: https://riannedeheide.github.io/PosterSAVI2026Yury.pdf  
The authors list the associated paper as in preparation: https://riannedeheide.github.io/research

## Pilot

The local generator uses likelihood-ratio e-processes for \(N(0,1)\) versus \(N(0.1,1)\), independent observations over time, and equicorrelated Gaussian innovations across streams. It compares the adaptive all-subsets statistic to:
- the simple max-Bonferroni test;
- the direct s=1,2,4 group power statistics.

All-null Type-I screen: K=200, 500 timepoints, 3,000 repetitions each. All-subsets had 0/3,000 rejections for both rho=0 and rho=0.3 (Wilson 95% upper endpoint 0.128%). These are finite-simulation screens, not proof of calibration.

| K | Signals q | All subsets | Max Bonferroni | Direct s=4 |
|---:|---:|---:|---:|---:|
| 20 | 1 | 33.8% | 48.4% | 45.2% |
| 20 | 20 | 96.4% | 98.2% | 98.2% |
| 100 | 20 | 81.4% | 92.8% | 90.6% |
| 100 | 100 | 97.6% | 99.2% | 99.2% |
| 200 | 20 | 66.8% | 83.2% | 81.2% |
| 200 | 100 | 94.0% | 98.4% | 98.2% |

Each power row uses 500 repetitions, 1,000 timepoints, alpha=.05, mean shift=.1 and rho=.3. The statistic selects a subset size after observing the data; the reported mean selected sizes are preserved in the JSON. Exact Wilson intervals are also preserved.

This benchmark is not an improvement over max-Bonferroni in power. The price is expected: simultaneous protection across every possible subset is stronger than one fixed group test. The result's potential value is selective group discovery with a non-exponential algorithm and an explicit simultaneous error bound.

## Scaling probe

One-vector timing on this runtime, excluding data generation:
- K=1,000: 0.321 ms
- K=10,000: 0.744 ms
- K=100,000: 9.604 ms

These are a microbenchmark, not a comparative production benchmark. The implementation uses NumPy sorting; a Rust implementation should be benchmarked separately.

## Tests and artifacts

11/11 local unit tests passed, including:
- exact match against exhaustive enumeration for K=7;
- order invariance;
- subset-size bounds;
- monotonicity;
- invalid-input rejection;
- reproducible simulations;
- Wilson interval boundaries.

Local executed hashes:
- source: \`7ce5ebfa8f17fe996a5668d4734a4651814e6886fc3bad65800fe3699f744b89\`
- tests: \`0d1914545744e8fba4e4a8ed7f3656197499ea6634a2c6f28cdee57fde227c79\`
- pilot output: \`99e02696b5456a224b6264b61953a8ae2c1e50e402025c272c21a6fdd760d2e2\`

## Next gates

1. Reconcile the exact theorem assumptions and whether the full-K normalizer controls every true-null subset under the intended familywise interpretation.
2. Test adversarial dependence, asynchronous observation schedules, and optional stopping.
3. Use the selected subset and simultaneous certificates to develop lower confidence bounds on true discoveries, then compare to dynamic e-closure.
4. Extend to heavy-tailed and misspecified likelihood ratios.
5. Compare against e-closure, BY, e-BH, and Bonferroni only on matched error targets.
6. Port only the reviewed statistic to Rust; maintain a Python reference for differential tests.
