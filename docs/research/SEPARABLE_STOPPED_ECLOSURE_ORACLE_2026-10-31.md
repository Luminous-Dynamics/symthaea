# Separable raw projective e-closure: exact O(N) intersection oracle

**Status:** exact finite-rational reference implementation. The membership oracle is linear in the number of active hypotheses. The utility optimizer remains exhaustive and exponential. Guarantee scope is simultaneous stopped-FDR, **not SupFDR**.

## 1. Projective raw e-collection

Let \(M_{i,t}\) be a valid coordinate e-process for null \(H_i\) in a common global filtration. Before hypothesis \(i\) activates, require \(M_{i,t}=1\). Fix global weights

\[
w_i\geq0,\qquad \sum_{i\geq1}w_i\leq1.
\]

For every finite intersection \(S\), define

\[
E_t^S
=1+\sum_{i\in S}w_i(M_{i,t}-1)
=1-\sum_{i\in S}w_i+\sum_{i\in S}w_iM_{i,t}.
\]

Under \(H_S=\bigcap_{i\in S}H_i\), each included coordinate is a valid e-process. The expression is a nonnegative convex/subconvex mixture with cash and is therefore a valid intersection e-process under arbitrary cross-coordinate dependence.

The dormant-one convention and nested active sets imply exact future-extension coherence:

\[
E_t^{S'}=E_t^{S'\cap I_t}.
\]

This is the raw projective construction in de Heide (2026). If these raw intersection processes are not time-monotone, the dynamic theorem gives simultaneous **stopped-FDR**, not SupFDR. Do not apply the SupFDR label to this branch.

## 2. New result: all-intersection membership has a closed-form separator

Fix an active universe \(I\), a nonempty rejection set \(R\subseteq I\), coordinate values \(M_i\), and significance level \(\alpha\). Dynamic e-closure requires

\[
\frac{|S\cap R|}{|R|}
\leq
\alpha\left[1+\sum_{i\in S}w_i(M_i-1)\right]
\qquad\text{for every }S\subseteq I.
\]

Rearrange:

\[
\sum_{i\in S}
\left[
\frac{\mathbf 1\{i\in R\}}{|R|}
-\alpha w_i(M_i-1)
\right]
\leq\alpha
\qquad\forall S\subseteq I.
\]

For arbitrary coefficients \(a_i\), the maximum of \(\sum_{i\in S}a_i\) over all subsets \(S\) is \(\sum_i(a_i)_+\), attained by choosing exactly the positive coefficients. Consequently the entire \(2^{|I|}\) intersection family is equivalent to one scalar inequality:

\[
\boxed{
\sum_{i\in I}
\left[
\frac{\mathbf 1\{i\in R\}}{|R|}
-\alpha w_i(M_i-1)
\right]_+
\leq\alpha.
}
\]

The minimum closed-eBH intersection slack is exactly

\[
\boxed{
\operatorname{slack}(R)
=\alpha-
\sum_{i\in I}
\left[
\frac{\mathbf 1\{i\in R\}}{|R|}
-\alpha w_i(M_i-1)
\right]_+.
}
\]

The worst intersection is the subset of active hypotheses with strictly positive coefficients; zero coefficients can be excluded for a canonical witness. For the empty rejection set, FDP is zero for every intersection, so it is always certified.

This is an exact O(N) separation oracle. It avoids enumerating every intersection for each candidate rejection set.

## 3. Resource-aware selection

The small reference optimizer chooses at each update

\[
R_t\in\arg\max_R\sum_{i\in R}v_i
\]

subject to:
- \(R_{t-1}\subseteq R\subseteq I_t\) (irrevocable / nested reports);
- \(\sum_{i\in R}c_i\leq B_t\) (predictable nondecreasing cumulative capacity);
- the O(N) positive-part certificate above.

The resulting reference selection costs \(O(2^N N)\) per update, rather than \(O(4^N)\) when every candidate also enumerates every intersection. This is a substantial reduction but is not polynomial-time optimization.

For fixed rejection cardinality \(k\), the certificate can be rewritten as a baseline plus per-hypothesis selection increments:

\[
b_i=\left[-\alpha w_i(M_i-1)\right]_+,
\quad
a_i(k)=\left[\frac1k-\alpha w_i(M_i-1)\right]_+.
\]

Then the certificate is

\[
\sum_i b_i+\sum_{i\in R}(a_i(k)-b_i)\leq\alpha.
\]

Thus, for each \(k\), maximizing utility under operational cost and this certificate becomes a two-budget cardinality-constrained selection problem. This exposes a route to exact Pareto dynamic programming, meet-in-the-middle, or branch-and-bound. With arbitrary values and costs the optimization is a knapsack-family problem; do not claim a linear-time optimal selector.

## 4. Guarantee boundary

The raw projective intersection e-processes are future-extension coherent. Under de Heide's dynamic e-closure theorem they support simultaneous stopped-FDR at global stopping times, subject to the common filtration and valid coordinate e-process contracts.

The raw values \(E_t^S\) may decrease as component evidence changes. No time-monotonicity is imposed here. Therefore this branch does not provide SupFDR or simultaneous FDR control over the supremum across all times. The separate adjusted branch (#9538) addresses that stronger target by applying a common running-maximum adjuster and checking time-monotonicity; its all-intersection membership calculation is not the same separable formula.

## 5. Implementation

The source module validation/separable_stopped_eclosure.py provides:
- exact Fraction positive-part separator;
- slow exhaustive all-intersections oracle for differential tests;
- nested capacity-aware small-stream reference optimizer;
- independent replay with O(N) certificate checks and exhaustive utility-optimality validation.

The test module validation/test_separable_stopped_eclosure.py covers:
- randomized exact differential tests against exhaustive subsets;
- uncertified high-value negative control;
- capacity-aware nested choices;
- future evidence prefix-invariance;
- valid-but-suboptimal receipt negative control;
- dormant-coordinate neutrality.

Reproduction:

    PYTHONPATH=validation python -m unittest -v test_separable_stopped_eclosure.py

The committed suite has not been executed in this environment. A separate development calculation independently reimplemented the identity and compared it to exhaustive subset enumeration on 475 randomized exact-rational cases with N=1..8, including e-values below one and arbitrary active subsets. All minimum slacks matched. This is development evidence, not CI or a theorem proof.

## 6. Benchmark and SOTA targets

- Membership verifier: O(N) per candidate; benchmark N=10², 10³, 10⁴, 10⁵.
- Action optimizer: exact oracle up to small N; differential-test a Pareto DP / meet-in-the-middle / MILP against it.
- Compare global weights: uniform, geometric, sparse, front-loaded, predeclared priority.
- Compare utility: uniform, heterogeneous, cost-aware and externally declared priority.
- Stress e-values below one, rare large evidence jumps, arbitrary dependence, common shocks, delayed hypotheses, all-null / sparse / dense alternatives and capacity saturation.
- Report stopped-FDR diagnostics, power, utility, cost, number of candidate sets, certificate time, p50/p95 latency and memory.
- Empirical simulations are diagnostics; e-process assumptions and the dynamic e-closure theorem supply validity.

## References

- de Heide (2026), [Dynamic e-closure for online hypotheses with any-time-valid evidence](https://arxiv.org/abs/2608.09927).
- Xu, Fischer & Ramdas (2026), [Online analogs of e-closure and compound e-values](https://proceedings.mlr.press/v337/xu26a.html).
- Bai & Ramdas (2026), [Recertifying decisions via e-closure](https://arxiv.org/abs/2610.35821).
