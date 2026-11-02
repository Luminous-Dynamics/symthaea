# Slack-anchored e-closure for process-valued compound evidence

## 1. Why this is the most promising next step

The online donation ledger has a global compound null-mass bound, but its current output is ordinary weighted e-BH on compound scores. That leaves an opportunity: turn the unused part of the globally precommitted candidate-weight budget into an *intersection-level local e-process*, then use e-closure to retain every baseline discovery while adding any larger certified set.

This follows the anchoring principle in Ramdas (2026), “Anchored multiple testing: a transparent use of e-closure to improve FDR procedures.” The general anchoring/e-closure idea is not claimed as new. The proposed contribution is its process-valued, forward-transfer local evidence construction and an exact finite-prefix solver for the resulting dynamic closure.

Reference: https://arxiv.org/abs/2610.11279

## 2. The compound intersection-mass lemma

Fix a countable global candidate order and deterministic weights \(\gamma_i\ge0\) with

\[
\Gamma_*=\sum_{i\ge1}\gamma_i<1,\qquad
\delta=1-\Gamma_*>0.
\]

Let \(X_{i,t}\) be an adapted, nonnegative, pathwise nondecreasing e-process under candidate i's null, valid in the common global filtration. Define the weighted compound score

\[
Z_{i,t}=\gamma_iX_{i,t}+I_{i,t}-O_{i,t},
\]

where \(I_i\) is cumulative credited capital and \(O_i\) is cumulative source-reserved capital. Assume the existing ledger conditions:

\[
0\le O_{i,t}\le\gamma_i\min(X_{i,t},1),\qquad
0\le\Delta O_{i,t}\le\gamma_i\Delta X_{i,t},
\]

with all capital transfers flowing strictly forward from source i to destination j>i, each reserved unit credited at most once, and incoming capital never re-donated.

The existing ledger argument gives pathwise nonnegativity and time monotonicity:

\[
Z_{i,t}\ge0,\qquad \Delta Z_{i,t}\ge0.
\]

Now take **any fixed finite intersection** \(S\) of hypotheses, and write \(k=\max S\). The key refinement is that the compound null-mass bound applies to arbitrary S, not only to the complete true-null set in a prefix.

Represent every credited amount as a directed forward edge from its original source to its destination. Edges entirely inside S cancel between \(\sum_{i\in S}I_i\) and \(\sum_{i\in S}O_i\). Outgoing edges from S to its complement only reduce the difference. Every incoming edge to S from outside S must originate at an index j<k with j not in S. Unassigned reserve is also an outflow with no compensating incoming edge. Consequently,

\[
\sum_{i\in S}(I_{i,t}-O_{i,t})
\le
\sum_{\substack{j\le k\\j\notin S}}O_{j,t}.
\]

Using the donor cap,

\[
\begin{aligned}
\sum_{i\in S}Z_{i,t}
&\le
\sum_{i\in S}\gamma_iX_{i,t}
+
\sum_{\substack{j\le k\\j\notin S}}
\gamma_j\min(X_{j,t},1).
\end{aligned}
\]

Under the intersection null \(H_S\), each i in S is a true null, so \(\mathbb E X_{i,t}\le1\). For j outside S, \(\min(X_{j,t},1)\le1\) pathwise. Therefore

\[
\boxed{
\mathbb E_{H_S}\left[\sum_{i\in S}Z_{i,t}\right]
\le
\sum_{i\le k}\gamma_i
=\Gamma_k
\le\Gamma_*.
}
\]

This is stronger in scope than a bound for one particular global null set: it holds for every fixed finite intersection S, under arbitrary dependence so long as every base e-process is valid in the *same global filtration*.

For a countable candidate universe and a possibly infinite true-null set N, apply the finite-intersection result to \(N\cap[m]\) and use monotone convergence. This gives

\[
\mathbb E\left[\sum_{i\in N}Z_{i,\infty}\right]\le\Gamma_*.
\]

The finite-active-roster procedure uses only candidates visible at each reporting time; dormant candidates carry zero evidence and zero capital until their predictable activation. The global weights and total mass remain fixed.

## 3. A time-monotone intersection e-collection from leftover mass

For every nonempty finite S, define

\[
\boxed{
F_{S,t}=\delta+\sum_{i\in S}Z_{i,t}.
}
\]

Under \(H_S\),

\[
\mathbb E F_{S,t}
\le\delta+\Gamma_*=1.
\]

Because every \(Z_i\) is adapted, nonnegative and pathwise nondecreasing, \(F_{S,t}\) is also adapted, nonnegative and pathwise nondecreasing. Thus it is a time-monotone intersection e-process under the same global-filtration assumptions. The terminal version is valid by monotone convergence.

The important design choice is that \(\delta=1-\Gamma_*\) is computed from the **globally precommitted infinite weight sequence**, not the current active family size or finite horizon. A convenient projective prior is

\[
\gamma_i=\frac{1-\delta}{i(i+1)},\qquad i=1,2,\ldots,
\]

whose infinite sum is exactly \(1-\delta\) by telescoping. Its first n weights are a prefix of the same sequence; no renormalization occurs when candidates arrive.

The constant \(\delta\) is the uniform leftover mass available for every finite intersection. More aggressive S-specific slack such as \(1-\Gamma_{\max S}\) may be valid for a known finite envelope, but it is not the same horizon-independent constant and needs a separate extension analysis. This implementation deliberately uses the uniform \(\delta\).

## 4. Slack-anchored closure

Let \(B_t\) be the ordinary weighted compound e-BH set, using the self-consistency condition

\[
i\in B_t
\quad\Longrightarrow\quad
Z_{i,t}\ge\frac{1}{\alpha|B_t|}.
\]

For every intersection S,

\[
\operatorname{FDP}_{S}(B_t)
=\frac{|B_t\cap S|}{|B_t|}
\le
\alpha\sum_{i\in S}Z_{i,t}
\le
\alpha F_{S,t}.
\]

Thus B_t is always certified by the local e-collection. Define the anchored rejection set \(\bar B_t\) as a measurable largest-cardinality superset of B_t satisfying

\[
\operatorname{FDP}_{S}(\bar B_t)
\le\alpha F_{S,t}
\quad\text{for every nonempty finite S}.
\]

The largest-set requirement matters: do not replace it with a union of all certified sets, because a union of certified sets need not itself be certified. The original B_t remains an allowed output, and \(\bar B_t\) contains every baseline discovery.

For the actual true-null set N, the closure constraint gives

\[
\operatorname{FDP}_{N}(\bar B_t)
\le\alpha\left(\delta+\sum_{i\in N}Z_{i,t}\right).
\]

The right side is bounded by the terminal time-monotone intersection process. Taking a supremum over reporting times and then expectation gives

\[
\boxed{
\mathbb E[\sup_t\operatorname{FDP}_{N}(\bar B_t)]
\le\alpha.
}
\]

This is a SupFDR consequence of the compound intersection-mass lemma and e-closure; it is not an independent new multiple-testing theorem. It remains conditional on valid base e-processes, a common global filtration, predictable arrivals, a fixed global weight sequence and the original source-only forward-flow accounting assumptions.

## 5. Exact closure condition

For a fixed nonempty rejection set R of size r, the local closure inequality is

\[
\frac{|R\cap S|}{r}
\le\alpha\left(\delta+\sum_{i\in S}Z_i\right)
\quad\text{for all nonempty S}.
\]

Rearrange:

\[
\sum_{i\in S}
\left(\alpha Z_i-\frac{\mathbf1\{i\in R\}}r\right)
+\alpha\delta\ge0
\quad\text{for every S}.
\]

The minimum over all subsets S is obtained by choosing every negative coefficient. Coefficients for i outside R are nonnegative because Z_i≥0. Therefore all closure constraints are equivalent to the single exact deficit condition

\[
\boxed{
\sum_{i\in R}\left(\frac1r-\alpha Z_i\right)_+
\le\alpha\delta.
}
\]

This condition is an anchored version of the offline e-BH donation/closure identity in Ramdas (2026, Section 7). The distinction is that the local e-collection here is built from the process-valued compound scores and global unused weight mass.

## 6. Largest certified superset in O(n log n)

Sort the current active scores in descending order with a deterministic candidate-ID tie-break:

\[
Z_{(1)}\ge\cdots\ge Z_{(n)}.
\]

For any fixed cardinality r, the top-r set minimizes the positive-part deficit, because it uses the r largest scores. It also contains the baseline e-BH set whenever r≥|B_t|.

Let

\[
q_r=\#\left\{i\le r:Z_{(i)}\ge\frac{1}{\alpha r}\right\}.
\]

Only ranks k=q_r+1,...,r contribute a positive deficit. With prefix sums \(P_r=\sum_{k\le r}Z_{(k)}\),

\[
D_r
=\frac{r-q_r}{r}
-\alpha(P_r-P_{q_r}).
\]

The largest certified set is the top-r set for the largest r satisfying

\[
r\ge |B_t|,\qquad D_r\le\alpha\delta.
\]

Algorithm:
1. Sort once in O(n log n).
2. Build exact prefix sums in O(n).
3. For each r=1,...,n, binary-search q_r in O(log n).
4. Keep the largest feasible r at or above the baseline size.

Total time is O(n log n), memory O(n). Scan every r: D_r need not be assumed monotone.

The implementation in benchmarks/anchored_compound_ebh.py uses fractions.Fraction and rejects floats. Its exponential oracle enumerates all subsets for small n.

## 7. Executable validation

The test file benchmarks/test_anchored_compound_ebh.py covers:
- direct all-subsets closure checks against the positive-part deficit identity;
- exact solver parity with exhaustive largest-certified-superset search;
- random rational score vectors;
- preservation of every baseline e-BH discovery;
- a strict-addition witness;
- empty baseline, ties, \(\delta=0\), \(\delta=1\);
- fixed global telescoping weights without horizon renormalization;
- pathwise monotonicity of local intersection processes;
- fail-closed input validation.

The local exact-rational research checks completed:
- 133,320 all-subset closure-equivalence comparisons using the committed test grid;
- 9,360 exhaustive small-vector solver comparisons against brute force using the committed test grid;
- 144 randomized solver/brute-force comparisons across n=1..8;
- 112,320 zero-score padding comparisons, all preserving the old decision prefix;
- strict-addition, baseline-retention and future-high-score extension witnesses passed.

These were executed as local exact-rational reference checks in a separate Python reimplementation of the committed algorithms, not as the committed unittest file, Rust tests or GitHub Actions. The test suite is authored for independent reproduction.

## 8. Projectivity and limitations

The *weight sequence* and \(\delta\) are horizon-independent. The compound score for an existing candidate remains stable only if the underlying evidence and fixed-destination transfer rules preserve that candidate's prior history under extension.

The anchored rejection set itself is **not automatically initial-segment projective**. Adding a high-scoring future candidate can change e-BH rank thresholds and allow old candidates to enter a larger certified set. This is not a defect in the local e-process bound; it is a decision-set coherence question. The project must report separately:
- fixed global weight / local-certificate projectivity;
- neutral zero-score padding behavior;
- arbitrary future high-score extension behavior;
- the exact decision-stability contract required by each application.

The current algorithm operates on the finite active set at each reporting time and checks all its subsets. For the countable global null set, the SupFDR proof uses the monotone limit of the local processes over finite intersections. It does not assume that every finite-horizon rejection set remains unchanged when new evidence arrives.

## 9. Benchmark programme

Compare on paired, valid base-process streams:
1. weighted compound e-BH with total global weight mass one (where an anchor slack is not available);
2. weighted compound e-BH with fixed global mass \(1-\delta\), no closure augmentation;
3. slack-anchored compound e-BH for a predeclared grid \(\delta\in\{0.01,0.05,0.10,0.20\}\);
4. two-phase exact-minimum-credit allocation;
5. two-phase allocation plus slack-anchored closure.

The δ grid must be fixed before confirmatory testing or tuned on separate training seeds. Use independent, shared-control/common-shock, balanced-negative, block-dependent and time-varying conditionally valid nulls. Include sparse, dense weak, heterogeneous, immediate-borderline, late-arrival, far-recipient, switching and no-useful-recipient alternatives.

Report SupFDR, terminal FDR, stopped-FDR, FDX and P(any false) separately; power, time-to-first-discovery, additional rejections beyond baseline, reserve/credit utilization, worst-regime regret, and p50/p95/p99 decision/verifier latency. Freeze at least 50,000 paired streams per confirmatory cell, with more for rare-event targets and simultaneous intervals across the preregistered grid.

## Claim boundary

The exact closure condition and O(n log n) solver are deterministic results. The statistical guarantee follows conditionally from the compound intersection-mass lemma and the time-monotone e-closure argument. The anchored e-closure principle itself is prior work (Ramdas, 2026). No power gain, admissibility, strict dominance, SOTA result, or full horizon-projectivity claim is made until the matched benchmark and extension tests complete.