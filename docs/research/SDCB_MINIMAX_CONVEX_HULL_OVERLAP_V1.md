# SDCB convex-hull overlap: exact finite-outcome oracle v1

**Status:** analytic, exact-rational known-answer fixture. It is a research benchmark artifact, not a Gaussian or physical-sensor result.

Machine-readable fixture: `docs/research/SDCB_MINIMAX_CONVEX_HULL_OVERLAP_V1.json`.

## Why this fixture matters

The pairwise overlap authority in #14133 correctly detects whether an individual alternative observation law equals an individual null law. For general law families, however, pairwise disjointness does not imply that a uniformly size-controlled test can achieve worst-case power above alpha.

A convex combination of alternative laws can equal a convex combination of null laws. That gives an adversarial mixture witness and an exact maximin power ceiling.

This is a general finite-outcome result. It must not be misrepresented as a claim that distinct finite Gaussian mixtures can coincide: finite Gaussian-mixture identifiability is a separate mathematical constraint.

## Frozen challenge

Observation space:

\[
\mathcal Y=\{y_1,y_2\}.
\]

Null family:

\[
B_1=(1/3,2/3).
\]

Alternative family:

\[
A_1=(1,0),\qquad A_2=(0,1).
\]

Significance level:

\[
\alpha=0.1.
\]

No individual alternative equals the null. Yet:

\[
B_1=\tfrac13 A_1+\tfrac23 A_2.
\]

## Exact upper bound

For any randomized test \(\phi:\mathcal Y\to[0,1]\) with null size at most alpha,

\[
\begin{aligned}
\min\{E_{A_1}[\phi],E_{A_2}[\phi]\}
&\leq \tfrac13E_{A_1}[\phi]+\tfrac23E_{A_2}[\phi]\\
&=E_{B_1}[\phi]\\
&\leq\alpha.
\end{aligned}
\]

Thus every uniformly valid test has worst-case power at most 0.1.

## Matching primal test

Choose:

\[
\phi(y_1)=\phi(y_2)=0.1.
\]

Then:

| Quantity | Exact value |
|---|---:|
| Size under B1 | 1/10 |
| Power under A1 | 1/10 |
| Power under A2 | 1/10 |
| Worst-case power | **1/10** |

The upper bound is attained, so the exact maximin value is:

\[
\boxed{\beta^*=\alpha=0.1}.
\]

## Matching dual certificate

In the primal/dual formulation in #14185, choose alternative mixture weights

\[
\lambda=(1/3,2/3)
\]

and unnormalized null weight

\[
w=(1).
\]

The two mixture laws are identical, so the positive-part integral vanishes. The dual objective is:

\[
\alpha\sum_jw_j+
\int\left(\sum_i\lambda_ip_i-\sum_jw_jq_j\right)_+dy
=0.1+0=0.1.
\]

The primal feasible test attains 0.1 and the dual feasible weights upper-bound the value by 0.1. The primal-dual gap is exactly zero.

## Candidate GLRT / p_sup diagnostic

For the candidate score

\[
T(y)=\max_i\log p_{A_i}(y)-\log p_{B_1}(y),
\]

the score at \(y_1\) exceeds that at \(y_2\). The null-tail p-values are:

\[
p_{\sup}(y_1)=1/3,\qquad p_{\sup}(y_2)=1.
\]

At alpha 0.1, the p_sup-GLRT rejects neither outcome and has zero power. It remains a valid conservative test; this fixture demonstrates that it is not a maximin oracle and that validity alone does not certify decision usefulness.

## Architectural consequence

The SDCB test path should distinguish:

1. **Pairwise intersection:** a null member and alternative member are the same law.
2. **Convex-hull intersection:** mixture laws coincide despite pairwise-disjoint family members.
3. **Separated mixtures:** a positive total-variation lower bound may give a nontrivial upper bound on robust power.
4. **Unresolved mixture geometry:** no separation or intersection claim is justified.

For any alternative mixture \(Q_A\) and null mixture \(Q_B\),

\[
\beta^*\leq\alpha+\operatorname{TV}(Q_A,Q_B).
\]

This supplies a useful power upper-bound diagnostic. The full primal/dual oracle, not this single TV inequality, remains the authority for the maximin value.

## Scope

- Exact rational finite-outcome fixture only.
- No continuous nuisance-family guarantee.
- No Gaussian-mixture equality claim.
- No physical sensor, metrological, CI, or SOTA qualification.


## Standalone Rust known-answer verifier

The branch also contains:

- `scripts/research/sdcb_convex_hull_overlap.rs` — fixture-specific verifier using exact `i128` rational arithmetic. It checks probability-vector validity, pairwise disjointness, exact convex-mixture equality, the constant randomized primal test, the matching dual certificate, and the p_sup-GLRT zero-power result.
- `scripts/research/run_sdcb_convex_hull_overlap_tests.sh` — dependency-free `rustc --test` and executable harness.

Seven Rust unit tests are authored. The source and harness were fetched back from GitHub and their expected checks inspected. The authoring environment does not have `rustc`, so compilation and execution have **not** occurred. The earlier independent `Fraction` replay and JSON/fixture checks are not Rust qualification.


## Latest head workflow state

At PR head `fdc10385d98e9f776c6d309933ba855b725b6928`, GitHub recorded CI and PR Governance as completed with conclusion **skipped**. The combined status list was empty. Neither is a passing qualification result. The seven Rust tests remain authored but uncompiled/unexecuted because `rustc` is unavailable in the authoring environment.
