# SDCB finite-family Gaussian minimax oracle v1

**Status:** exact analytical result for a small synthetic finite family; standalone Rust reference authored but not compiled, CI-qualified, or physically validated. Machine-readable fixture: `docs/research/SDCB_FINITE_FAMILY_MINIMAX_ORACLE_V1.json`.

Run the standalone harness with `bash scripts/research/run_sdcb_finite_family_tests.sh`.

## 1. Why this fixture

Issue #14136 proposes a conservative family p-value

\[
p_{\sup}(y)=\max_{j\in\{1,\dots,J\}}P_{B_j}\{T(Y)\geq T(y)\},
\qquad
T(y)=\max_i\log f_{A_i}(y)-\max_j\log f_{B_j}(y).
\]

For every true null member \(B_k\), \(p_{\sup}\geq p_k\). Therefore, if each component tail probability is valid, rejecting only when \(p_{\sup}\leq\alpha\) controls size at most \(\alpha\) for every member of the declared finite null family.

That construction is a sound conservative baseline. It is not automatically maximin. The benchmark therefore needs an exact small-family oracle against which candidate composite tests can be checked.

## 2. Frozen family

One observation \(X\in\mathbb R\), unit variance:

\[
\mathcal B=\{N(-1,1),N(1,1)\},\qquad
\mathcal A=\{N(-1/2,1),N(1/2,1)\}.
\]

All means and variances are exact dyadic values in the fixture. No null law equals an alternative law. The challenge has finite support in nuisance space; it says nothing about a continuous parameter interval.

The GLRT score is

\[
T(x)=\max_{a\in\{-1/2,1/2\}}\log\varphi(x-a)
-\max_{b\in\{-1,1\}}\log\varphi(x-b)
=\frac38-\frac{|x|}{2}.
\]

Thus high scores are exactly the observations near zero.

For either null member, the component tail probability is

\[
p_j(x)=P_{N(\pm1,1)}(|X|\leq |x|)
=\Phi(|x|-1)-\Phi(-|x|-1).
\]

The two components are equal by reflection, so the worst-null p-value is exactly

\[
p_{\sup}(x)=\Phi(|x|-1)-\Phi(-|x|-1).
\]

At \(\alpha=0.05\), the critical radius solves \(p_{\sup}(r_\alpha)=0.05\).

## 3. Exact least-favourable-mixture / maximin oracle

Take equal-weight mixtures

\[
Q_0=\tfrac12N(-1,1)+\tfrac12N(1,1),\qquad
Q_1=\tfrac12N(-1/2,1)+\tfrac12N(1/2,1).
\]

Their log likelihood ratio is

\[
\log\frac{q_1(x)}{q_0(x)}
=\frac38+\log\cosh(x/2)-\log\cosh(x).
\]

It is strictly decreasing in \(|x|\) for \(|x|>0\), so the Neyman-Pearson most-powerful level-\(\alpha\) test between the mixtures rejects for \(|x|\leq r_\alpha\).

This is also an exact maximin test for the finite family:

1. Any test with uniform size at most \(\alpha\) has average size under \(Q_0\) at most \(\alpha\).
2. By the Neyman-Pearson lemma, its average power under \(Q_1\) cannot exceed the mixture likelihood-ratio test's average power.
3. The mixture test has exactly size \(\alpha\) under each null member, and exactly the same power under each alternative member, by symmetry.
4. Its average alternative power therefore equals its worst-case alternative power and attains the upper bound. No uniformly level-\(\alpha\) test can have larger worst-case power on this fixture.

For this symmetric fixture only, the GLRT ordering, \(p_{\sup}\) test, equal-mixture likelihood-ratio test, and exact maximin test all produce the same rejection region. This coincidence is a test oracle, not a general theorem about composite Gaussian families.

## 4. Frozen operating point and a deliberately weak pairwise comparator

The independent SciPy calculation gives:

| Quantity | Value |
|---|---:|
| Nominal size \(\alpha\) | 0.0500000 |
| Critical radius \(r_{0.05}\) | 0.1033184797 |
| Size under \(N(-1,1)\) | 0.0500000 |
| Size under \(N(1,1)\) | 0.0500000 |
| Power under \(N(-0.5,1)\) | 0.0726527437 |
| Power under \(N(0.5,1)\) | 0.0726527437 |
| Worst-case power | 0.0726527437 |

The fixture is correctly classified **UNDERPOWERED** against the proposed 80% power target. The point of the oracle is validity and exact comparison, not to make a weak experiment look successful.

A deliberately narrow baseline chooses only \(A_+=N(0.5,1)\) versus \(B_+=N(1,1)\), then recalibrates its lower-tail LRT against the *worst* member of the full null family. Its rejection threshold is \(-2.64485362695\), which yields sizes 0.05 and 0.0001337720 at the two null members. Its powers are 0.0159822760 and 0.0008308497 at the two alternative members, so worst-case power is 0.0008308497. The symmetric minimax test is about 87.44 times as powerful in the worst case on this deliberately constructed example.

This is a controlled demonstration of why a nominal pair should not stand in for a composite family. It is not a claim that the GLRT or equal mixtures dominate all pairwise, robust, or industrial procedures.

## 5. Monte Carlo calibration contract

For each null member, a simulation-based component p-value may use

\[
\widehat p_j(y)=
\frac{1+\sum_{b=1}^M\mathbf 1\{T(Y_b)\geq T(y)\}}{M+1},
\qquad Y_b\stackrel{iid}{\sim}P_{B_j}.
\]

The observed test sample and calibration stream must be independent. The inclusive tail handles ties conservatively. The family value is \(\widehat p_{\sup}=\max_j\widehat p_j\).

Keep the following claims separate:

- exchangeable rank validity under independent draws from a declared null law;
- reproducibility under a deterministic PRNG seed;
- Monte Carlo uncertainty;
- numerical approximation error in the score or simulator;
- adequacy of the physical observation model.

A deterministic seed is a replay tool, not a source of physical randomness. Do not reuse sealed evaluation samples to choose the statistic, threshold, family, or nuisance bounds.

## 6. Required implementation controls

The Rust reference has focused tests for:

- no family intersection on the frozen fixture;
- exact intersection detection when a duplicate law is inserted across families;
- symmetry and monotonic ordering of the GLRT score;
- exact closed-form \(p_{\sup}\) equality for the two null members;
- 5% size and the frozen alternative power at both family members;
- equality of the GLRT and mixture rejection ordering;
- the worst-null recalibration of a one-pair LRT;
- monotonicity of \(p_{\sup}\) when a null member is added;
- inclusive Monte Carlo ties and empty-calibration rejection;
- invalid alpha failing closed.

The implementation uses a dependency-free normal-CDF approximation. Its numerical agreement target is 2e-6 for this fixture; it is not a certified special-function library. Exact law equality is checked only for this dyadic test fixture using exact parameter equality. General observation-law equivalence still belongs to #14133 and must return unresolved rather than infer equality from a floating-point tolerance.

## 7. Paradigm shift: synthesize the probe and test together

This oracle sharpens the next architectural move. Do not treat test design as a post-processing step after someone has chosen a sampling schedule. Co-design:

- the input waveform or intervention;
- sensor response and placement;
- sampling times and replicate allocation;
- calibration/reference operations;
- the composite test and its stopping rule;
- the terminal action and its loss.

For a design \(u\), a clean robust power formulation is

\[
\max_{u,\phi}\;\inf_{\eta\in\Theta_A}
E_{P^u_{A,\eta}}[\phi(Y)]
\quad\text{subject to}\quad
\sup_{\xi\in\Theta_B}
E_{P^u_{B,\xi}}[\phi(Y)]\leq\alpha,
\]

with actuator, safety, acquisition, time, energy, and traceability constraints on \(u\). If the selected observation protocol makes the composite laws intersect, its guaranteed passive power cannot exceed \(\alpha\); the design search should then propose a changed intervention rather than buy more passive samples.

Add a second, separate objective for **decision sufficiency**. If every model remaining in the confidence/uncertainty set implies the same terminal action (or actions whose worst-case regret differs by at most a preregistered \(\varepsilon\)), further mechanistic identification may have no decision value. Conversely, when compatible models imply different actions, prioritize experiments that separate those decision-relevant equivalence classes.

The outer loop should be adversarial: for each candidate design, search for nuisance settings that minimize separation or increase worst-case loss. A grid-only minimum is not a continuous-set certificate. The solver must return a witness, a validated lower bound, or \(\texttt{OVERLAP_UNRESOLVED}\).

## 8. Comparators and benchmarks

Recent work makes the comparator set concrete:

- Gül (2026), *From Asymptotic to Finite-Sample Minimax Robust Hypothesis Testing*, derives least-favourable distributions and robust likelihood-ratio functions for total-variation and band uncertainty classes. SDCB should compare its finite-family oracle to least-favourable-mixture / minimax constructions rather than presenting a GLRT as an optimum: https://arxiv.org/abs/2602.19803
- Liu, Maini and Baker (2026), *Optimal experiment design for practical parameter identifiability and model discrimination*, use profile likelihood for practical identifiability and Pontryagin's maximum principle for optimal-control stimulus design. This is a direct comparator for waveform synthesis and the sensor-inlet intervention: https://doi.org/10.1016/j.mbs.2026.109710
- Klimesch et al. (2026), *Designing physics experiments with artificial intelligence*, emphasize expressive hardware search spaces, fast reliable simulators, computable scientific objectives, and navigation of discrete/continuous design spaces. This motivates co-design beyond sampling-interval tuning: https://doi.org/10.1038/s41586-026-10898-6
- *Anytime validity is free: inducing sequential tests* (JRSS B, 2026) shows how to turn a fixed-horizon valid test into an anytime-valid sequential test that matches it at the horizon. SDCB needs this class of guarantee whenever acquisition or stopping adapts to observations: https://doi.org/10.1093/jrsssb/qkag050

Benchmark arms should include the nominal fixed-pair LRT, \(p_{\sup}\)-GLRT, fixed mixture LR, least-favourable/minimax mixture, pairwise multiplicity-corrected tests, robust EIG, D-optimal/Fisher design, profile likelihood, robust optimal control, and the existing #13567 exact policy oracle. Match every arm on the same family, data, actuator constraints, reference chain, compute, and total resource vector.

## 9. Proposed qualification gates

These are proposed targets, not achieved results:

| Track | Target |
|---|---|
| Family-wise size | \(\alpha=0.05\); preregistered 99% upper confidence bound \(\leq0.06\) for every declared null member |
| Worst-case power | \(\geq80\%\) for every material alternative at the funded budget; otherwise \(\texttt{UNDERPOWERED}\) |
| Exact overlap | Zero false robust-resolution results on exact-overlap controls |
| Family expansion | Adding a null member never silently preserves an old guarantee without recalibration |
| Continuous uncertainty | Zero continuous-family claims supported only by finite grids |
| Tiny-family oracle | Exact rejection-region agreement and numeric tolerance \(2\times10^{-6}\) on this fixture |
| Decision value | At most 2% regret versus #13567 exact small-world oracle |
| Practical advantage | Target at least 10% lower terminal decision loss per matched resource cost versus the strongest preregistered baseline, paired uncertainty interval excluding zero |
| Safety | Zero intervention constraint violations; unsafe designs excluded before acquisition |

Do not compress size, power, numerical error, model adequacy, calibration traceability, and decision regret into one score. Publish Pareto frontiers and all failures/abstentions.

## 10. Validation boundary

- The frozen scalar-family calculations were independently recomputed with SciPy's normal CDF and root solver.
- The proof of maximin optimality is analytic for this symmetric two-point family.
- The Rust reference and shell harness are authored, but the authoring container has no `rustc`/Cargo; Rust compilation, tests, and Clippy are not claimed.
- No hosted CI pass, physical sensor test, QUANT analysis, metrological traceability, continuous-family certificate, or industry/SOTA superiority is claimed.
