# SDCB finite-alphabet TV-ball minimax fixture v1

**Status:** exact analytical result for one binary-alphabet problem; Rust known-answer verifier authored but not compiled or CI-qualified.

Machine-readable specification: `docs/research/SDCB_TV_BALL_MINIMAX_V1.json`.

## Why this benchmark matters

Robust hypothesis testing has direct modern comparators for total-variation and band uncertainty sets, including least-favourable distributions and robust likelihood-ratio functions. Gül (2026) explicitly studies unequal robustness parameters. Separate 2026 work studies sample complexity under Huber, subtractive and TV contamination, including instability as uncertainty radii change.

- Gül, *From Asymptotic to Finite-Sample Minimax Robust Hypothesis Testing*: https://arxiv.org/abs/2602.19803
- Vallinayagam, Pensia & Jog, *On the Sample Complexity of Robust Binary Hypothesis Testing*: https://arxiv.org/abs/2605.24741

SDCB must benchmark against these uncertainty geometries, not only finite hand-picked Gaussian scenarios.

## Frozen uncertainty sets

Observation space \(\mathcal Y=\{0,1\}\). Use the convention

\[
\operatorname{TV}(P,Q)=\tfrac12\sum_{y\in\mathcal Y}|P(y)-Q(y)|.
\]

For Bernoulli distributions, this equals \(|p-p_0|\).

Nominal alternative: \(p_A^0=4/5\), radius \(\epsilon_A=3/20\).

\[
\mathcal P_A=\{\operatorname{Bern}(p):p\in[13/20,19/20]\}.
\]

Nominal null: \(p_B^0=1/5\), radius \(\epsilon_B=1/20\).

\[
\mathcal P_B=\{\operatorname{Bern}(q):q\in[3/20,1/4]\}.
\]

The radii are deliberately unequal. For this two-outcome simplex, every linear operating characteristic reaches its extremum at one of the interval endpoints; this is an analytic endpoint result, not grid evidence.

## Exact maximin test

At \(\alpha=1/10\), use the randomized test

\[
\phi(0)=0,\qquad \phi(1)=2/5.
\]

Worst-null size:

\[
\sup_{q\in[3/20,1/4]}q\,\phi(1)=\tfrac14\cdot\tfrac25=\tfrac1{10}.
\]

Worst-alternative power:

\[
\inf_{p\in[13/20,19/20]}p\,\phi(1)
=\tfrac{13}{20}\cdot\tfrac25=\tfrac{13}{50}=0.26.
\]

The least-favourable boundary laws are

\[
P_A^*=(7/20,13/20),\qquad P_B^*=(3/4,1/4).
\]

## Matching dual certificate

Use alternative mixture weight \(\lambda=1\), null weight \(w=13/5\), and slack \(z=(0,0)\).

Pointwise:

\[
P_A^*(0) - wP_B^*(0)=7/20-(13/5)(3/4)<0,
\]

\[
P_A^*(1) - wP_B^*(1)=13/20-(13/5)(1/4)=0.
\]

Thus the positive-part integral is zero, and the dual objective is

\[
\alpha w=\tfrac1{10}\cdot\tfrac{13}{5}=\tfrac{13}{50}.
\]

The primal and dual values agree exactly, proving \(\beta^*=13/50\) for this uncertainty-set challenge.

## p_sup-GLRT comparison

The endpoint-family GLRT score ratios are:

\[
T(y=0):\quad 7/17,\qquad
T(y=1):\quad 19/5.
\]

So \(y=1\) is the highest-score outcome, but its worst-null tail probability is

\[
p_{\sup}(1)=\sup_{q\in[3/20,1/4]}q=1/4>\alpha.
\]

For \(y=0\), the inclusive tail contains both outcomes and the p-value is 1. Therefore the valid p_sup rule rejects neither outcome at alpha 0.1 and has zero power. The minimax randomized rule uses a fractional rejection probability at the least-favourable boundary and achieves 0.26 worst-case power.

This is a fixture-specific comparison, not a general claim that p_sup is always suboptimal.

## Industry / research interpretation

- Treat the TV-ball as an uncertainty-set model, not as a synonym for Huber contamination.
- Keep subtractive contamination and band uncertainty separate unless an exact mapping is proven.
- Compare against the least-favourable robust likelihood-ratio constructions under their stated assumptions.
- Sweep the two uncertainty radii independently and publish power/sample-complexity sensitivity; do not quote only the nominal operating point.
- For larger finite alphabets, certify the complete TV-polytope extreme set or solve the robust inner LP directly. A sampled grid is not a TV-ball certificate.

## Implementation and validation

- `scripts/research/sdcb_tv_ball_minimax.rs` — exact rational known-answer verifier and eight authored tests.
- `scripts/research/run_sdcb_tv_ball_minimax_tests.sh` — standalone Rust test/executable harness.
- An independent Python `Fraction` replay confirms the interval endpoints, worst-null size, worst-alternative power, dual certificate and p_sup tails.
- Rust compilation/tests remain unrun because `rustc` is unavailable in the authoring environment.
- This does not establish a general TV-polytope solver, continuous-density result, physical sensor qualification or SOTA superiority.
