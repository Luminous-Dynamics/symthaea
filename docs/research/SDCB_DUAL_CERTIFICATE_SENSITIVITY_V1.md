# SDCB dual-certificate sensitivity at a least-favourable active-set switch

**Status:** exact analytical/rational known-answer fixture; Rust verifier authored but not compiled or CI-qualified.

Machine-readable artifact: `docs/research/SDCB_DUAL_CERTIFICATE_SENSITIVITY_V1.json`.

## Why this matters

The least-favourable alternative and null weights are more than proof metadata. They identify which physical/model-family members currently bottleneck the experiment's robust test power. Their changes with the design variable can guide stimulus, sensor and sampling co-design.

This fixture makes that idea testable: robust power has an exact piecewise-rational objective, and both least-favourable families switch at the same design point.

## Frozen family

Let \(u\in[0,1]\), \(\alpha=1/10\), with binary observations \(Y\in\{0,1\}\).

Alternative family:
\[
A_1(u)=\operatorname{Bern}(3/4+u/10),\qquad
A_2(u)=\operatorname{Bern}(17/20-u/10).
\]

Null family:
\[
B_1(u)=\operatorname{Bern}(1/5+u/10),\qquad
B_2(u)=\operatorname{Bern}(3/10-u/10).
\]

Define
\[
p_{\min}(u)=\min(p_{A_1}(u),p_{A_2}(u)),\quad
q_{\max}(u)=\max(p_{B_1}(u),p_{B_2}(u)).
\]

The exact robust randomized test is
\[
\phi(0)=0,\qquad \phi(1)=\alpha/q_{\max}(u),
\]
and its maximin power is
\[
V(u)=\alpha\,p_{\min}(u)/q_{\max}(u).
\]

## Active least-favourable member switch

| Design region | Least-favourable alternative | Least-favourable null |
|---|---|---|
| \(u<1/2\) | A1 | B2 |
| \(u=1/2\) | A1 and A2 | B1 and B2 |
| \(u>1/2\) | A2 | B1 |

The active sets switch at the same point. At \(u=1/2\), A1=A2=Bernoulli(4/5) and B1=B2=Bernoulli(1/4), so the optimal dual certificate is non-unique.

## Exact operating points and derivatives

| \(u\) | \(p_{\min}\) | \(q_{\max}\) | \(\phi(1)\) | \(V(u)\) | \(dV/du\) |
|---:|---:|---:|---:|---:|---:|
| 0 | 3/4 | 3/10 | 1/3 | 1/4 | 7/60 |
| 1/4 | 31/40 | 11/40 | 4/11 | 31/110 | 84/605 |
| 1/2 | 4/5 | 1/4 | 2/5 | **8/25** | nondifferentiable |
| 3/4 | 31/40 | 11/40 | 4/11 | 31/110 | -84/605 |
| 1 | 3/4 | 3/10 | 1/3 | 1/4 | -7/60 |

For a smooth active branch:
\[
\frac{dV}{du}=\alpha\frac{p' q-p q'}{q^2}.
\]

At the active-set switch, the exact one-sided derivatives are:
\[
V'_-(1/2)=21/125,\qquad V'_+(1/2)=-21/125.
\]

The Clarke generalized gradient is therefore:
\[
\partial_C V(1/2)=[-21/125,21/125].
\]

The strict cusp shows that \(u=1/2\) is the unique global maximizer on this frozen interval. An optimizer that insists on one smooth gradient at this point is wrong.

With \(h=1/10000\), the exact one-sided finite-difference quotients are \(4200/25001\) and \(-4200/25001\), converging to the one-sided derivatives from the appropriate sides.

## Dual certificates and non-uniqueness

For \(u<1/2\), one extreme optimal dual certificate is:
\[
\lambda=(1,0),\quad w=(0,p_{\min}/q_{\max}).
\]

For \(u>1/2\):
\[
\lambda=(0,1),\quad w=(p_{\min}/q_{\max},0).
\]

At the switch, every pair
\[
\lambda=(t,1-t),\quad
w=(s,16/5-s),
\]
with \(t\in[0,1]\), \(s\in[0,16/5]\), is an optimal dual certificate for this fixture. For example, \(\lambda=(1/3,2/3)\) and \(w=(8/5,8/5)\) give the same exact objective.

This is why #14226 requires set-valued sensitivities at active-set changes and non-unique dual solutions. A single canonical tie-break may be useful for replay, but must not be mistaken for a unique scientific bottleneck.

## Implementation

- `scripts/research/sdcb_dual_certificate_sensitivity.rs` — exact-rational fixture, active-set selection, primal/dual verification, smooth derivatives, switch derivatives and one-sided finite differences.
- `scripts/research/run_sdcb_dual_certificate_sensitivity_tests.sh` — standalone Rust test/executable harness.
- Eight Rust tests cover known answers, least-favourable member switches, primal/dual equality, dual non-uniqueness, smooth derivatives, kink derivatives, finite differences and global maximum on the frozen interval.

## Validation boundary

An independent Python `Fraction` replay confirmed all five operating points, the exact one-sided derivatives, finite-difference quotients and active-set switch. Rust tests are authored but **not compiled or executed** because `rustc` is unavailable in the authoring environment.

This is a one-dimensional exact finite-law sensitivity fixture. It is not a general robust-LP sensitivity solver, continuous waveform optimizer, physical experiment result, or SOTA claim.
