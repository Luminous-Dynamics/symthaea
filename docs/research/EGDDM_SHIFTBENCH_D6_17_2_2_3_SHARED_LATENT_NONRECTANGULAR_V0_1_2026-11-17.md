# EGDDM ShiftBench D6.17.2.2.3-alpha — Shared-latent cross-hypothesis joint-kernel model

**Date:** 2026-11-17  
**Status:** shared-latent kernel validator prototype; synthetic two-hypothesis/two-latent known-answer fixture; Rust execution pending  
**Related:** issue #12682 (shared latent/non-rectangular ambiguity), parent #12630, stochastic policy scorer PR #12686, joint-kernel validator PR #12655.

## 1. Why this matters

D6.17.2.2.1 makes a deliberate product-set assumption: each hypothesis's joint-kernel ambiguity set can vary independently of the others. That is useful and exact for the declared set, but it does not represent every physical system.

A common cause can alter the observation channel under several competing hypotheses at once:
- shared power supply or clock;
- common firmware, reference instrument or calibration procedure;
- site/environmental condition;
- common network, operator or supply chain;
- same device fleet or test fixture.

If one latent cause determines the kernel mixture for all hypotheses, then the likelihood uncertainty sets are coupled. Taking each hypothesis's marginal interval independently replaces a constrained joint model by a rectangular superset. The result remains conservative if all marginal intervals are valid, but it may be materially over-conservative: it can combine extremes that no single shared latent state can produce, inflating robust risk and suppressing useful actions.

Rectangularity is a substantive assumption in robust dynamic programming, not a harmless implementation detail. Iyengar (2005) establishes robust dynamic-programming results under rectangular sets of conditional probability measures and makes clear why the ambiguity structure matters: https://doi.org/10.1287/moor.1040.0129. Robust sequential experimental design under misspecification is also an active 2026 research area (Wen et al., ICML 2026): https://proceedings.mlr.press/v306/wen26i.html.

## 2. Shared-latent model

The alpha prototype has one evaluator-owned latent-state probability vector shared across all hypotheses and registered acquisition channels:

\[
w_z\in[\underline w_z,\overline w_z],\qquad
w_z^0\in[\underline w_z,\overline w_z],\qquad
\sum_z w_z=1,\quad \sum_z w_z^0=1.
\]

For every hypothesis h and latent state z, the evaluator supplies an exact conditional joint distribution over complete option-outcome vectors:

\[
q_{h,z}(j)\ge0,\qquad \sum_{j\in J_h}q_{h,z}(j)=1.
\]

The same mixture weight vector w is applied to every hypothesis:

\[
q_h(j;w)=\sum_z w_z q_{h,z}(j).
\]

For any event A in the joint outcome space:

\[
\underline q_h(A)=\min_{w\in\mathcal W}
\sum_z w_z q_{h,z}(A),\quad
\overline q_h(A)=\max_{w\in\mathcal W}
\sum_z w_z q_{h,z}(A),
\]

with nominal probability obtained from w0. These are linear optimizations over one bounded simplex. Starting at all lower weights and allocating remaining mass to the smallest or largest event coefficients gives the exact bounds for this alpha model.

The v0.1 conditional kernels are point distributions. This is intentional: it isolates cross-hypothesis dependence through a shared latent mixture. Conditional-kernel interval ambiguity, temporal latent dynamics, latent-state discovery and drift are later extensions. The nominal marginal projection is rounded half-up to the nearest basis point for reporting; posterior-risk probes retain the unrounded latent-weight × conditional-likelihood integer numerators.

## 3. Synthetic known-answer challenge

There are two hypotheses (safe-world and common-cause-hazard), two binary gate observations, and two latent states. The evaluator's shared weights satisfy:

\[
w_0\in[0.4,0.6],\qquad w_1=1-w_0,\qquad w_0^0=w_1^0=0.5.
\]

The conditional pass/pass probabilities are:

| Hypothesis | Latent z0 | Latent z1 |
|---|---:|---:|
| safe-world | 20% | 80% |
| common-cause-hazard | 10% | 50% |

The other three outcome-vector probabilities are supplied in the fixture and complete each conditional distribution to 100%. The resulting shared-mixture pass/pass intervals are:
- safe-world: [44%, 56%], nominal 50%;
- common-cause-hazard: [26%, 34%], nominal 30%.

The per-gate pass marginals are:
- safe-world: [50%, 60%], nominal 55%;
- common-cause-hazard: [36%, 44%], nominal 40%.

With a fixed prior of 50% safe-world and 50% hazard, consider a release action justified only under safe-world, after observing pass/pass.

### Shared-latent robust posterior risk

For a shared latent mixture:

\[
R(w_0)=
\frac{0.5(0.1w_0+0.5(1-w_0))}
{0.5(0.2w_0+0.8(1-w_0))+0.5(0.1w_0+0.5(1-w_0))}.
\]

The maximum over w0 in [0.4, 0.6] occurs at w0=0.4:

\[
R_{\mathrm{shared}}^{\max}=
\frac{0.34}{0.56+0.34}=37.78\%.
\]

### Independent rectangular hull comparator

If the same hypothesis-specific marginal intervals are treated as independently selectable, the risk upper bound can combine hazard likelihood 34% with safe likelihood 44%:

\[
R_{\mathrm{rect}}^{\max}=
\frac{0.34}{0.34+0.44}=43.59\%.
\]

The rectangular hull raises this synthetic worst-case posterior risk by 5.81 percentage points. This is not a real risk reduction produced by software: it is the difference between a coupled mathematical set and a larger rectangular outer approximation. The shared model is only preferable when independent evaluator evidence supports the common-latent assumption. Otherwise the rectangular/Fréchet fallback remains the safer uncertainty envelope.


### Machine-replayed risk probe

The evaluator fixture contains a risk query for action release, safe only under safe-world, with observed vector gate-a=pass and gate-b=pass. The Rust evaluator calculates:

| Measure | Result |
|---|---:|
| Lower support of pass/pass under prior and shared latent set | 35.00% |
| Nominal posterior hazard risk | 37.50% |
| Worst shared-latent posterior hazard risk | 37.78% |
| Worst independent rectangular-hull risk | 43.59% |
| Rectangular excess | 5.81 percentage points |

The query is part of the evaluator-only oracle, not the candidate policy. The report records per-hypothesis likelihood intervals and the exact observed vector. Risk calculations use integer numerators before final basis-point rounding; nominal channel projections round half-up to the nearest basis point.

## 4. What the validator implements

The Rust binary shared-latent-kernel-validator.rs:
- parses candidate channel marginals and option/factor registry separately from evaluator latent-state and conditional-kernel records;
- requires exactly one evaluator-owned shared latent-weight simplex;
- requires the exact hypothesis × latent-state product of conditional kernels;
- requires every conditional kernel to cover every registered option exactly;
- rejects duplicate joint outcome vectors, unknown outcomes, non-normalized rows, stale/revoked latent states and stale/revoked conditional kernels;
- derives each per-hypothesis channel marginal using the same shared latent weights;
- rejects candidate channel intervals that underclaim the coupled mixture interval or disagree with its nominal point;
- computes pairwise joint-event intervals under the same shared weights and checks them against the resulting marginal Fréchet envelopes;
- emits latent-weight, kernel-provenance, marginal and pairwise reports;
- accepts evaluator-only risk probes containing a prior credal set, a safe-hypothesis set and a complete observed outcome vector, then computes exact shared-latent worst-case posterior risk alongside the rectangular-hull comparator.

The candidate cannot supply latent states or conditional kernels. The oracle has one shared weight vector, not a separate weight vector per hypothesis. The fixture has four conditional kernels: 2 hypotheses × 2 latent states.

## 5. Benchmarks and acceptance

**Model correctness**
- Exact channel marginal consistency for all hypotheses and latent states.
- Exact conditional probability normalization and complete option coverage.
- Exact pairwise shared-mixture event bounds.
- Zero accepted stale/revoked latent or kernel records.
- Zero candidate oracle injection or outcome omission.
- Cross-hypothesis coupling receipt identifies the exact latent model class.

**Coupling challenge**
- Independent-kernel case: shared and product results should coincide when the latent mixture does not couple hypotheses.
- Positive shared-latent case: shared risk/cost bounds should be no wider than the product rectangular hull when the shared set is a subset of that hull.
- Anti-correlated latent case: quantify how policy rankings change when independent extremes are impossible.
- Latent omission/drift: D6.18 unknown-world reserve must catch unsupported model closure.
- If evidence cannot support a shared latent explanation, the evaluator must fall back to the conservative product/Fréchet representation.

**Proposed targets (not achieved)**
- Exact equality against an independent reference implementation for all enumerated small fixtures.
- Zero marginal underclaim and conditional-normalization violations accepted.
- Correct shared-vs-rectangular risk on all known-answer controls.
- 100% stale/revoked and oracle-injection negative controls rejected.
- Calibration coverage and interval width assessed separately on held-out common-cause families before using this model in a policy scorer.

## 6. Next steps

1. Extend the shared-latent validator to evaluator-owned prior/action query probes and calculate coupled robust posterior-risk bounds directly.
2. Integrate the shared-latent risk evaluator into D6.17.2.2.1 without changing its product-set behavior.
3. Add shared-latent cost optimization and policy search for small trees.
4. Add multiple latent components and explicit coupling across time windows.
5. Validate latent weights and conditional kernels against repeated measurements with leave-one-common-cause-family-out splits.
6. Combine with D6.18 open-world reserve and D6.17.2.3 authenticated run receipts.

## 7. Claim ceiling

This alpha validates a finite evaluator-owned shared latent-weight simplex and point conditional joint kernels, deriving exact mixture marginals and pairwise event intervals. It does not estimate latent weights from data, prove that the latent-state set is complete, calculate general dynamic non-rectangular robust-control values, authenticate calibration, model drift or authorize real-world actions.

Rust compilation, tests, rustfmt, Clippy and CI have not been executed here. Keep this branch draft until exact-head validation and independent held-out repeated-measurement trials.
