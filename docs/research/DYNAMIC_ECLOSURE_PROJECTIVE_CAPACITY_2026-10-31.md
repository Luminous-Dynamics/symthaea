# Projective dynamic e-closure with capacity-aware selection

**Status:** exact finite-horizon reference implementation. The pathwise verifier checks algebraic coherence, persistence, closure membership and utility optimality. It does not prove stochastic validity of arbitrary input coordinate processes.

## Why this is a separate research step

De Heide (2026), *Dynamic e-closure for online hypotheses with any-time-valid evidence: closure principles and projective mergers*, treats two interleaved clocks:
- hypotheses can arrive over time;
- evidence for already active hypotheses can continue to evolve and be inspected at global stopping times.

The key theorem is stronger than ordinary online e-closure. Fixed-intersection e-process validity alone is insufficient because a data-dependent stopping rule can switch between different intersection indices. Future-extension coherence prevents this by ensuring that adding inactive future hypotheses cannot reduce the current certificate.

The paper proves:
- future-extension coherence plus valid intersection e-processes gives simultaneous stopped-FDR control;
- adding pathwise time monotonicity gives simultaneous SupFDR control and setwise persistence;
- any measurable selection from the certified family inherits the relevant guarantee;
- nested reporting remains an additional operational choice, not a consequence of setwise persistence.

The paper's distinction between stopped FDR and SupFDR is important. It provides a concrete counterexample where stopped FDR is controlled but SupFDR is 3/4 at nominal 1/2.

## Projective global-weight construction

Let \(M_{i,t}\) be a nonnegative coordinate e-process for null \(H_i\), valid in one common global filtration. Before activation, require \(M_{i,t}=1\). Fix global, nonnegative weights \(w_i\) satisfying

\[
\sum_{i\geq1}w_i\leq1.
\]

For every finite nonempty intersection \(S\), define

\[
E_t^S=1+\sum_{i\in S}w_i(M_{i,t}-1).
\]

Under \(H_S=\cap_{i\in S}H_i\), each coordinate process in the sum is valid. Since the weights are fixed and globally summable,

\[
E_t^S
=1-\sum_{i\in S}w_i+\sum_{i\in S}w_iM_{i,t}
\]

is a nonnegative e-process under arbitrary cross-coordinate dependence. Inactive coordinates contribute zero because their process value is one. Therefore

\[
E_t^S=E_t^{S\cap I_t}
\]

whenever the active intersection is nonempty: exact neutral-padding projectivity, stronger than future-extension coherence.

The weights must not be renormalized over the currently active hypotheses. Renormalization changes the coefficients of existing coordinates and can destroy future-extension coherence. For an open-ended countable hypothesis universe, the cited paper also proves a rigidity result: within its pointwise arbitrary-dependence admissible merger class, a nontrivial symmetric merger is impossible. Global asymmetric weights or additional structure are necessary.

## Exact persistent adjuster

The raw projective intersection processes can decrease as evidence evolves. To obtain time-monotone e-processes, the implementation uses an exact step adjuster

\[
A(x)=\sum_{k=1}^{K}\mathbf 1\{x\geq 2^k\},
\qquad x\geq1.
\]

Its integral budget is

\[
\int_1^\infty \frac{A(x)}{x^2}\,dx
=\sum_{k=1}^{K}2^{-k}
=1-2^{-K}\leq1.
\]

For each intersection,

\[
\widetilde E_t^S
=A\!\left(1\vee\max_{u\leq t}E_u^S\right).
\]

The adjuster theorem makes this a nonnegative time-monotone e-process. Applying the same path transform to all projective intersections preserves neutral-padding equality, hence future-extension coherence. The dynamic closure is therefore setwise persistent and controls simultaneous SupFDR under the paper's theorem assumptions.

A general exact step adjuster is also supported. For increasing thresholds \(c_j>1\) and nonnegative increments \(a_j\), require

\[
\sum_j\frac{a_j}{c_j}\leq1,
\qquad
A(x)=\sum_j a_j\mathbf 1\{x\geq c_j\}.
\]

The adjuster is predeclared; tuning it after looking at the same evidence is not covered by this implementation or theorem contract.

## Capacity-aware dynamic decisions

At global update \(t\), let \(I_t\) be the active hypothesis set. Select

\[
R_t\in\arg\max_R\sum_{i\in R}v_i
\]

subject to

\[
R_{t-1}\subseteq R\subseteq I_t,\qquad
\sum_{i\in R}c_i\leq B_t,\qquad
R\in\mathfrak R_{\alpha,t}(\widetilde{\mathbf E}),
\]

where the current dynamic e-closure menu is

\[
\mathfrak R_{\alpha,t}(\widetilde{\mathbf E})
=\left\{R\subseteq I_t:
\operatorname{FDP}_S(R)\leq\alpha\widetilde E_t^S
\quad\forall\,\varnothing\neq S\subseteq I_t
\right\}.
\]

The initial implementation requires fixed/predeclared values and costs, nonnegative costs, irrevocable nested reporting, and nondecreasing cumulative capacity. Because the certified menus are setwise persistent and capacity is cumulative, an earlier choice remains available as a feasible candidate.

The statistical guarantee is inherited from the dynamic SupFDR theorem if:
1. every coordinate process is a valid e-process for its null in the common global filtration;
2. activation is represented by the deterministic nested active sets and dormant-one convention;
3. weights are fixed globally and sum to at most one;
4. the adjuster integral budget is valid;
5. the selected rejection sets remain inside the dynamic certified menus.

The finite verifier does not prove conditions 1–2 from one observed path. They must be supplied by the upstream evidence-generation and filtration contract.

## Exact fixture

Two hypotheses arrive on separate updates. Coordinate paths:

| Global time | Active set | \(M_{1,t}\) | \(M_{2,t}\) |
|---:|---|---:|---:|
| 0 | {1} | 1 | 1 (dormant) |
| 1 | {1,2} | 8 | 1 (dormant) |
| 2 | {1,2} | 8 | 16 |

Weights are \(w_1=1/2\), \(w_2=1/4\), and \(\alpha=1/2\). The all-success sample path can arise from valid one-step null martingales: coordinate 1 pays 8 with probability 1/8 or 0 otherwise at activation, and coordinate 2 independently pays 16 with probability 1/16 or 0 otherwise.

Using thresholds (2,4,8,16) and unit increments (integral budget 15/16), the adjusted intersection values evolve as:

| Intersection | Time 0 | Time 1 | Time 2 |
|---|---:|---:|---:|
| {1} | 0 | 2 | 2 |
| {2} | 0 | 0 | 2 |
| {1,2} | 0 | 2 | 3 |

With values (10,8), costs (1,1), and cumulative capacity (0,1,2), exact selection gives:
- time 0: no rejection (zero capacity);
- time 1: reject {1};
- time 2: reject {1,2}.

Every active intersection is checked at each update. The decisions are nested, certified, resource-feasible, and utility-optimal for the declared objective. This is a deterministic sample-path illustration under an explicitly valid martingale construction, not an empirical SupFDR estimate.

## Implementation

- validation/dynamic_eclosure_capacity_oracle.py
  - exact projective intersection path construction;
  - fixed global weight and dormant-one checks;
  - exact finite step-adjuster budget and running maximum;
  - pathwise future-extension coherence and time-monotonicity verifier;
  - capacity-aware nested action selection;
  - independent replay and exhaustive utility-optimality check.
- validation/test_dynamic_eclosure_capacity_oracle.py
  - exact two-hypothesis path;
  - coherence and time-monotonicity negative controls;
  - dormant-one and global weight-budget negative controls;
  - adjuster integral-budget negative control;
  - valid-but-suboptimal decision negative control;
  - future-evidence prefix-invariance test.

Reproduction:

    PYTHONPATH=validation python -m unittest -v test_dynamic_eclosure_capacity_oracle.py

The test file has not been executed in this environment. Do not report CI as passing.

## Complexity and limits

For \(N\) hypotheses, the prototype materializes \(2^N-1\) intersection paths. At each update, it enumerates up to \(2^N\) action sets and checks up to \(2^N-1\) active intersections. The reference implementation is exponential and intended only for small \(N\) differential tests.

It does not solve per-period queue stability, stochastic capacity outages, adaptive weights, or filtration lifting for coordinate processes valid only in local filtrations. It assumes the coordinate paths are already globally valid e-processes.

## Paradigm-level research opportunities

1. **Fairness versus projectivity:** global weights are an irrevocable evidence budget. Explore principled deterministic weight hierarchies, department/group budgets and predeclared reallocation between epochs, while preserving coherence.
2. **Beyond pointwise mergers:** use direct intersection betting strategies, richer dependence models or history-dependent mergers to escape the global weight budget, with a proof under the actual filtration.
3. **Better adjusters:** optimize the step-adjuster frontier under the integral constraint using an independent pilot objective; freeze the adjuster before confirmatory evidence.
4. **Resource-aware persistent closure:** optimize value under cumulative capacity while measuring value loss from the globally summable evidence weights.
5. **Asynchronous production evidence:** build a common global filtration and replay layer for shared controls, delayed outcomes, arm activation, adaptive observation scheduling and optional stopping.

## Benchmark programme

Compare:
- dynamic projective e-closure with fixed global weights;
- dynamic e-closure with equal / priority / group-budget weights;
- persistent closed e-BH for a fixed family;
- online e-closure with one final e-value per arriving hypothesis;
- stopped-FDR closure without time-monotone adjustment;
- adjusted running-max closure;
- LORD / SAFFRON / SCORE and online donation baselines where their guarantee class matches.

Stress:
- new hypotheses arriving while old evidence continues to update;
- independent, block, common-shock and shared-control dependence;
- delayed and missing outcomes;
- random global stopping times;
- inactive neutral padding and horizon extension;
- time-monotonicity and coherence violations;
- capacity saturation, outages and bursts;
- fixed versus misspecified utilities and global weight allocation.

Report simultaneous stopped FDR and SupFDR separately, plus power, discoveries, expected utility, resource utilization, queue delay, p50/p95 latency, memory, number of intersection constraints, and certificate-replay cost. Use 5,000 exploratory and 50,000 frozen confirmatory streams, common random numbers and independent held-out seeds. Empirical SupFDR is diagnostic; validity comes from the e-process and dynamic-closure contracts.

## References

- de Heide (2026), [Dynamic e-closure for online hypotheses with any-time-valid evidence](https://arxiv.org/abs/2608.09927).
- Xu, Fischer & Ramdas (2026), [Online analogs of e-closure and compound e-values](https://proceedings.mlr.press/v337/xu26a.html).
- Bai & Ramdas (2026), [Recertifying decisions via e-closure](https://arxiv.org/abs/2610.35821).
- Kuang, Gang & Xia (2026), [SCORE](https://proceedings.mlr.press/v306/kuang26c.html).
