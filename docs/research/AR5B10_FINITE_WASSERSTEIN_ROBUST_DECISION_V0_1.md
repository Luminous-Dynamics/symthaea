# AR-5B10: Finite Wasserstein Robust Terminal Decisions v0.1

**Status:** implementation draft; tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → this branch.  
**Claim ceiling:** solves the finite-support 1-Wasserstein worst-case expectation problem under a frozen finite posterior, symmetric metric, and radius using a one-dimensional piecewise-linear dual and reconstructed primal transport plan. Calculations use f64. No ambiguity-set calibration, learned OT geometry, sequential robust recursion, or SOTA claim is included.

## 1. Why add Wasserstein after total variation?

The TV reference in #9782 treats every unit of probability moved between distinct worlds as equally expensive. A finite Wasserstein ambiguity set instead assigns a declared ground cost to moving mass between worlds.

This gives the modeler an explicit geometry: nearby worlds can exchange mass more cheaply than distant worlds. The geometry is a scientific assumption and must be justified and sensitivity-tested; a sophisticated metric can encode the wrong structure just as easily as a misspecified posterior.

For finite nominal support p over worlds i and a ground metric c(i,j), the ambiguity set is:

```
W_1(q,p) = min_{gamma >= 0} sum_ij gamma_ij c(i,j)
subject to sum_j gamma_ij = p_i
           sum_i gamma_ij = q_j
```

The robust risk of one terminal decision with loss L(j) is:

```
max_q sum_j q_j L(j)
subject to W_1(q,p) <= rho
```

## 2. Finite dual and why this implementation needs no external solver

For finite support, eliminating q and dualizing the single transport-budget constraint gives:

```
min_{lambda >= 0}
  lambda * rho
  + sum_i p_i * max_j (L(j) - lambda * c(i,j))
```

This is a one-dimensional convex piecewise-linear function. Its finite breakpoints occur where two destination lines for one source intersect:

```
lambda = (L(j) - L(k)) / (c(i,j) - c(i,k))
```

The implementation enumerates non-negative breakpoints and zero, evaluates the dual objective, then reconstructs a primal coupling from the maximizing destinations at the selected multiplier. If the optimum is at a positive multiplier, ties permit mass allocation until the declared transport budget is met. If the optimum is at zero, the minimum-cost maximizers give a feasible worst-case distribution without spending unnecessary radius.

The report retains:
- the explicit target distribution q;
- the full nonzero source-to-target transport plan;
- actual transport cost;
- dual radius multiplier;
- dual upper bound;
- primal/dual gap;
- nominal and robust terminal decisions.

The implementation checks row-mass conservation, total target mass, metric constraints, radius feasibility, and primal/dual agreement. The support is capped at 32 worlds to bound the breakpoint enumeration.

## 3. Analytic two-world cross-check

For two worlds at distance d, nominal p=(p0,1-p0), and loss L=(L0,L1), if L1>L0 then the worst-case target mass on world 1 is:

```
q1 = min(1, p1 + rho/d)
```

until the mass is exhausted. The symmetric result applies when L0>L1. With p=(0.9,0.1), d=2, rho=0.2 and L=(0,40), the solution is q=(0.8,0.2), transport cost 0.2, and worst-case risk 8.

The unit tests exercise this closed form, zero-radius equivalence, a radius large enough to reach each decision's global worst-case world, metric violations, support-order mismatch, and report mutation. This is a design-level analytical cross-check, not executed validation; an independent external LP comparison is still required before promotion.

## 4. Important limitations

- The ground cost must be a metric in v0.1: finite, symmetric, zero diagonal, strictly positive off-diagonal, and triangle inequality. This avoids silently calling an arbitrary directed cost matrix a Wasserstein metric.
- World IDs and ordering must exactly match the canonical world support.
- The ambiguity radius has physical meaning only relative to the units and calibration of the ground metric.
- The radius and metric are caller-declared. No empirical calibration or generalization guarantee is provided.
- This is terminal minimax decision-making, not sequential robust control.
- A single global distribution over complete trajectories, a non-rectangular ambiguity set, and rectangular stagewise ambiguity lead to different dynamic-programming semantics. Any sequential extension must choose and document one.
- The solver uses f64 and numerical tolerances; results are finite LP optima subject to floating-point error, not symbolic exact rationals.

## 5. What the next scientific comparison should be

Compare on identical finite-support problems:
1. Nominal Bayes.
2. Fixed TV ball (#9782).
3. Finite Wasserstein ball (this PR).
4. DRO-BAS, when its exponential-family and duality conditions fit.
5. DRO-RoBAS / MMD ambiguity, with posterior and radius calibration.
6. Loss-aware OT-DRO, with ambiguity geometry learned only on training/validation campaigns and frozen for held-out evaluation.

Use two axes of shift:
- **Probability shift:** move mass among already declared worlds.
- **Support/model shift:** add omitted worlds or change the outcome mechanism. A metric on the original support cannot protect against omitted support unless the ambiguity contract explicitly includes a reserve/outside-support state or a model-expansion operation.

Relevant references:
- DRO-BAS, ICML 2025: https://proceedings.mlr.press/v267/dellaporta25a.html
- DRO-RoBAS, Entropy 2026: https://doi.org/10.3390/e28040430
- Loss-Aware OT-DRO, ICML 2026: https://proceedings.mlr.press/v306/ohnemus26a.html
- Gao & Kleywegt, Wasserstein distributionally robust stochastic optimization: https://doi.org/10.1287/moor.2022.1275

## 6. Proposed acceptance targets (not results)

- 100% transport plan row sums reconstruct p and column sums reconstruct q within a preregistered tolerance.
- Every q is in the simplex and every transport plan cost is <= rho.
- Primal and dual objectives agree within a scale-aware tolerance.
- 100% agreement with an independent LP solver on randomized finite cases within numerical tolerance.
- Zero-radius results reproduce nominal Bayes decisions.
- Radius and metric sensitivity curves are reported.
- Held-out decision regret and false-safe rates improve versus nominal Bayes/TV at equal all-in cost before any superiority claim.
- Omitted-support shifts are tested separately; in-support Wasserstein guarantees must never be presented as out-of-support protection.

## 7. Exact-head validation

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_wasserstein_decision -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_robust_decision -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_sequential_policy -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No Rust compile, test, clippy, CI, benchmark, external LP comparison, or SOTA pass is claimed.
