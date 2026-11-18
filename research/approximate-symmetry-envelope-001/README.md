# Approximate symmetry error envelope reference v0.1

First executable slice for [SYMM-ERROR-ENVELOPE-001 #127xx](https://github.com/Luminous-Dynamics/symthaea/issues/127xx).

## Paradigm shift

Replace a binary "symmetric / not symmetric" flag with a domain- and horizon-bounded error contract.

An approximate symmetry is not an equivalence relation. It may support an epsilon-bounded lifted comparison, but it does not permit exact quotient pruning. This implementation makes that distinction structural: `hard_prune_authorized` is always false.

## Discrete error propagation

For a discrete map \\(\Phi\\), an action \\(T\\), and a certified domain \\(D\\), assume:

\\[
\sup_{x\in D}\|\Phi(Tx)-T\Phi(x)\|\leq\delta,
\qquad
\|\Phi(x)-\Phi(y)\|\leq L\|x-y\|.
\\]

The paired-trajectory error obeys:

\\[
e_{k+1}\leq L e_k+\delta.
\\]

The Rust reference propagates this recurrence using checked fixed-point integers and ceiling division at every step. This deliberately over-approximates rather than rounding error downward.

It then computes objective and constraint error envelopes from declared Lipschitz factors and direct task-equivariance defects.

## Fail-closed decision semantics

An `epsilon_lift_candidate` requires all of the following:
- an independently established domain-invariance flag;
- the accumulated state error fits inside the certified boundary margin;
- state, objective and constraint errors fit their declared budgets;
- constraint slack exceeds the constraint-error bound;
- the objective gap exceeds twice the objective-error bound.

Otherwise the report abstains or remains unresolved. Even a fully exact zero-defect result is only an equivariance candidate; another complete task verifier must qualify boundary/initial conditions, admissible sets, objectives, constraints and outputs before any exact quotient operation. This crate never grants hard-prune authority.

The input contains the numerical bounds, but **does not prove they are true**. A later independent qualification layer must derive defect, Lipschitz and boundary bounds from interval analysis, analytic bounds, validated numerics or conservative physical measurements.

## Reproduction

    cargo fmt --manifest-path research/approximate-symmetry-envelope-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/approximate-symmetry-envelope-001/Cargo.toml
    cargo clippy --locked --manifest-path research/approximate-symmetry-envelope-001/Cargo.toml --all-targets -- -D warnings

## Benchmark ladder

Compare against direct empirical equivariance residuals and exact toy maps under:
- contraction, neutral and expansive maps (`L < 1`, `L = 1`, `L > 1`);
- constant and state-dependent residual bounds;
- horizons 1, 2, 4, 8, 16, 32 and 64;
- shrinking boundary margins and domain-exit controls;
- objective gaps and constraint slack immediately below/at/above their safety thresholds.

Report bound coverage (target 100% on validated toy systems), tightness, false-safe count (target 0), abstention, certified horizon, and receipt overhead. The existing Symtropy dense RATTLE reference is a future physical comparator, not itself evidence for approximate symmetry.

## Claim boundary

Finite-horizon discrete error arithmetic only. No proof of the supplied defect/Lipschitz/domain bounds, continuous-time flow theorem implementation, probability model, physical safety authorization, approximate group closure, exact task invariance, hard-prune authorization or production readiness.
