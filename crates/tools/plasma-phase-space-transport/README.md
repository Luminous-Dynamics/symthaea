# Plasma phase-space transport reference

A standalone, dependency-free Rust small-N reference for balanced 1-Wasserstein distance between nonnegative empirical measures on the one-dimensional periodic cylinder S^1_L x R.

## Ground metric

The position coordinate uses the shortest periodic arc:

\[
d_{S^1_L}(x,x')=\min(\delta_x,L-\delta_x),\qquad
\delta_x=|x-x'|\bmod L.
\]

The dimensionless product ground metric is:

\[
d((x,v),(x',v'))=
\sqrt{
\left(\frac{d_{S^1_L}(x,x')}{\ell_x}\right)^2+
\left(\frac{v-v'}{\ell_v}\right)^2
}.
\]

Domain length, position scale and velocity scale are part of the frozen benchmark identity.

## Measure and mass contract

This first reference is **nonnegative, balanced transport only**. It rejects negative signed delta-f weights. Co-located positive particles are canonicalized and aggregated before transport, preserving particle permutation and split/merge invariance.

Original total masses are compared using a strict relative mismatch tolerance before both measures are normalized to probability mass. The original masses and mismatch are reported. A material mismatch is an error; it is not silently normalized away. Two empty measures have zero cost by convention; one empty and one nonempty measure is rejected.

Signed measures remain on the characteristic-MMD lane. This crate does not reinterpret signed weights as positive transport mass.

## Algorithm

The exact finite transportation LP is evaluated by a successive-shortest-augmenting-path min-cost-flow method. Bellman-Ford is used on the residual network because reverse arcs have negative costs. The solver reports augmenting-path count, shortest-path relaxation count, and N-by-M ground-cost evaluations. Probability weights are normalized with compensated summation, and the solver drains every representable source residual rather than stopping at a fixed total-flow epsilon. If any destination residual remains, it returns an unresolved-mass error instead of reporting a partial W1 value.

This is a reference implementation for small supports. Its dense bipartite graph and repeated Bellman-Ford searches are not a production-scale Sinkhorn or NUFFT-accelerated method. The result is the W1 cost between probability-normalized measures under the declared ground metric.

## Tests

The tests cover:
- identical measures;
- particle permutation and co-located positive split/merge;
- shortest-arc seam behavior;
- velocity translation under the declared velocity scale;
- a changed position/velocity correlation with unchanged marginals;
- strict mass mismatch rejection;
- signed-weight rejection;
- empty measures;
- invalid scales and non-finite data;
- residual-graph rerouting behavior;
- a sub-1e-12 normalized mass atom whose large velocity separation contributes order-one W1.

## Run

\`\`\`sh
cargo fmt --manifest-path crates/tools/plasma-phase-space-transport/Cargo.toml -- --check
cargo test --locked --manifest-path crates/tools/plasma-phase-space-transport/Cargo.toml
cargo clippy --locked --manifest-path crates/tools/plasma-phase-space-transport/Cargo.toml --all-targets -- -D warnings
\`\`\`

W1 is a separate distribution-fidelity axis from characteristic MMD. Neither replaces field residuals, trajectory error, spectra, physical invariants, convergence or complete solver time-to-solution.
