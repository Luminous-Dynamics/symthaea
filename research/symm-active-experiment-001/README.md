# Active symmetry experiment reference v0.1

Implements the finite, deterministic reference slice for [SYMM-ACTIVE-EXPERIMENT-001 #12663](https://github.com/Luminous-Dynamics/symthaea/issues/12663).

## Paradigm shift

Passive symmetry identification asks which hypothesis best fits the observations already collected. Active symmetry falsification asks which safe, admissible next observation will most efficiently distinguish the hypotheses that remain.

V0 freezes a finite prediction table before planning. It does not execute physical experiments. The planner cannot infer a hypothesis's response; the caller must provide each predicted outcome.

## Exact adaptive objective

The exact policy solver enumerates every informative affordable probe at each decision node. Its lexicographic objective is:

1. minimize the worst-case number of surviving hypotheses at the depth limit;
2. minimize worst-case cumulative probe cost;
3. break ties deterministically by probe ID.

This is a bounded finite decision-tree search, not an information-gain heuristic. Search-node exhaustion is fail-closed: the solver returns an error and does not return a partial policy as qualified.

The simpler one-step greedy minimax selector and a fixed non-adaptive schedule evaluator are retained as baselines.

## Frozen D4 decision-tree fixture

The test fixture contains four hypotheses and three probes. An adaptive policy identifies the true hypothesis in at most two probes (worst-case cost 2). Every fixed two-probe schedule leaves at least two hypotheses indistinguishable. This is a small exact counterexample to the idea that choosing a good static probe order is equivalent to adaptive experiment design.

All four possible true hypotheses are replayed through the full policy tree. Each remains in the active set at every branch and is the sole terminal hypothesis.

## Evidence and replay

The receipt carries:
- exact canonical bytes for the complete hypothesis/probe/outcome/cost table;
- versioned solver budget;
- the complete adaptive policy tree;
- worst-case survivor and cost values at each node;
- total search nodes visited.

The verifier recomputes the exact policy from the frozen problem and budget. The encoding is versioned but is not a cryptographic digest or signature.

## Reproduction

    cargo fmt --manifest-path research/symm-active-experiment-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symm-active-experiment-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symm-active-experiment-001/Cargo.toml --all-targets -- -D warnings

## Claim boundary

Finite deterministic outcomes only. No probability model, noisy observation, interval prediction, posterior calibration, physical safety authorization, data acquisition, continuous group, unknown outcome handling beyond fail-closed branch absence, causal intervention, physical symmetry proof, or production search-pruning authority is implemented. Bounded-noise elimination belongs to [#12667](https://github.com/Luminous-Dynamics/symthaea/issues/12667).
