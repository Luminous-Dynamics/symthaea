# Active symmetry experiment reference v0.1

First executable finite reference for SYMM-ACTIVE-EXPERIMENT-001 #12663.

## Core capability

Given a finite hypothesis family and a frozen deterministic probe/outcome table, this crate computes an exact minimum worst-case cost adaptive decision tree. It uses dynamic programming over candidate-subset masks and considers only probes explicitly marked admissible.

This differs from ranking probes by one-step information gain: the optimal first probe accounts for the entire future adaptive policy and its worst-case accumulated cost.

## Included baselines

- Exact minimax decision-tree search (correctness target).
- Greedy minimum-largest-outcome-partition probe.
- Maximum entropy probe under an explicit uniform-hypothesis prior.

Greedy and entropy recommendations are non-authoritative comparison arms. The uniform prior is an assumption; it is not a posterior.

## Certificate and replay

The certificate carries canonical bytes for the complete hypothesis/probe/cost/admissibility/outcome contract, the full adaptive tree, worst-case cost, explored subset count and work units. The independent verifier recomputes the exact plan from the original problem and compares the complete tree. Observation replay narrows hypotheses by following the matching branch.

Canonical bytes are deterministic but are not a cryptographic digest or signature. The exact deterministic observation table is assumed to be the declared finite model.

## Frozen fixture

The example includes four competing hypotheses and three probes. Exact minimax chooses the balanced orientation probe first and identifies any hypothesis in worst-case cost 2. It is cross-checked against a separate exhaustive recursive oracle.

## Reproduction

cargo fmt --manifest-path research/symmetry-active-experiment-001/Cargo.toml --all -- --check

cargo test --locked --manifest-path research/symmetry-active-experiment-001/Cargo.toml

cargo clippy --locked --manifest-path research/symmetry-active-experiment-001/Cargo.toml --all-targets -- -D warnings

cargo run --release --locked --manifest-path research/symmetry-active-experiment-001/Cargo.toml --example decision_tree_probe

## Claim boundary

Finite deterministic experiments only. It does not model noisy measurements, causal interventions, continuous group posteriors, physical experiment safety, or proof that a surviving candidate is a physical symmetry. No probe may be physically executed without a separate safety/authorization gate.


## Exact static probe-basis oracle (v0.2)

The crate now includes an exact non-adaptive comparator, `solve_exact_static_probe_design`. It enumerates every subset of admissible probes under an explicit subset budget and minimizes total acquisition cost, then probe count, then lexical probe-ID sequence.

The certificate records:
- the selected minimum-cost probe set;
- every pair that can be distinguished by the admissible probe family;
- pairs that remain intrinsically indistinguishable even with every admissible probe;
- the exact number of subsets examined;
- an optimality flag that is true only after exhaustive completion.

The verifier recomputes the complete subset search from the original contract. If the subset budget is too small, the solver returns an error; it never returns an incomplete exploration as optimal.

### Static versus adaptive result on the frozen four-hypothesis fixture

The three unit-cost probes produce:
- exact minimum static pair-separating basis: all three probes, cost **3**;
- exact adaptive minimax policy: worst-case cumulative cost **2**.

This is a fixture-specific demonstration that a fixed probe set and an adaptive decision tree optimize different objectives. It does not imply that adaptive policies always win, nor does it model noisy costs or probabilities.

### Benchmark ladder

For every frozen finite case, report separately:
1. **Identifiability ceiling:** which candidate pairs can be distinguished by the entire available probe family?
2. **Minimum static basis:** exact minimum-cost set of probes that separates every resolvable pair (on bounded small cases).
3. **Greedy static basis:** weighted set-cover baseline and its gap from the exact static optimum.
4. **Adaptive minimax:** minimum worst-case cumulative probe cost under outcome-dependent selection.
5. **Myopic / entropy / learned acquisition:** policy cost and risk under separately declared priors and noise models.

Do not collapse these into one information-gain score. Each has different assumptions and operational meaning.
