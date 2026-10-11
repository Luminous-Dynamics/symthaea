# symthaea-discovery

Domain-neutral discovery contracts for Symthaea.

This crate provides the shared language needed to connect existing scientific-method, uncertainty, Pareto, expected-information-gain, external-solver, and evidence systems without rewriting them into one monolith.

## Scope

- Candidate identity and deterministic specifications
- Scalar predictions with explicit units, provenance, fidelity, and uncertainty
- Evidence references that distinguish literature, datasets, models, simulations, experiments, replications, device validation, and field observations
- Separate objectives and hard constraints for Pareto-oriented discovery
- Constraint assessment using the unique highest-fidelity matching prediction
- Experiment proposals with expected information gain and generic resource estimates
- Deterministic `DiscoveryCertificate` review artifacts with BLAKE3 digests
- Adapter traits for candidate generation, evaluation, and next-experiment selection

## Non-goals

This crate does **not**:

- execute experiments or physical actions;
- invoke solvers itself;
- synthesize or procure materials;
- replace existing domain implementations;
- collapse multiple objectives into a single hidden score;
- promote simulation evidence into experimental evidence;
- silently choose between conflicting equal-fidelity predictions.

Every `DiscoveryCertificate` repeats an explicit computational-only capability classification and is intended for human review.

## Integration path

Initial adapters should wrap, not move, existing implementations:

1. `src/scientific_method.rs` for hypothesis/test/update lifecycle;
2. Spark expected-information-gain experiment planning;
3. existing Pareto/NSGA-II implementations;
4. `symthaea-sim-bridge` simulation evidence and uncertainty;
5. `symthaea-process-discovery` certificate/search patterns;
6. domain-specific discovery crates such as energy, materials, biology, and engineering.

The first energy consumer should use these contracts to express application-dependent candidates and objectives while leaving numerical truth to existing or external physics models.
