# symthaea-discovery-sim-bridge

A narrow adapter between `symthaea-sim-bridge` and the domain-neutral `symthaea-discovery` contracts.

## Evidence boundary

This crate converts a `SimulationResult` only when `SimulationResult::is_engineering_evidence()` is true. That requires a converged external-solver result with at least one metric and complete backend, solver-version, input-digest, output-digest, and parser-version provenance.

`DryRun` and `Unknown` results are rejected. They are not downgraded into weaker discovery evidence, because doing so would erase the semantic distinction that `symthaea-sim-bridge` already enforces between deterministic fixtures and real external-solver output.

## Mapping

- each normalized `SimulationMetric` becomes one `Prediction`;
- fidelity is exactly `ExternalSimulation`;
- evidence kind is exactly `ExternalSimulation`;
- metric uncertainty is preserved when present, otherwise the run-level uncertainty is inherited explicitly;
- backend + solver version + input/output digests become model provenance;
- parser version and the full external provenance tuple are retained on the evidence reference;
- adapter warnings remain warnings on the returned record rather than being mislabeled as scientific assumptions.

## Authority boundary

This crate does not run a solver, spawn a process, perform network access, mutate candidates, promote evidence, deploy a result, or authorize real-world action. It only translates already-validated solver output into reviewable discovery records.
