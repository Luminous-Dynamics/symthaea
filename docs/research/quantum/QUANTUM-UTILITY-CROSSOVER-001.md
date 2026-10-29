# QUANTUM-UTILITY-CROSSOVER-001
## A fail-closed decision gate for quantum application utility

**Status:** research protocol and implementation draft  
**Scope:** application-level comparison, not a quantum device simulator  
**Code:** `crates/domains/symthaea-quantum-comp/src/quantum_utility.rs`  
**Related work:** QARCH-001 (photonic primitive and route envelopes), QARCH-002 (bounded adaptive fusion)

## Research question

For one precisely specified workload, can a quantum or quantum-inspired pipeline return an answer that passes the same acceptance test as the strongest credible classical baseline at lower end-to-end wall time, energy, or cost?

A device's gate fidelity, photon count, sampling rate, or circuit size is useful diagnostic evidence, but none alone establishes application utility. Application comparisons must include setup and compilation, data movement and encoding, state preparation, execution, feed-forward and retries, readout, mitigation, post-processing, and independent verification. State explicitly which terms are measured, modeled, omitted, or amortized.

This gate compares one resource objective per report. Run it separately for wall time, energy, and monetary cost, then retain the three reports rather than hiding tradeoffs in an arbitrary scalar score.

## How Symthaea contributes

### 1. Physics-informed workload discovery

The existing `symthaea-core` physics area includes decoherence and classical-mechanics representations, and the physics bridge contains SI dimensional signatures. These are candidate sources of constraints and observables—not yet a calibrated quantum-hardware predictor.

The research path is:

1. Formulate a scientific question as measurable observables with tolerances and units.
2. Use physics models to derive conservation constraints, symmetries, reduced coordinates, candidate Hamiltonians, and noise hypotheses.
3. Generate workloads at increasing sizes and retain exact workload fingerprints.
4. Generate the strongest plausible classical reduction and baseline before proposing quantum execution.
5. Pass only explicit, provenance-tagged physical parameters to architecture-specific models.

Do not silently equate photon loss, matter-qubit T1/T2, phase noise, thermal noise, logical error rate, or classical simulator uncertainty. An adapter must declare units, semantics, calibration domain, uncertainty, data source, and model version. The existence of a decoherence module does not mean that its values predict a particular photonic device.

### 2. Co-design instead of device-first optimization

QARCH-001 supplies a first-order primitive/path model. QARCH-002 supplies a bounded adaptive fusion state model. Their outputs can contribute to a photonic attempt-cycle estimate only after consistent physical assumptions and interfaces are specified. Add hardware-specific terms for detector efficiency and dark counts, indistinguishability, feed-forward latency, memory lifetime, readout, error mitigation, and verification as evidence becomes available.

The crossover gate is deliberately backend-agnostic. A future adapter may feed it estimates from photonic fusion, circuit execution, annealing, or another architecture, but must not disguise an analytic estimate as a backend observation.

### 3. Where to look for genuinely new uses

Prioritize experiments that exploit structure the target problem already has:

- **Quantum-native simulation:** molecular electronic structure, materials, and many-body observables, compared against domain-specific classical methods such as selected configuration interaction, tensor networks/DMRG, quantum Monte Carlo, and exact diagonalization where feasible.
- **Quantum measurement and sensing:** experiment design where the target is a physical parameter and the score is estimation error at matched resource and confidence.
- **Photonic sampling and network tasks:** workload-specific sampling quality, verified output utility, loss tolerance, and total cost—not sample count alone.
- **Calibration and control:** use classical models to propose the next informative device experiment, then evaluate actual calibration time and predictive improvement. This may create value even before any end-user computational speedup is demonstrated.

General optimization, machine learning, and cognitive applications remain candidate domains. Symthaea should use challenge instances and matched classical solvers to test each one instead of assuming quantum advantage. Quantum-inspired HDC findings do not, by themselves, imply a speedup from quantum hardware.

## Estimator definition and assumptions

For one candidate, let:

- (S) be one-time setup cost for this problem instance;
- (A) be the total resource cost of one attempt, including recurring post-processing and verification;
- (p) be the probability that one independent attempt produces an answer accepted by the task's verifier;
- (q) be the declared output-quality interval;
- (q_{min}) be the quality threshold preregistered for both candidates.

Under independent, stationary attempts, expected cost per accepted answer is:

[
E[C] = S + \frac{A}{p}
]

With conservative input intervals, the lower endpoint is (S_L + A_L/p_U); the upper endpoint is (S_U + A_U/p_L) when (p_L > 0). If (p_L=0), the upper endpoint remains unbounded. If (p_U=0), the candidate is unreachable under its declared model. Overflow or invalid bounds fail closed.

This is an elementary envelope calculation, not a full stochastic simulator. It assumes independent attempts and stationary acceptance probability. Stateful adaptive retry behavior must be evaluated by QARCH-002 first and represented faithfully before using this reduction. Correlated attempts need a richer model.

The quality gate is independent of the resource comparison:

- **Meets:** the entire quality interval is at least (q_{min}).
- **Fails:** even the upper quality bound is below (q_{min}).
- **Uncertain:** the interval crosses the threshold.

Resource dominance is reported only if both candidates meet the shared quality threshold and the conservative expected-resource intervals are separated. The report keeps evidence provenance separate from this numeric decision. Dominance from analytic models or synthetic fixtures is only a modeled hypothesis, never measured quantum advantage.

The input cost intervals are *declared conservative bounds*, not automatically confidence intervals. If they represent statistical uncertainty, the manifest must say how the interval was constructed and provide the sampling protocol.

## Benchmark protocol

### Tier 0 — contract correctness

- Same workload fingerprint, resource objective, and quality threshold on both sides.
- No unit mismatch and no silent missing cost terms.
- Invalid intervals, NaNs, overflow, empty fingerprints, and invalid probabilities are rejected.
- A zero lower success probability cannot produce a finite upper expected cost.
- A zero upper success probability cannot win by reporting zero cost.
- Any candidate below the quality gate is barred from resource dominance.

### Tier 1 — analytic fixtures and negative controls

- Test closed-form examples with tolerances appropriate to floating-point calculations.
- Use tiny cases for which expected attempt counts and expected resource are computed independently by hand.
- Mutate workload IDs, resource units, probability bounds, quality intervals, and costs.
- Include adversarial controls where the quantum path is faster only because encoding, retries, verification, or failures were omitted.
- Target: zero false dominance decisions across the test suite and mutation/negative-control set.

### Tier 2 — physics calibration and simulation

For each hardware architecture, publish the physical model, units, calibration data, valid operating range, and uncertainty. Split data into calibration and held-out sets by device, operating regime, or workload; do not randomly leak near-identical traces across both sets. Include out-of-distribution tests and report residuals by condition.

Targets:
- Closed-form fixtures match to numerical tolerance.
- Held-out measurements fall inside the declared prediction interval at the coverage level promised by the interval construction.
- Any calibration miss, out-of-range input, or unsupported adapter mapping blocks a measured-utility claim.
- Simulated outputs must be compared with an independent simulator or analytic control wherever feasible.

### Tier 3 — application-level comparison

Choose one well-specified scientific task, and preregister the accepted observable, tolerance, confidence requirement, workload sizes, and primary resource metric. Compare:
- the quantum pipeline;
- the strongest credible optimized classical baseline(s) for that domain;
- a simple baseline for sanity;
- ablations for every material component of the quantum workflow.

Record end-to-end wall time, energy and monetary cost where measurable, solution quality, accepted-answer rate, failure rate, peak memory, setup/amortization, backend configuration, and all discarded attempts. Report raw observations and uncertainty, not just a best run. The quantum and classical outputs must pass the same acceptance test.

### Tier 4 — crossover and reproduction

A candidate target for a first *useful* crossover is at least a 2x improvement in a preregistered primary resource objective while matching the quality gate and remaining within declared constraints for the other resources. This is a proposed research target, not an industry standard. Require repeated randomized or interleaved runs, held-out instances/scales, a full cost manifest, and independent reproduction before describing the result as demonstrated utility. A claim of asymptotic advantage requires scaling evidence and a credible classical comparison, not only a small-instance crossover.

## Comparison to current practice

Treat external benchmark suites as baselines and interop targets, not competitors to replace:

- [Open QBench](https://quantum.psnc.pl/en/openqbench/) provides a cross-stack benchmark framework for hybrid workflows, high-level programs, and compiled programs across gate-based, photonic, and annealing platforms.
- [AppQSim](https://doi.org/10.1103/qt55-7d6r) defines application-oriented Hamiltonian-simulation benchmarks across several scientific settings.
- [The Quantum Optimization Benchmarking Library](https://www.nature.com/articles/s43588-026-00991-1) organizes optimization tasks into model-independent problem classes and emphasizes strong classical references.
- [Quantum benchmarking (Nature Reviews Physics)](https://www.nature.com/articles/s42254-024-00796-z) explains why a single benchmark can mislead and why capability and application metrics must be interpreted in context.

The proposed niche for Symthaea is to connect these layers: physics constraints and workload generation, physical resource envelopes, architecture selection, full-stack expected cost, and reproducible utility evidence. It should export results in interoperable formats rather than inventing a private score that cannot be compared with established suites.

## Current implementation boundary

The first Rust module adds a validated interval estimator and comparison decision. It includes unit tests for the exact cost formula, separated and overlapping bounds, quality rejection, zero-success probability, workload/unit mismatch, invalid ranges, and overflow. The tests are authored but have not been executed in this environment.

It does not currently run a quantum backend, implement a photonic hardware simulator, import calibration data, estimate confidence intervals, discover a quantum algorithm, or demonstrate advantage. The next concrete integration step is a provenance-bearing manifest adapter for QARCH-001/QARCH-002 and a small set of physics-derived workload fixtures with known classical answers.
