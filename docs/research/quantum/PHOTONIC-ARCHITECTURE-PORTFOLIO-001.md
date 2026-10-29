# Photonic Architecture Portfolio 001: a falsifiable design space for Symthaea

**Status:** research proposal and component-envelope implementation plan; no architecture-performance result is claimed.  
**Tracking issue:** [QARCH-001](https://github.com/Luminous-Dynamics/symthaea/issues/8878)  
**Scope:** photonic resource generation, routing, fusion, encoding, error correction, architecture-level control, and application validation.

## Executive thesis

Do not bet on one photonic encoding or benchmark. Build an evidence-carrying architecture laboratory able to compare distinct families under their own valid physical models and then compare them at matched logical/application boundaries. The potentially differentiating Symthaea contribution is loss-aware cross-layer co-design: have the controller jointly reason about component calibration, optical routing depth, fusion scheduling, resource-state choice, code/layout parameters, feedforward latency and all-in resource cost. This is a testable architecture-control hypothesis, not a claim of a new quantum hardware primitive.

The first objective is not to assert a new universal photonic computer. It is to determine **which design is best for which workload, hardware assumptions and target logical error**, and to make it cheap to falsify a design before fabricating or running a large device.

## Six architecture families and what they can actually answer

| Family | Representative approach | Main opportunity | Bottleneck to model explicitly | Appropriate first workloads |
|---|---|---|---|---|
| A. Integrated discrete-variable (DV) dual rail | Heralded single photons, silicon/silicon-nitride photonics, linear optics and fusion | Fabricable networks, high-quality conditional operations, modular interconnects | End-to-end loss, source heralding, switching and detector yield, calibration/heater cost | Small verified circuits, fusion primitives, networked state preparation |
| B. Emitter-driven time-bin fusion | Deterministic or reconfigurable entangled-photon sources, synchronous fusion, adaptive repeat-until-success (RUS), foliated/Floquet colour-code lattices | Reduce optical depth and multiplexing; use source-side feedback to change subsequent emission | Emitter noise/distinguishability, feedforward, synchronous timing, spin coherence, loss along the whole path | Logical-state resource estimates, fusion-network threshold studies, low-depth QEC cycles |
| C. Continuous-variable (CV) cluster states with GKP encoding | Squeezed-light cluster states with bosonic Gottesman–Kitaev–Preskill (GKP) qubits and measurement-based operations | Deterministic Gaussian operations, natural continuous-variable processing | Available squeezing, finite-energy GKP state quality, loss, non-Gaussian injection/subtraction, homodyne noise and decoder | Small CV circuits, GKP preparation/teleportation, loss and decoder sweeps |
| D. Hybrid DV–CV/bosonic | Hybrid logical encodings and encoded measurement/fusion designs | Potentially combine single-photon erasure/loss handling with bosonic-code operations and more deterministic encoded measurements | Encoding/preparation overhead, which operations are genuinely near-deterministic, optical loss, conversion and feedforward | Matched encoded-gate and resource-overhead comparisons |
| E. Specialized Gaussian-boson sampling / photonic Ising | Gaussian boson sampling (GBS) or related photonic sampling mapped to a restricted optimization task | May offer an efficient specialized physical sampling primitive | Distribution-to-objective relevance, sampling bias, loss, classical simulation/baseline strength, post-selection and verification | Task-specific sampling diagnostics and small exact optimization instances |
| F. Programmable interferometers and modular photonic networks | Reconfigurable integrated interferometers, chip-to-chip/fibre interconnects, tunable optical mesh | Reconfiguration and modular scaling; different applications can share components | Heater power/drift, phase calibration, insertion loss, chip-to-chip loss, tuning time and routing congestion | Reconfiguration, networked primitives, calibration stability and throughput |

These rows are not interchangeable competitors. A component platform is not an error-correction architecture; GBS is not a universal fault-tolerant algorithm by default; conditional gate fidelity is not unconditional task success. Any comparison must preserve those distinctions.

## Research-grounded starting points

### A. Integrated DV silicon photonics

The 2025 *Nature* integrated-platform paper reports 99.98% ± 0.01% state-preparation/measurement fidelity, 99.50% ± 0.25% HOM interference visibility, 99.22% ± 0.12% two-qubit fusion fidelity, and 99.72% ± 0.04% chip-to-chip interconnect fidelity, **conditional on photon detection and not accounting for loss**. These are useful component reference points, not an end-to-end logical success rate. Use the paper's component table and its loss caveats as a reproducibility target, then model total optical path loss and detection yield separately.

Reference: PsiQuantum team, “A manufacturable platform for photonic quantum computing,” *Nature* 641 (2025), https://doi.org/10.1038/s41586-025-08820-7.

### B. Emitter/time-bin fusion-based computation

The May 2026 *npj Quantum Information* paper proposes a quantum-dot implementation combining deterministic entangled-photon sources, time-bin encoding, synchronous low-connectivity fusion, adaptive RUS fusions and a foliated Floquet colour-code architecture. It provides detailed physical-noise modelling and logical-cycle resource estimates; those estimates remain a theoretical architecture proposal rather than a completed universal fault-tolerant processor demonstration.

Reference: Chan et al., “Practical blueprint for low-depth photonic quantum computing with quantum dots,” *npj Quantum Information* 12 (2026), https://doi.org/10.1038/s41534-026-01258-3.

### C. GKP/CV and hybrid strategies

The 2025 integrated GKP-source paper reports an integrated silicon-nitride photonic source exhibiting mode-level signatures relevant to GKP encoding. It is a component milestone; it should not be described as fault-tolerant logical computation on its own.

Reference: “Integrated photonic source of Gottesman–Kitaev–Preskill qubits,” *Nature* (2025), https://doi.org/10.1038/s41586-025-09044-5.

A 2026 review of hybrid photonic quantum computing surveys DV, CV and bosonic encodings and discusses trade-offs in loss, determinism, error correction and resource states. Use it to identify architectural candidates, not as an empirical ranking of complete machines.

Reference: “Photonic hybrid quantum computing,” *Newton* 2(3) (2026), https://doi.org/10.1016/j.newton.2025.100359.

### D. Emitter-aware threshold studies

A 2025 *PRX Quantum* study reports proposed emitter-tailored fusion architectures with simulated fault-tolerance thresholds of 8% photon loss and 4% emitter distinguishability under its specified model. Such thresholds depend on the lattice, error model, decoder and assumptions; they are not a directly comparable device-level loss measurement.

Reference: “Tailoring Fusion-Based Photonic Quantum Computing Schemes to Quantum Emitters,” *PRX Quantum* 6, 020304 (2025), https://doi.org/10.1103/PRXQuantum.6.020304.

## Proposed Symthaea-specific hypotheses

### H1 — loss-aware compilation can beat ideal-fidelity compilation

A compiler that minimizes optical depth and likely-erasure exposure, while respecting required fusion connectivity and code constraints, may improve logical throughput at fixed physical parameters compared with a compiler selecting layouts from gate fidelity alone.

**Falsifier:** under matched source, loss, detector and gate parameters, the confidence interval for all-in logical throughput shows no improvement, or any improvement disappears after adding routing/switch overhead.

### H2 — adaptive RUS should be selected by a break-even rule, not preference

RUS can reduce the need to discard an entire resource state after an unsuccessful fusion, but repeated attempts consume time, emitter control budget and memory coherence. Compare adaptive RUS with fixed-attempt and restart-whole-resource policies using measured attempt latency, loss, feedforward and error correlations.

**Falsifier:** lower mean resource count comes with worse tail latency, logical error, or total resource use at the predeclared target.

### H3 — hybrid encoding should be chosen only when end-to-end savings offset conversion costs

Treat encoded-gate quality, encoding success, conversion, loss, and recovery as part of one cost boundary. Do not assume “hybrid” is superior because one sub-operation is nearly deterministic.

**Falsifier:** at matched logical error and workload, the hybrid Pareto frontier is dominated by an eligible DV or CV design.

### H4 — robust design choice should survive component uncertainty

Choose architecture and route using calibration intervals and uncertainty, not a single ideal component number. Report best, nominal and worst-case performance across documented parameter intervals and sensitivity to correlated drift where evidence supports it.

**Falsifier:** the selected design changes under small, plausible perturbations or its apparent benefit disappears once calibration uncertainty is propagated.

## Work plan

1. **Component envelope (initial prototype):** implemented in the dependency-free Rust module photonic_architecture. It provides dB-to-transmission, separate source/path/detector/gate yield factors, idealized probabilistic-source multiplexing, expected restart cost, and deterministic parameter-interval propagation. These calculations assume independent events and are not a fault-tolerance simulator.
2. **Topology-level routing (initial prototype):** the same module implements directed routes under three policies: fewest hops, minimum nominal loss, and minimum worst-case loss from declared loss intervals. A small fixture demonstrates that a nominally attractive route can be dominated under worst-case loss. It ignores crosstalk, wavelength contention, interference, multi-commodity congestion, fusion dependencies and code constraints; those need separate models.
3. **Published-data reproduction:** encode paper-derived values with source, date, units, conditionality and uncertainty. Reproduce selected component calculations and label assumptions not specified by the paper.
4. **Family-specific resource estimators:** add DV-FBQC and emitter/time-bin resource models first; add CV/GKP only with explicit squeezing, finite-energy, loss and decoder models; add hybrid only after declaring its encoded operation contract. Do not assign a universal architecture score.
5. **Loss-aware fusion compilation:** for matched small fusion graphs, compare hop-minimal routing, fidelity-only layout and robust loss-aware layout. Include fixed-RUS, adaptive-RUS and restart controls as separate policies.
5. **Logical-code simulation:** sweep code distance and physical error channels; report logical error per QEC cycle, threshold fit and uncertainty, decoder runtime, photons/resource states, feedforward latency, optical depth and component counts.
6. **Application benchmarks:** choose tasks per architecture. Use QED-C's application-oriented prototype benchmark suite for basic circuits where compatible, exact classical simulators for small circuits, and strong task-specific classical baselines for optimization/sampling. Published sampling advantage is not evidence of application advantage.
7. **Hardware-in-the-loop:** only after model qualification, feed calibration and device receipts into the estimator. Preserve raw results, exact code/configuration digests, environment, measured/unmeasured cost boundaries, and run failures.

## Minimum benchmark scorecard

### Component level

- Source valid-state probability and heralding efficiency; photon-number purity where relevant.
- HOM visibility and its measurement uncertainty.
- Conditional gate/fusion fidelity **and** unconditional success/yield.
- Propagation, switch, filter, coupling and interconnect loss (dB).
- Detector efficiency and photon-number resolution.
- Feedforward latency, reset latency, component drift and recalibration overhead.
- CV-specific squeezing and GKP finite-energy/state-quality measures where appropriate.

### Architecture and fault tolerance

- End-to-end resource-state yield including every declared loss channel.
- Optical depth and number of active/passive components per photon.
- Logical error per QEC cycle versus code distance with all declared noise channels active together.
- Threshold sensitivity to correlated loss, mode mismatch and drift.
- Photons/resource states, sources, detectors, switches and memory qubits per logical cycle.
- Logical clock rate and tail latency, not only gate time.
- Resource and energy cost per *verified logical operation* at a predeclared target error, plus failure probability.

### Application level

- Task correctness or distributional quality against a declared metric.
- Classical best-known and exact optimum on tractable subsets.
- Sample count, post-selection rate, verification cost, compilation, calibration and queue costs.
- Held-out instance/generalization performance; raw per-instance outcomes and uncertainty.
- A matched budget boundary across all methods, with missing energy/cost marked “unknown” rather than zero.

There is no universal SOTA number for these mixed architecture families. Numerical targets should be derived from the selected logical-code threshold, logical target error, workload, source, detector and loss assumptions—not a cross-paradigm composite score.

## Promotion rules

- Code/model checks must pass before the model is considered implemented.
- Reproduced paper values must separate measured component data from simulation, extrapolation and assumptions.
- Compare on at least two regimes: nominal and stressed hardware conditions; sweep loss, detector efficiency, source success, gate failure and feedforward latency.
- Include ablations proving which control choices change the outcome.
- Predeclare the primary endpoint and noninferiority constraints before held-out confirmation runs.
- No architecture gets promoted from one improved component metric if end-to-end logical success or application quality worsens.
- Publish null/negative results; do not claim quantum advantage without a matched and strong classical comparison at the application layer.

## Tooling references

- QED-C application-oriented quantum benchmarks (prototype set; not formal performance standards): https://github.com/sri-international/qc-app-oriented-benchmarks
- Integrated photonic platform benchmark data associated with the 2025 *Nature* paper: https://github.com/PsiQ/2404_17570
- OpenQBench, an open-source framework spanning gate-based, photonic and annealing paradigms: https://quantum.psnc.pl/en/openqbench/

This document defines a falsifiable research program. It does not assert a new quantum device, a superior architecture, SOTA, or quantum advantage.
