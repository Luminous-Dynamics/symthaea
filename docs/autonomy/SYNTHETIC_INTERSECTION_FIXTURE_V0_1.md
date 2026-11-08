# Synthetic intersection fixture v0.1

**Status:** synthetic scenario-design contract; execution is not yet implemented by this document  
**Symthaea owner:** [SAM-SCENARIO-001 #10043](https://github.com/Luminous-Dynamics/symthaea/issues/10043)  
**Symtropy execution owner:** [ROAD-AUTONOMY-SCENARIO-001 #3387](https://github.com/Luminous-Dynamics/symtropy/issues/3387)  
**Generator:** `symthaea_vehicle::scenario_discovery`

## Purpose and claim boundary

This is the first small vertical slice for counterexample-guided evaluation. Symthaea emits a versioned candidate with explicit units, fixed actor/road semantics, seed, generator identity and content identity. Symtropy is responsible for bounded execution, runtime identity, observables and exact replay.

The fixture is synthetic. It is not calibrated to any intersection, city, vehicle, pedestrian population or sensor. Its ranges are design-space bounds, not probability distributions. A candidate's position in the generated set must never be interpreted as a real-world frequency or collision probability.

The generator does not contain a simulator, collision detector, controller, perception stack or vehicle physics. It does not classify candidates as safe/unsafe.

## Fixed geometry and actor script

Coordinate convention: metres in a local right-handed planar frame; +x is the ego vehicle's nominal direction of travel and +y is the pedestrian's crossing direction.

| Element | Fixed contract |
|---|---:|
| Road geometry ID | `two-approach-one-marked-crossing-synthetic-v0.1` |
| Ego initial position | (−40 m, 0 m) |
| Ego lane centre | y = 0 m |
| Ego heading | +x |
| Crossing actor | Pedestrian |
| Pedestrian initial position | (0 m, −6 m) |
| Pedestrian movement | +y at 1.4 m/s, 12 m path |
| Time step | 0.02 s |
| Horizon | 8.0 s |

The actor starts its crossing at the candidate's `actor_crossing_start_s`. This is a scripted kinematic actor specification, not a biological movement model. Cyclists, turning vehicles, multiple actors, traffic lights and multi-lane geometry are intentionally separate future families.

## Candidate factors

The generator exposes two deterministic search baselines: (1) affine-permuted stratification with deterministic within-bin jitter and (2) a seed-bound SplitMix64 pseudorandom sampler. Both generate the same typed candidate contract and enforce identical factor bounds. The random sampler is a reproducible search baseline, not a statement of real-world distributions.

| Factor | Unit | Synthetic design range | Interpretation |
|---|---|---:|---|
| Ego initial speed | m/s | 6–22 | Ego speed at scenario start |
| Actor crossing start | s | 0.8–4.0 | Time at which scripted pedestrian begins crossing |
| Occlusion fraction | unitless | 0–0.9 | Fraction of actor observation mask occluded |
| Sensor delay | s | 0–0.25 | Delay applied to the actor observation stream |
| Sensor dropout duration | s | 0–0.4 | Deterministic dropout window beginning at crossing start |
| Lateral localization bias | m | −1–1 | Bias in estimated lateral position, not physical displacement |
| Brake actuation delay | s | 0–0.4 | Delay between brake request and simulated brake response |

All ranges currently carry `SyntheticAssumption` semantics. No normal, uniform, empirical or geographic distribution is asserted. The numerical intervals are not operational limits.

## Candidate identity and reproducibility

The Rust generators return these identity inputs for each candidate:

- schema version and generator-policy ID (`stratified-affine-lhs-v0.1` or `seeded-splitmix64-random-v0.1`);
- seed, ordinal and complete generation budget;
- fixed geometry and actor script;
- exact IEEE-754 factor bit patterns, time step and horizon;
- range semantics.

The candidate ID is a BLAKE3 content identity over those values, with floating-point values encoded as little-endian bit patterns. It changes when a factor, seed, ordinal, budget or generator contract changes. It is not a signature, source-attestation token or cross-version replay guarantee.

`identity_is_valid()` checks the fixed contract, bounded factor values and candidate identity. An execution artifact must additionally bind the candidate ID to the exact Symtropy engine profile, build/toolchain, capability descriptor, seed/clock policy, run output and replay identity under #3387 / #1172.

The maximum generator budget is 4,096 candidates. A zero or over-limit budget is rejected. This limit protects bounded local generation; it does not define a statistically sufficient sample size.

## Execution adapter obligations

Symtropy should reject or explicitly mark unsupported any candidate whose required semantics cannot be represented exactly. It must not silently reinterpret:

- coordinate frames, units, actor start time or actor path;
- observation occlusion, sensor delay or dropout timing;
- localization bias versus physical position;
- brake actuation delay versus command-generation delay;
- requested time step or horizon.

The execution result should record ego and actor pose/velocity traces, sensor-observation timing and availability, localization estimate/error, brake request/response timing, closest actor/ego separation, and contact events only where the engine provides declared contact semantics. Closest separation is not automatically a collision label. An unsupported collision model must produce an explicit unsupported result, not a synthetic collision claim.

Run a fresh-process replay of each frozen candidate before treating its output as reproducible. Preserve incomplete, unsupported, timed-out and invalid runs with typed outcomes and exact exclusion reasons.

## Scenario interchange target

The internal candidate object remains the source-authored, typed experiment contract. The execution bridge should project it into a pinned ASAM representation where the required semantics are expressible:

- OpenSCENARIO DSL 2.2.0 for parameterized/logical scenario descriptions and composition;
- OpenSCENARIO XML 1.4.0 for concrete simulator-interchange scenarios where the selected backend supports the relevant constructs.

The adapter must version the mapping profile and preserve the original candidate ID. It must report every loss, approximation and unsupported construct. Do not claim that DSL-to-XML, XML-to-engine or engine-to-internal-candidate round trips are semantically equivalent unless an explicit conformance suite proves the exact subset. A format conversion does not validate the physics or actor behavior.

See [ASAM OpenSCENARIO DSL](https://www.asam.net/standards/detail/openscenario-dsl/) and [ASAM OpenSCENARIO XML](https://www.asam.net/standards/detail/openscenario-xml/).

## Counterfactual outcome interpretation

A candidate that produces contact in one executed rollout is not automatically a counterexample showing that the tested controller could have prevented it. The next classification layer is [SAM-AVOID-001 #10083](https://github.com/Luminous-Dynamics/symthaea/issues/10083), which must compose #6247 vehicle dynamics and #2022 kinodynamic reachability:

- `avoidable-under-model` requires a concrete admissible evasive trajectory witness replayed under the same declared model/actor/actuator contract;
- `unavoidable-under-model` requires complete enumeration, an analytic bound or a sound conservative reachability result covering the full declared admissible action set;
- `indeterminate` is mandatory for incomplete search, unsupported semantics, OOD dynamics, missing actuator assumptions or exhausted budgets.

Do not use a search reward or the tested controller's own confidence to label avoidability. Do not infer legal fault or real-world preventability from this model-relative label.

## Benchmark sequence

1. Freeze this schema and both candidate-generator revisions.
2. Add explicit positive-control fixtures and seeded challenge fixtures as separate named cases; do not assume a generated boundary candidate is a known failure.
3. Execute the same exact candidates for every controller and RTA ablation. Pair runs by candidate ID, initial state, sensor stream and compute budget.
4. Add boundary and controller-disagreement search only after the random and stratified baselines replay.
5. Minimize observed counterexamples while preserving an exact backend-defined trigger predicate and parent run lineage.
6. Freeze minimized cases into regression. Keep final holdout families isolated by generator policy, topology, actor script and fault composition.
7. Use cluster-aware inference from Symtropy #3385 only when the target estimand and sampling weights have been specified. Adversarial candidate counts are not exposure-weighted risk estimates.

Report safety/constraint outcomes, task progress, interventions, fallback availability, false interventions, uncertainty/calibration, p50/p95/p99 runtime, deadline misses, search cost, time-to-first-counterexample, unique failure families, minimization ratio and replay reproducibility separately. Keep the full per-candidate table and denominators.

## Required controls

The next implementation slice must add explicit, separately named controls:

- **Positive control:** pedestrian visible, no sensor delay/dropout, zero localization bias and zero brake delay, with an initial state selected to permit a controlled stop under the declared test controller.
- **Sensor-loss control:** same geometry and initial state with an explicit observation dropout.
- **Timing-boundary control:** crossing onset chosen to create a declared near-conflict under a stated controller/engine pair.
- **Non-discovery control:** an actor timing and ego initial state pair with a backend-measured non-conflict outcome.

The positive and non-discovery controls must be executed before using discovery yield as a benchmark metric. A seeded challenge is only a challenge candidate until Symtropy returns an observed outcome under a declared engine.

## Claim ceiling

Passing generator tests establishes only that candidate generation, bounded factors and content identities behave as specified. Passing execution/replay tests establishes only the named synthetic scenario semantics on a named Symtropy build. Neither result establishes real-world frequency, collision probability, perception quality, physical-model fidelity, controller safety, ODD validity, roadworthiness, legal authorization or SAE Level 4/5 capability.
