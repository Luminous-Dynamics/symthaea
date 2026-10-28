# Energy Architecture Generation v0.1

**Status:** implementation spike; deterministic synthetic templates only. This is not yet an open-ended Symthaea design agent, optimizer, or validated engineering tool.

Related work:
- Symthaea [ENERGY-ARCHITECT-001 #8496](https://github.com/Luminous-Dynamics/symthaea/issues/8496)
- Symtropy [Energy System Design Studio #2721](https://github.com/Luminous-Dynamics/symtropy/issues/2721) and [candidate contract PR #2722](https://github.com/Luminous-Dynamics/symtropy/pull/2722)
- Existing Symthaea [scheduler benchmark corpus #7248](https://github.com/Luminous-Dynamics/symthaea/issues/7248) and [SAM battery cross-validation #7305](https://github.com/Luminous-Dynamics/symthaea/issues/7305)
- Sol-Atlas [exact dispatch oracle #309](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/309)

## What exists in this change

The new module `crates/domains/symthaea-engineering/src/energy_architecture.rs` currently provides:

- Serializable, strongly typed candidate DTOs matching the v0.1 Symtropy interchange field names.
- An independent structural validator for candidate IDs, evidence class/source references, port directions, carrier and unit compatibility, heat temperature grades, resource/service references, scenario chronology and hard-constraint/objective metadata.
- Three deterministic synthetic topology templates: grid-only; PV plus grid; PV plus grid plus battery.
- Negative controls for carrier mismatch, chronology gaps and implicit evidence-class promotion.
- Stable candidate IDs and a fixed UTC two-interval synthetic stress case. Weak candidates are retained for unbiased comparison, not silently dropped.

It does **not** yet use Symthaea's cognitive loop to invent arbitrary topologies, does not dispatch these designs, and does not simulate a real site. The reference templates exist to stabilize the cross-repository interface before adding search.

## Reuse existing ownership

- `symthaea-engineering`: candidate generation/orchestration and requirement binding.
- `symthaea-grid-physics`: existing electrical/battery/scheduling model for cases inside its declared radial real-power scope.
- `symthaea-sim-bridge`: solver-agnostic request/result and coupled-simulation boundary.
- `symthaea-digital-twin`: state/telemetry/uncertainty representation; it is not itself a physical solver.
- `symthaea-formal-safety`: safety-case/evidence representation, without implying certification.
- Symtropy: the versioned candidate contract and deterministic time-coupled twin scenarios in explicitly supported physics scope.
- Sol-Atlas: resource evidence, typed multi-carrier energy accounting, exact bounded dispatch reference and per-objective reports.
- Mycelix: provenance, rights, review and approval records; cryptographic identity does not prove source truth.

Do not introduce a second storage scheduler, a second evidence catalog, or a duplicate physics kernel for this integration.

## Design loop to implement next

1. **Requirements:** define essential and flexible services, temperature grades, jurisdiction/site, time horizon, resource boundaries and hard constraints.
2. **Candidate generation:** expand from fixed templates to structured graph transformations and diverse candidates. A change is allowed only when its new nodes/ports/claims remain typed and provenance-bound.
3. **Structural validation:** reject unknown nodes/ports, carrier/unit mismatch, incompatible heat grades, unsupported parameter evidence, chronology gaps or missing service samples.
4. **Exact subproblem:** route representable small cases through Sol-Atlas's bounded dispatch oracle and preserve the oracle's quantization/model ceiling.
5. **Twin evaluation:** map each candidate to Symtropy only where the physical model exists. Record model version, time-step, initial state, raw trajectory and residuals.
6. **Independent re-audit:** recompute constraints and summary outputs from raw snapshots rather than trusting the generator's report.
7. **Comparison:** retain infeasible candidates with structured counterexamples; report service shortfall, cost, carbon, water/material/logistics where evidence supports them, curtailment, recovery time, uncertainty and compute separately.
8. **Evidence-aware selection:** a proposal with unknown inputs cannot out-rank another simply because missing cost/impact values default to zero.
9. **Human review:** simulation cannot authorize real controls, protection settings, construction, procurement or operational change.

## Benchmark targets

Targets only; no measurements have been claimed:

- All deterministic templates pass the structural validator and serialize/deserialize byte-stably under pinned builds.
- 100% detection of the implemented negative controls.
- Preserve all generated/invalid candidates and rejection reasons; never report validity rate using only successful outputs.
- Add at least 10,000 deterministic generated candidate/scenario pairs before publishing aggregate design-quality results.
- Compare against random/diverse graphs, hand-built transparent baselines and constrained exact solutions with identical information/compute budgets.
- Report candidate validity rate, architectural diversity, feasible Pareto coverage, essential unserved energy, critical deficit duration, objective regret, uncertainty/sensitivity, model discrepancy, runtime/memory and compute energy where measurable.
- Split by site/time/episode for held-out qualification; do not tune the generator against the test set.
- Ablate HDC similarity, cognitive search, evidence-aware ranking and active experiment selection. Each must improve validity, diversity, decision quality or computational efficiency versus the simpler baseline to justify its complexity.

## Qualification boundary

Syntactic/structural validation is weaker than energy conservation; energy accounting is weaker than calibrated physics; calibrated simulation is weaker than software/controller/power hardware-in-the-loop; HIL is weaker than independently reviewed field evidence. The report must keep those levels separate.

External comparison lanes include [SAM](https://sam.nlr.gov/) for supported component/storage trajectories, [REopt](https://developer.nlr.gov/docs/energy-optimization/reopt/) for matched site/microgrid optimization cases, and [PyPSA](https://docs.pypsa.org/latest/) / [GenX](https://genxproject.github.io/GenX.jl/stable/User_Guide/workflow/) for matching network/planning subproblems. Do not claim to outperform these tools without pinned versions, matched assumptions, full results and reproducible artifacts.


## Deterministic byte export

The companion binary `energy_architecture_candidates` makes the reference generator's output consumable by the cross-repository pipeline without reconstructing the candidates by hand:

```sh
# Export all validated candidates as one JSON array.
cargo run -p symthaea-engineering --bin energy_architecture_candidates -- --output /tmp/energy-candidates.json

# Export one exact candidate object for the current Sol-Atlas adapter.
cargo run -p symthaea-engineering --bin energy_architecture_candidates -- --candidate-id synthetic-pv-battery-grid --output /tmp/energy-candidate.json

# Bind the exact exported bytes for the integration manifest/receipt.
sha256sum /tmp/energy-candidate.json
```

The binary validates every generated candidate with the existing structural validator before writing. Unknown candidate IDs and invalid candidates fail with a nonzero exit; no fallback candidate is substituted. JSON output is deterministic for a pinned build and code path, but consumers must still validate against the pinned shared Symtropy schema and bind receipts to the exact bytes consumed.

This is a serialized bridge primitive, not proof of cross-repository integration: the command still needs to be executed on the exact PR head, its bytes validated against the shared schema, and the single-candidate export compiled by Sol-Atlas and audited against its actual run report.


## Bounded parametric design-space baseline

A second, deliberately transparent generator path now sweeps a 36-point Cartesian design space around the existing PV+battery+grid topology:

| Variable | Declared values |
|---|---|
| PV rated power | 1, 2, 3 kW |
| Battery energy capacity | 1, 2, 3 kWh |
| Battery charge/discharge power | 1, 2 kW |
| Grid import power | 1, 2 kW |

The enumerator mutates typed numeric claims and scenario resource availability together, assigns stable candidate IDs, runs the structural validator, and keeps a structured rejection record if a variant cannot be formed or validated. The design-space manifest reports the declared axes, requested/emitted/rejected candidates and an explicit synthetic-only claim ceiling.

This is a bounded sizing sweep, not an unbounded architecture invention agent, nor a performance ranking. All 36 variants retain the same topology and two-interval synthetic scenario, so the immediate purpose is to establish a reproducible baseline and compare the downstream exact-dispatch/twin pipeline on more than one hand-picked point. Next steps should add an uncertainty-aware scenario family and controlled topology edits only with explicit connectivity and model-coverage checks.

Example commands:

```sh
# Export the full 36-point bounded search manifest and materialize all exact
# candidate documents from the same in-memory search result.
cargo run -p symthaea-engineering --bin energy_architecture_candidates -- \
  --design-space --candidate-output-dir /tmp/energy-candidates \
  --output /tmp/energy-design-space.json

# Export one sizing variant as the exact candidate object consumed downstream.
cargo run -p symthaea-engineering --bin energy_architecture_candidates -- \
  --candidate-id synthetic-parametric-pv-1kw-battery-1kwh-battery-power-1kw-grid-1kw \
  --output /tmp/energy-candidate.json
```

The unit suite asserts deterministic order/content, 36 unique IDs, schema-DTO round trips, structural validity of every emitted candidate, axis coverage, synchronized source limits and outage zeros, and preservation of unknown economic objectives.
