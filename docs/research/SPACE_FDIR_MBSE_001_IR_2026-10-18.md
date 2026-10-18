# SPACE-FDIR-MBSE-001: design-derived fault obligations and campaign IR

**Status:** prototype specification + one synthetic communications example  
**Model schema:** 0.1  
**Claim ceiling:** typed traceability and deterministic synthetic event generation only; not flight assurance, complete FMEA/FMECA, or a validated physical model.

## Paradigm shift

Do not create the fault list only after the nominal spacecraft architecture exists. Treat each modeled failure mode as a contract joining the design requirement, component/interface, initiating assumptions, injected event, expected telemetry, monitor deadline, permitted mitigation, protected invariants, and evidence obligations.

The first implementation is intentionally a small Rust intermediate representation (IR), not a SysML v2 parser. A future importer must identify the exact source export/tool version and preserve its revision and content digest. The IR is a versioned projection of the design authority, not a replacement for it.

## Included synthetic fixture

- `crates/domains/symthaea-orbital/fixtures/space_fdir_mbse_001/source_design.txt` is the compact source-side design artifact.
- `crates/domains/symthaea-orbital/fixtures/space_fdir_mbse_001/model.json` is its typed projection.
- The `source_design.sha256` field is the SHA-256 digest of the exact source-design bytes (UTF-8, including the final newline): `748da4fc495778c2b46ef7397ea4a3f435637f837b2aa8e61d10ef4b4d996656`.
- The critical example injects a communications blackout at zero-based simulator step 4 for 3 steps. The expected link signal changes; geometric line-of-sight does not.
- The monitor's fault-specific alarm has a one-step deadline and explicitly named negative controls, including line-of-sight loss without a radio blackout.
- Mitigation actions link to command-authority and power-budget invariants; the expected telemetry label is separate from the simulator's injection receipt.

This example is synthetic and not calibrated to any flight system.

## Output semantics

`MissionFaultModel::validate` fails closed on unsupported schema versions, empty identity fields, malformed source SHA-256, duplicate fault/signal IDs, non-finite delta-v, zero-length blackouts and blackout end-step overflow.

`compile_campaign` emits:
1. a deterministic `FaultScenario` when an injection is representable by the currently implemented orbital fault adapter;
2. an explicit list of `CoverageGap` values for missing requirement trace, event assumption, telemetry-label semantics, observable signal, monitor/negative control, mitigation, protected invariant, or evidence reference; and
3. a `PerturbationSchedule` mapping communications blackouts and velocity impulses to the simulator's corresponding event types.

**A generated schedule does not close a coverage gap.** A critical failure mode with an injection but no observable signal or monitor still remains open. A scenario/injection receipt establishes only that the synthetic event adapter ran; telemetry detection and diagnosis must be scored separately against the declared label semantics. The initial `critical_coverage_closed` predicate means only that the IR has a non-empty critical denominator and all declared structural obligations are present. It does not authenticate reference targets or prove that a model is physically correct.

## Proposed acceptance benchmarks (targets, not results)

| Gate | Proposed target | Evidence needed |
|---|---|---|
| Critical traceability | 100% of the frozen critical-mode denominator has requirement, assumption, injection, observable signal, monitor, mitigation, invariant and evidence links—or an explicit reviewed exclusion in a future schema | Independent completeness audit; frozen denominator |
| Event execution | 100% of scheduled events produce the expected typed receipt, or fail closed before execution | Event receipts + negative controls |
| Diagnosis latency | 100% of synthetic critical injections detected within each monitor's frozen step/time bound | Separate evaluator and reference oracle |
| Attribution | Zero false radio-blackout diagnoses in nominal-link and line-of-sight-only negative controls | Counterfactual scenario pairs |
| Invariant protection | Zero violations of the explicitly modeled invariants across the frozen campaign | Independent invariant monitor; scenario-by-scenario reports |
| Change impact | Any model/source/assumption/monitor/mitigation revision invalidates dependent receipts | Exact source/model digests and dependency-aware stale-evidence tests |
| Design usefulness | Better fault discriminability / faster detection or fewer unmitigated critical modes under a fixed mass, power, cost and compute envelope | Frozen pre-change baseline and reviewed design trade study |
| Onboard viability | Report worst-case execution time, memory, event throughput and degraded-mode behavior on target hardware | Target-hardware measurements; host benchmarks do not substitute |

These are internal engineering targets, not aerospace certification criteria. Do not report aggregate accuracy alone: include false alarms per operational time, detection delay, calibration/abstention, unseen-fault behavior, per-mode coverage, compute and memory costs, and uncertainty where the data supports it.

## Independent validation plan

1. Use the manifest to generate simulator-side scenarios, but keep the reference oracle and telemetry evaluator independently implemented.
2. Run paired nominal, faulted, confounder, missing-sensor, stale-model and unsupported-event controls.
3. Keep event receipts, monitor outputs, telemetry labels and invariant checks as separate evidence classes.
4. Freeze source/model digest, fixture set, split policy, operating mode and thresholds before reporting comparative results.
5. Only then compare design alternatives—for example, whether an added independent radio-health signal improves fault attribution enough to justify its mass, power, cost and integration burden.

## Boundaries and next tranche

This tranche does not parse SysML v2, resolve external requirement/evidence references, calculate fault probabilities, model radio physics, establish diagnosability for unmodeled failure modes, prove mitigation safety, control real hardware, or authorize flight operations. An ID's presence is not proof that the referenced artifact exists or is authoritative.

Next: add independent external-reference resolution and dependency digests, add an explicit reviewed-exclusion state, generate paired positive/negative scenario manifests, and connect monitor/telemetry evaluation without collapsing simulator injection receipts into detection results. A SysML v2 adapter should be admitted only after a precise exported source form and tool version are pinned.


## Follow-on: pairwise critical fault-signature preflight

The adjacent SPACE-FDIR-DIAG-001 work adds optional canonical `signature_value` labels to observable signals and produces one pairwise finding for every unordered pair of declared critical fault modes:

- `Distinguishable`: at least one shared signal ID and unit has explicit, non-empty, different canonical values;
- `Indistinguishable`: the two complete declared signature sets are exactly equal;
- `NotDemonstrated`: no valid separator is found and missing outcomes, incomparable units, or different signal sets prevent a stronger conclusion.

Unknown outcomes never count as separators. Different signal IDs or units do not count as evidence of a distinguishing observation. The report binds the model revision and source-design digest. This is conservative structural lint over declared categorical signatures only; it does not prove physical observability or classifier accuracy.

### Additional proposed benchmarks

- Pair denominator is exactly (n(n-1)/2) for (n) critical modes; the zero/one-mode report does not claim pairwise diagnosability.
- Zero complete-signature collisions go unreported.
- Every pair that lacks a verified structural separator remains visibly unresolved or indistinguishable until explicitly reviewed.
- Sensor recommendations name the fault pair separated and are evaluated against fixed mass, power, cost, bandwidth, calibration, independence, and compute assumptions.
- Full minimum-sensor-set and common-mode cut-set analysis continues to belong to the existing decision-specific PIE-009N diagnosability/sensor-cut oracle (#2032); this module intentionally does not implement another generic cut-set algorithm.


## Follow-on: temporal diagnosability and prediction confidence

The temporal analyzer extends the static pairwise signature preflight with optional time-indexed predictions per observed signal. Each sample is relative to the fault injection and contains a categorical value plus a declared confidence in integer PPM. This is a declared model-prediction confidence, not a calibrated empirical probability unless a separate calibration study establishes that interpretation.

The policy freezes a maximum observation horizon, the latest acceptable response step, and a minimum confidence threshold. The analyzer reports the earliest qualifying divergence and distinguishing signal IDs, or one of:
- `LateDivergence`: a valid separation exists but only after the response deadline;
- `IndistinguishableThroughHorizon`: complete, sufficiently confident sampled signatures match throughout the declared horizon;
- `NotDemonstrated`: missing/unknown values, low-confidence predictions, nonmatching sample offsets, or incomparable signal mappings prevent a supported conclusion.

Only the same signal ID, same unit, and same relative sample offset can be compared. Unknown, empty, missing-confidence, or below-threshold samples never count as separation. Reports bind the exact model/source identity. No real elapsed time is inferred from simulator step counts.

### Temporal benchmark additions (proposed)

- 100% of unordered critical pairs receive a temporal disposition for the frozen horizon.
- Zero low-confidence or unknown samples are counted as separators.
- Zero late divergences are reported as timely.
- Report per-pair earliest separation steps and aggregate minimum/median/p95 only when actual scenario outputs support those summaries.
- Evaluate sensitivity across frozen confidence thresholds and horizon/deadline changes; all changed policies produce distinct report identities.
- Keep physical sensor calibration, empirical fault-detection probability, false-alarm rate, and real-time/onboard performance outside this structural model until measured evidence exists.
