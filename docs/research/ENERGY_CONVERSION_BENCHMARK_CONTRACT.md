# Energy-Conversion Benchmark Contract (v1)

**Status:** proposed machine-readable contract; verifier and empirical reference cases remain follow-on work  
**Schema:** [energy-conversion-benchmark.schema.json](schemas/energy-conversion-benchmark.schema.json)  
**Synthetic fixture:** [energy-conversion-benchmark.synthetic.json](examples/energy-conversion-benchmark.synthetic.json)  
**Bias calibration protocol:** [Differential-Bias Calibration and Coverage Contract](ENERGY_CONVERSION_BIAS_CALIBRATION.md)

## Purpose

This contract makes each energy-conversion benchmark run reviewable and replayable. It binds results to a registered mission, explicit system boundary, exact candidate and solver revision, time intervals, energy accounting, uncertainty assumptions, and source artifacts.

It is deliberately one run per candidate. Compare candidates only after validating their run records against the same benchmark definition and boundary. JSON Schema validation checks record shape and basic field constraints; it **does not** establish physical correctness, verify hashes, check arithmetic, or validate uncertainty propagation.

## Required semantics

### 1. Freeze the mission and system boundary first

Before candidate search, register the mission ID and revision, duty-cycle ID, timebase, required operating phases, expected duration, ambient conditions when material, and acceptance thresholds. Include startup and shutdown in a complete-cycle benchmark unless the explicitly registered mission has a justified alternative.

The boundary must define:
- the energy sources counted as external input and their basis (for example, fuel lower heating value or measured electrical input);
- the exact useful output crossing the boundary;
- whether auxiliary consumption is reported separately and never double-counted as useful output;
- signed change in stored energy (positive means stored energy increased);
- upstream scope (device-only, tank/well-to-useful-output, lifecycle, or a precisely defined custom scope).

A result with a different boundary, fuel-energy basis, or auxiliary policy is not directly comparable. Do not blend ideal-cycle upper bounds, calibrated predictions, and measured system output into a single leaderboard without clearly separated evidence classes.

### 2. Register time intervals and energy flows

Every interval must report start/end time and phase. Energy quantities use joules in this contract; power samples, if provided by a future extension, must use watts and be integrated over the same interval by a declared method.

The independent verifier must check interval indices are unique, intervals are time ordered, the first interval starts at the registered origin, every interval has positive duration, adjacent intervals have neither gaps nor overlaps, the final end time matches the registered mission duration, and required phases actually occur. JSON Schema cannot enforce all these cross-record constraints.

For every interval, recompute first-law closure from the canonical energy fields:

```text
residual = external_input
           - net_useful_output
           - auxiliary
           - rejected_or_loss
           - stored_energy_change
throughput = external_input + max(-stored_energy_change, 0)
relative_error = abs(residual) / throughput
```

Reject intervals without positive throughput, non-finite values, negative flow quantities, energy-balance failures, or numeric overflow. Tolerances must be registered before inspecting candidate scores and justified from numerical and/or measurement uncertainty; never relax them to admit a preferred design. Validate each interval before aggregate summation and validate aggregate closure again.

Recompute aggregate metrics from the interval records. Cached totals and efficiency fields are claims to verify, not trusted facts. Only report net cycle efficiency when stored energy is charge-sustaining within the preregistered rule and the cycle's external input is positive. A transient interval with net storage discharge must not be assigned a naive external-input-only efficiency.

### 3. Report uncertainty without hiding correlation

The schema carries standard uncertainty for each interval energy quantity plus a stated propagation/correlation method. This is metadata until an implementation actually computes the uncertainty.

A qualifying implementation must:
- identify whether values are measured, derived, simulated, or synthetic;
- record calibration and uncertainty sources, coverage factor, units, and assumptions;
- propagate uncertainty to the conservation residual and reported comparison metrics;
- account for correlated quantities using covariance information or a documented Monte Carlo propagation, where relevant;
- keep validation data independent of optimizer tuning data;
- mark comparative conclusions **indeterminate**, not favorable, when material uncertainties or correlations are unknown.

Do not assume errors are independent merely to simplify the calculation. A narrow confidence interval is only justified when the measurement model and assumptions support it. Follow an accepted measurement-uncertainty framework such as the JCGM Guide to the Expression of Uncertainty in Measurement (GUM) and state which method was used: https://www.iso.org/sites/JCGM/GUM-introduction.htm.

### 4. Compare candidates using uncertainty-aware evidence

The Rust verifier's `compare_candidates` function compares two validated records only if they use the same benchmark, mission schedule, system boundary, preregistered policy, and evidence class. It requires the candidate policy's `baseline_id` to identify the supplied baseline candidate.

The current v1 comparison metric is explicitly registered as `net_cycle_efficiency`. The comparator also requires matching solver method, version, source revision, model fidelity, calibration dataset IDs, and random seed. This is intentionally strict for a direct comparison under a common evaluation method. Cross-solver or cross-fidelity comparisons need an independently validated bridge/correlation study rather than silently comparing different model quality. Heat-pump coefficient of performance, specific fuel consumption, thrust-specific metrics, economic cost, and lifecycle outcomes are not aliases for `net_cycle_efficiency` and are not supported by this comparator; the contract rejects an unimplemented metric rather than forcing all technologies into one ratio.

For the currently implemented uncertainty path, both records must declare `correlation_model: "independent"`. The calculation combines the per-interval standard uncertainties for total external input and useful output by root-sum-of-squares, then uses the first-order delta method for `eta = useful_output / external_input`. It combines candidate and baseline efficiency standard uncertainties in quadrature and subtracts the registered coverage factor times that combined uncertainty from the point-estimate difference.

The verifier first subtracts the registered worst-case `maximum_unmodeled_differential_bias_efficiency` from the observed efficiency difference after subtracting expanded measurement uncertainty. This is a **bounded worst-case offset**, not a standard uncertainty to combine by root-sum-of-squares. The registered `differential_bias_bound_basis` records the evidence/reasoning behind the bound; text is an audit aid, not authentication. Policy fields `differential_bias_calibration_artifact_id` and `differential_bias_calibration_artifact_sha256` now require an ID/digest pair matching an entry in `data_provenance`. This catches missing references and digest mismatches inside a manifest, but does not fetch artifact bytes, recompute the digest, authenticate the author, or validate calibration quality. The synthetic oracle separately scores whether a declared bound covers a known injected discrepancy. That property validates the decision rule under controlled assumptions; it does not calibrate the bound for physical use. See the [Differential-Bias Calibration and Coverage Contract](ENERGY_CONVERSION_BIAS_CALIBRATION.md) for the evidence needed before a real bound is decision-grade. The verifier reports `LowerBoundClearsRegisteredMargin` only if the adjusted conservative lower bound exceeds `minimum_superiority_margin`, an **absolute efficiency-fraction difference** (0.02 means two percentage points). Otherwise it reports `LowerBoundDoesNotClearMargin`, which does not prove equivalence or inferiority. Synthetic/not-for-decision records and correlated or unknown uncertainty models yield `Indeterminate`. These enum values describe a calculation against the supplied records only; they do not authenticate the records, verify actual preregistration anchoring, or establish physical superiority.

This is a deliberately limited first-order method, not a universal uncertainty solution. Its independence assumption includes energy channels, intervals, and candidate/baseline results. Shared sensors, calibration, ambient conditions, fuel measurements, or model parameters can induce correlation; those cases require a covariance model or distribution-propagation method such as Monte Carlo before a decision is justified. A coverage factor is not itself a guaranteed confidence probability unless the underlying distributions and degrees of freedom support that interpretation. See the [NIST law of propagation of uncertainty](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty) and [JCGM 101 Monte Carlo supplement](https://www.bipm.org/en/doi/10.59161/jcgm101-2008).

### 4. Preserve provenance and make runs replayable

Each non-synthetic source artifact must carry a SHA-256 hash of its exact bytes. Store URI or stable source identity, retrieval time where relevant, license/access terms, units, transformation steps, and raw-data references. Candidate configuration and solver revisions must be immutable and versioned. Record random seeds where applicable, solver version, model fidelity, calibration datasets, and all generated inputs/outputs needed for replay.

The independent verifier should recalculate all inexpensive invariants without relying on the optimizer's own validity result. It must fail closed on a missing artifact, hash mismatch, unsupported unit/boundary, inconsistent revision, schedule defect, or stale cached metric.

The synthetic_fixture evidence class and not_for_decision=true are useful for testing tooling only. They must never qualify as a measured or published baseline. The included synthetic example deliberately uses invented values and must not be used to support efficiency or superiority claims.

## Comparison protocol

1. Register the mission, boundary, hard constraints, output metric, tolerances, baseline, and preference rule before running optimization.
2. Validate the baseline and candidate manifests and all referenced artifacts at their exact hashes.
3. Recompute per-interval and aggregate conservation, storage balance, and metrics independently.
4. Evaluate equivalent mission performance and constraints. Report net useful output, startup/shutdown cost, auxiliary energy, losses, storage-state change, uncertainty, mass/volume, cost, durability, and emissions where defensible and relevant.
5. Compare performance distributions or justified uncertainty bounds, not just point estimates. If the uncertainty-aware difference does not meet a preregistered superiority threshold, record **no demonstrated improvement**.
6. Compare the Symthaea search against simple search baselines at equal model-evaluation budgets. Preserve failed candidates, constraint violations, random seeds, and all attempts; do not report winners alone.
7. Escalate model fidelity only for promising candidates, then require independently reviewed component and integrated-system testing before making physical performance claims.

## Public mission-profile entry points

A prescribed schedule is not the same as an energy-conversion measurement dataset. Keep the distinction in the evidence label.

- The US EPA publishes dynamometer drive schedules, including US06 and SC03: https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules. These can define repeatable vehicle demand profiles, but do not alone provide a tested powertrain's energy-flow measurements.
- NREL's Transportation Secure Data Center provides processed, second-by-second vehicle drive-cycle data and associated documentation: https://www.nrel.gov/transportation/secure-transportation-data/tsdc-drive-cycle-data.html. Availability and fields differ by survey; verify licensing, variables, and coverage before selecting a benchmark. GPS/duty-cycle data alone cannot establish fuel or electrical conversion efficiency.
- NREL's Fleet DNA describes commercial-fleet operating data and tools: https://www.nrel.gov/transportation/fleettest-fleet-dna.html. Determine whether a specific public dataset contains the required energy channels before treating it as an energy-conversion reference case.

The E1 gate remains: reproduce at least one authoritative or measured reference case with a declared boundary, uncertainty, provenance, and acceptance tolerance before running a novelty-focused search.

## Implementation boundary

The standalone `symthaea-energy-bench` crate now implements an initial independent semantic verifier for parsed v1 records. It checks required mission phases, timeline continuity and duration, time-ordered interval indices, finite/non-negative flow and uncertainty values, per-interval first-law closure, aggregate closure, charge-sustaining storage before exposing net cycle efficiency, basic provenance IDs/hash syntax, stale cached totals, and preregistration timestamp ordering. It also compares candidate and baseline records under a matched mission/boundary/policy and reports whether a first-order uncertainty lower bound clears the declared margin in the independent-input case. Its SHA-256 helper checks exact bytes supplied by the caller.

Run the targeted tests with `cargo test -p symthaea-energy-bench`. The workspace must also pass `cargo fmt --all -- --check`. Neither command was executable in the current authoring environment because a Rust toolchain was absent; do not treat authored tests or queued CI as passes.

Important limits remain:
- Apply the published JSON Schema (Draft 2020-12) separately. The Rust typed parser ignores unknown fields, so semantic validation does not replace schema validation.
- URI resolution, artifact-byte retrieval, and immutable pre-search policy anchoring are caller responsibilities. A self-reported timestamp/boolean cannot prove preregistration without an independently verifiable commit or signed append-only registry.
- First-order uncertainty propagation is implemented only for the explicit mutually-independent input case. The verifier returns indeterminate for unknown/correlated cases; it does not yet support covariance or Monte Carlo propagation. Comparison enum values describe the numerical rule result, not authenticated or physical evidence.
- The crate does not integrate raw sensor data, verify URI-retrieved artifact bytes automatically, validate a solver's physical fidelity, or replace engineering/hardware validation. The comparison result is a numerical rule outcome, not a physical superiority claim. The v1 metric is limited to charge-sustaining net cycle efficiency; other technologies need domain-appropriate metrics and separately validated comparison logic.

The benchmark pipeline is therefore more than a record contract, but it is not yet an automated end-to-end physically validated discovery system.
