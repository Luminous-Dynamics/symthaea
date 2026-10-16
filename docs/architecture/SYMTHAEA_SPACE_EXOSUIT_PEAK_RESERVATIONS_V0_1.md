# SX-034: Time-Windowed Peak Resource Reservations v0.1

**Status:** experimental software contract analyser; synthetic declarations only  
**Implementation:** suit_peak_reservations  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/7550  
**Predecessor:** SX-033 cumulative budgets, PR #7547

## Paradigm shift: total sufficiency is not peak feasibility

SX-033 asks whether cumulative required consumption plus a declared reserve fits conservative supply. A system can satisfy that sum while two simultaneously active tasks overbook power, coolant throughput, data rate, compute budget or shared equipment.

SX-034 models those as separate, dimensionally typed peak resource pools. It represents time in mission-relative integer milliseconds, not wall clock, and each reservation occupies a half-open interval [start, end). An event-driven sweep partitions each pool at reservation boundaries and incrementally updates active required/optional demand. It processes endings before starts at the same tick, so half-open semantics are exact without rescanning every reservation at every slice.

## Conservative capacity and task priority

For each pool:

- Conservative capacity is the declared interval's lower bound.
- Reserve is a caller-declared fraction of that lower bound, rounded up to the next integer quantum.
- Usable capacity is lower bound minus reserve.
- Each active reservation contributes its declared upper demand bound.
- Required and optional task demand are summed independently for each elementary time slice.

The classification rule is ordered: if required demand alone exceeds usable capacity, the slice is RequiredOversubscribed and blocks. Otherwise, if required plus optional demand exceeds capacity, the slice is OptionalOversubscribed and degrades. Otherwise the slice is within declared capacity.

This analysis does not rearrange tasks, issue commands, shed loads, or infer that optional work can be rescheduled. It only makes the declared concurrency conflict explicit. Window counts refer to elementary slices formed from distinct reservation endpoints; adjacent overloaded slices are not merged into a higher-level interval in v0.1.

## Explicit binding and determinism

A plan pins the exact mission/configuration revision and SX-031/SX-032 digests. Each pool also pins an exact provider component/revision, a non-empty set of exact SX-030 connection IDs, and a non-empty set of interface kinds that actually provide that resource. Those connections must exist, involve the declared provider, contain a compatible finding for a declared interface kind, meet the SX-030 software compatibility policy, and cover the task's required environment. A generic route to a provider is not enough.

The evaluator also checks pool and reservation evidence floors, provider revision, path reachability, environment scope and resource-kind identity. It rejects duplicate pool IDs, duplicate reservations, duplicate provider/resource pools, mismatched resource kinds, invalid or zero-length windows, invalid quantity intervals, unknown task/pool references and arithmetic overflow. It does not silently combine distinct supply pools.

Plan, report and receipt SHA-256 digests have distinct domain tags and deterministic ordering. Non-overbooked slices retain active-reservation count but omit the full ID list to keep normal-case output compact; overloaded slices include sorted active IDs for diagnosis. Receipt verification recomputes the SX-032 task plan and the reservation analysis from the exact inputs. The integration replay independently reimplements the canonical digest encodings. Digests are change evidence, not signatures, evidence authentication or qualification.

## Current synthetic tests

The unit module includes cases for:
- required plus optional overlap and optional-only overbooking;
- required overbooking after optional demand is removed;
- adjacent half-open windows, where end-at-tick-t does not overlap start-at-tick-t;
- conservative capacity minus reserve, including ceil rounding;
- empty windows and invalid quantity/capacity intervals;
- mismatched resource types;
- stale task receipts/configuration revisions;
- provider revision and environment mismatch;
- duplicate provider/resource pools and mutated reservation receipts;
- a deterministic 2,000-case overlap/boundary corpus that checks exact peak concurrent demand and resulting block/pass status;
- exact resource-interface-kind binding and missing provider-connection negative controls.

The integration replay includes positive, optional-overbooking and required-overbooking cases plus an independent plan/report/receipt digest oracle. All IDs, capacity values, demand numbers and evidence references are synthetic.

## Proposed acceptance targets

These are goals, not measured results. Timing thresholds should become CI gates only after freezing reference hardware, OS, Rust toolchain, build profile, deterministic seeds and corpus manifest.

| Benchmark | Initial target |
|---|---|
| Seeded required oversubscription | 100% detected |
| Optional-only oversubscription with required load under capacity | 100% degraded, not blocked |
| Non-overlapping reservations below capacity | 100% free from false overbooking |
| Boundary semantics | All start/end/equal-tick, invalid, zero-length and near-overflow cases covered |
| Input permutation | Identical report/receipt across 1,000 equivalent reservation/pool permutations |
| Generated overlap corpus | 2,000 seeded cases, exact expected overlap sets and zero unexpected panics |
| Scale target | 10,000 reservations / 100 pools p95 under 250 ms on documented runner |

For each 100% correctness result, record numerator, denominator, seed and corpus manifest. Preserve positive cases and known-negative controls; do not optimise away fault coverage for speed.

## Research basis

NASA's software engineering handbook guidance recommends quantitative margins for critical resources, robust responses to resource oversubscription, and timely measurement/visibility into resource use. Its oversubscription guidance discusses reducing non-essential demand to preserve critical capability. SX-034 adopts those as design motivation, not as a claim that this module is an operational scheduler or complies with NASA flight-software rules.

- NASA SWEHB 9.12, Resource Margins: https://swehb.nasa.gov/spaces/SWEHBVC/pages/85426336/9.12%2BResource%2BMargins
- NASA SWEHB 9.13, Resource Oversubscription: https://swehb.nasa.gov/spaces/SWEHBVD/pages/102695804/9.13%2BResource%2BOversubscription
- NASA SWEHB 9.14, Resource Usage Measurement: https://swehb.nasa.gov/spaces/SWEHBVC/pages/85426340/9.14%2BResource%2BUsage%2BMeasurement
- NASA Systems Engineering Handbook appendix: https://www.nasa.gov/reference/system-engineering-handbook-appendix/

## Deliberate exclusions

This is not an EDF/RM scheduler, resource broker, feedback controller, power-system model, thermal simulation, coolant simulation, RF propagation model, workload simulator or safety case. Capacity and occupancy envelopes are caller-declared uncertainty bounds; correlation, transient response, task migration, preemption and fail-safe response require separate models and evidence.
