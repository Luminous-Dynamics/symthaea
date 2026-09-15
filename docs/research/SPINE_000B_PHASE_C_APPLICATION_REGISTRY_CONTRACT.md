# SPINE-000B — Phase-C Application Registry Contract v1

**Status:** preregistered measurement-only registry; runtime instrumentation pending

**Authority:** measurement-only

**Issue:** #3261

The Phase-C application registry freezes how the current integrated `SubsystemOutput` is consumed by production code. It is an observational map, not a causal attribution table.

## Core laws

1. Every current scalar integration channel appears exactly once in `scalar_sources`.
2. Every defined `output_flags::*` bit appears exactly once in `flag_sources`.
3. Every flag is explicitly `CONSUMED`, `FEATURE_GATED_CONSUMED`, or `UNCONSUMED_IN_PHASE_C`.
4. Every production operation has a stable `operation_id` and `destination_id`.
5. `applied_argument` describes the exact operand that crosses the production operation boundary after casts/constants are resolved.
6. Before/after evidence observes the live destination around the real operation. It is never recomputed from the source value.
7. One source may fan out into multiple applications. Runtime `application_index` follows actual production execution order and is assigned only to operations that execute.
8. Flag polarity is explicit: `FLAG_SET` and `FLAG_CLEAR` are distinct canonical source conditions.
9. Complex/external effects may be `NOT_OBSERVED_AT_BOUNDARY`; do not invent scalar proxies.
10. An application receipt remains cycle-level evidence. It must not be attached to one manager as a causal consequence merely because that manager had leave-one-out integration influence.

## Scalar cast boundary

Current Phase C casts integrated `f64` confidence, learning-rate, and exploration values to `f32` before their feedback-helper calls. Their registry entries therefore require `F32_BITS` applied operands even though the upstream integrated channels are `f64`.

The helpers then update feedback consensus and synchronize `f64` destinations. Runtime instrumentation must observe the live destination bits before and after the helper call.

## Flag absence can be an application

Under `vision-manifold`, Phase C explicitly clears `carryover.quality.last_request_geodesic` when `REQUEST_GEODESIC` is absent while subsystem integration is active. The registry therefore contains a `FLAG_CLEAR` application for this bit. Runtime evidence must not assume all flag applications are positive-edge (`FLAG_SET`) effects.

## Unconsumed flags

`HAS_TELEMETRY` is produced by managers but currently has no Phase-C `integrated.has_flag(output_flags::HAS_TELEMETRY)` consumer. It is frozen as `UNCONSUMED_IN_PHASE_C`. If production later gains a consumer, this registry must fail and begin a new lineage.

## Feature-gated effects

`REQUEST_GEODESIC` is gated by `vision-manifold`. `REQUEST_BROADCAST` is gated by `all(swarm,vision-manifold)` in current Phase C. Disabled features produce no canonical application receipt for code that does not exist in that subject.

## External and complex effects

`REQUEST_BROADCAST` constructs a payload containing a wall-clock timestamp. Wall-clock data is excluded from canonical SPINE receipt identity. The registry records the boundary operation as `NOT_OBSERVED_AT_BOUNDARY` without inventing a canonical payload proxy.

Geodesic simulation and mental-movie construction are likewise represented as named boundary operations; richer semantic evidence for their internal products requires separate qualification.

## Qualification boundary

A registry PASS can establish only:

> For the exact subject, every current `SubsystemOutput` scalar/flag source has a frozen Phase-C consumption classification and the named consumed operations are source-consistent with production code.

It does not establish that an operation ran on an empirical cycle, that a particular subsystem caused the downstream state change, or that the state change was beneficial.
