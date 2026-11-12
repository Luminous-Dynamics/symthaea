# AIRROAD-POST-FAULT-SUPPLY-001 — V0.1 sustained electrical supply envelope

**Status:** executable integer power/energy reference and evidence composition contract  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11766  
**Fault/protection owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11645  
**Topology/load owner:** https://github.com/Luminous-Dynamics/symthaea/pull/11642

## 1. Decision boundary

This module answers one bounded question:

> Given an exact post-fault topology/path evidence set, lower-bounded non-storage source capacity, lower-bounded storage discharge power and usable storage energy, and upper-bounded time-indexed critical-load demand, can electrical supply be sustained for the entire declared profile?

It does **not** answer whether the aircraft can fly, retain adequate thrust, satisfy flight-control moments or remain safe. A sustained electrical supply result is not a mission/flight PASS.

## 2. Piecewise-constant hard-bound calculation

For a segment:

```
P_storage_required = max(0, P_load_upper - P_source_lower)
```

The instantaneous segment is electrically supportable only if:

```
P_storage_required <= P_storage_discharge_lower
```

Cumulative energy consumption is:

```
E_storage_required = sum(P_storage_required * duration_seconds)
```

All powers are integer watts and durations are integer seconds, so energy accounting is exact integer joules. The usable storage energy is one cumulative lower-bound budget shared by every segment. No charging, regeneration, recharge, source recovery or load shedding is assumed.

## 3. Evidence composition

The profile binds:
- aircraft configuration and operating-point digests;
- exact post-fault topology and fault-event digests;
- critical-load manifest digest and stable critical-load IDs;
- source-capability and storage-state digests;
- per-segment source/load/storage power bounds;
- confirmed available/unavailable/unknown post-fault path evidence for each critical load.

A confirmed unavailable path is a non-sustained result. An unknown path is NOT EVALUABLE. Missing or mismatched identities are rejected before calculation. Segment timelines must start at zero and be contiguous; segment and load ordering is canonical so result attribution is deterministic.

The implementation consumes evidence from #11645, #11640/#11642, #11663 and #11737. It does not create another topology, protection, source, storage or flight-control model.

## 4. Result semantics

- `SUSTAINED_ELECTRICAL_SUPPLY`: every declared load path is confirmed, every segment's storage power need is within its lower bound, and cumulative energy stays within the usable energy lower bound.
- `NOT_SUSTAINED_PATH`: a required load has a confirmed unavailable post-fault path.
- `NOT_EVALUABLE_PATH`: a required load's path is unknown.
- `NOT_SUSTAINED_POWER_LIMIT`: at least one segment's deficit exceeds the storage discharge-power lower bound.
- `NOT_SUSTAINED_ENERGY_LIMIT`: cumulative storage energy exceeds the usable energy lower bound.
- Profile identity, manifest, timeline or arithmetic errors produce typed validation errors, never PASS.

A separate aircraft control/mission-authority record must be joined from the appropriate flight/allocator owner before making any joint mission-capability statement. This module intentionally reports electrical supply only.

## 5. Frozen benchmark corpus

Machine-readable cases: `docs/research/fixtures/airroad-post-fault-supply-v0.1.json`. Independent integer/rational oracle: `tools/airroad-post-fault-supply-qualifier/verify_post_fault_supply.py`.

Frozen cases include:
1. Sustainable two-segment supply: 20 kW × 60 s + 5 kW × 120 s = 1.8 MJ, with 2.0 MJ available.
2. Reachable path but 5 kW instantaneous power shortfall.
3. Adequate instantaneous discharge power but energy exhaustion after 50 s of a 360 s segment.
4. Confirmed unavailable path.
5. Unknown path that cannot become PASS.
6. Aggregation of two critical-load demands.
7. Profile identity, load-manifest, duplicate-ID and non-contiguous-timeline negative controls.

## 6. Industry/SOTA comparison

Tang et al. (2026) demonstrate multi-machine DC-side cascade mechanisms and HIL evidence on two- and three-machine grid-forming converter clusters. Their work is a microgrid baseline, not an aircraft validation result. [DOI 10.1016/j.epsr.2026.113503](https://doi.org/10.1016/j.epsr.2026.113503).

Wu et al. (30 October 2026) review more-electric-aircraft system integration and explicitly distinguish path reachability from sustained critical-load supply after protection and reconfiguration. They call for joint validation of interruption, bus voltage, critical-load supply duration, source capacity and storage limits. [DOI 10.3390/aerospace13111003](https://doi.org/10.3390/aerospace13111003).

NASA's 2025 piloted DEP fault-recovery study evaluates multiple fan failures in a flight simulator and coordinates health management, fault detection, isolation and accommodation. It is a fault-recovery/flight-simulator baseline, not electrical PHIL evidence. [NASA/TM-20240013567](https://ntrs.nasa.gov/citations/20240013567).

NASA AREAL supplies a 200 kW, nominal 600–700 VDC, reconfigurable electrified-aircraft testbed for nominal and faulted architecture studies. Luminous should pursue a verified low-energy reference and qualified partner tests, not claim to replace such facilities. [NASA AREAL](https://www.nasa.gov/eap-labs-and-testbeds/areal/).

## 7. Targets and claim ceiling

Research targets:
- 100% exact integer agreement across the frozen corpus.
- Zero sustained-supply outputs on path loss, power shortfall or energy depletion.
- Zero path-reachability-only PASS outcomes.
- Zero implicit source-recovery or charging credit.
- Deterministic, digest-bound replay and exact segment-level deficit reporting.

These checks validate only piecewise-constant power/energy accounting over the declared hard-bound inputs. They do not calculate fault current, protection clearing, dynamic bus stability, battery chemistry, thermal behavior, rotor thrust, control authority, safe flight, airworthiness or certification. No energized fault-injection testing is authorized.
