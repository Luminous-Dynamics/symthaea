# Symthaea Subterranean Capability Campaign VI

**Date:** 2026-07-20  
**Theme:** Mission-scale autonomy, logistics truth, persistent maintenance, and restart continuity

## Purpose

Campaigns I–V closed the immediate control, safety, simulator, recovery,
geology, sensing, return-assurance, and multi-agent truth gaps. Campaign VI
addresses the next temporal scale: a platform that remains underground long
enough to have multiple destinations, dependencies, consumables, accumulated
damage, incomplete work, and process restarts.

The campaign's central rule is:

> A requested job is not mission authority until its route, resources,
> contingency, return reserve, hardware state, and prerequisites have all been
> admitted independently of the learned controller.

The mission executive selects what may be attempted. It does not replace the
hazard supervisor, team right-of-way, recovery planner, moral gate, or physical
command arbiter.

## Patch sets 40–50

### 40. Bounded tunnel graph and conservative routing

Adds explicit surface, junction, workface, relay-bay, refuge, and service-bay
nodes. Passage edges carry length, energy, obstruction, water, roof,
confidence, directionality, traversability, and monotonic revision.

Routing uses a deterministic bounded O(V²) search. Conservative route cost can
prefer a longer passage over a short high-risk passage. Blocked passages are
never routed, and a stale revision cannot silently reopen one.

### 41. Dependency-aware work scheduler

Adds bounded work orders for survey, boring, roof stabilization, dewatering,
relay deployment, sample extraction, spoil clearing, return, and maintenance.

Orders have priorities, deadlines, prerequisites, estimates, explicit status,
and progress. Dependencies must complete before dependent work can start.
Safety preemption suspends active work rather than pretending it completed.

### 42. Mission logistics admission

Adds a separate logistics gate covering:

- outbound and return energy,
- work energy,
- contingency,
- protected battery reserve,
- sealant,
- relays,
- roof supports,
- sample capacity,
- spoil capacity,
- coolant availability.

A geometrically reachable workface can still be refused. Admission reports the
complete resource envelope and a structured refusal reason.

### 43. Persistent maintenance and actuator derating

Adds health for cutter, auger, tracks, thermal pump, dewatering pump, pressure
seal, and communications hardware. Damage accumulates from the actual command
and measured plant load.

A failed component loses only the authority it cannot physically provide. For
example, a failed cutter is held at zero while healthy tracks may retain return
mobility. Repair consumes finite spare parts and lubricant.

### 44. Long-horizon mission executive

Composes topology, scheduling, logistics, and maintenance into an inspectable
mission-level assessment. The executive can:

- execute an admitted work order,
- return to base,
- hold position,
- yield to physical safety,
- refuse work for route/resource reasons.

It cannot issue raw actuator commands.

### 45. Runtime embodiment integration

Admitted work now changes the controller's effective mission context in live
deployment. Hazards and team right-of-way suspend work. Recovery commands are
then limited by actual component health before reaching the plant.

Only hazard-free, Green-tier, nominal-fallback frames advance work progress.
Actual post-arbitration commands accumulate hardware wear.

### 46. Operational checkpoint

Adds a combined versioned checkpoint containing:

- learned controller projection,
- tunnel topology,
- work queue and progress,
- logistics ledger,
- component health and maintenance resources,
- current and surface graph position,
- route-cost policy.

Both controller and mission state are validated before activation. Unsupported
schemas, malformed graphs, impossible scheduler states, invalid resources, and
unknown positions are rejected.

### 47. Mission-scale operational evidence

Every retained control frame can now include:

- executive directive,
- active and queued work,
- completion/failure counts,
- admission result and refusal reason,
- outbound and return route evidence,
- route risk and confidence,
- required and residual battery,
- minimum component health,
- maintenance and abort state,
- payload, spoil, and coolant state.

Summary evidence counts active-work frames, admission refusals, maintenance-due
frames, mission-abort frames, completed work, and minimum observed health.

### 48. Long-horizon acceptance contracts

Adds deterministic cross-module gates for:

1. safer-route preference,
2. underfunded-work rejection,
3. mid-work reserve abort,
4. failed-hardware authority removal,
5. checkpoint continuity.

### 49. Mission autonomy and authority protocol

Documents authority order, admission, progress, preemption, restart semantics,
and claim boundaries.

### 50. Mechanical formatting normalization

Applies the repository formatter after the functional commits so review can
separate behavior from formatting.

## Authority order

Campaign VI preserves the existing order of authority:

1. malformed observation and physical hazard assessment,
2. moral/consciousness/manual safety caps,
3. team right-of-way and accepted rescue state,
4. return-reserve and geological caution,
5. mission-work admission,
6. learned nominal policy,
7. verified recovery planning,
8. mechanical-health derating,
9. simulator/plant.

A later layer may narrow an earlier layer's requested command. It may not grant
authority that an earlier layer denied.

## Validation status

Offline validation uses API-compatible stand-ins because this standalone
archive omits the real workspace path dependencies. This verifies Rust types,
exhaustive matches, deterministic domain behavior, and the complete crate test
suite. The authoritative Rust 1.94 workspace build, real HDC/FEP integration,
Clippy, serialization, and controlled-hardware timing benchmark remain required
before merge.
