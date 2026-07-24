# Subterranean Mission Autonomy Protocol

## 1. Scope

This protocol defines how long-duration work becomes permitted mission intent.
It does not define transport authentication, cryptographic identity, low-level
motor control, or the physical correctness of a real vehicle.

## 2. Topology truth

Mission planning uses a bounded tunnel graph. Every passage has explicit
traversability, revision, risk, energy, and confidence. Depth is never treated
as proof of connectivity.

A stale edge revision cannot reopen a passage. A blocked edge cannot appear in
a route. Equal-cost route ties are deterministic.

## 3. Work-order truth

A work order identifies one bounded unit of work, its destination, priority,
deadline, prerequisites, resource estimate, and progress.

- Missing prerequisites reject submission.
- Incomplete prerequisites block activation.
- Preemption suspends; it does not complete.
- Work progresses only on nominal, hazard-free, Green-tier frames.

## 4. Admission truth

Before work receives mission authority, the logistics planner must fund:

1. travel to the target,
2. the work itself,
3. travel back to the surface,
4. contingency,
5. protected battery reserve,
6. every finite consumable required by the work.

Admission is re-evaluated while work is active. A shrinking battery or resource
margin can therefore abort a previously valid job.

## 5. Maintenance truth

Component health persists across work orders and checkpoints. Health is updated
from the command that actually reached the plant, not the command originally
requested by cognition.

Failed hardware is removed from command authority. Unrelated healthy hardware
is preserved where doing so remains safe. Maintenance consumes finite spares
and lubricant.

## 6. Preemption order

Physical hazards and malformed sensing preempt work immediately. Team
right-of-way suspends work. Critical maintenance returns to base when mobility
and cooling remain available; otherwise the platform holds position and
reports immobilization.

Mission preemption never suppresses the independent hazard-specific recovery
planner.

## 7. Restart continuity

The operational checkpoint binds learned control and mission state under one
schema version. Loading is fail-closed:

- both halves are validated before activation,
- graph endpoints and capacities are checked,
- scheduler invariants are checked,
- resource and health values must be finite and bounded,
- current and surface nodes must exist.

Checkpoint persistence is not a cryptographic authenticity claim. Signed and
rollback-resistant storage belongs at the deployment boundary.

## 8. Evidence

Operational evidence records actual command, physical state, safety decision,
mission admission, route envelope, work state, logistics, and health. Final
mission success may not be inferred from a mission enum alone.

## 9. Explicit non-claims

Campaign VI does not claim:

- optimal mine planning,
- certified geotechnical engineering,
- verified vehicle dynamics,
- authenticated maps or work orders,
- real-time certification on target hardware,
- autonomous legal authority to excavate or rescue.
