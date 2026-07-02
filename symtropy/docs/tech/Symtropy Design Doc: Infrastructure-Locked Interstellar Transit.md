# Symtropy Design Doc: Infrastructure-Locked Interstellar Transit

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Working Title

**No Casual FTL**

## Core Thesis

*Symtropy* should reject casual ship-mounted faster-than-light travel.

Interstellar movement must not be a personal vehicle upgrade, a convenience button, or an arcade warp jump.

Distance is part of the game’s moral and mechanical architecture.

If distance becomes trivial, then these systems lose weight:

```text
regional scarcity
communication latency
local law
settlement dependency
rescue windows
resource chokepoints
faction borders
transmission delay
worldline isolation
infrastructure sovereignty
```

The design rule:

```text
No ship carries civilization casually between stars.
Civilization builds, governs, powers, witnesses, and risks the crossing.
```

Interstellar transit in *Symtropy* is not a movement ability.

It is a macro-infrastructure transaction.

---

# 1. Transit Layers

*Symtropy* should separate travel into three layers.

```text
1. In-system sub-light travel
2. Atlas Gate interstellar transit
3. Timeline / worldline translation
```

Each layer has different costs, risks, and political consequences.

---

# 2. Layer One: In-System Sub-Light Travel

In-system travel remains physical.

Ships, rovers, landers, ferries, drones, and orbital craft move using conventional propulsion.

They require:

```text
fuel
thrust
life support
thermal control
battery margin
repair capacity
navigation windows
orbital timing
radiation shielding
crew endurance
```

There is no emergency warp button.

If a vessel fails mid-transit, the player must solve a physical problem:

```text
patch a coolant leak
reroute battery draw
repair a firmware lock
stabilize oxygen cycling
manually align antenna gain
restart a navigation computer
fix a thruster gimbal fault
```

Design rule:

```text
Sub-light travel preserves the frontier.
```

In-system travel should feel like survival inside infrastructure.

---

# 3. Layer Two: Atlas Gates

Atlas Gates are massive interstellar transit infrastructures.

They are not ship components.

They are faction-controlled, energy-intensive, legally governed, and historically dangerous.

An Atlas Gate requires:

```text
orbital access
regional power routing
gate authority permission
navigation target validation
Archive Witness handshake
shard synchronization
transit budget allocation
hazard review
Chronicle logging
```

The player does not “jump.”

The player petitions, powers, verifies, and commits a transit event.

Design rule:

```text
An interstellar jump is a public infrastructure act.
```

---

# 4. Atlas Gate Transaction Flow

A gate transit should behave like a volatile Device Bus transaction.

Example interaction:

```sh
$ write /dev/sym/gates/atlas_4/jump_target "Vesta Forge"

DENIED:
SEGMENT_UNSYNCHRONIZED.
Legitimacy token expired.
Archive Witness handshake required.
```

The player requests witness validation:

```sh
$ request-witness --target gate_authority

WITNESS_CONNECTED.
VALIDATING TARGET HISTORY...
CHECKING POWER VECTOR...
CHECKING SHARD CONTINUITY...
```

Then commits:

```sh
$ sym-gate commit /dev/sym/gates/atlas_4

POWER VECTOR STAGED.
TRANSIT WINDOW OPEN.
JUMP COMMITTED.
```

This should not feel like menu fast travel.

It should feel like operating dangerous public machinery.

---

# 5. Gate Authority and Politics

Atlas Gates create choke points.

Factions can control:

```text
access
pricing
route priority
transit permits
refugee movement
military quarantine
cargo inspection
archive logging
witness requirements
destination censorship
```

Possible gate-holding factions:

```text
Helion Directorate
Starward Mandate
Lumen Archive
Belt Rescue Compact
Utility Sovereigns
Continuance Office
Quarantine Authority
```

Each faction should treat gate access differently.

## Helion Directorate

```text
high order
strict permits
energy rationing
military priority
heavy inspection
```

## Lumen Archive

```text
witness-first transit
history-preserving
slow but trusted
requires source-chain integrity
```

## Utility Sovereign

```text
fast commercial access
subscription permissions
opaque routing costs
possible debt capture
```

## Quarantine Authority

```text
route lockdowns
biosecurity scans
hard denial under uncertainty
forced isolation corridors
```

Design rule:

```text
Who controls the gate controls the meaning of distance.
```

---

# 6. Gate Failure Modes

Atlas Gates should be powerful but dangerous.

Failure modes:

```text
segment unsynchronized
power vector collapse
witness handshake failure
destination seed mismatch
route legitimacy revoked
Null Reinforcement Loop
Archive corruption
partial shard split
transit delay
false destination attestation
quarantine reroute
```

## Null-Infected Gate

A Null-infected gate may appear to execute correctly.

But the destination may be wrong.

Possible outcomes:

```text
High-Entropy Fusion Zone
unmapped quarantine timeline
dead version of target system
collapsed archive mirror
Null Bloom-consumed destination
fragmented arrival shard
false civic authority overlay
```

Field Deck reading:

```text
NULL:
Gate accepted transit command.
Destination witness chain incomplete.
Transit route contains recursive authority echo.
```

Design rule:

```text
The most dangerous gate is not the one that refuses.
It is the one that politely agrees.
```

---

# 7. Arrival Knowledge Rule

A successful Atlas Gate transit should not give the player a complete destination map.

The gate provides an authenticated arrival envelope.

It does not provide verified reality.

## Arrival Envelope

The Field Deck receives:

```text
destination identity
gate signature
arrival vector
known orbital body
old archive map fragments
hazard advisories
route legitimacy proof
last confirmed witness timestamp
```

Example:

```json
{
  "arrival_envelope": {
    "destination": "Vesta Forge",
    "gate_authority": "atlas_4",
    "witness_status": "partial",
    "last_verified": "2171-09-14T03:22:00Z",
    "archive_map_age": "18 years",
    "hazard_flags": ["industrial_ruin", "radiation_uncertain", "labor_conflict_legacy"],
    "topography_status": "unverified"
  }
}
```

The Field Deck should clearly distinguish:

```text
known
archived
inferred
unverified
contradicted
Null-suspect
```

Design rule:

```text
Arrival proves where you are.
It does not prove what is still true.
```

---

# 8. Scout-Based Sector Compilation

After arrival, the player must build a fresh topographical directory.

Methods:

```text
long-range scout drone pings
orbital lidar sweep
ground rover survey
manual tower triangulation
radio beacon deployment
local witness interviews
old archive comparison
environmental sampling
Field Deck scan routes
```

Directory status should improve over time:

```text
NO MAP
ARCHIVE MAP
ROUGH TOPOLOGY
UNVERIFIED DIRECTORY
PARTIAL VERIFIED MAP
ATTESTED LOCAL MAP
```

Example Field Deck output:

```text
SCAN:
Archive map loaded.

DIAG:
Terrain mismatch detected.
Old bridge absent.
New thermal plume detected.

ARCHIVE:
Last verified survey: 18 years ago.

NULL:
No active spoofing detected, but source chain incomplete.
```

Design rule:

```text
A map is a claim, not the world.
```

---

# 9. Timeline Translation vs Space Flight

*Symtropy* can reduce dependence on conventional FTL by using worldline or timeline translation.

Instead of only moving horizontally across space, players may move vertically through histories.

A worldline terminal may allow access to:

```text
parallel ruins
alternate civilizational outcomes
nearby but historically divergent settlements
quarantine branches
failed restoration timelines
unburned archives
Null-consumed variants
```

This supports exploration without trivializing physical distance.

Design rule:

```text
Exploration is not only going far.
It is learning which version of near survived.
```

---

# 10. Transit as Chronicle Event

Major transit should always be recorded.

Example Chronicle event:

```json
{
  "event_type": "AtlasGateTransit",
  "origin": "Hearth System",
  "destination": "Vesta Forge",
  "gate": "/dev/sym/gates/atlas_4",
  "operator": "player",
  "witness_status": "partial",
  "power_budget": "regional_solar_reroute",
  "civic_authority": "Helion Directorate transit permit",
  "arrival_status": "successful_unverified",
  "map_status": "archive_only",
  "chronicle_line": "The player crossed the gate and learned that arrival is not the same as knowing where they stood."
}
```

Transit is not just movement.

It is historical continuity under stress.

---

# 11. Seedworks Implication

Seedworks v0.1 should not implement interstellar transit.

But it should foreshadow the doctrine.

Possible foreshadowing elements:

```text
broken miniature gate schematic
old Atlas Gate permit
Field Deck archive entry
distant gate-control faction reference
unverified arrival log
failed transit Chronicle fragment
Null-corrupted destination warning
```

Example archive fragment:

```text
ARCHIVE:
Atlas Gate transit records incomplete.
Destination acknowledged receipt.
No second witness returned.
```

This lets the player understand that distance, witness, and verification matter long before they ever cross a star.

---

# 12. Design Boundaries

## Allowed

```text
Atlas Gates
orbital rail gates
energy-intensive transit
worldline translation
timeline confluence
scout-based mapping
arrival uncertainty
faction-controlled routes
quarantine branches
Null-corrupted transits
```

## Forbidden

```text
casual ship-mounted FTL
personal hyperdrives
instant galaxy hopping
warp drives on scrap rovers
FTL as ordinary cargo upgrade
full destination map after jump
riskless transit cutscenes
```

---

# 13. Final Principles

```text
Distance is gameplay.
Transit is governance.
Maps are claims.
Arrival requires verification.
```

Final line:

```text
The gate does not make the stars small.
It makes distance political.
```
