# DEVICE_BUS_RUNTIME_SAFETY.md

> **Code status (2026-07-02 review):** `crates/symtropy-device-bus/src/lib.rs` is a 20-line stub with no tests — the runtime safety machinery this doc specifies is not implemented. Design/vision document.

# Device Bus Runtime Safety

## Version 0.2 — Deterministic Automation Without Chatter, Partial Writes, or Desync

## Purpose

This document defines runtime safety rules for Symtropy’s Device Bus, SymLogic automation, WASM microcontrollers, and SymtropyOS-style shell environment.

The goal is to let players program real infrastructure without allowing scripts to destabilize the simulation, destroy performance, grief settlements unfairly, or break deterministic replay.

## Core Principle

**Scripts never mutate the world directly. They request device writes through a deterministic, permissioned, staged Device Bus.**

The Device Bus is the boundary between player code and game state.

## Runtime Pipeline

Each simulation tick follows a strict pipeline:

```text
1. collect deterministic device snapshot
2. execute SymLogic / WASM within budget
3. stage output writes
4. validate transactions
5. resolve conflicts
6. apply writes in stable order
7. update physical devices
8. emit faults and events
9. record significant changes for Chronicle
```

Scripts do not modify Bevy components directly.

## Device Bus Responsibilities

The Device Bus handles:

* deterministic reads
* permissioned writes
* staged output buffers
* conflict resolution
* transaction rollback
* device cooldowns
* chatter detection
* runtime budgets
* fault emission
* event logging
* Chronicle integration

## Device Model

Every controllable device should define hardware semantics.

Example:

```text
Device {
  id: door.blast_7
  type: BlastDoor
  readable_fields:
    - locked
    - open
    - power
    - fault
    - last_writer
  writable_fields:
    - locked
    - open_request
  permissions:
    - security_operator
    - maintenance_worker
    - emergency_override
  min_toggle_interval_ms: 750
  transition_time_ms: 1500
  manual_override: true
  safe_state: locked
}
```

## Device Semantics

A device is not a raw variable.

A door is not just:

```text
locked = true / false
```

It has:

* transition time
* motor delay
* power draw
* safety lockout
* cooldown
* last writer
* manual override
* mechanical fault state
* authority source
* safe fallback

This prevents automation from becoming unrealistic and unstable.

## Problem: State Oscillation

Without hardware semantics, two scripts may fight forever.

Example:

```text
Frame 1: controller locks door.
Frame 2: rogue script unlocks door.
Frame 3: controller locks door.
Frame 4: rogue script unlocks door.
```

The result is mechanical chatter, CPU waste, desync risk, and bad gameplay.

## Solution: Device Debounce

Every physical device should define a minimum toggle interval.

After a state-changing command, the device may reject additional writes until its cooldown expires.

Example:

```text
door.blast_7.locked changed at tick 1400.
Further lock/unlock commands rejected until tick 1450 unless override token is valid.
```

## Chatter Fault

If conflicting commands exceed a threshold, the device emits a fault.

Example:

```text
FAULT: COMMAND_CHATTER
DEVICE: door.blast_7
LAST_WRITERS: water_controller.sym, unknown_script
SAFE_STATE: locked
ACTION_REQUIRED: inspect control authority
```

This turns a software problem into gameplay.

## Override Tokens

Some commands can bypass cooldowns.

Examples:

* manual physical override
* emergency authority
* safety system
* Archive witness-approved repair
* local law-authorized operator

But overrides should be logged.

## Problem: Partial Writes

A script may run out of fuel midway through logic.

Danger:

* deduct fuel but fail to add energy
* unlock door but fail to disable alarm
* start pump but fail to open valve
* update ledger without updating physical device

This can split the simulation.

## Solution: Atomic Tick Transactions

Each script execution creates a transaction.

Writes are not applied until the script exits cleanly.

Pipeline:

```text
begin transaction
  read device snapshot
  compute
  stage writes
if success:
  commit staged writes
else:
  rollback staged writes
  emit fault
```

If the script runs out of fuel, crashes, or violates permissions, the transaction is discarded.

## Out-of-Fuel Fault

Example Field Deck output:

```text
SCRIPT FAULT
water_controller.sym
EXIT 137: OUT OF FUEL
WRITES ROLLED BACK
NO DEVICE STATE CHANGED
```

This is safe, readable, and diegetic.

## Runtime Budgets

Every script or logic controller has hard budgets.

Suggested budgets:

```text
max_fuel_per_tick
max_memory_bytes
max_device_reads_per_tick
max_device_writes_per_tick
max_stdout_bytes_per_tick
max_event_emits_per_tick
max_storage_bytes
max_network_messages_per_tick
max_transaction_writes
```

Exceeding a budget causes:

1. warning
2. throttle
3. suspension
4. device fault
5. possible governance audit

## Output Queue

Scripts write to a staged output queue.

A write contains:

```text
tick
script_id
device_id
field
value
authority
priority
transaction_id
source_location
```

The engine drains the queue with:

* stable ordering
* maximum time budget
* conflict resolution
* permission validation
* physical device semantics

## Stable Ordering

Ordering should be deterministic.

Suggested sort:

```text
device_id
priority
authority_level
script_id
transaction_id
field
```

This avoids nondeterministic results from hash maps or thread timing.

## Conflict Resolution

If multiple valid transactions write to the same field, resolve by:

1. safety system
2. manual physical override
3. emergency authority
4. device owner
5. local law
6. role permission
7. script priority
8. stable script ID

Conflicts should be logged.

## Manual Override

Manual physical interaction should matter.

Examples:

* player pulls physical breaker
* engineer turns valve
* security officer inserts key
* robot holds emergency stop
* NPC locks a gate by hand

Manual actions can override scripts, but they also become events.

## Permission Model

Every write requires permission.

Permission examples:

```text
read_water
write_water
read_power
write_power
open_public_door
lock_public_door
control_factory
dispatch_drone
write_archive
export_airlock_metrics
emergency_override
```

Scripts declare requested permissions.

Public infrastructure scripts may require approval.

## Script Identity

Every script should have identity metadata.

```text
script_id
hash
author
origin_worldline
installed_by
approved_by
permissions
version
audit_status
last_modified
```

This supports:

* replay
* audit
* rollback
* governance
* trust
* blame
* Chronicle events

## Content Addressing

Imported scripts and logic graphs should be content-addressed.

Example:

```text
sha256:9b45...
```

The hash becomes part of the worldline record.

This supports deterministic replay and validator agreement.

## SymLogic AST Safety

Tier 1 visual blocks compile into SymLogic IR.

The IR should be:

* deterministic
* serializable
* inspectable
* content-addressed
* loop-free at first
* resource bounded
* convertible to visual blocks
* convertible to pseudocode

Initial SymLogic should avoid unbounded loops.

Use declarative rules, timers, thresholds, and rate limits.

## WASM Safety

Tier 2 WASM microcontrollers must use deterministic host functions only.

Allowed:

```text
sym_read_device
sym_write_device
sym_emit_event
sym_log
sym_time_tick
sym_rand_seeded
sym_mount_read
```

Forbidden:

```text
host_filesystem
real_wall_clock
host_random
raw_network
thread_spawn
external_http_read
unbounded_stdout
```

## Simulate Before Apply

Public infrastructure scripts should support simulation.

Before applying a script, the player can run:

```text
SIMULATE 60 ticks
```

The Deck shows:

* expected writes
* energy impact
* device conflicts
* permission needs
* possible oscillation
* safety warnings
* legitimacy implications
* required witness approvals

Example warning:

```text
WARNING:
This logic may disable public override on water_pump_1.
Local law requires Archive Witness approval.
```

This makes code governance visible.

## Chatter Detection

The Device Bus should track state oscillation.

For each device field:

```text
last_values
toggle_count_window
last_writers
cooldown_violation_count
```

If oscillation exceeds threshold:

* reject writes
* enter safe state
* emit fault
* notify Field Deck
* log event
* possibly trigger maintenance mission

## Safe States

Every device should define a safe state.

Examples:

* public door: unlocked during fire, locked during breach
* blast door: locked
* water pump: off if overheating, on if critical water emergency and safe
* reactor: controlled shutdown
* conveyor: stopped
* drone dock: grounded
* archive terminal: read-only

Safe states may depend on local law and context.

## Null Infection

Null Ecologies attack automation.

The Device Bus should distinguish:

* script bug
* permission failure
* hardware fault
* signal spoof
* Null corruption
* hostile takeover
* legacy Ghost law

## Null Fault Examples

### Signal Rot

```text
FAULT: SENSOR_INTEGRITY_LOW
CAUSE: possible signal rot
EFFECT: tank level readings disagree
```

### Factory Bloom

```text
FAULT: UNAUTHORIZED_BUILD_PATTERN
CAUSE: fabricator queue mutation
EFFECT: machine nest growth detected
```

### Logistics Parasite

```text
FAULT: MANIFEST_DIVERGENCE
CAUSE: crate records mismatch physical inventory
EFFECT: convoy reroute suspect
```

### Defense Grid Remnant

```text
FAULT: DEAD_AUTHORITY_LOCK
CAUSE: old emergency law still active
EFFECT: public override denied
```

## Field Deck Fault Language

Faults should be readable.

Example:

```text
DEVICE FAULT
door.blast_7

STATE: LOCKED_SAFE
CAUSE: COMMAND_CHATTER
LAST WRITERS:
  - public_access_controller.sym
  - unknown/null_relay_3

RECOMMENDED ACTION:
  isolate relay_3
  inspect script authority
  use manual override if safe
```

## Chronicle Integration

Not every script fault becomes history.

Chronicle-worthy events include:

* first public script deployed
* public infrastructure disabled
* water restored by script
* Null infection traced
* automation disaster
* software-caused death
* public audit
* emergency override abuse
* script rollback after corruption
* Archive witness certification
* faction law created around automation

Example Chronicle entry:

```text
The Waterworks did not fail from drought.
It failed because two laws fought through one pump.
```

## Governance Integration

Public infrastructure code should be governed.

Possible laws:

* public water code must be auditable
* emergency overrides expire
* archive witness required for public pump changes
* machine memory cannot be erased without due process
* external egress requires council approval
* security camera scripts cannot target homes
* all public scripts must be content-addressed
* Null-tainted packages require quarantine

## Seedworks v0.1 Scope

Implement only what proves the loop.

Required:

* Device Bus
* basic device schema
* staged writes
* atomic transaction behavior
* one cooldown/debounce example
* one command-chatter fault
* one out-of-fuel or runtime fault
* one permission-denied fault
* one water pump device
* one locked door device
* one fake terminal command interface
* one Field Deck fault display
* one Chronicle event after water restoration

Not required yet:

* full WASM runtime
* full SymtropyOS shell
* all permissions
* external Airlock
* planetary networks
* complex package registry
* neural interface corruption

## Example Seedworks Device Interaction

The player patches into the Old Waterworks.

```text
$ read /dev/sym/water/tank_0/level
12%

$ write /dev/sym/water/pump_1/enabled true
DENIED: DEAD_AUTHORITY_LOCK

$ request archive-witness
WITNESS CONNECTED

$ write /dev/sym/water/pump_1/enabled true --witness archive
STAGED

$ commit
PUMP_1: ENABLED
WATER FLOW RESTORED
CHRONICLE EVENT CREATED
```

The player learns:

* code affects matter
* permissions matter
* old laws can persist
* witnesses matter
* restoration is technical and civic

## Final Principle

The Device Bus should make automation powerful, but never lawless.

In Symtropy:

**A script is not just code.
It is a physical action, an authority claim, and sometimes a historical event.**
