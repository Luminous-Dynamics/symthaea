# WASM_FIRST_AUTOMATION_ARCHITECTURE.md

# WASM-First Automation Architecture

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Version 0.1 — Shippable In-World Computing

## Purpose

This document defines the practical implementation path for Symtropy’s in-world automation and SymtropyOS-style computing.

The goal is to preserve the fantasy of real terminals and programmable infrastructure while avoiding the complexity of making a full Linux emulator the foundation of gameplay.

The recommended path is:

**Visual logic first. Deterministic AST second. WASM microcontrollers third. SymtropyOS shell fourth. RISC-V/Linux emulation only as optional prestige content later.**

## Core Recommendation

Symtropy should be **WASM-first**, not RISC-V-first.

A RISC-V emulator is exciting and may be useful for special Ghost Civilization terminals, high-end mainframes, or modded prestige systems.

But the shippable core should use:

* deterministic visual logic
* constrained AST/IR evaluation
* WASM-backed microcontrollers
* custom virtual filesystem
* `/dev/sym/*` device bus
* OS-like shell layer
* content-addressed packages

This gives players the feeling of real computing without forcing early development into emulator, kernel, and rootfs complexity.

## Why WASM-First

WASM-first provides:

* strong sandboxing
* high performance
* deterministic control
* memory limits
* fuel/cycle budgeting
* language flexibility
* easier script packaging
* easier validation
* better fit for Bevy integration
* easier cross-platform support

The goal is not to simulate a real consumer operating system.

The goal is to make computation inside Symtropy feel real, useful, physical, and replayable.

## Automation Tiers

## Tier 0 — Physical Logic

No code.

Examples:

* switches
* breakers
* fuses
* levers
* valves
* mechanical relays
* pressure plates
* timers

Use cases:

* doors
* lights
* basic pump control
* simple alarms
* emergency shutoffs

This is for players who do not want programming.

## Tier 1 — SymLogic Blocks

Visual automation.

Players build simple logic using blocks or node graphs.

Example:

```text
IF tank_0.level < 35%
AND grid_0.power_available
THEN pump_1.enabled = true
ELSE pump_1.enabled = false
```

Internally, this compiles to deterministic AST/IR.

## Tier 2 — WASM Microcontrollers

Scriptable devices.

Used for:

* conveyor sorting
* greenhouse control
* local drone docks
* turret filters
* repair bots
* factory cells
* water controllers
* access panels

Scripts run under strict resource limits.

## Tier 3 — SymtropyOS Shell

A terminal environment that feels like an operating system.

It exposes:

* filesystem
* logs
* shell commands
* mounted blueprints
* packages
* `/dev/sym/*` devices
* local services

It may be backed by WASM services rather than a full Linux kernel.

## Tier 4 — Mainframe Clusters

Settlement-scale systems.

Used for:

* public water
* power grid
* archive
* logistics
* faction database
* voting terminals
* drone fleet coordination
* factory planning

## Tier 5 — MycelixNet / Airlock Gateways

Planetary and worldline-scale computing.

Used for:

* settlement ledgers
* archive replication
* faction treaties
* worldline records
* external observability
* signed metrics export

## Visual Logic Compilation

Visual logic should not generate raw text code as its primary representation.

It should compile to a deterministic intermediate representation.

Pipeline:

```text
Visual Blocks
  ↓
SymLogic IR / AST
  ↓
AST Interpreter
  ↓
Optional WASM Compilation
  ↓
Device Bus Writes
```

This keeps the visual system stable even if the scripting backend changes.

## Why AST First

AST-first allows:

* deterministic execution
* easy validation
* readable debugging
* safe permission analysis
* visual-to-text conversion
* text-to-visual conversion later
* no hidden syntax errors
* simpler replay
* easier AI/NPC inspection

Example AST:

```text
Rule {
  when: LessThan(Read("water.tank_0.level"), Const(0.35)),
  then: Write("water.pump_1.enabled", true),
  otherwise: Write("water.pump_1.enabled", false)
}
```

## SymLogic IR Goals

SymLogic IR should be:

* small
* deterministic
* serializable
* content-addressed
* inspectable by NPCs
* convertible to UI blocks
* executable without WASM
* compilable to WASM later
* safe for public infrastructure

## SymLogic IR Operations

Initial operations:

```text
Read(device.field)
Write(device.field, value)
Const(value)
And(a, b)
Or(a, b)
Not(a)
LessThan(a, b)
GreaterThan(a, b)
Equals(a, b)
Add(a, b)
Subtract(a, b)
Min(a, b)
Max(a, b)
Timer(duration)
RateLimit(count, interval)
EmitEvent(name)
```

Do not start with loops.

Avoid unbounded behavior.

## WASM Microcontroller Runtime

WASM devices should run under hard limits.

Suggested limits:

```text
max_fuel_per_tick
max_memory_pages
max_device_reads_per_tick
max_device_writes_per_tick
max_stdout_bytes_per_tick
max_storage_bytes
max_imported_functions
max_runtime_before_suspend
```

The runtime should provide only deterministic host functions.

Allowed host functions:

```text
sym_read_device
sym_write_device
sym_emit_event
sym_log
sym_time_tick
sym_rand_seeded
sym_mount_read
```

Forbidden host functions:

```text
real_wall_clock
host_filesystem
raw_network
host_random
thread_spawn
external_http_read
unbounded_stdout
```

## Device Bus

All scripts interact with the world through a deterministic Device Bus.

Scripts do not directly mutate Bevy components.

The bus exposes device fields.

Example paths:

```text
/dev/sym/water/tank_0/level
/dev/sym/water/pump_1/enabled
/dev/sym/power/grid_0/load
/dev/sym/factory/belt_4/speed
/dev/sym/door/blast_7/locked
/dev/sym/archive/events
```

## Deterministic Sync Point

Each simulation tick:

1. collect deterministic device snapshot
2. run SymLogic rules and WASM scripts within budget
3. queue output writes
4. validate permissions
5. stage writes
6. apply writes in stable order within frame budget
7. emit events
8. record significant changes for Chronicle

## Staged Output Queue

Scripts should not mutate game state directly at the sync point.

They should write to a staged output queue.

The engine drains the queue with:

* stable ordering
* permission validation
* time-slice budget
* conflict resolution
* overflow handling

If too many writes occur:

* throttle device
* suspend script
* emit warning
* create maintenance fault
* log abuse if relevant

This prevents many scripts from stalling the main thread.

## Conflict Resolution

If multiple scripts write to the same device:

Priority order may depend on:

* physical ownership
* local law
* emergency authority
* device lock
* role permission
* manual override
* script priority
* timestamp / tick order
* safety system

Example:

A public water controller and a rogue script both try to control the pump.

The pump accepts the write from the authorized water controller unless emergency override or sabotage changes the device state.

## Permissions

Scripts need permissions.

Permission examples:

```text
read_water
write_water
read_power
write_power
read_archive
write_archive
open_doors
control_factory
dispatch_drones
send_mesh_message
export_airlock_metrics
```

A script package should declare requested permissions.

Public infrastructure scripts may require approval.

## Content-Addressed Scripts

Every imported script or logic graph should be content-addressed.

Pipeline:

```text
Player IDE / Visual Tool
  ↓
Game Client Import
  ↓
Validate
  ↓
Hash
  ↓
Create Blueprint Asset
  ↓
Store in Worldline Archive
  ↓
Mount in Device
```

The hash becomes part of worldline history.

This supports replay and audit.

## Package Format

A script package should include:

```text
name
version
hash
author
license
permissions_requested
devices_supported
entrypoint
runtime
resource_limits
signatures
audit_status
known_risks
```

Package installation uses `sym-get`, not real internet package managers.

Example:

```sh
sym-get install pump-controller@sha256:abcd...
sym-get verify pump-controller
```

## SymtropyOS Shell

The shell is an interface layer, not necessarily a full OS.

Initial commands:

```text
ls
cat
echo
grep
tail
head
status
sym-dev
sym-log
sym-mount
sym-get
sym-run
sym-perms
sym-archive
```

Example:

```sh
sym-dev read /dev/sym/water/tank_0/level
sym-dev write /dev/sym/water/pump_1/enabled 1
sym-log tail waterworks
sym-get verify pump-controller
```

The shell should feel authentic while remaining controlled.

## RISC-V / Linux Emulator Role

A RISC-V or Linux-like emulator may still exist later.

Best use cases:

* Ghost Civilization prestige terminals
* rare ancient mainframes
* advanced modded worlds
* educational challenge dungeons
* special campaign moments
* high-end faction compute labs

But it should not be required for basic automation.

Recommended stance:

**WASM is the gameplay runtime.
RISC-V/Linux is a prestige simulation layer.**

## Null Infection

Null Ecologies should attack software visibly.

Infection modes:

### Signal Rot

* false sensor readings
* text jitter
* glyph corruption
* repeated logs
* phantom device states

### Factory Bloom

* fabricators ignore queues
* belts reroute goods
* drones build machine nests
* power draw spikes

### Logistics Parasite

* warehouse manifests lie
* crates disappear
* convoy routes change
* duplicate orders appear

### Defense Grid Remnant

* old laws lock doors
* turrets target civilians
* public override disabled
* access control refuses living authority

## Terminal Glitch Language

Null corruption should be visible.

Examples:

* amber text shifts to red
* characters smear
* unknown unicode appears
* lines repeat
* command prompt changes
* fake logs appear
* device names mutate
* cursor moves without input
* screen sync pulses with enemy signal

This makes software horror legible.

## Field Deck Integration

The Field Deck is the player’s primary interface to this system.

Deck modes:

* DIAG for device bus
* ARCHIVE for logs and records
* MESH for local network
* OFFLINE for trusted verification
* SCAN for signals and hazards

The Field Deck can:

* mount cartridges
* inspect device state
* run simple scripts
* access terminals
* verify packages
* recover logs
* show permissions
* display Chronicle entries

## Seedworks Implementation Path

Do not start with full WASM.

Build in stages.

### Milestone 1 — Fake Terminal, Real Device Bus

Commands:

```text
status
ls-dev
read water.tank_0.level
write water.pump_1.enabled 1
log tail water
```

Goal:

Prove gameplay.

### Milestone 2 — SymLogic Blocks

Visual logic for pump control.

Goal:

Support non-coders.

### Milestone 3 — AST Interpreter

Run serialized logic graphs.

Goal:

Deterministic automation.

### Milestone 4 — WASM Microcontroller

Run small scripts with budget limits.

Goal:

Advanced automation.

### Milestone 5 — SymtropyOS Shell

Expose shell over device bus and package system.

Goal:

Mainframe fantasy.

### Milestone 6 — Ghost Terminal Dungeon

Use terminal interaction to restore water or unlock a door.

Goal:

Magic moment.

## Example Seedworks Mission

The Old Waterworks pump is locked by a dead emergency authority.

The player powers a terminal with their rover battery.

```sh
$ status
POWER: external rover battery
PUMP_1: locked
FILTER_2: degraded
TANK_0: 12%
PUBLIC_OVERRIDE: disabled

$ cat last_order.txt
Emergency Automation Authority extended.
Public override suspended until threat level falls below RED.

$ sym-dev write /dev/sym/water/pump_1/enabled 1
Permission denied.

$ sym-archive request-witness
Archivist witness requested.

$ sym-dev write /dev/sym/water/pump_1/enabled 1 --witness archive
Override restored.
```

The player learns:

* computing is physical
* code is governance
* old laws can persist after society dies
* archive witnesses matter
* water restoration is technical and civic

## Final Principle

The goal is not to build Linux inside a game.

The goal is to make computation into a material of civilization.

Symtropy’s automation should feel real because it has:

* power cost
* device permissions
* physical location
* script identity
* replay history
* security limits
* social legitimacy
* Chronicle memory

Code is not separate from civilization.

Code is one of the ways civilization touches matter.
