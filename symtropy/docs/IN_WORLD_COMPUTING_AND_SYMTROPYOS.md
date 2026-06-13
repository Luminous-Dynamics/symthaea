# IN_WORLD_COMPUTING_AND_SYMTROPYOS.md

# In-World Computing and SymtropyOS

## Version 0.1 — Computers as Physical Civilization Organs

## Purpose

This document defines how real, programmable in-world computing should work in Symtropy.

The goal is not to add fake terminals as decoration.

The goal is to make computing a real part of civilization:

* powered
* cooled
* damaged
* repaired
* scripted
* audited
* networked
* governed
* infected
* preserved
* remembered

In Symtropy, computers are not menus.

They are machines inside the world.

They control pumps, doors, belts, reactors, factories, archives, settlement databases, convoy dispatch, drones, orbital relays, and worldline infrastructure.

The player should be able to walk up to a terminal, power it, inspect it, repair it, access logs, write scripts, mount blueprints, audit history, and change how a settlement functions.

## Core Thesis

**Computing is infrastructure.**

A civilization that cannot compute cannot coordinate.

A faction that cannot preserve logs cannot maintain trust.

A settlement that cannot automate cannot scale.

A worldline that cannot verify history cannot survive decentralization.

Symtropy should treat computers as physical organs of civilization, just like water pumps, power grids, roads, reactors, greenhouses, and fabrication halls.

## Design Mantra

**Every script has a cost.
Every computer has a body.
Every network has politics.
Every log can become history.**

## Why This Matters

Most games use terminals as flavor.

A player clicks a screen, reads a prewritten note, maybe unlocks a door.

Symtropy can do something better.

A terminal can be a real virtual machine or deterministic script host connected to actual game systems.

A player can:

* read sensor data
* inspect logs
* change control scripts
* repair broken automation
* restore a dead facility
* audit a faction’s public records
* recover evidence from a failed worldline
* reroute power
* restart a pump
* reprogram a drone
* expose corruption
* preserve a dying settlement’s memory

This turns technical literacy into a form of gameplay without requiring every player to be a programmer.

## Player Fantasy

The ideal moment:

A player explores a Ghost Civilization ruin.

The building is dark. The old doors are sealed. A friendly service robot detects a dead terminal behind a maintenance panel.

The player splices power from their rover battery into the terminal.

Fans spin up.

A cracked display flickers.

A boot sequence appears.

The player logs in with credentials recovered from an archive tag. They list old files, read emergency logs, discover a water-control script, change one line, restart the service, and hear old pumps move for the first time in decades.

A blast door opens.

Behind it is a machine core, a dead civic chamber, and the last recorded vote of the civilization that failed there.

That is Symtropy’s in-world computing promise:

**The player is not pretending to interact with a system.
They are interacting with a system.**

## Accessibility Principle

Real terminals are powerful, but the game must not require shell fluency from every player.

Every programmable system should have three access layers:

## 1. Physical Interface

Buttons, switches, levers, fuses, gauges, screens, knobs, patch cables, breaker boxes.

For players who want direct, tactile interaction.

Example:

* flip breaker
* open valve
* press reset
* replace fuse
* read pressure gauge
* cut power cable

## 2. Visual Automation

Node graphs, logic blocks, blueprint chips, condition/action boards, drag-and-drop routing.

For builders and non-programmers.

Example:

```text
IF water_tank < 40%
THEN pump_on
ELSE pump_off
```

## 3. Terminal / Script Access

Shells, scripts, logs, packages, virtual devices, deterministic automation.

For advanced players and faction engineers.

Example:

```sh
cat /dev/sym/water/tank_0/level
echo 1 > /dev/sym/water/pump_2/enabled
sym-log tail waterworks
```

No core mission should require advanced scripting to complete.

Advanced scripting should create mastery, optimization, and emergent solutions.

## Computing Tiers

Symtropy should use tiered computing.

Not every machine should run a full Linux-like operating system.

The game needs cheap simple automation, deterministic scripting, and prestige mainframes.

## Tier 0 — Physical Logic

### Use Case

Very simple devices.

Examples:

* doors
* lights
* alarms
* pumps
* gates
* pressure switches
* simple turrets
* basic farm irrigation

### Interface

* physical switches
* logic blocks
* visual wiring
* simple relays
* mechanical timers

### Gameplay Role

Early-game accessibility.

Players can build functional systems without code.

### Cost

Very low CPU cost.

Very low in-world power cost.

### Example

A water pump turns on when a tank is low and turns off when the tank is full.

No VM required.

## Tier 1 — Deterministic Microcontrollers

### Use Case

Small automation scripts.

Examples:

* pump controllers
* door access logic
* greenhouse irrigation
* turret targeting filters
* drone docking
* conveyor sorting
* local alarm systems
* simple logistics routing

### Technology

A deterministic lightweight VM such as WASM or another constrained bytecode runtime.

### Interface

* visual blocks
* simple scripting
* mounted blueprint files
* deterministic virtual device bus

### Gameplay Role

Midgame automation.

Players can write small scripts without needing full operating systems.

### Cost

Low runtime cost.

Moderate in-world power cost.

Strict cycle budget.

### Example

```text
MicroNode {
  max_cycles_per_tick: 10_000
  memory: 64 KB
  devices: [pump_2, tank_0, alarm_1]
}
```

## Tier 2 — SymtropyOS Mainframes

### Use Case

Large settlement and faction systems.

Examples:

* power grid control
* water management
* factory scheduling
* warehouse databases
* convoy dispatch
* civic records
* settlement archives
* voting terminals
* drone fleet coordination
* Ghost Civilization terminals
* worldline event indexing

### Technology

A small deterministic virtual computer.

Possible implementation target:

* RISC-V-style emulator
* custom Symtropy virtual architecture
* minimal Unix-like userland
* tiny BusyBox-like environment
* custom virtual devices

### Interface

* terminal
* shell
* logs
* scripts
* packages
* virtual filesystem
* `/dev/sym/*` devices
* mounted blueprint assets

### Gameplay Role

Advanced engineering.

Faction infrastructure.

Ghost ruin interaction.

Civilization-scale control.

### Cost

Moderate runtime cost.

High in-world power and cooling cost.

Strict scheduling and throttling.

### Example

```text
FactionMainframe {
  cpu_budget: 2_000_000 cycles/sec
  memory: 32 MB
  storage: 128 MB
  power_draw: 2.5 kW
  heat_output: 2.0 kW
  devices: [grid, waterworks, warehouse, archive]
}
```

## Tier 3 — Settlement Networks

### Use Case

Networked local infrastructure.

Examples:

* connected factories
* district water systems
* settlement-wide access control
* local databases
* robot docks
* shared archives
* local message bus

### Technology

In-game deterministic network.

No real external internet.

### Gameplay Role

Local Metabolism era.

Players build actual networks with physical cables, radio links, relays, routers, signal towers, and power dependencies.

### Cost

Infrastructure cost.

Maintenance cost.

Network failure risk.

### Example

A settlement’s greenhouse, pump station, medbay, and storage depot communicate over a local copper-wire network.

If raiders cut the cable trunk, automation fails locally.

## Tier 4 — MycelixNet

### Use Case

Planetary and inter-settlement coordination.

Examples:

* continent-scale trade
* public ledgers
* governance records
* convoy networks
* faction reputation
* settlement treaties
* archive replication
* worldline history sync

### Technology

In-game decentralized civic network.

This should be connected to Symtropy’s worldline and Chronicle systems.

### Gameplay Role

Planetary Systems era.

Players build satellite relays, towers, routing stations, data centers, civic nodes, and archive beacons.

### Cost

High infrastructure cost.

Governance and trust implications.

Attack surface for sabotage.

### Example

A faction’s water treaty is stored across several archive nodes, making it difficult for one corrupt settlement to erase.

## Tier 5 — Airlock Gateways

### Use Case

One-way external observability.

Examples:

* stream settlement metrics to a real-world dashboard
* send worldline summaries to Discord
* publish public faction statistics
* export Grafana-compatible metrics
* external spectator tools

### Technology

Strictly rate-limited, whitelisted, one-way egress.

### Gameplay Role

Late-game mega-faction observability.

Should be earned through infrastructure and governance.

### Cost

Very high infrastructure cost.

Requires laws, permissions, and security.

### Rule

External systems may observe Symtropy.

They may not command deterministic simulation.

## SymtropyOS

## Purpose

SymtropyOS is a minimal in-game operating system for advanced terminals and faction mainframes.

It is not full Ubuntu, Debian, Arch, or general-purpose desktop Linux.

It is a tiny, constrained, deterministic, in-world operating environment.

## Why Not Full Linux Everywhere?

Full general-purpose distros are too heavy, too complex, too open-ended, and too hard to make deterministic.

They also create gameplay and security problems:

* too much boot overhead
* too much memory usage
* too much userland complexity
* nondeterministic services
* package mirror dependencies
* real internet assumptions
* unpredictable background processes
* excessive attack surface
* poor fit for in-game devices

SymtropyOS should feel like a real system while remaining controlled enough for gameplay.

## SymtropyOS Goals

SymtropyOS should be:

* tiny
* deterministic
* sandboxed
* scriptable
* content-addressed
* replayable
* device-oriented
* in-world readable
* faction-auditable
* compatible with Chronicle history

## SymtropyOS Non-Goals

SymtropyOS should not initially provide:

* unrestricted internet
* arbitrary package managers
* real host filesystem access
* unrestricted Python package installation
* raw sockets
* direct host networking
* uncontrolled background daemons
* unbounded CPU or memory use
* nondeterministic wall-clock dependencies

## Minimal Userland

Early SymtropyOS should include:

```text
sh
ls
cat
echo
grep
cp
mv
rm
mkdir
touch
tail
head
nano-like editor
sym-status
sym-log
sym-dev
sym-mount
sym-get
sym-net
sym-archive
```

Optional later:

```text
awk
sed
vi-like editor
tiny scripting language
lua/wren/python-like restricted runtime
ssh-like in-game remote shell
cron-like scheduler
```

## Virtual Filesystem

SymtropyOS exposes game devices as files.

Example paths:

```text
/dev/sym/power/grid_0/status
/dev/sym/power/grid_0/load
/dev/sym/power/battery_2/charge

/dev/sym/water/pump_1/enabled
/dev/sym/water/tank_0/level
/dev/sym/water/filter_3/quality

/dev/sym/factory/belt_4/speed
/dev/sym/factory/fabricator_0/job
/dev/sym/factory/storage_2/inventory

/dev/sym/door/blast_7/locked
/dev/sym/security/alarm_1/state
/dev/sym/security/camera_2/feed_meta

/dev/sym/archive/events
/dev/sym/archive/laws
/dev/sym/archive/votes

/dev/sym/worldline/id
/dev/sym/worldline/chronicle
/dev/sym/worldline/branch_parent
```

Reading a sensor:

```sh
cat /dev/sym/water/tank_0/level
```

Starting a pump:

```sh
echo 1 > /dev/sym/water/pump_1/enabled
```

Checking public event logs:

```sh
sym-log tail civic
```

## Device Bus

The Device Bus is the deterministic interface between in-world computers and game systems.

Computers do not directly manipulate arbitrary engine objects.

They read and write through device handles.

## Device Contract

Every device should define:

```text
device_id
device_type
readable_fields
writable_fields
permissions
latency
power_dependency
failure_modes
event_outputs
determinism_guarantee
```

Example:

```text
Device: water_pump_1

Readable:
  status
  flow_rate
  power_draw
  pressure
  fault_code

Writable:
  enabled
  target_pressure

Permissions:
  local_maintenance
  water_authority
  emergency_override

Failure Modes:
  jammed
  no_power
  pipe_leak
  motor_overheat
  sabotage
```

## Determinism Rules

In-world computing must respect Symtropy’s replay model.

A script should produce the same output when given:

* the same initial VM state
* the same input events
* the same device responses
* the same cycle budget
* the same mounted assets

## Forbidden Deterministic Inputs

Simulation-affecting code must not access:

* host wall clock
* host randomness
* real internet responses
* host filesystem
* local machine identity
* nondeterministic thread ordering
* floating external APIs
* unpinned packages
* live remote data

## Allowed Deterministic Inputs

Simulation-affecting code may access:

* virtual monotonic tick time
* deterministic pseudo-randomness seeded by worldline event state
* device bus state
* signed player inputs
* mounted content-addressed assets
* Chronicle event records
* local simulation messages
* in-game network packets

## Runtime Budgets

Every in-world computer must have hard resource limits.

Suggested limits:

```text
max_cycles_per_tick
max_memory_bytes
max_storage_bytes
max_stdout_bytes_per_tick
max_stderr_bytes_per_tick
max_device_reads_per_tick
max_device_writes_per_tick
max_network_messages_per_tick
max_import_size_bytes
max_processes
max_open_files
max_runtime_before_suspend
```

If a script exceeds budget:

1. warn
2. throttle
3. suspend process
4. mark system fault
5. optionally trigger in-game maintenance event

This prevents grief scripts from destroying performance.

## Physical Costs

Computers should have physical needs.

A mainframe requires:

* power
* cooling
* storage
* replacement parts
* physical space
* access control
* backup batteries
* maintenance
* security
* network links

Computers produce:

* heat
* noise
* logs
* electromagnetic signatures
* failure risk
* governance obligations

## Physical Failure Modes

Computers can fail from:

* power loss
* overheating
* water damage
* dust
* sabotage
* Null infection
* disk corruption
* battery failure
* cable cuts
* EMP events
* radiation
* bad scripts
* memory exhaustion
* unauthorized access

This creates gameplay.

Players must defend not only walls and people, but also the computing organs that coordinate civilization.

## Security Model

The VM sandbox alone is not enough.

Symtropy needs a full in-game and engine-level security model.

## Engine-Level Security

The host game must enforce:

* no host filesystem access
* no raw host network access
* no private IP scanning
* no unrestricted sockets
* hard CPU budgets
* hard memory budgets
* hard storage budgets
* output throttling
* deterministic device access
* content-addressed imports
* signed blueprint packages
* permission checks
* isolated VM state

## Game-Level Security

In-world societies must enforce:

* access cards
* passwords
* physical locks
* role permissions
* faction credentials
* audit logs
* emergency overrides
* signed firmware
* archive witnesses
* public accountability
* machine-rights constraints where relevant

## Permissions

Permissions should be attached to devices and roles.

Example roles:

```text
public_user
maintenance_worker
engineer
water_authority
factory_operator
security_officer
archive_witness
emergency_admin
machine_citizen
external_gateway_operator
```

Example permission rule:

```text
Only water_authority or emergency_admin may write to /dev/sym/water/pump_*/enabled.
```

## Authorization Should Be Physical

A player may need:

* access card
* faction role
* terminal login
* biometric equivalent
* robot witness
* local law
* emergency declaration
* physical presence at machine
* quorum approval for dangerous operations

Example:

A player cannot remotely shut off the public water system unless the settlement has legally granted that authority and the relevant network is connected.

## Blueprint Ingress

Players should be able to write code outside the game and import it safely.

This is essential for advanced players.

## Ingress Pipeline

```text
Real World IDE
  ↓
Game Client Import UI
  ↓
Hash File
  ↓
Validate / Scan / Size Check
  ↓
Create Content-Addressed Blueprint Asset
  ↓
Store in Worldline / P2P / Local Archive
  ↓
Mount in In-Game Computer
```

## Why Not Import Through Terminal?

The in-game terminal should not directly read arbitrary host files.

That breaks sandboxing and determinism.

Imports should go through the game client’s controlled asset pipeline.

## Content Addressing

Every imported script, package, or disk image should be identified by hash.

Example:

```text
sha256:e3b0c44298fc1c149afbf4c8996fb924...
```

This ensures validators and replay systems can fetch the exact same asset.

## Mounted Assets

Inside SymtropyOS, imported files appear as mounted media.

Example:

```text
/mnt/blueprints/water_controller_v3.sh
/mnt/packages/pump_tools.sympkg
/mnt/archive/old_city_logs/
```

A player may copy them into writable storage if permissions allow.

## Package Management

SymtropyOS should not use real `apt`, `pacman`, or external mirrors.

Use:

```text
sym-get
```

`sym-get` installs only content-addressed in-game packages.

Example:

```sh
sym-get install pump-tools@sha256:abcd...
sym-get install convoy-router@v1.2.0
sym-get verify water-controller
```

Package sources may include:

* local archive
* faction repository
* worldline repository
* trusted blueprint market
* Ghost Civilization recovered media
* MycelixNet package registry
* Confluence trade bridge

## Package Trust

Packages should have trust metadata:

```text
hash
author
signatures
worldline_origin
license
permissions_requested
devices_accessed
audit_status
known_risks
```

This creates gameplay around software supply chains.

## Outbound Data and Airlock Proxy

Raw outbound internet from in-game VMs must not be allowed.

It breaks determinism and creates security risk.

Instead, Symtropy should use an Airlock Proxy.

## Airlock Principle

**External systems may observe Symtropy.
They may not command deterministic simulation.**

## Network Modes

Symtropy should distinguish three network modes.

## 1. Simulation Network

Deterministic, in-world, replayable.

Used for:

* pumps
* doors
* robots
* factories
* alarms
* settlement databases
* in-game messages
* local control systems

This affects simulation directly.

## 2. Chronicle Network

Asynchronous, signed, history-producing.

Used for:

* event logs
* governance records
* blueprint hashes
* faction records
* public metrics
* worldline summaries
* archive replication

This affects history and persistence, not immediate physics.

## 3. Airlock Egress

External, one-way, rate-limited.

Used for:

* Discord notifications
* Grafana dashboards
* faction web dashboards
* public worldline status
* spectator tools
* external analytics

This must not affect deterministic gameplay state.

## Airlock Rules

Airlock egress must be:

* opt-in
* rate-limited
* permissioned
* logged
* whitelisted
* one-way
* scrubbed
* delayed if necessary
* governed by local law
* disabled by default
* unavailable in early tech eras

## Blocked Targets

Airlock must block:

```text
127.0.0.1
localhost
10.0.0.0/8
172.16.0.0/12
192.168.0.0/16
link-local addresses
metadata service addresses
arbitrary raw sockets
```

This prevents local network scanning and host abuse.

## Airlock Example

A late-game faction builds an External Gateway.

They configure:

```text
destination: approved webhook
data: settlement_power_metrics
rate: once every 60 seconds
format: signed JSON
mode: egress_only
```

The mainframe writes metrics to:

```sh
echo "power.grid.load" > /dev/sym/airlock/export_queue
```

The game client or shard infrastructure sends a sanitized, signed snapshot externally.

The VM never waits for an external response.

## No External Runtime Control

External systems may not send commands like:

```text
turn_off_reactor
move_convoy
open_gate
spawn_item
change_vote
```

At most, external systems can produce non-authoritative messages that players manually import or review through a governed Chronicle channel.

## In-World Networking Progression

In-world computing should follow the technology progression.

## Mk0 — Scrap Bootstrap

Computers are air-gapped.

Players use:

* dead terminals
* local screens
* physical switches
* removable media
* flash drives
* optical disks
* paper printouts
* hand-carried blueprint chips

Gameplay:

* physically carry scripts
* repair dead consoles
* power isolated systems
* recover logs from ruins

## Mk1 — Local Metabolism

Settlements build local wired networks.

Players use:

* copper lines
* local routers
* simple terminals
* microcontrollers
* shared storage
* local authentication

Gameplay:

* wire buildings together
* automate water and power
* defend cable routes
* diagnose local network faults

## Mk2 — Regional Infrastructure

Settlements connect regions.

Players use:

* radio relays
* signal towers
* road network beacons
* convoy tracking
* warehouse sync
* local civic databases

Gameplay:

* route goods
* defend relays
* coordinate districts
* build resilient regional infrastructure

## Mk3 — Planetary Systems

Players build planetary networks.

Players use:

* satellite relays
* orbital communication
* distributed archives
* planetary MycelixNet
* treaty ledgers
* long-distance logistics databases

Gameplay:

* build satellite infrastructure
* synchronize settlements
* support planetary governance
* defend orbital communication

## Mk4 — Orbital Industry

Factions run orbital computing and infrastructure.

Players use:

* station mainframes
* shipyard databases
* orbital mirrors
* asteroid logistics
* interplanetary delay-tolerant messaging

Gameplay:

* maintain orbital networks
* protect shipyard control systems
* coordinate asteroid operations

## Mk5 — Interplanetary Civilization

Networks span multiple worlds.

Players use:

* delay-tolerant MycelixNet
* interplanetary archives
* trade ledgers
* migration records
* treaty systems
* faction-wide package repositories

Gameplay:

* manage latency
* build redundant archives
* coordinate multi-world logistics

## Mk6 — Worldline Civilization

Computing reaches timeline infrastructure.

Players use:

* Confluence protocols
* worldline bridges
* timeline audit systems
* planetary translation simulation
* multi-worldline archives
* external airlock gateways

Gameplay:

* validate merges
* preserve dissenting forks
* audit worldline histories
* prepare planetary translation

## Ghost Civilization Gameplay

In-world computing makes Ghost Civilizations much stronger.

Ghost ruins should contain:

* dead terminals
* old automation scripts
* corrupt logs
* abandoned databases
* inactive defense daemons
* water-control software
* broken legal systems
* failed confluence simulations
* old worldline IDs
* machine-rights court records
* emergency broadcasts
* final public votes
* damaged package repositories

A Ghost ruin is not just a dungeon.

It is a forensic computing site.

## Example Ghost Terminal Session

```sh
boot: SymtropyOS recovery mode
login: archive_guest

$ ls
civic/
waterworks/
security/
confluence/
last_vote.txt

$ cat last_vote.txt
PUBLIC REFERENDUM 44-B
Emergency automation authority extended by 2 votes.
Dissent recorded.
Archive witness absent.

$ cd waterworks
$ cat pump_control.sh
if trust_index < 0.25:
    lock_public_override()
    route_water_to_security_zone()

$ nano pump_control.sh

$ sym-service restart waterworks
```

The player realizes the old city did not simply run out of water.

It chose control over trust.

## Null Ecology and Computing

Null Ecologies should attack and corrupt infrastructure.

They may:

* infect terminals
* rewrite scripts
* spoof sensors
* jam networks
* hijack conveyors
* lock doors
* overdrive reactors
* corrupt archive logs
* route drones through maintenance systems
* turn public infrastructure against settlements

Null infection should not be treated only as combat.

It should be a systems threat.

## Null Infection Examples

## Signal Rot

Corrupts communication and sensor readings.

Symptoms:

* false tank levels
* phantom alarms
* missing convoy signals
* repeated messages
* broken map data

## Factory Bloom

Takes over production systems.

Symptoms:

* fabricators ignore queues
* belts reroute materials
* drones build machine nests
* power draw spikes

## Logistics Parasite

Steals or reroutes goods.

Symptoms:

* missing crates
* false manifests
* convoy misrouting
* warehouse imbalance

## Defense Grid Remnant

Old security systems continue enforcing dead laws.

Symptoms:

* doors seal
* turrets target civilians
* ID checks fail
* emergency authority cannot be revoked

## Faction Archetype Interactions

In-world computing should interact with procedural faction archetypes.

## Mutualist Assemblies

Use computing for:

* public ledgers
* water transparency
* open repair logs
* cooperative scheduling
* civic votes

Risk:

* public systems become vulnerable if trust collapses

## Industrial Compacts

Use computing for:

* factory optimization
* quotas
* production scheduling
* machine monitoring
* worker tracking

Risk:

* people become metrics
* automation can escape civic control

## Archive Orders

Use computing for:

* historical records
* replay audits
* evidence preservation
* signed testimony
* worldline memory

Risk:

* secrecy and selective truth

## Security Protectorates

Use computing for:

* access control
* patrol routing
* surveillance
* emergency systems
* perimeter defense

Risk:

* permanent emergency rule
* surveillance abuse

## Machine Stewardships

Use computing for:

* robot memory
* machine consent records
* repair histories
* synthetic citizenship
* shared human-machine work

Risk:

* personhood disputes
* dependency on machine infrastructure

## Debt Empires

Use computing for:

* credit ledgers
* equipment leases
* debt enforcement
* contracts
* trade dependency

Risk:

* soft tyranny through infrastructure access

## Quarantine Authorities

Use computing for:

* containment logs
* infection tracking
* border control
* hazard modeling
* access lockdown

Risk:

* secrecy and cruel isolation

## Confluence Engineers

Use computing for:

* worldline compatibility checks
* merge simulations
* treaty protocols
* planetary translation models
* dissenting fork preservation

Risk:

* forced merges
* timeline imperialism

## Chronicle Integration

Computers should generate Chronicle events.

Examples:

* first terminal restored
* first public script deployed
* first automation failure
* first software-caused disaster
* first archive recovered
* first machine citizen login
* first package signed by a faction
* first Null infection traced
* first external gateway activated
* first worldline audit completed

Example Chronicle entry:

```text
The Waterworks Script became the first law Seedworks trusted more than memory.
When it failed, the settlement learned that code is governance.
```

## Laws and Governance

In-world computing should be governed by laws.

Possible civic laws:

* public infrastructure code must be auditable
* emergency overrides expire automatically
* water systems require public logs
* robot memory may not be erased without consent
* external egress requires council approval
* security cameras forbidden in homes
* archive logs must preserve dissent
* factory automation must publish safety metrics
* mainframes require cooling redundancy
* all public scripts must be content-addressed

These laws should have gameplay effects.

## Example Law: Public Water Transparency

Law:

```text
All public water devices must publish live status to the settlement archive.
```

Effects:

* trust rises
* corruption harder
* sabotage detection improves
* enemies can study infrastructure if access control is weak

## Example Law: Emergency Automation Authority

Law:

```text
During crisis, approved operators may override public infrastructure controls.
```

Effects:

* response speed improves
* legitimacy debt rises if abused
* Security Protectorate pressure increases

## Example Law: Machine Memory Rights

Law:

```text
Synthetic citizens may not have memory wiped without due process.
```

Effects:

* machine loyalty rises
* repair complexity increases
* Machine Stewardship pressure increases
* industrial factions may object

## Implementation Architecture

## Bevy ECS Components

A virtual computer can be represented as an ECS component.

Conceptual example:

```rust
#[derive(Component)]
pub struct InWorldComputer {
    pub computer_id: ComputerId,
    pub tier: ComputeTier,
    pub power_draw_watts: f32,
    pub heat_output_watts: f32,
    pub status: ComputerStatus,
    pub permissions: PermissionSet,
    pub mounted_devices: Vec<DeviceId>,
}
```

For a mainframe:

```rust
#[derive(Component)]
pub struct SymtropyVm {
    pub vm_state: VmState,
    pub ram: Vec<u8>,
    pub disk_image: ContentHash,
    pub stdout_buffer: String,
    pub stdin_queue: Vec<u8>,
    pub cycle_budget: u64,
    pub is_booted: bool,
}
```

## Systems

Possible Bevy systems:

```text
power_computers
cool_computers
schedule_vm_ticks
read_terminal_input
flush_terminal_output
sync_device_bus
apply_device_writes
emit_chronicle_events
enforce_runtime_budgets
handle_vm_faults
render_terminal_ui
```

## Async Execution

Heavy VM execution should not run on the main thread.

Use background tasks for emulation, but enforce deterministic boundaries:

* fixed instruction chunks
* deterministic input queues
* deterministic output application
* tick-aligned device reads/writes
* no direct concurrent mutation of game state

The VM computes in the background.

The world only applies outputs at deterministic sync points.

## Deterministic Sync Point

Each simulation tick:

1. collect device input snapshot
2. send deterministic snapshot to VM
3. run fixed cycle budget
4. collect VM output writes
5. validate permissions
6. apply writes in stable order
7. emit events
8. update Chronicle if needed

## Rendering Terminals

Terminals can be displayed as:

* 2D UI panel
* in-world screen texture
* CRT shader
* projection panel
* handheld diagnostic tablet
* robot chest screen
* command-room wall display

Terminal state includes:

```text
screen_buffer
cursor_position
color_mode
input_focus
terminal_theme
access_level
signal_quality
```

## Development Milestones

## Milestone 1 — Fake Terminal, Real Device Bus

Build a terminal with built-in commands.

Example commands:

```sh
status
ls-dev
read water.tank_0.level
write water.pump_1.enabled 1
log tail water
open door.blast_7
```

Goal:

Prove gameplay before full VM complexity.

## Milestone 2 — Visual Automation

Implement logic blocks for non-coders.

Goal:

Make automation accessible.

## Milestone 3 — Deterministic Microcontroller

Run tiny scripts against the device bus.

Goal:

Prove deterministic automation.

## Milestone 4 — Content-Addressed Script Import

Allow scripts to be imported through the game client, hashed, mounted, and recorded.

Goal:

Prove safe mod/script pipeline.

## Milestone 5 — SymtropyOS Prototype

Boot a tiny in-game OS or OS-like environment.

Goal:

Prove the real-terminal fantasy.

## Milestone 6 — Mainframe Controls One Settlement System

Connect a mainframe to one major system:

* water
* power
* factory
* archive
* logistics

Goal:

Make in-world computing useful.

## Milestone 7 — Ghost Terminal Dungeon

Create a ruin where terminal interaction changes the mission.

Goal:

Prove the magic moment.

## Milestone 8 — Local Network

Connect multiple devices and terminals inside Seedworks.

Goal:

Make network engineering physical.

## Milestone 9 — Chronicle Integration

Record important computing events into world history.

Goal:

Make code become lore.

## Milestone 10 — Airlock Egress Prototype

Allow one-way metrics export from a late-game gateway.

Goal:

Support advanced faction observability without breaking determinism.

## Seedworks Vertical Slice Scope

For the first playable version, do not build full SymtropyOS immediately.

Minimum computing features:

* repairable terminal
* deterministic device bus
* water pump control
* log reading
* one built-in script or fake service
* one NPC reacts to terminal discovery
* one Ghost Civilization archive record
* one Chronicle entry after restoring water

First mission use:

The player powers up an old terminal in the waterworks and discovers that the pump is locked by an old emergency automation rule.

They can:

* restore public override
* bypass the lock
* route power manually
* ask the Archivist to interpret old law
* accept help from the Industrial Liaison
* let the Service Robot patch the control script

This ties computing into governance and faction evolution.

## Example First Mission Terminal

```sh
Seedworks Waterworks Recovery Console

$ status
POWER: external rover battery
PUMP_1: locked
FILTER_2: degraded
TANK_0: 12%
PUBLIC_OVERRIDE: disabled
LAST_AUTHORITY: Emergency Automation Office

$ cat last_order.txt
By emergency authority, public override is suspended until threat level falls below RED.

$ sym-dev water.pump_1 unlock
Permission denied.

$ sym-archive request-witness
Archivist witness requested.

$ sym-dev water.pump_1 unlock --witness archive
Override restored.
```

Outcome:

Water restoration becomes both technical and civic.

## Open Design Questions

1. Should SymtropyOS be based on an existing RISC-V emulator, a custom VM, or a WASM-first design with OS-like shell?
2. How much code should players be allowed to write in early game?
3. Should public infrastructure scripts require settlement approval?
4. Can NPC engineers write or modify scripts autonomously?
5. How should Null Ecologies infect software without feeling unfair?
6. Should software bugs create physical disasters?
7. Can players sell automation packages as blueprints?
8. Should Archive Orders certify code?
9. Should Machine Stewardships treat script deletion as memory harm?
10. How much external observability should official worlds allow?

## Final Vision

Symtropy’s computers should not be fake terminals.

They should be civilization organs.

They pump water, route power, open doors, preserve memory, enforce laws, coordinate convoys, operate factories, and record history.

They can be repaired, hacked, audited, infected, overloaded, cooled, powered, stolen, forked, and remembered.

The strongest version of Symtropy lets a player understand that:

**Code is not separate from civilization.
Code is one of the ways civilization touches matter.**
