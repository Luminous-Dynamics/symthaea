# NETWORKING_STACK_DECISION.md

# Symtropy Networking Stack Decision

## Version 0.1 — Lightyear for Real-Time Play, Chronicle for Durable History

## Decision Summary

Symtropy should use a layered networking architecture.

The recommended stack is:

```text id="tdrftj"
Bevy + Lightyear
  → local real-time multiplayer

Symtropy Device Bus
  → deterministic infrastructure transactions

Local Chronicle Backend
  → signed event history for Seedworks v0.1

Mycelix / Holochain Bridge
  → future civic identity, source chains, worldline history, and Confluence
```

## Core Decision

Use **Lightyear** for the fast real-time gameplay layer.

Do not use Lightyear as the durable truth layer for civilization, governance, archives, source chains, or worldline history.

Lightyear should handle bodies, bullets, drones, vehicles, local replication, prediction, interpolation, and moment-to-moment multiplayer responsiveness.

It should not decide what a society remembers.

## Design Mantra

**Lightyear moves bodies.
Device Bus commits machines.
Chronicle remembers meaning.
Mycelix carries civilization.**

## Why Lightyear

Symtropy needs a real-time multiplayer system that can support:

* player movement
* first-person combat
* co-op raids
* Null drones
* vehicles
* local physics interactions
* replicated tools
* Field Deck presence
* local shard state
* prediction / rollback-style responsiveness
* interpolation
* interest management

Building this from scratch would consume too much time.

Lightyear is a strong candidate because it is Bevy-native and focused on networked gameplay concerns.

## What Lightyear Should Own

Lightyear should own **Local Real-Time Truth**.

Examples:

* player position
* player rotation
* weapon raised / lowered
* tool use
* Null drone movement
* projectile replication
* vehicle control
* local combat state
* basic damage events
* Field Deck raised / lowered state
* Panic Drop state
* nearby terminal interaction state
* visible shared UI-state messages

Lightyear answers:

```text id="kilq1c"
What is happening right now in the room?
```

## What Lightyear Should Not Own

Lightyear should not own durable civilization truth.

Do not make it responsible for:

* public votes
* faction charters
* script legitimacy
* source chains
* worldline forks
* Confluence
* Chronicle entries
* settlement laws
* Archive Witness signatures
* long-term identity
* permanent blueprint ownership
* Mycelix governance state
* public infrastructure legal authority

Those belong to Chronicle / Mycelix / Holochain-style layers.

Lightyear answers the present.

Chronicle answers history.

## Truth Layers

## 1. Local Real-Time Truth

Implementation:

```text id="e18j6a"
Bevy + Lightyear
```

Used for:

* movement
* aiming
* shooting
* tool use
* drones
* vehicles
* local interactions
* animation
* short combat windows

Properties:

* fast
* responsive
* ephemeral
* not globally permanent by default

## 2. Device Transaction Truth

Implementation:

```text id="g1z6cc"
Symtropy Device Bus
```

Used for:

* `/dev/sym/*` device reads/writes
* staged output queues
* SymLogic AST
* WASM microcontroller execution
* deterministic integer fuel
* atomic transactions
* command-chatter faults
* device cooldowns
* permission checks

Properties:

* deterministic
* replayable
* transaction-based
* machine-authoritative

## 3. Chronicle / Civic Truth

Implementation:

```text id="id8b34"
Local Chronicle Backend first
Mycelix / Holochain bridge later
```

Used for:

* public votes
* Archive Witness events
* water restoration
* script installation
* credentials
* laws
* NPC deaths
* faction shifts
* settlement founding
* public infrastructure changes

Properties:

* signed
* append-only
* history-producing
* socially meaningful

## 4. Worldline Truth

Implementation:

```text id="abuy7o"
Mycelix / Holochain-style agent-centric persistence
```

Used for:

* worldline forks
* Confluence
* migration
* settlement charters
* treaties
* inter-settlement identity
* archive replication
* planetary-scale history

Properties:

* asynchronous
* agent-centric
* forkable
* validated by civic rules
* not real-time

## Seedworks v0.1 Networking Target

The first multiplayer target should be modest.

Recommended scope:

```text id="6hrxqn"
2–4 players
one listen-server or local shard authority
one Firstlight Basin mission area
one Old Waterworks interior
one Null drone encounter
one Field Deck patch interaction
one shared terminal / screen-state message
one water restoration Chronicle event
```

Do not attempt full decentralized worldline networking in v0.1.

The goal is to prove that co-op Seedworks feels good.

## Recommended v0.1 Topology

Use a **local shard authority** model.

Options:

* listen server
* hosted dedicated shard
* local co-op authority

The authoritative shard handles:

* real-time entity state
* local combat
* device interaction ordering
* mission state
* immediate world state

After the mission, durable outcomes are committed to the Chronicle layer.

## Why Not Fully Decentralized Real-Time First?

Fully decentralized real-time combat is extremely hard.

Problems:

* latency
* cheating
* prediction disputes
* physics divergence
* authority conflicts
* rollback complexity
* griefing
* bandwidth
* synchronized deterministic execution across hardware

Symtropy should become decentralized through worldline history, source chains, self-hosted shards, portable identity, and forkable records.

It does not need fully decentralized 60 FPS combat on day one.

## Ephemeral Action, Durable History

Combat is ephemeral until it produces meaningful consequences.

Example:

During a raid:

* players move
* shoot
* dodge
* repair
* revive
* patch cables
* fight Null drones

This is handled by Lightyear.

After the raid:

* pump restored
* Archive Witness signed
* one NPC wounded
* machine core recovered
* public override changed

This becomes Chronicle history.

## Old Waterworks Example

### During Combat

Handled by Lightyear:

```text id="t6j0et"
Player A shoots drone.
Player B raises Field Deck.
Player C repairs cable.
Null drone moves through hallway.
Door opens.
Player B panic-drops Deck.
```

### During Device Commit

Handled by Device Bus:

```text id="r324q3"
sym-dev read water.tank_0.level
sym-dev write water.pump_1.enabled true
DENIED: DEAD_AUTHORITY_LOCK
Archive Witness requested
Override restored
Pump enabled
```

### After Mission

Handled by Chronicle:

```text id="72tfjg"
Old Waterworks restored under Archive Witness.
Public override restored.
Null signal isolated.
Seedworks water level recovering.
```

## Field Deck Share Mode

Share Mode should not stream pixels.

Do not send video of the Deck screen.

Instead, send serialized UI state over the real-time layer.

Example payload:

```json id="l01hb8"
{
  "screen": "diagnostics",
  "device": "water.pump_1",
  "fields": {
    "status": "LOCKED",
    "power": "OFF",
    "authority": "DEAD_AUTHORITY_LOCK",
    "null_signal": 0.42
  },
  "selected": "request_archive_witness",
  "privacy_mask": ["credentials", "identity_key"]
}
```

Each client renders the amber Field Deck UI locally.

Benefits:

* low bandwidth
* readable
* privacy-aware
* accessible
* deterministic-friendly
* compatible with localization
* visually consistent

## Networked Field Deck States

Lightyear should replicate simple Field Deck states:

```text id="qvj35e"
DeckDown
DeckGlance
DeckRaised
TerminalFocus
PatchCableConnected
ShareModeActive
PanicDrop
```

Other players should see:

* Deck raised
* cable connected
* screen glow
* broad warning state
* share indicator
* panic drop action
* patch target

They should not automatically see private credentials or medical data.

## Device Bus and Networking

The Device Bus should run on the authoritative local shard.

Scripts and SymLogic rules do not directly mutate remote client state.

Flow:

```text id="lxgwdy"
client requests device interaction
  ↓
local shard validates permission
  ↓
Device Bus stages transaction
  ↓
transaction commits or rolls back
  ↓
result replicated to clients
  ↓
important event sent to Chronicle
```

## Deterministic Integer Fuel

Automation fuel must be deterministic.

Bad:

```text id="uu4w4n"
run script for 2 milliseconds
```

Good:

```text id="v6refo"
run script for 20,000 deterministic fuel units
```

Fuel should be:

* integer-based
* runtime-counted
* architecture-agnostic
* independent of CPU speed
* identical across peers when replayed

If fuel runs out:

* rollback staged writes
* emit fault
* commit no partial state
* show Field Deck warning

## Atomic Device Transactions

Every script execution creates a transaction.

```text id="hryp27"
begin transaction
  read snapshot
  compute
  stage writes
if success:
  commit
else:
  rollback
```

This prevents partial economic or infrastructure updates.

## Lightyear Boundary Rule

Lightyear may replicate the result of a committed Device Bus transaction.

It should not be the authority that decides whether a public infrastructure script was legally valid.

That belongs to Device Bus and Chronicle.

## Chronicle Boundary Rule

Chronicle may record that a Device Bus transaction happened.

It should not run every frame of combat or every movement update.

That belongs to Lightyear.

## Mycelix / Holochain Boundary Rule

Mycelix/Holochain should validate civic and worldline records.

It should not be in the 60 FPS combat hot path.

## Transport Strategy

For v0.1, keep transport simple.

Recommended starting path:

* local network / direct co-op
* listen server
* small session
* one mission area
* basic replicated gameplay

Future transports may support:

* dedicated community shards
* Steam networking
* self-hosted servers
* WebTransport / web clients
* regional shard servers
* hybrid P2P layers

Do not overbuild transport before gameplay is fun.

## Interest Management

Seedworks should use spatial interest.

Players only need detailed state for:

* nearby teammates
* visible enemies
* local devices
* active mission terminals
* nearby projectiles
* relevant Field Deck shared UI
* current settlement region

Do not replicate the whole settlement to every client in full fidelity.

## What Gets Replicated

In the Old Waterworks slice, replicate:

* player transforms
* player animation state
* tool state
* weapon state
* Null drone state
* interactable panel state
* Field Deck state
* patch cable state
* water pump result
* door state
* mission objective state
* shared UI-state payloads

## What Does Not Need Full Replication

For v0.1, do not fully replicate:

* entire settlement economy
* all NPC schedules
* all historical records
* all device internals
* all scripts
* all Chronicle logs
* all worldline state

Only replicate what the mission needs.

## Chronicle Event After Multiplayer Mission

At mission end, generate one durable event.

Example:

```json id="n4vkro"
{
  "event_type": "WaterworksRestored",
  "worldline_id": "seedworks.local.001",
  "region": "FirstlightBasin.OldWaterworks",
  "participants": ["agent_a", "agent_b", "agent_c"],
  "device_events": ["pump_1_override_restored"],
  "witnesses": ["archive_witness_mara"],
  "npc_outcomes": ["technician_ivo_wounded"],
  "summary": "The Old Waterworks were restored under Archive Witness after a Null incursion.",
  "signatures": ["..."]
}
```

This is the bridge from real-time action to durable history.

## Development Milestones

## Milestone 1 — Singleplayer Device Bus

No networking yet.

* Field Deck
* fake terminal
* water pump
* dead authority lock
* Archive Witness
* local Chronicle event

## Milestone 2 — Lightyear Movement Prototype

* two players
* replicated movement
* replicated Deck raised/down state
* panic drop visible to teammate

## Milestone 3 — Co-op Old Waterworks

* two to four players
* Null drone encounter
* patch cable interaction
* shared mission objective
* water restored

## Milestone 4 — Share Mode Payload

* Systems Operator shares diagnostic UI state
* teammate renders amber interface locally
* privacy mask works

## Milestone 5 — Device Transaction Replication

* committed Device Bus results replicate
* permission denial visible
* device fault visible
* door/pump state consistent

## Milestone 6 — Chronicle Commit

* mission outcome becomes signed local Chronicle event
* participants listed
* Archive Witness listed
* event visible in Field Deck Archive mode

## Milestone 7 — Mock Source Chains

* Field Deck records local event chain
* agents sign mission outcome
* local file backend persists records

## Milestone 8 — Mycelix / Holochain Bridge

* optional later
* civic records and identity begin using agent-centric backend
* not in combat hot path

## Risks

## Risk 1 — Networking Before Fun

Do not build deep networking before the singleplayer Old Waterworks loop feels good.

## Risk 2 — Durable History Too Early

Do not require Mycelix/Holochain before local Chronicle proves the record format.

## Risk 3 — Real-Time Decentralization Trap

Do not attempt fully decentralized physics and combat before a local shard model works.

## Risk 4 — Over-Replicating Device State

Only replicate what players can observe or affect.

## Risk 5 — Pixel Streaming Share Mode

Never stream Deck video if serialized UI state will do.

## Final Decision

Use Lightyear for Seedworks real-time multiplayer.

Keep the deterministic Device Bus independent.

Keep Chronicle history independent.

Keep Mycelix/Holochain as the future civic/worldline persistence layer.

The architecture should remain:

```text id="4gwq2m"
Lightyear = now
Device Bus = machine truth
Chronicle = remembered meaning
Mycelix/Holochain = civilization memory
```

## Final Principle

Symtropy should not make every packet sacred.

It should make the right outcomes meaningful.

**Fast bodies.
Deterministic machines.
Signed memory.
Forkable worlds.**
