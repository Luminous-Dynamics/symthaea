# MULTIPLAYER_TRUTH_MODEL.md

# Symtropy Multiplayer Truth Model

## Version 0.1 — Ephemeral Action, Durable History

## Purpose

This document defines how Symtropy handles multiplayer truth across real-time gameplay, deterministic automation, Field Deck identity, civic governance, worldline history, and eventual Mycelix/Holochain-style agent-centric persistence.

Symtropy should not force every part of the game into one global consensus layer.

That would destroy performance, responsiveness, and playability.

Instead, Symtropy uses different truth layers for different kinds of events.

Fast action stays local and responsive.

Deterministic infrastructure uses replayable transactions.

Civic and historical events become signed records.

Worldlines preserve durable meaning.

## Core Principle

**Ephemeral action, durable history.**

A firefight does not need to become a blockchain.

A public water override does.

A player’s footstep is ephemeral.

A public vote is durable.

A bullet impact may be temporary.

A restored waterworks, dead NPC, recovered black box, or signed Archive Witness event becomes history.

## Design Mantra

**Real-time locally.
Deterministic regionally.
Agent-centric civically.
Chronicle globally.**

## Four Truth Layers

Symtropy should separate truth into four layers.

## 1. Local Real-Time Truth

This layer handles fast gameplay.

Used for:

* player movement
* first-person combat
* bullets
* melee
* drones
* physics impulses
* immediate damage
* doors animating
* vehicles
* short combat encounters
* local NPC movement
* moment-to-moment tool use

Implementation:

* Bevy simulation
* local shard authority
* rollback/prediction where needed
* fast P2P or host-authoritative local session
* not stored as permanent civic history by default

This layer must be fast above all else.

It should not wait for DHT gossip, global consensus, or civic validation.

## 2. Device Transaction Truth

This layer handles deterministic infrastructure changes.

Used for:

* Device Bus reads/writes
* SymLogic execution
* WASM microcontroller execution
* staged output queues
* atomic script transactions
* out-of-fuel rollback
* device cooldowns
* command chatter faults
* permission checks
* public infrastructure control
* local machine authority

Implementation:

* custom Symtropy Device Bus
* deterministic integer fuel
* stable ordering
* content-addressed scripts
* atomic tick transactions
* local transaction logs
* deterministic replay where required

This layer answers:

**What did the machine accept as a valid state change?**

## 3. Chronicle / Civic Truth

This layer handles meaningful events that should be remembered.

Used for:

* public votes
* laws
* Archive Witness signatures
* water restoration
* credentials
* script installs
* public infrastructure changes
* treaty events
* NPC deaths
* faction schisms
* settlement founding
* public override abuse
* recovered Ghost records
* worldline branch declarations

Implementation:

* signed events
* append-only records
* Field Deck source chains
* local Chronicle backend at first
* Mycelix/Holochain bridge later
* content-addressed event payloads

This layer answers:

**What should society remember?**

## 4. Worldline / Confluence Truth

This layer handles large-scale persistence and cross-world identity.

Used for:

* worldline forks
* migrations
* Confluence events
* settlement charters
* planetary treaties
* archive replication
* worldline ancestry
* major faction transformations
* planetary translation authority
* cross-world blueprint legitimacy

Implementation:

* agent-centric network
* Mycelix/Holochain-style persistence
* source chains
* DHT validation
* asynchronous history exchange
* not in the real-time hot path

This layer answers:

**What future does this worldline claim as valid?**

## What Must Not Be Put on Chain

Do not put these directly into civic/source-chain history:

* every footstep
* every aim update
* every bullet
* every animation frame
* every temporary physics impulse
* every raw terminal keystroke
* every UI cursor move
* every local prediction correction
* every short-lived combat object

These belong to Local Real-Time Truth.

Only outcomes become durable when they matter.

## What Should Become History

These should become Chronicle or civic records:

* public water restored
* pump override approved
* Archive Witness signed
* script installed on public infrastructure
* machine core recovered
* Ghost terminal record preserved
* settlement vote passed
* emergency power declared
* emergency power revoked or abused
* NPC leader killed
* worldline fork chosen
* faction charter created
* Confluence signal accepted

## Example: Old Waterworks Mission

During the mission:

* players move
* fight Null drones
* dodge attacks
* repair cables
* use tools
* open panels
* carry parts
* patch terminals

This is Local Real-Time Truth.

When the Systems Operator patches into the waterworks terminal:

```text id="hr88sz"
sym-dev read water.tank_0.level
sym-dev write water.pump_1.enabled true
```

This becomes Device Transaction Truth.

When the old law denies access:

```text id="8pr30k"
DENIED: DEAD_AUTHORITY_LOCK
```

That is a deterministic device event.

When the player requests an Archive Witness and restores public override:

```text id="ajzwtz"
ARCHIVE_WITNESS_SIGNED
PUMP_1_ENABLED
WATER_FLOW_RESTORED
```

That becomes Chronicle / Civic Truth.

After the mission, the world remembers:

```text id="qmsrwp"
The Waterworks were restored under Archive Witness after the dead emergency authority was overturned.
```

That is Durable History.

## Field Deck Source Chains

Every Field Deck should maintain a local append-only source chain.

This is the player’s portable identity and event history.

A Field Deck source chain is not necessarily full Holochain in v0.1.

It should be designed so it can map cleanly to Holochain/Mycelix later.

## Source Chain Entry

Conceptual structure:

```text id="rpdkvb"
DeckSourceEntry {
  deck_id
  agent_id
  worldline_id
  previous_hash
  event_hash
  event_type
  timestamp_or_chronicle_tick
  signature
}
```

## Example Event Types

```text id="5mbr4s"
ScriptImported
ScriptInstalled
DevicePatched
PublicVoteCast
ArchiveWitnessRequested
ArchiveWitnessSigned
CredentialGranted
CredentialRevoked
BlueprintMounted
WaterPumpOverrideRestored
ChronicleEntryAccepted
WorldlineForkJoined
```

## Why Field Deck Chains Matter

The Field Deck is the player’s root of trust.

It records:

* what the player signed
* what credentials they received
* what public votes they cast
* what scripts they installed
* what devices they patched
* what witnesses they requested
* what worldline branches they joined

This makes identity portable without requiring one central server.

## Custom Inner Layer, Holochain Outer Layer

Symtropy should not embed full Holochain directly into the real-time gameplay loop.

Recommended architecture:

```text id="r1vixo"
Bevy Real-Time Simulation
  ↓
Symtropy Device Transaction Layer
  ↓
Local Chronicle Backend
  ↓
Mycelix / Holochain Bridge
  ↓
Worldline / Civic Persistence
```

## Why Custom Inner Layer

The real-time and deterministic infrastructure layers need:

* frame responsiveness
* atomic tick transactions
* out-of-fuel rollback
* device cooldowns
* stable ordering
* script fuel counting
* combat responsiveness
* direct Bevy integration

These are game-engine concerns.

They should be controlled by Symtropy.

## Why Mycelix / Holochain Outer Layer

The civic and worldline layers need:

* agent identity
* source chains
* signed records
* validation rules
* DHT persistence
* decentralized governance
* cross-world history
* portable credentials
* asynchronous replication

These are agent-centric network concerns.

They fit Mycelix/Holochain well.

## Boundary Rule

**Holochain/Mycelix validates history.
The Device Bus validates machine transactions.
Bevy validates moment-to-moment gameplay.**

Do not collapse these responsibilities.

## Deterministic Fuel

WASM and SymLogic automation must never use wall-clock time to decide execution budget.

Fuel must be:

* integer-based
* deterministic
* architecture-agnostic
* counted by runtime operations
* identical across peers
* independent of host CPU speed

Bad:

```text id="zbi5kw"
Run script for 2 milliseconds.
```

Good:

```text id="tavy9f"
Run script for 20,000 deterministic fuel units.
```

If fuel runs out, every peer must reach the same result.

## Out-of-Fuel Rule

If a script exhausts fuel:

* rollback all staged writes for that tick
* emit deterministic fault
* apply no partial device state
* show readable Field Deck warning
* optionally trigger maintenance or audit event

Example:

```text id="g2heti"
SCRIPT FAULT
water_controller.sym
EXIT 137: OUT OF FUEL
WRITES ROLLED BACK
NO DEVICE STATE CHANGED
```

This prevents divergent world states.

## Atomic Device Transactions

Every script execution creates a transaction.

```text id="jzqafu"
begin transaction
  read deterministic snapshot
  compute within fuel budget
  stage writes
if success:
  commit staged writes
else:
  rollback staged writes
  emit fault
```

Scripts never mutate world state directly.

## Stable Device Ordering

All device writes must apply in deterministic order.

Suggested order:

```text id="78txz5"
device_id
field
priority
authority_level
script_id
transaction_id
```

Avoid nondeterministic hash map iteration.

Avoid host thread order.

Avoid wall-clock arrival order.

## Share Mode

Field Deck Share Mode should not stream pixels.

Streaming video from one player’s Deck screen to teammates would waste bandwidth and create synchronization issues.

Instead, Share Mode streams serialized UI state.

## Share Mode Payload

Example:

```json id="u2xxuq"
{
  "screen": "diagnostics",
  "device": "water.pump_1",
  "fields": {
    "status": "LOCKED",
    "power": "OFF",
    "authority": "DEAD_AUTHORITY_LOCK",
    "null_signal": 0.42
  },
  "cursor": [12, 4],
  "selected": "request_archive_witness",
  "privacy_mask": ["credentials", "identity_key"]
}
```

Each teammate renders the amber vector interface locally.

## Benefits

UI-state sharing is:

* low bandwidth
* privacy-aware
* deterministic-friendly
* easier to localize
* easier to mask
* compatible with accessibility modes
* robust under poor network conditions

## Privacy Modes

Share Mode should support masking:

* credentials
* identity keys
* medical data
* private messages
* faction secrets
* restricted Archive records
* hidden vote choices

A shared screen may show:

```text id="zwwx0t"
PRIVATE FIELD MASKED
CREDENTIAL VIEW ACTIVE
```

## Co-op Interface Visibility

Different interfaces have different visibility.

## Public Terminals

Visible to nearby players by default.

Used for:

* settlement maps
* public logs
* votes
* public diagnostics
* command tables
* workshop screens

## Personal Field Deck

Visible physically, but detailed reading requires:

* shoulder proximity
* share permission
* public screen mode
* squad mirror
* or faction law requirement

## Sensitive Screens

Masked unless explicitly shared or lawfully disclosed.

## Combat Windows

Real-time combat should be compressed into durable history only after meaningful outcomes.

Example:

```text id="gkyjqp"
Combat Window:
  squad enters waterworks
  fights drones
  repairs conduit
  loses one NPC
  recovers machine core
  restores pump
```

Durable Chronicle summary:

```text id="ztudlf"
The Firstlight squad restored the Old Waterworks after a Null incursion.
Archivist Mara witnessed the override.
Technician Ivo was wounded.
Machine core recovered.
```

This preserves meaning without logging every bullet.

## Chronicle Transaction

A Chronicle transaction may include:

```text id="zsqc9s"
event_type
worldline_id
region_id
participants
device_events
npc_outcomes
items_recovered
scripts_installed
witnesses
source_hashes
signatures
summary_text
raw_event_refs
```

## Validation Levels

Not every event needs the same validation strictness.

## Soft Events

Flavor, local logs, low consequence.

Examples:

* player note
* minor discovery
* private journal
* ambient NPC comment

Validation:

* local only
* no broad consensus

## Medium Events

Affect settlement or faction.

Examples:

* device repair
* local trade
* script install
* public work order

Validation:

* local witnesses
* device logs
* source-chain signatures

## Hard Events

Affect worldline identity.

Examples:

* public vote
* law passed
* faction split
* settlement founded
* worldline fork
* Confluence treaty

Validation:

* quorum
* Archive Witnesses
* Mycelix/Holochain backend
* stronger signature rules

## Forking

Worldline forks should not be treated as database failure.

They are a core feature.

A fork may occur when:

* factions split
* players reject a law
* governance fails
* Confluence is disputed
* a settlement chooses incompatible future
* a modded ruleset diverges
* rollback is socially impossible

Agents carry their source chains into the new branch.

The worldline records ancestry.

## Confluence

Confluence is the partial or full reconciliation of worldlines.

It should merge:

* source-chain histories
* settlement records
* faction treaties
* Archive records
* device history summaries
* NPC continuity records
* player credentials

It should not try to perfectly merge every raw real-time event.

Confluence reconciles meaning, not every footstep.

## High-Entropy Fusion Zones

When worldlines merge with contradictions, the game can express them physically.

Examples:

* duplicate ruins
* conflicting land claims
* ghost structures
* unstable devices
* NPCs with contradictory memories
* two laws claiming authority over one pump
* Archive disputes
* timeline scars

This turns database conflict into gameplay.

## Local Chronicle Backend

For Seedworks v0.1, use a local backend.

Suggested interface:

```rust id="w9u0zz"
trait ChronicleBackend {
    fn append_event(&self, event: ChronicleEvent) -> Result<EventHash>;
    fn get_event(&self, hash: EventHash) -> Result<ChronicleEvent>;
    fn verify_event(&self, hash: EventHash) -> Result<VerificationStatus>;
    fn list_agent_events(&self, agent: AgentId) -> Result<Vec<EventHash>>;
}
```

Implementations:

```text id="c5oeaq"
LocalChronicleBackend
FileChronicleBackend
MockP2PChronicleBackend
MycelixHolochainChronicleBackend
```

This allows v0.1 to ship without full network complexity.

## Seedworks v0.1 Scope

Implement:

* local real-time simulation
* fake terminal
* real Device Bus
* deterministic integer fuel placeholder
* staged device writes
* atomic rollback
* Field Deck source-chain mock
* local Chronicle backend
* one Archive Witness event
* one water restoration event
* one public vote event
* Share Mode payload mock, not video
* no official Holochain hot-path integration

Do not implement yet:

* full DHT
* full Holochain conductor
* planetary MycelixNet
* real Confluence
* external Airlock
* full source-chain migration
* cross-worldline validation

## Future Integration Path

## Phase 1 — Local Truth

Build fun local loop.

* Field Deck
* Device Bus
* waterworks mission
* local Chronicle

## Phase 2 — Deterministic Infrastructure

Add more devices and scripts.

* SymLogic
* WASM fuel
* content-addressed scripts
* script audits

## Phase 3 — Mock P2P Worldline

Add multiplayer history mock.

* signed events
* shared Chronicle
* player source chains
* local validation

## Phase 4 — Mycelix / Holochain Bridge

Add agent-centric backend.

* credentials
* public laws
* archives
* settlement records
* faction identity

## Phase 5 — Worldline Forks

Add explicit branching.

* fork ancestry
* migration
* divergent histories
* branch credentials

## Phase 6 — Confluence

Add worldline reconciliation.

* treaty merge
* Archive disputes
* high-entropy fusion zones
* contradictory histories as gameplay

## Final Architecture Statement

Symtropy is not one kind of network.

It is layered truth.

* The body acts in real time.
* The machine commits deterministic transactions.
* The Deck signs identity.
* The settlement remembers civic events.
* The worldline preserves history.
* Confluence reconciles meaning.

The player experiences this as one world.

The engine treats it as many kinds of truth.

## Final Principle

A multiplayer world does not need every moment to be permanent.

It needs the right moments to become history.

**Ephemeral action.
Deterministic infrastructure.
Signed memory.
Forkable futures.**
