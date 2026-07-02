---

title: Device Bus Substrate Systems
status: implementation
milestone: seedworks-v0.1-to-v0.2
scope: substrate systems
owner: design/engineering
depends_on:

* SEEDWORKS_PLAYABLE_SLICE_SPEC.md
* SEEDWORKS_ARCHITECTURE.md
  recommended_path: docs/seedworks/01_milestone_v0_1_old_waterworks/DEVICE_BUS_SUBSTRATE_SYSTEMS.md

---

> **Code status (2026-07-02 review):** the `status: implementation` above does not hold — `crates/symtropy-device-bus/src/lib.rs` is a 20-line stub with no tests. This is substantially still a design document.

# Symtropy Design Doc: Device Bus Substrate Systems

## Power, Audio, and Proof-of-Repair

## Core Thesis

Symtropy treats infrastructure, law, computation, and memory as one continuum.

Therefore, power cannot be a binary resource.

Sound cannot be mere atmosphere.

Labor cannot be abstract money.

These systems must become auditable substrate layers on the Device Bus.

Core rule:

```text
Everything that sustains civilization must be measurable, interruptible, contestable, and remembered.
```

The three substrate systems are:

```text
/dev/sym/power/*
/dev/sym/audio/*
/dev/sym/labor/*
```

They should all follow the same grammar:

```text
physical measurement
→ Device Bus state
→ Field Deck interpretation
→ civic consequence
→ Chronicle record
```

---

# 1. Why These Systems Matter

The Seedworks v0.1 slice already proves:

```text
repair is physical
repair is registered
repair is witnessed
repair becomes precedent
```

The next substrate layer proves why repair can fail even when the player “did everything right.”

A pump may fail because power sagged.

A terminal may lie because command chatter shifted its relay cadence.

A player may be trusted in a new settlement because their prior repairs are cryptographically witnessed.

Design rule:

```text
The world should not ask “did the player press the repair button?”
It should ask “what physical, civic, and historical conditions made repair valid?”
```

---

# 2. Substrate System A: Thermodynamic Power Grid

## Device Path

```text
/dev/sym/power/*
```

Power is not a passive on/off flag.

Power is:

```text
voltage
load
line loss
heat
thermal bleed
battery stress
clock stability
brownout risk
device priority
```

Core rule:

```text
Computing is physical.
Power, heat, limits, risk.
```

---

## 2.1 Hybrid Power Model

Power attenuation should use a hybrid model.

Do not calculate line loss from raw Bevy coordinate distance every frame.

Do not make attenuation purely abstract from regional pressure vectors.

Use:

```text
physical cable graph
+ cached cable lengths
+ material / gauge / condition
+ load demand
+ environmental stress
+ regional pressure modifiers
= voltage, heat, line loss, device instability
```

Design rule:

```text
The cable should know how long it is.
The region should know why that length is dangerous today.
```

---

## 2.2 Authoring Layer

In Bevy, power lines may be authored as physical objects or logical connections.

Example connection:

```text
transformer_2 → pump_1
transformer_2 → conveyor_3
battery_bank_0 → camp_terminal_1
```

Initial length can be calculated from endpoint coordinates:

```text
length_m = distance(endpoint_a, endpoint_b)
```

After authoring, the power system stores the connection as a graph edge.

Do not recompute expensive physical relationships every frame unless the cable is moved, cut, repaired, extended, or damaged.

---

## 2.3 Simulation Layer

Example power edge:

```json
{
  "edge_id": "line_transformer_2_to_pump_1",
  "from": "/dev/sym/power/transformer_2",
  "to": "/dev/sym/water/pump_1",
  "length_m": 42.7,
  "material": "copper",
  "gauge": "medium",
  "condition": "oxidized",
  "insulation": "worn",
  "environment": "wet",
  "max_kw": 60,
  "current_kw": 45,
  "line_loss_kw_per_m": 0.031,
  "thermal_bleed": "high"
}
```

The graph should be deterministic and low-frequency.

Recommended update rate:

```text
power graph update: 2–10 Hz
visual effects/audio: frame-rate interpolated
critical failures: event-driven
```

Design rule:

```text
Power should feel continuous to the player but simulate as a stable graph.
```

---

## 2.4 Regional Pressure Vector

Regional pressure modifies grid behavior but does not replace physical graph truth.

Pressure vectors may include:

```text
storm_intensity
ambient_heat
settlement_demand
maintenance_neglect
sabotage_pressure
Null_pressure
emergency_rationing
faction_priority_load
```

Example:

```text
Base line loss comes from length, material, gauge, and condition.
Storm pressure increases instability.
Settlement scarcity increases load.
Null pressure increases command chatter risk.
```

Design rule:

```text
Geometry tells the system where loss can happen.
Pressure tells the system why today is worse.
```

---

## 2.5 Voltage Sag Bands

Voltage should degrade gradually.

Recommended bands:

```text
95–100%:
stable operation

90–95%:
minor warnings, noncritical jitter

75–90%:
clock drift, multi-tick transactions, pump efficiency loss

50–75%:
brownout mode, noncritical devices suspended

<50%:
fail-safe, crash, emergency shutdown, or dead-authority fallback
```

Field Deck example:

```text
NODE: /dev/sym/power/transformer_2/load
STATUS: OVERLOADED
VOLTAGE: 84%
LINE_LOSS: 1.4 kW/m
THERMAL_BLEED: HIGH
SCRIPT_CLOCK: DEGRADED
TRANSACTION_LATENCY: +3 TICKS

CONNECTED_DEVICES:
- /dev/sym/water/pump_1            draw: 45 kW
- /dev/sym/logistics/conveyor_3    draw: 12 kW
```

---

## 2.6 WASM / Script Runtime Impact

WASM microcontrollers should not become nondeterministic.

Do not slow them using real wall-clock time.

Instead, each controller receives deterministic cycles per simulation tick.

Example:

```json
{
  "controller": "/dev/sym/water/pump_1/controller",
  "voltage_percent": 84,
  "cycles_per_tick": 40,
  "nominal_cycles_per_tick": 64,
  "transaction_delay_ticks": 3,
  "state": "clock_drift"
}
```

Design rule:

```text
Voltage may slow computation.
It must not break determinism.
```

---

## 2.7 Power Failure Consequences

Low power may cause:

```text
pump startup failure
slow valve response
delayed Device Bus writes
terminal brownout
cargo manifest timeout
Field Deck charging limits
witness cartridge read failure
Null prompt persistence
repair-grade downgrade
```

Example:

```text
DIAG:
Patch conduit initialized.
Pump restart delayed by transformer voltage sag.

CIVIC:
Public override authorized, but infrastructure cannot yet execute command.
```

Design rule:

```text
A legitimate command can still fail physically.
```

---

# 3. Substrate System B: Audio as Active Bus Metric

## Device Path

```text
/dev/sym/audio/*
```

Sound is not only ambience.

Sound is machine testimony.

A machine’s pitch, vibration, rhythm, and silence can expose faults before visual UI does.

Core rule:

```text
Sound is diagnostics before it is atmosphere.
```

---

## 3.1 Auditable Acoustic Fields

Every major machine may expose acoustic telemetry.

Example:

```json
{
  "node": "/dev/sym/audio/pump_1",
  "source": "/dev/sym/water/pump_1",
  "acoustic_frequency_hz": 61.4,
  "vibration_amplitude": 0.72,
  "rhythm_pattern": "irregular_triplet",
  "bearing_wear": 0.63,
  "valve_drag": 0.41,
  "null_chatter_confidence": 0.18
}
```

This does not require a full acoustic physics simulation.

For v0.1, authored machine states can drive audio variables.

---

## 3.2 Field Deck Audio Readings

The player can use Field Deck modes to interpret sound.

### SCAN

```text
SCAN:
Pump vibration irregular.
Primary resonance below expected range.
```

### DIAG

```text
DIAG:
Valve cadence desynchronized from controller clock.
Bearing wear likely.
```

### ARCHIVE

```text
ARCHIVE:
Acoustic signature resembles 2113 authority-lock failure recordings.
```

### CIVIC

```text
CIVIC:
Machine testimony may support maintenance claim.
Expert witness recommended before adjudication.
```

### NULL

```text
NULL:
Command chatter possible but unconfirmed.
Rhythm pattern repeats at nonmechanical interval.
```

Design rule:

```text
The Field Deck should not replace listening.
It should teach the player what listening means.
```

---

## 3.3 Origin-Specific Audio Perception

Different origins should hear different things.

### Basin-Born Technician

```text
ORIGIN NOTE:
The pump is knocking too low.
That valve is dragging against load.
```

### Archive Apprentice

```text
ORIGIN NOTE:
The cadence matches archived failure pattern from the 2113 lock event.
```

### Corporate Utility Defector

```text
ORIGIN NOTE:
Relay rhythm resembles proprietary load-shedding firmware.
```

### Continuance Ghost Origin

```text
GHOST ORIGIN NOTE:
Irregular cadence may indicate unauthorized override stress.
Emergency lock may be preserving machine integrity.
```

Design rule:

```text
What the player hears should be shaped by what their life taught them to trust.
```

---

## 3.4 Sound of Failure

Machines should have identifiable acoustic states.

Examples:

```text
healthy pump:
low stable hum

oxidized bearing:
deep knocking undertone

valve drag:
slow scrape at cycle edge

voltage sag:
pitch droop under load

Null command chatter:
rhythmic relay clicking with unnatural repetition

dead authority loop:
steady repetition without adaptive response

manifest sorter jam:
stuttering gate clack and rollback chirp
```

Design rule:

```text
Expert players should eventually diagnose danger before the UI confirms it.
```

---

## 3.5 v0.1 Audio Scope

For Seedworks v0.1, implement only:

```text
pump hum stable
pump hum stressed
relay chatter loop
Field Deck text response to audio state
one origin-specific audio note
```

Do not implement:

```text
full acoustic propagation
complex sound occlusion
spectral analysis minigame
large audio-device network
```

Design rule:

```text
The first audio system should make one pump sound sick.
```

---

# 4. Substrate System C: Proof-of-Repair

## Device Path

```text
/dev/sym/labor/*
```

Proof-of-Repair is the portable legitimacy layer for maintenance labor.

It is not gold.

It is not generic social credit.

It is not a universal currency.

It is a signed, witnessed, non-fungible record that the player performed infrastructure work under specific conditions.

Core rule:

```text
Proof-of-Repair is not money.
It is portable legitimacy.
```

---

## 4.1 Why Traditional Currency Fails

Symtropy has:

```text
fragmented settlements
worldline forks
damaged archives
dead authorities
local charters
faction sovereignty
scarce infrastructure
contested truth
```

A universal coin would flatten the world.

A player should not buy trust with abstract numbers.

They earn access by carrying verifiable histories.

Design rule:

```text
The question is not “how rich are you?”
The question is “what can the world prove you repaired?”
```

---

## 4.2 Proof-of-Repair Receipt

When the player repairs infrastructure, the Device Bus may generate a signed receipt.

Example:

```json
{
  "receipt_type": "ProofOfRepair",
  "receipt_id": "por_firstlight_waterworks_0001",
  "site": "Old Waterworks",
  "node": "/dev/sym/water/patch_conduit_alpha",
  "work_type": "public_water_restoration",
  "repair_grade": "RoughEmergencySeal",
  "authority_basis": "ArchiveWitnessCartridge",
  "witnesses": [
    "ArchiveWitnessCartridge_03",
    "FirstlightPublicRepairCharter",
    "Mara"
  ],
  "risk_accepted": "inspection_required",
  "chronicle_event": "evt_00000142",
  "actor": "player_local",
  "transferability": "non_transferable_reputation",
  "issued_at": "2168-FirstlightBasin-local"
}
```

Design rule:

```text
A receipt should prove work, context, witnesses, risk, and consequence.
```

---

## 4.3 Field Deck Source Chain

Proof-of-Repair is committed to the player’s Field Deck source chain.

Example:

```text
FIELD DECK:
Proof-of-Repair committed.

SITE:
Old Waterworks

WORK:
Temporary public water restoration

STATUS:
Witnessed / inspection required

CIVIC VALUE:
Recognized by Firstlight Public Repair Charter
```

If the player dies before syncing, the receipt may become vulnerable.

This connects directly to death and source-chain recovery.

Design rule:

```text
Labor history is something you carry.
```

---

## 4.4 Trade and Access Effects

Proof-of-Repair may unlock:

```text
fuel trust discount
blueprint access
repair contract priority
settlement entry clearance
public tool library access
Archive Witness review priority
reduced suspicion at infrastructure gates
NPC trust changes
faction-specific offers
```

It should not behave as a generic spendable token.

Bad:

```text
Spend 3 repair coins to buy rifle.
```

Better:

```text
Merchant console recognizes prior public water restoration.
Fuel depot grants trusted technician access.
```

Field Deck example:

```text
CIVIC:
Proof-of-Repair recognized.

RECEIPT:
Old Waterworks public restoration.

LOCAL EFFECT:
Fuel cartridge access unlocked under repair-worker trust clause.
```

Design rule:

```text
Proof-of-Repair opens doors because someone believes the receipt, not because the receipt is money.
```

---

## 4.5 Receipt Risk

Receipts can be challenged.

Challenge reasons:

```text
witness missing
source chain damaged
repair later failed
receipt forged
Null contamination suspected
faction rejects authority basis
repair violated local charter
labor exploited others
```

Example:

```text
CIVIC:
Proof-of-Repair disputed.

Reason:
Repair restored water through unwitnessed manual bypass.

Local effect:
Technician trust bonus reduced.
Continuance review required.
```

Design rule:

```text
Reputation should remain accountable to evidence.
```

---

## 4.6 v0.1 Scope

Seedworks v0.1 should generate one Proof-of-Repair receipt at the end of the Old Waterworks slice.

Required fields:

```text
site
node
repair path
repair grade
authority basis
Chronicle event ID
witness status
```

Example v0.1 receipt:

```json
{
  "receipt_type": "ProofOfRepair",
  "receipt_id": "por_old_waterworks_v0_0001",
  "site": "Old Waterworks",
  "node": "/dev/sym/water/patch_conduit_alpha",
  "work_type": "temporary_public_water_restoration",
  "repair_grade": "RoughEmergencySeal",
  "authority_basis": "ArchiveWitnessCartridge",
  "chronicle_event": "evt_00000004",
  "witness_status": "partial_valid",
  "transferability": "non_transferable_reputation"
}
```

Do not implement full merchant redemption yet.

Just show:

```text
SOURCE CHAIN UPDATED:
Proof-of-Repair added.
```

Design rule:

```text
In v0.1, Proof-of-Repair should feel like a seed, not an economy.
```

---

# 5. Shared Data Pattern

All three substrate systems should produce Device Bus nodes.

## Power Node

```json
{
  "node": "/dev/sym/power/transformer_2",
  "status": "overloaded",
  "voltage_percent": 84,
  "thermal_bleed": "high",
  "connected_devices": [
    "/dev/sym/water/pump_1"
  ]
}
```

## Audio Node

```json
{
  "node": "/dev/sym/audio/pump_1",
  "status": "stressed",
  "frequency_hz": 61.4,
  "vibration_amplitude": 0.72,
  "pattern": "irregular_triplet"
}
```

## Labor Node

```json
{
  "node": "/dev/sym/labor/proof_of_repair/por_old_waterworks_v0_0001",
  "status": "committed",
  "site": "Old Waterworks",
  "repair_grade": "RoughEmergencySeal",
  "witness_status": "partial_valid"
}
```

Design rule:

```text
Different substrates. Same grammar.
```

---

# 6. Relationship to Chronicle

Chronicle records should reference substrate systems when they change history.

Examples:

```text
Power:
Transformer sag delayed public water restoration.

Audio:
Pump acoustic signature revealed hidden valve drag.

Labor:
Proof-of-Repair committed after witnessed restoration.
```

Example Chronicle event:

```json
{
  "event_type": "ProofOfRepairIssued",
  "site": "Old Waterworks",
  "actor": "player_local",
  "receipt_id": "por_old_waterworks_v0_0001",
  "repair_grade": "RoughEmergencySeal",
  "authority_basis": "ArchiveWitnessCartridge",
  "chronicle_line": "The settlement did not only receive water. It received proof of who carried the repair."
}
```

Design rule:

```text
The Chronicle records substrate changes when they alter legitimacy.
```

---

# 7. Relationship to Seedworks v0.1

For v0.1, implement only thin versions.

## Required

```text
one power warning
one audio diagnostic
one Proof-of-Repair receipt
```

## Optional

```text
voltage sag affecting command latency
origin-specific audio note
power state affecting pump restart delay
```

## Deferred

```text
full grid simulation
full acoustic propagation
merchant redemption
cross-settlement labor economy
worldline-fork receipt arbitration
```

Design rule:

```text
v0.1 should hint that the substrate is real without becoming a power-grid simulator.
```

---

# 8. Final Principles

```text
Power is not a toggle.
It is thermodynamic permission.

Sound is not decoration.
It is machine testimony.

Labor is not money.
It is witnessed transformation.

A cable has history.
A pump has a voice.
A repair leaves a receipt.

The Device Bus is where physics becomes accountable.
```

Final line:

```text
The settlement survived because the cable held, the pump spoke, and someone could prove they had carried the repair.
```
