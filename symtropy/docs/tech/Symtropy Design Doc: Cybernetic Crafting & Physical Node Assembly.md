# Symtropy Design Doc: Cybernetic Crafting & Physical Node Assembly

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Working Title

**Crafting as Registration**

## Core Thesis

Crafting in *Symtropy* is not a magic menu where raw materials become objects.

Crafting is the process of turning matter into accountable infrastructure.

A built object is not truly part of the world until it is:

```text
physically assembled
powered
registered
diagnosed
authorized
governed
remembered
```

A pipe patch, pump controller, valve, turret, battery, greenhouse node, or sensor mast is not only a mesh.

It is a physical device, a software node, a civic object, and a liability.

Core rule:

```text
A crafted object is not finished when it exists.
It is finished when the settlement can account for it.
```

---

# 1. Design Position

## Benchmark

*Space Engineers* makes construction satisfying because building is physical.

The player places a frame, brings materials, welds it into being, and watches a structure emerge in space.

*Symtropy* should preserve that tactile satisfaction, but not become a generic block-building sandbox.

The difference:

```text
Space Engineers:
Build a machine by assembling blocks.

Symtropy:
Build a machine by assembling matter, registering authority, and accepting consequence.
```

## Symtropy Difference

Every constructed object must answer:

```text
What is it?
Who built it?
Who may operate it?
What does it affect?
What law governs it?
What happens if it fails?
Who is responsible?
```

This turns construction into survival gameplay, civic gameplay, and infrastructure simulation at the same time.

---

# 2. The Cybernetic Crafting Loop

The full crafting loop has six stages.

```text
1. Acquire blueprint
2. Stage materials
3. Place physical frame
4. Assemble / weld / seal
5. Initialize software node
6. Register civic authority
```

The player should feel each stage physically.

---

## Stage 1: Acquire Blueprint

High-tier machines require concrete blueprint artifacts.

Blueprints are not abstract unlocks. They exist as physical media:

```text
data cartridges
etched repair cards
archive disks
machine-worn schematics
signed civic templates
black-market firmware slugs
field-expedient patch diagrams
```

A blueprint has provenance.

Example blueprint metadata:

```json
{
  "blueprint_id": "bp.water.patch_conduit.mk0",
  "name": "Patch Conduit Mk0",
  "class": "waterworks_repair",
  "source": "Old Waterworks Archive",
  "license": "commons_repair_use",
  "safety_profile": "low_pressure_only",
  "requires_witness": false,
  "firmware_seal": "open",
  "known_faults": ["leak_under_high_pressure", "corrosion_after_30_days"]
}
```

## Blueprint Politics

Different factions treat blueprints differently.

### Utility Sovereign Blueprint

```text
high reliability
opaque firmware
subscription locks
remote disable risk
warranty enforcement
possible vendor capture
```

### Mutualist Assembly Blueprint

```text
open-source
repairable
requires public safety witness
slower certification
trusted by commons factions
```

### Quarantine Authority Blueprint

```text
biosecure
restricted deployment
heavy logging
locks under uncertain contamination
```

### Black-Market Blueprint

```text
fast
cheap
unverified
possible Null contamination
possible hidden command channels
```

Design rule:

```text
Blueprints are political objects.
```

---

## Stage 2: Stage Materials

Materials must physically exist in the settlement logistics system.

No invisible inventory abstraction for major infrastructure.

Materials may be stored in:

```text
tool pouch
field crate
cargo locker
fabricator bay
conveyor line
salvage pallet
settlement depot
```

Each material may have quality and history:

```text
steel plate
copper coil
ceramic seal
valve actuator
sensor diode
firmware chip
biofilter membrane
insulation braid
```

Material state matters:

```text
corroded
clean
irradiated
contaminated
salvaged
certified
counterfeit
faction-marked
```

Example:

```json
{
  "material_id": "mat.copper_coil.salvaged",
  "condition": "oxidized",
  "conductivity": 0.72,
  "contamination": "low",
  "source": "ruined_pump_house",
  "faction_claim": "unregistered_salvage"
}
```

Design rule:

```text
Materials remember where they came from.
```

---

## Stage 3: Place Physical Frame

The player projects a localized wireframe using the Field Deck or structural assembly tool.

The frame is not freeform creative-mode building.

It must attach to valid anchors:

```text
pipe socket
wall bracket
floor mount
device rail
valve flange
power trunk
waterworks junction
foundation plate
```

For Seedworks v0.1, only allow repair frames.

Examples:

```text
Patch Conduit Mk0
Valve Handle Replacement
Sensor Mast Tripod
Temporary Power Bridge
Filter Housing Clamp
```

The player sees:

```text
ghost frame
anchor points
alignment guides
missing materials
pressure warnings
legal warnings
```

Example warning:

```text
ANCHOR VALID.
PIPE PRESSURE: LOW.
MATERIALS: 3/4 PRESENT.
CIVIC STATUS: EMERGENCY REPAIR PERMITTED.
```

Design rule:

```text
Early crafting should repair the world before it expands the world.
```

---

## Stage 4: Assemble / Weld / Seal

The player physically completes the frame.

Possible tools:

```text
hand welder
sealant injector
torque wrench
plasma cutter
fiber splice tool
ceramic patch press
biofilm applicator
```

The interaction should be tactile:

```text
hold tool against seam
manage heat
watch material fuse
listen for seal tone
avoid overburn
tighten bolts in sequence
clear rust before attachment
```

Quality depends on player action, tool condition, and material quality.

Example assembly outputs:

```text
clean seal
rough seal
leaky seal
overheated joint
misaligned bracket
temporary fix
certified repair
```

Design rule:

```text
Crafting quality should come from touch, not only ingredient count.
```

---

## Stage 5: Initialize Software Node

A newly assembled device is physically present but logically dead.

It must be initialized into the local Device Bus.

The player plugs a patch cable into the device port.

Field Deck output:

```sh
$ read /dev/sym/water/unmapped_node_7

STATUS: UNINITIALIZED
POWER_FEED: DETECTED
FLOW_CONTACT: DETECTED
AUTHORITY: NULL_LOGIC_EMPTY
PRESSURE_CLASS: LOW
FIRMWARE: NONE
```

The player initializes it:

```sh
$ sym-dev initialize /dev/sym/water/unmapped_node_7 --name patch_conduit_alpha

NODE INITIALIZED.
DEVICE CLASS: WATER_PATCH_CONDUIT
AUTHORITY REQUIRED: LOCAL_REPAIR_TOKEN
SAFETY PROFILE: LOW_PRESSURE_ONLY
```

The node becomes visible:

```text
/dev/sym/water/patch_conduit_alpha
```

Now the object can publish and receive state.

Example:

```json
{
  "node": "/dev/sym/water/patch_conduit_alpha",
  "status": "ACTIVE",
  "seal_quality": 0.81,
  "flow_delta": 0.36,
  "leak_risk": "medium",
  "authority": "emergency_repair_token",
  "operator": "player",
  "chronicle_pending": true
}
```

Design rule:

```text
A machine is not alive to the settlement until it has a name on the bus.
```

---

## Stage 6: Register Civic Authority

Some devices require civic authorization before operation.

Examples:

```text
water valves
defense turrets
quarantine doors
habitat pressure systems
species release pods
power distribution trunks
medical fabricators
```

Authorization can come from:

```text
emergency token
settlement charter
faction permit
Archive Witness
temporary repair license
Rights Floor override
public vote
operator credential
```

Example:

```sh
$ sym-civic authorize /dev/sym/water/patch_conduit_alpha --token emergency_repair_token

AUTHORIZATION ACCEPTED.
SCOPE: TEMPORARY_REPAIR
DURATION: 72 HOURS
REVIEW REQUIRED: YES
```

This creates future gameplay.

A temporary repair may later require:

```text
inspection
permanent certification
faction hearing
material replacement
decommissioning
legal review
```

Design rule:

```text
A repair can solve an emergency and create a political obligation.
```

---

# 3. Logistics as Device Bus

Cargo systems are also devices.

Conveyor lines, pumps, bins, sorters, fabricators, and depots should exist as physical/logical bus nodes.

Example paths:

```text
/dev/sym/logistics/conveyor_line_4
/dev/sym/logistics/sorter_alpha
/dev/sym/storage/salvage_bin_2
/dev/sym/fabricator/workbench_1
/dev/sym/water/pump_main
/dev/sym/power/battery_stack_3
```

## Logistics Faults

Because logistics are bus-connected, they can fail logically and physically.

Example failures:

```text
COMMAND_CHATTER
ROUTE_CONFLICT
MATERIAL_MISMATCH
SORTER_LOOP
JAMMED_LINE
POWER_SPIKE
AUTHORITY_DENIED
NULL_SCRIPT_ECHO
```

Example Field Deck reading:

```text
SCAN:
Storage sorter cycling rapidly.

DIAG:
Command chatter detected between sorter_alpha and conveyor_line_4.

CIVIC:
Automation exceeds local energy budget.

NULL:
Routing loop resembles prior Null Logic failure.
```

Gameplay consequence:

```text
materials stop moving
fabricator stalls
power spikes
pipe repair delayed
fire risk increases
operator must manually clear jam
```

Design rule:

```text
Automation is infrastructure. Infrastructure can argue with itself.
```

---

# 4. Panic Drop

During initialization, welding, or cable-linked diagnostics, the player is vulnerable.

Movement may be restricted by:

```text
patch cable
welding stance
tool bracing
open panel
active diagnostic session
high-voltage safety lock
```

If danger appears, the player can trigger a Panic Drop.

## Panic Drop Behavior

```text
tool line disconnects
Field Deck drops to chest lanyard
cable snaps free or retracts
current operation aborts
node enters safe state
player regains weapon/tool control
repair quality may suffer
```

Example:

```text
WARNING:
Null Choir drone signature detected.

HOLD: continue initialization.
TAP: panic drop.
```

Panic Drop should be dramatic but forgiving.

It makes crafting tactically tense without turning it into punishment.

Design rule:

```text
The player should fear being interrupted, not fear using the system.
```

---

# 5. Seedworks v0.1 Crafting Scope

The first playable slice should not implement full base building.

It should implement one complete repair-crafting loop.

## MVP Crafting Object

```text
Patch Conduit Mk0
```

## MVP Scenario

A broken pipe segment prevents the player from interacting successfully with the main water pump console.

The player must:

```text
1. Scan broken pipe.
2. Find or receive Patch Conduit Mk0 blueprint.
3. Gather required materials.
4. Project repair frame onto pipe break.
5. Weld or seal conduit into place.
6. Plug Field Deck into conduit port.
7. Initialize node.
8. Authorize temporary repair.
9. Run pump diagnostic.
10. Restore partial water flow.
11. Trigger Chronicle record.
```

## MVP Materials

```text
2 steel bands
1 ceramic seal
1 copper contact strip
1 firmware tab
```

## MVP Tools

```text
Field Deck Mk0
hand welder
sealant injector
patch cable
```

## MVP Device Path

```text
/dev/sym/water/patch_conduit_alpha
```

## MVP Field Deck Output

```text
SCAN:
Pipe fracture detected.
Flow interrupted.
Manual patch required.

DIAG:
Patch Conduit Mk0 compatible.
Seal quality will affect pressure stability.

CIVIC:
Emergency repair permitted under temporary water access clause.

NULL:
Automated pump restart blocked until local flow path is verified.
```

## MVP Chronicle Lines

If clean repair:

```text
The player gave the waterworks a new vein and named it before the settlement.
```

If rough repair:

```text
The pipe held, but the settlement could hear the weakness in the seal.
```

If unauthorized bypass:

```text
The player made the water move before the law agreed it should.
```

---

# 6. What To Defer

Do not build these in v0.1:

```text
freeform base building
large block grids
turret automation
complex conveyors
factory production chains
multiplayer construction rights
structural collapse simulation
full WASM scripting
blueprint economy
subscription firmware systems
```

These are later systems.

Seedworks v0.1 should prove the core sentence:

```text
To repair infrastructure, the player must touch matter, initialize logic, and confront authority.
```

---

# 7. Crafting Design Rules

## Rule 1: Repair Before Expansion

The first crafting loops should repair broken systems, not create sprawling bases.

```text
repair pipe
restore valve
patch cable
mount sensor
seal leak
replace fuse
```

## Rule 2: Every Built Object Has a Bus Identity

If the object changes settlement state, it needs a device path.

```text
No invisible infrastructure.
```

## Rule 3: Blueprints Have Provenance

A blueprint should carry history, license, and trust.

```text
No neutral schematics.
```

## Rule 4: Automation Has Failure Modes

Conveyors, valves, pumps, and sorters can misbehave.

```text
A system that moves matter can create politics.
```

## Rule 5: The Field Deck Is the Workbench

The most important crafting interface is not a menu.

It is the Field Deck touching the world.

```text
Crafting is diagnosis made physical.
```

---

# 8. Final Line

```text
A machine is not built when the weld cools.
It is built when the world knows what it is allowed to do.
```
# Addendum: Assembly Mechanics, Quality Propagation, and Infrastructure Consequence

## Purpose

The original Cybernetic Crafting document defines the correct thesis:

```text
Crafting is registration.
```

This addendum makes the physical assembly stage implementable.

The crafting loop must prove one complete chain:

```text
material condition
→ assembly procedure
→ repair quality
→ Device Bus performance
→ civic consequence
→ Chronicle memory
```

If this chain works, *Symtropy* has a crafting system.

If it does not, crafting risks becoming either a generic menu, a shallow QTE, or decorative interaction.

---

# 1. Stage 4 Revision: Assembly as Diagnostic Repair Procedure

The assembly stage should not be a conventional crafting minigame.

It should be a short, tactile, first-person repair procedure with visible mechanical steps.

## MVP Repair Procedure: Patch Conduit Mk0

The player repairs a fractured pipe using the following sequence:

```text
1. Clean contact surface.
2. Align patch frame.
3. Brace steel bands.
4. Seat ceramic seal.
5. Apply controlled weld heat.
6. Cool and pressure-test.
7. Inspect seal result.
```

Each step should be simple, physical, and readable.

The goal is not to make the player perform expert engineering.

The goal is to make the player feel that repair quality emerges from touch, attention, material condition, and pressure.

---

# 2. Repair Step Mechanics

## Step 1: Clean Contact Surface

The player uses a scraper, brush, or cutter to remove rust, biofilm, or debris from the pipe.

Variables affected:

```text
surface_cleanliness
biofilm_remaining
corrosion_remaining
```

Possible outcomes:

```text
clean contact
partial contact
contaminated contact
```

Gameplay effect:

```text
Poor cleaning reduces seal quality and increases contamination risk.
```

Field Deck example:

```text
DIAG:
Contact surface contaminated.
Seal adhesion penalty likely.
```

---

## Step 2: Align Patch Frame

The player projects the Patch Conduit Mk0 frame and aligns it to anchor points.

Variables affected:

```text
alignment_error
anchor_validity
frame_stress
```

Possible outcomes:

```text
precise alignment
acceptable alignment
misaligned frame
```

Gameplay effect:

```text
Poor alignment increases leak risk and mechanical stress.
```

Field Deck example:

```text
SCAN:
Anchor points valid.
Alignment drift: 11 degrees.
```

---

## Step 3: Brace Steel Bands

The player tightens steel bands around the conduit.

Variables affected:

```text
brace_tension
tension_balance
band_integrity
```

Possible outcomes:

```text
balanced brace
over-tightened brace
loose brace
uneven brace
```

Gameplay effect:

```text
Over-tightening risks cracking the ceramic seal.
Under-tightening risks pressure leakage.
```

Field Deck example:

```text
DIAG:
Brace tension uneven.
Pressure cycling may loosen patch.
```

---

## Step 4: Seat Ceramic Seal

The player places the ceramic seal into the joint.

Variables affected:

```text
seal_fit
seal_condition
contamination_between_layers
```

Possible outcomes:

```text
clean seat
rough seat
damaged seal
```

Gameplay effect:

```text
Seal quality determines short-term pressure stability.
```

Field Deck example:

```text
SCAN:
Ceramic seal seated.
Micro-gap detected on lower edge.
```

---

## Step 5: Apply Controlled Weld Heat

The player welds or heat-fuses the patch.

This should not be a twitch QTE.

It should be a short heat-control interaction.

The player holds the tool on the seam while watching a diegetic heat band or listening for audio pitch.

Variables affected:

```text
heat_applied
heat_variance
overburn
underfusion
tool_stability
```

Possible outcomes:

```text
clean weld
cold weld
overheated joint
uneven weld
```

Gameplay effect:

```text
Cold welds fail under pressure.
Overheated joints damage material and firmware contact points.
```

Field Deck example:

```text
DIAG:
Thermal variance high.
Recommend slow pass across upper seam.
```

---

## Step 6: Cool and Pressure-Test

The player initiates a low-pressure test.

Variables affected:

```text
initial_leak_rate
pressure_stability
seal_resonance
flow_contact
```

Possible outcomes:

```text
pressure stable
minor seep
active leak
seal resonance warning
```

Gameplay effect:

```text
The test determines whether the node can be safely initialized.
```

Field Deck example:

```text
SCAN:
Low-pressure flow detected.
Leak rate within emergency tolerance.
```

---

## Step 7: Inspect Seal Result

The player receives a readable repair grade.

Repair grades:

```text
Certified Seal
Clean Emergency Seal
Rough Emergency Seal
Leaky Seal
Unsafe Seal
```

The grade should not be merely cosmetic.

It feeds directly into Device Bus performance and civic review.

---

# 3. Quality Propagation Chain

Repair quality should propagate through the whole system.

## Material Condition

Example material state:

```json
{
  "material": "copper_contact_strip",
  "condition": "oxidized",
  "conductivity": 0.72,
  "contamination": "low",
  "source": "salvaged_pump_house"
}
```

Material condition affects:

```text
assembly difficulty
seal reliability
node initialization stability
inspection outcome
future maintenance timer
```

## Assembly Quality

Player procedure creates a repair result:

```json
{
  "assembly_quality": 0.76,
  "surface_cleanliness": 0.82,
  "alignment_error": 0.09,
  "brace_tension_balance": 0.71,
  "weld_integrity": 0.79,
  "seal_fit": 0.73
}
```

## Node Performance

The repair result becomes Device Bus state:

```json
{
  "node": "/dev/sym/water/patch_conduit_alpha",
  "status": "active",
  "flow_delta": 0.34,
  "leak_risk": "medium",
  "pressure_limit": "low",
  "maintenance_due": "soon",
  "authority_scope": "temporary_repair"
}
```

## Civic Consequence

The repair result creates obligations:

```json
{
  "civic_status": "temporary_emergency_repair",
  "review_required": true,
  "inspection_deadline": "72_hours",
  "liability": "operator_and_settlement_shared",
  "faction_claims": [
    "Watershed Commons requests permanent restoration.",
    "Continuance Office requires inspection.",
    "Utility Sovereign disputes unauthorized modification."
  ]
}
```

## Chronicle Memory

The outcome is remembered:

```text
The player gave the waterworks a temporary vein, but the seal still carried the history of salvaged metal.
```

Design rule:

```text
Repair quality is not a score.
It is a future problem made visible.
```

---

# 4. Assembly Input Model

The repair interaction should use a small number of readable inputs.

Recommended inputs:

```text
hold tool steady
move along seam
rotate part into alignment
tighten brace
release heat before overburn
switch Field Deck mode
trigger pressure test
panic drop if threatened
```

Avoid:

```text
complex button combos
abstract QTE prompts
opaque progress bars
random failure rolls
long crafting animations with no player judgment
```

The player should understand why the repair succeeded or failed.

Design rule:

```text
A bad repair should feel diagnosable, not arbitrary.
```

---

# 5. Panic Drop and Quality Loss

Panic Drop should interrupt assembly without hard-failing the mission.

If the player triggers Panic Drop during repair:

Possible consequences:

```text
weld pass incomplete
brace tension uneven
seal contamination increases
node initialization aborted
tool cable damaged
repair grade capped
```

Example:

```text
PANIC DROP:
Operation aborted.
Field Deck safe.
Patch frame remains attached.
Weld integrity incomplete.
Resume possible after threat cleared.
```

Design rule:

```text
Panic Drop preserves player agency but leaves physical evidence of interruption.
```

---

# 6. Unauthorized Bypass Consequence

The unauthorized bypass outcome should become a real civic conflict, not a flavor line.

## Situation

The player initializes the patch conduit and restores water before receiving full civic authorization.

Immediate mechanical result:

```text
water flow restored
settlement pressure improves
public health risk decreases
legal status unresolved
```

Immediate civic result:

```text
Continuance Office flags repair
Utility Sovereign may contest control
Watershed Commons may support action
Quarantine Authority may demand inspection
```

## Possible Follow-Up States

```text
Approved After Review
Retroactive Fine
Temporary Token Revoked
Public Hearing Triggered
Faction Claim Filed
Node Placed Under Watch
Repair Converted to Commons Asset
Repair Seized by Private Authority
```

## Example Chronicle Event

```json
{
  "event_type": "UnauthorizedInfrastructureRepair",
  "node": "/dev/sym/water/patch_conduit_alpha",
  "operator": "player",
  "procedure": "emergency_bypass",
  "mechanical_result": {
    "water_flow": "restored_partial",
    "seal_quality": "rough_emergency_seal",
    "leak_risk": "medium"
  },
  "civic_result": {
    "authorization": "retroactive_review_required",
    "liability": "operator_shared",
    "faction_conflict": [
      "continuance_office",
      "watershed_commons",
      "utility_sovereign"
    ]
  },
  "chronicle_line": "The player made the water move before the law agreed it should."
}
```

Design rule:

```text
Unauthorized repair should be useful, understandable, and politically expensive.
```

---

# 7. Logistics Faults as Future System Preview

The logistics fault section should remain in the design bible, but it should be labeled as a future expansion.

For Seedworks v0.1, do not implement full conveyor automation.

## Deferred Logistics Systems

```text
conveyor routing
storage sorters
factory loops
automated material requests
multi-node material priority
Null-infected logistics chatter
settlement-scale cargo arbitration
```

## Keep for Future

```text
COMMAND_CHATTER
ROUTE_CONFLICT
MATERIAL_MISMATCH
SORTER_LOOP
JAMMED_LINE
POWER_SPIKE
AUTHORITY_DENIED
NULL_SCRIPT_ECHO
```

These are valuable, but they belong after the Patch Conduit loop proves:

```text
physical repair
Device Bus initialization
civic authorization
Chronicle memory
```

Design rule:

```text
Do not build the factory before the first pipe knows its name.
```

---

# 8. MVP Implementation Target

Seedworks v0.1 should prove one repair end-to-end.

## Required MVP

```text
one broken pipe
one patch conduit blueprint
four material types
one assembly procedure
one repair grade
one Device Bus node
one authorization decision
one Chronicle record
```

## Not Required Yet

```text
crafting trees
full material economy
conveyor networks
complex factory automation
large-scale base building
animal AI
full faction courts
multiplayer construction rights
```

## MVP Success Test

The prototype succeeds if a player can say:

```text
I found a broken water pipe.
I physically patched it.
My repair quality mattered.
The machine appeared on the system bus.
The law cared how I restored it.
The settlement remembered what I did.
```

---

# 9. Updated Final Principle

The original final line remains strong:

```text
A machine is not built when the weld cools.
It is built when the world knows what it is allowed to do.
```

Add the implementation principle:

```text
A repair is not a progress bar.
It is a chain of evidence.
```
