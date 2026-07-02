---
title: Symtropy Resource Chains Game Design Document
status: canonical-draft
version: 0.1
milestone: seedworks-v0.1-to-v1.0
scope: resource chains, material provenance, logistics, fabrication, civic authorization, settlement metabolism, worldline history
recommended_path: docs/seedworks/00_canon/SYMTROPY_RESOURCE_CHAINS_GAME_DOC_V0_1.md
---

# Symtropy Resource Chains Game Design Document

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Working Title

**Matter Becomes History**

## Core Thesis

Resource chains in *Symtropy* are not generic production recipes.

They are the playable metabolism of civilization.

A resource is not complete when it is mined, looted, harvested, refined, or fabricated.

A resource becomes complete when the world can answer:

```text
Where did it come from?
Who touched it?
What did it repair?
Who controls it now?
Who was excluded by its movement?
What debt or legitimacy did it create?
What will fail if the chain breaks?
What will history remember about its use?
```

Core rule:

```text
A resource is matter under obligation.
```

Symtropy should therefore reject the shallow chain:

```text
ore -> ingot -> item
```

and replace it with:

```text
source -> extraction -> condition -> transport -> processing -> fabrication -> installation -> authorization -> operation -> consequence -> memory
```

The resource economy should make players feel that civilization is not an abstract spreadsheet.

It is water in pipes, charge in batteries, medicine in cold storage, seals on valves, food in kitchens, steel in bridges, testimony in archives, and trust in the hands that move it all.

---

# 1. Purpose

This document defines the first full resource-chain model for *Symtropy: Seedworks*.

It answers:

```text
What counts as a resource?
How do resources move through the world?
How do materials remember provenance?
How do resource chains connect to crafting, Device Bus, Field Deck, factions, settlement state, ecology, and Chronicle history?
Which chains belong in Seedworks v0.1?
Which chains should be deferred to v0.4 factory/logistics and v1.0 planetary systems?
How do we keep resource gameplay legible instead of overwhelming?
```

The design goal is not economic complexity for its own sake.

The goal is playable causality.

The player should understand:

```text
This pump failed because the seal was counterfeit.
The seal was counterfeit because the public workshop lacked certified ceramic.
The workshop lacked ceramic because the convoy road washed out.
The road washed out because the wetland was drained.
The wetland was drained because the old Utility Sovereign rerouted water under contract.
The contract is still enforced by dead authority.
Repairing the pump means reopening the argument.
```

That is a Symtropy resource chain.

---

# 2. Relationship to Existing Symtropy Systems

Resource chains should integrate with the following core systems.

## 2.1 First 30 Minutes

The opening Seedworks loop already implies a resource chain:

```text
storm
power instability
failing water pump
repair tool
salvage parts
fabricate or repair pump component
transport part to waterworks
fight drones
restore water
return to settlement
public vote
```

Resource-chain design should make this sequence systemic rather than scripted.

## 2.2 Cybernetic Crafting

Cybernetic crafting defines the principle that construction is not magic-menu conversion.

The full crafting loop is:

```text
1. Acquire blueprint
2. Stage materials
3. Place physical frame
4. Assemble / weld / seal
5. Initialize software node
6. Register civic authority
```

Resource chains supply the matter, provenance, logistics, and consequences that feed that loop.

## 2.3 Device Bus

Resource chains should expose major physical infrastructure as readable and writable systems.

Examples:

```text
/dev/sym/storage/salvage_bin_2
/dev/sym/logistics/conveyor_line_4
/dev/sym/fabricator/workbench_1
/dev/sym/water/patch_conduit_alpha
/dev/sym/power/battery_stack_3
/dev/sym/bio/nursery_willow_01
/dev/sym/civic/resource_allocation/water
```

A resource chain becomes fully Symtropy-native when it can be inspected, interrupted, repaired, authorized, audited, and remembered.

## 2.4 Settlement State Vector

Every major resource chain should affect one or more settlement metrics:

```text
power
water
food
health
repair
trust
legitimacy
safety
stress
entropy
ecology
knowledge
logistics
signal
harmony
```

A chain that only changes inventory count is too shallow.

## 2.5 Chronicle and Worldline History

Not every item movement becomes history.

But meaningful chain outcomes should become Chronicle events:

```text
public water restored
emergency medicine delivered
bridge reopened
contaminated materials exposed
illegal water diversion discovered
black-market firmware installed
species released into watershed
convoy lost
public repair certified
Utility Sovereign lock bypassed
```

Rule:

```text
Do not record every crate.
Record the crate that changed what society could become.
```

---

# 3. Resource Design Principles

## 3.1 No Neutral Resources

Every important resource should carry at least one of:

```text
condition
origin
ownership claim
contamination risk
faction mark
license restriction
civic status
ecological consequence
repair history
```

Example:

```json
{
  "resource_id": "mat.copper_contact_strip.salvaged",
  "condition": "oxidized",
  "conductivity": 0.72,
  "source_site": "old_pump_house",
  "faction_claim": "unregistered_salvage",
  "contamination": "low",
  "certification": "uncertified",
  "chronicle_relevance": "minor"
}
```

Design rule:

```text
Matter remembers.
```

## 3.2 Physical Before Abstract

Major resources should physically exist in the world.

They may be carried, stored, hauled, dropped, stolen, jammed, flooded, burned, contaminated, or witnessed.

Use invisible inventory only for small personal items or low-stakes abstractions.

Good:

```text
steel bands in a field crate
ceramic seal in a tool pouch
water tanks on a rover
medicine fridge in a clinic truck
seed tray in a nursery trailer
battery module on a pallet
```

Weak:

```text
+10 steel
+5 water
+3 medicine
```

## 3.3 Provenance Creates Gameplay

Where a resource came from should affect how factions interpret it.

Example:

```text
Salvaged corporate ceramic seal:
  + reliable material tolerance
  - disputed ownership
  - possible firmware compatibility lock

Public workshop ceramic seal:
  + trusted by commons factions
  + open repair record
  - slower production
  - lower early precision

Black-market ceramic seal:
  + immediate availability
  - hidden defect risk
  - possible Null contamination
```

Design rule:

```text
A material source is a political source.
```

## 3.4 Chains Must Fail Legibly

A chain is only interesting if the player can diagnose why it failed.

Failure should surface through Field Deck readings, world state, NPC behavior, and physical signs.

Example:

```text
Problem:
The fabricator stalls.

Visible signs:
The workbench light flickers.
A conveyor is cycling the same bin.
A worker is arguing with a sorter terminal.

Field Deck:
DIAG: MATERIAL_MISMATCH detected.
CIVIC: certified ceramic reserved for clinic oxygen pump.
NULL: sorter loop resembles prior command chatter event.
```

## 3.5 Repair Before Expansion

The first resource chains should repair broken infrastructure before enabling freeform expansion.

Seedworks should first teach:

```text
restore power
patch pipe
clean water
move medicine
reopen road
repair fabricator
stabilize greenhouse
```

Only later should it expand into:

```text
large factories
regional rail
orbital logistics
planetary supply networks
inter-worldline trade
```

## 3.6 Every Chain Should Create a Human Question

The best resource chains ask a social question:

```text
Who gets the last battery?
Who may use clean water first?
Who owns salvaged parts from a dead corporate site?
Should uncertified repairs be allowed during crisis?
Can refugees access public fabrication queues?
Should biological restoration delay mechanical throughput?
Can a machine refuse unsafe material flow?
```

Design rule:

```text
A chain is complete when it creates a decision, not when it creates an item.
```

---

# 4. Resource Chain Taxonomy

Symtropy resources should be grouped into ten chain families.

```text
1. Water Chains
2. Power Chains
3. Food and Bio-Metabolism Chains
4. Repair Material Chains
5. Fabrication and Tooling Chains
6. Computing and Firmware Chains
7. Logistics and Route Chains
8. Care and Medicine Chains
9. Ecological Restoration Chains
10. Legitimacy and Witness Chains
```

These families overlap.

A ceramic seal may belong to repair, fabrication, water, legitimacy, and faction politics at the same time.

---

# 5. Chain Family 1 — Water Chains

## Core Fantasy

```text
Water is not loot.
Water is the first constitution.
```

## Chain Shape

```text
source water
-> intake
-> settling
-> filtration
-> testing
-> storage
-> allocation
-> distribution
-> sanitation return
-> ecological consequence
-> civic record
```

## Early Resources

```text
raw basin water
contaminated canal water
stored cistern water
filter membrane
ceramic seal
chlorine substitute / sterilization agent
biofilter culture
pressure valve
pump actuator
water ledger token
```

## Gameplay Uses

```text
restore drinking water
power medbay sanitation
irrigate greenhouse
cool machines
clean contaminated parts
supply refugee camp
trade with nearby settlement
run biological restoration
```

## Failure Modes

```text
contamination
tank leak
pump failure
filter exhaustion
private diversion
ration dispute
Null-locked valve
biofilter die-off
root intrusion
sabotage
```

## Field Deck Example

```text
SCAN:
Water flow intermittent.
Visible sediment load high.

DIAG:
Filter membrane saturated.
Pump actuator current draw unstable.

CIVIC:
Emergency water allocation clause active.
Public review required in 72 hours.

NULL:
Automated rationing script still enforcing expired household classes.
```

## Civic Questions

```text
Who receives water first during shortage?
Can a temporary repair remain in use after emergency expiry?
Can a private contractor meter public water?
Should ecological flow be restored before human storage is full?
```

---

# 6. Chain Family 2 — Power Chains

## Core Fantasy

```text
A grid is a promise made of copper.
```

## Chain Shape

```text
generation
-> conditioning
-> storage
-> priority routing
-> device load
-> heat/cooling
-> outage response
-> public allocation
```

## Early Resources

```text
solar panel shard
charge controller
battery cell
copper cable
ceramic insulator
fuse block
inverter board
manual breaker
field generator
thermal sink
```

## Gameplay Uses

```text
run water pump
keep medicine cold
power fabricator
charge Field Deck
light streets
operate perimeter sensors
run settlement mesh
stabilize greenhouse climate
```

## Failure Modes

```text
brownout
overload
short circuit
battery fire
priority conflict
private compute drain
cooling failure
cable theft
Null load loop
```

## First Seedworks Conflict

```text
The last stable battery reserve cannot power everything.
The player must help decide between:

medbay refrigeration
fabricator restart
perimeter defense
water pump priming
settlement signal relay
```

Each choice creates consequences.

```text
Medbay first:
  health improves
  water repair delayed
  repair faction frustrated

Fabricator first:
  repair speed improves
  clinic risk rises

Defense first:
  safety improves
  trust may fall if water remains offline

Water first:
  public morale rises
  medicine risk increases
```

---

# 7. Chain Family 3 — Food and Bio-Metabolism Chains

## Core Fantasy

```text
A settlement eats its ethics.
```

## Chain Shape

```text
seed / culture / animal stock
-> soil or growth medium
-> water
-> nutrients
-> labor
-> harvest
-> storage
-> kitchen
-> ration / meal
-> health / morale / waste return
```

## Early Resources

```text
seed packet
mycorrhizal ampoule
compost feedstock
nutrient salts
greenhouse tray
pollinator habitat
food crate
cool storage cell
shared kitchen ration
```

## Gameplay Uses

```text
feed NPCs
support work crews
stabilize refugees
reduce stress
recover soil
supply medicine production
trade surplus
```

## Failure Modes

```text
greenhouse heat stress
water shortage
fungal imbalance
pest outbreak
ration theft
kitchen power loss
nutrient lockout
seed ownership dispute
invasive biology risk
```

## Design Note

Food should not be pure hunger-meter gameplay.

It should tie into:

```text
care
labor capacity
trust
ritual
hospital recovery
refugee dignity
cultural identity
ecological health
```

---

# 8. Chain Family 4 — Repair Material Chains

## Core Fantasy

```text
The broken world can still be made accountable.
```

## Chain Shape

```text
salvage source
-> extraction
-> sorting
-> cleaning
-> testing
-> certification
-> staging
-> repair installation
-> inspection
-> maintenance record
```

## Early Resources

```text
steel band
copper contact strip
ceramic seal
rubberized gasket
valve handle
sensor diode
pressure gauge
insulation braid
pipe clamp
bearing cartridge
```

## Material States

```text
clean
corroded
oxidized
cracked
heat-warped
irradiated
contaminated
certified
uncertified
counterfeit
faction-marked
Null-suspect
```

## Gameplay Uses

```text
pipe patch
valve replacement
power bridge
sensor mast
fabricator repair
rover maintenance
bridge repair
terminal restoration
```

## Failure Modes

```text
seal leak
misalignment
counterfeit tolerance
material contamination
salvage ownership claim
improper certification
wrong pressure class
faction refusal
```

## Design Rule

```text
Repair materials are not ingredients.
They are evidence.
```

---

# 9. Chain Family 5 — Fabrication and Tooling Chains

## Core Fantasy

```text
Make the missing part, then prove it belongs in the world.
```

## Chain Shape

```text
blueprint
-> certified materials
-> tool calibration
-> fabrication work
-> QA witness
-> firmware mount if needed
-> installation
-> civic registration
```

## Early Resources

```text
repair blueprint
fabricator head
tolerance jig
caliper bench
scrap filament
PCB blank
motor winding
quality witness tag
```

## Blueprint Provenance Types

```text
commons repair blueprint
Utility Sovereign locked blueprint
Archive-certified schematic
Quarantine-restricted template
black-market patch diagram
field-expedient repair card
Ghost ruin recovered schematic
```

## Gameplay Uses

```text
fabricate pump actuator
produce valve handle
make replacement sensor mast
repair rover component
print greenhouse bracket
craft temporary bridge anchor
```

## Failure Modes

```text
fabricator miscalibration
bad tolerance
wrong material class
firmware mismatch
uncertified output
power interruption
hidden vendor lock
Null script echo
```

## Field Deck Example

```text
DIAG:
Fabricator output tolerance outside waterworks pressure spec.

CIVIC:
Part may be installed under emergency repair clause only.
Permanent certification requires public QA witness.

NULL:
Blueprint contains opaque post-install callback.
```

---

# 10. Chain Family 6 — Computing and Firmware Chains

## Core Fantasy

```text
Software is also a resource, and it can become a ruin.
```

## Chain Shape

```text
script / firmware / access key
-> provenance check
-> sandbox validation
-> Device Bus mount
-> deterministic fuel test
-> authority review
-> installation
-> log monitoring
-> Chronicle event if public
```

## Early Resources

```text
firmware tab
boot cartridge
controller script
access credential
repair token
archive log bundle
source-chain fragment
diagnostic patch
```

## Gameplay Uses

```text
initialize patch conduit
restart pump controller
unlock dead terminal
repair sorter loop
patch drone dock
restore public water ledger
verify Archive Witness event
```

## Failure Modes

```text
out-of-fuel script
staged write rollback
authority denied
expired credential
opaque vendor function
command chatter
Null certainty injection
log corruption
```

## Design Rule

```text
A script that moves water is a political object.
```

---

# 11. Chain Family 7 — Logistics and Route Chains

## Core Fantasy

```text
Civilization moves on maintained routes.
```

## Chain Shape

```text
source site
-> pickup
-> load
-> route selection
-> convoy movement
-> checkpoint / hazard
-> depot intake
-> sorting
-> dispatch
-> delivery
```

## Early Resources and Infrastructure

```text
field crate
cargo strap
hand cart
rover bed
storage bin
salvage pallet
road marker
bridge panel
route token
warehouse ledger
```

## Gameplay Uses

```text
move repair parts
haul water
deliver medicine
transport refugees
recover archive cores
supply greenhouse
support road repair
```

## Failure Modes

```text
mud route blocked
bridge washed out
vehicle breakdown
cargo theft
checkpoint denial
convoy ambush
sorter jam
warehouse mislabel
fuel shortage
route legitimacy dispute
```

## Design Rule

```text
Logistics is not the boring part between missions.
Logistics is where missions become civilization.
```

---

# 12. Chain Family 8 — Care and Medicine Chains

## Core Fantasy

```text
Care depends on cold storage, clean water, trust, and time.
```

## Chain Shape

```text
medicine source
-> cold chain
-> sterile handling
-> triage priority
-> clinic power
-> treatment
-> recovery
-> follow-up
-> public accountability
```

## Early Resources

```text
medicine vial
sterile bandage
cooling cell
clinic water allotment
triage tag
field splint
bio-monitor patch
sanitation kit
```

## Gameplay Uses

```text
stabilize injured NPC
prevent outbreak
keep medbay functional
support work crew recovery
justify emergency power allocation
```

## Failure Modes

```text
medicine spoilage
clinic power loss
water contamination
triage dispute
care labor shortage
private hoarding
refugee exclusion
unauthorized genetic treatment
```

## Civic Questions

```text
Who receives scarce medicine first?
Can a work crew demand treatment priority because they repair water?
Can a clinic refuse undocumented refugees?
Can emergency modification be performed without full consent?
```

---

# 13. Chain Family 9 — Ecological Restoration Chains

## Core Fantasy

```text
Ecology becomes gameplay when restoration creates obligations.
```

## Chain Shape

```text
species / culture / soil unit
-> quarantine review
-> habitat preparation
-> deployment
-> monitoring
-> ecological effect
-> faction response
-> long-term maintenance
-> Chronicle precedent
```

## Early Resources

```text
white willow seed packet
mycorrhizal spore ampoule
biofilter culture
deinococcus remediation vial
wetland reed bundle
earthworm cocoon case
soil carbon inoculant
pollinator shelter kit
```

## Gameplay Uses

```text
reduce water toxins
stabilize riverbank
restore soil
remediate radiation
support greenhouse
lower mechanical filtration burden
create habitat corridors
```

## Failure Modes

```text
invasive spread
alien ecology contamination
root intrusion
biosecurity panic
corporate seed claim
ritual objection
slow effect during emergency
quarantine lock
```

## Example Chain

```text
polluted stream
-> Field Deck scan
-> willow deployment proposal
-> nursery propagation
-> public objection from pipe maintenance crew
-> planting
-> toxin load drops
-> root intrusion risk rises
-> waterworks filtration burden decreases
-> Chronicle records biological repair precedent
```

Design rule:

```text
A seed is not loot.
It is a civilization decision waiting to germinate.
```

---

# 14. Chain Family 10 — Legitimacy and Witness Chains

## Core Fantasy

```text
Authority is a resource, and it can run out.
```

## Chain Shape

```text
credential
-> claim
-> witness
-> authorization
-> action
-> review
-> precedent
-> Chronicle memory
```

## Early Resources

```text
emergency repair token
Archive Witness signature
public vote record
operator credential
settlement permit
source-chain proof
temporary water access clause
faction work order
```

## Gameplay Uses

```text
authorize water repair
install public device
override dead authority
allocate scarce power
certify fabricated part
grant refugee access
open archive door
```

## Failure Modes

```text
expired authority
dead law lock
forged witness
lost source chain
faction refusal
public distrust
emergency drift
Chronicle contradiction
```

## Design Rule

```text
Legitimacy should be as playable as steel.
```

---

# 15. The First Vertical Slice: Waterworks Patch Chain

## Purpose

The first complete resource-chain implementation should be the **Waterworks Patch Chain**.

It should teach the whole Symtropy grammar in one playable arc.

## Scenario

A fractured pipe prevents the main pump from restoring flow to Seedworks Outpost.

The pump console is not enough.

The player must repair matter, initialize logic, and confront authority.

## Chain Summary

```text
Broken pipe
-> Field Deck scan
-> Patch Conduit Mk0 blueprint
-> salvage materials
-> material condition check
-> transport to pipe
-> project repair frame
-> weld / seal conduit
-> initialize Device Bus node
-> authorize temporary repair
-> run pump diagnostic
-> restore partial water flow
-> trigger NPC response
-> record Chronicle line
-> public vote over permanent water doctrine
```

## Required Materials

```text
2 steel bands
1 ceramic seal
1 copper contact strip
1 firmware tab
```

## Required Tools

```text
Field Deck Mk0
hand welder
sealant injector
patch cable
basic repair tool
```

## Required Infrastructure

```text
broken waterworks pipe
nearby salvage field
fabricator or repair bench
settlement depot
Device Bus node
Field Deck interface
water pump console
local Chronicle backend
```

## Step-by-Step Gameplay

### Step 1 — Discover Failure

```text
SCAN:
Pipe fracture detected.
Flow path interrupted.
Manual patch required.
```

Player sees water dripping, pressure gauge trembling, NPCs arguing nearby.

### Step 2 — Find Blueprint

Blueprint source options:

```text
friendly service robot provides emergency repair card
Archive terminal contains old public schematic
worker NPC teaches field-expedient version
Utility Sovereign crate contains locked commercial version
```

### Step 3 — Gather Materials

Material locations:

```text
steel bands: collapsed maintenance rack
ceramic seal: old pump house locker
copper contact strip: damaged power junction
firmware tab: dead controller cabinet
```

Material variations:

```text
clean ceramic seal -> best outcome
cracked ceramic seal -> higher leak risk
corporate ceramic seal -> reliable but ownership dispute
black-market firmware tab -> faster but Null suspicion
```

### Step 4 — Stage Materials

Player places materials into the repair frame or nearby field crate.

```text
ANCHOR VALID.
PIPE PRESSURE: LOW.
MATERIALS: 4/4 PRESENT.
CIVIC STATUS: EMERGENCY REPAIR PERMITTED.
```

### Step 5 — Assemble

Player welds bands, injects sealant, seats contact strip, and mounts firmware tab.

Quality variables:

```text
seal_quality
alignment_quality
thermal_damage
material_condition
tool_condition
panic_interruption
```

### Step 6 — Initialize Node

```sh
$ sym-dev initialize /dev/sym/water/unmapped_node_7 --name patch_conduit_alpha

NODE INITIALIZED.
DEVICE CLASS: WATER_PATCH_CONDUIT
AUTHORITY REQUIRED: LOCAL_REPAIR_TOKEN
SAFETY PROFILE: LOW_PRESSURE_ONLY
```

### Step 7 — Authorize Temporary Repair

```sh
$ sym-civic authorize /dev/sym/water/patch_conduit_alpha --token emergency_repair_token

AUTHORIZATION ACCEPTED.
SCOPE: TEMPORARY_REPAIR
DURATION: 72 HOURS
REVIEW REQUIRED: YES
```

### Step 8 — Restore Flow

```text
DIAG:
Patch conduit active.
Flow restored at 36% baseline.
Leak risk: medium.

CIVIC:
Temporary repair created review obligation.
```

### Step 9 — Consequence

NPCs react:

```text
medic: water pressure is enough for clinic sanitation
repair worker: seal needs inspection before full pressure
refugee advocate: public tap must reopen immediately
security officer: unauthorized crowding risk at cistern
archive witness: temporary repair must be recorded
```

### Step 10 — Chronicle

Possible Chronicle lines:

```text
Clean repair:
The player gave the waterworks a new vein and named it before the settlement.

Rough repair:
The pipe held, but the settlement could hear the weakness in the seal.

Unauthorized bypass:
The player made the water move before the law agreed it should.

Corporate part used:
Water returned through a part whose ownership was already an argument.
```

---

# 16. Resource Chain Data Model

## Resource Item

```rust
struct ResourceItem {
    id: ResourceItemId,
    kind: ResourceKind,
    display_name: String,
    stack_behavior: StackBehavior,
    mass_kg: f32,
    volume_liters: f32,
    condition: MaterialCondition,
    contamination: ContaminationState,
    provenance: ProvenanceRecord,
    ownership_claims: Vec<OwnershipClaim>,
    certifications: Vec<CertificationTag>,
    faction_marks: Vec<FactionMark>,
    civic_flags: Vec<CivicFlag>,
    device_bus_path: Option<DevicePath>,
    chronicle_relevance: ChronicleRelevance,
}
```

## Provenance Record

```rust
struct ProvenanceRecord {
    source_site: SiteId,
    recovered_by: Option<AgentId>,
    recovery_tick: ChronicleTick,
    prior_owner: Option<ActorId>,
    extraction_method: ExtractionMethod,
    witness_status: WitnessStatus,
    source_chain_entry: Option<DeckSourceEntryId>,
}
```

## Resource Chain Node

```rust
struct ResourceChainNode {
    node_id: ChainNodeId,
    chain_family: ChainFamily,
    input_resources: Vec<ResourceRequirement>,
    output_resources: Vec<ResourceOutput>,
    required_tools: Vec<ToolRequirement>,
    required_infrastructure: Vec<DevicePath>,
    required_authority: AuthorityRequirement,
    failure_modes: Vec<ChainFailureMode>,
    settlement_effects: SettlementDelta,
    chronicle_hooks: Vec<ChronicleHook>,
}
```

## Settlement Delta

```rust
struct SettlementDelta {
    power: f32,
    water: f32,
    food: f32,
    health: f32,
    repair: f32,
    trust: f32,
    legitimacy: f32,
    safety: f32,
    stress: f32,
    entropy: f32,
    ecology: f32,
    knowledge: f32,
    logistics: f32,
    signal: f32,
    harmony: f32,
}
```

## Chain Failure Mode

```rust
enum ChainFailureMode {
    MissingMaterial,
    WrongMaterialClass,
    LowQualityMaterial,
    ContaminatedMaterial,
    CounterfeitMaterial,
    ToolFailure,
    PowerShortage,
    RouteBlocked,
    AuthorityDenied,
    WitnessRequired,
    DeviceBusFault,
    NullChatter,
    FactionDispute,
    EcologicalObjection,
    EmergencyExpiry,
}
```

---

# 17. Field Deck Resource Interface

The Field Deck should reveal resource-chain information in layers.

## SCAN

Shows physical state.

```text
Steel band detected.
Corrosion visible.
Mass: 1.8 kg.
```

## DIAG

Shows functional suitability.

```text
Conductivity degraded.
Compatible with low-pressure water repair only.
Estimated seal risk: medium.
```

## ARCHIVE

Shows history.

```text
Recovered from Old Pump House maintenance rack.
Rack last inspected under Emergency Water Act 2087.
```

## CIVIC

Shows authority and claim.

```text
Ownership claim disputed.
Utility Sovereign contract expired but not formally overturned.
Emergency public repair use likely defensible under witness.
```

## NULL

Shows anomaly and manipulation risk.

```text
WARNING:
Material certification tag repeats impossible timestamp.
Possible counterfeit or archive corruption.
```

## REPAIR

Shows how to use it.

```text
Patch Conduit Mk0 requires:
2 steel bands
1 ceramic seal
1 copper contact strip
1 firmware tab
```

## WITNESS

Shows whether use should be recorded.

```text
Using disputed corporate salvage in public water system may create Chronicle-relevant precedent.
Archive Witness recommended.
```

---

# 18. UI / UX Design Rules

## 18.1 Do Not Show the Whole Economy at Once

The player should first see:

```text
what is broken
what is missing
where a missing thing might be
what happens when they install it
```

Do not initially show:

```text
complete planetary supply graph
all hidden trophic consequences
full provenance lattice
faction economy dashboard
worldline trade futures
```

## 18.2 Use Physical World Cues Before Menus

Examples:

```text
empty racks show missing tools
water stains show leaks
flickering lights show power shortage
NPC queues show scarcity
crates blocking hallways show logistics failure
steam and heat shimmer show overload
birds avoid polluted stream
workers mark unsafe parts with paint
```

## 18.3 Make Resource Chains Audible

Sound should teach chain state.

Examples:

```text
healthy pump rhythm
misaligned bearing scrape
low battery inverter whine
water hammer in pipe
fabricator calibration chirp
sorter loop clatter
clinic fridge alarm
wetland insect return after restoration
```

## 18.4 Make Chain Quality Visible

A clean repair should look and sound different from a rough repair.

```text
clean repair:
steady flow, smooth seal, green diagnostic pulse

rough repair:
pipe shudder, intermittent drip, amber warning, NPC concern

illegal bypass:
water moves, but civic warning remains red
```

---

# 19. NPC and Faction Reactions

Resource chains should change social behavior.

## Example: Waterworks Patch Outcome

### Repair Guild Mechanic

```text
Approves high-quality physical repair.
Dislikes sloppy emergency shortcuts.
Respects visible tool competence.
```

### Archive Witness

```text
Approves recorded authorization.
Objects to undocumented public infrastructure changes.
```

### Refugee Advocate

```text
Prioritizes immediate public access.
Objects if repair benefits registered citizens only.
```

### Utility Sovereign Agent

```text
Claims proprietary part use creates service liability.
Offers clean restoration under contract.
```

### Continuance Officer

```text
Approves order and controlled distribution.
Objects to crowd-driven access.
May push emergency ration authority.
```

### Ecologist

```text
Approves watershed repair.
Objects if mechanical flow damages wetland recovery.
```

Design rule:

```text
The same resource outcome should create different truths for different people.
```

---

# 20. Progression Model

## Mk0 — Scrap Bootstrap

Core chains:

```text
hand salvage
basic water repair
small battery routing
field crates
repair bench
simple food storage
one biological restoration loop
local emergency tokens
```

Player fantasy:

```text
I can keep this place alive with my hands.
```

## Mk1 — Local Metabolism

Core chains:

```text
stable water distribution
microgrid storage
greenhouse food loop
certified parts
small fabricator
rover logistics
clinic cold chain
public repair ledger
```

Player fantasy:

```text
The settlement has a metabolism now.
```

## Mk2 — Regional Infrastructure

Core chains:

```text
roads
warehouses
convoys
trade contracts
regional water compacts
tool libraries
repair guild certification
medicine routes
route safety protocols
```

Player fantasy:

```text
We are no longer one outpost. We are a region.
```

## Mk3 — Planetary Systems

Core chains:

```text
watershed-scale management
large factories
rail corridors
advanced ecology
city food networks
regional governance
multi-settlement charters
public infrastructure budgets
```

Player fantasy:

```text
A planet is becoming governable again.
```

## Mk4 — Orbital Industry

Core chains:

```text
launch propellant
orbital fabrication
satellite relays
debris tracking
vacuum materials
closed-loop habitat supplies
lunar or asteroid feedstocks
```

Player fantasy:

```text
Space is not escape. It is exposed maintenance.
```

## Mk5 — Interplanetary Civilization

Core chains:

```text
propellant depots
habitat life support
deep-space rescue law
interplanetary cargo
radiation medicine
closed-loop water
archive replication
```

Player fantasy:

```text
Every airlock is a border crossing.
```

## Mk6 — Worldline Civilization

Core chains:

```text
portable blueprints
source-chain identity
cross-worldline trade
Chronicle reconciliation
Confluence treaty resources
planetary translation prerequisites
```

Player fantasy:

```text
Civilizations can carry their promises between futures.
```

---

# 21. Implementation Milestones

## v0.1 — Waterworks Resource Slice

Implement:

```text
ResourceItem data structure
basic material condition
4 MVP materials
field crates
salvage pickup
repair frame staging
Patch Conduit Mk0
Device Bus node initialization
emergency authorization
water restoration settlement delta
one Chronicle line
NPC reaction stubs
```

Do not implement:

```text
full economy simulation
regional trade
complex warehouses
large factory lines
freeform base building
multiplayer cargo claims
full provenance graph UI
```

## v0.2 — Mission and Convoy Chains

Add:

```text
cargo rover
mission cargo manifests
medicine delivery
convoy ambushes
route hazards
checkpoint authority
lost cargo recovery
basic warehouse intake
```

## v0.3 — Civic Resource Governance

Add:

```text
public allocation votes
emergency ration law
repair certification hearings
legitimacy debt from resource abuse
faction budget priorities
refugee access disputes
```

## v0.4 — Factory and Logistics

Add:

```text
warehouses
sorters
simple conveyors
fabricator queues
route planning
production chains
power-load scheduling
logistics faults
```

## v0.5 — Procedural Resource Politics

Add:

```text
resource pressure vectors
faction archetype drift
black markets
sabotage
labor strikes
private utility capture
public works movements
```

## v0.6 — Signed Resource History

Add:

```text
source-chain resource provenance
portable blueprint legitimacy
Archive Witness evidence bundles
worldline resource claims
local Chronicle backend expansion
```

## v1.0 — Settlement Metabolism

A complete playable resource system should support:

```text
survival chains
repair chains
power chains
food chains
medicine chains
ecological chains
logistics chains
civic authorization
faction reaction
Chronicle consequence
```

---

# 22. Design Risks

## Risk 1 — Too Much Spreadsheet

If resource chains become mostly menus, Symtropy loses its body.

Mitigation:

```text
show physical crates
show pipes
show racks
show broken machines
show NPC queues
show visible water quality
show real routes
```

## Risk 2 — Too Much Friction

If every small item needs legal review, players will hate the system.

Mitigation:

```text
only public survival infrastructure needs heavy authorization
small personal crafting stays lightweight
emergency tokens allow temporary action
review becomes future gameplay, not immediate punishment
```

## Risk 3 — Early Onboarding Overload

If the first hour explains provenance, civic authority, ecology, and source chains all at once, players will bounce.

Mitigation:

```text
first action: fix pipe
second feeling: water matters
third discovery: law can block repair
fourth consequence: someone remembers
```

## Risk 4 — Factory Game Erases Civic Game

If optimization dominates, players may ignore politics.

Mitigation:

```text
resource throughput affects trust and legitimacy
private efficiency can create public resentment
unsafe automation can create Null drift
public witnessing can slow but stabilize chains
```

## Risk 5 — Civic Game Erases Action

If governance slows everything, players may feel trapped in hearings.

Mitigation:

```text
allow emergency action
make action produce review obligations
resolve some disputes through field evidence
keep combat, salvage, repair, and transport tactile
```

---

# 23. Acceptance Tests

A resource-chain implementation is Symtropy-ready if the player can answer:

```text
What resource do I need?
Where can I get it?
What condition is it in?
How do I move it?
What can I build or repair with it?
Who might object?
What happens if I use a bad version?
What changes in the settlement after I succeed?
Does the world remember the outcome?
```

## v0.1 Acceptance Test

The Waterworks Patch Chain succeeds if:

```text
1. Player scans broken pipe.
2. Player learns Patch Conduit Mk0 requirements.
3. Player recovers four physical materials.
4. At least one material can have degraded condition.
5. Player stages materials at pipe.
6. Player performs tactile assembly.
7. Player initializes Device Bus node.
8. Player authorizes temporary repair.
9. Water metric improves.
10. NPC reactions change.
11. Chronicle line records outcome.
12. Later review obligation is created.
```

---

# 24. Final Design Rules

```text
1. A resource is matter under obligation.
2. A chain is complete when its consequence is legible.
3. Materials remember where they came from.
4. Repair materials are evidence, not ingredients.
5. Logistics is where missions become civilization.
6. Legitimacy is as playable as steel.
7. Ecology becomes gameplay when restoration creates obligations.
8. Do not record every crate. Record the crate that changed history.
9. First repair the world. Then let players expand it.
10. Matter becomes history when a society depends on it.
```

---

# 25. Closing Line

```text
Symtropy resources do not merely build machines.
They build the reasons machines are allowed to matter.
```
