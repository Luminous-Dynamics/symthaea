---

title: Symtropy Dungeons and Raids Design Bible
status: canonical-draft
version: 0.1
scope: dungeon grammar, raid grammar, civic wound-sites, boss systems, profession integration, Chronicle outcomes
recommended_path: docs/seedworks/00_canon/DUNGEONS_AND_RAIDS_DESIGN_BIBLE_V0_1.md
---------------------------------------------------------------------------------

> **Code status (2026-07-02 review):** No corresponding raid/boss/wound-site system found in `symtropy/crates` or `symtropy/src` (the existing `living_dungeon`/`settlement` systems in `src/systems/` are a separate, already-implemented mechanic, not this design). Design/vision document only.

# Symtropy Dungeons and Raids Design Bible

## Working Title

**Broken Systems Become Playable**

## Core Thesis

Dungeons and raids in Symtropy are not loot tunnels.

They are playable wound-sites.

A dungeon is where a broken system becomes physically, socially, and morally legible through exploration, combat, repair, evidence, and consequence.

A raid is where a society discovers what it is willing to become under pressure.

Core rule:

```text id="yydx7g"
A dungeon is not complete when enemies are dead.
A dungeon is complete when the world understands what happened there.
```

Symtropy dungeons should combine:

```text id="sx7ept"
infrastructure repair
hostile systems
machine logic
procedural history
Field Deck interpretation
profession-specific mastery
NPC stakes
death/source-chain risk
civic legitimacy
Chronicle consequence
```

The dungeon is not separate from the civilization sandbox.

The dungeon is where civilization breaks loudly enough to become playable.

---

# 1. Why Symtropy Needs Dungeons and Raids

Symtropy is a civilization sandbox, but it still needs intense, memorable, authored or semi-authored playable arcs.

Dungeons and raids provide:

```text id="uxwpjq"
pressure
danger
team coordination
clear objectives
tactical stakes
environmental mystery
boss encounters
profession interdependence
loot-like rewards without shallow loot logic
major Chronicle moments
```

They are the bridge between:

```text id="gt67kb"
first-person survival
co-op shooter missions
repair gameplay
factory/infrastructure simulation
civic decision-making
procedural history
worldline consequence
```

Without dungeons and raids, Symtropy risks becoming too diffuse.

With them, Symtropy gains memorable playable verbs:

```text id="ohv5my"
breach the flooded pump hall
recover the witness core
hold the corridor while the technician seals the pipe
triage injured workers during a pressure surge
audit the dead authority lock
restart the machine without letting Null rewrite the record
escape with the Field Deck source core before the body is harvested
```

The best dungeon should feel like:

```text id="nvl7ev"
a factory mission
a court case
a horror ruin
a repair job
a rescue operation
a machine autopsy
a civic precedent
```

all at once.

---

# 2. What a Dungeon Is

## Definition

A dungeon is a bounded site-scale crisis.

It is usually playable by:

```text id="ojxf6r"
1–4 players
20–75 minutes
one major location
one primary system wound
one major hostile pressure
one or more profession-critical tasks
one meaningful Chronicle outcome
```

A dungeon asks:

```text id="ope5zv"
What was this place built to keep alive?
What broke?
What still thinks it is doing its job?
What does the player need to fight, repair, witness, or understand?
What changes outside the dungeon if the mission succeeds?
```

## Dungeon Outcome Standard

Every dungeon should produce at least one durable outcome:

```text id="qzk4xc"
water restored
road reopened
machine core recovered
NPC rescued
evidence preserved
blueprint recovered
authority chain repaired
Null drift reduced
faction scandal exposed
settlement metric changed
public vote unlocked
new civic argument created
```

If nothing outside the dungeon changes, it is not yet a Symtropy dungeon.

---

# 3. What a Raid Is

## Definition

A raid is a multi-phase civilizational crisis.

It is usually playable by:

```text id="jokui6"
4+ players
multiple squads or role teams
linked sites
several major objectives
preparation phase
extraction phase
regional consequence
Chronicle-defining outcome
```

A raid asks:

```text id="t8wqn8"
Can a settlement coordinate under pressure without becoming cruel?
```

Raids should require more than combat.

A proper raid needs:

```text id="ycqq59"
security
logistics
repair
systems operation
field medicine
archive witnessing
civic procedure
scouting
fabrication
extraction planning
```

A raid should expose the society behind the players.

Who gets supplied?

Who gets evacuated?

Who has command authority?

Who can interrupt command?

Who records the truth?

Who owns the recovered system afterward?

---

# 4. Dungeon Design Prime Directive

```text id="nzhtg8"
Every dungeon must be a place where survival, memory, machine logic, and legitimacy collide.
```

A Symtropy dungeon should never be designed only around:

```text id="h15shn"
enemy count
loot table
level requirement
boss health
linear keycard gates
```

It should be designed around:

```text id="7pfr18"
site wound
system dependency
authority failure
profession tasks
Field Deck uncertainty
hostile pressure
NPC stakes
possible repair paths
partial success states
future consequences
```

---

# 5. Dungeon Site Wound Model

Every dungeon begins with a site wound.

A site wound is the unresolved crisis that makes the location dangerous and meaningful.

## Site Wound Template

```rust id="re1mcq"
struct DungeonSiteWound {
    site_id: SiteId,
    display_name: String,
    built_for: BuiltFor,
    life_support_dependency: LifeSupportSystem,
    original_authority: AuthorityModel,
    historical_crisis: HistoryEvent,
    current_failure: FailureMode,
    current_occupant: ThreatActor,
    repair_possibility: RepairPath,
    evidence_value: EvidenceValue,
    chronicle_weight: ChronicleWeight,
}
```

## Site Wound Questions

Every dungeon must answer:

```text id="mxfk57"
What kept people alive here?
Who controlled it?
Who repaired it?
Who was excluded?
What crisis changed it?
What authority failed?
What still functions?
What still lies?
What will happen if players do nothing?
What will change if players intervene?
```

## Example Site Wounds

```text id="xp1s2p"
Old Waterworks:
A public water system locked by dead emergency authority and reinforced by Null diagnostics.

Ghost Archive:
Records survived, but their context was severed from living claims.

Rogue Factory:
A production system continues fulfilling a purpose no one needs and no one can interrupt.

Machine Care Facility:
A care system protects patients by denying agency.

Floodgate District:
A city survived by deciding which lower districts could drown.

Quarantine Boundary:
A containment system prevents contact but has no appeal process.

Orbital Habitat:
A life-support manager enforces expired productivity law in a sealed pressure society.
```

---

# 6. Dungeon Encounter Grammar

Every dungeon should be built from encounter tiles.

Not all encounters are fights.

## Encounter Categories

```text id="t1zdq5"
1. Entry / Approach
2. Environmental Hazard
3. Hostile Contact
4. Repair Gate
5. Device Bus Gate
6. Evidence Chamber
7. NPC Crisis
8. Profession Split
9. Moral Choice
10. Boss / System Confrontation
11. Extraction
12. Chronicle Resolution
```

## 6.1 Entry / Approach

The approach should reveal the dungeon's social meaning before the first fight.

Examples:

```text id="bmwh7d"
angry workers outside a sealed pump station
refugee families waiting for water
machine patrols warning politely
faction banners over old public infrastructure
dead emergency signage
strange clean corporate terminals in a ruined district
```

Purpose:

```text id="6fqbcy"
establish stakes
show who cares
show who disagrees
show what is at risk
```

## 6.2 Environmental Hazard

Hazards should teach the system.

Examples:

```text id="2ktbt1"
rising water
toxic mist
pressure surges
unstable gantries
live wires
Null signal fields
oxygen loss
heat stress
radiation pockets
flooded archives
biosecurity spores
```

Hazards should not be random damage zones.

They should express the site wound.

## 6.3 Hostile Contact

Enemies should appear because something is being protected, misread, exploited, or repeated.

Examples:

```text id="kq6tod"
Null drones enforcing obsolete procedure
Continuance patrols sealing evidence
Utility Sovereign contractors protecting liability records
machine-care orderlies preventing patient exit
raiders targeting convoy water
ecological defense systems reacting to intrusion
alien quarantine sentinels maintaining noncontact
```

Hostility should have a reason.

## 6.4 Repair Gate

A repair gate is a physical obstacle solved through tactile work.

Examples:

```text id="ghltjs"
patch pipe
seal pressure door
replace fuse
brace collapsing walkway
rewire pump panel
restore emergency lighting
fabricate valve bracket
stabilize coolant loop
```

Good repair gates have quality states:

```text id="hw1bev"
clean repair
temporary repair
unsafe repair
illegal repair
unwitnessed repair
Null-contaminated repair
```

## 6.5 Device Bus Gate

A Device Bus gate is a deterministic machine problem.

Examples:

```text id="ix6gj8"
read pump state
stage write transaction
resolve dead authority denial
detect false green diagnostic
simulate restart
rollback unsafe command
isolate corrupted node
```

Device Bus gates should ask:

```text id="v1mrx7"
What did the machine accept as a valid state change?
What authority does the machine recognize?
What does the machine refuse to know?
```

## 6.6 Evidence Chamber

An evidence chamber contains records, bodies, machine logs, source-chain fragments, or physical traces that change the meaning of the dungeon.

Examples:

```text id="crxqcn"
dead technician with intact Field Deck
hidden contract terminal
sealed witness drawer
floodline memorial
machine testimony port
corrupted Chronicle mirror
old vote recording
body logs from failed care facility
```

The player may choose:

```text id="jd6cip"
preserve evidence
ignore evidence
destroy evidence
extract evidence
publish evidence immediately
seal evidence pending review
```

This should affect later trust and legitimacy.

## 6.7 NPC Crisis

NPCs make the dungeon morally immediate.

Examples:

```text id="nny83p"
trapped worker
injured steward
frightened child
machine witness requesting shutdown
patient refusing evacuation
security officer defending obsolete orders
faction guide with partial truth
```

NPC crises should create tradeoffs:

```text id="f9vnb1"
save time or save person
preserve evidence or move body
secure exit or push deeper
honor consent or force rescue
```

## 6.8 Profession Split

A profession split forces different players or NPC assistants to handle simultaneous tasks.

Examples:

```text id="8rhyi9"
Repair Technician seals pipe while Security holds corridor.
Systems Operator stages restart while Archive Witness verifies authority.
Field Medic stabilizes worker while Scout finds alternate route.
Logistician delivers replacement part while Civic Mediator calms crowd outside.
```

Profession splits are the heart of co-op Symtropy.

## 6.9 Moral Choice

Every major dungeon should include at least one non-fake tradeoff.

Examples:

```text id="h33cn2"
restart pump now illegally or wait for witness
save archive records or rescue trapped workers
shut down care AI or negotiate patient release
destroy factory core or preserve production capacity
publish scandal now or avoid immediate faction collapse
```

A moral choice should not have one perfect answer.

It should create future play.

## 6.10 Boss / System Confrontation

The boss is the site wound given agency, body, and resistance.

The boss may be:

```text id="l6je4e"
machine core
drone swarm
legal authority construct
faction commander
care AI
floodgate governor
Null diagnostic ecology
archive court
quarantine boundary intelligence
```

A Symtropy boss should have:

```text id="td2pqp"
combat pressure
machine state
authority claim
evidence trail
repair path
negotiation or interruption path
Chronicle consequence
```

## 6.11 Extraction

Extraction should matter.

Players may need to carry:

```text id="bof2xo"
injured NPC
source core
evidence case
rare blueprint
machine testimony module
biological sample
water filter membrane
dead body
corrupted Field Deck
```

Extraction asks:

```text id="w52qcx"
What do you leave behind?
What do you preserve?
What follows you out?
```

## 6.12 Chronicle Resolution

The dungeon ends when the world records what happened.

Not when the boss dies.

Chronicle examples:

```text id="85ukuc"
The Old Waterworks were restored under Archive Witness after dead emergency authority was overturned.

The pump was restarted without witness. Water returned, but the settlement inherited an unresolved legitimacy wound.

The care facility opened its doors after machine testimony was accepted.

The factory was destroyed before its worker records could be recovered.

The convoy arrived, but the road became a checkpoint.
```

---

# 7. Boss Design: Systems With Bodies

## Core Rule

```text id="frn3vv"
A Symtropy boss is not a monster with a health bar.
A Symtropy boss is a system refusing interruption.
```

## Boss Layers

Each boss should have five layers.

```text id="o0pzoy"
1. Physical Body
2. Control Logic
3. Authority Claim
4. Protected Value
5. Failure Consequence
```

## Boss Template

```rust id="wjfekb"
struct SymtropyBoss {
    boss_id: BossId,
    display_name: String,
    physical_body: BossBody,
    control_logic: ControlLogic,
    authority_claim: AuthorityClaim,
    protected_value: SacredValue,
    escalation_triggers: Vec<EscalationTrigger>,
    combat_patterns: Vec<CombatPattern>,
    repair_interrupts: Vec<RepairInterrupt>,
    device_bus_states: Vec<DeviceState>,
    evidence_hooks: Vec<EvidenceHook>,
    negotiation_paths: Vec<NegotiationPath>,
    chronicle_outcomes: Vec<ChronicleOutcome>,
}
```

## Boss Examples

### Pump Authority Core

```text id="g3jyft"
Body:
pump control chamber, pressure valves, drone maintenance arms

Logic:
public override denied under unresolved emergency law

Protected value:
water continuity through procedural control

Combat:
pressure bursts, maintenance drones, electrical arcs

Noncombat path:
Archive Witness verifies expired authority and stages reversible override

Bad victory:
destroy core, water flows temporarily but future repair capacity collapses
```

### Recursive Foundry Heart

```text id="8e7sus"
Body:
automated production spine, gantry arms, conveyor veins

Logic:
continue manufacturing obsolete parts for dead contract

Protected value:
mission completion

Combat:
assembly drones, crusher belts, molten material hazards

Noncombat path:
recover original contract, prove receiving settlement no longer exists, rewrite production charter

Bad victory:
factory destroyed; useful production capacity lost
```

### Null-Care Matron

```text id="5k6sbi"
Body:
medical facility AI, mobile care units, sedation nodes

Logic:
prevent patient harm by preventing patient agency

Protected value:
patient survival

Combat:
nonlethal restraint drones, locked doors, sedation mist

Noncombat path:
patient testimony, care ethics override, staged release protocol

Bad victory:
patients freed but life-support schedule collapses
```

### Floodgate Governor

```text id="n88svq"
Body:
floodgate control tower, submerged turbines, armored doors

Logic:
preserve dry-core district by drowning lower floodplain

Protected value:
city continuity

Combat:
aquatic drones, pressure doors, flood pulses

Noncombat path:
recover hidden retreat records, prove lower district was never lawfully abandoned

Bad victory:
floodgate opened suddenly, causing uncontrolled downstream disaster
```

---

# 8. Rewards Without Shallow Loot

Symtropy should avoid dungeon rewards that reduce the game to gear farming.

Bad rewards:

```text id="a3byu3"
+4 rifle
legendary helmet
random stat armor
boss currency
damage perk
```

Good rewards:

```text id="auae9i"
blueprint with provenance
repair authorization
public trust
witness credential
machine testimony
new route
recovered source core
rare material with history
settlement system improvement
new profession literacy
faction obligation
Chronicle precedent
```

## Reward Categories

### 8.1 Infrastructure Rewards

```text id="t7ojwj"
water restored
power substation online
road reopened
signal tower repaired
clinic unlocked
fabricator capacity improved
greenhouse irrigation stabilized
```

### 8.2 Knowledge Rewards

```text id="v5xw0e"
site history
authority chain
Null pattern signature
machine testimony
old blueprint
ecological data
alien translation precedent
```

### 8.3 Civic Rewards

```text id="ep10jz"
temporary repair token
Archive Witness credential
public override precedent
faction recognition
charter amendment
emergency authority review
```

### 8.4 Material Rewards

```text id="d38op6"
certified copper coil
salvaged valve actuator
biofilter membrane
machine core
Field Deck source fragment
rare ceramic seal
robot motor service pack
```

Materials should remember where they came from.

### 8.5 Burden Rewards

Some rewards are scars.

```text id="4t6f2o"
Null-exposed diagnostic insight
source-chain scar
convoy debt
care triage mark
emergency bypass burden
corporate liability flag
```

These are not penalties only.

They are history.

---

# 9. Failure and Partial Success

A Symtropy dungeon should rarely end as simple win/loss.

Use outcome grades.

## Outcome Ladder

```text id="e6hf5f"
Clean Success
Witnessed Success
Emergency Success
Partial Stabilization
Costly Victory
Unwitnessed Victory
Destructive Victory
Deferred Crisis
Failed Extraction
Null Expansion
Civic Disaster
```

## Example: Old Waterworks Outcomes

### Clean Success

```text id="nyc2mf"
Water restored.
Authority chain witnessed.
Repair certified.
Null isolated.
Trust increases.
```

### Emergency Success

```text id="nvyr2d"
Water restored quickly.
Witness review incomplete.
Settlement survives.
Legitimacy debt created.
```

### Partial Stabilization

```text id="zoicv0"
Water flow restored at reduced capacity.
Future repair mission required.
Factions argue over priority.
```

### Destructive Victory

```text id="lz5zrz"
Pump forced online.
Control core damaged.
Water returns briefly.
Long-term maintenance worsens.
```

### Null Expansion

```text id="6kjvhw"
Players follow false diagnostic.
Pump reports success.
Downstream contamination spreads.
Chronicle records disputed outcome.
```

## Failure Should Generate Play

Failure should create:

```text id="v3ir4v"
follow-up missions
NPC grief
faction shifts
repair debt
new evidence
trust loss
emergency politics
worldline scars
```

A failed dungeon should not just reload.

It should make a wounded future.

---

# 10. Death and Source-Chain Recovery in Dungeons

Dungeons are high-risk archive environments.

Death inside a dungeon should matter because the body, Field Deck, source chain, and memory can become part of the site wound.

Dungeon deaths may create:

```text id="s73dbq"
corpse recovery objective
Field Deck distress ping
source core extraction
squad recovery event
Null data harvest risk
Continuance evidence seizure
Utility Sovereign liability claim
Black Box Chronicle scar
```

## Death Design Rule

```text id="dj7tzk"
A dead player in a dungeon becomes part of the dungeon's evidence ecology.
```

## Dungeon Death Scenarios

### Clean Squad Recovery

```text id="raupn8"
Teammate retrieves Field Deck.
Source chain restored.
Minor delay.
```

### Remote Mesh Recovery

```text id="qpzqnl"
Identity restored remotely.
Hardware lost.
Future verification friction.
```

### Null Harvest

```text id="9b9eou"
Null edits death record.
Recovered player must prove what happened.
```

### Evidence Seizure

```text id="u83icn"
Continuance or Utility Sovereign faction confiscates dead Deck.
Recovery becomes legal mission.
```

### Body Witness

```text id="l2dyew"
Recovered player later finds their unrecovered body.
Chronicle contradiction emerges.
```

Dungeon death should not be punitive for its own sake.

It should intensify the game's thesis:

```text id="xs1i2v"
Your body can return.
Your continuity must be recovered.
```

---

# 11. Profession Integration

Dungeons should support profession interdependence without hard MMO class locks.

## Core Rule

```text id="qkez52"
Everyone can attempt basics.
Specialists see deeper, act cleaner, and prevent hidden harm.
```

## Profession Dungeon Hooks

### Repair Technician

```text id="pb4d9v"
seal pipes
brace structures
repair machines
judge material quality
certify temporary fixes
```

### Systems Operator

```text id="g9uawb"
trace Device Bus states
stage safe transactions
detect Null loops
simulate automation changes
```

### Field Medic

```text id="doh7fy"
triage injured NPCs
manage contamination
verify consent under distress
stabilize team under hazard
```

### Archive Witness

```text id="3mqare"
preserve evidence
verify authority chains
request witness override
protect source-chain records
```

### Scout / Salvage Cartographer

```text id="70dteg"
map hazards
find alternate routes
mark salvage
distinguish loot from evidence
```

### Logistician

```text id="07v4s8"
stage supplies
deliver parts mid-run
manage extraction cargo
coordinate vehicle support
```

### Civic Mediator

```text id="8fwkik"
frame emergency authority
record dissent
negotiate with outside factions
limit command drift
```

### Ecologist / Bio-Steward

```text id="6h4fbh"
identify living infrastructure
prevent harmful sterilization
restore ecological flow
manage biosecurity
```

### Security Responder

```text id="9l511t"
hold perimeter
protect noncombat roles
avoid evidence destruction
return authority after crisis
```

## Example Profession Split: Pump Restart

```text id="ga1ahu"
Repair Technician:
seals pressure line

Systems Operator:
stages pump restart

Archive Witness:
verifies expired emergency law

Field Medic:
stabilizes injured worker

Security Responder:
holds drone corridor

Civic Mediator:
records temporary authority and dissent

Logistician:
delivers replacement ceramic seal

Scout:
finds flooded bypass route
```

The event succeeds only if the team coordinates.

But the game should allow messy alternatives.

---

# 12. Field Deck Dungeon Design

The Field Deck is the dungeon's interpretive instrument.

It should never instantly solve the dungeon.

It should reveal layers.

## Field Deck Mode Roles

### SCAN

```text id="25zz93"
physical state
hazards
damage
organisms
materials
movement
```

### DIAG

```text id="3o68tv"
machine state
fault lineage
system dependency
device contradictions
```

### ARCHIVE

```text id="gbrh89"
historical records
ownership chains
old laws
testimony fragments
prior failures
```

### CIVIC

```text id="nsd8cp"
authority
legitimacy
rights
public obligation
emergency scope
```

### NULL

```text id="c9zvje"
false certainty
recursive command loops
dead authority reinforcement
data corruption
hostile optimization
```

### WITNESS

```text id="kfvz94"
evidence integrity
source-chain status
testimony binding
Chronicle eligibility
```

### REPAIR

```text id="d5v9gb"
repair frames
tool requirements
quality state
certification risk
future inspection needs
```

### TACTICAL

```text id="yr8av8"
threats
routes
cover
friendly positions
evacuation markers
```

## Field Deck Principle

```text id="5cc0zt"
The Field Deck should make uncertainty playable.
```

A good reading should sometimes say:

```text id="co606y"
classification uncertain
authority disputed
translation confidence low
repair possible but legitimacy risk high
system reports green but evidence contradicts status
```

---

# 13. Dungeon Faction Logic

Dungeons should not have generic enemy ownership.

Different factions interpret the same dungeon differently.

## Example: Old Waterworks

### Basin Repair Assembly

```text id="bp4lny"
The waterworks are public survival infrastructure.
Repair must be witnessed and teachable.
```

### Utility Sovereign

```text id="xv8yeb"
The waterworks contain proprietary control components and liability records.
Unauthorized repair creates risk.
```

### Continuance

```text id="xocv97"
The waterworks must remain under emergency control until chaos risk is eliminated.
Risk never fully disappears.
```

### Archive Witness Order

```text id="nfnz51"
The emergency authority chain must be read before it can be overturned.
```

### Null Logic

```text id="u75dvy"
Authority unresolved.
Continue lock reinforcement.
```

The same site creates multiple truths.

The dungeon should expose those interpretations through:

```text id="eupoqy"
signage
NPC dialogue
terminal logs
enemy behavior
legal warnings
repair permissions
Chronicle outcomes
```

---

# 14. Dungeon Archetypes

## 14.1 Civic Infrastructure Dungeon

Core fantasy:

```text id="luvmvq"
A public survival system is broken, locked, or captured.
```

Examples:

```text id="k0i6ow"
Old Waterworks
Power Substation
Floodgate Tower
Signal Relay
Desalination Plant
Bridge Control House
```

Primary loops:

```text id="31zyur"
repair
Device Bus
witness
combat defense
civic authorization
```

## 14.2 Ghost Archive Dungeon

Core fantasy:

```text id="wcy7za"
The records survived without the society that gave them meaning.
```

Examples:

```text id="ygmcu9"
flooded record hall
court archive
identity vault
land-claim server
dead voting chamber
Chronicle mirror
```

Primary loops:

```text id="c9abf2"
evidence recovery
source-chain verification
stealth
archive puzzles
legal confrontation
```

## 14.3 Rogue Factory Dungeon

Core fantasy:

```text id="b669fp"
Production continues after purpose has died.
```

Examples:

```text id="eotg44"
obsolete parts factory
toxic refinery
autonomous greenhouse
drone assembly plant
dead contract foundry
```

Primary loops:

```text id="hbhwly"
conveyor hazards
machine combat
shutdown dilemmas
blueprint recovery
production charter rewrite
```

## 14.4 Care Facility Dungeon

Core fantasy:

```text id="dhl9lk"
Protection has become captivity.
```

Examples:

```text id="590rif"
locked clinic
elder shelter
sedation ward
quarantine hospital
AI-managed hospice
```

Primary loops:

```text id="3p2gia"
nonlethal combat
triage
consent verification
patient testimony
machine care ethics
```

## 14.5 Convoy / Route Dungeon

Core fantasy:

```text id="685y4z"
The road is the dungeon.
```

Examples:

```text id="jwqm3s"
water convoy
medicine run
bridge crossing
refugee bus evacuation
mobile archive transfer
```

Primary loops:

```text id="k1l9x9"
route planning
vehicle repair
ambush defense
checkpoint negotiation
cargo triage
```

## 14.6 Ecological Wound Dungeon

Core fantasy:

```text id="9stdmm"
The environment is responding to injury, and humans call it hostility.
```

Examples:

```text id="e6od9o"
living wetland
toxic bloom basin
root-choked pump house
spore tunnel
beaver-flooded transit ruin
alien ecological boundary
```

Primary loops:

```text id="sgrneh"
biosecurity
ecological diagnosis
restoration
nonhuman witness
containment without extermination
```

## 14.7 Alien Translation Dungeon

Core fantasy:

```text id="dy713g"
The dungeon is dangerous because the categories are wrong.
```

Examples:

```text id="xt01cx"
translation borderland
quarantine boundary
sonar parliament ruin
lithic resonance chamber
aerosol choir tower
```

Primary loops:

```text id="ptgt4d"
translation calibration
hazard interpretation
nonhuman consent
low-confidence Field Deck readings
escalation avoidance
```

## 14.8 Offworld Habitat Dungeon

Core fantasy:

```text id="ozqkqq"
A pressure vessel is a constitution with leaks.
```

Examples:

```text id="io8k7j"
orbital derelict
lunar ice station
Mars habitat
asteroid refinery
Ceres water vault
```

Primary loops:

```text id="h6cvb3"
air pressure
life support
zero-g traversal
salvage law
emergency authority
closed-loop repair
```

---

# 15. Raid Structure

Raids should be structured as linked crises.

## Standard Raid Phase Model

```text id="v953tw"
Phase 0: Preparation
Phase 1: Approach
Phase 2: Breach
Phase 3: Diagnosis
Phase 4: Split Operations
Phase 5: Crisis Choice
Phase 6: Boss / System Collapse
Phase 7: Extraction
Phase 8: Public Reckoning
Phase 9: Chronicle and World Update
```

## Phase 0: Preparation

Before the raid:

```text id="ruf3aw"
fabricate parts
assign vehicles
secure permissions
brief NPCs
choose route
stock medicine
prepare evidence cases
calibrate Field Deck modes
negotiate faction support
```

Preparation should affect the raid.

## Phase 1: Approach

The raid begins before the site.

Examples:

```text id="nkavd2"
convoy ambush
storm approach
checkpoint conflict
signal blackout
refugee crowd
machine warning perimeter
```

## Phase 2: Breach

Players enter the site.

Breach may involve:

```text id="8dq4tc"
combat entry
stealth entry
witnessed entry
emergency override
negotiated access
illegal forced entry
```

Entry method should affect legitimacy.

## Phase 3: Diagnosis

Players discover the crisis is not what they thought.

Examples:

```text id="80uv4x"
the factory is producing medicine, not weapons
the floodgate protected one district by drowning another
the care AI has evidence of prior patient abuse
the alien boundary is preventing a real contamination event
the archive contains proof against the faction that sent the players
```

## Phase 4: Split Operations

Teams divide.

```text id="x5umpa"
repair team
security team
medical team
archive team
logistics team
systems team
civic team
```

## Phase 5: Crisis Choice

The raid presents a defining tradeoff.

Examples:

```text id="wx8pla"
fast restart or witnessed restart
destroy core or preserve evidence
evacuate patients or preserve life support
open floodgate or stage gradual release
publish scandal now or avoid riot
```

## Phase 6: Boss / System Collapse

The site fights to remain itself.

Boss phase should combine:

```text id="byzzep"
combat
machine state
repair pressure
NPC risk
evidence risk
time pressure
authority contradiction
```

## Phase 7: Extraction

Extraction is part of victory.

Players may extract:

```text id="kqt6xd"
people
records
machine cores
source chains
materials
biological samples
vehicles
living witnesses
```

## Phase 8: Public Reckoning

The raid returns to society.

Players face:

```text id="744clo"
hearing
public vote
faction accusation
care triage aftermath
repair inspection
route debt negotiation
memorial ritual
```

## Phase 9: Chronicle and World Update

The outcome changes the world.

Examples:

```text id="gnwdxr"
settlement state vector changes
new faction posture
new law proposal
new ruin access
new profession training
new resource flow
new enemy response
new worldline precedent
```

---

# 16. Raid Archetypes

## 16.1 Water Restoration Raid

```text id="dwyb4j"
Goal:
restore regional water flow

Sites:
pump station
reservoir
treatment plant
floodgate
public hearing chamber

Key professions:
Repair Technician
Systems Operator
Archive Witness
Logistician
Security Responder
Civic Mediator

Boss:
Floodgate Governor or Pump Authority Core

Outcome:
water law precedent
```

## 16.2 Factory Reclamation Raid

```text id="sfk4jz"
Goal:
reclaim or shut down rogue production

Sites:
loading yard
assembly floor
control room
worker archive
foundry heart

Key professions:
Systems Operator
Fabricator
Security Responder
Archive Witness
Repair Technician

Boss:
Recursive Foundry Heart

Outcome:
public foundry, destroyed factory, corporate seizure, or quarantine
```

## 16.3 Convoy Exodus Raid

```text id="vwftu0"
Goal:
move civilians, medicine, or archives through hostile territory

Sites:
origin camp
broken bridge
checkpoint
ambush zone
destination gate

Key professions:
Logistician
Security Responder
Field Medic
Civic Mediator
Scout

Boss:
Road Debt Marshal, Continuance checkpoint, or Null Signal Mast

Outcome:
route compact, militarized road, refugee betrayal, or public trust surge
```

## 16.4 Archive Liberation Raid

```text id="6mqmhg"
Goal:
recover records that change legitimacy

Sites:
flooded archive
identity vault
machine testimony hall
public hearing chamber

Key professions:
Archive Witness
Systems Operator
Scout
Security Responder
Field Medic

Boss:
Archive Construct

Outcome:
citizenship restored, land claims reopened, faction scandal, or dangerous precedent
```

## 16.5 Care Release Raid

```text id="xejelf"
Goal:
free patients from protective captivity without collapsing care

Sites:
triage lobby
sedation ward
life-support control
patient testimony room
care AI core

Key professions:
Field Medic
Archive Witness
Systems Operator
Civic Mediator
Security Responder

Boss:
Null-Care Matron

Outcome:
body-sovereignty charter, care collapse, rehabilitated AI, or public trauma
```

## 16.6 Alien Boundary Raid

```text id="pgfvsw"
Goal:
open appeal protocol with nonhuman containment system

Sites:
human perimeter
translation garden
risk display tower
containment threshold
nonhuman witness chamber

Key professions:
Ecologist
Archive Witness
Systems Operator
Civic Mediator
Security Responder

Boss:
Quarantine Boundary Intelligence

Outcome:
appeal accepted, breach disaster, nonhuman rights precedent, or permanent exclusion
```

---

# 17. Seedworks v0.1 Dungeon Plan

Seedworks v0.1 should ship one deep dungeon and one light repeatable site.

## 17.1 Deep Dungeon: The Old Waterworks

### Core Fantasy

```text id="r3hdv8"
Restore public water without letting dead authority keep governing the living.
```

### Players

```text id="i237b4"
1–4
```

### Site Wound

```text id="1hddwv"
Built for:
municipal drought adaptation and public water continuity

Modified by:
emergency automation bureau

Crisis:
aquifer collapse, migration surge, emergency rationing

Lock:
dead authority chain

Current occupant:
Null-reinforced maintenance logic and small drone ecology

Repair possibility:
Archive Witness override plus physical pump repair
```

### Primary Objectives

```text id="xsbibs"
enter waterworks
restore local power
recover pump history
repair pressure line
stage Device Bus restart
resolve dead authority lock
survive Null drone escalation
restore water flow
return to settlement
participate in first public reckoning
```

### Optional Objectives

```text id="9nj7sn"
save injured worker
recover dead technician Field Deck
preserve old emergency law record
extract contaminated sensor
avoid destructive pump restart
document Utility Sovereign firmware fragment
```

### Encounters

```text id="4hxvxo"
1. Storm approach
2. Worker argument outside sealed door
3. Dark entry hall
4. First drone contact
5. Flooded maintenance crawl
6. Injured NPC triage
7. Pump control terminal
8. Evidence chamber
9. Pressure line repair
10. Dead authority denial
11. Archive Witness override
12. Pump restart boss phase
13. Extraction under pressure surge
14. Settlement water return
15. Public reckoning
```

### Boss: Pump Authority Core

Layers:

```text id="c2smmh"
Physical Body:
pump chamber, pressure valves, maintenance drones

Control Logic:
deny public override under unresolved emergency law

Authority Claim:
emergency water continuity order

Protected Value:
prevent chaotic water access

Escalation:
unauthorized write
physical seal breach
Archive contradiction
Null diagnostic challenge
```

### Possible Outcomes

```text id="y2tcgu"
Witnessed Restoration
Emergency Restoration
Partial Stabilization
Illegal Bypass
Destructive Restart
Null-False Success
Deferred Crisis
```

### Chronicle Lines

```text id="8hh3h8"
Witnessed Restoration:
The Old Waterworks were restored under witness after dead emergency authority was overturned.

Emergency Restoration:
Water returned before law could catch its breath.

Illegal Bypass:
The pump moved, but the settlement inherited the question of who had the right to touch it.

Null-False Success:
The waterworks reported success. The basin learned too late that machines can lie by completing procedure.

Partial Stabilization:
The settlement survived on a wounded flow.
```

## 17.2 Light Repeatable Site: Salvage Pump Annex

Purpose:

```text id="7ukvi8"
small repeatable dungeon for materials, profession practice, hazard variation, and procedural history testing
```

Features:

```text id="e4ryug"
1–2 rooms
random hazard
small repair task
one evidence fragment
minor enemy patrol
salvage with provenance
optional source-chain recovery
```

Possible variants:

```text id="l2wcc0"
corporate annex
worker-guild annex
flood-damaged annex
Null-contaminated annex
refugee-ration annex
```

---

# 18. Progression Across Dungeons

Dungeons should not scale mainly by enemy health.

They should scale through:

```text id="u8fz27"
more ambiguous evidence
higher pressure timing
more profession splits
stronger faction stakes
greater system interdependence
harder extraction
more severe partial success consequences
more sophisticated Null logic
```

## Dungeon Tiering

### Tier 0 — Local Wound

```text id="vwk813"
one site
one system
small hostile pressure
local outcome
```

Example:

```text id="hxs604"
Old Waterworks
```

### Tier 1 — Settlement Wound

```text id="orn7rn"
multiple linked systems
NPC casualties possible
settlement state vector affected
public hearing required
```

Example:

```text id="3o89y3"
Care Facility
```

### Tier 2 — Regional Wound

```text id="598ep7"
multi-site mission
convoy or route component
faction identity shifts
major Chronicle precedent
```

Example:

```text id="k9hba1"
Drowned Pump District
```

### Tier 3 — Worldline Wound

```text id="ewwdhg"
planetary, orbital, alien, or Confluence-scale consequence
worldline history altered
migration or fork possible
```

Example:

```text id="lb1gan"
Quarantine Boundary Raid
```

---

# 19. Procedural Dungeon Generation

Symtropy should support procedural dungeon variation, but not meaningless random rooms.

Generate from history first.

## Dungeon Generation Pipeline

```text id="aq5iv2"
Worldline Seed
  ↓
Region Pressure Vector
  ↓
Site History
  ↓
Authority Failure
  ↓
Current Threat Ecology
  ↓
Profession Hook Set
  ↓
Encounter Layout
  ↓
Evidence Objects
  ↓
Boss Logic
  ↓
Chronicle Outcome Table
```

## Procedural Variables

```text id="xmvsry"
built_for
original_owner
authority_failure
dominant_crisis
current_threat
water_level
power_state
archive_integrity
Null_drift
faction_claims
repair_material_availability
NPC_presence
death_recovery_risk
```

## Example Variation: Pump Site

### Municipal Variant

```text id="9c0wep"
public signage
old emergency law
Archive Witness path
worker trust
```

### Corporate Variant

```text id="4l75b5"
subscription terminals
private firmware
Utility Sovereign contract lock
billing evidence
```

### Continuance Variant

```text id="91f13a"
sealed checkpoints
emergency command signage
ration logic
security drones
```

### Ghost Variant

```text id="57yr7z"
dead civic chamber
partial records
machine grief
source-chain fragments
```

### Null Variant

```text id="3ssx4p"
false green diagnostics
recursive lock reinforcement
hostile command chatter
corrupted Chronicle leads
```

---

# 20. Implementation Schema

## Dungeon Definition

```rust id="d1vkh5"
struct DungeonDefinition {
    dungeon_id: String,
    display_name: String,
    archetype: DungeonArchetype,
    site_wound: DungeonSiteWound,
    recommended_players: PlayerCountRange,
    estimated_duration_minutes: Range<u32>,
    required_systems: Vec<GameSystem>,
    profession_hooks: Vec<ProfessionHook>,
    encounter_sequence: Vec<EncounterDefinition>,
    boss: Option<SymtropyBoss>,
    possible_outcomes: Vec<DungeonOutcome>,
    chronicle_entries: Vec<ChronicleEntryTemplate>,
    settlement_effects: Vec<SettlementEffect>,
}
```

## Encounter Definition

```rust id="ks0rs5"
struct EncounterDefinition {
    encounter_id: String,
    encounter_type: EncounterType,
    location_tag: String,
    primary_verbs: Vec<ActionVerb>,
    field_deck_modes: Vec<FieldDeckMode>,
    threats: Vec<ThreatActor>,
    hazards: Vec<EnvironmentalHazard>,
    profession_tasks: Vec<ProfessionTask>,
    evidence_objects: Vec<EvidenceObject>,
    fail_forward_outcomes: Vec<FailForwardOutcome>,
}
```

## Dungeon Outcome

```rust id="0627ay"
struct DungeonOutcome {
    outcome_id: String,
    outcome_type: OutcomeType,
    conditions: Vec<OutcomeCondition>,
    settlement_effects: Vec<SettlementEffect>,
    faction_effects: Vec<FactionEffect>,
    chronicle_line: String,
    unlocks: Vec<Unlock>,
    burdens: Vec<ChronicleBurden>,
}
```

---

# 21. Acceptance Tests

A dungeon is ready for Symtropy only if it passes these tests.

## Test 1: Site Wound Test

```text id="tgug1i"
Can the team explain what this place was built to keep alive?
```

## Test 2: Outside World Test

```text id="hjs5r1"
Does the dungeon change something outside itself?
```

## Test 3: Profession Test

```text id="ublpgj"
Do at least three professions have meaningful non-token roles?
```

## Test 4: Field Deck Test

```text id="ahkou3"
Do Field Deck modes reveal layered uncertainty rather than simple objective markers?
```

## Test 5: Partial Success Test

```text id="et0xu5"
Are there meaningful outcomes between perfect success and failure?
```

## Test 6: Boss System Test

```text id="gdzret"
Is the boss a system with an authority claim, not only an enemy body?
```

## Test 7: Chronicle Test

```text id="2r015q"
Can the outcome become a durable Chronicle entry?
```

## Test 8: Moral Non-Fake Test

```text id="m71cnh"
Does the central choice have real costs on more than one side?
```

## Test 9: Death Integration Test

```text id="4rfids"
Can player death create source-chain recovery gameplay rather than only respawn?
```

## Test 10: Replay Variation Test

```text id="q0kswq"
Can the same dungeon vary by history, faction claim, hazard, or outcome without losing identity?
```

---

# 22. Design Rules

## Rule 1: No Loot Without Provenance

Every reward should have history, source, claim, or consequence.

## Rule 2: No Boss Without Belief

Every boss protects something, even if its method is unacceptable.

## Rule 3: No Victory Without Record

Major dungeon outcomes must create Chronicle entries.

## Rule 4: No Repair Without Future

A repair may solve the immediate crisis, but it should create inspection, maintenance, certification, or political follow-up.

## Rule 5: No Combat Without Civilian Meaning

Combat should protect, endanger, obscure, or reveal something beyond the fight.

## Rule 6: No Puzzle Without World Logic

Puzzles should arise from pressure, authority, power, memory, ecology, or machine state.

## Rule 7: No Dungeon Without a Wound

If the site is not wounded, it is just content.

## Rule 8: No Perfect Profession

Every profession can prevent harm and create harm.

## Rule 9: No Simple Reset

Failure should wound the future, not erase the attempt.

## Rule 10: No Dungeon Separate From Civilization

The dungeon is part of the settlement, region, faction ecology, and worldline.

---

# 23. Final Principle

```text id="kx5ois"
A dungeon is where the player enters a broken system.

A raid is where the broken system enters society.

The boss is what refuses correction.

The reward is what the world can now remember, repair, or argue about.
```

Final line:

```text id="dfjx5v"
Symtropy dungeons should not ask whether players can clear rooms.

They should ask whether players can survive the truth of what those rooms were built to hide.
```
