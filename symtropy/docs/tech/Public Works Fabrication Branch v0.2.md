---

title: Public Works Fabrication Branch v0.2
status: canonical-draft
milestone: seedworks-v0.2
scope: fabrication, recipes, repair grades, tool access, robotics dependencies
owner: design/engineering
depends_on:

* TECH_UNLOCK_TABLE_V0_1_TO_V0_3.md
* TECH_TREE_DEPENDENCY_SPINE.md
* DEVICE_BUS_SUBSTRATE_SYSTEMS.md
* ROBOTICS_PLATFORM_TECH_TREE_ADDENDUM.md
  recommended_path: docs/seedworks/00_canon/PUBLIC_WORKS_FABRICATION_BRANCH_V0_2.md

---

> **Code status (2026-07-02 review):** The only matching code is a thin `fabrication_allowed: bool` gate in `symtropy-sim-bridge`'s unrelated MK0 Bootstrapper Protocol — not this doc's public infrastructure fabrication system. Design/vision document only.

# Symtropy: Public Works Fabrication Branch v0.2

## Working Title

**The First Workshop**

## Core Thesis

v0.1 proves that one pipe can be repaired.

v0.2 proves that the settlement can learn to repeat repair.

Public Works Fabrication is not a crafting bench.

It is the first moment where individual repair becomes civic capacity.

Core rule:

```text
Fabrication is not the player making items.
Fabrication is the settlement learning what it can safely reproduce, certify, authorize, and maintain.
```

---

# 1. Purpose

The Public Works Fabrication Branch turns the Old Waterworks outcome into a repeatable local technology loop.

The player should move from:

```text
salvage → emergency repair → witnessed outcome
```

to:

```text
recovered materials → public fabrication → certified repair → stronger Proof-of-Repair → new settlement capability
```

v0.2 should teach that tools require:

```text
materials
power
recipes
bench access
quality checks
public authority
operator legitimacy
maintenance plans
```

Design rule:

```text
The settlement does not unlock better tools because it is richer.
It unlocks better tools because it can prove it knows how to use them responsibly.
```

---

# 2. Primary Unlock Condition

The Public Works Fabrication Branch unlocks after the v0.1 Old Waterworks loop reaches an accepted end-state.

Required:

```text
Old Waterworks outcome recorded
Chronicle JSONL event accepted
Proof-of-Repair receipt issued
Firstlight Public Repair Charter recognizes the repair
Field Deck source chain intact or recovered
```

Optional modifiers:

```text
repair grade
witness quality
power condition
cargo condition
chosen repair path
NPC trust
```

Example Field Deck message:

```text
CIVIC:
Proof-of-Repair recognized.

SITE:
Old Waterworks.

ACCESS UNLOCKED:
Public Works Fabrication Bench.

NOTE:
Certified fabrication requires material audit and pressure-test procedure.
```

Design rule:

```text
The first fabrication bench should feel earned by repair history, not purchased.
```

---

# 3. New v0.2 Facilities

## 3.1 Public Works Fabrication Bench

Device path:

```text
/dev/sym/fabrication/public_works_bench_01
```

Role:

```text
small infrastructure parts
certified repair tools
standard pipe components
basic housings
robotics subcomponents
```

Requires:

```text
Proof-of-Repair accepted
stable settlement power
public tool access
basic cargo ledger
operator identity
```

Player verbs:

```text
inspect
load material
select recipe
calibrate
fabricate
inspect output
certify
reject
log
```

Design rule:

```text
A bench is not a menu.
It is a public machine under charter.
```

---

## 3.2 Settlement Fabricator Bay

Device path:

```text
/dev/sym/fabrication/settlement_fabricator_bay
```

Role:

```text
larger assemblies
standardized parts
repeatable public works components
robotics frames
tooling cluster upgrades
```

Requires:

```text
Public Works Fabrication Bench active
power readout stable
cargo ledger audit available
safety charter approval
```

Design rule:

```text
The bay is where scavenged repair becomes repeatable manufacturing.
```

---

## 3.3 Public Tool Library

Device path:

```text
/dev/sym/tools/public_library
```

Role:

```text
shared tools
borrowed repair kits
certified gauges
inspection instruments
robot maintenance tools
```

Requires:

```text
Proof-of-Repair
repair-worker access class
tool checkout ledger
return policy
operator trust
```

Design rule:

```text
A public tool is not free.
It is entrusted.
```

---

## 3.4 Cargo Ledger Audit Station

Device path:

```text
/dev/sym/logistics/audit_station_01
```

Role:

```text
material verification
manifest disputes
contamination flagging
ownership disputes
cargo condition confirmation
```

Requires:

```text
Field Deck manifest reader
Chronicle reference
local cargo authority
basic sensor suite
```

Design rule:

```text
Fabrication begins by asking whether the material is what the crate says it is.
```

---

## 3.5 Fuel Depot Trust Console

Device path:

```text
/dev/sym/fuel/trust_console_01
```

Role:

```text
Proof-of-Repair recognition
fuel access
technician routing
emergency work assignment
```

Requires:

```text
Proof-of-Repair receipt
settlement charter
source-chain audit
```

Design rule:

```text
Proof-of-Repair opens doors. It is not a coin.
```

---

# 4. v0.2 Material Classes

v0.2 should introduce a small but meaningful material grammar.

## Material Classes

```text
scrap_metal
copper_conduit
ceramic_seal_blank
rubberized_gasket
filter_mesh
battery_cell
sensor_lens
control_board_blank
motor_winding_wire
recovered_bearing
biochar_filter_media
```

## Material States

```text
clean
oxidized
contaminated
bent
cracked
unverified
Null_suspect
evidence_grade
```

## Material Quality

```text
Q0 — Unverified Scrap
Q1 — Usable Salvage
Q2 — Bench-Cleaned
Q3 — Certified Public Works Grade
Q4 — Precision-Ready
```

v0.2 should mostly operate in Q1–Q3.

Q4 is foreshadowed for Mk0.5 tooling.

Design rule:

```text
Better fabrication begins with knowing what the material has survived.
```

---

# 5. Recipe Schema

Every recipe should use the same production format.

```text
Recipe:
Milestone:
Facility:
Inputs:
Required Tools:
Required Power:
Required Field Deck Mode:
Required Legitimacy:
Procedure:
Quality Checks:
Possible Outputs:
Failure Modes:
Chronicle Trigger:
Unlocks Next:
```

Design rule:

```text
No recipe without a procedure.
No procedure without a quality check.
No quality check without a consequence.
```

---

# 6. Core v0.2 Recipes

## Recipe 1: Certified Pipe Gauge

Milestone:

```text
v0.2
```

Facility:

```text
Public Works Fabrication Bench
```

Inputs:

```text
scrap_metal Q2
sensor_lens Q1
control_board_blank Q1
```

Required tools:

```text
calibration jig
Field Deck DIAG mode
pressure reference sample
```

Required power:

```text
stable bench power above 90%
```

Required legitimacy:

```text
Proof-of-Repair recognized
Public Tool Library access
```

Procedure:

```text
load cleaned metal
cut gauge body
seat sensor lens
flash measurement firmware
calibrate against pressure reference
seal gauge casing
register tool to public library
```

Quality checks:

```text
measurement drift
casing seal
firmware checksum
public tool ID
```

Possible outputs:

```text
Certified Pipe Gauge
Drifting Pipe Gauge
Uncertified Gauge Body
Rejected Scrap
```

Failure modes:

```text
CALIBRATION_DRIFT
FIRMWARE_CHECKSUM_FAIL
PUBLIC_TOOL_ID_MISSING
```

Unlocks next:

```text
Certified Seal Kit
Pressure Test Rig
Certified Repair Grade
```

Design rule:

```text
The first advanced tool should measure repair quality, not make repair easier by magic.
```

---

## Recipe 2: Certified Seal Kit

Facility:

```text
Public Works Fabrication Bench
```

Inputs:

```text
ceramic_seal_blank Q2
rubberized_gasket Q2
filter_mesh Q1
```

Required tools:

```text
Certified Pipe Gauge
thermal curing tray
seal inspection lamp
```

Required power:

```text
stable thermal cycle
```

Required legitimacy:

```text
public works access
repair-worker class
```

Procedure:

```text
inspect seal blank
trim gasket
seat gasket into ceramic ring
thermal cure
cool under pressure
inspect deformation
package with public seal ID
```

Quality checks:

```text
thermal deformation
gasket compression
seal ID logged
```

Possible outputs:

```text
Certified Seal Kit
Clean Emergency Seal
Warped Seal
Unsafe Seal
```

Failure modes:

```text
THERMAL_CYCLE_INTERRUPTED
GASKET_COMPRESSION_FAIL
SEAL_ID_NOT_LOGGED
```

Unlocks next:

```text
Certified Pipe Splice
Pressure Test Rig
Biofilter Housing
```

Design rule:

```text
Certification is the difference between “it held” and “the settlement may rely on it.”
```

---

## Recipe 3: Pressure Test Rig

Facility:

```text
Settlement Fabricator Bay
```

Inputs:

```text
scrap_metal Q3
copper_conduit Q2
Certified Pipe Gauge
battery_cell Q2
control_board_blank Q2
```

Required tools:

```text
Certified Pipe Gauge
DIAG mode
Public Works Fabrication Bench
```

Required power:

```text
bench power above 90%
```

Required legitimacy:

```text
public works charter
tool library registration
safety charter approval
```

Procedure:

```text
fabricate pressure chamber
install gauge mount
install pump interface valve
connect control board
flash test routine
register safety cutoff
run dry test
run wet test
commit rig ID
```

Quality checks:

```text
pressure accuracy
leak detection
emergency cutoff
operator safety log
```

Possible outputs:

```text
Certified Pressure Test Rig
Manual Pressure Test Rig
Unsafe Pressure Vessel
```

Failure modes:

```text
PRESSURE_SPIKE
EMERGENCY_CUTOFF_FAIL
LEAK_TEST_FAIL
```

Unlocks next:

```text
Certified Pipe Splice
Certified Repair Grade
Basic Transformer Repair Kit
```

Design rule:

```text
The rig makes repair repeatable because it lets failure happen safely before the public depends on the part.
```

---

## Recipe 4: Biofilter Housing

Facility:

```text
Public Works Fabrication Bench
```

Inputs:

```text
filter_mesh Q2
biochar_filter_media Q1
ceramic_seal_blank Q2
copper_conduit Q1
```

Required tools:

```text
Certified Seal Kit
DIAG mode
Cargo Ledger Audit
```

Required power:

```text
low
```

Required legitimacy:

```text
public water charter
contamination audit
```

Procedure:

```text
audit filter media
assemble housing
seat mesh layers
install seal
flush-test with graywater sample
log contamination state
```

Quality checks:

```text
flow rate
contamination flag
seal quality
filter media verification
```

Possible outputs:

```text
Biofilter Housing
Emergency Filter Housing
Contaminated Filter Assembly
```

Failure modes:

```text
CONTAMINATION_UNVERIFIED
FLOW_RATE_LOW
SEAL_LEAK
```

Unlocks next:

```text
Cold-Chain Vault
Hybrid Filter Alpha prerequisite
Tideborn trade readiness
```

Design rule:

```text
Before humanity can host alien filtration, it must prove it can build honest human filtration.
```

---

## Recipe 5: Basic Transformer Repair Kit

Facility:

```text
Public Works Fabrication Bench
```

Inputs:

```text
copper_conduit Q2
rubberized_gasket Q2
control_board_blank Q1
recovered_bearing Q1
```

Required tools:

```text
DIAG mode
Substrate Summary Page
Certified Pipe Gauge optional
```

Required power:

```text
bench power above 85%
```

Required legitimacy:

```text
public power access
safety authorization
```

Procedure:

```text
clean copper contacts
fabricate insulation spacer
test board continuity
assemble repair kit
register kit hazard class
```

Quality checks:

```text
contact resistance
insulation rating
voltage warning label
```

Possible outputs:

```text
Basic Transformer Repair Kit
Emergency Contact Kit
Unsafe Electrical Kit
```

Failure modes:

```text
CONTACT_RESISTANCE_HIGH
INSULATION_FAIL
VOLTAGE_LABEL_MISSING
```

Unlocks next:

```text
better voltage stability
robot charging dock
Public Works Fabrication reliability
```

Design rule:

```text
The workshop must learn to stabilize the power that makes the workshop possible.
```

---

## Recipe 6: Robot Crawler Motor Service Pack

Facility:

```text
Public Works Fabrication Bench
```

Inputs:

```text
motor_winding_wire Q2
recovered_bearing Q2
control_board_blank Q1
scrap_metal Q2
```

Required tools:

```text
basic winding jig
Field Deck DIAG
Public Tool Library access
```

Required power:

```text
stable bench power above 90%
```

Required legitimacy:

```text
robotics work permit
public inspection route authorization pending
```

Procedure:

```text
clean bearing
wind motor coil
mount rotor housing
flash low-autonomy controller
test torque under load
register motor pack
```

Quality checks:

```text
torque output
coil resistance
thermal rise
controller checksum
```

Possible outputs:

```text
Crawler Motor Service Pack
Weak Motor Pack
Overheating Motor Pack
Rejected Winding
```

Failure modes:

```text
WINDING_SHORT
TORQUE_LOW
THERMAL_RISE_HIGH
```

Unlocks next:

```text
mk0-scout Cable-Crawler
mk0.5-loom foreshadow
```

Design rule:

```text
The first robot begins as a repaired motor, not a personality.
```

---

## Recipe 7: Gantry Anchor Certification Kit

Facility:

```text
Public Works Fabrication Bench
```

Inputs:

```text
scrap_metal Q3
copper_conduit Q1
sensor_lens Q1
rubberized_gasket Q1
```

Required tools:

```text
Pressure Test Rig optional
Certified Pipe Gauge
DIAG mode
```

Required power:

```text
low
```

Required legitimacy:

```text
cargo movement charter
public safety approval
```

Procedure:

```text
fabricate anchor collar
install load indicator
seal anchor face
run static load test
register anchor point
```

Quality checks:

```text
load rating
mount stability
anchor ID
cargo movement log compatibility
```

Possible outputs:

```text
Certified Gantry Anchor
Manual Anchor
Unsafe Anchor
```

Failure modes:

```text
LOAD_RATING_LOW
ANCHOR_ID_MISSING
STATIC_TEST_FAIL
```

Unlocks next:

```text
mk0-gantry Sky-Hook
corpse / Field Deck recovery route
heavy cargo assist
```

Design rule:

```text
A gantry is safe only when the ceiling has agreed to become infrastructure.
```

---

# 7. Repair Grades v0.2

v0.1 repair grades are mostly emergency-grade.

v0.2 introduces certified repair outcomes.

## Repair Grade Ladder

```text
FAILED_REPAIR
No stable function.

UNSAFE_REPAIR
Works briefly but should not be authorized.

ROUGH_EMERGENCY_SEAL
Works under emergency justification.
Inspection required.

CLEAN_EMERGENCY_SEAL
Works cleanly but lacks full certification.

CERTIFIED_PUBLIC_WORKS_REPAIR
Meets public tool, material, pressure, and witness requirements.

PRECEDENT_REPAIR
Repair is strong enough to become a reusable public procedure.
```

## Grade Requirements

| Grade                         | Material | Tool       | Witness          | Test           | Civic Status       |
| ----------------------------- | -------- | ---------- | ---------------- | -------------- | ------------------ |
| FAILED_REPAIR                 | any      | any        | none             | fails          | none               |
| UNSAFE_REPAIR                 | Q0–Q1    | improvised | none             | untested       | denied             |
| ROUGH_EMERGENCY_SEAL          | Q1       | basic      | partial          | visual only    | temporary          |
| CLEAN_EMERGENCY_SEAL          | Q1–Q2    | basic      | partial/full     | basic pressure | temporary accepted |
| CERTIFIED_PUBLIC_WORKS_REPAIR | Q2–Q3    | certified  | full             | pressure rig   | authorized         |
| PRECEDENT_REPAIR              | Q3       | certified  | full + Chronicle | repeatable     | public procedure   |

Design rule:

```text
Better repair grades are not bigger numbers.
They are stronger claims about what the public may safely depend on.
```

---

# 8. Proof-of-Repair v0.2 Upgrade

v0.1 issues one Proof-of-Repair receipt.

v0.2 allows stronger receipts.

## v0.2 Receipt Fields

```json
{
  "receipt_type": "ProofOfRepair",
  "receipt_id": "por_public_works_0007",
  "site": "Old Waterworks",
  "node": "/dev/sym/water/patch_conduit_alpha",
  "work_type": "certified_public_water_repair",
  "repair_grade": "CERTIFIED_PUBLIC_WORKS_REPAIR",
  "materials": [
    "ceramic_seal_blank_Q2",
    "copper_conduit_Q2"
  ],
  "tools_used": [
    "Certified Pipe Gauge",
    "Pressure Test Rig"
  ],
  "witnesses": [
    "ArchiveWitnessCartridge_03",
    "PublicWorksBench_01",
    "Mara"
  ],
  "authority_basis": "Firstlight Public Repair Charter",
  "inspection_status": "passed",
  "chronicle_event": "evt_00000118",
  "transferability": "non_transferable_reputation",
  "recognized_access": [
    "Public Tool Library",
    "Fuel Depot Trust Console",
    "Repair Worker Access Class"
  ]
}
```

Design rule:

```text
v0.2 Proof-of-Repair should distinguish heroic emergency labor from certified public work.
```

---

# 9. Public Works Access Classes

v0.2 should introduce access classes rather than generic unlocks.

## Access Class 0 — Emergency Helper

Granted by:

```text
participation in v0.1 Old Waterworks repair
```

Allows:

```text
basic tool checkout
manual repair assistance
emergency cargo carry
```

## Access Class 1 — Recognized Repair Worker

Granted by:

```text
Proof-of-Repair accepted
```

Allows:

```text
Public Works Fabrication Bench use
Certified Seal Kit recipe
Fuel Depot Trust Console
```

## Access Class 2 — Certified Public Works Operator

Granted by:

```text
successful certified repair
pressure test passed
Chronicle entry accepted
```

Allows:

```text
Pressure Test Rig
Basic Transformer Repair Kit
robotics preparation recipes
cargo ledger dispute handling
```

## Access Class 3 — Public Works Steward

Granted by:

```text
multiple recognized repairs
public vote or charter appointment
```

Allows:

```text
tool library governance
robot route permission proposals
fabricator bay scheduling
infrastructure hearing participation
```

Design rule:

```text
Access is not level-gating.
It is public trust made operational.
```

---

# 10. First v0.2 Mission Chain

## Mission 1 — Reopen the Bench

Premise:

```text
The settlement agrees to let the player reopen the locked Public Works Fabrication Bench after the Old Waterworks repair.
```

Player actions:

```text
present Proof-of-Repair
restore bench power
audit bench tool manifest
clear dead-authority lock
run calibration test
```

Outcome:

```text
Public Works Fabrication Bench active
```

Chronicle:

```text
The bench opened not because someone owned it, but because the repair had been witnessed.
```

---

## Mission 2 — Certify the Seal

Premise:

```text
The player must fabricate a Certified Seal Kit to replace or reinforce the emergency waterworks repair.
```

Player actions:

```text
recover ceramic seal blank
audit material condition
fabricate seal kit
run inspection
install seal
pressure-test repair
```

Outcome:

```text
Old Waterworks repair upgraded from emergency to certified
```

Chronicle:

```text
The first repair saved the water. The second taught the settlement how to trust it.
```

---

## Mission 3 — The Tool Library Dispute

Premise:

```text
The Public Tool Library opens, but factions disagree over who may borrow certified tools.
```

Player actions:

```text
review charter
inspect tool checkout ledger
hear worker concern
decide access policy
publish tool rule
```

Possible outcomes:

```text
open commons access
restricted certified-operator access
Industrial Compact priority access
emergency-only tool access
```

Chronicle:

```text
The tool was public. The argument was what public meant.
```

---

## Mission 4 — Prepare the First Robot

Premise:

```text
The settlement can now fabricate a motor service pack for the mk0-scout Cable-Crawler, but route authorization is unresolved.
```

Player actions:

```text
fabricate motor service pack
inspect overhead cable route
submit route permission
test crawler dock
decide whether machine visual logs count as evidence
```

Outcome:

```text
mk0-scout becomes playable or visible-locked depending on civic decision
```

Chronicle:

```text
The first robot waited at the edge of the route, not for power, but for permission.
```

---

# 11. v0.2 Failure States

Fabrication should fail meaningfully.

```text
MATERIAL_UNVERIFIED:
recipe may complete, but output cannot be certified

POWER_SAG_DURING_CURE:
seal warps or becomes emergency-grade only

TOOL_ID_MISSING:
public tool cannot enter library

CARGO_LEDGER_DIVERGENCE:
material exists physically but manifest contradicts ownership or condition

PRESSURE_TEST_FAIL:
repair works visually but cannot receive certified status

CHARTER_ACCESS_DENIED:
player has materials and tools but lacks civic permission

WITNESS_CHAIN_BROKEN:
repair cannot produce higher-grade Proof-of-Repair

NULL_RECIPE_CONTAMINATION:
recipe interface shows subtly altered procedure

LABOR_DISPUTE_TRIGGERED:
automation or tool policy creates faction pressure
```

Design rule:

```text
A fabrication failure should produce a repair, audit, faction, or Chronicle problem — not just wasted materials.
```

---

# 12. Robotics Dependencies Created by v0.2

Public Works Fabrication should prepare robotics without making robotics dominate v0.2.

## `mk0-scout` Cable-Crawler Requires

```text
Robot Crawler Motor Service Pack
battery charger
Field Deck remote view
overhead cable route
public inspection permission
```

## `mk0-gantry` Sky-Hook Requires

```text
Gantry Anchor Certification Kit
cargo ledger integration
operator training
safety charter approval
```

## `mk0-aegis` Acoustic Boundary Requires

```text
audio bus node
microphone mesh
privacy boundary
public alert protocol
```

## `mk0-agora` Civic Kiosk Requires

```text
Chronicle link
public terminal
charter database
vote log integrity
```

Design rule:

```text
v0.2 should make the first robot plausible before making the first robot powerful.
```

---

# 13. Implementation Tickets

## F1 — Public Works Bench Stub

Add one usable fabrication bench.

Acceptance:

```text
bench is visible
locked until Proof-of-Repair accepted
unlocks after v0.1 outcome
supports at least one recipe
```

---

## F2 — Recipe Data Model

Implement minimal recipe schema.

Acceptance:

```text
recipe has inputs
recipe has tool requirements
recipe has power requirement
recipe has legitimacy requirement
recipe has possible outputs
recipe has failure modes
```

---

## F3 — Material Quality States

Add Q0–Q3 material quality.

Acceptance:

```text
material can be unverified, usable, bench-cleaned, or public works grade
recipe output depends on material quality
```

---

## F4 — Certified Seal Kit Recipe

Implement first complete recipe.

Acceptance:

```text
player loads materials
fabricates seal
receives output quality
can upgrade Old Waterworks repair
```

---

## F5 — Pressure Test Result

Add pressure-test result state.

Acceptance:

```text
repair can pass/fail pressure test
result influences repair grade
result influences Proof-of-Repair receipt
```

---

## F6 — Proof-of-Repair v0.2 Receipt

Upgrade receipt schema.

Acceptance:

```text
receipt includes tools used
materials used
repair grade
inspection status
recognized access
```

---

## F7 — Tool Library Access

Add Public Tool Library stub.

Acceptance:

```text
recognized repair worker can check out a certified tool
tool checkout is logged
tool can be returned or disputed
```

---

## F8 — First Robotics Dependency

Add visible `mk0-scout` dock dependency.

Acceptance:

```text
dock remains locked until motor service pack and route permission exist
Field Deck explains missing requirements
```

---

## F9 — Fabrication Failure Messages

Add failure outputs.

Acceptance:

```text
power sag can downgrade recipe
unverified material can block certification
missing witness can block stronger Proof-of-Repair
```

---

# 14. v0.2 Acceptance Test

v0.2 Public Works Fabrication succeeds if the player can:

```text
1. Present Proof-of-Repair to reopen the bench.
2. Audit material condition.
3. Fabricate a Certified Seal Kit.
4. Pressure-test the Old Waterworks repair.
5. Upgrade repair grade.
6. Receive a stronger Proof-of-Repair.
7. Unlock Public Tool Library access.
8. See the first robotics dependency become visible.
```

v0.2 fails if:

```text
fabrication feels like a generic crafting menu
recipe quality does not matter
civic authorization does not matter
Proof-of-Repair does not change access
robots appear without dependency or permission
```

---

# 15. Out of Scope for v0.2

Do not implement:

```text
full factory automation
large conveyor logistics
full robotics autonomy
alien hybrid fabrication
Shared Tool Embassy
large conveyor logistics
full robotics autonomy
alien hybrid fabrication
Shared Tool Embassy
full market economy
full regional trade
Atlas Gates
full multiplayer persistence
```

Foreshadow only:

```text
Shared Tool Embassy
Hybrid Filter Alpha
mk0.5-tooling cluster
regional technician passport
robot testimony dispute
```

Design rule:

```text
v0.2 should feel like the workshop door opening, not the whole industrial age arriving.
```

---

# 16. Final Principles

```text
v0.1:
The player repairs the pipe.

v0.2:
The settlement learns to repair pipes without requiring a miracle.

Fabrication is not crafting.
It is repeatable public trust.

A recipe is a procedure.
A tool is a responsibility.
A certified part is a promise.
A public bench is a political machine.

The first workshop should not make the player powerful.
It should make the settlement less fragile.
```

Final line:

```text
The bench did not give the settlement technology.
It gave the settlement a way to prove that the next repair would not depend on luck.
```
