---

title: Infrastructure Loom UI/UX Spec
status: canonical-draft
milestone: seedworks-v0.1-to-v0.3
scope: UI/UX, tech tree visualization, Field Deck modes, player interaction
owner: design/UI/engineering
depends_on:

* TECH_UNLOCK_TABLE_V0_1_TO_V0_3.md
* TECH_TREE_DEPENDENCY_SPINE.md
* PUBLIC_WORKS_FABRICATION_BRANCH_V0_2.md
* ROBOTICS_PLATFORM_TECH_TREE_ADDENDUM.md
* DEVICE_BUS_SUBSTRATE_SYSTEMS.md
  recommended_path: docs/seedworks/04_engine/INFRASTRUCTURE_LOOM_UI_UX_SPEC.md

---

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

# Symtropy UI/UX Spec: The Infrastructure Loom

## Working Title

**The Infrastructure Loom**

## Core Thesis

The tech tree in *Symtropy* must not look or behave like a fantasy skill tree.

It is not a menu of bonuses.

It is an in-world diagnostic, civic, and historical instrument that shows what the settlement can safely become.

Core rule:

```text
The tech tree should not ask:
“What have you unlocked?”

It should ask:
“What can your civilization now safely bear?”
```

---

# 1. Purpose

The Infrastructure Loom is the player-facing visualization of the Seedworks technology tree.

It shows:

```text
material dependencies
power readiness
computational capacity
civic legitimacy
maintenance support
Chronicle consequence
robotic agency
xeno-translation boundaries
```

It should make the player feel that technology is not abstract progress.

Technology is:

```text
built
powered
authorized
witnessed
maintained
remembered
disputed
```

Design rule:

```text
Every unlock is a public claim about what the world can now survive.
```

---

# 2. Core Visual Metaphor

The Infrastructure Loom should look like a living hybrid of:

```text
circuit diagram
subway map
waterworks schematic
public charter ledger
machine diagnostic graph
root system
civic archive
```

It should not look like:

```text
floating fantasy nodes
RPG skill constellations
abstract perk cards
shopping menu
research spreadsheet
```

The player should feel like they are standing in front of a public infrastructure board, not browsing a store.

Design rule:

```text
The Loom is a machine for seeing dependency.
```

---

# 3. Primary Layout

The Loom has three human trunk branches.

```text
Thermodynamic Material Fabrication
Computational Field Architecture
Socio-Civic Legitimacy Chains
```

These converge into:

```text
Robotics / Automation / Embodied Infrastructure
```

Beyond that sits:

```text
Xeno-Translation / Shared Tool Embassy
```

High-level topology:

```text
             XENO-TRANSLATION / SHARED TOOL EMBASSY
                              ▲
                              │
        ROBOTICS / AUTOMATION / EMBODIED INFRASTRUCTURE
              ▲               ▲               ▲
              │               │               │
 MATERIAL FABRICATION — COMPUTATION — LEGITIMACY
```

Design rule:

```text
Robotics is where human disciplines become embodied.
Xeno-translation is where human disciplines meet non-human obligations.
```

---

# 4. Access Points

The Infrastructure Loom should not exist only as a pause menu.

It should be accessible through several in-world interfaces.

## Field Deck

Portable, personal, limited view.

Best for:

```text
node details
dependency inspection
current mission requirements
repair consequence
Proof-of-Repair links
```

## Public Works Wall Terminal

Large public display.

Best for:

```text
settlement-wide readiness
public votes
major unlock branches
facility dependencies
v0.2 workshop planning
```

## Public Works Fabrication Bench

Material/fabrication slice.

Best for:

```text
recipes
material quality
tool access
repair grades
bench readiness
```

## Civic Kiosk

Legitimacy slice.

Best for:

```text
charters
permission envelopes
public access classes
vote locks
rights floor warnings
```

## Robot Dock

Robotics slice.

Best for:

```text
autonomy level
permission envelope
power needs
maintenance state
route authorization
witness capacity
```

## Shared Tool Embassy

Xeno-translation slice.

Best for:

```text
translation pools
alien exchange status
consent protocols
hybrid component risks
metabolic compatibility
```

Design rule:

```text
Different machines show different truths about the same future.
```

---

# 5. Field Deck Modes

The same Loom should change meaning depending on Field Deck mode.

## SCAN Mode

Shows physical reality.

Highlights:

```text
facilities
machines
pipes
benches
cargo routes
robot docks
missing components
material flow
physical blockages
```

Example:

```text
SCAN:
Public Works Fabrication Bench detected.

Missing:
- ceramic seal blanks
- certified pressure gauge
- stable bench power
```

Visual language:

```text
thin physical outlines
material routes
facility silhouettes
missing component icons
```

Design rule:

```text
SCAN answers: What exists?
```

---

## DIAG Mode

Shows technical health.

Highlights:

```text
power readiness
script budget
clock drift
thermal warnings
failure probability
device bus health
repair grade risk
```

Example:

```text
DIAG:
Pressure Test Rig unavailable.

Reason:
Transformer voltage below safe threshold.
Bench power stability: 84%.
Required: 90%.
```

Visual language:

```text
voltage bands
warning overlays
heat maps
failure risk pulsing
script budget gauges
```

Design rule:

```text
DIAG answers: What can fail?
```

---

## ARCHIVE Mode

Shows historical dependency.

Highlights:

```text
previous repairs
Chronicle events
old precedents
dead authority locks
lost blueprints
source-chain history
```

Example:

```text
ARCHIVE:
Public Works Fabrication Bench previously opened under Emergency Charter 2113.

Current conflict:
Authority expired.
Living charter required.
```

Visual language:

```text
root systems
old glowing paths
historical layers
date stamps
precedent tags
```

Design rule:

```text
ARCHIVE answers: What made this possible before?
```

---

## CIVIC Mode

Shows legitimacy.

Highlights:

```text
charters
public votes
access classes
permission envelopes
Proof-of-Repair requirements
rights floor warnings
faction objections
```

Example:

```text
CIVIC:
mk0-scout Cable-Crawler locked.

Missing:
- public inspection route authorization
- operator clearance
- machine witness policy
```

Visual language:

```text
locks
charter ribbons
vote markers
witness seals
public access indicators
```

Design rule:

```text
CIVIC answers: What is allowed?
```

---

## NULL Mode

Shows corruption, dead authority, and unsafe shortcuts.

Highlights:

```text
false unlocks
spoofed dependencies
dead-authority loops
unsafe bypasses
corrupted witness chains
Null drift
```

Example:

```text
NULL:
Unsafe shortcut detected.

Claim:
Fabrication Bench can open without witness chain.

Risk:
Recipe contamination.
Public tool ID spoof.
```

Visual language:

```text
red branch fractures
recursive lines
false nodes
glitching labels
broken witness glyphs
```

Design rule:

```text
NULL answers: What is pretending to be possible?
```

---

## WITNESS Mode

Shows evidence requirements.

Highlights:

```text
Proof-of-Repair
witness cartridges
machine logs
NPC testimony
Chronicle event IDs
repair-grade evidence
audit trails
```

Example:

```text
WITNESS:
Certified Seal Kit recipe available.

Evidence required:
- material audit
- pressure test pass
- tool checkout log
- recognized repair worker status
```

Visual language:

```text
seal stamps
evidence chains
signed nodes
source-chain paths
receipt links
```

Design rule:

```text
WITNESS answers: What can be proven?
```

---

# 6. Node Anatomy

Every tech node should use the same structure.

## Node Header

```text
Technology name
Milestone
Status
Discipline
Dependency layer
```

Example:

```text
mk0-scout Cable-Crawler
Milestone: v0.2
Status: VISIBLE_LOCKED
Discipline: Robotics / Field Architecture / Legitimacy
Layer: Robotics and Automation
```

## Readiness Bars

Every node has six readiness categories:

```text
MATERIAL
POWER
COMPUTATION
LEGITIMACY
MAINTENANCE
CONSEQUENCE
```

Example:

```text
MATERIAL:      78% — motor service pack missing
POWER:         71% — dock voltage stable
COMPUTATION:   58% — remote view available, route logging incomplete
LEGITIMACY:    49% — public route authorization missing
MAINTENANCE:   62% — crawler dock damaged
CONSEQUENCE:   55% — witness dispute possible
```

Design rule:

```text
A locked node should explain itself.
```

---

# 7. Node Status States

## PLAYABLE

The player can use this system directly.

Visual:

```text
solid node
clear dependency lines
Field Deck action available
```

## VISIBLE_LOCKED

The object or facility exists but is not usable.

Visual:

```text
solid silhouette
locked access ring
missing dependencies listed
```

## FORESHADOWED

The system is referenced but not interactable.

Visual:

```text
faint outline
distant branch
lore or archive note only
```

## STUB

Simplified implementation exists.

Visual:

```text
half-lit node
limited action warning
```

## ROADMAP

Designed but not in current build.

Visual:

```text
distant horizon layer
no direct player action
```

## DEFERRED

Explicitly out of scope.

Visual:

```text
dark branch
scope protection warning
```

Design rule:

```text
The player should know whether a lock is diegetic, technical, civic, or production-scope.
```

---

# 8. Dependency Lines

Dependency lines should be typed.

## Material Line

Represents:

```text
parts
tools
materials
facility access
```

Visual:

```text
copper / orange line
pipe-like
```

## Power Line

Represents:

```text
voltage
battery
transformer
thermal load
```

Visual:

```text
amber electrical trace
pulse speed shows stability
```

## Computation Line

Represents:

```text
Device Bus
script budget
Field Deck mode
WASM controller
mesh link
```

Visual:

```text
cyan circuit trace
data packets
```

## Legitimacy Line

Represents:

```text
charter
vote
Proof-of-Repair
witness
permission envelope
```

Visual:

```text
gold civic ribbon
seal icons
```

## Maintenance Line

Represents:

```text
repair procedure
inspection
spare parts
tool library
operator training
```

Visual:

```text
gray-white service route
wrench markers
```

## Consequence Line

Represents:

```text
Chronicle trigger
faction pressure
public outcome
risk propagation
```

Visual:

```text
thin white-gold line
ends in Chronicle glyph
```

Design rule:

```text
A line should tell the player what kind of dependency they are missing before they open the node.
```

---

# 9. Player Interaction

## Basic Actions

The Loom supports:

```text
pan
zoom
filter
select node
trace dependency
pin objective
simulate consequence
compare modes
open source chain
jump to facility
```

## Node Actions

Depending on node state, the player can:

```text
inspect
track requirement
view evidence
open charter
start recipe
request vote
submit witness
simulate risk
mark for later
```

Example:

```text
SELECT NODE:
Public Works Fabrication Bench

ACTIONS:
- Track missing power dependency
- View Proof-of-Repair requirement
- Open Charter Article 7
- Pin objective: restore bench power
```

Design rule:

```text
Every Loom interaction should produce a next action in the world.
```

---

# 10. Locked Node Explanation Pattern

Every locked node should answer:

```text
What is this?
Why is it locked?
What can I do now?
What evidence is missing?
What facility is missing?
What risk does it create?
What Chronicle event could unlock it?
```

Example:

```text
NODE:
Public Works Fabrication Bench

STATUS:
VISIBLE_LOCKED

WHAT:
A public machine for producing certified infrastructure parts.

WHY LOCKED:
Proof-of-Repair not yet accepted by Firstlight Public Repair Charter.

DO NOW:
Restore Old Waterworks and commit repair outcome.

EVIDENCE MISSING:
Archive Witness Cartridge signature.

FACILITY MISSING:
Bench power stable above 90%.

RISK:
Dead authority lock may spoof recipe access.

CHRONICLE UNLOCK:
OldWaterworksOutcomeRecorded
```

Design rule:

```text
A locked node should become a quest, not a frustration.
```

---

# 11. v0.1 Loom Scope

v0.1 should include only a tiny Loom.

## Required Nodes

```text
Field Deck Mk0
Patch Cable
Patch Conduit Mk0
Archive Witness Cartridge
Old Waterworks Pump
Thermodynamic Power Readout
Pump Audio Diagnostic
Chronicle JSONL v0
Proof-of-Repair Receipt
Public Works Fabrication Bench
```

## Required Branches

```text
Survival Repair
Field Deck / Device Bus
Power / Audio / Labor Substrate
Public Fabrication foreshadow
```

## Required Interaction

```text
select locked Public Works Fabrication Bench
see why it is locked
complete Old Waterworks repair
see Proof-of-Repair appear
see bench become future unlock
```

v0.1 should not include:

```text
full tech tree
all robotics nodes
all xeno-translation nodes
full regional horizon map
full player-customizable research routing
```

Design rule:

```text
v0.1 Loom should show that the future exists, not ask the player to manage all of it.
```

---

# 12. v0.2 Loom Scope

v0.2 expands the Loom into workshop planning.

## New Playable Nodes

```text
Public Works Fabrication Bench
Certified Seal Kit
Certified Pipe Gauge
Pressure Test Rig
Cargo Ledger Audit Station
Public Tool Library
Fuel Depot Trust Console
mk0-scout Cable-Crawler visible-locked/playable
mk0-gantry Sky-Hook visible-locked/playable
mk0-aegis Acoustic Boundary
mk0-agora Civic Kiosk
```

## Key Interaction

```text
player opens bench
selects Certified Seal Kit
sees material/power/civic requirements
fabricates part
pressure-tests repair
upgrades Proof-of-Repair
unlocks tool access
reveals robotics dependencies
```

Design rule:

```text
v0.2 Loom should make repeatable repair legible.
```

---

# 13. v0.3 Loom Scope

v0.3 introduces xeno-translation and living infrastructure.

## New Playable Nodes

```text
Shared Tool Embassy
Translation Pool
Metabolic Stabilizer
Rights Forum Terminal
Hybrid Filter Alpha
Bio-Electric Converter
Tideborn Chemical Memory Block
Canopy Root Wrapper Script
Translation Collapse
Overgrowth Without Consent
```

## Key Interaction

```text
player selects Hybrid Filter Alpha
sees human prerequisite stack
sees alien metabolic exchange requirement
sees Rights Forum consent requirement
sees Translation Collapse risk
calibrates Translation Pool
licenses hybrid block
```

Design rule:

```text
v0.3 Loom should make alien technology feel like a treaty under pressure.
```

---

# 14. Robotics Node UX

Robotics nodes require special display fields.

```text
platform class
autonomy level
permission envelope
power draw
script budget
mobility domain
sensor suite
witness capacity
civic restrictions
maintenance state
failure modes
```

Example:

```text
NODE:
mk0-scout Cable-Crawler

AUTONOMY:
L2 — Supervised Routine

PERMISSION:
Public inspection route required.

MISSING:
Motor Service Pack
Route Authorization
Crawler Dock Repair

WITNESS:
Visual log only.
May support but not replace human testimony.

FAILURE:
WITNESS_REJECTED
ROUTE_BOUNDARY_DENIED
POWER_SAG_MOTOR_DROP
```

Design rule:

```text
A robot node should be half machine spec, half civic permit.
```

---

# 15. Xeno-Tech Node UX

Xeno-tech nodes require special display fields.

```text
alien source
metabolic need
human wrapper
translation confidence
consent status
quarantine status
hybrid risk
rights forum status
failure mode
```

Example:

```text
NODE:
Hybrid Filter Alpha

ALIEN SOURCE:
Tideborn Water-Civic

METABOLIC NEED:
pH stability and flow continuity

HUMAN WRAPPER:
Bio-Electric Converter

CONSENT:
Rights Forum license pending

RISK:
2.4% Null Drift
Overgrowth Without Consent if maintenance window ignored
```

Design rule:

```text
Alien technology should never look like a normal recipe.
```

---

# 16. Null Corruption UX

NULL Mode should reveal dangerous false futures.

## Corruption Types

```text
false shortcut
dead-authority unlock
spoofed witness
unsafe recipe mutation
recursive dependency loop
permission bypass
fake Proof-of-Repair
```

Example:

```text
NULL MODE:
Fabrication Bench access appears unlocked.

WARNING:
Witness chain broken.
Recipe source cannot be verified.
Opening branch may contaminate public tool ledger.
```

Design rule:

```text
NULL Mode should tempt the player with convenience that reality cannot safely bear.
```

---

# 17. Chronicle Integration

Every major unlock should point to a Chronicle event.

Examples:

```text
OldWaterworksOutcomeRecorded
ProofOfRepairIssued
PublicWorksBenchReopened
CertifiedSealInstalled
ToolLibraryAccessPolicyPublished
RobotRouteAuthorized
MachineTestimonyAccepted
SharedToolEmbassyOpened
HybridFilterLicensed
TranslationCollapseContained
```

The Loom should show:

```text
what event unlocked this node
what event this node could produce
what faction memory it may alter
```

Design rule:

```text
A node is not fully unlocked until the Chronicle can remember why.
```

---

# 18. Loom Readiness Summary

The Loom should have a high-level readiness dashboard.

```text
Material readiness
Power readiness
Computation readiness
Legitimacy readiness
Maintenance readiness
Consequence readiness
```

Example:

```text
INFRASTRUCTURE LOOM READINESS

MATERIAL:      68%
POWER:         71%
COMPUTATION:   58%
LEGITIMACY:    49%
MAINTENANCE:   62%
CONSEQUENCE:   55%

OVERALL:
Developing

BLOCKER:
Legitimacy readiness below safe threshold.
```

Design rule:

```text
The readiness summary should tell the player what kind of civilization problem they currently have.
```

---

# 19. UI Copy Tone

The Loom should speak like a public infrastructure system.

Good tone:

```text
direct
procedural
civic
slightly worn
human but not cute
serious without being sterile
```

Avoid:

```text
gamey perk language
corporate gamification
loot language
“upgrade purchased”
“skill acquired”
“+5% efficiency”
```

Preferred phrases:

```text
Access recognized.
Witness missing.
Public authorization required.
Material unverified.
Power unstable.
Repair grade insufficient.
Charter conflict detected.
Chronicle precedent available.
```

Design rule:

```text
The UI should sound like a machine that knows public trust is heavy.
```

---

# 20. First Implementation Slice

## Minimal v0.1 Implementation

Implement:

```text
small node graph
one locked bench node
one repair node
one Proof-of-Repair node
one power node
one audio node
one Chronicle node
mode filter buttons
node detail panel
dependency trace highlight
```

## Required Screens

```text
Field Deck Loom View
Public Works Wall Terminal View
Node Detail Panel
Locked Dependency Panel
Proof-of-Repair Link Panel
```

## First Build Goal

The player should be able to:

```text
1. Open the Loom.
2. Select Public Works Fabrication Bench.
3. See it is locked by missing Proof-of-Repair.
4. Complete Old Waterworks repair.
5. Return to Loom.
6. See Proof-of-Repair node activated.
7. See Public Works Fabrication Bench shift from FORESHADOWED to VISIBLE_LOCKED or AVAILABLE_FOR_V0_2.
```

Design rule:

```text
The first Loom does not need many nodes.
It needs one node that changes because of what the player did.
```

---

# 21. Final Principles

```text
The tech tree is not a menu.
It is a civic diagnostic map.

The future is not bought.
It is made supportable.

Every node should have matter, power, computation, legitimacy, maintenance, and consequence.

Every lock should explain itself.
Every unlock should point back to history.
Every shortcut should carry risk.
Every future should ask what the settlement can safely bear.

SCAN shows what exists.
DIAG shows what can fail.
ARCHIVE shows what came before.
CIVIC shows what is allowed.
NULL shows what is lying.
WITNESS shows what can be proven.
```

Final line:

```text
The Infrastructure Loom did not show the player a list of upgrades.
It showed the settlement learning which futures it could carry without breaking.
```
