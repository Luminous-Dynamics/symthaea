---

title: Symtropy Profession Loops and Legibility Progression
status: canonical-draft
version: 0.1
scope: Seedworks professions, skillful gameplay loops, Field Deck literacy, witness reputation, co-op role differentiation
recommended_path: docs/seedworks/00_canon/PROFESSION_LOOPS_AND_LEGIBILITY_PROGRESSION_V0_1.md
---------------------------------------------------------------------------------------------

# Symtropy Profession Loops and Legibility Progression

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Working Title

**Professions Are Ways of Reading the World**

## Core Thesis

Symtropy should have profession mechanics, but not conventional RPG classes.

A profession in Symtropy is not a stat package.

A profession is a disciplined way of noticing, interpreting, touching, repairing, recording, and being held accountable by the world.

The player should not become powerful because numbers go up.

The player should become more capable because the world becomes more legible, their actions become more precise, and their history becomes more socially meaningful.

Core rule:

```text
Symtropy progression should make the world more legible, not the player more powerful.
```

---

# 1. Why Standard RPG Skills Are Wrong for Symtropy

Conventional RPG skill systems usually ask:

```text
Can the character do the thing?
```

Symtropy should instead ask:

```text
Does the player understand what kind of thing this is?
Who will trust the action?
What does the system actually accept?
What consequence will become history?
```

A normal RPG might turn a waterworks dispute into:

```text
Persuasion check passed.
Court authorizes bypass.
```

That is the wrong game.

Symtropy should turn the same moment into:

```text
The Basin Court distrusts emergency bypasses.
The Archive Witness record shows a prior bypass became permanent.
The repair technician can prove this bypass is reversible.
The Systems Operator can show the Device Bus rollback path.
The Field Medic argues the medbay cannot survive another hour without water.
The Convoy Union offers temporary tanker relief if route credits are honored.
The court authorizes a temporary bypass under public witness.
The Chronicle records the burden.
```

The drama is not whether the player has enough charisma.

The drama is whether the player can make reality legible enough for a society to act without lying to itself.

---

# 2. Profession Design Principle

A profession should create skillful play through four layers:

```text
1. Physical execution
2. Pattern recognition
3. Procedural judgment
4. Civic consequence
```

A good player should prevent harm that a careless player would not even know they caused.

Bad profession design:

```text
+10% repair speed
+5% rifle damage
Unlock terminal tier 3
Pass water engineering check
```

Good profession design:

```text
DIAG mode distinguishes pressure surge from corrosion.
Repair frame shows future certification risk.
CIVIC mode identifies legitimacy debt before override.
Chronicle warns that this action repeats an old failure pattern.
NPCs trust or distrust the procedure based on prior record.
```

Final profession rule:

```text
A profession is skillful when it changes what the player can read, not only what the player can do.
```

---

# 3. The Universal Profession Loop

Every profession should follow the same hidden structure.

```text
1. Notice a problem others might miss.
2. Interpret what kind of problem it really is.
3. Choose a procedure.
4. Execute a tactile, social, or analytical action under pressure.
5. Observe the system response.
6. Accept consequence.
7. Leave evidence, trust, repair, debt, or memory behind.
```

This creates a shared grammar across professions while allowing each role to feel distinct.

## Universal Profession Fields

```rust
struct ProfessionLoop {
    profession_id: ProfessionId,
    display_name: String,
    core_fantasy: String,
    primary_tools: Vec<ToolId>,
    primary_field_deck_modes: Vec<FieldDeckMode>,
    minute_loop: Vec<ActionVerb>,
    mastery_patterns: Vec<ReadablePattern>,
    common_failures: Vec<FailureMode>,
    civic_consequences: Vec<CivicConsequence>,
    settlement_metrics_affected: Vec<SettlementMetric>,
    chronicle_signatures: Vec<ChronicleOutcome>,
    co_op_dependencies: Vec<ProfessionDependency>,
}
```

---

# 4. Progression Without Character Levels

Symtropy should avoid a global character level.

Instead, progression should exist across five layers.

## 4.1 Origin Bias

Origin defines what the player notices first.

Origin is not a class.

It is a lived history.

Examples:

```text
Worker-Guild Mechanic:
Sees tool marks, machine wear, repair lineage, and bad maintenance earlier.

Archive Apprentice:
Sees authority chains, missing records, witness requirements, and legitimacy gaps earlier.

Refugee Charter Child:
Sees gatekeeping, ration systems, temporary credentials, and exclusion policy earlier.

Corporate Utility Defector:
Sees hidden subscription locks, private firmware claims, and liability traps earlier.

Null-Touched Survivor:
Sees false certainty, diagnostic loops, and corrupted machine confidence earlier.
```

Origin should affect:

```text
Field Deck emphasis
NPC opening trust
faction suspicion
starting obligations
dialogue framing
first-mission interpretation
```

## 4.2 Field Deck Literacy

Field Deck literacy controls how deeply the player can interpret readings.

Early game:

```text
SCAN:
Pressure unstable.
```

Later, with waterworks literacy:

```text
DIAG:
Pressure instability pattern suggests intermittent valve chatter, not pipe fracture.
```

Later, with civic procedure literacy:

```text
CIVIC:
Emergency override possible, but prior precedent shows high legitimacy debt unless witnessed.
```

Later, with Null experience:

```text
NULL:
Pump reports green status while downstream sensor contradicts flow state.
Possible false-certainty injection.
```

Field Deck literacy should not hide the world behind arbitrary locks.

It should progressively reduce ambiguity for players who have earned interpretive depth.

## 4.3 Witness Reputation

Witness reputation is not XP.

It is a social memory profile.

It records what kinds of actions the player has taken and whether institutions trust their evidence.

Examples:

```text
Preserved evidence chain during a failed repair.
Honored route credits during convoy crisis.
Filed ecological review instead of cutting living remediation roots.
Returned emergency authority after crisis.
Bypassed public process under pressure.
Lost source-chain continuity during death recovery.
```

Witness reputation should be qualitative.

Not:

```text
Basin Court Reputation: 72
```

Better:

```text
Basin Court Profile:
- Trusted for reversible emergency repairs.
- Distrusted on unilateral machine overrides.
- Recognized by two water witnesses.
- Prior bypass still under review.
```

## 4.4 Relationship History

NPCs should respond to what the player has done, not to persuasion stats.

Example:

```text
Mara Venn does not trust the player because they have high Charisma.
She trusts the player because they previously audited a steward labor contract and refused to erase worker testimony.
```

Relationship history should track:

```text
help given
harm caused
promises kept
promises broken
witnessed courage
procedural betrayal
care decisions
resource allocation choices
faction-specific memory
```

## 4.5 Chronicle Burden

Some progression should be burdensome.

A player should carry public scars.

Examples:

```text
Emergency Bypass Burden:
The player has used emergency authority before. Some factions trust their decisiveness. Others fear a pattern.

Source-Chain Scar:
The player was partially reconstructed after death. Archive systems request additional witness review.

Null Contact Residue:
NULL mode becomes more sensitive, but machine stewards distrust some readings.

Care Triage Mark:
The player made a hard medical allocation. Care factions respect their honesty. Families may still grieve.

Convoy Debt:
The player saved one convoy by sacrificing another route promise. Logistics factions remember both.
```

Burden gives progression emotional weight.

The player does not only gain capability.

They become historically entangled.

---

# 5. Profession Categories

Seedworks should support many professions eventually, but only a few should be deep in v0.1.

Recommended profession families:

```text
1. Repair Technician
2. Systems Operator
3. Field Medic
4. Archive Witness
5. Scout / Salvage Cartographer
6. Logistics / Convoy Planner
7. Civic Mediator
8. Ecologist / Bio-Steward
9. Fabricator / Materials Specialist
10. Security / Threat Response
```

Each profession should support basic participation by all players and deeper mastery by specialists.

Core rule:

```text
Everyone can do the basics.
Specialists see deeper consequences.
```

---

# 6. Profession Loop: Repair Technician

## Core Fantasy

```text
I can hear what the machine is trying to survive.
```

## Primary Tools

```text
Field Deck
hand welder
sealant injector
torque wrench
patch cable
repair frame projector
pressure tester
```

## Primary Field Deck Modes

```text
SCAN
DIAG
REPAIR
CIVIC
NULL
```

## Minute Loop

```text
scan damage
identify fault lineage
inspect material condition
choose repair frame
brace tool
clean surface
align frame
weld or seal
pressure-test
initialize node
request temporary authority
record repair
```

## Mastery Patterns

A skilled Repair Technician learns to distinguish:

```text
corrosion from pressure fatigue
pipe fracture from valve chatter
bad material from bad installation
temporary seal from certifiable repair
machine failure from authority failure
unsafe bypass from reversible emergency repair
```

## Skillful Inputs

```text
tool stability
heat control
surface preparation
alignment timing
brace tension
sealant pressure
cooldown discipline
pressure test timing
```

## Good Failure Modes

Repair failure should not be binary.

Possible outcomes:

```text
Certified Seal
Clean Emergency Seal
Rough Emergency Seal
Leaky Seal
Unsafe Seal
```

Consequences:

```text
seal holds but requires inspection
repair works but cannot be certified
pump restarts but downstream valve is stressed
workers distrust the repair lineage
Archive flags missing witness
Utility Sovereign disputes unauthorized modification
```

## Chronicle Examples

```text
The player gave the waterworks a new vein and named it before the settlement.

The pipe held, but the settlement could hear the weakness in the seal.

The player made the water move before the law agreed it should.
```

---

# 7. Profession Loop: Systems Operator

## Core Fantasy

```text
I do not hack the world. I negotiate with machines in their own grammar.
```

## Primary Tools

```text
Field Deck
patch cable
terminal
Device Bus console
script cartridge
logic analyzer
portable battery
```

## Primary Field Deck Modes

```text
DIAG
CIVIC
NULL
ARCHIVE
SHARE
```

## Minute Loop

```text
plug into terminal
read device state
trace dependencies
identify authority lock
simulate transaction
stage write
detect command chatter
commit or abort
watch physical effect
log outcome
```

## Mastery Patterns

A skilled Systems Operator learns to distinguish:

```text
machine denial from civic denial
dead authority lock from active safety lock
false green status from real stability
script fault from power fault
Null loop from ordinary automation bug
unsafe write from reversible staged transaction
```

## Skillful Inputs

```text
dependency tracing
transaction staging
fuel budgeting
rollback planning
signal comparison
manual override timing
privacy masking in Share Mode
```

## Good Failure Modes

```text
device accepts command but upstream system rejects it
script works but exceeds local energy budget
automation creates sorter loop
machine enters safe state
public infrastructure requires witness before commit
credentials are valid but morally disputed
```

## Chronicle Examples

```text
The operator made the pump answer, but not before the old law spoke through it.

The player refused the clean green lie and found the broken sensor behind it.

The waterworks accepted the command only after the settlement learned what it meant.
```

---

# 8. Profession Loop: Field Medic

## Core Fantasy

```text
Keeping people alive is not the same as optimizing bodies.
```

## Primary Tools

```text
med scanner
Field Deck
triage tags
portable stretcher
saline kit
sanitation kit
cooling blanket
medicine case
care ledger
```

## Primary Field Deck Modes

```text
SCAN
DIAG
CARE
CIVIC
WITNESS
```

## Minute Loop

```text
arrive at injury site
secure immediate danger
check breathing
stop bleeding
scan environment
classify risk
choose triage priority
request supplies
move or stabilize patient
record care decision
face social response
```

## Mastery Patterns

A skilled Field Medic learns to distinguish:

```text
shock from exhaustion
contamination from infection
panic from hypoxia
heat stress from trauma
care refusal from cognitive impairment
medical emergency from body-sovereignty dispute
```

## Skillful Inputs

```text
triage speed
resource prioritization
movement safety
sanitation discipline
patient stabilization
consent handling
family communication
```

## Good Failure Modes

```text
patient survives but medicine stock collapses
triage choice saves many but creates family grievance
treatment requires consent that is difficult to verify
clinic trust rises while security faction loses patience
care decision becomes public precedent
```

## Chronicle Examples

```text
The medic saved the body and inherited the argument over what care had cost.

The settlement learned that triage is not arithmetic.

A life was preserved, but the care ledger began to accuse everyone.
```

---

# 9. Profession Loop: Archive Witness

## Core Fantasy

```text
Truth is infrastructure.
```

## Primary Tools

```text
Field Deck
evidence case
witness seal
portable scanner
voice recorder
source-chain verifier
archive terminal
public hearing marker
```

## Primary Field Deck Modes

```text
ARCHIVE
CIVIC
WITNESS
NULL
SCAN
```

## Minute Loop

```text
inspect site
recover logs
verify source chain
interview witnesses
compare authority claims
detect missing records
flag contradiction
request hearing
bind evidence to Chronicle
authorize or block action
```

## Mastery Patterns

A skilled Archive Witness learns to distinguish:

```text
missing record from destroyed record
legal continuity from dead authority
testimony conflict from deliberate forgery
procedural delay from necessary caution
public memory from institutional self-protection
```

## Skillful Inputs

```text
evidence handling
timeline reconstruction
contradiction mapping
witness sequencing
chain-of-custody protection
hearing timing
```

## Good Failure Modes

```text
truth preserved but repair delayed
bad evidence chain weakens later civic action
witness process angers emergency factions
public trust rises but speed falls
old law is overturned but new precedent is unstable
```

## Chronicle Examples

```text
The witness did not fix the pump. They fixed the right to know why it was locked.

The settlement waited for truth and paid in thirst.

The old authority ended when its record was finally read aloud.
```

---

# 10. Profession Loop: Scout / Salvage Cartographer

## Core Fantasy

```text
The broken world can still be read.
```

## Primary Tools

```text
Field Deck
map slate
climbing line
hazard flags
salvage tags
compact drone
cutting tool
sample containers
```

## Primary Field Deck Modes

```text
SCAN
ARCHIVE
NULL
DIAG
TACTICAL
```

## Minute Loop

```text
enter ruin
listen for structural stress
map safe route
mark hazards
identify salvage
tag evidence
avoid contamination
extract useful parts
return route knowledge
update settlement map
```

## Mastery Patterns

A skilled Scout learns to distinguish:

```text
loot from evidence
safe route from temporarily quiet route
salvageable machine from sacred wreck
structural stress from ambient noise
Null ambush pattern from ordinary hazard
```

## Skillful Inputs

```text
route memory
weight management
hazard anticipation
quiet movement
visibility discipline
tagging accuracy
extraction timing
```

## Good Failure Modes

```text
valuable part recovered but evidence chain broken
safe route found but crosses disputed territory
map helps convoy but exposes refuge camp
salvage claim triggers faction dispute
```

## Chronicle Examples

```text
The scout brought back a map, and the map became an argument.

The ruin yielded parts, but not without losing some of its testimony.

The first safe path was not the most honest one.
```

---

# 11. Profession Loop: Logistics / Convoy Planner

## Core Fantasy

```text
Civilization is what arrives on time.
```

## Primary Tools

```text
route board
Field Deck
cargo manifest
vehicle terminal
fuel ledger
weather scanner
checkpoint credentials
convoy radio
```

## Primary Field Deck Modes

```text
SCAN
CIVIC
LOGISTICS
ARCHIVE
TACTICAL
```

## Minute Loop

```text
read settlement needs
inspect cargo
choose route
assign vehicle
check weather and threats
negotiate passage
drive or escort convoy
handle breakdown
verify delivery
record route outcome
```

## Mastery Patterns

A skilled Logistician learns to distinguish:

```text
urgent cargo from politically loud cargo
fast route from reliable route
checkpoint delay from faction pressure
ambush risk from weather risk
public delivery from private capture
```

## Skillful Inputs

```text
load balancing
fuel discipline
route planning
timing windows
risk tradeoff judgment
checkpoint negotiation
vehicle maintenance awareness
```

## Good Failure Modes

```text
convoy arrives late but intact
cargo arrives but route debt is owed
medical supplies saved while fabricator stalls
water delivered but convoy law favors insiders
route becomes militarized after repeated attacks
```

## Chronicle Examples

```text
The convoy arrived, and with it came the question of who had been left waiting.

The road was reopened, but the toll was memory.

The settlement learned that logistics is care with wheels.
```

---

# 12. Profession Loop: Civic Mediator

## Core Fantasy

```text
A society is a repair procedure for conflict.
```

## Primary Tools

```text
Field Deck
charter slate
public proposal board
hearing marker
faction ledger
rights floor reference
emergency authority clock
```

## Primary Field Deck Modes

```text
CIVIC
ARCHIVE
WITNESS
AFFECT
NULL
```

## Minute Loop

```text
read dispute
identify stakeholders
surface hidden cost
compare charter clauses
propose procedure
record dissent
call vote or witness
manage emergency scope
track legitimacy debt
```

## Mastery Patterns

A skilled Civic Mediator learns to distinguish:

```text
disagreement from exclusion
urgency from manipulation
consensus from paralysis
command from Continuance drift
law from dead authority
compromise from hidden coercion
```

## Skillful Inputs

```text
stakeholder sequencing
proposal wording
emergency scope definition
rights-floor interpretation
timing of vote versus action
dissent recording
```

## Good Failure Modes

```text
law passes but excluded faction radicalizes
emergency action succeeds but accrues legitimacy debt
public vote resolves water but worsens machine-rights dispute
slow process preserves legitimacy but costs resources
```

## Chronicle Examples

```text
The mediator did not end the conflict. They gave it a shape that could be survived.

The settlement voted with dry mouths.

The law held, but its shadow grew longer.
```

---

# 13. Profession Loop: Ecologist / Bio-Steward

## Core Fantasy

```text
The watershed is also machinery, but alive.
```

## Primary Tools

```text
Field Deck
water test kit
soil probe
species registry
sample jars
biosecurity tags
seed capsule
habitat marker
```

## Primary Field Deck Modes

```text
SCAN
DIAG
ARCHIVE
CIVIC
NULL
ECOLOGY
```

## Minute Loop

```text
scan soil or water
identify organism or ecological function
test contamination
compare trophic state
choose intervention
deploy species or restraint
monitor effect
negotiate objection
record ecological obligation
```

## Mastery Patterns

A skilled Bio-Steward learns to distinguish:

```text
obstruction from living remediation
invasive spread from healthy succession
pollution symptom from pollution source
ecosystem recovery from greenwashing
species utility from species agency
```

## Skillful Inputs

```text
sample timing
contamination control
habitat placement
intervention restraint
metric interpretation
faction communication
biosecurity discipline
```

## Good Failure Modes

```text
water quality improves but roots damage pipe
species helps one faction and threatens another
quarantine authority blocks deployment
restoration creates future maintenance duty
living system refuses instrumental use
```

## Chronicle Examples

```text
The pump was taught to bend around the living filter.

The player refused to mistake obstruction for failure.

The settlement learned that restoration creates obligations.
```

---

# 14. Profession Loop: Fabricator / Materials Specialist

## Core Fantasy

```text
Matter remembers where it came from.
```

## Primary Tools

```text
fabricator
Field Deck
caliper bench
materials scanner
provenance tags
quality witness station
part registry
```

## Primary Field Deck Modes

```text
DIAG
SCAN
CIVIC
ARCHIVE
REPAIR
```

## Minute Loop

```text
inspect material
check contamination
verify provenance
choose substitute
calibrate fabricator
produce component
quality test
label part
route to repair or logistics
```

## Mastery Patterns

A skilled Fabricator learns to distinguish:

```text
clean salvage from contaminated salvage
cheap substitute from dangerous substitute
certifiable part from emergency-only part
counterfeit material from old stock
material claim from legal trap
```

## Skillful Inputs

```text
calibration
tolerance reading
material selection
quality testing
contamination handling
part labeling
provenance review
```

## Good Failure Modes

```text
part fits but wears quickly
component works but cannot be certified
salvaged material carries faction claim
contaminated part spreads risk
fabrication shortcut creates later breakdown
```

## Chronicle Examples

```text
The part worked, but its metal remembered the ruin.

The settlement built with what it had, and what it had came with claims.

A fabricated valve became a legal object before it became a repair.
```

---

# 15. Profession Loop: Security / Threat Response

## Core Fantasy

```text
Protect without becoming Continuance.
```

## Primary Tools

```text
weapon
shield
Field Deck
nonlethal restraint
route markers
threat scanner
evacuation beacon
evidence recorder
```

## Primary Field Deck Modes

```text
TACTICAL
SCAN
CIVIC
NULL
WITNESS
```

## Minute Loop

```text
read threat
secure civilians
mark routes
hold perimeter
disable drones
choose lethal or nonlethal response
preserve evidence
protect repair crew
return authority after crisis
```

## Mastery Patterns

A skilled responder learns to distinguish:

```text
enemy from frightened system
active threat from obsolete defense loop
civilian panic from infiltration
security need from emergency-power drift
protection from control
```

## Skillful Inputs

```text
positioning
suppression
escort timing
target prioritization
nonlethal discipline
threat de-escalation
evidence preservation
authority handoff
```

## Good Failure Modes

```text
settlement safe but trust falls
threat neutralized but evidence destroyed
emergency powers remain too long
security faction gains weight
drones disabled but machine testimony lost
```

## Chronicle Examples

```text
The perimeter held, and then the settlement had to ask why the barricades stayed.

The player saved the convoy without letting fear write the law.

Protection became dangerous when it forgot to end.
```

---

# 16. Co-op Profession Design

Professions should create interdependence without hard class dependency.

Bad co-op design:

```text
Only Engineer can repair.
Only Medic can heal.
Only Archivist can open records.
Only Governor can vote.
```

Good co-op design:

```text
Everyone can attempt basic actions.
Specialists reduce ambiguity, improve quality, prevent hidden harm, and create better evidence.
```

## Example: Old Waterworks Co-op

Repair Technician:

```text
physically patches pipe
reads seal quality
prevents unsafe pressure restart
```

Systems Operator:

```text
stages pump transaction
detects dead authority lock
prevents false green restart
```

Archive Witness:

```text
verifies old emergency law
requests witness override
records legitimacy chain
```

Field Medic:

```text
argues medbay need
stabilizes injured worker
creates care urgency evidence
```

Logistician:

```text
delivers ceramic seal
routes water tanker backup
tracks delay cost
```

Civic Mediator:

```text
frames temporary repair authority
records dissent
limits emergency scope
```

Security Responder:

```text
holds corridor against Null drones
protects repair crew
avoids evidence destruction
```

Each player changes the same mission through a different literacy.

---

# 17. MVP Scope for Seedworks v0.1

Do not implement all professions deeply at first.

Seedworks v0.1 should prove four deep professions:

```text
Repair Technician
Systems Operator
Field Medic
Archive Witness
```

Support lighter versions of:

```text
Scout / Salvage Cartographer
Logistics Runner
Civic Mediator
Ecologist
Security Responder
```

## v0.1 Deep Profession Targets

### Repair Technician

Must support:

```text
one complete pipe repair loop
repair quality grades
Field Deck repair feedback
temporary certification
Chronicle outcome
```

### Systems Operator

Must support:

```text
one Device Bus terminal
read/write staged transaction
dead authority denial
rollback or safe abort
Archive Witness request path
```

### Field Medic

Must support:

```text
one injured NPC
triage choice
medicine scarcity
settlement metric effect
relationship consequence
```

### Archive Witness

Must support:

```text
one recoverable record
one dead authority conflict
one witness event
one Chronicle line
one legitimacy impact
```

## v0.1 Design Goal

The first playable mission should prove:

```text
Matter can break.
Machines can refuse.
Bodies can suffer.
Records can decide.
The player can repair, but not without consequence.
```

---

# 18. Progression UI

Avoid traditional skill trees as the primary UI.

Recommended interfaces:

```text
Field Deck Literacy Page
Witness Profile
Origin Lens
Profession Notes
Chronicle Burdens
Relationship Memory
Credential Wallet
```

## Field Deck Literacy Page

Shows what the player has learned to interpret.

Example:

```text
Waterworks Literacy:
- recognizes low-pressure seal instability
- distinguishes valve chatter from fracture
- understands temporary repair certification limits
- can read pump authority state
```

## Witness Profile

Shows social trust by domain.

Example:

```text
Witness Profile:
Basin Court:
  trusts reversible repairs
  distrusts unilateral overrides

Repair Guild:
  recognizes clean emergency seal
  requests formal apprenticeship

Refuge Compact:
  remembers water delivery during storm
  expects future aid
```

## Chronicle Burdens

Shows consequential public memory.

Example:

```text
Burdens:
- Emergency Bypass Under Review
- Null-Exposed Diagnostic Record
- Convoy Route Debt
```

## Credential Wallet

Shows formal permissions.

Example:

```text
Temporary Waterworks Operator
Archive Witness Assistant
Emergency Repair Token
Quarantine Escort Pending
```

---

# 19. Design Rules

## Rule 1: No Abstract Power Without Cost

Every new capability should create at least one new responsibility, visibility, or risk.

## Rule 2: No Skill Check Should Replace Understanding

A profession can clarify, warn, or stabilize.

It should not solve civic conflict by hidden number.

## Rule 3: Every Profession Must Produce Evidence

Professional actions should leave logs, marks, testimonies, quality states, or Chronicle hooks.

## Rule 4: Failure Should Be Informative

Failure should reveal how the world works.

A failed repair should teach pressure, material, authority, or trust.

## Rule 5: Combat Feeds Civilization

Security professions may include combat, but combat outcomes should affect roads, trust, safety, evidence, legitimacy, and future faction posture.

## Rule 6: Specialists See More, Not Everything

Mastery increases legibility.

It should not remove ambiguity entirely.

## Rule 7: Professions Should Create Future Problems

Good professional play does not erase consequence.

It creates better, more honest consequences.

---

# 20. Final Principle

```text
A profession in Symtropy is not a build.

It is a way of becoming responsible for a kind of truth.
```

Final line:

```text
The best player is not the one who can do everything.
The best player is the one who knows what kind of harm their profession can prevent.
```
