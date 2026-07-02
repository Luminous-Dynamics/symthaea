# Symtropy Design Doc: Earth Species Scope Firewall & Trophic Systems

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Working Title

**Seedworks: Earth Species as Living Infrastructure**

## Design Thesis

Earth species in *Symtropy* are not decorative biodiversity.

They are rare, functional, politically contested biological systems that interact directly with soil, water, atmosphere, settlement resilience, cognition, and governance.

The game should avoid simulating thousands of cosmetic organisms. Instead, it should use a curated catalog of high-impact Earth species whose ecological functions are legible, systemic, and emotionally meaningful.

The guiding production rule:

```text
Every Earth species must earn its simulation cost.
```

A species belongs in the game only if it changes at least one core system:

```text
soil recovery
water purification
erosion control
toxin remediation
pollination
waste cycling
food-web stability
biosecurity
civic rights
uplift politics
settlement sensing
```

---

# 1. Scope Firewall

## Hard Rule

*Symtropy* should not attempt to model Earth’s full biodiversity.

There are millions of real species on Earth. Simulating them individually would create production bloat, UI clutter, database complexity, and weak gameplay identity.

Instead, *Symtropy* uses two layers:

## Layer A: Visible Earth Species Catalog

A curated set of approximately **30–50 iconic Earth species**.

These are the species the player sees, scans, deploys, protects, negotiates over, or recovers from genetic vaults.

Examples:

```text
white willow
vetiver grass
mycorrhizal fungi
Deinococcus radiodurans
earthworms
beavers
corvids
dogs
kelp
phytoplankton
lichen
termites
wolves
vultures
duckweed
```

## Layer B: Abstract Trophic Matrix

A hidden ecological simulation layer representing broader food-web roles.

This can support approximately **100–160 trophic slots** without requiring every organism to be rendered, animated, or individually authored.

These slots represent ecological functions such as:

```text
wetland decomposer guild
temperate browsing herbivore guild
pelagic filter-feeder guild
desert detritivore guild
tundra lichen-fungal guild
deep crust chemosynthetic guild
```

This allows the world to behave ecologically without forcing the player or development team to manage thousands of species.

---

# 2. Species Design Standard

Every playable Earth species must define:

## Species Identity

```rust
SpeciesId
CommonName
ScientificName
BiomeTags
TrophicRole
ConservationStatus
GeneticSource
FactionSensitivity
```

## Ecological Function

```rust
SoilEffect
WaterEffect
ToxinEffect
ErosionEffect
BiomassEffect
FoodWebEffect
InfrastructureEffect
CivicEffect
```

## Gameplay Role

```rust
Deployable
Recoverable
Tradable
Protected
UpliftCandidate
QuarantineRisk
InvasiveRisk
SacredRelic
```

## Narrative Role

```rust
AssociatedFaction
ConflictUse
EthicalQuestion
FieldDeckEntries
ChronicleHooks
```

---

# 3. Core Species Tiers

## Tier 1: Micro-Engineers

Invisible or semi-visible organisms that modify baseline ecological variables.

They are not cosmetic particles. They are foundational infrastructure.

### Example: Mycorrhizal Fungi

Role:

```text
soil networking
nutrient exchange
forest recovery
toxin buffering
plant resilience
```

Gameplay function:

```text
accelerates soil recovery
reduces fertilizer dependency
improves drought tolerance
enables forest succession
```

Risk:

```text
may become invasive in alien ecologies
may transmit Earth biochemistry into non-Earth biospheres
may become politically controlled by seed corporations
```

### Example: Deinococcus radiodurans

Role:

```text
radiation remediation
biohazard reduction
ruin recovery
quarantine cleanup
```

Gameplay function:

```text
reduces radiation hazard ticks
helps reclaim irradiated ship hulls
enables toxic-zone access
supports post-disaster restoration
```

Risk:

```text
biosecurity panic
mutation anxiety
militarized cleanup contracts
```

---

## Tier 2: Keystone Flora

Plants function as living infrastructure.

They pump, filter, shade, anchor, cool, bind, and signal.

### Example: White Willow

Role:

```text
riparian filtration
water uptake
heavy-metal buffering
streambank stabilization
```

Gameplay function:

```text
reduces water toxin load
stabilizes polluted waterways
lowers mechanical filtration costs
creates habitat for wetland recovery
```

Risk:

```text
water diversion conflict
root intrusion into human infrastructure
possible incompatibility with alien hydrology
```

### Example: Vetiver Grass

Role:

```text
erosion control
slope stabilization
flood mitigation
soil retention
```

Gameplay function:

```text
reduces landslide probability
protects riverbanks
stabilizes low-tier settlement structures
slows flash-flood damage
```

Risk:

```text
monoculture dependency
misuse as cheap ecological bandage instead of watershed repair
```

---

## Tier 3: Structural Fauna

Animals that alter terrain, hydrology, decomposition, and settlement geometry.

They are ecological automation routines with agency.

### Example: Eurasian Beaver

Role:

```text
wetland engineering
water retention
aquifer recharge
flood slowing
habitat creation
```

Gameplay function:

```text
creates organic dams
raises local water table
expands wetland mesh
buffers drought
increases biodiversity recovery rate
```

Risk:

```text
conflict with farmers
settlement flooding
political dispute over autonomous animal engineering
```

### Example: Earthworms

Role:

```text
soil aeration
decomposition
nutrient cycling
compaction reversal
```

Gameplay function:

```text
improves soil quality
reduces greenhouse input costs
accelerates barren crust recovery
feeds decomposer pool
```

Risk:

```text
invasive outside native soil systems
can disrupt alien or fragile fungal layers
```

---

## Tier 4: Cognitive Clades

Species connected to uplift, companion intelligence, scouting, sensing, and civic ambiguity.

These are not merely pets or units. They sit near the boundary between fauna, tool, citizen, and ally.

### Example: Corvid Lineage

Role:

```text
aerial scouting
ruin mapping
signal-independent intelligence
adaptive tool use
```

Gameplay function:

```text
tracks drone swarms
maps ruins during electronic jamming
locates hidden data caches
warns of atmospheric changes
```

Civic question:

```text
When does training become employment?
When does uplift become citizenship?
```

### Example: K-9 Sentinel Strains

Role:

```text
hazard detection
companion sensing
audio warning
toxin tracking
subterranean patrol
```

Gameplay function:

```text
detects radiation leaks
warns of toxic weather before instruments stabilize
tracks missing people
supports rescue missions
```

Civic question:

```text
Are augmented animals equipment, partners, or protected workers?
```

---

# 4. Trophic Matrix Model

The visible species catalog is supported by a hidden trophic matrix.

The matrix tracks ecosystem function across major biome templates:

```text
marine / pelagic
freshwater / wetland
arid / desert
tropical rainforest
temperate / taiga
grassland / savanna
tundra / polar
deep crust / subterranean
```

Each biome tracks five trophic layers:

```text
producers
primary consumers
secondary consumers
apex regulators
decomposers
```

The simulation does not need to render every species.

Instead, each biome tile stores abstract trophic health values:

```rust
ProducerHealth
GrazerPressure
PredatorRegulation
ApexIntegrity
DecomposerCapacity
ToxinLoad
WaterStress
SoilCarbon
InvasivePressure
ExtinctionDebt
```

Visible species act as levers that modify these hidden values.

Example:

```text
Recovering wolves does not require simulating every wolf.
It restores ApexIntegrity in a temperate biome.
This reduces herbivore overpressure.
This protects vegetation.
This lowers erosion.
This protects water infrastructure.
```

---

# 5. Device Bus Integration

Earth species should communicate with the world simulation through the Symtropy Device Bus.

Example device paths:

```text
/dev/sym/bio/species/{species_id}
/dev/sym/bio/tile/{tile_id}/soil
/dev/sym/bio/tile/{tile_id}/water
/dev/sym/bio/tile/{tile_id}/toxins
/dev/sym/bio/tile/{tile_id}/trophic
/dev/sym/bio/vault/{vault_id}
/dev/sym/civic/bio_rights/{species_id}
```

## Example: White Willow Output

```json
{
  "species_id": "earth.white_willow",
  "tile_id": "firstlight.riparian.042",
  "water_toxin_delta": -0.07,
  "heavy_metal_capture": 0.12,
  "erosion_delta": -0.04,
  "water_table_delta": 0.03,
  "infrastructure_risk": "root_intrusion_low",
  "civic_flags": ["riparian_restoration", "water_commons_asset"]
}
```

## Example: Deinococcus Output

```json
{
  "species_id": "earth.deinococcus_radiodurans",
  "tile_id": "ghost_ruin.reactor_shell.011",
  "radiation_delta": -0.05,
  "biohazard_delta": -0.03,
  "quarantine_status": "controlled_remediation",
  "mutation_watch": true,
  "civic_flags": ["quarantine_authority_review"]
}
```

---

# 6. Field Deck Interface

The Field Deck should not immediately reveal a full scientific diagram.

Discovery should be staged.

## SCAN

Shows observed organism state.

```text
Species detected.
High water uptake.
Root-zone toxin concentration increasing.
Local stream toxicity decreasing.
```

## DIAG

Shows inferred ecological function.

```text
Likely phytoremediation behavior.
Heavy metal load accumulating in root tissue.
Mechanical filtration burden reduced by 12%.
```

## ARCHIVE

Shows historical human knowledge.

```text
White willow was used in riparian restoration and phytoremediation.
Pre-collapse records indicate high tolerance for wet soils and pollutant uptake.
```

## CIVIC

Shows political and legal implications.

```text
Species classified as living infrastructure.
Ownership claim disputed between Watershed Commons and Utility Sovereign contractor.
```

## NULL

Shows misuse or risk.

```text
WARNING:
Species deployment being used to justify continued upstream pollution.
Restoration effect does not neutralize source harm.
```

## Food-Web Discovery Rule

The player should not receive a complete food-web map instantly.

Instead:

```text
SCAN reveals organism condition.
DIAG reveals probable ecological role.
ARCHIVE reveals known Earth relationships.
Observation reveals local predator/prey/decomposer links.
CIVIC reveals governance consequences.
```

This keeps ecology mysterious, learnable, and systemic.

---

# 7. Genetic Vault Gameplay

Earth species are rare because Earth’s biosphere was damaged, fragmented, displaced, or politically enclosed.

Players recover species through:

```text
seed vaults
cryo-embryo banks
ruined biolabs
ark ships
monastery gardens
black-market gene caches
community seed libraries
parallel biome archives
```

Finding viable Earth genetics should feel like discovering industrial infrastructure, sacred relics, and political contraband at the same time.

Example loot:

```text
White Willow Seed Packet
Viable Earthworm Cocoons
Mycorrhizal Spore Ampoule
Corvid Embryo Archive
Kelp Culture Disk
Deinococcus Remediation Vial
```

Each recovered species creates a choice:

```text
restore
quarantine
patent
share
weaponize
ritualize
archive
destroy
```

---

# 8. Faction Reactions

## The Quiet Green

Views Earth species as sacred survivors and treaty-bearing ancestors.

Likely stance:

```text
restore carefully
share through commons
oppose patents
oppose careless alien deployment
```

## Utility Sovereigns

View species as infrastructure assets and biological IP.

Likely stance:

```text
license genes
optimize traits
patent firmware biology
deploy for profit
```

## Quarantine Authorities

View species as biosecurity risk.

Likely stance:

```text
restrict movement
sterilize unknowns
require containment
delay ecological restoration
```

## Uplift Collectives

View cognitive clades as possible civic persons.

Likely stance:

```text
protect corvids
protect dogs
oppose ownership of uplifted animals
demand representation protocols
```

## Watershed Commons

View keystone flora and fauna as hydrological partners.

Likely stance:

```text
restore beavers
plant willows
protect wetlands
oppose hard-channelized water systems
```

---

# 9. Seedworks v0.1 Vertical Slice

## Goal

Prove that biological actors can modify settlement infrastructure metrics through the Device Bus.

## Recommended MVP Species

### 1. White Willow

Primary mechanic:

```text
water filtration
heavy metal uptake
riparian stabilization
```

### 2. Deinococcus radiodurans

Primary mechanic:

```text
radiation remediation
biohazard reduction
ruin reclamation
```

### Optional 3. Earthworms

Primary mechanic:

```text
soil recovery
decomposer pool
greenhouse support
```

## MVP Tile Types

```text
polluted stream
ruined reactor shell
barren soil patch
settlement waterworks
seed vault chamber
```

## MVP Gameplay Loop

```text
1. Player scans polluted area.
2. Field Deck identifies toxin/radiation problem.
3. Player recovers viable Earth species from vault.
4. Player deploys species under controlled conditions.
5. Device Bus records biological remediation.
6. Settlement metrics improve.
7. Factions react.
8. Player must decide whether to scale, restrict, share, or privatize the organism.
```

## MVP Success Criteria

```text
Biology visibly changes infrastructure values.
Field Deck makes ecological function understandable.
Faction response creates political consequence.
Player understands species as living infrastructure, not decoration.
```

---

# 10. Why Not Start With Animal AI?

Animal AI pathfinding is useful later, but it is not the best first proof.

For Seedworks v0.1, the core risk is not whether an animal can walk.

The core risk is whether ecology can become infrastructure.

Therefore the first implementation should prioritize:

```text
plant remediation
microbial remediation
soil/water/toxin metrics
Field Deck readability
faction interpretation
```

Animal AI should come after the biological infrastructure loop works.

Suggested later animal test:

```text
beaver wetland engineering
corvid ruin scouting
K-9 hazard sensing
```

---

# 11. Design Principle

Earth species should never feel like collectibles.

They should feel like:

```text
living machines
ancestral survivors
ecological treaties
political assets
biosecurity risks
civic beings
infrastructure partners
```

Final design line:

```text
A seed is not loot.
It is a civilization decision waiting to germinate.
```
# Addendum: Ecological Conflict Resolution Grammar

## Purpose

The Earth Species system should not only answer:

```text
What does this organism do?
```

It must also answer:

```text
What happens when two valid claims collide over a living system?
```

A species in *Symtropy* is not simply a resource, unit, collectible, or biome decoration.

It is a living infrastructure actor that can create disputes between:

```text
ecology
settlement survival
property claims
faction law
biosecurity
alien biospheres
uplift rights
water politics
historical memory
```

The player’s role is not to “optimize nature.”

The player’s role is to resolve contested relationships without pretending that every value can be collapsed into a single resource score.

---

# 1. Lead Example: The Wolf Cascade

The Earth species system should be introduced through a simple trophic cascade.

```text
Recovering wolves does not require simulating every wolf.

It restores ApexIntegrity in a temperate biome.

This reduces herbivore overpressure.

This protects vegetation.

This lowers erosion.

This protects water infrastructure.
```

This is the core design pattern.

Visible species create emotional and narrative meaning.

The hidden trophic matrix carries systemic consequence.

The Device Bus makes those consequences legible.

The Chronicle remembers what the player chose when ecological, civic, and infrastructural claims collided.

---

# 2. Conflict Resolution Loop

Every major species dispute should use the same high-level grammar.

## Step 1: Detect

The Field Deck identifies a biological or ecological event.

```text
SCAN:
White willow root-zone toxin load rising.

DIAG:
Heavy metals being removed from stream water.

ARCHIVE:
Species historically used in riparian remediation.

CIVIC:
Upstream pollution license remains active.

NULL:
Restoration effect being used to justify continued contamination.
```

## Step 2: Identify Claims

The system surfaces competing claims.

Example claims:

```text
Watershed Commons:
The stream must be restored and protected.

Utility Sovereign:
Existing license permits upstream discharge.

Quiet Green:
The willow grove is a living restoration commons.

Quarantine Authority:
Bioaccumulated toxins create disposal risk.

Settlement Council:
Water supply must remain stable during transition.
```

## Step 3: Choose Procedural Frame

The player does not simply click “good” or “bad.”

They choose the process by which the dispute will be handled.

Possible frames:

```text
emergency order
restoration hearing
rights floor review
biosecurity quarantine
commons negotiation
technical audit
treaty council
direct action
temporary injunction
```

## Step 4: Apply Resolution Verb

The player chooses an action with both mechanical and civic consequences.

Resolution verbs:

```text
pause
permit
restrict
relocate
restore
quarantine
license
share
compensate
sanctuarize
monitor
escalate
ritualize
archive
decommission
```

## Step 5: Record Chronicle Outcome

The Chronicle records not only the result, but the meaning of the result.

Example:

```json
{
  "event_type": "EcologicalClaimResolved",
  "species_id": "earth.white_willow",
  "tile_id": "firstlight.riparian.042",
  "conflict": "riparian_restoration_vs_pollution_license",
  "resolution": "upstream_discharge_restricted",
  "procedure": "restoration_hearing",
  "affected_factions": [
    "watershed_commons",
    "utility_sovereign",
    "quiet_green",
    "settlement_council"
  ],
  "ecological_result": {
    "water_toxin_delta": -0.11,
    "erosion_delta": -0.04,
    "bioaccumulation_risk": "medium"
  },
  "civic_result": {
    "utility_sovereign_reputation_delta": -0.08,
    "watershed_commons_trust_delta": 0.14,
    "rights_floor_precedent": "living_infrastructure_consultation_required"
  },
  "chronicle_line": "The settlement chose to stop poisoning the stream instead of asking the willow to forgive it forever."
}
```

---

# 3. Species Disposition System

Recovered species should have formal disposition states.

These states are not always mutually exclusive.

A species can be restored by one faction, patented by another, ritually protected by a third, and quarantined by law.

## Core Dispositions

```text
Archived
Quarantined
Restored
Shared
Licensed
Patented
Ritual-Protected
Weaponized
Sanctuarized
Extinct-in-Wild
Invasive-Watch
Civic-Candidate
```

## Example: White Willow

Possible simultaneous states:

```text
Restored by Watershed Commons
Ritual-Protected by Quiet Green
Licensed by Utility Sovereign
Invasive-Watch by Quarantine Authority
```

This creates durable conflict.

The player is not choosing a single final fate.

They are shaping a contested status that can evolve.

## Disposition Change Rules

Disposition changes require different procedures.

```text
Archive → Restore:
requires viable genetic material, habitat suitability, and release approval.

Restore → Sanctuarize:
requires ecological success and civic recognition.

Restore → Invasive-Watch:
triggered by spread beyond approved biome boundary.

Shared → Patented:
requires legal capture, corporate influence, or failed commons defense.

Patented → Commons:
requires legal challenge, treaty override, or faction victory.

Quarantined → Restored:
requires biosecurity review and controlled release.

Civic-Candidate → Protected Personhood:
requires demonstrated agency, representation protocol, or uplift hearing.
```

---

# 4. Cognitive Clade Upgrade

Cognitive clades should not be defined by useful abilities alone.

They must have agency pressure.

A corvid that only scouts is a drone with feathers.

A K-9 that only detects hazards is a sensor with fur.

The design becomes interesting when they can disagree.

## Corvid Lineage Mechanics

Corvids may:

```text
refuse a scouting route
hide discovered objects
trade information
warn other corvids about player behavior
remember betrayal
recognize faction symbols
develop local route traditions
spread rumors between settlements
misclassify danger if stressed
protect juveniles over mission success
```

Field Deck example:

```text
SCAN:
Corvid scout returned without entering target ruin.

DIAG:
Avoidance pattern suggests remembered hazard, not disobedience.

ARCHIVE:
Corvid collectives historically display social learning and object permanence.

CIVIC:
Task refusal by uplift-adjacent clade may require representation review.

NULL:
Punitive retraining protocol inappropriate under agency uncertainty.
```

## K-9 Sentinel Mechanics

K-9 sentinels may:

```text
contradict sensor readings
refuse to enter contaminated zones
protect a child instead of completing patrol
detect fear before radiation
bond with specific handlers
suffer stress from repeated hazard deployment
trigger welfare review
be claimed as equipment by one faction and kin by another
```

Field Deck example:

```text
SCAN:
K-9 sentinel refusing reactor corridor entry.

DIAG:
Biometric stress spike precedes visible radiation reading.

ARCHIVE:
Working dogs historically detected hazards before instruments stabilized.

CIVIC:
Repeated forced deployment may violate uplift welfare protocol.

NULL:
Override command treats protected partner as disposable equipment.
```

## Core Principle

```text
Cognitive clades become interesting when usefulness conflicts with consent.
```

---

# 5. Device Bus Update Cadence

The Device Bus should not receive constant per-root, per-microbe, per-animal spam.

Ecological systems need layered update rates.

## Recommended Cadences

### Individual Event Layer

Used for direct player-visible events.

Examples:

```text
species deployed
species scanned
species dies
vault opened
animal refuses task
quarantine breach
faction claim filed
```

Cadence:

```text
event-driven
```

### Tile Ecological Layer

Used for local soil, water, toxin, and biomass changes.

Examples:

```text
riparian tile water quality
soil carbon
erosion probability
decomposer pool
toxin load
```

Cadence:

```text
every simulation tick, but aggregated before bus publish
```

### Trophic Matrix Layer

Used for biome-scale food-web changes.

Examples:

```text
ProducerHealth
GrazerPressure
ApexIntegrity
DecomposerCapacity
ExtinctionDebt
InvasivePressure
```

Cadence:

```text
slow tick
daily / seasonal / quest-phase update
```

### Chronicle Layer

Used only when an action becomes public consequence.

Examples:

```text
species restored
license restricted
quarantine declared
rights precedent created
cognitive clade protected
ecological disaster recorded
```

Cadence:

```text
rare, durable, signed event
```

## Bus Design Rule

```text
The Device Bus tracks state.
The Chronicle records consequence.
```

---

# 6. Example Conflict: Willow vs Pollution License

## Situation

A polluted stream is being restored by a White Willow grove.

The grove is reducing water toxicity, but the upstream factory still holds an active discharge license.

## Claims

```text
Watershed Commons:
Stop the pollution at source.

Utility Sovereign:
License is legal and economically necessary.

Quiet Green:
The willow grove is being used as a sacrificial filter.

Settlement Council:
Water supply must not collapse.

Quarantine Authority:
Bioaccumulated toxins in willow tissue are hazardous.
```

## Player Options

### Option A: Let License Continue

Immediate result:

```text
settlement water remains stable
Utility Sovereign approval rises
willow toxin load increases
Quiet Green trust falls
long-term biohazard risk increases
```

Chronicle line:

```text
The settlement chose clean numbers and dirty roots.
```

### Option B: Restrict Discharge

Immediate result:

```text
factory output drops
water quality improves
Utility Sovereign approval falls
Watershed Commons trust rises
willow recovery improves
```

Chronicle line:

```text
The settlement stopped asking the grove to absorb what law refused to name.
```

### Option C: Quarantine Grove

Immediate result:

```text
biohazard risk contained
water filtration benefit decreases
stream toxicity rises temporarily
Quarantine Authority approval rises
Quiet Green outrage rises
```

Chronicle line:

```text
The grove was fenced not because it failed, but because it had succeeded too well.
```

### Option D: Commons Hearing

Immediate result:

```text
resolution delayed
faction representatives gather
new evidence unlocks
public legitimacy rises
short-term ecological harm continues
```

Chronicle line:

```text
The settlement admitted the stream had more than one claimant.
```

---

# 7. Example Conflict: Corvid Refusal

## Situation

A corvid scout collective refuses to enter a ruin after previous birds failed to return.

The player needs the ruin map to avoid Null drone patrols.

## Claims

```text
Player Settlement:
The map is needed for survival.

Corvid Collective:
The route is remembered as death.

Uplift Collective:
Refusal is meaningful and must be respected.

Security Faction:
Birds are trained assets and should obey.

Field Deck:
Hazard evidence incomplete.
```

## Player Options

### Option A: Force Deployment

Result:

```text
map chance increases
corvid trust collapses
future refusals become more likely
uplift faction hostility increases
possible Chronicle stain
```

### Option B: Negotiate Alternate Route

Result:

```text
mission delayed
corvid trust increases
partial map gained
new hazard route discovered
```

### Option C: Send Human Team Instead

Result:

```text
human risk increases
corvid trust increases
settlement morale mixed
possible rescue mission triggered
```

### Option D: Abandon Ruin

Result:

```text
immediate risk avoided
resource opportunity lost
corvid trust increases
security faction approval falls
```

Core principle:

```text
Refusal is not a failure state.
It is relationship data.
```

---

# 8. Design Rule for Species Politics

A species becomes systemically meaningful when it can be:

```text
useful to one faction
sacred to another
dangerous to a third
legally ambiguous to a fourth
ecologically necessary to everyone
```

That is the ideal *Symtropy* species.

Not a collectible.

Not a stat modifier.

A living dispute.

---

# 9. New Closing Principle

The previous closing line should remain:

```text
A seed is not loot.
It is a civilization decision waiting to germinate.
```

But the refined systems principle should be:

```text
Ecology becomes gameplay when restoration creates obligations.
```

# Addendum: Eco Benchmark & 30-Minute Onboarding Firewall

## Purpose

*Eco* is the clearest existing high-water mark for ecological multiplayer civilization simulation.

It proves that players can understand and care about:

```text
ecological collapse
collective governance
resource extraction
pollution
law
economics
shared-world consequence
```

*Symtropy* should not dismiss *Eco*.

It should learn from it.

The goal is not to copy *Eco*.

The goal is to move one layer deeper.

Where *Eco* makes ecological and legal systems playable through accessible menus, *Symtropy* makes ecology, law, interface, memory, and infrastructure part of the same diegetic cybernetic substrate.

---

# 1. Comparative Design Position

## Eco’s Strength

*Eco* is strong because its premise is immediately legible:

```text
Players share a fragile world.
Players need resources.
Resource extraction damages the ecosystem.
Players create laws to survive together.
```

This is elegant.

A new player can understand the fantasy quickly.

That accessibility is not a weakness.

It is the benchmark.

## Symtropy’s Differentiation

*Symtropy* should not merely simulate ecological consequences.

It should simulate the systems that decide whether ecological consequence is seen, governed, denied, archived, or inherited.

The difference:

```text
Eco:
A society manages a shared ecosystem.

Symtropy:
A broken civilization negotiates reality through devices, laws, species, witnesses, memories, and infrastructure.
```

---

# 2. What Symtropy Must Preserve From Eco

## Immediate Physical Legibility

The player must understand the first action before understanding the full system.

The first 10 minutes should not ask the player to understand:

```text
trophic matrix abstraction
rights floor ambiguity
device bus governance
Chronicle signatures
faction legitimacy
species disposition states
```

The first 10 minutes should ask the player to do something physical:

```text
walk to a broken waterworks panel
pull open a corroded hatch
plug in a copper diagnostic cable
raise the Field Deck
watch amber text blink awake
hear water begin moving again
```

The systems can be deep.

The first action must be simple.

## Visible Cause and Effect

The first ecological loop must be obvious:

```text
dirty water
scan it
restore partial flow
deploy biological filtration
water quality improves
someone objects
the Chronicle remembers
```

A player should feel the system before they understand the architecture.

---

# 3. Where Symtropy Goes Deeper

## Law as Infrastructure

In *Symtropy*, law is not only a menu rule.

Law is an executable constraint over settlement infrastructure.

Example:

```text
A water pump is not simply on or off.

It can be:
physically powered
cryptographically locked
legally restricted
faction-contested
ecologically harmful
historically misclassified
overrideable only under emergency procedure
```

A law should be able to affect:

```text
pump access
valve priority
water rationing
species deployment
quarantine boundaries
repair authorization
data visibility
override rights
```

Design rule:

```text
A law is real when it changes what the world lets the player do.
```

## Interface Sovereignty

The Field Deck is not just a UI.

It is a political instrument.

Different factions may alter:

```text
what data is visible
what history is accessible
which warnings appear first
what actions are framed as legal
what ecological harm is hidden
what claims are considered legitimate
```

The player is not only reading the world.

The player is reading through a contested instrument.

Design rule:

```text
The interface is part of the conflict.
```

## Causal History

Every Seedworks site should have a procedural or authored causal history.

The player is not entering a blank survival map.

The player is entering a broken continuity.

A site should answer:

```text
What failed here?
Who benefited?
Who was locked out?
What law kept running after its authors died?
What ecological wound is still being misread as noise?
What did the previous settlement refuse to see?
```

Design rule:

```text
A ruin is a system that continued after responsibility ended.
```

## Architectural Permanence

The player’s Field Deck should preserve:

```text
local source chains
signed observations
Chronicle events
blueprints
species records
witness statements
faction precedents
```

If a settlement collapses, is captured, or becomes illegitimate, the player should not lose all meaning.

They should be able to carry evidence, memory, and partial continuity elsewhere.

Design rule:

```text
Worlds may fail.
Witness should survive.
```

---

# 4. The Onboarding Wall

The greatest risk to *Symtropy* is not lack of depth.

The greatest risk is revealing depth too early.

The player should not meet the architecture as an explanation.

The player should meet it as a tool.

## Forbidden First-30-Minute Concepts

Do not explain these in the first 30 minutes:

```text
full trophic matrix
all Field Deck modes
complete faction topology
cryptographic governance model
Chronicle event schema
species disposition ontology
alien personhood doctrine
WASM microcontroller architecture
```

These may exist under the hood.

They should not be front-loaded.

## Required First-30-Minute Feelings

The player should feel:

```text
this place is broken
my tools are physical
the interface is alive with history
water matters
law can block repair
biology can repair what machines cannot
someone disagrees with my solution
my choice will be remembered
```

## Required First-30-Minute Actions

The first playable experience should include:

```text
walk
inspect
plug in
scan
restore partial power
read a Field Deck warning
deploy or authorize one biological intervention
face one faction objection
make one reversible civic choice
see one ecological metric change
receive one Chronicle line
```

---

# 5. First Playable Build Priority

The first playable test should prioritize the Field Deck pipeline before advanced movement or combat polish.

## Priority 1: Field Deck Render-Texture Pipeline

The Field Deck must feel like a real object in the world.

It should support:

```text
held physical screen
offscreen render texture
diegetic amber/green diagnostic text
mode switching
scan target lock
signal noise
local source-chain display
simple civic warning
simple Chronicle event display
```

Without this, *Symtropy* becomes another menu sim.

## Priority 2: Tactile First-Person Interaction

Movement must feel responsive enough that the player trusts the body.

Minimum requirements:

```text
stable walking
clean looking
smooth object focus
believable hand/device raise
cable plug interaction
light camera sway
no nausea
no floaty delay
```

Do not over-polish combat recoil yet.

The important tactile fantasy is not shooting.

The important tactile fantasy is:

```text
I connected my instrument to a broken civilization, and the world answered.
```

## Priority 3: First Ecological Loop

Implement one biological infrastructure loop:

```text
White Willow Grove
Polluted Stream
Waterworks Panel
Field Deck Scan
Toxin Delta
Faction Objection
Chronicle Record
```

Optional second biological loop:

```text
Deinococcus Remediation Vial
Irradiated Ruin Panel
Radiation Delta
Quarantine Flag
```

---

# 6. The First 10-Minute Spine

The first playable prototype should follow this spine:

## Minute 0–2: Arrival

The player enters a broken wetland waterworks site.

Visible elements:

```text
stagnant water
dead pumps
corroded signage
distant settlement domes
willow roots invading concrete
faint bird movement
```

## Minute 2–5: Contact

The player finds a rusted diagnostic port.

They physically plug in the Field Deck cable.

The Deck wakes.

```text
SCAN:
Water flow interrupted.
Toxin load elevated.
Root system detected in pump channel.
```

## Minute 5–8: Interpretation

The player switches to DIAG.

```text
DIAG:
White willow roots filtering heavy metals.
Mechanical obstruction partially caused by living remediation system.
```

The player now understands the conflict:

```text
The plant is helping.
The plant is also blocking old infrastructure.
Removing it may restore flow but damage filtration.
```

## Minute 8–12: First Choice

The player chooses one of three actions:

```text
cut roots and restore full mechanical flow
reroute pump around root zone
pause repair and file ecological review
```

Each choice changes metrics.

Each choice creates faction response.

Each choice records a Chronicle line.

---

# 7. Example First Choice Outcomes

## Cut Roots

Immediate result:

```text
water pressure improves
toxin load rises
Quiet Green trust falls
Utility Sovereign approval rises
```

Chronicle line:

```text
The settlement made the water move, but asked it to forget what the roots had caught.
```

## Reroute Pump

Immediate result:

```text
water pressure partially improves
toxin reduction preserved
repair cost increases
Watershed Commons trust rises
```

Chronicle line:

```text
The pump was taught to bend around the living filter.
```

## File Ecological Review

Immediate result:

```text
repair delayed
legitimacy rises
short-term settlement frustration rises
new evidence unlocks
```

Chronicle line:

```text
The player refused to mistake obstruction for failure.
```

---

# 8. Production Rule

Every first-slice system must pass the Patch Cable Test.

## Patch Cable Test

A system belongs in Seedworks v0.1 only if it makes the first cable-plug moment more meaningful.

Allowed:

```text
Field Deck scan modes
water quality metric
one species intervention
one legal lock
one faction objection
one Chronicle event
```

Deferred:

```text
full species catalog
full trophic matrix
animal AI
complex combat
multiplayer governance
large biome simulation
deep crafting economy
alien diplomacy
worldline migration
```

The first slice should not prove the whole game.

It should prove the sentence:

```text
A broken world can be read, repaired, contested, and remembered through a physical interface.
```

---

# 9. Updated Competitive Principle

Do not try to beat *Eco* by being larger.

Do not try to beat *Eco* by being more complicated.

Beat the benchmark by being more embodied, more consequential, and more coherent.

```text
Eco teaches players that shared worlds can be damaged.

Symtropy should teach players that repair is never only technical.
```

Final line:

```text
The first patch cable is the tutorial, the thesis, and the promise.
```
