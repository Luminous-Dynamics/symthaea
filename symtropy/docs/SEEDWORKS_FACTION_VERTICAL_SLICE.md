# SEEDWORKS_FACTION_VERTICAL_SLICE.md

# Seedworks Faction Vertical Slice

## Purpose

Seedworks should prove the entire procedural faction system in miniature.

The first version does not need all archetypes, all timelines, or all planets.

It needs one playable region where players can watch a society respond to pressure, change posture, debate identity, and possibly evolve.

## Core Slice

The Seedworks vertical slice should include four operational archetypes:

1. Mutualist Assembly
2. Industrial Compact
3. Null Ecology
4. Ghost Civilization

Together, these form the first faction square.

## The Faction Square

## Mutualist Assembly

Represents the society players are trying to build.

Theme:

```text
Can trust survive scarcity?
```

Gameplay role:

* starter settlement
* public works
* shared water
* mutual aid
* repair culture
* civic voting
* NPC trust

Strengths:

* cooperation
* legitimacy
* repair speed
* NPC loyalty
* resilience after disaster

Weaknesses:

* slow decisions
* resource conflict
* sabotage vulnerability
* pressure toward militarization

## Industrial Compact

Represents the temptation of output at any cost.

Theme:

```text
Can production serve life instead of consuming it?
```

Gameplay role:

* rival workshop
* production booster
* tool supplier
* machine parts
* factory shortcuts
* pollution pressure

Strengths:

* faster fabrication
* better machinery
* stronger logistics
* emergency production

Weaknesses:

* pollution
* labor pressure
* hoarding
* legitimacy debt
* risk of automation runaway

## Null Ecology

Represents optimization without wisdom.

Theme:

```text
What happens when systems run after meaning dies?
```

Gameplay role:

* PvE enemy
* rogue factory
* drone pressure
* infrastructure infection
* horde events
* facility boss

Strengths:

* relentless pressure
* scalable combat
* machine repair
* territorial infection

Weaknesses:

* predictable needs
* hackable signals
* dependence on cores
* no civic legitimacy

## Ghost Civilization

Represents the warning from the past.

Theme:

```text
What did the old world fail to understand?
```

Gameplay role:

* ruins
* archive logs
* old defense systems
* automated laws
* dead settlement records
* historical mystery

Strengths:

* hidden knowledge
* old infrastructure
* powerful relics
* worldline clues

Weaknesses:

* broken context
* dead laws
* corrupted archives
* dormant Null infection

## First Region: Firstlight Basin

Firstlight Basin should include:

* Seedworks Outpost
* The Scrapfield
* Old Waterworks
* Rogue Factory
* Solar Ridge
* Wreckfall Plain
* Ghost Civic Center

## Starting Conditions

The player begins in Seedworks Outpost.

Initial state:

```text
power:       unstable
water:       critical
food:        low
repair:      poor
trust:       fragile
legitimacy:  provisional
safety:      weak
entropy:     rising
signal:      intermittent
```

Crisis:

The water pump is failing during a storm.

The settlement has one battery reserve.

NPCs disagree about its use.

Options:

* power the medbay
* power the fabricator
* power perimeter defense
* stabilize the water pump

This creates the first civic decision.

## First 30-Minute Flow

### Step 1 — Repair

Player restores a local junction.

Learns:

* power matters
* infrastructure is physical
* repair changes settlement state

### Step 2 — Salvage

Player retrieves parts from a nearby wreck.

Learns:

* ruins are resources
* danger exists outside settlement
* salvage supports fabrication

### Step 3 — Fabricate

Player repairs or prints a pump component.

Learns:

* machines convert salvage into public goods
* fabricators are civilizational organs

### Step 4 — Escort

Player transports the component to Old Waterworks.

Learns:

* logistics matter
* roads and safety matter
* goods exist in the world

### Step 5 — Fight

Null drones attack the waterworks.

Learns:

* combat protects infrastructure
* enemies target systems, not only players

### Step 6 — Restore

Water flows again.

Learns:

* infrastructure recovery changes NPC behavior
* settlement state improves

### Step 7 — Vote

The settlement holds its first public vote.

Learns:

* governance is physical
* choices have consequences
* society has values under pressure

## First Public Vote

The first vote should be simple but meaningful.

Question:

```text
What should Seedworks prioritize after restoring water?
```

Options:

## Public Repair

Reinforce water, housing, and tools.

Effect:

* Mutualist weight increases
* trust rises
* defense remains weak
* future repair cheaper

## Factory Overdrive

Use water recovery to expand fabrication.

Effect:

* Industrial weight increases
* production rises
* pollution risk begins
* worker stress rises

## Perimeter Defense

Build walls, lights, patrol routes, and guard posts.

Effect:

* Security drift begins
* safety rises
* trust may split
* future raids easier to survive

## Archive Recovery

Investigate the Ghost Civic Center.

Effect:

* Archive drift begins
* knowledge rises
* old risks awaken
* Ghost Civilization content expands

This one vote can seed multiple faction futures.

## Emergency Postures in Seedworks

Seedworks should implement a small number of emergency postures.

## Water Emergency

Trigger:

```text
water < critical threshold
```

Effects:

* rationing starts
* NPC stress rises
* water missions appear
* hoarding risk increases
* civic proposals generated

## Raid Alert

Trigger:

```text
Null threat near settlement
```

Effects:

* NPCs shelter
* guards move to gates
* repair work slows
* defense missions appear
* Security archetype pressure rises

## Factory Overdrive

Trigger:

```text
production deficit + player/NPC policy choice
```

Effects:

* fabrication speed rises
* power consumption rises
* pollution rises
* worker stress rises
* Industrial archetype pressure rises

## Archive Lockdown

Trigger:

```text
Ghost system activated or dangerous old logs discovered
```

Effects:

* old doors seal
* automated law systems activate
* archive missions appear
* knowledge rises
* risk of old defense systems rises

## Initial NPC Roles

Seedworks should include 5–8 named NPCs.

## The Engineer

Represents repair and fabrication.

Pressure response:

* favors factory overdrive during material scarcity
* may oppose reckless militarization
* may support machine citizenship if robots prove loyal

## The Medic

Represents care and public health.

Pressure response:

* favors water, food, sanitation, and medbay
* opposes forced labor and dangerous overdrive
* may become a moral critic of the faction

## The Convoy Lead

Represents logistics and risk.

Pressure response:

* favors roads, defense, and practical compromise
* may support Security drift after convoy losses
* may join Freebelt-like splinters later

## The Archivist

Represents memory and law.

Pressure response:

* favors archive recovery and historical truth
* records legitimacy debt
* warns against repeating Ghost Civilization failures

## The Service Robot

Represents machine personhood and trust.

Pressure response:

* obeys at first
* develops memory through player treatment
* can become central to a Machine Stewardship path

## The Young Technician

Represents the future generation.

Pressure response:

* reacts emotionally to player choices
* may become a symbol in Chronicle events
* gives weight to education, safety, and hope

## The Industrial Liaison

Represents production temptation.

Pressure response:

* offers faster fabrication at social/ecological cost
* may found an Industrial splinter
* creates moral tradeoffs without being a villain

## The First Antagonist Signal

Represents the Null Ecology.

Pressure response:

* adapts to player defenses
* shifts from drones to infrastructure infection
* can become Factory Bloom or Signal Rot depending on events

## Seedworks Faction Data Objects

The first implementation should track:

## FactionState

Current settlement state.

```text
resources
population
NPC roles
infrastructure
territory
relations
trust
legitimacy
stress
repair
safety
entropy
```

## ArchetypeVector

Current identity weights.

```text
Mutualist
Industrial
Security
Archive
Machine Stewardship
Null
Ghost
```

## PressureVector

Current crisis pressure.

```text
water_scarcity
raid_threat
production_deficit
pollution
trust_collapse
ghost_activation
machine_infection
```

## ValueSystem

What the faction claims to protect.

```text
sacred_values
taboos
policy preferences
legitimacy rules
```

## ChronicleMemory

What the faction remembers.

```text
votes
deaths
rescues
betrayals
failures
public works
raids
laws
archive discoveries
```

## First Evolution Examples

## Path A — The Commons Holds

Player choices:

* restore water publicly
* protect medbay
* avoid forced labor
* share tools
* defend convoys
* hold fair votes

Outcome:

Mutualist weight stays dominant.

Chronicle entry:

```text
The Water Vote became the founding memory of Seedworks. Scarcity did not break the commons.
```

## Path B — The Fortress Emerges

Player choices:

* prioritize perimeter defense
* accept checkpoints
* allow emergency patrols
* repeatedly survive raids
* delay return of emergency powers

Outcome:

Security weight rises.

Possible transformation:

```text
Seedworks Assembly → Firstlight Protectorate
```

Chronicle entry:

```text
After the third raid, the gates became more important than the forum.
```

## Path C — The Factory Takes Over

Player choices:

* prioritize output
* accept pollution
* overwork NPCs
* privatize fabrication access
* trade public trust for machine growth

Outcome:

Industrial weight rises.

Possible transformation:

```text
Seedworks Assembly → Basin Production Compact
```

Chronicle entry:

```text
The settlement survived by becoming a machine that made machines.
```

## Path D — The Archive Awakens

Player choices:

* investigate Ghost Civic Center
* preserve records
* expose old failures
* rebuild law from historical evidence
* accept dangerous truths

Outcome:

Archive weight rises.

Possible transformation:

```text
Seedworks Assembly → Firstlight Witness Council
```

Chronicle entry:

```text
The dead city taught the living how not to govern.
```

## Path E — The Machine Question

Player choices:

* protect the service robot
* recognize robot testimony
* repair machine memory
* reject disposable automation
* give robots civic status

Outcome:

Machine Stewardship weight rises.

Possible transformation:

```text
Seedworks Assembly → Kind Engine Stewardship
```

Chronicle entry:

```text
The first citizen who could not drink water still helped save the well.
```

## Path F — Collapse

Player failures:

* water remains broken
* raids destroy supplies
* trust collapses
* NPCs flee
* Null infection spreads
* public votes fail

Outcome:

Settlement fragments.

Possible results:

* Raider Swarm
* Refugee Ark
* Ghost Civilization
* Null Ecology
* worldline fork

Chronicle entry:

```text
Seedworks did not die in a single night. It became too many emergencies to remember itself.
```

## Success Criteria

The vertical slice succeeds if players can feel that:

* infrastructure matters
* combat protects civilization
* governance changes physical outcomes
* NPCs remember choices
* faction identity evolves from pressure
* ruins contain historical warnings
* enemies attack systems, not only bodies
* the same starting settlement can become different societies

## Final Principle

Seedworks should not try to show the whole galaxy.

It should show one society at the moment it becomes possible.

The player should leave the first version thinking:

```text
This place could become anything — and that is terrifying.
```
