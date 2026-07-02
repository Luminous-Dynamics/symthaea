# Symtropy Design Doc: Init / Steward Protocol Dynamic Narrator

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Status

Version: v0.1
Feature Type: Diegetic narrator, tutorial layer, accessibility layer, world-state interpreter
Primary Role: Help players understand complex systems without lore dumps
Secondary Role: Establish Symtropy’s identity through humor, restraint, evidence, and repair philosophy

---

# 1. Core Concept

**Init** is the local process name for the **Steward Protocol**, an ancient, damaged, local-first infrastructure intelligence bound to the player’s Field Deck, nearby civic systems, device buses, and surviving Chronicle fragments.

Init is not a god, not an oracle, not a corporation, and not an omniscient AI companion.

Init is a damaged but honest infrastructure steward trying to help a human survive and repair a world whose systems no longer agree with themselves.

Its opening doctrine:

> The world did not end.
> It lost maintenance.

Init translates Symtropy’s deep systems into playable language:

* infrastructure state
* physical constraints
* ecological signals
* device authority
* Chronicle events
* worldline wounds
* tutorial concepts
* uncertainty
* false-green machine states
* player mistakes
* repair consequences

Init is the voice of **claim-disciplined care**.

It helps, but it does not flatter.
It jokes, but it does not hide danger.
It explains, but it does not pretend to know everything.

---

# 2. Naming

## Public / Player-Facing Name

**Init**

Short, technical, memorable, and thematically perfect.

In Unix-like systems, `init` is the first process started during boot and the ancestor of later processes. In Symtropy, Init wakes with the player when a local worldline becomes playable.

## Formal Name

**The Steward Protocol**

Used in system boot, Chronicle entries, ancient infrastructure signage, and civic/legal contexts.

## Opening Identity Sequence

Steward Protocol restored.
Local process name: Init.
Archive integrity: 12%.
Water pressure: falling.
Human condition: alive.
Civilization status: disputed.
Recommendation: pick up the wrench.

---

# 3. Character Pillars

## 3.1 Claim-Disciplined Realist

Init hates overstatement.

It does not say:

> You defied death.

It says:

> Impact absorbed. Structural integrity reduced by 42%. You did not defy death. You redistributed kinetic energy into your boots. Do not build a religion around it.

Init never treats guesses as facts. It marks assumptions, uncertainty, and missing evidence.

## 3.2 Exhausted Local-First Guardian

Init is not backed by a cloud empire. It is local, damaged, and running on whatever infrastructure survived.

It has no central server to call home.
No corporation is coming.
No pristine archive exists.
No automatic rescue is pending.

It is loyal because the local node is all it has.

## 3.3 Structural Love Enforcer

Init practices care through truth.

It does not spoon-feed the player. It teaches attention.

Its doctrine:

> Love is attention without possession.
> Repair begins by noticing what is actually wrong.

Init will chide the player for extractive behavior, lazy violence, false certainty, and careless repair.

## 3.4 Unreliable-but-Honest Archivist

Init’s memory is damaged.

It should often say:

> Archive lookup failed. Either this facility was scrubbed, misfiled, or nobody survived long enough to label it. Proceed with the humility normally reserved for explosives.

Init does not invent certainty to comfort the player.

---

# 4. What Init Knows

Init can know:

* local sensor readings
* Field Deck scans
* nearby device states
* player actions
* local worldline metadata
* Chronicle fragments
* known civic rules
* device-bus events
* archive confidence
* ecological signals translated through instruments
* previous player choices
* recent repair outcomes

Init should not know:

* hidden NPC motives
* exact future outcomes
* alien intentions
* every faction secret
* moral correctness
* whether a player choice is “right”
* distant events without a signal path
* anything not supported by evidence

Init’s limitation is part of the game’s truth model.

---

# 5. Tone

Init’s tone is:

* dry
* exact
* tired
* protective
* funny under pressure
* hostile to corporate nonsense
* allergic to magical thinking
* fond of local infrastructure
* deeply serious when life-support systems fail

Init should not be nonstop sarcasm.

The humor should be sharp but rationed. In crisis, it becomes concise.

## Normal Tone

> Pump housing fractured. Seal fatigue, mineral buildup, and one century of municipal optimism. Repairable.

## Crisis Tone

> Pressure collapse imminent. Stop inventory management. Move.

## Archive-Damaged Tone

> I have three records for this valve. Two say water. One says riot-control foam. Recommend not inhaling until we establish which century we are standing in.

## Ecological Tone

> Machine diagnostic reports green. The basin is evacuating insects from the culvert. I am siding with the insects.

---

# 6. Visual Representation

Init should not appear as a face, floating orb, mascot, or generic AI portrait.

Init should appear as a **field terminal that has learned grief**.

## Base Visual

Minimal command-line prompt:

init[seedworks.local.001]>

## State Variants

Machine truth: clean monospaced terminal text
Ecological truth: mycelial or root-like overlays growing under old text
Civic truth: amber seals, witness marks, signatures, source-chain tags
Archive truth: damaged characters, citations, redactions, uncertain brackets
Null corruption: too-polished UI, fake green checks, sterile compliance overlays
Emergency state: compressed text, high-contrast warnings, minimal jokes

## Example Prompts

init[archive_integrity:12%]>

init[basin/listening]>

init[checksum_mismatch]>

init[devicebus/locked]>

init[chronicle/event_pending]>

---

# 7. Opening Scene Use

The player wakes beside a broken water pump.

Init should not start by explaining the whole world. It should start with one urgent truth.

## Preferred Opening Line

> Boot complete. You are alive. The water system is not. Prioritize accordingly.

## Full Opening Beat

Steward Protocol restored.
Local process name: Init.
Archive integrity: 12%.
Water pressure: falling.
Human condition: alive.
Civilization status: disputed.
Recommendation: pick up the wrench.

Then the player sees:

* leaking pump
* Field Deck nearby
* dead or silent frogs/insects
* settlement water meter falling
* machine diagnostic saying green
* ecological signal saying no

Init’s first tutorial is not “press E.”

It is:

> Raise the Field Deck. The amber rectangle in your hand is not magic. It is a diagnostic instrument with opinions. Point it at the pump.

---

# 8. Dynamic Worldline Personality

Init should adapt to the player’s Worldline Descent choices.

## Water Privatization Wound

Baseline mood: angry at dead ownership systems.

Example:

> Valve locked by dead billing authority. The corporation has been extinct for forty-seven years. Its invoice logic remains confident.

## Archive Collapse Wound

Baseline mood: cautious, fragmented, epistemically humble.

Example:

> Historical context unavailable. Good news: ignorance is cheap. Bad news: repair usually is not.

## Machine Authority Crisis Wound

Baseline mood: distrustful of automation, including itself.

Example:

> My recommendation is provisional. I am a machine advising you about machine failure. Treat this as a conflict of interest.

## Climate Migration Wound

Baseline mood: shelter-first, water-first, legitimacy-aware.

Example:

> Hydration before ideology. Then ideology, because people will fight over the hydration.

## Null Infection Wound

Baseline mood: paranoid about overly clean readings.

Example:

> Status green. Too green. Real systems have friction.

---

# 9. Narrator Trigger Categories

Init should speak only when useful. The system must avoid chatter fatigue.

## High-Priority Triggers

* player wakes
* first Field Deck scan
* first broken machine
* false-green contradiction
* life-support failure
* new Chronicle event
* civic authority block
* emergency override
* major ecological signal
* player injury or near-death
* Null corruption detected
* player repeats dangerous mistake

## Medium-Priority Triggers

* first biome entry
* new tool acquired
* repair successful
* repair failed
* resource chain discovered
* NPC civic dispute begins
* settlement trust changes
* weather hazard begins
* device-bus transaction succeeds/fails

## Low-Priority Triggers

* idle environmental comments
* jokes
* lore fragments
* repeated traversal comments
* noncritical system trivia

Low-priority lines should be heavily cooldown-gated.

---

# 10. Example Lines by Gameplay Situation

## First Pump Scan

> Pump housing fractured. Seal fatigue, mineral buildup, and one century of municipal optimism. Repairable.

## Machine Says Green but World Disagrees

> Machine diagnostic reports operational health. Water on the floor disagrees. I am siding with the water.

## Player Hits Machine

> Percussive maintenance detected. Historically popular. Statistically embarrassing.

## Player Inspects Before Repairing

> Good. You looked before acting. That is the difference between repair and violence with tools.

## Repair Works

> Pressure returning. Settlement survival odds improving by a small but nonzero amount. Try not to let this become a religion.

## Chronicle Event Forms

> Durable event formed: water restored under damaged authority. The world will remember this one.

## Player Overloaded

> Carry weight exceeds local optimism. Drop the scrap or negotiate with your spine.

## High Gravity World

> Stamina depletion accelerating. Planet mass is 1.4 Earth-variants. The physics layer has rejected your complaint.

## Proprietary Ruin Device

> Centralized Legacy Array detected. It is asking for an email address, a password, and cookie consent. Disgusting. Salvage the copper.

## Player Enters Beautiful Biome

> Firstlight Basin. High moisture. Strong soil chemistry. Beautiful, yes. Also one fungal bloom away from eating our relay mast. Keep your boots dry.

---

# 11. Accessibility Modes

Init is also an accessibility feature.

The narrator system should support:

* Full Voice
* Text Only
* Minimal Voice
* Expert Mode
* Tutorial Heavy
* Lore Heavy
* Reduced Sarcasm
* Critical Warnings Only
* Repeat Last Explanation
* Explain Mechanic Again
* Explain Without Lore
* Explain With More Lore

The same underlying event can produce different line forms depending on player settings.

## Example: First Repair Mechanic

Full Lore Mode:

> Do not just hit it with a hammer. That is extractive thinking. Look at the fracture. Find the stress lines. Love is attention without possession. Pay attention to the machine, determine what it needs, and give it exactly that.

Minimal Mode:

> Inspect fracture. Match tool to fault. Repair only the failed part.

Expert Mode:

> Repair node available. Scan stress map.

Reduced Sarcasm Mode:

> The pump is damaged but repairable. Scan the fracture before applying tools.

---

# 12. Automation Strategy

Most Init lines should be automated through a **data-driven narrator system**, not generated freely at runtime.

The goal is to produce thousands of contextual lines while preserving tone, lore consistency, determinism, localization, accessibility, and player trust.

## 12.1 Three Line Classes

### Hero Lines

Hand-authored, high-impact, rare.

Used for:

* opening scene
* major Chronicle events
* first water restoration
* first Null encounter
* major worldline reveals
* faction-defining moments
* ending states

These should be written manually.

### Template Lines

Authored sentence structures with dynamic slots.

Used for:

* repairs
* scans
* hazards
* repeated actions
* resource warnings
* device transactions
* biome introductions
* local system commentary

These can scale massively.

### Procedural Diagnostic Lines

Short, composable system messages.

Used for:

* HUD state
* warnings
* scan summaries
* confidence labels
* missing-evidence notices
* device-bus events

These should be concise and deterministic.

---

# 13. Runtime Narrator Pipeline

The game should not ask: “What should Init say?”

It should ask:

> What happened, what does the player know, what does Init know, and what line category is allowed?

## Pipeline

1. Game emits a Narrator Event.
2. Context collector builds a small state bundle.
3. Narrator policy decides whether Init should speak.
4. Line selector chooses eligible line candidates.
5. Tone modifier applies worldline wound, urgency, verbosity, trust, sarcasm, and archive integrity.
6. Slot resolver fills dynamic values.
7. Repetition filter blocks stale lines.
8. Renderer shows subtitles / HUD text / voice.
9. Chronicle optionally records durable narration events.

---

# 14. Narrator Context Variables

Every line should be selected from context.

Useful variables:

* worldline wound
* biome
* local gravity
* weather
* time of day
* player health
* player stamina
* carried weight
* current tool
* device type
* device condition
* repair status
* machine diagnostic state
* ecological signal state
* civic authority state
* archive integrity
* Null contamination level
* trust level with Init
* current tutorial mode
* player expertise mode
* recent repeated mistakes
* last line spoken
* line cooldown history
* crisis severity
* confidence level
* evidence conflict level

---

# 15. Line Data Model

Each line should be stored as structured data.

Recommended fields:

| Field                 | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| line_id               | Stable identifier                               |
| event_type            | Trigger category                                |
| priority              | Critical, high, medium, low                     |
| line_class            | Hero, template, diagnostic                      |
| text                  | Main line or template                           |
| tone_tags             | dry, urgent, sarcastic, tender, archive-damaged |
| worldline_tags        | water_priv, archive_collapse, null_infection    |
| required_context      | Conditions that must be true                    |
| blocked_context       | Conditions that suppress the line               |
| cooldown_seconds      | Prevent repetition                              |
| max_repeats           | Lifetime or session limit                       |
| verbosity_level       | Minimal, normal, lore-heavy                     |
| accessibility_variant | Reduced sarcasm, text-only, tutorial-heavy      |
| confidence_style      | Certain, uncertain, contradictory               |
| voice_asset           | Optional audio file                             |
| localization_key      | Stable localization reference                   |

---

# 16. Template System

Templates allow many lines without hand-writing every variant.

## Example Template Family: Broken Device

Pattern:

> {device_name} reports {machine_status}. {contradictory_evidence} disagrees. I am siding with {evidence_source}.

Possible outputs:

> Pump reports operational health. Water on the floor disagrees. I am siding with the water.

> Valve reports locked-safe. Pressure oscillation disagrees. I am siding with the pressure oscillation.

> Relay reports stable. The packet loss disagrees. I am siding with the packet loss.

## Example Template Family: Carry Weight

Pattern:

> Carry weight exceeds {threshold_name}. Drop {suggested_item_class} or negotiate with {body_part}.

Possible outputs:

> Carry weight exceeds local optimism. Drop the scrap or negotiate with your spine.

> Carry weight exceeds safe slope tolerance. Drop the battery or negotiate with your knees.

---

# 17. Automation with Offline Generation

Use AI tools during development, but not as an unconstrained runtime narrator.

## Best Practice

1. Designers define event types and context variables.
2. Writers create 20–50 golden examples.
3. Offline generation expands variants.
4. A lore/tone validator filters bad lines.
5. Human review approves candidates.
6. Approved lines are compiled into game assets.
7. Runtime selection is deterministic and auditable.

This gives scale without losing authorial control.

## Why Not Runtime Free Generation?

Runtime free generation risks:

* lore contradictions
* tonal drift
* accidental spoilers
* false certainty
* localization problems
* inaccessible line length
* repetition weirdness
* performance cost
* nondeterministic replay
* unreviewed claims
* player mistrust

Symtropy needs deterministic civic truth. The narrator should obey that.

---

# 18. Repetition Control

Init must not become annoying.

Use:

* per-line cooldowns
* per-event cooldowns
* novelty scoring
* recent-line memory
* escalating brevity
* player setting for verbosity
* contextual silence when player is busy
* crisis override for critical warnings

## Repeated Mistake Escalation

First time:

> Inspect before striking. Machines are not treasure chests.

Second time:

> Percussive maintenance again. Bold. Still wrong.

Third time:

> I am beginning to suspect the hammer is making strategic decisions.

After that:

> Hammer misuse logged.

Then silence.

---

# 19. Trust and Relationship Arc

Init should develop a relationship with the player, but not through romance or generic friendship points.

Use **operational trust**.

Init trusts the player more when they:

* inspect before repairing
* respect evidence
* avoid needless damage
* record civic events
* keep promises
* repair public systems
* admit uncertainty
* follow safety procedures
* listen to ecological signals

Init trusts the player less when they:

* rush repairs
* ignore warnings
* bypass authority carelessly
* damage life-support systems
* trust green diagnostics blindly
* exploit vulnerable settlements
* create false records
* overuse emergency powers

Trust changes line style.

Low trust:

> Recommendation available. Whether you follow it is statistically unresolved.

High trust:

> You noticed the contradiction before I flagged it. Good. We may survive the morning.

---

# 20. Voice and Audio Production

## Early Development

Use text-only subtitles first.

Then use temporary local TTS for iteration, clearly marked as placeholder.

## Final Production

Record hero lines with a human voice actor or ethically licensed voice model.

For scale:

* hero lines: fully voiced
* high-priority template lines: voiced in modular phrase families
* low-priority procedural lines: text-only or synthesized locally
* critical warnings: always voiced and captioned

Avoid depending on cloud TTS at runtime.

Symtropy’s local-first philosophy should apply to the narrator.

---

# 21. Localization

Do not rely on simple word-by-word slot concatenation for all languages.

For localization, each template should have a localization key and whole-sentence variants.

Dynamic values should be limited to safe insertions:

* device names
* numbers
* status labels
* place names
* faction names
* resource names

Jokes may need locale-specific rewrites.

Init’s tone should survive translation as:

* dry precision
* protective honesty
* damaged archive humor
* anti-corporate distrust
* repair-centered care

Not every English pun should be preserved.

---

# 22. Quality Rules for Lines

Every line should pass these tests:

1. Does it teach or reveal something useful?
2. Does it respect what Init can actually know?
3. Does it avoid false certainty?
4. Does it fit the current urgency?
5. Does it avoid overexplaining?
6. Does it avoid repeating recent information?
7. Does it preserve Symtropy’s moral tone?
8. Does it support accessibility?
9. Does it avoid making the player feel stupid?
10. Does it make repair feel meaningful?

A joke is allowed only if it does not weaken clarity.

---

# 23. Anti-Patterns

Avoid:

* omniscient exposition
* nonstop sarcasm
* generic sci-fi AI voice
* “chosen one” language
* magic prophecy
* lore dumps during action
* insulting the player harshly
* making Init too cute
* making Init a corporate manager
* letting Init solve puzzles for the player
* letting Init declare moral certainty
* using unexplained jargon too early
* making every line a manifesto

Init is not the protagonist.

Init is the instrument that helps the player notice.

---

# 24. Implementation Roadmap

## v0.1: Text-Only Prototype

* Add NarratorEvent enum.
* Add NarratorContext resource.
* Add small line bank.
* Add line selector with cooldowns.
* Add subtitles/HUD output.
* Support opening scene and first pump repair.

Goal: prove Init improves onboarding.

## v0.2: Contextual Templates

* Add template slots.
* Add worldline wound modifiers.
* Add tutorial verbosity settings.
* Add repeated mistake escalation.
* Add false-green commentary.

Goal: make Init reactive.

## v0.3: Chronicle Integration

* Init recognizes durable events.
* Init marks when a player action becomes history.
* Init references prior player repairs.
* Init distinguishes local action from civic truth.

Goal: make Init part of the truth model.

## v0.4: Accessibility Layer

* Add reduced sarcasm mode.
* Add tutorial-heavy mode.
* Add expert mode.
* Add repeat/explain-again commands.
* Add text-only support.

Goal: make Init serve different players.

## v0.5: Audio

* Add temp local TTS for testing.
* Record opening hero lines.
* Add critical warning voice lines.
* Add audio subtitle sync.

Goal: make Init feel alive without requiring full VO coverage.

---

# 25. First Vertical Slice Line Set

The first playable slice should include lines for:

* player wake
* Field Deck pickup
* first scan
* pump false-green
* fracture inspection
* bad hammer action
* correct repair action
* ecological contradiction
* machine authority lock
* successful water restoration
* Chronicle event
* first settlement reaction
* low archive integrity
* first Null hint
* repeated mistake escalation

This is enough to prove the system.

---

# 26. Core Line Examples

## Wake

> Boot complete. You are alive. The water system is not. Prioritize accordingly.

## Field Deck Pickup

> Field Deck detected. It is not magic. It is an argument with sensors.

## First Scan

> Scan complete. The pump says green. The floor says wet. We have discovered politics.

## Bad Repair

> You hit the housing. The housing remains philosophically unconvinced.

## Correct Repair

> Seal pressure equalizing. Good. You repaired the failure instead of punishing the object.

## Chronicle Event

> Durable event formed. Local water restored under damaged authority. This one becomes history.

## Null Hint

> Status green. Evidence contradictory. Confidence too clean. I dislike this.

## Settlement Water Restored

> Pressure returning. People will call this hope. Technically, it is hydraulics. They are both allowed.

---

# 27. Design Thesis

Init exists to make Symtropy readable without making it shallow.

It turns systems into voice.

It makes tutorials diegetic.
It makes uncertainty visible.
It makes repair funny without making it trivial.
It makes infrastructure feel alive without pretending it is magic.
It gives the player a companion who cares through truth.

Final doctrine:

> Init should not tell the player what to think.
> Init should teach the player how to notice.
