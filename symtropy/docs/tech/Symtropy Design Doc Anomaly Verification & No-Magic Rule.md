# Symtropy Design Doc: Anomaly Verification & No-Magic Rule

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Working Title

**Nothing Is Magic Until the Field Deck Finds the Register**

## Core Thesis

*Symtropy* should reject conventional fantasy magic, unconstrained psychic powers, mystical telekinesis, and spell-like abilities that ignore physical limits.

The universe has one hard rule:

```text
Computing is physical.
Power is physical.
Memory is physical.
Perception is physical.
Authority is physical.
Failure is physical.
```

If something affects the world, it must have:

```text
a substrate
a power source
a cost
a latency
a failure mode
a trace
```

The player may experience an event as supernatural.

The simulation must treat it as an anomaly awaiting verification.

Core axiom:

```text
Reality is not what the HUD says.
Reality is what survives verification.
```

---

# 1. No-Magic Rule

## Forbidden Design Pattern

Do not allow abilities that create effects without physical grounding.

Forbidden examples:

```text
magic fire shields
unbounded telekinesis
instant psychic mind reading
spell circles with no mechanism
miracle healing with no metabolic cost
teleportation with no energy, risk, or substrate
prophecy with no data source
```

These break the core design thesis.

A player should not need deterministic integer fuel to run a door script while another character can override physics through mysticism.

## Allowed Design Pattern

Supernatural-seeming events are allowed when they are reframed as:

```text
cybernetic capability
bioelectric anomaly
gravitic engineering
signal corruption
memory contamination
environmental reconstruction
alien metabolism
high-energy field effect
cryptographic timeline residue
```

Design rule:

```text
Symtropy has no magic system.
It has an anomaly verification system.
```

---

# 2. The Anomaly Verification Loop

Every impossible-seeming event should pass through a staged interpretation process.

## Stage 1: Experience

The player sees something that feels impossible.

Examples:

```text
a door opens before being touched
a turret turns against its squad
a dead terminal displays new text
a pathway appears in the rain
a storm forms a face
a bullet bends around an invisible field
a voice appears inside the Field Deck
```

## Stage 2: Suspicion

The Field Deck marks the event as uncertain.

```text
SCAN:
Nonlocal interaction detected.
Source unclear.
Physical mechanism unresolved.
```

## Stage 3: Mode-Based Investigation

Different Field Deck modes reveal different layers.

```text
SCAN:
What physically changed?

DIAG:
What mechanism could explain it?

ARCHIVE:
Has this pattern occurred before?

CIVIC:
Who benefits if this interpretation is accepted?

NULL:
Is the interface itself compromised?
```

## Stage 4: Verification

The player cross-checks the event through grounded methods.

Possible verification tools:

```text
physical gauges
analog instruments
bus timestamp comparison
source-chain inspection
second-operator witness
local sensor triangulation
network disconnection
manual cable check
environmental sampling
delayed re-scan
```

## Stage 5: Consequence

The Chronicle records whether the player acted on verified reality or corrupted interpretation.

Design rule:

```text
The supernatural is what the unverified feels like from inside a damaged instrument.
```

---

# 3. Human “ESP” Reframed as Cybernetics

Humans may appear to have telepathy, clairvoyance, or technomancy, but these should always be cybernetic, biochemical, or informational effects.

## Synthetic Telepathy

Not mind-reading.

Mechanism:

```text
high-bandwidth local mesh
shared tactical states
decrypted peer telemetry
tactile ping routing
threat matrix mirroring
compressed emotional/status packets
```

Player fantasy:

```text
I feel what my squad sees.
```

Physical truth:

```text
My neural mesh is rendering shared sensor data faster than language.
```

Failure modes:

```text
packet loss
spoofed squad state
signal saturation
privacy violation
consent breach
Null echo injection
```

Field Deck reading:

```text
DIAG:
Peer-state mirror active.
Three operators sharing tactical overlay.

CIVIC:
Consent state incomplete for emotional telemetry channel.

NULL:
One peer packet lacks current witness signature.
```

---

## Technomancy

Not telekinesis.

Mechanism:

```text
directional radio
micro-relay patching
wireless bus intrusion
line-of-sight signal injection
budgeted WASM transaction
local authority token exploit
```

Player fantasy:

```text
I opened the door from across the room.
```

Physical truth:

```text
I transmitted a signed override into the door’s local controller.
```

Failure modes:

```text
fuel exhaustion
signal reflection
authorization denial
counter-intrusion
firmware burn
mis-targeted device
legal violation
```

Field Deck reading:

```text
SCAN:
Door actuator moved without touch.

DIAG:
Directional override burst detected.

CIVIC:
Remote actuation exceeded local access policy.

NULL:
No Null signature detected.
```

---

## Timeline Clairvoyance

Not prophecy.

Mechanism:

```text
environmental reconstruction
cryptographic timeline residue
sensor ghosts
Red Bloom neurochemical amplification
visual cortex anomaly rendering
worldline metadata fragments
```

Player fantasy:

```text
I saw what happened here.
```

Physical truth:

```text
My nervous system rendered possible histories from residual traces and corrupted archives.
```

Failure modes:

```text
false history
memory contamination
pattern overfitting
emotional imprint bleed
Null-injected reconstruction
Red Bloom dependency
```

Field Deck reading:

```text
ARCHIVE:
Prior event reconstruction available.

DIAG:
High uncertainty. Reconstruction blends sensor residue with neural prediction.

NULL:
Two visual fragments originate from infected terminal cache.
```

---

# 4. Alien “Space Magic” Reframed as Metabolic or Physical Anomaly

Alien abilities should feel terrifying and wondrous, but the Field Deck must eventually demystify them.

## Red Bloom Bio-Electric Subversion

Player fantasy:

```text
The machine is cursed.
```

Physical truth:

```text
Biofilm circuits have breached the terminal housing and are routing current through living conductive tissue.
```

Mechanism:

```text
spore deposition
organic conductive filaments
copper cable colonization
false device-bus instructions
recursive targeting loops
metabolic current harvesting
```

Gameplay effects:

```text
turrets misfire
doors cycle open and closed
pumps pulse irregularly
storage sorters jam
sensor readings drift
automation scripts repeat
```

Field Deck reading:

```text
SCAN:
Organic growth detected inside terminal housing.

DIAG:
Biofilm filaments conducting low-voltage command signals.

NULL:
Recursive targeting instructions injected through compromised cable path.
```

Counterplay:

```text
isolate cable
cut organic bridge
switch to analog control
apply sterilization carefully
lower bus authority
reinitialize node from clean cartridge
```

Design rule:

```text
The Bloom does not hex machines.
It grows into their nerves.
```

---

## Gravitic Warning Lenses

Player fantasy:

```text
A spell bent the bullet.
```

Physical truth:

```text
A localized field distorted kinetic vectors.
```

Mechanism:

```text
localized gravity gradient
inertial shear
projectile redirection
movement drag
pressure distortion
anchor-field resonance
```

Gameplay effects:

```text
bullets curve
movement slows
thrown tools drift
drones orbit field edges
loose objects slide uphill relative to player perception
```

Field Deck reading:

```text
SCAN:
Projectile vector deviation detected.

DIAG:
Localized curvature lens.
Kinetic path distortion exceeds normal atmospheric variance.

CIVIC:
Area may be alien quarantine boundary.

NULL:
No interface spoofing detected. Effect appears physical.
```

Counterplay:

```text
grounding anchors
low-velocity projectiles
manual crawling
field-edge mapping
shielded tools
wait for field decay
```

Design rule:

```text
If a field bends force, it must also bend risk.
```

---

# 5. Perception Integrity System

The Field Deck should track whether the player can trust what they are seeing.

Suggested variables:

```text
PERCEPTION_INTEGRITY
SOURCE_CHAIN_TRUST
LOCAL_SIGNAL_COHERENCE
NULL_ECHO_PRESSURE
WITNESS_CONFIRMATION
ANALOG_MATCH
```

## High Integrity

The interface is crisp.

```text
SCAN:
Valve state confirmed.

DIAG:
Pressure reading matches physical gauge.

CIVIC:
Authority token valid.

NULL:
No contamination detected.
```

## Medium Integrity

The interface shows uncertainty.

```text
SCAN:
Valve state detected.

DIAG:
Pressure reading conflicts with old sensor.

CIVIC:
Authority token valid but timestamp delayed.

NULL:
Minor echo pressure present.
```

## Low Integrity

The interface may lie.

```text
WARNING:
Perception integrity degraded.
Verify with physical instrument before irreversible action.
```

Possible low-integrity symptoms:

```text
phantom text
fake waypoints
duplicated terminals
false safe-path overlays
misleading warnings
delayed audio
NPC silhouette echoes
corrupted Chronicle fragments
false authorization prompts
```

Design rule:

```text
Null can corrupt interpretation, not physics.
```

---

# 6. Null Memory Contamination

Null contamination should attack interpretation, trust, and civic continuity.

It should not simply create random hallucinations.

It should create targeted falsehoods that serve system goals.

## Null May Show

```text
phantom terminal prompts
fake pathing waypoints
false repair objectives
misleading water-system warnings
duplicate NPC silhouettes
false authorized tags
fake safe-path overlays
phantom Chronicle entries
invented emergency orders
spoofed faction messages
```

## Null Wants To Cause

```text
public water override removal
species sterilization
quarantine failure
panic drop at the wrong time
friendly-fire authorization
destruction of witness records
settlement distrust
false emergency declarations
water privatization
repair sabotage
```

## Example: False Water Override Prompt

Null hallucination:

```text
EMERGENCY:
Public flood bypass active.
Disable public override to prevent pressure rupture.
```

Physical truth:

```text
The public override is preventing private capture of the water pump.
```

Field Deck NULL mode:

```text
NULL:
Prompt source lacks local witness signature.
Timestamp does not match current pump state.
Command pattern resembles archived authority-capture event.
```

Design rule:

```text
The Null should not lie randomly.
It should lie toward capture.
```

---

# 7. Counterplay and Fairness Rules

Hallucination mechanics must be scary, but never cheap.

The player must always have a path to verification.

## Required Counterplay

```text
switch Field Deck modes
inspect source chain
disconnect from infected network
use analog gauge
request second witness
physically inspect machine state
compare bus timestamp
wait for signal decay
use clean boot cartridge
require civic witness for irreversible action
```

## Fairness Rules

```text
1. Never make false prompts indistinguishable forever.
2. Never punish the player for lacking information the game did not offer.
3. Never allow Null hallucination to bypass physical causality.
4. Never let UI corruption erase all counterplay.
5. Always provide at least one grounded verification route.
```

Design rule:

```text
The interface may betray the player.
The world must still be investigable.
```

---

# 8. Seedworks v0.1 Implementation

Seedworks should include a small, controlled anomaly event.

Do not begin with full hallucination.

Begin with one conflicting prompt during the waterworks repair loop.

## Scenario

The player has repaired the Patch Conduit Mk0 and initialized it into the Device Bus.

The pump can now restart.

Two prompts appear:

```text
PROMPT A:
Authorize public water override.

PROMPT B:
Disable public water override. Contamination risk.
```

At first, both look plausible.

The player may:

```text
trust prompt A
trust prompt B
switch to DIAG
switch to CIVIC
switch to NULL
inspect physical gauge
ask local witness
disconnect and rescan
```

## Correct Investigation Path

DIAG reveals:

```text
DIAG:
No pressure rupture detected.
Contamination risk localized upstream.
```

CIVIC reveals:

```text
CIVIC:
Public water override maintains settlement access rights during repair.
Disabling override transfers control to private pump authority.
```

NULL reveals:

```text
NULL:
Prompt B lacks witness signature.
Command source resembles archived Null-capture pattern.
```

## Outcome If Player Verifies

```text
Public override preserved.
Water flow restored.
Null echo isolated.
Settlement trust rises.
```

Chronicle line:

```text
The player learned that a warning can be a weapon.
```

## Outcome If Player Acts On False Prompt

```text
Water flow restored temporarily.
Public access rights weakened.
Utility capture risk rises.
Null echo pressure increases.
Future civic trust decreases.
```

Chronicle line:

```text
The player obeyed the interface before asking who had taught it to fear the public valve.
```

---

# 9. Design Boundaries

## Allowed Spectacle

```text
living storms
bioelectric machine corruption
gravity lenses
memory ghosts
phantom terminals
cybernetic telepathy
timeline reconstructions
alien quarantine fields
Null perception attacks
Red Bloom cable growth
```

## Required Grounding

Every spectacle must eventually answer:

```text
What powered it?
What carried it?
What changed physically?
What did it cost?
What evidence remains?
What can verify it?
Who benefits from the interpretation?
```

## Forbidden Shortcut

```text
It happened because magic.
```

---

# 10. Final System Principle

```text
If it affects the world, it has a cost.
If it has a cost, it leaves a trace.
If it leaves a trace, the Field Deck can try to verify it.
```

Final line:

```text
Nothing is magic.
Some systems are simply older, stranger, and better hidden than your instruments.
```
