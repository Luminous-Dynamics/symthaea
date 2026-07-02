# Symtropy Design Doc: Registered Infrastructure Adjudication

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Working Title

**When Two Factions Want the Same Pipe**

## Core Thesis

In *Symtropy*, infrastructure is never merely technical.

A pump, pipe, gate, bridge, turret, species-release pod, oxygen valve, water override, or settlement mainframe node is a physical object, a software endpoint, a civic claim, and a future liability.

Once infrastructure is registered to the Device Bus, it becomes visible to law.

Core rule:

```text
If a device can affect the settlement, the settlement can argue over it.
```

Registered infrastructure adjudication is the system that resolves competing claims over devices, repairs, permissions, obligations, and consequences.

The goal is not to simulate bureaucracy for its own sake.

The goal is to make repair politically meaningful.

---

# 1. Why This System Exists

Previous systems establish that:

```text
crafting creates registered devices
registered devices publish state
Device Bus nodes affect settlement life
repairs can be authorized or unauthorized
factions can disagree over infrastructure
Chronicle records durable consequence
```

This document answers the missing question:

```text
What happens after a repair creates a dispute?
```

Example:

```text
The player patches a water pipe.

The water moves again.

Watershed Commons praises the repair.

Continuance Office says the node lacks inspection.

Utility Sovereign claims the pipe is part of its licensed grid.

Quarantine Authority flags the salvaged ceramic seal as contaminated.

The settlement still needs water.
```

This is the design space.

---

# 2. Registered Infrastructure

A device becomes registered infrastructure when it has:

```text
physical presence
Device Bus path
operator history
power or flow relationship
authority status
risk profile
affected population
Chronicle eligibility
```

Example device path:

```text
/dev/sym/water/patch_conduit_alpha
```

Example registered node:

```json
{
  "node": "/dev/sym/water/patch_conduit_alpha",
  "class": "water_patch_conduit",
  "status": "active",
  "operator": "player",
  "seal_quality": "rough_emergency_seal",
  "authority_scope": "temporary_repair",
  "inspection_required": true,
  "affected_systems": ["public_water_access", "wetland_flow", "pump_main"],
  "claim_status": "contested"
}
```

Design rule:

```text
A device without a claim is machinery.
A device with a claim is infrastructure.
```

---

# 3. Claim Types

Factions do not all argue in the same language.

Each claim has a type.

## Technical Claim

A claim about safety, reliability, or performance.

Examples:

```text
seal is unstable
pressure exceeds rating
firmware is unverified
material is contaminated
node causes command chatter
```

## Civic Claim

A claim about legitimacy, law, access, or procedure.

Examples:

```text
repair was unauthorized
public access must be preserved
private control violates charter
emergency token expired
inspection deadline missed
```

## Ecological Claim

A claim about water, soil, species, habitat, or long-term restoration.

Examples:

```text
pump restart will drain wetland
pipe reroute harms willow roots
contaminated flow threatens downstream ecology
species-release pod affects alien biosphere
```

## Property Claim

A claim about ownership, license, debt, salvage, or jurisdiction.

Examples:

```text
Utility Sovereign owns the pump
salvaged part belongs to old contractor
blueprint license forbids modification
repair created unpaid liability
```

## Rights Floor Claim

A claim that invokes fundamental protections.

Examples:

```text
public water access cannot be privatized
uplifted worker consent required
biospheric agency uncertainty present
quarantine cannot override survival rights without review
```

## Emergency Claim

A claim that procedure must yield to immediate survival.

Examples:

```text
settlement will lose water in 2 hours
oxygen pressure falling
flood gate must open now
fire suppression unavailable
```

Design rule:

```text
Conflict becomes playable when claims are typed.
```

---

# 4. Claim Object Schema

Every formal dispute creates claim objects.

```json
{
  "claim_id": "claim.water.patch_alpha.utility_001",
  "node": "/dev/sym/water/patch_conduit_alpha",
  "claimant": "Utility Sovereign",
  "claim_type": "property",
  "priority": "medium",
  "evidence": [
    "old license record",
    "pump serial match",
    "blueprint firmware clause"
  ],
  "requested_action": "transfer_control",
  "risk_if_ignored": "legal_escalation",
  "rights_floor_conflict": false
}
```

Another claim:

```json
{
  "claim_id": "claim.water.patch_alpha.commons_001",
  "node": "/dev/sym/water/patch_conduit_alpha",
  "claimant": "Watershed Commons",
  "claim_type": "ecological",
  "priority": "high",
  "evidence": [
    "downstream flow reading",
    "wetland dependency map",
    "public water access metric"
  ],
  "requested_action": "maintain_public_override",
  "risk_if_ignored": "settlement_water_inequity",
  "rights_floor_conflict": true
}
```

Design rule:

```text
A dispute is evidence plus requested action.
```

---

# 5. Adjudication Loop

The adjudication loop has seven steps.

```text
1. Trigger dispute
2. Gather claims
3. Surface evidence
4. Choose procedural frame
5. Select resolution
6. Apply consequences
7. Record precedent
```

---

## Step 1: Trigger Dispute

A dispute may be triggered by:

```text
unauthorized repair
device failure
faction claim
expired emergency token
inspection failure
ecological harm
public access change
Null contamination
material provenance conflict
rights floor ambiguity
```

Example trigger:

```text
Player initializes Patch Conduit Mk0 without full civic authorization.
Water flow restored.
Node enters contested state.
```

Field Deck:

```text
CIVIC:
Registered infrastructure dispute detected.

Node:
/dev/sym/water/patch_conduit_alpha

Status:
ACTIVE / TEMPORARY / CONTESTED
```

---

## Step 2: Gather Claims

The system identifies interested parties.

Possible claimants:

```text
Watershed Commons
Utility Sovereign
Continuance Office
Quarantine Authority
Quiet Green
Settlement Council
Archive Witness
Uplift Collective
local residents
alien ecological witness
```

Field Deck:

```text
CIVIC:
Four claims pending.

1. Continuance Office: inspection required.
2. Watershed Commons: preserve public access.
3. Utility Sovereign: license conflict.
4. Quarantine Authority: material contamination review.
```

---

## Step 3: Surface Evidence

The player can inspect evidence through Field Deck modes.

## SCAN

Physical state.

```text
Seal quality: rough emergency.
Flow restored: partial.
Leak risk: medium.
```

## DIAG

Technical interpretation.

```text
Repair stable under low pressure.
High-pressure restart not recommended.
```

## ARCHIVE

Historical records.

```text
Pipe segment formerly operated under Utility Sovereign maintenance contract.
Contract status unresolved after settlement collapse.
```

## CIVIC

Claims and authority.

```text
Emergency repair permitted.
Permanent operation requires review.
Public water override active.
```

## NULL

Corruption check.

```text
No active Null spoofing detected.
One license record has timestamp mismatch.
```

Design rule:

```text
Adjudication should make evidence playable, not hidden in lore text.
```

---

## Step 4: Choose Procedural Frame

The player does not simply choose a faction.

The player chooses the procedure.

Procedural frames:

```text
Emergency Order
Technical Inspection
Commons Hearing
Rights Floor Review
Quarantine Hold
Archive Witness Review
Faction Arbitration
Settlement Vote
Direct Action
Temporary Injunction
```

Each frame changes what evidence matters.

## Emergency Order

Fast, survival-focused.

```text
prioritizes immediate function
weak legitimacy
may create later liability
```

## Technical Inspection

Engineering-focused.

```text
prioritizes safety
may ignore social access questions
slower than emergency order
```

## Commons Hearing

Public legitimacy-focused.

```text
brings residents and ecological claimants into view
slower
builds trust
```

## Rights Floor Review

Fundamental protection-focused.

```text
used when access, consent, personhood, survival, or public commons are at stake
can override property claims
```

## Quarantine Hold

Biosecurity-focused.

```text
pauses operation or limits flow
may protect settlement
may harm access
```

## Archive Witness Review

Continuity-focused.

```text
checks source chains, old contracts, prior failures, and Chronicle records
useful against forged claims
```

Design rule:

```text
The procedure is part of the player choice.
```

---

# 6. Resolution Verbs

After choosing a frame, the player chooses a resolution verb.

Core verbs:

```text
approve
deny
pause
restrict
transfer
share
inspect
quarantine
decommission
retroactively authorize
convert to commons
lease
sanctuarize
monitor
escalate
```

## Example: Patch Conduit

Possible resolutions:

```text
Approve temporary public operation.
Restrict to low-pressure flow.
Require inspection within 72 hours.
Deny Utility Sovereign transfer claim.
Convert node to Watershed Commons stewardship.
Quarantine salvaged seal material.
Escalate license dispute to Archive Witness.
```

Design rule:

```text
A good resolution should change both the machine and the politics around the machine.
```

---

# 7. Example Full Dispute: Patch Conduit Mk0

## Situation

The player repaired a broken water pipe using salvaged materials.

The repair restored partial water flow before full authorization.

## Device State

```json
{
  "node": "/dev/sym/water/patch_conduit_alpha",
  "status": "active",
  "repair_grade": "rough_emergency_seal",
  "flow_delta": 0.34,
  "leak_risk": "medium",
  "pressure_limit": "low",
  "authority_scope": "emergency_repair",
  "inspection_required": true
}
```

## Claims

### Watershed Commons

```text
Keep water flowing.
Maintain public override.
Do not transfer control to private pump authority.
```

### Continuance Office

```text
Temporary repair requires inspection.
Node must be certified or decommissioned within 72 hours.
```

### Utility Sovereign

```text
Pipe segment belongs to old licensed grid.
Operation requires proprietary maintenance key.
```

### Quarantine Authority

```text
Ceramic seal was salvaged from contaminated site.
Flow should remain restricted until material assay.
```

## Player Frames

```text
Emergency Order
Technical Inspection
Commons Hearing
Rights Floor Review
Archive Witness Review
```

## Possible Outcomes

### Outcome A: Emergency Public Operation

```text
water restored quickly
public access preserved
inspection deadline created
Utility Sovereign anger increases
leak risk remains
```

Chronicle:

```text
The player made the water move before every office agreed it had the right to.
```

### Outcome B: Technical Restriction

```text
node limited to low-pressure flow
safety improves
settlement receives less water
Continuance Office trust rises
Watershed Commons frustrated
```

Chronicle:

```text
The pipe was allowed to live only within the limits of its weakest seal.
```

### Outcome C: Transfer to Utility Sovereign

```text
repair stabilized with proprietary key
water flow improves
public override weakened
Utility Sovereign influence rises
Rights Floor concern triggered
```

Chronicle:

```text
The water returned through a gate someone else now owned.
```

### Outcome D: Commons Hearing

```text
decision delayed
new evidence unlocked
public legitimacy rises
short-term water stress continues
Utility Sovereign claim scrutinized
```

Chronicle:

```text
The settlement chose to hear the pipe as a public question.
```

### Outcome E: Quarantine Hold

```text
contamination risk contained
water flow reduced
Quarantine Authority trust rises
settlement stress rises
material assay quest unlocked
```

Chronicle:

```text
The repair was not rejected. It was asked to prove what it carried.
```

---

# 8. Consequence Types

Every adjudication result can affect multiple layers.

## Mechanical Consequences

```text
flow rate
pressure limit
leak risk
maintenance timer
device reliability
power draw
automation budget
```

## Civic Consequences

```text
public access
operator liability
inspection deadline
faction trust
legal precedent
rights floor review
ownership status
```

## Ecological Consequences

```text
wetland flow
toxin transport
species habitat
soil saturation
downstream access
biosphere agency concern
```

## Narrative Consequences

```text
Chronicle line
local rumor
faction memory
future permit friction
new witness requirement
site reputation
```

Design rule:

```text
No adjudication outcome should be only narrative or only mechanical.
```

---

# 9. Infrastructure Status Tags

Registered nodes should carry durable status tags.

```text
UNREGISTERED
INITIALIZED
AUTHORIZED
TEMPORARY
CONTESTED
UNDER_REVIEW
QUARANTINED
COMMONS_ASSET
PRIVATE_LICENSED
RIGHTS_FLOOR_PROTECTED
NULL_SUSPECT
DECOMMISSIONED
ARCHIVED
```

Example:

```json
{
  "node": "/dev/sym/water/patch_conduit_alpha",
  "tags": [
    "INITIALIZED",
    "TEMPORARY",
    "CONTESTED",
    "UNDER_REVIEW"
  ]
}
```

Tags drive future interactions.

Example:

```text
A TEMPORARY node may expire.
A CONTESTED node may trigger faction events.
A NULL_SUSPECT node requires verification before automation.
A COMMONS_ASSET node cannot be privately locked without review.
```

Design rule:

```text
Status tags are the memory of infrastructure.
```

---

# 10. Rights Floor Integration

Some disputes are not ordinary faction disputes.

A Rights Floor claim can override normal procedure when fundamental protections are involved.

Rights Floor triggers:

```text
public water access threatened
oxygen access threatened
forced labor implied
uplift consent ignored
biospheric agency uncertainty present
ecological witness excluded
quarantine used as political capture
private lockout of survival infrastructure
```

Example:

```text
Utility Sovereign claims the water pump.

CIVIC:
Property claim detected.

RIGHTS FLOOR:
Public water access dependency present.
Private lockout prohibited without survival-equivalent alternative.
```

Design rule:

```text
Not every valid claim is allowed to win.
```

---

# 11. Null and Adjudication

Null contamination can attack adjudication by corrupting evidence, prompts, or source chains.

Possible attacks:

```text
fake license record
false emergency declaration
spoofed witness signature
phantom inspection failure
fake contamination warning
duplicated faction claim
altered node status tag
```

Counterplay:

```text
NULL mode inspection
Archive Witness review
analog gauge check
second operator witness
source-chain comparison
network isolation
delayed re-scan
```

Example:

```text
NULL:
Utility Sovereign license record contains recursive timestamp echo.
Archive verification recommended before transfer.
```

Design rule:

```text
The Null does not only attack machines.
It attacks the procedures by which machines become trusted.
```

---

# 12. Minimal Seedworks v0.1 Implementation

Seedworks should implement a very small adjudication system.

## Required

```text
one registered device
three claimants
two procedural frames
three resolution outcomes
one Rights Floor warning
one Chronicle event
```

## Recommended MVP

Device:

```text
/dev/sym/water/patch_conduit_alpha
```

Claimants:

```text
Watershed Commons
Continuance Office
Utility Sovereign
```

Procedural frames:

```text
Emergency Order
Technical Inspection
```

Optional third frame:

```text
Commons Hearing
```

Resolution outcomes:

```text
Temporary Public Operation
Low-Pressure Technical Restriction
Private License Transfer Blocked by Rights Floor
```

Field Deck requirement:

```text
CIVIC mode displays claims.
DIAG mode displays repair safety.
NULL mode checks for spoofing.
Chronicle records selected outcome.
```

Design rule:

```text
The MVP does not need a court.
It needs a pipe that becomes a civic problem.
```

---

# 13. Example MVP Field Deck Flow

## After Repair

```text
CIVIC:
Node initialized without permanent authorization.
Registered infrastructure dispute opened.
```

## Claims

```text
1. Watershed Commons:
Preserve public water access.

2. Continuance Office:
Require inspection within 72 hours.

3. Utility Sovereign:
Transfer node to licensed grid authority.
```

## Player Chooses Emergency Order

```text
Emergency operation authorized.
Scope: public water access.
Duration: 72 hours.
Inspection required.
Private transfer denied pending Rights Floor review.
```

## Chronicle

```text
The settlement accepted an imperfect repair because thirst was not waiting for paperwork.
```

---

# 14. Design Boundaries

## Allowed

```text
small claim sets
typed disputes
procedural frames
status tags
temporary authorization
retroactive review
Rights Floor overrides
Chronicle precedent
```

## Deferred

```text
full legal simulation
complex court UI
large faction parliament
multiplayer voting
formal constitutional litigation
hundreds of concurrent disputes
procedural lawyers
```

The system should feel like civic infrastructure under stress, not a paperwork simulator.

---

# 15. Final Principles

```text
Infrastructure is where politics becomes physical.

Authorization is not ownership.

Repair creates obligation.

A working machine can still be illegitimate.

A legal machine can still be harmful.

A contested device is a story engine.
```

Final line:

```text
The pipe did not belong to whoever touched it first.
It belonged to everyone whose life changed when the water moved.
```
