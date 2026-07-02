# Symtropy Design Doc: Death, Reconstitution, and Source-Chain Recovery

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

## Working Title

**The Flesh Is Ephemeral. The Data Is Sacred.**

## Core Thesis

Player death in *Symtropy* should not behave like an arcade respawn.

Death is not a temporary inconvenience where the player reappears with their full inventory, identity, permissions, map memory, and civic authority intact.

In *Symtropy*, the body can be reconstituted.

The source chain must be recovered, witnessed, or permanently scarred.

Core rule:

```text
The player does not only lose health when they die.
They lose local continuity.
```

The death system should split the player across four layers:

```text
body
hardware
identity
memory
```

A dead body is physical.

A Field Deck is recoverable hardware.

A source chain is civic identity.

A Chronicle record is what survives if recovery fails.

Design rule:

```text
Your legacy is not your body.
It is what survives verification.
```

---

# 1. Why Death Must Be Diegetic

The whole *Symtropy* universe is built around:

```text
physical computation
Device Bus authority
Field Deck source chains
Chronicle memory
Archive loss
dead authority
infrastructure legitimacy
```

A magical respawn would break the world’s grammar.

If a player can die and instantly regain all permissions, maps, credentials, and logs, then the game teaches the opposite of its own thesis.

Death must therefore create a playable crisis of continuity.

Design rule:

```text
Death is an archive event before it is a fail state.
```

---

# 2. The Four-Layer Death Protocol

When the player suffers fatal damage, the game executes a four-layer split.

## Layer 1: Body Drop

The player’s body collapses physically in the world.

The corpse remains where it fell, subject to:

```text
gravity
hazards
enemy patrols
radiation
flooding
Null contamination
salvage risk
environmental decay
```

The original Field Deck remains attached to the body unless destroyed, stolen, or forcibly separated.

The dead Deck emits a low-power distress ping.

```text
SIGNAL:
Operator body offline.
Field Deck source core intact.
Mesh distress ping active.
```

Design rule:

```text
A corpse is not a marker.
It is a vulnerable archive container.
```

---

## Layer 2: Reconstitution

The player wakes at the nearest valid reconstitution site:

```text
settlement medical bay
fabricator pod
rover med-console
emergency clone berth
low-grade revival cot
```

The new body is alive but not fully trusted.

The player receives a fresh, blank, uncalibrated Field Deck shell.

It can read public data but cannot prove private authority.

Field Deck startup:

```sh
$ read /dev/sym/identity/status

STATUS: UNVERIFIED_AVATAR
LOCAL_SOURCE_CHAIN: MISSING
ROOT_FIELD_DECK: NOT PRESENT
PUBLIC_DIRECTORIES: READ_ONLY
PRIVATE_CREDENTIALS: REJECTED
CIVIC_STATUS: CITIZENSHIP_PENDING_VERIFICATION
```

Design rule:

```text
Reconstitution restores embodiment.
It does not restore legitimacy.
```

---

## Layer 3: Authority Loss

Until the original source chain is recovered or witnessed, the player is limited.

Restrictions:

```text
cannot issue permissioned Device Bus writes
cannot override public infrastructure
cannot cast civic votes
cannot authorize repairs
cannot modify settlement laws
cannot execute private scripts
cannot access sealed personal archives
cannot validate Chronicle testimony alone
```

Allowed actions:

```text
walk
fight
use basic tools
read public directories
scan physical objects
request help
recover body
receive temporary escort token
perform low-authority repairs
```

Design rule:

```text
A body without verified memory can act, but it cannot govern.
```

---

## Layer 4: Memory Blindness

The blank Deck lacks accumulated local knowledge.

Lost or restricted modes:

```text
ARCHIVE mode: blank or public-only
CIVIC mode: no private authority context
NULL mode: reduced source-chain comparison
DIAG mode: basic physical diagnostics only
personal waypoints: missing
uncommitted logs: missing
local trust annotations: missing
```

The world becomes colder and less readable.

Example:

```text
ARCHIVE:
No local source chain found.
Historical overlays unavailable.

CIVIC:
Operator identity unverified.
Authority interpretation limited to public charter fragments.

NULL:
Cannot compare against personal contamination baseline.
```

Design rule:

```text
Death should make the player feel how much of reality was being carried by memory.
```

---

# 3. Recovery Paths

Death should create a retrieval problem, not an automatic reset.

There are four major recovery paths.

---

## Path A: Physical Recovery

The player returns to their corpse and recovers the original Field Deck.

Steps:

```text
1. Wake as UNVERIFIED_AVATAR.
2. Follow distress ping.
3. Avoid or fight hazards near body.
4. Unclip original Field Deck.
5. Reconnect it to new body rig.
6. Restore source chain.
7. Resolve any contamination warnings.
```

Successful recovery:

```sh
$ sym-identity restore --source recovered_deck

SOURCE CHAIN RESTORED.
CREDENTIALS REBOUND.
ARCHIVE CACHE AVAILABLE.
CIVIC AUTHORITY RESTORED.
```

Design rule:

```text
Recovering your body is really recovering your witness.
```

---

## Path B: Squad Recovery

A living teammate recovers the dead player’s Field Deck.

Options:

```text
carry Deck back physically
extract source core
plug into corpse Deck with patch cable
broadcast signed recovery bundle
escort unverified player to body
```

Co-op Field Deck sequence:

```sh
$ sym-share recover /dev/sym/deck/downed_operator_b

DEAD DECK FOUND.
SOURCE CORE INTACT.
UNCOMMITTED LOGS PRESENT.
WITNESS SIGNATURE REQUIRED.

$ sym-share broadcast --target settlement_mesh

RECOVERY BUNDLE SENT.
IDENTITY REBIND AVAILABLE AT MEDICAL BAY.
```

Design rule:

```text
A teammate can save your life.
A witness can save your continuity.
```

---

## Path C: Remote Mesh Sync

If physical recovery is impossible, a teammate can patch into the dead Deck and transmit a recovery bundle.

This saves:

```text
credentials
uncommitted logs
local source chain
recent mission evidence
Chronicle testimony
```

But may lose:

```text
physical Field Deck hardware
rare cartridges
local cache fragments
damaged files
personal annotations
high-risk private keys
```

Result:

```text
identity restored
hardware lost
source chain marked as remotely recovered
future verification friction possible
```

Design rule:

```text
Remote recovery preserves continuity but leaves a scar.
```

---

## Path D: Black Box Chronicle

If the Deck is lost, destroyed, corrupted, or harvested before recovery, the player’s identity may be partially reconstructed from settlement backups and witness records.

This is not a clean recovery.

It creates a Chronicle scar.

Example event:

```json
{
  "event_type": "SourceChainSevered",
  "worldline_id": "seedworks.local.001",
  "region": "FirstlightBasin.OldWaterworks",
  "operator": "agent_b",
  "deck_status": "lost",
  "recovery_status": "partial_from_mesh_backup",
  "authority_result": "credentials_revoked_pending_review",
  "chronicle_line": "Technician Ivo returned with a body, but the waterworks still held the part of him that could prove what happened."
}
```

Design rule:

```text
If memory cannot be recovered, history must record the wound.
```

---

# 4. Enemy Interaction With Dead Decks

Dead Field Decks are valuable.

Hostile systems may attempt to seize, corrupt, or sanitize them.

## Null Data Harvest

Null systems may attempt to:

```text
compress uncommitted logs
inject false certainty
spoof source-chain fragments
corrupt local evidence
alter personal waypoints
poison future NULL mode comparisons
create false Chronicle leads
```

Field Deck warning on recovery:

```text
NULL:
Recovered source chain contains foreign certainty injection.
Manual review required before civic testimony.
```

Design rule:

```text
Null does not only kill operators.
It edits what their death means.
```

---

## Continuance Recovery

Continuance patrols may confiscate a dead Deck as evidence or contraband.

They may:

```text
seal the source core
classify the operator as compromised
demand procedural review
restore only approved logs
use the Deck to prove unauthorized action
```

Design rule:

```text
The Continuance saves records by controlling them.
```

---

## Utility Sovereign Recovery

Utility actors may treat the Deck as liability evidence or proprietary intrusion record.

They may:

```text
claim the death occurred in licensed territory
lock recovery behind service fees
extract proof of unauthorized access
offer clean restoration in exchange for contract acceptance
```

Design rule:

```text
A corpse in private infrastructure becomes a billing event.
```

---

# 5. Death States

The player should have clear post-death states.

```text
ALIVE_VERIFIED
DOWNED
DEAD_DECK_INTACT
RECONSTITUTED_UNVERIFIED
SOURCE_CHAIN_RECOVERED
REMOTE_RECOVERY_SCARRED
BLACK_BOX_RECONSTRUCTED
SOURCE_CHAIN_COMPROMISED
PERMANENT_ARCHIVE_LOSS
```

Example:

```json
{
  "operator_state": "RECONSTITUTED_UNVERIFIED",
  "body_status": "lost_in_old_waterworks",
  "deck_status": "distress_ping_active",
  "authority": "read_only",
  "archive_mode": "unavailable",
  "civic_mode": "limited",
  "recovery_route": "physical_or_squad"
}
```

Design rule:

```text
Death should be a state machine, not a loading screen.
```

---

# 6. What Gets Lost

Death should not always mean losing everything.

Loss should be layered.

## Always Lost Temporarily

```text
location control
current body
current carried consumables
local physical position
full authority
full Field Deck modes
```

## Recoverable

```text
Field Deck
credentials
source chain
mission logs
personal annotations
uncommitted evidence
rare cartridges attached to rig
```

## Potentially Permanently Lost

```text
destroyed hardware
uncommitted local-only records
untransmitted evidence
corrupted testimony
mission-critical witness fragments
private keys without backup
```

Design rule:

```text
The player should fear unrecovered evidence more than lost loot.
```

---

# 7. Fairness Rules

The death system must be serious but not cruel.

## Required Fairness

```text
corpse location is traceable by distress ping unless jammed
first death teaches recovery safely
player always has basic agency after reconstitution
source-chain loss is warned before becoming permanent
squad recovery is supported
solo recovery has a viable route
irreversible archive loss requires clear cause
```

## Avoid

```text
random permanent identity deletion
unavoidable full credential loss
opaque corpse despawn
punishing experimentation too early
turning death into inventory busywork
making recovery impossible without co-op
```

Design rule:

```text
Death should create dread, not resentment.
```

---

# 8. Seedworks v0.1 Implementation

Seedworks should test recoverable data-severance.

Do not start with strict permadeath.

## MVP Death Test

Scenario:

The player fails inside the Old Waterworks.

They wake at Firstlight Basin camp terminal.

New Field Deck state:

```text
STATUS: UNVERIFIED_AVATAR
LOCAL_SOURCE_CHAIN: MISSING
AUTHORITY: READ_ONLY
ARCHIVE: UNAVAILABLE
CIVIC: LIMITED
```

Objective:

```text
Recover original Field Deck from body inside waterworks.
```

The corpse emits:

```text
LOW POWER DISTRESS PING:
source_core_intact
distance: approximate
signal_quality: unstable
```

The player returns, retrieves the original Deck, and restores their identity.

Successful recovery Chronicle:

```text
The operator returned to the place they died and recovered the part of themselves the settlement could still verify.
```

If the player delays too long:

```text
NULL:
Distress ping weakening.
Uncommitted source chain at risk.
```

If the player fails to recover:

```text
BLACK BOX EVENT:
Local source chain severed.
Partial identity restored from camp backup.
Authority review required.
```

Design rule:

```text
Seedworks should teach that death is survivable, but continuity is recoverable only through action.
```

---

# 9. Permadeath and Hardcore Modes

Permadeath should not be the default v0.1 behavior.

It can exist later as:

```text
hardcore shard rule
expedition contract
black-box no-backup mission
iron witness mode
roguelike worldline branch
optional survival server setting
```

Permadeath should mean:

```text
no valid backup
source chain destroyed
no witness recovery
Chronicle records final loss
```

Example:

```json
{
  "event_type": "PermanentOperatorLoss",
  "operator": "agent_b",
  "cause": "deck_destroyed_no_witness",
  "worldline": "iron_witness.exp_04",
  "chronicle_line": "No one returned with the Deck, so the settlement kept only the shape of the absence."
}
```

Design rule:

```text
Permadeath is not file deletion.
It is final archive loss.
```

---

# 10. Relationship to Chronicle

The Chronicle should record meaningful death events, not every minor downing.

Chronicle-worthy death events:

```text
source chain recovered under fire
Field Deck lost to Null
teammate performed witness recovery
corpse abandoned in disputed infrastructure
identity restored from partial backup
operator died during unauthorized repair
death caused infrastructure adjudication
```

Example:

```json
{
  "event_type": "FieldDeckRecovered",
  "site": "Old Waterworks",
  "operator": "agent_b",
  "recovered_by": "agent_a",
  "deck_status": "intact_but_contaminated",
  "source_chain": "restored_after_null_review",
  "chronicle_line": "The squad did not retrieve a body first. They retrieved the truth it carried."
}
```

Design rule:

```text
The Chronicle records deaths that change what the world can prove.
```

---

# 11. Field Deck UX States

## Verified State

```text
OPERATOR VERIFIED
SOURCE CHAIN ACTIVE
AUTHORITY: FULL
ARCHIVE: LOCAL CACHE AVAILABLE
CIVIC: CREDENTIALS VALID
```

## Unverified State

```text
OPERATOR UNVERIFIED
SOURCE CHAIN MISSING
AUTHORITY: READ ONLY
ARCHIVE: PUBLIC ONLY
CIVIC: LIMITED
RECOVERY OBJECTIVE: ACTIVE
```

## Compromised Recovery

```text
SOURCE CHAIN RECOVERED
NULL REVIEW REQUIRED
CIVIC TESTIMONY LOCKED
ARCHIVE CACHE PARTIAL
```

## Black Box Recovery

```text
IDENTITY PARTIALLY RECONSTRUCTED
UNCOMMITTED LOGS LOST
CREDENTIALS REISSUED UNDER REVIEW
CHRONICLE SCAR RECORDED
```

Design rule:

```text
The UI should make identity loss feel operational, not abstract.
```

---

# 12. Final Principles

```text
The body is recoverable.

The Deck is sacred hardware.

The source chain is civic identity.

The Chronicle is the last witness.

A corpse is an archive container.

A respawn is not a reset.

Death is not failure unless the world can no longer verify what mattered.
```

Final line:

```text
The flesh came back clean.
The truth had to be carried home.
```
# Addendum: Resonatia Bastion Fallback Reconstitution

## Purpose

This addendum extends the death and reconstitution system with a final safety layer:

```text
Resonatia Bastions may act as last-resort continuity recovery sites when local respawn, corpse recovery, squad recovery, and source-chain restoration fail.
```

A Bastion fallback is not a normal respawn.

It is a civilizational reconstruction procedure.

Core rule:

```text
Respawn is medical.
Bastion fallback is historical reconstruction.
```

---

# 1. Design Thesis

A Resonatia Bastion should be able to save the player from total loss, but it must never erase consequence.

The Bastion can rebuild a body.

It can reconstruct enough identity to return the player to the world.

It can gather witness fragments from the Chronicle, settlement mesh, and public Device Bus logs.

But it cannot pretend nothing happened.

Design rule:

```text
A Bastion can save your life, but it cannot restore an unbroken past.
```

---

# 2. When Bastion Fallback Activates

Bastion fallback should only become available when ordinary recovery options fail.

Trigger conditions:

```text
local medical cot unavailable
field camp destroyed
rover med-bay disabled
corpse unreachable
original Field Deck destroyed
original Field Deck captured
squad recovery impossible
settlement mesh cannot fully verify identity
source chain partially severed
```

The Bastion is not the first option.

It is the last institution that still remembers enough of you to try.

Design rule:

```text
The Bastion does not replace recovery missions.
It exists for when recovery becomes impossible.
```

---

# 3. Bastion Reconstruction Inputs

A Bastion reconstructs identity from distributed evidence.

Possible inputs:

```text
last Field Deck sync
Chronicle fragments
Archive Witness signatures
public Device Bus logs
settlement mesh backups
teammate testimony
medical baseline records
mission authorization records
registered infrastructure interactions
prior civic votes
```

The Bastion asks:

```text
Who was this operator?
What did they last prove?
What authority can safely return?
What evidence was lost?
What claims are still under review?
```

Design rule:

```text
Identity is reconstructed from witnesses, not assumed.
```

---

# 4. Bastion Recovery Output

A Bastion fallback restores:

```text
physical body
basic operator identity
public citizenship status
minimum Field Deck functionality
core mission continuity
```

It does not automatically restore:

```text
full private credentials
uncommitted local logs
rare evidence fragments
lost personal annotations
contested infrastructure authority
corrupted source-chain sections
all prior map certainty
```

Example Field Deck state:

```text
STATUS: BASTION_RECONSTITUTED
SOURCE_CHAIN: PARTIAL
AUTHORITY: PROVISIONAL
ARCHIVE: PUBLIC + VERIFIED FRAGMENTS
CIVIC: UNDER REVIEW
NULL: CONTAMINATION BASELINE RESET
CHRONICLE: BLACK_BOX_REVIEW_OPEN
```

Design rule:

```text
The player returns alive, but legally and historically scarred.
```

---

# 5. Real-Time Reconstruction Delay

Bastion fallback may take real-world time, but it should not become a punishment timer.

The delay represents:

```text
identity arbitration
medical fabrication
witness reconciliation
source-chain review
Chronicle reconstruction
authority reissuance
```

Recommended framing:

```text
BASTION CONTINUITY RECONSTRUCTION ACTIVE
Estimated reconstruction: 18 minutes
Status: collecting witnesses / resolving identity fragments / preparing provisional body
```

During the delay, the player should still have useful options.

Possible activities:

```text
observe through public cameras
review Chronicle fragments
choose which identity claims to prioritize
send limited map pings to squadmates
play a low-authority scout drone
prepare a recovery warrant
inspect what was lost
authorize public disclosure of partial testimony
```

Design rule:

```text
Real time may create gravity.
It must not create boredom.
```

---

# 6. Reconstitution Tier Model

## Tier 1: Local Camp Reconstitution

```text
fast
low ceremony
body restored
authority read-only
recover original Field Deck to restore continuity
```

Best for Seedworks v0.1.

## Tier 2: Rover / Field Med-Bay Reconstitution

```text
fast
resource-limited
depends on power and medical substrate
good for expeditions
```

## Tier 3: Settlement Medical Bay Reconstitution

```text
stable
trusted
better backup access
may restore partial authority faster
```

## Tier 4: Resonatia Bastion Fallback

```text
slow
high legitimacy
expensive
reconstructs identity from distributed witnesses
creates Chronicle review
returns provisional authority
```

## Tier 5: Black Box Bastion Reconstruction

```text
very slow
source chain fractured
uncommitted logs lost
credentials reissued under review
permanent Chronicle scar
```

Design rule:

```text
The farther recovery moves from the body and Deck, the more history must stand in for memory.
```

---

# 7. Costs and Consequences

Bastion fallback should cost more than time.

Costs may include:

```text
biomass substrate
medical fabrication capacity
Archive Witness attention
settlement energy budget
Chronicle review burden
temporary authority suspension
faction scrutiny
public disclosure of partial death record
```

Possible consequences:

```text
operator credentials provisional
recent testimony locked pending review
contested infrastructure permissions suspended
faction trust adjusted
Black Box scar added to Chronicle
source-chain uncertainty visible in Field Deck
```

Design rule:

```text
The Bastion preserves continuity at the price of certainty.
```

---

# 8. Faction Reactions

## Resonatia Bastion

Position:

```text
No person should vanish merely because local infrastructure failed.
```

Risk:

```text
Bastion reconstruction may become too powerful if treated as flawless resurrection.
```

## Continuance

Position:

```text
Identity reconstruction requires command review.
```

Risk:

```text
They may demand custody over provisional operators.
```

## Utility Sovereigns

Position:

```text
Bastion recovery creates unpaid medical and infrastructure debt.
```

Risk:

```text
They may attempt to attach service claims to reconstructed bodies.
```

## Watershed Commons

Position:

```text
A community should help recover its people before outsourcing memory to distant institutions.
```

Risk:

```text
They may distrust centralized Bastion identity arbitration.
```

## Archive Witnesses

Position:

```text
Reconstruction must distinguish verified record from compassionate assumption.
```

Risk:

```text
They may delay restoration while seeking cleaner proof.
```

Design rule:

```text
Even resurrection has politics.
```

---

# 9. Bastion Failure Modes

A Bastion can fail imperfectly.

Possible failure modes:

```text
missing witness quorum
contradictory source chains
Null-suspect testimony
medical substrate shortage
identity collision
faction challenge
black-box evidence gap
delayed authority reissue
```

Example:

```text
BASTION REVIEW:
Operator identity reconstructed to 78% confidence.
Recent waterworks testimony unavailable.
Civic authority restricted pending witness review.
```

Design rule:

```text
A failed reconstruction should create playable uncertainty, not arbitrary deletion.
```

---

# 10. Seedworks v0.1 Guidance

Do not implement full Bastion fallback in Seedworks v0.1.

Seedworks should test:

```text
death
local reconstitution
UNVERIFIED_AVATAR state
corpse recovery
Field Deck recovery
source-chain restoration
```

Bastion should be foreshadowed only.

Example Field Deck text:

```text
BASTION CONTINUITY NODE:
Unavailable in Firstlight Basin.

Fallback recovery requires regional Bastion link.
Recover original Field Deck to restore authority.
```

Or:

```text
ARCHIVE:
Nearest Resonatia Bastion link: inactive.
Last successful continuity reconstruction: 2166.
```

Design rule:

```text
Teach local continuity before introducing civilizational recovery.
```

---

# 11. Chronicle Events

## Bastion Reconstruction

```json
{
  "event_type": "BastionReconstitution",
  "operator": "agent_b",
  "source_chain_status": "partial",
  "deck_status": "destroyed",
  "reconstruction_basis": [
    "settlement_mesh_backup",
    "Archive Witness fragment",
    "teammate testimony",
    "public Device Bus logs"
  ],
  "authority_result": "provisional",
  "chronicle_line": "The Bastion returned the operator from the witnesses that remained."
}
```

## Black Box Bastion Recovery

```json
{
  "event_type": "BlackBoxReconstruction",
  "operator": "agent_b",
  "source_chain_status": "fractured",
  "lost_records": [
    "uncommitted waterworks logs",
    "private route annotations",
    "local Null contamination baseline"
  ],
  "authority_result": "under_review",
  "chronicle_line": "The body returned, but the missing hours became part of the settlement's wound."
}
```

---

# 12. Final Principles

```text
Local respawn restores the body.

Field Deck recovery restores the person.

Squad recovery restores continuity through trust.

Bastion fallback restores continuity through civilization.

Black Box reconstruction restores life with a scar.

No resurrection should erase consequence.

No person should vanish merely because one machine failed.
```

Final line:

```text
When the Deck was gone, the Bastion asked the world what it could still prove of you.
```
