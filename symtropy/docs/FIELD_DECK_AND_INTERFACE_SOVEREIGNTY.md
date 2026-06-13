# FIELD_DECK_INTERACTION_MODEL.md

# Field Deck Interaction Model

## Version 0.2 — Focus, Co-op, Privacy, and Panic Drop

## Purpose

This document defines how the player actually uses the Field Deck moment-to-moment.

The Field Deck is not just a prop, inventory menu, or smartphone replacement. It is the player’s physical interface with machines, public systems, squad communication, local identity, field diagnostics, archives, and worldline truth.

The challenge is to preserve immersion without making the game frustrating.

The design answer is:

**Use diegetic framing for meaning, but use focused 2D interaction for precision.**

## Core Principle

**Use 2D for precision. Use 3D for spatial understanding. Use physical interaction for risk.**

The Field Deck can be a physical object in the world while still presenting readable interfaces when the player enters Focus Mode.

The player should feel like they are using a rugged device, but they should not be forced to edit complex logic graphs on a tiny tilted 3D screen during a firefight.

## Interface Layers

Field Deck interaction uses three layers.

## 1. World Layer

The Deck exists physically in the player’s hands.

Other players can see it.

Enemies can interrupt it.

Cables, cartridges, batteries, and damage are physical.

Used for:

* raising/lowering the Deck
* patching into machines
* inserting cartridges
* aiming the scanner
* seeing screen glow
* seeing teammate interaction state

## 2. Focus Layer

The Deck is raised and the player focuses on its screen.

The camera slightly stabilizes and zooms toward the display.

The UI becomes readable and precise.

Used for:

* diagnostics
* maps
* logs
* voting
* terminal commands
* device status
* logic block editing
* package verification
* archive review

This is technically a clean 2D UI, but fictionally it is the player focusing on the Deck screen.

## 3. Spatial Overlay Layer

The Deck or settlement terminal projects a simplified 3D topology into the world.

Used for:

* pipe paths
* power routes
* relay coverage
* device graph previews
* signal strength
* blocked conduits
* infected nodes
* squad pings
* physical system layout

The spatial layer is not for dense editing.

It is for understanding where things are.

## Visual Logic Editing

Tier 1 SymLogic Blocks should be edited in **Focus Mode**, not primarily as a free-floating 3D node graph.

### Why Focus Mode

Complex automation needs:

* readable text
* accurate selection
* controller support
* mouse support
* accessibility scaling
* error highlighting
* permission warnings
* simulation preview
* undo/redo
* side-by-side device state

A full 3D graph is beautiful in concept art but poor for dense logic editing.

## Recommended Design

### Editing

Use 2D Focus Mode on:

* Field Deck
* settlement terminal
* workbench screen
* command table
* vehicle console

### Previewing

Use 3D projection for:

* system topology
* signal paths
* dependency graph
* physical device location
* route planning
* live fault overlays

### Quick Field Tweaks

Use compact Deck controls for:

* enable/disable
* set threshold
* reset fault
* run diagnostic
* toggle emergency override
* mount cartridge
* verify package

## Example Flow

The player opens a water controller.

In Focus Mode they edit:

```text
IF tank_0.level < 35%
AND grid_0.power_available
THEN pump_1.enabled = true
ELSE pump_1.enabled = false
```

Then they press **Preview Topology**.

The Deck projects:

* tank
* pump
* pipe
* power feed
* relay
* jammed valve
* blocked access panel

The player understands both the logic and the physical system.

## Field Deck Modes

The Deck should have physical modes.

## OFFLINE

No radio.

No mesh.

No external sync.

Lowest power.

Used for trusted verification and Null-safe diagnostics.

## MESH

Local squad and settlement mesh.

Used for:

* squad text
* short-range pings
* screen mirroring
* relay drones
* local beacons

## DIAG

Device diagnostics.

Used for:

* pumps
* doors
* fabricators
* batteries
* rovers
* terminals
* grid nodes

## ARCHIVE

Logs and history.

Used for:

* Chronicle entries
* public laws
* old votes
* Ghost Civilization records
* black boxes
* witness cartridges

## SCAN

Environmental and signal scanning.

Used for:

* radiation
* temperature
* structural faults
* hidden cables
* Null signals
* resource hints

## EMERGENCY

High-power distress mode.

Used for:

* beacon
* vitals broadcast
* location ping
* rescue signal

Drains battery quickly.

## Input Contexts

The Field Deck requires a clear input context system.

Inputs should never ambiguously control two systems at once.

## Avatar Mode

Deck down.

Controls:

* movement
* camera
* weapon
* tools
* interact
* jump
* sprint

## Deck Glance Mode

Deck partly raised.

The player can still walk slowly.

Used for:

* compass check
* battery check
* quick map glance
* current objective
* squad ping
* warning readout

Movement remains possible but slower.

## Deck Raised Mode

Deck fully raised.

Movement is slowed or limited.

Used for:

* map
* diagnostics
* cartridge mount
* public vote
* log reading
* device status

## Terminal Focus Mode

Full typing and focused interaction.

Movement stops or becomes extremely limited.

Used for:

* shell commands
* SymLogic editing
* package verification
* script review
* archive search
* detailed diagnostics

## Patch Cable Mode

Player is physically connected to a machine.

Constraints:

* limited movement radius
* cable can snag
* cable can be cut
* cable can disconnect
* player may need protection

## Vehicle Console Mode

Deck or vehicle screen controls a rover, drone, crane, or remote device.

Movement controls may shift to vehicle/device controls.

## Neural Overlay Mode

Late-game interface.

High capability, but vulnerable to signal spoofing and cognitive corruption.

## Accessibility HUD Mode

Optional mode for accessibility and comfort.

Can expose essential survival data without requiring constant Deck raising.

## Panic Drop

Panic Drop is mandatory.

No menu or terminal mode should trap the player.

The player should instantly leave any Deck state when they:

* press sprint
* raise weapon
* dodge
* take damage
* trigger panic key
* fall
* get grabbed
* receive critical warning

The Deck drops onto its chest lanyard.

Normal avatar control returns immediately.

The interaction may be interrupted, but never at the cost of player agency.

## Rule

**Combat recovery must be faster than interface immersion.**

## Glance vs Raise Threshold

The Deck should not be required for every tiny survival check.

## Minimal Gear Overlay Handles

* critical health warning
* oxygen danger
* radiation danger
* heat/cold danger
* current tool mode
* weapon ammo when weapon is raised
* immediate squad distress
* low battery warning

## Deck Required For

* why a system is failing
* how to fix a machine
* exact map detail
* terminal access
* public ledgers
* voting
* diagnostics
* archives
* code/scripts
* route planning
* package verification

The player should raise the Deck when asking:

**Why is this happening?**

or:

**How do I change it?**

## Co-op Visibility

The Field Deck should be visible to other players, but not always fully readable.

## Public Screens

Settlement terminals, workshop screens, command tables, and vehicle consoles are shared by default.

Nearby players can read:

* maps
* logs
* device states
* public scripts
* public votes
* Chronicle entries
* diagnostics
* warnings

This supports collaborative engineering.

## Personal Field Deck

Other players can physically see the Deck, but default readability is limited.

A nearby teammate may see:

* mode
* warning color
* major error text
* big map shapes
* ACCESS DENIED
* NULL SIGNAL
* PUMP LOCKED
* battery warning

They should not automatically read:

* private credentials
* medical data
* private messages
* identity keys
* hidden inventory
* restricted Archive records
* faction secrets

## Share Mode

The player can intentionally mirror their Deck screen to squadmates.

Requirements may include:

* local mesh connection
* squad permission
* enough battery
* no heavy jamming
* no privacy-locked content

Share Mode supports callouts:

“Mirror my Deck. I’m patching the door. Tell me when the drones push left.”

## Private Mode

Some screens mask sensitive fields.

Used for:

* credentials
* medical data
* identity keys
* private messages
* restricted Archive testimony
* faction secrets
* voting privacy

A teammate looking over the shoulder may see:

```text
PRIVATE FIELD MASKED
CREDENTIAL VIEW ACTIVE
```

## Forced Visibility

Some societies may require public visibility.

Examples:

* Security Protectorate requires checkpoint credential display.
* Mutualist Assembly requires public water ledger transparency.
* Archive Order requires witness mode during evidence extraction.
* Industrial Compact requires worker quota dashboard visibility.

Interface visibility becomes political.

## Systems Operator Role

The Field Deck creates a real co-op role.

The Systems Operator is not just a hacker minigame player.

They are a battlefield infrastructure specialist.

During missions they can:

* place relay pucks
* patch into terminals
* read device truth
* open locked doors
* isolate infected machines
* reroute power
* scan Null signals
* verify credentials
* mirror diagnostics
* recover black-box logs
* trigger emergency shutdowns
* run local scripts
* certify evidence

They are vulnerable while working.

Squadmates protect them.

## Example Combat Scenario

A squad raids a Null-infected foundry.

The Systems Operator crouches behind a concrete barrier.

They patch the Field Deck into a door panel.

The screen reads:

```text
DOOR_AUTHORITY: corrupted
LOCAL POWER: unstable
NULL SIGNAL: rising
OVERRIDE: pending witness
```

A teammate holds the hallway.

Another places a relay puck.

The operator runs an isolation command.

The door opens.

The squad pushes through.

This makes hacking spatial, dangerous, and cooperative.

## Screen-Space vs In-World Rendering

The rendering model should support both immersion and usability.

## In-World Screen

Used when:

* looking at another player’s Deck
* seeing a terminal from a distance
* watching public screens
* sharing diagnostics in 3D space

## Screen-Space Focus

Used when:

* player is actively operating the Deck
* editing logic
* typing commands
* reading dense logs
* browsing maps
* voting
* reviewing scripts

This is acceptable because the fiction is focused attention.

## Render Strategy

For v0.1:

* amber vector display
* monospaced text
* simple UI panels
* subtle scanlines
* low-power glow
* jitter under Null infection
* offscreen render texture for in-world screens
* screen-space UI for focused interaction

Do not overbuild holograms early.

The Mk0 Deck should feel primitive.

## Null Interface Effects

Null corruption should be legible.

Effects:

* text jitter
* red pulse through amber screen
* repeated logs
* phantom command echo
* false signal pings
* scrambled device names
* cursor movement
* flickering map topology
* warning tones
* screen burn-in shapes

Accessibility settings must allow these effects to be reduced.

## Interface Sovereignty

The Field Deck is part of politics.

Core question:

**Who controls what reality looks like?**

Different societies may modify interface behavior.

## Mutualist Interface

* public water dashboards
* repair task boards
* transparent ledgers
* shared diagnostics

## Industrial Interface

* quotas
* output dashboards
* production warnings
* efficiency overlays

## Security Interface

* credential checks
* restricted maps
* access revocation
* patrol warnings
* surveillance notices

## Archive Interface

* signed logs
* witness mode
* cold verification
* source confidence
* evidence chains

## Null Interface

* false certainty
* corrupted truth
* fake ally markers
* phantom safety
* recursive prompts

## Seedworks v0.1 Scope

Implement:

* raise/lower Deck
* Deck Glance Mode
* Deck Raised Mode
* Terminal Focus Mode
* Panic Drop
* amber display style
* basic map
* diagnostics panel
* one patch cable interaction
* one public terminal
* one shared terminal screen
* limited co-op shoulder visibility
* one Share Mode prototype
* one privacy-mask example
* one Null glitch effect
* one waterworks control flow

Do not implement full neural overlays yet.

## Success Criteria

The system works if players say:

* I feel exposed when using the Deck in danger.
* I can still escape instantly.
* I understand why the Deck matters.
* I can use diagnostics without fighting the UI.
* My teammate can help me operate a terminal.
* Private information feels protected.
* Public information feels politically meaningful.
* Null corruption feels scary but readable.

## Final Principle

The Field Deck should make one idea tangible:

**Reality is not what the HUD says.
Reality is what survives verification.**
