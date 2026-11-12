# Spore Boot Ecology v0.3

**Status:** implementation tranche 1 — state receipt + deterministic composer  
**Scope:** beautiful, state-aware boot visuals that never become a boot dependency

## Design intent

Spore boot visuals should feel like one evolving visual organism rather than a collection of prerecorded splash screens. A machine receives a stable visual identity at installation, while every startup is procedurally varied by the previous system state and boot lineage.

The low-level state model stays factual. Organic language belongs in the renderer and UX, not in system telemetry.

Installation vocabulary:

- **Incubation** — preparing the target and system closure
- **Inoculation** — installing/deploying Spore/NixOS to the target
- **Germination** — first boot
- **Attunement** — optional mesh participation
- ordinary boot — starting / returning / growth
- resume — relighting

No boot-critical component depends on claims about machine consciousness.

## Architecture

```text
system facts
    |
    v
BootStateReceipt ----------------------+
    |                                   |
    |                    MorphologyLineage
    |                                   |
    +-------------> BootEcologyComposer |
                         |               |
                         v               |
                     BootGenome <--------+
                         |
                         v
                 procedural renderer
                         |
                         v
                      DRM/KMS

health gate ---> BootOutcome ---> MorphologyLineage
```

The renderer has **no authority** over boot. It consumes state and may disappear at any moment without affecting startup.

## Persistent identity and variation

A random 256-bit `machine_visual_seed` is generated once for visual identity. It is explicitly not an authentication key.

Each `BootGenome` derives a new deterministic seed from:

- machine visual seed
- monotonic boot counter
- previous termination category
- previous uptime
- generation transition
- generation health
- storage recovery state
- coarse OOM/thermal event counts
- hardware-topology changed/not-changed state
- coarse mesh state
- morphology lineage

This produces a recognizable family resemblance across boots without repeating the same sequence.

## Morphology families

The v0.3 composer exposes twelve visual grammars:

| Family | Character | Typical use |
| --- | --- | --- |
| `CentralSpore` | single luminous seed radiating outward | first boot / germination |
| `MycelialFan` | curved branching fans | normal boot |
| `LichenCells` | slow tessellating organic cells | normal boot |
| `ConstellationHyphae` | sparse bright nodes joined by fine threads | normal boot / mesh |
| `RiverDelta` | directional bifurcating flow | normal boot |
| `AnastomoticWeb` | branches fuse into loops | mature/connected system |
| `FairyRing` | concentric growth fronts | new NixOS generation |
| `HdcOrganic` | geometric field resolving into organic structure | alternate normal boot |
| `CrystalThaw` | crystalline stillness becoming fluid growth | hibernate resume |
| `KintsugiRepair` | interrupted geometry joined with warm gold seams | recovery / rollback |
| `MemoryGarden` | mature persistent topology | long-lived healthy lineage |
| `MinimalRelight` | existing structure softly illuminates | suspend resume |

These are not videos. They consume shared continuous morphology parameters such as curvature, branching probability, anastomosis probability, node density, turbulence, palette balance, glow radius, growth speed, repair intensity, maturity, and mesh opacity.

## Previous-state mapping

| Previous fact | Visual consequence |
| --- | --- |
| clean poweroff | balanced procedural growth |
| clean reboot | compact growth with continuity |
| new generation | visible gold growth ring |
| rollback | failed growth retracts; known-good topology returns |
| suspend | relight existing structure; do not regrow it |
| hibernate | crystal-thaw relight |
| power loss | asymmetry followed by repair |
| kernel panic/watchdog | repair topology with higher turbulence |
| filesystem journal replay | subtle repair seams |
| filesystem repair | stronger gold repair seams |
| thermal emergency | warmer initial palette cooling toward normal |
| long previous uptime | more mature/anastomotic morphology |
| hardware topology changed | new branch family buds into the network |
| mesh disabled | organism remains complete locally; no distant links |
| mesh enabled, no peers | dim latent mesh links |
| peers present | distant nodes illuminate and connect |

Recovery must be calm and informative, not alarming. Diagnostic details remain available separately.

## Sequence stages

A morphology family receives semantic stages rather than hard-coded scenes:

- `Blackout`
- `DormantCore`
- `Relight`
- `Germinate`
- `Grow`
- `Anastomose`
- `Repair`
- `GrowthRing`
- `HardwareBud`
- `RetractFailedGrowth`
- `MeshLink`
- `Settle`
- `Handoff`

The same stage can look different in every morphology family.

## Safety invariants

The v0.3 `BootRenderPolicy` establishes the contract for later renderer integration:

1. **Fail open.** Renderer failure must never fail boot.
2. **Short acquisition timeout.** If DRM/KMS cannot be claimed quickly, skip animation.
3. **Hard renderer deadline.** Animation lifetime is bounded independently of system startup.
4. **Progress is optional.** No FIFO/event producer may be required for the renderer to start or stop.
5. **Release before compositor.** DRM ownership is relinquished before the display manager/Wayland compositor claims the display.
6. **No personal telemetry.** Boot ecology receipts contain system lifecycle facts only.
7. **Known-good advances only after health success.** A failed candidate generation can never overwrite the last-known-good semantic reference.

## Current / Previous / Last Known Good

The boot experience should expose three semantic system references even if systemd-boot retains a larger generation history:

- **Current** — generation selected for this boot
- **Previous** — immediately preceding built generation
- **Last Known Good** — most recent generation that crossed the post-boot health gate

The visual lineage follows the same rule: speculative candidate growth is not committed to the lineage until the health gate succeeds.

On rollback, the candidate branch can visibly retract before the last-known-good topology resumes.

## Persistence model

`MorphologyLineage` stores only abstract history:

- successful boot count
- recovery mark count
- bounded maturity
- last genome seed
- last-known-good generation identifier

It does **not** store framebuffers, screenshots, user content, journal text, peer identities, or biometrics.

A successful long-lived system therefore develops a subtly more mature appearance over time without unbounded state growth.

## Renderer roadmap

### v0.3a — state/composer foundation (this PR)

- `BootStateReceipt`
- `MorphologyLineage`
- deterministic `BootGenome`
- 12 morphology families
- state-aware stage composition
- fail-open renderer policy
- known-good health-gate semantics
- deterministic unit tests

### v0.3b — exact preview renderer

Extend `symthaea-quicken-fb` with a renderer mode that can write the exact frames used at boot to image files or a frame directory. Approval must happen against the real renderer, not concept art.

Target CLI:

```text
quicken-fb preview --receipt receipt.json --lineage lineage.json --out frames/
quicken-fb run     --receipt receipt.json --lineage lineage.json --device auto
```

### v0.3c — renderer quality

- resolution-independent geometry
- curved/anti-aliased hyphae
- anastomosis and loop formation
- central spore morphology
- multi-pass software glow or lightweight GPU path where available
- stable node identity and traveling pulses
- kintsugi repair seams
- growth-ring rendering for Nix generations
- persistent-map/double-buffer DRM path
- deterministic frame capture tests

### v0.3d — boot lifecycle hardening

- automatic KMS card/connector discovery
- truly nonblocking progress/event source
- renderer acquisition timeout
- hard process deadline
- explicit compositor handoff protocol
- QEMU boot test with no progress writer
- QEMU boot test with progress events
- test that renderer crash still reaches `graphical.target`
- test that timeout still reaches `graphical.target`

### v0.3e — shutdown continuity

A clean shutdown writes the abstract morphology handoff state for the next boot:

- poweroff: activity contracts toward dormant spores
- reboot: quick contraction followed by immediate regrowth
- suspend: topology dims but remains structurally continuous
- hibernate: topology crystallizes
- unexpected loss: no closing receipt exists; next boot infers interruption from system facts

The next startup therefore continues the previous visual story instead of playing an unrelated splash.

## Non-goals

- delaying boot to finish an animation
- rendering private system data
- requiring network connectivity
- treating a crash as emotional trauma
- using visual seeds as security keys
- hiding recovery diagnostics from advanced users
- making the boot renderer responsible for rollback decisions

The goal is simpler: **make real system continuity legible and beautiful.**
