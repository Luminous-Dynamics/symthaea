# Spore Visual Composition v0.3.3 — Perceptual Hierarchy, Restraint, and Continuity

## Status

Design-frozen successor to Spore Boot Ecology v0.3.2 (`#238`).

This document does **not** change boot authority, boot truth, qualification status, DRM ownership, handoff policy, or physical-host enablement. It defines the next visual-only contract so implementation can improve perceptual quality without adding another renderer state machine or another source of boot truth.

The core invariant remains:

> Spore may observe boot; Spore must never be required for boot.

The visual invariant added here is:

> Topology tells the story. Light reveals the topology. Holography gives it space. Text confirms facts. Everything else must earn its presence.

---

## 1. Why v0.3.3 exists

v0.3.2 deliberately built the exact renderer as independently disposable layers:

1. organic procedural topology;
2. holographic spatial field;
3. membrane / caustics / bloom fidelity;
4. sparse factual identity microtype.

That separation is correct, but it creates a new perceptual risk: each layer can be individually tasteful while their simultaneous independent gains produce a frame with too many competing focal signals.

The v0.3.3 goal is **not more visual effects**. It is coordinated attention.

The current semantic model already contains an important signal that should become useful to rendering: every `BootStage` carries a bounded `intensity`. The next renderer should consume that existing value rather than inventing a parallel visual-state protocol.

The current base ecology also composes visual parameters that should increasingly have exact pixel consequences rather than existing only as latent genome metadata. The first high-value example is `node_density`: endpoint glows should respect the genome's intended density instead of allowing every mature branch to become an equally bright node.

---

## 2. Luminous visual constitution

All future Spore boot, inoculation, greeter-continuity, and desktop-continuity work should satisfy these principles.

### V1 — Truth before spectacle

A visual transition may dramatize a factual state but may not manufacture a factual state. Decorative completion cannot imply successful boot, health, Last Known Good promotion, or physical authority.

### V2 — Topology before effects

The organic topology is the primary visual identity. Membranes, bloom, caustics, spectral echoes, scanline sheen, field chords, and microtype are supporting layers.

### V3 — One hero at a time

Each semantic stage has one dominant visual story. A recovery frame should read first as repair. An update frame should read first as new-generation growth. A mesh-return frame should read first as reconnection.

### V4 — Composition before brightness

State differentiation should prefer geometry, reveal/retraction, negative space, density, and motion direction before simply increasing luminance or saturation.

### V5 — Continuity before novelty

A mature machine should retain recognizable morphology across boots. A lifecycle event modifies an existing visual identity; it should not make the computer look like an unrelated machine every time.

### V6 — Motion follows causality

Movement originates from topology or semantic transitions. No layer receives motion merely because time is passing. Ambient motion must never resemble progress when authoritative progress is unknown, delayed, degraded, or failed.

### V7 — Handoff simplifies

The approach to compositor handoff must monotonically reduce visual complexity. Handoff is resolution, not a victory explosion.

### V8 — Color is redundant meaning

Leaf green, solar gold, holographic cyan, and mycelial white may reinforce state, but state must remain distinguishable when hue discrimination is weak or color is reduced.

### V9 — Text confirms; it does not rescue

`SPORE`, the lifecycle cue, and the current factual stage remain sparse. A user should be able to distinguish the broad lifecycle composition without reading the label.

### V10 — Accessibility remains designed

Calm / Standard / Rich profiles are different art directions over the same semantic frame, not "full experience" versus "effects disabled".

### V11 — Determinism is part of identity

For the same exact genome, semantic frame, profile, resolution, and renderer version, output remains deterministic.

### V12 — Presentation stays expendable

Every visual layer can be omitted, killed, timed out, or replaced without reducing bootability.

---

## 3. VisualCompositionBudget

v0.3.3 should add one small **renderer-local**, cognition-free, I/O-free pure projection:

```text
BootGenome + EcologyFrameState + elapsed renderer time
                         ↓
              VisualCompositionBudget
                         ↓
 topology / accent / mesh / holography / membrane /
 caustics / bloom / identity
```

This is not a new source of boot state. It is a deterministic renderer policy derived only from already-qualified presentation inputs.

Recommended shape:

```rust
struct VisualCompositionBudget {
    topology: f32,
    accent: f32,
    mesh: f32,
    holography: f32,
    membrane: f32,
    caustics: f32,
    bloom: f32,
    identity: f32,
}
```

All gains are finite and clamped to `[0, 1]`.

The policy should consume the current `BootStage.intensity` as a bounded semantic-energy input. Intensity must **not** become a claim of health, confidence, completion, or authority.

The budget coordinates layers; it must not alter `BootGenome`, stage order, live semantic time, handoff readiness, or any host-owned lifecycle decision.

### Hard policy invariants

1. `topology` remains the largest or tied-largest ordinary structural gain.
2. No stage can maximize all secondary layers simultaneously.
3. `Repair` / `RetractFailedGrowth` suppress unrelated holographic spectacle.
4. `GrowthRing` gives the generation accent priority over caustics and scanline texture.
5. `MeshLink` may elevate spatial connectivity, but cannot erase topology.
6. `Settle` reduces secondary motion and contrast.
7. `Handoff` monotonically simplifies all secondary layers to zero or near-zero.
8. Lower `BootStage.intensity` cannot increase a secondary layer for an otherwise identical frame.
9. Unknown/degraded live presentation may hold or simplify but may not fabricate forward decorative progress.
10. A zero/near-zero secondary budget should allow the renderer to skip that layer's work entirely.

---

## 4. Stage hero map

The exact constants belong in implementation and visual review, but the semantic hierarchy is frozen here.

| Stage | Hero | Supporting | Suppress |
| --- | --- | --- | --- |
| `Blackout` | substrate / absence | none | everything |
| `DormantCore` | focal seed + negative space | faint membrane | mesh, caustics, strong bloom |
| `Relight` | existing topology illumination | cyan spatial depth | new-growth spectacle |
| `Germinate` | first topology emergence | focal membrane | mesh, scanline texture |
| `Grow` | topology extension | restrained depth | strong caustics |
| `Anastomose` | topology joining | subtle field chords | unrelated ring spectacle |
| `Repair` | gold/white repair paths | quiet membrane | mesh, strong sweep, busy caustics |
| `GrowthRing` | new generation ring | established topology | competing gold effects |
| `HardwareBud` | one new local region | topology continuity | global celebration |
| `RetractFailedGrowth` | candidate retraction + stable known-good structure | restrained repair light | ambient progress cues |
| `MeshLink` | distant links reconnecting | cyan/green spatial field | heavy membrane detail |
| `Settle` | mature stable topology | low breathing | strong sweep / caustics |
| `Handoff` | visual resolution / aperture | factual microtype briefly | all decorative competition |

---

## 5. Lifecycle composition contracts

### Germination

First boot should begin with unusually strong negative space. A focal seed establishes position, then a small number of primary structures define the silhouette before finer branching arrives.

The visual should feel like **emergence**, not a loading spinner made organic.

### Ordinary return

A normal boot should not replay first birth. It begins with more already-established structure, quickly reveals the known morphology, then performs only the amount of growth justified by the current genome/stages.

### Relighting

Suspend/hibernate return should preserve topology and move primarily through illumination, local reconnection, and depth. It must not look like regrowth from zero.

### Update

A generation change should alter composition locally: a new ring, region, or structured extension appears around/within established morphology. The old structure remains legible so the user sees continuity plus change.

### Rollback

Rollback is one of Spore's strongest possible visual signatures.

Candidate growth retracts while known-good structure remains spatially stable. Solar-gold/white repair may seal the transition, but rollback must not look like success fireworks or catastrophic red failure.

### Recovery

Recovery should read as **repair under restraint**. Kintsugi marks are the hero. Holographic membranes, caustics, mesh links, and energy sweeps should step back while the repair path is active.

### Mesh return

Connectivity should appear as reconnection between already-existing distant points, not as generic network lines over the whole screen.

### Handoff

Handoff should resolve rather than explode:

```text
mature ecology
    ↓
secondary motion slows
    ↓
caustics / scanline sheen disappear
    ↓
holographic echoes converge toward primary geometry
    ↓
gold accents settle toward white / green structure
    ↓
peripheral illumination reduces
    ↓
one stable topology / aperture relationship remains
    ↓
display ownership transfers
```

No white-flash victory requirement belongs in the mature boot path.

---

## 6. Negative space and node-density discipline

The current visual genome includes `node_density`; v0.3.3 should make it consequential in exact pixels.

Endpoint/node glow should be deterministically sampled from the genome rather than emitted for every eligible mature branch.

Required properties:

- primary/root structures remain legible;
- higher node density produces more visible structural nodes under the same topology;
- lower node density creates meaningful dark space instead of merely dimming all nodes;
- recovery marks and explicitly semantic event nodes may override ordinary density suppression;
- node selection remains deterministic from the genome seed;
- changing node density must not change boot state or topology identity.

This is both an aesthetic improvement and a performance opportunity.

---

## 7. Palette realization

v0.3.3 should begin closing the gap between composed genome palette metadata and exact pixel output.

Priority order:

1. use existing `solar_gold_fraction` only for semantic gold accents, as today;
2. make `leaf_green_fraction` / `mycelial_white_fraction` influence the structural green↔white balance without overwhelming depth cues;
3. treat `color_temperature_k` only as a **small bounded tint**, never a wholesale palette replacement;
4. keep holographic cyan outside the normalized organic palette because cyan represents spatial projection/connectivity rather than material topology.

Do not make all palette fields visually strong merely because they exist. Perceptual review outranks parameter maximalism.

---

## 8. Fast and slow boot choreography

### Fast boots

A short real boot must preserve the semantic silhouette rather than play the full ceremony at high speed.

Compress by:

- dropping expendable ambient intervals first;
- preserving stage entry/exit landmarks;
- retaining at least one legible topology transition;
- allowing secondary effects to be skipped entirely;
- preserving bounded handoff simplification.

Never fake progress to make a short boot visually complete.

### Slow boots

When semantic progress is held, the renderer may use bounded ambient drift that does not imply forward state:

- membrane breathing;
- very small parallax change;
- established-node luminance variation;
- static-topology depth cues.

It must not reveal new factual topology, add generation rings, perform repair, or enter Handoff merely because wall-clock time passed.

This composes with the existing semantic visual clock / `Hold` / `AmbientDrift` work in the later Boot Ecology convergence stack.

---

## 9. Calm / Standard / Rich

These are presentation profiles only.

### Calm

- topology dominant;
- minimal node breathing;
- very restrained holography;
- no scanline sheen;
- low/zero caustics;
- reduced bloom;
- no loss of factual lifecycle differentiation.

### Standard

The intended reference composition and qualification profile.

### Rich

- fuller membrane depth;
- bounded caustics and spectral echoes;
- richer spatial connectivity;
- still subject to the same composition budget and one-hero rule.

Rich must not produce additional semantic claims or longer boot delays.

---

## 10. Exact visual evidence for v0.3.3

Do not introduce an automated "beauty score".

The existing exact-pixel lint remains appropriate for blank output and semantic-collapse regressions. v0.3.3 should add **descriptive, non-aesthetic** evidence that helps human review:

### Temporal contact sheets

For each representative lifecycle scenario, capture the same normalized semantic landmarks:

- early establishment;
- first meaningful transition;
- stage midpoint;
- late stage;
- pre-handoff / handoff boundary where authorized.

This makes temporal choreography reviewable without pretending a single still frame represents motion quality.

### Layer occupancy report

Optionally record deterministic descriptive metrics such as:

- non-background pixel fraction;
- bright-pixel fraction;
- bloom-active pixel fraction;
- secondary-layer enabled/disabled state;
- visual budget values;
- final pixel digest.

These metrics may catch regressions such as "everything is glowing" or "the frame became blank". They must not be described as aesthetic quality scores.

### Required human review questions

1. What receives first attention without reading text?
2. Can rollback/recovery/update/relight be distinguished by composition?
3. Does the topology remain the primary object?
4. Are cyan and gold competing?
5. Is negative space preserved?
6. Does handoff become calmer frame by frame?
7. Does a stalled boot remain visually alive without pretending progress?
8. Does Calm still look intentional and premium?

---

## 11. Performance contract

Composition budgeting should be able to reduce work, not only alpha.

If a secondary gain falls below a small deterministic epsilon, the renderer should skip the corresponding analytic layer/pass rather than draw fully and composite near-zero pixels.

The first implementation should add no per-frame heap allocation and no new external dependencies.

Existing renderer-cost evidence remains authoritative for CPU rendering performance. v0.3.3 should compare representative v0.3.2 and v0.3.3 exact renderer cost at the same resolution/frame set before increasing fidelity further.

---

## 12. Implementation sequence

### VC-01 — Pure composition policy

Add a small internal `visual_composition` module with bounded gains and pure tests. Consume existing `BootStage.intensity`; do not change boot protocol or authority.

### VC-02 — Thread budget through exact renderer layers

Apply the budget to:

- base topology / semantic accents / mesh;
- holographic field;
- membrane / caustics / bloom;
- sparse identity layer.

### VC-03 — Node-density realization

Make `MorphologyParameters.node_density` deterministically control ordinary endpoint glow density while preserving semantic repair/event highlights.

### VC-04 — Structural palette realization

Make green/white palette fractions have bounded structural consequences. Add temperature tint only if exact-gallery review supports it.

### VC-05 — Temporal galleries

Add representative temporal contact sheets and composition-budget metadata to exact visual evidence.

### VC-06 — Fast/slow convergence

After the later semantic visual-clock/projection stack qualifies, connect the same composition policy to truthful fast compression and slow ambient hold. Do not add another clock.

### VC-07 — Continuity profiles

Expose Calm / Standard / Rich as presentation-only quality profiles and carry the same morphology lineage into later greeter/desktop continuity work.

---

## 13. Tests required before aesthetic tuning

Pure tests should establish at least:

- every composition gain is finite and in `[0, 1]`;
- topology cannot be accidentally suppressed below the frozen minimum for ordinary non-blackout frames;
- lowering otherwise-identical stage intensity does not increase secondary gains;
- repair suppresses unrelated holographic/caustic competition;
- update prioritizes the generation accent;
- mesh stage prioritizes mesh without suppressing topology;
- Handoff secondary budgets monotonically decrease with `stage_progress`;
- deterministic budget for identical genome/frame/time;
- node-density endpoint selection is deterministic;
- node density 0 and 1 exercise valid bounded endpoints;
- visual-policy changes cannot change boot state, stage order, or render-policy authority.

Renderer tests should continue to assert deterministic exact pixels for frozen fixtures.

---

## 14. Explicit non-goals

v0.3.3 does **not** authorize:

- physical-host Spore enablement;
- additional boot delays;
- new health or readiness inference;
- journal parsing;
- user-content visualization;
- GPU/shader dependencies in the canonical early-boot path;
- consciousness/emotion/sentience claims;
- a visual-success signal being treated as boot success;
- replacing the existing semantic visual clock;
- adding another lifecycle state machine;
- an automated aesthetic score.

---

## Exit condition

v0.3.3 is ready for visual review only when:

1. v0.3.2 remains independently reproducible as its frozen comparison point;
2. composition policy tests pass;
3. exact renderer layers consume one shared budget rather than independently maximizing attention;
4. node density has a deterministic pixel consequence;
5. temporal contact sheets exist for healthy, first boot, update, rollback, recovery, relight, and mesh-return cases;
6. exact-pixel lint remains green;
7. renderer-cost evidence exists at representative resolutions;
8. human review confirms that semantic states read through composition before labels;
9. handoff visibly simplifies rather than culminates in spectacle;
10. no boot-authority or physical-enable boundary changed.

The target is not "more impressive effects." The target is a machine lifecycle that feels coherent, beautiful, truthful, restrained, and unmistakably Spore.
