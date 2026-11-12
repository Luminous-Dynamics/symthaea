# Spore Visual Fidelity v0.3.2 — Light, Membrane, Ceremony, Identity

## Goal

Spore should exceed its own concept art by doing things static artwork cannot do: remain truthful to machine state, preserve a machine-specific visual lineage, and transform continuously through installation, boot, recovery, resume, and handoff.

v0.3.2 therefore focuses on **fidelity and continuity**, not more decorative HUD chrome.

The boot safety invariant is unchanged:

> Spore may observe boot; Spore must never be required for boot.

## Renderer stack

The exact live and preview renderer is now deliberately layered:

1. **Organic topology** — deterministic spores, curved hyphae, branching, anastomosis, repair marks, rollback retraction, generation rings, mesh links.
2. **Holographic field** — projected membranes, segmented orbitals, sparse anchors, spectral echoes, energy sweeps.
3. **Fidelity pass** — seeded spore membrane detail, bounded caustic interference, thresholded low-resolution bloom.
4. **Identity layer** — sparse dependency-free factual microtype (`SPORE`, lifecycle cue, current stage).
5. **Lifecycle-specific wrapper** — Inoculation chamber and install-route signature when the renderer is used for installation. Inoculation deliberately bypasses boot-only Germination identity and supplies its own factual install identity.

Each layer is optional presentation. None is allowed to become a dependency of system startup.

## Bloom without a GPU dependency

Concept art gains much of its perceived depth from bloom. The early-boot renderer implements a small CPU bloom workspace instead of requiring a shader compiler or graphics stack.

Properties:

- source buffer is downsampled by 4× in each dimension;
- bloom storage therefore uses approximately 1/16 the full pixel count;
- only pixels above a luminance threshold contribute;
- a fixed five-tap separable blur is used;
- screen-style compositing can brighten but never darken the authoritative frame;
- the workspace is allocated once and reused between frames;
- stage-dependent gain boosts relight, repair, mesh return, germination, and generation transitions;
- handoff gain decreases instead of keeping a bright afterimage while DRM is being released.

This is a fidelity effect, never a progress or health signal.

## Spore membrane

The central focal organism receives a deterministic membrane shell derived from the same `BootGenome`.

The membrane is intentionally procedural rather than an image asset:

- two displaced elliptical shells suggest translucency;
- a bounded sixteen-node geodesic-like cell network creates recognisable surface structure;
- subtle inner light suggests volume underneath the membrane;
- shell geometry breathes slowly but deterministically;
- repair/update stages bias the membrane toward solar gold;
- relight/connectivity stages bias toward holographic cyan;
- ordinary healthy growth remains white/green.

The membrane is strongest for Central Spore and HDC Organic families and quieter for distributed families such as Fairy Ring.

## Bounded caustics

Three deterministic interference arcs move slowly through the focal field. Their purpose is to imply light passing through a projected translucent organism, not to create particle noise.

Rules:

- exactly three arcs;
- no unbounded particle allocation;
- low opacity;
- deterministic from genome + elapsed sequence time;
- slightly warmer during repair/update;
- slightly cooler during resume.

## Sparse factual microtype

Concept art also benefits from typography, but pulling a desktop font stack into early boot would be the wrong tradeoff. v0.3.2 adds a tiny built-in 5×7 uppercase raster alphabet with no external assets.

The final renderer draws only:

- `SPORE`;
- one lifecycle cue such as `GERMINATION`, `RELIGHTING`, `APPLYING GENERATION`, `RESTORING KNOWN GOOD`, or `RECOVERY`;
- one small current-stage label such as `WEAVING`, `REPAIRING`, `GENERATION RING`, or `HANDOFF`.

The labels fade before compositor takeover and are explicitly tested not to make consciousness/sentience claims. The ecology remains understandable without them.

## Installation is a family of ceremonies

The shared Inoculation phase model remains:

1. Attestation
2. Preparing Substrate
3. Weaving System
4. Seeding Security
5. Opening Channels
6. Personalizing
7. Finalizing
8. Complete

v0.3.2 adds a second dimension: **the installation route gets its own visual signature**.

### Web Portal — projected aperture

Tall nested portal membranes frame the incubation chamber with a slow verification sweep.

### USB Forge — seed forge

Solar-gold basin arcs and bounded rising write lines make physical image creation feel materially different from an ordinary network install.

### WSL2 Pivot — bridge bloom

Two overlapping projected fields become connected by a central bridge lattice, visually representing a transition between environments without claiming anything about user data.

### Asahi Handshake — orchard orbits

Five restrained petal-like orbital projections create a smooth handoff language appropriate to guided Apple Silicon conversion without copying vendor branding.

### LAN Inoculation — mesh seeding

Satellite nodes and converging field links make remote/LAN installation read as seeding a system through an existing network.

### Local Direct — substrate contours

Quiet layered substrate contours emphasize local disk/system formation with the least ornamental ceremony.

These route signatures wrap the same factual installation phase state and the same machine-specific morphology. They never change installation authority or success criteria.

## Inoculation narration policy

The web ceremony uses the same epistemic language as the framebuffer UI. Narration describes verifiable actions — trust establishment, reproducible transition environment, storage layout, derivations, runtime components, configuration, and first verified boot — rather than asserting that hardware has awakened or become sentient.

Optional Phi telemetry can remain visible as a research metric where enabled, but it is explicitly not an installation/boot success signal.

## Exact-pixel evidence

CI renders three independent exact-pixel review artifacts through the same presentation stack used by live DRM:

- **Boot lifecycle matrix** — sixteen system-history cases.
- **Inoculation phase matrix** — eight phases × four progress samples.
- **Inoculation path matrix** — six installation routes × three representative lifecycle phases.

The browser galleries are produced from the renderer's PPM output. Generated concept art is not used as validation evidence.

After gallery publication, `scripts/spore_visual_evidence.py` seals every review root. Each `.ppm`, `.png`, `.json`, and `.html` review file receives a SHA-256 entry in `EVIDENCE.sha256`. `evidence-manifest.json` records the exact source commit, file count, total bytes, and individual hashes, and CI verifies the seal after creation.

The seal proves which renderer bytes were produced and reviewed. It is not a boot authority, health signal, or security key.

## Renderer-cost evidence

The additional bloom, membrane, holographic and microtype layers must earn their visual cost.

`spore_render_probe` exercises the complete organic + holographic + fidelity + factual-identity stack with a fixed deterministic genome and records:

- resolution and frame count;
- sequence duration;
- mean frame cost;
- p50 frame cost;
- p95 frame cost;
- maximum frame cost;
- a BLAKE3 digest of the final rendered frame.

The initial CI probe intentionally runs at 640×360 for 24 frames and is **evidence-only**. It validates that measurements are finite and stores the report as a CI artifact, but imposes no arbitrary performance cutoff. Thresholds should be introduced only after representative 1080p and 1440p measurements exist on known physical hardware.

The boot deadline always has priority over visual completeness: expensive presentation may lose detail or frames; boot may not wait for it.

## What comes next

Candidate v0.3.3 work should prioritize continuity over ornament:

- write a bounded visual handoff receipt before compositor takeover;
- allow the greeter/session shell to continue the same `BootGenome` rather than cutting to an unrelated login screen;
- persist shutdown contraction anchors so the next healthy boot can germinate from the same abstract spore positions;
- add Calm / Standard / Rich presentation profiles with identical factual semantics;
- evaluate DRM page flips/double buffering for cleaner motion;
- measure 1080p/1440p CPU budgets before increasing bloom radius or effect count;
- add reduced-motion and reduced-bloom accessibility behavior.

The design test remains:

> Does this make the computer's real state and continuity more legible and beautiful, or is it merely decoration?
