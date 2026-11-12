# Spore Visual Language v0.3.1 — Organic Spatial Computing

## Intent

Spore should not look like a conventional boot splash with a decorative animation placed on top. The visual system should make the machine lifecycle legible as a changing spatial ecology while remaining factual, quiet, deterministic, and disposable to system startup.

The target feeling is **organic spatial computing**: the computer appears as a living-looking topology of relationships and state without claiming that the machine is alive or conscious.

The visual invariant remains:

> Spore may observe boot; Spore must never be required for boot.

## Beyond the concept art

The concept images establish a useful direction — luminous mycelium, solar gold, leaf green, mycelial white, restrained cyan holography — but the real renderer should exceed static concept art in three ways:

1. **Truthful motion** — geometry responds to actual lifecycle facts and progress rather than merely looking animated.
2. **Persistent lineage** — the machine develops a bounded visual identity across successful boots and recoveries.
3. **Spatial depth** — projected membranes, parallax anchors, spectral echoes, and field sweeps create a sense that the topology occupies a volume rather than a flat framebuffer.

## Layer model

The renderer is intentionally layered so each effect can fail or be disabled independently.

### L0 — Substrate

Near-black / deep-moss background. No texture should compete with the topology.

### L1 — Organic topology

The deterministic procedural ecology remains the primary visual identity:

- curved hyphae
- branching and anastomosis
- spores and node pulses
- generation rings
- kintsugi recovery marks
- rollback retraction
- mesh links

### L2 — Holographic membrane

A bounded projection field wraps the ecology with:

- nested segmented elliptical membranes
- slight independent rotations and breathing
- sparse shell anchors
- occasional inter-anchor field chords
- stage-sensitive cyan / green / gold spectral behavior

The membranes must feel coupled to the organism, not like rectangular HUD windows pasted on top.

### L3 — Spectral depth

Very small displaced cyan echoes around selected projected structures create diffraction / parallax cues. This must remain subtle enough that text and topology stay crisp.

### L4 — Energy sweep

A low-opacity expanding projected ring moves through the ecology. State-sensitive boosts are allowed during relight, update, repair, and mesh-link phases.

### L5 — Projection texture

Sparse, extremely low-opacity scanline sheen can suggest a projected field on high-DPI displays. It must not become a retro CRT filter.

### L6 — Fidelity pass (v0.3.2)

The real live/preview path now adds a bounded CPU fidelity layer after holography:

- thresholded quarter-resolution bloom;
- deterministic focal spore membrane cells;
- low-opacity volumetric-looking caustic arcs;
- stage-aware highlight gain that fades before compositor handoff.

See `SPORE_VISUAL_FIDELITY_V032.md` for the exact constraints and install-route ceremony work.

## Motion principles

- Prefer slow coherent motion over noisy particle motion.
- Movement should originate from topology or state transitions.
- Avoid perpetual spinning UI chrome.
- Avoid random flicker.
- Use easing for appearance, repair, retraction, and handoff.
- Preserve deterministic replay from the same genome and elapsed time.
- All visual budgets are rendering budgets, never boot delays.

## Holographic color discipline

The palette remains deliberately narrow:

- **Mycelial white** — established structure / completion
- **Leaf green** — growth / healthy continuity
- **Solar gold** — update / repair / verified transition accents
- **Holographic cyan** — projected spatial field / relight / connectivity

Cyan is an accent, not a replacement palette. Generic neon-blue cyberpunk treatment is explicitly out of scope.

## Distinct installation ceremony: Inoculation

Installation should have its own visual grammar rather than reusing ordinary boot.

### Inoculation phases

1. **Attestation** — projected chamber establishes and verifies the substrate boundary.
2. **Preparing substrate** — quiet base field and dormant network become visible.
3. **Weaving system** — the organic topology expands through the chamber.
4. **Seeding security** — solar-gold verified structures are introduced.
5. **Opening channels** — I/O and network pathways illuminate.
6. **Personalizing** — machine-specific morphology stabilizes from the persistent visual seed.
7. **Finalizing** — field geometry converges and the progress halo closes.
8. **Complete** — the chamber dissolves into the first-boot Germination lineage.

The current exact inoculation renderer represents this with a projected cylindrical field, orbital seals, module seeds, and a progress halo over the organic topology. v0.3.2 additionally gives Web Portal, USB Forge, WSL2 Pivot, Asahi Handshake, LAN Inoculation, and Local Direct installation distinct bounded route signatures while preserving the same factual phase model.

## Lifecycle differentiation

Different machine histories should alter composition rather than merely changing labels:

- clean return → balanced mature network
- update → new solar-gold growth ring
- rollback → candidate branch retraction + known-good restoration
- interrupted boot → asymmetry followed by repair
- suspend → topology remains, illumination returns
- hibernate → cooler projected field and thaw-like reconnection
- hardware change → new budding region
- mesh return → distant cyan/green links reconnect
- long healthy uptime → greater maturity and anastomosis

## Text policy

The framebuffer visual system should not depend on text to explain itself. Where text is eventually added, it should be sparse and factual:

- `SPORE`
- `GERMINATION`
- `INOCULATION`
- `SYSTEM INITIALIZING`
- factual phase names

Claims about machine consciousness, emotion, awakening, trauma, or sentience do not belong in the boot/install UI.

## Performance and safety constraints

- CPU-only path remains available and is the canonical early-boot renderer.
- Holographic effects are bounded analytic geometry, not unbounded particles.
- Bloom works from a reusable 1/16-pixel-count workspace rather than a full-resolution multi-pass GPU pipeline.
- No shader compiler, window system, cognitive runtime, network dependency, or external font stack is required for boot.
- The display manager can terminate the renderer at any instant.
- Exact preview uses the same presentation wrapper as the live DRM path.
- CI must render boot lifecycle, shared inoculation phases, and install-route ceremony galleries from exact pixels before physical-host enablement.

## Future visual tranches

After v0.3.2 is verified, candidate additions are:

- signed-distance-field microtype for sparse factual labels
- double-buffered DRM page flips for cleaner motion
- temporal supersampling / inexpensive anti-aliasing
- curved field ribbons derived from topology density
- shutdown-to-next-boot morphology continuity
- compositor/greeter continuation of the same visual genome
- accessibility profiles for reduced motion and reduced bloom

Every addition should pass the same test: **does this reveal the state and identity of the system more beautifully, or is it merely decoration?**
