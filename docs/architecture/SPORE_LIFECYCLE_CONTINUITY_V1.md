# Spore Lifecycle Continuity v1

Status: architecture contract; implementation follows qualified boot convergence

## Product goal

The machine should feel like one coherent environment across lifecycle transitions without weakening Linux security, boot reliability, lock semantics, or recovery.

The visual experience may appear continuous while every trust boundary remains real.

```text
Firmware / initrd
      |
      v
Germination (early DRM boot)
      |
      v
First Breath (greeter handoff)
      |
      v
Ambient Desktop
      |
      +--> Focus / Gaming / Presentation
      |
      v
Secure Lock
      |
      v
Dormancy (suspend)
      |
      v
Reawakening (resume)
      |
      v
Ambient Desktop
      |
      v
Shutdown / next-boot lineage
```

## Core rule: semantic continuity, not process continuity

Do not keep one renderer/process alive across all lifecycle phases.

Boot, greeter, desktop, lock, and recovery have different privilege and security requirements. Each surface owns its renderer and reconstructs the visual world from a bounded semantic handoff.

The continuity payload describes **what the visual world means**, not how the prior process rendered it.

No framebuffer, screenshot, unlocked desktop pixels, arbitrary application state, journal content, environment, or privileged handles cross the handoff.

## Minimal continuity payload

A future versioned payload may contain only presentation-safe fields such as:

```text
schema_version
scene_digest
visual_seed
boot_genome_digest
morphology_family
normalized_scene_phase
world_age_ticks
lifecycle_transition
health_class
quality_profile
reduced_motion
semantic_milestones_digest
```

Optional extensions must remain bounded and capability-reviewed.

### Explicit exclusions

Never include:

- screen/framebuffer pixels;
- usernames or account identifiers;
- filenames or document titles;
- process lists;
- application command lines;
- SSIDs or peer identities;
- clipboard content;
- arbitrary journal text;
- microphone/camera data;
- security credentials;
- Nix store paths unless a separate public/content digest is sufficient;
- private system topology beyond normalized presentation-safe classes.

## Content-addressed scene identity

The desktop renderer should continue an exact scene package by digest rather than by mutable display name.

```text
scene_digest = hash(scene package + semantic ABI version)
```

If the desktop does not have the referenced scene digest, it falls back safely to a built-in Stillness/Mycelium scene. Missing visual content never blocks login.

## Boot -> greeter: First Breath

The early DRM renderer reaches its handoff boundary and writes a tiny semantic continuity record before releasing display ownership.

Ordering:

```text
Boot Ecology RenderPlan
        |
        v
freeze semantic scene state
        |
        v
write continuity payload atomically
        |
        v
request/release DRM handoff
        |
        v
early renderer exits
        |
        v
greeter starts
        |
        v
greeter renderer validates payload
        |
        +-- invalid/missing -> built-in safe scene
        |
        v
reconstruct scene at equivalent phase
```

The payload is presentation state only. Display-manager startup remains governed by systemd/host lifecycle policy, not by continuity success.

## Greeter -> authenticated desktop

Authentication is a security boundary. The login transition must never expose the previous user's desktop or retain greeter authority inside the session.

The greeter may transfer only the presentation-safe continuity payload to the newly created user session.

The user-session ambient renderer creates its own GPU/Wayland resources and continues the world semantically.

A visually smooth transition can be achieved with matching scene phase, camera framing, palette, seed, and timing without sharing protected surfaces.

## Desktop -> secure lock

The lock screen must be rendered by a security-qualified lock surface/compositor path.

Do not pass the unlocked framebuffer to a lock-scene process.

Instead:

```text
Ambient desktop
      |
      v
snapshot semantic state only
      |
      v
unlocked surface becomes unavailable
      |
      v
secure lock acknowledged
      |
      v
lock renderer reconstructs visual scene
```

The lock scene may look like the same ecosystem entering twilight/dormancy, but all private desktop pixels remain behind the compositor's lock boundary.

Unknown lock state must never be displayed as securely locked.

## Lock -> unlock

Unlock continues in reverse:

1. authentication succeeds through the normal secure path;
2. compositor releases the lock boundary;
3. user-session ambient renderer receives only semantic transition state;
4. the existing user world resumes/reconstructs;
5. the lock renderer exits.

No lock-screen process receives arbitrary desktop authority.

## Suspend: Dormancy

Suspend preparation should be extremely short and never veto suspend for aesthetics.

The ambient renderer may:

- stop spawning new decorative work;
- settle visible motion for a bounded interval;
- persist a semantic checkpoint;
- write the lifecycle transition `Dormancy`;
- release unnecessary GPU/audio resources.

If any step fails, suspend proceeds.

The scene checkpoint is cosmetic state only.

## Resume: Reawakening

On resume, do not simulate every missed render frame.

Advance the semantic world analytically:

```text
checkpoint_tick + monotonic elapsed duration -> current semantic tick
```

Examples:

- sun/sky position advances directly;
- slow growth advances from deterministic rates;
- particles are regenerated from current state rather than replaying overnight particles;
- weather/audio-reactive state resumes from current context, not stale input.

This gives continuity with effectively zero overnight rendering cost.

## Shutdown -> next boot

Shutdown may update bounded `MorphologyLineage` and write a factual lifecycle marker such as clean reboot/poweroff.

The next boot may reflect this history artistically.

Examples:

```text
clean reboot          -> ordinary return/germination
unclean termination   -> restrained repair grammar
new qualified Nix gen -> new growth ring
rollback              -> retraction/restored-known-good grammar
hardware change       -> bounded new bud
```

These are interpretations of factual state, never substitutes for it.

## Time model

Use a deterministic semantic clock distinct from render FPS.

```text
semantic tick -> world state
presentation time -> interpolation/rendering
```

Persistent world decisions use integer/fixed-point or otherwise explicitly deterministic state where practical. Rendering may use floating point and need only perceptual equivalence across GPU backends.

Never promise byte-identical pixels across different drivers/GPUs.

## Accessibility profile is part of continuity

Reduced-motion/high-contrast preferences should survive every lifecycle transition.

A boot animation must not be highly animated and then only honor reduced-motion after login.

Profiles should be available early enough to select:

- Calm;
- Standard;
- Rich.

`Calm` substantially reduces camera/particle/motion effects while preserving the same semantic information and diagnostic visibility.

## Failure behavior

Every continuity consumer validates version, size, digest formats, enum ranges, and bounded counters before use.

Failure matrix:

```text
missing payload       -> built-in safe scene
unsupported version   -> built-in safe scene
corrupt payload       -> built-in safe scene
missing scene digest  -> built-in safe scene
ambient renderer dies -> ordinary desktop remains usable
lock visual fails     -> secure lock remains authoritative
resume visual fails   -> session resumes without ambient effects
```

A visual failure may reduce beauty. It may never reduce access to login, recovery, raw diagnostics, or secure locking.

## Performance rule

Every surface must have a zero/near-zero-work state.

- early boot renderer ends before later display ownership;
- obscured desktop ambient rendering reaches zero where compositor visibility permits;
- screen off -> zero rendering;
- suspend -> zero rendering;
- static scene -> event-driven/no unnecessary frame production;
- lock scene obeys reduced-motion/power policy;
- resume performs analytical advancement rather than hidden-frame replay.

## Future desktop adapter

The future Rust ambient runtime should consume this continuity ABI rather than depend directly on the boot renderer implementation.

Recommended boundary:

```text
Boot Ecology
    |
continuity ABI
    |
Spore Ambient Core
    |
    +--> Plasma/KWin adapter
    +--> generic Wayland layer-shell adapter
    +--> future compositor adapters
```

This preserves the current strategic decision: build an experience layer on mature compositors first and replace compositor infrastructure only if later requirements prove it necessary.

## Qualification

Lifecycle continuity is qualified independently at every boundary:

- boot -> greeter;
- greeter -> user session;
- desktop -> lock;
- lock -> desktop;
- desktop -> suspend;
- resume -> desktop;
- shutdown -> next boot lineage.

For each boundary test:

1. normal transition;
2. missing producer;
3. corrupt payload;
4. consumer crash;
5. timeout;
6. stale payload;
7. unsupported version;
8. reduced-motion profile;
9. recovery/diagnostic path;
10. presentation disabled entirely.

The strongest invariant remains:

> Removing every Spore visual/continuity component must not make the machine less bootable, less recoverable, or less securely lockable.
