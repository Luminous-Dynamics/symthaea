# Boot Ecology Live Adapter v1

Status: renderer-independent reducer implemented; exact #238 renderer integration follows qualification

## Purpose

Combine the stable, deterministic visual identity of Spore Boot Ecology with authoritative live boot observations without creating a second visual engine or allowing presentation state to become boot truth.

The core model is:

```text
historical lifecycle facts -> BootGenome / stable topology
live BootSnapshot          -> LiveEcologyReducer
                                      |
                                      v
                           LiveEcologyModulation
                                      |
                                      v
                         one exact EcologyRenderer
```

The BootGenome is chosen once for a renderer invocation. Live system changes modulate that genome; they do not continuously regenerate its morphology.

## Implemented semantic boundary

The renderer-independent reference reducer lives at:

```text
crates/domains/symthaea-boot-ecology-live
```

It depends only on `symthaea-boot-protocol` and carries no DRM, journal, systemd, framebuffer, or Boot Ecology renderer dependency.

Its bounded output contains:

```text
observation_sequence
semantic phase anchor
authoritative coarse health
fixed-point reveal floor
delayed-domain bitmask
degraded-domain bitmask
failed-domain bitmask
minimum diagnostics visibility
semantic accent + idempotency token
handoff-ready fact
```

A reducer instance belongs to one already-validated observation lineage. The caller must reset it explicitly when the boot-protocol receiver adopts a new lineage.

## Stable genome, dynamic modulation

A boot may begin with a genome selected from factual history such as first boot, clean return, rollback, recovery, hardware change, or resume. Once rendering starts, live telemetry must not change the morphology family, deterministic seed, or persistent visual lineage merely because a transient service becomes delayed or recovers.

Otherwise a two-second network delay could make the organism appear to change species mid-boot and deterministic preview/live parity would become difficult to reason about.

## Conservative semantic phase anchors

Do not map Linux boot directly to an alleged percentage. Boot is a dependency graph, not a linear progress bar.

The observer may advance `BootPhase` when a watched unit is `Starting` or `Ready`; therefore the visual adapter must not reinterpret phase entry as stronger readiness/availability semantics.

The reference mapping is deliberately neutral:

```text
BootPhase::Kernel      -> KernelPhase
BootPhase::Initrd      -> InitrdPhase
BootPhase::Storage     -> StoragePhase
BootPhase::Filesystems -> FilesystemsPhase
BootPhase::Security    -> SecurityPhase
BootPhase::Network     -> NetworkPhase
BootPhase::Services    -> ServicesPhase
BootPhase::Graphics    -> GraphicsPhase
BootPhase::Session     -> SessionPhase
BootPhase::Ready       -> SessionReady
```

Only explicit `BootPhase::Ready`, produced from the authoritative boot-ready path, carries readiness semantics.

A scene may omit an anchor visually, but it may not invent a more advanced factual anchor.

## Reveal floors are not percentages

The live reducer assigns monotonic fixed-point reveal floors in `[0, 1_000_000]` so the renderer has a deterministic ordering target.

These values are **presentation constants**, not estimates of completed Linux boot work and must never be displayed as a boot percentage.

They answer only:

> What minimum visual phase is justified by the most advanced authoritative phase observed so far?

## Elastic visual timeline

The current Ecology renderer can render deterministic state at an absolute sequence time. Convergence should preserve deterministic topology while replacing a purely wall-clock reveal policy with an elastic semantic clock.

Conceptually:

```text
historical genome -> visual plan
live phase anchor -> minimum truthful visual phase
renderer time     -> interpolation toward/within earned visual range
```

Rules:

1. factual progress may pull the visual **forward**;
2. a visual timer may never claim a factual anchor not yet observed;
3. semantic progress does not regress within one observation lineage;
4. `Unknown` remains distinct from `Normal`;
5. `Ready` comes only from authoritative boot observation;
6. loss of telemetry freezes/falls back presentation; it never manufactures progress;
7. handoff may compress or skip remaining decoration rather than delay the display manager.

## Fast boot behavior

When the OS advances faster than the nominal visual sequence, the renderer should smoothly catch up rather than forcing the machine to wait for every planned stage.

Example:

```text
StoragePhase
      |
40 ms later
      v
ServicesPhase
      |
60 ms later
      v
GraphicsPhase
```

The ecology can compress intermediate reveal while preserving topology and semantic ordering. At display handoff, remaining purely decorative stages may be skipped or shortened to the bounded handoff transition.

The renderer never extends boot to finish a movie.

## Slow boot behavior

When the OS remains at one phase longer than the normal visual transition, Spore should not continue marching toward `SessionReady`.

Instead the scene enters a bounded ambient hold at the last earned phase:

- already-grown structures breathe/pulse subtly;
- no stronger factual state is implied;
- ordinary delays remain calm;
- statistically unusual delay may increase diagnostic prominence;
- degraded/failed health may modulate color/activity without replacing explicit diagnostics.

This avoids both a frozen-looking splash screen and a dishonest progress bar.

## Health and diagnostics

The snapshot's `BootHealth` is copied unchanged into `LiveEcologyModulation`; the adapter cannot promote/bless it.

Suggested visual interpretation:

```text
Normal   -> ordinary ecology
Unknown  -> neutral/uncertain; never success-green semantics
Delayed  -> calm hold + status cue
Degraded -> structured diagnostics + restrained ecology modulation
Failed   -> diagnostics dominate; ecology becomes secondary
```

The adapter also applies a defensive **diagnostic floor** from bounded domain state:

```text
any Delayed domain          -> at least Status
any Degraded/Failed domain  -> Diagnostics
```

This affects only visibility. It does not rewrite authoritative global health. Thus an inconsistent future snapshot cannot be visually underreported even though the renderer still exposes the original health value for diagnosis.

Raw logs remain a separate Linux-native visibility layer.

## Domain modulation is bounded and severity-aware

Delayed, degraded, and failed domains are stored in three separate fixed bitmasks over the protocol's bounded `BootDomain` enum rather than unbounded vectors or event history.

Separating severity lets the renderer distinguish:

```text
Delayed  -> restrained hold / lower-frequency local activity
Degraded -> repair emphasis while normal operation remains possible
Failed   -> stronger failure emphasis subordinate to explicit diagnostics
```

without reparsing raw system state or inferring severity from global health.

Recovery clears transient masks on the next authoritative snapshot; it does not regenerate topology or rewrite persistent visual history.

## Semantic accent protocol

Transient visual accents are **not** driven by every telemetry packet.

`LiveEcologyModulation` carries:

```text
accent_token
accent = None | Progress | Delay | Degraded | Failed | Recovery | Ready
```

The token changes only when presentation-relevant semantics change. A renderer triggers an accent only when the token changes.

Examples:

```text
same phase/health/domains, newer sequence -> no new accent
phase advances                           -> Progress
new delay                                -> Delay
new degradation                          -> Degraded
new failure                              -> Failed
issue clears / health recovers           -> Recovery
explicit Ready                           -> Ready
```

New failure/degradation/delay semantics outrank ordinary progress accents so diagnostic meaning is not drowned out by decorative activity.

The first accepted snapshot establishes state without requiring a transient accent.

This prevents high-volume but semantically unchanged observation traffic from becoming animation noise or CPU/GPU work.

## Sequence integrity and idempotency

For one observation lineage:

```text
sequence < previous sequence
    -> reject rewind

sequence == previous sequence
+ same presentation semantics
    -> return exactly the cached modulation

sequence == previous sequence
+ changed presentation semantics
    -> reject equivocation
```

A duplicated datagram therefore cannot retrigger an accent, while the same sequence number cannot be reused to smuggle a different visual interpretation.

A new validated protocol observation lineage requires an explicit reducer reset.

## Recovery

Recovery does not regenerate the BootGenome and does not rewind the visual timeline.

Example:

```text
ServicesPhase + Network Degraded
        |
        v
repair emphasis + Diagnostics
        |
authoritative recovery snapshot
        v
ServicesPhase + Network Ready
        |
        v
same topology / same reveal floor
repair emphasis clears
Recovery accent once
diagnostics de-escalate according to authoritative health
```

The reference reducer has regression coverage for this behavior.

## Diagnostics and ecology use the same source

Do not let the beautiful renderer and diagnostics UI infer state independently.

Both consume the same validated `BootSnapshot` reducer output:

```text
BootSnapshot
    |
    +--> structured diagnostics
    |
    +--> LiveEcologyReducer -> LiveEcologyModulation
```

This prevents independent scraping/parsing paths from disagreeing.

## Historical versus live facts

Historical `BootStateReceipt` and live `BootSnapshot` serve different purposes.

Historical facts may choose:

- morphology family;
- persistent repair marks;
- update rings;
- rollback/recovery grammar;
- visual maturity;
- deterministic local visual identity.

Live facts may choose only transient presentation modulation and semantic progress.

A live failure must not rewrite persistent visual lineage as though the next boot history had already been committed. Host lifecycle policy records qualified historical outcomes later.

## Handoff

When the host requests display handoff:

1. stop accepting further decorative progression;
2. capture the final semantic continuity state;
3. perform only a bounded handoff visual if budget permits;
4. release DRM/restore CRTC;
5. emit post-release evidence;
6. exit.

`LiveEcologyModulation::handoff_ready` is true only for explicit `BootPhase::Ready`; it is presentation input, not permission to start the display manager.

The host's bounded stop/lifecycle policy remains authoritative.

## Deterministic replay

A visual incident should be reproducible from:

```text
BootGenome
+ ordered validated BootSnapshot trace
+ reducer version
+ fixed semantic clock rules
+ renderer version
```

The reducer test suite includes representative irregular traces and requires independent reducer instances to produce identical modulation sequences.

Exact semantic state should replay deterministically. Pixel output across different GPU/driver/backend implementations needs only perceptual equivalence unless using the exact CPU renderer/evidence path.

## Performance

Live modulation is O(number of bounded boot domains), with no journal scanning and no unbounded event history.

The adapter retains only bounded scalar/mask state for the current observation lineage:

- last accepted sequence;
- last presentation fingerprint;
- last modulation.

Its output uses fixed-size domain masks and scalar fields.

It must not turn telemetry volume into renderer complexity.

## Current reducer qualification gates

Implemented tests cover:

1. monotonic phase-anchor/reveal mapping;
2. conservative phase naming;
3. slow-boot hold;
4. meaningful-change-only accents;
5. explicit-Ready-only handoff fact;
6. health-driven diagnostics floor;
7. defensive domain-driven diagnostics visibility;
8. separate bounded delayed/degraded/failed masks;
9. recovery without topology/timeline rewind;
10. sequence/anchor rewind rejection;
11. same-sequence equivocation rejection;
12. explicit lineage reset behavior;
13. invalid snapshot rejection;
14. exact equal-sequence idempotency;
15. deterministic representative trace replay.

## Deferred exact-renderer gates

After #238/#257 qualify and the exact Ecology renderer is connected:

1. fast-boot compression;
2. bounded slow-boot ambient hold;
3. telemetry-loss rendering;
4. event-storm/rate-limit behavior;
5. exact preview/live RenderPlan parity where inputs are equivalent;
6. handoff during every semantic phase;
7. modulation overhead evidence relative to raster cost;
8. reduced-motion/high-contrast parity.

## Invariant

> Spore may make truthful progress beautiful; it may never make beautiful progress authoritative.
