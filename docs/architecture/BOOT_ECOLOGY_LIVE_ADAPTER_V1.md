# Boot Ecology Live Adapter v1

Status: convergence contract; implementation follows #238/#257 qualification

## Purpose

Combine the stable, deterministic visual identity of Spore Boot Ecology with authoritative live boot observations without creating a second visual engine or allowing presentation state to become boot truth.

The core model is:

```text
historical lifecycle facts -> BootGenome / stable topology
live BootSnapshot/Event    -> LiveEcologyModulation
                                      |
                                      v
                         one exact EcologyRenderer
```

The BootGenome is chosen once for a renderer invocation. Live system changes modulate that genome; they do not continuously regenerate its morphology.

## Stable genome, dynamic modulation

A boot may begin with a genome selected from factual history such as first boot, clean return, rollback, recovery, hardware change, or resume. Once rendering starts, live telemetry must not change the morphology family, deterministic seed, or persistent visual lineage merely because a transient service becomes delayed or recovers.

Otherwise a two-second network delay could make the organism appear to change species mid-boot and deterministic preview/live parity would become difficult to reason about.

Live state is therefore represented by a small derived modulation layer.

A future ecology-facing type should contain only bounded presentation facts such as:

```text
observation sequence
semantic boot anchor
coarse health
stalled/degraded domain classes
reveal floor
pulse request
repair/degraded emphasis
handoff readiness
```

It must not contain arbitrary journal strings, process metadata, unit dumps, network identifiers, paths, or authority decisions.

## Semantic anchors

Do not map Linux boot directly to an alleged percentage. Boot is a dependency graph, not a linear progress bar.

Map normalized boot facts to monotonic semantic anchors instead:

```text
BootPhase::Kernel      -> KernelActive
BootPhase::Initrd      -> InitrdActive
BootPhase::Storage     -> StorageAvailable
BootPhase::Filesystems -> FilesAvailable
BootPhase::Security    -> SecurityReady
BootPhase::Network     -> NetworkAvailable
BootPhase::Services    -> ServicesAvailable
BootPhase::Graphics    -> GraphicsAvailable
BootPhase::Session     -> SessionStarting
BootPhase::Ready       -> SessionReady
```

The exact renderer/RenderPlan assigns a visual location to each supported anchor. A scene may omit an anchor visually, but it may not invent a more advanced factual anchor.

## Elastic visual timeline

The current Ecology renderer can render deterministic state at an absolute sequence time. Convergence should preserve deterministic topology while replacing a purely wall-clock reveal policy with an elastic semantic clock.

Conceptually:

```text
historical genome -> visual plan
live boot anchor  -> minimum truthful visual phase
renderer time     -> interpolation toward that phase
```

Rules:

1. factual progress may pull the visual **forward**;
2. a visual timer may never claim a factual anchor not yet observed;
3. visual semantic progress does not regress within one observation lineage;
4. `Unknown` never renders as `Normal`;
5. `Ready` comes only from authoritative boot observation;
6. loss of telemetry freezes/falls back presentation; it never manufactures progress;
7. handoff may compress or skip remaining decoration rather than delay the display manager.

## Fast boot behavior

When the OS advances faster than the nominal visual sequence, the renderer should smoothly catch up rather than forcing the machine to wait for every planned stage.

Example:

```text
StorageAvailable
      |
40 ms later
      v
ServicesAvailable
      |
60 ms later
      v
GraphicsAvailable
```

The ecology can compress intermediate reveal while preserving topology and semantic ordering. At display handoff, remaining purely decorative stages may be skipped or shortened to the bounded handoff transition.

The renderer never extends boot to finish a movie.

## Slow boot behavior

When the OS remains at one anchor longer than the normal visual transition:

```text
NetworkAvailable not yet observed
```

Spore should not continue marching toward SessionReady.

Instead the scene enters a bounded ambient hold at the current truthful phase:

- already-grown structures breathe/pulse subtly;
- no new factual anchor is implied;
- ordinary delays remain calm;
- statistically unusual delay may increase diagnostic prominence;
- degraded/failed health may modulate color/activity without replacing explicit diagnostics.

This avoids both a frozen-looking splash screen and a dishonest progress bar.

## Progress elasticity

The renderer should maintain two distinct quantities:

```text
semantic_floor  = minimum phase justified by observed OS state
visual_phase    = smoothly rendered phase
```

`visual_phase` may approach the semantic floor under a bounded catch-up rate. Decorative interpolation between already-earned anchors is allowed, but it must stop before the next unearned factual anchor.

The final implementation should use fixed-point/integer semantic phase for state/replay and floating point only for local rendering interpolation.

## Event accents

Events may request ephemeral accents without mutating the stable genome.

Examples:

```text
DomainReady       -> one outward pulse
DomainRecovered   -> restrained repair pulse
DomainDelayed     -> slower local activity
DomainDegraded    -> bounded amber/repair emphasis
DomainFailed      -> diagnostic emphasis
BootReady         -> handoff-ready illumination
```

Event accents are rate-limited and presentation-only. A storm of service events must not create unbounded GPU/CPU work or visual noise.

## Health semantics

Suggested visual interpretation:

```text
Normal   -> ordinary ecology
Unknown  -> neutral/uncertain; never green-success semantics
Delayed  -> calm hold + subtle diagnostic cue
Degraded -> visible structured diagnostics + restrained ecology modulation
Failed   -> diagnostics dominate; ecology becomes secondary
```

Raw logs remain a separate Linux-native visibility layer.

## Diagnostics and ecology use the same source

Do not let the beautiful renderer and diagnostics UI infer state independently.

Both consume the same validated `BootSnapshot` reducer output:

```text
BootSnapshot
    |
    +--> structured diagnostics
    |
    +--> BootEcologyLiveAdapter -> LiveEcologyModulation
```

This prevents contradictions such as the diagnostics saying `Network: Failed` while the visual layer independently concludes that networking is healthy.

## Historical versus live facts

Historical `BootStateReceipt` and live `BootSnapshot` serve different purposes.

Historical facts may choose:

- morphology family;
- persistent repair marks;
- update rings;
- rollback/recovery grammar;
- visual maturity;
- deterministic machine visual identity.

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

The display manager never waits for a long visual conclusion.

## Deterministic replay

A visual incident should be reproducible from:

```text
BootGenome
+ ordered validated BootSnapshot/BootEvent trace
+ fixed semantic clock rules
+ renderer version
```

Exact semantic state should replay deterministically. Pixel output across different GPU/driver/backend implementations needs only perceptual equivalence unless using the exact CPU renderer/evidence path.

## Performance

Live modulation must be O(1) or O(number of bounded boot domains), with no journal scanning and no unbounded event history.

The adapter should retain only:

- current validated snapshot;
- last accepted sequence/lineage;
- bounded transient accent state;
- current semantic/visual phase.

It must not turn telemetry volume into renderer complexity.

## Qualification gates

Before replacing the current time-only live ecology schedule:

1. golden mapping tests for every `BootPhase` and `BootHealth`;
2. `Unknown`/telemetry-loss tests;
3. monotonic semantic-phase tests;
4. fast-boot compression tests;
5. slow-boot hold tests;
6. recovery/degraded event tests;
7. event-storm rate-limit test;
8. exact preview/live RenderPlan parity where inputs are equivalent;
9. handoff during every anchor;
10. performance evidence proving modulation overhead is negligible relative to raster cost.

## Invariant

> Spore may make truthful progress beautiful; it may never make beautiful progress authoritative.
