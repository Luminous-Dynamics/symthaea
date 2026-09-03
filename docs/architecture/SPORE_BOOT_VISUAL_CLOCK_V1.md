# Spore Boot Visual Clock v1

Status: renderer-independent fixed-point clock implemented by `symthaea-boot-visual-clock`; exact Boot Ecology renderer integration follows qualification.

## Purpose

The boot visual must feel alive on both very fast and unusually slow machines without becoming a fake progress bar or delaying login.

The clock therefore separates:

```text
authoritative boot phase
        |
        v
LiveEcologyModulation
        |
        v
truth band
        |
        v
ElasticVisualClock
        |
        v
render phase
```

It has no systemd, DRM, login, recovery, or boot-success authority.

## Fixed-point phase

The visual phase is an integer in:

```text
0 .. 1_000_000
```

This is a renderer timeline coordinate, **not a percentage of Linux boot work**.

The same fixed-point scale is used by the live ecology reducer so semantic replay does not depend on floating-point serialization.

## Truth bands

Every factual semantic anchor owns a closed visual band:

```text
[factual floor, decorative ceiling]
```

The floor is the minimum visual location justified by the observed boot phase.

The decorative ceiling is the furthest the visual may advance without implying the next unobserved phase.

For adjacent anchors:

```text
current.ceiling < next.floor
```

except `SessionReady`, whose floor and ceiling are both the end of the sequence.

The exact numeric constants are presentation policy and may be tuned only with versioned tests/evidence. They must never be exposed to users as boot-completion percentages.

## Clock modes

The reference clock reports one of four modes per step:

```text
CatchUp
AmbientDrift
Hold
Complete
```

### CatchUp

Used when authoritative state has advanced farther than the current rendered phase.

The visual moves monotonically toward the new factual floor at a bounded catch-up velocity.

It cannot overshoot the floor during that step.

### AmbientDrift

After the factual floor is reached, a healthy/normal boot may move gently toward the current decorative ceiling.

This gives slow normal boots subtle life without claiming the next system phase.

### Hold

No decorative forward phase is permitted when state is:

```text
Unknown
Delayed
Degraded
Failed
```

or diagnostics require more than Ambient visibility.

The exact renderer may still animate bounded local breathing/pulses inside the already-rendered state; the semantic timeline itself does not advance.

### Complete

Reported only once the visual phase has reached the end of the `SessionReady` band.

This is presentation completion only. It grants no permission to start a display manager, authenticate a user, bless a boot, or block/continue host lifecycle.

## Fast boot

If Linux advances rapidly:

```text
StoragePhase -> ServicesPhase -> GraphicsPhase
```

before the visual catches up, the latest authoritative floor pulls the clock forward at the catch-up rate.

The renderer does not force Linux to wait for omitted decorative intervals.

At host handoff, remaining decoration may be skipped entirely.

## Slow normal boot

If the machine remains in one healthy phase:

```text
NetworkPhase
```

for longer than expected, the clock may drift only inside the Network truth band and then stops at its ceiling.

The scene can continue low-cost local animation without moving the semantic timeline farther.

## Slow unhealthy boot

When delayed/degraded/failed/unknown state is active:

```text
catch up to already-earned factual floor
        |
        v
Hold
```

There is no decorative timeline drift.

Diagnostics/health modulation take precedence over decorative progress.

## Scheduling-gap bound

A renderer that is paused or starved must not consume an entire visual sequence from one giant elapsed-time sample.

`VisualClockPolicy::max_step_ms` caps the elapsed time used by any one `advance_ms()` call.

The current default is presentation tuning, not a correctness requirement. Qualification may change it.

## No rewind

The clock is monotonic within one boot observation lineage.

A new validated observation lineage requires an explicit clock/reducer reset.

The clock never rewinds merely because a service recovers, diagnostics close, or a new frame arrives late.

## Accessibility

The initial clock is intentionally independent from animation quality/motion profiles.

When integrated with the exact renderer:

- reduced motion may lower or disable decorative drift while keeping factual catch-up;
- high contrast must not alter semantic timing;
- Calm quality may reduce local activity but cannot change what phase is truthfully earned.

Accessibility changes presentation, not system facts.

## Performance

The clock is scalar fixed-point arithmetic with no allocations, journal access, I/O, event history, or GPU work.

The exact renderer should measure its overhead separately, but it is expected to remain negligible relative to rasterization/blit cost.

## Qualification gates

The current pure-clock tests require:

1. every truth-band floor equals the reducer's reveal floor;
2. every non-final ceiling lies below the next anchor floor;
3. factual jumps catch up monotonically without overshoot;
4. healthy slow boots cannot drift beyond their current truth band;
5. Unknown/Delayed/Degraded/Failed state does not decoratively advance;
6. long scheduling gaps are bounded by policy;
7. Ready may reach visual completion without granting host authority.

Exact-renderer qualification later adds:

1. perceptual fast-boot compression;
2. slow-boot ambient hold quality;
3. reduced-motion behavior;
4. handoff at every clock mode;
5. CPU/raster overhead evidence;
6. preview/live RenderPlan parity.

## Invariant

> The visual clock may interpolate inside truth already earned. It may never interpolate into truth the operating system has not reported.
