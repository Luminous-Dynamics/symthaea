# Spore Boot Exact Render Adapter v1

Status: contract frozen on convergence branch; implementation against #238 follows qualification

## Purpose

Connect the renderer-neutral Spore presentation seam to Boot Ecology v0.3.2 without importing Linux/systemd state into the exact raster engine and without allowing decorative timing to enter the terminal `Handoff` stage early.

The target pipeline is:

```text
BootSnapshot
    |
    v
LiveEcologyReducer
    |
    v
LiveEcologyModulation
    |
    v
ElasticVisualClock
    |
    v
EcologyFrameInput
    |
    v
ExactEcologyAdapter
    |
    v
#238 EcologyRenderer::render_at(...)
```

`EcologyFrameInput` is the last boot-live semantic object visible to the exact renderer adapter. The renderer itself continues to own deterministic topology and pixel generation from `BootGenome`.

## Why naive normalized-time mapping is unsafe

Boot Ecology v0.3.2 derives `EcologyFrameState` from absolute renderer sequence time. Each genome ends with:

```text
...
Settle
Handoff
```

The duration of earlier stages varies by historical boot facts. Therefore mapping a generic normalized visual phase directly across `genome.visual_budget_ms()` could enter `BootStageKind::Handoff` before authoritative live state has reached the corresponding lifecycle boundary.

The adapter MUST partition the timeline so ordinary decorative interpolation cannot enter the terminal segment early.

## Canonical phase partition

The live semantic clock defines:

```text
SessionPhase truth ceiling = 975_000
SessionReady floor         = 1_000_000
```

Define:

```text
PRE_HANDOFF_PHASE_MAX = 975_000
REVEAL_SCALE          = 1_000_000
HANDOFF_PHASE_SPAN    = 25_000
```

For a qualified `BootGenome`, the adapter locates the final `BootStageKind::Handoff` stage and validates that it is the last stage.

Let:

```text
pre_handoff_ms = start time of final Handoff stage
handoff_ms     = duration of final Handoff stage
total_ms       = pre_handoff_ms + handoff_ms
```

## Ordinary projection before Ready

When `EcologyFrameInput.handoff_ready == false`, project only into the pre-handoff timeline:

```text
phase = min(frame.visual_phase, PRE_HANDOFF_PHASE_MAX)

elapsed_ms =
    round_down(phase * pre_handoff_ms / PRE_HANDOFF_PHASE_MAX)
```

The result MUST be `<= pre_handoff_ms`.

The current #238 `frame_state()` treats the exact boundary at the preceding stage end as that preceding stage at progress 1.0, so `elapsed_ms == pre_handoff_ms` does not enter Handoff.

The adapter must still assert this property in tests instead of depending silently on incidental renderer implementation.

## Projection after explicit live Ready

When `frame.handoff_ready == true`, the adapter may use the final semantic band for Handoff.

For `visual_phase <= PRE_HANDOFF_PHASE_MAX`, projection remains in the pre-handoff segment.

For `visual_phase > PRE_HANDOFF_PHASE_MAX`:

```text
handoff_phase = visual_phase - PRE_HANDOFF_PHASE_MAX

handoff_elapsed =
    round_down(handoff_phase * handoff_ms / HANDOFF_PHASE_SPAN)

elapsed_ms = pre_handoff_ms + min(handoff_elapsed, handoff_ms)
```

Only an explicit live Ready fact can therefore allow ordinary semantic-clock progression into the exact renderer's terminal Handoff stage.

## Host RequestHandoff is a separate authority

Live observation and display ownership are different authority domains.

An authoritative host `RequestHandoff` means the display manager is actually taking ownership. It may therefore request a bounded final Handoff visual even when live telemetry never produced `BootPhase::Ready`.

This exception MUST remain a separate host-control path. It must NOT mutate `EcologyFrameInput`, manufacture `BootHealth::Normal`, emit a celebratory Ready accent, or record a successful boot outcome.

The required behavior is:

```text
host RequestHandoff
      |
      +--> stop accepting decorative progression
      +--> optional bounded terminal Handoff rendering
      +--> restore CRTC / release DRM
      +--> post-release evidence
      +--> exit
```

If the remaining handoff budget is insufficient, skip or truncate the decorative Handoff stage. Never delay the display manager to finish it.

## Health and diagnostics remain independent of terminal stage

A machine may legitimately reach the display handoff boundary with:

```text
health = Unknown
health = Degraded
health = Failed
```

The exact renderer may still visually hand off because the lifecycle boundary is factual. It must not render positive-health celebration unless the live reducer emitted the separate `Ready` accent, which is restricted to known `BootHealth::Normal`.

Thus:

```text
handoff stage != healthy boot
Ready accent   = known-normal presentation only
```

## Stable genome remains immutable

The adapter does not regenerate `BootGenome` from live events.

Historical state continues to choose:

- morphology family;
- topology seed;
- repair grammar;
- rollback grammar;
- update rings;
- mesh grammar;
- maturity.

Live frame input controls only:

- elastic sequence time;
- bounded visual accents;
- delayed/degraded/failed emphasis;
- diagnostics visibility;
- terminal-stage eligibility.

## Semantic frame evidence

`symthaea-boot-presentation` produces a deterministic BLAKE3 digest for each validated `EcologyFrameInput` and a streaming order-sensitive trace digest.

These hashes are evidence/correlation only. They are not credentials or boot-success proofs.

An exact-render replay bundle should eventually bind:

```text
BootGenome digest
presentation trace digest
renderer version
render policy
resolution
final exact-pixel digest where applicable
```

This lets a visual incident be reproduced without retaining raw journal text or arbitrary process metadata.

## Required adapter tests

Before connecting live DRM, qualification must cover at least:

1. every #238 representative `BootGenome` has exactly one final Handoff stage;
2. no non-ready semantic frame projects into Handoff;
3. `visual_phase == 975_000` remains pre-handoff;
4. `visual_phase == 1_000_000` with Ready reaches the end of Handoff;
5. fast Ready from a visually early phase catches up without reordering stages;
6. slow healthy SessionPhase may settle but cannot enter Handoff;
7. Unknown/Delayed/Degraded/Failed SessionPhase cannot enter Handoff through ordinary timing;
8. explicit host RequestHandoff can terminate from every semantic phase without waiting for a movie;
9. degraded/unknown terminal handoff never emits known-normal Ready celebration;
10. semantic frame digest + exact render input mapping replay deterministically;
11. preview and live DRM use the same adapter;
12. adapter overhead remains negligible relative to exact raster cost.

## Invariants

> Decorative time may move only inside truth already earned.

> The terminal Handoff stage requires either explicit live Ready or an actual host display-ownership handoff request.

> A display handoff is not a boot-health verdict.

> The exact pixel renderer never becomes a source of operating-system truth.
