# Spore Boot Ecology Convergence v2

Status: implementation contract; no physical-host enablement

## Goal

Converge the live boot-observability stack and Spore Boot Ecology into one boot experience without creating a second source of system truth, a second visual engine, or a second handoff authority.

The target architecture is:

```text
Linux / kernel / systemd / NixOS
              |
              v
      authoritative observer
              |
              v
   symthaea-boot-protocol v1
              |
      +-------+--------------------+
      |                            |
      v                            v
 diagnostics/presentation     ecology adapter
                                   |
                                   v
                        historical lifecycle facts
                              + live facts
                                   |
                                   v
                              BootGenome
                                   |
                              RenderPlan
                             /          \
                            v            v
                         preview       DRM/KMS
                                        |
                                        v
                              post-DRM release receipt
```

No arrow may point upward. Rendering success, visual maturity, morphology lineage, scene completion, handoff receipts, or diagnostics presentation never decide boot health, generation blessing, recovery eligibility, or login authority.

## One visual engine

Spore Boot Ecology v0.3.2 remains the exact visual engine.

The newer protocol/observer work must not grow a competing long-lived boot renderer. The legacy `MycelialNetwork` path may remain temporarily for compatibility and microbenchmark isolation, but it is not the product visual qualification path once convergence lands.

Canonical visual evidence comes from the exact ecology/fidelity renderer used by preview and live DRM/KMS.

## Ownership split

### `symthaea-boot-protocol`

Owns live, present-boot facts only:

- phase;
- bounded domains and states;
- aggregate health;
- observation lineage;
- monotonic sequence and elapsed time;
- bounded presentation-safe hints;
- wire-size/version validation.

It does not own morphology, colors, narrative stages, previous-boot history, update rings, repair scars, or artistic maturity.

### `symthaea-boot-ecology`

Owns artistic interpretation and bounded visual history:

- `BootStateReceipt` historical facts;
- `MorphologyLineage`;
- deterministic `BootGenome`;
- visual family/stages/parameters;
- accessibility/fidelity profiles;
- exact RenderPlan semantics.

It must not query systemd, parse journal text, inspect arbitrary `/proc`, or independently infer authoritative present-boot health after convergence.

### Host integration

Owns operational policy and recovery:

- actually booted Nix generation;
- Current/Previous/Last-Known-Good roots;
- display-manager stability gate;
- systemd service lifetime bounds;
- rescue/diagnostic/no-Spore paths;
- physical-host enablement.

The renderer is never part of the health predicate.

## Adapter contract

The convergence adapter is deliberately narrow. It accepts a validated protocol snapshot and produces only ecology-facing **live presentation facts**.

Suggested semantic mapping:

```text
BootPhase::Kernel       -> live stage: substrate/kernel
BootPhase::Initrd       -> live stage: substrate/initrd
BootPhase::Storage      -> live stage: storage
BootPhase::Filesystems  -> live stage: filesystem
BootPhase::Security     -> live stage: trust/security
BootPhase::Network      -> live stage: channels/network
BootPhase::Services     -> live stage: services
BootPhase::Graphics     -> live stage: graphics
BootPhase::Session      -> live stage: handoff/session
BootPhase::Ready        -> live stage: ready
```

Health maps independently:

```text
Unknown   -> unknown visual state; never render as healthy
Normal    -> ordinary ecology
Delayed   -> restrained delay cue
Degraded  -> ecology + diagnostic affordance
Failed    -> failure/repair cue + diagnostic affordance
```

The adapter must preserve `Unknown` rather than collapsing it into Normal.

### Historical vs live facts

Do not force historical lifecycle fields into the live protocol.

`BootStateReceipt` fields such as previous termination, generation transition, previous hardware fingerprint, previous uptime, and bounded morphology history remain historical inputs composed before/during boot.

Live protocol state overlays the current boot's factual progress and health.

A useful composition model is:

```text
HistoricalBootFacts
       +
ValidatedLiveBootSnapshot
       |
       v
EcologyCompositionInput
       |
       v
BootGenome / RenderPlan
```

The historical receipt may influence the chosen morphology. The live snapshot may influence current stage/intensity/diagnostic cue. Neither may manufacture system facts.

## Deterministic parity gate

Before removing any duplicated live-observation enum/path, create golden parity cases.

For every representative old factual input:

1. render/compose with the qualified v0.3.2 path;
2. convert equivalent facts through the new adapter;
3. compose the new path;
4. assert identical semantic `BootGenome`/RenderPlan where equivalence is intended;
5. explicitly document intentional differences.

At minimum cover:

- first boot;
- clean ordinary boot;
- resume/relight;
- updated generation;
- rollback/known-good restore;
- unclean power loss + repair;
- degraded storage;
- hardware change;
- mesh return;
- current-boot delayed domain;
- current-boot degraded domain;
- current-boot failed domain;
- unknown current health.

Pixel tests remain tolerance/golden evidence; semantic plan tests should be exact.

## Two-phase display handoff

The final handoff combines the existing host request marker with the newer post-DRM release receipt.

```text
display manager ExecStartPre
          |
          v
read renderer InvocationID
          |
          v
write RequestHandoff marker
          |
          v
renderer notices request
          |
          v
stop advancing scene
          |
          v
optional deterministic final frame
          |
          v
drop DrmFramebuffer
restore saved CRTC
          |
          v
write DisplayReleased receipt
  { InvocationID, PID, release timing }
          |
          v
renderer exits
          |
          v
systemctl stop returns
(or bounded TimeoutStopSec enforcement)
          |
          v
best-effort receipt correlation check
          |
          v
display manager proceeds regardless
```

### Authority rule

`systemctl stop` plus the service lifetime bound is the operational enforcement mechanism.

The receipt is evidence/correlation only. It must never become an unbounded prerequisite for login. A missing, corrupt, stale, mismatched, or unwritable receipt may generate diagnostics but must not hold the display manager.

### Invocation correlation

Use systemd's per-service-start `InvocationID` to correlate request and release evidence.

A receipt must only accept a canonical 32-hex InvocationID. PID alone is insufficient because PIDs are reusable. InvocationID is still not a security credential; it is a lifecycle correlation identifier.

The coordinator should consider a matching receipt meaningful only when all of the following agree:

- expected systemd unit;
- expected InvocationID;
- renderer process has exited / stop transaction completed;
- receipt schema is supported;
- receipt was created for the current handoff attempt.

Even then, the result is diagnostic evidence, not generation/boot authority.

## Performance evidence convergence

Keep distinct measurement tools for distinct questions.

### Exact ecology renderer cost

`spore_render_probe` from Boot Ecology is the canonical CPU visual-fidelity measurement because it exercises the exact organic + holographic + fidelity + identity path.

### Primitive microbenchmark

`spore-boot-bench` may continue measuring legacy mycelial simulation/raster primitives for regression localization. It must never be presented as the exact ecology renderer's headline performance result.

### Live renderer receipt

The live receipt owns measurements that the pure render probe cannot:

- DRM open;
- first completed live frame;
- exact live blit cost;
- frame deadline misses;
- post-DRM release;
- renderer uptime/lifetime;
- correlated systemd InvocationID.

### Whole-system evidence

`systemd-analyze` and selected monotonic unit properties own:

- kernel/initrd/userspace timing context;
- display-manager timing;
- graphical target timing;
- time-to-session comparisons.

The target remains: Spore ON should be statistically indistinguishable from Spore OFF in usable-session latency except for a small, explicitly bounded display ownership handoff.

## Qualification order

Do not merge convergence merely because the adapter compiles.

1. qualify Boot Ecology v0.3.2 exact visual/evidence gates;
2. qualify boot protocol/observer/control stack Q1;
3. qualify current host fail-open QEMU gates;
4. land convergence adapter with exact semantic parity tests;
5. make preview and DRM consume the same converged RenderPlan;
6. add two-phase InvocationID-correlated handoff evidence;
7. run VM failure matrix;
8. gather exact renderer and whole-boot measurements;
9. optimize measured bottlenecks only;
10. perform compatibility-preserving `quicken-fb` -> `spore-boot` rename;
11. physical `nixos-rebuild test` canary;
12. boot-only canary;
13. default enablement only after review/blessing.

## Deletion gate

Convergence is incomplete until duplicated authority/observation paths can be removed.

After parity qualification:

- Boot Ecology must not query or infer live system health independently;
- the legacy live-growth telemetry adapter may be removed or kept only behind compatibility mode;
- one exact visual RenderPlan path must feed preview and DRM;
- one live protocol must feed diagnostics and ecology;
- one host lifecycle policy must own handoff/recovery/LKG.

The end state should have fewer concepts and fewer independent state machines than the pre-convergence branches, not more.
