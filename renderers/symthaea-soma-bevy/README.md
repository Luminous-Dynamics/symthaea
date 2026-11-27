# Symthaea Soma Bevy renderer adapter

**Status:** transition-aware source/target implementation added; exact Bevy build and runtime qualification pending  
**Bevy:** pinned to `=0.20.1`  
**Scope:** isolated 2D renderer for immutable Symthaea Soma scene frames

## Isolation

This package defines its own Cargo workspace and lives under `renderers/`, outside the root workspace globs. It does not add Bevy to `symthaea-soma`, and it does not change the existing Bevy 0.15 dashboard or Bevy 0.18 Muse integration.

The dependency uses Bevy's `2d` feature profile only. The exact Bevy patch version is pinned in `Cargo.toml`; no `Cargo.lock` has been generated because Cargo is unavailable in the current execution environment. Reproducible transitive dependency resolution remains pending.

## Host integration

The host is responsible for the trusted policy and clock boundary:

1. Create and insert `PresentationPolicyState` using the active session's default-deny `PresentationPolicy`.
2. Create `SceneClock` and call `update(clock_domain_id, now_monotonic_ns)` before every Bevy `Update`, using the same monotonic clock domain as the canonical cue.
3. Compile a `CanonicalCue` with `compile_scene`, then run the receipt ledger's `record_authorized` step. Construct `AuthorizedSceneSubmission` from the scene frame, the exact refreshed permit returned by authorization, and the `Authorized` receipt.
4. Enqueue the submission in `SceneSubmissionQueue`.
5. Read `LatestSceneSubmissionAck`. `Accepted` means the adapter accepted and queued ECS render entities; it is not evidence that the compositor displayed pixels. Record the exact accepted permit as `Submitted` using the receipt ledger.
6. Integrate a qualified compositor/frame observer separately to record `Observed`; use an output-stop observation to record `Stopped`.

The host should serialize policy recheck/authorization, queue submission and receipt updates so that a concurrent pause or revocation cannot race the renderer handoff. This crate does not claim a lock-free or cross-process authorization protocol.

## Fail-closed behavior

- Requires a fresh host clock update in the current app update; missing/stale clock hides output.
- Rejects wrong clock domains and clock regression.
- Rechecks the opaque permit against the live policy before creating scene entities and rechecks both source and target permits on every update during a handoff.
- A changed policy result (including reduced-motion or intensity changes) invalidates the authorized frame; the host must reauthorize and compile a new frame.
- Same-form updates and cross-form transitions use separate scene groups, host-clock transition weights, and Bevy's explicit alpha-blend material mode.
- If the source loses authority but the target remains valid, the source is removed and the target continues by hard cut. If the target loses authority, both groups are removed; previous output is never restored.
- Superseding an unfinished transition uses a hard cut instead of an ambiguous three-scene overlap.
- Every rendered entity is removed when policy recheck fails or the effective-permit expiry is reached.
- Reduced-motion scenes contain only static elements.
- No camera/microphone access, network transmission, audio output, haptics or physical actuators are implemented.
- No receipt state is advanced to Observed or Stopped by this adapter.

## Run and qualification

From this directory, once a Rust toolchain and dependency access are available:

```sh
cargo test --locked
cargo check --locked
```

Generate and review the package lockfile first, then use the exact committed lockfile for subsequent checks. A future visual example should use synthetic canonical cues and a test-only consent grant; it must not enable sensors or physical output.

## Current tests

Fifteen Rust adapter tests are authored, including monotonic clock regression, exact expiry, cross-domain rejection, authorized-intensity clamping, zero-intensity output, animation caps, and transition alpha bounds. The main Soma event, projection, policy, receipt, scene, turn-control and transition modules contain 109 authored tests, for 124 authored Rust tests total. None has been compiled or executed. Current draft-PR GitHub Actions results must be read from the exact head and are not inferred from this document.
