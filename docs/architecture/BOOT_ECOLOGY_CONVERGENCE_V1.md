# Spore Boot Ecology Convergence v1

This document defines how the typed boot-observability work and Spore Boot Ecology converge without creating two competing sources of boot truth.

## One-way authority graph

The architecture is deliberately one-way:

```text
Linux / kernel / systemd / NixOS
              |
              v
      authoritative observer
              |
              v
   symthaea-boot-protocol v1
              |
      +-------+--------+
      |                |
      v                v
 diagnostics UI   boot-ecology adapter
                       |
                       v
                 BootGenome / RenderPlan
                       |
                +------+------+
                |             |
                v             v
            preview       DRM/KMS renderer
```

No arrow is allowed to point upward. Renderers, genomes, visual lineage, receipts used for aesthetics, and diagnostics presentation never decide whether a service succeeded, a generation is good, or the machine is bootable.

## Protocol versus ecology

`symthaea-boot-protocol` owns only normalized present-boot observations:

- boot phase;
- bounded boot domains;
- domain state and aggregate health;
- monotonic sequence and elapsed time;
- bounded presentation-safe hints;
- transport envelope limits.

The protocol does **not** own visual concepts such as morphology, palette, repair scars, bloom, rings, spore count, animation families, or aesthetic maturity.

Spore Boot Ecology owns artistic interpretation:

- deterministic machine visual identity;
- morphology family selection;
- update rings and rollback retraction;
- repair/germination/relighting visual grammar;
- bounded persistent visual lineage;
- presentation profiles and renderer fidelity.

The ecology must consume protocol state through an explicit adapter. It must not parse journal text, inspect systemd units directly, or infer authoritative health independently.

## Historical lifecycle receipts

Historical lifecycle state is distinct from live observation.

A bounded boot-history receipt may contain coarse factual fields required to compose the next boot, such as previous termination class, previous uptime bucket, Nix generation transition, coarse hardware-topology digest, and whether the previous candidate was blessed.

Historical receipts:

1. are not transported as arbitrary live protocol details;
2. contain no user content, filenames, journal text, process names, SSIDs, peer identities, or secrets;
3. are schema-versioned and bounded;
4. may influence visual composition but never override Linux/systemd/NixOS truth;
5. may be deleted without affecting bootability.

## Health and Last Known Good

Boot presentation and generation blessing remain separate concerns.

- The protocol may report normalized readiness/health from an authoritative observer.
- A host-specific health gate decides whether the active NixOS generation qualifies for Last Known Good.
- Boot Ecology may visualize the candidate, blessing, or rollback transition only after receiving the factual result.
- Successful rendering is never part of the health predicate.

## Transport contract

The planned v1 transport is local Unix datagram.

Receivers must:

1. reject a datagram larger than `MAX_WIRE_BYTES` before deserialization;
2. reject unsupported protocol versions;
3. validate bounded details and snapshots;
4. ignore duplicate/older sequence numbers;
5. reject elapsed-time regressions within one observation lineage;
6. treat transport loss as presentation loss only.

Snapshot replacement is authoritative only after validation and may never move a reducer backward in time.

## Integration with Boot Ecology v0.3.2

The v0.3.2 Boot Ecology PR should remain qualification-frozen while its renderer, exact galleries, declarative package, and host fail-open tests qualify.

After qualification, convergence happens as a separate, revertible patch set:

1. add a small adapter from `BootSnapshot`/`BootEvent` to the factual state inputs consumed by `symthaea-boot-ecology`;
2. remove duplicated live-observation enums only after parity tests exist;
3. retain the existing historical `BootStateReceipt` fields that are genuinely lifecycle/history rather than live protocol state;
4. prove identical visual plans for equivalent old/new factual inputs with golden deterministic tests;
5. keep preview and DRM on the same `RenderPlan` path;
6. do not enable the physical host as part of the convergence PR.

## Qualification invariant

For every future boot-experience patch, the strongest availability property is:

> A machine with a broken or absent Spore presentation stack must be no less bootable than the same machine with Spore completely removed.

VM fault injection, physical canaries, and code review should be designed around proving that property rather than merely proving that the renderer works.
