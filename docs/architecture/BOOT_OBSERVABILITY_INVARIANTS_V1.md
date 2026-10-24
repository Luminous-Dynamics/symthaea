# Boot Observability Invariants v1

This document freezes the safety and UX boundaries for the Symthaea/Spore boot experience before richer diagnostics or ambient continuity are added.

## Authority boundary

1. Linux, systemd, the kernel, and NixOS remain authoritative for boot state.
2. Symthaea presentation components may observe normalized state but never determine whether the machine is healthy, bootable, authenticated, or recovered.
3. A successful animation is not evidence of a successful boot.
4. Persistent visual state is never promoted to system truth.

## Availability boundary

1. Failure of `symthaea-quicken-fb` must not prevent boot or login.
2. Failure of any future boot observer, input helper, receipt recorder, ambient handoff, or diagnostics renderer must not prevent boot or login.
3. A diagnostics boot path must remain available without any Symthaea presentation component.
4. Raw Linux/systemd diagnostics must remain accessible independently of the graphical boot renderer.

## Privacy boundary

1. Ambient consumers receive normalized boot domains and health states rather than raw journal lines.
2. The typed boot protocol must not carry user content, environment variables, arbitrary command lines, SSIDs, filesystem contents, or secrets.
3. Optional diagnostic details are bounded, single-line, presentation-safe hints; authoritative detail remains in the journal.
4. Input helpers, if introduced, recognize only explicit mode-control keys and must not log or forward ordinary typed input.

## Presentation model

The user-visible layers are:

- **Ambient**: the normal living boot animation.
- **Diagnostics**: structured human-readable boot state.
- **Raw**: the native Linux/systemd diagnostic surface.

Presentation may simplify state but must never contradict authoritative state. Unknown is rendered as unknown, never healthy.

## Failure escalation

- Normal boots remain ambient and quiet.
- Unusual delays may surface contextual state without abandoning the ambient presentation.
- Degraded boots may automatically open structured diagnostics.
- Critical failures must prioritize recovery and raw diagnostics over aesthetics.

## Reversibility and rollback

Every boot-experience PR must be independently revertible. The diagnostics boot path must not depend on the same code path as the ambient renderer.

## Protocol rule

The boot protocol is versioned. Receivers reject unsupported versions and malformed or over-budget messages. Unknown future fields must not cause the machine's boot path itself to fail.
