# symthaea-boot-protocol

A small, typed protocol for presenting Linux/NixOS boot state to Symthaea/Spore user interfaces.

The protocol is intentionally **not** a log transport. Linux/systemd remain authoritative and raw diagnostic detail remains in the journal. This crate carries bounded, normalized state such as boot phase, domain health, delay, degradation, and recovery.

## Design goals

- versioned messages;
- no unsafe code;
- deterministic serialization model;
- bounded optional diagnostic hints;
- no user content, SSIDs, arbitrary command lines, environment variables, or filesystem contents;
- renderer failure can never block boot;
- future transports can use the same types (Unix datagram is the planned v1 transport).

See `docs/architecture/BOOT_OBSERVABILITY_INVARIANTS_V1.md` for the frozen authority and recovery boundaries.
