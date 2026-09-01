# Spore Boot Renderer Protocol Adapter v1

Status: draft implementation contract.

## Purpose

Connect `symthaea-quicken-fb` to `symthaea-boot-protocol` without making the framebuffer renderer part of Linux/NixOS boot authority and without removing the existing installer progress FIFO.

## Data flow

```text
systemd / NixOS
    |
    v
symthaea-boot-observer
    |  lineage-bound WireMessage
    +------------------------------> /run/symthaea/boot-events.sock
    |
    +-- atomic Snapshot -----------> /run/symthaea-boot/state-v1.json
                                         |
                                         v
                                  quicken-fb receiver
                                         |
                                  WireStateReducer
                                         |
                                 BootVisualState
                                         |
                                  mycelial renderer
```

No arrow may point upward from the renderer into systemd, boot health, generation blessing, authentication, recovery, or observer state.

## Lineage reset rule

An event can never establish a new observation lineage. If the live socket reports `AwaitingSnapshot` or `ForeignObservation`, the renderer may reset only from the independently selected snapshot side channel, and only when that validated snapshot carries the same `ObservationId` as the message being considered.

This keeps a delayed old datagram from sequence-poisoning a restarted observer and keeps an event-only packet from silently acquiring presentation authority.

## Wire safety

The receiver allocates `MAX_WIRE_BYTES + 1` bytes and rejects an oversized datagram before JSON decoding. Unsupported versions, malformed JSON, protocol validation errors, stale messages, missing state files, and unavailable sockets affect presentation only.

## Socket trust boundary

The renderer binds `/run/symthaea/boot-events.sock` beneath a root-owned `0770 root:symthaea-boot` runtime directory with service umask `0007`. The hardened observer retains `DynamicUser` and receives only the static `symthaea-boot` supplementary group required to write the socket.

The receiver probes an existing Unix socket before unlinking it. A live endpoint returns `AddrInUse`; only a stale socket endpoint is removed. Non-socket files and symlinks at the configured path are never deleted by the receiver.

## Visual semantics

The typed boot snapshot currently contributes two presentation effects only:

1. semantic state changes trigger a mycelial pulse;
2. phase and aggregate health define a minimum growth rate.

Installer I/O remains independent and combines with boot telemetry using `max(installer_io_rate, boot_growth_floor)`. `BootReady` does not terminate the renderer and the renderer cannot declare boot completion.

The health multiplier deliberately makes degraded/failed states less exuberant without introducing a false green/healthy state. A later diagnostics-overlay tranche will provide explicit text/status presentation.

## Non-goals of this tranche

- no raw journal rendering;
- no F1/F2/Esc VT controller;
- no delay-learning policy;
- no historical boot receipts;
- no desktop/wgpu handoff;
- no compositor ownership;
- no change to authoritative boot/recovery behavior.

## Qualification gates

Before leaving draft:

```text
cargo check -p symthaea-boot-protocol --all-targets
cargo test -p symthaea-boot-protocol
cargo check -p symthaea-boot-observer --all-targets
cargo test -p symthaea-boot-observer
cargo check -p symthaea-quicken-fb --all-targets
cargo test -p symthaea-quicken-fb
cargo clippy -p symthaea-quicken-fb --all-targets -- -D warnings
```

Nix evaluation/VM qualification must additionally prove:

- animation boots with telemetry absent;
- observer boots with renderer absent;
- renderer loads a late snapshot after observer starts first;
- observer restart changes observation lineage without accepting old queued packets;
- a malformed/oversized packet does not terminate the renderer;
- a non-socket file at the event path is not deleted;
- no physical-host enablement occurs from this PR.
