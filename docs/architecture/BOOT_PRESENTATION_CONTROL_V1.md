# Spore Boot Presentation Control v1

Status: draft protocol boundary.

## Goal

Make boot-detail visibility user-controllable without turning key presses into privileged boot authority.

The model has three presentation modes:

```text
Ambient < Diagnostics < RawLogs
```

`Ambient` is the normal living Spore scene. `Diagnostics` is the future structured, human-readable boot overlay. `RawLogs` means a request to hand display ownership to a genuine Linux log VT.

## Separation of concerns

```text
keyboard/input adapter
        |
        v
PresentationRequest
        |
        v
PresentationArbiter <--- automatic diagnostic floor
        |
        v
effective presentation mode
        |
        +--> renderer overlay
        |
        +--> future VT coordinator
```

The input adapter is not allowed to call `chvt`, manipulate DRM, restart services, or interpret system health. The arbiter is not allowed to access input devices, systemd, journald, or DRM.

## Automatic visibility rule

Automatic policy may raise the minimum mode from `Ambient` to `Diagnostics` when the boot is delayed/degraded/failed. It **cannot** automatically force `RawLogs`.

This gives the system permission to make trouble visible without unexpectedly dumping a normal user into a raw terminal. Raw-log mode must be an explicit user action or a separately qualified recovery path.

## User preference vs policy floor

The effective mode is the more revealing of:

- the latest monotonically sequenced user request; and
- the automatic policy floor.

Example:

```text
user requests Ambient
policy requires Diagnostics
=> Diagnostics

health recovers
policy returns to Ambient
=> Ambient
```

If the user explicitly chose Diagnostics, recovery does not silently overwrite that preference.

## Raw-log handoff

This PR does **not** implement VT switching. The later coordinator must perform a two-phase transition:

1. request renderer quiescence / DRM release and receive acknowledgement;
2. activate the dedicated log VT.

Returning to Ambient/Diagnostics reverses that sequence: activate the graphics VT, reacquire DRM, then restore the deterministic scene state.

No fixed sleep is an acceptable substitute for the acknowledgement boundary.

## Wire constraints

User presentation requests use a small versioned message with a 256-byte application ceiling and monotonic sequence. This protocol is about presentation only and carries no arbitrary strings, commands, unit names, file paths, or shell input.

Filesystem/socket permissions provide the local trust boundary; the wire format itself is not authentication.

## Key mapping target

The later input adapter should map key-down events only:

```text
F1  -> Ambient
F2  -> Diagnostics
Esc -> RawLogs
```

All other keyboard events must be discarded immediately and never logged or retained.

## Qualification gates

Before any input/VT implementation is enabled on hardware:

- stale requests cannot rewind user mode;
- policy cannot force RawLogs;
- automatic Diagnostics can override an Ambient request;
- user Diagnostics survives policy recovery;
- oversized/malformed control packets are rejected;
- no ordinary key data is recorded;
- input daemon failure leaves normal boot unaffected;
- VT coordinator failure has an independent recovery path;
- no component in this presentation stack can determine Linux boot health.
