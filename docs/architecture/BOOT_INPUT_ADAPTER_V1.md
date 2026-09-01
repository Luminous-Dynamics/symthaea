# Spore Boot Input Adapter v1

Status: draft, not wired into hardware boot.

## Purpose

Translate a deliberately tiny subset of Linux input events into typed presentation requests while guaranteeing that keyboard observation cannot persist into the logged-in desktop.

## Lifetime model

`BootInputAdapter` is a library, not a daemon. Its intended owner is the early `quicken-fb` renderer (or an equally short-lived boot coordinator). When that process exits, all evdev file descriptors are dropped.

This is deliberate. A persistent privileged keyboard service would have visibility into passwords, messages, terminal commands, and other post-login user content even if its code promised to ignore them. The safer design is to make continued observation structurally impossible by lifetime.

## Recognized input

Only key-down (`value == 1`) events for these keys are translated:

```text
F1  -> Ambient
F2  -> Diagnostics
Esc -> RawLogs request
```

Releases, repeats, and every unrelated key/event are discarded immediately. Device names are not retained or logged. The adapter never calls `grab()`, so it does not suppress normal kernel/console input handling.

## Device discovery

The adapter uses `evdev::enumerate()` and retains only devices that advertise all three control keys. Devices are opened nonblocking and rescanned periodically so keyboard hotplug does not require a long-lived hotplug daemon.

## Authority boundary

The adapter emits `PresentationRequest` only. It cannot:

- switch VTs;
- access journald;
- manipulate DRM/KMS;
- change systemd state;
- declare boot health;
- invoke shell commands;
- authenticate a recovery action.

`RawLogs` remains a request consumed by the separately qualified presentation/VT coordinator.

## Resource bounds

At most 16 recognized requests are emitted from one poll call. This prevents a malfunctioning input device from producing unbounded allocation/work in a single renderer frame. Key-repeat events are ignored, further limiting accidental request storms.

## Qualification

Before wiring to `quicken-fb`:

- compile/test against the pinned evdev API;
- verify unrelated synthetic keys map to no request;
- verify release/repeat values map to no request;
- verify the adapter never grabs a device;
- exercise a uinput-backed integration test in a privileged CI/VM lane if available;
- prove the adapter lifetime ends before any login credential input can occur;
- no physical-host enablement until the raw-VT coordinator has its acknowledgement/fail-open design.
