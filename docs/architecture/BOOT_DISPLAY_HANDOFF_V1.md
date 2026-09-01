# Boot Display Handoff v1

Status: draft qualification contract

## Purpose

The Spore boot renderer is presentation only. It must release DRM/KMS promptly before any later display owner (display manager, compositor, recovery UI, or raw-log VT) needs the device, without becoming an authority over login or boot completion.

## Renderer-side release boundary

On a systemd/user-requested fast handoff, the renderer:

1. stops advancing the visual simulation;
2. writes one deterministic black frame;
3. drops `DrmFramebuffer`;
4. `DrmFramebuffer::Drop` restores the saved CRTC and destroys renderer framebuffer resources;
5. only after Drop returns, optionally writes `boot-display-released-v1.json` atomically;
6. exits.

The receipt is evidence that the renderer reached its post-Drop code path. It is **not** sufficient authorization for another component to assume ownership of the device. A future coordinator must also establish that the renderer process/unit is no longer capable of drawing.

## Fail-open rule

The renderer is decorative. Its stop timeout is bounded by systemd. Failure to produce a receipt, failure to serialize it, receipt corruption, or renderer failure must never prevent login, recovery, or shutdown.

`TimeoutStopSec` belongs to the service section and is deliberately short. The current v1 default budget is 1000 ms while physical and VM measurements are collected.

## Separate ceremonies

Installation completion and session handoff are intentionally different paths:

- **Installation completion** may use the longer contraction/flash/fade ceremony.
- **Display handoff** uses the bounded fast-release path and must not spend seconds animating while the login compositor is waiting.

## Receipt schema

```json
{
  "version": 1,
  "renderer_pid": 1234,
  "release_us": 420,
  "renderer_uptime_us": 2100000,
  "reason": "signal"
}
```

The receipt contains no user data, host name, serial number, process list, journal content, command line, or hardware identifier.

## Systemd topology is not frozen yet

Do **not** add `Conflicts=display-manager.service` blindly. `Conflicts=` is a negative requirement dependency; when conflicting units are pulled into the same start transaction, systemd may remove one of the non-required jobs. We must qualify the exact trigger topology in a VM before replacing the historical `StopWhenUnneeded` wiring.

Likewise, `Before=display-manager.service` on a `Type=simple` renderer does not mean the renderer remains active until the display manager is ready. It only orders service startup. The next gate must be triggered at the point where the display manager is actually ready to claim the display.

## Qualification gates for the coordinator

The future coordinator may become enabled only after tests prove:

- renderer present / normal release;
- renderer absent;
- renderer already exited;
- renderer hung;
- missing/corrupt receipt;
- DRM open failure;
- display manager restart;
- recovery/rescue boot;
- headless system;
- raw-log VT request;
- bounded timeout fallback;
- no cycle in the systemd transaction graph.

## Performance quantities

Measure separately:

- signal/request to renderer loop observation;
- final-frame time;
- `DrmFramebuffer` Drop/release time;
- receipt write time;
- renderer process exit time;
- later display-owner start latency.

Do not optimize based solely on total boot time: the handoff budget needs its own distribution (median, p95, p99/max).
