# Input Injection Validation Runbook

Companion to `capture-validation-runbook.md`, same purpose: track
real-hardware validation status for `xenia-inject`'s backends, one row
per OS/compositor, updated only with actual measured results.

## Harness

`xenia-peer/crates/xenia-inject/examples/inject_bench.rs`
(`cargo run -p xenia-inject --features xdg-portal --example inject_bench`)
constructs the backend under test (which blocks on the operator clicking
through any consent dialog), sends a handful of real pointer moves, a
click, a keypress, and a touch sequence, and reports pass/fail per call.
There's no capture loop to verify events landed anywhere specific on
screen — the harness proves the portal/backend session negotiation and
each `Notify*` call actually succeed against a live compositor, not just
compile.

## Status

| Task | Status |
|---|---|
| Harness shipped | ✅ `xenia-peer/crates/xenia-inject/examples/inject_bench.rs` |
| KDE-Wayland (`xdg-portal` / RemoteDesktop) | ✅ **first real run 2026-07-02** — 8/8 checks passed, `VERDICT: PASS` (see below) |
| GNOME-Wayland (`xdg-portal` / RemoteDesktop) | ⬜ pending an operator (same backend, should work identically per portal spec, unmeasured) |
| wlroots (`wayland-virtual`) | ⬜ not implemented (scaffold only, `WaylandInputInjector` in `lib.rs:340-396`) |
| `uinput` | ⬜ not implemented (scaffold only, `UinputInjector` in `lib.rs:408-464`); needs `/dev/uinput` access this account likely doesn't have |

### KDE-Wayland results (2026-07-02, `XdgPortalInjector` via `ashpd` 0.9.3)

All 8 checks passed on the first run: pointer move to two positions,
button down/up, key press/release, and a touch down→motion→up sequence.
Session negotiation (`CreateSession` → `SelectDevices` → `Start`) and
every `NotifyPointerMotion`/`NotifyPointerButton`/`NotifyKeyboardKeycode`/
`NotifyTouchDown`/`NotifyTouchMotion`/`NotifyTouchUp` call succeeded.

**Resolved 2026-07-02 (second run):** the operator watched the screen
during a repeat run and confirmed a real consent dialog does appear —
the session genuinely is gated on an interactive Allow/Deny prompt, not
silently granted. The dialog is just fast to click through (the whole
harness, including the click, completes in well under the operator's
perception of a "long pause"), which is why the first run's timing alone
looked ambiguous from the process side. `journalctl`'s `"MegaAuth: Failed
to lookup permissions"` / `"Only stream input"` lines were a red herring —
they describe KDE's internal permission-cache lookup finding no prior
grant (expected on a fresh session, not evidence of a skipped prompt).
No further action needed here; do keep this in mind for future sessions
though — timing alone (a run "completing too fast") is not reliable
evidence that a consent step was skipped without directly watching the
screen or authoritatively inspecting portal internals.

### `ashpd` vs. `scap`'s hand-rolled `dbus` approach

`xenia-capture`'s `scap` dependency hand-rolls D-Bus request/response
matching (`portal.rs::handle_req_response`) and has a real, confirmed
race there (registers the signal match *after* sending the method call —
see `capture-validation-runbook.md`'s `create_session` panic writeup).
`XdgPortalInjector` uses `ashpd` instead specifically to avoid
reintroducing that bug class in new code; no equivalent race was hit
during this validation, consistent with `ashpd` being a more mature
abstraction over the same underlying D-Bus request/response pattern.

## Full end-to-end pipeline (all 3 layers, done 2026-07-02)

Per xenia-peer's `ROADMAP.md`, `XdgPortalInjector` was layer 1 of three
needed for input injection to work end-to-end from a real viewer. All
three are now done:

1. ✅ **The backend itself** — this runbook.
2. ✅ **Daemon receive-loop.** `apps/xenia-peer/src/main.rs` splits its
   transport post-handshake and runs a dedicated recv task that opens
   inbound envelopes via `LaneSession::open_input`, gates each decoded
   `InputEvent` through `M1RuntimeSession::allow_input_flow`, and hands
   it to the injector selected by `--input-backend {noop,log,xdg-portal}`
   (default `noop`).
3. ✅ **Viewer capture.** `apps/xenia-viewer/src/gui.rs` reads egui
   pointer motion/buttons and a keymap covering letters/digits/nav/
   F-keys, normalizes against the rendered image's actual on-screen
   rect, and sends `InputEvent`s to the daemon over a split outbound
   path alongside the existing frame-receive loop.

### Live end-to-end proof (2026-07-02, KDE-Wayland, `--input-backend log`)

Real `xenia-peer` daemon + real `xenia-viewer --gui` viewer, connected
over TCP, operator moved the mouse and typed inside the actual GUI
window. The daemon's `LoggingInjector` recorded:

- **1,280** pointer-motion events (`Pointer { pressed: false, .. }`),
  x/y varying continuously and staying within `[0.0, 1.0]`.
- **27** button-press/release events (`button: 0` = left, matching the
  `egui::PointerButton::Primary → 0` mapping).
- **202** key press/release events, evdev codes matching what was
  typed (e.g. `code: 30` = A, `code: 34` = G, `code: 37` = K).

This is the concrete proof the whole path — egui capture → seal →
transport → lane-envelope open → bincode decode → M1 consent gate →
inject — works for real, without needing the `xdg-portal` backend (and
its consent dialog) in the loop.

### Stretch validation: `--input-backend xdg-portal` in the same live loop (2026-07-02)

Same setup, `--input-backend xdg-portal` instead of `log`. The operator
moved the mouse inside the real `xenia-viewer --gui` window: a
RemoteDesktop consent dialog appeared (confirming the gate is live, not
a cached/skipped grant from the earlier `inject_bench` runs), the
operator approved it, and their **real host mouse cursor moved** in
response to the viewer's captured pointer motion. This is the complete
proof of the full pipeline through the real OS-level backend, not just
`LoggingInjector`.

### Real (non-bypassed) M1 consent ceremony (2026-07-02)

Every other live test this session used `--m1-preprod-auto-consent` to
skip the ceremony. Separately verified the real Approve/Deny path
(`--consent-port`, no bypass flag):

- **Deny**: daemon exited immediately; viewer received **0** frames.
- **Approve**: 15 frames streamed and byte-verified via the real
  ceremony (not the bypass flag), clean exit.

Confirms the M1 gate that protects both frame flow and input injection
is real end-to-end, not just exercised via the pre-production shortcut.

### Full-VM isolation test (2026-07-02)

Same-machine testing (above) means the operator's real cursor and the
injected cursor are literally the same pointer, which is awkward for
anything beyond a one-shot smoke test. Set up a real, reusable NixOS
VM (`/etc/nixos/hosts/xenia-test-vm`, KDE Plasma 6 + Wayland, SSH
enabled, `/srv/luminous-dynamics/xenia` shared read-only via 9p so the
already-built host binaries run as-is in the guest without a rebuild)
and ran the daemon (`--input-backend xdg-portal`) inside the VM while
the viewer stayed on the real host, connected over a forwarded
WebSocket port.

Result: real host cursor moved **only** from the operator's own hand;
the injected input only ever moved the VM's own cursor, visible inside
the QEMU window. Full isolation confirmed, with the strongest possible
guarantee (separate kernel, not just a separate compositor process).

Two caveats surfaced, both artifacts of this specific test setup, not
xenia-peer bugs:

- The daemon inside the VM used synthetic `TestCapture` (fixed
  320×200), while `XdgPortalInjector` injects into the VM's real
  ~1024×768 desktop -- these two are normally coupled (both derived
  from the same real screen), so a normalized viewer position landed
  at the wrong relative spot inside the VM. Fix for a coordinate-
  accurate test: pass `--width`/`--height` matching the VM's real
  resolution, or (more completely) build with the `scap` feature and
  use real capture inside the VM so the viewer shows the actual VM
  desktop and both sides agree on screen size.
- `--transport auto`'s advertisement-based QUIC upgrade doesn't work
  through the VM's single-port NAT forward (QEMU usermode networking
  only forwards the one port you declare); use `--transport ws`
  explicitly on both daemon and viewer for VM-based testing.

The VM config is a reusable asset going forward, not a one-off.

`WaylandInputInjector`/`UinputInjector` remain scaffold stubs — out of
scope for this pass.
