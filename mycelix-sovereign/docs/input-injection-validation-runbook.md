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

**Open finding, not resolved:** the whole run completed in under two
seconds with no visible pause for an interactive consent dialog, unlike
the capture (`ScreenCast.Start`) validation earlier the same day, which
clearly blocked on the operator's click. `journalctl` for
`plasma-xdg-desktop-portal-kde` around the run shows:

```
MegaAuth: Failed to lookup permissions: "No entry for remote-desktop"
Only stream input
```

This suggests KDE's portal implementation may take a lighter-weight (or
non-modal) path for input-only `RemoteDesktop` sessions specifically
(no paired `ScreenCast` stream), but this is not confirmed — the operator
was asked directly whether a dialog appeared and did not respond in time
to confirm either way. **Do not treat this as proof the session was
properly consent-gated** until someone watches the screen during a run
and confirms what actually happens. If it turns out no prompt appears at
all for unsandboxed native binaries requesting input-only access, that's
a real finding worth its own investigation (portal consent bypass for a
class of requests), not just a validation footnote.

### `ashpd` vs. `scap`'s hand-rolled `dbus` approach

`xenia-capture`'s `scap` dependency hand-rolls D-Bus request/response
matching (`portal.rs::handle_req_response`) and has a real, confirmed
race there (registers the signal match *after* sending the method call —
see `capture-validation-runbook.md`'s `create_session` panic writeup).
`XdgPortalInjector` uses `ashpd` instead specifically to avoid
reintroducing that bug class in new code; no equivalent race was hit
during this validation, consistent with `ashpd` being a more mature
abstraction over the same underlying D-Bus request/response pattern.

## What's NOT validated by this harness

Per xenia-peer's `ROADMAP.md`, `XdgPortalInjector` is layer 1 of three
needed for input injection to work end-to-end from a real viewer:

1. ✅ **The backend itself** — this runbook.
2. ❌ **Daemon receive-loop.** `apps/xenia-peer/src/main.rs` has no code
   path that receives `RawInput` from a connected viewer during an
   active session; `xenia-inject` isn't a dependency of `xenia-peer` at
   all yet.
3. ❌ **Viewer capture.** `apps/xenia-viewer/src/gui.rs` explicitly notes
   real mouse/keyboard capture isn't wired ("that's M2").

This runbook only speaks to (1). A real end-to-end "move the mouse in
the viewer, see it move on the host" proof needs (2) and (3) built first.
