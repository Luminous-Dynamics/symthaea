# ADR 0001 — Cross-platform screen capture backend

**Status:** Accepted — **pending upstream Linux-build fix** (2026-04-19, see Upstream blocker below)
**Date:** 2026-04-19
**Deciders:** Luminous Dynamics core team

## Context

`xenia-capture` is an Apache-2.0 OR MIT crate in the xenia-peer workspace. **It already exists** — a 444-LOC implementation landed in `xenia-peer` commit history during Phase I.C (pre-existing `ScreenCapture` trait + `CapturedFrame` + `CaptureError` + `TestCapture` + `BlankCapture` + scaffold stubs for `WlrootsCapture`/`PortalCapture`, 4 passing tests). This ADR therefore decides which *implementation* to wire behind the existing trait, not whether to create the trait.

**Target matrix (revised 2026-04-19):**
- Windows 10/11 (Windows Graphics Capture)
- macOS 12+ (ScreenCaptureKit era)
- Linux Wayland via PipeWire / xdg-desktop-portal (GNOME, KDE, and wlroots-portal-compatible compositors)
- **Linux X11: explicitly out of scope.** Supersedes the original draft of this ADR which listed X11. The reason is the xenia-peer repo's ADR-001 Decision 2: X11's core design permits any client to read any other client's keyboard input and screen contents without elevation, which fundamentally undoes the end-to-end consent + sealed-transport security model. Running Xenia on X11 bypasses the threat model upstream of the wire. If an X11 compatibility gap emerges later with enough demand to justify the security cost, a separate `xenia-peer-x11` crate can be authored as a community fork — the reference implementation will not ship one.

**Target performance:** ≥15 FPS at 1080p; capture-to-wire latency <100 ms.

**Candidates identified in the W0 plan:**
- `xcap` (nashaofu/xcap) — https://github.com/nashaofu/xcap
- `crabgrab` (AugmendTech/crabgrab) — https://github.com/AugmendTech/crabgrab

**Research surfaced a third candidate:**
- `scap` (CapSoftware/scap) — https://github.com/CapSoftware/scap — built and used in production by Cap, an open-source Loom alternative.

### Findings (research agent, 2026-04-19)

| Criterion | xcap | crabgrab | scap |
|---|---|---|---|
| Latest release | 0.9.4 (2026-04-09) | 0.8.0 (2024); repo **archived 2024-10** | 0.1.0-beta.1 (2025-08-04) |
| Maintenance | Active — commits this month | **Dead — archived** | Alive; Cap depends on it in prod; slower cadence |
| Windows | WGC ✅ | WGC ✅ | WGC ✅ |
| macOS | ✅ (standard Screen Recording permission) | ScreenCaptureKit + Metal/IOSurface ✅ | ScreenCaptureKit, with built-in `has_permission()`/`request_permission()` ✅ |
| Linux X11 | ✅ | ❌ | Via XDG portal fallback (not first-class) |
| Linux Wayland | **"Limited" per their own matrix** ⚠️ | ❌ | **PipeWire-native via xdg-desktop-portal ✅** |
| Built-in encoding | None (raw frames) | None (GPU surfaces) | None (raw BGRA) |
| FPS / latency claims | None documented | Low-latency by design (zero-copy GPU) | Configurable; 60 FPS example |
| License | Apache-2.0 | Apache-2.0 | MIT |
| Stars (rough) | ~950 | ~160 (archived) | ~600 |

### Disqualifications

- **`crabgrab` is archived (read-only as of 2024-10)**. Last commit 2024-06-14. Committing a commercial 2026+ product to a dead dependency is untenable.

### Comparison — xcap vs scap

- **xcap** has the largest star count, most recent release, and simplest API. Its critical weakness is **Wayland support is labelled "limited"** by the maintainer — which means a 2026+ Linux-desktop story that is already failing on GNOME-Wayland, KDE-Wayland, and Sway. xcap's Wayland path would either remain weak (ceding a large fraction of Linux users) or require us to fork and rewrite.

- **scap** has a weaker Windows/macOS track record (beta API, 8 months since last commit at time of ADR) but its Linux story is architecturally correct: **xdg-desktop-portal / PipeWire** is what Wayland compositors converge on. For a product with a multi-year horizon in the NIS2 era (EU public sector ships predominantly on GNOME-Wayland), this is the right bet.

- **scap's beta status is the primary risk.** API churn is likely. Cap (CapSoftware's consumer product) hard-depends on it in production, which gives us some confidence the API won't thrash arbitrarily, but we should be prepared to pin a specific commit and upstream patches.

## Decision

**Primary backend: `scap`**, wired into the pre-existing `ScreenCapture` trait in `xenia-peer/crates/xenia-capture` as a new `scap-backend` Cargo feature.

**Fallback backend: `xcap`**, if scap's beta proves too unstable for the Suite beta timeline.

The existing `ScreenCapture` trait (not a new `FrameProducer` trait — the prior art already provides the right abstraction) ensures either backend can be swapped without changes to downstream callers. Configuration in the NixOS module exposes the choice as `services.mycelix-sovereign.xenia.captureBackend = "scap" | "xcap"` (default `"scap"`).

### Integration details — scap's blocking API

scap's `Capturer::get_next_frame()` is **blocking** (`mpsc::Receiver::recv`, no `try_recv` exposed). The existing `ScreenCapture::capture()` trait returns `Ok(None)` when no new frame is available (non-blocking poll). To reconcile, the scap backend runs the `Capturer` on a dedicated worker thread and forwards frames via an `mpsc` channel that the trait's `capture()` reads with `try_recv`. This design has two additional benefits:

- scap's `Capturer` is `!Send` on Windows (upstream issue #145). Constructing it inside the worker thread sidesteps the Send bound entirely.
- Linux portal-cancel is not surfaced as a distinct error upstream — a worker-thread supervisor can map `RecvError` / `Disconnected` to the trait's `CaptureError::Backend` cleanly without polluting callers.

BGRA→RGBA conversion is done in-place via `chunks_exact_mut(4).swap(0, 2)` on each frame as it crosses the channel boundary (the trait's `CapturedFrame` documents RGBA, top-left origin).

## Consequences

- **Unlocks cross-platform Suite beta.** W0 can proceed.
- **Commits us to a beta dependency.** We must pin a specific `scap` version (and possibly a specific commit hash via `[patch]`), track upstream, and be prepared to upstream fixes or fork if scap goes unmaintained.
- **Windows fidelity is shared across xcap and scap both using WGC**, so xcap fallback is viable on Windows with near-zero rework.
- **macOS permissions UX** is the same story on both (ScreenCaptureKit's mandatory permission dialog).
- **Xenia admin UI** must surface capture-backend permission state clearly (first-run permission grant on macOS can add 200-500 ms latency — this is a UX reality, not a bug).

## Pending validation (W0 week 1-2)

Before this ADR is locked in implementation, measure on real hardware:

1. **Capture-to-wire latency at 1080p60** on: Win11 + WGC, macOS 14 + ScreenCaptureKit, GNOME-Wayland + PipeWire, KDE-Wayland + PipeWire. (X11 dropped — see context.)
2. **Cursor rendering correctness** on mixed-DPI multi-monitor (retina + external 4K).
3. **HDR→SDR tone-mapping behavior** on macOS and Windows 11 HDR-enabled displays.
4. **First-frame latency after permission grant** on macOS (the ScreenCaptureKit permission dialog adds a variable cost that ships directly into user-perceived responsiveness).
5. **PipeWire portal picker cancellation mid-session** — scap does not surface this as a distinct error upstream (upstream issue #170). Our worker-thread wrapper must map channel disconnect → `CaptureError::Backend` cleanly without hanging.
6. **Windows `!Send` Capturer** — verify the worker-thread construction pattern holds up on Windows (the Capturer is `!Send` there per upstream issue #145; we construct it inside the worker).

If (1) fails to clear 15 FPS at 1080p on any target OS, escalate: evaluate whether it is a scap bug (patch upstream), an xcap opportunity (fallback), or a genuine hardware limit (reduce resolution default in Xenia admin console).

## Upstream blocker — scap 0.1.0-beta.1 does not compile on Linux (2026-04-19)

When the scap scaffold was wired into `xenia-peer/crates/xenia-capture` and `cargo check --features scap-backend` was run inside `nix develop` (providing `pipewire`, `dbus`, `wayland`, `libspa` system libs), scap itself failed to compile with 8 errors in `scap/src/capturer/engine/linux/mod.rs`:

- `Frame::XBGR(XBGRFrame { .. })` — variant not found on `Frame` enum
- `Frame::BGRx(BGRxFrame { .. })` — variant not found on `Frame` enum
- `display_time: timestamp as u64` — expected `SystemTime`, found `u64` (four sites)

Both the published `=0.1.0-beta.1` and the tip of `main` (commit `c03f15a4`, "fix windows build", post-beta.1) fail with identical errors. The Windows-build fix did not address Linux. This is an upstream regression: the `Frame` enum was simplified (variants removed) without updating the Linux capturer engine.

**Action taken:**
1. `scap-backend` feature marked SCAFFOLDED BUT NOT USABLE on Linux in the crate's `Cargo.toml`. macOS/Windows paths not verified from our NixOS host.
2. Our `src/scap_backend.rs` wrapper is kept in-tree — the worker-thread / BGRA→RGBA / permission-mapping design is sound and will compile against a working upstream without changes.
3. `cargo check -p xenia-capture` (baseline, feature OFF) passes.

**Decision path forward:**
- **Option 1 (watch-and-wait, default):** Monitor `CapSoftware/scap` for 0.1.0-beta.2 or a merged PR that fixes the Linux engine. Revisit when released.
- **Option 2 (escalate to fallback):** If scap beta.2 slips past W0 week 2, flip primary backend to `xcap` without architectural change — the `ScreenCapture` trait moat makes the swap a single new backend module plus a default-feature flip. `xcap`'s Wayland limitations become an accepted year-1 trade-off.
- **Option 3 (upstream fix):** Author and submit a PR to scap fixing the Linux engine. Effort estimate: 1-2 days. Only worth it if option 2 is undesirable and option 1 is not moving.

**Lesson validated:** This is exactly the failure mode the trait-moat architecture was designed to survive. If we had written scap calls directly into xenia-peer-core's session pipeline, upstream's break would have forced us into option 3 immediately. With the moat, we can proceed to other W0 work while the upstream situation resolves itself.

## Known upstream issues to track

Open scap issues relevant to our integration (as of 2026-04-19, ordered by severity for our use):

- **#172** Linux (Ubuntu 22) build failure — `pipewire`/`libspa-sys` ABI. Mitigation: pin scap version, require `nix develop` or equivalent for Linux builds.
- **#158** Wayland / Niri and other non-GNOME compositors — portal path issues.
- **#170** Portal cancel not surfaced as distinct error. Mitigation: worker-thread wrapper (see Integration details above).
- **#145** `Capturer` is `!Send` on Windows. Mitigation: construct inside worker thread (already planned).
- **#159** Capture sometimes returns 0-byte data, after which frame latency 4-5×. Mitigation: validate frame length in the worker before forwarding.
- **#140** macOS yellow recording border cannot be fully suppressed. UX consideration; document in admin console.
- **#118** Multi-display: `Display` has no position info; monitor ordering/identification ambiguous. Our existing `MonitorDescriptor` has `x_offset` / `y_offset` — may need platform-specific enrichment.
- **#182** `scap::targets::Window` import broken on 0.1.0-beta.1. We target primary display (not windows) in year 1, so not blocking.

## Alternatives rejected

- **`crabgrab`** — archived; disqualified.
- **Bare-metal OS APIs** (DXGI/DX11 on Windows, ScreenCaptureKit on macOS, Wayland portals directly) — rejected per product plan: "utilize an existing cross-platform Rust screen capture crate rather than writing bare-metal OS APIs from scratch." Engineering velocity > marginal latency gain.
- **FFmpeg `libavdevice`** — heavy C dependency, not idiomatic Rust, hurts cross-compile. Rejected.
- **`screenshots` crate** — deprecated; author redirects to xcap.
- **`lamco-pipewire` as primary** — Linux-only, not a candidate for primary.

## References

- [xcap on GitHub](https://github.com/nashaofu/xcap) · [crates.io](https://crates.io/crates/xcap)
- [scap on GitHub](https://github.com/CapSoftware/scap) · [crates.io](https://crates.io/crates/scap)
- [crabgrab on GitHub](https://github.com/AugmendTech/crabgrab) (archived)
- [xdg-desktop-portal / PipeWire screencast spec](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.ScreenCast.html)
- MYCELIX_SOVEREIGN_PLAN.md §6 W0
