# ADR 0001 — Cross-platform screen capture backend

**Status:** Accepted (pending real-hardware validation)
**Date:** 2026-04-19
**Deciders:** Luminous Dynamics core team

## Context

`xenia-capture` is a new MIT+Apache-2.0 crate in the Xenia family. It wraps a cross-platform screen-capture backend behind a `FrameProducer` trait aligned to `xenia-wire::RawFrame`. The Xenia admin and employee sides both depend on it; its OS coverage directly determines Suite OS coverage.

**Target matrix:**
- Windows 10/11
- macOS 12+ (ScreenCaptureKit era)
- Linux X11
- Linux Wayland (GNOME, KDE; wlr-screencopy-protocol-aware compositors out of scope year 1)

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

**Primary backend: `scap`.**

**Fallback backend: `xcap`**, if scap's beta proves too unstable for the Suite beta timeline.

The `xenia-capture` crate exposes a `FrameProducer` trait so that either backend (or a future third) can be swapped without changes to downstream callers. Configuration in the NixOS module exposes the choice as `services.mycelix-sovereign.xenia.captureBackend = "scap" | "xcap"` (default `"scap"`).

Linux-Wayland compositors not covered by PipeWire/portal (if any surface as important in the design-partner shortlist) are deferred to a dedicated `lamco-pipewire`-style backend behind the same trait in year 2.

## Consequences

- **Unlocks cross-platform Suite beta.** W0 can proceed.
- **Commits us to a beta dependency.** We must pin a specific `scap` version (and possibly a specific commit hash via `[patch]`), track upstream, and be prepared to upstream fixes or fork if scap goes unmaintained.
- **Windows fidelity is shared across xcap and scap both using WGC**, so xcap fallback is viable on Windows with near-zero rework.
- **macOS permissions UX** is the same story on both (ScreenCaptureKit's mandatory permission dialog).
- **Xenia admin UI** must surface capture-backend permission state clearly (first-run permission grant on macOS can add 200-500 ms latency — this is a UX reality, not a bug).

## Pending validation (W0 week 1-2)

Before this ADR is locked in implementation, measure on real hardware:

1. **Capture-to-wire latency at 1080p60** on: Win11 + WGC, macOS 14 + ScreenCaptureKit, GNOME-Wayland + PipeWire, KDE-Wayland + PipeWire, X11 baseline.
2. **Cursor rendering correctness** on mixed-DPI multi-monitor (retina + external 4K).
3. **HDR→SDR tone-mapping behavior** on macOS and Windows 11 HDR-enabled displays.
4. **First-frame latency after permission grant** on macOS (the ScreenCaptureKit permission dialog adds a variable cost that ships directly into user-perceived responsiveness).
5. **PipeWire portal picker cancellation mid-session** — what does scap do if the user cancels the share-picker dialog? Does the `FrameProducer` error cleanly or hang?

If (1) fails to clear 15 FPS at 1080p on any target OS, escalate: evaluate whether it is a scap bug (patch upstream), an xcap opportunity (fallback), or a genuine hardware limit (reduce resolution default in Xenia admin console).

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
