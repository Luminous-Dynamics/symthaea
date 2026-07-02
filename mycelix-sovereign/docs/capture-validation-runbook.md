# Runtime Validation Runbook

*Capture (W0) + xenia-admin resolve_did (W1). Keep this file as the single operator-facing verification playbook for Mycelix Sovereign; new routes get appended at the bottom.*

---

## xenia-admin: resolve_did end-to-end (W1 Stream A)

### Architectural verification (done 2026-04-19)

Native E2E probe using `holochain_client = "=0.9.0-dev.20"` against the shared dev conductor ran all four layers:

1. **Admin WebSocket** at `ws://localhost:33800` — connected OK.
2. **Issue app-authentication token** for `mycelix-unified` — OK, 64-byte token.
3. **App WebSocket** at `ws://localhost:8888` with the token + authorize_signing_credentials for the identity cell — OK.
4. **`did_registry::resolve_did(did)` zome call** — responded correctly:
   - With a valid-format DID (`did:mycelix:<agent_pub_key>`): `Ok(None)` returned as 1-byte MessagePack nil (0xc0) — the DID isn't registered yet, and the zome signals that cleanly.
   - With a bogus DID string: zome-side rejection with `Guest("Invalid agent pub key in DID")` at `did_registry/coordinator/src/lib.rs:293` — proves the zome received and processed input.

The zome-call path is architecturally verified end-to-end against the live conductor. The browser path via `mycelix-leptos-client` takes the same route with **unsigned zome calls** (zeroed signature + raw agent_pub_key as provenance), which works against conductors with Unrestricted capability grants — confirmed present on this conductor via `hc sandbox call --running=33800 list-capability-grants mycelix-unified` (empty function lists = unrestricted).

### Browser click-through (pending human)

The browser-click loop is straightforward now; gating on an operator with hands on the dev box. Steps:

1. **Conductor check.** Verify the shared conductor is running:
   ```sh
   ss -tlnp | grep -E ':(8888|33800) '
   ```
   Both `127.0.0.1:8888` (app) and `127.0.0.1:33800` (admin) should be LISTEN.

2. **Obtain an app auth token.** The admin interface requires no auth on localhost but uses a specific protocol. The fastest path is the same native probe used in architectural verification:
   ```sh
   mkdir -p /tmp/xenia-token && cd /tmp/xenia-token
   # Copy Cargo.toml + src/main.rs from this runbook's appendix, OR run
   # the full e2e-probe which also calls resolve_did to validate
   cargo run --quiet 2>/dev/null | grep -A1 "token"
   ```
   Output: `OK (token 64 bytes)` followed by a base64-ish token string. Save it.

3. **Inject token into `index.html`.** Edit `xenia-peer/crates/xenia-admin/index.html` — uncomment the `window.__HC_AUTH_TOKEN` line and paste the token (as a base64 string, NOT the raw bytes):
   ```html
   <script>
     window.__HC_AUTH_TOKEN = "<token from step 2>";
   </script>
   ```
   Tokens expire after 300s per the probe config; re-issue as needed.

4. **Build + serve.**
   ```sh
   cd /srv/luminous-dynamics/xenia-peer/crates/xenia-admin
   ~/.cargo/bin/trunk serve
   ```
   Wait for `applying new distribution` / `✅ success`.

5. **Browser test.** Open `http://localhost:8134/login`. Paste the real mycelix-unified agent DID:
   ```
   did:mycelix:uhCAkOf3jQ4Eq1A3H0xVn4S2dOZLyT-JtcDPE3HpPIFrkI7eF4bIF
   ```
   Click **Sign in**. Expected: "DID not found in the identity registry" banner (the DID isn't registered on chain yet — to get a positive result, first call `create_did()` via any identity client to register it).

6. **Confirm the footnote** under the form shows `auth: token set`. If it shows `auth: none (...)`, the window global didn't stick — check index.html script ordering.

### Runbook-only probe (appendix)

If you want the token without the full resolve_did call, the probe binary in `/tmp/e2e-probe/src/main.rs` (reproduced below for easy re-creation; recreate from this runbook after any rustc version change):

```rust
use anyhow::{Context, Result};
use holochain_client::{AdminWebsocket, IssueAppAuthenticationTokenPayload};
use std::net::Ipv4Addr;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let admin = AdminWebsocket::connect((Ipv4Addr::LOCALHOST, 33800), None)
        .await.context("admin connect")?;
    let resp = admin.issue_app_auth_token(IssueAppAuthenticationTokenPayload {
        installed_app_id: "mycelix-unified".to_string(),
        expiry_seconds: 300,
        single_use: false,
    }).await?;
    // Tokens are Vec<u8>; base64-encode for copy-paste into window.__HC_AUTH_TOKEN
    use base64::{engine::general_purpose::STANDARD, Engine};
    println!("{}", STANDARD.encode(&resp.token));
    Ok(())
}
```

Cargo.toml:
```toml
[dependencies]
holochain_client = "=0.9.0-dev.20"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
anyhow = "1"
base64 = "0.22"
```

If `cargo build` fails on `constant_time_eq requires rustc 1.95.0` with your toolchain:
```sh
cargo update -p constant_time_eq@0.4.3 --precise 0.4.2
```

### Findings from this verification pass

**Positive:**
- Conductor exposes `mycelix-unified` with `identity` role as scaffold defaults expected
- Capability grants are Unrestricted (compatible with mycelix-leptos-client's unsigned-call pattern)
- `did_registry::resolve_did` zome is live and validates input correctly
- All four network layers (admin WS, app WS, token issue, signing creds) work together

**Known limitation:**
- The scaffold footer shows `auth: none` with no token set, which is what you get out of the box. The token has to be issued + pasted on a per-deployment-per-300s basis. A future enhancement: have the admin console fetch its own token via the admin WebSocket (requires no changes to mycelix-leptos-client, just an extra step in the LoginPage flow — ticket this for W1 tail-end).

---

# Capture Backend Validation Runbook — W0 scap on 4 OSes

**Purpose:** verify `scap` (via `xenia-capture`) meets Mycelix Sovereign W0 acceptance criteria on every target OS after [CapSoftware/scap#183](https://github.com/CapSoftware/scap/pull/183) merges.

**Owner:** each platform's owner runs their row and posts results back to the ADR 0001 measurements section.

**Tool:** `cargo run -p xenia-capture --features scap-backend --example capture_bench` (lives in `xenia-peer/crates/xenia-capture/examples/capture_bench.rs`).

## Target matrix

| OS / compositor | Acceptance | Pass criteria |
|---|---|---|
| Windows 11 + WGC | primary target | ≥30 fps native; TCC-style prompt absent; `!Send` Capturer doesn't panic |
| macOS 14+ + ScreenCaptureKit | primary target | ≥30 fps native; first-run TCC prompt within 5s of run; subsequent runs <500ms first-frame |
| Linux GNOME-Wayland + PipeWire | primary target | ≥15 fps native; portal picker appears; cancel→clean `CaptureError::Backend`; no hang |
| Linux KDE-Wayland + PipeWire | primary target | ≥15 fps native; same as GNOME |

Linux X11 is **out of scope** per xenia-peer ADR-001 Decision 2 (threat-model grounds).

## Universal acceptance criteria

- `VERDICT: PASS` printed by the harness (effective_fps ≥ 15, errors == 0, frame bytes > 0)
- Conversion correctness: first frame dumped via `DUMP_FRAME=/tmp/first.rgba` has alpha byte (index 3) == 255 on every 4-byte pixel (quick check: `od -An -tu1 -N4 /tmp/first.rgba` — last number should be 255)
- No panics, no unhandled errors in stderr

## Per-OS runbook

### macOS 14+

1. On a **clean** machine (no prior scap Screen Recording grant), check initial state:
   ```
   tccutil reset ScreenCapture
   ```
2. Grant Xcode / your terminal "Screen Recording" in System Settings → Privacy & Security → Screen Recording.
3. Run:
   ```
   cargo run -p xenia-capture --features scap-backend --example capture_bench
   ```
4. **First run** expected to trigger TCC prompt if cleared. Grant, re-run. Record **first-frame latency** with and without prompt.
5. `DUMP_FRAME=/tmp/first.rgba cargo run ...` — verify alpha bytes.
6. **Known UX:** scap issue #140 — yellow "recording" border cannot be suppressed. Document in admin console copy.

**Report columns:** effective_fps, first_frame_latency_ms (with-prompt + without-prompt), frame_bytes, errors, panic-yn.

### Windows 11

1. Ensure Rust 1.85+ (`rustup show`).
2. Ensure Windows 10 1903 or later (WGC requirement).
3. Run from PowerShell or Command Prompt:
   ```
   cargo run -p xenia-capture --features scap-backend --example capture_bench
   ```
4. WGC picker should be silent — no dialog expected (unlike macOS TCC).
5. Multi-monitor: verify primary display is captured by default (`target: None` in `ScapOptions`).
6. Dump first frame with `DUMP_FRAME` env and verify alpha.
7. **Known issue:** scap #145 — `Capturer` is `!Send`. Our worker thread constructs it locally; verify no panic at start.

**Report columns:** effective_fps, first_frame_latency_ms, frame_bytes, errors, panic-yn, WGC vs DXGI.

### Linux + GNOME-Wayland

1. Inside `xenia-peer/` run `nix develop` (provides `pipewire`, `dbus`, `wayland-protocols`).
2. Verify your session:
   ```
   echo $XDG_SESSION_TYPE   # expect: wayland
   echo $XDG_CURRENT_DESKTOP  # expect: GNOME
   ```
3. Verify portal is installed:
   ```
   which xdg-desktop-portal xdg-desktop-portal-gnome
   systemctl --user status xdg-desktop-portal
   ```
4. Run the harness:
   ```
   cargo run -p xenia-capture --features scap-backend --example capture_bench
   ```
5. **Portal picker appears** shortly after start. Select "Entire screen" or a monitor. Observe **first-frame latency** (includes picker interaction time).
6. **Portal-cancel test:** run again; in the picker, click Cancel. Harness must exit cleanly with `FAIL` verdict and report the error — NOT hang.
7. Dump first frame via `DUMP_FRAME`; verify alpha bytes (on Linux, format is likely BGRx or BGR0 — our `frame_to_rgba` force-sets alpha to 255, so this should pass).

**Report columns:** effective_fps, first_frame_latency_ms (initial vs subsequent), frame_bytes, errors, portal-cancel-clean-yn, which VideoFrame variant was observed (add a `tracing` debug log if needed).

### Linux + KDE-Wayland (Plasma 6)

1. Same as GNOME but ensure `xdg-desktop-portal-kde` is the active portal (not `xdg-desktop-portal-gnome`):
   ```
   which xdg-desktop-portal-kde
   ```
2. Same harness invocation; same checks.
3. **Known issue:** scap #158 tracks portal-path issues on non-GNOME compositors. Report any discrepancies so we can contribute upstream if needed.

**Report columns:** same as GNOME + KDE-specific notes.

## Failure modes to verify

From ADR 0001 §Known upstream issues:

- [ ] **#170** Portal cancel mid-session → clean error, no hang
- [ ] **#159** 0-byte frames dropped (our `length_matches()` check)
- [ ] **#145** Windows `!Send` Capturer doesn't panic (worker-thread construction)
- [ ] **#151** FPS / output_type options respected on Win/Mac (ignored on Linux — document the observed variant)
- [ ] **#140** macOS yellow border present (UX doc)
- [ ] **#118** Multi-display position ambiguity (MonitorDescriptor x/y offsets)

## After validation

1. Paste results into [ADR 0001](adr/0001-screen-capture-backend.md) "Pending validation" subsection (replacing the pending checklist with measured values).
2. Update [MYCELIX_SOVEREIGN_PLAN.md §12](../../MYCELIX_SOVEREIGN_PLAN.md) — mark "Real-hardware validation" completed.
3. If any target fails, open ADR 0002 to either (a) adopt xcap fallback for that specific OS, or (b) document the scop-workaround path.

## Escalation

- scap upstream fix (PR #183) **not merged** within W0 week 2 → switch xenia-capture to xcap as primary, keep scap_backend.rs for future re-adoption. Blast radius contained by the `ScreenCapture` trait moat.
- scap merges but a specific OS underperforms → document as known limitation in the release notes; do not block W0 beta.

## Status

| Task | Status |
|---|---|
| Harness shipped | ✅ `xenia-peer/crates/xenia-capture/examples/capture_bench.rs` — now self-contained (generates its own on-screen activity, retries past a known upstream D-Bus race, measures steady-state fps) |
| Linux local (dev box) | ✅ validated 2026-07-02 on KDE-Wayland (see below); GNOME attempted 2026-07-02/03, blocked on a real environmental gap (see below), not yet passing |
| macOS | ⬜ needs operator with Mac 14+ |
| Windows 11 | ⬜ needs operator with Win 11 |
| GNOME-Wayland | 🟡 **BLOCKED 2026-07-03** — genuine environmental gap in the disposable NixOS test VM, not a code defect. `capture_bench` fails identically on repeated runs: `VERDICT: FAIL`, 0 frames, 11 errors, `scap capturer build panicked after 5 attempts (known upstream D-Bus request/response race): LinCapError { msg: "Did not get response" }`. See below for root cause and what's needed to actually clear this. |
| KDE-Wayland | ✅ **PASS 2026-07-02** — 22.97/23.10 fps fully unattended (two consecutive runs), ≥15fps bar cleared with no human interaction and no retries needed |

### KDE-Wayland results (2026-07-02, `scap` fork branch `fix/linux-engine-two-level-frame-enum`)

Before this run, `cargo build -p xenia-capture --features scap-backend` failed
outright (`FrameData` unresolved in `scap_backend.rs`, 6 sites — fixed in
`70e7267`). This is the first time this feature has ever actually been
compiled with a real toolchain, let alone run against a live display.

Six runs of `capture_bench` total after the fix (2 debug, 4 release —
`opt-level` changed from `"z"` to `3` for the release profile partway
through, see workspace `Cargo.toml`):

| Run | Profile | Frames | Elapsed | Effective fps | Dims | Bytes/frame | First-frame latency | Errors |
|---|---|---|---|---|---|---|---|---|
| 1 | debug | 77 | 30.0s | 2.57 | 1920×1080 | 8,294,400 | 7.63s | 0 |
| 2 | debug | 30 | 30.0s | 1.00 | 1920×1080 | 8,294,400 | 3.76s | 0 |
| 3 | release (`opt-level=z`) | 261 | 30.0s | 8.70 | 1920×1080 | 8,294,400 | 3.55s | 0 |
| 4 | release (`opt-level=3`) | 0 | 10.1s | — | — | — | — | 11 (panic, see below) |
| 5 | release (`opt-level=3`) | 0 | 10.1s | — | — | — | — | 11 (panic, see below) |
| 6 | release (`opt-level=3`) | 10 | 30.0s | 0.33 | 1920×1080 | 8,294,400 | 2.79s | 0 |

**Capture itself works**: correct native resolution, zero decode/backend
errors on every run that got past session creation. **Performance does not
meet the ≥15fps acceptance bar on any run**, but the run-to-run variance
(0.33–8.70 fps) is far larger than any effect attributable to the debug/
release or `opt-level` change — **these numbers should not be trusted as a
measurement of xenia's real capture ceiling.** `ps`/`uptime` during runs 4–6
showed load average 16–26 on a 12-core machine, with ~10 concurrent `rustc`
jobs (other sessions building bevy/symtropy/symthaea) and 3 other active
Claude Code sessions competing for the same CPU — `scap`'s worker thread and
the compositor itself were almost certainly starved intermittently. Redo
this measurement on an idle machine before drawing any conclusion from the
fps numbers; `opt-level = 3` is kept regardless since it's the correct
choice for a CPU-bound per-frame hot path independent of this noisy data.

Two of the six runs failed completely before producing any frames,
independent of `opt-level`: `LinuxCapturer::new` panicked on
`create_session` with `LinCapError { msg: "Did not get response" }` — this
step precedes the interactive source-picker (which scap gives 2 minutes
for), so it isn't a human-reaction-time issue. It recurred several more
times later in the same session (roughly half of all attempts, including
after a full restart of both `xdg-desktop-portal.service` and
`plasma-xdg-desktop-portal-kde.service`, which ruled out stale portal
state as the cause). Root cause not diagnosed — a D-Bus request/response
race in scap's synchronous polling loop (`portal.rs`'s `handle_req_response`)
is the leading candidate. Retrying gets past it reliably. Also worth
flagging upstream: scap panicking here instead of returning `Result`
cleanly is itself a robustness gap — a transient portal hiccup shouldn't
crash the capture worker thread.

### The real fps story: it was never a capture defect

`perf record -g` on a run that got past session creation showed 23% of all
CPU samples in `__memmove_avx_unaligned_erms` (glibc's memcpy), with the
rest opaque inside the stripped `scap`/`pipewire` worker thread. Rebuilding
with `CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_DEBUG=true`
for symbolication kept hitting the `create_session` flake (4 fails in a
row, independent of the portal restart above) before landing a clean
profile — so instead of continuing to fight `perf`, a one-line diagnostic
in `frame_to_rgba` printed which `scap::frame::VideoFrame` variant
PipeWire actually negotiates: **`BGRx`**. That's the cheap in-place
byte-swap path (`chunk.swap(0, 2)`, no allocation) — not the expensive
per-pixel `RGB` (3-byte) expansion path that does a real
`extend_from_slice`-per-pixel copy. So the pixel-format conversion was
never the bottleneck either, ruling out the two most likely code-level
suspects.

That left one hypothesis: PipeWire's KDE ScreenCast implementation is
damage-driven — it only pushes a new frame when on-screen content
visibly changes, not a fixed-rate stream. All six runs above were
measured against a static, idle desktop. Tested directly: with the
operator actively moving the mouse / interacting with a window during
the benchmark, two runs measured **12.40 fps** (hit the full 300-frame
target early, in 24.2s) and then **16.76 fps — VERDICT: PASS**, clearing
the ≥15fps bar outright, with first-frame latency dropping to ~1.9s from
the earlier 2.8–7.6s range.

**Conclusion: `xenia-capture`'s ScapCapture backend is validated on
KDE-Wayland.** The apparent fps deficiency in every earlier run this
session — debug vs. release, `opt-level=z` vs. `3`, heavy system load or
not — was a benchmark methodology gap (idle desktop → correctly throttled
damage-driven capture), not a defect in xenia's code or a real performance
ceiling. Anyone re-running this validation on another OS/compositor should
keep the screen actively changing (mouse movement, a moving window, video
playback) for a meaningful measurement — a static screen will
under-report fps regardless of how fast the pipeline actually is.

### `capture_bench` hardened: fully self-contained, no human required

Two follow-up fixes landed the same day, both in `capture_bench.rs` /
`scap_backend.rs`:

1. **Synthetic on-screen activity.** The harness now opens a small
   override-redirect XWayland window and repaints it (~30Hz) for the
   entire run, generating real compositor damage without depending on a
   human moving the mouse. XWayland is effectively universal on Linux
   desktops (KWin here runs with `--xwayland`), so this works even on a
   pure-Wayland compositor as long as Xwayland is enabled; if no X11
   connection is available it prints a warning and falls back to the old
   "move the mouse yourself" behavior.
2. **Retry past the `create_session` D-Bus race.** Root-caused (not just
   worked around): `scap`'s `handle_req_response` (`portal.rs:274-308`)
   sends the D-Bus method call and only registers the signal match for
   the reply *afterward* — a classic TOCTOU race. If the portal replies
   before the match is registered (plausible for `create_session`, which
   needs no user interaction and should be near-instant), the reply is
   silently missed and `LinuxCapturer::new` panics despite the portal
   having actually succeeded. `scap_worker` now wraps the `Capturer::build`
   call in `catch_unwind` and retries up to 5 times (200ms apart) before
   giving up for real. Also surfaced why fps looked artificially low even
   on otherwise-good runs: each failed attempt burns ~10s of scap's own
   internal timeout, and the harness used to measure fps from process
   start rather than from the first successful frame — a slow-but-
   eventually-successful setup made a perfectly fine pipeline look slow.
   Fixed: `DURATION_SECS` now measures a steady-state window starting at
   first frame, with a separate 90s `setup_timeout` bounding the
   pre-first-frame wait.

Two fully unattended runs after both fixes: **22.97 fps** and **23.10
fps** — `VERDICT: PASS` both times, no retries needed on either run, no
mouse movement, no manual intervention. This exceeds even the best
human-assisted result above (16.76 fps), consistent with the earlier
finding that active, continuous on-screen change (not just occasional
mouse movement) gives PipeWire the most consistent stream of damage
events to work with.

### GNOME-Wayland attempt (2026-07-02/03): blocked, not a code defect

The host's real desktop is KDE (already validated above), so GNOME
required a dedicated VM — a new `xenia-test-vm-gnome` NixOS host
(`/etc/nixos/hosts/xenia-test-vm-gnome/default.nix` in the operator's
personal system flake, separate repo from this monorepo), mirroring the
existing `xenia-test-vm` (KDE) VM but with `services.desktopManager.gnome`
instead of `plasma6`. Building and booting it surfaced three real,
non-obvious infrastructure issues before capture_bench could even run —
worth recording since they'll bite anyone else building a NixOS test VM
for this kind of validation:

1. **QEMU SLIRP's built-in DNS proxy silently drops EDNS0 queries.**
   `cargo`/`nix` fetches inside the guest intermittently hung for 5s per
   lookup then failed with "Name or service not known", even though ICMP
   to the proxy (`10.0.2.3`) was fine and the *very first* lookup after
   boot sometimes succeeded. Root cause: glibc's resolver defaults to
   `options edns0`, and QEMU's usermode DNS proxy doesn't reliably answer
   EDNS0-flagged queries. Fixed with
   `networking.resolvconf.extraOptions = [ "no-edns0" ];` — confirmed via
   five back-to-back `getent hosts` calls with `no-edns0` vs. reliably
   failing without it.
2. **The qemu-vm module's default writable-store overlay is tmpfs-backed
   and capped at half of `virtualisation.memorySize`.** With
   `memorySize = 4096`, that's a 2G `/nix/.rw-store` — nowhere near enough
   for `nix develop`'s full GNOME devShell fetch (ffmpeg/gstreamer/gtk/
   pipewire/mesa/etc.), which failed partway through with "No space left
   on device" even though the VM's actual 16G disk was 98% empty. Fixed
   with `virtualisation.writableStoreUseTmpfs = false;`, which backs the
   overlay by the real disk instead of RAM.
3. **GNOME's ScreenCast picker needs a working GBM/EGL path, which this
   VM's virtio-gpu doesn't provide.** Even after both fixes above,
   `capture_bench` failed identically on two separate runs:
   `VERDICT: FAIL`, 0 frames, `LinCapError { msg: "Did not get response" }`
   after all 5 of `scap`'s built-in D-Bus race retries. `journalctl --user
   -u xdg-desktop-portal-gnome` showed the real cause: `Failed to
   associate portal window with parent window`, `Vulkan: ... Unable to
   open device ... VK_ERROR_INCOMPATIBLE_DRIVER`, `MESA: error: ZINK:
   failed to choose pdev`. `xdg-desktop-portal-gnome`'s picker dialog
   renders via GTK4's GL path (Zink, GL-over-Vulkan), which in turn needs
   a GBM device — and GBM needs a DRM **render** node. This VM's guest
   only exposes `/dev/dri/card0` (a plain KMS/modesetting device, from
   virtio-gpu's default 2D-only mode); there is no `renderD*` node at all.
   `hardware.graphics.enable = true` was already on by default (verified:
   adding it explicitly produced a byte-identical build, i.e. a no-op) and
   the guest does have a full set of Vulkan ICDs registered including
   `lvp_icd.x86_64.json` (lavapipe, software Vulkan) — but GTK4's EGL/GBM
   initialization path doesn't fall through to a pure-software route
   automatically, so the picker window never renders and the D-Bus call
   that depends on it never gets a response. This is a genuinely different
   failure mode from KDE's (KDE's picker worked fine on the same class of
   VM, and KDE's own D-Bus race was purely a timing issue that retries
   resolved — GNOME's is a hard rendering dependency that no amount of
   retrying fixes).

**What would actually clear this**: give the guest a real render node —
either QEMU-side 3D-accelerated virtio-gpu (`virtio-gpu-gl`/virgl, needing
a `-display ...,gl=on` or `egl-headless` backend and a host DRM render
node, both of which this host has: `/dev/dri/renderD128`/`renderD129`
exist) or confirm/force a genuinely software-only EGL path for the
portal's picker. Neither was attempted — this is real, scoped follow-up
work (GPU-accelerated headless QEMU display config, not a quick fix), not
something to force through speculatively. **Conclusion: this is a real
gap in the GNOME test VM's environment, not a defect in xenia-capture or
xenia-peer.** The KDE-Wayland validation above stands as the one
completed real-hardware/real-desktop pass for the ScapCapture backend;
GNOME-Wayland remains unvalidated pending either the render-node fix above
or an operator with a real GNOME-Wayland desktop.
