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
| Harness shipped | ✅ `xenia-peer/crates/xenia-capture/examples/capture_bench.rs` |
| Linux local (dev box) | ⬜ pending PR #183 merge (upstream broken on Linux) |
| macOS | ⬜ needs operator with Mac 14+ |
| Windows 11 | ⬜ needs operator with Win 11 |
| GNOME-Wayland | ⬜ pending #183 merge |
| KDE-Wayland | ⬜ pending #183 merge |
