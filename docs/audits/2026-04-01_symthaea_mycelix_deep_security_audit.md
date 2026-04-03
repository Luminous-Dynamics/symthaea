# Deep Security Audit: Symthaea + Mycelix

Date: 2026-04-01
Repo: /srv/luminous-dynamics
Scope: Symthaea installer toolchain + Mycelix local P2P services (code + Nix configs)
Reviewer: Codex (agentic code review)

This audit focuses on *real, exploitable* network and privilege boundaries (not “internal correctness” of the consciousness / Holochain logic).

## Executive Finding

The “Sovereign Inoculation” install pathway is currently shaped like a remote-control implant:

- The custom installer ISO enables remote root access with a static password and advertises itself via mDNS.
- A WebSocket-to-SSH relay is/was exposed without authentication, and (previously) disabled SSH host-key verification.
- The eval API binds to all interfaces with wildcard CORS.
- Mycelix Gun server accepts unauthenticated writes and binds to all interfaces by default.

The inner engines (Symthaea cognition, Mycelix causality loops) can be mathematically correct and still be operationally compromised because the *outer shell* is permissive-by-default and unauthenticated.

This report is written to feed directly into the “Stop the Bleeding” remediation PRs.

## What These Components Were Being Used For (Intent)

- `ssh-relay` (`symthaea-spore`) exists because the browser cannot run `nix`, `nixos-anywhere`, or hold raw SSH connections. The relay provides a WebSocket protocol that:
  - receives “actions” from the portal (probe hardware, list disks, install, etc.)
  - opens an SSH session to the target
  - streams stdout/stderr back to the portal for progress visualization

- `eval-api` (`symthaea-spore`) exists so the portal can evaluate a generated NixOS configuration and display:
  - closure size / store paths
  - a config preview (helps the user sanity-check before installing)

- `installer-iso.nix` exists to enable a “boot target machine, then use any device” flow by:
  - enabling SSH on the live ISO
  - (previous design) auto-starting the WebSocket relay on the ISO
  - advertising the target via Avahi so phones/laptops can find it as `sovereign-inoculation.local`

- `gun-p2p-server.js` (`mycelix-core`) appears to be a dev/test “real P2P database” service used for:
  - agent registration
  - message passing / broadcast
  - a simple reputation namespace

## Threat Model Assumed

- Attacker on the same Wi‑Fi / LAN as a user during install.
- Attacker can lure the user to a malicious website while local services are running.
- Attacker can scan common ports / mDNS and attempt opportunistic takeovers.
- We assume the install target is a *high-value asset* (fresh OS install moment is maximal privilege).

## Findings (Ordered by Severity)

### CRITICAL-1: Custom Installer ISO Has Static Root Password + Remote Root SSH

Evidence:
- `symthaea/nix/installer-iso.nix:56-65` enables SSH with `PermitRootLogin = "yes"` and sets `users.users.root.initialPassword = "sovereign"`.

Impact:
- Any attacker on the same network can SSH in as root using a known password during install.
- This is a full machine compromise at the most sensitive moment (disk wipe + OS bootstrap).

Recommendation:
- Remove static root password from the ISO.
- Generate a high-entropy *one-time* password on boot, store in `/run`, and display on the console.
- Prefer firewall allowlist to only TCP/22 during install.

Status:
- Remediation planned (Stop-the-Bleeding Day 1).

### CRITICAL-2: ISO Auto-Starts a WebSocket Relay and Broadcasts It via Avahi

Evidence:
- `symthaea/nix/installer-iso.nix:66-77` auto-starts the relay on port 8094.
- `symthaea/nix/installer-iso.nix:79-106` enables Avahi and publishes a service pointing at port 8094.
- The console help text explicitly instructs users to use the relay (and repeats the static password). (`symthaea/nix/installer-iso.nix:147-192`)

Impact:
- Remote attackers discover the target via mDNS and gain a ready-made control channel.
- Even if SSH is fixed, this advertises “an install surface is available here” to the whole subnet.

Recommendation:
- Stop auto-starting relay on the ISO (or bind relay to `127.0.0.1` only if it must exist).
- Disable Avahi broadcasting by default; rely on console-displayed IP for discovery.

Status:
- Remediation planned (Stop-the-Bleeding Day 1).

### CRITICAL-3: Eval API Binds to 0.0.0.0 With Wildcard CORS

Evidence:
- CORS wildcard: `symthaea/crates/symthaea-spore/src/bin/eval_api.rs:299-302`
- Bind-all-interfaces: `symthaea/crates/symthaea-spore/src/bin/eval_api.rs:312`
- IP rate limiting trusts `x-forwarded-for` header: `symthaea/crates/symthaea-spore/src/bin/eval_api.rs:250-267`

Impact:
- Any host on the LAN can hit the eval endpoint and force expensive `nix` evaluation work (DoS), or probe environment/system behavior.
- Wildcard CORS makes it callable from arbitrary browser origins, increasing CSRF/drive-by abuse risk.
- Trusting `x-forwarded-for` allows trivial rate-limit bypass/spoofing.

Recommendation:
- Bind to `127.0.0.1` by default.
- Replace wildcard CORS with an explicit allowlist of local portal origins.
- Use socket remote address for rate limiting; only trust proxy headers when explicitly configured.

Status:
- Remediation planned (Stop-the-Bleeding Day 1).

### CRITICAL-4: Mycelix Gun Server Accepts Unauthenticated Writes and Binds Publicly

Evidence:
- Listens with no host binding: `mycelix-core/gun-p2p-server.js:191-214` (`server.listen(PORT, () => ...)`)
- No auth on write endpoints:
  - `POST /api/agent/register` (`mycelix-core/gun-p2p-server.js:52-72`)
  - `POST /api/message/send` (`mycelix-core/gun-p2p-server.js:74-100`)
  - `POST /api/reputation/update` (`mycelix-core/gun-p2p-server.js:115-141`)
  - `socket.on('gun-data', ...)` writes arbitrary paths (`mycelix-core/gun-p2p-server.js:179-182`)

Impact:
- Remote hosts can write arbitrary data into your Gun namespace, poison “reputation”, spam messages, and potentially fill disks (`radisk: true`, `file: 'gundata'`).

Recommendation:
- Bind to `127.0.0.1` by default; require explicit opt-in to expose externally.
- Restrict browser origins (CORS / Socket.IO origin controls).
- (Next layer) Add a write-token or capability gating for mutating operations.

Status:
- Remediation planned (Stop-the-Bleeding Day 1).

### HIGH-1: Symthaea Web Remote Install UI Defaults to Insecure ISO Relay + Password

Evidence:
- Defaults to mDNS host + relay port: `symthaea/crates/symthaea-web/src/pages/remote_install.rs:106-108`
- Defaults to root password “sovereign”: `symthaea/crates/symthaea-web/src/pages/remote_install.rs:108`
- Connects over plaintext `ws://` and immediately issues a `"connect"` action: `symthaea/crates/symthaea-web/src/pages/remote_install.rs:124-153`

Impact:
- Normalizes insecure defaults in UX.
- If the ISO remains discoverable, users are taught to keep using a static credential.

Recommendation:
- Remove insecure defaults; require the user to paste a one-time password.
- Align the UI with the secured relay protocol (token-auth, localhost relay).

Status:
- Remediation planned (Stop-the-Bleeding Day 1).

### HIGH-2: Sovereign Scan WebSocket Server Has No Auth (Local Data Exfil Risk)

Evidence:
- Scan server binds to localhost and sends results on connect with no auth:
  - `symthaea/crates/symthaea-scan/src/main.rs` (`serve_websocket(...)`)
- The installer UI auto-connects to it:
  - `symthaea/crates/symthaea-web/src/pages/install.rs:860-905`

Impact:
- Any website a user visits can attempt a CSWSH-style connection to `ws://127.0.0.1:7799` and read scan results while the scanner is running.
- The scan output includes installed applications and system details (privacy + targeting).

Recommendation:
- Add token-auth (same pattern as ssh-relay): require `"auth"` before sending scan results.
- Update the web UI to send the token.

Status:
- Not in the original Stop-the-Bleeding checklist, but strongly recommended as the same class of vulnerability.

### MEDIUM-1: Documentation Claims Security Posture That Code Does Not Match

Evidence:
- `docs/SECURITY_VERIFICATION_RESULTS.md` includes a “Transport & API Security” table claiming:
  - “CORS wildcard → allowlist”
  - “0.0.0.0 → 127.0.0.1”
  - “API auth: none → bearer token”
- In-repo code currently contradicts these claims (e.g. eval-api is wildcard + 0.0.0.0).

Impact:
- False confidence / operational risk: teams assume mitigations exist and deploy accordingly.

Recommendation:
- Treat docs as aspirational until enforced by tests/CI.
- Add regression tests or lightweight “bind/cors” checks (or a security checklist gate) for binaries that listen on sockets.

Status:
- Requires follow-up after Day 1 patches (docs update + regression).

### MEDIUM-2: Routing Registry vs DNA Manifests Drift (Breaks Cross-Cluster Calls)

Evidence:
- Registry uses `"bridge"` in several clusters:
  - `crates/mycelix-bridge-common/src/routing_registry.rs` (e.g. `FINANCE_LOCAL_ZOMES` includes `"bridge"`)
- But the DNA manifests use different coordinator zome names:
  - Finance uses `finance_bridge`: `mycelix-finance/dna/dna.yaml`
  - Supplychain uses `bridge_coordinator`: `mycelix-supplychain/holochain/dna/dna.yaml`
  - Knowledge/Energy/Manufacturing use `<cluster>_bridge` (see their zome Cargo.tomls)

Impact:
- Cross-cluster bridge dispatch can silently fail (or hit the wrong zome), leading to “security by accident” behaviors where governance/dispatch paths don’t match design.

Recommendation:
- Align `routing_registry` to the real manifest names (source of truth = DNA/hApp manifests).

Status:
- Scheduled for Day 2 “architectural drift” fixes.

### MEDIUM-3: Symthaea ↔ Mycelix Conductor Adapter Calls Likely-Nonexistent Roles/Zomes

Evidence:
- Governance role default: `"mycelix-governance"` (`symthaea/crates/symthaea-mycelix-conductor/src/lib.rs`)
- Uses zome `"agora"` and fn `"create_proposal"`:
  - `symthaea/crates/symthaea-mycelix-conductor/src/lib.rs` (SubmitProposal path)

Impact:
- Governance calls may be no-ops or always fail; if errors are swallowed upstream this can create a “phantom governance” system (UI says it acted, chain never did).

Recommendation:
- Align role + zome function names to actual governance DNA manifests.
- Make failures loud (do not return fake success).

Status:
- Scheduled for Day 2.

### LEGAL-1: Licensing Is Internally Contradictory (AGPL vs BSL vs MIT/Apache)

Evidence:
- Root `LICENSE` is Business Source License 1.1 (Change License Apache-2.0).
- Many source files carry `SPDX-License-Identifier: AGPL-3.0-or-later`.
- Some packages declare different licenses (e.g. `symthaea/crates/symthaea-spore/Cargo.toml` declares `license = "MIT"`).
- Repo root also contains `COMMERCIAL_LICENSE.md` / `LICENSING_FAQ.md` (present on disk; tracking status unclear).

Impact:
- You cannot truthfully communicate “the repo is AGPLv3+” if the root license is BSL and subprojects differ.
- This is a legal risk for contributors and downstream adopters.

Recommendation:
- Decide the repo-wide licensing intent (you stated AGPLv3+).
- Execute a structured reconciliation (root license, per-package metadata, and third-party exceptions).

Status:
- Scheduled for Day 3.

## Stop-the-Bleeding Remediation (Day 1) — Approved Scope

1) SSH relay
- Localhost-only bind
- Mandatory token auth
- SSH host-key verification enforced (known_hosts)
- Remove arbitrary exec channel

2) Installer ISO
- Remove static root password
- Generate one-time root SSH password on boot + display on console
- Remove public relay and Avahi broadcast (or force localhost bind)

3) Eval API + Gun server
- Bind both to localhost by default
- Replace wildcard CORS with explicit localhost origin allowlist

## Notes

- This audit is based on repository code + config inspection. It does not claim a live penetration test was run.
- The repo worktree is currently very dirty; security PRs should stage only explicitly intended paths.

