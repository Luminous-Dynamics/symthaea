# Mycelix Sovereign — Suite Plan

**Date:** 2026-04-19 (rev 2)
**Decision:** Path 2 reframed as **Secure Sovereign Operations Suite**, commercial name **Mycelix Sovereign**.
**Supersedes:** rev 1 ("Xenia PAM" wedge, same day).

---

## 1. Strategic frame

The broad "sovereign workplace" corner (Nextcloud, OpenDesk, Euro-Office, Proton Business) is **ceded to incumbents** — they will own docs, storage, and general office productivity. That market is crowded and their products are good enough.

We own a narrower, higher-value lane:

> **The secure-operations layer of a sovereign workplace.**
> Privileged remote access, encrypted communications, and AI-assisted triage — with wire-level verifiable consent and post-quantum cryptography by default.

Euro-Office/Nextcloud sells the workplace. **Mycelix Sovereign** is what their customers deploy alongside for the operations that cannot touch a foreign cloud, cannot tolerate an un-auditable admin override, and cannot afford to be decrypted in 2030.

### Suite composition (year 1)

| Component | Role | Existing state |
|---|---|---|
| **Xenia** | PAM / verifiable-consent remote support | `xenia-wire 0.2.0-alpha.2` on crates.io |
| **Mycelix Pulse** | PQC-encrypted operations email | Phase 0/1 shipped in worktree `session-pulse-readiness` (18 commits, not merged) |
| **Athena L1** | Consciousness-gated AI triage agent | Symthaea REPL with sandboxed tool-use (read_file, list_dir) |
| **Mycelix Identity** | DID + MFA + VC-based authority | Zome logic tested; 2,100+ cross-cluster dispatch points ready |

### Live regulatory alignment

Three live deadlines — all apply to the Suite, not just PAM:

| Deadline | Driver | Suite hook |
|---|---|---|
| **2026** | NIS2 Article 21 audits (EU) | Xenia consent ledger + Pulse encrypted comms + L1 audit trail |
| **Jan 2027** | CNSA 2.0 for new national-security systems | ML-KEM + Ed25519 hybrid across every component |
| **Jan 2030** | CISA TLS-1.3-or-PQC-successor | Already PQC today across wire + mail |

---

## 2. Naming — decided

**Commercial suite name: Mycelix Sovereign.**

Rejected:
- **Xenia Enterprise** — Xenia is one of four suite components. Trap-name as scope expanded.
- **Verifiable.ops** — punchy for marketing but orphans the Mycelix brand; violates the "tie back to parent/network" constraint.

Component naming inside the suite:

| Internal | Customer-facing | Notes |
|---|---|---|
| `xenia-wire` + `xenia-peer` + admin-side crates | **Xenia** (PAM) | Keeps crates.io brand and permissive license story |
| `mycelix-pulse` | **Pulse** (mail) | Keeps `mail.mycelix.net` brand |
| Symthaea REPL + action executor | **Athena L1** | Rename for enterprise audience — Symthaea stays the research/consciousness brand |
| `mycelix-identity` | **Identity** | Unchanged |

Year-2 expansion slots cleanly: *Mycelix Sovereign* (Suite year 1) → *Mycelix Civic*, *Mycelix Commons* as add-on suites.

### Why "Athena" for L1

"Symthaea" is our consciousness-research brand — load-bearing in academic and community channels. The L1 commercial surface needs a name that doesn't invite "consciousness-gated AI" scrutiny in an enterprise procurement meeting. **Athena** (classical wisdom, unclaimed in SecOps/PAM) lets Symthaea-core evolve inside while the commercial shell stays grounded. Confirm final name W2.

---

## 3. Deployment model — decided

**Self-hosted first.** Zero infrastructure cost to us; honors the sovereign threat model; "customer-managed keys / BYOK / HYOK" compliance language is native rather than bolt-on.

### Year-1 deployment artifacts

1. **NixOS module** — `flake.nix` output `mycelix-sovereign.nixosModules.default`. Declarative config, reproducible, plays to our NixOS depth.
2. **Docker Compose bundle** — for non-NixOS customers. `docker-compose.yml` + per-service images + `.env.example`.
3. **Air-gapped installer** — tarball + offline NixOS flake lock for classified/high-security environments.

### Year-2 "Managed" tier (explicitly not SaaS)

When we offer managed hosting later, it is:

- **Strictly single-tenant** — one customer, one VPS, one key root
- **Customer-paid VPS upfront** — no cross-subsidy, no noisy neighbors, no shared control plane
- **Same deployment artifact** they could have run themselves

This preserves the compliance story (customer keys never touch another tenant's memory) and caps our operational liability.

---

## 4. Licensing — decided

| Layer | License | Rationale |
|---|---|---|
| `xenia-wire` protocol | MIT + Apache-2.0 | Shipped; adoption-first; standards track |
| `xenia-peer` transport | MIT + Apache-2.0 | Same |
| `xenia-handshake` crypto | MIT + Apache-2.0 | Same |
| `xenia-capture` (new, cross-platform screen) | MIT + Apache-2.0 | Protocol-adjacent; broad adoption |
| **`xenia-ledger` (new)** | **AGPL-3.0** | Application-layer; the "verifiable consent" moat |
| **Admin console** (Leptos) | **AGPL-3.0** | Application-layer |
| **Pulse SMTP bridge / admin tools** | **AGPL-3.0** | Application-layer |
| **Athena L1 runtime** | **AGPL-3.0** | Application-layer |
| **Mycelix Sovereign meta-repo** (NixOS module, installer, Docker bundle) | **AGPL-3.0** | Bundles the above |

AGPL-3.0 on the app layer prevents a hyperscaler wrapping the suite as a managed SaaS without contributing back. The permissive wire keeps adoption frictionless — anyone can build a Xenia client on any license terms.

Commercial / dual-license carve-out reserved for customers whose counsel blocks AGPL (standard play); priced in year 2.

---

## 5. Reality check — what we have today

From integration audit (2026-04-19):

| Journey | State | Gap |
|---|---|---|
| DID login | Zome ready, no frontend wires it | ~3 wk |
| PQC email (Pulse) | Phase 0/1 in `session-pulse-readiness` worktree; main has DNA only | ~6 wk (merge + productionize) |
| Xenia remote support | Alpha on crates.io; synthetic frames, ML-KEM placeholder | ~6-8 wk (cross-platform via `scap` — see [ADR 0001](mycelix-sovereign/docs/adr/0001-screen-capture-backend.md)) |
| Athena L1 agent | REPL + tool-use works; no KB, no ticket API | ~4 wk |
| Unified shell / auth | Sensorium is a shell; no cross-cluster auth | ~5 wk |
| Device enrollment | Holon is a sensor bridge; no DID binding | **year 2** |

**Honest Suite-beta estimate: ~17 weeks** (parallel streams; see §6).

---

## 6. Phased execution — ~17 weeks to Suite beta

Two parallel work streams from W1 onward. Pulse integrates into W1 (merge) + W2 (productionize) per user direction.

### W0 — Xenia cross-platform close (5 wk, single stream)

The suite's headline differentiator rests on Xenia. Nothing else is blocked until this ships.

**Cross-platform screen capture — use an existing Rust crate, not bare-metal OS APIs.**

**Spike complete (2026-04-19).** See [ADR 0001](mycelix-sovereign/docs/adr/0001-screen-capture-backend.md). Key findings:

- **`crabgrab` is archived** (read-only since Oct 2024, last commit Jun 2024) → **disqualified**.
- **`xcap`** is active but Wayland support is explicitly "limited" by its maintainer.
- **`scap`** (CapSoftware/scap, used in production by the Cap Loom-alternative) covers all four OSes cleanly: WGC on Windows, ScreenCaptureKit on macOS, PipeWire-native on Linux Wayland, X11 via XDG portal fallback.

**Decision:** `scap` is primary backend; `xcap` is fallback if scap beta proves unstable. Real-hardware validation (FPS / latency / cursor / HDR / permission UX) pending in W0 wk 1-2.

**Deliverables:**

- [x] **W0.1 (completed 2026-04-19):** Backend selection ADR committed ([ADR 0001](mycelix-sovereign/docs/adr/0001-screen-capture-backend.md))
- [ ] **W0.2 (2 days):** Real-hardware validation on Win11 + macOS 14 + GNOME-Wayland + KDE-Wayland + X11; amend ADR 0001 with measurements, or open ADR 0002 if scap fails any target
- [ ] **`xenia-capture` crate** (MIT+Apache) — wraps scap; exposes `FrameProducer` trait aligned to `xenia-wire::RawFrame`
- [ ] **ML-KEM-768 handshake wired** — thread `xenia-handshake` through `xenia-peer` (currently "crate-shipped, wiring-pending" per memory)
- [ ] **Consent state machine enforced** — frame transmission gated on consent transitions; replaces UI mockup at `xenia-viewer-web/www/consent.html`
- [ ] **`xenia-ledger` crate (AGPL)** — append-only, hash-chained, signed consent + session events; ships with external verifier binary
- [ ] **`xenia-viewer-web` wired to real sessions** — drop synthetic RGBA, render live capture

**Exit criteria:**
- Linux admin views macOS employee's screen via `xenia-viewer-web` (cross-platform proven on the critical OS pair)
- External auditor independently verifies a session's ledger signatures using only the published `xenia-ledger` verifier
- ML-KEM handshake trace captured and matches `xenia-wire` SPEC draft-03

### W1 — Identity + Admin Console + Pulse merge (4 wk, two streams)

**Stream A — Identity + Admin Console**

- [ ] DID login on a Leptos app (first frontend that actually calls `resolve_did` + persists session); start with Xenia admin console
- [ ] Cross-cluster auth context threaded via `mycelix-bridge-common`
- [ ] Admin console MVP: device list, active sessions, historical ledger review, policy CRUD
- [ ] Standards mapping doc: **NIS2 Art. 21** + **ISO/IEC 27001 A.5.15** + **NIST SP 800-53 AC-17 / AC-6(9)**

**Stream B — Pulse merge to main**

- [ ] Merge `session-pulse-readiness` worktree → main (18 commits, Phase 0+1 per memory)
- [ ] Resolve getrandom / `mycelix-zkp-core` WASM blocker (env blocker #7 from the Pulse memory)
- [ ] Run H1–H5 runbook for two-agent end-to-end send/receive on main
- [ ] Remove `#[ignore]` on `phase0_alice_sends_bob_receives`; CI green

**Exit criteria:**
- **A:** Admin logs in with DID, sees three test devices, reviews a session ledger, exports audit report
- **B:** Alice sends mail to Bob across two machines; both have sovereign DIDs; main branch CI green

### W2 — Athena L1 + Pulse productionization (4 wk, two streams)

**Stream A — Athena L1 agent**

- [ ] **KB connector** — wire `mycelix-knowledge` (claims, graph, query, factcheck) into Athena system prompt; extend `top_grounded_facts()` to KB-backed answers
- [ ] **Ticket API** — REST shim accepts email-forward / webhook → Athena turn
- [ ] **Escalation handoff** — Athena emits `{status: escalate, context}` → routes to admin console
- [ ] **Per-tenant sandbox** — harden `/tmp/symthaea/repl-session/` → `/var/lib/mycelix-sovereign/<tenant>/athena/` + network policy
- [ ] **Enterprise name confirmed** — Athena vs alternatives, shipped

**Stream B — Pulse productionization**

- [ ] **PQC envelope sealing** in mail-bridge (main currently stores plaintext CIDs; design exists — wire it)
- [ ] **Epoch PQ ratchet state machine** — key rotation between messages
- [ ] **Real SMTP bridge merged to main** (exists in worktree per memory; needs finalization)
- [ ] **Two real users across different machines** exchange mail (beyond conductor sweettest)

**Exit criteria:**
- **A:** Ticket arrives → Athena triages using KB → resolves or escalates with context preserved
- **B:** Two non-lab users exchange a message; envelopes PQC-sealed; ratchet advances; replay-resistant

### W3 — Suite integration + design partners (4 wk)

- [ ] **Unified auth across all four components** — single DID-backed session for Xenia + Pulse + Athena + Identity
- [ ] **NixOS module** — `flake.nix` output bundling all four services + shared config schema
- [ ] **Docker Compose bundle** — for non-NixOS deployments
- [ ] **Threat model** — STRIDE across the full Suite
- [ ] **FIPS 203 / 204 compliance statement** — "where we are, path forward" (not the cert)
- [ ] **SOC 2 Type II readiness gap analysis** — not the cert, the playbook
- [ ] **External cryptographer review** — Xenia wire + Pulse envelope + ledger chain
- [ ] **3 design-partner conversations** — EU public sector / healthcare-or-legal / privacy-forward org
- [ ] **Public demo** — `sovereign.mycelix.net` with live admin console + recorded consent ceremony

**Exit criteria:** 3 letters of interest or scoped paid pilots; NixOS module installs on a fresh VM in <15 min; external crypto review with no blocker findings.

**Total: ~17 weeks to Suite beta.**

---

## 7. Year-2 expansion

Once the Suite has anchor customers:

- **Device enrollment + posture** — MDM replacement tied to DID (Holon evolves from sensor bridge)
- **Praxis** — NICE cybersecurity training; customer admins keep certifications current via living-credential decay
- **Craft** — living certifications for in-house SecOps teams
- **Commons / Governance / Civic** — the full "sovereign workplace" story re-emerges here for customers wanting the platform
- **Managed tier** — single-tenant VPS offering for customers who want sovereignty without ops burden

Each layers on top of an established customer base rather than leading the pitch.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Holochain enterprise track record** | Lead with Xenia (Apache/MIT, crates.io, standards-aligned) + PQC. Holochain becomes an implementation detail. |
| **PQC not yet mandatory** | Use "harvest-now-decrypt-later" urgency (NSA/NIST verbatim). CNSA 2.0 Jan 2027 close enough for FY27 procurement. |
| **AI agent needs SOC 2 / ISO 42001** (healthcare/fintech) | Position "architectural protection (data never leaves substrate)" as stronger than policy-based SOC 2. Pursue SOC 2 Type II in year 2. |
| **"Consciousness-gated" scares CISOs** | Rename to "adaptive trust tiers" / "continuous-authorization PAM" for enterprise. Keep movement framing for community/philosophy channels. |
| **"Decentralized = no throat to choke"** | Per-tenant authority anchor + verifiable consent ledger = *more* accountability. Lead with third-party-verifiable ledger. |
| **Cross-platform capture quality varies by OS** (Wayland restrictions, macOS permissions UX) | W0 wk-1 spike quantifies per-OS FPS/latency/permissions. Fall back to "Linux+macOS first, Windows W3.5" if Windows capture fidelity is poor. |
| **Pulse merge may surface integration bugs invisible in worktree** | W1 Stream B starts with main CI green as hard gate. If merge breaks, halt Stream B; Stream A keeps moving. |
| **AGPL blocks some enterprise buyers** | Dual-license option for app layer (commercial exception, priced). Announce with open-source launch. |
| **"Secure Operations Suite" confused with SIEM** (Splunk, Rapid7) | We're not a SIEM; we're the substrate they feed. Position against BeyondTrust/CyberArk (PAM) and Proton Business (secure comms). |
| **Concurrent-session monorepo hazards** | Every sub-phase closes with commits. Use worktrees when editing shared `bridge-common`, `zkp-core` (memory: proven hazard). |

---

## 9. Metrics to track

| Metric | Target (end of W3) |
|---|---|
| Xenia sessions end-to-end per week | >10 |
| Pulse messages end-to-end (non-lab) per week | >50 |
| Athena L1 auto-resolve rate (simulated tickets) | >40% |
| Unified auth coverage (4 components) | 100% |
| NixOS module install on fresh VM | <15 min |
| NIS2 Art. 21 control coverage | >70% |
| External cryptographer review | Complete, no blocker findings |
| Design-partner conversations | 3+ |
| Published standards-mapping doc | 1 |

---

## 10. Resolved decisions

| Question | Decision |
|---|---|
| Scope | **Suite** (PAM + Pulse + Athena L1 + Identity), not PAM-only wedge |
| Suite name | **Mycelix Sovereign** |
| L1 agent name | **Athena** (confirm W2) |
| RDP producer | Cross-platform Windows/macOS/Linux via **`scap`** (primary) / `xcap` (fallback); `crabgrab` disqualified — see [ADR 0001](mycelix-sovereign/docs/adr/0001-screen-capture-backend.md) |
| License — protocol | MIT + Apache-2.0 |
| License — app layer | **AGPL-3.0** |
| Deployment — year 1 | Self-hosted first (NixOS module + Docker bundle) |
| Deployment — year 2 | Managed = single-tenant dedicated VPS (never multi-tenant SaaS) |

---

## 11. Still-open questions

1. **Windows agent priority** — if xcap Windows backend delivers <15 FPS or has awkward permission UX, is Windows a Suite-beta blocker or year-2?
2. **Where does the ledger verifier run?** — Customer-hosted tool (simplest, in-scope now) vs neutral third party (stronger claim, partnership work) vs Holochain DHT (platform-native, more moving parts)
3. **FIPS 203 formal validation timeline** — "working toward" is sellable for 12-18 months; real validation is $$$. Decide path in W3.
4. **Commercial / dual-license pricing** for the AGPL exception — year-2 question but capture inbound interest from day one.
5. **Athena training / KB seed data** — what's in the KB day one? Default `mycelix-knowledge` facts? Customer-imported SOPs? Public NIST/ISO docs bundled?
6. **Design-partner shortlist** — who specifically to contact? Need 10 named orgs across three categories.

---

## 12. Next actions (this week)

- [x] **Stand up `mycelix-sovereign` meta-repo** at `/srv/luminous-dynamics/mycelix-sovereign/` — AGPL LICENSE, NixOS flake skeleton, README, NixOS module skeletons for all 4 components (2026-04-19)
- [x] **xcap vs crabgrab spike** — ADR 0001 committed; `scap` selected, `crabgrab` disqualified (2026-04-19)
- [ ] **Real-hardware validation** of `scap` on 5-OS matrix — amend ADR 0001 with measurements
- [ ] **Draft NIS2 Art. 21 mapping skeleton** — 1-day doc; concrete deliverable for W1
- [ ] **Identify design-partner shortlist** — name 10 orgs (3 per category)
- [ ] **`xenia-capture` crate scaffold** — new MIT+Apache crate wrapping scap behind a `FrameProducer` trait
- [ ] **`xenia-ledger` crate scaffold** — new AGPL crate (location TBD: standalone or under `xenia-peer/`)

---

## Appendix A — rev 1 → rev 2 changelog

- **Scope**: "Xenia PAM" wedge → "Mycelix Sovereign" Secure Sovereign Operations Suite (4 components)
- **Pulse**: deferred to year 2 (rev 1) → integrated W1 merge + W2 productionize (rev 2)
- **RDP producer**: Linux-only (rev 1) → cross-platform Win/Mac/Linux via xcap or crabgrab (rev 2)
- **License**: open question → AGPL-3.0 app layer, MIT+Apache protocol
- **Deployment**: open question → self-hosted first; year-2 managed is single-tenant VPS only
- **Timeline**: 14 wk PAM beta → 17 wk Suite beta
- **Commercial name**: working "Xenia PAM" → **Mycelix Sovereign**
- **L1 name**: "Symthaea L1" → **Athena L1** (enterprise surface; research brand preserved internally)
- **Screen capture backend** (ADR 0001): "xcap or crabgrab" (rev 2 open) → **`scap` primary / `xcap` fallback**; crabgrab archived → disqualified

## Appendix B — sources

Research agents, 2026-04-19:
- Integration audit (6 user-journey assessments)
- Market research (8 competitors; NIS2 + CNSA 2.0 + CISA deadlines; buyer language)

External (from market-research agent):
- [Computerworld — Nextcloud sovereignty](https://www.computerworld.com/article/4064116/)
- [Tech.eu — Euro-Office launch](https://tech.eu/2026/03/27/europe-builds-microsoft-compatible-euro-office-to-reclaim-digital-sovereignty/)
- [ActiveMind Legal — Microsoft Sovereign Cloud](https://www.activemind.legal/guides/microsoft-sovereign-cloud/)
- [Diamatix — NIS2 audits 2026 readiness](https://diamatix.com/nis2-audits-2026-readiness/)
- [PRNewswire — $15B post-quantum migration](https://www.prnewswire.com/news-releases/the-15-billion-post-quantum-migration-302730679.html)
- [Happenings Community — Holochain 2025 reality check](https://happeningscommunity.substack.com/p/the-holochain-ecosystem-in-2025-a)
- [Fini Labs — SOC 2 AI support regulated 2026](https://www.usefini.com/guides/best-soc-2-compliant-ai-support-platforms-regulated-industries-2026)

Internal:
- `CLAUDE.md`
- `memory/xenia_*.md` (Xenia arc, 20+ commits Apr 18)
- `memory/pulse_phase0_shipped_apr19.md` (18-commit worktree, Phase 0+1)
- `memory/mycelix_demo_readiness_apr18.md` (Type 1 civ substrate north-star)
