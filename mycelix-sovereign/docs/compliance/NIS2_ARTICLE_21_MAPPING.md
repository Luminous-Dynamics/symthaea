# NIS2 Article 21 — Mycelix Sovereign Control Mapping (skeleton)

**Status:** Draft v0.1 — skeleton
**Date:** 2026-04-19
**Audience:** EU design-partner CISOs, procurement, compliance auditors preparing NIS2 Art. 21(2) evidence

**What this is:** a per-clause mapping of NIS2 Article 21(2) requirements (a)–(j) against Mycelix Sovereign's delivered and planned controls. Coverage, evidence pointers, and gaps are stated honestly per clause.

**What this is not:** a certification. Mycelix Sovereign is pre-alpha. No SOC 2, no ISO 27001, no third-party penetration test has been performed yet. This document is the pre-procurement gap analysis a prospect can use to scope their own Art. 21 evidence package — not a substitute for an auditor-signed report.

## Scope

| | |
|---|---|
| **Product version** | Pre-alpha (W0 of plan) |
| **Directive** | (EU) 2022/2555 — NIS2 |
| **Article** | 21(2), the 10 risk-management measures ("essential" and "important" entities alike) |
| **Transposition deadline** | 17 October 2024 (already past); audit cycle 2026 |
| **ENISA guidance** | *Guidelines on security measures for operators of essential services* (Sept 2023) |

## Our component map (reference)

| Component | License | Role |
|---|---|---|
| `xenia-wire` / `xenia-peer` / `xenia-handshake` / `xenia-capture` / `xenia-inject` | Apache-2.0 OR MIT | Consent-native remote access protocol + transport + PQC handshake + cross-platform capture + input injection |
| `xenia-ledger` | **AGPL-3.0-or-later** | Append-only, blake3-chained, Ed25519-signed verifiable consent ledger |
| `mycelix-pulse` | Apache-2.0 OR MIT (core) / AGPL (admin) | PQC-encrypted operations email |
| `mycelix-identity` | Apache-2.0 OR MIT | DID + MFA + verifiable credentials |
| `symthaea-core` (Athena L1 runtime) | AGPL-3.0-or-later (commercial wrapper) | Consciousness-gated AI triage with sandboxed tool-use |
| `mycelix-sovereign` meta-repo | **AGPL-3.0-or-later** | NixOS module + Docker bundle + installer bundling all of the above |

---

## Article 21(2) mapping

### (a) Policies on risk analysis and information system security

**Required:** entities shall have documented policies on risk analysis and on information-system security.

**Coverage (suite-provided):**
- *Controls, not policies.* Mycelix Sovereign ships enforcement primitives; **the customer authors the policy text.** Suite provides:
  - `mycelix-identity` tier configuration (5-tier privilege model: Observer → Guardian) backing risk-weighted authority
  - `xenia-ledger` as the evidence backbone for policy-compliance claims
  - `athena` runtime configured with customer-provided SOPs as first-order knowledge

**Gap:** The suite does not ship a "default risk-analysis policy" or "default infosec policy" template. Design-partner engagement will surface whether we should add reference templates (ENISA-aligned) in W3+.

**Evidence pointer:** None yet. A reference-policy-template contribution is a candidate for the design-partner phase.

---

### (b) Incident handling

**Required:** policies and procedures covering detection, analysis, containment, response, recovery, and post-incident review.

**Coverage:**
- **Detection:** `xenia-ledger` records every consent event with cryptographic integrity. Anomalies (unexpected Revocations, `ConsentProtocolViolation` entries) are first-class detectable events. Third-party auditor can run `Verifier::verify_chain` offline.
- **Analysis:** Athena L1 runtime ingests ledger + Pulse threads + mycelix-knowledge KB → structured triage output with consciousness-gated authority tier.
- **Containment:** Xenia consent state machine halts frame transmission on any `ConsentViolation` variant (`RevocationBeforeApproval`, `ContradictoryResponse`, `StaleResponseForUnknownRequest` per xenia-wire SPEC draft-03 §12.6).
- **Recovery:** Append-only ledger by construction — no incident can rewrite the past. Post-incident analysts rebuild the session timeline from cryptographically-linked entries.
- **Post-incident review:** ledger entries are portable and externally verifiable.

**Gap:** No built-in alerting / paging integration (e.g., PagerDuty webhooks, SIEM forwarders). These are year-2 integrations; the customer's existing IR tooling bridges to the ledger via its portable serialization.

**Evidence pointer:** `xenia-peer/crates/xenia-ledger/src/lib.rs` (chain + Verifier + 9 tests covering every tamper vector).

---

### (c) Business continuity — backup management, disaster recovery, crisis management

**Required:** backup, restore, DR planning.

**Coverage:** *Intentionally out of scope for the suite core.* Ledger entries and Pulse envelopes are portable serializable artifacts — customer's existing backup stack (Borg, Restic, Veeam, Tanium, etc.) covers them like any other filesystem data.

**Gap:** No bundled backup/DR primitive. A reference `restic` / `borg` backup policy for `/var/lib/mycelix-sovereign/` is a 1-page doc we should ship in W3.

**Evidence pointer:** (Pending — W3 deliverable.)

---

### (d) Supply chain security

**Required:** security of suppliers and service providers, including assessment and monitoring.

**Coverage:**
- **All protocol-layer dependencies** (`xenia-wire` and its deps) are public Apache/MIT crates, auditable at crates.io with content-addressed git provenance. PR [CapSoftware/scap#183](https://github.com/CapSoftware/scap/pull/183) demonstrates live supplier-engagement when upstream bugs block us.
- **AGPL on app-layer components** means any downstream redistributor must ship source modifications back, creating a contractual-level supply-chain transparency requirement.
- **Deterministic builds via Nix** — the NixOS module's `flake.lock` pins every transitive dep to a content hash.
- **`mycelix-supplychain`** (year-2 cluster, 8 zomes) will be the portable supply-chain provenance layer across products.

**Gap:** No SBOM export yet. `cargo cyclonedx` / `cargo sbom` integration is a W3 deliverable; trivial to add.

**Evidence pointer:** `flake.lock` + SBOM export pending.

---

### (e) Security in network and information systems acquisition, development, and maintenance, including vulnerability handling and disclosure

**Required:** secure dev lifecycle + coordinated vuln disclosure.

**Coverage:**
- **SDLC:** The suite is developed with mandatory `unsafe_code = "deny"` at workspace level (see `xenia-peer/Cargo.toml:31`), `#![deny(unsafe_code)]` on security-sensitive crates (xenia-ledger). Reviewed commits only to `main`; `.githooks/pre-commit` cross-project commit guard in the monorepo.
- **CVD policy:** Apache/MIT crates already have `SECURITY.md` in upstream repos (see `xenia-peer/SECURITY.md` via ROADMAP); 90-day coordinated disclosure policy.
- **Vulnerability handling:** upstream-dep vulnerabilities surface via `cargo audit` (to be wired into CI in W2+).
- **Fuzzing:** `xenia-wire` ships cargo-fuzz targets including `fuzz_observe_consent`.

**Gap:** No public vulnerability-report triage process yet (security@ email, PGP key, response SLA). Committed to establish one in W3 before first design-partner deployment.

**Evidence pointer:** `xenia-peer/SECURITY.md` (upstream); `mycelix-sovereign/SECURITY.md` — *to be written W3*.

---

### (f) Policies and procedures to assess the effectiveness of cybersecurity risk-management measures

**Required:** measurable evaluation of whether the other controls actually work.

**Coverage:** **This is the `xenia-ledger` core claim.** Every privileged session produces a third-party-verifiable, tamper-evident record. Assessing control effectiveness reduces to:

1. `Verifier::verify_chain(&entries, &operator_public_key)` — integrity
2. Count `ConsentKind::Violation` entries in a time window — anomaly rate
3. Compare declared-policy privilege tier (mycelix-identity) against realized privilege grants (ledger) — drift detection

An auditor with the public key can do all three offline with no operator cooperation.

**Gap:** No built-in dashboard for these queries; admin console (W1 deliverable) will provide them.

**Evidence pointer:** `xenia-peer/crates/xenia-ledger/src/lib.rs:Verifier` + 9 tamper-test cases (`cargo test -p xenia-ledger`).

---

### (g) Basic cyber hygiene practices and cybersecurity training

**Required:** user-facing training + hygiene procedures.

**Coverage:**
- **In-suite:** Athena L1 runtime can surface contextual hygiene prompts at the moment of a privileged action (e.g., "you're about to install X on an unattended production box — proceed?"). Consciousness-gated so it only fires above the user's tier.
- **Training content:** `mycelix-praxis` (year-2) ships NICE Cybersecurity framework curriculum (K-to-PhD coverage per project CLAUDE.md). Customers can assign specific modules; completion is recorded as a living `mycelix-craft` credential that decays via the Ebbinghaus curve if not refreshed.

**Gap:** `mycelix-praxis` integration with Athena's policy engine is a year-2 deliverable. Today, a customer bolts their existing KnowBe4 / SANS / internal LMS alongside the suite.

**Evidence pointer:** `mycelix-praxis/apps/leptos/` (NICE curriculum) + `mycelix-craft` (living credentials) — year-2.

---

### (h) Policies and procedures regarding the use of cryptography and, where appropriate, encryption

**Required:** crypto governance + encryption where risk warrants.

**Coverage:** **Non-negotiable core property of the suite — PQC-by-default at every layer.**

| Layer | Cryptography |
|---|---|
| Xenia wire handshake | Ed25519 + **ML-KEM-768 hybrid** (via `xenia-handshake`, RustCrypto `ml-kem 0.3.0-rc.2`) |
| Xenia wire transport | ChaCha20-Poly1305 AEAD (SPEC §5) |
| Xenia consent ledger | Ed25519 signatures + blake3 hashes |
| Pulse email envelopes | PQC hybrid (Kyber + classical; epoch ratchet per `PULSE_READINESS_PLAN`) |
| Identity DIDs | Ed25519 anchor keys |

**CNSA 2.0 alignment:** ML-KEM-768 meets NSA CNSA 2.0 key-establishment requirement for new national-security systems (deadline Jan 2027). Ed25519 is NOT CNSA 2.0 compliant for *signatures* (CNSA 2.0 requires ML-DSA / SLH-DSA / LMS / XMSS for signatures); ML-DSA integration is a year-2 deliverable tracked in `xenia-ledger/README.md` ("PQC signature option").

**Gap:** Signatures are classical (Ed25519) today — CNSA 2.0 sensitive-system deployments need PQC signatures. Year-2.

**Evidence pointer:** `xenia-peer/crates/xenia-handshake/src/lib.rs`; `xenia-wire` SPEC.md draft-03 §5.

---

### (i) Human resources security, access control policies, and asset management

**Required:** personnel-level security + RBAC + asset inventory.

**Coverage:**
- **HR security:** Customer responsibility; suite enforces downstream.
- **Access control:** `mycelix-identity` 5-tier privilege model (Observer → Seeker → Contributor → Steward → Guardian) with configurable vote weights (default / constitutional / budget / emergency presets per `VoteWeightConfig`). Sub-Passport automatic effective-tier recovery (6h cooldown, 3:1 correction ratio). Every privilege elevation is recorded in `xenia-ledger`.
- **Asset management:** `mycelix-commons` + `mycelix-civic/robotics-dispatch` (RoboticAsset entity) cover physical + digital asset registries in the broader Mycelix platform; narrow IT-asset subset for Sovereign is a year-2 cluster (device-enrollment track).

**Gap:** Device enrollment with DID-binding ("Holon evolves from sensor bridge") is explicitly deferred to year 2 in the plan.

**Evidence pointer:** `mycelix-identity/` zomes (13); `mycelix-identity/dna/` entry types.

---

### (j) Use of MFA or continuous authentication, secured voice/video/text communications, and secured emergency communication

**Required:** MFA + secure comms channels.

**Coverage:**
- **MFA:** `mycelix-identity` enforces MFA by default (NixOS module option `services.mycelix-sovereign.identity.mfaRequired = true`). Configurable off for dev/bootstrap only.
- **Continuous authentication:** the consciousness-gated tier system (renamed **adaptive trust tiers** for enterprise) **is continuous authentication.** A privilege grant is not binary; it is recomputed per action against the operator's current 4D reputation profile (identity / reputation / community / engagement). A tier decay during a live session reduces authority mid-action.
- **Secured text:** `mycelix-pulse` (PQC email), `mycelix-civic/resonance-feed` (encrypted feed), `mycelix-hearth` messaging.
- **Secured voice/video:** Xenia is a secured *remote-access* channel with frame-level consent; it is not yet a general voice/video conferencing platform. A `mycelix-meet` cluster is deferred to year-2+.
- **Secured emergency:** Xenia's consent-revocation path surfaces as a red-button for a user to kill an active admin session; the revocation is cryptographically recorded. No out-of-band emergency channel yet.

**Gap:** Voice/video conferencing absent. `mycelix-meet` is an enterprise-request-driven year-2 cluster.

**Evidence pointer:** `mycelix-sovereign/nixos-modules/identity.nix` (mfaRequired=true default); `mycelix-identity/` (tier + decay logic).

---

## Coverage summary

| Clause | Summary | Status |
|---|---|---|
| (a) Risk policies | Controls provided; customer authors policies | Partial — need templates |
| (b) Incident handling | Ledger + consent state machine + Athena triage | Strong — missing alert integrations |
| (c) Business continuity | Portable artifacts; reuse customer backup stack | Partial — need ref backup policy |
| (d) Supply chain | Auditable deps, Nix lockfiles, upstream engagement | Strong — missing SBOM export |
| (e) SDLC + vuln handling | `unsafe_code=deny`, fuzzing, Apache-SECURITY.md | Partial — need public CVD process |
| (f) Effectiveness assessment | **`xenia-ledger` is the core deliverable** | Strong |
| (g) Hygiene + training | Athena in-moment; Praxis curriculum year-2 | Partial — year-2 full |
| (h) Cryptography | **PQC hybrid at every layer (KEM)**; classical sigs | Strong on KEMs; gap on sigs |
| (i) HR + access control + asset | 5-tier privilege model + Sub-Passport decay | Partial — device enrollment year-2 |
| (j) MFA + continuous auth + secure comms | MFA-default, tier-based continuous auth, PQC email | Partial — no voice/video |

**Honest headline:** Mycelix Sovereign provides strong, novel coverage for (b), (f), (h-KEM) — the verifiable-consent + PQC differentiation. Clauses (c), (d-SBOM), (g), (j-voice) have visible year-1 or year-2 gaps we are not hiding.

---

## What a design-partner engagement looks like

This document becomes a two-column worksheet per prospect:

| NIS2 Art. 21 clause | Mycelix Sovereign coverage | Prospect's existing coverage (gap/overlap) | Joint deliverable for beta |
|---|---|---|---|
| (b) incident | ledger + Athena | SIEM (Splunk) | ledger → Splunk forwarder |
| (c) backup | portable artifacts | Veeam | ref `backup-mycelix-sovereign.md` |
| ... | ... | ... | ... |

Each beta deployment closes one or more rows of the "joint deliverable" column. The prospect gets real Art. 21 coverage; we get product-shaping feedback.

---

## References

- [NIS2 Directive (EU) 2022/2555 — official text (EUR-Lex)](https://eur-lex.europa.eu/eli/dir/2022/2555/oj)
- [ENISA — *Guidelines on security measures for operators of essential services* (2023)](https://www.enisa.europa.eu/publications/guidelines-on-security-measures)
- [BeyondTrust — *Address the NIS2 Directive with Privileged Access Management*](https://www.beyondtrust.com/resources/whitepapers/address-the-nis2-directive-with-privileged-access-management)
- [Diamatix — *NIS2 audits 2026 readiness*](https://diamatix.com/nis2-audits-2026-readiness/)
- MYCELIX_SOVEREIGN_PLAN.md §1 (live regulatory deadlines)
- ADR 0001 (screen capture backend)
- `xenia-peer/crates/xenia-ledger/README.md`
- `xenia-wire/SPEC.md` draft-03
