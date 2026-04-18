# Mycelix × Nation-State Coexistence

Mycelix is architected as a **post-state civilization OS**. The constitution (`SOVEREIGN_PROFILE.md`, `mycelix-governance/ANTI_TYRANNY_DESIGN.md`) deliberately rejects state-style single-axis metrics and geographic jurisdiction. Consciousness-tier gating replaces geography.

Individual users, however, live under nation-states **today**. Without designed answers for tax reporting, jurisdiction verification, and government-ID interop, early adopters face real coercion (audits, prosecution, banking exclusion) that Mycelix would otherwise have no designed response to. Ad-hoc responses under duress produce worse outcomes than principled pre-designed ones.

This document describes the three **opt-in** extensions that let a Mycelix user coexist with their current nation-state without corrupting Mycelix's core consciousness-gated, pseudonymous, post-state architecture.

---

## The non-negotiables

**State-interop lives in separate clusters, never in core zomes.**

1. Governance weight never reads legal-ID credentials.
2. The default DID (`did:mycelix:primary`) never holds government IDs.
3. Core finance (MYCEL / TEND / SAP) never depends on fiat oracles.
4. Nothing here is mandatory to use Mycelix — a user who ignores every extension loses no existing capability.

If any of these slip, the architecture is captured.

---

## The dual-DID rule

```
did:mycelix:primary  ──  governance, MYCEL, TEND, social graph, content, vote weight
did:mycelix:legal    ──  ONLY state-facing bridges:
                           - tax export
                           - KYC'd fiat gateway
                           - regulated-industry credentials
                           - (future) court-order response
```

The two DIDs share **no linkability on-chain**. Any cross-link is **ZKP-only**: the primary DID can prove "I also control a legal DID that holds credential X from issuer Y" without revealing *which* legal DID.

This is the cryptographic airlock that keeps state reach from leaking into consensus.

---

## The three extensions

### 1. Tax export (extends existing `mycelix-workspace/observatory/src/lib/tax-export.ts`)

Was already TEND-only + SA-SARS-specific (259 lines of production TS). Extended to:

- Cover SAP, marketplace, attribution receipts (not just TEND).
- Add a user-held **classification overlay** (Gift / BarterLabor / Wage / Distribution / CommonsContribution / InternalTransfer / Sale / Purchase).
- Add a **FMV anchor** — per-transaction fair-market-value snapshot. For SAP, requires a user-selected fiat feed (no hardcoded default).
- Add a **novel-position** framework for four un-precedented tax questions: TEND as barter income, demurrage treatment, MYCEL accrual recognition, compost redistribution as constructive receipt. Each offers multiple stances with cite-able legal reasoning; user picks, output embeds the choice.
- Add jurisdiction templates: US IRS Schedule C, SA SARS IT12 (extended from existing). UK + EU DAC8 in a second wave.
- Enforce the dual-DID guard: `generateTaxExport` refuses `did:mycelix:primary`.

### 2. Jurisdiction proof (extends `crates/mycelix-zkp-core/`)

Privacy-preserving location proof. Verifier learns "user is in jurisdiction set S" — nothing more.

- Reuses existing Winterfell STARK range-proof circuit (`src/circuits/range_proof.rs`, ~1ms prove / 0.3ms verify / 7.5KB proofs).
- Reuses the sovereign-proof commitment pattern (`src/sovereign.rs`).
- Adds `src/circuits/jurisdiction_proof.rs`, `src/location_attestation.rs`, `src/jurisdiction_registry.rs`.
- Five attestation trust tiers (T0 self → T1 phone-GPS → T2 civic-bridge → T3 hardware-TEE → T4 notary). **Default minimum T1** (avoids bootstrap paradox). Verifiers may require higher.
- Location never reaches the DHT — proofs are presented peer-to-peer in session.
- **Trust model caveat**: the ZKP proves *containment of the attested value*, not that the attestation is truthful. A user can fake GPS. Security comes from the attester, not the proof. Documented loudly.

### 3. Lawful-identity cluster (`mycelix-lawful-identity/`)

The legal-DID namespace. Isolated from primary DID. No cross-indexing.

- Extends the existing `mycelix-identity/crates/eidas-zkp/` (W3C VC 2.0 + DASTARK + Dilithium5 + Merkle selective disclosure already present). Passport / mDL / SSN credentials are just eIDAS credentials with specific claim shapes — see `mycelix-lawful-identity/docs/GOV_ID_CLAIM_SHAPES.md`.
- `zomes/legal-did/` — isolated DID namespace, distinct DHT partition via role.
- `zomes/issuer-trust-tier/` — three tiers (Sovereign / RegulatedIntermediary / Peer). No tier ever influences governance.
- `zomes/cross-did-zkp/` — primary proves control of a legal DID holding X credential without revealing which legal DID. Fresh verifier-supplied nonce mandatory.
- `cli/` — first-run onboarding surfaces both honest caveats below.
- `docs/THREAT_MODEL.md` — four adversary vectors explicitly addressed.

---

## Threat model — the honest caveats

The dual-DID design mathematically neutralizes two vectors and explicitly does **not** cover two others:

| Vector | Status | Why |
|--------|--------|-----|
| 1. Cryptographic leakage in the ZKP | **Mitigated** | Link-resistance test asserts legal DID absent from public inputs; only `{issuer_pk, claim_predicate, fresh_nonce}` exposed. |
| 2. Replay / correlation | **Mitigated** | Fresh verifier-supplied nonce per proof; reused nonces rejected. |
| 3. Network metadata (IP correlation) | **NOT mitigated** | Same host → same gossip IP → trivially linkable. True unlinkability requires separate physical agents or Tor/I2P. |
| 4. Device compulsion / rubber-hose | **NOT mitigated** | Local keystore holds both keys. Physical seizure + compelled unlock links them instantly. |

Protection is against **mass passive on-chain surveillance**. Not against **targeted endpoint compromise or network-layer deanonymization**.

---

## What this explicitly does NOT add

1. No jurisdiction-tagged governance weight.
2. No court-order key disclosure protocol in core (parked; separate `mycelix-lawful-response` cluster if ever built).
3. No KYC on any primary-DID function.
4. No sanctions list integration anywhere (sanctions are a sovereignty claim; delegating to OFAC would hand the state a chokepoint).
5. No constitutional amendment — none of this touches governance rules.
6. No default-on state interop — every piece requires explicit user opt-in.

---

## Implementation status — 14 commits on main, live on shared conductor

As of 2026-04-18 the three subsystems are complete and the happ is
installed + enabled on the Mycelix shared ecosystem conductor. The
`lawful-id` CLI exercises all seven zome functions via `--live`.

### File map (what shipped)

| Path | What |
|------|------|
| `mycelix-workspace/observatory/src/lib/tax-export.ts` | Existing SA-SARS module, extended with dual-DID guard |
| `mycelix-workspace/observatory/src/lib/classification.ts` | 8-category transaction classification overlay (22 tests) |
| `mycelix-workspace/observatory/src/lib/novel-position.ts` | Cite-able tax-position framework, 4 novel questions (12 tests) |
| `mycelix-workspace/observatory/src/lib/fiat-feeds.ts` | 5 feed adapters, user-selected, client-side only (22 tests) |
| `mycelix-workspace/observatory/src/lib/fmv-anchor.ts` | Per-transaction FMV, immutable once set (18 tests) |
| `mycelix-workspace/observatory/src/lib/coexistence-golden-path.test.ts` | End-to-end integration test (13 tests) |
| `crates/mycelix-zkp-core/src/circuits/jurisdiction_proof.rs` | STARK jurisdiction containment proof (8 tests) |
| `crates/mycelix-zkp-core/src/location_attestation.rs` | T0-T4 attestation trust tiers (8 tests) |
| `crates/mycelix-zkp-core/src/jurisdiction_registry.rs` | US + SA seed registries (9 tests) |
| `mycelix-lawful-identity/` | Cluster — 3 zomes, `lawful-id` CLI with live conductor wiring |
| `mycelix-lawful-identity/tests/src/link_resistance.rs` | 1000-proof Vector-1 sweep (10 tests) |
| `mycelix-lawful-identity/tests/src/unlinkability.rs` | Vector-2 byte-distinctness + chi-squared (5 tests) |
| `mycelix-lawful-identity/QUICKSTART.md` | End-to-end install + CLI walkthrough |
| `mycelix-lawful-identity/docs/THREAT_MODEL.md` | Four adversary vectors with honest caveats |
| `mycelix-lawful-identity/docs/GOV_ID_CLAIM_SHAPES.md` | Claim-key conventions for passport / mDL / SSN |

### What changed vs the original plan

- **Subsystem 1 — fiat feeds**: originally planned as a Rust zome module inside `mycelix-finance/zomes/price-oracle/`, but WASM zomes can't do HTTP and the price-oracle is TEND-focused. Reshaped as client-side TypeScript modules in observatory — user-local by design, no central fiat-feed choice ever lands on chain.
- **Tax-export cluster**: the original plan called for a new `mycelix-tax-export` Rust cluster. Superseded by extending the existing `mycelix-workspace/observatory/src/lib/tax-export.ts` (SA-SARS, tested, production). No parallel cluster scaffolded.
- **Gov-ID schemas crate**: originally planned. Superseded by extending the existing `mycelix-identity/crates/eidas-zkp/` which already handles W3C VC 2.0 + DASTARK + Dilithium5 + Merkle selective disclosure. Gov-ID credentials are just eIDAS credentials with specific claim keys; see `mycelix-lawful-identity/docs/GOV_ID_CLAIM_SHAPES.md`.

### Test totals

- **Rust**: 56+ structural tests (8 jurisdiction + 8 attestation + 9 registry + 10 link-resistance + 5 unlinkability + 1 smoke + 6 CLI + additional zome-level unit tests).
- **TypeScript**: 292 tests across 14 observatory test files, 0 failures.
- **Live conductor**: all 7 zome functions exercised via `lawful-id --live` against the shared ecosystem conductor.

Port reservations (in `.claude/rules/PORTS.md`):
- 8132 — tax-export frontend (extends observatory)
- 8133 — mycelix-lawful-identity frontend

---

*Coexistence without capture. The airlock holds as long as nothing drills through it.*
