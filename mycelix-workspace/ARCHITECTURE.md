# Mycelix Architecture

This is the architecture document that was missing: a from-scratch
explanation of how Mycelix's cluster/bridge/gating design actually works,
written for someone who has never seen this repo before. Everything here was
verified against the current source as of 2026-07-03 (file paths and
counts may drift — re-check before citing exact numbers in a PR).

If you only read one other doc, read `mycelix-workspace/CLAUDE.md`'s "hApp
Status" table for current per-cluster build/test status — this document
explains the *shape* of the system, that one tracks its *state*.

## 1. The core idea: a fractal of small Holochain apps, not one monolith

Mycelix is not one big application. It's a set of **domain clusters**
(commons, governance, health, finance, identity, hearth, praxis, craft, ...)
— each cluster is its own DNA (or small group of DNAs) with its own zomes,
its own entry types, and its own test suite. A cluster owns one coherent
slice of the system (e.g. `mycelix-governance` owns proposals/voting/treasury;
`mycelix-identity` owns DIDs/recovery/MFA).

Clusters are combined at deploy time into **hApp bundles** — most notably
`happs/mycelix-unified-happ.yaml`, which packages many cluster DNAs into one
installable hApp so they can share a single running conductor and call each
other directly (`CallTargetCell::OtherRole`) instead of going over a network
hop. This is why the live conductor on this host has one hApp bundle
containing ~10 cluster DNAs rather than ~10 separate hApps.

Why split into clusters instead of one giant DNA: each cluster's WASM,
validation rules, and test suite stay independently buildable and
independently upgradable. The cost is that clusters need a disciplined way
to call each other — that's the bridge layer (§2).

## 2. Cross-cluster calls: the bridge pattern

Every cluster that needs to talk to another cluster does it through a
**`<cluster>-bridge` coordinator zome** (e.g. `commons-bridge`,
`civic-bridge`, `hearth-bridge`, `personal-bridge`, `governance-bridge`).
This is the only sanctioned path between clusters — a governance zome never
calls a health zome directly; it calls its own bridge, which calls the
target cluster's bridge.

The bridge zomes are hand-written per cluster (not code-generated), but they
share a common foundation: `crates/mycelix-bridge-common/`. That crate
provides, as plain HDK-free Rust so it's unit-testable without a conductor:

- **`routing_registry.rs`** — the canonical allowlist of which zome/fn pairs
  one cluster is permitted to call on another, keyed by
  `CrossClusterRole` (an enum covering all clusters: Commons, Civic,
  Identity, Hearth, Personal, Finance, Governance, Music, Health, Energy,
  Knowledge, Climate, Craft, Manufacturing, Supplychain, Praxis, and more).
  This exists specifically to replace hardcoded per-bridge `ALLOWED_*_ZOMES`
  constants with one source of truth — though not every bridge has been
  migrated onto it yet (see "known gap" below).
- **`consciousness_profile.rs`** — the gating logic (§3).
- **`migration.rs`** — the schema-versioning framework (§4).
- **`dispatch.rs`** (via `dispatch_call_checked`) — the actual cross-cluster
  call primitive: validates the target zome/fn against an allowlist, applies
  rate limiting, and only then dispatches.

**Apparent gap that turned out not to be one, verified 2026-07-03** — worth
recording so it isn't re-flagged without re-checking: `commons-bridge` and
`civic-bridge`'s `dispatch_call` bodies look very different at a glance
(commons-bridge does `detect_sub_cluster()` / `is_local_zome()` /
cross-DNA routing before dispatching; civic-bridge just checks for
empty-string zome/fn names). This is NOT a validation gap — commons
genuinely has two sub-cluster DNAs (`commons_land` + `commons_care`) that
need cross-DNA routing, and civic has one DNA and doesn't. Both call
`dispatch_call_checked(&input, ALLOWED_ZOMES)` with a real,
cluster-specific allowlist, and `dispatch_call_checked` itself is what
actually enforces the allowlist (`crates/mycelix-bridge-common/src/lib.rs`,
`if !allowed_zomes.contains(&input.zome.as_str())`) — so the allowlist
enforcement is identical in both; commons-bridge just has legitimately more
work to do before it gets there. Real remaining gap: the `*-bridge`
coordinators were largely written before `routing_registry.rs` existed, so
none of them consume it yet — each still hand-maintains its own
`ALLOWED_ZOMES` constant rather than reading from the single registry. That
means keeping N allowlists in sync by hand is still a real (lower-severity)
maintenance risk, just not a live bypass. If you're touching bridge
dispatch logic, check whether the cluster you're editing has already been
migrated onto
`routing_registry::is_allowed()` — if not, consider migrating it as part of
your change rather than adding another divergent copy.

## 3. Consciousness / Sovereign gating

Mycelix gates governance-weight actions (voting, proposals, constitutional
changes, emergency powers) behind a multi-dimensional trust profile instead
of one-agent-one-vote or a single reputation score. Two generations of this
exist in the code, both live in `mycelix-bridge-common`:

- **4D `ConsciousnessProfile`** (`consciousness_profile.rs`) — the original
  design. Four dimensions, each 0.0–1.0:
  1. **Identity** — MFA assurance level (Anonymous → Critical)
  2. **Reputation** — cross-hApp aggregated reputation with exponential decay
  3. **Community** — peer trust attestations, weighted by attestor tier
  4. **Engagement** — domain-specific participation, computed locally
  A weighted combination maps to one of five `ConsciousnessTier`s:
  `Observer` (< 0.3, read-only) → `Participant` (≥ 0.3, basic proposals) →
  `Citizen` (≥ 0.4, voting rights) → `Steward` (≥ 0.6, constitutional
  actions) → `Guardian` (≥ 0.8, emergency powers).
  `evaluate_governance()` is the pure function that does this mapping — it
  has no HDK dependency, so it's testable without a conductor, and it's the
  most heavily property-tested code in the repo (adversarial proptests,
  unclamped/out-of-range-input proptests, a dedicated security-regression
  suite covering replay attacks, expired-credential reuse, clock-skew abuse,
  and attestation tampering).
  Note some source comments call this "legacy" relative to §3's newer 8D
  system, but it is not dead code — it's still the primary gate wired into
  the governance bridge.
- **8D Sovereign Profile** (`sovereign-profile` crate, published on
  crates.io as `sovereign-profile` v0.1.2) — a newer, richer civic identity:
  EpistemicIntegrity, ThermodynamicYield, NetworkResilience, EconomicVelocity,
  CivicParticipation, Stewardship, SemanticResonance, DomainCompetence. This
  crate holds *no key material* — it's purely the scoring/decay/weights
  layer that sits on top of whatever identity system authenticates the
  agent (see the identity module note in `mycelix-identity`'s own docs for
  where key material actually lives).

**MATL** (Mycelix Adaptive Trust Layer) is the aggregate trust score used in
federated-learning contexts specifically: `Composite = 0.4·PoGQ +
0.3·Consistency + 0.3·Reputation`. 45% is this formula's theoretical
worst-case Byzantine-tolerance ceiling; the empirically *validated* figure
(from `mycelix-core`'s 0TML federated-learning benchmarks) is 34% — cite 34%
when describing tested behavior, 45% only when describing the formula's
design ceiling. See `mycelix-core/CLAUDE.md`.

## 4. Schema evolution: the `Migratable` trait

Every bridge entry type carries a `schema_version: u8` field. Holochain
doesn't support in-place mutation of DHT entries, so "migration" here means
the *runtime* upgrade path applied when an old-shaped entry is deserialized
— not a rewrite of data at rest. The trait, in
`crates/mycelix-bridge-common/src/migration.rs`:

```rust
pub trait Migratable: Sized {
    const CURRENT_VERSION: u8;
    fn migrate_from(json: &str, from_version: u8) -> Result<Self, MigrationError>;
}
```

`MigrationError` distinguishes "version not recognized" from "data
corrupt", so callers can tell a genuinely-too-new entry from a broken one.
If you're adding a new field to an existing entry type, this is the
mechanism to use — bump `CURRENT_VERSION`, implement `migrate_from` to
backfill a sensible default for the new field, and add a test that a
v(N-1)-shaped payload migrates correctly. See `migration_tests.rs` for the
existing pattern to follow.

## 5. Identity: where key material actually lives (and doesn't, yet)

This is the area with the biggest gap between backend and frontend, so it's
worth being explicit:

- **Real, tested backend** in `mycelix-identity/zomes/`:
  `did_registry` (W3C-DID-shaped documents, key rotation,
  `claim_recovered_did`), `recovery` (both guardian/trustee threshold-vote
  social recovery *and* progressive self-recovery via hashed verification
  anchors — phone/email/passkey/device/biometric — with proptest-covered
  state machines), and `mfa` (a 5-level assurance ladder from Anonymous to
  ConstitutionallyCritical).
- **No frontend wiring yet.** The shared Leptos UI shell
  (`crates/mycelix-leptos-core/src/local_identity.rs`) generates a
  throwaway per-browser identity — it does not call `did_registry::create_did`,
  does not know about the recovery zomes, and is not the conductor's real
  agent key. A first slice of real cryptographic identity for this module is
  tracked separately (see project task history around 2026-07-03); the DID
  registry and recovery zomes above are the backend it should eventually
  drive.
- **`mycelix-crypto`** (`mycelix-identity/crates/mycelix-crypto/`) is the
  crypto-agile Ed25519/PQC library backing the DID system, but its `wasm`
  Cargo feature currently exposes *types only* (no signing/keygen) — all
  real crypto operations are gated `#[cfg(feature = "native")]`. A
  browser-usable signing path is a prerequisite for wiring the frontend
  directly into this crate.

## 6. Directory map

```
mycelix-workspace/
├── happs/                    # hApp manifests bundling multiple cluster DNAs
│   └── mycelix-unified-happ.yaml   # the big bundle — most clusters, one conductor
├── crates/
│   ├── mycelix-bridge-common/      # §2, §3, §4 — the shared bridge/gating/migration kernel
│   ├── mycelix-bridge-entry-types/ # shared DHT entry types + error_messages
│   ├── mycelix-leptos-core/        # shared Leptos UI shell (app_shell, identity, gating primitives)
│   ├── mycelix-leptos-ui/          # shared Leptos UI kit (empty_state, toasts, data_table, ...)
│   ├── mycelix-leptos-client/      # WS/Tauri transports connecting a frontend to a conductor
│   ├── mycelix-zome-helpers/       # shared zome-side utilities
│   └── sweettest-harness/          # multi-conductor integration test harness
├── sdk/                      # Rust SDK
├── sdk-ts/                   # TypeScript SDK (published as @mycelix/sdk)
├── sdk-python/, sdk-wasm/    # Python and WASM SDKs
├── mycelix-<cluster>/        # one directory per domain cluster (commons, governance,
│                              # health, identity, hearth, finance, praxis, craft, ...) —
│                              # each is its own Cargo workspace with zomes/ + apps/leptos/
└── scripts/sync-to-standalone.sh   # publishes this whole tree to the public GitHub repo
```

Note: some `mycelix-<domain>/` directories at this level (e.g.
`mycelix-property`, `mycelix-housing`, `mycelix-water`, `mycelix-care`,
`mycelix-justice`, `mycelix-media`, `mycelix-mutualaid`, `mycelix-emergency`)
are **not** independent top-level clusters — they're leftover partial
duplicates from an earlier consolidation attempt, per
`scripts/sync-to-standalone.sh`'s own comments, which explicitly excludes
some of them from the public sync to avoid shipping dead duplicate code.
Their real, current home is as constituent domains inside `mycelix-commons`
(property/housing/water/food/care/mutualaid/transport) or `mycelix-civic`
(justice/emergency/media). If you're not sure whether a directory is live,
check `mycelix-workspace/CLAUDE.md`'s hApp Status table first.

## 7. Where to go next

- **Per-cluster build/test status, current as of last verification**:
  `mycelix-workspace/CLAUDE.md`
- **Port allocations**: `.claude/rules/PORTS.md` (private monorepo) — not
  synced to the public repo; the public repo's deployment docs are in
  `mycelix-pulse/docs/guides/deployment.md` for the one cluster (Pulse/mail)
  with a documented self-host path.
- **Contributing / license**: `CONTRIBUTING.md`, `LICENSE`,
  `COMMERCIAL_LICENSE.md` (repository root)
- **Security disclosure**: `SECURITY.md`
