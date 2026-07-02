# Mycelix Ecosystem Review — 2026-07-01

A high-level architecture and risk audit of the Mycelix ecosystem (16+ Holochain
clusters, shared bridge infrastructure, frontends, and the Pulse/SMTP gateway),
commissioned to produce one actionable improvement doc instead of another
one-off status report.

## How this review was done

Five parallel research passes: (1) synthesis of the ~24 existing self-authored
status/audit docs already in this repo, to establish what's already known vs.
stale; (2) shared bridge infrastructure (`mycelix-bridge-common`,
`mycelix-bridge-entry-types`, `mycelix-zkp-core`) plus a repo-wide security
sweep; (3) core social/governance clusters (commons, civic, hearth, identity,
governance, personal, attribution); (4) economic/health/comms clusters
(finance, health, pulse, marketplace, supplychain); (5) knowledge/frontend/
newer clusters (praxis, craft, music, knowledge, desci, energy, climate,
space, core, prism). The two most severe cross-cutting claims (mycelix-health
"missing," the stray nested git repo) were independently re-verified by hand
before writing this doc. Everything else is agent-sourced with file:line
citations where available — treat single-agent findings as **PLAUSIBLE**
unless marked verified, and re-check before acting on anything load-bearing.

---

## Executive summary

The Mycelix ecosystem is large, genuinely tested in places (515 passing tests
in `mycelix-bridge-common` alone, ~2,900+ zome tests across commons/civic
combined, real fuzz targets and a Kani formal-verification module), and the
core cross-cluster architecture (`CallTargetCell::OtherRole` + a centralized
`routing_registry`) is a sound design. But three systemic problems undercut
all of it:

1. **Version control is not actually protecting most of this code.** The
   `mycelix-workspace/` directory — which contains essentially the entire
   Mycelix codebase — has its own independent, disconnected `.git` repository
   (authored by a different tool, "Antigravity AI," not this monorepo's normal
   workflow), meaning none of it is tracked by the outer `/srv/luminous-dynamics`
   repo's history. **Verified.** This is the single highest-priority fix.
2. **Money- and provenance-moving code paths skip caller-identity checks**
   in multiple clusters (finance, supplychain, marketplace), while the one
   cluster that does it correctly (currency-mint) shows the pattern is known
   and just wasn't applied consistently.
3. **Documentation has drifted far ahead of — and in some cases directly
   contradicts — the code.** Zome counts, test counts, framework choices
   (Axum vs. Actix-web), and even whether a cluster's code is present at all
   (mycelix-health) are wrong in root-level docs that future coding agents
   will reasonably trust and build on.

None of the findings below suggest the architecture is wrong. They suggest
the ecosystem grew faster than its safety nets (git hygiene, auth checks,
doc accuracy, CI enforcement) could keep up — which is fixable without a
redesign.

---

## P0 — Critical (fix before anything else)

### 1. `mycelix-workspace/` has a disconnected, untracked git history — VERIFIED
`mycelix-workspace/.git` is a real, independent repository (not a submodule
gitlink) with `user.name = "Antigravity AI"`, whose own history is a single
`"Initialize repository in workspace root"` commit (2026-06-15) that bulk-
imported the entire tree. It is **not** registered in the outer repo's
`.gitmodules`. Practically: `commons`, `civic`, `hearth`, `personal`,
`attribution`, `praxis`, `craft`, `music`, `knowledge`, `desci`, `energy`,
`climate`, `space`, `core`, and `prism` — i.e. nearly the whole ecosystem —
have no durable, recoverable version history through the outer monorepo, and
whatever history exists in the nested repo is one flat commit. A `git clean`,
disk failure, or accidental directory delete anywhere under
`mycelix-workspace/` is currently unrecoverable beyond that one snapshot.
This directly violates `.claude/rules/CONCURRENT_SESSIONS.md`'s "commit
frequently" guidance and CLAUDE.md Rule 8 — but those rules can't help if the
directory isn't even reachable from the outer repo's `git add`.

**Fix**: decide deliberately whether `mycelix-workspace/` should be (a) a
real git submodule pointing at its own remote (matches the pattern already
used for `mycelix-health`), or (b) folded into the outer repo as regular
tracked files (`git rm -rf --cached` the nested `.git`, then `git add`).
Either is fine; leaving it as an orphaned nested repo is not. Do this before
any other cleanup work in this doc, since further edits to these clusters
should land in real history.

### 2. `mycelix-supplychain`: provenance forgery + escrow drain — unverified by hand, high confidence
- `logistics/coordinator/src/lib.rs:127-207` (`add_tracking_event`) records
  `reported_by` from caller input but never checks it against the shipment's
  expected sender/recipient — any agent can post a fabricated "Delivered"
  event.
- `payments/coordinator/src/lib.rs:170-190` (`release_escrow`) and
  `:100-119`/`:209` have no payer/payee ownership check — any agent can
  release escrowed funds.
- This is a **separate payments implementation from `mycelix-finance`**,
  bridged only best-effort, so it's also a duplication/drift risk (see P1 #4).

**Fix**: add `agent_info()`-bound ownership checks before any state
transition in `logistics` and `payments`; this cluster should not be
considered demo-safe until fixed.

### 3. `mycelix-finance`: no caller-identity checks on payments/treasury/staking — high confidence
`payments::channel_transfer`, `treasury::update_treasury_balance`,
`staking::withdraw_stake`/`update_stake_trust`/`slash_stake` never call
`agent_info()` to verify the caller owns the balance/stake being mutated —
matched only by a caller-supplied ID string. Contrast: `currency-mint`
(the actual minting authority) *does* call `agent_info()` at 4+ sites and
gates minting behind governance-proposal authorization — so the pattern is
known in this same crate family, just not applied to payments/treasury/
staking. Additionally, `staking::compute_bytes_hash` uses
`std::collections::hash_map::DefaultHasher` (SipHash — explicitly
non-cryptographic, unstable across Rust versions) to produce what the module
doc comment calls "cryptographic evidence" for slashing events.

**Fix**: add `agent_info()` ownership checks to payments/treasury/staking on
the same pattern currency-mint already uses; swap `DefaultHasher` for
blake3 (already a dependency elsewhere in the workspace, e.g. Pulse).

### 4. `mycelix-identity`: recovery-flow DID spoofing
`zomes/recovery/coordinator/src/lib.rs:130-193` (`initiate_recovery`) checks
`config.trustees.contains(&input.initiator_did)` but never binds
`initiator_did` to the actual calling agent's pubkey via `agent_info()`. Any
caller who knows a trustee's DID string can initiate account recovery *as*
that trustee. The same file's `revoke_recovery_config` (~line 657) does the
correct check (`config.owner != agent_info.agent_initial_pubkey`), so this
reads as an oversight in one function rather than a missing pattern.

**Fix**: mirror the ownership check from `revoke_recovery_config` into
`initiate_recovery` (resolve `initiator_did` → pubkey, compare against
caller). Small fix, high impact — this is the account-recovery path.

### 5. `mycelix-health`: registered submodule, never initialized; unified hApp manifest broken — VERIFIED, reframed
Initial pass flagged this as "code doesn't exist." On verification: it's less
dire but still broken. `mycelix-health` **is** a real, reachable upstream
repo (`https://github.com/Luminous-Dynamics/mycelix-health.git`, confirmed
live via `git ls-remote`) and **is** registered as a proper git submodule at
the top level (`/srv/luminous-dynamics/mycelix-health`, pinned commit
`8b861e32c8...`) — but that submodule was never checked out
(`git submodule update --init` not run), so the directory doesn't exist
locally. Separately, `mycelix-workspace/mycelix-health/` is an empty, untracked,
unrelated stray directory (not the submodule, not wired to it). Meanwhile
`mycelix-workspace/happs/{happ,mycelix-unified-happ}.yaml:148-153` declare a
DNA role pointing at `./health/dna/health.dna` — a path that doesn't match
*either* location, so **the unified 21-role hApp is unbuildable as
configured today**, independent of whether the submodule is checked out.
`mycelix-workspace/CLAUDE.md` and `ECOSYSTEM_STATUS.md` describe health as
"Built, 15 zomes, 81K LOC" (and `ECOSYSTEM_STATUS.md` even contradicts
itself in the same file, separately calling it "Scaffolded") — both are
misleading to anyone (human or agent) planning against them.

**Fix**: run `git submodule update --init mycelix-health`, fix the manifest
path to point at wherever the checked-out submodule actually lands relative
to `mycelix-workspace/`, and correct the two contradictory status lines.

---

## P1 — High

### 1. Cross-cluster dispatch: caller-controlled `fn_name`, zome-level-only allowlisting
The intended security model is one centralized allowlist
(`crates/mycelix-bridge-common/src/routing_registry.rs`, 1,693 lines, well
tested — `cargo test -p mycelix-bridge-common --lib` → 515 passed). In
practice, dispatch structs in at least `mycelix-governance`'s bridge
(`DispatchPersonalCallInput`/`DispatchIdentityCallInput`, in
`cross_cluster.rs:33-36`) accept caller-supplied `zome_name` **and**
`fn_name`, but only validate `fn_name` for length (1-256 chars) — not
against a function-level allowlist. Once a target zome passes the allowlist,
*any* exported function in it can be invoked, not just the intended one.
Today's call sites appear to hardcode `fn_name` server-side rather than
accept it live from remote callers, so exploitability looks low right now —
but the struct shape itself is the vulnerability, and it will bite the first
time someone wires a caller-facing path through it.

**Fix**: extend `routing_registry` to allowlist (zome, function) pairs, not
just zomes, and update the governance/personal/identity dispatch structs to
enforce it.

### 2. `routing_registry` adoption is partial — allowlist drift risk
Only 4 clusters (`commons-bridge`, `civic-bridge`, `hearth-bridge`,
`personal-bridge`) actually call `routing_registry::is_allowed`/
`get_allowed_zomes`. At least 8+ more dispatch `CallTargetCell::OtherRole`
directly without it: `mycelix-identity`, `mycelix-governance` (which
re-implements its own local `ALLOWED_PERSONAL_ZOMES`/etc. consts — the exact
pattern the registry's doc comment says it was built to replace),
`mycelix-finance`, `mycelix-praxis`, `mycelix-supplychain`,
`mycelix-marketplace`, `mycelix-pulse`. Two allowlists (governance's local
one and the registry's) can silently diverge with nothing enforcing sync.

**Fix**: migrate the remaining clusters onto `routing_registry`, delete the
local `ALLOWED_*` consts, and add a CI/lint check that fails on any new
`CallTargetCell::OtherRole` call site that doesn't route through it.

### 3. `mycelix-marketplace`: arbitration is structurally dead
`arbitration/coordinator/src/lib.rs:99-104` hardcodes an empty arbitrator
`Vec` ("placeholder"), so `finalize_arbitration`'s all-voted gate (:225-229)
can never pass, and the self-dealing exclusion that exists in that same
unreachable code (:109-111) is not enforced by the integrity zome at all
(`arbitration/integrity/src/lib.rs:160-222`). Separately,
`transactions::confirm_transaction`/`mark_shipped`/`confirm_delivery` never
verify caller identity against buyer/seller (only `dispute_transaction`/
`cancel_transaction` do).

**Fix**: wire real arbitrator assignment, move the self-dealing check into
the integrity zome, and add buyer/seller checks to the three unchecked
transaction-state functions.

### 4. Three independent, drift-prone value-moving implementations
`mycelix-finance/zomes/payments`, `mycelix-supplychain/holochain/zomes/payments`,
and `mycelix-marketplace`'s settlement bridge (which calls into finance,
non-fatally swallowing failures) are three separate paths for moving value,
sharing no code and only 3 of ~30 finance/marketplace/supplychain zomes even
depend on `mycelix-bridge-common`. Combined with P0 #2/#3, "consciousness-
gated high-stakes actions" should be treated as **aspirational** for
this cluster group, not implemented, until these are consolidated or at
least uniformly gated.

**Fix**: pick one authoritative ledger (finance's `currency-mint`/`payments`
is the natural candidate given it's the only one with real auth) and have
supplychain/marketplace call into it rather than maintaining parallel state.

### 5. `mycelix-prism`: unsanitized externally-fetched HTML injected via `inner_html`
The prior SSRF fix (`is_private_ipv4`/`validate_proxy_url` blocking loopback,
RFC1918, cloud metadata endpoints) is confirmed still in place in both
`prism-proxy/src/main.rs:20-52` and `prism-serve/src/main.rs:25-68` — good.
But it's duplicated verbatim across the two crates (drift risk if one gets
patched and not the other), and separately,
`prism-ui/src/components/sentient_overlay.rs:38-53` (`annotate_html`) and
`pages/content_router.rs:48` inject proxy-relayed external page HTML via
`inner_html` with no sanitization (no `ammonia` or equivalent found
anywhere in `prism-proxy`/`prism-serve`). The SSRF fix blocks internal
targets, not payload content — any external page containing `<script>`
served through the proxy executes in the WASM app's origin. This is a live
XSS gap of the same severity class as the SSRF issue it sits next to.

**Fix**: add an allowlist-based HTML sanitizer (`ammonia`) between the proxy
and `inner_html` injection; factor `validate_proxy_url`/`is_private_ipv4`
into one shared crate used by both `prism-proxy` and `prism-serve`.

### 6. `mycelix-desci`: wildcard CORS, no auth middleware, placeholder rate limiter
`src/api/src/main.rs:143-149` sets `allow_origin("*")` for `GET, POST, PUT,
DELETE` including `Authorization` header pass-through; no auth/JWT/API-key
middleware exists in `handlers/` or `middleware/` for mutating endpoints
(e.g. claims creation). `middleware/rate_limit.rs` is an explicit,
self-documented placeholder ("integrate a proper rate limiting solution
like tower-governor"). Also, the docs describing this as "REST API
(Actix-web)" are wrong — it's actually Axum.

**Fix**: replace wildcard CORS with an explicit origin allowlist, add auth
middleware on mutating routes, implement the already-suggested
`tower-governor` rate limiting, and correct the framework name in docs.

### 7. `mycelix-pulse`: DKIM signing is a no-op; Holochain bridge defaults to a stub
`crates/pulse-smtp-gateway/src/outbound.rs:110-127` (`dkim_sign()`) ignores
the configured RSA/Ed25519 keys and returns the raw message unmodified
(`TODO(phase-5b)`) — any doc language implying Phase 5A ships DKIM signing
is inaccurate. `StubZomeBridge` (`src/zome.rs:65-193`, self-documented as
"logs + accepts everything") is the *default* build; the real
`holochain_client` integration (`src/zome.rs:249-264`) is feature-gated
behind `holochain-bridge`, which is **not** a default feature. Both facts
match this repo's own memory of "Phase 5B not yet deployed," so this isn't
new information so much as confirmation the gap is real and specifically
located — but worth being precise about in any external-facing readiness
claims. Minor: NixOS systemd hardening
(`_infrastructure/nixos/pulse-gateway-host.nix:247-280`) is otherwise solid
(`ProtectSystem=strict`, `NoNewPrivileges`, scoped `CapabilityBoundingSet`)
but is missing `RestrictAddressFamilies`.

**Fix**: none urgent — this is correctly scoped as Phase 5B work already.
Just make sure no readiness doc overstates DKIM/bridge status, and add
`RestrictAddressFamilies` to the systemd unit while touching that file.

---

## P2 — Medium (worth fixing, lower urgency)

- **`mycelix-governance`**: 29 zome directories exist on disk, only 9 are
  wired into `dna/dna.yaml`; 20 are orphaned stubs (several literally 0 LOC:
  `singularity`, `intuition`, `autopoietic_amendment`, `s2s_bridge`,
  `sentinel`, `steward_consent`, `sovereign_beacon`, etc.), plus a stray
  `Cargo.toml.DISABLED`. Low risk (not compiled into the live DNA) but pure
  clutter and doc/code confusion. Separately, `execution::create_timelock`
  is a public `#[hdk_extern]` with no internal-only restriction — nothing
  stops a direct call with an arbitrary `proposal_id`/`actions` pair outside
  the normal tally-approval flow; worth confirming
  `execution/integrity/src/lib.rs:548`'s `validate_create_timelock` actually
  cross-checks against an approved tally.
- **`mycelix-identity`**: `name-registry`, `reputation-aggregator`, and
  `web-of-trust` exist as substantial code on disk but aren't wired into the
  DNA manifest — worth checking whether that's intentional or a regression,
  since (unlike governance's stubs) this looks like real lost functionality.
- **Unwrap/panic density**: `hearth-kinship` has 80+ `.unwrap()`/`.expect()`
  calls outside tests, worth converting to `ExternResult` errors given
  hearth handles emergency/care data. Note: a repo-wide unwrap/TODO grep
  initially flagged Praxis at "5,544 unwraps" / "1,406 TODOs," but a
  follow-up pass found the real number is much lower — the vendored
  `.cargo/registry/` tree is accidentally present inside
  `mycelix-praxis/` and inflates any naive grep across that directory.
  **Action**: `.gitignore` (or delete) that vendored path, then redo the
  unwrap/TODO sweep with clean numbers before trusting them for prioritization.
- **ZKP policy compliance**: clean today (`arkworks|halo2|groth16|bellman`
  grep across all of `mycelix-workspace` returns only doc-comment/algorithm-
  name false positives — no real dependency violations of the
  DASTARK-only rule). No CI enforcement exists, though — add a
  `cargo-deny` banned-crate check so this stays true automatically.
- **Working-tree/git mismatch inside social clusters**: independent of the
  P0 nested-repo issue, `commons`, `civic`, `hearth`, `personal`, and
  `attribution` show 0 tracked files even measured from *inside*
  `mycelix-workspace`'s own nested repo in some checks — recommend
  re-verifying git state for each cluster once P0 #1 is resolved, since the
  fix for #1 may also resolve this.
- **Disabled/deprecated code accumulation**: `disabled-crates/`
  (`binius-zome-verifier`, `mycelix-leptos-core`, `mycelix-leptos-ui`),
  `_deprecated/experimental-zomes-2026-06-13/` (an early, apparently
  superseded identity-zome prototype), `mycelix-governance/Cargo.toml.DISABLED`,
  `mycelix-supplychain/rust.disabled` — none individually dangerous, but
  worth a one-time confirm-and-delete pass; `rust.disabled` in particular
  means supplychain's coverage numbers in any doc may not reflect what
  `cargo test --workspace` actually runs.
- **Test-only `unsafe { std::mem::zeroed() }`** in
  `mycelix-pulse/backend/api/src/{trust/decay.rs,search/encrypted_search.rs}`
  to satisfy a constructor signature in calculation-only tests — confined to
  `#[cfg(test)]`, never dereferenced today, but fragile; replace with a
  proper mock/trait-object type.

---

## Architecture decisions already made — do not re-flag these as bugs

- `mycelix-space` has zero cross-cluster bridge calls **by design** —
  orbital mechanics is deliberately standalone.
- `mycelix-marketplace` and `mycelix-music` are intentionally kept separate
  from a Holochain-only architecture (Node.js/Solidity components) — a
  proposed merge was explicitly evaluated and rejected.
- 15 of 37 TypeScript SDK integration modules are intentionally
  "local-only" (in-memory Map + LocalBridge), not yet wired to `callZome` —
  this is a valid local-first pattern, not incompleteness (an earlier audit
  mislabeled these as stubs; that mislabel has since been corrected in
  `ECOSYSTEM_STATUS.md`).
- The dual-DID architecture (governance identity vs. legal identity) is a
  hard non-negotiable: governance weight must never read legal-ID
  credentials. Anyone auditing `mycelix-lawful-identity` should treat this
  as a constraint, not a gap.
- Cross-cluster calls always route through `CallTargetCell::OtherRole`,
  intended to be centralized in `routing_registry.rs` — the *adoption gap*
  (P1 #2) is real, but the pattern itself is the documented, correct one.

---

## Documentation & process hygiene

This repo has accumulated **~24 overlapping status/roadmap/audit docs**
across `/srv/luminous-dynamics/` and `mycelix-workspace/`, with a clear
pattern: a new `*_REPORT.md`/`*_STATUS.md`/`*_CERTIFICATE.md` gets created
nearly every session instead of updating one canonical doc. Findings from
the doc-synthesis pass:

- Two files claim to be *the* single source of truth
  (`ECOSYSTEM_STATUS.md` and `BETA_PROMOTION_CRITERIA.md`) without
  cross-referencing each other.
- Six docs with ecosystem-generic names (`FINAL_READINESS_CERTIFICATE.md`,
  `FINAL_INTEGRATION_ROADMAP.md`, `SYSTEMIC_CONSCIOUSNESS_REPORT.md`,
  `MYCELIX_READINESS_REPORT.md`, plus `TECH_DEBT.md`/
  `DEPENDENCY_AUDIT_REPORT.md`) are actually narrowly scoped to Prism or a
  single dependency conflict, are **untracked in git**, and read as
  ungrounded ("session sign-off") writing rather than evidence-based audits
  — no commit references, grandiose language. Treat these as noise.
- The credible, evidence-based docs
  (`ECOSYSTEM_STATUS.md`, `DNA_SIZE_AUDIT.md`, `HAPP_PORTFOLIO_STRATEGY.md`,
  `MYCELIX_IMPROVEMENT_PLAN.md`, `MYCELIX_DEMO_READINESS.md`,
  `MYCELIX_TIER_3_4_PRODUCT_GATE.md`, `MYCELIX_STATE_COEXISTENCE.md`) cite
  commit hashes and file-level evidence, but are split across two
  directories and already cross-reference each other loosely.
- `DNA_SIZE_AUDIT.md` contradicts itself internally: an update box near the
  top gives different totals than the (uncorrected) executive-summary table
  below it.
- `HAPP_PORTFOLIO_STRATEGY.md` (Feb 14) calls Music "Dormant" —
  contradicted by fresh code inspection showing recent commits shipping a
  consciousness-bridge feature. Stale, don't trust that doc's per-cluster
  activity labels.
- Doc-vs-disk drift is close to universal: commons has 50 zome dirs vs. 39
  documented, civic 20 vs. 18, praxis 14 vs. 10, knowledge 10 vs. 8
  (undocumented `invention` zome), desci is Axum not Actix-web as
  documented, and `mycelix-core`'s actual scope (Solidity contracts,
  Kubernetes/Terraform deployment configs, a standalone Tauri desktop app,
  a full 0TML research tree with its own zkSTARK work) is far larger than
  the one-line "0TML federated learning research" summary suggests.

**Recommendation**: consolidate to one living `ECOSYSTEM_STATUS.md` (already
the most disciplined/self-correcting of the bunch — it strikes through
resolved items in place) plus one dated, commit-linked `TECH_DEBT.md`.
Archive or delete the six untracked/narrowly-scoped docs listed above.
Retire `BETA_PROMOTION_CRITERIA.md`'s stale status table but keep its
checklist as a reusable template. Fold `MYCELIX_IMPROVEMENT_PLAN.md`,
`MYCELIX_DEMO_READINESS.md`, and `MYCELIX_TIER_3_4_PRODUCT_GATE.md` into a
single living roadmap once their currently-open items resolve.

---

## Recommended action plan

**Phase 0 — stop the bleeding (do first, low effort/high leverage)**
1. Resolve the `mycelix-workspace` nested-git-repo issue (P0 #1).
2. `git submodule update --init mycelix-health` + fix the unified hApp
   manifest path (P0 #5).
3. `.gitignore`/delete the vendored `.cargo/registry/` tree inside
   `mycelix-praxis/` so future greps/CI aren't polluted (P2).

**Phase 1 — close the exploitable gaps (this sprint)**
4. Add `agent_info()` ownership checks: `mycelix-supplychain` logistics +
   payments (P0 #2), `mycelix-finance` payments/treasury/staking (P0 #3),
   `mycelix-identity` `initiate_recovery` (P0 #4).
5. Sanitize proxied HTML in `mycelix-prism` before `inner_html` (P1 #5).
6. Lock down `mycelix-desci`'s CORS/auth/rate-limiting (P1 #6).
7. Wire real arbitrator assignment in `mycelix-marketplace` (P1 #3).

**Phase 2 — architectural convergence (next sprint or two)**
8. Migrate remaining clusters onto `routing_registry`; delete local
   allowlist consts; add function-level allowlisting (P1 #1, #2).
9. Consolidate the three parallel value-moving implementations onto one
   ledger (P1 #4).
10. Add CI enforcement: ZKP banned-crate check, routing-registry-usage
    lint, `cargo audit` gate (referenced in `BETA_PROMOTION_CRITERIA.md`
    but apparently never actually run).

**Phase 3 — cleanup and doc hygiene (ongoing, low urgency)**
11. Delete/finish orphaned governance zome stubs and identity's orphaned
    (but substantial) `name-registry`/`reputation-aggregator`/`web-of-trust`.
12. Prune `disabled-crates/`, `_deprecated/`, and cluster-local
    `.DISABLED`/`.disabled` files after a one-time confirmation pass.
13. Execute the doc-consolidation plan above.

---

## Appendix: per-cluster quick reference

| Cluster | P0/P1 issues found | Notes |
|---|---|---|
| commons | — | Largest cluster (50 zome dirs), test-heavy; caught in the P0 #1 git issue |
| civic | — | 20 zome dirs vs. 18 documented; caught in P0 #1 |
| hearth | P2 (unwrap density) | 11 zomes, consistent `validate()` coverage |
| identity | P0 #4 (recovery DID spoofing), P2 (orphaned zomes) | Thin test density relative to size |
| governance | P1 #1 (fn_name trust gap), P2 (20 orphan stubs) | Voting logic itself is strong (Sybil defense, fail-closed gating) |
| personal | — | Small, no red flags found |
| attribution | — | Small, high test density, DASTARK-consistent ZKP structure |
| finance | P0 #3 | currency-mint is well-gated; payments/treasury/staking are not |
| health | P0 #5 | Submodule uninitialized + manifest path broken; docs overstate status |
| pulse | P1 #7 | Correctly scoped Phase 5B gaps (DKIM, real zome bridge); systemd hardening solid |
| marketplace | P1 #3 | Arbitration structurally dead; transaction-state checks incomplete |
| supplychain | P0 #2 | Forgeable provenance events, unguarded escrow release |
| praxis | — | 14 zomes vs. 10 documented; TODO/unwrap counts inflated by vendored deps, needs re-audit after cleanup |
| craft | — | Cleanest cluster found in this pass |
| music | — | Much larger infra footprint (k8s/terraform/gateway) than docs suggest |
| knowledge | — | 10 zomes vs. 8 documented (undocumented `invention` zome) |
| desci | P1 #6 | Framework doc wrong (Axum not Actix-web); real auth/CORS gaps |
| energy / climate / space | — | Clean, but none wired into `mycelix-bridge-common` — confirm intentional |
| core | — | Scope far larger than documented (contracts, k8s, desktop app, 0TML zkSTARK); needs a dedicated follow-up pass |
| prism | P1 #5 | SSRF fix confirmed intact; new unsanitized-HTML XSS gap found |
| shared (bridge-common, zkp-core) | P1 #1, #2 | Well-tested core, partial adoption across clusters |

---

## Update — 2026-07-02: P0 fixes landed, architecture corrected

All six P0/P1 findings above were fixed and verified compiling (P0 #1-#5,
P1 #3, #5, #6). Notably: the P0 #3 finance fix, P0 #4 identity fix, and P1 #3
marketplace fix were initially applied to a location that turned out to be a
stale duplicate — see this doc's own commit history for the re-applied,
correct fixes.

**Critical correction to this doc's own framing**: this review originally
treated `mycelix-workspace/mycelix-*` as the untracked frontier and
`/srv/luminous-dynamics/mycelix-*` (top-level) as canonical for most
clusters. That was backwards. Per explicit user direction, **all Mycelix
work belongs in `mycelix-workspace/`** — this matches root `CLAUDE.md` Rule
7's own phrasing and the intent of the unfinished May 31 `cd20b9a7be`
"consolidate Mycelix" commit. On 2026-07-02, 15 clusters (hearth,
governance, personal, attribution, craft, energy, manufacturing, commons,
civic, identity, finance, knowledge, climate, supplychain, music) plus
`mycelix-space` were moved from top-level into `mycelix-workspace/`, with
every symlink, Cargo path dependency, `flake.nix` reference, and the
`sync-to-standalone.sh` script itself updated and verified. Still deferred
to a follow-up session (too large/live/special-case for a single pass):
`mycelix-praxis` (live production), `mycelix-marketplace`, `mycelix-desci`,
`mycelix-core`, `mycelix-health` (submodule).

**A related, broader pattern was found and partially fixed the same day**:
the same "duplicate tree, one side goes stale" shape existed outside
Mycelix too — Sol Atlas, `sovereign-profile`, `spark-engine`,
`symthaea-hdc-ltc`, `symthaea-hdc-crypto`, `symthaea-mycelix-bridge`, and
the Symtropy bridge crates all had it. Six of those were resolved (verified
via rustfmt-normalized diffing, not raw `diff` — raw line counts had
initially made several look more diverged than they actually were).
`symthaea-hdc-ltc` and `symthaea-hdc-crypto` had *genuine* divergence
(the latter carrying a real, un-backported security fix) — those aren't a
Mycelix concern but are worth knowing about if this pattern resurfaces.

**This doc (`MYCELIX_REVIEW.md`) is the current source of truth** for what's
fixed vs. still open in this ecosystem. `MYCELIX_IMPROVEMENT_PLAN.md`,
`MYCELIX_DEMO_READINESS.md`, `MYCELIX_TIER_3_4_PRODUCT_GATE.md`, and
`BETA_PROMOTION_CRITERIA.md`'s status table have all been marked stale with
pointers back here, rather than force-merged into one doc under time
pressure — they cover genuinely different purposes (compile triage, demo
vision, investment gate) and re-verifying every claim in all three well
enough to merge safely wasn't feasible in the time available. Still open:
tasks #9-13 from the original review (routing_registry migration for the
remaining clusters, CI enforcement for the ZKP/registry-adoption rules,
ledger consolidation across finance/supplychain/marketplace, further
cleanup, and the marketplace freeze-or-retire decision that's been pending
since April).

## Update — 2026-07-02 (task #9: routing_registry migration)

The CI advisory job (`routing-registry-adoption` in
`.github/workflows/security-checks.yml`) grep found 55 files calling
`CallTargetCell::OtherRole` without referencing `routing_registry`. Reading
each of the ~26 real coordinator files directly (after excluding test
files, SDK clients, out-of-scope hApps, and stale archives already
identified by the earlier grep) found they are **not** one uniform bug —
three distinct shapes:

- **Pattern A — duplicated/drifted allowlist** (the original finding). A
  coordinator exposes a generic relay taking caller-supplied `zome_name` and
  gates it with a *local* `const ALLOWED_X_ZOMES` instead of
  `routing_registry::get_allowed_zomes()`. Only one file actually matched
  this: `mycelix-governance/zomes/bridge/coordinator/src/cross_cluster.rs`
  (5 hardcoded lists — Personal, Identity, Commons, Civic, Finance). **Fixed**
  — all 5 consts now source from `routing_registry`; content was verified
  byte-identical before the swap, so this is a pure architecture fix with no
  behavior change. Reference pattern followed:
  `mycelix-personal/zomes/personal-bridge/coordinator/src/lib.rs`.

- **Pattern B — no gate at all, but no injection surface either.** The
  overwhelming majority (23 of the ~26 files) make hardcoded,
  compile-time-fixed `OtherRole` calls — the target zome/function are Rust
  literals, never derived from caller input. There is nothing to swap for a
  registry lookup because there's no local allowlist duplicating one; the
  "gate" question doesn't really apply since the caller can't steer the
  target at all. Confirmed across: `mycelix-hearth/hearth-kinship`,
  `mycelix-governance/restorative-justice`, `mycelix-manufacturing/bridge`,
  `mycelix-manufacturing/planning`, `mycelix-energy/bridge`,
  `mycelix-energy/regenerative`, `mycelix-commons/boundary-contracts`,
  `mycelix-commons/care-circles`, `mycelix-identity/bridge`,
  `mycelix-identity/reputation-aggregator`, `mycelix-finance/bridge`,
  `mycelix-finance/payments`, `mycelix-finance/price-oracle`,
  `mycelix-finance/treasury`, `mycelix-knowledge/markets_integration`,
  `mycelix-climate/bridge`, `mycelix-supplychain/payments`,
  `mycelix-space/observations`, `mycelix-pulse/mail-bridge`. Not fixed —
  there's no mechanical fix; closing this would mean deciding whether every
  fixed-target cross-cluster call site should defensively call
  `routing_registry::is_allowed()` before calling, which is a design
  question (does a fixed compile-time target even need runtime gating?),
  not a drift fix. Left as a separate, open question for a future session.

- **False positives.** 2 files the grep matched that don't actually dispatch
  `OtherRole` at all: `mycelix-supplychain/holochain/zomes/inventory/coordinator`
  (only `CallTargetCell::Local`) and `mycelix-identity/zomes/trust_credential/coordinator`
  (zero `OtherRole` calls — the only mention is a doc comment describing how
  *other* clusters call *into* this zome, not calls it makes outward).

- **New finding — not in the original review, more severe than Pattern A.**
  `mycelix-music/dnas/mycelix-music/zomes/music-bridge/coordinator/src/lib.rs`'s
  `cross_cluster_dispatch` function takes `target_role`, `target_zome`, AND
  `fn_name` **all as caller-supplied input**. It validates `target_role`
  against a local hardcoded array (`allowed_targets`), but **never validates
  `target_zome` or `fn_name` against anything** — any zome/function within an
  allowed cluster (identity, finance, governance, civic, commons_land,
  commons_care) can be invoked by any Citizen+ tier agent. This is a real gap,
  not just drift risk. Not fixed here: the correct fix needs new logic (parse
  `target_role: String` into a `CrossClusterRole`, then call
  `routing_registry::is_allowed(CrossClusterRole::Music, target, &input.target_zome)`
  before dispatching) rather than a mechanical const swap, which is outside
  this pass's approved scope. **Recommend prioritizing this over the
  Pattern-B design question above** — it's an active gap, not a hypothetical
  one.

Net result: 1 file fixed (governance), 1 new higher-priority finding surfaced
(music cross-cluster dispatch), and the "55 advisory warnings" figure is now
understood to be almost entirely Pattern B (no mechanical fix applicable) or
noise, not evidence of widespread drift.
