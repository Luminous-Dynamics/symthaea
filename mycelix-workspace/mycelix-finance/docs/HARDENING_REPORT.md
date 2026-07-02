# Mycelix Finance: Hardening Report

Consolidated report covering 13 rounds of security and correctness hardening on the mycelix-finance workspace.

---

## Overview

- **13 rounds** of hardening across all 7 zomes (TEND, payments, currency-mint, treasury, recognition, staking, bridge) plus shared infrastructure and types
- **~23K LOC source** (zomes + types + shared) and **~17K LOC tests** across 26 test files
- **560+ test functions** including unit tests, integration tests, sweettest e2e, and property-based tests
- **16 crates** with `#![deny(unsafe_code)]` (all integrity + coordinator crates, types, shared)
- **0 clippy warnings** on workspace build
- **All 19 race conditions** identified and addressed (18 fixed, 1 mitigated)
- **47 `is_finite()` guards** across 13 source files preventing NaN/Inf injection
- **72 link types**, **55+ anchor key patterns**, **35 entry types** (see `LINK_TOPOLOGY.md`)

---

## Defense Layers

### 1. Integrity Validation

Every integrity zome enforces structural correctness at the Holochain validation level, before coordinator logic runs:

- **Status transitions**: `CurrencyStatus.can_transition_to()`, `StakeStatus.can_transition_to()`, `EscrowStatus.can_transition_to()`, `ExchangeStatus` FSM with allowed transition matrix. Invalid transitions are rejected at the DHT level.
- **`is_finite()` guards**: All `f64`/`f32` fields in entry types are validated on create and update. NaN, Infinity, and negative-Infinity are rejected. 47 guards across 13 files.
- **Immutable field enforcement**: Fields like `provider_did`, `receiver_did`, `currency_id`, `hours`, and `created_at` cannot be changed on update. Only mutable fields (status, balance, timestamps) are permitted to differ.
- **Constitutional limits**: Credit limits (TEND), inalienable reserve (25% commons), collateral ratios (SAP), recognition allocation caps (max 10/cycle), stake minimums, and demurrage rate bounds are enforced in integrity validation.

### 2. Optimistic Locking

All balance mutation paths use read-modify-write retry loops to prevent lost updates:

- **Pattern**: `follow_update_chain(link_hash)` returns the latest `(Record, ActionHash)`. The mutation function applies its change and calls `update_entry` with the retrieved action hash. If a concurrent update has already advanced the chain, the operation detects the stale hash and retries.
- **MAX_RETRIES = 3** with exponential backoff conceptually (retry is immediate in practice since Holochain operations are fast).
- **Scope**: `mutate_balance()` (currency-mint), `update_balance_after_exchange()` (TEND), `credit_sap()`/`debit_sap()` (payments), `credit_treasury()`/`debit_treasury()` (treasury), `contribute_to_pool()` (treasury), `contribute_to_commons()`/`receive_compost()`/`request_allocation()` (treasury).
- **97 call sites** use `follow_update_chain` across the workspace.

### 3. Deterministic Fork Resolution

When Holochain's update chain forks (two agents update the same entry concurrently), `follow_update_chain` in `mycelix-finance-shared` uses **lowest ActionHash wins** as the deterministic tiebreaker:

- At each hop, if multiple updates exist, the update with the lexicographically smallest `ActionHash` is selected.
- All nodes converge to the same fork branch regardless of DHT propagation order.
- This replaces the previous `updates.last()` approach which was non-deterministic across agents.

### 4. Create-Then-Verify (Idempotent Deduplication)

For operations that must happen exactly once (confirmations, ratings, disputes, redistribution), the pattern is:

- **Create** the entry and its link unconditionally.
- **Verify** via `get_links` that only one entry exists for the key. If duplicates are found, the later creator detects the conflict and the duplicate is effectively orphaned (no links point to it from the canonical anchor).
- Applied to: `confirm_minted_exchange`, `rate_exchange`, `open_minted_dispute`, `open_dispute`, `redistribute_compost`.
- This eliminates TOCTOU (Time-Of-Check-Time-Of-Use) vulnerabilities where two concurrent callers both pass the "does it exist?" check.

### 5. Crash Recovery

Two pending-adjustment entry types provide crash recovery for non-atomic multi-step operations:

- **`PendingBalanceAdjustment`** (TEND): Created before `confirm_exchange` updates provider and receiver balances. Each side is marked complete after its update succeeds. If the process crashes between the two updates, a governance-gated `recover_pending_adjustments(currency_id)` function can retry the incomplete side.
- **`PendingMintedAdjustment`** (currency-mint): Same pattern for minted currency exchange confirmations.
- Combined with idempotent confirmation (layer 4) and optimistic locking (layer 2), the zero-sum invariant is recoverable even after crashes mid-confirmation.

### 6. Input Validation

All `#[hdk_extern]` entry points validate inputs before any state mutation:

- **`validate_id()`**: Checks that string IDs are non-empty, within length bounds, contain only allowed characters (alphanumeric, hyphens, underscores, colons), and do not contain path traversal sequences. 31 call sites across 8 files.
- **`validate_did_format()`**: Checks that DID strings match the expected `did:` prefix format.
- **`verify_caller_is_did()`**: Confirms `agent_info().agent_initial_pubkey` matches the DID suffix, preventing DID spoofing in payment, exit, and exchange operations.

### 7. Governance Gating

Operations that require governance authorization verify proposal existence via cross-cluster bridge calls:

- **`verify_governance_agent()`** / **`verify_governance_or_bootstrap()`**: 38 call sites across 10 files. Checks the `GOVERNANCE_AGENTS_ANCHOR` link registry in 3 zomes (TEND, recognition, staking).
- **Bootstrap mode**: When no governance agents are registered (initial network bootstrap), any agent is permitted. Once the first governance agent is registered, all subsequent calls require authorization.
- **Cross-cluster bridge**: Governance proposal verification calls `CallTargetCell::OtherRole` to the governance cluster for operations like currency creation, treasury allocation execution, and SAP minting.

---

## Round-by-Round Summary

| Round | Focus | Key Changes |
|-------|-------|-------------|
| **1** | Types crate foundation | Created `mycelix-finance-types` with all entry types, status enums, constitutional constants. 68 unit tests. |
| **2** | Shared infrastructure | `mycelix-finance-shared` with `anchor_hash()`, `follow_update_chain()`, `verify_caller_is_did()`, `validate_id()`. 24 unit tests + 2 proptests. |
| **3** | Integrity zome validation | Wrote validation callbacks for all 7 integrity zomes. Status FSMs, immutable field checks, `is_finite()` guards. 218 integrity tests. |
| **4** | Query-to-link migration | Replaced all `query()` calls (source-chain-only) with anchor-based `get_links()` for DHT-wide visibility. Defined 72 link types across 7 zomes. |
| **5** | Link topology documentation | Mapped all anchor key patterns, cross-zome call paths, and read/write ownership. Produced `LINK_TOPOLOGY.md`. |
| **6** | Coordinator test suite | Integration tests for all 7 coordinator zomes using mock HDK. 186 coordinator tests covering CRUD, edge cases, error paths. |
| **7** | Property-based testing | Added 7 proptest properties (types + shared) for demurrage monotonicity, balance arithmetic, ID validation, and round-trip serialization. |
| **8** | Economics stress testing | 32 stress tests in `economics_stress_test.rs` covering overflow, underflow, rounding, extreme demurrage, and multi-currency edge cases. |
| **9** | Race condition audit | Systematic audit identifying 19 race conditions across 6 coordinator zomes. Produced `RACE_CONDITION_AUDIT.md` with severity ratings and fix recommendations. |
| **10** | Critical race fixes (first wave) | Fixed RC-10 (pure-read `get_minted_balance`), RC-16/RC-17 (status transition FSMs), RC-18 (immutable escrow signatures), RC-19 (create-then-verify dedup), deterministic fork resolution (lowest ActionHash). |
| **11** | Optimistic locking + remaining races | Fixed RC-1 through RC-8 (optimistic locking on all balance mutations), RC-11 (idempotent `redistribute_compost`), RC-12 through RC-15 (deterministic winner in get-or-create). All 19 RCs addressed. |
| **12** | Crash recovery + pending adjustments | Added `PendingBalanceAdjustment` (TEND) and `PendingMintedAdjustment` (currency-mint) for crash recovery. Governance-gated recovery functions. |
| **13** | Sweettest e2e + final audit | Added 22 sweettest scenarios (11 integration + 11 consciousness gating). Final clippy pass, `#![deny(unsafe_code)]` on all 16 crates, zero warnings. Produced `SWEETTEST_PLAN.md` with 9 additional scenarios. |

---

## Race Condition Fix Matrix

| RC | Zome | Severity | Pattern | Fix | Status |
|----|------|----------|---------|-----|--------|
| RC-1 | currency-mint | CRITICAL | `mutate_balance()` lost update | Optimistic locking with retry (round 11) | FIXED |
| RC-2 | currency-mint | CRITICAL | `update_minted_balance()` inherits RC-1 | Optimistic locking with retry (round 11) | FIXED |
| RC-3 | TEND | CRITICAL | `update_balance_after_exchange()` lost update | Optimistic locking with retry (round 11) | FIXED |
| RC-4 | TEND | CRITICAL | `update_hearth_balance()` lost update | Optimistic locking with retry (round 11) | FIXED |
| RC-5 | payments | CRITICAL | `credit_sap()`/`debit_sap()` double-spend | Optimistic locking with retry (round 11) | FIXED |
| RC-6 | treasury | CRITICAL | `credit_treasury()`/`debit_treasury()` lost update | Optimistic locking with retry (round 11) | FIXED |
| RC-7 | treasury | CRITICAL | `contribute_to_pool()` lost update | Optimistic locking with retry (round 11) | FIXED |
| RC-8 | treasury | CRITICAL | `request_allocation()` overdraw (25% reserve violation) | Optimistic locking with retry (round 11) | FIXED |
| RC-9 | currency-mint | HIGH | `apply_minted_demurrage()` double deduction | Mitigated by RC-10 fix (no read-triggered amplification) | MITIGATED |
| RC-10 | currency-mint | HIGH | `get_minted_balance()` side-effect on read | Pure read function, demurrage via explicit call only (round 10) | FIXED |
| RC-11 | currency-mint | HIGH | `redistribute_compost()` non-atomic multi-update | Idempotency guard (round 11) | FIXED |
| RC-12 | currency-mint | HIGH | `get_or_create_minted_balance()` duplicate creation | Deterministic lowest-ActionHash winner (round 11) | FIXED |
| RC-13 | TEND | HIGH | `get_or_create_balance()` duplicate creation | Deterministic lowest-ActionHash winner (round 11) | FIXED |
| RC-14 | TEND | HIGH | `get_or_create_hearth_balance()` duplicate creation | Deterministic lowest-ActionHash winner (round 11) | FIXED |
| RC-15 | recognition | HIGH | `get_or_create_allocation()` duplicate creation (limit bypass) | Deterministic lowest-ActionHash winner (round 11) | FIXED |
| RC-16 | currency-mint | MEDIUM | Currency lifecycle concurrent transitions | `CurrencyStatus.can_transition_to()` in integrity (round 10) | FIXED |
| RC-17 | staking | MEDIUM | Concurrent slash + unbond | `StakeStatus.can_transition_to()` + `EscrowStatus.can_transition_to()` (round 10) | FIXED |
| RC-18 | staking | MEDIUM | Escrow signature collection lost signatures | Signatures as immutable link entries (round 10) | FIXED |
| RC-19 | currency-mint, TEND | MEDIUM | Duplicate confirmation/dispute/rating TOCTOU | Create-then-verify pattern (round 10) | FIXED |

---

## Test Coverage

### By Category

| Category | Count | Location |
|----------|-------|----------|
| Types unit tests | 68 | `types/src/lib.rs` |
| Economics stress tests | 32 | `types/tests/economics_stress_test.rs` |
| Shared unit tests | 24 | `zomes/shared/src/lib.rs` |
| Property-based tests (proptest) | 7 | `types/src/lib.rs` (5), `zomes/shared/src/lib.rs` (2) |
| Integrity validation tests | 218 | 7 integrity zomes |
| Bridge coordinator tests | 23 | `tests/bridge_test.rs` |
| Currency-mint coordinator tests | 14 | `tests/currency_mint_test/*.rs` (7 files) |
| Payments coordinator tests | 44 | `tests/payments_test.rs` |
| Recognition coordinator tests | 21 | `tests/recognition_test.rs` |
| Staking coordinator tests | 24 | `tests/staking_test.rs` |
| TEND coordinator tests | 39 | `tests/tend_test.rs` |
| Treasury coordinator tests | 24 | `tests/treasury_test.rs` |
| Sweettest integration (e2e) | 11 | `tests/sweettest_integration.rs` |
| Sweettest consciousness gating (e2e) | 11 | `tests/sweettest_consciousness_gating.rs` |
| **Total** | **560+** | |

### Property-Based Tests (Proptest)

7 properties, each with 256 cases:

1. **Demurrage monotonicity** -- longer elapsed time produces larger deduction
2. **Demurrage non-negative** -- deductions are never negative
3. **Balance arithmetic identity** -- credit then debit returns to original
4. **Balance overflow safety** -- extreme values do not wrap or panic
5. **ID validation roundtrip** -- valid IDs always pass, invalid IDs always fail
6. **`anchor_hash` determinism** -- same input always produces same hash
7. **`anchor_hash` collision resistance** -- distinct inputs produce distinct hashes

### Edge Case Coverage

- **NaN/Inf injection**: 47 `is_finite()` guards reject malformed floats at the integrity level
- **Demurrage rounding**: Stress tests verify sub-cent precision over long time periods
- **Overflow**: `f64` arithmetic is checked for overflow in balance operations; proptest verifies no panics
- **Status transition exhaustiveness**: Every `(from, to)` pair in each status FSM is tested (valid transitions succeed, invalid transitions are rejected)
- **Empty/missing data**: Tests cover zero balances, empty link results, missing entries, and unregistered DIDs

---

## Remaining Risks

### 1. RC-9: Demurrage Double-Deduction (Low Severity)

Concurrent explicit `apply_minted_demurrage` calls can still race, potentially applying demurrage twice for the same period. **Mitigated** by the RC-10 fix (reads no longer trigger demurrage), so this can only occur if two cron jobs or governance calls fire simultaneously. In practice, demurrage is applied by a single periodic process.

**Severity**: Low. Worst case is a small over-deduction that can be corrected by governance.

### 2. No Rate Limiting on Public Functions

Holochain `#[hdk_extern]` functions have no built-in rate limiting. A malicious agent could flood the DHT with `record_exchange` or `recognize_member` calls. Holochain's validation rules reject invalid entries, but valid-but-spammy entries would propagate.

**Mitigation**: Consciousness gating limits write operations to agents with sufficient tier. Recognition has a per-cycle allocation cap (max 10). Credit limits bound exchange amounts. However, these are logical limits, not throughput limits.

### 3. Bridge Permissive Defaults When Governance Unreachable

If the governance cluster is unreachable (network partition, not yet installed), bridge calls to verify governance authorization fall back to permissive defaults. This is intentional for bootstrap but could be exploited during network instability.

**Mitigation**: `verify_governance_or_bootstrap()` checks for at least one registered governance agent. Once the governance cluster is operational and agents are registered, the bootstrap bypass is permanently disabled.

### 4. Sweettest E2E Gaps

22 sweettest scenarios exist, but the `SWEETTEST_PLAN.md` identifies 9 additional scenarios that are not yet implemented. In particular:

- **Multi-agent balance visibility** (query-to-link regression)
- **Concurrent exchange race** (RC-1 through RC-5 under real DHT conditions)
- **Stale-get regression** (follow_update_chain correctness across agents)

These scenarios require a running Holochain conductor and cannot be validated in unit tests.

### 5. Non-Atomic Provider/Receiver Balance Updates

Confirming an exchange updates two balances (provider credit, receiver debit) sequentially. If the process crashes between the two updates, the zero-sum invariant is temporarily broken. The `PendingBalanceAdjustment` mechanism (defense layer 5) makes this recoverable, but recovery requires a governance-gated function call -- it is not automatic.

---

## Architecture Decisions

### Why Optimistic Locking Over Append-Only Ledger

An **append-only balance ledger** (where each transaction creates an immutable `BalanceChange` entry and current balance is computed by summing the chain) would eliminate lost-update races entirely. However:

1. **O(n) balance queries**: Every `get_balance` call would need to scan all change entries. With active communities, this grows unbounded.
2. **Snapshot complexity**: Periodic snapshot entries would mitigate query cost but introduce their own consistency challenges.
3. **Holochain DHT cost**: Each append creates a new DHT entry. High-frequency operations (demurrage on every member, recognition cycles) would produce significant DHT load.
4. **Existing codebase**: The mutable-entry pattern was deeply embedded across all 7 zomes. A ledger rewrite would have been a ground-up redesign.

**Optimistic locking** provides sufficient concurrency safety for the expected load (community-scale, not exchange-scale) while preserving O(1) balance lookups and the existing entry structure. The retry loop (MAX_RETRIES=3) handles the rare concurrent-update case without blocking.

### Why Deterministic Lowest-Hash Over Timestamp-Based Resolution

When `follow_update_chain` encounters a fork (two updates to the same entry), we need a deterministic rule so all agents converge:

- **Timestamp-based**: Unreliable because Holochain agents have unsynchronized clocks. An agent with a skewed clock could systematically win or lose forks.
- **Lowest ActionHash**: Deterministic, requires no clock synchronization, and is uniformly distributed (ActionHash is a cryptographic hash). No agent can predict or manipulate which update will win.
- **Trade-off**: The "losing" update is silently discarded. In balance operations, this is acceptable because the optimistic locking retry loop will re-read and re-apply the change on top of the winner.

### Why Permissive Bridge Defaults

When the governance cluster is unreachable, bridge authorization calls return `Ok(true)` (permissive) rather than `Err` (restrictive). Rationale:

1. **Bootstrap problem**: A new network has no governance agents. If bridge calls were restrictive by default, no operations could proceed until governance is configured -- creating a chicken-and-egg deadlock.
2. **Availability over consistency**: In a community mutual credit system, blocking all financial operations during a temporary network partition is more harmful than allowing operations that might later be audited.
3. **Auditability**: All operations create DHT entries regardless of authorization mode. If unauthorized operations slip through during a partition, they are visible in the DHT and can be reversed by governance after connectivity is restored.
4. **Transition**: Once the first governance agent is registered via `register_governance_agent`, the bootstrap bypass is disabled permanently for that zome. This is a one-way transition from permissive to gated.

---

## File Reference

| File | Purpose |
|------|---------|
| `docs/RACE_CONDITION_AUDIT.md` | Detailed analysis of all 19 race conditions |
| `docs/SWEETTEST_PLAN.md` | E2E test scenarios (22 implemented + 9 planned) |
| `docs/LINK_TOPOLOGY.md` | Complete DHT link type and anchor key mapping |
| `types/src/lib.rs` | Entry types, status FSMs, constitutional constants, 68 tests + 5 proptests |
| `zomes/shared/src/lib.rs` | `anchor_hash`, `follow_update_chain`, `validate_id`, `verify_caller_is_did`, 24 tests + 2 proptests |
| `zomes/*/integrity/src/lib.rs` | Validation callbacks (7 zomes, 218 tests) |
| `zomes/*/coordinator/src/` | Coordinator logic with optimistic locking, crash recovery, governance gating |
| `tests/` | Integration and sweettest files (26 files, 342+ tests) |
| `types/tests/economics_stress_test.rs` | 32 stress tests for financial edge cases |
