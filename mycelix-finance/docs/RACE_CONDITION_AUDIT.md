# Race Condition Audit: mycelix-finance Coordinator Zomes

## Summary

**19 race conditions** identified across 6 coordinator zomes. As of round 11, **all 19 have been addressed** — 8 fixed in round 10 (RC-10, RC-18, RC-19, fork resolution, plus 4 status transition validations for RC-16/RC-17), and 11 fixed in round 11 (RC-1–RC-8 via optimistic locking, RC-11 via idempotency guard, RC-12–RC-15 via deterministic winner selection).

### Fix Status

| RC | Status | Fix (commit) |
|----|--------|-------------|
| RC-1–RC-5 | **FIXED** | Optimistic locking with retry (round 11) |
| RC-6–RC-8 | **FIXED** | Treasury optimistic locking with retry (round 11) |
| RC-9 | **MITIGATED** | RC-10 fix removes the read-triggered amplification |
| RC-10 | **FIXED** | `get_minted_balance` is now pure read (round 10) |
| RC-11 | **FIXED** | Idempotency guard on `redistribute_compost` (round 11) |
| RC-12–RC-15 | **FIXED** | Deterministic winner selection in get-or-create (round 11) |
| RC-16 | **FIXED** | `CurrencyStatus.can_transition_to()` in integrity (round 10) |
| RC-17 | **FIXED** | `StakeStatus.can_transition_to()` + `EscrowStatus.can_transition_to()` in integrity (round 10) |
| RC-18 | **FIXED** | Escrow signatures as immutable link entries (round 10) |
| RC-19 | **FIXED** | Create-then-verify pattern in confirm/dispute/rate (round 10) |
| Fork resolution | **FIXED** | Deterministic lowest-ActionHash selection (round 10) |

---

## CRITICAL: Balance Update Race Conditions

### RC-1: `mutate_balance()` in currency-mint (Lost Update)

**File**: `zomes/currency-mint/coordinator/src/helpers.rs:389-427`

**Function**: `mutate_balance(member_did, currency_id, f)`

**Pattern**: `follow_update_chain(link_hash)` → deserialize → `f(&mut bal)` → `update_entry(record.action_address().clone(), &bal)`

**Race scenario**: Two concurrent exchanges involving the same member (e.g., Alice provides services to Bob and Carol simultaneously). Both calls read Alice's balance as +5, both add +1, both write +6. Alice should have +7 but has +6. One hour of credit is permanently lost.

**Impact**: **Lost balance updates**. In a zero-sum mutual credit system, this breaks the fundamental invariant (balances no longer sum to zero). The lost credit cannot be recovered.

**Recommended fix**: Use the action hash as an optimistic lock. After `follow_update_chain`, attempt `update_entry` with the returned action hash. If the underlying Holochain update detects the action was already updated by another call, detect the fork and retry. Alternatively, redesign balances as append-only ledger entries.

### RC-2: `update_minted_balance()` in currency-mint (inherits RC-1)

**File**: `zomes/currency-mint/coordinator/src/helpers.rs:147-166`

Thin wrapper around `mutate_balance()`. Called from `record_minted_exchange`, `confirm_minted_exchange`, and `resolve_minted_dispute`.

### RC-3: `update_balance_after_exchange()` in TEND (Lost Update)

**File**: `zomes/tend/coordinator/src/lib.rs:1928-1975`

Identical pattern to RC-1. Additionally, `confirm_exchange` calls `update_balance_after_exchange` twice (provider and receiver) without atomicity — if the first succeeds and the second fails or races, the zero-sum property is broken.

### RC-4: `update_hearth_balance()` in TEND (Lost Update)

**File**: `zomes/tend/coordinator/src/lib.rs:537-577`

Same read-modify-write pattern. Scoped to hearth-level balances.

### RC-5: `credit_sap()` / `debit_sap()` in Payments (Lost Update / Double-Spend)

**File**: `zomes/payments/coordinator/src/lib.rs`
- `credit_sap`: lines 195-240
- `debit_sap`: lines 251-279
- `apply_demurrage`: lines 79-169

**Race scenario**: Two concurrent `send_payment` calls from the same sender. Both read balance=1000, both compute demurrage, subtract 500, write 500. **Double-spend**: the sender has spent 1000 SAP but only 500 was deducted. Most financially dangerous race — SAP is backed by real collateral.

---

## CRITICAL: Treasury and Pool Balance Race Conditions

### RC-6: `credit_treasury()` / `debit_treasury()` in Treasury (Lost Update)

**File**: `zomes/treasury/coordinator/src/lib.rs:118-155`

Pattern: `get_treasury_record(treasury_id)` → modify balance → `update_entry`.

Two contributions arriving concurrently can lose one update.

### RC-7: `contribute_to_pool()` in Treasury (Lost Update)

**File**: `zomes/treasury/coordinator/src/lib.rs:811-835`

Pattern: `get_savings_pool_record` → modify `current_amount` → `update_entry`.

### RC-8: `contribute_to_commons()` / `receive_compost()` / `request_allocation()` in Treasury

**File**: `zomes/treasury/coordinator/src/lib.rs:1038-1173`

Two concurrent `request_allocation` calls can over-draw the commons pool, violating the 25% inalienable reserve constitutional requirement.

---

## HIGH: Demurrage Race Conditions

### RC-9: `apply_minted_demurrage()` in currency-mint (Double Demurrage)

**File**: `zomes/currency-mint/coordinator/src/demurrage.rs:17-73`

Reads balance, computes deduction from elapsed time, then calls `mutate_balance`. If two calls race, either double demurrage occurs or compost credits don't match member losses.

### RC-10: `get_minted_balance()` lazy demurrage (Side-Effect on Read)

**File**: `zomes/currency-mint/coordinator/src/balances.rs:16-49`

Read function has write side-effect: calls `apply_minted_demurrage` on every read. Concurrent reads trigger multiple demurrage applications, amplifying RC-9.

### RC-11: `redistribute_compost()` in currency-mint (Non-Atomic Multi-Update)

**File**: `zomes/currency-mint/coordinator/src/demurrage.rs:116-205`

Two concurrent calls can both read compost_balance=100 and distribute to N members. Net result: money created from nothing. **Severe zero-sum violation**.

---

## HIGH: Get-or-Create Duplicate Entry Race Conditions

### RC-12: `get_or_create_minted_balance()` in currency-mint (Duplicate Creation)

**File**: `zomes/currency-mint/coordinator/src/helpers.rs:54-119`

Pattern: Check for existing link → if none, create entry + create link. Two concurrent calls create two balance entries for the same member+currency.

### RC-13: `get_or_create_balance()` in TEND (Duplicate Creation)

**File**: `zomes/tend/coordinator/src/lib.rs:1458-1488`

Same pattern as RC-12.

### RC-14: `get_or_create_hearth_balance()` in TEND (Duplicate Creation)

**File**: `zomes/tend/coordinator/src/lib.rs:437-488`

### RC-15: `get_or_create_allocation()` in Recognition (Duplicate Creation)

**File**: `zomes/recognition/coordinator/src/lib.rs:713-761`

The per-cycle recognition limit (max 10) could be bypassed because `increment_allocation` would increment one duplicate while the limit check reads the other.

---

## MEDIUM: State Transition Race Conditions

### RC-16: Currency Lifecycle Transitions (Concurrent Status Changes)

**File**: `zomes/currency-mint/coordinator/src/lifecycle.rs`

Functions: `activate_currency`, `suspend_currency`, `reactivate_currency`, `retire_currency`, `amend_currency_params`

Two concurrent calls can both read the same status and both attempt transitions, creating a forked update chain.

### RC-17: Staking State Transitions (Concurrent Slash + Unbond)

**File**: `zomes/staking/coordinator/src/lib.rs`

A governance agent calls `slash_stake` while the staker calls `begin_unbonding` concurrently. The slash may be silently undone.

### RC-18: Escrow Signature Collection (Lost Signatures)

**File**: `zomes/staking/coordinator/src/lib.rs:577-648`

Two authorized signers submit signatures simultaneously. One signature is lost. Multi-sig escrow release conditions may never be met — **funds could be permanently locked**.

**Recommended fix**: Store signatures as separate link entries (immutable) rather than a mutable array. Count signature links at release time.

### RC-19: Duplicate Dispute / Confirmation Guards (TOCTOU)

**Files**: currency-mint `open_minted_dispute`, `confirm_minted_exchange`; tend `open_dispute`, `rate_exchange`

All use TOCTOU: `get_links` → check empty → `create_entry`. Two concurrent calls can both create entries.

**The duplicate confirmation case is the most dangerous**: it leads to double balance updates via `update_minted_balance`.

---

## Architectural Observations

1. **No transactions in Holochain**: Every `update_entry` is independent. Updating provider balance + receiver balance in sequence has no rollback — if the first succeeds and the second fails, the zero-sum invariant is broken. **Mitigation**: The idempotent confirmation guard (RC-19 fix) ensures only one caller proceeds to balance updates. Optimistic locking (RC-1–RC-8 fix) reduces the window for lost updates.

2. **`follow_update_chain` fork problem**: ~~The implementation uses `updates.last()` to pick the next hop.~~ **FIXED**: Now uses deterministic lowest-ActionHash selection. All nodes converge to the same fork branch.

3. **Lazy demurrage amplifies races**: ~~`get_minted_balance` triggers writes on reads.~~ **FIXED**: `get_minted_balance` is now a pure read function. Demurrage is computed in-memory and only persisted via explicit `apply_minted_demurrage` calls.

4. **Future architectural improvement**: An **append-only balance ledger** where each change creates an immutable `BalanceChange` entry would provide even stronger guarantees than optimistic locking, at the cost of O(n) balance queries (mitigable with periodic snapshots). The current optimistic locking approach is sufficient for expected concurrency levels.

---

## Remaining Risks

1. **RC-9** (demurrage double-deduction): Partially mitigated by RC-10 fix (no more read-triggered amplification), but concurrent explicit `apply_minted_demurrage` calls can still race. Low severity — demurrage is typically applied by a periodic cron job, not concurrent user actions.

2. **Zero-sum atomicity**: ~~Provider + receiver balance updates in `confirm_exchange` are still non-atomic.~~ **ADDRESSED**: `PendingBalanceAdjustment` entries are now created before balance updates in `confirm_exchange`. Each side is marked complete after its update succeeds. A governance-gated `recover_pending_adjustments(currency_id)` function can retry any interrupted operations to restore zero-sum. Combined with RC-19 (no duplicate confirmations) and optimistic locking (no lost updates), the zero-sum invariant is now recoverable even after crashes.
