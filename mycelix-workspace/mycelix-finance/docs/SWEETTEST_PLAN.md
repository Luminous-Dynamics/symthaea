# Sweettest E2E Test Plan

Priority sweettest scenarios targeting multi-agent correctness after the
query→link migration and hardening work. These tests verify behavior that
**unit tests cannot cover** — specifically DHT visibility across agents.

## Running

```bash
cargo test --test sweettest_integration -- --ignored
cargo test --test sweettest_consciousness_gating -- --ignored
```

Requires Holochain conductor via `nix develop`.

## Existing Coverage (22 tests)

**sweettest_integration.rs** (11 tests):
- Exchange recording, balance queries, service listings
- Recognition lifecycle (initialize, recognize, score computation)
- Treasury (commons pool, contributions, inalienable reserve)
- Complete lifecycle (multi-step flow)

**sweettest_consciousness_gating.rs** (11 tests):
- DID-match enforcement on payments/deposits/collateral
- Governance mint authorization
- Read vs write gating
- Fee tier reflection
- Cache consistency under repeated failures

## New Scenarios Needed

### 1. Multi-Agent Balance Visibility (HIGH PRIORITY)

**Why**: The query→link migration was specifically to fix cross-agent
visibility. Old `query()` only searched the caller's source chain.

```
Agent A: record_minted_exchange(provider=A, receiver=B, hours=2)
Agent B: get_minted_balance(B) → should see -2
Agent C: get_currency_exchanges(currency) → should see the exchange
```

### 2. Duplicate ID Prevention (HIGH PRIORITY)

**Why**: We added duplicate-ID checks on create_treasury,
create_savings_pool, create_commons_pool, open_payment_channel.

```
Agent A: create_treasury("treasury-1") → Ok
Agent A: create_treasury("treasury-1") → Err("already exists")
Agent B: create_treasury("treasury-1") → Err("already exists")
```

### 3. Confirmation Flow (Multi-Agent)

**Why**: Exchange confirmation requires receiver agent to call
confirm_minted_exchange, which updates balances via links.

```
Agent A: record_minted_exchange(receiver=B, hours=3, requires_confirmation=true)
Agent A: get_minted_balance(A) → 0 (not confirmed yet)
Agent B: list_pending_for_receiver(B) → [exchange]
Agent B: confirm_minted_exchange(id) → Ok
Agent A: get_minted_balance(A) → +3
Agent B: get_minted_balance(B) → -3
Agent B: list_pending_for_receiver(B) → [] (removed from pending)
```

### 4. Cancellation After Timeout

```
Agent A: record_minted_exchange(receiver=B, timeout=1hr)
# Wait or mock time
Agent A: cancel_expired_exchange(id) → Ok
Agent B: list_pending_for_receiver(B) → [] (removed)
Agent B: confirm_minted_exchange(id) → should fail (link deleted)
```

### 5. Treasury Savings Pool (Multi-Agent)

```
Agent A: create_treasury("t1") + create_savings_pool("pool1")
Agent B: join_savings_pool("pool1") → Ok (finds pool via link)
Agent B: contribute_to_pool("pool1", 100) → Ok
Agent A: get_savings_pool("pool1") → sees B as member, total=100
```

### 6. Payment Channel Lifecycle (Multi-Agent)

```
Agent A: open_payment_channel(party_a=A, party_b=B)
Agent B: channel_transfer(channel_id, amount=50) → Ok
Agent A: get_channels(A) → sees channel with updated balance
Agent A: close_payment_channel(channel_id) → Ok
```

### 7. Collateral Bridge Deposit (Multi-Agent)

```
Agent A: deposit_collateral(ETH, 1000, rate=2.0) → Pending
Agent A: confirm_deposit(id) → Confirmed, SAP minted
Agent B: get_deposit(id) → sees Confirmed status (via link)
```

### 8. Stale-Get Regression

Verify that `follow_update_chain` works correctly after updates:

```
Agent A: create_entry + update_entry (e.g., update pool member list)
Agent B: get_entry → should see UPDATED version, not original
```

### 9. Concurrent Exchange Race (STRETCH)

```
Agent A: record_minted_exchange(provider=A, receiver=B, hours=39)
Agent C: record_minted_exchange(provider=A, receiver=D, hours=39)
# If credit_limit=40, at most one should succeed
# (Currently both may succeed — see concurrent update audit)
```

## Implementation Notes

- Use `SweetConductor::from_standard()` with 2-3 agents
- Mirror types already exist in `sweettest_integration.rs` — reuse
- Each test is `#[ignore]` (requires conductor) and `#[tokio::test(flavor = "multi_thread")]`
- DNA path: `mycelix-finance/dna/mycelix_finance.dna`
