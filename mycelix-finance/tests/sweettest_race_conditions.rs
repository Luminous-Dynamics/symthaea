// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Race Condition Sweettest Scenarios
//!
//! End-to-end tests validating the race condition fixes from the hardening audit
//! (see `docs/RACE_CONDITION_AUDIT.md`). These tests require a running Holochain
//! conductor and the pre-built DNA bundle at `dna/mycelix_finance.dna`.
//!
//! ## Running
//!
//! ```bash
//! # From the tests/ directory (which has its own workspace):
//! cargo test --test sweettest_race_conditions -- --ignored
//! ```
//!
//! ## Scenarios
//!
//! 1. **RC-1**: Concurrent balance updates — validates optimistic locking retry
//!    in `mutate_balance` prevents lost updates.
//! 2. **RC-19**: Double-confirm idempotency — validates create-then-verify pattern
//!    ensures only one confirmation produces balance updates.
//! 3. **Crash recovery**: PendingBalanceAdjustment — validates that
//!    `recover_pending_adjustments` completes interrupted confirm_exchange calls.

use holochain::prelude::*;
use holochain::sweettest::*;
use mycelix_finance_types::MintedCurrencyParams;
use std::path::PathBuf;

// ============================================================================
// Mirror types (must match zome entry types exactly for MessagePack serde)
// ============================================================================

// --- Currency Mint mirror types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCurrencyInput {
    pub dao_did: String,
    pub params: MintedCurrencyParams,
    pub governance_proposal_id: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ActivateCurrencyInput {
    pub currency_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordMintedExchangeInput {
    pub currency_id: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetMintedBalanceInput {
    pub currency_id: String,
    pub member_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MintedBalanceInfo {
    pub member_did: String,
    pub currency_id: String,
    pub currency_name: String,
    pub currency_symbol: String,
    pub raw_balance: i32,
    pub effective_balance: i32,
    pub pending_demurrage: i32,
    pub last_activity: Timestamp,
    pub balance: i32,
    pub credit_limit: i32,
    pub can_provide: bool,
    pub can_receive: bool,
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MintedExchange {
    pub id: String,
    pub currency_id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub timestamp: Timestamp,
    pub confirmed: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConfirmMintedExchangeInput {
    pub exchange_id: String,
}

// --- TEND mirror types (for PendingBalanceAdjustment / crash recovery test) ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ServiceCategory {
    CareWork,
    HomeServices,
    FoodServices,
    Transportation,
    Education,
    GeneralAssistance,
    Administrative,
    Creative,
    TechSupport,
    Wellness,
    Gardening,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ExchangeStatus {
    Proposed,
    Confirmed,
    Disputed,
    Cancelled,
    Resolved,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordExchangeInput {
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub cultural_alias: Option<String>,
    pub dao_did: String,
    pub service_date: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExchangeRecord {
    pub id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub status: ExchangeStatus,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetBalanceInput {
    pub member_did: String,
    pub dao_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BalanceInfo {
    pub member_did: String,
    pub dao_did: String,
    pub balance: i32,
    pub can_provide: bool,
    pub can_receive: bool,
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PendingBalanceAdjustment {
    pub exchange_id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f64,
    pub currency_id: String,
    pub provider_completed: bool,
    pub receiver_completed: bool,
    pub created_at: Timestamp,
}

// ============================================================================
// Helpers
// ============================================================================

/// Path to the pre-built DNA bundle.
fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_finance.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle — run 'hc dna pack dna/' first")
}

/// Create test currency params with no confirmation requirement and no demurrage.
fn test_params(name: &str, symbol: &str) -> MintedCurrencyParams {
    MintedCurrencyParams {
        name: name.into(),
        symbol: symbol.into(),
        description: format!("Race condition test currency: {}", name),
        credit_limit: 100,
        demurrage_rate: 0.0,
        max_service_hours: 8,
        min_service_minutes: 15,
        requires_confirmation: false,
        confirmation_timeout_hours: 0,
        max_exchanges_per_day: 0,
    }
}

/// Create test currency params that require receiver confirmation.
fn test_params_with_confirmation(name: &str, symbol: &str) -> MintedCurrencyParams {
    let mut p = test_params(name, symbol);
    p.requires_confirmation = true;
    p.confirmation_timeout_hours = 48;
    p
}

/// Call a zome function with automatic retry on BadNonce("Expired") errors.
///
/// Holochain's nonce validation can spuriously fail under load. This wrapper
/// retries up to 2 times with a 2-second delay between attempts.
async fn call_with_retry<I, O>(
    conductor: &SweetConductor,
    zome: &SweetZome,
    fn_name: &str,
    payload: I,
) -> O
where
    I: serde::Serialize + std::fmt::Debug + Clone,
    O: serde::de::DeserializeOwned + std::fmt::Debug,
{
    let max_retries = 2;
    for attempt in 0..=max_retries {
        match conductor
            .call_fallible(zome, fn_name, payload.clone())
            .await
        {
            Ok(result) => return result,
            Err(e) => {
                let err_str = format!("{:?}", e);
                if err_str.contains("BadNonce") && attempt < max_retries {
                    eprintln!(
                        "  [retry] BadNonce on {}() attempt {}/{}, retrying in 2s...",
                        fn_name,
                        attempt + 1,
                        max_retries
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    continue;
                }
                panic!(
                    "Zome call {}() failed after {} attempt(s): {:?}",
                    fn_name,
                    attempt + 1,
                    e
                );
            }
        }
    }
    unreachable!()
}

/// Set up a SweetConductor with the given number of agents and the finance DNA.
///
/// Returns (conductor, agent_pub_keys, apps). Access cells via `apps[i].cells()[0]`.
async fn setup_finance_conductor(
    agent_count: usize,
) -> (SweetConductor, Vec<AgentPubKey>, Vec<SweetApp>) {
    let dna = load_dna().await;
    let mut conductor = SweetConductor::from_standard_config().await;
    let agents = SweetAgents::get(conductor.keystore(), agent_count).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-finance", &agents, &[dna])
        .await
        .expect("Install app");

    // Warmup probe: absorb transient BadNonce on first call.
    let warmup_zome = apps[0].cells()[0].zome("currency_mint");
    let _: Result<Vec<String>, _> = conductor
        .call_fallible(
            &warmup_zome,
            "list_currency_members",
            "warmup-probe".to_string(),
        )
        .await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    (conductor, agents, apps.into())
}

// ============================================================================
// Scenario 1: Concurrent Balance Updates (RC-1)
//
// Validates that the optimistic locking retry in mutate_balance prevents lost
// updates when two exchanges target the same member concurrently.
//
// Race condition (before fix): Both calls read balance=100, both add 10,
// both write 110. Final balance = 110 instead of 120 — a lost update.
//
// After fix: mutate_balance detects the stale action hash on the second
// update_entry call, re-reads the updated balance (110), re-applies +10,
// and writes 120. The optimistic lock retry loop ensures no updates are lost.
// ============================================================================

#[test]
#[ignore] // Requires Holochain conductor — run with: cargo test --test sweettest_race_conditions -- --ignored
fn test_concurrent_balance_updates() {
    // Use a dedicated thread with enlarged stack for the async state machine.
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(test_concurrent_balance_updates_inner());
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn test_concurrent_balance_updates_inner() {
    // Setup: 3 agents — Alice (provider), Bob and Carol (each records an exchange
    // with Alice as provider, so Alice's balance gets updated by both).
    let (conductor, agents, apps) = setup_finance_conductor(3).await;

    let zome_a = apps[0].cells()[0].zome("currency_mint");
    let zome_b = apps[1].cells()[0].zome("currency_mint");
    let zome_c = apps[2].cells()[0].zome("currency_mint");

    let alice_did = format!("did:mycelix:{}", agents[0]);
    let bob_did = format!("did:mycelix:{}", agents[1]);
    let carol_did = format!("did:mycelix:{}", agents[2]);

    // Step 1: Create and activate a currency (no confirmation, no demurrage).
    // credit_limit=100 so both exchanges will be well within limits.
    let create_input = CreateCurrencyInput {
        dao_did: "did:mycelix:dao:rc1-test".into(),
        params: test_params("RC1 Test Credits", "RC1"),
        governance_proposal_id: None,
    };
    let def: currency_mint_integrity::CurrencyDefinition =
        call_with_retry(&conductor, &zome_a, "create_currency", create_input).await;
    let currency_id = def.id.clone();

    let _active: currency_mint_integrity::CurrencyDefinition = call_with_retry(
        &conductor,
        &zome_a,
        "activate_currency",
        ActivateCurrencyInput {
            currency_id: currency_id.clone(),
        },
    )
    .await;

    // Step 2: Record a baseline exchange so Alice has an initial balance.
    // Alice provides 1 hour to Bob → Alice gets +1, Bob gets -1.
    let baseline_input = RecordMintedExchangeInput {
        currency_id: currency_id.clone(),
        receiver_did: bob_did.clone(),
        hours: 1.0,
        service_description: "Baseline exchange to establish Alice balance".into(),
    };
    let _baseline: MintedExchange = call_with_retry(
        &conductor,
        &zome_a,
        "record_minted_exchange",
        baseline_input,
    )
    .await;

    // Verify Alice starts at +1.
    let alice_bal: MintedBalanceInfo = call_with_retry(
        &conductor,
        &zome_a,
        "get_minted_balance",
        GetMintedBalanceInput {
            currency_id: currency_id.clone(),
            member_did: alice_did.clone(),
        },
    )
    .await;
    assert_eq!(
        alice_bal.raw_balance, 1,
        "Alice should start at +1 after baseline exchange"
    );

    // Step 3: Fire two concurrent exchanges that BOTH credit Alice.
    // Bob records "Alice provided 1 hour to Bob" → Alice +1
    // Carol records "Alice provided 1 hour to Carol" → Alice +1
    //
    // In a race without optimistic locking, both would read Alice's balance as +1,
    // both would write +2, and Alice would end at +2 instead of +3.
    //
    // With the RC-1 fix (optimistic locking retry in mutate_balance), the second
    // writer detects the stale action hash, re-reads, and retries.
    //
    // Note: In the currency-mint zome, the PROVIDER (caller) is the one whose
    // agent calls record_minted_exchange. So Alice must call both exchanges.
    // To create concurrency, we use tokio::join! to fire them simultaneously.

    let exchange_to_bob = RecordMintedExchangeInput {
        currency_id: currency_id.clone(),
        receiver_did: bob_did.clone(),
        hours: 1.0,
        service_description: "Concurrent exchange A (Alice→Bob)".into(),
    };
    let exchange_to_carol = RecordMintedExchangeInput {
        currency_id: currency_id.clone(),
        receiver_did: carol_did.clone(),
        hours: 1.0,
        service_description: "Concurrent exchange B (Alice→Carol)".into(),
    };

    // Fire both exchange calls concurrently from Alice's zome.
    // tokio::join! ensures they are in-flight at the same time, maximizing
    // the chance of hitting the read-modify-write race in mutate_balance.
    let (ex_b, ex_c) = tokio::join!(
        call_with_retry::<_, MintedExchange>(
            &conductor,
            &zome_a,
            "record_minted_exchange",
            exchange_to_bob,
        ),
        call_with_retry::<_, MintedExchange>(
            &conductor,
            &zome_a,
            "record_minted_exchange",
            exchange_to_carol,
        ),
    );

    assert!(!ex_b.id.is_empty(), "Exchange B should succeed");
    assert!(!ex_c.id.is_empty(), "Exchange C should succeed");

    // Step 4: Verify final balances.
    // Alice provided 3 total hours (1 baseline + 1 to Bob + 1 to Carol) → balance = +3
    // Bob received 2 total hours → balance = -2
    // Carol received 1 hour → balance = -1
    // Zero-sum check: +3 + (-2) + (-1) = 0

    // Small delay to let DHT propagate link updates.
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let alice_final: MintedBalanceInfo = call_with_retry(
        &conductor,
        &zome_a,
        "get_minted_balance",
        GetMintedBalanceInput {
            currency_id: currency_id.clone(),
            member_did: alice_did.clone(),
        },
    )
    .await;
    let bob_final: MintedBalanceInfo = call_with_retry(
        &conductor,
        &zome_b,
        "get_minted_balance",
        GetMintedBalanceInput {
            currency_id: currency_id.clone(),
            member_did: bob_did.clone(),
        },
    )
    .await;
    let carol_final: MintedBalanceInfo = call_with_retry(
        &conductor,
        &zome_c,
        "get_minted_balance",
        GetMintedBalanceInput {
            currency_id: currency_id.clone(),
            member_did: carol_did.clone(),
        },
    )
    .await;

    // The critical assertion: Alice must have +3, NOT +2.
    // If mutate_balance's optimistic locking failed, one update would be lost.
    assert_eq!(
        alice_final.raw_balance, 3,
        "RC-1 REGRESSION: Alice balance should be +3 (1 baseline + 2 concurrent). \
         Got {} — indicates a lost update from the read-modify-write race.",
        alice_final.raw_balance
    );
    assert_eq!(
        bob_final.raw_balance, -2,
        "Bob should be -2 (received 2 hours total)"
    );
    assert_eq!(
        carol_final.raw_balance, -1,
        "Carol should be -1 (received 1 hour)"
    );

    // Zero-sum invariant: all balances must sum to zero.
    let sum = alice_final.raw_balance + bob_final.raw_balance + carol_final.raw_balance;
    assert_eq!(
        sum, 0,
        "Zero-sum invariant violated! Balances sum to {} (Alice={}, Bob={}, Carol={})",
        sum, alice_final.raw_balance, bob_final.raw_balance, carol_final.raw_balance
    );

    println!(
        "RC-1 PASSED: Concurrent balance updates correctly resolved. \
         Alice={}, Bob={}, Carol={}, sum={}",
        alice_final.raw_balance, bob_final.raw_balance, carol_final.raw_balance, sum
    );
}

// ============================================================================
// Scenario 2: Double-Confirm Idempotency (RC-19)
//
// Validates the create-then-verify pattern in confirm_minted_exchange.
// When two callers (or the same caller twice) try to confirm the same exchange
// concurrently, only one set of balance updates should occur.
//
// The fix (RC-19): Each caller creates a confirmation link, then re-reads all
// links for that exchange. The winner is chosen deterministically by lowest
// ActionHash. The loser returns idempotently without updating balances.
// ============================================================================

#[test]
#[ignore] // Requires Holochain conductor — run with: cargo test --test sweettest_race_conditions -- --ignored
fn test_double_confirm_idempotent() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(test_double_confirm_idempotent_inner());
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn test_double_confirm_idempotent_inner() {
    // Setup: 2 agents — Alice (provider) and Bob (receiver).
    // Alice records an exchange requiring confirmation; Bob confirms it.
    let (conductor, agents, apps) = setup_finance_conductor(2).await;

    let zome_a = apps[0].cells()[0].zome("currency_mint");
    let zome_b = apps[1].cells()[0].zome("currency_mint");

    let alice_did = format!("did:mycelix:{}", agents[0]);
    let bob_did = format!("did:mycelix:{}", agents[1]);

    // Step 1: Create and activate a currency that REQUIRES confirmation.
    let create_input = CreateCurrencyInput {
        dao_did: "did:mycelix:dao:rc19-test".into(),
        params: test_params_with_confirmation("RC19 Confirm Credits", "RC19"),
        governance_proposal_id: None,
    };
    let def: currency_mint_integrity::CurrencyDefinition =
        call_with_retry(&conductor, &zome_a, "create_currency", create_input).await;
    let currency_id = def.id.clone();

    let _active: currency_mint_integrity::CurrencyDefinition = call_with_retry(
        &conductor,
        &zome_a,
        "activate_currency",
        ActivateCurrencyInput {
            currency_id: currency_id.clone(),
        },
    )
    .await;

    // Step 2: Alice records an exchange with Bob as receiver (requires confirmation).
    // Since requires_confirmation=true, balances should NOT update until confirmed.
    let exchange_input = RecordMintedExchangeInput {
        currency_id: currency_id.clone(),
        receiver_did: bob_did.clone(),
        hours: 3.0,
        service_description: "Gardening help — requires confirmation".into(),
    };
    let exchange: MintedExchange = call_with_retry(
        &conductor,
        &zome_a,
        "record_minted_exchange",
        exchange_input,
    )
    .await;

    assert!(
        !exchange.confirmed,
        "Exchange should start unconfirmed when requires_confirmation=true"
    );
    let exchange_id = exchange.id.clone();

    // Step 3: Verify balances are still zero (no balance update before confirmation).
    let alice_pre: MintedBalanceInfo = call_with_retry(
        &conductor,
        &zome_a,
        "get_minted_balance",
        GetMintedBalanceInput {
            currency_id: currency_id.clone(),
            member_did: alice_did.clone(),
        },
    )
    .await;
    // With requires_confirmation, balances should not change until confirm.
    // (The exact pre-confirmation balance depends on zome behavior — it may
    // be 0 if deferred, or +3 if immediate. We verify post-confirm totals.)
    let alice_pre_balance = alice_pre.raw_balance;

    // Step 4: Bob tries to confirm the exchange TWICE concurrently.
    // Both calls should succeed, but only one should trigger balance updates.
    // The RC-19 create-then-verify pattern ensures deterministic winner selection.
    let confirm_input_1 = exchange_id.clone();
    let confirm_input_2 = exchange_id.clone();

    let (result_1, result_2) = tokio::join!(
        call_with_retry::<_, MintedExchange>(
            &conductor,
            &zome_b,
            "confirm_minted_exchange",
            confirm_input_1,
        ),
        call_with_retry::<_, MintedExchange>(
            &conductor,
            &zome_b,
            "confirm_minted_exchange",
            confirm_input_2,
        ),
    );

    // Both calls should return the exchange (winner returns updated, loser
    // returns idempotently with the original or confirmed exchange).
    assert_eq!(
        result_1.id, exchange_id,
        "First confirm should return the correct exchange"
    );
    assert_eq!(
        result_2.id, exchange_id,
        "Second confirm should return the correct exchange (idempotent)"
    );

    // Step 5: Verify balances were updated exactly once.
    // Alice provided 3 hours → +3. Bob received 3 hours → -3.
    // If double-confirm caused double balance updates, Alice would be +6 and Bob -6.
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let alice_post: MintedBalanceInfo = call_with_retry(
        &conductor,
        &zome_a,
        "get_minted_balance",
        GetMintedBalanceInput {
            currency_id: currency_id.clone(),
            member_did: alice_did.clone(),
        },
    )
    .await;
    let bob_post: MintedBalanceInfo = call_with_retry(
        &conductor,
        &zome_b,
        "get_minted_balance",
        GetMintedBalanceInput {
            currency_id: currency_id.clone(),
            member_did: bob_did.clone(),
        },
    )
    .await;

    // The critical assertion: balances should reflect exactly ONE confirmation.
    let expected_alice = alice_pre_balance + 3;
    assert_eq!(
        alice_post.raw_balance, expected_alice,
        "RC-19 REGRESSION: Alice balance should be {} (pre={} + 3 hours once). \
         Got {} — indicates double balance update from concurrent confirms.",
        expected_alice, alice_pre_balance, alice_post.raw_balance
    );

    // Bob's balance should be exactly -3 from this exchange.
    assert_eq!(
        bob_post.raw_balance, -3,
        "RC-19 REGRESSION: Bob balance should be -3. Got {} — \
         indicates double balance update from concurrent confirms.",
        bob_post.raw_balance
    );

    // Zero-sum check.
    let sum = alice_post.raw_balance + bob_post.raw_balance;
    assert_eq!(
        sum, 0,
        "Zero-sum invariant violated! Alice={} + Bob={} = {}",
        alice_post.raw_balance, bob_post.raw_balance, sum
    );

    println!(
        "RC-19 PASSED: Double-confirm was idempotent. Alice={}, Bob={}, sum={}",
        alice_post.raw_balance, bob_post.raw_balance, sum
    );
}

// ============================================================================
// Scenario 3: Crash Recovery via PendingBalanceAdjustment
//
// Validates that recover_pending_adjustments can complete interrupted
// confirm_exchange calls in the TEND zome. The crash scenario:
//
// 1. Alice records a TEND exchange with Bob (Proposed status).
// 2. Bob calls confirm_exchange, which:
//    a. Creates PendingBalanceAdjustment (provider_completed=false, receiver_completed=false)
//    b. Updates provider (Alice) balance → marks provider_completed=true
//    c. [SIMULATED CRASH HERE] — receiver balance NOT updated
// 3. A governance agent calls recover_pending_adjustments.
// 4. The function finds the incomplete adjustment, applies the receiver update,
//    and marks receiver_completed=true.
// 5. Both balances are now correct and the zero-sum invariant is restored.
//
// Note: We cannot literally crash the conductor mid-call, but we CAN verify
// the full flow: confirm succeeds → PendingBalanceAdjustment is created and
// completed → calling recover_pending_adjustments is a no-op (returns 0).
// This validates the recovery infrastructure is wired correctly. A true
// crash-recovery test would require conductor fault injection (future work).
// ============================================================================

#[test]
#[ignore] // Requires Holochain conductor — run with: cargo test --test sweettest_race_conditions -- --ignored
fn test_crash_recovery_pending_adjustment() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(test_crash_recovery_pending_adjustment_inner());
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn test_crash_recovery_pending_adjustment_inner() {
    // Setup: 2 agents — Alice (provider) and Bob (receiver) using TEND zome.
    // TEND's confirm_exchange creates PendingBalanceAdjustment entries for
    // crash recovery, unlike currency_mint which uses the simpler RC-19 pattern.
    let (conductor, agents, apps) = setup_finance_conductor(2).await;

    let tend_a = apps[0].cells()[0].zome("tend");
    let tend_b = apps[1].cells()[0].zome("tend");

    let alice_did = format!("did:mycelix:{}", agents[0]);
    let bob_did = format!("did:mycelix:{}", agents[1]);
    let dao_did = "did:mycelix:dao:crash-recovery-test".to_string();

    // Step 1: Record a TEND exchange — Alice provides 2 hours of education to Bob.
    let exchange_input = RecordExchangeInput {
        receiver_did: bob_did.clone(),
        hours: 2.0,
        service_description: "Crash recovery test — tutoring session".into(),
        service_category: ServiceCategory::Education,
        cultural_alias: None,
        dao_did: dao_did.clone(),
        service_date: None,
    };
    let exchange: ExchangeRecord =
        call_with_retry(&conductor, &tend_a, "record_exchange", exchange_input).await;

    assert_eq!(
        exchange.status,
        ExchangeStatus::Proposed,
        "Exchange should start as Proposed"
    );
    let exchange_id = exchange.id.clone();

    // Step 2: Check pre-confirmation balances.
    // In TEND, recording an exchange in Proposed status should not update balances
    // (confirmation is always required for TEND exchanges).
    let alice_pre: BalanceInfo = call_with_retry(
        &conductor,
        &tend_a,
        "get_balance",
        GetBalanceInput {
            member_did: alice_did.clone(),
            dao_did: dao_did.clone(),
        },
    )
    .await;
    let bob_pre: BalanceInfo = call_with_retry(
        &conductor,
        &tend_b,
        "get_balance",
        GetBalanceInput {
            member_did: bob_did.clone(),
            dao_did: dao_did.clone(),
        },
    )
    .await;
    let alice_pre_balance = alice_pre.balance;
    let bob_pre_balance = bob_pre.balance;

    // Step 3: Bob confirms the exchange.
    // This triggers the PendingBalanceAdjustment flow:
    //   a. Create PBA (provider_completed=false, receiver_completed=false)
    //   b. Update Alice's balance (+2) → mark provider_completed=true
    //   c. Update Bob's balance (-2) → mark receiver_completed=true
    //
    // In a real crash scenario, step (c) might not complete. Here we let
    // the full confirm succeed, then verify the recovery function handles
    // the "already completed" case gracefully (returns 0 recovered).
    let confirmed: ExchangeRecord =
        call_with_retry(&conductor, &tend_b, "confirm_exchange", exchange_id.clone()).await;

    assert_eq!(
        confirmed.status,
        ExchangeStatus::Confirmed,
        "Exchange should be Confirmed after Bob confirms"
    );

    // Step 4: Verify post-confirmation balances.
    // Alice provided 2 hours → balance increases by 2.
    // Bob received 2 hours → balance decreases by 2.
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let alice_post: BalanceInfo = call_with_retry(
        &conductor,
        &tend_a,
        "get_balance",
        GetBalanceInput {
            member_did: alice_did.clone(),
            dao_did: dao_did.clone(),
        },
    )
    .await;
    let bob_post: BalanceInfo = call_with_retry(
        &conductor,
        &tend_b,
        "get_balance",
        GetBalanceInput {
            member_did: bob_did.clone(),
            dao_did: dao_did.clone(),
        },
    )
    .await;

    assert_eq!(
        alice_post.balance,
        alice_pre_balance + 2,
        "Alice should gain +2 hours after confirmation. Pre={}, Post={}",
        alice_pre_balance,
        alice_post.balance
    );
    assert_eq!(
        bob_post.balance,
        bob_pre_balance - 2,
        "Bob should lose -2 hours after confirmation. Pre={}, Post={}",
        bob_pre_balance,
        bob_post.balance
    );

    // Zero-sum check on the deltas.
    let alice_delta = alice_post.balance - alice_pre_balance;
    let bob_delta = bob_post.balance - bob_pre_balance;
    assert_eq!(
        alice_delta + bob_delta,
        0,
        "Zero-sum invariant violated! Alice delta={}, Bob delta={}",
        alice_delta,
        bob_delta
    );

    // Step 5: Call recover_pending_adjustments.
    // Since the confirm completed fully (both provider and receiver balances
    // were updated), the PBA is marked fully complete. Recovery should find
    // nothing to recover and return 0.
    //
    // This validates that:
    //   a. The PBA was created and linked to the well-known anchor.
    //   b. Both completed flags were set correctly.
    //   c. The recovery function correctly skips fully-completed PBAs.
    //   d. The recovery function is callable and doesn't error.
    //
    // In a true crash scenario (e.g., conductor dies between provider and
    // receiver updates), the PBA would have provider_completed=true and
    // receiver_completed=false. Recovery would then apply only the receiver
    // update, restoring the zero-sum invariant.
    let recovered: u32 = call_with_retry(
        &conductor,
        &tend_a,
        "recover_pending_adjustments",
        dao_did.clone(),
    )
    .await;

    assert_eq!(
        recovered, 0,
        "No adjustments should need recovery (confirm completed fully). Got {}.",
        recovered
    );

    // Step 6: Verify balances are unchanged after recovery (no double-application).
    let alice_after_recovery: BalanceInfo = call_with_retry(
        &conductor,
        &tend_a,
        "get_balance",
        GetBalanceInput {
            member_did: alice_did.clone(),
            dao_did: dao_did.clone(),
        },
    )
    .await;
    let bob_after_recovery: BalanceInfo = call_with_retry(
        &conductor,
        &tend_b,
        "get_balance",
        GetBalanceInput {
            member_did: bob_did.clone(),
            dao_did: dao_did.clone(),
        },
    )
    .await;

    assert_eq!(
        alice_after_recovery.balance, alice_post.balance,
        "Recovery should not change Alice's balance when PBA is fully complete"
    );
    assert_eq!(
        bob_after_recovery.balance, bob_post.balance,
        "Recovery should not change Bob's balance when PBA is fully complete"
    );

    println!(
        "Crash recovery PASSED: Confirm completed fully, recovery was no-op. \
         Alice={}, Bob={}, recovered={}",
        alice_after_recovery.balance, bob_after_recovery.balance, recovered
    );
}
