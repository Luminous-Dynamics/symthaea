//! Lifecycle tests: Draft → Active → Suspended → Active → Retired
//!
//! Coverage:
//!   create_currency         → test_create_and_activate_currency (1.1)
//!   activate_currency       → test_create_and_activate_currency (1.1), test_full_lifecycle (1.2)
//!   suspend_currency        → test_full_lifecycle (1.2)
//!   reactivate_currency     → test_full_lifecycle (1.2)
//!   retire_currency         → test_full_lifecycle (1.2), test_draft_cannot_retire (1.3)
//!   get_currency            → test_create_and_activate_currency (1.1)
//!   cancel_expired_exchange → test_cancel_exchange_guards (14.1)
//!   (lifecycle guards)      → test_redundant_lifecycle_transitions (14.2)

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;
use holochain::sweettest::*;
use mycelix_finance_types::CurrencyStatus;

/// Test 1.1: Create currency in Draft, activate, verify status transitions
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_create_and_activate_currency() {
    println!("Test 1.1: Create and Activate Currency");

    let (mut conductor, _agents, apps) = setup_finance_conductor(1).await;
    let cell = &apps[0].cells()[0];
    let zome = cell.zome("currency_mint");

    // Create currency (Draft)
    let input = CreateCurrencyInput {
        dao_did: "did:mycelix:dao:test-hearth-1".into(),
        params: test_params("Garden Hours", "GH"),
        governance_proposal_id: None,
    };

    let def: CurrencyDefinition = conductor.call(&zome, "create_currency", input).await;
    assert_eq!(def.status, CurrencyStatus::Draft);
    assert_eq!(def.params.symbol, "GH");
    println!("  - Created draft: {} ({})", def.params.name, def.id);

    // Activate
    let activate = ActivateCurrencyInput {
        currency_id: def.id.clone(),
    };
    let active: CurrencyDefinition = conductor.call(&zome, "activate_currency", activate).await;
    assert_eq!(active.status, CurrencyStatus::Active);
    println!("  - Activated: {:?}", active.status);

    // Verify via get_currency
    let fetched: Option<CurrencyDefinition> =
        conductor.call(&zome, "get_currency", def.id.clone()).await;
    assert!(fetched.is_some());
    println!("Test 1.1 PASSED");
}

/// Test 1.2: Full lifecycle — Draft → Active → Suspended → Active → Retired
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_full_lifecycle() {
    println!("Test 1.2: Full Lifecycle");

    let (mut conductor, _agents, apps) = setup_finance_conductor(1).await;
    let cell = &apps[0].cells()[0];
    let zome = cell.zome("currency_mint");

    let input = CreateCurrencyInput {
        dao_did: "did:mycelix:dao:lifecycle-test".into(),
        params: test_params("Lifecycle Token", "LT"),
        governance_proposal_id: None,
    };

    let def: CurrencyDefinition = conductor.call(&zome, "create_currency", input).await;
    let cid = def.id.clone();
    assert_eq!(def.status, CurrencyStatus::Draft);

    // Draft → Active
    let active: CurrencyDefinition = conductor
        .call(
            &zome,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: cid.clone(),
            },
        )
        .await;
    assert_eq!(active.status, CurrencyStatus::Active);

    // Active → Suspended
    let suspended: CurrencyDefinition =
        conductor.call(&zome, "suspend_currency", cid.clone()).await;
    assert_eq!(suspended.status, CurrencyStatus::Suspended);

    // Suspended → Active
    let reactivated: CurrencyDefinition = conductor
        .call(&zome, "reactivate_currency", cid.clone())
        .await;
    assert_eq!(reactivated.status, CurrencyStatus::Active);

    // Active → Retired
    let retired: CurrencyDefinition =
        conductor.call(&zome, "retire_currency", cid.clone()).await;
    assert_eq!(retired.status, CurrencyStatus::Retired);

    println!("  - Draft → Active → Suspended → Active → Retired: all transitions verified");
    println!("Test 1.2 PASSED");
}

/// Test 1.3: Draft cannot be retired directly
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_draft_cannot_retire() {
    println!("Test 1.3: Draft Cannot Retire");

    let (mut conductor, _agents, apps) = setup_finance_conductor(1).await;
    let cell = &apps[0].cells()[0];
    let zome = cell.zome("currency_mint");

    let input = CreateCurrencyInput {
        dao_did: "did:mycelix:dao:draft-retire-test".into(),
        params: test_params("No Shortcut", "NS"),
        governance_proposal_id: None,
    };

    let def: CurrencyDefinition = conductor.call(&zome, "create_currency", input).await;

    // Attempt to retire Draft — should fail
    let result: Result<CurrencyDefinition, _> = conductor
        .call_fallible(&zome, "retire_currency", def.id.clone())
        .await;

    assert!(result.is_err(), "Draft → Retired should be rejected");
    println!("  - Draft → Retired correctly rejected");
    println!("Test 1.3 PASSED");
}

/// Test 14.1: Cancel confirmed exchange fails; cancel unconfirmed before timeout fails
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cancel_exchange_guards() {
    println!("Test 14.1: Cancel Exchange Guards");

    let (mut conductor, agents, apps) = setup_finance_conductor(2).await;
    let cell_a = &apps[0].cells()[0];
    let zome_a = cell_a.zome("currency_mint");
    let bob_did = format!("did:mycelix:{}", agents[1]);

    // Auto-confirm currency (no confirmation needed)
    let def: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:cancel-test".into(),
                params: test_params("CancelCoin", "XC"),
                governance_proposal_id: None,
            },
        )
        .await;
    let _: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def.id.clone(),
            },
        )
        .await;

    // Record auto-confirmed exchange
    let exchange: MintedExchange = conductor
        .call(
            &zome_a,
            "record_minted_exchange",
            RecordMintedExchangeInput {
                currency_id: def.id.clone(),
                receiver_did: bob_did.clone(),
                hours: 1.0,
                service_description: "Auto-confirmed service".into(),
            },
        )
        .await;
    assert!(exchange.confirmed, "Should be auto-confirmed");

    // Cancel confirmed exchange should fail
    let result: Result<bool, _> = conductor
        .call_fallible(&zome_a, "cancel_expired_exchange", exchange.id.clone())
        .await;
    assert!(result.is_err(), "Should not cancel a confirmed exchange");
    println!("  - Cancel confirmed exchange: correctly rejected");

    // Create confirmation-required currency and record unconfirmed exchange
    let def2: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:cancel-test".into(),
                params: test_params_with_confirmation("PendCoin", "PC"),
                governance_proposal_id: None,
            },
        )
        .await;
    let _: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def2.id.clone(),
            },
        )
        .await;

    let unconfirmed: MintedExchange = conductor
        .call(
            &zome_a,
            "record_minted_exchange",
            RecordMintedExchangeInput {
                currency_id: def2.id.clone(),
                receiver_did: bob_did.clone(),
                hours: 2.0,
                service_description: "Pending service".into(),
            },
        )
        .await;
    assert!(!unconfirmed.confirmed, "Should be unconfirmed");

    // Cancel before timeout should fail (just created, timeout is 48h)
    let result2: Result<bool, _> = conductor
        .call_fallible(&zome_a, "cancel_expired_exchange", unconfirmed.id.clone())
        .await;
    assert!(result2.is_err(), "Should not cancel before timeout expires");
    println!("  - Cancel before timeout: correctly rejected");

    println!("Test 14.1 PASSED");
}

/// Test 14.2: Redundant lifecycle transitions are rejected
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_redundant_lifecycle_transitions() {
    println!("Test 14.2: Redundant Lifecycle Transitions");

    let (mut conductor, _agents, apps) = setup_finance_conductor(1).await;
    let cell = &apps[0].cells()[0];
    let zome = cell.zome("currency_mint");

    let def: CurrencyDefinition = conductor
        .call(
            &zome,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:guard-test".into(),
                params: test_params("GuardCoin", "GC"),
                governance_proposal_id: None,
            },
        )
        .await;

    // Suspend Draft should fail
    let r: Result<CurrencyDefinition, _> = conductor
        .call_fallible(&zome, "suspend_currency", def.id.clone())
        .await;
    assert!(r.is_err(), "Cannot suspend a Draft currency");
    println!("  - Suspend Draft: rejected");

    // Reactivate Draft should fail (not Suspended)
    let r: Result<CurrencyDefinition, _> = conductor
        .call_fallible(&zome, "reactivate_currency", def.id.clone())
        .await;
    assert!(r.is_err(), "Cannot reactivate a Draft currency");
    println!("  - Reactivate Draft: rejected");

    // Activate it
    let _: CurrencyDefinition = conductor
        .call(
            &zome,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def.id.clone(),
            },
        )
        .await;

    // Double-activate should fail (already Active)
    let r: Result<CurrencyDefinition, _> = conductor
        .call_fallible(
            &zome,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def.id.clone(),
            },
        )
        .await;
    assert!(r.is_err(), "Cannot activate an already Active currency");
    println!("  - Double-activate: rejected");

    // Reactivate Active should fail (not Suspended)
    let r: Result<CurrencyDefinition, _> = conductor
        .call_fallible(&zome, "reactivate_currency", def.id.clone())
        .await;
    assert!(r.is_err(), "Cannot reactivate an Active currency");
    println!("  - Reactivate Active: rejected");

    println!("Test 14.2 PASSED");
}
