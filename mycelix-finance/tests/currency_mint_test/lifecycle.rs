//! Lifecycle tests: Draft → Active → Suspended → Active → Retired
//!
//! Coverage:
//!   create_currency         → scenario 1.1
//!   activate_currency       → scenario 1.1, 1.2, 14.2
//!   suspend_currency        → scenario 1.2, 14.2
//!   reactivate_currency     → scenario 1.2, 14.2
//!   retire_currency         → scenario 1.2, 1.3
//!   get_currency            → scenario 1.1
//!   cancel_expired_exchange → scenario 14.1
//!   (lifecycle guards)      → scenario 14.2
//!
//! All 5 scenarios share a single SweetConductor (2 agents).

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;
use mycelix_finance_types::CurrencyStatus;

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_lifecycle_all() {
    let (conductor, agents, apps) = setup_finance_conductor(2).await;
    let cell_a = &apps[0].cells()[0];
    let zome_a = cell_a.zome("currency_mint");
    let bob_did = format!("did:mycelix:{}", agents[1]);

    // ── Scenario 1.1: Create currency in Draft, activate, verify ──────────
    println!("Scenario 1.1: Create and Activate Currency");
    {
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:test-hearth-1".into(),
            params: test_params("Garden Hours", "GH"),
            governance_proposal_id: None,
        };

        let def: CurrencyDefinition = conductor.call(&zome_a, "create_currency", input).await;
        assert_eq!(def.status, CurrencyStatus::Draft);
        assert_eq!(def.params.symbol, "GH");
        println!("  - Created draft: {} ({})", def.params.name, def.id);

        let activate = ActivateCurrencyInput {
            currency_id: def.id.clone(),
        };
        let active: CurrencyDefinition =
            conductor.call(&zome_a, "activate_currency", activate).await;
        assert_eq!(active.status, CurrencyStatus::Active);
        println!("  - Activated: {:?}", active.status);

        let fetched: Option<CurrencyDefinition> =
            conductor.call(&zome_a, "get_currency", def.id.clone()).await;
        assert!(fetched.is_some());
        println!("Scenario 1.1 PASSED");
    }

    // ── Scenario 1.2: Full lifecycle — Draft → Active → Suspended → Active → Retired ──
    println!("Scenario 1.2: Full Lifecycle");
    {
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:lifecycle-test".into(),
            params: test_params("Lifecycle Token", "LT"),
            governance_proposal_id: None,
        };

        let def: CurrencyDefinition = conductor.call(&zome_a, "create_currency", input).await;
        let cid = def.id.clone();
        assert_eq!(def.status, CurrencyStatus::Draft);

        let active: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: cid.clone(),
                },
            )
            .await;
        assert_eq!(active.status, CurrencyStatus::Active);

        let suspended: CurrencyDefinition =
            conductor.call(&zome_a, "suspend_currency", cid.clone()).await;
        assert_eq!(suspended.status, CurrencyStatus::Suspended);

        let reactivated: CurrencyDefinition = conductor
            .call(&zome_a, "reactivate_currency", cid.clone())
            .await;
        assert_eq!(reactivated.status, CurrencyStatus::Active);

        let retired: CurrencyDefinition =
            conductor.call(&zome_a, "retire_currency", cid.clone()).await;
        assert_eq!(retired.status, CurrencyStatus::Retired);

        println!("  - Draft → Active → Suspended → Active → Retired: all transitions verified");
        println!("Scenario 1.2 PASSED");
    }

    // ── Scenario 1.3: Draft cannot be retired directly ────────────────────
    println!("Scenario 1.3: Draft Cannot Retire");
    {
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:draft-retire-test".into(),
            params: test_params("No Shortcut", "NS"),
            governance_proposal_id: None,
        };

        let def: CurrencyDefinition = conductor.call(&zome_a, "create_currency", input).await;

        let result: Result<CurrencyDefinition, _> = conductor
            .call_fallible(&zome_a, "retire_currency", def.id.clone())
            .await;

        assert!(result.is_err(), "Draft → Retired should be rejected");
        println!("  - Draft → Retired correctly rejected");
        println!("Scenario 1.3 PASSED");
    }

    // ── Scenario 14.1: Cancel exchange guards ─────────────────────────────
    println!("Scenario 14.1: Cancel Exchange Guards");
    {
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

        println!("Scenario 14.1 PASSED");
    }

    // ── Scenario 14.2: Redundant lifecycle transitions are rejected ───────
    println!("Scenario 14.2: Redundant Lifecycle Transitions");
    {
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
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
            .call_fallible(&zome_a, "suspend_currency", def.id.clone())
            .await;
        assert!(r.is_err(), "Cannot suspend a Draft currency");
        println!("  - Suspend Draft: rejected");

        // Reactivate Draft should fail (not Suspended)
        let r: Result<CurrencyDefinition, _> = conductor
            .call_fallible(&zome_a, "reactivate_currency", def.id.clone())
            .await;
        assert!(r.is_err(), "Cannot reactivate a Draft currency");
        println!("  - Reactivate Draft: rejected");

        // Activate it
        let _: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        // Double-activate should fail (already Active)
        let r: Result<CurrencyDefinition, _> = conductor
            .call_fallible(
                &zome_a,
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
            .call_fallible(&zome_a, "reactivate_currency", def.id.clone())
            .await;
        assert!(r.is_err(), "Cannot reactivate an Active currency");
        println!("  - Reactivate Active: rejected");

        println!("Scenario 14.2 PASSED");
    }

    // ── Scenario 15: Update chain regression — amend then read returns latest ──
    println!("Scenario 15: Update Chain Regression (follow_update_chain)");
    {
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:update-chain-test".into(),
                    params: test_params("UpdateChain", "UC"),
                    governance_proposal_id: None,
                },
            )
            .await;

        // Activate
        let _: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        // Amend params: change credit_limit from 40 → 99 and description
        let mut new_params = test_params("UpdateChain", "UC");
        new_params.credit_limit = 99;
        new_params.description = "Amended to verify update chain".into();

        let amended: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "amend_currency_params",
                AmendCurrencyParamsInput {
                    currency_id: def.id.clone(),
                    new_params,
                    governance_proposal_id: None,
                },
            )
            .await;
        assert_eq!(amended.params.credit_limit, 99);

        // THE KEY ASSERTION: get_currency must return the AMENDED version,
        // not the original. This is the regression test for follow_update_chain.
        let fetched: Option<CurrencyDefinition> = conductor
            .call(&zome_a, "get_currency", def.id.clone())
            .await;
        let f = fetched.expect("Currency should exist");
        assert_eq!(
            f.params.credit_limit, 99,
            "get_currency must return latest version (credit_limit=99), not original (40)"
        );
        assert_eq!(
            f.params.description, "Amended to verify update chain",
            "get_currency must return latest description"
        );
        println!("  - get_currency returned amended params (credit_limit=99)");

        // Amend AGAIN to verify multi-hop update chain
        let mut params_v3 = test_params("UpdateChain", "UC");
        params_v3.credit_limit = 150;
        params_v3.description = "Second amendment".into();

        let _: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "amend_currency_params",
                AmendCurrencyParamsInput {
                    currency_id: def.id.clone(),
                    new_params: params_v3,
                    governance_proposal_id: None,
                },
            )
            .await;

        let fetched2: Option<CurrencyDefinition> = conductor
            .call(&zome_a, "get_currency", def.id.clone())
            .await;
        let f2 = fetched2.expect("Currency should exist");
        assert_eq!(
            f2.params.credit_limit, 150,
            "get_currency must follow multi-hop update chain to latest (credit_limit=150)"
        );
        println!("  - Multi-hop update chain verified (credit_limit=150)");

        println!("Scenario 15 PASSED");
    }

    println!("=== All lifecycle scenarios PASSED ===");
}
