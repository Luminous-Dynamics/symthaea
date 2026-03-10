//! Exchange recording, confirmation flow, rate limiting, and edge case tests.
//!
//! Coverage:
//!   record_minted_exchange    → test_record_exchange_zero_sum (2.1), test_rate_limit_enforcement (12.1),
//!                               test_self_exchange_rejected (E1), test_nan_hours_rejected (E2),
//!                               test_credit_limit_enforcement (E4), test_empty_description_rejected (E5),
//!                               test_min_service_minutes (E14), test_max_service_hours (E15),
//!                               test_nan_inf_hours_rejected (E16), test_zero_negative_hours_rejected (E17),
//!                               test_daily_rate_limit (E20)
//!   confirm_minted_exchange   → test_confirmation_flow (3.1), test_confirm_blocked_when_suspended (3.2),
//!                               test_two_party_confirmation_flow (13.1), test_double_confirm_idempotent (E13),
//!                               test_confirm_on_suspended_currency (E18), test_confirm_by_non_receiver_rejected (E24)
//!   cancel_expired_exchange   → test_cancel_expired_exchange (14.1), test_cancel_by_non_participant_rejected (E26),
//!                               test_cancel_no_timeout_rejected (E28)
//!   list_pending_exchanges    → test_two_party_confirmation_flow (13.1)
//!   list_pending_for_receiver → test_two_party_confirmation_flow (13.1)
//!   get_exchange              → test_get_exchange (15.1)
//!   get_currency_exchanges    → test_pagination_cursor (18.2)
//!   get_member_exchanges      → test_pagination_cursor (18.2)
//!   (suspended guard)         → test_exchange_blocked_when_suspended (2.2)

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;

/// Consolidated exchange tests — single conductor setup, 22 scenarios.
///
/// Runs block_on inside a thread with 16MB stack because the consolidated
/// async state machine (22 scenarios, 1633 lines) exceeds default stack sizes.
#[test]
#[ignore]
fn test_exchanges_all() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(test_exchanges_all_inner());
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn test_exchanges_all_inner() {
    let (conductor, agents, apps) = setup_finance_conductor(3).await;

    let zome_a = apps[0].cells()[0].zome("currency_mint");
    let zome_b = apps[1].cells()[0].zome("currency_mint");
    let zome_c = apps[2].cells()[0].zome("currency_mint");

    let alice_did = format!("did:mycelix:{}", agents[0]);
    let bob_did = format!("did:mycelix:{}", agents[1]);
    let _charlie_did = format!("did:mycelix:{}", agents[2]);

    // ── Scenario 2.1: Record exchange between two members, verify zero-sum balances ──
    {
        println!("Test 2.1: Exchange Recording & Zero-Sum");

        let receiver_did = bob_did.clone();

        // Create + activate currency
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:exchange-test".into(),
            params: test_params("Meal Credits", "MC"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition =
            call_with_retry(&conductor, &zome_a, "create_currency", input).await;
        let _active: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        // Record exchange: creator provides 2 hours to receiver
        let exchange_input = RecordMintedExchangeInput {
            currency_id: def.id.clone(),
            receiver_did: receiver_did.clone(),
            hours: 2.0,
            service_description: "Cooked dinner for the family".into(),
        };

        let exchange: MintedExchange = conductor
            .call(&zome_a, "record_minted_exchange", exchange_input)
            .await;
        assert_eq!(exchange.hours, 2.0);
        assert!(!exchange.confirmed || !def.params.requires_confirmation);
        println!("  - Exchange recorded: {} hours", exchange.hours);

        // Check provider balance (should be +2)
        let provider_did = alice_did.clone();
        let provider_bal: MintedBalanceInfo = conductor
            .call(
                &zome_a,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: provider_did.clone(),
                },
            )
            .await;
        assert_eq!(provider_bal.balance, 2, "Provider should gain +2");

        // Check receiver balance (should be -2)
        let receiver_bal: MintedBalanceInfo = conductor
            .call(
                &zome_b,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: receiver_did.clone(),
                },
            )
            .await;
        assert_eq!(receiver_bal.balance, -2, "Receiver should lose -2");

        // Zero-sum check
        assert_eq!(
            provider_bal.balance + receiver_bal.balance,
            0,
            "Balances must sum to zero"
        );
        println!(
            "  - Zero-sum verified: {} + {} = 0",
            provider_bal.balance, receiver_bal.balance
        );
        println!("Test 2.1 PASSED");
    }

    // ── Scenario 2.2: Cannot exchange in a Suspended currency ──
    {
        println!("Test 2.2: Suspended Currency Blocks Exchanges");

        let receiver_did = bob_did.clone();

        // Create + activate + suspend
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:suspend-test".into(),
            params: test_params("Paused Coin", "PC"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition = conductor.call(&zome_a, "create_currency", input).await;
        let _: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "suspend_currency", def.id.clone())
            .await;

        // Attempt exchange — should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did,
                    hours: 1.0,
                    service_description: "Should fail".into(),
                },
            )
            .await;

        assert!(
            result.is_err(),
            "Exchange in Suspended currency should fail"
        );
        println!("  - Exchange correctly rejected in Suspended currency");
        println!("Test 2.2 PASSED");
    }

    // ── Scenario 3.1: Two-party confirmation flow ──
    {
        println!("Test 3.1: Two-Party Confirmation Flow");

        let receiver_did = bob_did.clone();

        // Create currency with confirmation required
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:confirm-test".into(),
            params: test_params_with_confirmation("Confirm Coin", "CC"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition = conductor
            .call(&zome_a, "create_currency", input)
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

        // Record exchange — should be unconfirmed
        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: receiver_did.clone(),
                    hours: 1.5,
                    service_description: "Tutoring session".into(),
                },
            )
            .await;
        assert!(!exchange.confirmed, "Should start unconfirmed");
        println!("  - Exchange created unconfirmed: {}", exchange.id);

        // Receiver confirms
        let confirmed: MintedExchange = conductor
            .call(
                &zome_b,
                "confirm_minted_exchange",
                ConfirmMintedExchangeInput {
                    exchange_id: exchange.id.clone(),
                },
            )
            .await;
        assert!(confirmed.confirmed, "Should now be confirmed");
        println!("  - Exchange confirmed by receiver");

        // Check balances — should reflect the exchange now
        let provider_did = alice_did.clone();
        let provider_bal: MintedBalanceInfo = conductor
            .call(
                &zome_a,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: provider_did,
                },
            )
            .await;
        // Balance is rounded hours as i32
        assert_eq!(
            provider_bal.balance, 2,
            "Provider should gain rounded hours"
        );

        let receiver_bal: MintedBalanceInfo = conductor
            .call(
                &zome_b,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: receiver_did,
                },
            )
            .await;
        assert_eq!(
            receiver_bal.balance, -2,
            "Receiver should lose rounded hours"
        );

        println!(
            "  - Balances updated after confirmation: +{} / {}",
            provider_bal.balance, receiver_bal.balance
        );
        println!("Test 3.1 PASSED");
    }

    // ── Scenario 3.2: Confirm blocked when currency is Suspended ──
    {
        println!("Test 3.2: Confirm Blocked When Suspended");

        let receiver_did = bob_did.clone();

        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:suspend-confirm-test".into(),
            params: test_params_with_confirmation("Suspendable", "SU"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition = conductor
            .call(&zome_a, "create_currency", input)
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

        // Record exchange while Active
        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did,
                    hours: 1.0,
                    service_description: "Pre-suspend exchange".into(),
                },
            )
            .await;
        assert!(!exchange.confirmed);

        // Suspend currency
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "suspend_currency", def.id.clone())
            .await;

        // Attempt to confirm — should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_b,
                "confirm_minted_exchange",
                ConfirmMintedExchangeInput {
                    exchange_id: exchange.id.clone(),
                },
            )
            .await;

        assert!(
            result.is_err(),
            "Confirmation should be blocked for Suspended currency"
        );
        println!("  - Confirm correctly blocked for Suspended currency");
        println!("Test 3.2 PASSED");
    }

    // ── Scenario 12.1: max_exchanges_per_day enforcement ──
    {
        println!("Test 12.1: Rate Limit Enforcement");

        // Create currency with rate limit of 3 exchanges per day
        let mut params = test_params("RateLimitCoin", "RL");
        params.max_exchanges_per_day = 3;
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:ratelimit".into(),
                    params,
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

        // Record 3 exchanges (all should succeed)
        for i in 1..=3 {
            let ex: MintedExchange = conductor
                .call(
                    &zome_a,
                    "record_minted_exchange",
                    RecordMintedExchangeInput {
                        currency_id: def.id.clone(),
                        receiver_did: bob_did.clone(),
                        hours: 1.0,
                        service_description: format!("Service #{}", i),
                    },
                )
                .await;
            assert!(ex.confirmed, "Exchange #{} should succeed", i);
            println!("  - Exchange #{} succeeded", i);
        }

        // 4th exchange should fail (rate limit exceeded)
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 1.0,
                    service_description: "Service #4 (should fail)".into(),
                },
            )
            .await;

        assert!(
            result.is_err(),
            "4th exchange should be rejected by rate limit"
        );
        println!("  - Exchange #4 correctly rejected (rate limit 3/day)");

        println!("Test 12.1 PASSED");
    }

    // ── Scenario 13.1: Full two-party confirmation flow with pending list queries ──
    {
        println!("Test 13.1: Two-Party Confirmation Flow");

        // Create currency with confirmation required
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:confirm-test-13".into(),
                    params: test_params_with_confirmation("ConfirmCoin13", "C3"),
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

        // Alice records exchange (Bob is receiver) — should be unconfirmed
        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 3.0,
                    service_description: "Teaching math".into(),
                },
            )
            .await;
        assert!(
            !exchange.confirmed,
            "Exchange should be unconfirmed (requires_confirmation=true)"
        );
        println!("  - Exchange recorded (unconfirmed): {}", exchange.id);

        // Balances should be unchanged (no balance update until confirmed)
        let alice_bal: MintedBalanceInfo = conductor
            .call(
                &zome_a,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                },
            )
            .await;
        assert_eq!(
            alice_bal.balance, 0,
            "Alice balance unchanged before confirm"
        );

        // Pending exchanges should show this exchange
        let pending: Vec<MintedExchange> = conductor
            .call(&zome_a, "list_pending_exchanges", def.id.clone())
            .await;
        assert!(
            pending.iter().any(|e| e.id == exchange.id),
            "Exchange should appear in pending list"
        );
        println!("  - Pending list shows {} exchange(s)", pending.len());

        // Bob's pending list should show this exchange
        let bob_pending: Vec<MintedExchange> = conductor
            .call(&zome_b, "list_pending_for_receiver", bob_did.clone())
            .await;
        assert!(
            bob_pending.iter().any(|e| e.id == exchange.id),
            "Exchange should appear in Bob's pending list"
        );
        println!(
            "  - Bob's pending list shows {} exchange(s)",
            bob_pending.len()
        );

        // Bob confirms the exchange
        let confirmed: MintedExchange = conductor
            .call(&zome_b, "confirm_minted_exchange", exchange.id.clone())
            .await;
        assert!(confirmed.confirmed, "Exchange should now be confirmed");
        println!("  - Exchange confirmed by Bob");

        // Balances should now reflect the exchange
        let alice_bal2: MintedBalanceInfo = conductor
            .call(
                &zome_a,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                },
            )
            .await;
        assert_eq!(alice_bal2.balance, 3, "Alice should have +3 (provider)");

        let bob_bal: MintedBalanceInfo = conductor
            .call(
                &zome_b,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: bob_did.clone(),
                },
            )
            .await;
        assert_eq!(bob_bal.balance, -3, "Bob should have -3 (receiver)");
        println!(
            "  - Balances: Alice={}, Bob={}",
            alice_bal2.balance, bob_bal.balance
        );

        // Verify confirmed exchange no longer appears in Bob's pending list
        let bob_pending_after: Vec<MintedExchange> = conductor
            .call(&zome_b, "list_pending_for_receiver", bob_did.clone())
            .await;
        assert!(
            !bob_pending_after.iter().any(|e| e.id == exchange.id),
            "Confirmed exchange must not appear in receiver's pending list"
        );
        println!("  - Confirmed exchange removed from pending index");

        println!("Test 13.1 PASSED");
    }

    // ── Scenario 15.1: get_exchange returns exchange by ID, None for non-existent ──
    {
        println!("Test 15.1: Get Exchange");

        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:getex-test".into(),
                    params: test_params("GetExCoin", "GE"),
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

        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 1.5,
                    service_description: "Test exchange".into(),
                },
            )
            .await;

        // get_exchange should find it
        let fetched: Option<MintedExchange> = conductor
            .call(&zome_a, "get_exchange", exchange.id.clone())
            .await;
        assert!(fetched.is_some(), "Should find exchange by ID");
        let f = fetched.unwrap();
        assert_eq!(f.id, exchange.id);
        assert_eq!(f.hours, 1.5);
        println!("  - get_exchange found: {} ({} hours)", f.id, f.hours);

        // Non-existent should return None
        let missing: Option<MintedExchange> = conductor
            .call(&zome_a, "get_exchange", "nonexistent-id".to_string())
            .await;
        assert!(
            missing.is_none(),
            "Non-existent exchange should return None"
        );
        println!("  - get_exchange for non-existent: None");

        println!("Test 15.1 PASSED");
    }

    // ── Scenario 18.2: Pagination — get_currency_exchanges and get_member_exchanges with cursor ──
    {
        println!("Test 18.2: Pagination Cursor");

        // Create + activate
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:pagination-test".into(),
            params: test_params("Page Test", "PT"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition = conductor.call(&zome_a, "create_currency", input).await;
        let _: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        // Record 3 exchanges with small delays for distinct timestamps
        let mut exchange_timestamps = Vec::new();
        for i in 1..=3 {
            let ex: MintedExchange = conductor
                .call(
                    &zome_a,
                    "record_minted_exchange",
                    RecordMintedExchangeInput {
                        currency_id: def.id.clone(),
                        receiver_did: bob_did.clone(),
                        hours: 1.0,
                        service_description: format!("Pagination exchange {}", i),
                    },
                )
                .await;
            exchange_timestamps.push(ex.timestamp);
        }
        println!("  - Recorded 3 exchanges");

        // Get all (no cursor)
        let all: Vec<MintedExchange> = conductor
            .call(
                &zome_a,
                "get_currency_exchanges",
                PaginatedCurrencyInput {
                    currency_id: def.id.clone(),
                    limit: None,
                    after_timestamp: None,
                },
            )
            .await;
        assert_eq!(all.len(), 3, "All 3 exchanges returned without cursor");

        // Get with cursor after first exchange → should skip the first
        let after_first: Vec<MintedExchange> = conductor
            .call(
                &zome_a,
                "get_currency_exchanges",
                PaginatedCurrencyInput {
                    currency_id: def.id.clone(),
                    limit: None,
                    after_timestamp: Some(exchange_timestamps[0]),
                },
            )
            .await;
        assert!(
            after_first.len() <= 2,
            "Cursor should filter out first exchange: got {}",
            after_first.len()
        );
        println!(
            "  - Cursor after first: {} exchanges (expected ≤2)",
            after_first.len()
        );

        // Get with limit=1
        let limited: Vec<MintedExchange> = conductor
            .call(
                &zome_a,
                "get_currency_exchanges",
                PaginatedCurrencyInput {
                    currency_id: def.id.clone(),
                    limit: Some(1),
                    after_timestamp: None,
                },
            )
            .await;
        assert_eq!(limited.len(), 1, "Limit=1 returns exactly 1");

        // Test get_member_exchanges with cursor
        let alice_exchanges: Vec<MintedExchange> = conductor
            .call(
                &zome_a,
                "get_member_exchanges",
                GetMemberExchangesInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                    limit: None,
                    after_timestamp: None,
                },
            )
            .await;
        assert_eq!(
            alice_exchanges.len(),
            3,
            "Alice is provider of all 3 exchanges"
        );
        println!("  - Alice member exchanges: {}", alice_exchanges.len());

        println!("Test 18.2 PASSED");
    }

    // ── Scenario E1: Self-exchange is rejected ──
    {
        println!("Test E1: Self-Exchange Rejected");

        let self_did = alice_did.clone();

        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:self-exchange-test".into(),
                    params: test_params("SelfCoin", "SF"),
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

        // Attempt self-exchange — should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: self_did,
                    hours: 1.0,
                    service_description: "Self-exchange attempt".into(),
                },
            )
            .await;

        assert!(result.is_err(), "Self-exchange should be rejected");
        println!("  - Self-exchange correctly rejected");
        println!("Test E1 PASSED");
    }

    // ── Scenario E2: NaN hours rejected ──
    {
        println!("Test E2: NaN Hours Rejected");

        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:nan-test".into(),
                    params: test_params("NaNcoin", "NN"),
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

        // Attempt NaN hours — should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: f32::NAN,
                    service_description: "NaN hours attempt".into(),
                },
            )
            .await;

        assert!(result.is_err(), "NaN hours should be rejected");
        println!("  - NaN hours correctly rejected");

        // Attempt infinity hours — should also fail
        let result2: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: f32::INFINITY,
                    service_description: "Infinity hours attempt".into(),
                },
            )
            .await;

        assert!(result2.is_err(), "Infinity hours should be rejected");
        println!("  - Infinity hours correctly rejected");

        println!("Test E2 PASSED");
    }

    // ── Scenario E4: Credit limit enforcement — provider ceiling and receiver floor ──
    {
        println!("Test E4: Credit Limit Enforcement");

        // Create currency with credit_limit=40
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:credit-limit-test".into(),
                    params: test_params("LimitCoin", "LC"),
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

        // Record 5 exchanges of 8 hours each = 40 total (at the limit)
        for i in 1..=5 {
            let _: MintedExchange = conductor
                .call(
                    &zome_a,
                    "record_minted_exchange",
                    RecordMintedExchangeInput {
                        currency_id: def.id.clone(),
                        receiver_did: bob_did.clone(),
                        hours: 8.0,
                        service_description: format!("Max service #{}", i),
                    },
                )
                .await;
        }
        println!("  - Provider at +40 (credit limit)");

        // Next exchange should fail — provider would exceed +40
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 1.0,
                    service_description: "Over the limit".into(),
                },
            )
            .await;

        assert!(
            result.is_err(),
            "Provider exceeding credit limit should be rejected"
        );
        println!("  - Provider ceiling enforced: +41 rejected");

        // Verify receiver is at -40 (debt limit)
        let receiver_bal: MintedBalanceInfo = conductor
            .call(
                &zome_a,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: bob_did.clone(),
                },
            )
            .await;
        assert_eq!(receiver_bal.balance, -40, "Receiver should be at -40");
        println!("  - Receiver at debt floor: {}", receiver_bal.balance);

        println!("Test E4 PASSED");
    }

    // ── Scenario E5: Empty service description is rejected ──
    {
        println!("Test E5: Empty Description Rejected");

        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:empty-desc-test".into(),
                    params: test_params("DescCoin", "DS"),
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

        // Attempt exchange with empty description — should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 1.0,
                    service_description: "".into(),
                },
            )
            .await;

        assert!(
            result.is_err(),
            "Empty service description should be rejected"
        );
        println!("  - Empty description correctly rejected");

        println!("Test E5 PASSED");
    }

    // ── Scenario E13: Double-confirm is idempotent — no double-credit ──
    {
        println!("Test E13: Double-Confirm Idempotent");

        // Create confirmation-required currency
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:double-confirm".into(),
                    params: test_params_with_confirmation("DoubleConf", "DC"),
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

        // Alice records exchange → unconfirmed
        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 2.0,
                    service_description: "Double confirm test".into(),
                },
            )
            .await;
        assert!(!exchange.confirmed);

        // Bob confirms first time
        let confirmed: MintedExchange = conductor
            .call(&zome_b, "confirm_minted_exchange", exchange.id.clone())
            .await;
        assert!(confirmed.confirmed);
        println!("  - First confirm: success");

        // Check balances after first confirm
        let alice_bal: MintedBalanceInfo = conductor
            .call(
                &zome_a,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                },
            )
            .await;
        assert_eq!(alice_bal.balance, 2, "Alice +2 after first confirm");

        // Bob confirms second time — should be idempotent (no-op)
        let confirmed2: MintedExchange = conductor
            .call(&zome_b, "confirm_minted_exchange", exchange.id.clone())
            .await;
        assert!(confirmed2.confirmed);
        println!("  - Second confirm: no-op (idempotent)");

        // Balances should be unchanged — no double-credit
        let alice_bal2: MintedBalanceInfo = conductor
            .call(
                &zome_a,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                },
            )
            .await;
        assert_eq!(
            alice_bal2.balance, 2,
            "Alice should still be +2 (no double-credit)"
        );

        let bob_bal: MintedBalanceInfo = conductor
            .call(
                &zome_b,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: bob_did.clone(),
                },
            )
            .await;
        assert_eq!(
            bob_bal.balance, -2,
            "Bob should still be -2 (no double-debit)"
        );
        println!(
            "  - Balances unchanged: Alice={}, Bob={}",
            alice_bal2.balance, bob_bal.balance
        );

        // Zero-sum check
        let stats: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;
        assert_eq!(stats.net_sum, 0, "Zero-sum preserved after double confirm");
        println!("  - Zero-sum: net_sum={}", stats.net_sum);

        println!("Test E13 PASSED");
    }

    // ── Scenario E14: min_service_minutes boundary enforcement ──
    {
        println!("Test E14: Min Service Minutes");

        // Default min_service_minutes = 15 (0.25 hours)
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:min-minutes".into(),
                    params: test_params("MinCoin", "MN"),
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

        // 0.2 hours = 12 minutes → below 15-minute minimum → should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 0.2,
                    service_description: "Too short".into(),
                },
            )
            .await;
        assert!(
            result.is_err(),
            "12 minutes should be rejected (min is 15)"
        );
        println!("  - 0.2h (12 min) rejected");

        // 0.25 hours = 15 minutes → exactly at minimum → should succeed
        let ok: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 0.25,
                    service_description: "Exactly at minimum".into(),
                },
            )
            .await;
        assert_eq!(ok.hours, 0.25);
        println!("  - 0.25h (15 min) accepted");

        println!("Test E14 PASSED");
    }

    // ── Scenario E15: max_service_hours boundary enforcement ──
    {
        println!("Test E15: Max Service Hours");

        // Default max_service_hours = 8
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:max-hours".into(),
                    params: test_params("MaxCoin", "MX"),
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

        // 8.0 hours → at maximum → should succeed
        let ok: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 8.0,
                    service_description: "Full day service".into(),
                },
            )
            .await;
        assert_eq!(ok.hours, 8.0);
        println!("  - 8.0h accepted (at max)");

        // 9.0 hours → exceeds maximum → should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 9.0,
                    service_description: "Exceeds max".into(),
                },
            )
            .await;
        assert!(result.is_err(), "9 hours should be rejected (max is 8)");
        println!("  - 9.0h rejected (exceeds max)");

        println!("Test E15 PASSED");
    }

    // ── Scenario E16: NaN and Infinity hours are rejected by is_finite() guard ──
    {
        println!("Test E16: NaN/Inf Hours Rejected");

        // Create and activate currency
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:nan-test-e16".into(),
                    params: test_params("NanTest", "NT"),
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

        // NaN hours — should be rejected
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: f32::NAN,
                    service_description: "NaN hours".into(),
                },
            )
            .await;
        assert!(result.is_err(), "NaN hours should be rejected");
        println!("  - f32::NAN rejected");

        // Positive infinity — should be rejected
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: f32::INFINITY,
                    service_description: "Inf hours".into(),
                },
            )
            .await;
        assert!(result.is_err(), "Positive infinity hours should be rejected");
        println!("  - f32::INFINITY rejected");

        // Negative infinity — should be rejected
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: f32::NEG_INFINITY,
                    service_description: "Neg inf hours".into(),
                },
            )
            .await;
        assert!(
            result.is_err(),
            "Negative infinity hours should be rejected"
        );
        println!("  - f32::NEG_INFINITY rejected");

        println!("Test E16 PASSED");
    }

    // ── Scenario E17: Zero and negative hours are rejected ──
    {
        println!("Test E17: Zero/Negative Hours Rejected");

        // Create and activate currency
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:zero-test".into(),
                    params: test_params("ZeroTest", "ZT"),
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

        // Zero hours — should be rejected
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 0.0,
                    service_description: "Zero hours".into(),
                },
            )
            .await;
        assert!(result.is_err(), "Zero hours should be rejected");
        println!("  - 0.0 hours rejected");

        // Negative hours — should be rejected
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: -1.0,
                    service_description: "Negative hours".into(),
                },
            )
            .await;
        assert!(result.is_err(), "Negative hours should be rejected");
        println!("  - -1.0 hours rejected");

        println!("Test E17 PASSED");
    }

    // ── Scenario E18: Confirm exchange blocked when currency is suspended mid-flow ──
    {
        println!("Test E18: Confirm Blocked on Suspended Currency");

        // Create currency with requires_confirmation
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:susp-confirm".into(),
                    params: test_params_with_confirmation("SuspConfirm", "SC"),
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

        // Record exchange (unconfirmed due to requires_confirmation)
        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 2.0,
                    service_description: "Service before suspension".into(),
                },
            )
            .await;
        assert!(!exchange.confirmed, "Should be unconfirmed");
        println!("  - Unconfirmed exchange recorded");

        // Suspend the currency mid-flow
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "suspend_currency", def.id.clone())
            .await;
        println!("  - Currency suspended");

        // Bob tries to confirm — should fail because currency is Suspended
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(&zome_b, "confirm_minted_exchange", exchange.id.clone())
            .await;
        assert!(
            result.is_err(),
            "Confirm should fail when currency is Suspended"
        );
        println!("  - Confirm on suspended currency correctly rejected");

        // Verify balance unchanged (no credits applied)
        let bal: MintedBalanceInfo = conductor
            .call(
                &zome_a,
                "get_minted_balance",
                GetMintedBalanceInput {
                    currency_id: def.id.clone(),
                    member_did: bob_did.clone(),
                },
            )
            .await;
        assert_eq!(bal.balance, 0, "Balance should be unchanged");
        println!("  - Receiver balance still 0 (no phantom credits)");

        println!("Test E18 PASSED");
    }

    // ── Scenario E20: Daily rate limit enforcement (max_exchanges_per_day) ──
    {
        println!("Test E20: Daily Rate Limit Enforcement");

        // Create currency with max_exchanges_per_day = 2
        let mut params = test_params("RateLimit2", "R2");
        params.max_exchanges_per_day = 2;
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:rate-limit".into(),
                    params,
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

        // Exchange 1 — should succeed
        let ex1: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 1.0,
                    service_description: "First exchange".into(),
                },
            )
            .await;
        assert!(ex1.confirmed);
        println!("  - Exchange 1/2 accepted");

        // Exchange 2 — should succeed
        let ex2: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 1.0,
                    service_description: "Second exchange".into(),
                },
            )
            .await;
        assert!(ex2.confirmed);
        println!("  - Exchange 2/2 accepted");

        // Exchange 3 — should be rejected (daily limit reached)
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 1.0,
                    service_description: "Third exchange".into(),
                },
            )
            .await;
        assert!(
            result.is_err(),
            "3rd exchange should be rejected (daily limit = 2)"
        );
        println!("  - Exchange 3 rejected (daily limit reached)");

        println!("Test E20 PASSED");
    }

    // ── Scenario E24: Only the receiver can confirm — provider attempting confirm is rejected ──
    {
        println!("Test E24: Confirm by Non-Receiver Rejected");

        // Create currency with requires_confirmation
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:auth-confirm".into(),
                    params: test_params_with_confirmation("AuthConfirm", "AC"),
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

        // Alice records exchange to Bob (unconfirmed)
        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 2.0,
                    service_description: "Service needing confirmation".into(),
                },
            )
            .await;
        assert!(!exchange.confirmed);
        println!("  - Unconfirmed exchange recorded");

        // Alice (provider) tries to confirm — should be rejected
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(&zome_a, "confirm_minted_exchange", exchange.id.clone())
            .await;
        assert!(
            result.is_err(),
            "Provider should not be able to confirm their own exchange"
        );
        println!("  - Provider confirm correctly rejected");

        println!("Test E24 PASSED");
    }

    // ── Scenario E26: Only exchange participants can cancel — third party rejected ──
    {
        println!("Test E26: Cancel by Non-Participant Rejected");

        // Create currency with confirmation + timeout
        let mut params = test_params_with_confirmation("CancelAuth", "CA");
        params.confirmation_timeout_hours = 1;
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:cancel-auth".into(),
                    params,
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

        // Alice records exchange to Bob (unconfirmed)
        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 2.0,
                    service_description: "Service for cancel test".into(),
                },
            )
            .await;
        assert!(!exchange.confirmed);
        println!("  - Unconfirmed exchange recorded (Alice→Bob)");

        // Charlie (non-participant) tries to cancel — should be rejected
        let result: Result<bool, _> = conductor
            .call_fallible(&zome_c, "cancel_expired_exchange", exchange.id.clone())
            .await;
        assert!(
            result.is_err(),
            "Non-participant should not be able to cancel"
        );
        println!("  - Charlie (non-participant) cancel correctly rejected");

        println!("Test E26 PASSED");
    }

    // ── Scenario E28: Cancel on currency with no timeout (timeout_hours=0) is rejected ──
    {
        println!("Test E28: Cancel With No Timeout Rejected");

        // Create currency: requires_confirmation=true but timeout=0 (never expires)
        let mut params = test_params("NoTimeout", "NO");
        params.requires_confirmation = true;
        params.confirmation_timeout_hours = 0;
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:no-timeout".into(),
                    params,
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

        // Record unconfirmed exchange
        let exchange: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 2.0,
                    service_description: "Service with no timeout".into(),
                },
            )
            .await;
        assert!(!exchange.confirmed);
        println!("  - Unconfirmed exchange recorded (timeout=0)");

        // Try to cancel — should fail because timeout=0 means "never expires"
        let result: Result<bool, _> = conductor
            .call_fallible(&zome_a, "cancel_expired_exchange", exchange.id.clone())
            .await;
        assert!(
            result.is_err(),
            "Cancel should fail when timeout is 0 (exchanges never expire)"
        );
        println!("  - Cancel with no timeout correctly rejected");

        println!("Test E28 PASSED");
    }
}
