// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Demurrage, compost accumulation, and redistribution tests.
//!
//! Coverage:
//!   apply_minted_demurrage → Scenario 16.1, E7, E23
//!   apply_demurrage_all    → Scenario 18.1, E9, E23
//!   get_demurrage_report   → Scenario 16.1, 18.1
//!   get_compost_balance    → Scenario 6.1, 16.1
//!   redistribute_compost   → Scenario 16.2, E8, E25, E29

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;

/// Consolidated demurrage test: 10 scenarios sharing a single conductor.
///
/// Runs block_on inside a thread with 16MB stack because the consolidated
/// async state machine (10 scenarios, 776 lines) exceeds default stack sizes.
#[test]
#[ignore]
fn test_demurrage_all() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(test_demurrage_all_inner());
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn test_demurrage_all_inner() {
    let (conductor, agents, apps) = setup_finance_conductor(2).await;

    let zome_a = apps[0].cells()[0].zome("currency_mint");
    let zome_b = apps[1].cells()[0].zome("currency_mint");
    let alice_did = format!("did:mycelix:{}", agents[0]);
    let bob_did = format!("did:mycelix:{}", agents[1]);

    // ── Scenario 6.1: Compost Zero-Sum ──────────────────────────────────
    {
        println!("Scenario 6.1: Compost Zero-Sum");

        let receiver_did = bob_did.clone();

        // Create currency with 2% demurrage
        let def: CurrencyDefinition = call_with_retry(
            &conductor,
            &zome_a,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:compost-test".into(),
                params: test_params("CompostCoin", "CC"),
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

        // Record exchange so provider has positive balance
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: receiver_did.clone(),
                    hours: 5.0,
                    service_description: "Compost test service".into(),
                },
            )
            .await;

        // Check compost balance exists (starts at 0 before any demurrage applied)
        let compost: CompostBalance = conductor
            .call(&zome_a, "get_compost_balance", def.id.clone())
            .await;
        println!(
            "  - Compost balance: {} (before demurrage)",
            compost.accumulated
        );

        // Verify stats net_sum is 0
        let stats: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;
        assert_eq!(stats.net_sum, 0, "Zero-sum invariant must hold");
        println!("  - Net sum: {} (zero-sum OK)", stats.net_sum);
        println!("Scenario 6.1 PASSED");
    }

    // ── Scenario 16.1: Demurrage & Compost Cycle ────────────────────────
    {
        println!("Scenario 16.1: Demurrage & Compost Cycle");

        // Create currency with 2% demurrage
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:demurrage-test".into(),
                    params: test_params("DemurCoin", "DM"),
                    governance_proposal_id: None,
                },
            )
            .await;
        assert_eq!(def.params.demurrage_rate, 0.02);

        let _: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        // Alice provides 5 hours to Bob (Alice gets +5, Bob gets -5)
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 5.0,
                    service_description: "Gardening".into(),
                },
            )
            .await;

        // Verify Alice has +5 before demurrage
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
        assert_eq!(alice_bal.balance, 5, "Alice should have +5");

        // Apply demurrage to Alice (positive balance = subject to demurrage)
        let dem_result: DemurrageResult = conductor
            .call(
                &zome_a,
                "apply_minted_demurrage",
                ApplyDemurrageInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                },
            )
            .await;
        assert_eq!(dem_result.previous_balance, 5);
        assert!(dem_result.deduction >= 0, "Deduction must be non-negative");
        assert_eq!(dem_result.demurrage_rate, 0.02);
        println!(
            "  - Demurrage applied: prev={}, deduction={}, new={}",
            dem_result.previous_balance, dem_result.deduction, dem_result.new_balance
        );

        // Apply demurrage to Bob (negative balance = exempt)
        let bob_dem: DemurrageResult = conductor
            .call(
                &zome_a,
                "apply_minted_demurrage",
                ApplyDemurrageInput {
                    currency_id: def.id.clone(),
                    member_did: bob_did.clone(),
                },
            )
            .await;
        assert_eq!(
            bob_dem.deduction, 0,
            "Negative balance exempt from demurrage"
        );
        println!("  - Bob (debtor) demurrage: deduction=0 (exempt)");

        // Check compost balance
        let compost: CompostBalance = conductor
            .call(&zome_a, "get_compost_balance", def.id.clone())
            .await;
        assert!(compost.accumulated >= 0, "Compost should be non-negative");
        println!("  - Compost balance: {}", compost.accumulated);

        // Get demurrage report
        let report: DemurrageReport = conductor
            .call(&zome_a, "get_demurrage_report", def.id.clone())
            .await;
        assert_eq!(report.demurrage_rate, 0.02);
        println!(
            "  - Report: rate={}, pending={}, affected={}",
            report.demurrage_rate, report.total_pending_demurrage, report.affected_members
        );

        println!("Scenario 16.1 PASSED");
    }

    // ── Scenario 16.2: Redistribute Compost ─────────────────────────────
    {
        println!("Scenario 16.2: Redistribute Compost");

        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:compost-redist".into(),
                    params: test_params("CompostCoin", "CP"),
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

        // Create some exchange history
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 4.0,
                    service_description: "Service".into(),
                },
            )
            .await;

        // Redistribute compost (may be 0 if no demurrage has accumulated yet)
        let result: RedistributeCompostResult = conductor
            .call(&zome_a, "redistribute_compost", def.id.clone())
            .await;
        assert!(
            result.total_redistributed >= 0,
            "Redistributed must be non-negative"
        );
        // result.recipients is u32 — always non-negative by type
        println!(
            "  - Redistributed: {} to {} recipients (per_member={})",
            result.total_redistributed, result.recipients, result.per_member_amount
        );

        println!("Scenario 16.2 PASSED");
    }

    // ── Scenario 18.1: Batch Demurrage + Report ─────────────────────────
    {
        println!("Scenario 18.1: Batch Demurrage + Report");

        // Create + activate a currency WITH demurrage
        let mut params = test_params("Decay Test", "DT");
        params.demurrage_rate = 0.02;
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:demurrage-batch".into(),
            params,
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition = conductor.call(&zome_a, "create_currency", input).await;
        let activate = ActivateCurrencyInput {
            currency_id: def.id.clone(),
        };
        let _: CurrencyDefinition = conductor.call(&zome_a, "activate_currency", activate).await;

        // Alice provides 3 hours to Bob → Alice +3, Bob -3
        let ex_input = RecordMintedExchangeInput {
            currency_id: def.id.clone(),
            receiver_did: bob_did.clone(),
            hours: 3.0,
            service_description: "Batch demurrage test exchange 1".into(),
        };
        let _: MintedExchange = conductor
            .call(&zome_a, "record_minted_exchange", ex_input)
            .await;

        // Bob provides 1 hour to Alice → Bob +1, Alice -1 → net: Alice +2, Bob -2
        let ex_input2 = RecordMintedExchangeInput {
            currency_id: def.id.clone(),
            receiver_did: format!("did:mycelix:{}", agents[0].clone()),
            hours: 1.0,
            service_description: "Batch demurrage test exchange 2".into(),
        };
        let _: MintedExchange = conductor
            .call(&zome_b, "record_minted_exchange", ex_input2)
            .await;
        println!("  - Two exchanges: Alice +2, Bob -2");

        // Get demurrage report (read-only, no mutation)
        let report: DemurrageReport = conductor
            .call(&zome_a, "get_demurrage_report", def.id.clone())
            .await;
        assert_eq!(report.demurrage_rate, 0.02);
        println!(
            "  - Report: rate={}, affected={}, total_pending={}",
            report.demurrage_rate, report.affected_members, report.total_pending_demurrage
        );

        // Apply batch demurrage
        let batch_results: Vec<DemurrageResult> = conductor
            .call(&zome_a, "apply_demurrage_all", def.id.clone())
            .await;
        println!(
            "  - Batch results: {} members affected",
            batch_results.len()
        );

        // Verify zero-sum still holds
        let stats: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;
        assert_eq!(stats.net_sum, 0, "Zero-sum preserved after batch demurrage");
        println!("  - Zero-sum invariant: net_sum={}", stats.net_sum);

        println!("Scenario 18.1 PASSED");
    }

    // ── Scenario E7: Demurrage Zero-Balance No-Op ───────────────────────
    {
        println!("Scenario E7: Demurrage Zero-Balance No-Op");

        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:zero-demurrage".into(),
                    params: test_params("ZeroDemCoin", "ZD"),
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

        // Apply demurrage to member with zero balance (never traded)
        let dem: DemurrageResult = conductor
            .call(
                &zome_a,
                "apply_minted_demurrage",
                ApplyDemurrageInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                },
            )
            .await;

        assert_eq!(dem.previous_balance, 0, "Should start at 0");
        assert_eq!(dem.deduction, 0, "Zero balance → zero deduction");
        assert_eq!(dem.new_balance, 0, "Balance unchanged");
        println!(
            "  - Zero-balance demurrage: prev={}, deduction={}, new={}",
            dem.previous_balance, dem.deduction, dem.new_balance
        );

        // Compost should also be 0
        let compost: CompostBalance = conductor
            .call(&zome_a, "get_compost_balance", def.id.clone())
            .await;
        assert_eq!(
            compost.accumulated, 0,
            "No compost from zero-balance demurrage"
        );
        println!("  - Compost: {}", compost.accumulated);

        println!("Scenario E7 PASSED");
    }

    // ── Scenario E8: Redistribute Compost Idempotent ────────────────────
    {
        println!("Scenario E8: Redistribute Compost Idempotent");

        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:redist-idem".into(),
                    params: test_params("RedistCoin", "RI"),
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

        // Create exchange history (2 members)
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 5.0,
                    service_description: "Redistribute test".into(),
                },
            )
            .await;

        // First redistribute
        let r1: RedistributeCompostResult = conductor
            .call(&zome_a, "redistribute_compost", def.id.clone())
            .await;
        println!(
            "  - First: redistributed={}, recipients={}, remainder={}",
            r1.total_redistributed, r1.recipients, r1.remainder_kept
        );

        // Verify zero-sum after first redistribute
        let stats1: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;
        assert_eq!(stats1.net_sum, 0, "Zero-sum after first redistribute");

        // Second redistribute (should be safe, likely 0 to distribute)
        let r2: RedistributeCompostResult = conductor
            .call(&zome_a, "redistribute_compost", def.id.clone())
            .await;
        println!(
            "  - Second: redistributed={}, recipients={}, remainder={}",
            r2.total_redistributed, r2.recipients, r2.remainder_kept
        );

        // Verify zero-sum still holds
        let stats2: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;
        assert_eq!(stats2.net_sum, 0, "Zero-sum after second redistribute");
        println!("  - Zero-sum preserved through double redistribute");

        println!("Scenario E8 PASSED");
    }

    // ── Scenario E9: Batch Demurrage Member Coverage ────────────────────
    {
        println!("Scenario E9: Batch Demurrage Member Coverage");

        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:batch-coverage".into(),
                    params: test_params("BatchCoin", "BA"),
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

        // Alice provides 3 hours to Bob → Alice +3, Bob -3
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 3.0,
                    service_description: "Batch coverage exchange 1".into(),
                },
            )
            .await;

        // Bob provides 5 hours to Alice → Bob +5, Alice -5 → net: Alice -2, Bob +2
        let _: MintedExchange = conductor
            .call(
                &zome_b,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: alice_did.clone(),
                    hours: 5.0,
                    service_description: "Batch coverage exchange 2".into(),
                },
            )
            .await;

        // Now: Alice=-2, Bob=+2
        // apply_demurrage_all should process Bob (positive) but not Alice (negative)
        let results: Vec<DemurrageResult> = conductor
            .call(&zome_a, "apply_demurrage_all", def.id.clone())
            .await;

        // Results should contain Bob (positive balance) but not Alice (negative)
        let bob_results: Vec<&DemurrageResult> =
            results.iter().filter(|r| r.member_did == bob_did).collect();
        let alice_results: Vec<&DemurrageResult> = results
            .iter()
            .filter(|r| r.member_did == alice_did)
            .collect();

        assert!(
            !bob_results.is_empty(),
            "Bob (positive balance) should be in batch results"
        );
        // Alice has negative balance — she should either not appear or have deduction=0
        for r in &alice_results {
            assert_eq!(
                r.deduction, 0,
                "Negative-balance member should have 0 deduction"
            );
        }
        println!(
            "  - Batch: {} total results, Bob entries={}, Alice entries={}",
            results.len(),
            bob_results.len(),
            alice_results.len()
        );

        // All deductions should be non-negative
        for r in &results {
            assert!(r.deduction >= 0, "Deduction must be non-negative");
        }

        // Second batch call should be safe
        let results2: Vec<DemurrageResult> = conductor
            .call(&zome_a, "apply_demurrage_all", def.id.clone())
            .await;
        println!(
            "  - Second batch: {} results (safe re-call)",
            results2.len()
        );

        // Zero-sum preserved
        let stats: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;
        assert_eq!(stats.net_sum, 0, "Zero-sum after double batch demurrage");
        println!("  - Zero-sum preserved");

        println!("Scenario E9 PASSED");
    }

    // ── Scenario E23: Demurrage on Non-Active Currency Rejected ─────────
    {
        println!("Scenario E23: Demurrage on Non-Active Currency Rejected");

        // Create, activate, do an exchange to create a positive balance
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:demurrage-guard".into(),
                    params: test_params("DemGuard", "DG"),
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
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 3.0,
                    service_description: "Service before suspension".into(),
                },
            )
            .await;
        println!("  - Exchange recorded, provider has positive balance");

        // Suspend the currency
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "suspend_currency", def.id.clone())
            .await;

        // apply_minted_demurrage on Suspended — should fail
        let result: Result<DemurrageResult, _> = conductor
            .call_fallible(
                &zome_a,
                "apply_minted_demurrage",
                ApplyDemurrageInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                },
            )
            .await;
        assert!(
            result.is_err(),
            "Demurrage should be rejected on Suspended currency"
        );
        println!("  - apply_minted_demurrage on Suspended: rejected");

        // apply_demurrage_all on Suspended — should also fail
        let result: Result<Vec<DemurrageResult>, _> = conductor
            .call_fallible(&zome_a, "apply_demurrage_all", def.id.clone())
            .await;
        assert!(
            result.is_err(),
            "Batch demurrage should be rejected on Suspended currency"
        );
        println!("  - apply_demurrage_all on Suspended: rejected");

        // Retire and try again
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "reactivate_currency", def.id.clone())
            .await;
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "retire_currency", def.id.clone())
            .await;

        let result: Result<DemurrageResult, _> = conductor
            .call_fallible(
                &zome_a,
                "apply_minted_demurrage",
                ApplyDemurrageInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                },
            )
            .await;
        assert!(
            result.is_err(),
            "Demurrage should be rejected on Retired currency"
        );
        println!("  - apply_minted_demurrage on Retired: rejected");

        println!("Scenario E23 PASSED");
    }

    // ── Scenario E25: Redistribute Compost on Non-Active Rejected ───────
    {
        println!("Scenario E25: Redistribute Compost on Non-Active Rejected");

        // Create, activate, exchange (to have members and potential compost)
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:redist-guard".into(),
                    params: test_params("RedistGuard", "RG"),
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
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 3.0,
                    service_description: "Service before suspension".into(),
                },
            )
            .await;

        // Suspend
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "suspend_currency", def.id.clone())
            .await;

        // redistribute_compost on Suspended — should fail
        let result: Result<RedistributeCompostResult, _> = conductor
            .call_fallible(&zome_a, "redistribute_compost", def.id.clone())
            .await;
        assert!(
            result.is_err(),
            "Redistribute should be rejected on Suspended currency"
        );
        println!("  - redistribute_compost on Suspended: rejected");

        // Reactivate → Retire → try again
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "reactivate_currency", def.id.clone())
            .await;
        let _: CurrencyDefinition = conductor
            .call(&zome_a, "retire_currency", def.id.clone())
            .await;

        let result: Result<RedistributeCompostResult, _> = conductor
            .call_fallible(&zome_a, "redistribute_compost", def.id.clone())
            .await;
        assert!(
            result.is_err(),
            "Redistribute should be rejected on Retired currency"
        );
        println!("  - redistribute_compost on Retired: rejected");

        println!("Scenario E25 PASSED");
    }

    // ── Scenario E29: Redistribute With Insufficient Compost ────────────
    {
        println!("Scenario E29: Redistribute With Insufficient Compost");

        // Create and activate currency
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:insuff-compost".into(),
                    params: test_params("InsuffCompost", "IC"),
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

        // Do an exchange to create members (but compost is 0 — no demurrage applied yet)
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 3.0,
                    service_description: "Service for compost test".into(),
                },
            )
            .await;

        // Redistribute with zero compost — should return gracefully (not error)
        let result: RedistributeCompostResult = conductor
            .call(&zome_a, "redistribute_compost", def.id.clone())
            .await;
        assert_eq!(result.total_redistributed, 0, "Nothing to redistribute");
        assert_eq!(result.per_member_amount, 0, "No per-member amount");
        println!(
            "  - Zero compost: redistributed={}, per_member={}, remainder={}",
            result.total_redistributed, result.per_member_amount, result.remainder_kept
        );

        println!("Scenario E29 PASSED");
    }
}
