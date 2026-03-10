//! Currency stats and zero-sum proof tests.
//!
//! Coverage:
//!   get_currency_stats → scenario 4.1, 17.1
//!
//! Both scenarios share a single SweetConductor (2 agents).

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;

#[test]
#[ignore]
fn test_stats_all() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(test_stats_all_inner());
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn test_stats_all_inner() {
    let (conductor, agents, apps) = setup_finance_conductor(2).await;
    let cell_a = &apps[0].cells()[0];
    let zome_a = cell_a.zome("currency_mint");
    let bob_did = format!("did:mycelix:{}", agents[1]);

    // ── Scenario 4.1: Currency stats reflect exchange activity ─────────
    println!("Scenario 4.1: Currency Stats");
    {
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:stats-test-41".into(),
            params: test_params("Stats Coin", "ST"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition =
            call_with_retry(&conductor, &zome_a, "create_currency", input).await;
        let _: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        for i in 1..=3 {
            let _: MintedExchange = conductor
                .call(
                    &zome_a,
                    "record_minted_exchange",
                    RecordMintedExchangeInput {
                        currency_id: def.id.clone(),
                        receiver_did: bob_did.clone(),
                        hours: i as f32,
                        service_description: format!("Service #{}", i),
                    },
                )
                .await;
        }

        let stats: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;
        assert_eq!(stats.total_exchanges, 3);
        assert_eq!(stats.member_count, 2);
        assert_eq!(stats.net_sum, 0, "Zero-sum: credits + debts = 0");
        println!(
            "  - Stats: {} exchanges, {} members, net_sum={}",
            stats.total_exchanges, stats.member_count, stats.net_sum
        );
        println!("Scenario 4.1 PASSED");
    }

    // ── Scenario 17.1: Zero-sum proof end-to-end ──────────────────────
    println!("Scenario 17.1: Currency Stats — Zero-Sum Proof");
    {
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:stats-test-171".into(),
                    params: test_params("StatsCoin", "S2"),
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

        for i in 1..=3 {
            let _: MintedExchange = conductor
                .call(
                    &zome_a,
                    "record_minted_exchange",
                    RecordMintedExchangeInput {
                        currency_id: def.id.clone(),
                        receiver_did: bob_did.clone(),
                        hours: i as f32,
                        service_description: format!("Service #{}", i),
                    },
                )
                .await;
        }

        let stats: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;

        assert_eq!(stats.member_count, 2, "Two members (Alice and Bob)");
        assert_eq!(stats.total_exchanges, 3, "Three exchanges recorded");
        assert_eq!(stats.confirmed_exchanges, 3, "All auto-confirmed");
        assert_eq!(stats.pending_exchanges, 0, "No pending");
        assert!(stats.total_credit > 0, "Alice has positive credit");
        assert!(stats.total_debt < 0, "Bob has negative debt");

        assert_eq!(
            stats.net_sum, 0,
            "Zero-sum invariant: credit + debt + compost = 0"
        );
        println!(
            "  - Stats: members={}, exchanges={}, credit={}, debt={}, compost={}, net={}",
            stats.member_count,
            stats.total_exchanges,
            stats.total_credit,
            stats.total_debt,
            stats.compost_balance,
            stats.net_sum
        );
        println!("  - ZERO-SUM INVARIANT VERIFIED");
        println!("Scenario 17.1 PASSED");
    }

    println!("=== All stats scenarios PASSED ===");
}
