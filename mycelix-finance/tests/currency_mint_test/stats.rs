//! Currency stats and zero-sum proof tests.
//!
//! Coverage:
//!   get_currency_stats → test_currency_stats (4.1), test_currency_stats_zero_sum (17.1)

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;
use holochain::sweettest::*;
use mycelix_finance_types::CurrencyStatus;

/// Test 4.1: Currency stats reflect exchange activity
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_currency_stats() {
    println!("Test 4.1: Currency Stats");

    let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path)
        .await
        .expect("Load DNA");
    let mut conductor = SweetConductor::from_standard_config().await;

    let agents = SweetAgents::get(conductor.keystore(), 2).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-finance", &agents, &[dna])
        .await
        .expect("Install app");

    let cell = &apps[0].cells()[0];
    let zome = cell.zome("currency_mint");
    let receiver_did = format!("did:mycelix:{}", agents[1]);

    // Create + activate
    let input = CreateCurrencyInput {
        dao_did: "did:mycelix:dao:stats-test".into(),
        params: test_params("Stats Coin", "ST"),
        governance_proposal_id: None,
    };
    let def: CurrencyDefinition = conductor.call(&zome, "create_currency", input).await;
    let _: CurrencyDefinition = conductor
        .call(
            &zome,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def.id.clone(),
            },
        )
        .await;

    // Record a few exchanges
    for i in 1..=3 {
        let _: MintedExchange = conductor
            .call(
                &zome,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: receiver_did.clone(),
                    hours: i as f32,
                    service_description: format!("Service #{}", i),
                },
            )
            .await;
    }

    // Get stats
    let stats: CurrencyStats = conductor
        .call(&zome, "get_currency_stats", def.id.clone())
        .await;
    assert_eq!(stats.total_exchanges, 3);
    assert_eq!(stats.member_count, 2);
    assert_eq!(stats.net_sum, 0, "Zero-sum: credits + debts = 0");
    println!(
        "  - Stats: {} exchanges, {} members, net_sum={}",
        stats.total_exchanges, stats.member_count, stats.net_sum
    );
    println!("Test 4.1 PASSED");
}

/// Test 17.1: get_currency_stats verifies zero-sum invariant end-to-end
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_currency_stats_zero_sum() {
    println!("Test 17.1: Currency Stats — Zero-Sum Proof");

    let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path)
        .await
        .expect("Load DNA");
    let mut conductor = SweetConductor::from_standard_config().await;
    let agents = SweetAgents::get(conductor.keystore(), 2).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-finance", &agents, &[dna])
        .await
        .expect("Install app");

    let cell_a = &apps[0].cells()[0];
    let zome_a = cell_a.zome("currency_mint");
    let bob_did = format!("did:mycelix:{}", agents[1]);

    let def: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:stats-test".into(),
                params: test_params("StatsCoin", "ST"),
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

    // Record multiple exchanges
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

    // Get stats
    let stats: CurrencyStats = conductor
        .call(&zome_a, "get_currency_stats", def.id.clone())
        .await;

    assert_eq!(stats.member_count, 2, "Two members (Alice and Bob)");
    assert_eq!(stats.total_exchanges, 3, "Three exchanges recorded");
    assert_eq!(stats.confirmed_exchanges, 3, "All auto-confirmed");
    assert_eq!(stats.pending_exchanges, 0, "No pending");
    assert!(stats.total_credit > 0, "Alice has positive credit");
    assert!(stats.total_debt < 0, "Bob has negative debt");

    // THE ZERO-SUM PROOF: net_sum must be 0
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

    println!("Test 17.1 PASSED");
}
