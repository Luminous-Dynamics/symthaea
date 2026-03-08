//! Discovery tests: list_active_currencies, search_currencies, suspend/reactivate index.
//!
//! Coverage:
//!   list_active_currencies → test_discovery_active_only (10.1)
//!   search_currencies      → test_discovery_active_only (10.1)

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;
use holochain::sweettest::*;

/// Test 10.1: Active currencies appear in list and search; suspended/retired don't
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_discovery_active_only() {
    println!("Test 10.1: Discovery — Active Only");

    let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path)
        .await
        .expect("Load DNA");
    let mut conductor = SweetConductor::from_standard_config().await;
    let agents = SweetAgents::get(conductor.keystore(), 1).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-finance", &agents, &[dna])
        .await
        .expect("Install app");

    let cell = &apps[0].cells()[0];
    let zome = cell.zome("currency_mint");

    // Create two currencies
    let def_a: CurrencyDefinition = conductor
        .call(
            &zome,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:discovery".into(),
                params: test_params("AlphaCoin", "AC"),
                governance_proposal_id: None,
            },
        )
        .await;

    let def_b: CurrencyDefinition = conductor
        .call(
            &zome,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:discovery".into(),
                params: test_params("BetaCoin", "BC"),
                governance_proposal_id: None,
            },
        )
        .await;

    // Activate both
    let _: CurrencyDefinition = conductor
        .call(
            &zome,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def_a.id.clone(),
            },
        )
        .await;
    let _: CurrencyDefinition = conductor
        .call(
            &zome,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def_b.id.clone(),
            },
        )
        .await;

    // Both should appear in list
    let active: Vec<CurrencyDefinition> =
        conductor.call(&zome, "list_active_currencies", ()).await;
    assert!(
        active.iter().any(|c| c.id == def_a.id),
        "AlphaCoin should be in active list"
    );
    assert!(
        active.iter().any(|c| c.id == def_b.id),
        "BetaCoin should be in active list"
    );
    println!(
        "  - Both currencies in active list ({} total)",
        active.len()
    );

    // Search by name (case-insensitive)
    let search_results: Vec<CurrencyDefinition> = conductor
        .call(&zome, "search_currencies", "alpha".to_string())
        .await;
    assert!(
        search_results.iter().any(|c| c.id == def_a.id),
        "AlphaCoin should appear in search for 'alpha'"
    );
    assert!(
        !search_results.iter().any(|c| c.id == def_b.id),
        "BetaCoin should NOT appear in search for 'alpha'"
    );
    println!(
        "  - Search 'alpha' found {} result(s)",
        search_results.len()
    );

    // Suspend AlphaCoin
    let _: CurrencyDefinition = conductor
        .call(&zome, "suspend_currency", def_a.id.clone())
        .await;

    // AlphaCoin should no longer be in active list
    let active2: Vec<CurrencyDefinition> =
        conductor.call(&zome, "list_active_currencies", ()).await;
    assert!(
        !active2.iter().any(|c| c.id == def_a.id),
        "Suspended AlphaCoin should NOT be in active list"
    );
    assert!(
        active2.iter().any(|c| c.id == def_b.id),
        "BetaCoin should still be in active list"
    );
    println!("  - After suspend: AlphaCoin gone, BetaCoin remains");

    // Reactivate AlphaCoin
    let _: CurrencyDefinition = conductor
        .call(&zome, "reactivate_currency", def_a.id.clone())
        .await;

    let active3: Vec<CurrencyDefinition> =
        conductor.call(&zome, "list_active_currencies", ()).await;
    assert!(
        active3.iter().any(|c| c.id == def_a.id),
        "Reactivated AlphaCoin should be back in active list"
    );
    println!("  - After reactivate: AlphaCoin back in list");

    // Retire BetaCoin
    let _: CurrencyDefinition = conductor
        .call(&zome, "retire_currency", def_b.id.clone())
        .await;

    let active4: Vec<CurrencyDefinition> =
        conductor.call(&zome, "list_active_currencies", ()).await;
    assert!(
        !active4.iter().any(|c| c.id == def_b.id),
        "Retired BetaCoin should NOT be in active list"
    );
    println!("  - After retire: BetaCoin gone");

    println!("Test 10.1 PASSED");
}
