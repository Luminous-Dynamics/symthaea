//! Balance queries, portfolio, member listing, and DAO listing tests.
//!
//! Coverage:
//!   get_minted_balance     → (tested throughout exchanges.rs and stats.rs)
//!   get_member_portfolio   → test_portfolio_and_member_listing (11.1), test_multi_currency_portfolio (18.3)
//!   list_currency_members  → test_portfolio_and_member_listing (11.1), test_members_sorted_unique (E6)
//!   get_dao_currencies     → test_dao_currency_listing (5.1), test_multi_dao_isolation (E11)
//!   get_member_exchanges   → test_member_exchange_history (7.1)

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;
use holochain::sweettest::*;

/// Test 5.1: DAO can list all its currencies
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_dao_currency_listing() {
    println!("Test 5.1: DAO Currency Listing");

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
    let dao_did = "did:mycelix:dao:listing-test";

    // Create 3 currencies for the same DAO
    for (name, sym) in [("Alpha", "A"), ("Beta", "B"), ("Gamma", "G")] {
        let _: CurrencyDefinition = conductor
            .call(
                &zome,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: dao_did.into(),
                    params: test_params(name, sym),
                    governance_proposal_id: None,
                },
            )
            .await;
    }

    // List all
    let currencies: Vec<CurrencyDefinition> = conductor
        .call(&zome, "get_dao_currencies", dao_did.to_string())
        .await;

    assert_eq!(currencies.len(), 3, "Should have 3 currencies");
    println!("  - DAO has {} currencies", currencies.len());
    for c in &currencies {
        println!(
            "    - {} ({}) [{:?}]",
            c.params.name, c.params.symbol, c.status
        );
    }
    println!("Test 5.1 PASSED");
}

/// Test 7.1: Member exchange history with pagination
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_member_exchange_history() {
    println!("Test 7.1: Member Exchange History");

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
    let provider_did = format!("did:mycelix:{}", agents[0]);
    let receiver_did = format!("did:mycelix:{}", agents[1]);

    // Create and activate
    let def: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:history-test".into(),
                params: test_params("HistoryCoin", "HC"),
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

    // Record 5 exchanges
    for i in 1..=5 {
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: receiver_did.clone(),
                    hours: i as f32,
                    service_description: format!("History service #{}", i),
                },
            )
            .await;
    }

    // Get all member exchanges
    let all: Vec<MintedExchange> = conductor
        .call(
            &zome_a,
            "get_member_exchanges",
            GetMemberExchangesInput {
                currency_id: def.id.clone(),
                member_did: provider_did.clone(),
                limit: None,
                after_timestamp: None,
            },
        )
        .await;
    assert_eq!(all.len(), 5, "Provider should see all 5 exchanges");

    // Get with limit
    let limited: Vec<MintedExchange> = conductor
        .call(
            &zome_a,
            "get_member_exchanges",
            GetMemberExchangesInput {
                currency_id: def.id.clone(),
                member_did: provider_did.clone(),
                limit: Some(3),
                after_timestamp: None,
            },
        )
        .await;
    assert_eq!(limited.len(), 3, "Limit=3 should return 3 exchanges");

    // Get currency exchanges with pagination
    let page1: Vec<MintedExchange> = conductor
        .call(
            &zome_a,
            "get_currency_exchanges",
            PaginatedCurrencyInput {
                currency_id: def.id.clone(),
                limit: Some(3),
                after_timestamp: None,
            },
        )
        .await;
    assert_eq!(page1.len(), 3, "Page 1 should have 3 exchanges");

    // Cursor pagination: get exchanges after the oldest in page1
    if let Some(oldest) = page1.last() {
        let page2: Vec<MintedExchange> = conductor
            .call(
                &zome_a,
                "get_currency_exchanges",
                PaginatedCurrencyInput {
                    currency_id: def.id.clone(),
                    limit: Some(10),
                    after_timestamp: Some(oldest.timestamp),
                },
            )
            .await;
        assert!(
            page2.len() < 5,
            "Cursor should filter some results (got {})",
            page2.len()
        );
        println!("  - Page 2 with cursor: {} exchanges", page2.len());
    }

    println!(
        "  - All: {}, Limited: {}, Page1: {}",
        all.len(),
        limited.len(),
        page1.len()
    );
    println!("Test 7.1 PASSED");
}

/// Test 11.1: Member portfolio across multiple currencies + list_currency_members
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_portfolio_and_member_listing() {
    println!("Test 11.1: Portfolio & Member Listing");

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
    let alice_did = format!("did:mycelix:{}", agents[0]);
    let bob_did = format!("did:mycelix:{}", agents[1]);

    // Create and activate two currencies
    let def_x: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:portfolio".into(),
                params: test_params("CurrencyX", "CX"),
                governance_proposal_id: None,
            },
        )
        .await;
    let _: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def_x.id.clone(),
            },
        )
        .await;

    let def_y: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "create_currency",
            CreateCurrencyInput {
                dao_did: "did:mycelix:dao:portfolio".into(),
                params: test_params("CurrencyY", "CY"),
                governance_proposal_id: None,
            },
        )
        .await;
    let _: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def_y.id.clone(),
            },
        )
        .await;

    // Alice provides service to Bob in both currencies
    let _: MintedExchange = conductor
        .call(
            &zome_a,
            "record_minted_exchange",
            RecordMintedExchangeInput {
                currency_id: def_x.id.clone(),
                receiver_did: bob_did.clone(),
                hours: 2.0,
                service_description: "Gardening in CurrencyX".into(),
            },
        )
        .await;
    let _: MintedExchange = conductor
        .call(
            &zome_a,
            "record_minted_exchange",
            RecordMintedExchangeInput {
                currency_id: def_y.id.clone(),
                receiver_did: bob_did.clone(),
                hours: 1.5,
                service_description: "Tutoring in CurrencyY".into(),
            },
        )
        .await;

    // Alice's portfolio should show both currencies
    let portfolio: Vec<MintedBalanceInfo> = conductor
        .call(&zome_a, "get_member_portfolio", alice_did.clone())
        .await;
    assert!(
        portfolio.len() >= 2,
        "Alice should be in at least 2 currencies, got {}",
        portfolio.len()
    );
    println!("  - Alice's portfolio: {} currencies", portfolio.len());

    // list_currency_members for CurrencyX should include both Alice and Bob
    let members_x: Vec<String> = conductor
        .call(&zome_a, "list_currency_members", def_x.id.clone())
        .await;
    assert!(
        members_x.contains(&alice_did),
        "Alice should be a member of CurrencyX"
    );
    assert!(
        members_x.contains(&bob_did),
        "Bob should be a member of CurrencyX"
    );
    assert_eq!(members_x.len(), 2, "Exactly 2 members (no compost)");
    println!(
        "  - CurrencyX members: {} (compost excluded)",
        members_x.len()
    );

    // Alice's exchange history in CurrencyX
    let exchanges: Vec<MintedExchange> = conductor
        .call(
            &zome_a,
            "get_member_exchanges",
            GetMemberExchangesInput {
                currency_id: def_x.id.clone(),
                member_did: alice_did.clone(),
                limit: None,
                after_timestamp: None,
            },
        )
        .await;
    assert_eq!(exchanges.len(), 1, "Alice has 1 exchange in CurrencyX");
    println!("  - Alice's CurrencyX exchanges: {}", exchanges.len());

    println!("Test 11.1 PASSED");
}

/// Test 18.3: Multi-currency portfolio — member has balances in 2 currencies
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_multi_currency_portfolio() {
    println!("Test 18.3: Multi-Currency Portfolio");

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

    let zome_a = apps[0].cells()[0].zome("currency_mint");
    let bob_did = format!("did:mycelix:{}", agents[1].clone());
    let alice_did = format!("did:mycelix:{}", agents[0].clone());

    // Create + activate currency 1
    let input1 = CreateCurrencyInput {
        dao_did: "did:mycelix:dao:multi-portfolio".into(),
        params: test_params("Alpha Hours", "AH"),
        governance_proposal_id: None,
    };
    let def1: CurrencyDefinition = conductor.call(&zome_a, "create_currency", input1).await;
    let _: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def1.id.clone(),
            },
        )
        .await;

    // Create + activate currency 2
    let input2 = CreateCurrencyInput {
        dao_did: "did:mycelix:dao:multi-portfolio".into(),
        params: test_params("Beta Credits", "BC"),
        governance_proposal_id: None,
    };
    let def2: CurrencyDefinition = conductor.call(&zome_a, "create_currency", input2).await;
    let _: CurrencyDefinition = conductor
        .call(
            &zome_a,
            "activate_currency",
            ActivateCurrencyInput {
                currency_id: def2.id.clone(),
            },
        )
        .await;

    // Alice provides 2 hours to Bob in currency 1
    let _: MintedExchange = conductor
        .call(
            &zome_a,
            "record_minted_exchange",
            RecordMintedExchangeInput {
                currency_id: def1.id.clone(),
                receiver_did: bob_did.clone(),
                hours: 2.0,
                service_description: "Multi-portfolio test AH".into(),
            },
        )
        .await;

    // Alice provides 1 hour to Bob in currency 2
    let _: MintedExchange = conductor
        .call(
            &zome_a,
            "record_minted_exchange",
            RecordMintedExchangeInput {
                currency_id: def2.id.clone(),
                receiver_did: bob_did.clone(),
                hours: 1.0,
                service_description: "Multi-portfolio test BC".into(),
            },
        )
        .await;
    println!("  - Exchanges in 2 currencies");

    // Alice's portfolio should have 2 entries
    let portfolio: Vec<MintedBalanceInfo> = conductor
        .call(&zome_a, "get_member_portfolio", alice_did.clone())
        .await;
    assert!(
        portfolio.len() >= 2,
        "Alice should have at least 2 currency balances, got {}",
        portfolio.len()
    );

    // Verify both currencies present
    let symbols: Vec<&str> = portfolio.iter().map(|p| p.currency_symbol.as_str()).collect();
    assert!(symbols.contains(&"AH"), "Portfolio should contain AH");
    assert!(symbols.contains(&"BC"), "Portfolio should contain BC");
    println!(
        "  - Portfolio: {} currencies: {:?}",
        portfolio.len(),
        symbols
    );

    // Verify balances
    for entry in &portfolio {
        if entry.currency_symbol == "AH" {
            assert_eq!(entry.balance, 2, "Alice +2 in AH");
        } else if entry.currency_symbol == "BC" {
            assert_eq!(entry.balance, 1, "Alice +1 in BC");
        }
    }
    println!("  - Balances verified: AH=+2, BC=+1");

    println!("Test 18.3 PASSED");
}

/// Test E6: list_currency_members returns sorted, deduplicated results
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_members_sorted_unique() {
    println!("Test E6: Members Sorted & Unique");

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
                dao_did: "did:mycelix:dao:sorted-members".into(),
                params: test_params("SortCoin", "SO"),
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

    // Record multiple exchanges between same pair → should NOT duplicate members
    for i in 1..=3 {
        let _: MintedExchange = conductor
            .call(
                &zome_a,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did.clone(),
                    hours: 1.0,
                    service_description: format!("Sort test #{}", i),
                },
            )
            .await;
    }

    let members: Vec<String> = conductor
        .call(&zome_a, "list_currency_members", def.id.clone())
        .await;

    // Should be exactly 2 unique members (no duplicates from multiple exchanges)
    assert_eq!(members.len(), 2, "Should have exactly 2 unique members");

    // Should be sorted
    let mut sorted = members.clone();
    sorted.sort();
    assert_eq!(members, sorted, "Members should be sorted alphabetically");

    // Should not contain compost pseudo-member
    assert!(
        !members.iter().any(|m| m.contains("__compost__")),
        "Compost pseudo-member should be excluded"
    );
    println!(
        "  - {} unique members, sorted, compost excluded",
        members.len()
    );

    println!("Test E6 PASSED");
}

/// Test E11: Multi-DAO isolation — each DAO sees only its own currencies
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_multi_dao_isolation() {
    println!("Test E11: Multi-DAO Isolation");

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

    let dao_a = "did:mycelix:dao:isolation-alpha";
    let dao_b = "did:mycelix:dao:isolation-beta";

    // Create 2 currencies for DAO A
    for (name, sym) in [("Alpha1", "A1"), ("Alpha2", "A2")] {
        let _: CurrencyDefinition = conductor
            .call(
                &zome,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: dao_a.into(),
                    params: test_params(name, sym),
                    governance_proposal_id: None,
                },
            )
            .await;
    }

    // Create 3 currencies for DAO B
    for (name, sym) in [("Beta1", "B1"), ("Beta2", "B2"), ("Beta3", "B3")] {
        let _: CurrencyDefinition = conductor
            .call(
                &zome,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: dao_b.into(),
                    params: test_params(name, sym),
                    governance_proposal_id: None,
                },
            )
            .await;
    }

    // DAO A should see exactly 2
    let a_currencies: Vec<CurrencyDefinition> = conductor
        .call(&zome, "get_dao_currencies", dao_a.to_string())
        .await;
    assert_eq!(a_currencies.len(), 2, "DAO A should have 2 currencies");
    for c in &a_currencies {
        assert!(
            c.params.symbol.starts_with('A'),
            "DAO A should only contain Alpha currencies, found {}",
            c.params.symbol
        );
    }
    println!("  - DAO A: {} currencies (no cross-contamination)", a_currencies.len());

    // DAO B should see exactly 3
    let b_currencies: Vec<CurrencyDefinition> = conductor
        .call(&zome, "get_dao_currencies", dao_b.to_string())
        .await;
    assert_eq!(b_currencies.len(), 3, "DAO B should have 3 currencies");
    for c in &b_currencies {
        assert!(
            c.params.symbol.starts_with('B'),
            "DAO B should only contain Beta currencies, found {}",
            c.params.symbol
        );
    }
    println!("  - DAO B: {} currencies (no cross-contamination)", b_currencies.len());

    // Non-existent DAO should return empty
    let empty: Vec<CurrencyDefinition> = conductor
        .call(
            &zome,
            "get_dao_currencies",
            "did:mycelix:dao:nonexistent".to_string(),
        )
        .await;
    assert!(empty.is_empty(), "Non-existent DAO should have 0 currencies");
    println!("  - Non-existent DAO: 0 currencies");

    println!("Test E11 PASSED");
}
