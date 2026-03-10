//! Balance queries, portfolio, member listing, and DAO listing tests.
//!
//! Coverage:
//!   get_minted_balance     → (tested throughout exchanges.rs and stats.rs)
//!   get_member_portfolio   → scenario 11.1, 18.3
//!   list_currency_members  → scenario 11.1, E6
//!   get_dao_currencies     → scenario 5.1, E11
//!   get_member_exchanges   → scenario 7.1
//!
//! All 6 scenarios share a single SweetConductor (2 agents).

use super::common::*;
use currency_mint_integrity::CurrencyDefinition;

/// Runs block_on inside a thread with 16MB stack because the consolidated
/// async state machine exceeds default stack sizes.
#[test]
#[ignore]
fn test_balances_all() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(test_balances_all_inner());
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn test_balances_all_inner() {
    let (conductor, agents, apps) = setup_finance_conductor(2).await;
    let cell_a = &apps[0].cells()[0];
    let zome_a = cell_a.zome("currency_mint");
    let alice_did = format!("did:mycelix:{}", agents[0]);
    let bob_did = format!("did:mycelix:{}", agents[1]);

    // ── Scenario 5.1: DAO can list all its currencies ─────────────────
    println!("Scenario 5.1: DAO Currency Listing");
    {
        let dao_did = "did:mycelix:dao:listing-test";
        for (name, sym) in [("Alpha", "A"), ("Beta", "B"), ("Gamma", "G")] {
            let _: CurrencyDefinition = call_with_retry(
                &conductor,
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: dao_did.into(),
                    params: test_params(name, sym),
                    governance_proposal_id: None,
                },
            )
            .await;
        }

        let currencies: Vec<CurrencyDefinition> = conductor
            .call(&zome_a, "get_dao_currencies", dao_did.to_string())
            .await;

        assert_eq!(currencies.len(), 3, "Should have 3 currencies");
        println!("  - DAO has {} currencies", currencies.len());
        for c in &currencies {
            println!(
                "    - {} ({}) [{:?}]",
                c.params.name, c.params.symbol, c.status
            );
        }
        println!("Scenario 5.1 PASSED");
    }

    // ── Scenario 7.1: Member exchange history with pagination ─────────
    println!("Scenario 7.1: Member Exchange History");
    {
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

        for i in 1..=5 {
            let _: MintedExchange = conductor
                .call(
                    &zome_a,
                    "record_minted_exchange",
                    RecordMintedExchangeInput {
                        currency_id: def.id.clone(),
                        receiver_did: bob_did.clone(),
                        hours: i as f32,
                        service_description: format!("History service #{}", i),
                    },
                )
                .await;
        }

        let all: Vec<MintedExchange> = conductor
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
        assert_eq!(all.len(), 5, "Provider should see all 5 exchanges");

        let limited: Vec<MintedExchange> = conductor
            .call(
                &zome_a,
                "get_member_exchanges",
                GetMemberExchangesInput {
                    currency_id: def.id.clone(),
                    member_did: alice_did.clone(),
                    limit: Some(3),
                    after_timestamp: None,
                },
            )
            .await;
        assert_eq!(limited.len(), 3, "Limit=3 should return 3 exchanges");

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
        println!("Scenario 7.1 PASSED");
    }

    // ── Scenario 11.1: Portfolio across multiple currencies + member listing ──
    println!("Scenario 11.1: Portfolio & Member Listing");
    {
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

        let portfolio: Vec<MintedBalanceInfo> = conductor
            .call(&zome_a, "get_member_portfolio", alice_did.clone())
            .await;
        assert!(
            portfolio.len() >= 2,
            "Alice should be in at least 2 currencies, got {}",
            portfolio.len()
        );
        println!("  - Alice's portfolio: {} currencies", portfolio.len());

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

        println!("Scenario 11.1 PASSED");
    }

    // ── Scenario 18.3: Multi-currency portfolio ───────────────────────
    println!("Scenario 18.3: Multi-Currency Portfolio");
    {
        let def1: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:multi-portfolio".into(),
                    params: test_params("Alpha Hours", "AH"),
                    governance_proposal_id: None,
                },
            )
            .await;
        let _: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def1.id.clone(),
                },
            )
            .await;

        let def2: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:multi-portfolio".into(),
                    params: test_params("Beta Credits", "BC"),
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

        let portfolio: Vec<MintedBalanceInfo> = conductor
            .call(&zome_a, "get_member_portfolio", alice_did.clone())
            .await;
        assert!(
            portfolio.len() >= 2,
            "Alice should have at least 2 currency balances, got {}",
            portfolio.len()
        );

        let symbols: Vec<&str> = portfolio.iter().map(|p| p.currency_symbol.as_str()).collect();
        assert!(symbols.contains(&"AH"), "Portfolio should contain AH");
        assert!(symbols.contains(&"BC"), "Portfolio should contain BC");
        println!(
            "  - Portfolio: {} currencies: {:?}",
            portfolio.len(),
            symbols
        );

        for entry in &portfolio {
            if entry.currency_symbol == "AH" {
                assert_eq!(entry.balance, 2, "Alice +2 in AH");
            } else if entry.currency_symbol == "BC" {
                assert_eq!(entry.balance, 1, "Alice +1 in BC");
            }
        }
        println!("  - Balances verified: AH=+2, BC=+1");

        println!("Scenario 18.3 PASSED");
    }

    // ── Scenario E6: Members sorted & unique ──────────────────────────
    println!("Scenario E6: Members Sorted & Unique");
    {
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

        assert_eq!(members.len(), 2, "Should have exactly 2 unique members");

        let mut sorted = members.clone();
        sorted.sort();
        assert_eq!(members, sorted, "Members should be sorted alphabetically");

        assert!(
            !members.iter().any(|m| m.contains("__compost__")),
            "Compost pseudo-member should be excluded"
        );
        println!(
            "  - {} unique members, sorted, compost excluded",
            members.len()
        );

        println!("Scenario E6 PASSED");
    }

    // ── Scenario E11: Multi-DAO isolation ─────────────────────────────
    println!("Scenario E11: Multi-DAO Isolation");
    {
        let dao_a = "did:mycelix:dao:isolation-alpha";
        let dao_b = "did:mycelix:dao:isolation-beta";

        for (name, sym) in [("Alpha1", "A1"), ("Alpha2", "A2")] {
            let _: CurrencyDefinition = conductor
                .call(
                    &zome_a,
                    "create_currency",
                    CreateCurrencyInput {
                        dao_did: dao_a.into(),
                        params: test_params(name, sym),
                        governance_proposal_id: None,
                    },
                )
                .await;
        }

        for (name, sym) in [("Beta1", "B1"), ("Beta2", "B2"), ("Beta3", "B3")] {
            let _: CurrencyDefinition = conductor
                .call(
                    &zome_a,
                    "create_currency",
                    CreateCurrencyInput {
                        dao_did: dao_b.into(),
                        params: test_params(name, sym),
                        governance_proposal_id: None,
                    },
                )
                .await;
        }

        let a_currencies: Vec<CurrencyDefinition> = conductor
            .call(&zome_a, "get_dao_currencies", dao_a.to_string())
            .await;
        assert_eq!(a_currencies.len(), 2, "DAO A should have 2 currencies");
        for c in &a_currencies {
            assert!(
                c.params.symbol.starts_with('A'),
                "DAO A should only contain Alpha currencies, found {}",
                c.params.symbol
            );
        }
        println!(
            "  - DAO A: {} currencies (no cross-contamination)",
            a_currencies.len()
        );

        let b_currencies: Vec<CurrencyDefinition> = conductor
            .call(&zome_a, "get_dao_currencies", dao_b.to_string())
            .await;
        assert_eq!(b_currencies.len(), 3, "DAO B should have 3 currencies");
        for c in &b_currencies {
            assert!(
                c.params.symbol.starts_with('B'),
                "DAO B should only contain Beta currencies, found {}",
                c.params.symbol
            );
        }
        println!(
            "  - DAO B: {} currencies (no cross-contamination)",
            b_currencies.len()
        );

        let empty: Vec<CurrencyDefinition> = conductor
            .call(
                &zome_a,
                "get_dao_currencies",
                "did:mycelix:dao:nonexistent".to_string(),
            )
            .await;
        assert!(empty.is_empty(), "Non-existent DAO should have 0 currencies");
        println!("  - Non-existent DAO: 0 currencies");

        println!("Scenario E11 PASSED");
    }

    println!("=== All balance scenarios PASSED ===");
}
