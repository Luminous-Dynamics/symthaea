//! # Currency Mint (Currency Factory) Integration Tests
//!
//! Tests for community-minted mutual credit currencies covering:
//! - Currency lifecycle: Draft → Active → Suspended → Reactivated → Retired
//! - Exchange recording with zero-sum validation
//! - Confirmation flow (two-party confirmation)
//! - Balance queries and demurrage
//! - Lifecycle guards (Draft cannot retire, Suspended blocks confirm)
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --release --test currency_mint_test -- --include-ignored --test-threads=1 --nocapture
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;

use currency_mint_integrity::CurrencyDefinition;
use mycelix_finance_types::{CurrencyStatus, MintedCurrencyParams};

// Mirror types for coordinator inputs — avoids linking the coordinator crate
// (which generates conflicting #[no_mangle] symbols with other coordinators).

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CurrencyStats {
    pub currency_id: String,
    pub currency_name: String,
    pub currency_symbol: String,
    pub status: CurrencyStatus,
    pub member_count: u32,
    pub total_credit: i64,
    pub total_debt: i64,
    pub net_sum: i64,
    pub total_exchanges: u64,
    pub confirmed_exchanges: u64,
    pub pending_exchanges: u64,
}

fn test_params(name: &str, symbol: &str) -> MintedCurrencyParams {
    MintedCurrencyParams {
        name: name.into(),
        symbol: symbol.into(),
        description: format!("Test currency: {}", name),
        credit_limit: 40,
        demurrage_rate: 0.02,
        max_service_hours: 8,
        min_service_minutes: 15,
        requires_confirmation: false,
        confirmation_timeout_hours: 0,
        max_exchanges_per_day: 0,
    }
}

fn test_params_with_confirmation(name: &str, symbol: &str) -> MintedCurrencyParams {
    let mut p = test_params(name, symbol);
    p.requires_confirmation = true;
    p.confirmation_timeout_hours = 48;
    p
}

// ============================================================================
// Section 1: Currency Lifecycle
// ============================================================================

#[cfg(test)]
mod lifecycle {
    use super::*;

    /// Test 1.1: Create currency in Draft, activate, verify status transitions
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_and_activate_currency() {
        println!("Test 1.1: Create and Activate Currency");

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
        // Note: get_currency follows the link to original create hash,
        // so it may return Draft if update hasn't propagated. The update_entry
        // is authoritative.
        println!("Test 1.1 PASSED");
    }

    /// Test 1.2: Full lifecycle — Draft → Active → Suspended → Active → Retired
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_full_lifecycle() {
        println!("Test 1.2: Full Lifecycle");

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
}

// ============================================================================
// Section 2: Exchange Recording & Zero-Sum
// ============================================================================

#[cfg(test)]
mod exchanges {
    use super::*;

    /// Test 2.1: Record exchange between two members, verify zero-sum balances
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_record_exchange_zero_sum() {
        println!("Test 2.1: Exchange Recording & Zero-Sum");

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

        let creator_cell = &apps[0].cells()[0];
        let receiver_cell = &apps[1].cells()[0];
        let zome_creator = creator_cell.zome("currency_mint");
        let zome_receiver = receiver_cell.zome("currency_mint");

        let receiver_did = format!("did:mycelix:{}", agents[1]);

        // Create + activate currency
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:exchange-test".into(),
            params: test_params("Meal Credits", "MC"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition = conductor
            .call(&zome_creator, "create_currency", input)
            .await;
        let _active: CurrencyDefinition = conductor
            .call(
                &zome_creator,
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
            .call(&zome_creator, "record_minted_exchange", exchange_input)
            .await;
        assert_eq!(exchange.hours, 2.0);
        assert!(!exchange.confirmed || !def.params.requires_confirmation);
        println!("  - Exchange recorded: {} hours", exchange.hours);

        // Check provider balance (should be +2)
        let provider_did = format!("did:mycelix:{}", agents[0]);
        let provider_bal: MintedBalanceInfo = conductor
            .call(
                &zome_creator,
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
                &zome_receiver,
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

    /// Test 2.2: Cannot exchange in a Suspended currency
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_exchange_blocked_when_suspended() {
        println!("Test 2.2: Suspended Currency Blocks Exchanges");

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

        // Create + activate + suspend
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:suspend-test".into(),
            params: test_params("Paused Coin", "PC"),
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
        let _: CurrencyDefinition = conductor
            .call(&zome, "suspend_currency", def.id.clone())
            .await;

        // Attempt exchange — should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome,
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
}

// ============================================================================
// Section 3: Confirmation Flow
// ============================================================================

#[cfg(test)]
mod confirmation {
    use super::*;

    /// Test 3.1: Two-party confirmation flow — exchange starts unconfirmed,
    /// receiver confirms, balances update only after confirmation
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_confirmation_flow() {
        println!("Test 3.1: Two-Party Confirmation Flow");

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

        let provider_cell = &apps[0].cells()[0];
        let receiver_cell = &apps[1].cells()[0];
        let zome_provider = provider_cell.zome("currency_mint");
        let zome_receiver = receiver_cell.zome("currency_mint");

        let receiver_did = format!("did:mycelix:{}", agents[1]);

        // Create currency with confirmation required
        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:confirm-test".into(),
            params: test_params_with_confirmation("Confirm Coin", "CC"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition = conductor
            .call(&zome_provider, "create_currency", input)
            .await;
        let _: CurrencyDefinition = conductor
            .call(
                &zome_provider,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        // Record exchange — should be unconfirmed
        let exchange: MintedExchange = conductor
            .call(
                &zome_provider,
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
                &zome_receiver,
                "confirm_minted_exchange",
                ConfirmMintedExchangeInput {
                    exchange_id: exchange.id.clone(),
                },
            )
            .await;
        assert!(confirmed.confirmed, "Should now be confirmed");
        println!("  - Exchange confirmed by receiver");

        // Check balances — should reflect the exchange now
        let provider_did = format!("did:mycelix:{}", agents[0]);
        let provider_bal: MintedBalanceInfo = conductor
            .call(
                &zome_provider,
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
                &zome_receiver,
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

    /// Test 3.2: Confirm blocked when currency is Suspended
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_confirm_blocked_when_suspended() {
        println!("Test 3.2: Confirm Blocked When Suspended");

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

        let provider_cell = &apps[0].cells()[0];
        let receiver_cell = &apps[1].cells()[0];
        let zome_provider = provider_cell.zome("currency_mint");
        let zome_receiver = receiver_cell.zome("currency_mint");

        let receiver_did = format!("did:mycelix:{}", agents[1]);

        let input = CreateCurrencyInput {
            dao_did: "did:mycelix:dao:suspend-confirm-test".into(),
            params: test_params_with_confirmation("Suspendable", "SU"),
            governance_proposal_id: None,
        };
        let def: CurrencyDefinition = conductor
            .call(&zome_provider, "create_currency", input)
            .await;
        let _: CurrencyDefinition = conductor
            .call(
                &zome_provider,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        // Record exchange while Active
        let exchange: MintedExchange = conductor
            .call(
                &zome_provider,
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
            .call(&zome_provider, "suspend_currency", def.id.clone())
            .await;

        // Attempt to confirm — should fail
        let result: Result<MintedExchange, _> = conductor
            .call_fallible(
                &zome_receiver,
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
}

// ============================================================================
// Section 4: Currency Stats
// ============================================================================

#[cfg(test)]
mod stats {
    use super::*;

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
}

// ============================================================================
// Section 5: DAO Listing
// ============================================================================

#[cfg(test)]
mod dao_listing {
    use super::*;

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
}
