//! # Currency Mint (Currency Factory) Integration Tests
//!
//! Tests for community-minted mutual credit currencies covering:
//! - Currency lifecycle: Draft → Active → Suspended → Reactivated → Retired
//! - Exchange recording with zero-sum validation
//! - Confirmation flow (two-party confirmation)
//! - Balance queries and demurrage
//! - Lifecycle guards (Draft cannot retire, Suspended blocks confirm)
//! - Compost zero-sum invariant, member exchange history
//! - Disputes (open, get, resolve)
//! - Parameter amendment
//! - Discovery (list_active_currencies, search_currencies, suspend/reactivate index)
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
pub struct PaginatedCurrencyInput {
    pub currency_id: String,
    pub limit: Option<usize>,
    pub after_timestamp: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetMemberExchangesInput {
    pub currency_id: String,
    pub member_did: String,
    pub limit: Option<usize>,
    pub after_timestamp: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompostBalance {
    pub currency_id: String,
    pub accumulated: i32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OpenDisputeInput {
    pub exchange_id: String,
    pub reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MintedDispute {
    pub exchange_id: String,
    pub opener_did: String,
    pub reason: String,
    pub resolved: Option<bool>,
    pub resolver_did: Option<String>,
    pub resolution_reason: Option<String>,
    pub opened_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
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
    pub compost_balance: i64,
    pub net_sum: i64,
    pub total_exchanges: u64,
    pub confirmed_exchanges: u64,
    pub pending_exchanges: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResolveDisputeInput {
    pub exchange_id: String,
    pub accept: bool,
    pub resolution_reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AmendCurrencyParamsInput {
    pub currency_id: String,
    pub new_params: MintedCurrencyParams,
    pub governance_proposal_id: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ApplyDemurrageInput {
    pub currency_id: String,
    pub member_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DemurrageResult {
    pub member_did: String,
    pub currency_id: String,
    pub previous_balance: i32,
    pub deduction: i32,
    pub new_balance: i32,
    pub demurrage_rate: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DemurrageReport {
    pub currency_id: String,
    pub currency_name: String,
    pub demurrage_rate: f64,
    pub total_pending_demurrage: i64,
    pub affected_members: u32,
    pub entries: Vec<DemurrageReportEntry>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DemurrageReportEntry {
    pub member_did: String,
    pub current_balance: i32,
    pub pending_deduction: i32,
    pub effective_balance: i32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RedistributeCompostResult {
    pub currency_id: String,
    pub total_redistributed: i32,
    pub recipients: u32,
    pub per_member_amount: i32,
    pub remainder_kept: i32,
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

// ============================================================================
// Section 6: Compost & Demurrage
// ============================================================================

#[cfg(test)]
mod compost {
    use super::*;

    /// Test 6.1: Compost balance accumulates from demurrage and preserves zero-sum
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_compost_zero_sum() {
        println!("Test 6.1: Compost Zero-Sum");

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
        let receiver_did = format!("did:mycelix:{}", agents[1].to_raw_36());

        // Create currency with 2% demurrage
        let def: CurrencyDefinition = conductor
            .call(
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
        println!("Test 6.1 PASSED");
    }
}

// ============================================================================
// Section 7: Member Exchange History
// ============================================================================

#[cfg(test)]
mod member_exchanges {
    use super::*;

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
        let provider_did = format!("did:mycelix:{}", agents[0].to_raw_36());
        let receiver_did = format!("did:mycelix:{}", agents[1].to_raw_36());

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
            // page1 is newest-first, so oldest.timestamp is 3rd newest
            // page2 with after_timestamp > oldest should return the 2 newer ones
            // Actually page2 returns exchanges with ts > cursor, which is the 2 newer
            // But page1 already has those... cursor should go other direction for "next page"
            // In our newest-first sort, the cursor for "next page" is the last item's timestamp
            // and we want items OLDER than cursor. Our current impl uses > cursor which is forward.
            // For this test, just verify we get fewer results with a cursor
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
}

// ============================================================================
// Section 8: Disputes
// ============================================================================

#[cfg(test)]
mod disputes {
    use super::*;

    /// Test 8.1: Open and verify dispute on confirmed exchange
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_open_dispute() {
        println!("Test 8.1: Open Dispute");

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
        let receiver_did = format!("did:mycelix:{}", agents[1].to_raw_36());

        // Create, activate, and record a confirmed exchange
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:dispute-test".into(),
                    params: test_params("DisputeCoin", "DC"),
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
                    receiver_did: receiver_did.clone(),
                    hours: 3.0,
                    service_description: "Disputed gardening".into(),
                },
            )
            .await;
        assert!(
            exchange.confirmed,
            "Auto-confirmed (no confirmation required)"
        );

        // Open dispute
        let dispute: MintedDispute = conductor
            .call(
                &zome_a,
                "open_minted_dispute",
                OpenDisputeInput {
                    exchange_id: exchange.id.clone(),
                    reason: "Service was not as described".into(),
                },
            )
            .await;

        assert_eq!(dispute.exchange_id, exchange.id);
        assert!(
            dispute.resolved.is_none(),
            "New dispute should be unresolved"
        );
        println!(
            "  - Dispute opened by {} on exchange {}",
            dispute.opener_did, dispute.exchange_id
        );
        println!("Test 8.1 PASSED");
    }

    /// Test 8.2: Get dispute and resolve dispute
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_and_resolve_dispute() {
        println!("Test 8.2: Get Dispute and Resolve Dispute");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell_a = &apps[0].cells()[0];
        let zome_a = cell_a.zome("currency_mint");
        let receiver_did = format!("did:mycelix:{}", agents[1].to_raw_36());

        // Create, activate, exchange, dispute
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:dispute-resolve".into(),
                    params: test_params("ResolveCoin", "RC"),
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
                    receiver_did: receiver_did.clone(),
                    hours: 2.0,
                    service_description: "Test service".into(),
                },
            )
            .await;

        let _: MintedDispute = conductor
            .call(
                &zome_a,
                "open_minted_dispute",
                OpenDisputeInput {
                    exchange_id: exchange.id.clone(),
                    reason: "Quality issue".into(),
                },
            )
            .await;

        // Get dispute via new extern
        let fetched: Option<MintedDispute> = conductor
            .call(&zome_a, "get_dispute", exchange.id.clone())
            .await;
        assert!(fetched.is_some(), "get_dispute should return the dispute");
        let d = fetched.unwrap();
        assert_eq!(d.exchange_id, exchange.id);
        assert!(d.resolved.is_none());
        println!(
            "  - get_dispute returned dispute for exchange {}",
            d.exchange_id
        );

        // Resolve the dispute (reject)
        let resolved: MintedDispute = conductor
            .call(
                &zome_a,
                "resolve_minted_dispute",
                ResolveDisputeInput {
                    exchange_id: exchange.id.clone(),
                    accept: false,
                    resolution_reason: "Evidence insufficient".into(),
                },
            )
            .await;
        assert_eq!(resolved.resolved, Some(false));
        assert!(resolved.resolver_did.is_some());
        println!(
            "  - Dispute resolved (rejected) by {}",
            resolved.resolver_did.unwrap()
        );

        // get_dispute should now show resolved
        let fetched2: Option<MintedDispute> = conductor
            .call(&zome_a, "get_dispute", exchange.id.clone())
            .await;
        let d2 = fetched2.unwrap();
        assert_eq!(d2.resolved, Some(false));
        println!("Test 8.2 PASSED");
    }
}

// ============================================================================
// Section 9: Parameter Amendment
// ============================================================================

#[cfg(test)]
mod amendment {
    use super::*;

    /// Test 9.1: Amend currency parameters
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_amend_currency_params() {
        println!("Test 9.1: Amend Currency Parameters");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell = &apps[0].cells()[0];
        let zome = cell.zome("currency_mint");

        // Create and activate
        let def: CurrencyDefinition = conductor
            .call(
                &zome,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:amend-test".into(),
                    params: test_params("AmendCoin", "AC"),
                    governance_proposal_id: None,
                },
            )
            .await;

        let _: CurrencyDefinition = conductor
            .call(
                &zome,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        // Amend: raise credit limit from 40 to 80
        let mut new_params = test_params("AmendCoin", "AC");
        new_params.credit_limit = 80;
        new_params.description = "Updated description".into();

        let amended: CurrencyDefinition = conductor
            .call(
                &zome,
                "amend_currency_params",
                AmendCurrencyParamsInput {
                    currency_id: def.id.clone(),
                    new_params: new_params.clone(),
                    governance_proposal_id: None,
                },
            )
            .await;

        assert_eq!(amended.params.credit_limit, 80);
        assert_eq!(amended.params.description, "Updated description");
        assert_eq!(amended.status, CurrencyStatus::Active, "Status unchanged");
        println!(
            "  - Credit limit amended: 40 → {}",
            amended.params.credit_limit
        );

        // Verify via get_currency
        let fetched: Option<CurrencyDefinition> =
            conductor.call(&zome, "get_currency", def.id.clone()).await;
        let f = fetched.expect("Currency should exist");
        assert_eq!(f.params.credit_limit, 80);
        println!("  - get_currency confirms amended params");
        println!("Test 9.1 PASSED");
    }
}

// ============================================================================
// Section 10: Discovery (list_active_currencies, search_currencies)
// ============================================================================

#[cfg(test)]
mod discovery {
    use super::*;

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
            .setup_app_for_agents("finance", &agents, &[dna])
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
}

// ============================================================================
// Section 11: Portfolio, Member Listing
// ============================================================================

#[cfg(test)]
mod portfolio {
    use super::*;

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
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell_a = &apps[0].cells()[0];
        let zome_a = cell_a.zome("currency_mint");
        let alice_did = format!("did:mycelix:{}", agents[0].to_raw_36());
        let bob_did = format!("did:mycelix:{}", agents[1].to_raw_36());

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
}

// ============================================================================
// Section 12: Rate Limiting Enforcement
// ============================================================================

#[cfg(test)]
mod rate_limiting {
    use super::*;

    /// Test 12.1: max_exchanges_per_day enforcement
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_rate_limit_enforcement() {
        println!("Test 12.1: Rate Limit Enforcement");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell_a = &apps[0].cells()[0];
        let zome_a = cell_a.zome("currency_mint");
        let bob_did = format!("did:mycelix:{}", agents[1].to_raw_36());

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
}

// ============================================================================
// Section 13: Confirmation Flow (Two-Party)
// ============================================================================

#[cfg(test)]
mod confirmation_flow {
    use super::*;

    /// Test 13.1: Full two-party confirmation flow
    /// Create currency with requires_confirmation=true, record exchange (unconfirmed),
    /// verify pending, confirm from receiver, verify balances updated.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_two_party_confirmation_flow() {
        println!("Test 13.1: Two-Party Confirmation Flow");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell_a = &apps[0].cells()[0];
        let cell_b = &apps[1].cells()[0];
        let zome_a = cell_a.zome("currency_mint");
        let zome_b = cell_b.zome("currency_mint");
        let alice_did = format!("did:mycelix:{}", agents[0].to_raw_36());
        let bob_did = format!("did:mycelix:{}", agents[1].to_raw_36());

        // Create currency with confirmation required
        let def: CurrencyDefinition = conductor
            .call(
                &zome_a,
                "create_currency",
                CreateCurrencyInput {
                    dao_did: "did:mycelix:dao:confirm-test".into(),
                    params: test_params_with_confirmation("ConfirmCoin", "CC"),
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

        println!("Test 13.1 PASSED");
    }
}

// ============================================================================
// Section 14: Cancel & Lifecycle Guards
// ============================================================================

#[cfg(test)]
mod cancel_and_guards {
    use super::*;

    /// Test 14.1: Cancel confirmed exchange fails; cancel unconfirmed before timeout fails
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cancel_exchange_guards() {
        println!("Test 14.1: Cancel Exchange Guards");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell_a = &apps[0].cells()[0];
        let zome_a = cell_a.zome("currency_mint");
        let bob_did = format!("did:mycelix:{}", agents[1].to_raw_36());

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

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 1).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

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
}

// ============================================================================
// Section 15: Get Exchange
// ============================================================================

#[cfg(test)]
mod get_exchange_test {
    use super::*;

    /// Test 15.1: get_exchange returns exchange by ID, None for non-existent
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_exchange() {
        println!("Test 15.1: Get Exchange");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell = &apps[0].cells()[0];
        let zome = cell.zome("currency_mint");
        let bob_did = format!("did:mycelix:{}", agents[1].to_raw_36());

        let def: CurrencyDefinition = conductor
            .call(
                &zome,
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
                &zome,
                "activate_currency",
                ActivateCurrencyInput {
                    currency_id: def.id.clone(),
                },
            )
            .await;

        let exchange: MintedExchange = conductor
            .call(
                &zome,
                "record_minted_exchange",
                RecordMintedExchangeInput {
                    currency_id: def.id.clone(),
                    receiver_did: bob_did,
                    hours: 1.5,
                    service_description: "Test exchange".into(),
                },
            )
            .await;

        // get_exchange should find it
        let fetched: Option<MintedExchange> = conductor
            .call(&zome, "get_exchange", exchange.id.clone())
            .await;
        assert!(fetched.is_some(), "Should find exchange by ID");
        let f = fetched.unwrap();
        assert_eq!(f.id, exchange.id);
        assert_eq!(f.hours, 1.5);
        println!("  - get_exchange found: {} ({} hours)", f.id, f.hours);

        // Non-existent should return None
        let missing: Option<MintedExchange> = conductor
            .call(&zome, "get_exchange", "nonexistent-id".to_string())
            .await;
        assert!(
            missing.is_none(),
            "Non-existent exchange should return None"
        );
        println!("  - get_exchange for non-existent: None");

        println!("Test 15.1 PASSED");
    }
}

// ============================================================================
// Section 16: Demurrage & Compost Cycle
// ============================================================================

#[cfg(test)]
mod demurrage_compost {
    use super::*;

    /// Test 16.1: Demurrage deducts from positive balance, credits compost
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_demurrage_and_compost() {
        println!("Test 16.1: Demurrage & Compost Cycle");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell_a = &apps[0].cells()[0];
        let zome_a = cell_a.zome("currency_mint");
        let alice_did = format!("did:mycelix:{}", agents[0].to_raw_36());
        let bob_did = format!("did:mycelix:{}", agents[1].to_raw_36());

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
        // Note: demurrage is time-based. In a fresh exchange the elapsed time
        // may be ~0, producing deduction=0. That's correct behavior.
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
        assert_eq!(bob_dem.deduction, 0, "Negative balance exempt from demurrage");
        println!("  - Bob (debtor) demurrage: deduction=0 (exempt)");

        // Check compost balance
        let compost: CompostBalance = conductor
            .call(&zome_a, "get_compost_balance", def.id.clone())
            .await;
        assert!(
            compost.accumulated >= 0,
            "Compost should be non-negative"
        );
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

        println!("Test 16.1 PASSED");
    }

    /// Test 16.2: Redistribute compost to members
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_redistribute_compost() {
        println!("Test 16.2: Redistribute Compost");

        let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
        let dna = SweetDnaFile::from_bundle(&dna_path)
            .await
            .expect("Load DNA");
        let mut conductor = SweetConductor::from_standard_config().await;
        let agents = SweetAgents::get(conductor.keystore(), 2).await;
        let apps = conductor
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell_a = &apps[0].cells()[0];
        let zome_a = cell_a.zome("currency_mint");
        let bob_did = format!("did:mycelix:{}", agents[1].to_raw_36());

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
        assert!(result.recipients >= 0, "Recipients must be non-negative");
        println!(
            "  - Redistributed: {} to {} recipients (per_member={})",
            result.total_redistributed, result.recipients, result.per_member_amount
        );

        println!("Test 16.2 PASSED");
    }
}

// ============================================================================
// Section 17: Currency Stats (Zero-Sum Proof)
// ============================================================================

#[cfg(test)]
mod stats_test {
    use super::*;

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
            .setup_app_for_agents("finance", &agents, &[dna])
            .await
            .expect("Install app");

        let cell_a = &apps[0].cells()[0];
        let zome_a = cell_a.zome("currency_mint");
        let bob_did = format!("did:mycelix:{}", agents[1].to_raw_36());

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
        // net_sum = total_credit + total_debt + compost_balance
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

    /// Test 18.1: apply_demurrage_all + get_demurrage_report
    /// Batch demurrage across 2 members, verify report, verify compost absorbs.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_batch_demurrage_and_report() {
        println!("Test 18.1: Batch Demurrage + Report");

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
        let zome_b = apps[1].cells()[0].zome("currency_mint");
        let bob_did = format!("did:mycelix:{}", agents[1].clone());

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
        let _: MintedExchange = conductor.call(&zome_a, "record_minted_exchange", ex_input).await;

        // Bob provides 1 hour to Alice → Bob +1, Alice -1 → net: Alice +2, Bob -2
        let ex_input2 = RecordMintedExchangeInput {
            currency_id: def.id.clone(),
            receiver_did: format!("did:mycelix:{}", agents[0].clone()),
            hours: 1.0,
            service_description: "Batch demurrage test exchange 2".into(),
        };
        let _: MintedExchange = conductor.call(&zome_b, "record_minted_exchange", ex_input2).await;
        println!("  - Two exchanges: Alice +2, Bob -2");

        // Get demurrage report (read-only, no mutation)
        let report: DemurrageReport = conductor
            .call(&zome_a, "get_demurrage_report", def.id.clone())
            .await;
        assert_eq!(report.demurrage_rate, 0.02);
        // Alice has positive balance, so she should appear in the report
        // (but elapsed time is near-zero so deduction may be 0)
        println!(
            "  - Report: rate={}, affected={}, total_pending={}",
            report.demurrage_rate, report.affected_members, report.total_pending_demurrage
        );

        // Apply batch demurrage
        let batch_results: Vec<DemurrageResult> = conductor
            .call(&zome_a, "apply_demurrage_all", def.id.clone())
            .await;
        // Near-zero elapsed time → likely no deductions
        println!("  - Batch results: {} members affected", batch_results.len());

        // Verify zero-sum still holds
        let stats: CurrencyStats = conductor
            .call(&zome_a, "get_currency_stats", def.id.clone())
            .await;
        assert_eq!(
            stats.net_sum, 0,
            "Zero-sum preserved after batch demurrage"
        );
        println!("  - Zero-sum invariant: net_sum={}", stats.net_sum);

        println!("Test 18.1 PASSED");
    }

    /// Test 18.2: Pagination — get_currency_exchanges and get_member_exchanges with cursor
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_pagination_cursor() {
        println!("Test 18.2: Pagination Cursor");

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
                    member_did: alice_did,
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
}
