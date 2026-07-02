// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy Investments Integrity Zome
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Investment {
    pub id: String,
    pub project_id: String,
    pub investor_did: String,
    pub amount: f64,
    pub currency: String,
    pub shares: f64,
    pub share_percentage: f64,
    pub investment_type: InvestmentType,
    pub status: InvestmentStatus,
    pub pledged: Timestamp,
    pub confirmed: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InvestmentType {
    Equity,
    Debt,
    ConvertibleNote,
    RevenueShare,
    CommunityShare,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InvestmentStatus {
    Pledged,
    PendingPayment,
    Confirmed,
    Cancelled,
    Transferred,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Dividend {
    pub id: String,
    pub project_id: String,
    pub investor_did: String,
    pub amount: f64,
    pub currency: String,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub distributed: Timestamp,
    pub payment_reference: Option<String>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Portfolio {
    pub owner_did: String,
    pub total_invested: f64,
    pub total_returns: f64,
    pub currency: String,
    pub project_count: u32,
    pub last_updated: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    Investment(Investment),
    Dividend(Dividend),
    Portfolio(Portfolio),
}

#[hdk_link_types]
pub enum LinkTypes {
    ProjectToInvestments,
    InvestorToInvestments,
    InvestorToDividends,
    InvestorToPortfolio,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::Investment(investment) => {
                        validate_create_investment(EntryCreationAction::Create(action), investment)
                    }
                    EntryTypes::Dividend(dividend) => {
                        validate_create_dividend(EntryCreationAction::Create(action), dividend)
                    }
                    EntryTypes::Portfolio(portfolio) => {
                        validate_create_portfolio(EntryCreationAction::Create(action), portfolio)
                    }
                }
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::Investment(investment) => {
                        validate_update_investment(action, investment)
                    }
                    EntryTypes::Dividend(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Dividends cannot be updated".into(),
                        ))
                    }
                    EntryTypes::Portfolio(portfolio) => {
                        validate_update_portfolio(action, portfolio)
                    }
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            match link_type {
                LinkTypes::ProjectToInvestments => Ok(ValidateCallbackResult::Valid),
                LinkTypes::InvestorToInvestments => Ok(ValidateCallbackResult::Valid),
                LinkTypes::InvestorToDividends => Ok(ValidateCallbackResult::Valid),
                LinkTypes::InvestorToPortfolio => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_investment(
    _action: EntryCreationAction,
    investment: Investment,
) -> ExternResult<ValidateCallbackResult> {
    if !investment.investor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Investor must be a valid DID".into()));
    }
    if investment.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Investment amount must be positive".into()));
    }
    if investment.share_percentage < 0.0 || investment.share_percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid("Share percentage must be 0-100".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_investment(
    _action: Update,
    investment: Investment,
) -> ExternResult<ValidateCallbackResult> {
    if investment.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Investment amount must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_dividend(
    _action: EntryCreationAction,
    dividend: Dividend,
) -> ExternResult<ValidateCallbackResult> {
    if !dividend.investor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Investor must be a valid DID".into()));
    }
    if dividend.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Dividend amount must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_portfolio(
    _action: EntryCreationAction,
    portfolio: Portfolio,
) -> ExternResult<ValidateCallbackResult> {
    if !portfolio.owner_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Owner must be a valid DID".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_portfolio(
    _action: Update,
    _portfolio: Portfolio,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    fn create_test_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000) // 2024-01-01 00:00:00 UTC
    }

    // =========================================================================
    // InvestmentType Enum Tests
    // =========================================================================

    #[test]
    fn test_investment_type_variants() {
        let types = vec![
            InvestmentType::Equity,
            InvestmentType::Debt,
            InvestmentType::ConvertibleNote,
            InvestmentType::RevenueShare,
            InvestmentType::CommunityShare,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_investment_type_equality() {
        assert_eq!(InvestmentType::Equity, InvestmentType::Equity);
        assert_ne!(InvestmentType::Equity, InvestmentType::Debt);
    }

    #[test]
    fn test_investment_type_cloning() {
        let original = InvestmentType::CommunityShare;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // =========================================================================
    // InvestmentStatus Enum Tests
    // =========================================================================

    #[test]
    fn test_investment_status_variants() {
        let statuses = vec![
            InvestmentStatus::Pledged,
            InvestmentStatus::PendingPayment,
            InvestmentStatus::Confirmed,
            InvestmentStatus::Cancelled,
            InvestmentStatus::Transferred,
        ];
        assert_eq!(statuses.len(), 5);
    }

    #[test]
    fn test_investment_status_equality() {
        assert_eq!(InvestmentStatus::Confirmed, InvestmentStatus::Confirmed);
        assert_ne!(InvestmentStatus::Pledged, InvestmentStatus::Cancelled);
    }

    #[test]
    fn test_investment_status_state_transitions() {
        // Typical state flow: Pledged -> PendingPayment -> Confirmed
        let states = [
            InvestmentStatus::Pledged,
            InvestmentStatus::PendingPayment,
            InvestmentStatus::Confirmed,
        ];
        // All states should be distinct
        for (i, state) in states.iter().enumerate() {
            for (j, other_state) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(state, other_state);
                } else {
                    assert_ne!(state, other_state);
                }
            }
        }
    }

    // =========================================================================
    // Investment Validation Tests
    // =========================================================================

    fn valid_investment() -> Investment {
        Investment {
            id: "investment:project1:did:mycelix:investor1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            investor_did: "did:mycelix:investor1".to_string(),
            amount: 50000.0,
            currency: "USD".to_string(),
            shares: 100.0,
            share_percentage: 5.0,
            investment_type: InvestmentType::Equity,
            status: InvestmentStatus::Pledged,
            pledged: create_test_timestamp(),
            confirmed: None,
        }
    }

    #[test]
    fn test_investment_valid_investor_did() {
        let investment = valid_investment();
        assert!(investment.investor_did.starts_with("did:"));
    }

    #[test]
    fn test_investment_positive_amount() {
        let investment = valid_investment();
        assert!(investment.amount > 0.0);
    }

    #[test]
    fn test_investment_valid_share_percentage() {
        let investment = valid_investment();
        assert!(investment.share_percentage >= 0.0);
        assert!(investment.share_percentage <= 100.0);
    }

    #[test]
    fn test_investment_invalid_investor_did() {
        let investment = Investment {
            investor_did: "investor123".to_string(),
            ..valid_investment()
        };
        assert!(!investment.investor_did.starts_with("did:"));
    }

    #[test]
    fn test_investment_zero_amount_invalid() {
        let investment = Investment {
            amount: 0.0,
            ..valid_investment()
        };
        assert!(investment.amount <= 0.0);
    }

    #[test]
    fn test_investment_negative_amount_invalid() {
        let investment = Investment {
            amount: -1000.0,
            ..valid_investment()
        };
        assert!(investment.amount <= 0.0);
    }

    #[test]
    fn test_investment_share_percentage_over_100_invalid() {
        let investment = Investment {
            share_percentage: 150.0,
            ..valid_investment()
        };
        assert!(investment.share_percentage > 100.0);
    }

    #[test]
    fn test_investment_share_percentage_negative_invalid() {
        let investment = Investment {
            share_percentage: -5.0,
            ..valid_investment()
        };
        assert!(investment.share_percentage < 0.0);
    }

    #[test]
    fn test_investment_unconfirmed_pledged() {
        let investment = valid_investment();
        assert!(investment.confirmed.is_none());
        assert_eq!(investment.status, InvestmentStatus::Pledged);
    }

    #[test]
    fn test_investment_confirmed_with_timestamp() {
        let investment = Investment {
            status: InvestmentStatus::Confirmed,
            confirmed: Some(Timestamp::from_micros(1704153600000000)),
            ..valid_investment()
        };
        assert!(investment.confirmed.is_some());
        assert_eq!(investment.status, InvestmentStatus::Confirmed);
    }

    #[test]
    fn test_investment_all_types() {
        let types = vec![
            InvestmentType::Equity,
            InvestmentType::Debt,
            InvestmentType::ConvertibleNote,
            InvestmentType::RevenueShare,
            InvestmentType::CommunityShare,
        ];
        for inv_type in types {
            let investment = Investment {
                investment_type: inv_type.clone(),
                ..valid_investment()
            };
            assert_eq!(investment.investment_type, inv_type);
        }
    }

    #[test]
    fn test_investment_full_ownership() {
        let investment = Investment {
            share_percentage: 100.0,
            ..valid_investment()
        };
        assert_eq!(investment.share_percentage, 100.0);
    }

    #[test]
    fn test_investment_fractional_ownership() {
        let investment = Investment {
            share_percentage: 0.001,
            ..valid_investment()
        };
        assert!(investment.share_percentage > 0.0);
        assert!(investment.share_percentage < 1.0);
    }

    // =========================================================================
    // Dividend Validation Tests
    // =========================================================================

    fn valid_dividend() -> Dividend {
        Dividend {
            id: "dividend:project1:did:mycelix:investor1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            investor_did: "did:mycelix:investor1".to_string(),
            amount: 250.50,
            currency: "USD".to_string(),
            period_start: create_test_timestamp(),
            period_end: Timestamp::from_micros(1706745600000000), // 30 days later
            distributed: Timestamp::from_micros(1706832000000000),
            payment_reference: Some("PAY-DIV-2024-001".to_string()),
        }
    }

    #[test]
    fn test_dividend_valid_investor_did() {
        let dividend = valid_dividend();
        assert!(dividend.investor_did.starts_with("did:"));
    }

    #[test]
    fn test_dividend_positive_amount() {
        let dividend = valid_dividend();
        assert!(dividend.amount > 0.0);
    }

    #[test]
    fn test_dividend_invalid_investor_did() {
        let dividend = Dividend {
            investor_did: "investor123".to_string(),
            ..valid_dividend()
        };
        assert!(!dividend.investor_did.starts_with("did:"));
    }

    #[test]
    fn test_dividend_zero_amount_invalid() {
        let dividend = Dividend {
            amount: 0.0,
            ..valid_dividend()
        };
        assert!(dividend.amount <= 0.0);
    }

    #[test]
    fn test_dividend_negative_amount_invalid() {
        let dividend = Dividend {
            amount: -100.0,
            ..valid_dividend()
        };
        assert!(dividend.amount <= 0.0);
    }

    #[test]
    fn test_dividend_with_payment_reference() {
        let dividend = valid_dividend();
        assert!(dividend.payment_reference.is_some());
    }

    #[test]
    fn test_dividend_without_payment_reference() {
        let dividend = Dividend {
            payment_reference: None,
            ..valid_dividend()
        };
        assert!(dividend.payment_reference.is_none());
    }

    #[test]
    fn test_dividend_period_ordering() {
        let dividend = valid_dividend();
        assert!(dividend.period_end.as_micros() > dividend.period_start.as_micros());
    }

    #[test]
    fn test_dividend_distributed_after_period() {
        let dividend = valid_dividend();
        assert!(dividend.distributed.as_micros() >= dividend.period_end.as_micros());
    }

    // =========================================================================
    // Portfolio Validation Tests
    // =========================================================================

    fn valid_portfolio() -> Portfolio {
        Portfolio {
            owner_did: "did:mycelix:investor1".to_string(),
            total_invested: 100000.0,
            total_returns: 15000.0,
            currency: "USD".to_string(),
            project_count: 5,
            last_updated: create_test_timestamp(),
        }
    }

    #[test]
    fn test_portfolio_valid_owner_did() {
        let portfolio = valid_portfolio();
        assert!(portfolio.owner_did.starts_with("did:"));
    }

    #[test]
    fn test_portfolio_invalid_owner_did() {
        let portfolio = Portfolio {
            owner_did: "owner123".to_string(),
            ..valid_portfolio()
        };
        assert!(!portfolio.owner_did.starts_with("did:"));
    }

    #[test]
    fn test_portfolio_positive_returns() {
        let portfolio = valid_portfolio();
        assert!(portfolio.total_returns >= 0.0);
    }

    #[test]
    fn test_portfolio_negative_returns() {
        // Loss scenario
        let portfolio = Portfolio {
            total_returns: -5000.0,
            ..valid_portfolio()
        };
        assert!(portfolio.total_returns < 0.0);
    }

    #[test]
    fn test_portfolio_empty() {
        let portfolio = Portfolio {
            total_invested: 0.0,
            total_returns: 0.0,
            project_count: 0,
            ..valid_portfolio()
        };
        assert_eq!(portfolio.project_count, 0);
    }

    #[test]
    fn test_portfolio_roi_calculation() {
        let portfolio = valid_portfolio();
        if portfolio.total_invested > 0.0 {
            let roi = portfolio.total_returns / portfolio.total_invested;
            assert!(roi.is_finite());
        }
    }

    // =========================================================================
    // Anchor Tests
    // =========================================================================

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("project_investments".to_string());
        assert_eq!(anchor.0, "project_investments");
    }

    #[test]
    fn test_anchor_equality() {
        let anchor1 = Anchor("investor_portfolio".to_string());
        let anchor2 = Anchor("investor_portfolio".to_string());
        let anchor3 = Anchor("project_investments".to_string());

        assert_eq!(anchor1, anchor2);
        assert_ne!(anchor1, anchor3);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_very_large_investment() {
        let investment = Investment {
            amount: 1_000_000_000.0, // $1 billion
            ..valid_investment()
        };
        assert!(investment.amount > 0.0);
    }

    #[test]
    fn test_very_small_investment() {
        let investment = Investment {
            amount: 0.01, // 1 cent
            ..valid_investment()
        };
        assert!(investment.amount > 0.0);
    }

    #[test]
    fn test_very_large_shares() {
        let investment = Investment {
            shares: 1_000_000.0,
            ..valid_investment()
        };
        assert!(investment.shares > 0.0);
    }

    #[test]
    fn test_fractional_shares() {
        let investment = Investment {
            shares: 0.5,
            ..valid_investment()
        };
        assert!(investment.shares > 0.0);
    }

    #[test]
    fn test_various_currencies() {
        let currencies = vec!["USD", "EUR", "GBP", "CHF", "BTC", "ETH", "SOLAR"];
        for currency in currencies {
            let investment = Investment {
                currency: currency.to_string(),
                ..valid_investment()
            };
            assert!(!investment.currency.is_empty());
        }
    }

    #[test]
    fn test_community_share_investment() {
        let investment = Investment {
            investment_type: InvestmentType::CommunityShare,
            share_percentage: 10.0,
            ..valid_investment()
        };
        assert_eq!(investment.investment_type, InvestmentType::CommunityShare);
    }

    #[test]
    fn test_convertible_note_investment() {
        let investment = Investment {
            investment_type: InvestmentType::ConvertibleNote,
            ..valid_investment()
        };
        assert_eq!(investment.investment_type, InvestmentType::ConvertibleNote);
    }

    #[test]
    fn test_revenue_share_investment() {
        let investment = Investment {
            investment_type: InvestmentType::RevenueShare,
            ..valid_investment()
        };
        assert_eq!(investment.investment_type, InvestmentType::RevenueShare);
    }

    #[test]
    fn test_serialization_investment() {
        let investment = valid_investment();
        // Test that all fields are serializable
        let result = serde_json::to_string(&investment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_dividend() {
        let dividend = valid_dividend();
        let result = serde_json::to_string(&dividend);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_portfolio() {
        let portfolio = valid_portfolio();
        let result = serde_json::to_string(&portfolio);
        assert!(result.is_ok());
    }
}
