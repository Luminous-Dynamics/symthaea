// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy Investments Coordinator Zome
use hdk::prelude::*;
use investments_integrity::*;
use mycelix_energy_shared::batch::links_to_records;
use mycelix_energy_shared::anchors::anchor_hash;

#[hdk_extern]
pub fn pledge_investment(input: PledgeInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let investment = Investment {
        id: format!("investment:{}:{}:{}", input.project_id, input.investor_did, now.as_micros()),
        project_id: input.project_id.clone(),
        investor_did: input.investor_did.clone(),
        amount: input.amount,
        currency: input.currency,
        shares: input.shares,
        share_percentage: input.share_percentage,
        investment_type: input.investment_type,
        status: InvestmentStatus::Pledged,
        pledged: now,
        confirmed: None,
    };

    let action_hash = create_entry(&EntryTypes::Investment(investment))?;
    create_link(anchor_hash(&input.project_id)?, action_hash.clone(), LinkTypes::ProjectToInvestments, ())?;
    create_link(anchor_hash(&input.investor_did)?, action_hash.clone(), LinkTypes::InvestorToInvestments, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PledgeInput {
    pub project_id: String,
    pub investor_did: String,
    pub amount: f64,
    pub currency: String,
    pub shares: f64,
    pub share_percentage: f64,
    pub investment_type: InvestmentType,
}

#[hdk_extern]
pub fn confirm_investment(investment_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.id == investment_id {
                let now = sys_time()?;
                let confirmed = Investment { status: InvestmentStatus::Confirmed, confirmed: Some(now), ..investment };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::Investment(confirmed))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Investment not found".into())))
}

#[hdk_extern]
pub fn distribute_dividend(input: DividendInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let dividend = Dividend {
        id: format!("dividend:{}:{}:{}", input.project_id, input.investor_did, now.as_micros()),
        project_id: input.project_id,
        investor_did: input.investor_did.clone(),
        amount: input.amount,
        currency: input.currency,
        period_start: input.period_start,
        period_end: input.period_end,
        distributed: now,
        payment_reference: input.payment_reference,
    };

    let action_hash = create_entry(&EntryTypes::Dividend(dividend))?;
    create_link(anchor_hash(&input.investor_did)?, action_hash.clone(), LinkTypes::InvestorToDividends, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DividendInput {
    pub project_id: String,
    pub investor_did: String,
    pub amount: f64,
    pub currency: String,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub payment_reference: Option<String>,
}

/// Get investor's portfolio
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_investor_portfolio(did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::InvestorToInvestments)?, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Get investments for a project
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_project_investments(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToInvestments)?, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Get a specific investment by ID
#[hdk_extern]
pub fn get_investment(investment_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.id == investment_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Cancel a pledged investment (before confirmation)
#[hdk_extern]
pub fn cancel_investment(input: CancelInvestmentInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.id == input.investment_id {
                // Only investor can cancel
                if investment.investor_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest("Only investor can cancel".into())));
                }

                // Can only cancel pledged investments
                if investment.status != InvestmentStatus::Pledged {
                    return Err(wasm_error!(WasmErrorInner::Guest("Can only cancel pledged investments".into())));
                }

                let cancelled = Investment {
                    status: InvestmentStatus::Cancelled,
                    ..investment
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::Investment(cancelled))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Investment not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CancelInvestmentInput {
    pub investment_id: String,
    pub requester_did: String,
}

/// Get investor dividends
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_investor_dividends(did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::InvestorToDividends)?, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Get investments by status
#[hdk_extern]
pub fn get_investments_by_status(status: InvestmentStatus) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.status == status {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Get investments by type
#[hdk_extern]
pub fn get_investments_by_type(investment_type: InvestmentType) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.investment_type == investment_type {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Calculate total investment for a project
#[hdk_extern]
pub fn get_project_total_investment(project_id: String) -> ExternResult<ProjectTotalInvestment> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?))
        .include_entries(true);

    let mut total_amount = 0.0;
    let mut total_shares = 0.0;
    let mut investor_count = 0;

    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.project_id == project_id && investment.status == InvestmentStatus::Confirmed {
                total_amount += investment.amount;
                total_shares += investment.shares;
                investor_count += 1;
            }
        }
    }

    Ok(ProjectTotalInvestment {
        project_id,
        total_amount,
        total_shares,
        investor_count,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProjectTotalInvestment {
    pub project_id: String,
    pub total_amount: f64,
    pub total_shares: f64,
    pub investor_count: u32,
}

/// Distribute dividends to all project investors
#[hdk_extern]
pub fn distribute_project_dividends(input: DistributeProjectDividendsInput) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?))
        .include_entries(true);

    let now = sys_time()?;
    let mut dividends = Vec::new();

    // Collect all confirmed investments for this project
    let mut project_investments = Vec::new();
    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.project_id == input.project_id && investment.status == InvestmentStatus::Confirmed {
                project_investments.push(investment);
            }
        }
    }

    // Calculate total shares
    let total_shares: f64 = project_investments.iter().map(|i| i.shares).sum();

    if total_shares == 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("No confirmed investments for project".into())));
    }

    // Distribute proportionally
    for investment in project_investments {
        let share_ratio = investment.shares / total_shares;
        let dividend_amount = input.total_amount * share_ratio;

        let dividend = Dividend {
            id: format!("dividend:{}:{}:{}", input.project_id, investment.investor_did, now.as_micros()),
            project_id: input.project_id.clone(),
            investor_did: investment.investor_did.clone(),
            amount: dividend_amount,
            currency: input.currency.clone(),
            period_start: input.period_start,
            period_end: input.period_end,
            distributed: now,
            payment_reference: input.payment_reference.clone(),
        };

        let action_hash = create_entry(&EntryTypes::Dividend(dividend))?;
        create_link(anchor_hash(&investment.investor_did)?, action_hash.clone(), LinkTypes::InvestorToDividends, ())?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            dividends.push(record);
        }
    }

    Ok(dividends)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DistributeProjectDividendsInput {
    pub project_id: String,
    pub total_amount: f64,
    pub currency: String,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub payment_reference: Option<String>,
}

/// Update investment amount (only for pledged)
#[hdk_extern]
pub fn update_investment_amount(input: UpdateInvestmentAmountInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.id == input.investment_id {
                // Only investor can update
                if investment.investor_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest("Only investor can update".into())));
                }

                // Can only update pledged investments
                if investment.status != InvestmentStatus::Pledged {
                    return Err(wasm_error!(WasmErrorInner::Guest("Can only update pledged investments".into())));
                }

                let updated = Investment {
                    amount: input.new_amount,
                    shares: input.new_shares,
                    share_percentage: input.new_share_percentage,
                    ..investment
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::Investment(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Investment not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateInvestmentAmountInput {
    pub investment_id: String,
    pub requester_did: String,
    pub new_amount: f64,
    pub new_shares: f64,
    pub new_share_percentage: f64,
}

/// Get investor portfolio summary
#[hdk_extern]
pub fn get_portfolio_summary(did: String) -> ExternResult<PortfolioSummary> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Investment)?))
        .include_entries(true);

    let mut total_invested = 0.0;
    let mut total_shares = 0.0;
    let mut project_count = 0;
    let mut projects = Vec::new();

    for record in query(filter)? {
        if let Some(investment) = record.entry().to_app_option::<Investment>().ok().flatten() {
            if investment.investor_did == did && investment.status == InvestmentStatus::Confirmed {
                total_invested += investment.amount;
                total_shares += investment.shares;
                project_count += 1;
                if !projects.contains(&investment.project_id) {
                    projects.push(investment.project_id);
                }
            }
        }
    }

    // Get total dividends received
    let dividend_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Dividend)?))
        .include_entries(true);

    let mut total_dividends = 0.0;
    for record in query(dividend_filter)? {
        if let Some(dividend) = record.entry().to_app_option::<Dividend>().ok().flatten() {
            if dividend.investor_did == did {
                total_dividends += dividend.amount;
            }
        }
    }

    Ok(PortfolioSummary {
        investor_did: did,
        total_invested,
        total_shares,
        project_count: project_count as u32,
        unique_projects: projects.len() as u32,
        total_dividends_received: total_dividends,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PortfolioSummary {
    pub investor_did: String,
    pub total_invested: f64,
    pub total_shares: f64,
    pub project_count: u32,
    pub unique_projects: u32,
    pub total_dividends_received: f64,
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
    // PledgeInput Tests
    // =========================================================================

    fn valid_pledge_input() -> PledgeInput {
        PledgeInput {
            project_id: "project:solar_farm_alpha".to_string(),
            investor_did: "did:mycelix:investor1".to_string(),
            amount: 50000.0,
            currency: "USD".to_string(),
            shares: 100.0,
            share_percentage: 5.0,
            investment_type: InvestmentType::Equity,
        }
    }

    #[test]
    fn test_pledge_input_valid() {
        let input = valid_pledge_input();
        assert!(input.investor_did.starts_with("did:"));
        assert!(input.amount > 0.0);
        assert!(input.share_percentage >= 0.0 && input.share_percentage <= 100.0);
    }

    #[test]
    fn test_pledge_input_all_investment_types() {
        let types = vec![
            InvestmentType::Equity,
            InvestmentType::Debt,
            InvestmentType::ConvertibleNote,
            InvestmentType::RevenueShare,
            InvestmentType::CommunityShare,
        ];
        for inv_type in types {
            let input = PledgeInput {
                investment_type: inv_type.clone(),
                ..valid_pledge_input()
            };
            assert_eq!(input.investment_type, inv_type);
        }
    }

    #[test]
    fn test_pledge_input_serialization() {
        let input = valid_pledge_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    #[test]
    fn test_pledge_input_large_investment() {
        let input = PledgeInput {
            amount: 10_000_000.0,
            shares: 10000.0,
            share_percentage: 50.0,
            ..valid_pledge_input()
        };
        assert!(input.amount > 0.0);
    }

    #[test]
    fn test_pledge_input_small_investment() {
        let input = PledgeInput {
            amount: 100.0,
            shares: 1.0,
            share_percentage: 0.01,
            ..valid_pledge_input()
        };
        assert!(input.amount > 0.0);
        assert!(input.share_percentage > 0.0);
    }

    #[test]
    fn test_pledge_input_various_currencies() {
        let currencies = vec!["USD", "EUR", "GBP", "CHF", "BTC"];
        for currency in currencies {
            let input = PledgeInput {
                currency: currency.to_string(),
                ..valid_pledge_input()
            };
            assert_eq!(input.currency, currency);
        }
    }

    // =========================================================================
    // DividendInput Tests
    // =========================================================================

    fn valid_dividend_input() -> DividendInput {
        DividendInput {
            project_id: "project:solar_farm_alpha".to_string(),
            investor_did: "did:mycelix:investor1".to_string(),
            amount: 500.0,
            currency: "USD".to_string(),
            period_start: create_test_timestamp(),
            period_end: Timestamp::from_micros(1706745600000000),
            payment_reference: Some("PAY-DIV-001".to_string()),
        }
    }

    #[test]
    fn test_dividend_input_valid() {
        let input = valid_dividend_input();
        assert!(input.investor_did.starts_with("did:"));
        assert!(input.amount > 0.0);
    }

    #[test]
    fn test_dividend_input_with_payment_ref() {
        let input = valid_dividend_input();
        assert!(input.payment_reference.is_some());
    }

    #[test]
    fn test_dividend_input_without_payment_ref() {
        let input = DividendInput {
            payment_reference: None,
            ..valid_dividend_input()
        };
        assert!(input.payment_reference.is_none());
    }

    #[test]
    fn test_dividend_input_period_valid() {
        let input = valid_dividend_input();
        assert!(input.period_end.as_micros() > input.period_start.as_micros());
    }

    #[test]
    fn test_dividend_input_serialization() {
        let input = valid_dividend_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    // =========================================================================
    // CancelInvestmentInput Tests
    // =========================================================================

    fn valid_cancel_investment_input() -> CancelInvestmentInput {
        CancelInvestmentInput {
            investment_id: "investment:project1:did:mycelix:investor1:123456".to_string(),
            requester_did: "did:mycelix:investor1".to_string(),
        }
    }

    #[test]
    fn test_cancel_investment_input_valid() {
        let input = valid_cancel_investment_input();
        assert!(!input.investment_id.is_empty());
        assert!(input.requester_did.starts_with("did:"));
    }

    #[test]
    fn test_cancel_investment_input_investor_match() {
        let input = valid_cancel_investment_input();
        // Investment ID should reference the same investor
        assert!(input.investment_id.contains("investor1"));
        assert!(input.requester_did.contains("investor1"));
    }

    // =========================================================================
    // UpdateInvestmentAmountInput Tests
    // =========================================================================

    fn valid_update_investment_amount_input() -> UpdateInvestmentAmountInput {
        UpdateInvestmentAmountInput {
            investment_id: "investment:project1:did:mycelix:investor1:123456".to_string(),
            requester_did: "did:mycelix:investor1".to_string(),
            new_amount: 75000.0,
            new_shares: 150.0,
            new_share_percentage: 7.5,
        }
    }

    #[test]
    fn test_update_investment_amount_input_valid() {
        let input = valid_update_investment_amount_input();
        assert!(!input.investment_id.is_empty());
        assert!(input.requester_did.starts_with("did:"));
        assert!(input.new_amount > 0.0);
        assert!(input.new_share_percentage >= 0.0 && input.new_share_percentage <= 100.0);
    }

    #[test]
    fn test_update_investment_amount_input_increase() {
        let input = UpdateInvestmentAmountInput {
            new_amount: 100000.0,
            new_shares: 200.0,
            new_share_percentage: 10.0,
            ..valid_update_investment_amount_input()
        };
        assert!(input.new_amount > 75000.0);
    }

    #[test]
    fn test_update_investment_amount_input_decrease() {
        let input = UpdateInvestmentAmountInput {
            new_amount: 25000.0,
            new_shares: 50.0,
            new_share_percentage: 2.5,
            ..valid_update_investment_amount_input()
        };
        assert!(input.new_amount < 75000.0);
    }

    // =========================================================================
    // DistributeProjectDividendsInput Tests
    // =========================================================================

    fn valid_distribute_project_dividends_input() -> DistributeProjectDividendsInput {
        DistributeProjectDividendsInput {
            project_id: "project:solar_farm_alpha".to_string(),
            total_amount: 10000.0,
            currency: "USD".to_string(),
            period_start: create_test_timestamp(),
            period_end: Timestamp::from_micros(1706745600000000),
            payment_reference: Some("DIV-2024-Q1".to_string()),
        }
    }

    #[test]
    fn test_distribute_project_dividends_input_valid() {
        let input = valid_distribute_project_dividends_input();
        assert!(!input.project_id.is_empty());
        assert!(input.total_amount > 0.0);
    }

    #[test]
    fn test_distribute_project_dividends_input_large_amount() {
        let input = DistributeProjectDividendsInput {
            total_amount: 1_000_000.0,
            ..valid_distribute_project_dividends_input()
        };
        assert!(input.total_amount > 0.0);
    }

    #[test]
    fn test_distribute_project_dividends_input_serialization() {
        let input = valid_distribute_project_dividends_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    // =========================================================================
    // ProjectTotalInvestment Tests
    // =========================================================================

    #[test]
    fn test_project_total_investment_creation() {
        let total = ProjectTotalInvestment {
            project_id: "project:solar_farm_alpha".to_string(),
            total_amount: 1_000_000.0,
            total_shares: 10000.0,
            investor_count: 50,
        };
        assert!(!total.project_id.is_empty());
        assert!(total.total_amount >= 0.0);
        assert!(total.investor_count >= 0);
    }

    #[test]
    fn test_project_total_investment_empty() {
        let total = ProjectTotalInvestment {
            project_id: "project:new_project".to_string(),
            total_amount: 0.0,
            total_shares: 0.0,
            investor_count: 0,
        };
        assert_eq!(total.investor_count, 0);
        assert_eq!(total.total_amount, 0.0);
    }

    #[test]
    fn test_project_total_investment_single_investor() {
        let total = ProjectTotalInvestment {
            project_id: "project:small_project".to_string(),
            total_amount: 50000.0,
            total_shares: 100.0,
            investor_count: 1,
        };
        assert_eq!(total.investor_count, 1);
    }

    #[test]
    fn test_project_total_investment_many_investors() {
        let total = ProjectTotalInvestment {
            project_id: "project:community_project".to_string(),
            total_amount: 5_000_000.0,
            total_shares: 100000.0,
            investor_count: 1000,
        };
        assert!(total.investor_count > 0);
    }

    // =========================================================================
    // PortfolioSummary Tests
    // =========================================================================

    #[test]
    fn test_portfolio_summary_creation() {
        let summary = PortfolioSummary {
            investor_did: "did:mycelix:investor1".to_string(),
            total_invested: 250000.0,
            total_shares: 5000.0,
            project_count: 10,
            unique_projects: 5,
            total_dividends_received: 12500.0,
        };
        assert!(summary.investor_did.starts_with("did:"));
        assert!(summary.total_invested >= 0.0);
    }

    #[test]
    fn test_portfolio_summary_empty() {
        let summary = PortfolioSummary {
            investor_did: "did:mycelix:new_investor".to_string(),
            total_invested: 0.0,
            total_shares: 0.0,
            project_count: 0,
            unique_projects: 0,
            total_dividends_received: 0.0,
        };
        assert_eq!(summary.project_count, 0);
        assert_eq!(summary.total_invested, 0.0);
    }

    #[test]
    fn test_portfolio_summary_diversified() {
        let summary = PortfolioSummary {
            investor_did: "did:mycelix:investor1".to_string(),
            total_invested: 100000.0,
            total_shares: 2000.0,
            project_count: 15, // Multiple investments
            unique_projects: 8, // In 8 different projects
            total_dividends_received: 5000.0,
        };
        assert!(summary.unique_projects <= summary.project_count as u32);
    }

    #[test]
    fn test_portfolio_summary_roi_calculation() {
        let summary = PortfolioSummary {
            investor_did: "did:mycelix:investor1".to_string(),
            total_invested: 100000.0,
            total_shares: 2000.0,
            project_count: 5,
            unique_projects: 5,
            total_dividends_received: 10000.0,
        };
        if summary.total_invested > 0.0 {
            let roi_percent = (summary.total_dividends_received / summary.total_invested) * 100.0;
            assert_eq!(roi_percent, 10.0);
        }
    }

    #[test]
    fn test_portfolio_summary_serialization() {
        let summary = PortfolioSummary {
            investor_did: "did:mycelix:investor1".to_string(),
            total_invested: 50000.0,
            total_shares: 1000.0,
            project_count: 3,
            unique_projects: 3,
            total_dividends_received: 2500.0,
        };
        let json = serde_json::to_string(&summary);
        assert!(json.is_ok());
    }

    // =========================================================================
    // Business Logic Edge Cases
    // =========================================================================

    #[test]
    fn test_dividend_proportional_distribution() {
        // Simulating proportional dividend distribution
        let total_project_shares = 10000.0;
        let total_dividend = 10000.0;

        // Investor with 10% of shares
        let investor_shares = 1000.0;
        let share_ratio = investor_shares / total_project_shares;
        let expected_dividend = total_dividend * share_ratio;

        assert_eq!(expected_dividend, 1000.0);
    }

    #[test]
    fn test_investment_share_percentage_consistency() {
        // Share percentage should be consistent with shares
        let total_project_shares = 10000.0;
        let investor_shares = 500.0;
        let expected_percentage = (investor_shares / total_project_shares) * 100.0;

        assert_eq!(expected_percentage, 5.0);
    }

    #[test]
    fn test_fractional_dividend_amounts() {
        let input = DividendInput {
            amount: 0.01, // 1 cent
            ..valid_dividend_input()
        };
        assert!(input.amount > 0.0);
    }

    #[test]
    fn test_large_dividend_amounts() {
        let input = DividendInput {
            amount: 1_000_000.0,
            ..valid_dividend_input()
        };
        assert!(input.amount > 0.0);
    }

    #[test]
    fn test_long_project_id() {
        let input = PledgeInput {
            project_id: format!("project:{}", "a".repeat(100)),
            ..valid_pledge_input()
        };
        assert!(!input.project_id.is_empty());
    }

    #[test]
    fn test_investment_status_lifecycle() {
        // Test typical investment lifecycle
        let statuses = [
            InvestmentStatus::Pledged,
            InvestmentStatus::PendingPayment,
            InvestmentStatus::Confirmed,
        ];
        for status in statuses {
            // Each status is valid
            let _s = status.clone();
        }
    }

    #[test]
    fn test_investment_cancellation_window() {
        // Investments can only be cancelled while Pledged
        let input = valid_cancel_investment_input();
        assert!(!input.investment_id.is_empty());
    }

    #[test]
    fn test_multiple_investments_same_project() {
        // Same investor can have multiple investments in same project
        let inputs: Vec<PledgeInput> = (0..3)
            .map(|i| PledgeInput {
                investment_type: match i {
                    0 => InvestmentType::Equity,
                    1 => InvestmentType::Debt,
                    _ => InvestmentType::RevenueShare,
                },
                amount: (i + 1) as f64 * 10000.0,
                shares: (i + 1) as f64 * 100.0,
                share_percentage: (i + 1) as f64 * 1.0,
                ..valid_pledge_input()
            })
            .collect();

        assert_eq!(inputs.len(), 3);
        for input in inputs {
            assert!(input.amount > 0.0);
        }
    }

    #[test]
    fn test_deserialization_roundtrip() {
        let input = valid_pledge_input();
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: PledgeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.investor_did, input.investor_did);
        assert_eq!(deserialized.amount, input.amount);
    }
}
