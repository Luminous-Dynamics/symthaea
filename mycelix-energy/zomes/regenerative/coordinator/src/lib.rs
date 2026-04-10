// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regenerative Exit Coordinator Zome
use hdk::prelude::*;
use regenerative_integrity::*;

/// Create or retrieve an anchor entry hash for deterministic link bases
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn create_regenerative_contract(input: CreateContractInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let contract = RegenerativeContract {
        id: format!("regen:{}:{}", input.project_id, now.as_micros()),
        project_id: input.project_id.clone(),
        community_did: input.community_did.clone(),
        conditions: input.conditions,
        current_ownership_percentage: 0.0,
        target_ownership_percentage: input.target_ownership_percentage,
        reserve_account_balance: 0.0,
        currency: input.currency,
        status: ContractStatus::Active,
        created: now,
        last_assessment: now,
    };

    let action_hash = create_entry(&EntryTypes::RegenerativeContract(contract))?;
    create_link(anchor_hash(&input.project_id)?, action_hash.clone(), LinkTypes::ProjectToContract, ())?;
    create_link(anchor_hash(&input.community_did)?, action_hash.clone(), LinkTypes::CommunityToContract, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateContractInput {
    pub project_id: String,
    pub community_did: String,
    pub conditions: Vec<TransitionCondition>,
    pub target_ownership_percentage: f64,
    pub currency: String,
}

#[hdk_extern]
pub fn submit_readiness_assessment(input: AssessmentInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let overall = input.scores.iter().map(|s| s.score * 1.0).sum::<f64>() / input.scores.len() as f64;

    let assessment = ReadinessAssessment {
        id: format!("assess:{}:{}", input.contract_id, now.as_micros()),
        contract_id: input.contract_id.clone(),
        assessor_did: input.assessor_did,
        scores: input.scores,
        overall_readiness: overall,
        recommendations: input.recommendations,
        assessed: now,
    };

    let action_hash = create_entry(&EntryTypes::ReadinessAssessment(assessment))?;
    create_link(anchor_hash(&input.contract_id)?, action_hash.clone(), LinkTypes::ContractToAssessments, ())?;

    // Update contract with latest assessment
    update_contract_conditions(&input.contract_id, overall)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AssessmentInput {
    pub contract_id: String,
    pub assessor_did: String,
    pub scores: Vec<ConditionScore>,
    pub recommendations: Vec<String>,
}

fn update_contract_conditions(contract_id: &str, _readiness: f64) -> ExternResult<()> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.id == contract_id {
                let now = sys_time()?;
                let updated = RegenerativeContract { last_assessment: now, ..contract };
                update_entry(record.action_address().clone(), &EntryTypes::RegenerativeContract(updated))?;
                return Ok(());
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))
}

#[hdk_extern]
pub fn execute_ownership_transfer(input: TransferInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.id == input.contract_id {
                let now = sys_time()?;

                // Record the transfer
                let transfer = OwnershipTransfer {
                    id: format!("transfer:{}:{}", input.contract_id, now.as_micros()),
                    contract_id: input.contract_id.clone(),
                    from_percentage: contract.current_ownership_percentage,
                    to_percentage: input.new_percentage,
                    shares_transferred: input.shares_transferred,
                    transfer_price: input.transfer_price,
                    currency: contract.currency.clone(),
                    executed: now,
                };
                let transfer_hash = create_entry(&EntryTypes::OwnershipTransfer(transfer))?;
                create_link(anchor_hash(&input.contract_id)?, transfer_hash, LinkTypes::ContractToTransfers, ())?;

                // Update contract
                let new_status = if input.new_percentage >= contract.target_ownership_percentage {
                    ContractStatus::TransitionComplete
                } else {
                    ContractStatus::Active
                };

                let updated = RegenerativeContract {
                    current_ownership_percentage: input.new_percentage,
                    status: new_status,
                    ..contract
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::RegenerativeContract(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferInput {
    pub contract_id: String,
    pub new_percentage: f64,
    pub shares_transferred: f64,
    pub transfer_price: f64,
}

#[hdk_extern]
pub fn get_contract(contract_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.id == contract_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all contracts for a project
#[hdk_extern]
pub fn get_project_contracts(project_id: String) -> ExternResult<Vec<Record>> {
    let mut contracts = Vec::new();
    for link in get_links(LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToContract)?, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            contracts.push(record);
        }
    }
    Ok(contracts)
}

/// Get all contracts for a community
#[hdk_extern]
pub fn get_community_contracts(community_did: String) -> ExternResult<Vec<Record>> {
    let mut contracts = Vec::new();
    for link in get_links(LinkQuery::try_new(anchor_hash(&community_did)?, LinkTypes::CommunityToContract)?, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            contracts.push(record);
        }
    }
    Ok(contracts)
}

/// Get contract assessments
#[hdk_extern]
pub fn get_contract_assessments(contract_id: String) -> ExternResult<Vec<Record>> {
    let mut assessments = Vec::new();
    for link in get_links(LinkQuery::try_new(anchor_hash(&contract_id)?, LinkTypes::ContractToAssessments)?, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            assessments.push(record);
        }
    }
    Ok(assessments)
}

/// Get ownership transfers for a contract
#[hdk_extern]
pub fn get_contract_transfers(contract_id: String) -> ExternResult<Vec<Record>> {
    let mut transfers = Vec::new();
    for link in get_links(LinkQuery::try_new(anchor_hash(&contract_id)?, LinkTypes::ContractToTransfers)?, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            transfers.push(record);
        }
    }
    Ok(transfers)
}

/// Update contract status
#[hdk_extern]
pub fn update_contract_status(input: UpdateContractStatusInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.id == input.contract_id {
                let updated = RegenerativeContract {
                    status: input.new_status,
                    ..contract
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::RegenerativeContract(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateContractStatusInput {
    pub contract_id: String,
    pub new_status: ContractStatus,
}

/// Get contracts by status
#[hdk_extern]
pub fn get_contracts_by_status(status: ContractStatus) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.status == status {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Add to reserve account balance
#[hdk_extern]
pub fn add_to_reserve(input: AddToReserveInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.id == input.contract_id {
                let updated = RegenerativeContract {
                    reserve_account_balance: contract.reserve_account_balance + input.amount,
                    ..contract
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::RegenerativeContract(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddToReserveInput {
    pub contract_id: String,
    pub amount: f64,
}

/// Update contract conditions
#[hdk_extern]
pub fn update_conditions(input: UpdateConditionsInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.id == input.contract_id {
                let updated = RegenerativeContract {
                    conditions: input.conditions,
                    ..contract
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::RegenerativeContract(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateConditionsInput {
    pub contract_id: String,
    pub conditions: Vec<TransitionCondition>,
}

/// Get latest assessment for a contract
#[hdk_extern]
pub fn get_latest_assessment(contract_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::ReadinessAssessment)?))
        .include_entries(true);

    let mut latest: Option<(Timestamp, Record)> = None;
    for record in query(filter)? {
        if let Some(assessment) = record.entry().to_app_option::<ReadinessAssessment>().ok().flatten() {
            if assessment.contract_id == contract_id {
                match &latest {
                    None => latest = Some((assessment.assessed, record)),
                    Some((ts, _)) if assessment.assessed.as_micros() > ts.as_micros() => {
                        latest = Some((assessment.assessed, record));
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(latest.map(|(_, r)| r))
}

/// Calculate readiness progress for a contract
#[hdk_extern]
pub fn get_readiness_progress(contract_id: String) -> ExternResult<ReadinessProgress> {
    let contract = get_contract(contract_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))?;

    let contract_data = contract.entry().to_app_option::<RegenerativeContract>().ok().flatten()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid contract data".into())))?;

    // Get latest assessment
    let latest = get_latest_assessment(contract_id.clone())?;
    let overall_readiness = latest
        .and_then(|r| r.entry().to_app_option::<ReadinessAssessment>().ok().flatten())
        .map(|a| a.overall_readiness)
        .unwrap_or(0.0);

    // Calculate conditions met
    let conditions_count = contract_data.conditions.len() as u32;
    let conditions_met = if overall_readiness >= 80.0 {
        conditions_count
    } else if overall_readiness >= 50.0 {
        (conditions_count as f64 * 0.5) as u32
    } else {
        0
    };

    Ok(ReadinessProgress {
        contract_id,
        current_ownership_percentage: contract_data.current_ownership_percentage,
        target_ownership_percentage: contract_data.target_ownership_percentage,
        reserve_balance: contract_data.reserve_account_balance,
        overall_readiness,
        conditions_met,
        total_conditions: conditions_count,
        status: contract_data.status,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReadinessProgress {
    pub contract_id: String,
    pub current_ownership_percentage: f64,
    pub target_ownership_percentage: f64,
    pub reserve_balance: f64,
    pub overall_readiness: f64,
    pub conditions_met: u32,
    pub total_conditions: u32,
    pub status: ContractStatus,
}

/// Get regenerative exit summary across all contracts
#[hdk_extern]
pub fn get_regenerative_summary(_: ()) -> ExternResult<RegenerativeSummary> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?))
        .include_entries(true);

    let mut active_contracts = 0;
    let mut completed_contracts = 0;
    let mut total_reserve_balance = 0.0;
    let mut total_ownership_transferred = 0.0;

    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            match contract.status {
                ContractStatus::Active => active_contracts += 1,
                ContractStatus::TransitionComplete => {
                    completed_contracts += 1;
                    total_ownership_transferred += contract.current_ownership_percentage;
                }
                _ => {}
            }
            total_reserve_balance += contract.reserve_account_balance;
        }
    }

    // Count transfers
    let transfer_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::OwnershipTransfer)?))
        .include_entries(true);

    let total_transfers = query(transfer_filter)?.len() as u32;

    Ok(RegenerativeSummary {
        active_contracts,
        completed_contracts,
        total_reserve_balance,
        total_ownership_transferred,
        total_transfers,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegenerativeSummary {
    pub active_contracts: u32,
    pub completed_contracts: u32,
    pub total_reserve_balance: f64,
    pub total_ownership_transferred: f64,
    pub total_transfers: u32,
}

/// Pause a contract (stops further transfers)
#[hdk_extern]
pub fn pause_contract(contract_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.id == contract_id {
                if contract.status != ContractStatus::Active {
                    return Err(wasm_error!(WasmErrorInner::Guest("Can only pause active contracts".into())));
                }

                let updated = RegenerativeContract {
                    status: ContractStatus::Paused,
                    ..contract
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::RegenerativeContract(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))
}

/// Resume a paused contract
#[hdk_extern]
pub fn resume_contract(contract_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::RegenerativeContract)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(contract) = record.entry().to_app_option::<RegenerativeContract>().ok().flatten() {
            if contract.id == contract_id {
                if contract.status != ContractStatus::Paused {
                    return Err(wasm_error!(WasmErrorInner::Guest("Can only resume paused contracts".into())));
                }

                let updated = RegenerativeContract {
                    status: ContractStatus::Active,
                    ..contract
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::RegenerativeContract(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))
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

    fn valid_transition_condition() -> TransitionCondition {
        TransitionCondition {
            condition_type: ConditionType::CommunityReadiness,
            threshold: 0.8,
            current_value: 0.75,
            weight: 0.25,
            satisfied: false,
        }
    }

    fn valid_condition_score() -> ConditionScore {
        ConditionScore {
            condition_type: ConditionType::CommunityReadiness,
            score: 0.85,
            evidence: "Community training program completed".to_string(),
        }
    }

    // =========================================================================
    // CreateContractInput Tests
    // =========================================================================

    fn valid_create_contract_input() -> CreateContractInput {
        CreateContractInput {
            project_id: "project:solar_farm_alpha".to_string(),
            community_did: "did:mycelix:community1".to_string(),
            conditions: vec![
                valid_transition_condition(),
                TransitionCondition {
                    condition_type: ConditionType::FinancialSustainability,
                    threshold: 0.9,
                    current_value: 0.0,
                    weight: 0.3,
                    satisfied: false,
                },
            ],
            target_ownership_percentage: 100.0,
            currency: "USD".to_string(),
        }
    }

    #[test]
    fn test_create_contract_input_valid() {
        let input = valid_create_contract_input();
        assert!(input.community_did.starts_with("did:"));
        assert!(input.target_ownership_percentage > 0.0);
        assert!(!input.conditions.is_empty());
    }

    #[test]
    fn test_create_contract_input_multiple_conditions() {
        let input = valid_create_contract_input();
        assert!(input.conditions.len() >= 2);
    }

    #[test]
    fn test_create_contract_input_partial_ownership_target() {
        let input = CreateContractInput {
            target_ownership_percentage: 51.0,
            ..valid_create_contract_input()
        };
        assert!(input.target_ownership_percentage < 100.0);
    }

    #[test]
    fn test_create_contract_input_full_ownership_target() {
        let input = valid_create_contract_input();
        assert_eq!(input.target_ownership_percentage, 100.0);
    }

    #[test]
    fn test_create_contract_input_serialization() {
        let input = valid_create_contract_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    #[test]
    fn test_create_contract_input_various_currencies() {
        let currencies = vec!["USD", "EUR", "GBP", "CHF"];
        for currency in currencies {
            let input = CreateContractInput {
                currency: currency.to_string(),
                ..valid_create_contract_input()
            };
            assert_eq!(input.currency, currency);
        }
    }

    // =========================================================================
    // AssessmentInput Tests
    // =========================================================================

    fn valid_assessment_input() -> AssessmentInput {
        AssessmentInput {
            contract_id: "regen:project1:123456".to_string(),
            assessor_did: "did:mycelix:assessor1".to_string(),
            scores: vec![
                valid_condition_score(),
                ConditionScore {
                    condition_type: ConditionType::FinancialSustainability,
                    score: 0.9,
                    evidence: "Positive cash flow maintained".to_string(),
                },
            ],
            recommendations: vec![
                "Continue governance training".to_string(),
                "Build emergency reserve".to_string(),
            ],
        }
    }

    #[test]
    fn test_assessment_input_valid() {
        let input = valid_assessment_input();
        assert!(!input.contract_id.is_empty());
        assert!(input.assessor_did.starts_with("did:"));
        assert!(!input.scores.is_empty());
    }

    #[test]
    fn test_assessment_input_with_recommendations() {
        let input = valid_assessment_input();
        assert!(!input.recommendations.is_empty());
    }

    #[test]
    fn test_assessment_input_no_recommendations() {
        let input = AssessmentInput {
            recommendations: vec![],
            ..valid_assessment_input()
        };
        assert!(input.recommendations.is_empty());
    }

    #[test]
    fn test_assessment_input_score_calculation() {
        let input = valid_assessment_input();
        let avg: f64 = input.scores.iter().map(|s| s.score).sum::<f64>() / input.scores.len() as f64;
        assert!(avg >= 0.0 && avg <= 1.0);
    }

    #[test]
    fn test_assessment_input_all_condition_types() {
        let types = vec![
            ConditionType::CommunityReadiness,
            ConditionType::FinancialSustainability,
            ConditionType::OperationalCompetence,
            ConditionType::GovernanceMaturity,
            ConditionType::InvestorReturns,
            ConditionType::ReserveAccountFunded,
            ConditionType::MinimumOperatingHistory,
        ];
        let scores: Vec<ConditionScore> = types.into_iter()
            .map(|t| ConditionScore {
                condition_type: t,
                score: 0.8,
                evidence: "Test evidence".to_string(),
            })
            .collect();

        let input = AssessmentInput {
            scores,
            ..valid_assessment_input()
        };
        assert_eq!(input.scores.len(), 7);
    }

    // =========================================================================
    // TransferInput Tests
    // =========================================================================

    fn valid_transfer_input() -> TransferInput {
        TransferInput {
            contract_id: "regen:project1:123456".to_string(),
            new_percentage: 25.0,
            shares_transferred: 150.0,
            transfer_price: 75000.0,
        }
    }

    #[test]
    fn test_transfer_input_valid() {
        let input = valid_transfer_input();
        assert!(!input.contract_id.is_empty());
        assert!(input.new_percentage > 0.0);
        assert!(input.shares_transferred > 0.0);
    }

    #[test]
    fn test_transfer_input_large_transfer() {
        let input = TransferInput {
            new_percentage: 100.0,
            shares_transferred: 10000.0,
            transfer_price: 5_000_000.0,
            ..valid_transfer_input()
        };
        assert_eq!(input.new_percentage, 100.0);
    }

    #[test]
    fn test_transfer_input_small_transfer() {
        let input = TransferInput {
            new_percentage: 11.0,
            shares_transferred: 10.0,
            transfer_price: 5000.0,
            ..valid_transfer_input()
        };
        assert!(input.new_percentage > 0.0);
    }

    #[test]
    fn test_transfer_input_serialization() {
        let input = valid_transfer_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    // =========================================================================
    // UpdateContractStatusInput Tests
    // =========================================================================

    fn valid_update_contract_status_input() -> UpdateContractStatusInput {
        UpdateContractStatusInput {
            contract_id: "regen:project1:123456".to_string(),
            new_status: ContractStatus::TransitionInProgress,
        }
    }

    #[test]
    fn test_update_contract_status_input_valid() {
        let input = valid_update_contract_status_input();
        assert!(!input.contract_id.is_empty());
    }

    #[test]
    fn test_update_contract_status_input_all_statuses() {
        let statuses = vec![
            ContractStatus::Active,
            ContractStatus::TransitionInProgress,
            ContractStatus::TransitionComplete,
            ContractStatus::Paused,
            ContractStatus::Terminated,
        ];
        for status in statuses {
            let input = UpdateContractStatusInput {
                new_status: status.clone(),
                ..valid_update_contract_status_input()
            };
            assert_eq!(input.new_status, status);
        }
    }

    // =========================================================================
    // AddToReserveInput Tests
    // =========================================================================

    fn valid_add_to_reserve_input() -> AddToReserveInput {
        AddToReserveInput {
            contract_id: "regen:project1:123456".to_string(),
            amount: 10000.0,
        }
    }

    #[test]
    fn test_add_to_reserve_input_valid() {
        let input = valid_add_to_reserve_input();
        assert!(!input.contract_id.is_empty());
        assert!(input.amount > 0.0);
    }

    #[test]
    fn test_add_to_reserve_input_large_amount() {
        let input = AddToReserveInput {
            amount: 1_000_000.0,
            ..valid_add_to_reserve_input()
        };
        assert!(input.amount > 0.0);
    }

    #[test]
    fn test_add_to_reserve_input_small_amount() {
        let input = AddToReserveInput {
            amount: 100.0,
            ..valid_add_to_reserve_input()
        };
        assert!(input.amount > 0.0);
    }

    // =========================================================================
    // UpdateConditionsInput Tests
    // =========================================================================

    fn valid_update_conditions_input() -> UpdateConditionsInput {
        UpdateConditionsInput {
            contract_id: "regen:project1:123456".to_string(),
            conditions: vec![
                TransitionCondition {
                    condition_type: ConditionType::CommunityReadiness,
                    threshold: 0.85,
                    current_value: 0.80,
                    weight: 0.3,
                    satisfied: false,
                },
                TransitionCondition {
                    condition_type: ConditionType::FinancialSustainability,
                    threshold: 0.9,
                    current_value: 0.88,
                    weight: 0.4,
                    satisfied: false,
                },
            ],
        }
    }

    #[test]
    fn test_update_conditions_input_valid() {
        let input = valid_update_conditions_input();
        assert!(!input.contract_id.is_empty());
        assert!(!input.conditions.is_empty());
    }

    #[test]
    fn test_update_conditions_input_updated_values() {
        let input = valid_update_conditions_input();
        // Check that conditions have updated current values
        assert!(input.conditions.iter().all(|c| c.current_value >= 0.0));
    }

    // =========================================================================
    // ReadinessProgress Tests
    // =========================================================================

    #[test]
    fn test_readiness_progress_creation() {
        let progress = ReadinessProgress {
            contract_id: "regen:project1:123456".to_string(),
            current_ownership_percentage: 25.0,
            target_ownership_percentage: 100.0,
            reserve_balance: 100000.0,
            overall_readiness: 0.75,
            conditions_met: 2,
            total_conditions: 5,
            status: ContractStatus::Active,
        };
        assert!(!progress.contract_id.is_empty());
        assert!(progress.conditions_met <= progress.total_conditions);
    }

    #[test]
    fn test_readiness_progress_early_stage() {
        let progress = ReadinessProgress {
            contract_id: "regen:project1:123456".to_string(),
            current_ownership_percentage: 0.0,
            target_ownership_percentage: 100.0,
            reserve_balance: 10000.0,
            overall_readiness: 0.25,
            conditions_met: 0,
            total_conditions: 5,
            status: ContractStatus::Active,
        };
        assert_eq!(progress.current_ownership_percentage, 0.0);
        assert_eq!(progress.conditions_met, 0);
    }

    #[test]
    fn test_readiness_progress_near_completion() {
        let progress = ReadinessProgress {
            contract_id: "regen:project1:123456".to_string(),
            current_ownership_percentage: 90.0,
            target_ownership_percentage: 100.0,
            reserve_balance: 500000.0,
            overall_readiness: 0.95,
            conditions_met: 5,
            total_conditions: 5,
            status: ContractStatus::TransitionInProgress,
        };
        assert!(progress.overall_readiness >= 0.9);
        assert_eq!(progress.conditions_met, progress.total_conditions);
    }

    #[test]
    fn test_readiness_progress_completed() {
        let progress = ReadinessProgress {
            contract_id: "regen:project1:123456".to_string(),
            current_ownership_percentage: 100.0,
            target_ownership_percentage: 100.0,
            reserve_balance: 500000.0,
            overall_readiness: 1.0,
            conditions_met: 5,
            total_conditions: 5,
            status: ContractStatus::TransitionComplete,
        };
        assert_eq!(progress.current_ownership_percentage, progress.target_ownership_percentage);
        assert_eq!(progress.status, ContractStatus::TransitionComplete);
    }

    // =========================================================================
    // RegenerativeSummary Tests
    // =========================================================================

    #[test]
    fn test_regenerative_summary_creation() {
        let summary = RegenerativeSummary {
            active_contracts: 10,
            completed_contracts: 5,
            total_reserve_balance: 1_000_000.0,
            total_ownership_transferred: 350.0,
            total_transfers: 25,
        };
        assert!(summary.active_contracts >= 0);
        assert!(summary.completed_contracts >= 0);
    }

    #[test]
    fn test_regenerative_summary_empty() {
        let summary = RegenerativeSummary {
            active_contracts: 0,
            completed_contracts: 0,
            total_reserve_balance: 0.0,
            total_ownership_transferred: 0.0,
            total_transfers: 0,
        };
        assert_eq!(summary.active_contracts, 0);
        assert_eq!(summary.total_transfers, 0);
    }

    #[test]
    fn test_regenerative_summary_high_activity() {
        let summary = RegenerativeSummary {
            active_contracts: 100,
            completed_contracts: 50,
            total_reserve_balance: 50_000_000.0,
            total_ownership_transferred: 5000.0,
            total_transfers: 500,
        };
        assert!(summary.active_contracts > 0);
        assert!(summary.total_transfers > 0);
    }

    // =========================================================================
    // Business Logic Edge Cases
    // =========================================================================

    #[test]
    fn test_contract_id_format() {
        let project_id = "project:solar_farm";
        let timestamp = 1704067200000000_u64;
        let id = format!("regen:{}:{}", project_id, timestamp);
        assert!(id.starts_with("regen:"));
    }

    #[test]
    fn test_assessment_id_format() {
        let contract_id = "regen:project1:123";
        let timestamp = 1704067200000000_u64;
        let id = format!("assess:{}:{}", contract_id, timestamp);
        assert!(id.starts_with("assess:"));
    }

    #[test]
    fn test_transfer_id_format() {
        let contract_id = "regen:project1:123";
        let timestamp = 1704067200000000_u64;
        let id = format!("transfer:{}:{}", contract_id, timestamp);
        assert!(id.starts_with("transfer:"));
    }

    #[test]
    fn test_overall_readiness_calculation() {
        let scores = vec![0.8_f64, 0.9, 0.7, 0.85];
        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        assert!((avg - 0.8125).abs() < 0.001);
    }

    #[test]
    fn test_condition_weights_validation() {
        let input = valid_create_contract_input();
        let total_weight: f64 = input.conditions.iter().map(|c| c.weight).sum();
        // Weights should sum reasonably (doesn't need to be exactly 1.0 for this test)
        assert!(total_weight > 0.0);
    }

    #[test]
    fn test_ownership_increase_calculation() {
        let from_pct = 25.0;
        let to_pct = 50.0;
        let increase = to_pct - from_pct;
        assert_eq!(increase, 25.0);
    }

    #[test]
    fn test_reserve_accumulation() {
        let initial_reserve = 50000.0;
        let additions = vec![10000.0, 15000.0, 25000.0];
        let final_reserve: f64 = initial_reserve + additions.iter().sum::<f64>();
        assert_eq!(final_reserve, 100000.0);
    }

    #[test]
    fn test_deserialization_roundtrip_contract_input() {
        let input = valid_create_contract_input();
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: CreateContractInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.project_id, input.project_id);
        assert_eq!(deserialized.target_ownership_percentage, input.target_ownership_percentage);
    }

    #[test]
    fn test_deserialization_roundtrip_assessment_input() {
        let input = valid_assessment_input();
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: AssessmentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.contract_id, input.contract_id);
        assert_eq!(deserialized.scores.len(), input.scores.len());
    }

    #[test]
    fn test_pause_resume_contract_flow() {
        // Simulate pause/resume logic
        let initial_status = ContractStatus::Active;
        let paused_status = ContractStatus::Paused;
        let resumed_status = ContractStatus::Active;

        assert_eq!(initial_status, ContractStatus::Active);
        assert_eq!(paused_status, ContractStatus::Paused);
        assert_eq!(resumed_status, ContractStatus::Active);
    }

    #[test]
    fn test_transition_completion_check() {
        let current_ownership = 100.0;
        let target_ownership = 100.0;
        let is_complete = current_ownership >= target_ownership;
        assert!(is_complete);
    }
}
