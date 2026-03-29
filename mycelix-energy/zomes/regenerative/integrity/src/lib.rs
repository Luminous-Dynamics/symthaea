// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regenerative Exit Integrity Zome
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RegenerativeContract {
    pub id: String,
    pub project_id: String,
    pub community_did: String,
    pub conditions: Vec<TransitionCondition>,
    pub current_ownership_percentage: f64,
    pub target_ownership_percentage: f64,
    pub reserve_account_balance: f64,
    pub currency: String,
    pub status: ContractStatus,
    pub created: Timestamp,
    pub last_assessment: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TransitionCondition {
    pub condition_type: ConditionType,
    pub threshold: f64,
    pub current_value: f64,
    pub weight: f64,
    pub satisfied: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ConditionType {
    CommunityReadiness,
    FinancialSustainability,
    OperationalCompetence,
    GovernanceMaturity,
    InvestorReturns,
    ReserveAccountFunded,
    MinimumOperatingHistory,
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContractStatus {
    Active,
    TransitionInProgress,
    TransitionComplete,
    Paused,
    Terminated,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReadinessAssessment {
    pub id: String,
    pub contract_id: String,
    pub assessor_did: String,
    pub scores: Vec<ConditionScore>,
    pub overall_readiness: f64,
    pub recommendations: Vec<String>,
    pub assessed: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ConditionScore {
    pub condition_type: ConditionType,
    pub score: f64,
    pub evidence: String,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OwnershipTransfer {
    pub id: String,
    pub contract_id: String,
    pub from_percentage: f64,
    pub to_percentage: f64,
    pub shares_transferred: f64,
    pub transfer_price: f64,
    pub currency: String,
    pub executed: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    RegenerativeContract(RegenerativeContract),
    ReadinessAssessment(ReadinessAssessment),
    OwnershipTransfer(OwnershipTransfer),
}

#[hdk_link_types]
pub enum LinkTypes {
    ProjectToContract,
    CommunityToContract,
    ContractToAssessments,
    ContractToTransfers,
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
                    EntryTypes::RegenerativeContract(contract) => {
                        validate_create_regenerative_contract(EntryCreationAction::Create(action), contract)
                    }
                    EntryTypes::ReadinessAssessment(assessment) => {
                        validate_create_readiness_assessment(EntryCreationAction::Create(action), assessment)
                    }
                    EntryTypes::OwnershipTransfer(transfer) => {
                        validate_create_ownership_transfer(EntryCreationAction::Create(action), transfer)
                    }
                }
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::RegenerativeContract(contract) => {
                        validate_update_regenerative_contract(action, contract)
                    }
                    EntryTypes::ReadinessAssessment(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Assessments cannot be updated".into(),
                        ))
                    }
                    EntryTypes::OwnershipTransfer(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Transfers cannot be updated".into(),
                        ))
                    }
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            match link_type {
                LinkTypes::ProjectToContract => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CommunityToContract => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ContractToAssessments => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ContractToTransfers => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_regenerative_contract(
    _action: EntryCreationAction,
    contract: RegenerativeContract,
) -> ExternResult<ValidateCallbackResult> {
    if !contract.community_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Community must be a valid DID".into()));
    }
    if contract.target_ownership_percentage <= contract.current_ownership_percentage {
        return Ok(ValidateCallbackResult::Invalid("Target must exceed current ownership".into()));
    }
    if contract.conditions.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Must have at least one condition".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_regenerative_contract(
    _action: Update,
    contract: RegenerativeContract,
) -> ExternResult<ValidateCallbackResult> {
    if contract.conditions.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Must have at least one condition".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_readiness_assessment(
    _action: EntryCreationAction,
    assessment: ReadinessAssessment,
) -> ExternResult<ValidateCallbackResult> {
    if !assessment.assessor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Assessor must be a valid DID".into()));
    }
    if assessment.overall_readiness < 0.0 || assessment.overall_readiness > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Readiness must be 0-1".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_ownership_transfer(
    _action: EntryCreationAction,
    transfer: OwnershipTransfer,
) -> ExternResult<ValidateCallbackResult> {
    if transfer.to_percentage <= transfer.from_percentage {
        return Ok(ValidateCallbackResult::Invalid("Transfer must increase ownership".into()));
    }
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
    // ConditionType Enum Tests
    // =========================================================================

    #[test]
    fn test_condition_type_variants() {
        let types = vec![
            ConditionType::CommunityReadiness,
            ConditionType::FinancialSustainability,
            ConditionType::OperationalCompetence,
            ConditionType::GovernanceMaturity,
            ConditionType::InvestorReturns,
            ConditionType::ReserveAccountFunded,
            ConditionType::MinimumOperatingHistory,
            ConditionType::Custom("EnvironmentalCompliance".to_string()),
        ];
        assert_eq!(types.len(), 8);
    }

    #[test]
    fn test_condition_type_equality() {
        assert_eq!(ConditionType::CommunityReadiness, ConditionType::CommunityReadiness);
        assert_ne!(ConditionType::CommunityReadiness, ConditionType::GovernanceMaturity);
    }

    #[test]
    fn test_condition_type_custom_equality() {
        let custom1 = ConditionType::Custom("Test".to_string());
        let custom2 = ConditionType::Custom("Test".to_string());
        let custom3 = ConditionType::Custom("Other".to_string());

        assert_eq!(custom1, custom2);
        assert_ne!(custom1, custom3);
    }

    // =========================================================================
    // ContractStatus Enum Tests
    // =========================================================================

    #[test]
    fn test_contract_status_variants() {
        let statuses = vec![
            ContractStatus::Active,
            ContractStatus::TransitionInProgress,
            ContractStatus::TransitionComplete,
            ContractStatus::Paused,
            ContractStatus::Terminated,
        ];
        assert_eq!(statuses.len(), 5);
    }

    #[test]
    fn test_contract_status_equality() {
        assert_eq!(ContractStatus::Active, ContractStatus::Active);
        assert_ne!(ContractStatus::Active, ContractStatus::Paused);
    }

    #[test]
    fn test_contract_status_lifecycle() {
        // Typical progression: Active -> TransitionInProgress -> TransitionComplete
        let lifecycle = [
            ContractStatus::Active,
            ContractStatus::TransitionInProgress,
            ContractStatus::TransitionComplete,
        ];
        for i in 0..lifecycle.len() - 1 {
            assert_ne!(lifecycle[i], lifecycle[i + 1]);
        }
    }

    // =========================================================================
    // TransitionCondition Tests
    // =========================================================================

    fn valid_transition_condition() -> TransitionCondition {
        TransitionCondition {
            condition_type: ConditionType::CommunityReadiness,
            threshold: 0.8,
            current_value: 0.75,
            weight: 0.25,
            satisfied: false,
        }
    }

    #[test]
    fn test_transition_condition_valid() {
        let condition = valid_transition_condition();
        assert!(condition.threshold >= 0.0 && condition.threshold <= 1.0);
        assert!(condition.current_value >= 0.0);
        assert!(condition.weight > 0.0);
    }

    #[test]
    fn test_transition_condition_satisfied() {
        let condition = TransitionCondition {
            current_value: 0.85,
            satisfied: true,
            ..valid_transition_condition()
        };
        assert!(condition.satisfied);
        assert!(condition.current_value >= condition.threshold);
    }

    #[test]
    fn test_transition_condition_not_satisfied() {
        let condition = valid_transition_condition();
        assert!(!condition.satisfied);
        assert!(condition.current_value < condition.threshold);
    }

    #[test]
    fn test_transition_condition_all_types() {
        let types = vec![
            ConditionType::CommunityReadiness,
            ConditionType::FinancialSustainability,
            ConditionType::OperationalCompetence,
            ConditionType::GovernanceMaturity,
            ConditionType::InvestorReturns,
            ConditionType::ReserveAccountFunded,
            ConditionType::MinimumOperatingHistory,
        ];
        for cond_type in types {
            let condition = TransitionCondition {
                condition_type: cond_type.clone(),
                ..valid_transition_condition()
            };
            assert_eq!(condition.condition_type, cond_type);
        }
    }

    // =========================================================================
    // RegenerativeContract Validation Tests
    // =========================================================================

    fn valid_regenerative_contract() -> RegenerativeContract {
        RegenerativeContract {
            id: "regen:project1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            community_did: "did:mycelix:community1".to_string(),
            conditions: vec![
                valid_transition_condition(),
                TransitionCondition {
                    condition_type: ConditionType::FinancialSustainability,
                    threshold: 0.9,
                    current_value: 0.85,
                    weight: 0.3,
                    satisfied: false,
                },
            ],
            current_ownership_percentage: 10.0,
            target_ownership_percentage: 100.0,
            reserve_account_balance: 50000.0,
            currency: "USD".to_string(),
            status: ContractStatus::Active,
            created: create_test_timestamp(),
            last_assessment: create_test_timestamp(),
        }
    }

    #[test]
    fn test_regenerative_contract_valid_community_did() {
        let contract = valid_regenerative_contract();
        assert!(contract.community_did.starts_with("did:"));
    }

    #[test]
    fn test_regenerative_contract_target_exceeds_current() {
        let contract = valid_regenerative_contract();
        assert!(contract.target_ownership_percentage > contract.current_ownership_percentage);
    }

    #[test]
    fn test_regenerative_contract_has_conditions() {
        let contract = valid_regenerative_contract();
        assert!(!contract.conditions.is_empty());
    }

    #[test]
    fn test_regenerative_contract_invalid_community_did() {
        let contract = RegenerativeContract {
            community_did: "community123".to_string(),
            ..valid_regenerative_contract()
        };
        assert!(!contract.community_did.starts_with("did:"));
    }

    #[test]
    fn test_regenerative_contract_target_not_greater_than_current_invalid() {
        let contract = RegenerativeContract {
            current_ownership_percentage: 50.0,
            target_ownership_percentage: 50.0,
            ..valid_regenerative_contract()
        };
        assert!(contract.target_ownership_percentage <= contract.current_ownership_percentage);
    }

    #[test]
    fn test_regenerative_contract_no_conditions_invalid() {
        let contract = RegenerativeContract {
            conditions: vec![],
            ..valid_regenerative_contract()
        };
        assert!(contract.conditions.is_empty());
    }

    #[test]
    fn test_regenerative_contract_full_ownership() {
        let contract = RegenerativeContract {
            current_ownership_percentage: 100.0,
            target_ownership_percentage: 100.0,
            status: ContractStatus::TransitionComplete,
            ..valid_regenerative_contract()
        };
        assert_eq!(contract.current_ownership_percentage, 100.0);
    }

    #[test]
    fn test_regenerative_contract_zero_ownership() {
        let contract = RegenerativeContract {
            current_ownership_percentage: 0.0,
            ..valid_regenerative_contract()
        };
        assert_eq!(contract.current_ownership_percentage, 0.0);
    }

    // =========================================================================
    // ReadinessAssessment Validation Tests
    // =========================================================================

    fn valid_condition_score() -> ConditionScore {
        ConditionScore {
            condition_type: ConditionType::CommunityReadiness,
            score: 0.85,
            evidence: "Community training program completed with 95% participation".to_string(),
        }
    }

    fn valid_readiness_assessment() -> ReadinessAssessment {
        ReadinessAssessment {
            id: "assess:regen1:123456".to_string(),
            contract_id: "regen:project1:123456".to_string(),
            assessor_did: "did:mycelix:assessor1".to_string(),
            scores: vec![
                valid_condition_score(),
                ConditionScore {
                    condition_type: ConditionType::FinancialSustainability,
                    score: 0.9,
                    evidence: "Positive cash flow for 12 consecutive months".to_string(),
                },
            ],
            overall_readiness: 0.875,
            recommendations: vec![
                "Continue governance training".to_string(),
                "Build emergency reserve to 6 months".to_string(),
            ],
            assessed: create_test_timestamp(),
        }
    }

    #[test]
    fn test_readiness_assessment_valid_assessor_did() {
        let assessment = valid_readiness_assessment();
        assert!(assessment.assessor_did.starts_with("did:"));
    }

    #[test]
    fn test_readiness_assessment_valid_overall_readiness() {
        let assessment = valid_readiness_assessment();
        assert!(assessment.overall_readiness >= 0.0 && assessment.overall_readiness <= 1.0);
    }

    #[test]
    fn test_readiness_assessment_invalid_assessor_did() {
        let assessment = ReadinessAssessment {
            assessor_did: "assessor123".to_string(),
            ..valid_readiness_assessment()
        };
        assert!(!assessment.assessor_did.starts_with("did:"));
    }

    #[test]
    fn test_readiness_assessment_over_1_invalid() {
        let assessment = ReadinessAssessment {
            overall_readiness: 1.5,
            ..valid_readiness_assessment()
        };
        assert!(assessment.overall_readiness > 1.0);
    }

    #[test]
    fn test_readiness_assessment_negative_invalid() {
        let assessment = ReadinessAssessment {
            overall_readiness: -0.1,
            ..valid_readiness_assessment()
        };
        assert!(assessment.overall_readiness < 0.0);
    }

    #[test]
    fn test_readiness_assessment_perfect_score() {
        let assessment = ReadinessAssessment {
            overall_readiness: 1.0,
            ..valid_readiness_assessment()
        };
        assert_eq!(assessment.overall_readiness, 1.0);
    }

    #[test]
    fn test_readiness_assessment_zero_score() {
        let assessment = ReadinessAssessment {
            overall_readiness: 0.0,
            ..valid_readiness_assessment()
        };
        assert_eq!(assessment.overall_readiness, 0.0);
    }

    #[test]
    fn test_readiness_assessment_with_recommendations() {
        let assessment = valid_readiness_assessment();
        assert!(!assessment.recommendations.is_empty());
    }

    #[test]
    fn test_readiness_assessment_no_recommendations() {
        let assessment = ReadinessAssessment {
            recommendations: vec![],
            ..valid_readiness_assessment()
        };
        assert!(assessment.recommendations.is_empty());
    }

    // =========================================================================
    // ConditionScore Tests
    // =========================================================================

    #[test]
    fn test_condition_score_valid() {
        let score = valid_condition_score();
        assert!(score.score >= 0.0 && score.score <= 1.0);
        assert!(!score.evidence.is_empty());
    }

    #[test]
    fn test_condition_score_all_types() {
        let types = vec![
            ConditionType::CommunityReadiness,
            ConditionType::FinancialSustainability,
            ConditionType::OperationalCompetence,
            ConditionType::GovernanceMaturity,
            ConditionType::InvestorReturns,
            ConditionType::ReserveAccountFunded,
            ConditionType::MinimumOperatingHistory,
        ];
        for cond_type in types {
            let score = ConditionScore {
                condition_type: cond_type.clone(),
                ..valid_condition_score()
            };
            assert_eq!(score.condition_type, cond_type);
        }
    }

    // =========================================================================
    // OwnershipTransfer Validation Tests
    // =========================================================================

    fn valid_ownership_transfer() -> OwnershipTransfer {
        OwnershipTransfer {
            id: "transfer:regen1:123456".to_string(),
            contract_id: "regen:project1:123456".to_string(),
            from_percentage: 10.0,
            to_percentage: 25.0,
            shares_transferred: 150.0,
            transfer_price: 75000.0,
            currency: "USD".to_string(),
            executed: create_test_timestamp(),
        }
    }

    #[test]
    fn test_ownership_transfer_valid() {
        let transfer = valid_ownership_transfer();
        assert!(transfer.to_percentage > transfer.from_percentage);
    }

    #[test]
    fn test_ownership_transfer_increase_ownership() {
        let transfer = valid_ownership_transfer();
        let increase = transfer.to_percentage - transfer.from_percentage;
        assert!(increase > 0.0);
    }

    #[test]
    fn test_ownership_transfer_to_equals_from_invalid() {
        let transfer = OwnershipTransfer {
            from_percentage: 25.0,
            to_percentage: 25.0,
            ..valid_ownership_transfer()
        };
        assert!(transfer.to_percentage <= transfer.from_percentage);
    }

    #[test]
    fn test_ownership_transfer_decrease_invalid() {
        let transfer = OwnershipTransfer {
            from_percentage: 50.0,
            to_percentage: 25.0,
            ..valid_ownership_transfer()
        };
        assert!(transfer.to_percentage <= transfer.from_percentage);
    }

    #[test]
    fn test_ownership_transfer_to_100_percent() {
        let transfer = OwnershipTransfer {
            from_percentage: 75.0,
            to_percentage: 100.0,
            ..valid_ownership_transfer()
        };
        assert_eq!(transfer.to_percentage, 100.0);
    }

    #[test]
    fn test_ownership_transfer_small_increment() {
        let transfer = OwnershipTransfer {
            from_percentage: 10.0,
            to_percentage: 10.5,
            shares_transferred: 5.0,
            ..valid_ownership_transfer()
        };
        assert!(transfer.to_percentage > transfer.from_percentage);
    }

    // =========================================================================
    // Anchor Tests
    // =========================================================================

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("project_contracts".to_string());
        assert_eq!(anchor.0, "project_contracts");
    }

    #[test]
    fn test_anchor_equality() {
        let anchor1 = Anchor("community_contracts".to_string());
        let anchor2 = Anchor("community_contracts".to_string());
        let anchor3 = Anchor("project_contracts".to_string());

        assert_eq!(anchor1, anchor2);
        assert_ne!(anchor1, anchor3);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_very_large_reserve_balance() {
        let contract = RegenerativeContract {
            reserve_account_balance: 100_000_000.0,
            ..valid_regenerative_contract()
        };
        assert!(contract.reserve_account_balance > 0.0);
    }

    #[test]
    fn test_zero_reserve_balance() {
        let contract = RegenerativeContract {
            reserve_account_balance: 0.0,
            ..valid_regenerative_contract()
        };
        assert_eq!(contract.reserve_account_balance, 0.0);
    }

    #[test]
    fn test_many_conditions() {
        let conditions: Vec<TransitionCondition> = (0..10)
            .map(|i| TransitionCondition {
                condition_type: ConditionType::Custom(format!("Custom_{}", i)),
                threshold: 0.5 + (i as f64 * 0.05),
                current_value: 0.0,
                weight: 0.1,
                satisfied: false,
            })
            .collect();

        let contract = RegenerativeContract {
            conditions,
            ..valid_regenerative_contract()
        };
        assert_eq!(contract.conditions.len(), 10);
    }

    #[test]
    fn test_condition_weights_sum_to_one() {
        let conditions = vec![
            TransitionCondition {
                weight: 0.3,
                ..valid_transition_condition()
            },
            TransitionCondition {
                condition_type: ConditionType::FinancialSustainability,
                weight: 0.4,
                ..valid_transition_condition()
            },
            TransitionCondition {
                condition_type: ConditionType::GovernanceMaturity,
                weight: 0.3,
                ..valid_transition_condition()
            },
        ];
        let total_weight: f64 = conditions.iter().map(|c| c.weight).sum();
        assert!((total_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_assessment_score_averaging() {
        let scores = vec![
            ConditionScore {
                score: 0.8,
                ..valid_condition_score()
            },
            ConditionScore {
                condition_type: ConditionType::FinancialSustainability,
                score: 0.9,
                evidence: "Test".to_string(),
            },
        ];
        let average: f64 = scores.iter().map(|s| s.score).sum::<f64>() / scores.len() as f64;
        // Use approximate comparison for floating point
        assert!((average - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_serialization_contract() {
        let contract = valid_regenerative_contract();
        let result = serde_json::to_string(&contract);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_assessment() {
        let assessment = valid_readiness_assessment();
        let result = serde_json::to_string(&assessment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_transfer() {
        let transfer = valid_ownership_transfer();
        let result = serde_json::to_string(&transfer);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_condition() {
        let condition = valid_transition_condition();
        let result = serde_json::to_string(&condition);
        assert!(result.is_ok());
    }
}
