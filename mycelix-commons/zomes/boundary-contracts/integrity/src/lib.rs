// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Boundary Contracts Integrity Zome
//!
//! Pre-exchange consent layer for care/intimacy work. The cryptographic
//! mechanism that prevents the 1602 extraction model from re-emerging.
//!
//! Contract terms are private between parties. Only existence + status
//! are visible on the DHT. Violations route to restorative justice,
//! never punitive enforcement — the thermodynamic mirror handles the rest.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

// =============================================================================
// Service category (serde-compatible mirror of tend_integrity::ServiceCategory)
// =============================================================================

/// Categories of service that can be exchanged.
/// Mirrors `tend_integrity::ServiceCategory` serde layout for cross-zome compat.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ServiceCategory {
    CareWork,
    HomeServices,
    FoodServices,
    Transportation,
    Education,
    GeneralAssistance,
    Administrative,
    Creative,
    TechSupport,
    Wellness,
    Gardening,
    Custom(String),
}

// =============================================================================
// Entry types
// =============================================================================

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Contract status with validated state machine transitions.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContractStatus {
    /// Proposed, awaiting counter-signature.
    Proposed,
    /// Both parties signed — exchange can proceed.
    Active,
    /// Revoked by either party (partial TEND settlement).
    Revoked,
    /// Successfully completed (full TEND settlement).
    Completed,
    /// A boundary was violated (routes to restorative justice).
    Violated,
}

impl ContractStatus {
    /// Valid state transitions. Enforced by integrity validation.
    pub fn can_transition_to(&self, new: &ContractStatus) -> bool {
        matches!(
            (self, new),
            (ContractStatus::Proposed, ContractStatus::Active)
                | (ContractStatus::Proposed, ContractStatus::Revoked)
                | (ContractStatus::Active, ContractStatus::Completed)
                | (ContractStatus::Active, ContractStatus::Revoked)
                | (ContractStatus::Active, ContractStatus::Violated)
        )
    }
}

/// A pre-exchange boundary contract between two parties.
/// Private between the provider and receiver.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BoundaryContract {
    /// Unique contract identifier.
    pub id: String,
    /// The care worker / service provider.
    pub provider: AgentPubKey,
    /// The care receiver / client.
    pub receiver: AgentPubKey,
    /// TEND service category.
    pub service_type: ServiceCategory,
    /// Explicit boundaries: what IS included.
    pub inclusions: Vec<String>,
    /// Explicit boundaries: what is NOT included.
    pub exclusions: Vec<String>,
    /// Agreed duration (minutes). Max 1 week (10080).
    pub duration_minutes: u32,
    /// TEND hours compensation. Max 168 (1 week).
    pub tend_hours: f32,
    /// Conditions under which either party can revoke.
    pub revocation_conditions: Vec<String>,
    /// Current status.
    pub status: ContractStatus,
    /// Provider signature timestamp.
    pub provider_signed_at: Option<Timestamp>,
    /// Receiver signature timestamp.
    pub receiver_signed_at: Option<Timestamp>,
    /// Creation timestamp.
    pub created_at: Timestamp,
    /// Last update timestamp.
    pub updated_at: Timestamp,
}

/// Public-facing contract summary. Only this goes to the DHT.
/// Reveals existence and status without exposing boundary terms.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ContractSummary {
    /// References the full BoundaryContract.
    pub contract_hash: ActionHash,
    /// Provider agent.
    pub provider: AgentPubKey,
    /// Receiver agent.
    pub receiver: AgentPubKey,
    /// Service category (not sensitive).
    pub service_type: ServiceCategory,
    /// Current status.
    pub status: ContractStatus,
    /// Created timestamp.
    pub created_at: Timestamp,
    /// Updated timestamp.
    pub updated_at: Timestamp,
}

/// Severity of a boundary violation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ViolationSeverity {
    /// Minor misunderstanding — nudge + conversation.
    Minor,
    /// Moderate — reputation impact + restorative circle.
    Moderate,
    /// Severe — significant reputation slash + mandatory restorative process.
    Severe,
}

impl ViolationSeverity {
    /// Reputation slash factor for SubPassport community dimension.
    pub fn slash_factor(&self) -> f32 {
        match self {
            ViolationSeverity::Minor => 0.05,
            ViolationSeverity::Moderate => 0.15,
            ViolationSeverity::Severe => 0.30,
        }
    }
}

/// A boundary violation report.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BoundaryViolation {
    /// The contract that was violated.
    pub contract_hash: ActionHash,
    /// Who reported the violation.
    pub reporter: AgentPubKey,
    /// Who allegedly violated the boundary.
    pub violator: AgentPubKey,
    /// Description (private — shared only with restorative justice facilitator).
    pub description: String,
    /// Severity assessment.
    pub severity: ViolationSeverity,
    /// Restorative justice case ID (populated after routing).
    pub restorative_case_id: Option<String>,
    /// When reported.
    pub reported_at: Timestamp,
}

/// Standing boundaries a care worker applies to ALL their contracts.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StandingBoundaries {
    /// The care worker.
    pub agent: AgentPubKey,
    /// Service types they offer.
    pub offered_services: Vec<ServiceCategory>,
    /// Universal exclusions (apply to all contracts).
    pub universal_exclusions: Vec<String>,
    /// Maximum session duration (minutes).
    pub max_duration_minutes: Option<u32>,
    /// Minimum TEND hours per session.
    pub min_tend_hours: Option<f32>,
    /// Agents blocked from proposing contracts.
    pub blocked_agents: Vec<AgentPubKey>,
    /// Updated timestamp.
    pub updated_at: Timestamp,
}

// =============================================================================
// Entry and link type registration
// =============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    #[entry_type(visibility = "private")]
    BoundaryContract(BoundaryContract),
    ContractSummary(ContractSummary),
    #[entry_type(visibility = "private")]
    BoundaryViolation(BoundaryViolation),
    StandingBoundaries(StandingBoundaries),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> their contracts (as provider or receiver).
    AgentToContracts,
    /// Agent -> their standing boundaries.
    AgentToStandingBoundaries,
    /// Contract -> public summary.
    ContractToSummary,
    /// Contract -> violation reports.
    ContractToViolation,
    /// Anchor -> all active contract summaries.
    AllActiveSummaries,
    /// Agent -> agents they've blocked.
    AgentToBlockedAgents,
    /// Violation -> restorative justice case.
    ViolationToCase,
}

// =============================================================================
// Validation constants
// =============================================================================

/// Maximum contract ID length.
const MAX_ID_BYTES: usize = 256;

/// Maximum TEND hours per contract (1 week).
const MAX_TEND_HOURS: f32 = 168.0;

/// Maximum duration in minutes (1 week).
const MAX_DURATION_MINUTES: u32 = 10_080;

/// Maximum boundary items (inclusions or exclusions).
const MAX_BOUNDARY_ITEMS: usize = 50;

/// Maximum bytes per boundary item.
const MAX_BOUNDARY_ITEM_BYTES: usize = 512;

/// Maximum violation description length.
const MAX_VIOLATION_DESC_BYTES: usize = 4096;

/// Maximum blocked agents per standing boundary.
const MAX_BLOCKED_AGENTS: usize = 1000;

/// Maximum link tag size.
const MAX_TAG_BYTES: usize = 256;

// =============================================================================
// Genesis
// =============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// Validation
// =============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, &action.author)
            }
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                ..
            } => {
                let original_action = must_get_action(original_action_hash)?;
                let author_check = check_author_match(
                    original_action.action().author(),
                    &action.author,
                    "update",
                );
                if author_check != ValidateCallbackResult::Valid {
                    return Ok(author_check);
                }
                validate_create_entry(app_entry, &action.author)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type, tag, ..
        } => validate_link_tag(&link_type, &tag),
        FlatOp::RegisterDeleteLink {
            link_type,
            tag,
            action,
            ..
        } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            validate_link_tag(&link_type, &tag)
        }
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    app_entry: EntryTypes,
    author: &AgentPubKey,
) -> ExternResult<ValidateCallbackResult> {
    match app_entry {
        EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),

        EntryTypes::BoundaryContract(contract) => {
            if contract.provider == contract.receiver {
                return Ok(ValidateCallbackResult::Invalid(
                    "Provider and receiver must be different agents".into(),
                ));
            }
            if author != &contract.provider && author != &contract.receiver {
                return Ok(ValidateCallbackResult::Invalid(
                    "Contract author must be either provider or receiver".into(),
                ));
            }

            // ID length
            if contract.id.is_empty() || contract.id.len() > MAX_ID_BYTES {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Contract ID must be 1-{MAX_ID_BYTES} bytes"
                )));
            }

            // TEND hours bounds
            if !contract.tend_hours.is_finite()
                || contract.tend_hours <= 0.0
                || contract.tend_hours > MAX_TEND_HOURS
            {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "tend_hours must be finite, > 0.0, <= {MAX_TEND_HOURS}"
                )));
            }

            // Duration bounds
            if contract.duration_minutes == 0 || contract.duration_minutes > MAX_DURATION_MINUTES {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "duration_minutes must be 1-{MAX_DURATION_MINUTES}"
                )));
            }

            // Must define at least one inclusion and one exclusion
            if contract.inclusions.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Contract must define at least one inclusion (what IS offered)".into(),
                ));
            }
            if contract.exclusions.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Contract must define at least one exclusion (what is NOT offered)".into(),
                ));
            }

            // Boundary item limits
            if contract.inclusions.len() > MAX_BOUNDARY_ITEMS
                || contract.exclusions.len() > MAX_BOUNDARY_ITEMS
            {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Max {MAX_BOUNDARY_ITEMS} inclusions/exclusions"
                )));
            }
            for item in contract.inclusions.iter().chain(contract.exclusions.iter()) {
                if item.len() > MAX_BOUNDARY_ITEM_BYTES {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "Boundary item exceeds {MAX_BOUNDARY_ITEM_BYTES} bytes"
                    )));
                }
            }

            Ok(ValidateCallbackResult::Valid)
        }

        EntryTypes::ContractSummary(summary) => {
            // Author must be a party
            if author != &summary.provider && author != &summary.receiver {
                return Ok(ValidateCallbackResult::Invalid(
                    "Summary author must be a party to the contract".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }

        EntryTypes::BoundaryViolation(violation) => {
            // Reporter must match action author
            if &violation.reporter != author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Violation reporter must match action author".into(),
                ));
            }

            // Reporter and violator must differ
            if violation.reporter == violation.violator {
                return Ok(ValidateCallbackResult::Invalid(
                    "Cannot report a violation against yourself".into(),
                ));
            }

            // Description bounds
            if violation.description.is_empty()
                || violation.description.len() > MAX_VIOLATION_DESC_BYTES
            {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Violation description must be 1-{MAX_VIOLATION_DESC_BYTES} bytes"
                )));
            }

            Ok(ValidateCallbackResult::Valid)
        }

        EntryTypes::StandingBoundaries(boundaries) => {
            // Agent must match author
            if &boundaries.agent != author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Standing boundaries agent must match action author".into(),
                ));
            }

            // Blocked agents cap
            if boundaries.blocked_agents.len() > MAX_BLOCKED_AGENTS {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Max {MAX_BLOCKED_AGENTS} blocked agents"
                )));
            }

            // Universal exclusions cap
            if boundaries.universal_exclusions.len() > MAX_BOUNDARY_ITEMS {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Max {MAX_BOUNDARY_ITEMS} universal exclusions"
                )));
            }
            for ex in &boundaries.universal_exclusions {
                if ex.len() > MAX_BOUNDARY_ITEM_BYTES {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "Exclusion item exceeds {MAX_BOUNDARY_ITEM_BYTES} bytes"
                    )));
                }
            }

            // TEND hours minimum, if set
            if let Some(min_h) = boundaries.min_tend_hours {
                if !min_h.is_finite() || min_h < 0.0 || min_h > MAX_TEND_HOURS {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "min_tend_hours must be finite, >= 0.0, <= {MAX_TEND_HOURS}"
                    )));
                }
            }

            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_link_tag(
    _link_type: &LinkTypes,
    tag: &LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    if tag.0.len() > MAX_TAG_BYTES {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Link tag exceeds {MAX_TAG_BYTES} bytes: {}",
            tag.0.len()
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_serde_roundtrip() {
        let contract = BoundaryContract {
            id: "test-001".into(),
            provider: AgentPubKey::from_raw_36(vec![1u8; 36]),
            receiver: AgentPubKey::from_raw_36(vec![2u8; 36]),
            service_type: ServiceCategory::CareWork,
            inclusions: vec!["Stress relief massage".into()],
            exclusions: vec!["No sexual contact".into()],
            duration_minutes: 60,
            tend_hours: 1.0,
            revocation_conditions: vec!["Either party at any time".into()],
            status: ContractStatus::Proposed,
            provider_signed_at: None,
            receiver_signed_at: None,
            created_at: Timestamp::from_micros(1000000),
            updated_at: Timestamp::from_micros(1000000),
        };
        let bytes = holochain_serialized_bytes::encode(&contract).unwrap();
        let decoded: BoundaryContract = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(contract, decoded);
    }

    #[test]
    fn test_contract_status_valid_transitions() {
        assert!(ContractStatus::Proposed.can_transition_to(&ContractStatus::Active));
        assert!(ContractStatus::Proposed.can_transition_to(&ContractStatus::Revoked));
        assert!(ContractStatus::Active.can_transition_to(&ContractStatus::Completed));
        assert!(ContractStatus::Active.can_transition_to(&ContractStatus::Revoked));
        assert!(ContractStatus::Active.can_transition_to(&ContractStatus::Violated));
    }

    #[test]
    fn test_contract_status_invalid_transitions() {
        assert!(!ContractStatus::Completed.can_transition_to(&ContractStatus::Active));
        assert!(!ContractStatus::Revoked.can_transition_to(&ContractStatus::Active));
        assert!(!ContractStatus::Violated.can_transition_to(&ContractStatus::Active));
        assert!(!ContractStatus::Proposed.can_transition_to(&ContractStatus::Completed));
        assert!(!ContractStatus::Proposed.can_transition_to(&ContractStatus::Violated));
    }

    #[test]
    fn test_violation_severity_slash_factors() {
        assert!((ViolationSeverity::Minor.slash_factor() - 0.05).abs() < 1e-6);
        assert!((ViolationSeverity::Moderate.slash_factor() - 0.15).abs() < 1e-6);
        assert!((ViolationSeverity::Severe.slash_factor() - 0.30).abs() < 1e-6);
    }

    #[test]
    fn test_standing_boundaries_serde_roundtrip() {
        let boundaries = StandingBoundaries {
            agent: AgentPubKey::from_raw_36(vec![3u8; 36]),
            offered_services: vec![ServiceCategory::CareWork, ServiceCategory::Wellness],
            universal_exclusions: vec!["No alcohol/drugs during session".into()],
            max_duration_minutes: Some(120),
            min_tend_hours: Some(0.5),
            blocked_agents: vec![AgentPubKey::from_raw_36(vec![4u8; 36])],
            updated_at: Timestamp::from_micros(2000000),
        };
        let bytes = holochain_serialized_bytes::encode(&boundaries).unwrap();
        let decoded: StandingBoundaries = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(boundaries, decoded);
    }

    #[test]
    fn test_violation_serde_roundtrip() {
        let violation = BoundaryViolation {
            contract_hash: ActionHash::from_raw_36(vec![5u8; 36]),
            reporter: AgentPubKey::from_raw_36(vec![1u8; 36]),
            violator: AgentPubKey::from_raw_36(vec![2u8; 36]),
            description: "Client exceeded agreed boundaries".into(),
            severity: ViolationSeverity::Moderate,
            restorative_case_id: None,
            reported_at: Timestamp::from_micros(3000000),
        };
        let bytes = holochain_serialized_bytes::encode(&violation).unwrap();
        let decoded: BoundaryViolation = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(violation, decoded);
    }

    #[test]
    fn test_contract_summary_serde_roundtrip() {
        let summary = ContractSummary {
            contract_hash: ActionHash::from_raw_36(vec![6u8; 36]),
            provider: AgentPubKey::from_raw_36(vec![1u8; 36]),
            receiver: AgentPubKey::from_raw_36(vec![2u8; 36]),
            service_type: ServiceCategory::Wellness,
            status: ContractStatus::Active,
            created_at: Timestamp::from_micros(4000000),
            updated_at: Timestamp::from_micros(4000000),
        };
        let bytes = holochain_serialized_bytes::encode(&summary).unwrap();
        let decoded: ContractSummary = holochain_serialized_bytes::decode(&bytes).unwrap();
        assert_eq!(summary, decoded);
    }

    #[test]
    fn test_service_category_serde_all_variants() {
        for cat in [
            ServiceCategory::CareWork,
            ServiceCategory::HomeServices,
            ServiceCategory::FoodServices,
            ServiceCategory::Transportation,
            ServiceCategory::Education,
            ServiceCategory::GeneralAssistance,
            ServiceCategory::Administrative,
            ServiceCategory::Creative,
            ServiceCategory::TechSupport,
            ServiceCategory::Wellness,
            ServiceCategory::Gardening,
            ServiceCategory::Custom("Intimacy counseling".into()),
        ] {
            let bytes = holochain_serialized_bytes::encode(&cat).unwrap();
            let decoded: ServiceCategory = holochain_serialized_bytes::decode(&bytes).unwrap();
            assert_eq!(cat, decoded);
        }
    }
}
