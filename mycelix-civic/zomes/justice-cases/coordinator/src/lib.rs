// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cases Coordinator Zome
//!
//! Manages the full lifecycle of dispute cases including filing,
//! mediation, escalation, and closure.

use hdk::prelude::*;
use justice_cases_integrity::*;
use mycelix_bridge_common::{
    civic_requirement_proposal, GovernanceEligibility,
};
use mycelix_zome_helpers::{get_latest_record};


/// File a new case

#[hdk_extern]
pub fn file_case(case: Case) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "file_case")?;

    let action_hash = create_entry(&EntryTypes::Case(case.clone()))?;
    let record = get_latest_record(action_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created case".into())
    ))?;

    // Link from complainant
    let complainant_path = Path::from(format!("users/{}/cases", case.complainant));
    create_link(
        complainant_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::ComplainantToCases,
        (),
    )?;

    // Link from respondent
    let respondent_path = Path::from(format!("users/{}/cases", case.respondent));
    create_link(
        respondent_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::RespondentToCases,
        (),
    )?;

    // Link to all cases
    let all_cases_path = Path::from("cases/all");
    create_link(
        all_cases_path.path_entry_hash()?,
        action_hash,
        LinkTypes::AllCases,
        (),
    )?;

    Ok(record)
}

/// Get a case by its action hash
#[hdk_extern]
pub fn get_case(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Update case phase
#[hdk_extern]
pub fn update_case_phase(input: UpdatePhaseInput) -> ExternResult<Record> {
    let record = get(input.case_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Case not found".into())))?;

    let mut case: Case = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid case entry".into()
        )))?;

    case.phase = input.new_phase;
    case.updated_at = sys_time()?;
    case.phase_deadline = input.deadline;

    let action_hash = update_entry(input.case_hash, &case)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated case".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePhaseInput {
    pub case_hash: ActionHash,
    pub new_phase: CasePhase,
    pub deadline: Option<Timestamp>,
}

/// Update case status
#[hdk_extern]
pub fn update_case_status(input: UpdateStatusInput) -> ExternResult<Record> {
    let record = get(input.case_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Case not found".into())))?;

    let mut case: Case = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid case entry".into()
        )))?;

    case.status = input.new_status;
    case.updated_at = sys_time()?;

    let action_hash = update_entry(input.case_hash, &case)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated case".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateStatusInput {
    pub case_hash: ActionHash,
    pub new_status: CaseStatus,
}

/// Add a party to a case
#[hdk_extern]
pub fn add_party(input: AddPartyInput) -> ExternResult<Record> {
    let record = get(input.case_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Case not found".into())))?;

    let mut case: Case = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid case entry".into()
        )))?;

    let party = CaseParty {
        did: input.party_did,
        role: input.role,
        joined_at: sys_time()?,
    };

    case.parties.push(party);
    case.updated_at = sys_time()?;

    let action_hash = update_entry(input.case_hash, &case)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated case".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddPartyInput {
    pub case_hash: ActionHash,
    pub party_did: String,
    pub role: PartyRole,
}

/// Submit evidence for a case
#[hdk_extern]
pub fn submit_evidence(evidence: Evidence) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "submit_evidence")?;

    let action_hash = create_entry(&EntryTypes::Evidence(evidence.clone()))?;
    let record = get_latest_record(action_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created evidence".into())
    ))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/evidence", evidence.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash,
        LinkTypes::CaseToEvidence,
        (),
    )?;

    Ok(record)
}

/// Get all evidence for a case
#[hdk_extern]
pub fn get_case_evidence(case_id: String) -> ExternResult<Vec<Record>> {
    let case_path = Path::from(format!("cases/{}/evidence", case_id));
    let links = get_links(
        LinkQuery::try_new(case_path.path_entry_hash()?, LinkTypes::CaseToEvidence)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Initiate mediation for a case
#[hdk_extern]
pub fn initiate_mediation(mediation: Mediation) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Mediation(mediation.clone()))?;
    let record = get_latest_record(action_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created mediation".into())
    ))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/mediation", mediation.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash,
        LinkTypes::CaseToMediation,
        (),
    )?;

    Ok(record)
}

/// Get cases for a user (as complainant or respondent)
#[hdk_extern]
pub fn get_my_cases(did: String) -> ExternResult<Vec<Record>> {
    let user_path = Path::from(format!("users/{}/cases", did));
    let links = get_links(
        LinkQuery::try_new(user_path.path_entry_hash()?, LinkTypes::ComplainantToCases)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                records.push(record);
            }
        }
    }

    // Also get cases where user is respondent
    let respondent_links = get_links(
        LinkQuery::try_new(user_path.path_entry_hash()?, LinkTypes::RespondentToCases)?,
        GetStrategy::default(),
    )?;

    for link in respondent_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                // Avoid duplicates
                if !records
                    .iter()
                    .any(|r| r.action_address() == record.action_address())
                {
                    records.push(record);
                }
            }
        }
    }

    Ok(records)
}

/// Get all open cases
#[hdk_extern]
pub fn get_all_cases(_: ()) -> ExternResult<Vec<Record>> {
    let all_path = Path::from("cases/all");
    let links = get_links(
        LinkQuery::try_new(all_path.path_entry_hash()?, LinkTypes::AllCases)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn fake_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn make_context() -> CaseContext {
        CaseContext {
            happ: Some("mycelix-civic".to_string()),
            reference_id: Some("tx-123".to_string()),
            community: Some("Downtown DAO".to_string()),
            jurisdiction: Some("community-rules-v2".to_string()),
        }
    }

    fn make_case() -> Case {
        Case {
            id: "case-001".to_string(),
            title: "Property boundary dispute".to_string(),
            description: "Disagreement over shared garden boundary".to_string(),
            case_type: CaseType::PropertyDispute,
            complainant: "did:example:alice".to_string(),
            respondent: "did:example:bob".to_string(),
            parties: vec![],
            phase: CasePhase::Filed,
            status: CaseStatus::Active,
            severity: CaseSeverity::Moderate,
            context: make_context(),
            created_at: ts(),
            updated_at: ts(),
            phase_deadline: None,
        }
    }

    // ========================================================================
    // Coordinator input struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn update_phase_input_serde_roundtrip() {
        let input = UpdatePhaseInput {
            case_hash: fake_hash(),
            new_phase: CasePhase::Mediation,
            deadline: Some(ts()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdatePhaseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_phase, CasePhase::Mediation);
        assert!(decoded.deadline.is_some());
    }

    #[test]
    fn update_phase_input_no_deadline_serde() {
        let input = UpdatePhaseInput {
            case_hash: fake_hash(),
            new_phase: CasePhase::Closed,
            deadline: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdatePhaseInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_phase, CasePhase::Closed);
        assert!(decoded.deadline.is_none());
    }

    #[test]
    fn update_phase_input_all_phases() {
        for phase in [
            CasePhase::Filed,
            CasePhase::Negotiation,
            CasePhase::Mediation,
            CasePhase::Arbitration,
            CasePhase::Appeal,
            CasePhase::Enforcement,
            CasePhase::Closed,
        ] {
            let input = UpdatePhaseInput {
                case_hash: fake_hash(),
                new_phase: phase.clone(),
                deadline: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdatePhaseInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_phase, phase);
        }
    }

    #[test]
    fn update_status_input_serde_roundtrip() {
        let input = UpdateStatusInput {
            case_hash: fake_hash(),
            new_status: CaseStatus::Resolved,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, CaseStatus::Resolved);
    }

    #[test]
    fn update_status_input_all_statuses() {
        for status in [
            CaseStatus::Active,
            CaseStatus::OnHold,
            CaseStatus::AwaitingResponse,
            CaseStatus::InDeliberation,
            CaseStatus::DecisionRendered,
            CaseStatus::Enforcing,
            CaseStatus::Resolved,
            CaseStatus::Dismissed,
            CaseStatus::Withdrawn,
        ] {
            let input = UpdateStatusInput {
                case_hash: fake_hash(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    #[test]
    fn add_party_input_serde_roundtrip() {
        let input = AddPartyInput {
            case_hash: fake_hash(),
            party_did: "did:example:witness1".to_string(),
            role: PartyRole::Witness,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddPartyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.party_did, "did:example:witness1");
        assert_eq!(decoded.role, PartyRole::Witness);
    }

    #[test]
    fn add_party_input_empty_did() {
        let input = AddPartyInput {
            case_hash: fake_hash(),
            party_did: String::new(),
            role: PartyRole::Affected,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddPartyInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.party_did.is_empty());
    }

    #[test]
    fn add_party_input_unicode_did() {
        let input = AddPartyInput {
            case_hash: fake_hash(),
            party_did: "did:\u{00E9}\u{00F1}\u{00FC}".to_string(),
            role: PartyRole::Expert,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddPartyInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.party_did.contains('\u{00E9}'));
    }

    // ========================================================================
    // Integrity enum serde tests (all variants)
    // ========================================================================

    #[test]
    fn case_phase_all_variants_serde() {
        let variants = vec![
            CasePhase::Filed,
            CasePhase::Negotiation,
            CasePhase::Mediation,
            CasePhase::Arbitration,
            CasePhase::Appeal,
            CasePhase::Enforcement,
            CasePhase::Closed,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: CasePhase = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn case_status_all_variants_serde() {
        let variants = vec![
            CaseStatus::Active,
            CaseStatus::OnHold,
            CaseStatus::AwaitingResponse,
            CaseStatus::InDeliberation,
            CaseStatus::DecisionRendered,
            CaseStatus::Enforcing,
            CaseStatus::Resolved,
            CaseStatus::Dismissed,
            CaseStatus::Withdrawn,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: CaseStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn party_role_all_variants_serde() {
        let variants = vec![
            PartyRole::Complainant,
            PartyRole::Respondent,
            PartyRole::Witness,
            PartyRole::Expert,
            PartyRole::Intervenor,
            PartyRole::Affected,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: PartyRole = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn case_severity_all_variants_serde() {
        let variants = vec![
            CaseSeverity::Minor,
            CaseSeverity::Moderate,
            CaseSeverity::Serious,
            CaseSeverity::Critical,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: CaseSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn case_type_all_variants_serde() {
        let variants: Vec<CaseType> = vec![
            CaseType::ContractDispute,
            CaseType::ConductViolation,
            CaseType::PropertyDispute,
            CaseType::FinancialDispute,
            CaseType::GovernanceDispute,
            CaseType::IdentityDispute,
            CaseType::IPDispute,
            CaseType::Other {
                category: "Custom".to_string(),
            },
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: CaseType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn case_type_other_empty_category() {
        let ct = CaseType::Other {
            category: String::new(),
        };
        let json = serde_json::to_string(&ct).unwrap();
        let decoded: CaseType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ct);
    }

    #[test]
    fn case_type_other_unicode_category() {
        let ct = CaseType::Other {
            category: "\u{1F3E0} Housing".to_string(),
        };
        let json = serde_json::to_string(&ct).unwrap();
        let decoded: CaseType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ct);
    }

    // ========================================================================
    // Entry type serde roundtrip tests
    // ========================================================================

    #[test]
    fn case_serde_roundtrip() {
        let case = make_case();
        let json = serde_json::to_string(&case).unwrap();
        let decoded: Case = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "case-001");
        assert_eq!(decoded.title, "Property boundary dispute");
        assert_eq!(decoded.case_type, CaseType::PropertyDispute);
        assert_eq!(decoded.phase, CasePhase::Filed);
        assert_eq!(decoded.status, CaseStatus::Active);
        assert_eq!(decoded.severity, CaseSeverity::Moderate);
        assert!(decoded.phase_deadline.is_none());
        assert!(decoded.parties.is_empty());
    }

    #[test]
    fn case_with_deadline_and_parties() {
        let mut case = make_case();
        case.phase_deadline = Some(Timestamp::from_micros(86_400_000_000));
        case.parties.push(CaseParty {
            did: "did:example:witness".to_string(),
            role: PartyRole::Witness,
            joined_at: ts(),
        });
        let json = serde_json::to_string(&case).unwrap();
        let decoded: Case = serde_json::from_str(&json).unwrap();
        assert!(decoded.phase_deadline.is_some());
        assert_eq!(decoded.parties.len(), 1);
        assert_eq!(decoded.parties[0].role, PartyRole::Witness);
    }

    #[test]
    fn case_context_all_none() {
        let ctx = CaseContext {
            happ: None,
            reference_id: None,
            community: None,
            jurisdiction: None,
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let decoded: CaseContext = serde_json::from_str(&json).unwrap();
        assert!(decoded.happ.is_none());
        assert!(decoded.reference_id.is_none());
        assert!(decoded.community.is_none());
        assert!(decoded.jurisdiction.is_none());
    }

    #[test]
    fn case_context_all_some() {
        let ctx = make_context();
        let json = serde_json::to_string(&ctx).unwrap();
        let decoded: CaseContext = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.happ, Some("mycelix-civic".to_string()));
        assert_eq!(decoded.reference_id, Some("tx-123".to_string()));
    }

    #[test]
    fn case_party_serde_roundtrip() {
        let party = CaseParty {
            did: "did:example:expert42".to_string(),
            role: PartyRole::Expert,
            joined_at: Timestamp::from_micros(5_000_000),
        };
        let json = serde_json::to_string(&party).unwrap();
        let decoded: CaseParty = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.did, "did:example:expert42");
        assert_eq!(decoded.role, PartyRole::Expert);
    }

    // ========================================================================
    // Clone tests
    // ========================================================================

    #[test]
    fn case_clone_preserves_all_fields() {
        let case = make_case();
        let cloned = case.clone();
        assert_eq!(case.id, cloned.id);
        assert_eq!(case.title, cloned.title);
        assert_eq!(case.case_type, cloned.case_type);
        assert_eq!(case.phase, cloned.phase);
        assert_eq!(case.status, cloned.status);
        assert_eq!(case.severity, cloned.severity);
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn case_unicode_title_and_description() {
        let mut case = make_case();
        case.title = "\u{1F3DB} Dispute \u{2696}".to_string();
        case.description = "\u{4E89}\u{8BAE}\u{89E3}\u{51B3}".to_string();
        let json = serde_json::to_string(&case).unwrap();
        let decoded: Case = serde_json::from_str(&json).unwrap();
        assert!(decoded.title.contains('\u{2696}'));
        assert_eq!(decoded.description.len(), case.description.len());
    }

    #[test]
    fn case_empty_string_fields() {
        let case = Case {
            id: String::new(),
            title: String::new(),
            description: String::new(),
            case_type: CaseType::ContractDispute,
            complainant: String::new(),
            respondent: String::new(),
            parties: vec![],
            phase: CasePhase::Filed,
            status: CaseStatus::Active,
            severity: CaseSeverity::Minor,
            context: CaseContext {
                happ: None,
                reference_id: None,
                community: None,
                jurisdiction: None,
            },
            created_at: ts(),
            updated_at: ts(),
            phase_deadline: None,
        };
        let json = serde_json::to_string(&case).unwrap();
        let decoded: Case = serde_json::from_str(&json).unwrap();
        assert!(decoded.id.is_empty());
        assert!(decoded.title.is_empty());
    }

    #[test]
    fn case_many_parties() {
        let mut case = make_case();
        for i in 0..50 {
            case.parties.push(CaseParty {
                did: format!("did:example:party-{}", i),
                role: PartyRole::Affected,
                joined_at: ts(),
            });
        }
        let json = serde_json::to_string(&case).unwrap();
        let decoded: Case = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.parties.len(), 50);
    }
}
