// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Restorative Coordinator Zome
//!
//! Manages restorative justice circles, community healing processes,
//! and alternative dispute resolution focused on restoration over punishment.

use hdk::prelude::*;
use justice_restorative_integrity::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};
use mycelix_zome_helpers::{get_latest_record};


/// Create a restorative circle

#[hdk_extern]
pub fn create_circle(circle: RestorativeCircle) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "create_circle")?;
    let action_hash = create_entry(&EntryTypes::RestorativeCircle(circle.clone()))?;
    let record = get_latest_record(action_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Could not get created circle".into())
    ))?;

    // Link from case
    let case_path = Path::from(format!("cases/{}/circles", circle.case_id));
    create_link(
        case_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::CaseToRestorativeCircle,
        (),
    )?;

    // Index by facilitator
    let facilitator_path = Path::from(format!("facilitators/{}/circles", circle.facilitator));
    create_link(
        facilitator_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::CaseToRestorativeCircle,
        (),
    )?;

    // Index by status
    let status_path = Path::from(format!("circles/status/{:?}", circle.status));
    create_link(
        status_path.path_entry_hash()?,
        action_hash,
        LinkTypes::AllCases,
        (),
    )?;

    Ok(record)
}

/// Get restorative circle for a case
#[hdk_extern]
pub fn get_case_circle(case_id: String) -> ExternResult<Option<Record>> {
    let case_path = Path::from(format!("cases/{}/circles", case_id));
    let links = get_links(
        LinkQuery::try_new(
            case_path.path_entry_hash()?,
            LinkTypes::CaseToRestorativeCircle,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            return get(action_hash, GetOptions::default());
        }
    }

    Ok(None)
}

/// Record participant consent
#[hdk_extern]
pub fn record_consent(input: ConsentInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "record_consent")?;
    let record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Circle not found".into())
    ))?;

    let mut circle: RestorativeCircle = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    // Update participant consent
    for p in &mut circle.participants {
        if p.did == input.participant_did {
            p.consented = true;
        }
    }

    // Check if all consented - if so, move to Active
    let all_consented = circle.participants.iter().all(|p| p.consented);
    if all_consented && circle.status == CircleStatus::Forming {
        circle.status = CircleStatus::Active;
    }

    let action_hash = update_entry(input.circle_hash, &circle)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated circle".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConsentInput {
    pub circle_hash: ActionHash,
    pub participant_did: String,
}

/// Record a circle session
#[hdk_extern]
pub fn record_session(input: RecordSessionInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "record_session")?;
    let record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Circle not found".into())
    ))?;

    let mut circle: RestorativeCircle = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    // Add the session
    circle.sessions.push(input.session);

    // Update participant attendance
    for did in &input.attendees {
        for p in &mut circle.participants {
            if &p.did == did {
                let session_num = circle.sessions.len() as u32;
                p.attended_sessions.push(session_num);
            }
        }
    }

    let action_hash = update_entry(input.circle_hash, &circle)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated circle".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordSessionInput {
    pub circle_hash: ActionHash,
    pub session: CircleSession,
    pub attendees: Vec<String>,
}

/// Add an agreement to the circle
#[hdk_extern]
pub fn add_agreement(input: AddAgreementInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "add_agreement")?;
    let record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Circle not found".into())
    ))?;

    let mut circle: RestorativeCircle = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    circle.agreements.push(input.agreement);

    let action_hash = update_entry(input.circle_hash, &circle)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated circle".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddAgreementInput {
    pub circle_hash: ActionHash,
    pub agreement: String,
}

/// Update circle status
#[hdk_extern]
pub fn update_circle_status(input: UpdateCircleStatusInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_circle_status")?;
    let record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Circle not found".into())
    ))?;

    let mut circle: RestorativeCircle = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    circle.status = input.new_status;

    let action_hash = update_entry(input.circle_hash, &circle)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated circle".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCircleStatusInput {
    pub circle_hash: ActionHash,
    pub new_status: CircleStatus,
}

/// Complete the circle
#[hdk_extern]
pub fn complete_circle(input: CompleteCircleInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "complete_circle")?;
    let record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Circle not found".into())
    ))?;

    let mut circle: RestorativeCircle = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    circle.status = CircleStatus::Completed;

    // Add any final agreements
    for agreement in input.final_agreements {
        if !circle.agreements.contains(&agreement) {
            circle.agreements.push(agreement);
        }
    }

    let action_hash = update_entry(input.circle_hash, &circle)?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated circle".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteCircleInput {
    pub circle_hash: ActionHash,
    pub final_agreements: Vec<String>,
}

/// Get circles by facilitator
#[hdk_extern]
pub fn get_facilitator_circles(facilitator_did: String) -> ExternResult<Vec<Record>> {
    let facilitator_path = Path::from(format!("facilitators/{}/circles", facilitator_did));
    let links = get_links(
        LinkQuery::try_new(
            facilitator_path.path_entry_hash()?,
            LinkTypes::CaseToRestorativeCircle,
        )?,
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

/// Get circles by status
#[hdk_extern]
pub fn get_circles_by_status(status: CircleStatus) -> ExternResult<Vec<Record>> {
    let status_path = Path::from(format!("circles/status/{:?}", status));
    let links = get_links(
        LinkQuery::try_new(status_path.path_entry_hash()?, LinkTypes::AllCases)?,
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

// ============================================================================
// RESTITUTION — Structured Economic Justice via TEND
// ============================================================================

/// Input for creating a restitution agreement.
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRestitutionInput {
    pub circle_hash: ActionHash,
    pub agreement_id: String,
    pub description: String,
    pub restitution_type: RestitutionType,
    pub amount_hours: Option<f32>,
    pub from_did: String,
    pub to_did: String,
    pub deadline: Timestamp,
}

/// Input for recording a restitution payment.
#[derive(Serialize, Deserialize, Debug)]
pub struct RecordRestitutionPaymentInput {
    pub agreement_hash: ActionHash,
    pub tend_exchange_id: String,
}

/// Status of restitution completion for a circle.
#[derive(Serialize, Deserialize, Debug)]
pub struct RestitutionCompletionStatus {
    pub circle_hash: ActionHash,
    pub total_agreements: u32,
    pub completed: u32,
    pub pending: u32,
    pub defaulted: u32,
    pub all_complete: bool,
}

/// Create a structured restitution agreement linked to a restorative circle.
///
/// The circle must be in `AgreementReached` status or later. This replaces
/// the informal `agreements: Vec<String>` with tracked, verifiable terms
/// linked to the TEND mutual credit system.
#[hdk_extern]
pub fn create_restitution_agreement(input: CreateRestitutionInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "create_restitution_agreement")?;

    // Verify circle exists and is in appropriate status
    let circle_record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Circle not found".into())),
    )?;
    let circle: RestorativeCircle = circle_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    // Gate: restitution can only be created when agreement is reached
    match circle.status {
        CircleStatus::AgreementReached | CircleStatus::Monitoring | CircleStatus::Completed => {}
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Circle must be in AgreementReached/Monitoring/Completed status, got {:?}",
                circle.status
            ))));
        }
    }

    // Validate hours if provided
    if let Some(hours) = input.amount_hours {
        if hours <= 0.0 || !hours.is_finite() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Restitution hours must be positive and finite".into()
            )));
        }
    }

    let agreement = RestitutionAgreement {
        circle_hash: input.circle_hash.clone(),
        agreement_id: input.agreement_id,
        description: input.description,
        restitution_type: input.restitution_type,
        amount_hours: input.amount_hours,
        from_did: input.from_did,
        to_did: input.to_did,
        deadline: input.deadline,
        tend_exchange_id: None,
        status: RestitutionStatus::Pending,
        created_at: sys_time()?,
    };

    let action_hash = create_entry(&EntryTypes::RestitutionAgreement(agreement))?;

    // Link circle → restitution
    create_link(
        input.circle_hash,
        action_hash.clone(),
        LinkTypes::CircleToRestitution,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get created restitution agreement".into()
    )))
}

/// Record a TEND payment against a restitution agreement.
///
/// Updates the agreement status to `PaymentRecorded` and stores
/// the TEND exchange ID for cross-reference verification.
#[hdk_extern]
pub fn record_restitution_payment(input: RecordRestitutionPaymentInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "record_restitution_payment")?;

    let record = get(input.agreement_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest(
            "Restitution agreement not found".into()
        )),
    )?;
    let mut agreement: RestitutionAgreement = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid agreement entry".into()
        )))?;

    agreement.tend_exchange_id = Some(input.tend_exchange_id);
    agreement.status = RestitutionStatus::PaymentRecorded;

    let new_hash = update_entry(input.agreement_hash, &agreement)?;
    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not get updated agreement".into()
    )))
}

/// Check the restitution completion status for all agreements in a circle.
#[hdk_extern]
pub fn check_restitution_completion(circle_hash: ActionHash) -> ExternResult<RestitutionCompletionStatus> {
    let links = get_links(
        LinkQuery::try_new(circle_hash.clone(), LinkTypes::CircleToRestitution)?,
        GetStrategy::default(),
    )?;

    let mut total = 0u32;
    let mut completed = 0u32;
    let mut pending = 0u32;
    let mut defaulted = 0u32;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(agreement) = record
                    .entry()
                    .to_app_option::<RestitutionAgreement>()
                    .ok()
                    .flatten()
                {
                    total += 1;
                    match agreement.status {
                        RestitutionStatus::Completed | RestitutionStatus::Verified => {
                            completed += 1;
                        }
                        RestitutionStatus::Defaulted => defaulted += 1,
                        _ => pending += 1,
                    }
                }
            }
        }
    }

    Ok(RestitutionCompletionStatus {
        circle_hash,
        total_agreements: total,
        completed,
        pending,
        defaulted,
        all_complete: total > 0 && completed == total,
    })
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

    fn make_participant(did: &str, role: CircleRole, consented: bool) -> CircleParticipant {
        CircleParticipant {
            did: did.to_string(),
            role,
            consented,
            attended_sessions: vec![],
        }
    }

    fn make_circle() -> RestorativeCircle {
        RestorativeCircle {
            id: "circle-001".to_string(),
            case_id: "case-001".to_string(),
            facilitator: "did:example:facilitator".to_string(),
            participants: vec![
                make_participant("did:example:alice", CircleRole::HarmReceiver, false),
                make_participant("did:example:bob", CircleRole::HarmDoer, false),
            ],
            status: CircleStatus::Forming,
            sessions: vec![],
            agreements: vec![],
            created_at: ts(),
        }
    }

    // ========================================================================
    // Coordinator input struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn consent_input_serde_roundtrip() {
        let input = ConsentInput {
            circle_hash: fake_hash(),
            participant_did: "did:example:alice".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ConsentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.participant_did, "did:example:alice");
    }

    #[test]
    fn consent_input_unicode_did() {
        let input = ConsentInput {
            circle_hash: fake_hash(),
            participant_did: "did:example:\u{1F600}\u{4E16}\u{754C}".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ConsentInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.participant_did.contains('\u{1F600}'));
    }

    #[test]
    fn consent_input_empty_did() {
        let input = ConsentInput {
            circle_hash: fake_hash(),
            participant_did: String::new(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ConsentInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.participant_did.is_empty());
    }

    #[test]
    fn record_session_input_serde_roundtrip() {
        let session = CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec![
                "did:example:alice".to_string(),
                "did:example:bob".to_string(),
            ],
            summary: "Initial hearing of all parties".to_string(),
            next_steps: vec!["Schedule follow-up".to_string()],
        };
        let input = RecordSessionInput {
            circle_hash: fake_hash(),
            session,
            attendees: vec![
                "did:example:alice".to_string(),
                "did:example:bob".to_string(),
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordSessionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.session.session_number, 1);
        assert_eq!(decoded.session.summary, "Initial hearing of all parties");
        assert_eq!(decoded.attendees.len(), 2);
    }

    #[test]
    fn record_session_input_empty_attendees_and_next_steps() {
        let session = CircleSession {
            session_number: 0,
            held_at: ts(),
            attendees: vec![],
            summary: String::new(),
            next_steps: vec![],
        };
        let input = RecordSessionInput {
            circle_hash: fake_hash(),
            session,
            attendees: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordSessionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.session.session_number, 0);
        assert!(decoded.session.attendees.is_empty());
        assert!(decoded.session.next_steps.is_empty());
        assert!(decoded.attendees.is_empty());
    }

    #[test]
    fn record_session_input_max_session_number() {
        let session = CircleSession {
            session_number: u32::MAX,
            held_at: ts(),
            attendees: vec!["did:one".to_string()],
            summary: "Final session".to_string(),
            next_steps: vec![],
        };
        let input = RecordSessionInput {
            circle_hash: fake_hash(),
            session,
            attendees: vec!["did:one".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordSessionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.session.session_number, u32::MAX);
    }

    #[test]
    fn add_agreement_input_serde_roundtrip() {
        let input = AddAgreementInput {
            circle_hash: fake_hash(),
            agreement: "Both parties agree to mediated settlement terms".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddAgreementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.agreement,
            "Both parties agree to mediated settlement terms"
        );
    }

    #[test]
    fn add_agreement_input_empty_agreement() {
        let input = AddAgreementInput {
            circle_hash: fake_hash(),
            agreement: String::new(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddAgreementInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.agreement.is_empty());
    }

    #[test]
    fn update_circle_status_input_serde_roundtrip() {
        let input = UpdateCircleStatusInput {
            circle_hash: fake_hash(),
            new_status: CircleStatus::AgreementReached,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateCircleStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, CircleStatus::AgreementReached);
    }

    #[test]
    fn update_circle_status_all_variants_serde() {
        for status in [
            CircleStatus::Forming,
            CircleStatus::Active,
            CircleStatus::AgreementReached,
            CircleStatus::Monitoring,
            CircleStatus::Completed,
            CircleStatus::Discontinued,
        ] {
            let input = UpdateCircleStatusInput {
                circle_hash: fake_hash(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateCircleStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    #[test]
    fn complete_circle_input_serde_roundtrip() {
        let input = CompleteCircleInput {
            circle_hash: fake_hash(),
            final_agreements: vec![
                "Restitution payment of 500 credits".to_string(),
                "Public apology within 7 days".to_string(),
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.final_agreements.len(), 2);
        assert_eq!(
            decoded.final_agreements[0],
            "Restitution payment of 500 credits"
        );
    }

    #[test]
    fn complete_circle_input_empty_agreements_serde() {
        let input = CompleteCircleInput {
            circle_hash: fake_hash(),
            final_agreements: vec![],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteCircleInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.final_agreements.is_empty());
    }

    // ========================================================================
    // Integrity enum serde tests (all variants)
    // ========================================================================

    #[test]
    fn circle_status_all_variants_serde() {
        let variants = vec![
            CircleStatus::Forming,
            CircleStatus::Active,
            CircleStatus::AgreementReached,
            CircleStatus::Monitoring,
            CircleStatus::Completed,
            CircleStatus::Discontinued,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: CircleStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn circle_role_all_variants_serde() {
        let variants = vec![
            CircleRole::Facilitator,
            CircleRole::HarmDoer,
            CircleRole::HarmReceiver,
            CircleRole::CommunityMember,
            CircleRole::SupportPerson,
            CircleRole::Elder,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: CircleRole = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // Entry type serde roundtrip tests
    // ========================================================================

    #[test]
    fn restorative_circle_serde_roundtrip() {
        let circle = make_circle();
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: RestorativeCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "circle-001");
        assert_eq!(decoded.case_id, "case-001");
        assert_eq!(decoded.facilitator, "did:example:facilitator");
        assert_eq!(decoded.participants.len(), 2);
        assert_eq!(decoded.status, CircleStatus::Forming);
        assert!(decoded.sessions.is_empty());
        assert!(decoded.agreements.is_empty());
    }

    #[test]
    fn restorative_circle_with_sessions_and_agreements() {
        let mut circle = make_circle();
        circle.sessions.push(CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec!["did:example:alice".to_string()],
            summary: "Session 1 summary".to_string(),
            next_steps: vec!["Follow up".to_string()],
        });
        circle.agreements.push("Agreement one".to_string());
        circle.status = CircleStatus::AgreementReached;

        let json = serde_json::to_string(&circle).unwrap();
        let decoded: RestorativeCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.sessions.len(), 1);
        assert_eq!(decoded.agreements.len(), 1);
        assert_eq!(decoded.status, CircleStatus::AgreementReached);
    }

    #[test]
    fn circle_participant_serde_roundtrip() {
        let p = CircleParticipant {
            did: "did:example:alice".to_string(),
            role: CircleRole::HarmReceiver,
            consented: true,
            attended_sessions: vec![1, 2, 3],
        };
        let json = serde_json::to_string(&p).unwrap();
        let decoded: CircleParticipant = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.did, "did:example:alice");
        assert_eq!(decoded.role, CircleRole::HarmReceiver);
        assert!(decoded.consented);
        assert_eq!(decoded.attended_sessions, vec![1, 2, 3]);
    }

    #[test]
    fn circle_participant_empty_attended_sessions() {
        let p = CircleParticipant {
            did: String::new(),
            role: CircleRole::Elder,
            consented: false,
            attended_sessions: vec![],
        };
        let json = serde_json::to_string(&p).unwrap();
        let decoded: CircleParticipant = serde_json::from_str(&json).unwrap();
        assert!(decoded.did.is_empty());
        assert!(!decoded.consented);
        assert!(decoded.attended_sessions.is_empty());
    }

    #[test]
    fn circle_session_serde_roundtrip() {
        let session = CircleSession {
            session_number: 5,
            held_at: Timestamp::from_micros(1_000_000),
            attendees: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            summary: "Productive session with all parties present".to_string(),
            next_steps: vec!["Draft agreement".to_string(), "Schedule review".to_string()],
        };
        let json = serde_json::to_string(&session).unwrap();
        let decoded: CircleSession = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.session_number, 5);
        assert_eq!(decoded.attendees.len(), 3);
        assert_eq!(decoded.next_steps.len(), 2);
    }

    // ========================================================================
    // Clone / equality tests
    // ========================================================================

    #[test]
    fn restorative_circle_clone_equals_original() {
        let circle = make_circle();
        let cloned = circle.clone();
        assert_eq!(circle.id, cloned.id);
        assert_eq!(circle.case_id, cloned.case_id);
        assert_eq!(circle.status, cloned.status);
        assert_eq!(circle.participants.len(), cloned.participants.len());
    }

    #[test]
    fn circle_status_clone_preserves_variant() {
        let status = CircleStatus::Monitoring;
        let cloned = status.clone();
        assert_eq!(status, cloned);
    }

    // ========================================================================
    // Edge case / boundary value tests
    // ========================================================================

    #[test]
    fn restorative_circle_unicode_fields() {
        let circle = RestorativeCircle {
            id: "\u{1F600}".to_string(),
            case_id: "\u{4E16}\u{754C}".to_string(),
            facilitator: "\u{00E9}\u{00F1}".to_string(),
            participants: vec![],
            status: CircleStatus::Forming,
            sessions: vec![],
            agreements: vec!["\u{2764} Peace".to_string()],
            created_at: ts(),
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: RestorativeCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "\u{1F600}");
        assert_eq!(decoded.agreements[0], "\u{2764} Peace");
    }

    #[test]
    fn circle_session_unicode_summary() {
        let session = CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec![],
            summary: "\u{00C0}\u{00DF}\u{00E7} summary".to_string(),
            next_steps: vec!["\u{2714} complete".to_string()],
        };
        let json = serde_json::to_string(&session).unwrap();
        let decoded: CircleSession = serde_json::from_str(&json).unwrap();
        assert!(decoded.summary.contains('\u{00C0}'));
    }

    #[test]
    fn restorative_circle_many_participants() {
        let participants: Vec<CircleParticipant> = (0..100)
            .map(|i| {
                make_participant(
                    &format!("did:{}", i),
                    CircleRole::CommunityMember,
                    i % 2 == 0,
                )
            })
            .collect();
        let circle = RestorativeCircle {
            id: "large".to_string(),
            case_id: "case-big".to_string(),
            facilitator: "did:f".to_string(),
            participants,
            status: CircleStatus::Active,
            sessions: vec![],
            agreements: vec![],
            created_at: ts(),
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: RestorativeCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.participants.len(), 100);
        assert!(decoded.participants[0].consented);
        assert!(!decoded.participants[1].consented);
    }

    #[test]
    fn circle_participant_max_u32_session_number() {
        let p = CircleParticipant {
            did: "did:max".to_string(),
            role: CircleRole::Facilitator,
            consented: true,
            attended_sessions: vec![u32::MAX, 0, 1],
        };
        let json = serde_json::to_string(&p).unwrap();
        let decoded: CircleParticipant = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.attended_sessions[0], u32::MAX);
    }
}
