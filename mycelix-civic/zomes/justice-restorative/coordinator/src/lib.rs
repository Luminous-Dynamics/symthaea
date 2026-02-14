//! Restorative Coordinator Zome
//!
//! Manages restorative justice circles, community healing processes,
//! and alternative dispute resolution focused on restoration over punishment.

use hdk::prelude::*;
use justice_restorative_integrity::*;

/// Create a restorative circle
#[hdk_extern]
pub fn create_circle(circle: RestorativeCircle) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::RestorativeCircle(circle.clone()))?;
    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get created circle".into())))?;

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
        LinkQuery::try_new(case_path.path_entry_hash()?, LinkTypes::CaseToRestorativeCircle)?,
        GetStrategy::default()
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
    let record = get(input.circle_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Circle not found".into())))?;

    let mut circle: RestorativeCircle = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid circle entry".into())))?;

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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated circle".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConsentInput {
    pub circle_hash: ActionHash,
    pub participant_did: String,
}

/// Record a circle session
#[hdk_extern]
pub fn record_session(input: RecordSessionInput) -> ExternResult<Record> {
    let record = get(input.circle_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Circle not found".into())))?;

    let mut circle: RestorativeCircle = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid circle entry".into())))?;

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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated circle".into())))
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
    let record = get(input.circle_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Circle not found".into())))?;

    let mut circle: RestorativeCircle = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid circle entry".into())))?;

    circle.agreements.push(input.agreement);

    let action_hash = update_entry(input.circle_hash, &circle)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated circle".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddAgreementInput {
    pub circle_hash: ActionHash,
    pub agreement: String,
}

/// Update circle status
#[hdk_extern]
pub fn update_circle_status(input: UpdateCircleStatusInput) -> ExternResult<Record> {
    let record = get(input.circle_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Circle not found".into())))?;

    let mut circle: RestorativeCircle = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid circle entry".into())))?;

    circle.status = input.new_status;

    let action_hash = update_entry(input.circle_hash, &circle)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated circle".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCircleStatusInput {
    pub circle_hash: ActionHash,
    pub new_status: CircleStatus,
}

/// Complete the circle
#[hdk_extern]
pub fn complete_circle(input: CompleteCircleInput) -> ExternResult<Record> {
    let record = get(input.circle_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Circle not found".into())))?;

    let mut circle: RestorativeCircle = record.entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid circle entry".into())))?;

    circle.status = CircleStatus::Completed;

    // Add any final agreements
    for agreement in input.final_agreements {
        if !circle.agreements.contains(&agreement) {
            circle.agreements.push(agreement);
        }
    }

    let action_hash = update_entry(input.circle_hash, &circle)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not get updated circle".into())))
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
        LinkQuery::try_new(facilitator_path.path_entry_hash()?, LinkTypes::CaseToRestorativeCircle)?,
        GetStrategy::default()
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
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
        GetStrategy::default()
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
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

    // ========================================================================
    // Coordinator input struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn consent_input_serde_roundtrip() {
        let input = ConsentInput {
            circle_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            participant_did: "did:example:alice".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ConsentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.participant_did, "did:example:alice");
    }

    #[test]
    fn record_session_input_serde_roundtrip() {
        let session = CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec!["did:example:alice".to_string(), "did:example:bob".to_string()],
            summary: "Initial hearing of all parties".to_string(),
            next_steps: vec!["Schedule follow-up".to_string()],
        };
        let input = RecordSessionInput {
            circle_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            session,
            attendees: vec!["did:example:alice".to_string(), "did:example:bob".to_string()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordSessionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.session.session_number, 1);
        assert_eq!(decoded.session.summary, "Initial hearing of all parties");
        assert_eq!(decoded.attendees.len(), 2);
    }

    #[test]
    fn add_agreement_input_serde_roundtrip() {
        let input = AddAgreementInput {
            circle_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            agreement: "Both parties agree to mediated settlement terms".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddAgreementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.agreement, "Both parties agree to mediated settlement terms");
    }

    #[test]
    fn update_circle_status_input_serde_roundtrip() {
        let input = UpdateCircleStatusInput {
            circle_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            new_status: CircleStatus::AgreementReached,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateCircleStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, CircleStatus::AgreementReached);
    }

    #[test]
    fn complete_circle_input_serde_roundtrip() {
        let input = CompleteCircleInput {
            circle_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            final_agreements: vec![
                "Restitution payment of 500 credits".to_string(),
                "Public apology within 7 days".to_string(),
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CompleteCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.final_agreements.len(), 2);
        assert_eq!(decoded.final_agreements[0], "Restitution payment of 500 credits");
    }

    #[test]
    fn complete_circle_input_empty_agreements_serde() {
        let input = CompleteCircleInput {
            circle_hash: ActionHash::from_raw_36(vec![0u8; 36]),
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
}
