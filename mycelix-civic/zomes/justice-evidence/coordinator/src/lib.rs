//! Evidence Coordinator Zome
use hdk::prelude::*;
use justice_evidence_integrity::*;

/// Create a deterministic anchor hash from a string
fn anchor_hash(s: &str) -> ExternResult<EntryHash> {
    let hash = holo_hash::blake2b_256(s.as_bytes());
    Ok(EntryHash::from_raw_32(hash.to_vec()))
}

/// Helper to get records from links
fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

#[hdk_extern]
pub fn submit_evidence(evidence: Evidence) -> ExternResult<Record> {
    if evidence.title.is_empty() || evidence.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Title must be 1-256 characters".into())));
    }
    if evidence.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Description must be under 4096 characters".into())));
    }
    if evidence.complaint_id.is_empty() || evidence.complaint_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Complaint ID must be 1-256 characters".into())));
    }
    if evidence.submitter.is_empty() || evidence.submitter.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Submitter must be 1-256 characters".into())));
    }
    let action_hash = create_entry(&EntryTypes::Evidence(evidence.clone()))?;
    create_link(
        anchor_hash(&format!("complaint:{}", evidence.complaint_id))?,
        action_hash.clone(),
        LinkTypes::ComplaintToEvidence,
        (),
    )?;
    create_link(
        anchor_hash(&format!("submitter:{}", evidence.submitter))?,
        action_hash.clone(),
        LinkTypes::SubmitterToEvidence,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[hdk_extern]
pub fn get_complaint_evidence(complaint_id: String) -> ExternResult<Vec<Record>> {
    if complaint_id.is_empty() || complaint_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Complaint ID must be 1-256 characters".into())));
    }
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&format!("complaint:{}", complaint_id))?, LinkTypes::ComplaintToEvidence)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Verify evidence (by a juror or arbitrator)
#[hdk_extern]
pub fn verify_evidence(input: VerifyEvidenceInput) -> ExternResult<Record> {
    if input.evidence_id.is_empty() || input.evidence_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Evidence ID must be 1-256 characters".into())));
    }
    if input.verifier.is_empty() || input.verifier.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Verifier must be 1-256 characters".into())));
    }
    if let Some(ref notes) = input.notes {
        if notes.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest("Notes must be under 4096 characters".into())));
        }
    }
    let verification = EvidenceVerification {
        id: format!("verification:{}:{}", input.evidence_id, sys_time()?.as_micros()),
        evidence_id: input.evidence_id.clone(),
        verifier: input.verifier.clone(),
        status: input.status,
        notes: input.notes,
        verified_at: sys_time()?,
    };
    let action_hash = create_entry(&EntryTypes::EvidenceVerification(verification))?;
    create_link(
        anchor_hash(&format!("evidence:{}", input.evidence_id))?,
        action_hash.clone(),
        LinkTypes::EvidenceToVerification,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyEvidenceInput {
    pub evidence_id: String,
    pub verifier: String,
    pub status: VerificationStatus,
    pub notes: Option<String>,
}

/// Dispute evidence (challenge its validity)
#[hdk_extern]
pub fn dispute_evidence(input: DisputeEvidenceInput) -> ExternResult<Record> {
    if input.evidence_id.is_empty() || input.evidence_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Evidence ID must be 1-256 characters".into())));
    }
    if input.disputant.is_empty() || input.disputant.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Disputant must be 1-256 characters".into())));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Reason must be 1-4096 characters".into())));
    }
    let dispute = EvidenceDispute {
        id: format!("dispute:{}:{}", input.evidence_id, sys_time()?.as_micros()),
        evidence_id: input.evidence_id.clone(),
        disputant: input.disputant.clone(),
        reason: input.reason,
        created_at: sys_time()?,
        resolved: false,
    };
    let action_hash = create_entry(&EntryTypes::EvidenceDispute(dispute))?;
    create_link(
        anchor_hash(&format!("evidence:{}", input.evidence_id))?,
        action_hash.clone(),
        LinkTypes::EvidenceToDispute,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DisputeEvidenceInput {
    pub evidence_id: String,
    pub disputant: String,
    pub reason: String,
}

/// Get all verifications for an evidence item
#[hdk_extern]
pub fn get_evidence_verifications(evidence_id: String) -> ExternResult<Vec<Record>> {
    if evidence_id.is_empty() || evidence_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Evidence ID must be 1-256 characters".into())));
    }
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&format!("evidence:{}", evidence_id))?, LinkTypes::EvidenceToVerification)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get disputes for an evidence item
#[hdk_extern]
pub fn get_evidence_disputes(evidence_id: String) -> ExternResult<Vec<Record>> {
    if evidence_id.is_empty() || evidence_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Evidence ID must be 1-256 characters".into())));
    }
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&format!("evidence:{}", evidence_id))?, LinkTypes::EvidenceToDispute)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all evidence submitted by a party
#[hdk_extern]
pub fn get_evidence_by_submitter(submitter: String) -> ExternResult<Vec<Record>> {
    if submitter.is_empty() || submitter.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Submitter must be 1-256 characters".into())));
    }
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&format!("submitter:{}", submitter))?, LinkTypes::SubmitterToEvidence)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
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
    fn verify_evidence_input_serde_roundtrip() {
        let input = VerifyEvidenceInput {
            evidence_id: "ev-1".to_string(),
            verifier: "did:example:juror1".to_string(),
            status: VerificationStatus::Verified,
            notes: Some("Verified via chain analysis".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VerifyEvidenceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.evidence_id, "ev-1");
        assert_eq!(decoded.verifier, "did:example:juror1");
        assert_eq!(decoded.status, VerificationStatus::Verified);
        assert_eq!(decoded.notes, Some("Verified via chain analysis".to_string()));
    }

    #[test]
    fn verify_evidence_input_no_notes_serde() {
        let input = VerifyEvidenceInput {
            evidence_id: "ev-2".to_string(),
            verifier: "did:example:juror2".to_string(),
            status: VerificationStatus::Pending,
            notes: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: VerifyEvidenceInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.notes.is_none());
        assert_eq!(decoded.status, VerificationStatus::Pending);
    }

    #[test]
    fn dispute_evidence_input_serde_roundtrip() {
        let input = DisputeEvidenceInput {
            evidence_id: "ev-1".to_string(),
            disputant: "did:example:bob".to_string(),
            reason: "Document has been altered".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DisputeEvidenceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.evidence_id, "ev-1");
        assert_eq!(decoded.disputant, "did:example:bob");
        assert_eq!(decoded.reason, "Document has been altered");
    }

    // ========================================================================
    // Integrity enum serde tests (all variants)
    // ========================================================================

    #[test]
    fn evidence_type_all_variants_serde() {
        let variants = vec![
            EvidenceType::Document,
            EvidenceType::Testimony,
            EvidenceType::Transaction,
            EvidenceType::CrossReference,
            EvidenceType::Media,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: EvidenceType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn verification_status_all_variants_serde() {
        let variants = vec![
            VerificationStatus::Verified,
            VerificationStatus::Challenged,
            VerificationStatus::Rejected,
            VerificationStatus::Pending,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: VerificationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // Evidence entry struct serde roundtrip tests
    // ========================================================================

    #[test]
    fn evidence_serde_roundtrip() {
        let evidence = Evidence {
            id: "ev-1".to_string(),
            complaint_id: "case-1".to_string(),
            submitter: "did:example:alice".to_string(),
            evidence_type: EvidenceType::Document,
            title: "Contract PDF".to_string(),
            description: "Original signed contract".to_string(),
            content_hash: "sha256:abc123".to_string(),
            encrypted_content: None,
            submitted: ts(),
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let decoded: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "ev-1");
        assert_eq!(decoded.complaint_id, "case-1");
        assert_eq!(decoded.submitter, "did:example:alice");
        assert_eq!(decoded.evidence_type, EvidenceType::Document);
        assert_eq!(decoded.title, "Contract PDF");
        assert_eq!(decoded.content_hash, "sha256:abc123");
        assert!(decoded.encrypted_content.is_none());
    }

    #[test]
    fn evidence_with_encrypted_content_serde() {
        let evidence = Evidence {
            id: "ev-2".to_string(),
            complaint_id: "case-2".to_string(),
            submitter: "did:example:bob".to_string(),
            evidence_type: EvidenceType::Media,
            title: "Video Evidence".to_string(),
            description: "Encrypted surveillance footage".to_string(),
            content_hash: "sha256:def456".to_string(),
            encrypted_content: Some("base64-encrypted-data".to_string()),
            submitted: ts(),
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let decoded: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.encrypted_content, Some("base64-encrypted-data".to_string()));
    }
}
