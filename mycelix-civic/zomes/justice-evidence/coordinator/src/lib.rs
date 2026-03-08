//! Evidence Coordinator Zome
use hdk::prelude::*;
use justice_evidence_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_proposal, requirement_for_voting, GovernanceEligibility,
    GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("civic_bridge", requirement, action_name)
}

/// Create a deterministic anchor hash from a string
fn anchor_hash(s: &str) -> ExternResult<EntryHash> {
    let hash = holo_hash::blake2b_256(s.as_bytes());
    Ok(EntryHash::from_raw_32(hash.to_vec()))
}

/// Helper to get records from links
fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

#[hdk_extern]
pub fn submit_evidence(evidence: Evidence) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "submit_evidence")?;
    if evidence.title.is_empty() || evidence.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if evidence.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be under 4096 characters".into()
        )));
    }
    if evidence.complaint_id.is_empty() || evidence.complaint_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Complaint ID must be 1-256 characters".into()
        )));
    }
    if evidence.submitter.is_empty() || evidence.submitter.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Submitter must be 1-256 characters".into()
        )));
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Complaint ID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("complaint:{}", complaint_id))?,
            LinkTypes::ComplaintToEvidence,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Verify evidence (by a juror or arbitrator)
#[hdk_extern]
pub fn verify_evidence(input: VerifyEvidenceInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_voting(), "verify_evidence")?;
    if input.evidence_id.is_empty() || input.evidence_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Evidence ID must be 1-256 characters".into()
        )));
    }
    if input.verifier.is_empty() || input.verifier.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Verifier must be 1-256 characters".into()
        )));
    }
    if let Some(ref notes) = input.notes {
        if notes.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Notes must be under 4096 characters".into()
            )));
        }
    }
    let verification = EvidenceVerification {
        id: format!(
            "verification:{}:{}",
            input.evidence_id,
            sys_time()?.as_micros()
        ),
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "dispute_evidence")?;
    if input.evidence_id.is_empty() || input.evidence_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Evidence ID must be 1-256 characters".into()
        )));
    }
    if input.disputant.is_empty() || input.disputant.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Disputant must be 1-256 characters".into()
        )));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-4096 characters".into()
        )));
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Evidence ID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("evidence:{}", evidence_id))?,
            LinkTypes::EvidenceToVerification,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get disputes for an evidence item
#[hdk_extern]
pub fn get_evidence_disputes(evidence_id: String) -> ExternResult<Vec<Record>> {
    if evidence_id.is_empty() || evidence_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Evidence ID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("evidence:{}", evidence_id))?,
            LinkTypes::EvidenceToDispute,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all evidence submitted by a party
#[hdk_extern]
pub fn get_evidence_by_submitter(submitter: String) -> ExternResult<Vec<Record>> {
    if submitter.is_empty() || submitter.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Submitter must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("submitter:{}", submitter))?,
            LinkTypes::SubmitterToEvidence,
        )?,
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
        assert_eq!(
            decoded.notes,
            Some("Verified via chain analysis".to_string())
        );
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
        assert_eq!(
            decoded.encrypted_content,
            Some("base64-encrypted-data".to_string())
        );
    }

    // ========================================================================
    // EvidenceVerification serde roundtrip tests
    // ========================================================================

    #[test]
    fn evidence_verification_full_serde_roundtrip() {
        let v = EvidenceVerification {
            id: "ver-1".to_string(),
            evidence_id: "ev-1".to_string(),
            verifier: "did:example:juror1".to_string(),
            status: VerificationStatus::Verified,
            notes: Some("Confirmed authenticity".to_string()),
            verified_at: Timestamp::from_micros(1_000_000),
        };
        let json = serde_json::to_string(&v).unwrap();
        let decoded: EvidenceVerification = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "ver-1");
        assert_eq!(decoded.evidence_id, "ev-1");
        assert_eq!(decoded.verifier, "did:example:juror1");
        assert_eq!(decoded.status, VerificationStatus::Verified);
        assert_eq!(decoded.notes, Some("Confirmed authenticity".to_string()));
    }

    #[test]
    fn evidence_verification_no_notes_serde() {
        let v = EvidenceVerification {
            id: "ver-2".to_string(),
            evidence_id: "ev-2".to_string(),
            verifier: "did:example:juror2".to_string(),
            status: VerificationStatus::Pending,
            notes: None,
            verified_at: ts(),
        };
        let json = serde_json::to_string(&v).unwrap();
        let decoded: EvidenceVerification = serde_json::from_str(&json).unwrap();
        assert!(decoded.notes.is_none());
        assert_eq!(decoded.status, VerificationStatus::Pending);
    }

    // ========================================================================
    // EvidenceDispute serde roundtrip tests
    // ========================================================================

    #[test]
    fn evidence_dispute_full_serde_roundtrip() {
        let d = EvidenceDispute {
            id: "disp-1".to_string(),
            evidence_id: "ev-1".to_string(),
            disputant: "did:example:bob".to_string(),
            reason: "Document appears to be forged".to_string(),
            created_at: Timestamp::from_micros(500_000),
            resolved: false,
        };
        let json = serde_json::to_string(&d).unwrap();
        let decoded: EvidenceDispute = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "disp-1");
        assert_eq!(decoded.evidence_id, "ev-1");
        assert_eq!(decoded.disputant, "did:example:bob");
        assert_eq!(decoded.reason, "Document appears to be forged");
        assert!(!decoded.resolved);
    }

    #[test]
    fn evidence_dispute_resolved_true_serde() {
        let d = EvidenceDispute {
            id: "disp-2".to_string(),
            evidence_id: "ev-2".to_string(),
            disputant: "did:example:carol".to_string(),
            reason: "Timestamps inconsistent".to_string(),
            created_at: ts(),
            resolved: true,
        };
        let json = serde_json::to_string(&d).unwrap();
        let decoded: EvidenceDispute = serde_json::from_str(&json).unwrap();
        assert!(decoded.resolved);
    }

    // ========================================================================
    // Clone/equality tests
    // ========================================================================

    #[test]
    fn evidence_clone_equals_original() {
        let ev = Evidence {
            id: "ev-clone".to_string(),
            complaint_id: "case-clone".to_string(),
            submitter: "did:example:alice".to_string(),
            evidence_type: EvidenceType::Document,
            title: "Clone Test".to_string(),
            description: "Test clone equality".to_string(),
            content_hash: "sha256:clone".to_string(),
            encrypted_content: None,
            submitted: ts(),
        };
        assert_eq!(ev, ev.clone());
    }

    #[test]
    fn evidence_verification_clone_equals_original() {
        let v = EvidenceVerification {
            id: "ver-clone".to_string(),
            evidence_id: "ev-1".to_string(),
            verifier: "did:example:verifier".to_string(),
            status: VerificationStatus::Challenged,
            notes: Some("Under review".to_string()),
            verified_at: ts(),
        };
        assert_eq!(v, v.clone());
    }

    #[test]
    fn evidence_dispute_clone_equals_original() {
        let d = EvidenceDispute {
            id: "disp-clone".to_string(),
            evidence_id: "ev-1".to_string(),
            disputant: "did:example:disputant".to_string(),
            reason: "Forged".to_string(),
            created_at: ts(),
            resolved: false,
        };
        assert_eq!(d, d.clone());
    }

    #[test]
    fn evidence_ne_different_type() {
        let a = Evidence {
            id: "ev-a".to_string(),
            complaint_id: "c".to_string(),
            submitter: "did:example:a".to_string(),
            evidence_type: EvidenceType::Document,
            title: "T".to_string(),
            description: "D".to_string(),
            content_hash: "h".to_string(),
            encrypted_content: None,
            submitted: ts(),
        };
        let mut b = a.clone();
        b.evidence_type = EvidenceType::Testimony;
        assert_ne!(a, b);
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn evidence_all_types_serde() {
        for et in [
            EvidenceType::Document,
            EvidenceType::Testimony,
            EvidenceType::Transaction,
            EvidenceType::CrossReference,
            EvidenceType::Media,
        ] {
            let ev = Evidence {
                id: "ev".to_string(),
                complaint_id: "c".to_string(),
                submitter: "did:example:s".to_string(),
                evidence_type: et.clone(),
                title: "T".to_string(),
                description: "D".to_string(),
                content_hash: "h".to_string(),
                encrypted_content: None,
                submitted: ts(),
            };
            let json = serde_json::to_string(&ev).unwrap();
            let decoded: Evidence = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.evidence_type, et);
        }
    }

    #[test]
    fn verify_evidence_input_all_statuses_serde() {
        for status in [
            VerificationStatus::Verified,
            VerificationStatus::Challenged,
            VerificationStatus::Rejected,
            VerificationStatus::Pending,
        ] {
            let input = VerifyEvidenceInput {
                evidence_id: "ev-1".to_string(),
                verifier: "did:example:v".to_string(),
                status: status.clone(),
                notes: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: VerifyEvidenceInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    #[test]
    fn evidence_unicode_fields_serde() {
        let ev = Evidence {
            id: "ev-unicode".to_string(),
            complaint_id: "\u{6848}\u{4EF6}-001".to_string(),
            submitter: "did:example:\u{30A2}\u{30EA}\u{30B9}".to_string(),
            evidence_type: EvidenceType::Document,
            title: "\u{5408}\u{540C}\u{66F8}".to_string(),
            description: "\u{0410}\u{0440}\u{0431}\u{0438}\u{0442}\u{0440}\u{0430}\u{0436}"
                .to_string(),
            content_hash: "sha256:abc".to_string(),
            encrypted_content: None,
            submitted: ts(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let decoded: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, "\u{5408}\u{540C}\u{66F8}");
        assert_eq!(decoded.complaint_id, "\u{6848}\u{4EF6}-001");
    }

    #[test]
    fn dispute_evidence_input_long_reason_serde() {
        let input = DisputeEvidenceInput {
            evidence_id: "ev-long".to_string(),
            disputant: "did:example:disputant".to_string(),
            reason: "x".repeat(4000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DisputeEvidenceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.reason.len(), 4000);
    }

    #[test]
    fn evidence_empty_description_serde() {
        let ev = Evidence {
            id: "ev-empty".to_string(),
            complaint_id: "case-1".to_string(),
            submitter: "did:example:alice".to_string(),
            evidence_type: EvidenceType::Transaction,
            title: "Title".to_string(),
            description: "".to_string(),
            content_hash: "sha256:abc".to_string(),
            encrypted_content: None,
            submitted: ts(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let decoded: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.description, "");
    }

    #[test]
    fn evidence_verification_all_statuses_roundtrip() {
        for status in [
            VerificationStatus::Verified,
            VerificationStatus::Challenged,
            VerificationStatus::Rejected,
            VerificationStatus::Pending,
        ] {
            let v = EvidenceVerification {
                id: "v".to_string(),
                evidence_id: "e".to_string(),
                verifier: "did:example:v".to_string(),
                status: status.clone(),
                notes: None,
                verified_at: ts(),
            };
            let json = serde_json::to_string(&v).unwrap();
            let decoded: EvidenceVerification = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    #[test]
    fn evidence_timestamp_max_serde() {
        let ev = Evidence {
            id: "ev-ts".to_string(),
            complaint_id: "c".to_string(),
            submitter: "did:example:s".to_string(),
            evidence_type: EvidenceType::Media,
            title: "T".to_string(),
            description: "D".to_string(),
            content_hash: "h".to_string(),
            encrypted_content: None,
            submitted: Timestamp::from_micros(i64::MAX),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let decoded: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.submitted, Timestamp::from_micros(i64::MAX));
    }
}
