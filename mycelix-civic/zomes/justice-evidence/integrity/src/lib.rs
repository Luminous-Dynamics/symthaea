//! Evidence Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Evidence {
    pub id: String,
    pub complaint_id: String,
    pub submitter: String,
    pub evidence_type: EvidenceType,
    pub title: String,
    pub description: String,
    pub content_hash: String,
    pub encrypted_content: Option<String>,
    pub submitted: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EvidenceType {
    Document,
    Testimony,
    Transaction,
    CrossReference,
    Media,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EvidenceVerification {
    pub id: String,
    pub evidence_id: String,
    pub verifier: String,
    pub status: VerificationStatus,
    pub notes: Option<String>,
    pub verified_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    Verified,
    Challenged,
    Rejected,
    Pending,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EvidenceDispute {
    pub id: String,
    pub evidence_id: String,
    pub disputant: String,
    pub reason: String,
    pub created_at: Timestamp,
    pub resolved: bool,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Evidence(Evidence),
    EvidenceVerification(EvidenceVerification),
    EvidenceDispute(EvidenceDispute),
}

#[hdk_link_types]
pub enum LinkTypes {
    ComplaintToEvidence,
    SubmitterToEvidence,
    EvidenceToVerification,
    EvidenceToDispute,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. })
        | FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Evidence(evidence) => validate_evidence(&evidence),
            EntryTypes::EvidenceVerification(v) => validate_evidence_verification(&v),
            EntryTypes::EvidenceDispute(d) => validate_evidence_dispute(&d),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_evidence(evidence: &Evidence) -> ExternResult<ValidateCallbackResult> {
    if !evidence.submitter.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Submitter must be a valid DID".into(),
        ));
    }
    if evidence.submitter.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Submitter DID too long (max 256 chars)".into(),
        ));
    }
    if evidence.content_hash.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash required".into(),
        ));
    }
    if evidence.content_hash.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash too long (max 256 chars)".into(),
        ));
    }
    if evidence.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Description cannot be empty".into(),
        ));
    }
    if evidence.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description too long (max 4096 chars)".into(),
        ));
    }
    if evidence.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence ID too long (max 256 chars)".into(),
        ));
    }
    if evidence.complaint_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Complaint ID too long (max 256 chars)".into(),
        ));
    }
    if evidence.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Title too long (max 256 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_evidence_verification(
    v: &EvidenceVerification,
) -> ExternResult<ValidateCallbackResult> {
    if !v.verifier.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Verifier must be a valid DID".into(),
        ));
    }
    if v.verifier.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Verifier DID too long (max 256 chars)".into(),
        ));
    }
    if v.evidence_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence ID required".into(),
        ));
    }
    if v.evidence_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence ID too long (max 256 chars)".into(),
        ));
    }
    if v.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Verification ID too long (max 256 chars)".into(),
        ));
    }
    if let Some(ref notes) = v.notes {
        if notes.len() > 8192 {
            return Ok(ValidateCallbackResult::Invalid(
                "Notes too long (max 8192 chars)".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_evidence_dispute(d: &EvidenceDispute) -> ExternResult<ValidateCallbackResult> {
    if !d.disputant.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Disputant must be a valid DID".into(),
        ));
    }
    if d.disputant.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Disputant DID too long (max 256 chars)".into(),
        ));
    }
    if d.evidence_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence ID required".into(),
        ));
    }
    if d.evidence_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence ID too long (max 256 chars)".into(),
        ));
    }
    if d.reason.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute reason required".into(),
        ));
    }
    if d.reason.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute reason too long (max 4096 chars)".into(),
        ));
    }
    if d.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute ID too long (max 256 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RESULT HELPERS
    // ========================================================================

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    // ========================================================================
    // DATA CONSTRUCTION HELPERS
    // ========================================================================

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn make_evidence() -> Evidence {
        Evidence {
            id: "ev-1".into(),
            complaint_id: "complaint-1".into(),
            submitter: "did:example:alice".into(),
            evidence_type: EvidenceType::Document,
            title: "Contract PDF".into(),
            description: "The original contract".into(),
            content_hash: "sha256:abc123def456".into(),
            encrypted_content: None,
            submitted: ts(),
        }
    }

    fn make_verification() -> EvidenceVerification {
        EvidenceVerification {
            id: "ver-1".into(),
            evidence_id: "ev-1".into(),
            verifier: "did:example:verifier".into(),
            status: VerificationStatus::Verified,
            notes: Some("Authenticity confirmed".into()),
            verified_at: ts(),
        }
    }

    fn make_dispute() -> EvidenceDispute {
        EvidenceDispute {
            id: "disp-1".into(),
            evidence_id: "ev-1".into(),
            disputant: "did:example:bob".into(),
            reason: "Document appears to be forged".into(),
            created_at: ts(),
            resolved: false,
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_evidence_type_all_variants() {
        let variants = vec![
            EvidenceType::Document,
            EvidenceType::Testimony,
            EvidenceType::Transaction,
            EvidenceType::CrossReference,
            EvidenceType::Media,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let back: EvidenceType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_verification_status_all_variants() {
        let variants = vec![
            VerificationStatus::Verified,
            VerificationStatus::Challenged,
            VerificationStatus::Rejected,
            VerificationStatus::Pending,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let back: VerificationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_evidence() {
        let ev = make_evidence();
        let json = serde_json::to_string(&ev).unwrap();
        let back: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn serde_roundtrip_evidence_with_encrypted_content() {
        let mut ev = make_evidence();
        ev.encrypted_content = Some("base64:encryptedblob==".into());
        let json = serde_json::to_string(&ev).unwrap();
        let back: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn serde_roundtrip_evidence_verification() {
        let v = make_verification();
        let json = serde_json::to_string(&v).unwrap();
        let back: EvidenceVerification = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn serde_roundtrip_evidence_verification_no_notes() {
        let mut v = make_verification();
        v.notes = None;
        let json = serde_json::to_string(&v).unwrap();
        let back: EvidenceVerification = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn serde_roundtrip_evidence_dispute() {
        let d = make_dispute();
        let json = serde_json::to_string(&d).unwrap();
        let back: EvidenceDispute = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn serde_roundtrip_evidence_dispute_resolved() {
        let mut d = make_dispute();
        d.resolved = true;
        let json = serde_json::to_string(&d).unwrap();
        let back: EvidenceDispute = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
        assert!(back.resolved);
    }

    // ========================================================================
    // VALIDATE_EVIDENCE TESTS
    // ========================================================================

    #[test]
    fn valid_evidence_passes() {
        let result = validate_evidence(&make_evidence());
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_submitter_not_did_rejected() {
        let mut ev = make_evidence();
        ev.submitter = "alice".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Submitter must be a valid DID");
    }

    #[test]
    fn evidence_empty_submitter_rejected() {
        let mut ev = make_evidence();
        ev.submitter = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_submitter_did_prefix_only_passes() {
        // "did:" alone satisfies starts_with("did:")
        let mut ev = make_evidence();
        ev.submitter = "did:".into();
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_empty_content_hash_rejected() {
        let mut ev = make_evidence();
        ev.content_hash = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Content hash required");
    }

    #[test]
    fn evidence_whitespace_only_content_hash_rejected() {
        let mut ev = make_evidence();
        ev.content_hash = "   ".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_submitter_check_precedes_hash_check() {
        // When both are invalid, submitter check triggers first
        let mut ev = make_evidence();
        ev.submitter = "not-a-did".into();
        ev.content_hash = "".into();
        let result = validate_evidence(&ev);
        assert_eq!(invalid_msg(&result), "Submitter must be a valid DID");
    }

    #[test]
    fn evidence_all_evidence_types_valid() {
        let types = vec![
            EvidenceType::Document,
            EvidenceType::Testimony,
            EvidenceType::Transaction,
            EvidenceType::CrossReference,
            EvidenceType::Media,
        ];
        for et in types {
            let mut ev = make_evidence();
            ev.evidence_type = et;
            assert!(is_valid(&validate_evidence(&ev)));
        }
    }

    #[test]
    fn evidence_with_encrypted_content_passes() {
        let mut ev = make_evidence();
        ev.encrypted_content = Some("encrypted-blob".into());
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_without_encrypted_content_passes() {
        let mut ev = make_evidence();
        ev.encrypted_content = None;
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // VALIDATE_EVIDENCE_VERIFICATION TESTS
    // ========================================================================

    #[test]
    fn valid_verification_passes() {
        let result = validate_evidence_verification(&make_verification());
        assert!(is_valid(&result));
    }

    #[test]
    fn verification_verifier_not_did_rejected() {
        let mut v = make_verification();
        v.verifier = "verifier-person".into();
        let result = validate_evidence_verification(&v);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Verifier must be a valid DID");
    }

    #[test]
    fn verification_empty_verifier_rejected() {
        let mut v = make_verification();
        v.verifier = "".into();
        let result = validate_evidence_verification(&v);
        assert!(is_invalid(&result));
    }

    #[test]
    fn verification_verifier_did_prefix_only_passes() {
        let mut v = make_verification();
        v.verifier = "did:".into();
        let result = validate_evidence_verification(&v);
        assert!(is_valid(&result));
    }

    #[test]
    fn verification_empty_evidence_id_rejected() {
        let mut v = make_verification();
        v.evidence_id = "".into();
        let result = validate_evidence_verification(&v);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Evidence ID required");
    }

    #[test]
    fn verification_verifier_check_precedes_evidence_id_check() {
        let mut v = make_verification();
        v.verifier = "not-did".into();
        v.evidence_id = "".into();
        let result = validate_evidence_verification(&v);
        assert_eq!(invalid_msg(&result), "Verifier must be a valid DID");
    }

    #[test]
    fn verification_all_statuses_valid() {
        let statuses = vec![
            VerificationStatus::Verified,
            VerificationStatus::Challenged,
            VerificationStatus::Rejected,
            VerificationStatus::Pending,
        ];
        for s in statuses {
            let mut v = make_verification();
            v.status = s;
            assert!(is_valid(&validate_evidence_verification(&v)));
        }
    }

    #[test]
    fn verification_with_notes_none_passes() {
        let mut v = make_verification();
        v.notes = None;
        let result = validate_evidence_verification(&v);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // VALIDATE_EVIDENCE_DISPUTE TESTS
    // ========================================================================

    #[test]
    fn valid_dispute_passes() {
        let result = validate_evidence_dispute(&make_dispute());
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_disputant_not_did_rejected() {
        let mut d = make_dispute();
        d.disputant = "bob".into();
        let result = validate_evidence_dispute(&d);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Disputant must be a valid DID");
    }

    #[test]
    fn dispute_empty_disputant_rejected() {
        let mut d = make_dispute();
        d.disputant = "".into();
        let result = validate_evidence_dispute(&d);
        assert!(is_invalid(&result));
    }

    #[test]
    fn dispute_disputant_did_prefix_only_passes() {
        let mut d = make_dispute();
        d.disputant = "did:".into();
        let result = validate_evidence_dispute(&d);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_empty_evidence_id_rejected() {
        let mut d = make_dispute();
        d.evidence_id = "".into();
        let result = validate_evidence_dispute(&d);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Evidence ID required");
    }

    #[test]
    fn dispute_empty_reason_rejected() {
        let mut d = make_dispute();
        d.reason = "".into();
        let result = validate_evidence_dispute(&d);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Dispute reason required");
    }

    #[test]
    fn dispute_whitespace_only_reason_rejected() {
        let mut d = make_dispute();
        d.reason = "   \t\n  ".into();
        let result = validate_evidence_dispute(&d);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Dispute reason required");
    }

    #[test]
    fn dispute_disputant_check_precedes_evidence_id_check() {
        let mut d = make_dispute();
        d.disputant = "not-did".into();
        d.evidence_id = "".into();
        d.reason = "".into();
        let result = validate_evidence_dispute(&d);
        assert_eq!(invalid_msg(&result), "Disputant must be a valid DID");
    }

    #[test]
    fn dispute_evidence_id_check_precedes_reason_check() {
        let mut d = make_dispute();
        d.evidence_id = "".into();
        d.reason = "".into();
        let result = validate_evidence_dispute(&d);
        assert_eq!(invalid_msg(&result), "Evidence ID required");
    }

    #[test]
    fn dispute_resolved_true_passes() {
        let mut d = make_dispute();
        d.resolved = true;
        let result = validate_evidence_dispute(&d);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_resolved_false_passes() {
        let d = make_dispute();
        assert!(!d.resolved);
        let result = validate_evidence_dispute(&d);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // EDGE CASES: UNICODE
    // ========================================================================

    #[test]
    fn evidence_unicode_title_passes() {
        let mut ev = make_evidence();
        ev.title = "\u{1F4DC} \u{5408}\u{540C}\u{66F8}".into(); // scroll emoji + Japanese "contract"
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_unicode_description_passes() {
        let mut ev = make_evidence();
        ev.description = "\u{0410}\u{0440}\u{0431}\u{0438}\u{0442}\u{0440}\u{0430}\u{0436} \u{0434}\u{043E}\u{043A}\u{0443}\u{043C}\u{0435}\u{043D}\u{0442}".into();
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_unicode_reason_passes() {
        let mut d = make_dispute();
        d.reason = "\u{C704}\u{C870} \u{BB38}\u{C11C}".into(); // Korean "forged document"
        let result = validate_evidence_dispute(&d);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_unicode_submitter_without_did_prefix_rejected() {
        let mut ev = make_evidence();
        ev.submitter = "\u{30C7}\u{30B8}\u{30BF}\u{30EB}:alice".into(); // Japanese "digital:alice"
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_did_with_unicode_method_passes() {
        let mut ev = make_evidence();
        ev.submitter = "did:\u{4F8B}:alice".into(); // did:<kanji for example>:alice
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // EDGE CASES: MAX / BOUNDARY VALUES
    // ========================================================================

    #[test]
    fn evidence_very_long_content_hash_rejected() {
        let mut ev = make_evidence();
        ev.content_hash = "a".repeat(10_000);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Content hash too long (max 256 chars)"
        );
    }

    #[test]
    fn evidence_content_hash_at_limit_passes() {
        let mut ev = make_evidence();
        ev.content_hash = "a".repeat(256);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_content_hash_over_limit_rejected() {
        let mut ev = make_evidence();
        ev.content_hash = "a".repeat(257);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Content hash too long (max 256 chars)"
        );
    }

    #[test]
    fn evidence_very_long_submitter_did_rejected() {
        let mut ev = make_evidence();
        ev.submitter = format!("did:example:{}", "x".repeat(10_000));
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Submitter DID too long (max 256 chars)"
        );
    }

    #[test]
    fn evidence_submitter_did_at_limit_passes() {
        let mut ev = make_evidence();
        ev.submitter = format!("did:{}", "x".repeat(252));
        assert_eq!(ev.submitter.len(), 256);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_description_empty_rejected() {
        let mut ev = make_evidence();
        ev.description = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Description cannot be empty");
    }

    #[test]
    fn evidence_description_whitespace_only_rejected() {
        let mut ev = make_evidence();
        ev.description = "   ".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Description cannot be empty");
    }

    #[test]
    fn evidence_description_at_limit_passes() {
        let mut ev = make_evidence();
        ev.description = "d".repeat(4096);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_description_over_limit_rejected() {
        let mut ev = make_evidence();
        ev.description = "d".repeat(4097);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Description too long (max 4096 chars)"
        );
    }

    #[test]
    fn evidence_title_at_limit_passes() {
        let mut ev = make_evidence();
        ev.title = "t".repeat(256);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_title_over_limit_rejected() {
        let mut ev = make_evidence();
        ev.title = "t".repeat(257);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Title too long (max 256 chars)");
    }

    #[test]
    fn dispute_very_long_reason_rejected() {
        let mut d = make_dispute();
        d.reason = "x".repeat(100_000);
        let result = validate_evidence_dispute(&d);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Dispute reason too long (max 4096 chars)"
        );
    }

    #[test]
    fn dispute_reason_at_limit_passes() {
        let mut d = make_dispute();
        d.reason = "r".repeat(4096);
        let result = validate_evidence_dispute(&d);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_reason_over_limit_rejected() {
        let mut d = make_dispute();
        d.reason = "r".repeat(4097);
        let result = validate_evidence_dispute(&d);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Dispute reason too long (max 4096 chars)"
        );
    }

    #[test]
    fn dispute_evidence_id_at_limit_passes() {
        let mut d = make_dispute();
        d.evidence_id = "e".repeat(256);
        let result = validate_evidence_dispute(&d);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_evidence_id_over_limit_rejected() {
        let mut d = make_dispute();
        d.evidence_id = "e".repeat(257);
        let result = validate_evidence_dispute(&d);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Evidence ID too long (max 256 chars)");
    }

    #[test]
    fn verification_very_long_evidence_id_rejected() {
        let mut v = make_verification();
        v.evidence_id = "ev-".to_string() + &"9".repeat(10_000);
        let result = validate_evidence_verification(&v);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Evidence ID too long (max 256 chars)");
    }

    #[test]
    fn verification_evidence_id_at_limit_passes() {
        let mut v = make_verification();
        v.evidence_id = "e".repeat(256);
        let result = validate_evidence_verification(&v);
        assert!(is_valid(&result));
    }

    #[test]
    fn verification_notes_at_limit_passes() {
        let mut v = make_verification();
        v.notes = Some("n".repeat(8192));
        let result = validate_evidence_verification(&v);
        assert!(is_valid(&result));
    }

    #[test]
    fn verification_notes_over_limit_rejected() {
        let mut v = make_verification();
        v.notes = Some("n".repeat(8193));
        let result = validate_evidence_verification(&v);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Notes too long (max 8192 chars)");
    }

    // ========================================================================
    // EDGE CASES: EMPTY COLLECTIONS / SPECIAL VALUES
    // ========================================================================

    #[test]
    fn evidence_empty_id_still_passes() {
        // id is not validated
        let mut ev = make_evidence();
        ev.id = "".into();
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_empty_complaint_id_still_passes() {
        // complaint_id is not validated
        let mut ev = make_evidence();
        ev.complaint_id = "".into();
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_empty_title_still_passes() {
        // title is not validated by validate_evidence
        let mut ev = make_evidence();
        ev.title = "".into();
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_empty_description_rejected() {
        let mut ev = make_evidence();
        ev.description = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Description cannot be empty");
    }

    #[test]
    fn verification_empty_id_still_passes() {
        let mut v = make_verification();
        v.id = "".into();
        let result = validate_evidence_verification(&v);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_empty_id_still_passes() {
        let mut d = make_dispute();
        d.id = "".into();
        let result = validate_evidence_dispute(&d);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_timestamp_zero_passes() {
        let ev = make_evidence();
        assert_eq!(ev.submitted, Timestamp::from_micros(0));
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_timestamp_max_passes() {
        let mut ev = make_evidence();
        ev.submitted = Timestamp::from_micros(i64::MAX);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_timestamp_negative_passes() {
        let mut ev = make_evidence();
        ev.submitted = Timestamp::from_micros(-1_000_000);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // CLONE / EQUALITY TESTS
    // ========================================================================

    #[test]
    fn evidence_clone_equals_original() {
        let ev = make_evidence();
        let cloned = ev.clone();
        assert_eq!(ev, cloned);
    }

    #[test]
    fn evidence_verification_clone_equals_original() {
        let v = make_verification();
        let cloned = v.clone();
        assert_eq!(v, cloned);
    }

    #[test]
    fn evidence_dispute_clone_equals_original() {
        let d = make_dispute();
        let cloned = d.clone();
        assert_eq!(d, cloned);
    }

    #[test]
    fn evidence_type_equality() {
        assert_eq!(EvidenceType::Document, EvidenceType::Document);
        assert_ne!(EvidenceType::Document, EvidenceType::Testimony);
        assert_ne!(EvidenceType::Media, EvidenceType::Transaction);
    }

    #[test]
    fn verification_status_equality() {
        assert_eq!(VerificationStatus::Verified, VerificationStatus::Verified);
        assert_ne!(VerificationStatus::Verified, VerificationStatus::Rejected);
        assert_ne!(VerificationStatus::Pending, VerificationStatus::Challenged);
    }
}
