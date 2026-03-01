//! Verification Integrity Zome
//!
//! Defines entry types for design verification and safety claims,
//! integrating with the Knowledge hApp for epistemic classification.

use hdi::prelude::*;
use fabrication_common::*;
use fabrication_common::validation;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    DesignVerification(DesignVerification),
    #[entry_type(visibility = "public")]
    SafetyClaim(SafetyClaim),
    #[entry_type(visibility = "public")]
    VerificationRequest(VerificationRequest),
}

#[hdk_link_types]
pub enum LinkTypes {
    DesignToVerifications,
    DesignToClaims,
    VerifierToVerifications,
    OpenRequests,
    ClaimToKnowledge,
    RateLimitBucket,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DesignVerification {
    pub design_hash: ActionHash,
    pub verification_type: VerificationType,
    pub result: VerificationResult,
    pub evidence: Vec<ActionHash>,
    pub verifier: AgentPubKey,
    pub verifier_credentials: Vec<String>,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SafetyClaim {
    pub design_hash: ActionHash,
    pub claim_type: SafetyClaimType,
    pub claim_text: String,
    pub epistemic: ClaimEpistemic,
    pub supporting_evidence: Vec<String>,
    pub knowledge_claim_hash: Option<ActionHash>,
    pub author: AgentPubKey,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerificationRequest {
    pub design_hash: ActionHash,
    pub requester: AgentPubKey,
    pub target_safety_class: SafetyClass,
    pub bounty: Option<u64>,
    pub deadline: Option<Timestamp>,
    pub status: RequestStatus,
    pub created_at: Timestamp,
}

#[hdk_extern]
pub fn genesis_self_check(_: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(
            OpEntry::CreateEntry { app_entry, .. }
            | OpEntry::UpdateEntry { app_entry, .. }
        ) => match app_entry {
            EntryTypes::DesignVerification(v) => validate_verification(v),
            EntryTypes::SafetyClaim(c) => validate_safety_claim(c),
            EntryTypes::VerificationRequest(r) => validate_verification_request(r),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            let max_len: usize = 256;
            check!(validation::require_max_tag_len(&tag, max_len, &format!("{:?}", link_type)));
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            if action.author != *original_action.action().author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(op_update) => {
            let update_action = match op_update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(update_action.original_action_address.clone())?;
            if update_action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(op_delete) => {
            let original = must_get_action(op_delete.action.deletes_address.clone())?;
            if op_delete.action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate a DesignVerification entry.
fn validate_verification(v: DesignVerification) -> ExternResult<ValidateCallbackResult> {
    // --- verifier_credentials: max 32 items, each max 256 chars ---
    check!(validation::require_max_vec_len(
        &v.verifier_credentials, 32, "verifier_credentials"
    ));
    for cred in &v.verifier_credentials {
        check!(validation::require_max_len(cred, 256, "verifier credential"));
    }

    // --- evidence: max 64 items ---
    check!(validation::require_max_vec_len(&v.evidence, 64, "evidence"));

    // --- VerificationResult variant fields ---
    match &v.result {
        VerificationResult::Passed { confidence, notes } => {
            check!(validation::require_in_range(*confidence, 0.0, 1.0, "Passed.confidence"));
            check!(validation::require_max_len(notes, 4096, "Passed.notes"));
        }
        VerificationResult::Failed { reasons } => {
            check!(validation::require_max_vec_len(reasons, 32, "Failed.reasons"));
            for reason in reasons {
                check!(validation::require_max_len(reason, 1024, "Failed reason"));
            }
        }
        VerificationResult::ConditionalPass { conditions, confidence } => {
            check!(validation::require_in_range(*confidence, 0.0, 1.0, "ConditionalPass.confidence"));
            check!(validation::require_max_vec_len(conditions, 32, "ConditionalPass.conditions"));
            for cond in conditions {
                check!(validation::require_max_len(cond, 1024, "ConditionalPass condition"));
            }
        }
        VerificationResult::NeedsMoreEvidence => {}
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a SafetyClaim entry.
fn validate_safety_claim(c: SafetyClaim) -> ExternResult<ValidateCallbackResult> {
    // --- claim_text: non-empty (trim), max 4096 chars ---
    check!(validation::require_non_empty(&c.claim_text, "claim_text"));
    check!(validation::require_max_len(&c.claim_text, 4096, "claim_text"));

    // --- epistemic scores: each in 0.0..1.0 ---
    check!(validation::require_in_range(c.epistemic.empirical, 0.0, 1.0, "epistemic.empirical"));
    check!(validation::require_in_range(c.epistemic.normative, 0.0, 1.0, "epistemic.normative"));
    check!(validation::require_in_range(c.epistemic.mythic, 0.0, 1.0, "epistemic.mythic"));

    // --- supporting_evidence: max 64 items, each max 256 chars ---
    check!(validation::require_max_vec_len(
        &c.supporting_evidence, 64, "supporting_evidence"
    ));
    for ev in &c.supporting_evidence {
        check!(validation::require_max_len(ev, 256, "supporting evidence item"));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a VerificationRequest entry.
fn validate_verification_request(r: VerificationRequest) -> ExternResult<ValidateCallbackResult> {
    // --- bounty: if present, max 1_000_000_000 ---
    if let Some(bounty) = r.bounty {
        if bounty > 1_000_000_000 {
            return Ok(ValidateCallbackResult::Invalid(
                "bounty cannot exceed 1000000000".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers ----

    fn valid_verification() -> DesignVerification {
        DesignVerification {
            design_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            verification_type: VerificationType::StructuralAnalysis,
            result: VerificationResult::Passed {
                confidence: 0.95,
                notes: "All checks passed".to_string(),
            },
            evidence: vec![ActionHash::from_raw_36(vec![1u8; 36])],
            verifier: AgentPubKey::from_raw_36(vec![0u8; 36]),
            verifier_credentials: vec!["PE License #12345".to_string()],
            created_at: Timestamp::now(),
        }
    }

    fn valid_safety_claim() -> SafetyClaim {
        SafetyClaim {
            design_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            claim_type: SafetyClaimType::LoadCapacity("Supports 50kg".to_string()),
            claim_text: "This bracket supports up to 50kg static load".to_string(),
            epistemic: ClaimEpistemic {
                empirical: 0.8,
                normative: 0.5,
                mythic: 0.1,
            },
            supporting_evidence: vec!["FEA report v2.1".to_string()],
            knowledge_claim_hash: None,
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            created_at: Timestamp::now(),
        }
    }

    fn valid_verification_request() -> VerificationRequest {
        VerificationRequest {
            design_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            requester: AgentPubKey::from_raw_36(vec![0u8; 36]),
            target_safety_class: SafetyClass::Class2LoadBearing,
            bounty: Some(1000),
            deadline: None,
            status: RequestStatus::Open,
            created_at: Timestamp::now(),
        }
    }

    // ---- DesignVerification tests ----

    #[test]
    fn test_valid_verification_passes() {
        let result = validate_verification(valid_verification()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_too_many_credentials_rejected() {
        let mut v = valid_verification();
        v.verifier_credentials = (0..33).map(|i| format!("cred-{}", i)).collect();
        let result = validate_verification(v).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("verifier_credentials")));
    }

    #[test]
    fn test_nan_passed_confidence_rejected() {
        let mut v = valid_verification();
        v.result = VerificationResult::Passed {
            confidence: f32::NAN,
            notes: "ok".to_string(),
        };
        let result = validate_verification(v).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("Passed.confidence")));
    }

    #[test]
    fn test_nan_conditional_pass_confidence_rejected() {
        let mut v = valid_verification();
        v.result = VerificationResult::ConditionalPass {
            conditions: vec!["Retest after 24h".to_string()],
            confidence: f32::NAN,
        };
        let result = validate_verification(v).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("ConditionalPass.confidence")));
    }

    // ---- SafetyClaim tests ----

    #[test]
    fn test_valid_claim_passes() {
        let result = validate_safety_claim(valid_safety_claim()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_claim_text_rejected() {
        let mut c = valid_safety_claim();
        c.claim_text = "   ".to_string();
        let result = validate_safety_claim(c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("claim_text")));
    }

    #[test]
    fn test_nan_empirical_epistemic_rejected() {
        let mut c = valid_safety_claim();
        c.epistemic.empirical = f32::NAN;
        let result = validate_safety_claim(c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("epistemic.empirical")));
    }

    #[test]
    fn test_nan_normative_epistemic_rejected() {
        let mut c = valid_safety_claim();
        c.epistemic.normative = f32::NAN;
        let result = validate_safety_claim(c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("epistemic.normative")));
    }

    #[test]
    fn test_nan_mythic_epistemic_rejected() {
        let mut c = valid_safety_claim();
        c.epistemic.mythic = f32::NAN;
        let result = validate_safety_claim(c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("epistemic.mythic")));
    }

    #[test]
    fn test_too_many_supporting_evidence_rejected() {
        let mut c = valid_safety_claim();
        c.supporting_evidence = (0..65).map(|i| format!("evidence-{}", i)).collect();
        let result = validate_safety_claim(c).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("supporting_evidence")));
    }

    // ---- VerificationRequest tests ----

    #[test]
    fn test_valid_request_passes() {
        let result = validate_verification_request(valid_verification_request()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_bounty_too_large_rejected() {
        let mut r = valid_verification_request();
        r.bounty = Some(1_000_000_001);
        let result = validate_verification_request(r).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("bounty")));
    }

    // =========================================================================
    // Link tag validation tests
    // =========================================================================

    #[test]
    fn test_link_tag_at_max_passes() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validation::require_max_tag_len(&tag, 256, "test");
        assert!(result.is_err()); // Err(()) means "no validation issue found"
    }

    #[test]
    fn test_link_tag_over_max_rejected() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validation::require_max_tag_len(&tag, 256, "test");
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(msg)) if msg.contains("link tag")));
    }
}
