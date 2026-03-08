//! Fact-Check Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FactCheck {
    pub id: String,
    pub publication_id: String,
    pub claim_text: String,
    pub claim_location: String,
    pub epistemic_position: EpistemicPosition,
    pub verdict: FactCheckVerdict,
    pub evidence: Vec<EvidenceItem>,
    pub checker_did: String,
    pub checked: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicPosition {
    pub empirical: f64, // 0.0 to 1.0
    pub normative: f64, // 0.0 to 1.0
    pub mythic: f64,    // 0.0 to 1.0
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FactCheckVerdict {
    True,
    MostlyTrue,
    HalfTrue,
    MostlyFalse,
    False,
    Unverifiable,
    OutOfContext,
    Satire,
    Opinion,
    PartiallyTrue,
    Misleading,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EvidenceItem {
    pub source_type: SourceType,
    pub source_url: Option<String>,
    pub source_did: Option<String>,
    pub description: String,
    pub supports_claim: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SourceType {
    PrimarySource,
    SecondarySource,
    ExpertOpinion,
    OfficialDocument,
    ScientificStudy,
    EyewitnessAccount,
    DataAnalysis,
    Other(String),
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SourceCredibility {
    pub source_id: String,
    pub source_type: SourceType,
    pub credibility_score: f64,
    pub verification_count: u32,
    pub dispute_count: u32,
    pub last_assessed: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FactCheckDispute {
    pub id: String,
    pub fact_check_id: String,
    pub disputer_did: String,
    pub reason: String,
    pub counter_evidence: Vec<EvidenceItem>,
    pub status: DisputeStatus,
    pub created: Timestamp,
    pub resolved: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeStatus {
    Pending,
    Upheld,
    Rejected,
    PartiallyUpheld,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    FactCheck(FactCheck),
    SourceCredibility(SourceCredibility),
    FactCheckDispute(FactCheckDispute),
}

#[hdk_link_types]
pub enum LinkTypes {
    PublicationToFactChecks,
    CheckerToFactChecks,
    ClaimToFactCheck,
    FactCheckToDisputes,
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
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::FactCheck(check) => {
                    validate_create_fact_check(EntryCreationAction::Create(action), check)
                }
                EntryTypes::SourceCredibility(source) => {
                    validate_create_source_credibility(EntryCreationAction::Create(action), source)
                }
                EntryTypes::FactCheckDispute(dispute) => {
                    validate_create_fact_check_dispute(EntryCreationAction::Create(action), dispute)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::FactCheck(_) => Ok(ValidateCallbackResult::Invalid(
                    "Fact checks cannot be updated".into(),
                )),
                EntryTypes::SourceCredibility(source) => {
                    validate_update_source_credibility(action, source)
                }
                EntryTypes::FactCheckDispute(dispute) => {
                    validate_update_fact_check_dispute(action, dispute)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::PublicationToFactChecks
                | LinkTypes::CheckerToFactChecks
                | LinkTypes::ClaimToFactCheck
                | LinkTypes::FactCheckToDisputes => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_fact_check(
    _action: EntryCreationAction,
    check: FactCheck,
) -> ExternResult<ValidateCallbackResult> {
    if check.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "FactCheck ID cannot be empty".into(),
        ));
    }
    if check.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "FactCheck ID too long (max 256 chars)".into(),
        ));
    }
    if check.publication_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "FactCheck publication_id cannot be empty".into(),
        ));
    }
    if check.publication_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "FactCheck publication_id too long (max 256 chars)".into(),
        ));
    }
    if !check.checker_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Checker must be a valid DID".into(),
        ));
    }
    if check.checker_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Checker DID too long (max 256 chars)".into(),
        ));
    }
    if check.claim_text.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim text required".into(),
        ));
    }
    if check.claim_text.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim text too long (max 4096 chars)".into(),
        ));
    }
    if check.claim_location.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim location too long (max 256 chars)".into(),
        ));
    }
    // Validate evidence items
    for item in &check.evidence {
        if item.description.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Evidence description too long (max 4096 chars)".into(),
            ));
        }
        if let Some(ref url) = item.source_url {
            if url.len() > 2048 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Evidence source URL too long (max 2048 chars)".into(),
                ));
            }
        }
    }
    let ep = &check.epistemic_position;
    if !ep.empirical.is_finite() || !ep.normative.is_finite() || !ep.mythic.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "epistemic position values must be a finite number".into(),
        ));
    }
    if ep.empirical < 0.0
        || ep.empirical > 1.0
        || ep.normative < 0.0
        || ep.normative > 1.0
        || ep.mythic < 0.0
        || ep.mythic > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Epistemic values must be 0-1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_source_credibility(
    _action: EntryCreationAction,
    source: SourceCredibility,
) -> ExternResult<ValidateCallbackResult> {
    if source.source_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "SourceCredibility source_id cannot be empty".into(),
        ));
    }
    if source.source_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "SourceCredibility source_id too long (max 256 chars)".into(),
        ));
    }
    if !source.credibility_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "credibility_score must be a finite number".into(),
        ));
    }
    if source.credibility_score < 0.0 || source.credibility_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credibility must be 0-1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_source_credibility(
    _action: Update,
    source: SourceCredibility,
) -> ExternResult<ValidateCallbackResult> {
    if !source.credibility_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "credibility_score must be a finite number".into(),
        ));
    }
    if source.credibility_score < 0.0 || source.credibility_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credibility must be 0-1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_fact_check_dispute(
    _action: EntryCreationAction,
    dispute: FactCheckDispute,
) -> ExternResult<ValidateCallbackResult> {
    if dispute.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute ID cannot be empty".into(),
        ));
    }
    if dispute.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute ID too long (max 256 chars)".into(),
        ));
    }
    if dispute.fact_check_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute fact_check_id cannot be empty".into(),
        ));
    }
    if dispute.fact_check_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute fact_check_id too long (max 256 chars)".into(),
        ));
    }
    if !dispute.disputer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Disputer must be a valid DID".into(),
        ));
    }
    if dispute.disputer_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Disputer DID too long (max 256 chars)".into(),
        ));
    }
    if dispute.reason.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute reason required".into(),
        ));
    }
    if dispute.reason.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute reason too long (max 4096 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_fact_check_dispute(
    _action: Update,
    _dispute: FactCheckDispute,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

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
    // CONSTRUCTION HELPERS
    // ========================================================================

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn fake_entry_creation_action() -> EntryCreationAction {
        EntryCreationAction::Create(fake_create())
    }

    fn fake_update() -> Update {
        Update {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(0),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0u8; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn make_epistemic_position() -> EpistemicPosition {
        EpistemicPosition {
            empirical: 0.8,
            normative: 0.5,
            mythic: 0.1,
        }
    }

    fn make_evidence_item() -> EvidenceItem {
        EvidenceItem {
            source_type: SourceType::PrimarySource,
            source_url: Some("https://example.com/source".into()),
            source_did: Some("did:example:source".into()),
            description: "Primary evidence".into(),
            supports_claim: true,
        }
    }

    fn make_fact_check() -> FactCheck {
        FactCheck {
            id: "fc-1".into(),
            publication_id: "pub-1".into(),
            claim_text: "The earth revolves around the sun".into(),
            claim_location: "paragraph 3".into(),
            epistemic_position: make_epistemic_position(),
            verdict: FactCheckVerdict::True,
            evidence: vec![make_evidence_item()],
            checker_did: "did:example:checker".into(),
            checked: ts(),
        }
    }

    fn make_source_credibility() -> SourceCredibility {
        SourceCredibility {
            source_id: "src-1".into(),
            source_type: SourceType::ScientificStudy,
            credibility_score: 0.85,
            verification_count: 10,
            dispute_count: 1,
            last_assessed: ts(),
        }
    }

    fn make_dispute() -> FactCheckDispute {
        FactCheckDispute {
            id: "dispute-1".into(),
            fact_check_id: "fc-1".into(),
            disputer_did: "did:example:disputer".into(),
            reason: "The evidence cited is outdated".into(),
            counter_evidence: vec![make_evidence_item()],
            status: DisputeStatus::Pending,
            created: ts(),
            resolved: None,
        }
    }

    // ========================================================================
    // FACT CHECK VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_fact_check_passes() {
        let result = validate_create_fact_check(fake_entry_creation_action(), make_fact_check());
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_checker_did_not_did_rejected() {
        let mut fc = make_fact_check();
        fc.checker_did = "not-a-did".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Checker must be a valid DID");
    }

    #[test]
    fn fact_check_checker_did_empty_rejected() {
        let mut fc = make_fact_check();
        fc.checker_did = "".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Checker must be a valid DID");
    }

    #[test]
    fn fact_check_checker_did_prefix_only_passes() {
        // "did:" alone passes the starts_with check
        let mut fc = make_fact_check();
        fc.checker_did = "did:".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_claim_text_empty_rejected() {
        let mut fc = make_fact_check();
        fc.claim_text = "".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Claim text required");
    }

    #[test]
    fn fact_check_claim_text_whitespace_rejected() {
        let mut fc = make_fact_check();
        fc.claim_text = "   ".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Claim text required");
    }

    #[test]
    fn fact_check_epistemic_empirical_below_zero_rejected() {
        let mut fc = make_fact_check();
        fc.epistemic_position.empirical = -0.01;
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Epistemic values must be 0-1");
    }

    #[test]
    fn fact_check_epistemic_empirical_above_one_rejected() {
        let mut fc = make_fact_check();
        fc.epistemic_position.empirical = 1.01;
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Epistemic values must be 0-1");
    }

    #[test]
    fn fact_check_epistemic_normative_below_zero_rejected() {
        let mut fc = make_fact_check();
        fc.epistemic_position.normative = -0.5;
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Epistemic values must be 0-1");
    }

    #[test]
    fn fact_check_epistemic_normative_above_one_rejected() {
        let mut fc = make_fact_check();
        fc.epistemic_position.normative = 1.5;
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Epistemic values must be 0-1");
    }

    #[test]
    fn fact_check_epistemic_mythic_below_zero_rejected() {
        let mut fc = make_fact_check();
        fc.epistemic_position.mythic = -100.0;
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Epistemic values must be 0-1");
    }

    #[test]
    fn fact_check_epistemic_mythic_above_one_rejected() {
        let mut fc = make_fact_check();
        fc.epistemic_position.mythic = 100.0;
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Epistemic values must be 0-1");
    }

    #[test]
    fn fact_check_epistemic_all_zero_passes() {
        let mut fc = make_fact_check();
        fc.epistemic_position = EpistemicPosition {
            empirical: 0.0,
            normative: 0.0,
            mythic: 0.0,
        };
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_epistemic_all_one_passes() {
        let mut fc = make_fact_check();
        fc.epistemic_position = EpistemicPosition {
            empirical: 1.0,
            normative: 1.0,
            mythic: 1.0,
        };
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_multiple_failures_first_wins() {
        // id, publication_id, checker_did, claim_text all invalid; id check comes first
        let mut fc = make_fact_check();
        fc.id = "".into();
        fc.checker_did = "not-a-did".into();
        fc.claim_text = "".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "FactCheck ID cannot be empty");
    }

    #[test]
    fn fact_check_every_verdict_variant_passes() {
        let variants = vec![
            FactCheckVerdict::True,
            FactCheckVerdict::MostlyTrue,
            FactCheckVerdict::HalfTrue,
            FactCheckVerdict::MostlyFalse,
            FactCheckVerdict::False,
            FactCheckVerdict::Unverifiable,
            FactCheckVerdict::OutOfContext,
            FactCheckVerdict::Satire,
            FactCheckVerdict::Opinion,
            FactCheckVerdict::PartiallyTrue,
            FactCheckVerdict::Misleading,
        ];
        for variant in variants {
            let mut fc = make_fact_check();
            fc.verdict = variant;
            let result = validate_create_fact_check(fake_entry_creation_action(), fc);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn fact_check_empty_evidence_list_passes() {
        let mut fc = make_fact_check();
        fc.evidence = vec![];
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // SOURCE CREDIBILITY VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_source_credibility_passes() {
        let result = validate_create_source_credibility(
            fake_entry_creation_action(),
            make_source_credibility(),
        );
        assert!(is_valid(&result));
    }

    #[test]
    fn source_credibility_below_zero_rejected() {
        let mut sc = make_source_credibility();
        sc.credibility_score = -0.01;
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Credibility must be 0-1");
    }

    #[test]
    fn source_credibility_above_one_rejected() {
        let mut sc = make_source_credibility();
        sc.credibility_score = 1.01;
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Credibility must be 0-1");
    }

    #[test]
    fn source_credibility_exactly_zero_passes() {
        let mut sc = make_source_credibility();
        sc.credibility_score = 0.0;
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_valid(&result));
    }

    #[test]
    fn source_credibility_exactly_one_passes() {
        let mut sc = make_source_credibility();
        sc.credibility_score = 1.0;
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_valid(&result));
    }

    #[test]
    fn source_credibility_negative_large_rejected() {
        let mut sc = make_source_credibility();
        sc.credibility_score = -999.0;
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_invalid(&result));
    }

    #[test]
    fn source_credibility_large_positive_rejected() {
        let mut sc = make_source_credibility();
        sc.credibility_score = 999.0;
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_invalid(&result));
    }

    #[test]
    fn source_credibility_midpoint_passes() {
        let mut sc = make_source_credibility();
        sc.credibility_score = 0.5;
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // SOURCE CREDIBILITY UPDATE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn update_source_credibility_valid_passes() {
        let result = validate_update_source_credibility(fake_update(), make_source_credibility());
        assert!(is_valid(&result));
    }

    #[test]
    fn update_source_credibility_below_zero_rejected() {
        let mut sc = make_source_credibility();
        sc.credibility_score = -0.01;
        let result = validate_update_source_credibility(fake_update(), sc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Credibility must be 0-1");
    }

    #[test]
    fn update_source_credibility_above_one_rejected() {
        let mut sc = make_source_credibility();
        sc.credibility_score = 1.01;
        let result = validate_update_source_credibility(fake_update(), sc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Credibility must be 0-1");
    }

    #[test]
    fn update_source_credibility_boundary_zero_passes() {
        let mut sc = make_source_credibility();
        sc.credibility_score = 0.0;
        let result = validate_update_source_credibility(fake_update(), sc);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_source_credibility_boundary_one_passes() {
        let mut sc = make_source_credibility();
        sc.credibility_score = 1.0;
        let result = validate_update_source_credibility(fake_update(), sc);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // FACT CHECK DISPUTE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_dispute_passes() {
        let result =
            validate_create_fact_check_dispute(fake_entry_creation_action(), make_dispute());
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_disputer_did_not_did_rejected() {
        let mut dispute = make_dispute();
        dispute.disputer_did = "not-a-did".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Disputer must be a valid DID");
    }

    #[test]
    fn dispute_disputer_did_empty_rejected() {
        let mut dispute = make_dispute();
        dispute.disputer_did = "".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Disputer must be a valid DID");
    }

    #[test]
    fn dispute_disputer_did_prefix_only_passes() {
        let mut dispute = make_dispute();
        dispute.disputer_did = "did:".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_reason_empty_rejected() {
        let mut dispute = make_dispute();
        dispute.reason = "".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Dispute reason required");
    }

    #[test]
    fn dispute_reason_whitespace_rejected() {
        let mut dispute = make_dispute();
        dispute.reason = "   ".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Dispute reason required");
    }

    #[test]
    fn dispute_multiple_failures_first_wins() {
        // id, fact_check_id, disputer_did, reason all invalid; id check comes first
        let mut dispute = make_dispute();
        dispute.id = "".into();
        dispute.disputer_did = "bad".into();
        dispute.reason = "".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Dispute ID cannot be empty");
    }

    #[test]
    fn dispute_empty_counter_evidence_passes() {
        let mut dispute = make_dispute();
        dispute.counter_evidence = vec![];
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_every_status_variant_passes() {
        let variants = vec![
            DisputeStatus::Pending,
            DisputeStatus::Upheld,
            DisputeStatus::Rejected,
            DisputeStatus::PartiallyUpheld,
        ];
        for variant in variants {
            let mut dispute = make_dispute();
            dispute.status = variant;
            let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn dispute_with_resolved_timestamp_passes() {
        let mut dispute = make_dispute();
        dispute.resolved = Some(ts());
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // FACT CHECK DISPUTE UPDATE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn update_dispute_always_passes() {
        let result = validate_update_fact_check_dispute(fake_update(), make_dispute());
        assert!(is_valid(&result));
    }

    #[test]
    fn update_dispute_with_empty_reason_passes() {
        // Update validation has no checks; it always returns Valid
        let mut dispute = make_dispute();
        dispute.reason = "".into();
        let result = validate_update_fact_check_dispute(fake_update(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_dispute_with_bad_did_passes() {
        // Update validation has no checks
        let mut dispute = make_dispute();
        dispute.disputer_did = "not-a-did".into();
        let result = validate_update_fact_check_dispute(fake_update(), dispute);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // NEW FIELD-LEVEL VALIDATION TESTS (Fix #6)
    // ========================================================================

    #[test]
    fn fact_check_empty_id_rejected() {
        let mut fc = make_fact_check();
        fc.id = "".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "FactCheck ID cannot be empty");
    }

    #[test]
    fn fact_check_whitespace_id_rejected() {
        let mut fc = make_fact_check();
        fc.id = "  \t ".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "FactCheck ID cannot be empty");
    }

    #[test]
    fn fact_check_empty_publication_id_rejected() {
        let mut fc = make_fact_check();
        fc.publication_id = "".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "FactCheck publication_id cannot be empty"
        );
    }

    #[test]
    fn fact_check_whitespace_publication_id_rejected() {
        let mut fc = make_fact_check();
        fc.publication_id = "   ".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "FactCheck publication_id cannot be empty"
        );
    }

    #[test]
    fn source_credibility_empty_source_id_rejected() {
        let mut sc = make_source_credibility();
        sc.source_id = "".into();
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "SourceCredibility source_id cannot be empty"
        );
    }

    #[test]
    fn source_credibility_whitespace_source_id_rejected() {
        let mut sc = make_source_credibility();
        sc.source_id = "   ".into();
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "SourceCredibility source_id cannot be empty"
        );
    }

    #[test]
    fn dispute_empty_id_rejected() {
        let mut dispute = make_dispute();
        dispute.id = "".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Dispute ID cannot be empty");
    }

    #[test]
    fn dispute_whitespace_id_rejected() {
        let mut dispute = make_dispute();
        dispute.id = "  \t ".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Dispute ID cannot be empty");
    }

    #[test]
    fn dispute_empty_fact_check_id_rejected() {
        let mut dispute = make_dispute();
        dispute.fact_check_id = "".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Dispute fact_check_id cannot be empty"
        );
    }

    #[test]
    fn dispute_whitespace_fact_check_id_rejected() {
        let mut dispute = make_dispute();
        dispute.fact_check_id = "   ".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Dispute fact_check_id cannot be empty"
        );
    }

    #[test]
    fn fact_check_did_before_claim_text_rejection_order() {
        // With valid id and publication_id, checker_did fails before claim_text
        let mut fc = make_fact_check();
        fc.checker_did = "bad".into();
        fc.claim_text = "".into();
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Checker must be a valid DID");
    }

    #[test]
    fn dispute_did_before_reason_rejection_order() {
        // With valid id and fact_check_id, disputer_did fails before reason
        let mut dispute = make_dispute();
        dispute.disputer_did = "bad".into();
        dispute.reason = "".into();
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Disputer must be a valid DID");
    }

    // ========================================================================
    // STRING LENGTH LIMIT TESTS
    // ========================================================================

    #[test]
    fn fact_check_id_at_limit_passes() {
        let mut fc = make_fact_check();
        fc.id = "i".repeat(256);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_id_over_limit_rejected() {
        let mut fc = make_fact_check();
        fc.id = "i".repeat(257);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "FactCheck ID too long (max 256 chars)"
        );
    }

    #[test]
    fn fact_check_publication_id_at_limit_passes() {
        let mut fc = make_fact_check();
        fc.publication_id = "p".repeat(256);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_publication_id_over_limit_rejected() {
        let mut fc = make_fact_check();
        fc.publication_id = "p".repeat(257);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "FactCheck publication_id too long (max 256 chars)"
        );
    }

    #[test]
    fn fact_check_checker_did_at_limit_passes() {
        let mut fc = make_fact_check();
        fc.checker_did = format!("did:{}", "x".repeat(252));
        assert_eq!(fc.checker_did.len(), 256);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_checker_did_over_limit_rejected() {
        let mut fc = make_fact_check();
        fc.checker_did = format!("did:{}", "x".repeat(253));
        assert_eq!(fc.checker_did.len(), 257);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Checker DID too long (max 256 chars)");
    }

    #[test]
    fn fact_check_claim_text_at_limit_passes() {
        let mut fc = make_fact_check();
        fc.claim_text = "t".repeat(4096);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_claim_text_over_limit_rejected() {
        let mut fc = make_fact_check();
        fc.claim_text = "t".repeat(4097);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Claim text too long (max 4096 chars)");
    }

    #[test]
    fn fact_check_claim_location_at_limit_passes() {
        let mut fc = make_fact_check();
        fc.claim_location = "l".repeat(256);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_valid(&result));
    }

    #[test]
    fn fact_check_claim_location_over_limit_rejected() {
        let mut fc = make_fact_check();
        fc.claim_location = "l".repeat(257);
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Claim location too long (max 256 chars)"
        );
    }

    #[test]
    fn fact_check_evidence_description_over_limit_rejected() {
        let mut fc = make_fact_check();
        fc.evidence = vec![EvidenceItem {
            source_type: SourceType::PrimarySource,
            source_url: None,
            source_did: None,
            description: "d".repeat(4097),
            supports_claim: true,
        }];
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Evidence description too long (max 4096 chars)"
        );
    }

    #[test]
    fn fact_check_evidence_source_url_over_limit_rejected() {
        let mut fc = make_fact_check();
        fc.evidence = vec![EvidenceItem {
            source_type: SourceType::PrimarySource,
            source_url: Some("u".repeat(2049)),
            source_did: None,
            description: "ok".into(),
            supports_claim: true,
        }];
        let result = validate_create_fact_check(fake_entry_creation_action(), fc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Evidence source URL too long (max 2048 chars)"
        );
    }

    #[test]
    fn source_credibility_source_id_at_limit_passes() {
        let mut sc = make_source_credibility();
        sc.source_id = "s".repeat(256);
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_valid(&result));
    }

    #[test]
    fn source_credibility_source_id_over_limit_rejected() {
        let mut sc = make_source_credibility();
        sc.source_id = "s".repeat(257);
        let result = validate_create_source_credibility(fake_entry_creation_action(), sc);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "SourceCredibility source_id too long (max 256 chars)"
        );
    }

    #[test]
    fn dispute_id_at_limit_passes() {
        let mut dispute = make_dispute();
        dispute.id = "d".repeat(256);
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_id_over_limit_rejected() {
        let mut dispute = make_dispute();
        dispute.id = "d".repeat(257);
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Dispute ID too long (max 256 chars)");
    }

    #[test]
    fn dispute_fact_check_id_at_limit_passes() {
        let mut dispute = make_dispute();
        dispute.fact_check_id = "f".repeat(256);
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_fact_check_id_over_limit_rejected() {
        let mut dispute = make_dispute();
        dispute.fact_check_id = "f".repeat(257);
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Dispute fact_check_id too long (max 256 chars)"
        );
    }

    #[test]
    fn dispute_disputer_did_at_limit_passes() {
        let mut dispute = make_dispute();
        dispute.disputer_did = format!("did:{}", "x".repeat(252));
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_disputer_did_over_limit_rejected() {
        let mut dispute = make_dispute();
        dispute.disputer_did = format!("did:{}", "x".repeat(253));
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Disputer DID too long (max 256 chars)"
        );
    }

    #[test]
    fn dispute_reason_at_limit_passes() {
        let mut dispute = make_dispute();
        dispute.reason = "r".repeat(4096);
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_valid(&result));
    }

    #[test]
    fn dispute_reason_over_limit_rejected() {
        let mut dispute = make_dispute();
        dispute.reason = "r".repeat(4097);
        let result = validate_create_fact_check_dispute(fake_entry_creation_action(), dispute);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Dispute reason too long (max 4096 chars)"
        );
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(link_type: &LinkTypes, tag: &LinkTag) -> ValidateCallbackResult {
        let tag_len = tag.0.len();
        match link_type {
            LinkTypes::PublicationToFactChecks
            | LinkTypes::CheckerToFactChecks
            | LinkTypes::ClaimToFactCheck
            | LinkTypes::FactCheckToDisputes => {
                if tag_len > 256 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 256 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
        }
    }

    fn validate_delete_link_tag(tag: &LinkTag) -> ValidateCallbackResult {
        if tag.0.len() > 256 {
            ValidateCallbackResult::Invalid("Delete link tag too long (max 256 bytes)".into())
        } else {
            ValidateCallbackResult::Valid
        }
    }

    // -- PublicationToFactChecks (256-byte limit) boundary tests --

    #[test]
    fn link_tag_pub_to_factchecks_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::PublicationToFactChecks, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_pub_to_factchecks_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_create_link_tag(&LinkTypes::PublicationToFactChecks, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_pub_to_factchecks_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_create_link_tag(&LinkTypes::PublicationToFactChecks, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- FactCheckToDisputes (256-byte limit) boundary tests --

    #[test]
    fn link_tag_factcheck_to_disputes_at_limit_valid() {
        let tag = LinkTag::new(vec![0xCC; 256]);
        let result = validate_create_link_tag(&LinkTypes::FactCheckToDisputes, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_factcheck_to_disputes_over_limit_invalid() {
        let tag = LinkTag::new(vec![0xCC; 257]);
        let result = validate_create_link_tag(&LinkTypes::FactCheckToDisputes, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- DoS prevention: massive tags rejected for all link types --

    #[test]
    fn link_tag_dos_prevention_all_types() {
        let massive_tag = LinkTag::new(vec![0xFF; 10_000]);
        let all_types = [
            LinkTypes::PublicationToFactChecks,
            LinkTypes::CheckerToFactChecks,
            LinkTypes::ClaimToFactCheck,
            LinkTypes::FactCheckToDisputes,
        ];
        for lt in &all_types {
            let result = validate_create_link_tag(lt, &massive_tag);
            assert!(
                matches!(result, ValidateCallbackResult::Invalid(_)),
                "Massive tag should be rejected for {:?}",
                lt
            );
        }
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
