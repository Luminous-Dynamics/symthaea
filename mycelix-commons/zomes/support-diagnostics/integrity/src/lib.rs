//! Support Diagnostics Integrity Zome
//! Entry types, link types, and validation for diagnostic results,
//! privacy preferences, and cognitive updates in the Mycelix support domain.

use hdi::prelude::*;
use support_types::*;

// ============================================================================
// ANCHOR
// ============================================================================

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// DIAGNOSTIC RESULT
// ============================================================================

/// A diagnostic result produced by running a diagnostic check, optionally
/// linked to a support ticket.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DiagnosticResult {
    /// Optional link to the originating support ticket
    pub ticket_hash: Option<ActionHash>,
    /// The type of diagnostic that was run
    pub diagnostic_type: DiagnosticType,
    /// JSON-structured findings from the diagnostic
    pub findings: String,
    /// Severity of the diagnostic outcome
    pub severity: DiagnosticSeverity,
    /// Actionable recommendations based on findings
    pub recommendations: Vec<String>,
    /// Agent who ran the diagnostic
    pub agent: AgentPubKey,
    /// Whether PII has been scrubbed from findings
    pub scrubbed: bool,
    /// When the diagnostic was created
    pub created_at: Timestamp,
}

// ============================================================================
// PRIVACY PREFERENCE
// ============================================================================

/// Per-agent privacy preferences controlling what diagnostic data can be shared.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PrivacyPreference {
    /// The agent these preferences belong to
    pub agent: AgentPubKey,
    /// Sharing tier: LocalOnly, Anonymized, or Full
    pub sharing_tier: SharingTier,
    /// Which support categories this agent allows sharing for
    pub allowed_categories: Vec<SupportCategory>,
    /// Whether to share system information
    pub share_system_info: bool,
    /// Whether to share resolution patterns
    pub share_resolution_patterns: bool,
    /// Whether to share cognitive updates
    pub share_cognitive_updates: bool,
    /// When preferences were last updated
    pub updated_at: Timestamp,
}

// ============================================================================
// COGNITIVE UPDATE
// ============================================================================

/// A cognitive update encoding a resolution pattern as a BinaryHV for
/// cross-agent learning via the Symthaea bridge.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CognitiveUpdate {
    /// Support category this update relates to
    pub category: SupportCategory,
    /// BinaryHV encoding (must be exactly 2048 bytes)
    pub encoding: Vec<u8>,
    /// Phi (integrated information) value for this pattern
    pub phi: f64,
    /// Human-readable description of the resolution pattern
    pub resolution_pattern: String,
    /// Agent who created this update
    pub source_agent: AgentPubKey,
    /// When the update was created
    pub created_at: Timestamp,
}

// ============================================================================
// HELPER PROFILE
// ============================================================================

/// A helper profile representing an agent who can assist with support tickets.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HelperProfile {
    /// The agent this profile belongs to
    pub agent: AgentPubKey,
    /// Categories this helper has expertise in
    pub expertise_categories: Vec<SupportCategory>,
    /// Maximum number of concurrent tickets this helper can handle
    pub max_concurrent: u32,
    /// Preferred difficulty level for ticket assignments
    pub difficulty_preference: DifficultyLevel,
    /// Whether this helper is currently available
    pub available: bool,
    /// When the profile was created
    pub created_at: Timestamp,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    DiagnosticResult(DiagnosticResult),
    PrivacyPreference(PrivacyPreference),
    CognitiveUpdate(CognitiveUpdate),
    HelperProfile(HelperProfile),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Time-sharded diagnostics index (max 256 byte tag)
    ShardedDiagnostics,
    /// Agent to their diagnostic results (max 256 byte tag)
    AgentToDiagnostic,
    /// Ticket to its diagnostic results (max 256 byte tag)
    TicketToDiagnostic,
    /// Agent to their privacy preference (max 256 byte tag)
    AgentToPrivacyPreference,
    /// Category to cognitive updates, time-sharded (max 256 byte tag)
    CategoryToCognitiveUpdate,
    /// All cognitive updates index, time-sharded (max 256 byte tag)
    AllCognitiveUpdates,
    /// All registered helpers index (max 256 byte tag)
    AllHelpers,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::DiagnosticResult(r) => validate_create_diagnostic(action, r),
                EntryTypes::PrivacyPreference(p) => validate_create_privacy_preference(action, p),
                EntryTypes::CognitiveUpdate(u) => validate_create_cognitive_update(action, u),
                EntryTypes::HelperProfile(h) => validate_helper_profile(action, h),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::DiagnosticResult(r) => {
                    validate_update_diagnostic(action, r, original_action_hash)
                }
                EntryTypes::PrivacyPreference(p) => {
                    validate_update_privacy_preference(action, p, original_action_hash)
                }
                EntryTypes::CognitiveUpdate(u) => {
                    validate_update_cognitive_update(action, u, original_action_hash)
                }
                EntryTypes::HelperProfile(h) => {
                    validate_update_helper_profile(action, h, original_action_hash)
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
        } => match link_type {
            LinkTypes::ShardedDiagnostics => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ShardedDiagnostics link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToDiagnostic => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToDiagnostic link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TicketToDiagnostic => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TicketToDiagnostic link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToPrivacyPreference => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToPrivacyPreference link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CategoryToCognitiveUpdate => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CategoryToCognitiveUpdate link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllCognitiveUpdates => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllCognitiveUpdates link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllHelpers => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllHelpers link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action,
        } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

// ============================================================================
// VALIDATION HELPERS — DiagnosticResult
// ============================================================================

fn validate_create_diagnostic(
    _action: Create,
    r: DiagnosticResult,
) -> ExternResult<ValidateCallbackResult> {
    for rec in &r.recommendations {
        if rec.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Recommendation too long (max 256 chars per item)".into(),
            ));
        }
    }
    // findings must be valid JSON
    if serde_json::from_str::<serde_json::Value>(&r.findings).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "DiagnosticResult findings must be valid JSON".into(),
        ));
    }
    // findings max 32768 bytes
    if r.findings.len() > 32768 {
        return Ok(ValidateCallbackResult::Invalid(
            "DiagnosticResult findings must be 32768 bytes or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_diagnostic(
    _action: Update,
    r: DiagnosticResult,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    for rec in &r.recommendations {
        if rec.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Recommendation too long (max 256 chars per item)".into(),
            ));
        }
    }
    // Same validation as create
    if serde_json::from_str::<serde_json::Value>(&r.findings).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "DiagnosticResult findings must be valid JSON".into(),
        ));
    }
    if r.findings.len() > 32768 {
        return Ok(ValidateCallbackResult::Invalid(
            "DiagnosticResult findings must be 32768 bytes or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// VALIDATION HELPERS — PrivacyPreference
// ============================================================================

fn validate_create_privacy_preference(
    _action: Create,
    p: PrivacyPreference,
) -> ExternResult<ValidateCallbackResult> {
    if p.allowed_categories.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "PrivacyPreference allowed_categories must not be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_privacy_preference(
    _action: Update,
    p: PrivacyPreference,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    if p.allowed_categories.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "PrivacyPreference allowed_categories must not be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// VALIDATION HELPERS — CognitiveUpdate
// ============================================================================

fn validate_create_cognitive_update(
    _action: Create,
    u: CognitiveUpdate,
) -> ExternResult<ValidateCallbackResult> {
    if u.resolution_pattern.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "CognitiveUpdate resolution_pattern cannot be empty".into(),
        ));
    }
    if u.resolution_pattern.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "CognitiveUpdate resolution_pattern too long (max 4096 chars)".into(),
        ));
    }
    if u.encoding.len() != 2048 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "CognitiveUpdate encoding must be exactly 2048 bytes, got {}",
            u.encoding.len()
        )));
    }
    if !u.phi.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "CognitiveUpdate phi must be finite".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_cognitive_update(
    _action: Update,
    u: CognitiveUpdate,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    if u.resolution_pattern.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "CognitiveUpdate resolution_pattern cannot be empty".into(),
        ));
    }
    if u.resolution_pattern.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "CognitiveUpdate resolution_pattern too long (max 4096 chars)".into(),
        ));
    }
    if u.encoding.len() != 2048 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "CognitiveUpdate encoding must be exactly 2048 bytes, got {}",
            u.encoding.len()
        )));
    }
    if !u.phi.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "CognitiveUpdate phi must be finite".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// VALIDATION HELPERS — HelperProfile
// ============================================================================

fn validate_helper_profile(
    _action: Create,
    h: HelperProfile,
) -> ExternResult<ValidateCallbackResult> {
    if h.expertise_categories.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "HelperProfile must have at least one expertise category".into(),
        ));
    }
    if h.max_concurrent == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "HelperProfile max_concurrent must be at least 1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_helper_profile(
    _action: Update,
    h: HelperProfile,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    if h.expertise_categories.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "HelperProfile must have at least one expertise category".into(),
        ));
    }
    if h.max_concurrent == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "HelperProfile max_concurrent must be at least 1".into(),
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
    // HELPERS
    // ========================================================================

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xab; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xdb; 36])
    }

    fn fake_entry_hash() -> EntryHash {
        EntryHash::from_raw_36(vec![0xcd; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(1_700_000_000_000_000)
    }

    fn fake_create() -> Create {
        Create {
            author: fake_agent(),
            timestamp: fake_timestamp(),
            action_seq: 0,
            prev_action: fake_action_hash(),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        }
    }

    fn fake_update() -> Update {
        Update {
            author: fake_agent(),
            timestamp: fake_timestamp(),
            action_seq: 1,
            prev_action: fake_action_hash(),
            original_action_address: fake_action_hash(),
            original_entry_address: fake_entry_hash(),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        }
    }

    fn assert_valid(result: &ExternResult<ValidateCallbackResult>) {
        assert!(
            matches!(result, Ok(ValidateCallbackResult::Valid)),
            "Expected Valid, got {:?}",
            result
        );
    }

    fn assert_invalid(result: &ExternResult<ValidateCallbackResult>) {
        assert!(
            matches!(result, Ok(ValidateCallbackResult::Invalid(_))),
            "Expected Invalid, got {:?}",
            result
        );
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    fn make_diagnostic_result() -> DiagnosticResult {
        DiagnosticResult {
            ticket_hash: Some(fake_action_hash()),
            diagnostic_type: DiagnosticType::NetworkCheck,
            findings: r#"{"status":"ok","latency_ms":42}"#.to_string(),
            severity: DiagnosticSeverity::Healthy,
            recommendations: vec!["No action needed".to_string()],
            agent: fake_agent(),
            scrubbed: false,
            created_at: fake_timestamp(),
        }
    }

    fn make_privacy_preference() -> PrivacyPreference {
        PrivacyPreference {
            agent: fake_agent(),
            sharing_tier: SharingTier::Anonymized,
            allowed_categories: vec![SupportCategory::Network, SupportCategory::General],
            share_system_info: true,
            share_resolution_patterns: true,
            share_cognitive_updates: false,
            updated_at: fake_timestamp(),
        }
    }

    fn make_helper_profile() -> HelperProfile {
        HelperProfile {
            agent: fake_agent(),
            expertise_categories: vec![SupportCategory::Network, SupportCategory::Software],
            max_concurrent: 3,
            difficulty_preference: DifficultyLevel::Intermediate,
            available: true,
            created_at: fake_timestamp(),
        }
    }

    fn make_cognitive_update() -> CognitiveUpdate {
        CognitiveUpdate {
            category: SupportCategory::Software,
            encoding: vec![0u8; 2048],
            phi: 0.42,
            resolution_pattern: "Clear cache and restart".to_string(),
            source_agent: fake_agent(),
            created_at: fake_timestamp(),
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP — DiagnosticResult
    // ========================================================================

    #[test]
    fn serde_roundtrip_diagnostic_result() {
        let diag = make_diagnostic_result();
        let json = serde_json::to_string(&diag).unwrap();
        let back: DiagnosticResult = serde_json::from_str(&json).unwrap();
        assert_eq!(diag, back);
    }

    #[test]
    fn serde_roundtrip_diagnostic_result_no_ticket() {
        let mut diag = make_diagnostic_result();
        diag.ticket_hash = None;
        let json = serde_json::to_string(&diag).unwrap();
        let back: DiagnosticResult = serde_json::from_str(&json).unwrap();
        assert_eq!(diag, back);
        assert!(back.ticket_hash.is_none());
    }

    #[test]
    fn serde_roundtrip_diagnostic_result_scrubbed() {
        let mut diag = make_diagnostic_result();
        diag.scrubbed = true;
        let json = serde_json::to_string(&diag).unwrap();
        let back: DiagnosticResult = serde_json::from_str(&json).unwrap();
        assert!(back.scrubbed);
    }

    #[test]
    fn serde_roundtrip_diagnostic_result_empty_recommendations() {
        let mut diag = make_diagnostic_result();
        diag.recommendations = vec![];
        let json = serde_json::to_string(&diag).unwrap();
        let back: DiagnosticResult = serde_json::from_str(&json).unwrap();
        assert!(back.recommendations.is_empty());
    }

    #[test]
    fn serde_roundtrip_diagnostic_result_all_severities() {
        for severity in [
            DiagnosticSeverity::Healthy,
            DiagnosticSeverity::Warning,
            DiagnosticSeverity::Error,
            DiagnosticSeverity::Critical,
        ] {
            let mut diag = make_diagnostic_result();
            diag.severity = severity.clone();
            let json = serde_json::to_string(&diag).unwrap();
            let back: DiagnosticResult = serde_json::from_str(&json).unwrap();
            assert_eq!(back.severity, severity);
        }
    }

    #[test]
    fn serde_roundtrip_diagnostic_result_all_types() {
        for dtype in [
            DiagnosticType::NetworkCheck,
            DiagnosticType::DiskSpace,
            DiagnosticType::ServiceStatus,
            DiagnosticType::HolochainHealth,
            DiagnosticType::MemoryUsage,
            DiagnosticType::Custom("my-check".to_string()),
        ] {
            let mut diag = make_diagnostic_result();
            diag.diagnostic_type = dtype.clone();
            let json = serde_json::to_string(&diag).unwrap();
            let back: DiagnosticResult = serde_json::from_str(&json).unwrap();
            assert_eq!(back.diagnostic_type, dtype);
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP — PrivacyPreference
    // ========================================================================

    #[test]
    fn serde_roundtrip_privacy_preference() {
        let pref = make_privacy_preference();
        let json = serde_json::to_string(&pref).unwrap();
        let back: PrivacyPreference = serde_json::from_str(&json).unwrap();
        assert_eq!(pref, back);
    }

    #[test]
    fn serde_roundtrip_privacy_preference_all_tiers() {
        for tier in [
            SharingTier::LocalOnly,
            SharingTier::Anonymized,
            SharingTier::Full,
        ] {
            let mut pref = make_privacy_preference();
            pref.sharing_tier = tier.clone();
            let json = serde_json::to_string(&pref).unwrap();
            let back: PrivacyPreference = serde_json::from_str(&json).unwrap();
            assert_eq!(back.sharing_tier, tier);
        }
    }

    #[test]
    fn serde_roundtrip_privacy_preference_all_bools_false() {
        let mut pref = make_privacy_preference();
        pref.share_system_info = false;
        pref.share_resolution_patterns = false;
        pref.share_cognitive_updates = false;
        let json = serde_json::to_string(&pref).unwrap();
        let back: PrivacyPreference = serde_json::from_str(&json).unwrap();
        assert!(!back.share_system_info);
        assert!(!back.share_resolution_patterns);
        assert!(!back.share_cognitive_updates);
    }

    // ========================================================================
    // SERDE ROUNDTRIP — CognitiveUpdate
    // ========================================================================

    #[test]
    fn serde_roundtrip_cognitive_update() {
        let cu = make_cognitive_update();
        let json = serde_json::to_string(&cu).unwrap();
        let back: CognitiveUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(cu, back);
    }

    #[test]
    fn serde_roundtrip_cognitive_update_nonzero_encoding() {
        let mut cu = make_cognitive_update();
        cu.encoding = vec![0xff; 2048];
        let json = serde_json::to_string(&cu).unwrap();
        let back: CognitiveUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.encoding.len(), 2048);
        assert!(back.encoding.iter().all(|&b| b == 0xff));
    }

    #[test]
    fn serde_roundtrip_cognitive_update_negative_phi() {
        let mut cu = make_cognitive_update();
        cu.phi = -0.5;
        let json = serde_json::to_string(&cu).unwrap();
        let back: CognitiveUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.phi, -0.5);
    }

    // ========================================================================
    // VALIDATION — DiagnosticResult
    // ========================================================================

    #[test]
    fn valid_diagnostic_result_passes() {
        let result = validate_create_diagnostic(fake_create(), make_diagnostic_result());
        assert_valid(&result);
    }

    #[test]
    fn diagnostic_result_invalid_json_rejected() {
        let mut diag = make_diagnostic_result();
        diag.findings = "not valid json {{{".to_string();
        let result = validate_create_diagnostic(fake_create(), diag);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "DiagnosticResult findings must be valid JSON"
        );
    }

    #[test]
    fn diagnostic_result_empty_json_object_accepted() {
        let mut diag = make_diagnostic_result();
        diag.findings = "{}".to_string();
        let result = validate_create_diagnostic(fake_create(), diag);
        assert_valid(&result);
    }

    #[test]
    fn diagnostic_result_json_array_accepted() {
        let mut diag = make_diagnostic_result();
        diag.findings = r#"[1,2,3]"#.to_string();
        let result = validate_create_diagnostic(fake_create(), diag);
        assert_valid(&result);
    }

    #[test]
    fn diagnostic_result_json_string_accepted() {
        let mut diag = make_diagnostic_result();
        diag.findings = r#""just a string""#.to_string();
        let result = validate_create_diagnostic(fake_create(), diag);
        assert_valid(&result);
    }

    #[test]
    fn diagnostic_result_findings_exactly_32768_accepted() {
        let mut diag = make_diagnostic_result();
        // Create a valid JSON string that is exactly 32768 bytes
        // A JSON string "x...x" with 32766 x's + 2 quotes = 32768
        let inner = "x".repeat(32766);
        diag.findings = format!("\"{}\"", inner);
        assert_eq!(diag.findings.len(), 32768);
        let result = validate_create_diagnostic(fake_create(), diag);
        assert_valid(&result);
    }

    #[test]
    fn diagnostic_result_findings_32769_rejected() {
        let mut diag = make_diagnostic_result();
        let inner = "x".repeat(32767);
        diag.findings = format!("\"{}\"", inner);
        assert_eq!(diag.findings.len(), 32769);
        let result = validate_create_diagnostic(fake_create(), diag);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "DiagnosticResult findings must be 32768 bytes or fewer"
        );
    }

    #[test]
    fn diagnostic_result_update_valid() {
        let result =
            validate_update_diagnostic(fake_update(), make_diagnostic_result(), fake_action_hash());
        assert_valid(&result);
    }

    #[test]
    fn diagnostic_result_update_invalid_json_rejected() {
        let mut diag = make_diagnostic_result();
        diag.findings = "bad json".to_string();
        let result = validate_update_diagnostic(fake_update(), diag, fake_action_hash());
        assert_invalid(&result);
    }

    // ========================================================================
    // VALIDATION — PrivacyPreference
    // ========================================================================

    #[test]
    fn valid_privacy_preference_passes() {
        let result = validate_create_privacy_preference(fake_create(), make_privacy_preference());
        assert_valid(&result);
    }

    #[test]
    fn privacy_preference_empty_categories_rejected() {
        let mut pref = make_privacy_preference();
        pref.allowed_categories = vec![];
        let result = validate_create_privacy_preference(fake_create(), pref);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "PrivacyPreference allowed_categories must not be empty"
        );
    }

    #[test]
    fn privacy_preference_single_category_accepted() {
        let mut pref = make_privacy_preference();
        pref.allowed_categories = vec![SupportCategory::Security];
        let result = validate_create_privacy_preference(fake_create(), pref);
        assert_valid(&result);
    }

    #[test]
    fn privacy_preference_all_categories_accepted() {
        let mut pref = make_privacy_preference();
        pref.allowed_categories = vec![
            SupportCategory::Network,
            SupportCategory::Hardware,
            SupportCategory::Software,
            SupportCategory::Holochain,
            SupportCategory::Mycelix,
            SupportCategory::Security,
            SupportCategory::General,
        ];
        let result = validate_create_privacy_preference(fake_create(), pref);
        assert_valid(&result);
    }

    #[test]
    fn privacy_preference_update_valid() {
        let result = validate_update_privacy_preference(
            fake_update(),
            make_privacy_preference(),
            fake_action_hash(),
        );
        assert_valid(&result);
    }

    #[test]
    fn privacy_preference_update_empty_categories_rejected() {
        let mut pref = make_privacy_preference();
        pref.allowed_categories = vec![];
        let result = validate_update_privacy_preference(fake_update(), pref, fake_action_hash());
        assert_invalid(&result);
    }

    // ========================================================================
    // VALIDATION — CognitiveUpdate
    // ========================================================================

    #[test]
    fn valid_cognitive_update_passes() {
        let result = validate_create_cognitive_update(fake_create(), make_cognitive_update());
        assert_valid(&result);
    }

    #[test]
    fn cognitive_update_encoding_too_short_rejected() {
        let mut cu = make_cognitive_update();
        cu.encoding = vec![0u8; 2047];
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
        assert!(invalid_msg(&result).contains("exactly 2048 bytes"));
        assert!(invalid_msg(&result).contains("2047"));
    }

    #[test]
    fn cognitive_update_encoding_too_long_rejected() {
        let mut cu = make_cognitive_update();
        cu.encoding = vec![0u8; 2049];
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
        assert!(invalid_msg(&result).contains("exactly 2048 bytes"));
        assert!(invalid_msg(&result).contains("2049"));
    }

    #[test]
    fn cognitive_update_encoding_empty_rejected() {
        let mut cu = make_cognitive_update();
        cu.encoding = vec![];
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
    }

    #[test]
    fn cognitive_update_phi_nan_rejected() {
        let mut cu = make_cognitive_update();
        cu.phi = f64::NAN;
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
        assert_eq!(invalid_msg(&result), "CognitiveUpdate phi must be finite");
    }

    #[test]
    fn cognitive_update_phi_positive_infinity_rejected() {
        let mut cu = make_cognitive_update();
        cu.phi = f64::INFINITY;
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
        assert_eq!(invalid_msg(&result), "CognitiveUpdate phi must be finite");
    }

    #[test]
    fn cognitive_update_phi_negative_infinity_rejected() {
        let mut cu = make_cognitive_update();
        cu.phi = f64::NEG_INFINITY;
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
        assert_eq!(invalid_msg(&result), "CognitiveUpdate phi must be finite");
    }

    #[test]
    fn cognitive_update_phi_zero_accepted() {
        let mut cu = make_cognitive_update();
        cu.phi = 0.0;
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_valid(&result);
    }

    #[test]
    fn cognitive_update_phi_negative_accepted() {
        let mut cu = make_cognitive_update();
        cu.phi = -1.0;
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_valid(&result);
    }

    #[test]
    fn cognitive_update_phi_large_accepted() {
        let mut cu = make_cognitive_update();
        cu.phi = 999999.99;
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_valid(&result);
    }

    #[test]
    fn cognitive_update_update_valid() {
        let result = validate_update_cognitive_update(
            fake_update(),
            make_cognitive_update(),
            fake_action_hash(),
        );
        assert_valid(&result);
    }

    #[test]
    fn cognitive_update_update_wrong_encoding_rejected() {
        let mut cu = make_cognitive_update();
        cu.encoding = vec![0u8; 1024];
        let result = validate_update_cognitive_update(fake_update(), cu, fake_action_hash());
        assert_invalid(&result);
    }

    #[test]
    fn cognitive_update_update_nan_phi_rejected() {
        let mut cu = make_cognitive_update();
        cu.phi = f64::NAN;
        let result = validate_update_cognitive_update(fake_update(), cu, fake_action_hash());
        assert_invalid(&result);
    }

    // ========================================================================
    // LINK TAG BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn link_tag_256_bytes_is_within_limit() {
        // Verifies that a 256-byte tag would pass our validation threshold
        let tag = LinkTag::new(vec![0u8; 256]);
        assert!(tag.0.len() <= 256);
    }

    #[test]
    fn link_tag_257_bytes_exceeds_limit() {
        // Verifies that a 257-byte tag would fail our validation threshold
        let tag = LinkTag::new(vec![0u8; 257]);
        assert!(tag.0.len() > 256);
    }

    #[test]
    fn link_tag_empty_within_limit() {
        let tag = LinkTag::new(vec![]);
        assert!(tag.0.len() <= 256);
    }

    // ========================================================================
    // SERDE ROUNDTRIP — HelperProfile
    // ========================================================================

    #[test]
    fn serde_roundtrip_helper_profile() {
        let hp = make_helper_profile();
        let json = serde_json::to_string(&hp).unwrap();
        let back: HelperProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(hp, back);
    }

    // ========================================================================
    // VALIDATION — HelperProfile
    // ========================================================================

    #[test]
    fn valid_helper_profile_passes() {
        let result = validate_helper_profile(fake_create(), make_helper_profile());
        assert_valid(&result);
    }

    #[test]
    fn helper_empty_categories_rejected() {
        let mut hp = make_helper_profile();
        hp.expertise_categories = vec![];
        let result = validate_helper_profile(fake_create(), hp);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "HelperProfile must have at least one expertise category"
        );
    }

    #[test]
    fn helper_zero_max_concurrent_rejected() {
        let mut hp = make_helper_profile();
        hp.max_concurrent = 0;
        let result = validate_helper_profile(fake_create(), hp);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "HelperProfile max_concurrent must be at least 1"
        );
    }

    #[test]
    fn helper_single_category_accepted() {
        let mut hp = make_helper_profile();
        hp.expertise_categories = vec![SupportCategory::Security];
        let result = validate_helper_profile(fake_create(), hp);
        assert_valid(&result);
    }

    #[test]
    fn helper_all_categories_accepted() {
        let mut hp = make_helper_profile();
        hp.expertise_categories = vec![
            SupportCategory::Network,
            SupportCategory::Hardware,
            SupportCategory::Software,
            SupportCategory::Holochain,
            SupportCategory::Mycelix,
            SupportCategory::Security,
            SupportCategory::General,
        ];
        let result = validate_helper_profile(fake_create(), hp);
        assert_valid(&result);
    }

    #[test]
    fn helper_update_valid() {
        let result = validate_update_helper_profile(
            fake_update(),
            make_helper_profile(),
            fake_action_hash(),
        );
        assert_valid(&result);
    }

    #[test]
    fn helper_update_empty_categories_rejected() {
        let mut hp = make_helper_profile();
        hp.expertise_categories = vec![];
        let result = validate_update_helper_profile(fake_update(), hp, fake_action_hash());
        assert_invalid(&result);
    }

    // ========================================================================
    // LINK TAG BOUNDARY TESTS — AllHelpers
    // ========================================================================

    #[test]
    fn all_helpers_link_tag_256_bytes_within_limit() {
        let tag = LinkTag::new(vec![0u8; 256]);
        assert!(tag.0.len() <= 256);
    }

    #[test]
    fn all_helpers_link_tag_257_bytes_exceeds_limit() {
        let tag = LinkTag::new(vec![0u8; 257]);
        assert!(tag.0.len() > 256);
    }

    // ========================================================================
    // CLONE / EQUALITY TESTS
    // ========================================================================

    #[test]
    fn diagnostic_result_clone_equals() {
        let diag = make_diagnostic_result();
        let cloned = diag.clone();
        assert_eq!(diag, cloned);
    }

    #[test]
    fn privacy_preference_clone_equals() {
        let pref = make_privacy_preference();
        let cloned = pref.clone();
        assert_eq!(pref, cloned);
    }

    #[test]
    fn cognitive_update_clone_equals() {
        let cu = make_cognitive_update();
        let cloned = cu.clone();
        assert_eq!(cu, cloned);
    }

    #[test]
    fn helper_profile_clone_equals() {
        let hp = make_helper_profile();
        let cloned = hp.clone();
        assert_eq!(hp, cloned);
    }

    #[test]
    fn anchor_clone_equals() {
        let anchor = Anchor("test".to_string());
        let cloned = anchor.clone();
        assert_eq!(anchor, cloned);
    }

    // ========================================================================
    // DELETE AUTHORIZATION TESTS
    // ========================================================================

    #[test]
    fn delete_action_struct_has_deletes_address_field() {
        let delete = Delete {
            author: fake_agent(),
            timestamp: fake_timestamp(),
            action_seq: 1,
            prev_action: fake_action_hash(),
            deletes_address: fake_action_hash(),
            deletes_entry_address: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        };
        assert_eq!(delete.deletes_address, fake_action_hash());
        assert_eq!(delete.author, fake_agent());
    }

    #[test]
    fn delete_link_action_struct_has_link_add_address_field() {
        let delete_link = DeleteLink {
            author: fake_agent(),
            timestamp: fake_timestamp(),
            action_seq: 2,
            prev_action: fake_action_hash(),
            link_add_address: fake_action_hash(),
            base_address: AnyLinkableHash::from(fake_entry_hash()),
        };
        assert_eq!(delete_link.link_add_address, fake_action_hash());
        assert_eq!(delete_link.author, fake_agent());
    }

    // ========================================================================
    // STRING LENGTH BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn diagnostic_recommendation_at_limit_accepted() {
        let mut diag = make_diagnostic_result();
        diag.recommendations = vec!["x".repeat(256)];
        let result = validate_create_diagnostic(fake_create(), diag);
        assert_valid(&result);
    }

    #[test]
    fn diagnostic_recommendation_over_limit_rejected() {
        let mut diag = make_diagnostic_result();
        diag.recommendations = vec!["x".repeat(257)];
        let result = validate_create_diagnostic(fake_create(), diag);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "Recommendation too long (max 256 chars per item)"
        );
    }

    #[test]
    fn diagnostic_update_recommendation_over_limit_rejected() {
        let mut diag = make_diagnostic_result();
        diag.recommendations = vec!["x".repeat(257)];
        let result = validate_update_diagnostic(fake_update(), diag, fake_action_hash());
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "Recommendation too long (max 256 chars per item)"
        );
    }

    #[test]
    fn cognitive_update_resolution_pattern_empty_rejected() {
        let mut cu = make_cognitive_update();
        cu.resolution_pattern = String::new();
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "CognitiveUpdate resolution_pattern cannot be empty"
        );
    }

    #[test]
    fn cognitive_update_resolution_pattern_whitespace_rejected() {
        let mut cu = make_cognitive_update();
        cu.resolution_pattern = "   ".to_string();
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "CognitiveUpdate resolution_pattern cannot be empty"
        );
    }

    #[test]
    fn cognitive_update_resolution_pattern_at_limit_accepted() {
        let mut cu = make_cognitive_update();
        cu.resolution_pattern = "x".repeat(4096);
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_valid(&result);
    }

    #[test]
    fn cognitive_update_resolution_pattern_over_limit_rejected() {
        let mut cu = make_cognitive_update();
        cu.resolution_pattern = "x".repeat(4097);
        let result = validate_create_cognitive_update(fake_create(), cu);
        assert_invalid(&result);
        assert_eq!(
            invalid_msg(&result),
            "CognitiveUpdate resolution_pattern too long (max 4096 chars)"
        );
    }

    #[test]
    fn cognitive_update_update_resolution_pattern_empty_rejected() {
        let mut cu = make_cognitive_update();
        cu.resolution_pattern = String::new();
        let result = validate_update_cognitive_update(fake_update(), cu, fake_action_hash());
        assert_invalid(&result);
    }

    #[test]
    fn cognitive_update_update_resolution_pattern_over_limit_rejected() {
        let mut cu = make_cognitive_update();
        cu.resolution_pattern = "x".repeat(4097);
        let result = validate_update_cognitive_update(fake_update(), cu, fake_action_hash());
        assert_invalid(&result);
    }
}
