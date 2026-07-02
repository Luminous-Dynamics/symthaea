// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Constitution Integrity Zome
//! Defines entry types and validation for charter and amendments
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// The constitutional charter
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Charter {
    /// Charter identifier
    pub id: String,
    /// Charter version
    pub version: u32,
    /// Preamble
    pub preamble: String,
    /// Articles (JSON array)
    pub articles: String,
    /// Fundamental rights
    pub rights: Vec<String>,
    /// Amendment process
    pub amendment_process: String,
    /// Adoption timestamp
    pub adopted: Timestamp,
    /// Last amendment timestamp
    pub last_amended: Option<Timestamp>,
}

/// A constitutional amendment
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Amendment {
    /// Amendment identifier
    pub id: String,
    /// Charter version this amends
    pub charter_version: u32,
    /// Amendment type
    pub amendment_type: AmendmentType,
    /// Article being amended (if applicable)
    pub article: Option<String>,
    /// Original text
    pub original_text: Option<String>,
    /// New text
    pub new_text: String,
    /// Rationale
    pub rationale: String,
    /// Proposer's DID
    pub proposer: String,
    /// Linked proposal ID
    pub proposal_id: String,
    /// Amendment status
    pub status: AmendmentStatus,
    /// Creation timestamp
    pub created: Timestamp,
    /// Ratification timestamp
    pub ratified: Option<Timestamp>,
}

/// Types of amendments
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AmendmentType {
    /// Add new article
    AddArticle,
    /// Modify existing article
    ModifyArticle,
    /// Remove article
    RemoveArticle,
    /// Add right
    AddRight,
    /// Modify right
    ModifyRight,
    /// Remove right
    RemoveRight,
    /// Modify preamble
    ModifyPreamble,
    /// Modify amendment process itself
    ModifyProcess,
}

/// Status of an amendment
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AmendmentStatus {
    /// Draft stage
    Draft,
    /// Under deliberation
    Deliberation,
    /// Voting in progress
    Voting,
    /// Ratified and in effect
    Ratified,
    /// Rejected by vote
    Rejected,
    /// Withdrawn by proposer
    Withdrawn,
}

/// Governance parameter
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GovernanceParameter {
    /// Parameter name
    pub name: String,
    /// Current value (JSON)
    pub value: String,
    /// Value type
    pub value_type: ParameterType,
    /// Description
    pub description: String,
    /// Minimum value (if applicable)
    pub min_value: Option<String>,
    /// Maximum value (if applicable)
    pub max_value: Option<String>,
    /// Last update timestamp
    pub updated: Timestamp,
    /// Proposal that last changed this
    pub changed_by_proposal: Option<String>,
}

/// Types of parameters
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ParameterType {
    Integer,
    Float,
    Percentage,
    Duration,
    Boolean,
    String,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Charter(Charter),
    Amendment(Amendment),
    GovernanceParameter(GovernanceParameter),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Current charter
    CurrentCharter,
    /// Charter version history
    CharterHistory,
    /// Charter to amendments
    CharterToAmendment,
    /// Parameter name to parameter
    ParameterIndex,
    /// O(1) lookup: amendment ID anchor → amendment record
    AmendmentById,
}

// ── Pure check functions (no HDI host calls) ───────────────────────────

/// Check charter validity on creation.
pub fn check_create_charter(charter: &Charter) -> Result<(), String> {
    if charter.preamble.is_empty() {
        return Err("Charter must have a preamble".into());
    }
    if serde_json::from_str::<serde_json::Value>(&charter.articles).is_err() {
        return Err("Articles must be valid JSON".into());
    }
    if charter.rights.is_empty() {
        return Err("Charter must define fundamental rights".into());
    }
    Ok(())
}

/// Check charter validity on update (compares original and updated).
pub fn check_update_charter(original: &Charter, updated: &Charter) -> Result<(), String> {
    if updated.id != original.id {
        return Err("Cannot change charter ID".into());
    }
    if updated.version != original.version + 1 {
        return Err(format!(
            "Charter version must increment by 1 (expected {}, got {})",
            original.version + 1,
            updated.version
        ));
    }
    if updated.preamble.is_empty() {
        return Err("Charter must have a preamble".into());
    }
    if serde_json::from_str::<serde_json::Value>(&updated.articles).is_err() {
        return Err("Articles must be valid JSON".into());
    }
    if updated.rights.is_empty() {
        return Err("Charter must define fundamental rights".into());
    }
    if updated.adopted != original.adopted {
        return Err("Cannot change charter adoption date".into());
    }
    Ok(())
}

/// Check amendment validity on creation.
///
/// Immutable Core rights — aligned with Constitution Art. IV, Sec. 2.
/// These are NOT absolutely unamendable but require the enhanced
/// amendment process: 90% supermajority in both Global DAO houses,
/// ratified by 3/4 of all Sector and Regional DAOs.
///
/// At the integrity level, amendments targeting these rights are
/// FLAGGED (not rejected) — the coordinator must verify that the
/// enhanced process was followed before allowing the amendment.
const IMMUTABLE_CORE_RIGHTS: &[&str] = &[
    "veto override",
    "consciousness gating",
    "term limits",
    "emergency power limits",
    "permission-less enforcement",
    "fork rights",
    "right to exit",
    // Constitutional Immutable Core (Art. IV, Sec. 2):
    "core principles",
    "sovereignty",
    "golden veto sunset",
    "oversight funding",
];

/// Check if an amendment targets the Immutable Core (Art. IV, Sec. 2).
///
/// Returns Some(reason) if the amendment targets a protected right.
/// The amendment is not REJECTED at the integrity level — instead,
/// the coordinator must verify the enhanced amendment process
/// (90% supermajority + 3/4 ratification) was followed.
///
/// This aligns with the Constitution: the Immutable Core is not
/// absolutely unamendable, but requires an extraordinary process.
pub fn targets_immutable_core(amendment: &Amendment) -> Option<String> {
    let text_lower = amendment.new_text.to_lowercase();
    let original_lower = amendment
        .original_text
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    let article_lower = amendment
        .article
        .as_deref()
        .unwrap_or("")
        .to_lowercase();

    // Check RemoveRight/ModifyRight against immutable core rights
    if matches!(
        amendment.amendment_type,
        AmendmentType::RemoveRight | AmendmentType::ModifyRight
    ) {
        for right in IMMUTABLE_CORE_RIGHTS {
            if original_lower.contains(right) || text_lower.contains(right) {
                return Some(format!(
                    "Amendment targets Immutable Core right: '{}'. \
                     Requires enhanced process: 90% supermajority + 3/4 DAO ratification \
                     (Constitution Art. IV, Sec. 2).",
                    right
                ));
            }
        }
    }

    // Check RemoveArticle against immutable core topics
    if matches!(amendment.amendment_type, AmendmentType::RemoveArticle) {
        for right in IMMUTABLE_CORE_RIGHTS {
            if article_lower.contains(right) {
                return Some(format!(
                    "Amendment targets Immutable Core article: '{}'. \
                     Requires enhanced process (Art. IV, Sec. 2).",
                    right
                ));
            }
        }
    }

    // ModifyProcess cannot strip override mechanisms without enhanced process
    if matches!(amendment.amendment_type, AmendmentType::ModifyProcess) {
        for right in IMMUTABLE_CORE_RIGHTS {
            if original_lower.contains(right) && !text_lower.contains(right) {
                return Some(format!(
                    "Amendment removes '{}' from process — requires enhanced process \
                     (Art. IV, Sec. 2).",
                    right
                ));
            }
        }
    }

    None
}

pub fn check_create_amendment(amendment: &Amendment) -> Result<(), String> {
    if !amendment.proposer.starts_with("did:") {
        return Err("Proposer must be a valid DID".into());
    }
    if amendment.new_text.is_empty() {
        return Err("Amendment must have new text".into());
    }
    if amendment.rationale.is_empty() {
        return Err("Amendment must have rationale".into());
    }

    // ── IMMUTABLE CORE CHECK (Art. IV, Sec. 2) ──
    // Flag amendments targeting the Immutable Core. These are not rejected
    // at the integrity level — the coordinator verifies the enhanced process
    // (90% supermajority + 3/4 ratification) was followed.
    //
    // The amendment is VALID to create (so it can enter deliberation),
    // but the coordinator must check targets_immutable_core() before
    // transitioning to Ratified status.
    //
    // This aligns with the Constitution: even the Immutable Core can be
    // amended through an extraordinary democratic process.

    match amendment.amendment_type {
        AmendmentType::ModifyArticle
        | AmendmentType::ModifyRight
        | AmendmentType::ModifyPreamble => {
            if amendment.original_text.is_none() {
                return Err("Modification amendments must include original text".into());
            }
        }
        _ => {}
    }
    Ok(())
}

/// Check amendment validity on update (compares original and updated).
pub fn check_update_amendment(original: &Amendment, updated: &Amendment) -> Result<(), String> {
    if updated.id != original.id {
        return Err("Cannot change amendment ID".into());
    }
    if updated.proposer != original.proposer {
        return Err("Cannot change amendment proposer".into());
    }
    if updated.proposal_id != original.proposal_id {
        return Err("Cannot change amendment proposal_id".into());
    }
    if updated.status != original.status {
        let valid = matches!(
            (&original.status, &updated.status),
            (AmendmentStatus::Draft, AmendmentStatus::Deliberation)
                | (AmendmentStatus::Draft, AmendmentStatus::Withdrawn)
                | (AmendmentStatus::Deliberation, AmendmentStatus::Voting)
                | (AmendmentStatus::Deliberation, AmendmentStatus::Withdrawn)
                | (AmendmentStatus::Voting, AmendmentStatus::Ratified)
                | (AmendmentStatus::Voting, AmendmentStatus::Rejected)
        );
        if !valid {
            return Err(format!(
                "Invalid amendment status transition: {:?} -> {:?}",
                original.status, updated.status
            ));
        }
    }
    Ok(())
}

/// Check governance parameter validity on creation.
pub fn check_create_parameter(param: &GovernanceParameter) -> Result<(), String> {
    if param.name.is_empty() {
        return Err("Parameter name cannot be empty".into());
    }
    if serde_json::from_str::<serde_json::Value>(&param.value).is_err() {
        return Err("Parameter value must be valid JSON".into());
    }
    Ok(())
}

/// Check governance parameter validity on update.
pub fn check_update_parameter(param: &GovernanceParameter) -> Result<(), String> {
    if serde_json::from_str::<serde_json::Value>(&param.value).is_err() {
        return Err("Parameter value must be valid JSON".into());
    }
    Ok(())
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Charter(charter) => validate_create_charter(action, charter),
                EntryTypes::Amendment(amendment) => validate_create_amendment(action, amendment),
                EntryTypes::GovernanceParameter(param) => validate_create_parameter(action, param),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Charter(charter) => {
                    validate_update_charter(action, charter, original_action_hash)
                }
                EntryTypes::Amendment(amendment) => {
                    validate_update_amendment(action, amendment, original_action_hash)
                }
                EntryTypes::GovernanceParameter(param) => validate_update_parameter(action, param),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::CurrentCharter => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CharterHistory => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CharterToAmendment => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ParameterIndex => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AmendmentById => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::CurrentCharter => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate charter creation
fn validate_create_charter(
    _action: Create,
    charter: Charter,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_charter(&charter) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate charter update
fn validate_update_charter(
    _action: Update,
    charter: Charter,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_charter: Charter = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original charter not found".into()
        )))?;

    match check_update_charter(&original_charter, &charter) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate amendment creation
fn validate_create_amendment(
    _action: Create,
    amendment: Amendment,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_amendment(&amendment) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate amendment update — enforce status transition whitelist
fn validate_update_amendment(
    _action: Update,
    amendment: Amendment,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_amendment: Amendment = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original amendment not found".into()
        )))?;

    match check_update_amendment(&original_amendment, &amendment) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate governance parameter creation
fn validate_create_parameter(
    _action: Create,
    param: GovernanceParameter,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_parameter(&param) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

/// Validate governance parameter update
fn validate_update_parameter(
    _action: Update,
    param: GovernanceParameter,
) -> ExternResult<ValidateCallbackResult> {
    match check_update_parameter(&param) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(msg) => Ok(ValidateCallbackResult::Invalid(msg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn valid_charter() -> Charter {
        Charter {
            id: "charter-1".into(),
            version: 1,
            preamble: "We the people".into(),
            articles: r#"[{"title":"Article 1"}]"#.into(),
            rights: vec!["Right to dignity".into()],
            amendment_process: "2/3 vote".into(),
            adopted: ts(1000000),
            last_amended: None,
        }
    }

    fn valid_amendment() -> Amendment {
        Amendment {
            id: "amend-1".into(),
            charter_version: 1,
            amendment_type: AmendmentType::AddArticle,
            article: None,
            original_text: None,
            new_text: "New article text".into(),
            rationale: "Needed for clarity".into(),
            proposer: "did:key:z6Mk...".into(),
            proposal_id: "prop-1".into(),
            status: AmendmentStatus::Draft,
            created: ts(2000000),
            ratified: None,
        }
    }

    fn valid_parameter() -> GovernanceParameter {
        GovernanceParameter {
            name: "quorum".into(),
            value: r#"{"threshold": 0.67}"#.into(),
            value_type: ParameterType::Percentage,
            description: "Minimum quorum for votes".into(),
            min_value: Some("0.5".into()),
            max_value: Some("1.0".into()),
            updated: ts(3000000),
            changed_by_proposal: None,
        }
    }

    #[test]
    fn test_valid_charter_accepted() {
        assert!(check_create_charter(&valid_charter()).is_ok());
    }

    #[test]
    fn test_charter_preamble_required() {
        let mut c = valid_charter();
        c.preamble = String::new();
        let err = check_create_charter(&c).unwrap_err();
        assert!(err.contains("preamble"));
    }

    #[test]
    fn test_charter_articles_must_be_json() {
        let mut c = valid_charter();
        c.articles = "not json {{{".into();
        let err = check_create_charter(&c).unwrap_err();
        assert!(err.contains("JSON"));
    }

    #[test]
    fn test_charter_must_have_rights() {
        let mut c = valid_charter();
        c.rights = vec![];
        let err = check_create_charter(&c).unwrap_err();
        assert!(err.contains("rights"));
    }

    #[test]
    fn test_charter_update_version_must_increment() {
        let original = valid_charter();
        let mut updated = original.clone();
        updated.version = 5; // jump from 1 to 5 instead of 2
        let err = check_update_charter(&original, &updated).unwrap_err();
        assert!(err.contains("increment by 1"));
    }

    #[test]
    fn test_charter_update_id_immutable() {
        let original = valid_charter();
        let mut updated = original.clone();
        updated.version = 2;
        updated.id = "different-id".into();
        let err = check_update_charter(&original, &updated).unwrap_err();
        assert!(err.contains("charter ID"));
    }

    #[test]
    fn test_amendment_proposer_must_be_did() {
        let mut a = valid_amendment();
        a.proposer = "not-a-did".into();
        let err = check_create_amendment(&a).unwrap_err();
        assert!(err.contains("DID"));
    }

    #[test]
    fn test_amendment_new_text_required() {
        let mut a = valid_amendment();
        a.new_text = String::new();
        let err = check_create_amendment(&a).unwrap_err();
        assert!(err.contains("new text"));
    }

    #[test]
    fn test_amendment_modification_requires_original_text() {
        let mut a = valid_amendment();
        a.amendment_type = AmendmentType::ModifyArticle;
        a.original_text = None;
        let err = check_create_amendment(&a).unwrap_err();
        assert!(err.contains("original text"));
    }

    #[test]
    fn test_amendment_valid_status_transitions() {
        let transitions = vec![
            (AmendmentStatus::Draft, AmendmentStatus::Deliberation),
            (AmendmentStatus::Draft, AmendmentStatus::Withdrawn),
            (AmendmentStatus::Deliberation, AmendmentStatus::Voting),
            (AmendmentStatus::Deliberation, AmendmentStatus::Withdrawn),
            (AmendmentStatus::Voting, AmendmentStatus::Ratified),
            (AmendmentStatus::Voting, AmendmentStatus::Rejected),
        ];
        for (from, to) in transitions {
            let mut original = valid_amendment();
            original.status = from;
            let mut updated = original.clone();
            updated.status = to;
            assert!(
                check_update_amendment(&original, &updated).is_ok(),
                "Transition {:?} -> {:?} should be valid",
                original.status,
                updated.status
            );
        }
    }

    #[test]
    fn test_amendment_invalid_status_transition() {
        let mut original = valid_amendment();
        original.status = AmendmentStatus::Voting;
        let mut updated = original.clone();
        updated.status = AmendmentStatus::Draft;
        let err = check_update_amendment(&original, &updated).unwrap_err();
        assert!(err.contains("Invalid amendment status transition"));
    }

    #[test]
    fn test_parameter_name_required() {
        let mut p = valid_parameter();
        p.name = String::new();
        let err = check_create_parameter(&p).unwrap_err();
        assert!(err.contains("name"));
    }

    #[test]
    fn test_parameter_value_must_be_json() {
        let mut p = valid_parameter();
        p.value = "not json!!!".into();
        let err = check_create_parameter(&p).unwrap_err();
        assert!(err.contains("JSON"));
        // Also test the update path
        let err = check_update_parameter(&p).unwrap_err();
        assert!(err.contains("JSON"));
    }
}
