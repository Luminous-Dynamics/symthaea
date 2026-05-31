// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Fact-Check Integrity Zome
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
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::PublicationToFactChecks => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CheckerToFactChecks => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToFactCheck => Ok(ValidateCallbackResult::Valid),
            LinkTypes::FactCheckToDisputes => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
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
    if !check.checker_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Checker must be a valid DID".into(),
        ));
    }
    if check.claim_text.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim text required".into(),
        ));
    }
    let ep = &check.epistemic_position;
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
    if !dispute.disputer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Disputer must be a valid DID".into(),
        ));
    }
    if dispute.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute reason required".into(),
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
