// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Arbitration Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Arbitration {
    pub id: String,
    pub complaint_id: String,
    pub jurors: Vec<String>,
    pub status: ArbitrationStatus,
    pub started: Timestamp,
    pub deadline: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Judgment {
    pub id: String,
    pub arbitration_id: String,
    pub decision: JudgmentDecision,
    pub reasoning: String,
    pub penalties: Vec<Penalty>,
    pub issued: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ArbitrationStatus {
    JurorSelection,
    EvidenceReview,
    Deliberation,
    Decided,
    Appealed,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum JudgmentDecision {
    ForComplainant,
    ForRespondent,
    PartialForComplainant,
    Dismissed,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Penalty {
    pub penalty_type: String,
    pub amount: Option<f64>,
    pub description: String,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct JurorVote {
    pub id: String,
    pub arbitration_id: String,
    pub juror_id: String,
    pub decision: JudgmentDecision,
    pub reasoning: String,
    pub submitted: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Arbitration(Arbitration),
    Judgment(Judgment),
    JurorVote(JurorVote),
}

#[hdk_link_types]
pub enum LinkTypes {
    ComplaintToArbitration,
    ArbitrationToJudgment,
    JurorToArbitration,
    ArbitrationToVote,
}

#[hdk_extern]
pub fn validate_create_entry_arbitration(
    _action: EntryCreationAction,
    arb: Arbitration,
) -> ExternResult<ValidateCallbackResult> {
    if arb.jurors.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Jurors required".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate_create_entry_judgment(
    _action: EntryCreationAction,
    _judgment: Judgment,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
