// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Arbitration Coordinator Zome
use arbitration_integrity::*;
use hdk::prelude::*;

#[hdk_extern]
pub fn initiate_arbitration(input: InitiateArbitrationInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let arb = Arbitration {
        id: format!("arb:{}:{}", input.complaint_id, now.as_micros()),
        complaint_id: input.complaint_id.clone(),
        jurors: input.jurors,
        status: ArbitrationStatus::JurorSelection,
        started: now,
        deadline: Timestamp::from_micros(now.as_micros() as i64 + (7 * 24 * 3600 * 1_000_000)),
    };
    let action_hash = create_entry(&EntryTypes::Arbitration(arb))?;
    create_link(
        hash_entry(&input.complaint_id)?,
        action_hash.clone(),
        LinkTypes::ComplaintToArbitration,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateArbitrationInput {
    pub complaint_id: String,
    pub jurors: Vec<String>,
}

#[hdk_extern]
pub fn issue_judgment(input: IssueJudgmentInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let judgment = Judgment {
        id: format!("judgment:{}:{}", input.arbitration_id, now.as_micros()),
        arbitration_id: input.arbitration_id.clone(),
        decision: input.decision,
        reasoning: input.reasoning,
        penalties: input.penalties,
        issued: now,
    };
    let action_hash = create_entry(&EntryTypes::Judgment(judgment))?;
    create_link(
        hash_entry(&input.arbitration_id)?,
        action_hash.clone(),
        LinkTypes::ArbitrationToJudgment,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IssueJudgmentInput {
    pub arbitration_id: String,
    pub decision: JudgmentDecision,
    pub reasoning: String,
    pub penalties: Vec<Penalty>,
}

/// Get arbitration by ID
#[hdk_extern]
pub fn get_arbitration(arbitration_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Arbitration,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(arb) = record.entry().to_app_option::<Arbitration>().ok().flatten() {
            if arb.id == arbitration_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get arbitration for a complaint
#[hdk_extern]
pub fn get_arbitration_for_complaint(complaint_id: String) -> ExternResult<Vec<Record>> {
    let mut arbitrations = Vec::new();
    for link in get_links(
        GetLinksInputBuilder::try_new(
            hash_entry(&complaint_id)?,
            LinkTypes::ComplaintToArbitration,
        )?
        .build(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            arbitrations.push(record);
        }
    }
    Ok(arbitrations)
}

/// Get judgment for an arbitration
#[hdk_extern]
pub fn get_judgment(arbitration_id: String) -> ExternResult<Vec<Record>> {
    let mut judgments = Vec::new();
    for link in get_links(
        GetLinksInputBuilder::try_new(
            hash_entry(&arbitration_id)?,
            LinkTypes::ArbitrationToJudgment,
        )?
        .build(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            judgments.push(record);
        }
    }
    Ok(judgments)
}

/// Update arbitration status
#[hdk_extern]
pub fn update_arbitration_status(input: UpdateArbitrationStatusInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Arbitration,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(arb) = record.entry().to_app_option::<Arbitration>().ok().flatten() {
            if arb.id == input.arbitration_id {
                let updated = Arbitration {
                    status: input.new_status,
                    ..arb
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Arbitration(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Arbitration not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateArbitrationStatusInput {
    pub arbitration_id: String,
    pub new_status: ArbitrationStatus,
}

/// Extend deadline for arbitration
#[hdk_extern]
pub fn extend_deadline(input: ExtendDeadlineInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Arbitration,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(arb) = record.entry().to_app_option::<Arbitration>().ok().flatten() {
            if arb.id == input.arbitration_id {
                let new_deadline = Timestamp::from_micros(
                    arb.deadline.as_micros() as i64
                        + (input.extension_days as i64 * 24 * 3600 * 1_000_000),
                );
                let updated = Arbitration {
                    deadline: new_deadline,
                    ..arb
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Arbitration(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Arbitration not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExtendDeadlineInput {
    pub arbitration_id: String,
    pub extension_days: u32,
}

/// Assign additional juror to arbitration
#[hdk_extern]
pub fn assign_juror(input: AssignJurorInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Arbitration,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(mut arb) = record.entry().to_app_option::<Arbitration>().ok().flatten() {
            if arb.id == input.arbitration_id {
                if !arb.jurors.contains(&input.juror_id) {
                    arb.jurors.push(input.juror_id);
                }
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Arbitration(arb),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Arbitration not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AssignJurorInput {
    pub arbitration_id: String,
    pub juror_id: String,
}

/// Remove juror from arbitration (recusal or disqualification)
#[hdk_extern]
pub fn remove_juror(input: AssignJurorInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Arbitration,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(mut arb) = record.entry().to_app_option::<Arbitration>().ok().flatten() {
            if arb.id == input.arbitration_id {
                arb.jurors.retain(|j| j != &input.juror_id);
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Arbitration(arb),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Arbitration not found".into()
    )))
}

/// Get all arbitrations by status (for panel queue management)
#[hdk_extern]
pub fn get_arbitrations_by_status(status: ArbitrationStatus) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Arbitration,
        )?))
        .include_entries(true);

    let mut arbitrations = Vec::new();
    for record in query(filter)? {
        if let Some(arb) = record.entry().to_app_option::<Arbitration>().ok().flatten() {
            if arb.status == status {
                arbitrations.push(record);
            }
        }
    }
    Ok(arbitrations)
}

/// Submit juror vote on a decision
#[hdk_extern]
pub fn submit_juror_vote(input: JurorVoteInput) -> ExternResult<Record> {
    let vote = JurorVote {
        id: format!(
            "vote:{}:{}:{}",
            input.arbitration_id,
            input.juror_id,
            sys_time()?.as_micros()
        ),
        arbitration_id: input.arbitration_id.clone(),
        juror_id: input.juror_id.clone(),
        decision: input.decision,
        reasoning: input.reasoning,
        submitted: sys_time()?,
    };
    let action_hash = create_entry(&EntryTypes::JurorVote(vote))?;
    create_link(
        hash_entry(&input.arbitration_id)?,
        action_hash.clone(),
        LinkTypes::ArbitrationToVote,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JurorVoteInput {
    pub arbitration_id: String,
    pub juror_id: String,
    pub decision: JudgmentDecision,
    pub reasoning: String,
}

/// Get all votes for an arbitration
#[hdk_extern]
pub fn get_arbitration_votes(arbitration_id: String) -> ExternResult<Vec<Record>> {
    let mut votes = Vec::new();
    for link in get_links(
        GetLinksInputBuilder::try_new(hash_entry(&arbitration_id)?, LinkTypes::ArbitrationToVote)?
            .build(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            votes.push(record);
        }
    }
    Ok(votes)
}
