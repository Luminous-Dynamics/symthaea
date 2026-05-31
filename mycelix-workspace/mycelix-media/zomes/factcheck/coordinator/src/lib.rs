// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Fact-Check Coordinator Zome
use factcheck_integrity::*;
use hdk::prelude::*;

/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn submit_fact_check(input: SubmitFactCheckInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let check = FactCheck {
        id: format!("factcheck:{}:{}", input.publication_id, now.as_micros()),
        publication_id: input.publication_id.clone(),
        claim_text: input.claim_text.clone(),
        claim_location: input.claim_location,
        epistemic_position: input.epistemic_position,
        verdict: input.verdict,
        evidence: input.evidence,
        checker_did: input.checker_did.clone(),
        checked: now,
    };

    let action_hash = create_entry(&EntryTypes::FactCheck(check))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToFactChecks,
        (),
    )?;
    create_link(
        anchor_hash(&input.checker_did)?,
        action_hash.clone(),
        LinkTypes::CheckerToFactChecks,
        (),
    )?;

    // Link claim text for cross-reference
    create_link(
        anchor_hash(&input.claim_text)?,
        action_hash.clone(),
        LinkTypes::ClaimToFactCheck,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitFactCheckInput {
    pub publication_id: String,
    pub claim_text: String,
    pub claim_location: String,
    pub epistemic_position: EpistemicPosition,
    pub verdict: FactCheckVerdict,
    pub evidence: Vec<EvidenceItem>,
    pub checker_did: String,
}

#[hdk_extern]
pub fn get_publication_fact_checks(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut checks = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToFactChecks as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            checks.push(record);
        }
    }
    Ok(checks)
}

#[hdk_extern]
pub fn search_fact_checks_for_claim(claim_text: String) -> ExternResult<Vec<Record>> {
    let mut checks = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&claim_text)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::ClaimToFactCheck as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            checks.push(record);
        }
    }
    Ok(checks)
}

#[hdk_extern]
pub fn update_source_credibility(input: UpdateCredibilityInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let source = SourceCredibility {
        source_id: input.source_id,
        source_type: input.source_type,
        credibility_score: input.credibility_score,
        verification_count: input.verification_count,
        dispute_count: input.dispute_count,
        last_assessed: now,
    };

    let action_hash = create_entry(&EntryTypes::SourceCredibility(source))?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCredibilityInput {
    pub source_id: String,
    pub source_type: SourceType,
    pub credibility_score: f64,
    pub verification_count: u32,
    pub dispute_count: u32,
}

/// Get a specific fact check by ID
#[hdk_extern]
pub fn get_fact_check(fact_check_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.id == fact_check_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get fact checks by checker
#[hdk_extern]
pub fn get_checker_fact_checks(checker_did: String) -> ExternResult<Vec<Record>> {
    let mut checks = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&checker_did)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::CheckerToFactChecks as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            checks.push(record);
        }
    }
    Ok(checks)
}

/// Get fact checks by verdict
#[hdk_extern]
pub fn get_fact_checks_by_verdict(verdict: FactCheckVerdict) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.verdict == verdict {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Dispute a fact check
#[hdk_extern]
pub fn dispute_fact_check(input: DisputeFactCheckInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let dispute = FactCheckDispute {
        id: format!("dispute:{}:{}", input.fact_check_id, now.as_micros()),
        fact_check_id: input.fact_check_id.clone(),
        disputer_did: input.disputer_did,
        reason: input.reason,
        counter_evidence: input.counter_evidence,
        status: DisputeStatus::Pending,
        created: now,
        resolved: None,
    };

    let action_hash = create_entry(&EntryTypes::FactCheckDispute(dispute))?;
    create_link(
        anchor_hash(&input.fact_check_id)?,
        action_hash.clone(),
        LinkTypes::FactCheckToDisputes,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DisputeFactCheckInput {
    pub fact_check_id: String,
    pub disputer_did: String,
    pub reason: String,
    pub counter_evidence: Vec<EvidenceItem>,
}

/// Resolve a fact check dispute
#[hdk_extern]
pub fn resolve_dispute(input: ResolveDisputeInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheckDispute,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(dispute) = record
            .entry()
            .to_app_option::<FactCheckDispute>()
            .ok()
            .flatten()
        {
            if dispute.id == input.dispute_id {
                let now = sys_time()?;
                let updated = FactCheckDispute {
                    status: input.resolution,
                    resolved: Some(now),
                    ..dispute
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::FactCheckDispute(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Dispute not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub dispute_id: String,
    pub resolution: DisputeStatus,
}

/// Get disputes for a fact check
#[hdk_extern]
pub fn get_fact_check_disputes(fact_check_id: String) -> ExternResult<Vec<Record>> {
    let mut disputes = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&fact_check_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::FactCheckToDisputes as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            disputes.push(record);
        }
    }
    Ok(disputes)
}

/// Get source credibility by ID
#[hdk_extern]
pub fn get_source_credibility(source_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::SourceCredibility,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(source) = record
            .entry()
            .to_app_option::<SourceCredibility>()
            .ok()
            .flatten()
        {
            if source.source_id == source_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get checker stats
#[hdk_extern]
pub fn get_checker_stats(checker_did: String) -> ExternResult<CheckerStats> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    let mut total_checks = 0;
    let mut true_count = 0;
    let mut false_count = 0;
    let mut mixed_count = 0;

    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.checker_did == checker_did {
                total_checks += 1;
                match check.verdict {
                    FactCheckVerdict::True => true_count += 1,
                    FactCheckVerdict::False => false_count += 1,
                    FactCheckVerdict::PartiallyTrue | FactCheckVerdict::Misleading => {
                        mixed_count += 1
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(CheckerStats {
        checker_did,
        total_checks,
        true_count,
        false_count,
        mixed_count,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckerStats {
    pub checker_did: String,
    pub total_checks: u32,
    pub true_count: u32,
    pub false_count: u32,
    pub mixed_count: u32,
}

/// Add additional evidence to fact check
#[hdk_extern]
pub fn add_evidence(input: AddEvidenceInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.id == input.fact_check_id {
                // Only checker can add evidence
                if check.checker_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only checker can add evidence".into()
                    )));
                }

                let mut evidence = check.evidence.clone();
                evidence.extend(input.new_evidence);

                let updated = FactCheck { evidence, ..check };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::FactCheck(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Fact check not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddEvidenceInput {
    pub fact_check_id: String,
    pub requester_did: String,
    pub new_evidence: Vec<EvidenceItem>,
}

/// Update fact check verdict
#[hdk_extern]
pub fn update_verdict(input: UpdateVerdictInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::FactCheck,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(check) = record.entry().to_app_option::<FactCheck>().ok().flatten() {
            if check.id == input.fact_check_id {
                if check.checker_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only checker can update verdict".into()
                    )));
                }

                let updated = FactCheck {
                    verdict: input.new_verdict,
                    ..check
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::FactCheck(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Fact check not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateVerdictInput {
    pub fact_check_id: String,
    pub requester_did: String,
    pub new_verdict: FactCheckVerdict,
}
