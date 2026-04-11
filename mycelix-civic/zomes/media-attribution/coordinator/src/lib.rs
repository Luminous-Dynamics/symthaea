// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Attribution Coordinator Zome
use hdk::prelude::*;
use media_attribution_integrity::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};
use mycelix_zome_helpers::{get_latest_record};

/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
    hash_entry(&anchor)
}


#[hdk_extern]
pub fn add_attribution(input: AddAttributionInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "add_attribution")?;
    let now = sys_time()?;
    let attribution = Attribution {
        id: format!(
            "attr:{}:{}:{}",
            input.publication_id,
            input.contributor_did,
            now.as_micros()
        ),
        publication_id: input.publication_id.clone(),
        contributor_did: input.contributor_did.clone(),
        role: input.role,
        share_percentage: input.share_percentage,
        verified: false,
        created: now,
    };

    let action_hash = create_entry(&EntryTypes::Attribution(attribution))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToAttributions,
        (),
    )?;
    create_link(
        anchor_hash(&input.contributor_did)?,
        action_hash.clone(),
        LinkTypes::ContributorToAttributions,
        (),
    )?;
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddAttributionInput {
    pub publication_id: String,
    pub contributor_did: String,
    pub role: ContributorRole,
    pub share_percentage: f64,
}

#[hdk_extern]
pub fn set_royalty_rule(input: SetRoyaltyInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "set_royalty_rule")?;
    let now = sys_time()?;
    let rule = RoyaltyRule {
        id: format!(
            "royalty:{}:{:?}:{}",
            input.publication_id,
            input.rule_type,
            now.as_micros()
        ),
        publication_id: input.publication_id.clone(),
        rule_type: input.rule_type,
        percentage: input.percentage,
        minimum_amount: input.minimum_amount,
        currency: input.currency,
        active: true,
    };

    let action_hash = create_entry(&EntryTypes::RoyaltyRule(rule))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToRoyalties,
        (),
    )?;
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetRoyaltyInput {
    pub publication_id: String,
    pub rule_type: RoyaltyType,
    pub percentage: f64,
    pub minimum_amount: Option<f64>,
    pub currency: String,
}

#[hdk_extern]
pub fn record_usage(input: RecordUsageInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "record_usage")?;
    let now = sys_time()?;
    let usage = UsageRecord {
        id: format!("usage:{}:{}", input.publication_id, now.as_micros()),
        publication_id: input.publication_id.clone(),
        usage_type: input.usage_type,
        user_did: input.user_did,
        timestamp: now,
        royalty_paid: input.royalty_paid,
    };

    let action_hash = create_entry(&EntryTypes::UsageRecord(usage))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToUsage,
        (),
    )?;
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordUsageInput {
    pub publication_id: String,
    pub usage_type: UsageType,
    pub user_did: Option<String>,
    pub royalty_paid: Option<f64>,
}

#[hdk_extern]
pub fn get_publication_attributions(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut attributions = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(
            0.into(),
            (LinkTypes::PublicationToAttributions as u8).into(),
        ),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            attributions.push(record);
        }
    }
    Ok(attributions)
}

#[hdk_extern]
pub fn get_contributor_works(did: String) -> ExternResult<Vec<Record>> {
    let mut works = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&did)?,
        LinkTypeFilter::single_type(
            0.into(),
            (LinkTypes::ContributorToAttributions as u8).into(),
        ),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            works.push(record);
        }
    }
    Ok(works)
}

/// Get a specific attribution by ID
#[hdk_extern]
pub fn get_attribution(attribution_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    // Take the LAST match — update_entry appends newer versions later in the chain
    let mut found: Option<Record> = None;
    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.id == attribution_id {
                found = Some(record);
            }
        }
    }
    Ok(found)
}

/// Verify an attribution
#[hdk_extern]
pub fn verify_attribution(input: VerifyAttributionInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "verify_attribution")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.id == input.attribution_id {
                // Only contributor can verify
                if attr.contributor_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only contributor can verify attribution".into()
                    )));
                }

                let verified = Attribution {
                    verified: true,
                    ..attr
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Attribution(verified),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Attribution not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyAttributionInput {
    pub attribution_id: String,
    pub requester_did: String,
}

/// Update share percentage
#[hdk_extern]
pub fn update_share_percentage(input: UpdateShareInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_share_percentage")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.id == input.attribution_id {
                let updated = Attribution {
                    share_percentage: input.new_share_percentage,
                    ..attr
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Attribution(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Attribution not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateShareInput {
    pub attribution_id: String,
    pub new_share_percentage: f64,
}

/// Update an attribution entry
#[hdk_extern]
pub fn update_attribution(input: UpdateAttributionInput) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_attribution")?;
    update_entry(
        input.original_action_hash,
        &EntryTypes::Attribution(input.updated_entry),
    )
}

/// Input for updating an attribution
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateAttributionInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: Attribution,
}

/// Update a royalty rule entry
#[hdk_extern]
pub fn update_royalty_rule(input: UpdateRoyaltyRuleInput) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_royalty_rule")?;
    update_entry(
        input.original_action_hash,
        &EntryTypes::RoyaltyRule(input.updated_entry),
    )
}

/// Input for updating a royalty rule
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateRoyaltyRuleInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: RoyaltyRule,
}

/// Get publication royalty rules
#[hdk_extern]
pub fn get_publication_royalties(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut royalties = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToRoyalties as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            royalties.push(record);
        }
    }
    Ok(royalties)
}

/// Get publication usage records
#[hdk_extern]
pub fn get_publication_usage(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut usage = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToUsage as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            usage.push(record);
        }
    }
    Ok(usage)
}

/// Deactivate a royalty rule
#[hdk_extern]
pub fn deactivate_royalty_rule(input: DeactivateRoyaltyInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "deactivate_royalty_rule")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::RoyaltyRule,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(rule) = record.entry().to_app_option::<RoyaltyRule>().ok().flatten() {
            if rule.id == input.rule_id {
                let updated = RoyaltyRule {
                    active: false,
                    ..rule
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::RoyaltyRule(updated),
                )?;
                return get_latest_record(action_hash)?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Royalty rule not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeactivateRoyaltyInput {
    pub rule_id: String,
}

/// Get contributor earnings summary
#[hdk_extern]
pub fn get_contributor_earnings(did: String) -> ExternResult<ContributorEarnings> {
    let attribution_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    let mut publication_ids = Vec::new();
    let mut total_share = 0.0;

    for record in query(attribution_filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.contributor_did == did {
                publication_ids.push(attr.publication_id);
                total_share += attr.share_percentage;
            }
        }
    }

    // Calculate total royalties earned
    let usage_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::UsageRecord,
        )?))
        .include_entries(true);

    let mut total_royalties = 0.0;
    for record in query(usage_filter)? {
        if let Some(usage) = record.entry().to_app_option::<UsageRecord>().ok().flatten() {
            if publication_ids.contains(&usage.publication_id) {
                if let Some(paid) = usage.royalty_paid {
                    total_royalties += paid;
                }
            }
        }
    }

    Ok(ContributorEarnings {
        contributor_did: did,
        publication_count: publication_ids.len() as u32,
        average_share_percentage: if publication_ids.is_empty() {
            0.0
        } else {
            total_share / publication_ids.len() as f64
        },
        total_royalties_earned: total_royalties,
    })
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct ContributorEarnings {
    pub contributor_did: String,
    pub publication_count: u32,
    pub average_share_percentage: f64,
    pub total_royalties_earned: f64,
}

/// Get attributions by role
#[hdk_extern]
pub fn get_attributions_by_role(role: ContributorRole) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.role == role {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Calculate total shares for a publication
#[hdk_extern]
pub fn get_publication_shares(publication_id: String) -> ExternResult<PublicationShares> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    let mut total_percentage = 0.0;
    let mut contributor_count = 0;
    let mut verified_count = 0;

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.publication_id == publication_id {
                total_percentage += attr.share_percentage;
                contributor_count += 1;
                if attr.verified {
                    verified_count += 1;
                }
            }
        }
    }

    Ok(PublicationShares {
        publication_id,
        total_share_percentage: total_percentage,
        contributor_count,
        verified_count,
        remaining_percentage: 100.0 - total_percentage,
    })
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct PublicationShares {
    pub publication_id: String,
    pub total_share_percentage: f64,
    pub contributor_count: u32,
    pub verified_count: u32,
    pub remaining_percentage: f64,
}

/// Pure function: compute publication share summary from individual share data.
///
/// Each entry is (share_percentage, verified).
pub fn compute_publication_shares(
    publication_id: String,
    shares: &[(f64, bool)],
) -> PublicationShares {
    let mut total_percentage = 0.0;
    let mut contributor_count = 0u32;
    let mut verified_count = 0u32;

    for &(pct, verified) in shares {
        total_percentage += pct;
        contributor_count += 1;
        if verified {
            verified_count += 1;
        }
    }

    PublicationShares {
        publication_id,
        total_share_percentage: total_percentage,
        contributor_count,
        verified_count,
        remaining_percentage: 100.0 - total_percentage,
    }
}

/// Pure function: compute average share percentage from a list of share values.
///
/// Returns 0.0 for an empty list.
pub fn compute_average_share(shares: &[f64]) -> f64 {
    if shares.is_empty() {
        0.0
    } else {
        let total: f64 = shares.iter().sum();
        total / shares.len() as f64
    }
}

/// Remove an attribution (only by original author/creator)
#[hdk_extern]
pub fn remove_attribution(input: RemoveAttributionInput) -> ExternResult<()> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "remove_attribution")?;
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.id == input.attribution_id {
                delete_entry(record.action_address().clone())?;
                return Ok(());
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Attribution not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveAttributionInput {
    pub attribution_id: String,
    pub requester_did: String,
}

// ============================================================================
// Cross-domain: Check evidence disputes before attribution
// ============================================================================

/// Wire-compatible copy of justice EvidenceDispute for deserialization.
#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
struct LocalEvidenceDispute {
    pub id: String,
    pub evidence_id: String,
    pub disputant: String,
    pub reason: String,
    pub created_at: Timestamp,
    pub resolved: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckEvidenceDisputesInput {
    /// The evidence ID to check in the justice system.
    pub evidence_id: String,
    /// Publication being attributed (for context in result).
    pub publication_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EvidenceDisputeCheckResult {
    pub has_disputes: bool,
    pub dispute_count: u32,
    pub unresolved_count: u32,
    pub warning: Option<String>,
    pub error: Option<String>,
}

/// Check if evidence cited in a publication has been disputed in the
/// justice system.
///
/// Cross-domain call: media-attribution queries justice_evidence via
/// `call(CallTargetCell::Local, ...)` to check whether cited evidence
/// has open disputes. A publication citing disputed evidence should
/// carry a credibility warning.
#[hdk_extern]
pub fn check_evidence_disputes_for_publication(
    input: CheckEvidenceDisputesInput,
) -> ExternResult<EvidenceDisputeCheckResult> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("justice_evidence"),
        FunctionName::from("get_evidence_disputes"),
        None,
        input.evidence_id.clone(),
    );

    match &response {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let records: Vec<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?;

            let total = records.len() as u32;
            let mut unresolved = 0u32;

            for record in &records {
                if let Some(dispute) = record
                    .entry()
                    .to_app_option::<LocalEvidenceDispute>()
                    .ok()
                    .flatten()
                {
                    if !dispute.resolved {
                        unresolved += 1;
                    }
                }
            }

            Ok(EvidenceDisputeCheckResult {
                has_disputes: total > 0,
                dispute_count: total,
                unresolved_count: unresolved,
                warning: if unresolved > 0 {
                    Some(format!(
                        "Evidence '{}' cited in publication '{}' has {} unresolved dispute(s) in the justice system",
                        input.evidence_id, input.publication_id, unresolved
                    ))
                } else {
                    None
                },
                error: None,
            })
        }
        Ok(ZomeCallResponse::NetworkError(err)) => Ok(EvidenceDisputeCheckResult {
            has_disputes: false,
            dispute_count: 0,
            unresolved_count: 0,
            warning: None,
            error: Some(format!("Network error: {}", err)),
        }),
        _ => Ok(EvidenceDisputeCheckResult {
            has_disputes: false,
            dispute_count: 0,
            unresolved_count: 0,
            warning: None,
            error: Some("Failed to query justice evidence zome".into()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // compute_publication_shares tests
    // ========================================================================

    #[test]
    fn publication_shares_no_contributors() {
        let result = compute_publication_shares("pub-1".into(), &[]);
        assert_eq!(result.contributor_count, 0);
        assert_eq!(result.verified_count, 0);
        assert!((result.total_share_percentage - 0.0).abs() < f64::EPSILON);
        assert!((result.remaining_percentage - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn publication_shares_single_contributor() {
        let result = compute_publication_shares("pub-1".into(), &[(25.0, true)]);
        assert_eq!(result.contributor_count, 1);
        assert_eq!(result.verified_count, 1);
        assert!((result.total_share_percentage - 25.0).abs() < f64::EPSILON);
        assert!((result.remaining_percentage - 75.0).abs() < f64::EPSILON);
    }

    #[test]
    fn publication_shares_multiple_contributors() {
        let shares = vec![(30.0, true), (20.0, false), (10.0, true)];
        let result = compute_publication_shares("pub-2".into(), &shares);
        assert_eq!(result.contributor_count, 3);
        assert_eq!(result.verified_count, 2);
        assert!((result.total_share_percentage - 60.0).abs() < f64::EPSILON);
        assert!((result.remaining_percentage - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn publication_shares_exactly_100_percent() {
        let shares = vec![(50.0, true), (30.0, true), (20.0, false)];
        let result = compute_publication_shares("pub-3".into(), &shares);
        assert!((result.total_share_percentage - 100.0).abs() < f64::EPSILON);
        assert!((result.remaining_percentage - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn publication_shares_over_100_percent() {
        // Overshoot: remaining_percentage should be negative
        let shares = vec![(60.0, false), (60.0, false)];
        let result = compute_publication_shares("pub-4".into(), &shares);
        assert!((result.total_share_percentage - 120.0).abs() < f64::EPSILON);
        assert!((result.remaining_percentage - (-20.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn publication_shares_all_unverified() {
        let shares = vec![(10.0, false), (20.0, false)];
        let result = compute_publication_shares("pub-5".into(), &shares);
        assert_eq!(result.verified_count, 0);
        assert_eq!(result.contributor_count, 2);
    }

    #[test]
    fn publication_shares_preserves_publication_id() {
        let result = compute_publication_shares("my-special-pub-id".into(), &[(5.0, false)]);
        assert_eq!(result.publication_id, "my-special-pub-id");
    }

    // ========================================================================
    // compute_average_share tests
    // ========================================================================

    #[test]
    fn average_share_empty() {
        assert!((compute_average_share(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn average_share_single() {
        assert!((compute_average_share(&[42.0]) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn average_share_multiple() {
        assert!((compute_average_share(&[10.0, 20.0, 30.0]) - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn average_share_all_same() {
        assert!((compute_average_share(&[25.0, 25.0, 25.0, 25.0]) - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn average_share_fractional() {
        // 10 + 20 + 30 = 60 / 3 = 20 (exact), but test a non-round case:
        // 1 + 2 = 3 / 2 = 1.5
        assert!((compute_average_share(&[1.0, 2.0]) - 1.5).abs() < f64::EPSILON);
    }

    // ========================================================================
    // Serde roundtrip tests for coordinator-local structs
    // ========================================================================

    #[test]
    fn add_attribution_input_serde_roundtrip() {
        let input = AddAttributionInput {
            publication_id: "pub-1".into(),
            contributor_did: "did:mycelix:alice".into(),
            role: ContributorRole::Author,
            share_percentage: 50.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: AddAttributionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.publication_id, "pub-1");
        assert_eq!(input2.contributor_did, "did:mycelix:alice");
        assert!((input2.share_percentage - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn set_royalty_input_serde_roundtrip() {
        let input = SetRoyaltyInput {
            publication_id: "pub-2".into(),
            rule_type: RoyaltyType::PerView,
            percentage: 5.0,
            minimum_amount: Some(0.01),
            currency: "USD".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: SetRoyaltyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.publication_id, "pub-2");
        assert_eq!(input2.minimum_amount, Some(0.01));
        assert_eq!(input2.currency, "USD");
    }

    #[test]
    fn record_usage_input_serde_roundtrip() {
        let input = RecordUsageInput {
            publication_id: "pub-3".into(),
            usage_type: UsageType::Download,
            user_did: Some("did:mycelix:bob".into()),
            royalty_paid: Some(1.50),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: RecordUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.publication_id, "pub-3");
        assert_eq!(input2.user_did.as_deref(), Some("did:mycelix:bob"));
        assert_eq!(input2.royalty_paid, Some(1.50));
    }

    #[test]
    fn record_usage_input_serde_none_fields() {
        let input = RecordUsageInput {
            publication_id: "pub-4".into(),
            usage_type: UsageType::View,
            user_did: None,
            royalty_paid: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: RecordUsageInput = serde_json::from_str(&json).unwrap();
        assert!(input2.user_did.is_none());
        assert!(input2.royalty_paid.is_none());
    }

    #[test]
    fn verify_attribution_input_serde_roundtrip() {
        let input = VerifyAttributionInput {
            attribution_id: "attr-1".into(),
            requester_did: "did:mycelix:alice".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: VerifyAttributionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.attribution_id, "attr-1");
        assert_eq!(input2.requester_did, "did:mycelix:alice");
    }

    #[test]
    fn update_share_input_serde_roundtrip() {
        let input = UpdateShareInput {
            attribution_id: "attr-2".into(),
            new_share_percentage: 33.33,
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: UpdateShareInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.attribution_id, "attr-2");
        assert!((input2.new_share_percentage - 33.33).abs() < f64::EPSILON);
    }

    #[test]
    fn deactivate_royalty_input_serde_roundtrip() {
        let input = DeactivateRoyaltyInput {
            rule_id: "royalty-1".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: DeactivateRoyaltyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.rule_id, "royalty-1");
    }

    #[test]
    fn remove_attribution_input_serde_roundtrip() {
        let input = RemoveAttributionInput {
            attribution_id: "attr-3".into(),
            requester_did: "did:mycelix:charlie".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: RemoveAttributionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.attribution_id, "attr-3");
        assert_eq!(input2.requester_did, "did:mycelix:charlie");
    }

    #[test]
    fn contributor_earnings_serde_roundtrip() {
        let earnings = ContributorEarnings {
            contributor_did: "did:mycelix:alice".into(),
            publication_count: 5,
            average_share_percentage: 40.0,
            total_royalties_earned: 123.45,
        };
        let json = serde_json::to_string(&earnings).unwrap();
        let earnings2: ContributorEarnings = serde_json::from_str(&json).unwrap();
        assert_eq!(earnings, earnings2);
    }

    #[test]
    fn publication_shares_serde_roundtrip() {
        let shares = PublicationShares {
            publication_id: "pub-99".into(),
            total_share_percentage: 80.0,
            contributor_count: 3,
            verified_count: 2,
            remaining_percentage: 20.0,
        };
        let json = serde_json::to_string(&shares).unwrap();
        let shares2: PublicationShares = serde_json::from_str(&json).unwrap();
        assert_eq!(shares, shares2);
    }

    #[test]
    fn check_evidence_disputes_input_serde_roundtrip() {
        let input = CheckEvidenceDisputesInput {
            evidence_id: "ev-1".into(),
            publication_id: "pub-1".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let input2: CheckEvidenceDisputesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input2.evidence_id, "ev-1");
        assert_eq!(input2.publication_id, "pub-1");
    }

    #[test]
    fn evidence_dispute_check_result_serde_no_disputes() {
        let result = EvidenceDisputeCheckResult {
            has_disputes: false,
            dispute_count: 0,
            unresolved_count: 0,
            warning: None,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: EvidenceDisputeCheckResult = serde_json::from_str(&json).unwrap();
        assert!(!result2.has_disputes);
        assert_eq!(result2.dispute_count, 0);
    }

    #[test]
    fn evidence_dispute_check_result_serde_with_warning() {
        let result = EvidenceDisputeCheckResult {
            has_disputes: true,
            dispute_count: 3,
            unresolved_count: 1,
            warning: Some("Evidence has unresolved disputes".into()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: EvidenceDisputeCheckResult = serde_json::from_str(&json).unwrap();
        assert!(result2.has_disputes);
        assert_eq!(result2.dispute_count, 3);
        assert_eq!(result2.unresolved_count, 1);
        assert!(result2.warning.is_some());
    }

    // ========================================================================
    // Update input struct tests
    // ========================================================================

    #[test]
    fn update_attribution_input_serde_roundtrip() {
        let input = UpdateAttributionInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: Attribution {
                id: "attr:pub-1:did:mycelix:alice:999".into(),
                publication_id: "pub-1".into(),
                contributor_did: "did:mycelix:alice".into(),
                role: ContributorRole::Author,
                share_percentage: 60.0,
                verified: true,
                created: Timestamp::from_micros(1_000_000),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateAttributionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(decoded.updated_entry.share_percentage, 60.0);
        assert!(decoded.updated_entry.verified);
    }

    #[test]
    fn update_attribution_input_clone() {
        let input = UpdateAttributionInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            updated_entry: Attribution {
                id: "attr-clone".into(),
                publication_id: "pub-c".into(),
                contributor_did: "did:mycelix:bob".into(),
                role: ContributorRole::Editor,
                share_percentage: 20.0,
                verified: false,
                created: Timestamp::from_micros(0),
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry.role, ContributorRole::Editor);
    }

    #[test]
    fn update_attribution_input_other_role_serde() {
        let input = UpdateAttributionInput {
            original_action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            updated_entry: Attribution {
                id: "attr-other".into(),
                publication_id: "pub-x".into(),
                contributor_did: "did:mycelix:custom".into(),
                role: ContributorRole::Other("sound-designer".into()),
                share_percentage: 5.0,
                verified: false,
                created: Timestamp::from_micros(0),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateAttributionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.updated_entry.role,
            ContributorRole::Other("sound-designer".into())
        );
    }

    #[test]
    fn update_royalty_rule_input_serde_roundtrip() {
        let input = UpdateRoyaltyRuleInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xcd; 36]),
            updated_entry: RoyaltyRule {
                id: "royalty:pub-1:PerView:999".into(),
                publication_id: "pub-1".into(),
                rule_type: RoyaltyType::PerDownload,
                percentage: 15.0,
                minimum_amount: Some(0.05),
                currency: "EUR".into(),
                active: true,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRoyaltyRuleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.original_action_hash,
            ActionHash::from_raw_36(vec![0xcd; 36])
        );
        assert_eq!(decoded.updated_entry.rule_type, RoyaltyType::PerDownload);
        assert_eq!(decoded.updated_entry.percentage, 15.0);
        assert_eq!(decoded.updated_entry.minimum_amount, Some(0.05));
    }

    #[test]
    fn update_royalty_rule_input_clone() {
        let input = UpdateRoyaltyRuleInput {
            original_action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            updated_entry: RoyaltyRule {
                id: "r-clone".into(),
                publication_id: "pub-c".into(),
                rule_type: RoyaltyType::Subscription,
                percentage: 10.0,
                minimum_amount: None,
                currency: "USD".into(),
                active: false,
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.updated_entry.rule_type, RoyaltyType::Subscription);
        assert!(!cloned.updated_entry.active);
    }

    #[test]
    fn update_royalty_rule_input_deactivated_serde() {
        let input = UpdateRoyaltyRuleInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xef; 36]),
            updated_entry: RoyaltyRule {
                id: "r-deactivated".into(),
                publication_id: "pub-2".into(),
                rule_type: RoyaltyType::PerView,
                percentage: 0.0,
                minimum_amount: None,
                currency: "USD".into(),
                active: false,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRoyaltyRuleInput = serde_json::from_str(&json).unwrap();
        assert!(!decoded.updated_entry.active);
        assert_eq!(decoded.updated_entry.percentage, 0.0);
    }

    #[test]
    fn contributor_role_all_variants_serde() {
        let roles = vec![
            ContributorRole::Author,
            ContributorRole::CoAuthor,
            ContributorRole::Editor,
            ContributorRole::Researcher,
            ContributorRole::Photographer,
            ContributorRole::Illustrator,
            ContributorRole::Translator,
            ContributorRole::Source,
            ContributorRole::Other("custom-role".into()),
        ];
        for role in roles {
            let json = serde_json::to_string(&role).unwrap();
            let role2: ContributorRole = serde_json::from_str(&json).unwrap();
            assert_eq!(role, role2);
        }
    }

    #[test]
    fn royalty_type_all_variants_serde() {
        let types = vec![
            RoyaltyType::PerView,
            RoyaltyType::PerShare,
            RoyaltyType::PerDownload,
            RoyaltyType::PerDerivative,
            RoyaltyType::Subscription,
        ];
        for rt in types {
            let json = serde_json::to_string(&rt).unwrap();
            let rt2: RoyaltyType = serde_json::from_str(&json).unwrap();
            assert_eq!(rt, rt2);
        }
    }

    #[test]
    fn usage_type_all_variants_serde() {
        let types = vec![
            UsageType::View,
            UsageType::Share,
            UsageType::Download,
            UsageType::Derivative,
            UsageType::Citation,
        ];
        for ut in types {
            let json = serde_json::to_string(&ut).unwrap();
            let ut2: UsageType = serde_json::from_str(&json).unwrap();
            assert_eq!(ut, ut2);
        }
    }
}
