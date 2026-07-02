// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! CGC (Civic Gifting Credits) Coordinator Zome
//!
//! Implements Commons Charter Article I - Civic Gifting Credits
//!
//! This module provides the business logic for:
//! - Monthly CGC allocation (10 per verified member)
//! - Gift giving (recognition, not transfer)
//! - Recognition tracking (what you've received)
//! - High-activity flagging and audit support
//! - Cultural alias management
//!
//! Key Principle: CGCs are for RECOGNITION, not accumulation.
//! You cannot "spend" recognition you've received - you can only give
//! from your monthly allocation.

use hdk::prelude::*;

// Re-export integrity types for external use
pub use cgc_integrity::*;

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct GiftCgcInput {
    pub receiver_did: String,
    pub amount: u32,
    pub gratitude_message: String,
    pub contribution_type: ContributionType,
    pub cultural_alias: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AllocationStatus {
    pub member_did: String,
    pub cycle_id: String,
    pub total_allocated: u32,
    pub remaining: u32,
    pub gifted: u32,
    pub expires_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecognitionSummary {
    pub member_did: String,
    pub cycle_id: String,
    pub total_received: u32,
    pub unique_givers: u32,
    pub by_contribution_type: Vec<(String, u32)>,
    pub is_flagged: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GiftRecord {
    pub id: String,
    pub giver_did: String,
    pub receiver_did: String,
    pub amount: u32,
    pub gratitude_message: String,
    pub contribution_type: ContributionType,
    pub timestamp: Timestamp,
}

// =============================================================================
// CORE FUNCTIONS
// =============================================================================

/// Get or create allocation for current cycle
///
/// Every verified member receives 10 CGC per month (non-cumulative)
/// This function ensures an allocation exists for the current cycle
#[hdk_extern]
pub fn get_or_create_allocation(member_did: String) -> ExternResult<AllocationStatus> {
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    let cycle_id = get_current_cycle_id()?;

    // Try to find existing allocation
    if let Some(allocation) = find_allocation(&member_did, &cycle_id)? {
        return Ok(allocation_to_status(&allocation));
    }

    // Create new allocation for this cycle
    let now = sys_time()?;
    let cycle_end = get_cycle_end_timestamp(&cycle_id)?;

    let allocation = CgcAllocation {
        member_did: member_did.clone(),
        cycle_id: cycle_id.clone(),
        total_allocated: MONTHLY_CGC_ALLOCATION,
        remaining: MONTHLY_CGC_ALLOCATION,
        gifted: 0,
        created: now,
        expires: cycle_end,
    };

    let action_hash = create_entry(&EntryTypes::CgcAllocation(allocation.clone()))?;

    // Link to member
    create_link(
        anchor_hash(&format!("member:{}", member_did))?,
        action_hash,
        LinkTypes::MemberToAllocations,
        cycle_id.as_bytes().to_vec(),
    )?;

    Ok(allocation_to_status(&allocation))
}

/// Gift CGC to another member
///
/// This is the core action: giving recognition to someone for their contribution.
/// The gift comes from your monthly allocation, reducing your "remaining" balance.
/// The receiver gains RECOGNITION (tracked separately), not spendable credits.
#[hdk_extern]
pub fn gift_cgc(input: GiftCgcInput) -> ExternResult<GiftRecord> {
    if input.receiver_did.is_empty() || input.receiver_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Receiver DID must be 1-256 characters".into()
        )));
    }
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be greater than 0".into()
        )));
    }
    if input.gratitude_message.is_empty() || input.gratitude_message.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Gratitude message must be 1-2048 characters".into()
        )));
    }
    if let Some(ref alias) = input.cultural_alias {
        if alias.is_empty() || alias.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cultural alias must be 1-256 characters".into()
            )));
        }
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let giver_did = format!("did:mycelix:{}", caller);
    let cycle_id = get_current_cycle_id()?;
    let now = sys_time()?;

    // Validate receiver is different from giver
    if giver_did == input.receiver_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot gift CGC to yourself".into()
        )));
    }

    // Get or create giver's allocation
    let allocation = get_or_create_allocation(giver_did.clone())?;

    // Check if giver has enough remaining
    if allocation.remaining < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Insufficient CGC: have {} remaining, tried to gift {}",
            allocation.remaining, input.amount
        ))));
    }

    // Create the gift
    let gift_id = format!(
        "gift:{}:{}:{}",
        giver_did,
        input.receiver_did,
        now.as_micros()
    );
    let gift = CgcGift {
        id: gift_id.clone(),
        giver_did: giver_did.clone(),
        receiver_did: input.receiver_did.clone(),
        amount: input.amount,
        gratitude_message: input.gratitude_message.clone(),
        contribution_type: input.contribution_type.clone(),
        cultural_alias: input.cultural_alias,
        cycle_id: cycle_id.clone(),
        timestamp: now,
    };

    let gift_hash = create_entry(&EntryTypes::CgcGift(gift.clone()))?;

    // Link gift to giver
    create_link(
        anchor_hash(&format!("giver:{}", giver_did))?,
        gift_hash.clone(),
        LinkTypes::GiverToGifts,
        (),
    )?;

    // Link gift to receiver
    create_link(
        anchor_hash(&format!("receiver:{}", input.receiver_did))?,
        gift_hash.clone(),
        LinkTypes::ReceiverToGifts,
        (),
    )?;

    // Link gift to cycle
    create_link(
        anchor_hash(&format!("cycle:{}", cycle_id))?,
        gift_hash,
        LinkTypes::CycleToGifts,
        (),
    )?;

    // Update giver's allocation (decrease remaining)
    update_allocation_after_gift(&giver_did, &cycle_id, input.amount)?;

    // Update receiver's recognition
    update_recognition_after_gift(
        &input.receiver_did,
        &cycle_id,
        input.amount,
        &giver_did,
        &input.contribution_type,
    )?;

    Ok(GiftRecord {
        id: gift_id,
        giver_did,
        receiver_did: input.receiver_did,
        amount: input.amount,
        gratitude_message: input.gratitude_message,
        contribution_type: input.contribution_type,
        timestamp: now,
    })
}

/// Get gifts given by a member in current cycle
#[hdk_extern]
pub fn get_gifts_given(member_did: String) -> ExternResult<Vec<GiftRecord>> {
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("giver:{}", member_did))?,
            LinkTypes::GiverToGifts,
        )?,
        GetStrategy::default(),
    )?;

    let cycle_id = get_current_cycle_id()?;
    let mut gifts = Vec::new();

    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(gift) = record.entry().to_app_option::<CgcGift>().ok().flatten() {
                // Only include current cycle
                if gift.cycle_id == cycle_id {
                    gifts.push(GiftRecord {
                        id: gift.id,
                        giver_did: gift.giver_did,
                        receiver_did: gift.receiver_did,
                        amount: gift.amount,
                        gratitude_message: gift.gratitude_message,
                        contribution_type: gift.contribution_type,
                        timestamp: gift.timestamp,
                    });
                }
            }
        }
    }

    Ok(gifts)
}

/// Get gifts received by a member in current cycle
#[hdk_extern]
pub fn get_gifts_received(member_did: String) -> ExternResult<Vec<GiftRecord>> {
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("receiver:{}", member_did))?,
            LinkTypes::ReceiverToGifts,
        )?,
        GetStrategy::default(),
    )?;

    let cycle_id = get_current_cycle_id()?;
    let mut gifts = Vec::new();

    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(gift) = record.entry().to_app_option::<CgcGift>().ok().flatten() {
                if gift.cycle_id == cycle_id {
                    gifts.push(GiftRecord {
                        id: gift.id,
                        giver_did: gift.giver_did,
                        receiver_did: gift.receiver_did,
                        amount: gift.amount,
                        gratitude_message: gift.gratitude_message,
                        contribution_type: gift.contribution_type,
                        timestamp: gift.timestamp,
                    });
                }
            }
        }
    }

    Ok(gifts)
}

/// Get recognition summary for a member
#[hdk_extern]
pub fn get_recognition_summary(member_did: String) -> ExternResult<RecognitionSummary> {
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    let cycle_id = get_current_cycle_id()?;
    let gifts = get_gifts_received(member_did.clone())?;

    let total_received: u32 = gifts.iter().map(|g| g.amount).sum();
    let unique_givers: u32 = gifts
        .iter()
        .map(|g| g.giver_did.clone())
        .collect::<std::collections::HashSet<_>>()
        .len() as u32;

    // Aggregate by contribution type
    let mut by_type: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for gift in &gifts {
        let type_str = format!("{:?}", gift.contribution_type);
        *by_type.entry(type_str).or_insert(0) += gift.amount;
    }

    // Check if flagged
    let is_flagged = total_received > HIGH_ACTIVITY_THRESHOLD_TOTAL;

    Ok(RecognitionSummary {
        member_did,
        cycle_id,
        total_received,
        unique_givers,
        by_contribution_type: by_type.into_iter().collect(),
        is_flagged,
    })
}

/// Get high-activity report (for Audit Guild)
/// Returns members exceeding thresholds (Article I, Section 2.b)
#[hdk_extern]
pub fn get_high_activity_report(_cycle_id: String) -> ExternResult<Vec<RecognitionSummary>> {
    // In production: query all recognition records for the cycle
    // and filter to those exceeding thresholds

    // For now, return empty (would need index/anchor for all members)
    Ok(vec![])
}

// =============================================================================
// CULTURAL NAMING FUNCTIONS
// =============================================================================

/// Register a cultural alias for a DAO (Article I, Section 5)
#[hdk_extern]
pub fn register_cultural_alias(input: RegisterAliasInput) -> ExternResult<CulturalAlias> {
    if input.alias.is_empty() || input.alias.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Alias must be 1-256 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    if input.meaning.is_empty() || input.meaning.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Meaning must be 1-2048 characters".into()
        )));
    }
    let now = sys_time()?;

    let alias = CulturalAlias {
        alias: input.alias.to_uppercase(),
        dao_did: input.dao_did.clone(),
        meaning: input.meaning,
        registered_at: now,
    };

    let action_hash = create_entry(&EntryTypes::CulturalAlias(alias.clone()))?;

    // Link to DAO
    create_link(
        anchor_hash(&format!("dao:{}", input.dao_did))?,
        action_hash,
        LinkTypes::DaoToAliases,
        (),
    )?;

    Ok(alias)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterAliasInput {
    pub alias: String,
    pub dao_did: String,
    pub meaning: String,
}

/// Get all cultural aliases for a DAO
#[hdk_extern]
pub fn get_dao_aliases(dao_did: String) -> ExternResult<Vec<CulturalAlias>> {
    if dao_did.is_empty() || dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dao:{}", dao_did))?,
            LinkTypes::DaoToAliases,
        )?,
        GetStrategy::default(),
    )?;

    let mut aliases = Vec::new();
    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(alias) = record
                .entry()
                .to_app_option::<CulturalAlias>()
                .ok()
                .flatten()
            {
                aliases.push(alias);
            }
        }
    }

    Ok(aliases)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get current cycle ID (YYYY-MM format)
fn get_current_cycle_id() -> ExternResult<String> {
    let now = sys_time()?;
    let micros = now.as_micros();

    // Convert to date (simplified - in production use proper datetime)
    // Assuming microseconds since Unix epoch
    let secs = micros / 1_000_000;
    let days = secs / 86400;

    // Approximate year and month (simplified calculation)
    let years_since_1970 = days / 365;
    let year = 1970 + years_since_1970;

    let day_of_year = days % 365;
    let month = (day_of_year / 30) + 1;

    Ok(format!("{:04}-{:02}", year, month.min(12)))
}

/// Get timestamp for end of cycle
fn get_cycle_end_timestamp(cycle_id: &str) -> ExternResult<Timestamp> {
    let parts: Vec<&str> = cycle_id.split('-').collect();
    if parts.len() != 2 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid cycle ID".into()
        )));
    }

    let year: i64 = parts[0]
        .parse()
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid year".into())))?;
    let month: i64 = parts[1]
        .parse()
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid month".into())))?;

    // Calculate end of month (simplified)
    let next_month = if month == 12 { 1 } else { month + 1 };
    let next_year = if month == 12 { year + 1 } else { year };

    // Days since epoch to start of next month (simplified)
    let days = (next_year - 1970) * 365 + (next_month - 1) * 30;
    let micros = days * 86400 * 1_000_000;

    Ok(Timestamp::from_micros(micros))
}

/// Find allocation for a member in a cycle
fn find_allocation(member_did: &str, cycle_id: &str) -> ExternResult<Option<CgcAllocation>> {
    let mut query = LinkQuery::try_new(
        anchor_hash(&format!("member:{}", member_did))?,
        LinkTypes::MemberToAllocations,
    )?;
    query.tag_prefix = Some(LinkTag::new(cycle_id.as_bytes().to_vec()));
    let links = get_links(query, GetStrategy::default())?;

    for link in links {
        if let Some(record) = get(
            link.target.into_action_hash().ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(allocation) = record
                .entry()
                .to_app_option::<CgcAllocation>()
                .ok()
                .flatten()
            {
                if allocation.cycle_id == cycle_id {
                    return Ok(Some(allocation));
                }
            }
        }
    }

    Ok(None)
}

/// Update allocation after a gift is made
fn update_allocation_after_gift(member_did: &str, cycle_id: &str, amount: u32) -> ExternResult<()> {
    // Find and update the allocation
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("member:{}", member_did))?,
            LinkTypes::MemberToAllocations,
        )?,
        GetStrategy::default(),
    )?;

    for link in links {
        let action_hash = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;
        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            if let Some(mut allocation) = record
                .entry()
                .to_app_option::<CgcAllocation>()
                .ok()
                .flatten()
            {
                if allocation.cycle_id == cycle_id {
                    allocation.remaining = allocation.remaining.saturating_sub(amount);
                    allocation.gifted += amount;

                    update_entry(action_hash, &allocation)?;
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}

/// Update recognition after receiving a gift
fn update_recognition_after_gift(
    receiver_did: &str,
    cycle_id: &str,
    amount: u32,
    _giver_did: &str,
    contribution_type: &ContributionType,
) -> ExternResult<()> {
    // Find existing recognition or create new
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("recognition:{}", receiver_did))?,
            LinkTypes::MemberToRecognition,
        )?,
        GetStrategy::default(),
    )?;

    for link in links {
        let action_hash = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;
        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            if let Some(mut recognition) = record
                .entry()
                .to_app_option::<CgcRecognition>()
                .ok()
                .flatten()
            {
                if recognition.cycle_id == cycle_id {
                    recognition.total_received += amount;
                    recognition.unique_givers += 1; // Simplified

                    // Check for high-activity flag
                    if recognition.total_received > HIGH_ACTIVITY_THRESHOLD_TOTAL {
                        recognition.flagged_high_activity = true;
                    }

                    update_entry(action_hash, &recognition)?;
                    return Ok(());
                }
            }
        }
    }

    // Create new recognition record
    let recognition = CgcRecognition {
        member_did: receiver_did.to_string(),
        cycle_id: cycle_id.to_string(),
        total_received: amount,
        unique_givers: 1,
        max_from_single_giver: amount,
        flagged_high_activity: amount > HIGH_ACTIVITY_THRESHOLD_TOTAL,
        by_contribution_type: vec![(contribution_type.clone(), amount)],
    };

    let action_hash = create_entry(&EntryTypes::CgcRecognition(recognition))?;

    create_link(
        anchor_hash(&format!("recognition:{}", receiver_did))?,
        action_hash,
        LinkTypes::MemberToRecognition,
        cycle_id.as_bytes().to_vec(),
    )?;

    // Create flag if needed
    if amount > HIGH_ACTIVITY_THRESHOLD_TOTAL {
        create_high_activity_flag(receiver_did, cycle_id, amount, None)?;
    }

    Ok(())
}

/// Create a high-activity flag
fn create_high_activity_flag(
    member_did: &str,
    cycle_id: &str,
    total_received: u32,
    max_single_source: Option<u32>,
) -> ExternResult<()> {
    let now = sys_time()?;

    let flag_reason = if let Some(max) = max_single_source {
        if max > HIGH_ACTIVITY_THRESHOLD_SINGLE {
            FlagReason::ExceededSingleSourceThreshold
        } else {
            FlagReason::ExceededMonthlyThreshold
        }
    } else {
        FlagReason::ExceededMonthlyThreshold
    };

    let flag = CgcHighActivityFlag {
        member_did: member_did.to_string(),
        cycle_id: cycle_id.to_string(),
        flag_reason,
        total_received,
        max_single_source,
        flagged_at: now,
        audit_status: AuditStatus::PendingReview,
        appeal: None,
    };

    let action_hash = create_entry(&EntryTypes::CgcHighActivityFlag(flag))?;

    create_link(
        anchor_hash(&format!("flags:{}", member_did))?,
        action_hash,
        LinkTypes::MemberToFlags,
        (),
    )?;

    Ok(())
}

/// Create anchor hash using Anchor entry type
fn anchor_hash(anchor: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(anchor.to_string()))
}

fn allocation_to_status(allocation: &CgcAllocation) -> AllocationStatus {
    AllocationStatus {
        member_did: allocation.member_did.clone(),
        cycle_id: allocation.cycle_id.clone(),
        total_allocated: allocation.total_allocated,
        remaining: allocation.remaining,
        gifted: allocation.gifted,
        expires_at: allocation.expires,
    }
}

// =============================================================================
// EXPORTS FOR OTHER ZOMES
// =============================================================================

/// Get CGC recognition for reputation calculation (max 10% weight per Article I.4)
#[hdk_extern]
pub fn get_cgc_reputation_input(member_did: String) -> ExternResult<f32> {
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    let summary = get_recognition_summary(member_did)?;

    // Normalize to 0-1 scale (assuming 100 CGC is "excellent" recognition)
    let normalized = (summary.total_received as f32 / 100.0).min(1.0);

    // Apply max weight of 10%
    Ok(normalized * 0.10)
}
