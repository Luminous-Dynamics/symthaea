// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! CGC (Civic Gifting Credits) Integrity Zome
//!
//! Implements Commons Charter Article I - Civic Gifting Credits
//!
//! Core Principles:
//! - Each verified member receives 10 CGC per month
//! - Non-cumulative: expire at end of each monthly cycle
//! - Non-transferable: can only GIFT, not transfer balance
//! - Transparency: high-activity flagged for audit review
//! - Cultural naming: DAOs may use aliases (SPARK, EMBER, etc.)
//!
//! Constitutional Reference: Commons Charter v1.0, Article I

use hdi::prelude::*;

// =============================================================================
// CONSTANTS (Per Commons Charter Article I)
// =============================================================================

/// Monthly CGC allocation per member (Article I, Section 1)
pub const MONTHLY_CGC_ALLOCATION: u32 = 10;

/// Threshold for high-activity flagging (Article I, Section 2.b)
pub const HIGH_ACTIVITY_THRESHOLD_TOTAL: u32 = 100; // >100 received/month
pub const HIGH_ACTIVITY_THRESHOLD_SINGLE: u32 = 50; // >50 from single source

/// Maximum CGC reputation weight (Article I, Section 4)
pub const MAX_REPUTATION_WEIGHT: f32 = 0.10; // 10% max

// =============================================================================
// ENTRY TYPES
// =============================================================================

/// A CGC Gift - the core unit of gratitude recognition
///
/// Unlike transfers, a Gift represents recognition flowing FROM the giver's
/// monthly allocation TO another member. The giver's balance decreases,
/// but the receiver doesn't gain "spendable" credits - they gain RECOGNITION.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CgcGift {
    /// Unique identifier for this gift
    pub id: String,

    /// DID of the member giving recognition
    pub giver_did: String,

    /// DID of the member being recognized
    pub receiver_did: String,

    /// Amount of CGC gifted (from giver's monthly allocation)
    pub amount: u32,

    /// Reason for the gift (gratitude message)
    pub gratitude_message: String,

    /// Category of contribution being recognized
    pub contribution_type: ContributionType,

    /// Cultural name used (if any) - e.g., "SPARK", "EMBER"
    pub cultural_alias: Option<String>,

    /// The cycle (month) in which this gift was made
    pub cycle_id: String, // Format: "YYYY-MM"

    /// When the gift was given
    pub timestamp: Timestamp,
}

/// Categories of contribution that can be recognized with CGC
/// (Non-exhaustive - communities can define additional types)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContributionType {
    /// Helped with education, mentoring, knowledge sharing
    Education,
    /// Organized community events or coordination
    CommunityOrganizing,
    /// Provided care, support, emotional labor
    CareWork,
    /// Created art, music, culture
    CreativeWork,
    /// Technical contributions (code, infrastructure)
    TechnicalWork,
    /// Governance participation (beyond voting)
    GovernanceLabor,
    /// Environmental stewardship
    Stewardship,
    /// General gratitude (catch-all)
    Gratitude,
    /// Custom category defined by local DAO
    Custom(String),
}

/// Monthly allocation record - tracks a member's CGC for a cycle
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CgcAllocation {
    /// DID of the member
    pub member_did: String,

    /// The cycle (month) - Format: "YYYY-MM"
    pub cycle_id: String,

    /// Total allocated for this cycle (always 10 per Commons Charter)
    pub total_allocated: u32,

    /// Amount remaining to gift
    pub remaining: u32,

    /// Amount gifted this cycle
    pub gifted: u32,

    /// When the allocation was created
    pub created: Timestamp,

    /// When the allocation expires (end of cycle)
    pub expires: Timestamp,
}

/// Recognition received - aggregates gifts received by a member
/// This is what builds the member's "recognition score" (not spendable)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CgcRecognition {
    /// DID of the member who received recognition
    pub member_did: String,

    /// The cycle (month) - Format: "YYYY-MM"
    pub cycle_id: String,

    /// Total CGC received this cycle
    pub total_received: u32,

    /// Number of unique givers
    pub unique_givers: u32,

    /// Highest amount from a single giver (for flagging)
    pub max_from_single_giver: u32,

    /// Whether this exceeds high-activity thresholds
    pub flagged_high_activity: bool,

    /// Breakdown by contribution type
    pub by_contribution_type: Vec<(ContributionType, u32)>,
}

/// High-activity flag for audit review (Article I, Section 2.b)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CgcHighActivityFlag {
    /// DID of the flagged member
    pub member_did: String,

    /// The cycle being flagged
    pub cycle_id: String,

    /// Reason for flagging
    pub flag_reason: FlagReason,

    /// Total received that triggered the flag
    pub total_received: u32,

    /// Highest single-source amount (if applicable)
    pub max_single_source: Option<u32>,

    /// Timestamp of flag creation
    pub flagged_at: Timestamp,

    /// Status of audit review
    pub audit_status: AuditStatus,

    /// Appeal information (if filed)
    pub appeal: Option<Appeal>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FlagReason {
    /// Received >100 CGC in a month
    ExceededMonthlyThreshold,
    /// Received >50 CGC from a single source
    ExceededSingleSourceThreshold,
    /// Detected circular gifting pattern
    CircularGiftingPattern,
    /// Other suspicious pattern
    SuspiciousPattern(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AuditStatus {
    /// Pending quarterly audit review
    PendingReview,
    /// Under active investigation
    UnderInvestigation,
    /// Cleared - legitimate high-value contributor
    Cleared,
    /// Confirmed gaming - penalties applied
    ConfirmedGaming,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Appeal {
    /// Explanation from the member
    pub explanation: String,
    /// When the appeal was filed
    pub filed_at: Timestamp,
    /// Status of the appeal
    pub status: AppealStatus,
    /// Knowledge Council ruling (if decided)
    pub ruling: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AppealStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
}

/// Cultural alias registration (Article I, Section 5)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CulturalAlias {
    /// The alias name (e.g., "SPARK", "EMBER")
    pub alias: String,

    /// The DAO that registered this alias
    pub dao_did: String,

    /// Description of the cultural meaning
    pub meaning: String,

    /// When registered
    pub registered_at: Timestamp,
}

// =============================================================================
// ENTRY & LINK TYPE ENUMS
// =============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CgcGift(CgcGift),
    CgcAllocation(CgcAllocation),
    CgcRecognition(CgcRecognition),
    CgcHighActivityFlag(CgcHighActivityFlag),
    CulturalAlias(CulturalAlias),
    Anchor(Anchor),
}

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from giver DID to their gifts
    GiverToGifts,
    /// Link from receiver DID to gifts received
    ReceiverToGifts,
    /// Link from member DID to their allocations
    MemberToAllocations,
    /// Link from member DID to their recognition records
    MemberToRecognition,
    /// Link from member DID to any flags
    MemberToFlags,
    /// Link from cycle anchor to all gifts in that cycle
    CycleToGifts,
    /// Link from DAO to cultural aliases
    DaoToAliases,
    /// Link type for anchor/path infrastructure
    AnchorLinks,
}

// =============================================================================
// VALIDATION
// =============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                match app_entry {
                    EntryTypes::CgcGift(gift) => {
                        validate_create_gift(EntryCreationAction::Create(action), gift)
                    }
                    EntryTypes::CgcAllocation(allocation) => {
                        validate_create_allocation(EntryCreationAction::Create(action), allocation)
                    }
                    EntryTypes::CgcRecognition(recognition) => validate_create_recognition(
                        EntryCreationAction::Create(action),
                        recognition,
                    ),
                    EntryTypes::CgcHighActivityFlag(flag) => {
                        validate_create_flag(EntryCreationAction::Create(action), flag)
                    }
                    EntryTypes::CulturalAlias(alias) => {
                        validate_create_alias(EntryCreationAction::Create(action), alias)
                    }
                    // Anchors are always valid (just hash placeholders)
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => {
                match app_entry {
                    EntryTypes::CgcGift(_) => {
                        // Gifts cannot be updated (immutable gratitude)
                        Ok(ValidateCallbackResult::Invalid(
                            "CGC Gifts are immutable once given".into(),
                        ))
                    }
                    EntryTypes::CgcAllocation(allocation) => {
                        validate_update_allocation(action, allocation)
                    }
                    EntryTypes::CgcRecognition(recognition) => {
                        validate_update_recognition(action, recognition)
                    }
                    EntryTypes::CgcHighActivityFlag(flag) => validate_update_flag(action, flag),
                    EntryTypes::CulturalAlias(_) => Ok(ValidateCallbackResult::Invalid(
                        "Cultural aliases cannot be updated, only deprecated".into(),
                    )),
                    // Anchors cannot be updated
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                        "Anchors cannot be updated".into(),
                    )),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::GiverToGifts => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ReceiverToGifts => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MemberToAllocations => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MemberToRecognition => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MemberToFlags => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CycleToGifts => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DaoToAliases => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AnchorLinks => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_gift(
    _action: EntryCreationAction,
    gift: CgcGift,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DIDs
    if !gift.giver_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Giver must be a valid DID".into(),
        ));
    }
    if !gift.receiver_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Receiver must be a valid DID".into(),
        ));
    }

    // Cannot gift to yourself
    if gift.giver_did == gift.receiver_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot gift CGC to yourself".into(),
        ));
    }

    // Amount must be positive and within allocation
    if gift.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Gift amount must be positive".into(),
        ));
    }
    if gift.amount > MONTHLY_CGC_ALLOCATION {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Gift amount exceeds monthly allocation of {}",
            MONTHLY_CGC_ALLOCATION
        )));
    }

    // Gratitude message required and reasonable length
    if gift.gratitude_message.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Gratitude message is required".into(),
        ));
    }
    if gift.gratitude_message.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Gratitude message too long (max 1000 chars)".into(),
        ));
    }

    // Cycle ID must be valid format
    if !is_valid_cycle_id(&gift.cycle_id) {
        return Ok(ValidateCallbackResult::Invalid(
            "Cycle ID must be YYYY-MM format".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_allocation(
    _action: EntryCreationAction,
    allocation: CgcAllocation,
) -> ExternResult<ValidateCallbackResult> {
    if !allocation.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Member must be a valid DID".into(),
        ));
    }

    // Total allocated must match constitutional amount
    if allocation.total_allocated != MONTHLY_CGC_ALLOCATION {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Monthly allocation must be {} per Commons Charter",
            MONTHLY_CGC_ALLOCATION
        )));
    }

    // Remaining cannot exceed total
    if allocation.remaining > allocation.total_allocated {
        return Ok(ValidateCallbackResult::Invalid(
            "Remaining cannot exceed total allocated".into(),
        ));
    }

    // Gifted + remaining must equal total
    if allocation.gifted + allocation.remaining != allocation.total_allocated {
        return Ok(ValidateCallbackResult::Invalid(
            "Gifted + remaining must equal total allocated".into(),
        ));
    }

    if !is_valid_cycle_id(&allocation.cycle_id) {
        return Ok(ValidateCallbackResult::Invalid(
            "Cycle ID must be YYYY-MM format".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_allocation(
    _action: Update,
    allocation: CgcAllocation,
) -> ExternResult<ValidateCallbackResult> {
    // Can only decrease remaining (as gifts are made)
    if allocation.gifted + allocation.remaining != allocation.total_allocated {
        return Ok(ValidateCallbackResult::Invalid(
            "Gifted + remaining must equal total allocated".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_recognition(
    _action: EntryCreationAction,
    recognition: CgcRecognition,
) -> ExternResult<ValidateCallbackResult> {
    if !recognition.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Member must be a valid DID".into(),
        ));
    }
    if !is_valid_cycle_id(&recognition.cycle_id) {
        return Ok(ValidateCallbackResult::Invalid(
            "Cycle ID must be YYYY-MM format".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_recognition(
    _action: Update,
    _recognition: CgcRecognition,
) -> ExternResult<ValidateCallbackResult> {
    // Recognition can be updated as more gifts are received
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_flag(
    _action: EntryCreationAction,
    flag: CgcHighActivityFlag,
) -> ExternResult<ValidateCallbackResult> {
    if !flag.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Member must be a valid DID".into(),
        ));
    }

    // Verify flag reason matches thresholds
    match &flag.flag_reason {
        FlagReason::ExceededMonthlyThreshold => {
            if flag.total_received <= HIGH_ACTIVITY_THRESHOLD_TOTAL {
                return Ok(ValidateCallbackResult::Invalid(format!(
                    "Monthly threshold flag requires >{}",
                    HIGH_ACTIVITY_THRESHOLD_TOTAL
                )));
            }
        }
        FlagReason::ExceededSingleSourceThreshold => {
            if let Some(max) = flag.max_single_source {
                if max <= HIGH_ACTIVITY_THRESHOLD_SINGLE {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "Single source threshold flag requires >{}",
                        HIGH_ACTIVITY_THRESHOLD_SINGLE
                    )));
                }
            } else {
                return Ok(ValidateCallbackResult::Invalid(
                    "Single source flag requires max_single_source".into(),
                ));
            }
        }
        _ => {} // Other reasons validated differently
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_flag(
    _action: Update,
    _flag: CgcHighActivityFlag,
) -> ExternResult<ValidateCallbackResult> {
    // Flags can be updated by Audit Guild
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_alias(
    _action: EntryCreationAction,
    alias: CulturalAlias,
) -> ExternResult<ValidateCallbackResult> {
    // Alias must be alphanumeric and reasonable length
    if alias.alias.is_empty() || alias.alias.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Alias must be 1-20 characters".into(),
        ));
    }
    if !alias.alias.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Ok(ValidateCallbackResult::Invalid(
            "Alias must be alphanumeric".into(),
        ));
    }
    if !alias.dao_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DAO must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn is_valid_cycle_id(cycle_id: &str) -> bool {
    // Must be YYYY-MM format
    if cycle_id.len() != 7 {
        return false;
    }
    let parts: Vec<&str> = cycle_id.split('-').collect();
    if parts.len() != 2 {
        return false;
    }

    // Validate year
    if let Ok(year) = parts[0].parse::<u32>() {
        if year < 2020 || year > 2100 {
            return false;
        }
    } else {
        return false;
    }

    // Validate month
    if let Ok(month) = parts[1].parse::<u32>() {
        if month < 1 || month > 12 {
            return false;
        }
    } else {
        return false;
    }

    true
}
