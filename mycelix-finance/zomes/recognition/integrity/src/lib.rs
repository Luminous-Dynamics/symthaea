#![deny(unsafe_code)]
//! Recognition Integrity Zome
//!
//! Implements the MYCEL reputation system through weighted recognition events.
//! Each member can recognize up to 10 others per monthly cycle.
//! Recognition weight = recognizer's MYCEL score x base_weight.
//!
//! Constitutional: Recognition weighted by recognizer's MYCEL (prevents Sybil attacks).
//! Constitutional: MYCEL is non-transferable (soulbound).

use hdi::prelude::*;
pub use mycelix_finance_types::ContributionType;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Maximum recognitions per member per monthly cycle
pub const MAX_RECOGNITIONS_PER_CYCLE: u32 = 10;

/// Base weight for recognition (multiplied by recognizer's MYCEL)
pub const RECOGNITION_BASE_WEIGHT: f64 = 1.0;

/// Minimum MYCEL score required to give recognition (apprentices cannot)
pub const MIN_MYCEL_TO_GIVE: f64 = 0.3;

/// MYCEL thresholds for tier progression
pub const MYCEL_APPRENTICE_MAX: f64 = 0.3;
pub const MYCEL_STEWARD_MIN: f64 = 0.7;

/// Jubilee compression factor (applied every 4 years)
pub const JUBILEE_COMPRESSION: f64 = 0.8;

/// Passive decay rate (5% annual)
pub const PASSIVE_DECAY_RATE: f64 = 0.05;

// String length limits — prevent DHT bloat attacks
const MAX_DID_LEN: usize = 256;
const MAX_CYCLE_ID_LEN: usize = 32;

// =============================================================================
// ENTRY TYPES
// =============================================================================

/// A recognition event from one member to another
///
/// Recognition is the mechanism by which MYCEL reputation is earned.
/// Higher-MYCEL recognizers produce more valuable recognition (Sybil-resistant).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RecognitionEvent {
    /// DID of the member giving recognition
    pub recognizer_did: String,

    /// DID of the member receiving recognition
    pub recipient_did: String,

    /// Computed weight: recognizer's MYCEL × base_weight
    pub weight: f64,

    /// Type of contribution being recognized
    pub contribution_type: ContributionType,

    /// Monthly cycle ID (e.g., "2026-02")
    pub cycle_id: String,

    /// Recognizer's MYCEL score at time of recognition
    pub recognizer_mycel: f64,

    /// Timestamp
    pub timestamp: Timestamp,
}

/// A member's MYCEL state — soulbound, non-transferable reputation
///
/// MYCEL score is 0.0-1.0, computed from 4 weighted components:
/// - Participation (40%): tx activity, governance voting, commons engagement
/// - Recognition (20%): weighted recognition events from other members
/// - Validation (20%): quality of work as validator/contributor
/// - Longevity (20%): time active, capped at 24 months
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MemberMycelState {
    /// DID of the member
    pub member_did: String,

    /// Composite MYCEL score (0.0 - 1.0)
    pub mycel_score: f64,

    /// Participation component (0.0 - 1.0)
    pub participation: f64,

    /// Recognition component (0.0 - 1.0)
    pub recognition: f64,

    /// Validation component (0.0 - 1.0)
    pub validation: f64,

    /// Longevity component (0.0 - 1.0)
    pub longevity: f64,

    /// Active months count (for longevity calculation)
    pub active_months: u32,

    /// Whether this member is in apprentice mode
    pub is_apprentice: bool,

    /// DID of the vouching mentor (if apprentice)
    pub mentor_did: Option<String>,

    /// Total recognitions given in current cycle
    pub recognitions_given_this_cycle: u32,

    /// Current cycle ID
    pub current_cycle_id: String,

    /// When the MYCEL state was last updated
    pub last_updated: Timestamp,
}

/// Recognition allocation for tracking per-cycle limits
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RecognitionAllocation {
    /// DID of the recognizer
    pub recognizer_did: String,

    /// Monthly cycle ID
    pub cycle_id: String,

    /// Number of recognitions given in this cycle
    pub count: u32,
}

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// =============================================================================
// ENTRY & LINK TYPE ENUMS
// =============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    RecognitionEvent(RecognitionEvent),
    MemberMycelState(MemberMycelState),
    RecognitionAllocation(RecognitionAllocation),
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from recognizer DID to recognition events they gave
    RecognizerToEvents,
    /// Link from recipient DID to recognition events they received
    RecipientToEvents,
    /// Link from cycle anchor to recognition events in that cycle
    CycleToEvents,
    /// Link from member DID to their MemberMycelState
    MemberToMycelState,
    /// Link from recognizer+cycle to their allocation tracker
    RecognizerCycleToAllocation,
    /// General anchor links
    AnchorLinks,
    /// Link from governance_agents anchor to authorized agent pubkeys
    GovernanceAgents,
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
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::RecognitionEvent(event) => {
                    validate_create_recognition(EntryCreationAction::Create(action), event)
                }
                EntryTypes::MemberMycelState(state) => {
                    validate_create_mycel_state(EntryCreationAction::Create(action), state)
                }
                EntryTypes::RecognitionAllocation(alloc) => {
                    validate_create_allocation(EntryCreationAction::Create(action), alloc)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => {
                match app_entry {
                    EntryTypes::RecognitionEvent(_) => {
                        // Recognition events are immutable once created
                        Ok(ValidateCallbackResult::Invalid(
                            "Recognition events cannot be updated".into(),
                        ))
                    }
                    EntryTypes::MemberMycelState(state) => {
                        validate_update_mycel_state(action, state)
                    }
                    EntryTypes::RecognitionAllocation(alloc) => {
                        validate_update_allocation(action, alloc)
                    }
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Invalid(
                        "Anchors cannot be updated".into(),
                    )),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::RecognizerToEvents => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecipientToEvents => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CycleToEvents => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MemberToMycelState => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecognizerCycleToAllocation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AnchorLinks => Ok(ValidateCallbackResult::Valid),
            LinkTypes::GovernanceAgents => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_recognition(
    _action: EntryCreationAction,
    event: RecognitionEvent,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if event.recognizer_did.len() > MAX_DID_LEN || event.recipient_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if event.cycle_id.len() > MAX_CYCLE_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Cycle ID exceeds maximum length".into(),
        ));
    }

    // Validate recognizer DID
    if !event.recognizer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Recognizer must be a valid DID".into(),
        ));
    }

    // Validate recipient DID
    if !event.recipient_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Recipient must be a valid DID".into(),
        ));
    }

    // Cannot recognize yourself
    if event.recognizer_did == event.recipient_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot recognize yourself".into(),
        ));
    }

    // NaN/Infinity guard — NaN fails all comparisons, bypassing range checks
    if !event.recognizer_mycel.is_finite() || !event.weight.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Recognition weight and recognizer_mycel must be finite numbers".into(),
        ));
    }

    // Recognizer's MYCEL must be at least MIN_MYCEL_TO_GIVE
    if event.recognizer_mycel < MIN_MYCEL_TO_GIVE {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Recognizer MYCEL ({}) is below minimum {} required to give recognition",
            event.recognizer_mycel, MIN_MYCEL_TO_GIVE
        )));
    }

    // Weight must be in (0.0, 1.0] — max MYCEL is 1.0, base_weight is 1.0
    if event.weight <= 0.0 || event.weight > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recognition weight must be > 0.0 and <= 1.0".into(),
        ));
    }

    // Cycle ID format check (YYYY-MM): length 7 with dash at position 4
    if event.cycle_id.len() != 7 || event.cycle_id.as_bytes()[4] != b'-' {
        return Ok(ValidateCallbackResult::Invalid(
            "Cycle ID must be in YYYY-MM format".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_mycel_state(
    _action: EntryCreationAction,
    state: MemberMycelState,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if state.member_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if state.current_cycle_id.len() > MAX_CYCLE_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Cycle ID exceeds maximum length".into(),
        ));
    }
    if let Some(ref mentor) = state.mentor_did {
        if mentor.len() > MAX_DID_LEN {
            return Ok(ValidateCallbackResult::Invalid(
                "Mentor DID exceeds maximum length".into(),
            ));
        }
    }

    // Validate member DID
    if !state.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Member must be a valid DID".into(),
        ));
    }

    // NaN/Infinity guard — NaN passes range comparisons like `< 0.0 || > 1.0`
    if !state.mycel_score.is_finite()
        || !state.participation.is_finite()
        || !state.recognition.is_finite()
        || !state.validation.is_finite()
        || !state.longevity.is_finite()
    {
        return Ok(ValidateCallbackResult::Invalid(
            "MYCEL score and components must be finite numbers".into(),
        ));
    }

    // MYCEL score must be 0.0-1.0
    if state.mycel_score < 0.0 || state.mycel_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "MYCEL score must be between 0.0 and 1.0".into(),
        ));
    }

    // All components must be 0.0-1.0
    for (name, value) in [
        ("participation", state.participation),
        ("recognition", state.recognition),
        ("validation", state.validation),
        ("longevity", state.longevity),
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} component must be between 0.0 and 1.0",
                name
            )));
        }
    }

    // Apprentices must have a mentor
    if state.is_apprentice && state.mentor_did.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Apprentice must have a mentor DID".into(),
        ));
    }

    // Validate mentor DID if present
    if let Some(ref mentor) = state.mentor_did {
        if !mentor.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "Mentor must be a valid DID".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_mycel_state(
    _action: Update,
    state: MemberMycelState,
) -> ExternResult<ValidateCallbackResult> {
    // Validate member DID format (must remain a valid DID)
    if !state.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Member must be a valid DID".into(),
        ));
    }

    // NaN/Infinity guard
    if !state.mycel_score.is_finite()
        || !state.participation.is_finite()
        || !state.recognition.is_finite()
        || !state.validation.is_finite()
        || !state.longevity.is_finite()
    {
        return Ok(ValidateCallbackResult::Invalid(
            "MYCEL score and components must be finite numbers".into(),
        ));
    }

    // MYCEL score must be 0.0-1.0
    if state.mycel_score < 0.0 || state.mycel_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "MYCEL score must be between 0.0 and 1.0".into(),
        ));
    }

    // All components must be 0.0-1.0
    for (name, value) in [
        ("participation", state.participation),
        ("recognition", state.recognition),
        ("validation", state.validation),
        ("longevity", state.longevity),
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} component must be between 0.0 and 1.0",
                name
            )));
        }
    }

    // Note: member_did immutability (cannot change which member this state belongs to)
    // cannot be fully enforced here because integrity validation does not have access
    // to fetch the original entry from the DHT. The coordinator must enforce this.

    // Apprentices must have a mentor
    if state.is_apprentice && state.mentor_did.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Apprentice must have a mentor DID".into(),
        ));
    }

    // Validate mentor DID if present
    if let Some(ref mentor) = state.mentor_did {
        if !mentor.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "Mentor must be a valid DID".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_allocation(
    _action: EntryCreationAction,
    alloc: RecognitionAllocation,
) -> ExternResult<ValidateCallbackResult> {
    // String length checks — prevent DHT bloat
    if alloc.recognizer_did.len() > MAX_DID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "DID exceeds maximum length".into(),
        ));
    }
    if alloc.cycle_id.len() > MAX_CYCLE_ID_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            "Cycle ID exceeds maximum length".into(),
        ));
    }

    if !alloc.recognizer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Recognizer must be a valid DID".into(),
        ));
    }
    if alloc.count > MAX_RECOGNITIONS_PER_CYCLE {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Cannot exceed {} recognitions per cycle",
            MAX_RECOGNITIONS_PER_CYCLE
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_allocation(
    _action: Update,
    alloc: RecognitionAllocation,
) -> ExternResult<ValidateCallbackResult> {
    if alloc.count > MAX_RECOGNITIONS_PER_CYCLE {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Cannot exceed {} recognitions per cycle",
            MAX_RECOGNITIONS_PER_CYCLE
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}
