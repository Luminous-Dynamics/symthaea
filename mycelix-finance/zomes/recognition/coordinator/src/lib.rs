#![deny(unsafe_code)]
//! Recognition Coordinator Zome
//!
//! Implements MYCEL reputation through weighted recognition events.
//! Each member can recognize up to 10 others per monthly cycle.
//! Recognition weight = recognizer's MYCEL score x base_weight.
//!
//! Key functions:
//! - recognize_member: Give recognition to another member
//! - get_mycel_score: Get a member's current MYCEL state
//! - get_recognition_received: Get all recognitions received by a member
//! - initialize_member: Set up new member MYCEL state (apprentice or full)
//! - jubilee_normalize: Apply 4-year jubilee compression

use hdk::prelude::*;
use mycelix_finance_shared::{
    anchor_hash, follow_update_chain, verify_caller_is_did,
    verify_governance_or_bootstrap_from_links, GOVERNANCE_AGENTS_ANCHOR,
};

// Re-export integrity types for external use
pub use recognition_integrity::*;

/// DID method prefix for Mycelix identities
const DID_METHOD_PREFIX: &str = "did:mycelix:";
/// Maximum length for a DID string
const MAX_DID_LENGTH: usize = 256;

fn verify_governance_or_bootstrap() -> ExternResult<()> {
    let gov_links = get_links(
        LinkQuery::try_new(
            anchor_hash(GOVERNANCE_AGENTS_ANCHOR)?,
            LinkTypes::GovernanceAgents,
        )?,
        GetStrategy::default(),
    )?;
    verify_governance_or_bootstrap_from_links(gov_links)
}

/// Register a governance agent. Only existing governance agents can register
/// new ones (or anyone during bootstrap when no agents exist yet).
#[hdk_extern]
pub fn register_governance_agent(agent: AgentPubKey) -> ExternResult<ActionHash> {
    verify_governance_or_bootstrap()?;
    create_link(
        anchor_hash(GOVERNANCE_AGENTS_ANCHOR)?,
        agent,
        LinkTypes::GovernanceAgents,
        (),
    )
}

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RecognizeMemberInput {
    pub recipient_did: String,
    pub contribution_type: ContributionType,
    pub cycle_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitializeMemberInput {
    pub member_did: String,
    pub is_apprentice: bool,
    pub mentor_did: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMycelInput {
    pub member_did: String,
    pub participation: f64,
    pub recognition: f64,
    /// If None, validation is auto-fetched from TEND quality ratings via cross-zome call.
    /// If Some, uses the provided value (useful for testing or offline computation).
    pub validation_override: Option<f64>,
    pub active_months: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetRecognitionsInput {
    pub member_did: String,
    pub cycle_id: Option<String>,
    /// Maximum number of recognition events to return (default 100)
    pub limit: Option<usize>,
}

// =============================================================================
// CORE RECOGNITION FUNCTIONS
// =============================================================================

/// Recognize a member's contribution
///
/// Creates a RecognitionEvent weighted by the caller's MYCEL score.
/// Enforces per-cycle limits (max 10 per member per cycle).
/// Apprentices (MYCEL < 0.3) cannot give recognition.
#[hdk_extern]
pub fn recognize_member(input: RecognizeMemberInput) -> ExternResult<Record> {
    if input.recipient_did.is_empty() || input.recipient_did.len() > MAX_DID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Recipient DID must be 1-256 characters".into()
        )));
    }
    if input.cycle_id.len() != 7 || !input.cycle_id.contains('-') {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cycle ID must be in YYYY-MM format (e.g., 2026-02)".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("{}{}", DID_METHOD_PREFIX, caller);

    // Cannot recognize yourself
    if caller_did == input.recipient_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot recognize yourself".into()
        )));
    }

    // Get caller's MYCEL state
    let caller_mycel = get_mycel_state(&caller_did)?;

    // Enforce minimum MYCEL to give recognition
    if caller_mycel.mycel_score < MIN_MYCEL_TO_GIVE {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Your MYCEL score ({:.2}) is below the minimum ({}) required to give recognition",
            caller_mycel.mycel_score, MIN_MYCEL_TO_GIVE
        ))));
    }

    // Check per-cycle limit
    let allocation = get_or_create_allocation(&caller_did, &input.cycle_id)?;
    if allocation.count >= MAX_RECOGNITIONS_PER_CYCLE {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "You have already given {} recognitions this cycle (max {})",
            allocation.count, MAX_RECOGNITIONS_PER_CYCLE
        ))));
    }

    let now = sys_time()?;
    let weight = caller_mycel.mycel_score * RECOGNITION_BASE_WEIGHT;

    let event = RecognitionEvent {
        recognizer_did: caller_did.clone(),
        recipient_did: input.recipient_did.clone(),
        weight,
        contribution_type: input.contribution_type,
        cycle_id: input.cycle_id.clone(),
        recognizer_mycel: caller_mycel.mycel_score,
        timestamp: now,
    };

    let event_hash = create_entry(&EntryTypes::RecognitionEvent(event))?;

    // Link from recognizer to event
    create_link(
        anchor_hash(&format!("recognizer:{}", caller_did))?,
        event_hash.clone(),
        LinkTypes::RecognizerToEvents,
        (),
    )?;

    // Link from recipient to event
    create_link(
        anchor_hash(&format!("recipient:{}", input.recipient_did))?,
        event_hash.clone(),
        LinkTypes::RecipientToEvents,
        (),
    )?;

    // Link from cycle to event
    create_link(
        anchor_hash(&format!("cycle:{}", input.cycle_id))?,
        event_hash.clone(),
        LinkTypes::CycleToEvents,
        (),
    )?;

    // Update the allocation counter
    increment_allocation(&caller_did, &input.cycle_id)?;

    // Return the created record
    let record = get(event_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Failed to retrieve created recognition record".into())
    ))?;

    Ok(record)
}

/// Get all recognitions received by a member, optionally filtered by cycle (paginated, default limit 100)
///
// NOTE: RecognitionEvents are immutable and could use links_to_records, but this
// function applies cycle_id filtering with early-exit on limit during iteration.
// Batch-fetching all records then filtering would be wasteful when cycle_id is set,
// so sequential get() with inline filtering is kept intentionally.
#[hdk_extern]
pub fn get_recognition_received(
    input: GetRecognitionsInput,
) -> ExternResult<Vec<RecognitionEvent>> {
    if input.member_did.is_empty() || input.member_did.len() > MAX_DID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    let limit = input.limit.unwrap_or(100);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("recipient:{}", input.member_did))?,
            LinkTypes::RecipientToEvents,
        )?,
        GetStrategy::default(),
    )?;

    let mut events = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            // RecognitionEvents are immutable (never updated), so plain get() is correct here
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(event) =
                    record
                        .entry()
                        .to_app_option::<RecognitionEvent>()
                        .map_err(|e| {
                            wasm_error!(WasmErrorInner::Guest(format!(
                                "RecognitionEvent deserialization error: {:?}",
                                e
                            )))
                        })?
                {
                    // Filter by cycle if specified
                    if let Some(ref cycle) = input.cycle_id {
                        if event.cycle_id == *cycle {
                            events.push(event);
                        }
                    } else {
                        events.push(event);
                    }
                    if events.len() >= limit {
                        break;
                    }
                }
            }
        }
    }

    // Sort by timestamp (newest first)
    events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(events)
}

// =============================================================================
// MYCEL STATE FUNCTIONS
// =============================================================================

/// Get a member's current MYCEL score
#[hdk_extern]
pub fn get_mycel_score(member_did: String) -> ExternResult<MemberMycelState> {
    if member_did.is_empty() || member_did.len() > MAX_DID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    get_mycel_state(&member_did)
}

/// Initialize a new member's MYCEL state
///
/// Creates the initial MemberMycelState for a new community member.
/// Apprentices start at MYCEL 0.1, full members start at 0.3.
#[hdk_extern]
pub fn initialize_member(input: InitializeMemberInput) -> ExternResult<Record> {
    // Verify caller is the member being initialized (prevents spoofing)
    verify_caller_is_did(&input.member_did)?;

    if input.is_apprentice {
        let mentor = input
            .mentor_did
            .as_ref()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Apprentices must have a mentor DID".into()
            )))?;
        if mentor.is_empty() || mentor.len() > MAX_DID_LENGTH {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Mentor DID must be 1-256 characters".into()
            )));
        }
    }

    // Check if member already has a MYCEL state
    let existing = find_mycel_state(&input.member_did)?;
    if existing.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member already has a MYCEL state initialized".into()
        )));
    }

    let now = sys_time()?;
    let initial_score = if input.is_apprentice { 0.1 } else { 0.3 };

    let state = MemberMycelState {
        member_did: input.member_did.clone(),
        mycel_score: initial_score,
        participation: initial_score,
        recognition: 0.0,
        validation: 0.0,
        longevity: 0.0,
        active_months: 0,
        is_apprentice: input.is_apprentice,
        mentor_did: input.mentor_did,
        recognitions_given_this_cycle: 0,
        current_cycle_id: String::new(),
        last_updated: now,
    };

    let state_hash = create_entry(&EntryTypes::MemberMycelState(state))?;

    // Link from member to their MYCEL state
    create_link(
        anchor_hash(&format!("mycel:{}", input.member_did))?,
        state_hash.clone(),
        LinkTypes::MemberToMycelState,
        (),
    )?;

    let record = get(state_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Failed to retrieve created MYCEL state".into())
    ))?;

    Ok(record)
}

/// Update a member's MYCEL score components
///
/// Recalculates the composite MYCEL score from the 4 components:
/// Participation (40%), Recognition (20%), Validation (20%), Longevity (20%)
///
/// The Validation component is auto-fetched from TEND quality ratings via
/// cross-zome call unless `validation_override` is provided.
#[hdk_extern]
pub fn update_mycel_score(input: UpdateMycelInput) -> ExternResult<MemberMycelState> {
    // Verify caller is the member (prevents score manipulation by others)
    verify_caller_is_did(&input.member_did)?;

    let (current_state, action_hash) =
        find_mycel_state_with_hash(&input.member_did)?.ok_or(wasm_error!(
            WasmErrorInner::Guest("Member MYCEL state not found — initialize first".into())
        ))?;

    let now = sys_time()?;
    let participation = input.participation.clamp(0.0, 1.0);
    let recognition = input.recognition.clamp(0.0, 1.0);

    // Fetch validation from TEND quality ratings, or use override
    let validation = if let Some(v) = input.validation_override {
        v.clamp(0.0, 1.0)
    } else {
        fetch_validation_from_tend(&input.member_did)?
    };

    let longevity = (input.active_months as f64 / 24.0).min(1.0);

    let composite =
        participation * 0.40 + recognition * 0.20 + validation * 0.20 + longevity * 0.20;

    // Check if member should graduate from apprentice
    let is_apprentice = if current_state.is_apprentice && composite >= MYCEL_APPRENTICE_MAX {
        false // Graduate!
    } else {
        current_state.is_apprentice
    };

    let updated_state = MemberMycelState {
        mycel_score: composite.clamp(0.0, 1.0),
        participation,
        recognition,
        validation,
        longevity,
        active_months: input.active_months,
        is_apprentice,
        last_updated: now,
        ..current_state
    };

    update_entry(action_hash, &updated_state)?;

    Ok(updated_state)
}

/// Apply jubilee normalization to a member's MYCEL score
///
/// `new_mycel = 0.3 + (current - 0.3) * 0.8`
/// Compresses toward mean without resetting. Active members recover quickly.
///
/// Restricted to authorized governance agents (or any agent during bootstrap).
#[hdk_extern]
pub fn jubilee_normalize(member_did: String) -> ExternResult<MemberMycelState> {
    verify_governance_or_bootstrap()?;
    if member_did.is_empty() || member_did.len() > MAX_DID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    let (current_state, action_hash) = find_mycel_state_with_hash(&member_did)?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Member MYCEL state not found".into())),
    )?;

    let now = sys_time()?;
    let normalized_score =
        (0.3 + (current_state.mycel_score - 0.3) * JUBILEE_COMPRESSION).clamp(0.0, 1.0);

    let updated_state = MemberMycelState {
        mycel_score: normalized_score,
        last_updated: now,
        ..current_state
    };

    update_entry(action_hash, &updated_state)?;

    Ok(updated_state)
}

/// Dissolve a member's MYCEL state (on exit/death)
///
/// MYCEL dissolves immediately. Contribution history is preserved via
/// immutable RecognitionEvent entries in the DHT.
///
/// Restricted to authorized governance agents (or any agent during bootstrap).
#[hdk_extern]
pub fn dissolve_mycel(member_did: String) -> ExternResult<()> {
    verify_governance_or_bootstrap()?;
    if member_did.is_empty() || member_did.len() > MAX_DID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    let (_, action_hash) = find_mycel_state_with_hash(&member_did)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Member MYCEL state not found".into())
    ))?;

    let now = sys_time()?;

    // Set all scores to 0 (dissolved)
    let dissolved_state = MemberMycelState {
        member_did: member_did.clone(),
        mycel_score: 0.0,
        participation: 0.0,
        recognition: 0.0,
        validation: 0.0,
        longevity: 0.0,
        active_months: 0,
        is_apprentice: false,
        mentor_did: None,
        recognitions_given_this_cycle: 0,
        current_cycle_id: String::new(),
        last_updated: now,
    };

    update_entry(action_hash, &dissolved_state)?;

    Ok(())
}

// =============================================================================
// PASSIVE DECAY
// =============================================================================

/// Apply passive MYCEL decay for non-participation.
///
/// Per the Three-Currency Spec: 5% annual linear decay for non-participation.
/// Decay is proportional to time elapsed since last update.
///
/// `new_score = score - score * PASSIVE_DECAY_RATE * (elapsed_seconds / seconds_per_year)`
///
/// Restricted to authorized governance agents (or any agent during bootstrap).
#[hdk_extern]
pub fn apply_passive_decay(member_did: String) -> ExternResult<MemberMycelState> {
    verify_governance_or_bootstrap()?;
    if member_did.is_empty() || member_did.len() > MAX_DID_LENGTH {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    let (current_state, action_hash) = find_mycel_state_with_hash(&member_did)?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Member MYCEL state not found".into())),
    )?;

    // Don't decay dissolved members (score already 0)
    if current_state.mycel_score <= 0.0 {
        return Ok(current_state);
    }

    let now = sys_time()?;
    let elapsed_us = now.as_micros() - current_state.last_updated.as_micros();
    if elapsed_us <= 0 {
        return Ok(current_state);
    }

    let elapsed_seconds = elapsed_us as f64 / 1_000_000.0;
    let seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0;
    let years_elapsed = elapsed_seconds / seconds_per_year;

    // Linear decay: score -= score * rate * years
    let decay_amount = current_state.mycel_score * PASSIVE_DECAY_RATE * years_elapsed;
    let new_score = (current_state.mycel_score - decay_amount).max(0.0);

    // If decay is negligible, skip the update
    if (new_score - current_state.mycel_score).abs() < 1e-9 {
        return Ok(current_state);
    }

    let updated_state = MemberMycelState {
        mycel_score: new_score,
        last_updated: now,
        ..current_state
    };

    update_entry(action_hash, &updated_state)?;

    Ok(updated_state)
}

// =============================================================================
// APPRENTICE LIFECYCLE
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct OnboardApprenticeInput {
    /// DID of the new apprentice
    pub apprentice_did: String,
}

/// Onboard a new apprentice — the caller becomes the vouching mentor.
///
/// Per the Three-Currency Spec:
/// - Mentor must have MYCEL >= 0.3
/// - Apprentice starts at MYCEL 0.1
/// - Apprentice has limited TEND capacity (±10)
/// - Can receive recognition but cannot give
/// - Mentor shares MYCEL penalty if apprentice is slashed during onboarding
#[hdk_extern]
pub fn onboard_apprentice(input: OnboardApprenticeInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let mentor_did = format!("{}{}", DID_METHOD_PREFIX, caller);

    // Validate apprentice DID
    if !input.apprentice_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Apprentice must be a valid DID".into()
        )));
    }
    if mentor_did == input.apprentice_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot mentor yourself".into()
        )));
    }

    // Verify mentor has sufficient MYCEL
    let mentor_state = get_mycel_state(&mentor_did)?;
    if mentor_state.mycel_score < MYCEL_APPRENTICE_MAX {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Mentor MYCEL ({:.2}) must be at least {} to vouch for an apprentice",
            mentor_state.mycel_score, MYCEL_APPRENTICE_MAX
        ))));
    }

    // Check apprentice doesn't already have a state
    if find_mycel_state(&input.apprentice_did)?.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member already has a MYCEL state — cannot onboard as apprentice".into()
        )));
    }

    let now = sys_time()?;
    let state = MemberMycelState {
        member_did: input.apprentice_did.clone(),
        mycel_score: 0.1,
        participation: 0.1,
        recognition: 0.0,
        validation: 0.0,
        longevity: 0.0,
        active_months: 0,
        is_apprentice: true,
        mentor_did: Some(mentor_did),
        recognitions_given_this_cycle: 0,
        current_cycle_id: String::new(),
        last_updated: now,
    };

    let state_hash = create_entry(&EntryTypes::MemberMycelState(state))?;
    create_link(
        anchor_hash(&format!("mycel:{}", input.apprentice_did))?,
        state_hash.clone(),
        LinkTypes::MemberToMycelState,
        (),
    )?;

    get(state_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Graduate an apprentice to full member status.
///
/// Can be called when the apprentice's composite MYCEL score >= 0.3.
/// The caller must be the apprentice themselves or their mentor.
#[hdk_extern]
pub fn graduate_apprentice(apprentice_did: String) -> ExternResult<MemberMycelState> {
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("{}{}", DID_METHOD_PREFIX, caller);

    let (current_state, action_hash) =
        find_mycel_state_with_hash(&apprentice_did)?.ok_or(wasm_error!(WasmErrorInner::Guest(
            "Apprentice MYCEL state not found".into()
        )))?;

    if !current_state.is_apprentice {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member is not an apprentice".into()
        )));
    }

    // Caller must be the apprentice or their mentor
    let is_apprentice = caller_did == apprentice_did;
    let is_mentor = current_state.mentor_did.as_ref() == Some(&caller_did);
    if !is_apprentice && !is_mentor {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the apprentice or their mentor can trigger graduation".into()
        )));
    }

    if current_state.mycel_score < MYCEL_APPRENTICE_MAX {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "MYCEL score ({:.2}) must reach {} to graduate",
            current_state.mycel_score, MYCEL_APPRENTICE_MAX
        ))));
    }

    let now = sys_time()?;
    let graduated = MemberMycelState {
        is_apprentice: false,
        mentor_did: None, // Mentor relationship ends at graduation
        last_updated: now,
        ..current_state
    };

    update_entry(action_hash, &graduated)?;
    Ok(graduated)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get a member's MYCEL state, creating an apprentice state if not found
fn get_mycel_state(member_did: &str) -> ExternResult<MemberMycelState> {
    if let Some(state) = find_mycel_state(member_did)? {
        return Ok(state);
    }

    // If no state exists, return a default apprentice state
    // (the member should be properly initialized via initialize_member)
    let now = sys_time()?;
    Ok(MemberMycelState {
        member_did: member_did.to_string(),
        mycel_score: 0.1,
        participation: 0.1,
        recognition: 0.0,
        validation: 0.0,
        longevity: 0.0,
        active_months: 0,
        is_apprentice: true,
        mentor_did: None,
        recognitions_given_this_cycle: 0,
        current_cycle_id: String::new(),
        last_updated: now,
    })
}

fn find_mycel_state(member_did: &str) -> ExternResult<Option<MemberMycelState>> {
    Ok(find_mycel_state_with_hash(member_did)?.map(|(state, _)| state))
}

fn find_mycel_state_with_hash(
    member_did: &str,
) -> ExternResult<Option<(MemberMycelState, ActionHash)>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mycel:{}", member_did))?,
            LinkTypes::MemberToMycelState,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(link_hash)?;
            let state = record
                .entry()
                .to_app_option::<MemberMycelState>()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "MemberMycelState deserialization error: {:?}",
                        e
                    )))
                })?;
            if let Some(state) = state {
                return Ok(Some((state, record.action_address().clone())));
            }
        }
    }

    Ok(None)
}

fn get_or_create_allocation(
    recognizer_did: &str,
    cycle_id: &str,
) -> ExternResult<RecognitionAllocation> {
    let anchor_key = format!("alloc:{}:{}", recognizer_did, cycle_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&anchor_key)?,
            LinkTypes::RecognizerCycleToAllocation,
        )?,
        GetStrategy::default(),
    )?;

    // Pick the link with the lowest target ActionHash (deterministic winner)
    // to handle orphaned links from past race conditions (RC-15).
    if let Some(action_hash) = links
        .iter()
        .filter_map(|l| l.target.clone().into_action_hash())
        .min()
    {
        let record = follow_update_chain(action_hash)?;
        let alloc = record
            .entry()
            .to_app_option::<RecognitionAllocation>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "RecognitionAllocation deserialization error: {:?}",
                    e
                )))
            })?;
        if let Some(alloc) = alloc {
            return Ok(alloc);
        }
    }

    // Create new allocation for this cycle
    let alloc = RecognitionAllocation {
        recognizer_did: recognizer_did.to_string(),
        cycle_id: cycle_id.to_string(),
        count: 0,
    };

    let alloc_hash = create_entry(&EntryTypes::RecognitionAllocation(alloc.clone()))?;

    let anchor = anchor_hash(&anchor_key)?;
    create_link(
        anchor.clone(),
        alloc_hash.clone(),
        LinkTypes::RecognizerCycleToAllocation,
        (),
    )?;

    // RC-15: Race condition guard — re-read links to detect concurrent creators.
    let recheck_links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::RecognizerCycleToAllocation)?,
        GetStrategy::default(),
    )?;
    if recheck_links.len() > 1 {
        if let Some(winner_hash) = recheck_links
            .iter()
            .filter_map(|l| l.target.clone().into_action_hash())
            .min()
        {
            if winner_hash != alloc_hash {
                // We lost the race — return the winner's entry instead
                let record = follow_update_chain(winner_hash)?;
                let winner_alloc = record
                    .entry()
                    .to_app_option::<RecognitionAllocation>()
                    .map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "RecognitionAllocation deserialization error: {:?}",
                            e
                        )))
                    })?
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Winner allocation entry missing".into()
                    )))?;
                return Ok(winner_alloc);
            }
        }
    }

    Ok(alloc)
}

fn increment_allocation(recognizer_did: &str, cycle_id: &str) -> ExternResult<()> {
    let anchor_key = format!("alloc:{}:{}", recognizer_did, cycle_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&anchor_key)?,
            LinkTypes::RecognizerCycleToAllocation,
        )?,
        GetStrategy::default(),
    )?;

    // Pick the link with the lowest target ActionHash (deterministic winner)
    // to handle orphaned links from past race conditions (RC-15).
    if let Some(link_hash) = links
        .iter()
        .filter_map(|l| l.target.clone().into_action_hash())
        .min()
    {
        let record = follow_update_chain(link_hash)?;
        if let Some(mut alloc) = record
            .entry()
            .to_app_option::<RecognitionAllocation>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "RecognitionAllocation deserialization error: {:?}",
                    e
                )))
            })?
        {
            alloc.count += 1;
            update_entry(record.action_address().clone(), &alloc)?;
        }
    }

    Ok(())
}

/// Fetch validation score from TEND quality ratings via cross-zome call.
///
/// Calls TEND coordinator's `get_validation_score` to aggregate quality
/// ratings for the member. Falls back to 0.0 if the TEND zome is
/// unreachable (e.g., in unit tests without full DNA).
fn fetch_validation_from_tend(member_did: &str) -> ExternResult<f64> {
    #[derive(Serialize, Debug)]
    struct ValidationScoreInput {
        member_did: String,
        limit: Option<usize>,
    }
    match call(
        CallTargetCell::Local,
        ZomeName::from("tend"),
        FunctionName::from("get_validation_score"),
        None,
        ValidationScoreInput {
            member_did: member_did.to_string(),
            limit: None,
        },
    ) {
        Ok(ZomeCallResponse::Ok(result)) => result.decode::<f64>().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode TEND validation score: {:?}",
                e
            )))
        }),
        Ok(other) => {
            // Zome call was routed but returned non-Ok (e.g., Unauthorized)
            debug!(
                "TEND validation cross-zome call returned {:?}, defaulting to 0.0",
                other
            );
            Ok(0.0)
        }
        Err(_) => {
            // TEND zome unreachable — default to 0.0
            Ok(0.0)
        }
    }
}
