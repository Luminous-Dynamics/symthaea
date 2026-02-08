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

// Re-export integrity types for external use
pub use recognition_integrity::*;

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
    pub validation: f64,
    pub active_months: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetRecognitionsInput {
    pub member_did: String,
    pub cycle_id: Option<String>,
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
    if input.recipient_did.is_empty() || input.recipient_did.len() > 256 {
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
    let caller_did = format!("did:mycelix:{}", caller);

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
    let record = get(event_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to retrieve created recognition record".into()
        )))?;

    Ok(record)
}

/// Get all recognitions received by a member, optionally filtered by cycle
#[hdk_extern]
pub fn get_recognition_received(input: GetRecognitionsInput) -> ExternResult<Vec<RecognitionEvent>> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

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
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(event) = record.entry().to_app_option::<RecognitionEvent>().ok().flatten() {
                    // Filter by cycle if specified
                    if let Some(ref cycle) = input.cycle_id {
                        if event.cycle_id == *cycle {
                            events.push(event);
                        }
                    } else {
                        events.push(event);
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
    if member_did.is_empty() || member_did.len() > 256 {
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
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    if input.is_apprentice {
        let mentor = input.mentor_did.as_ref().ok_or(wasm_error!(
            WasmErrorInner::Guest("Apprentices must have a mentor DID".into())
        ))?;
        if mentor.is_empty() || mentor.len() > 256 {
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

    let record = get(state_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to retrieve created MYCEL state".into()
        )))?;

    Ok(record)
}

/// Update a member's MYCEL score components
///
/// Recalculates the composite MYCEL score from the 4 components:
/// Participation (40%), Recognition (20%), Validation (20%), Longevity (20%)
#[hdk_extern]
pub fn update_mycel_score(input: UpdateMycelInput) -> ExternResult<MemberMycelState> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    let (current_state, action_hash) = find_mycel_state_with_hash(&input.member_did)?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Member MYCEL state not found — initialize first".into()
        )))?;

    let now = sys_time()?;
    let participation = input.participation.clamp(0.0, 1.0);
    let recognition = input.recognition.clamp(0.0, 1.0);
    let validation = input.validation.clamp(0.0, 1.0);
    let longevity = (input.active_months as f64 / 24.0).min(1.0);

    let composite = participation * 0.40
        + recognition * 0.20
        + validation * 0.20
        + longevity * 0.20;

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
#[hdk_extern]
pub fn jubilee_normalize(member_did: String) -> ExternResult<MemberMycelState> {
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    let (current_state, action_hash) = find_mycel_state_with_hash(&member_did)?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Member MYCEL state not found".into()
        )))?;

    let now = sys_time()?;
    let normalized_score = (0.3 + (current_state.mycel_score - 0.3) * JUBILEE_COMPRESSION)
        .clamp(0.0, 1.0);

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
#[hdk_extern]
pub fn dissolve_mycel(member_did: String) -> ExternResult<()> {
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    let (_, action_hash) = find_mycel_state_with_hash(&member_did)?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Member MYCEL state not found".into()
        )))?;

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
// HELPER FUNCTIONS
// =============================================================================

fn anchor_hash(anchor: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(anchor.to_string()))
}

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

fn find_mycel_state_with_hash(member_did: &str) -> ExternResult<Option<(MemberMycelState, ActionHash)>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mycel:{}", member_did))?,
            LinkTypes::MemberToMycelState,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(state) = record.entry().to_app_option::<MemberMycelState>().ok().flatten() {
                    return Ok(Some((state, action_hash)));
                }
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

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(alloc) = record.entry().to_app_option::<RecognitionAllocation>().ok().flatten() {
                    return Ok(alloc);
                }
            }
        }
    }

    // Create new allocation for this cycle
    let alloc = RecognitionAllocation {
        recognizer_did: recognizer_did.to_string(),
        cycle_id: cycle_id.to_string(),
        count: 0,
    };

    let alloc_hash = create_entry(&EntryTypes::RecognitionAllocation(alloc.clone()))?;

    create_link(
        anchor_hash(&anchor_key)?,
        alloc_hash,
        LinkTypes::RecognizerCycleToAllocation,
        (),
    )?;

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

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(mut alloc) = record.entry().to_app_option::<RecognitionAllocation>().ok().flatten() {
                    alloc.count += 1;
                    update_entry(action_hash, &alloc)?;
                }
            }
        }
    }

    Ok(())
}
