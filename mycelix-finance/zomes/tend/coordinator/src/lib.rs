//! TEND (Time Exchange) Coordinator Zome
//!
//! Implements Commons Charter Article II, Section 2 - Time Exchange Module
//!
//! Key Features:
//! - Record time exchanges between community members
//! - Track balances (mutual credit: total always sums to zero)
//! - Enforce balance limits (±40 standard, ±10 apprentice)
//! - Service listings and requests marketplace
//! - Quality ratings for confirmed exchanges
//! - Dispute resolution with escalation stages
//!
//! Philosophy: All hours are equal. A doctor's hour = a gardener's hour.
//! This radical equality is the foundation of time banking.

use hdk::prelude::*;

// Re-export integrity types for external use
pub use tend_integrity::*;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Balance limit for apprentice-tier members (lower than standard)
pub const APPRENTICE_BALANCE_LIMIT: i32 = 10;

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordExchangeInput {
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub cultural_alias: Option<String>,
    pub dao_did: String,
    pub service_date: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExchangeRecord {
    pub id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub status: ExchangeStatus,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BalanceInfo {
    pub member_did: String,
    pub dao_did: String,
    pub balance: i32,
    pub can_provide: bool,   // Can still provide (balance < +limit)
    pub can_receive: bool,   // Can still receive (balance > -limit)
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateListingInput {
    pub dao_did: String,
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub estimated_hours: Option<f32>,
    pub availability: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRequestInput {
    pub dao_did: String,
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub estimated_hours: Option<f32>,
    pub urgency: Urgency,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetBalanceInput {
    pub member_did: String,
    pub dao_did: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RateExchangeInput {
    pub exchange_id: String,
    pub rating: u8,
    pub comment: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenDisputeInput {
    pub exchange_id: String,
    pub description: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub dispute_id: String,
    pub resolution: String,
}

// QualityRating, DisputeCase, DisputeStage, and TendLimitTier are imported
// from tend_integrity via `pub use tend_integrity::*` above.

// =============================================================================
// TIER-BASED LIMIT FUNCTION
// =============================================================================

/// Get the current TEND balance limit for a given tier
///
/// Returns the maximum absolute balance allowed for the given oracle tier.
/// Uses the TendLimitTier enum from the integrity zome which maps to
/// Normal(±40), Elevated(±60), High(±80), Emergency(±120).
/// Apprentice limit (±10) is handled separately via member tier checks.
#[hdk_extern]
pub fn get_current_tend_limit(tier: TendLimitTier) -> ExternResult<i32> {
    Ok(tier.limit())
}

// =============================================================================
// CORE EXCHANGE FUNCTIONS
// =============================================================================

/// Record a time exchange
///
/// Called by the PROVIDER after providing a service.
/// The exchange starts in "Proposed" status until the receiver confirms.
///
/// Effect on balances (after confirmation):
/// - Provider: +hours (credit)
/// - Receiver: -hours (debt)
///
/// Enforces apprentice balance limits (±10 TEND) in addition to the
/// standard balance limit (±40 TEND). For now, all members are checked
/// against BALANCE_LIMIT; apprentice enforcement uses APPRENTICE_BALANCE_LIMIT.
#[hdk_extern]
pub fn record_exchange(input: RecordExchangeInput) -> ExternResult<ExchangeRecord> {
    if input.receiver_did.is_empty() || input.receiver_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Receiver DID must be 1-256 characters".into())));
    }
    if input.hours <= 0.0 || input.hours > 168.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Hours must be between 0 and 168 (one week)".into())));
    }
    if input.service_description.is_empty() || input.service_description.len() > 1024 {
        return Err(wasm_error!(WasmErrorInner::Guest("Service description must be 1-1024 characters".into())));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("DAO DID must be 1-256 characters".into())));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let provider_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    // Validate not exchanging with self
    if provider_did == input.receiver_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot exchange time with yourself".into()
        )));
    }

    // Determine effective limits for each party
    // In production, tier would be fetched from the oracle/membership zome.
    // For now, check against both BALANCE_LIMIT (standard) and APPRENTICE_BALANCE_LIMIT.
    let provider_limit = get_effective_limit_for_member(&provider_did)?;
    let receiver_limit = get_effective_limit_for_member(&input.receiver_did)?;

    // Check provider's balance (can still earn if below +limit)
    let provider_balance = get_or_create_balance(provider_did.clone(), input.dao_did.clone())?;
    let new_provider_balance = provider_balance.balance + (input.hours as i32);
    if new_provider_balance > provider_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Exchange would exceed your credit limit of +{}. Current balance: {}",
            provider_limit, provider_balance.balance
        ))));
    }

    // Check receiver's balance (can still receive if above -limit)
    let receiver_balance = get_or_create_balance(input.receiver_did.clone(), input.dao_did.clone())?;
    let new_receiver_balance = receiver_balance.balance - (input.hours as i32);
    if new_receiver_balance < -receiver_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Receiver would exceed debt limit of -{}. Their balance: {}",
            receiver_limit, receiver_balance.balance
        ))));
    }

    // Create the exchange
    let exchange_id = format!("tend:{}:{}:{}", provider_did, input.receiver_did, now.as_micros());
    let exchange = TendExchange {
        id: exchange_id.clone(),
        provider_did: provider_did.clone(),
        receiver_did: input.receiver_did.clone(),
        hours: input.hours,
        service_description: input.service_description.clone(),
        service_category: input.service_category.clone(),
        cultural_alias: input.cultural_alias,
        dao_did: input.dao_did.clone(),
        timestamp: now,
        status: ExchangeStatus::Proposed,
        service_date: input.service_date,
    };

    let exchange_hash = create_entry(&EntryTypes::TendExchange(exchange.clone()))?;

    // Create links
    create_link(
        anchor_hash(&format!("provider:{}:{}", input.dao_did, provider_did))?,
        exchange_hash.clone(),
        LinkTypes::ProviderToExchanges,
        (),
    )?;

    create_link(
        anchor_hash(&format!("receiver:{}:{}", input.dao_did, input.receiver_did))?,
        exchange_hash.clone(),
        LinkTypes::ReceiverToExchanges,
        (),
    )?;

    create_link(
        anchor_hash(&format!("dao:{}", input.dao_did))?,
        exchange_hash.clone(),
        LinkTypes::DaoToExchanges,
        (),
    )?;

    // Create index link for lookup by exchange ID
    create_link(
        anchor_hash(&format!("exchange:{}", exchange_id))?,
        exchange_hash,
        LinkTypes::ExchangeIdToExchange,
        (),
    )?;

    Ok(ExchangeRecord {
        id: exchange_id,
        provider_did,
        receiver_did: input.receiver_did,
        hours: input.hours,
        service_description: input.service_description,
        service_category: input.service_category,
        status: ExchangeStatus::Proposed,
        timestamp: now,
    })
}

/// Confirm an exchange (called by receiver)
///
/// This finalizes the exchange and updates both balances.
/// Enforces apprentice balance limits before confirming.
#[hdk_extern]
pub fn confirm_exchange(exchange_id: String) -> ExternResult<ExchangeRecord> {
    if exchange_id.is_empty() || exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Exchange ID must be 1-256 characters".into())));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the exchange
    let exchange = find_exchange_by_id(&exchange_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Exchange not found".into())))?;

    // Verify caller is the receiver
    if exchange.receiver_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the receiver can confirm an exchange".into()
        )));
    }

    // Verify status is Proposed
    if exchange.status != ExchangeStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange is not in Proposed status".into()
        )));
    }

    // Re-check balance limits at confirmation time (balances may have changed)
    let provider_limit = get_effective_limit_for_member(&exchange.provider_did)?;
    let receiver_limit = get_effective_limit_for_member(&exchange.receiver_did)?;

    let provider_balance = get_or_create_balance(exchange.provider_did.clone(), exchange.dao_did.clone())?;
    let new_provider_balance = provider_balance.balance + (exchange.hours as i32);
    if new_provider_balance > provider_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot confirm: provider would exceed credit limit of +{}. Current balance: {}",
            provider_limit, provider_balance.balance
        ))));
    }

    let receiver_balance = get_or_create_balance(exchange.receiver_did.clone(), exchange.dao_did.clone())?;
    let new_receiver_balance = receiver_balance.balance - (exchange.hours as i32);
    if new_receiver_balance < -receiver_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot confirm: receiver would exceed debt limit of -{}. Current balance: {}",
            receiver_limit, receiver_balance.balance
        ))));
    }

    // Update balances
    update_balance_after_exchange(
        &exchange.provider_did,
        &exchange.dao_did,
        exchange.hours,
        true, // provider gains
    )?;

    update_balance_after_exchange(
        &exchange.receiver_did,
        &exchange.dao_did,
        exchange.hours,
        false, // receiver pays
    )?;

    // Update exchange status
    let updated_exchange = TendExchange {
        status: ExchangeStatus::Confirmed,
        ..exchange.clone()
    };

    // Find and update the entry
    update_exchange_entry(&exchange_id, &updated_exchange)?;

    Ok(ExchangeRecord {
        id: exchange.id,
        provider_did: exchange.provider_did,
        receiver_did: exchange.receiver_did,
        hours: exchange.hours,
        service_description: exchange.service_description,
        service_category: exchange.service_category,
        status: ExchangeStatus::Confirmed,
        timestamp: exchange.timestamp,
    })
}

/// Dispute an exchange
#[hdk_extern]
pub fn dispute_exchange(exchange_id: String) -> ExternResult<ExchangeRecord> {
    if exchange_id.is_empty() || exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Exchange ID must be 1-256 characters".into())));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    let exchange = find_exchange_by_id(&exchange_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Exchange not found".into())))?;

    // Either party can dispute
    if exchange.provider_did != caller_did && exchange.receiver_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only exchange participants can dispute".into()
        )));
    }

    let updated_exchange = TendExchange {
        status: ExchangeStatus::Disputed,
        ..exchange.clone()
    };

    update_exchange_entry(&exchange_id, &updated_exchange)?;

    Ok(ExchangeRecord {
        id: exchange.id,
        provider_did: exchange.provider_did,
        receiver_did: exchange.receiver_did,
        hours: exchange.hours,
        service_description: exchange.service_description,
        service_category: exchange.service_category,
        status: ExchangeStatus::Disputed,
        timestamp: exchange.timestamp,
    })
}

/// Cancel an exchange (only if still Proposed)
#[hdk_extern]
pub fn cancel_exchange(exchange_id: String) -> ExternResult<ExchangeRecord> {
    if exchange_id.is_empty() || exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Exchange ID must be 1-256 characters".into())));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    let exchange = find_exchange_by_id(&exchange_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Exchange not found".into())))?;

    // Only provider can cancel a proposed exchange
    if exchange.provider_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the provider can cancel a proposed exchange".into()
        )));
    }

    if exchange.status != ExchangeStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only cancel exchanges in Proposed status".into()
        )));
    }

    let updated_exchange = TendExchange {
        status: ExchangeStatus::Cancelled,
        ..exchange.clone()
    };

    update_exchange_entry(&exchange_id, &updated_exchange)?;

    Ok(ExchangeRecord {
        id: exchange.id,
        provider_did: exchange.provider_did,
        receiver_did: exchange.receiver_did,
        hours: exchange.hours,
        service_description: exchange.service_description,
        service_category: exchange.service_category,
        status: ExchangeStatus::Cancelled,
        timestamp: exchange.timestamp,
    })
}

// =============================================================================
// QUALITY RATING FUNCTIONS
// =============================================================================

/// Rate a confirmed exchange
///
/// Creates a QualityRating for a confirmed exchange. Only the receiver of
/// the exchange can rate the provider's service. Rating must be 1-5.
/// The rating is stored as a proper QualityRating entry type and linked
/// to the exchange and the rated member for retrieval.
#[hdk_extern]
pub fn rate_exchange(input: RateExchangeInput) -> ExternResult<Record> {
    // Validate rating range
    if input.rating < 1 || input.rating > 5 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Rating must be between 1 and 5".into()
        )));
    }

    if input.exchange_id.is_empty() || input.exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange ID must be 1-256 characters".into()
        )));
    }

    if let Some(ref comment) = input.comment {
        if comment.len() > 2048 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Comment must be under 2048 characters".into()
            )));
        }
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the exchange
    let exchange = find_exchange_by_id(&input.exchange_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Exchange not found".into())))?;

    // Exchange must be Confirmed
    if exchange.status != ExchangeStatus::Confirmed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only rate confirmed exchanges".into()
        )));
    }

    // Caller must be the receiver (rating the provider's service)
    if exchange.receiver_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the receiver can rate an exchange".into()
        )));
    }

    // Check for duplicate rating via ExchangeToRating link
    let existing_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("rating:{}", input.exchange_id))?,
            LinkTypes::ExchangeToRating,
        )?,
        GetStrategy::default(),
    )?;
    if !existing_links.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "This exchange has already been rated".into()
        )));
    }

    let now = sys_time()?;

    let quality_rating = QualityRating {
        exchange_id: input.exchange_id.clone(),
        rater_did: caller_did,
        provider_did: exchange.provider_did.clone(),
        rating: input.rating,
        comment: input.comment,
        timestamp: now,
    };

    // Store as a proper QualityRating entry type
    let rating_hash = create_entry(&EntryTypes::QualityRating(quality_rating))?;

    // Link from exchange to rating
    create_link(
        anchor_hash(&format!("rating:{}", input.exchange_id))?,
        rating_hash.clone(),
        LinkTypes::ExchangeToRating,
        (),
    )?;

    // Link from rated member (provider) to rating for aggregation
    create_link(
        anchor_hash(&format!("ratings_for:{}", exchange.provider_did))?,
        rating_hash.clone(),
        LinkTypes::ExchangeToRating,
        (),
    )?;

    // Return the Record
    let record = get(rating_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to retrieve created rating record".into()
        )))?;

    Ok(record)
}

// =============================================================================
// DISPUTE RESOLUTION FUNCTIONS
// =============================================================================

/// Open a dispute case for an exchange
///
/// Creates a DisputeCase in DirectNegotiation stage. The caller must be
/// a participant in the exchange (provider or receiver). The dispute is
/// stored as a proper DisputeCase entry and linked to the exchange and
/// both members for retrieval.
#[hdk_extern]
pub fn open_dispute(input: OpenDisputeInput) -> ExternResult<Record> {
    if input.exchange_id.is_empty() || input.exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange ID must be 1-256 characters".into()
        )));
    }
    if input.description.is_empty() || input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-4096 characters".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the exchange
    let exchange = find_exchange_by_id(&input.exchange_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Exchange not found".into())))?;

    // Caller must be a participant
    if exchange.provider_did != caller_did && exchange.receiver_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only exchange participants can open a dispute".into()
        )));
    }

    // Check for existing dispute on this exchange via ExchangeToDispute link
    let existing_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dispute_for_exchange:{}", input.exchange_id))?,
            LinkTypes::ExchangeToDispute,
        )?,
        GetStrategy::default(),
    )?;
    if !existing_links.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "A dispute already exists for this exchange".into()
        )));
    }

    // Determine the other party (respondent)
    let respondent_did = if exchange.provider_did == caller_did {
        exchange.receiver_did.clone()
    } else {
        exchange.provider_did.clone()
    };

    let now = sys_time()?;
    let dispute_id = format!("dispute:{}:{}", input.exchange_id, now.as_micros());

    let dispute_case = DisputeCase {
        id: dispute_id.clone(),
        exchange_id: input.exchange_id.clone(),
        complainant_did: caller_did.clone(),
        respondent_did: respondent_did.clone(),
        stage: DisputeStage::DirectNegotiation,
        description: input.description,
        mediator_dids: Vec::new(),
        resolution: None,
        opened_at: now,
        escalated_at: None,
        resolved_at: None,
    };

    // Store as a proper DisputeCase entry type
    let dispute_hash = create_entry(&EntryTypes::DisputeCase(dispute_case))?;

    // Link from exchange to dispute
    create_link(
        anchor_hash(&format!("dispute_for_exchange:{}", input.exchange_id))?,
        dispute_hash.clone(),
        LinkTypes::ExchangeToDispute,
        (),
    )?;

    // Link from dispute ID for direct lookup
    create_link(
        anchor_hash(&format!("dispute:{}", dispute_id))?,
        dispute_hash.clone(),
        LinkTypes::ExchangeToDispute,
        (),
    )?;

    // Link to complainant member
    create_link(
        anchor_hash(&format!("disputes_for:{}", caller_did))?,
        dispute_hash.clone(),
        LinkTypes::MemberToDisputes,
        (),
    )?;

    // Link to respondent member
    create_link(
        anchor_hash(&format!("disputes_for:{}", respondent_did))?,
        dispute_hash.clone(),
        LinkTypes::MemberToDisputes,
        (),
    )?;

    // Also mark the exchange as Disputed
    let updated_exchange = TendExchange {
        status: ExchangeStatus::Disputed,
        ..exchange.clone()
    };
    update_exchange_entry(&input.exchange_id, &updated_exchange)?;

    // Return the Record
    let record = get(dispute_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to retrieve created dispute record".into()
        )))?;

    Ok(record)
}

/// Escalate a dispute to the next resolution stage
///
/// Escalation path:
///   DirectNegotiation -> MediationPanel -> GovernanceVote
///
/// Only dispute participants can escalate. A dispute that is already
/// at GovernanceVote cannot be escalated further.
#[hdk_extern]
pub fn escalate_dispute(dispute_id: String) -> ExternResult<Record> {
    if dispute_id.is_empty() || dispute_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispute ID must be 1-256 characters".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the dispute
    let (dispute_case, action_hash) = find_dispute_by_id(&dispute_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Dispute not found".into())))?;

    // Verify caller is a participant
    if dispute_case.complainant_did != caller_did && dispute_case.respondent_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only dispute participants can escalate".into()
        )));
    }

    // Determine next stage
    let next_stage = match dispute_case.stage {
        DisputeStage::DirectNegotiation => DisputeStage::MediationPanel,
        DisputeStage::MediationPanel => DisputeStage::GovernanceVote,
        DisputeStage::GovernanceVote => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Dispute is already at the highest escalation level (GovernanceVote)".into()
            )));
        }
    };

    let now = sys_time()?;
    let updated_dispute = DisputeCase {
        stage: next_stage,
        escalated_at: Some(now),
        ..dispute_case
    };

    // Update the entry in place
    update_entry(action_hash, &updated_dispute)?;

    // Re-fetch the updated record via the link
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dispute:{}", dispute_id))?,
            LinkTypes::ExchangeToDispute,
        )?,
        GetStrategy::default(),
    )?;

    let link = links.first()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Dispute link not found".into())))?;
    let hash = link.target.clone().into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid dispute link target".into())))?;

    let record = get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to retrieve updated dispute record".into()
        )))?;

    Ok(record)
}

/// Resolve a dispute
///
/// Marks the dispute as resolved with the given resolution description.
/// Only dispute participants can resolve a dispute. The integrity zome's
/// DisputeStage enum does not have a Resolved variant — resolution is
/// indicated by the `resolved_at` timestamp and `resolution` field being set.
#[hdk_extern]
pub fn resolve_dispute(input: ResolveDisputeInput) -> ExternResult<Record> {
    if input.dispute_id.is_empty() || input.dispute_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispute ID must be 1-256 characters".into()
        )));
    }
    if input.resolution.is_empty() || input.resolution.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Resolution must be 1-4096 characters".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the dispute
    let (dispute_case, action_hash) = find_dispute_by_id(&input.dispute_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Dispute not found".into())))?;

    // Verify caller is a participant
    if dispute_case.complainant_did != caller_did && dispute_case.respondent_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only dispute participants can resolve".into()
        )));
    }

    // Cannot resolve an already-resolved dispute
    if dispute_case.resolved_at.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispute is already resolved".into()
        )));
    }

    let now = sys_time()?;
    let resolved_dispute = DisputeCase {
        resolution: Some(input.resolution),
        resolved_at: Some(now),
        ..dispute_case.clone()
    };

    // Update the entry in place
    update_entry(action_hash, &resolved_dispute)?;

    // Also update the exchange status to Resolved if it was Disputed
    if let Some(exchange) = find_exchange_by_id(&dispute_case.exchange_id)? {
        if exchange.status == ExchangeStatus::Disputed {
            let resolved_exchange = TendExchange {
                status: ExchangeStatus::Resolved,
                ..exchange
            };
            update_exchange_entry(&dispute_case.exchange_id, &resolved_exchange)?;
        }
    }

    // Re-fetch the updated record via the link
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dispute:{}", input.dispute_id))?,
            LinkTypes::ExchangeToDispute,
        )?,
        GetStrategy::default(),
    )?;

    let link = links.first()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Dispute link not found".into())))?;
    let hash = link.target.clone().into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid dispute link target".into())))?;

    let record = get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to retrieve resolved dispute record".into()
        )))?;

    Ok(record)
}

// =============================================================================
// BALANCE FUNCTIONS
// =============================================================================

/// Get or create balance for a member in a DAO (internal function)
fn get_or_create_balance(member_did: String, dao_did: String) -> ExternResult<BalanceInfo> {
    // Try to find existing balance
    if let Some(balance) = find_balance(&member_did, &dao_did)? {
        return Ok(balance_to_info(&balance));
    }

    // Create new balance (starts at 0)
    let now = sys_time()?;
    let balance = TendBalance {
        member_did: member_did.clone(),
        dao_did: dao_did.clone(),
        balance: 0,
        total_provided: 0.0,
        total_received: 0.0,
        exchange_count: 0,
        last_activity: now,
    };

    let action_hash = create_entry(&EntryTypes::TendBalance(balance.clone()))?;

    create_link(
        anchor_hash(&format!("balance:{}:{}", dao_did, member_did))?,
        action_hash,
        LinkTypes::MemberToBalance,
        (),
    )?;

    Ok(balance_to_info(&balance))
}

/// Get balance info for a member
#[hdk_extern]
pub fn get_balance(input: GetBalanceInput) -> ExternResult<BalanceInfo> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Member DID must be 1-256 characters".into())));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("DAO DID must be 1-256 characters".into())));
    }
    get_or_create_balance(input.member_did, input.dao_did)
}

/// Get all exchanges for a member in a DAO
#[hdk_extern]
pub fn get_my_exchanges(dao_did: String) -> ExternResult<Vec<ExchangeRecord>> {
    if dao_did.is_empty() || dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("DAO DID must be 1-256 characters".into())));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let member_did = format!("did:mycelix:{}", caller);

    let mut exchanges = Vec::new();

    // Get exchanges where member was provider
    let provider_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("provider:{}:{}", dao_did, member_did))?,
            LinkTypes::ProviderToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    for link in provider_links {
        if let Some(record) = get(link.target.into_action_hash().ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?, GetOptions::default())? {
            if let Some(exchange) = record.entry().to_app_option::<TendExchange>().ok().flatten() {
                exchanges.push(exchange_to_record(&exchange));
            }
        }
    }

    // Get exchanges where member was receiver
    let receiver_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("receiver:{}:{}", dao_did, member_did))?,
            LinkTypes::ReceiverToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    for link in receiver_links {
        if let Some(record) = get(link.target.into_action_hash().ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?, GetOptions::default())? {
            if let Some(exchange) = record.entry().to_app_option::<TendExchange>().ok().flatten() {
                // Avoid duplicates (shouldn't happen, but safety)
                if !exchanges.iter().any(|e| e.id == exchange.id) {
                    exchanges.push(exchange_to_record(&exchange));
                }
            }
        }
    }

    // Sort by timestamp (newest first)
    exchanges.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(exchanges)
}

// =============================================================================
// SERVICE MARKETPLACE FUNCTIONS
// =============================================================================

/// Create a service listing (offer to help)
#[hdk_extern]
pub fn create_listing(input: CreateListingInput) -> ExternResult<ServiceListing> {
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Title must be 1-256 characters".into())));
    }
    if input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Description must be under 4096 characters".into())));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("DAO DID must be 1-256 characters".into())));
    }
    if let Some(hours) = input.estimated_hours {
        if hours <= 0.0 || hours > 168.0 {
            return Err(wasm_error!(WasmErrorInner::Guest("Estimated hours must be between 0 and 168".into())));
        }
    }
    if let Some(ref avail) = input.availability {
        if avail.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest("Availability must be under 256 characters".into())));
        }
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let provider_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    let listing_id = format!("listing:{}:{}", provider_did, now.as_micros());
    let listing = ServiceListing {
        id: listing_id,
        provider_did: provider_did.clone(),
        dao_did: input.dao_did.clone(),
        title: input.title,
        description: input.description,
        category: input.category.clone(),
        estimated_hours: input.estimated_hours,
        availability: input.availability,
        active: true,
        created: now,
    };

    let listing_hash = create_entry(&EntryTypes::ServiceListing(listing.clone()))?;

    // Link to DAO
    create_link(
        anchor_hash(&format!("listings:{}", input.dao_did))?,
        listing_hash.clone(),
        LinkTypes::DaoToListings,
        (),
    )?;

    // Link to provider
    create_link(
        anchor_hash(&format!("my_listings:{}", provider_did))?,
        listing_hash.clone(),
        LinkTypes::ProviderToListings,
        (),
    )?;

    // Link to category
    create_link(
        anchor_hash(&format!("category:{}:{:?}", input.dao_did, input.category))?,
        listing_hash,
        LinkTypes::CategoryToListings,
        (),
    )?;

    Ok(listing)
}

/// Get all active listings in a DAO
#[hdk_extern]
pub fn get_dao_listings(dao_did: String) -> ExternResult<Vec<ServiceListing>> {
    if dao_did.is_empty() || dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("DAO DID must be 1-256 characters".into())));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("listings:{}", dao_did))?,
            LinkTypes::DaoToListings,
        )?,
        GetStrategy::default(),
    )?;

    let mut listings = Vec::new();
    for link in links {
        if let Some(record) = get(link.target.into_action_hash().ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?, GetOptions::default())? {
            if let Some(listing) = record.entry().to_app_option::<ServiceListing>().ok().flatten() {
                if listing.active {
                    listings.push(listing);
                }
            }
        }
    }

    Ok(listings)
}

/// Create a service request (ask for help)
#[hdk_extern]
pub fn create_request(input: CreateRequestInput) -> ExternResult<ServiceRequest> {
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Title must be 1-256 characters".into())));
    }
    if input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Description must be under 4096 characters".into())));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("DAO DID must be 1-256 characters".into())));
    }
    if let Some(hours) = input.estimated_hours {
        if hours <= 0.0 || hours > 168.0 {
            return Err(wasm_error!(WasmErrorInner::Guest("Estimated hours must be between 0 and 168".into())));
        }
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let requester_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    let request_id = format!("request:{}:{}", requester_did, now.as_micros());
    let request = ServiceRequest {
        id: request_id,
        requester_did,
        dao_did: input.dao_did.clone(),
        title: input.title,
        description: input.description,
        category: input.category,
        estimated_hours: input.estimated_hours,
        urgency: input.urgency,
        open: true,
        created: now,
    };

    let request_hash = create_entry(&EntryTypes::ServiceRequest(request.clone()))?;

    // Link to DAO
    create_link(
        anchor_hash(&format!("requests:{}", input.dao_did))?,
        request_hash,
        LinkTypes::DaoToRequests,
        (),
    )?;

    Ok(request)
}

/// Get all open requests in a DAO
#[hdk_extern]
pub fn get_dao_requests(dao_did: String) -> ExternResult<Vec<ServiceRequest>> {
    if dao_did.is_empty() || dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("DAO DID must be 1-256 characters".into())));
    }
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("requests:{}", dao_did))?,
            LinkTypes::DaoToRequests,
        )?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(record) = get(link.target.into_action_hash().ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?, GetOptions::default())? {
            if let Some(request) = record.entry().to_app_option::<ServiceRequest>().ok().flatten() {
                if request.open {
                    requests.push(request);
                }
            }
        }
    }

    Ok(requests)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn anchor_hash(anchor: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(anchor.to_string()))
}

fn find_balance(member_did: &str, dao_did: &str) -> ExternResult<Option<TendBalance>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("balance:{}:{}", dao_did, member_did))?,
            LinkTypes::MemberToBalance,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(record) = get(link.target.clone().into_action_hash().unwrap(), GetOptions::default())? {
            return Ok(record.entry().to_app_option::<TendBalance>().ok().flatten());
        }
    }

    Ok(None)
}

fn update_balance_after_exchange(
    member_did: &str,
    dao_did: &str,
    hours: f32,
    is_provider: bool,
) -> ExternResult<()> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("balance:{}:{}", dao_did, member_did))?,
            LinkTypes::MemberToBalance,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        let action_hash = link.target.clone().into_action_hash().ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;
        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            if let Some(mut balance) = record.entry().to_app_option::<TendBalance>().ok().flatten() {
                let now = sys_time()?;

                if is_provider {
                    balance.balance += hours as i32;
                    balance.total_provided += hours;
                } else {
                    balance.balance -= hours as i32;
                    balance.total_received += hours;
                }
                balance.exchange_count += 1;
                balance.last_activity = now;

                update_entry(action_hash, &balance)?;
            }
        }
    }

    Ok(())
}

/// Find an exchange by its ID using the ExchangeIdToExchange index
fn find_exchange_by_id(exchange_id: &str) -> ExternResult<Option<TendExchange>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("exchange:{}", exchange_id))?,
            LinkTypes::ExchangeIdToExchange,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record.entry().to_app_option::<TendExchange>().ok().flatten());
            }
        }
    }

    Ok(None)
}

/// Update an exchange entry by finding it via ID index and updating in place
fn update_exchange_entry(exchange_id: &str, exchange: &TendExchange) -> ExternResult<()> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("exchange:{}", exchange_id))?,
            LinkTypes::ExchangeIdToExchange,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            update_entry(action_hash, exchange)?;
            return Ok(());
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Exchange not found for update: {}",
        exchange_id
    ))))
}

/// Find a dispute case by its ID, returning both the deserialized case and the action hash
fn find_dispute_by_id(dispute_id: &str) -> ExternResult<Option<(DisputeCase, ActionHash)>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dispute:{}", dispute_id))?,
            LinkTypes::ExchangeToDispute,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(dispute_case) = record.entry().to_app_option::<DisputeCase>().ok().flatten() {
                    return Ok(Some((dispute_case, action_hash)));
                }
            }
        }
    }

    Ok(None)
}

/// Get the effective balance limit for a member
///
/// In production, this would query the membership/oracle zome to determine
/// the member's tier. For now, we check if the member has an apprentice
/// marker link. If not found, we default to the standard BALANCE_LIMIT.
fn get_effective_limit_for_member(member_did: &str) -> ExternResult<i32> {
    // Check if member has an apprentice tier marker
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("tier:apprentice:{}", member_did))?,
            LinkTypes::AnchorLinks,
        )?,
        GetStrategy::default(),
    )?;

    if !links.is_empty() {
        Ok(APPRENTICE_BALANCE_LIMIT)
    } else {
        // Default to standard limit
        // In production, dynamic limit would come from the oracle
        Ok(BALANCE_LIMIT)
    }
}

fn balance_to_info(balance: &TendBalance) -> BalanceInfo {
    BalanceInfo {
        member_did: balance.member_did.clone(),
        dao_did: balance.dao_did.clone(),
        balance: balance.balance,
        can_provide: balance.balance < BALANCE_LIMIT,
        can_receive: balance.balance > -BALANCE_LIMIT,
        total_provided: balance.total_provided,
        total_received: balance.total_received,
        exchange_count: balance.exchange_count,
    }
}

fn exchange_to_record(exchange: &TendExchange) -> ExchangeRecord {
    ExchangeRecord {
        id: exchange.id.clone(),
        provider_did: exchange.provider_did.clone(),
        receiver_did: exchange.receiver_did.clone(),
        hours: exchange.hours,
        service_description: exchange.service_description.clone(),
        service_category: exchange.service_category.clone(),
        status: exchange.status.clone(),
        timestamp: exchange.timestamp,
    }
}

// =============================================================================
// EXPORTS FOR OTHER ZOMES
// =============================================================================

/// Get TEND activity for reputation calculation (optional, max 5% weight per Commons Charter)
#[hdk_extern]
pub fn get_tend_reputation_input(input: GetBalanceInput) -> ExternResult<f32> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Member DID must be 1-256 characters".into())));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("DAO DID must be 1-256 characters".into())));
    }
    let balance = get_or_create_balance(input.member_did, input.dao_did)?;

    // Normalize based on exchange count (more exchanges = more active)
    // Cap at 50 exchanges for max score
    let activity_score = (balance.exchange_count as f32 / 50.0).min(1.0);

    // Apply max weight of 5%
    Ok(activity_score * 0.05)
}

/// Forgive a member's TEND balance on exit/death.
///
/// Sets their balance to zero. The community absorbs the micro-imbalance,
/// proven safe by 40+ years of LETS experience.
/// Returns the list of (dao_did, forgiven_amount) pairs.
#[hdk_extern]
pub fn forgive_balance(member_did: String) -> ExternResult<Vec<(String, i32)>> {
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Member DID must be 1-256 characters".into())));
    }
    if !member_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Member must be a valid DID".into())));
    }

    let mut forgiven = Vec::new();

    // Query all TendBalance entries to find ones for this member
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::TendBalance)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(balance) = record.entry().to_app_option::<TendBalance>().ok().flatten() {
            if balance.member_did == member_did && balance.balance != 0 {
                let forgiven_amount = balance.balance;
                let now = sys_time()?;
                let zeroed = TendBalance {
                    balance: 0,
                    last_activity: now,
                    ..balance.clone()
                };
                update_entry(record.action_address().clone(), &EntryTypes::TendBalance(zeroed))?;
                forgiven.push((balance.dao_did, forgiven_amount));
            }
        }
    }

    Ok(forgiven)
}
