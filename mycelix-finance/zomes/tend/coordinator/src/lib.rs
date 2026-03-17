#![deny(unsafe_code)]
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
use mycelix_finance_shared::{
    anchor_hash, follow_update_chain, pick_race_winner, rate_limit_anchor_key,
    verify_governance_or_bootstrap_from_links, verify_participant_tier,
    DEFAULT_RATE_LIMIT_PER_MINUTE, GOVERNANCE_AGENTS_ANCHOR,
};

// Re-export integrity types for external use
pub use tend_integrity::*;

// =============================================================================
// GOVERNANCE AGENT AUTHORIZATION
// =============================================================================

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

/// Verify the calling agent is a governance agent (used for cross-zome authorization).
/// Returns Ok(()) if authorized, Err if not.
#[hdk_extern]
pub fn verify_governance_agent(_: ()) -> ExternResult<()> {
    verify_governance_or_bootstrap()
}

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
    pub can_provide: bool, // Can still provide (balance < +limit)
    pub can_receive: bool, // Can still receive (balance > -limit)
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

const ORACLE_STATE_ANCHOR: &str = "tend:oracle_state";
const MAX_VITALITY: u32 = 100;

/// Update the oracle state (sets current vitality and limit tier).
///
/// Restricted to authorized governance agents (or any agent during bootstrap).
/// The oracle agent monitors network health metrics and calls this periodically.
#[hdk_extern]
pub fn update_oracle_state(vitality: u32) -> ExternResult<OracleState> {
    verify_governance_or_bootstrap()?;
    if vitality > MAX_VITALITY {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Vitality must be 0-{}",
            MAX_VITALITY
        ))));
    }

    let now = sys_time()?;
    let tier = OracleState::tier_from_vitality(vitality);
    let state = OracleState {
        vitality,
        tier,
        updated_at: now,
    };

    // Check for existing state to update
    let links = get_links(
        LinkQuery::try_new(anchor_hash(ORACLE_STATE_ANCHOR)?, LinkTypes::AnchorLinks)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(link_hash)?;
            update_entry(
                record.action_address().clone(),
                &EntryTypes::OracleState(state.clone()),
            )?;
            return Ok(state);
        }
    }

    // Create new oracle state
    let hash = create_entry(&EntryTypes::OracleState(state.clone()))?;
    create_link(
        anchor_hash(ORACLE_STATE_ANCHOR)?,
        hash,
        LinkTypes::AnchorLinks,
        (),
    )?;

    Ok(state)
}

/// Get the dynamic TEND limit based on current oracle state.
///
/// Returns the limit for the current network condition (Normal=±40, Elevated=±60, etc.).
/// Falls back to Normal (±40) if no oracle state exists.
#[hdk_extern]
pub fn get_dynamic_tend_limit(_: ()) -> ExternResult<i32> {
    Ok(read_current_tier()?.limit())
}

/// Internal: read the current tier from oracle state.
/// Follows the update chain to get the latest oracle state.
fn read_current_tier() -> ExternResult<TendLimitTier> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(ORACLE_STATE_ANCHOR)?, LinkTypes::AnchorLinks)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(action_hash)?;
            let state = record.entry().to_app_option::<OracleState>().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "OracleState deserialization error: {:?}",
                    e
                )))
            })?;
            if let Some(state) = state {
                return Ok(state.tier);
            }
        }
    }

    // Default to Normal if no oracle state
    Ok(TendLimitTier::Normal)
}

/// Get the current oracle state (vitality + tier).
/// Used by bridge coordinator for cross-cluster TEND limit queries.
#[hdk_extern]
pub fn get_oracle_state(_: ()) -> ExternResult<OracleStateResponse> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(ORACLE_STATE_ANCHOR)?, LinkTypes::AnchorLinks)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(action_hash)?;
            let state = record.entry().to_app_option::<OracleState>().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "OracleState deserialization error: {:?}",
                    e
                )))
            })?;
            if let Some(state) = state {
                return Ok(OracleStateResponse {
                    vitality: state.vitality,
                    tier_name: format!("{:?}", state.tier),
                    limit: state.tier.limit(),
                });
            }
        }
    }
    // Default: Normal state
    Ok(OracleStateResponse {
        vitality: 50,
        tier_name: "Normal".to_string(),
        limit: 40,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OracleStateResponse {
    pub vitality: u32,
    pub tier_name: String,
    pub limit: i32,
}

// =============================================================================
// HEARTH-SCOPED TEND (Phase 2)
// =============================================================================

/// Record a TEND exchange within a hearth (family unit).
///
/// Hearths use tighter credit limits (±20) and simpler governance.
/// The same zero-sum mutual credit physics apply.
#[hdk_extern]
pub fn record_hearth_exchange(input: RecordHearthExchangeInput) -> ExternResult<ExchangeRecord> {
    if !input.hours.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hours must be a finite number".into()
        )));
    }
    if input.hours <= 0.0 || input.hours > 8.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hours must be between 0 and 8".into()
        )));
    }
    if input.hours < 0.25 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Minimum service time is 15 minutes".into()
        )));
    }
    if input.hearth_did.is_empty() || !input.hearth_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hearth DID must be a valid DID".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let provider_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    if provider_did == input.receiver_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot exchange time with yourself".into()
        )));
    }

    // Verify both parties are members of this hearth via cross-zome call
    verify_hearth_membership(&provider_did, &input.hearth_did)?;
    verify_hearth_membership(&input.receiver_did, &input.hearth_did)?;

    // Check hearth balance limits (±HEARTH_TEND_CREDIT_LIMIT = ±20)
    let provider_bal =
        get_or_create_hearth_balance(provider_did.clone(), input.hearth_did.clone())?;
    let new_provider = provider_bal.balance + (input.hours.round() as i32);
    if new_provider > HEARTH_TEND_CREDIT_LIMIT {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Would exceed hearth credit limit of +{}. Current: {}",
            HEARTH_TEND_CREDIT_LIMIT, provider_bal.balance
        ))));
    }

    let receiver_bal =
        get_or_create_hearth_balance(input.receiver_did.clone(), input.hearth_did.clone())?;
    let new_receiver = receiver_bal.balance - (input.hours.round() as i32);
    if new_receiver < -HEARTH_TEND_CREDIT_LIMIT {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Receiver would exceed hearth debt limit of -{}. Current: {}",
            HEARTH_TEND_CREDIT_LIMIT, receiver_bal.balance
        ))));
    }

    // Create the exchange (uses DAO-scoped TendExchange with hearth_did as dao_did)
    let exchange_id = format!(
        "hearth-tend:{}:{}:{}",
        provider_did,
        input.receiver_did,
        now.as_micros()
    );
    let exchange = TendExchange {
        id: exchange_id.clone(),
        provider_did: provider_did.clone(),
        receiver_did: input.receiver_did.clone(),
        hours: input.hours,
        service_description: input.service_description.clone(),
        service_category: input.service_category.clone(),
        cultural_alias: input.cultural_alias,
        dao_did: input.hearth_did.clone(), // hearth acts as micro-DAO
        timestamp: now,
        status: ExchangeStatus::Confirmed, // Hearths use instant confirmation (high trust)
        service_date: None,
    };

    let hash = create_entry(&EntryTypes::TendExchange(exchange))?;
    create_link(
        anchor_hash(&format!("exchange:{}", exchange_id))?,
        hash,
        LinkTypes::ExchangeIdToExchange,
        (),
    )?;

    // Update hearth balances
    update_hearth_balance(&provider_did, &input.hearth_did, input.hours, true)?;
    update_hearth_balance(&input.receiver_did, &input.hearth_did, input.hours, false)?;

    Ok(ExchangeRecord {
        id: exchange_id,
        provider_did,
        receiver_did: input.receiver_did,
        hours: input.hours,
        service_description: input.service_description,
        service_category: input.service_category,
        status: ExchangeStatus::Confirmed,
        timestamp: now,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordHearthExchangeInput {
    pub receiver_did: String,
    pub hearth_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub cultural_alias: Option<String>,
}

/// Get a member's hearth TEND balance
#[hdk_extern]
pub fn get_hearth_balance(input: GetHearthBalanceInput) -> ExternResult<HearthTendBalance> {
    get_or_create_hearth_balance(input.member_did, input.hearth_did)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetHearthBalanceInput {
    pub member_did: String,
    pub hearth_did: String,
}

fn get_or_create_hearth_balance(
    member_did: String,
    hearth_did: String,
) -> ExternResult<HearthTendBalance> {
    let anchor_key = format!("hearth-balance:{}:{}", hearth_did, member_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::HearthToBalances)?,
        GetStrategy::default(),
    )?;
    // Pick the link with the lowest target ActionHash (deterministic winner)
    // to handle orphaned links from past race conditions (RC-14).
    if let Some(action_hash) = links
        .iter()
        .filter_map(|l| l.target.clone().into_action_hash())
        .min()
    {
        let record = follow_update_chain(action_hash)?;
        let bal = record
            .entry()
            .to_app_option::<HearthTendBalance>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "HearthTendBalance deserialization error: {:?}",
                    e
                )))
            })?;
        if let Some(bal) = bal {
            return Ok(bal);
        }
    }
    // Create new zero balance
    let now = sys_time()?;
    let bal = HearthTendBalance {
        member_did: member_did.clone(),
        hearth_did: hearth_did.clone(),
        balance: 0,
        total_provided: 0.0,
        total_received: 0.0,
        exchange_count: 0,
        last_activity: now,
    };
    let hash = create_entry(&EntryTypes::HearthTendBalance(bal.clone()))?;
    let anchor = anchor_hash(&anchor_key)?;
    let our_link_hash = create_link(
        anchor.clone(),
        hash.clone(),
        LinkTypes::HearthToBalances,
        (),
    )?;
    create_link(
        anchor_hash(&format!("my-hearths:{}", member_did))?,
        hash.clone(),
        LinkTypes::MemberToHearthBalance,
        (),
    )?;

    // RC-14: Race condition guard — re-read links to detect concurrent creators.
    let recheck_links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::HearthToBalances)?,
        GetStrategy::default(),
    )?;
    if recheck_links.len() > 1 {
        if let Some(winner_hash) = recheck_links
            .iter()
            .filter_map(|l| l.target.clone().into_action_hash())
            .min()
        {
            if winner_hash != hash {
                // We lost the race — delete our orphaned link to prevent drift
                let _ = delete_link(our_link_hash, GetOptions::default());
                // Return the winner's entry instead
                let record = follow_update_chain(winner_hash)?;
                let winner_bal = record
                    .entry()
                    .to_app_option::<HearthTendBalance>()
                    .map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "HearthTendBalance deserialization error: {:?}",
                            e
                        )))
                    })?
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Winner hearth balance entry missing".into()
                    )))?;
                return Ok(winner_bal);
            }
        }
    }

    Ok(bal)
}

/// Verify that a member belongs to a hearth via cross-zome call to hearth zome.
/// Falls through (allows) if hearth zome is unreachable (bootstrap/standalone mode).
fn verify_hearth_membership(member_did: &str, hearth_did: &str) -> ExternResult<()> {
    #[derive(Serialize, Debug)]
    struct MembershipQuery {
        member_did: String,
        hearth_did: String,
    }

    match call(
        CallTargetCell::Local,
        ZomeName::from("hearth_bridge"),
        FunctionName::from("is_hearth_member"),
        None,
        MembershipQuery {
            member_did: member_did.to_string(),
            hearth_did: hearth_did.to_string(),
        },
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            match result.decode::<bool>() {
                Ok(true) => Ok(()),
                Ok(false) => Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "{} is not a member of hearth {}",
                    member_did, hearth_did
                )))),
                Err(e) => {
                    // SECURITY NOTE: Decode error falls through permissively to support
                    // bootstrap/standalone mode where hearth zome schema may differ.
                    debug!("verify_hearth_membership: decode error for {}@{}: {:?}, allowing (bootstrap/standalone)", member_did, hearth_did, e);
                    Ok(())
                }
            }
        }
        Ok(other) => {
            // SECURITY NOTE: Hearth zome unreachable/unauthorized — allow in bootstrap/standalone mode.
            debug!("verify_hearth_membership: hearth_bridge returned {:?} for {}@{}, allowing (bootstrap/standalone)", other, member_did, hearth_did);
            Ok(())
        }
        Err(e) => {
            // SECURITY NOTE: Hearth zome unreachable — allow in bootstrap/standalone mode.
            debug!("verify_hearth_membership: hearth_bridge unreachable for {}@{}: {:?}, allowing (bootstrap/standalone)", member_did, hearth_did, e);
            Ok(())
        }
    }
}

fn update_hearth_balance(
    member_did: &str,
    hearth_did: &str,
    hours: f32,
    is_provider: bool,
) -> ExternResult<()> {
    let anchor_key = format!("hearth-balance:{}:{}", hearth_did, member_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::HearthToBalances)?,
        GetStrategy::default(),
    )?;

    // Pick the link with the lowest target ActionHash (deterministic winner)
    // to handle orphaned links from past race conditions (RC-14).
    let Some(original_hash) = links
        .iter()
        .filter_map(|l| l.target.clone().into_action_hash())
        .min()
    else {
        return Ok(());
    };

    for attempt in 0..MAX_BALANCE_RETRIES {
        let record = follow_update_chain(original_hash.clone())?;
        let mut bal = record
            .entry()
            .to_app_option::<HearthTendBalance>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "HearthTendBalance deserialization error: {:?}",
                    e
                )))
            })?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "HearthTendBalance entry missing".into()
            )))?;

        let now = sys_time()?;
        // See ZERO-SUM INVARIANT comment in update_balance_after_exchange.
        if is_provider {
            bal.balance += hours.round() as i32;
            bal.total_provided += hours;
        } else {
            bal.balance -= hours.round() as i32;
            bal.total_received += hours;
        }
        bal.exchange_count += 1;
        bal.last_activity = now;

        let expected_balance = bal.balance;
        let expected_count = bal.exchange_count;
        update_entry(record.action_address().clone(), &bal)?;

        // Verify our update won: re-read from the original link target
        let verify_record = follow_update_chain(original_hash.clone())?;
        if let Ok(Some(actual)) = verify_record.entry().to_app_option::<HearthTendBalance>() {
            if actual.balance == expected_balance && actual.exchange_count == expected_count {
                return Ok(());
            }
        }

        // Concurrent update detected — retry unless exhausted
        if attempt == MAX_BALANCE_RETRIES - 1 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Hearth balance update for {}:{} failed after {} retries due to concurrent modifications",
                hearth_did, member_did, MAX_BALANCE_RETRIES
            ))));
        }
        debug!(
            "update_hearth_balance: concurrent update detected for {}:{}, retry {}/{}",
            hearth_did,
            member_did,
            attempt + 1,
            MAX_BALANCE_RETRIES
        );
    }

    Ok(())
}

// =============================================================================
// CULTURAL ALIASES (Phase 3)
// =============================================================================

/// Register a cultural alias for a community's currency.
///
/// A farming co-op might register "Water Credits" as their TEND alias.
/// A care collective might use "Cuidado". The name is display-only —
/// the mutual credit physics are identical.
#[hdk_extern]
pub fn register_currency_alias(
    input: RegisterCurrencyAliasInput,
) -> ExternResult<CurrencyAliasEntry> {
    if input.alias_name.is_empty() || input.alias_name.len() > 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Alias name must be 1-64 characters".into()
        )));
    }
    if !input.dao_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO must be a valid DID".into()
        )));
    }

    let now = sys_time()?;
    let alias = CurrencyAliasEntry {
        dao_did: input.dao_did.clone(),
        alias_name: input.alias_name,
        base_currency: input.base_currency,
        display_symbol: input.display_symbol,
        description: input.description,
        created_at: now,
    };

    let hash = create_entry(&EntryTypes::CurrencyAliasEntry(alias.clone()))?;
    create_link(
        anchor_hash(&format!("alias:{}", input.dao_did))?,
        hash,
        LinkTypes::DaoToAlias,
        (),
    )?;

    Ok(alias)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterCurrencyAliasInput {
    pub dao_did: String,
    pub alias_name: String,
    pub base_currency: Currency,
    pub display_symbol: Option<String>,
    pub description: Option<String>,
}

/// Get the cultural alias registered for a community's currency.
#[hdk_extern]
pub fn get_currency_alias(dao_did: String) -> ExternResult<Option<CurrencyAliasEntry>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("alias:{}", dao_did))?,
            LinkTypes::DaoToAlias,
        )?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return record
                    .entry()
                    .to_app_option::<CurrencyAliasEntry>()
                    .map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "CurrencyAliasEntry deserialization error: {:?}",
                            e
                        )))
                    });
            }
        }
    }
    Ok(None)
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
    verify_participant_tier()?;
    if input.receiver_did.is_empty() || input.receiver_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Receiver DID must be 1-256 characters".into()
        )));
    }
    if !input.hours.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hours must be a finite number".into()
        )));
    }
    if input.hours <= 0.0 || input.hours > 8.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hours must be between 0 and 8 (MAX_SERVICE_HOURS)".into()
        )));
    }
    if input.hours < 0.25 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Minimum service time is 15 minutes (0.25 hours)".into()
        )));
    }
    if input.service_description.is_empty() || input.service_description.len() > 1024 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Service description must be 1-1024 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let provider_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    // Per-agent rate limit: reject if this agent has exceeded the exchange
    // recording limit within the current 60-second window.
    {
        let key = rate_limit_anchor_key("tend_exchange", &caller, now.as_micros());
        let anchor = anchor_hash(&key)?;
        let recent_links = get_links(
            LinkQuery::try_new(anchor.clone(), LinkTypes::DaoToExchanges)?,
            GetStrategy::default(),
        )?;
        if recent_links.len() >= DEFAULT_RATE_LIMIT_PER_MINUTE {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Rate limit exceeded: max {} TEND exchanges per minute",
                DEFAULT_RATE_LIMIT_PER_MINUTE
            ))));
        }
        // Record this operation in the rate-limit bucket
        create_link(
            anchor,
            AnyLinkableHash::from(caller.clone()),
            LinkTypes::DaoToExchanges,
            (),
        )?;
    }

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
    let new_provider_balance = provider_balance.balance + (input.hours.round() as i32);
    if new_provider_balance > provider_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Exchange would exceed your credit limit of +{}. Current balance: {}",
            provider_limit, provider_balance.balance
        ))));
    }

    // Check receiver's balance (can still receive if above -limit)
    let receiver_balance =
        get_or_create_balance(input.receiver_did.clone(), input.dao_did.clone())?;
    let new_receiver_balance = receiver_balance.balance - (input.hours.round() as i32);
    if new_receiver_balance < -receiver_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Receiver would exceed debt limit of -{}. Their balance: {}",
            receiver_limit, receiver_balance.balance
        ))));
    }

    // Create the exchange
    let exchange_id = format!(
        "tend:{}:{}:{}",
        provider_did,
        input.receiver_did,
        now.as_micros()
    );
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
        anchor_hash(&format!(
            "receiver:{}:{}",
            input.dao_did, input.receiver_did
        ))?,
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
    verify_participant_tier()?;
    if exchange_id.is_empty() || exchange_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange ID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the exchange
    let exchange = find_exchange_by_id(&exchange_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Exchange not found".into()
    )))?;

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

    // RC-19 fix: Create confirmation guard link FIRST, then verify we won
    // the race. This prevents duplicate balance updates from concurrent calls.
    let confirm_anchor = format!("tend-confirm:{}", exchange_id);
    let confirm_anchor_hash = anchor_hash(&confirm_anchor)?;

    // Quick pre-check: if confirmation already happened, return idempotently
    let pre_existing = get_links(
        LinkQuery::try_new(confirm_anchor_hash.clone(), LinkTypes::AnchorLinks)?,
        GetStrategy::default(),
    )?;
    if !pre_existing.is_empty() {
        // Already confirmed by a prior call — re-fetch to get current status
        let confirmed_ex = find_exchange_by_id(&exchange_id)?.ok_or(wasm_error!(
            WasmErrorInner::Guest("Exchange not found".into())
        ))?;
        return Ok(ExchangeRecord {
            id: confirmed_ex.id,
            provider_did: confirmed_ex.provider_did,
            receiver_did: confirmed_ex.receiver_did,
            hours: confirmed_ex.hours,
            service_description: confirmed_ex.service_description,
            service_category: confirmed_ex.service_category,
            status: ExchangeStatus::Confirmed,
            timestamp: confirmed_ex.timestamp,
        });
    }

    // Create the guard link BEFORE balance updates (claim our intent)
    let our_link_hash = create_link(
        confirm_anchor_hash.clone(),
        AnyLinkableHash::from(agent_info()?.agent_initial_pubkey),
        LinkTypes::AnchorLinks,
        (),
    )?;

    // Re-read links to detect concurrent confirmations (create-then-verify)
    let all_confirm_links = get_links(
        LinkQuery::try_new(confirm_anchor_hash, LinkTypes::AnchorLinks)?,
        GetStrategy::default(),
    )?;

    if all_confirm_links.len() > 1 {
        // Race detected — winner is lowest ActionHash (deterministic)
        let winner = pick_race_winner(&all_confirm_links)?;

        if winner.create_link_hash != our_link_hash {
            // We lost the race — clean up and return idempotently
            delete_link(our_link_hash, GetOptions::default())?;
            return Ok(ExchangeRecord {
                id: exchange.id,
                provider_did: exchange.provider_did,
                receiver_did: exchange.receiver_did,
                hours: exchange.hours,
                service_description: exchange.service_description,
                service_category: exchange.service_category,
                status: ExchangeStatus::Confirmed,
                timestamp: exchange.timestamp,
            });
        }
        // We won — best-effort cleanup of loser links
        for link in &all_confirm_links {
            if link.create_link_hash != our_link_hash {
                let _ = delete_link(link.create_link_hash.clone(), GetOptions::default());
            }
        }
    }

    // Only the race winner reaches here — re-check balance limits
    let provider_limit = get_effective_limit_for_member(&exchange.provider_did)?;
    let receiver_limit = get_effective_limit_for_member(&exchange.receiver_did)?;

    let provider_balance =
        get_or_create_balance(exchange.provider_did.clone(), exchange.dao_did.clone())?;
    let new_provider_balance = provider_balance.balance + (exchange.hours.round() as i32);
    if new_provider_balance > provider_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot confirm: provider would exceed credit limit of +{}. Current balance: {}",
            provider_limit, provider_balance.balance
        ))));
    }

    let receiver_balance =
        get_or_create_balance(exchange.receiver_did.clone(), exchange.dao_did.clone())?;
    let new_receiver_balance = receiver_balance.balance - (exchange.hours.round() as i32);
    if new_receiver_balance < -receiver_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot confirm: receiver would exceed debt limit of -{}. Current balance: {}",
            receiver_limit, receiver_balance.balance
        ))));
    }

    // Create a PendingBalanceAdjustment BEFORE balance updates for crash recovery.
    // If a crash occurs between the two updates, recover_pending_adjustments can
    // complete the interrupted operation and restore the zero-sum invariant.
    let now_ts = sys_time()?;
    let pending_adj = PendingBalanceAdjustment {
        exchange_id: exchange.id.clone(),
        provider_did: exchange.provider_did.clone(),
        receiver_did: exchange.receiver_did.clone(),
        hours: exchange.hours as f64,
        currency_id: exchange.dao_did.clone(),
        provider_completed: false,
        receiver_completed: false,
        created_at: now_ts,
    };
    let pending_adj_hash = create_entry(&EntryTypes::PendingBalanceAdjustment(pending_adj))?;

    // Link from a well-known anchor so recover_pending_adjustments can find them
    let pending_anchor = anchor_hash("pending-balance-adjustments")?;
    create_link(
        pending_anchor,
        pending_adj_hash.clone(),
        LinkTypes::PendingAdjustmentToExchange,
        (),
    )?;

    // Update balances — provider first
    update_balance_after_exchange(
        &exchange.provider_did,
        &exchange.dao_did,
        exchange.hours,
        true, // provider gains
    )?;

    // Mark provider side as completed
    let pending_adj_provider_done = PendingBalanceAdjustment {
        exchange_id: exchange.id.clone(),
        provider_did: exchange.provider_did.clone(),
        receiver_did: exchange.receiver_did.clone(),
        hours: exchange.hours as f64,
        currency_id: exchange.dao_did.clone(),
        provider_completed: true,
        receiver_completed: false,
        created_at: now_ts,
    };
    update_entry(
        pending_adj_hash.clone(),
        &EntryTypes::PendingBalanceAdjustment(pending_adj_provider_done),
    )?;

    // Update balances — receiver second
    update_balance_after_exchange(
        &exchange.receiver_did,
        &exchange.dao_did,
        exchange.hours,
        false, // receiver pays
    )?;

    // Mark both sides as completed
    let pending_adj_all_done = PendingBalanceAdjustment {
        exchange_id: exchange.id.clone(),
        provider_did: exchange.provider_did.clone(),
        receiver_did: exchange.receiver_did.clone(),
        hours: exchange.hours as f64,
        currency_id: exchange.dao_did.clone(),
        provider_completed: true,
        receiver_completed: true,
        created_at: now_ts,
    };
    update_entry(
        pending_adj_hash,
        &EntryTypes::PendingBalanceAdjustment(pending_adj_all_done),
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange ID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    let exchange = find_exchange_by_id(&exchange_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Exchange not found".into()
    )))?;

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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange ID must be 1-256 characters".into()
        )));
    }
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    let exchange = find_exchange_by_id(&exchange_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Exchange not found".into()
    )))?;

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
    let exchange = find_exchange_by_id(&input.exchange_id)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Exchange not found".into())
    ))?;

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

    // RC-19 fix: Create-then-verify pattern to prevent duplicate ratings
    // from concurrent calls. We create the rating link first, then check if
    // we won the race.
    let rating_anchor = format!("rating:{}", input.exchange_id);
    let rating_anchor_hash = anchor_hash(&rating_anchor)?;

    // Quick pre-check: if rating already exists, reject
    let pre_existing = get_links(
        LinkQuery::try_new(rating_anchor_hash.clone(), LinkTypes::ExchangeToRating)?,
        GetStrategy::default(),
    )?;
    if !pre_existing.is_empty() {
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

    // Link from exchange to rating — create BEFORE verifying uniqueness
    let our_link_hash = create_link(
        rating_anchor_hash.clone(),
        rating_hash.clone(),
        LinkTypes::ExchangeToRating,
        (),
    )?;

    // Re-read to detect concurrent rating creation
    let all_rating_links = get_links(
        LinkQuery::try_new(rating_anchor_hash, LinkTypes::ExchangeToRating)?,
        GetStrategy::default(),
    )?;

    if all_rating_links.len() > 1 {
        // Race detected — winner is lowest ActionHash (deterministic)
        let winner = pick_race_winner(&all_rating_links)?;

        if winner.create_link_hash != our_link_hash {
            // We lost the race — clean up and error
            delete_link(our_link_hash, GetOptions::default())?;
            return Err(wasm_error!(WasmErrorInner::Guest(
                "This exchange has already been rated".into()
            )));
        }
        // We won — best-effort cleanup of loser links
        for link in &all_rating_links {
            if link.create_link_hash != our_link_hash {
                let _ = delete_link(link.create_link_hash.clone(), GetOptions::default());
            }
        }
    }

    // Link from rated member (provider) to rating for aggregation
    create_link(
        anchor_hash(&format!("ratings_for:{}", exchange.provider_did))?,
        rating_hash.clone(),
        LinkTypes::ExchangeToRating,
        (),
    )?;

    // Return the Record
    let record = get(rating_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Failed to retrieve created rating record".into())
    ))?;

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
    let exchange = find_exchange_by_id(&input.exchange_id)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Exchange not found".into())
    ))?;

    // Caller must be a participant
    if exchange.provider_did != caller_did && exchange.receiver_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only exchange participants can open a dispute".into()
        )));
    }

    // RC-19 fix: Create-then-verify pattern to prevent duplicate disputes
    // from concurrent calls. We create the dispute link first, then check if
    // we won the race.
    let dispute_anchor = format!("dispute_for_exchange:{}", input.exchange_id);
    let dispute_anchor_hash = anchor_hash(&dispute_anchor)?;

    // Quick pre-check: if dispute already exists, reject
    let pre_existing = get_links(
        LinkQuery::try_new(dispute_anchor_hash.clone(), LinkTypes::ExchangeToDispute)?,
        GetStrategy::default(),
    )?;
    if !pre_existing.is_empty() {
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

    // Link from exchange to dispute — create BEFORE verifying uniqueness
    let our_link_hash = create_link(
        dispute_anchor_hash.clone(),
        dispute_hash.clone(),
        LinkTypes::ExchangeToDispute,
        (),
    )?;

    // Re-read to detect concurrent dispute creation
    let all_dispute_links = get_links(
        LinkQuery::try_new(dispute_anchor_hash, LinkTypes::ExchangeToDispute)?,
        GetStrategy::default(),
    )?;

    if all_dispute_links.len() > 1 {
        // Race detected — winner is lowest ActionHash (deterministic)
        let winner = pick_race_winner(&all_dispute_links)?;

        if winner.create_link_hash != our_link_hash {
            // We lost the race — clean up and error
            delete_link(our_link_hash, GetOptions::default())?;
            return Err(wasm_error!(WasmErrorInner::Guest(
                "A dispute already exists for this exchange".into()
            )));
        }
        // We won — best-effort cleanup of loser links
        for link in &all_dispute_links {
            if link.create_link_hash != our_link_hash {
                let _ = delete_link(link.create_link_hash.clone(), GetOptions::default());
            }
        }
    }

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
    let record = get(dispute_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Failed to retrieve created dispute record".into())
    ))?;

    Ok(record)
}

/// Minimum MYCEL score required to serve as a mediator
const MEDIATOR_MIN_MYCEL: f64 = 0.5;
/// Number of mediators assigned per dispute panel
const MEDIATOR_PANEL_SIZE: usize = 3;

#[derive(Serialize, Deserialize, Debug)]
pub struct EscalateDisputeInput {
    pub dispute_id: String,
    /// Candidate mediator DIDs. Required when escalating to MediationPanel.
    /// Must provide at least 3 candidates with MYCEL > 0.5.
    /// 3 are selected (first 3 that pass validation).
    pub candidate_mediators: Option<Vec<String>>,
}

/// Escalate a dispute to the next resolution stage
///
/// Escalation path:
///   DirectNegotiation -> MediationPanel -> GovernanceVote
///
/// When escalating to MediationPanel, candidate_mediators must be provided.
/// Each candidate is validated via cross-zome call to recognition to confirm
/// MYCEL > 0.5. The first 3 valid candidates are assigned as mediators.
///
/// Only dispute participants can escalate. A dispute that is already
/// at GovernanceVote cannot be escalated further.
#[hdk_extern]
pub fn escalate_dispute(input: EscalateDisputeInput) -> ExternResult<Record> {
    let dispute_id = &input.dispute_id;
    if dispute_id.is_empty() || dispute_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Dispute ID must be 1-256 characters".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Find the dispute
    let (dispute_case, action_hash) = find_dispute_by_id(dispute_id)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Dispute not found".into())
    ))?;

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

    // If escalating to MediationPanel, validate and assign mediators
    let mediator_dids = if next_stage == DisputeStage::MediationPanel {
        let candidates = input
            .candidate_mediators
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "candidate_mediators required when escalating to MediationPanel".into()
            )))?;

        let mut valid_mediators = Vec::new();
        for candidate in &candidates {
            // Exclude dispute parties
            if *candidate == dispute_case.complainant_did
                || *candidate == dispute_case.respondent_did
            {
                continue;
            }
            // Validate MYCEL score via cross-zome call
            if check_mediator_eligible(candidate)? {
                valid_mediators.push(candidate.clone());
                if valid_mediators.len() >= MEDIATOR_PANEL_SIZE {
                    break;
                }
            }
        }

        if valid_mediators.len() < MEDIATOR_PANEL_SIZE {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Need {} eligible mediators (MYCEL > {}), only found {}",
                MEDIATOR_PANEL_SIZE,
                MEDIATOR_MIN_MYCEL,
                valid_mediators.len()
            ))));
        }

        valid_mediators
    } else {
        dispute_case.mediator_dids.clone()
    };

    let now = sys_time()?;
    let updated_dispute = DisputeCase {
        stage: next_stage,
        escalated_at: Some(now),
        mediator_dids,
        ..dispute_case
    };

    // Update the entry in place
    update_entry(action_hash, &updated_dispute)?;

    // Re-fetch the updated record via follow_update_chain
    // (the link points to the original create action, not the latest update)
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dispute:{}", dispute_id))?,
            LinkTypes::ExchangeToDispute,
        )?,
        GetStrategy::default(),
    )?;

    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Dispute link not found".into()
    )))?;
    let hash = link
        .target
        .clone()
        .into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid dispute link target".into()
        )))?;

    let record = follow_update_chain(hash)?;

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
    let (dispute_case, action_hash) = find_dispute_by_id(&input.dispute_id)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Dispute not found".into())
    ))?;

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

    // Re-fetch the updated record via follow_update_chain
    // (the link points to the original create action, not the latest update)
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dispute:{}", input.dispute_id))?,
            LinkTypes::ExchangeToDispute,
        )?,
        GetStrategy::default(),
    )?;

    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Dispute link not found".into()
    )))?;
    let hash = link
        .target
        .clone()
        .into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid dispute link target".into()
        )))?;

    let record = follow_update_chain(hash)?;

    Ok(record)
}

// =============================================================================
// BALANCE FUNCTIONS
// =============================================================================

/// Get or create balance for a member in a DAO (internal function)
fn get_or_create_balance(member_did: String, dao_did: String) -> ExternResult<BalanceInfo> {
    // Try to find existing balance
    if let Some(balance) = find_balance(&member_did, &dao_did)? {
        let limit = get_effective_limit_for_member(&member_did)?;
        return Ok(balance_to_info(&balance, limit));
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

    let anchor = anchor_hash(&format!("balance:{}:{}", dao_did, member_did))?;
    let our_link_hash = create_link(
        anchor.clone(),
        action_hash.clone(),
        LinkTypes::MemberToBalance,
        (),
    )?;

    // RC-13: Race condition guard — re-read links to detect concurrent creators.
    // If multiple links exist, deterministically pick the one with the lowest
    // target ActionHash. If we're not the winner, delete our orphaned link and
    // return the winner's entry.
    let recheck_links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::MemberToBalance)?,
        GetStrategy::default(),
    )?;
    if recheck_links.len() > 1 {
        if let Some(winner_hash) = recheck_links
            .iter()
            .filter_map(|l| l.target.clone().into_action_hash())
            .min()
        {
            if winner_hash != action_hash {
                // We lost the race — delete our orphaned link to prevent drift
                let _ = delete_link(our_link_hash, GetOptions::default());
                // Return the winner's entry instead
                let record = follow_update_chain(winner_hash)?;
                let winner_bal = record
                    .entry()
                    .to_app_option::<TendBalance>()
                    .map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "TendBalance deserialization error: {:?}",
                            e
                        )))
                    })?
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Winner balance entry missing".into()
                    )))?;
                let limit = get_effective_limit_for_member(&member_did)?;
                return Ok(balance_to_info(&winner_bal, limit));
            }
        }
    }

    let limit = get_effective_limit_for_member(&member_did)?;
    Ok(balance_to_info(&balance, limit))
}

/// Get balance info for a member
#[hdk_extern]
pub fn get_balance(input: GetBalanceInput) -> ExternResult<BalanceInfo> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    get_or_create_balance(input.member_did, input.dao_did)
}

/// Get all exchanges for a member in a DAO (paginated, default limit 100)
#[hdk_extern]
pub fn get_my_exchanges(input: PaginatedDaoInput) -> ExternResult<Vec<ExchangeRecord>> {
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let limit = input.limit.unwrap_or(100);
    let caller = agent_info()?.agent_initial_pubkey;
    let member_did = format!("did:mycelix:{}", caller);

    let mut exchanges = Vec::new();

    // Get exchanges where member was provider
    let provider_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("provider:{}:{}", input.dao_did, member_did))?,
            LinkTypes::ProviderToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    for link in provider_links {
        let action_hash = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;
        let record = follow_update_chain(action_hash)?;
        if let Some(exchange) = record
            .entry()
            .to_app_option::<TendExchange>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "TendExchange deserialization error: {:?}",
                    e
                )))
            })?
        {
            exchanges.push(exchange_to_record(&exchange));
        }
    }

    // Get exchanges where member was receiver
    let receiver_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("receiver:{}:{}", input.dao_did, member_did))?,
            LinkTypes::ReceiverToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    for link in receiver_links {
        let action_hash = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;
        let record = follow_update_chain(action_hash)?;
        if let Some(exchange) = record
            .entry()
            .to_app_option::<TendExchange>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "TendExchange deserialization error: {:?}",
                    e
                )))
            })?
        {
            // Avoid duplicates (shouldn't happen, but safety)
            if !exchanges.iter().any(|e| e.id == exchange.id) {
                exchanges.push(exchange_to_record(&exchange));
            }
        }
    }

    // Sort by timestamp (newest first)
    exchanges.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    exchanges.truncate(limit);

    Ok(exchanges)
}

// =============================================================================
// SERVICE MARKETPLACE FUNCTIONS
// =============================================================================

/// Create a service listing (offer to help)
#[hdk_extern]
pub fn create_listing(input: CreateListingInput) -> ExternResult<ServiceListing> {
    verify_participant_tier()?;
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be under 4096 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    if let Some(hours) = input.estimated_hours {
        if !hours.is_finite() || hours <= 0.0 || hours > 168.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Estimated hours must be a finite number between 0 and 168".into()
            )));
        }
    }
    if let Some(ref avail) = input.availability {
        if avail.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Availability must be under 256 characters".into()
            )));
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

/// Input for paginated DAO queries
#[derive(Serialize, Deserialize, Debug)]
pub struct PaginatedDaoInput {
    pub dao_did: String,
    pub limit: Option<usize>,
}

/// Get all active listings in a DAO (paginated, default limit 100)
#[hdk_extern]
pub fn get_dao_listings(input: PaginatedDaoInput) -> ExternResult<Vec<ServiceListing>> {
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let limit = input.limit.unwrap_or(100);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("listings:{}", input.dao_did))?,
            LinkTypes::DaoToListings,
        )?,
        GetStrategy::default(),
    )?;

    let mut listings = Vec::new();
    for link in links {
        let action_hash = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;
        let record = follow_update_chain(action_hash)?;
        if let Some(listing) = record
            .entry()
            .to_app_option::<ServiceListing>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "ServiceListing deserialization error: {:?}",
                    e
                )))
            })?
        {
            if listing.active {
                listings.push(listing);
                if listings.len() >= limit {
                    break;
                }
            }
        }
    }

    Ok(listings)
}

/// Create a service request (ask for help)
#[hdk_extern]
pub fn create_request(input: CreateRequestInput) -> ExternResult<ServiceRequest> {
    verify_participant_tier()?;
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if input.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be under 4096 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    if let Some(hours) = input.estimated_hours {
        if !hours.is_finite() || hours <= 0.0 || hours > 168.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Estimated hours must be a finite number between 0 and 168".into()
            )));
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

/// Get all open requests in a DAO (paginated, default limit 100)
#[hdk_extern]
pub fn get_dao_requests(input: PaginatedDaoInput) -> ExternResult<Vec<ServiceRequest>> {
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
    }
    let limit = input.limit.unwrap_or(100);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("requests:{}", input.dao_did))?,
            LinkTypes::DaoToRequests,
        )?,
        GetStrategy::default(),
    )?;

    let mut requests = Vec::new();
    for link in links {
        let action_hash = link
            .target
            .into_action_hash()
            .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string())))?;
        let record = follow_update_chain(action_hash)?;
        if let Some(request) = record
            .entry()
            .to_app_option::<ServiceRequest>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "ServiceRequest deserialization error: {:?}",
                    e
                )))
            })?
        {
            if request.open {
                requests.push(request);
                if requests.len() >= limit {
                    break;
                }
            }
        }
    }

    Ok(requests)
}

// =============================================================================
// VALIDATION SCORE (for MYCEL component)
// =============================================================================

/// Input for paginated validation score queries
#[derive(Serialize, Deserialize, Debug)]
pub struct GetValidationScoreInput {
    pub member_did: String,
    /// Maximum number of ratings to consider (default 100)
    pub limit: Option<usize>,
}

/// Get aggregate validation score for a member based on their quality ratings.
///
/// Returns a score between 0.0 and 1.0 derived from the average of all
/// QualityRating entries linked to this member. This is called cross-zome
/// by the recognition coordinator to compute the MYCEL Validation component.
///
/// Scoring: average rating (1-5) mapped to 0.0-1.0 -> (avg - 1) / 4
/// Minimum 3 ratings required for a non-zero score (prevents gaming with
/// a single 5-star rating).
/// Paginated: default limit 100 ratings considered.
#[hdk_extern]
pub fn get_validation_score(input: GetValidationScoreInput) -> ExternResult<f64> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }

    let limit = input.limit.unwrap_or(100);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("ratings_for:{}", input.member_did))?,
            LinkTypes::ExchangeToRating,
        )?,
        GetStrategy::default(),
    )?;

    let mut total: f64 = 0.0;
    let mut count: u32 = 0;

    for link in links.into_iter().take(limit) {
        if let Some(action_hash) = link.target.into_action_hash() {
            // QualityRatings are immutable — plain get() is safe here
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(rating) =
                    record
                        .entry()
                        .to_app_option::<QualityRating>()
                        .map_err(|e| {
                            wasm_error!(WasmErrorInner::Guest(format!(
                                "QualityRating deserialization error: {:?}",
                                e
                            )))
                        })?
                {
                    total += rating.rating as f64;
                    count += 1;
                }
            }
        }
    }

    // Minimum 3 ratings for a non-zero score
    if count < 3 {
        return Ok(0.0);
    }

    let avg = total / count as f64;
    // Map 1-5 to 0.0-1.0
    let score = ((avg - 1.0) / 4.0).clamp(0.0, 1.0);
    Ok(score)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn find_balance(member_did: &str, dao_did: &str) -> ExternResult<Option<TendBalance>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("balance:{}:{}", dao_did, member_did))?,
            LinkTypes::MemberToBalance,
        )?,
        GetStrategy::default(),
    )?;

    // Pick the link with the lowest target ActionHash (deterministic winner)
    // to handle orphaned links from past race conditions (RC-13).
    if let Some(action_hash) = links
        .iter()
        .filter_map(|l| l.target.clone().into_action_hash())
        .min()
    {
        let record = follow_update_chain(action_hash)?;
        return record.entry().to_app_option::<TendBalance>().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "TendBalance deserialization error: {:?}",
                e
            )))
        });
    }

    Ok(None)
}

/// Maximum retries for optimistic-locking balance mutations.
const MAX_BALANCE_RETRIES: usize = 3;

fn update_balance_after_exchange(
    member_did: &str,
    dao_did: &str,
    hours: f32,
    is_provider: bool,
) -> ExternResult<()> {
    let anchor_key = format!("balance:{}:{}", dao_did, member_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::MemberToBalance)?,
        GetStrategy::default(),
    )?;

    // Pick the link with the lowest target ActionHash (deterministic winner)
    // to handle orphaned links from past race conditions (RC-13).
    let Some(original_hash) = links
        .iter()
        .filter_map(|l| l.target.clone().into_action_hash())
        .min()
    else {
        return Ok(());
    };

    for attempt in 0..MAX_BALANCE_RETRIES {
        let record = follow_update_chain(original_hash.clone())?;
        if let Some(mut balance) = record.entry().to_app_option::<TendBalance>().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "TendBalance deserialization error: {:?}",
                e
            )))
        })? {
            let now = sys_time()?;

            // ZERO-SUM INVARIANT: Both provider (+) and receiver (-) apply
            // the same `hours.round() as i32`, so the pair always nets to zero.
            // However, f32 has ~7 decimal digits of precision. A client sending
            // e.g. 2.5000001 would round to 3 instead of the intended 2 or 3,
            // and `total_provided`/`total_received` (f32 accumulators) will
            // drift from the integer `balance` over many exchanges. If this
            // becomes problematic, migrate hours to fixed-point (e.g. i32
            // quarter-hours) to eliminate rounding entirely.
            if is_provider {
                balance.balance += hours.round() as i32;
                balance.total_provided += hours;
            } else {
                balance.balance -= hours.round() as i32;
                balance.total_received += hours;
            }
            balance.exchange_count += 1;
            balance.last_activity = now;

            let expected_balance = balance.balance;
            let expected_count = balance.exchange_count;
            update_entry(record.action_address().clone(), &balance)?;

            // Verify our update won: re-read from the original link target
            let verify_record = follow_update_chain(original_hash.clone())?;
            if let Ok(Some(actual)) = verify_record.entry().to_app_option::<TendBalance>() {
                if actual.balance == expected_balance && actual.exchange_count == expected_count {
                    return Ok(());
                }
            }

            // Concurrent update detected — retry unless exhausted
            if attempt == MAX_BALANCE_RETRIES - 1 {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Balance update for {}:{} failed after {} retries due to concurrent modifications",
                    dao_did, member_did, MAX_BALANCE_RETRIES
                ))));
            }
            debug!(
                "update_balance_after_exchange: concurrent update detected for {}:{}, retry {}/{}",
                dao_did,
                member_did,
                attempt + 1,
                MAX_BALANCE_RETRIES
            );
        } else {
            return Ok(());
        }
    }

    Ok(())
}

/// Find an exchange by its ID using the ExchangeIdToExchange index.
/// Follows the update chain to get the latest version (exchanges are mutable — status changes).
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
            let record = follow_update_chain(action_hash)?;
            return record.entry().to_app_option::<TendExchange>().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "TendExchange deserialization error: {:?}",
                    e
                )))
            });
        }
    }

    Ok(None)
}

/// Update an exchange entry by finding it via ID index and updating in place.
/// Follows the update chain to ensure we update the latest version (prevents fork).
fn update_exchange_entry(exchange_id: &str, exchange: &TendExchange) -> ExternResult<()> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("exchange:{}", exchange_id))?,
            LinkTypes::ExchangeIdToExchange,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(link_hash)?;
            update_entry(record.action_address().clone(), exchange)?;
            return Ok(());
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Exchange not found for update: {}",
        exchange_id
    ))))
}

/// Check if a candidate mediator meets the MYCEL threshold (> 0.5)
/// via cross-zome call to the recognition zome.
fn check_mediator_eligible(mediator_did: &str) -> ExternResult<bool> {
    #[derive(Debug, serde::Deserialize)]
    struct MycelState {
        mycel_score: f64,
    }

    match call(
        CallTargetCell::Local,
        ZomeName::from("recognition"),
        FunctionName::from("get_mycel_score"),
        None,
        mediator_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => match result.decode::<MycelState>() {
            Ok(state) => Ok(state.mycel_score > MEDIATOR_MIN_MYCEL),
            Err(_) => Ok(false),
        },
        _ => Ok(false), // Recognition zome unreachable → ineligible
    }
}

/// Find a dispute case by its ID, returning both the deserialized case and the action hash.
/// Follows the update chain to get the latest version.
fn find_dispute_by_id(dispute_id: &str) -> ExternResult<Option<(DisputeCase, ActionHash)>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dispute:{}", dispute_id))?,
            LinkTypes::ExchangeToDispute,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(link_hash)?;
            let dispute_case = record.entry().to_app_option::<DisputeCase>().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "DisputeCase deserialization error: {:?}",
                    e
                )))
            })?;
            if let Some(dispute_case) = dispute_case {
                return Ok(Some((dispute_case, record.action_address().clone())));
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
        // Apprentice limit is always ±10 regardless of oracle state
        Ok(APPRENTICE_BALANCE_LIMIT)
    } else {
        // Dynamic limit from oracle state (counter-cyclical TEND expansion)
        Ok(read_current_tier()?.limit())
    }
}

fn balance_to_info(balance: &TendBalance, effective_limit: i32) -> BalanceInfo {
    BalanceInfo {
        member_did: balance.member_did.clone(),
        dao_did: balance.dao_did.clone(),
        balance: balance.balance,
        can_provide: balance.balance < effective_limit,
        can_receive: balance.balance > -effective_limit,
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
// CROSS-COMMUNITY TEND CLEARING
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordCrossDAOExchangeInput {
    /// DID of the provider's DAO
    pub provider_dao_did: String,
    /// DID of the receiver's DAO
    pub receiver_dao_did: String,
    /// Hours exchanged
    pub hours: f32,
}

/// Record an inter-DAO TEND exchange by updating the bilateral balance.
///
/// Provider's DAO gains credit, receiver's DAO gains debt.
/// DAOs are stored in canonical alphabetical order.
///
/// Restricted to authorized governance agents (or any agent during bootstrap).
#[hdk_extern]
pub fn record_cross_dao_exchange(
    input: RecordCrossDAOExchangeInput,
) -> ExternResult<BilateralBalance> {
    verify_governance_or_bootstrap()?;
    if input.provider_dao_did == input.receiver_dao_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cross-DAO exchange requires two different DAOs".into()
        )));
    }
    if !input.hours.is_finite() || input.hours <= 0.0 || input.hours > 168.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hours must be a finite number between 0 and 168".into()
        )));
    }

    // Canonical ordering: dao_a < dao_b alphabetically
    let (dao_a, dao_b, direction) = if input.provider_dao_did < input.receiver_dao_did {
        (&input.provider_dao_did, &input.receiver_dao_did, 1) // A provided → positive net
    } else {
        (&input.receiver_dao_did, &input.provider_dao_did, -1) // B provided → negative net
    };

    let anchor_key = format!("bilateral:{}:{}", dao_a, dao_b);
    let now = sys_time()?;
    let delta = (input.hours.round() as i32) * direction;

    // Try to find existing bilateral balance
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::DaoToBilateralBalance)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(link_hash)?;
            if let Some(mut bal) =
                record
                    .entry()
                    .to_app_option::<BilateralBalance>()
                    .map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "BilateralBalance deserialization error: {:?}",
                            e
                        )))
                    })?
            {
                bal.net_balance += delta;
                bal.total_exchanges += 1;
                bal.last_updated_at = now;
                update_entry(record.action_address().clone(), &bal)?;
                return Ok(bal);
            }
        }
    }

    // Create new bilateral balance
    let bal = BilateralBalance {
        dao_a_did: dao_a.clone(),
        dao_b_did: dao_b.clone(),
        net_balance: delta,
        total_exchanges: 1,
        last_settled_at: now,
        last_updated_at: now,
    };

    let hash = create_entry(&EntryTypes::BilateralBalance(bal.clone()))?;
    create_link(
        anchor_hash(&anchor_key)?,
        hash,
        LinkTypes::DaoToBilateralBalance,
        (),
    )?;

    Ok(bal)
}

/// Get the bilateral balance between two DAOs.
#[hdk_extern]
pub fn get_bilateral_balance(input: GetBilateralInput) -> ExternResult<Option<BilateralBalance>> {
    let (dao_a, dao_b) = if input.dao_a_did < input.dao_b_did {
        (&input.dao_a_did, &input.dao_b_did)
    } else {
        (&input.dao_b_did, &input.dao_a_did)
    };

    let anchor_key = format!("bilateral:{}:{}", dao_a, dao_b);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::DaoToBilateralBalance)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = follow_update_chain(action_hash)?;
            return record
                .entry()
                .to_app_option::<BilateralBalance>()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "BilateralBalance deserialization error: {:?}",
                        e
                    )))
                });
        }
    }
    Ok(None)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetBilateralInput {
    pub dao_a_did: String,
    pub dao_b_did: String,
}

/// Payload for cross-zome treasury SAP transfer call.
///
/// Sent to the treasury zome's `transfer_commons_sap` function to move
/// SAP from one DAO's commons pool to another during bilateral settlement.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransferPayload {
    pub from_dao: String,
    pub to_dao: String,
    pub amount: i32,
    pub reason: String,
}

/// Input for settle_bilateral_balance
#[derive(Serialize, Deserialize, Debug)]
pub struct SettleBilateralInput {
    pub dao_a_did: String,
    pub dao_b_did: String,
}

/// Settle a bilateral balance using a two-phase commit pattern.
///
/// Phase 1 (Prepare): Create a BilateralSettlement entry with status Pending,
/// recording the amount to settle. The bilateral balance is NOT zeroed yet.
///
/// Phase 2 (Commit): Call treasury to transfer SAP from the debtor DAO's
/// commons pool to the creditor DAO's commons pool.
/// - If the transfer SUCCEEDS: update settlement to Completed, zero the
///   bilateral balance.
/// - If the transfer FAILS: update settlement to Failed, the bilateral
///   balance remains unchanged (no debt is lost).
///
/// Returns the settlement Record on success.
///
/// NOTE: Governance operation -- should be called quarterly by an authorized agent.
/// Restricted to authorized governance agents (or any agent during bootstrap).
#[hdk_extern]
pub fn settle_bilateral_balance(input: SettleBilateralInput) -> ExternResult<Record> {
    verify_governance_or_bootstrap()?;
    let (dao_a, dao_b) = if input.dao_a_did < input.dao_b_did {
        (input.dao_a_did.clone(), input.dao_b_did.clone())
    } else {
        (input.dao_b_did.clone(), input.dao_a_did.clone())
    };

    let anchor_key = format!("bilateral:{}:{}", dao_a, dao_b);
    let now = sys_time()?;

    // Step 1: Find the bilateral balance
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::DaoToBilateralBalance)?,
        GetStrategy::default(),
    )?;

    let (balance_action_hash, bal) =
        {
            let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
                "No bilateral balance found between these DAOs".into()
            )))?;
            let link_hash = link.target.clone().into_action_hash().ok_or(wasm_error!(
                WasmErrorInner::Guest("Invalid bilateral balance link target".into())
            ))?;
            let record = follow_update_chain(link_hash)?;
            let bal = record
                .entry()
                .to_app_option::<BilateralBalance>()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "BilateralBalance deserialization error: {:?}",
                        e
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Bilateral balance entry missing".into()
                )))?;
            (record.action_address().clone(), bal)
        };

    if bal.net_balance == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Bilateral balance is already zero, nothing to settle".into()
        )));
    }

    // Determine debtor and creditor based on net_balance sign
    // Positive net_balance means DAO-A is owed by DAO-B (DAO-B is debtor)
    // Negative net_balance means DAO-B is owed by DAO-A (DAO-A is debtor)
    let (debtor_dao, creditor_dao) = if bal.net_balance > 0 {
        (dao_b.clone(), dao_a.clone())
    } else {
        (dao_a.clone(), dao_b.clone())
    };
    let settle_amount = bal.net_balance.abs();

    // Step 2: Phase 1 (Prepare) -- create BilateralSettlement with status Pending
    let settlement_id = format!("settlement:{}:{}:{}", dao_a, dao_b, now.as_micros());
    let settlement = BilateralSettlement {
        id: settlement_id.clone(),
        debtor_dao_did: debtor_dao.clone(),
        creditor_dao_did: creditor_dao.clone(),
        amount: settle_amount,
        status: SettlementStatus::Pending,
        created_at: now,
        completed_at: None,
    };

    let settlement_hash = create_entry(&EntryTypes::BilateralSettlement(settlement.clone()))?;

    // Link from settlement registry anchor
    create_link(
        anchor_hash(&format!("settlements:{}:{}", dao_a, dao_b))?,
        settlement_hash.clone(),
        LinkTypes::SettlementRegistry,
        (),
    )?;

    // Step 3: Phase 2 (Commit) -- attempt treasury SAP transfer
    let transfer_payload = TransferPayload {
        from_dao: debtor_dao.clone(),
        to_dao: creditor_dao.clone(),
        amount: settle_amount,
        reason: format!(
            "Bilateral TEND settlement: {} TEND-hours from {} to {}",
            settle_amount, debtor_dao, creditor_dao
        ),
    };

    let treasury_result = call(
        CallTargetCell::Local,
        ZomeName::from("treasury"),
        FunctionName::from("transfer_commons_sap"),
        None,
        transfer_payload,
    );

    let transfer_succeeded = matches!(treasury_result, Ok(ZomeCallResponse::Ok(_)));

    if transfer_succeeded {
        // Step 4a: Transfer succeeded -- update settlement to Completed, zero bilateral balance
        let completed_at = sys_time()?;
        let completed_settlement = BilateralSettlement {
            status: SettlementStatus::Completed,
            completed_at: Some(completed_at),
            ..settlement
        };
        update_entry(settlement_hash.clone(), &completed_settlement)?;

        // Zero the bilateral balance
        let zeroed = BilateralBalance {
            net_balance: 0,
            last_settled_at: completed_at,
            last_updated_at: completed_at,
            ..bal
        };
        update_entry(balance_action_hash, &zeroed)?;
    } else {
        // Step 4b: Transfer failed -- update settlement to Failed, balance unchanged
        let failed_at = sys_time()?;
        let failed_settlement = BilateralSettlement {
            status: SettlementStatus::Failed,
            completed_at: Some(failed_at),
            ..settlement
        };
        update_entry(settlement_hash.clone(), &failed_settlement)?;

        return Err(wasm_error!(WasmErrorInner::Guest(
            "Treasury SAP transfer failed. Settlement marked as Failed. \
             Bilateral balance remains unchanged -- no debt was lost."
                .into()
        )));
    }

    // Step 5: Return the settlement record
    let record = get(settlement_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Failed to retrieve settlement record".into())
    ))?;

    Ok(record)
}

// =============================================================================
// EXPORTS FOR OTHER ZOMES
// =============================================================================

/// Get TEND activity for reputation calculation (optional, max 5% weight per Commons Charter)
#[hdk_extern]
pub fn get_tend_reputation_input(input: GetBalanceInput) -> ExternResult<f32> {
    if input.member_did.is_empty() || input.member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    if input.dao_did.is_empty() || input.dao_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO DID must be 1-256 characters".into()
        )));
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
///
/// Restricted to authorized governance agents (or any agent during bootstrap).
#[hdk_extern]
pub fn forgive_balance(member_did: String) -> ExternResult<Vec<(String, i32)>> {
    verify_governance_or_bootstrap()?;
    if member_did.is_empty() || member_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member DID must be 1-256 characters".into()
        )));
    }
    if !member_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Member must be a valid DID".into()
        )));
    }

    let mut forgiven = Vec::new();

    // NOTE: query() only searches the CALLING AGENT's source chain.
    // TendBalance entries are authored by the agent who triggered the exchange
    // (the receiver in confirm_exchange), NOT the governance agent calling this.
    // This means query() will only find balances if the governance agent also
    // happened to participate in exchanges. For a proper implementation, we would
    // need a "member → DAO memberships" link index and use find_balance() per DAO.
    //
    // For now, this works when the same agent that created the balances also
    // calls forgive_balance (e.g., in single-agent test scenarios or when the
    // member themselves initiates exit via the payments zome).
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::TendBalance,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(balance) = record.entry().to_app_option::<TendBalance>().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "TendBalance deserialization error: {:?}",
                e
            )))
        })? {
            if balance.member_did == member_did && balance.balance != 0 {
                let forgiven_amount = balance.balance;
                let now = sys_time()?;
                let zeroed = TendBalance {
                    balance: 0,
                    last_activity: now,
                    ..balance.clone()
                };
                update_entry(
                    record.action_address().clone(),
                    &EntryTypes::TendBalance(zeroed),
                )?;
                forgiven.push((balance.dao_did, forgiven_amount));
            }
        }
    }

    Ok(forgiven)
}

// =============================================================================
// PENDING BALANCE ADJUSTMENT RECOVERY
// =============================================================================

/// Recover incomplete balance adjustments from interrupted confirm_exchange calls.
///
/// When confirm_exchange crashes between the provider and receiver balance updates,
/// the zero-sum invariant is broken. This function finds all PendingBalanceAdjustment
/// entries that are not fully completed and retries the missing balance updates.
///
/// Only callable by a governance agent (or during bootstrap when no agents exist).
///
/// Returns the number of adjustments that were recovered.
#[hdk_extern]
pub fn recover_pending_adjustments(currency_id: String) -> ExternResult<u32> {
    verify_governance_or_bootstrap()?;

    if currency_id.is_empty() || currency_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency ID must be 1-256 characters".into()
        )));
    }

    let pending_anchor = anchor_hash("pending-balance-adjustments")?;
    let links = get_links(
        LinkQuery::try_new(pending_anchor, LinkTypes::PendingAdjustmentToExchange)?,
        GetStrategy::default(),
    )?;

    let mut recovered: u32 = 0;

    for link in links {
        let Some(target_hash) = link.target.into_action_hash() else {
            continue;
        };

        // Follow the update chain to get the latest version of this entry
        let record = match follow_update_chain(target_hash) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let Some(adj) = record
            .entry()
            .to_app_option::<PendingBalanceAdjustment>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "PendingBalanceAdjustment deserialization error: {:?}",
                    e
                )))
            })?
        else {
            continue;
        };

        // Skip entries not matching the requested currency
        if adj.currency_id != currency_id {
            continue;
        }

        // Skip fully completed adjustments
        if adj.provider_completed && adj.receiver_completed {
            continue;
        }

        let hours = adj.hours as f32;
        let mut current_action = record.action_address().clone();

        // Retry missing balance updates
        if !adj.provider_completed {
            // Neither side completed — retry provider first
            update_balance_after_exchange(
                &adj.provider_did,
                &adj.currency_id,
                hours,
                true, // provider gains
            )?;

            // Mark provider done
            let updated = PendingBalanceAdjustment {
                provider_completed: true,
                ..adj.clone()
            };
            current_action = update_entry(
                current_action,
                &EntryTypes::PendingBalanceAdjustment(updated),
            )?;
        }

        if !adj.receiver_completed {
            update_balance_after_exchange(
                &adj.receiver_did,
                &adj.currency_id,
                hours,
                false, // receiver pays
            )?;

            // Mark both sides done
            let updated = PendingBalanceAdjustment {
                provider_completed: true,
                receiver_completed: true,
                ..adj.clone()
            };
            update_entry(
                current_action,
                &EntryTypes::PendingBalanceAdjustment(updated),
            )?;
        }

        recovered += 1;
    }

    Ok(recovered)
}
