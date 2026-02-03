//! Markets Bridge - Cross-hApp Prediction Market Integration
//!
//! Enables prediction markets about events in any Mycelix hApp:
//! - Marketplace: "Will this seller deliver on time?"
//! - Governance: "Will MIP-47 pass?"
//! - Knowledge: "Will this claim achieve E3 verification?"
//! - DeSci: "Will this experiment replicate?"
//! - Supply Chain: "Is this product authentic?"
//!
//! Integrates with Mycelix Bridge Protocol for event subscriptions
//! and automatic resolution triggers.

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// CROSS-HAPP MARKET SOURCES
// ============================================================================

/// Supported hApps for cross-hApp markets
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum HappSource {
    Governance,
    Marketplace,
    Knowledge,
    Identity,
    Finance,
    SupplyChain,
    DeSci,
    Justice,
    Energy,
    Media,
    Custom(String),
}

impl HappSource {
    pub fn zome_name(&self) -> String {
        match self {
            HappSource::Governance => "governance".to_string(),
            HappSource::Marketplace => "marketplace".to_string(),
            HappSource::Knowledge => "knowledge".to_string(),
            HappSource::Identity => "identity".to_string(),
            HappSource::Finance => "finance".to_string(),
            HappSource::SupplyChain => "supply_chain".to_string(),
            HappSource::DeSci => "desci".to_string(),
            HappSource::Justice => "justice".to_string(),
            HappSource::Energy => "energy".to_string(),
            HappSource::Media => "media".to_string(),
            HappSource::Custom(name) => name.clone(),
        }
    }
}

// ============================================================================
// PREDICTION REQUESTS
// ============================================================================

/// A request from another hApp for a prediction market
#[hdk_entry_helper]
#[derive(Clone)]
pub struct PredictionRequest {
    /// Unique request ID
    pub id: EntryHash,

    /// Source hApp making the request
    pub source_happ: HappSource,

    /// The question to be predicted
    pub question: String,

    /// Context data from source hApp
    pub context: serde_json::Value,

    /// How to resolve the outcome
    pub resolution_criteria: ResolutionCriteria,

    /// How urgent is this prediction?
    pub urgency: Urgency,

    /// Bounty offered for prediction
    pub bounty: u64,

    /// When the request expires
    pub expires_at: u64,

    /// Status of the request
    pub status: RequestStatus,

    /// If fulfilled, the resulting market
    pub resulting_market: Option<EntryHash>,

    /// Requesting agent
    pub requester: AgentPubKey,

    /// Created timestamp
    pub created_at: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResolutionCriteria {
    /// Resolve when specific on-chain event occurs
    OnChainEvent {
        event_type: String,
        event_source: HappSource,
        filter: Option<serde_json::Value>,
    },

    /// Resolve when knowledge claim reaches specified status
    KnowledgeClaim {
        claim_id: Option<EntryHash>,
        claim_type: String,
        required_epistemic_level: u8,
    },

    /// Resolve via expert consensus
    ExpertConsensus {
        required_experts: u32,
        min_expert_matl: f64,
        domain: String,
    },

    /// Resolve at specific time with data fetch
    TimeBased {
        resolve_at: u64,
        data_source: String,
        data_query: String,
    },

    /// Custom resolution logic
    Custom {
        description: String,
        validator_fn: String,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Urgency {
    Low,      // Weeks to resolve
    Medium,   // Days to resolve
    High,     // Hours to resolve
    Critical, // Needs immediate attention
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum RequestStatus {
    Pending,
    Accepted,
    MarketCreated,
    Resolved,
    Expired,
    Rejected { reason: String },
}

// ============================================================================
// EVENT SUBSCRIPTIONS
// ============================================================================

/// Subscribe to events from other hApps for automatic resolution
#[hdk_entry_helper]
#[derive(Clone)]
pub struct EventSubscription {
    /// The market that will be resolved
    pub market_id: EntryHash,

    /// Event source hApp
    pub source_happ: HappSource,

    /// Event type to listen for
    pub event_type: String,

    /// Filter criteria for matching events
    pub filter: Option<serde_json::Value>,

    /// How to map event data to market outcome
    pub outcome_mapping: OutcomeMapping,

    /// Status
    pub active: bool,

    /// Created timestamp
    pub created_at: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OutcomeMapping {
    /// Direct: event contains outcome field
    Direct { field_path: String },

    /// Conditional: map event fields to outcomes
    Conditional {
        conditions: Vec<OutcomeCondition>,
        default_outcome: String,
    },

    /// Threshold: compare value to threshold
    Threshold {
        field_path: String,
        threshold: f64,
        above_outcome: String,
        below_outcome: String,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutcomeCondition {
    pub field_path: String,
    pub operator: ComparisonOperator,
    pub value: serde_json::Value,
    pub outcome: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ComparisonOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Contains,
    StartsWith,
}

// ============================================================================
// BRIDGE EVENTS
// ============================================================================

/// Events broadcast via Bridge Protocol
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BridgeEvent {
    /// Market created from cross-hApp request
    MarketCreated {
        market_id: EntryHash,
        source_happ: HappSource,
        question: String,
    },

    /// Market resolved
    MarketResolved {
        market_id: EntryHash,
        outcome: String,
        confidence: f64,
        source_happ: Option<HappSource>,
    },

    /// Prediction request received
    PredictionRequested {
        request_id: EntryHash,
        source_happ: HappSource,
        question: String,
        bounty: u64,
    },

    /// Reputation update from prediction accuracy
    ReputationUpdate {
        agent: AgentPubKey,
        domain: String,
        delta: f64,
        reason: String,
    },
}

// ============================================================================
// ZOME FUNCTIONS
// ============================================================================

#[hdk_extern]
pub fn submit_prediction_request(input: SubmitRequestInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;
    let requester = agent_info()?.agent_initial_pubkey;

    let request = PredictionRequest {
        id: EntryHash::from_raw_36(vec![0; 36]),
        source_happ: input.source_happ.clone(),
        question: input.question.clone(),
        context: input.context,
        resolution_criteria: input.resolution_criteria,
        urgency: input.urgency,
        bounty: input.bounty,
        expires_at: input.expires_at.unwrap_or(now + 7 * 24 * 60 * 60 * 1_000_000),
        status: RequestStatus::Pending,
        resulting_market: None,
        requester: requester.clone(),
        created_at: now,
    };

    // In HDK 0.6, create_entry returns ActionHash - compute EntryHash from entry
    let entry_hash = hash_entry(&request)?;
    let _action_hash = create_entry(&EntryTypes::PredictionRequest(request.clone()))?;

    // Link to requester
    create_link(requester, entry_hash.clone(), LinkTypes::RequesterToRequest, ())?;

    // Link to pending requests anchor
    let anchor = anchor_hash("pending_requests")?;
    create_link(anchor, entry_hash.clone(), LinkTypes::AnchorToRequest, ())?;

    // Broadcast event via bridge
    emit_signal(serde_json::json!({
        "type": "bridge_event",
        "event": BridgeEvent::PredictionRequested {
            request_id: entry_hash.clone(),
            source_happ: input.source_happ,
            question: input.question,
            bounty: input.bounty,
        }
    }))?;

    Ok(entry_hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubmitRequestInput {
    pub source_happ: HappSource,
    pub question: String,
    pub context: serde_json::Value,
    pub resolution_criteria: ResolutionCriteria,
    pub urgency: Urgency,
    pub bounty: u64,
    pub expires_at: Option<u64>,
}

#[hdk_extern]
pub fn accept_request(request_id: EntryHash) -> ExternResult<EntryHash> {
    // Get the record to extract the action hash for update
    let record = get(request_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;

    let mut request: PredictionRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Request entry not found".into())))?;

    if request.status != RequestStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request is not pending".into()
        )));
    }

    request.status = RequestStatus::Accepted;

    // HDK 0.6: update_entry takes ActionHash, not EntryHash
    let action_hash = record.action_hashed().hash.clone();
    update_entry(action_hash, &request)?;

    Ok(request_id)
}

/// Input for creating a market in the markets zome
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateMarketInput {
    pub question: String,
    pub description: String,
    pub outcomes: Vec<String>,
    pub closes_at: u64,
    pub resolution_source: Option<String>,
    pub source_happ: Option<String>,
    pub initial_liquidity: Option<u64>,
    pub domain: Option<String>,
}

#[hdk_extern]
pub fn create_market_from_request(input: CreateFromRequestInput) -> ExternResult<EntryHash> {
    // Get the record to extract action hash for update
    let record = get(input.request_id.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;

    let mut request: PredictionRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Request entry not found".into())))?;

    if request.status != RequestStatus::Accepted {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request must be accepted first".into()
        )));
    }

    // Build market creation input from the prediction request
    let create_market_input = CreateMarketInput {
        question: request.question.clone(),
        description: serde_json::to_string(&request.context).unwrap_or_default(),
        outcomes: input.outcomes.clone(),
        closes_at: input.closes_at,
        resolution_source: Some(format!("{:?}", request.resolution_criteria)),
        source_happ: Some(request.source_happ.zome_name()),
        initial_liquidity: Some(request.bounty),
        domain: request.context.get("domain").and_then(|d| d.as_str()).map(String::from),
    };

    // Call markets zome to create the market - HDK 0.6 ZomeCallResponse handling
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("markets"),
        FunctionName::new("create_market"),
        None,
        create_market_input,
    )
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to create market: {:?}", e))))?;

    let market_id: EntryHash = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode market ID: {:?}", e))))?,
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            return Err(wasm_error!(WasmErrorInner::Guest("Unauthorized zome call".into())));
        }
        ZomeCallResponse::NetworkError(e) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!("Network error: {}", e))));
        }
        ZomeCallResponse::CountersigningSession(e) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!("Countersigning error: {}", e))));
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            return Err(wasm_error!(WasmErrorInner::Guest("Authentication failed".into())));
        }
    };

    request.status = RequestStatus::MarketCreated;
    request.resulting_market = Some(market_id.clone());

    // HDK 0.6: update_entry takes ActionHash
    let action_hash = record.action_hashed().hash.clone();
    update_entry(action_hash, &request)?;

    // Set up event subscription if applicable
    if let Some(subscription_config) = input.event_subscription {
        subscribe_to_event(SubscribeInput {
            market_id: market_id.clone(),
            source_happ: request.source_happ.clone(),
            event_type: subscription_config.event_type,
            filter: subscription_config.filter,
            outcome_mapping: subscription_config.outcome_mapping,
        })?;
    }

    // Broadcast event
    emit_signal(serde_json::json!({
        "type": "bridge_event",
        "event": BridgeEvent::MarketCreated {
            market_id: market_id.clone(),
            source_happ: request.source_happ,
            question: request.question,
        }
    }))?;

    Ok(market_id)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateFromRequestInput {
    pub request_id: EntryHash,
    pub outcomes: Vec<String>,
    pub closes_at: u64,
    pub event_subscription: Option<EventSubscriptionConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EventSubscriptionConfig {
    pub event_type: String,
    pub filter: Option<serde_json::Value>,
    pub outcome_mapping: OutcomeMapping,
}

#[hdk_extern]
pub fn subscribe_to_event(input: SubscribeInput) -> ExternResult<EntryHash> {
    let now = sys_time()?.as_micros() as u64;

    let subscription = EventSubscription {
        market_id: input.market_id.clone(),
        source_happ: input.source_happ,
        event_type: input.event_type,
        filter: input.filter,
        outcome_mapping: input.outcome_mapping,
        active: true,
        created_at: now,
    };

    // HDK 0.6: create_entry returns ActionHash, compute EntryHash from entry
    let entry_hash = hash_entry(&subscription)?;
    let _action_hash = create_entry(&EntryTypes::EventSubscription(subscription))?;

    // Link subscription to market
    create_link(
        input.market_id,
        entry_hash.clone(),
        LinkTypes::MarketToSubscription,
        (),
    )?;

    Ok(entry_hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubscribeInput {
    pub market_id: EntryHash,
    pub source_happ: HappSource,
    pub event_type: String,
    pub filter: Option<serde_json::Value>,
    pub outcome_mapping: OutcomeMapping,
}

/// Input for triggering market resolution
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TriggerResolutionInput {
    pub market_id: EntryHash,
    pub outcome: String,
    pub evidence: serde_json::Value,
}

/// Handle incoming bridge event (called by other hApps)
#[hdk_extern]
pub fn handle_bridge_event(event: serde_json::Value) -> ExternResult<()> {
    // Find subscriptions matching this event
    let event_type = event
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("unknown");

    let source_happ = event
        .get("source_happ")
        .and_then(|s| serde_json::from_value::<HappSource>(s.clone()).ok());

    // Query all active subscriptions to find matches
    let subscriptions = get_active_subscriptions()?;

    for subscription in subscriptions {
        // Check if subscription matches this event
        if subscription.event_type != event_type {
            continue;
        }

        if let Some(ref src) = source_happ {
            if subscription.source_happ != *src {
                continue;
            }
        }

        // Check filter criteria if present
        if let Some(ref filter) = subscription.filter {
            if !event_matches_filter(&event, filter) {
                continue;
            }
        }

        // Map event data to market outcome
        let outcome = match &subscription.outcome_mapping {
            OutcomeMapping::Direct { field_path } => {
                extract_field_value(&event, field_path)
                    .unwrap_or_else(|| "unknown".to_string())
            }
            OutcomeMapping::Conditional { conditions, default_outcome } => {
                let mut result = default_outcome.clone();
                for condition in conditions {
                    if check_condition(&event, condition) {
                        result = condition.outcome.clone();
                        break;
                    }
                }
                result
            }
            OutcomeMapping::Threshold { field_path, threshold, above_outcome, below_outcome } => {
                if let Some(value) = extract_field_value(&event, field_path)
                    .and_then(|v| v.parse::<f64>().ok())
                {
                    if value >= *threshold {
                        above_outcome.clone()
                    } else {
                        below_outcome.clone()
                    }
                } else {
                    "unknown".to_string()
                }
            }
        };

        // Trigger resolution for this market
        let resolution_input = TriggerResolutionInput {
            market_id: subscription.market_id.clone(),
            outcome: outcome.clone(),
            evidence: event.clone(),
        };

        // Call resolution zome to start resolution with this outcome - HDK 0.6 handling
        let response = call(
            CallTargetCell::Local,
            ZomeName::new("resolution"),
            FunctionName::new("trigger_automated_resolution"),
            None,
            resolution_input,
        )
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to trigger resolution: {:?}", e))))?;

        let _result: Result<EntryHash, _> = match response {
            ZomeCallResponse::Ok(extern_io) => extern_io.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode resolution result: {:?}", e)))),
            _ => Err(wasm_error!(WasmErrorInner::Guest("Resolution call failed".into()))),
        };

        // Deactivate subscription after triggering
        deactivate_subscription(subscription.market_id.clone())?;

        emit_signal(serde_json::json!({
            "type": "market_resolution_triggered",
            "market_id": subscription.market_id,
            "outcome": outcome,
            "event_type": event_type,
            "source_happ": source_happ,
        }))?;
    }

    emit_signal(serde_json::json!({
        "type": "bridge_event_received",
        "event_type": event_type,
        "source_happ": source_happ,
    }))?;

    Ok(())
}

/// Get all active event subscriptions
fn get_active_subscriptions() -> ExternResult<Vec<EventSubscription>> {
    let anchor = anchor_hash("active_subscriptions")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::MarketToSubscription)?,
        GetStrategy::default(),
    )?;

    let mut subscriptions = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_entry_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(sub) = record.entry().to_app_option::<EventSubscription>().ok().flatten() {
                    if sub.active {
                        subscriptions.push(sub);
                    }
                }
            }
        }
    }

    Ok(subscriptions)
}

/// Deactivate a subscription after it has triggered
fn deactivate_subscription(market_id: EntryHash) -> ExternResult<()> {
    let links = get_links(
        LinkQuery::try_new(market_id, LinkTypes::MarketToSubscription)?,
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(record) = get(entry_hash.clone(), GetOptions::default())? {
                if let Some(mut sub) = record.entry().to_app_option::<EventSubscription>().ok().flatten() {
                    sub.active = false;
                    // HDK 0.6: update_entry takes ActionHash, not EntryHash
                    let action_hash = record.action_hashed().hash.clone();
                    update_entry(action_hash, &sub)?;
                }
            }
        }
    }

    Ok(())
}

/// Check if event matches filter criteria
fn event_matches_filter(event: &serde_json::Value, filter: &serde_json::Value) -> bool {
    // Simple key-value matching for filter
    if let (Some(event_obj), Some(filter_obj)) = (event.as_object(), filter.as_object()) {
        for (key, expected_value) in filter_obj {
            if let Some(actual_value) = event_obj.get(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    } else {
        false
    }
}

/// Extract a field value from nested JSON using dot notation
fn extract_field_value(event: &serde_json::Value, field_path: &str) -> Option<String> {
    let parts: Vec<&str> = field_path.split('.').collect();
    let mut current = event;

    for part in parts {
        current = current.get(part)?;
    }

    current.as_str().map(String::from)
        .or_else(|| Some(current.to_string().trim_matches('"').to_string()))
}

/// Check a single condition against event data
fn check_condition(event: &serde_json::Value, condition: &OutcomeCondition) -> bool {
    let Some(actual_value) = event.get(&condition.field_path) else {
        return false;
    };

    match condition.operator {
        ComparisonOperator::Equals => actual_value == &condition.value,
        ComparisonOperator::NotEquals => actual_value != &condition.value,
        ComparisonOperator::GreaterThan => {
            if let (Some(a), Some(b)) = (actual_value.as_f64(), condition.value.as_f64()) {
                a > b
            } else {
                false
            }
        }
        ComparisonOperator::LessThan => {
            if let (Some(a), Some(b)) = (actual_value.as_f64(), condition.value.as_f64()) {
                a < b
            } else {
                false
            }
        }
        ComparisonOperator::GreaterOrEqual => {
            if let (Some(a), Some(b)) = (actual_value.as_f64(), condition.value.as_f64()) {
                a >= b
            } else {
                false
            }
        }
        ComparisonOperator::LessOrEqual => {
            if let (Some(a), Some(b)) = (actual_value.as_f64(), condition.value.as_f64()) {
                a <= b
            } else {
                false
            }
        }
        ComparisonOperator::Contains => {
            if let (Some(a), Some(b)) = (actual_value.as_str(), condition.value.as_str()) {
                a.contains(b)
            } else {
                false
            }
        }
        ComparisonOperator::StartsWith => {
            if let (Some(a), Some(b)) = (actual_value.as_str(), condition.value.as_str()) {
                a.starts_with(b)
            } else {
                false
            }
        }
    }
}

#[hdk_extern]
pub fn get_prediction_request(hash: EntryHash) -> ExternResult<Option<PredictionRequest>> {
    match get(hash, GetOptions::default())? {
        Some(record) => {
            let request: PredictionRequest = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Request not found".into())))?;
            Ok(Some(request))
        }
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn list_pending_requests(_: ()) -> ExternResult<Vec<PredictionRequest>> {
    let anchor = anchor_hash("pending_requests")?;
    let links =
        get_links(LinkQuery::try_new(anchor, LinkTypes::AnchorToRequest)?, GetStrategy::default())?;

    let mut requests = Vec::new();
    for link in links {
        if let Some(request) = get_prediction_request(link.target.into_entry_hash().unwrap())? {
            if request.status == RequestStatus::Pending {
                requests.push(request);
            }
        }
    }

    Ok(requests)
}

#[hdk_extern]
pub fn list_requests_by_happ(source: HappSource) -> ExternResult<Vec<PredictionRequest>> {
    let requests = list_pending_requests(())?;
    Ok(requests
        .into_iter()
        .filter(|r| r.source_happ == source)
        .collect())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn anchor_hash(anchor: &str) -> ExternResult<EntryHash> {
    use hdk::prelude::{hash_entry, Entry, AppEntryBytes, SerializedBytes, UnsafeBytes};
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", anchor).into_bytes()
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

// ============================================================================
// ENTRY AND LINK DEFINITIONS
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    PredictionRequest(PredictionRequest),
    EventSubscription(EventSubscription),
}

#[hdk_link_types]
pub enum LinkTypes {
    RequesterToRequest,
    AnchorToRequest,
    MarketToSubscription,
    HappToRequest,
}
