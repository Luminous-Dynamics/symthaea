//! TEND (Time Exchange) Integrity Zome
//!
//! Implements Commons Charter Article II, Section 2 - Time Exchange Module
//!
//! Core Principles:
//! - 1 TEND = 1 hour of service (all labor valued equally)
//! - Mutual credit: members can go negative (debt) or positive (credit)
//! - Balance limits: ±40 TEND (prevents excessive debt/credit)
//! - No interest: time doesn't compound, debt doesn't grow
//! - Community-scoped: each Local DAO has its own TEND ledger
//!
//! Constitutional Reference: Commons Charter v1.0, Article II, Section 2
//! MIP Template Reference: MIP-C-042

use hdi::prelude::*;

// =============================================================================
// CONSTANTS (Per MIP-C-042 Template)
// =============================================================================

/// Balance limit (both positive and negative)
pub const BALANCE_LIMIT: i32 = 40;

/// 1 TEND = 1 hour of service (in minutes for precision)
pub const TEND_UNIT_MINUTES: u32 = 60;

/// Maximum service duration in one transaction (8 hours)
pub const MAX_SERVICE_HOURS: u32 = 8;

/// Minimum service duration (15 minutes)
pub const MIN_SERVICE_MINUTES: u32 = 15;

/// Elevated balance limit (Stressed state)
pub const BALANCE_LIMIT_ELEVATED: i32 = 60;

/// High balance limit (Critical state)
pub const BALANCE_LIMIT_HIGH: i32 = 80;

/// Emergency balance limit (Failing state)
pub const BALANCE_LIMIT_EMERGENCY: i32 = 120;

/// Apprentice balance limit
pub const APPRENTICE_BALANCE_LIMIT: i32 = 10;

// =============================================================================
// TEND LIMIT TIERS
// =============================================================================

/// Tiered balance limits based on community/member state.
///
/// Normal operation uses ±40, but elevated tiers allow wider limits
/// during periods of community stress or emergency. The coordinator
/// zome is responsible for determining which tier applies dynamically.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TendLimitTier {
    /// Normal operation: ±40 TEND
    Normal,
    /// Stressed state: ±60 TEND
    Elevated,
    /// Critical state: ±80 TEND
    High,
    /// Failing state: ±120 TEND
    Emergency,
}

impl TendLimitTier {
    /// Returns the balance limit for this tier.
    pub fn limit(&self) -> i32 {
        match self {
            TendLimitTier::Normal => BALANCE_LIMIT,
            TendLimitTier::Elevated => BALANCE_LIMIT_ELEVATED,
            TendLimitTier::High => BALANCE_LIMIT_HIGH,
            TendLimitTier::Emergency => BALANCE_LIMIT_EMERGENCY,
        }
    }
}

// =============================================================================
// ENTRY TYPES
// =============================================================================

/// A Time Exchange transaction
///
/// When Alice provides 2 hours of service to Bob:
/// - Alice earns +2 TEND (credit)
/// - Bob spends -2 TEND (debt)
///
/// This is MUTUAL CREDIT: the total TEND in the system is always zero.
/// One person's credit is another's debt.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TendExchange {
    /// Unique identifier
    pub id: String,

    /// DID of the service provider (earns TEND)
    pub provider_did: String,

    /// DID of the service receiver (spends TEND)
    pub receiver_did: String,

    /// Amount of TEND exchanged (in hours, e.g., 2.5 = 2h 30m)
    pub hours: f32,

    /// Description of the service provided
    pub service_description: String,

    /// Category of service
    pub service_category: ServiceCategory,

    /// Cultural alias used (if any) - e.g., "CARE", "HOURS"
    pub cultural_alias: Option<String>,

    /// The DAO/community where this exchange occurred
    pub dao_did: String,

    /// When the exchange was recorded
    pub timestamp: Timestamp,

    /// Status of the exchange
    pub status: ExchangeStatus,

    /// Optional: when the service was actually performed (if different from recorded)
    pub service_date: Option<Timestamp>,
}

/// Categories of service that can be exchanged
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ServiceCategory {
    /// Childcare, eldercare, pet care
    CareWork,
    /// Home repairs, maintenance
    HomeServices,
    /// Cooking, meal prep
    FoodServices,
    /// Driving, moving help
    Transportation,
    /// Tutoring, teaching
    Education,
    /// General help, errands
    GeneralAssistance,
    /// Administrative, paperwork
    Administrative,
    /// Creative work (art, music, writing)
    Creative,
    /// Tech support, computer help
    TechSupport,
    /// Health and wellness (non-medical)
    Wellness,
    /// Gardening, landscaping
    Gardening,
    /// Custom category defined by DAO
    Custom(String),
}

/// Status of an exchange
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ExchangeStatus {
    /// Proposed by provider, awaiting receiver confirmation
    Proposed,
    /// Confirmed by both parties
    Confirmed,
    /// Disputed by one party
    Disputed,
    /// Cancelled before confirmation
    Cancelled,
    /// Resolved after dispute
    Resolved,
}

/// Member's TEND balance within a DAO
///
/// Balance can be positive (credit - you've provided more than received)
/// or negative (debt - you've received more than provided).
/// Both are limited to ±40 TEND.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TendBalance {
    /// DID of the member
    pub member_did: String,

    /// The DAO/community
    pub dao_did: String,

    /// Current balance (can be negative)
    pub balance: i32,  // Using i32 for negative support

    /// Total hours provided (lifetime)
    pub total_provided: f32,

    /// Total hours received (lifetime)
    pub total_received: f32,

    /// Number of exchanges participated in
    pub exchange_count: u32,

    /// Last activity timestamp
    pub last_activity: Timestamp,
}

/// Service listing - offering services to the community
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ServiceListing {
    /// Unique identifier
    pub id: String,

    /// DID of the service provider
    pub provider_did: String,

    /// The DAO/community
    pub dao_did: String,

    /// Title of the service
    pub title: String,

    /// Description
    pub description: String,

    /// Category
    pub category: ServiceCategory,

    /// Estimated hours (optional)
    pub estimated_hours: Option<f32>,

    /// Availability notes
    pub availability: Option<String>,

    /// Whether the listing is active
    pub active: bool,

    /// When listed
    pub created: Timestamp,
}

/// Service request - requesting services from the community
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ServiceRequest {
    /// Unique identifier
    pub id: String,

    /// DID of the requester
    pub requester_did: String,

    /// The DAO/community
    pub dao_did: String,

    /// Title of the request
    pub title: String,

    /// Description of what's needed
    pub description: String,

    /// Category
    pub category: ServiceCategory,

    /// Estimated hours needed
    pub estimated_hours: Option<f32>,

    /// Urgency level
    pub urgency: Urgency,

    /// Whether the request is still open
    pub open: bool,

    /// When requested
    pub created: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Urgency {
    /// Flexible timing
    Flexible,
    /// Within a week
    SoonPreferred,
    /// Within a few days
    Urgent,
    /// Immediate need
    Emergency,
}

/// Quality rating for a completed exchange
///
/// After an exchange is confirmed, either party can rate the experience.
/// Ratings feed into MYCEL reputation scores and help the community
/// identify reliable service providers.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct QualityRating {
    /// ID of the exchange being rated
    pub exchange_id: String,

    /// DID of the member submitting the rating
    pub rater_did: String,

    /// DID of the service provider being rated
    pub provider_did: String,

    /// Rating from 1-5 stars
    pub rating: u8,

    /// Optional comment explaining the rating
    pub comment: Option<String>,

    /// When the rating was submitted
    pub timestamp: Timestamp,
}

// =============================================================================
// DISPUTE RESOLUTION
// =============================================================================

/// Stage of a dispute case, following the three-tier resolution process.
///
/// Disputes escalate through stages with increasing community involvement:
/// 1. DirectNegotiation - 72 hours for the two parties to resolve directly
/// 2. MediationPanel - 3 random members with MYCEL > 0.5, 7 day window
/// 3. GovernanceVote - Full community vote, final and binding
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeStage {
    /// 72 hours between parties to resolve directly
    DirectNegotiation,
    /// 3 random members with MYCEL > 0.5, 7 days to mediate
    MediationPanel,
    /// Final, binding community governance vote
    GovernanceVote,
}

/// A dispute case for a contested exchange
///
/// When an exchange is disputed, a DisputeCase tracks the resolution
/// process through the three-tier escalation system defined in the
/// Commons Charter.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DisputeCase {
    /// Unique identifier for the dispute
    pub id: String,

    /// ID of the disputed exchange
    pub exchange_id: String,

    /// DID of the member who filed the dispute
    pub complainant_did: String,

    /// DID of the member the dispute is against
    pub respondent_did: String,

    /// Current stage of the dispute
    pub stage: DisputeStage,

    /// Description of the dispute
    pub description: String,

    /// DIDs of assigned mediators (populated in MediationPanel stage)
    pub mediator_dids: Vec<String>,

    /// Resolution outcome (populated when resolved)
    pub resolution: Option<String>,

    /// When the dispute was opened
    pub opened_at: Timestamp,

    /// When the dispute was escalated to the next stage (if applicable)
    pub escalated_at: Option<Timestamp>,

    /// When the dispute was resolved (if applicable)
    pub resolved_at: Option<Timestamp>,
}

// =============================================================================
// ENTRY & LINK TYPE ENUMS
// =============================================================================

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    TendExchange(TendExchange),
    TendBalance(TendBalance),
    ServiceListing(ServiceListing),
    ServiceRequest(ServiceRequest),
    Anchor(Anchor),
    QualityRating(QualityRating),
    DisputeCase(DisputeCase),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from provider DID to exchanges where they provided
    ProviderToExchanges,
    /// Link from receiver DID to exchanges where they received
    ReceiverToExchanges,
    /// Link from member DID to their balance
    MemberToBalance,
    /// Link from DAO to all exchanges in that community
    DaoToExchanges,
    /// Link from DAO to service listings
    DaoToListings,
    /// Link from DAO to service requests
    DaoToRequests,
    /// Link from provider to their listings
    ProviderToListings,
    /// Link from category anchor to listings
    CategoryToListings,
    /// Link from exchange ID to exchange entry (for lookup by ID)
    ExchangeIdToExchange,
    /// Link type for anchor/path infrastructure
    AnchorLinks,
    /// Link from exchange to quality ratings
    ExchangeToRating,
    /// Link from member DID to disputes they are involved in
    MemberToDisputes,
    /// Link from exchange to its dispute case
    ExchangeToDispute,
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
                    EntryTypes::TendExchange(exchange) => {
                        validate_create_exchange(EntryCreationAction::Create(action), exchange)
                    }
                    EntryTypes::TendBalance(balance) => {
                        validate_create_balance(EntryCreationAction::Create(action), balance)
                    }
                    EntryTypes::ServiceListing(listing) => {
                        validate_create_listing(EntryCreationAction::Create(action), listing)
                    }
                    EntryTypes::ServiceRequest(request) => {
                        validate_create_request(EntryCreationAction::Create(action), request)
                    }
                    EntryTypes::QualityRating(rating) => {
                        validate_create_quality_rating(EntryCreationAction::Create(action), rating)
                    }
                    EntryTypes::DisputeCase(dispute) => {
                        validate_create_dispute_case(EntryCreationAction::Create(action), dispute)
                    }
                    // Anchors are always valid (just hash placeholders)
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                match app_entry {
                    EntryTypes::TendExchange(exchange) => {
                        validate_update_exchange(action, exchange)
                    }
                    EntryTypes::TendBalance(balance) => {
                        validate_update_balance(action, balance)
                    }
                    EntryTypes::ServiceListing(listing) => {
                        validate_update_listing(action, listing)
                    }
                    EntryTypes::ServiceRequest(request) => {
                        validate_update_request(action, request)
                    }
                    EntryTypes::QualityRating(_) => {
                        // Ratings are immutable once created
                        Ok(ValidateCallbackResult::Invalid(
                            "Quality ratings cannot be updated".into(),
                        ))
                    }
                    EntryTypes::DisputeCase(dispute) => {
                        validate_update_dispute_case(action, dispute)
                    }
                    // Anchors cannot be updated
                    EntryTypes::Anchor(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Anchors cannot be updated".into(),
                        ))
                    }
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            match link_type {
                LinkTypes::ProviderToExchanges => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ReceiverToExchanges => Ok(ValidateCallbackResult::Valid),
                LinkTypes::MemberToBalance => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DaoToExchanges => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DaoToListings => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DaoToRequests => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ProviderToListings => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CategoryToListings => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ExchangeIdToExchange => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AnchorLinks => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ExchangeToRating => Ok(ValidateCallbackResult::Valid),
                LinkTypes::MemberToDisputes => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ExchangeToDispute => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_exchange(
    _action: EntryCreationAction,
    exchange: TendExchange,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DIDs
    if !exchange.provider_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Provider must be a valid DID".into()));
    }
    if !exchange.receiver_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Receiver must be a valid DID".into()));
    }
    if !exchange.dao_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("DAO must be a valid DID".into()));
    }

    // Cannot exchange with yourself
    if exchange.provider_did == exchange.receiver_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot exchange time with yourself".into(),
        ));
    }

    // Hours must be positive and within limits
    if exchange.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Hours must be positive".into()));
    }

    let minutes = (exchange.hours * 60.0) as u32;
    if minutes < MIN_SERVICE_MINUTES {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Minimum service duration is {} minutes", MIN_SERVICE_MINUTES),
        ));
    }
    if exchange.hours > MAX_SERVICE_HOURS as f32 {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Maximum service duration is {} hours per exchange", MAX_SERVICE_HOURS),
        ));
    }

    // Description required
    if exchange.service_description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Service description is required".into(),
        ));
    }
    if exchange.service_description.len() > 2000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Service description too long (max 2000 chars)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_exchange(
    _action: Update,
    exchange: TendExchange,
) -> ExternResult<ValidateCallbackResult> {
    // Only status can change (Proposed -> Confirmed/Disputed/Cancelled)
    // Core data (provider, receiver, hours) cannot change
    if exchange.hours <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Hours must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_balance(
    _action: EntryCreationAction,
    balance: TendBalance,
) -> ExternResult<ValidateCallbackResult> {
    if !balance.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Member must be a valid DID".into()));
    }
    if !balance.dao_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("DAO must be a valid DID".into()));
    }

    // Balance must be within limits
    if balance.balance.abs() > BALANCE_LIMIT {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Balance exceeds limit of ±{}", BALANCE_LIMIT),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_balance(
    _action: Update,
    balance: TendBalance,
) -> ExternResult<ValidateCallbackResult> {
    // Balance must stay within limits
    if balance.balance.abs() > BALANCE_LIMIT {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Balance would exceed limit of ±{}", BALANCE_LIMIT),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_listing(
    _action: EntryCreationAction,
    listing: ServiceListing,
) -> ExternResult<ValidateCallbackResult> {
    if !listing.provider_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Provider must be a valid DID".into()));
    }
    if !listing.dao_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("DAO must be a valid DID".into()));
    }
    if listing.title.is_empty() || listing.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid("Title must be 1-200 chars".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_listing(
    _action: Update,
    _listing: ServiceListing,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_request(
    _action: EntryCreationAction,
    request: ServiceRequest,
) -> ExternResult<ValidateCallbackResult> {
    if !request.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Requester must be a valid DID".into()));
    }
    if !request.dao_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("DAO must be a valid DID".into()));
    }
    if request.title.is_empty() || request.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid("Title must be 1-200 chars".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_request(
    _action: Update,
    _request: ServiceRequest,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_quality_rating(
    _action: EntryCreationAction,
    rating: QualityRating,
) -> ExternResult<ValidateCallbackResult> {
    // Validate rater DID
    if !rating.rater_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Rater must be a valid DID".into()));
    }

    // Validate provider DID
    if !rating.provider_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Provider must be a valid DID".into()));
    }

    // Cannot rate yourself
    if rating.rater_did == rating.provider_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot rate yourself".into(),
        ));
    }

    // Rating must be 1-5 stars
    if rating.rating < 1 || rating.rating > 5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rating must be between 1 and 5".into(),
        ));
    }

    // Exchange ID must not be empty
    if rating.exchange_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange ID is required".into(),
        ));
    }

    // Comment length check (if provided)
    if let Some(ref comment) = rating.comment {
        if comment.len() > 2000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Comment too long (max 2000 chars)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_dispute_case(
    _action: EntryCreationAction,
    dispute: DisputeCase,
) -> ExternResult<ValidateCallbackResult> {
    // Validate complainant DID
    if !dispute.complainant_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Complainant must be a valid DID".into(),
        ));
    }

    // Validate respondent DID
    if !dispute.respondent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Respondent must be a valid DID".into(),
        ));
    }

    // Cannot dispute with yourself
    if dispute.complainant_did == dispute.respondent_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot file a dispute against yourself".into(),
        ));
    }

    // Validate mediator DIDs (if any are present)
    for mediator_did in &dispute.mediator_dids {
        if !mediator_did.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "All mediator DIDs must be valid".into(),
            ));
        }
    }

    // Description is required
    if dispute.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute description is required".into(),
        ));
    }
    if dispute.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute description too long (max 5000 chars)".into(),
        ));
    }

    // Exchange ID must not be empty
    if dispute.exchange_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Exchange ID is required".into(),
        ));
    }

    // Dispute ID must not be empty
    if dispute.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute ID is required".into(),
        ));
    }

    // New disputes must start at DirectNegotiation stage
    if dispute.stage != DisputeStage::DirectNegotiation {
        return Ok(ValidateCallbackResult::Invalid(
            "New disputes must start at DirectNegotiation stage".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_dispute_case(
    _action: Update,
    dispute: DisputeCase,
) -> ExternResult<ValidateCallbackResult> {
    // Validate core DID fields remain valid
    if !dispute.complainant_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Complainant must be a valid DID".into(),
        ));
    }
    if !dispute.respondent_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Respondent must be a valid DID".into(),
        ));
    }

    // Validate mediator DIDs
    for mediator_did in &dispute.mediator_dids {
        if !mediator_did.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "All mediator DIDs must be valid".into(),
            ));
        }
    }

    // MediationPanel stage requires exactly 3 mediators
    if dispute.stage == DisputeStage::MediationPanel && dispute.mediator_dids.len() != 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "MediationPanel stage requires exactly 3 mediators".into(),
        ));
    }

    // Description must remain non-empty
    if dispute.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute description is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
