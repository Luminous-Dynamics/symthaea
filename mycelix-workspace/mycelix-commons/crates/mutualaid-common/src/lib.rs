// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mutual Aid Common - Shared Types for Mycelix Mutual Aid hApp
//!
//! This crate contains all the shared types, enums, and structures used across
//! the Mutual Aid hApp zomes. It implements community economic coordination:
//!
//! - Time Banking (1 hour = 1 hour, regardless of service type)
//! - Community Credit Circles (mutual credit with automatic clearing)
//! - Resource Sharing (tools, vehicles, spaces, equipment)
//! - Needs Matching (offers and requests with emergency flagging)
//! - Circle Governance (democratic decision-making)
//! - Cross-hApp Bridge (identity, governance, MATL integration)

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// TIME BANKING TYPES
// =============================================================================

/// Time credit representing hours of service owed or earned
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TimeCredit {
    /// Amount in hours (can be fractional)
    pub hours: f64,
    /// Who earned this credit
    pub earner: AgentPubKey,
    /// Who owes this credit
    pub debtor: AgentPubKey,
    /// Service category
    pub service_category: ServiceCategory,
    /// Description of the service performed
    pub description: String,
    /// When the service was performed
    pub performed_at: Timestamp,
    /// Expiration date (if any)
    pub expires_at: Option<Timestamp>,
}

/// Service categories for time banking
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ServiceCategory {
    // Care Work
    Childcare,
    Eldercare,
    PetCare,
    PersonalCare,

    // Home Services
    Cleaning,
    Cooking,
    Gardening,
    HomeRepair,
    MovingHelp,

    // Professional Skills
    LegalAdvice,
    FinancialAdvice,
    MedicalConsult,
    TechSupport,

    // Education
    Tutoring,
    LanguageTeaching,
    MusicLessons,
    ArtInstruction,

    // Transportation
    Driving,
    Delivery,
    Errands,

    // Creative
    Photography,
    Design,
    Writing,
    Crafts,

    // Community
    EventPlanning,
    Facilitation,
    Mediation,
    Organizing,

    /// Custom category
    Custom(String),
}

/// A service offer in the time bank
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ServiceOffer {
    /// Unique identifier
    pub id: String,
    /// Who is offering
    pub provider: AgentPubKey,
    /// Service category
    pub category: ServiceCategory,
    /// Title of the service
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Skills/qualifications
    pub qualifications: Vec<String>,
    /// Availability
    pub availability: Availability,
    /// Location constraints
    pub location: LocationConstraint,
    /// Minimum time unit (e.g., 0.5 hours)
    pub min_duration_hours: f64,
    /// Maximum time per session
    pub max_duration_hours: Option<f64>,
    /// Whether offer is currently active
    pub active: bool,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Last update timestamp
    pub updated_at: Timestamp,
}

/// A service request in the time bank
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ServiceRequest {
    /// Unique identifier
    pub id: String,
    /// Who needs the service
    pub requester: AgentPubKey,
    /// Service category needed
    pub category: ServiceCategory,
    /// Title of the request
    pub title: String,
    /// Detailed description of need
    pub description: String,
    /// Urgency level
    pub urgency: UrgencyLevel,
    /// When the service is needed
    pub needed_by: Option<Timestamp>,
    /// Estimated duration in hours
    pub estimated_hours: f64,
    /// Location
    pub location: LocationConstraint,
    /// Request status
    pub status: RequestStatus,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// A completed time exchange
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TimeExchange {
    /// Unique identifier
    pub id: String,
    /// Link to service offer (if from offer)
    pub offer_hash: Option<ActionHash>,
    /// Link to service request (if from request)
    pub request_hash: Option<ActionHash>,
    /// Service provider
    pub provider: AgentPubKey,
    /// Service recipient
    pub recipient: AgentPubKey,
    /// Hours exchanged
    pub hours: f64,
    /// Service category
    pub category: ServiceCategory,
    /// Description of what was done
    pub description: String,
    /// When the exchange occurred
    pub completed_at: Timestamp,
    /// Provider's rating of the experience
    pub provider_rating: Option<Rating>,
    /// Recipient's rating of the service
    pub recipient_rating: Option<Rating>,
    /// Both parties confirmed
    pub confirmed: bool,
}

/// Rating for time exchange
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Rating {
    /// Score (1-5)
    pub score: u8,
    /// Optional comment
    pub comment: Option<String>,
    /// Timestamp of rating
    pub rated_at: Timestamp,
}

// =============================================================================
// CREDIT CIRCLE TYPES
// =============================================================================

/// A mutual credit circle
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CreditCircle {
    /// Unique identifier
    pub id: String,
    /// Circle name
    pub name: String,
    /// Description and purpose
    pub description: String,
    /// Currency/unit name
    pub currency_name: String,
    /// Currency symbol
    pub currency_symbol: String,
    /// Default credit limit for new members
    pub default_credit_limit: i64,
    /// Maximum credit limit allowed
    pub max_credit_limit: i64,
    /// Transaction fee percentage (if any)
    pub transaction_fee_percent: f64,
    /// Monthly demurrage rate (if any, for negative balances)
    pub demurrage_rate_percent: f64,
    /// Geographic scope
    pub geographic_scope: Option<String>,
    /// Founding members
    pub founders: Vec<AgentPubKey>,
    /// Circle rules
    pub rules_hash: Option<ActionHash>,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Whether circle is active
    pub active: bool,
}

/// Credit line for a member within a circle
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CreditLine {
    /// Link to the credit circle
    pub circle_hash: ActionHash,
    /// Member agent
    pub member: AgentPubKey,
    /// Credit limit (how negative balance can go)
    pub credit_limit: i64,
    /// Current balance (can be negative)
    pub balance: i64,
    /// Total credit extended over time
    pub total_credit_extended: u64,
    /// Total credit received over time
    pub total_credit_received: u64,
    /// When member joined
    pub joined_at: Timestamp,
    /// Line status
    pub status: CreditLineStatus,
    /// Last activity
    pub last_activity: Timestamp,
}

/// Status of a credit line
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CreditLineStatus {
    /// Active and in good standing
    Active,
    /// Temporarily frozen
    Frozen,
    /// Suspended for rule violation
    Suspended,
    /// Voluntarily closed
    Closed,
}

/// A credit transaction within a circle
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CreditTransaction {
    /// Unique identifier
    pub id: String,
    /// Link to the credit circle
    pub circle_hash: ActionHash,
    /// Sender (balance decreases)
    pub from: AgentPubKey,
    /// Receiver (balance increases)
    pub to: AgentPubKey,
    /// Amount transferred
    pub amount: i64,
    /// Transaction type
    pub transaction_type: TransactionType,
    /// Description/memo
    pub memo: String,
    /// Link to related exchange (if any)
    pub related_exchange_hash: Option<ActionHash>,
    /// Transaction timestamp
    pub created_at: Timestamp,
    /// Whether transaction is confirmed
    pub confirmed: bool,
}

/// Types of credit transactions
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransactionType {
    /// Payment for goods/services
    Payment,
    /// Repayment of debt
    Repayment,
    /// Gift/donation
    Gift,
    /// Automatic clearing between members
    Clearing,
    /// Fee deduction
    Fee,
    /// Demurrage charge
    Demurrage,
    /// Adjustment by governance
    Adjustment,
}

/// Current balance snapshot for a member
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Balance {
    /// Member agent
    pub member: AgentPubKey,
    /// Link to circle
    pub circle_hash: ActionHash,
    /// Current balance
    pub balance: i64,
    /// Credit available (limit - abs(min(0, balance)))
    pub credit_available: i64,
    /// As of timestamp
    pub as_of: Timestamp,
}

// =============================================================================
// RESOURCE SHARING TYPES
// =============================================================================

/// A shared resource (tool, vehicle, space, equipment)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SharedResource {
    /// Unique identifier
    pub id: String,
    /// Resource owner
    pub owner: AgentPubKey,
    /// Resource name
    pub name: String,
    /// Description
    pub description: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Current condition
    pub condition: ResourceCondition,
    /// Photos (IPFS CIDs)
    pub photos: Vec<String>,
    /// Location
    pub location: LocationConstraint,
    /// Availability schedule
    pub availability: Availability,
    /// Sharing model
    pub sharing_model: SharingModel,
    /// Usage requirements/instructions
    pub usage_instructions: String,
    /// Insurance/liability notes
    pub liability_notes: Option<String>,
    /// Whether resource is currently available
    pub currently_available: bool,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Last update
    pub updated_at: Timestamp,
}

/// Types of shared resources
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResourceType {
    // Tools
    PowerTool,
    HandTool,
    GardenTool,
    CookingEquipment,
    CraftingSupplies,

    // Vehicles
    Car,
    Truck,
    Bicycle,
    Trailer,
    Boat,

    // Spaces
    MeetingRoom,
    Workshop,
    Kitchen,
    GardenPlot,
    StorageSpace,
    ParkingSpot,

    // Equipment
    CampingGear,
    SportsEquipment,
    MusicInstrument,
    Photography,
    Projector,

    // Other
    Custom(String),
}

/// Resource condition
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResourceCondition {
    Excellent,
    Good,
    Fair,
    NeedsRepair,
    BeingRepaired,
}

/// How the resource is shared
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SharingModel {
    /// Free to use
    pub free: bool,
    /// Deposit required (in local currency/credits)
    pub deposit: Option<i64>,
    /// Hourly rate (0 if free)
    pub hourly_rate: i64,
    /// Daily rate
    pub daily_rate: Option<i64>,
    /// Can pay in time credits
    pub accepts_time_credits: bool,
    /// Can pay in circle credits
    pub accepts_circle_credits: bool,
    /// Circle for credit payment (if applicable)
    pub circle_hash: Option<ActionHash>,
}

/// A booking for a shared resource
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Booking {
    /// Unique identifier
    pub id: String,
    /// Link to the resource
    pub resource_hash: ActionHash,
    /// Who is booking
    pub booker: AgentPubKey,
    /// Start time
    pub start_time: Timestamp,
    /// End time
    pub end_time: Timestamp,
    /// Purpose/notes
    pub purpose: String,
    /// Booking status
    pub status: BookingStatus,
    /// Payment method used
    pub payment_method: Option<PaymentMethod>,
    /// Payment transaction hash
    pub payment_hash: Option<ActionHash>,
    /// Created timestamp
    pub created_at: Timestamp,
}

/// Booking status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BookingStatus {
    Pending,
    Confirmed,
    Active,
    Completed,
    Cancelled,
    NoShow,
}

/// Payment method for resource usage
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentMethod {
    Free,
    TimeCredits(f64),
    CircleCredits {
        circle_hash: ActionHash,
        amount: i64,
    },
    External(String),
}

/// Usage record for tracking resource use
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Usage {
    /// Link to booking
    pub booking_hash: ActionHash,
    /// Actual start time
    pub actual_start: Timestamp,
    /// Actual end time
    pub actual_end: Option<Timestamp>,
    /// Condition before use
    pub condition_before: ResourceCondition,
    /// Condition after use
    pub condition_after: Option<ResourceCondition>,
    /// Usage notes
    pub notes: String,
    /// Issues reported
    pub issues: Vec<String>,
}

/// Maintenance record for shared resources
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Maintenance {
    /// Link to resource
    pub resource_hash: ActionHash,
    /// Who performed maintenance
    pub maintainer: AgentPubKey,
    /// Type of maintenance
    pub maintenance_type: MaintenanceType,
    /// Description of work done
    pub description: String,
    /// Cost (if any)
    pub cost: Option<i64>,
    /// Time spent (hours)
    pub hours_spent: f64,
    /// Parts used
    pub parts_used: Vec<String>,
    /// Date performed
    pub performed_at: Timestamp,
    /// Next maintenance due
    pub next_due: Option<Timestamp>,
}

/// Types of maintenance
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaintenanceType {
    Routine,
    Repair,
    Cleaning,
    Upgrade,
    SafetyCheck,
    Other(String),
}

// =============================================================================
// NEEDS MATCHING TYPES
// =============================================================================

/// A need that a member has
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Need {
    /// Unique identifier
    pub id: String,
    /// Who has this need
    pub requester: AgentPubKey,
    /// Need category
    pub category: NeedCategory,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Urgency level
    pub urgency: UrgencyLevel,
    /// Is this an emergency?
    pub emergency: bool,
    /// Quantity needed (if applicable)
    pub quantity: Option<u32>,
    /// Location
    pub location: LocationConstraint,
    /// Needed by date
    pub needed_by: Option<Timestamp>,
    /// Willing to reciprocate with
    pub reciprocity_offers: Vec<String>,
    /// Status
    pub status: NeedStatus,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// A offer to give something
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Offer {
    /// Unique identifier
    pub id: String,
    /// Who is offering
    pub offerer: AgentPubKey,
    /// Offer category
    pub category: NeedCategory,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Quantity available
    pub quantity: Option<u32>,
    /// Condition (for items)
    pub condition: Option<String>,
    /// Location
    pub location: LocationConstraint,
    /// Available until
    pub available_until: Option<Timestamp>,
    /// Asking for in return (can be empty for gifts)
    pub asking_for: Vec<String>,
    /// Status
    pub status: OfferStatus,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// Categories for needs and offers
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NeedCategory {
    // Basic Needs
    Food,
    Clothing,
    Housing,
    Healthcare,
    Transportation,

    // Household
    Furniture,
    Appliances,
    Kitchenware,
    Bedding,

    // Family
    BabyItems,
    ChildrensItems,
    PetSupplies,

    // Education
    SchoolSupplies,
    Books,
    Computers,

    // Work
    WorkClothes,
    Tools,
    Equipment,

    // Personal
    Hygiene,
    Medications,

    // Services
    Rides,
    MovingHelp,
    Childcare,
    PetSitting,

    /// Custom category
    Custom(String),
}

/// Need status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NeedStatus {
    Open,
    PartiallyMet,
    Matched,
    Fulfilled,
    Withdrawn,
    Expired,
}

/// Offer status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OfferStatus {
    Available,
    Reserved,
    Claimed,
    Completed,
    Withdrawn,
    Expired,
}

/// A match between a need and an offer
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Match {
    /// Unique identifier
    pub id: String,
    /// Link to the need
    pub need_hash: ActionHash,
    /// Link to the offer
    pub offer_hash: ActionHash,
    /// Who has the need
    pub requester: AgentPubKey,
    /// Who made the offer
    pub offerer: AgentPubKey,
    /// Match status
    pub status: MatchStatus,
    /// Quantity being matched
    pub quantity: Option<u32>,
    /// Notes about the match
    pub notes: String,
    /// When the match was made
    pub matched_at: Timestamp,
    /// Scheduled handoff time
    pub scheduled_handoff: Option<Timestamp>,
    /// Handoff location
    pub handoff_location: Option<String>,
}

/// Match status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MatchStatus {
    Proposed,
    Accepted,
    Scheduled,
    InProgress,
    Completed,
    Cancelled,
}

/// Record of a fulfilled match
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Fulfillment {
    /// Link to the match
    pub match_hash: ActionHash,
    /// Actual quantity given
    pub quantity_given: Option<u32>,
    /// Fulfillment notes
    pub notes: String,
    /// Requester confirmation
    pub requester_confirmed: bool,
    /// Offerer confirmation
    pub offerer_confirmed: bool,
    /// When fulfilled
    pub fulfilled_at: Timestamp,
    /// Optional gratitude message
    pub gratitude_message: Option<String>,
}

// =============================================================================
// GOVERNANCE TYPES
// =============================================================================

/// Circle membership
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Member {
    /// Member's agent public key
    pub agent: AgentPubKey,
    /// Display name
    pub display_name: String,
    /// Profile (link to Identity hApp if available)
    pub identity_hash: Option<ActionHash>,
    /// Member role(s)
    pub roles: Vec<MemberRole>,
    /// When they joined
    pub joined_at: Timestamp,
    /// Membership status
    pub status: MemberStatus,
    /// Endorsements from other members
    pub endorsement_count: u32,
    /// MATL reputation score (from exchanges)
    pub matl_score: Option<f64>,
}

/// Member roles
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MemberRole {
    /// Regular member
    Member,
    /// Can onboard new members
    Steward,
    /// Can manage resources
    ResourceManager,
    /// Can manage finances
    Treasurer,
    /// Can facilitate governance
    Facilitator,
    /// Founding member
    Founder,
    /// Custom role
    Custom(String),
}

/// Member status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MemberStatus {
    /// Prospective, awaiting approval
    Pending,
    /// Full member
    Active,
    /// Temporarily inactive
    Inactive,
    /// Suspended
    Suspended,
    /// Left the circle
    Departed,
}

/// A governance proposal
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Proposal {
    /// Unique identifier
    pub id: String,
    /// Who submitted the proposal
    pub proposer: AgentPubKey,
    /// Proposal title
    pub title: String,
    /// Full description
    pub description: String,
    /// Type of proposal
    pub proposal_type: ProposalType,
    /// If this modifies a rule, which one
    pub modifies_rule: Option<ActionHash>,
    /// Voting method
    pub voting_method: VotingMethod,
    /// Quorum required (percentage of members)
    pub quorum_percent: u8,
    /// Threshold to pass (percentage of votes)
    pub threshold_percent: u8,
    /// Voting start time
    pub voting_starts: Timestamp,
    /// Voting end time
    pub voting_ends: Timestamp,
    /// Current status
    pub status: ProposalStatus,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// Types of proposals
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProposalType {
    /// Add new rule
    AddRule,
    /// Modify existing rule
    ModifyRule,
    /// Remove rule
    RemoveRule,
    /// Change credit limits
    CreditLimitChange,
    /// Add new member
    MemberAdmission,
    /// Change member status
    MemberStatusChange,
    /// Resource policy change
    ResourcePolicy,
    /// General decision
    GeneralDecision,
    /// Emergency action
    Emergency,
    /// Custom proposal type
    Custom(String),
}

/// Voting methods
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VotingMethod {
    /// Simple majority
    Majority,
    /// 2/3 supermajority
    Supermajority,
    /// Everyone must agree
    Consensus,
    /// No objections (silence = consent)
    ConsentBased,
    /// Weighted by contribution
    ContributionWeighted,
}

/// Proposal status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProposalStatus {
    Draft,
    Discussion,
    Voting,
    Passed,
    Failed,
    Implemented,
    Withdrawn,
}

/// A vote on a proposal
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vote {
    /// Link to the proposal
    pub proposal_hash: ActionHash,
    /// Who is voting
    pub voter: AgentPubKey,
    /// The vote
    pub vote: VoteChoice,
    /// Optional reasoning
    pub reasoning: Option<String>,
    /// When the vote was cast
    pub voted_at: Timestamp,
}

/// Vote choices
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VoteChoice {
    Yes,
    No,
    Abstain,
    Block, // For consensus processes
}

/// Circle rule
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Rule {
    /// Unique identifier
    pub id: String,
    /// Rule title
    pub title: String,
    /// Rule text
    pub text: String,
    /// Rule category
    pub category: RuleCategory,
    /// Priority (for conflict resolution)
    pub priority: u8,
    /// Proposal that created this rule
    pub created_by_proposal: ActionHash,
    /// When the rule became active
    pub active_since: Timestamp,
    /// Whether rule is currently active
    pub active: bool,
    /// Superseded by (if replaced)
    pub superseded_by: Option<ActionHash>,
}

/// Rule categories
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RuleCategory {
    Membership,
    Credits,
    Resources,
    Conduct,
    Governance,
    Disputes,
    Privacy,
    Custom(String),
}

// =============================================================================
// BRIDGE TYPES
// =============================================================================

/// External identity link (to Identity hApp)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct IdentityLink {
    /// Local member hash
    pub member_hash: ActionHash,
    /// Identity hApp agent key
    pub identity_agent: AgentPubKey,
    /// DID (if available)
    pub did: Option<String>,
    /// Verification status
    pub verified: bool,
    /// Verification timestamp
    pub verified_at: Option<Timestamp>,
}

/// Dispute for governance resolution
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Dispute {
    /// Unique identifier
    pub id: String,
    /// Who filed the dispute
    pub complainant: AgentPubKey,
    /// Who is being disputed
    pub respondent: AgentPubKey,
    /// Related transaction/exchange/booking
    pub related_hash: Option<ActionHash>,
    /// Dispute type
    pub dispute_type: DisputeType,
    /// Description
    pub description: String,
    /// Evidence (IPFS CIDs)
    pub evidence: Vec<String>,
    /// Status
    pub status: DisputeStatus,
    /// Resolution (if any)
    pub resolution: Option<String>,
    /// Resolved by (governance proposal hash)
    pub resolved_by_proposal: Option<ActionHash>,
    /// Filed timestamp
    pub filed_at: Timestamp,
}

/// Types of disputes
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeType {
    ServiceQuality,
    NonPayment,
    ResourceDamage,
    MismatchedExpectations,
    NoShow,
    RuleViolation,
    Other(String),
}

/// Dispute status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeStatus {
    Filed,
    UnderReview,
    Mediation,
    GovernanceVote,
    Resolved,
    Closed,
}

/// MATL reputation data
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MatlReputation {
    /// Member agent
    pub member: AgentPubKey,
    /// Overall composite score
    pub composite_score: f64,
    /// Quality score (from ratings)
    pub quality_score: f64,
    /// Consistency score (reliability)
    pub consistency_score: f64,
    /// Reputation score (network trust)
    pub reputation_score: f64,
    /// Number of exchanges
    pub exchange_count: u32,
    /// Last updated
    pub updated_at: Timestamp,
}

/// Lightning network settlement record
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LightningSettlement {
    /// Link to credit transaction
    pub transaction_hash: ActionHash,
    /// Amount in satoshis
    pub amount_sats: u64,
    /// Lightning invoice
    pub invoice: String,
    /// Payment hash
    pub payment_hash: String,
    /// Status
    pub status: SettlementStatus,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Settled timestamp
    pub settled_at: Option<Timestamp>,
}

/// Settlement status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SettlementStatus {
    Pending,
    Paid,
    Expired,
    Failed,
}

// =============================================================================
// COMMON TYPES
// =============================================================================

/// Urgency levels
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Urgent,
    Emergency,
}

/// Request status (generic)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Open,
    Matched,
    InProgress,
    Completed,
    Cancelled,
    Expired,
}

/// Availability schedule
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Availability {
    /// Available days of week (0=Sunday, 6=Saturday)
    pub days: Vec<u8>,
    /// Start time (minutes from midnight)
    pub start_minutes: u16,
    /// End time (minutes from midnight)
    pub end_minutes: u16,
    /// Timezone offset from UTC
    pub timezone_offset_minutes: i16,
    /// Exceptions/blackout dates
    pub exceptions: Vec<DateRange>,
    /// Custom notes
    pub notes: Option<String>,
}

/// Date range for exceptions
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DateRange {
    pub start: Timestamp,
    pub end: Timestamp,
    pub reason: Option<String>,
}

/// Location constraint
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LocationConstraint {
    /// Can happen anywhere (remote)
    Remote,
    /// Specific address
    FixedLocation(String),
    /// Within radius of point
    WithinRadius { geohash: String, radius_km: f32 },
    /// At requester's location
    AtRequester,
    /// At provider's location
    AtProvider,
    /// To be determined
    ToBeArranged,
}

// =============================================================================
// UTILITY IMPLEMENTATIONS
// =============================================================================

impl Default for Availability {
    fn default() -> Self {
        Self {
            days: vec![1, 2, 3, 4, 5],  // Monday-Friday
            start_minutes: 540,         // 9:00 AM
            end_minutes: 1020,          // 5:00 PM
            timezone_offset_minutes: 0, // UTC
            exceptions: vec![],
            notes: None,
        }
    }
}

impl Default for SharingModel {
    fn default() -> Self {
        Self {
            free: true,
            deposit: None,
            hourly_rate: 0,
            daily_rate: None,
            accepts_time_credits: true,
            accepts_circle_credits: true,
            circle_hash: None,
        }
    }
}

impl Default for MatlReputation {
    fn default() -> Self {
        Self {
            member: AgentPubKey::from_raw_36(vec![0; 36]),
            composite_score: 0.5,
            quality_score: 0.5,
            consistency_score: 0.5,
            reputation_score: 0.5,
            exchange_count: 0,
            updated_at: Timestamp::from_micros(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_availability_default() {
        let avail = Availability::default();
        assert_eq!(avail.days, vec![1, 2, 3, 4, 5]);
        assert_eq!(avail.start_minutes, 540);
    }

    #[test]
    fn test_sharing_model_default() {
        let model = SharingModel::default();
        assert!(model.free);
        assert!(model.accepts_time_credits);
    }

    #[test]
    fn test_serialization() {
        let category = ServiceCategory::Tutoring;
        let json = serde_json::to_string(&category).unwrap();
        let parsed: ServiceCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(category, parsed);
    }
}
