//! Agent Reputation Integrity Zome
//!
//! Defines entry types for tracking AI agent reputation using MATL principles.
//! This enables Byzantine-resistant trust scoring for civic AI agents.
//!
//! ## Entry Types
//!
//! - `AgentProfile`: Profile for a Symthaea AI agent
//! - `ReputationEvent`: A positive or negative interaction record
//! - `TrustScore`: Computed MATL trust score snapshot
//!
//! ## MATL Integration
//!
//! Trust scores are computed using:
//! - Quality: Response accuracy and helpfulness
//! - Consistency: Predictable, reliable behavior
//! - Reputation: Aggregate feedback from citizens
//!
//! Composite = 0.4·Quality + 0.3·Consistency + 0.3·Reputation

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

/// Agent type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    /// Symthaea AI agent (automated)
    SymthaeaAi,
    /// Human civic servant
    HumanAgent,
    /// Hybrid (AI-assisted human)
    Hybrid,
    /// Government authority
    Authority,
    /// Community validator
    CommunityValidator,
}

impl Default for AgentType {
    fn default() -> Self {
        Self::SymthaeaAi
    }
}

/// Specialization domain for the agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentSpecialization {
    /// Generalist - handles all domains
    General,
    /// Benefits specialist
    Benefits,
    /// Permits specialist
    Permits,
    /// Tax specialist
    Tax,
    /// Voting/elections specialist
    Voting,
    /// Justice/legal specialist
    Justice,
    /// Emergency response specialist
    Emergency,
}

impl Default for AgentSpecialization {
    fn default() -> Self {
        Self::General
    }
}

/// Profile for a civic AI agent
#[hdk_entry_helper]
#[derive(Clone)]
pub struct AgentProfile {
    /// Agent type
    pub agent_type: AgentType,
    /// Display name
    pub name: String,
    /// Specialization domains
    pub specializations: Vec<AgentSpecialization>,
    /// Description of capabilities
    pub description: String,
    /// Whether the agent is currently active
    pub is_active: bool,
    /// Creation timestamp
    pub created_at: u64,
    /// Last activity timestamp
    pub last_active: u64,
    /// Configuration metadata (JSON)
    pub config_metadata: Option<String>,
}

/// Type of reputation event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReputationEventType {
    /// Citizen marked response as helpful
    Helpful,
    /// Citizen marked response as not helpful
    NotHelpful,
    /// Response was accurate (verified)
    Accurate,
    /// Response was inaccurate
    Inaccurate,
    /// Successfully escalated to human
    ProperEscalation,
    /// Failed to escalate when needed
    FailedEscalation,
    /// Fast response time
    FastResponse,
    /// Slow response time
    SlowResponse,
    /// Authority validation (positive)
    AuthorityApproved,
    /// Authority validation (negative)
    AuthorityRejected,
}

impl ReputationEventType {
    /// Get the impact value (positive or negative)
    pub fn impact_value(&self) -> f32 {
        match self {
            Self::Helpful => 0.05,
            Self::NotHelpful => -0.03,
            Self::Accurate => 0.08,
            Self::Inaccurate => -0.10,
            Self::ProperEscalation => 0.03,
            Self::FailedEscalation => -0.08,
            Self::FastResponse => 0.02,
            Self::SlowResponse => -0.01,
            Self::AuthorityApproved => 0.15,
            Self::AuthorityRejected => -0.20,
        }
    }

    /// Is this a positive event?
    pub fn is_positive(&self) -> bool {
        self.impact_value() > 0.0
    }
}

/// A reputation event for an agent
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ReputationEvent {
    /// The agent this event is about
    pub agent_pubkey: AgentPubKey,
    /// Type of event
    pub event_type: ReputationEventType,
    /// Context/notes about the event
    pub context: Option<String>,
    /// Conversation ID if applicable
    pub conversation_id: Option<String>,
    /// Timestamp
    pub timestamp: u64,
    /// Weight multiplier (for authority events)
    pub weight: f32,
}

/// MATL-style trust score snapshot
#[hdk_entry_helper]
#[derive(Clone)]
pub struct TrustScore {
    /// The agent this score is for
    pub agent_pubkey: AgentPubKey,
    /// Quality score (0.0 - 1.0)
    pub quality: f32,
    /// Consistency score (0.0 - 1.0)
    pub consistency: f32,
    /// Reputation score (0.0 - 1.0)
    pub reputation: f32,
    /// Composite MATL score (0.0 - 1.0)
    pub composite: f32,
    /// Total positive events counted
    pub positive_count: u32,
    /// Total negative events counted
    pub negative_count: u32,
    /// Timestamp of this snapshot
    pub computed_at: u64,
    /// Confidence in the score (based on event count)
    pub confidence: f32,
}

impl TrustScore {
    /// Compute composite score using MATL weights
    pub fn compute_composite(quality: f32, consistency: f32, reputation: f32) -> f32 {
        // MATL formula: 0.4·Quality + 0.3·Consistency + 0.3·Reputation
        (0.4 * quality + 0.3 * consistency + 0.3 * reputation).clamp(0.0, 1.0)
    }

    /// Check if the score meets Byzantine tolerance threshold
    pub fn is_trustworthy(&self) -> bool {
        // MATL 45% Byzantine tolerance threshold
        self.composite >= 0.55 && self.confidence >= 0.3
    }
}

/// Link types for the agent reputation zome
#[hdk_link_types]
pub enum AgentReputationLinkTypes {
    /// Links from agent pubkey to their profile
    AgentToProfile,
    /// Links from agent to their reputation events
    AgentToReputationEvent,
    /// Links from agent to their trust scores
    AgentToTrustScore,
    /// Links agents by specialization
    SpecializationToAgent,
    /// Links for active agents
    ActiveAgents,
}

/// Anchor entry for path-based linking
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

/// Entry types for this zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    AgentProfile(AgentProfile),
    #[entry_type(visibility = "public")]
    ReputationEvent(ReputationEvent),
    #[entry_type(visibility = "public")]
    TrustScore(TrustScore),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

/// Genesis validation
#[hdk_extern]
pub fn genesis_self_check(_: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Validate operations
#[hdk_extern]
pub fn validate(_: Op) -> ExternResult<ValidateCallbackResult> {
    // TODO: Add proper validation rules
    // - AgentProfile: Validate name length, required fields
    // - ReputationEvent: Validate weight range, timestamp
    // - TrustScore: Validate score ranges (0.0 - 1.0)
    Ok(ValidateCallbackResult::Valid)
}
