//! Collective Sensemaking Integrity Zome
//!
//! Entry types for distributed belief sharing, consensus, and emergent truth discovery.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

/// Anonymized belief share - shared version of a thought without personal identifiers
#[hdk_entry_helper]
#[derive(Clone)]
pub struct BeliefShare {
    /// Hash of the original thought (for deduplication)
    pub content_hash: String,
    /// The belief content (may be summarized/anonymized)
    pub content: String,
    /// Thought type
    pub belief_type: String,
    /// E/N/M/H classification
    pub epistemic_code: String,
    /// Confidence level
    pub confidence: f64,
    /// Domain/topic tags
    pub tags: Vec<String>,
    /// Timestamp of share
    pub shared_at: Timestamp,
    /// Optional: linked evidence hashes
    pub evidence_hashes: Vec<String>,
    /// HDC embedding for semantic similarity (16,384 dimensions, stored as Vec<f32>)
    /// This enables semantic pattern detection across the collective
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// Stance on this belief (if applicable)
    #[serde(default)]
    pub stance: Option<BeliefStance>,
}

/// Stance on a belief (for collective sensemaking)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BeliefStance {
    /// Strongly support this belief
    StronglyAgree,
    /// Agree with this belief
    Agree,
    /// Neutral/observational
    Neutral,
    /// Disagree with this belief
    Disagree,
    /// Strongly oppose this belief
    StronglyDisagree,
}

/// Validation vote on a shared belief
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ValidationVote {
    /// Reference to the belief share
    pub belief_share_hash: ActionHash,
    /// Vote type
    pub vote_type: ValidationVoteType,
    /// Optional evidence provided
    pub evidence: Option<String>,
    /// Voter's epistemic weight (from reputation)
    pub voter_weight: f64,
    /// Timestamp
    pub voted_at: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ValidationVoteType {
    /// Voter has independent evidence supporting this
    Corroborate,
    /// Voter has evidence contradicting this
    Contradict,
    /// Voter finds this plausible but unverified
    Plausible,
    /// Voter finds this implausible
    Implausible,
    /// Voter cannot assess
    Abstain,
}

/// Consensus record when collective agreement is reached
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ConsensusRecord {
    /// The belief share this consensus is about
    pub belief_share_hash: ActionHash,
    /// Consensus type achieved
    pub consensus_type: ConsensusType,
    /// Number of validators
    pub validator_count: u32,
    /// Weighted agreement score (0-1)
    pub agreement_score: f64,
    /// Summary of the consensus
    pub summary: String,
    /// When consensus was reached
    pub reached_at: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConsensusType {
    /// Strong agreement (>80% weighted)
    StrongConsensus,
    /// Moderate agreement (60-80% weighted)
    ModerateConsensus,
    /// Weak agreement (40-60% weighted)
    WeakConsensus,
    /// Contested (significant disagreement)
    Contested,
    /// Insufficient data
    Insufficient,
}

/// Emergent pattern detected across multiple beliefs
#[hdk_entry_helper]
#[derive(Clone)]
pub struct EmergentPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Description of the pattern
    pub description: String,
    /// Belief shares that form this pattern
    pub belief_hashes: Vec<ActionHash>,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Confidence in the pattern
    pub confidence: f64,
    /// When detected
    pub detected_at: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PatternType {
    /// Multiple independent sources converging
    Convergence,
    /// Systematic disagreement pattern
    Divergence,
    /// Temporal trend
    Trend,
    /// Cluster of related beliefs
    Cluster,
    /// Contradiction cluster
    ContradictionCluster,
}

/// Epistemic reputation for validators
#[hdk_entry_helper]
#[derive(Clone)]
pub struct EpistemicReputation {
    /// Agent this reputation is for
    pub agent: AgentPubKey,
    /// Domains of expertise
    pub domains: Vec<String>,
    /// Accuracy score (predictions that came true)
    pub accuracy_score: f64,
    /// Calibration score (confidence vs actual accuracy)
    pub calibration_score: f64,
    /// Total validations performed
    pub validation_count: u32,
    /// Last updated
    pub updated_at: Timestamp,
}

/// Relationship between two agents in the collective
///
/// Tracks trust, interaction history, and relationship quality for
/// weighted consensus calculations.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct AgentRelationship {
    /// The other agent in this relationship (relative to author)
    pub other_agent: AgentPubKey,
    /// Trust score (0.0-1.0) - how much we trust this agent's beliefs
    pub trust_score: f64,
    /// Number of interactions with this agent
    pub interaction_count: u32,
    /// Last interaction timestamp
    pub last_interaction: Timestamp,
    /// Current relationship stage
    pub relationship_stage: RelationshipStage,
    /// Domains where we've collaborated
    pub shared_domains: Vec<String>,
    /// Agreement ratio (how often we've agreed on votes)
    pub agreement_ratio: f64,
}

/// Stages of relationship development
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RelationshipStage {
    /// No established relationship
    NoRelation,
    /// Basic familiarity from occasional interactions
    Acquaintance,
    /// Regular collaboration, some trust established
    Collaborator,
    /// High trust, frequently align on beliefs
    TrustedPeer,
    /// Deep trust, often seek each other's validation
    PartnerInTruth,
}

/// Link types for the collective zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Tag to belief shares
    TagToBeliefShare,
    /// Belief share to votes
    BeliefShareToVotes,
    /// Belief share to consensus
    BeliefShareToConsensus,
    /// Pattern to beliefs
    PatternToBeliefs,
    /// Agent to reputation
    AgentToReputation,
    /// All belief shares (anchor)
    AllBeliefShares,
    /// All patterns (anchor)
    AllPatterns,
    /// Agent to their relationships with other agents
    AgentToRelationship,
}

/// Entry types
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "Anchor", visibility = "public")]
    Anchor(Anchor),
    #[entry_type(name = "BeliefShare", visibility = "public")]
    BeliefShare(BeliefShare),
    #[entry_type(name = "ValidationVote", visibility = "public")]
    ValidationVote(ValidationVote),
    #[entry_type(name = "ConsensusRecord", visibility = "public")]
    ConsensusRecord(ConsensusRecord),
    #[entry_type(name = "EmergentPattern", visibility = "public")]
    EmergentPattern(EmergentPattern),
    #[entry_type(name = "EpistemicReputation", visibility = "public")]
    EpistemicReputation(EpistemicReputation),
    #[entry_type(name = "AgentRelationship", visibility = "private")]
    AgentRelationship(AgentRelationship),
}

/// Validation callbacks
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::BeliefShare(share) => validate_belief_share(share),
                EntryTypes::ValidationVote(vote) => validate_vote(vote),
                EntryTypes::ConsensusRecord(consensus) => validate_consensus(consensus),
                EntryTypes::EmergentPattern(pattern) => validate_pattern(pattern),
                EntryTypes::EpistemicReputation(rep) => validate_reputation(rep),
                EntryTypes::AgentRelationship(rel) => validate_relationship(rel),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_belief_share(share: BeliefShare) -> ExternResult<ValidateCallbackResult> {
    // Content must not be empty
    if share.content.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Belief share content cannot be empty".into()));
    }

    // Confidence must be 0-1
    if share.confidence < 0.0 || share.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Confidence must be between 0 and 1".into()));
    }

    // Must have at least one tag
    if share.tags.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Belief share must have at least one tag".into()));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_vote(vote: ValidationVote) -> ExternResult<ValidateCallbackResult> {
    // Weight must be positive
    if vote.voter_weight < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Voter weight cannot be negative".into()));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_consensus(consensus: ConsensusRecord) -> ExternResult<ValidateCallbackResult> {
    // Agreement score must be 0-1
    if consensus.agreement_score < 0.0 || consensus.agreement_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Agreement score must be between 0 and 1".into()));
    }

    // Must have at least 3 validators for meaningful consensus
    if consensus.validator_count < 3 {
        return Ok(ValidateCallbackResult::Invalid("Consensus requires at least 3 validators".into()));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_pattern(pattern: EmergentPattern) -> ExternResult<ValidateCallbackResult> {
    // Must involve multiple beliefs
    if pattern.belief_hashes.len() < 2 {
        return Ok(ValidateCallbackResult::Invalid("Pattern must involve at least 2 beliefs".into()));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_reputation(rep: EpistemicReputation) -> ExternResult<ValidateCallbackResult> {
    // Scores must be 0-1
    if rep.accuracy_score < 0.0 || rep.accuracy_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Accuracy score must be between 0 and 1".into()));
    }

    if rep.calibration_score < 0.0 || rep.calibration_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Calibration score must be between 0 and 1".into()));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_relationship(rel: AgentRelationship) -> ExternResult<ValidateCallbackResult> {
    // Trust score must be 0-1
    if rel.trust_score < 0.0 || rel.trust_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Trust score must be between 0 and 1".into()));
    }

    // Agreement ratio must be 0-1
    if rel.agreement_ratio < 0.0 || rel.agreement_ratio > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Agreement ratio must be between 0 and 1".into()));
    }

    Ok(ValidateCallbackResult::Valid)
}
