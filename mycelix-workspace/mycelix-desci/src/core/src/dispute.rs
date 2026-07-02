// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dispute Resolution System
//!
//! Implements the dispute resolution process from Epistemic Charter §5.
//! Provides mechanisms for challenging claims, presenting evidence,
//! and reaching resolution through arbitration.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of challenge being made against a claim
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChallengeType {
    /// Factual inaccuracy - the data or conclusions are wrong
    Factual,
    /// Methodological issues - flawed study design or analysis
    Methodological,
    /// Ethical concerns - research ethics violations
    Ethical,
    /// Reproducibility failure - cannot replicate results
    Reproducibility,
    /// Plagiarism or attribution issues
    Attribution,
    /// Conflict of interest not disclosed
    ConflictOfInterest,
    /// Data integrity issues (fabrication, falsification)
    DataIntegrity,
    /// Scope creep - claim extends beyond evidence
    OverClaim,
}

impl ChallengeType {
    pub fn severity(&self) -> u8 {
        match self {
            Self::OverClaim => 1,
            Self::Attribution => 2,
            Self::ConflictOfInterest => 3,
            Self::Methodological => 4,
            Self::Reproducibility => 5,
            Self::Factual => 6,
            Self::Ethical => 7,
            Self::DataIntegrity => 8,
        }
    }

    pub fn requires_expert_review(&self) -> bool {
        self.severity() >= 4
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Factual => "Factual inaccuracy in data or conclusions",
            Self::Methodological => "Flawed study design or analysis",
            Self::Ethical => "Research ethics violations",
            Self::Reproducibility => "Results cannot be reproduced",
            Self::Attribution => "Plagiarism or attribution issues",
            Self::ConflictOfInterest => "Undisclosed conflict of interest",
            Self::DataIntegrity => "Data fabrication or falsification",
            Self::OverClaim => "Conclusions exceed the evidence",
        }
    }
}

/// Current status of a dispute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisputeStatus {
    /// Just filed, awaiting initial review
    Filed,
    /// Under initial screening
    Screening,
    /// Awaiting response from claim author
    AwaitingResponse,
    /// Under expert review by audit guild
    UnderReview,
    /// In arbitration phase
    Arbitration,
    /// Resolved with decision
    Resolved,
    /// Dismissed (invalid challenge)
    Dismissed,
    /// Withdrawn by challenger
    Withdrawn,
    /// Appealed after resolution
    Appealed,
}

impl DisputeStatus {
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Filed
                | Self::Screening
                | Self::AwaitingResponse
                | Self::UnderReview
                | Self::Arbitration
                | Self::Appealed
        )
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Resolved | Self::Dismissed | Self::Withdrawn)
    }
}

/// Outcome of a resolved dispute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionOutcome {
    /// Challenge upheld - claim is invalid/needs correction
    ChallengeUpheld,
    /// Challenge partially upheld - minor issues found
    PartiallyUpheld,
    /// Challenge rejected - claim stands
    ChallengeRejected,
    /// Insufficient evidence to decide either way
    Inconclusive,
    /// Both parties reached agreement
    MutualResolution,
}

/// Resolution details for a dispute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    /// Outcome of the dispute
    pub outcome: ResolutionOutcome,
    /// Detailed explanation of the decision
    pub explanation: String,
    /// Required actions (if any)
    pub required_actions: Vec<RequiredAction>,
    /// Timestamp of resolution
    pub resolved_at: DateTime<Utc>,
    /// Who made the final decision
    pub resolved_by: Vec<String>,
    /// Voting breakdown (if applicable)
    pub votes: Option<VotingRecord>,
}

/// Required action after dispute resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredAction {
    /// What action is required
    pub action_type: ActionType,
    /// Description of the action
    pub description: String,
    /// Deadline for the action
    pub deadline: Option<DateTime<Utc>>,
    /// Whether it's been completed
    pub completed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    /// Claim must be corrected
    Correction,
    /// Claim must be retracted
    Retraction,
    /// Additional disclosure required
    Disclosure,
    /// Apology or acknowledgment
    Acknowledgment,
    /// Data must be released
    DataRelease,
    /// Methodology clarification
    Clarification,
    /// No action required
    None,
}

/// Record of voting in arbitration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingRecord {
    /// Votes for upholding the challenge
    pub votes_for: usize,
    /// Votes against the challenge
    pub votes_against: usize,
    /// Abstentions
    pub abstentions: usize,
    /// Individual votes (arbiter -> vote)
    pub individual_votes: Vec<(String, Vote)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Vote {
    For,
    Against,
    Abstain,
}

/// Evidence submitted in a dispute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Unique ID for this evidence
    pub id: Uuid,
    /// Who submitted it
    pub submitted_by: String,
    /// Type of evidence
    pub evidence_type: EvidenceType,
    /// Description of the evidence
    pub description: String,
    /// Reference to supporting claims
    pub supporting_claims: Vec<Uuid>,
    /// IPFS or storage reference
    pub storage_ref: Option<String>,
    /// Hash of evidence content
    pub content_hash: String,
    /// When submitted
    pub submitted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Published paper or preprint
    Publication,
    /// Raw data
    Data,
    /// Analysis code or methodology
    Code,
    /// Expert testimony
    ExpertTestimony,
    /// Replication study results
    ReplicationStudy,
    /// Communication records
    Communication,
    /// Statistical reanalysis
    Reanalysis,
    /// Other documentation
    Document,
}

/// A dispute against a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dispute {
    /// Unique dispute ID
    pub id: Uuid,
    /// ID of the claim being challenged
    pub claim_id: Uuid,
    /// Who filed the challenge
    pub challenger: String,
    /// Type of challenge
    pub challenge_type: ChallengeType,
    /// Detailed description of the challenge
    pub challenge_description: String,
    /// Evidence supporting the challenge
    pub challenger_evidence: Vec<Evidence>,
    /// Response from claim author
    pub author_response: Option<AuthorResponse>,
    /// Current status
    pub status: DisputeStatus,
    /// Assigned arbiters (audit guild members)
    pub arbiters: Vec<String>,
    /// Resolution (if resolved)
    pub resolution: Option<Resolution>,
    /// Timeline of events
    pub timeline: Vec<DisputeEvent>,
    /// When filed
    pub filed_at: DateTime<Utc>,
    /// Deadline for response
    pub response_deadline: DateTime<Utc>,
    /// Deadline for resolution
    pub resolution_deadline: Option<DateTime<Utc>>,
}

/// Response from the claim author
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorResponse {
    /// The response text
    pub response: String,
    /// Evidence provided in defense
    pub evidence: Vec<Evidence>,
    /// When responded
    pub responded_at: DateTime<Utc>,
    /// Whether author acknowledges any issues
    pub acknowledges_issues: bool,
    /// Proposed resolution (if any)
    pub proposed_resolution: Option<String>,
}

/// Event in the dispute timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisputeEvent {
    /// When it happened
    pub timestamp: DateTime<Utc>,
    /// What happened
    pub event_type: DisputeEventType,
    /// Who did it
    pub actor: String,
    /// Description
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisputeEventType {
    Filed,
    EvidenceSubmitted,
    ResponseReceived,
    ArbiterAssigned,
    StatusChanged,
    CommentAdded,
    VoteCast,
    Resolved,
    Appealed,
}

impl Dispute {
    /// Create a new dispute
    pub fn new(
        claim_id: Uuid,
        challenger: String,
        challenge_type: ChallengeType,
        description: String,
    ) -> Self {
        let now = Utc::now();
        let id = Uuid::new_v4();

        let initial_event = DisputeEvent {
            timestamp: now,
            event_type: DisputeEventType::Filed,
            actor: challenger.clone(),
            description: format!("Dispute filed: {}", challenge_type.description()),
        };

        Self {
            id,
            claim_id,
            challenger,
            challenge_type,
            challenge_description: description,
            challenger_evidence: Vec::new(),
            author_response: None,
            status: DisputeStatus::Filed,
            arbiters: Vec::new(),
            resolution: None,
            timeline: vec![initial_event],
            filed_at: now,
            response_deadline: now + Duration::days(14), // 2 weeks to respond
            resolution_deadline: None,
        }
    }

    /// Add evidence to the dispute
    pub fn add_evidence(&mut self, evidence: Evidence) {
        let event = DisputeEvent {
            timestamp: Utc::now(),
            event_type: DisputeEventType::EvidenceSubmitted,
            actor: evidence.submitted_by.clone(),
            description: format!("Evidence submitted: {:?}", evidence.evidence_type),
        };
        self.challenger_evidence.push(evidence);
        self.timeline.push(event);
    }

    /// Record author's response
    pub fn respond(&mut self, response: AuthorResponse) {
        let event = DisputeEvent {
            timestamp: Utc::now(),
            event_type: DisputeEventType::ResponseReceived,
            actor: "claim_author".to_string(),
            description: "Author response received".to_string(),
        };
        self.author_response = Some(response);
        self.status = DisputeStatus::UnderReview;
        self.timeline.push(event);
    }

    /// Assign arbiters to the dispute
    pub fn assign_arbiters(&mut self, arbiters: Vec<String>) {
        for arbiter in &arbiters {
            let event = DisputeEvent {
                timestamp: Utc::now(),
                event_type: DisputeEventType::ArbiterAssigned,
                actor: "system".to_string(),
                description: format!("Arbiter assigned: {}", arbiter),
            };
            self.timeline.push(event);
        }
        self.arbiters = arbiters;
        self.status = DisputeStatus::Arbitration;
        self.resolution_deadline = Some(Utc::now() + Duration::days(30));
    }

    /// Resolve the dispute
    pub fn resolve(&mut self, resolution: Resolution) {
        let event = DisputeEvent {
            timestamp: Utc::now(),
            event_type: DisputeEventType::Resolved,
            actor: resolution.resolved_by.join(", "),
            description: format!("Dispute resolved: {:?}", resolution.outcome),
        };
        self.resolution = Some(resolution);
        self.status = DisputeStatus::Resolved;
        self.timeline.push(event);
    }

    /// Check if response deadline has passed
    pub fn is_response_overdue(&self) -> bool {
        self.author_response.is_none() && Utc::now() > self.response_deadline
    }

    /// Get the duration since filing
    pub fn age(&self) -> Duration {
        Utc::now() - self.filed_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dispute() {
        let claim_id = Uuid::new_v4();
        let dispute = Dispute::new(
            claim_id,
            "challenger@test.com".to_string(),
            ChallengeType::Methodological,
            "Insufficient sample size".to_string(),
        );

        assert_eq!(dispute.status, DisputeStatus::Filed);
        assert_eq!(dispute.claim_id, claim_id);
        assert!(dispute.timeline.len() == 1);
    }

    #[test]
    fn test_challenge_severity() {
        assert!(ChallengeType::DataIntegrity.severity() > ChallengeType::OverClaim.severity());
        assert!(ChallengeType::DataIntegrity.requires_expert_review());
        assert!(!ChallengeType::OverClaim.requires_expert_review());
    }

    #[test]
    fn test_add_evidence() {
        let mut dispute = Dispute::new(
            Uuid::new_v4(),
            "challenger@test.com".to_string(),
            ChallengeType::Reproducibility,
            "Cannot replicate Figure 3".to_string(),
        );

        let evidence = Evidence {
            id: Uuid::new_v4(),
            submitted_by: "challenger@test.com".to_string(),
            evidence_type: EvidenceType::ReplicationStudy,
            description: "Our replication attempt".to_string(),
            supporting_claims: vec![],
            storage_ref: Some("ipfs://Qm...".to_string()),
            content_hash: "abc123".to_string(),
            submitted_at: Utc::now(),
        };

        dispute.add_evidence(evidence);
        assert_eq!(dispute.challenger_evidence.len(), 1);
        assert_eq!(dispute.timeline.len(), 2);
    }

    #[test]
    fn test_dispute_resolution() {
        let mut dispute = Dispute::new(
            Uuid::new_v4(),
            "challenger@test.com".to_string(),
            ChallengeType::Factual,
            "Data error in Table 1".to_string(),
        );

        let resolution = Resolution {
            outcome: ResolutionOutcome::ChallengeUpheld,
            explanation: "Error confirmed in Table 1".to_string(),
            required_actions: vec![RequiredAction {
                action_type: ActionType::Correction,
                description: "Correct Table 1 values".to_string(),
                deadline: Some(Utc::now() + Duration::days(7)),
                completed: false,
            }],
            resolved_at: Utc::now(),
            resolved_by: vec!["arbiter1@audit.org".to_string()],
            votes: None,
        };

        dispute.resolve(resolution);
        assert_eq!(dispute.status, DisputeStatus::Resolved);
        assert!(dispute.resolution.is_some());
    }

    #[test]
    fn test_dispute_status() {
        assert!(DisputeStatus::Filed.is_active());
        assert!(DisputeStatus::Resolved.is_terminal());
        assert!(!DisputeStatus::UnderReview.is_terminal());
    }
}
