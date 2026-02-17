//! Proposal Management
//!
//! Handles proposal creation, validation, lifecycle transitions,
//! and execution coordination.

use super::types::*;
use super::{GovernanceError, GovernanceResult};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Proposal manager handling lifecycle and validation
#[derive(Debug, Default)]
pub struct ProposalManager {
    /// Active proposals by ID
    proposals: HashMap<String, Proposal>,

    /// Proposal history (archived)
    archived: HashMap<String, Proposal>,

    /// Governance configuration
    config: GovernanceConfig,

    /// Last proposal time per proposer
    last_proposal: HashMap<String, DateTime<Utc>>,
}

impl ProposalManager {
    /// Create a new proposal manager
    pub fn new(config: GovernanceConfig) -> Self {
        Self {
            proposals: HashMap::new(),
            archived: HashMap::new(),
            config,
            last_proposal: HashMap::new(),
        }
    }

    /// Create and register a new proposal
    pub fn create_proposal(
        &mut self,
        id: impl Into<String>,
        title: impl Into<String>,
        description: impl Into<String>,
        proposal_type: ProposalType,
        proposer_did: impl Into<String>,
        proposer_matl: f32,
        proposer_stake: f32,
    ) -> GovernanceResult<&Proposal> {
        let id = id.into();
        let proposer_did = proposer_did.into();

        // Check if proposal ID already exists
        if self.proposals.contains_key(&id) || self.archived.contains_key(&id) {
            return Err(GovernanceError::InvalidProposal(format!(
                "Proposal ID {} already exists",
                id
            )));
        }

        // Validate proposer eligibility
        self.validate_proposer(&proposer_did, proposer_matl, proposer_stake)?;

        // Check cooldown
        if let Some(last) = self.last_proposal.get(&proposer_did) {
            let elapsed = Utc::now() - *last;
            if elapsed < self.config.proposal_cooldown {
                return Err(GovernanceError::InvalidProposal(format!(
                    "Proposer must wait {:?} before creating another proposal",
                    self.config.proposal_cooldown - elapsed
                )));
            }
        }

        // Create proposal
        let proposal = Proposal::new(id.clone(), title, description, proposal_type, &proposer_did);

        // Register
        self.proposals.insert(id.clone(), proposal);
        self.last_proposal.insert(proposer_did, Utc::now());

        Ok(self.proposals.get(&id).unwrap())
    }

    /// Validate proposer eligibility
    fn validate_proposer(
        &self,
        _proposer_did: &str,
        matl_score: f32,
        stake: f32,
    ) -> GovernanceResult<()> {
        if matl_score < self.config.min_proposal_matl {
            return Err(GovernanceError::InsufficientEligibility(format!(
                "MATL score {} below minimum {}",
                matl_score, self.config.min_proposal_matl
            )));
        }

        if stake < self.config.min_proposal_stake {
            return Err(GovernanceError::InsufficientEligibility(format!(
                "Stake {} below minimum {}",
                stake, self.config.min_proposal_stake
            )));
        }

        Ok(())
    }

    /// Get a proposal by ID
    pub fn get_proposal(&self, id: &str) -> GovernanceResult<&Proposal> {
        self.proposals
            .get(id)
            .or_else(|| self.archived.get(id))
            .ok_or_else(|| GovernanceError::ProposalNotFound(id.to_string()))
    }

    /// Get a mutable proposal by ID
    pub fn get_proposal_mut(&mut self, id: &str) -> GovernanceResult<&mut Proposal> {
        self.proposals
            .get_mut(id)
            .ok_or_else(|| GovernanceError::ProposalNotFound(id.to_string()))
    }

    /// Add a sponsor to a proposal
    pub fn add_sponsor(
        &mut self,
        proposal_id: &str,
        sponsor_did: impl Into<String>,
        sponsor_matl: f32,
    ) -> GovernanceResult<()> {
        // Extract config value before mutable borrow
        let min_sponsor_matl = self.config.min_proposal_matl * 0.8;

        // Sponsors need reasonable MATL score
        if sponsor_matl < min_sponsor_matl {
            return Err(GovernanceError::InsufficientEligibility(
                "Sponsor MATL score too low".to_string(),
            ));
        }

        let proposal = self.get_proposal_mut(proposal_id)?;

        if proposal.status != ProposalStatus::Draft {
            return Err(GovernanceError::InvalidProposal(
                "Can only add sponsors during draft phase".to_string(),
            ));
        }

        proposal.add_sponsor(sponsor_did);
        Ok(())
    }

    /// Submit proposal for discussion
    pub fn submit_for_discussion(&mut self, proposal_id: &str) -> GovernanceResult<()> {
        // Extract config value before mutable borrow
        let required_sponsors = self.config.constitutional_sponsors as usize;

        let proposal = self.get_proposal_mut(proposal_id)?;

        if proposal.status != ProposalStatus::Draft {
            return Err(GovernanceError::InvalidProposal(
                "Proposal must be in draft status".to_string(),
            ));
        }

        // Check sponsor requirements for constitutional proposals
        if proposal.proposal_type == ProposalType::Constitutional {
            if proposal.sponsors.len() < required_sponsors {
                return Err(GovernanceError::InvalidProposal(format!(
                    "Constitutional proposals require {} sponsors, have {}",
                    required_sponsors,
                    proposal.sponsors.len()
                )));
            }
        }

        proposal.start_discussion();
        Ok(())
    }

    /// Transition proposal to voting phase
    pub fn start_voting(&mut self, proposal_id: &str) -> GovernanceResult<()> {
        let proposal = self.get_proposal_mut(proposal_id)?;

        if proposal.status != ProposalStatus::Discussion {
            return Err(GovernanceError::InvalidProposal(
                "Proposal must be in discussion phase".to_string(),
            ));
        }

        // Check if discussion period is complete
        if let Some(voting_starts) = proposal.voting_starts {
            if Utc::now() < voting_starts {
                return Err(GovernanceError::InvalidProposal(
                    "Discussion period not yet complete".to_string(),
                ));
            }
        }

        proposal.start_voting();
        Ok(())
    }

    /// End voting and transition to tallying
    pub fn end_voting(&mut self, proposal_id: &str) -> GovernanceResult<()> {
        let proposal = self.get_proposal_mut(proposal_id)?;

        if proposal.status != ProposalStatus::Voting {
            return Err(GovernanceError::VotingNotActive(proposal_id.to_string()));
        }

        if !proposal.voting_ended() {
            return Err(GovernanceError::InvalidProposal(
                "Voting period not yet ended".to_string(),
            ));
        }

        proposal.status = ProposalStatus::Tallying;
        Ok(())
    }

    /// Finalize proposal with tally results
    pub fn finalize_proposal(
        &mut self,
        proposal_id: &str,
        tally: &VoteTally,
    ) -> GovernanceResult<ProposalStatus> {
        let proposal = self.get_proposal_mut(proposal_id)?;

        if proposal.status != ProposalStatus::Tallying {
            return Err(GovernanceError::InvalidProposal(
                "Proposal must be in tallying phase".to_string(),
            ));
        }

        // Check quorum
        let quorum_threshold = proposal.proposal_type.quorum_threshold();
        if tally.participation_rate() < quorum_threshold {
            proposal.status = ProposalStatus::Expired;
            return Err(GovernanceError::QuorumNotReached {
                required: quorum_threshold,
                actual: tally.participation_rate(),
            });
        }

        // Check approval
        let approval_threshold = proposal.proposal_type.approval_threshold();
        if tally.approval_rate() >= approval_threshold {
            // Set execution time based on timelock
            let timelock = proposal.proposal_type.timelock_duration();
            proposal.execution_time = Some(Utc::now() + timelock);
            proposal.status = ProposalStatus::Approved;
        } else {
            proposal.status = ProposalStatus::Rejected;
        }

        Ok(proposal.status)
    }

    /// Apply guardian veto
    pub fn apply_veto(
        &mut self,
        proposal_id: &str,
        guardian_did: &str,
        reason: &str,
    ) -> GovernanceResult<()> {
        // Verify guardian
        if !self.config.guardians.contains(&guardian_did.to_string()) {
            return Err(GovernanceError::GuardianVeto(
                "Not a registered guardian".to_string(),
            ));
        }

        let proposal = self.get_proposal_mut(proposal_id)?;

        // Can only veto proposals requiring guardian approval during timelock
        if !proposal.proposal_type.requires_guardian_approval() {
            return Err(GovernanceError::GuardianVeto(
                "This proposal type does not require guardian approval".to_string(),
            ));
        }

        if proposal.status != ProposalStatus::Approved {
            return Err(GovernanceError::GuardianVeto(
                "Can only veto approved proposals in timelock".to_string(),
            ));
        }

        proposal.status = ProposalStatus::Vetoed;
        proposal.metadata.insert(
            "veto_reason".to_string(),
            serde_json::Value::String(reason.to_string()),
        );
        proposal.metadata.insert(
            "vetoed_by".to_string(),
            serde_json::Value::String(guardian_did.to_string()),
        );

        Ok(())
    }

    /// Execute an approved proposal
    pub fn execute_proposal(&mut self, proposal_id: &str) -> GovernanceResult<Vec<ActionResult>> {
        // First, validate and get required data without holding mutable borrow
        let (actions, execution_time, status) = {
            let proposal = self.get_proposal(proposal_id)?;
            (
                proposal.actions.clone(),
                proposal.execution_time,
                proposal.status,
            )
        };

        if status != ProposalStatus::Approved {
            return Err(GovernanceError::ExecutionError(
                "Proposal not approved".to_string(),
            ));
        }

        // Check timelock
        if let Some(exec_time) = execution_time {
            if Utc::now() < exec_time {
                return Err(GovernanceError::TimelockActive(format!(
                    "Execution available at {}",
                    exec_time
                )));
            }
        }

        // Execute actions
        let mut results = Vec::new();
        let mut failed = false;
        for action in &actions {
            let result = Self::execute_action_static(action);
            results.push(result.clone());

            // Stop on required action failure
            if !action.optional && !result.success {
                failed = true;
                break;
            }
        }

        // Update status
        let proposal = self.get_proposal_mut(proposal_id)?;
        proposal.status = if failed {
            ProposalStatus::Failed
        } else {
            ProposalStatus::Executed
        };

        Ok(results)
    }

    /// Execute a single action (placeholder - actual execution depends on action type)
    fn execute_action_static(action: &ProposalAction) -> ActionResult {
        // In a real implementation, this would dispatch to appropriate handlers
        ActionResult {
            action_type: action.action_type.clone(),
            success: true,
            message: format!("Executed {:?} on {}", action.action_type, action.target),
            data: None,
        }
    }

    /// Cancel a proposal (by proposer)
    pub fn cancel_proposal(&mut self, proposal_id: &str, requester_did: &str) -> GovernanceResult<()> {
        let proposal = self.get_proposal_mut(proposal_id)?;

        if proposal.proposer_did != requester_did {
            return Err(GovernanceError::InvalidProposal(
                "Only proposer can cancel".to_string(),
            ));
        }

        if proposal.status.is_final() {
            return Err(GovernanceError::InvalidProposal(
                "Cannot cancel finalized proposal".to_string(),
            ));
        }

        // Can only cancel before voting ends
        if proposal.status == ProposalStatus::Voting && proposal.voting_ended() {
            return Err(GovernanceError::InvalidProposal(
                "Cannot cancel after voting ended".to_string(),
            ));
        }

        proposal.status = ProposalStatus::Cancelled;
        Ok(())
    }

    /// Archive completed proposals
    pub fn archive_finalized(&mut self) {
        let finalized: Vec<String> = self
            .proposals
            .iter()
            .filter(|(_, p)| p.status.is_final())
            .map(|(id, _)| id.clone())
            .collect();

        for id in finalized {
            if let Some(proposal) = self.proposals.remove(&id) {
                self.archived.insert(id, proposal);
            }
        }
    }

    /// List active proposals
    pub fn list_active(&self) -> Vec<&Proposal> {
        self.proposals
            .values()
            .filter(|p| p.status.is_active())
            .collect()
    }

    /// List proposals by status
    pub fn list_by_status(&self, status: ProposalStatus) -> Vec<&Proposal> {
        self.proposals
            .values()
            .filter(|p| p.status == status)
            .collect()
    }

    /// List proposals needing action (status transition)
    pub fn list_needing_transition(&self) -> Vec<(&str, ProposalStatus)> {
        let now = Utc::now();
        let mut needs_transition = Vec::new();

        for (id, proposal) in &self.proposals {
            match proposal.status {
                ProposalStatus::Discussion => {
                    if let Some(voting_starts) = proposal.voting_starts {
                        if now >= voting_starts {
                            needs_transition.push((id.as_str(), ProposalStatus::Voting));
                        }
                    }
                }
                ProposalStatus::Voting => {
                    if proposal.voting_ended() {
                        needs_transition.push((id.as_str(), ProposalStatus::Tallying));
                    }
                }
                _ => {}
            }
        }

        needs_transition
    }
}

/// Result of executing a proposal action
#[derive(Debug, Clone)]
pub struct ActionResult {
    pub action_type: ActionType,
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GovernanceConfig {
        GovernanceConfig {
            min_proposal_matl: 0.5,
            min_proposal_stake: 100.0,
            constitutional_sponsors: 3,
            guardians: vec!["did:mycelix:guardian1".to_string()],
            delegation_enabled: true,
            max_delegation_depth: 3,
            proposal_cooldown: chrono::Duration::seconds(0), // No cooldown for tests
        }
    }

    #[test]
    fn test_create_proposal() {
        let mut manager = ProposalManager::new(test_config());

        let result = manager.create_proposal(
            "MIP-0001",
            "Test Proposal",
            "Description",
            ProposalType::Standard,
            "did:mycelix:alice",
            0.8,  // MATL score
            200.0, // stake
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();
        assert_eq!(proposal.id, "MIP-0001");
        assert_eq!(proposal.status, ProposalStatus::Draft);
    }

    #[test]
    fn test_proposal_eligibility_check() {
        let mut manager = ProposalManager::new(test_config());

        // Low MATL score should fail
        let result = manager.create_proposal(
            "MIP-0002",
            "Test",
            "Desc",
            ProposalType::Standard,
            "did:mycelix:bob",
            0.3,  // Below threshold
            200.0,
        );

        assert!(matches!(result, Err(GovernanceError::InsufficientEligibility(_))));
    }

    #[test]
    fn test_proposal_lifecycle() {
        let mut manager = ProposalManager::new(test_config());

        // Create
        manager.create_proposal(
            "MIP-0003",
            "Lifecycle Test",
            "Testing full lifecycle",
            ProposalType::Standard,
            "did:mycelix:charlie",
            0.8,
            200.0,
        ).unwrap();

        // Submit for discussion
        manager.submit_for_discussion("MIP-0003").unwrap();
        assert_eq!(
            manager.get_proposal("MIP-0003").unwrap().status,
            ProposalStatus::Discussion
        );

        // Start voting (normally would wait for discussion period)
        let proposal = manager.get_proposal_mut("MIP-0003").unwrap();
        proposal.voting_starts = Some(Utc::now() - chrono::Duration::hours(1));
        drop(proposal);

        manager.start_voting("MIP-0003").unwrap();
        assert_eq!(
            manager.get_proposal("MIP-0003").unwrap().status,
            ProposalStatus::Voting
        );
    }

    #[test]
    fn test_guardian_veto() {
        let mut manager = ProposalManager::new(test_config());

        manager.create_proposal(
            "MIP-EMERGENCY",
            "Emergency Fix",
            "Critical security patch",
            ProposalType::Emergency,
            "did:mycelix:alice",
            0.9,
            500.0,
        ).unwrap();

        // Advance to approved status
        let proposal = manager.get_proposal_mut("MIP-EMERGENCY").unwrap();
        proposal.status = ProposalStatus::Approved;
        proposal.execution_time = Some(Utc::now() + chrono::Duration::hours(1));
        drop(proposal);

        // Veto by guardian
        let result = manager.apply_veto(
            "MIP-EMERGENCY",
            "did:mycelix:guardian1",
            "Security concerns identified",
        );

        assert!(result.is_ok());
        assert_eq!(
            manager.get_proposal("MIP-EMERGENCY").unwrap().status,
            ProposalStatus::Vetoed
        );
    }

    #[test]
    fn test_finalize_with_quorum_failure() {
        let mut manager = ProposalManager::new(test_config());

        manager.create_proposal(
            "MIP-0004",
            "Low Turnout",
            "Not enough votes",
            ProposalType::Standard,
            "did:mycelix:dave",
            0.8,
            200.0,
        ).unwrap();

        // Advance to tallying
        let proposal = manager.get_proposal_mut("MIP-0004").unwrap();
        proposal.status = ProposalStatus::Tallying;
        drop(proposal);

        // Create tally with low participation
        let mut tally = VoteTally::default();
        tally.eligible_weight = 1000.0;
        tally.add_vote(VoteChoice::For, 50.0); // 5% participation

        let result = manager.finalize_proposal("MIP-0004", &tally);
        assert!(matches!(result, Err(GovernanceError::QuorumNotReached { .. })));
    }
}
