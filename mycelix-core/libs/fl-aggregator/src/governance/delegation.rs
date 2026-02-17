//! Vote Delegation System
//!
//! Manages delegation of voting power between identities with:
//! - Liquid democracy (transitive delegation)
//! - Delegation chains with depth limits
//! - Topic-specific delegation
//! - Revocation and override capabilities

use super::types::*;
use super::{GovernanceError, GovernanceResult};
use chrono::{DateTime, Duration, Utc};
use std::collections::{HashMap, HashSet};

/// Delegation manager
#[derive(Debug, Default)]
pub struct DelegationManager {
    /// Active delegations: delegator -> delegation info
    delegations: HashMap<String, Delegation>,

    /// Reverse index: delegate -> list of delegators
    delegates_of: HashMap<String, HashSet<String>>,

    /// Configuration
    config: DelegationConfig,
}

/// Delegation configuration
#[derive(Debug, Clone)]
pub struct DelegationConfig {
    /// Maximum delegation chain depth
    pub max_depth: u32,

    /// Whether delegation is enabled
    pub enabled: bool,

    /// Require minimum MATL for delegates
    pub min_delegate_matl: f32,

    /// Allow partial delegation (percentage of weight)
    pub allow_partial: bool,

    /// Allow topic-specific delegation
    pub allow_topic_specific: bool,

    /// Delegation expiry duration
    pub default_expiry: Duration,
}

impl Default for DelegationConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            enabled: true,
            min_delegate_matl: 0.4,
            allow_partial: true,
            allow_topic_specific: true,
            default_expiry: Duration::days(90),
        }
    }
}

/// A delegation of voting power
#[derive(Debug, Clone)]
pub struct Delegation {
    /// Delegator DID (person giving up voting power)
    pub delegator_did: String,

    /// Delegate DID (person receiving voting power)
    pub delegate_did: String,

    /// Fraction of voting power delegated (0.0-1.0)
    pub weight_fraction: f32,

    /// Topic restriction (None = all topics)
    pub topic: Option<DelegationTopic>,

    /// When delegation was created
    pub created_at: DateTime<Utc>,

    /// When delegation expires
    pub expires_at: Option<DateTime<Utc>>,

    /// Whether delegation is currently active
    pub active: bool,

    /// Reason for delegation
    pub reason: Option<String>,
}

/// Topics for delegation scope
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DelegationTopic {
    /// All governance matters
    All,

    /// FL-specific decisions
    FederatedLearning,

    /// Treasury and funding
    Treasury,

    /// Technical/protocol changes
    Technical,

    /// Membership decisions
    Membership,

    /// Constitutional matters
    Constitutional,

    /// Custom topic
    Custom(String),
}

impl DelegationTopic {
    /// Check if this topic matches a proposal type
    pub fn matches_proposal_type(&self, proposal_type: ProposalType) -> bool {
        match self {
            DelegationTopic::All => true,
            DelegationTopic::FederatedLearning => {
                matches!(proposal_type, ProposalType::ModelGovernance)
            }
            DelegationTopic::Treasury => matches!(proposal_type, ProposalType::Treasury),
            DelegationTopic::Technical => {
                matches!(
                    proposal_type,
                    ProposalType::Standard | ProposalType::Emergency
                )
            }
            DelegationTopic::Membership => matches!(proposal_type, ProposalType::Membership),
            DelegationTopic::Constitutional => {
                matches!(proposal_type, ProposalType::Constitutional)
            }
            DelegationTopic::Custom(_) => false,
        }
    }
}

/// Result of delegation resolution
#[derive(Debug, Clone)]
pub struct DelegationResolution {
    /// Original voter
    pub original_did: String,

    /// Final delegate who will vote
    pub effective_delegate: String,

    /// Chain of delegations
    pub delegation_chain: Vec<String>,

    /// Effective weight after chain
    pub effective_weight: f32,

    /// Depth of delegation chain
    pub depth: u32,
}

impl DelegationManager {
    /// Create a new delegation manager
    pub fn new(config: DelegationConfig) -> Self {
        Self {
            delegations: HashMap::new(),
            delegates_of: HashMap::new(),
            config,
        }
    }

    /// Create a delegation
    pub fn delegate(
        &mut self,
        delegator_did: impl Into<String>,
        delegate_did: impl Into<String>,
        weight_fraction: f32,
        topic: Option<DelegationTopic>,
        expiry: Option<DateTime<Utc>>,
    ) -> GovernanceResult<&Delegation> {
        if !self.config.enabled {
            return Err(GovernanceError::DelegationError(
                "Delegation is disabled".to_string(),
            ));
        }

        let delegator_did = delegator_did.into();
        let delegate_did = delegate_did.into();

        // Can't delegate to self
        if delegator_did == delegate_did {
            return Err(GovernanceError::DelegationError(
                "Cannot delegate to self".to_string(),
            ));
        }

        // Check for circular delegation
        if self.would_create_cycle(&delegator_did, &delegate_did)? {
            return Err(GovernanceError::DelegationError(
                "Would create circular delegation".to_string(),
            ));
        }

        // Check depth limit: the delegator's incoming chain depth + 1 (for this new delegation)
        // must not exceed max_depth
        let incoming_depth = self.get_delegation_depth(&delegator_did);
        if incoming_depth + 1 > self.config.max_depth {
            return Err(GovernanceError::DelegationError(format!(
                "Delegation chain would exceed max depth of {}",
                self.config.max_depth
            )));
        }

        // Validate weight fraction
        let weight = if self.config.allow_partial {
            weight_fraction.clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Create delegation
        let delegation = Delegation {
            delegator_did: delegator_did.clone(),
            delegate_did: delegate_did.clone(),
            weight_fraction: weight,
            topic,
            created_at: Utc::now(),
            expires_at: expiry.or_else(|| Some(Utc::now() + self.config.default_expiry)),
            active: true,
            reason: None,
        };

        // Update indices
        self.delegates_of
            .entry(delegate_did)
            .or_default()
            .insert(delegator_did.clone());

        self.delegations.insert(delegator_did.clone(), delegation);

        Ok(self.delegations.get(&delegator_did).unwrap())
    }

    /// Revoke a delegation
    pub fn revoke(&mut self, delegator_did: &str) -> GovernanceResult<()> {
        let delegation = self
            .delegations
            .get_mut(delegator_did)
            .ok_or_else(|| GovernanceError::DelegationError("No delegation found".to_string()))?;

        delegation.active = false;

        // Update reverse index
        if let Some(delegators) = self.delegates_of.get_mut(&delegation.delegate_did) {
            delegators.remove(delegator_did);
        }

        Ok(())
    }

    /// Check if delegation would create a cycle
    fn would_create_cycle(&self, delegator: &str, delegate: &str) -> GovernanceResult<bool> {
        let mut visited = HashSet::new();
        let mut current = delegate.to_string();

        while let Some(delegation) = self.delegations.get(&current) {
            if !delegation.active {
                break;
            }

            if delegation.delegate_did == delegator {
                return Ok(true);
            }

            if visited.contains(&current) {
                break;
            }

            visited.insert(current.clone());
            current = delegation.delegate_did.clone();
        }

        Ok(false)
    }

    /// Get delegation depth for a delegate
    fn get_delegation_depth(&self, delegate_did: &str) -> u32 {
        let mut max_depth = 0;

        if let Some(delegators) = self.delegates_of.get(delegate_did) {
            for delegator in delegators {
                if let Some(delegation) = self.delegations.get(delegator) {
                    if delegation.active {
                        let depth = 1 + self.get_delegation_depth(delegator);
                        max_depth = max_depth.max(depth);
                    }
                }
            }
        }

        max_depth
    }

    /// Resolve effective delegate for a voter
    pub fn resolve_delegate(
        &self,
        voter_did: &str,
        proposal_type: ProposalType,
        original_weight: f32,
    ) -> DelegationResolution {
        let mut chain = Vec::new();
        let mut current = voter_did.to_string();
        let mut effective_weight = original_weight;
        let mut depth = 0;

        while depth < self.config.max_depth {
            if let Some(delegation) = self.delegations.get(&current) {
                // Check if delegation is valid
                if !delegation.active {
                    break;
                }

                // Check expiry
                if let Some(expiry) = delegation.expires_at {
                    if Utc::now() > expiry {
                        break;
                    }
                }

                // Check topic match
                if let Some(topic) = &delegation.topic {
                    if !topic.matches_proposal_type(proposal_type) {
                        break;
                    }
                }

                chain.push(current.clone());
                effective_weight *= delegation.weight_fraction;
                current = delegation.delegate_did.clone();
                depth += 1;
            } else {
                break;
            }
        }

        DelegationResolution {
            original_did: voter_did.to_string(),
            effective_delegate: current,
            delegation_chain: chain,
            effective_weight,
            depth,
        }
    }

    /// Get all delegators for a delegate
    pub fn get_delegators(&self, delegate_did: &str) -> Vec<&Delegation> {
        self.delegates_of
            .get(delegate_did)
            .map(|delegators| {
                delegators
                    .iter()
                    .filter_map(|did| self.delegations.get(did))
                    .filter(|d| d.active)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get delegation for a delegator
    pub fn get_delegation(&self, delegator_did: &str) -> Option<&Delegation> {
        self.delegations.get(delegator_did).filter(|d| d.active)
    }

    /// Calculate total delegated weight for a delegate
    pub fn get_delegated_weight(
        &self,
        delegate_did: &str,
        proposal_type: ProposalType,
        weight_lookup: &HashMap<String, f32>,
    ) -> f32 {
        let mut total = 0.0;

        if let Some(delegators) = self.delegates_of.get(delegate_did) {
            for delegator_did in delegators {
                if let Some(delegation) = self.delegations.get(delegator_did) {
                    if !delegation.active {
                        continue;
                    }

                    // Check topic
                    if let Some(topic) = &delegation.topic {
                        if !topic.matches_proposal_type(proposal_type) {
                            continue;
                        }
                    }

                    // Check expiry
                    if let Some(expiry) = delegation.expires_at {
                        if Utc::now() > expiry {
                            continue;
                        }
                    }

                    if let Some(&weight) = weight_lookup.get(delegator_did) {
                        total += weight * delegation.weight_fraction;
                    }
                }
            }
        }

        total
    }

    /// Clean up expired delegations
    pub fn cleanup_expired(&mut self) {
        let now = Utc::now();
        let expired: Vec<String> = self
            .delegations
            .iter()
            .filter(|(_, d)| {
                if let Some(expiry) = d.expires_at {
                    now > expiry
                } else {
                    false
                }
            })
            .map(|(did, _)| did.clone())
            .collect();

        for did in expired {
            if let Some(delegation) = self.delegations.remove(&did) {
                if let Some(delegators) = self.delegates_of.get_mut(&delegation.delegate_did) {
                    delegators.remove(&did);
                }
            }
        }
    }

    /// Get delegation statistics
    pub fn get_stats(&self) -> DelegationStats {
        let active = self.delegations.values().filter(|d| d.active).count();
        let by_topic = self.delegations.values().filter(|d| d.active).fold(
            HashMap::new(),
            |mut acc, d| {
                let topic = d.topic.clone().unwrap_or(DelegationTopic::All);
                *acc.entry(topic).or_insert(0) += 1;
                acc
            },
        );

        DelegationStats {
            total_delegations: self.delegations.len(),
            active_delegations: active,
            unique_delegates: self.delegates_of.len(),
            by_topic,
        }
    }
}

/// Delegation statistics
#[derive(Debug, Clone)]
pub struct DelegationStats {
    pub total_delegations: usize,
    pub active_delegations: usize,
    pub unique_delegates: usize,
    pub by_topic: HashMap<DelegationTopic, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_delegation() {
        let mut manager = DelegationManager::new(DelegationConfig::default());

        let delegation = manager
            .delegate(
                "did:mycelix:alice",
                "did:mycelix:bob",
                1.0,
                None,
                None,
            )
            .unwrap();

        assert_eq!(delegation.delegator_did, "did:mycelix:alice");
        assert_eq!(delegation.delegate_did, "did:mycelix:bob");
        assert!(delegation.active);
    }

    #[test]
    fn test_circular_delegation_prevented() {
        let mut manager = DelegationManager::new(DelegationConfig::default());

        // Alice delegates to Bob
        manager
            .delegate("did:mycelix:alice", "did:mycelix:bob", 1.0, None, None)
            .unwrap();

        // Bob tries to delegate to Alice - should fail
        let result = manager.delegate(
            "did:mycelix:bob",
            "did:mycelix:alice",
            1.0,
            None,
            None,
        );

        assert!(matches!(
            result,
            Err(GovernanceError::DelegationError(_))
        ));
    }

    #[test]
    fn test_delegation_chain_resolution() {
        let mut manager = DelegationManager::new(DelegationConfig::default());

        // Alice -> Bob -> Charlie
        manager
            .delegate("did:mycelix:alice", "did:mycelix:bob", 1.0, None, None)
            .unwrap();
        manager
            .delegate("did:mycelix:bob", "did:mycelix:charlie", 0.8, None, None)
            .unwrap();

        let resolution =
            manager.resolve_delegate("did:mycelix:alice", ProposalType::Standard, 100.0);

        assert_eq!(resolution.effective_delegate, "did:mycelix:charlie");
        assert_eq!(resolution.depth, 2);
        assert!((resolution.effective_weight - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_topic_specific_delegation() {
        let mut manager = DelegationManager::new(DelegationConfig::default());

        manager
            .delegate(
                "did:mycelix:alice",
                "did:mycelix:treasury_expert",
                1.0,
                Some(DelegationTopic::Treasury),
                None,
            )
            .unwrap();

        // Should delegate for treasury
        let treasury_resolution =
            manager.resolve_delegate("did:mycelix:alice", ProposalType::Treasury, 100.0);
        assert_eq!(
            treasury_resolution.effective_delegate,
            "did:mycelix:treasury_expert"
        );

        // Should not delegate for other topics
        let technical_resolution =
            manager.resolve_delegate("did:mycelix:alice", ProposalType::Standard, 100.0);
        assert_eq!(technical_resolution.effective_delegate, "did:mycelix:alice");
    }

    #[test]
    fn test_revoke_delegation() {
        let mut manager = DelegationManager::new(DelegationConfig::default());

        manager
            .delegate("did:mycelix:alice", "did:mycelix:bob", 1.0, None, None)
            .unwrap();

        manager.revoke("did:mycelix:alice").unwrap();

        let resolution =
            manager.resolve_delegate("did:mycelix:alice", ProposalType::Standard, 100.0);
        assert_eq!(resolution.effective_delegate, "did:mycelix:alice");
        assert_eq!(resolution.depth, 0);
    }

    #[test]
    fn test_max_depth_limit() {
        let config = DelegationConfig {
            max_depth: 2,
            ..Default::default()
        };
        let mut manager = DelegationManager::new(config);

        manager
            .delegate("did:mycelix:a", "did:mycelix:b", 1.0, None, None)
            .unwrap();
        manager
            .delegate("did:mycelix:b", "did:mycelix:c", 1.0, None, None)
            .unwrap();

        // Third delegation should fail due to depth limit
        let result = manager.delegate("did:mycelix:c", "did:mycelix:d", 1.0, None, None);
        assert!(matches!(
            result,
            Err(GovernanceError::DelegationError(_))
        ));
    }
}
