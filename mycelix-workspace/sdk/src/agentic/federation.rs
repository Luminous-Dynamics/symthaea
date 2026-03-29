// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cross-Swarm Federation
//!
//! Trust propagation across swarm boundaries with federated consensus.
//!
//! ## Features
//!
//! - **Trust Attestations**: Cross-swarm trust vouching
//! - **Federated Consensus**: Multi-swarm decision making
//! - **Bridge Protocols**: Secure trust transfer mechanisms
//! - **Reputation Portability**: Carry trust across boundaries
//!
//! ## Security Model
//!
//! Federation uses a "trust but verify" model where:
//! - Each swarm maintains sovereign trust scores
//! - Cross-swarm attestations are weighted by source swarm reputation
//! - Byzantine tolerance applies at federation level
//! - Sybil resistance through stake requirements

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::matl::KVector;

// ============================================================================
// Configuration
// ============================================================================

/// Federation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Minimum swarm reputation to participate
    pub min_swarm_reputation: f64,
    /// Maximum attestations per agent per epoch
    pub max_attestations_per_epoch: u32,
    /// Attestation decay rate (per day)
    pub attestation_decay_rate: f64,
    /// Required confirmations for cross-swarm trust
    pub required_confirmations: u32,
    /// Byzantine tolerance threshold (0.0-0.5)
    pub byzantine_threshold: f64,
    /// Enable recursive trust (trust of trust)
    pub recursive_trust: bool,
    /// Maximum recursion depth
    pub max_recursion_depth: u32,
    /// Stake required for federation participation
    pub stake_requirement: u64,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            min_swarm_reputation: 0.3,
            max_attestations_per_epoch: 100,
            attestation_decay_rate: 0.02,
            required_confirmations: 3,
            byzantine_threshold: 0.33,
            recursive_trust: true,
            max_recursion_depth: 3,
            stake_requirement: 1000,
        }
    }
}

// ============================================================================
// Swarm Identity
// ============================================================================

/// Unique swarm identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SwarmId(pub String);

impl SwarmId {
    /// Create a new swarm identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Swarm profile for federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmProfile {
    /// Swarm identifier
    pub id: SwarmId,
    /// Human-readable name
    pub name: String,
    /// Swarm's aggregate reputation
    pub reputation: f64,
    /// Number of members
    pub member_count: u32,
    /// Total stake deposited
    pub total_stake: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Swarm's K-Vector aggregate
    pub aggregate_kvector: KVector,
    /// Federation endpoints
    pub endpoints: Vec<String>,
    /// Public verification key
    pub verification_key: Vec<u8>,
}

impl SwarmProfile {
    /// Calculate swarm's federation weight
    pub fn federation_weight(&self) -> f64 {
        let reputation_factor = self.reputation;
        let size_factor = (self.member_count as f64).ln().max(1.0) / 10.0;
        let stake_factor = (self.total_stake as f64).ln().max(1.0) / 20.0;

        (reputation_factor * 0.5 + size_factor * 0.3 + stake_factor * 0.2).min(1.0)
    }

    /// Check if swarm meets federation requirements
    pub fn meets_requirements(&self, config: &FederationConfig) -> bool {
        self.reputation >= config.min_swarm_reputation
            && self.total_stake >= config.stake_requirement
    }
}

// ============================================================================
// Trust Attestations
// ============================================================================

/// Cross-swarm trust attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustAttestation {
    /// Unique attestation ID
    pub id: String,
    /// Source swarm
    pub source_swarm: SwarmId,
    /// Target swarm
    pub target_swarm: SwarmId,
    /// Agent being attested
    pub agent_id: String,
    /// Attested trust level
    pub trust_level: f64,
    /// Confidence in attestation (0.0-1.0)
    pub confidence: f64,
    /// Evidence/justification
    pub evidence: AttestationEvidence,
    /// Timestamp
    pub timestamp: u64,
    /// Expiry timestamp
    pub expires_at: u64,
    /// Cryptographic signature
    pub signature: Vec<u8>,
    /// Number of confirmations received
    pub confirmations: u32,
    /// Confirming swarms
    pub confirming_swarms: HashSet<SwarmId>,
}

impl TrustAttestation {
    /// Check if attestation is expired
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }

    /// Check if fully confirmed
    pub fn is_confirmed(&self, required: u32) -> bool {
        self.confirmations >= required
    }

    /// Calculate effective trust (weighted by source reputation)
    pub fn effective_trust(&self, source_reputation: f64) -> f64 {
        self.trust_level * self.confidence * source_reputation
    }
}

/// Evidence supporting attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttestationEvidence {
    /// Based on interaction history
    InteractionHistory {
        /// Number of interactions
        interaction_count: u32,
        /// Success rate
        success_rate: f64,
        /// Time span of interactions (days)
        time_span_days: u32,
    },
    /// Based on K-Vector proof
    KVectorProof {
        /// Commitment to K-Vector
        commitment: Vec<u8>,
        /// ZK proof of properties
        proof: Vec<u8>,
    },
    /// Based on stake/collateral
    StakeBased {
        /// Amount staked
        stake_amount: u64,
        /// Lock duration (days)
        lock_duration: u32,
    },
    /// Delegation from trusted party
    Delegation {
        /// Delegator agent ID
        delegator: String,
        /// Delegator's trust level
        delegator_trust: f64,
    },
}

// ============================================================================
// Federated Consensus
// ============================================================================

/// Cross-swarm proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedProposal {
    /// Unique proposal ID
    pub id: String,
    /// Proposing swarm
    pub proposer_swarm: SwarmId,
    /// Proposal type
    pub proposal_type: FederatedProposalType,
    /// Description
    pub description: String,
    /// Required quorum (weighted)
    pub quorum: f64,
    /// Approval threshold
    pub approval_threshold: f64,
    /// Voting deadline
    pub deadline: u64,
    /// Current state
    pub state: FederatedProposalState,
    /// Votes received
    pub votes: HashMap<SwarmId, FederatedVote>,
    /// Creation timestamp
    pub created_at: u64,
}

/// Types of federated proposals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederatedProposalType {
    /// Add new swarm to federation.
    AddSwarm {
        /// ID of the swarm to add.
        swarm_id: SwarmId,
        /// Profile of the swarm to add.
        profile: Box<SwarmProfile>,
    },
    /// Remove swarm from federation.
    RemoveSwarm {
        /// ID of the swarm to remove.
        swarm_id: SwarmId,
        /// Reason for removal.
        reason: String,
    },
    /// Update federation parameters.
    UpdateConfig {
        /// Proposed configuration changes.
        changes: FederationConfigChanges,
    },
    /// Emergency action.
    Emergency {
        /// The emergency action to execute.
        action: EmergencyAction,
    },
    /// Custom proposal with opaque data.
    Custom {
        /// Serialized proposal data.
        data: Vec<u8>,
    },
}

/// Configuration changes proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfigChanges {
    /// New minimum swarm reputation, if changing.
    pub min_swarm_reputation: Option<f64>,
    /// New Byzantine threshold, if changing.
    pub byzantine_threshold: Option<f64>,
    /// New stake requirement, if changing.
    pub stake_requirement: Option<u64>,
}

/// Emergency actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyAction {
    /// Freeze all cross-swarm transfers
    FreezeTransfers,
    /// Isolate compromised swarm
    IsolateSwarm(SwarmId),
    /// Reset federation state
    Reset,
}

/// Federated proposal state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederatedProposalState {
    /// Open for voting
    Active,
    /// Passed
    Passed,
    /// Failed
    Failed,
    /// Executed
    Executed,
    /// Cancelled
    Cancelled,
}

/// Vote from a swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedVote {
    /// Voting swarm
    pub swarm_id: SwarmId,
    /// Vote decision
    pub decision: FederatedVoteDecision,
    /// Weight (based on swarm profile)
    pub weight: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Signature
    pub signature: Vec<u8>,
}

/// Vote decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederatedVoteDecision {
    /// Vote to approve the proposal.
    Approve,
    /// Vote to reject the proposal.
    Reject,
    /// Abstain from voting.
    Abstain,
}

// ============================================================================
// Trust Bridge
// ============================================================================

/// Trust bridge between swarms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustBridge {
    /// Bridge ID
    pub id: String,
    /// Source swarm
    pub source: SwarmId,
    /// Target swarm
    pub target: SwarmId,
    /// Bridge type
    pub bridge_type: BridgeType,
    /// Active status
    pub active: bool,
    /// Trust transfer capacity per epoch
    pub capacity: f64,
    /// Current utilization
    pub utilization: f64,
    /// Fee rate for transfers
    pub fee_rate: f64,
}

/// Bridge types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BridgeType {
    /// Bidirectional trust flow
    Bidirectional,
    /// One-way trust export
    ExportOnly,
    /// One-way trust import
    ImportOnly,
    /// Escrow-based with lockup
    Escrowed,
}

/// Trust transfer request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustTransfer {
    /// Transfer ID
    pub id: String,
    /// Source swarm
    pub source_swarm: SwarmId,
    /// Target swarm
    pub target_swarm: SwarmId,
    /// Agent ID
    pub agent_id: String,
    /// Trust amount to transfer
    pub trust_amount: f64,
    /// K-Vector snapshot
    pub kvector_snapshot: KVector,
    /// Proof of trust
    pub proof: Vec<u8>,
    /// Status
    pub status: TransferStatus,
    /// Timestamp
    pub timestamp: u64,
}

/// Transfer status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferStatus {
    /// Transfer is pending processing.
    Pending,
    /// Transfer has been confirmed by validators.
    Confirmed,
    /// Transfer completed successfully.
    Completed,
    /// Transfer failed.
    Failed,
    /// Transfer was reverted after completion.
    Reverted,
}

// ============================================================================
// Federation Engine
// ============================================================================

/// Main federation engine
#[derive(Debug)]
pub struct FederationEngine {
    /// Configuration
    config: FederationConfig,
    /// Registered swarms
    swarms: HashMap<SwarmId, SwarmProfile>,
    /// Active attestations
    attestations: HashMap<String, TrustAttestation>,
    /// Trust bridges
    bridges: HashMap<String, TrustBridge>,
    /// Pending transfers
    pending_transfers: VecDeque<TrustTransfer>,
    /// Completed transfers
    completed_transfers: Vec<TrustTransfer>,
    /// Active proposals
    proposals: HashMap<String, FederatedProposal>,
    /// Aggregated cross-swarm trust per agent
    cross_swarm_trust: HashMap<String, CrossSwarmTrust>,
    /// Event log
    events: VecDeque<FederationEvent>,
    /// Current timestamp
    current_time: u64,
}

/// Aggregated trust from multiple swarms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSwarmTrust {
    /// Agent ID
    pub agent_id: String,
    /// Home swarm
    pub home_swarm: SwarmId,
    /// Trust from each swarm
    pub swarm_trust: HashMap<SwarmId, f64>,
    /// Weighted aggregate trust
    pub aggregate_trust: f64,
    /// Last update timestamp
    pub last_updated: u64,
}

impl CrossSwarmTrust {
    /// Recalculate aggregate trust
    pub fn recalculate(&mut self, swarm_weights: &HashMap<SwarmId, f64>) {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (swarm_id, trust) in &self.swarm_trust {
            let weight = swarm_weights.get(swarm_id).copied().unwrap_or(0.0);
            weighted_sum += trust * weight;
            total_weight += weight;
        }

        self.aggregate_trust = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };
    }
}

/// Federation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederationEvent {
    /// A swarm joined the federation.
    SwarmJoined {
        /// Swarm that joined.
        swarm_id: SwarmId,
        /// When the swarm joined.
        timestamp: u64,
    },
    /// A swarm left the federation.
    SwarmLeft {
        /// Swarm that left.
        swarm_id: SwarmId,
        /// Reason for leaving.
        reason: String,
        /// When the swarm left.
        timestamp: u64,
    },
    /// A cross-swarm attestation was created.
    AttestationCreated {
        /// Attestation identifier.
        attestation_id: String,
        /// Source swarm.
        source: SwarmId,
        /// Target swarm.
        target: SwarmId,
        /// Creation time.
        timestamp: u64,
    },
    /// An attestation received sufficient confirmations.
    AttestationConfirmed {
        /// Attestation identifier.
        attestation_id: String,
        /// Confirmation time.
        timestamp: u64,
    },
    /// An attestation expired.
    AttestationExpired {
        /// Attestation identifier.
        attestation_id: String,
        /// Expiration time.
        timestamp: u64,
    },
    /// A trust transfer was initiated.
    TransferInitiated {
        /// Transfer identifier.
        transfer_id: String,
        /// Source swarm.
        source: SwarmId,
        /// Target swarm.
        target: SwarmId,
        /// Initiation time.
        timestamp: u64,
    },
    /// A trust transfer completed.
    TransferCompleted {
        /// Transfer identifier.
        transfer_id: String,
        /// Completion time.
        timestamp: u64,
    },
    /// A trust transfer failed.
    TransferFailed {
        /// Transfer identifier.
        transfer_id: String,
        /// Failure reason.
        reason: String,
        /// Failure time.
        timestamp: u64,
    },
    /// A federated proposal was created.
    ProposalCreated {
        /// Proposal identifier.
        proposal_id: String,
        /// Proposing swarm.
        proposer: SwarmId,
        /// Creation time.
        timestamp: u64,
    },
    /// A federated proposal passed.
    ProposalPassed {
        /// Proposal identifier.
        proposal_id: String,
        /// Time the proposal passed.
        timestamp: u64,
    },
    /// A federated proposal failed.
    ProposalFailed {
        /// Proposal identifier.
        proposal_id: String,
        /// Time the proposal failed.
        timestamp: u64,
    },
    /// A trust bridge was created between swarms.
    BridgeCreated {
        /// Bridge identifier.
        bridge_id: String,
        /// Source swarm.
        source: SwarmId,
        /// Target swarm.
        target: SwarmId,
        /// Creation time.
        timestamp: u64,
    },
    /// A trust bridge was deactivated.
    BridgeDeactivated {
        /// Bridge identifier.
        bridge_id: String,
        /// Deactivation reason.
        reason: String,
        /// Deactivation time.
        timestamp: u64,
    },
    /// An emergency action was triggered.
    EmergencyAction {
        /// The emergency action taken.
        action: EmergencyAction,
        /// When the action was triggered.
        timestamp: u64,
    },
}

impl FederationEngine {
    /// Create new federation engine
    pub fn new(config: FederationConfig) -> Self {
        Self {
            config,
            swarms: HashMap::new(),
            attestations: HashMap::new(),
            bridges: HashMap::new(),
            pending_transfers: VecDeque::new(),
            completed_transfers: Vec::new(),
            proposals: HashMap::new(),
            cross_swarm_trust: HashMap::new(),
            events: VecDeque::new(),
            current_time: 0,
        }
    }

    /// Register a swarm
    pub fn register_swarm(&mut self, profile: SwarmProfile) -> Result<(), FederationError> {
        if !profile.meets_requirements(&self.config) {
            return Err(FederationError::InsufficientRequirements {
                swarm_id: profile.id.clone(),
                reason: "Does not meet minimum reputation or stake requirements".to_string(),
            });
        }

        let swarm_id = profile.id.clone();
        self.swarms.insert(swarm_id.clone(), profile);
        self.events.push_back(FederationEvent::SwarmJoined {
            swarm_id,
            timestamp: self.current_time,
        });

        Ok(())
    }

    /// Remove a swarm
    pub fn remove_swarm(
        &mut self,
        swarm_id: &SwarmId,
        reason: String,
    ) -> Result<(), FederationError> {
        if !self.swarms.contains_key(swarm_id) {
            return Err(FederationError::SwarmNotFound(swarm_id.clone()));
        }

        self.swarms.remove(swarm_id);

        // Deactivate all bridges involving this swarm
        for bridge in self.bridges.values_mut() {
            if &bridge.source == swarm_id || &bridge.target == swarm_id {
                bridge.active = false;
            }
        }

        self.events.push_back(FederationEvent::SwarmLeft {
            swarm_id: swarm_id.clone(),
            reason,
            timestamp: self.current_time,
        });

        Ok(())
    }

    /// Create trust attestation
    pub fn create_attestation(
        &mut self,
        source_swarm: SwarmId,
        target_swarm: SwarmId,
        agent_id: String,
        trust_level: f64,
        confidence: f64,
        evidence: AttestationEvidence,
        duration_ms: u64,
        signature: Vec<u8>,
    ) -> Result<String, FederationError> {
        // Verify source swarm exists and is authorized
        let source = self
            .swarms
            .get(&source_swarm)
            .ok_or_else(|| FederationError::SwarmNotFound(source_swarm.clone()))?;

        if source.reputation < self.config.min_swarm_reputation {
            return Err(FederationError::InsufficientReputation {
                swarm_id: source_swarm,
                required: self.config.min_swarm_reputation,
                actual: source.reputation,
            });
        }

        let attestation_id = format!("attest-{}-{}", self.current_time, agent_id);
        let attestation = TrustAttestation {
            id: attestation_id.clone(),
            source_swarm: source_swarm.clone(),
            target_swarm: target_swarm.clone(),
            agent_id,
            trust_level: trust_level.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            evidence,
            timestamp: self.current_time,
            expires_at: self.current_time + duration_ms,
            signature,
            confirmations: 1, // Self-confirmation
            confirming_swarms: HashSet::from([source_swarm.clone()]),
        };

        self.attestations
            .insert(attestation_id.clone(), attestation);
        self.events.push_back(FederationEvent::AttestationCreated {
            attestation_id: attestation_id.clone(),
            source: source_swarm,
            target: target_swarm,
            timestamp: self.current_time,
        });

        Ok(attestation_id)
    }

    /// Confirm an attestation
    pub fn confirm_attestation(
        &mut self,
        attestation_id: &str,
        confirming_swarm: SwarmId,
        signature: Vec<u8>,
    ) -> Result<bool, FederationError> {
        let attestation = self
            .attestations
            .get_mut(attestation_id)
            .ok_or_else(|| FederationError::AttestationNotFound(attestation_id.to_string()))?;

        if attestation.is_expired(self.current_time) {
            return Err(FederationError::AttestationExpired(
                attestation_id.to_string(),
            ));
        }

        if !self.swarms.contains_key(&confirming_swarm) {
            return Err(FederationError::SwarmNotFound(confirming_swarm));
        }

        if attestation.confirming_swarms.contains(&confirming_swarm) {
            return Ok(false); // Already confirmed
        }

        attestation.confirming_swarms.insert(confirming_swarm);
        attestation.confirmations += 1;
        let _ = signature; // Would verify in production

        let is_fully_confirmed = attestation.is_confirmed(self.config.required_confirmations);

        // Clone attestation data before releasing the mutable borrow on self.attestations
        let attestation_clone = if is_fully_confirmed {
            Some(attestation.clone())
        } else {
            None
        };

        if let Some(ref att) = attestation_clone {
            // Update cross-swarm trust
            self.update_cross_swarm_trust(att);

            self.events
                .push_back(FederationEvent::AttestationConfirmed {
                    attestation_id: attestation_id.to_string(),
                    timestamp: self.current_time,
                });
        }

        Ok(is_fully_confirmed)
    }

    /// Update cross-swarm trust based on confirmed attestation
    fn update_cross_swarm_trust(&mut self, attestation: &TrustAttestation) {
        let source_reputation = self
            .swarms
            .get(&attestation.source_swarm)
            .map(|s| s.reputation)
            .unwrap_or(0.0);

        let effective_trust = attestation.effective_trust(source_reputation);

        let entry = self
            .cross_swarm_trust
            .entry(attestation.agent_id.clone())
            .or_insert_with(|| CrossSwarmTrust {
                agent_id: attestation.agent_id.clone(),
                home_swarm: attestation.target_swarm.clone(),
                swarm_trust: HashMap::new(),
                aggregate_trust: 0.0,
                last_updated: self.current_time,
            });

        entry
            .swarm_trust
            .insert(attestation.source_swarm.clone(), effective_trust);
        entry.last_updated = self.current_time;

        // Recalculate aggregate
        let swarm_weights: HashMap<SwarmId, f64> = self
            .swarms
            .iter()
            .map(|(id, profile)| (id.clone(), profile.federation_weight()))
            .collect();
        entry.recalculate(&swarm_weights);
    }

    /// Create trust bridge
    pub fn create_bridge(
        &mut self,
        source: SwarmId,
        target: SwarmId,
        bridge_type: BridgeType,
        capacity: f64,
        fee_rate: f64,
    ) -> Result<String, FederationError> {
        if !self.swarms.contains_key(&source) {
            return Err(FederationError::SwarmNotFound(source));
        }
        if !self.swarms.contains_key(&target) {
            return Err(FederationError::SwarmNotFound(target));
        }

        let bridge_id = format!(
            "bridge-{}-{}-{}",
            source.as_str(),
            target.as_str(),
            self.current_time
        );
        let bridge = TrustBridge {
            id: bridge_id.clone(),
            source: source.clone(),
            target: target.clone(),
            bridge_type,
            active: true,
            capacity,
            utilization: 0.0,
            fee_rate,
        };

        self.bridges.insert(bridge_id.clone(), bridge);
        self.events.push_back(FederationEvent::BridgeCreated {
            bridge_id: bridge_id.clone(),
            source,
            target,
            timestamp: self.current_time,
        });

        Ok(bridge_id)
    }

    /// Initiate trust transfer
    pub fn initiate_transfer(
        &mut self,
        source_swarm: SwarmId,
        target_swarm: SwarmId,
        agent_id: String,
        trust_amount: f64,
        kvector_snapshot: KVector,
        proof: Vec<u8>,
    ) -> Result<String, FederationError> {
        // Find active bridge
        let bridge = self
            .bridges
            .values()
            .find(|b| {
                b.active
                    && ((b.source == source_swarm && b.target == target_swarm)
                        || (b.bridge_type == BridgeType::Bidirectional
                            && b.target == source_swarm
                            && b.source == target_swarm))
            })
            .ok_or(FederationError::NoBridge {
                source: source_swarm.clone(),
                target: target_swarm.clone(),
            })?;

        if bridge.utilization + trust_amount > bridge.capacity {
            return Err(FederationError::BridgeCapacityExceeded {
                bridge_id: bridge.id.clone(),
                capacity: bridge.capacity,
                requested: trust_amount,
            });
        }

        let transfer_id = format!("transfer-{}-{}", self.current_time, agent_id);
        let transfer = TrustTransfer {
            id: transfer_id.clone(),
            source_swarm: source_swarm.clone(),
            target_swarm: target_swarm.clone(),
            agent_id,
            trust_amount,
            kvector_snapshot,
            proof,
            status: TransferStatus::Pending,
            timestamp: self.current_time,
        };

        self.pending_transfers.push_back(transfer);
        self.events.push_back(FederationEvent::TransferInitiated {
            transfer_id: transfer_id.clone(),
            source: source_swarm,
            target: target_swarm,
            timestamp: self.current_time,
        });

        Ok(transfer_id)
    }

    /// Process pending transfers
    pub fn process_transfers(&mut self) -> Vec<TransferResult> {
        let mut results = Vec::new();

        while let Some(mut transfer) = self.pending_transfers.pop_front() {
            // Verify proof (simplified - would be full ZK verification in production)
            let proof_valid = !transfer.proof.is_empty();

            if proof_valid {
                transfer.status = TransferStatus::Completed;

                // Update cross-swarm trust
                let entry = self
                    .cross_swarm_trust
                    .entry(transfer.agent_id.clone())
                    .or_insert_with(|| CrossSwarmTrust {
                        agent_id: transfer.agent_id.clone(),
                        home_swarm: transfer.source_swarm.clone(),
                        swarm_trust: HashMap::new(),
                        aggregate_trust: 0.0,
                        last_updated: self.current_time,
                    });

                entry
                    .swarm_trust
                    .insert(transfer.target_swarm.clone(), transfer.trust_amount);
                entry.last_updated = self.current_time;

                results.push(TransferResult {
                    transfer_id: transfer.id.clone(),
                    success: true,
                    error: None,
                });

                self.events.push_back(FederationEvent::TransferCompleted {
                    transfer_id: transfer.id.clone(),
                    timestamp: self.current_time,
                });

                self.completed_transfers.push(transfer);
            } else {
                transfer.status = TransferStatus::Failed;

                results.push(TransferResult {
                    transfer_id: transfer.id.clone(),
                    success: false,
                    error: Some("Invalid proof".to_string()),
                });

                self.events.push_back(FederationEvent::TransferFailed {
                    transfer_id: transfer.id,
                    reason: "Invalid proof".to_string(),
                    timestamp: self.current_time,
                });
            }
        }

        results
    }

    /// Create federated proposal
    pub fn create_proposal(
        &mut self,
        proposer_swarm: SwarmId,
        proposal_type: FederatedProposalType,
        description: String,
        quorum: f64,
        approval_threshold: f64,
        deadline: u64,
    ) -> Result<String, FederationError> {
        if !self.swarms.contains_key(&proposer_swarm) {
            return Err(FederationError::SwarmNotFound(proposer_swarm));
        }

        let proposal_id = format!("proposal-{}-{}", proposer_swarm.as_str(), self.current_time);
        let proposal = FederatedProposal {
            id: proposal_id.clone(),
            proposer_swarm: proposer_swarm.clone(),
            proposal_type,
            description,
            quorum,
            approval_threshold,
            deadline,
            state: FederatedProposalState::Active,
            votes: HashMap::new(),
            created_at: self.current_time,
        };

        self.proposals.insert(proposal_id.clone(), proposal);
        self.events.push_back(FederationEvent::ProposalCreated {
            proposal_id: proposal_id.clone(),
            proposer: proposer_swarm,
            timestamp: self.current_time,
        });

        Ok(proposal_id)
    }

    /// Vote on proposal
    pub fn vote_proposal(
        &mut self,
        proposal_id: &str,
        swarm_id: SwarmId,
        decision: FederatedVoteDecision,
        signature: Vec<u8>,
    ) -> Result<(), FederationError> {
        let swarm = self
            .swarms
            .get(&swarm_id)
            .ok_or_else(|| FederationError::SwarmNotFound(swarm_id.clone()))?;
        let weight = swarm.federation_weight();

        let proposal = self
            .proposals
            .get_mut(proposal_id)
            .ok_or_else(|| FederationError::ProposalNotFound(proposal_id.to_string()))?;

        if proposal.state != FederatedProposalState::Active {
            return Err(FederationError::ProposalNotActive(proposal_id.to_string()));
        }

        if self.current_time > proposal.deadline {
            return Err(FederationError::ProposalExpired(proposal_id.to_string()));
        }

        let vote = FederatedVote {
            swarm_id: swarm_id.clone(),
            decision,
            weight,
            timestamp: self.current_time,
            signature,
        };

        proposal.votes.insert(swarm_id, vote);

        Ok(())
    }

    /// Tally proposal votes and update state
    pub fn tally_proposal(
        &mut self,
        proposal_id: &str,
    ) -> Result<FederatedProposalState, FederationError> {
        let proposal = self
            .proposals
            .get_mut(proposal_id)
            .ok_or_else(|| FederationError::ProposalNotFound(proposal_id.to_string()))?;

        if proposal.state != FederatedProposalState::Active {
            return Ok(proposal.state);
        }

        // Calculate total weight and votes
        let total_weight: f64 = self.swarms.values().map(|s| s.federation_weight()).sum();

        let voting_weight: f64 = proposal.votes.values().map(|v| v.weight).sum();

        let approval_weight: f64 = proposal
            .votes
            .values()
            .filter(|v| v.decision == FederatedVoteDecision::Approve)
            .map(|v| v.weight)
            .sum();

        // Check quorum
        let quorum_met = voting_weight / total_weight >= proposal.quorum;

        // Check approval
        let approval_met = if voting_weight > 0.0 {
            approval_weight / voting_weight >= proposal.approval_threshold
        } else {
            false
        };

        // Update state if deadline passed or clear majority
        if self.current_time > proposal.deadline || voting_weight / total_weight > 0.67 {
            proposal.state = if quorum_met && approval_met {
                self.events.push_back(FederationEvent::ProposalPassed {
                    proposal_id: proposal_id.to_string(),
                    timestamp: self.current_time,
                });
                FederatedProposalState::Passed
            } else {
                self.events.push_back(FederationEvent::ProposalFailed {
                    proposal_id: proposal_id.to_string(),
                    timestamp: self.current_time,
                });
                FederatedProposalState::Failed
            };
        }

        Ok(proposal.state)
    }

    /// Get cross-swarm trust for an agent
    pub fn get_cross_swarm_trust(&self, agent_id: &str) -> Option<&CrossSwarmTrust> {
        self.cross_swarm_trust.get(agent_id)
    }

    /// Get all swarms
    pub fn swarms(&self) -> impl Iterator<Item = &SwarmProfile> {
        self.swarms.values()
    }

    /// Get swarm by ID
    pub fn get_swarm(&self, swarm_id: &SwarmId) -> Option<&SwarmProfile> {
        self.swarms.get(swarm_id)
    }

    /// Advance time and process decay
    pub fn tick(&mut self, elapsed_ms: u64) {
        self.current_time += elapsed_ms;

        // Expire old attestations
        let expired: Vec<String> = self
            .attestations
            .iter()
            .filter(|(_, a)| a.is_expired(self.current_time))
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired {
            self.attestations.remove(&id);
            self.events.push_back(FederationEvent::AttestationExpired {
                attestation_id: id,
                timestamp: self.current_time,
            });
        }

        // Reset bridge utilization each epoch (simplified)
        for bridge in self.bridges.values_mut() {
            bridge.utilization = 0.0;
        }

        // Trim old events
        while self.events.len() > 10000 {
            self.events.pop_front();
        }
    }

    /// Get recent events
    pub fn recent_events(&self, limit: usize) -> Vec<&FederationEvent> {
        self.events.iter().rev().take(limit).collect()
    }

    /// Get federation statistics
    pub fn stats(&self) -> FederationStats {
        FederationStats {
            swarm_count: self.swarms.len(),
            active_attestations: self.attestations.len(),
            active_bridges: self.bridges.values().filter(|b| b.active).count(),
            pending_transfers: self.pending_transfers.len(),
            completed_transfers: self.completed_transfers.len(),
            active_proposals: self
                .proposals
                .values()
                .filter(|p| p.state == FederatedProposalState::Active)
                .count(),
            tracked_agents: self.cross_swarm_trust.len(),
        }
    }
}

/// Transfer result
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Transfer identifier.
    pub transfer_id: String,
    /// Whether the transfer succeeded.
    pub success: bool,
    /// Error message if the transfer failed.
    pub error: Option<String>,
}

/// Federation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStats {
    /// Number of registered swarms.
    pub swarm_count: usize,
    /// Number of active attestations.
    pub active_attestations: usize,
    /// Number of active trust bridges.
    pub active_bridges: usize,
    /// Number of pending trust transfers.
    pub pending_transfers: usize,
    /// Number of completed trust transfers.
    pub completed_transfers: usize,
    /// Number of active proposals.
    pub active_proposals: usize,
    /// Number of agents with cross-swarm trust.
    pub tracked_agents: usize,
}

/// Federation errors
#[derive(Debug, Clone)]
pub enum FederationError {
    /// Swarm not found in the federation.
    SwarmNotFound(
        /// Swarm ID that was not found.
        SwarmId,
    ),
    /// Swarm does not meet federation requirements.
    InsufficientRequirements {
        /// Swarm ID.
        swarm_id: SwarmId,
        /// Reason for insufficiency.
        reason: String,
    },
    /// Swarm reputation is below the required threshold.
    InsufficientReputation {
        /// Swarm ID.
        swarm_id: SwarmId,
        /// Required reputation.
        required: f64,
        /// Actual reputation.
        actual: f64,
    },
    /// Attestation not found.
    AttestationNotFound(
        /// Attestation ID.
        String,
    ),
    /// Attestation has expired.
    AttestationExpired(
        /// Attestation ID.
        String,
    ),
    /// No bridge exists between the specified swarms.
    NoBridge {
        /// Source swarm.
        source: SwarmId,
        /// Target swarm.
        target: SwarmId,
    },
    /// Bridge capacity would be exceeded by the transfer.
    BridgeCapacityExceeded {
        /// Bridge ID.
        bridge_id: String,
        /// Maximum bridge capacity.
        capacity: f64,
        /// Requested transfer amount.
        requested: f64,
    },
    /// Proposal not found.
    ProposalNotFound(
        /// Proposal ID.
        String,
    ),
    /// Proposal is not in active state.
    ProposalNotActive(
        /// Proposal ID.
        String,
    ),
    /// Proposal voting deadline has passed.
    ProposalExpired(
        /// Proposal ID.
        String,
    ),
    /// Cryptographic signature verification failed.
    InvalidSignature,
    /// Proof verification failed.
    InvalidProof,
}

impl std::fmt::Display for FederationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SwarmNotFound(id) => write!(f, "Swarm not found: {}", id.as_str()),
            Self::InsufficientRequirements { swarm_id, reason } => {
                write!(
                    f,
                    "Swarm {} insufficient requirements: {}",
                    swarm_id.as_str(),
                    reason
                )
            }
            Self::InsufficientReputation {
                swarm_id,
                required,
                actual,
            } => {
                write!(
                    f,
                    "Swarm {} reputation {} below required {}",
                    swarm_id.as_str(),
                    actual,
                    required
                )
            }
            Self::AttestationNotFound(id) => write!(f, "Attestation not found: {}", id),
            Self::AttestationExpired(id) => write!(f, "Attestation expired: {}", id),
            Self::NoBridge { source, target } => {
                write!(
                    f,
                    "No bridge from {} to {}",
                    source.as_str(),
                    target.as_str()
                )
            }
            Self::BridgeCapacityExceeded {
                bridge_id,
                capacity,
                requested,
            } => {
                write!(
                    f,
                    "Bridge {} capacity {} exceeded by request {}",
                    bridge_id, capacity, requested
                )
            }
            Self::ProposalNotFound(id) => write!(f, "Proposal not found: {}", id),
            Self::ProposalNotActive(id) => write!(f, "Proposal not active: {}", id),
            Self::ProposalExpired(id) => write!(f, "Proposal expired: {}", id),
            Self::InvalidSignature => write!(f, "Invalid signature"),
            Self::InvalidProof => write!(f, "Invalid proof"),
        }
    }
}

impl std::error::Error for FederationError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_swarm(id: &str, reputation: f64) -> SwarmProfile {
        SwarmProfile {
            id: SwarmId::new(id),
            name: format!("Test Swarm {}", id),
            reputation,
            member_count: 100,
            total_stake: 10000,
            created_at: 0,
            last_activity: 0,
            aggregate_kvector: KVector::new_participant(),
            endpoints: vec!["http://localhost:8080".to_string()],
            verification_key: vec![1, 2, 3, 4],
        }
    }

    #[test]
    fn test_register_swarm() {
        let mut engine = FederationEngine::new(FederationConfig::default());
        let swarm = create_test_swarm("swarm-1", 0.5);

        assert!(engine.register_swarm(swarm).is_ok());
        assert_eq!(engine.swarms.len(), 1);
    }

    #[test]
    fn test_reject_low_reputation_swarm() {
        let mut engine = FederationEngine::new(FederationConfig::default());
        let swarm = create_test_swarm("swarm-1", 0.1); // Below threshold

        assert!(engine.register_swarm(swarm).is_err());
    }

    #[test]
    fn test_create_attestation() {
        let mut engine = FederationEngine::new(FederationConfig::default());

        let swarm1 = create_test_swarm("swarm-1", 0.5);
        let swarm2 = create_test_swarm("swarm-2", 0.5);
        engine.register_swarm(swarm1).unwrap();
        engine.register_swarm(swarm2).unwrap();

        let result = engine.create_attestation(
            SwarmId::new("swarm-1"),
            SwarmId::new("swarm-2"),
            "agent-1".to_string(),
            0.8,
            0.9,
            AttestationEvidence::InteractionHistory {
                interaction_count: 100,
                success_rate: 0.95,
                time_span_days: 30,
            },
            86400_000,
            vec![1, 2, 3],
        );

        assert!(result.is_ok());
        assert_eq!(engine.attestations.len(), 1);
    }

    #[test]
    fn test_confirm_attestation() {
        let mut engine = FederationEngine::new(FederationConfig {
            required_confirmations: 2,
            ..Default::default()
        });

        let swarm1 = create_test_swarm("swarm-1", 0.5);
        let swarm2 = create_test_swarm("swarm-2", 0.5);
        let swarm3 = create_test_swarm("swarm-3", 0.5);
        engine.register_swarm(swarm1).unwrap();
        engine.register_swarm(swarm2).unwrap();
        engine.register_swarm(swarm3).unwrap();

        let attestation_id = engine
            .create_attestation(
                SwarmId::new("swarm-1"),
                SwarmId::new("swarm-2"),
                "agent-1".to_string(),
                0.8,
                0.9,
                AttestationEvidence::InteractionHistory {
                    interaction_count: 100,
                    success_rate: 0.95,
                    time_span_days: 30,
                },
                86400_000,
                vec![1, 2, 3],
            )
            .unwrap();

        // Confirm from third swarm
        let confirmed = engine
            .confirm_attestation(&attestation_id, SwarmId::new("swarm-3"), vec![4, 5, 6])
            .unwrap();

        assert!(confirmed);

        // Cross-swarm trust should be updated
        let trust = engine.get_cross_swarm_trust("agent-1");
        assert!(trust.is_some());
    }

    #[test]
    fn test_create_bridge() {
        let mut engine = FederationEngine::new(FederationConfig::default());

        let swarm1 = create_test_swarm("swarm-1", 0.5);
        let swarm2 = create_test_swarm("swarm-2", 0.5);
        engine.register_swarm(swarm1).unwrap();
        engine.register_swarm(swarm2).unwrap();

        let result = engine.create_bridge(
            SwarmId::new("swarm-1"),
            SwarmId::new("swarm-2"),
            BridgeType::Bidirectional,
            100.0,
            0.01,
        );

        assert!(result.is_ok());
        assert_eq!(engine.bridges.len(), 1);
    }

    #[test]
    fn test_trust_transfer() {
        let mut engine = FederationEngine::new(FederationConfig::default());

        let swarm1 = create_test_swarm("swarm-1", 0.5);
        let swarm2 = create_test_swarm("swarm-2", 0.5);
        engine.register_swarm(swarm1).unwrap();
        engine.register_swarm(swarm2).unwrap();

        engine
            .create_bridge(
                SwarmId::new("swarm-1"),
                SwarmId::new("swarm-2"),
                BridgeType::Bidirectional,
                100.0,
                0.01,
            )
            .unwrap();

        let transfer_id = engine
            .initiate_transfer(
                SwarmId::new("swarm-1"),
                SwarmId::new("swarm-2"),
                "agent-1".to_string(),
                0.7,
                KVector::new_participant(),
                vec![1, 2, 3], // Non-empty proof
            )
            .unwrap();

        assert!(!transfer_id.is_empty());

        // Process transfers
        let results = engine.process_transfers();
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
    }

    #[test]
    fn test_federated_proposal() {
        let mut engine = FederationEngine::new(FederationConfig::default());

        let swarm1 = create_test_swarm("swarm-1", 0.5);
        let swarm2 = create_test_swarm("swarm-2", 0.6);
        let swarm3 = create_test_swarm("swarm-3", 0.4);
        engine.register_swarm(swarm1).unwrap();
        engine.register_swarm(swarm2).unwrap();
        engine.register_swarm(swarm3).unwrap();

        let proposal_id = engine
            .create_proposal(
                SwarmId::new("swarm-1"),
                FederatedProposalType::UpdateConfig {
                    changes: FederationConfigChanges {
                        min_swarm_reputation: Some(0.4),
                        byzantine_threshold: None,
                        stake_requirement: None,
                    },
                },
                "Lower minimum reputation".to_string(),
                0.5,
                0.5,
                1000000,
            )
            .unwrap();

        // Vote
        engine
            .vote_proposal(
                &proposal_id,
                SwarmId::new("swarm-1"),
                FederatedVoteDecision::Approve,
                vec![],
            )
            .unwrap();
        engine
            .vote_proposal(
                &proposal_id,
                SwarmId::new("swarm-2"),
                FederatedVoteDecision::Approve,
                vec![],
            )
            .unwrap();
        engine
            .vote_proposal(
                &proposal_id,
                SwarmId::new("swarm-3"),
                FederatedVoteDecision::Reject,
                vec![],
            )
            .unwrap();

        // Tally
        engine.current_time = 2000000; // Past deadline
        let state = engine.tally_proposal(&proposal_id).unwrap();

        assert_eq!(state, FederatedProposalState::Passed);
    }

    #[test]
    fn test_federation_stats() {
        let mut engine = FederationEngine::new(FederationConfig::default());

        let swarm1 = create_test_swarm("swarm-1", 0.5);
        engine.register_swarm(swarm1).unwrap();

        let stats = engine.stats();
        assert_eq!(stats.swarm_count, 1);
        assert_eq!(stats.active_attestations, 0);
    }
}
