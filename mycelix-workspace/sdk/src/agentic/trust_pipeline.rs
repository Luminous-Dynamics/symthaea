// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Trust Pipeline
//!
//! Orchestration layer that connects all trust modules into a cohesive flow.
//!
//! ## Pipeline Flow
//!
//! ```text
//! Agent Output
//!     ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ 1. REGISTER OUTPUT                                               │
//! │    - Create provenance node                                      │
//! │    - Compute content hash                                        │
//! │    - Link to parent knowledge                                    │
//! └─────────────────────────────────────────────────────────────────┘
//!     ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ 2. PHI-WEIGHTED CONSENSUS                                        │
//! │    - Collect votes from agents                                   │
//! │    - Weight by trust and coherence                               │
//! │    - Determine outcome                                           │
//! └─────────────────────────────────────────────────────────────────┘
//!     ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ 3. TRUST UPDATE                                                  │
//! │    - Convert consensus to trust delta                            │
//! │    - Update K-Vector dimensions                                  │
//! │    - Propagate through network                                   │
//! └─────────────────────────────────────────────────────────────────┘
//!     ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ 4. GENERATE ATTESTATION                                          │
//! │    - Create ZK proof of trust state                              │
//! │    - Include provenance chain                                    │
//! │    - Commit to K-Vector                                          │
//! └─────────────────────────────────────────────────────────────────┘
//!     ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ 5. CROSS-DOMAIN TRANSLATION (optional)                           │
//! │    - Translate for target domain                                 │
//! │    - Compute domain-specific trust                               │
//! │    - Generate translated attestation                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::epistemic::EmpiricalLevel;
use crate::matl::KVector;

use super::cross_domain::{translate_trust, TranslationResult, TrustDomain};
use super::multi_agent::AgentVote;
use super::phi_consensus::{
    compute_phi_weighted_consensus, PhiConsensusConfig, PhiConsensusResult, PhiConsensusStatus,
};
use super::provenance::{
    DerivationType, ProvenanceBuilder, ProvenanceChain, ProvenanceNode, ProvenanceRegistry,
};
use super::zk_trust::{
    KVectorCommitment, ProofStatement, ProverConfig, TrustProof, TrustProver, TrustVerifier,
};
use super::{AgentOutput, InstrumentalActor};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Pipeline Input
// ============================================================================

/// Input parameters for running the full trust pipeline
pub struct PipelineInput<'a> {
    /// Agent output to process
    pub output: AgentOutput,
    /// Agent identifier
    pub agent_id: &'a str,
    /// Parent provenance nodes
    pub parents: Vec<ProvenanceNode>,
    /// Type of derivation
    pub derivation: DerivationType,
    /// Votes from other agents
    pub votes: Vec<AgentVote>,
    /// Proof statement for attestation
    pub proof_statement: ProofStatement,
    /// Blinding factor for ZK proof
    pub blinding: &'a [u8; 32],
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Configuration for the trust pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Phi consensus configuration
    pub consensus_config: PhiConsensusConfig,

    /// Minimum consensus value to generate positive trust delta
    pub min_consensus_for_trust: f64,

    /// Maximum trust delta per pipeline run
    pub max_trust_delta: f32,

    /// Whether to require ZK attestation
    pub require_attestation: bool,

    /// Default domain for trust computation
    pub default_domain: Option<TrustDomain>,

    /// Enable provenance tracking
    pub track_provenance: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            consensus_config: PhiConsensusConfig::default(),
            min_consensus_for_trust: 0.5,
            max_trust_delta: 0.1,
            require_attestation: false,
            default_domain: None,
            track_provenance: true,
        }
    }
}

// ============================================================================
// Pipeline Stages
// ============================================================================

/// Stage 1: Registered output with provenance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisteredOutput {
    /// The agent output
    pub output: AgentOutput,

    /// Provenance node for this output
    pub provenance: ProvenanceNode,

    /// Parent provenance nodes (sources this derived from)
    pub parents: Vec<ProvenanceNode>,

    /// Registration timestamp
    pub registered_at: u64,

    /// Agent who created this output
    pub agent_id: String,
}

/// Stage 2: Consensus result with vote details
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusOutcome {
    /// The registered output being evaluated
    pub output: RegisteredOutput,

    /// Phi-weighted consensus result
    pub consensus: PhiConsensusResult,

    /// Individual votes received
    pub votes: Vec<AgentVote>,

    /// Whether consensus was reached
    pub consensus_reached: bool,

    /// Effective consensus value (0.0-1.0)
    pub effective_value: f64,
}

/// Stage 3: Trust update result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustUpdate {
    /// The consensus outcome
    pub outcome: ConsensusOutcome,

    /// Agent whose trust was updated
    pub agent_id: String,

    /// Previous K-Vector
    pub previous_kvector: KVector,

    /// Updated K-Vector
    pub updated_kvector: KVector,

    /// Trust delta applied
    pub trust_delta: TrustDelta,

    /// New trust score
    pub new_trust_score: f32,
}

/// Trust change applied to K-Vector
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct TrustDelta {
    /// Change to reputation (k_r)
    pub reputation_delta: f32,
    /// Change to performance (k_p)
    pub performance_delta: f32,
    /// Change to integrity (k_i)
    pub integrity_delta: f32,
    /// Change to historical (k_h)
    pub historical_delta: f32,
    /// Overall direction
    pub direction: TrustDirection,
    /// Reason for the change
    pub reason: String,
}

/// Direction of trust change
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum TrustDirection {
    /// Trust increased
    Increase,
    /// Trust decreased
    Decrease,
    /// Trust unchanged
    Unchanged,
}

/// Stage 4: Attestation with ZK proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustAttestation {
    /// The trust update this attests to
    pub update: TrustUpdate,

    /// ZK proof of trust state
    pub trust_proof: TrustProof,

    /// Commitment to K-Vector
    pub kvector_commitment: KVectorCommitment,

    /// Provenance chain for the output
    pub provenance_chain: Option<ProvenanceChain>,

    /// Attestation timestamp
    pub attested_at: u64,
}

/// Stage 5: Domain-translated attestation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranslatedAttestation {
    /// Original attestation
    pub original: TrustAttestation,

    /// Target domain
    pub target_domain: TrustDomain,

    /// Translation result
    pub translation: TranslationResult,

    /// Domain-specific proof (optional)
    pub domain_proof: Option<TrustProof>,
}

// ============================================================================
// Pipeline Engine
// ============================================================================

/// The unified trust pipeline engine
pub struct TrustPipeline {
    /// Configuration
    config: PipelineConfig,

    /// Provenance registry
    provenance_registry: ProvenanceRegistry,

    /// Trust prover
    prover: TrustProver,

    /// Trust verifier
    verifier: TrustVerifier,

    /// Known agents (for consensus)
    agents: HashMap<String, InstrumentalActor>,

    /// Current timestamp (for testing)
    current_timestamp: u64,
}

impl TrustPipeline {
    /// Create a new trust pipeline
    pub fn new(config: PipelineConfig) -> Self {
        let prover_config = ProverConfig {
            simulation_mode: true, // Default to simulation
            agent_id: "pipeline".to_string(),
            timestamp: 0,
        };

        Self {
            config,
            provenance_registry: ProvenanceRegistry::new(),
            prover: TrustProver::new(prover_config),
            verifier: TrustVerifier::new(),
            agents: HashMap::new(),
            current_timestamp: 0,
        }
    }

    /// Set current timestamp (for testing)
    pub fn set_timestamp(&mut self, timestamp: u64) {
        self.current_timestamp = timestamp;
    }

    /// Register an agent with the pipeline
    pub fn register_agent(&mut self, agent: InstrumentalActor) {
        self.agents
            .insert(agent.agent_id.as_str().to_string(), agent);
    }

    /// Get an agent by ID
    pub fn get_agent(&self, agent_id: &str) -> Option<&InstrumentalActor> {
        self.agents.get(agent_id)
    }

    /// Get mutable agent by ID
    pub fn get_agent_mut(&mut self, agent_id: &str) -> Option<&mut InstrumentalActor> {
        self.agents.get_mut(agent_id)
    }

    // ========================================================================
    // Stage 1: Register Output
    // ========================================================================

    /// Register an agent output with provenance tracking
    pub fn register_output(
        &mut self,
        output: AgentOutput,
        agent_id: &str,
        parents: Vec<ProvenanceNode>,
        derivation: DerivationType,
    ) -> Result<RegisteredOutput, PipelineError> {
        // Serialize output content
        let content = bincode::serialize(&output)
            .map_err(|e| PipelineError::SerializationError(e.to_string()))?;

        // Build provenance node
        let provenance = if parents.is_empty() {
            ProvenanceBuilder::original(&content, agent_id)
        } else {
            ProvenanceBuilder::derived(&content, agent_id, derivation).parents(parents.clone())
        }
        .epistemic(output.classification.empirical)
        .confidence(output.classification_confidence as f64)
        .build(self.current_timestamp);

        // Register in provenance registry if tracking enabled
        if self.config.track_provenance {
            // Register parents first if needed
            for parent in &parents {
                let _ = self.provenance_registry.register(parent.clone());
            }
            self.provenance_registry
                .register(provenance.clone())
                .map_err(|e| PipelineError::ProvenanceError(format!("{:?}", e)))?;
        }

        Ok(RegisteredOutput {
            output,
            provenance,
            parents,
            registered_at: self.current_timestamp,
            agent_id: agent_id.to_string(),
        })
    }

    // ========================================================================
    // Stage 2: Phi-Weighted Consensus
    // ========================================================================

    /// Run phi-weighted consensus on a registered output
    pub fn run_consensus(
        &self,
        registered: RegisteredOutput,
        votes: Vec<AgentVote>,
    ) -> Result<ConsensusOutcome, PipelineError> {
        if votes.is_empty() {
            return Err(PipelineError::NoVotes);
        }

        // Run phi-weighted consensus
        let consensus =
            compute_phi_weighted_consensus(&votes, &self.agents, &self.config.consensus_config);

        let consensus_reached = matches!(
            consensus.status,
            PhiConsensusStatus::Reached | PhiConsensusStatus::ReachedLowCoherence
        );

        // Use the underlying consensus value
        let effective_value = consensus.consensus.consensus_value;

        Ok(ConsensusOutcome {
            output: registered,
            consensus,
            votes,
            consensus_reached,
            effective_value,
        })
    }

    // ========================================================================
    // Stage 3: Trust Update
    // ========================================================================

    /// Process consensus result and update agent trust
    pub fn process_consensus(
        &mut self,
        outcome: ConsensusOutcome,
    ) -> Result<TrustUpdate, PipelineError> {
        let agent_id = outcome.output.agent_id.clone();

        let agent = self
            .agents
            .get(&agent_id)
            .ok_or_else(|| PipelineError::AgentNotFound(agent_id.clone()))?;

        let previous_kvector = agent.k_vector;

        // Compute trust delta based on consensus
        let trust_delta = self.compute_trust_delta(&outcome);

        // Apply delta to K-Vector
        let updated_kvector = self.apply_trust_delta(&previous_kvector, &trust_delta);

        // Update agent in registry
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.k_vector = updated_kvector;
        }

        let new_trust_score = updated_kvector.trust_score();

        Ok(TrustUpdate {
            outcome,
            agent_id,
            previous_kvector,
            updated_kvector,
            trust_delta,
            new_trust_score,
        })
    }

    /// Compute trust delta from consensus outcome
    fn compute_trust_delta(&self, outcome: &ConsensusOutcome) -> TrustDelta {
        let value = outcome.effective_value;
        let reached = outcome.consensus_reached;

        // Determine direction
        let direction = if value >= self.config.min_consensus_for_trust && reached {
            TrustDirection::Increase
        } else if value < 0.3 {
            TrustDirection::Decrease
        } else {
            TrustDirection::Unchanged
        };

        // Compute deltas based on epistemic level of output
        let epistemic_multiplier = match outcome.output.output.classification.empirical {
            EmpiricalLevel::E4PublicRepro => 1.5,
            EmpiricalLevel::E3Cryptographic => 1.2,
            EmpiricalLevel::E2PrivateVerify => 1.0,
            EmpiricalLevel::E1Testimonial => 0.7,
            EmpiricalLevel::E0Null => 0.3,
        };

        let base_delta = match direction {
            TrustDirection::Increase => {
                ((value - self.config.min_consensus_for_trust) * 0.2) as f32
            }
            TrustDirection::Decrease => {
                -((self.config.min_consensus_for_trust - value) * 0.1) as f32
            }
            TrustDirection::Unchanged => 0.0,
        };

        let adjusted_delta = (base_delta * epistemic_multiplier as f32)
            .clamp(-self.config.max_trust_delta, self.config.max_trust_delta);

        let reason = match direction {
            TrustDirection::Increase => format!(
                "Consensus reached ({:.0}%) on E{} output",
                value * 100.0,
                outcome.output.output.classification.empirical as u8
            ),
            TrustDirection::Decrease => format!("Low consensus ({:.0}%) on output", value * 100.0),
            TrustDirection::Unchanged => "Consensus inconclusive".to_string(),
        };

        TrustDelta {
            reputation_delta: adjusted_delta * 0.4,
            performance_delta: adjusted_delta * 0.3,
            integrity_delta: adjusted_delta * 0.2,
            historical_delta: adjusted_delta * 0.1,
            direction,
            reason,
        }
    }

    /// Apply trust delta to K-Vector
    fn apply_trust_delta(&self, kvector: &KVector, delta: &TrustDelta) -> KVector {
        KVector::new(
            (kvector.k_r + delta.reputation_delta).clamp(0.0, 1.0),
            kvector.k_a, // Activity unchanged by consensus
            (kvector.k_i + delta.integrity_delta).clamp(0.0, 1.0),
            (kvector.k_p + delta.performance_delta).clamp(0.0, 1.0),
            kvector.k_m, // Membership unchanged
            kvector.k_s, // Stake unchanged
            (kvector.k_h + delta.historical_delta).clamp(0.0, 1.0),
            kvector.k_topo,      // Topology unchanged
            kvector.k_v,         // Verification unchanged
            kvector.k_coherence, // Coherence unchanged by trust delta
        )
    }

    // ========================================================================
    // Stage 4: Generate Attestation
    // ========================================================================

    /// Generate ZK attestation for trust update
    pub fn generate_attestation(
        &mut self,
        update: TrustUpdate,
        statement: ProofStatement,
        blinding: &[u8; 32],
    ) -> Result<TrustAttestation, PipelineError> {
        // Update prover timestamp
        let prover_config = ProverConfig {
            simulation_mode: true,
            agent_id: update.agent_id.clone(),
            timestamp: self.current_timestamp,
        };
        self.prover = TrustProver::new(prover_config);

        // Generate proof
        let trust_proof = self
            .prover
            .prove(&update.updated_kvector, &statement, blinding)
            .map_err(|e| PipelineError::ProofError(format!("{:?}", e)))?;

        // Create commitment
        let kvector_commitment = self.prover.commit(&update.updated_kvector, blinding);

        // Register commitment with verifier
        self.verifier
            .register_commitment(kvector_commitment.clone());

        // Build provenance chain if tracking enabled
        let provenance_chain = if self.config.track_provenance {
            self.provenance_registry
                .build_chain(&update.outcome.output.provenance.node_id)
        } else {
            None
        };

        Ok(TrustAttestation {
            update,
            trust_proof,
            kvector_commitment,
            provenance_chain,
            attested_at: self.current_timestamp,
        })
    }

    // ========================================================================
    // Stage 5: Cross-Domain Translation
    // ========================================================================

    /// Translate attestation for a target domain
    pub fn translate_for_domain(
        &self,
        attestation: TrustAttestation,
        target_domain: TrustDomain,
        blinding: &[u8; 32],
    ) -> Result<TranslatedAttestation, PipelineError> {
        // Get source domain (default if not configured)
        let source_domain = self
            .config
            .default_domain
            .clone()
            .unwrap_or_else(|| TrustDomain {
                domain_id: "default".to_string(),
                name: "Default Domain".to_string(),
                description: "General-purpose trust domain".to_string(),
                dimension_relevance: super::cross_domain::DomainRelevance::balanced(),
                min_trust_threshold: 0.3,
                requires_verification: false,
                weight_overrides: None,
            });

        // Translate trust
        let translation = translate_trust(
            &attestation.update.updated_kvector,
            &source_domain,
            &target_domain,
        );

        // Optionally generate domain-specific proof
        let domain_proof = if self.config.require_attestation {
            let statement = ProofStatement::TrustExceedsThreshold {
                threshold: target_domain.min_trust_threshold,
            };

            let prover_config = ProverConfig {
                simulation_mode: true,
                agent_id: attestation.update.agent_id.clone(),
                timestamp: self.current_timestamp,
            };
            let prover = TrustProver::new(prover_config);

            prover
                .prove(&translation.target_kvector, &statement, blinding)
                .ok()
        } else {
            None
        };

        Ok(TranslatedAttestation {
            original: attestation,
            target_domain,
            translation,
            domain_proof,
        })
    }

    // ========================================================================
    // Full Pipeline Run
    // ========================================================================

    /// Run the complete pipeline from output to attestation
    pub fn run_full_pipeline(
        &mut self,
        input: PipelineInput<'_>,
    ) -> Result<TrustAttestation, PipelineError> {
        let PipelineInput {
            output,
            agent_id,
            parents,
            derivation,
            votes,
            proof_statement,
            blinding,
        } = input;
        // Stage 1: Register
        let registered = self.register_output(output, agent_id, parents, derivation)?;

        // Stage 2: Consensus
        let consensus = self.run_consensus(registered, votes)?;

        // Stage 3: Trust update
        let update = self.process_consensus(consensus)?;

        // Stage 4: Attestation
        let attestation = self.generate_attestation(update, proof_statement, blinding)?;

        Ok(attestation)
    }

    /// Run full pipeline with domain translation
    pub fn run_full_pipeline_with_translation(
        &mut self,
        input: PipelineInput<'_>,
        target_domain: TrustDomain,
    ) -> Result<TranslatedAttestation, PipelineError> {
        let blinding = input.blinding;
        let attestation = self.run_full_pipeline(input)?;

        self.translate_for_domain(attestation, target_domain, blinding)
    }

    // ========================================================================
    // Verification
    // ========================================================================

    /// Verify an attestation
    pub fn verify_attestation(&self, attestation: &TrustAttestation) -> bool {
        let result = self.verifier.verify(&attestation.trust_proof);
        result.is_valid()
    }

    /// Verify a translated attestation
    pub fn verify_translated_attestation(&self, attestation: &TranslatedAttestation) -> bool {
        // Verify original
        if !self.verify_attestation(&attestation.original) {
            return false;
        }

        // Verify domain proof if present
        if let Some(proof) = &attestation.domain_proof {
            let result = self.verifier.verify(proof);
            if !result.is_valid() {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Pipeline Errors
// ============================================================================

/// Errors that can occur in the trust pipeline
#[derive(Clone, Debug)]
pub enum PipelineError {
    /// Agent not found
    AgentNotFound(String),
    /// Serialization failed
    SerializationError(String),
    /// Provenance error
    ProvenanceError(String),
    /// No votes provided
    NoVotes,
    /// Proof generation failed
    ProofError(String),
    /// Domain translation failed
    TranslationError(String),
    /// Verification failed
    VerificationFailed,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AgentNotFound(id) => write!(f, "Agent not found: {}", id),
            Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
            Self::ProvenanceError(e) => write!(f, "Provenance error: {}", e),
            Self::NoVotes => write!(f, "No votes provided for consensus"),
            Self::ProofError(e) => write!(f, "Proof error: {}", e),
            Self::TranslationError(e) => write!(f, "Translation error: {}", e),
            Self::VerificationFailed => write!(f, "Verification failed"),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::epistemic_classifier::OutputContent;
    use crate::agentic::UncertaintyCalibration;
    use crate::agentic::{AgentClass, AgentConstraints, AgentId, AgentStatus, EpistemicStats};
    use crate::epistemic::{
        EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel, NormativeLevel,
    };

    fn create_test_agent(id: &str, trust: f32) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Autonomous,
            kredit_balance: 1000,
            kredit_cap: 10000,
            k_vector: KVector::new(
                trust, trust, trust, trust, trust, trust, trust, trust, trust, trust,
            ),
            behavior_log: Vec::new(),
            constraints: AgentConstraints::default(),
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            epistemic_stats: EpistemicStats::default(),
            output_history: Vec::new(),
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: Vec::new(),
        }
    }

    fn create_test_output(agent_id: &str) -> AgentOutput {
        AgentOutput {
            output_id: "output-1".to_string(),
            agent_id: agent_id.to_string(),
            content: OutputContent::Text("Test output".to_string()),
            classification: EpistemicClassificationExtended {
                empirical: EmpiricalLevel::E3Cryptographic,
                normative: NormativeLevel::N1Communal,
                materiality: MaterialityLevel::M1Temporal,
                harmonic: HarmonicLevel::H1Local,
            },
            classification_confidence: 0.8,
            timestamp: 1000,
            has_proof: false,
            proof_data: None,
            context_references: vec![],
        }
    }

    fn create_test_vote(agent_id: &str, value: f64) -> AgentVote {
        AgentVote {
            agent_id: agent_id.to_string(),
            value,
            confidence: 0.9,
            epistemic_level: 3,
            reasoning: None,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = TrustPipeline::new(config);
        assert!(pipeline.agents.is_empty());
    }

    #[test]
    fn test_agent_registration() {
        let mut pipeline = TrustPipeline::new(PipelineConfig::default());
        let agent = create_test_agent("agent-1", 0.7);
        pipeline.register_agent(agent);

        assert!(pipeline.get_agent("agent-1").is_some());
        assert!(pipeline.get_agent("unknown").is_none());
    }

    #[test]
    fn test_output_registration() {
        let mut pipeline = TrustPipeline::new(PipelineConfig::default());
        pipeline.set_timestamp(1000);

        let output = create_test_output("agent-1");
        let registered = pipeline
            .register_output(output, "agent-1", vec![], DerivationType::Original)
            .unwrap();

        assert_eq!(registered.agent_id, "agent-1");
        assert_eq!(registered.registered_at, 1000);
        assert!(registered.parents.is_empty());
    }

    #[test]
    fn test_consensus_runs() {
        let mut pipeline = TrustPipeline::new(PipelineConfig::default());
        pipeline.set_timestamp(1000);

        // Register agents
        pipeline.register_agent(create_test_agent("agent-1", 0.8));
        pipeline.register_agent(create_test_agent("voter-1", 0.7));
        pipeline.register_agent(create_test_agent("voter-2", 0.6));

        // Register output
        let output = create_test_output("agent-1");
        let registered = pipeline
            .register_output(output, "agent-1", vec![], DerivationType::Original)
            .unwrap();

        // Create approving votes
        let votes = vec![
            create_test_vote("voter-1", 0.9),
            create_test_vote("voter-2", 0.85),
        ];

        let outcome = pipeline.run_consensus(registered, votes).unwrap();

        // Consensus ran successfully
        assert!(outcome.effective_value > 0.0);
        assert_eq!(outcome.votes.len(), 2);
    }

    #[test]
    fn test_trust_update_mechanics() {
        let mut pipeline = TrustPipeline::new(PipelineConfig::default());
        pipeline.set_timestamp(1000);

        // Register agents
        pipeline.register_agent(create_test_agent("agent-1", 0.5));
        pipeline.register_agent(create_test_agent("voter-1", 0.8));

        // Register and get consensus
        let output = create_test_output("agent-1");
        let registered = pipeline
            .register_output(output, "agent-1", vec![], DerivationType::Original)
            .unwrap();

        let votes = vec![create_test_vote("voter-1", 1.0)];

        let outcome = pipeline.run_consensus(registered, votes).unwrap();
        let update = pipeline.process_consensus(outcome).unwrap();

        // Trust update was computed
        assert!(update.new_trust_score > 0.0);
        assert!(!update.trust_delta.reason.is_empty());
    }

    #[test]
    fn test_trust_delta_direction() {
        let pipeline = TrustPipeline::new(PipelineConfig::default());

        // Mock high consensus outcome
        let high_outcome = ConsensusOutcome {
            output: RegisteredOutput {
                output: create_test_output("agent-1"),
                provenance: ProvenanceBuilder::original(b"test", "agent-1").build(1000),
                parents: vec![],
                registered_at: 1000,
                agent_id: "agent-1".to_string(),
            },
            consensus: PhiConsensusResult {
                consensus: super::super::multi_agent::ConsensusResult {
                    consensus_value: 0.8,
                    confidence: 0.9,
                    participant_count: 3,
                    total_trust_weight: 2.1,
                    dissent: 0.1,
                    consensus_reached: true,
                    contributions: std::collections::HashMap::new(),
                },
                collective_phi: super::super::coherence_integration::CollectiveCoherenceResult {
                    population_coherence: 0.7,
                    average_individual_coherence: 0.65,
                    coherence_variance: 0.05,
                    emergent_integration: 0.1,
                    coherence_level:
                        super::super::coherence_integration::CollectiveCoherenceLevel::HighlyIntegrated,
                    agent_count: 3,
                    coherence_distribution: [0, 0, 0, 3, 0],
                },
                status: PhiConsensusStatus::Reached,
                phi_contributions: std::collections::HashMap::new(),
                divergent_agents: vec![],
                harmonic_confidence: 0.85,
                harmonic_weights: std::collections::HashMap::new(),
            },
            votes: vec![],
            consensus_reached: true,
            effective_value: 0.8,
        };

        let delta = pipeline.compute_trust_delta(&high_outcome);
        assert_eq!(delta.direction, TrustDirection::Increase);
        assert!(delta.reputation_delta > 0.0);
    }

    #[test]
    fn test_full_pipeline() {
        let mut pipeline = TrustPipeline::new(PipelineConfig::default());
        pipeline.set_timestamp(1000);

        // Register agents
        pipeline.register_agent(create_test_agent("agent-1", 0.6));
        pipeline.register_agent(create_test_agent("voter-1", 0.8));
        pipeline.register_agent(create_test_agent("voter-2", 0.7));

        let output = create_test_output("agent-1");
        let votes = vec![
            create_test_vote("voter-1", 0.9),
            create_test_vote("voter-2", 0.8),
        ];

        let blinding = [42u8; 32];
        let statement = ProofStatement::trust_above(0.5);

        let attestation = pipeline
            .run_full_pipeline(PipelineInput {
                output,
                agent_id: "agent-1",
                parents: vec![],
                derivation: DerivationType::Original,
                votes,
                proof_statement: statement,
                blinding: &blinding,
            })
            .unwrap();

        // Verify attestation
        assert!(pipeline.verify_attestation(&attestation));
        assert!(attestation.provenance_chain.is_some());
    }

    #[test]
    fn test_pipeline_with_domain_translation() {
        let mut pipeline = TrustPipeline::new(PipelineConfig::default());
        pipeline.set_timestamp(1000);

        // Register agents
        pipeline.register_agent(create_test_agent("agent-1", 0.7));
        pipeline.register_agent(create_test_agent("voter-1", 0.8));

        let output = create_test_output("agent-1");
        let votes = vec![create_test_vote("voter-1", 0.9)];

        let blinding = [42u8; 32];
        let statement = ProofStatement::trust_above(0.4);
        let target_domain = super::super::cross_domain::DomainTemplates::code_review();

        let translated = pipeline
            .run_full_pipeline_with_translation(
                PipelineInput {
                    output,
                    agent_id: "agent-1",
                    parents: vec![],
                    derivation: DerivationType::Original,
                    votes,
                    proof_statement: statement,
                    blinding: &blinding,
                },
                target_domain,
            )
            .unwrap();

        assert!(pipeline.verify_translated_attestation(&translated));
        assert!(translated.translation.translation_confidence > 0.0);
    }
}
