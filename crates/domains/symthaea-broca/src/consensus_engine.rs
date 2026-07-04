// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consensus Engine — Distributed Architectural Consensus
//!
//! Manages multi-node consensus for architectural changes.
//! Uses HDC vectors to reach agreement on "System Blueprints"
//! across a peer-to-peer network of Symthaea nodes.

use crate::architectural_memory::ArchitecturalMemory;
use crate::evolutionary_scaffolder::EvolutionResult;
use mycelix_zkp_core::types::AuthenticatedProof;
use std::collections::HashMap;
use symthaea_core::hdc::unified_hv::ContinuousHV;

#[derive(Debug, Clone)]
pub struct ChangeProposal {
    pub proposer_id: String,
    pub blueprint_hv: ContinuousHV,
    pub substrate_path: String,
    pub description: String,
    /// **NEW**: The empirical result of the evolution from the proposer.
    pub local_result: Option<EvolutionResult>,
    /// **NEW**: Zero-Knowledge proof of performance improvement.
    pub zk_proof: Option<AuthenticatedProof>,
}

#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub consensus_reached: bool,
    pub agreed_hv: Option<ContinuousHV>,
    pub disagreement_nodes: Vec<String>,
    pub confidence: f32,
}

pub struct ConsensusEngine {
    node_id: String,
    active_proposals: HashMap<String, Vec<ChangeProposal>>,
}

impl ConsensusEngine {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            active_proposals: HashMap::new(),
        }
    }

    /// Submit a local change proposal to the collective.
    pub fn propose_change(
        &mut self,
        path: &str,
        hv: ContinuousHV,
        desc: &str,
        result: Option<EvolutionResult>,
        zk_proof: Option<AuthenticatedProof>,
    ) {
        let proposal = ChangeProposal {
            proposer_id: self.node_id.clone(),
            blueprint_hv: hv,
            substrate_path: path.to_string(),
            description: desc.to_string(),
            local_result: result,
            zk_proof,
        };
        self.active_proposals
            .entry(path.to_string())
            .or_default()
            .push(proposal);
    }

    /// Vote on a proposal from another node using HDC similarity.
    pub fn vote_on_proposal(&self, local_hv: &ContinuousHV, proposal: &ChangeProposal) -> f32 {
        local_hv.similarity(&proposal.blueprint_hv)
    }

    /// **NEW**: Peer Verification.
    /// Simulate other nodes verifying the evolution result.
    pub fn vote_on_evolution(&self, _proposal: &ChangeProposal) -> bool {
        // In real use: each node runs the evolution in its own sandbox
        rand::random::<f32>() > 0.1 // 90% pass rate for stable mutations
    }

    /// Evaluate if the network has reached consensus on a blueprint.
    pub fn evaluate_consensus(
        &self,
        path: &str,
        threshold: f32,
        memory: &mut ArchitecturalMemory,
    ) -> ConsensusResult {
        let proposals = match self.active_proposals.get(path) {
            Some(p) if !p.is_empty() => p,
            _ => {
                return ConsensusResult {
                    consensus_reached: false,
                    agreed_hv: None,
                    disagreement_nodes: vec![],
                    confidence: 0.0,
                };
            }
        };

        // For demo: simple average of proposal HVs
        let mut sum_hv = proposals[0].blueprint_hv.clone();
        for p in &proposals[1..] {
            sum_hv.lerp_in_place(&p.blueprint_hv, 0.5, 0.5);
        }

        // Check if everyone is within threshold of the consensus average
        let mut disagreement_nodes = Vec::new();
        let mut min_similarity = 1.0f32;

        for p in proposals {
            let sim = p.blueprint_hv.similarity(&sum_hv);
            min_similarity = min_similarity.min(sim);
            if sim < threshold {
                disagreement_nodes.push(p.proposer_id.clone());
            } else {
                // If it passes locally, commit to architectural memory
                if let Some(ref best_local) = p.local_result {
                    let _ = memory.commit_evolution(best_local, &p.blueprint_hv);
                }
            }
        }

        ConsensusResult {
            consensus_reached: disagreement_nodes.is_empty(),
            agreed_hv: Some(sum_hv),
            disagreement_nodes,
            confidence: min_similarity,
        }
    }

    /// **NEW**: Internalize Consensus.
    /// Commit the agreed change to long-term memory if consensus is reached.
    pub fn internalize_consensus(
        &self,
        path: &str,
        result: &ConsensusResult,
        memory: &mut ArchitecturalMemory,
    ) -> anyhow::Result<()> {
        if result.consensus_reached
            && let Some(ref hv) = result.agreed_hv
            && let Some(proposals) = self.active_proposals.get(path)
            && let Some(best_local) = &proposals[0].local_result
        {
            memory.commit_evolution(best_local, hv)?;
            println!("🧬 Multi-agent consensus reached. Committed to global memory.");
        }
        Ok(())
    }
}
