//! # Collective Phi — Group Consciousness Measurement
//!
//! Measures whether a group of agents is *thinking together* (high integration)
//! or just averaging independent opinions (low integration).
//!
//! ## Algorithm
//!
//! Each agent contributes a 6D consciousness vector. We compute pairwise cosine
//! similarity to get inter-agent coherence, then:
//!
//! ```text
//! collective_phi = mean_individual_phi × inter_agent_coherence
//! ```
//!
//! **Fragile consensus**: When vote ratio > 0.7 but collective_phi < 0.3,
//! the consensus is structurally weak — agents agree on outcome but their
//! consciousness states are divergent.
//!
//! ## Scaling
//!
//! Pairwise cosine similarity is O(N²). For N > [`COLLECTIVE_PHI_MAX_SYNC`],
//! we randomly sample down to that limit. At N=100 with 6D vectors, this is
//! ~60K float ops — well under 1ms.
//!
//! ## WASM Float Determinism
//!
//! This module runs **off-chain only** — called from coordinator zomes (local
//! execution) and Symthaea (native Rust). The governance integrity zome does
//! NOT call collective phi; it only stores `AgentConsciousnessVector` as DHT
//! data. No float math crosses the validation boundary.

use serde::{Deserialize, Serialize};

/// Maximum agents to process synchronously. Above this, we sample.
pub const COLLECTIVE_PHI_MAX_SYNC: usize = 100;

/// A single agent's consciousness state vector for collective phi computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConsciousnessVector {
    /// Unique agent identifier.
    pub agent_id: String,
    /// Overall consciousness level (C_unified, 0.0–1.0).
    pub consciousness_level: f64,
    /// Meta-awareness / self-model accuracy (0.0–1.0).
    pub meta_awareness: f64,
    /// Internal coherence (cross-module agreement, 0.0–1.0).
    pub coherence: f64,
    /// Care harmony activation (0.0–1.0).
    pub care_activation: f64,
    /// Alignment of harmonic profile with group (0.0–1.0).
    pub harmonic_alignment: f64,
    /// Epistemic confidence (0.0–1.0).
    pub epistemic_confidence: f64,
}

impl AgentConsciousnessVector {
    /// Extract the 6D vector as an array for computation.
    fn as_array(&self) -> [f64; 6] {
        [
            self.consciousness_level,
            self.meta_awareness,
            self.coherence,
            self.care_activation,
            self.harmonic_alignment,
            self.epistemic_confidence,
        ]
    }

    /// Compute the individual phi proxy (mean of consciousness dimensions).
    fn individual_phi(&self) -> f64 {
        let arr = self.as_array();
        arr.iter().sum::<f64>() / arr.len() as f64
    }
}

/// Result of collective phi computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectivePhiResult {
    /// Inter-agent integrated information (collective consciousness).
    pub collective_phi: f64,
    /// Mean individual phi across all agents.
    pub mean_individual_phi: f64,
    /// Mean pairwise cosine similarity of consciousness vectors.
    pub inter_agent_coherence: f64,
    /// Whether the consensus is structurally fragile.
    pub fragile: bool,
    /// Number of agents in the computation.
    pub n_agents: usize,
}

/// Engine for computing collective phi from agent consciousness vectors.
pub struct CollectivePhiEngine {
    vectors: Vec<AgentConsciousnessVector>,
}

impl CollectivePhiEngine {
    /// Create a new engine with the given agent vectors.
    pub fn new(vectors: Vec<AgentConsciousnessVector>) -> Self {
        Self { vectors }
    }

    /// Compute collective phi.
    ///
    /// For N > [`COLLECTIVE_PHI_MAX_SYNC`], samples down to that limit
    /// using a deterministic stride-based selection (not random, for
    /// reproducibility in coordinator context).
    pub fn compute(&self) -> CollectivePhiResult {
        if self.vectors.is_empty() {
            return CollectivePhiResult {
                collective_phi: 0.0,
                mean_individual_phi: 0.0,
                inter_agent_coherence: 0.0,
                fragile: false,
                n_agents: 0,
            };
        }

        if self.vectors.len() == 1 {
            let phi = self.vectors[0].individual_phi();
            return CollectivePhiResult {
                collective_phi: phi,
                mean_individual_phi: phi,
                inter_agent_coherence: 1.0,
                fragile: false,
                n_agents: 1,
            };
        }

        // Sample if too many agents
        let sample: Vec<&AgentConsciousnessVector> = if self.vectors.len() > COLLECTIVE_PHI_MAX_SYNC
        {
            // Stride-based deterministic sampling
            let stride = self.vectors.len() / COLLECTIVE_PHI_MAX_SYNC;
            self.vectors.iter().step_by(stride.max(1)).take(COLLECTIVE_PHI_MAX_SYNC).collect()
        } else {
            self.vectors.iter().collect()
        };

        let n = sample.len();

        // Compute mean individual phi
        let mean_individual_phi: f64 =
            sample.iter().map(|v| v.individual_phi()).sum::<f64>() / n as f64;

        // Compute pairwise cosine similarity matrix → mean coherence
        let mut total_similarity = 0.0;
        let mut pair_count = 0u64;

        for i in 0..n {
            let a = sample[i].as_array();
            for j in (i + 1)..n {
                let b = sample[j].as_array();
                let sim = cosine_similarity(&a, &b);
                total_similarity += sim;
                pair_count += 1;
            }
        }

        let inter_agent_coherence = if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            1.0
        };

        let collective_phi = mean_individual_phi * inter_agent_coherence.max(0.0);

        // Fragile: high vote ratio but low collective phi
        // (We don't have vote ratio here, so fragile = low coherence despite nonzero phi)
        let fragile = collective_phi < 0.3 && mean_individual_phi > 0.3;

        CollectivePhiResult {
            collective_phi,
            mean_individual_phi,
            inter_agent_coherence,
            fragile,
            n_agents: self.vectors.len(),
        }
    }

    /// Compute collective phi with explicit vote ratio for fragile consensus detection.
    pub fn compute_with_vote_ratio(&self, vote_ratio: f64) -> CollectivePhiResult {
        let mut result = self.compute();
        // Override fragile detection with vote ratio
        result.fragile = vote_ratio > 0.7 && result.collective_phi < 0.3;
        result
    }
}

/// Cosine similarity between two 6D vectors.
fn cosine_similarity(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(id: &str, level: f64) -> AgentConsciousnessVector {
        AgentConsciousnessVector {
            agent_id: id.to_string(),
            consciousness_level: level,
            meta_awareness: level,
            coherence: level,
            care_activation: level,
            harmonic_alignment: level,
            epistemic_confidence: level,
        }
    }

    fn diverse_agent(
        id: &str,
        cl: f64,
        ma: f64,
        co: f64,
        ca: f64,
        ha: f64,
        ec: f64,
    ) -> AgentConsciousnessVector {
        AgentConsciousnessVector {
            agent_id: id.to_string(),
            consciousness_level: cl,
            meta_awareness: ma,
            coherence: co,
            care_activation: ca,
            harmonic_alignment: ha,
            epistemic_confidence: ec,
        }
    }

    #[test]
    fn test_empty_returns_zero() {
        let engine = CollectivePhiEngine::new(vec![]);
        let result = engine.compute();
        assert_eq!(result.collective_phi, 0.0);
        assert_eq!(result.n_agents, 0);
        assert!(!result.fragile);
    }

    #[test]
    fn test_single_agent() {
        let engine = CollectivePhiEngine::new(vec![agent("a1", 0.8)]);
        let result = engine.compute();
        assert!((result.collective_phi - 0.8).abs() < 1e-6);
        assert!((result.mean_individual_phi - 0.8).abs() < 1e-6);
        assert!((result.inter_agent_coherence - 1.0).abs() < 1e-6);
        assert_eq!(result.n_agents, 1);
    }

    #[test]
    fn test_identical_vectors_high_coherence() {
        let vectors = vec![agent("a1", 0.7), agent("a2", 0.7), agent("a3", 0.7)];
        let engine = CollectivePhiEngine::new(vectors);
        let result = engine.compute();

        assert!(
            result.inter_agent_coherence > 0.99,
            "Identical vectors should have coherence ~1.0: {}",
            result.inter_agent_coherence
        );
        assert!(
            (result.collective_phi - 0.7).abs() < 0.01,
            "Collective phi should equal individual phi for identical agents: {}",
            result.collective_phi
        );
    }

    #[test]
    fn test_orthogonal_vectors_low_coherence() {
        // Create agents with very different consciousness profiles
        let vectors = vec![
            diverse_agent("a1", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            diverse_agent("a2", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            diverse_agent("a3", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        ];
        let engine = CollectivePhiEngine::new(vectors);
        let result = engine.compute();

        assert!(
            result.inter_agent_coherence < 0.2,
            "Orthogonal vectors should have low coherence: {}",
            result.inter_agent_coherence
        );
        assert!(
            result.collective_phi < result.mean_individual_phi,
            "Collective phi should be less than mean individual phi for divergent agents"
        );
    }

    #[test]
    fn test_fragile_consensus_detection() {
        // High individual phi but orthogonal → low collective phi
        let vectors = vec![
            diverse_agent("a1", 0.9, 0.1, 0.1, 0.1, 0.1, 0.1),
            diverse_agent("a2", 0.1, 0.9, 0.1, 0.1, 0.1, 0.1),
            diverse_agent("a3", 0.1, 0.1, 0.9, 0.1, 0.1, 0.1),
        ];
        let engine = CollectivePhiEngine::new(vectors);
        let result = engine.compute_with_vote_ratio(0.85);

        assert!(
            result.fragile,
            "High vote ratio + low collective phi should be fragile"
        );
    }

    #[test]
    fn test_fragile_not_triggered_on_aligned() {
        let vectors = vec![agent("a1", 0.8), agent("a2", 0.8), agent("a3", 0.8)];
        let engine = CollectivePhiEngine::new(vectors);
        let result = engine.compute_with_vote_ratio(0.9);

        assert!(
            !result.fragile,
            "High coherence + high collective phi should not be fragile"
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0, 0.5, 0.3, 0.7, 0.2, 0.9];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = [0.0; 6];
        let b = [1.0, 0.5, 0.3, 0.7, 0.2, 0.9];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_large_group_sampling() {
        // 200 agents — should sample down to COLLECTIVE_PHI_MAX_SYNC
        let vectors: Vec<_> = (0..200)
            .map(|i| agent(&format!("a{}", i), 0.6 + (i as f64 * 0.001)))
            .collect();
        let engine = CollectivePhiEngine::new(vectors);
        let result = engine.compute();

        assert_eq!(result.n_agents, 200);
        assert!(result.collective_phi > 0.0);
        assert!(result.inter_agent_coherence > 0.9, "Similar agents should have high coherence even after sampling");
    }

    #[test]
    fn test_collective_phi_bounds() {
        // All dimensions at 1.0, identical agents
        let vectors = vec![agent("a1", 1.0), agent("a2", 1.0)];
        let engine = CollectivePhiEngine::new(vectors);
        let result = engine.compute();
        assert!(result.collective_phi <= 1.0);
        assert!(result.collective_phi >= 0.0);

        // All dimensions at 0.0
        let vectors = vec![agent("a1", 0.0), agent("a2", 0.0)];
        let engine = CollectivePhiEngine::new(vectors);
        let result = engine.compute();
        assert!(result.collective_phi >= 0.0);
    }
}
