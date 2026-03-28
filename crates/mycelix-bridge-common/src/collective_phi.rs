// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

/// Reputation threshold below which agents are excluded from collective phi.
pub const COLLECTIVE_PHI_BLACKLIST_THRESHOLD: f64 = 0.05;

/// Clamp a consciousness dimension to [0.0, 1.0], replacing NaN/Inf with 0.0.
fn sanitize_dim(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

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
    /// Optional reputation score (0.0–1.0). When provided, agents below
    /// the blacklist threshold (0.05) are excluded from collective phi,
    /// and individual phi is weighted by reputation.
    #[serde(default)]
    pub reputation: Option<f64>,
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
    ///
    /// All consciousness dimensions are clamped to [0.0, 1.0] and
    /// non-finite values are replaced with 0.0 to prevent corrupt
    /// cosine similarity results.
    pub fn new(vectors: Vec<AgentConsciousnessVector>) -> Self {
        let vectors = vectors
            .into_iter()
            .filter(|v| {
                v.reputation
                    .map_or(true, |r| r >= COLLECTIVE_PHI_BLACKLIST_THRESHOLD)
            })
            .map(|mut v| {
                v.consciousness_level = sanitize_dim(v.consciousness_level);
                v.meta_awareness = sanitize_dim(v.meta_awareness);
                v.coherence = sanitize_dim(v.coherence);
                v.care_activation = sanitize_dim(v.care_activation);
                v.harmonic_alignment = sanitize_dim(v.harmonic_alignment);
                v.epistemic_confidence = sanitize_dim(v.epistemic_confidence);
                v
            })
            .collect();
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
            self.vectors
                .iter()
                .step_by(stride.max(1))
                .take(COLLECTIVE_PHI_MAX_SYNC)
                .collect()
        } else {
            self.vectors.iter().collect()
        };

        let n = sample.len();

        // Compute mean individual phi, weighted by reputation when available.
        let mean_individual_phi: f64 = {
            let weighted_sum: f64 = sample
                .iter()
                .map(|v| {
                    let rep_weight = v.reputation.unwrap_or(1.0).clamp(0.0, 1.0);
                    v.individual_phi() * rep_weight
                })
                .sum();
            let weight_sum: f64 = sample
                .iter()
                .map(|v| v.reputation.unwrap_or(1.0).clamp(0.0, 1.0))
                .sum();
            if weight_sum > 1e-10 { weighted_sum / weight_sum } else { 0.0 }
        };

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
            reputation: None,
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
            reputation: None,
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
        assert!(
            result.inter_agent_coherence > 0.9,
            "Similar agents should have high coherence even after sampling"
        );
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

    // ── Red-Team: Cartel → Slash → Exclusion Pipeline ─────────────────

    #[test]
    fn test_redteam_cartel_slash_excludes_from_collective_phi() {
        use super::super::consciousness_profile::{
            apply_cartel_slash, ReputationState, CARTEL_MIN_SIZE,
            CARTEL_SLASH_MIN_CONFIDENCE,
        };

        // Setup: 5 honest agents + 3 cartel agents (all identical consciousness)
        let honest_ids: Vec<String> = (0..5).map(|i| format!("honest_{}", i)).collect();
        let cartel_ids: Vec<String> = (0..3).map(|i| format!("cartel_{}", i)).collect();

        // Build reputation states (all start at 1.0)
        let mut rep_states: std::collections::HashMap<String, ReputationState> =
            std::collections::HashMap::new();
        for id in honest_ids.iter().chain(cartel_ids.iter()) {
            rep_states.insert(id.clone(), ReputationState::new(1.0, 0));
        }

        // Phase 1: Compute collective phi WITH cartel agents (before detection)
        let all_vectors: Vec<AgentConsciousnessVector> = honest_ids
            .iter()
            .chain(cartel_ids.iter())
            .map(|id| AgentConsciousnessVector {
                agent_id: id.clone(),
                consciousness_level: 0.7,
                meta_awareness: 0.7,
                coherence: 0.7,
                care_activation: 0.7,
                harmonic_alignment: 0.7,
                epistemic_confidence: 0.7,
                reputation: Some(rep_states[id].score),
            })
            .collect();
        let phi_before = CollectivePhiEngine::new(all_vectors).compute();
        assert_eq!(phi_before.n_agents, 8, "All 8 agents should be included before slash");

        // Phase 2: Cartel detected with high confidence → slash
        let slash_results = apply_cartel_slash(
            &mut rep_states,
            &cartel_ids,
            0.85, // above CARTEL_SLASH_MIN_CONFIDENCE (0.7)
            1_000_000,
        );
        assert_eq!(slash_results.len(), 3, "All 3 cartel members should be slashed");
        for (id, score) in &slash_results {
            assert!(
                *score < 0.2,
                "Cartel member {} should be heavily slashed: {}",
                id,
                score
            );
        }

        // Phase 3: Recompute collective phi — cartel agents should be excluded
        // (reputation below COLLECTIVE_PHI_BLACKLIST_THRESHOLD = 0.05)
        // With 0.85 slash: 1.0 * (1 - 0.85) = 0.15, still above 0.05
        // So they won't be excluded, but their weight will be reduced
        let post_slash_vectors: Vec<AgentConsciousnessVector> = honest_ids
            .iter()
            .chain(cartel_ids.iter())
            .map(|id| AgentConsciousnessVector {
                agent_id: id.clone(),
                consciousness_level: 0.7,
                meta_awareness: 0.7,
                coherence: 0.7,
                care_activation: 0.7,
                harmonic_alignment: 0.7,
                epistemic_confidence: 0.7,
                reputation: Some(rep_states[id].score),
            })
            .collect();
        let phi_after = CollectivePhiEngine::new(post_slash_vectors).compute();

        // Verify: cartel agents still included (rep 0.15 > 0.05) but weighted less
        assert_eq!(phi_after.n_agents, 8);

        // Phase 4: Double-slash cartel (they keep attacking) → now below blacklist
        let _ = apply_cartel_slash(&mut rep_states, &cartel_ids, 0.85, 2_000_000);
        for id in &cartel_ids {
            assert!(
                rep_states[id].score < 0.05,
                "Double-slashed cartel member {} should be below blacklist: {}",
                id,
                rep_states[id].score
            );
            assert!(rep_states[id].blacklisted, "Should be blacklisted");
        }

        // Phase 5: Recompute — blacklisted agents excluded
        let final_vectors: Vec<AgentConsciousnessVector> = honest_ids
            .iter()
            .chain(cartel_ids.iter())
            .map(|id| AgentConsciousnessVector {
                agent_id: id.clone(),
                consciousness_level: 0.7,
                meta_awareness: 0.7,
                coherence: 0.7,
                care_activation: 0.7,
                harmonic_alignment: 0.7,
                epistemic_confidence: 0.7,
                reputation: Some(rep_states[id].score),
            })
            .collect();
        let phi_final = CollectivePhiEngine::new(final_vectors).compute();
        assert_eq!(
            phi_final.n_agents, 5,
            "Only 5 honest agents should remain after cartel blacklisted"
        );
    }

    #[test]
    fn test_redteam_below_threshold_cartel_not_slashed() {
        use super::super::consciousness_profile::{apply_cartel_slash, ReputationState};

        let mut rep_states: std::collections::HashMap<String, ReputationState> =
            std::collections::HashMap::new();
        let ids: Vec<String> = (0..3).map(|i| format!("agent_{}", i)).collect();
        for id in &ids {
            rep_states.insert(id.clone(), ReputationState::new(1.0, 0));
        }

        // Confidence 0.5 < threshold 0.7 → no slash
        let results = apply_cartel_slash(&mut rep_states, &ids, 0.5, 1_000_000);
        assert!(results.is_empty());
        for id in &ids {
            assert!((rep_states[id].score - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_redteam_too_small_cartel_not_slashed() {
        use super::super::consciousness_profile::{apply_cartel_slash, ReputationState};

        let mut rep_states: std::collections::HashMap<String, ReputationState> =
            std::collections::HashMap::new();
        rep_states.insert("a".to_string(), ReputationState::new(1.0, 0));
        rep_states.insert("b".to_string(), ReputationState::new(1.0, 0));
        let ids = vec!["a".to_string(), "b".to_string()];

        // Only 2 members < CARTEL_MIN_SIZE (3) → no slash
        let results = apply_cartel_slash(&mut rep_states, &ids, 0.9, 1_000_000);
        assert!(results.is_empty());
    }
}
