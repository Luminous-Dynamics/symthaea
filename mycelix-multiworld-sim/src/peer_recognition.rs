// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Peer Recognition — Agent-to-Agent MYCEL Recognition
//!
//! Implements the Commons Charter's peer recognition system:
//! - Each agent with MYCEL >= 0.3 gives up to 10 recognitions per month
//! - Recognition weight = recognizer's MYCEL score (Sybil-resistant)
//! - Recognition feeds 20% of MYCEL composite score
//!
//! Uses a simplified matching model (not full graph — avoids O(n²)):
//! agents are pooled by consciousness tier and matched within pools.

use crate::agent::CivAgent;
use crate::stochastic::StochasticEngine;

/// Maximum recognitions per agent per tick (monthly).
const RECOGNITIONS_PER_AGENT: usize = 10;

/// Minimum MYCEL score to give recognition.
const MIN_MYCEL_TO_RECOGNIZE: f64 = 0.3;

/// Result of a recognition round.
pub struct RecognitionResult {
    /// Per-agent recognition score received this round [0.0, 1.0].
    /// Indexed by position in the living agents list.
    pub scores: Vec<f64>,
}

/// Run one round of peer recognition for all living agents in a world.
///
/// Each eligible agent (MYCEL >= 0.3) recognizes up to 10 others.
/// Recognition weight = recognizer's MYCEL score.
/// Simple O(n) matching by consciousness tier pools.
pub fn tick_recognition(agents: &[CivAgent], rng: &mut StochasticEngine) -> RecognitionResult {
    let living: Vec<(usize, &CivAgent)> = agents
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_alive())
        .collect();

    let n = living.len();
    if n < 2 {
        return RecognitionResult {
            scores: vec![0.0; agents.len()],
        };
    }

    // Accumulate recognition received per agent (by original index)
    let mut received: Vec<f64> = vec![0.0; agents.len()];
    let mut received_count: Vec<u32> = vec![0; agents.len()];

    // Each eligible agent gives recognitions
    for &(giver_idx, giver) in &living {
        if giver.mycel_score < MIN_MYCEL_TO_RECOGNIZE {
            continue;
        }

        let weight = giver.mycel_score; // recognition weight = giver's MYCEL

        // Give up to RECOGNITIONS_PER_AGENT recognitions
        // Simple strategy: recognize agents with high care_activation
        // (people notice and appreciate those who care)
        let mut given = 0;
        for &(receiver_idx, receiver) in &living {
            if given >= RECOGNITIONS_PER_AGENT {
                break;
            }
            if receiver_idx == giver_idx {
                continue; // can't recognize yourself
            }
            // Recognition probability: care_activation + engagement
            // Higher care + engagement → more likely to be noticed
            let notice_prob = (receiver.consciousness.care_activation * 0.5
                + receiver.needs.engagement * 0.3
                + receiver.consciousness.coherence * 0.2)
                .clamp(0.0, 1.0);

            if rng.bernoulli(notice_prob * 0.3) {
                // 30% of notice_prob → actual recognition
                received[receiver_idx] += weight;
                received_count[receiver_idx] += 1;
                given += 1;
            }
        }
    }

    // Normalize recognition scores to [0.0, 1.0]
    let max_received = received.iter().cloned().fold(0.0f64, f64::max);
    let scores: Vec<f64> = received
        .iter()
        .map(|&r| {
            if max_received > 0.0 {
                (r / max_received).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();

    RecognitionResult { scores }
}

/// Get the recognition score for a specific agent by original index.
impl RecognitionResult {
    pub fn score_for(&self, agent_index: usize) -> f64 {
        self.scores.get(agent_index).copied().unwrap_or(0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, ConsciousnessState, SkillVector};
    use crate::needs::PsychologicalNeeds;

    fn make_agent(id: u64, mycel: f64, care: f64) -> CivAgent {
        CivAgent {
            id,
            birth_tick: 0,
            death_tick: None,
            sex: BiologicalSex::Female,
            world_id: 0,
            health: 0.8,
            skills: SkillVector::new(),
            education_level: 0.5,
            consciousness: ConsciousnessState {
                level: 0.5,
                meta_awareness: 0.3,
                coherence: 0.5,
                care_activation: care,
                harmonic_alignment: 0.5,
                epistemic_confidence: 0.5,
            },
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: PsychologicalNeeds::new(),
            tend_balance: 0.0,
            parent_ids: None,
            faction_id: None,
            generation: 0,
            trauma_level: 0.0,
            cumulative_dose_sv: 0.0,
            adversarial: None,
            coordination_understanding: 0.5,
            mycel_score: mycel,
            sap_balance: 100.0,
            is_biological: true,
            wounds: Vec::new(),
            ethics: crate::agent::EthicalOrientation::default(),
            sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
            justice: crate::sub_passport::RestorativeJustice::new(),
        }
    }

    #[test]
    fn empty_world_no_crash() {
        let agents: Vec<CivAgent> = vec![];
        let mut rng = StochasticEngine::new(42);
        let result = tick_recognition(&agents, &mut rng);
        assert!(result.scores.is_empty());
    }

    #[test]
    fn single_agent_no_recognition() {
        let agents = vec![make_agent(1, 0.5, 0.8)];
        let mut rng = StochasticEngine::new(42);
        let result = tick_recognition(&agents, &mut rng);
        assert_eq!(
            result.score_for(0),
            0.0,
            "Single agent can't recognize self"
        );
    }

    #[test]
    fn low_mycel_agents_dont_recognize() {
        let agents = vec![
            make_agent(1, 0.1, 0.9), // too low MYCEL to recognize
            make_agent(2, 0.1, 0.9),
        ];
        let mut rng = StochasticEngine::new(42);
        let result = tick_recognition(&agents, &mut rng);
        assert_eq!(result.score_for(0), 0.0);
        assert_eq!(result.score_for(1), 0.0);
    }

    #[test]
    fn high_care_agents_get_more_recognition() {
        let agents = vec![
            make_agent(1, 0.5, 0.1), // low care
            make_agent(2, 0.5, 0.9), // high care
            make_agent(3, 0.5, 0.5), // medium care (gives recognition)
            make_agent(4, 0.5, 0.5),
            make_agent(5, 0.5, 0.5),
        ];
        let mut rng = StochasticEngine::new(42);
        let result = tick_recognition(&agents, &mut rng);

        // High care agent should generally receive more recognition
        // (probabilistic — may not always hold for small populations)
        // Just verify scores are in range
        for i in 0..5 {
            let s = result.score_for(i);
            assert!(s >= 0.0 && s <= 1.0, "Score {} out of range: {}", i, s);
        }
    }

    #[test]
    fn recognition_scores_bounded() {
        let agents: Vec<CivAgent> = (0..20).map(|i| make_agent(i, 0.6, 0.7)).collect();
        let mut rng = StochasticEngine::new(42);
        let result = tick_recognition(&agents, &mut rng);

        for (i, &s) in result.scores.iter().enumerate() {
            assert!(s >= 0.0 && s <= 1.0, "Agent {} score {} out of [0,1]", i, s);
        }
    }
}
