// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness evolution: individual growth and collective Phi computation.
//!
//! Each tick evolves individual consciousness dimensions (level, meta_awareness,
//! coherence, care_activation, harmonic_alignment, epistemic_confidence) based on
//! education, age, social role, and cultural alignment. Collective Phi aggregates
//! individual Phi weighted by inter-agent coherence.

use crate::agent::CivAgent;
use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::CulturalProfile;

use serde::{Deserialize, Serialize};

/// Consciousness decay per tick — consciousness is earned, not permanent.
const DECAY_RATE: f64 = 0.0015;

/// Maximum number of agents to sample for pairwise coherence.
/// Bug #2 fix: increased from 100 to 200 for better statistical estimate.
const COHERENCE_SAMPLE_AGENTS: usize = 200;

/// Maximum number of pairwise comparisons for inter-agent coherence.
/// Bug #2 fix: increased from 50 to 200 for more stable coherence estimate.
const COHERENCE_SAMPLE_PAIRS: usize = 200;

/// History length: 120 ticks = 10 years at monthly resolution.
const PHI_HISTORY_LEN: usize = 120;

/// Collective Phi threshold for first HarmonyMilestone event.
const PHI_MILESTONE_THRESHOLD: f64 = 0.5;

/// Collective Phi threshold for consciousness viability.
const PHI_VIABLE_THRESHOLD: f64 = 0.1;

/// Fragile consensus: tier concentration above this fraction is suspicious.
const FRAGILE_TIER_THRESHOLD: f64 = 0.7;

/// Fragile consensus: collective Phi below this value means the consensus is fragile.
const FRAGILE_PHI_THRESHOLD: f64 = 0.3;

/// Individual and collective consciousness evolution engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEngine {
    /// Collective Phi for this world (mean_phi * inter_agent_coherence).
    pub collective_phi: f64,
    /// Inter-agent coherence (pairwise cosine similarity of 6D vectors, sampled).
    pub inter_agent_coherence: f64,
    /// Mean individual Phi across sampled agents.
    pub mean_phi: f64,
    /// Tier distribution [Observer, Participant, Contributor, Steward, Guardian].
    pub tier_distribution: [f64; 5],
    /// Fragile consensus detected (high tier agreement, low collective Phi).
    pub fragile_consensus: bool,
    /// History of collective Phi (last 120 ticks = 10 years).
    pub phi_history: Vec<f64>,
    /// Whether the Phi > 0.5 milestone has already been logged.
    phi_milestone_logged: bool,
}

impl ConsciousnessEngine {
    /// Create a new engine with default state.
    pub fn new() -> Self {
        Self {
            collective_phi: 0.0,
            inter_agent_coherence: 0.0,
            mean_phi: 0.0,
            tier_distribution: [0.0; 5],
            fragile_consensus: false,
            phi_history: Vec::with_capacity(PHI_HISTORY_LEN),
            phi_milestone_logged: false,
        }
    }

    /// Evolve individual consciousness and recompute collective metrics.
    ///
    /// Returns any milestone events generated this tick.
    pub fn tick_consciousness(
        &mut self,
        agents: &mut [CivAgent],
        culture: &CulturalProfile,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();

        // Pre-compute community-level stats for social moral emotions (immutable read before loop).
        // These drive guilt (survivor conscience) and outrage (moral isolation).
        let community_mean_trauma: f64 = {
            let living: Vec<_> = agents.iter().filter(|a| a.is_alive()).collect();
            let n = living.len().max(1) as f64;
            if living.is_empty() {
                0.0
            } else {
                living.iter().map(|a| a.trauma_level).sum::<f64>() / n
            }
        };
        let community_mean_ethics: [f64; 4] = {
            let living: Vec<_> = agents.iter().filter(|a| a.is_alive()).collect();
            let n = living.len().max(1) as f64;
            if living.is_empty() {
                [0.5; 4]
            } else {
                [
                    living.iter().map(|a| a.ethics.deontological).sum::<f64>() / n,
                    living
                        .iter()
                        .map(|a| a.ethics.consequentialist)
                        .sum::<f64>()
                        / n,
                    living.iter().map(|a| a.ethics.virtue_care).sum::<f64>() / n,
                    living.iter().map(|a| a.ethics.relational).sum::<f64>() / n,
                ]
            }
        };

        // --- Individual evolution ---
        for agent in agents.iter_mut().filter(|a| a.is_alive()) {
            // Snapshot ethics before this tick's modifications for dissonance tracking
            let ethics_before = agent.ethics.as_vec();

            // Decay all dimensions (consciousness is earned)
            agent.consciousness.level = (agent.consciousness.level - DECAY_RATE).max(0.0);
            agent.consciousness.meta_awareness =
                (agent.consciousness.meta_awareness - DECAY_RATE).max(0.0);
            agent.consciousness.coherence = (agent.consciousness.coherence - DECAY_RATE).max(0.0);
            agent.consciousness.care_activation =
                (agent.consciousness.care_activation - DECAY_RATE).max(0.0);
            agent.consciousness.harmonic_alignment =
                (agent.consciousness.harmonic_alignment - DECAY_RATE).max(0.0);
            agent.consciousness.epistemic_confidence =
                (agent.consciousness.epistemic_confidence - DECAY_RATE).max(0.0);

            // Level: grows with education (must exceed DECAY_RATE for educated agents)
            agent.consciousness.level =
                (agent.consciousness.level + agent.education_level * 0.003).min(1.0);

            // Level: governance participation bonus (tier >= 2 = Contributor)
            if agent.consciousness.tier() >= 2 {
                agent.consciousness.level = (agent.consciousness.level + 0.001).min(1.0);
            }

            // Meta-awareness: grows with age (wisdom of elders, > 40 years)
            let age_years = agent.age_years(current_tick);
            if age_years > 40.0 {
                agent.consciousness.meta_awareness =
                    (agent.consciousness.meta_awareness + 0.0005).min(1.0);
            }

            // Ethical age drift (Phase 3c): elders shift toward virtue/care and
            // relational orientations (Kohlberg 1981 stage theory, Tornstam 1989
            // gerotranscendence). Lifetime experience grows compassion and
            // community-orientation. Rate: 0.0001/tick ≈ 0.012/year after 50.
            if age_years > 50.0 {
                agent.ethics.virtue_care = (agent.ethics.virtue_care + 0.0001).min(1.0);
                agent.ethics.relational = (agent.ethics.relational + 0.0001).min(1.0);
            }

            // Coherence: emergent from education, trauma, experience, and cultural alignment.
            // Previously a static cultural-weight lookup; now multi-factor so coherence
            // is genuinely earned through education and resilience, not assigned.
            let skill_slice = agent.skills.as_slice();
            let strongest_idx = skill_slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let edu_factor = agent.education_level;
            let trauma_factor = 1.0 - agent.trauma_level;
            let experience_factor = (agent.age_years(current_tick) / 80.0).min(1.0);
            let cultural_factor = culture.harmony_weights[strongest_idx].min(1.0);
            agent.consciousness.coherence = (edu_factor * 0.4
                + trauma_factor * 0.2
                + experience_factor * 0.2
                + cultural_factor * 0.2)
                .clamp(0.0, 1.0);

            // Care activation: grows if agent has children or works in medicine/education
            if !agent.children_ids.is_empty() {
                agent.consciousness.care_activation =
                    (agent.consciousness.care_activation + 0.001).min(1.0);
            }
            let strongest = agent.skills.strongest();
            if strongest == "medicine" || strongest == "education" {
                agent.consciousness.care_activation =
                    (agent.consciousness.care_activation + 0.001).min(1.0);
            }

            // Harmonic alignment: cosine similarity between skill vector and cultural weights
            agent.consciousness.harmonic_alignment =
                cosine_similarity_8(&skill_slice, &culture.harmony_weights)
                    .max(0.0)
                    .min(1.0);

            // Epistemic confidence: grows with education
            agent.consciousness.epistemic_confidence =
                (agent.consciousness.epistemic_confidence + agent.education_level * 0.001).min(1.0);

            // Ethics-consciousness coupling: each ethical orientation accelerates
            // growth in a DIFFERENT consciousness dimension. This creates distinct
            // phi profiles without changing overall phi magnitude — the composition
            // shifts, not the level. Inspired by Symthaea's moral algebra insight
            // that consciousness gates morality, and morality shapes consciousness.
            // Virtue/care → care_activation (compassion practiced becomes embodied)
            // Relational → harmonic_alignment (Ubuntu resonance with community)
            // Deontological → coherence (principled consistency strengthens integration)
            // Consequentialist → epistemic_confidence (evidence-based → calibrated belief)
            agent.consciousness.care_activation =
                (agent.consciousness.care_activation + agent.ethics.virtue_care * 0.0007).min(1.0);
            agent.consciousness.harmonic_alignment = (agent.consciousness.harmonic_alignment
                + agent.ethics.relational * 0.0007)
                .min(1.0);
            agent.consciousness.coherence =
                (agent.consciousness.coherence + agent.ethics.deontological * 0.0005).min(1.0);
            agent.consciousness.epistemic_confidence = (agent.consciousness.epistemic_confidence
                + agent.ethics.consequentialist * 0.0005)
                .min(1.0);

            // Ethics-needs coupling (Maslow-inspired):
            // Psychological state shapes ethical orientation over time.
            // Stressed people crave certainty → deontological drift.
            // Socially fulfilled people become communal → relational drift.
            // Engaged people develop care → virtue_care drift.
            // Disengaged people optimize selfishly → consequentialist drift.
            // Rate: 0.0003/tick ≈ 0.036/year — slow but persistent.
            let nd = 0.00015;
            let stress_d = if agent.needs.allostatic_load > 0.5 {
                nd
            } else {
                -nd * 0.5
            };
            agent.ethics.modify_with_sacred_resistance(0, stress_d);
            let social_d = if agent.needs.social_satiation > 0.6 {
                nd
            } else {
                -nd * 0.5
            };
            agent.ethics.modify_with_sacred_resistance(3, social_d);
            let engage_d = if agent.needs.engagement > 0.6 {
                nd
            } else {
                -nd * 0.5
            };
            agent.ethics.modify_with_sacred_resistance(2, engage_d);
            let disengage_d = if agent.needs.engagement < 0.4 {
                nd
            } else {
                -nd * 0.3
            };
            agent.ethics.modify_with_sacred_resistance(1, disengage_d);

            // Ethical switching cost (cognitive dissonance):
            // Rapid moral transitions are psychologically expensive.
            // When ethics shift by > 0.05 in a single tick, allostatic load increases.
            // Ref: Festinger (1957) cognitive dissonance theory.
            let ethics_after = agent.ethics.as_vec();
            let ethics_delta: f64 = ethics_before
                .iter()
                .zip(ethics_after.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            if ethics_delta > 0.05 {
                let dissonance_cost = (ethics_delta - 0.05) * 0.15;
                agent.needs.allostatic_load =
                    (agent.needs.allostatic_load + dissonance_cost).min(1.0);
            }

            // --- Ethics normalization: cap total ethics sum at 2.1 → 2.0 ---
            // Prevents runaway growth where all 4 dims inflate simultaneously.
            // Berlin (1969) value pluralism: genuine ethical tensions mean you can't
            // be maximally committed to ALL frameworks. Trade-offs are irreducible.
            let eth_sum = agent.ethics.deontological
                + agent.ethics.consequentialist
                + agent.ethics.virtue_care
                + agent.ethics.relational;
            if eth_sum > 2.05 {
                let scale = 2.0 / eth_sum;
                agent.ethics.deontological = (agent.ethics.deontological * scale).max(0.05);
                agent.ethics.consequentialist = (agent.ethics.consequentialist * scale).max(0.05);
                agent.ethics.virtue_care = (agent.ethics.virtue_care * scale).max(0.05);
                agent.ethics.relational = (agent.ethics.relational * scale).max(0.05);
            }

            // --- Moral emotions: guilt, moral injury, outrage ---

            // ── Guilt ──────────────────────────────────────────────────────────
            // Source 1: Survivor conscience — virtue/care agents feel guilty when
            // their community suffers, even when personally secure.
            // "I have what others lack — why?" (Tangney 2002: prosocial guilt).
            // High virtue_care × community trauma → consistent low-level guilt.
            let survivor_guilt = agent.ethics.virtue_care * community_mean_trauma * 0.005;

            // Source 2: Hypocrisy gap — stress-driven pragmatism diverges from stated values.
            // Only significant at allostatic_load > 0.5 (severe stress = survival mode).
            let revealed = agent.ethics.revealed(agent.needs.allostatic_load);
            let hypocrisy_gap: f64 = agent
                .ethics
                .as_vec()
                .iter()
                .zip(revealed.as_vec().iter())
                .map(|(s, r)| (s - r).abs())
                .sum::<f64>();
            let hypocrisy_guilt = hypocrisy_gap * 0.15;

            // Combined: slow decay (0.98 ≈ 4-year half-life at monthly resolution)
            agent.needs.affect.guilt =
                (agent.needs.affect.guilt * 0.98 + survivor_guilt + hypocrisy_guilt).min(1.0);

            // Guilt → consequentialist self-correction: feeling guilty about pragmatic
            // compromises triggers moral self-regulation (Tangney 2002).
            if agent.needs.affect.guilt > 0.2 {
                let correction = agent.needs.affect.guilt * 0.0008;
                agent.ethics.consequentialist =
                    (agent.ethics.consequentialist - correction).max(0.05);
            }

            // ── Moral injury (harm) ────────────────────────────────────────────
            // Only SEVERE hypocrisy (gap > 0.4) causes lasting moral injury.
            // Recovery rate (0.005) is 5× accumulation rate (0.001) — most people heal.
            // Ref: Litz et al. (2009) moral injury and moral repair.
            if hypocrisy_gap > 0.4 {
                agent.needs.affect.harm = (agent.needs.affect.harm + 0.001).min(1.0);
                if agent.needs.affect.harm > 0.5 {
                    agent.trauma_level = (agent.trauma_level + 0.0005).min(1.0);
                    agent.consciousness.coherence =
                        (agent.consciousness.coherence - 0.0003).max(0.0);
                }
            } else {
                agent.needs.affect.harm = (agent.needs.affect.harm - 0.005).max(0.0);
            }

            // ── Outrage ────────────────────────────────────────────────────────
            // Moral outrage: righteous indignation when the agent's deepest ethical
            // commitment is a minority view in their community.
            // "My people no longer share what I hold sacred." (Graham et al. 2011).
            // Fires when agent's sacred dim is significantly above community mean
            // on that dimension AND the agent is strongly committed (> 0.6).
            let (sacred_idx, sacred_val) = agent.ethics.sacred_dimension();
            let community_sacred = community_mean_ethics[sacred_idx];
            let moral_isolation = (sacred_val - community_sacred).max(0.0);
            if moral_isolation > 0.15 && sacred_val > 0.6 {
                // Outrage proportional to isolation × commitment strength
                let outrage_rise = (moral_isolation - 0.15) * sacred_val * 0.008;
                agent.needs.affect.outrage = (agent.needs.affect.outrage + outrage_rise).min(1.0);
            } else {
                // Slow recovery when moral isolation is low or agent uncommitted
                agent.needs.affect.outrage = (agent.needs.affect.outrage * 0.985).max(0.0);
            }
        }

        // --- Ethical diffusion: social contact spreads ethical orientations ---
        // Each tick, a fraction of agents contact a neighbor and blend ethics.
        // High coordination_understanding agents are better synthesizers (absorb more).
        // This models Bandura (1977) social learning + Haidt (2012) moral foundations
        // transmission through community interaction.
        // Rate: ~10% of agents per tick contact a neighbor, blend at 0.005 rate.
        {
            let living_indices: Vec<usize> = agents
                .iter()
                .enumerate()
                .filter(|(_, a)| a.is_alive())
                .map(|(i, _)| i)
                .collect();
            let n_contacts = (living_indices.len() as f64 * 0.10).ceil() as usize;
            for _ in 0..n_contacts {
                if living_indices.len() < 2 {
                    break;
                }
                let ai = living_indices[(rng.next_u64() as usize) % living_indices.len()];
                let bi = living_indices[(rng.next_u64() as usize) % living_indices.len()];
                if ai == bi {
                    continue;
                }
                // Moral leadership: high-consciousness agents are ethical beacons.
                // Guardian (tier 4) transmits at 5x, Steward (tier 3) at 2x.
                // This models Gandhi/Tutu/King — moral leaders who shift populations.
                // Receivers absorb faster with high coordination_understanding.
                let tier_mult = |tier: u8| -> f64 {
                    match tier {
                        4 => 5.0,
                        3 => 2.0,
                        2 => 1.0,
                        _ => 0.3,
                    }
                };
                let transmit_a = tier_mult(agents[ai].consciousness.tier());
                let transmit_b = tier_mult(agents[bi].consciousness.tier());
                let absorb_a = 1.0 + agents[ai].coordination_understanding * 0.5;
                let absorb_b = 1.0 + agents[bi].coordination_understanding * 0.5;
                // Rate reduced 0.02→0.005: faster diffusion caused rapid convergence
                // in large populations (4000 × 10% × 0.02 ≈ population homogenizes
                // in ~50 ticks). At 0.005, convergence takes ~200 ticks = 16 years.
                let blend_a = 0.005 * absorb_a * transmit_b; // A absorbs B's ethics
                let blend_b = 0.005 * absorb_b * transmit_a; // B absorbs A's ethics
                let ethics_a = agents[ai].ethics.as_vec();
                let ethics_b = agents[bi].ethics.as_vec();
                // A absorbs from B
                agents[ai].ethics.deontological += (ethics_b[0] - ethics_a[0]) * blend_a;
                agents[ai].ethics.consequentialist += (ethics_b[1] - ethics_a[1]) * blend_a;
                agents[ai].ethics.virtue_care += (ethics_b[2] - ethics_a[2]) * blend_a;
                agents[ai].ethics.relational += (ethics_b[3] - ethics_a[3]) * blend_a;
                // B absorbs from A
                agents[bi].ethics.deontological += (ethics_a[0] - ethics_b[0]) * blend_b;
                agents[bi].ethics.consequentialist += (ethics_a[1] - ethics_b[1]) * blend_b;
                agents[bi].ethics.virtue_care += (ethics_a[2] - ethics_b[2]) * blend_b;
                agents[bi].ethics.relational += (ethics_a[3] - ethics_b[3]) * blend_b;
            }
        }

        // --- Collective computation ---
        let living: Vec<&CivAgent> = agents.iter().filter(|a| a.is_alive()).collect();
        let n = living.len();

        if n == 0 {
            self.mean_phi = 0.0;
            self.inter_agent_coherence = 0.0;
            self.collective_phi = 0.0;
            self.tier_distribution = [0.0; 5];
            self.fragile_consensus = false;
            self.push_phi_history();
            return events;
        }

        // Sample up to COHERENCE_SAMPLE_AGENTS agents (stride-based)
        let stride = if n > COHERENCE_SAMPLE_AGENTS {
            n / COHERENCE_SAMPLE_AGENTS
        } else {
            1
        };
        let sampled: Vec<&CivAgent> = living.iter().step_by(stride).copied().collect();

        // Mean Phi from sampled agents
        self.mean_phi =
            sampled.iter().map(|a| a.consciousness.phi()).sum::<f64>() / sampled.len() as f64;

        // Inter-agent coherence: pairwise cosine similarity of 6D consciousness vectors
        if sampled.len() >= 2 {
            let mut coherence_sum = 0.0;
            let pairs = COHERENCE_SAMPLE_PAIRS.min(sampled.len() * (sampled.len() - 1) / 2);
            let mut pair_count = 0u32;
            for _ in 0..pairs {
                let i = (rng.next_u64() as usize) % sampled.len();
                let mut j = (rng.next_u64() as usize) % sampled.len();
                if j == i {
                    j = (i + 1) % sampled.len();
                }
                coherence_sum += consciousness_cosine_similarity(
                    &sampled[i].consciousness,
                    &sampled[j].consciousness,
                );
                pair_count += 1;
            }
            self.inter_agent_coherence = if pair_count > 0 {
                coherence_sum / pair_count as f64
            } else {
                0.0
            };

            // Bug #2 fix: Add institutional coherence floor.
            // Even a diverse population has shared institutions (language, governance,
            // infrastructure) that maintain collective coherence. Without this floor,
            // collective_phi collapses toward zero for large populations because
            // pairwise cosine similarity drops with diversity — physically wrong.
            // Floor: 0.2 base + 0.1 dampened by population size.
            let n = living.len() as f64;
            let institutional_floor = 0.2 + 0.1 / (1.0 + (n / 500.0).max(0.0).ln().max(0.0));
            self.inter_agent_coherence = self.inter_agent_coherence.max(institutional_floor);
        } else {
            self.inter_agent_coherence = 1.0; // single agent is perfectly coherent with itself
        }

        // No finite civilization achieves perfect consciousness — cap at 0.85.
        self.collective_phi = (self.mean_phi * self.inter_agent_coherence).min(0.85);

        // Tier distribution from ALL living agents
        let mut counts = [0usize; 5];
        for a in &living {
            counts[a.consciousness.tier() as usize] += 1;
        }
        for i in 0..5 {
            self.tier_distribution[i] = counts[i] as f64 / n as f64;
        }

        // Fragile consensus: >70% at one tier but collective Phi < 0.3
        let max_tier_frac = self
            .tier_distribution
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        self.fragile_consensus =
            max_tier_frac > FRAGILE_TIER_THRESHOLD && self.collective_phi < FRAGILE_PHI_THRESHOLD;

        // History
        self.push_phi_history();

        // Milestone: collective Phi crosses 0.5 for the first time
        if !self.phi_milestone_logged && self.collective_phi >= PHI_MILESTONE_THRESHOLD {
            self.phi_milestone_logged = true;
            events.push(CivEvent::new(
                current_tick,
                None,
                CivEventType::HarmonyMilestone,
                format!(
                    "Collective Phi reached {:.3} (threshold {PHI_MILESTONE_THRESHOLD})",
                    self.collective_phi
                ),
            ));
        }

        events
    }

    /// Slope of phi_history (positive = improving consciousness).
    pub fn phi_trend(&self) -> f64 {
        linear_slope(&self.phi_history)
    }

    /// Whether civilization consciousness is viable (collective Phi > 0.1).
    pub fn consciousness_viable(&self) -> bool {
        self.collective_phi > PHI_VIABLE_THRESHOLD
    }

    fn push_phi_history(&mut self) {
        self.phi_history.push(self.collective_phi);
        if self.phi_history.len() > PHI_HISTORY_LEN {
            self.phi_history.remove(0);
        }
    }
}

impl Default for ConsciousnessEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Fix 5: Consciousness development with growth model.
///
/// Growth rate depends on education, mentoring, trauma, and age-gating.
/// Plateau: growth halves every 0.2 above current level (diminishing returns).
///
/// Ref: Kegan (1982) stages of adult development; Fischer & Bidell (2006)
/// dynamic skill theory.
pub fn tick_consciousness_development(agent: &mut CivAgent, has_mentor: bool) {
    let _age = if agent.birth_tick == 0 { 30.0 } else { 0.0 }; // approximate; caller should pass tick
                                                               // We use consciousness.level as a proxy for development stage

    let base_rate = 0.001;
    let education_bonus = if agent.education_level > 0.3 {
        0.002
    } else {
        0.0
    };
    let mentor_bonus = if has_mentor { 0.003 } else { 0.0 };
    let trauma_penalty = if agent.trauma_level > 0.3 {
        -0.005 * agent.trauma_level
    } else {
        0.0
    };

    let raw_rate = base_rate + education_bonus + mentor_bonus + trauma_penalty;
    if raw_rate <= 0.0 {
        return;
    }

    // Plateau: growth halves every 0.2 above current level
    let current = agent.consciousness.level;
    let plateau_factor = 0.5_f64.powf(current / 0.2);
    let effective_rate = raw_rate * plateau_factor;

    agent.consciousness.level = (agent.consciousness.level + effective_rate).min(1.0);
}

/// Fix 5 variant with explicit age gating via current_tick.
pub fn tick_consciousness_development_with_age(
    agent: &mut CivAgent,
    has_mentor: bool,
    current_tick: u32,
) {
    let age_years = agent.age_years(current_tick);
    if age_years < 15.0 {
        return; // No consciousness growth below 15 years
    }

    let base_rate = 0.001;
    let education_bonus = if agent.education_level > 0.3 {
        0.002
    } else {
        0.0
    };
    let mentor_bonus = if has_mentor { 0.003 } else { 0.0 };
    let trauma_penalty = if agent.trauma_level > 0.3 {
        -0.005 * agent.trauma_level
    } else {
        0.0
    };

    let raw_rate = base_rate + education_bonus + mentor_bonus + trauma_penalty;
    if raw_rate <= 0.0 {
        return;
    }

    // Plateau: growth halves every 0.2 above current level
    let current = agent.consciousness.level;
    let plateau_factor = 0.5_f64.powf(current / 0.2);
    let effective_rate = raw_rate * plateau_factor;

    agent.consciousness.level = (agent.consciousness.level + effective_rate).min(1.0);
}

// --- Utility functions ---

/// 6D consciousness vector for an agent.
fn consciousness_vec(c: &crate::agent::ConsciousnessState) -> [f64; 6] {
    [
        c.level,
        c.meta_awareness,
        c.coherence,
        c.care_activation,
        c.harmonic_alignment,
        c.epistemic_confidence,
    ]
}

/// Cosine similarity between two 6D consciousness vectors.
fn consciousness_cosine_similarity(
    a: &crate::agent::ConsciousnessState,
    b: &crate::agent::ConsciousnessState,
) -> f64 {
    let va = consciousness_vec(a);
    let vb = consciousness_vec(b);
    let dot: f64 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = va.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = vb.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = mag_a * mag_b;
    if denom < 1e-15 {
        return 0.0;
    }
    (dot / denom).clamp(0.0, 1.0)
}

/// Cosine similarity between two 8-element vectors.
fn cosine_similarity_8(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = mag_a * mag_b;
    if denom < 1e-15 {
        return 0.0;
    }
    dot / denom
}

/// Least-squares slope of a time series (x = 0, 1, 2, ...).
fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let sum_x: f64 = (0..n).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_xx: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();
    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    (nf * sum_xy - sum_x * sum_y) / denom
}

/// Tick consciousness growth across all worlds.
///
/// Extracted from `MultiWorldSimulator::tick_consciousness()`. Handles:
/// - Red team adversarial recruitment (maintain ~5% adversarial population)
/// - Per-agent consciousness decay (earned, not permanent; load-amplified)
/// - Growth with diminishing returns, burnout penalty, pharma boost
/// - Trust-weighted governance gating
/// - Phi ceiling (moral humility at 0.95)
pub fn tick_consciousness_all_worlds(
    worlds: &mut [crate::world::World],
    current_tick: u32,
    pharma_boost: f64,
    trust_weighted_governance: bool,
    rng: &mut crate::stochastic::StochasticEngine,
) {
    use crate::agent;
    use crate::red_team;

    // Red team: maintain adversarial population at ~5% if any adversaries exist.
    for world in worlds.iter_mut() {
        let has_adversaries = world.agents.iter().any(|a| a.adversarial.is_some());
        if has_adversaries {
            let living = world.agents.iter().filter(|a| a.is_alive()).count();
            let adv_count = world
                .agents
                .iter()
                .filter(|a| a.is_alive() && a.adversarial.is_some())
                .count();
            let target = (living as f64 * 0.05).ceil() as usize;
            if adv_count < target {
                let deficit = target - adv_count;
                let mut recruited = 0;
                for agent in world
                    .agents
                    .iter_mut()
                    .filter(|a| a.is_alive() && a.adversarial.is_none())
                {
                    if recruited >= deficit {
                        break;
                    }
                    agent.adversarial = Some(red_team::AdversarialStrategy::ProfileMaximizer);
                    recruited += 1;
                }
            }
        }
    }

    // Phase 7: Gradual consciousness growth for agents.
    for world in worlds.iter_mut() {
        let mean_edu: f64 = {
            let living: Vec<f64> = world
                .agents
                .iter()
                .filter(|a| a.is_alive())
                .map(|a| a.education_level)
                .collect();
            if living.is_empty() {
                0.0
            } else {
                living.iter().sum::<f64>() / living.len() as f64
            }
        };

        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
            let stage = agent.life_stage(current_tick);
            let growth_rate = match stage {
                agent::LifeStage::Child => 0.001,
                agent::LifeStage::Youth => 0.003,
                agent::LifeStage::Adult => 0.002,
                agent::LifeStage::Elder => 0.001,
            };

            let base_decay = if stage == agent::LifeStage::Adult || stage == agent::LifeStage::Elder
            {
                0.0015
            } else {
                0.0
            };
            let decay = base_decay * (1.0 + agent.needs.allostatic_load);
            let c = &mut agent.consciousness;
            c.level = (c.level - decay).max(0.0);
            c.meta_awareness = (c.meta_awareness - decay).max(0.0);
            c.coherence = (c.coherence - decay).max(0.0);
            c.care_activation = (c.care_activation - decay).max(0.0);
            c.harmonic_alignment = (c.harmonic_alignment - decay).max(0.0);
            c.epistemic_confidence = (c.epistemic_confidence - decay).max(0.0);

            let pharma = if world.infrastructure_level > 0.3 {
                pharma_boost
            } else {
                0.0
            };
            let gating_factor = if trust_weighted_governance {
                let phi = c.phi();
                let bonus = 0.5 / (1.0 + (-10.0 * (phi - 0.3)).exp());
                0.5 + bonus
            } else {
                0.5
            };
            let amplifier = (1.0 + mean_edu * 0.5 + pharma) * gating_factor;
            let burnout_penalty = if agent.needs.is_burnout() { 0.35 } else { 1.0 };
            let amplifier = amplifier * burnout_penalty;

            let amplifier = if let Some(strategy) = &agent.adversarial {
                let modifier = red_team::AdversarialModifier::for_strategy(*strategy, 0.01);
                amplifier * modifier.phi_growth_mult
            } else {
                amplifier
            };

            if agent.needs.social_satiation > 0.5 {
                c.care_activation = (c.care_activation + 0.0005).min(1.0);
            }
            let cap = 0.85;
            let dr = |dim: f64, rate: f64| -> f64 {
                let headroom = (1.0 - dim / cap).max(0.0);
                rate * amplifier * headroom
            };
            c.level = (c.level + dr(c.level, growth_rate)).min(cap);
            c.meta_awareness =
                (c.meta_awareness + dr(c.meta_awareness, growth_rate * 0.8)).min(cap);
            c.coherence = (c.coherence + dr(c.coherence, growth_rate * 0.6)).min(cap);
            c.care_activation =
                (c.care_activation + dr(c.care_activation, growth_rate * 0.7)).min(cap);
            c.harmonic_alignment =
                (c.harmonic_alignment + dr(c.harmonic_alignment, growth_rate * 0.5)).min(cap);
            c.epistemic_confidence =
                (c.epistemic_confidence + dr(c.epistemic_confidence, growth_rate * 0.4)).min(cap);

            let phi = c.phi();
            if phi > 0.95 {
                let scale = 0.95 / phi;
                c.level *= scale;
                c.meta_awareness *= scale;
                c.coherence *= scale;
                c.care_activation *= scale;
                c.harmonic_alignment *= scale;
                c.epistemic_confidence *= scale;
            }
        }
    }

    // ── Ethical dynamics: per-world moral psychology + diffusion ──────────────
    // This runs AFTER the consciousness growth loop because ethics modulate
    // consciousness dimensions (coupling below) and benefit from stable phi.
    for world in worlds.iter_mut() {
        // Pre-compute community-level stats for social moral emotions.
        let community_mean_trauma: f64 = {
            let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
            let n = living.len().max(1) as f64;
            if living.is_empty() {
                0.0
            } else {
                living.iter().map(|a| a.trauma_level).sum::<f64>() / n
            }
        };
        let community_mean_ethics: [f64; 4] = {
            let living: Vec<_> = world.agents.iter().filter(|a| a.is_alive()).collect();
            let n = living.len().max(1) as f64;
            if living.is_empty() {
                [0.5; 4]
            } else {
                [
                    living.iter().map(|a| a.ethics.deontological).sum::<f64>() / n,
                    living
                        .iter()
                        .map(|a| a.ethics.consequentialist)
                        .sum::<f64>()
                        / n,
                    living.iter().map(|a| a.ethics.virtue_care).sum::<f64>() / n,
                    living.iter().map(|a| a.ethics.relational).sum::<f64>() / n,
                ]
            }
        };

        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
            // Age drift: elders develop virtue/care and relational wisdom (Tornstam 1989).
            let age_in_years = if current_tick > agent.birth_tick {
                (current_tick - agent.birth_tick) as f64 / 12.0
            } else {
                0.0
            };
            if age_in_years > 50.0 {
                agent.ethics.virtue_care = (agent.ethics.virtue_care + 0.0001).min(1.0);
                agent.ethics.relational = (agent.ethics.relational + 0.0001).min(1.0);
            }

            // Ethics-consciousness coupling: each framework builds a different dimension.
            agent.consciousness.care_activation =
                (agent.consciousness.care_activation + agent.ethics.virtue_care * 0.0007).min(1.0);
            agent.consciousness.harmonic_alignment = (agent.consciousness.harmonic_alignment
                + agent.ethics.relational * 0.0007)
                .min(1.0);
            agent.consciousness.coherence =
                (agent.consciousness.coherence + agent.ethics.deontological * 0.0005).min(1.0);
            agent.consciousness.epistemic_confidence = (agent.consciousness.epistemic_confidence
                + agent.ethics.consequentialist * 0.0005)
                .min(1.0);

            // Ethics-needs coupling: symmetric dead-zone coupling prevents one-directional
            // drift in moderate conditions. Each variable has a HIGH trigger (dim rises),
            // LOW trigger (dim falls), and NEUTRAL zone (no change).
            // Berlin (1969) value pluralism: stability in moderate conditions is natural.
            let nd = 0.00015;
            // Stress: high → deontological (need rules); low → virtue (safety enables care)
            let stress_d = if agent.needs.allostatic_load > 0.6 {
                nd
            } else if agent.needs.allostatic_load < 0.2 {
                -nd * 0.3
            } else {
                0.0
            };
            agent.ethics.modify_with_sacred_resistance(0, stress_d); // deontological
                                                                     // Social: high satiation → relational; isolated → relational falls
            let social_d = if agent.needs.social_satiation > 0.7 {
                nd
            } else if agent.needs.social_satiation < 0.3 {
                -nd * 0.5
            } else {
                0.0
            };
            agent.ethics.modify_with_sacred_resistance(3, social_d); // relational
                                                                     // Engagement: zero-sum — high → virtue up + conseq down; low → conseq up + virtue down
            if agent.needs.engagement > 0.7 {
                agent.ethics.modify_with_sacred_resistance(2, nd); // virtue up
                agent.ethics.modify_with_sacred_resistance(1, -nd * 0.4); // conseq slightly down
            } else if agent.needs.engagement < 0.3 {
                agent.ethics.modify_with_sacred_resistance(1, nd); // conseq up
                agent.ethics.modify_with_sacred_resistance(2, -nd * 0.4); // virtue slightly down
            }
            // Neutral zone [0.3, 0.7]: no engagement-driven pressure

            // Ethics normalization: prevent all-dimensions inflation.
            // Berlin (1969): genuine ethical tensions make maximal commitment impossible.
            let eth_sum = agent.ethics.deontological
                + agent.ethics.consequentialist
                + agent.ethics.virtue_care
                + agent.ethics.relational;
            if eth_sum > 2.05 {
                let scale = 2.0 / eth_sum;
                agent.ethics.deontological = (agent.ethics.deontological * scale).max(0.05);
                agent.ethics.consequentialist = (agent.ethics.consequentialist * scale).max(0.05);
                agent.ethics.virtue_care = (agent.ethics.virtue_care * scale).max(0.05);
                agent.ethics.relational = (agent.ethics.relational * scale).max(0.05);
            }

            // ── Guilt ────────────────────────────────────────────────────────
            // Survivor conscience: virtue/care agents feel guilty witnessing suffering.
            let survivor_guilt = agent.ethics.virtue_care * community_mean_trauma * 0.005;
            // Hypocrisy gap: stress-driven pragmatism vs stated values.
            let revealed = agent.ethics.revealed(agent.needs.allostatic_load);
            let hypocrisy_gap: f64 = agent
                .ethics
                .as_vec()
                .iter()
                .zip(revealed.as_vec().iter())
                .map(|(s, r)| (s - r).abs())
                .sum::<f64>();
            agent.needs.affect.guilt =
                (agent.needs.affect.guilt * 0.98 + survivor_guilt + hypocrisy_gap * 0.15).min(1.0);
            // Guilt self-corrects consequentialist drift.
            if agent.needs.affect.guilt > 0.2 {
                let correction = agent.needs.affect.guilt * 0.0008;
                agent.ethics.consequentialist =
                    (agent.ethics.consequentialist - correction).max(0.05);
            }

            // ── Moral injury (harm) ──────────────────────────────────────────
            // Severe hypocrisy (> 0.4 gap) causes lasting moral injury; heals 5× faster.
            if hypocrisy_gap > 0.4 {
                agent.needs.affect.harm = (agent.needs.affect.harm + 0.001).min(1.0);
            } else {
                agent.needs.affect.harm = (agent.needs.affect.harm - 0.005).max(0.0);
            }

            // ── Outrage ──────────────────────────────────────────────────────
            // Moral outrage: principled resistance when community leans heavily consequentialist
            // while the agent retains virtue/deontological commitments.
            // Works in converged societies (no sacred-dim comparison needed).
            // "My community rationalizes everything — I cannot accept this." (Graham 2011).
            let community_conseq = community_mean_ethics[1];
            let agent_non_conseq = agent.ethics.virtue_care + agent.ethics.deontological;
            // community_pressure: how far above neutral (0.5) the community is
            let community_pressure = (community_conseq - 0.5).max(0.0);
            // moral_resistance: agent's combined non-consequentialist commitment above 0.5
            let moral_resistance = (agent_non_conseq - 0.5).max(0.0);
            if community_pressure > 0.1 && moral_resistance > 0.05 {
                let outrage_rise = community_pressure * moral_resistance * 0.012;
                agent.needs.affect.outrage = (agent.needs.affect.outrage + outrage_rise).min(1.0);
            } else {
                agent.needs.affect.outrage = (agent.needs.affect.outrage * 0.985).max(0.0);
            }
        }

        // ── Ethical diffusion: social contact spreads ethics ─────────────────
        // 10% of agents per tick contact a random neighbor and blend ethics.
        // Rate 0.005 (was 0.02) to preserve diversity in large populations.
        {
            let living_indices: Vec<usize> = world
                .agents
                .iter()
                .enumerate()
                .filter(|(_, a)| a.is_alive())
                .map(|(i, _)| i)
                .collect();
            let n_contacts = (living_indices.len() as f64 * 0.10).ceil() as usize;
            for _ in 0..n_contacts {
                if living_indices.len() < 2 {
                    break;
                }
                let ai = living_indices[(rng.next_u64() as usize) % living_indices.len()];
                let bi = living_indices[(rng.next_u64() as usize) % living_indices.len()];
                if ai == bi {
                    continue;
                }
                let tier_mult = |tier: u8| -> f64 {
                    match tier {
                        4 => 5.0,
                        3 => 2.0,
                        2 => 1.0,
                        _ => 0.3,
                    }
                };
                let transmit_a = tier_mult(world.agents[ai].consciousness.tier());
                let transmit_b = tier_mult(world.agents[bi].consciousness.tier());
                let absorb_a = 1.0 + world.agents[ai].coordination_understanding * 0.5;
                let absorb_b = 1.0 + world.agents[bi].coordination_understanding * 0.5;
                let blend_a = 0.005 * absorb_a * transmit_b;
                let blend_b = 0.005 * absorb_b * transmit_a;
                let ea = world.agents[ai].ethics.as_vec();
                let eb = world.agents[bi].ethics.as_vec();
                world.agents[ai].ethics.deontological += (eb[0] - ea[0]) * blend_a;
                world.agents[ai].ethics.consequentialist += (eb[1] - ea[1]) * blend_a;
                world.agents[ai].ethics.virtue_care += (eb[2] - ea[2]) * blend_a;
                world.agents[ai].ethics.relational += (eb[3] - ea[3]) * blend_a;
                world.agents[bi].ethics.deontological += (ea[0] - eb[0]) * blend_b;
                world.agents[bi].ethics.consequentialist += (ea[1] - eb[1]) * blend_b;
                world.agents[bi].ethics.virtue_care += (ea[2] - eb[2]) * blend_b;
                world.agents[bi].ethics.relational += (ea[3] - eb[3]) * blend_b;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::stochastic::StochasticEngine;
    use crate::world::CulturalProfile;

    fn make_agent(id: u64, birth_tick: u32) -> CivAgent {
        CivAgent {
            id,
            birth_tick,
            death_tick: None,
            sex: BiologicalSex::Female,
            world_id: 0,
            health: 1.0,
            skills: SkillVector::new(),
            education_level: 0.5,
            consciousness: ConsciousnessState::nascent(),
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: crate::needs::PsychologicalNeeds::new(),
            tend_balance: 0.0,
            parent_ids: None,
            faction_id: None,
            generation: 0,
            trauma_level: 0.0,
            cumulative_dose_sv: 0.0,
            adversarial: None,
            coordination_understanding: 0.0,
            mycel_score: 0.1,
            sap_balance: 100.0,
            is_biological: true,
            wounds: Vec::new(),
            ethics: crate::agent::EthicalOrientation::default(),
            sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
            justice: crate::sub_passport::RestorativeJustice::new(),
        }
    }

    #[test]
    fn test_individual_consciousness_growth() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();
        let mut agents = vec![make_agent(0, 0)];
        agents[0].education_level = 0.8;

        let initial_level = agents[0].consciousness.level;
        for tick in 1..=120 {
            engine.tick_consciousness(&mut agents, &culture, tick, &mut rng);
        }
        // Education drives consciousness level up over time
        assert!(
            agents[0].consciousness.level > initial_level,
            "Level should grow with education: {} vs {}",
            agents[0].consciousness.level,
            initial_level
        );
    }

    #[test]
    fn test_collective_phi_from_agents() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();

        // Create agents with elevated consciousness
        let mut agents: Vec<CivAgent> = (0..20)
            .map(|i| {
                let mut a = make_agent(i, 0);
                a.consciousness.level = 0.7;
                a.consciousness.meta_awareness = 0.6;
                a.consciousness.coherence = 0.5;
                a.consciousness.care_activation = 0.5;
                a.consciousness.harmonic_alignment = 0.5;
                a.consciousness.epistemic_confidence = 0.5;
                a
            })
            .collect();

        engine.tick_consciousness(&mut agents, &culture, 1, &mut rng);

        assert!(engine.mean_phi > 0.0, "mean_phi should be positive");
        assert!(
            engine.collective_phi > 0.0,
            "collective_phi should be positive"
        );
        assert!(
            engine.inter_agent_coherence > 0.0,
            "coherence should be positive for similar agents"
        );
    }

    #[test]
    fn test_tier_distribution_sums_to_one() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();

        let mut agents: Vec<CivAgent> = (0..50).map(|i| make_agent(i, 0)).collect();
        engine.tick_consciousness(&mut agents, &culture, 100, &mut rng);

        let sum: f64 = engine.tier_distribution.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "tier distribution should sum to 1.0, was {sum}"
        );
    }

    #[test]
    fn test_fragile_consensus_detection() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();

        // All agents at tier 0 (nascent) with low phi — classic fragile consensus.
        // Low education ensures emergent coherence stays low (phi < 0.3).
        let mut agents: Vec<CivAgent> = (0..30)
            .map(|i| {
                let mut a = make_agent(i, 0);
                a.consciousness = ConsciousnessState::nascent();
                a.education_level = 0.05;
                a
            })
            .collect();

        engine.tick_consciousness(&mut agents, &culture, 1, &mut rng);

        // Nascent agents are all tier 0 (>70% at one level) and collective phi < 0.3
        assert!(
            engine.fragile_consensus,
            "Should detect fragile consensus when all agents are nascent"
        );
    }

    #[test]
    fn test_phi_history_tracking() {
        let mut engine = ConsciousnessEngine::new();
        let mut rng = StochasticEngine::new(42);
        let culture = CulturalProfile::earth_default();
        let mut agents = vec![make_agent(0, 0)];

        for tick in 1..=10 {
            engine.tick_consciousness(&mut agents, &culture, tick, &mut rng);
        }

        assert_eq!(
            engine.phi_history.len(),
            10,
            "Should have 10 history entries after 10 ticks"
        );

        // Run 200 more to test truncation
        for tick in 11..=200 {
            engine.tick_consciousness(&mut agents, &culture, tick, &mut rng);
        }
        assert!(
            engine.phi_history.len() <= PHI_HISTORY_LEN,
            "History should be capped at {PHI_HISTORY_LEN}, was {}",
            engine.phi_history.len()
        );
    }

    #[test]
    fn test_consciousness_viable_threshold() {
        let mut engine = ConsciousnessEngine::new();
        assert!(
            !engine.consciousness_viable(),
            "Zero collective phi should not be viable"
        );

        engine.collective_phi = 0.05;
        assert!(!engine.consciousness_viable());

        engine.collective_phi = 0.15;
        assert!(engine.consciousness_viable());
    }

    #[test]
    fn test_phi_trend_positive_for_growing_series() {
        let mut engine = ConsciousnessEngine::new();
        engine.phi_history = (0..20).map(|i| i as f64 * 0.01).collect();
        assert!(
            engine.phi_trend() > 0.0,
            "Trend should be positive for increasing series"
        );
    }

    #[test]
    fn test_consciousness_grows_over_time() {
        // Fix 5: Consciousness Development
        let mut agent = make_agent(0, 0);
        agent.education_level = 0.5;
        let initial_level = agent.consciousness.level;

        // 100 ticks of development with mentor
        for _ in 0..100 {
            tick_consciousness_development(&mut agent, true);
        }

        assert!(
            agent.consciousness.level > initial_level,
            "Consciousness should grow: {} vs {}",
            agent.consciousness.level,
            initial_level
        );
    }

    #[test]
    fn test_consciousness_age_gate() {
        // Fix 5: No growth below 15 years
        let mut child = make_agent(0, 100); // born at tick 100
        child.education_level = 0.8;
        let initial_level = child.consciousness.level;

        // Tick at age 10 (tick 220)
        tick_consciousness_development_with_age(&mut child, true, 220);
        assert!(
            (child.consciousness.level - initial_level).abs() < 1e-10,
            "Child should not grow consciousness"
        );

        // Tick at age 20 (tick 340)
        tick_consciousness_development_with_age(&mut child, true, 340);
        assert!(
            child.consciousness.level > initial_level,
            "Adult should grow consciousness"
        );
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let c = ConsciousnessState {
            level: 0.5,
            meta_awareness: 0.5,
            coherence: 0.5,
            care_activation: 0.5,
            harmonic_alignment: 0.5,
            epistemic_confidence: 0.5,
        };
        let sim = consciousness_cosine_similarity(&c, &c);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Identical vectors should have similarity 1.0, was {sim}"
        );
    }
}
