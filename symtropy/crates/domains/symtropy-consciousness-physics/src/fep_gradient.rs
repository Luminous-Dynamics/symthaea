// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Free Energy Gradient: agents move to minimize prediction error.
//!
//! This is the missing ingredient the definitive experiment identified.
//! Thermodynamic enforcement creates the pressure. This module creates
//! the RESPONSE — agents that seek low-cost zones by following the
//! free energy gradient downhill.
//!
//! An agent that truly minimizes variational free energy will:
//! 1. Move toward areas where predictions are accurate (low surprise)
//! 2. Move toward resonant partners (lower processing cost)
//! 3. Move toward energy sources (reduce depletion threat)
//! 4. Avoid high-entropy zones (Leviathan, collision areas)
//!
//! Implementation: sample free energy in 8 compass directions,
//! move in the direction that reduces it most. Pure gradient descent
//! on the variational free energy landscape.

use crate::harmony_field::{HarmonyField, NUM_HARMONIES};
use nalgebra::SVector;

/// Compute the preferred movement direction for an agent to minimize free energy.
///
/// Returns a unit direction vector (or zero if no gradient found).
///
/// `position`: agent's current position
/// `energy_fraction`: agent's energy / max_energy [0, 1]
/// `harmony`: agent's harmony activation vector
/// `nearby_agents`: list of (position, harmony_activations) for other agents
/// `energy_wells`: list of (position, remaining_fraction) for energy wells
/// `danger_source`: optional position of the threat (Leviathan)
/// `danger_level`: [0, 1]
/// Original Φ-independent gradient (backward compatible).
pub fn free_energy_gradient<const D: usize>(
    position: &SVector<f64, D>,
    energy_fraction: f64,
    harmony: &[f64; NUM_HARMONIES],
    nearby_agents: &[(SVector<f64, D>, [f64; NUM_HARMONIES])],
    energy_wells: &[(SVector<f64, D>, f64)],
    danger_source: Option<&SVector<f64, D>>,
    danger_level: f64,
) -> SVector<f64, D> {
    free_energy_gradient_phi(
        position,
        energy_fraction,
        None,
        harmony,
        nearby_agents,
        energy_wells,
        danger_source,
        danger_level,
    )
}

/// Extended version with Φ as an explicit parameter.
///
/// When `phi` is provided, it modulates:
/// 1. Cooperation urgency: scaled by (1 + phi) — higher consciousness = stronger social drive
/// 2. Resonance gating: only attracted to agents whose harmony resonance × phi > 0.3
/// 3. Danger sensitivity: higher phi = detect danger earlier (wider avoidance)
///
/// Call with phi = None for the Φ-independent baseline.
#[allow(clippy::too_many_arguments)]
pub fn free_energy_gradient_phi<const D: usize>(
    position: &SVector<f64, D>,
    energy_fraction: f64,
    phi: Option<f64>,
    harmony: &[f64; NUM_HARMONIES],
    nearby_agents: &[(SVector<f64, D>, [f64; NUM_HARMONIES])],
    energy_wells: &[(SVector<f64, D>, f64)],
    danger_source: Option<&SVector<f64, D>>,
    danger_level: f64,
) -> SVector<f64, D> {
    let mut gradient = SVector::zeros();
    let mut total_weight = 0.0;

    let phi_val = phi.unwrap_or(0.5);

    // === 1. Seek resonant partners ===
    // Φ-coupled: higher consciousness = stronger social drive
    // This is the key wiring: cooperation_urgency now scales with Φ
    let cooperation_urgency = (1.0 - energy_fraction) * (0.5 + phi_val); // Φ amplifies social drive

    for (agent_pos, agent_harmony) in nearby_agents {
        let delta = agent_pos - position;
        let dist = delta.norm();
        if !(1.0..=100.0).contains(&dist) {
            continue;
        }

        let resonance = HarmonyField::<D>::resonance(harmony, agent_harmony);
        // Φ-coupled resonance gating: need resonance × phi_val > 0.15 to feel attraction
        // High Φ agents are attracted to weaker resonances (more socially aware)
        // Low/zero Φ agents only attracted to very strong resonances
        let effective_resonance = resonance * (0.3 + phi_val * 0.7);
        if effective_resonance > 0.15 {
            let attraction = effective_resonance * cooperation_urgency * 2.0;
            gradient += delta / dist * attraction;
            total_weight += attraction;
        }
    }

    // === 2. Seek energy wells (reduce depletion threat) ===
    // Weight increases sharply as energy drops below 50%
    let energy_urgency = if energy_fraction < 0.5 {
        (0.5 - energy_fraction) * 4.0 // 0→2 as energy drops 50%→0%
    } else {
        0.0
    };

    for (well_pos, well_remaining) in energy_wells {
        if *well_remaining < 0.01 {
            continue;
        } // depleted well
        let delta = well_pos - position;
        let dist = delta.norm();
        if !(1.0..=200.0).contains(&dist) {
            continue;
        }

        let attraction = energy_urgency * well_remaining;
        gradient += delta / dist * attraction;
        total_weight += attraction;
    }

    // === 3. Flee from danger (avoid high-entropy zones) ===
    // Φ-coupled: higher consciousness = detect danger at lower thresholds
    let danger_threshold = 0.5 - phi_val * 0.4; // Φ=1 detects at 0.1, Φ=0 detects at 0.5
    #[allow(clippy::collapsible_if)]
    if let Some(danger_pos) = danger_source {
        if danger_level > danger_threshold {
            let delta = position - danger_pos; // AWAY from danger
            let dist = delta.norm();
            if dist > 1.0 && dist < 300.0 {
                let repulsion = danger_level * 3.0;
                gradient += delta / dist * repulsion;
                total_weight += repulsion;
            }
        }
    }

    // === 4. Exploration (when energy is high and no strong gradient) ===
    // Low free energy = confident predictions = explore for new information
    if energy_fraction > 0.7 && total_weight < 0.5 {
        // Gentle random walk (deterministic from position for reproducibility)
        let phase = position[0] * 0.1 + position[1 % D] * 0.07;
        let mut explore = SVector::zeros();
        explore[0] = phase.sin() * 0.3;
        if D > 1 {
            explore[1] = phase.cos() * 0.3;
        }
        gradient += explore;
    }

    // Normalize to unit direction
    let norm = gradient.norm();
    if norm > 1e-6 {
        gradient / norm
    } else {
        SVector::zeros()
    }
}

/// Learnable FEP gradient weights per agent.
///
/// Replaces hardcoded constants with per-agent adaptive weights, updated by
/// a reward-modulated plasticity rule (NOT Hebbian — this uses global reward,
/// not local pre/post correlations).
///
/// Initialize with `default()` for backward-compatible values matching the
/// current hardcoded constants.
#[derive(Debug, Clone)]
pub struct LearnedFepWeights {
    /// Weight for cooperation/social drive (default: 2.0).
    pub cooperation: f64,
    /// Weight for energy well attraction (default: 4.0).
    pub energy_well: f64,
    /// Weight for danger avoidance (default: 3.0).
    pub danger: f64,
    /// Weight for exploration when energy high (default: 0.3).
    pub exploration: f64,
    /// Learning rate for reward-modulated updates.
    pub learning_rate: f64,
    /// Momentum accumulator per weight.
    pub momentum: [f64; 4],
    /// Previous energy fraction (for reward computation).
    pub prev_energy: f64,
}

impl Default for LearnedFepWeights {
    fn default() -> Self {
        Self {
            cooperation: 2.0,
            energy_well: 4.0,
            danger: 3.0,
            exploration: 0.3,
            learning_rate: 0.01,
            momentum: [0.0; 4],
            prev_energy: 1.0,
        }
    }
}

impl LearnedFepWeights {
    /// Update weights via reward-modulated plasticity.
    ///
    /// Reward = current_energy - prev_energy (positive = gained energy).
    /// Each weight is nudged toward values that produced positive outcomes.
    /// Uses momentum (0.9) for stability.
    ///
    /// `contributions`: [cooperation_contrib, well_contrib, danger_contrib, explore_contrib]
    /// — the raw magnitude each component contributed to the last gradient.
    pub fn update(&mut self, current_energy: f64, contributions: [f64; 4]) {
        let reward = current_energy - self.prev_energy;
        self.prev_energy = current_energy;

        if reward.abs() < 1e-6 {
            return;
        } // no signal

        let lr = self.learning_rate;
        let beta = 0.9; // momentum coefficient
        // Update each weight with reward-modulated plasticity
        let grads: [f64; 4] = std::array::from_fn(|i| {
            let grad = reward * contributions[i];
            self.momentum[i] = beta * self.momentum[i] + (1.0 - beta) * grad;
            lr * self.momentum[i]
        });
        self.cooperation = (self.cooperation + grads[0]).clamp(0.01, 20.0);
        self.energy_well = (self.energy_well + grads[1]).clamp(0.01, 20.0);
        self.danger = (self.danger + grads[2]).clamp(0.01, 20.0);
        self.exploration = (self.exploration + grads[3]).clamp(0.01, 20.0);
    }

    /// Update weights using a windowed reward signal (from AgentMemory).
    ///
    /// The windowed reward averages energy trajectory over ~100 ticks,
    /// providing a much stronger gradient signal than per-tick deltas.
    /// This is the recommended update method for experiments with memory.
    pub fn update_windowed(&mut self, windowed_reward: f64, contributions: [f64; 4]) {
        if windowed_reward.abs() < 1e-4 {
            return;
        } // no signal

        let lr = self.learning_rate * 5.0; // stronger signal = can use higher LR
        let beta = 0.9;
        let grads: [f64; 4] = std::array::from_fn(|i| {
            let grad = windowed_reward * contributions[i];
            self.momentum[i] = beta * self.momentum[i] + (1.0 - beta) * grad;
            lr * self.momentum[i]
        });
        self.cooperation = (self.cooperation + grads[0]).clamp(0.01, 20.0);
        self.energy_well = (self.energy_well + grads[1]).clamp(0.01, 20.0);
        self.danger = (self.danger + grads[2]).clamp(0.01, 20.0);
        self.exploration = (self.exploration + grads[3]).clamp(0.01, 20.0);
    }

    /// Summary of learned weight drift from defaults.
    pub fn drift_from_default(&self) -> f64 {
        let d = Self::default();
        (self.cooperation - d.cooperation).abs()
            + (self.energy_well - d.energy_well).abs()
            + (self.danger - d.danger).abs()
            + (self.exploration - d.exploration).abs()
    }
}

/// Extended FEP gradient with learnable weights.
///
/// Same as `free_energy_gradient_phi` but uses per-agent `LearnedFepWeights`
/// instead of hardcoded constants. Returns both the gradient direction and
/// the per-component contribution magnitudes (for weight update).
#[allow(clippy::too_many_arguments)]
pub fn free_energy_gradient_learned<const D: usize>(
    position: &SVector<f64, D>,
    energy_fraction: f64,
    phi: Option<f64>,
    harmony: &[f64; NUM_HARMONIES],
    nearby_agents: &[(SVector<f64, D>, [f64; NUM_HARMONIES])],
    energy_wells: &[(SVector<f64, D>, f64)],
    danger_source: Option<&SVector<f64, D>>,
    danger_level: f64,
    weights: &LearnedFepWeights,
) -> (SVector<f64, D>, [f64; 4]) {
    let mut gradient: SVector<f64, D> = SVector::zeros();
    let phi_val = phi.unwrap_or(0.5);
    let mut contributions = [0.0f64; 4];

    // === 1. Seek resonant partners (cooperation) ===
    let cooperation_urgency = (1.0 - energy_fraction) * (0.5 + phi_val);
    let mut coop_grad: SVector<f64, D> = SVector::zeros();
    for (agent_pos, agent_harmony) in nearby_agents {
        let delta = agent_pos - position;
        let dist = delta.norm();
        if !(1.0..=100.0).contains(&dist) {
            continue;
        }
        let resonance = HarmonyField::<D>::resonance(harmony, agent_harmony);
        let effective_resonance = resonance * (0.3 + phi_val * 0.7);
        if effective_resonance > 0.15 {
            let attraction = effective_resonance * cooperation_urgency;
            coop_grad += delta / dist * attraction;
        }
    }
    contributions[0] = coop_grad.norm();
    gradient += coop_grad * weights.cooperation;

    // === 2. Seek energy wells ===
    let energy_urgency = if energy_fraction < 0.5 {
        (0.5 - energy_fraction) * 4.0
    } else {
        0.0
    };
    let mut well_grad: SVector<f64, D> = SVector::zeros();
    for (well_pos, well_remaining) in energy_wells {
        if *well_remaining < 0.01 {
            continue;
        }
        let delta = well_pos - position;
        let dist = delta.norm();
        if !(1.0..=200.0).contains(&dist) {
            continue;
        }
        well_grad += delta / dist * (energy_urgency * *well_remaining);
    }
    contributions[1] = well_grad.norm();
    gradient += well_grad * weights.energy_well;

    // === 3. Flee danger ===
    let danger_threshold = 0.5 - phi_val * 0.4;
    let mut danger_grad: SVector<f64, D> = SVector::zeros();
    #[allow(clippy::collapsible_if)]
    if let Some(danger_pos) = danger_source {
        if danger_level > danger_threshold {
            let delta = position - danger_pos;
            let dist = delta.norm();
            if dist > 1.0 && dist < 300.0 {
                danger_grad += delta / dist * danger_level;
            }
        }
    }
    contributions[2] = danger_grad.norm();
    gradient += danger_grad * weights.danger;

    // === 4. Exploration ===
    let mut explore_grad: SVector<f64, D> = SVector::zeros();
    if energy_fraction > 0.7 && gradient.norm() < 0.5 {
        let phase = position[0] * 0.1 + position[1 % D] * 0.07;
        explore_grad[0] = phase.sin();
        if D > 1 {
            explore_grad[1] = phase.cos();
        }
    }
    contributions[3] = explore_grad.norm();
    gradient += explore_grad * weights.exploration;

    // Normalize
    let norm = gradient.norm();
    let dir = if norm > 1e-6 {
        gradient / norm
    } else {
        SVector::zeros()
    };
    (dir, contributions)
}

/// Compute a scalar "free energy" estimate at a position.
///
/// Lower is better. Agents should move to minimize this.
pub fn estimate_free_energy(
    energy_fraction: f64,
    prediction_error: f64,
    nearest_resonant_dist: f64,
    nearest_well_dist: f64,
    danger_level: f64,
) -> f64 {
    // Free energy = surprise + expected future cost
    let surprise = prediction_error;
    let depletion_risk = (1.0 - energy_fraction).powi(2); // quadratic urgency
    let isolation_cost = (nearest_resonant_dist / 50.0).min(1.0); // normalized
    let well_distance_cost = (nearest_well_dist / 100.0).min(1.0);
    let threat_cost = danger_level;

    // Weighted sum
    surprise * 2.0
        + depletion_risk * 3.0
        + isolation_cost * 1.5
        + well_distance_cost * 1.0
        + threat_cost * 2.5
}

/// Φ-gravity: gravitational attraction proportional to integration metrics.
///
/// F = G × Φ_self × Φ_other / r² (Plummer-softened)
///
/// High-Φ agents attract each other. Low-Φ agents are gravitationally
/// invisible. Creates natural hierarchy — "integration wells" that
/// pull others into orbit.
///
/// Returns the gravitational acceleration vector (add to velocity).
pub fn phi_gravity<const D: usize>(
    position: &SVector<f64, D>,
    self_phi: f64,
    nearby_agents: &[(SVector<f64, D>, f64)], // (position, phi)
    gravitational_constant: f64,              // coupling strength (default ~0.5)
) -> SVector<f64, D> {
    let mut accel = SVector::zeros();
    let softening = 1.0; // Plummer softening to prevent singularity

    for (other_pos, other_phi) in nearby_agents {
        let delta = other_pos - position;
        let dist_sq = delta.norm_squared() + softening * softening;
        let dist = dist_sq.sqrt();

        if dist < 1e-6 {
            continue;
        }

        // F = G × Φ₁ × Φ₂ / r² (Plummer-softened)
        let force_mag = gravitational_constant * self_phi * other_phi / dist_sq;

        // Direction: toward the other agent
        accel += delta / dist * force_mag;
    }

    // Clamp to prevent extreme accelerations
    let mag = accel.norm();
    if mag > 20.0 {
        accel *= 20.0 / mag;
    }
    accel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_energy_seeks_wells() {
        let pos = SVector::from([0.0, 0.0]);
        let harmony = [0.5; NUM_HARMONIES];
        let wells = vec![
            (SVector::from([50.0, 0.0]), 1.0), // well to the right
        ];

        let dir = free_energy_gradient(&pos, 0.1, &harmony, &[], &wells, None, 0.0);

        // Should move toward the well (positive x)
        assert!(
            dir[0] > 0.0,
            "should seek well when low energy, got x={}",
            dir[0]
        );
    }

    #[test]
    fn full_energy_ignores_wells() {
        let pos = SVector::from([0.0, 0.0]);
        let harmony = [0.5; NUM_HARMONIES];
        let wells = vec![(SVector::from([50.0, 0.0]), 1.0)];

        let dir = free_energy_gradient(&pos, 0.9, &harmony, &[], &wells, None, 0.0);

        // Should not strongly seek well when energy is high
        assert!(
            dir[0].abs() < 0.5,
            "should not urgently seek well at 90% energy"
        );
    }

    #[test]
    fn flees_from_danger() {
        let pos = SVector::from([0.0, 0.0]);
        let harmony = [0.5; NUM_HARMONIES];
        let danger = SVector::from([50.0, 0.0]); // danger to the right

        let dir = free_energy_gradient(&pos, 0.5, &harmony, &[], &[], Some(&danger), 0.8);

        // Should flee (negative x)
        assert!(dir[0] < 0.0, "should flee from danger, got x={}", dir[0]);
    }

    #[test]
    fn seeks_resonant_partner_when_low_energy() {
        let pos = SVector::from([0.0, 0.0]);
        let harmony = [0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.0];
        let partner_harmony = [0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.0]; // same = high resonance
        let agents = vec![(SVector::from([30.0, 0.0]), partner_harmony)];

        let dir = free_energy_gradient(&pos, 0.2, &harmony, &agents, &[], None, 0.0);

        assert!(dir[0] > 0.0, "should seek resonant partner when low energy");
    }

    #[test]
    fn does_not_seek_dissonant_agent() {
        let pos = SVector::from([0.0, 0.0]);
        let harmony = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let other_harmony = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]; // orthogonal
        let agents = vec![(SVector::from([30.0, 0.0]), other_harmony)];

        let dir = free_energy_gradient(&pos, 0.2, &harmony, &agents, &[], None, 0.0);

        // Orthogonal harmonies = resonance ~0 = no attraction
        assert!(dir[0].abs() < 0.3, "should not seek dissonant agent");
    }

    #[test]
    fn free_energy_decreases_with_energy() {
        let fe_low = estimate_free_energy(0.1, 0.5, 30.0, 50.0, 0.0);
        let fe_high = estimate_free_energy(0.9, 0.5, 30.0, 50.0, 0.0);
        assert!(fe_low > fe_high, "low energy = higher free energy");
    }

    #[test]
    fn free_energy_increases_with_danger() {
        let fe_safe = estimate_free_energy(0.5, 0.5, 30.0, 50.0, 0.0);
        let fe_danger = estimate_free_energy(0.5, 0.5, 30.0, 50.0, 0.9);
        assert!(fe_danger > fe_safe, "danger increases free energy");
    }

    #[test]
    fn gradient_is_unit_or_zero() {
        let pos = SVector::from([0.0, 0.0]);
        let harmony = [0.5; NUM_HARMONIES];
        let wells = vec![(SVector::from([50.0, 30.0]), 1.0)];

        let dir = free_energy_gradient(&pos, 0.2, &harmony, &[], &wells, None, 0.0);
        let norm = dir.norm();
        assert!(norm < 1.01, "gradient should be unit length, got {}", norm);
    }
}
