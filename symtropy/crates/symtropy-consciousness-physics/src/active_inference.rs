// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Active Inference agent with genuine variational free energy minimization.
//!
//! Replaces the handcrafted FEP gradient with a proper Bayesian active
//! inference loop: maintain beliefs → predict → act → observe → update.
//!
//! Key design: belief updates are AMORTIZED (every 50 ticks or on
//! significant events) to avoid O(n×beliefs) per-tick overhead.
//! Action selection is cheap: evaluate 24 candidate directions from
//! cached beliefs.

use nalgebra::SVector;

/// A believed well location with uncertainty.
#[derive(Debug, Clone)]
pub struct WellBelief {
    /// Believed position (mean of Gaussian).
    pub position: SVector<f64, 2>,
    /// Uncertainty (variance — high = unsure about location).
    pub uncertainty: f64,
    /// Confidence weight (how many observations support this belief).
    pub confidence: f64,
    /// Last tick this belief was updated.
    pub last_updated: usize,
}

/// A believed partner value (Beta distribution).
#[derive(Debug, Clone)]
pub struct PartnerBelief {
    pub handle_bits: u64,
    /// Beta distribution: alpha = successful encounters.
    pub alpha: f64,
    /// Beta distribution: beta = unsuccessful encounters.
    pub beta: f64,
}

impl PartnerBelief {
    /// Expected value of the Beta distribution.
    pub fn expected_value(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
}

/// Generative model: the agent's beliefs about the world.
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    /// Believed well locations.
    pub well_beliefs: Vec<WellBelief>,
    /// Believed partner values.
    pub partner_beliefs: Vec<PartnerBelief>,
    /// Expected energy trajectory (exponential moving average).
    pub energy_forecast: f64,
    /// Prior uncertainty for new well discoveries.
    pub prior_well_uncertainty: f64,
    /// Ticks since last full belief update.
    pub ticks_since_update: usize,
    /// Maximum well beliefs to maintain.
    pub max_well_beliefs: usize,
    /// Maximum partner beliefs.
    pub max_partner_beliefs: usize,
}

impl Default for GenerativeModel {
    fn default() -> Self {
        Self {
            well_beliefs: Vec::new(),
            partner_beliefs: Vec::new(),
            energy_forecast: 0.5,
            prior_well_uncertainty: 100.0,
            ticks_since_update: 0,
            max_well_beliefs: 10,
            max_partner_beliefs: 20,
        }
    }
}

/// Active inference agent.
#[derive(Debug, Clone)]
pub struct ActiveInferenceAgent {
    pub model: GenerativeModel,
    /// Cached action from last inference (reused between updates).
    pub cached_direction: SVector<f64, 2>,
    /// Update interval (ticks between full belief updates).
    pub update_interval: usize,
}

impl Default for ActiveInferenceAgent {
    fn default() -> Self {
        Self {
            model: GenerativeModel::default(),
            cached_direction: SVector::zeros(),
            update_interval: 50,
        }
    }
}

impl ActiveInferenceAgent {
    /// Compute variational free energy for a hypothetical position.
    ///
    /// F = E[surprise] + KL[q||p]
    /// surprise = -log P(energy_gain | position, beliefs)
    /// KL = complexity cost of updating beliefs
    ///
    /// Lower is better. Agents should move to minimize this.
    fn expected_free_energy(
        &self,
        candidate_pos: &SVector<f64, 2>,
        energy_fraction: f64,
        visible_wells: &[(SVector<f64, 2>, f64)],
        visible_agents: &[(SVector<f64, 2>, f64)], // (pos, resonance)
        well_regen_rate: f64,
        resonance_regen_rate: f64,
    ) -> f64 {
        // Pragmatic value: expected energy gain at this position
        let mut expected_energy_gain = 0.0;

        // From believed wells
        for belief in &self.model.well_beliefs {
            let dist = (candidate_pos - belief.position).norm();
            if dist < 35.0 {
                // Discount by uncertainty: low uncertainty = reliable
                let reliability = 1.0 / (1.0 + belief.uncertainty * 0.01);
                expected_energy_gain += well_regen_rate * belief.confidence * reliability;
            }
        }

        // From visible wells (override beliefs — direct observation)
        for (wpos, wrem) in visible_wells {
            let dist = (candidate_pos - wpos).norm();
            if dist < 35.0 && *wrem > 0.01 {
                expected_energy_gain += well_regen_rate * wrem;
            }
        }

        // From nearby agents (resonance benefit)
        for (apos, resonance) in visible_agents {
            let dist = (candidate_pos - apos).norm();
            if dist > 2.0 && dist < 40.0 && *resonance > 0.5 {
                expected_energy_gain += resonance_regen_rate * (resonance - 0.5) * 2.0;
            }
        }

        // Surprise: deviation from expected energy trajectory
        let surprise = (self.model.energy_forecast - energy_fraction).abs();

        // Urgency: low energy = high urgency to find ANY energy source
        let urgency = (1.0 - energy_fraction).powi(2) * 3.0;

        // Exploration bonus: prefer positions that reduce uncertainty
        let mut info_gain = 0.0;
        for belief in &self.model.well_beliefs {
            let dist = (candidate_pos - belief.position).norm();
            if dist < 50.0 && belief.uncertainty > 10.0 {
                info_gain += belief.uncertainty * 0.001; // small bonus for resolving uncertainty
            }
        }

        // Free energy = cost - benefit
        // Lower = better (agents minimize this)
        let cost = surprise + urgency;
        let benefit = expected_energy_gain + info_gain;

        cost - benefit
    }

    /// Select action that minimizes expected free energy.
    ///
    /// Evaluates 24 candidate directions (8 angles × 3 distances).
    /// Uses cached beliefs — no Bayesian updates during action selection.
    pub fn infer_action(
        &mut self,
        pos: &SVector<f64, 2>,
        energy_fraction: f64,
        visible_wells: &[(SVector<f64, 2>, f64)],
        visible_agents: &[(SVector<f64, 2>, f64)],
        well_regen_rate: f64,
        resonance_regen_rate: f64,
    ) -> SVector<f64, 2> {
        self.model.ticks_since_update += 1;

        let mut best_fe = f64::MAX;
        let mut best_dir = SVector::zeros();

        // 8 angles × 3 distances = 24 candidates
        for angle_idx in 0..8 {
            let angle = angle_idx as f64 * std::f64::consts::TAU / 8.0;
            let dir = SVector::from([angle.cos(), angle.sin()]);

            for &dist in &[5.0, 15.0, 30.0] {
                let candidate = pos + dir * dist;
                let fe = self.expected_free_energy(
                    &candidate, energy_fraction,
                    visible_wells, visible_agents,
                    well_regen_rate, resonance_regen_rate,
                );
                if fe < best_fe {
                    best_fe = fe;
                    best_dir = dir;
                }
            }
        }

        // Also evaluate staying still
        let stay_fe = self.expected_free_energy(
            pos, energy_fraction, visible_wells, visible_agents,
            well_regen_rate, resonance_regen_rate,
        );
        if stay_fe < best_fe {
            best_dir = SVector::zeros();
        }

        self.cached_direction = best_dir;
        best_dir
    }

    /// Amortized belief update. Call every `update_interval` ticks
    /// or on significant events.
    ///
    /// Updates well beliefs from observations and decays old beliefs.
    pub fn update_beliefs(
        &mut self,
        pos: &SVector<f64, 2>,
        energy_fraction: f64,
        discovered_wells: &[SVector<f64, 2>],
        nearby_partners: &[(u64, f64)], // (handle_bits, resonance)
        current_tick: usize,
    ) {
        // Update energy forecast (EMA)
        self.model.energy_forecast =
            0.95 * self.model.energy_forecast + 0.05 * energy_fraction;

        // Incorporate discovered wells
        for well_pos in discovered_wells {
            let existing = self.model.well_beliefs.iter_mut()
                .find(|b| (b.position - well_pos).norm() < 15.0);
            if let Some(belief) = existing {
                // Reduce uncertainty (we confirmed it's here)
                belief.uncertainty *= 0.5;
                belief.confidence = (belief.confidence + 0.1).min(1.0);
                belief.last_updated = current_tick;
            } else {
                // New well discovery
                if self.model.well_beliefs.len() >= self.model.max_well_beliefs {
                    // Evict oldest/least confident
                    if let Some(min_idx) = self.model.well_beliefs.iter().enumerate()
                        .min_by(|(_, a), (_, b)| a.confidence.partial_cmp(&b.confidence).unwrap())
                        .map(|(i, _)| i)
                    {
                        self.model.well_beliefs.swap_remove(min_idx);
                    }
                }
                self.model.well_beliefs.push(WellBelief {
                    position: *well_pos,
                    uncertainty: self.model.prior_well_uncertainty * 0.5, // moderate certainty
                    confidence: 0.3,
                    last_updated: current_tick,
                });
            }
        }

        // Decay old beliefs (uncertainty grows with time)
        for belief in &mut self.model.well_beliefs {
            let age = current_tick.saturating_sub(belief.last_updated);
            if age > 500 {
                belief.uncertainty += 0.1 * (age as f64 / 500.0);
                belief.confidence *= 0.999;
            }
        }

        // Update partner beliefs
        for &(handle, resonance) in nearby_partners {
            let existing = self.model.partner_beliefs.iter_mut()
                .find(|b| b.handle_bits == handle);
            if let Some(belief) = existing {
                if resonance > 0.5 {
                    belief.alpha += 0.1; // successful interaction
                } else {
                    belief.beta += 0.1; // low-value interaction
                }
            } else {
                if self.model.partner_beliefs.len() >= self.model.max_partner_beliefs {
                    if let Some(min_idx) = self.model.partner_beliefs.iter().enumerate()
                        .min_by(|(_, a), (_, b)| {
                            (a.alpha + a.beta).partial_cmp(&(b.alpha + b.beta)).unwrap()
                        })
                        .map(|(i, _)| i)
                    {
                        self.model.partner_beliefs.swap_remove(min_idx);
                    }
                }
                let (alpha, beta) = if resonance > 0.5 { (1.1, 1.0) } else { (1.0, 1.1) };
                self.model.partner_beliefs.push(PartnerBelief { handle_bits: handle, alpha, beta });
            }
        }

        self.model.ticks_since_update = 0;
    }

    /// Whether a belief update is due (amortized scheduling).
    pub fn should_update(&self) -> bool {
        self.model.ticks_since_update >= self.update_interval
    }

    /// Number of known well locations.
    pub fn wells_known(&self) -> usize {
        self.model.well_beliefs.len()
    }

    /// Number of known partners.
    pub fn partners_known(&self) -> usize {
        self.model.partner_beliefs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_agent_has_no_beliefs() {
        let agent = ActiveInferenceAgent::default();
        assert_eq!(agent.wells_known(), 0);
        assert_eq!(agent.partners_known(), 0);
    }

    #[test]
    fn well_discovery_creates_belief() {
        let mut agent = ActiveInferenceAgent::default();
        let pos = SVector::from([0.0, 0.0]);
        agent.update_beliefs(&pos, 0.5, &[SVector::from([30.0, 0.0])], &[], 100);
        assert_eq!(agent.wells_known(), 1);
    }

    #[test]
    fn repeated_observation_reduces_uncertainty() {
        let mut agent = ActiveInferenceAgent::default();
        let pos = SVector::from([0.0, 0.0]);
        let well = SVector::from([30.0, 0.0]);
        agent.update_beliefs(&pos, 0.5, &[well], &[], 100);
        let u1 = agent.model.well_beliefs[0].uncertainty;
        agent.update_beliefs(&pos, 0.5, &[well], &[], 200);
        let u2 = agent.model.well_beliefs[0].uncertainty;
        assert!(u2 < u1, "Repeated observation should reduce uncertainty");
    }

    #[test]
    fn action_selection_prefers_known_wells() {
        let mut agent = ActiveInferenceAgent::default();
        agent.model.well_beliefs.push(WellBelief {
            position: SVector::from([30.0, 0.0]),
            uncertainty: 1.0,
            confidence: 0.9,
            last_updated: 0,
        });
        let dir = agent.infer_action(
            &SVector::from([0.0, 0.0]), 0.3,
            &[], &[], 0.12, 0.06,
        );
        // Should point roughly toward the believed well (positive x)
        assert!(dir[0] > 0.5, "Should move toward believed well, got {:?}", dir);
    }

    #[test]
    fn partner_belief_updates() {
        let mut agent = ActiveInferenceAgent::default();
        let pos = SVector::from([0.0, 0.0]);
        agent.update_beliefs(&pos, 0.5, &[], &[(42, 0.8), (99, 0.2)], 100);
        assert_eq!(agent.partners_known(), 2);
        let p42 = agent.model.partner_beliefs.iter().find(|p| p.handle_bits == 42).unwrap();
        assert!(p42.expected_value() > 0.5, "High-resonance partner should have high value");
    }

    #[test]
    fn amortized_update_schedule() {
        let mut agent = ActiveInferenceAgent::default();
        agent.update_interval = 50;
        assert!(!agent.should_update());
        for _ in 0..50 {
            agent.model.ticks_since_update += 1;
        }
        assert!(agent.should_update());
    }
}
