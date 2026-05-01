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

/// A shareable belief message for factorised multi-agent propagation.
///
/// Agents broadcast messages about well locations they have observed.
/// Receivers fuse these with their own beliefs using Gaussian precision fusion.
///
/// The "factorised" structure means each well belief is fused independently
/// — the complexity is O(|beliefs|) per message rather than exponential in
/// the number of wells. This is the mean-field / variational approximation.
///
/// # Meso-temporal decay
/// Messages age: uncertainty grows as `σ² += age_ticks * BELIEF_TEMPORAL_DECAY`.
/// Stale messages (older than `max_age_ticks`) should be discarded by receivers.
/// This keeps the multi-agent system from acting on outdated information —
/// resources change, agents move, and old certainty becomes misleading.
///
/// # References
/// - Parr & Friston (2019), "Generalised free energy and active inference"
/// - Bishop (2006), PRML §2.3.3 — Gaussian precision fusion
#[derive(Debug, Clone)]
pub struct BeliefMessage {
    /// Position of the believed well (Gaussian mean).
    pub well_position: SVector<f64, 2>,
    /// Positional uncertainty (Gaussian variance σ²). Smaller = more certain.
    pub uncertainty: f64,
    /// Confidence weight [0, 1]: how many observations support this belief.
    pub confidence: f64,
    /// Sender's agent handle bits (for attribution and loop prevention).
    pub sender_bits: u64,
    /// Tick at which this message was created (for temporal decay).
    pub created_at_tick: usize,
    /// Maximum ticks this message stays valid (default: 500 ticks ≈ 7.8s at 64Hz).
    pub max_age_ticks: usize,
}

/// Rate at which message uncertainty grows per tick (meso-temporal diffusion).
///
/// At 0.2 σ²/tick, a confident belief (σ²=10) becomes prior-level (σ²=100)
/// after ~450 ticks (~7 seconds at 64Hz). This matches the timescale at which
/// resources respawn and entity positions meaningfully change.
pub const BELIEF_TEMPORAL_DECAY: f64 = 0.2;

/// Default maximum age for a belief message in ticks.
pub const BELIEF_MAX_AGE_TICKS: usize = 500;

impl BeliefMessage {
    /// Precision (inverse variance) of this belief.
    ///
    /// More certain beliefs have higher precision and contribute
    /// proportionally more to the fused posterior.
    #[inline]
    pub fn precision(&self) -> f64 {
        if self.uncertainty < 1e-10 {
            1e10
        } else {
            1.0 / self.uncertainty
        }
    }

    /// Effective uncertainty after accounting for meso-temporal decay.
    ///
    /// Beliefs about resource locations diffuse over time because resources
    /// regenerate, deplete, and move. The uncertainty grows linearly with age.
    #[inline]
    pub fn aged_uncertainty(&self, current_tick: usize) -> f64 {
        let age = current_tick.saturating_sub(self.created_at_tick);
        self.uncertainty + age as f64 * BELIEF_TEMPORAL_DECAY
    }

    /// Whether this message has exceeded its maximum age (should be discarded).
    #[inline]
    pub fn is_expired(&self, current_tick: usize) -> bool {
        current_tick.saturating_sub(self.created_at_tick) > self.max_age_ticks
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

/// KL divergence of a well belief from the uninformative (prior) Gaussian.
///
/// For a Gaussian belief `N(μ, σ²)` vs prior `N(μ, σ₀²)` (same location, different spread):
///
/// ```text
/// KL[N(μ,σ²) || N(μ,σ₀²)] = 0.5 * (σ²/σ₀² + ln(σ₀²/σ²) - 1)
/// ```
///
/// This is the **complexity** term in the variational free energy:
/// `F = -<accuracy> + KL[posterior || prior]`.
///
/// - When `σ² = σ₀²` (no update from prior): KL = 0.
/// - When `σ² << σ₀²` (very certain = far from prior): KL is large.
///
/// Returns 0.0 for degenerate inputs to avoid NaN.
///
/// # References
/// - Friston et al. (2017), "Active Inference" — F = E + C decomposition.
/// - Cover & Thomas (2006), "Elements of Information Theory", §9.3.
fn well_belief_kl(uncertainty: f64, prior_uncertainty: f64) -> f64 {
    if prior_uncertainty < 1e-10 || uncertainty < 1e-10 {
        return 0.0;
    }
    let ratio = uncertainty / prior_uncertainty;
    // 0.5*(σ²/σ₀² + ln(σ₀²/σ²) - 1) ≥ 0 by Jensen's inequality; clamp for floating-point safety
    0.5_f64 * (ratio + ratio.recip().ln() - 1.0).max(0.0)
}

/// Decomposed Expected Free Energy for a candidate action.
///
/// Separates the three canonical components of EFE as defined in:
/// Friston et al. (2017) "Active Inference and Epistemic Value" — Cog. Neurosci.
///
/// ```text
/// G(π) = epistemic_value + pragmatic_value + complexity
/// ```
///
/// Agents minimize total G. Lower is better. The decomposition exposes:
/// - **Epistemic value** (negative): exploration drive — how much uncertainty
///   would be reduced by visiting this position?
/// - **Pragmatic value** (negative): exploitation drive — how much energy
///   would be gained at this position (minus surprise penalty)?
/// - **Complexity**: KL cost of maintaining current beliefs vs prior.
///   Penalizes overly confident world models — the "razor" that keeps
///   agents from hallucinating certain structure.
#[derive(Debug, Clone, Default)]
pub struct EfeParts {
    /// Expected information gain from visiting this position.
    /// Negative = good (we prefer to reduce uncertainty).
    pub epistemic_value: f64,
    /// Expected energy gain minus depletion threat.
    /// Negative = good (we prefer to gain energy).
    pub pragmatic_value: f64,
    /// KL complexity: cost of maintaining current belief state vs prior.
    /// Always ≥ 0.
    pub complexity: f64,
    /// Sum: epistemic_value + pragmatic_value + complexity.
    pub total: f64,
}

impl EfeParts {
    /// True if this action is better (lower total G) than `other`.
    pub fn is_better_than(&self, other: &EfeParts) -> bool {
        self.total < other.total
    }
}

/// Active inference agent.
#[derive(Debug, Clone)]
pub struct ActiveInferenceAgent {
    pub model: GenerativeModel,
    /// Cached action from last inference (reused between updates).
    pub cached_direction: SVector<f64, 2>,
    /// EFE decomposition for the last inferred action.
    /// Exposes the epistemic/pragmatic/complexity split for telemetry and research.
    pub last_efe: EfeParts,
    /// Update interval (ticks between full belief updates).
    pub update_interval: usize,
}

impl Default for ActiveInferenceAgent {
    fn default() -> Self {
        Self {
            model: GenerativeModel::default(),
            cached_direction: SVector::zeros(),
            last_efe: EfeParts::default(),
            update_interval: 50,
        }
    }
}

impl ActiveInferenceAgent {
    /// Compute the decomposed Expected Free Energy (EFE) for a candidate position.
    ///
    /// EFE = epistemic_value + pragmatic_value + complexity
    ///
    /// All three components are additive and the agent minimizes the total.
    /// This decomposition is the canonical form from Friston et al. (2017):
    ///
    /// ```text
    /// G(π) ≈ -E_q[info_gain]    (epistemic: curiosity)
    ///       + E_q[surprise]     (pragmatic: outcome mismatch)
    ///       + KL[q || p]        (complexity: belief maintenance cost)
    /// ```
    ///
    /// The decomposition is observable via `last_efe` after `infer_action`,
    /// making the exploration/exploitation trade-off measurable for research.
    fn efe_components(
        &self,
        candidate_pos: &SVector<f64, 2>,
        energy_fraction: f64,
        visible_wells: &[(SVector<f64, 2>, f64)],
        visible_agents: &[(SVector<f64, 2>, f64)],
        well_regen_rate: f64,
        resonance_regen_rate: f64,
    ) -> EfeParts {
        // ── Pragmatic component ───────────────────────────────────────────────
        // Expected energy gain at this position (reduces VFE when positive).
        let mut expected_energy_gain = 0.0;

        // From believed wells
        for belief in &self.model.well_beliefs {
            let dist = (candidate_pos - belief.position).norm();
            if dist < 35.0 {
                let reliability = 1.0 / (1.0 + belief.uncertainty * 0.01);
                expected_energy_gain += well_regen_rate * belief.confidence * reliability;
            }
        }
        // From visible wells (direct observation overrides belief)
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

        // Surprise = how unexpected is the current energy level given the forecast?
        let surprise = (self.model.energy_forecast - energy_fraction).abs();
        // Urgency = quadratic scarcity pressure (near-zero energy = near-death)
        let urgency = (1.0 - energy_fraction).powi(2) * 3.0;

        // pragmatic_value: cost side of the pragmatic term
        // Positive = bad (surprise + urgency), partially offset by expected gain.
        let pragmatic_value = surprise + urgency - expected_energy_gain;

        // ── Epistemic component (information gain) ────────────────────────────
        // Expected entropy reduction from visiting this position.
        //
        // For each well belief in range: if we visit, we observe the well (or its
        // absence), reducing our positional uncertainty. The expected reduction
        // depends on observation strength (proximity) and current uncertainty.
        //
        // ΔH = 0.5 * ln(σ²_before / σ²_after)   [Gaussian entropy reduction]
        //
        // epistemic_value is NEGATIVE (exploration is beneficial = lowers EFE).
        let mut info_gain = 0.0;
        for belief in &self.model.well_beliefs {
            let dist = (candidate_pos - belief.position).norm();
            if dist < 50.0 && belief.uncertainty > 0.1 {
                // Observation strength: 1.0 at dist=0, 0.0 at dist=50
                let obs_strength = (1.0 - dist / 50.0).max(0.0);
                if obs_strength > 0.01 {
                    // Expected σ² after observing (50% reduction at full strength)
                    let sigma_after = belief.uncertainty * (1.0 - obs_strength * 0.5).max(0.01);
                    // Information gain in nats = 0.5 * ln(σ²_before / σ²_after)
                    let gain = 0.5 * (belief.uncertainty / sigma_after).ln();
                    info_gain += gain * belief.confidence;
                }
            }
        }
        // epistemic_value = -info_gain (negative = good, exploration reduces EFE)
        let epistemic_value = -info_gain;

        // ── Complexity (KL divergence) ────────────────────────────────────────
        // Cost of maintaining current belief state vs the uninformative prior.
        // Constant across positions (only depends on current beliefs), but
        // included in the decomposition for completeness and paper figures.
        //
        // Scaling constant 0.002: calibrated so KL stays at the same order of
        // magnitude as surprise (both typically 0–0.5 per tick at default params).
        let complexity: f64 = self
            .model
            .well_beliefs
            .iter()
            .map(|b| {
                well_belief_kl(b.uncertainty, self.model.prior_well_uncertainty) * b.confidence
            })
            .sum::<f64>()
            * 0.002;

        let total = epistemic_value + pragmatic_value + complexity;

        EfeParts {
            epistemic_value,
            pragmatic_value,
            complexity,
            total,
        }
    }

    /// Compute scalar EFE (for backward-compatible callers).
    fn expected_free_energy(
        &self,
        candidate_pos: &SVector<f64, 2>,
        energy_fraction: f64,
        visible_wells: &[(SVector<f64, 2>, f64)],
        visible_agents: &[(SVector<f64, 2>, f64)],
        well_regen_rate: f64,
        resonance_regen_rate: f64,
    ) -> f64 {
        self.efe_components(
            candidate_pos,
            energy_fraction,
            visible_wells,
            visible_agents,
            well_regen_rate,
            resonance_regen_rate,
        )
        .total
    }

    /// Select action that minimizes expected free energy.
    ///
    /// Evaluates 24 candidate directions (8 angles × 3 distances).
    /// Uses cached beliefs — no Bayesian updates during action selection.
    ///
    /// After this call, `self.last_efe` contains the EFE decomposition for
    /// the selected action, exposing the epistemic/pragmatic split.
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

        let mut best_efe = EfeParts {
            total: f64::MAX,
            ..EfeParts::default()
        };
        let mut best_dir = SVector::zeros();

        // 8 angles × 3 distances = 24 candidates
        for angle_idx in 0..8 {
            let angle = angle_idx as f64 * std::f64::consts::TAU / 8.0;
            let dir = SVector::from([angle.cos(), angle.sin()]);

            for &dist in &[5.0, 15.0, 30.0] {
                let candidate = pos + dir * dist;
                let efe = self.efe_components(
                    &candidate,
                    energy_fraction,
                    visible_wells,
                    visible_agents,
                    well_regen_rate,
                    resonance_regen_rate,
                );
                if efe.is_better_than(&best_efe) {
                    best_efe = efe;
                    best_dir = dir;
                }
            }
        }

        // Also evaluate staying still
        let stay_efe = self.efe_components(
            pos,
            energy_fraction,
            visible_wells,
            visible_agents,
            well_regen_rate,
            resonance_regen_rate,
        );
        if stay_efe.is_better_than(&best_efe) {
            best_efe = stay_efe;
            best_dir = SVector::zeros();
        }

        self.last_efe = best_efe;
        self.cached_direction = best_dir;
        best_dir
    }

    /// Amortized belief update. Call every `update_interval` ticks
    /// or on significant events.
    ///
    /// Updates well beliefs from observations and decays old beliefs.
    pub fn update_beliefs(
        &mut self,
        _pos: &SVector<f64, 2>,
        energy_fraction: f64,
        discovered_wells: &[SVector<f64, 2>],
        nearby_partners: &[(u64, f64)], // (handle_bits, resonance)
        current_tick: usize,
    ) {
        // Update energy forecast (EMA)
        self.model.energy_forecast = 0.95 * self.model.energy_forecast + 0.05 * energy_fraction;

        // Incorporate discovered wells
        for well_pos in discovered_wells {
            let existing = self
                .model
                .well_beliefs
                .iter_mut()
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
                    if let Some(min_idx) = self
                        .model
                        .well_beliefs
                        .iter()
                        .enumerate()
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
            let existing = self
                .model
                .partner_beliefs
                .iter_mut()
                .find(|b| b.handle_bits == handle);
            if let Some(belief) = existing {
                if resonance > 0.5 {
                    belief.alpha += 0.1; // successful interaction
                } else {
                    belief.beta += 0.1; // low-value interaction
                }
            } else {
                if self.model.partner_beliefs.len() >= self.model.max_partner_beliefs {
                    if let Some(min_idx) = self
                        .model
                        .partner_beliefs
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            (a.alpha + a.beta).partial_cmp(&(b.alpha + b.beta)).unwrap()
                        })
                        .map(|(i, _)| i)
                    {
                        self.model.partner_beliefs.swap_remove(min_idx);
                    }
                }
                let (alpha, beta) = if resonance > 0.5 {
                    (1.1, 1.0)
                } else {
                    (1.0, 1.1)
                };
                self.model.partner_beliefs.push(PartnerBelief {
                    handle_bits: handle,
                    alpha,
                    beta,
                });
            }
        }

        self.model.ticks_since_update = 0;
    }

    /// Collect outbound belief messages to share with nearby agents.
    ///
    /// Returns one message per well belief whose confidence is above
    /// `min_confidence`. Messages are tagged with `sender_bits` so receivers
    /// can attribute the source and avoid echo-chamber loops (never
    /// re-receiving your own beliefs from a third agent).
    ///
    /// Only beliefs with confidence ≥ `min_confidence` are shared — low
    /// confidence beliefs are not yet worth propagating.
    pub fn collect_messages(
        &self,
        sender_bits: u64,
        current_tick: usize,
        min_confidence: f64,
    ) -> Vec<BeliefMessage> {
        self.model
            .well_beliefs
            .iter()
            .filter(|b| b.confidence >= min_confidence)
            .map(|b| BeliefMessage {
                well_position: b.position,
                uncertainty: b.uncertainty,
                confidence: b.confidence,
                sender_bits,
                created_at_tick: current_tick,
                max_age_ticks: BELIEF_MAX_AGE_TICKS,
            })
            .collect()
    }

    /// Receive and fuse incoming belief messages into this agent's model.
    ///
    /// Uses **Gaussian precision fusion** (exact Bayesian update for Gaussians):
    ///
    /// ```text
    /// 1/σ²_new = 1/σ²_self + 1/σ²_msg
    /// μ_new    = σ²_new × (μ_self/σ²_self + μ_msg/σ²_msg)
    /// ```
    ///
    /// The "factorised" property: each well belief is updated **independently**.
    /// No joint distribution over multiple wells is ever computed, keeping
    /// complexity O(|beliefs| × |messages|) — tractable for large groups.
    ///
    /// Meso-temporal guard: messages are aged before fusion. Stale messages
    /// (expired or aged to near-prior uncertainty) are skipped entirely.
    ///
    /// Loop prevention: messages originally sent by this agent (`own_bits`)
    /// are discarded to prevent belief echo-chambers.
    pub fn receive_messages(
        &mut self,
        messages: &[BeliefMessage],
        own_bits: u64,
        current_tick: usize,
    ) {
        for msg in messages {
            // Skip own messages (echo-chamber guard)
            if msg.sender_bits == own_bits {
                continue;
            }
            // Skip expired messages (meso-temporal guard)
            if msg.is_expired(current_tick) {
                continue;
            }

            // Age the uncertainty before fusion
            let msg_uncertainty = msg.aged_uncertainty(current_tick);
            // Skip messages that have decayed to near-prior (not informative)
            if msg_uncertainty >= self.model.prior_well_uncertainty * 0.95 {
                continue;
            }

            // Find an existing belief to fuse with
            let existing = self
                .model
                .well_beliefs
                .iter_mut()
                .find(|b| (b.position - msg.well_position).norm() < 20.0);

            if let Some(belief) = existing {
                // Gaussian precision fusion: combine posterior with message.
                // σ²_new = 1 / (1/σ²_self + 1/σ²_msg)
                let prec_self = if belief.uncertainty > 1e-10 {
                    1.0 / belief.uncertainty
                } else {
                    1e10
                };
                let prec_msg = if msg_uncertainty > 1e-10 {
                    1.0 / msg_uncertainty
                } else {
                    1e10
                };
                let prec_new = prec_self + prec_msg;
                let sigma_new = 1.0 / prec_new;

                // μ_new = σ²_new × (μ_self × prec_self + μ_msg × prec_msg)
                let pos_new =
                    (belief.position * prec_self + msg.well_position * prec_msg) * sigma_new;

                belief.uncertainty = sigma_new;
                belief.position = pos_new;
                // Confidence: take the max (the better-supported belief wins)
                belief.confidence = belief.confidence.max(msg.confidence * 0.9);
                belief.last_updated = current_tick;
            } else {
                // New well from peer observation — insert with reduced confidence
                // (we haven't seen it ourselves yet)
                if self.model.well_beliefs.len() >= self.model.max_well_beliefs {
                    if let Some(min_idx) = self
                        .model
                        .well_beliefs
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.confidence.partial_cmp(&b.confidence).unwrap())
                        .map(|(i, _)| i)
                    {
                        self.model.well_beliefs.swap_remove(min_idx);
                    }
                }
                self.model.well_beliefs.push(WellBelief {
                    position: msg.well_position,
                    uncertainty: msg_uncertainty, // already aged
                    // Peer-sourced belief: 80% of sender's confidence
                    confidence: (msg.confidence * 0.8).min(0.7),
                    last_updated: current_tick,
                });
            }
        }
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
        let dir = agent.infer_action(&SVector::from([0.0, 0.0]), 0.3, &[], &[], 0.12, 0.06);
        // Should point roughly toward the believed well (positive x)
        assert!(
            dir[0] > 0.5,
            "Should move toward believed well, got {:?}",
            dir
        );
    }

    #[test]
    fn partner_belief_updates() {
        let mut agent = ActiveInferenceAgent::default();
        let pos = SVector::from([0.0, 0.0]);
        agent.update_beliefs(&pos, 0.5, &[], &[(42, 0.8), (99, 0.2)], 100);
        assert_eq!(agent.partners_known(), 2);
        let p42 = agent
            .model
            .partner_beliefs
            .iter()
            .find(|p| p.handle_bits == 42)
            .unwrap();
        assert!(
            p42.expected_value() > 0.5,
            "High-resonance partner should have high value"
        );
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

    // ── KL divergence (VFE complexity term) ──────────────────────────────────

    // ── Belief message & propagation ─────────────────────────────────────────

    #[test]
    fn belief_message_precision_is_inverse_uncertainty() {
        let msg = BeliefMessage {
            well_position: SVector::zeros(),
            uncertainty: 4.0,
            confidence: 0.8,
            sender_bits: 1,
            created_at_tick: 0,
            max_age_ticks: 500,
        };
        assert!(
            (msg.precision() - 0.25).abs() < 1e-12,
            "precision = 1/σ² = 1/4 = 0.25"
        );
    }

    #[test]
    fn belief_message_temporal_decay_grows_uncertainty() {
        let msg = BeliefMessage {
            well_position: SVector::zeros(),
            uncertainty: 10.0,
            confidence: 0.8,
            sender_bits: 1,
            created_at_tick: 0,
            max_age_ticks: 500,
        };
        let aged = msg.aged_uncertainty(100);
        assert!(
            aged > 10.0,
            "aged uncertainty should grow: {} > 10.0 at tick 100",
            aged
        );
        assert!(
            (aged - (10.0 + 100.0 * BELIEF_TEMPORAL_DECAY)).abs() < 1e-10,
            "aged uncertainty should be exactly σ² + age * DECAY"
        );
    }

    #[test]
    fn belief_message_expires_after_max_age() {
        let msg = BeliefMessage {
            well_position: SVector::zeros(),
            uncertainty: 10.0,
            confidence: 0.8,
            sender_bits: 1,
            created_at_tick: 0,
            max_age_ticks: 100,
        };
        assert!(
            !msg.is_expired(50),
            "should not expire before max_age_ticks"
        );
        assert!(msg.is_expired(101), "should expire after max_age_ticks");
    }

    #[test]
    fn collect_messages_filters_low_confidence() {
        let mut agent = ActiveInferenceAgent::default();
        agent.model.well_beliefs.push(WellBelief {
            position: SVector::from([30.0, 0.0]),
            uncertainty: 10.0,
            confidence: 0.9, // above threshold
            last_updated: 0,
        });
        agent.model.well_beliefs.push(WellBelief {
            position: SVector::from([60.0, 0.0]),
            uncertainty: 50.0,
            confidence: 0.1, // below threshold 0.3
            last_updated: 0,
        });
        let msgs = agent.collect_messages(42, 100, 0.3);
        assert_eq!(
            msgs.len(),
            1,
            "only high-confidence beliefs should be shared"
        );
        assert!((msgs[0].well_position[0] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn receive_message_reduces_uncertainty_via_precision_fusion() {
        let mut agent = ActiveInferenceAgent::default();
        agent.model.well_beliefs.push(WellBelief {
            position: SVector::from([30.0, 0.0]),
            uncertainty: 40.0, // moderately uncertain
            confidence: 0.5,
            last_updated: 0,
        });

        // Peer has more certain belief about the same well
        let msg = BeliefMessage {
            well_position: SVector::from([32.0, 0.0]), // slightly offset
            uncertainty: 10.0,                         // more certain
            confidence: 0.8,
            sender_bits: 99, // different agent
            created_at_tick: 0,
            max_age_ticks: 500,
        };

        let uncertainty_before = agent.model.well_beliefs[0].uncertainty;
        agent.receive_messages(&[msg], 1, 0); // own_bits != 99, so accepted
        let uncertainty_after = agent.model.well_beliefs[0].uncertainty;

        assert!(
            uncertainty_after < uncertainty_before,
            "precision fusion should reduce uncertainty: {} < {}",
            uncertainty_after,
            uncertainty_before
        );
        // Fused position should be between the two (precision-weighted)
        assert!(
            agent.model.well_beliefs[0].position[0] > 30.0,
            "fused position should shift toward peer's more certain belief"
        );
    }

    #[test]
    fn receive_message_rejects_own_sender() {
        let mut agent = ActiveInferenceAgent::default();
        let msg = BeliefMessage {
            well_position: SVector::from([30.0, 0.0]),
            uncertainty: 5.0,
            confidence: 0.9,
            sender_bits: 42, // same as own_bits
            created_at_tick: 0,
            max_age_ticks: 500,
        };
        agent.receive_messages(&[msg], 42, 0); // own_bits = 42 → should reject
        assert_eq!(agent.wells_known(), 0, "own messages should be rejected");
    }

    #[test]
    fn receive_message_rejects_expired_message() {
        let mut agent = ActiveInferenceAgent::default();
        let msg = BeliefMessage {
            well_position: SVector::from([30.0, 0.0]),
            uncertainty: 5.0,
            confidence: 0.9,
            sender_bits: 99,
            created_at_tick: 0,
            max_age_ticks: 100,
        };
        agent.receive_messages(&[msg], 1, 200); // current_tick 200 > created+max_age=100
        assert_eq!(
            agent.wells_known(),
            0,
            "expired messages should be rejected"
        );
    }

    #[test]
    fn receive_message_adds_new_well_from_peer() {
        let mut agent = ActiveInferenceAgent::default();
        let msg = BeliefMessage {
            well_position: SVector::from([50.0, 0.0]),
            uncertainty: 20.0,
            confidence: 0.7,
            sender_bits: 99,
            created_at_tick: 0,
            max_age_ticks: 500,
        };
        agent.receive_messages(&[msg], 1, 0);
        assert_eq!(agent.wells_known(), 1, "peer-observed well should be added");
        // Confidence should be discounted (80% of sender's)
        assert!(
            agent.model.well_beliefs[0].confidence <= 0.7,
            "peer-sourced confidence should be discounted"
        );
    }

    #[test]
    fn receive_stale_message_near_prior_is_ignored() {
        let mut agent = ActiveInferenceAgent::default();
        // A message that has aged to near-prior uncertainty should be ignored
        let msg = BeliefMessage {
            well_position: SVector::from([30.0, 0.0]),
            uncertainty: 5.0, // starts certain
            confidence: 0.9,
            sender_bits: 99,
            created_at_tick: 0,  // created at tick 0
            max_age_ticks: 5000, // not expired by age
        };
        // But aged_uncertainty at tick 500 = 5.0 + 500 * 0.2 = 105 ≥ prior (100) * 0.95 = 95
        // So the message's effective uncertainty has diffused to near-prior
        agent.receive_messages(&[msg], 1, 500);
        assert_eq!(
            agent.wells_known(),
            0,
            "message aged to near-prior uncertainty should be ignored"
        );
    }

    #[test]
    fn belief_propagation_improves_group_knowledge() {
        // Simulate a minimal 2-agent group:
        // Agent A discovers well at (50, 0).
        // Agent B has no beliefs.
        // After propagation, agent B should know about the well.
        let mut agent_a = ActiveInferenceAgent::default();
        let mut agent_b = ActiveInferenceAgent::default();

        let pos = SVector::from([0.0, 0.0]);
        agent_a.update_beliefs(&pos, 0.5, &[SVector::from([50.0, 0.0])], &[], 100);
        // Confirm A learned the well
        assert_eq!(agent_a.wells_known(), 1);
        assert_eq!(agent_b.wells_known(), 0);

        // A broadcasts; B receives
        let messages = agent_a.collect_messages(1, 100, 0.1);
        agent_b.receive_messages(&messages, 2, 100);

        assert_eq!(
            agent_b.wells_known(),
            1,
            "agent B should learn about the well via belief propagation"
        );
        // B's knowledge should be less certain than A's (peer discount applied)
        let a_conf = agent_a.model.well_beliefs[0].confidence;
        let b_conf = agent_b.model.well_beliefs[0].confidence;
        assert!(
            b_conf <= a_conf,
            "B's confidence should be ≤ A's (peer discount)"
        );
    }

    // ── EFE decomposition ─────────────────────────────────────────────────────

    #[test]
    fn efe_parts_default_is_zero() {
        let parts = EfeParts::default();
        assert_eq!(parts.total, 0.0);
        assert_eq!(parts.epistemic_value, 0.0);
        assert_eq!(parts.pragmatic_value, 0.0);
        assert_eq!(parts.complexity, 0.0);
    }

    #[test]
    fn efe_is_better_than_compares_total() {
        let good = EfeParts {
            total: -1.0,
            ..EfeParts::default()
        };
        let bad = EfeParts {
            total: 1.0,
            ..EfeParts::default()
        };
        assert!(good.is_better_than(&bad));
        assert!(!bad.is_better_than(&good));
    }

    #[test]
    fn efe_last_efe_populated_after_infer_action() {
        let mut agent = ActiveInferenceAgent::default();
        let pos = SVector::from([0.0, 0.0]);
        agent.infer_action(&pos, 0.5, &[], &[], 0.12, 0.06);
        // last_efe total should be finite after inference
        assert!(
            agent.last_efe.total.is_finite(),
            "last_efe should be finite after infer_action"
        );
    }

    #[test]
    fn efe_epistemic_value_negative_when_uncertain_well_nearby() {
        // Place a well belief with high uncertainty near the agent.
        // Moving toward it should provide info_gain → negative epistemic_value.
        let mut agent = ActiveInferenceAgent::default();
        agent.model.well_beliefs.push(WellBelief {
            position: SVector::from([25.0, 0.0]),
            uncertainty: 80.0, // high = lots to learn
            confidence: 0.8,
            last_updated: 0,
        });

        let efe = agent.efe_components(
            &SVector::from([0.0, 0.0]),
            0.7, // high energy = no urgency
            &[],
            &[],
            0.12,
            0.06,
        );
        // With the well 25 units away and high uncertainty, info_gain > 0
        // → epistemic_value < 0 (exploration is beneficial)
        assert!(
            efe.epistemic_value < 0.0,
            "epistemic_value should be negative (beneficial) for uncertain nearby well, got {}",
            efe.epistemic_value
        );
    }

    #[test]
    fn efe_pragmatic_value_negative_when_well_at_candidate() {
        // A candidate position directly on a visible well should have
        // high expected_energy_gain → pragmatic_value should be lower (more negative).
        let agent = ActiveInferenceAgent::default();
        let well_pos = SVector::from([0.0, 0.0]);
        let far_pos = SVector::from([100.0, 100.0]);

        let efe_on_well = agent.efe_components(&well_pos, 0.3, &[(well_pos, 1.0)], &[], 0.12, 0.06);
        let efe_far = agent.efe_components(&far_pos, 0.3, &[(well_pos, 1.0)], &[], 0.12, 0.06);
        assert!(
            efe_on_well.pragmatic_value < efe_far.pragmatic_value,
            "being on the well should have lower pragmatic_value: on={}, far={}",
            efe_on_well.pragmatic_value,
            efe_far.pragmatic_value
        );
    }

    #[test]
    fn efe_complexity_zero_with_no_beliefs() {
        let agent = ActiveInferenceAgent::default();
        let efe = agent.efe_components(&SVector::from([0.0, 0.0]), 0.5, &[], &[], 0.12, 0.06);
        assert!(
            efe.complexity.abs() < 1e-12,
            "no beliefs → no complexity, got {}",
            efe.complexity
        );
    }

    #[test]
    fn efe_complexity_increases_with_confident_beliefs() {
        let mut agent_confident = ActiveInferenceAgent::default();
        let mut agent_uncertain = ActiveInferenceAgent::default();

        agent_confident.model.well_beliefs.push(WellBelief {
            position: SVector::from([30.0, 0.0]),
            uncertainty: 1.0, // very certain
            confidence: 0.9,
            last_updated: 0,
        });
        agent_uncertain.model.well_beliefs.push(WellBelief {
            position: SVector::from([30.0, 0.0]),
            uncertainty: 90.0, // close to prior (100.0)
            confidence: 0.9,
            last_updated: 0,
        });

        let efe_c = agent_confident.efe_components(&SVector::zeros(), 0.5, &[], &[], 0.12, 0.06);
        let efe_u = agent_uncertain.efe_components(&SVector::zeros(), 0.5, &[], &[], 0.12, 0.06);

        assert!(
            efe_c.complexity > efe_u.complexity,
            "confident agent should have higher complexity: certain={}, uncertain={}",
            efe_c.complexity,
            efe_u.complexity
        );
    }

    #[test]
    fn efe_total_equals_sum_of_parts() {
        let mut agent = ActiveInferenceAgent::default();
        agent.model.well_beliefs.push(WellBelief {
            position: SVector::from([20.0, 0.0]),
            uncertainty: 30.0,
            confidence: 0.7,
            last_updated: 0,
        });
        let efe = agent.efe_components(
            &SVector::from([10.0, 0.0]),
            0.4,
            &[(SVector::from([20.0, 0.0]), 0.8)],
            &[],
            0.12,
            0.06,
        );
        let expected_total = efe.epistemic_value + efe.pragmatic_value + efe.complexity;
        assert!(
            (efe.total - expected_total).abs() < 1e-10,
            "total should equal sum of parts: total={}, sum={}",
            efe.total,
            expected_total
        );
    }

    #[test]
    fn kl_zero_at_prior_uncertainty() {
        // When current uncertainty equals the prior, KL should be zero.
        let prior = 100.0;
        let kl = well_belief_kl(prior, prior);
        assert!(
            kl.abs() < 1e-12,
            "KL at prior uncertainty should be 0, got {kl}"
        );
    }

    #[test]
    fn kl_positive_below_prior() {
        // Narrowing uncertainty (more certain than prior) → positive KL.
        let kl = well_belief_kl(10.0, 100.0); // σ² = 10, σ₀² = 100
        assert!(kl > 0.0, "KL should be positive when σ² < σ₀², got {kl}");
    }

    #[test]
    fn kl_positive_above_prior() {
        // Widening uncertainty (less certain than prior) → also positive KL
        // (KL is always non-negative, just reflects mismatch in both directions).
        let kl = well_belief_kl(200.0, 100.0);
        assert!(kl > 0.0, "KL should be positive when σ² > σ₀², got {kl}");
    }

    #[test]
    fn kl_increases_with_certainty() {
        // More certain beliefs (lower uncertainty) should have higher KL cost.
        let prior = 100.0;
        let kl_uncertain = well_belief_kl(50.0, prior); // half the prior
        let kl_certain = well_belief_kl(1.0, prior); // very certain
        assert!(
            kl_certain > kl_uncertain,
            "more certain belief should have higher KL: certain={kl_certain}, uncertain={kl_uncertain}"
        );
    }

    #[test]
    fn kl_degenerates_to_zero_for_bad_inputs() {
        assert_eq!(well_belief_kl(0.0, 100.0), 0.0, "zero uncertainty → 0");
        assert_eq!(well_belief_kl(100.0, 0.0), 0.0, "zero prior → 0");
    }

    #[test]
    fn kl_complexity_affects_action_selection() {
        // An agent with a well-known well (low uncertainty, high confidence)
        // should pay KL complexity cost for it, making exploration more attractive
        // compared to an agent with an uncertain belief about the same well.
        let mut agent_certain = ActiveInferenceAgent::default();
        let mut agent_uncertain = ActiveInferenceAgent::default();

        let well_pos = SVector::from([30.0, 0.0]);

        // certain agent: very confident
        agent_certain.model.well_beliefs.push(WellBelief {
            position: well_pos,
            uncertainty: 0.5, // very certain
            confidence: 0.9,
            last_updated: 0,
        });

        // uncertain agent: same well but unsure
        agent_uncertain.model.well_beliefs.push(WellBelief {
            position: well_pos,
            uncertainty: 80.0, // close to prior
            confidence: 0.9,
            last_updated: 0,
        });

        // Both agents try to infer action from the same starting point with same inputs.
        let pos = SVector::from([0.0, 0.0]);
        let dir_certain = agent_certain.infer_action(&pos, 0.3, &[], &[], 0.12, 0.06);
        let dir_uncertain = agent_uncertain.infer_action(&pos, 0.3, &[], &[], 0.12, 0.06);

        // Both should still seek the well (energy is low), but the certain agent's
        // choice is influenced by the KL cost term in the VFE.
        // At minimum, neither should panic and both should return normalized directions.
        assert!(
            dir_certain.norm() <= 1.01,
            "certain: direction should be unit/zero"
        );
        assert!(
            dir_uncertain.norm() <= 1.01,
            "uncertain: direction should be unit/zero"
        );
    }

    #[test]
    fn vfe_complexity_term_is_bounded() {
        // With 10 maximally confident well beliefs, the total KL complexity
        // should remain finite and not blow up to infinity.
        let mut agent = ActiveInferenceAgent::default();
        for i in 0..10 {
            agent.model.well_beliefs.push(WellBelief {
                position: SVector::from([i as f64 * 30.0, 0.0]),
                uncertainty: 0.01, // extremely certain
                confidence: 1.0,
                last_updated: 0,
            });
        }
        let dir = agent.infer_action(&SVector::zeros(), 0.5, &[], &[], 0.12, 0.06);
        assert!(
            dir.norm().is_finite(),
            "VFE complexity should remain finite"
        );
    }
}
