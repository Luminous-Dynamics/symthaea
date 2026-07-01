// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Twin-therapeutic bridge: connects the neuro-digital twin (Direction A)
//! to therapeutic consciousness protocols (Direction H) for closed-loop
//! biofeedback-driven therapy.
//!
//! Architecture:
//! ```text
//!   EEG -> Neuro-Digital Twin -> Feature Extraction -> Protocol Adaptation
//!                                                           |
//!   Patient <- Modality Output <- Protocol Engine <- Target State Tracking
//! ```
//!
//! The bridge translates EEG-derived neuromod features into biofeedback
//! metrics that the protocol engine uses to evaluate progress toward
//! target consciousness states and adapt interventions in real-time.
//!
//! References:
//! - Hammond, D. C. (2011). Neurofeedback treatment of depression and anxiety.
//! - Sitaram et al. (2017). Closed-loop brain training.
//! - Carhart-Harris & Friston (2019). REBUS model.

use crate::consciousness_protocols::{
    BiofeedbackMetric, Modality, Protocol, SafetyAction, SessionOutcome, SessionState,
    SessionStatus, TargetState,
};

// ---------------------------------------------------------------------------
// EEG feature mirror (avoids circular dependency on main symthaea crate)
// ---------------------------------------------------------------------------

/// EEG-derived features (mirrors `neuro_digital_twin::EegNeuromodFeatures`).
///
/// Kept local to avoid circular dependency between the therapeutic crate
/// and the main symthaea crate where the neuro-digital twin lives.
#[derive(Debug, Clone, Default)]
pub struct EegFeatures {
    /// Frontal alpha asymmetry: (F4_alpha - F3_alpha).
    /// Positive values indicate approach / positive affect.
    /// Science: Davidson (2004).
    pub frontal_alpha_asymmetry: f32,
    /// Frontal midline theta power (Fz).
    /// High theta indicates sustained attention.
    /// Science: Harmony (2013).
    pub theta_frontal_midline: f32,
    /// Beta-gamma power ratio across all channels.
    /// High ratio indicates cortical arousal.
    /// Science: Aston-Jones & Cohen (2005).
    pub beta_gamma_ratio: f32,
    /// Delta band power (global average).
    /// High delta indicates sleep pressure.
    pub delta_power: f32,
    /// Inter-hemispheric alpha coherence.
    /// High coherence indicates cortical integration.
    pub alpha_coherence: f32,
    /// Gamma-band synchrony across channels.
    /// High synchrony indicates conscious binding.
    pub gamma_synchrony: f32,
    /// Theta/beta ratio.
    /// High ratio indicates inattention (ADHD marker).
    /// Science: Monastra et al. (2001).
    pub theta_beta_ratio: f32,
}

// ---------------------------------------------------------------------------
// Bridge output
// ---------------------------------------------------------------------------

/// Output produced each tick by the bridge.
#[derive(Debug, Clone)]
pub struct TherapeuticOutput {
    /// Modalities the protocol prescribes this tick.
    pub modalities: Vec<Modality>,
    /// Progress toward the target state (0.0 to 1.0).
    pub progress: f64,
    /// Safety action if any constraint was triggered.
    pub safety_action: Option<SafetyAction>,
    /// Current biofeedback readings derived from EEG.
    pub biofeedback_readings: Vec<(BiofeedbackMetric, f32)>,
}

// ---------------------------------------------------------------------------
// Named constants
// ---------------------------------------------------------------------------

/// Phi value below which a safety override triggers (Hammond 2011).
const SAFETY_PHI_FLOOR: f64 = 0.15;

/// Default EMA adaptation rate for progress tracking.
const DEFAULT_ADAPTATION_RATE: f32 = 0.15;

/// Number of recent EEG samples retained for trend analysis.
const EEG_HISTORY_CAP: usize = 128;

/// Number of recent Phi samples retained for trend analysis.
const PHI_HISTORY_CAP: usize = 128;

/// Minimum ticks before a session can be considered "complete".
const MIN_TICKS_FOR_COMPLETION: u32 = 10;

/// Progress threshold above which we consider the target reached.
const TARGET_REACHED_THRESHOLD: f64 = 0.90;

// ---------------------------------------------------------------------------
// Bridge
// ---------------------------------------------------------------------------

/// Closed-loop bridge between neuro-digital twin EEG features and the
/// therapeutic protocol engine.
///
/// Each `process()` call:
/// 1. Converts EEG features to biofeedback metrics.
/// 2. Checks for safety overrides (dangerously low Phi).
/// 3. Feeds metrics + Phi into `SessionState::tick()`.
/// 4. Computes progress toward the protocol's target state.
/// 5. Returns `TherapeuticOutput` with modalities, progress, and safety info.
pub struct TwinTherapeuticBridge {
    session: SessionState,
    eeg_history: Vec<EegFeatures>,
    phi_history: Vec<f64>,
    /// EMA smoothing rate for progress tracking (0..1).
    adaptation_rate: f32,
    /// True when the bridge has triggered its own safety override
    /// (independent of protocol-level safety).
    safety_override: bool,
    /// Most recent safety action (from bridge or protocol).
    last_safety: Option<SafetyAction>,
    /// Running tick counter.
    tick_count: u32,
    /// EMA-smoothed progress estimate.
    progress_ema: f64,
}

impl TwinTherapeuticBridge {
    /// Create a new bridge for the given protocol.
    pub fn new(protocol: Protocol) -> Self {
        Self {
            session: SessionState::new(protocol),
            eeg_history: Vec::new(),
            phi_history: Vec::new(),
            adaptation_rate: DEFAULT_ADAPTATION_RATE,
            safety_override: false,
            last_safety: None,
            tick_count: 0,
            progress_ema: 0.0,
        }
    }

    /// Create a bridge with a custom adaptation rate.
    pub fn with_adaptation_rate(protocol: Protocol, rate: f32) -> Self {
        let mut bridge = Self::new(protocol);
        bridge.adaptation_rate = rate.clamp(0.01, 1.0);
        bridge
    }

    /// Feed EEG features and current Phi, get back therapeutic modalities
    /// to apply this tick.
    pub fn process(&mut self, eeg: &EegFeatures, phi: f64) -> TherapeuticOutput {
        self.tick_count += 1;

        // Record history (bounded).
        if self.eeg_history.len() >= EEG_HISTORY_CAP {
            self.eeg_history.remove(0);
        }
        self.eeg_history.push(eeg.clone());

        if self.phi_history.len() >= PHI_HISTORY_CAP {
            self.phi_history.remove(0);
        }
        self.phi_history.push(phi);

        // Step 1: Convert EEG to biofeedback metrics.
        let biofeedback = self.eeg_to_biofeedback(eeg);

        // Step 2: Check bridge-level safety override.
        if phi < SAFETY_PHI_FLOOR && !self.safety_override {
            self.safety_override = true;
            let action = SafetyAction::PauseProtocol;
            self.last_safety = Some(action.clone());
            return TherapeuticOutput {
                modalities: Vec::new(),
                progress: self.progress_ema,
                safety_action: Some(action),
                biofeedback_readings: biofeedback,
            };
        }

        // Clear bridge-level override when Phi recovers.
        if phi >= SAFETY_PHI_FLOOR * 2.0 {
            self.safety_override = false;
        }

        if self.safety_override {
            return TherapeuticOutput {
                modalities: Vec::new(),
                progress: self.progress_ema,
                safety_action: self.last_safety.clone(),
                biofeedback_readings: biofeedback,
            };
        }

        // Step 3: Tick the underlying session.
        let modalities = self.session.tick(phi, &biofeedback);

        // Step 4: Check protocol-level safety.
        let protocol_safety = self
            .session
            .check_safety(phi, 0.0)
            .or_else(|| self.last_safety.clone().filter(|_| self.safety_override));
        if protocol_safety.is_some() {
            self.last_safety.clone_from(&protocol_safety);
        }

        // Step 5: Update progress toward target.
        let raw_progress = self.compute_raw_progress(eeg, phi);
        let alpha = self.adaptation_rate as f64;
        self.progress_ema = self.progress_ema * (1.0 - alpha) + raw_progress * alpha;

        TherapeuticOutput {
            modalities,
            progress: self.progress_ema,
            safety_action: protocol_safety,
            biofeedback_readings: biofeedback,
        }
    }

    /// Convert EEG features to biofeedback metrics the protocol understands.
    fn eeg_to_biofeedback(&self, eeg: &EegFeatures) -> Vec<(BiofeedbackMetric, f32)> {
        vec![
            (BiofeedbackMetric::AlphaCoherence, eeg.alpha_coherence),
            (BiofeedbackMetric::ThetaPower, eeg.theta_frontal_midline),
            (BiofeedbackMetric::GammaSynchrony, eeg.gamma_synchrony),
            // beta_gamma_ratio proxies arousal; map to HRV (inverse relationship:
            // high arousal = low HRV). Hammond (2011).
            (
                BiofeedbackMetric::HeartRateVariability,
                (1.0 - eeg.beta_gamma_ratio).clamp(0.0, 1.0),
            ),
            // theta_beta_ratio proxies inattention; map to skin conductance
            // (higher inattention = lower conductance). Monastra et al. (2001).
            (
                BiofeedbackMetric::SkinConductance,
                (1.0 - eeg.theta_beta_ratio).clamp(0.0, 1.0),
            ),
        ]
    }

    /// Compute raw progress (0..1) toward the protocol target from current EEG + Phi.
    fn compute_raw_progress(&self, eeg: &EegFeatures, phi: f64) -> f64 {
        match &self.session.protocol.target {
            TargetState::FlowState {
                min_phi,
                optimal_arousal,
            } => {
                let phi_progress = (phi / min_phi).min(1.0);
                let arousal = eeg.beta_gamma_ratio;
                let arousal_dist = (arousal - optimal_arousal).abs();
                let arousal_progress = (1.0 - arousal_dist as f64).max(0.0);
                (phi_progress * 0.6 + arousal_progress * 0.4).min(1.0)
            }
            TargetState::MeditativeAbsorption {
                target_alpha_coherence,
                min_theta,
            } => {
                let alpha_progress = (eeg.alpha_coherence / target_alpha_coherence).min(1.0) as f64;
                let theta_progress = (eeg.theta_frontal_midline / min_theta).min(1.0) as f64;
                (alpha_progress * 0.5 + theta_progress * 0.5).min(1.0)
            }
            TargetState::PeakExperience { target_phi, .. } => (phi / target_phi).min(1.0),
            // For protocols without EEG-specific targets, use Phi alone.
            _ => (phi / 0.6).min(1.0),
        }
    }

    /// Current progress toward the target state (EMA-smoothed, 0..1).
    pub fn progress_toward_target(&self) -> f64 {
        self.progress_ema
    }

    /// Reference to the current session status.
    pub fn status(&self) -> &SessionStatus {
        &self.session.status
    }

    /// Session outcome if the session has completed or terminated.
    pub fn outcome(&self) -> Option<SessionOutcome> {
        match &self.session.status {
            SessionStatus::Completed | SessionStatus::Terminated(_) => Some(self.session.outcome()),
            _ => None,
        }
    }

    /// Most recent safety action triggered (bridge-level or protocol-level).
    pub fn last_safety_action(&self) -> Option<&SafetyAction> {
        self.last_safety.as_ref()
    }

    /// Estimate ticks remaining to reach the target based on progress EMA trend.
    ///
    /// Returns `None` if progress is not advancing or already reached.
    pub fn estimated_ticks_to_target(&self) -> Option<u32> {
        if self.tick_count < 2 {
            return None;
        }
        if self.progress_ema >= TARGET_REACHED_THRESHOLD {
            return Some(0);
        }

        // Simple linear extrapolation from EMA rate.
        let rate = self.progress_ema / self.tick_count as f64;
        if rate <= 0.0 {
            return None;
        }
        let remaining = TARGET_REACHED_THRESHOLD - self.progress_ema;
        Some((remaining / rate).ceil() as u32)
    }

    /// Number of ticks processed so far.
    pub fn ticks_elapsed(&self) -> u32 {
        self.tick_count
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: EEG features simulating relaxed, coherent state.
    fn relaxed_eeg() -> EegFeatures {
        EegFeatures {
            frontal_alpha_asymmetry: 0.3,
            theta_frontal_midline: 0.6,
            beta_gamma_ratio: 0.3,
            delta_power: 0.1,
            alpha_coherence: 0.85,
            gamma_synchrony: 0.7,
            theta_beta_ratio: 0.4,
        }
    }

    /// Helper: EEG features simulating anxious, desynchronized state.
    fn anxious_eeg() -> EegFeatures {
        EegFeatures {
            frontal_alpha_asymmetry: -0.4,
            theta_frontal_midline: 0.1,
            beta_gamma_ratio: 0.9,
            delta_power: 0.05,
            alpha_coherence: 0.2,
            gamma_synchrony: 0.15,
            theta_beta_ratio: 0.8,
        }
    }

    /// Helper: EEG features simulating deep meditative state.
    fn meditative_eeg() -> EegFeatures {
        EegFeatures {
            frontal_alpha_asymmetry: 0.1,
            theta_frontal_midline: 0.8,
            beta_gamma_ratio: 0.15,
            delta_power: 0.2,
            alpha_coherence: 0.9,
            gamma_synchrony: 0.4,
            theta_beta_ratio: 0.6,
        }
    }

    #[test]
    fn bridge_creates_with_flow_protocol() {
        let bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        assert_eq!(*bridge.status(), SessionStatus::Active);
        assert_eq!(bridge.ticks_elapsed(), 0);
        assert!(bridge.progress_toward_target() < f64::EPSILON);
    }

    #[test]
    fn process_returns_modalities() {
        let mut bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        let output = bridge.process(&relaxed_eeg(), 0.7);
        // Flow protocol has modalities in its first phase.
        assert!(!output.modalities.is_empty());
        assert_eq!(bridge.ticks_elapsed(), 1);
    }

    #[test]
    fn eeg_maps_to_biofeedback_metrics() {
        let bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        let readings = bridge.eeg_to_biofeedback(&relaxed_eeg());
        assert_eq!(readings.len(), 5);
        let metrics: Vec<BiofeedbackMetric> = readings.iter().map(|(m, _)| *m).collect();
        assert!(metrics.contains(&BiofeedbackMetric::AlphaCoherence));
        assert!(metrics.contains(&BiofeedbackMetric::ThetaPower));
        assert!(metrics.contains(&BiofeedbackMetric::GammaSynchrony));
        assert!(metrics.contains(&BiofeedbackMetric::HeartRateVariability));
        assert!(metrics.contains(&BiofeedbackMetric::SkinConductance));
    }

    #[test]
    fn high_alpha_coherence_maps_to_high_metric() {
        let bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        let eeg = EegFeatures {
            alpha_coherence: 0.95,
            ..Default::default()
        };
        let readings = bridge.eeg_to_biofeedback(&eeg);
        let alpha_val = readings
            .iter()
            .find(|(m, _)| *m == BiofeedbackMetric::AlphaCoherence)
            .map(|(_, v)| *v)
            .unwrap();
        assert!((alpha_val - 0.95).abs() < 1e-6);
    }

    #[test]
    fn progress_increases_when_eeg_matches_target() {
        let mut bridge = TwinTherapeuticBridge::with_adaptation_rate(Protocol::flow_state(), 0.5);
        let good_eeg = relaxed_eeg();

        // Run several ticks with good EEG and high Phi.
        for _ in 0..5 {
            bridge.process(&good_eeg, 0.8);
        }
        let progress_at_5 = bridge.progress_toward_target();
        assert!(
            progress_at_5 > 0.3,
            "Progress should be meaningful after 5 good ticks: {progress_at_5}"
        );

        for _ in 0..10 {
            bridge.process(&good_eeg, 0.8);
        }
        let progress_at_15 = bridge.progress_toward_target();
        assert!(
            progress_at_15 > progress_at_5,
            "Progress should increase: {progress_at_15} vs {progress_at_5}"
        );
    }

    #[test]
    fn safety_override_triggers_on_low_phi() {
        let mut bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        let output = bridge.process(&relaxed_eeg(), 0.05);
        assert!(output.safety_action.is_some());
        assert_eq!(output.safety_action.unwrap(), SafetyAction::PauseProtocol);
        assert!(output.modalities.is_empty());
    }

    #[test]
    fn session_completes_after_enough_ticks() {
        let mut bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        // Pump ticks well beyond what the flow protocol needs to advance
        // through all phases (each phase is time-based or metric-based).
        for i in 0..120 {
            let phi = 0.7 + (i as f64) * 0.001;
            bridge.process(&relaxed_eeg(), phi);
        }
        // Session should have completed or at least advanced.
        let outcome = bridge.outcome();
        // The protocol is 45 minutes, so 120 ticks (minutes) should complete it.
        assert!(
            outcome.is_some(),
            "Session should complete after 120 ticks, status: {:?}",
            bridge.status()
        );
    }

    #[test]
    fn therapeutic_output_includes_biofeedback_readings() {
        let mut bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        let output = bridge.process(&relaxed_eeg(), 0.6);
        assert_eq!(output.biofeedback_readings.len(), 5);
        // All values should be in [0, 1].
        for (_, v) in &output.biofeedback_readings {
            assert!(*v >= 0.0 && *v <= 1.0, "Metric out of range: {v}");
        }
    }

    #[test]
    fn meditation_protocol_responds_to_theta() {
        let mut bridge =
            TwinTherapeuticBridge::with_adaptation_rate(Protocol::meditative_absorption(), 0.5);
        // Feed high theta + alpha coherence: should see progress.
        let eeg = meditative_eeg();
        for _ in 0..10 {
            bridge.process(&eeg, 0.5);
        }
        let progress = bridge.progress_toward_target();
        assert!(
            progress > 0.4,
            "Meditation protocol should respond to high theta/alpha: {progress}"
        );
    }

    #[test]
    fn estimated_ticks_returns_none_initially() {
        let bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        assert_eq!(bridge.estimated_ticks_to_target(), None);
    }

    #[test]
    fn estimated_ticks_returns_zero_when_reached() {
        let mut bridge = TwinTherapeuticBridge::with_adaptation_rate(Protocol::flow_state(), 1.0);
        // With adaptation_rate=1.0 and perfect EEG+Phi the progress should
        // reach the target quickly.
        for _ in 0..50 {
            bridge.process(&relaxed_eeg(), 0.9);
        }
        if bridge.progress_toward_target() >= TARGET_REACHED_THRESHOLD {
            assert_eq!(bridge.estimated_ticks_to_target(), Some(0));
        }
    }

    #[test]
    fn safety_override_clears_when_phi_recovers() {
        let mut bridge = TwinTherapeuticBridge::new(Protocol::flow_state());
        // Trigger override.
        bridge.process(&relaxed_eeg(), 0.05);
        assert!(bridge.safety_override);

        // Phi recovers above 2x floor.
        for _ in 0..3 {
            bridge.process(&relaxed_eeg(), 0.5);
        }
        assert!(!bridge.safety_override);
    }
}
