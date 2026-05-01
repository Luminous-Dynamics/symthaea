// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for the browser agent.
//!
//! Bridges the synchronous cognitive loop to the async CDP session using
//! tokio channels. The cognitive loop calls `step()` synchronously; the
//! bridge encodes the last observation, applies safety gating, and queues
//! actions for async execution.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    MotorSafetyLevel, GROUNDING_SENSORIMOTOR,
};

use crate::encoder::BrowserHdcEncoder;
use crate::observation::PageObservation;
use crate::safety::BrowserSafetyPolicy;

/// Grounding level for browser perception (higher number = less grounded).
/// Web content is second-hand information, not direct sensory experience.
/// Uses the same scale as GROUNDING_SOCIAL (2) since web is mediated info.
const GROUNDING_BROWSER: u8 = 2;

/// Browser embodiment bridge for the cognitive loop.
///
/// Unlike physical platforms (vehicle, helicopter), the browser bridge
/// operates on a different timescale — page loads take seconds, not
/// milliseconds. The bridge maintains the last observation and encodes
/// it on demand.
pub struct BrowserBridge {
    encoder: BrowserHdcEncoder,
    safety: BrowserSafetyPolicy,
    last_observation: Option<PageObservation>,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_prediction_error: f32,
    /// Number of actions blocked by safety policy this session.
    actions_blocked: usize,
}

impl BrowserBridge {
    /// Create a new browser bridge.
    pub fn new(genesis: &GenesisSeed, safety: BrowserSafetyPolicy) -> Self {
        Self {
            encoder: BrowserHdcEncoder::new(genesis),
            safety,
            last_observation: None,
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            last_prediction_error: 0.0,
            actions_blocked: 0,
        }
    }

    /// Update the observation from a CDP fetch (called from async context).
    pub fn update_observation(&mut self, observation: PageObservation) {
        self.last_observation = Some(observation);
    }

    /// Get the current observation, if any.
    pub fn observation(&self) -> Option<&PageObservation> {
        self.last_observation.as_ref()
    }

    /// Set a safety override (immune system or external halt).
    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    /// Clear the safety override.
    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    /// Cognitive loop step: encode observation, check safety, return result.
    ///
    /// The `thought_hv` is not directly used for action selection here
    /// (that happens in the cognitive loop's output phase). Instead, we
    /// compute prediction error between consecutive observations.
    pub fn step(&mut self, _thought_hv: &ContinuousHV, _dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };

        // Encode current observation
        let perception = match &self.last_observation {
            Some(obs) => self.encoder.encode(obs),
            None => ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION),
        };

        // Prediction error from consecutive observations
        let pred_error = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0_f32
        };
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        EmbodimentResult {
            num_actuators: 4,    // navigate, click, type, scroll
            control_effort: 0.0, // browser has no physical effort
            success: self.last_observation.is_some(),
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_BROWSER,
            observation_confidence: grounding_from_prediction_error(pred_error),
        }
    }

    /// Encode the current page state as a 16,384D perception HV.
    pub fn encode_perception(&mut self) -> ContinuousHV {
        let hv = match &self.last_observation {
            Some(obs) => self.encoder.encode(obs),
            None => ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION),
        };
        self.last_perception = Some(hv.clone());
        hv
    }

    /// Reset the bridge state.
    pub fn reset(&mut self) {
        self.last_observation = None;
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.last_prediction_error = 0.0;
        self.actions_blocked = 0;
    }

    /// Current safety level.
    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }

    /// Total cognitive steps completed.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Number of actions blocked by safety policy.
    pub fn actions_blocked(&self) -> usize {
        self.actions_blocked
    }

    /// Get a reference to the safety policy.
    pub fn safety_policy(&self) -> &BrowserSafetyPolicy {
        &self.safety
    }

    /// Check whether an action would be allowed under current policy and Phi.
    pub fn would_allow(&self, action: &crate::actions::BrowserAction, phi: f64) -> bool {
        self.safety.is_action_allowed(action, phi)
    }

    /// Record that an action was blocked.
    pub fn record_blocked(&mut self) {
        self.actions_blocked += 1;
    }

    /// Telemetry snapshot.
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: 0.0,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "browser".to_string(),
            num_actuators: 4,
            epistemic_grounding: grounding_label(GROUNDING_BROWSER).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_without_observation() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.05, 0.7);
        assert!(!r.success); // no observation loaded
        assert_eq!(r.num_actuators, 4);
    }

    #[test]
    fn test_step_with_observation() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());

        bridge.update_observation(PageObservation {
            url: "https://example.com".into(),
            title: "Example".into(),
            elements: vec![],
            focused_element: None,
        });

        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.05, 0.7);
        assert!(r.success);
    }

    #[test]
    fn test_safety_gating() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.05, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_perception_encoding() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());

        bridge.update_observation(PageObservation {
            url: "https://example.com".into(),
            title: "Test".into(),
            elements: vec![],
            focused_element: None,
        });

        let p = bridge.encode_perception();
        assert_eq!(p.dim(), 16384);
    }

    #[test]
    fn test_reset() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.05, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
        assert_eq!(bridge.actions_blocked(), 0);
    }

    #[test]
    fn test_telemetry() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.05, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "browser");
    }
}
