// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Browser embodiment telemetry for the synchronous cognitive loop.
//!
//! The bridge stores the latest bounded observation and exposes its HDC
//! encoding. It does not select or asynchronously dispatch actions; action
//! proposals are executed separately by [`crate::executor::BrowserExecutor`].
//! Consecutive-observation distance is reported as perceptual change/novelty,
//! not as sensor confidence.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, MotorSafetyLevel, grounding_label,
};

use crate::encoder::BrowserHdcEncoder;
use crate::observation::PageObservation;
use crate::safety::BrowserSafetyPolicy;

const GROUNDING_BROWSER: u8 = 2;

pub struct BrowserBridge {
    encoder: BrowserHdcEncoder,
    safety: BrowserSafetyPolicy,
    last_observation: Option<PageObservation>,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_observation_delta: f32,
    last_observation_confidence: f32,
    actions_blocked: usize,
}

impl BrowserBridge {
    pub fn new(genesis: &GenesisSeed, safety: BrowserSafetyPolicy) -> Self {
        Self {
            encoder: BrowserHdcEncoder::new(genesis),
            safety,
            last_observation: None,
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            last_observation_delta: 0.0,
            last_observation_confidence: 0.0,
            actions_blocked: 0,
        }
    }

    pub fn update_observation(&mut self, observation: PageObservation) {
        self.last_observation_confidence = observation_confidence(&observation);
        self.last_observation = Some(observation);
    }

    pub fn observation(&self) -> Option<&PageObservation> {
        self.last_observation.as_ref()
    }

    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    /// Encode the current observation and report perceptual change.
    ///
    /// `thought_hv` and `dt` are accepted for compatibility with the common
    /// embodiment interface. Browser action selection occurs outside this
    /// bridge and page-load timing is asynchronous.
    pub fn step(&mut self, _thought_hv: &ContinuousHV, _dt: f32, phi: f64) -> EmbodimentResult {
        let bounded_phi = if phi.is_finite() {
            phi.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let phi_level = MotorSafetyLevel::from_phi(bounded_phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };

        let perception = match &self.last_observation {
            Some(observation) => self.encoder.encode(observation),
            None => ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION),
        };

        let observation_delta = if let Some(previous) = &self.last_perception {
            (1.0 - perception.similarity(previous).max(0.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.last_observation_delta = observation_delta;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        EmbodimentResult {
            num_actuators: 4,
            control_effort: 0.0,
            success: self.last_observation.is_some(),
            // The common field is named prediction_error. Until an
            // action-conditioned browser world model exists, this is the
            // consecutive-observation delta and must be interpreted as such.
            prediction_error: observation_delta,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_BROWSER,
            observation_confidence: self.last_observation_confidence,
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        let vector = match &self.last_observation {
            Some(observation) => self.encoder.encode(observation),
            None => ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION),
        };
        self.last_perception = Some(vector.clone());
        vector
    }

    pub fn reset(&mut self) {
        self.last_observation = None;
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.last_observation_delta = 0.0;
        self.last_observation_confidence = 0.0;
        self.actions_blocked = 0;
    }

    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    pub fn actions_blocked(&self) -> usize {
        self.actions_blocked
    }

    pub fn observation_delta(&self) -> f32 {
        self.last_observation_delta
    }

    pub fn observation_confidence(&self) -> f32 {
        self.last_observation_confidence
    }

    pub fn safety_policy(&self) -> &BrowserSafetyPolicy {
        &self.safety
    }

    pub fn would_allow(&self, action: &crate::actions::BrowserAction, phi: f64) -> bool {
        self.safety.is_action_allowed(action, phi)
    }

    pub fn record_blocked(&mut self) {
        self.actions_blocked += 1;
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: 0.0,
            prediction_error: self.last_observation_delta,
            safety_level: self.current_safety,
            platform: "browser".to_string(),
            num_actuators: 4,
            epistemic_grounding: grounding_label(GROUNDING_BROWSER).to_string(),
            observation_confidence: self.last_observation_confidence,
            platform_specific: Vec::new(),
        }
    }
}

fn observation_confidence(observation: &PageObservation) -> f32 {
    let mut confidence = 0.45_f32;
    if url::Url::parse(&observation.url).is_ok() {
        confidence += 0.20;
    }
    if !observation.title.is_empty() {
        confidence += 0.10;
    }
    if !observation.elements.is_empty() {
        confidence += 0.15;
    }
    confidence.min(0.90)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(url: &str, title: &str) -> PageObservation {
        PageObservation {
            url: url.into(),
            title: title.into(),
            elements: Vec::new(),
            focused_element: None,
        }
    }

    #[test]
    fn step_without_observation_reports_no_sensor_confidence() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());
        let thought = ContinuousHV::random(16384, 42);
        let result = bridge.step(&thought, 0.05, 0.7);
        assert!(!result.success);
        assert_eq!(result.observation_confidence, 0.0);
    }

    #[test]
    fn observation_change_does_not_reduce_sensor_confidence() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());
        let thought = ContinuousHV::random(16384, 42);

        bridge.update_observation(observation("https://example.com", "First"));
        bridge.step(&thought, 0.05, 0.7);
        let first_confidence = bridge.observation_confidence();

        bridge.update_observation(observation("https://other.example", "Second"));
        let result = bridge.step(&thought, 0.05, 0.7);
        assert!(result.prediction_error > 0.0);
        assert_eq!(result.observation_confidence, first_confidence);
    }

    #[test]
    fn non_finite_phi_fails_to_red_safety() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());
        let thought = ContinuousHV::random(16384, 42);
        let result = bridge.step(&thought, 0.05, f64::NAN);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn reset_clears_telemetry() {
        let genesis = GenesisSeed::from_phrase("test-browser");
        let mut bridge = BrowserBridge::new(&genesis, BrowserSafetyPolicy::default());
        let thought = ContinuousHV::random(16384, 42);
        bridge.step(&thought, 0.05, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
        assert_eq!(bridge.observation_confidence(), 0.0);
    }
}
