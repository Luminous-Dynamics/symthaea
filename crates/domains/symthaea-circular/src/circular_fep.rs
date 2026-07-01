// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Free Energy Principle agent for waste classification action selection.
//!
//! Selects actions based on expected free energy: epistemic value (reducing
//! uncertainty about waste classification) and pragmatic value (minimizing
//! contamination risk). Includes NRC-style safety gating.

use serde::{Deserialize, Serialize};

/// Actions the circular economy agent can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircularAction {
    /// Classify an unidentified waste item (high epistemic value).
    ClassifyWaste,
    /// Alert about detected contamination (high pragmatic value).
    AlertContamination,
    /// Recommend routing to appropriate processing stream.
    RecommendRoute,
    /// Predict decomposition timeline for the current item.
    PredictDecomposition,
    /// No action needed.
    NoOp,
}

/// NRC-inspired safety levels for waste processing operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SafetyLevel {
    /// Normal operations, no hazards detected.
    Green,
    /// Elevated attention, minor contamination possible.
    Yellow,
    /// Active concern, significant contamination detected.
    Orange,
    /// Emergency, hazardous contamination requiring immediate action.
    Red,
}

/// Observation state for the circular economy FEP agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularObservation {
    /// Encoded waste hypervector (if a waste item is present).
    pub waste_hv: Option<Vec<f32>>,
    /// Sensor readings: (temperature_c, moisture_pct, oxygen_pct).
    pub sensor_readings: Option<(f64, f64, f64)>,
    /// Whether contamination has been detected.
    pub contamination_detected: bool,
    /// Confidence in the current classification (0-1).
    pub classification_confidence: f32,
    /// Contamination score from the detector (0-1), if available.
    pub contamination_score: Option<f32>,
}

/// Free Energy Principle agent for waste classification and routing decisions.
///
/// Selects actions by minimizing expected free energy:
/// - High epistemic value: actions that reduce uncertainty (classify unknown items).
/// - High pragmatic value: actions that prevent harm (alert on contamination).
///
/// Safety gating ensures hazardous contamination triggers immediate alerts
/// regardless of other considerations.
pub struct CircularFepAgent;

impl CircularFepAgent {
    /// Create a new FEP agent.
    pub fn new() -> Self {
        Self
    }

    /// Select the best action given the current observation.
    ///
    /// Priority order:
    /// 1. If contamination detected → AlertContamination (safety-first)
    /// 2. If waste present but low confidence → ClassifyWaste (epistemic)
    /// 3. If sensor readings present and classification done → PredictDecomposition
    /// 4. If high confidence → RecommendRoute (pragmatic)
    /// 5. Otherwise → NoOp
    pub fn select_action(&self, observation: &CircularObservation) -> CircularAction {
        // Safety-first: contamination always triggers alert
        if observation.contamination_detected {
            return CircularAction::AlertContamination;
        }

        // Epistemic: reduce uncertainty about unknown waste
        if observation.waste_hv.is_some() && observation.classification_confidence < 0.5 {
            return CircularAction::ClassifyWaste;
        }

        // If well-classified and sensors available, predict decomposition
        if observation.sensor_readings.is_some() && observation.classification_confidence >= 0.5 {
            return CircularAction::PredictDecomposition;
        }

        // High confidence with waste present → route it
        if observation.waste_hv.is_some() && observation.classification_confidence > 0.8 {
            return CircularAction::RecommendRoute;
        }

        CircularAction::NoOp
    }

    /// Determine safety level from the observation.
    ///
    /// Maps contamination severity to NRC-style safety tiers:
    /// - Green: no contamination (score < 0.2 or absent)
    /// - Yellow: mild contamination (0.2 - 0.5)
    /// - Orange: significant contamination (0.5 - 0.8)
    /// - Red: hazardous contamination (> 0.8)
    pub fn safety_gate(&self, observation: &CircularObservation) -> SafetyLevel {
        let score = observation.contamination_score.unwrap_or(0.0);

        if !observation.contamination_detected && score < 0.2 {
            SafetyLevel::Green
        } else if score < 0.5 {
            SafetyLevel::Yellow
        } else if score < 0.8 {
            SafetyLevel::Orange
        } else {
            SafetyLevel::Red
        }
    }
}

impl Default for CircularFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn no_waste_obs() -> CircularObservation {
        CircularObservation {
            waste_hv: None,
            sensor_readings: None,
            contamination_detected: false,
            classification_confidence: 0.0,
            contamination_score: None,
        }
    }

    #[test]
    fn test_low_confidence_classify() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            waste_hv: Some(vec![0.0; 100]),
            classification_confidence: 0.3,
            ..no_waste_obs()
        };
        assert_eq!(agent.select_action(&obs), CircularAction::ClassifyWaste);
    }

    #[test]
    fn test_contamination_alert() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            contamination_detected: true,
            classification_confidence: 0.9,
            ..no_waste_obs()
        };
        assert_eq!(
            agent.select_action(&obs),
            CircularAction::AlertContamination
        );
    }

    #[test]
    fn test_high_confidence_with_sensors_predict() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            waste_hv: Some(vec![0.0; 100]),
            sensor_readings: Some((25.0, 55.0, 21.0)),
            classification_confidence: 0.85,
            ..no_waste_obs()
        };
        assert_eq!(
            agent.select_action(&obs),
            CircularAction::PredictDecomposition
        );
    }

    #[test]
    fn test_high_confidence_no_sensors_route() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            waste_hv: Some(vec![0.0; 100]),
            classification_confidence: 0.9,
            ..no_waste_obs()
        };
        assert_eq!(agent.select_action(&obs), CircularAction::RecommendRoute);
    }

    #[test]
    fn test_no_waste_noop() {
        let agent = CircularFepAgent::new();
        let obs = no_waste_obs();
        assert_eq!(agent.select_action(&obs), CircularAction::NoOp);
    }

    #[test]
    fn test_contamination_overrides_low_confidence() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            waste_hv: Some(vec![0.0; 100]),
            contamination_detected: true,
            classification_confidence: 0.1,
            ..no_waste_obs()
        };
        // Contamination should override the classify action
        assert_eq!(
            agent.select_action(&obs),
            CircularAction::AlertContamination
        );
    }

    #[test]
    fn test_safety_gate_green() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            contamination_score: Some(0.05),
            ..no_waste_obs()
        };
        assert_eq!(agent.safety_gate(&obs), SafetyLevel::Green);
    }

    #[test]
    fn test_safety_gate_yellow() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            contamination_detected: true,
            contamination_score: Some(0.35),
            ..no_waste_obs()
        };
        assert_eq!(agent.safety_gate(&obs), SafetyLevel::Yellow);
    }

    #[test]
    fn test_safety_gate_orange() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            contamination_detected: true,
            contamination_score: Some(0.65),
            ..no_waste_obs()
        };
        assert_eq!(agent.safety_gate(&obs), SafetyLevel::Orange);
    }

    #[test]
    fn test_safety_gate_red() {
        let agent = CircularFepAgent::new();
        let obs = CircularObservation {
            contamination_detected: true,
            contamination_score: Some(0.95),
            ..no_waste_obs()
        };
        assert_eq!(agent.safety_gate(&obs), SafetyLevel::Red);
    }

    #[test]
    fn test_safety_gate_no_score_green() {
        let agent = CircularFepAgent::new();
        let obs = no_waste_obs();
        assert_eq!(agent.safety_gate(&obs), SafetyLevel::Green);
    }
}
