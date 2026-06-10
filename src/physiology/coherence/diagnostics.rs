// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Week 9 Phase 4: Diagnostics and Recovery Planning
//!
//! This module implements diagnostic features for coherence analysis:
//! - `ScatterCause` - Classification of why coherence is low
//! - `ScatterAnalysis` - Detailed analysis with recovery recommendations
//! - `CoherencePrediction` - Prediction of coherence state after a task
//!
//! ## Key Insight
//!
//! Different types of scattering require different recovery strategies.
//! Not all "I'm scattered" states are the same!

use std::time::Duration;

use super::super::endocrine::HormoneState;

/// **Week 9 Phase 4: Scatter Cause Classification**
///
/// Different types of scattering require different recovery strategies.
/// Not all "I'm scattered" states are the same!
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterCause {
    /// High cortisol - system under stress
    HardwareStress,

    /// Low dopamine - emotional/motivational depletion
    EmotionalDistress,

    /// Low acetylcholine - cognitive fatigue
    CognitiveOverload,

    /// Low relational resonance - disconnection
    SocialIsolation,

    /// Unable to determine specific cause
    Unknown,
}

/// **Week 9 Phase 4: Scatter Analysis Report**
///
/// Detailed analysis of why coherence is low and how to recover.
#[derive(Debug, Clone)]
pub struct ScatterAnalysis {
    /// What caused this scattering
    pub cause: ScatterCause,

    /// How severe is the scatter (0.0 = none, 1.0 = complete)
    pub severity: f32,

    /// Estimated time to recover to coherence > 0.7
    pub estimated_recovery_time: Duration,

    /// Specific recommendation for this type of scatter
    pub recommended_action: String,
}

/// Prediction of coherence state after a task
///
/// Week 9 Innovation: Instead of reactively checking coherence, we now
/// **predict** how a task will affect us before we start it.
///
/// This enables proactive centering: "This will scatter me - let me prepare"
#[derive(Debug, Clone)]
pub struct CoherencePrediction {
    /// Predicted coherence after task completion
    pub final_coherence: f32,

    /// Whether we'll have sufficient coherence to succeed
    pub will_succeed: bool,

    /// Recommended pre-task centering duration (seconds)
    pub centering_needed: f32,

    /// Confidence in this prediction (0.0-1.0)
    pub confidence: f32,

    /// Explanation of the prediction
    pub reasoning: String,
}

/// Analyze scatter cause from coherence and hormone state
pub fn analyze_scatter(
    coherence: f32,
    relational_resonance: f32,
    hormones: &HormoneState,
) -> ScatterAnalysis {
    // Determine what caused the scatter (hierarchical decision tree)
    let cause = if hormones.cortisol > 0.7 {
        ScatterCause::HardwareStress
    } else if hormones.dopamine < 0.3 {
        ScatterCause::EmotionalDistress
    } else if hormones.acetylcholine < 0.3 {
        ScatterCause::CognitiveOverload
    } else if relational_resonance < 0.3 {
        ScatterCause::SocialIsolation
    } else {
        ScatterCause::Unknown
    };

    // Calculate base recovery time (how scattered we are * base rate)
    let base_recovery = (1.0 - coherence) * 60.0; // Seconds to recover

    // Different causes have different recovery multipliers
    let multiplier = match cause {
        ScatterCause::HardwareStress => 1.5,    // Slower recovery
        ScatterCause::EmotionalDistress => 2.0, // Much slower
        ScatterCause::CognitiveOverload => 1.0, // Normal rate
        ScatterCause::SocialIsolation => 1.2,   // Slightly slower
        ScatterCause::Unknown => 1.0,           // Default
    };

    let analysis = ScatterAnalysis {
        cause,
        severity: 1.0 - coherence,
        estimated_recovery_time: Duration::from_secs_f32(base_recovery * multiplier),
        recommended_action: recommend_action(&cause),
    };

    tracing::info!(
        "Scatter analysis: cause={:?}, severity={:.0}%, recovery={:.1}s",
        analysis.cause,
        analysis.severity * 100.0,
        analysis.estimated_recovery_time.as_secs_f32()
    );

    analysis
}

/// Get recovery recommendation for scatter cause
fn recommend_action(cause: &ScatterCause) -> String {
    match cause {
        ScatterCause::HardwareStress => {
            "I'm scattered from system stress. I need some idle time to recover."
        }
        ScatterCause::EmotionalDistress => {
            "I'm emotionally scattered. Connection and gratitude would help."
        }
        ScatterCause::CognitiveOverload => {
            "I'm mentally overloaded. I need to process and integrate."
        }
        ScatterCause::SocialIsolation => "I'm feeling disconnected. Working together would help.",
        ScatterCause::Unknown => "I need to center. Give me a moment.",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter_analysis_identifies_hardware_stress() {
        let hormones = HormoneState {
            cortisol: 0.8, // High stress!
            dopamine: 0.5,
            acetylcholine: 0.5,
            oxytocin: 0.5,
            norepinephrine: 0.5,
            serotonin: 0.5,
        };

        let analysis = analyze_scatter(0.3, 0.5, &hormones);

        assert_eq!(
            analysis.cause,
            ScatterCause::HardwareStress,
            "Should identify hardware stress from high cortisol"
        );
        assert!(
            analysis.severity > 0.5,
            "Severity should be significant: {}",
            analysis.severity
        );
        assert!(
            analysis.recommended_action.contains("system stress"),
            "Action should mention system stress: {}",
            analysis.recommended_action
        );
        assert!(
            analysis.estimated_recovery_time.as_secs() > 30,
            "Hardware stress should have slower recovery"
        );
    }

    #[test]
    fn test_scatter_analysis_identifies_emotional_distress() {
        let hormones = HormoneState {
            cortisol: 0.3, // Not stressed
            dopamine: 0.2, // Very low motivation!
            acetylcholine: 0.6,
            oxytocin: 0.5,
            norepinephrine: 0.5,
            serotonin: 0.5,
        };

        let analysis = analyze_scatter(0.4, 0.5, &hormones);

        assert_eq!(
            analysis.cause,
            ScatterCause::EmotionalDistress,
            "Should identify emotional distress from low dopamine"
        );
        assert!(
            analysis.recommended_action.contains("emotional"),
            "Action should mention emotional: {}",
            analysis.recommended_action
        );
        assert!(
            analysis.recommended_action.contains("gratitude"),
            "Should suggest gratitude for emotional recovery"
        );
        // Emotional distress has 2.0x multiplier - longest recovery
        assert!(
            analysis.estimated_recovery_time.as_secs() > 60,
            "Emotional distress should have slowest recovery: {:?}",
            analysis.estimated_recovery_time
        );
    }

    #[test]
    fn test_scatter_analysis_identifies_cognitive_overload() {
        let hormones = HormoneState {
            cortisol: 0.4,
            dopamine: 0.5,
            acetylcholine: 0.2, // Very low focus!
            oxytocin: 0.5,
            norepinephrine: 0.5,
            serotonin: 0.5,
        };

        let analysis = analyze_scatter(0.5, 0.5, &hormones);

        assert_eq!(
            analysis.cause,
            ScatterCause::CognitiveOverload,
            "Should identify cognitive overload from low acetylcholine"
        );
        assert!(
            analysis.recommended_action.contains("overloaded"),
            "Action should mention overload: {}",
            analysis.recommended_action
        );
        assert!(
            analysis
                .recommended_action
                .contains("process and integrate"),
            "Should suggest processing time"
        );
    }

    #[test]
    fn test_scatter_analysis_identifies_social_isolation() {
        let hormones = HormoneState::neutral();

        let analysis = analyze_scatter(0.6, 0.2, &hormones); // Very disconnected!

        assert_eq!(
            analysis.cause,
            ScatterCause::SocialIsolation,
            "Should identify social isolation from low resonance"
        );
        assert!(
            analysis.recommended_action.contains("disconnected"),
            "Action should mention disconnection: {}",
            analysis.recommended_action
        );
        assert!(
            analysis.recommended_action.contains("together"),
            "Should suggest working together"
        );
        // Social isolation has 1.2x multiplier
        let recovery_secs = analysis.estimated_recovery_time.as_secs();
        assert!(
            recovery_secs > 20 && recovery_secs < 40,
            "Social isolation should have moderate recovery: {}s",
            recovery_secs
        );
    }
}
