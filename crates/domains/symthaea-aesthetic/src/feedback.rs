// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Aesthetic feedback: AestheticScore → neuromodulator-compatible deltas.
//!
//! This module bridges the aesthetic evaluation system with the neuromodulator
//! bath. It translates multi-dimensional beauty scores into the chemical signals
//! that drive learning, exploration, and satisfaction in the cognitive loop.

use crate::{AestheticConfig, AestheticFeedback, AestheticScore};

/// Compute aesthetic feedback from a score and its delta relative to expectation.
///
/// This is the core feedback function used by `AestheticTracker::process()`.
/// Exposed separately for testing and for systems that manage their own EMA.
///
/// # Arguments
/// * `score` — The current aesthetic score.
/// * `ema` — The running average aesthetic expectation.
/// * `config` — Feedback scaling parameters.
/// * `harmony_activations` — Current Eight Harmony activations.
pub fn compute_feedback(
    score: &AestheticScore,
    ema: f32,
    config: &AestheticConfig,
    harmony_activations: &[f32; 8],
) -> AestheticFeedback {
    let config = config.clone().sanitized();
    let expectation = if ema.is_finite() {
        ema.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let intrinsic = score.intrinsic_composite();
    let delta = intrinsic - expectation;
    let surprise = score.surprise_against(expectation);

    // Dopamine: reward prediction error
    let dopamine_delta = if delta > config.reward_threshold {
        (delta * config.dopamine_scale).min(0.15)
    } else if delta < -config.reward_threshold {
        (delta * config.dopamine_scale * 0.5).max(-0.05)
    } else {
        0.0
    };

    // Serotonin: proportional to harmony alignment
    let harmony = if score.harmony.is_finite() {
        score.harmony.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let serotonin_delta = harmony * config.serotonin_scale;

    // Surprise signal for exploration system
    let surprise_signal = surprise * config.surprise_scale;

    // Project onto active harmonies
    let mut harmony_projection = [0.0f32; 8];
    for i in 0..8 {
        let activation = if harmony_activations[i].is_finite() {
            harmony_activations[i].clamp(0.0, 1.0)
        } else {
            0.0
        };
        harmony_projection[i] = activation * intrinsic;
    }

    AestheticFeedback {
        dopamine_delta,
        serotonin_delta,
        surprise_signal,
        harmony_projection,
    }
}

/// Describe the aesthetic feedback in human-readable terms.
///
/// Useful for logging, debugging, and the Observatory dashboard.
pub fn describe_feedback(feedback: &AestheticFeedback) -> String {
    let da_desc = if feedback.dopamine_delta > 0.05 {
        "rewarding"
    } else if feedback.dopamine_delta < -0.02 {
        "disappointing"
    } else {
        "neutral"
    };

    let ne_desc = if feedback.surprise_signal > 0.05 {
        "novel"
    } else {
        "expected"
    };

    let dominant_harmony = feedback
        .harmony_projection
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let harmony_names = [
        "ResonantCoherence",
        "PanSentientFlourishing",
        "IntegralWisdom",
        "InfinitePlay",
        "UniversalInterconnectedness",
        "SacredReciprocity",
        "EvolutionaryProgression",
        "SacredStillness",
    ];

    format!(
        "Aesthetic: {} + {} (dominant harmony: {})",
        da_desc, ne_desc, harmony_names[dominant_harmony]
    )
}

/// Blend multiple aesthetic feedbacks (e.g., from multiple modalities).
///
/// Averages all feedback signals. Useful when generating visual art + music
/// simultaneously and wanting a unified feedback signal.
pub fn blend_feedbacks(feedbacks: &[AestheticFeedback]) -> AestheticFeedback {
    if feedbacks.is_empty() {
        return AestheticFeedback::neutral();
    }

    let n = feedbacks.len() as f32;
    let mut result = AestheticFeedback::neutral();

    for f in feedbacks {
        result.dopamine_delta += f.dopamine_delta / n;
        result.serotonin_delta += f.serotonin_delta / n;
        result.surprise_signal += f.surprise_signal / n;
        for i in 0..8 {
            result.harmony_projection[i] += f.harmony_projection[i] / n;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_delta_positive_dopamine() {
        let score = AestheticScore::uniform(0.8);
        let ema = 0.5; // well below current
        let config = AestheticConfig::default();
        let harmonies = [0.5; 8];

        let feedback = compute_feedback(&score, ema, &config, &harmonies);
        assert!(
            feedback.dopamine_delta > 0.0,
            "should reward: {}",
            feedback.dopamine_delta
        );
    }

    #[test]
    fn negative_delta_negative_dopamine() {
        let score = AestheticScore::uniform(0.2);
        let ema = 0.8;
        let config = AestheticConfig::default();
        let harmonies = [0.5; 8];

        let feedback = compute_feedback(&score, ema, &config, &harmonies);
        assert!(feedback.dopamine_delta < 0.0);
        assert!(feedback.dopamine_delta >= -0.05, "should be mild");
    }

    #[test]
    fn within_threshold_no_dopamine() {
        let score = AestheticScore::uniform(0.51);
        let ema = score.intrinsic_composite() - 0.01;
        let config = AestheticConfig::default();
        let harmonies = [0.5; 8];

        let feedback = compute_feedback(&score, ema, &config, &harmonies);
        assert_eq!(feedback.dopamine_delta, 0.0);
    }

    #[test]
    fn blend_averages() {
        let f1 = AestheticFeedback {
            dopamine_delta: 0.10,
            serotonin_delta: 0.04,
            surprise_signal: 0.08,
            harmony_projection: [1.0; 8],
        };
        let f2 = AestheticFeedback {
            dopamine_delta: 0.0,
            serotonin_delta: 0.02,
            surprise_signal: 0.0,
            harmony_projection: [0.0; 8],
        };

        let blended = blend_feedbacks(&[f1, f2]);
        assert!((blended.dopamine_delta - 0.05).abs() < 0.001);
        assert!((blended.serotonin_delta - 0.03).abs() < 0.001);
        assert!((blended.harmony_projection[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn blend_empty_neutral() {
        let blended = blend_feedbacks(&[]);
        assert_eq!(blended, AestheticFeedback::neutral());
    }

    #[test]
    fn describe_produces_string() {
        let feedback = AestheticFeedback {
            dopamine_delta: 0.1,
            serotonin_delta: 0.03,
            surprise_signal: 0.08,
            harmony_projection: [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
        };
        let desc = describe_feedback(&feedback);
        assert!(desc.contains("rewarding"));
        assert!(desc.contains("novel"));
        assert!(desc.contains("InfinitePlay"));
    }
}
