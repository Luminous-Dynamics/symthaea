// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Attachment style formation from caregiver interaction patterns.
//!
//! Determines the emergent attachment style from accumulated caregiver
//! interaction statistics. Follows the classic 4-category model:
//!
//! - **Secure**: High reliability + quality + warmth
//! - **Anxious**: Moderate reliability, inconsistent quality
//! - **Avoidant**: Low reliability + quality
//! - **Disorganized**: Frightening caregiver behavior (overrides other dimensions)
//!
//! Also provides a probability distribution over styles for softer classification.
//!
//! Science: Ainsworth et al. (1978), Main & Solomon (1990).

use crate::attachment::AttachmentSystem;
use crate::types::AttachmentStyle;

/// Determine attachment style from caregiver interaction statistics.
///
/// # Arguments
/// * `caregiver_reliability` - P(respond | distress), range 0-1
/// * `response_quality` - EMA of response quality, range 0-1
/// * `warmth` - EMA of emotional warmth, range 0-1
/// * `is_frightening` - Whether caregiver has exhibited frightening behavior
/// * `age_months` - Current developmental age in months
///
/// # Returns
/// The most likely attachment style given the input statistics.
pub fn determine_style(
    caregiver_reliability: f64,
    response_quality: f64,
    warmth: f64,
    is_frightening: bool,
    age_months: f64,
) -> AttachmentStyle {
    // Pre-attachment phase: no differentiated style yet
    if age_months < 6.0 {
        return AttachmentStyle::Forming;
    }

    // Frightening caregiver -> disorganized (overrides all else)
    if is_frightening {
        return AttachmentStyle::Disorganized;
    }

    // Secure: high on all three dimensions
    if caregiver_reliability > 0.7 && response_quality > 0.6 && warmth > 0.5 {
        return AttachmentStyle::Secure;
    }

    // Avoidant: consistently low across dimensions
    if caregiver_reliability < 0.3 && response_quality < 0.4 {
        return AttachmentStyle::Avoidant;
    }

    // Default: anxious (inconsistent care that doesn't meet secure or avoidant thresholds)
    AttachmentStyle::Anxious
}

/// Compute a probability distribution over attachment styles.
///
/// Uses softmax-like scoring based on how well the attachment system's
/// statistics match each style's prototypical profile.
pub fn style_probability_distribution(system: &AttachmentSystem) -> [(AttachmentStyle, f64); 4] {
    let r = system.caregiver_reliability;
    let q = system.response_quality_ema;
    let w = system.internal_working_model.self_worthy; // warmth proxy

    // Score each style: how well does the data match the prototype?
    let secure_score = r * q * w; // All high = secure
    let anxious_score = (1.0 - (r - 0.5).abs()) * q * 0.5; // Moderate reliability
    let avoidant_score = (1.0 - r) * (1.0 - q) * (1.0 - w); // All low = avoidant
    let disorganized_score = if system.frightening_history {
        0.8
    } else {
        0.05
    };

    // Normalize to probabilities
    let total = secure_score + anxious_score + avoidant_score + disorganized_score;
    if total < 1e-10 {
        return [
            (AttachmentStyle::Secure, 0.25),
            (AttachmentStyle::Anxious, 0.25),
            (AttachmentStyle::Avoidant, 0.25),
            (AttachmentStyle::Disorganized, 0.25),
        ];
    }

    [
        (AttachmentStyle::Secure, secure_score / total),
        (AttachmentStyle::Anxious, anxious_score / total),
        (AttachmentStyle::Avoidant, avoidant_score / total),
        (AttachmentStyle::Disorganized, disorganized_score / total),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CaregiverAction;

    #[test]
    fn test_forming_before_six_months() {
        assert_eq!(
            determine_style(0.9, 0.9, 0.9, false, 3.0),
            AttachmentStyle::Forming
        );
    }

    #[test]
    fn test_secure_high_all() {
        assert_eq!(
            determine_style(0.9, 0.8, 0.7, false, 12.0),
            AttachmentStyle::Secure
        );
    }

    #[test]
    fn test_avoidant_low_all() {
        assert_eq!(
            determine_style(0.1, 0.2, 0.1, false, 12.0),
            AttachmentStyle::Avoidant
        );
    }

    #[test]
    fn test_anxious_moderate() {
        assert_eq!(
            determine_style(0.5, 0.5, 0.4, false, 12.0),
            AttachmentStyle::Anxious
        );
    }

    #[test]
    fn test_disorganized_frightening_overrides() {
        // Even with high reliability, frightening -> disorganized
        assert_eq!(
            determine_style(0.9, 0.9, 0.9, true, 12.0),
            AttachmentStyle::Disorganized
        );
    }

    #[test]
    fn test_probability_distribution_sums_to_one() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(12.0);
        for _ in 0..100 {
            sys.process_interaction(&CaregiverAction::reliable());
        }

        let dist = style_probability_distribution(&sys);
        let sum: f64 = dist.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_probability_distribution_secure_dominant() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(12.0);
        for _ in 0..1000 {
            sys.process_interaction(&CaregiverAction::reliable());
        }

        let dist = style_probability_distribution(&sys);
        let secure_p = dist
            .iter()
            .find(|(s, _)| *s == AttachmentStyle::Secure)
            .unwrap()
            .1;
        let anxious_p = dist
            .iter()
            .find(|(s, _)| *s == AttachmentStyle::Anxious)
            .unwrap()
            .1;
        assert!(
            secure_p > anxious_p,
            "Secure should dominate: secure={secure_p}, anxious={anxious_p}"
        );
    }

    #[test]
    fn test_probability_all_positive() {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(12.0);
        for _ in 0..100 {
            sys.process_interaction(&CaregiverAction::inconsistent());
        }

        let dist = style_probability_distribution(&sys);
        for (style, p) in &dist {
            assert!(
                *p >= 0.0,
                "{style:?} probability should be non-negative: {p}"
            );
        }
    }

    #[test]
    fn test_boundary_conditions() {
        // Exactly at secure threshold
        assert_eq!(
            determine_style(0.71, 0.61, 0.51, false, 12.0),
            AttachmentStyle::Secure
        );
        // Just below secure threshold
        assert_ne!(
            determine_style(0.69, 0.59, 0.49, false, 12.0),
            AttachmentStyle::Secure
        );
    }
}
