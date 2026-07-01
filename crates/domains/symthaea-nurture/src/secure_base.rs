// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Secure base behavior: caregiver presence modulates exploration.
//!
//! Securely attached infants use the caregiver as a "secure base" from which to
//! explore. When the caregiver is present and the attachment is secure, exploration
//! radius is wide. Without the caregiver, exploration contracts according to
//! attachment style.
//!
//! Science: Ainsworth (1967) "Infancy in Uganda"; Bowlby (1988) "A Secure Base".

use serde::{Deserialize, Serialize};

use crate::attachment::AttachmentSystem;
use crate::types::AttachmentStyle;

/// Secure base state tracking exploration and comfort.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureBase {
    /// How far the infant will explore from the caregiver (0-1).
    pub exploration_radius: f64,
    /// Comfort derived from caregiver proximity (0-1).
    pub comfort_from_proximity: f64,
    /// Exploration quality: joy/curiosity during exploration (0-1).
    pub exploration_quality: f64,
}

impl SecureBase {
    /// Create a new SecureBase with moderate defaults.
    pub fn new() -> Self {
        Self {
            exploration_radius: 0.3,
            comfort_from_proximity: 0.5,
            exploration_quality: 0.5,
        }
    }

    /// Update exploration parameters based on attachment and caregiver presence.
    ///
    /// When the caregiver is present:
    /// - Secure: wide exploration (0.3 + 0.7 * security) with high quality
    /// - Anxious: moderate exploration but frequent check-backs
    /// - Avoidant: moderate exploration but low comfort
    /// - Disorganized: restricted, confused exploration
    ///
    /// When the caregiver is absent:
    /// - Secure: moderate exploration continues (internalized security)
    /// - Anxious: severely restricted (hypervigilant scanning)
    /// - Avoidant: appears to explore but without joy (defensive autonomy)
    /// - Disorganized: restricted and disoriented
    pub fn update(&mut self, attachment: &AttachmentSystem, caregiver_present: bool) {
        if caregiver_present {
            // Secure attachment -> wider exploration
            self.exploration_radius = 0.3 + 0.7 * attachment.security_score;
            self.comfort_from_proximity = attachment.security_score;

            self.exploration_quality = match attachment.style {
                AttachmentStyle::Secure => 0.4 + 0.6 * attachment.security_score,
                AttachmentStyle::Anxious => 0.3, // Distracted by monitoring caregiver
                AttachmentStyle::Avoidant => 0.4, // Functional but joyless
                AttachmentStyle::Disorganized => 0.2, // Confused
                AttachmentStyle::Forming => 0.3 + 0.3 * attachment.security_score,
            };
        } else {
            // Without caregiver, exploration depends on attachment style
            self.exploration_radius = match attachment.style {
                AttachmentStyle::Secure => 0.5,       // Internalized security
                AttachmentStyle::Anxious => 0.1,      // Hypervigilant, restricted
                AttachmentStyle::Avoidant => 0.7,     // Appears to explore (defensive)
                AttachmentStyle::Disorganized => 0.2, // Disoriented
                AttachmentStyle::Forming => 0.3,      // Default
            };

            self.comfort_from_proximity = 0.0; // No caregiver, no proximity comfort

            self.exploration_quality = match attachment.style {
                AttachmentStyle::Secure => 0.4,       // Still some joy
                AttachmentStyle::Anxious => 0.1,      // Anxious scanning, no joy
                AttachmentStyle::Avoidant => 0.2,     // Mechanical, not joyful
                AttachmentStyle::Disorganized => 0.1, // Frozen/confused
                AttachmentStyle::Forming => 0.2,
            };
        }
    }

    /// Effective exploration capacity: radius * quality.
    pub fn effective_exploration(&self) -> f64 {
        self.exploration_radius * self.exploration_quality
    }
}

impl Default for SecureBase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CaregiverAction;

    fn build_attachment(style_action: &CaregiverAction) -> AttachmentSystem {
        let mut sys = AttachmentSystem::new();
        sys.advance_age(12.0);
        for _ in 0..1000 {
            sys.process_interaction(style_action);
        }
        sys
    }

    #[test]
    fn test_secure_wide_exploration_with_caregiver() {
        let secure = build_attachment(&CaregiverAction::reliable());
        let mut base = SecureBase::new();
        base.update(&secure, true);

        assert!(
            base.exploration_radius > 0.7,
            "Secure + caregiver should have wide exploration, got {}",
            base.exploration_radius
        );
        assert!(
            base.comfort_from_proximity > 0.7,
            "Secure should have high comfort from proximity, got {}",
            base.comfort_from_proximity
        );
        assert!(
            base.exploration_quality > 0.7,
            "Secure should have high exploration quality, got {}",
            base.exploration_quality
        );
    }

    #[test]
    fn test_anxious_restricted_without_caregiver() {
        let anxious = build_attachment(&CaregiverAction::inconsistent());
        let mut base = SecureBase::new();
        base.update(&anxious, false);

        assert!(
            base.exploration_radius < 0.2,
            "Anxious without caregiver should be restricted, got {}",
            base.exploration_radius
        );
        assert!(
            base.exploration_quality < 0.2,
            "Anxious without caregiver should have low quality, got {}",
            base.exploration_quality
        );
    }

    #[test]
    fn test_avoidant_appears_to_explore_without_caregiver() {
        let avoidant = build_attachment(&CaregiverAction::unresponsive());
        let mut base = SecureBase::new();
        base.update(&avoidant, false);

        assert!(
            base.exploration_radius > 0.6,
            "Avoidant should appear to explore widely, got {}",
            base.exploration_radius
        );
        assert!(
            base.exploration_quality < 0.3,
            "Avoidant exploration should be joyless, got {}",
            base.exploration_quality
        );
    }

    #[test]
    fn test_caregiver_presence_increases_exploration() {
        let secure = build_attachment(&CaregiverAction::reliable());
        let mut with_cg = SecureBase::new();
        with_cg.update(&secure, true);

        let mut without_cg = SecureBase::new();
        without_cg.update(&secure, false);

        assert!(
            with_cg.exploration_radius > without_cg.exploration_radius,
            "Caregiver presence should increase exploration: {} vs {}",
            with_cg.exploration_radius,
            without_cg.exploration_radius
        );
    }

    #[test]
    fn test_effective_exploration_secure_vs_anxious() {
        let secure = build_attachment(&CaregiverAction::reliable());
        let anxious = build_attachment(&CaregiverAction::inconsistent());

        let mut s_base = SecureBase::new();
        s_base.update(&secure, true);

        let mut a_base = SecureBase::new();
        a_base.update(&anxious, true);

        assert!(
            s_base.effective_exploration() > a_base.effective_exploration(),
            "Secure should have higher effective exploration: {} vs {}",
            s_base.effective_exploration(),
            a_base.effective_exploration()
        );
    }

    #[test]
    fn test_disorganized_restricted() {
        let disorganized = build_attachment(&CaregiverAction::frightening());
        let mut base = SecureBase::new();
        base.update(&disorganized, false);

        assert!(
            base.exploration_radius < 0.3,
            "Disorganized should have restricted exploration, got {}",
            base.exploration_radius
        );
    }

    #[test]
    fn test_no_comfort_without_caregiver() {
        let secure = build_attachment(&CaregiverAction::reliable());
        let mut base = SecureBase::new();
        base.update(&secure, false);

        assert!(
            base.comfort_from_proximity < 0.01,
            "No comfort without caregiver, got {}",
            base.comfort_from_proximity
        );
    }
}
