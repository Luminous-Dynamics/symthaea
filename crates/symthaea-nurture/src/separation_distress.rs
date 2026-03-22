// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 4-stage separation distress cascade (Bowlby, 1969).
//!
//! Progression follows Bowlby's observed sequence:
//! Calm -> Quiet -> Vocalization -> Protest -> Despair
//!
//! Escalation rate is modulated by attachment style:
//! - Anxious: 2x faster (hypervigilant, rapid protest)
//! - Avoidant: 0.3x slower (suppressed distress)
//! - Disorganized: 1.5x (unpredictable escalation)
//! - Secure: 1.0x (normal, proportionate)
//!
//! Each stage has a characteristic neuromodulatory signature.
//!
//! Science: Bowlby (1969/1982), Robertson & Bowlby (1952), Spitz (1946).

use crate::attachment::AttachmentSystem;
use crate::types::{AttachmentNeuromodulation, AttachmentStyle, SeparationStage};

/// Advance the separation distress cascade based on elapsed minutes.
///
/// Time thresholds (for secure/forming attachment):
/// - Calm -> Quiet: 5 min
/// - Quiet -> Vocalization: 15 min
/// - Vocalization -> Protest: 30 min
/// - Protest -> Despair: 60 min
///
/// These thresholds are scaled by attachment-style escalation rate.
pub fn advance_separation(
    current: SeparationStage,
    minutes_elapsed: f64,
    attachment: &AttachmentSystem,
) -> SeparationStage {
    // Attachment style modulates escalation speed
    let escalation_rate = match attachment.style {
        AttachmentStyle::Anxious => 2.0,      // Hypervigilant, rapid protest
        AttachmentStyle::Avoidant => 0.3,     // Suppressed, stays quiet longer
        AttachmentStyle::Disorganized => 1.5, // Unpredictable, somewhat fast
        AttachmentStyle::Secure => 1.0,       // Normal, proportionate
        AttachmentStyle::Forming => 1.0,      // Default
    };

    let effective_minutes = minutes_elapsed * escalation_rate;

    match current {
        SeparationStage::Calm if effective_minutes > 5.0 => SeparationStage::Quiet,
        SeparationStage::Quiet if effective_minutes > 15.0 => SeparationStage::Vocalization,
        SeparationStage::Vocalization if effective_minutes > 30.0 => SeparationStage::Protest,
        SeparationStage::Protest if effective_minutes > 60.0 => SeparationStage::Despair,
        other => other,
    }
}

/// Get the neuromodulatory signature for a separation stage.
///
/// - Calm: neutral (no stress response)
/// - Quiet: mild NE increase
/// - Vocalization: moderate NE + mild oxytocin seeking
/// - Protest: full separation response (high NE, 5-HT dip, adenosine)
/// - Despair: anaclitic depression (5-HT depletion, high adenosine)
pub fn neuromod_for_stage(stage: SeparationStage) -> AttachmentNeuromodulation {
    match stage {
        SeparationStage::Calm => AttachmentNeuromodulation::neutral(),
        SeparationStage::Quiet => AttachmentNeuromodulation {
            oxytocin_burst: 0.0,
            noradrenaline_separation: 0.1,
            serotonin_despair: 0.0,
            dopamine_reunion: 0.0,
            adenosine_stress: 0.05,
        },
        SeparationStage::Vocalization => AttachmentNeuromodulation {
            oxytocin_burst: 0.05, // Seeking connection
            noradrenaline_separation: 0.2,
            serotonin_despair: -0.05,
            dopamine_reunion: 0.0,
            adenosine_stress: 0.1,
        },
        SeparationStage::Protest => AttachmentNeuromodulation::separation(),
        SeparationStage::Despair => AttachmentNeuromodulation::despair(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_attachment(style: AttachmentStyle) -> AttachmentSystem {
        let mut sys = AttachmentSystem::new();
        sys.style = style;
        sys
    }

    #[test]
    fn test_calm_to_quiet_at_5_minutes() {
        let attachment = make_attachment(AttachmentStyle::Secure);
        let result = advance_separation(SeparationStage::Calm, 4.0, &attachment);
        assert_eq!(result, SeparationStage::Calm, "Should stay calm at 4 min");

        let result = advance_separation(SeparationStage::Calm, 6.0, &attachment);
        assert_eq!(result, SeparationStage::Quiet, "Should be quiet at 6 min");
    }

    #[test]
    fn test_quiet_to_vocalization_at_15_minutes() {
        let attachment = make_attachment(AttachmentStyle::Secure);
        let result = advance_separation(SeparationStage::Quiet, 14.0, &attachment);
        assert_eq!(
            result,
            SeparationStage::Quiet,
            "Should stay quiet at 14 min"
        );

        let result = advance_separation(SeparationStage::Quiet, 16.0, &attachment);
        assert_eq!(
            result,
            SeparationStage::Vocalization,
            "Should vocalize at 16 min"
        );
    }

    #[test]
    fn test_vocalization_to_protest_at_30_minutes() {
        let attachment = make_attachment(AttachmentStyle::Secure);
        let result = advance_separation(SeparationStage::Vocalization, 29.0, &attachment);
        assert_eq!(result, SeparationStage::Vocalization);

        let result = advance_separation(SeparationStage::Vocalization, 31.0, &attachment);
        assert_eq!(result, SeparationStage::Protest);
    }

    #[test]
    fn test_protest_to_despair_at_60_minutes() {
        let attachment = make_attachment(AttachmentStyle::Secure);
        let result = advance_separation(SeparationStage::Protest, 59.0, &attachment);
        assert_eq!(result, SeparationStage::Protest);

        let result = advance_separation(SeparationStage::Protest, 61.0, &attachment);
        assert_eq!(result, SeparationStage::Despair);
    }

    #[test]
    fn test_anxious_faster_escalation() {
        let anxious = make_attachment(AttachmentStyle::Anxious);
        // Anxious at 2x rate: 3 min * 2 = 6 effective min -> should be quiet
        let result = advance_separation(SeparationStage::Calm, 3.0, &anxious);
        assert_eq!(
            result,
            SeparationStage::Quiet,
            "Anxious should escalate at 3 min (2x rate)"
        );

        // Secure at same time: 3 min * 1 = 3 effective min -> still calm
        let secure = make_attachment(AttachmentStyle::Secure);
        let result = advance_separation(SeparationStage::Calm, 3.0, &secure);
        assert_eq!(
            result,
            SeparationStage::Calm,
            "Secure should stay calm at 3 min"
        );
    }

    #[test]
    fn test_avoidant_slower_escalation() {
        let avoidant = make_attachment(AttachmentStyle::Avoidant);
        // Avoidant at 0.3x rate: 10 min * 0.3 = 3 effective min -> still calm
        let result = advance_separation(SeparationStage::Calm, 10.0, &avoidant);
        assert_eq!(
            result,
            SeparationStage::Calm,
            "Avoidant should stay calm at 10 min (0.3x rate)"
        );

        // Avoidant needs 5/0.3 = ~17 min to become quiet
        let result = advance_separation(SeparationStage::Calm, 18.0, &avoidant);
        assert_eq!(
            result,
            SeparationStage::Quiet,
            "Avoidant should be quiet at 18 min"
        );
    }

    #[test]
    fn test_despair_is_terminal() {
        let attachment = make_attachment(AttachmentStyle::Secure);
        let result = advance_separation(SeparationStage::Despair, 1000.0, &attachment);
        assert_eq!(result, SeparationStage::Despair, "Despair is terminal");
    }

    #[test]
    fn test_neuromod_calm_neutral() {
        let nm = neuromod_for_stage(SeparationStage::Calm);
        assert!((nm.oxytocin_burst - 0.0).abs() < 1e-6);
        assert!((nm.noradrenaline_separation - 0.0).abs() < 1e-6);
        assert!((nm.serotonin_despair - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_neuromod_escalation_pattern() {
        let quiet = neuromod_for_stage(SeparationStage::Quiet);
        let vocal = neuromod_for_stage(SeparationStage::Vocalization);
        let protest = neuromod_for_stage(SeparationStage::Protest);
        let despair = neuromod_for_stage(SeparationStage::Despair);

        // NE should generally increase through protest
        assert!(vocal.noradrenaline_separation > quiet.noradrenaline_separation);
        assert!(protest.noradrenaline_separation > vocal.noradrenaline_separation);

        // Adenosine stress should increase
        assert!(despair.adenosine_stress > protest.adenosine_stress);

        // Serotonin should deplete in despair
        assert!(despair.serotonin_despair < protest.serotonin_despair);
    }

    #[test]
    fn test_disorganized_escalation_moderate() {
        let disorganized = make_attachment(AttachmentStyle::Disorganized);
        // 1.5x rate: 4 min * 1.5 = 6 -> quiet
        let result = advance_separation(SeparationStage::Calm, 4.0, &disorganized);
        assert_eq!(
            result,
            SeparationStage::Quiet,
            "Disorganized escalates at 1.5x"
        );
    }
}
