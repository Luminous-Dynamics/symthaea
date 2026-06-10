// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for the nurture developmental model.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_neuromodulators::NeuromodulatorBath;

/// Developmental age in months, providing conversion to other time units.
///
/// All developmental milestones and critical period curves are parameterized
/// by months, following WHO convention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentalAge {
    /// Age in months (non-negative).
    pub months: f64,
}

impl DevelopmentalAge {
    /// Create a new developmental age, clamping negative values to zero.
    pub fn new(months: f64) -> Self {
        Self {
            months: months.max(0.0),
        }
    }

    /// Convert to years.
    pub fn years(&self) -> f64 {
        self.months / 12.0
    }

    /// Convert to weeks (using 4.33 weeks/month).
    pub fn weeks(&self) -> f64 {
        self.months * 4.33
    }

    /// Convert to days (using 30.44 days/month).
    pub fn days(&self) -> f64 {
        self.months * 30.44
    }

    /// Convert to seconds.
    pub fn seconds(&self) -> f64 {
        self.months * 30.44 * 86400.0
    }
}

/// Bowlby attachment style, emergent from caregiver interaction history.
///
/// - **Forming**: Pre-attachment (< 6 months), no differentiated style yet.
/// - **Secure**: Reliable, warm caregiver -> confident exploration.
/// - **Anxious**: Inconsistent caregiver -> hypervigilance, protest on separation.
/// - **Avoidant**: Unresponsive caregiver -> compulsive self-reliance.
/// - **Disorganized**: Frightening caregiver -> approach-avoidance conflict.
///
/// Science: Ainsworth et al. (1978) Strange Situation; Main & Solomon (1990).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttachmentStyle {
    /// Not yet established (< 6 months)
    Forming,
    /// Reliable caregiver -> confidence
    Secure,
    /// Inconsistent caregiver -> hypervigilance
    Anxious,
    /// Unresponsive caregiver -> self-reliance
    Avoidant,
    /// Frightening caregiver -> confusion
    Disorganized,
}

impl AttachmentStyle {
    /// Static string for telemetry (avoids `format!("{:?}")`).
    pub fn as_str(&self) -> &'static str {
        match self {
            AttachmentStyle::Forming => "Forming",
            AttachmentStyle::Secure => "Secure",
            AttachmentStyle::Anxious => "Anxious",
            AttachmentStyle::Avoidant => "Avoidant",
            AttachmentStyle::Disorganized => "Disorganized",
        }
    }
}

/// Bowlby's 4-stage separation distress cascade.
///
/// Progression: Calm -> Quiet -> Vocalization -> Protest -> Despair.
/// Maps to Spitz (1946) anaclitic depression at the extreme end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeparationStage {
    /// Secure, no distress
    Calm,
    /// Mild unease, scanning for caregiver
    Quiet,
    /// Calling out, attention-seeking
    Vocalization,
    /// Active distress, crying
    Protest,
    /// Withdrawal, flat affect (Spitz anaclitic depression)
    Despair,
}

impl SeparationStage {
    /// Returns distress level in [0, 1].
    pub fn distress_level(&self) -> f64 {
        match self {
            Self::Calm => 0.0,
            Self::Quiet => 0.15,
            Self::Vocalization => 0.4,
            Self::Protest => 0.75,
            Self::Despair => 1.0,
        }
    }
}

/// Developmental milestone categories (WHO classification).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MilestoneCategory {
    /// Large muscle movement: rolling, crawling, walking.
    GrossMotor,
    /// Hand and finger dexterity: grasping, pincer grip, drawing.
    FineMotor,
    /// Speech and language development from cooing to complex sentences.
    Language,
    /// Social-emotional development: attachment, play, theory of mind.
    Social,
    /// Cognitive development: object permanence, symbolic play, reasoning.
    Cognitive,
}

/// A single caregiver interaction action.
///
/// Each dimension captures a different facet of caregiving quality:
/// - `responsiveness`: P(respond | distress) proxy
/// - `warmth`: emotional quality (tone, facial expression)
/// - `attunement`: accuracy of reading infant signals
/// - `contact`: physical proximity/holding
/// - `vocalizing`: verbal engagement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaregiverAction {
    /// How quickly and consistently the caregiver responds (0-1).
    pub responsiveness: f64,
    /// Emotional quality of the response (0-1).
    pub warmth: f64,
    /// Accuracy of reading infant state (0-1).
    pub attunement: f64,
    /// Physical contact present.
    pub contact: bool,
    /// Speaking or singing to infant.
    pub vocalizing: bool,
}

impl CaregiverAction {
    /// Average of responsiveness, warmth, and attunement.
    pub fn quality(&self) -> f64 {
        (self.responsiveness + self.warmth + self.attunement) / 3.0
    }

    /// Reliable, warm, attuned caregiver (secure-promoting).
    pub fn reliable() -> Self {
        Self {
            responsiveness: 0.9,
            warmth: 0.85,
            attunement: 0.8,
            contact: true,
            vocalizing: true,
        }
    }

    /// Inconsistent caregiver (anxious-promoting).
    pub fn inconsistent() -> Self {
        Self {
            responsiveness: 0.4,
            warmth: 0.6,
            attunement: 0.3,
            contact: false,
            vocalizing: false,
        }
    }

    /// Emotionally unavailable caregiver (avoidant-promoting).
    pub fn unresponsive() -> Self {
        Self {
            responsiveness: 0.1,
            warmth: 0.2,
            attunement: 0.1,
            contact: false,
            vocalizing: false,
        }
    }

    /// Frightening caregiver (disorganized-promoting).
    pub fn frightening() -> Self {
        Self {
            responsiveness: 0.3,
            warmth: 0.0,
            attunement: 0.1,
            contact: true,
            vocalizing: false,
        }
    }
}

/// Internal Working Model (Bowlby, 1969; Bretherton & Munholland, 1999).
///
/// Three core beliefs about self, others, and world, each encoded as a scalar
/// and simultaneously maintained as a 16,384D hypervector for HDC operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalWorkingModel {
    /// "Am I worthy of care?" (0-1)
    pub self_worthy: f64,
    /// "Can I count on others?" (0-1)
    pub other_reliable: f64,
    /// "Is the world safe?" (0-1)
    pub world_safe: f64,
    /// 16,384D HDC encoding of the working model
    pub model_hv: ContinuousHV,
}

/// Neuromodulatory signature for attachment events.
///
/// Different attachment events produce characteristic neurotransmitter patterns:
/// - Separation: NE surge, 5-HT dip, adenosine accumulation
/// - Reunion: oxytocin burst, DA reward, NE reduction
/// - Despair: serotonin depletion, adenosine stress
/// - Co-regulation: oxytocin-mediated calming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentNeuromodulation {
    /// Oxytocin surge from contact/reunion (0-1).
    pub oxytocin_burst: f32,
    /// Noradrenaline change: positive = arousal during separation, negative = calming.
    pub noradrenaline_separation: f32,
    /// Serotonin change: negative = depletion in despair, positive = restoration.
    pub serotonin_despair: f32,
    /// Dopamine reward signal on reunion (0-1).
    pub dopamine_reunion: f32,
    /// Adenosine stress accumulation: positive = increase, negative = clearance.
    pub adenosine_stress: f32,
}

impl AttachmentNeuromodulation {
    /// Neuromodulatory signature for caregiver separation.
    pub fn separation() -> Self {
        Self {
            oxytocin_burst: 0.0,
            noradrenaline_separation: 0.3,
            serotonin_despair: -0.1,
            dopamine_reunion: 0.0,
            adenosine_stress: 0.2,
        }
    }

    /// Neuromodulatory signature for caregiver reunion.
    pub fn reunion() -> Self {
        Self {
            oxytocin_burst: 0.4,
            noradrenaline_separation: -0.2,
            serotonin_despair: 0.0,
            dopamine_reunion: 0.3,
            adenosine_stress: -0.1,
        }
    }

    /// Neuromodulatory signature for prolonged separation (despair).
    pub fn despair() -> Self {
        Self {
            oxytocin_burst: 0.0,
            noradrenaline_separation: 0.1,
            serotonin_despair: -0.3,
            dopamine_reunion: 0.0,
            adenosine_stress: 0.4,
        }
    }

    /// Neuromodulatory signature for co-regulation, scaled by synchrony.
    pub fn coregulation(synchrony: f64) -> Self {
        let s = synchrony as f32;
        Self {
            oxytocin_burst: 0.2 * s,
            noradrenaline_separation: -0.1 * s,
            serotonin_despair: 0.1 * s,
            dopamine_reunion: 0.1 * s,
            adenosine_stress: -0.05 * s,
        }
    }

    /// Apply this neuromodulatory signature to a NeuromodulatorBath.
    pub fn apply_to_bath(&self, bath: &mut NeuromodulatorBath) {
        if self.oxytocin_burst > 0.0 {
            bath.oxytocin.produce(self.oxytocin_burst);
        }
        if self.noradrenaline_separation > 0.0 {
            bath.noradrenaline.produce(self.noradrenaline_separation);
        }
        if self.noradrenaline_separation < 0.0 {
            bath.noradrenaline.level =
                (bath.noradrenaline.level + self.noradrenaline_separation).max(0.0);
        }
        if self.serotonin_despair < 0.0 {
            bath.serotonin.level = (bath.serotonin.level + self.serotonin_despair).max(0.0);
        }
        if self.serotonin_despair > 0.0 {
            bath.serotonin.produce(self.serotonin_despair);
        }
        if self.dopamine_reunion > 0.0 {
            bath.dopamine.produce(self.dopamine_reunion);
        }
        if self.adenosine_stress > 0.0 {
            bath.adenosine.produce(self.adenosine_stress);
        }
        if self.adenosine_stress < 0.0 {
            bath.adenosine.level = (bath.adenosine.level + self.adenosine_stress).max(0.0);
        }
    }

    /// Returns the zero/neutral neuromodulation (no effect).
    pub fn neutral() -> Self {
        Self {
            oxytocin_burst: 0.0,
            noradrenaline_separation: 0.0,
            serotonin_despair: 0.0,
            dopamine_reunion: 0.0,
            adenosine_stress: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_developmental_age_conversions() {
        let age = DevelopmentalAge::new(12.0);
        assert!((age.years() - 1.0).abs() < 1e-10);
        assert!((age.weeks() - 51.96).abs() < 0.1);
        assert!((age.days() - 365.28).abs() < 0.1);
        assert!(age.seconds() > 0.0);
    }

    #[test]
    fn test_developmental_age_negative_clamped() {
        let age = DevelopmentalAge::new(-5.0);
        assert!((age.months - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_separation_stage_distress_levels() {
        assert!((SeparationStage::Calm.distress_level() - 0.0).abs() < 1e-10);
        assert!((SeparationStage::Quiet.distress_level() - 0.15).abs() < 1e-10);
        assert!((SeparationStage::Vocalization.distress_level() - 0.4).abs() < 1e-10);
        assert!((SeparationStage::Protest.distress_level() - 0.75).abs() < 1e-10);
        assert!((SeparationStage::Despair.distress_level() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distress_monotonic() {
        let stages = [
            SeparationStage::Calm,
            SeparationStage::Quiet,
            SeparationStage::Vocalization,
            SeparationStage::Protest,
            SeparationStage::Despair,
        ];
        for i in 1..stages.len() {
            assert!(stages[i].distress_level() > stages[i - 1].distress_level());
        }
    }

    #[test]
    fn test_caregiver_action_quality() {
        let reliable = CaregiverAction::reliable();
        assert!(reliable.quality() > 0.8);

        let unresponsive = CaregiverAction::unresponsive();
        assert!(unresponsive.quality() < 0.2);

        let inconsistent = CaregiverAction::inconsistent();
        assert!(inconsistent.quality() > 0.3);
        assert!(inconsistent.quality() < 0.5);
    }

    #[test]
    fn test_caregiver_presets_ordering() {
        let reliable = CaregiverAction::reliable();
        let inconsistent = CaregiverAction::inconsistent();
        let unresponsive = CaregiverAction::unresponsive();
        let frightening = CaregiverAction::frightening();

        assert!(reliable.quality() > inconsistent.quality());
        assert!(inconsistent.quality() > unresponsive.quality());
        assert!(frightening.warmth < 0.01);
    }

    #[test]
    fn test_neuromod_separation_pattern() {
        let sep = AttachmentNeuromodulation::separation();
        assert!(
            sep.noradrenaline_separation > 0.0,
            "NE should rise on separation"
        );
        assert!(sep.serotonin_despair < 0.0, "5-HT should dip on separation");
        assert!(
            sep.oxytocin_burst < 0.01,
            "No oxytocin burst during separation"
        );
    }

    #[test]
    fn test_neuromod_reunion_pattern() {
        let reu = AttachmentNeuromodulation::reunion();
        assert!(reu.oxytocin_burst > 0.0, "Oxytocin burst on reunion");
        assert!(reu.dopamine_reunion > 0.0, "DA reward on reunion");
        assert!(
            reu.noradrenaline_separation < 0.0,
            "NE should decrease on reunion"
        );
    }

    #[test]
    fn test_neuromod_despair_pattern() {
        let desp = AttachmentNeuromodulation::despair();
        assert!(
            desp.serotonin_despair < -0.2,
            "Major 5-HT depletion in despair"
        );
        assert!(
            desp.adenosine_stress > 0.3,
            "Adenosine stress load in despair"
        );
    }

    #[test]
    fn test_neuromod_coregulation_scales_with_synchrony() {
        let low = AttachmentNeuromodulation::coregulation(0.2);
        let high = AttachmentNeuromodulation::coregulation(0.8);
        assert!(high.oxytocin_burst > low.oxytocin_burst);
        assert!(high.dopamine_reunion > low.dopamine_reunion);
    }

    #[test]
    fn test_neuromod_apply_to_bath() {
        let mut bath = NeuromodulatorBath::default();
        let initial_oxy = bath.oxytocin.level;
        let reunion = AttachmentNeuromodulation::reunion();
        reunion.apply_to_bath(&mut bath);
        assert!(
            bath.oxytocin.level > initial_oxy,
            "Oxytocin should increase on reunion"
        );
        assert!(bath.dopamine.level > 0.5, "DA should increase on reunion");
    }

    #[test]
    fn test_neuromod_neutral_no_effect() {
        let mut bath = NeuromodulatorBath::default();
        let before_oxy = bath.oxytocin.level;
        let before_da = bath.dopamine.level;
        let neutral = AttachmentNeuromodulation::neutral();
        neutral.apply_to_bath(&mut bath);
        assert!((bath.oxytocin.level - before_oxy).abs() < 1e-6);
        assert!((bath.dopamine.level - before_da).abs() < 1e-6);
    }

    #[test]
    fn test_attachment_style_enum_variants() {
        // Ensure all variants are distinct
        let styles = [
            AttachmentStyle::Forming,
            AttachmentStyle::Secure,
            AttachmentStyle::Anxious,
            AttachmentStyle::Avoidant,
            AttachmentStyle::Disorganized,
        ];
        for i in 0..styles.len() {
            for j in (i + 1)..styles.len() {
                assert_ne!(styles[i], styles[j]);
            }
        }
    }

    #[test]
    fn test_milestone_category_variants() {
        let cats = [
            MilestoneCategory::GrossMotor,
            MilestoneCategory::FineMotor,
            MilestoneCategory::Language,
            MilestoneCategory::Social,
            MilestoneCategory::Cognitive,
        ];
        assert_eq!(cats.len(), 5);
    }

    #[test]
    fn test_neuromod_separation_ne_increases_in_bath() {
        let mut bath = NeuromodulatorBath::default();
        let initial_ne = bath.noradrenaline.level;
        let sep = AttachmentNeuromodulation::separation();
        sep.apply_to_bath(&mut bath);
        assert!(
            bath.noradrenaline.level > initial_ne,
            "NE should increase on separation: before={initial_ne}, after={}",
            bath.noradrenaline.level
        );
    }

    #[test]
    fn test_neuromod_reunion_ne_decreases_in_bath() {
        let mut bath = NeuromodulatorBath::default();
        // First elevate NE via separation
        AttachmentNeuromodulation::separation().apply_to_bath(&mut bath);
        let elevated_ne = bath.noradrenaline.level;
        // Then reunion should bring it down
        AttachmentNeuromodulation::reunion().apply_to_bath(&mut bath);
        assert!(
            bath.noradrenaline.level < elevated_ne,
            "NE should decrease on reunion: elevated={elevated_ne}, after={}",
            bath.noradrenaline.level
        );
    }

    #[test]
    fn test_neuromod_despair_serotonin_drops_in_bath() {
        let mut bath = NeuromodulatorBath::default();
        let initial_5ht = bath.serotonin.level;
        let desp = AttachmentNeuromodulation::despair();
        desp.apply_to_bath(&mut bath);
        assert!(
            bath.serotonin.level < initial_5ht,
            "5-HT should decrease in despair: before={initial_5ht}, after={}",
            bath.serotonin.level
        );
    }

    #[test]
    fn test_neuromod_separation_reunion_cycle() {
        let mut bath = NeuromodulatorBath::default();
        let baseline_oxy = bath.oxytocin.level;

        // Separation: NE up, 5-HT down
        AttachmentNeuromodulation::separation().apply_to_bath(&mut bath);
        let post_sep_ne = bath.noradrenaline.level;
        let post_sep_5ht = bath.serotonin.level;

        // Reunion: oxytocin burst, DA up, NE down
        AttachmentNeuromodulation::reunion().apply_to_bath(&mut bath);
        assert!(
            bath.oxytocin.level > baseline_oxy,
            "Oxytocin should surge on reunion"
        );
        assert!(
            bath.noradrenaline.level < post_sep_ne,
            "NE should drop on reunion"
        );
        assert!(bath.dopamine.level > 0.5, "DA reward burst on reunion");

        // Co-regulation: further calming
        AttachmentNeuromodulation::coregulation(0.9).apply_to_bath(&mut bath);
        assert!(
            bath.oxytocin.level > baseline_oxy,
            "Oxytocin should remain elevated after co-regulation"
        );
    }

    #[test]
    fn test_neuromod_adenosine_stress_accumulates() {
        let mut bath = NeuromodulatorBath::default();
        let initial_ade = bath.adenosine.level;

        // Separation adds stress
        AttachmentNeuromodulation::separation().apply_to_bath(&mut bath);
        let post_sep = bath.adenosine.level;
        assert!(
            post_sep > initial_ade,
            "Adenosine should increase on separation"
        );

        // Despair adds more stress
        AttachmentNeuromodulation::despair().apply_to_bath(&mut bath);
        assert!(
            bath.adenosine.level > post_sep,
            "Adenosine should accumulate further in despair"
        );
    }

    #[test]
    fn test_neuromod_coregulation_zero_synchrony_no_effect() {
        let mut bath = NeuromodulatorBath::default();
        let before = bath.oxytocin.level;
        AttachmentNeuromodulation::coregulation(0.0).apply_to_bath(&mut bath);
        assert!(
            (bath.oxytocin.level - before).abs() < 1e-6,
            "Zero synchrony should produce no neuromodulatory effect"
        );
    }
}
