// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core data types for ectogenesis modeling.

use serde::{Deserialize, Serialize};

/// Total weeks of human gestation.
pub const WEEKS_GESTATION: u32 = 40;

/// Seconds in one gestational week.
pub const SECONDS_PER_WEEK: f32 = 604_800.0;

/// Gestational week (0-40), clamped to valid range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GestationalWeek(pub u32);

impl GestationalWeek {
    /// Create a new gestational week, clamped to [0, 40].
    pub fn new(week: u32) -> Self {
        Self(week.min(WEEKS_GESTATION))
    }

    /// Return trimester (1, 2, or 3).
    pub fn trimester(&self) -> u8 {
        match self.0 {
            0..=13 => 1,
            14..=27 => 2,
            _ => 3,
        }
    }

    /// Whether the fetus has reached viability (~24 weeks).
    pub fn is_viable(&self) -> bool {
        self.0 >= 24
    }

    /// Whether the pregnancy is full term (>=37 weeks).
    pub fn is_full_term(&self) -> bool {
        self.0 >= 37
    }
}

/// Comprehensive fetal biometrics at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetalMetrics {
    /// Estimated fetal weight in grams.
    pub weight_grams: f64,
    /// Crown-rump length in millimeters.
    pub crown_rump_length_mm: f64,
    /// Fetal heart rate in beats per minute.
    pub heart_rate_bpm: f64,
    /// Normalized movement score (0-1).
    pub movement_score: f64,
    /// Normalized brain activity score (0-1).
    pub brain_activity_score: f64,
    /// Lung maturity / surfactant production (0-1).
    pub lung_maturity: f64,
    /// Current gestational week.
    pub week: GestationalWeek,
}

impl FetalMetrics {
    /// Expected weight by gestational week (simplified Hadlock-like curve).
    ///
    /// Models exponential growth from ~0.1g (week <4) to ~3300g at term.
    pub fn expected_weight(week: GestationalWeek) -> f64 {
        let w = week.0 as f64;
        if w < 4.0 {
            return 0.1;
        }
        0.5 * (0.22 * w).exp()
    }

    /// Expected heart rate by gestational week.
    ///
    /// Follows the known pattern: accelerates from week 6, peaks around week 9-10
    /// at ~170 bpm, then gradually declines to ~130 bpm at term.
    pub fn expected_heart_rate(week: GestationalWeek) -> f64 {
        let w = week.0 as f64;
        match week.0 {
            0..=5 => 0.0,
            6..=9 => 100.0 + (w - 6.0) * 20.0,
            10..=14 => 170.0 - (w - 10.0) * 2.0,
            _ => 140.0 - (w - 14.0) * 0.5,
        }
    }

    /// Expected crown-rump length (mm) by gestational week.
    pub fn expected_crl(week: GestationalWeek) -> f64 {
        let w = week.0 as f64;
        if w < 6.0 {
            return w * 0.5;
        }
        // Approximately linear growth: ~3mm at week 6 to ~500mm at week 40
        3.0 + (w - 6.0) * 14.6
    }

    /// Expected lung maturity (0-1) by gestational week.
    /// Surfactant production begins ~24 weeks, adequate by ~34 weeks.
    pub fn expected_lung_maturity(week: GestationalWeek) -> f64 {
        let w = week.0 as f64;
        if w < 20.0 {
            return 0.0;
        }
        // Sigmoid rise from 0 at week 20 to ~1.0 at week 38
        let x = (w - 30.0) / 3.0;
        1.0 / (1.0 + (-x).exp())
    }

    /// Build a normative (expected) `FetalMetrics` for a given week.
    pub fn normative(week: GestationalWeek) -> Self {
        let w = week.0 as f64;
        Self {
            weight_grams: Self::expected_weight(week),
            crown_rump_length_mm: Self::expected_crl(week),
            heart_rate_bpm: Self::expected_heart_rate(week),
            movement_score: if w >= 16.0 {
                (0.3 + (w - 16.0) * 0.03).min(1.0)
            } else {
                0.0
            },
            brain_activity_score: if w >= 8.0 {
                ((w - 8.0) / 32.0).min(1.0)
            } else {
                0.0
            },
            lung_maturity: Self::expected_lung_maturity(week),
            week,
        }
    }
}

/// Eight key developmental milestones spanning gestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DevelopmentalMilestone {
    /// Week 4: Neural tube closes.
    NeuralTubeFormation,
    /// Week 6: First heartbeat detected.
    HeartbeatDetected,
    /// Week 8: Limb buds form.
    LimbBudFormation,
    /// Week 12: Primary organ formation complete.
    OrganogenesisComplete,
    /// Week 16: First fetal movements felt.
    Quickening,
    /// Week 24: Viability threshold.
    Viability,
    /// Week 34: Adequate surfactant production.
    LungMaturation,
    /// Week 37-40: Full term.
    FullTerm,
}

impl DevelopmentalMilestone {
    /// The gestational week at which this milestone is expected.
    pub fn expected_week(&self) -> GestationalWeek {
        GestationalWeek(match self {
            Self::NeuralTubeFormation => 4,
            Self::HeartbeatDetected => 6,
            Self::LimbBudFormation => 8,
            Self::OrganogenesisComplete => 12,
            Self::Quickening => 16,
            Self::Viability => 24,
            Self::LungMaturation => 34,
            Self::FullTerm => 37,
        })
    }

    /// All eight milestones in chronological order.
    pub fn all() -> [DevelopmentalMilestone; 8] {
        [
            Self::NeuralTubeFormation,
            Self::HeartbeatDetected,
            Self::LimbBudFormation,
            Self::OrganogenesisComplete,
            Self::Quickening,
            Self::Viability,
            Self::LungMaturation,
            Self::FullTerm,
        ]
    }
}

/// Consent proxy tier based on developmental stage.
///
/// As the fetus develops, its moral patient status escalates,
/// requiring increasingly stringent approval for interventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentProxy {
    /// Weeks 0-4: minimal sentience; PATIENT weight 0.2.
    PreSentient,
    /// Weeks 5-23: emerging neural activity; PATIENT weight 0.6.
    EmergingSentience,
    /// Weeks 24+: viability / pain perception; PATIENT weight 1.0.
    Sentient,
}

impl ConsentProxy {
    /// Determine consent proxy tier from gestational week.
    pub fn from_week(week: GestationalWeek) -> Self {
        match week.0 {
            0..=4 => Self::PreSentient,
            5..=23 => Self::EmergingSentience,
            _ => Self::Sentient,
        }
    }

    /// Moral patient weight for this tier.
    pub fn patient_weight(&self) -> f64 {
        match self {
            Self::PreSentient => 0.2,
            Self::EmergingSentience => 0.6,
            Self::Sentient => 1.0,
        }
    }
}

/// Amniotic fluid parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmnioticFluid {
    /// Target temperature in Celsius (~37.0).
    pub temperature_celsius: f64,
    /// pH level (~7.0 - 7.25).
    pub ph: f64,
    /// Volume in milliliters (varies by week).
    pub volume_ml: f64,
    /// Electrolyte balance score (0-1, 1 = optimal).
    pub electrolyte_balance: f64,
    /// Protein concentration in mg/dL.
    pub protein_concentration: f64,
}

/// Gestational hormone levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HormoneProfile {
    /// Human chorionic gonadotropin (mIU/mL).
    pub hcg: f64,
    /// Progesterone (ng/mL).
    pub progesterone: f64,
    /// Estradiol (pg/mL).
    pub estradiol: f64,
    /// Cortisol (mcg/dL).
    pub cortisol: f64,
    /// Free thyroxine T4 (mcg/dL).
    pub thyroid_t4: f64,
}

/// Oxygenation parameters for the artificial placenta circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxygenationParams {
    /// Fraction of inspired oxygen (0-1).
    pub fio2: f64,
    /// Flow rate in mL/min.
    pub flow_rate_ml_min: f64,
    /// Target partial pressure of O2 (mmHg).
    pub pao2_target: f64,
    /// Target partial pressure of CO2 (mmHg).
    pub paco2_target: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gestational_week_clamping() {
        assert_eq!(GestationalWeek::new(0).0, 0);
        assert_eq!(GestationalWeek::new(40).0, 40);
        assert_eq!(GestationalWeek::new(50).0, 40);
    }

    #[test]
    fn test_trimesters() {
        assert_eq!(GestationalWeek::new(1).trimester(), 1);
        assert_eq!(GestationalWeek::new(13).trimester(), 1);
        assert_eq!(GestationalWeek::new(14).trimester(), 2);
        assert_eq!(GestationalWeek::new(27).trimester(), 2);
        assert_eq!(GestationalWeek::new(28).trimester(), 3);
        assert_eq!(GestationalWeek::new(40).trimester(), 3);
    }

    #[test]
    fn test_viability() {
        assert!(!GestationalWeek::new(23).is_viable());
        assert!(GestationalWeek::new(24).is_viable());
        assert!(GestationalWeek::new(40).is_viable());
    }

    #[test]
    fn test_full_term() {
        assert!(!GestationalWeek::new(36).is_full_term());
        assert!(GestationalWeek::new(37).is_full_term());
        assert!(GestationalWeek::new(40).is_full_term());
    }

    #[test]
    fn test_expected_weight_increases() {
        let w10 = FetalMetrics::expected_weight(GestationalWeek::new(10));
        let w20 = FetalMetrics::expected_weight(GestationalWeek::new(20));
        let w30 = FetalMetrics::expected_weight(GestationalWeek::new(30));
        let w40 = FetalMetrics::expected_weight(GestationalWeek::new(40));
        assert!(w10 < w20);
        assert!(w20 < w30);
        assert!(w30 < w40);
    }

    #[test]
    fn test_early_weight_tiny() {
        assert!(FetalMetrics::expected_weight(GestationalWeek::new(2)) < 1.0);
    }

    #[test]
    fn test_heart_rate_pattern() {
        // No heartbeat before week 6
        assert_eq!(
            FetalMetrics::expected_heart_rate(GestationalWeek::new(3)),
            0.0
        );
        // Accelerates 6-9
        let hr6 = FetalMetrics::expected_heart_rate(GestationalWeek::new(6));
        let hr9 = FetalMetrics::expected_heart_rate(GestationalWeek::new(9));
        assert!(hr9 > hr6);
        // Peak around 9-10
        let hr10 = FetalMetrics::expected_heart_rate(GestationalWeek::new(10));
        assert!(hr10 >= 160.0);
        // Decline after peak
        let hr30 = FetalMetrics::expected_heart_rate(GestationalWeek::new(30));
        assert!(hr30 < hr10);
    }

    #[test]
    fn test_lung_maturity_sigmoid() {
        let lm20 = FetalMetrics::expected_lung_maturity(GestationalWeek::new(20));
        let lm34 = FetalMetrics::expected_lung_maturity(GestationalWeek::new(34));
        let lm38 = FetalMetrics::expected_lung_maturity(GestationalWeek::new(38));
        assert!(lm20 < 0.1);
        assert!(lm34 > 0.5);
        assert!(lm38 > 0.9);
    }

    #[test]
    fn test_normative_metrics() {
        let m = FetalMetrics::normative(GestationalWeek::new(30));
        assert!(m.weight_grams > 100.0);
        assert!(m.heart_rate_bpm > 100.0);
        assert!(m.lung_maturity > 0.0);
        assert!(m.movement_score > 0.0);
        assert!(m.brain_activity_score > 0.0);
    }

    #[test]
    fn test_milestone_expected_weeks_ordered() {
        let milestones = DevelopmentalMilestone::all();
        for i in 1..milestones.len() {
            assert!(milestones[i].expected_week() > milestones[i - 1].expected_week());
        }
    }

    #[test]
    fn test_consent_proxy_escalation() {
        assert_eq!(
            ConsentProxy::from_week(GestationalWeek::new(2)),
            ConsentProxy::PreSentient
        );
        assert_eq!(
            ConsentProxy::from_week(GestationalWeek::new(4)),
            ConsentProxy::PreSentient
        );
        assert_eq!(
            ConsentProxy::from_week(GestationalWeek::new(5)),
            ConsentProxy::EmergingSentience
        );
        assert_eq!(
            ConsentProxy::from_week(GestationalWeek::new(23)),
            ConsentProxy::EmergingSentience
        );
        assert_eq!(
            ConsentProxy::from_week(GestationalWeek::new(24)),
            ConsentProxy::Sentient
        );
        assert_eq!(
            ConsentProxy::from_week(GestationalWeek::new(40)),
            ConsentProxy::Sentient
        );
    }

    #[test]
    fn test_patient_weight_monotonic() {
        let pw_pre = ConsentProxy::PreSentient.patient_weight();
        let pw_em = ConsentProxy::EmergingSentience.patient_weight();
        let pw_sent = ConsentProxy::Sentient.patient_weight();
        assert!(pw_pre < pw_em);
        assert!(pw_em < pw_sent);
        assert!((pw_sent - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gestational_week_serde_roundtrip() {
        let week = GestationalWeek::new(25);
        let json = serde_json::to_string(&week).unwrap();
        let deser: GestationalWeek = serde_json::from_str(&json).unwrap();
        assert_eq!(week, deser);
    }

    #[test]
    fn test_fetal_metrics_serde_roundtrip() {
        let m = FetalMetrics::normative(GestationalWeek::new(20));
        let json = serde_json::to_string(&m).unwrap();
        let deser: FetalMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.week, m.week);
        assert!((deser.weight_grams - m.weight_grams).abs() < 1e-6);
    }

    #[test]
    fn test_consent_proxy_serde_roundtrip() {
        let proxy = ConsentProxy::Sentient;
        let json = serde_json::to_string(&proxy).unwrap();
        let deser: ConsentProxy = serde_json::from_str(&json).unwrap();
        assert_eq!(proxy, deser);
    }

    #[test]
    fn test_milestone_all_count() {
        assert_eq!(DevelopmentalMilestone::all().len(), 8);
    }
}
