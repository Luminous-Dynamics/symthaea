// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Developmental profile with domain scores and normative z-scores.
//!
//! Inspired by symthaea-psych-bench's CognitiveProfile, this module provides
//! a structured assessment of developmental progress across 5 domains. Each
//! domain gets a raw score (percentage of expected milestones achieved),
//! a z-score (standard deviations from mean), and a percentile rank.
//!
//! Science: Bayley Scales of Infant Development (BSID-III, Bayley 2006).

use serde::{Deserialize, Serialize};

use crate::milestones::DevelopmentalMilestoneDb;
use crate::types::{DevelopmentalAge, MilestoneCategory};

/// A score for a single developmental domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainScore {
    /// Domain category.
    pub category: MilestoneCategory,
    /// Raw score (0-100): percentage of expected milestones achieved.
    pub raw_score: f64,
    /// Standard deviations from mean.
    pub z_score: f64,
    /// Percentile rank (0-100).
    pub percentile: f64,
    /// Names of milestones achieved.
    pub milestones_achieved: Vec<String>,
    /// Names of milestones expected at this age.
    pub milestones_expected: Vec<String>,
}

/// Overall developmental profile across all domains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentalProfile {
    /// Current developmental age.
    pub age: DevelopmentalAge,
    /// Per-domain scores.
    pub domain_scores: Vec<DomainScore>,
    /// Overall developmental quotient (weighted average of domain scores).
    pub overall_developmental_quotient: f64,
}

impl DevelopmentalProfile {
    /// Assess developmental progress against normative milestones.
    ///
    /// # Arguments
    /// * `age` - Current developmental age
    /// * `milestones_achieved` - Names of milestones the infant has achieved
    ///
    /// # Returns
    /// A comprehensive profile with per-domain scores and overall DQ.
    pub fn assess(age: &DevelopmentalAge, milestones_achieved: &[String]) -> Self {
        let categories = [
            MilestoneCategory::GrossMotor,
            MilestoneCategory::FineMotor,
            MilestoneCategory::Language,
            MilestoneCategory::Social,
            MilestoneCategory::Cognitive,
        ];

        let all_expected = DevelopmentalMilestoneDb::milestones_for_age(age);

        let mut domain_scores = Vec::new();

        for category in &categories {
            let expected: Vec<String> = all_expected
                .iter()
                .filter(|m| m.category == *category)
                .map(|m| m.name.clone())
                .collect();

            let achieved: Vec<String> = expected
                .iter()
                .filter(|name| milestones_achieved.iter().any(|a| a == *name))
                .cloned()
                .collect();

            let raw_score = if expected.is_empty() {
                100.0 // No milestones expected yet = on track
            } else {
                (achieved.len() as f64 / expected.len() as f64) * 100.0
            };

            // Assume normative mean = 100, SD = 15 (standard DQ scale)
            let z_score = Self::z_score_from_raw(raw_score, 75.0, 15.0);
            let percentile = Self::percentile_from_z(z_score);

            domain_scores.push(DomainScore {
                category: *category,
                raw_score,
                z_score,
                percentile,
                milestones_achieved: achieved,
                milestones_expected: expected,
            });
        }

        let overall_dq = if domain_scores.is_empty() {
            100.0
        } else {
            domain_scores.iter().map(|d| d.raw_score).sum::<f64>() / domain_scores.len() as f64
        };

        Self {
            age: age.clone(),
            domain_scores,
            overall_developmental_quotient: overall_dq,
        }
    }

    /// Compute z-score from raw score using mean and standard deviation.
    pub fn z_score_from_raw(raw: f64, mean: f64, sd: f64) -> f64 {
        if sd.abs() < 1e-10 {
            return 0.0;
        }
        (raw - mean) / sd
    }

    /// Approximate percentile from z-score using the error function approximation.
    ///
    /// Uses Abramowitz & Stegun (1964) approximation to the normal CDF.
    pub fn percentile_from_z(z: f64) -> f64 {
        // Approximation of the standard normal CDF
        // Using the logistic approximation: Phi(z) ~ 1 / (1 + exp(-1.7 * z))
        let cdf = 1.0 / (1.0 + (-1.7 * z).exp());
        (cdf * 100.0).clamp(0.1, 99.9)
    }

    /// Check if any domain shows significant delay (z < -2).
    pub fn has_significant_delay(&self) -> bool {
        self.domain_scores.iter().any(|d| d.z_score < -2.0)
    }

    /// Get domains that are delayed (z < -1).
    pub fn delayed_domains(&self) -> Vec<MilestoneCategory> {
        self.domain_scores
            .iter()
            .filter(|d| d.z_score < -1.0)
            .map(|d| d.category)
            .collect()
    }

    /// Get domains that are advanced (z > 1).
    pub fn advanced_domains(&self) -> Vec<MilestoneCategory> {
        self.domain_scores
            .iter()
            .filter(|d| d.z_score > 1.0)
            .map(|d| d.category)
            .collect()
    }

    /// Get the domain score for a specific category.
    pub fn domain(&self, category: MilestoneCategory) -> Option<&DomainScore> {
        self.domain_scores.iter().find(|d| d.category == category)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assess_all_achieved() {
        let age = DevelopmentalAge::new(12.0);
        let expected = DevelopmentalMilestoneDb::milestones_for_age(&age);
        let achieved: Vec<String> = expected.iter().map(|m| m.name.clone()).collect();

        let profile = DevelopmentalProfile::assess(&age, &achieved);

        assert!(
            profile.overall_developmental_quotient > 90.0,
            "All milestones achieved should give high DQ, got {}",
            profile.overall_developmental_quotient
        );
    }

    #[test]
    fn test_assess_none_achieved() {
        let age = DevelopmentalAge::new(12.0);
        let profile = DevelopmentalProfile::assess(&age, &[]);

        assert!(
            profile.overall_developmental_quotient < 10.0,
            "No milestones achieved should give low DQ, got {}",
            profile.overall_developmental_quotient
        );
    }

    #[test]
    fn test_assess_at_birth() {
        let age = DevelopmentalAge::new(0.0);
        let profile = DevelopmentalProfile::assess(&age, &[]);

        // No milestones expected at birth -> all domains at 100
        assert!(
            (profile.overall_developmental_quotient - 100.0).abs() < 1e-10,
            "No expectations at birth, DQ should be 100, got {}",
            profile.overall_developmental_quotient
        );
    }

    #[test]
    fn test_five_domains() {
        let age = DevelopmentalAge::new(12.0);
        let profile = DevelopmentalProfile::assess(&age, &[]);
        assert_eq!(profile.domain_scores.len(), 5);
    }

    #[test]
    fn test_z_score_from_raw() {
        let z = DevelopmentalProfile::z_score_from_raw(90.0, 75.0, 15.0);
        assert!((z - 1.0).abs() < 1e-10);

        let z = DevelopmentalProfile::z_score_from_raw(60.0, 75.0, 15.0);
        assert!((z - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_from_z() {
        let p0 = DevelopmentalProfile::percentile_from_z(0.0);
        assert!(
            (p0 - 50.0).abs() < 5.0,
            "z=0 should be ~50th percentile, got {p0}"
        );

        let p_pos = DevelopmentalProfile::percentile_from_z(2.0);
        assert!(p_pos > 90.0, "z=2 should be >90th percentile, got {p_pos}");

        let p_neg = DevelopmentalProfile::percentile_from_z(-2.0);
        assert!(p_neg < 10.0, "z=-2 should be <10th percentile, got {p_neg}");
    }

    #[test]
    fn test_percentile_monotonic() {
        let mut prev = 0.0;
        for z in -30..=30 {
            let z_f = z as f64 / 10.0;
            let p = DevelopmentalProfile::percentile_from_z(z_f);
            assert!(
                p >= prev,
                "Percentile should be monotonic: z={z_f}, prev={prev}, p={p}"
            );
            prev = p;
        }
    }

    #[test]
    fn test_has_significant_delay() {
        let age = DevelopmentalAge::new(24.0);
        // Achieve nothing -> significant delay
        let profile = DevelopmentalProfile::assess(&age, &[]);
        assert!(profile.has_significant_delay());
    }

    #[test]
    fn test_no_delay_when_on_track() {
        let age = DevelopmentalAge::new(12.0);
        let expected = DevelopmentalMilestoneDb::milestones_for_age(&age);
        let achieved: Vec<String> = expected.iter().map(|m| m.name.clone()).collect();

        let profile = DevelopmentalProfile::assess(&age, &achieved);
        assert!(!profile.has_significant_delay());
    }

    #[test]
    fn test_delayed_domains() {
        let age = DevelopmentalAge::new(24.0);
        let profile = DevelopmentalProfile::assess(&age, &[]);
        let delayed = profile.delayed_domains();
        assert!(
            !delayed.is_empty(),
            "Should have delayed domains with no milestones"
        );
    }

    #[test]
    fn test_domain_lookup() {
        let age = DevelopmentalAge::new(12.0);
        let profile = DevelopmentalProfile::assess(&age, &[]);
        let motor = profile.domain(MilestoneCategory::GrossMotor);
        assert!(motor.is_some());
    }

    #[test]
    fn test_partial_achievement() {
        let age = DevelopmentalAge::new(12.0);
        // Achieve only motor milestones
        let motor_milestones: Vec<String> = DevelopmentalMilestoneDb::milestones_for_age(&age)
            .iter()
            .filter(|m| m.category == MilestoneCategory::GrossMotor)
            .map(|m| m.name.clone())
            .collect();

        let profile = DevelopmentalProfile::assess(&age, &motor_milestones);

        let motor_score = profile.domain(MilestoneCategory::GrossMotor).unwrap();
        let language_score = profile.domain(MilestoneCategory::Language).unwrap();

        assert!(
            motor_score.raw_score > language_score.raw_score,
            "Motor should be higher than language: motor={}, lang={}",
            motor_score.raw_score,
            language_score.raw_score
        );
    }

    #[test]
    fn test_dq_between_0_and_100() {
        for months in [0, 6, 12, 24, 36, 48, 60] {
            let age = DevelopmentalAge::new(months as f64);
            let all = DevelopmentalMilestoneDb::milestones_for_age(&age);
            let half: Vec<String> = all
                .iter()
                .take(all.len() / 2)
                .map(|m| m.name.clone())
                .collect();
            let profile = DevelopmentalProfile::assess(&age, &half);
            assert!(
                profile.overall_developmental_quotient >= 0.0
                    && profile.overall_developmental_quotient <= 100.0,
                "DQ should be 0-100 at {}mo, got {}",
                months,
                profile.overall_developmental_quotient
            );
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let age = DevelopmentalAge::new(12.0);
        let achieved = vec!["Social smile".to_string(), "Head lift".to_string()];
        let profile = DevelopmentalProfile::assess(&age, &achieved);

        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: DevelopmentalProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.domain_scores.len(),
            profile.domain_scores.len()
        );
        assert!(
            (deserialized.overall_developmental_quotient - profile.overall_developmental_quotient)
                .abs()
                < 1e-10
        );
    }
}
