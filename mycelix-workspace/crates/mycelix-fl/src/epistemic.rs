//! Epistemic quality grading for federated learning gradients.
//!
//! Implements the E-N-M (Empirical-Normative-Materiality) 3D epistemic model
//! from the Mycelix Protocol. Each gradient submission is classified on all
//! three axes to determine its epistemic quality score.
//!
//! # Axes
//! - **E (Empirical)**: How verifiable is the gradient? E0 (unverifiable) → E4 (reproducible)
//! - **N (Normative)**: Who agrees this update is binding? N0 (personal) → N3 (axiomatic)
//! - **M (Materiality)**: How long does this update matter? M0 (ephemeral) → M3 (foundational)
//!
//! # Status
//! Stub implementation — axis classification is based on heuristics.

use serde::{Deserialize, Serialize};

/// E-axis score (empirical verifiability)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EmpiricalAxis {
    /// E0: Unverifiable subjective claim
    E0 = 0,
    /// E1: Anecdotal evidence
    E1 = 1,
    /// E2: Peer-reviewed claim
    E2 = 2,
    /// E3: Replicable experiment
    E3 = 3,
    /// E4: Publicly reproducible
    E4 = 4,
}

/// N-axis score (normative consensus)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NormativeAxis {
    /// N0: Personal preference
    N0 = 0,
    /// N1: Community norm
    N1 = 1,
    /// N2: Network consensus
    N2 = 2,
    /// N3: Axiomatic / constitutional
    N3 = 3,
}

/// M-axis score (materiality / permanence)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaterialityAxis {
    /// M0: Ephemeral (discard after use)
    M0 = 0,
    /// M1: Session-scoped
    M1 = 1,
    /// M2: Long-term relevant
    M2 = 2,
    /// M3: Foundational (preserve forever)
    M3 = 3,
}

/// E-N-M epistemic classification of a gradient update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicGrade {
    pub empirical: EmpiricalAxis,
    pub normative: NormativeAxis,
    pub materiality: MaterialityAxis,
    /// Composite quality score in [0.0, 1.0]
    pub composite_score: f32,
}

impl EpistemicGrade {
    /// Create a grade from axis values.
    pub fn new(e: EmpiricalAxis, n: NormativeAxis, m: MaterialityAxis) -> Self {
        let composite = Self::compute_composite(e, n, m);
        Self {
            empirical: e,
            normative: n,
            materiality: m,
            composite_score: composite,
        }
    }

    /// Compute composite score as weighted average of normalized axes.
    fn compute_composite(e: EmpiricalAxis, n: NormativeAxis, m: MaterialityAxis) -> f32 {
        let e_norm = e as u8 as f32 / 4.0; // [0, 1]
        let n_norm = n as u8 as f32 / 3.0; // [0, 1]
        let m_norm = m as u8 as f32 / 3.0; // [0, 1]
        // Weights: E=50%, N=30%, M=20%
        (0.5 * e_norm + 0.3 * n_norm + 0.2 * m_norm).clamp(0.0, 1.0)
    }
}

/// Epistemic quality grader for gradient updates.
pub struct EpistemicGrader {
    /// Minimum composite score required for submission (default: 0.25)
    pub min_quality: f32,
}

impl Default for EpistemicGrader {
    fn default() -> Self {
        Self { min_quality: 0.25 }
    }
}

impl EpistemicGrader {
    pub fn new(min_quality: f32) -> Self {
        Self { min_quality }
    }

    /// Grade a gradient submission based on available metadata.
    ///
    /// # Heuristics (stub)
    /// - `has_proof`: presence of zkSTARK proof → higher E-axis
    /// - `reputation`: reputation score → N-axis (network consensus)
    /// - `round`: training round → M-axis (later rounds are less material)
    pub fn grade(
        &self,
        has_proof: bool,
        reputation: f32,
        quality_score: f32,
        _round: u32,
    ) -> EpistemicGrade {
        // E-axis: based on proof and quality score
        let e = if has_proof && quality_score > 0.8 {
            EmpiricalAxis::E3
        } else if has_proof || quality_score > 0.6 {
            EmpiricalAxis::E2
        } else if quality_score > 0.4 {
            EmpiricalAxis::E1
        } else {
            EmpiricalAxis::E0
        };

        // N-axis: based on reputation (network consensus)
        let n = if reputation > 0.8 {
            NormativeAxis::N2
        } else if reputation > 0.5 {
            NormativeAxis::N1
        } else {
            NormativeAxis::N0
        };

        // M-axis: all FL gradient updates are session-scoped (M1)
        let m = MaterialityAxis::M1;

        EpistemicGrade::new(e, n, m)
    }

    /// Check if a grade meets the minimum quality threshold.
    pub fn is_acceptable(&self, grade: &EpistemicGrade) -> bool {
        grade.composite_score >= self.min_quality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade_with_proof_high_quality() {
        let grader = EpistemicGrader::default();
        let grade = grader.grade(true, 0.9, 0.9, 1);
        assert_eq!(grade.empirical, EmpiricalAxis::E3);
        assert_eq!(grade.normative, NormativeAxis::N2);
        assert!(grade.composite_score > 0.5);
    }

    #[test]
    fn test_grade_no_proof_low_quality() {
        let grader = EpistemicGrader::default();
        let grade = grader.grade(false, 0.2, 0.2, 1);
        assert_eq!(grade.empirical, EmpiricalAxis::E0);
        assert_eq!(grade.normative, NormativeAxis::N0);
    }

    #[test]
    fn test_composite_score_range() {
        let grader = EpistemicGrader::default();
        for &has_proof in &[true, false] {
            for &rep in &[0.0_f32, 0.5, 1.0] {
                for &quality in &[0.0_f32, 0.5, 1.0] {
                    let grade = grader.grade(has_proof, rep, quality, 1);
                    assert!(grade.composite_score >= 0.0);
                    assert!(grade.composite_score <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_acceptability_threshold() {
        let grader = EpistemicGrader::new(0.5);
        let acceptable = grader.grade(true, 0.9, 0.9, 1);
        let rejected = grader.grade(false, 0.1, 0.1, 1);
        assert!(grader.is_acceptable(&acceptable));
        assert!(!grader.is_acceptable(&rejected));
    }
}
