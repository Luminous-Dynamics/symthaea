// SPDX-License-Identifier: AGPL-3.0-or-later
//! Collision-risk review boundary for surface vessels.
//!
//! This module does not encode COLREG give-way/stand-on manoeuvres. It only decides whether
//! relative-motion evidence is sufficient to remain in routine monitoring or must be escalated
//! to a separate, verified COLREG/navigation policy. Thresholds are operator/policy inputs.

use serde::{Deserialize, Serialize};

/// Relative-motion evidence for one contact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactMotionObservation {
    /// Current range to contact in metres.
    pub range_m: f32,
    /// Distance at closest point of approach in metres, if reliably estimated.
    pub dcpa_m: Option<f32>,
    /// Time to closest point of approach in seconds. Negative means CPA is in the past.
    pub tcpa_s: Option<f32>,
    /// Number of independent/meaningfully distinct observation sources contributing to the track.
    pub source_count: u8,
}

impl ContactMotionObservation {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.range_m.is_finite() || self.range_m < 0.0 {
            return Err("range_m must be finite and non-negative");
        }
        if self.dcpa_m.is_some_and(|v| !v.is_finite() || v < 0.0) {
            return Err("dcpa_m must be finite and non-negative when present");
        }
        if self.tcpa_s.is_some_and(|v| !v.is_finite()) {
            return Err("tcpa_s must be finite when present");
        }
        Ok(())
    }
}

/// Operator/verified-policy thresholds for escalating a contact to COLREG review.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CollisionReviewPolicy {
    pub immediate_proximity_m: f32,
    pub minimum_dcpa_m: f32,
    pub lookahead_s: f32,
    pub minimum_source_count: u8,
}

impl CollisionReviewPolicy {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.immediate_proximity_m.is_finite() || self.immediate_proximity_m <= 0.0 {
            return Err("immediate_proximity_m must be finite and positive");
        }
        if !self.minimum_dcpa_m.is_finite() || self.minimum_dcpa_m <= 0.0 {
            return Err("minimum_dcpa_m must be finite and positive");
        }
        if !self.lookahead_s.is_finite() || self.lookahead_s <= 0.0 {
            return Err("lookahead_s must be finite and positive");
        }
        if self.minimum_source_count == 0 {
            return Err("minimum_source_count must be at least one");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollisionReviewDisposition {
    NoTrigger,
    Monitor,
    ColregReviewRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollisionReviewReason {
    InsufficientRelativeMotionEvidence,
    InsufficientSourceDiversity,
    ImmediateProximity,
    ProjectedCloseApproach,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollisionReviewAssessment {
    pub disposition: CollisionReviewDisposition,
    pub reasons: Vec<CollisionReviewReason>,
}

/// Evaluate whether a contact must be escalated to a COLREG/navigation policy layer.
///
/// The fail-closed behavior for missing DCPA/TCPA or insufficient source diversity is
/// intentional: this boundary refuses to infer safe passage from scanty relative-motion data.
pub fn assess_collision_review(
    observation: ContactMotionObservation,
    policy: CollisionReviewPolicy,
) -> Result<CollisionReviewAssessment, &'static str> {
    observation.validate()?;
    policy.validate()?;

    let mut reasons = Vec::new();
    if observation.source_count < policy.minimum_source_count {
        reasons.push(CollisionReviewReason::InsufficientSourceDiversity);
    }

    let (dcpa_m, tcpa_s) = match (observation.dcpa_m, observation.tcpa_s) {
        (Some(dcpa), Some(tcpa)) => (dcpa, tcpa),
        _ => {
            reasons.push(CollisionReviewReason::InsufficientRelativeMotionEvidence);
            return Ok(CollisionReviewAssessment {
                disposition: CollisionReviewDisposition::ColregReviewRequired,
                reasons,
            });
        }
    };

    if observation.range_m <= policy.immediate_proximity_m {
        reasons.push(CollisionReviewReason::ImmediateProximity);
    }
    if (0.0..=policy.lookahead_s).contains(&tcpa_s) && dcpa_m <= policy.minimum_dcpa_m {
        reasons.push(CollisionReviewReason::ProjectedCloseApproach);
    }

    if !reasons.is_empty() {
        return Ok(CollisionReviewAssessment {
            disposition: CollisionReviewDisposition::ColregReviewRequired,
            reasons,
        });
    }

    // A recently passed CPA or a predicted approach outside the configured safety boundary is
    // still worth monitoring, but this layer does not claim legal or navigational clearance.
    let disposition = if tcpa_s < 0.0 || tcpa_s <= policy.lookahead_s {
        CollisionReviewDisposition::Monitor
    } else {
        CollisionReviewDisposition::NoTrigger
    };
    Ok(CollisionReviewAssessment {
        disposition,
        reasons: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> CollisionReviewPolicy {
        CollisionReviewPolicy {
            immediate_proximity_m: 100.0,
            minimum_dcpa_m: 250.0,
            lookahead_s: 900.0,
            minimum_source_count: 2,
        }
    }

    #[test]
    fn scanty_motion_evidence_requires_review() {
        let assessment = assess_collision_review(
            ContactMotionObservation {
                range_m: 1_000.0,
                dcpa_m: None,
                tcpa_s: None,
                source_count: 1,
            },
            policy(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            CollisionReviewDisposition::ColregReviewRequired
        );
        assert!(assessment
            .reasons
            .contains(&CollisionReviewReason::InsufficientRelativeMotionEvidence));
        assert!(assessment
            .reasons
            .contains(&CollisionReviewReason::InsufficientSourceDiversity));
    }

    #[test]
    fn projected_close_approach_requires_review_without_choosing_manoeuvre() {
        let assessment = assess_collision_review(
            ContactMotionObservation {
                range_m: 1_500.0,
                dcpa_m: Some(80.0),
                tcpa_s: Some(300.0),
                source_count: 2,
            },
            policy(),
        )
        .unwrap();
        assert_eq!(
            assessment.disposition,
            CollisionReviewDisposition::ColregReviewRequired
        );
        assert_eq!(
            assessment.reasons,
            vec![CollisionReviewReason::ProjectedCloseApproach]
        );
    }

    #[test]
    fn distant_future_contact_does_not_create_a_manoeuvre_claim() {
        let assessment = assess_collision_review(
            ContactMotionObservation {
                range_m: 5_000.0,
                dcpa_m: Some(1_000.0),
                tcpa_s: Some(2_000.0),
                source_count: 3,
            },
            policy(),
        )
        .unwrap();
        assert_eq!(assessment.disposition, CollisionReviewDisposition::NoTrigger);
        assert!(assessment.reasons.is_empty());
    }
}
