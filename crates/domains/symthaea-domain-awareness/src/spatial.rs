// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Time-bounded spatial safety volumes and fail-closed trajectory assessment.
//!
//! This module describes where an operation is authorized or where protected
//! space exists. It does not classify an intruder and does not grant actuation
//! authority.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Purpose of a spatial volume. None of these states imply hostile intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpatialVolumeKind {
    /// Space containing people/assets that should be protected from hazardous motion.
    ProtectedArea,
    /// Space in which the local system is authorized to operate.
    AuthorizedOperatingArea,
    /// Space local motion planning must avoid.
    KeepOutArea,
    /// Bounded authorized transit path.
    Corridor,
    /// Temporary safety/emergency area.
    EmergencyArea,
    /// Geometry is known but its semantic status is uncertain.
    UncertainArea,
}

/// Axis-aligned 3-D bounds in a declared coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisAlignedBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl AxisAlignedBounds {
    pub fn validate(&self) -> bool {
        self.min.iter().chain(self.max.iter()).all(|v| v.is_finite())
            && (0..3).all(|axis| self.min[axis] <= self.max[axis])
    }

    pub fn contains(&self, point: [f64; 3]) -> bool {
        self.validate()
            && point.iter().all(|v| v.is_finite())
            && (0..3).all(|axis| point[axis] >= self.min[axis] && point[axis] <= self.max[axis])
    }
}

/// A spatial rule with explicit temporal validity and authority provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialVolume {
    pub volume_id: Uuid,
    pub kind: SpatialVolumeKind,
    pub coordinate_frame: String,
    pub bounds: AxisAlignedBounds,
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    /// Identifier for the authority/source that declared the volume.
    pub authority_ref: String,
    pub evidence_refs: Vec<String>,
}

impl SpatialVolume {
    pub fn validate(&self) -> bool {
        !self.coordinate_frame.trim().is_empty()
            && self.bounds.validate()
            && self.valid_from_ms <= self.valid_until_ms
            && !self.authority_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|r| !r.trim().is_empty())
    }

    pub fn active_at(&self, timestamp_ms: u64) -> bool {
        self.validate() && (self.valid_from_ms..=self.valid_until_ms).contains(&timestamp_ms)
    }

    pub fn contains_at(&self, point: [f64; 3], timestamp_ms: u64) -> bool {
        self.active_at(timestamp_ms) && self.bounds.contains(point)
    }
}

/// One predicted or observed trajectory sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySample {
    pub timestamp_ms: u64,
    pub position: [f64; 3],
}

impl TrajectorySample {
    pub fn validate(&self) -> bool {
        self.position.iter().all(|v| v.is_finite())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialSafetyStatus {
    Safe,
    Restricted,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialSafetyReason {
    Safe,
    InvalidVolumeEvidence,
    InvalidTrajectory,
    MissingAuthorizedVolume,
    OutsideAuthorizedVolume,
    KeepOutIntersection,
    ProtectedVolumeIncursion,
    CoordinateFrameMismatch,
}

/// Assessment is advisory evidence for an independent safety boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialSafetyAssessment {
    pub status: SpatialSafetyStatus,
    pub reason: SpatialSafetyReason,
    pub offending_sample_index: Option<usize>,
    pub volume_id: Option<Uuid>,
}

impl SpatialSafetyAssessment {
    pub const fn safe() -> Self {
        Self {
            status: SpatialSafetyStatus::Safe,
            reason: SpatialSafetyReason::Safe,
            offending_sample_index: None,
            volume_id: None,
        }
    }
}

/// Pure spatial safety evaluator. It produces no actuator commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialSafetyPolicy {
    pub require_authorized_operating_volume: bool,
    /// Treat entry into a ProtectedArea as a restricted condition. This says
    /// nothing about the identity or intent of the moving object.
    pub restrict_protected_volume_incursion: bool,
}

impl Default for SpatialSafetyPolicy {
    fn default() -> Self {
        Self {
            require_authorized_operating_volume: true,
            restrict_protected_volume_incursion: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialSafetyKernel {
    policy: SpatialSafetyPolicy,
}

impl SpatialSafetyKernel {
    pub const fn new(policy: SpatialSafetyPolicy) -> Self {
        Self { policy }
    }

    pub fn assess(
        &self,
        coordinate_frame: &str,
        trajectory: &[TrajectorySample],
        volumes: &[SpatialVolume],
    ) -> SpatialSafetyAssessment {
        if coordinate_frame.trim().is_empty()
            || trajectory.is_empty()
            || trajectory.iter().any(|sample| !sample.validate())
        {
            return SpatialSafetyAssessment {
                status: SpatialSafetyStatus::Incomplete,
                reason: SpatialSafetyReason::InvalidTrajectory,
                offending_sample_index: None,
                volume_id: None,
            };
        }
        if volumes.iter().any(|volume| !volume.validate()) {
            return SpatialSafetyAssessment {
                status: SpatialSafetyStatus::Incomplete,
                reason: SpatialSafetyReason::InvalidVolumeEvidence,
                offending_sample_index: None,
                volume_id: None,
            };
        }
        if volumes
            .iter()
            .any(|volume| volume.coordinate_frame != coordinate_frame)
        {
            return SpatialSafetyAssessment {
                status: SpatialSafetyStatus::Incomplete,
                reason: SpatialSafetyReason::CoordinateFrameMismatch,
                offending_sample_index: None,
                volume_id: None,
            };
        }

        for (index, sample) in trajectory.iter().enumerate() {
            for volume in volumes.iter().filter(|volume| {
                volume.kind == SpatialVolumeKind::KeepOutArea && volume.active_at(sample.timestamp_ms)
            }) {
                if volume.bounds.contains(sample.position) {
                    return SpatialSafetyAssessment {
                        status: SpatialSafetyStatus::Restricted,
                        reason: SpatialSafetyReason::KeepOutIntersection,
                        offending_sample_index: Some(index),
                        volume_id: Some(volume.volume_id),
                    };
                }
            }

            if self.policy.restrict_protected_volume_incursion {
                for volume in volumes.iter().filter(|volume| {
                    volume.kind == SpatialVolumeKind::ProtectedArea
                        && volume.active_at(sample.timestamp_ms)
                }) {
                    if volume.bounds.contains(sample.position) {
                        return SpatialSafetyAssessment {
                            status: SpatialSafetyStatus::Restricted,
                            reason: SpatialSafetyReason::ProtectedVolumeIncursion,
                            offending_sample_index: Some(index),
                            volume_id: Some(volume.volume_id),
                        };
                    }
                }
            }

            if self.policy.require_authorized_operating_volume {
                let authorized = volumes.iter().any(|volume| {
                    matches!(
                        volume.kind,
                        SpatialVolumeKind::AuthorizedOperatingArea
                            | SpatialVolumeKind::Corridor
                            | SpatialVolumeKind::EmergencyArea
                    ) && volume.contains_at(sample.position, sample.timestamp_ms)
                });
                if !authorized {
                    let any_active_authorized = volumes.iter().any(|volume| {
                        matches!(
                            volume.kind,
                            SpatialVolumeKind::AuthorizedOperatingArea
                                | SpatialVolumeKind::Corridor
                                | SpatialVolumeKind::EmergencyArea
                        ) && volume.active_at(sample.timestamp_ms)
                    });
                    return SpatialSafetyAssessment {
                        status: SpatialSafetyStatus::Restricted,
                        reason: if any_active_authorized {
                            SpatialSafetyReason::OutsideAuthorizedVolume
                        } else {
                            SpatialSafetyReason::MissingAuthorizedVolume
                        },
                        offending_sample_index: Some(index),
                        volume_id: None,
                    };
                }
            }
        }

        SpatialSafetyAssessment::safe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(kind: SpatialVolumeKind, min: [f64; 3], max: [f64; 3]) -> SpatialVolume {
        SpatialVolume {
            volume_id: Uuid::new_v4(),
            kind,
            coordinate_frame: "local-enu".to_string(),
            bounds: AxisAlignedBounds { min, max },
            valid_from_ms: 0,
            valid_until_ms: 10_000,
            authority_ref: "authority:test".to_string(),
            evidence_refs: vec!["evidence:volume".to_string()],
        }
    }

    #[test]
    fn trajectory_inside_authorized_volume_is_safe() {
        let authorized = volume(
            SpatialVolumeKind::AuthorizedOperatingArea,
            [-100.0, -100.0, 0.0],
            [100.0, 100.0, 100.0],
        );
        let samples = [TrajectorySample {
            timestamp_ms: 1_000,
            position: [0.0, 0.0, 20.0],
        }];
        let result = SpatialSafetyKernel::default().assess("local-enu", &samples, &[authorized]);
        assert_eq!(result.status, SpatialSafetyStatus::Safe);
    }

    #[test]
    fn missing_authorized_volume_fails_restricted() {
        let samples = [TrajectorySample {
            timestamp_ms: 1_000,
            position: [0.0, 0.0, 20.0],
        }];
        let result = SpatialSafetyKernel::default().assess("local-enu", &samples, &[]);
        assert_eq!(result.status, SpatialSafetyStatus::Restricted);
        assert_eq!(result.reason, SpatialSafetyReason::MissingAuthorizedVolume);
    }

    #[test]
    fn keep_out_intersection_restricts_without_classifying_intent() {
        let authorized = volume(
            SpatialVolumeKind::AuthorizedOperatingArea,
            [-100.0, -100.0, 0.0],
            [100.0, 100.0, 100.0],
        );
        let keep_out = volume(
            SpatialVolumeKind::KeepOutArea,
            [-5.0, -5.0, 0.0],
            [5.0, 5.0, 50.0],
        );
        let samples = [TrajectorySample {
            timestamp_ms: 1_000,
            position: [0.0, 0.0, 20.0],
        }];
        let result = SpatialSafetyKernel::default().assess(
            "local-enu",
            &samples,
            &[authorized, keep_out],
        );
        assert_eq!(result.status, SpatialSafetyStatus::Restricted);
        assert_eq!(result.reason, SpatialSafetyReason::KeepOutIntersection);
    }

    #[test]
    fn invalid_volume_evidence_is_incomplete_not_ignored() {
        let mut invalid = volume(
            SpatialVolumeKind::AuthorizedOperatingArea,
            [-100.0, -100.0, 0.0],
            [100.0, 100.0, 100.0],
        );
        invalid.authority_ref.clear();
        let samples = [TrajectorySample {
            timestamp_ms: 1_000,
            position: [0.0, 0.0, 20.0],
        }];
        let result = SpatialSafetyKernel::default().assess("local-enu", &samples, &[invalid]);
        assert_eq!(result.status, SpatialSafetyStatus::Incomplete);
    }
}
