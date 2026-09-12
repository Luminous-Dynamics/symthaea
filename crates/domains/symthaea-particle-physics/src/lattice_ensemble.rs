// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducible ensemble-run lineage for lattice gauge calculations.
//!
//! This module deliberately separates a predeclared run plan, the executed run
//! record, and downstream statistical qualification. It contains no gauge-field
//! solver and cannot make an ensemble scientifically trustworthy by itself.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatticeSamplerKind {
    CabibboMarinariMetropolis,
    CabibboMarinariHeatbath,
    HybridMonteCarlo,
    ExternalReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatticeBoundaryCondition {
    Periodic,
    Open,
    Twisted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngLineage {
    pub algorithm: String,
    pub implementation: String,
    pub version: String,
    pub stream_id: String,
    /// Commitment to seed material before the run, normally `sha256:<hex>`.
    pub seed_commitment: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnsemblePlan {
    pub ensemble_id: String,
    pub dims: [usize; 4],
    pub beta: f64,
    pub action_definition: String,
    pub boundary_conditions: [LatticeBoundaryCondition; 4],
    pub sampler: LatticeSamplerKind,
    /// Required for the current Cabibbo-Marinari Metropolis proposal family.
    pub proposal_max_angle: Option<f64>,
    /// Number of complete sweeps discarded before any scheduled measurement.
    pub thermalization_sweeps: u64,
    /// Complete sweeps between retained measurements.
    pub measurement_stride: u64,
    pub planned_measurements: usize,
    pub rng: RngLineage,
    pub code_revision: String,
    /// Canonical serialized run configuration digest.
    pub config_digest: String,
    /// Caller-supplied RFC3339 timestamp or equivalent immutable time label.
    pub planned_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnsembleRunRecord {
    pub plan: EnsemblePlan,
    pub completed_sweeps: u64,
    pub attempted_updates: u64,
    pub accepted_updates: u64,
    /// Actual retained measurement sweep numbers. A completed run must match the
    /// predeclared schedule exactly.
    pub measurement_sweeps: Vec<u64>,
    /// Digest of the immutable output ensemble/measurement artifact.
    pub output_artifact_digest: String,
    pub started_at: String,
    pub completed_at: String,
    /// Post-run digest of disclosed seed material. Keeping the seed hidden until
    /// completion can prevent accidental benchmark steering while preserving replay.
    pub seed_reveal_digest: Option<String>,
    /// Evidence that the action/update implementation matched its independent oracle.
    pub action_parity_evidence_id: String,
    /// Evidence for the sampler transition law / proposal theorem.
    pub sampler_evidence_id: String,
    /// Post-run autocorrelation / ESS / blocking analysis evidence.
    pub chain_statistics_evidence_id: Option<String>,
    /// Exact-head build/test qualification receipt when the run is scientific evidence.
    pub exact_head_ci_evidence_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleRecordError {
    EmptyField(&'static str),
    InvalidExtent([usize; 4]),
    InvalidBeta(f64),
    InvalidProposalMaxAngle(Option<f64>),
    InvalidMeasurementStride(u64),
    InvalidMeasurementCount(usize),
    InvalidSha256Digest { field: &'static str, value: String },
    MeasurementScheduleOverflow,
    MeasurementScheduleMismatch,
    IncompleteRun { required_sweep: u64, completed_sweep: u64 },
    AcceptedExceedsAttempted { accepted: u64, attempted: u64 },
    MissingScientificQualification(&'static str),
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), EnsembleRecordError> {
    if value.trim().is_empty() {
        Err(EnsembleRecordError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), EnsembleRecordError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(EnsembleRecordError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(EnsembleRecordError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

impl EnsemblePlan {
    pub fn validate(&self) -> Result<(), EnsembleRecordError> {
        require_nonempty(&self.ensemble_id, "ensemble_id")?;
        require_nonempty(&self.action_definition, "action_definition")?;
        require_nonempty(&self.rng.algorithm, "rng.algorithm")?;
        require_nonempty(&self.rng.implementation, "rng.implementation")?;
        require_nonempty(&self.rng.version, "rng.version")?;
        require_nonempty(&self.rng.stream_id, "rng.stream_id")?;
        require_nonempty(&self.code_revision, "code_revision")?;
        require_nonempty(&self.planned_at, "planned_at")?;
        require_sha256(&self.rng.seed_commitment, "rng.seed_commitment")?;
        require_sha256(&self.config_digest, "config_digest")?;

        if self.dims.iter().any(|&n| n == 0) {
            return Err(EnsembleRecordError::InvalidExtent(self.dims));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(EnsembleRecordError::InvalidBeta(self.beta));
        }
        if self.measurement_stride == 0 {
            return Err(EnsembleRecordError::InvalidMeasurementStride(0));
        }
        if self.planned_measurements == 0 {
            return Err(EnsembleRecordError::InvalidMeasurementCount(0));
        }
        if self.sampler == LatticeSamplerKind::CabibboMarinariMetropolis {
            match self.proposal_max_angle {
                Some(angle) if angle.is_finite() && angle > 0.0 && angle <= PI => {}
                value => return Err(EnsembleRecordError::InvalidProposalMaxAngle(value)),
            }
        }
        self.measurement_schedule()?;
        Ok(())
    }

    /// Predeclared retained-measurement sweeps.
    ///
    /// The first retained sample occurs one full `measurement_stride` after the
    /// end of the thermalization region; there is no measurement exactly at the
    /// burn-in boundary.
    pub fn measurement_schedule(&self) -> Result<Vec<u64>, EnsembleRecordError> {
        let mut out = Vec::with_capacity(self.planned_measurements);
        for i in 1..=self.planned_measurements {
            let offset = self
                .measurement_stride
                .checked_mul(i as u64)
                .ok_or(EnsembleRecordError::MeasurementScheduleOverflow)?;
            let sweep = self
                .thermalization_sweeps
                .checked_add(offset)
                .ok_or(EnsembleRecordError::MeasurementScheduleOverflow)?;
            out.push(sweep);
        }
        Ok(out)
    }
}

impl EnsembleRunRecord {
    pub fn validate_execution(&self) -> Result<(), EnsembleRecordError> {
        self.plan.validate()?;
        require_nonempty(&self.started_at, "started_at")?;
        require_nonempty(&self.completed_at, "completed_at")?;
        require_nonempty(&self.action_parity_evidence_id, "action_parity_evidence_id")?;
        require_nonempty(&self.sampler_evidence_id, "sampler_evidence_id")?;
        require_sha256(&self.output_artifact_digest, "output_artifact_digest")?;

        let expected = self.plan.measurement_schedule()?;
        if self.measurement_sweeps != expected {
            return Err(EnsembleRecordError::MeasurementScheduleMismatch);
        }
        let required_sweep = *expected
            .last()
            .ok_or(EnsembleRecordError::InvalidMeasurementCount(0))?;
        if self.completed_sweeps < required_sweep {
            return Err(EnsembleRecordError::IncompleteRun {
                required_sweep,
                completed_sweep: self.completed_sweeps,
            });
        }
        if self.accepted_updates > self.attempted_updates {
            return Err(EnsembleRecordError::AcceptedExceedsAttempted {
                accepted: self.accepted_updates,
                attempted: self.attempted_updates,
            });
        }
        if let Some(seed) = &self.seed_reveal_digest {
            require_sha256(seed, "seed_reveal_digest")?;
        }
        Ok(())
    }

    pub fn acceptance_rate(&self) -> Option<f64> {
        (self.attempted_updates > 0)
            .then_some(self.accepted_updates as f64 / self.attempted_updates as f64)
    }

    /// Stronger gate for using this run as scientific numerical evidence.
    ///
    /// This checks that critical evidence references are present; it does not
    /// evaluate whether their scientific conclusions are adequate.
    pub fn validate_for_scientific_use(&self) -> Result<(), EnsembleRecordError> {
        self.validate_execution()?;
        if self.sampler != LatticeSamplerKind::ExternalReference && self.attempted_updates == 0 {
            return Err(EnsembleRecordError::MissingScientificQualification(
                "nonzero sampler update count",
            ));
        }
        let seed = self
            .seed_reveal_digest
            .as_deref()
            .ok_or(EnsembleRecordError::MissingScientificQualification(
                "seed_reveal_digest",
            ))?;
        require_sha256(seed, "seed_reveal_digest")?;
        require_nonempty(
            self.chain_statistics_evidence_id.as_deref().ok_or(
                EnsembleRecordError::MissingScientificQualification(
                    "chain_statistics_evidence_id",
                ),
            )?,
            "chain_statistics_evidence_id",
        )?;
        require_nonempty(
            self.exact_head_ci_evidence_id.as_deref().ok_or(
                EnsembleRecordError::MissingScientificQualification("exact_head_ci_evidence_id"),
            )?,
            "exact_head_ci_evidence_id",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn plan() -> EnsemblePlan {
        EnsemblePlan {
            ensemble_id: "pure-su3-beta6-4x4x4x8".into(),
            dims: [4, 4, 4, 8],
            beta: 6.0,
            action_definition: "wilson_pure_gauge_v1".into(),
            boundary_conditions: [LatticeBoundaryCondition::Periodic; 4],
            sampler: LatticeSamplerKind::CabibboMarinariMetropolis,
            proposal_max_angle: Some(0.18),
            thermalization_sweeps: 1_000,
            measurement_stride: 100,
            planned_measurements: 4,
            rng: RngLineage {
                algorithm: "qualification-rng".into(),
                implementation: "external".into(),
                version: "v1".into(),
                stream_id: "stream-0".into(),
                seed_commitment: digest('a'),
            },
            code_revision: "0123456789abcdef0123456789abcdef01234567".into(),
            config_digest: digest('b'),
            planned_at: "2026-09-12T10:00:00Z".into(),
        }
    }

    #[test]
    fn measurement_schedule_is_unambiguous() {
        let p = plan();
        assert_eq!(p.measurement_schedule().unwrap(), vec![1100, 1200, 1300, 1400]);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn metropolis_plan_requires_proposal_width() {
        let mut p = plan();
        p.proposal_max_angle = None;
        assert!(matches!(
            p.validate(),
            Err(EnsembleRecordError::InvalidProposalMaxAngle(None))
        ));
    }

    #[test]
    fn execution_must_match_predeclared_measurement_schedule() {
        let p = plan();
        let mut record = EnsembleRunRecord {
            measurement_sweeps: p.measurement_schedule().unwrap(),
            plan: p,
            completed_sweeps: 1_400,
            attempted_updates: 10_000,
            accepted_updates: 7_000,
            output_artifact_digest: digest('c'),
            started_at: "2026-09-12T10:01:00Z".into(),
            completed_at: "2026-09-12T10:10:00Z".into(),
            seed_reveal_digest: Some(digest('d')),
            action_parity_evidence_id: "LQCD-011".into(),
            sampler_evidence_id: "LQCD-013".into(),
            chain_statistics_evidence_id: Some("LQCD-003:run-1".into()),
            exact_head_ci_evidence_id: Some("ci:run-1".into()),
        };
        assert!(record.validate_for_scientific_use().is_ok());
        record.measurement_sweeps[2] = 1_301;
        assert!(matches!(
            record.validate_execution(),
            Err(EnsembleRecordError::MeasurementScheduleMismatch)
        ));
    }

    #[test]
    fn execution_is_not_scientific_without_post_run_chain_evidence() {
        let p = plan();
        let record = EnsembleRunRecord {
            measurement_sweeps: p.measurement_schedule().unwrap(),
            plan: p,
            completed_sweeps: 1_400,
            attempted_updates: 10_000,
            accepted_updates: 7_000,
            output_artifact_digest: digest('c'),
            started_at: "2026-09-12T10:01:00Z".into(),
            completed_at: "2026-09-12T10:10:00Z".into(),
            seed_reveal_digest: Some(digest('d')),
            action_parity_evidence_id: "LQCD-011".into(),
            sampler_evidence_id: "LQCD-013".into(),
            chain_statistics_evidence_id: None,
            exact_head_ci_evidence_id: Some("ci:run-1".into()),
        };
        assert!(record.validate_execution().is_ok());
        assert!(matches!(
            record.validate_for_scientific_use(),
            Err(EnsembleRecordError::MissingScientificQualification(
                "chain_statistics_evidence_id"
            ))
        ));
    }

    #[test]
    fn malformed_digests_fail_closed() {
        let mut p = plan();
        p.config_digest = "sha256:not-a-digest".into();
        assert!(matches!(
            p.validate(),
            Err(EnsembleRecordError::InvalidSha256Digest { .. })
        ));
    }
}
