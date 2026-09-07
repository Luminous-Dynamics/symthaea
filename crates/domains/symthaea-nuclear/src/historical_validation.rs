// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time-separated nuclear-mass validation over explicitly frozen data vintages.
//!
//! The current in-tree AME2020 corpus does not carry per-nucleus measurement
//! dates or earlier AME release membership. Deriving a historical benchmark from
//! that one table would therefore invent chronology. This module instead accepts
//! two explicit frozen mass-table snapshots and constructs a holdout only from
//! nuclei that are measured in the later snapshot but were not measured in the
//! earlier one.
//!
//! Freezing data alone is not enough. A model can still leak future knowledge
//! through a later physical prior, feature table, implementation change, or model
//! selection decision. Every scored model therefore carries a declared knowledge
//! cutoff plus immutable implementation/dependency evidence IDs, and that cutoff
//! must not postdate the earlier mass-table release.
//!
//! These local contracts still cannot prove external publication timestamps or
//! prove that the running code is byte-identical to an archived implementation.
//! Results are therefore explicitly `RetrospectiveDataVintageBacktest` with
//! `DeclaredChronologyOnly`, never a claim of prospective prediction.

use crate::discovery::MeasuredNucleus;
use crate::duflo_zuker::dz_binding_energy;
use crate::ml_mass::{MlMassConfig, MlMassFitError, MlMassPredictor};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotAuthority {
    /// Content is expected to correspond to an externally retained artifact.
    ExternalArtifactSha256 { sha256_hex: String },
    /// Test/fixture data. This can exercise mechanics but is not historical evidence.
    SyntheticFixture,
}

impl SnapshotAuthority {
    fn validate(&self) -> Result<(), HistoricalValidationError> {
        match self {
            Self::ExternalArtifactSha256 { sha256_hex }
                if sha256_hex.len() == 64
                    && sha256_hex.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
            {
                Ok(())
            }
            Self::ExternalArtifactSha256 { .. } => {
                Err(HistoricalValidationError::InvalidSnapshotSha256)
            }
            Self::SyntheticFixture => Ok(()),
        }
    }

    fn is_external(&self) -> bool {
        matches!(self, Self::ExternalArtifactSha256 { .. })
    }

    fn same_external_artifact(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::ExternalArtifactSha256 { sha256_hex: left },
                Self::ExternalArtifactSha256 { sha256_hex: right },
            ) => left.eq_ignore_ascii_case(right),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HistoricalMassDatum {
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev_bits: u64,
    pub is_measured: bool,
}

impl HistoricalMassDatum {
    pub fn binding_energy_mev(self) -> f64 {
        f64::from_bits(self.binding_energy_mev_bits)
    }

    fn to_nucleus(self) -> MeasuredNucleus {
        MeasuredNucleus {
            z: self.z,
            n: self.n,
            binding_energy_mev: self.binding_energy_mev(),
            is_measured: self.is_measured,
        }
    }
}

impl From<&MeasuredNucleus> for HistoricalMassDatum {
    fn from(nucleus: &MeasuredNucleus) -> Self {
        Self {
            z: nucleus.z,
            n: nucleus.n,
            binding_energy_mev_bits: nucleus.binding_energy_mev.to_bits(),
            is_measured: nucleus.is_measured,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalSnapshotMetadata {
    pub source_id: String,
    /// Calendar date encoded as YYYYMMDD.
    pub release_yyyymmdd: u32,
    /// External immutable/timestamped record intended to support the declared date.
    pub chronology_evidence_id: String,
    pub authority: SnapshotAuthority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalMassSnapshot {
    metadata: HistoricalSnapshotMetadata,
    data: Vec<HistoricalMassDatum>,
}

impl HistoricalMassSnapshot {
    pub fn new(
        source_id: impl Into<String>,
        release_yyyymmdd: u32,
        chronology_evidence_id: impl Into<String>,
        authority: SnapshotAuthority,
        nuclei: &[MeasuredNucleus],
    ) -> Result<Self, HistoricalValidationError> {
        let mut snapshot = Self {
            metadata: HistoricalSnapshotMetadata {
                source_id: source_id.into(),
                release_yyyymmdd,
                chronology_evidence_id: chronology_evidence_id.into(),
                authority,
            },
            data: nuclei.iter().map(HistoricalMassDatum::from).collect(),
        };
        snapshot.data.sort_unstable();
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn metadata(&self) -> &HistoricalSnapshotMetadata {
        &self.metadata
    }

    pub fn data(&self) -> &[HistoricalMassDatum] {
        &self.data
    }

    pub fn validate(&self) -> Result<(), HistoricalValidationError> {
        if self.metadata.source_id.trim().is_empty() {
            return Err(HistoricalValidationError::EmptySourceId);
        }
        if self.metadata.chronology_evidence_id.trim().is_empty() {
            return Err(HistoricalValidationError::EmptyChronologyEvidenceId);
        }
        if !valid_yyyymmdd(self.metadata.release_yyyymmdd) {
            return Err(HistoricalValidationError::InvalidReleaseDate);
        }
        self.metadata.authority.validate()?;
        if self.data.is_empty() {
            return Err(HistoricalValidationError::EmptySnapshot);
        }

        let mut seen = BTreeSet::new();
        for datum in &self.data {
            if datum.z == 0 && datum.n == 0 {
                return Err(HistoricalValidationError::InvalidCoordinate {
                    z: datum.z,
                    n: datum.n,
                });
            }
            if !datum.binding_energy_mev().is_finite() {
                return Err(HistoricalValidationError::NonFiniteBindingEnergy {
                    z: datum.z,
                    n: datum.n,
                });
            }
            if !seen.insert((datum.z, datum.n)) {
                return Err(HistoricalValidationError::DuplicateNucleus {
                    z: datum.z,
                    n: datum.n,
                });
            }
        }
        Ok(())
    }
}

fn valid_yyyymmdd(value: u32) -> bool {
    let year = value / 10_000;
    let month = (value / 100) % 100;
    let day = value % 100;
    if year < 1900 || !(1..=12).contains(&month) || day == 0 {
        return false;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    day <= max_day
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalModelKnowledge {
    pub model_id: String,
    /// Latest date on which any declared implementation choice, physical prior,
    /// feature table, calibration constant, or model dependency became available.
    pub latest_dependency_yyyymmdd: u32,
    /// Immutable external record for the implementation intended by this model id.
    pub frozen_implementation_evidence_id: String,
    /// Immutable external records for physical/model dependencies used by the run.
    pub dependency_evidence_ids: Vec<String>,
}

impl HistoricalModelKnowledge {
    pub fn new(
        model_id: impl Into<String>,
        latest_dependency_yyyymmdd: u32,
        frozen_implementation_evidence_id: impl Into<String>,
        dependency_evidence_ids: Vec<String>,
    ) -> Result<Self, HistoricalValidationError> {
        let knowledge = Self {
            model_id: model_id.into(),
            latest_dependency_yyyymmdd,
            frozen_implementation_evidence_id: frozen_implementation_evidence_id.into(),
            dependency_evidence_ids,
        };
        knowledge.validate()?;
        Ok(knowledge)
    }

    fn validate(&self) -> Result<(), HistoricalValidationError> {
        if self.model_id.trim().is_empty() {
            return Err(HistoricalValidationError::EmptyModelId);
        }
        if !valid_yyyymmdd(self.latest_dependency_yyyymmdd) {
            return Err(HistoricalValidationError::InvalidModelKnowledgeDate);
        }
        if self.frozen_implementation_evidence_id.trim().is_empty() {
            return Err(HistoricalValidationError::EmptyImplementationEvidenceId);
        }
        if self.dependency_evidence_ids.is_empty()
            || self
                .dependency_evidence_ids
                .iter()
                .any(|value| value.trim().is_empty())
        {
            return Err(HistoricalValidationError::EmptyDependencyEvidenceId);
        }
        let unique: BTreeSet<_> = self.dependency_evidence_ids.iter().collect();
        if unique.len() != self.dependency_evidence_ids.len() {
            return Err(HistoricalValidationError::DuplicateDependencyEvidenceId);
        }
        Ok(())
    }

    fn validate_for_training_vintage(
        &self,
        earlier_release_yyyymmdd: u32,
    ) -> Result<(), HistoricalValidationError> {
        self.validate()?;
        if self.latest_dependency_yyyymmdd > earlier_release_yyyymmdd {
            return Err(HistoricalValidationError::ModelKnowledgePostdatesTrainingSnapshot {
                latest_dependency_yyyymmdd: self.latest_dependency_yyyymmdd,
                training_release_yyyymmdd: earlier_release_yyyymmdd,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoricalEvidenceScope {
    /// Both snapshots identify externally retained artifacts, but chronology is
    /// still only declared until the evidence IDs are verified outside this crate.
    ExternalArtifactsDeclaredChronology,
    /// At least one snapshot is a synthetic fixture.
    SyntheticFixtureOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoricalChronologyInterpretation {
    DeclaredChronologyOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoricalBenchmarkInterpretation {
    /// Current code is retrospectively evaluated on an earlier data vintage. This
    /// is stronger than random CV but is not itself proof of a prospective forecast.
    RetrospectiveDataVintageBacktest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalRevision {
    pub z: u16,
    pub n: u16,
    pub earlier_binding_energy_mev_bits: u64,
    pub later_binding_energy_mev_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalBenchmarkReceipt {
    pub earlier: HistoricalSnapshotMetadata,
    pub later: HistoricalSnapshotMetadata,
    pub training: Vec<HistoricalMassDatum>,
    pub holdout_newly_measured: Vec<HistoricalMassDatum>,
    pub revised_previously_measured: Vec<HistoricalRevision>,
    pub evidence_scope: HistoricalEvidenceScope,
    pub chronology_interpretation: HistoricalChronologyInterpretation,
    pub benchmark_interpretation: HistoricalBenchmarkInterpretation,
}

pub struct HistoricalBenchmarkSplit {
    training: Vec<MeasuredNucleus>,
    holdout: Vec<MeasuredNucleus>,
    receipt: HistoricalBenchmarkReceipt,
}

impl HistoricalBenchmarkSplit {
    pub fn from_snapshots(
        earlier: &HistoricalMassSnapshot,
        later: &HistoricalMassSnapshot,
    ) -> Result<Self, HistoricalValidationError> {
        earlier.validate()?;
        later.validate()?;
        if earlier.metadata.release_yyyymmdd >= later.metadata.release_yyyymmdd {
            return Err(HistoricalValidationError::NonIncreasingSnapshotChronology);
        }
        if earlier
            .metadata
            .authority
            .same_external_artifact(&later.metadata.authority)
        {
            return Err(HistoricalValidationError::IndistinguishableSnapshots);
        }

        let earlier_by_coord: BTreeMap<(u16, u16), HistoricalMassDatum> = earlier
            .data
            .iter()
            .copied()
            .map(|datum| ((datum.z, datum.n), datum))
            .collect();

        let mut training = earlier
            .data
            .iter()
            .copied()
            .filter(|datum| datum.is_measured)
            .map(HistoricalMassDatum::to_nucleus)
            .collect::<Vec<_>>();
        training.sort_by_key(|nucleus| (nucleus.z, nucleus.n));
        if training.is_empty() {
            return Err(HistoricalValidationError::EmptyHistoricalTrainingSet);
        }

        let mut holdout = Vec::new();
        let mut revisions = Vec::new();
        for later_datum in later.data.iter().copied().filter(|datum| datum.is_measured) {
            match earlier_by_coord.get(&(later_datum.z, later_datum.n)).copied() {
                Some(earlier_datum) if earlier_datum.is_measured => {
                    if earlier_datum.binding_energy_mev_bits
                        != later_datum.binding_energy_mev_bits
                    {
                        revisions.push(HistoricalRevision {
                            z: later_datum.z,
                            n: later_datum.n,
                            earlier_binding_energy_mev_bits: earlier_datum.binding_energy_mev_bits,
                            later_binding_energy_mev_bits: later_datum.binding_energy_mev_bits,
                        });
                    }
                }
                Some(_) | None => holdout.push(later_datum.to_nucleus()),
            }
        }
        holdout.sort_by_key(|nucleus| (nucleus.z, nucleus.n));
        revisions.sort_by_key(|revision| (revision.z, revision.n));
        if holdout.is_empty() {
            return Err(HistoricalValidationError::EmptyHistoricalHoldoutSet);
        }

        let training_coords: BTreeSet<_> = training
            .iter()
            .map(|nucleus| (nucleus.z, nucleus.n))
            .collect();
        if holdout
            .iter()
            .any(|nucleus| training_coords.contains(&(nucleus.z, nucleus.n)))
        {
            return Err(HistoricalValidationError::HistoricalTrainHoldoutOverlap);
        }

        let evidence_scope = if earlier.metadata.authority.is_external()
            && later.metadata.authority.is_external()
        {
            HistoricalEvidenceScope::ExternalArtifactsDeclaredChronology
        } else {
            HistoricalEvidenceScope::SyntheticFixtureOnly
        };
        let receipt = HistoricalBenchmarkReceipt {
            earlier: earlier.metadata.clone(),
            later: later.metadata.clone(),
            training: training.iter().map(HistoricalMassDatum::from).collect(),
            holdout_newly_measured: holdout.iter().map(HistoricalMassDatum::from).collect(),
            revised_previously_measured: revisions,
            evidence_scope,
            chronology_interpretation: HistoricalChronologyInterpretation::DeclaredChronologyOnly,
            benchmark_interpretation:
                HistoricalBenchmarkInterpretation::RetrospectiveDataVintageBacktest,
        };

        Ok(Self {
            training,
            holdout,
            receipt,
        })
    }

    pub fn training(&self) -> &[MeasuredNucleus] {
        &self.training
    }

    pub fn holdout(&self) -> &[MeasuredNucleus] {
        &self.holdout
    }

    pub fn receipt(&self) -> &HistoricalBenchmarkReceipt {
        &self.receipt
    }

    pub fn evaluate_declared_model<F>(
        &self,
        knowledge: &HistoricalModelKnowledge,
        mut predict_binding_energy: F,
    ) -> Result<HistoricalValidationReport, HistoricalValidationError>
    where
        F: FnMut(u16, u16) -> f64,
    {
        knowledge.validate_for_training_vintage(self.receipt.earlier.release_yyyymmdd)?;

        let mut errors = Vec::with_capacity(self.holdout.len());
        for nucleus in &self.holdout {
            let predicted = predict_binding_energy(nucleus.z, nucleus.n);
            if !predicted.is_finite() {
                return Err(HistoricalValidationError::NonFinitePrediction {
                    z: nucleus.z,
                    n: nucleus.n,
                });
            }
            errors.push(predicted - nucleus.binding_energy_mev);
        }
        let n = errors.len() as f64;
        let bias_mev = errors.iter().sum::<f64>() / n;
        let mae_mev = errors.iter().map(|error| error.abs()).sum::<f64>() / n;
        let rms_mev = (errors.iter().map(|error| error * error).sum::<f64>() / n).sqrt();
        let max_abs_error_mev = errors
            .iter()
            .map(|error| error.abs())
            .fold(0.0, f64::max);

        Ok(HistoricalValidationReport {
            method: knowledge.model_id.clone(),
            model_knowledge: knowledge.clone(),
            earlier_release_yyyymmdd: self.receipt.earlier.release_yyyymmdd,
            later_release_yyyymmdd: self.receipt.later.release_yyyymmdd,
            n_training: self.training.len(),
            n_newly_measured_holdout: self.holdout.len(),
            n_revised_previously_measured: self.receipt.revised_previously_measured.len(),
            bias_mev,
            mae_mev,
            rms_mev,
            max_abs_error_mev,
            evidence_scope: self.receipt.evidence_scope,
            chronology_interpretation: self.receipt.chronology_interpretation,
            benchmark_interpretation: self.receipt.benchmark_interpretation,
            chronology_note: "snapshot/model dates and evidence IDs are externally supplied declarations; this crate does not independently prove publication chronology or that the running implementation is identical to the archived implementation".to_string(),
        })
    }

    pub fn evaluate_dz10(
        &self,
        knowledge: &HistoricalModelKnowledge,
    ) -> Result<HistoricalValidationReport, HistoricalValidationError> {
        require_model_id(knowledge, "DZ10")?;
        self.evaluate_declared_model(knowledge, dz_binding_energy)
    }

    pub fn evaluate_rf(
        &self,
        config: MlMassConfig,
        knowledge: &HistoricalModelKnowledge,
    ) -> Result<HistoricalValidationReport, HistoricalValidationError> {
        require_model_id(knowledge, "DZ10+RF-historical-fit")?;
        knowledge.validate_for_training_vintage(self.receipt.earlier.release_yyyymmdd)?;
        let predictor = MlMassPredictor::fit_measured_with_config(&self.training, config)
            .map_err(HistoricalValidationError::MlMassFit)?;
        self.evaluate_declared_model(knowledge, |z, n| predictor.predict(z, n).binding_energy)
    }

    /// Same-target comparison only. There is deliberately no winner/ranking field.
    pub fn compare_dz10_and_rf(
        &self,
        config: MlMassConfig,
        dz10_knowledge: &HistoricalModelKnowledge,
        rf_knowledge: &HistoricalModelKnowledge,
    ) -> Result<HistoricalModelComparison, HistoricalValidationError> {
        Ok(HistoricalModelComparison {
            receipt: self.receipt.clone(),
            entries: vec![
                self.evaluate_dz10(dz10_knowledge)?,
                self.evaluate_rf(config, rf_knowledge)?,
            ],
        })
    }
}

fn require_model_id(
    knowledge: &HistoricalModelKnowledge,
    expected: &'static str,
) -> Result<(), HistoricalValidationError> {
    if knowledge.model_id != expected {
        return Err(HistoricalValidationError::ModelIdMismatch {
            expected,
            actual: knowledge.model_id.clone(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoricalValidationReport {
    pub method: String,
    pub model_knowledge: HistoricalModelKnowledge,
    pub earlier_release_yyyymmdd: u32,
    pub later_release_yyyymmdd: u32,
    pub n_training: usize,
    pub n_newly_measured_holdout: usize,
    pub n_revised_previously_measured: usize,
    pub bias_mev: f64,
    pub mae_mev: f64,
    pub rms_mev: f64,
    pub max_abs_error_mev: f64,
    pub evidence_scope: HistoricalEvidenceScope,
    pub chronology_interpretation: HistoricalChronologyInterpretation,
    pub benchmark_interpretation: HistoricalBenchmarkInterpretation,
    pub chronology_note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoricalModelComparison {
    pub receipt: HistoricalBenchmarkReceipt,
    pub entries: Vec<HistoricalValidationReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HistoricalValidationError {
    EmptySourceId,
    EmptyChronologyEvidenceId,
    InvalidSnapshotSha256,
    InvalidReleaseDate,
    EmptySnapshot,
    DuplicateNucleus { z: u16, n: u16 },
    InvalidCoordinate { z: u16, n: u16 },
    NonFiniteBindingEnergy { z: u16, n: u16 },
    NonIncreasingSnapshotChronology,
    IndistinguishableSnapshots,
    EmptyHistoricalTrainingSet,
    EmptyHistoricalHoldoutSet,
    HistoricalTrainHoldoutOverlap,
    EmptyModelId,
    InvalidModelKnowledgeDate,
    EmptyImplementationEvidenceId,
    EmptyDependencyEvidenceId,
    DuplicateDependencyEvidenceId,
    ModelKnowledgePostdatesTrainingSnapshot {
        latest_dependency_yyyymmdd: u32,
        training_release_yyyymmdd: u32,
    },
    ModelIdMismatch {
        expected: &'static str,
        actual: String,
    },
    NonFinitePrediction { z: u16, n: u16 },
    MlMassFit(MlMassFitError),
}

impl fmt::Display for HistoricalValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySourceId => write!(f, "historical snapshot source id must not be empty"),
            Self::EmptyChronologyEvidenceId => write!(
                f,
                "historical snapshot chronology evidence id must not be empty"
            ),
            Self::InvalidSnapshotSha256 => {
                write!(f, "historical snapshot SHA-256 must be exactly 64 hex digits")
            }
            Self::InvalidReleaseDate => write!(f, "invalid historical snapshot YYYYMMDD date"),
            Self::EmptySnapshot => write!(f, "historical mass snapshot is empty"),
            Self::DuplicateNucleus { z, n } => {
                write!(f, "duplicate historical snapshot nucleus at Z={z}, N={n}")
            }
            Self::InvalidCoordinate { z, n } => {
                write!(f, "invalid historical snapshot coordinate Z={z}, N={n}")
            }
            Self::NonFiniteBindingEnergy { z, n } => write!(
                f,
                "non-finite historical snapshot binding energy at Z={z}, N={n}"
            ),
            Self::NonIncreasingSnapshotChronology => write!(
                f,
                "later historical snapshot must declare a strictly later release date"
            ),
            Self::IndistinguishableSnapshots => write!(
                f,
                "historical snapshots identify the same external artifact"
            ),
            Self::EmptyHistoricalTrainingSet => {
                write!(f, "earlier snapshot contains no measured training nuclei")
            }
            Self::EmptyHistoricalHoldoutSet => write!(
                f,
                "later snapshot contains no newly measured nuclei for historical adjudication"
            ),
            Self::HistoricalTrainHoldoutOverlap => write!(
                f,
                "historical benchmark placed one coordinate in both training and holdout"
            ),
            Self::EmptyModelId => write!(f, "historical benchmark model id must not be empty"),
            Self::InvalidModelKnowledgeDate => write!(f, "invalid historical model knowledge date"),
            Self::EmptyImplementationEvidenceId => write!(
                f,
                "historical model implementation evidence id must not be empty"
            ),
            Self::EmptyDependencyEvidenceId => write!(
                f,
                "historical model dependency evidence ids must be non-empty"
            ),
            Self::DuplicateDependencyEvidenceId => write!(
                f,
                "historical model dependency evidence ids must be unique"
            ),
            Self::ModelKnowledgePostdatesTrainingSnapshot {
                latest_dependency_yyyymmdd,
                training_release_yyyymmdd,
            } => write!(
                f,
                "model knowledge cutoff {latest_dependency_yyyymmdd} postdates training snapshot {training_release_yyyymmdd}"
            ),
            Self::ModelIdMismatch { expected, actual } => write!(
                f,
                "historical model id mismatch: expected {expected}, received {actual}"
            ),
            Self::NonFinitePrediction { z, n } => write!(
                f,
                "historical benchmark predictor returned non-finite energy at Z={z}, N={n}"
            ),
            Self::MlMassFit(error) => write!(f, "historical RF fit failed: {error}"),
        }
    }
}

impl std::error::Error for HistoricalValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn nucleus(z: u16, n: u16, binding_energy_mev: f64, is_measured: bool) -> MeasuredNucleus {
        MeasuredNucleus {
            z,
            n,
            binding_energy_mev,
            is_measured,
        }
    }

    fn earlier_fixture() -> HistoricalMassSnapshot {
        HistoricalMassSnapshot::new(
            "fixture-old",
            20160101,
            "fixture:old-date",
            SnapshotAuthority::SyntheticFixture,
            &[
                nucleus(6, 6, 92.1, true),
                nucleus(8, 8, 127.6, true),
                nucleus(10, 10, 160.6, true),
                nucleus(12, 12, 198.3, true),
                nucleus(8, 9, 131.7, false),
            ],
        )
        .unwrap()
    }

    fn later_fixture() -> HistoricalMassSnapshot {
        HistoricalMassSnapshot::new(
            "fixture-new",
            20200101,
            "fixture:new-date",
            SnapshotAuthority::SyntheticFixture,
            &[
                nucleus(6, 6, 92.1, true),
                nucleus(8, 8, 127.7, true), // revision, not a historical holdout
                nucleus(10, 10, 160.6, true),
                nucleus(12, 12, 198.3, true),
                nucleus(8, 9, 131.8, true), // estimated -> measured
                nucleus(14, 14, 236.5, true), // absent -> measured
            ],
        )
        .unwrap()
    }

    fn dz_knowledge() -> HistoricalModelKnowledge {
        HistoricalModelKnowledge::new(
            "DZ10",
            20150101,
            "fixture:dz-implementation",
            vec!["fixture:dz-reference".to_string()],
        )
        .unwrap()
    }

    fn rf_knowledge() -> HistoricalModelKnowledge {
        HistoricalModelKnowledge::new(
            "DZ10+RF-historical-fit",
            20150101,
            "fixture:rf-implementation",
            vec![
                "fixture:dz-reference".to_string(),
                "fixture:deformation-reference".to_string(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn historical_holdout_contains_only_newly_measured_coordinates() {
        let split = HistoricalBenchmarkSplit::from_snapshots(&earlier_fixture(), &later_fixture())
            .unwrap();
        assert_eq!(split.training().len(), 4);
        assert_eq!(split.holdout().len(), 2);
        assert!(split.holdout().iter().any(|nucleus| (nucleus.z, nucleus.n) == (8, 9)));
        assert!(split.holdout().iter().any(|nucleus| (nucleus.z, nucleus.n) == (14, 14)));
        assert_eq!(split.receipt().revised_previously_measured.len(), 1);
        assert_eq!(
            split.receipt().evidence_scope,
            HistoricalEvidenceScope::SyntheticFixtureOnly
        );
        assert_eq!(
            split.receipt().chronology_interpretation,
            HistoricalChronologyInterpretation::DeclaredChronologyOnly
        );
        assert_eq!(
            split.receipt().benchmark_interpretation,
            HistoricalBenchmarkInterpretation::RetrospectiveDataVintageBacktest
        );
    }

    #[test]
    fn historical_rf_fit_uses_only_earlier_measured_snapshot() {
        let split = HistoricalBenchmarkSplit::from_snapshots(&earlier_fixture(), &later_fixture())
            .unwrap();
        let comparison = split
            .compare_dz10_and_rf(
                MlMassConfig {
                    n_trees: 4,
                    max_depth: 3,
                    min_samples: 1,
                    seed: 17,
                },
                &dz_knowledge(),
                &rf_knowledge(),
            )
            .unwrap();
        assert_eq!(comparison.entries.len(), 2);
        for entry in &comparison.entries {
            assert_eq!(entry.n_training, 4);
            assert_eq!(entry.n_newly_measured_holdout, 2);
            assert!(entry.rms_mev.is_finite());
            assert_eq!(
                entry.chronology_interpretation,
                HistoricalChronologyInterpretation::DeclaredChronologyOnly
            );
            assert_eq!(
                entry.benchmark_interpretation,
                HistoricalBenchmarkInterpretation::RetrospectiveDataVintageBacktest
            );
        }
    }

    #[test]
    fn future_model_knowledge_cannot_enter_earlier_vintage_backtest() {
        let split = HistoricalBenchmarkSplit::from_snapshots(&earlier_fixture(), &later_fixture())
            .unwrap();
        let future = HistoricalModelKnowledge::new(
            "DZ10",
            20170101,
            "fixture:future-implementation",
            vec!["fixture:future-reference".to_string()],
        )
        .unwrap();
        assert!(matches!(
            split.evaluate_dz10(&future),
            Err(HistoricalValidationError::ModelKnowledgePostdatesTrainingSnapshot { .. })
        ));
    }

    #[test]
    fn revisions_of_already_measured_nuclei_are_not_counted_as_new_predictions() {
        let split = HistoricalBenchmarkSplit::from_snapshots(&earlier_fixture(), &later_fixture())
            .unwrap();
        let revision = &split.receipt().revised_previously_measured[0];
        assert_eq!((revision.z, revision.n), (8, 8));
        assert!(split
            .holdout()
            .iter()
            .all(|nucleus| (nucleus.z, nucleus.n) != (8, 8)));
    }

    #[test]
    fn chronology_must_be_strictly_increasing() {
        let earlier = earlier_fixture();
        let same_date = HistoricalMassSnapshot::new(
            "fixture-other",
            20160101,
            "fixture:same-date",
            SnapshotAuthority::SyntheticFixture,
            &[nucleus(14, 14, 236.5, true)],
        )
        .unwrap();
        assert!(matches!(
            HistoricalBenchmarkSplit::from_snapshots(&earlier, &same_date),
            Err(HistoricalValidationError::NonIncreasingSnapshotChronology)
        ));
    }

    #[test]
    fn identical_external_artifact_cannot_masquerade_as_two_vintages() {
        let digest = "a".repeat(64);
        let earlier = HistoricalMassSnapshot::new(
            "source-a",
            20160101,
            "evidence:old",
            SnapshotAuthority::ExternalArtifactSha256 {
                sha256_hex: digest.clone(),
            },
            &[nucleus(6, 6, 92.1, true)],
        )
        .unwrap();
        let later = HistoricalMassSnapshot::new(
            "source-b",
            20200101,
            "evidence:new",
            SnapshotAuthority::ExternalArtifactSha256 { sha256_hex: digest },
            &[nucleus(8, 8, 127.6, true)],
        )
        .unwrap();
        assert!(matches!(
            HistoricalBenchmarkSplit::from_snapshots(&earlier, &later),
            Err(HistoricalValidationError::IndistinguishableSnapshots)
        ));
    }

    #[test]
    fn external_snapshot_requires_sha256_shaped_identity() {
        assert!(matches!(
            HistoricalMassSnapshot::new(
                "bad-digest",
                20160101,
                "evidence:date",
                SnapshotAuthority::ExternalArtifactSha256 {
                    sha256_hex: "1234".to_string(),
                },
                &[nucleus(6, 6, 92.1, true)],
            ),
            Err(HistoricalValidationError::InvalidSnapshotSha256)
        ));
    }

    #[test]
    fn invalid_calendar_dates_fail_closed() {
        assert!(matches!(
            HistoricalMassSnapshot::new(
                "bad-date",
                20210229,
                "evidence:date",
                SnapshotAuthority::SyntheticFixture,
                &[nucleus(6, 6, 92.1, true)],
            ),
            Err(HistoricalValidationError::InvalidReleaseDate)
        ));
    }
}
