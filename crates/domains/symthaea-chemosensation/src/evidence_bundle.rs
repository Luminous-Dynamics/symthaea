// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bind decision receipts to the exact archived traces used as evidence.
//!
//! Decision receipts and trace archives are individually auditable, but a final
//! experiment result also needs an explicit statement of *which* traces supported
//! it and how each trace was used. This module provides that linkage and executes
//! the preregistered deviation policy when checking admissibility.

use crate::{
    ChemicalDecisionReceipt, ChemicalEvidenceLevel, ChemicalTraceArchive, DeviationDisposition,
    ExperimentDecision, ProtocolDeviation, TraceArchiveDigest, TraceArchiveError,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const EVIDENCE_BUNDLE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEvidenceRef {
    pub digest: TraceArchiveDigest,
    pub acquisition_protocol_id: String,
    pub run_id: String,
    pub original_acquisition_evidence: ChemicalEvidenceLevel,
    /// Evidence class under which this archive is used in the decision.
    ///
    /// This may equal the original acquisition class, or it may be
    /// `RecordedReplay`. No other relabeling is admissible.
    pub used_as: ChemicalEvidenceLevel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceEvidenceRefError {
    BlankAcquisitionProtocolId,
    BlankRunId,
    ReplayCannotBeOriginalAcquisition,
    InvalidEvidenceRelabel {
        original: ChemicalEvidenceLevel,
        used_as: ChemicalEvidenceLevel,
    },
    ArchiveInvalid(TraceArchiveError),
    ArchiveDigestMismatch,
    ArchiveProtocolMismatch,
    ArchiveRunMismatch,
    ArchiveOriginEvidenceMismatch,
}

impl From<TraceArchiveError> for TraceEvidenceRefError {
    fn from(value: TraceArchiveError) -> Self {
        Self::ArchiveInvalid(value)
    }
}

impl TraceEvidenceRef {
    pub fn from_archive(
        archive: &ChemicalTraceArchive,
        used_as: ChemicalEvidenceLevel,
    ) -> Result<Self, TraceEvidenceRefError> {
        archive.verify()?;
        let reference = Self {
            digest: archive.manifest.digest,
            acquisition_protocol_id: archive.manifest.protocol_id.clone(),
            run_id: archive.manifest.run_id.clone(),
            original_acquisition_evidence: archive.manifest.acquisition_evidence,
            used_as,
        };
        reference.verify_self()?;
        Ok(reference)
    }

    pub fn verify_self(&self) -> Result<(), TraceEvidenceRefError> {
        if self.acquisition_protocol_id.trim().is_empty() {
            return Err(TraceEvidenceRefError::BlankAcquisitionProtocolId);
        }
        if self.run_id.trim().is_empty() {
            return Err(TraceEvidenceRefError::BlankRunId);
        }
        if self.original_acquisition_evidence == ChemicalEvidenceLevel::RecordedReplay {
            return Err(TraceEvidenceRefError::ReplayCannotBeOriginalAcquisition);
        }
        if self.used_as != self.original_acquisition_evidence
            && self.used_as != ChemicalEvidenceLevel::RecordedReplay
        {
            return Err(TraceEvidenceRefError::InvalidEvidenceRelabel {
                original: self.original_acquisition_evidence,
                used_as: self.used_as,
            });
        }
        Ok(())
    }

    /// Cross-check this reference against the archive it names.
    pub fn verify_against_archive(
        &self,
        archive: &ChemicalTraceArchive,
    ) -> Result<(), TraceEvidenceRefError> {
        self.verify_self()?;
        archive.verify()?;
        if self.digest != archive.manifest.digest {
            return Err(TraceEvidenceRefError::ArchiveDigestMismatch);
        }
        if self.acquisition_protocol_id != archive.manifest.protocol_id {
            return Err(TraceEvidenceRefError::ArchiveProtocolMismatch);
        }
        if self.run_id != archive.manifest.run_id {
            return Err(TraceEvidenceRefError::ArchiveRunMismatch);
        }
        if self.original_acquisition_evidence != archive.manifest.acquisition_evidence {
            return Err(TraceEvidenceRefError::ArchiveOriginEvidenceMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemicalEvidenceBundle {
    pub schema_version: u16,
    pub decision: ChemicalDecisionReceipt,
    pub traces: Vec<TraceEvidenceRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deviations: Vec<ProtocolDeviation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceBundleError {
    UnsupportedSchemaVersion { actual: u16 },
    InvalidDecisionReceipt,
    EmptyTraceSet,
    InvalidTraceRef(TraceEvidenceRefError),
    DuplicateTraceDigest(TraceArchiveDigest),
    DuplicateRunRef { protocol_id: String, run_id: String },
    DecisionProtocolMismatch {
        expected: String,
        actual: String,
        run_id: String,
    },
    DecisionEvidenceMismatch {
        expected: ChemicalEvidenceLevel,
        actual: ChemicalEvidenceLevel,
        run_id: String,
    },
    DeviationProtocolMismatch,
    DiscardedRunReferenced(String),
    ConfirmedDecisionUsesSeparatedEvidence(String),
    ConfirmedDecisionHasProtocolWideSeparation,
    MissingArchive(TraceArchiveDigest),
    UnexpectedArchive(TraceArchiveDigest),
    DuplicateArchiveDigest(TraceArchiveDigest),
}

impl From<TraceEvidenceRefError> for EvidenceBundleError {
    fn from(value: TraceEvidenceRefError) -> Self {
        Self::InvalidTraceRef(value)
    }
}

impl ChemicalEvidenceBundle {
    pub fn new(
        decision: ChemicalDecisionReceipt,
        traces: Vec<TraceEvidenceRef>,
        deviations: Vec<ProtocolDeviation>,
    ) -> Result<Self, EvidenceBundleError> {
        let bundle = Self {
            schema_version: EVIDENCE_BUNDLE_SCHEMA_VERSION,
            decision,
            traces,
            deviations,
        };
        bundle.verify()?;
        Ok(bundle)
    }

    /// Verify structural linkage between the decision, trace references,
    /// evidence classes, and any declared protocol deviations.
    pub fn verify(&self) -> Result<(), EvidenceBundleError> {
        if self.schema_version != EVIDENCE_BUNDLE_SCHEMA_VERSION {
            return Err(EvidenceBundleError::UnsupportedSchemaVersion {
                actual: self.schema_version,
            });
        }
        if !self.decision.verify_self() {
            return Err(EvidenceBundleError::InvalidDecisionReceipt);
        }
        if self.traces.is_empty() {
            return Err(EvidenceBundleError::EmptyTraceSet);
        }

        let mut digests = BTreeSet::new();
        let mut runs = BTreeSet::new();
        for trace in &self.traces {
            trace.verify_self()?;
            if !digests.insert(trace.digest.0) {
                return Err(EvidenceBundleError::DuplicateTraceDigest(trace.digest));
            }
            let run_key = (
                trace.acquisition_protocol_id.clone(),
                trace.run_id.clone(),
            );
            if !runs.insert(run_key.clone()) {
                return Err(EvidenceBundleError::DuplicateRunRef {
                    protocol_id: run_key.0,
                    run_id: run_key.1,
                });
            }
            if trace.acquisition_protocol_id != self.decision.protocol_id {
                return Err(EvidenceBundleError::DecisionProtocolMismatch {
                    expected: self.decision.protocol_id.clone(),
                    actual: trace.acquisition_protocol_id.clone(),
                    run_id: trace.run_id.clone(),
                });
            }
            if trace.used_as != self.decision.evidence {
                return Err(EvidenceBundleError::DecisionEvidenceMismatch {
                    expected: self.decision.evidence,
                    actual: trace.used_as,
                    run_id: trace.run_id.clone(),
                });
            }
        }

        for deviation in &self.deviations {
            if deviation.protocol_id != self.decision.protocol_id
                || deviation.protocol_version != self.decision.version
            {
                return Err(EvidenceBundleError::DeviationProtocolMismatch);
            }

            let affected_refs: Vec<&TraceEvidenceRef> = if deviation.affected_run_ids.is_empty() {
                Vec::new()
            } else {
                self.traces
                    .iter()
                    .filter(|trace| deviation.affected_run_ids.contains(&trace.run_id))
                    .collect()
            };

            if deviation.disposition == DeviationDisposition::RestartAffectedRun {
                if let Some(trace) = affected_refs.first() {
                    return Err(EvidenceBundleError::DiscardedRunReferenced(
                        trace.run_id.clone(),
                    ));
                }
                continue;
            }

            if self.decision.decision == ExperimentDecision::Confirmed {
                if deviation.affected_run_ids.is_empty() {
                    return Err(EvidenceBundleError::ConfirmedDecisionHasProtocolWideSeparation);
                }
                if let Some(trace) = affected_refs.first() {
                    return Err(EvidenceBundleError::ConfirmedDecisionUsesSeparatedEvidence(
                        trace.run_id.clone(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// End-to-end verification when the referenced archives are available.
    ///
    /// This first checks the bundle's decision/deviation semantics, then requires
    /// an exact archive set and re-runs each archive's own integrity and
    /// trace-invariant verification.
    pub fn verify_with_archives(
        &self,
        archives: &[ChemicalTraceArchive],
    ) -> Result<(), EvidenceBundleError> {
        self.verify()?;

        let mut supplied_digests = BTreeSet::new();
        for archive in archives {
            if !supplied_digests.insert(archive.manifest.digest.0) {
                return Err(EvidenceBundleError::DuplicateArchiveDigest(
                    archive.manifest.digest,
                ));
            }
        }

        for trace_ref in &self.traces {
            let archive = archives
                .iter()
                .find(|archive| archive.manifest.digest == trace_ref.digest)
                .ok_or(EvidenceBundleError::MissingArchive(trace_ref.digest))?;
            trace_ref.verify_against_archive(archive)?;
        }

        for archive in archives {
            if !self
                .traces
                .iter()
                .any(|trace_ref| trace_ref.digest == archive.manifest.digest)
            {
                return Err(EvidenceBundleError::UnexpectedArchive(
                    archive.manifest.digest,
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationState, ChemicalChannel, ChemicalDecisionProtocol, ChemicalModality,
        ChemicalObservation, ChemicalTrace, DeviationDisposition, EvaluationPartition,
        ExpectedBiasDirection, GateDirection, MeasurementUnit, MetricGate, MetricObservation,
        ProtocolDeviation, SamplingContext, SamplingPhase, SensorHealth,
    };

    fn archive(evidence: ChemicalEvidenceLevel, run_id: &str) -> ChemicalTraceArchive {
        let sampling = SamplingContext::new("od001-v1", run_id, SamplingPhase::Exposure, 0)
            .unwrap();
        let observation = ChemicalObservation::new(
            1_000,
            ChemicalModality::Olfactory,
            "nose-a",
            vec![ChemicalChannel {
                name: "mox-0".into(),
                raw_value: 42_000.0,
                unit: MeasurementUnit::Ohms,
                calibration: CalibrationState::identity("session-a"),
                health: SensorHealth::default(),
            }],
        )
        .with_sampling(sampling);
        let trace = ChemicalTrace::new(observation).unwrap();
        ChemicalTraceArchive::from_trace(&trace, evidence).unwrap()
    }

    fn decision(evidence: ChemicalEvidenceLevel, value: f64) -> ChemicalDecisionReceipt {
        ChemicalDecisionProtocol::new(
            "od001-v1",
            "1.0.0",
            evidence,
            EvaluationPartition::Holdout,
            vec![MetricGate::new("accuracy", GateDirection::AtLeast, 0.8, Some(0.5)).unwrap()],
        )
        .unwrap()
        .evaluate(
            evidence,
            EvaluationPartition::Holdout,
            &[MetricObservation::new("accuracy", value)],
        )
        .unwrap()
    }

    #[test]
    fn heldout_decision_can_bind_to_exact_heldout_archive() {
        let archive = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let trace_ref = TraceEvidenceRef::from_archive(
            &archive,
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();

        let bundle = ChemicalEvidenceBundle::new(
            decision(ChemicalEvidenceLevel::HeldOutPhysicalObservation, 0.9),
            vec![trace_ref],
            vec![],
        )
        .unwrap();
        assert!(bundle.verify_with_archives(&[archive]).is_ok());
    }

    #[test]
    fn replay_decision_preserves_origin_but_uses_replay_evidence_class() {
        let archive = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let trace_ref =
            TraceEvidenceRef::from_archive(&archive, ChemicalEvidenceLevel::RecordedReplay)
                .unwrap();
        assert_eq!(
            trace_ref.original_acquisition_evidence,
            ChemicalEvidenceLevel::HeldOutPhysicalObservation
        );

        let bundle = ChemicalEvidenceBundle::new(
            decision(ChemicalEvidenceLevel::RecordedReplay, 0.9),
            vec![trace_ref],
            vec![],
        )
        .unwrap();
        assert!(bundle.verify_with_archives(&[archive]).is_ok());
    }

    #[test]
    fn evidence_cannot_be_relabelled_as_a_different_physical_class() {
        let archive = archive(ChemicalEvidenceLevel::BenchPhysicalObservation, "run-a");
        assert_eq!(
            TraceEvidenceRef::from_archive(
                &archive,
                ChemicalEvidenceLevel::HeldOutPhysicalObservation,
            ),
            Err(TraceEvidenceRefError::InvalidEvidenceRelabel {
                original: ChemicalEvidenceLevel::BenchPhysicalObservation,
                used_as: ChemicalEvidenceLevel::HeldOutPhysicalObservation,
            })
        );
    }

    #[test]
    fn decision_protocol_must_match_trace_protocol() {
        let archive = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let mut trace_ref = TraceEvidenceRef::from_archive(
            &archive,
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();
        trace_ref.acquisition_protocol_id = "other-v1".into();

        assert!(matches!(
            ChemicalEvidenceBundle::new(
                decision(ChemicalEvidenceLevel::HeldOutPhysicalObservation, 0.9),
                vec![trace_ref],
                vec![],
            ),
            Err(EvidenceBundleError::DecisionProtocolMismatch { .. })
        ));
    }

    #[test]
    fn decision_evidence_must_match_every_trace_use() {
        let archive = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let trace_ref =
            TraceEvidenceRef::from_archive(&archive, ChemicalEvidenceLevel::RecordedReplay)
                .unwrap();
        assert!(matches!(
            ChemicalEvidenceBundle::new(
                decision(ChemicalEvidenceLevel::HeldOutPhysicalObservation, 0.9),
                vec![trace_ref],
                vec![],
            ),
            Err(EvidenceBundleError::DecisionEvidenceMismatch { .. })
        ));
    }

    #[test]
    fn restart_deviation_makes_affected_run_inadmissible() {
        let archive = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let trace_ref = TraceEvidenceRef::from_archive(
            &archive,
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();
        let deviation = ProtocolDeviation::new(
            "od001-v1",
            "1.0.0",
            1_500,
            vec!["run-a".into()],
            vec!["pump_timeout".into()],
            "predeclared timeout triggered",
            false,
            ExpectedBiasDirection::Neutral,
            DeviationDisposition::RestartAffectedRun,
        )
        .unwrap();

        assert_eq!(
            ChemicalEvidenceBundle::new(
                decision(ChemicalEvidenceLevel::HeldOutPhysicalObservation, 0.9),
                vec![trace_ref],
                vec![deviation],
            ),
            Err(EvidenceBundleError::DiscardedRunReferenced("run-a".into()))
        );
    }

    #[test]
    fn confirmed_decision_cannot_use_exploratory_affected_run() {
        let archive = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let trace_ref = TraceEvidenceRef::from_archive(
            &archive,
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();
        let deviation = ProtocolDeviation::new(
            "od001-v1",
            "1.0.0",
            1_500,
            vec!["run-a".into()],
            vec!["threshold".into()],
            "threshold changed after outcome inspection",
            true,
            ExpectedBiasDirection::TowardConfirmation,
            DeviationDisposition::ExploratoryOnly,
        )
        .unwrap();

        assert_eq!(
            ChemicalEvidenceBundle::new(
                decision(ChemicalEvidenceLevel::HeldOutPhysicalObservation, 0.9),
                vec![trace_ref],
                vec![deviation],
            ),
            Err(EvidenceBundleError::ConfirmedDecisionUsesSeparatedEvidence(
                "run-a".into()
            ))
        );
    }

    #[test]
    fn end_to_end_verification_detects_archive_tampering() {
        let mut archive = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let trace_ref = TraceEvidenceRef::from_archive(
            &archive,
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();
        let bundle = ChemicalEvidenceBundle::new(
            decision(ChemicalEvidenceLevel::HeldOutPhysicalObservation, 0.9),
            vec![trace_ref],
            vec![],
        )
        .unwrap();

        archive.observations[0].channels[0].raw_value += 1.0;
        assert!(matches!(
            bundle.verify_with_archives(&[archive]),
            Err(EvidenceBundleError::InvalidTraceRef(
                TraceEvidenceRefError::ArchiveInvalid(TraceArchiveError::DigestMismatch)
            ))
        ));
    }

    #[test]
    fn end_to_end_verification_requires_exact_archive_set() {
        let archive_a = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let archive_b = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-b");
        let trace_ref = TraceEvidenceRef::from_archive(
            &archive_a,
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();
        let bundle = ChemicalEvidenceBundle::new(
            decision(ChemicalEvidenceLevel::HeldOutPhysicalObservation, 0.9),
            vec![trace_ref],
            vec![],
        )
        .unwrap();

        assert_eq!(
            bundle.verify_with_archives(&[]),
            Err(EvidenceBundleError::MissingArchive(archive_a.manifest.digest))
        );
        assert_eq!(
            bundle.verify_with_archives(&[archive_a, archive_b.clone()]),
            Err(EvidenceBundleError::UnexpectedArchive(
                archive_b.manifest.digest
            ))
        );
    }

    #[test]
    fn duplicate_supplied_archive_is_rejected() {
        let archive = archive(ChemicalEvidenceLevel::HeldOutPhysicalObservation, "run-a");
        let trace_ref = TraceEvidenceRef::from_archive(
            &archive,
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();
        let bundle = ChemicalEvidenceBundle::new(
            decision(ChemicalEvidenceLevel::HeldOutPhysicalObservation, 0.9),
            vec![trace_ref],
            vec![],
        )
        .unwrap();

        assert_eq!(
            bundle.verify_with_archives(&[archive.clone(), archive.clone()]),
            Err(EvidenceBundleError::DuplicateArchiveDigest(
                archive.manifest.digest
            ))
        );
    }
}
