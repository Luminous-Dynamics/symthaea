// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-sample evidence and conservative uncertainty propagation for EAC-004.
//!
//! A nonlinear curve/table may be assembled from more than one measurement,
//! fit, datasheet, or numerical lineage. `StateDependentParameter::source`
//! remains useful as dataset-level provenance, but it is not sufficient to prove
//! the lineage of every supplied point. This module binds every sample to its own
//! source and optional absolute uncertainty interval.
//!
//! Interpolation never becomes measurement authority. A 1D interpolated result
//! records the two source samples that support it and, when both endpoints carry
//! uncertainty, propagates a conservative hull spanning both endpoint intervals.
//! Missing endpoint uncertainty stays unknown rather than being fabricated.

use crate::nonlinear::{
    EvaluationDisposition, NonlinearModelError, StateDependentParameter, StateEvaluation,
};
use crate::{ParameterSource, UncertaintyInterval, ValidationError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

/// Evidence attached to exactly one supplied nonlinear sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateSampleEvidence {
    pub sample_index: usize,
    pub source: ParameterSource,
    /// Absolute interval in the parameter response unit.
    pub uncertainty: Option<UncertaintyInterval>,
}

impl StateSampleEvidence {
    pub fn new(sample_index: usize, source: ParameterSource) -> Self {
        Self {
            sample_index,
            source,
            uncertainty: None,
        }
    }

    pub fn with_uncertainty(mut self, uncertainty: UncertaintyInterval) -> Self {
        self.uncertainty = Some(uncertainty);
        self
    }
}

/// A state-dependent parameter whose individual samples retain their own
/// provenance/uncertainty rather than inheriting only one dataset-level source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBoundStateParameter {
    pub parameter: StateDependentParameter,
    pub sample_evidence: Vec<StateSampleEvidence>,
}

impl EvidenceBoundStateParameter {
    pub fn validate(&self) -> Result<(), NonlinearEvidenceError> {
        self.parameter.validate()?;

        if self.sample_evidence.len() != self.parameter.samples.len() {
            return Err(NonlinearEvidenceError::EvidenceCountMismatch {
                expected: self.parameter.samples.len(),
                actual: self.sample_evidence.len(),
            });
        }

        let mut seen = BTreeSet::new();
        for evidence in &self.sample_evidence {
            if evidence.sample_index >= self.parameter.samples.len() {
                return Err(NonlinearEvidenceError::SampleIndexOutOfRange {
                    sample_index: evidence.sample_index,
                    sample_count: self.parameter.samples.len(),
                });
            }
            if !seen.insert(evidence.sample_index) {
                return Err(NonlinearEvidenceError::DuplicateSampleEvidence(
                    evidence.sample_index,
                ));
            }
            evidence.source.validate()?;
            if let Some(interval) = evidence.uncertainty {
                interval.validate()?;
                let nominal = self.parameter.samples[evidence.sample_index].value;
                if !interval.contains(nominal) {
                    return Err(NonlinearEvidenceError::SampleUncertaintyDoesNotContainNominal {
                        sample_index: evidence.sample_index,
                        nominal,
                        lower: interval.lower,
                        upper: interval.upper,
                    });
                }
            }
        }

        // Equal lengths plus unique, in-range indices proves complete coverage.
        Ok(())
    }

    pub fn evaluate(
        &self,
        coordinates: &[f64],
    ) -> Result<EvidenceBoundStateEvaluation, NonlinearEvidenceError> {
        self.validate()?;
        let evaluation = self.parameter.evaluate(coordinates)?;

        match evaluation.disposition {
            EvaluationDisposition::ExactSample => {
                let sample_index = self
                    .parameter
                    .samples
                    .iter()
                    .position(|sample| sample.coordinates == coordinates)
                    .ok_or(NonlinearEvidenceError::ExactSampleEvidenceNotFound)?;
                let evidence = self.evidence_for(sample_index)?;
                let uncertainty = evidence.uncertainty;
                Ok(EvidenceBoundStateEvaluation {
                    evaluation,
                    authority: EvaluationAuthority::SourceSample,
                    contributing_samples: vec![sample_index],
                    sources: vec![evidence.source.clone()],
                    uncertainty,
                    uncertainty_basis: if uncertainty.is_some() {
                        UncertaintyBasis::ExactSampleInterval
                    } else {
                        UncertaintyBasis::Unavailable
                    },
                })
            }
            EvaluationDisposition::LinearInterpolated {
                lower_sample,
                upper_sample,
            } => {
                let lower = self.evidence_for(lower_sample)?;
                let upper = self.evidence_for(upper_sample)?;
                let (uncertainty, uncertainty_basis) = match (lower.uncertainty, upper.uncertainty) {
                    (Some(a), Some(b)) => (
                        Some(UncertaintyInterval {
                            lower: a.lower.min(b.lower),
                            upper: a.upper.max(b.upper),
                        }),
                        UncertaintyBasis::ConservativeEndpointHull,
                    ),
                    _ => (None, UncertaintyBasis::Unavailable),
                };

                Ok(EvidenceBoundStateEvaluation {
                    evaluation,
                    authority: EvaluationAuthority::DerivedInterpolation,
                    contributing_samples: vec![lower_sample, upper_sample],
                    sources: vec![lower.source.clone(), upper.source.clone()],
                    uncertainty,
                    uncertainty_basis,
                })
            }
        }
    }

    fn evidence_for(&self, sample_index: usize) -> Result<&StateSampleEvidence, NonlinearEvidenceError> {
        self.sample_evidence
            .iter()
            .find(|evidence| evidence.sample_index == sample_index)
            .ok_or(NonlinearEvidenceError::MissingSampleEvidence(sample_index))
    }
}

/// Authority of the returned value. This deliberately prevents an interpolated
/// result between measured samples from being represented as a measured value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationAuthority {
    SourceSample,
    DerivedInterpolation,
}

/// How the returned uncertainty was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyBasis {
    ExactSampleInterval,
    /// Hull of the two endpoint intervals. This is deliberately more
    /// conservative than pretending the uncertainty itself varies linearly.
    ConservativeEndpointHull,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBoundStateEvaluation {
    pub evaluation: StateEvaluation,
    pub authority: EvaluationAuthority,
    pub contributing_samples: Vec<usize>,
    pub sources: Vec<ParameterSource>,
    pub uncertainty: Option<UncertaintyInterval>,
    pub uncertainty_basis: UncertaintyBasis,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum NonlinearEvidenceError {
    #[error(transparent)]
    Model(#[from] NonlinearModelError),
    #[error(transparent)]
    Parameter(#[from] ValidationError),
    #[error("sample evidence count mismatch: expected {expected}, got {actual}")]
    EvidenceCountMismatch { expected: usize, actual: usize },
    #[error("sample evidence index {sample_index} is outside sample count {sample_count}")]
    SampleIndexOutOfRange {
        sample_index: usize,
        sample_count: usize,
    },
    #[error("duplicate evidence for sample {0}")]
    DuplicateSampleEvidence(usize),
    #[error("missing evidence for sample {0}")]
    MissingSampleEvidence(usize),
    #[error(
        "sample {sample_index} uncertainty [{lower}, {upper}] does not contain nominal {nominal}"
    )]
    SampleUncertaintyDoesNotContainNominal {
        sample_index: usize,
        nominal: f64,
        lower: f64,
        upper: f64,
    },
    #[error("exact sample evaluation could not be mapped back to source evidence")]
    ExactSampleEvidenceNotFound,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear::{
        ExtrapolationPolicy, InterpolationPolicy, StateAxis, StateAxisKind, StateAxisUnit,
        StateResponseKind, StateSample,
    };
    use crate::{ParameterSourceKind, PhysicalUnit};

    fn measured(name: &str) -> ParameterSource {
        ParameterSource::new(ParameterSourceKind::Measured, format!("measurement {name}"))
            .with_evidence_ref(format!("measurement-run:{name}"))
    }

    fn curve() -> EvidenceBoundStateParameter {
        let parameter = StateDependentParameter {
            id: "bl-x-evidence".into(),
            response: StateResponseKind::ForceFactor,
            response_unit: PhysicalUnit::TeslaMeter,
            axes: vec![StateAxis::new(
                StateAxisKind::Displacement,
                StateAxisUnit::Meter,
                -0.01,
                0.01,
            )],
            samples: vec![
                StateSample::new(vec![-0.01], 3.5),
                StateSample::new(vec![0.0], 5.0),
                StateSample::new(vec![0.01], 4.0),
            ],
            source: ParameterSource::new(ParameterSourceKind::Measured, "assembled Bl(x) dataset")
                .with_evidence_ref("dataset:bl-x"),
            interpolation: InterpolationPolicy::Linear1D,
            extrapolation: ExtrapolationPolicy::Reject,
        };

        EvidenceBoundStateParameter {
            parameter,
            sample_evidence: vec![
                StateSampleEvidence::new(0, measured("neg"))
                    .with_uncertainty(UncertaintyInterval::new(3.3, 3.7).unwrap()),
                StateSampleEvidence::new(1, measured("zero"))
                    .with_uncertainty(UncertaintyInterval::new(4.9, 5.1).unwrap()),
                StateSampleEvidence::new(2, measured("pos"))
                    .with_uncertainty(UncertaintyInterval::new(3.8, 4.2).unwrap()),
            ],
        }
    }

    #[test]
    fn exact_sample_retains_sample_level_lineage_and_uncertainty() {
        let result = curve().evaluate(&[0.0]).unwrap();
        assert_eq!(result.authority, EvaluationAuthority::SourceSample);
        assert_eq!(result.contributing_samples, vec![1]);
        assert_eq!(result.sources[0].evidence_ref.as_deref(), Some("measurement-run:zero"));
        assert_eq!(result.uncertainty, Some(UncertaintyInterval { lower: 4.9, upper: 5.1 }));
        assert_eq!(result.uncertainty_basis, UncertaintyBasis::ExactSampleInterval);
    }

    #[test]
    fn interpolation_is_derived_and_uses_conservative_endpoint_hull() {
        let result = curve().evaluate(&[-0.005]).unwrap();
        assert_eq!(result.authority, EvaluationAuthority::DerivedInterpolation);
        assert_eq!(result.contributing_samples, vec![0, 1]);
        assert_eq!(result.sources.len(), 2);
        assert_eq!(result.uncertainty, Some(UncertaintyInterval { lower: 3.3, upper: 5.1 }));
        assert_eq!(
            result.uncertainty_basis,
            UncertaintyBasis::ConservativeEndpointHull
        );
    }

    #[test]
    fn missing_endpoint_uncertainty_stays_unknown() {
        let mut bound = curve();
        bound.sample_evidence[1].uncertainty = None;
        let result = bound.evaluate(&[-0.005]).unwrap();
        assert_eq!(result.authority, EvaluationAuthority::DerivedInterpolation);
        assert_eq!(result.uncertainty, None);
        assert_eq!(result.uncertainty_basis, UncertaintyBasis::Unavailable);
    }

    #[test]
    fn distinct_sample_lineages_are_preserved() {
        let result = curve().evaluate(&[0.005]).unwrap();
        let refs: Vec<_> = result
            .sources
            .iter()
            .map(|source| source.evidence_ref.as_deref().unwrap())
            .collect();
        assert_eq!(refs, vec!["measurement-run:zero", "measurement-run:pos"]);
    }

    #[test]
    fn incomplete_sample_evidence_fails_closed() {
        let mut bound = curve();
        bound.sample_evidence.pop();
        assert!(matches!(
            bound.validate(),
            Err(NonlinearEvidenceError::EvidenceCountMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_sample_evidence_fails_closed() {
        let mut bound = curve();
        bound.sample_evidence[2].sample_index = 1;
        assert_eq!(
            bound.validate(),
            Err(NonlinearEvidenceError::DuplicateSampleEvidence(1))
        );
    }

    #[test]
    fn sample_uncertainty_must_contain_nominal() {
        let mut bound = curve();
        bound.sample_evidence[0].uncertainty = Some(UncertaintyInterval::new(2.0, 3.0).unwrap());
        assert!(matches!(
            bound.validate(),
            Err(NonlinearEvidenceError::SampleUncertaintyDoesNotContainNominal {
                sample_index: 0,
                ..
            })
        ));
    }
}
