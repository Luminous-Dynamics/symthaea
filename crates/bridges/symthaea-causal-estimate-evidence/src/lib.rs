//! Evidence contracts for numerical causal-effect estimates.
//!
//! Identification and estimation are intentionally separate. An identified
//! estimand says a causal quantity is recoverable under a structural model; it
//! does not provide a numerical effect magnitude. Likewise, a numerical `0.0`
//! is a valid estimate only when an estimator actually produced it with usable
//! data and diagnostics. Insufficient samples, degenerate treatment variance,
//! positivity failures, or unavailable diagnostics must remain explicit
//! `NotEstimable` states rather than being encoded as zero effects.

use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_causal_reasoning::counterfactual::{IdentificationMethod, UnidentifiedReason};
use symthaea_earth_causal_query::{CausalEffectRequest, EarthCausalQueryOutcome};

pub type Result<T> = std::result::Result<T, EstimateError>;

#[derive(Debug, Clone, PartialEq)]
pub enum EstimateError {
    EmptyField(&'static str),
    NonFinite { field: &'static str, value: f64 },
    Negative { field: &'static str, value: f64 },
    InvalidInterval { lower: f64, point: f64, upper: f64 },
    InvalidProbabilityRange { lower: f64, upper: f64 },
    InvalidSampleCount(usize),
    InvalidEffectiveSampleSize { effective: f64, sample_count: usize },
    DegenerateTreatmentVariance(f64),
    NumericalEstimateRequiresIdentifiedEstimand,
}

impl Display for EstimateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::NonFinite { field, value } => write!(f, "{field} must be finite, got {value}"),
            Self::Negative { field, value } => write!(f, "{field} must be non-negative, got {value}"),
            Self::InvalidInterval { lower, point, upper } => write!(
                f,
                "effect interval must satisfy lower <= point <= upper, got {lower} <= {point} <= {upper}"
            ),
            Self::InvalidProbabilityRange { lower, upper } => write!(
                f,
                "probability range must satisfy 0 <= lower <= upper <= 1, got [{lower}, {upper}]"
            ),
            Self::InvalidSampleCount(count) => {
                write!(f, "a numerical effect estimate requires at least two samples, got {count}")
            }
            Self::InvalidEffectiveSampleSize {
                effective,
                sample_count,
            } => write!(
                f,
                "effective sample size must be in (0, {sample_count}], got {effective}"
            ),
            Self::DegenerateTreatmentVariance(value) => write!(
                f,
                "treatment variance is too small for a numerical estimate: {value}"
            ),
            Self::NumericalEstimateRequiresIdentifiedEstimand => write!(
                f,
                "a numerical causal estimate requires an identified estimand; assumption-dependent and unidentified queries remain separate states"
            ),
        }
    }
}

impl Error for EstimateError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(EstimateError::EmptyField(field));
    }
    Ok(())
}

fn finite_non_negative(value: f64, field: &'static str) -> Result<f64> {
    if !value.is_finite() {
        return Err(EstimateError::NonFinite { field, value });
    }
    if value < 0.0 {
        return Err(EstimateError::Negative { field, value });
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatorFamily {
    LinearRegression,
    RegressionAdjustment,
    FrontdoorPathProduct,
    InverseProbabilityWeighting,
    DoublyRobust,
    InstrumentalVariables,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstimatorRef {
    pub family: EstimatorFamily,
    pub implementation: String,
    pub version: String,
    pub artifact_digest: Option<String>,
}

impl EstimatorRef {
    pub fn new(
        family: EstimatorFamily,
        implementation: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<Self> {
        let implementation = implementation.into();
        let version = version.into();
        non_empty(&implementation, "estimator implementation")?;
        non_empty(&version, "estimator version")?;
        Ok(Self {
            family,
            implementation,
            version,
            artifact_digest: None,
        })
    }

    pub fn with_artifact_digest(mut self, digest: impl Into<String>) -> Result<Self> {
        let digest = digest.into();
        non_empty(&digest, "estimator artifact digest")?;
        self.artifact_digest = Some(digest);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetRef {
    pub dataset_id: String,
    pub digest: String,
    pub variable_ids: Vec<String>,
    pub sample_count: usize,
}

impl DatasetRef {
    pub fn new(
        dataset_id: impl Into<String>,
        digest: impl Into<String>,
        variable_ids: Vec<String>,
        sample_count: usize,
    ) -> Result<Self> {
        let dataset_id = dataset_id.into();
        let digest = digest.into();
        non_empty(&dataset_id, "dataset id")?;
        non_empty(&digest, "dataset digest")?;
        for variable in &variable_ids {
            non_empty(variable, "dataset variable id")?;
        }
        Ok(Self {
            dataset_id,
            digest,
            variable_ids,
            sample_count,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectInterval {
    pub point: f64,
    pub lower: f64,
    pub upper: f64,
    pub unit: String,
}

impl EffectInterval {
    pub fn new(point: f64, lower: f64, upper: f64, unit: impl Into<String>) -> Result<Self> {
        for (field, value) in [("effect point", point), ("effect lower", lower), ("effect upper", upper)] {
            if !value.is_finite() {
                return Err(EstimateError::NonFinite { field, value });
            }
        }
        if lower > point || point > upper {
            return Err(EstimateError::InvalidInterval { lower, point, upper });
        }
        let unit = unit.into();
        non_empty(&unit, "effect unit")?;
        Ok(Self {
            point,
            lower,
            upper,
            unit,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimateDiagnostics {
    pub treatment_variance: f64,
    pub outcome_variance: f64,
    pub standard_error: Option<f64>,
    pub effective_sample_size: Option<f64>,
    pub propensity_range: Option<(f64, f64)>,
    pub warnings: Vec<String>,
}

impl EstimateDiagnostics {
    pub fn new(treatment_variance: f64, outcome_variance: f64) -> Result<Self> {
        let treatment_variance = finite_non_negative(treatment_variance, "treatment variance")?;
        let outcome_variance = finite_non_negative(outcome_variance, "outcome variance")?;
        Ok(Self {
            treatment_variance,
            outcome_variance,
            standard_error: None,
            effective_sample_size: None,
            propensity_range: None,
            warnings: Vec::new(),
        })
    }

    pub fn with_standard_error(mut self, standard_error: f64) -> Result<Self> {
        self.standard_error = Some(finite_non_negative(standard_error, "standard error")?);
        Ok(self)
    }

    pub fn with_effective_sample_size(
        mut self,
        effective_sample_size: f64,
        sample_count: usize,
    ) -> Result<Self> {
        if !effective_sample_size.is_finite()
            || effective_sample_size <= 0.0
            || effective_sample_size > sample_count as f64
        {
            return Err(EstimateError::InvalidEffectiveSampleSize {
                effective: effective_sample_size,
                sample_count,
            });
        }
        self.effective_sample_size = Some(effective_sample_size);
        Ok(self)
    }

    pub fn with_propensity_range(mut self, lower: f64, upper: f64) -> Result<Self> {
        if !lower.is_finite()
            || !upper.is_finite()
            || lower < 0.0
            || upper > 1.0
            || lower > upper
        {
            return Err(EstimateError::InvalidProbabilityRange { lower, upper });
        }
        self.propensity_range = Some((lower, upper));
        Ok(self)
    }

    pub fn with_warning(mut self, warning: impl Into<String>) -> Result<Self> {
        let warning = warning.into();
        non_empty(&warning, "diagnostic warning")?;
        self.warnings.push(warning);
        Ok(self)
    }

    pub fn validate_for_estimate(&self, sample_count: usize) -> Result<()> {
        if sample_count < 2 {
            return Err(EstimateError::InvalidSampleCount(sample_count));
        }
        // The current causal estimator uses ~1e-10 as its regression degeneracy
        // threshold. Planetary Perception refuses to reinterpret that failure as
        // a valid zero effect.
        if self.treatment_variance.abs() < 1e-10 {
            return Err(EstimateError::DegenerateTreatmentVariance(
                self.treatment_variance,
            ));
        }
        if let Some(effective) = self.effective_sample_size {
            if effective <= 0.0 || effective > sample_count as f64 {
                return Err(EstimateError::InvalidEffectiveSampleSize {
                    effective,
                    sample_count,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum IdentificationEvidence {
    Identified {
        estimand_description: String,
        method: IdentificationMethod,
        identification_confidence: f64,
        adjustment_evidence_ids: Vec<String>,
    },
    Unidentified {
        reason: UnidentifiedReason,
        missing: Vec<String>,
        suggestions: Vec<String>,
    },
    AssumptionRequired {
        condition: String,
        testability: f64,
        plausibility: f64,
        estimand_description: String,
        adjustment_evidence_ids: Vec<String>,
    },
}

impl IdentificationEvidence {
    pub fn from_query(outcome: &EarthCausalQueryOutcome) -> Result<Self> {
        match outcome {
            EarthCausalQueryOutcome::Identified {
                estimand_description,
                method,
                identification_confidence,
                adjustment_evidence_ids,
            } => {
                if !identification_confidence.is_finite() {
                    return Err(EstimateError::NonFinite {
                        field: "identification confidence",
                        value: *identification_confidence,
                    });
                }
                Ok(Self::Identified {
                    estimand_description: estimand_description.clone(),
                    method: *method,
                    identification_confidence: *identification_confidence,
                    adjustment_evidence_ids: adjustment_evidence_ids.clone(),
                })
            }
            EarthCausalQueryOutcome::Unidentified {
                reason,
                missing,
                suggestions,
            } => Ok(Self::Unidentified {
                reason: reason.clone(),
                missing: missing.clone(),
                suggestions: suggestions.clone(),
            }),
            EarthCausalQueryOutcome::AssumptionRequired {
                assumption,
                estimand_description,
                adjustment_evidence_ids,
                plausibility,
            } => {
                for (field, value) in [
                    ("assumption testability", assumption.testability),
                    ("assumption plausibility", *plausibility),
                ] {
                    if !value.is_finite() {
                        return Err(EstimateError::NonFinite { field, value });
                    }
                }
                Ok(Self::AssumptionRequired {
                    condition: assumption.condition.clone(),
                    testability: assumption.testability,
                    plausibility: *plausibility,
                    estimand_description: estimand_description.clone(),
                    adjustment_evidence_ids: adjustment_evidence_ids.clone(),
                })
            }
        }
    }

    pub const fn is_identified(&self) -> bool {
        matches!(self, Self::Identified { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotEstimableReason {
    InsufficientSamples,
    DegenerateTreatmentVariance,
    MissingVariables,
    PositivityViolation,
    DiagnosticsUnavailable,
    IdentificationNotEstablished,
    Other(String),
}

#[derive(Debug, Clone)]
pub enum NumericalEstimateStatus {
    Estimated {
        effect: EffectInterval,
        diagnostics: EstimateDiagnostics,
    },
    NotEstimable {
        reason: NotEstimableReason,
        note: String,
    },
}

/// Evidence record for one attempt to numerically estimate a causal effect.
///
/// Identification confidence and effect uncertainty remain separate. There is
/// intentionally no policy score or execution authority in this type.
#[derive(Debug, Clone)]
pub struct CausalEffectEstimateEvidence {
    pub id: String,
    pub request: CausalEffectRequest,
    pub identification: IdentificationEvidence,
    pub estimator: EstimatorRef,
    pub dataset: DatasetRef,
    pub status: NumericalEstimateStatus,
}

impl CausalEffectEstimateEvidence {
    pub fn estimated(
        id: impl Into<String>,
        request: CausalEffectRequest,
        identification_outcome: &EarthCausalQueryOutcome,
        estimator: EstimatorRef,
        dataset: DatasetRef,
        effect: EffectInterval,
        diagnostics: EstimateDiagnostics,
    ) -> Result<Self> {
        let id = id.into();
        non_empty(&id, "causal estimate evidence id")?;
        let identification = IdentificationEvidence::from_query(identification_outcome)?;
        if !identification.is_identified() {
            return Err(EstimateError::NumericalEstimateRequiresIdentifiedEstimand);
        }
        diagnostics.validate_for_estimate(dataset.sample_count)?;
        Ok(Self {
            id,
            request,
            identification,
            estimator,
            dataset,
            status: NumericalEstimateStatus::Estimated {
                effect,
                diagnostics,
            },
        })
    }

    pub fn not_estimable(
        id: impl Into<String>,
        request: CausalEffectRequest,
        identification_outcome: &EarthCausalQueryOutcome,
        estimator: EstimatorRef,
        dataset: DatasetRef,
        reason: NotEstimableReason,
        note: impl Into<String>,
    ) -> Result<Self> {
        let id = id.into();
        let note = note.into();
        non_empty(&id, "causal estimate evidence id")?;
        non_empty(&note, "not-estimable note")?;
        Ok(Self {
            id,
            request,
            identification: IdentificationEvidence::from_query(identification_outcome)?,
            estimator,
            dataset,
            status: NumericalEstimateStatus::NotEstimable { reason, note },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> CausalEffectRequest {
        CausalEffectRequest::new("rainfall", "vegetation", vec!["soil".into()]).unwrap()
    }

    fn identified() -> EarthCausalQueryOutcome {
        EarthCausalQueryOutcome::Identified {
            estimand_description: "E[Y|do(X)]".into(),
            method: IdentificationMethod::BackdoorAdjustment,
            identification_confidence: 0.9,
            adjustment_evidence_ids: vec!["soil".into()],
        }
    }

    fn estimator() -> EstimatorRef {
        EstimatorRef::new(
            EstimatorFamily::RegressionAdjustment,
            "symthaea-causal-reasoning",
            "0.1.0",
        )
        .unwrap()
    }

    fn dataset(n: usize) -> DatasetRef {
        DatasetRef::new(
            "wetland-series-v1",
            "sha256:fixture",
            vec!["rainfall".into(), "vegetation".into(), "soil".into()],
            n,
        )
        .unwrap()
    }

    #[test]
    fn genuine_zero_effect_is_representable_when_diagnostics_are_valid() {
        let evidence = CausalEffectEstimateEvidence::estimated(
            "estimate-1",
            request(),
            &identified(),
            estimator(),
            dataset(100),
            EffectInterval::new(0.0, -0.05, 0.05, "NDVI/unit-rainfall").unwrap(),
            EstimateDiagnostics::new(0.2, 0.1)
                .unwrap()
                .with_standard_error(0.025)
                .unwrap(),
        )
        .unwrap();

        assert!(matches!(
            evidence.status,
            NumericalEstimateStatus::Estimated { .. }
        ));
    }

    #[test]
    fn degenerate_treatment_variance_cannot_be_encoded_as_zero_effect() {
        let result = CausalEffectEstimateEvidence::estimated(
            "estimate-2",
            request(),
            &identified(),
            estimator(),
            dataset(100),
            EffectInterval::new(0.0, 0.0, 0.0, "NDVI/unit-rainfall").unwrap(),
            EstimateDiagnostics::new(0.0, 0.1).unwrap(),
        );
        assert_eq!(
            result.unwrap_err(),
            EstimateError::DegenerateTreatmentVariance(0.0)
        );
    }

    #[test]
    fn insufficient_samples_remain_not_estimable_instead_of_zero() {
        let evidence = CausalEffectEstimateEvidence::not_estimable(
            "estimate-3",
            request(),
            &identified(),
            estimator(),
            dataset(1),
            NotEstimableReason::InsufficientSamples,
            "one sample cannot support this estimate",
        )
        .unwrap();
        assert!(matches!(
            evidence.status,
            NumericalEstimateStatus::NotEstimable {
                reason: NotEstimableReason::InsufficientSamples,
                ..
            }
        ));
    }

    #[test]
    fn unidentified_query_cannot_be_promoted_to_numeric_effect() {
        let outcome = EarthCausalQueryOutcome::Unidentified {
            reason: UnidentifiedReason::NotConnected,
            missing: vec![],
            suggestions: vec![],
        };
        let result = CausalEffectEstimateEvidence::estimated(
            "estimate-4",
            request(),
            &outcome,
            estimator(),
            dataset(100),
            EffectInterval::new(0.2, 0.1, 0.3, "NDVI/unit-rainfall").unwrap(),
            EstimateDiagnostics::new(0.2, 0.1).unwrap(),
        );
        assert_eq!(
            result.unwrap_err(),
            EstimateError::NumericalEstimateRequiresIdentifiedEstimand
        );
    }

    #[test]
    fn identification_confidence_is_not_effect_uncertainty() {
        let evidence = CausalEffectEstimateEvidence::estimated(
            "estimate-5",
            request(),
            &identified(),
            estimator(),
            dataset(100),
            EffectInterval::new(0.4, 0.2, 0.6, "NDVI/unit-rainfall").unwrap(),
            EstimateDiagnostics::new(0.3, 0.2).unwrap(),
        )
        .unwrap();

        match (&evidence.identification, &evidence.status) {
            (
                IdentificationEvidence::Identified {
                    identification_confidence,
                    ..
                },
                NumericalEstimateStatus::Estimated { effect, .. },
            ) => {
                assert_eq!(*identification_confidence, 0.9);
                assert_eq!(effect.lower, 0.2);
                assert_eq!(effect.upper, 0.6);
            }
            _ => panic!("expected identified numerical estimate"),
        }
    }
}
