//! Append-only forecast verification and calibration contracts.
//!
//! Forecasts must exist before their validity window, verification refers back
//! to the immutable forecast id, and calibration reports preserve misses as well
//! as hits. This crate does not produce a universal model trust score.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_earth_observation::{EvidenceRef, EvidenceStage};
use symthaea_scenario_outcomes::OutcomeEstimate;

pub type Result<T> = std::result::Result<T, CalibrationError>;

#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationError {
    EmptyField(&'static str),
    NonFinite { field: &'static str, value: f64 },
    InvalidProbability(f64),
    InvalidForecastWindow { issued_at: i64, valid_from: i64, valid_until: i64 },
    DuplicateForecast(String),
    UnknownForecast(String),
    DuplicateVerification(String),
    VerificationOutsideWindow { observed_for: i64, valid_from: i64, valid_until: i64 },
    MissingVerificationEvidence,
    VerificationEvidenceNotVerificationStage(String),
    VerificationKindMismatch,
    UnitMismatch { expected: String, got: String },
    NoForecastsForModelTarget,
}

impl Display for CalibrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::NonFinite { field, value } => write!(f, "{field} must be finite, got {value}"),
            Self::InvalidProbability(value) => write!(f, "probability must be in [0,1], got {value}"),
            Self::InvalidForecastWindow { issued_at, valid_from, valid_until } => write!(
                f,
                "forecast must be issued before a non-empty validity window: issued={issued_at}, valid_from={valid_from}, valid_until={valid_until}"
            ),
            Self::DuplicateForecast(id) => write!(f, "forecast {id} already exists"),
            Self::UnknownForecast(id) => write!(f, "forecast {id} does not exist"),
            Self::DuplicateVerification(id) => write!(f, "forecast {id} is already verified"),
            Self::VerificationOutsideWindow { observed_for, valid_from, valid_until } => write!(
                f,
                "verification target time {observed_for} is outside [{valid_from}, {valid_until}]"
            ),
            Self::MissingVerificationEvidence => write!(f, "forecast verification requires evidence"),
            Self::VerificationEvidenceNotVerificationStage(id) => write!(
                f,
                "verification evidence {id} must be explicitly marked EvidenceStage::Verification"
            ),
            Self::VerificationKindMismatch => write!(f, "forecast and verification kinds do not match"),
            Self::UnitMismatch { expected, got } => write!(f, "unit mismatch: expected {expected}, got {got}"),
            Self::NoForecastsForModelTarget => write!(f, "no forecasts exist for requested model/target"),
        }
    }
}

impl Error for CalibrationError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(CalibrationError::EmptyField(field));
    }
    Ok(())
}

fn finite(value: f64, field: &'static str) -> Result<()> {
    if !value.is_finite() {
        return Err(CalibrationError::NonFinite { field, value });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForecastValue {
    Numeric(OutcomeEstimate),
    BinaryProbability(f64),
}

impl ForecastValue {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Numeric(estimate) => {
                finite(estimate.point, "numeric forecast point")?;
                if let Some(lower) = estimate.lower {
                    finite(lower, "numeric forecast lower")?;
                }
                if let Some(upper) = estimate.upper {
                    finite(upper, "numeric forecast upper")?;
                }
                non_empty(&estimate.unit, "numeric forecast unit")
            }
            Self::BinaryProbability(probability) => {
                if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                    return Err(CalibrationError::InvalidProbability(*probability));
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForecastRecord {
    pub id: String,
    pub model_id: String,
    pub model_version: String,
    pub target_id: String,
    pub scenario_id: Option<String>,
    pub issued_at_unix_ms: i64,
    pub valid_from_unix_ms: i64,
    pub valid_until_unix_ms: i64,
    pub value: ForecastValue,
    pub model_artifact_digest: Option<String>,
    pub assumptions: Vec<String>,
}

impl ForecastRecord {
    pub fn new(
        id: impl Into<String>,
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        target_id: impl Into<String>,
        issued_at_unix_ms: i64,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
        value: ForecastValue,
    ) -> Result<Self> {
        let id = id.into();
        let model_id = model_id.into();
        let model_version = model_version.into();
        let target_id = target_id.into();
        non_empty(&id, "forecast id")?;
        non_empty(&model_id, "forecast model id")?;
        non_empty(&model_version, "forecast model version")?;
        non_empty(&target_id, "forecast target id")?;
        if valid_until_unix_ms < valid_from_unix_ms || issued_at_unix_ms > valid_from_unix_ms {
            return Err(CalibrationError::InvalidForecastWindow {
                issued_at: issued_at_unix_ms,
                valid_from: valid_from_unix_ms,
                valid_until: valid_until_unix_ms,
            });
        }
        value.validate()?;
        Ok(Self {
            id,
            model_id,
            model_version,
            target_id,
            scenario_id: None,
            issued_at_unix_ms,
            valid_from_unix_ms,
            valid_until_unix_ms,
            value,
            model_artifact_digest: None,
            assumptions: Vec::new(),
        })
    }

    pub fn with_scenario_id(mut self, scenario_id: impl Into<String>) -> Result<Self> {
        let scenario_id = scenario_id.into();
        non_empty(&scenario_id, "scenario id")?;
        self.scenario_id = Some(scenario_id);
        Ok(self)
    }

    pub fn with_artifact_digest(mut self, digest: impl Into<String>) -> Result<Self> {
        let digest = digest.into();
        non_empty(&digest, "forecast model artifact digest")?;
        self.model_artifact_digest = Some(digest);
        Ok(self)
    }

    pub fn with_assumption(mut self, assumption: impl Into<String>) -> Result<Self> {
        let assumption = assumption.into();
        non_empty(&assumption, "forecast assumption")?;
        self.assumptions.push(assumption);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum VerifiedOutcome {
    Numeric { value: f64, unit: String },
    Binary(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForecastVerification {
    pub forecast_id: String,
    /// Time in the physical world to which the observed outcome applies.
    pub observed_for_unix_ms: i64,
    /// Time verification was recorded; this may be later than the target time.
    pub recorded_at_unix_ms: i64,
    pub outcome: VerifiedOutcome,
    pub evidence: Vec<EvidenceRef>,
}

impl ForecastVerification {
    pub fn new(
        forecast_id: impl Into<String>,
        observed_for_unix_ms: i64,
        recorded_at_unix_ms: i64,
        outcome: VerifiedOutcome,
        evidence: Vec<EvidenceRef>,
    ) -> Result<Self> {
        let forecast_id = forecast_id.into();
        non_empty(&forecast_id, "forecast verification id")?;
        if evidence.is_empty() {
            return Err(CalibrationError::MissingVerificationEvidence);
        }
        for reference in &evidence {
            if reference.stage != EvidenceStage::Verification {
                return Err(CalibrationError::VerificationEvidenceNotVerificationStage(
                    reference.id.clone(),
                ));
            }
        }
        match &outcome {
            VerifiedOutcome::Numeric { value, unit } => {
                finite(*value, "verified numeric outcome")?;
                non_empty(unit, "verified numeric outcome unit")?;
            }
            VerifiedOutcome::Binary(_) => {}
        }
        Ok(Self {
            forecast_id,
            observed_for_unix_ms,
            recorded_at_unix_ms,
            outcome,
            evidence,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumericForecastScore {
    pub error: f64,
    pub absolute_error: f64,
    pub squared_error: f64,
    pub interval_hit: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryForecastScore {
    pub brier_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForecastScore {
    Numeric(NumericForecastScore),
    Binary(BinaryForecastScore),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedForecast {
    pub forecast: ForecastRecord,
    pub verification: ForecastVerification,
    pub score: ForecastScore,
}

fn score(forecast: &ForecastRecord, verification: &ForecastVerification) -> Result<ForecastScore> {
    match (&forecast.value, &verification.outcome) {
        (ForecastValue::Numeric(estimate), VerifiedOutcome::Numeric { value, unit }) => {
            if &estimate.unit != unit {
                return Err(CalibrationError::UnitMismatch {
                    expected: estimate.unit.clone(),
                    got: unit.clone(),
                });
            }
            let error = estimate.point - *value;
            let interval_hit = match (estimate.lower, estimate.upper) {
                (Some(lower), Some(upper)) => Some(*value >= lower && *value <= upper),
                _ => None,
            };
            Ok(ForecastScore::Numeric(NumericForecastScore {
                error,
                absolute_error: error.abs(),
                squared_error: error * error,
                interval_hit,
            }))
        }
        (ForecastValue::BinaryProbability(probability), VerifiedOutcome::Binary(observed)) => {
            let y = if *observed { 1.0 } else { 0.0 };
            let error = probability - y;
            Ok(ForecastScore::Binary(BinaryForecastScore {
                brier_score: error * error,
            }))
        }
        _ => Err(CalibrationError::VerificationKindMismatch),
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ForecastCalibrationLedger {
    forecasts: Vec<ForecastRecord>,
    verified: Vec<VerifiedForecast>,
}

impl ForecastCalibrationLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn forecasts(&self) -> &[ForecastRecord] {
        &self.forecasts
    }

    pub fn verified(&self) -> &[VerifiedForecast] {
        &self.verified
    }

    pub fn register_forecast(&mut self, forecast: ForecastRecord) -> Result<()> {
        if self.forecasts.iter().any(|existing| existing.id == forecast.id) {
            return Err(CalibrationError::DuplicateForecast(forecast.id));
        }
        self.forecasts.push(forecast);
        Ok(())
    }

    pub fn verify(&mut self, verification: ForecastVerification) -> Result<&VerifiedForecast> {
        if self
            .verified
            .iter()
            .any(|existing| existing.forecast.id == verification.forecast_id)
        {
            return Err(CalibrationError::DuplicateVerification(
                verification.forecast_id,
            ));
        }

        let Some(forecast) = self
            .forecasts
            .iter()
            .find(|forecast| forecast.id == verification.forecast_id)
            .cloned()
        else {
            return Err(CalibrationError::UnknownForecast(
                verification.forecast_id,
            ));
        };

        if verification.observed_for_unix_ms < forecast.valid_from_unix_ms
            || verification.observed_for_unix_ms > forecast.valid_until_unix_ms
        {
            return Err(CalibrationError::VerificationOutsideWindow {
                observed_for: verification.observed_for_unix_ms,
                valid_from: forecast.valid_from_unix_ms,
                valid_until: forecast.valid_until_unix_ms,
            });
        }

        let score = score(&forecast, &verification)?;
        self.verified.push(VerifiedForecast {
            forecast,
            verification,
            score,
        });
        Ok(self.verified.last().expect("just pushed verified forecast"))
    }

    pub fn pending_forecast_ids(&self) -> Vec<&str> {
        let verified: HashSet<&str> = self
            .verified
            .iter()
            .map(|entry| entry.forecast.id.as_str())
            .collect();
        self.forecasts
            .iter()
            .filter(|forecast| !verified.contains(forecast.id.as_str()))
            .map(|forecast| forecast.id.as_str())
            .collect()
    }

    pub fn report(&self, model_id: &str, target_id: &str) -> Result<ModelTargetCalibrationReport> {
        let all: Vec<&ForecastRecord> = self
            .forecasts
            .iter()
            .filter(|forecast| forecast.model_id == model_id && forecast.target_id == target_id)
            .collect();
        if all.is_empty() {
            return Err(CalibrationError::NoForecastsForModelTarget);
        }

        let verified: Vec<&VerifiedForecast> = self
            .verified
            .iter()
            .filter(|entry| entry.forecast.model_id == model_id && entry.forecast.target_id == target_id)
            .collect();

        let mut numeric_count = 0usize;
        let mut abs_error_sum = 0.0;
        let mut squared_error_sum = 0.0;
        let mut interval_scored = 0usize;
        let mut interval_hits = 0usize;
        let mut binary_count = 0usize;
        let mut brier_sum = 0.0;

        for entry in &verified {
            match &entry.score {
                ForecastScore::Numeric(score) => {
                    numeric_count += 1;
                    abs_error_sum += score.absolute_error;
                    squared_error_sum += score.squared_error;
                    if let Some(hit) = score.interval_hit {
                        interval_scored += 1;
                        if hit {
                            interval_hits += 1;
                        }
                    }
                }
                ForecastScore::Binary(score) => {
                    binary_count += 1;
                    brier_sum += score.brier_score;
                }
            }
        }

        Ok(ModelTargetCalibrationReport {
            model_id: model_id.to_string(),
            target_id: target_id.to_string(),
            total_forecasts: all.len(),
            verified_forecasts: verified.len(),
            pending_forecasts: all.len() - verified.len(),
            numeric_count,
            mean_absolute_error: (numeric_count > 0).then(|| abs_error_sum / numeric_count as f64),
            root_mean_squared_error: (numeric_count > 0)
                .then(|| (squared_error_sum / numeric_count as f64).sqrt()),
            interval_scored,
            interval_coverage: (interval_scored > 0)
                .then(|| interval_hits as f64 / interval_scored as f64),
            binary_count,
            mean_brier_score: (binary_count > 0).then(|| brier_sum / binary_count as f64),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelTargetCalibrationReport {
    pub model_id: String,
    pub target_id: String,
    pub total_forecasts: usize,
    pub verified_forecasts: usize,
    pub pending_forecasts: usize,
    pub numeric_count: usize,
    pub mean_absolute_error: Option<f64>,
    pub root_mean_squared_error: Option<f64>,
    pub interval_scored: usize,
    pub interval_coverage: Option<f64>,
    pub binary_count: usize,
    pub mean_brier_score: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verification_ref(id: &str) -> EvidenceRef {
        EvidenceRef::new(id, EvidenceStage::Verification).unwrap()
    }

    fn numeric_forecast(id: &str, point: f64, lower: f64, upper: f64) -> ForecastRecord {
        ForecastRecord::new(
            id,
            "watershed-model",
            "1.0",
            "wetland-area",
            0,
            100,
            200,
            ForecastValue::Numeric(OutcomeEstimate::interval(point, lower, upper, "ha").unwrap()),
        )
        .unwrap()
    }

    #[test]
    fn forecast_cannot_be_registered_after_its_window_starts() {
        assert!(matches!(
            ForecastRecord::new(
                "late",
                "m",
                "1",
                "target",
                101,
                100,
                200,
                ForecastValue::BinaryProbability(0.7),
            ),
            Err(CalibrationError::InvalidForecastWindow { .. })
        ));
    }

    #[test]
    fn verification_requires_explicit_verification_stage_evidence() {
        let ordinary = EvidenceRef::new("obs", EvidenceStage::Observation).unwrap();
        assert_eq!(
            ForecastVerification::new(
                "f",
                150,
                210,
                VerifiedOutcome::Binary(true),
                vec![ordinary],
            )
            .unwrap_err(),
            CalibrationError::VerificationEvidenceNotVerificationStage("obs".into())
        );
    }

    #[test]
    fn numeric_hit_and_error_are_recorded_without_a_trust_score() {
        let mut ledger = ForecastCalibrationLedger::new();
        ledger.register_forecast(numeric_forecast("f1", 100.0, 90.0, 110.0)).unwrap();
        ledger
            .verify(
                ForecastVerification::new(
                    "f1",
                    150,
                    220,
                    VerifiedOutcome::Numeric {
                        value: 105.0,
                        unit: "ha".into(),
                    },
                    vec![verification_ref("v1")],
                )
                .unwrap(),
            )
            .unwrap();

        let ForecastScore::Numeric(score) = &ledger.verified()[0].score else {
            panic!("expected numeric score");
        };
        assert_eq!(score.absolute_error, 5.0);
        assert_eq!(score.interval_hit, Some(true));
    }

    #[test]
    fn binary_forecasts_use_brier_score() {
        let mut ledger = ForecastCalibrationLedger::new();
        ledger
            .register_forecast(
                ForecastRecord::new(
                    "f1",
                    "flood-model",
                    "1.0",
                    "flood-next-pass",
                    0,
                    100,
                    200,
                    ForecastValue::BinaryProbability(0.8),
                )
                .unwrap(),
            )
            .unwrap();
        ledger
            .verify(
                ForecastVerification::new(
                    "f1",
                    150,
                    220,
                    VerifiedOutcome::Binary(true),
                    vec![verification_ref("v1")],
                )
                .unwrap(),
            )
            .unwrap();

        let ForecastScore::Binary(score) = &ledger.verified()[0].score else {
            panic!("expected binary score");
        };
        assert!((score.brier_score - 0.04).abs() < 1e-12);
    }

    #[test]
    fn bad_forecasts_remain_in_calibration_history() {
        let mut ledger = ForecastCalibrationLedger::new();
        ledger.register_forecast(numeric_forecast("good", 100.0, 90.0, 110.0)).unwrap();
        ledger.register_forecast(numeric_forecast("bad", 200.0, 190.0, 210.0)).unwrap();

        for (id, observed, evidence) in [("good", 100.0, "vg"), ("bad", 100.0, "vb")] {
            ledger
                .verify(
                    ForecastVerification::new(
                        id,
                        150,
                        220,
                        VerifiedOutcome::Numeric {
                            value: observed,
                            unit: "ha".into(),
                        },
                        vec![verification_ref(evidence)],
                    )
                    .unwrap(),
                )
                .unwrap();
        }

        let report = ledger.report("watershed-model", "wetland-area").unwrap();
        assert_eq!(report.total_forecasts, 2);
        assert_eq!(report.verified_forecasts, 2);
        assert!((report.mean_absolute_error.unwrap() - 50.0).abs() < 1e-12);
        assert!((report.interval_coverage.unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pending_forecasts_are_visible_not_dropped() {
        let mut ledger = ForecastCalibrationLedger::new();
        ledger.register_forecast(numeric_forecast("pending", 100.0, 90.0, 110.0)).unwrap();
        assert_eq!(ledger.pending_forecast_ids(), vec!["pending"]);
        let report = ledger.report("watershed-model", "wetland-area").unwrap();
        assert_eq!(report.pending_forecasts, 1);
        assert_eq!(report.verified_forecasts, 0);
    }
}
