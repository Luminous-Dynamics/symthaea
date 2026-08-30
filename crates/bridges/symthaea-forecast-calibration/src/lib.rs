//! Physical-world forecast provenance adapter for the Symthaea Futures Laboratory.
//!
//! Planetary Perception does not define a second forecast language or scoring
//! system. Forecast distributions, abstentions, and proper scoring rules come
//! from `symthaea-futures-core` / `symthaea-futures-calibration`. This bridge
//! adds Earth-facing wall-clock binding and verification evidence.

use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_earth_observation::{EvidenceRef, EvidenceStage};
use symthaea_futures_calibration::{
    BrierScore, Crps, FiniteScore, LogScore, ScoringRule, ScoringRuleKind,
};
use symthaea_futures_core::{
    AbstentionReason, ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion,
};

pub type Result<T> = std::result::Result<T, CalibrationBridgeError>;

#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationBridgeError {
    EmptyField(&'static str),
    ZeroTickDuration,
    ClockOverflow,
    InvalidVerificationWindow { valid_from: i64, valid_until: i64 },
    TargetOutsideVerificationWindow { target: i64, valid_from: i64, valid_until: i64 },
    DistributionTickMismatch { record_tick: u64, distribution_tick: u64 },
    DistributionHorizonMismatch { record_horizon: u64, distribution_horizon: u64 },
    DuplicateForecast(String),
    UnknownForecast(String),
    DuplicateResolution(String),
    VerificationOutsideWindow { observed_for: i64, valid_from: i64, valid_until: i64 },
    MissingVerificationEvidence,
    VerificationEvidenceNotVerificationStage(String),
    Scoring(String),
    NoForecastsForModelTarget,
}

impl Display for CalibrationBridgeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::ZeroTickDuration => write!(f, "physical forecast tick duration must be > 0 ms"),
            Self::ClockOverflow => write!(f, "physical forecast clock mapping overflowed i64 milliseconds"),
            Self::InvalidVerificationWindow { valid_from, valid_until } => write!(
                f,
                "verification window requires valid_from <= valid_until, got {valid_from}..={valid_until}"
            ),
            Self::TargetOutsideVerificationWindow { target, valid_from, valid_until } => write!(
                f,
                "forecast target time {target} lies outside verification window {valid_from}..={valid_until}"
            ),
            Self::DistributionTickMismatch { record_tick, distribution_tick } => write!(
                f,
                "record issued tick {record_tick} differs from canonical forecast tick {distribution_tick}"
            ),
            Self::DistributionHorizonMismatch { record_horizon, distribution_horizon } => write!(
                f,
                "record horizon {record_horizon} differs from canonical forecast horizon {distribution_horizon}"
            ),
            Self::DuplicateForecast(id) => write!(f, "forecast {id} already exists"),
            Self::UnknownForecast(id) => write!(f, "forecast {id} does not exist"),
            Self::DuplicateResolution(id) => write!(f, "forecast {id} is already resolved"),
            Self::VerificationOutsideWindow { observed_for, valid_from, valid_until } => write!(
                f,
                "verification target time {observed_for} is outside {valid_from}..={valid_until}"
            ),
            Self::MissingVerificationEvidence => write!(f, "forecast verification requires explicit verification evidence"),
            Self::VerificationEvidenceNotVerificationStage(id) => write!(
                f,
                "verification evidence {id} must be EvidenceStage::Verification"
            ),
            Self::Scoring(message) => write!(f, "Futures Laboratory scoring failed: {message}"),
            Self::NoForecastsForModelTarget => write!(f, "no forecasts exist for requested model/target/rule"),
        }
    }
}

impl Error for CalibrationBridgeError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(CalibrationBridgeError::EmptyField(field));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalTimeBinding {
    pub tick_zero_unix_ms: i64,
    pub tick_duration_ms: u64,
}

impl PhysicalTimeBinding {
    pub fn new(tick_zero_unix_ms: i64, tick_duration_ms: u64) -> Result<Self> {
        if tick_duration_ms == 0 {
            return Err(CalibrationBridgeError::ZeroTickDuration);
        }
        Ok(Self { tick_zero_unix_ms, tick_duration_ms })
    }

    pub fn unix_ms_for_tick(&self, tick: u64) -> Result<i64> {
        let value = i128::from(self.tick_zero_unix_ms)
            + i128::from(tick) * i128::from(self.tick_duration_ms);
        i64::try_from(value).map_err(|_| CalibrationBridgeError::ClockOverflow)
    }
}

#[derive(Debug, Clone)]
pub struct PhysicalForecastRecord {
    pub id: String,
    pub model_id: String,
    pub model_version: String,
    pub target_id: String,
    pub scenario_id: Option<String>,
    pub issued_tick: u64,
    pub horizon: Horizon,
    pub clock: PhysicalTimeBinding,
    pub valid_from_unix_ms: i64,
    pub valid_until_unix_ms: i64,
    pub output: ForecastOutput,
    pub scoring_rule: ScoringRuleKind,
    pub model_artifact_digest: Option<String>,
    pub assumptions: Vec<String>,
}

impl PhysicalForecastRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        target_id: impl Into<String>,
        issued_tick: u64,
        horizon: Horizon,
        clock: PhysicalTimeBinding,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
        output: ForecastOutput,
        scoring_rule: ScoringRuleKind,
    ) -> Result<Self> {
        let id = id.into();
        let model_id = model_id.into();
        let model_version = model_version.into();
        let target_id = target_id.into();
        non_empty(&id, "forecast id")?;
        non_empty(&model_id, "forecast model id")?;
        non_empty(&model_version, "forecast model version")?;
        non_empty(&target_id, "forecast target id")?;
        if valid_until_unix_ms < valid_from_unix_ms {
            return Err(CalibrationBridgeError::InvalidVerificationWindow {
                valid_from: valid_from_unix_ms,
                valid_until: valid_until_unix_ms,
            });
        }

        let target_tick = issued_tick.checked_add(horizon.0).ok_or(CalibrationBridgeError::ClockOverflow)?;
        let target_unix_ms = clock.unix_ms_for_tick(target_tick)?;
        if target_unix_ms < valid_from_unix_ms || target_unix_ms > valid_until_unix_ms {
            return Err(CalibrationBridgeError::TargetOutsideVerificationWindow {
                target: target_unix_ms,
                valid_from: valid_from_unix_ms,
                valid_until: valid_until_unix_ms,
            });
        }

        if let ForecastOutput::Distribution(distribution) = &output {
            if distribution.issued_at_tick() != issued_tick {
                return Err(CalibrationBridgeError::DistributionTickMismatch {
                    record_tick: issued_tick,
                    distribution_tick: distribution.issued_at_tick(),
                });
            }
            if distribution.horizon() != horizon {
                return Err(CalibrationBridgeError::DistributionHorizonMismatch {
                    record_horizon: horizon.0,
                    distribution_horizon: distribution.horizon().0,
                });
            }
        }

        Ok(Self {
            id,
            model_id,
            model_version,
            target_id,
            scenario_id: None,
            issued_tick,
            horizon,
            clock,
            valid_from_unix_ms,
            valid_until_unix_ms,
            output,
            scoring_rule,
            model_artifact_digest: None,
            assumptions: Vec::new(),
        })
    }

    pub fn issued_at_unix_ms(&self) -> Result<i64> {
        self.clock.unix_ms_for_tick(self.issued_tick)
    }

    pub fn target_unix_ms(&self) -> Result<i64> {
        let target_tick = self.issued_tick.checked_add(self.horizon.0).ok_or(CalibrationBridgeError::ClockOverflow)?;
        self.clock.unix_ms_for_tick(target_tick)
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

#[derive(Debug, Clone)]
pub struct PhysicalForecastVerification {
    pub forecast_id: String,
    pub observed_for_unix_ms: i64,
    pub recorded_at_unix_ms: i64,
    pub actual: OutcomeRegion,
    pub evidence: Vec<EvidenceRef>,
}

impl PhysicalForecastVerification {
    pub fn new(
        forecast_id: impl Into<String>,
        observed_for_unix_ms: i64,
        recorded_at_unix_ms: i64,
        actual: OutcomeRegion,
        evidence: Vec<EvidenceRef>,
    ) -> Result<Self> {
        let forecast_id = forecast_id.into();
        non_empty(&forecast_id, "forecast verification id")?;
        if evidence.is_empty() {
            return Err(CalibrationBridgeError::MissingVerificationEvidence);
        }
        for reference in &evidence {
            if reference.stage != EvidenceStage::Verification {
                return Err(CalibrationBridgeError::VerificationEvidenceNotVerificationStage(reference.id.clone()));
            }
        }
        Ok(Self { forecast_id, observed_for_unix_ms, recorded_at_unix_ms, actual, evidence })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForecastResolutionScore {
    ProperScore(FiniteScore),
    Abstained(AbstentionReason),
}

#[derive(Debug, Clone)]
pub struct ResolvedPhysicalForecast {
    pub forecast: PhysicalForecastRecord,
    pub verification: PhysicalForecastVerification,
    pub resolution: ForecastResolutionScore,
}

fn proper_score(rule: ScoringRuleKind, forecast: &ForecastDistribution, actual: &OutcomeRegion) -> Result<FiniteScore> {
    let score = match rule {
        ScoringRuleKind::Brier => BrierScore.score(forecast, actual),
        ScoringRuleKind::Crps => Crps.score(forecast, actual),
        ScoringRuleKind::LogScore => LogScore::default().score(forecast, actual),
    };
    score.map_err(|error| CalibrationBridgeError::Scoring(error.to_string()))
}

#[derive(Debug, Clone, Default)]
pub struct PhysicalForecastLedger {
    forecasts: Vec<PhysicalForecastRecord>,
    resolved: Vec<ResolvedPhysicalForecast>,
}

impl PhysicalForecastLedger {
    pub fn new() -> Self { Self::default() }
    pub fn forecasts(&self) -> &[PhysicalForecastRecord] { &self.forecasts }
    pub fn resolved(&self) -> &[ResolvedPhysicalForecast] { &self.resolved }

    pub fn register(&mut self, forecast: PhysicalForecastRecord) -> Result<()> {
        if self.forecasts.iter().any(|existing| existing.id == forecast.id) {
            return Err(CalibrationBridgeError::DuplicateForecast(forecast.id));
        }
        self.forecasts.push(forecast);
        Ok(())
    }

    pub fn resolve(&mut self, verification: PhysicalForecastVerification) -> Result<&ResolvedPhysicalForecast> {
        if self.resolved.iter().any(|existing| existing.forecast.id == verification.forecast_id) {
            return Err(CalibrationBridgeError::DuplicateResolution(verification.forecast_id));
        }
        let Some(forecast) = self.forecasts.iter().find(|forecast| forecast.id == verification.forecast_id).cloned() else {
            return Err(CalibrationBridgeError::UnknownForecast(verification.forecast_id));
        };
        if verification.observed_for_unix_ms < forecast.valid_from_unix_ms || verification.observed_for_unix_ms > forecast.valid_until_unix_ms {
            return Err(CalibrationBridgeError::VerificationOutsideWindow {
                observed_for: verification.observed_for_unix_ms,
                valid_from: forecast.valid_from_unix_ms,
                valid_until: forecast.valid_until_unix_ms,
            });
        }
        let resolution = match &forecast.output {
            ForecastOutput::Distribution(distribution) => ForecastResolutionScore::ProperScore(
                proper_score(forecast.scoring_rule, distribution, &verification.actual)?,
            ),
            ForecastOutput::Abstain(reason) => ForecastResolutionScore::Abstained(*reason),
        };
        self.resolved.push(ResolvedPhysicalForecast { forecast, verification, resolution });
        Ok(self.resolved.last().expect("resolved forecast was just pushed"))
    }

    pub fn pending_forecast_ids(&self) -> Vec<&str> {
        self.forecasts.iter()
            .filter(|forecast| !self.resolved.iter().any(|resolved| resolved.forecast.id == forecast.id))
            .map(|forecast| forecast.id.as_str())
            .collect()
    }

    pub fn report(&self, model_id: &str, target_id: &str, scoring_rule: ScoringRuleKind) -> Result<PhysicalForecastReport> {
        let all: Vec<&PhysicalForecastRecord> = self.forecasts.iter()
            .filter(|forecast| forecast.model_id == model_id && forecast.target_id == target_id && forecast.scoring_rule == scoring_rule)
            .collect();
        if all.is_empty() {
            return Err(CalibrationBridgeError::NoForecastsForModelTarget);
        }
        let resolved: Vec<&ResolvedPhysicalForecast> = self.resolved.iter()
            .filter(|entry| entry.forecast.model_id == model_id && entry.forecast.target_id == target_id && entry.forecast.scoring_rule == scoring_rule)
            .collect();
        let scores: Vec<f64> = resolved.iter().filter_map(|entry| match entry.resolution {
            ForecastResolutionScore::ProperScore(score) => Some(score.get()),
            ForecastResolutionScore::Abstained(_) => None,
        }).collect();
        let abstained = resolved.len() - scores.len();
        let mean_proper_score = (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64);
        Ok(PhysicalForecastReport {
            model_id: model_id.to_string(),
            target_id: target_id.to_string(),
            scoring_rule,
            total_forecasts: all.len(),
            resolved_forecasts: resolved.len(),
            pending_forecasts: all.len() - resolved.len(),
            scored_forecasts: scores.len(),
            abstained_forecasts: abstained,
            mean_proper_score,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalForecastReport {
    pub model_id: String,
    pub target_id: String,
    pub scoring_rule: ScoringRuleKind,
    pub total_forecasts: usize,
    pub resolved_forecasts: usize,
    pub pending_forecasts: usize,
    pub scored_forecasts: usize,
    pub abstained_forecasts: usize,
    pub mean_proper_score: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_futures_core::{ForecastDistribution, OutcomeSpaceId};

    fn clock() -> PhysicalTimeBinding { PhysicalTimeBinding::new(1_000, 100).unwrap() }
    fn evidence(stage: EvidenceStage) -> EvidenceRef { EvidenceRef::new("verify-1", stage).unwrap() }

    fn boolean_distribution() -> ForecastDistribution {
        ForecastDistribution::try_from_raw(
            2,
            Horizon(3),
            OutcomeSpaceId("flood-within-horizon".into()),
            vec![
                (0.8, OutcomeRegion::Boolean(true), vec![]),
                (0.2, OutcomeRegion::Boolean(false), vec![]),
            ],
            0.0,
        ).unwrap()
    }

    fn record(output: ForecastOutput) -> PhysicalForecastRecord {
        PhysicalForecastRecord::new(
            "forecast-1", "wetland-model", "0.1.0", "flood-within-horizon",
            2, Horizon(3), clock(), 1_490, 1_510, output, ScoringRuleKind::Brier,
        ).unwrap()
    }

    #[test]
    fn physical_clock_binding_is_explicit_and_checked() {
        assert_eq!(clock().unix_ms_for_tick(5).unwrap(), 1_500);
        assert_eq!(PhysicalTimeBinding::new(0, 0).unwrap_err(), CalibrationBridgeError::ZeroTickDuration);
    }

    #[test]
    fn canonical_distribution_tick_must_match_physical_record() {
        let distribution = ForecastDistribution::try_from_raw(
            3, Horizon(3), OutcomeSpaceId("flood-within-horizon".into()),
            vec![(0.5, OutcomeRegion::Boolean(true), vec![]), (0.5, OutcomeRegion::Boolean(false), vec![])], 0.0,
        ).unwrap();
        assert!(matches!(
            PhysicalForecastRecord::new(
                "forecast-x", "model", "1", "target", 2, Horizon(3), clock(), 1_490, 1_510,
                ForecastOutput::Distribution(distribution), ScoringRuleKind::Brier,
            ),
            Err(CalibrationBridgeError::DistributionTickMismatch { .. })
        ));
    }

    #[test]
    fn uses_futures_lab_multi_class_brier_convention() {
        let mut ledger = PhysicalForecastLedger::new();
        ledger.register(record(ForecastOutput::Distribution(boolean_distribution()))).unwrap();
        let verification = PhysicalForecastVerification::new(
            "forecast-1", 1_500, 1_600, OutcomeRegion::Boolean(true),
            vec![evidence(EvidenceStage::Verification)],
        ).unwrap();
        let resolved = ledger.resolve(verification).unwrap();
        let ForecastResolutionScore::ProperScore(score) = resolved.resolution else { panic!("expected proper score") };
        assert!((score.get() - 0.08).abs() < 1e-12);
    }

    #[test]
    fn verification_requires_verification_stage_evidence() {
        assert!(matches!(
            PhysicalForecastVerification::new(
                "forecast-1", 1_500, 1_600, OutcomeRegion::Boolean(true),
                vec![evidence(EvidenceStage::Observation)],
            ),
            Err(CalibrationBridgeError::VerificationEvidenceNotVerificationStage(_))
        ));
    }

    #[test]
    fn abstention_is_retained_and_not_scored_as_failure_sentinel() {
        let mut ledger = PhysicalForecastLedger::new();
        ledger.register(record(ForecastOutput::Abstain(AbstentionReason::OutOfDistributionScenario))).unwrap();
        ledger.resolve(PhysicalForecastVerification::new(
            "forecast-1", 1_500, 1_600, OutcomeRegion::Boolean(true),
            vec![evidence(EvidenceStage::Verification)],
        ).unwrap()).unwrap();
        let report = ledger.report("wetland-model", "flood-within-horizon", ScoringRuleKind::Brier).unwrap();
        assert_eq!(report.scored_forecasts, 0);
        assert_eq!(report.abstained_forecasts, 1);
        assert_eq!(report.mean_proper_score, None);
    }

    #[test]
    fn pending_forecasts_remain_visible() {
        let mut ledger = PhysicalForecastLedger::new();
        ledger.register(record(ForecastOutput::Distribution(boolean_distribution()))).unwrap();
        assert_eq!(ledger.pending_forecast_ids(), vec!["forecast-1"]);
    }
}
