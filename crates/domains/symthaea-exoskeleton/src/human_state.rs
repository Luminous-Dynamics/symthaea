// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-preserving human-state twin for Space Exosuit research (SX-023).
//!
//! This module fuses observations for simulation, diagnostics, and planning only.
//! It is not a medical monitor and exposes no actuator or life-support commands.
//! Missing, stale, invalid, or contradictory inputs remain explicitly unknown;
//! this module never manufactures a nominal/healthy state.

use serde::{Deserialize, Serialize};

pub const HUMAN_METRIC_COUNT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HumanMetric {
    HeartRateBpm,
    RespiratoryRateBpm,
    CoreTemperatureC,
    SkinTemperatureC,
    PotableWaterRemainingMl,
    LocalFatigueFraction,
    WholeBodyMetabolicPowerW,
    PerceivedExertion0To10,
}

impl HumanMetric {
    pub const ALL: [Self; HUMAN_METRIC_COUNT] = [
        Self::HeartRateBpm,
        Self::RespiratoryRateBpm,
        Self::CoreTemperatureC,
        Self::SkinTemperatureC,
        Self::PotableWaterRemainingMl,
        Self::LocalFatigueFraction,
        Self::WholeBodyMetabolicPowerW,
        Self::PerceivedExertion0To10,
    ];

    /// Input-domain guardrails, not medical thresholds or healthy ranges.
    fn accepts_value(self, value: f64) -> bool {
        match self {
            Self::HeartRateBpm => (0.0..=500.0).contains(&value),
            Self::RespiratoryRateBpm => (0.0..=300.0).contains(&value),
            Self::CoreTemperatureC | Self::SkinTemperatureC => (-100.0..=100.0).contains(&value),
            Self::PotableWaterRemainingMl | Self::WholeBodyMetabolicPowerW => value >= 0.0,
            Self::LocalFatigueFraction => (0.0..=1.0).contains(&value),
            Self::PerceivedExertion0To10 => (0.0..=10.0).contains(&value),
        }
    }

    fn disagreement_tolerance(self, tolerances: HumanStateTolerances) -> f64 {
        match self {
            Self::HeartRateBpm => tolerances.heart_rate_bpm,
            Self::RespiratoryRateBpm => tolerances.respiratory_rate_bpm,
            Self::CoreTemperatureC => tolerances.core_temperature_c,
            Self::SkinTemperatureC => tolerances.skin_temperature_c,
            Self::PotableWaterRemainingMl => tolerances.potable_water_ml,
            Self::LocalFatigueFraction => tolerances.local_fatigue_fraction,
            Self::WholeBodyMetabolicPowerW => tolerances.metabolic_power_w,
            Self::PerceivedExertion0To10 => tolerances.perceived_exertion,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanObservationSource {
    Instrument,
    WearerReported,
    ModelEstimate,
    ModelPrediction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanStateEvidenceKind {
    Observed,
    WearerReported,
    Estimated,
    Predicted,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanStateObservation {
    pub metric: HumanMetric,
    pub value: f64,
    /// Absolute uncertainty in the metric's declared unit, never a confidence interval.
    pub uncertainty_abs: f64,
    /// Mission-elapsed monotonic seconds, not a wall-clock timestamp.
    pub observed_at_s: f64,
    /// Producer-supplied confidence in [0,1]; not a calibration guarantee.
    pub confidence: f64,
    pub source_id: String,
    pub source: HumanObservationSource,
}

impl HumanStateObservation {
    pub fn is_valid(&self) -> bool {
        self.metric.accepts_value(self.value)
            && self.value.is_finite()
            && self.uncertainty_abs.is_finite()
            && self.uncertainty_abs >= 0.0
            && (self.value - self.uncertainty_abs).is_finite()
            && (self.value + self.uncertainty_abs).is_finite()
            && self.observed_at_s.is_finite()
            && self.observed_at_s >= 0.0
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && !self.source_id.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanStateTolerances {
    pub heart_rate_bpm: f64,
    pub respiratory_rate_bpm: f64,
    pub core_temperature_c: f64,
    pub skin_temperature_c: f64,
    pub potable_water_ml: f64,
    pub local_fatigue_fraction: f64,
    pub metabolic_power_w: f64,
    pub perceived_exertion: f64,
}

impl HumanStateTolerances {
    fn is_valid(self) -> bool {
        [
            self.heart_rate_bpm,
            self.respiratory_rate_bpm,
            self.core_temperature_c,
            self.skin_temperature_c,
            self.potable_water_ml,
            self.local_fatigue_fraction,
            self.metabolic_power_w,
            self.perceived_exertion,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanStateConfig {
    pub maximum_observation_age_s: f64,
    pub minimum_confidence: f64,
    pub disagreement_tolerances: HumanStateTolerances,
}

impl HumanStateConfig {
    /// Explicitly illustrative software-simulation settings, not suit or medical limits.
    pub fn simulation_reference() -> Self {
        Self {
            maximum_observation_age_s: 5.0,
            minimum_confidence: 0.70,
            disagreement_tolerances: HumanStateTolerances {
                heart_rate_bpm: 15.0,
                respiratory_rate_bpm: 8.0,
                core_temperature_c: 0.8,
                skin_temperature_c: 3.0,
                potable_water_ml: 100.0,
                local_fatigue_fraction: 0.20,
                metabolic_power_w: 100.0,
                perceived_exertion: 2.0,
            },
        }
    }

    pub fn is_valid(&self) -> bool {
        self.maximum_observation_age_s.is_finite()
            && self.maximum_observation_age_s >= 0.0
            && self.minimum_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_confidence)
            && self.disagreement_tolerances.is_valid()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanMetricStatus {
    Fresh,
    Conflicted,
    Stale,
    LowConfidence,
    Invalid,
    InsufficientEvidence,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedHumanMetric {
    pub metric: HumanMetric,
    pub status: HumanMetricStatus,
    /// None for unknown, invalid, or conflicted evidence. Never defaults to zero.
    pub value: Option<f64>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub confidence: Option<f64>,
    pub evidence_kind: HumanStateEvidenceKind,
    pub source_ids: Vec<String>,
    /// Kept separate from instrument/model fusion; self-report is not silently erased.
    pub wearer_reported_values: Vec<f64>,
}

impl FusedHumanMetric {
    fn unknown(metric: HumanMetric, status: HumanMetricStatus) -> Self {
        Self {
            metric,
            status,
            value: None,
            lower_bound: None,
            upper_bound: None,
            confidence: None,
            evidence_kind: HumanStateEvidenceKind::Unknown,
            source_ids: Vec::new(),
            wearer_reported_values: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanStateFusionReport {
    pub evaluated_at_s: f64,
    pub metrics: Vec<FusedHumanMetric>,
    pub rejected_invalid_observations: usize,
    pub rejected_future_dated_observations: usize,
    pub excluded_stale_observations: usize,
    pub excluded_low_confidence_observations: usize,
}

impl HumanStateFusionReport {
    pub fn metric(&self, metric: HumanMetric) -> Option<&FusedHumanMetric> {
        self.metrics.iter().find(|entry| entry.metric == metric)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanStateError {
    InvalidConfig,
    InvalidEvaluationTime,
}

/// Fuses only fresh, valid, sufficiently confident evidence.
///
/// Readings from one source are reduced to that source's latest reading for each
/// metric, so a time series from one sensor is not incorrectly treated as
/// simultaneous independent evidence. Conflicting sources are surfaced, not
/// averaged into a deceptively reassuring value.
pub fn fuse_human_state(
    observations: &[HumanStateObservation],
    evaluated_at_s: f64,
    config: HumanStateConfig,
) -> Result<HumanStateFusionReport, HumanStateError> {
    if !config.is_valid() {
        return Err(HumanStateError::InvalidConfig);
    }
    if !evaluated_at_s.is_finite() || evaluated_at_s < 0.0 {
        return Err(HumanStateError::InvalidEvaluationTime);
    }

    let mut rejected_invalid = 0;
    let mut rejected_future = 0;
    let mut excluded_stale = 0;
    let mut excluded_low_confidence = 0;
    let mut fused_metrics = Vec::with_capacity(HUMAN_METRIC_COUNT);

    for metric in HumanMetric::ALL {
        let mut accepted: Vec<HumanStateObservation> = Vec::new();
        let mut saw_any = false;
        let mut saw_invalid = false;
        let mut saw_future = false;
        let mut saw_stale = false;
        let mut saw_low_confidence = false;

        for observation in observations.iter().filter(|item| item.metric == metric) {
            saw_any = true;
            if !observation.is_valid() {
                rejected_invalid += 1;
                saw_invalid = true;
                continue;
            }
            if observation.observed_at_s > evaluated_at_s {
                rejected_future += 1;
                saw_future = true;
                continue;
            }
            if evaluated_at_s - observation.observed_at_s > config.maximum_observation_age_s {
                excluded_stale += 1;
                saw_stale = true;
                continue;
            }
            if observation.confidence < config.minimum_confidence || observation.confidence <= 0.0 {
                excluded_low_confidence += 1;
                saw_low_confidence = true;
                continue;
            }

            if let Some(existing) = accepted
                .iter_mut()
                .find(|item| item.source_id == observation.source_id)
            {
                if observation.observed_at_s > existing.observed_at_s {
                    *existing = observation.clone();
                }
            } else {
                accepted.push(observation.clone());
            }
        }

        if accepted.is_empty() {
            let status = match (saw_invalid || saw_future, saw_stale, saw_low_confidence) {
                (true, false, false) => HumanMetricStatus::Invalid,
                (false, true, false) => HumanMetricStatus::Stale,
                (false, false, true) => HumanMetricStatus::LowConfidence,
                (false, false, false) if !saw_any => HumanMetricStatus::Missing,
                _ => HumanMetricStatus::InsufficientEvidence,
            };
            fused_metrics.push(FusedHumanMetric::unknown(metric, status));
            continue;
        }

        let tolerance = metric.disagreement_tolerance(config.disagreement_tolerances);
        let conflict = accepted.iter().enumerate().any(|(index, left)| {
            accepted.iter().skip(index + 1).any(|right| {
                (left.value - right.value).abs()
                    > tolerance + left.uncertainty_abs + right.uncertainty_abs
            })
        });

        let mut source_ids: Vec<String> =
            accepted.iter().map(|item| item.source_id.clone()).collect();
        source_ids.sort();
        source_ids.dedup();
        let mut wearer_reported_values: Vec<f64> = accepted
            .iter()
            .filter(|item| item.source == HumanObservationSource::WearerReported)
            .map(|item| item.value)
            .collect();
        wearer_reported_values.sort_by(f64::total_cmp);

        if conflict {
            fused_metrics.push(FusedHumanMetric {
                metric,
                status: HumanMetricStatus::Conflicted,
                value: None,
                lower_bound: None,
                upper_bound: None,
                confidence: None,
                evidence_kind: HumanStateEvidenceKind::Unknown,
                source_ids,
                wearer_reported_values,
            });
            continue;
        }

        // Incremental confidence-weighted mean avoids summing large raw readings.
        let mut estimate = accepted[0].value;
        let mut total_weight = accepted[0].confidence;
        let mut lower = accepted[0].value - accepted[0].uncertainty_abs;
        let mut upper = accepted[0].value + accepted[0].uncertainty_abs;
        let mut conservative_confidence = accepted[0].confidence;
        let mut invalid_weight = false;
        for item in accepted.iter().skip(1) {
            let next_weight = total_weight + item.confidence;
            if !next_weight.is_finite() || next_weight <= 0.0 {
                rejected_invalid += accepted.len();
                invalid_weight = true;
                break;
            }
            estimate = estimate * (total_weight / next_weight)
                + item.value * (item.confidence / next_weight);
            total_weight = next_weight;
            lower = lower.min(item.value - item.uncertainty_abs);
            upper = upper.max(item.value + item.uncertainty_abs);
            conservative_confidence = conservative_confidence.min(item.confidence);
        }
        if invalid_weight {
            fused_metrics.push(FusedHumanMetric::unknown(
                metric,
                HumanMetricStatus::Invalid,
            ));
            continue;
        }

        // Derived overflow is invalid evidence, not a reason to emit infinity.
        let uncertainty = (estimate - lower).max(upper - estimate);
        if !estimate.is_finite()
            || !lower.is_finite()
            || !upper.is_finite()
            || !uncertainty.is_finite()
            || estimate < lower
            || estimate > upper
        {
            fused_metrics.push(FusedHumanMetric::unknown(
                metric,
                HumanMetricStatus::Invalid,
            ));
            continue;
        }

        let evidence_kind = if accepted.len() > 1 {
            HumanStateEvidenceKind::Estimated
        } else {
            match accepted[0].source {
                HumanObservationSource::Instrument => HumanStateEvidenceKind::Observed,
                HumanObservationSource::WearerReported => HumanStateEvidenceKind::WearerReported,
                HumanObservationSource::ModelEstimate => HumanStateEvidenceKind::Estimated,
                HumanObservationSource::ModelPrediction => HumanStateEvidenceKind::Predicted,
            }
        };

        fused_metrics.push(FusedHumanMetric {
            metric,
            status: HumanMetricStatus::Fresh,
            value: Some(estimate),
            lower_bound: Some(lower),
            upper_bound: Some(upper),
            confidence: Some(conservative_confidence),
            evidence_kind,
            source_ids,
            wearer_reported_values,
        });
    }

    Ok(HumanStateFusionReport {
        evaluated_at_s,
        metrics: fused_metrics,
        rejected_invalid_observations: rejected_invalid,
        rejected_future_dated_observations: rejected_future,
        excluded_stale_observations: excluded_stale,
        excluded_low_confidence_observations: excluded_low_confidence,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HydrationLedger {
    pub starting_ml: f64,
    pub replenished_ml: f64,
    pub consumed_ml: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HydrationLedgerError {
    InvalidAmount,
    InventoryOverdraw,
    ArithmeticOverflow,
}

impl HydrationLedger {
    pub fn new(starting_ml: f64) -> Result<Self, HydrationLedgerError> {
        if !starting_ml.is_finite() || starting_ml < 0.0 {
            return Err(HydrationLedgerError::InvalidAmount);
        }
        Ok(Self {
            starting_ml,
            replenished_ml: 0.0,
            consumed_ml: 0.0,
        })
    }

    pub fn remaining_ml(&self) -> f64 {
        (self.starting_ml + self.replenished_ml - self.consumed_ml).max(0.0)
    }

    /// Records replenishment after a resource transfer is independently verified.
    pub fn record_replenishment(&mut self, amount_ml: f64) -> Result<(), HydrationLedgerError> {
        if !amount_ml.is_finite() || amount_ml < 0.0 {
            return Err(HydrationLedgerError::InvalidAmount);
        }
        let replenished = self.replenished_ml + amount_ml;
        if !replenished.is_finite() || !(self.starting_ml + replenished).is_finite() {
            return Err(HydrationLedgerError::ArithmeticOverflow);
        }
        self.replenished_ml = replenished;
        Ok(())
    }

    /// Refuses overdraw and leaves the ledger unchanged on every error.
    pub fn record_consumption(&mut self, amount_ml: f64) -> Result<(), HydrationLedgerError> {
        if !amount_ml.is_finite() || amount_ml < 0.0 {
            return Err(HydrationLedgerError::InvalidAmount);
        }
        if amount_ml > self.remaining_ml() {
            return Err(HydrationLedgerError::InventoryOverdraw);
        }
        let consumed = self.consumed_ml + amount_ml;
        if !consumed.is_finite() {
            return Err(HydrationLedgerError::ArithmeticOverflow);
        }
        self.consumed_ml = consumed;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        metric: HumanMetric,
        value: f64,
        uncertainty_abs: f64,
        observed_at_s: f64,
        confidence: f64,
        source_id: &str,
        source: HumanObservationSource,
    ) -> HumanStateObservation {
        HumanStateObservation {
            metric,
            value,
            uncertainty_abs,
            observed_at_s,
            confidence,
            source_id: source_id.to_string(),
            source,
        }
    }

    #[test]
    fn missing_metric_stays_unknown_instead_of_becoming_nominal() {
        let report = fuse_human_state(&[], 100.0, HumanStateConfig::simulation_reference()).unwrap();
        let metric = report.metric(HumanMetric::CoreTemperatureC).unwrap();
        assert_eq!(metric.status, HumanMetricStatus::Missing);
        assert_eq!(metric.value, None);
        assert_eq!(metric.evidence_kind, HumanStateEvidenceKind::Unknown);
    }

    #[test]
    fn stale_observation_cannot_contribute_to_fused_value() {
        let old = observation(
            HumanMetric::HeartRateBpm, 90.0, 2.0, 10.0, 0.99, "heart-a",
            HumanObservationSource::Instrument,
        );
        let report = fuse_human_state(&[old], 100.0, HumanStateConfig::simulation_reference()).unwrap();
        assert_eq!(
            report.metric(HumanMetric::HeartRateBpm).unwrap().status,
            HumanMetricStatus::Stale
        );
        assert_eq!(report.metric(HumanMetric::HeartRateBpm).unwrap().value, None);
        assert_eq!(report.excluded_stale_observations, 1);
    }

    #[test]
    fn conflicting_fresh_core_temperature_sources_remain_conflicted() {
        let inputs = [
            observation(
                HumanMetric::CoreTemperatureC, 37.0, 0.1, 99.0, 0.99, "temp-a",
                HumanObservationSource::Instrument,
            ),
            observation(
                HumanMetric::CoreTemperatureC, 40.0, 0.1, 99.0, 0.98, "temp-b",
                HumanObservationSource::Instrument,
            ),
        ];
        let report = fuse_human_state(&inputs, 100.0, HumanStateConfig::simulation_reference()).unwrap();
        let fused = report.metric(HumanMetric::CoreTemperatureC).unwrap();
        assert_eq!(fused.status, HumanMetricStatus::Conflicted);
        assert_eq!(fused.value, None);
        assert_eq!(fused.source_ids.len(), 2);
    }

    #[test]
    fn latest_sample_from_one_sensor_replaces_its_older_sample() {
        let inputs = [
            observation(
                HumanMetric::HeartRateBpm, 80.0, 1.0, 96.0, 0.95, "heart-a",
                HumanObservationSource::Instrument,
            ),
            observation(
                HumanMetric::HeartRateBpm, 84.0, 1.0, 99.0, 0.95, "heart-a",
                HumanObservationSource::Instrument,
            ),
            observation(
                HumanMetric::HeartRateBpm, 85.0, 1.0, 99.0, 0.95, "heart-b",
                HumanObservationSource::Instrument,
            ),
        ];
        let report = fuse_human_state(&inputs, 100.0, HumanStateConfig::simulation_reference()).unwrap();
        let fused = report.metric(HumanMetric::HeartRateBpm).unwrap();
        assert_eq!(fused.status, HumanMetricStatus::Fresh);
        assert_eq!(fused.source_ids.len(), 2);
        assert!(fused.value.unwrap() > 84.0);
        assert!(fused.value.unwrap() < 85.1);
    }

    #[test]
    fn wearer_report_is_preserved_separately_from_instrument_evidence() {
        let inputs = [
            observation(
                HumanMetric::PerceivedExertion0To10, 7.0, 1.0, 99.0, 0.99, "wearer",
                HumanObservationSource::WearerReported,
            ),
            observation(
                HumanMetric::PerceivedExertion0To10, 6.5, 0.5, 99.0, 0.90, "protocol",
                HumanObservationSource::Instrument,
            ),
        ];
        let report = fuse_human_state(&inputs, 100.0, HumanStateConfig::simulation_reference()).unwrap();
        let fused = report.metric(HumanMetric::PerceivedExertion0To10).unwrap();
        assert_eq!(fused.status, HumanMetricStatus::Fresh);
        assert_eq!(fused.evidence_kind, HumanStateEvidenceKind::Estimated);
        assert_eq!(fused.wearer_reported_values, vec![7.0]);
    }

    #[test]
    fn future_dated_and_nan_readings_fail_closed() {
        let inputs = [
            observation(
                HumanMetric::RespiratoryRateBpm, 20.0, 1.0, 101.0, 0.99, "future",
                HumanObservationSource::Instrument,
            ),
            observation(
                HumanMetric::RespiratoryRateBpm, f64::NAN, 1.0, 99.0, 0.99, "nan",
                HumanObservationSource::Instrument,
            ),
        ];
        let report = fuse_human_state(&inputs, 100.0, HumanStateConfig::simulation_reference()).unwrap();
        assert_eq!(
            report.metric(HumanMetric::RespiratoryRateBpm).unwrap().status,
            HumanMetricStatus::InsufficientEvidence
        );
        assert_eq!(report.metric(HumanMetric::RespiratoryRateBpm).unwrap().value, None);
        assert_eq!(report.rejected_future_dated_observations, 1);
        assert_eq!(report.rejected_invalid_observations, 1);
    }

    #[test]
    fn invalid_normalized_metrics_are_not_accepted() {
        let input = observation(
            HumanMetric::LocalFatigueFraction, 1.5, 0.1, 99.0, 0.99, "fatigue",
            HumanObservationSource::ModelEstimate,
        );
        let report = fuse_human_state(&[input], 100.0, HumanStateConfig::simulation_reference()).unwrap();
        assert_eq!(
            report.metric(HumanMetric::LocalFatigueFraction).unwrap().status,
            HumanMetricStatus::Invalid
        );
        assert_eq!(report.metric(HumanMetric::LocalFatigueFraction).unwrap().value, None);
    }

    #[test]
    fn hydration_ledger_conserves_inventory_and_refuses_overdraw_atomically() {
        let mut ledger = HydrationLedger::new(500.0).unwrap();
        ledger.record_consumption(120.0).unwrap();
        ledger.record_replenishment(200.0).unwrap();
        assert_eq!(ledger.remaining_ml(), 580.0);
        let before = ledger;
        assert_eq!(
            ledger.record_consumption(581.0),
            Err(HydrationLedgerError::InventoryOverdraw)
        );
        assert_eq!(ledger, before);
        ledger.record_consumption(580.0).unwrap();
        assert_eq!(ledger.remaining_ml(), 0.0);
    }

    #[test]
    fn invalid_hydration_amount_does_not_mutate_ledger() {
        let mut ledger = HydrationLedger::new(250.0).unwrap();
        let before = ledger;
        assert_eq!(
            ledger.record_replenishment(f64::INFINITY),
            Err(HydrationLedgerError::InvalidAmount)
        );
        assert_eq!(ledger, before);
    }
}
