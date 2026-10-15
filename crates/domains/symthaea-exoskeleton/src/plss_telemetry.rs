// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-preserving PLSS telemetry fusion for Space Exosuit research (SX-025).
//!
//! This is a read-only, simulation-tagged evidence layer. Instrument readings,
//! model estimates, and model predictions are fused separately; one class is
//! never allowed to make another class look like an observation. Missing,
//! stale, invalid, low-confidence, or conflicted inputs never become a numeric
//! nominal value. No actuator, life-support, or alert-command interface exists.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PLSS_TELEMETRY_METRIC_COUNT: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PlssTelemetryMetric {
    CircuitCo2PartialPressurePa,
    OxygenPartialPressurePa,
    CircuitPressurePa,
    GasTemperatureK,
    VentilationFlowLMin,
    RelativeHumidityPercent,
}

impl PlssTelemetryMetric {
    pub const ALL: [Self; PLSS_TELEMETRY_METRIC_COUNT] = [
        Self::CircuitCo2PartialPressurePa,
        Self::OxygenPartialPressurePa,
        Self::CircuitPressurePa,
        Self::GasTemperatureK,
        Self::VentilationFlowLMin,
        Self::RelativeHumidityPercent,
    ];

    pub const fn unit(self) -> PlssTelemetryUnit {
        match self {
            Self::CircuitCo2PartialPressurePa
            | Self::OxygenPartialPressurePa
            | Self::CircuitPressurePa => PlssTelemetryUnit::Pascal,
            Self::GasTemperatureK => PlssTelemetryUnit::Kelvin,
            Self::VentilationFlowLMin => PlssTelemetryUnit::LiterPerMinute,
            Self::RelativeHumidityPercent => PlssTelemetryUnit::Percent,
        }
    }

    /// Encoding/domain guardrails only; these are not atmosphere safety limits.
    fn accepts_value(self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match self {
            Self::CircuitCo2PartialPressurePa | Self::OxygenPartialPressurePa => {
                (0.0..=10_000_000.0).contains(&value)
            }
            Self::CircuitPressurePa => (0.0..=100_000_000.0).contains(&value),
            Self::GasTemperatureK => value > 0.0 && value <= 2_000.0,
            Self::VentilationFlowLMin => (0.0..=1_000_000.0).contains(&value),
            Self::RelativeHumidityPercent => (0.0..=100.0).contains(&value),
        }
    }

    fn disagreement_tolerance(self, tolerances: PlssTelemetryTolerances) -> f64 {
        match self {
            Self::CircuitCo2PartialPressurePa => tolerances.co2_partial_pressure_pa,
            Self::OxygenPartialPressurePa => tolerances.oxygen_partial_pressure_pa,
            Self::CircuitPressurePa => tolerances.circuit_pressure_pa,
            Self::GasTemperatureK => tolerances.gas_temperature_k,
            Self::VentilationFlowLMin => tolerances.ventilation_flow_l_min,
            Self::RelativeHumidityPercent => tolerances.relative_humidity_percent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PlssTelemetryUnit {
    Pascal,
    Kelvin,
    LiterPerMinute,
    Percent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PlssTelemetrySourceKind {
    Instrument,
    ModelEstimate,
    ModelPrediction,
}

impl PlssTelemetrySourceKind {
    pub const ALL: [Self; 3] = [
        Self::Instrument,
        Self::ModelEstimate,
        Self::ModelPrediction,
    ];
}

/// A single time-stamped observation. Mission-elapsed seconds must be monotonic;
/// confidence is producer-supplied and is not a calibration guarantee.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryObservation {
    pub metric: PlssTelemetryMetric,
    pub unit: PlssTelemetryUnit,
    pub value: f64,
    /// Absolute uncertainty in the metric's declared unit, not a confidence interval.
    pub uncertainty_abs: f64,
    pub observed_at_s: f64,
    pub confidence: f64,
    pub source_id: String,
    pub source_kind: PlssTelemetrySourceKind,
}

impl PlssTelemetryObservation {
    pub fn is_valid(&self) -> bool {
        self.unit == self.metric.unit()
            && self.metric.accepts_value(self.value)
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
pub struct PlssTelemetryTolerances {
    pub co2_partial_pressure_pa: f64,
    pub oxygen_partial_pressure_pa: f64,
    pub circuit_pressure_pa: f64,
    pub gas_temperature_k: f64,
    pub ventilation_flow_l_min: f64,
    pub relative_humidity_percent: f64,
}

impl PlssTelemetryTolerances {
    fn is_valid(self) -> bool {
        [
            self.co2_partial_pressure_pa,
            self.oxygen_partial_pressure_pa,
            self.circuit_pressure_pa,
            self.gas_temperature_k,
            self.ventilation_flow_l_min,
            self.relative_humidity_percent,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryConfig {
    pub maximum_observation_age_s: f64,
    pub minimum_confidence: f64,
    pub disagreement_tolerances: PlssTelemetryTolerances,
}

impl PlssTelemetryConfig {
    /// Illustrative software-only settings. These are not human exposure limits,
    /// instrument calibration claims, or requirements for any qualified spacesuit.
    pub fn simulation_reference() -> Self {
        Self {
            maximum_observation_age_s: 5.0,
            minimum_confidence: 0.70,
            disagreement_tolerances: PlssTelemetryTolerances {
                co2_partial_pressure_pa: 250.0,
                oxygen_partial_pressure_pa: 1_000.0,
                circuit_pressure_pa: 2_000.0,
                gas_temperature_k: 2.0,
                ventilation_flow_l_min: 15.0,
                relative_humidity_percent: 5.0,
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
pub enum PlssTelemetryEvidenceLevel {
    SimulationOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlssTelemetryStatus {
    Fresh,
    Conflicted,
    Stale,
    LowConfidence,
    Invalid,
    InsufficientEvidence,
    Missing,
}

/// A result is for exactly one metric and one provenance class. A model estimate
/// can be inspected next to an instrument result, but is never fused into it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedPlssTelemetryMetric {
    pub metric: PlssTelemetryMetric,
    pub unit: PlssTelemetryUnit,
    pub source_kind: PlssTelemetrySourceKind,
    pub status: PlssTelemetryStatus,
    /// None for missing, invalid, or conflicted evidence. Never defaults to zero.
    pub value: Option<f64>,
    /// Declared-uncertainty envelope; not a calibrated statistical interval.
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub confidence: Option<f64>,
    pub source_ids: Vec<String>,
    /// Number of distinct, accepted source IDs after latest-sample reduction.
    pub accepted_source_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryFusionReport {
    pub evaluated_at_s: f64,
    pub evidence_level: PlssTelemetryEvidenceLevel,
    /// The exact policy used to evaluate this report.
    pub config: PlssTelemetryConfig,
    /// Stable order: metric order in PLSS_TELEMETRY_METRIC_COUNT, then source kind.
    pub metrics: Vec<FusedPlssTelemetryMetric>,
    pub rejected_invalid_observations: usize,
    pub rejected_future_dated_observations: usize,
    pub excluded_stale_observations: usize,
    pub excluded_low_confidence_observations: usize,
}

impl PlssTelemetryFusionReport {
    pub fn metric(
        &self,
        metric: PlssTelemetryMetric,
        source_kind: PlssTelemetrySourceKind,
    ) -> Option<&FusedPlssTelemetryMetric> {
        self.metrics
            .iter()
            .find(|entry| entry.metric == metric && entry.source_kind == source_kind)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlssTelemetryError {
    InvalidConfig,
    InvalidEvaluationTime,
    InvalidSourceRevision,
}

/// Fuse only fresh, valid, sufficiently confident observations of the same
/// metric and provenance class. Sequential samples from one source contribute
/// only its newest eligible sample. Equal-time contradictory samples from one
/// source are a data-integrity conflict and fail closed.
pub fn fuse_plss_telemetry(
    observations: &[PlssTelemetryObservation],
    evaluated_at_s: f64,
    config: PlssTelemetryConfig,
) -> Result<PlssTelemetryFusionReport, PlssTelemetryError> {
    if !config.is_valid() {
        return Err(PlssTelemetryError::InvalidConfig);
    }
    if !evaluated_at_s.is_finite() || evaluated_at_s < 0.0 {
        return Err(PlssTelemetryError::InvalidEvaluationTime);
    }

    let mut rejected_invalid = 0;
    let mut rejected_future = 0;
    let mut excluded_stale = 0;
    let mut excluded_low_confidence = 0;
    let capacity = PLSS_TELEMETRY_METRIC_COUNT * PlssTelemetrySourceKind::ALL.len();
    let mut fused = Vec::with_capacity(capacity);

    for metric in PlssTelemetryMetric::ALL {
        for source_kind in PlssTelemetrySourceKind::ALL {
            let matching = observations
                .iter()
                .filter(|item| item.metric == metric && item.source_kind == source_kind);
            let mut saw_any = false;
            let mut saw_invalid = false;
            let mut saw_future = false;
            let mut saw_stale = false;
            let mut saw_low_confidence = false;
            let mut latest_by_source: Vec<(PlssTelemetryObservation, bool)> = Vec::new();

            for observation in matching {
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
                if observation.confidence < config.minimum_confidence
                    || observation.confidence <= 0.0
                {
                    excluded_low_confidence += 1;
                    saw_low_confidence = true;
                    continue;
                }

                if let Some((previous, tied_conflict)) = latest_by_source
                    .iter_mut()
                    .find(|(previous, _)| previous.source_id == observation.source_id)
                {
                    if observation.observed_at_s > previous.observed_at_s {
                        *previous = observation.clone();
                        *tied_conflict = false;
                    } else if observation.observed_at_s == previous.observed_at_s
                        && !same_reading(previous, observation)
                    {
                        *tied_conflict = true;
                    }
                } else {
                    latest_by_source.push((observation.clone(), false));
                }
            }

            if latest_by_source.is_empty() {
                let status = match (saw_invalid || saw_future, saw_stale, saw_low_confidence) {
                    (true, false, false) => PlssTelemetryStatus::Invalid,
                    (false, true, false) => PlssTelemetryStatus::Stale,
                    (false, false, true) => PlssTelemetryStatus::LowConfidence,
                    (false, false, false) if !saw_any => PlssTelemetryStatus::Missing,
                    _ => PlssTelemetryStatus::InsufficientEvidence,
                };
                fused.push(FusedPlssTelemetryMetric {
                    metric,
                    unit: metric.unit(),
                    source_kind,
                    status,
                    value: None,
                    lower_bound: None,
                    upper_bound: None,
                    confidence: None,
                    source_ids: Vec::new(),
                    accepted_source_count: 0,
                });
                continue;
            }

            latest_by_source.sort_by(|(left, _), (right, _)| left.source_id.cmp(&right.source_id));
            let source_conflict = latest_by_source.iter().any(|(_, tied_conflict)| *tied_conflict);
            let accepted: Vec<PlssTelemetryObservation> = latest_by_source
                .iter()
                .map(|(observation, _)| observation.clone())
                .collect();
            let conflict = source_conflict || accepted.iter().enumerate().any(|(index, left)| {
                accepted.iter().skip(index + 1).any(|right| {
                    (left.value - right.value).abs()
                        > metric.disagreement_tolerance(config.disagreement_tolerances)
                            + left.uncertainty_abs
                            + right.uncertainty_abs
                })
            });
            let source_ids: Vec<String> = accepted
                .iter()
                .map(|item| item.source_id.clone())
                .collect();

            if conflict {
                fused.push(FusedPlssTelemetryMetric {
                    metric,
                    unit: metric.unit(),
                    source_kind,
                    status: PlssTelemetryStatus::Conflicted,
                    value: None,
                    lower_bound: None,
                    upper_bound: None,
                    confidence: None,
                    accepted_source_count: accepted.len(),
                    source_ids,
                });
                continue;
            }

            let mut estimate = accepted[0].value;
            let mut total_weight = accepted[0].confidence;
            let mut lower = accepted[0].value - accepted[0].uncertainty_abs;
            let mut upper = accepted[0].value + accepted[0].uncertainty_abs;
            let mut conservative_confidence = accepted[0].confidence;
            let mut invalid_weight = false;

            for item in accepted.iter().skip(1) {
                let next_weight = total_weight + item.confidence;
                if !next_weight.is_finite() || next_weight <= 0.0 {
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

            let uncertainty = (estimate - lower).max(upper - estimate);
            if invalid_weight
                || !estimate.is_finite()
                || !lower.is_finite()
                || !upper.is_finite()
                || !uncertainty.is_finite()
                || estimate < lower
                || estimate > upper
            {
                fused.push(FusedPlssTelemetryMetric {
                    metric,
                    unit: metric.unit(),
                    source_kind,
                    status: PlssTelemetryStatus::Invalid,
                    value: None,
                    lower_bound: None,
                    upper_bound: None,
                    confidence: None,
                    accepted_source_count: accepted.len(),
                    source_ids,
                });
                continue;
            }

            fused.push(FusedPlssTelemetryMetric {
                metric,
                unit: metric.unit(),
                source_kind,
                status: PlssTelemetryStatus::Fresh,
                value: Some(estimate),
                lower_bound: Some(lower),
                upper_bound: Some(upper),
                confidence: Some(conservative_confidence),
                accepted_source_count: accepted.len(),
                source_ids,
            });
        }
    }

    Ok(PlssTelemetryFusionReport {
        evaluated_at_s,
        evidence_level: PlssTelemetryEvidenceLevel::SimulationOnly,
        config,
        metrics: fused,
        rejected_invalid_observations: rejected_invalid,
        rejected_future_dated_observations: rejected_future,
        excluded_stale_observations: excluded_stale,
        excluded_low_confidence_observations: excluded_low_confidence,
    })
}


/// Reproducibility metadata for one fusion evaluation. SHA-256 fingerprints
/// detect input/result changes, but do not authenticate sensors or prove that
/// a caller supplied a truthful source revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlssTelemetryReceipt {
    /// Must be the exact 40-character Git SHA or 64-character Git object ID.
    pub source_revision: String,
    pub input_observation_count: usize,
    pub accepted_source_metric_pairs: usize,
    pub rejected_invalid_observations: usize,
    pub rejected_future_dated_observations: usize,
    pub excluded_stale_observations: usize,
    pub excluded_low_confidence_observations: usize,
    pub input_digest_sha256: String,
    pub result_digest_sha256: String,
}

/// Evaluate and produce a deterministic receipt tied to a caller-supplied,
/// syntactically valid source revision. Callers should supply the exact checked
/// out commit SHA; this function validates identifier form, not Git authenticity.
pub fn evaluate_plss_telemetry_with_receipt(
    observations: &[PlssTelemetryObservation],
    evaluated_at_s: f64,
    config: PlssTelemetryConfig,
    source_revision: &str,
) -> Result<(PlssTelemetryFusionReport, PlssTelemetryReceipt), PlssTelemetryError> {
    let valid_revision_length = source_revision.len() == 40 || source_revision.len() == 64;
    if !valid_revision_length || !source_revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PlssTelemetryError::InvalidSourceRevision);
    }

    let report = fuse_plss_telemetry(observations, evaluated_at_s, config)?;
    let receipt = PlssTelemetryReceipt {
        source_revision: source_revision.to_ascii_lowercase(),
        input_observation_count: observations.len(),
        accepted_source_metric_pairs: report
            .metrics
            .iter()
            .map(|metric| metric.accepted_source_count)
            .sum(),
        rejected_invalid_observations: report.rejected_invalid_observations,
        rejected_future_dated_observations: report.rejected_future_dated_observations,
        excluded_stale_observations: report.excluded_stale_observations,
        excluded_low_confidence_observations: report.excluded_low_confidence_observations,
        input_digest_sha256: digest_input_observations(observations),
        result_digest_sha256: digest_fusion_report(&report),
    };
    Ok((report, receipt))
}

struct StableSha256(Sha256);

impl StableSha256 {
    fn new() -> Self {
        Self(Sha256::new())
    }

    fn write_raw(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn write_u64(&mut self, value: u64) {
        self.write_raw(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn write_f64(&mut self, value: f64) {
        self.write_u64(value.to_bits());
    }

    fn write_str(&mut self, value: &str) {
        self.write_usize(value.len());
        self.write_raw(value.as_bytes());
    }

    fn write_optional_f64(&mut self, value: Option<f64>) {
        match value {
            Some(value) => {
                self.write_raw(&[1]);
                self.write_f64(value);
            }
            None => self.write_raw(&[0]),
        }
    }

    fn finish(self) -> String {
        let digest = self.0.finalize();
        let encoded = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        format!("sha256:{encoded}")
    }
}

fn digest_input_observations(observations: &[PlssTelemetryObservation]) -> String {
    let mut canonical = observations.to_vec();
    canonical.sort_by(|left, right| {
        left.metric
            .cmp(&right.metric)
            .then(left.source_kind.cmp(&right.source_kind))
            .then(left.source_id.cmp(&right.source_id))
            .then(left.observed_at_s.total_cmp(&right.observed_at_s))
            .then(left.unit.cmp(&right.unit))
            .then(left.value.total_cmp(&right.value))
            .then(left.uncertainty_abs.total_cmp(&right.uncertainty_abs))
            .then(left.confidence.total_cmp(&right.confidence))
    });

    let mut hash = StableSha256::new();
    hash.write_str("symthaea.plss.telemetry.input.v1");
    hash.write_usize(canonical.len());
    for observation in &canonical {
        hash.write_str(&format!("{:?}", observation.metric));
        hash.write_str(&format!("{:?}", observation.unit));
        hash.write_f64(observation.value);
        hash.write_f64(observation.uncertainty_abs);
        hash.write_f64(observation.observed_at_s);
        hash.write_f64(observation.confidence);
        hash.write_str(&observation.source_id);
        hash.write_str(&format!("{:?}", observation.source_kind));
    }
    hash.finish()
}

fn digest_fusion_report(report: &PlssTelemetryFusionReport) -> String {
    let mut hash = StableSha256::new();
    hash.write_str("symthaea.plss.telemetry.result.v1");
    hash.write_f64(report.evaluated_at_s);
    hash.write_str(&format!("{:?}", report.evidence_level));
    hash.write_f64(report.config.maximum_observation_age_s);
    hash.write_f64(report.config.minimum_confidence);
    hash.write_f64(report.config.disagreement_tolerances.co2_partial_pressure_pa);
    hash.write_f64(report.config.disagreement_tolerances.oxygen_partial_pressure_pa);
    hash.write_f64(report.config.disagreement_tolerances.circuit_pressure_pa);
    hash.write_f64(report.config.disagreement_tolerances.gas_temperature_k);
    hash.write_f64(report.config.disagreement_tolerances.ventilation_flow_l_min);
    hash.write_f64(report.config.disagreement_tolerances.relative_humidity_percent);
    hash.write_usize(report.rejected_invalid_observations);
    hash.write_usize(report.rejected_future_dated_observations);
    hash.write_usize(report.excluded_stale_observations);
    hash.write_usize(report.excluded_low_confidence_observations);
    hash.write_usize(report.metrics.len());

    for metric in &report.metrics {
        hash.write_str(&format!("{:?}", metric.metric));
        hash.write_str(&format!("{:?}", metric.unit));
        hash.write_str(&format!("{:?}", metric.source_kind));
        hash.write_str(&format!("{:?}", metric.status));
        hash.write_optional_f64(metric.value);
        hash.write_optional_f64(metric.lower_bound);
        hash.write_optional_f64(metric.upper_bound);
        hash.write_optional_f64(metric.confidence);
        hash.write_usize(metric.accepted_source_count);
        hash.write_usize(metric.source_ids.len());
        for source_id in &metric.source_ids {
            hash.write_str(source_id);
        }
    }
    hash.finish()
}

fn same_reading(left: &PlssTelemetryObservation, right: &PlssTelemetryObservation) -> bool {
    left.metric == right.metric
        && left.unit == right.unit
        && left.value == right.value
        && left.uncertainty_abs == right.uncertainty_abs
        && left.observed_at_s == right.observed_at_s
        && left.confidence == right.confidence
        && left.source_id == right.source_id
        && left.source_kind == right.source_kind
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        metric: PlssTelemetryMetric,
        value: f64,
        uncertainty_abs: f64,
        observed_at_s: f64,
        confidence: f64,
        source_id: &str,
        source_kind: PlssTelemetrySourceKind,
    ) -> PlssTelemetryObservation {
        PlssTelemetryObservation {
            metric,
            unit: metric.unit(),
            value,
            uncertainty_abs,
            observed_at_s,
            confidence,
            source_id: source_id.to_string(),
            source_kind,
        }
    }

    fn fuse(observations: &[PlssTelemetryObservation]) -> PlssTelemetryFusionReport {
        fuse_plss_telemetry(
            observations,
            100.0,
            PlssTelemetryConfig::simulation_reference(),
        )
        .unwrap()
    }

    #[test]
    fn missing_instrument_data_stays_missing_instead_of_becoming_model_value() {
        let report = fuse(&[]);
        let value = report
            .metric(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(value.status, PlssTelemetryStatus::Missing);
        assert_eq!(value.value, None);
        assert_eq!(report.evidence_level, PlssTelemetryEvidenceLevel::SimulationOnly);
    }

    #[test]
    fn instrument_observation_and_model_estimate_are_never_fused_together() {
        let report = fuse(&[
            observation(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                900.0,
                20.0,
                99.0,
                0.99,
                "co2-instrument",
                PlssTelemetrySourceKind::Instrument,
            ),
            observation(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                18_000.0,
                100.0,
                99.0,
                0.99,
                "bulk-model",
                PlssTelemetrySourceKind::ModelEstimate,
            ),
        ]);
        let measured = report
            .metric(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        let modeled = report
            .metric(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                PlssTelemetrySourceKind::ModelEstimate,
            )
            .unwrap();
        assert_eq!(measured.status, PlssTelemetryStatus::Fresh);
        assert_eq!(measured.value, Some(900.0));
        assert_eq!(modeled.status, PlssTelemetryStatus::Fresh);
        assert_eq!(modeled.value, Some(18_000.0));
        assert_eq!(report.evidence_level, PlssTelemetryEvidenceLevel::SimulationOnly);
    }

    #[test]
    fn conflicting_fresh_instruments_have_no_fused_point_estimate() {
        let report = fuse(&[
            observation(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                1_000.0,
                10.0,
                99.0,
                0.99,
                "co2-a",
                PlssTelemetrySourceKind::Instrument,
            ),
            observation(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                5_000.0,
                10.0,
                99.0,
                0.95,
                "co2-b",
                PlssTelemetrySourceKind::Instrument,
            ),
        ]);
        let fused = report
            .metric(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(fused.status, PlssTelemetryStatus::Conflicted);
        assert_eq!(fused.value, None);
        assert_eq!(fused.source_ids, vec!["co2-a", "co2-b"]);
    }

    #[test]
    fn latest_sample_from_one_source_replaces_older_samples() {
        let report = fuse(&[
            observation(
                PlssTelemetryMetric::VentilationFlowLMin,
                100.0,
                2.0,
                98.0,
                0.95,
                "flow-a",
                PlssTelemetrySourceKind::Instrument,
            ),
            observation(
                PlssTelemetryMetric::VentilationFlowLMin,
                130.0,
                2.0,
                100.0,
                0.95,
                "flow-a",
                PlssTelemetrySourceKind::Instrument,
            ),
        ]);
        let fused = report
            .metric(
                PlssTelemetryMetric::VentilationFlowLMin,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(fused.status, PlssTelemetryStatus::Fresh);
        assert_eq!(fused.value, Some(130.0));
        assert_eq!(fused.accepted_source_count, 1);
        assert_eq!(fused.source_ids, vec!["flow-a"]);
    }

    #[test]
    fn contradictory_equal_time_samples_from_one_source_fail_closed() {
        let report = fuse(&[
            observation(
                PlssTelemetryMetric::CircuitPressurePa,
                50_000.0,
                100.0,
                99.0,
                0.95,
                "pressure-a",
                PlssTelemetrySourceKind::Instrument,
            ),
            observation(
                PlssTelemetryMetric::CircuitPressurePa,
                55_000.0,
                100.0,
                99.0,
                0.95,
                "pressure-a",
                PlssTelemetrySourceKind::Instrument,
            ),
        ]);
        let fused = report
            .metric(
                PlssTelemetryMetric::CircuitPressurePa,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(fused.status, PlssTelemetryStatus::Conflicted);
        assert_eq!(fused.value, None);
    }

    #[test]
    fn stale_future_invalid_and_low_confidence_inputs_do_not_contribute() {
        let mut wrong_unit = observation(
            PlssTelemetryMetric::GasTemperatureK,
            290.0,
            0.1,
            99.0,
            0.99,
            "wrong-unit",
            PlssTelemetrySourceKind::Instrument,
        );
        wrong_unit.unit = PlssTelemetryUnit::Pascal;
        let report = fuse(&[
            observation(
                PlssTelemetryMetric::GasTemperatureK,
                290.0,
                0.1,
                90.0,
                0.99,
                "stale",
                PlssTelemetrySourceKind::Instrument,
            ),
            observation(
                PlssTelemetryMetric::GasTemperatureK,
                290.0,
                0.1,
                101.0,
                0.99,
                "future",
                PlssTelemetrySourceKind::Instrument,
            ),
            observation(
                PlssTelemetryMetric::GasTemperatureK,
                f64::NAN,
                0.1,
                99.0,
                0.99,
                "nan",
                PlssTelemetrySourceKind::Instrument,
            ),
            wrong_unit,
            observation(
                PlssTelemetryMetric::GasTemperatureK,
                290.0,
                0.1,
                99.0,
                0.2,
                "low-confidence",
                PlssTelemetrySourceKind::Instrument,
            ),
        ]);
        assert_eq!(report.rejected_invalid_observations, 2);
        assert_eq!(report.rejected_future_dated_observations, 1);
        assert_eq!(report.excluded_stale_observations, 1);
        assert_eq!(report.excluded_low_confidence_observations, 1);
        let fused = report
            .metric(
                PlssTelemetryMetric::GasTemperatureK,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(fused.status, PlssTelemetryStatus::InsufficientEvidence);
        assert_eq!(fused.value, None);
    }

    #[test]
    fn output_source_order_is_stable_when_input_order_changes() {
        let a = observation(
            PlssTelemetryMetric::RelativeHumidityPercent,
            30.0,
            1.0,
            99.0,
            0.9,
            "sensor-a",
            PlssTelemetrySourceKind::Instrument,
        );
        let b = observation(
            PlssTelemetryMetric::RelativeHumidityPercent,
            31.0,
            1.0,
            99.0,
            0.95,
            "sensor-b",
            PlssTelemetrySourceKind::Instrument,
        );
        let first = fuse(&[a.clone(), b.clone()]);
        let second = fuse(&[b, a]);
        assert_eq!(first.metrics, second.metrics);
    }

    #[test]
    fn sha256_implementation_matches_standard_test_vector() {
        let mut hash = StableSha256::new();
        hash.write_raw(b"abc");
        assert_eq!(
            hash.finish(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn receipt_is_reproducible_and_input_order_independent() {
        let a = observation(
            PlssTelemetryMetric::RelativeHumidityPercent,
            30.0,
            1.0,
            99.0,
            0.90,
            "humidity-a",
            PlssTelemetrySourceKind::Instrument,
        );
        let b = observation(
            PlssTelemetryMetric::RelativeHumidityPercent,
            31.0,
            1.0,
            99.0,
            0.95,
            "humidity-b",
            PlssTelemetrySourceKind::Instrument,
        );
        let revision = "a".repeat(40);
        let (_, first) = evaluate_plss_telemetry_with_receipt(
            &[a.clone(), b.clone()],
            100.0,
            PlssTelemetryConfig::simulation_reference(),
            &revision,
        )
        .unwrap();
        let (_, second) = evaluate_plss_telemetry_with_receipt(
            &[b, a],
            100.0,
            PlssTelemetryConfig::simulation_reference(),
            &revision,
        )
        .unwrap();
        assert_eq!(first.source_revision, revision);
        assert_eq!(first.input_observation_count, 2);
        assert_eq!(first.input_digest_sha256, second.input_digest_sha256);
        assert_eq!(first.result_digest_sha256, second.result_digest_sha256);
        assert!(first.input_digest_sha256.starts_with("sha256:"));
        assert!(first.result_digest_sha256.starts_with("sha256:"));
    }

    #[test]
    fn changing_an_observation_changes_input_and_result_fingerprints() {
        let first_observation = observation(
            PlssTelemetryMetric::VentilationFlowLMin,
            100.0,
            2.0,
            99.0,
            0.95,
            "flow-a",
            PlssTelemetrySourceKind::Instrument,
        );
        let changed_observation = observation(
            PlssTelemetryMetric::VentilationFlowLMin,
            101.0,
            2.0,
            99.0,
            0.95,
            "flow-a",
            PlssTelemetrySourceKind::Instrument,
        );
        let revision = "b".repeat(40);
        let (_, first) = evaluate_plss_telemetry_with_receipt(
            &[first_observation],
            100.0,
            PlssTelemetryConfig::simulation_reference(),
            &revision,
        )
        .unwrap();
        let (_, second) = evaluate_plss_telemetry_with_receipt(
            &[changed_observation],
            100.0,
            PlssTelemetryConfig::simulation_reference(),
            &revision,
        )
        .unwrap();
        assert_ne!(first.input_digest_sha256, second.input_digest_sha256);
        assert_ne!(first.result_digest_sha256, second.result_digest_sha256);
    }

    #[test]
    fn changing_configuration_changes_result_fingerprint() {
        let revision = "c".repeat(40);
        let (_, first) = evaluate_plss_telemetry_with_receipt(
            &[],
            100.0,
            PlssTelemetryConfig::simulation_reference(),
            &revision,
        )
        .unwrap();
        let mut config = PlssTelemetryConfig::simulation_reference();
        config.maximum_observation_age_s += 1.0;
        let (_, second) = evaluate_plss_telemetry_with_receipt(
            &[],
            100.0,
            config,
            &revision,
        )
        .unwrap();
        assert_eq!(first.input_digest_sha256, second.input_digest_sha256);
        assert_ne!(first.result_digest_sha256, second.result_digest_sha256);
    }

    #[test]
    fn receipt_requires_a_git_object_id_shape() {
        assert_eq!(
            evaluate_plss_telemetry_with_receipt(
                &[],
                100.0,
                PlssTelemetryConfig::simulation_reference(),
                "not-a-revision",
            ),
            Err(PlssTelemetryError::InvalidSourceRevision)
        );
    }

    #[test]
    fn invalid_config_and_future_evaluation_time_are_rejected() {
        let mut config = PlssTelemetryConfig::simulation_reference();
        config.minimum_confidence = 2.0;
        assert_eq!(
            fuse_plss_telemetry(&[], 100.0, config),
            Err(PlssTelemetryError::InvalidConfig)
        );
        assert_eq!(
            fuse_plss_telemetry(
                &[],
                f64::NAN,
                PlssTelemetryConfig::simulation_reference()
            ),
            Err(PlssTelemetryError::InvalidEvaluationTime)
        );
    }
}
