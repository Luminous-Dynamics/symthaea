// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference-data validation for reduced-order flight models.
//!
//! A simulator should not become more trusted merely because it has more state.
//! This module compares predicted scalar signals with independently identified
//! reference samples, separates calibration from held-out validation/test data,
//! and reports bias, RMSE, normalized RMSE, peak error, and uncertainty coverage.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FlightModelSignal {
    MainRotorRpm,
    TailRotorRpm,
    VerticalVelocity,
    HorizontalVelocity,
    YawRate,
    RotorThrust,
    HubMoment,
    FuelFlow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationPartition {
    Calibration,
    Validation,
    IndependentTest,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlightModelValidationSample {
    pub scenario_id_hash: u64,
    pub time_s: f64,
    pub signal: FlightModelSignal,
    pub partition: ValidationPartition,
    pub reference_value: f64,
    pub predicted_value: f64,
    pub reference_scale: f64,
    pub predicted_standard_deviation: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ValidationSignalGate {
    pub signal: FlightModelSignal,
    pub maximum_rmse: f64,
    pub maximum_normalized_rmse: f64,
    pub maximum_absolute_bias: f64,
    pub maximum_peak_error: f64,
    pub minimum_two_sigma_coverage: f64,
}

impl ValidationSignalGate {
    fn validate(&self) -> bool {
        self.maximum_rmse.is_finite()
            && self.maximum_rmse >= 0.0
            && self.maximum_normalized_rmse.is_finite()
            && self.maximum_normalized_rmse >= 0.0
            && self.maximum_absolute_bias.is_finite()
            && self.maximum_absolute_bias >= 0.0
            && self.maximum_peak_error.is_finite()
            && self.maximum_peak_error >= 0.0
            && self.minimum_two_sigma_coverage.is_finite()
            && (0.0..=1.0).contains(&self.minimum_two_sigma_coverage)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightModelValidationConfig {
    pub minimum_samples_per_signal: usize,
    pub require_independent_test_partition: bool,
    pub gates: Vec<ValidationSignalGate>,
}

impl FlightModelValidationConfig {
    pub fn validate(&self) -> Result<(), ModelValidationError> {
        if self.minimum_samples_per_signal == 0
            || self.gates.is_empty()
            || self.gates.iter().any(|gate| !gate.validate())
        {
            return Err(ModelValidationError::InvalidConfiguration);
        }
        let mut signals = self
            .gates
            .iter()
            .map(|gate| gate.signal)
            .collect::<Vec<_>>();
        signals.sort();
        signals.dedup();
        if signals.len() != self.gates.len() {
            return Err(ModelValidationError::DuplicateSignalGate);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationMetricStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignalValidationMetrics {
    pub signal: FlightModelSignal,
    pub status: ValidationMetricStatus,
    pub sample_count: usize,
    pub independent_test_samples: usize,
    pub bias: f64,
    pub rmse: f64,
    pub normalized_rmse: f64,
    pub peak_absolute_error: f64,
    pub two_sigma_coverage: Option<f64>,
    pub failed_metrics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightModelValidationReport {
    pub status: ValidationMetricStatus,
    pub total_samples: usize,
    pub rejected_samples: usize,
    pub signals: Vec<SignalValidationMetrics>,
    pub canonical_digest_fnv1a64: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelValidationError {
    InvalidConfiguration,
    DuplicateSignalGate,
    NonFiniteSample,
    InvalidReferenceScale,
    InvalidUncertainty,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct FlightModelValidator {
    config: FlightModelValidationConfig,
    samples: Vec<FlightModelValidationSample>,
    rejected_samples: usize,
}

impl FlightModelValidator {
    pub fn new(config: FlightModelValidationConfig) -> Result<Self, ModelValidationError> {
        config.validate()?;
        Ok(Self {
            config,
            samples: Vec::new(),
            rejected_samples: 0,
        })
    }

    pub fn push(
        &mut self,
        sample: FlightModelValidationSample,
    ) -> Result<(), ModelValidationError> {
        if [
            sample.time_s,
            sample.reference_value,
            sample.predicted_value,
            sample.reference_scale,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            self.rejected_samples = self.rejected_samples.saturating_add(1);
            return Err(ModelValidationError::NonFiniteSample);
        }
        if sample.time_s < 0.0 || sample.reference_scale <= 0.0 {
            self.rejected_samples = self.rejected_samples.saturating_add(1);
            return Err(ModelValidationError::InvalidReferenceScale);
        }
        if sample
            .predicted_standard_deviation
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            self.rejected_samples = self.rejected_samples.saturating_add(1);
            return Err(ModelValidationError::InvalidUncertainty);
        }
        self.samples.push(sample);
        Ok(())
    }

    pub fn report(&self) -> Result<FlightModelValidationReport, ModelValidationError> {
        self.config.validate()?;
        let mut by_signal = BTreeMap::<FlightModelSignal, Vec<FlightModelValidationSample>>::new();
        for sample in &self.samples {
            by_signal.entry(sample.signal).or_default().push(*sample);
        }
        let gates = self
            .config
            .gates
            .iter()
            .map(|gate| (gate.signal, *gate))
            .collect::<BTreeMap<_, _>>();
        let mut signals = Vec::new();
        for gate in &self.config.gates {
            let samples = by_signal
                .get(&gate.signal)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            signals.push(metrics_for_signal(
                *gate,
                samples,
                self.config.minimum_samples_per_signal,
                self.config.require_independent_test_partition,
            ));
        }
        let status = if signals
            .iter()
            .any(|metrics| metrics.status == ValidationMetricStatus::Fail)
        {
            ValidationMetricStatus::Fail
        } else if signals
            .iter()
            .any(|metrics| metrics.status == ValidationMetricStatus::Incomplete)
        {
            ValidationMetricStatus::Incomplete
        } else {
            ValidationMetricStatus::Pass
        };
        let digest_payload = serde_json::to_vec(&(&gates, &signals))
            .map_err(|_| ModelValidationError::SerializationFailed)?;
        Ok(FlightModelValidationReport {
            status,
            total_samples: self.samples.len(),
            rejected_samples: self.rejected_samples,
            signals,
            canonical_digest_fnv1a64: fnv1a64(&digest_payload),
        })
    }
}

fn metrics_for_signal(
    gate: ValidationSignalGate,
    samples: &[FlightModelValidationSample],
    minimum_samples: usize,
    require_independent_test: bool,
) -> SignalValidationMetrics {
    let sample_count = samples.len();
    let independent_test_samples = samples
        .iter()
        .filter(|sample| sample.partition == ValidationPartition::IndependentTest)
        .count();
    let mut failed_metrics = Vec::new();
    if sample_count < minimum_samples {
        failed_metrics.push("insufficient_samples".into());
    }
    if require_independent_test && independent_test_samples == 0 {
        failed_metrics.push("missing_independent_test_partition".into());
    }
    let errors = samples
        .iter()
        .map(|sample| sample.predicted_value - sample.reference_value)
        .collect::<Vec<_>>();
    let bias = mean(&errors);
    let rmse = mean(&errors.iter().map(|error| error * error).collect::<Vec<_>>()).sqrt();
    let normalized_rmse = if samples.is_empty() {
        0.0
    } else {
        (samples
            .iter()
            .map(|sample| {
                let error = sample.predicted_value - sample.reference_value;
                (error / sample.reference_scale).powi(2)
            })
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt()
    };
    let peak_absolute_error = errors
        .iter()
        .map(|error| error.abs())
        .fold(0.0_f64, f64::max);
    let uncertainty_samples = samples
        .iter()
        .filter_map(|sample| {
            sample
                .predicted_standard_deviation
                .map(|sigma| (sample.predicted_value - sample.reference_value).abs() <= 2.0 * sigma)
        })
        .collect::<Vec<_>>();
    let two_sigma_coverage = if uncertainty_samples.is_empty() {
        None
    } else {
        Some(
            uncertainty_samples
                .iter()
                .filter(|covered| **covered)
                .count() as f64
                / uncertainty_samples.len() as f64,
        )
    };

    if rmse > gate.maximum_rmse {
        failed_metrics.push("rmse".into());
    }
    if normalized_rmse > gate.maximum_normalized_rmse {
        failed_metrics.push("normalized_rmse".into());
    }
    if bias.abs() > gate.maximum_absolute_bias {
        failed_metrics.push("bias".into());
    }
    if peak_absolute_error > gate.maximum_peak_error {
        failed_metrics.push("peak_error".into());
    }
    if gate.minimum_two_sigma_coverage > 0.0 && two_sigma_coverage.is_none() {
        failed_metrics.push("missing_uncertainty_evidence".into());
    } else if two_sigma_coverage.is_some_and(|coverage| coverage < gate.minimum_two_sigma_coverage)
    {
        failed_metrics.push("two_sigma_coverage".into());
    }

    let incomplete = failed_metrics.iter().any(|failure| {
        failure == "insufficient_samples"
            || failure == "missing_independent_test_partition"
            || failure == "missing_uncertainty_evidence"
    });
    let failed = failed_metrics.iter().any(|failure| {
        failure != "insufficient_samples"
            && failure != "missing_independent_test_partition"
            && failure != "missing_uncertainty_evidence"
    });
    let status = if failed {
        ValidationMetricStatus::Fail
    } else if incomplete {
        ValidationMetricStatus::Incomplete
    } else {
        ValidationMetricStatus::Pass
    };
    SignalValidationMetrics {
        signal: gate.signal,
        status,
        sample_count,
        independent_test_samples,
        bias,
        rmse,
        normalized_rmse,
        peak_absolute_error,
        two_sigma_coverage,
        failed_metrics,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn fnv1a64(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FlightModelValidationConfig {
        FlightModelValidationConfig {
            minimum_samples_per_signal: 2,
            require_independent_test_partition: true,
            gates: vec![ValidationSignalGate {
                signal: FlightModelSignal::MainRotorRpm,
                maximum_rmse: 10.0,
                maximum_normalized_rmse: 0.05,
                maximum_absolute_bias: 5.0,
                maximum_peak_error: 20.0,
                minimum_two_sigma_coverage: 0.8,
            }],
        }
    }

    fn sample(predicted: f64, partition: ValidationPartition) -> FlightModelValidationSample {
        FlightModelValidationSample {
            scenario_id_hash: 1,
            time_s: 1.0,
            signal: FlightModelSignal::MainRotorRpm,
            partition,
            reference_value: 3_300.0,
            predicted_value: predicted,
            reference_scale: 3_300.0,
            predicted_standard_deviation: Some(5.0),
        }
    }

    #[test]
    fn held_out_accurate_samples_pass() {
        let mut validator = FlightModelValidator::new(config()).unwrap();
        validator
            .push(sample(3_302.0, ValidationPartition::Validation))
            .unwrap();
        validator
            .push(sample(3_298.0, ValidationPartition::IndependentTest))
            .unwrap();
        assert_eq!(
            validator.report().unwrap().status,
            ValidationMetricStatus::Pass
        );
    }

    #[test]
    fn large_bias_fails() {
        let mut validator = FlightModelValidator::new(config()).unwrap();
        validator
            .push(sample(3_400.0, ValidationPartition::Validation))
            .unwrap();
        validator
            .push(sample(3_400.0, ValidationPartition::IndependentTest))
            .unwrap();
        assert_eq!(
            validator.report().unwrap().status,
            ValidationMetricStatus::Fail
        );
    }

    #[test]
    fn missing_test_partition_is_incomplete() {
        let mut validator = FlightModelValidator::new(config()).unwrap();
        validator
            .push(sample(3_300.0, ValidationPartition::Validation))
            .unwrap();
        validator
            .push(sample(3_300.0, ValidationPartition::Validation))
            .unwrap();
        assert_eq!(
            validator.report().unwrap().status,
            ValidationMetricStatus::Incomplete
        );
    }

    #[test]
    fn nonfinite_sample_is_rejected() {
        let mut validator = FlightModelValidator::new(config()).unwrap();
        let mut invalid = sample(f64::NAN, ValidationPartition::IndependentTest);
        invalid.time_s = 1.0;
        assert_eq!(
            validator.push(invalid),
            Err(ModelValidationError::NonFiniteSample)
        );
    }
}
