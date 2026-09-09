// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! R4.0 measurement-only predictive baselines for humanoid transition evidence.
//!
//! This module deliberately starts with the hardest baseline to beat honestly:
//! persistence. It predicts that every semantic physical state value and
//! privileged root position remain unchanged across the recorded transition.
//!
//! Evaluation is dual-domain:
//! - raw physical residuals are retained per exact observation contract;
//! - heterogeneous units are aggregated only after normalization by contract
//!   span or the exact V1 quantization step `(max - min) / (bins - 1)`;
//! - policy-facing state is also compared in the existing R3.1 HDC geometry;
//! - privileged world-root trajectory error remains outside policy HDC.
//!
//! No result in this module affects control, safety, authority, or actuation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Write as _;

use symthaea_core::hdc::sensorimotor_contingencies::{
    SensorimotorComponentV1, SensorimotorFrameV1, SensorimotorHdcEncoderV1,
    SensorimotorMeasurementV1, SensorimotorObservationV1, SensorimotorQuantityV1,
    SensorimotorSubjectV1, SensorimotorUnitV1,
};

use crate::morphology::HumanoidMorphology;
use crate::transition_evidence::{
    HumanoidTransitionEvidenceErrorV1, HumanoidTransitionEvidenceV1,
};

const PREDICTION_STATE_DOMAIN_V1: &[u8] = b"symthaea.humanoid.predicted-state.v1\0";
const PREDICTION_REPORT_DOMAIN_V1: &[u8] = b"symthaea.humanoid.prediction-report.v1\0";

/// Versioned baseline family. Later learned predictors should use a separate
/// model/evidence identity rather than being smuggled in as another baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictionBaselineV1 {
    /// Predict post-state values and privileged root position are unchanged.
    Persistence,
}

impl HumanoidPredictionBaselineV1 {
    pub const fn predictor_id(self) -> &'static str {
        match self {
            Self::Persistence => "symthaea.humanoid.predictor.persistence.v1",
        }
    }
}

/// One exact-contract physical prediction residual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidScalarPredictionResidualV1 {
    pub contract_digest_hex: String,
    pub physical_role_digest_hex: String,
    pub subject: SensorimotorSubjectV1,
    pub quantity: SensorimotorQuantityV1,
    pub frame: SensorimotorFrameV1,
    pub component: SensorimotorComponentV1,
    pub unit: SensorimotorUnitV1,
    pub contract_min: f64,
    pub contract_max: f64,
    pub contract_bins: u16,
    pub predicted_value: f64,
    pub actual_value: f64,
    /// `predicted - actual`, in the declared physical unit.
    pub signed_error: f64,
    pub absolute_error: f64,
    pub contract_span: f64,
    /// Exact spacing between adjacent V1 quantization bins.
    pub quantization_step: f64,
    pub span_normalized_absolute_error: f64,
    pub absolute_error_bins: f64,
    /// True when exact scalar bits differ, independent of HDC quantization.
    pub changed: bool,
    /// V1 HDC saturates such values at the contract boundary. These flags keep
    /// that information visible beside HDC similarity.
    pub predicted_outside_contract: bool,
    pub actual_outside_contract: bool,
}

/// Content-addressed evaluation of one predictor against one causal transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidPredictionReportV1 {
    pub schema_id: String,
    pub baseline: HumanoidPredictionBaselineV1,
    pub predictor_id: String,
    pub transition_digest_hex: String,
    pub prediction_state_digest_hex: String,
    pub morphology: HumanoidMorphology,
    pub applied_dt_seconds: f64,

    pub compared_measured_count: usize,
    pub predicted_missing_count: usize,
    pub actual_missing_count: usize,
    pub both_missing_count: usize,
    pub changed_scalar_count: usize,
    pub predicted_saturation_count: usize,
    pub actual_saturation_count: usize,

    /// Dimensionless aggregate errors. Raw errors remain in `residuals` and are
    /// never averaged across incompatible units.
    pub mean_span_normalized_absolute_error: f64,
    pub max_span_normalized_absolute_error: f64,
    pub mean_absolute_error_bins: f64,
    pub max_absolute_error_bins: f64,

    /// Similarity between predicted policy HDC and actual post-state policy HDC.
    /// Persistence therefore measures how much the semantic state changed in the
    /// exact existing R3.1 representation.
    pub policy_hdc_similarity: Option<f32>,

    /// Persistence predicts privileged world-root position is unchanged. This
    /// metric is intentionally separate from policy HDC.
    pub privileged_root_position_component_error_m: [Option<f64>; 3],
    pub privileged_root_position_error_norm_m: Option<f64>,

    /// Canonically ordered by exact contract digest.
    pub residuals: Vec<HumanoidScalarPredictionResidualV1>,
    pub report_digest_hex: String,
}

impl HumanoidPredictionReportV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.prediction-report.v1";

    pub fn validate(&self) -> Result<(), HumanoidPredictionBaselineErrorV1> {
        if self.schema_id != Self::SCHEMA_ID
            || self.predictor_id != self.baseline.predictor_id()
            || self.transition_digest_hex.trim().is_empty()
            || self.prediction_state_digest_hex.trim().is_empty()
        {
            return Err(HumanoidPredictionBaselineErrorV1::SchemaMismatch);
        }
        if !self.applied_dt_seconds.is_finite() || self.applied_dt_seconds <= 0.0 {
            return Err(HumanoidPredictionBaselineErrorV1::NonFiniteMetric);
        }
        if self.compared_measured_count != self.residuals.len() {
            return Err(HumanoidPredictionBaselineErrorV1::SummaryMismatch);
        }

        for value in [
            self.mean_span_normalized_absolute_error,
            self.max_span_normalized_absolute_error,
            self.mean_absolute_error_bins,
            self.max_absolute_error_bins,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(HumanoidPredictionBaselineErrorV1::NonFiniteMetric);
            }
        }
        if self
            .policy_hdc_similarity
            .is_some_and(|similarity| !similarity.is_finite())
        {
            return Err(HumanoidPredictionBaselineErrorV1::NonFiniteMetric);
        }
        if self
            .privileged_root_position_component_error_m
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
            || self
                .privileged_root_position_error_norm_m
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(HumanoidPredictionBaselineErrorV1::NonFiniteMetric);
        }

        let mut previous_digest: Option<&str> = None;
        for residual in &self.residuals {
            residual.validate()?;
            if previous_digest
                .is_some_and(|previous| previous >= residual.contract_digest_hex.as_str())
            {
                return Err(HumanoidPredictionBaselineErrorV1::ResidualOrderMismatch);
            }
            previous_digest = Some(&residual.contract_digest_hex);
        }

        let changed = self
            .residuals
            .iter()
            .filter(|residual| residual.changed)
            .count();
        let predicted_saturation = self
            .residuals
            .iter()
            .filter(|residual| residual.predicted_outside_contract)
            .count();
        let actual_saturation = self
            .residuals
            .iter()
            .filter(|residual| residual.actual_outside_contract)
            .count();
        if changed != self.changed_scalar_count
            || predicted_saturation != self.predicted_saturation_count
            || actual_saturation != self.actual_saturation_count
        {
            return Err(HumanoidPredictionBaselineErrorV1::SummaryMismatch);
        }

        let (mean_span, max_span, mean_bins, max_bins) = aggregate_residuals(&self.residuals)?;
        if mean_span.to_bits() != self.mean_span_normalized_absolute_error.to_bits()
            || max_span.to_bits() != self.max_span_normalized_absolute_error.to_bits()
            || mean_bins.to_bits() != self.mean_absolute_error_bins.to_bits()
            || max_bins.to_bits() != self.max_absolute_error_bins.to_bits()
        {
            return Err(HumanoidPredictionBaselineErrorV1::SummaryMismatch);
        }

        if self.report_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidPredictionBaselineErrorV1::DigestMismatch);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidPredictionBaselineErrorV1> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PREDICTION_REPORT_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, self.baseline.predictor_id());
        feed_str(&mut hasher, &self.predictor_id);
        feed_str(&mut hasher, &self.transition_digest_hex);
        feed_str(&mut hasher, &self.prediction_state_digest_hex);
        feed_str(&mut hasher, self.morphology.schema_id());
        hasher.update(&self.applied_dt_seconds.to_bits().to_le_bytes());

        for count in [
            self.compared_measured_count,
            self.predicted_missing_count,
            self.actual_missing_count,
            self.both_missing_count,
            self.changed_scalar_count,
            self.predicted_saturation_count,
            self.actual_saturation_count,
        ] {
            hasher.update(&(count as u64).to_le_bytes());
        }

        for value in [
            self.mean_span_normalized_absolute_error,
            self.max_span_normalized_absolute_error,
            self.mean_absolute_error_bins,
            self.max_absolute_error_bins,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        feed_option_f32(&mut hasher, self.policy_hdc_similarity);
        for value in self.privileged_root_position_component_error_m {
            feed_option_f64(&mut hasher, value);
        }
        feed_option_f64(&mut hasher, self.privileged_root_position_error_norm_m);

        hasher.update(&(self.residuals.len() as u64).to_le_bytes());
        for residual in &self.residuals {
            residual.feed_digest(&mut hasher)?;
        }
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

impl HumanoidScalarPredictionResidualV1 {
    fn validate(&self) -> Result<(), HumanoidPredictionBaselineErrorV1> {
        if self.contract_digest_hex.len() != 64
            || self.physical_role_digest_hex.len() != 64
            || !(2..=4096).contains(&self.contract_bins)
        {
            return Err(HumanoidPredictionBaselineErrorV1::SchemaMismatch);
        }
        for value in [
            self.contract_min,
            self.contract_max,
            self.predicted_value,
            self.actual_value,
            self.signed_error,
            self.absolute_error,
            self.contract_span,
            self.quantization_step,
            self.span_normalized_absolute_error,
            self.absolute_error_bins,
        ] {
            if !value.is_finite() {
                return Err(HumanoidPredictionBaselineErrorV1::NonFiniteMetric);
            }
        }
        if self.contract_max <= self.contract_min
            || self.absolute_error < 0.0
            || self.contract_span <= 0.0
            || self.quantization_step <= 0.0
            || self.span_normalized_absolute_error < 0.0
            || self.absolute_error_bins < 0.0
        {
            return Err(HumanoidPredictionBaselineErrorV1::NonFiniteMetric);
        }

        let expected_span = self.contract_max - self.contract_min;
        let expected_quantization_step = expected_span / f64::from(self.contract_bins - 1);
        let expected_signed_error = self.predicted_value - self.actual_value;
        let expected_absolute_error = expected_signed_error.abs();
        let expected_span_normalized = expected_absolute_error / expected_span;
        let expected_bins = expected_absolute_error / expected_quantization_step;
        let expected_changed = self.predicted_value.to_bits() != self.actual_value.to_bits();
        let expected_predicted_outside =
            self.predicted_value < self.contract_min || self.predicted_value > self.contract_max;
        let expected_actual_outside =
            self.actual_value < self.contract_min || self.actual_value > self.contract_max;

        if self.contract_span.to_bits() != expected_span.to_bits()
            || self.quantization_step.to_bits() != expected_quantization_step.to_bits()
            || self.signed_error.to_bits() != expected_signed_error.to_bits()
            || self.absolute_error.to_bits() != expected_absolute_error.to_bits()
            || self.span_normalized_absolute_error.to_bits() != expected_span_normalized.to_bits()
            || self.absolute_error_bins.to_bits() != expected_bins.to_bits()
            || self.changed != expected_changed
            || self.predicted_outside_contract != expected_predicted_outside
            || self.actual_outside_contract != expected_actual_outside
        {
            return Err(HumanoidPredictionBaselineErrorV1::SummaryMismatch);
        }
        Ok(())
    }

    fn feed_digest(
        &self,
        hasher: &mut blake3::Hasher,
    ) -> Result<(), HumanoidPredictionBaselineErrorV1> {
        self.validate()?;
        feed_str(hasher, &self.contract_digest_hex);
        feed_str(hasher, &self.physical_role_digest_hex);
        feed_str(
            hasher,
            &serde_json::to_string(&self.subject)
                .map_err(|_| HumanoidPredictionBaselineErrorV1::Serialization)?,
        );
        feed_str(
            hasher,
            &serde_json::to_string(&self.quantity)
                .map_err(|_| HumanoidPredictionBaselineErrorV1::Serialization)?,
        );
        feed_str(
            hasher,
            &serde_json::to_string(&self.frame)
                .map_err(|_| HumanoidPredictionBaselineErrorV1::Serialization)?,
        );
        feed_str(
            hasher,
            &serde_json::to_string(&self.component)
                .map_err(|_| HumanoidPredictionBaselineErrorV1::Serialization)?,
        );
        feed_str(
            hasher,
            &serde_json::to_string(&self.unit)
                .map_err(|_| HumanoidPredictionBaselineErrorV1::Serialization)?,
        );
        hasher.update(&self.contract_min.to_bits().to_le_bytes());
        hasher.update(&self.contract_max.to_bits().to_le_bytes());
        hasher.update(&self.contract_bins.to_le_bytes());
        for value in [
            self.predicted_value,
            self.actual_value,
            self.signed_error,
            self.absolute_error,
            self.contract_span,
            self.quantization_step,
            self.span_normalized_absolute_error,
            self.absolute_error_bins,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        hasher.update(&[
            u8::from(self.changed),
            u8::from(self.predicted_outside_contract),
            u8::from(self.actual_outside_contract),
        ]);
        Ok(())
    }
}

/// Evaluate the persistence baseline against one fully validated privileged-truth
/// causal transition.
pub fn evaluate_persistence_baseline_v1(
    transition: &HumanoidTransitionEvidenceV1,
) -> Result<HumanoidPredictionReportV1, HumanoidPredictionBaselineErrorV1> {
    transition
        .validate()
        .map_err(HumanoidPredictionBaselineErrorV1::Transition)?;
    if !transition.r4_privileged_truth_ready() {
        return Err(HumanoidPredictionBaselineErrorV1::TransitionNotPrivilegedTruth);
    }

    let pre = index_observations(&transition.pre_state.policy_observations)?;
    let post = index_observations(&transition.post_state.policy_observations)?;
    if pre.keys().ne(post.keys()) {
        return Err(HumanoidPredictionBaselineErrorV1::ContractSetMismatch);
    }

    let mut residuals = Vec::new();
    let mut predicted_missing_count = 0usize;
    let mut actual_missing_count = 0usize;
    let mut both_missing_count = 0usize;

    for (digest, predicted) in &pre {
        let actual = post
            .get(digest)
            .ok_or(HumanoidPredictionBaselineErrorV1::ContractSetMismatch)?;
        match (*predicted, *actual) {
            (
                SensorimotorObservationV1::Measured(predicted_measurement),
                SensorimotorObservationV1::Measured(actual_measurement),
            ) => residuals.push(build_residual(predicted_measurement, actual_measurement)?),
            (SensorimotorObservationV1::Missing { .. }, SensorimotorObservationV1::Measured(_)) => {
                predicted_missing_count += 1;
            }
            (SensorimotorObservationV1::Measured(_), SensorimotorObservationV1::Missing { .. }) => {
                actual_missing_count += 1;
            }
            (
                SensorimotorObservationV1::Missing { .. },
                SensorimotorObservationV1::Missing { .. },
            ) => {
                both_missing_count += 1;
            }
        }
    }

    if residuals.is_empty() {
        return Err(HumanoidPredictionBaselineErrorV1::NoComparableMeasurements);
    }
    residuals.sort_by(|left, right| left.contract_digest_hex.cmp(&right.contract_digest_hex));

    let changed_scalar_count = residuals
        .iter()
        .filter(|residual| residual.changed)
        .count();
    let predicted_saturation_count = residuals
        .iter()
        .filter(|residual| residual.predicted_outside_contract)
        .count();
    let actual_saturation_count = residuals
        .iter()
        .filter(|residual| residual.actual_outside_contract)
        .count();
    let (mean_span, max_span, mean_bins, max_bins) = aggregate_residuals(&residuals)?;

    let hdc_encoder = SensorimotorHdcEncoderV1;
    let predicted_hdc = hdc_encoder
        .encode_observations(&transition.pre_state.policy_observations)
        .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;
    let actual_hdc = hdc_encoder
        .encode_observations(&transition.post_state.policy_observations)
        .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;
    let policy_hdc_similarity = match (predicted_hdc, actual_hdc) {
        (Some(predicted), Some(actual)) => Some(predicted.similarity(&actual)),
        _ => None,
    };

    let (root_components, root_norm) = privileged_root_persistence_error(transition);
    let prediction_state_digest_hex = persistence_prediction_digest(transition)?;

    let mut report = HumanoidPredictionReportV1 {
        schema_id: HumanoidPredictionReportV1::SCHEMA_ID.to_string(),
        baseline: HumanoidPredictionBaselineV1::Persistence,
        predictor_id: HumanoidPredictionBaselineV1::Persistence
            .predictor_id()
            .to_string(),
        transition_digest_hex: transition.transition_digest_hex.clone(),
        prediction_state_digest_hex,
        morphology: transition.morphology,
        applied_dt_seconds: transition.applied_dt_seconds(),
        compared_measured_count: residuals.len(),
        predicted_missing_count,
        actual_missing_count,
        both_missing_count,
        changed_scalar_count,
        predicted_saturation_count,
        actual_saturation_count,
        mean_span_normalized_absolute_error: mean_span,
        max_span_normalized_absolute_error: max_span,
        mean_absolute_error_bins: mean_bins,
        max_absolute_error_bins: max_bins,
        policy_hdc_similarity,
        privileged_root_position_component_error_m: root_components,
        privileged_root_position_error_norm_m: root_norm,
        residuals,
        report_digest_hex: String::new(),
    };
    report.report_digest_hex = report.compute_digest_hex()?;
    report.validate()?;
    Ok(report)
}

fn index_observations<'a>(
    observations: &'a [SensorimotorObservationV1],
) -> Result<BTreeMap<[u8; 32], &'a SensorimotorObservationV1>, HumanoidPredictionBaselineErrorV1> {
    let mut indexed = BTreeMap::new();
    for observation in observations {
        observation
            .validate()
            .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;
        let digest = observation
            .address()
            .semantic_digest()
            .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;
        if indexed.insert(digest, observation).is_some() {
            return Err(HumanoidPredictionBaselineErrorV1::DuplicateContract);
        }
    }
    Ok(indexed)
}

fn build_residual(
    predicted: &SensorimotorMeasurementV1,
    actual: &SensorimotorMeasurementV1,
) -> Result<HumanoidScalarPredictionResidualV1, HumanoidPredictionBaselineErrorV1> {
    predicted
        .validate()
        .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;
    actual
        .validate()
        .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;
    let predicted_digest = predicted
        .address
        .semantic_digest()
        .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;
    let actual_digest = actual
        .address
        .semantic_digest()
        .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;
    if predicted_digest != actual_digest {
        return Err(HumanoidPredictionBaselineErrorV1::ContractSetMismatch);
    }

    let contract = &predicted.address.value_contract;
    let span = contract.max - contract.min;
    let quantization_step = span / f64::from(contract.bins - 1);
    let signed_error = predicted.value - actual.value;
    let absolute_error = signed_error.abs();
    let physical_role_digest = predicted
        .address
        .physical_role_digest()
        .map_err(HumanoidPredictionBaselineErrorV1::Sensorimotor)?;

    let residual = HumanoidScalarPredictionResidualV1 {
        contract_digest_hex: digest_hex(&predicted_digest),
        physical_role_digest_hex: digest_hex(&physical_role_digest),
        subject: predicted.address.subject.clone(),
        quantity: predicted.address.quantity.clone(),
        frame: predicted.address.frame.clone(),
        component: predicted.address.component,
        unit: contract.unit.clone(),
        contract_min: contract.min,
        contract_max: contract.max,
        contract_bins: contract.bins,
        predicted_value: predicted.value,
        actual_value: actual.value,
        signed_error,
        absolute_error,
        contract_span: span,
        quantization_step,
        span_normalized_absolute_error: absolute_error / span,
        absolute_error_bins: absolute_error / quantization_step,
        changed: predicted.value.to_bits() != actual.value.to_bits(),
        predicted_outside_contract: predicted.value < contract.min || predicted.value > contract.max,
        actual_outside_contract: actual.value < contract.min || actual.value > contract.max,
    };
    residual.validate()?;
    Ok(residual)
}

fn aggregate_residuals(
    residuals: &[HumanoidScalarPredictionResidualV1],
) -> Result<(f64, f64, f64, f64), HumanoidPredictionBaselineErrorV1> {
    if residuals.is_empty() {
        return Err(HumanoidPredictionBaselineErrorV1::NoComparableMeasurements);
    }
    let count = residuals.len() as f64;
    let mut sum_span = 0.0;
    let mut max_span: f64 = 0.0;
    let mut sum_bins = 0.0;
    let mut max_bins: f64 = 0.0;
    for residual in residuals {
        residual.validate()?;
        sum_span += residual.span_normalized_absolute_error;
        max_span = max_span.max(residual.span_normalized_absolute_error);
        sum_bins += residual.absolute_error_bins;
        max_bins = max_bins.max(residual.absolute_error_bins);
    }
    Ok((sum_span / count, max_span, sum_bins / count, max_bins))
}

fn privileged_root_persistence_error(
    transition: &HumanoidTransitionEvidenceV1,
) -> ([Option<f64>; 3], Option<f64>) {
    let components = std::array::from_fn(|index| {
        match (
            transition.pre_state.privileged_root_position_world_m[index],
            transition.post_state.privileged_root_position_world_m[index],
        ) {
            (Some(predicted), Some(actual)) => Some(predicted - actual),
            _ => None,
        }
    });
    let norm = if components.iter().all(Option::is_some) {
        Some(
            components
                .iter()
                .map(|value| value.expect("all root components present").powi(2))
                .sum::<f64>()
                .sqrt(),
        )
    } else {
        None
    };
    (components, norm)
}

fn persistence_prediction_digest(
    transition: &HumanoidTransitionEvidenceV1,
) -> Result<String, HumanoidPredictionBaselineErrorV1> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREDICTION_STATE_DOMAIN_V1);
    feed_str(
        &mut hasher,
        HumanoidPredictionBaselineV1::Persistence.predictor_id(),
    );
    feed_str(&mut hasher, transition.morphology.schema_id());
    let indexed = index_observations(&transition.pre_state.policy_observations)?;
    hasher.update(&(indexed.len() as u64).to_le_bytes());
    for (digest, observation) in indexed {
        hasher.update(&digest);
        match observation {
            SensorimotorObservationV1::Measured(measurement) => {
                hasher.update(&[1]);
                hasher.update(&measurement.value.to_bits().to_le_bytes());
            }
            SensorimotorObservationV1::Missing { .. } => {
                hasher.update(&[0]);
            }
        }
    }
    for value in transition.pre_state.privileged_root_position_world_m {
        feed_option_f64(&mut hasher, value);
    }
    Ok(digest_hex(hasher.finalize().as_bytes()))
}

fn feed_option_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn feed_option_f32(hasher: &mut blake3::Hasher, value: Option<f32>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Debug)]
pub enum HumanoidPredictionBaselineErrorV1 {
    Transition(HumanoidTransitionEvidenceErrorV1),
    TransitionNotPrivilegedTruth,
    Sensorimotor(&'static str),
    DuplicateContract,
    ContractSetMismatch,
    NoComparableMeasurements,
    SchemaMismatch,
    SummaryMismatch,
    ResidualOrderMismatch,
    NonFiniteMetric,
    Serialization,
    DigestMismatch,
}

impl fmt::Display for HumanoidPredictionBaselineErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transition(error) => write!(f, "invalid transition evidence: {error}"),
            Self::TransitionNotPrivilegedTruth => {
                write!(
                    f,
                    "R4.0 baseline requires privileged-truth transition evidence"
                )
            }
            Self::Sensorimotor(message) => write!(f, "sensorimotor schema: {message}"),
            Self::DuplicateContract => write!(f, "duplicate exact sensorimotor contract in frame"),
            Self::ContractSetMismatch => write!(f, "pre/post sensorimotor contract sets differ"),
            Self::NoComparableMeasurements => {
                write!(f, "transition has no comparable measured values")
            }
            Self::SchemaMismatch => write!(f, "unsupported prediction report schema"),
            Self::SummaryMismatch => {
                write!(f, "prediction report summary does not match residuals")
            }
            Self::ResidualOrderMismatch => {
                write!(f, "prediction residuals are not canonically ordered")
            }
            Self::NonFiniteMetric => write!(f, "prediction report contains an invalid metric"),
            Self::Serialization => write!(f, "prediction report field serialization failed"),
            Self::DigestMismatch => write!(f, "prediction report content digest mismatch"),
        }
    }
}

impl std::error::Error for HumanoidPredictionBaselineErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_action_evidence::InstrumentedSimpleHumanoidSimulator;
    use crate::simulator::HumanoidPhysicsSimulator;
    use crate::transition_evidence::capture_privileged_transition_v1;
    use crate::types::HumanoidCommand;

    fn command(morphology: HumanoidMorphology, gain: f32) -> HumanoidCommand {
        let mut torques = vec![0.0f32; morphology.num_actuators()];
        if !torques.is_empty() {
            torques[0] = gain;
        }
        if torques.len() > 5 {
            torques[5] = -0.6 * gain;
        }
        if torques.len() > 8 {
            torques[8] = 0.45 * gain;
        }
        HumanoidCommand { torques }
    }

    fn capture(seed: u64, gain: f32) -> HumanoidTransitionEvidenceV1 {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.reset_with_perturbation(0.0, seed);
        capture_privileged_transition_v1(
            &mut simulator,
            &command(morphology, gain),
            0.025,
            "simple-sim-clock",
            "baseline-episode",
        )
        .unwrap()
    }

    #[test]
    fn persistence_baseline_reports_physical_hdc_and_root_metrics() {
        let transition = capture(31, 0.8);
        let report = evaluate_persistence_baseline_v1(&transition).unwrap();
        report.validate().unwrap();

        assert_eq!(
            report.compared_measured_count,
            transition.pre_state.policy_observations.len()
        );
        assert!(report.changed_scalar_count > 0);
        assert!(report.mean_absolute_error_bins > 0.0);
        assert!(report.policy_hdc_similarity.is_some_and(f32::is_finite));
        assert!(
            report
                .privileged_root_position_error_norm_m
                .is_some_and(f64::is_finite)
        );
        assert_eq!(
            report.predictor_id,
            HumanoidPredictionBaselineV1::Persistence.predictor_id()
        );
    }

    #[test]
    fn deterministic_transition_produces_bit_exact_report() {
        let left = evaluate_persistence_baseline_v1(&capture(99, 0.7)).unwrap();
        let right = evaluate_persistence_baseline_v1(&capture(99, 0.7)).unwrap();

        assert_eq!(left.report_digest_hex, right.report_digest_hex);
        assert_eq!(
            left.prediction_state_digest_hex,
            right.prediction_state_digest_hex
        );
        assert_eq!(
            left.mean_absolute_error_bins.to_bits(),
            right.mean_absolute_error_bins.to_bits()
        );
        assert_eq!(
            left.policy_hdc_similarity.map(f32::to_bits),
            right.policy_hdc_similarity.map(f32::to_bits)
        );
    }

    #[test]
    fn persistence_prediction_is_action_blind_but_evaluation_is_not() {
        let quiet_transition = capture(123, 0.1);
        let strong_transition = capture(123, 0.9);
        let quiet = evaluate_persistence_baseline_v1(&quiet_transition).unwrap();
        let strong = evaluate_persistence_baseline_v1(&strong_transition).unwrap();

        assert_eq!(
            quiet.prediction_state_digest_hex,
            strong.prediction_state_digest_hex
        );
        assert_ne!(
            quiet_transition.transition_digest_hex,
            strong_transition.transition_digest_hex
        );
        assert_ne!(quiet.report_digest_hex, strong.report_digest_hex);
    }

    #[test]
    fn report_digest_detects_residual_tampering() {
        let transition = capture(7, 0.75);
        let mut report = evaluate_persistence_baseline_v1(&transition).unwrap();
        report.residuals[0].actual_value += 0.01;
        assert!(report.validate().is_err());
    }

    #[test]
    fn residual_uses_exact_v1_bin_spacing_and_exposes_saturation() {
        use symthaea_core::hdc::sensorimotor_contingencies::{
            SensorimotorAddressV1, SensorimotorValueContractV1,
        };

        let address = SensorimotorAddressV1::new(
            SensorimotorSubjectV1::BodyRoot,
            SensorimotorQuantityV1::AngularVelocity,
            SensorimotorFrameV1::Body,
            SensorimotorComponentV1::X,
            SensorimotorValueContractV1 {
                unit: SensorimotorUnitV1::RadianPerSecond,
                min: -20.0,
                max: 20.0,
                bins: 401,
            },
        );
        let predicted = SensorimotorMeasurementV1 {
            address: address.clone(),
            value: 0.0,
        };
        let actual = SensorimotorMeasurementV1 {
            address,
            value: 25.0,
        };
        let residual = build_residual(&predicted, &actual).unwrap();

        residual.validate().unwrap();
        assert_eq!(residual.contract_min.to_bits(), (-20.0f64).to_bits());
        assert_eq!(residual.contract_max.to_bits(), 20.0f64.to_bits());
        assert_eq!(residual.contract_bins, 401);
        assert!((residual.quantization_step - 0.1).abs() < 1.0e-12);
        assert!((residual.absolute_error_bins - 250.0).abs() < 1.0e-9);
        assert!(!residual.predicted_outside_contract);
        assert!(residual.actual_outside_contract);
    }
}
