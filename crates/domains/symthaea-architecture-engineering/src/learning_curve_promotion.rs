//! Split-conformal learning-curve prediction and guarded fidelity promotion.
//!
//! The model is intentionally transparent: for each rung it regresses final metric on the
//! latest observed score, plus the score-per-resource slope once two rungs are available.
//! A disjoint residual-calibration split supplies a finite-sample absolute-residual quantile.
//!
//! Promotion exploits conservative bounds but reserves guard slots for candidates with the
//! strongest optimistic bounds. This is a calibrated baseline, not a replacement for LCPFN,
//! Gaussian-process curve models or HiP-BO. Marginal conformal coverage relies on exchangeability
//! between calibration and evaluation architectures; distribution shift can invalidate it.

use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

use super::nasbench201_trajectory::ProxyDirection;

const CALIBRATION_DOMAIN: &[u8] = b"symthaea.learning-curve-calibration.v0.1";
const CURVE_DATA_DOMAIN: &[u8] = b"symthaea.learning-curve-samples.v0.1";
const SELECTION_DOMAIN: &[u8] = b"symthaea.learning-curve-promotion-selection.v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurveFidelityLevel {
    pub id: String,
    pub cumulative_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LearningCurveCalibrationConfig {
    pub benchmark_id: String,
    pub target_metric_id: String,
    pub metric_direction: ProxyDirection,
    pub source_artifact_sha256: String,
    pub model_fit_split_id: String,
    pub residual_calibration_split_id: String,
    pub evaluation_split_id: String,
    pub fidelity_levels: Vec<CurveFidelityLevel>,
    /// Miscoverage level for the split-conformal residual interval.
    pub alpha: f64,
    pub minimum_model_fit_samples: usize,
    pub minimum_residual_calibration_samples: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LearningCurveCalibrationSample {
    pub architecture_id: String,
    /// One observed metric value for every declared rung, including the final rung.
    pub curve_values: Vec<f64>,
    /// Must agree with the final curve value.
    pub final_value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurveRungModel {
    pub fidelity_index: usize,
    pub coefficients: Vec<f64>,
    /// Inverse X'X, used to form a candidate-dependent normalized interval radius.
    pub inverse_gram: Vec<Vec<f64>>,
    pub normalized_residual_quantile: f64,
    pub fit_sample_count: usize,
    pub residual_calibration_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LearningCurveCalibrator {
    pub config: LearningCurveCalibrationConfig,
    pub model_fit_data_sha256: String,
    pub residual_calibration_data_sha256: String,
    pub rung_models: Vec<CurveRungModel>,
    pub calibration_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurvePrediction {
    pub architecture_id: String,
    pub fidelity_index: usize,
    pub observed_curve: Vec<f64>,
    pub predicted_final_value: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub interval_radius: f64,
    pub conservative_bound: f64,
    pub optimistic_bound: f64,
    pub leverage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionReason {
    ConservativeBound,
    OptimisticGuard,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedCurveCandidate {
    pub architecture_id: String,
    pub reason: PromotionReason,
    pub predicted_final_value: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub interval_radius: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurvePromotionSelection {
    pub calibration_sha256: String,
    pub fidelity_index: usize,
    pub requested_promotion_count: usize,
    pub guard_slot_count: usize,
    pub selected: Vec<SelectedCurveCandidate>,
    pub all_predictions: Vec<CurvePrediction>,
    pub selection_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearningCurvePromotionError {
    pub field: String,
    pub message: String,
}

impl LearningCurvePromotionError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for LearningCurvePromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for LearningCurvePromotionError {}

impl LearningCurveCalibrator {
    /// Fit on development architectures and calibrate normalized residuals on a disjoint set.
    pub fn fit(
        config: LearningCurveCalibrationConfig,
        model_fit_samples: &[LearningCurveCalibrationSample],
        residual_calibration_samples: &[LearningCurveCalibrationSample],
    ) -> Result<Self, LearningCurvePromotionError> {
        validate_config(&config)?;
        validate_samples(
            model_fit_samples,
            &config,
            config.minimum_model_fit_samples,
            "model_fit_samples",
        )?;
        validate_samples(
            residual_calibration_samples,
            &config,
            config.minimum_residual_calibration_samples,
            "residual_calibration_samples",
        )?;

        let model_ids: HashSet<&str> = model_fit_samples
            .iter()
            .map(|sample| sample.architecture_id.as_str())
            .collect();
        if residual_calibration_samples
            .iter()
            .any(|sample| model_ids.contains(sample.architecture_id.as_str()))
        {
            return Err(LearningCurvePromotionError::new(
                "residual_calibration_samples",
                "model-fit and residual-calibration architecture IDs must be disjoint",
            ));
        }

        let model_fit_data_sha256 =
            digest_samples("model-fit", model_fit_samples, &config.fidelity_levels);
        let residual_calibration_data_sha256 = digest_samples(
            "residual-calibration",
            residual_calibration_samples,
            &config.fidelity_levels,
        );

        let mut rung_models = Vec::with_capacity(config.fidelity_levels.len() - 1);
        for fidelity_index in 0..(config.fidelity_levels.len() - 1) {
            let fit_rows: Vec<Vec<f64>> = model_fit_samples
                .iter()
                .map(|sample| curve_features(&sample.curve_values, fidelity_index, &config.fidelity_levels))
                .collect();
            let fit_targets: Vec<f64> = model_fit_samples
                .iter()
                .map(|sample| sample.final_value)
                .collect();
            let (coefficients, inverse_gram) = fit_linear_model(&fit_rows, &fit_targets)?;

            let mut normalized_residuals = Vec::with_capacity(residual_calibration_samples.len());
            for sample in residual_calibration_samples {
                let features = curve_features(
                    &sample.curve_values,
                    fidelity_index,
                    &config.fidelity_levels,
                );
                let prediction = dot(&coefficients, &features);
                let leverage = quadratic_form(&features, &inverse_gram)?;
                let normalization = (1.0 + leverage.max(0.0)).sqrt();
                let residual = (sample.final_value - prediction).abs() / normalization;
                if !residual.is_finite() {
                    return Err(LearningCurvePromotionError::new(
                        "residual_calibration_samples",
                        "normalized residual is non-finite",
                    ));
                }
                normalized_residuals.push(residual);
            }
            normalized_residuals.sort_by(f64::total_cmp);
            let quantile_index = ((normalized_residuals.len() + 1) as f64
                * (1.0 - config.alpha))
                .ceil() as usize;
            if quantile_index == 0 || quantile_index > normalized_residuals.len() {
                return Err(LearningCurvePromotionError::new(
                    "alpha",
                    "finite-sample conformal quantile is unavailable at this calibration sample count",
                ));
            }
            let normalized_residual_quantile = normalized_residuals[quantile_index - 1];
            rung_models.push(CurveRungModel {
                fidelity_index,
                coefficients,
                inverse_gram,
                normalized_residual_quantile,
                fit_sample_count: model_fit_samples.len(),
                residual_calibration_count: residual_calibration_samples.len(),
            });
        }

        let calibration_sha256 = digest_calibrator(
            &config,
            &model_fit_data_sha256,
            &residual_calibration_data_sha256,
            &rung_models,
        );
        Ok(Self {
            config,
            model_fit_data_sha256,
            residual_calibration_data_sha256,
            rung_models,
            calibration_sha256,
        })
    }

    /// Recompute and verify the frozen model and interval-calibration digest.
    pub fn verify_integrity(&self) -> Result<(), LearningCurvePromotionError> {
        validate_config(&self.config)?;
        let valid_digest = |value: &str| {
            value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        };
        if !valid_digest(&self.model_fit_data_sha256)
            || !valid_digest(&self.residual_calibration_data_sha256)
            || self.rung_models.len() + 1 != self.config.fidelity_levels.len()
        {
            return Err(LearningCurvePromotionError::new(
                "calibrator",
                "calibration provenance or rung-model count is invalid",
            ));
        }
        for (index, model) in self.rung_models.iter().enumerate() {
            let expected_parameters = if index == 0 { 2 } else { 3 };
            if model.fidelity_index != index
                || model.coefficients.len() != expected_parameters
                || model.inverse_gram.len() != expected_parameters
                || model.inverse_gram.iter().any(|row| row.len() != expected_parameters)
                || model.coefficients.iter().any(|value| !value.is_finite())
                || model.inverse_gram.iter().flatten().any(|value| !value.is_finite())
                || !model.normalized_residual_quantile.is_finite()
                || model.normalized_residual_quantile < 0.0
                || model.fit_sample_count < self.config.minimum_model_fit_samples
                || model.residual_calibration_count
                    < self.config.minimum_residual_calibration_samples
            {
                return Err(LearningCurvePromotionError::new(
                    format!("rung_models[{index}]"),
                    "model dimensions, coefficients, quantile or calibration counts are invalid",
                ));
            }
        }
        let expected = digest_calibrator(
            &self.config,
            &self.model_fit_data_sha256,
            &self.residual_calibration_data_sha256,
            &self.rung_models,
        );
        if expected != self.calibration_sha256 {
            return Err(LearningCurvePromotionError::new(
                "calibration_sha256",
                "calibrator content does not match its digest",
            ));
        }
        Ok(())
    }

    /// Predict final-fidelity performance from a candidate's observed prefix.
    /// The prefix must end before final fidelity.
    pub fn predict_final(
        &self,
        architecture_id: &str,
        observed_curve: &[f64],
    ) -> Result<CurvePrediction, LearningCurvePromotionError> {
        self.verify_integrity()?;
        if architecture_id.trim().is_empty() {
            return Err(LearningCurvePromotionError::new(
                "architecture_id",
                "must be non-empty",
            ));
        }
        if observed_curve.is_empty() || observed_curve.len() >= self.config.fidelity_levels.len() {
            return Err(LearningCurvePromotionError::new(
                "observed_curve",
                "must contain a non-empty prefix and stop before final fidelity",
            ));
        }
        if observed_curve.iter().any(|value| !value.is_finite()) {
            return Err(LearningCurvePromotionError::new(
                "observed_curve",
                "all observations must be finite",
            ));
        }

        let fidelity_index = observed_curve.len() - 1;
        let model = self.rung_models.get(fidelity_index).ok_or_else(|| {
            LearningCurvePromotionError::new("fidelity_index", "no calibration model for this rung")
        })?;
        let features = curve_features(observed_curve, fidelity_index, &self.config.fidelity_levels);
        let predicted_final_value = dot(&model.coefficients, &features);
        let leverage = quadratic_form(&features, &model.inverse_gram)?;
        if !predicted_final_value.is_finite() || !leverage.is_finite() {
            return Err(LearningCurvePromotionError::new(
                "prediction",
                "prediction or leverage is non-finite",
            ));
        }
        let interval_radius = model.normalized_residual_quantile
            * (1.0 + leverage.max(0.0)).sqrt();
        let lower_bound = predicted_final_value - interval_radius;
        let upper_bound = predicted_final_value + interval_radius;
        let (conservative_bound, optimistic_bound) = match self.config.metric_direction {
            ProxyDirection::Maximize => (lower_bound, upper_bound),
            ProxyDirection::Minimize => (upper_bound, lower_bound),
        };
        Ok(CurvePrediction {
            architecture_id: architecture_id.to_owned(),
            fidelity_index,
            observed_curve: observed_curve.to_vec(),
            predicted_final_value,
            lower_bound,
            upper_bound,
            interval_radius,
            conservative_bound,
            optimistic_bound,
            leverage,
        })
    }
}

/// Select the top conservative-bound candidates, then spend guard slots on the strongest
/// optimistic-bound candidates not already selected. This explicitly protects uncertain
/// candidates whose final performance could plausibly be strong.
pub fn select_curve_promotions(
    calibrator: &LearningCurveCalibrator,
    predictions: &[CurvePrediction],
    promotion_count: usize,
    guard_slot_count: usize,
) -> Result<CurvePromotionSelection, LearningCurvePromotionError> {
    calibrator.verify_integrity()?;
    if predictions.is_empty() || promotion_count == 0 || promotion_count > predictions.len() {
        return Err(LearningCurvePromotionError::new(
            "promotion_count",
            "must be in 1..=prediction_count",
        ));
    }
    if guard_slot_count > promotion_count {
        return Err(LearningCurvePromotionError::new(
            "guard_slot_count",
            "cannot exceed promotion_count",
        ));
    }
    let fidelity_index = predictions[0].fidelity_index;
    let mut ids = HashSet::with_capacity(predictions.len());
    for (index, prediction) in predictions.iter().enumerate() {
        if prediction.architecture_id.trim().is_empty()
            || !ids.insert(prediction.architecture_id.as_str())
        {
            return Err(LearningCurvePromotionError::new(
                format!("predictions[{index}].architecture_id"),
                "candidate IDs must be non-empty and unique",
            ));
        }
        if prediction.fidelity_index != fidelity_index
            || prediction.fidelity_index >= calibrator.rung_models.len()
            || !prediction.predicted_final_value.is_finite()
            || !prediction.lower_bound.is_finite()
            || !prediction.upper_bound.is_finite()
            || !prediction.interval_radius.is_finite()
            || prediction.interval_radius < 0.0
            || prediction.lower_bound > prediction.upper_bound
        {
            return Err(LearningCurvePromotionError::new(
                format!("predictions[{index}]"),
                "predictions must share a valid rung and contain finite ordered bounds",
            ));
        }
        let recomputed = calibrator.predict_final(
            &prediction.architecture_id,
            &prediction.observed_curve,
        )?;
        if recomputed != *prediction {
            return Err(LearningCurvePromotionError::new(
                format!("predictions[{index}]"),
                "prediction fields do not match the frozen calibrator and observed curve",
            ));
        }
        let expected_conservative = match calibrator.config.metric_direction {
            ProxyDirection::Maximize => prediction.lower_bound,
            ProxyDirection::Minimize => prediction.upper_bound,
        };
        let expected_optimistic = match calibrator.config.metric_direction {
            ProxyDirection::Maximize => prediction.upper_bound,
            ProxyDirection::Minimize => prediction.lower_bound,
        };
        if prediction.conservative_bound.to_bits() != expected_conservative.to_bits()
            || prediction.optimistic_bound.to_bits() != expected_optimistic.to_bits()
        {
            return Err(LearningCurvePromotionError::new(
                format!("predictions[{index}]"),
                "objective-oriented bounds do not match metric direction",
            ));
        }
    }

    let mut exploitation_order: Vec<&CurvePrediction> = predictions.iter().collect();
    exploitation_order.sort_by(|left, right| {
        let order = match calibrator.config.metric_direction {
            ProxyDirection::Maximize => right
                .conservative_bound
                .total_cmp(&left.conservative_bound),
            ProxyDirection::Minimize => left
                .conservative_bound
                .total_cmp(&right.conservative_bound),
        };
        order.then_with(|| left.architecture_id.cmp(&right.architecture_id))
    });

    let exploitation_count = promotion_count - guard_slot_count;
    let mut selected_ids = HashSet::with_capacity(promotion_count);
    let mut selected = Vec::with_capacity(promotion_count);
    for prediction in exploitation_order.iter().take(exploitation_count) {
        selected_ids.insert(prediction.architecture_id.as_str());
        selected.push(selected_candidate(
            prediction,
            PromotionReason::ConservativeBound,
        ));
    }

    let mut guard_order: Vec<&CurvePrediction> = predictions
        .iter()
        .filter(|prediction| !selected_ids.contains(prediction.architecture_id.as_str()))
        .collect();
    guard_order.sort_by(|left, right| {
        let order = match calibrator.config.metric_direction {
            ProxyDirection::Maximize => right.optimistic_bound.total_cmp(&left.optimistic_bound),
            ProxyDirection::Minimize => left.optimistic_bound.total_cmp(&right.optimistic_bound),
        };
        order.then_with(|| left.architecture_id.cmp(&right.architecture_id))
    });
    for prediction in guard_order.into_iter().take(guard_slot_count) {
        selected_ids.insert(prediction.architecture_id.as_str());
        selected.push(selected_candidate(prediction, PromotionReason::OptimisticGuard));
    }

    if selected.len() != promotion_count {
        return Err(LearningCurvePromotionError::new(
            "selection",
            "could not produce the requested number of unique promotions",
        ));
    }

    let mut all_predictions = predictions.to_vec();
    all_predictions.sort_by(|left, right| left.architecture_id.cmp(&right.architecture_id));
    let selection_sha256 = digest_selection(
        &calibrator.calibration_sha256,
        fidelity_index,
        promotion_count,
        guard_slot_count,
        &selected,
        &all_predictions,
    );
    Ok(CurvePromotionSelection {
        calibration_sha256: calibrator.calibration_sha256.clone(),
        fidelity_index,
        requested_promotion_count: promotion_count,
        guard_slot_count,
        selected,
        all_predictions,
        selection_sha256,
    })
}

fn selected_candidate(
    prediction: &CurvePrediction,
    reason: PromotionReason,
) -> SelectedCurveCandidate {
    SelectedCurveCandidate {
        architecture_id: prediction.architecture_id.clone(),
        reason,
        predicted_final_value: prediction.predicted_final_value,
        lower_bound: prediction.lower_bound,
        upper_bound: prediction.upper_bound,
        interval_radius: prediction.interval_radius,
    }
}

fn validate_config(
    config: &LearningCurveCalibrationConfig,
) -> Result<(), LearningCurvePromotionError> {
    for (field, value) in [
        ("benchmark_id", config.benchmark_id.as_str()),
        ("target_metric_id", config.target_metric_id.as_str()),
        ("model_fit_split_id", config.model_fit_split_id.as_str()),
        (
            "residual_calibration_split_id",
            config.residual_calibration_split_id.as_str(),
        ),
        ("evaluation_split_id", config.evaluation_split_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(LearningCurvePromotionError::new(field, "must be non-empty"));
        }
    }
    let splits = [
        config.model_fit_split_id.as_str(),
        config.residual_calibration_split_id.as_str(),
        config.evaluation_split_id.as_str(),
    ];
    if splits.iter().collect::<BTreeSet<_>>().len() != splits.len() {
        return Err(LearningCurvePromotionError::new(
            "splits",
            "model-fit, residual-calibration and evaluation splits must be pairwise disjoint",
        ));
    }
    if config.source_artifact_sha256.len() != 64
        || !config
            .source_artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(LearningCurvePromotionError::new(
            "source_artifact_sha256",
            "must be exactly 64 hexadecimal characters",
        ));
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 || config.alpha >= 0.5 {
        return Err(LearningCurvePromotionError::new(
            "alpha",
            "must be finite and in (0, 0.5)",
        ));
    }
    if config.fidelity_levels.len() < 2 {
        return Err(LearningCurvePromotionError::new(
            "fidelity_levels",
            "at least two fidelity levels are required",
        ));
    }
    if config.fidelity_levels.iter().any(|level| level.id.trim().is_empty()) {
        return Err(LearningCurvePromotionError::new(
            "fidelity_levels",
            "every level ID must be non-empty",
        ));
    }
    let ids: BTreeSet<&str> = config
        .fidelity_levels
        .iter()
        .map(|level| level.id.as_str())
        .collect();
    if ids.len() != config.fidelity_levels.len() {
        return Err(LearningCurvePromotionError::new(
            "fidelity_levels",
            "fidelity IDs must be unique",
        ));
    }
    if config
        .fidelity_levels
        .windows(2)
        .any(|pair| pair[1].cumulative_cost_units <= pair[0].cumulative_cost_units)
    {
        return Err(LearningCurvePromotionError::new(
            "fidelity_levels",
            "cumulative resource costs must strictly increase",
        ));
    }
    let maximum_required = if config.fidelity_levels.len() == 2 { 2 } else { 3 };
    if config.minimum_model_fit_samples < maximum_required
        || config.minimum_residual_calibration_samples == 0
    {
        return Err(LearningCurvePromotionError::new(
            "minimum_samples",
            "minimum fit count must cover the largest model and calibration count must be positive",
        ));
    }
    Ok(())
}

fn validate_samples(
    samples: &[LearningCurveCalibrationSample],
    config: &LearningCurveCalibrationConfig,
    minimum: usize,
    field: &str,
) -> Result<(), LearningCurvePromotionError> {
    if samples.len() < minimum {
        return Err(LearningCurvePromotionError::new(
            field,
            format!("requires at least {minimum} architecture samples"),
        ));
    }
    let mut ids = HashSet::with_capacity(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        if sample.architecture_id.trim().is_empty()
            || !ids.insert(sample.architecture_id.as_str())
        {
            return Err(LearningCurvePromotionError::new(
                format!("{field}[{index}].architecture_id"),
                "architecture IDs must be non-empty and unique within a split",
            ));
        }
        if sample.curve_values.len() != config.fidelity_levels.len() {
            return Err(LearningCurvePromotionError::new(
                format!("{field}[{index}].curve_values"),
                "curve must have one finite observation per fidelity level",
            ));
        }
        if sample.curve_values.iter().any(|value| !value.is_finite())
            || !sample.final_value.is_finite()
        {
            return Err(LearningCurvePromotionError::new(
                format!("{field}[{index}]"),
                "curve and final value must be finite",
            ));
        }
        let curve_final = sample.curve_values[sample.curve_values.len() - 1];
        if (curve_final - sample.final_value).abs() > 1e-8 {
            return Err(LearningCurvePromotionError::new(
                format!("{field}[{index}].final_value"),
                "final_value must match the final fidelity curve point",
            ));
        }
    }
    Ok(())
}

fn curve_features(
    curve: &[f64],
    fidelity_index: usize,
    fidelity_levels: &[CurveFidelityLevel],
) -> Vec<f64> {
    let current = curve[fidelity_index];
    if fidelity_index == 0 {
        vec![1.0, current]
    } else {
        let previous = curve[fidelity_index - 1];
        let cost_delta = (fidelity_levels[fidelity_index].cumulative_cost_units
            - fidelity_levels[fidelity_index - 1].cumulative_cost_units) as f64;
        let slope = (current - previous) / cost_delta;
        vec![1.0, current, slope]
    }
}

fn fit_linear_model(
    rows: &[Vec<f64>],
    targets: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), LearningCurvePromotionError> {
    if rows.is_empty() || rows.len() != targets.len() {
        return Err(LearningCurvePromotionError::new(
            "model_fit_samples",
            "regression rows and targets must be non-empty and aligned",
        ));
    }
    let parameter_count = rows[0].len();
    if parameter_count < 2
        || rows.iter().any(|row| row.len() != parameter_count || row.iter().any(|x| !x.is_finite()))
        || targets.iter().any(|target| !target.is_finite())
    {
        return Err(LearningCurvePromotionError::new(
            "model_fit_samples",
            "regression matrix is malformed or non-finite",
        ));
    }
    let mut gram = vec![vec![0.0; parameter_count]; parameter_count];
    let mut rhs = vec![0.0; parameter_count];
    for (row, target) in rows.iter().zip(targets) {
        for left in 0..parameter_count {
            rhs[left] += row[left] * target;
            for right in 0..parameter_count {
                gram[left][right] += row[left] * row[right];
            }
        }
    }
    let inverse_gram = invert_matrix(&gram)?;
    let coefficients = inverse_gram
        .iter()
        .map(|row| row.iter().zip(&rhs).map(|(left, right)| left * right).sum())
        .collect::<Vec<f64>>();
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(LearningCurvePromotionError::new(
            "model_fit_samples",
            "regression coefficients are non-finite",
        ));
    }
    Ok((coefficients, inverse_gram))
}

fn invert_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, LearningCurvePromotionError> {
    let size = matrix.len();
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
        return Err(LearningCurvePromotionError::new(
            "model_fit_samples",
            "normal-equation matrix must be square and non-empty",
        ));
    }
    let mut augmented = vec![vec![0.0; size * 2]; size];
    for row in 0..size {
        for column in 0..size {
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][size + row] = 1.0;
    }
    for pivot in 0..size {
        let best_row = (pivot..size)
            .max_by(|left, right| {
                augmented[*left][pivot]
                    .abs()
                    .total_cmp(&augmented[*right][pivot].abs())
            })
            .unwrap_or(pivot);
        if augmented[best_row][pivot].abs() <= 1e-12 {
            return Err(LearningCurvePromotionError::new(
                "model_fit_samples",
                "learning-curve regression design is rank-deficient",
            ));
        }
        augmented.swap(pivot, best_row);
        let scale = augmented[pivot][pivot];
        for column in 0..(size * 2) {
            augmented[pivot][column] /= scale;
        }
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            for column in 0..(size * 2) {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }
    let inverse: Vec<Vec<f64>> = augmented
        .iter()
        .map(|row| row[size..].to_vec())
        .collect();
    if inverse.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LearningCurvePromotionError::new(
            "model_fit_samples",
            "inverse normal-equation matrix is non-finite",
        ));
    }
    Ok(inverse)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn quadratic_form(
    features: &[f64],
    matrix: &[Vec<f64>],
) -> Result<f64, LearningCurvePromotionError> {
    if matrix.len() != features.len() || matrix.iter().any(|row| row.len() != features.len()) {
        return Err(LearningCurvePromotionError::new(
            "prediction",
            "feature vector and leverage matrix dimension mismatch",
        ));
    }
    let transformed: Vec<f64> = matrix.iter().map(|row| dot(row, features)).collect();
    let value = dot(features, &transformed);
    if !value.is_finite() {
        return Err(LearningCurvePromotionError::new(
            "prediction",
            "leverage is non-finite",
        ));
    }
    Ok(value)
}

fn digest_samples(
    role: &str,
    samples: &[LearningCurveCalibrationSample],
    fidelity_levels: &[CurveFidelityLevel],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CURVE_DATA_DOMAIN);
    update_string(&mut hasher, role);
    let mut ordered: Vec<&LearningCurveCalibrationSample> = samples.iter().collect();
    ordered.sort_by(|left, right| left.architecture_id.cmp(&right.architecture_id));
    hasher.update((ordered.len() as u64).to_be_bytes());
    for sample in ordered {
        update_string(&mut hasher, &sample.architecture_id);
        hasher.update((sample.curve_values.len() as u64).to_be_bytes());
        for (index, value) in sample.curve_values.iter().enumerate() {
            update_string(&mut hasher, &fidelity_levels[index].id);
            hasher.update(value.to_bits().to_be_bytes());
        }
        hasher.update(sample.final_value.to_bits().to_be_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_calibrator(
    config: &LearningCurveCalibrationConfig,
    fit_data_sha256: &str,
    calibration_data_sha256: &str,
    models: &[CurveRungModel],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CALIBRATION_DOMAIN);
    update_string(&mut hasher, &config.benchmark_id);
    update_string(&mut hasher, &config.target_metric_id);
    hasher.update([direction_byte(config.metric_direction)]);
    update_string(&mut hasher, &config.source_artifact_sha256);
    update_string(&mut hasher, &config.model_fit_split_id);
    update_string(&mut hasher, &config.residual_calibration_split_id);
    update_string(&mut hasher, &config.evaluation_split_id);
    hasher.update(config.alpha.to_bits().to_be_bytes());
    hasher.update((config.minimum_model_fit_samples as u64).to_be_bytes());
    hasher.update((config.minimum_residual_calibration_samples as u64).to_be_bytes());
    hasher.update((config.fidelity_levels.len() as u64).to_be_bytes());
    for level in &config.fidelity_levels {
        update_string(&mut hasher, &level.id);
        hasher.update(level.cumulative_cost_units.to_be_bytes());
    }
    update_string(&mut hasher, fit_data_sha256);
    update_string(&mut hasher, calibration_data_sha256);
    hasher.update((models.len() as u64).to_be_bytes());
    for model in models {
        hasher.update((model.fidelity_index as u64).to_be_bytes());
        hasher.update((model.coefficients.len() as u64).to_be_bytes());
        for coefficient in &model.coefficients {
            hasher.update(coefficient.to_bits().to_be_bytes());
        }
        for row in &model.inverse_gram {
            hasher.update((row.len() as u64).to_be_bytes());
            for value in row {
                hasher.update(value.to_bits().to_be_bytes());
            }
        }
        hasher.update(model.normalized_residual_quantile.to_bits().to_be_bytes());
        hasher.update((model.fit_sample_count as u64).to_be_bytes());
        hasher.update((model.residual_calibration_count as u64).to_be_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_selection(
    calibration_sha256: &str,
    fidelity_index: usize,
    promotion_count: usize,
    guard_slot_count: usize,
    selected: &[SelectedCurveCandidate],
    predictions: &[CurvePrediction],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SELECTION_DOMAIN);
    update_string(&mut hasher, calibration_sha256);
    hasher.update((fidelity_index as u64).to_be_bytes());
    hasher.update((promotion_count as u64).to_be_bytes());
    hasher.update((guard_slot_count as u64).to_be_bytes());
    hasher.update((selected.len() as u64).to_be_bytes());
    for candidate in selected {
        update_string(&mut hasher, &candidate.architecture_id);
        hasher.update([match candidate.reason {
            PromotionReason::ConservativeBound => 0,
            PromotionReason::OptimisticGuard => 1,
        }]);
        hasher.update(candidate.predicted_final_value.to_bits().to_be_bytes());
        hasher.update(candidate.lower_bound.to_bits().to_be_bytes());
        hasher.update(candidate.upper_bound.to_bits().to_be_bytes());
        hasher.update(candidate.interval_radius.to_bits().to_be_bytes());
    }
    hasher.update((predictions.len() as u64).to_be_bytes());
    for prediction in predictions {
        update_string(&mut hasher, &prediction.architecture_id);
        hasher.update((prediction.fidelity_index as u64).to_be_bytes());
        hasher.update((prediction.observed_curve.len() as u64).to_be_bytes());
        for value in &prediction.observed_curve {
            hasher.update(value.to_bits().to_be_bytes());
        }
        hasher.update(prediction.predicted_final_value.to_bits().to_be_bytes());
        hasher.update(prediction.lower_bound.to_bits().to_be_bytes());
        hasher.update(prediction.upper_bound.to_bits().to_be_bytes());
        hasher.update(prediction.interval_radius.to_bits().to_be_bytes());
        hasher.update(prediction.leverage.to_bits().to_be_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

fn direction_byte(direction: ProxyDirection) -> u8 {
    match direction {
        ProxyDirection::Maximize => 0,
        ProxyDirection::Minimize => 1,
    }
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().copied() {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> LearningCurveCalibrationConfig {
        LearningCurveCalibrationConfig {
            benchmark_id: "synthetic-learning-curves".into(),
            target_metric_id: "validation_accuracy".into(),
            metric_direction: ProxyDirection::Maximize,
            source_artifact_sha256: "a".repeat(64),
            model_fit_split_id: "development-fit".into(),
            residual_calibration_split_id: "development-calibration".into(),
            evaluation_split_id: "held-out-evaluation".into(),
            fidelity_levels: vec![
                CurveFidelityLevel {
                    id: "epoch_5".into(),
                    cumulative_cost_units: 5,
                },
                CurveFidelityLevel {
                    id: "epoch_15".into(),
                    cumulative_cost_units: 15,
                },
                CurveFidelityLevel {
                    id: "epoch_200".into(),
                    cumulative_cost_units: 200,
                },
            ],
            alpha: 0.1,
            minimum_model_fit_samples: 20,
            minimum_residual_calibration_samples: 20,
        }
    }

    fn samples(prefix: &str, count: usize, noise_scale: f64) -> Vec<LearningCurveCalibrationSample> {
        (0..count)
            .map(|index| {
                let initial = 10.0 + ((index * 7) % 19) as f64;
                let slope = 0.2 + ((index * 11) % 13) as f64 * 0.1;
                let middle = initial + slope * 10.0;
                let noise = (((index * 3) % 5) as f64 - 2.0) * noise_scale;
                let final_value = 14.0 + 2.0 * middle + 4.0 * slope + noise;
                LearningCurveCalibrationSample {
                    architecture_id: format!("{prefix}-{index:03}"),
                    curve_values: vec![initial, middle, final_value],
                    final_value,
                }
            })
            .collect()
    }

    fn fitted() -> LearningCurveCalibrator {
        LearningCurveCalibrator::fit(config(), &samples("fit", 60, 0.0), &samples("cal", 60, 0.3))
            .unwrap()
    }

    #[test]
    fn calibration_uses_disjoint_splits_and_builds_every_intermediate_rung_model() {
        let predictor = fitted();
        assert_eq!(predictor.rung_models.len(), 2);
        assert_eq!(predictor.rung_models[0].coefficients.len(), 2);
        assert_eq!(predictor.rung_models[1].coefficients.len(), 3);
        assert_eq!(predictor.rung_models[1].fit_sample_count, 60);
        assert_eq!(predictor.rung_models[1].residual_calibration_count, 60);
        assert_eq!(predictor.calibration_sha256.len(), 64);
    }

    #[test]
    fn curve_slope_can_reverse_raw_early_metric_order() {
        let predictor = fitted();
        let slow = predictor.predict_final("slow", &[30.0, 55.0]).unwrap();
        let fast = predictor.predict_final("fast", &[56.0, 57.0]).unwrap();
        assert!(fast.observed_curve[1] > slow.observed_curve[1]);
        assert!(slow.predicted_final_value > fast.predicted_final_value);
    }

    #[test]
    fn prediction_interval_is_candidate_specific_and_direction_aware() {
        let predictor = fitted();
        let in_range = predictor.predict_final("in-range", &[20.0, 25.0]).unwrap();
        let extrapolated = predictor.predict_final("extrapolated", &[100.0, 180.0]).unwrap();
        assert!(extrapolated.interval_radius > in_range.interval_radius);

        let mut minimizing_config = config();
        minimizing_config.metric_direction = ProxyDirection::Minimize;
        let minimizing = LearningCurveCalibrator::fit(
            minimizing_config,
            &samples("fit", 60, 0.0),
            &samples("cal", 60, 0.3),
        )
        .unwrap();
        let prediction = minimizing.predict_final("latency", &[20.0, 25.0]).unwrap();
        assert_eq!(prediction.conservative_bound, prediction.upper_bound);
        assert_eq!(prediction.optimistic_bound, prediction.lower_bound);
    }

    #[test]
    fn guard_slot_preserves_uncertain_candidate_that_lcb_would_drop() {
        let predictor = fitted();
        let predictions = vec![
            predictor.predict_final("stable-a", &[20.0, 25.0]).unwrap(),
            predictor.predict_final("stable-b", &[21.0, 26.0]).unwrap(),
            predictor.predict_final("uncertain-c", &[45.0, 30.0]).unwrap(),
        ];
        let selection =
            select_curve_promotions(&predictor, &predictions, 2, 1).unwrap();
        assert_eq!(selection.selected.len(), 2);
        let guard = selection
            .selected
            .iter()
            .find(|candidate| candidate.reason == PromotionReason::OptimisticGuard)
            .unwrap();
        let conservative_top = predictions
            .iter()
            .max_by(|left, right| left.conservative_bound.total_cmp(&right.conservative_bound))
            .unwrap();
        assert_eq!(guard.architecture_id, "uncertain-c");
        assert_ne!(guard.architecture_id, conservative_top.architecture_id);
        assert!(guard.interval_radius > predictions[1].interval_radius);
        assert_eq!(selection.selection_sha256.len(), 64);
    }

    #[test]
    fn conformal_quantile_uses_finite_sample_order_statistic() {
        let predictor = fitted();
        for model in &predictor.rung_models {
            assert!(model.normalized_residual_quantile.is_finite());
            assert!(model.normalized_residual_quantile >= 0.0);
        }
    }

    #[test]
    fn fit_rejects_overlapping_architecture_ids_between_fit_and_calibration() {
        let fit = samples("fit", 60, 0.0);
        let mut calibration = samples("cal", 60, 0.3);
        calibration[0].architecture_id = fit[0].architecture_id.clone();
        assert!(LearningCurveCalibrator::fit(config(), &fit, &calibration).is_err());
    }

    #[test]
    fn fit_rejects_split_overlap_invalid_curves_and_rank_deficient_designs() {
        let mut invalid_config = config();
        invalid_config.evaluation_split_id = invalid_config.residual_calibration_split_id.clone();
        assert!(LearningCurveCalibrator::fit(
            invalid_config,
            &samples("fit", 60, 0.0),
            &samples("cal", 60, 0.3),
        )
        .is_err());

        let mut invalid_samples = samples("fit", 60, 0.0);
        invalid_samples[0].curve_values[1] = f64::NAN;
        assert!(LearningCurveCalibrator::fit(
            config(),
            &invalid_samples,
            &samples("cal", 60, 0.3),
        )
        .is_err());

        let mut rank_deficient = samples("flat-fit", 60, 0.0);
        for sample in &mut rank_deficient {
            sample.curve_values[0] = 1.0;
            sample.curve_values[1] = 1.0;
            sample.curve_values[2] = sample.final_value;
        }
        assert!(LearningCurveCalibrator::fit(
            config(),
            &rank_deficient,
            &samples("cal", 60, 0.3),
        )
        .is_err());
    }

    #[test]
    fn malformed_prediction_sets_duplicate_ids_and_bad_slots_are_rejected() {
        let predictor = fitted();
        let prediction = predictor.predict_final("a", &[20.0, 25.0]).unwrap();
        assert!(select_curve_promotions(&predictor, &[prediction.clone(), prediction.clone()], 1, 0).is_err());
        assert!(select_curve_promotions(&predictor, &[prediction.clone()], 1, 2).is_err());
        let mut wrong_rung = prediction.clone();
        wrong_rung.fidelity_index = 9;
        assert!(select_curve_promotions(&predictor, &[wrong_rung], 1, 0).is_err());
    }

    #[test]
    fn calibrator_integrity_rejects_modified_coefficients() {
        let mut predictor = fitted();
        predictor.rung_models[0].coefficients[0] += 1.0;
        assert!(predictor.verify_integrity().is_err());
    }

    #[test]
    fn promotion_selector_rejects_tampered_prediction_bounds() {
        let predictor = fitted();
        let mut prediction = predictor.predict_final("candidate", &[20.0, 25.0]).unwrap();
        prediction.lower_bound -= 1.0;
        assert!(select_curve_promotions(&predictor, &[prediction], 1, 0).is_err());
    }

    #[test]
    fn final_fidelity_prefix_is_not_predicted_and_digest_is_seed_stable() {
        let first = fitted();
        let second = fitted();
        assert_eq!(first.calibration_sha256, second.calibration_sha256);
        assert!(first.predict_final("final", &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn source_split_and_artifact_provenance_change_calibration_digest() {
        let first = fitted();
        let mut altered = config();
        altered.source_artifact_sha256 = "c".repeat(64);
        let second = LearningCurveCalibrator::fit(
            altered,
            &samples("fit", 60, 0.0),
            &samples("cal", 60, 0.3),
        )
        .unwrap();
        assert_ne!(first.calibration_sha256, second.calibration_sha256);
    }
}
