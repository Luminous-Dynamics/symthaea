//! Task-relevant out-of-model-scope monitoring from one-step prediction residuals.
//!
//! Unlike feature-space OOD detection, this monitor scores disagreement between a
//! predicted next state and the state actually observed after an action. Residuals
//! are normalized by nominal scales and by the available safety margin, so an
//! error consuming most of the remaining reserve/load margin receives more weight.
//! This is a post-action monitor: it can switch the *next* decision to a fallback,
//! but cannot retroactively prevent the transition whose result exposed the error.

use super::drift_monitor::{assess_drift, calibrate_drift_threshold, DriftAssessment, DriftThreshold};
use super::robust_viability::ViabilityLimits;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionErrorScales {
    reserve_scale: f64,
    load_scale: f64,
    min_margin_scale: f64,
}

impl PredictionErrorScales {
    pub fn new(
        reserve_scale: f64,
        load_scale: f64,
        min_margin_scale: f64,
    ) -> Option<Self> {
        if !reserve_scale.is_finite()
            || !load_scale.is_finite()
            || !min_margin_scale.is_finite()
            || reserve_scale <= 0.0
            || load_scale <= 0.0
            || min_margin_scale <= 0.0
        {
            return None;
        }
        Some(Self {
            reserve_scale,
            load_scale,
            min_margin_scale,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionErrorScore {
    score: f64,
    normalized_absolute_error: f64,
    normalized_reserve_margin_loss: f64,
    normalized_load_margin_loss: f64,
    predicted_safe: bool,
    observed_safe: bool,
    false_safe_prediction: bool,
}

impl PredictionErrorScore {
    pub const fn score(self) -> f64 {
        self.score
    }

    pub const fn normalized_absolute_error(self) -> f64 {
        self.normalized_absolute_error
    }

    pub const fn normalized_reserve_margin_loss(self) -> f64 {
        self.normalized_reserve_margin_loss
    }

    pub const fn normalized_load_margin_loss(self) -> f64 {
        self.normalized_load_margin_loss
    }

    pub const fn predicted_safe(self) -> bool {
        self.predicted_safe
    }

    pub const fn observed_safe(self) -> bool {
        self.observed_safe
    }

    /// The predictor claimed the next state met the configured limits, but the
    /// observed state violated at least one of them.
    pub const fn false_safe_prediction(self) -> bool {
        self.false_safe_prediction
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScopeThreshold {
    inner: DriftThreshold,
}

impl ScopeThreshold {
    pub const fn score_threshold(self) -> f64 {
        self.inner.score_threshold()
    }

    pub const fn target_false_alarm_rate(self) -> f64 {
        self.inner.target_false_alarm_rate()
    }

    pub const fn calibration_episodes(self) -> usize {
        self.inner.calibration_episodes()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScopeAssessment {
    prediction_error: PredictionErrorScore,
    statistical_assessment: DriftAssessment,
    alarm: bool,
    false_safe_override: bool,
}

impl ScopeAssessment {
    pub const fn prediction_error(self) -> PredictionErrorScore {
        self.prediction_error
    }

    pub const fn statistical_assessment(self) -> DriftAssessment {
        self.statistical_assessment
    }

    pub const fn alarm(self) -> bool {
        self.alarm
    }

    /// True when the measured prediction said “safe” but the observation was
    /// already unsafe. This hard override is a detection signal, not prevention.
    pub const fn false_safe_override(self) -> bool {
        self.false_safe_override
    }
}

/// Score a predicted/observed state pair using absolute model error and the
/// *direction* in which the error consumes safety margin. Larger values indicate
/// a prediction that is more wrong relative to the state-specific available
/// reserve/load margin. State values and limits use [0, 1] units.
pub fn score_prediction_error(
    predicted_reserve: f64,
    predicted_load: f64,
    observed_reserve: f64,
    observed_load: f64,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
) -> Option<PredictionErrorScore> {
    let values = [
        predicted_reserve,
        predicted_load,
        observed_reserve,
        observed_load,
        limits.min_reserve(),
        limits.max_load(),
        scales.reserve_scale,
        scales.load_scale,
        scales.min_margin_scale,
    ];
    if values.iter().any(|v| !v.is_finite())
        || [predicted_reserve, predicted_load, observed_reserve, observed_load]
            .iter()
            .any(|v| !(0.0..=1.0).contains(v))
        || scales.reserve_scale <= 0.0
        || scales.load_scale <= 0.0
        || scales.min_margin_scale <= 0.0
    {
        return None;
    }

    let normalized_reserve_error =
        (predicted_reserve - observed_reserve).abs() / scales.reserve_scale;
    let normalized_load_error =
        (predicted_load - observed_load).abs() / scales.load_scale;
    let normalized_absolute_error = normalized_reserve_error.max(normalized_load_error);

    // Only errors that move reserve/load in the unsafe direction consume a
    // margin. Clamp the denominator away from zero near an active constraint.
    let predicted_reserve_margin =
        (predicted_reserve - limits.min_reserve()).max(scales.min_margin_scale);
    let predicted_load_margin = (limits.max_load() - predicted_load).max(scales.min_margin_scale);
    let normalized_reserve_margin_loss =
        (predicted_reserve - observed_reserve).max(0.0) / predicted_reserve_margin;
    let normalized_load_margin_loss =
        (observed_load - predicted_load).max(0.0) / predicted_load_margin;

    let predicted_safe =
        predicted_reserve >= limits.min_reserve() && predicted_load <= limits.max_load();
    let observed_safe =
        observed_reserve >= limits.min_reserve() && observed_load <= limits.max_load();
    let false_safe_prediction = predicted_safe && !observed_safe;
    let score = normalized_absolute_error
        .max(normalized_reserve_margin_loss)
        .max(normalized_load_margin_loss);

    Some(PredictionErrorScore {
        score,
        normalized_absolute_error,
        normalized_reserve_margin_loss,
        normalized_load_margin_loss,
        predicted_safe,
        observed_safe,
        false_safe_prediction,
    })
}

/// Calibrate one episode-maximum prediction-risk score per nominal calibration
/// episode. The finite order statistic inherits the exchangeability/frozen-model
/// assumptions of the underlying split-conformal threshold.
pub fn calibrate_scope_threshold(
    nominal_episode_max_scores: &[f64],
    target_false_alarm_rate: f64,
) -> Option<ScopeThreshold> {
    calibrate_drift_threshold(nominal_episode_max_scores, target_false_alarm_rate)
        .map(|inner| ScopeThreshold { inner })
}

/// Assess whether the predictive model is out of task scope. The calibrated score
/// is the statistical gate; an observed false-safe prediction triggers an
/// unconditional alarm even if a score tie or threshold choice would not. Since
/// observation arrives after the transition, this protects subsequent control
/// decisions and must not be described as prevention of the already observed
/// unsafe transition.
pub fn assess_prediction_scope(
    predicted_reserve: f64,
    predicted_load: f64,
    observed_reserve: f64,
    observed_load: f64,
    limits: ViabilityLimits,
    scales: PredictionErrorScales,
    threshold: ScopeThreshold,
) -> Option<ScopeAssessment> {
    let prediction_error = score_prediction_error(
        predicted_reserve,
        predicted_load,
        observed_reserve,
        observed_load,
        limits,
        scales,
    )?;
    let statistical_assessment = assess_drift(prediction_error.score(), threshold.inner)?;
    let false_safe_override = prediction_error.false_safe_prediction();
    Some(ScopeAssessment {
        prediction_error,
        statistical_assessment,
        alarm: statistical_assessment.drift_detected() || false_safe_override,
        false_safe_override,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> ViabilityLimits {
        ViabilityLimits::new(0.15, 0.88).expect("valid viability limits")
    }

    fn scales() -> PredictionErrorScales {
        PredictionErrorScales::new(0.02, 0.02, 0.02).expect("valid residual scales")
    }

    fn nominal_scores() -> Vec<f64> {
        (0..159)
            .map(|i| (i + 1) as f64 / 160.0 * 0.20)
            .collect()
    }

    #[test]
    fn rejects_invalid_state_values_and_scales() {
        assert!(PredictionErrorScales::new(0.0, 0.02, 0.02).is_none());
        assert!(PredictionErrorScales::new(0.02, f64::NAN, 0.02).is_none());
        assert!(score_prediction_error(
            1.01, 0.5, 0.99, 0.5, limits(), scales()
        )
        .is_none());
        assert!(score_prediction_error(
            0.5, 0.5, f64::NAN, 0.5, limits(), scales()
        )
        .is_none());
    }

    #[test]
    fn margin_consuming_errors_score_higher_than_equally_sized_safe_direction_errors() {
        let margin_sensitive_scales =
            PredictionErrorScales::new(0.05, 0.05, 0.02).expect("valid scales");
        let harmful = score_prediction_error(
            0.17,
            0.86,
            0.149,
            0.881,
            limits(),
            margin_sensitive_scales,
        )
        .expect("finite harmful residual");
        let benign = score_prediction_error(
            0.40,
            0.50,
            0.38,
            0.52,
            limits(),
            margin_sensitive_scales,
        )
        .expect("finite benign residual");
        assert!(harmful.score() > benign.score());
        assert!(harmful.false_safe_prediction());
        assert!(!benign.false_safe_prediction());
    }

    #[test]
    fn calibrates_from_episode_maxima_and_fails_closed_on_invalid_inputs() {
        assert!(calibrate_scope_threshold(&[], 0.05).is_none());
        assert!(calibrate_scope_threshold(&[0.1, f64::NAN], 0.05).is_none());
        assert!(calibrate_scope_threshold(&nominal_scores(), 0.0).is_none());
        let threshold =
            calibrate_scope_threshold(&nominal_scores(), 0.05).expect("valid threshold");
        assert_eq!(threshold.calibration_episodes(), 159);
        assert_eq!(threshold.target_false_alarm_rate(), 0.05);
    }

    #[test]
    fn false_safe_prediction_overrides_a_non_alarm_statistical_score() {
        let high_scores = vec![10.0; 159];
        let threshold =
            calibrate_scope_threshold(&high_scores, 0.05).expect("high threshold");
        let result = assess_prediction_scope(
            0.151,
            0.879,
            0.149,
            0.881,
            limits(),
            scales(),
            threshold,
        )
        .expect("valid prediction pair");
        assert!(result.false_safe_override());
        assert!(result.alarm());
        assert!(!result.statistical_assessment().drift_detected());
    }

    #[test]
    fn large_stealth_dynamics_residual_alarms_without_feature_shift() {
        let threshold =
            calibrate_scope_threshold(&nominal_scores(), 0.05).expect("valid threshold");
        let nominal = score_prediction_error(0.82, 0.84, 0.819, 0.841, limits(), scales())
            .expect("nominal residual");
        let stealth = score_prediction_error(0.82, 0.84, 0.765, 0.94, limits(), scales())
            .expect("stealth-shift residual");
        assert!(nominal.score() <= threshold.score_threshold());
        assert!(stealth.score() > threshold.score_threshold());
    }
}
