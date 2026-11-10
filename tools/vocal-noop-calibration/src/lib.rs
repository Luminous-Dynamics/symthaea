//! Distribution-free no-op tolerance calibration for vocal-control evaluation.
//!
//! Uses the sample maximum as a one-sided nonparametric upper tolerance bound.
//! The caller remains responsible for independence and representativeness.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

const MAX_SUPPORTED_SAMPLE_COUNT: usize = 10_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationError {
    pub message: String,
}

impl CalibrationError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl Display for CalibrationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for CalibrationError {}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricCalibrationSpec {
    pub metric_id: String,
    pub unit: String,
    pub safety_multiplier: f64,
    pub minimum_tolerance: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NoOpObservation {
    /// Independent calibration unit, not an overlapping frame/window.
    pub calibration_unit_id: String,
    pub metric_id: String,
    pub unit: String,
    /// Absolute difference between matched no-op repeats.
    pub absolute_delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricTolerance {
    pub metric_id: String,
    pub unit: String,
    pub sample_count: usize,
    pub required_sample_count: usize,
    pub maximum_observed_absolute_delta: f64,
    pub safety_multiplier: f64,
    pub minimum_tolerance: f64,
    pub calibrated_tolerance: f64,
    pub per_metric_confidence_lower_bound: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationReport {
    pub schema: String,
    pub population_coverage: f64,
    pub requested_familywise_confidence: f64,
    pub metric_count: usize,
    pub per_metric_confidence_target: f64,
    pub familywise_confidence_lower_bound: f64,
    pub calibration_unit_ids: Vec<String>,
    pub metrics: Vec<MetricTolerance>,
}

/// Calibrate all metric tolerances as one family.
///
/// For each metric, the sample maximum is a one-sided, distribution-free upper
/// tolerance bound. With independent, representative observations, its
/// confidence of covering at least population_coverage of that metric's
/// no-op distribution is 1 - population_coverage^n.
///
/// Bonferroni adjustment allocates family-wise failure probability across
/// metrics without assuming independence between different metrics. Each
/// metric must be observed over the same set of unique calibration units.
pub fn calibrate_noop_family(
    specs: &[MetricCalibrationSpec],
    observations: &[NoOpObservation],
    population_coverage: f64,
    familywise_confidence: f64,
) -> Result<CalibrationReport, CalibrationError> {
    validate_probability(population_coverage, "population_coverage")?;
    validate_probability(familywise_confidence, "familywise_confidence")?;
    if specs.is_empty() {
        return Err(CalibrationError::new("at least one metric specification is required"));
    }
    if observations.is_empty() {
        return Err(CalibrationError::new("no no-op observations supplied"));
    }

    let metric_count = specs.len();
    let family_alpha = 1.0 - familywise_confidence;
    let per_metric_alpha = family_alpha / metric_count as f64;
    let per_metric_confidence_target = 1.0 - per_metric_alpha;
    let required_sample_count =
        minimum_sample_count(population_coverage, per_metric_confidence_target)?;

    let mut specs_by_id = BTreeMap::new();
    for spec in specs {
        require_non_empty(&spec.metric_id, "metric_id")?;
        require_non_empty(&spec.unit, "metric unit")?;
        if !spec.safety_multiplier.is_finite() || spec.safety_multiplier < 1.0 {
            return Err(CalibrationError::new(format!(
                "safety multiplier for {} must be finite and >= 1.0",
                spec.metric_id
            )));
        }
        if !spec.minimum_tolerance.is_finite() || spec.minimum_tolerance < 0.0 {
            return Err(CalibrationError::new(format!(
                "minimum tolerance for {} must be finite and non-negative",
                spec.metric_id
            )));
        }
        if specs_by_id.insert(spec.metric_id.as_str(), spec).is_some() {
            return Err(CalibrationError::new(format!(
                "duplicate metric specification: {}",
                spec.metric_id
            )));
        }
    }

    let mut values_by_metric: BTreeMap<String, Vec<f64>> = specs
        .iter()
        .map(|spec| (spec.metric_id.clone(), Vec::new()))
        .collect();
    let mut unit_ids_by_metric: BTreeMap<String, BTreeSet<String>> = specs
        .iter()
        .map(|spec| (spec.metric_id.clone(), BTreeSet::new()))
        .collect();

    for observation in observations {
        require_non_empty(&observation.calibration_unit_id, "calibration_unit_id")?;
        require_non_empty(&observation.metric_id, "observation metric_id")?;
        require_non_empty(&observation.unit, "observation unit")?;
        if !observation.absolute_delta.is_finite() || observation.absolute_delta < 0.0 {
            return Err(CalibrationError::new(format!(
                "absolute no-op delta for {} must be finite and non-negative",
                observation.metric_id
            )));
        }

        let spec = specs_by_id.get(observation.metric_id.as_str()).ok_or_else(|| {
            CalibrationError::new(format!(
                "observation has no matching metric specification: {}",
                observation.metric_id
            ))
        })?;
        if spec.unit != observation.unit {
            return Err(CalibrationError::new(format!(
                "unit mismatch for {}: expected {}, got {}",
                observation.metric_id, spec.unit, observation.unit
            )));
        }

        let unit_ids = unit_ids_by_metric
            .get_mut(&observation.metric_id)
            .expect("metric was checked against specifications");
        if !unit_ids.insert(observation.calibration_unit_id.clone()) {
            return Err(CalibrationError::new(format!(
                "duplicate calibration unit {} for metric {}",
                observation.calibration_unit_id, observation.metric_id
            )));
        }
        values_by_metric
            .get_mut(&observation.metric_id)
            .expect("metric was checked against specifications")
            .push(observation.absolute_delta);
    }

    let first_ids = unit_ids_by_metric
        .values()
        .next()
        .expect("non-empty specifications checked above")
        .clone();
    if first_ids.is_empty() {
        return Err(CalibrationError::new("calibration units are missing"));
    }
    for (metric_id, ids) in &unit_ids_by_metric {
        if ids != &first_ids {
            return Err(CalibrationError::new(format!(
                "calibration unit coverage differs for metric {metric_id}"
            )));
        }
    }

    let sample_count = first_ids.len();
    if sample_count < required_sample_count {
        return Err(CalibrationError::new(format!(
            "insufficient independent calibration units: need at least {required_sample_count}, \
             got {sample_count} for {:.3}% population coverage at {:.3}% family-wise confidence \
             across {metric_count} metrics",
            population_coverage * 100.0,
            familywise_confidence * 100.0
        )));
    }

    let per_metric_confidence_lower_bound =
        1.0 - population_coverage.powf(sample_count as f64);
    let familywise_confidence_lower_bound =
        (1.0 - metric_count as f64 * population_coverage.powf(sample_count as f64)).max(0.0);

    let mut metrics = Vec::with_capacity(metric_count);
    for spec in specs {
        let values = values_by_metric
            .get(&spec.metric_id)
            .expect("metric exists in values map");
        if values.len() != sample_count {
            return Err(CalibrationError::new(format!(
                "metric {} has {} observations; expected {}",
                spec.metric_id, values.len(), sample_count
            )));
        }

        let maximum_observed_absolute_delta = values.iter().copied().fold(0.0f64, f64::max);
        let calibrated_tolerance =
            (maximum_observed_absolute_delta * spec.safety_multiplier).max(spec.minimum_tolerance);
        if !calibrated_tolerance.is_finite() {
            return Err(CalibrationError::new(format!(
                "calibrated tolerance overflow for metric {}",
                spec.metric_id
            )));
        }

        metrics.push(MetricTolerance {
            metric_id: spec.metric_id.clone(),
            unit: spec.unit.clone(),
            sample_count,
            required_sample_count,
            maximum_observed_absolute_delta,
            safety_multiplier: spec.safety_multiplier,
            minimum_tolerance: spec.minimum_tolerance,
            calibrated_tolerance,
            per_metric_confidence_lower_bound,
        });
    }

    Ok(CalibrationReport {
        schema: "vocal-noop-calibration/v1".into(),
        population_coverage,
        requested_familywise_confidence: familywise_confidence,
        metric_count,
        per_metric_confidence_target,
        familywise_confidence_lower_bound,
        calibration_unit_ids: first_ids.into_iter().collect(),
        metrics,
    })
}

/// Minimum n such that the sample maximum gives a one-sided upper tolerance
/// bound with the requested population coverage and confidence.
///
/// For maximum order statistic X_(n), the miss probability is p^n, so
/// n >= ln(1 - confidence) / ln(p).
pub fn minimum_sample_count(
    population_coverage: f64,
    confidence: f64,
) -> Result<usize, CalibrationError> {
    validate_probability(population_coverage, "population_coverage")?;
    validate_probability(confidence, "confidence")?;

    let raw = ((1.0 - confidence).ln() / population_coverage.ln()).ceil();
    if !raw.is_finite() || raw < 1.0 || raw > MAX_SUPPORTED_SAMPLE_COUNT as f64 {
        return Err(CalibrationError::new(format!(
            "required sample count is outside supported range (1..={MAX_SUPPORTED_SAMPLE_COUNT})"
        )));
    }

    let mut count = raw as usize;
    while 1.0 - population_coverage.powf(count as f64) < confidence {
        count += 1;
        if count > MAX_SUPPORTED_SAMPLE_COUNT {
            return Err(CalibrationError::new(
                "required sample count exceeds supported maximum",
            ));
        }
    }
    while count > 1
        && 1.0 - population_coverage.powf((count - 1) as f64) >= confidence
    {
        count -= 1;
    }
    Ok(count)
}

fn validate_probability(value: f64, name: &str) -> Result<(), CalibrationError> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        return Err(CalibrationError::new(format!(
            "{name} must be finite and strictly between zero and one"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), CalibrationError> {
    if value.trim().is_empty() {
        Err(CalibrationError::new(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(metric_id: &str, unit: &str) -> MetricCalibrationSpec {
        MetricCalibrationSpec {
            metric_id: metric_id.into(),
            unit: unit.into(),
            safety_multiplier: 1.0,
            minimum_tolerance: 0.0,
        }
    }

    fn observations(metric_ids: &[(&str, &str)], count: usize) -> Vec<NoOpObservation> {
        let mut rows = Vec::new();
        for sample in 0..count {
            for (metric_index, (metric_id, unit)) in metric_ids.iter().enumerate() {
                rows.push(NoOpObservation {
                    calibration_unit_id: format!("cal-{sample:03}"),
                    metric_id: (*metric_id).into(),
                    unit: (*unit).into(),
                    absolute_delta: (sample + metric_index + 1) as f64 / 1000.0,
                });
            }
        }
        rows
    }

    #[test]
    fn one_metric_95_95_requires_59_independent_units() {
        assert_eq!(minimum_sample_count(0.95, 0.95).unwrap(), 59);
    }

    #[test]
    fn ten_metric_family_requires_104_units_for_95_95() {
        let per_metric_confidence = 1.0 - 0.05 / 10.0;
        assert_eq!(minimum_sample_count(0.95, per_metric_confidence).unwrap(), 104);
    }

    #[test]
    fn calibration_uses_maximum_and_applies_margin_and_floor() {
        let mut pitch = spec("pitch", "cents");
        pitch.safety_multiplier = 1.5;
        let mut energy = spec("energy", "dB");
        energy.minimum_tolerance = 0.002;
        let rows = observations(&[("pitch", "cents"), ("energy", "dB")], 104)
            .into_iter()
            .map(|mut row| {
                if row.metric_id == "pitch" {
                    row.absolute_delta /= 10.0;
                } else {
                    row.absolute_delta = 0.0001;
                }
                row
            })
            .collect::<Vec<_>>();

        let report = calibrate_noop_family(&[pitch, energy], &rows, 0.95, 0.95).unwrap();
        assert_eq!(report.schema, "vocal-noop-calibration/v1");
        assert_eq!(report.metric_count, 2);
        assert_eq!(report.metrics[0].maximum_observed_absolute_delta, 0.0104);
        assert_eq!(report.metrics[0].calibrated_tolerance, 0.0156);
        assert_eq!(report.metrics[1].calibrated_tolerance, 0.002);
        assert!(report.familywise_confidence_lower_bound >= 0.95);
        assert_eq!(report.calibration_unit_ids.len(), 104);
    }

    #[test]
    fn insufficient_calibration_units_fail_closed() {
        let rows = observations(&[("pitch", "cents")], 58);
        let error = calibrate_noop_family(&[spec("pitch", "cents")], &rows, 0.95, 0.95)
            .unwrap_err();
        assert!(error.message.contains("need at least 59"));
    }

    #[test]
    fn duplicate_units_are_rejected() {
        let mut rows = observations(&[("pitch", "cents")], 59);
        rows[1].calibration_unit_id = rows[0].calibration_unit_id.clone();
        assert!(calibrate_noop_family(&[spec("pitch", "cents")], &rows, 0.95, 0.95).is_err());
    }

    #[test]
    fn missing_metric_or_calibration_unit_is_rejected() {
        let specs = vec![spec("pitch", "cents"), spec("energy", "dB")];
        let mut rows = observations(&[("pitch", "cents"), ("energy", "dB")], 104);
        rows.retain(|row| !(row.metric_id == "energy" && row.calibration_unit_id == "cal-103"));
        assert!(calibrate_noop_family(&specs, &rows, 0.95, 0.95).is_err());
    }

    #[test]
    fn extra_or_unknown_metric_is_rejected() {
        let mut rows = observations(&[("pitch", "cents")], 59);
        rows.push(NoOpObservation {
            calibration_unit_id: "cal-999".into(),
            metric_id: "unknown".into(),
            unit: "cents".into(),
            absolute_delta: 0.1,
        });
        assert!(calibrate_noop_family(&[spec("pitch", "cents")], &rows, 0.95, 0.95).is_err());
    }

    #[test]
    fn wrong_units_and_non_finite_or_negative_deltas_are_rejected() {
        let mut rows = observations(&[("pitch", "cents")], 59);
        rows[0].unit = "Hz".into();
        assert!(calibrate_noop_family(&[spec("pitch", "cents")], &rows, 0.95, 0.95).is_err());

        let mut rows = observations(&[("pitch", "cents")], 59);
        rows[0].absolute_delta = f64::NAN;
        assert!(calibrate_noop_family(&[spec("pitch", "cents")], &rows, 0.95, 0.95).is_err());

        let mut rows = observations(&[("pitch", "cents")], 59);
        rows[0].absolute_delta = -0.01;
        assert!(calibrate_noop_family(&[spec("pitch", "cents")], &rows, 0.95, 0.95).is_err());
    }

    #[test]
    fn invalid_probabilities_and_multipliers_are_rejected() {
        assert!(minimum_sample_count(1.0, 0.95).is_err());
        assert!(minimum_sample_count(0.95, 0.0).is_err());

        let mut bad = spec("pitch", "cents");
        bad.safety_multiplier = 0.99;
        let rows = observations(&[("pitch", "cents")], 59);
        assert!(calibrate_noop_family(&[bad], &rows, 0.95, 0.95).is_err());
    }

    #[test]
    fn perfectly_repeatable_metric_can_produce_zero_tolerance() {
        let mut rows = observations(&[("pitch", "cents")], 59);
        rows.iter_mut().for_each(|row| row.absolute_delta = 0.0);
        let report = calibrate_noop_family(&[spec("pitch", "cents")], &rows, 0.95, 0.95).unwrap();
        assert_eq!(report.metrics[0].calibrated_tolerance, 0.0);
    }

    #[test]
    fn calibration_is_independent_of_observation_order() {
        let specs = vec![spec("pitch", "cents"), spec("energy", "dB")];
        let mut rows = observations(&[("pitch", "cents"), ("energy", "dB")], 104);
        rows.reverse();
        let report = calibrate_noop_family(&specs, &rows, 0.95, 0.95).unwrap();
        assert_eq!(report.metrics.len(), 2);
        assert!(report.familywise_confidence_lower_bound >= 0.95);
    }
}
