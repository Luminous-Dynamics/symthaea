//! Interval-aware, three-way assurance for vocal control measurements.
//!
//! This crate consumes uncertainty intervals produced by external, versioned
//! analyzers. It does not estimate intervals, inspect audio, or validate the
//! statistical calibration method that produced them.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const SCHEMA_V1: &str = "vocal-interval-assurance/v1";
const COVERAGE_EPSILON: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssuranceError {
    pub message: String,
}

impl AssuranceError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for AssuranceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for AssuranceError {}

/// A producer-supplied uncertainty interval for one measured acoustic delta.
///
/// The interval is interpreted conservatively: the full interval must satisfy
/// a rule to be verified compliant. Coverage and provenance fields are
/// declarations made by the upstream analysis pipeline; this crate checks
/// their consistency but cannot prove that the claims are statistically true.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementIntervalV1 {
    pub metric_id: String,
    pub unit: String,
    pub lower_delta: f64,
    pub upper_delta: f64,
    pub familywise_coverage: f64,
    pub interval_family_id: String,
    pub uncertainty_method_id: String,
    pub calibration_receipt_id: String,
    pub source_case_id: String,
    pub trial_id: String,
    pub input_manifest_digest: String,
}

/// The requested metric must respond inside this signed delta interval.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetConstraintV1 {
    pub metric_id: String,
    pub unit: String,
    pub expected_min_delta: f64,
    pub expected_max_delta: f64,
}

/// A collateral metric must remain inside the symmetric absolute-delta bound.
#[derive(Clone, Debug, PartialEq)]
pub struct CollateralConstraintV1 {
    pub metric_id: String,
    pub unit: String,
    pub absolute_tolerance: f64,
}

/// Exact identity and constraint universe for one control trial.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlAssuranceSpecV1 {
    pub control_id: String,
    pub source_case_id: String,
    pub trial_id: String,
    pub input_manifest_digest: String,
    pub interval_family_id: String,
    pub minimum_familywise_coverage: f64,
    pub target: TargetConstraintV1,
    pub collateral: Vec<CollateralConstraintV1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssuranceDecision {
    VerifiedCompliant,
    VerifiedNoncompliant,
    Indeterminate,
}

impl AssuranceDecision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VerifiedCompliant => "VERIFIED_COMPLIANT",
            Self::VerifiedNoncompliant => "VERIFIED_NONCOMPLIANT",
            Self::Indeterminate => "INDETERMINATE",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricAssuranceResultV1 {
    pub metric_id: String,
    pub unit: String,
    pub lower_delta: f64,
    pub upper_delta: f64,
    pub decision: AssuranceDecision,
    pub uncertainty_method_id: String,
    pub calibration_receipt_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlAssuranceReportV1 {
    pub schema: String,
    pub control_id: String,
    pub source_case_id: String,
    pub trial_id: String,
    pub input_manifest_digest: String,
    pub interval_family_id: String,
    pub familywise_coverage: f64,
    pub overall_decision: AssuranceDecision,
    pub metrics: Vec<MetricAssuranceResultV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionSummaryV1 {
    pub schema: String,
    pub trial_count: usize,
    pub verified_compliant: usize,
    pub verified_noncompliant: usize,
    pub indeterminate: usize,
    pub indeterminate_fraction: f64,
}

/// Evaluate one requested control response and all declared collateral guards.
///
/// Missing, extra, duplicated, stale, malformed, unit-incompatible or
/// provenance-incomplete observations are errors. A caller must not reinterpret
/// such an error as a compliant outcome.
pub fn evaluate_control(
    spec: &ControlAssuranceSpecV1,
    observations: &[MeasurementIntervalV1],
) -> Result<ControlAssuranceReportV1, AssuranceError> {
    validate_spec(spec)?;

    let mut constraints = BTreeMap::new();
    constraints.insert(
        spec.target.metric_id.as_str(),
        Constraint::Target {
            unit: spec.target.unit.as_str(),
            min: spec.target.expected_min_delta,
            max: spec.target.expected_max_delta,
        },
    );

    for collateral in &spec.collateral {
        if constraints
            .insert(
                collateral.metric_id.as_str(),
                Constraint::Collateral {
                    unit: collateral.unit.as_str(),
                    tolerance: collateral.absolute_tolerance,
                },
            )
            .is_some()
        {
            return Err(AssuranceError::new(format!(
                "duplicate target/collateral metric ID: {}",
                collateral.metric_id
            )));
        }
    }

    if observations.len() != constraints.len() {
        return Err(AssuranceError::new(format!(
            "expected exactly {} metric observations, received {}",
            constraints.len(),
            observations.len()
        )));
    }

    let mut by_metric: BTreeMap<&str, &MeasurementIntervalV1> = BTreeMap::new();
    let mut family_coverage = None;

    for observation in observations {
        validate_observation(observation)?;

        if observation.source_case_id != spec.source_case_id {
            return Err(AssuranceError::new(format!(
                "source-case mismatch for metric {}",
                observation.metric_id
            )));
        }
        if observation.trial_id != spec.trial_id {
            return Err(AssuranceError::new(format!(
                "trial mismatch for metric {}",
                observation.metric_id
            )));
        }
        if observation.input_manifest_digest != spec.input_manifest_digest {
            return Err(AssuranceError::new(format!(
                "input-manifest digest mismatch for metric {}",
                observation.metric_id
            )));
        }
        if observation.interval_family_id != spec.interval_family_id {
            return Err(AssuranceError::new(format!(
                "interval-family mismatch for metric {}",
                observation.metric_id
            )));
        }
        if observation.familywise_coverage + COVERAGE_EPSILON
            < spec.minimum_familywise_coverage
        {
            return Err(AssuranceError::new(format!(
                "family-wise coverage for metric {} is below the required minimum",
                observation.metric_id
            )));
        }

        match family_coverage {
            None => family_coverage = Some(observation.familywise_coverage),
            Some(value)
                if (value - observation.familywise_coverage).abs() <= COVERAGE_EPSILON => {}
            Some(_) => {
                return Err(AssuranceError::new(
                    "all observations in one interval family must declare the same coverage",
                ));
            }
        }

        if !constraints.contains_key(observation.metric_id.as_str()) {
            return Err(AssuranceError::new(format!(
                "unexpected metric observation: {}",
                observation.metric_id
            )));
        }
        if by_metric
            .insert(observation.metric_id.as_str(), observation)
            .is_some()
        {
            return Err(AssuranceError::new(format!(
                "duplicate metric observation: {}",
                observation.metric_id
            )));
        }
    }

    let mut metric_results = Vec::with_capacity(constraints.len());
    for (metric_id, constraint) in constraints {
        let observation = by_metric
            .get(metric_id)
            .copied()
            .ok_or_else(|| AssuranceError::new(format!("missing metric observation: {metric_id}")))?;

        let expected_unit = constraint.unit();
        if observation.unit != expected_unit {
            return Err(AssuranceError::new(format!(
                "unit mismatch for metric {metric_id}: expected {expected_unit}, received {}",
                observation.unit
            )));
        }

        let decision = match constraint {
            Constraint::Target { min, max, .. } => classify_target(
                observation.lower_delta,
                observation.upper_delta,
                min,
                max,
            ),
            Constraint::Collateral { tolerance, .. } => classify_collateral(
                observation.lower_delta,
                observation.upper_delta,
                tolerance,
            ),
        };

        metric_results.push(MetricAssuranceResultV1 {
            metric_id: metric_id.to_owned(),
            unit: observation.unit.clone(),
            lower_delta: observation.lower_delta,
            upper_delta: observation.upper_delta,
            decision,
            uncertainty_method_id: observation.uncertainty_method_id.clone(),
            calibration_receipt_id: observation.calibration_receipt_id.clone(),
        });
    }

    let overall_decision = aggregate_decisions(
        metric_results.iter().map(|metric| metric.decision),
    );
    let familywise_coverage = family_coverage
        .ok_or_else(|| AssuranceError::new("no interval-family coverage was supplied"))?;

    Ok(ControlAssuranceReportV1 {
        schema: SCHEMA_V1.to_owned(),
        control_id: spec.control_id.clone(),
        source_case_id: spec.source_case_id.clone(),
        trial_id: spec.trial_id.clone(),
        input_manifest_digest: spec.input_manifest_digest.clone(),
        interval_family_id: spec.interval_family_id.clone(),
        familywise_coverage,
        overall_decision,
        metrics: metric_results,
    })
}

/// Summarize three-way outcomes across distinct control trials.
///
/// Repeated reports for the same control/source/trial identity are rejected so
/// a caller cannot inflate a denominator by duplicating an already scored run.
pub fn summarize_decisions(
    reports: &[ControlAssuranceReportV1],
) -> Result<DecisionSummaryV1, AssuranceError> {
    if reports.is_empty() {
        return Err(AssuranceError::new(
            "at least one control-assurance report is required",
        ));
    }

    let mut seen = BTreeSet::new();
    let mut verified_compliant = 0usize;
    let mut verified_noncompliant = 0usize;
    let mut indeterminate = 0usize;

    for report in reports {
        if report.schema != SCHEMA_V1 {
            return Err(AssuranceError::new(format!(
                "unsupported report schema: {}",
                report.schema
            )));
        }
        let identity = (
            report.control_id.as_str(),
            report.source_case_id.as_str(),
            report.trial_id.as_str(),
        );
        if !seen.insert(identity) {
            return Err(AssuranceError::new(
                "duplicate control/source/trial report identity",
            ));
        }

        match report.overall_decision {
            AssuranceDecision::VerifiedCompliant => verified_compliant += 1,
            AssuranceDecision::VerifiedNoncompliant => verified_noncompliant += 1,
            AssuranceDecision::Indeterminate => indeterminate += 1,
        }
    }

    let trial_count = reports.len();
    Ok(DecisionSummaryV1 {
        schema: SCHEMA_V1.to_owned(),
        trial_count,
        verified_compliant,
        verified_noncompliant,
        indeterminate,
        indeterminate_fraction: indeterminate as f64 / trial_count as f64,
    })
}

enum Constraint<'a> {
    Target {
        unit: &'a str,
        min: f64,
        max: f64,
    },
    Collateral {
        unit: &'a str,
        tolerance: f64,
    },
}

impl Constraint<'_> {
    fn unit(&self) -> &str {
        match self {
            Self::Target { unit, .. } | Self::Collateral { unit, .. } => unit,
        }
    }
}

fn validate_spec(spec: &ControlAssuranceSpecV1) -> Result<(), AssuranceError> {
    require_non_empty(&spec.control_id, "control_id")?;
    require_non_empty(&spec.source_case_id, "source_case_id")?;
    require_non_empty(&spec.trial_id, "trial_id")?;
    require_non_empty(&spec.input_manifest_digest, "input_manifest_digest")?;
    require_non_empty(&spec.interval_family_id, "interval_family_id")?;
    validate_coverage(
        spec.minimum_familywise_coverage,
        "minimum_familywise_coverage",
    )?;

    require_non_empty(&spec.target.metric_id, "target metric_id")?;
    require_non_empty(&spec.target.unit, "target unit")?;
    if !spec.target.expected_min_delta.is_finite()
        || !spec.target.expected_max_delta.is_finite()
        || spec.target.expected_min_delta >= spec.target.expected_max_delta
    {
        return Err(AssuranceError::new(
            "target response bounds must be finite and strictly increasing",
        ));
    }
    if spec.target.expected_min_delta <= 0.0
        && spec.target.expected_max_delta >= 0.0
    {
        return Err(AssuranceError::new(
            "target response interval must exclude zero so a no-op cannot comply",
        ));
    }

    for collateral in &spec.collateral {
        require_non_empty(&collateral.metric_id, "collateral metric_id")?;
        require_non_empty(&collateral.unit, "collateral unit")?;
        if !collateral.absolute_tolerance.is_finite()
            || collateral.absolute_tolerance < 0.0
        {
            return Err(AssuranceError::new(format!(
                "collateral tolerance for {} must be finite and non-negative",
                collateral.metric_id
            )));
        }
    }
    Ok(())
}

fn validate_observation(
    observation: &MeasurementIntervalV1,
) -> Result<(), AssuranceError> {
    require_non_empty(&observation.metric_id, "observation metric_id")?;
    require_non_empty(&observation.unit, "observation unit")?;
    require_non_empty(&observation.interval_family_id, "interval_family_id")?;
    require_non_empty(&observation.uncertainty_method_id, "uncertainty_method_id")?;
    require_non_empty(&observation.calibration_receipt_id, "calibration_receipt_id")?;
    require_non_empty(&observation.source_case_id, "observation source_case_id")?;
    require_non_empty(&observation.trial_id, "observation trial_id")?;
    require_non_empty(
        &observation.input_manifest_digest,
        "observation input_manifest_digest",
    )?;
    if !observation.lower_delta.is_finite()
        || !observation.upper_delta.is_finite()
        || observation.lower_delta > observation.upper_delta
    {
        return Err(AssuranceError::new(format!(
            "measurement interval for {} must have finite, ordered bounds",
            observation.metric_id
        )));
    }
    validate_coverage(
        observation.familywise_coverage,
        "observation familywise_coverage",
    )?;
    Ok(())
}

fn validate_coverage(value: f64, field: &str) -> Result<(), AssuranceError> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(AssuranceError::new(format!(
            "{field} must be finite and in (0, 1]"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), AssuranceError> {
    if value.trim().is_empty() {
        return Err(AssuranceError::new(format!("{field} must not be empty")));
    }
    Ok(())
}

fn classify_target(
    lower: f64,
    upper: f64,
    expected_min: f64,
    expected_max: f64,
) -> AssuranceDecision {
    if lower >= expected_min && upper <= expected_max {
        AssuranceDecision::VerifiedCompliant
    } else if upper < expected_min || lower > expected_max {
        AssuranceDecision::VerifiedNoncompliant
    } else {
        AssuranceDecision::Indeterminate
    }
}

fn classify_collateral(
    lower: f64,
    upper: f64,
    absolute_tolerance: f64,
) -> AssuranceDecision {
    if lower >= -absolute_tolerance && upper <= absolute_tolerance {
        AssuranceDecision::VerifiedCompliant
    } else if upper < -absolute_tolerance || lower > absolute_tolerance {
        AssuranceDecision::VerifiedNoncompliant
    } else {
        AssuranceDecision::Indeterminate
    }
}

fn aggregate_decisions(
    decisions: impl Iterator<Item = AssuranceDecision>,
) -> AssuranceDecision {
    let mut has_indeterminate = false;
    for decision in decisions {
        match decision {
            AssuranceDecision::VerifiedNoncompliant => {
                return AssuranceDecision::VerifiedNoncompliant;
            }
            AssuranceDecision::Indeterminate => has_indeterminate = true,
            AssuranceDecision::VerifiedCompliant => {}
        }
    }
    if has_indeterminate {
        AssuranceDecision::Indeterminate
    } else {
        AssuranceDecision::VerifiedCompliant
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ControlAssuranceSpecV1 {
        ControlAssuranceSpecV1 {
            control_id: "vibrato-depth".to_owned(),
            source_case_id: "case-01".to_owned(),
            trial_id: "trial-01".to_owned(),
            input_manifest_digest: "sha256:input-manifest-01".to_owned(),
            interval_family_id: "family-01".to_owned(),
            minimum_familywise_coverage: 0.95,
            target: TargetConstraintV1 {
                metric_id: "vibrato-depth-delta".to_owned(),
                unit: "cents".to_owned(),
                expected_min_delta: 8.0,
                expected_max_delta: 12.0,
            },
            collateral: vec![
                CollateralConstraintV1 {
                    metric_id: "mean-f0-delta".to_owned(),
                    unit: "cents".to_owned(),
                    absolute_tolerance: 3.0,
                },
                CollateralConstraintV1 {
                    metric_id: "phrase-energy-delta".to_owned(),
                    unit: "dB".to_owned(),
                    absolute_tolerance: 0.5,
                },
            ],
        }
    }

    fn observation(
        metric_id: &str,
        unit: &str,
        lower_delta: f64,
        upper_delta: f64,
    ) -> MeasurementIntervalV1 {
        MeasurementIntervalV1 {
            metric_id: metric_id.to_owned(),
            unit: unit.to_owned(),
            lower_delta,
            upper_delta,
            familywise_coverage: 0.95,
            interval_family_id: "family-01".to_owned(),
            uncertainty_method_id: "cluster-bootstrap-v1".to_owned(),
            calibration_receipt_id: format!("calibration-{metric_id}"),
            source_case_id: "case-01".to_owned(),
            trial_id: "trial-01".to_owned(),
            input_manifest_digest: "sha256:input-manifest-01".to_owned(),
        }
    }

    fn valid_observations() -> Vec<MeasurementIntervalV1> {
        vec![
            observation("vibrato-depth-delta", "cents", 9.0, 11.0),
            observation("mean-f0-delta", "cents", -1.0, 2.0),
            observation("phrase-energy-delta", "dB", -0.2, 0.3),
        ]
    }

    fn decision(
        spec: &ControlAssuranceSpecV1,
        observations: &[MeasurementIntervalV1],
    ) -> AssuranceDecision {
        evaluate_control(spec, observations).unwrap().overall_decision
    }

    #[test]
    fn wholly_contained_target_and_collateral_intervals_are_compliant() {
        let report = evaluate_control(&spec(), &valid_observations()).unwrap();
        assert_eq!(report.schema, SCHEMA_V1);
        assert_eq!(
            report.overall_decision,
            AssuranceDecision::VerifiedCompliant
        );
        assert_eq!(report.metrics.len(), 3);
        assert_eq!(
            report.metrics[0].metric_id,
            "mean-f0-delta",
            "report metric order must be deterministic"
        );
    }

    #[test]
    fn target_interval_disjoint_below_expected_range_is_noncompliant() {
        let spec = spec();
        let mut observations = valid_observations();
        observations[0].lower_delta = 2.0;
        observations[0].upper_delta = 4.0;
        assert_eq!(
            decision(&spec, &observations),
            AssuranceDecision::VerifiedNoncompliant
        );
    }

    #[test]
    fn target_interval_disjoint_above_expected_range_is_noncompliant() {
        let spec = spec();
        let mut observations = valid_observations();
        observations[0].lower_delta = 13.0;
        observations[0].upper_delta = 14.0;
        assert_eq!(
            decision(&spec, &observations),
            AssuranceDecision::VerifiedNoncompliant
        );
    }

    #[test]
    fn target_interval_overlapping_boundary_is_indeterminate() {
        let spec = spec();
        let mut observations = valid_observations();
        observations[0].lower_delta = 7.0;
        observations[0].upper_delta = 9.0;
        assert_eq!(
            decision(&spec, &observations),
            AssuranceDecision::Indeterminate
        );
    }

    #[test]
    fn collateral_interval_entirely_outside_tolerance_is_noncompliant() {
        let spec = spec();
        let mut observations = valid_observations();
        observations[1].lower_delta = 4.0;
        observations[1].upper_delta = 5.0;
        assert_eq!(
            decision(&spec, &observations),
            AssuranceDecision::VerifiedNoncompliant
        );
    }

    #[test]
    fn collateral_interval_crossing_tolerance_boundary_is_indeterminate() {
        let spec = spec();
        let mut observations = valid_observations();
        observations[1].lower_delta = 2.0;
        observations[1].upper_delta = 4.0;
        assert_eq!(
            decision(&spec, &observations),
            AssuranceDecision::Indeterminate
        );
    }

    #[test]
    fn definite_failure_dominates_another_indeterminate_metric() {
        let spec = spec();
        let mut observations = valid_observations();
        observations[0].lower_delta = 7.0;
        observations[0].upper_delta = 9.0;
        observations[1].lower_delta = 4.0;
        observations[1].upper_delta = 5.0;
        assert_eq!(
            decision(&spec, &observations),
            AssuranceDecision::VerifiedNoncompliant
        );
    }

    #[test]
    fn missing_metric_is_rejected() {
        let spec = spec();
        let mut observations = valid_observations();
        observations.pop();
        assert!(evaluate_control(&spec, &observations).is_err());
    }

    #[test]
    fn extra_and_duplicate_metrics_are_rejected() {
        let spec = spec();
        let mut extra = valid_observations();
        extra.push(observation("unrequested", "dB", 0.0, 0.0));
        assert!(evaluate_control(&spec, &extra).is_err());

        let mut duplicate = valid_observations();
        duplicate.push(duplicate[0].clone());
        assert!(evaluate_control(&spec, &duplicate).is_err());
    }

    #[test]
    fn mismatched_trial_digest_or_interval_family_is_rejected() {
        let spec = spec();

        let mut stale_trial = valid_observations();
        stale_trial[0].trial_id = "other-trial".to_owned();
        assert!(evaluate_control(&spec, &stale_trial).is_err());

        let mut stale_digest = valid_observations();
        stale_digest[0].input_manifest_digest = "sha256:other-input".to_owned();
        assert!(evaluate_control(&spec, &stale_digest).is_err());

        let mut wrong_family = valid_observations();
        wrong_family[0].interval_family_id = "other-family".to_owned();
        assert!(evaluate_control(&spec, &wrong_family).is_err());
    }

    #[test]
    fn below_minimum_or_inconsistent_family_coverage_is_rejected() {
        let spec = spec();

        let mut low_coverage = valid_observations();
        low_coverage[0].familywise_coverage = 0.90;
        assert!(evaluate_control(&spec, &low_coverage).is_err());

        let mut inconsistent = valid_observations();
        inconsistent[0].familywise_coverage = 0.96;
        assert!(evaluate_control(&spec, &inconsistent).is_err());
    }

    #[test]
    fn units_and_provenance_are_required() {
        let spec = spec();

        let mut wrong_unit = valid_observations();
        wrong_unit[0].unit = "Hz".to_owned();
        assert!(evaluate_control(&spec, &wrong_unit).is_err());

        let mut missing_method = valid_observations();
        missing_method[0].uncertainty_method_id.clear();
        assert!(evaluate_control(&spec, &missing_method).is_err());

        let mut missing_calibration = valid_observations();
        missing_calibration[0].calibration_receipt_id.clear();
        assert!(evaluate_control(&spec, &missing_calibration).is_err());
    }

    #[test]
    fn malformed_intervals_are_rejected() {
        let spec = spec();

        let mut reversed = valid_observations();
        reversed[0].lower_delta = 12.0;
        reversed[0].upper_delta = 10.0;
        assert!(evaluate_control(&spec, &reversed).is_err());

        let mut non_finite = valid_observations();
        non_finite[0].upper_delta = f64::NAN;
        assert!(evaluate_control(&spec, &non_finite).is_err());
    }

    #[test]
    fn target_rule_must_exclude_zero_and_collateral_tolerance_must_be_valid() {
        let mut zero_target = spec();
        zero_target.target.expected_min_delta = -1.0;
        zero_target.target.expected_max_delta = 1.0;
        assert!(evaluate_control(&zero_target, &valid_observations()).is_err());

        let mut invalid_collateral = spec();
        invalid_collateral.collateral[0].absolute_tolerance = f64::INFINITY;
        assert!(evaluate_control(&invalid_collateral, &valid_observations()).is_err());
    }

    #[test]
    fn zero_tolerance_requires_an_exactly_zero_interval() {
        let mut spec = spec();
        spec.collateral[0].absolute_tolerance = 0.0;
        let mut exact = valid_observations();
        exact[1].lower_delta = 0.0;
        exact[1].upper_delta = 0.0;
        assert_eq!(
            decision(&spec, &exact),
            AssuranceDecision::VerifiedCompliant
        );

        let mut uncertain = valid_observations();
        uncertain[1].lower_delta = -0.1;
        uncertain[1].upper_delta = 0.1;
        assert_eq!(
            decision(&spec, &uncertain),
            AssuranceDecision::Indeterminate
        );
    }

    #[test]
    fn decision_summary_reports_indeterminate_fraction_and_rejects_duplicates() {
        let first = evaluate_control(&spec(), &valid_observations()).unwrap();

        let mut second_spec = spec();
        second_spec.source_case_id = "case-02".to_owned();
        second_spec.trial_id = "trial-02".to_owned();
        let mut second_observations = valid_observations();
        for observation in &mut second_observations {
            observation.source_case_id = "case-02".to_owned();
            observation.trial_id = "trial-02".to_owned();
        }
        second_observations[0].lower_delta = 7.0;
        second_observations[0].upper_delta = 9.0;
        let second = evaluate_control(&second_spec, &second_observations).unwrap();

        let summary = summarize_decisions(&[first.clone(), second]).unwrap();
        assert_eq!(summary.trial_count, 2);
        assert_eq!(summary.verified_compliant, 1);
        assert_eq!(summary.indeterminate, 1);
        assert_eq!(summary.indeterminate_fraction, 0.5);
        assert!(summarize_decisions(&[first.clone(), first]).is_err());
        assert!(summarize_decisions(&[]).is_err());
    }
}
