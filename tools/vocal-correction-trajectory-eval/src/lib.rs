//! Assurance evaluation for bounded closed-loop vocal correction trajectories.
//!
//! This crate evaluates a supplied sequence of render receipts and uncertainty
//! intervals. It does not render audio, estimate controls, generate confidence
//! intervals, or choose corrective actions.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const TRAJECTORY_SCHEMA_V1: &str = "vocal-correction-trajectory/v1";
const COVERAGE_EPSILON: f64 = 1e-12;
const MAX_CORRECTION_BUDGET: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectoryError {
    pub message: String,
}

impl TrajectoryError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for TrajectoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for TrajectoryError {}

/// Accepted interval for one target acoustic metric.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetConstraintV1 {
    pub metric_id: String,
    pub unit: String,
    pub accepted_min: f64,
    pub accepted_max: f64,
    /// Minimum decrease in conservative distance-to-target required for a
    /// target that remains outside its accepted interval.
    pub minimum_progress: f64,
    /// Maximum permitted increase in conservative distance-to-target between
    /// correction steps. A previously compliant target may never regress.
    pub maximum_allowed_regression: f64,
}

/// Absolute-delta guard for a protected acoustic property.
#[derive(Clone, Debug, PartialEq)]
pub struct CollateralConstraintV1 {
    pub metric_id: String,
    pub unit: String,
    pub absolute_tolerance: f64,
}

/// Frozen identity and policy for one complete correction trajectory.
///
/// frozen_context_digest must commit to the source case, lyrics, selected
/// pronunciation, score/alignment, voice identity/authorization, renderer and
/// voicebank artifacts, analyzer artifacts, preprocessing, calibration
/// profile, and all other inputs that must not change during the trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct TrajectorySpecV1 {
    pub trajectory_id: String,
    pub source_case_id: String,
    pub control_id: String,
    pub frozen_context_digest: String,
    pub baseline_output_digest: String,
    pub minimum_familywise_coverage: f64,
    pub maximum_corrections: usize,
    pub maximum_action_norm: f64,
    pub targets: Vec<TargetConstraintV1>,
    pub collateral: Vec<CollateralConstraintV1>,
}

/// One uncertainty interval emitted by a versioned external acoustic analyzer.
///
/// The interval-family ID and coverage are producer declarations. This crate
/// checks their structural consistency but cannot prove their statistical
/// calibration or independence.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricIntervalV1 {
    pub metric_id: String,
    pub unit: String,
    pub lower_delta: f64,
    pub upper_delta: f64,
    pub interval_family_id: String,
    pub familywise_coverage: f64,
    pub uncertainty_method_id: String,
    pub calibration_receipt_id: String,
}

/// Evidence for the baseline render or one subsequent correction render.
///
/// Iteration zero is the untouched baseline and has no parent or action.
/// Every later iteration must reference the exact prior output and carry a
/// non-empty action digest plus a bounded, non-zero action norm.
#[derive(Clone, Debug, PartialEq)]
pub struct IterationEvidenceV1 {
    pub iteration_index: usize,
    pub source_case_id: String,
    pub frozen_context_digest: String,
    pub parent_output_digest: Option<String>,
    pub output_digest: String,
    pub action_digest: Option<String>,
    pub action_norm: f64,
    pub measurements: Vec<MetricIntervalV1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetMetricDecision {
    Compliant,
    NeedsCorrection,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollateralMetricDecision {
    Compliant,
    Noncompliant,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryOutcome {
    Converged,
    NeedsCorrection,
    Indeterminate,
    Noncompliant,
    Stalled,
    IterationLimitReached,
}

impl TrajectoryOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "CONVERGED",
            Self::NeedsCorrection => "NEEDS_CORRECTION",
            Self::Indeterminate => "INDETERMINATE",
            Self::Noncompliant => "NONCOMPLIANT",
            Self::Stalled => "STALLED",
            Self::IterationLimitReached => "ITERATION_LIMIT_REACHED",
        }
    }

    const fn is_terminal(self) -> bool {
        !matches!(self, Self::NeedsCorrection)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProgressCheck {
    Progress,
    NoProgress,
    Regression,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TargetMetricResultV1 {
    pub metric_id: String,
    pub unit: String,
    pub lower_delta: f64,
    pub upper_delta: f64,
    pub decision: TargetMetricDecision,
    /// Conservative upper bound on distance to the accepted interval.
    /// None is used only for indeterminate intervals.
    pub distance_upper_bound: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollateralMetricResultV1 {
    pub metric_id: String,
    pub unit: String,
    pub lower_delta: f64,
    pub upper_delta: f64,
    pub decision: CollateralMetricDecision,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IterationResultV1 {
    pub iteration_index: usize,
    pub output_digest: String,
    pub action_digest: Option<String>,
    pub action_norm: f64,
    pub outcome: TrajectoryOutcome,
    pub targets: Vec<TargetMetricResultV1>,
    pub collateral: Vec<CollateralMetricResultV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrajectoryReportV1 {
    pub schema: String,
    pub trajectory_id: String,
    pub source_case_id: String,
    pub frozen_context_digest: String,
    pub initial_output_digest: String,
    pub final_output_digest: String,
    pub corrections_used: usize,
    pub outcome: TrajectoryOutcome,
    pub iterations: Vec<IterationResultV1>,
}

/// Evaluate a complete or partial sequence of baseline/correction renders.
///
/// A returned NeedsCorrection means the supplied trace is internally
/// qualified so far and another bounded correction may be attempted. It is not
/// a quality pass. Indeterminate and all other terminal outcomes stop the
/// trajectory; any supplied later mutation is rejected.
pub fn evaluate_trajectory(
    spec: &TrajectorySpecV1,
    evidence: &[IterationEvidenceV1],
) -> Result<TrajectoryReportV1, TrajectoryError> {
    validate_spec(spec)?;

    if evidence.is_empty() {
        return Err(TrajectoryError::new(
            "at least the baseline iteration is required",
        ));
    }
    if evidence.len() > spec.maximum_corrections.saturating_add(1) {
        return Err(TrajectoryError::new(
            "trajectory exceeds the declared maximum correction budget",
        ));
    }

    let mut results: Vec<IterationResultV1> = Vec::with_capacity(evidence.len());
    let mut previous_output_digest: Option<&str> = None;
    let mut corrections_used = 0usize;
    let mut final_outcome = TrajectoryOutcome::NeedsCorrection;

    for (expected_index, iteration) in evidence.iter().enumerate() {
        validate_iteration_identity(
            spec,
            iteration,
            expected_index,
            previous_output_digest,
        )?;

        let observations = index_observations(spec, &iteration.measurements)?;
        let (targets, collateral) = evaluate_metrics(spec, &observations)?;
        let mut outcome = aggregate_metric_decisions(&targets, &collateral);

        if expected_index == 0 {
            // The baseline has no action and establishes initial residuals
            // and protected-region deltas.
        } else {
            corrections_used += 1;

            if iteration.action_norm > spec.maximum_action_norm {
                outcome = TrajectoryOutcome::Noncompliant;
            } else if iteration.output_digest
                == iteration.parent_output_digest.as_deref().unwrap_or_default()
            {
                outcome = TrajectoryOutcome::Stalled;
            } else if !matches!(
                outcome,
                TrajectoryOutcome::Indeterminate | TrajectoryOutcome::Noncompliant
            ) {
                let previous_targets = &results[expected_index - 1].targets;
                match check_target_progress(
                    &spec.targets,
                    previous_targets,
                    &targets,
                )? {
                    ProgressCheck::Progress => {}
                    ProgressCheck::NoProgress => outcome = TrajectoryOutcome::Stalled,
                    ProgressCheck::Regression => outcome = TrajectoryOutcome::Noncompliant,
                }
            }
        }

        let result = IterationResultV1 {
            iteration_index: iteration.iteration_index,
            output_digest: iteration.output_digest.clone(),
            action_digest: iteration.action_digest.clone(),
            action_norm: iteration.action_norm,
            outcome,
            targets,
            collateral,
        };
        results.push(result);
        final_outcome = outcome;

        if outcome.is_terminal() && expected_index + 1 < evidence.len() {
            return Err(TrajectoryError::new(format!(
                "iteration {} is terminal ({}), but later mutation evidence was supplied",
                expected_index,
                outcome.as_str()
            )));
        }

        previous_output_digest = Some(iteration.output_digest.as_str());
    }

    if final_outcome == TrajectoryOutcome::NeedsCorrection
        && corrections_used >= spec.maximum_corrections
    {
        final_outcome = TrajectoryOutcome::IterationLimitReached;
    }

    let final_output_digest = evidence
        .last()
        .map(|iteration| iteration.output_digest.clone())
        .ok_or_else(|| TrajectoryError::new("trajectory lost its final output identity"))?;

    Ok(TrajectoryReportV1 {
        schema: TRAJECTORY_SCHEMA_V1.to_owned(),
        trajectory_id: spec.trajectory_id.clone(),
        source_case_id: spec.source_case_id.clone(),
        frozen_context_digest: spec.frozen_context_digest.clone(),
        initial_output_digest: spec.baseline_output_digest.clone(),
        final_output_digest,
        corrections_used,
        outcome: final_outcome,
        iterations: results,
    })
}

fn validate_spec(spec: &TrajectorySpecV1) -> Result<(), TrajectoryError> {
    require_non_empty(&spec.trajectory_id, "trajectory_id")?;
    require_non_empty(&spec.source_case_id, "source_case_id")?;
    require_non_empty(&spec.control_id, "control_id")?;
    validate_digest(&spec.frozen_context_digest, "frozen_context_digest")?;
    validate_digest(&spec.baseline_output_digest, "baseline_output_digest")?;

    if !spec.minimum_familywise_coverage.is_finite()
        || spec.minimum_familywise_coverage <= 0.0
        || spec.minimum_familywise_coverage > 1.0
    {
        return Err(TrajectoryError::new(
            "minimum_familywise_coverage must be finite and in (0, 1]",
        ));
    }
    if spec.maximum_corrections > MAX_CORRECTION_BUDGET {
        return Err(TrajectoryError::new(format!(
            "maximum_corrections exceeds the supported budget of {MAX_CORRECTION_BUDGET}"
        )));
    }
    if !spec.maximum_action_norm.is_finite() || spec.maximum_action_norm <= 0.0 {
        return Err(TrajectoryError::new(
            "maximum_action_norm must be finite and positive",
        ));
    }
    if spec.targets.is_empty() {
        return Err(TrajectoryError::new(
            "at least one target metric is required",
        ));
    }

    let mut metric_ids = BTreeSet::new();
    for target in &spec.targets {
        require_non_empty(&target.metric_id, "target metric_id")?;
        require_non_empty(&target.unit, "target unit")?;
        if !metric_ids.insert(target.metric_id.as_str()) {
            return Err(TrajectoryError::new(format!(
                "duplicate target/collateral metric ID: {}",
                target.metric_id
            )));
        }
        if !target.accepted_min.is_finite()
            || !target.accepted_max.is_finite()
            || target.accepted_min >= target.accepted_max
        {
            return Err(TrajectoryError::new(format!(
                "accepted range for target {} must be finite and increasing",
                target.metric_id
            )));
        }
        if !target.minimum_progress.is_finite() || target.minimum_progress <= 0.0 {
            return Err(TrajectoryError::new(format!(
                "minimum_progress for target {} must be finite and positive",
                target.metric_id
            )));
        }
        if !target.maximum_allowed_regression.is_finite()
            || target.maximum_allowed_regression < 0.0
        {
            return Err(TrajectoryError::new(format!(
                "maximum_allowed_regression for target {} must be finite and non-negative",
                target.metric_id
            )));
        }
    }

    for collateral in &spec.collateral {
        require_non_empty(&collateral.metric_id, "collateral metric_id")?;
        require_non_empty(&collateral.unit, "collateral unit")?;
        if !metric_ids.insert(collateral.metric_id.as_str()) {
            return Err(TrajectoryError::new(format!(
                "duplicate target/collateral metric ID: {}",
                collateral.metric_id
            )));
        }
        if !collateral.absolute_tolerance.is_finite()
            || collateral.absolute_tolerance < 0.0
        {
            return Err(TrajectoryError::new(format!(
                "absolute_tolerance for collateral {} must be finite and non-negative",
                collateral.metric_id
            )));
        }
    }
    Ok(())
}

fn validate_iteration_identity(
    spec: &TrajectorySpecV1,
    iteration: &IterationEvidenceV1,
    expected_index: usize,
    previous_output_digest: Option<&str>,
) -> Result<(), TrajectoryError> {
    if iteration.iteration_index != expected_index {
        return Err(TrajectoryError::new(format!(
            "expected iteration index {expected_index}, received {}",
            iteration.iteration_index
        )));
    }
    if iteration.source_case_id != spec.source_case_id {
        return Err(TrajectoryError::new(format!(
            "source-case identity changed at iteration {expected_index}"
        )));
    }
    if iteration.frozen_context_digest != spec.frozen_context_digest {
        return Err(TrajectoryError::new(format!(
            "frozen context changed at iteration {expected_index}"
        )));
    }
    validate_digest(&iteration.output_digest, "iteration output_digest")?;

    if expected_index == 0 {
        if iteration.parent_output_digest.is_some() {
            return Err(TrajectoryError::new(
                "baseline iteration must not declare a parent output",
            ));
        }
        if iteration.action_digest.is_some() || iteration.action_norm != 0.0 {
            return Err(TrajectoryError::new(
                "baseline iteration must not contain a correction action",
            ));
        }
        if iteration.output_digest != spec.baseline_output_digest {
            return Err(TrajectoryError::new(
                "baseline output digest does not match the trajectory specification",
            ));
        }
    } else {
        let expected_parent = previous_output_digest
            .ok_or_else(|| TrajectoryError::new("correction iteration has no prior output"))?;
        let actual_parent = iteration
            .parent_output_digest
            .as_deref()
            .ok_or_else(|| TrajectoryError::new("correction iteration is missing its parent output"))?;
        validate_digest(actual_parent, "parent_output_digest")?;
        if actual_parent != expected_parent {
            return Err(TrajectoryError::new(format!(
                "parent-output digest mismatch at iteration {expected_index}"
            )));
        }
        let action_digest = iteration
            .action_digest
            .as_deref()
            .ok_or_else(|| TrajectoryError::new("correction iteration is missing action_digest"))?;
        validate_digest(action_digest, "action_digest")?;
        if !iteration.action_norm.is_finite() || iteration.action_norm <= 0.0 {
            return Err(TrajectoryError::new(
                "correction action_norm must be finite and positive",
            ));
        }
    }
    Ok(())
}

fn index_observations<'a>(
    spec: &TrajectorySpecV1,
    observations: &'a [MetricIntervalV1],
) -> Result<BTreeMap<&'a str, &'a MetricIntervalV1>, TrajectoryError> {
    let expected_count = spec.targets.len() + spec.collateral.len();
    if observations.len() != expected_count {
        return Err(TrajectoryError::new(format!(
            "expected exactly {expected_count} target/collateral observations, received {}",
            observations.len()
        )));
    }

    let mut expected_units = BTreeMap::new();
    for target in &spec.targets {
        expected_units.insert(target.metric_id.as_str(), target.unit.as_str());
    }
    for collateral in &spec.collateral {
        expected_units.insert(collateral.metric_id.as_str(), collateral.unit.as_str());
    }

    let mut indexed = BTreeMap::new();
    let mut family_id: Option<&str> = None;
    let mut family_coverage: Option<f64> = None;

    for observation in observations {
        require_non_empty(&observation.metric_id, "observation metric_id")?;
        require_non_empty(&observation.unit, "observation unit")?;
        require_non_empty(&observation.interval_family_id, "interval_family_id")?;
        require_non_empty(&observation.uncertainty_method_id, "uncertainty_method_id")?;
        require_non_empty(&observation.calibration_receipt_id, "calibration_receipt_id")?;

        if !observation.lower_delta.is_finite()
            || !observation.upper_delta.is_finite()
            || observation.lower_delta > observation.upper_delta
        {
            return Err(TrajectoryError::new(format!(
                "measurement interval for {} must have finite, ordered bounds",
                observation.metric_id
            )));
        }
        if !observation.familywise_coverage.is_finite()
            || observation.familywise_coverage <= 0.0
            || observation.familywise_coverage > 1.0
        {
            return Err(TrajectoryError::new(
                "observation familywise_coverage must be finite and in (0, 1]",
            ));
        }
        if observation.familywise_coverage + COVERAGE_EPSILON
            < spec.minimum_familywise_coverage
        {
            return Err(TrajectoryError::new(format!(
                "family-wise coverage for metric {} is below the required minimum",
                observation.metric_id
            )));
        }

        match family_id {
            None => family_id = Some(observation.interval_family_id.as_str()),
            Some(expected) if expected == observation.interval_family_id.as_str() => {}
            Some(_) => {
                return Err(TrajectoryError::new(
                    "all metrics in one iteration must share an interval_family_id",
                ));
            }
        }
        match family_coverage {
            None => family_coverage = Some(observation.familywise_coverage),
            Some(expected)
                if (expected - observation.familywise_coverage).abs() <= COVERAGE_EPSILON => {}
            Some(_) => {
                return Err(TrajectoryError::new(
                    "all metrics in one iteration must declare the same family-wise coverage",
                ));
            }
        }

        let expected_unit = expected_units
            .get(observation.metric_id.as_str())
            .copied()
            .ok_or_else(|| {
                TrajectoryError::new(format!(
                    "unexpected metric observation: {}",
                    observation.metric_id
                ))
            })?;
        if observation.unit != expected_unit {
            return Err(TrajectoryError::new(format!(
                "unit mismatch for {}: expected {}, received {}",
                observation.metric_id, expected_unit, observation.unit
            )));
        }
        if indexed
            .insert(observation.metric_id.as_str(), observation)
            .is_some()
        {
            return Err(TrajectoryError::new(format!(
                "duplicate metric observation: {}",
                observation.metric_id
            )));
        }
    }

    for metric_id in expected_units.keys() {
        if !indexed.contains_key(metric_id) {
            return Err(TrajectoryError::new(format!(
                "missing metric observation: {metric_id}"
            )));
        }
    }
    Ok(indexed)
}

fn evaluate_metrics(
    spec: &TrajectorySpecV1,
    observations: &BTreeMap<&str, &MetricIntervalV1>,
) -> Result<
    (
        Vec<TargetMetricResultV1>,
        Vec<CollateralMetricResultV1>,
    ),
    TrajectoryError,
> {
    let mut targets = Vec::with_capacity(spec.targets.len());
    for constraint in &spec.targets {
        let observation = observations
            .get(constraint.metric_id.as_str())
            .copied()
            .ok_or_else(|| {
                TrajectoryError::new(format!(
                    "missing target observation: {}",
                    constraint.metric_id
                ))
            })?;
        let decision = classify_target(
            observation.lower_delta,
            observation.upper_delta,
            constraint.accepted_min,
            constraint.accepted_max,
        );
        let distance_upper_bound = match decision {
            TargetMetricDecision::Compliant => Some(0.0),
            TargetMetricDecision::NeedsCorrection => Some(target_distance_upper_bound(
                observation.lower_delta,
                observation.upper_delta,
                constraint.accepted_min,
                constraint.accepted_max,
            )),
            TargetMetricDecision::Indeterminate => None,
        };
        targets.push(TargetMetricResultV1 {
            metric_id: constraint.metric_id.clone(),
            unit: observation.unit.clone(),
            lower_delta: observation.lower_delta,
            upper_delta: observation.upper_delta,
            decision,
            distance_upper_bound,
        });
    }

    let mut collateral = Vec::with_capacity(spec.collateral.len());
    for constraint in &spec.collateral {
        let observation = observations
            .get(constraint.metric_id.as_str())
            .copied()
            .ok_or_else(|| {
                TrajectoryError::new(format!(
                    "missing collateral observation: {}",
                    constraint.metric_id
                ))
            })?;
        collateral.push(CollateralMetricResultV1 {
            metric_id: constraint.metric_id.clone(),
            unit: observation.unit.clone(),
            lower_delta: observation.lower_delta,
            upper_delta: observation.upper_delta,
            decision: classify_collateral(
                observation.lower_delta,
                observation.upper_delta,
                constraint.absolute_tolerance,
            ),
        });
    }
    Ok((targets, collateral))
}

fn classify_target(
    lower: f64,
    upper: f64,
    accepted_min: f64,
    accepted_max: f64,
) -> TargetMetricDecision {
    if lower >= accepted_min && upper <= accepted_max {
        TargetMetricDecision::Compliant
    } else if upper < accepted_min || lower > accepted_max {
        TargetMetricDecision::NeedsCorrection
    } else {
        TargetMetricDecision::Indeterminate
    }
}

fn target_distance_upper_bound(
    lower: f64,
    upper: f64,
    accepted_min: f64,
    accepted_max: f64,
) -> f64 {
    if upper < accepted_min {
        accepted_min - lower
    } else if lower > accepted_max {
        upper - accepted_max
    } else {
        // This helper is called only for a decisive NeedsCorrection interval.
        0.0
    }
}

fn classify_collateral(
    lower: f64,
    upper: f64,
    tolerance: f64,
) -> CollateralMetricDecision {
    if lower >= -tolerance && upper <= tolerance {
        CollateralMetricDecision::Compliant
    } else if upper < -tolerance || lower > tolerance {
        CollateralMetricDecision::Noncompliant
    } else {
        CollateralMetricDecision::Indeterminate
    }
}

fn aggregate_metric_decisions(
    targets: &[TargetMetricResultV1],
    collateral: &[CollateralMetricResultV1],
) -> TrajectoryOutcome {
    if collateral
        .iter()
        .any(|metric| metric.decision == CollateralMetricDecision::Noncompliant)
    {
        return TrajectoryOutcome::Noncompliant;
    }
    if targets
        .iter()
        .any(|metric| metric.decision == TargetMetricDecision::Indeterminate)
        || collateral
            .iter()
            .any(|metric| metric.decision == CollateralMetricDecision::Indeterminate)
    {
        return TrajectoryOutcome::Indeterminate;
    }
    if targets
        .iter()
        .all(|metric| metric.decision == TargetMetricDecision::Compliant)
    {
        TrajectoryOutcome::Converged
    } else {
        TrajectoryOutcome::NeedsCorrection
    }
}

/// Return true when every target is non-regressing and at least one unsatisfied
/// target achieves its declared minimum progress or becomes compliant.
fn check_target_progress(
    constraints: &[TargetConstraintV1],
    previous: &[TargetMetricResultV1],
    current: &[TargetMetricResultV1],
) -> Result<ProgressCheck, TrajectoryError> {
    if constraints.len() != previous.len() || constraints.len() != current.len() {
        return Err(TrajectoryError::new(
            "target result cardinality changed between iterations",
        ));
    }

    let mut any_progress = false;
    for (index, constraint) in constraints.iter().enumerate() {
        let before = &previous[index];
        let after = &current[index];

        if before.metric_id != constraint.metric_id || after.metric_id != constraint.metric_id {
            return Err(TrajectoryError::new(
                "target result order does not match the frozen specification",
            ));
        }

        if before.decision == TargetMetricDecision::Compliant {
            if after.decision != TargetMetricDecision::Compliant {
                return Ok(ProgressCheck::Regression);
            }
            continue;
        }

        if after.decision == TargetMetricDecision::Indeterminate {
            return Err(TrajectoryError::new(format!(
                "target {} became indeterminate during progress evaluation",
                constraint.metric_id
            )));
        }

        let before_distance = before.distance_upper_bound.ok_or_else(|| {
            TrajectoryError::new(format!(
                "previous target {} lacks a distance bound",
                constraint.metric_id
            ))
        })?;
        let after_distance = after.distance_upper_bound.ok_or_else(|| {
            TrajectoryError::new(format!(
                "current target {} lacks a distance bound",
                constraint.metric_id
            ))
        })?;

        if after_distance > before_distance + constraint.maximum_allowed_regression {
            return Ok(ProgressCheck::Regression);
        }

        if after.decision == TargetMetricDecision::Compliant
            || before_distance - after_distance >= constraint.minimum_progress
        {
            any_progress = true;
        }
    }
    Ok(if any_progress {
        ProgressCheck::Progress
    } else {
        ProgressCheck::NoProgress
    })
}

fn validate_digest(value: &str, field: &str) -> Result<(), TrajectoryError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(TrajectoryError::new(format!(
            "{field} must be a 64-character lowercase hexadecimal digest"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), TrajectoryError> {
    if value.trim().is_empty() {
        return Err(TrajectoryError::new(format!("{field} must not be empty")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn spec() -> TrajectorySpecV1 {
        TrajectorySpecV1 {
            trajectory_id: "trajectory-01".to_owned(),
            source_case_id: "case-01".to_owned(),
            control_id: "vibrato-depth".to_owned(),
            frozen_context_digest: digest('a'),
            baseline_output_digest: digest('b'),
            minimum_familywise_coverage: 0.95,
            maximum_corrections: 3,
            maximum_action_norm: 2.0,
            targets: vec![
                TargetConstraintV1 {
                    metric_id: "vibrato-depth".to_owned(),
                    unit: "cents".to_owned(),
                    accepted_min: 8.0,
                    accepted_max: 12.0,
                    minimum_progress: 1.0,
                    maximum_allowed_regression: 0.5,
                },
                TargetConstraintV1 {
                    metric_id: "mean-f0".to_owned(),
                    unit: "cents".to_owned(),
                    accepted_min: -2.0,
                    accepted_max: 2.0,
                    minimum_progress: 0.5,
                    maximum_allowed_regression: 0.25,
                },
            ],
            collateral: vec![
                CollateralConstraintV1 {
                    metric_id: "phrase-energy".to_owned(),
                    unit: "dB".to_owned(),
                    absolute_tolerance: 0.5,
                },
            ],
        }
    }

    fn metric(
        metric_id: &str,
        unit: &str,
        lower_delta: f64,
        upper_delta: f64,
    ) -> MetricIntervalV1 {
        MetricIntervalV1 {
            metric_id: metric_id.to_owned(),
            unit: unit.to_owned(),
            lower_delta,
            upper_delta,
            interval_family_id: "family-01".to_owned(),
            familywise_coverage: 0.95,
            uncertainty_method_id: "cluster-bootstrap-v1".to_owned(),
            calibration_receipt_id: format!("calibration-{metric_id}"),
        }
    }

    fn baseline_metrics(
        vibrato: (f64, f64),
        mean_f0: (f64, f64),
        energy: (f64, f64),
    ) -> Vec<MetricIntervalV1> {
        vec![
            metric("vibrato-depth", "cents", vibrato.0, vibrato.1),
            metric("mean-f0", "cents", mean_f0.0, mean_f0.1),
            metric("phrase-energy", "dB", energy.0, energy.1),
        ]
    }

    fn iteration(
        index: usize,
        parent: Option<String>,
        output: char,
        action: Option<char>,
        action_norm: f64,
        metrics: Vec<MetricIntervalV1>,
    ) -> IterationEvidenceV1 {
        IterationEvidenceV1 {
            iteration_index: index,
            source_case_id: "case-01".to_owned(),
            frozen_context_digest: digest('a'),
            parent_output_digest: parent,
            output_digest: digest(output),
            action_digest: action.map(digest),
            action_norm,
            measurements: metrics,
        }
    }

    fn baseline(
        vibrato: (f64, f64),
        mean_f0: (f64, f64),
        energy: (f64, f64),
    ) -> IterationEvidenceV1 {
        iteration(
            0,
            None,
            'b',
            None,
            0.0,
            baseline_metrics(vibrato, mean_f0, energy),
        )
    }

    fn correction(
        index: usize,
        parent: char,
        output: char,
        vibrato: (f64, f64),
        mean_f0: (f64, f64),
        energy: (f64, f64),
    ) -> IterationEvidenceV1 {
        iteration(
            index,
            Some(digest(parent)),
            output,
            Some(char::from_u32('c' as u32 + index as u32).unwrap()),
            0.5,
            baseline_metrics(vibrato, mean_f0, energy),
        )
    }

    #[test]
    fn baseline_already_inside_all_ranges_is_converged() {
        let report = evaluate_trajectory(
            &spec(),
            &[baseline((9.0, 11.0), (-1.0, 1.0), (-0.1, 0.1))],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Converged);
        assert_eq!(report.corrections_used, 0);
        assert_eq!(report.final_output_digest, digest('b'));
    }

    #[test]
    fn one_bounded_correction_can_converge() {
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (-0.1, 0.1)),
                correction(
                    1,
                    'b',
                    'd',
                    (9.0, 11.0),
                    (-1.0, 1.0),
                    (-0.2, 0.2),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Converged);
        assert_eq!(report.corrections_used, 1);
        assert_eq!(report.iterations[1].outcome, TrajectoryOutcome::Converged);
    }

    #[test]
    fn multiple_corrections_must_make_progress_on_at_least_one_target() {
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((16.0, 17.0), (5.0, 6.0), (-0.1, 0.1)),
                correction(
                    1,
                    'b',
                    'd',
                    (13.0, 14.0),
                    (3.0, 4.0),
                    (-0.1, 0.1),
                ),
                correction(
                    2,
                    'd',
                    'e',
                    (9.0, 11.0),
                    (-1.0, 1.0),
                    (-0.2, 0.2),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Converged);
        assert_eq!(report.corrections_used, 2);
    }

    #[test]
    fn protected_metric_violation_is_noncompliant_at_any_iteration() {
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (-0.1, 0.1)),
                correction(
                    1,
                    'b',
                    'd',
                    (9.0, 11.0),
                    (-1.0, 1.0),
                    (0.8, 0.9),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Noncompliant);
    }

    #[test]
    fn protected_metric_boundary_overlap_is_indeterminate() {
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (-0.1, 0.1)),
                correction(
                    1,
                    'b',
                    'd',
                    (9.0, 11.0),
                    (-1.0, 1.0),
                    (0.4, 0.7),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Indeterminate);
    }

    #[test]
    fn target_boundary_overlap_is_indeterminate_and_stops() {
        let report = evaluate_trajectory(
            &spec(),
            &[baseline((7.0, 9.0), (4.0, 5.0), (0.0, 0.0))],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Indeterminate);
    }

    #[test]
    fn target_regression_beyond_allowance_is_noncompliant() {
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)),
                correction(
                    1,
                    'b',
                    'd',
                    (18.0, 19.0),
                    (4.0, 5.0),
                    (0.0, 0.0),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Noncompliant);
    }

    #[test]
    fn previously_compliant_target_may_not_regress_outside_its_range() {
        let spec = spec();
        let report = evaluate_trajectory(
            &spec,
            &[
                baseline((14.0, 15.0), (0.0, 0.0), (0.0, 0.0)),
                correction(
                    1,
                    'b',
                    'd',
                    (9.0, 11.0),
                    (3.0, 4.0),
                    (0.0, 0.0),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Noncompliant);
    }

    #[test]
    fn no_progress_is_stalled() {
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)),
                correction(
                    1,
                    'b',
                    'd',
                    (14.0, 15.0),
                    (4.0, 5.0),
                    (0.0, 0.0),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Stalled);
    }

    #[test]
    fn identical_output_digest_is_stalled_even_when_analyzer_noise_looks_better() {
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)),
                correction(
                    1,
                    'b',
                    'b',
                    (9.0, 11.0),
                    (-1.0, 1.0),
                    (0.0, 0.0),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Stalled);
    }

    #[test]
    fn action_magnitude_above_declared_bound_is_noncompliant() {
        let mut step = correction(
            1,
            'b',
            'd',
            (9.0, 11.0),
            (-1.0, 1.0),
            (0.0, 0.0),
        );
        step.action_norm = 2.01;
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)),
                step,
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::Noncompliant);
    }

    #[test]
    fn correction_budget_exhaustion_is_not_convergence() {
        let mut spec = spec();
        spec.maximum_corrections = 1;
        let report = evaluate_trajectory(
            &spec,
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)),
                correction(
                    1,
                    'b',
                    'd',
                    (13.0, 14.0),
                    (3.0, 4.0),
                    (0.0, 0.0),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::IterationLimitReached);
        assert_eq!(
            report.iterations[1].outcome,
            TrajectoryOutcome::NeedsCorrection
        );
    }

    #[test]
    fn valid_partial_trajectory_can_request_another_correction() {
        let report = evaluate_trajectory(
            &spec(),
            &[
                baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)),
                correction(
                    1,
                    'b',
                    'd',
                    (13.0, 14.0),
                    (3.0, 4.0),
                    (0.0, 0.0),
                ),
            ],
        )
        .unwrap();
        assert_eq!(report.outcome, TrajectoryOutcome::NeedsCorrection);
    }

    #[test]
    fn iteration_indices_must_be_contiguous() {
        let step = correction(
            2,
            'b',
            'd',
            (9.0, 11.0),
            (-1.0, 1.0),
            (0.0, 0.0),
        );
        assert!(evaluate_trajectory(
            &spec(),
            &[baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)), step]
        )
        .is_err());
    }

    #[test]
    fn stale_parent_output_is_rejected() {
        let mut step = correction(
            1,
            'x',
            'd',
            (9.0, 11.0),
            (-1.0, 1.0),
            (0.0, 0.0),
        );
        step.parent_output_digest = Some(digest('x'));
        assert!(evaluate_trajectory(
            &spec(),
            &[baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)), step]
        )
        .is_err());
    }

    #[test]
    fn changed_frozen_context_is_rejected() {
        let mut step = correction(
            1,
            'b',
            'd',
            (9.0, 11.0),
            (-1.0, 1.0),
            (0.0, 0.0),
        );
        step.frozen_context_digest = digest('z');
        assert!(evaluate_trajectory(
            &spec(),
            &[baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0)), step]
        )
        .is_err());
    }

    #[test]
    fn mutation_after_convergence_is_rejected() {
        let first = baseline((9.0, 11.0), (-1.0, 1.0), (0.0, 0.0));
        let second = correction(
            1,
            'b',
            'd',
            (9.0, 11.0),
            (-1.0, 1.0),
            (0.0, 0.0),
        );
        assert!(evaluate_trajectory(&spec(), &[first, second]).is_err());
    }

    #[test]
    fn mutation_after_indeterminate_evidence_is_rejected() {
        let first = baseline((7.0, 9.0), (4.0, 5.0), (0.0, 0.0));
        let second = correction(
            1,
            'b',
            'd',
            (9.0, 11.0),
            (-1.0, 1.0),
            (0.0, 0.0),
        );
        assert!(evaluate_trajectory(&spec(), &[first, second]).is_err());
    }

    #[test]
    fn missing_duplicate_and_extra_metrics_are_rejected() {
        let mut missing = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        missing.measurements.pop();
        assert!(evaluate_trajectory(&spec(), &[missing]).is_err());

        let mut duplicate = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        duplicate.measurements.push(duplicate.measurements[0].clone());
        assert!(evaluate_trajectory(&spec(), &[duplicate]).is_err());

        let mut extra = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        extra.measurements.push(metric("unrequested", "dB", 0.0, 0.0));
        assert!(evaluate_trajectory(&spec(), &[extra]).is_err());
    }

    #[test]
    fn family_coverage_and_metric_units_must_match() {
        let mut bad_coverage = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        bad_coverage.measurements[0].familywise_coverage = 0.90;
        assert!(evaluate_trajectory(&spec(), &[bad_coverage]).is_err());

        let mut bad_unit = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        bad_unit.measurements[0].unit = "Hz".to_owned();
        assert!(evaluate_trajectory(&spec(), &[bad_unit]).is_err());

        let mut wrong_family = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        wrong_family.measurements[0].interval_family_id = "other-family".to_owned();
        assert!(evaluate_trajectory(&spec(), &[wrong_family]).is_err());
    }

    #[test]
    fn reversed_nonfinite_and_unlabelled_measurements_are_rejected() {
        let mut reversed = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        reversed.measurements[0].lower_delta = 15.0;
        reversed.measurements[0].upper_delta = 14.0;
        assert!(evaluate_trajectory(&spec(), &[reversed]).is_err());

        let mut nonfinite = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        nonfinite.measurements[0].upper_delta = f64::INFINITY;
        assert!(evaluate_trajectory(&spec(), &[nonfinite]).is_err());

        let mut unlabelled = baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0));
        unlabelled.measurements[0].uncertainty_method_id.clear();
        assert!(evaluate_trajectory(&spec(), &[unlabelled]).is_err());
    }

    #[test]
    fn invalid_digest_and_empty_trajectory_are_rejected() {
        let mut malformed = spec();
        malformed.frozen_context_digest = "not-a-digest".to_owned();
        assert!(evaluate_trajectory(
            &malformed,
            &[baseline((14.0, 15.0), (4.0, 5.0), (0.0, 0.0))]
        )
        .is_err());
        assert!(evaluate_trajectory(&spec(), &[]).is_err());
    }
}
