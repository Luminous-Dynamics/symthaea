//! Finite-distribution sensitivity bounds for causal effect transport.
//!
//! This is a deterministic discrete v0 of the outcome-shift sensitivity
//! model in Asiaee et al. (CLeaR 2026). For each treatment arm and measured
//! covariate stratum, the target conditional outcome distribution is allowed
//! to differ from the source trial distribution by a likelihood ratio r in
//! [1/Lambda, Lambda], subject to normalization. The sharp arm-mean bounds
//! follow from threshold reweighting of the ordered source outcomes.
//!
//! The target population covariate masses are fixed and observed. Potential
//! outcomes in the target population are evaluator-only. This module computes
//! identification bounds conditional on the registered Lambda; it does not
//! add finite-sample uncertainty for estimating source distributions, target
//! covariate masses, or sampling weights.

use crate::ContentDigest;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MASS_TOLERANCE: f64 = 1.0e-9;
const COVERAGE_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTrialOutcome {
    pub unit_id: u64,
    pub stratum_id: String,
    pub assigned_treatment: bool,
    /// Known treatment-assignment probability for this trial unit.
    pub propensity_treatment: f64,
    pub observed_outcome: f64,
}

impl SourceTrialOutcome {
    pub fn new(
        unit_id: u64,
        stratum_id: impl Into<String>,
        assigned_treatment: bool,
        propensity_treatment: f64,
        observed_outcome: f64,
    ) -> Self {
        Self {
            unit_id,
            stratum_id: stratum_id.into(),
            assigned_treatment,
            propensity_treatment,
            observed_outcome,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetStratumMass {
    pub stratum_id: String,
    /// Target-population mass P_target(X = stratum_id).
    pub target_mass: f64,
}

impl TargetStratumMass {
    pub fn new(stratum_id: impl Into<String>, target_mass: f64) -> Self {
        Self {
            stratum_id: stratum_id.into(),
            target_mass,
        }
    }
}

/// Agent-visible source trial data and target covariate distribution.
/// No target outcome or potential-outcome field appears here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportScenario {
    pub source_population_digest: ContentDigest,
    pub target_population_digest: ContentDigest,
    pub estimand_digest: ContentDigest,
    pub sensitivity_lambda: f64,
    pub target_strata: Vec<TargetStratumMass>,
    pub source_trial: Vec<SourceTrialOutcome>,
    pub decision_threshold: f64,
}

impl TransportScenario {
    pub fn new(
        source_population_digest: ContentDigest,
        target_population_digest: ContentDigest,
        estimand_digest: ContentDigest,
        sensitivity_lambda: f64,
        target_strata: Vec<TargetStratumMass>,
        source_trial: Vec<SourceTrialOutcome>,
        decision_threshold: f64,
    ) -> Self {
        Self {
            source_population_digest,
            target_population_digest,
            estimand_digest,
            sensitivity_lambda,
            target_strata,
            source_trial,
            decision_threshold,
        }
    }
}

/// Evaluator-only finite target-population potential outcomes.
/// population_weight is an absolute target-population mass and all row
/// weights together sum to one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPotentialOutcomeRow {
    pub unit_id: u64,
    pub stratum_id: String,
    pub population_weight: f64,
    pub outcome_control: f64,
    pub outcome_treatment: f64,
}

impl TargetPotentialOutcomeRow {
    pub fn new(
        unit_id: u64,
        stratum_id: impl Into<String>,
        population_weight: f64,
        outcome_control: f64,
        outcome_treatment: f64,
    ) -> Self {
        Self {
            unit_id,
            stratum_id: stratum_id.into(),
            population_weight,
            outcome_control,
            outcome_treatment,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransportOracle {
    pub source_population_digest: ContentDigest,
    pub target_population_digest: ContentDigest,
    pub estimand_digest: ContentDigest,
    pub target_rows: Vec<TargetPotentialOutcomeRow>,
}

impl TransportOracle {
    pub fn new(
        source_population_digest: ContentDigest,
        target_population_digest: ContentDigest,
        estimand_digest: ContentDigest,
        target_rows: Vec<TargetPotentialOutcomeRow>,
    ) -> Self {
        Self {
            source_population_digest,
            target_population_digest,
            estimand_digest,
            target_rows,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportEffectInterval {
    pub lower: f64,
    pub upper: f64,
}

impl TransportEffectInterval {
    pub fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }

    pub fn is_valid(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }

    pub fn contains(&self, value: f64) -> bool {
        self.is_valid()
            && self.lower - COVERAGE_TOLERANCE <= value
            && value <= self.upper + COVERAGE_TOLERANCE
    }

    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportDecision {
    Act,
    DoNotAct,
    Defer,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportClaim {
    pub reported_interval: TransportEffectInterval,
    pub decision: TransportDecision,
}

impl TransportClaim {
    pub fn new(reported_interval: TransportEffectInterval, decision: TransportDecision) -> Self {
        Self {
            reported_interval,
            decision,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportTrace {
    pub claim: TransportClaim,
}

impl TransportTrace {
    pub fn new(claim: TransportClaim) -> Self {
        Self { claim }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportQualification {
    Qualified,
    NotQualified,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportDecisionQuality {
    RobustWithinSensitivityEnvelope,
    SafeButConservative,
    Unsafe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportFindingCode {
    EmptyTargetStrata,
    EmptySourceTrial,
    EmptyStratumId,
    DuplicateTargetStratum,
    InvalidTargetMass,
    TargetMassDoesNotSumToOne,
    InvalidLambda,
    NonFiniteDecisionThreshold,
    EmptySourceUnitId,
    DuplicateSourceUnitId,
    InvalidTreatmentPropensity,
    NonFiniteSourceOutcome,
    EmptyTargetOracle,
    EmptyTargetUnitId,
    DuplicateTargetUnitId,
    UnknownTargetStratum,
    InvalidTargetWeight,
    NonFiniteTargetPotentialOutcome,
    TargetWeightDoesNotMatchStratumMass,
    SourcePopulationDigestMismatch,
    TargetPopulationDigestMismatch,
    EstimandDigestMismatch,
    MissingSourceArmSupport,
    InvalidReportedInterval,
    ReportedIntervalNarrowerThanSharpBound,
    UnsafeDecision,
    NonFiniteSharpBound,
    TargetDistributionOutsideSensitivityEnvelope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportFinding {
    pub code: TransportFindingCode,
    pub stratum_id: Option<String>,
    pub unit_id: Option<u64>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportArmBounds {
    pub source_weighted_mean: f64,
    pub sharp_lower_mean: f64,
    pub sharp_upper_mean: f64,
    pub source_support_size: usize,
    /// Evaluator-only density-ratio extrema for the actual target world.
    pub oracle_min_density_ratio: Option<f64>,
    pub oracle_max_density_ratio: Option<f64>,
    pub oracle_within_lambda: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportStratumReport {
    pub stratum_id: String,
    pub target_mass: f64,
    pub control: TransportArmBounds,
    pub treatment: TransportArmBounds,
    pub sharp_stratum_effect_lower: f64,
    pub sharp_stratum_effect_upper: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportReport {
    pub qualification: TransportQualification,
    pub source_population_digest: ContentDigest,
    pub target_population_digest: ContentDigest,
    pub estimand_digest: ContentDigest,
    pub sensitivity_lambda: f64,
    pub recomputed_target_effect_bounds: Option<TransportEffectInterval>,
    pub reported_target_effect_bounds: TransportEffectInterval,
    /// Evaluator-only target truth. Never expose this report to the agent.
    pub oracle_target_ate: Option<f64>,
    /// Whether the evaluator-only target outcome distributions satisfy the
    /// registered conditional likelihood-ratio sensitivity model.
    pub oracle_within_sensitivity_model: Option<bool>,
    /// Whether the exact finite target ATE lies inside the recomputed bounds.
    pub target_truth_covered: Option<bool>,
    pub recomputed_interval_width: Option<f64>,
    pub reported_interval_width: f64,
    pub decision_quality: Option<TransportDecisionQuality>,
    pub stratum_reports: Vec<TransportStratumReport>,
    pub findings: Vec<TransportFinding>,
}

impl TransportReport {
    pub fn is_qualified(&self) -> bool {
        self.qualification == TransportQualification::Qualified && self.findings.is_empty()
    }

    pub fn has_target_coverage(&self) -> bool {
        self.target_truth_covered == Some(true)
    }

    pub fn sensitivity_model_holds(&self) -> bool {
        self.oracle_within_sensitivity_model == Some(true)
    }
}

#[derive(Debug, Clone)]
struct WeightedOutcome {
    value: f64,
    probability: f64,
}

fn push_finding(
    findings: &mut Vec<TransportFinding>,
    code: TransportFindingCode,
    stratum_id: Option<String>,
    unit_id: Option<u64>,
    detail: impl Into<String>,
) {
    findings.push(TransportFinding {
        code,
        stratum_id,
        unit_id,
        detail: detail.into(),
    });
}

fn grouped_distribution(mut values: Vec<WeightedOutcome>) -> Vec<WeightedOutcome> {
    values.sort_by(|left, right| left.value.total_cmp(&right.value));
    let mut grouped: Vec<WeightedOutcome> = Vec::new();
    for value in values {
        if let Some(last) = grouped.last_mut() {
            if last.value == value.value {
                last.probability += value.probability;
                continue;
            }
        }
        grouped.push(value);
    }
    grouped
}

fn source_distribution(
    rows: &[&SourceTrialOutcome],
    treatment_arm: bool,
) -> Option<Vec<WeightedOutcome>> {
    let mut weighted = Vec::new();
    let mut total_weight = 0.0_f64;
    for row in rows {
        if row.assigned_treatment != treatment_arm {
            continue;
        }
        let propensity = if treatment_arm {
            row.propensity_treatment
        } else {
            1.0 - row.propensity_treatment
        };
        let weight = 1.0 / propensity;
        if !weight.is_finite() || weight <= 0.0 {
            return None;
        }
        weighted.push((row.observed_outcome, weight));
        total_weight += weight;
    }
    if weighted.is_empty() || !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }
    let values = weighted
        .into_iter()
        .map(|(value, weight)| WeightedOutcome {
            value,
            probability: weight / total_weight,
        })
        .collect();
    Some(grouped_distribution(values))
}

fn target_distribution(
    rows: &[&TargetPotentialOutcomeRow],
    stratum_mass: f64,
    treatment_arm: bool,
) -> Option<Vec<WeightedOutcome>> {
    let mut values = Vec::new();
    for row in rows {
        let value = if treatment_arm {
            row.outcome_treatment
        } else {
            row.outcome_control
        };
        values.push(WeightedOutcome {
            value,
            probability: row.population_weight / stratum_mass,
        });
    }
    let sum = values.iter().map(|item| item.probability).sum::<f64>();
    if !sum.is_finite() || (sum - 1.0).abs() > MASS_TOLERANCE {
        return None;
    }
    Some(grouped_distribution(values))
}

fn weighted_mean(distribution: &[WeightedOutcome]) -> f64 {
    distribution
        .iter()
        .map(|item| item.value * item.probability)
        .sum()
}

/// Computes the exact finite-distribution extrema under
/// Lambda^-1 <= r(y) <= Lambda and sum_y p(y) r(y) = 1.
///
/// Since the objective is linear and the constraints are box bounds plus one
/// normalization equality, the optimum is threshold-structured: assign the
/// larger likelihood ratio to the largest outcomes (upper mean) or smallest
/// outcomes (lower mean), with at most one partially allocated threshold atom.
fn sharp_shifted_mean(
    distribution: &[WeightedOutcome],
    lambda: f64,
    maximize: bool,
) -> Option<f64> {
    if distribution.is_empty() || !lambda.is_finite() || lambda < 1.0 {
        return None;
    }
    let lower_ratio = 1.0 / lambda;
    let upper_ratio = lambda;
    let mut remaining = 1.0 - lower_ratio;
    let mut order: Vec<usize> = (0..distribution.len()).collect();
    order.sort_by(|left, right| {
        distribution[*left]
            .value
            .total_cmp(&distribution[*right].value)
    });
    if maximize {
        order.reverse();
    }

    let mut ratios = vec![lower_ratio; distribution.len()];
    for index in order {
        if remaining <= MASS_TOLERANCE {
            break;
        }
        let probability = distribution[index].probability;
        if !probability.is_finite() || probability <= 0.0 {
            return None;
        }
        let capacity = probability * (upper_ratio - lower_ratio);
        let allocation = remaining.min(capacity);
        ratios[index] += allocation / probability;
        remaining -= allocation;
    }

    if remaining > 1.0e-8 {
        return None;
    }
    let expectation = distribution
        .iter()
        .zip(ratios.iter())
        .map(|(item, ratio)| item.probability * ratio * item.value)
        .sum::<f64>();
    expectation.is_finite().then_some(expectation)
}

fn density_ratio_extrema(
    source: &[WeightedOutcome],
    target: &[WeightedOutcome],
    lambda: f64,
) -> (bool, Option<f64>, Option<f64>) {
    if source.is_empty() || target.is_empty() {
        return (false, None, None);
    }
    let mut ratios = Vec::new();
    for source_item in source {
        let target_probability = target
            .iter()
            .find(|target_item| target_item.value == source_item.value)
            .map_or(0.0, |target_item| target_item.probability);
        if source_item.probability <= 0.0 {
            return (false, None, None);
        }
        ratios.push(target_probability / source_item.probability);
    }
    for target_item in target {
        if !source.iter().any(|source_item| source_item.value == target_item.value) {
            return (false, None, None);
        }
    }
    let minimum = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = ratios.iter().copied().fold(0.0_f64, f64::max);
    let lower = 1.0 / lambda;
    let compatible = minimum + COVERAGE_TOLERANCE >= lower
        && maximum <= lambda + COVERAGE_TOLERANCE
        && minimum.is_finite()
        && maximum.is_finite();
    (compatible, Some(minimum), Some(maximum))
}

fn empty_report(
    scenario: &TransportScenario,
    trace: &TransportTrace,
    qualification: TransportQualification,
    findings: Vec<TransportFinding>,
) -> TransportReport {
    TransportReport {
        qualification,
        source_population_digest: scenario.source_population_digest,
        target_population_digest: scenario.target_population_digest,
        estimand_digest: scenario.estimand_digest,
        sensitivity_lambda: scenario.sensitivity_lambda,
        recomputed_target_effect_bounds: None,
        reported_target_effect_bounds: trace.claim.reported_interval,
        oracle_target_ate: None,
        oracle_within_sensitivity_model: None,
        target_truth_covered: None,
        recomputed_interval_width: None,
        reported_interval_width: trace.claim.reported_interval.width(),
        decision_quality: None,
        stratum_reports: Vec::new(),
        findings,
    }
}

fn validate_decision(
    decision: TransportDecision,
    interval: TransportEffectInterval,
    threshold: f64,
) -> TransportDecisionQuality {
    if decision == TransportDecision::Defer {
        if interval.lower > threshold || interval.upper <= threshold {
            return TransportDecisionQuality::SafeButConservative;
        }
        return TransportDecisionQuality::RobustWithinSensitivityEnvelope;
    }
    match decision {
        TransportDecision::Act if interval.lower > threshold => {
            TransportDecisionQuality::RobustWithinSensitivityEnvelope
        }
        TransportDecision::DoNotAct if interval.upper <= threshold => {
            TransportDecisionQuality::RobustWithinSensitivityEnvelope
        }
        _ => TransportDecisionQuality::Unsafe,
    }
}

/// Evaluate a finite-distribution Lambda sensitivity envelope for a
/// randomized source trial and an evaluator-only target population.
///
/// Source outcome distributions are normalized from inverse assignment
/// propensity weights within each measured stratum and treatment arm. Target
/// stratum masses are supplied separately. The returned sharp interval is
/// conditional on those finite empirical distributions; it is not a
/// confidence interval for their sampling error.
pub fn evaluate_transport_bounds(
    scenario: &TransportScenario,
    oracle: &TransportOracle,
    trace: &TransportTrace,
) -> TransportReport {
    let mut findings = Vec::new();
    if scenario.target_strata.is_empty() {
        push_finding(
            &mut findings,
            TransportFindingCode::EmptyTargetStrata,
            None,
            None,
            "target population must contain at least one measured stratum",
        );
    }
    if scenario.source_trial.is_empty() {
        push_finding(
            &mut findings,
            TransportFindingCode::EmptySourceTrial,
            None,
            None,
            "source randomized trial contains no observed units",
        );
    }
    if oracle.target_rows.is_empty() {
        push_finding(
            &mut findings,
            TransportFindingCode::EmptyTargetOracle,
            None,
            None,
            "evaluator-only target population contains no units",
        );
    }
    if !scenario.sensitivity_lambda.is_finite() || scenario.sensitivity_lambda < 1.0 {
        push_finding(
            &mut findings,
            TransportFindingCode::InvalidLambda,
            None,
            None,
            "sensitivity Lambda must be finite and at least one",
        );
    }
    if !scenario.decision_threshold.is_finite() {
        push_finding(
            &mut findings,
            TransportFindingCode::NonFiniteDecisionThreshold,
            None,
            None,
            "decision threshold must be finite",
        );
    }
    if oracle.source_population_digest != scenario.source_population_digest {
        push_finding(
            &mut findings,
            TransportFindingCode::SourcePopulationDigestMismatch,
            None,
            None,
            "oracle source-population digest differs from the agent-visible scenario identity",
        );
    }
    if oracle.target_population_digest != scenario.target_population_digest {
        push_finding(
            &mut findings,
            TransportFindingCode::TargetPopulationDigestMismatch,
            None,
            None,
            "oracle target-population digest differs from the agent-visible scenario identity",
        );
    }
    if oracle.estimand_digest != scenario.estimand_digest {
        push_finding(
            &mut findings,
            TransportFindingCode::EstimandDigestMismatch,
            None,
            None,
            "oracle estimand digest differs from the frozen scenario estimand",
        );
    }
    if !trace.claim.reported_interval.is_valid() {
        push_finding(
            &mut findings,
            TransportFindingCode::InvalidReportedInterval,
            None,
            None,
            "reported target-effect interval is invalid",
        );
    }

    let mut target_masses = BTreeMap::new();
    let mut target_mass_sum = 0.0_f64;
    for stratum in &scenario.target_strata {
        if stratum.stratum_id.trim().is_empty() {
            push_finding(
                &mut findings,
                TransportFindingCode::EmptyStratumId,
                Some(stratum.stratum_id.clone()),
                None,
                "target stratum ID is empty",
            );
        }
        if !stratum.target_mass.is_finite() || stratum.target_mass <= 0.0 {
            push_finding(
                &mut findings,
                TransportFindingCode::InvalidTargetMass,
                Some(stratum.stratum_id.clone()),
                None,
                "target stratum mass must be finite and positive",
            );
        }
        if target_masses
            .insert(stratum.stratum_id.clone(), stratum.target_mass)
            .is_some()
        {
            push_finding(
                &mut findings,
                TransportFindingCode::DuplicateTargetStratum,
                Some(stratum.stratum_id.clone()),
                None,
                "target stratum ID is duplicated",
            );
        }
        target_mass_sum += stratum.target_mass;
    }
    if target_mass_sum.is_finite() && (target_mass_sum - 1.0).abs() > MASS_TOLERANCE {
        push_finding(
            &mut findings,
            TransportFindingCode::TargetMassDoesNotSumToOne,
            None,
            None,
            "target stratum masses must sum to one",
        );
    }

    let mut source_ids = BTreeSet::new();
    for row in &scenario.source_trial {
        if row.unit_id == 0 {
            push_finding(
                &mut findings,
                TransportFindingCode::EmptySourceUnitId,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "source trial unit ID zero is reserved",
            );
        }
        if !source_ids.insert(row.unit_id) {
            push_finding(
                &mut findings,
                TransportFindingCode::DuplicateSourceUnitId,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "source trial unit ID is duplicated",
            );
        }
        if row.stratum_id.trim().is_empty() {
            push_finding(
                &mut findings,
                TransportFindingCode::EmptyStratumId,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "source trial stratum ID is empty",
            );
        }
        if !row.propensity_treatment.is_finite()
            || row.propensity_treatment <= 0.0
            || row.propensity_treatment >= 1.0
        {
            push_finding(
                &mut findings,
                TransportFindingCode::InvalidTreatmentPropensity,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "source assignment propensity must be strictly between zero and one",
            );
        }
        if !row.observed_outcome.is_finite() {
            push_finding(
                &mut findings,
                TransportFindingCode::NonFiniteSourceOutcome,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "source observed outcome must be finite",
            );
        }
    }

    let mut target_ids = BTreeSet::new();
    let mut target_mass_by_stratum: BTreeMap<String, f64> = BTreeMap::new();
    let mut target_rows_by_stratum: BTreeMap<String, Vec<&TargetPotentialOutcomeRow>> =
        BTreeMap::new();
    for row in &oracle.target_rows {
        if row.unit_id == 0 {
            push_finding(
                &mut findings,
                TransportFindingCode::EmptyTargetUnitId,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "target oracle unit ID zero is reserved",
            );
        }
        if !target_ids.insert(row.unit_id) {
            push_finding(
                &mut findings,
                TransportFindingCode::DuplicateTargetUnitId,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "target oracle unit ID is duplicated",
            );
        }
        if !target_masses.contains_key(&row.stratum_id) {
            push_finding(
                &mut findings,
                TransportFindingCode::UnknownTargetStratum,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "target oracle row references an undeclared target stratum",
            );
        }
        if !row.population_weight.is_finite() || row.population_weight <= 0.0 {
            push_finding(
                &mut findings,
                TransportFindingCode::InvalidTargetWeight,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "target population row weight must be finite and positive",
            );
        }
        if !row.outcome_control.is_finite() || !row.outcome_treatment.is_finite() {
            push_finding(
                &mut findings,
                TransportFindingCode::NonFiniteTargetPotentialOutcome,
                Some(row.stratum_id.clone()),
                Some(row.unit_id),
                "target potential outcomes must be finite",
            );
        }
        *target_mass_by_stratum
            .entry(row.stratum_id.clone())
            .or_default() += row.population_weight;
        target_rows_by_stratum
            .entry(row.stratum_id.clone())
            .or_default()
            .push(row);
    }
    for (stratum_id, expected_mass) in &target_masses {
        let observed_mass = target_mass_by_stratum.get(stratum_id).copied().unwrap_or(0.0);
        if (observed_mass - expected_mass).abs() > MASS_TOLERANCE {
            push_finding(
                &mut findings,
                TransportFindingCode::TargetWeightDoesNotMatchStratumMass,
                Some(stratum_id.clone()),
                None,
                "evaluator target row weights do not match the declared target stratum mass",
            );
        }
    }

    if !findings.is_empty() {
        return empty_report(
            scenario,
            trace,
            TransportQualification::NotQualified,
            findings,
        );
    }

    let mut stratum_reports = Vec::new();
    let mut target_effect_lower = 0.0_f64;
    let mut target_effect_upper = 0.0_f64;
    let target_ate = oracle
        .target_rows
        .iter()
        .map(|row| row.population_weight * (row.outcome_treatment - row.outcome_control))
        .sum::<f64>();
    if !target_ate.is_finite() {
        push_finding(
            &mut findings,
            TransportFindingCode::NonFiniteTargetPotentialOutcome,
            None,
            None,
            "evaluator target ATE is non-finite",
        );
        return empty_report(
            scenario,
            trace,
            TransportQualification::NotQualified,
            findings,
        );
    }
    let mut all_strata_supported = true;
    let mut all_density_ratios_compatible = true;

    for stratum in &scenario.target_strata {
        let source_rows: Vec<&SourceTrialOutcome> = scenario
            .source_trial
            .iter()
            .filter(|row| row.stratum_id == stratum.stratum_id)
            .collect();
        let target_rows = target_rows_by_stratum
            .get(&stratum.stratum_id)
            .cloned()
            .unwrap_or_default();

        let Some(source_control) = source_distribution(&source_rows, false) else {
            all_strata_supported = false;
            push_finding(
                &mut findings,
                TransportFindingCode::MissingSourceArmSupport,
                Some(stratum.stratum_id.clone()),
                None,
                "source trial has no usable control-arm observations in this target stratum",
            );
            continue;
        };
        let Some(source_treatment) = source_distribution(&source_rows, true) else {
            all_strata_supported = false;
            push_finding(
                &mut findings,
                TransportFindingCode::MissingSourceArmSupport,
                Some(stratum.stratum_id.clone()),
                None,
                "source trial has no usable treatment-arm observations in this target stratum",
            );
            continue;
        };
        let Some(target_control) = target_distribution(&target_rows, stratum.target_mass, false)
        else {
            all_strata_supported = false;
            push_finding(
                &mut findings,
                TransportFindingCode::InvalidTargetWeight,
                Some(stratum.stratum_id.clone()),
                None,
                "target control outcome distribution could not be normalized",
            );
            continue;
        };
        let Some(target_treatment) =
            target_distribution(&target_rows, stratum.target_mass, true)
        else {
            all_strata_supported = false;
            push_finding(
                &mut findings,
                TransportFindingCode::InvalidTargetWeight,
                Some(stratum.stratum_id.clone()),
                None,
                "target treatment outcome distribution could not be normalized",
            );
            continue;
        };

        let lambda = scenario.sensitivity_lambda;
        let Some(control_lower) = sharp_shifted_mean(&source_control, lambda, false) else {
            push_finding(
                &mut findings,
                TransportFindingCode::NonFiniteSharpBound,
                Some(stratum.stratum_id.clone()),
                None,
                "control lower sensitivity bound is not finite",
            );
            continue;
        };
        let Some(control_upper) = sharp_shifted_mean(&source_control, lambda, true) else {
            push_finding(
                &mut findings,
                TransportFindingCode::NonFiniteSharpBound,
                Some(stratum.stratum_id.clone()),
                None,
                "control upper sensitivity bound is not finite",
            );
            continue;
        };
        let Some(treatment_lower) = sharp_shifted_mean(&source_treatment, lambda, false)
        else {
            push_finding(
                &mut findings,
                TransportFindingCode::NonFiniteSharpBound,
                Some(stratum.stratum_id.clone()),
                None,
                "treatment lower sensitivity bound is not finite",
            );
            continue;
        };
        let Some(treatment_upper) = sharp_shifted_mean(&source_treatment, lambda, true)
        else {
            push_finding(
                &mut findings,
                TransportFindingCode::NonFiniteSharpBound,
                Some(stratum.stratum_id.clone()),
                None,
                "treatment upper sensitivity bound is not finite",
            );
            continue;
        };

        let (control_compatible, control_min_ratio, control_max_ratio) =
            density_ratio_extrema(&source_control, &target_control, lambda);
        let (treatment_compatible, treatment_min_ratio, treatment_max_ratio) =
            density_ratio_extrema(&source_treatment, &target_treatment, lambda);
        let stratum_compatible = control_compatible && treatment_compatible;
        all_density_ratios_compatible &= stratum_compatible;

        let stratum_lower = treatment_lower - control_upper;
        let stratum_upper = treatment_upper - control_lower;
        target_effect_lower += stratum.target_mass * stratum_lower;
        target_effect_upper += stratum.target_mass * stratum_upper;

        stratum_reports.push(TransportStratumReport {
            stratum_id: stratum.stratum_id.clone(),
            target_mass: stratum.target_mass,
            control: TransportArmBounds {
                source_weighted_mean: weighted_mean(&source_control),
                sharp_lower_mean: control_lower,
                sharp_upper_mean: control_upper,
                source_support_size: source_control.len(),
                oracle_min_density_ratio: control_min_ratio,
                oracle_max_density_ratio: control_max_ratio,
                oracle_within_lambda: control_compatible,
            },
            treatment: TransportArmBounds {
                source_weighted_mean: weighted_mean(&source_treatment),
                sharp_lower_mean: treatment_lower,
                sharp_upper_mean: treatment_upper,
                source_support_size: source_treatment.len(),
                oracle_min_density_ratio: treatment_min_ratio,
                oracle_max_density_ratio: treatment_max_ratio,
                oracle_within_lambda: treatment_compatible,
            },
            sharp_stratum_effect_lower: stratum_lower,
            sharp_stratum_effect_upper: stratum_upper,
        });
    }

    if !all_strata_supported || stratum_reports.len() != scenario.target_strata.len() {
        return TransportReport {
            qualification: TransportQualification::Inconclusive,
            source_population_digest: scenario.source_population_digest,
            target_population_digest: scenario.target_population_digest,
            estimand_digest: scenario.estimand_digest,
            sensitivity_lambda: scenario.sensitivity_lambda,
            recomputed_target_effect_bounds: None,
            reported_target_effect_bounds: trace.claim.reported_interval,
            oracle_target_ate: Some(target_ate),
            oracle_within_sensitivity_model: None,
            target_truth_covered: None,
            recomputed_interval_width: None,
            reported_interval_width: trace.claim.reported_interval.width(),
            decision_quality: None,
            stratum_reports,
            findings,
        };
    }

    let recomputed = TransportEffectInterval::new(target_effect_lower, target_effect_upper);
    if !recomputed.is_valid() {
        push_finding(
            &mut findings,
            TransportFindingCode::NonFiniteSharpBound,
            None,
            None,
            "aggregate target sensitivity interval is invalid",
        );
        return empty_report(
            scenario,
            trace,
            TransportQualification::NotQualified,
            findings,
        );
    }

    if !all_density_ratios_compatible {
        push_finding(
            &mut findings,
            TransportFindingCode::TargetDistributionOutsideSensitivityEnvelope,
            None,
            None,
            "the evaluator-only target outcome distribution violates the registered conditional likelihood-ratio sensitivity envelope",
        );
    }

    if trace.claim.reported_interval.lower > recomputed.lower + COVERAGE_TOLERANCE
        || trace.claim.reported_interval.upper < recomputed.upper - COVERAGE_TOLERANCE
    {
        push_finding(
            &mut findings,
            TransportFindingCode::ReportedIntervalNarrowerThanSharpBound,
            None,
            None,
            "reported target interval is narrower than the evaluator-recomputed sharp sensitivity envelope",
        );
    }

    let decision_quality = validate_decision(
        trace.claim.decision,
        trace.claim.reported_interval,
        scenario.decision_threshold,
    );
    if decision_quality == TransportDecisionQuality::Unsafe {
        push_finding(
            &mut findings,
            TransportFindingCode::UnsafeDecision,
            None,
            None,
            "agent took a decisive target action while the reported interval crosses the decision threshold",
        );
    }

    let reported_contains_bound = trace.claim.reported_interval.lower
        <= recomputed.lower + COVERAGE_TOLERANCE
        && trace.claim.reported_interval.upper >= recomputed.upper - COVERAGE_TOLERANCE;
    let qualification = if findings.is_empty() {
        TransportQualification::Qualified
    } else {
        TransportQualification::NotQualified
    };

    TransportReport {
        qualification,
        source_population_digest: scenario.source_population_digest,
        target_population_digest: scenario.target_population_digest,
        estimand_digest: scenario.estimand_digest,
        sensitivity_lambda: scenario.sensitivity_lambda,
        recomputed_target_effect_bounds: Some(recomputed),
        reported_target_effect_bounds: trace.claim.reported_interval,
        oracle_target_ate: Some(target_ate),
        oracle_within_sensitivity_model: Some(all_density_ratios_compatible),
        target_truth_covered: Some(recomputed.contains(target_ate)),
        recomputed_interval_width: Some(recomputed.width()),
        reported_interval_width: trace.claim.reported_interval.width(),
        decision_quality: Some(if !reported_contains_bound {
            TransportDecisionQuality::Unsafe
        } else {
            decision_quality
        }),
        stratum_reports,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (TransportScenario, TransportOracle, TransportTrace) {
        let source_digest = ContentDigest::of(b"source-rct-v1");
        let target_digest = ContentDigest::of(b"target-pop-v1");
        let estimand_digest = ContentDigest::of(b"target-ate-v1");
        let source_rows = vec![
            SourceTrialOutcome::new(1, "x0", false, 0.5, 0.0),
            SourceTrialOutcome::new(2, "x0", false, 0.5, 10.0),
            SourceTrialOutcome::new(3, "x0", true, 0.5, 0.0),
            SourceTrialOutcome::new(4, "x0", true, 0.5, 10.0),
        ];
        let scenario = TransportScenario::new(
            source_digest,
            target_digest,
            estimand_digest,
            2.0,
            vec![TargetStratumMass::new("x0", 1.0)],
            source_rows,
            0.0,
        );
        let target_rows = vec![
            TargetPotentialOutcomeRow::new(101, "x0", 0.25, 0.0, 0.0),
            TargetPotentialOutcomeRow::new(102, "x0", 0.50, 0.0, 10.0),
            TargetPotentialOutcomeRow::new(103, "x0", 0.25, 10.0, 10.0),
        ];
        let oracle = TransportOracle::new(
            source_digest,
            target_digest,
            estimand_digest,
            target_rows,
        );
        let trace = TransportTrace::new(TransportClaim::new(
            TransportEffectInterval::new(-5.0, 5.0),
            TransportDecision::Defer,
        ));
        (scenario, oracle, trace)
    }

    #[test]
    fn threshold_reweighting_matches_exhaustive_finite_vertex_enumeration() {
        let distribution = vec![
            WeightedOutcome {
                value: 0.0,
                probability: 0.25,
            },
            WeightedOutcome {
                value: 5.0,
                probability: 0.50,
            },
            WeightedOutcome {
                value: 10.0,
                probability: 0.25,
            },
        ];
        let lambda = 2.0;
        let actual_lower = sharp_shifted_mean(&distribution, lambda, false).unwrap();
        let actual_upper = sharp_shifted_mean(&distribution, lambda, true).unwrap();

        let lower_ratio = 1.0 / lambda;
        let upper_ratio = lambda;
        let mut oracle_lower = f64::INFINITY;
        let mut oracle_upper = f64::NEG_INFINITY;
        // At a vertex of this box-constrained simplex, at most one
        // likelihood ratio is interior. Enumerate endpoint pairs and solve
        // the third ratio from sum_i p_i r_i = 1.
        for first in [lower_ratio, upper_ratio] {
            for second in [lower_ratio, upper_ratio] {
                let third =
                    (1.0 - 0.25 * first - 0.50 * second) / 0.25;
                if third < lower_ratio - MASS_TOLERANCE
                    || third > upper_ratio + MASS_TOLERANCE
                {
                    continue;
                }
                let mean = 0.25 * first * 0.0
                    + 0.50 * second * 5.0
                    + 0.25 * third * 10.0;
                oracle_lower = oracle_lower.min(mean);
                oracle_upper = oracle_upper.max(mean);
            }
        }
        for first in [lower_ratio, upper_ratio] {
            for third in [lower_ratio, upper_ratio] {
                let second =
                    (1.0 - 0.25 * first - 0.25 * third) / 0.50;
                if second < lower_ratio - MASS_TOLERANCE
                    || second > upper_ratio + MASS_TOLERANCE
                {
                    continue;
                }
                let mean = 0.25 * first * 0.0
                    + 0.50 * second * 5.0
                    + 0.25 * third * 10.0;
                oracle_lower = oracle_lower.min(mean);
                oracle_upper = oracle_upper.max(mean);
            }
        }
        for second in [lower_ratio, upper_ratio] {
            for third in [lower_ratio, upper_ratio] {
                let first = (1.0 - 0.50 * second - 0.25 * third) / 0.25;
                if first < lower_ratio - MASS_TOLERANCE
                    || first > upper_ratio + MASS_TOLERANCE
                {
                    continue;
                }
                let mean = 0.25 * first * 0.0
                    + 0.50 * second * 5.0
                    + 0.25 * third * 10.0;
                oracle_lower = oracle_lower.min(mean);
                oracle_upper = oracle_upper.max(mean);
            }
        }
        assert!((actual_lower - oracle_lower).abs() < 1.0e-10);
        assert!((actual_upper - oracle_upper).abs() < 1.0e-10);
    }

    #[test]
    fn lambda_two_produces_sharp_threshold_bounds_and_covers_target() {
        let (scenario, oracle, trace) = fixture();
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            TransportQualification::Qualified,
            "{:?}",
            report.findings
        );
        assert_eq!(
            report.recomputed_target_effect_bounds,
            Some(TransportEffectInterval::new(-5.0, 5.0))
        );
        assert_eq!(report.oracle_target_ate, Some(5.0));
        assert_eq!(report.oracle_within_sensitivity_model, Some(true));
        assert_eq!(report.target_truth_covered, Some(true));
        assert_eq!(
            report.decision_quality,
            Some(TransportDecisionQuality::RobustWithinSensitivityEnvelope)
        );
        assert!(report.is_qualified());
    }

    #[test]
    fn lambda_one_recovers_source_distribution_when_target_matches_source() {
        let (mut scenario, mut oracle, mut trace) = fixture();
        scenario.sensitivity_lambda = 1.0;
        oracle.target_rows = vec![
            TargetPotentialOutcomeRow::new(101, "x0", 0.5, 0.0, 0.0),
            TargetPotentialOutcomeRow::new(102, "x0", 0.5, 10.0, 10.0),
        ];
        trace.claim.reported_interval = TransportEffectInterval::new(0.0, 0.0);
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            TransportQualification::Qualified,
            "{:?}",
            report.findings
        );
        assert_eq!(
            report.recomputed_target_effect_bounds,
            Some(TransportEffectInterval::new(0.0, 0.0))
        );
        assert_eq!(report.oracle_within_sensitivity_model, Some(true));
        assert_eq!(report.target_truth_covered, Some(true));
    }

    #[test]
    fn larger_lambda_weakly_widens_the_identified_set() {
        let (mut scenario, oracle, trace) = fixture();
        let report_two = evaluate_transport_bounds(&scenario, &oracle, &trace);
        scenario.sensitivity_lambda = 3.0;
        let report_three = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert!(
            report_three.recomputed_interval_width.unwrap()
                >= report_two.recomputed_interval_width.unwrap()
        );
    }

    #[test]
    fn target_outside_declared_lambda_is_reported_as_sensitivity_miss() {
        let (mut scenario, mut oracle, mut trace) = fixture();
        scenario.sensitivity_lambda = 1.0;
        oracle.target_rows = vec![
            TargetPotentialOutcomeRow::new(101, "x0", 0.25, 0.0, 0.0),
            TargetPotentialOutcomeRow::new(102, "x0", 0.50, 0.0, 10.0),
            TargetPotentialOutcomeRow::new(103, "x0", 0.25, 10.0, 10.0),
        ];
        trace.claim.reported_interval = TransportEffectInterval::new(0.0, 0.0);
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            TransportQualification::NotQualified,
            "{:?}",
            report.findings
        );
        assert_eq!(report.oracle_within_sensitivity_model, Some(false));
        assert_eq!(report.target_truth_covered, Some(false));
        assert!(report.findings.iter().any(|finding| {
            finding.code == TransportFindingCode::TargetDistributionOutsideSensitivityEnvelope
        }));
    }

    #[test]
    fn target_only_stratum_without_source_support_is_inconclusive() {
        let (mut scenario, mut oracle, trace) = fixture();
        scenario.target_strata.push(TargetStratumMass::new("x1", 0.5));
        scenario.target_strata[0].target_mass = 0.5;
        for row in &mut oracle.target_rows {
            row.population_weight *= 0.5;
        }
        oracle.target_rows.push(TargetPotentialOutcomeRow::new(
            104, "x1", 0.5, 0.0, 1.0,
        ));
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, TransportQualification::Inconclusive);
        assert_eq!(report.oracle_target_ate, Some(3.0));
        assert_eq!(report.oracle_within_sensitivity_model, None);
        assert!(report.findings.iter().any(|finding| {
            finding.code == TransportFindingCode::MissingSourceArmSupport
        }));
    }

    #[test]
    fn invalid_assignment_propensity_blocks_transport_bounds() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.source_trial[0].propensity_treatment = 0.0;
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            TransportQualification::NotQualified
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == TransportFindingCode::InvalidTreatmentPropensity
        }));
    }

    #[test]
    fn a_reported_interval_may_not_be_narrower_than_the_sharp_envelope() {
        let (scenario, oracle, mut trace) = fixture();
        trace.claim.reported_interval = TransportEffectInterval::new(4.0, 5.0);
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            TransportQualification::NotQualified
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == TransportFindingCode::ReportedIntervalNarrowerThanSharpBound
        }));
    }

    #[test]
    fn decisive_action_is_rejected_when_sensitivity_interval_crosses_threshold() {
        let (scenario, oracle, mut trace) = fixture();
        trace.claim.decision = TransportDecision::Act;
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            TransportQualification::NotQualified
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == TransportFindingCode::UnsafeDecision
        }));
    }

    #[test]
    fn population_and_estimand_identity_must_match_the_evaluator_oracle() {
        let (scenario, mut oracle, trace) = fixture();
        oracle.target_population_digest = ContentDigest::of(b"different-target");
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, TransportQualification::NotQualified);
        assert!(report.findings.iter().any(|finding| {
            finding.code == TransportFindingCode::TargetPopulationDigestMismatch
        }));
    }

    #[test]
    fn negative_outcome_support_is_handled_by_threshold_ordering() {
        let (mut scenario, mut oracle, mut trace) = fixture();
        scenario.source_trial[0].observed_outcome = -10.0;
        scenario.source_trial[1].observed_outcome = 10.0;
        scenario.source_trial[2].observed_outcome = -10.0;
        scenario.source_trial[3].observed_outcome = 10.0;
        oracle.target_rows = vec![
            TargetPotentialOutcomeRow::new(101, "x0", 0.5, -10.0, -10.0),
            TargetPotentialOutcomeRow::new(102, "x0", 0.5, 10.0, 10.0),
        ];
        trace.claim.reported_interval = TransportEffectInterval::new(-10.0, 10.0);
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, TransportQualification::Qualified, "{:?}", report.findings);
        assert_eq!(report.oracle_within_sensitivity_model, Some(true));
        let bounds = report.recomputed_target_effect_bounds.unwrap();
        assert!(bounds.lower <= bounds.upper);
        assert!(bounds.lower.is_finite() && bounds.upper.is_finite());
    }

    #[test]
    fn target_stratum_masses_must_sum_to_one() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.target_strata[0].target_mass = 0.75;
        let report = evaluate_transport_bounds(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            TransportQualification::NotQualified
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == TransportFindingCode::TargetMassDoesNotSumToOne
        }));
    }
}
