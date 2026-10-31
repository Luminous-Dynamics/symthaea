//! Pareto-native, evidence-bound evaluation for multi-objective search.
//!
//! Objective dimensions remain separate: this module does not collapse them into an
//! arbitrary weighted score. Every objective is transformed into a maximization utility
//! using frozen scale anchors; feasibility is a separate gate; nondominated candidates
//! remain visible; and exact hypervolume is reported only against an explicit reference
//! point. Validated plans support exact hypervolume for two through four dimensions.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "symthaea.architecture-engineering.pareto-objectives";
const VERSION: &str = "0.1.0";
const MIN_DIMENSIONS: usize = 2;
const MAX_EXACT_HYPERVOLUME_DIMENSIONS: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoError {
    pub path: String,
    pub message: String,
}

impl ParetoError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ParetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for ParetoError {}

/// Raw objective direction. All internal comparisons operate on maximization utilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

/// Frozen objective definition and scale anchors.
/// The anchors define units only; measured values are not clipped to the anchor interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoObjectiveSpec {
    pub objective_id: String,
    pub direction: ObjectiveDirection,
    pub scale_lower: f64,
    pub scale_upper: f64,
}

/// One candidate's raw objective vector and independent feasibility decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoCandidate {
    pub candidate_id: String,
    /// Raw values in exactly the same order as objective_specs.
    pub objective_values: Vec<f64>,
    pub feasible: bool,
    /// Empty for feasible candidates; non-empty, explicit reasons for infeasible candidates.
    pub constraint_violations: Vec<String>,
    /// Digest of the independently retained evaluation/evidence artifact.
    pub evidence_sha256: String,
}

/// Immutable analysis definition and complete candidate set for one Pareto comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoAnalysisPlan {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    /// Declared dimension order is semantically significant and is never silently reordered.
    pub objective_specs: Vec<ParetoObjectiveSpec>,
    /// Hypervolume reference point expressed in transformed maximization-utility coordinates.
    pub reference_point_utility: Vec<f64>,
    /// Canonically sorted by candidate_id.
    pub candidates: Vec<ParetoCandidate>,
}

impl ParetoAnalysisPlan {
    pub fn new(
        campaign_id: impl Into<String>,
        objective_specs: Vec<ParetoObjectiveSpec>,
        reference_point_utility: Vec<f64>,
        mut candidates: Vec<ParetoCandidate>,
    ) -> Result<Self, ParetoError> {
        candidates.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
        let plan = Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            campaign_id: campaign_id.into(),
            objective_specs,
            reference_point_utility,
            candidates,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), ParetoError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(ParetoError::new(
                "plan.schema",
                "unsupported Pareto analysis schema/version",
            ));
        }
        if self.campaign_id.trim().is_empty() {
            return Err(ParetoError::new("plan.campaign_id", "campaign ID must be non-empty"));
        }
        let dimensions = self.objective_specs.len();
        if !(MIN_DIMENSIONS..=MAX_EXACT_HYPERVOLUME_DIMENSIONS).contains(&dimensions) {
            return Err(ParetoError::new(
                "plan.objective_specs",
                format!(
                    "exact hypervolume currently requires {MIN_DIMENSIONS}..={MAX_EXACT_HYPERVOLUME_DIMENSIONS} objectives"
                ),
            ));
        }
        let mut objective_ids = BTreeSet::new();
        for (index, objective) in self.objective_specs.iter().enumerate() {
            if objective.objective_id.trim().is_empty() {
                return Err(ParetoError::new(
                    format!("plan.objective_specs[{index}].objective_id"),
                    "objective ID must be non-empty",
                ));
            }
            if !objective_ids.insert(objective.objective_id.as_str()) {
                return Err(ParetoError::new(
                    format!("plan.objective_specs[{index}].objective_id"),
                    "objective IDs must be unique",
                ));
            }
            if !objective.scale_lower.is_finite()
                || !objective.scale_upper.is_finite()
                || objective.scale_lower >= objective.scale_upper
                || !(objective.scale_upper - objective.scale_lower).is_finite()
            {
                return Err(ParetoError::new(
                    format!("plan.objective_specs[{index}].scale"),
                    "scale anchors must be finite and lower must be less than upper",
                ));
            }
        }
        if self.reference_point_utility.len() != dimensions {
            return Err(ParetoError::new(
                "plan.reference_point_utility",
                "reference point dimension differs from the objective count",
            ));
        }
        for (index, value) in self.reference_point_utility.iter().enumerate() {
            if !value.is_finite() {
                return Err(ParetoError::new(
                    format!("plan.reference_point_utility[{index}]"),
                    "reference point coordinates must be finite",
                ));
            }
        }
        if self.candidates.is_empty() {
            return Err(ParetoError::new("plan.candidates", "at least one candidate is required"));
        }
        let mut candidate_ids = BTreeSet::new();
        let mut previous_id: Option<&str> = None;
        for (index, candidate) in self.candidates.iter().enumerate() {
            if candidate.candidate_id.trim().is_empty() {
                return Err(ParetoError::new(
                    format!("plan.candidates[{index}].candidate_id"),
                    "candidate ID must be non-empty",
                ));
            }
            if !candidate_ids.insert(candidate.candidate_id.as_str()) {
                return Err(ParetoError::new(
                    format!("plan.candidates[{index}].candidate_id"),
                    "candidate IDs must be unique",
                ));
            }
            if previous_id.is_some_and(|previous| previous >= candidate.candidate_id.as_str()) {
                return Err(ParetoError::new(
                    format!("plan.candidates[{index}].candidate_id"),
                    "candidate list must be sorted by candidate_id",
                ));
            }
            previous_id = Some(candidate.candidate_id.as_str());

            if candidate.objective_values.len() != dimensions {
                return Err(ParetoError::new(
                    format!("plan.candidates[{index}].objective_values"),
                    "objective vector has the wrong dimension",
                ));
            }
            for (objective_index, value) in candidate.objective_values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(ParetoError::new(
                        format!("plan.candidates[{index}].objective_values[{objective_index}]"),
                        "objective values must be finite",
                    ));
                }
            }
            validate_sha256(
                &candidate.evidence_sha256,
                &format!("plan.candidates[{index}].evidence_sha256"),
            )?;
            if candidate.feasible && !candidate.constraint_violations.is_empty() {
                return Err(ParetoError::new(
                    format!("plan.candidates[{index}].constraint_violations"),
                    "feasible candidates cannot carry constraint violations",
                ));
            }
            if !candidate.feasible
                && (candidate.constraint_violations.is_empty()
                    || candidate.constraint_violations.iter().any(|reason| reason.trim().is_empty()))
            {
                return Err(ParetoError::new(
                    format!("plan.candidates[{index}].constraint_violations"),
                    "infeasible candidates require explicit non-empty violation reasons",
                ));
            }
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, ParetoError> {
        self.validate()?;
        digest_serializable(self)
    }

    fn utility_vector(&self, candidate: &ParetoCandidate) -> Result<Vec<f64>, ParetoError> {
        let mut utility = Vec::with_capacity(self.objective_specs.len());
        for (index, (objective, raw)) in self.objective_specs.iter()
            .zip(&candidate.objective_values).enumerate()
        {
            let range = objective.scale_upper - objective.scale_lower;
            let transformed = match objective.direction {
                ObjectiveDirection::Maximize => (*raw - objective.scale_lower) / range,
                ObjectiveDirection::Minimize => (objective.scale_upper - *raw) / range,
            };
            if !transformed.is_finite() {
                return Err(ParetoError::new(
                    format!("candidate[{}].objective_values[{index}]", candidate.candidate_id),
                    "objective transformation overflowed",
                ));
            }
            utility.push(transformed);
        }
        Ok(utility)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoCandidateResult {
    pub candidate_id: String,
    pub utility_vector: Vec<f64>,
    pub feasible: bool,
    pub constraint_violations: Vec<String>,
    /// Digest of the independently retained evaluation/evidence artifact.
    pub evidence_sha256: String,
    pub on_pareto_front: bool,
    /// Deterministic witness: lexicographically first feasible candidate that dominates this one.
    pub dominated_by: Option<String>,
}

/// Result for one explicitly declared hypervolume reference point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferencePointHypervolume {
    pub reference_point_utility: Vec<f64>,
    pub hypervolume: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoAnalysisSummary {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub objective_ids: Vec<String>,
    pub feasible_candidate_count: usize,
    pub infeasible_candidate_count: usize,
    pub dominated_candidate_count: usize,
    pub nondominated_candidate_ids: Vec<String>,
    pub candidate_results: Vec<ParetoCandidateResult>,
    pub reference_point_utility: Vec<f64>,
    /// Exact Lebesgue measure of the union of boxes between the feasible front and reference.
    pub hypervolume: f64,
    pub plan_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct ParetoReceiptPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    objective_ids: &'a [String],
    feasible_candidate_count: usize,
    infeasible_candidate_count: usize,
    dominated_candidate_count: usize,
    nondominated_candidate_ids: &'a [String],
    candidate_results: &'a [ParetoCandidateResult],
    reference_point_utility: &'a [f64],
    hypervolume: f64,
    plan_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoReferenceSensitivity {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub plan_sha256: String,
    /// Includes the plan's primary reference point as the first entry.
    pub reference_results: Vec<ReferencePointHypervolume>,
    pub minimum_hypervolume: f64,
    pub maximum_hypervolume: f64,
    pub hypervolume_range: f64,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct SensitivityReceiptPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    plan_sha256: &'a str,
    reference_results: &'a [ReferencePointHypervolume],
    minimum_hypervolume: f64,
    maximum_hypervolume: f64,
    hypervolume_range: f64,
}

pub fn analyze_pareto(plan: &ParetoAnalysisPlan) -> Result<ParetoAnalysisSummary, ParetoError> {
    plan.validate()?;
    let plan_sha256 = plan.plan_sha256()?;
    let objective_ids = plan.objective_specs.iter()
        .map(|objective| objective.objective_id.clone())
        .collect::<Vec<_>>();

    let mut utilities = Vec::with_capacity(plan.candidates.len());
    for candidate in &plan.candidates {
        utilities.push(plan.utility_vector(candidate)?);
    }

    let feasible_indices = plan.candidates.iter().enumerate()
        .filter_map(|(index, candidate)| candidate.feasible.then_some(index))
        .collect::<Vec<_>>();

    let mut nondominated = Vec::new();
    let mut dominated_by = vec![None; plan.candidates.len()];
    for &candidate_index in &feasible_indices {
        let candidate_utility = &utilities[candidate_index];
        for &other_index in &feasible_indices {
            if candidate_index == other_index {
                continue;
            }
            if dominates(&utilities[other_index], candidate_utility) {
                dominated_by[candidate_index] = Some(
                    plan.candidates[other_index].candidate_id.clone()
                );
                break;
            }
        }
        if dominated_by[candidate_index].is_none() {
            nondominated.push(candidate_index);
        }
    }

    let nondominated_candidate_ids = nondominated.iter()
        .map(|index| plan.candidates[*index].candidate_id.clone())
        .collect::<Vec<_>>();

    let candidate_results = plan.candidates.iter().enumerate()
        .map(|(index, candidate)| ParetoCandidateResult {
            candidate_id: candidate.candidate_id.clone(),
            utility_vector: utilities[index].clone(),
            feasible: candidate.feasible,
            constraint_violations: candidate.constraint_violations.clone(),
            evidence_sha256: candidate.evidence_sha256.clone(),
            on_pareto_front: nondominated.contains(&index),
            dominated_by: dominated_by[index].clone(),
        })
        .collect::<Vec<_>>();

    let front_points = nondominated.iter()
        .map(|index| utilities[*index].clone())
        .collect::<Vec<_>>();
    let hypervolume = exact_hypervolume(&front_points, &plan.reference_point_utility)?;

    let feasible_candidate_count = feasible_indices.len();
    let infeasible_candidate_count = plan.candidates.len() - feasible_candidate_count;
    let dominated_candidate_count = feasible_candidate_count - nondominated.len();

    let receipt_sha256 = digest_serializable(&ParetoReceiptPayload {
        schema: SCHEMA,
        version: VERSION,
        campaign_id: &plan.campaign_id,
        objective_ids: &objective_ids,
        feasible_candidate_count,
        infeasible_candidate_count,
        dominated_candidate_count,
        nondominated_candidate_ids: &nondominated_candidate_ids,
        candidate_results: &candidate_results,
        reference_point_utility: &plan.reference_point_utility,
        hypervolume,
        plan_sha256: &plan_sha256,
    })?;

    Ok(ParetoAnalysisSummary {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        campaign_id: plan.campaign_id.clone(),
        objective_ids,
        feasible_candidate_count,
        infeasible_candidate_count,
        dominated_candidate_count,
        nondominated_candidate_ids,
        candidate_results,
        reference_point_utility: plan.reference_point_utility.clone(),
        hypervolume,
        plan_sha256,
        receipt_sha256,
    })
}

/// Recompute hypervolume under explicitly supplied reference points without changing the front.
/// The primary point from the plan is included first. This exposes sensitivity rather than
/// allowing the analyst to choose an unreported point after seeing which score looks best.
pub fn analyze_reference_sensitivity(
    plan: &ParetoAnalysisPlan,
    additional_reference_points: &[Vec<f64>],
) -> Result<ParetoReferenceSensitivity, ParetoError> {
    plan.validate()?;
    let primary = analyze_pareto(plan)?;
    let dimensions = plan.objective_specs.len();

    let mut references = Vec::with_capacity(additional_reference_points.len() + 1);
    references.push(plan.reference_point_utility.clone());
    for (index, reference) in additional_reference_points.iter().enumerate() {
        if reference.len() != dimensions {
            return Err(ParetoError::new(
                format!("additional_reference_points[{index}]"),
                "reference point dimension differs from the objective count",
            ));
        }
        if reference.iter().any(|value| !value.is_finite()) {
            return Err(ParetoError::new(
                format!("additional_reference_points[{index}]"),
                "reference point coordinates must be finite",
            ));
        }
        if references.iter().any(|existing| same_vector(existing, reference)) {
            return Err(ParetoError::new(
                format!("additional_reference_points[{index}]"),
                "reference points must be unique",
            ));
        }
        references.push(reference.clone());
    }

    let mut reference_results = Vec::with_capacity(references.len());
    for reference in references {
        let hypervolume = if same_vector(&reference, &plan.reference_point_utility) {
            primary.hypervolume
        } else {
            let mut points = Vec::new();
            for candidate in &plan.candidates {
                if candidate.feasible {
                    points.push(plan.utility_vector(candidate)?);
                }
            }
            let front = nondominated_vectors(&points);
            exact_hypervolume(&front, &reference)?
        };
        reference_results.push(ReferencePointHypervolume {
            reference_point_utility: reference,
            hypervolume,
        });
    }

    let minimum_hypervolume = reference_results.iter()
        .map(|result| result.hypervolume)
        .fold(f64::INFINITY, f64::min);
    let maximum_hypervolume = reference_results.iter()
        .map(|result| result.hypervolume)
        .fold(f64::NEG_INFINITY, f64::max);
    let hypervolume_range = maximum_hypervolume - minimum_hypervolume;
    if !hypervolume_range.is_finite() {
        return Err(ParetoError::new("hypervolume_range", "hypervolume range overflowed"));
    }
    let plan_sha256 = plan.plan_sha256()?;
    let receipt_sha256 = digest_serializable(&SensitivityReceiptPayload {
        schema: SCHEMA,
        version: VERSION,
        campaign_id: &plan.campaign_id,
        plan_sha256: &plan_sha256,
        reference_results: &reference_results,
        minimum_hypervolume,
        maximum_hypervolume,
        hypervolume_range,
    })?;

    Ok(ParetoReferenceSensitivity {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        campaign_id: plan.campaign_id.clone(),
        plan_sha256,
        reference_results,
        minimum_hypervolume,
        maximum_hypervolume,
        hypervolume_range,
        receipt_sha256,
    })
}

/// Strict Pareto dominance in transformed maximization-utility space.
/// Equality in every dimension is a tie, not strict dominance.
fn dominates(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right)
        .all(|(left_value, right_value)| left_value >= right_value)
        && left.iter().zip(right)
            .any(|(left_value, right_value)| left_value > right_value)
}

fn nondominated_vectors(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    points.iter().enumerate()
        .filter_map(|(index, point)| {
            let dominated = points.iter().enumerate().any(|(other_index, other)| {
                index != other_index && dominates(other, point)
            });
            (!dominated).then(|| point.clone())
        })
        .collect()
}

/// Exact recursive hypervolume of the union of axis-aligned boxes [reference, point].
/// Inputs are maximization vectors. Points that do not strictly exceed the reference in
/// every dimension contribute zero volume. Complexity is intended for small archived fronts
/// and at most four objectives, not millions of points.
fn exact_hypervolume(points: &[Vec<f64>], reference: &[f64]) -> Result<f64, ParetoError> {
    if points.is_empty() {
        return Ok(0.0);
    }
    if reference.is_empty() {
        return Err(ParetoError::new("reference_point", "reference point cannot be empty"));
    }
    for (index, point) in points.iter().enumerate() {
        if point.len() != reference.len() {
            return Err(ParetoError::new(
                format!("points[{index}]"),
                "point dimension differs from reference dimension",
            ));
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(ParetoError::new(
                format!("points[{index}]"),
                "hypervolume point coordinates must be finite",
            ));
        }
    }
    if reference.iter().any(|value| !value.is_finite()) {
        return Err(ParetoError::new("reference_point", "reference point must be finite"));
    }
    hypervolume_recursive(points, reference)
}

fn hypervolume_recursive(points: &[Vec<f64>], reference: &[f64]) -> Result<f64, ParetoError> {
    let dimensions = reference.len();
    if dimensions == 1 {
        let maximum = points.iter()
            .filter_map(|point| point.first().copied())
            .fold(reference[0], f64::max);
        let volume = (maximum - reference[0]).max(0.0);
        if !volume.is_finite() {
            return Err(ParetoError::new("hypervolume", "one-dimensional volume overflowed"));
        }
        return Ok(volume);
    }

    let last = dimensions - 1;
    let mut levels = points.iter()
        .filter_map(|point| (point[last] > reference[last]).then_some(point[last]))
        .collect::<Vec<_>>();
    if levels.is_empty() {
        return Ok(0.0);
    }
    levels.push(reference[last]);
    levels.sort_by(f64::total_cmp);
    levels.dedup_by(|left, right| *left == *right);

    let mut volume = 0.0;
    for pair in levels.windows(2) {
        let lower = pair[0];
        let upper = pair[1];
        let height = upper - lower;
        if height <= 0.0 {
            continue;
        }
        let active = points.iter()
            .filter(|point| point[last] >= upper)
            .map(|point| point[..last].to_vec())
            .collect::<Vec<_>>();
        let cross_section = hypervolume_recursive(&active, &reference[..last])?;
        let slab = cross_section * height;
        volume += slab;
        if !slab.is_finite() || !volume.is_finite() {
            return Err(ParetoError::new("hypervolume", "recursive volume calculation overflowed"));
        }
    }
    Ok(volume)
}

fn same_vector(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right)
            .all(|(left_value, right_value)| left_value == right_value)
}

fn validate_sha256(value: &str, path: &str) -> Result<(), ParetoError> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ParetoError::new(
            path,
            "expected a lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, ParetoError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ParetoError::new("serialization", format!("cannot serialize receipt: {error}"))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Inverted generational distance plus in maximization-utility coordinates.
///
/// For each reference point, this averages the nearest one-sided distance from the reference
/// point to the approximation set. A point that is better than the reference in every
/// objective has zero one-sided distance. The caller must supply a frozen reference set.
pub fn inverted_generational_distance_plus(
    approximation_utility: &[Vec<f64>],
    reference_utility: &[Vec<f64>],
) -> Result<f64, ParetoError> {
    let dimensions = validate_indicator_fronts(approximation_utility, reference_utility)?;
    let mut total_distance = 0.0;

    for (reference_index, reference) in reference_utility.iter().enumerate() {
        let mut nearest = f64::INFINITY;
        for (approximation_index, approximation) in approximation_utility.iter().enumerate() {
            let mut squared_distance = 0.0;
            for dimension in 0..dimensions {
                let gap = (reference[dimension] - approximation[dimension]).max(0.0);
                squared_distance += gap * gap;
            }
            if !squared_distance.is_finite() {
                return Err(ParetoError::new(
                    format!("approximation_utility[{approximation_index}]"),
                    "IGD+ squared distance overflowed",
                ));
            }
            nearest = nearest.min(squared_distance.sqrt());
        }
        total_distance += nearest;
        if !total_distance.is_finite() {
            return Err(ParetoError::new(
                format!("reference_utility[{reference_index}]"),
                "IGD+ aggregation overflowed",
            ));
        }
    }
    let result = total_distance / reference_utility.len() as f64;
    if !result.is_finite() {
        return Err(ParetoError::new("igd_plus", "IGD+ result is not finite"));
    }
    Ok(result)
}

/// Additive epsilon indicator I_epsilon+(A, R) in maximization-utility coordinates.
///
/// Returns the smallest additive utility shift required for the approximation set A to
/// weakly cover every point in the frozen reference set R. Negative values mean the
/// approximation set strictly exceeds the reference set under this indicator.
pub fn additive_epsilon_indicator(
    approximation_utility: &[Vec<f64>],
    reference_utility: &[Vec<f64>],
) -> Result<f64, ParetoError> {
    let dimensions = validate_indicator_fronts(approximation_utility, reference_utility)?;
    let mut worst_reference_gap = f64::NEG_INFINITY;

    for (reference_index, reference) in reference_utility.iter().enumerate() {
        let mut best_approximation_gap = f64::INFINITY;
        for (approximation_index, approximation) in approximation_utility.iter().enumerate() {
            let mut maximum_gap = f64::NEG_INFINITY;
            for dimension in 0..dimensions {
                let gap = reference[dimension] - approximation[dimension];
                if !gap.is_finite() {
                    return Err(ParetoError::new(
                        format!("approximation_utility[{approximation_index}]"),
                        "additive epsilon difference overflowed",
                    ));
                }
                maximum_gap = maximum_gap.max(gap);
            }
            best_approximation_gap = best_approximation_gap.min(maximum_gap);
        }
        worst_reference_gap = worst_reference_gap.max(best_approximation_gap);
        if !worst_reference_gap.is_finite() {
            return Err(ParetoError::new(
                format!("reference_utility[{reference_index}]"),
                "additive epsilon aggregation overflowed",
            ));
        }
    }
    Ok(worst_reference_gap)
}

fn validate_indicator_fronts(
    approximation_utility: &[Vec<f64>],
    reference_utility: &[Vec<f64>],
) -> Result<usize, ParetoError> {
    if approximation_utility.is_empty() {
        return Err(ParetoError::new(
            "approximation_utility",
            "approximation set cannot be empty",
        ));
    }
    if reference_utility.is_empty() {
        return Err(ParetoError::new(
            "reference_utility",
            "reference set cannot be empty",
        ));
    }
    let dimensions = reference_utility[0].len();
    if !(MIN_DIMENSIONS..=MAX_EXACT_HYPERVOLUME_DIMENSIONS).contains(&dimensions) {
        return Err(ParetoError::new(
            "reference_utility",
            "indicator dimensions must be between two and four",
        ));
    }
    for (set_name, points) in [
        ("approximation_utility", approximation_utility),
        ("reference_utility", reference_utility),
    ] {
        for (index, point) in points.iter().enumerate() {
            if point.len() != dimensions {
                return Err(ParetoError::new(
                    format!("{set_name}[{index}]"),
                    "point dimension differs from the reference front",
                ));
            }
            if point.iter().any(|value| !value.is_finite()) {
                return Err(ParetoError::new(
                    format!("{set_name}[{index}]"),
                    "indicator coordinates must be finite",
                ));
            }
        }
    }
    Ok(dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn specs() -> Vec<ParetoObjectiveSpec> {
        vec![
            ParetoObjectiveSpec {
                objective_id: "quality".to_owned(),
                direction: ObjectiveDirection::Maximize,
                scale_lower: 0.0,
                scale_upper: 1.0,
            },
            ParetoObjectiveSpec {
                objective_id: "robustness".to_owned(),
                direction: ObjectiveDirection::Maximize,
                scale_lower: 0.0,
                scale_upper: 1.0,
            },
            ParetoObjectiveSpec {
                objective_id: "energy".to_owned(),
                direction: ObjectiveDirection::Minimize,
                scale_lower: 0.0,
                scale_upper: 100.0,
            },
        ]
    }

    fn candidate(
        id: &str,
        quality: f64,
        robustness: f64,
        energy: f64,
        feasible: bool,
        violations: Vec<&str>,
        evidence: &str,
    ) -> ParetoCandidate {
        ParetoCandidate {
            candidate_id: id.to_owned(),
            objective_values: vec![quality, robustness, energy],
            feasible,
            constraint_violations: violations.into_iter().map(str::to_owned).collect(),
            evidence_sha256: evidence.to_owned(),
        }
    }

    fn plan(candidates: Vec<ParetoCandidate>) -> ParetoAnalysisPlan {
        ParetoAnalysisPlan::new(
            "pareto-test-campaign",
            specs(),
            vec![0.0, 0.0, 0.0],
            candidates,
        ).unwrap()
    }

    #[test]
    fn mixed_maximize_minimize_objectives_produce_expected_utility_and_front() {
        let summary = analyze_pareto(&plan(vec![
            candidate("a", 0.8, 0.4, 80.0, true, vec![], H1),
            candidate("b", 0.4, 0.8, 20.0, true, vec![], H2),
            candidate("c", 0.7, 0.3, 90.0, true, vec![], H3),
            candidate("d", 0.9, 0.9, 95.0, false, vec!["thermal limit".into()], H4),
        ])).unwrap();

        assert_eq!(summary.nondominated_candidate_ids, vec!["a", "b"]);
        assert_eq!(summary.feasible_candidate_count, 3);
        assert_eq!(summary.infeasible_candidate_count, 1);
        assert_eq!(summary.dominated_candidate_count, 1);
        let c = summary.candidate_results.iter().find(|value| value.candidate_id == "c").unwrap();
        assert!(c.dominated_by.as_deref() == Some("a"));
        let b = summary.candidate_results.iter().find(|value| value.candidate_id == "b").unwrap();
        assert!((b.utility_vector[2] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn exact_three_dimensional_hypervolume_matches_known_union() {
        let volume = exact_hypervolume(
            &[
                vec![0.8, 0.4, 0.4],
                vec![0.4, 0.8, 0.4],
                vec![0.4, 0.4, 0.8],
            ],
            &[0.0, 0.0, 0.0],
        ).unwrap();
        // Inclusion/exclusion: 3*(.8*.4*.4) - 3*(.4*.4*.4) + (.4*.4*.4).
        let expected = 3.0 * 0.8 * 0.4 * 0.4
            - 3.0 * 0.4 * 0.4 * 0.4
            + 0.4 * 0.4 * 0.4;
        assert!((volume - expected).abs() < 1e-12);
    }

    #[test]
    fn exact_two_dimensional_hypervolume_matches_known_union() {
        let volume = exact_hypervolume(
            &[vec![0.8, 0.4], vec![0.4, 0.8]],
            &[0.0, 0.0],
        ).unwrap();
        assert!((volume - 0.48).abs() < 1e-12);
    }

    #[test]
    fn infeasible_candidates_never_contribute_to_the_front_or_hypervolume() {
        let with_infeasible = plan(vec![
            candidate("a", 0.7, 0.7, 50.0, true, vec![], H1),
            candidate("unsafe", 1.0, 1.0, 0.0, false, vec!["unsafe configuration".into()], H2),
        ]);
        let without_infeasible = plan(vec![
            candidate("a", 0.7, 0.7, 50.0, true, vec![], H1),
        ]);
        let left = analyze_pareto(&with_infeasible).unwrap();
        let right = analyze_pareto(&without_infeasible).unwrap();
        assert_eq!(left.nondominated_candidate_ids, vec!["a"]);
        assert_eq!(left.hypervolume, right.hypervolume);
    }

    #[test]
    fn reference_point_sensitivity_is_explicit_and_receipted() {
        let benchmark = plan(vec![
            candidate("a", 0.8, 0.4, 80.0, true, vec![], H1),
            candidate("b", 0.4, 0.8, 20.0, true, vec![], H2),
        ]);
        let result = analyze_reference_sensitivity(
            &benchmark,
            &[vec![0.1, 0.1, 0.1], vec![0.2, 0.2, 0.2]],
        ).unwrap();
        assert_eq!(result.reference_results.len(), 3);
        assert!(result.hypervolume_range > 0.0);
        assert_eq!(result.receipt_sha256.len(), 64);
    }

    #[test]
    fn candidate_input_order_does_not_change_plan_or_analysis_receipts() {
        let a = candidate("a", 0.8, 0.4, 80.0, true, vec![], H1);
        let b = candidate("b", 0.4, 0.8, 20.0, true, vec![], H2);
        let left = plan(vec![a.clone(), b.clone()]);
        let right = plan(vec![b, a]);
        let left_result = analyze_pareto(&left).unwrap();
        let right_result = analyze_pareto(&right).unwrap();
        assert_eq!(left_result.plan_sha256, right_result.plan_sha256);
        assert_eq!(left_result.receipt_sha256, right_result.receipt_sha256);
    }

    #[test]
    fn malformed_dimension_feasibility_and_evidence_are_rejected() {
        let mut bad = candidate("bad", 0.5, 0.5, 50.0, true, vec![], H1);
        bad.objective_values.pop();
        assert!(ParetoAnalysisPlan::new(
            "campaign", specs(), vec![0.0, 0.0, 0.0], vec![bad],
        ).is_err());

        let unsafe_candidate = candidate(
            "unsafe", 0.5, 0.5, 50.0, false, vec![], H2,
        );
        assert!(ParetoAnalysisPlan::new(
            "campaign", specs(), vec![0.0, 0.0, 0.0], vec![unsafe_candidate],
        ).is_err());

        let mut invalid_evidence = candidate("bad-hash", 0.5, 0.5, 50.0, true, vec![], H3);
        invalid_evidence.evidence_sha256 = "not-a-digest".to_owned();
        assert!(ParetoAnalysisPlan::new(
            "campaign", specs(), vec![0.0, 0.0, 0.0], vec![invalid_evidence],
        ).is_err());
    }

    #[test]
    fn exact_four_dimensional_single_box_volume_is_computed() {
        let volume = exact_hypervolume(
            &[vec![0.5, 0.5, 0.5, 0.5]],
            &[0.0, 0.0, 0.0, 0.0],
        ).unwrap();
        assert!((volume - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn hypervolume_ignores_candidate_that_fails_one_reference_coordinate() {
        let volume = exact_hypervolume(
            &[vec![0.9, 0.1], vec![0.2, 0.8]],
            &[0.0, 0.3],
        ).unwrap();
        assert!((volume - 0.1).abs() < 1e-12);
    }

    #[test]
    fn signed_zero_reference_points_are_semantically_equal() {
        let benchmark = plan(vec![
            candidate("a", 0.8, 0.4, 80.0, true, vec![], H1),
        ]);
        assert!(analyze_reference_sensitivity(
            &benchmark,
            &[vec![-0.0, 0.0, -0.0]],
        ).is_err());
    }

    #[test]
    fn igd_plus_and_additive_epsilon_match_known_maximization_case() {
        let reference = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let approximation = vec![vec![0.8, 0.0], vec![0.0, 0.8]];
        let igd = inverted_generational_distance_plus(&approximation, &reference).unwrap();
        let epsilon = additive_epsilon_indicator(&approximation, &reference).unwrap();
        assert!((igd - 0.2).abs() < 1e-12);
        assert!((epsilon - 0.2).abs() < 1e-12);
    }

    #[test]
    fn igd_plus_and_epsilon_reward_reference_coverage_without_scalarization() {
        let reference = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let exact = reference.clone();
        assert_eq!(inverted_generational_distance_plus(&exact, &reference).unwrap(), 0.0);
        assert_eq!(additive_epsilon_indicator(&exact, &reference).unwrap(), 0.0);

        let better = vec![vec![1.1, 0.0], vec![0.0, 1.1]];
        assert_eq!(inverted_generational_distance_plus(&better, &reference).unwrap(), 0.0);
        assert!((additive_epsilon_indicator(&better, &reference).unwrap() + 0.1).abs() < 1e-12);
    }

    #[test]
    fn indicators_reject_empty_or_inconsistent_reference_sets() {
        assert!(inverted_generational_distance_plus(&[], &[vec![0.0, 1.0]]).is_err());
        assert!(additive_epsilon_indicator(&[vec![0.0, 1.0]], &[]).is_err());
        assert!(inverted_generational_distance_plus(
            &[vec![0.0, 1.0, 0.5]],
            &[vec![0.0, 1.0]],
        ).is_err());
    }

    #[test]
    fn duplicate_reference_points_are_rejected() {
        let benchmark = plan(vec![
            candidate("a", 0.8, 0.4, 80.0, true, vec![], H1),
        ]);
        assert!(analyze_reference_sensitivity(
            &benchmark,
            &[vec![0.0, 0.0, 0.0]],
        ).is_err());
    }
}
