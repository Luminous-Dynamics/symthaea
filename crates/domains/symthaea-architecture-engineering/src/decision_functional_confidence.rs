//! Simultaneous decision-functional confidence bounds for a finite policy frontier.
//!
//! This V1 implementation estimates only the bounded expectations consumed by
//! the policy-frontier solver. It does not estimate or certify the complete path
//! distribution. Calibration and confirmation must share a target law by design;
//! bounded transport and off-policy estimation are intentionally unsupported.
//!
//! SHA-256 binds artifact identity, not data authenticity, IID sampling,
//! randomization integrity, target-law equality, or absence of interference.

use super::policy_frontier_lp::{
    analyze_policy_frontier, PolicyFrontierError, PolicyFrontierPlan, PolicyFrontierReport,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

const REQUEST_SCHEMA: &str =
    "symthaea.architecture-engineering.decision-functional-calibration-request";
const REQUEST_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str =
    "symthaea.architecture-engineering.decision-functional-calibration-report";
const REPORT_VERSION: &str = "0.1.0";
const TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionFunctionalError {
    pub path: String,
    pub message: String,
}

impl DecisionFunctionalError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }

    fn from_frontier(error: PolicyFrontierError) -> Self {
        Self {
            path: error.path,
            message: error.message,
        }
    }
}

impl fmt::Display for DecisionFunctionalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for DecisionFunctionalError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CalibrationTargetRelation {
    SameTargetLawByDesign,
    BoundedTransport,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CalibrationEvidenceMode {
    RandomizedPolicyExecution,
    SimulatorGroundTruth,
    OffPolicyEstimate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecisionFunctionalKind {
    NullRejection,
    AlternativePower,
    EvaluatorCost,
    DecisionUtility,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NullPolicyModelSamples {
    pub policy_id: String,
    pub model_id: String,
    /// Unique independent sampling-unit IDs, aligned with rejection_values.
    /// Clustered/repeated data must be aggregated to the independent unit first.
    pub sampling_unit_ids: Vec<String>,
    /// Per-unit rejection indicator or bounded conditional rejection probability.
    pub rejection_values: Vec<f64>,
    pub assignment_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlternativePolicyModelSamples {
    pub policy_id: String,
    pub model_id: String,
    /// Digest of the policy-specific utility/cost/outcome functional contract.
    pub utility_contract_sha256: String,
    /// Unique independent sampling-unit IDs, aligned across the three value vectors.
    pub sampling_unit_ids: Vec<String>,
    /// Per-unit rejection indicator or bounded conditional rejection probability.
    pub rejection_values: Vec<f64>,
    pub evaluator_cost_values: Vec<f64>,
    pub decision_utility_values: Vec<f64>,
    pub evaluator_cost_lower_bound: f64,
    pub evaluator_cost_upper_bound: f64,
    pub decision_utility_lower_bound: f64,
    pub decision_utility_upper_bound: f64,
    pub assignment_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionFunctionalCalibrationRequest {
    pub schema: String,
    pub version: String,
    pub calibration_id: String,
    /// SHA-256 of the canonical nominal PolicyFrontierPlan.
    pub source_plan_sha256: String,
    /// Shared semantic identity for the declared path-level functional definitions.
    pub functional_contract_sha256: String,
    /// Equal design digests encode the SAME_TARGET_LAW_BY_DESIGN assertion.
    /// They do not prove that the actual data follow this design.
    pub calibration_target_design_sha256: String,
    pub confirmatory_target_design_sha256: String,
    pub target_relation: CalibrationTargetRelation,
    pub evidence_mode: CalibrationEvidenceMode,
    /// Null-family simultaneous coverage failure budget; must be < nominal alpha.
    pub null_family_coverage_failure_probability: f64,
    /// Alternative-family simultaneous coverage failure budget.
    pub alternative_family_coverage_failure_probability: f64,
    pub calibration_independent_of_confirmatory_stream: bool,
    pub independent_sampling_units_declared: bool,
    /// V1 rejects unmodeled interference. This is a design declaration, not proof.
    pub no_interference_within_sampling_unit_declared: bool,
    pub null_rows: Vec<NullPolicyModelSamples>,
    pub alternative_rows: Vec<AlternativePolicyModelSamples>,
}

impl DecisionFunctionalCalibrationRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        calibration_id: impl Into<String>,
        source_plan_sha256: impl Into<String>,
        functional_contract_sha256: impl Into<String>,
        calibration_target_design_sha256: impl Into<String>,
        confirmatory_target_design_sha256: impl Into<String>,
        target_relation: CalibrationTargetRelation,
        evidence_mode: CalibrationEvidenceMode,
        null_family_coverage_failure_probability: f64,
        alternative_family_coverage_failure_probability: f64,
        calibration_independent_of_confirmatory_stream: bool,
        independent_sampling_units_declared: bool,
        no_interference_within_sampling_unit_declared: bool,
        null_rows: Vec<NullPolicyModelSamples>,
        alternative_rows: Vec<AlternativePolicyModelSamples>,
    ) -> Result<Self, DecisionFunctionalError> {
        let mut request = Self {
            schema: REQUEST_SCHEMA.into(),
            version: REQUEST_VERSION.into(),
            calibration_id: calibration_id.into(),
            source_plan_sha256: source_plan_sha256.into(),
            functional_contract_sha256: functional_contract_sha256.into(),
            calibration_target_design_sha256: calibration_target_design_sha256.into(),
            confirmatory_target_design_sha256: confirmatory_target_design_sha256.into(),
            target_relation,
            evidence_mode,
            null_family_coverage_failure_probability,
            alternative_family_coverage_failure_probability,
            calibration_independent_of_confirmatory_stream,
            independent_sampling_units_declared,
            no_interference_within_sampling_unit_declared,
            null_rows,
            alternative_rows,
        };
        request.canonicalize()?;
        Ok(request)
    }

    pub fn canonicalize(&mut self) -> Result<(), DecisionFunctionalError> {
        self.null_rows.sort_by(|left, right| {
            (&left.policy_id, &left.model_id).cmp(&(&right.policy_id, &right.model_id))
        });
        self.alternative_rows.sort_by(|left, right| {
            (&left.policy_id, &left.model_id).cmp(&(&right.policy_id, &right.model_id))
        });
        self.validate_structure()
    }

    fn validate_structure(&self) -> Result<(), DecisionFunctionalError> {
        if self.schema != REQUEST_SCHEMA || self.version != REQUEST_VERSION {
            return Err(DecisionFunctionalError::new(
                "request.schema",
                "unsupported decision-functional request schema/version",
            ));
        }
        if self.calibration_id.trim().is_empty() {
            return Err(DecisionFunctionalError::new(
                "request.calibration_id",
                "calibration ID must be non-empty",
            ));
        }
        validate_hash(&self.source_plan_sha256, "request.source_plan_sha256")?;
        validate_hash(
            &self.functional_contract_sha256,
            "request.functional_contract_sha256",
        )?;
        validate_hash(
            &self.calibration_target_design_sha256,
            "request.calibration_target_design_sha256",
        )?;
        validate_hash(
            &self.confirmatory_target_design_sha256,
            "request.confirmatory_target_design_sha256",
        )?;

        if self.target_relation != CalibrationTargetRelation::SameTargetLawByDesign {
            return Err(DecisionFunctionalError::new(
                "request.target_relation",
                "V1 qualifies only SAME_TARGET_LAW_BY_DESIGN; bounded transport and unknown target relations are not supported",
            ));
        }
        if self.calibration_target_design_sha256 != self.confirmatory_target_design_sha256 {
            return Err(DecisionFunctionalError::new(
                "request.confirmatory_target_design_sha256",
                "calibration and confirmation design identities differ; a transport certificate is required",
            ));
        }
        if self.evidence_mode == CalibrationEvidenceMode::OffPolicyEstimate {
            return Err(DecisionFunctionalError::new(
                "request.evidence_mode",
                "V1 does not support off-policy estimates or importance-weighted outcomes",
            ));
        }
        if !self.calibration_independent_of_confirmatory_stream {
            return Err(DecisionFunctionalError::new(
                "request.calibration_independent_of_confirmatory_stream",
                "calibration must be independent of the confirmatory stream",
            ));
        }
        if !self.independent_sampling_units_declared {
            return Err(DecisionFunctionalError::new(
                "request.independent_sampling_units_declared",
                "V1 requires independently sampled units; aggregate clusters before calibration",
            ));
        }
        if !self.no_interference_within_sampling_unit_declared {
            return Err(DecisionFunctionalError::new(
                "request.no_interference_within_sampling_unit_declared",
                "V1 does not qualify outcomes with unmodeled interference",
            ));
        }
        validate_open_probability(
            self.null_family_coverage_failure_probability,
            "request.null_family_coverage_failure_probability",
        )?;
        validate_open_probability(
            self.alternative_family_coverage_failure_probability,
            "request.alternative_family_coverage_failure_probability",
        )?;
        if self.null_rows.is_empty() || self.alternative_rows.is_empty() {
            return Err(DecisionFunctionalError::new(
                "request.rows",
                "null and alternative calibration rows must both be non-empty",
            ));
        }

        validate_unique_row_ids(
            self.null_rows.iter().map(|row| (&row.policy_id, &row.model_id)),
            "request.null_rows",
        )?;
        validate_unique_row_ids(
            self.alternative_rows
                .iter()
                .map(|row| (&row.policy_id, &row.model_id)),
            "request.alternative_rows",
        )?;

        if self.evidence_mode == CalibrationEvidenceMode::RandomizedPolicyExecution {
            validate_randomized_assignment(
                self.null_rows.iter().map(|row| (&row.model_id, &row.policy_id, row.assignment_probability, row.sampling_unit_ids.as_slice())),
                "request.null_rows",
            )?;
            validate_randomized_assignment(
                self.alternative_rows.iter().map(|row| (&row.model_id, &row.policy_id, row.assignment_probability, row.sampling_unit_ids.as_slice())),
                "request.alternative_rows",
            )?;
        }

        for (index, row) in self.null_rows.iter().enumerate() {
            validate_sampling_units(
                &row.sampling_unit_ids,
                &row.rejection_values,
                &format!("request.null_rows[{index}]"),
            )?;
            validate_probability(row.assignment_probability, &format!("request.null_rows[{index}].assignment_probability"))?;
            if row.assignment_probability <= 0.0 {
                return Err(DecisionFunctionalError::new(
                    format!("request.null_rows[{index}].assignment_probability"),
                    "every candidate policy must have positive assignment probability",
                ));
            }
            for (sample_index, value) in row.rejection_values.iter().enumerate() {
                validate_bounded(
                    *value,
                    0.0,
                    1.0,
                    &format!("request.null_rows[{index}].rejection_values[{sample_index}]"),
                )?;
            }
        }

        for (index, row) in self.alternative_rows.iter().enumerate() {
            let path = format!("request.alternative_rows[{index}]");
            validate_hash(
                &row.utility_contract_sha256,
                &format!("{path}.utility_contract_sha256"),
            )?;
            validate_sampling_units(
                &row.sampling_unit_ids,
                &row.rejection_values,
                &path,
            )?;
            if row.evaluator_cost_values.len() != row.sampling_unit_ids.len()
                || row.decision_utility_values.len() != row.sampling_unit_ids.len()
            {
                return Err(DecisionFunctionalError::new(
                    format!("{path}.samples"),
                    "rejection, cost and utility vectors must align one-to-one with sampling units",
                ));
            }
            validate_probability(
                row.assignment_probability,
                &format!("{path}.assignment_probability"),
            )?;
            if row.assignment_probability <= 0.0 {
                return Err(DecisionFunctionalError::new(
                    format!("{path}.assignment_probability"),
                    "every candidate policy must have positive assignment probability",
                ));
            }
            validate_range(
                row.evaluator_cost_lower_bound,
                row.evaluator_cost_upper_bound,
                &format!("{path}.evaluator_cost_bounds"),
            )?;
            validate_range(
                row.decision_utility_lower_bound,
                row.decision_utility_upper_bound,
                &format!("{path}.decision_utility_bounds"),
            )?;
            for (sample_index, value) in row.rejection_values.iter().enumerate() {
                validate_bounded(
                    *value,
                    0.0,
                    1.0,
                    &format!("{path}.rejection_values[{sample_index}]"),
                )?;
            }
            for (sample_index, value) in row.evaluator_cost_values.iter().enumerate() {
                validate_bounded(
                    *value,
                    row.evaluator_cost_lower_bound,
                    row.evaluator_cost_upper_bound,
                    &format!("{path}.evaluator_cost_values[{sample_index}]"),
                )?;
            }
            for (sample_index, value) in row.decision_utility_values.iter().enumerate() {
                validate_bounded(
                    *value,
                    row.decision_utility_lower_bound,
                    row.decision_utility_upper_bound,
                    &format!("{path}.decision_utility_values[{sample_index}]"),
                )?;
            }
        }
        Ok(())
    }

    pub fn request_sha256(&self) -> Result<String, DecisionFunctionalError> {
        let mut canonical = self.clone();
        canonical.canonicalize()?;
        digest(&canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionFunctionalConfidenceComponent {
    pub policy_id: String,
    pub model_id: String,
    pub functional: DecisionFunctionalKind,
    pub sample_size: usize,
    pub allocated_failure_probability: f64,
    pub support_lower_bound: f64,
    pub support_upper_bound: f64,
    pub empirical_mean: f64,
    pub confidence_radius: f64,
    pub lower_confidence_bound: f64,
    pub upper_confidence_bound: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionFunctionalCalibrationReport {
    pub schema: String,
    pub version: String,
    pub calibration_id: String,
    pub source_plan_sha256: String,
    pub request_sha256: String,
    pub functional_contract_sha256: String,
    pub calibration_target_design_sha256: String,
    pub confirmatory_target_design_sha256: String,
    pub evidence_mode: CalibrationEvidenceMode,
    pub target_relation: CalibrationTargetRelation,
    pub nominal_significance_level: f64,
    pub conditional_significance_level: f64,
    pub null_family_coverage_failure_probability: f64,
    pub alternative_family_coverage_failure_probability: f64,
    pub worst_case_unconditional_type_i_upper_bound: f64,
    pub robustified_plan_sha256: String,
    pub components: Vec<DecisionFunctionalConfidenceComponent>,
    pub frontier: PolicyFrontierReport,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    calibration_id: &'a str,
    source_plan_sha256: &'a str,
    request_sha256: &'a str,
    functional_contract_sha256: &'a str,
    calibration_target_design_sha256: &'a str,
    confirmatory_target_design_sha256: &'a str,
    evidence_mode: CalibrationEvidenceMode,
    target_relation: CalibrationTargetRelation,
    nominal_significance_level: f64,
    conditional_significance_level: f64,
    null_family_coverage_failure_probability: f64,
    alternative_family_coverage_failure_probability: f64,
    worst_case_unconditional_type_i_upper_bound: f64,
    robustified_plan_sha256: &'a str,
    components: &'a [DecisionFunctionalConfidenceComponent],
    frontier: &'a PolicyFrontierReport,
}

fn validate_open_probability(value: f64, path: &str) -> Result<(), DecisionFunctionalError> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(DecisionFunctionalError::new(
            path,
            "failure probability must be finite and strictly between zero and one",
        ));
    }
    Ok(())
}

fn validate_probability(value: f64, path: &str) -> Result<(), DecisionFunctionalError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(DecisionFunctionalError::new(
            path,
            "probability must be finite and lie in [0, 1]",
        ));
    }
    Ok(())
}

fn validate_hash(value: &str, path: &str) -> Result<(), DecisionFunctionalError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DecisionFunctionalError::new(
            path,
            "expected a 64-character hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn validate_range(lower: f64, upper: f64, path: &str) -> Result<(), DecisionFunctionalError> {
    let width = upper - lower;
    if !lower.is_finite() || !upper.is_finite() || !width.is_finite() || lower > upper {
        return Err(DecisionFunctionalError::new(
            path,
            "support bounds and their width must be finite and ordered",
        ));
    }
    Ok(())
}

fn validate_bounded(
    value: f64,
    lower: f64,
    upper: f64,
    path: &str,
) -> Result<(), DecisionFunctionalError> {
    if !value.is_finite() || value < lower || value > upper {
        return Err(DecisionFunctionalError::new(
            path,
            "observation must be finite and lie within the declared support",
        ));
    }
    Ok(())
}

fn validate_sampling_units(
    ids: &[String],
    values: &[f64],
    path: &str,
) -> Result<(), DecisionFunctionalError> {
    if ids.is_empty() || ids.len() != values.len() {
        return Err(DecisionFunctionalError::new(
            format!("{path}.sampling_unit_ids"),
            "sampling-unit IDs must be non-empty and align with observations",
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, id) in ids.iter().enumerate() {
        if id.trim().is_empty() || !seen.insert(id.as_str()) {
            return Err(DecisionFunctionalError::new(
                format!("{path}.sampling_unit_ids[{index}]"),
                "sampling-unit IDs must be non-empty and unique within the row",
            ));
        }
    }
    Ok(())
}

fn validate_randomized_assignment<'a>(
    rows: impl Iterator<Item = (&'a String, &'a String, f64, &'a [String])>,
    path: &str,
) -> Result<(), DecisionFunctionalError> {
    let mut by_model: BTreeMap<&str, (f64, BTreeSet<&str>)> = BTreeMap::new();
    for (model_id, _policy_id, probability, unit_ids) in rows {
        let entry = by_model
            .entry(model_id.as_str())
            .or_insert_with(|| (0.0, BTreeSet::new()));
        entry.0 += probability;
        for unit_id in unit_ids {
            if !entry.1.insert(unit_id.as_str()) {
                return Err(DecisionFunctionalError::new(
                    format!("{path}.{model_id}.sampling_unit_ids"),
                    "a sampling unit appears in more than one randomized policy arm for this model",
                ));
            }
        }
    }
    for (model_id, (probability_sum, _)) in by_model {
        if (probability_sum - 1.0).abs() > TOLERANCE {
            return Err(DecisionFunctionalError::new(
                format!("{path}.{model_id}.assignment_probability"),
                "randomized policy-arm assignment probabilities must sum to one per model",
            ));
        }
    }
    Ok(())
}

fn validate_unique_row_ids<'a>(
    rows: impl Iterator<Item = (&'a String, &'a String)>,
    path: &str,
) -> Result<(), DecisionFunctionalError> {
    let mut seen = BTreeSet::new();
    for (index, (policy_id, model_id)) in rows.enumerate() {
        if policy_id.trim().is_empty()
            || model_id.trim().is_empty()
            || !seen.insert((policy_id.as_str(), model_id.as_str()))
        {
            return Err(DecisionFunctionalError::new(
                format!("{path}[{index}]"),
                "policy/model row IDs must be non-empty and unique",
            ));
        }
    }
    Ok(())
}

fn mean(values: &[f64]) -> Result<f64, DecisionFunctionalError> {
    if values.is_empty() {
        return Err(DecisionFunctionalError::new(
            "samples",
            "at least one observation is required",
        ));
    }
    let mut current = values[0];
    for (index, value) in values.iter().enumerate().skip(1) {
        current += (*value - current) / (index as f64 + 1.0);
    }
    current.is_finite().then_some(current).ok_or_else(|| {
        DecisionFunctionalError::new("samples.mean", "online mean reconstruction is non-finite")
    })
}

fn hoeffding_radius(
    lower: f64,
    upper: f64,
    sample_size: usize,
    failure_probability: f64,
) -> Result<f64, DecisionFunctionalError> {
    if sample_size == 0 {
        return Err(DecisionFunctionalError::new(
            "samples",
            "sample size must be positive",
        ));
    }
    validate_open_probability(failure_probability, "confidence.failure_probability")?;
    let width = upper - lower;
    let factor = ((-failure_probability.ln()) / (2.0 * sample_size as f64)).sqrt();
    let raw = width * factor;
    if !raw.is_finite() || raw >= width {
        Ok(width)
    } else {
        Ok(raw.max(0.0))
    }
}

fn upper_bound(mean: f64, radius: f64, upper: f64) -> f64 {
    if radius >= upper - mean {
        upper
    } else {
        (mean + radius).min(upper)
    }
}

fn lower_bound(mean: f64, radius: f64, lower: f64) -> f64 {
    if radius >= mean - lower {
        lower
    } else {
        (mean - radius).max(lower)
    }
}

fn confidence_component(
    policy_id: &str,
    model_id: &str,
    functional: DecisionFunctionalKind,
    values: &[f64],
    lower: f64,
    upper: f64,
    failure_probability: f64,
    use_upper: bool,
) -> Result<DecisionFunctionalConfidenceComponent, DecisionFunctionalError> {
    let empirical_mean = mean(values)?;
    let radius = hoeffding_radius(lower, upper, values.len(), failure_probability)?;
    let (lower_confidence_bound, upper_confidence_bound) = if use_upper {
        (lower, upper_bound(empirical_mean, radius, upper))
    } else {
        (lower_bound(empirical_mean, radius, lower), upper)
    };
    Ok(DecisionFunctionalConfidenceComponent {
        policy_id: policy_id.into(),
        model_id: model_id.into(),
        functional,
        sample_size: values.len(),
        allocated_failure_probability: failure_probability,
        support_lower_bound: lower,
        support_upper_bound: upper,
        empirical_mean,
        confidence_radius: radius,
        lower_confidence_bound,
        upper_confidence_bound,
    })
}

fn conditional_alpha(alpha: f64, delta: f64) -> Result<f64, DecisionFunctionalError> {
    if delta >= alpha {
        return Err(DecisionFunctionalError::new(
            "request.null_family_coverage_failure_probability",
            "total null-side confidence failure must be strictly less than nominal alpha",
        ));
    }
    let result = (alpha - delta) / (1.0 - delta);
    if !result.is_finite() || !(0.0..1.0).contains(&result) {
        return Err(DecisionFunctionalError::new(
            "report.conditional_significance_level",
            "conditional alpha calculation is invalid",
        ));
    }
    Ok(result)
}

fn robustify_plan(
    nominal: &PolicyFrontierPlan,
    request: &DecisionFunctionalCalibrationRequest,
    components: &[DecisionFunctionalConfidenceComponent],
) -> Result<PolicyFrontierPlan, DecisionFunctionalError> {
    let mut plan = nominal.clone();
    plan.canonicalize()
        .map_err(DecisionFunctionalError::from_frontier)?;
    let source_sha256 = plan.plan_sha256().map_err(DecisionFunctionalError::from_frontier)?;
    if source_sha256 != request.source_plan_sha256 {
        return Err(DecisionFunctionalError::new(
            "request.source_plan_sha256",
            "calibration request is not bound to this canonical nominal plan",
        ));
    }

    let expected_null_rows = plan.policies.len() * plan.policies[0].null_models.len();
    let expected_alt_rows = plan.policies.len() * plan.policies[0].alternatives.len();
    if request.null_rows.len() != expected_null_rows
        || request.alternative_rows.len() != expected_alt_rows
    {
        return Err(DecisionFunctionalError::new(
            "request.rows",
            "calibration rows must cover exactly every policy × null/alternative model pair",
        ));
    }

    let policies = plan
        .policies
        .iter()
        .map(|policy| policy.policy_id.as_str())
        .collect::<BTreeSet<_>>();
    let null_models = plan.policies[0]
        .null_models
        .iter()
        .map(|row| row.model_id.as_str())
        .collect::<BTreeSet<_>>();
    let alternatives = plan.policies[0]
        .alternatives
        .iter()
        .map(|row| row.model_id.as_str())
        .collect::<BTreeSet<_>>();

    for row in &request.null_rows {
        if !policies.contains(row.policy_id.as_str())
            || !null_models.contains(row.model_id.as_str())
        {
            return Err(DecisionFunctionalError::new(
                "request.null_rows",
                "null calibration contains an undeclared policy or model",
            ));
        }
    }
    let null_keys = request
        .null_rows
        .iter()
        .map(|row| (row.policy_id.as_str(), row.model_id.as_str()))
        .collect::<BTreeSet<_>>();
    let expected_null_keys = policies
        .iter()
        .flat_map(|policy| null_models.iter().map(move |model| (*policy, *model)))
        .collect::<BTreeSet<_>>();
    if null_keys != expected_null_keys {
        return Err(DecisionFunctionalError::new(
            "request.null_rows",
            "null calibration does not cover the exact policy/model family",
        ));
    }

    for row in &request.alternative_rows {
        if !policies.contains(row.policy_id.as_str())
            || !alternatives.contains(row.model_id.as_str())
        {
            return Err(DecisionFunctionalError::new(
                "request.alternative_rows",
                "alternative calibration contains an undeclared policy or model",
            ));
        }
    }
    let alternative_keys = request
        .alternative_rows
        .iter()
        .map(|row| (row.policy_id.as_str(), row.model_id.as_str()))
        .collect::<BTreeSet<_>>();
    let expected_alternative_keys = policies
        .iter()
        .flat_map(|policy| alternatives.iter().map(move |model| (*policy, *model)))
        .collect::<BTreeSet<_>>();
    if alternative_keys != expected_alternative_keys {
        return Err(DecisionFunctionalError::new(
            "request.alternative_rows",
            "alternative calibration does not cover the exact policy/model family",
        ));
    }

    let null_upper = components
        .iter()
        .filter(|component| component.functional == DecisionFunctionalKind::NullRejection)
        .map(|component| {
            (
                (component.policy_id.as_str(), component.model_id.as_str()),
                component.upper_confidence_bound,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let alt_power_lower = components
        .iter()
        .filter(|component| component.functional == DecisionFunctionalKind::AlternativePower)
        .map(|component| {
            (
                (component.policy_id.as_str(), component.model_id.as_str()),
                component.lower_confidence_bound,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let alt_cost_upper = components
        .iter()
        .filter(|component| component.functional == DecisionFunctionalKind::EvaluatorCost)
        .map(|component| {
            (
                (component.policy_id.as_str(), component.model_id.as_str()),
                component.upper_confidence_bound,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let alt_utility_lower = components
        .iter()
        .filter(|component| component.functional == DecisionFunctionalKind::DecisionUtility)
        .map(|component| {
            (
                (component.policy_id.as_str(), component.model_id.as_str()),
                component.lower_confidence_bound,
            )
        })
        .collect::<BTreeMap<_, _>>();

    let alpha = plan.significance_level;
    plan.significance_level = conditional_alpha(
        alpha,
        request.null_family_coverage_failure_probability,
    )?;

    for policy in &mut plan.policies {
        for null_row in &mut policy.null_models {
            null_row.null_size = *null_upper
                .get(&(policy.policy_id.as_str(), null_row.model_id.as_str()))
                .ok_or_else(|| {
                    DecisionFunctionalError::new(
                        "components.null_rejection",
                        "missing simultaneous null upper confidence bound",
                    )
                })?;
        }
        for alt_row in &mut policy.alternatives {
            let key = (policy.policy_id.as_str(), alt_row.model_id.as_str());
            alt_row.rejection_power = *alt_power_lower.get(&key).ok_or_else(|| {
                DecisionFunctionalError::new(
                    "components.alternative_power",
                    "missing simultaneous alternative-power lower confidence bound",
                )
            })?;
            alt_row.expected_evaluator_cost = *alt_cost_upper.get(&key).ok_or_else(|| {
                DecisionFunctionalError::new(
                    "components.evaluator_cost",
                    "missing simultaneous evaluator-cost upper confidence bound",
                )
            })?;
            alt_row.decision_utility = *alt_utility_lower.get(&key).ok_or_else(|| {
                DecisionFunctionalError::new(
                    "components.decision_utility",
                    "missing simultaneous decision-utility lower confidence bound",
                )
            })?;
        }
    }

    plan.canonicalize()
        .map_err(DecisionFunctionalError::from_frontier)?;
    Ok(plan)
}

/// Calibrate all declared policy/model decision functionals and run the existing frontier LP.
///
/// The null-family delta is Bonferroni-allocated over every policy × null-model row.
/// The alternative-family delta is Bonferroni-allocated over every policy × alternative-model
/// row × three functionals. Calibration must be independent of confirmation and share the
/// confirmatory target law by design. V1 does not implement a transport correction.
pub fn analyze_decision_functional_frontier(
    nominal: &PolicyFrontierPlan,
    input_request: &DecisionFunctionalCalibrationRequest,
) -> Result<DecisionFunctionalCalibrationReport, DecisionFunctionalError> {
    let mut request = input_request.clone();
    request.canonicalize()?;
    let mut canonical_nominal = nominal.clone();
    canonical_nominal
        .canonicalize()
        .map_err(DecisionFunctionalError::from_frontier)?;
    let source_plan_sha256 = canonical_nominal
        .plan_sha256()
        .map_err(DecisionFunctionalError::from_frontier)?;
    if source_plan_sha256 != request.source_plan_sha256 {
        return Err(DecisionFunctionalError::new(
            "request.source_plan_sha256",
            "request is not bound to the supplied canonical nominal plan",
        ));
    }

    let policy_count = canonical_nominal.policies.len();
    let null_model_count = canonical_nominal.policies[0].null_models.len();
    let alternative_model_count = canonical_nominal.policies[0].alternatives.len();
    let null_row_count = policy_count * null_model_count;
    let alternative_functional_count = policy_count * alternative_model_count * 3;
    let null_delta_per_row =
        request.null_family_coverage_failure_probability / null_row_count as f64;
    let alternative_delta_per_functional =
        request.alternative_family_coverage_failure_probability
            / alternative_functional_count as f64;

    let mut components = Vec::with_capacity(null_row_count + alternative_functional_count);
    for row in &request.null_rows {
        components.push(confidence_component(
            &row.policy_id,
            &row.model_id,
            DecisionFunctionalKind::NullRejection,
            &row.rejection_values,
            0.0,
            1.0,
            null_delta_per_row,
            true,
        )?);
    }

    for row in &request.alternative_rows {
        let policy = canonical_nominal
            .policies
            .iter()
            .find(|policy| policy.policy_id == row.policy_id)
            .ok_or_else(|| {
                DecisionFunctionalError::new(
                    "request.alternative_rows.policy_id",
                    "policy is not in the nominal frontier",
                )
            })?;
        if row.utility_contract_sha256 != policy.utility_contract_sha256 {
            return Err(DecisionFunctionalError::new(
                format!(
                    "request.alternative_rows.{}.utility_contract_sha256",
                    row.policy_id
                ),
                "calibration utility/cost contract differs from the nominal policy receipt",
            ));
        }
        let nominal_alt = policy
            .alternatives
            .iter()
            .find(|alternative| alternative.model_id == row.model_id)
            .ok_or_else(|| {
                DecisionFunctionalError::new(
                    "request.alternative_rows.model_id",
                    "alternative model is not in the nominal frontier",
                )
            })?;
        if nominal_alt.expected_evaluator_cost < row.evaluator_cost_lower_bound
            || nominal_alt.expected_evaluator_cost > row.evaluator_cost_upper_bound
            || nominal_alt.decision_utility < row.decision_utility_lower_bound
            || nominal_alt.decision_utility > row.decision_utility_upper_bound
        {
            return Err(DecisionFunctionalError::new(
                format!("request.alternative_rows.{}.support_bounds", row.policy_id),
                "nominal cost/utility receipts must lie within declared path-level support bounds",
            ));
        }
        components.push(confidence_component(
            &row.policy_id,
            &row.model_id,
            DecisionFunctionalKind::AlternativePower,
            &row.rejection_values,
            0.0,
            1.0,
            alternative_delta_per_functional,
            false,
        )?);
        components.push(confidence_component(
            &row.policy_id,
            &row.model_id,
            DecisionFunctionalKind::EvaluatorCost,
            &row.evaluator_cost_values,
            row.evaluator_cost_lower_bound,
            row.evaluator_cost_upper_bound,
            alternative_delta_per_functional,
            true,
        )?);
        components.push(confidence_component(
            &row.policy_id,
            &row.model_id,
            DecisionFunctionalKind::DecisionUtility,
            &row.decision_utility_values,
            row.decision_utility_lower_bound,
            row.decision_utility_upper_bound,
            alternative_delta_per_functional,
            false,
        )?);
    }

    components.sort_by(|left, right| {
        (
            &left.policy_id,
            &left.model_id,
            left.functional as u8,
        )
            .cmp(&(
                &right.policy_id,
                &right.model_id,
                right.functional as u8,
            ))
    });

    let robustified_plan = robustify_plan(&canonical_nominal, &request, &components)?;
    let frontier = analyze_policy_frontier(&robustified_plan)
        .map_err(DecisionFunctionalError::from_frontier)?;
    let delta = request.null_family_coverage_failure_probability;
    let unconditional_type_i =
        delta + (1.0 - delta) * frontier.worst_case_null_size;
    if !unconditional_type_i.is_finite()
        || unconditional_type_i > canonical_nominal.significance_level + TOLERANCE
    {
        return Err(DecisionFunctionalError::new(
            "report.worst_case_unconditional_type_i_upper_bound",
            "reconstructed unconditional Type-I upper bound exceeds nominal alpha",
        ));
    }
    let request_sha256 = request.request_sha256()?;
    let robustified_sha256 = robustified_plan
        .plan_sha256()
        .map_err(DecisionFunctionalError::from_frontier)?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        calibration_id: &request.calibration_id,
        source_plan_sha256: &source_plan_sha256,
        request_sha256: &request_sha256,
        functional_contract_sha256: &request.functional_contract_sha256,
        calibration_target_design_sha256: &request.calibration_target_design_sha256,
        confirmatory_target_design_sha256: &request.confirmatory_target_design_sha256,
        evidence_mode: request.evidence_mode,
        target_relation: request.target_relation,
        nominal_significance_level: canonical_nominal.significance_level,
        conditional_significance_level: robustified_plan.significance_level,
        null_family_coverage_failure_probability: delta,
        alternative_family_coverage_failure_probability: request
            .alternative_family_coverage_failure_probability,
        worst_case_unconditional_type_i_upper_bound: unconditional_type_i,
        robustified_plan_sha256: &robustified_sha256,
        components: &components,
        frontier: &frontier,
    };
    let report_sha256 = digest(&payload)?;
    Ok(DecisionFunctionalCalibrationReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        calibration_id: request.calibration_id,
        source_plan_sha256,
        request_sha256,
        functional_contract_sha256: request.functional_contract_sha256,
        calibration_target_design_sha256: request.calibration_target_design_sha256,
        confirmatory_target_design_sha256: request.confirmatory_target_design_sha256,
        evidence_mode: request.evidence_mode,
        target_relation: request.target_relation,
        nominal_significance_level: canonical_nominal.significance_level,
        conditional_significance_level: robustified_plan.significance_level,
        null_family_coverage_failure_probability: delta,
        alternative_family_coverage_failure_probability: request
            .alternative_family_coverage_failure_probability,
        worst_case_unconditional_type_i_upper_bound: unconditional_type_i,
        robustified_plan_sha256: robustified_sha256,
        components,
        frontier,
        report_sha256,
    })
}

pub fn verify_decision_functional_report(
    nominal: &PolicyFrontierPlan,
    request: &DecisionFunctionalCalibrationRequest,
    report: &DecisionFunctionalCalibrationReport,
) -> Result<(), DecisionFunctionalError> {
    if analyze_decision_functional_frontier(nominal, request)? != *report {
        return Err(DecisionFunctionalError::new(
            "report",
            "stored decision-functional report differs from complete deterministic reconstruction",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, DecisionFunctionalError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| DecisionFunctionalError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const UTILITY_HASH: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const PLAN_HASH: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DESIGN_HASH: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const FUNCTION_HASH: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn policy(
        id: &str,
        null_size: f64,
        alt_a: (f64, f64, f64),
        alt_b: (f64, f64, f64),
    ) -> super::super::policy_frontier_lp::FrontierPolicyReceipt {
        super::super::policy_frontier_lp::FrontierPolicyReceipt {
            policy_id: id.into(),
            utility_contract_sha256: UTILITY_HASH.into(),
            null_models: vec![super::super::policy_frontier_lp::FrontierNullReceipt {
                model_id: "null-a".into(),
                null_size,
            }],
            alternatives: vec![
                super::super::policy_frontier_lp::FrontierAlternativeReceipt {
                    model_id: "alt-a".into(),
                    rejection_power: alt_a.0,
                    expected_evaluator_cost: alt_a.1,
                    decision_utility: alt_a.2,
                },
                FrontierAlternativeReceipt {
                    model_id: "alt-b".into(),
                    rejection_power: alt_b.0,
                    expected_evaluator_cost: alt_b.1,
                    decision_utility: alt_b.2,
                },
            ],
        }
    }

    fn nominal_plan() -> PolicyFrontierPlan {
        PolicyFrontierPlan::new(
            "dfce-frontier-fixture",
            0.05,
            Some(10.0),
            vec![
                policy("policy-a", 0.001, (0.8, 2.0, 6.0), (0.7, 3.0, 5.0)),
                policy("policy-b", 0.002, (0.7, 4.0, 7.0), (0.9, 5.0, 8.0)),
            ],
        )
        .unwrap()
    }

    fn units(prefix: &str, n: usize) -> Vec<String> {
        (0..n).map(|index| format!("{prefix}-{index}")).collect()
    }

    fn repeated(value: f64, n: usize) -> Vec<f64> {
        vec![value; n]
    }

    fn request(plan: &PolicyFrontierPlan) -> DecisionFunctionalCalibrationRequest {
        let n = 20_000;
        let mut null_rows = Vec::new();
        for (policy_id, value) in [("policy-a", 0.001), ("policy-b", 0.002)] {
            null_rows.push(NullPolicyModelSamples {
                policy_id: policy_id.into(),
                model_id: "null-a".into(),
                sampling_unit_ids: units(policy_id, n),
                rejection_values: repeated(value, n),
                assignment_probability: 0.5,
            });
        }
        let mut alternative_rows = Vec::new();
        for policy_id in ["policy-a", "policy-b"] {
            for (model_id, power, cost, utility) in [
                ("alt-a", 0.80, 2.0, 6.0),
                ("alt-b", 0.85, 3.0, 7.0),
            ] {
                alternative_rows.push(AlternativePolicyModelSamples {
                    policy_id: policy_id.into(),
                    model_id: model_id.into(),
                    utility_contract_sha256: UTILITY_HASH.into(),
                    sampling_unit_ids: units(&format!("{policy_id}-{model_id}"), n),
                    rejection_values: repeated(power, n),
                    evaluator_cost_values: repeated(cost, n),
                    decision_utility_values: repeated(utility, n),
                    evaluator_cost_lower_bound: 0.0,
                    evaluator_cost_upper_bound: 10.0,
                    decision_utility_lower_bound: -5.0,
                    decision_utility_upper_bound: 10.0,
                    assignment_probability: 0.5,
                });
            }
        }
        DecisionFunctionalCalibrationRequest::new(
            "dfce-calibration-fixture",
            plan.plan_sha256().unwrap(),
            FUNCTION_HASH,
            DESIGN_HASH,
            DESIGN_HASH,
            CalibrationTargetRelation::SameTargetLawByDesign,
            CalibrationEvidenceMode::RandomizedPolicyExecution,
            0.005,
            0.05,
            true,
            true,
            true,
            null_rows,
            alternative_rows,
        )
        .unwrap()
    }

    #[test]
    fn computes_simultaneous_bounds_and_unconditional_type_i_budget() {
        let plan = nominal_plan();
        let request = request(&plan);
        let report = analyze_decision_functional_frontier(&plan, &request).unwrap();
        assert_eq!(report.components.len(), 14);
        assert!((report.conditional_significance_level - (0.05 - 0.005) / 0.995).abs() < 1e-12);
        assert!(report.worst_case_unconditional_type_i_upper_bound <= 0.05 + TOLERANCE);
        assert!(report.components.iter().filter(|component| {
            component.functional == DecisionFunctionalKind::NullRejection
        }).all(|component| component.upper_confidence_bound < report.conditional_significance_level));
        verify_decision_functional_report(&plan, &request, &report).unwrap();
    }

    #[test]
    fn zero_width_functional_has_zero_radius() {
        let radius = hoeffding_radius(3.0, 3.0, 10, 0.01).unwrap();
        assert_eq!(radius, 0.0);
    }

    #[test]
    fn rejects_unknown_or_transport_target_relation() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.target_relation = CalibrationTargetRelation::Unknown;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
        request.target_relation = CalibrationTargetRelation::BoundedTransport;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_independent_but_differently_designed_confirmation() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.confirmatory_target_design_sha256 = PLAN_HASH.into();
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_off_policy_without_a_separate_weighted_estimator() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.evidence_mode = CalibrationEvidenceMode::OffPolicyEstimate;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_calibration_dependence_and_non_independent_units() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.calibration_independent_of_confirmatory_stream = false;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
        request.calibration_independent_of_confirmatory_stream = true;
        request.independent_sampling_units_declared = false;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_randomized_assignment_without_full_arm_support() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.null_rows[0].assignment_probability = 0.2;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_sampling_units_reused_across_randomized_policy_arms() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.null_rows[1].sampling_unit_ids[0] =
            request.null_rows[0].sampling_unit_ids[0].clone();
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_duplicate_sampling_units() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.null_rows[0].sampling_unit_ids[1] =
            request.null_rows[0].sampling_unit_ids[0].clone();
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_missing_policy_model_rows() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.alternative_rows.pop();
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_out_of_range_and_non_finite_observations() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.alternative_rows[0].decision_utility_values[0] = f64::NAN;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
        request.alternative_rows[0].decision_utility_values[0] = 100.0;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn rejects_null_coverage_failure_that_consumes_alpha() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        request.null_family_coverage_failure_probability = 0.05;
        assert!(analyze_decision_functional_frontier(&plan, &request).is_err());
    }

    #[test]
    fn reordered_rows_reconstruct_identical_report() {
        let plan = nominal_plan();
        let mut request = request(&plan);
        let first = analyze_decision_functional_frontier(&plan, &request).unwrap();
        request.null_rows.reverse();
        request.alternative_rows.reverse();
        let second = analyze_decision_functional_frontier(&plan, &request).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn report_tampering_is_detected() {
        let plan = nominal_plan();
        let request = request(&plan);
        let mut report = analyze_decision_functional_frontier(&plan, &request).unwrap();
        report.worst_case_unconditional_type_i_upper_bound += 0.001;
        assert!(verify_decision_functional_report(&plan, &request, &report).is_err());
    }
}
