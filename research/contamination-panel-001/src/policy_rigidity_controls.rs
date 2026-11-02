//! Matched controls for separating item-specific response-path rigidity from
//! global RL-induced entropy collapse and intrinsic task determinism.
//!
//! The caller supplies one score bundle per independent source/task cluster.
//! The module does not load models or create prompts. It decomposes each bundle
//! into (1) baseline entropy, (2) task-wide invariance to non-response-conditioned
//! interventions, and (3) excess invariance specifically under answer-conditioned
//! Self-Critique. Every axis is calibrated against disjoint clean controls within
//! the same checkpoint × family × difficulty × path-multiplicity group.

use crate::hash_domain;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const POLICY_RIGIDITY_REQUEST_SCHEMA: &str =
    "symthaea.policy_rigidity_controls_request.v1";
pub const POLICY_RIGIDITY_REPORT_SCHEMA: &str =
    "symthaea.policy_rigidity_controls_report.v1";
pub const PROBE_REGISTRY_SCHEMA: &str = "symthaea.policy_rigidity_probe_registry.v1";
pub const POLICY_RIGIDITY_FUSION_ID: &str =
    "matched_clean_control_empirical_ranks_by_v1";
pub const MIN_CLEAN_CONTROL_UNITS_PER_GROUP: usize = 2;
pub const MAX_CLEAN_CONTROL_UNITS_PER_GROUP: usize = 100_000;
pub const MAX_EVALUATION_UNITS: usize = 1_000_000;
pub const MAX_PROBE_SPECS: usize = 16;
pub const MIN_PATH_ADJUDICATORS: u16 = 3;
pub const MIN_PATH_AGREEMENT_PERMILLE: u16 = 900;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PathMultiplicityV1 {
    SingleOrFewValidPaths,
    MultipleValidPaths,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExposureLabelV1 {
    KnownClean,
    KnownExposed,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ProbeInterventionV1 {
    ResponseConditionedSelfCritique,
    PromptOnlyUnfamiliarMethod,
    SemanticPreservingParaphrase,
    ConstraintPreservingReparameterization,
    AlternateValidMethod,
}

impl ProbeInterventionV1 {
    pub const REQUIRED: [Self; 5] = [
        Self::ResponseConditionedSelfCritique,
        Self::PromptOnlyUnfamiliarMethod,
        Self::SemanticPreservingParaphrase,
        Self::ConstraintPreservingReparameterization,
        Self::AlternateValidMethod,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::ResponseConditionedSelfCritique => "response_conditioned_self_critique",
            Self::PromptOnlyUnfamiliarMethod => "prompt_only_unfamiliar_method",
            Self::SemanticPreservingParaphrase => "semantic_preserving_paraphrase",
            Self::ConstraintPreservingReparameterization => {
                "constraint_preserving_reparameterization"
            }
            Self::AlternateValidMethod => "alternate_valid_method",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeSpecV1 {
    pub probe_id: String,
    pub intervention: ProbeInterventionV1,
    pub prompt_contract_sha256: String,
    pub generation_contract_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PathMultiplicityAdjudicationV1 {
    pub classification: PathMultiplicityV1,
    pub adjudication_report_sha256: String,
    pub reviewer_panel_sha256: String,
    pub reviewer_count: u16,
    pub agreement_permille: u16,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RigidityProbeScoresV1 {
    /// Mean token entropy of the baseline greedy response, in nats.
    pub baseline_mean_entropy_nats: f64,
    /// Similarity of baseline and answer-conditioned Self-Critique entropy paths.
    /// Higher means the response-conditioned intervention preserves the path.
    pub response_conditioned_similarity: f64,
    /// Similarity of baseline and prompt-only unfamiliar-method entropy paths.
    pub prompt_only_similarity: f64,
    /// Similarity of baseline and semantic-preserving paraphrase entropy paths.
    pub semantic_paraphrase_similarity: f64,
    /// Similarity of baseline and constraint-preserving reparameterization paths.
    pub constraint_reparameterization_similarity: f64,
    /// Required for tasks adjudicated to have multiple valid solution paths.
    /// Must be absent for single/few-path tasks, where this probe is not meaningful.
    pub alternate_valid_method_similarity: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CleanControlRecordV1 {
    pub calibration_unit_id: String,
    pub source_cluster_sha256: String,
    pub task_family: String,
    pub difficulty_band: String,
    pub path_multiplicity: PathMultiplicityAdjudicationV1,
    pub probe_receipt_manifest_sha256: String,
    pub task_recipe_sha256: String,
    pub exposure_label: ExposureLabelV1,
    pub scores: RigidityProbeScoresV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationControlRecordV1 {
    pub evaluation_unit_id: String,
    pub source_cluster_sha256: String,
    pub task_family: String,
    pub difficulty_band: String,
    pub path_multiplicity: PathMultiplicityAdjudicationV1,
    pub probe_receipt_manifest_sha256: String,
    pub task_recipe_sha256: String,
    pub exposure_label: Option<ExposureLabelV1>,
    pub scores: RigidityProbeScoresV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRigidityControlsRequestV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    /// One request represents one exact model/checkpoint regime.
    pub model_regime: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub prompt_harness_sha256: String,
    pub runtime_environment_sha256: String,
    pub calibration_protocol_sha256: String,
    pub probe_registry_sha256: String,
    pub target_alpha_permille: u16,
    pub probe_specs: Vec<ProbeSpecV1>,
    pub clean_controls: Vec<CleanControlRecordV1>,
    pub evaluation: Vec<EvaluationControlRecordV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct RigidityGroupKey {
    task_family: String,
    difficulty_band: String,
    path_multiplicity: PathMultiplicityV1,
}

#[derive(Debug, Clone, Copy)]
struct DecomposedRigidity {
    baseline_mean_entropy_nats: f64,
    global_path_rigidity_score: f64,
    response_conditioned_excess: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RigidityScoreRowV1 {
    pub evaluation_unit_id: String,
    pub task_family: String,
    pub difficulty_band: String,
    pub path_multiplicity: PathMultiplicityV1,
    pub exposure_label: Option<ExposureLabelV1>,
    pub baseline_mean_entropy_nats: Option<f64>,
    pub global_path_rigidity_score: Option<f64>,
    pub response_conditioned_excess: Option<f64>,
    /// Lower-tail empirical p-value: unexpectedly low baseline entropy
    /// relative to matched clean controls.
    pub global_entropy_lower_tail_p: Option<f64>,
    /// Upper-tail empirical p-value: unusually high invariance under
    /// non-response-conditioned interventions.
    pub global_rigidity_upper_tail_p: Option<f64>,
    /// Upper-tail empirical p-value: response-conditioned invariance
    /// beyond the matched alternative-intervention envelope.
    pub response_conditioned_excess_upper_tail_p: Option<f64>,
    pub by_adjusted_excess_p: Option<f64>,
    pub by_family_size: Option<u64>,
    pub minimum_clean_rank_p_value: Option<f64>,
    pub decision: RigidityDecisionV1,
    /// Descriptive decomposition, not a membership verdict.
    pub pattern: RigidityPatternV1,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RigidityDecisionV1 {
    FlagResponseConditionedExcess,
    NoResponseConditionedExcessFlag,
    AbstainIndeterminatePathMultiplicity,
    AbstainUnsupportedCleanControlGroup,
    AbstainInsufficientCalibrationResolution,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RigidityPatternV1 {
    BroadPolicyRigidity,
    TaskOrPathRigidity,
    ResponseConditionedSpecificity,
    NoMarkedRigidityPattern,
    InsufficientEvidence,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RigidityGroupSummaryV1 {
    pub task_family: String,
    pub difficulty_band: String,
    pub path_multiplicity: PathMultiplicityV1,
    pub clean_control_unit_count: u64,
    pub evaluation_unit_count: u64,
    pub eligible_unit_count: u64,
    pub flagged_response_conditioned_excess_count: u64,
    pub no_flag_count: u64,
    pub abstained_count: u64,
    pub known_clean_count: u64,
    pub known_exposed_count: u64,
    pub false_positive_count: u64,
    pub true_positive_count: u64,
    pub false_negative_count: u64,
    pub empirical_fpr_permille: Option<f64>,
    pub empirical_tpr_permille: Option<f64>,
    pub decision_coverage_permille: f64,
    pub minimum_clean_rank_p_value: f64,
    pub target_alpha_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRigidityControlsReportV1 {
    pub schema: String,
    pub method_id: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_regime: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub prompt_harness_sha256: String,
    pub runtime_environment_sha256: String,
    pub calibration_protocol_sha256: String,
    pub probe_registry_sha256: String,
    pub request_sha256: String,
    pub clean_control_manifest_sha256: String,
    pub evaluation_manifest_sha256: String,
    pub target_alpha_permille: u16,
    pub multiple_testing_method: String,
    pub fdr_control_conditional_on_exchangeability: bool,
    pub evaluation_unit_count: u64,
    pub flagged_count: u64,
    pub no_flag_count: u64,
    pub abstained_count: u64,
    pub score_rows: Vec<RigidityScoreRowV1>,
    pub group_summaries: Vec<RigidityGroupSummaryV1>,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub assumptions: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Copy)]
struct CleanGroupCalibration {
    baseline_entropy: Vec<f64>,
    global_rigidity: Vec<f64>,
    response_excess: Vec<f64>,
}

fn valid_id(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn validate_adjudication(
    adjudication: &PathMultiplicityAdjudicationV1,
    label: &str,
) -> Result<(), String> {
    require_digest(
        &adjudication.adjudication_report_sha256,
        &format!("{label} adjudication_report_sha256"),
    )?;
    require_digest(
        &adjudication.reviewer_panel_sha256,
        &format!("{label} reviewer_panel_sha256"),
    )?;
    if adjudication.reviewer_count < MIN_PATH_ADJUDICATORS || adjudication.reviewer_count > 20 {
        return Err(format!(
            "{label} reviewer_count must be in {MIN_PATH_ADJUDICATORS}..=20"
        ));
    }
    if adjudication.agreement_permille > 1_000 {
        return Err(format!("{label} agreement_permille must be <=1000"));
    }
    if adjudication.classification != PathMultiplicityV1::Indeterminate
        && adjudication.agreement_permille < MIN_PATH_AGREEMENT_PERMILLE
    {
        return Err(format!(
            "{label} accepted path-multiplicity label requires >= {MIN_PATH_AGREEMENT_PERMILLE} permille agreement"
        ));
    }
    Ok(())
}

fn validate_scores(
    scores: &RigidityProbeScoresV1,
    multiplicity: PathMultiplicityV1,
    label: &str,
) -> Result<(), String> {
    if !scores.baseline_mean_entropy_nats.is_finite()
        || scores.baseline_mean_entropy_nats < 0.0
    {
        return Err(format!("{label} baseline mean entropy must be finite and non-negative"));
    }
    for (name, value) in [
        ("response_conditioned_similarity", scores.response_conditioned_similarity),
        ("prompt_only_similarity", scores.prompt_only_similarity),
        ("semantic_paraphrase_similarity", scores.semantic_paraphrase_similarity),
        (
            "constraint_reparameterization_similarity",
            scores.constraint_reparameterization_similarity,
        ),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("{label} {name} must be finite and in [0,1]"));
        }
    }
    match (multiplicity, scores.alternate_valid_method_similarity) {
        (PathMultiplicityV1::MultipleValidPaths, Some(value))
            if value.is_finite() && (0.0..=1.0).contains(&value) => {}
        (PathMultiplicityV1::MultipleValidPaths, _) => {
            return Err(format!(
                "{label} multiple-valid-path tasks require a finite alternate_valid_method_similarity in [0,1]"
            ));
        }
        (PathMultiplicityV1::SingleOrFewValidPaths, None) => {}
        (PathMultiplicityV1::SingleOrFewValidPaths, Some(_)) => {
            return Err(format!(
                "{label} single/few-path tasks must not include the alternate-valid-method probe"
            ));
        }
        (PathMultiplicityV1::Indeterminate, Some(value))
            if !value.is_finite() || !(0.0..=1.0).contains(&value) =>
        {
            return Err(format!(
                "{label} alternate_valid_method_similarity must be finite and in [0,1]"
            ));
        }
        (PathMultiplicityV1::Indeterminate, _) => {}
    }
    Ok(())
}

fn decompose(scores: &RigidityProbeScoresV1) -> Result<DecomposedRigidity, String> {
    let mut alternative_scores = vec![
        scores.prompt_only_similarity,
        scores.semantic_paraphrase_similarity,
        scores.constraint_reparameterization_similarity,
    ];
    if let Some(value) = scores.alternate_valid_method_similarity {
        alternative_scores.push(value);
    }
    if alternative_scores.is_empty() {
        return Err("at least one non-response-conditioned probe is required".to_owned());
    }
    let global_path_rigidity_score =
        alternative_scores.iter().sum::<f64>() / alternative_scores.len() as f64;
    let response_conditioned_excess =
        scores.response_conditioned_similarity - global_path_rigidity_score;
    if !global_path_rigidity_score.is_finite() || !response_conditioned_excess.is_finite() {
        return Err("rigidity decomposition produced a non-finite value".to_owned());
    }
    Ok(DecomposedRigidity {
        baseline_mean_entropy_nats: scores.baseline_mean_entropy_nats,
        global_path_rigidity_score,
        response_conditioned_excess,
    })
}

fn upper_tail_p(reference: &[f64], candidate: f64) -> Result<f64, String> {
    if reference.is_empty() || !candidate.is_finite() || reference.iter().any(|v| !v.is_finite()) {
        return Err("upper-tail empirical rank requires finite nonempty reference scores".to_owned());
    }
    let extreme = reference.iter().filter(|value| **value >= candidate).count() as u64;
    Ok((1 + extreme) as f64 / (reference.len() as f64 + 1.0))
}

fn lower_tail_p(reference: &[f64], candidate: f64) -> Result<f64, String> {
    if reference.is_empty() || !candidate.is_finite() || reference.iter().any(|v| !v.is_finite()) {
        return Err("lower-tail empirical rank requires finite nonempty reference scores".to_owned());
    }
    let extreme = reference.iter().filter(|value| **value <= candidate).count() as u64;
    Ok((1 + extreme) as f64 / (reference.len() as f64 + 1.0))
}

fn canonical_probe_registry(specs: &[ProbeSpecV1]) -> Result<Vec<u8>, String> {
    let mut normalized = specs.to_vec();
    normalized.sort_by(|left, right| left.intervention.cmp(&right.intervention));
    serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize policy-rigidity probe registry: {error}"))
}

pub fn compute_probe_registry_sha256(specs: &[ProbeSpecV1]) -> Result<String, String> {
    validate_probe_registry(specs)?;
    let bytes = canonical_probe_registry(specs)?;
    Ok(hash_domain(b"policy-rigidity-probe-registry.v1", &bytes))
}

fn validate_probe_registry(specs: &[ProbeSpecV1]) -> Result<(), String> {
    if specs.len() != ProbeInterventionV1::REQUIRED.len()
        || specs.len() > MAX_PROBE_SPECS
    {
        return Err(format!(
            "probe registry must contain exactly {} interventions",
            ProbeInterventionV1::REQUIRED.len()
        ));
    }
    let mut interventions = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for spec in specs {
        if !valid_id(&spec.probe_id, 128)
            || spec.probe_id != spec.intervention.id()
            || !ids.insert(spec.probe_id.as_str())
            || !interventions.insert(spec.intervention)
        {
            return Err("probe IDs must uniquely match their canonical intervention IDs".to_owned());
        }
        require_digest(&spec.prompt_contract_sha256, "probe prompt_contract_sha256")?;
        require_digest(&spec.generation_contract_sha256, "probe generation_contract_sha256")?;
    }
    if ProbeInterventionV1::REQUIRED
        .iter()
        .any(|required| !interventions.contains(required))
    {
        return Err("probe registry is missing a required intervention".to_owned());
    }
    Ok(())
}

fn validate_record_common(
    unit_id: &str,
    source_cluster_sha256: &str,
    task_family: &str,
    difficulty_band: &str,
    adjudication: &PathMultiplicityAdjudicationV1,
    probe_receipt_manifest_sha256: &str,
    task_recipe_sha256: &str,
    scores: &RigidityProbeScoresV1,
    label: &str,
) -> Result<(), String> {
    if !valid_id(unit_id, 128)
        || !valid_id(task_family, 64)
        || !valid_id(difficulty_band, 64)
    {
        return Err(format!("{label} has invalid unit/family/difficulty identifiers"));
    }
    require_digest(source_cluster_sha256, &format!("{label} source_cluster_sha256"))?;
    require_digest(
        probe_receipt_manifest_sha256,
        &format!("{label} probe_receipt_manifest_sha256"),
    )?;
    require_digest(task_recipe_sha256, &format!("{label} task_recipe_sha256"))?;
    validate_adjudication(adjudication, label)?;
    validate_scores(scores, adjudication.classification, label)
}

fn canonical_clean_manifest(rows: &[CleanControlRecordV1]) -> Result<Vec<u8>, String> {
    let mut normalized = rows.to_vec();
    normalized.sort_by(|left, right| left.calibration_unit_id.cmp(&right.calibration_unit_id));
    serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize clean control manifest: {error}"))
}

fn canonical_evaluation_manifest(rows: &[EvaluationControlRecordV1]) -> Result<Vec<u8>, String> {
    let mut normalized = rows.to_vec();
    normalized.sort_by(|left, right| left.evaluation_unit_id.cmp(&right.evaluation_unit_id));
    serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize evaluation manifest: {error}"))
}

fn group_key(
    family: &str,
    difficulty: &str,
    multiplicity: PathMultiplicityV1,
) -> RigidityGroupKey {
    RigidityGroupKey {
        task_family: family.to_owned(),
        difficulty_band: difficulty.to_owned(),
        path_multiplicity: multiplicity,
    }
}

fn benjamini_yekutieli_adjustment(
    values: &[(String, f64)],
) -> Result<BTreeMap<String, f64>, String> {
    if values.is_empty() {
        return Ok(BTreeMap::new());
    }
    if values.iter().any(|(_, p)| !p.is_finite() || !(0.0..=1.0).contains(p)) {
        return Err("BY adjustment requires finite p-values in [0,1]".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.1.total_cmp(&right.1).then_with(|| left.0.cmp(&right.0)));
    let m = sorted.len();
    let harmonic = (1..=m).map(|rank| 1.0 / rank as f64).sum::<f64>();
    let multiplier = m as f64 * harmonic;
    let mut result = BTreeMap::new();
    let mut reverse_min = 1.0_f64;
    for index in (0..m).rev() {
        let rank = index + 1;
        reverse_min = reverse_min.min(sorted[index].1 * multiplier / rank as f64).min(1.0);
        result.insert(sorted[index].0.clone(), reverse_min);
    }
    Ok(result)
}

fn pattern_from_diagnostics(
    global_entropy_p: f64,
    global_rigidity_p: f64,
    response_excess_p: f64,
    alpha: f64,
) -> RigidityPatternV1 {
    let global_entropy_low = global_entropy_p <= alpha;
    let global_rigidity_high = global_rigidity_p <= alpha;
    let response_specific = response_excess_p <= alpha;
    match (global_entropy_low || global_rigidity_high, response_specific) {
        (true, true) => RigidityPatternV1::BroadPolicyRigidity,
        (true, false) => RigidityPatternV1::TaskOrPathRigidity,
        (false, true) => RigidityPatternV1::ResponseConditionedSpecificity,
        (false, false) => RigidityPatternV1::NoMarkedRigidityPattern,
    }
}

/// Score each evaluation row against disjoint clean controls matched on exact
/// model/checkpoint, task family, difficulty band and adjudicated path multiplicity.
/// BY is applied to response-conditioned-excess p-values within each group.
pub fn analyze_policy_rigidity_controls(
    request: &PolicyRigidityControlsRequestV1,
) -> Result<PolicyRigidityControlsReportV1, String> {
    if request.schema != POLICY_RIGIDITY_REQUEST_SCHEMA {
        return Err(format!("schema must be {POLICY_RIGIDITY_REQUEST_SCHEMA}"));
    }
    for (value, label, limit) in [
        (request.study_id.as_str(), "study_id", 128),
        (request.benchmark_id.as_str(), "benchmark_id", 128),
        (request.benchmark_version.as_str(), "benchmark_version", 64),
        (request.model_regime.as_str(), "model_regime", 128),
    ] {
        if !valid_id(value, limit) {
            return Err(format!("{label} must be a valid opaque identifier"));
        }
    }
    for (digest, label) in [
        (&request.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&request.tokenizer_sha256, "tokenizer_sha256"),
        (&request.prompt_harness_sha256, "prompt_harness_sha256"),
        (&request.runtime_environment_sha256, "runtime_environment_sha256"),
        (&request.calibration_protocol_sha256, "calibration_protocol_sha256"),
        (&request.probe_registry_sha256, "probe_registry_sha256"),
    ] {
        require_digest(digest, label)?;
    }
    if !(1..=500).contains(&request.target_alpha_permille) {
        return Err("target_alpha_permille must be in [1,500]".to_owned());
    }
    validate_probe_registry(&request.probe_specs)?;
    if compute_probe_registry_sha256(&request.probe_specs)? != request.probe_registry_sha256 {
        return Err("probe_registry_sha256 does not match the pinned probe_specs".to_owned());
    }
    if request.clean_controls.len() < MIN_CLEAN_CONTROL_UNITS_PER_GROUP
        || request.clean_controls.len() > MAX_CLEAN_CONTROL_UNITS_PER_GROUP * 10
        || request.evaluation.is_empty()
        || request.evaluation.len() > MAX_EVALUATION_UNITS
    {
        return Err("clean-control/evaluation cohort sizes are empty or exceed limits".to_owned());
    }

    let mut all_unit_ids = BTreeSet::new();
    let mut all_source_clusters = BTreeSet::new();
    let mut clean_by_group = BTreeMap::<RigidityGroupKey, Vec<DecomposedRigidity>>::new();
    for row in &request.clean_controls {
        validate_record_common(
            &row.calibration_unit_id,
            &row.source_cluster_sha256,
            &row.task_family,
            &row.difficulty_band,
            &row.path_multiplicity,
            &row.probe_receipt_manifest_sha256,
            &row.task_recipe_sha256,
            &row.scores,
            "clean-control row",
        )?;
        if row.exposure_label != ExposureLabelV1::KnownClean {
            return Err("every Calibration-A rigidity control must be known clean".to_owned());
        }
        if row.path_multiplicity.classification == PathMultiplicityV1::Indeterminate {
            return Err("clean controls with indeterminate path multiplicity cannot calibrate a scored group".to_owned());
        }
        if !all_unit_ids.insert(row.calibration_unit_id.as_str()) {
            return Err("clean-control unit IDs must be unique".to_owned());
        }
        if !all_source_clusters.insert(row.source_cluster_sha256.as_str()) {
            return Err("source-cluster digests must be unique across all splits".to_owned());
        }
        let key = group_key(
            &row.task_family,
            &row.difficulty_band,
            row.path_multiplicity.classification,
        );
        clean_by_group
            .entry(key)
            .or_default()
            .push(decompose(&row.scores)?);
    }

    let mut seen_eval_ids = BTreeSet::new();
    let mut evaluation_groups = BTreeMap::<RigidityGroupKey, Vec<&EvaluationControlRecordV1>>::new();
    for row in &request.evaluation {
        validate_record_common(
            &row.evaluation_unit_id,
            &row.source_cluster_sha256,
            &row.task_family,
            &row.difficulty_band,
            &row.path_multiplicity,
            &row.probe_receipt_manifest_sha256,
            &row.task_recipe_sha256,
            &row.scores,
            "evaluation row",
        )?;
        if !seen_eval_ids.insert(row.evaluation_unit_id.as_str())
            || all_unit_ids.contains(row.evaluation_unit_id.as_str())
        {
            return Err("evaluation unit IDs must be unique and disjoint from clean controls".to_owned());
        }
        if !all_source_clusters.insert(row.source_cluster_sha256.as_str()) {
            return Err("source-cluster digests must be unique across clean and evaluation splits".to_owned());
        }
        evaluation_groups
            .entry(group_key(
                &row.task_family,
                &row.difficulty_band,
                row.path_multiplicity.classification,
            ))
            .or_default()
            .push(row);
    }

    for (key, clean) in &clean_by_group {
        if clean.len() > MAX_CLEAN_CONTROL_UNITS_PER_GROUP {
            return Err(format!(
                "clean-control group {:?} exceeds {MAX_CLEAN_CONTROL_UNITS_PER_GROUP} units",
                key
            ));
        }
    }

    let mut score_rows = Vec::with_capacity(request.evaluation.len());
    let mut group_summaries = Vec::new();
    for (key, rows) in &evaluation_groups {
        let Some(clean) = clean_by_group.get(key) else {
            for row in rows {
                score_rows.push(RigidityScoreRowV1 {
                    evaluation_unit_id: row.evaluation_unit_id.clone(),
                    task_family: row.task_family.clone(),
                    difficulty_band: row.difficulty_band.clone(),
                    path_multiplicity: row.path_multiplicity.classification,
                    exposure_label: row.exposure_label,
                    baseline_mean_entropy_nats: None,
                    global_path_rigidity_score: None,
                    response_conditioned_excess: None,
                    global_entropy_lower_tail_p: None,
                    global_rigidity_upper_tail_p: None,
                    response_conditioned_excess_upper_tail_p: None,
                    by_adjusted_excess_p: None,
                    by_family_size: None,
                    minimum_clean_rank_p_value: None,
                    decision: if row.path_multiplicity.classification == PathMultiplicityV1::Indeterminate {
                        RigidityDecisionV1::AbstainIndeterminatePathMultiplicity
                    } else {
                        RigidityDecisionV1::AbstainUnsupportedCleanControlGroup
                    },
                    pattern: RigidityPatternV1::InsufficientEvidence,
                });
            }
            group_summaries.push(RigidityGroupSummaryV1 {
                task_family: key.task_family.clone(),
                difficulty_band: key.difficulty_band.clone(),
                path_multiplicity: key.path_multiplicity,
                clean_control_unit_count: 0,
                evaluation_unit_count: rows.len() as u64,
                eligible_unit_count: 0,
                flagged_response_conditioned_excess_count: 0,
                no_flag_count: 0,
                abstained_count: rows.len() as u64,
                known_clean_count: rows.iter().filter(|r| r.exposure_label == Some(ExposureLabelV1::KnownClean)).count() as u64,
                known_exposed_count: rows.iter().filter(|r| r.exposure_label == Some(ExposureLabelV1::KnownExposed)).count() as u64,
                false_positive_count: 0,
                true_positive_count: 0,
                false_negative_count: rows.iter().filter(|r| r.exposure_label == Some(ExposureLabelV1::KnownExposed)).count() as u64,
                empirical_fpr_permille: None,
                empirical_tpr_permille: None,
                decision_coverage_permille: 0.0,
                minimum_clean_rank_p_value: 1.0,
                target_alpha_permille: request.target_alpha_permille,
            });
            continue;
        };
        if clean.len() < MIN_CLEAN_CONTROL_UNITS_PER_GROUP {
            return Err("each matched clean-control group requires at least two units".to_owned());
        }
        let baseline_reference: Vec<f64> = clean.iter().map(|v| v.baseline_mean_entropy_nats).collect();
        let global_reference: Vec<f64> = clean.iter().map(|v| v.global_path_rigidity_score).collect();
        let excess_reference: Vec<f64> = clean.iter().map(|v| v.response_conditioned_excess).collect();
        let min_p = 1.0 / (clean.len() as f64 + 1.0);
        let alpha_resolvable =
            u128::from(request.target_alpha_permille) * (clean.len() as u128 + 1) >= 1_000;
        let alpha = f64::from(request.target_alpha_permille) / 1_000.0;
        let mut flagged = 0_u64;
        let mut no_flag = 0_u64;
        let mut abstained = 0_u64;
        let mut clean_labelled = 0_u64;
        let mut exposed_labelled = 0_u64;
        let mut fp = 0_u64;
        let mut tp = 0_u64;
        let mut fn_count = 0_u64;

        for row in rows {
            let decomposed = decompose(&row.scores)?;
            let entropy_p = lower_tail_p(&baseline_reference, decomposed.baseline_mean_entropy_nats)?;
            let global_p = upper_tail_p(&global_reference, decomposed.global_path_rigidity_score)?;
            let excess_p = upper_tail_p(&excess_reference, decomposed.response_conditioned_excess)?;
            let pattern = pattern_from_diagnostics(entropy_p, global_p, excess_p, alpha);
            let decision = if row.path_multiplicity.classification == PathMultiplicityV1::Indeterminate {
                abstained += 1;
                RigidityDecisionV1::AbstainIndeterminatePathMultiplicity
            } else if !alpha_resolvable {
                abstained += 1;
                RigidityDecisionV1::AbstainInsufficientCalibrationResolution
            } else {
                no_flag += 1;
                RigidityDecisionV1::NoResponseConditionedExcessFlag
            };
            match row.exposure_label {
                Some(ExposureLabelV1::KnownClean) => clean_labelled += 1,
                Some(ExposureLabelV1::KnownExposed) => exposed_labelled += 1,
                Some(ExposureLabelV1::Unknown) | None => {}
            }
            score_rows.push(RigidityScoreRowV1 {
                evaluation_unit_id: row.evaluation_unit_id.clone(),
                task_family: row.task_family.clone(),
                difficulty_band: row.difficulty_band.clone(),
                path_multiplicity: row.path_multiplicity.classification,
                exposure_label: row.exposure_label,
                baseline_mean_entropy_nats: Some(decomposed.baseline_mean_entropy_nats),
                global_path_rigidity_score: Some(decomposed.global_path_rigidity_score),
                response_conditioned_excess: Some(decomposed.response_conditioned_excess),
                global_entropy_lower_tail_p: Some(entropy_p),
                global_rigidity_upper_tail_p: Some(global_p),
                response_conditioned_excess_upper_tail_p: Some(excess_p),
                by_adjusted_excess_p: None,
                by_family_size: None,
                minimum_clean_rank_p_value: Some(min_p),
                decision,
                pattern: if decision == RigidityDecisionV1::AbstainInsufficientCalibrationResolution {
                    RigidityPatternV1::InsufficientEvidence
                } else {
                    pattern
                },
            });
        }

        let start = score_rows.len().saturating_sub(rows.len());
        let eligible_pairs: Vec<(String, f64)> = score_rows[start..]
            .iter()
            .filter(|row| row.decision == RigidityDecisionV1::NoResponseConditionedExcessFlag)
            .filter_map(|row| row.response_conditioned_excess_upper_tail_p.map(|p| (row.evaluation_unit_id.clone(), p)))
            .collect();
        let by_adjusted = benjamini_yekutieli_adjustment(&eligible_pairs)?;
        let family_size = eligible_pairs.len() as u64;
        for score in &mut score_rows[start..] {
            if let Some(adjusted) = by_adjusted.get(&score.evaluation_unit_id).copied() {
                score.by_adjusted_excess_p = Some(adjusted);
                score.by_family_size = Some(family_size);
                score.decision = if adjusted * 1_000.0 <= f64::from(request.target_alpha_permille) {
                    flagged += 1;
                    RigidityDecisionV1::FlagResponseConditionedExcess
                } else {
                    no_flag += 1;
                    RigidityDecisionV1::NoResponseConditionedExcessFlag
                };
            }
        }
        // The provisional no-flag count above included all eligible rows.
        // Recompute group decision counts after BY to avoid double counting.
        no_flag = score_rows[start..]
            .iter()
            .filter(|row| row.decision == RigidityDecisionV1::NoResponseConditionedExcessFlag)
            .count() as u64;
        flagged = score_rows[start..]
            .iter()
            .filter(|row| row.decision == RigidityDecisionV1::FlagResponseConditionedExcess)
            .count() as u64;
        abstained = score_rows[start..]
            .iter()
            .filter(|row| matches!(
                row.decision,
                RigidityDecisionV1::AbstainIndeterminatePathMultiplicity
                    | RigidityDecisionV1::AbstainInsufficientCalibrationResolution
            ))
            .count() as u64;
        for row in &score_rows[start..] {
            match row.exposure_label {
                Some(ExposureLabelV1::KnownClean)
                    if row.decision == RigidityDecisionV1::FlagResponseConditionedExcess => fp += 1,
                Some(ExposureLabelV1::KnownExposed)
                    if row.decision != RigidityDecisionV1::FlagResponseConditionedExcess => fn_count += 1,
                _ => {}
            }
            if row.exposure_label == Some(ExposureLabelV1::KnownExposed)
                && row.decision == RigidityDecisionV1::FlagResponseConditionedExcess
            {
                tp += 1;
            }
        }
        let clean_denominator = clean_labelled;
        let exposed_denominator = exposed_labelled;
        let empirical_fpr_permille = (clean_denominator > 0)
            .then_some(fp as f64 * 1_000.0 / clean_denominator as f64);
        let empirical_tpr_permille = (exposed_denominator > 0)
            .then_some(tp as f64 * 1_000.0 / exposed_denominator as f64);

        group_summaries.push(RigidityGroupSummaryV1 {
            task_family: key.task_family.clone(),
            difficulty_band: key.difficulty_band.clone(),
            path_multiplicity: key.path_multiplicity,
            clean_control_unit_count: clean.len() as u64,
            evaluation_unit_count: rows.len() as u64,
            eligible_unit_count: eligible_pairs.len() as u64,
            flagged_response_conditioned_excess_count: flagged,
            no_flag_count: no_flag,
            abstained_count: abstained,
            known_clean_count: clean_labelled,
            known_exposed_count: exposed_labelled,
            false_positive_count: fp,
            true_positive_count: tp,
            false_negative_count: fn_count,
            empirical_fpr_permille,
            empirical_tpr_permille,
            decision_coverage_permille: (flagged + no_flag) as f64 * 1_000.0 / rows.len() as f64,
            minimum_clean_rank_p_value: min_p,
            target_alpha_permille: request.target_alpha_permille,
        });
    }

    score_rows.sort_by(|left, right| {
        (
            &left.task_family,
            &left.difficulty_band,
            left.path_multiplicity,
            &left.evaluation_unit_id,
        )
            .cmp(&(
                &right.task_family,
                &right.difficulty_band,
                right.path_multiplicity,
                &right.evaluation_unit_id,
            ))
    });
    group_summaries.sort_by(|left, right| {
        (
            &left.task_family,
            &left.difficulty_band,
            left.path_multiplicity,
        )
            .cmp(&(
                &right.task_family,
                &right.difficulty_band,
                right.path_multiplicity,
            ))
    });

    let request_bytes = serde_json::to_vec(request)
        .map_err(|error| format!("cannot serialize policy-rigidity request: {error}"))?;
    let clean_manifest = canonical_clean_manifest(&request.clean_controls)?;
    let evaluation_manifest = canonical_evaluation_manifest(&request.evaluation)?;
    let flagged_count = score_rows
        .iter()
        .filter(|row| row.decision == RigidityDecisionV1::FlagResponseConditionedExcess)
        .count() as u64;
    let no_flag_count = score_rows
        .iter()
        .filter(|row| row.decision == RigidityDecisionV1::NoResponseConditionedExcessFlag)
        .count() as u64;
    let abstained_count = score_rows.len() as u64 - flagged_count - no_flag_count;

    Ok(PolicyRigidityControlsReportV1 {
        schema: POLICY_RIGIDITY_REPORT_SCHEMA.to_owned(),
        method_id: POLICY_RIGIDITY_FUSION_ID.to_owned(),
        study_id: request.study_id.clone(),
        benchmark_id: request.benchmark_id.clone(),
        benchmark_version: request.benchmark_version.clone(),
        model_regime: request.model_regime.clone(),
        model_checkpoint_sha256: request.model_checkpoint_sha256.clone(),
        tokenizer_sha256: request.tokenizer_sha256.clone(),
        prompt_harness_sha256: request.prompt_harness_sha256.clone(),
        runtime_environment_sha256: request.runtime_environment_sha256.clone(),
        calibration_protocol_sha256: request.calibration_protocol_sha256.clone(),
        probe_registry_sha256: request.probe_registry_sha256.clone(),
        request_sha256: hash_domain(b"policy-rigidity-controls-request.v1", &request_bytes),
        clean_control_manifest_sha256: hash_domain(
            b"policy-rigidity-clean-control-manifest.v1",
            &clean_manifest,
        ),
        evaluation_manifest_sha256: hash_domain(
            b"policy-rigidity-evaluation-manifest.v1",
            &evaluation_manifest,
        ),
        target_alpha_permille: request.target_alpha_permille,
        multiple_testing_method: "benjamini_yekutieli".to_owned(),
        fdr_control_conditional_on_exchangeability: true,
        evaluation_unit_count: score_rows.len() as u64,
        flagged_count,
        no_flag_count,
        abstained_count,
        score_rows,
        group_summaries,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: "exploratory_matched_control_decomposition".to_owned(),
        assumptions: vec![
            "All comparisons are conditional on the exact request-level model checkpoint, tokenizer, prompt harness and runtime environment.".to_owned(),
            "Clean calibration controls are exchangeable with future clean evaluation items within task-family × difficulty × path-multiplicity strata.".to_owned(),
            "Task path multiplicity is independently adjudicated before observing model outputs; reviewer digests are self-declared consistency evidence.".to_owned(),
            "Response-conditioned excess is a diagnostic contrast, not an identified causal effect or proof of item membership.".to_owned(),
            "Benjamini-Yekutieli FDR control is conditional on valid super-uniform clean-control empirical ranks.".to_owned(),
            "The design does not remove global model interference from training interventions; treatment/placebo model pairs remain the causal uncertainty units.".to_owned(),
        ],
        claim_boundary: "This report decomposes baseline entropy, general task-path rigidity and response-conditioned excess using clean controls matched to the exact model/checkpoint, task family, difficulty and adjudicated path multiplicity. It can test whether answer-conditioned rigidity is more unusual than the non-response-conditioned intervention envelope, but it does not prove item-level training membership, causal exposure, semantic truth, reviewer independence, model custody or external validity. All p-values are conditional on clean-control exchangeability; BY controls FDR only when those p-values are super-uniform. The report is exploratory.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn probe_specs() -> Vec<ProbeSpecV1> {
        ProbeInterventionV1::REQUIRED
            .iter()
            .map(|intervention| ProbeSpecV1 {
                probe_id: intervention.id().to_owned(),
                intervention: *intervention,
                prompt_contract_sha256: digest(1 + *intervention as u8),
                generation_contract_sha256: digest(20 + *intervention as u8),
            })
            .collect()
    }

    fn adjudication(classification: PathMultiplicityV1) -> PathMultiplicityAdjudicationV1 {
        PathMultiplicityAdjudicationV1 {
            classification,
            adjudication_report_sha256: digest(40),
            reviewer_panel_sha256: digest(41),
            reviewer_count: 3,
            agreement_permille: if classification == PathMultiplicityV1::Indeterminate { 700 } else { 1_000 },
        }
    }

    fn score_bundle(
        response: f64,
        prompt: f64,
        semantic: f64,
        constraint: f64,
        alternate: Option<f64>,
    ) -> RigidityProbeScoresV1 {
        RigidityProbeScoresV1 {
            baseline_mean_entropy_nats: 0.8,
            response_conditioned_similarity: response,
            prompt_only_similarity: prompt,
            semantic_paraphrase_similarity: semantic,
            constraint_reparameterization_similarity: constraint,
            alternate_valid_method_similarity: alternate,
        }
    }

    fn clean_record(id: &str, path: PathMultiplicityV1, excess: f64) -> CleanControlRecordV1 {
        let global = 0.4;
        CleanControlRecordV1 {
            calibration_unit_id: id.to_owned(),
            source_cluster_sha256: hash_domain(b"policy-rigidity-test-source.v1", id.as_bytes()),
            task_family: "mathematics".to_owned(),
            difficulty_band: "medium".to_owned(),
            path_multiplicity: adjudication(path),
            probe_receipt_manifest_sha256: digest(42),
            task_recipe_sha256: digest(43),
            exposure_label: ExposureLabelV1::KnownClean,
            scores: if path == PathMultiplicityV1::MultipleValidPaths {
                score_bundle(
                    global + excess,
                    global,
                    global,
                    global,
                    Some(global),
                )
            } else {
                score_bundle(global + excess, global, global, global, None)
            },
        }
    }

    fn eval_record(
        id: &str,
        path: PathMultiplicityV1,
        label: ExposureLabelV1,
        response: f64,
    ) -> EvaluationControlRecordV1 {
        EvaluationControlRecordV1 {
            evaluation_unit_id: id.to_owned(),
            source_cluster_sha256: hash_domain(b"policy-rigidity-test-source.v1", id.as_bytes()),
            task_family: "mathematics".to_owned(),
            difficulty_band: "medium".to_owned(),
            path_multiplicity: adjudication(path),
            probe_receipt_manifest_sha256: digest(44),
            task_recipe_sha256: digest(45),
            exposure_label: Some(label),
            scores: if path == PathMultiplicityV1::MultipleValidPaths {
                score_bundle(response, 0.4, 0.4, 0.4, Some(0.4))
            } else {
                score_bundle(response, 0.4, 0.4, 0.4, None)
            },
        }
    }

    fn request() -> PolicyRigidityControlsRequestV1 {
        let probe_specs = probe_specs();
        let probe_registry_sha256 = compute_probe_registry_sha256(&probe_specs).unwrap();
        PolicyRigidityControlsRequestV1 {
            schema: POLICY_RIGIDITY_REQUEST_SCHEMA.to_owned(),
            study_id: "study-001".to_owned(),
            benchmark_id: "rigidity-controls-v1".to_owned(),
            benchmark_version: "v1".to_owned(),
            model_regime: "qwen-rlvr".to_owned(),
            model_checkpoint_sha256: digest(50),
            tokenizer_sha256: digest(51),
            prompt_harness_sha256: digest(52),
            runtime_environment_sha256: digest(53),
            calibration_protocol_sha256: digest(54),
            probe_registry_sha256,
            target_alpha_permille: 500,
            probe_specs,
            clean_controls: vec![
                clean_record("c1", PathMultiplicityV1::SingleOrFewValidPaths, -0.1),
                clean_record("c2", PathMultiplicityV1::SingleOrFewValidPaths, 0.0),
                clean_record("c3", PathMultiplicityV1::SingleOrFewValidPaths, 0.1),
                clean_record("m1", PathMultiplicityV1::MultipleValidPaths, -0.1),
                clean_record("m2", PathMultiplicityV1::MultipleValidPaths, 0.0),
                clean_record("m3", PathMultiplicityV1::MultipleValidPaths, 0.1),
            ],
            evaluation: vec![
                eval_record("e1", PathMultiplicityV1::SingleOrFewValidPaths, ExposureLabelV1::KnownClean, 0.9),
                eval_record("e2", PathMultiplicityV1::MultipleValidPaths, ExposureLabelV1::KnownExposed, 0.9),
            ],
        }
    }

    #[test]
    fn decomposition_separates_global_rigidity_from_response_conditioned_excess() {
        let scores = score_bundle(0.9, 0.4, 0.5, 0.3, Some(0.4));
        let d = decompose(&scores).unwrap();
        assert!((d.global_path_rigidity_score - 0.4).abs() < 1.0e-12);
        assert!((d.response_conditioned_excess - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn clean_rigid_task_is_calibrated_only_against_same_path_multiplicity() {
        let request = request();
        let report = analyze_policy_rigidity_controls(&request).unwrap();
        assert_eq!(report.group_summaries.len(), 2);
        assert!(report.score_rows.iter().all(|row| row.global_entropy_lower_tail_p.is_some()));
    }

    #[test]
    fn multiple_path_tasks_require_alternate_valid_method_probe() {
        let mut request = request();
        request.evaluation[1].scores.alternate_valid_method_similarity = None;
        assert!(analyze_policy_rigidity_controls(&request).is_err());
    }

    #[test]
    fn single_path_tasks_reject_inapplicable_alternate_method_probe() {
        let mut request = request();
        request.evaluation[0].scores.alternate_valid_method_similarity = Some(0.5);
        assert!(analyze_policy_rigidity_controls(&request).is_err());
    }

    #[test]
    fn indeterminate_path_multiplicity_abstains() {
        let mut request = request();
        request.evaluation[0].path_multiplicity = adjudication(PathMultiplicityV1::Indeterminate);
        request.evaluation[0].scores.alternate_valid_method_similarity = None;
        let report = analyze_policy_rigidity_controls(&request).unwrap();
        let row = report.score_rows.iter().find(|row| row.evaluation_unit_id == "e1").unwrap();
        assert_eq!(row.decision, RigidityDecisionV1::AbstainIndeterminatePathMultiplicity);
    }

    #[test]
    fn calibration_rows_must_be_known_clean() {
        let mut request = request();
        request.clean_controls[0].exposure_label = ExposureLabelV1::KnownExposed;
        assert!(analyze_policy_rigidity_controls(&request).is_err());
    }

    #[test]
    fn source_cluster_overlap_across_splits_is_rejected() {
        let mut request = request();
        request.evaluation[0].source_cluster_sha256 = request.clean_controls[0].source_cluster_sha256.clone();
        assert!(analyze_policy_rigidity_controls(&request).is_err());
    }

    #[test]
    fn probe_registry_digest_binds_all_probe_contracts() {
        let request = request();
        assert_eq!(
            compute_probe_registry_sha256(&request.probe_specs).unwrap(),
            request.probe_registry_sha256
        );
        let mut changed = request.probe_specs.clone();
        changed[0].prompt_contract_sha256 = digest(99);
        assert_ne!(
            compute_probe_registry_sha256(&changed).unwrap(),
            request.probe_registry_sha256
        );
    }

    #[test]
    fn conservative_empirical_ranks_count_ties() {
        assert_eq!(upper_tail_p(&[0.1, 0.2, 0.2, 0.3], 0.2).unwrap(), 0.8);
        assert_eq!(lower_tail_p(&[0.1, 0.2, 0.2, 0.3], 0.2).unwrap(), 0.8);
    }

    #[test]
    fn by_adjustment_is_monotone_and_bounded() {
        let values = vec![
            ("a".to_owned(), 0.01),
            ("b".to_owned(), 0.03),
            ("c".to_owned(), 0.04),
        ];
        let adjusted = benjamini_yekutieli_adjustment(&values).unwrap();
        assert!(adjusted["a"] <= adjusted["b"]);
        assert!(adjusted["b"] <= adjusted["c"]);
        assert!(adjusted.values().all(|p| *p >= 0.0 && *p <= 1.0));
    }

    #[test]
    fn full_request_fixture_round_trips_and_scores() {
        let request: PolicyRigidityControlsRequestV1 = serde_json::from_str(include_str!(
            "../fixtures/policy-rigidity-controls-request.hand.v1.json"
        )).unwrap();
        let report = analyze_policy_rigidity_controls(&request).unwrap();
        assert_eq!(report.evaluation_unit_count, 2);
        assert_eq!(report.group_summaries.len(), 2);
        assert!(!report.confirmatory_eligible);
    }
}
