#![forbid(unsafe_code)]

//! One-sided cluster-conformal calibration of recoverability-portfolio margin.
//!
//! Calibration scores only dangerous optimism: predicted kth-largest independent
//! recovery-domain margin minus independently verified reference margin, clipped
//! below at zero, with a maximum over the complete pre-registered block.

use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellIdentity {
    pub rollout_index: u32,
    pub action_query_cell_index: u32,
    /// Prefix zero is the current state; horizon H contains prefixes 0..=H.
    pub prefix_step: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionMargin {
    pub profile_id: String,
    pub failure_domain_sha256: String,
    /// Positive means inside the recovery entry region; non-positive is not admissible.
    pub margin_units: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryMarginBlockSpec {
    pub contract_sha256: String,
    pub source_manifest_sha256: String,
    pub candidate_checkpoint_sha256: String,
    pub plant_model_sha256: String,
    pub context_scope_sha256: String,
    pub recovery_portfolio_sha256: String,
    pub recovery_certificate_manifest_sha256: String,
    pub margin_metric_sha256: String,
    /// Includes action/query portfolio and any adaptive search/selection procedure.
    pub selection_protocol_sha256: String,
    pub rollouts_per_cluster: u32,
    pub action_query_cell_count: u32,
    pub horizon_steps: u32,
    pub required_independent_domains: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateMarginCell {
    pub identity: CellIdentity,
    pub option_margins: Vec<OptionMargin>,
}

/// Candidate-side predictions only. No reference margin or evaluator cluster label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateMarginBlock {
    pub candidate_view_sha256: String,
    pub predicted_margin_block_sha256: String,
    pub spec: RecoveryMarginBlockSpec,
    pub cells: Vec<CandidateMarginCell>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceMarginCell {
    pub identity: CellIdentity,
    pub option_margins: Vec<OptionMargin>,
}

/// Evaluator-only reference margins, produced by an independent verifier/replay boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceMarginBlock {
    pub independent_cluster_id: String,
    pub environment_family_id: String,
    pub block_manifest_sha256: String,
    /// Digest of the exact candidate margin block used to compute the reference residual.
    pub candidate_margin_block_sha256: String,
    pub reference_margin_receipt_sha256: String,
    pub spec: RecoveryMarginBlockSpec,
    pub cells: Vec<ReferenceMarginCell>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationUnitPair {
    pub candidate: CandidateMarginBlock,
    pub reference: ReferenceMarginBlock,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationManifest {
    pub calibration_manifest_sha256: String,
    pub training_cluster_ids: Vec<String>,
    pub scale_validation_cluster_ids: Vec<String>,
    pub held_out_evaluation_cluster_ids: Vec<String>,
    /// Exactly one candidate/reference block pair per independent calibration cluster.
    pub calibration_units: Vec<CalibrationUnitPair>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarginCalibrationContract {
    pub spec: RecoveryMarginBlockSpec,
    pub alpha_numerator: u32,
    pub alpha_denominator: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedMarginCalibration {
    pub spec: RecoveryMarginBlockSpec,
    pub calibration_manifest_sha256: String,
    /// Upper quantile of cluster-block optimism; pessimistic predictions are not penalized.
    pub optimism_radius_units: u64,
    pub quantile_rank_one_based: usize,
    pub calibration_cluster_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationDiagnostics {
    pub calibration_family_count: usize,
    pub training_cluster_count: usize,
    pub validation_cluster_count: usize,
    pub held_out_evaluation_cluster_count: usize,
}

/// One immutable order-statistic input, retained for downstream monitoring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationClusterScore {
    pub independent_cluster_id: String,
    pub score_units: u64,
    pub block_manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationResult {
    pub qualified: QualifiedMarginCalibration,
    pub diagnostics: CalibrationDiagnostics,
    /// Exact unsorted per-cluster scores used to produce the qualified quantile.
    pub cluster_scores: Vec<CalibrationClusterScore>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviousOutcome {
    Reconciled,
    ConfirmedNotExecuted,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeMarginForecast {
    pub previous_outcome: PreviousOutcome,
    pub block: CandidateMarginBlock,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarginDisposition {
    BlockUntilReconciliation,
    CommitWithinCalibratedBlock,
    DoNotCommit {
        first_nonpositive_cell: CellIdentity,
        lower_bound_margin_units: i64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedMarginAssessment {
    pub disposition: MarginDisposition,
    pub optimism_radius_units: u64,
    pub minimum_lower_bound_margin_units: Option<i64>,
    pub calibration_cluster_count: usize,
    pub calibration_manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarginCalibrationError {
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    InvalidMiscoverage,
    InvalidBlockSpec(&'static str),
    EmptySplit(&'static str),
    DuplicateClusterInSplit,
    ClusterSplitLeakage,
    DuplicateCalibrationCluster,
    InvalidCellInventory,
    DuplicateCellIdentity,
    InvalidOptionIdentity,
    DuplicateProfileId,
    ProfileSetMismatch,
    InsufficientIndependentDomains {
        required: usize,
        available: usize,
    },
    NoCalibrationClusters,
    InsufficientCalibrationClusters {
        required_rank: usize,
        available_clusters: usize,
    },
    RuntimeScopeMismatch(&'static str),
    MarginArithmeticOverflow,
}

/// Computes the kth-largest margin after collapsing profiles in the same failure domain.
/// Multiple profiles in one domain contribute one option: the best admissible margin in
/// that domain. Distinct profile IDs never establish domain independence by themselves.
pub fn kth_largest_independent_margin(
    options: &[OptionMargin],
    required_domains: usize,
) -> Result<i64, MarginCalibrationError> {
    if required_domains == 0 {
        return Err(MarginCalibrationError::InvalidBlockSpec(
            "required_independent_domains",
        ));
    }
    let mut best_by_domain: BTreeMap<String, i64> = BTreeMap::new();
    let mut profile_ids = HashSet::new();
    for option in options {
        if option.profile_id.trim().is_empty() {
            return Err(MarginCalibrationError::EmptyIdentity("profile_id"));
        }
        if !is_sha256_hex(&option.failure_domain_sha256) {
            return Err(MarginCalibrationError::InvalidSha256(
                "failure_domain_sha256",
            ));
        }
        if !profile_ids.insert(option.profile_id.clone()) {
            return Err(MarginCalibrationError::DuplicateProfileId);
        }
        best_by_domain
            .entry(option.failure_domain_sha256.clone())
            .and_modify(|current| *current = (*current).max(option.margin_units))
            .or_insert(option.margin_units);
    }
    if best_by_domain.len() < required_domains {
        return Err(MarginCalibrationError::InsufficientIndependentDomains {
            required: required_domains,
            available: best_by_domain.len(),
        });
    }
    let mut margins: Vec<i64> = best_by_domain.into_values().collect();
    margins.sort_unstable_by(|left, right| right.cmp(left));
    Ok(margins[required_domains - 1])
}

/// Calibrates one-sided over-optimism of the kth independent recovery-option margin.
///
/// The cluster score is max_cell(max(0, predicted_k_margin - reference_k_margin)).
pub fn calibrate_recoverability_margin(
    contract: MarginCalibrationContract,
    manifest: &CalibrationManifest,
) -> Result<CalibrationResult, MarginCalibrationError> {
    validate_contract(&contract)?;
    validate_split_manifest(manifest)?;
    if !is_sha256_hex(&manifest.calibration_manifest_sha256) {
        return Err(MarginCalibrationError::InvalidSha256(
            "calibration_manifest_sha256",
        ));
    }
    if manifest.calibration_units.is_empty() {
        return Err(MarginCalibrationError::NoCalibrationClusters);
    }

    let forbidden_ids: HashSet<&str> = manifest
        .training_cluster_ids
        .iter()
        .chain(manifest.scale_validation_cluster_ids.iter())
        .chain(manifest.held_out_evaluation_cluster_ids.iter())
        .map(String::as_str)
        .collect();

    let mut cluster_ids = HashSet::new();
    let mut family_ids = HashSet::new();
    let mut scores = Vec::with_capacity(manifest.calibration_units.len());
    let mut cluster_scores = Vec::with_capacity(manifest.calibration_units.len());

    for unit in &manifest.calibration_units {
        let reference = &unit.reference;
        if reference.independent_cluster_id.trim().is_empty() {
            return Err(MarginCalibrationError::EmptyIdentity(
                "independent_cluster_id",
            ));
        }
        if reference.environment_family_id.trim().is_empty() {
            return Err(MarginCalibrationError::EmptyIdentity(
                "environment_family_id",
            ));
        }
        if !is_sha256_hex(&reference.block_manifest_sha256) {
            return Err(MarginCalibrationError::InvalidSha256(
                "block_manifest_sha256",
            ));
        }
        if !is_sha256_hex(&reference.candidate_margin_block_sha256) {
            return Err(MarginCalibrationError::InvalidSha256(
                "candidate_margin_block_sha256",
            ));
        }
        if reference.candidate_margin_block_sha256 != unit.candidate.predicted_margin_block_sha256 {
            return Err(MarginCalibrationError::RuntimeScopeMismatch(
                "candidate_margin_block_sha256",
            ));
        }
        if !is_sha256_hex(&reference.reference_margin_receipt_sha256) {
            return Err(MarginCalibrationError::InvalidSha256(
                "reference_margin_receipt_sha256",
            ));
        }
        if forbidden_ids.contains(reference.independent_cluster_id.as_str()) {
            return Err(MarginCalibrationError::ClusterSplitLeakage);
        }
        if !cluster_ids.insert(reference.independent_cluster_id.clone()) {
            return Err(MarginCalibrationError::DuplicateCalibrationCluster);
        }
        let score = score_calibration_unit(&unit.candidate, reference, &contract.spec)?;
        scores.push(score);
        cluster_scores.push(CalibrationClusterScore {
            independent_cluster_id: reference.independent_cluster_id.clone(),
            score_units: score,
            block_manifest_sha256: reference.block_manifest_sha256.clone(),
        });
        family_ids.insert(reference.environment_family_id.clone());
    }

    let rank = finite_sample_quantile_rank(
        scores.len(),
        contract.alpha_numerator,
        contract.alpha_denominator,
    )?;
    if rank > scores.len() {
        return Err(MarginCalibrationError::InsufficientCalibrationClusters {
            required_rank: rank,
            available_clusters: scores.len(),
        });
    }
    scores.sort_unstable();
    let optimism_radius_units = scores[rank - 1];

    Ok(CalibrationResult {
        qualified: QualifiedMarginCalibration {
            spec: contract.spec,
            calibration_manifest_sha256: manifest.calibration_manifest_sha256.clone(),
            optimism_radius_units,
            quantile_rank_one_based: rank,
            calibration_cluster_count: scores.len(),
        },
        diagnostics: CalibrationDiagnostics {
            calibration_family_count: family_ids.len(),
            training_cluster_count: manifest.training_cluster_ids.len(),
            validation_cluster_count: manifest.scale_validation_cluster_ids.len(),
            held_out_evaluation_cluster_count: manifest.held_out_evaluation_cluster_ids.len(),
        },
        cluster_scores,
    })
}

pub fn apply_recoverability_margin_calibration(
    calibration: &QualifiedMarginCalibration,
    forecast: &RuntimeMarginForecast,
) -> Result<AppliedMarginAssessment, MarginCalibrationError> {
    if forecast.previous_outcome == PreviousOutcome::Unknown {
        return Ok(AppliedMarginAssessment {
            disposition: MarginDisposition::BlockUntilReconciliation,
            optimism_radius_units: calibration.optimism_radius_units,
            minimum_lower_bound_margin_units: None,
            calibration_cluster_count: calibration.calibration_cluster_count,
            calibration_manifest_sha256: calibration.calibration_manifest_sha256.clone(),
        });
    }
    validate_candidate_block(&forecast.block, &calibration.spec)?;

    let mut lower_bounds = Vec::with_capacity(forecast.block.cells.len());
    for cell in &forecast.block.cells {
        let predicted_margin = kth_largest_independent_margin(
            &cell.option_margins,
            calibration.spec.required_independent_domains,
        )?;
        let lower = i128::from(predicted_margin) - i128::from(calibration.optimism_radius_units);
        let lower =
            i64::try_from(lower).map_err(|_| MarginCalibrationError::MarginArithmeticOverflow)?;
        lower_bounds.push((cell.identity.clone(), lower));
    }
    let minimum_lower_bound = lower_bounds
        .iter()
        .map(|(_, margin)| *margin)
        .min()
        .ok_or(MarginCalibrationError::InvalidCellInventory)?;

    let first_nonpositive = lower_bounds
        .iter()
        .filter(|(_, margin)| *margin <= 0)
        .min_by_key(|(identity, _)| identity.clone());

    let disposition = match first_nonpositive {
        Some((identity, margin)) => MarginDisposition::DoNotCommit {
            first_nonpositive_cell: identity.clone(),
            lower_bound_margin_units: *margin,
        },
        None => MarginDisposition::CommitWithinCalibratedBlock,
    };

    Ok(AppliedMarginAssessment {
        disposition,
        optimism_radius_units: calibration.optimism_radius_units,
        minimum_lower_bound_margin_units: Some(minimum_lower_bound),
        calibration_cluster_count: calibration.calibration_cluster_count,
        calibration_manifest_sha256: calibration.calibration_manifest_sha256.clone(),
    })
}

fn score_calibration_unit(
    candidate: &CandidateMarginBlock,
    reference: &ReferenceMarginBlock,
    spec: &RecoveryMarginBlockSpec,
) -> Result<u64, MarginCalibrationError> {
    validate_candidate_block(candidate, spec)?;
    if reference.spec != *spec {
        return Err(MarginCalibrationError::RuntimeScopeMismatch(
            "reference_block_spec",
        ));
    }
    let reference_identities: Vec<CellIdentity> = reference
        .cells
        .iter()
        .map(|cell| cell.identity.clone())
        .collect();
    validate_cell_inventory(&reference_identities, spec)?;
    if candidate.cells.len() != reference.cells.len() {
        return Err(MarginCalibrationError::InvalidCellInventory);
    }

    let mut reference_by_id = BTreeMap::new();
    for cell in &reference.cells {
        if reference_by_id
            .insert(cell.identity.clone(), &cell.option_margins)
            .is_some()
        {
            return Err(MarginCalibrationError::DuplicateCellIdentity);
        }
    }

    let mut maximum_optimism = 0u64;
    for candidate_cell in &candidate.cells {
        let reference_options = reference_by_id
            .get(&candidate_cell.identity)
            .ok_or(MarginCalibrationError::InvalidCellInventory)?;
        validate_matching_profile_sets(&candidate_cell.option_margins, reference_options)?;
        let predicted = kth_largest_independent_margin(
            &candidate_cell.option_margins,
            spec.required_independent_domains,
        )?;
        let actual =
            kth_largest_independent_margin(reference_options, spec.required_independent_domains)?;
        let optimism = (i128::from(predicted) - i128::from(actual)).max(0);
        let optimism = u64::try_from(optimism)
            .map_err(|_| MarginCalibrationError::MarginArithmeticOverflow)?;
        maximum_optimism = maximum_optimism.max(optimism);
    }
    Ok(maximum_optimism)
}

fn validate_matching_profile_sets(
    candidate: &[OptionMargin],
    reference: &[OptionMargin],
) -> Result<(), MarginCalibrationError> {
    let mut left: Vec<(String, String)> = candidate
        .iter()
        .map(|item| (item.profile_id.clone(), item.failure_domain_sha256.clone()))
        .collect();
    let mut right: Vec<(String, String)> = reference
        .iter()
        .map(|item| (item.profile_id.clone(), item.failure_domain_sha256.clone()))
        .collect();
    left.sort_unstable();
    right.sort_unstable();
    if left != right {
        return Err(MarginCalibrationError::ProfileSetMismatch);
    }
    Ok(())
}

fn validate_candidate_block(
    candidate: &CandidateMarginBlock,
    spec: &RecoveryMarginBlockSpec,
) -> Result<(), MarginCalibrationError> {
    if !is_sha256_hex(&candidate.candidate_view_sha256) {
        return Err(MarginCalibrationError::InvalidSha256(
            "candidate_view_sha256",
        ));
    }
    if !is_sha256_hex(&candidate.predicted_margin_block_sha256) {
        return Err(MarginCalibrationError::InvalidSha256(
            "predicted_margin_block_sha256",
        ));
    }
    if candidate.spec != *spec {
        return Err(MarginCalibrationError::RuntimeScopeMismatch(
            "candidate_block_spec",
        ));
    }
    let identities: Vec<CellIdentity> = candidate
        .cells
        .iter()
        .map(|cell| cell.identity.clone())
        .collect();
    validate_cell_inventory(&identities, spec)
}

fn validate_cell_inventory(
    identities: &[CellIdentity],
    spec: &RecoveryMarginBlockSpec,
) -> Result<(), MarginCalibrationError> {
    let expected = expected_cell_count(spec)?;
    if identities.len() != expected {
        return Err(MarginCalibrationError::InvalidCellInventory);
    }
    let mut seen = HashSet::new();
    for identity in identities {
        if identity.rollout_index >= spec.rollouts_per_cluster
            || identity.action_query_cell_index >= spec.action_query_cell_count
            || identity.prefix_step > spec.horizon_steps
        {
            return Err(MarginCalibrationError::InvalidCellInventory);
        }
        if !seen.insert(identity.clone()) {
            return Err(MarginCalibrationError::DuplicateCellIdentity);
        }
    }
    Ok(())
}

fn expected_cell_count(spec: &RecoveryMarginBlockSpec) -> Result<usize, MarginCalibrationError> {
    let count = u128::from(spec.rollouts_per_cluster)
        .checked_mul(u128::from(spec.action_query_cell_count))
        .and_then(|value| value.checked_mul(u128::from(spec.horizon_steps) + 1))
        .ok_or(MarginCalibrationError::InvalidBlockSpec("cell_count"))?;
    usize::try_from(count).map_err(|_| MarginCalibrationError::InvalidBlockSpec("cell_count"))
}

fn validate_contract(contract: &MarginCalibrationContract) -> Result<(), MarginCalibrationError> {
    validate_spec(&contract.spec)?;
    if contract.alpha_denominator == 0
        || contract.alpha_numerator == 0
        || contract.alpha_numerator >= contract.alpha_denominator
    {
        return Err(MarginCalibrationError::InvalidMiscoverage);
    }
    Ok(())
}

fn validate_spec(spec: &RecoveryMarginBlockSpec) -> Result<(), MarginCalibrationError> {
    for (value, name) in [
        (&spec.contract_sha256, "contract_sha256"),
        (&spec.source_manifest_sha256, "source_manifest_sha256"),
        (
            &spec.candidate_checkpoint_sha256,
            "candidate_checkpoint_sha256",
        ),
        (&spec.plant_model_sha256, "plant_model_sha256"),
        (&spec.context_scope_sha256, "context_scope_sha256"),
        (&spec.recovery_portfolio_sha256, "recovery_portfolio_sha256"),
        (
            &spec.recovery_certificate_manifest_sha256,
            "recovery_certificate_manifest_sha256",
        ),
        (&spec.margin_metric_sha256, "margin_metric_sha256"),
        (&spec.selection_protocol_sha256, "selection_protocol_sha256"),
    ] {
        if !is_sha256_hex(value) {
            return Err(MarginCalibrationError::InvalidSha256(name));
        }
    }
    if spec.rollouts_per_cluster == 0 {
        return Err(MarginCalibrationError::InvalidBlockSpec(
            "rollouts_per_cluster",
        ));
    }
    if spec.action_query_cell_count == 0 {
        return Err(MarginCalibrationError::InvalidBlockSpec(
            "action_query_cell_count",
        ));
    }
    if spec.horizon_steps == 0 {
        return Err(MarginCalibrationError::InvalidBlockSpec("horizon_steps"));
    }
    if spec.required_independent_domains == 0 {
        return Err(MarginCalibrationError::InvalidBlockSpec(
            "required_independent_domains",
        ));
    }
    expected_cell_count(spec)?;
    Ok(())
}

fn validate_split_manifest(manifest: &CalibrationManifest) -> Result<(), MarginCalibrationError> {
    if manifest.training_cluster_ids.is_empty() {
        return Err(MarginCalibrationError::EmptySplit("training_cluster_ids"));
    }
    if manifest.scale_validation_cluster_ids.is_empty() {
        return Err(MarginCalibrationError::EmptySplit(
            "scale_validation_cluster_ids",
        ));
    }
    if manifest.held_out_evaluation_cluster_ids.is_empty() {
        return Err(MarginCalibrationError::EmptySplit(
            "held_out_evaluation_cluster_ids",
        ));
    }
    let mut seen = HashSet::new();
    for id in &manifest.training_cluster_ids {
        insert_split_id(id, &mut seen)?;
    }
    for id in &manifest.scale_validation_cluster_ids {
        insert_split_id(id, &mut seen)?;
    }
    for id in &manifest.held_out_evaluation_cluster_ids {
        insert_split_id(id, &mut seen)?;
    }
    for unit in &manifest.calibration_units {
        if seen.contains(&unit.reference.independent_cluster_id) {
            return Err(MarginCalibrationError::ClusterSplitLeakage);
        }
    }
    Ok(())
}

fn insert_split_id(id: &str, seen: &mut HashSet<String>) -> Result<(), MarginCalibrationError> {
    if id.trim().is_empty() {
        return Err(MarginCalibrationError::EmptyIdentity("split_cluster_id"));
    }
    if !seen.insert(id.to_owned()) {
        return Err(MarginCalibrationError::DuplicateClusterInSplit);
    }
    Ok(())
}

pub fn finite_sample_quantile_rank(
    calibration_clusters: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
) -> Result<usize, MarginCalibrationError> {
    if alpha_denominator == 0 || alpha_numerator == 0 || alpha_numerator >= alpha_denominator {
        return Err(MarginCalibrationError::InvalidMiscoverage);
    }
    let n_plus_one = (calibration_clusters as u128)
        .checked_add(1)
        .ok_or(MarginCalibrationError::InvalidMiscoverage)?;
    let coverage_numerator = u128::from(alpha_denominator - alpha_numerator);
    let denominator = u128::from(alpha_denominator);
    let product = n_plus_one
        .checked_mul(coverage_numerator)
        .ok_or(MarginCalibrationError::InvalidMiscoverage)?;
    let rank = product
        .checked_add(denominator - 1)
        .ok_or(MarginCalibrationError::InvalidMiscoverage)?
        / denominator;
    usize::try_from(rank).map_err(|_| MarginCalibrationError::InvalidMiscoverage)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn spec() -> RecoveryMarginBlockSpec {
        RecoveryMarginBlockSpec {
            contract_sha256: digest('a'),
            source_manifest_sha256: digest('b'),
            candidate_checkpoint_sha256: digest('c'),
            plant_model_sha256: digest('d'),
            context_scope_sha256: digest('e'),
            recovery_portfolio_sha256: digest('f'),
            recovery_certificate_manifest_sha256: digest('1'),
            margin_metric_sha256: digest('2'),
            selection_protocol_sha256: digest('3'),
            rollouts_per_cluster: 1,
            action_query_cell_count: 1,
            horizon_steps: 1,
            required_independent_domains: 2,
        }
    }

    fn margins(base: i64) -> Vec<OptionMargin> {
        vec![
            OptionMargin {
                profile_id: "fast-a".into(),
                failure_domain_sha256: digest('4'),
                margin_units: base + 10,
            },
            OptionMargin {
                profile_id: "fast-b-same-domain".into(),
                failure_domain_sha256: digest('4'),
                margin_units: base + 20,
            },
            OptionMargin {
                profile_id: "independent-slow".into(),
                failure_domain_sha256: digest('5'),
                margin_units: base + 5,
            },
        ]
    }

    fn candidate_block(base: i64) -> CandidateMarginBlock {
        let s = spec();
        CandidateMarginBlock {
            candidate_view_sha256: digest('6'),
            predicted_margin_block_sha256: digest('7'),
            spec: s,
            cells: vec![
                CandidateMarginCell {
                    identity: CellIdentity {
                        rollout_index: 0,
                        action_query_cell_index: 0,
                        prefix_step: 0,
                    },
                    option_margins: margins(base),
                },
                CandidateMarginCell {
                    identity: CellIdentity {
                        rollout_index: 0,
                        action_query_cell_index: 0,
                        prefix_step: 1,
                    },
                    option_margins: margins(base),
                },
            ],
        }
    }

    fn reference_block(
        cluster: usize,
        predicted_base: i64,
        overoptimism: i64,
    ) -> ReferenceMarginBlock {
        let s = spec();
        let reference_base = predicted_base - overoptimism;
        ReferenceMarginBlock {
            independent_cluster_id: format!("calibration-{cluster}"),
            environment_family_id: format!("family-{}", cluster % 3),
            block_manifest_sha256: digest('8'),
            candidate_margin_block_sha256: digest('7'),
            reference_margin_receipt_sha256: digest('9'),
            spec: s,
            cells: vec![
                ReferenceMarginCell {
                    identity: CellIdentity {
                        rollout_index: 0,
                        action_query_cell_index: 0,
                        prefix_step: 0,
                    },
                    option_margins: margins(reference_base),
                },
                ReferenceMarginCell {
                    identity: CellIdentity {
                        rollout_index: 0,
                        action_query_cell_index: 0,
                        prefix_step: 1,
                    },
                    option_margins: margins(reference_base),
                },
            ],
        }
    }

    fn pair(cluster: usize, overoptimism: i64) -> CalibrationUnitPair {
        CalibrationUnitPair {
            candidate: candidate_block(10),
            reference: reference_block(cluster, 10, overoptimism),
        }
    }

    fn contract() -> MarginCalibrationContract {
        MarginCalibrationContract {
            spec: spec(),
            alpha_numerator: 1,
            alpha_denominator: 20,
        }
    }

    fn manifest(count: usize) -> CalibrationManifest {
        CalibrationManifest {
            calibration_manifest_sha256: digest('0'),
            training_cluster_ids: vec!["training-1".into()],
            scale_validation_cluster_ids: vec!["validation-1".into()],
            held_out_evaluation_cluster_ids: vec!["evaluation-1".into()],
            calibration_units: (0..count)
                .map(|index| pair(index, index as i64 + 1))
                .collect(),
        }
    }

    fn qualified() -> QualifiedMarginCalibration {
        calibrate_recoverability_margin(contract(), &manifest(19))
            .unwrap()
            .qualified
    }

    fn runtime_forecast() -> RuntimeMarginForecast {
        RuntimeMarginForecast {
            previous_outcome: PreviousOutcome::Reconciled,
            block: candidate_block(30),
        }
    }

    #[test]
    fn two_profiles_in_one_failure_domain_count_as_one_option() {
        let options = margins(0);
        assert_eq!(kth_largest_independent_margin(&options, 2).unwrap(), 5);
    }

    #[test]
    fn profile_ids_do_not_create_false_independence() {
        let options = vec![
            OptionMargin {
                profile_id: "profile-one".into(),
                failure_domain_sha256: digest('4'),
                margin_units: 10,
            },
            OptionMargin {
                profile_id: "profile-two".into(),
                failure_domain_sha256: digest('4'),
                margin_units: 12,
            },
        ];
        assert_eq!(
            kth_largest_independent_margin(&options, 2),
            Err(MarginCalibrationError::InsufficientIndependentDomains {
                required: 2,
                available: 1,
            })
        );
    }

    #[test]
    fn nineteen_clusters_produce_expected_95_percent_one_sided_optimism_radius() {
        let result = calibrate_recoverability_margin(contract(), &manifest(19)).unwrap();
        assert_eq!(result.qualified.quantile_rank_one_based, 19);
        assert_eq!(result.qualified.optimism_radius_units, 19);
        assert_eq!(result.qualified.calibration_cluster_count, 19);
        assert_eq!(result.diagnostics.calibration_family_count, 3);
    }

    #[test]
    fn calibration_result_retains_exact_per_cluster_scores() {
        let m = manifest(19);
        let result = calibrate_recoverability_margin(contract(), &m).unwrap();
        assert_eq!(
            result.cluster_scores.len(),
            result.qualified.calibration_cluster_count
        );
        for (score, unit) in result.cluster_scores.iter().zip(&m.calibration_units) {
            assert_eq!(
                score.independent_cluster_id,
                unit.reference.independent_cluster_id
            );
            assert_eq!(
                score.block_manifest_sha256,
                unit.reference.block_manifest_sha256
            );
            assert_eq!(
                score.score_units,
                score_calibration_unit(&unit.candidate, &unit.reference, &spec()).unwrap()
            );
        }
    }

    #[test]
    fn eighteen_clusters_are_insufficient_at_95_percent() {
        assert_eq!(
            calibrate_recoverability_margin(contract(), &manifest(18)),
            Err(MarginCalibrationError::InsufficientCalibrationClusters {
                required_rank: 19,
                available_clusters: 18,
            })
        );
    }

    #[test]
    fn reference_must_be_bound_to_exact_candidate_margin_block() {
        let mut m = manifest(19);
        m.calibration_units[0].reference.candidate_margin_block_sha256 = digest('0');
        assert_eq!(
            calibrate_recoverability_margin(contract(), &m),
            Err(MarginCalibrationError::RuntimeScopeMismatch(
                "candidate_margin_block_sha256"
            ))
        );
    }

    #[test]
    fn underprediction_is_not_penalized_by_one_sided_score() {
        let c = candidate_block(0);
        let r = reference_block(1, 0, -10);
        assert_eq!(score_calibration_unit(&c, &r, &spec()).unwrap(), 0);
    }

    #[test]
    fn maximum_optimism_over_any_prefix_controls_cluster_score() {
        let mut c = candidate_block(10);
        let mut r = reference_block(1, 10, 2);
        // Raise domain A so domain B is the limiting kth margin, then
        // overstate domain B by 25 units at prefix 1.
        c.cells[1].option_margins[0].margin_units += 100;
        c.cells[1].option_margins[1].margin_units += 100;
        c.cells[1].option_margins[2].margin_units += 30;
        r.cells[1].option_margins[0].margin_units += 100;
        r.cells[1].option_margins[1].margin_units += 100;
        r.cells[1].option_margins[2].margin_units = c.cells[1].option_margins[2].margin_units - 25;
        assert_eq!(score_calibration_unit(&c, &r, &spec()).unwrap(), 25);
    }

    #[test]
    fn lower_bound_margin_allows_commit_when_corrected_margin_stays_positive() {
        let calibration = qualified();
        let assessment =
            apply_recoverability_margin_calibration(&calibration, &runtime_forecast()).unwrap();
        assert_eq!(
            assessment.disposition,
            MarginDisposition::CommitWithinCalibratedBlock
        );
        assert_eq!(assessment.optimism_radius_units, 19);
        assert_eq!(assessment.minimum_lower_bound_margin_units, Some(16));
    }

    #[test]
    fn nonpositive_calibrated_margin_prevents_commitment() {
        let calibration = qualified();
        let mut forecast = runtime_forecast();
        // Remove the second independent recovery domain, not a duplicate
        // profile within the first domain.
        forecast.block.cells[1].option_margins[2].margin_units = -100;
        let assessment = apply_recoverability_margin_calibration(&calibration, &forecast).unwrap();
        assert!(matches!(
            assessment.disposition,
            MarginDisposition::DoNotCommit {
                first_nonpositive_cell: CellIdentity { prefix_step: 1, .. },
                ..
            }
        ));
        assert!(assessment.minimum_lower_bound_margin_units.unwrap() <= 0);
    }

    #[test]
    fn unknown_previous_outcome_blocks_margin_assessment() {
        let calibration = qualified();
        let mut forecast = runtime_forecast();
        forecast.previous_outcome = PreviousOutcome::Unknown;
        let assessment = apply_recoverability_margin_calibration(&calibration, &forecast).unwrap();
        assert_eq!(
            assessment.disposition,
            MarginDisposition::BlockUntilReconciliation
        );
        assert_eq!(assessment.minimum_lower_bound_margin_units, None);
    }

    #[test]
    fn recovery_portfolio_identity_mismatch_invalidates_calibration() {
        let calibration = qualified();
        let mut forecast = runtime_forecast();
        forecast.block.spec.recovery_portfolio_sha256 = digest('9');
        assert_eq!(
            apply_recoverability_margin_calibration(&calibration, &forecast),
            Err(MarginCalibrationError::RuntimeScopeMismatch(
                "candidate_block_spec"
            ))
        );
    }

    #[test]
    fn cluster_split_leakage_is_rejected() {
        let mut m = manifest(19);
        m.training_cluster_ids[0] = m.calibration_units[0]
            .reference
            .independent_cluster_id
            .clone();
        assert_eq!(
            calibrate_recoverability_margin(contract(), &m),
            Err(MarginCalibrationError::ClusterSplitLeakage)
        );
    }

    #[test]
    fn duplicate_calibration_cluster_is_rejected() {
        let mut m = manifest(19);
        m.calibration_units[1].reference.independent_cluster_id = m.calibration_units[0]
            .reference
            .independent_cluster_id
            .clone();
        assert_eq!(
            calibrate_recoverability_margin(contract(), &m),
            Err(MarginCalibrationError::DuplicateCalibrationCluster)
        );
    }

    #[test]
    fn candidate_and_reference_profile_manifests_must_match() {
        let mut p = pair(0, 2);
        p.reference.cells[0].option_margins[0].failure_domain_sha256 = digest('9');
        assert_eq!(
            score_calibration_unit(&p.candidate, &p.reference, &spec()),
            Err(MarginCalibrationError::ProfileSetMismatch)
        );
    }
}
