#![forbid(unsafe_code)]

//! Cluster-level simultaneous conformal calibration for fixed trajectory blocks.
//!
//! One max residual score is admitted per independent calibration cluster.
//! This crate checks the declared protocol; an independent evidence boundary
//! must recompute and authenticate the underlying prediction/target bytes.

use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectoryBlockSpec {
    /// Observable, pre-registered context/support cell.
    pub context_scope_sha256: String,
    /// Frozen coordinate scaling and L-infinity metric identity.
    pub state_metric_sha256: String,
    /// Includes action/query portfolio and any adaptive search/selection procedure.
    pub selection_protocol_sha256: String,
    /// Complete evaluation block per independent cluster.
    pub rollouts_per_cluster: u32,
    pub action_query_cell_count: u32,
    pub horizon_steps: u32,
    pub state_dimension: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationContract {
    pub contract_sha256: String,
    pub source_manifest_sha256: String,
    pub candidate_checkpoint_sha256: String,
    pub block: TrajectoryBlockSpec,
    /// Miscoverage alpha = alpha_numerator / alpha_denominator.
    pub alpha_numerator: u32,
    pub alpha_denominator: u32,
}

/// One evaluator-side aggregate score for one independent cluster block.
/// This is not an episode or transition row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterBlockScore {
    pub independent_cluster_id: String,
    /// Evaluator-only label; never passed to candidate inference.
    pub environment_family_id: String,
    pub context_scope_sha256: String,
    pub state_metric_sha256: String,
    pub selection_protocol_sha256: String,
    /// Digest of the complete block inventory and source prediction/target records.
    pub block_manifest_sha256: String,
    /// Digest from the independent residual-score computation boundary.
    pub aggregate_score_receipt_sha256: String,
    pub rollouts_per_cluster: u32,
    pub action_query_cell_count: u32,
    pub horizon_steps: u32,
    pub state_dimension: u32,
    /// Maximum coordinate-wise absolute residual over the entire frozen block.
    pub max_abs_error_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationManifest {
    /// Cluster IDs used to fit model parameters.
    pub training_cluster_ids: Vec<String>,
    /// Cluster IDs used to choose/freeze any residual scale or normalization.
    pub scale_validation_cluster_ids: Vec<String>,
    /// Evaluator-only held-out IDs; included to prevent calibration/test leakage.
    pub held_out_evaluation_cluster_ids: Vec<String>,
    /// Exactly one aggregate score per independent calibration cluster.
    pub calibration_units: Vec<ClusterBlockScore>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedTrajectoryEnvelope {
    /// Sent to runtime only after the evidence-ingestion boundary authenticates the manifest.
    pub contract: CalibrationContract,
    pub radius_units: u64,
    pub quantile_rank_one_based: usize,
    pub calibration_cluster_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationDiagnostics {
    /// Evaluator-only diagnostics; do not expose hidden family labels to the candidate.
    pub calibration_family_count: usize,
    pub training_cluster_count: usize,
    pub validation_cluster_count: usize,
    pub held_out_evaluation_cluster_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationResult {
    pub envelope: QualifiedTrajectoryEnvelope,
    pub diagnostics: CalibrationDiagnostics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateTrajectoryForecast {
    /// Candidate-visible prediction identity, materialized before oracle residuals are attached.
    pub candidate_view_sha256: String,
    pub predicted_trajectory_sha256: String,
    pub contract_sha256: String,
    pub source_manifest_sha256: String,
    pub candidate_checkpoint_sha256: String,
    pub block: TrajectoryBlockSpec,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppliedTrajectoryEnvelope {
    pub candidate_view_sha256: String,
    pub predicted_trajectory_sha256: String,
    pub radius_units: u64,
    pub horizon_steps: u32,
    pub state_dimension: u32,
    pub action_query_cell_count: u32,
    pub calibration_cluster_count: usize,
    pub nominal_coverage_numerator: u32,
    pub nominal_coverage_denominator: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    InvalidMiscoverage,
    InvalidBlockSpec(&'static str),
    DuplicateClusterInSplit,
    ClusterSplitLeakage,
    DuplicateCalibrationCluster,
    InvalidCalibrationInventory(&'static str),
    NoCalibrationClusters,
    InsufficientCalibrationClusters {
        required_rank: usize,
        available_clusters: usize,
    },
    RuntimeScopeMismatch(&'static str),
}

pub fn calibrate_cluster_trajectory_envelope(
    contract: CalibrationContract,
    manifest: &CalibrationManifest,
) -> Result<CalibrationResult, CalibrationError> {
    validate_contract(&contract)?;
    validate_split_manifest(manifest)?;

    if manifest.calibration_units.is_empty() {
        return Err(CalibrationError::NoCalibrationClusters);
    }

    let mut calibration_ids = HashSet::new();
    let mut families = HashSet::new();
    let mut scores = Vec::with_capacity(manifest.calibration_units.len());

    let forbidden_ids: HashSet<&str> = manifest
        .training_cluster_ids
        .iter()
        .chain(manifest.scale_validation_cluster_ids.iter())
        .chain(manifest.held_out_evaluation_cluster_ids.iter())
        .map(String::as_str)
        .collect();

    for unit in &manifest.calibration_units {
        if unit.independent_cluster_id.trim().is_empty() {
            return Err(CalibrationError::EmptyIdentity("independent_cluster_id"));
        }
        if unit.environment_family_id.trim().is_empty() {
            return Err(CalibrationError::EmptyIdentity("environment_family_id"));
        }
        if !is_sha256_hex(&unit.block_manifest_sha256) {
            return Err(CalibrationError::InvalidSha256("block_manifest_sha256"));
        }
        if !is_sha256_hex(&unit.aggregate_score_receipt_sha256) {
            return Err(CalibrationError::InvalidSha256(
                "aggregate_score_receipt_sha256",
            ));
        }
        if forbidden_ids.contains(unit.independent_cluster_id.as_str()) {
            return Err(CalibrationError::ClusterSplitLeakage);
        }
        if !calibration_ids.insert(unit.independent_cluster_id.clone()) {
            return Err(CalibrationError::DuplicateCalibrationCluster);
        }
        validate_unit_inventory(unit, &contract.block)?;
        families.insert(unit.environment_family_id.clone());
        scores.push(unit.max_abs_error_units);
    }

    let rank = finite_sample_quantile_rank(
        scores.len(),
        contract.alpha_numerator,
        contract.alpha_denominator,
    )?;
    if rank > scores.len() {
        return Err(CalibrationError::InsufficientCalibrationClusters {
            required_rank: rank,
            available_clusters: scores.len(),
        });
    }

    scores.sort_unstable();
    let radius_units = scores[rank - 1];

    Ok(CalibrationResult {
        envelope: QualifiedTrajectoryEnvelope {
            contract,
            radius_units,
            quantile_rank_one_based: rank,
            calibration_cluster_count: scores.len(),
        },
        diagnostics: CalibrationDiagnostics {
            calibration_family_count: families.len(),
            training_cluster_count: manifest.training_cluster_ids.len(),
            validation_cluster_count: manifest.scale_validation_cluster_ids.len(),
            held_out_evaluation_cluster_count: manifest.held_out_evaluation_cluster_ids.len(),
        },
    })
}

/// Applies the frozen cluster-level envelope only to an exactly matching runtime block.
pub fn apply_trajectory_envelope(
    calibration: &QualifiedTrajectoryEnvelope,
    forecast: &CandidateTrajectoryForecast,
) -> Result<AppliedTrajectoryEnvelope, CalibrationError> {
    if !is_sha256_hex(&forecast.candidate_view_sha256) {
        return Err(CalibrationError::InvalidSha256("candidate_view_sha256"));
    }
    if !is_sha256_hex(&forecast.predicted_trajectory_sha256) {
        return Err(CalibrationError::InvalidSha256("predicted_trajectory_sha256"));
    }
    if forecast.contract_sha256 != calibration.contract.contract_sha256 {
        return Err(CalibrationError::RuntimeScopeMismatch("contract_sha256"));
    }
    if forecast.source_manifest_sha256 != calibration.contract.source_manifest_sha256 {
        return Err(CalibrationError::RuntimeScopeMismatch("source_manifest_sha256"));
    }
    if forecast.candidate_checkpoint_sha256 != calibration.contract.candidate_checkpoint_sha256 {
        return Err(CalibrationError::RuntimeScopeMismatch(
            "candidate_checkpoint_sha256",
        ));
    }
    if forecast.block != calibration.contract.block {
        return Err(CalibrationError::RuntimeScopeMismatch("trajectory_block_spec"));
    }

    Ok(AppliedTrajectoryEnvelope {
        candidate_view_sha256: forecast.candidate_view_sha256.clone(),
        predicted_trajectory_sha256: forecast.predicted_trajectory_sha256.clone(),
        radius_units: calibration.radius_units,
        horizon_steps: calibration.contract.block.horizon_steps,
        state_dimension: calibration.contract.block.state_dimension,
        action_query_cell_count: calibration.contract.block.action_query_cell_count,
        calibration_cluster_count: calibration.calibration_cluster_count,
        nominal_coverage_numerator: calibration.contract.alpha_denominator
            - calibration.contract.alpha_numerator,
        nominal_coverage_denominator: calibration.contract.alpha_denominator,
    })
}

pub fn finite_sample_quantile_rank(
    calibration_clusters: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
) -> Result<usize, CalibrationError> {
    if alpha_denominator == 0
        || alpha_numerator == 0
        || alpha_numerator >= alpha_denominator
    {
        return Err(CalibrationError::InvalidMiscoverage);
    }
    let n_plus_one = (calibration_clusters as u128)
        .checked_add(1)
        .ok_or(CalibrationError::InvalidMiscoverage)?;
    let coverage_numerator = u128::from(alpha_denominator - alpha_numerator);
    let denominator = u128::from(alpha_denominator);
    let product = n_plus_one
        .checked_mul(coverage_numerator)
        .ok_or(CalibrationError::InvalidMiscoverage)?;
    let rank = product
        .checked_add(denominator - 1)
        .ok_or(CalibrationError::InvalidMiscoverage)?
        / denominator;
    usize::try_from(rank).map_err(|_| CalibrationError::InvalidMiscoverage)
}

fn validate_contract(contract: &CalibrationContract) -> Result<(), CalibrationError> {
    for (value, name) in [
        (&contract.contract_sha256, "contract_sha256"),
        (&contract.source_manifest_sha256, "source_manifest_sha256"),
        (
            &contract.candidate_checkpoint_sha256,
            "candidate_checkpoint_sha256",
        ),
        (
            &contract.block.context_scope_sha256,
            "context_scope_sha256",
        ),
        (&contract.block.state_metric_sha256, "state_metric_sha256"),
        (
            &contract.block.selection_protocol_sha256,
            "selection_protocol_sha256",
        ),
    ] {
        if !is_sha256_hex(value) {
            return Err(CalibrationError::InvalidSha256(name));
        }
    }
    if contract.alpha_denominator == 0
        || contract.alpha_numerator == 0
        || contract.alpha_numerator >= contract.alpha_denominator
    {
        return Err(CalibrationError::InvalidMiscoverage);
    }
    if contract.block.rollouts_per_cluster == 0 {
        return Err(CalibrationError::InvalidBlockSpec("rollouts_per_cluster"));
    }
    if contract.block.action_query_cell_count == 0 {
        return Err(CalibrationError::InvalidBlockSpec(
            "action_query_cell_count",
        ));
    }
    if contract.block.horizon_steps == 0 {
        return Err(CalibrationError::InvalidBlockSpec("horizon_steps"));
    }
    if contract.block.state_dimension == 0 {
        return Err(CalibrationError::InvalidBlockSpec("state_dimension"));
    }
    Ok(())
}

fn validate_split_manifest(manifest: &CalibrationManifest) -> Result<(), CalibrationError> {
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
        if seen.contains(&unit.independent_cluster_id) {
            return Err(CalibrationError::ClusterSplitLeakage);
        }
    }
    Ok(())
}

fn insert_split_id(id: &str, seen: &mut HashSet<String>) -> Result<(), CalibrationError> {
    if id.trim().is_empty() {
        return Err(CalibrationError::EmptyIdentity("split_cluster_id"));
    }
    if !seen.insert(id.to_owned()) {
        return Err(CalibrationError::DuplicateClusterInSplit);
    }
    Ok(())
}

fn validate_unit_inventory(
    unit: &ClusterBlockScore,
    block: &TrajectoryBlockSpec,
) -> Result<(), CalibrationError> {
    if unit.context_scope_sha256 != block.context_scope_sha256 {
        return Err(CalibrationError::InvalidCalibrationInventory(
            "context_scope_sha256",
        ));
    }
    if unit.state_metric_sha256 != block.state_metric_sha256 {
        return Err(CalibrationError::InvalidCalibrationInventory(
            "state_metric_sha256",
        ));
    }
    if unit.selection_protocol_sha256 != block.selection_protocol_sha256 {
        return Err(CalibrationError::InvalidCalibrationInventory(
            "selection_protocol_sha256",
        ));
    }
    if unit.rollouts_per_cluster != block.rollouts_per_cluster {
        return Err(CalibrationError::InvalidCalibrationInventory(
            "rollouts_per_cluster",
        ));
    }
    if unit.action_query_cell_count != block.action_query_cell_count {
        return Err(CalibrationError::InvalidCalibrationInventory(
            "action_query_cell_count",
        ));
    }
    if unit.horizon_steps != block.horizon_steps {
        return Err(CalibrationError::InvalidCalibrationInventory("horizon_steps"));
    }
    if unit.state_dimension != block.state_dimension {
        return Err(CalibrationError::InvalidCalibrationInventory(
            "state_dimension",
        ));
    }
    Ok(())
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

    fn block() -> TrajectoryBlockSpec {
        TrajectoryBlockSpec {
            context_scope_sha256: digest('a'),
            state_metric_sha256: digest('b'),
            selection_protocol_sha256: digest('c'),
            rollouts_per_cluster: 4,
            action_query_cell_count: 3,
            horizon_steps: 8,
            state_dimension: 12,
        }
    }

    fn contract() -> CalibrationContract {
        CalibrationContract {
            contract_sha256: digest('d'),
            source_manifest_sha256: digest('e'),
            candidate_checkpoint_sha256: digest('f'),
            block: block(),
            alpha_numerator: 1,
            alpha_denominator: 20,
        }
    }

    fn unit(index: usize, score: u64) -> ClusterBlockScore {
        ClusterBlockScore {
            independent_cluster_id: format!("calibration-cluster-{index}"),
            environment_family_id: format!("family-{}", index % 3),
            context_scope_sha256: digest('a'),
            state_metric_sha256: digest('b'),
            selection_protocol_sha256: digest('c'),
            block_manifest_sha256: digest('1'),
            aggregate_score_receipt_sha256: digest('2'),
            rollouts_per_cluster: 4,
            action_query_cell_count: 3,
            horizon_steps: 8,
            state_dimension: 12,
            max_abs_error_units: score,
        }
    }

    fn manifest(count: usize) -> CalibrationManifest {
        CalibrationManifest {
            training_cluster_ids: vec!["training-cluster-1".into()],
            scale_validation_cluster_ids: vec!["validation-cluster-1".into()],
            held_out_evaluation_cluster_ids: vec!["evaluation-cluster-1".into()],
            calibration_units: (0..count)
                .map(|index| unit(index, index as u64 + 10))
                .collect(),
        }
    }

    fn qualified() -> QualifiedTrajectoryEnvelope {
        calibrate_cluster_trajectory_envelope(contract(), &manifest(19))
            .unwrap()
            .envelope
    }

    fn forecast() -> CandidateTrajectoryForecast {
        CandidateTrajectoryForecast {
            candidate_view_sha256: digest('9'),
            predicted_trajectory_sha256: digest('8'),
            contract_sha256: digest('d'),
            source_manifest_sha256: digest('e'),
            candidate_checkpoint_sha256: digest('f'),
            block: block(),
        }
    }

    #[test]
    fn nineteen_independent_clusters_support_nominal_95_percent_block_coverage() {
        let result = calibrate_cluster_trajectory_envelope(contract(), &manifest(19)).unwrap();
        assert_eq!(result.envelope.quantile_rank_one_based, 19);
        assert_eq!(result.envelope.radius_units, 28);
        assert_eq!(result.envelope.calibration_cluster_count, 19);
        assert_eq!(result.diagnostics.calibration_family_count, 3);
    }

    #[test]
    fn eighteen_clusters_cannot_emit_a_finite_95_percent_tube() {
        assert_eq!(
            calibrate_cluster_trajectory_envelope(contract(), &manifest(18)),
            Err(CalibrationError::InsufficientCalibrationClusters {
                required_rank: 19,
                available_clusters: 18,
            })
        );
    }

    #[test]
    fn repeated_rows_from_one_cluster_cannot_inflate_independent_support() {
        let mut m = manifest(19);
        m.calibration_units[1].independent_cluster_id =
            m.calibration_units[0].independent_cluster_id.clone();
        assert_eq!(
            calibrate_cluster_trajectory_envelope(contract(), &m),
            Err(CalibrationError::DuplicateCalibrationCluster)
        );
    }

    #[test]
    fn training_validation_and_evaluation_cluster_leakage_is_rejected() {
        let mut m = manifest(19);
        m.training_cluster_ids[0] = m.calibration_units[0].independent_cluster_id.clone();
        assert_eq!(
            calibrate_cluster_trajectory_envelope(contract(), &m),
            Err(CalibrationError::ClusterSplitLeakage)
        );
    }

    #[test]
    fn incomplete_rollout_block_is_rejected() {
        let mut m = manifest(19);
        m.calibration_units[0].rollouts_per_cluster = 3;
        assert_eq!(
            calibrate_cluster_trajectory_envelope(contract(), &m),
            Err(CalibrationError::InvalidCalibrationInventory(
                "rollouts_per_cluster"
            ))
        );
    }

    #[test]
    fn one_large_prefix_error_controls_the_joint_cluster_radius() {
        let mut m = manifest(19);
        m.calibration_units[0].max_abs_error_units = 200;
        let result = calibrate_cluster_trajectory_envelope(contract(), &m).unwrap();
        assert_eq!(result.envelope.radius_units, 200);
    }

    #[test]
    fn candidate_checkpoint_mismatch_invalidates_the_tube() {
        let mut f = forecast();
        f.candidate_checkpoint_sha256 = digest('0');
        assert_eq!(
            apply_trajectory_envelope(&qualified(), &f),
            Err(CalibrationError::RuntimeScopeMismatch(
                "candidate_checkpoint_sha256"
            ))
        );
    }

    #[test]
    fn observable_context_scope_mismatch_invalidates_the_tube() {
        let mut f = forecast();
        f.block.context_scope_sha256 = digest('0');
        assert_eq!(
            apply_trajectory_envelope(&qualified(), &f),
            Err(CalibrationError::RuntimeScopeMismatch(
                "trajectory_block_spec"
            ))
        );
    }

    #[test]
    fn changed_selection_protocol_invalidates_the_tube() {
        let mut f = forecast();
        f.block.selection_protocol_sha256 = digest('0');
        assert_eq!(
            apply_trajectory_envelope(&qualified(), &f),
            Err(CalibrationError::RuntimeScopeMismatch(
                "trajectory_block_spec"
            ))
        );
    }

    #[test]
    fn longer_horizon_or_changed_dimension_invalidates_the_tube() {
        let mut f = forecast();
        f.block.horizon_steps = 9;
        assert_eq!(
            apply_trajectory_envelope(&qualified(), &f),
            Err(CalibrationError::RuntimeScopeMismatch(
                "trajectory_block_spec"
            ))
        );
        let mut f2 = forecast();
        f2.block.state_dimension = 16;
        assert_eq!(
            apply_trajectory_envelope(&qualified(), &f2),
            Err(CalibrationError::RuntimeScopeMismatch(
                "trajectory_block_spec"
            ))
        );
    }

    #[test]
    fn malformed_aggregate_score_receipt_is_rejected() {
        let mut m = manifest(19);
        m.calibration_units[0].aggregate_score_receipt_sha256 = "unverified".into();
        assert_eq!(
            calibrate_cluster_trajectory_envelope(contract(), &m),
            Err(CalibrationError::InvalidSha256(
                "aggregate_score_receipt_sha256"
            ))
        );
    }

    #[test]
    fn alpha_must_be_strictly_between_zero_and_one() {
        let mut c = contract();
        c.alpha_numerator = 0;
        assert_eq!(
            calibrate_cluster_trajectory_envelope(c, &manifest(19)),
            Err(CalibrationError::InvalidMiscoverage)
        );
    }

    #[test]
    fn applied_tube_binds_to_candidate_forecast_digest() {
        let tube = apply_trajectory_envelope(&qualified(), &forecast()).unwrap();
        assert_eq!(tube.radius_units, 28);
        assert_eq!(tube.horizon_steps, 8);
        assert_eq!(tube.state_dimension, 12);
        assert_eq!(tube.action_query_cell_count, 3);
        assert_eq!(tube.candidate_view_sha256, digest('9'));
        assert_eq!(tube.predicted_trajectory_sha256, digest('8'));
        assert_eq!(tube.nominal_coverage_numerator, 19);
        assert_eq!(tube.nominal_coverage_denominator, 20);
    }
}
