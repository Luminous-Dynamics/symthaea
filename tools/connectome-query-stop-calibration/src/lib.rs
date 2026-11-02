#![forbid(unsafe_code)]

//! Finite-sample, volume-disjoint calibration of query-specific future proofreading drift.
//!
//! The calibration unit is one independent source-volume trajectory within a frozen query
//! stratum and future edit horizon. For each trajectory, the nonnegative residual is
//! max(0, future_max_drift_ratio - observed_max_drift_ratio). A split-conformal upper
//! order statistic is added to the target's observed drift ratio.
//!
//! The result is a marginal statement under exchangeability of calibration and target
//! trajectories within the declared stratum. It is not conditional coverage, does not
//! survive arbitrary domain shift, and does not certify biological completeness.

use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationTrajectory {
    pub trajectory_id: String,
    pub source_volume_id: String,
    /// Highest shared animal/session/archive grouping that must be independent across units.
    pub independence_cluster_id: String,
    pub evidence_sha256: String,
    /// Maximum endpoint drift relative to its rolling-origin anchor in the observed window,
    /// already divided by the query's pre-registered materiality threshold.
    pub observed_max_drift_ratio: f64,
    /// Maximum endpoint drift relative to the same rolling-origin anchor in the future horizon,
    /// already divided by the identical materiality threshold.
    pub future_max_drift_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryStopCalibrationInput {
    pub profile_id: String,
    pub calibration_profile_sha256: String,
    pub group_disjointness_report_sha256: String,
    pub exchangeability_assessment_sha256: String,
    pub query_id: String,
    pub query_definition_sha256: String,
    pub analysis_code_sha256: String,
    pub metric_id: String,
    pub unit_id: String,
    /// Frozen stratum should encode all declared transport conditions (for example species,
    /// workflow and volume class). It is an identity, not an inferred embedding.
    pub calibration_stratum_id: String,
    /// Positive materiality tolerance in the original metric's units.
    pub materiality_threshold: f64,
    /// Fixed future horizon, expressed as committed edit count after the rolling origin.
    pub future_horizon_edits: u64,
    /// Miscoverage target in (0,1), fixed before looking at calibration scores.
    pub alpha: f64,
    pub target_source_volume_id: String,
    pub target_independence_cluster_id: String,
    /// Supplied by an independent group-disjoint split auditor; adapters must verify the receipt.
    pub target_is_disjoint_from_calibration: bool,
    /// Explicit, evidence-backed assessment that target and calibration trajectory units are
    /// exchangeable within the frozen stratum and horizon.
    pub target_exchangeability_supported: bool,
    /// Observed historical max-drift ratio for the target rolling origin.
    pub target_observed_max_drift_ratio: f64,
    /// Current identified-set hull width divided by the query materiality threshold.
    pub target_current_uncertainty_width_ratio: f64,
    pub calibration_trajectories: Vec<CalibrationTrajectory>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopCalibrationStatus {
    InsufficientCalibration,
    NotCalibratedGroupOverlap,
    NotCalibratedExchangeability,
    ProjectedStableWithinHorizon,
    ProjectedUnstableWithinHorizon,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryStopCalibrationReport {
    pub report_version: &'static str,
    pub profile_id: String,
    pub calibration_profile_sha256: String,
    pub group_disjointness_report_sha256: String,
    pub exchangeability_assessment_sha256: String,
    pub target_exchangeability_supported: bool,
    pub query_id: String,
    pub query_definition_sha256: String,
    pub analysis_code_sha256: String,
    pub metric_id: String,
    pub unit_id: String,
    pub calibration_stratum_id: String,
    pub future_horizon_edits: u64,
    pub alpha: f64,
    pub target_source_volume_id: String,
    pub calibration_trajectory_count: usize,
    /// 1-based conformal rank k = ceil((n+1)*(1-alpha)).
    pub order_statistic_rank: usize,
    /// None when k > n; no finite sample bound is available.
    pub residual_quantile_ratio: Option<f64>,
    pub target_observed_max_drift_ratio: f64,
    pub target_current_uncertainty_width_ratio: f64,
    /// None when calibration is insufficient.
    pub projected_future_drift_upper_ratio: Option<f64>,
    pub status: StopCalibrationStatus,
    /// Always false: a marginal future-drift bound is not universal or biological completeness.
    pub universal_connectome_completeness_claimed: bool,
    /// Always false: this is a calibrated finite horizon, not a statement about all future edits.
    pub all_future_edits_certified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopCalibrationError {
    InvalidIdentifier { kind: &'static str, value: String },
    InvalidSha256 { kind: &'static str, value: String },
    InvalidAlpha,
    InvalidMaterialityThreshold,
    InvalidFutureHorizon,
    InvalidRatio { kind: &'static str, value: String },
    DuplicateTrajectoryId(String),
    DuplicateSourceVolume(String),
    DuplicateIndependenceCluster(String),
    TargetVolumeAppearsInCalibration(String),
    TargetClusterAppearsInCalibration(String),
    EmptySourceVolume(String),
    RankOverflow,
}

impl fmt::Display for StopCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { kind, value } => {
                write!(f, "invalid {kind} identifier: {value:?}")
            }
            Self::InvalidSha256 { kind, value } => {
                write!(f, "invalid SHA-256 for {kind}: {value:?}")
            }
            Self::InvalidAlpha => write!(f, "alpha must be finite and strictly between 0 and 1"),
            Self::InvalidMaterialityThreshold => {
                write!(f, "materiality threshold must be positive and finite")
            }
            Self::InvalidFutureHorizon => write!(f, "future edit horizon must be positive"),
            Self::InvalidRatio { kind, value } => {
                write!(f, "{kind} ratio must be finite and non-negative, got {value}")
            }
            Self::DuplicateTrajectoryId(value) => {
                write!(f, "duplicate calibration trajectory ID: {value}")
            }
            Self::DuplicateSourceVolume(value) => {
                write!(f, "source volume occurs more than once in calibration: {value}")
            }
            Self::DuplicateIndependenceCluster(value) => {
                write!(f, "independence cluster occurs more than once in calibration: {value}")
            }
            Self::TargetVolumeAppearsInCalibration(value) => {
                write!(f, "target source volume also appears in calibration: {value}")
            }
            Self::TargetClusterAppearsInCalibration(value) => {
                write!(f, "target independence cluster also appears in calibration: {value}")
            }
            Self::EmptySourceVolume(value) => {
                write!(f, "trajectory {value} has an empty source volume ID")
            }
            Self::RankOverflow => write!(f, "finite-sample conformal order-statistic rank overflow"),
        }
    }
}

impl std::error::Error for StopCalibrationError {}

/// 1-based finite-sample split-conformal upper quantile rank.
///
/// If this rank exceeds n, the caller must return no finite bound / abstain.
pub fn conformal_order_statistic_rank(
    sample_count: usize,
    alpha: f64,
) -> Result<usize, StopCalibrationError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(StopCalibrationError::InvalidAlpha);
    }
    let rank_float = ((sample_count as f64) + 1.0) * (1.0 - alpha);
    if !rank_float.is_finite() || rank_float > (usize::MAX as f64) {
        return Err(StopCalibrationError::RankOverflow);
    }
    let rank = rank_float.ceil();
    if rank < 1.0 {
        return Ok(1);
    }
    Ok(rank as usize)
}

/// Calibrate a finite-horizon upper bound on future query drift.
///
/// The residual score for calibration trajectory i is:
/// max(0, future_max_drift_ratio_i - observed_max_drift_ratio_i).
///
/// For target observed ratio O*, the projected upper bound is O* + q_(k), where q_(k) is
/// the k-th ordered calibration residual. If k > n, status is InsufficientCalibration.
/// A group overlap is retained as NotCalibratedGroupOverlap and never converted to stability.
pub fn calibrate_query_stop(
    input: &QueryStopCalibrationInput,
) -> Result<QueryStopCalibrationReport, StopCalibrationError> {
    validate_input(input)?;

    let n = input.calibration_trajectories.len();
    let rank = conformal_order_statistic_rank(n, input.alpha)?;
    let mut base = report_base(input, n, rank);

    if !input.target_is_disjoint_from_calibration {
        base.status = StopCalibrationStatus::NotCalibratedGroupOverlap;
        return Ok(base);
    }
    if !input.target_exchangeability_supported {
        base.status = StopCalibrationStatus::NotCalibratedExchangeability;
        return Ok(base);
    }

    if rank > n {
        base.status = StopCalibrationStatus::InsufficientCalibration;
        return Ok(base);
    }

    let mut residuals: Vec<f64> = input
        .calibration_trajectories
        .iter()
        .map(|trajectory| {
            (trajectory.future_max_drift_ratio - trajectory.observed_max_drift_ratio).max(0.0)
        })
        .collect();
    residuals.sort_by(f64::total_cmp);

    let quantile = residuals[rank - 1];
    let upper = input.target_observed_max_drift_ratio + quantile;
    if !upper.is_finite() {
        return Err(StopCalibrationError::InvalidRatio {
            kind: "projected future drift upper",
            value: upper.to_string(),
        });
    }

    base.residual_quantile_ratio = Some(quantile);
    base.projected_future_drift_upper_ratio = Some(upper);
    base.status = if input.target_current_uncertainty_width_ratio <= 1.0 && upper <= 1.0 {
        StopCalibrationStatus::ProjectedStableWithinHorizon
    } else {
        StopCalibrationStatus::ProjectedUnstableWithinHorizon
    };
    Ok(base)
}

fn report_base(
    input: &QueryStopCalibrationInput,
    n: usize,
    rank: usize,
) -> QueryStopCalibrationReport {
    QueryStopCalibrationReport {
        report_version: "0.1",
        profile_id: input.profile_id.clone(),
        calibration_profile_sha256: input.calibration_profile_sha256.clone(),
        group_disjointness_report_sha256: input.group_disjointness_report_sha256.clone(),
        exchangeability_assessment_sha256: input.exchangeability_assessment_sha256.clone(),
        target_exchangeability_supported: input.target_exchangeability_supported,
        query_id: input.query_id.clone(),
        query_definition_sha256: input.query_definition_sha256.clone(),
        analysis_code_sha256: input.analysis_code_sha256.clone(),
        metric_id: input.metric_id.clone(),
        unit_id: input.unit_id.clone(),
        calibration_stratum_id: input.calibration_stratum_id.clone(),
        future_horizon_edits: input.future_horizon_edits,
        alpha: input.alpha,
        target_source_volume_id: input.target_source_volume_id.clone(),
        calibration_trajectory_count: n,
        order_statistic_rank: rank,
        residual_quantile_ratio: None,
        target_observed_max_drift_ratio: input.target_observed_max_drift_ratio,
        target_current_uncertainty_width_ratio: input.target_current_uncertainty_width_ratio,
        projected_future_drift_upper_ratio: None,
        status: StopCalibrationStatus::InsufficientCalibration,
        universal_connectome_completeness_claimed: false,
        all_future_edits_certified: false,
    }
}

fn validate_input(input: &QueryStopCalibrationInput) -> Result<(), StopCalibrationError> {
    for (kind, value) in [
        ("profile", &input.profile_id),
        ("query", &input.query_id),
        ("metric", &input.metric_id),
        ("unit", &input.unit_id),
        ("calibration stratum", &input.calibration_stratum_id),
        ("target source volume", &input.target_source_volume_id),
        ("target independence cluster", &input.target_independence_cluster_id),
    ] {
        validate_id(kind, value)?;
    }
    validate_sha("calibration profile", &input.calibration_profile_sha256)?;
    validate_sha("group-disjointness report", &input.group_disjointness_report_sha256)?;
    validate_sha("exchangeability assessment", &input.exchangeability_assessment_sha256)?;
    validate_sha("query definition", &input.query_definition_sha256)?;
    validate_sha("analysis code", &input.analysis_code_sha256)?;

    if !input.materiality_threshold.is_finite() || input.materiality_threshold <= 0.0 {
        return Err(StopCalibrationError::InvalidMaterialityThreshold);
    }
    if input.future_horizon_edits == 0 {
        return Err(StopCalibrationError::InvalidFutureHorizon);
    }
    if !input.alpha.is_finite() || input.alpha <= 0.0 || input.alpha >= 1.0 {
        return Err(StopCalibrationError::InvalidAlpha);
    }
    validate_ratio(
        "target observed max drift",
        input.target_observed_max_drift_ratio,
    )?;
    validate_ratio(
        "target current uncertainty width",
        input.target_current_uncertainty_width_ratio,
    )?;

    let mut trajectory_ids = BTreeSet::new();
    let mut source_volumes = BTreeSet::new();
    let mut independence_clusters = BTreeSet::new();
    for trajectory in &input.calibration_trajectories {
        validate_id("trajectory", &trajectory.trajectory_id)?;
        validate_id("independence cluster", &trajectory.independence_cluster_id)?;
        if trajectory.source_volume_id.trim().is_empty()
            || trajectory.source_volume_id.trim() != trajectory.source_volume_id
        {
            return Err(StopCalibrationError::EmptySourceVolume(
                trajectory.trajectory_id.clone(),
            ));
        }
        validate_sha("calibration trajectory evidence", &trajectory.evidence_sha256)?;
        validate_ratio(
            "calibration observed max drift",
            trajectory.observed_max_drift_ratio,
        )?;
        validate_ratio(
            "calibration future max drift",
            trajectory.future_max_drift_ratio,
        )?;

        if !trajectory_ids.insert(trajectory.trajectory_id.clone()) {
            return Err(StopCalibrationError::DuplicateTrajectoryId(
                trajectory.trajectory_id.clone(),
            ));
        }
        if !source_volumes.insert(trajectory.source_volume_id.clone()) {
            return Err(StopCalibrationError::DuplicateSourceVolume(
                trajectory.source_volume_id.clone(),
            ));
        }
        if !independence_clusters.insert(trajectory.independence_cluster_id.clone()) {
            return Err(StopCalibrationError::DuplicateIndependenceCluster(
                trajectory.independence_cluster_id.clone(),
            ));
        }
        if trajectory.source_volume_id == input.target_source_volume_id {
            return Err(StopCalibrationError::TargetVolumeAppearsInCalibration(
                trajectory.source_volume_id.clone(),
            ));
        }
        if trajectory.independence_cluster_id == input.target_independence_cluster_id {
            return Err(StopCalibrationError::TargetClusterAppearsInCalibration(
                trajectory.independence_cluster_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_ratio(kind: &'static str, value: f64) -> Result<(), StopCalibrationError> {
    if !value.is_finite() || value < 0.0 {
        return Err(StopCalibrationError::InvalidRatio {
            kind,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_sha(kind: &'static str, value: &str) -> Result<(), StopCalibrationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(StopCalibrationError::InvalidSha256 {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), StopCalibrationError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(StopCalibrationError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn trajectory(index: usize, observed: f64, future: f64) -> CalibrationTrajectory {
        CalibrationTrajectory {
            trajectory_id: format!("trajectory-{index}"),
            source_volume_id: format!("volume-{index}"),
            independence_cluster_id: format!("cluster-{index}"),
            evidence_sha256: SHA_C.to_owned(),
            observed_max_drift_ratio: observed,
            future_max_drift_ratio: future,
        }
    }

    fn input(n: usize, alpha: f64) -> QueryStopCalibrationInput {
        QueryStopCalibrationInput {
            profile_id: "profile-v1".to_owned(),
            calibration_profile_sha256: SHA_A.to_owned(),
            group_disjointness_report_sha256: SHA_B.to_owned(),
            exchangeability_assessment_sha256: SHA_C.to_owned(),
            query_id: "pathway-strength".to_owned(),
            query_definition_sha256: SHA_C.to_owned(),
            analysis_code_sha256: SHA_A.to_owned(),
            metric_id: "pathway-strength".to_owned(),
            unit_id: "normalized-strength".to_owned(),
            calibration_stratum_id: "mouse-volume-workflow-v1".to_owned(),
            materiality_threshold: 0.05,
            future_horizon_edits: 100,
            alpha,
            target_source_volume_id: "target-volume".to_owned(),
            target_independence_cluster_id: "target-cluster".to_owned(),
            target_is_disjoint_from_calibration: true,
            target_exchangeability_supported: true,
            target_observed_max_drift_ratio: 0.20,
            target_current_uncertainty_width_ratio: 0.30,
            calibration_trajectories: (0..n)
                .map(|index| trajectory(index, 0.1, 0.2 + index as f64 * 0.01))
                .collect(),
        }
    }

    #[test]
    fn exact_finite_sample_rank_requires_nineteen_for_ninety_five_percent() {
        assert_eq!(conformal_order_statistic_rank(18, 0.05).unwrap(), 19);
        assert_eq!(conformal_order_statistic_rank(19, 0.05).unwrap(), 19);
        assert_eq!(conformal_order_statistic_rank(99, 0.05).unwrap(), 95);
    }

    #[test]
    fn insufficient_calibration_abstains_instead_of_returning_a_finite_bound() {
        let report = calibrate_query_stop(&input(18, 0.05)).unwrap();
        assert_eq!(report.order_statistic_rank, 19);
        assert_eq!(report.residual_quantile_ratio, None);
        assert_eq!(report.projected_future_drift_upper_ratio, None);
        assert_eq!(report.status, StopCalibrationStatus::InsufficientCalibration);
    }

    #[test]
    fn finite_conformal_bound_uses_nonnegative_residual_order_statistic() {
        let mut value = input(19, 0.05);
        for trajectory in &mut value.calibration_trajectories {
            trajectory.future_max_drift_ratio = trajectory.observed_max_drift_ratio + 0.10;
        }
        let report = calibrate_query_stop(&value).unwrap();
        assert!((report.residual_quantile_ratio.unwrap() - 0.10).abs() < 1e-12);
        assert!((report.projected_future_drift_upper_ratio.unwrap() - 0.30).abs() < 1e-12);
        assert_eq!(report.status, StopCalibrationStatus::ProjectedStableWithinHorizon);
    }

    #[test]
    fn high_future_residual_prevents_projected_stability() {
        let mut value = input(19, 0.05);
        for trajectory in &mut value.calibration_trajectories {
            trajectory.future_max_drift_ratio = trajectory.observed_max_drift_ratio + 0.90;
        }
        let report = calibrate_query_stop(&value).unwrap();
        assert!((report.projected_future_drift_upper_ratio.unwrap() - 1.10).abs() < 1e-12);
        assert_eq!(report.status, StopCalibrationStatus::ProjectedUnstableWithinHorizon);
    }

    #[test]
    fn current_uncertainty_width_above_tolerance_blocks_stability() {
        let mut value = input(19, 0.05);
        for trajectory in &mut value.calibration_trajectories {
            trajectory.future_max_drift_ratio = trajectory.observed_max_drift_ratio;
        }
        value.target_current_uncertainty_width_ratio = 1.01;
        let report = calibrate_query_stop(&value).unwrap();
        assert!((report.projected_future_drift_upper_ratio.unwrap() - 0.20).abs() < 1e-12);
        assert_eq!(report.status, StopCalibrationStatus::ProjectedUnstableWithinHorizon);
    }

    #[test]
    fn group_overlap_blocks_calibration_even_when_sample_size_is_sufficient() {
        let mut value = input(19, 0.05);
        value.target_is_disjoint_from_calibration = false;
        let report = calibrate_query_stop(&value).unwrap();
        assert_eq!(report.status, StopCalibrationStatus::NotCalibratedGroupOverlap);
        assert_eq!(report.residual_quantile_ratio, None);
    }

    #[test]
    fn unsupported_exchangeability_abstains_even_with_disjoint_volumes() {
        let mut value = input(19, 0.05);
        value.target_exchangeability_supported = false;
        let report = calibrate_query_stop(&value).unwrap();
        assert_eq!(
            report.status,
            StopCalibrationStatus::NotCalibratedExchangeability
        );
        assert_eq!(report.residual_quantile_ratio, None);
        assert_eq!(report.projected_future_drift_upper_ratio, None);
    }

    #[test]
    fn target_volume_must_not_appear_in_calibration() {
        let mut value = input(19, 0.05);
        value.calibration_trajectories[0].source_volume_id = "target-volume".to_owned();
        assert_eq!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::TargetVolumeAppearsInCalibration(
                "target-volume".to_owned()
            ))
        );
    }

    #[test]
    fn repeated_rows_from_one_source_volume_cannot_inflate_sample_size() {
        let mut value = input(19, 0.05);
        value.calibration_trajectories[1].source_volume_id =
            value.calibration_trajectories[0].source_volume_id.clone();
        assert_eq!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::DuplicateSourceVolume(
                "volume-0".to_owned()
            ))
        );
    }

    #[test]
    fn distinct_volumes_from_one_independence_cluster_do_not_count_as_independent() {
        let mut value = input(19, 0.05);
        value.calibration_trajectories[1].independence_cluster_id =
            value.calibration_trajectories[0].independence_cluster_id.clone();
        assert_eq!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::DuplicateIndependenceCluster(
                "cluster-0".to_owned()
            ))
        );
    }

    #[test]
    fn target_independence_cluster_must_be_disjoint_from_calibration() {
        let mut value = input(19, 0.05);
        value.calibration_trajectories[0].independence_cluster_id =
            "target-cluster".to_owned();
        assert_eq!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::TargetClusterAppearsInCalibration(
                "target-cluster".to_owned()
            ))
        );
    }

    #[test]
    fn duplicate_trajectory_id_is_rejected() {
        let mut value = input(19, 0.05);
        value.calibration_trajectories[1].trajectory_id =
            value.calibration_trajectories[0].trajectory_id.clone();
        assert_eq!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::DuplicateTrajectoryId(
                "trajectory-0".to_owned()
            ))
        );
    }

    #[test]
    fn alpha_outside_open_unit_interval_is_rejected() {
        for alpha in [0.0, 1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                conformal_order_statistic_rank(20, alpha),
                Err(StopCalibrationError::InvalidAlpha)
            );
        }
    }

    #[test]
    fn zero_horizon_and_invalid_materiality_threshold_are_rejected() {
        let mut value = input(19, 0.05);
        value.future_horizon_edits = 0;
        assert_eq!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::InvalidFutureHorizon)
        );

        let mut value = input(19, 0.05);
        value.materiality_threshold = 0.0;
        assert_eq!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::InvalidMaterialityThreshold)
        );
    }

    #[test]
    fn negative_or_non_finite_drift_ratios_are_rejected() {
        let mut value = input(19, 0.05);
        value.target_observed_max_drift_ratio = -0.1;
        assert!(matches!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::InvalidRatio { .. })
        ));

        let mut value = input(19, 0.05);
        value.calibration_trajectories[0].future_max_drift_ratio = f64::NAN;
        assert!(matches!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::InvalidRatio { .. })
        ));
    }

    #[test]
    fn empty_calibration_returns_insufficient_calibration() {
        let report = calibrate_query_stop(&input(0, 0.05)).unwrap();
        assert_eq!(report.calibration_trajectory_count, 0);
        assert_eq!(report.order_statistic_rank, 1);
        assert_eq!(report.status, StopCalibrationStatus::InsufficientCalibration);
    }

    #[test]
    fn deterministic_quantile_is_invariant_to_calibration_row_order() {
        let mut value = input(19, 0.05);
        let forward = calibrate_query_stop(&value).unwrap();
        value.calibration_trajectories.reverse();
        let reverse = calibrate_query_stop(&value).unwrap();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn calibration_evidence_digest_is_required() {
        let mut value = input(19, 0.05);
        value.calibration_trajectories[0].evidence_sha256 = "bad".to_owned();
        assert!(matches!(
            calibrate_query_stop(&value),
            Err(StopCalibrationError::InvalidSha256 {
                kind: "calibration trajectory evidence",
                ..
            })
        ));
    }
}
