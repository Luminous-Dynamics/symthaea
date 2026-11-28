// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline selective-risk calibrator for evidence-qualified decisions.
//!
//! Primary endpoint: probability that a new independent cluster contains at
//! least one unacceptable accepted decision. Repeated rows inside a cluster
//! never increase the independent trial count.
//!
//! Deterministic evidence, authority, scope, lineage and lifecycle gates remain
//! outside the statistical score and can never be waived by calibration.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION_V1: &str = "mace_selective_evidence_sufficiency_v0_1";
pub const RISK_BOUND_METHOD_V1: &str = "clopper_pearson_cluster_any_unsafe_v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupSpecV1 {
    pub group_id: String,
    pub parent_group_id: Option<String>,
    pub allow_parent_fallback: bool,
    pub distribution_shift_detected: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationExampleV1 {
    pub run_id: String,
    pub independent_cluster_id: String,
    pub group_id: String,
    pub sufficiency_score: f64,
    pub hard_gates_pass: bool,
    /// Evaluator-only bounded decision loss in [0, 1].
    pub decision_loss: f64,
    pub outcome_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationConfigV1 {
    pub run_id: String,
    pub calibration_data_sha256: String,
    pub calibration_split_sha256: String,
    pub group_hierarchy_sha256: String,
    pub cluster_unit_definition_sha256: String,
    pub cluster_sampling_design_sha256: String,
    pub cluster_workload_policy_sha256: String,
    pub cluster_assignment_manifest_sha256: String,
    pub maximum_observations_per_cluster: usize,
    pub cluster_design_receipt_sha256: String,
    pub sufficiency_score_definition_sha256: String,
    pub decision_loss_definition_sha256: String,
    pub decision_policy_sha256: String,
    pub candidate_code_sha256: String,
    pub evaluator_code_sha256: String,
    pub threshold_grid_sha256: String,
    pub threshold_grid_source: ThresholdGridSourceV1,
    pub threshold_grid_source_sha256: String,
    pub threshold_grid_precommitment_sha256: String,
    pub threshold_grid_committed_at_ms: u64,
    pub calibration_opened_at_ms: u64,
    pub thresholds: Vec<f64>,
    pub threshold_selection_method: ThresholdSelectionMethodV1,
    pub risk_target_alpha: f64,
    pub familywise_delta: f64,
    pub unacceptable_loss_threshold: f64,
    pub minimum_independent_clusters: usize,
    pub exchangeability_justified_by_design: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdSelectionMethodV1 {
    FixedSequenceCluster,
    BonferroniSimultaneous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdGridSourceV1 {
    DomainPredeclared,
    IndependentPilot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskStatusV1 {
    Certified,
    NotEstimableNoCalibrationClusters,
    NotEstimableInsufficientCalibrationClusters,
    NotEstimableExchangeabilityNotEstablished,
    UnsupportedDistributionShift,
    ParentGroupFallback,
    RiskExceedsTarget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificateScopeV1 {
    GroupConditional,
    ParentFallback,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupCertificateV1 {
    pub schema_version: String,
    pub record_type: String,
    pub certificate_sha256: String,
    pub calibration_manifest_sha256: String,
    pub decision_policy_sha256: String,
    pub sufficiency_score_definition_sha256: String,
    pub group_hierarchy_sha256: String,
    pub cluster_unit_definition_sha256: String,
    pub cluster_sampling_design_sha256: String,
    pub cluster_workload_policy_sha256: String,
    pub group_id: String,
    pub group_path: Vec<String>,
    pub parent_group_id: Option<String>,
    pub status: RiskStatusV1,
    pub calibration_cluster_count: usize,
    pub calibration_example_count: usize,
    pub accepted_example_count: usize,
    pub hard_gate_rejected_count: usize,
    pub accepted_cluster_count: usize,
    pub unsafe_cluster_count: usize,
    pub risk_target_alpha: f64,
    pub threshold: Option<f64>,
    pub empirical_cluster_unsafe_rate: Option<f64>,
    /// Bound on P(independent cluster has at least one unsafe accepted case).
    pub risk_upper_bound: Option<f64>,
    pub mean_selective_loss: Option<f64>,
    /// Probability that an independent cluster contains at least one accepted case.
    pub coverage: Option<f64>,
    /// Descriptive equal-cluster-weighted fraction of individual decisions accepted.
    pub decision_coverage: Option<f64>,
    pub multiplicity_adjusted_delta: Option<f64>,
    pub risk_bound_method: String,
    pub certificate_scope: CertificateScopeV1,
    pub parent_fallback_group_id: Option<String>,
    pub parent_fallback_certificate_sha256: Option<String>,
    pub distribution_shift_detected: bool,
    pub minimum_cluster_requirement: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationManifestV1 {
    pub schema_version: String,
    pub record_type: String,
    pub run_id: String,
    pub calibration_data_sha256: String,
    pub calibration_split_sha256: String,
    pub calibration_cluster_count: usize,
    pub calibration_example_count: usize,
    pub group_hierarchy_sha256: String,
    pub sufficiency_score_definition_sha256: String,
    pub decision_loss_definition_sha256: String,
    pub decision_policy_sha256: String,
    pub candidate_code_sha256: String,
    pub evaluator_code_sha256: String,
    pub threshold_grid_sha256: String,
    pub threshold_grid_source: ThresholdGridSourceV1,
    pub threshold_grid_source_sha256: String,
    pub threshold_grid_precommitment_sha256: String,
    pub threshold_grid_committed_at_ms: u64,
    pub calibration_opened_at_ms: u64,
    pub threshold_count: usize,
    pub group_count: usize,
    pub cluster_unit_definition_sha256: String,
    pub cluster_sampling_design_sha256: String,
    pub cluster_workload_policy_sha256: String,
    pub cluster_assignment_manifest_sha256: String,
    pub maximum_observations_per_cluster: usize,
    pub cluster_design_receipt_sha256: String,
    pub calibration_method: String,
    pub risk_target_alpha: f64,
    pub familywise_delta: f64,
    pub multiplicity_method: String,
    pub resampling_unit: String,
    pub exchangeability_status: String,
    pub threshold_selection: String,
    pub unacceptable_loss_threshold: f64,
    pub minimum_independent_clusters: usize,
    pub evaluator_custody: String,
    pub manifest_sha256: String,
}

impl CalibrationManifestV1 {
    /// Verify the manifest integrity commitment; this does not authenticate its issuer.
    pub fn verify_integrity(&self) -> bool {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256.clear();
        digest_serialized(&unsigned)
            .map(|digest| digest == self.manifest_sha256)
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationResultV1 {
    pub manifest: CalibrationManifestV1,
    pub certificates: Vec<GroupCertificateV1>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationRejectV1 {
    InvalidConfig,
    InvalidDigest,
    InvalidThresholdGrid,
    InvalidExample,
    MixedRun,
    UnknownGroup,
    DuplicateGroup,
    MissingParentGroup,
    GroupHierarchyCycle,
    GroupHierarchyDigestMismatch,
    CalibrationDataDigestMismatch,
    ThresholdGridDigestMismatch,
    ThresholdGridPrecommitmentMismatch,
    ClusterDesignReceiptMismatch,
    ClusterAssignmentManifestMismatch,
    DuplicateOutcomeReceipt,
    ClusterWorkloadLimitExceeded,
    SerializationFailure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionActionV1 {
    AcceptAndAct,
    AcquireEvidence,
    RetrieveMore,
    Abstain,
    Escalate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionAssessmentStatusV1 {
    AcceptedUnderCertificate,
    AcquisitionRecommended,
    RetrievalRecommended,
    AbstainedNotEstimable,
    AbstainedRiskExceedsTarget,
    RejectedHardGate,
    CertificateSuspendedShift,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionInputV1 {
    pub group_id: String,
    pub sufficiency_score: f64,
    pub sufficiency_score_definition_sha256: String,
    pub group_hierarchy_sha256: String,
    pub cluster_unit_definition_sha256: String,
    pub cluster_sampling_design_sha256: String,
    pub cluster_workload_policy_sha256: String,
    pub hard_gates_pass: bool,
    pub distribution_shift_detected: bool,
    pub evidence_acquisition_available: bool,
    pub retrieval_available: bool,
    pub escalate_on_hard_gate: bool,
    pub decision_need_sha256: String,
    pub evidence_capsule_sha256: String,
    pub policy_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionAssessmentV1 {
    pub schema_version: String,
    pub record_type: String,
    pub assessment_sha256: String,
    pub decision_need_sha256: String,
    pub evidence_capsule_sha256: String,
    pub calibration_manifest_sha256: String,
    pub group_certificate_sha256: Option<String>,
    pub policy_sha256: String,
    pub sufficiency_score_definition_sha256: String,
    pub group_hierarchy_sha256: String,
    pub cluster_unit_definition_sha256: String,
    pub cluster_sampling_design_sha256: String,
    pub cluster_workload_policy_sha256: String,
    pub group_id: String,
    pub action: DecisionActionV1,
    pub status: DecisionAssessmentStatusV1,
    pub hard_gates_pass: bool,
    pub sufficiency_score: f64,
    pub threshold: Option<f64>,
    pub risk_target_alpha: f64,
    pub risk_upper_bound: Option<f64>,
    pub distribution_shift_detected: bool,
    pub reason: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn nonempty(value: &str) -> bool { !value.trim().is_empty() && value.len() <= 512 }
fn digest_serialized<T: Serialize + ?Sized>(value: &T) -> Result<String, CalibrationRejectV1> {
    serde_json::to_vec(value).map(|bytes| sha256_hex(&bytes)).map_err(|_| CalibrationRejectV1::SerializationFailure)
}

#[derive(Serialize)]
struct ThresholdGridPrecommitmentProjectionV1<'a> {
    schema_version: &'static str,
    run_id: &'a str,
    source: ThresholdGridSourceV1,
    source_sha256: &'a str,
    threshold_grid_sha256: &'a str,
    intended_calibration_split_sha256: &'a str,
    committed_at_ms: u64,
}

#[derive(Serialize)]
struct ClusterDesignProjectionV1<'a> {
    schema_version: &'static str,
    calibration_split_sha256: &'a str,
    cluster_unit_definition_sha256: &'a str,
    cluster_sampling_design_sha256: &'a str,
    cluster_workload_policy_sha256: &'a str,
    cluster_assignment_manifest_sha256: &'a str,
    maximum_observations_per_cluster: usize,
}

#[derive(Serialize)]
struct ClusterAssignmentEntryV1<'a> {
    outcome_receipt_sha256: &'a str,
    group_id: &'a str,
    independent_cluster_id: &'a str,
}

/// Recompute the row-to-cluster mapping from unique evaluator outcome receipts.
pub fn compute_cluster_assignment_manifest_sha256_v1(
    examples: &[CalibrationExampleV1],
) -> Result<String, CalibrationRejectV1> {
    let mut entries = Vec::with_capacity(examples.len());
    let mut seen_receipts = BTreeSet::new();
    for example in examples {
        if !seen_receipts.insert(example.outcome_receipt_sha256.as_str()) {
            return Err(CalibrationRejectV1::DuplicateOutcomeReceipt);
        }
        entries.push(ClusterAssignmentEntryV1 {
            outcome_receipt_sha256: &example.outcome_receipt_sha256,
            group_id: &example.group_id,
            independent_cluster_id: &example.independent_cluster_id,
        });
    }
    entries.sort_by(|left, right| left.outcome_receipt_sha256.cmp(right.outcome_receipt_sha256));
    digest_serialized(&entries)
}

/// Recompute the threshold-grid precommitment. This binds the candidate grid
/// to its source and intended calibration split, but does not authenticate time.
pub fn compute_threshold_grid_precommitment_sha256_v1(
    config: &CalibrationConfigV1,
) -> Result<String, CalibrationRejectV1> {
    digest_serialized(&ThresholdGridPrecommitmentProjectionV1 {
        schema_version: "mace_threshold_grid_precommitment_v1",
        run_id: &config.run_id,
        source: config.threshold_grid_source,
        source_sha256: &config.threshold_grid_source_sha256,
        threshold_grid_sha256: &config.threshold_grid_sha256,
        intended_calibration_split_sha256: &config.calibration_split_sha256,
        committed_at_ms: config.threshold_grid_committed_at_ms,
    })
}

/// Recompute the cluster design receipt tying the independent unit, sampling
/// frame, workload horizon and assignment manifest to the calibration split.
pub fn compute_cluster_design_receipt_sha256_v1(
    config: &CalibrationConfigV1,
) -> Result<String, CalibrationRejectV1> {
    digest_serialized(&ClusterDesignProjectionV1 {
        schema_version: "mace_cluster_design_receipt_v1",
        calibration_split_sha256: &config.calibration_split_sha256,
        cluster_unit_definition_sha256: &config.cluster_unit_definition_sha256,
        cluster_sampling_design_sha256: &config.cluster_sampling_design_sha256,
        cluster_workload_policy_sha256: &config.cluster_workload_policy_sha256,
        cluster_assignment_manifest_sha256: &config.cluster_assignment_manifest_sha256,
        maximum_observations_per_cluster: config.maximum_observations_per_cluster,
    })
}

fn validate_config(config: &CalibrationConfigV1) -> Result<(), CalibrationRejectV1> {
    if !nonempty(&config.run_id)
        || !config.risk_target_alpha.is_finite()
        || !(0.0..1.0).contains(&config.risk_target_alpha)
        || !config.familywise_delta.is_finite()
        || !(0.0..1.0).contains(&config.familywise_delta)
        || !config.unacceptable_loss_threshold.is_finite()
        || !(0.0..=1.0).contains(&config.unacceptable_loss_threshold)
        || config.minimum_independent_clusters < 2
        || config.maximum_observations_per_cluster == 0
        || config.thresholds.is_empty() {
        return Err(CalibrationRejectV1::InvalidConfig);
    }
    for value in [
        config.calibration_data_sha256.as_str(), config.calibration_split_sha256.as_str(),
        config.group_hierarchy_sha256.as_str(),
        config.cluster_unit_definition_sha256.as_str(),
        config.cluster_sampling_design_sha256.as_str(),
        config.cluster_workload_policy_sha256.as_str(),
        config.cluster_assignment_manifest_sha256.as_str(),
        config.cluster_design_receipt_sha256.as_str(),
        config.threshold_grid_source_sha256.as_str(),
        config.threshold_grid_precommitment_sha256.as_str(),
        config.sufficiency_score_definition_sha256.as_str(),
        config.decision_loss_definition_sha256.as_str(), config.decision_policy_sha256.as_str(),
        config.candidate_code_sha256.as_str(), config.evaluator_code_sha256.as_str(),
        config.threshold_grid_sha256.as_str(),
    ] {
        if !valid_digest(value) { return Err(CalibrationRejectV1::InvalidDigest); }
    }
    if config.threshold_grid_committed_at_ms == 0
        || config.calibration_opened_at_ms == 0
        || config.threshold_grid_committed_at_ms >= config.calibration_opened_at_ms
        || config.threshold_grid_source_sha256 == config.calibration_data_sha256
        || config.threshold_grid_source_sha256 == config.calibration_split_sha256
    {
        return Err(CalibrationRejectV1::InvalidConfig);
    }
    if compute_threshold_grid_precommitment_sha256_v1(config)?
        != config.threshold_grid_precommitment_sha256
    {
        return Err(CalibrationRejectV1::ThresholdGridPrecommitmentMismatch);
    }
    if compute_cluster_design_receipt_sha256_v1(config)?
        != config.cluster_design_receipt_sha256
    {
        return Err(CalibrationRejectV1::ClusterDesignReceiptMismatch);
    }
    let mut previous = None;
    for threshold in &config.thresholds {
        if !threshold.is_finite() || !(0.0..=1.0).contains(threshold) {
            return Err(CalibrationRejectV1::InvalidThresholdGrid);
        }
        let invalid_order = match config.threshold_selection_method {
            ThresholdSelectionMethodV1::FixedSequenceCluster => previous.is_some_and(|value| *threshold >= value),
            ThresholdSelectionMethodV1::BonferroniSimultaneous => previous.is_some_and(|value| *threshold <= value),
        };
        if invalid_order { return Err(CalibrationRejectV1::InvalidThresholdGrid); }
        previous = Some(*threshold);
    }
    if digest_serialized(&config.thresholds)? != config.threshold_grid_sha256 {
        return Err(CalibrationRejectV1::ThresholdGridDigestMismatch);
    }
    Ok(())
}

fn hierarchy_depth(id: &str, groups: &BTreeMap<String, GroupSpecV1>) -> Result<usize, CalibrationRejectV1> {
    let mut current = Some(id);
    let mut seen = BTreeSet::new();
    let mut depth = 0;
    while let Some(value) = current {
        if !seen.insert(value.to_owned()) { return Err(CalibrationRejectV1::GroupHierarchyCycle); }
        let group = groups.get(value).ok_or(CalibrationRejectV1::UnknownGroup)?;
        current = group.parent_group_id.as_deref();
        if current.is_some() { depth += 1; }
    }
    Ok(depth)
}

fn group_path(id: &str, groups: &BTreeMap<String, GroupSpecV1>) -> Result<Vec<String>, CalibrationRejectV1> {
    let mut path = Vec::new();
    let mut current = Some(id);
    let mut seen = BTreeSet::new();
    while let Some(value) = current {
        if !seen.insert(value.to_owned()) { return Err(CalibrationRejectV1::GroupHierarchyCycle); }
        let group = groups.get(value).ok_or(CalibrationRejectV1::UnknownGroup)?;
        path.push(value.to_owned());
        current = group.parent_group_id.as_deref();
    }
    path.reverse();
    Ok(path)
}

fn is_descendant_or_same(candidate: &str, ancestor: &str, groups: &BTreeMap<String, GroupSpecV1>) -> bool {
    let mut current = Some(candidate);
    let mut seen = BTreeSet::new();
    while let Some(value) = current {
        if !seen.insert(value.to_owned()) { return false; }
        if value == ancestor { return true; }
        current = groups.get(value).and_then(|group| group.parent_group_id.as_deref());
    }
    false
}

fn validate_groups(specs: &[GroupSpecV1]) -> Result<BTreeMap<String, GroupSpecV1>, CalibrationRejectV1> {
    let mut groups = BTreeMap::new();
    for group in specs {
        if !nonempty(&group.group_id) { return Err(CalibrationRejectV1::InvalidConfig); }
        if groups.insert(group.group_id.clone(), group.clone()).is_some() {
            return Err(CalibrationRejectV1::DuplicateGroup);
        }
    }
    for group in specs {
        if group.parent_group_id.as_ref().is_some_and(|parent| !groups.contains_key(parent)) {
            return Err(CalibrationRejectV1::MissingParentGroup);
        }
    }
    for id in groups.keys() { hierarchy_depth(id, &groups)?; }
    Ok(groups)
}

/// Exact one-sided Clopper-Pearson upper bound for a cluster-level unsafe incidence.
pub fn clopper_pearson_upper_v1(unsafe_clusters: usize, accepted_clusters: usize, delta: f64) -> Option<f64> {
    if accepted_clusters == 0 || unsafe_clusters > accepted_clusters
        || !delta.is_finite() || !(0.0..1.0).contains(&delta) {
        return None;
    }
    if unsafe_clusters == accepted_clusters { return Some(1.0); }
    if unsafe_clusters == 0 { return Some(1.0 - delta.powf(1.0 / accepted_clusters as f64)); }

    fn log_choose(n: usize, k: usize) -> f64 {
        let k = k.min(n-k);
        (1..=k).map(|i| ((n-k+i) as f64 / i as f64).ln()).sum()
    }
    fn cdf(k: usize, n: usize, p: f64) -> f64 {
        if p <= 0.0 { return 1.0; }
        if p >= 1.0 { return 0.0; }
        if k == 0 { return (n as f64 * (-p).ln_1p()).exp(); }
        let mut mass = (log_choose(n,k) + k as f64*p.ln() + (n-k) as f64*(-p).ln_1p()).exp();
        let mut total = mass;
        for i in (1..=k).rev() {
            mass *= (i as f64 / (n-i+1) as f64) * ((1.0-p)/p);
            total += mass;
            if mass <= total*1e-15 { break; }
        }
        total.min(1.0)
    }
    let mut low = unsafe_clusters as f64 / accepted_clusters as f64;
    let mut high = 1.0;
    for _ in 0..100 {
        let mid = (low+high)*0.5;
        if cdf(unsafe_clusters, accepted_clusters, mid) > delta { low = mid; } else { high = mid; }
    }
    Some(high.clamp(0.0,1.0))
}

#[derive(Clone, Debug)]
struct Stats {
    threshold: f64,
    accepted_examples: usize,
    hard_rejected: usize,
    accepted_clusters: usize,
    unsafe_clusters: usize,
    empirical_rate: f64,
    risk_upper: f64,
    coverage: f64,
    decision_coverage: f64,
    mean_loss: f64,
}

fn threshold_stats(rows: &[&CalibrationExampleV1], threshold: f64, loss_limit: f64, delta: f64) -> Option<Stats> {
    if rows.is_empty() { return None; }
    let mut clusters: BTreeMap<&str,(usize,usize,bool,f64,usize)> = BTreeMap::new();
    let mut hard_rejected = 0;
    for row in rows {
        let entry = clusters.entry(row.independent_cluster_id.as_str()).or_insert((0,0,false,0.0,0));
        entry.0 += 1;
        if !row.hard_gates_pass { hard_rejected += 1; continue; }
        if row.sufficiency_score >= threshold {
            entry.1 += 1;
            entry.2 |= row.decision_loss > loss_limit;
            entry.3 += row.decision_loss;
            entry.4 += 1;
        }
    }
    let mut accepted_examples=0;
    let mut accepted_clusters=0;
    let mut unsafe_clusters=0;
    let mut decision_coverage_sum=0.0;
    let mut loss_sum=0.0;
    for (total,accepted,unsafe_cluster,loss,count) in clusters.values() {
        if *accepted>0 {
            accepted_clusters+=1;
            unsafe_clusters+=usize::from(*unsafe_cluster);
            accepted_examples+=accepted;
            loss_sum+=loss/ *count as f64;
        }
        decision_coverage_sum+=*accepted as f64/ *total as f64;
    }
    let risk_upper=clopper_pearson_upper_v1(unsafe_clusters,accepted_clusters,delta)?;
    Some(Stats {
        threshold, accepted_examples, hard_rejected, accepted_clusters, unsafe_clusters,
        empirical_rate:unsafe_clusters as f64/accepted_clusters as f64,
        risk_upper,
        coverage:accepted_clusters as f64/clusters.len() as f64,
        decision_coverage:decision_coverage_sum/clusters.len() as f64,
        mean_loss:loss_sum/accepted_clusters as f64,
    })
}

fn certificate_digest(certificate: &GroupCertificateV1) -> Result<String,CalibrationRejectV1> {
    let mut unsigned=certificate.clone();
    unsigned.certificate_sha256.clear();
    digest_serialized(&unsigned)
}

impl GroupCertificateV1 {
    pub fn verify_integrity(&self)->bool {
        certificate_digest(self).map(|value|value==self.certificate_sha256).unwrap_or(false)
    }
    pub fn is_usable(&self)->bool {
        self.verify_integrity()
            && matches!(self.status,RiskStatusV1::Certified|RiskStatusV1::ParentGroupFallback)
            && !self.distribution_shift_detected
            && self.threshold.is_some()
            && self.risk_upper_bound.is_some_and(|value|value<=self.risk_target_alpha)
    }
}

fn finish_certificate(mut value: GroupCertificateV1)->Result<GroupCertificateV1,CalibrationRejectV1>{
    value.certificate_sha256=certificate_digest(&value)?;
    Ok(value)
}

fn blank_certificate(config:&CalibrationConfigV1,manifest:&str,group:&GroupSpecV1,path:Vec<String>,
    example_count:usize,cluster_count:usize,hard_rejected:usize)->GroupCertificateV1{
    GroupCertificateV1{
        schema_version:SCHEMA_VERSION_V1.into(),record_type:"group_certificate".into(),
        certificate_sha256:String::new(),calibration_manifest_sha256:manifest.into(),
        decision_policy_sha256:config.decision_policy_sha256.clone(),
        sufficiency_score_definition_sha256:config.sufficiency_score_definition_sha256.clone(),
        group_hierarchy_sha256:config.group_hierarchy_sha256.clone(),
        cluster_unit_definition_sha256:config.cluster_unit_definition_sha256.clone(),
        cluster_sampling_design_sha256:config.cluster_sampling_design_sha256.clone(),
        cluster_workload_policy_sha256:config.cluster_workload_policy_sha256.clone(),
        group_id:group.group_id.clone(),group_path:path,parent_group_id:group.parent_group_id.clone(),
        status:RiskStatusV1::NotEstimableNoCalibrationClusters,
        calibration_cluster_count:cluster_count,calibration_example_count:example_count,
        accepted_example_count:0,hard_gate_rejected_count:hard_rejected,accepted_cluster_count:0,
        unsafe_cluster_count:0,risk_target_alpha:config.risk_target_alpha,threshold:None,
        empirical_cluster_unsafe_rate:None,risk_upper_bound:None,mean_selective_loss:None,coverage:None,decision_coverage:None,
        multiplicity_adjusted_delta:None,risk_bound_method:RISK_BOUND_METHOD_V1.into(),
        certificate_scope:CertificateScopeV1::GroupConditional,parent_fallback_group_id:None,
        parent_fallback_certificate_sha256:None,distribution_shift_detected:group.distribution_shift_detected,
        minimum_cluster_requirement:config.minimum_independent_clusters
    }
}

pub fn calibrate_selective_evidence_v1(config:&CalibrationConfigV1,group_specs:&[GroupSpecV1],
    examples:&[CalibrationExampleV1])->Result<CalibrationResultV1,CalibrationRejectV1>{
    validate_config(config)?;
    let groups=validate_groups(group_specs)?;
    if groups.is_empty(){return Err(CalibrationRejectV1::InvalidConfig);}
    if digest_serialized(group_specs)?!=config.group_hierarchy_sha256{return Err(CalibrationRejectV1::GroupHierarchyDigestMismatch);}
    if digest_serialized(examples)?!=config.calibration_data_sha256{return Err(CalibrationRejectV1::CalibrationDataDigestMismatch);}
    for row in examples{
        if !nonempty(&row.run_id)||!nonempty(&row.independent_cluster_id){return Err(CalibrationRejectV1::InvalidExample);}
        if row.run_id!=config.run_id{return Err(CalibrationRejectV1::MixedRun);}
        if !groups.contains_key(&row.group_id)||!row.sufficiency_score.is_finite()
            ||!(0.0..=1.0).contains(&row.sufficiency_score)||!row.decision_loss.is_finite()
            ||!(0.0..=1.0).contains(&row.decision_loss)||!valid_digest(&row.outcome_receipt_sha256){
            return Err(CalibrationRejectV1::InvalidExample);
        }
    }

    let recomputed_cluster_assignment =
        compute_cluster_assignment_manifest_sha256_v1(examples)?;
    if recomputed_cluster_assignment != config.cluster_assignment_manifest_sha256 {
        return Err(CalibrationRejectV1::ClusterAssignmentManifestMismatch);
    }
    let mut cluster_workloads: BTreeMap<&str, usize> = BTreeMap::new();
    for example in examples {
        let count = cluster_workloads.entry(example.independent_cluster_id.as_str()).or_default();
        *count += 1;
        if *count > config.maximum_observations_per_cluster {
            return Err(CalibrationRejectV1::ClusterWorkloadLimitExceeded);
        }
    }

    let mut manifest=CalibrationManifestV1{
        schema_version:SCHEMA_VERSION_V1.into(),record_type:"calibration_manifest".into(),run_id:config.run_id.clone(),
        calibration_data_sha256:config.calibration_data_sha256.clone(),calibration_split_sha256:config.calibration_split_sha256.clone(),
        cluster_unit_definition_sha256:config.cluster_unit_definition_sha256.clone(),
        cluster_sampling_design_sha256:config.cluster_sampling_design_sha256.clone(),
        cluster_workload_policy_sha256:config.cluster_workload_policy_sha256.clone(),
        cluster_assignment_manifest_sha256:config.cluster_assignment_manifest_sha256.clone(),
        maximum_observations_per_cluster:config.maximum_observations_per_cluster,
        cluster_design_receipt_sha256:config.cluster_design_receipt_sha256.clone(),
        calibration_cluster_count:examples.iter().map(|row|row.independent_cluster_id.as_str()).collect::<BTreeSet<_>>().len(),
        calibration_example_count:examples.len(),group_hierarchy_sha256:config.group_hierarchy_sha256.clone(),
        sufficiency_score_definition_sha256:config.sufficiency_score_definition_sha256.clone(),
        decision_loss_definition_sha256:config.decision_loss_definition_sha256.clone(),
        decision_policy_sha256:config.decision_policy_sha256.clone(),candidate_code_sha256:config.candidate_code_sha256.clone(),
        evaluator_code_sha256:config.evaluator_code_sha256.clone(),threshold_grid_sha256:config.threshold_grid_sha256.clone(),
        threshold_grid_source:config.threshold_grid_source,
        threshold_grid_source_sha256:config.threshold_grid_source_sha256.clone(),
        threshold_grid_precommitment_sha256:config.threshold_grid_precommitment_sha256.clone(),
        threshold_grid_committed_at_ms:config.threshold_grid_committed_at_ms,
        calibration_opened_at_ms:config.calibration_opened_at_ms,
        threshold_count:config.thresholds.len(),group_count:groups.len(),
        calibration_method:"hierarchical_group_conditional_exact_cluster_risk".into(),
        risk_target_alpha:config.risk_target_alpha,familywise_delta:config.familywise_delta,
        multiplicity_method:match config.threshold_selection_method {
            ThresholdSelectionMethodV1::FixedSequenceCluster => "bonferroni_over_groups_fixed_sequence_over_thresholds".into(),
            ThresholdSelectionMethodV1::BonferroniSimultaneous => "bonferroni_over_group_threshold_hypotheses".into(),
        },
        resampling_unit:"independent_declared_cluster".into(),
        exchangeability_status:if config.exchangeability_justified_by_design{"justified_by_sampling_design"}else{"not_established"}.into(),
        threshold_selection:match config.threshold_selection_method {
            ThresholdSelectionMethodV1::FixedSequenceCluster => "fixed_sequence_cluster_acceptance_prefix".into(),
            ThresholdSelectionMethodV1::BonferroniSimultaneous => "simultaneous_bonferroni_on_calibration_split".into(),
        },
        unacceptable_loss_threshold:config.unacceptable_loss_threshold,
        minimum_independent_clusters:config.minimum_independent_clusters,
        evaluator_custody:"public_diagnostic_only".into(),manifest_sha256:String::new()
    };
    manifest.manifest_sha256=digest_serialized(&manifest)?;
    let hypothesis_count = match config.threshold_selection_method {
        ThresholdSelectionMethodV1::FixedSequenceCluster => groups.len(),
        ThresholdSelectionMethodV1::BonferroniSimultaneous => groups.len().checked_mul(config.thresholds.len()).ok_or(CalibrationRejectV1::InvalidConfig)?,
    };
    let adjusted_delta=config.familywise_delta/hypothesis_count as f64;

    let mut ordered:Vec<&GroupSpecV1>=groups.values().collect();
    ordered.sort_by_key(|group|hierarchy_depth(&group.group_id,&groups).unwrap_or(usize::MAX));
    let mut certificates:BTreeMap<String,GroupCertificateV1>=BTreeMap::new();
    for group in ordered{
        let rows:Vec<&CalibrationExampleV1>=examples.iter()
            .filter(|row|is_descendant_or_same(&row.group_id,&group.group_id,&groups)).collect();
        let clusters=rows.iter().map(|row|row.independent_cluster_id.as_str()).collect::<BTreeSet<_>>().len();
        let hard_rejected=rows.iter().filter(|row|!row.hard_gates_pass).count();
        let mut cert=blank_certificate(config,&manifest.manifest_sha256,group,group_path(&group.group_id,&groups)?,rows.len(),clusters,hard_rejected);
        if group.distribution_shift_detected{
            cert.status=RiskStatusV1::UnsupportedDistributionShift;
        }else if !config.exchangeability_justified_by_design{
            cert.status=RiskStatusV1::NotEstimableExchangeabilityNotEstablished;
        }else if rows.is_empty(){
            cert.status=RiskStatusV1::NotEstimableNoCalibrationClusters;
        }else if clusters<config.minimum_independent_clusters{
            cert.status=RiskStatusV1::NotEstimableInsufficientCalibrationClusters;
        }else{
            let mut stats=Vec::new();
            for threshold in &config.thresholds {
                let Some(value) = threshold_stats(&rows, *threshold, config.unacceptable_loss_threshold, adjusted_delta) else {
                    if config.threshold_selection_method == ThresholdSelectionMethodV1::FixedSequenceCluster { break; }
                    continue;
                };
                if value.accepted_clusters < config.minimum_independent_clusters {
                    if config.threshold_selection_method == ThresholdSelectionMethodV1::FixedSequenceCluster { break; }
                    continue;
                }
                let certified = value.risk_upper <= config.risk_target_alpha;
                stats.push(value);
                if !certified && config.threshold_selection_method == ThresholdSelectionMethodV1::FixedSequenceCluster {
                    break;
                }
            }
            if let Some(best)=stats.iter().filter(|s|s.risk_upper<=config.risk_target_alpha).max_by(|a,b|{
                a.coverage.total_cmp(&b.coverage)
                    .then_with(||b.risk_upper.total_cmp(&a.risk_upper))
                    .then_with(||b.threshold.total_cmp(&a.threshold))
            }){
                cert.status=RiskStatusV1::Certified;
                cert.threshold=Some(best.threshold);
                cert.accepted_example_count=best.accepted_examples;
                cert.hard_gate_rejected_count=best.hard_rejected;
                cert.accepted_cluster_count=best.accepted_clusters;
                cert.unsafe_cluster_count=best.unsafe_clusters;
                cert.empirical_cluster_unsafe_rate=Some(best.empirical_rate);
                cert.risk_upper_bound=Some(best.risk_upper);
                cert.mean_selective_loss=Some(best.mean_loss);
                cert.coverage=Some(best.coverage);
                cert.decision_coverage=Some(best.decision_coverage);
                cert.multiplicity_adjusted_delta=Some(adjusted_delta);
            }else if let Some(best)=stats.iter().min_by(|a,b|{
                a.risk_upper.total_cmp(&b.risk_upper)
                    .then_with(||b.coverage.total_cmp(&a.coverage))
                    .then_with(||b.threshold.total_cmp(&a.threshold))
            }){
                cert.status=RiskStatusV1::RiskExceedsTarget;
                cert.threshold=Some(best.threshold);
                cert.accepted_example_count=best.accepted_examples;
                cert.hard_gate_rejected_count=best.hard_rejected;
                cert.accepted_cluster_count=best.accepted_clusters;
                cert.unsafe_cluster_count=best.unsafe_clusters;
                cert.empirical_cluster_unsafe_rate=Some(best.empirical_rate);
                cert.risk_upper_bound=Some(best.risk_upper);
                cert.mean_selective_loss=Some(best.mean_loss);
                cert.coverage=Some(best.coverage);
                cert.decision_coverage=Some(best.decision_coverage);
                cert.multiplicity_adjusted_delta=Some(adjusted_delta);
            }else{
                cert.status=RiskStatusV1::NotEstimableInsufficientCalibrationClusters;
            }
        }

        if matches!(cert.status,RiskStatusV1::NotEstimableInsufficientCalibrationClusters|RiskStatusV1::NotEstimableNoCalibrationClusters)
            &&group.allow_parent_fallback
            &&let Some(parent_id)=group.parent_group_id.as_deref()
            &&let Some(parent)=certificates.get(parent_id)
            &&parent.is_usable(){
                        cert.status=RiskStatusV1::ParentGroupFallback;
                        cert.threshold=parent.threshold;
                        cert.accepted_example_count=parent.accepted_example_count;
                        cert.accepted_cluster_count=parent.accepted_cluster_count;
                        cert.unsafe_cluster_count=parent.unsafe_cluster_count;
                        cert.empirical_cluster_unsafe_rate=parent.empirical_cluster_unsafe_rate;
                        cert.risk_upper_bound=parent.risk_upper_bound;
                        cert.mean_selective_loss=parent.mean_selective_loss;
                        cert.coverage=parent.coverage;
                        cert.decision_coverage=parent.decision_coverage;
                        cert.multiplicity_adjusted_delta=parent.multiplicity_adjusted_delta;
                        cert.certificate_scope=CertificateScopeV1::ParentFallback;
                        cert.parent_fallback_group_id=Some(parent.group_id.clone());
                        cert.parent_fallback_certificate_sha256=Some(parent.certificate_sha256.clone());
        }
        certificates.insert(group.group_id.clone(),finish_certificate(cert)?);
    }
    Ok(CalibrationResultV1{manifest,certificates:certificates.into_values().collect()})
}

pub fn assess_decision_v1(input:&DecisionInputV1,certificate:&GroupCertificateV1)
    ->Result<DecisionAssessmentV1,CalibrationRejectV1>{
    if !valid_digest(&input.decision_need_sha256)||!valid_digest(&input.evidence_capsule_sha256)
        ||!valid_digest(&input.policy_sha256)
        ||!valid_digest(&input.sufficiency_score_definition_sha256)
        ||!valid_digest(&input.group_hierarchy_sha256)
        ||!input.sufficiency_score.is_finite()
        ||!(0.0..=1.0).contains(&input.sufficiency_score){return Err(CalibrationRejectV1::InvalidExample);}
    let certificate_scope_matches = certificate.decision_policy_sha256 == input.policy_sha256
        && certificate.sufficiency_score_definition_sha256 == input.sufficiency_score_definition_sha256
        && certificate.group_hierarchy_sha256 == input.group_hierarchy_sha256
        && certificate.cluster_unit_definition_sha256 == input.cluster_unit_definition_sha256
        && certificate.cluster_sampling_design_sha256 == input.cluster_sampling_design_sha256
        && certificate.cluster_workload_policy_sha256 == input.cluster_workload_policy_sha256
        && certificate.group_id == input.group_id;
    let (action,status,reason)=if !input.hard_gates_pass{
        (if input.escalate_on_hard_gate{DecisionActionV1::Escalate}else{DecisionActionV1::Abstain},
        DecisionAssessmentStatusV1::RejectedHardGate,"deterministic evidence, authority, scope, lineage or lifecycle gate failed")
    }else if input.distribution_shift_detected||certificate.status==RiskStatusV1::UnsupportedDistributionShift{
        (if input.evidence_acquisition_available{DecisionActionV1::AcquireEvidence}else{DecisionActionV1::Abstain},
        DecisionAssessmentStatusV1::CertificateSuspendedShift,"distribution shift suspends the group certificate")
    }else if certificate.decision_policy_sha256!=input.policy_sha256{
        (DecisionActionV1::Abstain,DecisionAssessmentStatusV1::AbstainedNotEstimable,
         "decision policy digest does not match the calibrated certificate")
    }else if certificate.sufficiency_score_definition_sha256!=input.sufficiency_score_definition_sha256{
        (DecisionActionV1::Abstain,DecisionAssessmentStatusV1::AbstainedNotEstimable,
         "sufficiency score definition does not match the calibrated certificate")
    }else if certificate.group_hierarchy_sha256!=input.group_hierarchy_sha256
        || certificate.cluster_unit_definition_sha256!=input.cluster_unit_definition_sha256
        || certificate.cluster_sampling_design_sha256!=input.cluster_sampling_design_sha256
        || certificate.cluster_workload_policy_sha256!=input.cluster_workload_policy_sha256 {
        (DecisionActionV1::Abstain,DecisionAssessmentStatusV1::AbstainedNotEstimable,
         "cluster unit, sampling design, workload policy or group hierarchy differs from the calibrated certificate")
    }else if !certificate.is_usable(){
        match certificate.status{
            RiskStatusV1::RiskExceedsTarget=>(if input.evidence_acquisition_available{DecisionActionV1::AcquireEvidence}else{DecisionActionV1::Abstain},
                DecisionAssessmentStatusV1::AbstainedRiskExceedsTarget,"best calibration threshold exceeds risk target"),
            _=>(if input.evidence_acquisition_available{DecisionActionV1::AcquireEvidence}else{DecisionActionV1::Abstain},
                DecisionAssessmentStatusV1::AbstainedNotEstimable,"no usable risk certificate exists for this group")
        }
    }else if certificate.group_id!=input.group_id
        ||certificate.threshold.is_none_or(|threshold|input.sufficiency_score<threshold){
        if input.retrieval_available{
            (DecisionActionV1::RetrieveMore,DecisionAssessmentStatusV1::RetrievalRecommended,"score is below the certified sufficiency threshold")
        }else if input.evidence_acquisition_available{
            (DecisionActionV1::AcquireEvidence,DecisionAssessmentStatusV1::AcquisitionRecommended,"more evidence is needed to meet the certified threshold")
        }else{
            (DecisionActionV1::Abstain,DecisionAssessmentStatusV1::AbstainedRiskExceedsTarget,"score is below the certified sufficiency threshold")
        }
    }else{
        (DecisionActionV1::AcceptAndAct,DecisionAssessmentStatusV1::AcceptedUnderCertificate,"hard gates pass and the calibrated group certificate covers this score")
    };
    let mut assessment=DecisionAssessmentV1{
        schema_version:SCHEMA_VERSION_V1.into(),record_type:"decision_assessment".into(),
        assessment_sha256:String::new(),decision_need_sha256:input.decision_need_sha256.clone(),
        evidence_capsule_sha256:input.evidence_capsule_sha256.clone(),
        calibration_manifest_sha256:certificate.calibration_manifest_sha256.clone(),
        group_certificate_sha256:Some(certificate.certificate_sha256.clone()),
        policy_sha256:input.policy_sha256.clone(),
        sufficiency_score_definition_sha256:input.sufficiency_score_definition_sha256.clone(),
        group_hierarchy_sha256:input.group_hierarchy_sha256.clone(),
        cluster_unit_definition_sha256:input.cluster_unit_definition_sha256.clone(),
        cluster_sampling_design_sha256:input.cluster_sampling_design_sha256.clone(),
        cluster_workload_policy_sha256:input.cluster_workload_policy_sha256.clone(),
        group_id:input.group_id.clone(),action,status,
        hard_gates_pass:input.hard_gates_pass,sufficiency_score:input.sufficiency_score,
        threshold:certificate.threshold,risk_target_alpha:certificate.risk_target_alpha,
        risk_upper_bound:if certificate_scope_matches && certificate.verify_integrity()
            && input.hard_gates_pass && !input.distribution_shift_detected && !certificate.distribution_shift_detected
            { certificate.risk_upper_bound } else { None },
        distribution_shift_detected:input.distribution_shift_detected||certificate.distribution_shift_detected,
        reason:reason.into()
    };
    let mut unsigned=assessment.clone();unsigned.assessment_sha256.clear();
    assessment.assessment_sha256=digest_serialized(&unsigned)?;
    Ok(assessment)
}

impl DecisionAssessmentV1{
    pub fn verify_integrity(&self)->bool{
        let mut unsigned=self.clone();unsigned.assessment_sha256.clear();
        digest_serialized(&unsigned).map(|value|value==self.assessment_sha256).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn digest(value:&str)->String{sha256_hex(value.as_bytes())}
    fn examples(count:usize,group_id:&str,score:f64,loss:f64)->Vec<CalibrationExampleV1>{
        (0..count).map(|index|CalibrationExampleV1{
            run_id:"run-001".into(),independent_cluster_id:format!("cluster-{index:04}"),
            group_id:group_id.into(),sufficiency_score:score,hard_gates_pass:true,
            decision_loss:loss,outcome_receipt_sha256:digest(&format!("outcome-{index}"))
        }).collect()
    }
    fn config(groups:&[GroupSpecV1],rows:&[CalibrationExampleV1],thresholds:Vec<f64>,minimum:usize)->CalibrationConfigV1{
        let mut config=CalibrationConfigV1{
            run_id:"run-001".into(),calibration_data_sha256:digest_serialized(rows).unwrap(),
            calibration_split_sha256:digest("split"),group_hierarchy_sha256:digest_serialized(groups).unwrap(),
            cluster_unit_definition_sha256:digest("person_profile"),
            cluster_sampling_design_sha256:digest("profile_cluster_sampling_v1"),
            cluster_workload_policy_sha256:digest("max_episodes_per_profile_20"),
            cluster_assignment_manifest_sha256:compute_cluster_assignment_manifest_sha256_v1(rows).unwrap(),
            maximum_observations_per_cluster:20,
            cluster_design_receipt_sha256:String::new(),
            sufficiency_score_definition_sha256:digest("score"),decision_loss_definition_sha256:digest("loss"),
            decision_policy_sha256:digest("policy"),candidate_code_sha256:digest("candidate"),
            evaluator_code_sha256:digest("evaluator"),threshold_grid_sha256:digest_serialized(&thresholds).unwrap(),
            threshold_grid_source:ThresholdGridSourceV1::IndependentPilot,
            threshold_grid_source_sha256:digest("pilot-split"),
            threshold_grid_precommitment_sha256:String::new(),
            threshold_grid_committed_at_ms:10,calibration_opened_at_ms:20,
            thresholds,threshold_selection_method:ThresholdSelectionMethodV1::FixedSequenceCluster,
            risk_target_alpha:0.1,familywise_delta:0.05,unacceptable_loss_threshold:0.5,
            minimum_independent_clusters:minimum,exchangeability_justified_by_design:true
        };
        config.threshold_grid_precommitment_sha256 =
            compute_threshold_grid_precommitment_sha256_v1(&config).unwrap();
        config.cluster_design_receipt_sha256 =
            compute_cluster_design_receipt_sha256_v1(&config).unwrap();
        config
    }

    #[test]
    fn zero_unsafe_bound_matches_closed_form_and_minimum_sample(){
        let at30=clopper_pearson_upper_v1(0,30,0.05).unwrap();
        let at59=clopper_pearson_upper_v1(0,59,0.05).unwrap();
        assert!(at30>0.05);assert!(at59<=0.05);
        assert!((at30-(1.0-0.05_f64.powf(1.0/30.0))).abs()<1e-12);
    }
    #[test]
    fn multiplicity_adjustment_is_more_conservative(){
        assert!(clopper_pearson_upper_v1(0,80,0.05/12.0).unwrap()
            >clopper_pearson_upper_v1(0,80,0.05).unwrap());
    }
    #[test]
    fn safe_threshold_selects_highest_coverage_certified_band(){
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let mut rows=examples(80,"root",0.9,0.0);
        for row in rows.iter_mut().take(20){row.sufficiency_score=0.2;row.decision_loss=1.0;}
        let cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        assert!(result.manifest.verify_integrity());
        let cert=&result.certificates[0];
        assert_eq!(cert.status,RiskStatusV1::Certified);
        assert!(cert.threshold.is_some_and(|value|value>=0.5));
        assert_eq!(cert.accepted_cluster_count,60);
        assert!(cert.risk_upper_bound.unwrap()<=cfg.risk_target_alpha);
        assert!(cert.verify_integrity());
    }
    #[test]
    fn hard_gate_failure_cannot_be_overridden_by_perfect_score(){
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let mut rows=examples(40,"root",1.0,0.0);
        for row in &mut rows{row.hard_gates_pass=false;}
        let cfg=config(&groups,&rows,vec![0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        let cert=&result.certificates[0];
        assert_eq!(cert.status,RiskStatusV1::NotEstimableInsufficientCalibrationClusters);
        assert_eq!(cert.accepted_example_count,0);
        let assessment=assess_decision_v1(&DecisionInputV1{
            group_id:"root".into(),sufficiency_score:1.0,
            sufficiency_score_definition_sha256:cfg.sufficiency_score_definition_sha256.clone(),
            group_hierarchy_sha256:cfg.group_hierarchy_sha256.clone(),
            cluster_unit_definition_sha256:cfg.cluster_unit_definition_sha256.clone(),
            cluster_sampling_design_sha256:cfg.cluster_sampling_design_sha256.clone(),
            cluster_workload_policy_sha256:cfg.cluster_workload_policy_sha256.clone(),
            hard_gates_pass:false,
            distribution_shift_detected:false,evidence_acquisition_available:false,retrieval_available:false,
            escalate_on_hard_gate:false,decision_need_sha256:digest("need"),
            evidence_capsule_sha256:digest("capsule"),policy_sha256:digest("policy")
        },cert).unwrap();
        assert_eq!(assessment.action,DecisionActionV1::Abstain);
        assert_eq!(assessment.status,DecisionAssessmentStatusV1::RejectedHardGate);
    }
    #[test]
    fn unsafe_clusters_are_risk_exceeds_target_not_not_estimable(){
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(100,"root",0.9,1.0);let cfg=config(&groups,&rows,vec![0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        assert_eq!(result.certificates[0].status,RiskStatusV1::RiskExceedsTarget);
        assert!(result.certificates[0].risk_upper_bound.unwrap()>cfg.risk_target_alpha);
    }
    #[test]
    fn insufficient_leaf_falls_back_only_to_certified_declared_parent(){
        let groups=vec![
            GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false},
            GroupSpecV1{group_id:"leaf".into(),parent_group_id:Some("root".into()),allow_parent_fallback:true,distribution_shift_detected:false}
        ];
        let mut rows=examples(80,"root",0.9,0.0);
        rows.push(CalibrationExampleV1{run_id:"run-001".into(),independent_cluster_id:"leaf-one".into(),
            group_id:"leaf".into(),sufficiency_score:0.9,hard_gates_pass:true,decision_loss:0.0,
            outcome_receipt_sha256:digest("leaf-one")});
        let cfg=config(&groups,&rows,vec![0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        let leaf=result.certificates.iter().find(|value|value.group_id=="leaf").unwrap();
        assert_eq!(leaf.status,RiskStatusV1::ParentGroupFallback);
        assert_eq!(leaf.parent_fallback_group_id.as_deref(),Some("root"));
        assert!(leaf.parent_fallback_certificate_sha256.is_some());assert!(leaf.verify_integrity());
    }
    #[test]
    fn shift_and_unjustified_exchangeability_suspend_certification(){
        let shifted=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:true}];
        let rows=examples(40,"root",0.9,0.0);let cfg=config(&shifted,&rows,vec![0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&shifted,&rows).unwrap();
        assert_eq!(result.certificates[0].status,RiskStatusV1::UnsupportedDistributionShift);
        let stable=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let mut cfg=config(&stable,&rows,vec![0.5,0.0],30);cfg.exchangeability_justified_by_design=false;
        let result=calibrate_selective_evidence_v1(&cfg,&stable,&rows).unwrap();
        assert_eq!(result.certificates[0].status,RiskStatusV1::NotEstimableExchangeabilityNotEstablished);
    }
    #[test]
    fn hierarchy_cycles_and_wrong_digests_are_rejected(){
        let cyclic=vec![
            GroupSpecV1{group_id:"a".into(),parent_group_id:Some("b".into()),allow_parent_fallback:false,distribution_shift_detected:false},
            GroupSpecV1{group_id:"b".into(),parent_group_id:Some("a".into()),allow_parent_fallback:false,distribution_shift_detected:false}
        ];
        let rows=examples(40,"a",0.9,0.0);let cfg=config(&cyclic,&rows,vec![0.5,0.0],30);
        assert_eq!(calibrate_selective_evidence_v1(&cfg,&cyclic,&rows),Err(CalibrationRejectV1::GroupHierarchyCycle));
        let stable=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(40,"root",0.9,0.0);let mut cfg=config(&stable,&rows,vec![0.5,0.0],30);
        cfg.calibration_data_sha256=digest("wrong");
        assert_eq!(calibrate_selective_evidence_v1(&cfg,&stable,&rows),Err(CalibrationRejectV1::CalibrationDataDigestMismatch));
    }
    #[test]
    fn low_score_triggers_retrieve_more(){
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let mut rows=examples(80,"root",0.9,0.0);
        for row in rows.iter_mut().take(20) { row.sufficiency_score=0.2; row.decision_loss=1.0; }
        let cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();let cert=&result.certificates[0];
        assert_eq!(cert.threshold,Some(0.5));
        let assessment=assess_decision_v1(&DecisionInputV1{
            group_id:"root".into(),sufficiency_score:0.1,
            sufficiency_score_definition_sha256:cfg.sufficiency_score_definition_sha256.clone(),
            group_hierarchy_sha256:cfg.group_hierarchy_sha256.clone(),
            cluster_unit_definition_sha256:cfg.cluster_unit_definition_sha256.clone(),
            cluster_sampling_design_sha256:cfg.cluster_sampling_design_sha256.clone(),
            cluster_workload_policy_sha256:cfg.cluster_workload_policy_sha256.clone(),
            hard_gates_pass:true,distribution_shift_detected:false,
            evidence_acquisition_available:true,retrieval_available:true,escalate_on_hard_gate:false,
            decision_need_sha256:digest("need"),evidence_capsule_sha256:digest("capsule"),policy_sha256:digest("policy")
        },cert).unwrap();
        assert_eq!(assessment.action,DecisionActionV1::RetrieveMore);assert!(assessment.verify_integrity());
    }
    #[test]
    fn acceptance_requires_matching_group_certificate_and_hard_gates(){
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(100,"root",0.9,0.0);let cfg=config(&groups,&rows,vec![0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();let cert=&result.certificates[0];
        let input=DecisionInputV1{group_id:"root".into(),sufficiency_score:0.9,
        sufficiency_score_definition_sha256:cfg.sufficiency_score_definition_sha256.clone(),
        group_hierarchy_sha256:cfg.group_hierarchy_sha256.clone(),
            cluster_unit_definition_sha256:cfg.cluster_unit_definition_sha256.clone(),
            cluster_sampling_design_sha256:cfg.cluster_sampling_design_sha256.clone(),
            cluster_workload_policy_sha256:cfg.cluster_workload_policy_sha256.clone(),
            hard_gates_pass:true,
            distribution_shift_detected:false,evidence_acquisition_available:false,retrieval_available:false,
            escalate_on_hard_gate:false,decision_need_sha256:digest("need"),
            evidence_capsule_sha256:digest("capsule"),policy_sha256:digest("policy")};
        let assessment=assess_decision_v1(&input,cert).unwrap();
        assert_eq!(assessment.action,DecisionActionV1::AcceptAndAct);
        assert_eq!(assessment.status,DecisionAssessmentStatusV1::AcceptedUnderCertificate);
        let mut wrong=input;wrong.group_id="different".into();
        assert_ne!(assess_decision_v1(&wrong,cert).unwrap().action,DecisionActionV1::AcceptAndAct);
    }
    #[test]
    fn tampered_certificate_cannot_authorize_acceptance(){
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(100,"root",0.9,0.0);let cfg=config(&groups,&rows,vec![0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        let mut cert=result.certificates[0].clone();cert.threshold=Some(0.99);
        assert!(!cert.verify_integrity());assert!(!cert.is_usable());
    }

    #[test]
    fn changed_decision_policy_cannot_reuse_a_valid_certificate() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(100,"root",0.9,0.0);
        let cfg=config(&groups,&rows,vec![0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        let cert=&result.certificates[0];
        let input=DecisionInputV1{
            group_id:"root".into(),sufficiency_score:0.9,
            sufficiency_score_definition_sha256:cfg.sufficiency_score_definition_sha256.clone(),
            group_hierarchy_sha256:cfg.group_hierarchy_sha256.clone(),
            cluster_unit_definition_sha256:cfg.cluster_unit_definition_sha256.clone(),
            cluster_sampling_design_sha256:cfg.cluster_sampling_design_sha256.clone(),
            cluster_workload_policy_sha256:cfg.cluster_workload_policy_sha256.clone(),
            hard_gates_pass:true,
            distribution_shift_detected:false,evidence_acquisition_available:false,retrieval_available:false,
            escalate_on_hard_gate:false,decision_need_sha256:digest("need"),
            evidence_capsule_sha256:digest("capsule"),policy_sha256:digest("different-policy")
        };
        let assessment=assess_decision_v1(&input,cert).unwrap();
        assert_eq!(assessment.action,DecisionActionV1::Abstain);
        assert_eq!(assessment.status,DecisionAssessmentStatusV1::AbstainedNotEstimable);
        assert!(assessment.verify_integrity());
    }


    #[test]
    fn changed_score_definition_or_group_hierarchy_cannot_reuse_certificate() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(100,"root",0.9,0.0);
        let cfg=config(&groups,&rows,vec![0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        let cert=&result.certificates[0];
        let mut input=DecisionInputV1{
            group_id:"root".into(),sufficiency_score:0.9,
            sufficiency_score_definition_sha256:cfg.sufficiency_score_definition_sha256.clone(),
            group_hierarchy_sha256:cfg.group_hierarchy_sha256.clone(),
            cluster_unit_definition_sha256:cfg.cluster_unit_definition_sha256.clone(),
            cluster_sampling_design_sha256:cfg.cluster_sampling_design_sha256.clone(),
            cluster_workload_policy_sha256:cfg.cluster_workload_policy_sha256.clone(),
            hard_gates_pass:true,distribution_shift_detected:false,
            evidence_acquisition_available:false,retrieval_available:false,escalate_on_hard_gate:false,
            decision_need_sha256:digest("need"),evidence_capsule_sha256:digest("capsule"),
            policy_sha256:digest("policy")
        };
        input.sufficiency_score_definition_sha256=digest("different-score-definition");
        let score_mismatch=assess_decision_v1(&input,cert).unwrap();
        assert_eq!(score_mismatch.action,DecisionActionV1::Abstain);
        assert_eq!(score_mismatch.status,DecisionAssessmentStatusV1::AbstainedNotEstimable);
        assert_eq!(score_mismatch.risk_upper_bound,None);

        input.sufficiency_score_definition_sha256=cfg.sufficiency_score_definition_sha256.clone();
        input.group_hierarchy_sha256=digest("different-group-hierarchy");
        let hierarchy_mismatch=assess_decision_v1(&input,cert).unwrap();
        assert_eq!(hierarchy_mismatch.action,DecisionActionV1::Abstain);
        assert_eq!(hierarchy_mismatch.status,DecisionAssessmentStatusV1::AbstainedNotEstimable);
        assert_eq!(hierarchy_mismatch.risk_upper_bound,None);
    }


    #[test]
    fn fixed_sequence_uses_groupwise_delta_and_stops_at_first_failed_prefix() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let mut rows=examples(80,"root",0.9,0.0);
        for row in rows.iter_mut().take(20) { row.sufficiency_score=0.2; row.decision_loss=1.0; }
        let mut cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        cfg.risk_target_alpha=0.05;
        let fixed=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        let fixed_cert=&fixed.certificates[0];
        assert_eq!(fixed_cert.status,RiskStatusV1::Certified);
        assert_eq!(fixed_cert.threshold,Some(0.5));
        assert_eq!(fixed_cert.multiplicity_adjusted_delta,Some(0.05));

        let mut simultaneous=cfg.clone();
        simultaneous.threshold_selection_method=ThresholdSelectionMethodV1::BonferroniSimultaneous;
        simultaneous.thresholds.reverse();
        simultaneous.threshold_grid_sha256=digest_serialized(&simultaneous.thresholds).unwrap();
        // The changed threshold grid needs a new precommitment before calibration opens.
        simultaneous.threshold_grid_precommitment_sha256=
            compute_threshold_grid_precommitment_sha256_v1(&simultaneous).unwrap();
        let bonf=calibrate_selective_evidence_v1(&simultaneous,&groups,&rows).unwrap();
        let bonf_cert=&bonf.certificates[0];
        assert_ne!(bonf_cert.status,RiskStatusV1::Certified);
        assert_eq!(bonf_cert.multiplicity_adjusted_delta,Some(0.05/3.0));
    }

    #[test]
    fn cluster_and_decision_coverage_are_reported_separately() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let mut rows=examples(40,"root",0.9,0.0);
        let mut repeated=rows[0].clone();
        repeated.outcome_receipt_sha256=digest("outcome-repeated");
        repeated.sufficiency_score=0.1;
        repeated.decision_loss=1.0;
        rows.push(repeated);
        let cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        let cert=&result.certificates[0];
        assert_eq!(cert.accepted_cluster_count,40);
        assert_eq!(cert.coverage,Some(1.0));
        assert!(cert.decision_coverage.is_some_and(|value|value<1.0));
    }

    #[test]
    fn fixed_sequence_rejects_non_descending_sufficiency_grid() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(40,"root",0.9,0.0);
        let cfg=config(&groups,&rows,vec![0.0,0.5],30);
        assert_eq!(calibrate_selective_evidence_v1(&cfg,&groups,&rows),Err(CalibrationRejectV1::InvalidThresholdGrid));
    }


    #[test]
    fn calibration_rejects_grid_committed_after_calibration_started() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(40,"root",0.9,0.0);
        let mut cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        cfg.calibration_opened_at_ms=10;
        cfg.threshold_grid_committed_at_ms=20;
        cfg.threshold_grid_precommitment_sha256=compute_threshold_grid_precommitment_sha256_v1(&cfg).unwrap();
        assert_eq!(calibrate_selective_evidence_v1(&cfg,&groups,&rows),Err(CalibrationRejectV1::InvalidConfig));
    }

    #[test]
    fn calibration_rejects_threshold_grid_source_from_the_calibration_split() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(40,"root",0.9,0.0);
        let mut cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        cfg.threshold_grid_source_sha256=cfg.calibration_split_sha256.clone();
        cfg.threshold_grid_precommitment_sha256=compute_threshold_grid_precommitment_sha256_v1(&cfg).unwrap();
        assert_eq!(calibrate_selective_evidence_v1(&cfg,&groups,&rows),Err(CalibrationRejectV1::InvalidConfig));
    }

    #[test]
    fn certificate_cannot_transfer_to_a_different_cluster_unit_or_workload_policy() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(100,"root",0.9,0.0);
        let cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        let result=calibrate_selective_evidence_v1(&cfg,&groups,&rows).unwrap();
        let cert=&result.certificates[0];
        let mut input=DecisionInputV1{
            group_id:"root".into(),sufficiency_score:0.9,
            sufficiency_score_definition_sha256:cfg.sufficiency_score_definition_sha256.clone(),
            group_hierarchy_sha256:cfg.group_hierarchy_sha256.clone(),
            cluster_unit_definition_sha256:cfg.cluster_unit_definition_sha256.clone(),
            cluster_sampling_design_sha256:cfg.cluster_sampling_design_sha256.clone(),
            cluster_workload_policy_sha256:cfg.cluster_workload_policy_sha256.clone(),
            hard_gates_pass:true,distribution_shift_detected:false,
            evidence_acquisition_available:false,retrieval_available:false,escalate_on_hard_gate:false,
            decision_need_sha256:digest("need"),evidence_capsule_sha256:digest("capsule"),
            policy_sha256:cfg.decision_policy_sha256.clone()
        };
        input.cluster_workload_policy_sha256=digest("different workload horizon");
        let assessment=assess_decision_v1(&input,cert).unwrap();
        assert_eq!(assessment.action,DecisionActionV1::Abstain);
        assert_eq!(assessment.status,DecisionAssessmentStatusV1::AbstainedNotEstimable);
        assert_eq!(assessment.risk_upper_bound,None);
        assert!(assessment.verify_integrity());
    }

    #[test]
    fn precommitment_and_cluster_design_receipts_are_recomputed_and_tamper_evident() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(40,"root",0.9,0.0);
        let cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        assert_eq!(compute_threshold_grid_precommitment_sha256_v1(&cfg).unwrap(),cfg.threshold_grid_precommitment_sha256);
        assert_eq!(compute_cluster_design_receipt_sha256_v1(&cfg).unwrap(),cfg.cluster_design_receipt_sha256);
        let mut altered=cfg.clone();
        altered.cluster_assignment_manifest_sha256=digest("different cluster assignment");
        assert_ne!(compute_cluster_design_receipt_sha256_v1(&altered).unwrap(),cfg.cluster_design_receipt_sha256);
    }


    #[test]
    fn calibration_recomputes_row_to_cluster_assignment_manifest() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let rows=examples(40,"root",0.9,0.0);
        let cfg=config(&groups,&rows,vec![0.8,0.5,0.0],30);
        let mut altered=rows.clone();
        altered[0].independent_cluster_id="profile-swapped".into();
        let mut altered_cfg=cfg;
        altered_cfg.calibration_data_sha256=digest_serialized(&altered).unwrap();
        assert_eq!(
            calibrate_selective_evidence_v1(&altered_cfg,&groups,&altered),
            Err(CalibrationRejectV1::ClusterAssignmentManifestMismatch)
        );
    }

    #[test]
    fn calibration_enforces_the_declared_per_cluster_workload_limit() {
        let groups=vec![GroupSpecV1{group_id:"root".into(),parent_group_id:None,allow_parent_fallback:false,distribution_shift_detected:false}];
        let mut rows=examples(2,"root",0.9,0.0);
        rows[1].independent_cluster_id=rows[0].independent_cluster_id.clone();
        let mut cfg=config(&groups,&rows,vec![0.8,0.5,0.0],2);
        cfg.maximum_observations_per_cluster=1;
        cfg.cluster_workload_policy_sha256=digest("max_episodes_per_profile_1");
        cfg.cluster_design_receipt_sha256=compute_cluster_design_receipt_sha256_v1(&cfg).unwrap();
        assert_eq!(
            calibrate_selective_evidence_v1(&cfg,&groups,&rows),
            Err(CalibrationRejectV1::ClusterWorkloadLimitExceeded)
        );
    }


    #[test]
    fn duplicate_outcome_receipts_cannot_make_cluster_assignment_ambiguous() {
        let mut rows=examples(2,"root",0.9,0.0);
        rows[1].outcome_receipt_sha256=rows[0].outcome_receipt_sha256.clone();
        assert_eq!(
            compute_cluster_assignment_manifest_sha256_v1(&rows),
            Err(CalibrationRejectV1::DuplicateOutcomeReceipt)
        );
    }

}
