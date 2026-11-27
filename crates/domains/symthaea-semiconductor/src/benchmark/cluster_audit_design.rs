// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finite-population design-based risk estimation under common-cause cluster
//! sampling. The randomized units are complete, predeclared cohorts; every
//! population unit in a selected cohort is audited by a distinct reference
//! method. No IID-wafer outcome model or online-controller propensity is used.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const CLUSTER_AUDIT_FRAME_SCHEMA_V1: &str =
    "luminous.semi-bench.cluster-audit-frame.v1";
pub const CLUSTER_AUDIT_DESIGN_SCHEMA_V1: &str =
    "luminous.semi-bench.cluster-audit-design.v1";
pub const CLUSTER_AUDIT_RECORD_SCHEMA_V1: &str =
    "luminous.semi-bench.cluster-audit-record.v1";
pub const CLUSTER_AUDIT_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.cluster-audit-input.v1";
pub const CLUSTER_AUDIT_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.cluster-audit-report.v1";
pub const CLUSTER_AUDIT_RANDOMIZATION_ALGORITHM_V1: &str =
    "sha256-counter-fisher-yates-rejection.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterAuditStatusV1 {
    Complete,
    ExactCensus,
    VarianceUnavailable,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceAuditQualificationV1 {
    Qualified,
    Unknown,
    Blocked,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditFrameUnitV1 {
    pub unit_id: String,
    pub population_unit_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditFrameClusterV1 {
    pub cluster_id: String,
    pub stratum_id: String,
    /// A predeclared common-cause cohort membership receipt; it is not
    /// inferred from observed losses.
    pub common_cause_membership_receipt_sha256: String,
    pub units: Vec<ClusterAuditFrameUnitV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditPopulationFrameV1 {
    pub schema: String,
    pub population_manifest_sha256: String,
    /// Operational method from which the VM predictions were evaluated.
    pub operational_measurement_method_sha256: String,
    pub frame_frozen_at_unix_seconds: i64,
    pub clusters: Vec<ClusterAuditFrameClusterV1>,
    pub frame_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditStratumPlanV1 {
    pub stratum_id: String,
    /// Number of whole clusters sampled by SRSWOR in this stratum.
    pub audited_cluster_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditDesignV1 {
    pub schema: String,
    pub design_id: String,
    pub randomization_algorithm: String,
    pub population_manifest_sha256: String,
    pub frame_sha256: String,
    /// 32-byte seed, hex encoded, revealed only after audit closeout.
    pub randomization_seed_reveal_hex: String,
    pub seed_commitment_sha256: String,
    pub seed_committed_at_unix_seconds: i64,
    pub randomization_performed_at_unix_seconds: i64,
    pub audit_closed_at_unix_seconds: i64,
    pub seed_revealed_at_unix_seconds: i64,
    /// Hard physical reference-metrology capacity, in audited units.
    pub maximum_audit_unit_capacity: usize,
    pub confidence_delta: f64,
    pub reference_method_sha256: String,
    /// Must be distinct from the operational VM measurement method.
    pub independent_reference_method_declared: bool,
    /// Must be true: audit cluster selection is independent of operational
    /// selection, K1 qualification and controller feedback tracking.
    pub independent_of_operational_process_declared: bool,
    /// Commitment, seed-generation and randomization chronology evidence.
    pub randomization_provenance_sha256: String,
    pub stratum_plans: Vec<ClusterAuditStratumPlanV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditUnitOutcomeV1 {
    pub unit_id: String,
    /// Bounded interval loss/miscoverage indicator or other predeclared loss.
    pub loss: Option<f64>,
    pub reference_qualification: ReferenceAuditQualificationV1,
    pub reference_label_id: String,
    pub reference_label_revision: u64,
    pub reference_label_payload_sha256: String,
    pub reference_measurement_event_id: String,
    pub reference_measurement_receipt_sha256: String,
    pub reference_qualification_report_sha256: String,
    pub reference_qualification_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditRecordV1 {
    pub schema: String,
    pub cluster_id: String,
    pub stratum_id: String,
    pub audited_at_unix_seconds: i64,
    pub reference_method_sha256: String,
    /// Must contain every unit in the selected cohort: v1 audits whole cohorts.
    pub unit_outcomes: Vec<ClusterAuditUnitOutcomeV1>,
    pub cluster_audit_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditInputV1 {
    pub schema: String,
    pub run_id: String,
    pub frame: ClusterAuditPopulationFrameV1,
    pub design: ClusterAuditDesignV1,
    pub audited_clusters: Vec<ClusterAuditRecordV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditStratumReportV1 {
    pub stratum_id: String,
    pub population_cluster_count: usize,
    pub audited_cluster_count: usize,
    pub cluster_size_units: usize,
    pub population_unit_count: usize,
    pub sampled_stratum_mean_loss: f64,
    pub population_weight: f64,
    pub first_order_cluster_inclusion_probability: f64,
    pub second_order_cluster_inclusion_probability: Option<f64>,
    pub sample_variance_of_cluster_means: Option<f64>,
    pub design_variance_of_stratum_mean: Option<f64>,
    pub hoeffding_radius: f64,
    pub audited_cluster_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterAuditReportV1 {
    pub schema: String,
    pub run_id: String,
    pub input_sha256: String,
    pub frame_sha256: String,
    pub design_sha256: String,
    pub status: ClusterAuditStatusV1,
    pub reason_codes: Vec<String>,
    pub population_cluster_count: usize,
    pub population_unit_count: usize,
    pub audited_cluster_count: usize,
    pub audited_unit_count: usize,
    pub maximum_audit_unit_capacity: usize,
    pub audit_capacity_remaining: usize,
    pub population_risk_estimate: Option<f64>,
    pub design_variance_estimate: Option<f64>,
    pub design_standard_error: Option<f64>,
    /// Simultaneous finite-population Hoeffding radius, valid without
    /// an IID-within-cluster loss assumption.
    pub hoeffding_radius: Option<f64>,
    pub hoeffding_lower: Option<f64>,
    pub hoeffding_upper: Option<f64>,
    pub exact_census: bool,
    pub stratum_reports: Vec<ClusterAuditStratumReportV1>,
    pub randomization_replay_verified: bool,
    pub audit_selection_independent_of_operational_process_declared: bool,
    pub reference_method_sha256: String,
    pub source_authenticity_proven: bool,
    pub physical_cluster_independence_proven: bool,
    pub population_claim_authorized_by_qualified_operational_subset_alone: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ClusterAuditErrorV1(pub String);

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, ClusterAuditErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ClusterAuditErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), ClusterAuditErrorV1> {
    if value.trim().is_empty() {
        return Err(ClusterAuditErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn hex_seed(value: &str) -> Result<[u8; 32], ClusterAuditErrorV1> {
    if !valid_sha256(value) {
        return Err(ClusterAuditErrorV1(
            "randomization seed reveal must be 32 bytes of lowercase hex".into(),
        ));
    }
    let mut seed = [0_u8; 32];
    for (index, slot) in seed.iter_mut().enumerate() {
        let start = index * 2;
        *slot = u8::from_str_radix(&value[start..start + 2], 16)
            .map_err(|_| ClusterAuditErrorV1("randomization seed hex is invalid".into()))?;
    }
    Ok(seed)
}

fn canonical_frame_digest(
    frame: &ClusterAuditPopulationFrameV1,
) -> Result<String, ClusterAuditErrorV1> {
    let mut canonical = frame.clone();
    canonical.frame_sha256.clear();
    canonical.clusters.sort_by(|a, b| {
        (&a.stratum_id, &a.cluster_id).cmp(&(&b.stratum_id, &b.cluster_id))
    });
    for cluster in &mut canonical.clusters {
        cluster.units.sort_by(|a, b| a.unit_id.cmp(&b.unit_id));
    }
    json_sha256(&canonical)
}

fn frame_digest(frame: &ClusterAuditPopulationFrameV1) -> Result<String, ClusterAuditErrorV1> {
    canonical_frame_digest(frame)
}

fn canonical_cluster_record_digest(
    record: &ClusterAuditRecordV1,
) -> Result<String, ClusterAuditErrorV1> {
    let mut canonical = record.clone();
    canonical.cluster_audit_receipt_sha256.clear();
    canonical.unit_outcomes.sort_by(|a, b| a.unit_id.cmp(&b.unit_id));
    json_sha256(&canonical)
}

pub fn seal_cluster_audit_frame_v1(
    frame: &ClusterAuditPopulationFrameV1,
) -> Result<String, ClusterAuditErrorV1> {
    frame_digest(frame)
}

pub fn seal_cluster_audit_record_v1(
    record: &ClusterAuditRecordV1,
) -> Result<String, ClusterAuditErrorV1> {
    canonical_cluster_record_digest(record)
}

fn canonical_design_digest(
    design: &ClusterAuditDesignV1,
) -> Result<String, ClusterAuditErrorV1> {
    let mut canonical = design.clone();
    canonical.stratum_plans.sort_by(|a, b| a.stratum_id.cmp(&b.stratum_id));
    json_sha256(&canonical)
}

fn input_digest(input: &ClusterAuditInputV1) -> Result<String, ClusterAuditErrorV1> {
    let mut canonical = input.clone();
    canonical.frame.clusters.sort_by(|a, b| {
        (&a.stratum_id, &a.cluster_id).cmp(&(&b.stratum_id, &b.cluster_id))
    });
    for cluster in &mut canonical.frame.clusters {
        cluster.units.sort_by(|a, b| a.unit_id.cmp(&b.unit_id));
    }
    canonical.design.stratum_plans.sort_by(|a, b| a.stratum_id.cmp(&b.stratum_id));
    canonical.audited_clusters.sort_by(|a, b| a.cluster_id.cmp(&b.cluster_id));
    for record in &mut canonical.audited_clusters {
        record.unit_outcomes.sort_by(|a, b| a.unit_id.cmp(&b.unit_id));
    }
    json_sha256(&canonical)
}

fn report_digest(report: &ClusterAuditReportV1) -> Result<String, ClusterAuditErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn seed_commitment(
    seed: &[u8; 32],
    design_id: &str,
    population_manifest_sha256: &str,
    frame_sha256: &str,
    randomization_algorithm: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"LUMINOUS-SEMI-CLUSTER-AUDIT-COMMIT-V2\0");
    hasher.update(seed);
    hasher.update([0]);
    hasher.update(design_id.as_bytes());
    hasher.update([0]);
    hasher.update(population_manifest_sha256.as_bytes());
    hasher.update([0]);
    hasher.update(frame_sha256.as_bytes());
    hasher.update([0]);
    hasher.update(randomization_algorithm.as_bytes());
    format!("{:x}", hasher.finalize())
}

struct HashCounterRandomV1 {
    seed: [u8; 32],
    design_id: String,
    stratum_id: String,
    counter: u64,
}

impl HashCounterRandomV1 {
    fn new(seed: [u8; 32], design_id: &str, stratum_id: &str) -> Self {
        Self {
            seed,
            design_id: design_id.into(),
            stratum_id: stratum_id.into(),
            counter: 0,
        }
    }

    fn next_u64(&mut self) -> Result<u64, ClusterAuditErrorV1> {
        let mut hasher = Sha256::new();
        hasher.update(b"LUMINOUS-SEMI-CLUSTER-AUDIT-RNG-V1\0");
        hasher.update(self.seed);
        hasher.update(self.design_id.as_bytes());
        hasher.update([0]);
        hasher.update(self.stratum_id.as_bytes());
        hasher.update(self.counter.to_be_bytes());
        self.counter = self
            .counter
            .checked_add(1)
            .ok_or_else(|| ClusterAuditErrorV1("random counter overflow".into()))?;
        let digest = hasher.finalize();
        let mut first = [0_u8; 8];
        first.copy_from_slice(&digest[..8]);
        Ok(u64::from_be_bytes(first))
    }

    /// Uniform integer in [0,bound), using rejection sampling to avoid modulo bias.
    fn below(&mut self, bound: u64) -> Result<u64, ClusterAuditErrorV1> {
        if bound == 0 {
            return Err(ClusterAuditErrorV1("random bound must be positive".into()));
        }
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let value = self.next_u64()?;
            if value < zone {
                return Ok(value % bound);
            }
        }
    }
}

fn expected_sample(
    cluster_ids: &[String],
    sample_count: usize,
    seed: [u8; 32],
    design_id: &str,
    stratum_id: &str,
) -> Result<BTreeSet<String>, ClusterAuditErrorV1> {
    if sample_count > cluster_ids.len() {
        return Err(ClusterAuditErrorV1(format!(
            "stratum {stratum_id} requests more clusters than its frame contains"
        )));
    }
    let mut shuffled = cluster_ids.to_vec();
    shuffled.sort();
    let mut rng = HashCounterRandomV1::new(seed, design_id, stratum_id);
    for index in 0..sample_count {
        let remaining = (shuffled.len() - index) as u64;
        let offset = rng.below(remaining)? as usize;
        shuffled.swap(index, index + offset);
    }
    Ok(shuffled.into_iter().take(sample_count).collect())
}

fn validate_frame(
    frame: &ClusterAuditPopulationFrameV1,
) -> Result<
    (
        BTreeMap<String, ClusterAuditFrameClusterV1>,
        BTreeMap<String, Vec<String>>,
        BTreeMap<String, usize>,
        usize,
    ),
    ClusterAuditErrorV1,
> {
    if frame.schema != CLUSTER_AUDIT_FRAME_SCHEMA_V1 {
        return Err(ClusterAuditErrorV1("cluster-audit frame schema is unsupported".into()));
    }
    if !valid_sha256(&frame.population_manifest_sha256)
        || !valid_sha256(&frame.operational_measurement_method_sha256)
        || !valid_sha256(&frame.frame_sha256)
        || frame_digest(frame)? != frame.frame_sha256
    {
        return Err(ClusterAuditErrorV1(
            "cluster-audit frame digest or method identity is invalid".into(),
        ));
    }
    let mut clusters = BTreeMap::new();
    let mut strata: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut cluster_sizes: BTreeMap<String, usize> = BTreeMap::new();
    let mut unit_ids = BTreeSet::new();
    let mut total_units = 0_usize;

    for cluster in &frame.clusters {
        require_text(&cluster.cluster_id, "frame.cluster_id")?;
        require_text(&cluster.stratum_id, "frame.stratum_id")?;
        if !valid_sha256(&cluster.common_cause_membership_receipt_sha256) {
            return Err(ClusterAuditErrorV1(format!(
                "cluster {} has malformed common-cause membership receipt",
                cluster.cluster_id
            )));
        }
        if cluster.units.is_empty() {
            return Err(ClusterAuditErrorV1(format!(
                "cluster {} has an empty unit frame",
                cluster.cluster_id
            )));
        }
        let mut local_units = BTreeSet::new();
        for unit in &cluster.units {
            require_text(&unit.unit_id, "frame.unit_id")?;
            if !valid_sha256(&unit.population_unit_receipt_sha256)
                || !local_units.insert(unit.unit_id.as_str())
                || !unit_ids.insert(unit.unit_id.as_str())
            {
                return Err(ClusterAuditErrorV1(format!(
                    "cluster {} has a duplicate unit or malformed unit receipt",
                    cluster.cluster_id
                )));
            }
        }
        if clusters.insert(cluster.cluster_id.clone(), cluster.clone()).is_some() {
            return Err(ClusterAuditErrorV1("duplicate cluster ID in population frame".into()));
        }
        strata
            .entry(cluster.stratum_id.clone())
            .or_default()
            .push(cluster.cluster_id.clone());
        let size = cluster.units.len();
        if cluster_sizes
            .insert(cluster.stratum_id.clone(), size)
            .is_some_and(|stored| stored != size)
        {
            return Err(ClusterAuditErrorV1(format!(
                "stratum {} contains unequal cohort sizes; stratify the frame by cohort size",
                cluster.stratum_id
            )));
        }
        total_units = total_units
            .checked_add(size)
            .ok_or_else(|| ClusterAuditErrorV1("population unit count overflow".into()))?;
    }

    for ids in strata.values_mut() {
        ids.sort();
    }
    if clusters.is_empty() {
        return Err(ClusterAuditErrorV1("population frame has no clusters".into()));
    }
    Ok((clusters, strata, cluster_sizes, total_units))
}

fn validate_design(
    design: &ClusterAuditDesignV1,
    frame: &ClusterAuditPopulationFrameV1,
) -> Result<[u8; 32], ClusterAuditErrorV1> {
    if design.schema != CLUSTER_AUDIT_DESIGN_SCHEMA_V1
        || design.randomization_algorithm != CLUSTER_AUDIT_RANDOMIZATION_ALGORITHM_V1
    {
        return Err(ClusterAuditErrorV1("cluster-audit design schema/algorithm is unsupported".into()));
    }
    require_text(&design.design_id, "design.design_id")?;
    for (field, value) in [
        ("design.population_manifest_sha256", design.population_manifest_sha256.as_str()),
        ("design.frame_sha256", design.frame_sha256.as_str()),
        ("design.seed_commitment_sha256", design.seed_commitment_sha256.as_str()),
        ("design.reference_method_sha256", design.reference_method_sha256.as_str()),
        ("design.randomization_provenance_sha256", design.randomization_provenance_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            return Err(ClusterAuditErrorV1(format!("{field}: malformed SHA-256")));
        }
    }
    if design.population_manifest_sha256 != frame.population_manifest_sha256
        || design.frame_sha256 != frame.frame_sha256
        || design.reference_method_sha256 == frame.operational_measurement_method_sha256
    {
        return Err(ClusterAuditErrorV1(
            "cluster-audit design does not bind the frozen population frame or independent reference method"
                .into(),
        ));
    }
    if !design.independent_reference_method_declared
        || !design.independent_of_operational_process_declared
        || design.maximum_audit_unit_capacity == 0
        || !design.confidence_delta.is_finite()
        || !(0.0..1.0).contains(&design.confidence_delta)
    {
        return Err(ClusterAuditErrorV1(
            "cluster-audit independence, capacity or confidence policy is invalid".into(),
        ));
    }
    if !(frame.frame_frozen_at_unix_seconds <= design.seed_committed_at_unix_seconds
        && design.seed_committed_at_unix_seconds < design.randomization_performed_at_unix_seconds
        && design.randomization_performed_at_unix_seconds <= design.audit_closed_at_unix_seconds
        && design.audit_closed_at_unix_seconds <= design.seed_revealed_at_unix_seconds)
    {
        return Err(ClusterAuditErrorV1(
            "cluster-audit seed commitment/randomization/reveal chronology is invalid".into(),
        ));
    }
    let seed = hex_seed(&design.randomization_seed_reveal_hex)?;
    let expected_commitment = seed_commitment(
        &seed,
        &design.design_id,
        &design.population_manifest_sha256,
        &design.frame_sha256,
        &design.randomization_algorithm,
    );
    if expected_commitment != design.seed_commitment_sha256 {
        return Err(ClusterAuditErrorV1(
            "revealed random seed does not match its precommitted digest".into(),
        ));
    }
    Ok(seed)
}

fn validate_cluster_record(
    record: &ClusterAuditRecordV1,
    frame_cluster: &ClusterAuditFrameClusterV1,
    design: &ClusterAuditDesignV1,
) -> Result<Option<Vec<f64>>, ClusterAuditErrorV1> {
    if record.schema != CLUSTER_AUDIT_RECORD_SCHEMA_V1 {
        return Err(ClusterAuditErrorV1("cluster audit record schema is unsupported".into()));
    }
    if record.cluster_id != frame_cluster.cluster_id
        || record.stratum_id != frame_cluster.stratum_id
        || record.reference_method_sha256 != design.reference_method_sha256
        || record.audited_at_unix_seconds < design.randomization_performed_at_unix_seconds
        || record.audited_at_unix_seconds > design.audit_closed_at_unix_seconds
    {
        return Err(ClusterAuditErrorV1(format!(
            "audit record {} does not match the selected frozen cluster/design",
            record.cluster_id
        )));
    }
    if !valid_sha256(&record.cluster_audit_receipt_sha256)
        || canonical_cluster_record_digest(record)? != record.cluster_audit_receipt_sha256
    {
        return Err(ClusterAuditErrorV1(format!(
            "cluster audit record {} digest mismatch",
            record.cluster_id
        )));
    }
    if record.unit_outcomes.len() != frame_cluster.units.len() {
        return Err(ClusterAuditErrorV1(format!(
            "selected cluster {} is not fully audited",
            record.cluster_id
        )));
    }
    let expected_units: BTreeSet<&str> =
        frame_cluster.units.iter().map(|unit| unit.unit_id.as_str()).collect();
    let mut actual_units = BTreeSet::new();
    let mut losses = Vec::with_capacity(record.unit_outcomes.len());
    let mut all_qualified = true;
    for outcome in &record.unit_outcomes {
        require_text(&outcome.unit_id, "audit.unit_id")?;
        if !expected_units.contains(outcome.unit_id.as_str())
            || !actual_units.insert(outcome.unit_id.as_str())
            || outcome.reference_label_revision == 0
            || outcome.reference_measurement_event_id.trim().is_empty()
            || outcome.reference_label_id.trim().is_empty()
            || !valid_sha256(&outcome.reference_label_payload_sha256)
            || !valid_sha256(&outcome.reference_measurement_receipt_sha256)
            || !valid_sha256(&outcome.reference_qualification_report_sha256)
            || !valid_sha256(&outcome.reference_qualification_receipt_sha256)
        {
            return Err(ClusterAuditErrorV1(format!(
                "cluster {} has a duplicate/foreign unit or malformed reference receipt",
                record.cluster_id
            )));
        }
        match (outcome.reference_qualification, outcome.loss) {
            (ReferenceAuditQualificationV1::Qualified, Some(loss))
                if loss.is_finite() && (0.0..=1.0).contains(&loss) =>
            {
                losses.push(loss);
            }
            (ReferenceAuditQualificationV1::Qualified, _) => {
                return Err(ClusterAuditErrorV1(format!(
                    "cluster {} has a qualified reference outcome without a bounded finite loss",
                    record.cluster_id
                )));
            }
            (ReferenceAuditQualificationV1::Unknown, None)
            | (ReferenceAuditQualificationV1::Blocked, None) => {
                all_qualified = false;
            }
            _ => {
                return Err(ClusterAuditErrorV1(format!(
                    "cluster {} has inconsistent reference qualification and loss",
                    record.cluster_id
                )));
            }
        }
    }
    if actual_units != expected_units {
        return Err(ClusterAuditErrorV1(format!(
            "cluster {} is missing one or more population units",
            record.cluster_id
        )));
    }
    if !all_qualified {
        return Ok(None);
    }
    Ok(Some(losses))
}

fn validate_plans(
    plans: &[ClusterAuditStratumPlanV1],
    strata: &BTreeMap<String, Vec<String>>,
) -> Result<(), ClusterAuditErrorV1> {
    let mut plan_ids = BTreeSet::new();
    for plan in plans {
        require_text(&plan.stratum_id, "design.stratum_id")?;
        let Some(frame_ids) = strata.get(&plan.stratum_id) else {
            return Err(ClusterAuditErrorV1(format!(
                "design references unknown stratum {}",
                plan.stratum_id
            )));
        };
        if !plan_ids.insert(plan.stratum_id.as_str())
            || plan.audited_cluster_count == 0
            || plan.audited_cluster_count > frame_ids.len()
        {
            return Err(ClusterAuditErrorV1(format!(
                "stratum {} has duplicate or invalid cluster sample count",
                plan.stratum_id
            )));
        }
    }
    if plan_ids.len() != strata.len()
        || strata.keys().any(|id| !plan_ids.contains(id.as_str()))
    {
        return Err(ClusterAuditErrorV1(
            "cluster-audit plan must cover every population stratum exactly once".into(),
        ));
    }
    Ok(())
}

fn empty_report(
    input: &ClusterAuditInputV1,
    status: ClusterAuditStatusV1,
    reason: &str,
    population_cluster_count: usize,
    population_unit_count: usize,
    audited_cluster_count: usize,
    audited_unit_count: usize,
    randomization_replay_verified: bool,
) -> Result<ClusterAuditReportV1, ClusterAuditErrorV1> {
    let mut report = ClusterAuditReportV1 {
        schema: CLUSTER_AUDIT_REPORT_SCHEMA_V1.into(),
        run_id: input.run_id.clone(),
        input_sha256: input_digest(input)?,
        frame_sha256: input.frame.frame_sha256.clone(),
        design_sha256: canonical_design_digest(&input.design)?,
        status,
        reason_codes: vec![reason.into()],
        population_cluster_count,
        population_unit_count,
        audited_cluster_count,
        audited_unit_count,
        maximum_audit_unit_capacity: input.design.maximum_audit_unit_capacity,
        audit_capacity_remaining: input
            .design
            .maximum_audit_unit_capacity
            .saturating_sub(audited_unit_count),
        population_risk_estimate: None,
        design_variance_estimate: None,
        design_standard_error: None,
        hoeffding_radius: None,
        hoeffding_lower: None,
        hoeffding_upper: None,
        exact_census: false,
        stratum_reports: Vec::new(),
        randomization_replay_verified,
        audit_selection_independent_of_operational_process_declared: input
            .design
            .independent_of_operational_process_declared,
        reference_method_sha256: input.design.reference_method_sha256.clone(),
        source_authenticity_proven: false,
        physical_cluster_independence_proven: false,
        population_claim_authorized_by_qualified_operational_subset_alone: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Reconstructs the precommitted, stratified SRSWOR sample and estimates a
/// finite-population mean using whole-cohort census outcomes.
pub fn evaluate_cluster_audit_v1(
    input: &ClusterAuditInputV1,
) -> Result<ClusterAuditReportV1, ClusterAuditErrorV1> {
    if input.schema != CLUSTER_AUDIT_INPUT_SCHEMA_V1 {
        return Err(ClusterAuditErrorV1("cluster-audit input schema is unsupported".into()));
    }
    require_text(&input.run_id, "input.run_id")?;
    let (frame_clusters, strata, cluster_sizes, population_units) =
        validate_frame(&input.frame)?;
    let population_clusters = frame_clusters.len();
    let seed = validate_design(&input.design, &input.frame)?;
    validate_plans(&input.design.stratum_plans, &strata)?;

    let plans: BTreeMap<&str, usize> = input
        .design
        .stratum_plans
        .iter()
        .map(|plan| (plan.stratum_id.as_str(), plan.audited_cluster_count))
        .collect();
    let mut expected_clusters = BTreeSet::new();
    for (stratum_id, cluster_ids) in &strata {
        let count = plans[stratum_id.as_str()];
        expected_clusters.extend(expected_sample(
            cluster_ids,
            count,
            seed,
            &input.design.design_id,
            stratum_id,
        )?);
    }

    let mut records = BTreeMap::new();
    let mut audited_unit_count = 0_usize;
    let mut cluster_losses: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut missing_reference = false;
    for record in &input.audited_clusters {
        if records.insert(record.cluster_id.as_str(), record).is_some() {
            return Err(ClusterAuditErrorV1("duplicate audited cluster record".into()));
        }
        let Some(frame_cluster) = frame_clusters.get(&record.cluster_id) else {
            return Err(ClusterAuditErrorV1(format!(
                "audit record {} references a foreign population cluster",
                record.cluster_id
            )));
        };
        let losses = validate_cluster_record(record, frame_cluster, &input.design)?;
        audited_unit_count = audited_unit_count
            .checked_add(frame_cluster.units.len())
            .ok_or_else(|| ClusterAuditErrorV1("audited unit count overflow".into()))?;
        match losses {
            Some(values) => {
                cluster_losses.insert(record.cluster_id.clone(), values);
            }
            None => missing_reference = true,
        }
    }
    let actual_clusters: BTreeSet<String> = records.keys().map(|id| (*id).to_string()).collect();
    if actual_clusters != expected_clusters {
        return Err(ClusterAuditErrorV1(
            "observed audit cluster sample differs from the precommitted randomization replay"
                .into(),
        ));
    }
    if audited_unit_count > input.design.maximum_audit_unit_capacity {
        return Err(ClusterAuditErrorV1(
            "whole-cluster audit exceeds the hard reference-metrology capacity".into(),
        ));
    }
    if missing_reference {
        return empty_report(
            input,
            ClusterAuditStatusV1::Unknown,
            "reference_audit_contains_unknown_or_blocked_measurement",
            population_clusters,
            population_units,
            actual_clusters.len(),
            audited_unit_count,
            true,
        );
    }

    let total_strata = strata.len();
    let log_term = (2.0 * total_strata as f64 / input.design.confidence_delta).ln();
    let mut population_estimate = 0.0;
    let mut total_design_variance = 0.0;
    let mut design_variance_available = true;
    let mut combined_hoeffding_radius = 0.0;
    let mut stratum_reports = Vec::with_capacity(total_strata);
    let mut exact_census = true;

    for (stratum_id, cluster_ids) in &strata {
        let population_cluster_count = cluster_ids.len();
        let audited_cluster_count = plans[stratum_id.as_str()];
        let cluster_size = cluster_sizes[stratum_id];
        let population_unit_count = population_cluster_count
            .checked_mul(cluster_size)
            .ok_or_else(|| ClusterAuditErrorV1("stratum population size overflow".into()))?;
        let mut means = Vec::with_capacity(audited_cluster_count);
        let mut audited_ids = Vec::with_capacity(audited_cluster_count);
        for cluster_id in expected_clusters.iter().filter(|id| {
            frame_clusters[*id].stratum_id == *stratum_id
        }) {
            let values = cluster_losses.get(cluster_id).ok_or_else(|| {
                ClusterAuditErrorV1(format!(
                    "selected cluster {cluster_id} has no qualified reference outcome"
                ))
            })?;
            let mean = values.iter().sum::<f64>() / cluster_size as f64;
            means.push(mean);
            audited_ids.push(cluster_id.clone());
        }
        audited_ids.sort();

        let stratum_mean = means.iter().sum::<f64>() / audited_cluster_count as f64;
        let weight = population_unit_count as f64 / population_units as f64;
        population_estimate += weight * stratum_mean;

        let first_order = audited_cluster_count as f64 / population_cluster_count as f64;
        let second_order = if audited_cluster_count >= 2 && population_cluster_count >= 2 {
            Some(
                audited_cluster_count as f64
                    * (audited_cluster_count - 1) as f64
                    / (population_cluster_count as f64 * (population_cluster_count - 1) as f64),
            )
        } else {
            None
        };

        let is_census = audited_cluster_count == population_cluster_count;
        exact_census &= is_census;
        let (sample_variance, stratum_variance, radius) = if is_census {
            (Some(0.0), Some(0.0), 0.0)
        } else {
            let radius = (log_term / (2.0 * audited_cluster_count as f64)).sqrt();
            if audited_cluster_count < 2 {
                design_variance_available = false;
                (None, None, radius)
            } else {
                let sample_mean = stratum_mean;
                let variance = means
                    .iter()
                    .map(|value| (value - sample_mean).powi(2))
                    .sum::<f64>()
                    / (audited_cluster_count - 1) as f64;
                let design_variance = (1.0 - first_order) * variance
                    / audited_cluster_count as f64;
                total_design_variance += weight.powi(2) * design_variance;
                (Some(variance), Some(design_variance), radius)
            }
        };
        combined_hoeffding_radius += weight * radius;
        stratum_reports.push(ClusterAuditStratumReportV1 {
            stratum_id: stratum_id.clone(),
            population_cluster_count,
            audited_cluster_count,
            cluster_size_units: cluster_size,
            population_unit_count,
            sampled_stratum_mean_loss: stratum_mean,
            population_weight: weight,
            first_order_cluster_inclusion_probability: first_order,
            second_order_cluster_inclusion_probability: second_order,
            sample_variance_of_cluster_means: sample_variance,
            design_variance_of_stratum_mean: stratum_variance,
            hoeffding_radius: radius,
            audited_cluster_ids: audited_ids,
        });
    }

    let design_variance = design_variance_available.then_some(total_design_variance);
    let design_standard_error = design_variance.map(f64::sqrt);
    let mut report = ClusterAuditReportV1 {
        schema: CLUSTER_AUDIT_REPORT_SCHEMA_V1.into(),
        run_id: input.run_id.clone(),
        input_sha256: input_digest(input)?,
        frame_sha256: input.frame.frame_sha256.clone(),
        design_sha256: canonical_design_digest(&input.design)?,
        status: if exact_census {
            ClusterAuditStatusV1::ExactCensus
        } else if design_variance_available {
            ClusterAuditStatusV1::Complete
        } else {
            ClusterAuditStatusV1::VarianceUnavailable
        },
        reason_codes: if design_variance_available {
            Vec::new()
        } else {
            vec!["one_or_more_strata_have_fewer_than_two_sampled_clusters".into()]
        },
        population_cluster_count: population_clusters,
        population_unit_count: population_units,
        audited_cluster_count: actual_clusters.len(),
        audited_unit_count,
        maximum_audit_unit_capacity: input.design.maximum_audit_unit_capacity,
        audit_capacity_remaining: input.design.maximum_audit_unit_capacity - audited_unit_count,
        population_risk_estimate: Some(population_estimate),
        design_variance_estimate: design_variance,
        design_standard_error,
        hoeffding_radius: Some(combined_hoeffding_radius.min(1.0)),
        hoeffding_lower: Some((population_estimate - combined_hoeffding_radius).clamp(0.0, 1.0)),
        hoeffding_upper: Some((population_estimate + combined_hoeffding_radius).clamp(0.0, 1.0)),
        exact_census,
        stratum_reports,
        randomization_replay_verified: true,
        audit_selection_independent_of_operational_process_declared: true,
        reference_method_sha256: input.design.reference_method_sha256.clone(),
        source_authenticity_proven: false,
        physical_cluster_independence_proven: false,
        population_claim_authorized_by_qualified_operational_subset_alone: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn unit_outcome(
        unit_id: &str,
        loss: f64,
        qualification: ReferenceAuditQualificationV1,
    ) -> ClusterAuditUnitOutcomeV1 {
        ClusterAuditUnitOutcomeV1 {
            unit_id: unit_id.into(),
            loss: if qualification == ReferenceAuditQualificationV1::Qualified {
                Some(loss)
            } else {
                None
            },
            reference_qualification: qualification,
            reference_label_id: format!("reference-label-{unit_id}"),
            reference_label_revision: 1,
            reference_label_payload_sha256: digest('c'),
            reference_measurement_event_id: format!("reference-event-{unit_id}"),
            reference_measurement_receipt_sha256: digest('a'),
            reference_qualification_report_sha256: digest('d'),
            reference_qualification_receipt_sha256: digest('b'),
        }
    }

    fn frame_cluster(id: &str, stratum: &str, size: usize) -> ClusterAuditFrameClusterV1 {
        ClusterAuditFrameClusterV1 {
            cluster_id: id.into(),
            stratum_id: stratum.into(),
            common_cause_membership_receipt_sha256: digest('c'),
            units: (0..size)
                .map(|index| ClusterAuditFrameUnitV1 {
                    unit_id: format!("{id}-unit-{index}"),
                    population_unit_receipt_sha256: digest('d'),
                })
                .collect(),
        }
    }

    fn make_record(cluster: &ClusterAuditFrameClusterV1, losses: &[f64]) -> ClusterAuditRecordV1 {
        let mut record = ClusterAuditRecordV1 {
            schema: CLUSTER_AUDIT_RECORD_SCHEMA_V1.into(),
            cluster_id: cluster.cluster_id.clone(),
            stratum_id: cluster.stratum_id.clone(),
            audited_at_unix_seconds: 150,
            reference_method_sha256: digest('9'),
            unit_outcomes: cluster
                .units
                .iter()
                .zip(losses)
                .map(|(unit, loss)| {
                    unit_outcome(&unit.unit_id, *loss, ReferenceAuditQualificationV1::Qualified)
                })
                .collect(),
            cluster_audit_receipt_sha256: String::new(),
        };
        record.cluster_audit_receipt_sha256 =
            seal_cluster_audit_record_v1(&record).unwrap();
        record
    }

    fn make_input() -> ClusterAuditInputV1 {
        let clusters = vec![
            frame_cluster("a", "small", 2),
            frame_cluster("b", "small", 2),
            frame_cluster("c", "small", 2),
            frame_cluster("d", "large", 3),
            frame_cluster("e", "large", 3),
            frame_cluster("f", "large", 3),
        ];
        let mut frame = ClusterAuditPopulationFrameV1 {
            schema: CLUSTER_AUDIT_FRAME_SCHEMA_V1.into(),
            population_manifest_sha256: digest('1'),
            operational_measurement_method_sha256: digest('2'),
            frame_frozen_at_unix_seconds: 100,
            clusters,
            frame_sha256: String::new(),
        };
        frame.frame_sha256 = seal_cluster_audit_frame_v1(&frame).unwrap();
        let seed = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let seed_bytes = hex_seed(seed).unwrap();
        let design_id = "audit-design-001";
        let commitment = seed_commitment(
            &seed_bytes,
            design_id,
            &frame.population_manifest_sha256,
            &frame.frame_sha256,
            CLUSTER_AUDIT_RANDOMIZATION_ALGORITHM_V1,
        );
        let design = ClusterAuditDesignV1 {
            schema: CLUSTER_AUDIT_DESIGN_SCHEMA_V1.into(),
            design_id: design_id.into(),
            randomization_algorithm: CLUSTER_AUDIT_RANDOMIZATION_ALGORITHM_V1.into(),
            population_manifest_sha256: frame.population_manifest_sha256.clone(),
            frame_sha256: frame.frame_sha256.clone(),
            randomization_seed_reveal_hex: seed.into(),
            seed_commitment_sha256: commitment,
            seed_committed_at_unix_seconds: 110,
            randomization_performed_at_unix_seconds: 120,
            audit_closed_at_unix_seconds: 200,
            seed_revealed_at_unix_seconds: 210,
            maximum_audit_unit_capacity: 10,
            confidence_delta: 0.05,
            reference_method_sha256: digest('9'),
            independent_reference_method_declared: true,
            independent_of_operational_process_declared: true,
            randomization_provenance_sha256: digest('8'),
            stratum_plans: vec![
                ClusterAuditStratumPlanV1 {
                    stratum_id: "small".into(),
                    audited_cluster_count: 2,
                },
                ClusterAuditStratumPlanV1 {
                    stratum_id: "large".into(),
                    audited_cluster_count: 2,
                },
            ],
        };
        let mut audited = Vec::new();
        for plan in &design.stratum_plans {
            let mut ids = frame
                .clusters
                .iter()
                .filter(|cluster| cluster.stratum_id == plan.stratum_id)
                .map(|cluster| cluster.cluster_id.clone())
                .collect::<Vec<_>>();
            let selected = expected_sample(
                &ids,
                plan.audited_cluster_count,
                seed_bytes,
                &design_id,
                &plan.stratum_id,
            )
            .unwrap();
            ids.retain(|id| selected.contains(id));
            for id in ids {
                let cluster = frame
                    .clusters
                    .iter()
                    .find(|cluster| cluster.cluster_id == id)
                    .unwrap();
                let losses = cluster
                    .units
                    .iter()
                    .enumerate()
                    .map(|(index, _)| (index as f64 + 1.0) / (cluster.units.len() as f64 + 1.0))
                    .collect::<Vec<_>>();
                audited.push(make_record(cluster, &losses));
            }
        }
        ClusterAuditInputV1 {
            schema: CLUSTER_AUDIT_INPUT_SCHEMA_V1.into(),
            run_id: "cluster-audit-known-answer-001".into(),
            frame,
            design,
            audited_clusters: audited,
        }
    }

    #[test]
    fn two_strata_cluster_census_known_answer_and_capacity() {
        let input = make_input();
        let report = evaluate_cluster_audit_v1(&input).unwrap();
        assert_eq!(report.status, ClusterAuditStatusV1::Complete);
        assert_eq!(report.population_unit_count, 15);
        assert_eq!(report.audited_unit_count, 10);
        assert_eq!(report.audit_capacity_remaining, 0);
        assert!(report.population_risk_estimate.is_some());
        assert!(report.design_variance_estimate.is_some());
        assert!(report.hoeffding_radius.is_some());
        assert!(report.randomization_replay_verified);
    }

    #[test]
    fn census_of_all_clusters_is_exact_and_has_zero_radius() {
        let mut input = make_input();
        for plan in &mut input.design.stratum_plans {
            plan.audited_cluster_count = 3;
        }
        input.design.maximum_audit_unit_capacity = 15;
        let seed = hex_seed(&input.design.randomization_seed_reveal_hex).unwrap();
        input.audited_clusters.clear();
        for plan in &input.design.stratum_plans {
            let ids = input
                .frame
                .clusters
                .iter()
                .filter(|cluster| cluster.stratum_id == plan.stratum_id)
                .map(|cluster| cluster.cluster_id.clone())
                .collect::<Vec<_>>();
            for id in ids {
                let cluster = input.frame.clusters.iter().find(|c| c.cluster_id == id).unwrap();
                let values = vec![0.25; cluster.units.len()];
                input.audited_clusters.push(make_record(cluster, &values));
            }
        }
        let report = evaluate_cluster_audit_v1(&input).unwrap();
        assert_eq!(report.status, ClusterAuditStatusV1::ExactCensus);
        assert_eq!(report.hoeffding_radius, Some(0.0));
        assert_eq!(report.design_variance_estimate, Some(0.0));
        assert_eq!(report.hoeffding_lower, report.population_risk_estimate);
        assert_eq!(report.hoeffding_upper, report.population_risk_estimate);
        let _ = seed;
    }

    #[test]
    fn within_cluster_dependence_does_not_create_fake_iid_wafer_precision() {
        let mut input = make_input();
        for record in &mut input.audited_clusters {
            let common_loss = record.unit_outcomes[0].loss.unwrap();
            for outcome in &mut record.unit_outcomes {
                outcome.loss = Some(common_loss);
            }
            record.cluster_audit_receipt_sha256 =
                seal_cluster_audit_record_v1(record).unwrap();
        }
        let report = evaluate_cluster_audit_v1(&input).unwrap();
        assert!(report.design_standard_error.is_some());
        assert!(report.hoeffding_radius.unwrap() > 0.0);
        assert!(!report.physical_cluster_independence_proven);
    }

    #[test]
    fn seed_commitment_mismatch_is_rejected() {
        let mut input = make_input();
        input.design.randomization_seed_reveal_hex =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        assert!(evaluate_cluster_audit_v1(&input).is_err());
    }

    #[test]
    fn one_cluster_per_non_census_stratum_has_no_design_variance() {
        let mut input = make_input();
        for plan in &mut input.design.stratum_plans {
            plan.audited_cluster_count = 1;
        }
        input.design.maximum_audit_unit_capacity = 5;
        let seed = hex_seed(&input.design.randomization_seed_reveal_hex).unwrap();
        input.audited_clusters.clear();
        for plan in &input.design.stratum_plans {
            let ids = input
                .frame
                .clusters
                .iter()
                .filter(|cluster| cluster.stratum_id == plan.stratum_id)
                .map(|cluster| cluster.cluster_id.clone())
                .collect::<Vec<_>>();
            let selected = expected_sample(
                &ids,
                1,
                seed,
                &input.design.design_id,
                &plan.stratum_id,
            )
            .unwrap();
            for id in selected {
                let cluster = input.frame.clusters.iter().find(|c| c.cluster_id == id).unwrap();
                input.audited_clusters.push(make_record(cluster, &vec![0.5; cluster.units.len()]));
            }
        }
        let report = evaluate_cluster_audit_v1(&input).unwrap();
        assert_eq!(report.status, ClusterAuditStatusV1::VarianceUnavailable);
        assert_eq!(report.design_variance_estimate, None);
        assert!(report.hoeffding_radius.is_some());
    }

    #[test]
    fn non_qualified_reference_outcome_returns_unknown_not_a_point_estimate() {
        let mut input = make_input();
        let record = &mut input.audited_clusters[0];
        record.unit_outcomes[0].reference_qualification =
            ReferenceAuditQualificationV1::Unknown;
        record.unit_outcomes[0].loss = None;
        record.cluster_audit_receipt_sha256 =
            seal_cluster_audit_record_v1(record).unwrap();
        let report = evaluate_cluster_audit_v1(&input).unwrap();
        assert_eq!(report.status, ClusterAuditStatusV1::Unknown);
        assert_eq!(report.population_risk_estimate, None);
        assert!(!report.randomization_replay_verified);
    }
}
