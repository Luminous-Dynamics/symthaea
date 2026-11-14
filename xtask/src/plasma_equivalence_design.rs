use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ClaimDisposition, ScopeEvidence};

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct SampleIdentity {
    pub(super) specimen_id: String,
    pub(super) cluster_id: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct ExcludedSample {
    arm: String,
    specimen_id: String,
    cluster_id: String,
    reason_ref: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct PowerReceipt {
    receipt_id: String,
    calculation_sha256: String,
    plan_sha256: String,
    design_id: String,
    endpoint_id: String,
    independent_unit: String,
    grouping_scheme: String,
    planned_baseline_clusters: usize,
    planned_candidate_clusters: usize,
    target_power: f64,
    target_precision_half_width: f64,
    assumptions_sha256: String,
    effect_under_alternative: f64,
    variance_low: f64,
    variance_central: f64,
    variance_high: f64,
    method_ref: String,
    claim_ceiling: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct EquivalenceAnalysisResult {
    pub(super) result_id: String,
    pub(super) result_sha256: String,
    pub(super) design_id: String,
    pub(super) plan_sha256: String,
    pub(super) endpoint_id: String,
    pub(super) analysis_profile: String,
    margin_lower: f64,
    margin_upper: f64,
    alpha: f64,
    confidence_level: f64,
    sample_manifest_sha256: String,
    baseline_sample_count: usize,
    candidate_sample_count: usize,
    baseline_cluster_count: usize,
    candidate_cluster_count: usize,
    ci_lower: f64,
    ci_upper: f64,
    outcome: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct DesignSourceBinding {
    pub(super) role: String,
    pub(super) cohort_role: String,
    pub(super) source_kind: String,
    pub(super) source_id: String,
    pub(super) artifact_sha256: String,
    pub(super) method_receipt: Option<ScopeEvidence>,
    pub(super) samples: Vec<SampleIdentity>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(super) struct EquivalenceDesignReceipt {
    pub(super) design_id: String,
    pub(super) plan_sha256: String,
    pub(super) endpoint_id: String,
    #[serde(default)]
    pub(super) endpoint_transform_id: Option<String>,
    #[serde(default)]
    pub(super) observation_endpoint_id: Option<String>,
    #[serde(default)]
    pub(super) endpoint_transform_sha256: Option<String>,
    #[serde(default)]
    pub(super) receipt_sha256: Option<String>,
    estimand: String,
    quantity_unit: String,
    pub(super) analysis_profile: String,
    grouping_scheme: String,
    independent_unit: String,
    margin_lower: f64,
    margin_upper: f64,
    alpha: f64,
    confidence_level: f64,
    frozen_at: i64,
    first_outcome_available_at: i64,
    design_changed_after_outcome: bool,
    planned_min_baseline_specimens: usize,
    planned_min_candidate_specimens: usize,
    planned_min_baseline_clusters: usize,
    planned_min_candidate_clusters: usize,
    max_specimen_attrition_rate: f64,
    max_cluster_attrition_rate: f64,
    target_power: f64,
    target_precision_half_width: f64,
    assumptions_sha256: String,
    effect_under_alternative: f64,
    variance_low: f64,
    variance_central: f64,
    variance_high: f64,
    randomization_plan_ref: String,
    analysis_plan_ref: String,
    exclusion_protocol_ref: String,
    time_source_ref: String,
    power_receipt: PowerReceipt,
    baseline_samples: Vec<SampleIdentity>,
    candidate_samples: Vec<SampleIdentity>,
    excluded_samples: Vec<ExcludedSample>,
    sample_manifest_sha256: String,
    baseline_method_receipt: ScopeEvidence,
    candidate_method_receipt: ScopeEvidence,
    #[serde(default)]
    pub(super) additional_source_bindings: Vec<DesignSourceBinding>,
    pub(super) analysis_result: EquivalenceAnalysisResult,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<DesignCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DesignCase {
    id: String,
    receipt: EquivalenceDesignReceipt,
    expected_disposition: String,
    expected_equivalence: String,
    rationale: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Disposition {
    Pass,
    Blocked,
    Rejected,
}

impl Disposition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Blocked => "BLOCKED",
            Self::Rejected => "REJECTED",
        }
    }
}

pub(super) struct Decision {
    pub(super) disposition: Disposition,
    pub(super) equivalence: &'static str,
    pub(super) reason: &'static str,
}

fn sha256_shape(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn unique_samples(samples: &[SampleIdentity]) -> bool {
    let mut specimens = HashSet::new();
    samples.iter().all(|sample| {
        nonempty(&sample.specimen_id)
            && nonempty(&sample.cluster_id)
            && specimens.insert(sample.specimen_id.as_str())
    })
}

fn specimen_set(samples: &[SampleIdentity]) -> HashSet<&str> {
    samples.iter().map(|sample| sample.specimen_id.as_str()).collect()
}

fn cluster_set(samples: &[SampleIdentity]) -> HashSet<&str> {
    samples.iter().map(|sample| sample.cluster_id.as_str()).collect()
}

fn attrition_rate(included: usize, excluded: usize) -> f64 {
    let total = included + excluded;
    if total == 0 { 1.0 } else { excluded as f64 / total as f64 }
}

pub(super) fn audit_receipt(receipt: &EquivalenceDesignReceipt) -> Decision {
    let blocked = |reason| Decision {
        disposition: Disposition::Blocked,
        equivalence: "NOT_EVALUATED",
        reason,
    };
    let rejected = |reason| Decision {
        disposition: Disposition::Rejected,
        equivalence: "NOT_EVALUATED",
        reason,
    };

    for value in [
        receipt.design_id.as_str(),
        receipt.endpoint_id.as_str(),
        receipt.estimand.as_str(),
        receipt.quantity_unit.as_str(),
        receipt.analysis_profile.as_str(),
        receipt.independent_unit.as_str(),
        receipt.power_receipt.receipt_id.as_str(),
        receipt.power_receipt.method_ref.as_str(),
        receipt.randomization_plan_ref.as_str(),
        receipt.analysis_plan_ref.as_str(),
        receipt.exclusion_protocol_ref.as_str(),
        receipt.time_source_ref.as_str(),
        receipt.analysis_result.result_id.as_str(),
    ] {
        if !nonempty(value) {
            return blocked("design, endpoint, estimand, unit, analysis or linked receipt identity is missing");
        }
    }

    if !matches!(receipt.grouping_scheme.as_str(), "independent_arms" | "paired_by_cluster") {
        return rejected("grouping scheme is unsupported by this audit profile");
    }
    let expected_analysis_profile = match receipt.grouping_scheme.as_str() {
        "independent_arms" => "two_sample_tost_v1",
        "paired_by_cluster" => "paired_tost_v1",
        _ => unreachable!("grouping scheme checked above"),
    };
    if receipt.analysis_profile != expected_analysis_profile {
        return rejected("analysis profile is not registered for the declared independent-unit design");
    }
    if !matches!(receipt.independent_unit.as_str(), "material_lot" | "adhesive_batch" | "plasma_head" | "production_day" | "production_line" | "manufacturing_site" | "paired_cluster") {
        return blocked("independent experimental unit is unknown or unregistered");
    }
    if receipt.grouping_scheme == "paired_by_cluster"
        && (receipt.independent_unit != "paired_cluster"
            || receipt.planned_min_baseline_clusters != receipt.planned_min_candidate_clusters)
    {
        return rejected("paired-by-cluster design must declare paired_cluster and equal paired-cluster minima");
    }
    if receipt.grouping_scheme == "independent_arms" && receipt.independent_unit == "paired_cluster" {
        return rejected("independent-arm design cannot declare a paired-cluster independent unit");
    }

    if !sha256_shape(&receipt.plan_sha256)
        || !sha256_shape(&receipt.sample_manifest_sha256)
        || !sha256_shape(&receipt.assumptions_sha256)
        || !sha256_shape(&receipt.power_receipt.calculation_sha256)
        || !sha256_shape(&receipt.power_receipt.assumptions_sha256)
        || !sha256_shape(&receipt.analysis_result.result_sha256)
    {
        return blocked("plan, manifest, power or analysis receipt digest is missing or malformed");
    }

    if receipt.design_changed_after_outcome || receipt.frozen_at >= receipt.first_outcome_available_at {
        return rejected("the design was changed or frozen only after outcome availability");
    }
    if receipt.margin_lower >= receipt.margin_upper
        || !receipt.margin_lower.is_finite()
        || !receipt.margin_upper.is_finite()
    {
        return rejected("equivalence margins are non-finite or not strictly ordered");
    }
    if !receipt.alpha.is_finite() || receipt.alpha <= 0.0 || receipt.alpha >= 0.5 {
        return rejected("alpha is outside the supported two-one-sided-test range");
    }
    if !receipt.confidence_level.is_finite()
        || (receipt.confidence_level - (1.0 - 2.0 * receipt.alpha)).abs() > 1e-9
    {
        return rejected("confidence level is inconsistent with the declared two-one-sided-test alpha");
    }
    if !receipt.max_specimen_attrition_rate.is_finite()
        || !(0.0..1.0).contains(&receipt.max_specimen_attrition_rate)
        || !receipt.max_cluster_attrition_rate.is_finite()
        || !(0.0..1.0).contains(&receipt.max_cluster_attrition_rate)
        || !receipt.target_power.is_finite()
        || receipt.target_power <= 0.0
        || receipt.target_power >= 1.0
        || !receipt.target_precision_half_width.is_finite()
        || receipt.target_precision_half_width <= 0.0
        || !receipt.effect_under_alternative.is_finite()
        || receipt.effect_under_alternative == 0.0
        || !receipt.variance_low.is_finite()
        || !receipt.variance_central.is_finite()
        || !receipt.variance_high.is_finite()
        || receipt.variance_low <= 0.0
        || receipt.variance_central < receipt.variance_low
        || receipt.variance_high < receipt.variance_central
    {
        return rejected("attrition, power or precision planning limits are invalid");
    }
    if receipt.planned_min_baseline_specimens == 0
        || receipt.planned_min_candidate_specimens == 0
        || receipt.planned_min_baseline_clusters == 0
        || receipt.planned_min_candidate_clusters == 0
    {
        return rejected("planned specimen and independent-cluster minimums must be positive");
    }

    let power = &receipt.power_receipt;
    if power.plan_sha256 != receipt.plan_sha256
        || power.design_id != receipt.design_id
        || power.endpoint_id != receipt.endpoint_id
        || power.independent_unit != receipt.independent_unit
        || power.grouping_scheme != receipt.grouping_scheme
        || power.planned_baseline_clusters != receipt.planned_min_baseline_clusters
        || power.planned_candidate_clusters != receipt.planned_min_candidate_clusters
        || (power.target_power - receipt.target_power).abs() > 1e-9
        || (power.target_precision_half_width - receipt.target_precision_half_width).abs() > 1e-9
        || power.assumptions_sha256 != receipt.assumptions_sha256
        || !close(power.effect_under_alternative, receipt.effect_under_alternative)
        || !close(power.variance_low, receipt.variance_low)
        || !close(power.variance_central, receipt.variance_central)
        || !close(power.variance_high, receipt.variance_high)
    {
        return rejected("power/precision planning receipt does not match the frozen design contract");
    }
    if power.claim_ceiling != "planning_only" {
        return rejected("power/precision receipt exceeds its permitted planning-only claim ceiling");
    }

    if !unique_samples(&receipt.baseline_samples) || !unique_samples(&receipt.candidate_samples) {
        return rejected("included specimen identities are blank or duplicated within an arm");
    }
    let baseline_specimens = specimen_set(&receipt.baseline_samples);
    let candidate_specimens = specimen_set(&receipt.candidate_samples);
    if !baseline_specimens.is_disjoint(&candidate_specimens) {
        return rejected("the same specimen identity appears in both baseline and candidate arms");
    }

    let baseline_clusters = cluster_set(&receipt.baseline_samples);
    let candidate_clusters = cluster_set(&receipt.candidate_samples);
    let paired_count = baseline_clusters.intersection(&candidate_clusters).count();

    if receipt.grouping_scheme == "independent_arms"
        && !baseline_clusters.is_disjoint(&candidate_clusters)
    {
        return rejected("independent-arm design reuses a manufacturing cluster across arms");
    }
    if receipt.grouping_scheme == "paired_by_cluster"
        && (baseline_clusters != candidate_clusters || baseline_clusters.is_empty())
    {
        return rejected("paired-by-cluster design does not contain exact baseline/candidate cluster pairs");
    }

    let mut excluded_ids = HashSet::new();
    let mut excluded_by_arm = [HashSet::new(), HashSet::new()];
    let mut excluded_clusters_by_arm = [HashSet::new(), HashSet::new()];
    for excluded in &receipt.excluded_samples {
        if !matches!(excluded.arm.as_str(), "baseline" | "candidate")
            || !nonempty(&excluded.specimen_id)
            || !nonempty(&excluded.cluster_id)
            || !nonempty(&excluded.reason_ref)
        {
            return blocked("an excluded specimen lacks a valid arm, identity or reason reference");
        }
        if !excluded_ids.insert(excluded.specimen_id.as_str())
            || baseline_specimens.contains(excluded.specimen_id.as_str())
            || candidate_specimens.contains(excluded.specimen_id.as_str())
        {
            return rejected("an excluded specimen is duplicated or also included in the analysis");
        }
        let arm = if excluded.arm == "baseline" { 0 } else { 1 };
        excluded_by_arm[arm].insert(excluded.specimen_id.as_str());
        excluded_clusters_by_arm[arm].insert(excluded.cluster_id.as_str());
    }

    if receipt.grouping_scheme == "paired_by_cluster"
        && excluded_clusters_by_arm[0] != excluded_clusters_by_arm[1]
    {
        return rejected("paired-by-cluster attrition does not remove the same cluster pairs from both arms");
    }

    let baseline_total = receipt.baseline_samples.len() + excluded_by_arm[0].len();
    let candidate_total = receipt.candidate_samples.len() + excluded_by_arm[1].len();
    if attrition_rate(receipt.baseline_samples.len(), excluded_by_arm[0].len())
        > receipt.max_specimen_attrition_rate
        || attrition_rate(receipt.candidate_samples.len(), excluded_by_arm[1].len())
            > receipt.max_specimen_attrition_rate
    {
        return DispositionDecision::blocked("specimen attrition exceeds the preregistered allowance");
    }

    let baseline_all_clusters: HashSet<&str> = baseline_clusters
        .union(&excluded_clusters_by_arm[0])
        .copied()
        .collect();
    let candidate_all_clusters: HashSet<&str> = candidate_clusters
        .union(&excluded_clusters_by_arm[1])
        .copied()
        .collect();
    let baseline_excluded_cluster_count = baseline_all_clusters.len() - baseline_clusters.len();
    let candidate_excluded_cluster_count = candidate_all_clusters.len() - candidate_clusters.len();
    if attrition_rate(baseline_clusters.len(), baseline_excluded_cluster_count)
        > receipt.max_cluster_attrition_rate
        || attrition_rate(candidate_clusters.len(), candidate_excluded_cluster_count)
            > receipt.max_cluster_attrition_rate
    {
        return blocked("independent-cluster attrition exceeds the preregistered allowance");
    }

    if receipt.baseline_samples.len() < receipt.planned_min_baseline_specimens
        || receipt.candidate_samples.len() < receipt.planned_min_candidate_specimens
        || baseline_clusters.len() < receipt.planned_min_baseline_clusters
        || candidate_clusters.len() < receipt.planned_min_candidate_clusters
    {
        return blocked("actual specimen or independent-cluster count is below the frozen minimum");
    }

    if receipt.grouping_scheme == "paired_by_cluster"
        && paired_count < receipt.planned_min_baseline_clusters
    {
        return blocked("actual independent cluster-pair count is below the frozen minimum");
    }

    let baseline_receipt = &receipt.baseline_method_receipt;
    let candidate_receipt = &receipt.candidate_method_receipt;
    let expected_auxiliary_roles: &[(&str, &str, &str)] =
        match receipt.endpoint_transform_id.as_deref() {
            None => &[],
            Some("unit_preserving_strength_projection_v1") => &[],
            Some("aged_strength_retention_ratio_v1") => &[
                ("baseline_unaged_strength", "baseline_unaged_strength", "observation"),
                ("candidate_unaged_strength", "candidate_unaged_strength", "observation"),
                ("baseline_aged_strength::conditioning", "baseline_aged_strength", "conditioning_method"),
                ("baseline_aged_strength::exposure", "baseline_aged_strength", "exposure_record"),
                ("candidate_aged_strength::conditioning", "candidate_aged_strength", "conditioning_method"),
                ("candidate_aged_strength::exposure", "candidate_aged_strength", "exposure_record"),
            ],
            Some("predeclared_failure_mode_fraction_v1") => {
                return blocked("cluster-aware proportion-equivalence analysis is not yet qualified; failure-mode fraction remains blocked");
            }
            Some(_) => return rejected("endpoint transform is not registered for design source bindings"),
        };
    if receipt.additional_source_bindings.len() != expected_auxiliary_roles.len() {
        return if expected_auxiliary_roles.is_empty() {
            rejected("direct endpoint design must not contain unregistered auxiliary sources")
        } else {
            blocked("design receipt is missing required auxiliary source bindings")
        };
    }

    let mut auxiliary_roles = HashSet::new();
    let mut auxiliary_ids = HashSet::from([
        baseline_receipt.id.as_str(),
        candidate_receipt.id.as_str(),
    ]);
    let mut auxiliary_digests = HashSet::from([
        baseline_receipt.raw_data_sha256.as_deref().unwrap_or(""),
        candidate_receipt.raw_data_sha256.as_deref().unwrap_or(""),
    ]);
    let mut cohort_sample_maps = HashMap::<&str, HashMap<&str, &str>>::new();
    let baseline_primary: HashMap<&str, &str> = receipt.baseline_samples
        .iter().map(|sample| (sample.specimen_id.as_str(), sample.cluster_id.as_str())).collect();
    let candidate_primary: HashMap<&str, &str> = receipt.candidate_samples
        .iter().map(|sample| (sample.specimen_id.as_str(), sample.cluster_id.as_str())).collect();

    for binding in &receipt.additional_source_bindings {
        if !nonempty(&binding.role)
            || !nonempty(&binding.cohort_role)
            || !nonempty(&binding.source_id)
        {
            return blocked("auxiliary source role, cohort role or receipt identity is missing");
        }
        let Some((_, expected_cohort, expected_kind)) = expected_auxiliary_roles
            .iter()
            .find(|(role, _, _)| *role == binding.role.as_str())
        else {
            return rejected("auxiliary source role is not registered for this endpoint transform");
        };
        if !auxiliary_roles.insert(binding.role.as_str()) {
            return rejected("auxiliary source role is duplicated");
        }
        if binding.cohort_role != *expected_cohort || binding.source_kind != *expected_kind {
            return rejected("auxiliary source cohort or kind does not match its registered role");
        }
        if !sha256_shape(&binding.artifact_sha256) {
            return blocked("auxiliary source artifact digest is missing or malformed");
        }
        if !auxiliary_ids.insert(binding.source_id.as_str())
            || !auxiliary_digests.insert(binding.artifact_sha256.as_str())
        {
            return rejected("auxiliary source receipt or artifact is reused");
        }
        if !unique_samples(&binding.samples) {
            return blocked("auxiliary source specimen/cluster mapping is incomplete");
        }

        let mut this_cohort = HashMap::<&str, &str>::new();
        for sample in &binding.samples {
            if this_cohort.insert(sample.specimen_id.as_str(), sample.cluster_id.as_str()).is_some() {
                return rejected("auxiliary source repeats a specimen identity");
            }
        }
        if let Some(existing) = cohort_sample_maps.get(binding.cohort_role.as_str()) {
            if existing != &this_cohort {
                return rejected("sources assigned to one cohort disagree about specimen/cluster lineage");
            }
        } else {
            cohort_sample_maps.insert(binding.cohort_role.as_str(), this_cohort);
        }

        match (binding.source_kind.as_str(), binding.method_receipt.as_ref()) {
            ("observation", Some(method_receipt)) => {
                if method_receipt.id != binding.source_id
                    || method_receipt.raw_data_sha256.as_deref() != Some(binding.artifact_sha256.as_str())
                    || method_receipt.endpoint != observation_endpoint
                    || method_receipt.method_id != baseline_receipt.method_id
                    || method_receipt.edition != baseline_receipt.edition
                    || method_receipt.evidence_kind != baseline_receipt.evidence_kind
                    || method_receipt.material_pair_class != baseline_receipt.material_pair_class
                    || method_receipt.geometry != baseline_receipt.geometry
                    || method_receipt.protocol_id != baseline_receipt.protocol_id
                    || method_receipt.protocol_sha256 != baseline_receipt.protocol_sha256
                    || !same_members(
                        &method_receipt.specimen_ids,
                        &binding.samples.iter().map(|sample| sample.specimen_id.clone()).collect::<Vec<_>>(),
                    )
                {
                    return rejected("auxiliary observation receipt does not match its exact source manifest or method scope");
                }
                match decide(method_receipt).disposition {
                    ClaimDisposition::Supported => {}
                    ClaimDisposition::Blocked => return blocked("auxiliary observation receipt is not method-qualified"),
                    ClaimDisposition::Rejected => return rejected("auxiliary observation receipt exceeds its method claim scope"),
                }
            }
            ("conditioning_method", Some(method_receipt)) => {
                if method_receipt.id != binding.source_id
                    || method_receipt.raw_data_sha256.as_deref() != Some(binding.artifact_sha256.as_str())
                    || method_receipt.evidence_kind != "conditioning_practice"
                    || method_receipt.requested_claim != "conditioning_protocol_reference"
                    || !same_members(
                        &method_receipt.specimen_ids,
                        &binding.samples.iter().map(|sample| sample.specimen_id.clone()).collect::<Vec<_>>(),
                    )
                {
                    return rejected("conditioning method receipt does not match its exact exposure cohort");
                }
                match decide(method_receipt).disposition {
                    ClaimDisposition::Supported => {}
                    ClaimDisposition::Blocked => return blocked("conditioning method receipt is not qualified"),
                    ClaimDisposition::Rejected => return rejected("conditioning method receipt exceeds its registered scope"),
                }
            }
            ("exposure_record", None) => {}
            _ => return rejected("auxiliary source kind and method receipt presence are inconsistent"),
        }

        let primary = match binding.cohort_role.as_str() {
            "baseline_unaged_strength" | "baseline_aged_strength" => &baseline_primary,
            "candidate_unaged_strength" | "candidate_aged_strength" => &candidate_primary,
            _ => return rejected("auxiliary cohort role is not registered"),
        };
        match binding.cohort_role.as_str() {
            "baseline_unaged_strength" | "candidate_unaged_strength" => {
                let cluster_set: HashSet<&str> = binding.samples.iter().map(|s| s.cluster_id.as_str()).collect();
                let primary_cluster_set: HashSet<&str> = primary.values().copied().collect();
                let planned_specimens = if binding.cohort_role == "baseline_unaged_strength" {
                    receipt.planned_min_baseline_specimens
                } else {
                    receipt.planned_min_candidate_specimens
                };
                let planned_clusters = if binding.cohort_role == "baseline_unaged_strength" {
                    receipt.planned_min_baseline_clusters
                } else {
                    receipt.planned_min_candidate_clusters
                };
                if cluster_set != primary_cluster_set
                    || binding.samples.len() < planned_specimens
                    || cluster_set.len() < planned_clusters
                    || binding.samples.iter().any(|sample| primary.contains_key(sample.specimen_id.as_str()))
                {
                    return rejected("unaged auxiliary cohort must meet the frozen minima, use the corresponding arm's clusters and contain distinct specimens");
                }
            }
            "baseline_aged_strength" | "candidate_aged_strength" => {
                let same_manifest = binding.samples.len() == primary.len()
                    && binding.samples.iter().all(|sample| {
                        primary.get(sample.specimen_id.as_str()) == Some(&sample.cluster_id.as_str())
                    });
                if !same_manifest {
                    return rejected("conditioning/exposure evidence must bind the exact aged specimen cohort");
                }
            }
            _ => unreachable!(),
        }
    }

    if receipt.endpoint_transform_id.as_deref() == Some("aged_strength_retention_ratio_v1") {
        let Some(baseline_unaged) = cohort_sample_maps.get("baseline_unaged_strength") else {
            return blocked("baseline unaged source manifest is missing");
        };
        let Some(candidate_unaged) = cohort_sample_maps.get("candidate_unaged_strength") else {
            return blocked("candidate unaged source manifest is missing");
        };
        let baseline_aged = &baseline_primary;
        let candidate_aged = &candidate_primary;
        let cohorts = [baseline_unaged, candidate_unaged, baseline_aged, candidate_aged];
        let mut all_specimens = HashSet::new();
        for cohort in cohorts {
            for specimen in cohort.keys() {
                if !all_specimens.insert(*specimen) {
                    return rejected("unaged and aged source cohorts reuse a physical specimen identity");
                }
            }
        }
        let baseline_unaged_clusters: HashSet<&str> = baseline_unaged.values().copied().collect();
        let candidate_unaged_clusters: HashSet<&str> = candidate_unaged.values().copied().collect();
        let baseline_aged_clusters: HashSet<&str> = baseline_aged.values().copied().collect();
        let candidate_aged_clusters: HashSet<&str> = candidate_aged.values().copied().collect();
        if baseline_unaged_clusters != baseline_aged_clusters
            || candidate_unaged_clusters != candidate_aged_clusters
            || !baseline_aged_clusters.is_disjoint(&candidate_aged_clusters)
        {
            return rejected("four-cohort retention design does not preserve within-arm lot pairing and between-arm independence");
        }
    }
    if expected_auxiliary_roles
        .iter()
        .any(|(role, _, _)| !auxiliary_roles.contains(role))
    {
        return blocked("one or more required auxiliary source roles are missing");
    }
    let result = &receipt.analysis_result;
    if !nonempty(&result.result_id)
        || !sha256_shape(&result.result_sha256)
        || result.design_id != receipt.design_id
        || result.plan_sha256 != receipt.plan_sha256
        || result.endpoint_id != receipt.endpoint_id
        || result.analysis_profile != receipt.analysis_profile
        || result.sample_manifest_sha256 != receipt.sample_manifest_sha256
        || result.baseline_sample_count != receipt.baseline_samples.len()
        || result.candidate_sample_count != receipt.candidate_samples.len()
        || result.baseline_cluster_count != baseline_clusters.len()
        || result.candidate_cluster_count != candidate_clusters.len()
        || !close(result.margin_lower, receipt.margin_lower)
        || !close(result.margin_upper, receipt.margin_upper)
        || !close(result.alpha, receipt.alpha)
        || !close(result.confidence_level, receipt.confidence_level)
    {
        return rejected("analysis result receipt does not match the frozen plan, sample manifest, cluster counts or equivalence contract");
    }
    if !result.ci_lower.is_finite() || !result.ci_upper.is_finite() || result.ci_lower > result.ci_upper {
        return rejected("analysis interval is non-finite or reversed");
    }
    let equivalent = result.ci_lower >= receipt.margin_lower
        && result.ci_upper <= receipt.margin_upper;
    let outcome = if equivalent { "EQUIVALENCE_ESTABLISHED" } else { "EQUIVALENCE_NOT_ESTABLISHED" };
    if result.outcome != outcome {
        return rejected("reported equivalence outcome does not match the declared interval and equivalence margins");
    }

    // Included + excluded totals remain available for the attrition contract;
    // only the included samples and clusters feed the analysis-result counts.
    let _ = (baseline_total, candidate_total);
    Decision {
        disposition: Disposition::Pass,
        equivalence: outcome,
        reason: "frozen design, independent-unit grouping, method receipts, actual sample manifest and analysis-result lineage are structurally consistent; numerical validity is not certified",
    }
}

pub(super) fn validate_bridge(
    receipt: &EquivalenceDesignReceipt,
    endpoint: &str,
    baseline_method_receipt: &ScopeEvidence,
    candidate_method_receipt: &ScopeEvidence,
) -> Decision {
    let decision = audit_receipt(receipt);
    if decision.disposition != Disposition::Pass {
        return decision;
    }
    if receipt.endpoint_id != endpoint
        || !same_scope_evidence(&receipt.baseline_method_receipt, baseline_method_receipt)
        || !same_scope_evidence(&receipt.candidate_method_receipt, candidate_method_receipt)
    {
        return Decision {
            disposition: Disposition::Rejected,
            equivalence: "NOT_EVALUATED",
            reason: "design receipt is not bound to the exact method-qualified endpoint receipts used by the transfer bridge",
        };
    }
    decision
}

fn same_scope_evidence(left: &ScopeEvidence, right: &ScopeEvidence) -> bool {
    left.id == right.id
        && left.method_id == right.method_id
        && left.edition == right.edition
        && left.evidence_kind == right.evidence_kind
        && left.material_pair_class == right.material_pair_class
        && left.geometry == right.geometry
        && left.endpoint == right.endpoint
        && left.requested_claim == right.requested_claim
        && left.protocol_id == right.protocol_id
        && left.protocol_sha256 == right.protocol_sha256
        && left.raw_data_sha256 == right.raw_data_sha256
        && left.suitability_review_sha256 == right.suitability_review_sha256
        && left.specimen_ids == right.specimen_ids
}

fn specimen_set_from_strings<'a>(values: &'a [String]) -> HashSet<&'a str> {
    values.iter().map(String::as_str).collect()
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9
}

struct DispositionDecision;

impl DispositionDecision {
    fn blocked(reason: &'static str) -> Decision {
        Decision { disposition: Disposition::Blocked, equivalence: "NOT_EVALUATED", reason }
    }
}

fn audit_case(case: &DesignCase) -> Decision {
    let decision = audit_receipt(&case.receipt);
    if decision.disposition.as_str() != case.expected_disposition {
        return decision;
    }
    if decision.equivalence != case.expected_equivalence {
        return Decision {
            disposition: Disposition::Rejected,
            equivalence: decision.equivalence,
            reason: "fixture equivalence outcome mismatch",
        };
    }
    decision
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-equivalence-design-fixture-v0.1" {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }

    let mut counts = [0usize; 3];
    for case in &fixture.cases {
        let decision = audit_case(case);
        let index = match decision.disposition {
            Disposition::Pass => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        println!("{}: {} / {} — {}", case.id, decision.disposition.as_str(), decision.equivalence, decision.reason);
        if decision.disposition.as_str() != case.expected_disposition {
            bail!("case {}: expected {}, got {}", case.id, case.expected_disposition, decision.disposition.as_str());
        }
        if decision.equivalence != case.expected_equivalence {
            bail!("case {}: expected equivalence {}, got {}", case.id, case.expected_equivalence, decision.equivalence);
        }
        if !nonempty(&case.rationale) {
            bail!("case {} has an empty rationale", case.id);
        }
    }

    println!("PLASMA-EQUIVALENCE-DESIGN S0 — {} cases; PASS={} BLOCKED={} REJECTED={}", fixture.cases.len(), counts[0], counts[1], counts[2]);
    println!("Claim ceiling: design/result structural consistency only; no power, statistical-model, physical-equivalence or production qualification claim.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_equivalence_design_fixture_v0_1.json");

    #[test]
    fn frozen_design_fixture_matches_all_dispositions() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid equivalence design fixture");
        assert_eq!(fixture.schema_version, "plasma-equivalence-design-fixture-v0.1");
        for case in &fixture.cases {
            let result = audit_case(case);
            assert_eq!(result.disposition.as_str(), case.expected_disposition, "{}", case.id);
            assert_eq!(result.equivalence, case.expected_equivalence, "{}", case.id);
        }
    }

    #[test]
    fn coupon_count_cannot_replace_independent_cluster_count() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid equivalence design fixture");
        let case = fixture.cases.iter().find(|case| case.id == "many-coupons-too-few-lots").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Blocked);
    }

    #[test]
    fn post_outcome_design_change_is_rejected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid equivalence design fixture");
        let case = fixture.cases.iter().find(|case| case.id == "plan-frozen-after-outcomes").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Rejected);
    }

    #[test]
    fn paired_clusters_are_counted_once_not_twice() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid equivalence design fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-paired-by-lot-design").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Pass);
    }

    #[test]
    fn aged_retention_design_binds_unaged_conditioning_and_exposure_sources() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../data/research/plasma_endpoint_transform_fixture_v0_1.json"))
                .expect("valid endpoint transform fixture");
        let case = fixture["cases"].as_array().expect("cases").iter()
            .find(|case| case["id"] == "valid-aged-retention-four-cohort-graph")
            .expect("aged retention case exists");
        let receipt: EquivalenceDesignReceipt =
            serde_json::from_value(case["graph"]["design_receipt"].clone())
                .expect("valid aged retention design receipt");
        assert_eq!(audit_receipt(&receipt).disposition, Disposition::Pass);
        assert_eq!(receipt.additional_source_bindings.len(), 6);
    }

    #[test]
    fn missing_unaged_design_source_binding_blocks_retention() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../data/research/plasma_endpoint_transform_fixture_v0_1.json"))
                .expect("valid endpoint transform fixture");
        let case = fixture["cases"].as_array().expect("cases").iter()
            .find(|case| case["id"] == "valid-aged-retention-four-cohort-graph")
            .expect("aged retention case exists");
        let mut value = case["graph"]["design_receipt"].clone();
        let bindings = value["additional_source_bindings"].as_array_mut().expect("bindings");
        bindings.retain(|binding| binding["role"] != "baseline_unaged_strength");
        let receipt: EquivalenceDesignReceipt =
            serde_json::from_value(value).expect("partial aged retention receipt parses");
        assert_eq!(audit_receipt(&receipt).disposition, Disposition::Blocked);
    }

    #[test]
    fn unaged_design_source_must_preserve_its_arm_lot_mapping() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../data/research/plasma_endpoint_transform_fixture_v0_1.json"))
                .expect("valid endpoint transform fixture");
        let case = fixture["cases"].as_array().expect("cases").iter()
            .find(|case| case["id"] == "valid-aged-retention-four-cohort-graph")
            .expect("aged retention case exists");
        let mut value = case["graph"]["design_receipt"].clone();
        let binding = value["additional_source_bindings"].as_array_mut().expect("bindings").iter_mut()
            .find(|binding| binding["role"] == "baseline_unaged_strength").expect("unaged binding");
        binding["samples"][0]["cluster_id"] = serde_json::Value::String("UNRELATED-LOT".to_owned());
        let receipt: EquivalenceDesignReceipt =
            serde_json::from_value(value).expect("mismatched aged retention receipt parses");
        assert_eq!(audit_receipt(&receipt).disposition, Disposition::Rejected);
    }

    #[test]
    fn power_assumptions_are_bound_to_the_frozen_plan() {    #[test]
    fn power_assumptions_are_bound_to_the_frozen_plan() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid equivalence design fixture");
        let case = fixture.cases.iter().find(|case| case.id == "power-assumption-digest-mismatch").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Rejected);
    }

    #[test]
    fn missing_randomization_protocol_blocks_the_design() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid equivalence design fixture");
        let case = fixture.cases.iter().find(|case| case.id == "missing-randomization-protocol-ref").expect("case exists");
        assert_eq!(audit_receipt(&case.receipt).disposition, Disposition::Blocked);
    }

    #[test]
    fn failure_fraction_design_is_blocked_until_clustered_equivalence_is_qualified() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid equivalence design fixture");
        let valid = fixture.cases.iter()
            .find(|case| case.id == "valid-independent-lot-arms")
            .expect("valid design case exists");
        let mut receipt: EquivalenceDesignReceipt =
            serde_json::from_value(serde_json::to_value(&valid.receipt).expect("serialize receipt"))
                .expect("clone receipt");
        receipt.endpoint_id = "failure_mode_fraction".to_owned();
        receipt.endpoint_transform_id = Some("predeclared_failure_mode_fraction_v1".to_owned());
        receipt.observation_endpoint_id = Some("frp_joint_failure_mode_classification".to_owned());
        assert_eq!(audit_receipt(&receipt).disposition, Disposition::Blocked);
    }

    #[test]
    fn design_pass_does_not_require_equivalence_to_be_established() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid equivalence design fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-design-but-interval-outside-margin").expect("case exists");
        let result = audit_case(case);
        assert_eq!(result.disposition, Disposition::Pass);
        assert_eq!(result.equivalence, "EQUIVALENCE_NOT_ESTABLISHED");
    }
}
