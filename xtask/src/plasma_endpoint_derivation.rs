use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ClaimDisposition, ScopeEvidence};

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Sample {
    specimen_id: String,
    cluster_id: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct StrengthAggregate {
    receipt_id: String,
    result_sha256: String,
    sample_manifest_sha256: String,
    scope_evidence: ScopeEvidence,
    samples: Vec<Sample>,
    mean_strength_mpa: f64,
    conditioning_profile_id: Option<String>,
    conditioning_profile_sha256: Option<String>,
    conditioning_receipt: Option<ScopeEvidence>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct RetentionArm {
    profile_sha256: String,
    initial: StrengthAggregate,
    aged: StrengthAggregate,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct RetentionResult {
    result_id: String,
    result_sha256: String,
    design_id: String,
    plan_sha256: String,
    formula_sha256: String,
    input_manifest_sha256: String,
    uncertainty_receipt_sha256: String,
    analysis_profile: String,
    baseline_retention: f64,
    candidate_retention: f64,
    candidate_minus_baseline: f64,
    ci_lower: f64,
    ci_upper: f64,
    margin_lower: f64,
    margin_upper: f64,
    outcome: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct FailureClassification {
    specimen_id: String,
    cluster_id: String,
    category_ids: Vec<String>,
    rater_count: usize,
    adjudication_sha256: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct FailureModeResult {
    result_id: String,
    result_sha256: String,
    design_id: String,
    plan_sha256: String,
    taxonomy_sha256: String,
    input_manifest_sha256: String,
    uncertainty_receipt_sha256: String,
    analysis_profile: String,
    baseline_numerator: usize,
    baseline_denominator: usize,
    candidate_numerator: usize,
    candidate_denominator: usize,
    baseline_fraction: f64,
    candidate_fraction: f64,
    candidate_minus_baseline: f64,
    ci_lower: f64,
    ci_upper: f64,
    margin_lower: f64,
    margin_upper: f64,
    outcome: String,
    claim_ceiling: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct DirectObservation {
    endpoint_id: String,
    quantity_unit: String,
    result_sha256: String,
    value: f64,
    scope_evidence: ScopeEvidence,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum EndpointPayload {
    DirectObservation {
        receipt: DirectObservation,
    },
    AgedStrengthRetention {
        endpoint_id: String,
        estimand: String,
        quantity_unit: String,
        formula_id: String,
        formula_sha256: String,
        design_id: String,
        plan_sha256: String,
        frozen_at: i64,
        first_outcome_available_at: i64,
        minimum_independent_clusters: usize,
        baseline: RetentionArm,
        candidate: RetentionArm,
        result: RetentionResult,
    },
    FailureModeFraction {
        endpoint_id: String,
        estimand: String,
        quantity_unit: String,
        taxonomy_id: String,
        taxonomy_sha256: String,
        category_id: String,
        denominator_policy: String,
        grouping_scheme: String,
        design_id: String,
        plan_sha256: String,
        frozen_at: i64,
        first_outcome_available_at: i64,
        minimum_independent_clusters: usize,
        baseline_scope_evidence: ScopeEvidence,
        candidate_scope_evidence: ScopeEvidence,
        baseline: Vec<FailureClassification>,
        candidate: Vec<FailureClassification>,
        result: FailureModeResult,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    payload: EndpointPayload,
    expected_disposition: String,
    expected_outcome: String,
    rationale: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
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

struct Decision {
    disposition: Disposition,
    outcome: &'static str,
    reason: &'static str,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn sha256_shape(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn near(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9
}

fn samples_valid(samples: &[Sample]) -> bool {
    let mut seen = HashSet::new();
    samples.iter().all(|sample| {
        nonempty(&sample.specimen_id)
            && nonempty(&sample.cluster_id)
            && seen.insert(sample.specimen_id.as_str())
    })
}

fn specimen_ids(samples: &[Sample]) -> HashSet<&str> {
    samples.iter().map(|sample| sample.specimen_id.as_str()).collect()
}

fn cluster_ids(samples: &[Sample]) -> HashSet<&str> {
    samples.iter().map(|sample| sample.cluster_id.as_str()).collect()
}

fn same_set(left: &HashSet<&str>, right: &HashSet<&str>) -> bool {
    left == right
}

fn validate_scope(evidence: &ScopeEvidence) -> std::result::Result<(), Disposition> {
    match decide(evidence).disposition {
        ClaimDisposition::Supported => Ok(()),
        ClaimDisposition::Blocked => Err(Disposition::Blocked),
        ClaimDisposition::Rejected => Err(Disposition::Rejected),
    }
}

fn validate_strength_aggregate(
    aggregate: &StrengthAggregate,
    require_conditioning: bool,
) -> std::result::Result<(), Disposition> {
    if !nonempty(&aggregate.receipt_id)
        || !sha256_shape(&aggregate.result_sha256)
        || !sha256_shape(&aggregate.sample_manifest_sha256)
    {
        return Err(Disposition::Blocked);
    }
    if !aggregate.mean_strength_mpa.is_finite() || aggregate.mean_strength_mpa <= 0.0 {
        return Err(Disposition::Rejected);
    }
    if !samples_valid(&aggregate.samples) || aggregate.samples.is_empty() {
        return Err(Disposition::Rejected);
    }
    let ids = specimen_ids(&aggregate.samples);
    let receipt_ids: HashSet<&str> = aggregate
        .scope_evidence
        .specimen_ids
        .iter()
        .map(String::as_str)
        .collect();
    if ids != receipt_ids {
        return Err(Disposition::Rejected);
    }
    if aggregate.scope_evidence.endpoint != "apparent_shear_strength"
        || aggregate.scope_evidence.requested_claim != "comparative_screening"
    {
        return Err(Disposition::Rejected);
    }
    validate_scope(&aggregate.scope_evidence)?;

    match (&aggregate.conditioning_receipt, require_conditioning) {
        (Some(conditioning), true) => {
            if aggregate.conditioning_profile_id.as_deref().is_none_or(str::is_empty)
                || !aggregate.conditioning_profile_sha256.as_deref().is_some_and(sha256_shape)
                || conditioning.method_id != "ASTM_D1151"
                || conditioning.endpoint != "moisture_temperature_exposure_protocol"
                || conditioning.requested_claim != "conditioning_protocol_reference"
                || specimen_ids(&aggregate.samples)
                    != conditioning.specimen_ids.iter().map(String::as_str).collect()
            {
                return Err(Disposition::Rejected);
            }
            validate_scope(conditioning)?;
        }
        (None, true) => return Err(Disposition::Blocked),
        (Some(_), false) => return Err(Disposition::Rejected),
        (None, false) => {
            if aggregate.conditioning_profile_id.is_some() || aggregate.conditioning_profile_sha256.is_some() {
                return Err(Disposition::Rejected);
            }
        }
    }
    Ok(())
}

fn validate_retention_arm(
    arm: &RetentionArm,
    minimum_clusters: usize,
) -> std::result::Result<f64, Disposition> {
    if !sha256_shape(&arm.profile_sha256) {
        return Err(Disposition::Blocked);
    }
    validate_strength_aggregate(&arm.initial, false)?;
    validate_strength_aggregate(&arm.aged, true)?;

    let initial_ids = specimen_ids(&arm.initial.samples);
    let aged_ids = specimen_ids(&arm.aged.samples);
    if !initial_ids.is_disjoint(&aged_ids) {
        return Err(Disposition::Rejected);
    }

    let initial_clusters = cluster_ids(&arm.initial.samples);
    let aged_clusters = cluster_ids(&arm.aged.samples);
    if !same_set(&initial_clusters, &aged_clusters) {
        return Err(Disposition::Rejected);
    }
    if initial_clusters.len() < minimum_clusters {
        return Err(Disposition::Blocked);
    }

    Ok(arm.aged.mean_strength_mpa / arm.initial.mean_strength_mpa)
}

fn audit_direct(receipt: &DirectObservation) -> Decision {
    if !nonempty(&receipt.endpoint_id)
        || !nonempty(&receipt.quantity_unit)
        || !sha256_shape(&receipt.result_sha256)
    {
        return Decision {
            disposition: Disposition::Blocked,
            outcome: "NOT_EVALUATED",
            reason: "direct observation receipt identity, unit or result digest is missing",
        };
    }
    if receipt.endpoint_id != receipt.scope_evidence.endpoint
        || !receipt.value.is_finite()
    {
        return Decision {
            disposition: Disposition::Rejected,
            outcome: "NOT_EVALUATED",
            reason: "direct endpoint/value does not match the exact measured method endpoint",
        };
    }
    match validate_scope(&receipt.scope_evidence) {
        Ok(()) => Decision {
            disposition: Disposition::Pass,
            outcome: "DIRECT_OBSERVATION_VALID",
            reason: "direct endpoint matches its exact method/edition/material/geometry scope",
        },
        Err(Disposition::Blocked) => Decision {
            disposition: Disposition::Blocked,
            outcome: "NOT_EVALUATED",
            reason: "direct observation is missing required method-scope evidence",
        },
        Err(Disposition::Rejected) => Decision {
            disposition: Disposition::Rejected,
            outcome: "NOT_EVALUATED",
            reason: "direct observation exceeds or contradicts its method scope",
        },
        Err(Disposition::Pass) => unreachable!(),
    }
}

fn audit_retention(
    endpoint_id: &str,
    estimand: &str,
    quantity_unit: &str,
    formula_id: &str,
    formula_sha256: &str,
    design_id: &str,
    plan_sha256: &str,
    frozen_at: i64,
    first_outcome_available_at: i64,
    minimum_independent_clusters: usize,
    baseline: &RetentionArm,
    candidate: &RetentionArm,
    result: &RetentionResult,
) -> Decision {
    let blocked = |reason| Decision { disposition: Disposition::Blocked, outcome: "NOT_EVALUATED", reason };
    let rejected = |reason| Decision { disposition: Disposition::Rejected, outcome: "NOT_EVALUATED", reason };

    if !nonempty(endpoint_id) || !nonempty(estimand) || !nonempty(quantity_unit)
        || !nonempty(design_id) || !nonempty(result.result_id)
    {
        return blocked("derived retention endpoint identity or unit is missing");
    }
    if endpoint_id != "aged_strength_retention"
        || estimand != "candidate_minus_baseline_ratio_of_means"
        || quantity_unit != "dimensionless"
        || formula_id != "ratio_of_means_v1"
    {
        return rejected("retention endpoint is not the registered ratio-of-means estimand");
    }
    if !sha256_shape(formula_sha256) || !sha256_shape(plan_sha256)
        || !sha256_shape(&result.result_sha256)
        || !sha256_shape(&result.plan_sha256)
        || !sha256_shape(&result.formula_sha256)
        || !sha256_shape(&result.input_manifest_sha256)
        || !sha256_shape(&input_manifest_sha256)
        || !sha256_shape(&uncertainty_receipt_sha256)
        || !sha256_shape(&result.uncertainty_receipt_sha256)
    {
        return blocked("retention formula, plan, input-manifest or result digest is missing/malformed");
    }
    if frozen_at >= first_outcome_available_at {
        return rejected("retention derivation design was not frozen before outcome availability");
    }
    if minimum_independent_clusters == 0 {
        return rejected("minimum independent-cluster count must be positive");
    }
    if result.design_id != design_id || result.plan_sha256 != plan_sha256
        || result.formula_sha256 != formula_sha256
        || result.input_manifest_sha256 != input_manifest_sha256
        || result.uncertainty_receipt_sha256 != uncertainty_receipt_sha256
        || result.analysis_profile != uncertainty_method_id
        || uncertainty_method_id != "cluster_bootstrap_diff_of_retention_v1"
    {
        return rejected("retention result is not bound to the frozen design/formula receipt");
    }

    let baseline_ratio = match validate_retention_arm(baseline, minimum_independent_clusters) {
        Ok(value) => value,
        Err(Disposition::Blocked) => return blocked("baseline initial/aged measurements, conditioning or cluster evidence is incomplete"),
        Err(Disposition::Rejected) => return rejected("baseline retention input method, destructive cohorts or cluster pairing is contradictory"),
        Err(Disposition::Pass) => unreachable!(),
    };
    let candidate_ratio = match validate_retention_arm(candidate, minimum_independent_clusters) {
        Ok(value) => value,
        Err(Disposition::Blocked) => return blocked("candidate initial/aged measurements, conditioning or cluster evidence is incomplete"),
        Err(Disposition::Rejected) => return rejected("candidate retention input method, destructive cohorts or cluster pairing is contradictory"),
        Err(Disposition::Pass) => unreachable!(),
    };

    let baseline_clusters = cluster_ids(&baseline.initial.samples);
    let candidate_clusters = cluster_ids(&candidate.initial.samples);
    let baseline_specimens: HashSet<&str> = specimen_ids(&baseline.initial.samples)
        .union(&specimen_ids(&baseline.aged.samples)).copied().collect();
    let candidate_specimens: HashSet<&str> = specimen_ids(&candidate.initial.samples)
        .union(&specimen_ids(&candidate.aged.samples)).copied().collect();
    if !baseline_clusters.is_disjoint(&candidate_clusters)
        || !baseline_specimens.is_disjoint(&candidate_specimens)
    {
        return rejected("baseline and candidate retention cohorts reuse a specimen or independent manufacturing cluster");
    }

    if baseline.initial.scope_evidence.method_id != baseline.aged.scope_evidence.method_id
        || baseline.initial.scope_evidence.edition != baseline.aged.scope_evidence.edition
        || baseline.initial.scope_evidence.protocol_id != baseline.aged.scope_evidence.protocol_id
        || baseline.initial.scope_evidence.protocol_sha256 != baseline.aged.scope_evidence.protocol_sha256
        || baseline.aged.conditioning_profile_id != candidate.aged.conditioning_profile_id
        || baseline.aged.conditioning_profile_sha256 != candidate.aged.conditioning_profile_sha256
        || baseline.initial.scope_evidence.method_id != candidate.initial.scope_evidence.method_id
        || baseline.initial.scope_evidence.edition != candidate.initial.scope_evidence.edition
        || baseline.initial.scope_evidence.material_pair_class != candidate.initial.scope_evidence.material_pair_class
        || baseline.initial.scope_evidence.geometry != candidate.initial.scope_evidence.geometry
        || baseline.aged.scope_evidence.method_id != candidate.aged.scope_evidence.method_id
        || baseline.aged.scope_evidence.edition != candidate.aged.scope_evidence.edition
        || baseline.aged.scope_evidence.material_pair_class != candidate.aged.scope_evidence.material_pair_class
        || baseline.aged.scope_evidence.geometry != candidate.aged.scope_evidence.geometry
        || candidate.initial.scope_evidence.protocol_id != candidate.aged.scope_evidence.protocol_id
        || candidate.initial.scope_evidence.protocol_sha256 != candidate.aged.scope_evidence.protocol_sha256
    {
        return rejected("baseline/candidate strength observations do not share the same method and joint scope");
    }

    if !near(result.baseline_retention, baseline_ratio)
        || !near(result.candidate_retention, candidate_ratio)
        || !near(result.candidate_minus_baseline, candidate_ratio - baseline_ratio)
    {
        return rejected("reported retention ratios or difference do not reconcile to the declared ratio-of-means inputs");
    }
    if !result.ci_lower.is_finite() || !result.ci_upper.is_finite()
        || result.ci_lower > result.ci_upper
        || !result.margin_lower.is_finite() || !result.margin_upper.is_finite()
        || result.margin_lower >= result.margin_upper
    {
        return rejected("retention interval or equivalence margins are invalid");
    }
    let equivalent = result.ci_lower >= result.margin_lower && result.ci_upper <= result.margin_upper;
    let outcome = if equivalent { "EQUIVALENCE_ESTABLISHED" } else { "EQUIVALENCE_NOT_ESTABLISHED" };
    if result.outcome != outcome {
        return rejected("retention equivalence outcome does not match the interval and frozen margin");
    }

    Decision {
        disposition: Disposition::Pass,
        outcome,
        reason: "ratio-of-means endpoint is structurally bound to destructive initial/aged cohorts, conditioning and cluster lineage; interval validity is not independently certified",
    }
}

fn validate_failure_arm(
    samples: &[FailureClassification],
    evidence: &ScopeEvidence,
    minimum_clusters: usize,
) -> std::result::Result<(usize, usize), Disposition> {
    if samples.is_empty() || !nonempty(&evidence.id) {
        return Err(Disposition::Blocked);
    }
    if evidence.method_id != "ASTM_D5573" || evidence.edition != "D5573-99R19" {
        return Err(Disposition::Blocked);
    }
    if evidence.material_pair_class != "frp_smc_joint" {
        return Err(Disposition::Blocked);
    }
    if evidence.geometry != "failure_surface_classification"
        || evidence.endpoint != "frp_joint_failure_mode_classification"
        || evidence.requested_claim != "failure_mode_classification_reference"
    {
        return Err(Disposition::Rejected);
    }
    let ids: Vec<String> = samples.iter().map(|sample| sample.specimen_id.clone()).collect();
    if !unique_strings(&ids) || evidence.specimen_ids.iter().collect::<HashSet<_>>()
        != ids.iter().collect::<HashSet<_>>()
    {
        return Err(Disposition::Rejected);
    }
    validate_scope(evidence)?;

    let allowed = [
        "adhesive_failure",
        "cohesive_failure",
        "thin_layer_cohesive_failure",
        "fiber_tear_failure",
        "light_fiber_tear_failure",
        "stock_break_failure",
        "mixed_failure",
        "unknown_unclassifiable",
    ];
    let mut clusters = HashSet::new();
    let mut numerator = 0usize;
    for sample in samples {
        if !nonempty(&sample.specimen_id) || !nonempty(&sample.cluster_id)
            || sample.category_ids.is_empty() || !unique_strings(&sample.category_ids)
        {
            return Err(Disposition::Rejected);
        }
        if sample.category_ids.iter().any(|id| !allowed.contains(&id.as_str())) {
            return Err(Disposition::Rejected);
        }
        if sample.category_ids.contains(&"unknown_unclassifiable".to_owned())
            && sample.category_ids.len() != 1
        {
            return Err(Disposition::Rejected);
        }
        if sample.rater_count == 0 {
            return Err(Disposition::Blocked);
        }
        if sample.rater_count > 1 && !sample.adjudication_sha256.as_deref().is_some_and(sha256_shape) {
            return Err(Disposition::Blocked);
        }
        if sample.rater_count == 1 && sample.adjudication_sha256.is_some() {
            return Err(Disposition::Rejected);
        }
        if sample.category_ids.contains(&"adhesive_failure".to_owned()) {
            numerator += 1;
        }
        clusters.insert(sample.cluster_id.as_str());
    }
    if clusters.len() < minimum_clusters {
        return Err(Disposition::Blocked);
    }
    Ok((numerator, samples.len()))
}

fn unique_strings(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    values.iter().all(|value| nonempty(value) && seen.insert(value.as_str()))
}

fn audit_failure_fraction(
    endpoint_id: &str,
    estimand: &str,
    quantity_unit: &str,
    taxonomy_id: &str,
    taxonomy_sha256: &str,
    category_id: &str,
    denominator_policy: &str,
    grouping_scheme: &str,
    design_id: &str,
    plan_sha256: &str,
    frozen_at: i64,
    first_outcome_available_at: i64,
    minimum_independent_clusters: usize,
    baseline_scope_evidence: &ScopeEvidence,
    candidate_scope_evidence: &ScopeEvidence,
    baseline: &[FailureClassification],
    candidate: &[FailureClassification],
    result: &FailureModeResult,
) -> Decision {
    let blocked = |reason| Decision { disposition: Disposition::Blocked, outcome: "NOT_EVALUATED", reason };
    let rejected = |reason| Decision { disposition: Disposition::Rejected, outcome: "NOT_EVALUATED", reason };
    if !nonempty(endpoint_id) || !nonempty(estimand) || !nonempty(taxonomy_id)
        || !nonempty(design_id) || !nonempty(result.result_id)
    {
        return blocked("failure-mode endpoint/taxonomy/design identity is missing");
    }
    if endpoint_id != "failure_mode_fraction"
        || estimand != "candidate_minus_baseline_specimen_fraction"
        || quantity_unit != "fraction"
        || category_id != "adhesive_failure"
        || denominator_policy != "all_tested_specimens_including_unknown"
        || !matches!(grouping_scheme, "independent_arms" | "paired_by_cluster")
    {
        return rejected("failure-mode fraction estimand, category or denominator policy is not the registered descriptive contract");
    }
    if !sha256_shape(taxonomy_sha256) || !sha256_shape(plan_sha256)
        || !sha256_shape(&result.result_sha256) || !sha256_shape(&result.plan_sha256)
        || !sha256_shape(&result.taxonomy_sha256) || !sha256_shape(&result.input_manifest_sha256)
        || !sha256_shape(input_manifest_sha256)
        || !sha256_shape(uncertainty_receipt_sha256)
        || !sha256_shape(&result.uncertainty_receipt_sha256)
    {
        return blocked("failure taxonomy, plan, manifest or result digest is missing/malformed");
    }
    if frozen_at >= first_outcome_available_at {
        return rejected("failure-mode design was not frozen before outcome availability");
    }
    if minimum_independent_clusters == 0 {
        return rejected("minimum independent-cluster count must be positive");
    }
    if result.claim_ceiling != "descriptive_fraction_only" {
        return rejected("failure-mode classification fraction is promoted beyond its descriptive claim ceiling");
    }
    if result.design_id != design_id || result.plan_sha256 != plan_sha256
        || result.taxonomy_sha256 != taxonomy_sha256
        || result.input_manifest_sha256 != input_manifest_sha256
        || result.uncertainty_receipt_sha256 != uncertainty_receipt_sha256
        || result.analysis_profile != uncertainty_method_id
        || uncertainty_method_id != "cluster_bootstrap_diff_of_fractions_v1"
    {
        return rejected("failure-mode result is not bound to the frozen design and taxonomy");
    }

    let (base_num, base_den) = match validate_failure_arm(baseline, baseline_scope_evidence, minimum_independent_clusters) {
        Ok(value) => value,
        Err(Disposition::Blocked) => return blocked("baseline classifications, adjudication or independent clusters are incomplete"),
        Err(Disposition::Rejected) => return rejected("baseline classification evidence is outside D5573 scope or has invalid specimen/category lineage"),
        Err(Disposition::Pass) => unreachable!(),
    };
    let (cand_num, cand_den) = match validate_failure_arm(candidate, candidate_scope_evidence, minimum_independent_clusters) {
        Ok(value) => value,
        Err(Disposition::Blocked) => return blocked("candidate classifications, adjudication or independent clusters are incomplete"),
        Err(Disposition::Rejected) => return rejected("candidate classification evidence is outside D5573 scope or has invalid specimen/category lineage"),
        Err(Disposition::Pass) => unreachable!(),
    };

    let base_ids: HashSet<&str> = baseline.iter().map(|sample| sample.specimen_id.as_str()).collect();
    let candidate_ids: HashSet<&str> = candidate.iter().map(|sample| sample.specimen_id.as_str()).collect();
    let base_clusters: HashSet<&str> = baseline.iter().map(|sample| sample.cluster_id.as_str()).collect();
    let candidate_clusters: HashSet<&str> = candidate.iter().map(|sample| sample.cluster_id.as_str()).collect();
    if !base_ids.is_disjoint(&candidate_ids) {
        return rejected("baseline and candidate failure-classification specimens overlap");
    }
    match grouping_scheme {
        "independent_arms" if !base_clusters.is_disjoint(&candidate_clusters) => {
            return rejected("independent classification arms reuse a manufacturing cluster");
        }
        "paired_by_cluster" if !same_set(&base_clusters, &candidate_clusters) => {
            return rejected("paired classification arms do not share the exact cluster-pair set");
        }
        _ => {}
    }

    let base_fraction = base_num as f64 / base_den as f64;
    let candidate_fraction = cand_num as f64 / cand_den as f64;
    if result.baseline_numerator != base_num || result.baseline_denominator != base_den
        || result.candidate_numerator != cand_num || result.candidate_denominator != cand_den
        || !near(result.baseline_fraction, base_fraction)
        || !near(result.candidate_fraction, candidate_fraction)
        || !near(result.candidate_minus_baseline, candidate_fraction - base_fraction)
    {
        return rejected("failure-mode numerator, denominator or fraction does not reconcile to specimen-level categories");
    }
    if !result.ci_lower.is_finite() || !result.ci_upper.is_finite()
        || result.ci_lower > result.ci_upper
        || !result.margin_lower.is_finite() || !result.margin_upper.is_finite()
        || result.margin_lower >= result.margin_upper
        || result.ci_lower < -1.0 || result.ci_upper > 1.0
        || result.margin_lower < -1.0 || result.margin_upper > 1.0
    {
        return rejected("failure-mode fraction interval or margins are invalid");
    }
    let equivalent = result.ci_lower >= result.margin_lower && result.ci_upper <= result.margin_upper;
    let outcome = if equivalent { "EQUIVALENCE_ESTABLISHED" } else { "EQUIVALENCE_NOT_ESTABLISHED" };
    if result.outcome != outcome {
        return rejected("failure-mode equivalence outcome does not reconcile to the interval and margin");
    }

    Decision {
        disposition: Disposition::Pass,
        outcome,
        reason: "failure-mode fraction reconciles to specimen-level D5573 classifications with unknowns in the denominator; no acceptability decision is implied",
    }
}

fn audit_case(case: &Case) -> Decision {
    match &case.payload {
        EndpointPayload::DirectObservation { receipt } => audit_direct(receipt),
        EndpointPayload::AgedStrengthRetention {
            endpoint_id, estimand, quantity_unit, formula_id, formula_sha256, design_id, plan_sha256,
            frozen_at, first_outcome_available_at, minimum_independent_clusters, baseline, candidate, result,
        } => audit_retention(
            endpoint_id, estimand, quantity_unit, formula_id, formula_sha256, design_id, plan_sha256,
            *frozen_at, *first_outcome_available_at, *minimum_independent_clusters, baseline, candidate, result,
        ),
        EndpointPayload::FailureModeFraction {
            endpoint_id, estimand, quantity_unit, taxonomy_id, taxonomy_sha256, category_id,
            denominator_policy, grouping_scheme, design_id, plan_sha256, frozen_at,
            first_outcome_available_at, minimum_independent_clusters, input_manifest_sha256,
            uncertainty_method_id, uncertainty_receipt_sha256, baseline_scope_evidence,
            candidate_scope_evidence, baseline, candidate, result,
        } => audit_failure_fraction(
            endpoint_id, estimand, quantity_unit, taxonomy_id, taxonomy_sha256, category_id,
            denominator_policy, grouping_scheme, design_id, plan_sha256, *frozen_at,
            *first_outcome_available_at, *minimum_independent_clusters, baseline_scope_evidence,
            candidate_scope_evidence, baseline, candidate, result,
        ),
    }
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-endpoint-derivation-fixture-v0.1" {
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
        println!("{}: {} / {} — {}", case.id, decision.disposition.as_str(), decision.outcome, decision.reason);
        if decision.disposition.as_str() != case.expected_disposition {
            bail!("case {}: expected {}, got {}", case.id, case.expected_disposition, decision.disposition.as_str());
        }
        if decision.outcome != case.expected_outcome {
            bail!("case {}: expected outcome {}, got {}", case.id, case.expected_outcome, decision.outcome);
        }
        if !nonempty(&case.rationale) {
            bail!("case {} has an empty rationale", case.id);
        }
    }
    println!(
        "PLASMA-ENDPOINT-DERIVATION S0 — {} cases; PASS={} BLOCKED={} REJECTED={}",
        fixture.cases.len(), counts[0], counts[1], counts[2]
    );
    println!("Claim ceiling: typed endpoint lineage and deterministic count/arithmetic reconciliation only; no statistical or physical qualification.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_endpoint_derivation_fixture_v0_1.json");

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint derivation fixture");
        for case in &fixture.cases {
            let decision = audit_case(case);
            assert_eq!(decision.disposition.as_str(), case.expected_disposition, "{}", case.id);
            assert_eq!(decision.outcome, case.expected_outcome, "{}", case.id);
        }
    }

    #[test]
    fn aged_retention_uses_distinct_destructive_cohorts() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint derivation fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-aged-strength-retention").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Pass);
    }

    #[test]
    fn d5573_unknown_failure_modes_remain_in_denominator() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint derivation fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-failure-mode-fraction-includes-unknown").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Pass);
    }

    #[test]
    fn d5573_does_not_apply_to_polypropylene_joint() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint derivation fixture");
        let case = fixture.cases.iter().find(|case| case.id == "d5573-on-polypropylene-blocked").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn descriptive_failure_fraction_is_not_acceptance() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint derivation fixture");
        let case = fixture.cases.iter().find(|case| case.id == "failure-fraction-acceptability-escalation").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }
}
