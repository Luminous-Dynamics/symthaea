//! Typed derived-endpoint audit for aged bond retention and FRP failure-mode distributions.
//! S0 validates lineage, study design, denominators and claim ceilings. It does
//! not calculate statistics, classify images, infer failure causality or accept failures.
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ClaimDisposition, ScopeEvidence};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<DerivedCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DerivedCase {
    id: String,
    kind: String,
    expected_disposition: String,
    expected_claim: String,
    rationale: String,
    aged_retention: Option<AgedRetentionEvidence>,
    failure_mode_distribution: Option<FailureModeDistributionEvidence>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StudyDesign {
    design_id: String,
    design_sha256: String,
    analysis_plan_sha256: String,
    sample_manifest_sha256: String,
    analysis_profile: String,
    estimand: String,
    quantity_unit: String,
    grouping_scheme: String,
    independent_unit: String,
    minimum_independent_clusters: usize,
    maximum_specimen_attrition_rate: f64,
    maximum_cluster_attrition_rate: f64,
    confidence_level: f64,
    minimum_confidence_level: f64,
    frozen_at: i64,
    first_outcome_available_at: i64,
    changed_after_outcome: bool,
    time_source_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecimenUnit {
    specimen_id: String,
    independent_cluster_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Exclusion {
    arm: String,
    specimen_id: String,
    independent_cluster_id: String,
    reason_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConditioningEvidence {
    receipt: ScopeEvidence,
    exposure_profile_id: String,
    exposure_profile_sha256: String,
    raw_trace_sha256: String,
    duration_hours: f64,
    temperature_c: f64,
    relative_humidity_pct: f64,
    cycle_protocol_id: Option<String>,
    cycle_count: u32,
    start_time: i64,
    end_time: i64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DerivedRatioResult {
    result_id: String,
    result_sha256: String,
    derivation_code_id: String,
    derivation_code_sha256: String,
    derivation_definition_id: String,
    derivation_definition_sha256: String,
    input_manifest_sha256: String,
    analysis_profile: String,
    estimate: f64,
    lower_bound: f64,
    upper_bound: f64,
    minimum_acceptable_ratio: f64,
    outcome: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgedRetentionEvidence {
    endpoint_id: String,
    material_pair_class: String,
    geometry: String,
    baseline_receipt: ScopeEvidence,
    conditioned_receipt: ScopeEvidence,
    #[serde(default)]
    conditioning: Option<ConditioningEvidence>,
    requested_claim: String,
    design: StudyDesign,
    baseline_samples: Vec<SpecimenUnit>,
    conditioned_samples: Vec<SpecimenUnit>,
    exclusions: Vec<Exclusion>,
    result: DerivedRatioResult,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassificationAssignment {
    specimen_id: String,
    independent_cluster_id: String,
    category_id: String,
    confidence: f64,
    adjudication_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CategoryCount {
    category_id: String,
    count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DistributionFraction {
    category_id: String,
    observed_fraction: f64,
    lower_bound: f64,
    upper_bound: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FailureModeAnalysisResult {
    result_id: String,
    result_sha256: String,
    derivation_code_id: String,
    derivation_code_sha256: String,
    category_registry_id: String,
    category_registry_sha256: String,
    input_manifest_sha256: String,
    analysis_profile: String,
    fractions: Vec<DistributionFraction>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FailureModeDistributionEvidence {
    endpoint_id: String,
    material_pair_class: String,
    geometry: String,
    classification_receipt: ScopeEvidence,
    image_manifest_id: String,
    image_manifest_sha256: String,
    classifier_id: String,
    classifier_version: String,
    adjudication_policy_sha256: String,
    reviewer_authority_ref: String,
    category_registry_id: String,
    category_registry_sha256: String,
    category_ids: Vec<String>,
    mandatory_nonterminal_categories: Vec<String>,
    denominator_specimen_ids: Vec<String>,
    sample_manifest_sha256: String,
    minimum_assignment_confidence: f64,
    assignments: Vec<ClassificationAssignment>,
    category_counts: Vec<CategoryCount>,
    design: StudyDesign,
    exclusions: Vec<Exclusion>,
    result: FailureModeAnalysisResult,
    requested_claim: String,
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
    claim: &'static str,
    reason: &'static str,
}

impl Decision {
    fn blocked(reason: &'static str) -> Self {
        Self { disposition: Disposition::Blocked, claim: "NOT_ESTABLISHED", reason }
    }

    fn rejected(reason: &'static str) -> Self {
        Self { disposition: Disposition::Rejected, claim: "REJECTED_BY_CLAIM_CEILING", reason }
    }

    fn pass(claim: &'static str, reason: &'static str) -> Self {
        Self { disposition: Disposition::Pass, claim, reason }
    }
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn sha256_shape(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn unique_nonempty(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    values.iter().all(|value| nonempty(value) && seen.insert(value.as_str()))
}

fn validate_design(design: &StudyDesign, required_profile: &str) -> std::result::Result<(), Decision> {
    for value in [
        design.design_id.as_str(),
        design.analysis_profile.as_str(),
        design.estimand.as_str(),
        design.quantity_unit.as_str(),
        design.independent_unit.as_str(),
        design.time_source_ref.as_str(),
    ] {
        if !nonempty(value) {
            return Err(Decision::blocked("derived-endpoint study design is missing identity, estimand, unit or time-source reference"));
        }
    }
    if !sha256_shape(&design.design_sha256) || !sha256_shape(&design.analysis_plan_sha256) {
        return Err(Decision::blocked("design or analysis-plan digest is missing or malformed"));
    }
    if design.analysis_profile != required_profile {
        return Err(Decision::rejected("analysis profile is not registered for the declared derived endpoint"));
    }
    if design.grouping_scheme == "single_arm_clustered" && design.independent_unit == "paired_cluster" {
        return Err(Decision::rejected("single-cohort classification cannot declare paired clusters"));
    }
    if !matches!(design.grouping_scheme.as_str(), "independent_arms" | "paired_by_cluster" | "single_arm_clustered") {
        return Err(Decision::rejected("unsupported grouping scheme"));
    }
    if !matches!(
        design.independent_unit.as_str(),
        "material_lot" | "adhesive_batch" | "film_roll" | "production_day" | "production_line" | "manufacturing_site"
    ) {
        return Err(Decision::blocked("independent unit is not a registered manufacturing replication unit"));
    }
    if design.minimum_independent_clusters == 0 {
        return Err(Decision::rejected("minimum independent-cluster count must be positive"));
    }
    if !design.maximum_specimen_attrition_rate.is_finite()
        || !(0.0..1.0).contains(&design.maximum_specimen_attrition_rate)
        || !design.maximum_cluster_attrition_rate.is_finite()
        || !(0.0..1.0).contains(&design.maximum_cluster_attrition_rate)
    {
        return Err(Decision::rejected("specimen and cluster attrition allowances must be finite and in [0, 1)"));
    }
    if !design.confidence_level.is_finite()
        || !design.minimum_confidence_level.is_finite()
        || !(0.0..=1.0).contains(&design.minimum_confidence_level)
        || design.confidence_level < design.minimum_confidence_level
    {
        return Err(Decision::blocked("derived-endpoint confidence is below its preregistered minimum"));
    }
    if design.changed_after_outcome || design.frozen_at >= design.first_outcome_available_at {
        return Err(Decision::rejected("derived-endpoint analysis plan was changed or frozen after outcomes were available"));
    }
    Ok(())
}

fn validate_observation(
    receipt: &ScopeEvidence,
    expected_method: &str,
    expected_edition: &str,
    material_pair_class: &str,
    geometry: &str,
    endpoint: &str,
    claim: &str,
) -> std::result::Result<(), Decision> {
    match decide(receipt).disposition {
        ClaimDisposition::Supported => {}
        ClaimDisposition::Blocked => return Err(Decision::blocked("parent method observation is incomplete or outside the registered edition scope")),
        ClaimDisposition::Rejected => return Err(Decision::rejected("parent method observation exceeds its registered evidence or claim scope")),
    }
    if receipt.method_id != expected_method
        || receipt.edition != expected_edition
        || receipt.material_pair_class != material_pair_class
        || receipt.geometry != geometry
        || receipt.endpoint != endpoint
        || receipt.requested_claim != claim
    {
        return Err(Decision::rejected("parent observation does not match the exact method, edition, material, geometry, endpoint and claim required by the derivation"));
    }
    Ok(())
}

fn validate_sample_cohorts(
    baseline: &[SpecimenUnit],
    treated: &[SpecimenUnit],
    exclusions: &[Exclusion],
    design: &StudyDesign,
) -> std::result::Result<(), Decision> {
    let baseline_ids: Vec<String> = baseline.iter().map(|s| s.specimen_id.clone()).collect();
    let treated_ids: Vec<String> = treated.iter().map(|s| s.specimen_id.clone()).collect();
    if !unique_nonempty(&baseline_ids)
        || !unique_nonempty(&treated_ids)
        || baseline.iter().any(|s| !nonempty(&s.independent_cluster_id))
        || treated.iter().any(|s| !nonempty(&s.independent_cluster_id))
    {
        return Err(Decision::rejected("sample or independent-cluster identities are blank or duplicated"));
    }
    if baseline_ids.iter().any(|id| treated_ids.contains(id)) {
        return Err(Decision::rejected("baseline and derived-outcome specimen identities overlap"));
    }

    let baseline_clusters: HashSet<&str> = baseline.iter().map(|s| s.independent_cluster_id.as_str()).collect();
    let treated_clusters: HashSet<&str> = treated.iter().map(|s| s.independent_cluster_id.as_str()).collect();
    match design.grouping_scheme.as_str() {
        "independent_arms" => {
            if !baseline_clusters.is_disjoint(&treated_clusters) {
                return Err(Decision::rejected("independent baseline and outcome arms share manufacturing clusters"));
            }
        }
        "paired_by_cluster" => {
            if baseline_clusters.is_empty() || baseline_clusters != treated_clusters {
                return Err(Decision::rejected("paired derived-endpoint cohorts do not have exact matching independent clusters"));
            }
        }
        _ => unreachable!("grouping scheme was validated above"),
    }

    let mut exclusion_ids = HashSet::new();
    let mut excluded_baseline_specimens = HashSet::new();
    let mut excluded_conditioned_specimens = HashSet::new();
    let mut excluded_baseline_clusters = HashSet::new();
    let mut excluded_conditioned_clusters = HashSet::new();
    for exclusion in exclusions {
        if !matches!(exclusion.arm.as_str(), "baseline" | "conditioned")
            || !nonempty(&exclusion.specimen_id)
            || !nonempty(&exclusion.independent_cluster_id)
            || !nonempty(&exclusion.reason_ref)
        {
            return Err(Decision::blocked("excluded specimen lacks a valid cohort, cluster identity or exclusion reason"));
        }
        if !exclusion_ids.insert(exclusion.specimen_id.as_str())
            || baseline_ids.contains(&exclusion.specimen_id)
            || treated_ids.contains(&exclusion.specimen_id)
        {
            return Err(Decision::rejected("excluded specimen identity is duplicated or also appears in an analysis cohort"));
        }
        if exclusion.arm == "baseline" {
            excluded_baseline_specimens.insert(exclusion.specimen_id.as_str());
            excluded_baseline_clusters.insert(exclusion.independent_cluster_id.as_str());
        } else {
            excluded_conditioned_specimens.insert(exclusion.specimen_id.as_str());
            excluded_conditioned_clusters.insert(exclusion.independent_cluster_id.as_str());
        }
    }
    if design.grouping_scheme == "paired_by_cluster"
        && excluded_baseline_clusters != excluded_conditioned_clusters
    {
        return Err(Decision::rejected("paired cohort attrition does not remove the same independent clusters from baseline and conditioned arms"));
    }
    let baseline_specimen_attrition = excluded_baseline_specimens.len() as f64
        / (baseline.len() + excluded_baseline_specimens.len()).max(1) as f64;
    let conditioned_specimen_attrition = excluded_conditioned_specimens.len() as f64
        / (treated.len() + excluded_conditioned_specimens.len()).max(1) as f64;
    if baseline_specimen_attrition > design.maximum_specimen_attrition_rate
        || conditioned_specimen_attrition > design.maximum_specimen_attrition_rate
    {
        return Err(Decision::blocked("specimen attrition exceeds the preregistered aged-retention allowance"));
    }
    let baseline_all_clusters: HashSet<&str> = baseline_clusters
        .union(&excluded_baseline_clusters)
        .copied()
        .collect();
    let conditioned_all_clusters: HashSet<&str> = treated_clusters
        .union(&excluded_conditioned_clusters)
        .copied()
        .collect();
    let baseline_cluster_attrition = (baseline_all_clusters.len() - baseline_clusters.len()) as f64
        / baseline_all_clusters.len().max(1) as f64;
    let conditioned_cluster_attrition = (conditioned_all_clusters.len() - treated_clusters.len()) as f64
        / conditioned_all_clusters.len().max(1) as f64;
    if baseline_cluster_attrition > design.maximum_cluster_attrition_rate
        || conditioned_cluster_attrition > design.maximum_cluster_attrition_rate
    {
        return Err(Decision::blocked("independent-cluster attrition exceeds the preregistered aged-retention allowance"));
    }

    let independent_count = match design.grouping_scheme.as_str() {
        "independent_arms" => baseline_clusters.len().min(treated_clusters.len()),
        "paired_by_cluster" => baseline_clusters.len(),
        _ => 0,
    };
    if independent_count < design.minimum_independent_clusters {
        return Err(Decision::blocked("actual independent manufacturing clusters are below the preregistered minimum"));
    }
    Ok(())
}

fn validate_conditioning(
    conditioning: &ConditioningEvidence,
    expected_specimens: &[String],
) -> std::result::Result<(), Decision> {
    let receipt = &conditioning.receipt;
    match decide(receipt).disposition {
        ClaimDisposition::Supported => {}
        ClaimDisposition::Blocked => return Err(Decision::blocked("conditioning-practice reference is incomplete or outside the registered edition scope")),
        ClaimDisposition::Rejected => return Err(Decision::rejected("conditioning evidence kind or claim exceeds the registered practice scope")),
    }
    if !matches!(
        (receipt.method_id.as_str(), receipt.edition.as_str(), receipt.endpoint.as_str()),
        ("ASTM_D1151", "D1151-00R22", "moisture_temperature_exposure_protocol")
            | ("ASTM_D1183", "D1183-03R19", "cyclic_temperature_humidity_exposure_protocol")
    ) {
        return Err(Decision::rejected("aged-retention derivation requires a registered D1151 or D1183 conditioning-practice receipt"));
    }
    if receipt.requested_claim != "conditioning_protocol_reference"
        || receipt.material_pair_class != "adhesive_bonded_specimens"
        || receipt.geometry != if receipt.method_id == "ASTM_D1151" { "continuous_environmental_conditioning" } else { "cyclic_environmental_conditioning" }
    {
        return Err(Decision::rejected("conditioning practice does not match the declared adhesive-bond specimen scope"));
    }
    if !nonempty(&conditioning.exposure_profile_id)
        || !sha256_shape(&conditioning.exposure_profile_sha256)
        || !sha256_shape(&conditioning.raw_trace_sha256)
        || !conditioning.duration_hours.is_finite()
        || conditioning.duration_hours <= 0.0
        || !conditioning.temperature_c.is_finite()
        || !conditioning.relative_humidity_pct.is_finite()
        || !(0.0..=100.0).contains(&conditioning.relative_humidity_pct)
        || conditioning.end_time <= conditioning.start_time
        || (receipt.method_id == "ASTM_D1183" && conditioning.cycle_count == 0)
        || (receipt.method_id == "ASTM_D1151" && conditioning.cycle_count > 0 && conditioning.cycle_protocol_id.is_none())
    {
        return Err(Decision::blocked("conditioning exposure profile, raw trace, duration, environment or time interval is incomplete"));
    }
    if receipt.method_id == "ASTM_D1183"
        && (conditioning.cycle_count == 0 || conditioning.cycle_protocol_id.as_deref().is_none_or(str::is_empty))
    {
        return Err(Decision::blocked("D1183 cyclic conditioning requires a positive cycle count and an explicit cycle-protocol identity"));
    }
    if receipt.method_id == "ASTM_D1151" && conditioning.cycle_count != 0 {
        return Err(Decision::rejected("D1151 continuous conditioning cannot be represented as a cyclic exposure"));
    }
    let specimen_ids = receipt.specimen_ids.clone();
    if !unique_nonempty(&specimen_ids)
        || specimen_ids.len() != expected_specimens.len()
        || !specimen_ids.iter().all(|id| expected_specimens.contains(id))
    {
        return Err(Decision::rejected("conditioning specimen set does not exactly match the post-conditioning measurement cohort"));
    }
    Ok(())
}

fn audit_aged_retention(evidence: &AgedRetentionEvidence) -> Decision {
    if evidence.endpoint_id != "aged_strength_retention_ratio" {
        return Decision::rejected("aged-retention result uses an unsupported endpoint identity");
    }
    if evidence.requested_claim != "aged_strength_retention" {
        return Decision::rejected("aged-retention derivation cannot be promoted to service life, time-to-failure or another unsupported claim");
    }
    if evidence.material_pair_class != "rigid_plastic_pair" || evidence.geometry != "single_lap" {
        return Decision::blocked("this S0 aged-retention profile is limited to rigid-plastic single-lap specimens");
    }

    let baseline = &evidence.baseline_receipt;
    let conditioned = &evidence.conditioned_receipt;
    for receipt in [baseline, conditioned] {
        if let Err(decision) = validate_observation(
            receipt,
            "ASTM_D3163",
            "D3163-01R23",
            &evidence.material_pair_class,
            &evidence.geometry,
            "apparent_shear_strength",
            "comparative_screening",
        ) {
            return decision;
        }
    }
    if baseline.id == conditioned.id
        || baseline.raw_data_sha256 == conditioned.raw_data_sha256
        || baseline.protocol_id != conditioned.protocol_id
        || baseline.protocol_sha256 != conditioned.protocol_sha256
    {
        return Decision::rejected("baseline and conditioned D3163 observations reuse identity/data or use different protocols");
    }

    if let Err(decision) = validate_design(&evidence.design, "paired_cluster_retention_ratio_v1") {
        return decision;
    }
    if evidence.design.grouping_scheme != "paired_by_cluster" {
        return Decision::rejected("the first aged-retention profile requires matched independent manufacturing clusters");
    }
    if let Err(decision) = validate_sample_cohorts(
        &evidence.baseline_samples,
        &evidence.conditioned_samples,
        &evidence.exclusions,
        &evidence.design,
    ) {
        return decision;
    }

    let baseline_ids: Vec<String> = evidence.baseline_samples.iter().map(|s| s.specimen_id.clone()).collect();
    let aged_ids: Vec<String> = evidence.conditioned_samples.iter().map(|s| s.specimen_id.clone()).collect();
    for (receipt, expected) in [(baseline, &baseline_ids), (conditioned, &aged_ids)] {
        if !unique_nonempty(&receipt.specimen_ids)
            || receipt.specimen_ids.len() != expected.len()
            || !receipt.specimen_ids.iter().all(|id| expected.contains(id))
        {
            return Decision::rejected("D3163 observation receipt does not exactly match the derived endpoint cohort manifest");
        }
    }

    let Some(conditioning) = evidence.conditioning.as_ref() else {
        return Decision::blocked("aged-strength retention requires an explicit D1151 or D1183 conditioning receipt");
    };
    if let Err(decision) = validate_conditioning(conditioning, &aged_ids) {
        return decision;
    }

    let result = &evidence.result;
    if ![
        result.result_id.as_str(),
        result.derivation_code_id.as_str(),
        result.derivation_definition_id.as_str(),
    ].iter().all(|v| nonempty(v))
        || ![
            result.result_sha256.as_str(),
            result.derivation_code_sha256.as_str(),
            result.derivation_definition_sha256.as_str(),
            result.input_manifest_sha256.as_str(),
        ].iter().all(|v| sha256_shape(v))
    {
        return Decision::blocked("derived-retention result, code, definition or input-manifest receipt is incomplete");
    }
    if result.analysis_profile != evidence.design.analysis_profile
        || result.input_manifest_sha256 != evidence.design.sample_manifest_sha256
    {
        return Decision::rejected("retention derivation is not bound to the preregistered design and input manifest");
    }
    if ![
        result.estimate,
        result.lower_bound,
        result.upper_bound,
        result.minimum_acceptable_ratio,
    ].iter().all(|v| v.is_finite())
        || result.minimum_acceptable_ratio <= 0.0
        || result.lower_bound < 0.0
        || result.upper_bound < result.lower_bound
        || result.estimate < result.lower_bound
        || result.estimate > result.upper_bound
    {
        return Decision::rejected("retention estimate or interval is non-finite, negative, reversed or inconsistent");
    }
    if result.lower_bound >= result.minimum_acceptable_ratio {
        return Decision::pass(
            "AGED_STRENGTH_RETENTION_ESTABLISHED",
            "D3163 baseline/conditioned observations and D1151/D1183 conditioning are linked to a preregistered matched-cluster retention derivation",
        );
    }
    if result.upper_bound < result.minimum_acceptable_ratio {
        return Decision::rejected("upper confidence bound is below the predeclared minimum aged-strength-retention ratio");
    }
    Decision::blocked("aged-strength-retention interval crosses the minimum acceptable ratio")
}

fn clusters_from_map<'a>(
    assignments: &'a HashMap<&'a str, &'a str>,
) -> HashSet<&'a str> {
    assignments.values().copied().collect()
}

fn audit_failure_mode_distribution(evidence: &FailureModeDistributionEvidence) -> Decision {
    if evidence.endpoint_id != "frp_failure_mode_distribution" {
        return Decision::rejected("failure-mode result uses an unsupported endpoint identity");
    }
    if evidence.requested_claim != "failure_mode_distribution" {
        return Decision::rejected("D5573 classification supports descriptive failure-mode distribution only, never acceptability or product disposition");
    }
    if evidence.material_pair_class != "frp_smc_joint" || evidence.geometry != "failure_surface_classification" {
        return Decision::blocked("this S0 failure-mode profile is limited to the registered SMC FRP classification scope");
    }
    if let Err(decision) = validate_observation(
        &evidence.classification_receipt,
        "ASTM_D5573",
        "D5573-99R19",
        "frp_smc_joint",
        "failure_surface_classification",
        "frp_joint_failure_mode_classification",
        "failure_mode_classification_reference",
    ) {
        return decision;
    }
    for value in [
        evidence.image_manifest_id.as_str(),
        evidence.classifier_id.as_str(),
        evidence.classifier_version.as_str(),
        evidence.reviewer_authority_ref.as_str(),
        evidence.category_registry_id.as_str(),
    ] {
        if !nonempty(value) {
            return Decision::blocked("failure-mode inspection, classifier, adjudication authority or category registry identity is missing");
        }
    }
    for digest in [
        evidence.image_manifest_sha256.as_str(),
        evidence.adjudication_policy_sha256.as_str(),
        evidence.category_registry_sha256.as_str(),
    ] {
        if !sha256_shape(digest) {
            return Decision::blocked("image, adjudication policy or category registry digest is missing or malformed");
        }
    }
    if !unique_nonempty(&evidence.category_ids)
        || !unique_nonempty(&evidence.mandatory_nonterminal_categories)
    {
        return Decision::rejected("failure-mode category IDs are blank or duplicated");
    }
    for required in ["mixed", "unclear", "invalid"] {
        if !evidence.mandatory_nonterminal_categories.iter().any(|v| v == required)
            || !evidence.category_ids.iter().any(|v| v == required)
        {
            return Decision::blocked("denominator taxonomy must explicitly account for mixed, unclear and invalid classifications");
        }
    }

    let denominator = &evidence.denominator_specimen_ids;
    if !unique_nonempty(denominator) || denominator.len() != evidence.classification_receipt.specimen_ids.len() {
        return Decision::rejected("failure-mode denominator is empty, duplicated or differs from the D5573 classification specimen manifest");
    }
    if !denominator.iter().all(|id| evidence.classification_receipt.specimen_ids.contains(id)) {
        return Decision::rejected("D5573 classification receipt does not cover the complete declared denominator");
    }
    if !sha256_shape(&evidence.sample_manifest_sha256)
        || !evidence.minimum_assignment_confidence.is_finite()
        || !(0.0..=1.0).contains(&evidence.minimum_assignment_confidence)
    {
        return Decision::blocked("failure-mode sample manifest or minimum classification-confidence threshold is incomplete");
    }
    if evidence.classification_receipt.raw_data_sha256.as_deref()
        != Some(evidence.image_manifest_sha256.as_str())
    {
        return Decision::rejected("D5573 classification receipt does not bind the exact image/inspection manifest");
    }
    if let Err(decision) = validate_design(&evidence.design, "clustered_failure_mode_distribution_v1") {
        return decision;
    }
    if evidence.design.grouping_scheme != "single_arm_clustered" {
        return Decision::rejected("failure-mode distribution requires a single-cohort clustered design, not a two-arm contrast");
    }
    if evidence.design.sample_manifest_sha256 != evidence.sample_manifest_sha256 {
        return Decision::rejected("failure-mode study design and classification evidence do not bind the same denominator manifest");
    }

    let mut assignments = HashMap::<&str, &str>::new();
    let mut assignment_clusters = HashMap::<&str, &str>::new();
    let mut observed_counts = HashMap::<&str, usize>::new();
    for assignment in &evidence.assignments {
        if !nonempty(&assignment.specimen_id)
            || !nonempty(&assignment.independent_cluster_id)
            || !nonempty(&assignment.category_id)
            || !nonempty(&assignment.adjudication_ref)
            || !assignment.confidence.is_finite()
            || !(0.0..=1.0).contains(&assignment.confidence)
        {
            return Decision::blocked("classification assignment has incomplete identity, category, adjudication or confidence");
        }
        if !denominator.iter().any(|id| id == &assignment.specimen_id)
            || !evidence.category_ids.iter().any(|id| id == &assignment.category_id)
        {
            return Decision::rejected("classification assignment refers to a specimen outside the denominator or an unknown category");
        }
        if assignment.confidence < evidence.minimum_assignment_confidence
            && assignment.category_id != "unclear"
            && assignment.category_id != "invalid"
        {
            return Decision::rejected("low-confidence classification must remain in the explicit unclear or invalid denominator category");
        }
        if assignments.insert(&assignment.specimen_id, &assignment.category_id).is_some() {
            return Decision::rejected("a specimen has multiple top-level failure-mode assignments; mixed cases must be explicitly represented in their category");
        }
        assignment_clusters.insert(&assignment.specimen_id, &assignment.independent_cluster_id);
        *observed_counts.entry(&assignment.category_id).or_insert(0) += 1;
    }
    if assignments.len() != denominator.len() {
        return Decision::blocked("classification denominator contains specimens without a final classified, mixed, unclear or invalid assignment");
    }
    let mut exclusion_ids = HashSet::new();
    let mut exclusion_clusters = HashSet::new();
    for exclusion in &evidence.exclusions {
        if exclusion.arm != "cohort"
            || !nonempty(&exclusion.specimen_id)
            || !nonempty(&exclusion.independent_cluster_id)
            || !nonempty(&exclusion.reason_ref)
        {
            return Decision::blocked("failure-mode exclusion lacks cohort identity, cluster or reason");
        }
        if !exclusion_ids.insert(exclusion.specimen_id.as_str())
            || denominator.contains(&exclusion.specimen_id)
        {
            return Decision::rejected("failure-mode exclusion identity is duplicated or also included in the denominator");
        }
        exclusion_clusters.insert(exclusion.independent_cluster_id.as_str());
    }
    let specimen_attrition = exclusion_ids.len() as f64
        / (denominator.len() + exclusion_ids.len()).max(1) as f64;
    if specimen_attrition > evidence.design.maximum_specimen_attrition_rate {
        return Decision::blocked("failure-mode specimen attrition exceeds the preregistered allowance");
    }
    let all_clusters: HashSet<&str> = clusters_from_map(&assignment_clusters)
        .union(&exclusion_clusters)
        .copied()
        .collect();
    let included_cluster_count = clusters_for_assignments(&assignment_clusters).len();
    let cluster_attrition = (all_clusters.len() - included_cluster_count) as f64
        / all_clusters.len().max(1) as f64;
    if cluster_attrition > evidence.design.maximum_cluster_attrition_rate {
        return Decision::blocked("failure-mode cluster attrition exceeds the preregistered allowance");
    }

    let mut declared_counts = HashMap::<&str, usize>::new();
    for category in &evidence.category_counts {
        if !evidence.category_ids.iter().any(|id| id == &category.category_id)
            || declared_counts.insert(&category.category_id, category.count).is_some()
        {
            return Decision::rejected("category count uses an unknown category or duplicate category entry");
        }
    }
    if declared_counts.len() != evidence.category_ids.len()
        || evidence.category_ids.iter().any(|id| !declared_counts.contains_key(id.as_str()))
    {
        return Decision::blocked("category count table omits a declared category, including nonterminal categories");
    }
    let total: usize = declared_counts.values().sum();
    if total != denominator.len()
        || evidence.category_ids.iter().any(|id| {
            declared_counts.get(id.as_str()).copied().unwrap_or(0)
                != observed_counts.get(id.as_str()).copied().unwrap_or(0)
        })
    {
        return Decision::rejected("category counts do not sum to and reproduce the complete specimen denominator");
    }

    let clusters: HashSet<&str> = assignment_clusters.values().copied().collect();
    if clusters.len() < evidence.design.minimum_independent_clusters {
        return Decision::blocked("failure-mode distribution has fewer independent manufacturing clusters than preregistered");
    }

    let result = &evidence.result;
    if !nonempty(&result.result_id)
        || !nonempty(&result.derivation_code_id)
        || !sha256_shape(&result.result_sha256)
        || !sha256_shape(&result.derivation_code_sha256)
        || !sha256_shape(&result.category_registry_sha256)
        || !sha256_shape(&result.input_manifest_sha256)
    {
        return Decision::blocked("failure-mode distribution result or derivation provenance is incomplete");
    }
    if result.category_registry_id != evidence.category_registry_id
        || result.category_registry_sha256 != evidence.category_registry_sha256
        || result.input_manifest_sha256 != evidence.classification_receipt.raw_data_sha256.as_deref().unwrap_or("")
        || result.analysis_profile != evidence.design.analysis_profile
    {
        return Decision::rejected("failure-mode distribution result is not bound to the exact category registry, D5573 source and study plan");
    }
    if result.fractions.len() != evidence.category_ids.len() {
        return Decision::blocked("failure-mode result does not report every declared category");
    }
    let mut fraction_ids = HashSet::new();
    let mut fraction_sum = 0.0;
    for fraction in &result.fractions {
        if !evidence.category_ids.iter().any(|id| id == &fraction.category_id)
            || !fraction_ids.insert(fraction.category_id.as_str())
        {
            return Decision::rejected("failure-mode result has an unknown or duplicated category fraction");
        }
        if !fraction.observed_fraction.is_finite()
            || !fraction.lower_bound.is_finite()
            || !fraction.upper_bound.is_finite()
            || fraction.lower_bound < 0.0
            || fraction.upper_bound > 1.0
            || fraction.lower_bound > fraction.observed_fraction
            || fraction.observed_fraction > fraction.upper_bound
        {
            return Decision::rejected("failure-mode fraction or interval is non-finite or inconsistent");
        }
        let expected = observed_counts.get(fraction.category_id.as_str()).copied().unwrap_or(0) as f64
            / denominator.len() as f64;
        if (fraction.observed_fraction - expected).abs() > 1e-9 {
            return Decision::rejected("reported observed fraction does not reproduce the complete classified denominator");
        }
        fraction_sum += fraction.observed_fraction;
    }
    if fraction_ids.len() != evidence.category_ids.len() || (fraction_sum - 1.0).abs() > 1e-9 {
        return Decision::rejected("failure-mode fractions do not form a complete descriptive composition summing to one");
    }

    Decision::pass(
        "DESCRIPTIVE_FAILURE_MODE_DISTRIBUTION",
        "D5573 classifications, inspection/adjudication evidence and the complete specimen denominator are linked to a preregistered descriptive distribution; no acceptability inference is made",
    )
}

fn audit_case(case: &DerivedCase) -> Decision {
    match case.kind.as_str() {
        "aged_strength_retention" => {
            if case.failure_mode_distribution.is_some() || case.aged_retention.is_none() {
                return Decision::rejected("case must contain exactly one aged-retention evidence object");
            }
            audit_aged_retention(case.aged_retention.as_ref().expect("checked above"))
        }
        "failure_mode_distribution" => {
            if case.aged_retention.is_some() || case.failure_mode_distribution.is_none() {
                return Decision::rejected("case must contain exactly one failure-mode distribution evidence object");
            }
            audit_failure_mode_distribution(case.failure_mode_distribution.as_ref().expect("checked above"))
        }
        _ => Decision::rejected("unknown derived endpoint kind"),
    }
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-derived-endpoints-fixture-v0.1" {
        bail!("unsupported derived-endpoints fixture schema: {}", fixture.schema_version);
    }
    let mut counts = [0usize; 3];
    let mut ids = HashSet::new();
    for case in &fixture.cases {
        if !ids.insert(case.id.as_str()) {
            bail!("duplicate derived-endpoint case identity: {}", case.id);
        }
        let decision = audit_case(case);
        let index = match decision.disposition {
            Disposition::Pass => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        println!("{}: {} / {} — {}", case.id, decision.disposition.as_str(), decision.claim, decision.reason);
        if decision.disposition.as_str() != case.expected_disposition {
            bail!("case {}: expected {}, got {}", case.id, case.expected_disposition, decision.disposition.as_str());
        }
        if !nonempty(&case.rationale) {
            bail!("case {} has an empty rationale", case.id);
        }
    }
    println!("PLASMA-DERIVED-ENDPOINTS S0 — {} cases; PASS={} BLOCKED={} REJECTED={}", fixture.cases.len(), counts[0], counts[1], counts[2]);
    println!("Aged retention is not service life. Failure-mode classification is descriptive, not acceptability.");
    println!("Claim ceiling: typed derivation lineage only; no independent statistical or physical validation is claimed.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../data/research/plasma_derived_endpoints_fixture_v0_1.json");

    fn fixture() -> Fixture {
        serde_json::from_str(FIXTURE).expect("valid derived-endpoint fixture")
    }

    #[test]
    fn frozen_fixture_matches_all_dispositions_and_claims() {
        let fixture = fixture();
        for case in &fixture.cases {
            let decision = audit_case(case);
            assert_eq!(decision.disposition.as_str(), case.expected_disposition, "{}", case.id);
            assert_eq!(decision.claim, case.expected_claim, "{}", case.id);
        }
    }

    #[test]
    fn lap_shear_without_conditioning_cannot_be_aged_retention() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "aged-strength-without-conditioning").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn retention_requires_exact_baseline_and_conditioned_cohorts() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "aged-retention-cohort-mismatch").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }

    #[test]
    fn conditioning_protocol_must_bind_exact_aged_cohort() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "conditioning-cohort-mismatch").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }

    #[test]
    fn failure_mode_distribution_counts_low_confidence_as_unclear() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "low-confidence-classification-not-unclear").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }

    #[test]
    fn d5573_distribution_requires_complete_nonterminal_denominator() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "failure-mode-denominator-omits-unclear").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn failure_mode_distribution_cannot_be_promoted_to_acceptability() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "failure-mode-acceptability-overclaim").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }

    #[test]
    fn survival_claim_cannot_be_inferred_from_retention_ratio() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "aged-retention-to-service-life-overclaim").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }
}
