//! Constrained substrate-integrity audit for plasma surface activation.
//! S0 checks declared hard limits and evidence lineage; it does not execute
//! measurement methods, calculate confidence intervals, or authorize production.
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ClaimDisposition, ScopeEvidence};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<SurfaceIntegrityCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SubstrateProfile {
    substrate_grade: String,
    supplier_id: String,
    lot_id: String,
    product_function: String,
    thickness_mm: f64,
    surface_finish: String,
    profile_sha256: String,
    constraint_plan_sha256: String,
    minimum_confidence_level: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExposureProfile {
    plasma_article_id: String,
    plasma_head_id: String,
    gas_species: String,
    gas_purity_pct: f64,
    delivered_power_w: f64,
    gas_flow_l_min: f64,
    standoff_mm: f64,
    line_speed_mm_s: f64,
    pass_count: u32,
    treatment_to_bond_min: f64,
    maximum_treatment_to_bond_min: f64,
    exposure_profile_sha256: String,
    raw_trace_sha256: String,
    time_source_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SampleUnit {
    specimen_id: String,
    independent_cluster_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RatioAnalysis {
    analysis_id: String,
    analysis_profile: String,
    plan_sha256: String,
    constraint_plan_sha256: String,
    sample_manifest_sha256: String,
    result_sha256: String,
    substrate_profile_sha256: String,
    exposure_profile_sha256: String,
    time_source_ref: String,
    grouping_scheme: String,
    independent_unit: String,
    minimum_independent_clusters: usize,
    minimum_confidence_level: f64,
    confidence_level: f64,
    frozen_at: i64,
    first_outcome_available_at: i64,
    design_changed_after_outcome: bool,
    baseline_samples: Vec<SampleUnit>,
    treated_samples: Vec<SampleUnit>,
    result_limit_direction: String,
    result_limit_value: f64,
    ratio_lower: f64,
    ratio_upper: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegrityConstraint {
    constraint_id: String,
    endpoint: String,
    limit_direction: String,
    limit_value: f64,
    ratio_lower: f64,
    ratio_upper: f64,
    baseline_method_receipt: ScopeEvidence,
    treated_method_receipt: ScopeEvidence,
    analysis: RatioAnalysis,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceIntegrityCase {
    id: String,
    substrate: SubstrateProfile,
    exposure: ExposureProfile,
    integrity_constraints: Vec<IntegrityConstraint>,
    contact_angle_diagnostic: Option<ScopeEvidence>,
    claimed_bond_outcome_passed: bool,
    expected_disposition: String,
    rationale: String,
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
    reason: &'static str,
    eligible: &'static str,
}

impl Decision {
    fn blocked(reason: &'static str) -> Self {
        Self { disposition: Disposition::Blocked, reason, eligible: "NOT_ESTABLISHED" }
    }

    fn rejected(reason: &'static str) -> Self {
        Self { disposition: Disposition::Rejected, reason, eligible: "REJECTED_BY_HARD_CONSTRAINT" }
    }

    fn pass() -> Self {
        Self {
            disposition: Disposition::Pass,
            reason: "all declared substrate-integrity hard constraints have method-scoped, preregistered, cluster-aware results within their application-specific limits",
            eligible: "ELIGIBLE_WITHIN_DECLARED_LIMITS",
        }
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

fn sample_manifest_is_valid(
    samples: &[SampleUnit],
    receipt: &ScopeEvidence,
) -> bool {
    let specimen_ids: Vec<String> = samples.iter().map(|sample| sample.specimen_id.clone()).collect();
    unique_nonempty(&specimen_ids)
        && samples.iter().all(|sample| nonempty(&sample.independent_cluster_id))
        && {
            let receipt_ids: Vec<String> = receipt.specimen_ids.clone();
            unique_nonempty(&receipt_ids)
                && receipt_ids.len() == specimen_ids.len()
                && receipt_ids.iter().all(|id| specimen_ids.contains(id))
        }
}

fn cluster_ids(samples: &[SampleUnit]) -> HashSet<&str> {
    samples.iter().map(|sample| sample.independent_cluster_id.as_str()).collect()
}

fn specimen_ids(samples: &[SampleUnit]) -> Vec<String> {
    samples.iter().map(|sample| sample.specimen_id.clone()).collect()
}

fn audit_method_pair(
    constraint: &IntegrityConstraint,
) -> std::result::Result<(), Decision> {
    for receipt in [&constraint.baseline_method_receipt, &constraint.treated_method_receipt] {
        match decide(receipt).disposition {
            ClaimDisposition::Supported => {}
            ClaimDisposition::Blocked => return Err(Decision::blocked("barrier or mechanical method receipt is incomplete or outside the registered edition map")),
            ClaimDisposition::Rejected => return Err(Decision::rejected("method receipt is incompatible with the registered evidence kind, specimen scope or endpoint")),
        }
    }

    let baseline = &constraint.baseline_method_receipt;
    let treated = &constraint.treated_method_receipt;
    if baseline.endpoint != constraint.endpoint || treated.endpoint != constraint.endpoint {
        return Err(Decision::rejected("constraint endpoint does not match both baseline and treated method receipts"));
    }
    if baseline.id == treated.id
        || baseline.raw_data_sha256 == treated.raw_data_sha256
        || baseline.method_id != treated.method_id
        || baseline.edition != treated.edition
        || baseline.evidence_kind != treated.evidence_kind
        || baseline.material_pair_class != treated.material_pair_class
        || baseline.geometry != treated.geometry
        || baseline.endpoint != treated.endpoint
        || baseline.requested_claim != treated.requested_claim
        || baseline.protocol_id != treated.protocol_id
        || baseline.protocol_sha256 != treated.protocol_sha256
    {
        return Err(Decision::rejected("baseline and treated measurements do not share one exact method/edition/protocol scope or reuse an observation identity"));
    }
    Ok(())
}

fn audit_constraint(
    constraint: &IntegrityConstraint,
    substrate: &SubstrateProfile,
    exposure: &ExposureProfile,
) -> std::result::Result<(), Decision> {
    if !nonempty(&constraint.constraint_id) || !nonempty(&constraint.endpoint) {
        return Err(Decision::blocked("constraint identity or endpoint is missing"));
    }
    if !matches!(constraint.limit_direction.as_str(), "maximum_ratio" | "minimum_ratio") {
        return Err(Decision::rejected("constraint direction must be an explicit maximum_ratio or minimum_ratio"));
    }
    if !constraint.limit_value.is_finite()
        || constraint.limit_value <= 0.0
        || !constraint.ratio_lower.is_finite()
        || !constraint.ratio_upper.is_finite()
        || constraint.ratio_lower < 0.0
        || constraint.ratio_upper < constraint.ratio_lower
    {
        return Err(Decision::rejected("hard limit or ratio interval is non-finite, negative or reversed"));
    }

    let analysis = &constraint.analysis;
    for value in [
        analysis.analysis_id.as_str(),
        analysis.analysis_profile.as_str(),
        analysis.time_source_ref.as_str(),
    ] {
        if !nonempty(value) {
            return Err(Decision::blocked("analysis identity, profile or trusted-time source reference is missing"));
        }
    }
    for digest in [
        analysis.plan_sha256.as_str(),
        analysis.constraint_plan_sha256.as_str(),
        analysis.sample_manifest_sha256.as_str(),
        analysis.result_sha256.as_str(),
        analysis.substrate_profile_sha256.as_str(),
        analysis.exposure_profile_sha256.as_str(),
    ] {
        if !sha256_shape(digest) {
            return Err(Decision::blocked("analysis, plan, sample, substrate or exposure digest is missing or malformed"));
        }
    }

    if analysis.substrate_profile_sha256 != substrate.profile_sha256
        || analysis.exposure_profile_sha256 != exposure.exposure_profile_sha256
        || analysis.constraint_plan_sha256 != substrate.constraint_plan_sha256
        || analysis.time_source_ref != exposure.time_source_ref
    {
        return Err(Decision::rejected("analysis receipt is bound to a different substrate, exposure, constraint plan or time-source reference"));
    }
    if analysis.design_changed_after_outcome || analysis.frozen_at >= analysis.first_outcome_available_at {
        return Err(Decision::rejected("surface-integrity plan was changed or frozen after outcomes became available"));
    }
    if !analysis.minimum_confidence_level.is_finite()
        || !(0.0..=1.0).contains(&analysis.minimum_confidence_level)
        || !analysis.confidence_level.is_finite()
        || analysis.confidence_level < analysis.minimum_confidence_level
    {
        return Err(Decision::blocked("analysis interval confidence is below the preregistered minimum"));
    }
    if analysis.result_limit_direction != constraint.limit_direction
        || (analysis.result_limit_value - constraint.limit_value).abs() > 1e-9
        || (analysis.ratio_lower - constraint.ratio_lower).abs() > 1e-9
        || (analysis.ratio_upper - constraint.ratio_upper).abs() > 1e-9
    {
        return Err(Decision::rejected("analysis result does not reproduce the predeclared hard limit and reported ratio interval"));
    }
    if !matches!(analysis.grouping_scheme.as_str(), "independent_arms" | "paired_by_cluster") {
        return Err(Decision::rejected("unsupported independent-unit grouping scheme"));
    }
    if !matches!(
        analysis.independent_unit.as_str(),
        "material_lot" | "film_roll" | "production_day" | "production_line" | "manufacturing_site"
    ) {
        return Err(Decision::blocked("independent unit is not a registered manufacturing group"));
    }
    if analysis.minimum_independent_clusters == 0 {
        return Err(Decision::rejected("minimum independent-cluster count must be positive"));
    }

    if !sample_manifest_is_valid(&analysis.baseline_samples, &constraint.baseline_method_receipt)
        || !sample_manifest_is_valid(&analysis.treated_samples, &constraint.treated_method_receipt)
    {
        return Err(Decision::rejected("analysis sample identities do not exactly match the method-qualified baseline and treated receipts"));
    }
    let baseline_specimens = specimen_ids(&analysis.baseline_samples);
    let treated_specimens = specimen_ids(&analysis.treated_samples);
    if baseline_specimens.iter().any(|id| treated_specimens.contains(id)) {
        return Err(Decision::rejected("baseline and treated specimen identities overlap"));
    }

    let baseline_clusters = cluster_ids(&analysis.baseline_samples);
    let treated_clusters = cluster_ids(&analysis.treated_samples);
    match analysis.grouping_scheme.as_str() {
        "independent_arms" => {
            if !baseline_clusters.is_disjoint(&treated_clusters) {
                return Err(Decision::rejected("independent arms share manufacturing clusters"));
            }
        }
        "paired_by_cluster" => {
            if baseline_clusters != treated_clusters || baseline_clusters.is_empty() {
                return Err(Decision::rejected("paired design does not have exact baseline/treated manufacturing-cluster pairs"));
            }
        }
        _ => unreachable!("grouping scheme was validated above"),
    }
    let effective_clusters = match analysis.grouping_scheme.as_str() {
        "independent_arms" => baseline_clusters.len().min(treated_clusters.len()),
        "paired_by_cluster" => baseline_clusters.len(),
        _ => 0,
    };
    if effective_clusters < analysis.minimum_independent_clusters {
        return Err(Decision::blocked("actual independent manufacturing clusters are below the preregistered minimum"));
    }

    let interval_passes = match constraint.limit_direction.as_str() {
        "maximum_ratio" => {
            if constraint.ratio_lower > constraint.limit_value {
                return Err(Decision::rejected("lower confidence bound already exceeds the maximum permitted substrate-degradation ratio"));
            }
            if constraint.ratio_upper <= constraint.limit_value {
                true
            } else {
                return Err(Decision::blocked("uncertainty interval crosses the maximum permitted substrate-degradation ratio"));
            }
        }
        "minimum_ratio" => {
            if constraint.ratio_upper < constraint.limit_value {
                return Err(Decision::rejected("upper confidence bound is below the minimum required substrate-retention ratio"));
            }
            if constraint.ratio_lower >= constraint.limit_value {
                true
            } else {
                return Err(Decision::blocked("uncertainty interval crosses the minimum required substrate-retention ratio"));
            }
        }
        _ => unreachable!("limit direction validated above"),
    };
    if !interval_passes {
        return Err(Decision::blocked("hard substrate-integrity constraint was not established"));
    }
    Ok(())
}

fn audit_case(case: &SurfaceIntegrityCase) -> Decision {
    let substrate = &case.substrate;
    let exposure = &case.exposure;
    for value in [
        substrate.substrate_grade.as_str(),
        substrate.supplier_id.as_str(),
        substrate.lot_id.as_str(),
        substrate.product_function.as_str(),
        substrate.surface_finish.as_str(),
    ] {
        if !nonempty(value) {
            return Decision::blocked("substrate identity, lot or product-function declaration is incomplete");
        }
    }
    if !sha256_shape(&substrate.profile_sha256)
        || !sha256_shape(&substrate.constraint_plan_sha256)
    {
        return Decision::blocked("substrate or constraint-plan digest is missing or malformed");
    }
    if !substrate.thickness_mm.is_finite()
        || substrate.thickness_mm <= 0.0
        || !substrate.minimum_confidence_level.is_finite()
        || !(0.0..=1.0).contains(&substrate.minimum_confidence_level)
    {
        return Decision::rejected("substrate thickness or minimum confidence requirement is invalid");
    }

    for value in [
        exposure.plasma_article_id.as_str(),
        exposure.plasma_head_id.as_str(),
        exposure.gas_species.as_str(),
        exposure.time_source_ref.as_str(),
    ] {
        if !nonempty(value) {
            return Decision::blocked("delivered plasma article, head, gas or time-source identity is missing");
        }
    }
    if !sha256_shape(&exposure.exposure_profile_sha256)
        || !sha256_shape(&exposure.raw_trace_sha256)
    {
        return Decision::blocked("exposure profile or raw process-trace digest is missing or malformed");
    }
    if !exposure.gas_purity_pct.is_finite()
        || !(0.0..=100.0).contains(&exposure.gas_purity_pct)
        || !exposure.delivered_power_w.is_finite()
        || exposure.delivered_power_w <= 0.0
        || !exposure.gas_flow_l_min.is_finite()
        || exposure.gas_flow_l_min < 0.0
        || !exposure.standoff_mm.is_finite()
        || exposure.standoff_mm <= 0.0
        || !exposure.line_speed_mm_s.is_finite()
        || exposure.line_speed_mm_s <= 0.0
        || exposure.pass_count == 0
        || !exposure.treatment_to_bond_min.is_finite()
        || exposure.treatment_to_bond_min < 0.0
        || !exposure.maximum_treatment_to_bond_min.is_finite()
        || exposure.maximum_treatment_to_bond_min < 0.0
    {
        return Decision::rejected("plasma exposure values are outside the structural process-record contract");
    }
    if exposure.treatment_to_bond_min > exposure.maximum_treatment_to_bond_min {
        return Decision::rejected("treatment-to-bond time exceeds the declared material/application process window");
    }

    if case.integrity_constraints.is_empty() {
        return Decision::blocked("contact-angle or process telemetry alone cannot establish substrate integrity; at least one hard substrate-property constraint is required");
    }
    let mut seen_ids = HashSet::new();
    let mut seen_endpoints = HashSet::new();
    let mut seen_artifacts = HashSet::new();
    for constraint in &case.integrity_constraints {
        if !seen_ids.insert(constraint.constraint_id.as_str())
            || !seen_endpoints.insert(constraint.endpoint.as_str())
        {
            return Decision::rejected("duplicate integrity constraint ID or endpoint");
        }
        for receipt in [&constraint.baseline_method_receipt, &constraint.treated_method_receipt] {
            if let Some(raw) = receipt.raw_data_sha256.as_deref() {
                if !seen_artifacts.insert(raw.to_ascii_lowercase()) {
                    return Decision::rejected("the same raw observation artifact is reused across substrate-integrity measurements");
                }
            }
        }
        if let Err(decision) = audit_method_pair(constraint) {
            return decision;
        }
        if let Err(decision) = audit_constraint(constraint, substrate, exposure) {
            return decision;
        }
    }

    if let Some(diagnostic) = &case.contact_angle_diagnostic {
        if diagnostic.method_id != "WATER_CONTACT_ANGLE_PROTOCOL"
            || diagnostic.requested_claim != "wettability_observation"
        {
            return Decision::rejected("optional contact-angle diagnostic is not typed as a wettability observation");
        }
        if decide(diagnostic).disposition != ClaimDisposition::Supported {
            return Decision::blocked("optional contact-angle diagnostic is outside its registered observation scope");
        }
    }

    Decision::pass()
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-surface-integrity-fixture-v0.1" {
        bail!("unsupported surface-integrity fixture schema: {}", fixture.schema_version);
    }
    let mut counts = [0usize; 3];
    let mut seen = HashSet::new();
    for case in &fixture.cases {
        if !seen.insert(case.id.as_str()) {
            bail!("duplicate surface-integrity case identity: {}", case.id);
        }
        let decision = audit_case(case);
        let index = match decision.disposition {
            Disposition::Pass => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        println!("{}: {} / {} — {}", case.id, decision.disposition.as_str(), decision.eligible, decision.reason);
        if decision.disposition.as_str() != case.expected_disposition {
            bail!("case {}: expected {}, got {}", case.id, case.expected_disposition, decision.disposition.as_str());
        }
        if nonempty(&case.rationale) == false {
            bail!("case {} has an empty rationale", case.id);
        }
    }
    println!("PLASMA-SURFACE-INTEGRITY S0 — {} cases; PASS={} BLOCKED={} REJECTED={}", fixture.cases.len(), counts[0], counts[1], counts[2]);
    println!("Hard constraints are application-specific; a low contact angle or good bond result cannot offset a substrate-integrity violation.");
    println!("Claim ceiling: structural hard-limit evidence only; no physical substrate integrity or product qualification is claimed.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../data/research/plasma_surface_integrity_fixture_v0_1.json");

    fn fixture() -> Fixture {
        serde_json::from_str(FIXTURE).expect("valid surface-integrity fixture")
    }

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let fixture = fixture();
        assert_eq!(fixture.schema_version, "plasma-surface-integrity-fixture-v0.1");
        for case in &fixture.cases {
            assert_eq!(audit_case(case).disposition.as_str(), case.expected_disposition, "{}", case.id);
        }
    }

    #[test]
    fn excellent_contact_angle_cannot_offset_barrier_damage() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "wettability-improves-barrier-hard-limit-fails").expect("case exists");
        assert!(case.claimed_bond_outcome_passed);
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }

    #[test]
    fn uncertainty_crossing_hard_limit_blocks_window_qualification() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "barrier-interval-crosses-limit").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn contact_angle_only_is_not_a_surface_integrity_pass() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "contact-angle-only").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn d882_thickness_scope_is_checked_against_profile() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "d882-outside-thickness-profile").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn method_receipt_sample_mismatch_rejects() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "method-receipt-sample-mismatch").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }

    #[test]
    fn visual_inspection_alone_cannot_replace_barrier_and_mechanical_constraints() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "visual-only-no-hard-measurement").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Blocked);
    }

    #[test]
    fn treatment_delay_above_declared_window_rejects() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|case| case.id == "treatment-to-bond-window-exceeded").expect("case exists");
        assert_eq!(audit_case(case).disposition, Disposition::Rejected);
    }
}
