use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ScopeDisposition, ScopeEvidence};
use crate::plasma_equivalence_design::{
    validate_bridge, Disposition as DesignDisposition, EquivalenceDesignReceipt,
};

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct SpecimenCluster {
    specimen_id: String,
    cluster_id: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct AgingExposure {
    profile_id: String,
    exposure_receipt_id: String,
    protocol_id: String,
    protocol_sha256: String,
    conditioning_mode: String,
    exposure_receipt_sha256: String,
    conditions_sha256: String,
    conditioning_method_receipt: ScopeEvidence,
    temperature_c: f64,
    relative_humidity_pct: f64,
    duration_hours: f64,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct ObservationNode {
    role: String,
    receipt: ScopeEvidence,
    specimen_clusters: Vec<SpecimenCluster>,
    aging_exposure: Option<AgingExposure>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct AnalysisReceipt {
    result_id: String,
    graph_id: String,
    result_sha256: String,
    design_id: String,
    design_sha256: String,
    plan_sha256: String,
    transform_sha256: String,
    graph_sha256: String,
    analysis_profile: String,
    analysis_method_ref: String,
    quantity_unit: String,
    confidence_level: f64,
    margin_lower: f64,
    margin_upper: f64,
    interval_lower: f64,
    interval_upper: f64,
    source_receipt_ids: Vec<String>,
    source_artifact_digests: Vec<String>,
    outcome: String,
    claim_ceiling: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct EndpointGraph {
    graph_id: String,
    output_endpoint: String,
    transform_id: String,
    transform_version: String,
    formula_id: String,
    transform_sha256: String,
    preregistered_at: i64,
    first_outcome_available_at: i64,
    changed_after_outcome: bool,
    inputs: Vec<ObservationNode>,
    classification_taxonomy_id: Option<String>,
    classification_taxonomy_sha256: Option<String>,
    target_failure_modes: Vec<String>,
    design_receipt: EquivalenceDesignReceipt,
    analysis: AnalysisReceipt,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphCase {
    id: String,
    graph: EndpointGraph,
    expected_disposition: String,
    expected_outcome: String,
    rationale: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<GraphCase>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Disposition {
    Pass,
    Blocked,
    Rejected,
}

impl Disposition {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Blocked => "BLOCKED",
            Self::Rejected => "REJECTED",
        }
    }
}

pub(super) struct Decision {
    pub(super) disposition: Disposition,
    pub(super) outcome: &'static str,
    pub(super) reason: &'static str,
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

fn unique_strings(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    !values.is_empty()
        && values
            .iter()
            .all(|value| nonempty(value) && seen.insert(value.as_str()))
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9
}

fn expected_roles(graph: &EndpointGraph) -> Option<&'static [&'static str]> {
    match graph.transform_id.as_str() {
        "unit_preserving_strength_projection_v1"
            if graph.output_endpoint == "initial_lap_shear_mpa" =>
        {
            Some(&["baseline_strength", "candidate_strength"])
        }
        "aged_strength_retention_ratio_v1"
            if graph.output_endpoint == "aged_strength_retention" =>
        {
            Some(&[
                "baseline_unaged_strength",
                "baseline_aged_strength",
                "candidate_unaged_strength",
                "candidate_aged_strength",
            ])
        }
        "predeclared_failure_mode_fraction_v1"
            if graph.output_endpoint == "failure_mode_fraction" =>
        {
            Some(&[
                "baseline_failure_classification",
                "candidate_failure_classification",
            ])
        }
        _ => None,
    }
}

fn method_is_joint_strength(receipt: &ScopeEvidence) -> bool {
    matches!(receipt.method_id.as_str(), "ASTM_D3163" | "ASTM_D5868")
        && receipt.endpoint == "apparent_shear_strength"
        && receipt.evidence_kind == "measurement"
}

fn validate_node(node: &ObservationNode) -> std::result::Result<(), Disposition> {
    if !nonempty(&node.role) || !unique_strings(&node.receipt.specimen_ids) {
        return Err(Disposition::Blocked);
    }
    let mut mapped = Vec::new();
    let mut specimens = HashSet::new();
    let mut clusters = HashSet::new();
    for item in &node.specimen_clusters {
        if !nonempty(&item.specimen_id) || !nonempty(&item.cluster_id) {
            return Err(Disposition::Blocked);
        }
        if !specimens.insert(item.specimen_id.as_str()) {
            return Err(Disposition::Rejected);
        }
        clusters.insert(item.cluster_id.as_str());
        mapped.push(item.specimen_id.clone());
    }
    if !unique_strings(&mapped) || !same_string_members(&mapped, &node.receipt.specimen_ids) {
        return Err(Disposition::Rejected);
    }
    if clusters.len() < 3 {
        return Err(Disposition::Blocked);
    }

    match decide(&node.receipt).disposition {
        ScopeDisposition::Supported => {}
        ScopeDisposition::Blocked => return Err(Disposition::Blocked),
        ScopeDisposition::Rejected => return Err(Disposition::Rejected),
    }
    Ok(())
}

fn same_string_members(left: &[String], right: &[String]) -> bool {
    if left.len() != right.len() || !unique_strings(left) || !unique_strings(right) {
        return false;
    }
    let left: HashSet<&str> = left.iter().map(String::as_str).collect();
    let right: HashSet<&str> = right.iter().map(String::as_str).collect();
    left == right
}

fn specimen_set(node: &ObservationNode) -> HashSet<&str> {
    node.receipt
        .specimen_ids
        .iter()
        .map(String::as_str)
        .collect()
}

fn cluster_set(node: &ObservationNode) -> HashSet<&str> {
    node.specimen_clusters
        .iter()
        .map(|sample| sample.cluster_id.as_str())
        .collect()
}

fn validate_strength_method(node: &ObservationNode) -> std::result::Result<(), Disposition> {
    if !method_is_joint_strength(&node.receipt) {
        return Err(Disposition::Rejected);
    }
    Ok(())
}

fn validate_aging_exposure(node: &ObservationNode) -> std::result::Result<(), Disposition> {
    let Some(exposure) = &node.aging_exposure else {
        return Err(Disposition::Blocked);
    };
    if !nonempty(&exposure.profile_id)
        || !nonempty(&exposure.exposure_receipt_id)
        || !nonempty(&exposure.protocol_id)
    {
        return Err(Disposition::Blocked);
    }
    if !sha256_shape(&exposure.protocol_sha256)
        || !sha256_shape(&exposure.exposure_receipt_sha256)
        || !sha256_shape(&exposure.conditions_sha256)
    {
        return Err(Disposition::Blocked);
    }
    if !same_string_members(
        &exposure.conditioning_method_receipt.specimen_ids,
        &node.receipt.specimen_ids,
    ) {
        return Err(Disposition::Rejected);
    }
    let method = &exposure.conditioning_method_receipt;
    if !matches!(method.method_id.as_str(), "ASTM_D1151" | "ASTM_D1183")
        || method.evidence_kind != "conditioning_practice"
        || method.requested_claim != "conditioning_protocol_reference"
        || exposure.protocol_id != method.protocol_id.as_deref().unwrap_or("")
        || exposure.protocol_sha256 != method.protocol_sha256.as_deref().unwrap_or("")
        || (method.method_id == "ASTM_D1151" && exposure.conditioning_mode != "continuous")
        || (method.method_id == "ASTM_D1183" && exposure.conditioning_mode != "cyclic")
    {
        return Err(Disposition::Rejected);
    }
    match decide(method).disposition {
        ScopeDisposition::Supported => {}
        ScopeDisposition::Blocked => return Err(Disposition::Blocked),
        ScopeDisposition::Rejected => return Err(Disposition::Rejected),
    }
    if !exposure.temperature_c.is_finite()
        || !exposure.relative_humidity_pct.is_finite()
        || !exposure.duration_hours.is_finite()
        || exposure.duration_hours <= 0.0
        || !(0.0..=100.0).contains(&exposure.relative_humidity_pct)
    {
        return Err(Disposition::Rejected);
    }
    Ok(())
}

fn validate_direct(graph: &EndpointGraph, nodes: &[&ObservationNode]) -> std::result::Result<(), Disposition> {
    if nodes.len() != 2 {
        return Err(Disposition::Rejected);
    }
    for node in nodes {
        validate_strength_method(node)?;
        if node.aging_exposure.is_some() {
            return Err(Disposition::Rejected);
        }
    }
    let baseline = nodes.iter().find(|node| node.role == "baseline_strength").ok_or(Disposition::Blocked)?;
    let candidate = nodes.iter().find(|node| node.role == "candidate_strength").ok_or(Disposition::Blocked)?;
    if !specimen_set(baseline).is_disjoint(&specimen_set(candidate))
        || !cluster_set(baseline).is_disjoint(&cluster_set(candidate))
    {
        return Err(Disposition::Rejected);
    }
    if !same_method_scope(&baseline.receipt, &candidate.receipt) {
        return Err(Disposition::Rejected);
    }
    if graph.analysis.analysis_profile != "two_sample_tost_v1" {
        return Err(Disposition::Blocked);
    }
    Ok(())
}

fn validate_aged_retention(nodes: &[&ObservationNode]) -> std::result::Result<(), Disposition> {
    if nodes.len() != 4 {
        return Err(Disposition::Blocked);
    }
    let get = |role: &str| nodes.iter().find(|node| node.role == role).copied().ok_or(Disposition::Blocked);
    let bu = get("baseline_unaged_strength")?;
    let ba = get("baseline_aged_strength")?;
    let cu = get("candidate_unaged_strength")?;
    let ca = get("candidate_aged_strength")?;

    for node in [bu, ba, cu, ca] {
        validate_strength_method(node)?;
    }
    validate_aging_exposure(ba)?;
    validate_aging_exposure(ca)?;
    if bu.aging_exposure.is_some() {
        return Err(Disposition::Rejected);
    }
    if cu.aging_exposure.is_some() {
        return Err(Disposition::Rejected);
    }

    if !same_method_scope(&bu.receipt, &ba.receipt)
        || !same_method_scope(&cu.receipt, &ca.receipt)
        || !same_method_scope(&bu.receipt, &cu.receipt)
    {
        return Err(Disposition::Rejected);
    }

    if !cluster_set(bu).is_disjoint(&cluster_set(cu))
        || cluster_set(bu) != cluster_set(ba)
        || cluster_set(cu) != cluster_set(ca)
    {
        return Err(Disposition::Rejected);
    }

    let mut all_specimens = HashSet::new();
    for node in [bu, ba, cu, ca] {
        for specimen in &node.receipt.specimen_ids {
            if !all_specimens.insert(specimen.as_str()) {
                return Err(Disposition::Rejected);
            }
        }
    }

    let base_exposure = ba.aging_exposure.as_ref().ok_or(Disposition::Blocked)?;
    let candidate_exposure = ca.aging_exposure.as_ref().ok_or(Disposition::Blocked)?;
    if base_exposure.profile_id != candidate_exposure.profile_id
        || base_exposure.exposure_receipt_id == candidate_exposure.exposure_receipt_id
        || base_exposure.exposure_receipt_sha256 == candidate_exposure.exposure_receipt_sha256
        || base_exposure.protocol_id != candidate_exposure.protocol_id
        || base_exposure.protocol_sha256 != candidate_exposure.protocol_sha256
        || base_exposure.conditions_sha256 != candidate_exposure.conditions_sha256
        || !same_method_scope(
            &base_exposure.conditioning_method_receipt,
            &candidate_exposure.conditioning_method_receipt,
        )
        || base_exposure.conditioning_method_receipt.id == candidate_exposure.conditioning_method_receipt.id
        || base_exposure.conditioning_method_receipt.raw_data_sha256
            == candidate_exposure.conditioning_method_receipt.raw_data_sha256
        || !close(base_exposure.temperature_c, candidate_exposure.temperature_c)
        || !close(base_exposure.relative_humidity_pct, candidate_exposure.relative_humidity_pct)
        || !close(base_exposure.duration_hours, candidate_exposure.duration_hours)
        || base_exposure.exposure_receipt_sha256 == candidate_exposure.exposure_receipt_sha256
    {
        return Err(Disposition::Rejected);
    }
    Ok(())
}

fn validate_failure_fraction(
    graph: &EndpointGraph,
    nodes: &[&ObservationNode],
) -> std::result::Result<(), Disposition> {
    if nodes.len() != 2 {
        return Err(Disposition::Rejected);
    }
    for node in nodes {
        if node.receipt.method_id != "ASTM_D5573"
            || node.receipt.endpoint != "frp_joint_failure_mode_classification"
            || node.receipt.evidence_kind != "classification_practice"
            || node.receipt.requested_claim != "failure_mode_classification_reference"
        {
            return Err(Disposition::Rejected);
        }
        if node.aging_exposure.is_some() {
            return Err(Disposition::Rejected);
        }
    }
    let baseline = nodes.iter().find(|node| node.role == "baseline_failure_classification").ok_or(Disposition::Blocked)?;
    let candidate = nodes.iter().find(|node| node.role == "candidate_failure_classification").ok_or(Disposition::Blocked)?;
    if baseline.receipt.material_pair_class != "frp_smc_joint"
        || candidate.receipt.material_pair_class != "frp_smc_joint"
        || baseline.receipt.geometry != "failure_surface_classification"
        || candidate.receipt.geometry != "failure_surface_classification"
    {
        return Err(Disposition::Rejected);
    }
    if !specimen_set(baseline).is_disjoint(&specimen_set(candidate))
        || !cluster_set(baseline).is_disjoint(&cluster_set(candidate))
    {
        return Err(Disposition::Rejected);
    }
    if !same_method_scope(&baseline.receipt, &candidate.receipt) {
        return Err(Disposition::Rejected);
    }
    if !nonempty(graph.classification_taxonomy_id.as_deref().unwrap_or(""))
        || !graph
            .classification_taxonomy_sha256
            .as_deref()
            .is_some_and(sha256_shape)
        || !unique_strings(&graph.target_failure_modes)
    {
        return Err(Disposition::Blocked);
    }
    // A generic two-proportion score interval does not account for the declared
    // manufacturing-cluster structure or establish an equivalence design.
    // Keep this endpoint BLOCKED until Statistics registers and validates a
    // cluster-aware proportion-equivalence profile and its independent reference.
    if graph.analysis.analysis_profile != "clustered_two_proportion_equivalence_pending_v1" {
        return Err(Disposition::Blocked);
    }
    Err(Disposition::Blocked)
}

fn same_method_scope(left: &ScopeEvidence, right: &ScopeEvidence) -> bool {
    left.method_id == right.method_id
        && left.edition == right.edition
        && left.evidence_kind == right.evidence_kind
        && left.material_pair_class == right.material_pair_class
        && left.geometry == right.geometry
        && left.endpoint == right.endpoint
        && left.requested_claim == right.requested_claim
        && left.protocol_id == right.protocol_id
        && left.protocol_sha256 == right.protocol_sha256
}

pub(super) fn audit_graph(graph: &EndpointGraph) -> Decision {
    let blocked = |reason| Decision { disposition: Disposition::Blocked, outcome: "NOT_EVALUATED", reason };
    let rejected = |reason| Decision { disposition: Disposition::Rejected, outcome: "NOT_EVALUATED", reason };

    if !nonempty(&graph.graph_id)
        || !nonempty(&graph.output_endpoint)
        || !nonempty(&graph.transform_version)
        || !sha256_shape(&graph.transform_sha256)
    {
        return blocked("graph, output, version or transform digest is missing");
    }
    if graph.changed_after_outcome || graph.preregistered_at >= graph.first_outcome_available_at {
        return rejected("endpoint transform or analysis plan was not frozen before outcome availability");
    }
    if !nonempty(&graph.formula_id) {
        return blocked("a transform formula identifier is required");
    }
    if graph.inputs.is_empty() {
        return blocked("endpoint graph has no source observations");
    }

    let Some(roles) = expected_roles(graph) else {
        return rejected("endpoint transform/output combination is not in the versioned S0 registry");
    };
    let expected_formula = match graph.transform_id.as_str() {
        "unit_preserving_strength_projection_v1" => "identity_strength_unit_projection_v1",
        "aged_strength_retention_ratio_v1" => "mean_aged_strength_over_mean_unaged_strength_v1",
        "predeclared_failure_mode_fraction_v1" => "target_modes_over_classified_specimens_v1",
        _ => unreachable!("transform registry checked above"),
    };
    if graph.formula_id != expected_formula {
        return rejected("formula identifier does not match the registered endpoint transformation");
    }
    if graph.inputs.len() != roles.len() {
        return blocked("endpoint graph does not contain the exact required number of source observations");
    }

    let mut role_set = HashSet::new();
    let mut receipt_ids = HashSet::new();
    let mut raw_digests = HashSet::new();
    let mut source_ids = Vec::new();
    let mut source_digests = Vec::new();
    let mut nodes = Vec::new();

    for node in &graph.inputs {
        if !roles.contains(&node.role.as_str()) || !role_set.insert(node.role.as_str()) {
            return rejected("input role is duplicated or not admitted by the endpoint transform");
        }
        if !receipt_ids.insert(node.receipt.id.as_str()) {
            return rejected("source observation receipt ID is reused");
        }
        let Some(raw_digest) = node.receipt.raw_data_sha256.as_deref() else {
            return blocked("source observation has no raw-data digest");
        };
        if !raw_digests.insert(raw_digest.to_ascii_lowercase()) {
            return rejected("source observation raw-data artifact is reused");
        }
        match validate_node(node) {
            Ok(()) => {}
            Err(Disposition::Blocked) => return blocked("source observation scope, specimen mapping or support is incomplete"),
            Err(Disposition::Rejected) => return rejected("source observation scope or specimen mapping is contradictory"),
            Err(Disposition::Pass) => unreachable!(),
        }
        source_ids.push(node.receipt.id.clone());
        source_digests.push(raw_digest.to_owned());
        if let Some(exposure) = &node.aging_exposure {
            let condition_receipt = &exposure.conditioning_method_receipt;
            if !receipt_ids.insert(condition_receipt.id.as_str()) {
                return rejected("conditioning-method receipt ID is reused");
            }
            let Some(condition_raw_digest) = condition_receipt.raw_data_sha256.as_deref() else {
                return blocked("conditioning-method observation has no raw-data digest");
            };
            if !raw_digests.insert(condition_raw_digest.to_ascii_lowercase()) {
                return rejected("conditioning-method raw-data artifact is reused");
            }
            source_ids.push(condition_receipt.id.clone());
            source_digests.push(condition_raw_digest.to_owned());
            if !receipt_ids.insert(exposure.exposure_receipt_id.as_str()) {
                return rejected("aging exposure receipt ID is reused");
            }
            source_ids.push(exposure.exposure_receipt_id.clone());
            source_digests.push(exposure.exposure_receipt_sha256.clone());
        }
        nodes.push(node);
    }

    for role in roles {
        if !role_set.contains(role) {
            return blocked("a required source observation role is missing");
        }
    }

    let required_ceiling = match graph.transform_id.as_str() {
        "unit_preserving_strength_projection_v1" => "comparative_joint_strength_observation_only",
        "aged_strength_retention_ratio_v1" => "aged_strength_retention_observation_only",
        "predeclared_failure_mode_fraction_v1" => "failure_mode_fraction_observation_only_not_acceptability",
        _ => unreachable!("transform registry checked above"),
    };
    if graph.analysis.claim_ceiling != required_ceiling {
        return rejected("derived claim ceiling exceeds or differs from the registered transform ceiling");
    }

    let transform_check = match graph.transform_id.as_str() {
        "unit_preserving_strength_projection_v1" => validate_direct(graph, &nodes),
        "aged_strength_retention_ratio_v1" => validate_aged_retention(&nodes),
        "predeclared_failure_mode_fraction_v1" => validate_failure_fraction(graph, &nodes),
        _ => unreachable!("transform registry checked above"),
    };
    match transform_check {
        Ok(()) => {}
        Err(Disposition::Blocked) => return blocked("transform inputs or required analysis profile are incomplete"),
        Err(Disposition::Rejected) => return rejected("transform inputs, cohorts or method scopes are contradictory"),
        Err(Disposition::Pass) => unreachable!(),
    }

    let (baseline_design_observation, candidate_design_observation) =
        match graph.transform_id.as_str() {
            "unit_preserving_strength_projection_v1" => (
                nodes.iter().find(|node| node.role == "baseline_strength"),
                nodes.iter().find(|node| node.role == "candidate_strength"),
            ),
            "aged_strength_retention_ratio_v1" => (
                nodes.iter().find(|node| node.role == "baseline_aged_strength"),
                nodes.iter().find(|node| node.role == "candidate_aged_strength"),
            ),
            "predeclared_failure_mode_fraction_v1" => (
                nodes.iter().find(|node| node.role == "baseline_failure_classification"),
                nodes.iter().find(|node| node.role == "candidate_failure_classification"),
            ),
            _ => unreachable!("transform registry checked above"),
        };
    let (Some(baseline_observation), Some(candidate_observation)) =
        (baseline_design_observation, candidate_design_observation)
    else {
        return blocked("design receipt cannot be linked to its baseline/candidate observation roles");
    };

    let Some(design_receipt_sha256) = graph.design_receipt.receipt_sha256.as_deref() else {
        return blocked("full equivalence-design receipt digest is missing");
    };
    let Some(transform_sha256) = graph.design_receipt.endpoint_transform_sha256.as_deref() else {
        return blocked("design receipt does not bind the endpoint transform digest");
    };
    if !sha256_shape(design_receipt_sha256)
        || transform_sha256 != graph.transform_sha256
        || graph.design_receipt.endpoint_transform_id.as_deref() != Some(graph.transform_id.as_str())
        || graph.design_receipt.observation_endpoint_id.as_deref()
            != Some(baseline_observation.receipt.endpoint.as_str())
        || analysis_design_id_mismatch(&graph.analysis, &graph.design_receipt)
    {
        return rejected("equivalence-design receipt is not bound to this graph, output endpoint and exact transform");
    }

    match validate_bridge(
        &graph.design_receipt,
        &graph.output_endpoint,
        &baseline_observation.receipt,
        &candidate_observation.receipt,
    ).disposition {
        DesignDisposition::Pass => {}
        DesignDisposition::Blocked => return blocked("equivalence-design receipt or its exact observation binding is incomplete"),
        DesignDisposition::Rejected => return rejected("equivalence-design receipt contradicts the endpoint transformation or observations"),
    }

    let analysis = &graph.analysis;
    if !nonempty(&analysis.result_id)
        || !nonempty(&analysis.graph_id)
        || analysis.graph_id != graph.graph_id
        || !nonempty(&analysis.design_id)
        || !nonempty(&analysis.analysis_method_ref)
        || !nonempty(&analysis.quantity_unit)
    {
        return blocked("analysis result, design, method or output unit reference is missing");
    }
    for digest in [
        analysis.result_sha256.as_str(),
        analysis.design_sha256.as_str(),
        analysis.plan_sha256.as_str(),
        analysis.transform_sha256.as_str(),
        analysis.graph_sha256.as_str(),
    ] {
        if !sha256_shape(digest) {
            return blocked("analysis or design digest is missing or malformed");
        }
    }
    if analysis.transform_sha256 != graph.transform_sha256
        || analysis.analysis_profile != expected_analysis_profile(graph)
        || !unique_strings(&analysis.source_receipt_ids)
        || !unique_strings(&analysis.source_artifact_digests)
        || analysis.source_receipt_ids.iter().collect::<HashSet<_>>()
            != source_ids.iter().collect::<HashSet<_>>()
        || analysis.source_artifact_digests.iter().collect::<HashSet<_>>()
            != source_digests.iter().collect::<HashSet<_>>()
    {
        return rejected("analysis result does not bind the exact transform and source observation receipts");
    }
    let expected_unit = match graph.transform_id.as_str() {
        "unit_preserving_strength_projection_v1" => "MPa",
        "aged_strength_retention_ratio_v1" => "ratio",
        "predeclared_failure_mode_fraction_v1" => "proportion",
        _ => unreachable!("transform registry checked above"),
    };
    if analysis.quantity_unit != expected_unit {
        return rejected("analysis result quantity unit does not match the derived endpoint");
    }
    if !close(analysis.confidence_level, 0.90)
        || !analysis.margin_lower.is_finite()
        || !analysis.margin_upper.is_finite()
        || analysis.margin_lower >= analysis.margin_upper
        || !analysis.interval_lower.is_finite()
        || !analysis.interval_upper.is_finite()
        || analysis.interval_lower > analysis.interval_upper
    {
        return rejected("analysis interval, confidence or equivalence margin is invalid");
    }

    let graph_binding_sha = analysis.graph_sha256.as_str();
    if !sha256_shape(graph_binding_sha) {
        return blocked("graph binding digest is missing");
    }

    let equivalent = analysis.interval_lower >= analysis.margin_lower
        && analysis.interval_upper <= analysis.margin_upper;
    let outcome = if equivalent { "EQUIVALENCE_ESTABLISHED" } else { "EQUIVALENCE_NOT_ESTABLISHED" };
    if analysis.outcome != outcome {
        return rejected("analysis outcome contradicts its declared interval and margin");
    }

    Decision {
        disposition: Disposition::Pass,
        outcome,
        reason: "source observations, specimen/cluster lineage, registered transform and analysis result are structurally linked; numerical validity is not certified",
    }
}

fn analysis_design_id_mismatch(
    analysis: &AnalysisReceipt,
    design: &EquivalenceDesignReceipt,
) -> bool {
    analysis.design_id != design.design_id
        || analysis.design_sha256 != design.receipt_sha256.as_deref().unwrap_or("")
        || analysis.plan_sha256 != design.plan_sha256
        || analysis.result_id != design.analysis_result.result_id
        || analysis.result_sha256 != design.analysis_result.result_sha256
        || analysis.analysis_profile != design.analysis_profile
        || analysis.design_id != design.analysis_result.design_id
        || analysis.plan_sha256 != design.analysis_result.plan_sha256
}

fn expected_analysis_profile(graph: &EndpointGraph) -> &'static str {
    match graph.transform_id.as_str() {
        "unit_preserving_strength_projection_v1" | "aged_strength_retention_ratio_v1" => "two_sample_tost_v1",
        "predeclared_failure_mode_fraction_v1" => "clustered_two_proportion_equivalence_pending_v1",
        _ => "unsupported",
    }
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-endpoint-transform-fixture-v0.1" {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }
    let mut counts = [0usize; 3];
    for case in &fixture.cases {
        let decision = audit_graph(&case.graph);
        let index = match decision.disposition {
            Disposition::Pass => 0,
            Disposition::Blocked => 1,
            Disposition::Rejected => 2,
        };
        counts[index] += 1;
        println!("{}: {} / {} — {}", case.id, decision.disposition.as_str(), decision.outcome, decision.reason);
        if decision.disposition.as_str() != case.expected_disposition || decision.outcome != case.expected_outcome {
            bail!("case {}: expected {}/{} but got {}/{}", case.id, case.expected_disposition, case.expected_outcome, decision.disposition.as_str(), decision.outcome);
        }
        if !nonempty(&case.rationale) {
            bail!("case {} has an empty rationale", case.id);
        }
    }
    println!("PLASMA-ENDPOINT-TRANSFORM S0 — {} cases; PASS={} BLOCKED={} REJECTED={}", fixture.cases.len(), counts[0], counts[1], counts[2]);
    println!("Claim ceiling: typed source and transformation lineage only; numerical transformation and physical qualification remain external.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../data/research/plasma_endpoint_transform_fixture_v0_1.json");

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        assert_eq!(fixture.schema_version, "plasma-endpoint-transform-fixture-v0.1");
        for case in &fixture.cases {
            let result = audit_graph(&case.graph);
            assert_eq!(result.disposition.as_str(), case.expected_disposition, "{}", case.id);
            assert_eq!(result.outcome, case.expected_outcome, "{}", case.id);
        }
    }

    #[test]
    fn aged_strength_requires_four_separate_observation_cohorts() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "aged-retention-missing-unaged-reference").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Blocked);
    }

    #[test]
    fn failure_mode_fraction_stays_blocked_until_clustered_proportion_method_is_qualified() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-failure-mode-fraction-classification").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Blocked);
    }

    #[test]
    fn failure_classification_cannot_claim_acceptability() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "failure-mode-claim-ceiling-escalation").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Rejected);
    }

    #[test]
    fn repeated_specimen_or_raw_artifact_is_rejected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "aged-retention-reuses-raw-observation").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Rejected);
    }

    #[test]
    fn registered_d5868_frp_joint_strength_is_admissible() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-frp-d5868-direct-strength").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Pass);
    }

    #[test]
    fn constitutive_response_method_cannot_replace_joint_strength() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "d5656-constitutive-response-not-joint-strength").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Rejected);
    }

    #[test]
    fn unsupported_d5573_material_family_is_blocked() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "d5573-general-frp-scope-not-admitted").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Blocked);
    }

    #[test]
    fn missing_design_digest_blocks_a_derived_endpoint() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "missing-design-receipt-digest").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Blocked);
    }

    #[test]
    fn analysis_result_must_bind_every_source_receipt() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "analysis-result-input-lineage-mismatch").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Rejected);
    }

    #[test]
    fn invalid_aging_method_or_protocol_cannot_support_retention() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "conditioning-receipt-protocol-mismatch").expect("case exists");
        assert_eq!(audit_graph(&case.graph).disposition, Disposition::Rejected);
    }

    #[test]
    fn design_receipt_for_a_different_transform_is_rejected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-direct-initial-strength").expect("case exists");
        let mut graph = case.graph.clone();
        graph.design_receipt.endpoint_transform_id = Some("aged_strength_retention_ratio_v1".to_owned());
        assert_eq!(audit_graph(&graph).disposition, Disposition::Rejected);
    }

    #[test]
    fn design_receipt_digest_must_match_the_analysis_link() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-direct-initial-strength").expect("case exists");
        let mut graph = case.graph.clone();
        graph.analysis.design_sha256 = "ff".repeat(32);
        assert_eq!(audit_graph(&graph).disposition, Disposition::Rejected);
    }

    #[test]
    fn aged_retention_binds_conditioning_and_exposure_receipts() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-aged-retention-four-cohort-graph").expect("case exists");
        let result = audit_graph(&case.graph);
        assert_eq!(result.disposition, Disposition::Pass);
        assert_eq!(case.graph.analysis.source_receipt_ids.len(), 8);
        assert_eq!(case.graph.analysis.source_artifact_digests.len(), 8);
    }

    #[test]
    fn valid_graph_can_fail_to_establish_equivalence() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid endpoint transform fixture");
        let case = fixture.cases.iter().find(|case| case.id == "valid-direct-strength-outside-margin").expect("case exists");
        let result = audit_graph(&case.graph);
        assert_eq!(result.disposition, Disposition::Pass);
        assert_eq!(result.outcome, "EQUIVALENCE_NOT_ESTABLISHED");
    }
}
