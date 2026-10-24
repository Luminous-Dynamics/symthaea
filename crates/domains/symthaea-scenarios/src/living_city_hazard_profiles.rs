// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Versioned, evidence-linked hazard profiles that connect the benchmark
//! registry to the infrastructure dependency graph.
//!
//! A profile provides declared scenario assumptions and initiating failures;
//! it does not calculate hazard intensity or determine physical failure
//! thresholds. Those must come from an evidence-backed profile and appropriate
//! validated domain solvers. The output is a deterministic dependency pre-screen.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::living_city_benchmarks::{
    BenchmarkDefinition, BenchmarkProfileBinding, Hazard, LIVING_CITY_BENCHMARKS,
    validate_benchmark_registry,
};
use crate::living_city_dependencies::{
    DependencyAssessment, ServiceDependency, ServiceNode, assess_dependency_failures,
};
use crate::living_city_evidence::{
    EvidenceKind, EvidenceManifest, sha256_hex,
};

pub const HAZARD_PROFILE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HazardParameter {
    pub hazard: Hazard,
    /// Domain-specific, stable name such as design_dry_bulb_temperature.
    pub parameter_name: String,
    pub value: f64,
    pub unit: String,
    /// ID of an item in the exact evidence manifest bound to this profile.
    pub evidence_item_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitiatingFailure {
    pub service_id: String,
    pub cause: Hazard,
    pub evidence_item_id: String,
    /// Explain whether this is a source-backed event, a scenario assumption,
    /// or a result imported from a domain model.
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HazardProfile {
    pub schema_version: u32,
    pub profile_id: String,
    pub profile_revision: String,
    pub benchmark_id: String,
    pub study_area_id: String,
    pub jurisdiction: String,
    /// Declared simulation period/design horizon; domain-specific profiles must
    /// state the actual time basis and any averaging/window semantics.
    pub time_horizon: String,
    pub hazard_parameters: Vec<HazardParameter>,
    pub initiating_failures: Vec<InitiatingFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HazardProfileError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for HazardProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid living-city hazard profile {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for HazardProfileError {}

fn invalid(field: &'static str, reason: &'static str) -> HazardProfileError {
    HazardProfileError { field, reason }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

impl HazardProfile {
    /// Canonical profile bytes normalize the two unordered semantic collections.
    /// Parameter values and profile fields otherwise preserve their typed meaning.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, HazardProfileError> {
        let mut canonical = self.clone();
        canonical.hazard_parameters.sort_by(|left, right| {
            (left.hazard, left.parameter_name.as_str(), left.evidence_item_id.as_str())
                .cmp(&(right.hazard, right.parameter_name.as_str(), right.evidence_item_id.as_str()))
        });
        canonical.initiating_failures.sort_by(|left, right| {
            (left.service_id.as_str(), left.cause, left.evidence_item_id.as_str(), left.rationale.as_str())
                .cmp(&(right.service_id.as_str(), right.cause, right.evidence_item_id.as_str(), right.rationale.as_str()))
        });
        serde_json::to_vec(&canonical)
            .map_err(|_| invalid("canonical_bytes", "profile serialization failed"))
    }

    pub fn sha256(&self) -> Result<String, HazardProfileError> {
        Ok(sha256_hex(&self.canonical_bytes()?))
    }

    pub fn binding(&self) -> Result<BenchmarkProfileBinding, HazardProfileError> {
        Ok(BenchmarkProfileBinding {
            benchmark_id: self.benchmark_id.clone(),
            profile_id: format!("{}/{}", self.profile_id, self.profile_revision),
            profile_sha256: self.sha256()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HazardProfileAssessment {
    pub benchmark_id: String,
    pub profile_id: String,
    pub profile_sha256: String,
    pub dependency_assessment: DependencyAssessment,
}

/// Validate a profile against the versioned benchmark, exact evidence manifest,
/// declared service graph, and matching profile binding, then propagate declared
/// initiating failures through the graph.
pub fn evaluate_hazard_profile(
    profile: &HazardProfile,
    binding: &BenchmarkProfileBinding,
    manifest: &EvidenceManifest,
    nodes: &[ServiceNode],
    dependencies: &[ServiceDependency],
    definitions: &[BenchmarkDefinition],
) -> Result<HazardProfileAssessment, HazardProfileError> {
    if profile.schema_version != HAZARD_PROFILE_SCHEMA_VERSION {
        return Err(invalid("schema_version", "unsupported hazard profile schema version"));
    }
    for (field, value) in [
        ("profile_id", profile.profile_id.as_str()),
        ("profile_revision", profile.profile_revision.as_str()),
        ("benchmark_id", profile.benchmark_id.as_str()),
        ("study_area_id", profile.study_area_id.as_str()),
        ("jurisdiction", profile.jurisdiction.as_str()),
        ("time_horizon", profile.time_horizon.as_str()),
    ] {
        if !non_empty(value) {
            return Err(invalid(field, "must be explicitly provided"));
        }
    }

    manifest
        .validate()
        .map_err(|_| invalid("evidence_manifest", "evidence manifest is invalid"))?;
    if manifest.scenario_id.as_str() != profile.benchmark_id.as_str() {
        return Err(invalid(
            "scenario_id",
            "manifest and hazard profile must identify the same benchmark scenario",
        ));
    }

    validate_benchmark_registry(definitions)
        .map_err(|_| invalid("benchmark_registry", "benchmark registry is invalid"))?;
    let definition = definitions
        .iter()
        .find(|definition| definition.id == profile.benchmark_id.as_str())
        .ok_or_else(|| invalid("benchmark_id", "profile references an unregistered benchmark"))?;

    // Validate the full dependency graph even if this profile has no initial failures.
    assess_dependency_failures(nodes, dependencies, &[])
        .map_err(|_| invalid("dependency_graph", "service graph is invalid"))?;

    if binding.benchmark_id.as_str() != profile.benchmark_id.as_str() {
        return Err(invalid("binding.benchmark_id", "binding refers to a different benchmark"));
    }
    let expected_profile_id = format!("{}/{}", profile.profile_id, profile.profile_revision);
    if binding.profile_id.as_str() != expected_profile_id.as_str() {
        return Err(invalid("binding.profile_id", "binding refers to a different profile revision"));
    }
    let profile_sha256 = profile.sha256()?;
    if binding.profile_sha256 != profile_sha256 {
        return Err(invalid("binding.profile_sha256", "profile digest does not match canonical profile bytes"));
    }

    let declared_hazards: BTreeSet<Hazard> = definition.hazards.iter().copied().collect();
    let mut parameter_keys = BTreeSet::new();
    let mut covered_hazards = BTreeSet::new();

    for parameter in &profile.hazard_parameters {
        if !declared_hazards.contains(&parameter.hazard) {
            return Err(invalid(
                "hazard_parameters.hazard",
                "profile parameter references a hazard not declared by its benchmark",
            ));
        }
        if !non_empty(&parameter.parameter_name) || !non_empty(&parameter.unit) {
            return Err(invalid(
                "hazard_parameters",
                "parameter name and unit must be explicit",
            ));
        }
        if !parameter.value.is_finite() {
            return Err(invalid(
                "hazard_parameters.value",
                "NaN and infinite hazard parameter values are prohibited",
            ));
        }
        if !parameter_keys.insert((parameter.hazard, parameter.parameter_name.as_str())) {
            return Err(invalid(
                "hazard_parameters",
                "duplicate hazard/name parameter pair",
            ));
        }
        let evidence = manifest
            .items
            .iter()
            .find(|item| item.id.as_str() == parameter.evidence_item_id.as_str())
            .ok_or_else(|| invalid("hazard_parameters.evidence_item_id", "parameter evidence ID is absent from manifest"))?;
        if evidence.kind == EvidenceKind::Missing {
            return Err(invalid(
                "hazard_parameters.evidence_item_id",
                "a missing-source record cannot substantiate a numeric hazard parameter",
            ));
        }
        covered_hazards.insert(parameter.hazard);
    }

    if covered_hazards != declared_hazards {
        return Err(invalid(
            "hazard_parameters",
            "every hazard declared by the benchmark requires at least one evidenced parameter",
        ));
    }
    if definition.hazards.is_empty() && !profile.hazard_parameters.is_empty() {
        return Err(invalid(
            "hazard_parameters",
            "baseline profiles may not declare hazard stress parameters",
        ));
    }
    if definition.hazards.is_empty() && !profile.initiating_failures.is_empty() {
        return Err(invalid(
            "initiating_failures",
            "baseline profiles must not inject hazard-caused initiating failures",
        ));
    }

    let mut failure_keys = BTreeSet::new();
    let mut initiating_service_ids = BTreeSet::new();
    for failure in &profile.initiating_failures {
        if !non_empty(&failure.service_id) || !non_empty(&failure.rationale) {
            return Err(invalid(
                "initiating_failures",
                "service ID and rationale must be explicit",
            ));
        }
        if !declared_hazards.contains(&failure.cause) {
            return Err(invalid(
                "initiating_failures.cause",
                "initiating failure cause is not declared by this benchmark",
            ));
        }
        if !failure_keys.insert((
            failure.service_id.as_str(),
            failure.cause,
            failure.evidence_item_id.as_str(),
        )) {
            return Err(invalid(
                "initiating_failures",
                "duplicate service/cause/evidence tuple",
            ));
        }
        if !nodes.iter().any(|node| node.id.as_str() == failure.service_id.as_str()) {
            return Err(invalid(
                "initiating_failures.service_id",
                "initiating failure references an unknown service",
            ));
        }
        let evidence = manifest
            .items
            .iter()
            .find(|item| item.id.as_str() == failure.evidence_item_id.as_str())
            .ok_or_else(|| invalid("initiating_failures.evidence_item_id", "failure evidence ID is absent from manifest"))?;
        if evidence.kind == EvidenceKind::Missing {
            return Err(invalid(
                "initiating_failures.evidence_item_id",
                "a missing-source record cannot substantiate an initiating failure",
            ));
        }
        initiating_service_ids.insert(failure.service_id.clone());
    }

    let initial_failures: Vec<String> = initiating_service_ids.into_iter().collect();
    let dependency_assessment = assess_dependency_failures(nodes, dependencies, &initial_failures)
        .map_err(|_| invalid("dependency_graph", "dependency assessment rejected the graph or initiating failures"))?;

    Ok(HazardProfileAssessment {
        benchmark_id: profile.benchmark_id.clone(),
        profile_id: expected_profile_id,
        profile_sha256,
        dependency_assessment,
    })
}

/// Resolve a benchmark by stable ID from the initial built-in registry.
pub fn builtin_benchmark(id: &str) -> Option<&'static BenchmarkDefinition> {
    LIVING_CITY_BENCHMARKS.iter().find(|definition| definition.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::living_city_benchmarks::{
        BenchmarkClass, LIVING_CITY_BENCHMARKS,
    };
    use crate::living_city_dependencies::{DependencyMode, ServiceDomain};
    use crate::living_city_evidence::{EvidenceItem, EVIDENCE_SCHEMA_VERSION};

    const HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn evidence_item(id: &str, kind: EvidenceKind) -> EvidenceItem {
        EvidenceItem {
            id: id.into(),
            kind,
            source_uri: format!("fixture://{id}"),
            content_sha256: (kind != EvidenceKind::Missing).then(|| HASH.into()),
            license_or_basis: if kind == EvidenceKind::Missing { String::new() } else { "synthetic fixture".into() },
            units: "declared engineering unit".into(),
            spatial_scope: "synthetic study area".into(),
            temporal_scope: "synthetic design basis".into(),
            acquired_at_utc: if kind == EvidenceKind::Missing { String::new() } else { "2026-01-01T00:00:00Z".into() },
            transformation_chain: vec![],
            uncertainty_note: "synthetic test data; not field evidence".into(),
            missing_reason: (kind == EvidenceKind::Missing).then(|| "fixture gap".into()),
        }
    }

    fn heat_manifest() -> EvidenceManifest {
        EvidenceManifest {
            schema_version: EVIDENCE_SCHEMA_VERSION,
            project_id: "living-city-fixture".into(),
            scenario_id: "heat-design-event".into(),
            study_boundary_id: "synthetic-area".into(),
            coordinate_reference_system: "synthetic-local-crs".into(),
            items: vec![
                evidence_item("heat-basis", EvidenceKind::Assumed),
                evidence_item("outage-basis", EvidenceKind::Assumed),
            ],
        }
    }

    fn graph() -> (Vec<ServiceNode>, Vec<ServiceDependency>) {
        let nodes = vec![
            ServiceNode { id: "grid".into(), domain: ServiceDomain::Electricity },
            ServiceNode { id: "cooling-centre".into(), domain: ServiceDomain::Cooling },
            ServiceNode { id: "hospital".into(), domain: ServiceDomain::Healthcare },
        ];
        let edges = vec![
            ServiceDependency {
                dependent_id: "cooling-centre".into(),
                prerequisite_id: "grid".into(),
                mode: DependencyMode::Required,
            },
            ServiceDependency {
                dependent_id: "hospital".into(),
                prerequisite_id: "cooling-centre".into(),
                mode: DependencyMode::Required,
            },
        ];
        (nodes, edges)
    }

    fn heat_profile() -> HazardProfile {
        HazardProfile {
            schema_version: HAZARD_PROFILE_SCHEMA_VERSION,
            profile_id: "synthetic-heat-profile".into(),
            profile_revision: "r1".into(),
            benchmark_id: "heat-design-event".into(),
            study_area_id: "synthetic-area".into(),
            jurisdiction: "test jurisdiction".into(),
            time_horizon: "synthetic 24-hour design event".into(),
            hazard_parameters: vec![HazardParameter {
                hazard: Hazard::ExtremeHeat,
                parameter_name: "design_dry_bulb_temperature".into(),
                value: 42.0,
                unit: "degree_Celsius".into(),
                evidence_item_id: "heat-basis".into(),
            }],
            initiating_failures: vec![InitiatingFailure {
                service_id: "grid".into(),
                cause: Hazard::ExtremeHeat,
                evidence_item_id: "outage-basis".into(),
                rationale: "synthetic negative-control fixture for heat-correlated outage".into(),
            }],
        }
    }

    fn binding_for(profile: &HazardProfile) -> BenchmarkProfileBinding {
        profile.binding().unwrap()
    }

    #[test]
    fn evidence_profile_binds_benchmark_manifest_and_failure_graph() {
        let profile = heat_profile();
        let manifest = heat_manifest();
        let (nodes, edges) = graph();
        let result = evaluate_hazard_profile(
            &profile,
            &binding_for(&profile),
            &manifest,
            &nodes,
            &edges,
            LIVING_CITY_BENCHMARKS,
        )
        .unwrap();
        assert_eq!(result.benchmark_id, "heat-design-event");
        assert_eq!(result.dependency_assessment.failed_ids(), vec!["cooling-centre", "grid", "hospital"]);
        assert_eq!(result.profile_sha256, profile.sha256().unwrap());
    }

    #[test]
    fn canonical_profile_digest_ignores_unordered_collection_order() {
        let profile = heat_profile();
        let mut reordered = profile.clone();
        reordered.hazard_parameters.reverse();
        reordered.initiating_failures.reverse();
        assert_eq!(profile.sha256().unwrap(), reordered.sha256().unwrap());
    }

    #[test]
    fn changed_profile_cannot_reuse_old_binding_digest() {
        let profile = heat_profile();
        let binding = binding_for(&profile);
        let manifest = heat_manifest();
        let (nodes, edges) = graph();
        let mut modified = profile.clone();
        modified.hazard_parameters[0].value = 43.0;
        assert_eq!(
            evaluate_hazard_profile(&modified, &binding, &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "binding.profile_sha256"
        );
    }

    #[test]
    fn unsupported_or_unregistered_benchmark_is_rejected() {
        let mut profile = heat_profile();
        profile.benchmark_id = "unknown-benchmark".into();
        let mut manifest = heat_manifest();
        manifest.scenario_id = "unknown-benchmark".into();
        let (nodes, edges) = graph();
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "benchmark_id"
        );
    }

    #[test]
    fn missing_or_wrong_scenario_evidence_is_rejected() {
        let profile = heat_profile();
        let mut manifest = heat_manifest();
        manifest.scenario_id = "drought-design-case".into();
        let (nodes, edges) = graph();
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "scenario_id"
        );

        manifest = heat_manifest();
        manifest.items[0].kind = EvidenceKind::Missing;
        manifest.items[0].content_sha256 = None;
        manifest.items[0].missing_reason = Some("not collected".into());
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "hazard_parameters.evidence_item_id"
        );
    }

    #[test]
    fn undeclared_hazard_nonfinite_parameter_and_duplicate_parameter_fail() {
        let mut profile = heat_profile();
        let manifest = heat_manifest();
        let (nodes, edges) = graph();

        profile.hazard_parameters[0].hazard = Hazard::Drought;
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "hazard_parameters.hazard"
        );

        profile = heat_profile();
        profile.hazard_parameters[0].value = f64::NAN;
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "hazard_parameters.value"
        );

        profile = heat_profile();
        profile.hazard_parameters.push(profile.hazard_parameters[0].clone());
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "hazard_parameters"
        );
    }

    #[test]
    fn unknown_failure_service_cause_or_evidence_fails_closed() {
        let manifest = heat_manifest();
        let (nodes, edges) = graph();
        let mut profile = heat_profile();
        profile.initiating_failures[0].service_id = "not-in-graph".into();
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "initiating_failures.service_id"
        );

        profile = heat_profile();
        profile.initiating_failures[0].cause = Hazard::Drought;
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "initiating_failures.cause"
        );

        profile = heat_profile();
        profile.initiating_failures[0].evidence_item_id = "not-in-manifest".into();
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "initiating_failures.evidence_item_id"
        );
    }

    #[test]
    fn compound_profile_requires_each_hazard_and_propagates_outage() {
        let mut profile = heat_profile();
        profile.benchmark_id = "heat-plus-72h-grid-outage".into();
        profile.hazard_parameters.push(HazardParameter {
            hazard: Hazard::GridOutage72Hours,
            parameter_name: "critical_asset_outage_duration".into(),
            value: 72.0,
            unit: "hour".into(),
            evidence_item_id: "outage-basis".into(),
        });
        profile.initiating_failures[0].cause = Hazard::GridOutage72Hours;

        let mut manifest = heat_manifest();
        manifest.scenario_id = "heat-plus-72h-grid-outage".into();
        let (nodes, edges) = graph();
        let result = evaluate_hazard_profile(
            &profile,
            &binding_for(&profile),
            &manifest,
            &nodes,
            &edges,
            LIVING_CITY_BENCHMARKS,
        )
        .unwrap();
        assert_eq!(
            result.dependency_assessment.failed_ids(),
            vec!["cooling-centre", "grid", "hospital"]
        );

        profile.hazard_parameters.pop();
        assert_eq!(
            evaluate_hazard_profile(
                &profile,
                &binding_for(&profile),
                &manifest,
                &nodes,
                &edges,
                LIVING_CITY_BENCHMARKS,
            )
            .unwrap_err()
            .field,
            "hazard_parameters"
        );
    }

    #[test]
    fn baseline_profile_cannot_inject_hazard_failures() {
        let mut profile = heat_profile();
        profile.benchmark_id = "baseline-operating-envelope".into();
        profile.initiating_failures.clear();
        profile.hazard_parameters.clear();
        let mut manifest = heat_manifest();
        manifest.scenario_id = "baseline-operating-envelope".into();
        let (nodes, edges) = graph();
        let result = evaluate_hazard_profile(
            &profile,
            &binding_for(&profile),
            &manifest,
            &nodes,
            &edges,
            LIVING_CITY_BENCHMARKS,
        )
        .unwrap();
        assert!(result.dependency_assessment.failed_ids().is_empty());
        assert_eq!(
            LIVING_CITY_BENCHMARKS
                .iter()
                .find(|definition| definition.id == result.benchmark_id)
                .unwrap()
                .class,
            BenchmarkClass::Baseline
        );
    }

    #[test]
    fn profile_rejects_empty_required_metadata() {
        let mut profile = heat_profile();
        profile.study_area_id.clear();
        let manifest = heat_manifest();
        let (nodes, edges) = graph();
        assert_eq!(
            evaluate_hazard_profile(&profile, &binding_for(&profile), &manifest, &nodes, &edges, LIVING_CITY_BENCHMARKS)
                .unwrap_err()
                .field,
            "study_area_id"
        );
    }
}
