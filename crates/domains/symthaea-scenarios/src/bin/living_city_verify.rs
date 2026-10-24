// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline integrity verifier for a retained Living City run package.
//!
//! Usage: living_city_verify <run-directory>
//!
//! The directory contains bundle.json, a typed evidence manifest, a typed run
//! receipt, a receipt digest sidecar, and the source/output files referenced by
//! bundle.json. File paths must remain within the canonical run directory.
//!
//! A successful report means only that the package is internally consistent and
//! its retained bytes match the declared digests. It does not authenticate the
//! author, validate physics, certify code compliance, or approve construction.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use symthaea_scenarios::living_city_assessment_package::{
    CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID, CITY_ASSESSMENT_PACKAGE_SCHEMA_VERSION,
    CITY_METRICS_PAYLOAD_EVIDENCE_ID, CityAssessmentPackage,
};
use symthaea_scenarios::living_city_benchmarks::{
    BenchmarkProfileBinding, LIVING_CITY_BENCHMARKS,
};
use symthaea_scenarios::living_city_dependencies::{
    ServiceDependency, ServiceNode,
};
use symthaea_scenarios::living_city_evidence::{
    EVIDENCE_SCHEMA_VERSION, RECEIPT_SCHEMA_VERSION, EvidenceItem, EvidenceKind,
    EvidenceManifest, GateStatus, OutputArtifact, ScenarioRunReceipt, ToolExecution,
    ToolRunStatus, ValidationGate, evidence_manifest_sha256, sha256_hex,
    sha256_reader_hex, scenario_run_receipt_sha256, verify_manifest_source_digests,
    verify_receipt_manifest_link, verify_receipt_output_digests,
    verify_scenario_run_receipt_digest,
};
use symthaea_scenarios::living_city_hazard_profiles::{
    HazardProfile, evaluate_hazard_profile,
};
use symthaea_scenarios::living_city_grid_reference::{
    GRID_REFERENCE_CASE_IDS, GRID_REFERENCE_SOLVER_ENVELOPE, GRID_REFERENCE_SUITE_ID,
    GRID_REFERENCE_SUITE_VERSION, GRID_REFERENCE_VALIDATION_KIND, run_grid_reference_suite,
};

const BUNDLE_SCHEMA_VERSION: u32 = 3;
const SERVICE_GRAPH_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VerificationBundle {
    schema_version: u32,
    manifest_path: String,
    receipt_path: String,
    receipt_sha256_path: String,
    hazard_profile_path: String,
    profile_binding_path: String,
    service_graph_path: String,
    city_assessment_path: String,
    source_files: Vec<FileBinding>,
    output_files: Vec<FileBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceGraphPackage {
    schema_version: u32,
    nodes: Vec<ServiceNode>,
    dependencies: Vec<ServiceDependency>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileBinding {
    id: String,
    path: String,
}

fn canonical_service_graph_bytes(
    graph: &ServiceGraphPackage,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut canonical = graph.clone();
    canonical.nodes.sort_by(|left, right| left.id.cmp(&right.id));
    canonical.dependencies.sort();
    Ok(serde_json::to_vec(&canonical)?)
}

/// Bind the profile, selected profile binding, and service graph into the
/// receipt's configuration digest. Length prefixes make component boundaries
/// unambiguous. The evidence-manifest digest remains a separate receipt field.
fn scenario_configuration_sha256(
    profile: &HazardProfile,
    binding: &BenchmarkProfileBinding,
    graph: &ServiceGraphPackage,
    city_assessment: &CityAssessmentPackage,
) -> Result<String, Box<dyn Error>> {
    let profile_bytes = profile.canonical_bytes()?;
    let binding_bytes = serde_json::to_vec(binding)?;
    let graph_bytes = canonical_service_graph_bytes(graph)?;
    let city_assessment_bytes = city_assessment.canonical_bytes()?;
    // Pin the built-in solver reference suite/case identity into the same
    // config digest that binds the scenario profile and dependency graph.
    let reference_suite = format!(
        "{}:{}:{}",
        GRID_REFERENCE_SUITE_VERSION,
        GRID_REFERENCE_SUITE_ID,
        GRID_REFERENCE_CASE_IDS.join(",")
    )
    .into_bytes();
    let mut bytes = Vec::new();
    for component in [
        &profile_bytes,
        &binding_bytes,
        &graph_bytes,
        &city_assessment_bytes,
        &reference_suite,
    ] {
        bytes.extend_from_slice(&(component.len() as u64).to_le_bytes());
        bytes.extend_from_slice(component);
    }
    Ok(sha256_hex(&bytes))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn safe_file_path(root: &Path, relative_path: &str) -> io::Result<PathBuf> {
    let relative = Path::new(relative_path);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(invalid_input(
            "package file path must be a non-empty relative path without parent traversal",
        ));
    }

    let candidate = root.join(relative);
    let resolved = fs::canonicalize(candidate)?;
    if !resolved.starts_with(root) {
        return Err(invalid_input(
            "package file resolves outside the run directory (path traversal or external symlink)",
        ));
    }
    if !resolved.is_file() {
        return Err(invalid_input("package path must resolve to a regular file"));
    }
    Ok(resolved)
}

fn read_package_file(root: &Path, relative_path: &str) -> io::Result<Vec<u8>> {
    fs::read(safe_file_path(root, relative_path)?)
}

fn read_json<T: for<'de> Deserialize<'de>>(
    root: &Path,
    relative_path: &str,
) -> Result<T, Box<dyn Error>> {
    let bytes = read_package_file(root, relative_path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn compute_file_digest_map(
    root: &Path,
    bindings: &[FileBinding],
    collection_name: &'static str,
) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut digests = BTreeMap::new();
    for binding in bindings {
        if binding.id.trim().is_empty() {
            return Err(invalid_input(format!(
                "{collection_name} contains an empty artifact ID"
            ))
            .into());
        }
        if digests.contains_key(&binding.id) {
            return Err(invalid_input(format!(
                "{collection_name} contains duplicate artifact ID {:?}",
                binding.id
            ))
            .into());
        }

        let path = safe_file_path(root, &binding.path)?;
        let file = File::open(path)?;
        let actual_sha256 = sha256_reader_hex(file)?;
        digests.insert(binding.id.clone(), actual_sha256);
    }
    Ok(digests)
}

fn verify_run_package(run_directory: &Path) -> Result<serde_json::Value, Box<dyn Error>> {
    let root = fs::canonicalize(run_directory)?;
    if !root.is_dir() {
        return Err(invalid_input("run-directory must be a directory").into());
    }

    let bundle: VerificationBundle = read_json(&root, "bundle.json")?;
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION {
        return Err(invalid_input("unsupported verification bundle schema version").into());
    }

    let manifest: EvidenceManifest = read_json(&root, &bundle.manifest_path)?;
    let receipt: ScenarioRunReceipt = read_json(&root, &bundle.receipt_path)?;
    let profile: HazardProfile = read_json(&root, &bundle.hazard_profile_path)?;
    let profile_binding: BenchmarkProfileBinding =
        read_json(&root, &bundle.profile_binding_path)?;
    let service_graph: ServiceGraphPackage =
        read_json(&root, &bundle.service_graph_path)?;
    let city_assessment: CityAssessmentPackage =
        read_json(&root, &bundle.city_assessment_path)?;

    if service_graph.schema_version != SERVICE_GRAPH_SCHEMA_VERSION {
        return Err(invalid_input("unsupported service-graph schema version").into());
    }
    manifest.validate()?;
    receipt.validate()?;

    let receipt_digest_sidecar = read_package_file(&root, &bundle.receipt_sha256_path)?;
    let expected_receipt_sha256 = std::str::from_utf8(&receipt_digest_sidecar)?.trim();
    verify_scenario_run_receipt_digest(&receipt, expected_receipt_sha256)?;
    verify_receipt_manifest_link(&receipt, &manifest)?;

    let source_digests =
        compute_file_digest_map(&root, &bundle.source_files, "source_files")?;
    verify_manifest_source_digests(&manifest, &source_digests)?;

    let output_digests =
        compute_file_digest_map(&root, &bundle.output_files, "output_files")?;
    verify_receipt_output_digests(&receipt, &output_digests)?;

    // Recompute the non-compensatory city feasibility result from the exact
    // retained assessment package and its source-linked manifest, rather than
    // trusting a generator-supplied "feasible" flag.
    if city_assessment.schema_version != CITY_ASSESSMENT_PACKAGE_SCHEMA_VERSION {
        return Err(invalid_input("unsupported city-assessment package schema version").into());
    }
    if !source_digests.contains_key(CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID)
        || !source_digests.contains_key(CITY_METRICS_PAYLOAD_EVIDENCE_ID)
    {
        return Err(invalid_input(
            "bundle must retain both the canonical city-assessment package and its exact metrics payload",
        )
        .into());
    }
    let city_assessment_result = city_assessment.assess(&manifest)?;

    // The profile's input evidence is not considered usable until all retained
    // source bytes passed the independent digest checks above.
    let cascade = evaluate_hazard_profile(
        &profile,
        &profile_binding,
        &manifest,
        &service_graph.nodes,
        &service_graph.dependencies,
        LIVING_CITY_BENCHMARKS,
    )?;
    let actual_configuration_sha256 = scenario_configuration_sha256(
        &profile,
        &profile_binding,
        &service_graph,
        &city_assessment,
    )?;
    if actual_configuration_sha256 != receipt.configuration_sha256 {
        return Err(invalid_input(
            "receipt configuration digest does not match canonical hazard profile, binding and service graph",
        )
        .into());
    }

    let manifest_sha256 = evidence_manifest_sha256(&manifest)?;
    let receipt_sha256 = scenario_run_receipt_sha256(&receipt)?;
    let missing_sources: Vec<&str> = manifest
        .items
        .iter()
        .filter(|item| {
            item.kind
                == symthaea_scenarios::living_city_evidence::EvidenceKind::Missing
        })
        .map(|item| item.id.as_str())
        .collect();
    let failed_services = cascade.dependency_assessment.failed_ids();
    let degraded_services = cascade.dependency_assessment.degraded_ids();
    let graph_sha256 = sha256_hex(&canonical_service_graph_bytes(&service_graph)?);
    let grid_reference_suite = match run_grid_reference_suite() {
        Ok(results) => serde_json::json!({
            "status": "passed",
            "suite_version": GRID_REFERENCE_SUITE_VERSION,
            "suite_id": GRID_REFERENCE_SUITE_ID,
            "validation_kind": GRID_REFERENCE_VALIDATION_KIND,
            "solver_envelope": GRID_REFERENCE_SOLVER_ENVELOPE,
            "declared_case_ids": GRID_REFERENCE_CASE_IDS,
            "case_count": results.len(),
            "cases": results.iter().map(|result| serde_json::json!({
                "case_id": result.case_id,
                "validation_kind": result.validation_kind,
                "solver_envelope": result.solver_envelope,
                "analytical_voltage_v": result.analytical_voltage_v,
                "solver_voltage_v": result.solver_voltage_v,
                "absolute_voltage_error_v": result.absolute_voltage_error_v,
                "expected_branch_power_kw": result.expected_branch_power_kw,
                "solver_branch_power_kw": result.solver_branch_power_kw,
                "absolute_branch_power_error_kw": result.absolute_branch_power_error_kw,
                "checked_bus_count": result.checked_bus_count,
                "max_absolute_voltage_error_v": result.max_absolute_voltage_error_v,
                "max_absolute_branch_power_error_kw": result.max_absolute_branch_power_error_kw
            })).collect::<Vec<_>>(),
            "scope": "Four analytical linearized DistFlow regression cases (real load, reactive load, two-segment radial feeder, and DER voltage rise). This is not nonlinear AC power-flow validation or a real-outage simulation."
        }),
        Err(error) => serde_json::json!({
            "status": "failed",
            "suite_version": GRID_REFERENCE_SUITE_VERSION,
            "suite_id": GRID_REFERENCE_SUITE_ID,
            "case_id": error.case_id,
            "field": error.field,
            "reason": error.to_string(),
            "scope": "Four analytical linearized DistFlow regression cases (real load, reactive load, two-segment radial feeder, and DER voltage rise). This is not nonlinear AC power-flow validation or a real-outage simulation."
        }),
    };
    let grid_reference_passed = grid_reference_suite["status"] == "passed";
    let city_feasible = city_assessment_result.is_feasible();
    let promotion_state = match receipt.promotion_state().as_str() {
        "failed" => "failed",
        "blocked" => "blocked",
        _ if !city_feasible => "failed",
        _ if grid_reference_passed => "eligible_for_next_workflow_stage",
        _ => "failed",
    };
    let city_violations: Vec<serde_json::Value> = city_assessment_result
        .violations()
        .iter()
        .map(|violation| {
            serde_json::json!({
                "metric_id": violation.metric,
                "actual": violation.actual,
                "bound": match violation.bound {
                    symthaea_scenarios::living_city::ConstraintBound::AtLeast => "at_least",
                    symthaea_scenarios::living_city::ConstraintBound::AtMost => "at_most",
                },
                "threshold": violation.threshold,
            })
        })
        .collect();
    let city_design_feasibility = serde_json::json!({
        "status": if city_feasible { "feasible" } else { "infeasible" },
        "scenario_id": city_assessment_result.scenario_id(),
        "study_area_id": city_assessment.study_area_id,
        "jurisdiction": city_assessment.jurisdiction,
        "floor_ratification_ref": city_assessment.floor_ratification_ref,
        "metric_evidence": city_assessment.metric_evidence,
        "metrics": city_assessment.metrics,
        "floors": city_assessment.floors,
        "violations": city_violations,
        "scope": "Constraint evaluation of evidence-linked supplied metrics against explicitly ratified floors. This does not independently validate the physical models or establish community authorization beyond the retained ratification record."
    });

    Ok(serde_json::json!({
        "schema_version": 3,
        "package_integrity": "verified",
        "run_id": receipt.run_id,
        "scenario_id": receipt.scenario_id,
        "repository": receipt.repository,
        "commit_sha": receipt.commit_sha,
        "manifest_sha256": manifest_sha256,
        "receipt_sha256": receipt_sha256,
        "configuration_sha256": actual_configuration_sha256,
        "workflow_promotion_state": promotion_state,
        "city_design_feasibility": city_design_feasibility,
        "domain_reference_suite": grid_reference_suite,
        "hazard_profile_id": cascade.profile_id,
        "hazard_profile_sha256": cascade.profile_sha256,
        "service_graph_sha256": graph_sha256,
        "dependency_propagation_rounds": cascade.dependency_assessment.propagation_rounds,
        "failed_service_ids": failed_services,
        "degraded_service_ids": degraded_services,
        "verified_source_count": source_digests.len(),
        "verified_output_count": output_digests.len(),
        "explicit_missing_source_ids": missing_sources,
        "scope": "Package integrity, profile/manifest binding and declared dependency propagation only. This report does not authenticate receipt authorship, validate physical models or solver claims, establish jurisdictional code compliance, replace professional review, or authorize construction."
    }))
}
fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let run_directory = match (args.next(), args.next()) {
        (Some(path), None) => PathBuf::from(path),
        _ => {
            return Err(invalid_input("usage: living_city_verify <run-directory>").into());
        }
    };

    let report = verify_run_package(&run_directory)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("living_city_verify: verification failed: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_scenarios::living_city::{CityFloors, CityMetrics};
    use symthaea_scenarios::living_city_assessment_package::REQUIRED_CITY_METRIC_IDS;
    use symthaea_scenarios::living_city_assessment_package::{
        CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID, CITY_METRICS_PAYLOAD_EVIDENCE_ID,
        CityAssessmentPackage, MetricEvidenceBinding, REQUIRED_CITY_METRIC_IDS,
    };
    use symthaea_scenarios::living_city_benchmarks::Hazard;
    use symthaea_scenarios::living_city_dependencies::{DependencyMode, ServiceDomain};
    use symthaea_scenarios::living_city_hazard_profiles::{
        HAZARD_PROFILE_SCHEMA_VERSION, HazardParameter, InitiatingFailure,
    };

    #[test]
    fn verifier_rejects_empty_absolute_and_parent_traversal_paths() {
        let root = Path::new("/does/not/need/to/exist/for-this-guard-test");
        assert!(safe_file_path(root, "").is_err());
        assert!(safe_file_path(root, "/etc/passwd").is_err());
        assert!(safe_file_path(root, "../outside.txt").is_err());
        assert!(safe_file_path(root, "subdir/../../outside.txt").is_err());
    }

    #[test]
    fn bundle_parser_rejects_unknown_fields() {
        let bundle = r#"{
            "schema_version": 3,
            "manifest_path": "manifest.json",
            "receipt_path": "receipt.json",
            "receipt_sha256_path": "receipt.sha256",
            "hazard_profile_path": "hazard-profile.json",
            "profile_binding_path": "profile-binding.json",
            "service_graph_path": "service-graph.json",
            "city_assessment_path": "city-assessment.json",
            "source_files": [],
            "output_files": [],
            "unexpected_authority": "trusted"
        }"#;
        assert!(serde_json::from_str::<VerificationBundle>(bundle).is_err());
    }

    fn write_test_package(essential_access_share: f64) -> PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let root = env::temp_dir().join(format!(
            "symthaea-living-city-verify-{}-{unique}",
            process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("sources")).unwrap();
        fs::create_dir_all(root.join("outputs")).unwrap();

        let heat_bytes = b"synthetic heat design basis";
        let outage_bytes = b"synthetic correlated outage basis";
        let city_model_bytes = b"synthetic model evidence; not field validation";
        let floor_ratification_bytes =
            b"synthetic governance test record; not a real community authorization";
        let output_bytes = b"synthetic living-city solver output bytes";
        fs::write(root.join("sources/heat.dat"), heat_bytes).unwrap();
        fs::write(root.join("sources/outage.dat"), outage_bytes).unwrap();
        fs::write(root.join("sources/city-model-output.dat"), city_model_bytes).unwrap();
        fs::write(
            root.join("sources/floor-ratification.dat"),
            floor_ratification_bytes,
        )
        .unwrap();
        fs::write(root.join("outputs/metrics.dat"), output_bytes).unwrap();

        let metrics = CityMetrics {
            essential_access_share,
            housing_cost_burden_p90: 0.25,
            critical_service_uptime: 0.9999,
            water_service_reliability: 0.999,
            heat_safe_access_share: 0.96,
            lifecycle_emissions_kg_co2e_per_person_year: 5_000.0,
            water_withdrawal_l_per_person_day: 120.0,
            habitat_connectivity_index: 0.65,
            involuntary_displacement_risk: 0.0,
            lifecycle_cost_per_person_year: 12_000.0,
        };
        let metrics_payload_bytes = serde_json::to_vec(&metrics).unwrap();
        fs::write(
            root.join("sources/city-metrics.json"),
            &metrics_payload_bytes,
        )
        .unwrap();

        let city_assessment = CityAssessmentPackage {
            schema_version: CITY_ASSESSMENT_PACKAGE_SCHEMA_VERSION,
            scenario_id: "heat-plus-72h-grid-outage".into(),
            study_area_id: "synthetic-area".into(),
            jurisdiction: "test jurisdiction".into(),
            metrics,
            floors: CityFloors {
                minimum_essential_access_share: 0.90,
                maximum_housing_cost_burden_p90: 0.30,
                minimum_critical_service_uptime: 0.999,
                minimum_water_service_reliability: 0.995,
                minimum_heat_safe_access_share: 0.95,
                maximum_involuntary_displacement_risk: 0.01,
            },
            metrics_payload_evidence_item_id: CITY_METRICS_PAYLOAD_EVIDENCE_ID.into(),
            metric_evidence: REQUIRED_CITY_METRIC_IDS
                .iter()
                .map(|metric_id| MetricEvidenceBinding {
                    metric_id: (*metric_id).to_owned(),
                    evidence_item_ids: vec!["city-model-output".into()],
                    derivation_note: "synthetic test lineage; not a validated physical derivation".into(),
                })
                .collect(),
            floor_ratification_ref: "fixture://floor-ratification".into(),
            floor_ratification_evidence_item_id: "floor-ratification".into(),
        };
        let city_assessment_bytes = city_assessment.canonical_bytes().unwrap();
        fs::write(root.join("city-assessment.json"), &city_assessment_bytes).unwrap();

        let mut manifest_items = Vec::new();
        manifest_items.push(EvidenceItem {
            id: "heat-basis".into(),
            kind: EvidenceKind::Assumed,
            source_uri: "fixture://heat-basis".into(),
            content_sha256: Some(sha256_hex(heat_bytes)),
            license_or_basis: "synthetic test fixture assumption".into(),
            units: "degree_Celsius".into(),
            spatial_scope: "synthetic site".into(),
            temporal_scope: "synthetic heat design event".into(),
            acquired_at_utc: "2026-01-01T00:00:00Z".into(),
            transformation_chain: vec!["fixture-generator/1".into()],
            uncertainty_note: "assumption only; not field evidence".into(),
            missing_reason: None,
        });
        manifest_items.push(EvidenceItem {
            id: "outage-basis".into(),
            kind: EvidenceKind::Assumed,
            source_uri: "fixture://outage-basis".into(),
            content_sha256: Some(sha256_hex(outage_bytes)),
            license_or_basis: "synthetic test fixture assumption".into(),
            units: "hour".into(),
            spatial_scope: "synthetic critical services".into(),
            temporal_scope: "synthetic 72-hour outage".into(),
            acquired_at_utc: "2026-01-01T00:00:00Z".into(),
            transformation_chain: vec!["fixture-generator/1".into()],
            uncertainty_note: "assumption only; not field evidence".into(),
            missing_reason: None,
        });
        manifest_items.push(EvidenceItem {
            id: "city-model-output".into(),
            kind: EvidenceKind::Simulated,
            source_uri: "fixture://city-model-output".into(),
            content_sha256: Some(sha256_hex(city_model_bytes)),
            license_or_basis: "synthetic test fixture".into(),
            units: "mixed; metric schemas declare units".into(),
            spatial_scope: "synthetic study area".into(),
            temporal_scope: "synthetic design scenario".into(),
            acquired_at_utc: "2026-01-01T00:00:00Z".into(),
            transformation_chain: vec!["fixture-generator/1".into()],
            uncertainty_note: "simulated fixture only; not real-city evidence".into(),
            missing_reason: None,
        });
        manifest_items.push(EvidenceItem {
            id: "floor-ratification".into(),
            kind: EvidenceKind::GovernanceDecision,
            source_uri: "fixture://floor-ratification".into(),
            content_sha256: Some(sha256_hex(floor_ratification_bytes)),
            license_or_basis: "synthetic test fixture only".into(),
            units: "decision record".into(),
            spatial_scope: "synthetic study area".into(),
            temporal_scope: "synthetic approval event".into(),
            acquired_at_utc: "2026-01-01T00:00:00Z".into(),
            transformation_chain: vec!["fixture-generator/1".into()],
            uncertainty_note: "does not represent real community consent or lawful ratification".into(),
            missing_reason: None,
        });
        manifest_items.push(EvidenceItem {
            id: CITY_METRICS_PAYLOAD_EVIDENCE_ID.into(),
            kind: EvidenceKind::Simulated,
            source_uri: "fixture://city-metrics-payload".into(),
            content_sha256: Some(sha256_hex(&metrics_payload_bytes)),
            license_or_basis: "canonical synthetic metric payload".into(),
            units: "units embedded in metric schema".into(),
            spatial_scope: "synthetic study area".into(),
            temporal_scope: "synthetic design scenario".into(),
            acquired_at_utc: "2026-01-01T00:00:00Z".into(),
            transformation_chain: vec!["serde-json-canonical-metrics/1".into()],
            uncertainty_note: "synthetic fixture values only".into(),
            missing_reason: None,
        });
        manifest_items.push(EvidenceItem {
            id: CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID.into(),
            kind: EvidenceKind::Derived,
            source_uri: "fixture://city-assessment-package".into(),
            content_sha256: Some(sha256_hex(&city_assessment_bytes)),
            license_or_basis: "canonical synthetic assessment package".into(),
            units: "typed city-assessment schema".into(),
            spatial_scope: "synthetic study area".into(),
            temporal_scope: "synthetic design scenario".into(),
            acquired_at_utc: "2026-01-01T00:00:00Z".into(),
            transformation_chain: vec!["living-city-assessment-package/1".into()],
            uncertainty_note: "synthetic fixture package; not a real city assessment".into(),
            missing_reason: None,
        });
        let manifest = EvidenceManifest {
            schema_version: EVIDENCE_SCHEMA_VERSION,
            project_id: "integration-test-project".into(),
            scenario_id: "heat-plus-72h-grid-outage".into(),
            study_boundary_id: "synthetic-boundary".into(),
            coordinate_reference_system: "synthetic-local-crs".into(),
            items: manifest_items,
        };
        let manifest_digest = evidence_manifest_sha256(&manifest).unwrap();
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let profile = HazardProfile {
            schema_version: HAZARD_PROFILE_SCHEMA_VERSION,
            profile_id: "synthetic-compound-profile".into(),
            profile_revision: "r1".into(),
            benchmark_id: "heat-plus-72h-grid-outage".into(),
            study_area_id: "synthetic-area".into(),
            jurisdiction: "test jurisdiction".into(),
            time_horizon: "synthetic 72-hour stress window".into(),
            hazard_parameters: vec![
                HazardParameter {
                    hazard: Hazard::ExtremeHeat,
                    parameter_name: "design_dry_bulb_temperature".into(),
                    value: 42.0,
                    unit: "degree_Celsius".into(),
                    evidence_item_id: "heat-basis".into(),
                },
                HazardParameter {
                    hazard: Hazard::GridOutage72Hours,
                    parameter_name: "critical_asset_outage_duration".into(),
                    value: 72.0,
                    unit: "hour".into(),
                    evidence_item_id: "outage-basis".into(),
                },
            ],
            initiating_failures: vec![InitiatingFailure {
                service_id: "grid".into(),
                cause: Hazard::GridOutage72Hours,
                evidence_item_id: "outage-basis".into(),
                rationale: "synthetic test condition; not a real outage prediction".into(),
            }],
        };
        let profile_binding = profile.binding().unwrap();
        fs::write(
            root.join("hazard-profile.json"),
            serde_json::to_vec_pretty(&profile).unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("profile-binding.json"),
            serde_json::to_vec_pretty(&profile_binding).unwrap(),
        )
        .unwrap();

        let service_graph = ServiceGraphPackage {
            schema_version: SERVICE_GRAPH_SCHEMA_VERSION,
            nodes: vec![
                ServiceNode {
                    id: "grid".into(),
                    domain: ServiceDomain::Electricity,
                },
                ServiceNode {
                    id: "cooling-centre".into(),
                    domain: ServiceDomain::Cooling,
                },
                ServiceNode {
                    id: "hospital".into(),
                    domain: ServiceDomain::Healthcare,
                },
            ],
            dependencies: vec![
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
            ],
        };
        fs::write(
            root.join("service-graph.json"),
            serde_json::to_vec_pretty(&service_graph).unwrap(),
        )
        .unwrap();
        let configuration_sha256 = scenario_configuration_sha256(
            &profile,
            &profile_binding,
            &service_graph,
            &city_assessment,
        )
        .unwrap();

        let output_digest = sha256_hex(output_bytes);
        let receipt = ScenarioRunReceipt {
            schema_version: RECEIPT_SCHEMA_VERSION,
            run_id: format!("integration-run-{essential_access_share:.2}"),
            scenario_id: "heat-plus-72h-grid-outage".into(),
            repository: "Luminous-Dynamics/symthaea".into(),
            commit_sha: "b".repeat(40),
            dependency_lock_sha256: "1".repeat(64),
            build_profile: "test-fixture".into(),
            toolchain: "test-fixture".into(),
            evidence_manifest_sha256: manifest_digest,
            configuration_sha256,
            seed: Some(7),
            tools: vec![ToolExecution {
                tool_id: "synthetic-model".into(),
                version: "fixture/1".into(),
                executable_sha256: Some("3".repeat(64)),
                arguments_sha256: "4".repeat(64),
                status: ToolRunStatus::Succeeded,
            }],
            outputs: vec![OutputArtifact {
                id: "metrics".into(),
                media_type: "application/octet-stream".into(),
                sha256: output_digest.clone(),
            }],
            gates: vec![ValidationGate {
                gate_id: "artifact-integrity".into(),
                required: true,
                status: GateStatus::Pass,
                reason: None,
                not_applicable_approval_ref: None,
                verifier_id: Some("test-verifier/1".into()),
                evidence_sha256: Some(output_digest),
            }],
        };
        let receipt_digest = scenario_run_receipt_sha256(&receipt).unwrap();
        fs::write(
            root.join("receipt.json"),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
        fs::write(root.join("receipt.sha256"), format!("{receipt_digest}\n")).unwrap();

        let bundle = serde_json::json!({
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "manifest_path": "manifest.json",
            "receipt_path": "receipt.json",
            "receipt_sha256_path": "receipt.sha256",
            "hazard_profile_path": "hazard-profile.json",
            "profile_binding_path": "profile-binding.json",
            "service_graph_path": "service-graph.json",
            "city_assessment_path": "city-assessment.json",
            "source_files": [
                {"id": "heat-basis", "path": "sources/heat.dat"},
                {"id": "outage-basis", "path": "sources/outage.dat"},
                {"id": "city-model-output", "path": "sources/city-model-output.dat"},
                {"id": "floor-ratification", "path": "sources/floor-ratification.dat"},
                {"id": CITY_METRICS_PAYLOAD_EVIDENCE_ID, "path": "sources/city-metrics.json"},
                {"id": CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID, "path": "city-assessment.json"}
            ],
            "output_files": [
                {"id": "metrics", "path": "outputs/metrics.dat"}
            ]
        });
        fs::write(
            root.join("bundle.json"),
            serde_json::to_vec_pretty(&bundle).unwrap(),
        )
        .unwrap();
        root
    }

    fn write_valid_test_package() -> PathBuf {
        write_test_package(0.95)
    }

    #[test]
    fn valid_run_package_verifies_end_to_end() {
        let root = write_valid_test_package();
        let report = verify_run_package(&root).unwrap();
        assert_eq!(report["package_integrity"], "verified");
        assert_eq!(report["verified_source_count"], 6);
        assert_eq!(report["verified_output_count"], 1);
        assert_eq!(report["city_design_feasibility"]["status"], "feasible");
        assert_eq!(
            report["city_design_feasibility"]["scenario_id"],
            "heat-plus-72h-grid-outage"
        );
        assert_eq!(report["city_design_feasibility"]["violations"], serde_json::json!([]));
        assert_eq!(report["hazard_profile_id"], "synthetic-compound-profile/r1");
        assert_eq!(report["domain_reference_suite"]["status"], "passed");
        assert_eq!(
            report["domain_reference_suite"]["case_count"],
            GRID_REFERENCE_CASE_IDS.len()
        );
        assert_eq!(
            report["domain_reference_suite"]["declared_case_ids"],
            serde_json::json!(GRID_REFERENCE_CASE_IDS)
        );
        assert_eq!(
            report["domain_reference_suite"]["validation_kind"],
            GRID_REFERENCE_VALIDATION_KIND
        );
        assert_eq!(
            report["domain_reference_suite"]["solver_envelope"],
            GRID_REFERENCE_SOLVER_ENVELOPE
        );
        assert_eq!(
            report["workflow_promotion_state"],
            "eligible_for_next_workflow_stage"
        );
        assert_eq!(report["failed_service_ids"], serde_json::json!(["cooling-centre", "grid", "hospital"]));
        assert!(report["dependency_propagation_rounds"].as_u64().unwrap() >= 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn infeasible_city_floors_block_promotion_even_when_artifacts_are_valid() {
        let root = write_test_package(0.60);
        let report = verify_run_package(&root).unwrap();
        assert_eq!(report["package_integrity"], "verified");
        assert_eq!(report["city_design_feasibility"]["status"], "infeasible");
        assert_eq!(report["workflow_promotion_state"], "failed");
        assert!(
            report["city_design_feasibility"]["violations"]
                .as_array()
                .unwrap()
                .iter()
                .any(|violation| violation["metric_id"] == "essential_access_share")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tampered_metrics_payload_is_rejected_before_feasibility_promotion() {
        let root = write_valid_test_package();
        let path = root.join("sources/city-metrics.json");
        let bytes = fs::read(&path).unwrap();
        let mut metrics: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        metrics["essential_access_share"] = serde_json::Value::from(0.60);
        fs::write(&path, serde_json::to_vec(&metrics).unwrap()).unwrap();
        assert!(verify_run_package(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn profile_tampering_is_rejected_by_end_to_end_verifier() {
        let root = write_valid_test_package();
        let path = root.join("hazard-profile.json");
        let bytes = fs::read(&path).unwrap();
        let mut profile: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        profile["hazard_parameters"][0]["value"] = serde_json::Value::from(45.0);
        fs::write(&path, serde_json::to_vec_pretty(&profile).unwrap()).unwrap();
        assert!(verify_run_package(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn changed_service_graph_is_rejected_by_configuration_binding() {
        let root = write_valid_test_package();
        let path = root.join("service-graph.json");
        let bytes = fs::read(&path).unwrap();
        let mut graph: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        graph["nodes"].as_array_mut().unwrap().push(serde_json::json!({
            "id": "unbound-extra-service",
            "domain": "communications"
        }));
        fs::write(&path, serde_json::to_vec_pretty(&graph).unwrap()).unwrap();
        assert!(verify_run_package(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn corrupt_output_is_rejected_by_end_to_end_verifier() {
        let root = write_valid_test_package();
        fs::write(root.join("outputs/metrics.dat"), b"tampered metrics").unwrap();
        assert!(verify_run_package(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn graph_collection_order_does_not_change_the_configuration_digest() {
        let root = write_valid_test_package();
        let path = root.join("service-graph.json");
        let bytes = fs::read(&path).unwrap();
        let mut graph: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        graph["nodes"].as_array_mut().unwrap().reverse();
        graph["dependencies"].as_array_mut().unwrap().reverse();
        fs::write(&path, serde_json::to_vec_pretty(&graph).unwrap()).unwrap();
        let report = verify_run_package(&root).unwrap();
        assert_eq!(report["package_integrity"], "verified");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn package_path_escape_is_rejected_end_to_end() {
        let root = write_valid_test_package();
        let bundle_path = root.join("bundle.json");
        let bytes = fs::read(&bundle_path).unwrap();
        let mut bundle: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        bundle["source_files"][0]["path"] = serde_json::Value::String("../outside.dat".into());
        fs::write(&bundle_path, serde_json::to_vec_pretty(&bundle).unwrap()).unwrap();
        assert!(verify_run_package(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn receipt_digest_sidecar_mismatch_is_rejected() {
        let root = write_valid_test_package();
        fs::write(root.join("receipt.sha256"), format!("{}\n", "0".repeat(64))).unwrap();
        assert!(verify_run_package(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bundle_parser_accepts_the_declared_schema() {
        let bundle = r#"{
            "schema_version": 3,
            "manifest_path": "manifest.json",
            "receipt_path": "receipt.json",
            "receipt_sha256_path": "receipt.sha256",
            "hazard_profile_path": "hazard-profile.json",
            "profile_binding_path": "profile-binding.json",
            "service_graph_path": "service-graph.json",
            "city_assessment_path": "city-assessment.json",
            "source_files": [{"id": "heat-basis", "path": "sources/heat.geojson"}],
            "output_files": [{"id": "metrics", "path": "outputs/metrics.json"}]
        }"#;
        let parsed = serde_json::from_str::<VerificationBundle>(bundle).unwrap();
        assert_eq!(parsed.schema_version, BUNDLE_SCHEMA_VERSION);
        assert_eq!(parsed.source_files.len(), 1);
        assert_eq!(parsed.output_files.len(), 1);
    }
}
