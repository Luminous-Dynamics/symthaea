// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-linked package for a city-scenario metric set and its ratified floors.
//!
//! The exact canonical metric payload, the package itself, source provenance for
//! each metric, and the governance decision for the floors are all explicit.
//! A well-formed/integrity-verified package is not a physical validation or
//! community approval by itself; the ratification record must be verified and
//! the decision owner remains accountable.

use serde::{Deserialize, Serialize};

use crate::living_city::{
    CityFloors, CityMetrics, ScenarioAssessment, assess_scenario,
};
use crate::living_city_evidence::{
    EvidenceKind, EvidenceManifest, sha256_hex,
};

pub const CITY_ASSESSMENT_PACKAGE_SCHEMA_VERSION: u32 = 1;
pub const CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID: &str = "city-assessment-package";
pub const CITY_METRICS_PAYLOAD_EVIDENCE_ID: &str = "city-metrics-payload";

pub const REQUIRED_CITY_METRIC_IDS: [&str; 10] = [
    "essential_access_share",
    "housing_cost_burden_p90",
    "critical_service_uptime",
    "water_service_reliability",
    "heat_safe_access_share",
    "lifecycle_emissions_kg_co2e_per_person_year",
    "water_withdrawal_l_per_person_day",
    "habitat_connectivity_index",
    "involuntary_displacement_risk",
    "lifecycle_cost_per_person_year",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricEvidenceBinding {
    pub metric_id: String,
    pub evidence_item_ids: Vec<String>,
    /// Describe the derivation/source relationship; this is provenance metadata,
    /// not an independently verified proof that the derivation is correct.
    pub derivation_note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CityAssessmentPackage {
    pub schema_version: u32,
    pub scenario_id: String,
    pub study_area_id: String,
    pub jurisdiction: String,
    pub metrics: CityMetrics,
    pub floors: CityFloors,
    /// Evidence item whose retained bytes must equal canonical serialization of
    /// the metrics object. This prevents the embedded values being changed
    /// without changing the exact source payload.
    pub metrics_payload_evidence_item_id: String,
    pub metric_evidence: Vec<MetricEvidenceBinding>,
    /// Human/governance decision identifier; not inferred from a model score.
    pub floor_ratification_ref: String,
    pub floor_ratification_evidence_item_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CityAssessmentPackageError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for CityAssessmentPackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid city assessment package {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for CityAssessmentPackageError {}

fn invalid(field: &'static str, reason: &'static str) -> CityAssessmentPackageError {
    CityAssessmentPackageError { field, reason }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

impl CityAssessmentPackage {
    /// Canonical compact JSON representation. The retained city-assessment
    /// package file is required to contain these exact bytes, not pretty JSON.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CityAssessmentPackageError> {
        let mut canonical = self.clone();
        canonical.metric_evidence.sort_by(|a, b| a.metric_id.cmp(&b.metric_id));
        for binding in &mut canonical.metric_evidence {
            binding.evidence_item_ids.sort();
        }
        serde_json::to_vec(&canonical)
            .map_err(|_| invalid("canonical_bytes", "package serialization failed"))
    }

    pub fn sha256(&self) -> Result<String, CityAssessmentPackageError> {
        Ok(sha256_hex(&self.canonical_bytes()?))
    }

    /// Validate source lineage and the community/governance binding, then
    /// evaluate the configured non-compensatory floors.
    pub fn assess(
        &self,
        manifest: &EvidenceManifest,
    ) -> Result<ScenarioAssessment, CityAssessmentPackageError> {
        if self.schema_version != CITY_ASSESSMENT_PACKAGE_SCHEMA_VERSION {
            return Err(invalid("schema_version", "unsupported city-assessment schema version"));
        }
        for (field, value) in [
            ("scenario_id", self.scenario_id.as_str()),
            ("study_area_id", self.study_area_id.as_str()),
            ("jurisdiction", self.jurisdiction.as_str()),
            ("metrics_payload_evidence_item_id", self.metrics_payload_evidence_item_id.as_str()),
            ("floor_ratification_ref", self.floor_ratification_ref.as_str()),
            ("floor_ratification_evidence_item_id", self.floor_ratification_evidence_item_id.as_str()),
        ] {
            if !non_empty(value) {
                return Err(invalid(field, "must be explicitly provided"));
            }
        }

        manifest
            .validate()
            .map_err(|_| invalid("evidence_manifest", "evidence manifest is invalid"))?;
        if manifest.scenario_id != self.scenario_id {
            return Err(invalid(
                "scenario_id",
                "city assessment and evidence manifest must refer to the same scenario",
            ));
        }

        let expected_package_sha256 = self.sha256()?;
        let package_item = manifest
            .items
            .iter()
            .find(|item| item.id == CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID)
            .ok_or_else(|| invalid(
                "city_assessment_package_evidence",
                "manifest must bind the exact serialized city-assessment package",
            ))?;
        if package_item.kind == EvidenceKind::Missing
            || package_item.content_sha256.as_deref() != Some(expected_package_sha256.as_str())
        {
            return Err(invalid(
                "city_assessment_package_evidence",
                "manifest digest must match the canonical city-assessment package bytes",
            ));
        }

        let metrics_payload_item = manifest
            .items
            .iter()
            .find(|item| item.id == self.metrics_payload_evidence_item_id)
            .ok_or_else(|| invalid(
                "metrics_payload_evidence_item_id",
                "metrics payload evidence is absent from manifest",
            ))?;
        if metrics_payload_item.kind == EvidenceKind::Missing {
            return Err(invalid(
                "metrics_payload_evidence_item_id",
                "missing-source records cannot substantiate city metric values",
            ));
        }
        let metrics_bytes = serde_json::to_vec(&self.metrics)
            .map_err(|_| invalid("metrics", "metrics serialization failed"))?;
        let expected_metrics_sha256 = sha256_hex(&metrics_bytes);
        if metrics_payload_item.content_sha256.as_deref() != Some(expected_metrics_sha256.as_str()) {
            return Err(invalid(
                "metrics_payload_evidence_item_id",
                "retained metric payload digest must match canonical metrics values",
            ));
        }

        let ratification_item = manifest
            .items
            .iter()
            .find(|item| item.id == self.floor_ratification_evidence_item_id)
            .ok_or_else(|| invalid(
                "floor_ratification_evidence_item_id",
                "floor-ratification evidence is absent from manifest",
            ))?;
        if ratification_item.kind != EvidenceKind::GovernanceDecision {
            return Err(invalid(
                "floor_ratification_evidence_item_id",
                "locally defined floors must cite evidence classified as governance_decision",
            ));
        }
        if ratification_item.source_uri != self.floor_ratification_ref {
            return Err(invalid(
                "floor_ratification_ref",
                "ratification reference must match the referenced governance evidence URI",
            ));
        }

        validate_metric_bindings(self, manifest)?;

        assess_scenario(self.scenario_id.clone(), self.metrics, self.floors)
            .map_err(|_| invalid("metrics_or_floors", "metrics/floors failed numeric validation"))
    }
}

fn validate_metric_bindings(
    package: &CityAssessmentPackage,
    manifest: &EvidenceManifest,
) -> Result<(), CityAssessmentPackageError> {
    let mut seen = std::collections::BTreeSet::new();
    for binding in &package.metric_evidence {
        if !REQUIRED_CITY_METRIC_IDS.contains(&binding.metric_id.as_str()) {
            return Err(invalid("metric_evidence.metric_id", "unknown city metric ID"));
        }
        if !seen.insert(binding.metric_id.as_str()) {
            return Err(invalid("metric_evidence.metric_id", "each metric ID may be bound once"));
        }
        if !non_empty(&binding.derivation_note) {
            return Err(invalid("metric_evidence.derivation_note", "derivation note must be explicit"));
        }
        if binding.evidence_item_ids.is_empty() {
            return Err(invalid("metric_evidence.evidence_item_ids", "each metric needs source evidence"));
        }
        let mut ids = std::collections::BTreeSet::new();
        for evidence_id in &binding.evidence_item_ids {
            if !ids.insert(evidence_id.as_str()) {
                return Err(invalid("metric_evidence.evidence_item_ids", "duplicate evidence ID"));
            }
            if evidence_id == &package.metrics_payload_evidence_item_id
                || evidence_id == CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID
            {
                return Err(invalid(
                    "metric_evidence.evidence_item_ids",
                    "metric lineage must cite supporting inputs/outputs, not the package that merely records the value",
                ));
            }
            let item = manifest
                .items
                .iter()
                .find(|item| item.id == *evidence_id)
                .ok_or_else(|| invalid(
                    "metric_evidence.evidence_item_ids",
                    "metric source evidence ID is absent from manifest",
                ))?;
            if item.kind == EvidenceKind::Missing {
                return Err(invalid(
                    "metric_evidence.evidence_item_ids",
                    "a missing-source record cannot substantiate a metric",
                ));
            }
        }
    }
    if seen.len() != REQUIRED_CITY_METRIC_IDS.len()
        || REQUIRED_CITY_METRIC_IDS.iter().any(|id| !seen.contains(id))
    {
        return Err(invalid(
            "metric_evidence",
            "all ten decision-bearing metrics must have explicit source bindings",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::living_city_evidence::{
        EVIDENCE_SCHEMA_VERSION, EvidenceItem,
    };

    fn metrics() -> CityMetrics {
        CityMetrics {
            essential_access_share: 0.95,
            housing_cost_burden_p90: 0.25,
            critical_service_uptime: 0.9999,
            water_service_reliability: 0.999,
            heat_safe_access_share: 0.96,
            lifecycle_emissions_kg_co2e_per_person_year: 5_000.0,
            water_withdrawal_l_per_person_day: 120.0,
            habitat_connectivity_index: 0.65,
            involuntary_displacement_risk: 0.0,
            lifecycle_cost_per_person_year: 12_000.0,
        }
    }

    fn floors() -> CityFloors {
        CityFloors {
            minimum_essential_access_share: 0.90,
            maximum_housing_cost_burden_p90: 0.30,
            minimum_critical_service_uptime: 0.999,
            minimum_water_service_reliability: 0.995,
            minimum_heat_safe_access_share: 0.95,
            maximum_involuntary_displacement_risk: 0.01,
        }
    }

    fn item(id: &str, kind: EvidenceKind, bytes: &[u8]) -> EvidenceItem {
        EvidenceItem {
            id: id.into(),
            kind,
            source_uri: format!("fixture://{id}"),
            content_sha256: Some(sha256_hex(bytes)),
            license_or_basis: "synthetic test fixture".into(),
            units: "fixture-defined".into(),
            spatial_scope: "synthetic test area".into(),
            temporal_scope: "synthetic scenario".into(),
            acquired_at_utc: "2026-01-01T00:00:00Z".into(),
            transformation_chain: vec!["fixture-generator/1".into()],
            uncertainty_note: "synthetic only; not field evidence".into(),
            missing_reason: None,
        }
    }

    fn package() -> CityAssessmentPackage {
        let metrics = metrics();
        CityAssessmentPackage {
            schema_version: CITY_ASSESSMENT_PACKAGE_SCHEMA_VERSION,
            scenario_id: "fixture-city-scenario".into(),
            study_area_id: "fixture-study-area".into(),
            jurisdiction: "fixture jurisdiction".into(),
            metrics_payload_evidence_item_id: CITY_METRICS_PAYLOAD_EVIDENCE_ID.into(),
            floors: floors(),
            metrics,
            metric_evidence: REQUIRED_CITY_METRIC_IDS
                .iter()
                .map(|metric_id| MetricEvidenceBinding {
                    metric_id: (*metric_id).into(),
                    evidence_item_ids: vec!["city-model-inputs".into()],
                    derivation_note: "synthetic fixture source for package validation".into(),
                })
                .collect(),
            floor_ratification_ref: "fixture://floor-ratification".into(),
            floor_ratification_evidence_item_id: "floor-ratification".into(),
        }
    }

    fn manifest(package: &CityAssessmentPackage) -> EvidenceManifest {
        let metrics_bytes = serde_json::to_vec(&package.metrics).unwrap();
        let package_bytes = package.canonical_bytes().unwrap();
        EvidenceManifest {
            schema_version: EVIDENCE_SCHEMA_VERSION,
            project_id: "fixture-city-project".into(),
            scenario_id: package.scenario_id.clone(),
            study_boundary_id: package.study_area_id.clone(),
            coordinate_reference_system: "synthetic CRS".into(),
            items: vec![
                item(CITY_ASSESSMENT_PACKAGE_EVIDENCE_ID, EvidenceKind::Derived, &package_bytes),
                item(CITY_METRICS_PAYLOAD_EVIDENCE_ID, EvidenceKind::Simulated, &metrics_bytes),
                item("city-model-inputs", EvidenceKind::Measured, b"synthetic inputs"),
                item(
                    "floor-ratification",
                    EvidenceKind::GovernanceDecision,
                    b"synthetic governance record; not a real community authorization",
                ),
            ],
        }
    }

    #[test]
    fn canonical_package_binds_metrics_floors_and_provenance() {
        let input = package();
        assert!(input.assess(&manifest(&input)).unwrap().is_feasible());
    }

    #[test]
    fn stale_package_digest_is_rejected() {
        let input = package();
        let mut evidence = manifest(&input);
        evidence.items[0].content_sha256 = Some("a".repeat(64));
        assert_eq!(
            input.assess(&evidence).unwrap_err().field,
            "city_assessment_package_evidence"
        );
    }

    #[test]
    fn floors_must_cite_governance_decision_evidence() {
        let input = package();
        let mut evidence = manifest(&input);
        evidence.items[3].kind = EvidenceKind::Assumed;
        assert_eq!(
            input.assess(&evidence).unwrap_err().field,
            "floor_ratification_evidence_item_id"
        );
    }

    #[test]
    fn ratification_reference_must_match_governance_evidence_source() {
        let mut input = package();
        input.floor_ratification_ref = "fixture://unrelated-record".into();
        let evidence = manifest(&input);
        assert_eq!(
            input.assess(&evidence).unwrap_err().field,
            "floor_ratification_ref"
        );
    }

    #[test]
    fn every_metric_needs_non_missing_lineage() {
        let mut input = package();
        input.metric_evidence.pop();
        assert_eq!(
            input.assess(&manifest(&input)).unwrap_err().field,
            "metric_evidence"
        );

        input = package();
        input.metric_evidence[0].evidence_item_ids = vec!["not-listed".into()];
        assert_eq!(
            input.assess(&manifest(&input)).unwrap_err().field,
            "metric_evidence.evidence_item_ids"
        );
    }

    #[test]
    fn infeasible_metrics_are_a_valid_assessment_but_fail_the_floors() {
        let mut input = package();
        input.metrics.essential_access_share = 0.60;
        let assessment = input.assess(&manifest(&input)).unwrap();
        assert!(!assessment.is_feasible());
        assert!(assessment.violations().iter().any(|violation| violation.metric == "essential_access_share"));
    }
}
