// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_maritime_core::{
    DependencyGovernance, RegenerativeCapability, RegenerativeClosureModel,
    RegenerativeDependency, RegenerativeDependencyKind, RegenerativeGenomeRequirementV1,
    RegenerativeGenomeV1, RegenerativeHorizon, RegenerativeSubstitutionEvidenceV1,
    REGENERATIVE_GENOME_SCHEMA_V1, REGENERATIVE_SUBSTITUTION_SCHEMA_V1,
};

const CAPABILITY_ID: &str = "persistent-ocean-infrastructure";
const CAPABILITY_BINDING: &str = "capability:persistent-ocean-infrastructure:v1";
const V1_MODEL_ID: &str = "manta-forge-v1";
const V1_MODEL_BINDING: &str = "model:manta-forge-v1:sha256:example";
const V2_MODEL_ID: &str = "manta-forge-v2";
const V2_MODEL_BINDING: &str = "model:manta-forge-v2:sha256:example";
const V1_GENOME_BINDING: &str = "genome:manta-v1:sha256:example";
const V2_GENOME_BINDING: &str = "genome:manta-v2:sha256:example";
const SUBSTITUTION_BINDING: &str = "substitution:electronics-local-v1";

fn dependency(
    id: &str,
    kind: RegenerativeDependencyKind,
    governance: DependencyGovernance,
    demand: u64,
    production: u64,
    recycling: u64,
    stockpile: u64,
) -> RegenerativeDependency {
    RegenerativeDependency {
        dependency_id: id.into(),
        kind,
        governance,
        demand_units_per_period: demand,
        local_production_units_per_period: production,
        recycling_units_per_period: recycling,
        stockpile_units: stockpile,
        unit_mass_grams: Some(1),
        evidence_binding: format!("dependency:{id}:v1"),
    }
}

fn source_model() -> RegenerativeClosureModel {
    RegenerativeClosureModel {
        model_id: V1_MODEL_ID.into(),
        period_duration_ms: 1,
        dependencies: vec![
            dependency(
                "structural-material",
                RegenerativeDependencyKind::Material,
                DependencyGovernance::Ordinary,
                100,
                80,
                20,
                0,
            ),
            dependency(
                "control-electronics-imported",
                RegenerativeDependencyKind::Component,
                DependencyGovernance::Ordinary,
                10,
                0,
                0,
                50,
            ),
            dependency(
                "control-electronics-local",
                RegenerativeDependencyKind::Component,
                DependencyGovernance::Ordinary,
                10,
                10,
                0,
                0,
            ),
            dependency(
                "qualified-reactor-fuel-service",
                RegenerativeDependencyKind::ExternalService,
                DependencyGovernance::SafeguardedExternal,
                1,
                0,
                0,
                30,
            ),
        ],
        capabilities: vec![RegenerativeCapability {
            capability_id: CAPABILITY_ID.into(),
            essential: true,
            dependency_ids: BTreeSet::from([
                "control-electronics-imported".into(),
                "qualified-reactor-fuel-service".into(),
                "structural-material".into(),
            ]),
            evidence_binding: CAPABILITY_BINDING.into(),
        }],
        evidence_binding: V1_MODEL_BINDING.into(),
    }
}

fn requirement(
    dependency_id: &str,
    substitutions: Vec<String>,
) -> RegenerativeGenomeRequirementV1 {
    RegenerativeGenomeRequirementV1 {
        requirement_id: match dependency_id {
            "control-electronics-imported" | "control-electronics-local" => {
                "req-electronics".into()
            }
            "qualified-reactor-fuel-service" => "req-reactor-service".into(),
            "structural-material" => "req-structure".into(),
            other => panic!("unexpected test dependency: {other}"),
        },
        capability_id: CAPABILITY_ID.into(),
        baseline_dependency_id: dependency_id.into(),
        design_binding: format!("design:{dependency_id}:v1"),
        metrology_profile_binding: format!("metrology:{dependency_id}:v1"),
        requalification_profile_binding: format!("requalification:{dependency_id}:v1"),
        disassembly_profile_binding: format!("disassembly:{dependency_id}:v1"),
        recovery_profile_binding: format!("recovery:{dependency_id}:v1"),
        qualified_substitution_bindings: substitutions,
    }
}

fn genome_v1() -> RegenerativeGenomeV1 {
    RegenerativeGenomeV1 {
        schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
        genome_id: "manta-genome-v1".into(),
        lineage_parent_binding: None,
        closure_model_id: V1_MODEL_ID.into(),
        closure_model_evidence_binding: V1_MODEL_BINDING.into(),
        requirements: vec![
            requirement(
                "control-electronics-imported",
                vec![SUBSTITUTION_BINDING.into()],
            ),
            requirement("qualified-reactor-fuel-service", Vec::new()),
            requirement("structural-material", Vec::new()),
        ],
        evidence_binding: V1_GENOME_BINDING.into(),
    }
}

fn substitution() -> RegenerativeSubstitutionEvidenceV1 {
    RegenerativeSubstitutionEvidenceV1 {
        schema_version: REGENERATIVE_SUBSTITUTION_SCHEMA_V1,
        source_model_id: V1_MODEL_ID.into(),
        source_model_evidence_binding: V1_MODEL_BINDING.into(),
        derived_model_id: V2_MODEL_ID.into(),
        derived_model_evidence_binding: V2_MODEL_BINDING.into(),
        capability_id: CAPABILITY_ID.into(),
        capability_evidence_binding: CAPABILITY_BINDING.into(),
        baseline_dependency_id: "control-electronics-imported".into(),
        substitute_dependency_id: "control-electronics-local".into(),
        qualification_binding: SUBSTITUTION_BINDING.into(),
    }
}

fn genome_v2() -> RegenerativeGenomeV1 {
    RegenerativeGenomeV1 {
        schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
        genome_id: "manta-genome-v2".into(),
        lineage_parent_binding: Some(V1_GENOME_BINDING.into()),
        closure_model_id: V2_MODEL_ID.into(),
        closure_model_evidence_binding: V2_MODEL_BINDING.into(),
        requirements: vec![
            requirement(
                "control-electronics-local",
                vec![SUBSTITUTION_BINDING.into()],
            ),
            requirement("qualified-reactor-fuel-service", Vec::new()),
            requirement("structural-material", Vec::new()),
        ],
        evidence_binding: V2_GENOME_BINDING.into(),
    }
}

#[test]
fn explicit_substitution_creates_a_distinct_successor_genome_with_longer_horizon() {
    let source = source_model();
    let v1 = genome_v1();
    let v1_report = v1.evaluate_supportability(&source).unwrap();
    assert_eq!(v1_report.essential_horizon, RegenerativeHorizon::FinitePeriods(5));
    assert_eq!(v1_report.limiting_requirement_ids, vec!["req-electronics"]);

    let derived = substitution().apply_to_model(&source).unwrap();
    assert_eq!(derived.model_id, V2_MODEL_ID);
    assert_eq!(derived.evidence_binding, V2_MODEL_BINDING);

    // The source model is immutable evidence; deriving v2 does not rewrite v1.
    let source_capability = &source.capabilities[0];
    assert!(source_capability
        .dependency_ids
        .contains("control-electronics-imported"));
    assert!(!source_capability
        .dependency_ids
        .contains("control-electronics-local"));

    let derived_capability = &derived.capabilities[0];
    assert!(!derived_capability
        .dependency_ids
        .contains("control-electronics-imported"));
    assert!(derived_capability
        .dependency_ids
        .contains("control-electronics-local"));

    let v2 = genome_v2();
    let v2_report = v2.evaluate_supportability(&derived).unwrap();
    assert_eq!(v2.lineage_parent_binding.as_deref(), Some(V1_GENOME_BINDING));
    assert_eq!(v2.evidence_binding, V2_GENOME_BINDING);
    assert_eq!(v2_report.essential_horizon, RegenerativeHorizon::FinitePeriods(30));
    assert_eq!(v2_report.limiting_requirement_ids, vec!["req-reactor-service"]);
}

#[test]
fn successor_identifiers_match_the_recipe_free_mycelix_lineage_contract() {
    let v2 = genome_v2();
    let sub = substitution();

    assert_eq!(v2.genome_id, "manta-genome-v2");
    assert_eq!(v2.evidence_binding, "genome:manta-v2:sha256:example");
    assert_eq!(
        v2.lineage_parent_binding.as_deref(),
        Some("genome:manta-v1:sha256:example")
    );
    assert_eq!(v2.closure_model_evidence_binding, "model:manta-forge-v2:sha256:example");
    assert_eq!(sub.qualification_binding, "substitution:electronics-local-v1");
}

#[test]
fn explicit_successor_evolution_does_not_make_safeguarded_service_substitutable() {
    let derived = substitution().apply_to_model(&source_model()).unwrap();
    let reactor = derived
        .dependencies
        .iter()
        .find(|dependency| dependency.dependency_id == "qualified-reactor-fuel-service")
        .unwrap();
    assert_eq!(reactor.governance, DependencyGovernance::SafeguardedExternal);
    assert_eq!(reactor.local_production_units_per_period, 0);
    assert_eq!(reactor.recycling_units_per_period, 0);

    let reactor_requirement = genome_v2()
        .requirements
        .into_iter()
        .find(|requirement| requirement.requirement_id == "req-reactor-service")
        .unwrap();
    assert!(reactor_requirement.qualified_substitution_bindings.is_empty());
}
