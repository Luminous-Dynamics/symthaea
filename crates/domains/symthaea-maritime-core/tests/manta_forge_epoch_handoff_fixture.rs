use std::collections::BTreeSet;
use symthaea_maritime_core::{
    qualify_regenerative_epoch_handoff, DependencyGovernance, RegenerativeCapability,
    RegenerativeClosureModel, RegenerativeDependency, RegenerativeDependencyKind,
    RegenerativeEpochExternalAdmissionQualificationV1, RegenerativeEpochHandoffEvidenceV1,
    RegenerativeEpochTransferQualificationV1, RegenerativeGenomeRequirementV1,
    RegenerativeGenomeV1, REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1,
    REGENERATIVE_GENOME_SCHEMA_V1,
};

const FIXTURE: &str = include_str!("../fixtures/manta-forge-epoch-handoff-v1.txt");

fn scalar(key: &str) -> &str {
    FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing fixture key {key}"))
}

fn governance(value: &str) -> DependencyGovernance {
    match value {
        "ordinary" => DependencyGovernance::Ordinary,
        "safeguarded_external" => DependencyGovernance::SafeguardedExternal,
        other => panic!("unexpected governance {other}"),
    }
}

fn dependency_kind(value: &str) -> RegenerativeDependencyKind {
    match value {
        "metrology" => RegenerativeDependencyKind::Metrology,
        "external_service" => RegenerativeDependencyKind::ExternalService,
        "component" => RegenerativeDependencyKind::Component,
        other => panic!("unexpected dependency kind {other}"),
    }
}

fn model(side: &str) -> RegenerativeClosureModel {
    let model_fields: Vec<_> = scalar(&format!("{side}_model")).split('|').collect();
    assert_eq!(model_fields.len(), 2);
    let dependency_prefix = format!("{side}_dependency=");
    let dependencies = FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix(&dependency_prefix))
        .map(|record| {
            let fields: Vec<_> = record.split('|').collect();
            let expected_len = if side == "source" { 4 } else { 3 };
            assert_eq!(fields.len(), expected_len);
            RegenerativeDependency {
                dependency_id: fields[0].into(),
                governance: governance(fields[1]),
                kind: dependency_kind(fields[2]),
                demand_units_per_period: 1,
                local_production_units_per_period: 0,
                recycling_units_per_period: 0,
                stockpile_units: if side == "source" {
                    fields[3].parse().unwrap()
                } else {
                    0
                },
                unit_mass_grams: None,
                evidence_binding: format!("dependency:{}", fields[0]),
            }
        })
        .collect();
    let capability_fields: Vec<_> = scalar(&format!("{side}_capability")).split('|').collect();
    assert_eq!(capability_fields.len(), 2);
    RegenerativeClosureModel {
        model_id: model_fields[0].into(),
        period_duration_ms: 1,
        dependencies,
        capabilities: vec![RegenerativeCapability {
            capability_id: capability_fields[0].into(),
            essential: true,
            dependency_ids: capability_fields[1]
                .split(',')
                .map(str::to_owned)
                .collect::<BTreeSet<_>>(),
            evidence_binding: format!("capability:{}", capability_fields[0]),
        }],
        evidence_binding: model_fields[1].into(),
    }
}

fn genome(side: &str, model: &RegenerativeClosureModel) -> RegenerativeGenomeV1 {
    let fields: Vec<_> = scalar(&format!("{side}_genome")).split('|').collect();
    let expected_len = if side == "source" { 2 } else { 3 };
    assert_eq!(fields.len(), expected_len);
    let capability_id = model.capabilities[0].capability_id.clone();
    let mut requirements = model
        .dependencies
        .iter()
        .map(|dependency| RegenerativeGenomeRequirementV1 {
            requirement_id: format!("req-{}", dependency.dependency_id),
            capability_id: capability_id.clone(),
            baseline_dependency_id: dependency.dependency_id.clone(),
            design_binding: format!("design:{}", dependency.dependency_id),
            metrology_profile_binding: format!("metrology-profile:{}", dependency.dependency_id),
            requalification_profile_binding: format!("requal:{}", dependency.dependency_id),
            disassembly_profile_binding: format!("disassembly:{}", dependency.dependency_id),
            recovery_profile_binding: format!("recovery:{}", dependency.dependency_id),
            qualified_substitution_bindings: Vec::new(),
        })
        .collect::<Vec<_>>();
    requirements.sort_by(|left, right| left.requirement_id.cmp(&right.requirement_id));
    RegenerativeGenomeV1 {
        schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
        genome_id: fields[0].into(),
        lineage_parent_binding: if side == "source" {
            None
        } else {
            Some(fields[2].into())
        },
        closure_model_id: model.model_id.clone(),
        closure_model_evidence_binding: model.evidence_binding.clone(),
        requirements,
        evidence_binding: fields[1].into(),
    }
}

#[test]
fn shared_epoch_handoff_fixture_pins_semantic_transfer_qualification() {
    assert_eq!(scalar("schema"), "manta-forge-epoch-handoff-v1");
    let source_model = model("source");
    let successor_model = model("successor");
    let source_genome = genome("source", &source_model);
    let successor_genome = genome("successor", &successor_model);

    let source_epoch: Vec<_> = scalar("source_epoch").split('|').collect();
    let successor_epoch: Vec<_> = scalar("successor_epoch").split('|').collect();
    let handoff: Vec<_> = scalar("handoff").split('|').collect();
    assert_eq!(source_epoch.len(), 2);
    assert_eq!(successor_epoch.len(), 2);
    assert_eq!(handoff.len(), 3);

    let transfer_qualifications = FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("transfer="))
        .map(|record| {
            let fields: Vec<_> = record.split('|').collect();
            assert_eq!(fields.len(), 6);
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: fields[0].into(),
                successor_dependency_id: fields[1].into(),
                transfer_qualification_binding: fields[4].into(),
                safeguarded_continuity_binding: (!fields[5].is_empty())
                    .then(|| fields[5].to_owned()),
            }
        })
        .collect();
    let external_admission_qualifications = FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("external="))
        .map(|record| {
            let fields: Vec<_> = record.split('|').collect();
            assert_eq!(fields.len(), 5);
            RegenerativeEpochExternalAdmissionQualificationV1 {
                successor_dependency_id: fields[0].into(),
                inventory_evidence_binding: fields[2].into(),
                admission_qualification_binding: fields[3].into(),
                safeguarded_admission_binding: (!fields[4].is_empty())
                    .then(|| fields[4].to_owned()),
            }
        })
        .collect();

    let evidence = RegenerativeEpochHandoffEvidenceV1 {
        schema_version: REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1,
        handoff_id: handoff[0].into(),
        source_epoch_id: source_epoch[0].into(),
        source_epoch_evidence_binding: source_epoch[1].into(),
        successor_epoch_id: successor_epoch[0].into(),
        successor_epoch_evidence_binding: successor_epoch[1].into(),
        source_genome_id: source_genome.genome_id.clone(),
        source_genome_evidence_binding: source_genome.evidence_binding.clone(),
        successor_genome_id: successor_genome.genome_id.clone(),
        successor_genome_evidence_binding: successor_genome.evidence_binding.clone(),
        dynamic_handoff_receipt_binding: handoff[2].into(),
        transfer_qualifications,
        external_admission_qualifications,
        evidence_binding: handoff[1].into(),
    };

    let report = qualify_regenerative_epoch_handoff(
        &evidence,
        &source_genome,
        &source_model,
        &successor_genome,
        &successor_model,
    )
    .unwrap();

    assert_eq!(
        report.qualified_transfer_count,
        scalar("expected_qualified_transfer_count").parse().unwrap()
    );
    assert_eq!(report.cross_id_transfer_count, 3);
    assert_eq!(
        report.safeguarded_transfer_count,
        scalar("expected_safeguarded_transfer_count")
            .parse()
            .unwrap()
    );
    assert_eq!(
        report.external_admission_count,
        scalar("expected_external_admission_count").parse().unwrap()
    );
    assert_eq!(report.safeguarded_external_admission_count, 0);
    assert_eq!(report.dynamic_handoff_receipt_binding, handoff[2]);

    // Symtropy independently consumes these same bytes to prove runtime inventory
    // conservation and the resulting successor inventory quantities.
    assert_eq!(scalar("expected_source_final_tick"), "1");
    assert!(FIXTURE.contains("expected_successor_inventory=spares-v2|84"));
}
