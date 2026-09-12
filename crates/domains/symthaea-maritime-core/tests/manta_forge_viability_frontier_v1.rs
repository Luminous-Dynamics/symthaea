use std::collections::BTreeSet;
use symthaea_maritime_core::{
    assess_regenerative_bootstrap_reserve, validate_regenerative_bootstrap_reserve_policy,
    DependencyGovernance, RegenerativeBootstrapReserveObservationV1,
    RegenerativeBootstrapReservePolicyV1, RegenerativeBootstrapReserveQuantityV1,
    RegenerativeBootstrapReserveRequirementV1, RegenerativeCapability, RegenerativeClosureModel,
    RegenerativeDependency, RegenerativeDependencyKind, RegenerativeEpochHandoffEvidenceV1,
    RegenerativeEpochTransferQualificationV1, RegenerativeGenomeRequirementV1,
    RegenerativeGenomeV1, REGENERATIVE_BOOTSTRAP_RESERVE_SCHEMA_V1,
    REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1, REGENERATIVE_GENOME_SCHEMA_V1,
};

const FIXTURE: &str = include_str!("../fixtures/manta-forge-viability-frontier-v1.txt");
const SYMTROPY_FRONTIER_HEAD: &str = "c3c7f09853075457f430f61eba7dee0a603f29bf";

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrontierCase {
    recovery_tick: u64,
    tooling_stock: u64,
    expected_role_window: Option<u64>,
    expected_handoff_window: Option<u64>,
}

fn parse_window(value: &str) -> Option<u64> {
    if value == "none" {
        None
    } else {
        Some(value.parse().unwrap())
    }
}

fn cases() -> Vec<FrontierCase> {
    FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("case="))
        .map(|line| {
            let mut recovery_tick = None;
            let mut tooling_stock = None;
            let mut expected_role_window = None;
            let mut expected_handoff_window = None;
            for field in line.split('|') {
                let (key, value) = field.split_once(':').unwrap();
                match key {
                    "recovery_tick" => recovery_tick = Some(value.parse().unwrap()),
                    "tooling_stock" => tooling_stock = Some(value.parse().unwrap()),
                    "expected_role_window" => expected_role_window = Some(parse_window(value)),
                    "expected_handoff_window" => {
                        expected_handoff_window = Some(parse_window(value))
                    }
                    other => panic!("unknown frontier field {other}"),
                }
            }
            FrontierCase {
                recovery_tick: recovery_tick.unwrap(),
                tooling_stock: tooling_stock.unwrap(),
                expected_role_window: expected_role_window.unwrap(),
                expected_handoff_window: expected_handoff_window.unwrap(),
            }
        })
        .collect()
}

fn dependency(
    id: &str,
    kind: RegenerativeDependencyKind,
    governance: DependencyGovernance,
) -> RegenerativeDependency {
    RegenerativeDependency {
        dependency_id: id.into(),
        kind,
        governance,
        demand_units_per_period: 1,
        local_production_units_per_period: 0,
        recycling_units_per_period: 0,
        stockpile_units: 1,
        unit_mass_grams: None,
        evidence_binding: format!("dependency:{id}:frontier-policy"),
    }
}

fn model(generation: &str) -> RegenerativeClosureModel {
    let forge = format!("forge-tooling-{generation}");
    let reactor = format!("reactor-service-{generation}");
    RegenerativeClosureModel {
        model_id: format!("manta-{generation}-frontier-policy"),
        period_duration_ms: 1,
        dependencies: vec![
            dependency(
                &forge,
                RegenerativeDependencyKind::Tooling,
                DependencyGovernance::Ordinary,
            ),
            dependency(
                &reactor,
                RegenerativeDependencyKind::ExternalService,
                DependencyGovernance::SafeguardedExternal,
            ),
        ],
        capabilities: vec![RegenerativeCapability {
            capability_id: format!("bootstrap-{generation}"),
            essential: true,
            dependency_ids: BTreeSet::from([forge, reactor]),
            evidence_binding: format!("capability:bootstrap-{generation}:frontier-policy"),
        }],
        evidence_binding: format!("model:manta-{generation}-frontier-policy"),
    }
}

fn requirement(id: &str, capability: &str, dependency: &str) -> RegenerativeGenomeRequirementV1 {
    RegenerativeGenomeRequirementV1 {
        requirement_id: id.into(),
        capability_id: capability.into(),
        baseline_dependency_id: dependency.into(),
        design_binding: format!("design:{id}"),
        metrology_profile_binding: format!("metrology-profile:{id}"),
        requalification_profile_binding: format!("requalification:{id}"),
        disassembly_profile_binding: format!("disassembly:{id}"),
        recovery_profile_binding: format!("recovery:{id}"),
        qualified_substitution_bindings: Vec::new(),
    }
}

fn genome(generation: &str, parent: Option<&str>) -> RegenerativeGenomeV1 {
    let capability = format!("bootstrap-{generation}");
    let mut requirements = vec![
        requirement(
            &format!("req-{generation}-forge"),
            &capability,
            &format!("forge-tooling-{generation}"),
        ),
        requirement(
            &format!("req-{generation}-reactor"),
            &capability,
            &format!("reactor-service-{generation}"),
        ),
    ];
    requirements.sort_by(|left, right| left.requirement_id.cmp(&right.requirement_id));
    RegenerativeGenomeV1 {
        schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
        genome_id: format!("manta-genome-{generation}-frontier-policy"),
        lineage_parent_binding: parent.map(str::to_owned),
        closure_model_id: format!("manta-{generation}-frontier-policy"),
        closure_model_evidence_binding: format!("model:manta-{generation}-frontier-policy"),
        requirements,
        evidence_binding: format!("genome:manta-{generation}-frontier-policy"),
    }
}

fn handoff(case: &FrontierCase, v2: &RegenerativeGenomeV1, v3: &RegenerativeGenomeV1) -> RegenerativeEpochHandoffEvidenceV1 {
    RegenerativeEpochHandoffEvidenceV1 {
        schema_version: REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1,
        handoff_id: format!(
            "handoff-frontier-r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        source_epoch_id: "manta-v2-frontier".into(),
        source_epoch_evidence_binding: format!(
            "epoch:manta-v2-frontier:tooling-{}",
            case.tooling_stock
        ),
        successor_epoch_id: "manta-v3-frontier".into(),
        successor_epoch_evidence_binding: format!(
            "epoch:manta-v3-frontier:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        source_genome_id: v2.genome_id.clone(),
        source_genome_evidence_binding: v2.evidence_binding.clone(),
        successor_genome_id: v3.genome_id.clone(),
        successor_genome_evidence_binding: v3.evidence_binding.clone(),
        dynamic_handoff_receipt_binding: format!(
            "symtropy-handoff:{SYMTROPY_FRONTIER_HEAD}:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        transfer_qualifications: vec![
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "forge-tooling-v2".into(),
                successor_dependency_id: "forge-tooling-v3".into(),
                transfer_qualification_binding: "qualification:forge-frontier-v2-v3".into(),
                safeguarded_continuity_binding: None,
            },
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "reactor-service-v2".into(),
                successor_dependency_id: "reactor-service-v3".into(),
                transfer_qualification_binding: "qualification:reactor-frontier-v2-v3".into(),
                safeguarded_continuity_binding: Some(
                    "safeguarded:reactor-frontier-v2-v3".into(),
                ),
            },
        ],
        external_admission_qualifications: Vec::new(),
        evidence_binding: format!(
            "handoff-qualification:frontier:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
    }
}

fn policy(case: &FrontierCase, handoff: &RegenerativeEpochHandoffEvidenceV1) -> RegenerativeBootstrapReservePolicyV1 {
    RegenerativeBootstrapReservePolicyV1 {
        schema_version: REGENERATIVE_BOOTSTRAP_RESERVE_SCHEMA_V1,
        policy_id: format!(
            "bootstrap-policy-frontier-r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        handoff_qualification_evidence_binding: handoff.evidence_binding.clone(),
        requirements: vec![
            RegenerativeBootstrapReserveRequirementV1 {
                source_dependency_id: "forge-tooling-v2".into(),
                successor_dependency_id: "forge-tooling-v3".into(),
                minimum_transfer_units: 1,
                reserve_requirement_binding: "bootstrap-requirement:forge-min-1".into(),
            },
            RegenerativeBootstrapReserveRequirementV1 {
                source_dependency_id: "reactor-service-v2".into(),
                successor_dependency_id: "reactor-service-v3".into(),
                minimum_transfer_units: 1,
                reserve_requirement_binding: "bootstrap-requirement:reactor-min-1".into(),
            },
        ],
        evidence_binding: "bootstrap-policy:manta-forge-frontier-v1".into(),
    }
}

fn observation(
    case: &FrontierCase,
    handoff: &RegenerativeEpochHandoffEvidenceV1,
) -> RegenerativeBootstrapReserveObservationV1 {
    let observation_tick = case
        .expected_role_window
        .unwrap_or_else(|| (case.recovery_tick + 1).min(9));
    RegenerativeBootstrapReserveObservationV1 {
        observation_id: format!(
            "frontier-observation-r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        dynamic_handoff_receipt_binding: handoff.dynamic_handoff_receipt_binding.clone(),
        source_final_tick: observation_tick,
        source_final_observation_tick: Some(observation_tick),
        construction_and_qualification_available: case.expected_role_window.is_some(),
        quantities: vec![
            RegenerativeBootstrapReserveQuantityV1 {
                source_dependency_id: "forge-tooling-v2".into(),
                successor_dependency_id: "forge-tooling-v3".into(),
                available_transfer_units: case.tooling_stock.saturating_sub(observation_tick),
            },
            RegenerativeBootstrapReserveQuantityV1 {
                source_dependency_id: "reactor-service-v2".into(),
                successor_dependency_id: "reactor-service-v3".into(),
                available_transfer_units: 8u64.saturating_sub(observation_tick),
            },
        ],
        evidence_binding: format!(
            "symtropy-frontier:{SYMTROPY_FRONTIER_HEAD}:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
    }
}

#[test]
fn semantic_bootstrap_policy_explains_the_dynamic_frontier_gap() {
    let cases = cases();
    assert_eq!(cases.len(), 42);
    let v2_model = model("v2");
    let v3_model = model("v3");
    let v2_genome = genome("v2", Some("genome:manta-v1-frontier-policy"));
    let v3_genome = genome("v3", Some(v2_genome.evidence_binding.as_str()));

    let mut role_overlap_count = 0usize;
    let mut reproduction_ready_count = 0usize;
    let mut overlap_without_bootstrap_count = 0usize;

    for case in &cases {
        let handoff = handoff(case, &v2_genome, &v3_genome);
        let policy = policy(case, &handoff);
        validate_regenerative_bootstrap_reserve_policy(
            &policy,
            &handoff,
            &v2_genome,
            &v2_model,
            &v3_genome,
            &v3_model,
        )
        .unwrap();

        let assessment = assess_regenerative_bootstrap_reserve(
            &policy,
            &handoff,
            &observation(case, &handoff),
        )
        .unwrap();

        let expected_ready = case.expected_handoff_window.is_some();
        assert_eq!(assessment.reproduction_ready, expected_ready, "{case:?}");
        assert!(assessment.fresh_completed_observation);
        if case.expected_role_window.is_some() {
            role_overlap_count += 1;
            assert!(assessment.construction_and_qualification_available);
        }
        if assessment.reproduction_ready {
            reproduction_ready_count += 1;
            assert!(assessment.bootstrap_reserve_satisfied);
            assert!(assessment.deficits.is_empty());
            assert_eq!(case.expected_role_window, case.expected_handoff_window);
        } else if case.expected_role_window.is_some() {
            overlap_without_bootstrap_count += 1;
            assert!(!assessment.bootstrap_reserve_satisfied);
            assert!(!assessment.deficits.is_empty());
        }
    }

    assert_eq!(role_overlap_count, 25);
    assert_eq!(reproduction_ready_count, 18);
    assert_eq!(overlap_without_bootstrap_count, 7);
}

#[test]
fn stale_observation_cannot_be_reproduction_ready_even_with_sufficient_reserve() {
    let case = FrontierCase {
        recovery_tick: 3,
        tooling_stock: 5,
        expected_role_window: Some(4),
        expected_handoff_window: Some(4),
    };
    let v2_model = model("v2");
    let v3_model = model("v3");
    let v2_genome = genome("v2", Some("genome:manta-v1-frontier-policy"));
    let v3_genome = genome("v3", Some(v2_genome.evidence_binding.as_str()));
    let handoff = handoff(&case, &v2_genome, &v3_genome);
    let policy = policy(&case, &handoff);
    validate_regenerative_bootstrap_reserve_policy(
        &policy,
        &handoff,
        &v2_genome,
        &v2_model,
        &v3_genome,
        &v3_model,
    )
    .unwrap();

    let mut observation = observation(&case, &handoff);
    observation.source_final_observation_tick = Some(3);
    let assessment = assess_regenerative_bootstrap_reserve(&policy, &handoff, &observation).unwrap();
    assert!(assessment.bootstrap_reserve_satisfied);
    assert!(!assessment.fresh_completed_observation);
    assert!(!assessment.reproduction_ready);
}
