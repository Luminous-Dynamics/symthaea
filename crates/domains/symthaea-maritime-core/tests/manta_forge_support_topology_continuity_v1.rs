use std::collections::BTreeSet;
use symthaea_maritime_core::*;

const FIXTURE: &str =
    include_str!("../fixtures/manta-forge-support-topology-adversary-v1.txt");

fn scalar(key: &str) -> u64 {
    FIXTURE
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing fixture scalar {key}"))
        .parse()
        .unwrap()
}

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
        unit_mass_grams: None,
        evidence_binding: format!("dependency:{id}:evidence"),
    }
}

fn model(generation: u64, successor_template: bool) -> RegenerativeClosureModel {
    let stock = |source: u64| if successor_template { 0 } else { source };
    RegenerativeClosureModel {
        model_id: format!("closure-v{generation}-topology"),
        period_duration_ms: scalar("period_duration_ms"),
        dependencies: vec![
            dependency(
                &format!("controller-support-v{generation}"),
                RegenerativeDependencyKind::ProcessInput,
                DependencyGovernance::Ordinary,
                scalar("controller_support_demand_units_per_period"),
                0,
                0,
                stock(scalar("controller_support_inventory_units")),
            ),
            dependency(
                &format!("forge-tooling-v{generation}"),
                RegenerativeDependencyKind::Tooling,
                DependencyGovernance::Ordinary,
                scalar("forge_tooling_demand_units_per_period"),
                0,
                0,
                stock(scalar("forge_tooling_inventory_units")),
            ),
            dependency(
                &format!("local-controller-v{generation}"),
                RegenerativeDependencyKind::Component,
                DependencyGovernance::Ordinary,
                1,
                1,
                0,
                0,
            ),
            dependency(
                &format!("metrology-v{generation}"),
                RegenerativeDependencyKind::Metrology,
                DependencyGovernance::Ordinary,
                1,
                1,
                0,
                0,
            ),
            dependency(
                &format!("reactor-service-v{generation}"),
                RegenerativeDependencyKind::ExternalService,
                DependencyGovernance::SafeguardedExternal,
                scalar("reactor_service_demand_units_per_period"),
                0,
                0,
                stock(scalar("reactor_service_inventory_units")),
            ),
            dependency(
                &format!("structural-stock-v{generation}"),
                RegenerativeDependencyKind::Material,
                DependencyGovernance::Ordinary,
                1,
                0,
                1,
                0,
            ),
        ],
        capabilities: vec![
            RegenerativeCapability {
                capability_id: format!("operation-v{generation}"),
                essential: true,
                dependency_ids: BTreeSet::from([
                    format!("reactor-service-v{generation}"),
                    format!("structural-stock-v{generation}"),
                ]),
                evidence_binding: format!("capability:operation:v{generation}"),
            },
            RegenerativeCapability {
                capability_id: format!("successor-construction-v{generation}"),
                essential: true,
                dependency_ids: BTreeSet::from([
                    format!("forge-tooling-v{generation}"),
                    format!("local-controller-v{generation}"),
                    format!("structural-stock-v{generation}"),
                ]),
                evidence_binding: format!("capability:construction:v{generation}"),
            },
            RegenerativeCapability {
                capability_id: format!("successor-qualification-v{generation}"),
                essential: true,
                dependency_ids: BTreeSet::from([
                    format!("metrology-v{generation}"),
                    format!("reactor-service-v{generation}"),
                ]),
                evidence_binding: format!("capability:qualification:v{generation}"),
            },
        ],
        evidence_binding: format!("model:closure-v{generation}-topology"),
    }
}

fn flow_claim(
    dependency_id: String,
    flow_kind: RegenerativeFlowKindV1,
    prerequisite_dependency_id: String,
) -> RegenerativeFlowSupportClaimV1 {
    RegenerativeFlowSupportClaimV1 {
        dependency_id: dependency_id.clone(),
        flow_kind,
        capability_binding: format!("flow-capability:{dependency_id}:{flow_kind:?}"),
        prerequisite_dependency_ids: vec![prerequisite_dependency_id],
        external_input_binding: None,
        metrology_binding: format!("flow-metrology:{dependency_id}:{flow_kind:?}"),
        qualification_binding: format!("flow-qualification:{dependency_id}:{flow_kind:?}"),
        bootstrap_binding: None,
    }
}

fn support(generation: u64, hidden: bool) -> RegenerativeFlowSupportV1 {
    let controller_prerequisite = if hidden {
        format!("controller-support-v{generation}")
    } else {
        format!("reactor-service-v{generation}")
    };
    let mut claims = vec![
        flow_claim(
            format!("local-controller-v{generation}"),
            RegenerativeFlowKindV1::Production,
            controller_prerequisite,
        ),
        flow_claim(
            format!("metrology-v{generation}"),
            RegenerativeFlowKindV1::Production,
            format!("reactor-service-v{generation}"),
        ),
        flow_claim(
            format!("structural-stock-v{generation}"),
            RegenerativeFlowKindV1::Recycling,
            format!("forge-tooling-v{generation}"),
        ),
    ];
    claims.sort_by(|left, right| {
        (left.dependency_id.as_str(), left.flow_kind)
            .cmp(&(right.dependency_id.as_str(), right.flow_kind))
    });
    RegenerativeFlowSupportV1 {
        schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
        support_id: format!(
            "support:v{generation}:{}",
            if hidden { "hidden" } else { "direct" }
        ),
        closure_model_id: format!("closure-v{generation}-topology"),
        closure_model_evidence_binding: format!("model:closure-v{generation}-topology"),
        claims,
        evidence_binding: format!(
            "flow-support:v{generation}:{}",
            if hidden { "hidden" } else { "direct" }
        ),
    }
}

fn profile(generation: u64) -> RegenerativeLineageViabilityProfileV1 {
    RegenerativeLineageViabilityProfileV1 {
        schema_version: REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
        profile_id: format!("profile-v{generation}-topology"),
        closure_model_id: format!("closure-v{generation}-topology"),
        closure_model_evidence_binding: format!("model:closure-v{generation}-topology"),
        operational_capability_ids: vec![format!("operation-v{generation}")],
        successor_construction_capability_ids: vec![format!(
            "successor-construction-v{generation}"
        )],
        successor_qualification_capability_ids: vec![format!(
            "successor-qualification-v{generation}"
        )],
        evidence_binding: format!("profile-evidence:v{generation}:topology"),
    }
}

fn mapping_policy(hidden: bool) -> RegenerativeRoleSupportClosureContinuityPolicyV1 {
    let mut source_ids = vec![
        "forge-tooling-v3",
        "local-controller-v3",
        "metrology-v3",
        "reactor-service-v3",
        "structural-stock-v3",
    ];
    if hidden {
        source_ids.push("controller-support-v3");
    }
    let mut mappings: Vec<_> = source_ids
        .into_iter()
        .map(|source| {
            let successor = source.replace("-v3", "-v4");
            RegenerativeSupportClosureMappingV1 {
                source_dependency_id: source.into(),
                successor_dependency_id: successor.clone(),
                topology_equivalence_binding: format!("topology-map:{source}:{successor}"),
            }
        })
        .collect();
    mappings.sort_by(|left, right| {
        (
            left.source_dependency_id.as_str(),
            left.successor_dependency_id.as_str(),
        )
            .cmp(&(
                right.source_dependency_id.as_str(),
                right.successor_dependency_id.as_str(),
            ))
    });
    RegenerativeRoleSupportClosureContinuityPolicyV1 {
        policy_id: format!(
            "role-support-closure-{}",
            if hidden { "hidden" } else { "direct" }
        ),
        evidence_binding: format!(
            "role-support-closure-evidence:{}",
            if hidden { "hidden" } else { "direct" }
        ),
        mappings,
    }
}

fn handoff_evidence() -> RegenerativeEpochHandoffEvidenceV1 {
    RegenerativeEpochHandoffEvidenceV1 {
        schema_version: REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1,
        handoff_id: "handoff-topology-v3-v4".into(),
        source_epoch_id: "epoch-v3".into(),
        source_epoch_evidence_binding: "epoch:v3:topology".into(),
        successor_epoch_id: "epoch-v4".into(),
        successor_epoch_evidence_binding: "epoch:v4:topology".into(),
        source_genome_id: "genome-v3".into(),
        source_genome_evidence_binding: "genome:v3:topology".into(),
        successor_genome_id: "genome-v4".into(),
        successor_genome_evidence_binding: "genome:v4:topology".into(),
        dynamic_handoff_receipt_binding: "symtropy:pr-840:topology-handoff".into(),
        transfer_qualifications: vec![
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "forge-tooling-v3".into(),
                successor_dependency_id: "forge-tooling-v4".into(),
                transfer_qualification_binding: "transfer:forge-tooling:v3-v4".into(),
                safeguarded_continuity_binding: None,
            },
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "reactor-service-v3".into(),
                successor_dependency_id: "reactor-service-v4".into(),
                transfer_qualification_binding: "transfer:reactor-service:v3-v4".into(),
                safeguarded_continuity_binding: Some("safeguard:reactor-service:v3-v4".into()),
            },
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "structural-stock-v3".into(),
                successor_dependency_id: "structural-stock-v4".into(),
                transfer_qualification_binding: "transfer:structural-stock:v3-v4".into(),
                safeguarded_continuity_binding: None,
            },
        ],
        external_admission_qualifications: Vec::new(),
        evidence_binding: "handoff-evidence:topology-v3-v4".into(),
    }
}

fn basis_report(
    source_model: &RegenerativeClosureModel,
    successor_model: &RegenerativeClosureModel,
) -> RegenerativeIntergenerationalSupportBasisReportV1 {
    let evidence = handoff_evidence();
    let handoff_report = RegenerativeEpochHandoffQualificationReportV1 {
        handoff_id: evidence.handoff_id.clone(),
        source_genome_id: evidence.source_genome_id.clone(),
        successor_genome_id: evidence.successor_genome_id.clone(),
        dynamic_handoff_receipt_binding: evidence.dynamic_handoff_receipt_binding.clone(),
        qualified_transfer_count: 3,
        cross_id_transfer_count: 3,
        safeguarded_transfer_count: 1,
        external_admission_count: 0,
        safeguarded_external_admission_count: 0,
    };
    let policy = RegenerativeIntergenerationalSupportBasisPolicyV1 {
        policy_id: "basis-policy-topology-v1".into(),
        evidence_binding: "basis-policy:topology:v1".into(),
        requirements: vec![
            RegenerativeSupportBasisRequirementV1 {
                source_dependency_id: "forge-tooling-v3".into(),
                successor_dependency_id: "forge-tooling-v4".into(),
                quantity_basis_equivalence_binding: "quantity-basis:forge:v3-v4".into(),
            },
            RegenerativeSupportBasisRequirementV1 {
                source_dependency_id: "reactor-service-v3".into(),
                successor_dependency_id: "reactor-service-v4".into(),
                quantity_basis_equivalence_binding: "quantity-basis:reactor:v3-v4".into(),
            },
        ],
    };
    assess_intergenerational_support_basis(
        &policy,
        &evidence,
        &handoff_report,
        source_model,
        successor_model,
    )
    .unwrap()
}

fn role_report(
    role: RegenerativeLineageRoleV1,
    horizon: RegenerativeHorizon,
    roots: &[&str],
) -> RegenerativeLineageRoleReportV1 {
    RegenerativeLineageRoleReportV1 {
        role,
        conservative_horizon: horizon,
        limiting_requirement_ids: vec![format!("requirement:{role:?}")],
        root_limiting_dependency_ids: roots.iter().map(|root| (*root).into()).collect(),
        fully_modeled_support: true,
        externally_conditioned_requirement_ids: Vec::new(),
    }
}

fn lineage_report() -> RegenerativeLineageViabilityReportV1 {
    let h4 = RegenerativeHorizon::FinitePeriods(scalar("source_horizon_periods"));
    RegenerativeLineageViabilityReportV1 {
        profile_id: "profile-v3-topology".into(),
        genome_id: "genome-v3".into(),
        closure_model_id: "closure-v3-topology".into(),
        flow_support_id: "support:v3:direct".into(),
        operation: role_report(
            RegenerativeLineageRoleV1::Operation,
            RegenerativeHorizon::FinitePeriods(100),
            &["reactor-service-v3"],
        ),
        successor_construction: role_report(
            RegenerativeLineageRoleV1::SuccessorConstruction,
            h4,
            &["forge-tooling-v3"],
        ),
        successor_qualification: role_report(
            RegenerativeLineageRoleV1::SuccessorQualification,
            RegenerativeHorizon::FinitePeriods(100),
            &["reactor-service-v3"],
        ),
        successor_reproduction_horizon: h4,
        regenerative_viability_horizon: h4,
        limiting_roles: vec![RegenerativeLineageRoleV1::SuccessorConstruction],
        fully_modeled_regenerative_viability: true,
    }
}

fn policies() -> Vec<RegenerativeReproductionPolicySpecV1> {
    (1..=4)
        .map(|maturity_periods| RegenerativeReproductionPolicySpecV1 {
            reproduction_policy_id: format!("maturity-{maturity_periods}"),
            reproduction_policy_evidence_binding: format!(
                "reproduction-policy:maturity-{maturity_periods}:v1"
            ),
            maturity_periods,
        })
        .collect()
}

fn expected_direct_surface() -> Vec<(u64, u64, u64, u64, u64)> {
    FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("direct_policy="))
        .map(|line| {
            let mut values = [0u64; 5];
            for field in line.split('|') {
                let (key, raw) = field.split_once(':').unwrap();
                let value: u64 = raw.parse().unwrap();
                match key {
                    "maturity_periods" => values[0] = value,
                    "founded_descendants" => values[1] = value,
                    "maturity_completed_descendants" => values[2] = value,
                    "descendant_reproduction_transitions" => values[3] = value,
                    "terminal_residual_periods" => values[4] = value,
                    other => panic!("unknown fixture field {other}"),
                }
            }
            (values[0], values[1], values[2], values[3], values[4])
        })
        .collect()
}

#[test]
fn direct_support_closure_isomorphic_and_root_continuous_authorizes_h4_surface() {
    let source_model = model(3, false);
    let successor_model = model(4, true);
    let basis = basis_report(&source_model, &successor_model);
    assert!(basis.scalar_runway_projection_safe);

    let continuity = qualify_regenerative_role_support_closure_continuity(
        &mapping_policy(false),
        &profile(3),
        &source_model,
        &support(3, false),
        &profile(4),
        &successor_model,
        &support(4, false),
        &handoff_evidence(),
        &basis,
    )
    .unwrap();
    assert!(continuity.scalar_runway_projection_safe);
    assert_eq!(
        continuity.source_finite_root_dependency_ids,
        vec!["forge-tooling-v3", "reactor-service-v3"]
    );

    let surface = evaluate_closure_qualified_regenerative_policy_sensitivity_surface(
        &lineage_report(),
        &policies(),
        &basis,
        &continuity,
    )
    .unwrap();
    let observed: Vec<_> = surface
        .policy_points
        .iter()
        .map(|point| {
            let projection = &point.generation_policy_projection;
            let finite = |value| match value {
                RegenerativeGenerationCountV1::Finite(value) => value,
                RegenerativeGenerationCountV1::IndefiniteUnderStaticModel => {
                    panic!("expected finite topology fixture")
                }
            };
            (
                projection.maturity_periods,
                finite(projection.founded_descendant_generations),
                finite(projection.maturity_completed_descendant_generations),
                finite(projection.descendant_reproduction_transitions),
                projection.terminal_generation_residual_periods.unwrap(),
            )
        })
        .collect();
    assert_eq!(observed, expected_direct_surface());
}

#[test]
fn hidden_finite_root_not_transferred_blocks_scalar_projection() {
    let source_model = model(3, false);
    let successor_model = model(4, true);
    let basis = basis_report(&source_model, &successor_model);
    let error = qualify_regenerative_role_support_closure_continuity(
        &mapping_policy(true),
        &profile(3),
        &source_model,
        &support(3, true),
        &profile(4),
        &successor_model,
        &support(4, true),
        &handoff_evidence(),
        &basis,
    )
    .unwrap_err();
    assert_eq!(
        error,
        RegenerativeSupportClosureContinuityError::FiniteRootNotTransferred {
            source_dependency_id: "controller-support-v3".into(),
            successor_dependency_id: "controller-support-v4".into(),
        }
    );
}
