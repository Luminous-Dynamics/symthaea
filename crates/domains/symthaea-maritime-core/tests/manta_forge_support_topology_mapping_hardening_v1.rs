include!("manta_forge_support_topology_continuity_v1.rs");

#[test]
fn duplicate_source_mapping_cannot_overwrite_graph_identity() {
    let source_model = model(3, false);
    let successor_model = model(4, true);
    let basis = basis_report(&source_model, &successor_model);
    let mut policy = mapping_policy(false);

    policy.mappings.push(RegenerativeSupportClosureMappingV1 {
        source_dependency_id: "forge-tooling-v3".into(),
        successor_dependency_id: "local-controller-v4".into(),
        topology_equivalence_binding: "topology-map:duplicate-source:forge-to-controller".into(),
    });
    policy.mappings.sort_by(|left, right| {
        (
            left.source_dependency_id.as_str(),
            left.successor_dependency_id.as_str(),
        )
            .cmp(&(
                right.source_dependency_id.as_str(),
                right.successor_dependency_id.as_str(),
            ))
    });

    assert_eq!(
        qualify_regenerative_role_support_closure_continuity(
            &policy,
            &profile(3),
            &source_model,
            &support(3, false),
            &profile(4),
            &successor_model,
            &support(4, false),
            &handoff_evidence(),
            &basis,
        ),
        Err(RegenerativeSupportClosureContinuityError::DuplicateSourceMapping {
            dependency_id: "forge-tooling-v3".into(),
        })
    );
}
