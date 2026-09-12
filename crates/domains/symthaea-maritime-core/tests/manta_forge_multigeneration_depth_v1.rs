include!("manta_forge_successor_depth_frontier_v1.rs");

const SYMTROPY_MULTIGENERATION_HEAD: &str =
    "d2453d2818b87cbfb3a22e979c9d1b77391fc686";

fn semantic_descendant_handoff(
    case: &SuccessorDepthCase,
    source_generation: u64,
    successor_generation: u64,
    source_genome: &RegenerativeGenomeV1,
    successor_genome: &RegenerativeGenomeV1,
) -> RegenerativeEpochHandoffEvidenceV1 {
    RegenerativeEpochHandoffEvidenceV1 {
        schema_version: REGENERATIVE_EPOCH_HANDOFF_SCHEMA_V1,
        handoff_id: format!(
            "handoff-manta-v{source_generation}-v{successor_generation}-r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        source_epoch_id: format!(
            "manta-v{source_generation}-frontier-r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        source_epoch_evidence_binding: format!(
            "epoch:manta-v{source_generation}-frontier:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        successor_epoch_id: format!(
            "manta-v{successor_generation}-frontier-r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        successor_epoch_evidence_binding: format!(
            "epoch:manta-v{successor_generation}-frontier:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        source_genome_id: source_genome.genome_id.clone(),
        source_genome_evidence_binding: source_genome.evidence_binding.clone(),
        successor_genome_id: successor_genome.genome_id.clone(),
        successor_genome_evidence_binding: successor_genome.evidence_binding.clone(),
        dynamic_handoff_receipt_binding: format!(
            "symtropy-pr:787:{SYMTROPY_MULTIGENERATION_HEAD}:v{source_generation}-v{successor_generation}:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
        transfer_qualifications: vec![
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: format!("forge-tooling-v{source_generation}"),
                successor_dependency_id: format!("forge-tooling-v{successor_generation}"),
                transfer_qualification_binding: format!(
                    "qualification:forge-v{source_generation}-v{successor_generation}:r{}-t{}",
                    case.recovery_tick, case.tooling_stock
                ),
                safeguarded_continuity_binding: None,
            },
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: format!("reactor-service-v{source_generation}"),
                successor_dependency_id: format!("reactor-service-v{successor_generation}"),
                transfer_qualification_binding: format!(
                    "qualification:reactor-v{source_generation}-v{successor_generation}:r{}-t{}",
                    case.recovery_tick, case.tooling_stock
                ),
                safeguarded_continuity_binding: Some(format!(
                    "safeguarded:reactor-v{source_generation}-v{successor_generation}:r{}-t{}",
                    case.recovery_tick, case.tooling_stock
                )),
            },
            RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: format!("structural-stock-v{source_generation}"),
                successor_dependency_id: format!("structural-stock-v{successor_generation}"),
                transfer_qualification_binding: format!(
                    "qualification:structure-v{source_generation}-v{successor_generation}:r{}-t{}",
                    case.recovery_tick, case.tooling_stock
                ),
                safeguarded_continuity_binding: None,
            },
        ],
        external_admission_qualifications: Vec::new(),
        evidence_binding: format!(
            "handoff-qualification:v{source_generation}-v{successor_generation}:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        ),
    }
}

#[test]
fn static_horizon_recurses_into_semantically_qualified_descendant_generations() {
    let cases = successor_depth_cases();
    assert_eq!(cases.len(), 42);

    let mut generation_depth_counts = [0usize; 5];
    let mut qualified_descendant_handoffs = 0usize;

    for case in &cases {
        let Some(expected_depth) = case.expected_successor_horizon else {
            continue;
        };

        let handoff_tick = case.recovery_tick + 1;
        let mut forge = case.tooling_stock - handoff_tick;
        let mut reactor = 8 - handoff_tick;
        assert_eq!(forge.min(reactor), expected_depth);

        let mut generation = 3u64;
        let mut parent_binding = "genome:manta-v2-success-control".to_string();
        let mut realized_generations = 0u64;

        loop {
            assert!(forge > 0);
            assert!(reactor > 0);
            let generation_name = format!("v{generation}");
            let current_model = model(&generation_name, forge, 0, reactor, 18);
            let current_support = support(&generation_name);
            let current_genome = genome(&generation_name, Some(parent_binding.as_str()));
            let current_profile = profile(&generation_name);
            let current_report = evaluate_regenerative_lineage_viability(
                &current_profile,
                &current_genome,
                &current_model,
                &current_support,
            )
            .unwrap();

            let remaining_depth = expected_depth - realized_generations;
            assert_eq!(
                current_report.regenerative_viability_horizon,
                RegenerativeHorizon::FinitePeriods(remaining_depth),
                "recursive horizon mismatch for {case:?} generation v{generation}"
            );
            assert_eq!(
                current_report.successor_reproduction_horizon,
                RegenerativeHorizon::FinitePeriods(remaining_depth),
                "recursive reproduction horizon mismatch for {case:?} generation v{generation}"
            );

            realized_generations += 1;
            if forge == 1 || reactor == 1 {
                assert_eq!(realized_generations, expected_depth);
                break;
            }

            // One complete generation period consumes one finite tooling unit and
            // one safeguarded-service unit before the next descendant handoff.
            let next_forge = forge - 1;
            let next_reactor = reactor - 1;
            let successor_generation = generation + 1;
            let successor_name = format!("v{successor_generation}");
            let successor_model = model(&successor_name, next_forge, 0, next_reactor, 18);
            let successor_genome = genome(
                &successor_name,
                Some(current_genome.evidence_binding.as_str()),
            );
            let handoff = semantic_descendant_handoff(
                case,
                generation,
                successor_generation,
                &current_genome,
                &successor_genome,
            );
            let qualified = qualify_regenerative_epoch_handoff(
                &handoff,
                &current_genome,
                &current_model,
                &successor_genome,
                &successor_model,
            )
            .unwrap();
            assert_eq!(qualified.qualified_transfer_count, 3);
            assert_eq!(qualified.cross_id_transfer_count, 3);
            assert_eq!(qualified.safeguarded_transfer_count, 1);
            assert_eq!(qualified.external_admission_count, 0);

            qualified_descendant_handoffs += 1;
            parent_binding = current_genome.evidence_binding;
            forge = next_forge;
            reactor = next_reactor;
            generation = successor_generation;
        }

        assert_eq!(realized_generations, expected_depth);
        generation_depth_counts[expected_depth as usize] += 1;
    }

    assert_eq!(generation_depth_counts[1], 6);
    assert_eq!(generation_depth_counts[2], 5);
    assert_eq!(generation_depth_counts[3], 4);
    assert_eq!(generation_depth_counts[4], 3);
    assert_eq!(generation_depth_counts[1..].iter().sum::<usize>(), 18);

    // This excludes the 18 already-qualified parent v2->v3 handoffs. It covers
    // the later v3->v4, v4->v5, and v5->v6 semantic transitions independently.
    assert_eq!(qualified_descendant_handoffs, 22);
}
