include!("manta_forge_multigeneration_depth_v1.rs");

const REPRODUCTION_POLICY_FIXTURE: &str =
    include_str!("../fixtures/manta-forge-reproduction-policy-frontier-v1.txt");
const SYMTROPY_POLICY_HEAD: &str = "e49453c04dd56226bd53bd9b21f7e3c43703a667";

#[derive(Debug, Clone, PartialEq, Eq)]
struct PolicyExpectation {
    maturity_periods: u64,
    depth_counts: [usize; 5],
    expected_total_handoffs: usize,
}

fn reproduction_policy_expectations() -> Vec<PolicyExpectation> {
    REPRODUCTION_POLICY_FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("policy="))
        .map(|line| {
            let mut maturity_periods = None;
            let mut depth_counts = [0usize; 5];
            let mut expected_total_handoffs = None;
            for field in line.split('|') {
                let (key, value) = field.split_once(':').unwrap();
                match key {
                    "maturity_periods" => maturity_periods = Some(value.parse().unwrap()),
                    "depth1_cases" => depth_counts[1] = value.parse().unwrap(),
                    "depth2_cases" => depth_counts[2] = value.parse().unwrap(),
                    "depth3_cases" => depth_counts[3] = value.parse().unwrap(),
                    "depth4_cases" => depth_counts[4] = value.parse().unwrap(),
                    "expected_total_handoffs" => {
                        expected_total_handoffs = Some(value.parse().unwrap())
                    }
                    other => panic!("unknown policy fixture field {other}"),
                }
            }
            PolicyExpectation {
                maturity_periods: maturity_periods.unwrap(),
                depth_counts,
                expected_total_handoffs: expected_total_handoffs.unwrap(),
            }
        })
        .collect()
}

fn static_generation_depth_under_policy(
    case: &SuccessorDepthCase,
    maturity_periods: u64,
) -> (u64, usize) {
    assert!(maturity_periods > 0);
    let handoff_tick = case.recovery_tick + 1;
    let mut forge = case.tooling_stock - handoff_tick;
    let mut reactor = 8 - handoff_tick;
    assert!(forge > 0);
    assert!(reactor > 0);

    let mut generation = 3u64;
    let mut parent_binding = "genome:manta-v2-success-control".to_string();
    let mut realized_generations = 1u64;
    let mut successful_handoffs = 1usize; // parent v2 -> v3

    loop {
        let generation_name = format!("v{generation}");
        let current_model = model(&generation_name, forge, 0, reactor, 18);
        let current_support = support(&generation_name);
        let current_genome = genome(&generation_name, Some(parent_binding.as_str()));
        let current_profile = profile(&generation_name);
        let report = evaluate_regenerative_lineage_viability(
            &current_profile,
            &current_genome,
            &current_model,
            &current_support,
        )
        .unwrap();
        let current_horizon = forge.min(reactor);
        assert_eq!(
            report.regenerative_viability_horizon,
            RegenerativeHorizon::FinitePeriods(current_horizon)
        );

        // A generation may survive exactly `maturity_periods` complete periods and
        // still be unable to reproduce if doing so consumes the final bootstrap unit.
        if current_horizon <= maturity_periods {
            break;
        }

        let next_forge = forge - maturity_periods;
        let next_reactor = reactor - maturity_periods;
        assert!(next_forge > 0);
        assert!(next_reactor > 0);
        let successor_generation = generation + 1;
        let successor_name = format!("v{successor_generation}");
        let successor_model = model(&successor_name, next_forge, 0, next_reactor, 18);
        let successor_genome = genome(
            &successor_name,
            Some(current_genome.evidence_binding.as_str()),
        );
        let mut handoff = semantic_descendant_handoff(
            case,
            generation,
            successor_generation,
            &current_genome,
            &successor_genome,
        );
        handoff.dynamic_handoff_receipt_binding = format!(
            "symtropy-pr:794:{SYMTROPY_POLICY_HEAD}:maturity-{maturity_periods}:v{generation}-v{successor_generation}:r{}-t{}",
            case.recovery_tick, case.tooling_stock
        );
        handoff.evidence_binding = format!(
            "handoff-qualification:policy-m{maturity_periods}:v{generation}-v{successor_generation}:r{}-t{}",
            case.recovery_tick, case.tooling_stock
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
        assert_eq!(qualified.safeguarded_transfer_count, 1);

        parent_binding = current_genome.evidence_binding;
        forge = next_forge;
        reactor = next_reactor;
        generation = successor_generation;
        realized_generations += 1;
        successful_handoffs += 1;
    }

    (realized_generations, successful_handoffs)
}

#[test]
fn semantic_policy_frontier_matches_dynamic_maturity_sweep() {
    let cases = successor_depth_cases();
    let expectations = reproduction_policy_expectations();
    assert_eq!(cases.len(), 42);
    assert_eq!(expectations.len(), 4);

    let mut previous_handoffs = None;
    for expectation in expectations {
        let mut counts = [0usize; 5];
        let mut total_handoffs = 0usize;
        let mut reproductive_cases = 0usize;

        for case in &cases {
            let Some(temporal_horizon) = case.expected_successor_horizon else {
                continue;
            };
            reproductive_cases += 1;
            let (generation_depth, handoffs) =
                static_generation_depth_under_policy(case, expectation.maturity_periods);
            assert!(generation_depth <= temporal_horizon);
            counts[generation_depth as usize] += 1;
            total_handoffs += handoffs;
        }

        assert_eq!(reproductive_cases, 18);
        assert_eq!(counts, expectation.depth_counts);
        assert_eq!(total_handoffs, expectation.expected_total_handoffs);
        if let Some(previous) = previous_handoffs {
            assert!(total_handoffs < previous);
        }
        previous_handoffs = Some(total_handoffs);
    }
}

#[test]
fn temporal_runway_equivalence_requires_the_one_period_policy() {
    for case in successor_depth_cases()
        .into_iter()
        .filter(|case| case.expected_successor_horizon.is_some())
    {
        let temporal_horizon = case.expected_successor_horizon.unwrap();
        let (one_period_depth, _) = static_generation_depth_under_policy(&case, 1);
        assert_eq!(one_period_depth, temporal_horizon);

        let (two_period_depth, _) = static_generation_depth_under_policy(&case, 2);
        assert!(two_period_depth <= temporal_horizon);
    }
}
