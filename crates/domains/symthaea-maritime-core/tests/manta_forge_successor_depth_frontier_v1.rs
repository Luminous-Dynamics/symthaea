include!("manta_forge_lineage_success_control_v1.rs");

const SUCCESSOR_DEPTH_FIXTURE: &str =
    include_str!("../fixtures/manta-forge-successor-depth-frontier-v1.txt");

#[derive(Debug, Clone, PartialEq, Eq)]
struct SuccessorDepthCase {
    recovery_tick: u64,
    tooling_stock: u64,
    expected_successor_horizon: Option<u64>,
}

fn parse_depth(value: &str) -> Option<u64> {
    if value == "none" {
        None
    } else {
        Some(value.parse().unwrap())
    }
}

fn successor_depth_cases() -> Vec<SuccessorDepthCase> {
    SUCCESSOR_DEPTH_FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("case="))
        .map(|line| {
            let mut recovery_tick = None;
            let mut tooling_stock = None;
            let mut expected_successor_horizon = None;
            for field in line.split('|') {
                let (key, value) = field.split_once(':').unwrap();
                match key {
                    "recovery_tick" => recovery_tick = Some(value.parse().unwrap()),
                    "tooling_stock" => tooling_stock = Some(value.parse().unwrap()),
                    "expected_successor_horizon" => {
                        expected_successor_horizon = Some(parse_depth(value))
                    }
                    other => panic!("unknown successor-depth field {other}"),
                }
            }
            SuccessorDepthCase {
                recovery_tick: recovery_tick.unwrap(),
                tooling_stock: tooling_stock.unwrap(),
                expected_successor_horizon: expected_successor_horizon.unwrap(),
            }
        })
        .collect()
}

#[test]
fn static_successor_viability_matches_dynamic_depth_frontier() {
    let cases = successor_depth_cases();
    assert_eq!(cases.len(), 42);

    let mut exact_horizon_counts = [0usize; 5];
    for case in &cases {
        let Some(expected_horizon) = case.expected_successor_horizon else {
            continue;
        };

        let handoff_tick = case.recovery_tick + 1;
        let transferred_forge = case.tooling_stock - handoff_tick;
        let transferred_reactor = 8 - handoff_tick;
        assert!(transferred_forge > 0);
        assert!(transferred_reactor > 0);

        let v3_model = model(
            "v3",
            transferred_forge,
            0,
            transferred_reactor,
            18,
        );
        let v3_support = support("v3");
        let v3_genome = genome("v3", Some("genome:manta-v2-success-control"));
        let v3_profile = profile("v3");
        let report = evaluate_regenerative_lineage_viability(
            &v3_profile,
            &v3_genome,
            &v3_model,
            &v3_support,
        )
        .unwrap();

        assert_eq!(
            report.regenerative_viability_horizon,
            RegenerativeHorizon::FinitePeriods(expected_horizon),
            "static successor horizon mismatch for {case:?}"
        );
        assert_eq!(
            report.successor_reproduction_horizon,
            RegenerativeHorizon::FinitePeriods(expected_horizon),
            "static successor reproduction horizon mismatch for {case:?}"
        );
        assert_eq!(
            report.operation.conservative_horizon,
            RegenerativeHorizon::FinitePeriods(expected_horizon),
            "static successor operation horizon mismatch for {case:?}"
        );

        exact_horizon_counts[expected_horizon as usize] += 1;
    }

    assert_eq!(exact_horizon_counts[1], 6);
    assert_eq!(exact_horizon_counts[2], 5);
    assert_eq!(exact_horizon_counts[3], 4);
    assert_eq!(exact_horizon_counts[4], 3);
    assert_eq!(exact_horizon_counts[1..].iter().sum::<usize>(), 18);
    assert_eq!(exact_horizon_counts[2..].iter().sum::<usize>(), 12);
    assert_eq!(exact_horizon_counts[3..].iter().sum::<usize>(), 7);
    assert_eq!(exact_horizon_counts[4..].iter().sum::<usize>(), 3);
}
