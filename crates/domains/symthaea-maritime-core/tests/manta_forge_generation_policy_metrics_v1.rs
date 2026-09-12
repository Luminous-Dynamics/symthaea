include!("manta_forge_generation_policy_projection_v1.rs");

const POLICY_METRICS_FIXTURE: &str =
    include_str!("../fixtures/manta-forge-generation-policy-metrics-v1.txt");

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedPolicyMetrics {
    maturity_periods: u64,
    founded_descendants: u64,
    maturity_completed_descendants: u64,
    descendant_reproduction_transitions: u64,
}

fn expected_policy_metrics() -> Vec<ExpectedPolicyMetrics> {
    POLICY_METRICS_FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("policy="))
        .map(|line| {
            let mut maturity_periods = None;
            let mut founded_descendants = None;
            let mut maturity_completed_descendants = None;
            let mut descendant_reproduction_transitions = None;
            for field in line.split('|') {
                let (key, value) = field.split_once(':').unwrap();
                match key {
                    "maturity_periods" => maturity_periods = Some(value.parse().unwrap()),
                    "founded_descendants" => founded_descendants = Some(value.parse().unwrap()),
                    "maturity_completed_descendants" => {
                        maturity_completed_descendants = Some(value.parse().unwrap())
                    }
                    "descendant_reproduction_transitions" => {
                        descendant_reproduction_transitions = Some(value.parse().unwrap())
                    }
                    other => panic!("unknown policy metric field {other}"),
                }
            }
            ExpectedPolicyMetrics {
                maturity_periods: maturity_periods.unwrap(),
                founded_descendants: founded_descendants.unwrap(),
                maturity_completed_descendants: maturity_completed_descendants.unwrap(),
                descendant_reproduction_transitions: descendant_reproduction_transitions.unwrap(),
            }
        })
        .collect()
}

#[test]
fn closed_form_policy_metrics_match_dynamic_metric_fixture() {
    let horizons = temporal_horizons();
    let expectations = expected_policy_metrics();
    assert_eq!(horizons.len(), 18);
    assert_eq!(expectations.len(), 4);

    for expected in expectations {
        let mut founded = 0u64;
        let mut matured = 0u64;
        let mut transitions = 0u64;
        for horizon in &horizons {
            let projection = project_regenerative_generation_policy(
                RegenerativeHorizon::FinitePeriods(*horizon),
                expected.maturity_periods,
            )
            .unwrap();
            founded += finite(projection.founded_descendant_generations);
            matured += finite(projection.maturity_completed_descendant_generations);
            transitions += finite(projection.descendant_reproduction_transitions);
        }

        assert_eq!(founded, expected.founded_descendants);
        assert_eq!(matured, expected.maturity_completed_descendants);
        assert_eq!(transitions, expected.descendant_reproduction_transitions);
        assert_eq!(founded, 18 + transitions);
        assert!(matured <= founded);
    }
}
