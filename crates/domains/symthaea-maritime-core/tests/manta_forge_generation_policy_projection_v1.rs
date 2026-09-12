use symthaea_maritime_core::{
    project_regenerative_generation_policy, RegenerativeGenerationCountV1,
    RegenerativeHorizon,
};

const DEPTH_FIXTURE: &str =
    include_str!("../fixtures/manta-forge-successor-depth-frontier-v1.txt");
const POLICY_FIXTURE: &str =
    include_str!("../fixtures/manta-forge-reproduction-policy-frontier-v1.txt");

fn finite(value: RegenerativeGenerationCountV1) -> u64 {
    match value {
        RegenerativeGenerationCountV1::Finite(value) => value,
        RegenerativeGenerationCountV1::IndefiniteUnderStaticModel => {
            panic!("fixture projection must be finite")
        }
    }
}

fn temporal_horizons() -> Vec<u64> {
    DEPTH_FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("case="))
        .filter_map(|line| {
            let value = line
                .split('|')
                .find_map(|field| {
                    let (key, value) = field.split_once(':').unwrap();
                    (key == "expected_successor_horizon").then_some(value)
                })
                .unwrap();
            (value != "none").then(|| value.parse().unwrap())
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PolicyRow {
    maturity_periods: u64,
    expected_depth_counts: [u64; 5],
    expected_total_handoffs: u64,
}

fn policy_rows() -> Vec<PolicyRow> {
    POLICY_FIXTURE
        .lines()
        .filter_map(|line| line.strip_prefix("policy="))
        .map(|line| {
            let mut maturity_periods = None;
            let mut expected_depth_counts = [0u64; 5];
            let mut expected_total_handoffs = None;
            for field in line.split('|') {
                let (key, value) = field.split_once(':').unwrap();
                match key {
                    "maturity_periods" => maturity_periods = Some(value.parse().unwrap()),
                    "depth1_cases" => expected_depth_counts[1] = value.parse().unwrap(),
                    "depth2_cases" => expected_depth_counts[2] = value.parse().unwrap(),
                    "depth3_cases" => expected_depth_counts[3] = value.parse().unwrap(),
                    "depth4_cases" => expected_depth_counts[4] = value.parse().unwrap(),
                    "expected_total_handoffs" => {
                        expected_total_handoffs = Some(value.parse().unwrap())
                    }
                    other => panic!("unknown policy fixture field {other}"),
                }
            }
            PolicyRow {
                maturity_periods: maturity_periods.unwrap(),
                expected_depth_counts,
                expected_total_handoffs: expected_total_handoffs.unwrap(),
            }
        })
        .collect()
}

#[test]
fn closed_form_projection_reproduces_every_policy_frontier_row() {
    let horizons = temporal_horizons();
    let rows = policy_rows();
    assert_eq!(horizons.len(), 18);
    assert_eq!(rows.len(), 4);

    for row in rows {
        let mut founded_depth_counts = [0u64; 5];
        let mut total_founded = 0u64;
        for horizon in &horizons {
            let projection = project_regenerative_generation_policy(
                RegenerativeHorizon::FinitePeriods(*horizon),
                row.maturity_periods,
            )
            .unwrap();
            let founded = finite(projection.founded_descendant_generations);
            founded_depth_counts[founded as usize] += 1;
            total_founded += founded;
        }
        assert_eq!(founded_depth_counts, row.expected_depth_counts);
        assert_eq!(total_founded, row.expected_total_handoffs);
    }
}

#[test]
fn terminal_generation_semantics_are_not_hidden_by_founded_depth() {
    let horizons = temporal_horizons();

    // With four-period maturity, all 18 cases have a founded v3, but only the
    // three H=4 cases actually complete the maturity interval. Those three still
    // cannot found v4 because completing maturity consumes their final finite reserve.
    let mut founded_total = 0u64;
    let mut matured_total = 0u64;
    let mut transition_total = 0u64;
    for horizon in &horizons {
        let projection = project_regenerative_generation_policy(
            RegenerativeHorizon::FinitePeriods(*horizon),
            4,
        )
        .unwrap();
        founded_total += finite(projection.founded_descendant_generations);
        matured_total += finite(projection.maturity_completed_descendant_generations);
        transition_total += finite(projection.descendant_reproduction_transitions);
    }
    assert_eq!(founded_total, 18);
    assert_eq!(matured_total, 3);
    assert_eq!(transition_total, 0);
}

#[test]
fn two_period_policy_exposes_terminal_immaturity_separately_from_reproduction() {
    let horizons = temporal_horizons();
    let mut founded_total = 0u64;
    let mut matured_total = 0u64;
    let mut transition_total = 0u64;
    for horizon in &horizons {
        let projection = project_regenerative_generation_policy(
            RegenerativeHorizon::FinitePeriods(*horizon),
            2,
        )
        .unwrap();
        founded_total += finite(projection.founded_descendant_generations);
        matured_total += finite(projection.maturity_completed_descendant_generations);
        transition_total += finite(projection.descendant_reproduction_transitions);
    }

    assert_eq!(founded_total, 25);
    assert_eq!(matured_total, 15);
    assert_eq!(transition_total, 7);
    assert_eq!(founded_total, 18 + transition_total);
}
