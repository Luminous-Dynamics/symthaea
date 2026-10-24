//! Cost-sensitive evidence selection under non-uniform priors.
//!
//! This is a finite deterministic benchmark for action-ranking semantics. The score
//! is expected reduction in multiclass Brier risk under a declared hypothesis prior;
//! it is not expected information gain over stochastic observations.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
struct CandidateAction {
    id: &'static str,
    /// One deterministic outcome per hypothesis, in the prior-weight order below.
    outcomes: [&'static str; 3],
    cost_units: u32,
    /// Supplied by the frozen test scenario; not a cryptographic validation result.
    provenance_validated: bool,
}

const HYPOTHESIS_PRIOR_WEIGHTS: [u64; 3] = [6, 3, 1];

fn expected_brier_reduction(action: CandidateAction) -> f64 {
    expected_brier_reduction_with_prior(action, HYPOTHESIS_PRIOR_WEIGHTS)
}

fn expected_brier_reduction_with_prior(action: CandidateAction, prior_weights: [u64; 3]) -> f64 {
    if !action.provenance_validated {
        return 0.0;
    }

    let total_weight: u64 = prior_weights.iter().sum();
    let mut group_weights: BTreeMap<&str, u64> = BTreeMap::new();
    let mut group_square_sums: BTreeMap<&str, u128> = BTreeMap::new();
    let mut prior_square_sum = 0_u128;

    for (weight, outcome) in prior_weights.iter().copied().zip(action.outcomes) {
        let weight_squared = u128::from(weight) * u128::from(weight);
        *group_weights.entry(outcome).or_default() += weight;
        *group_square_sums.entry(outcome).or_default() += weight_squared;
        prior_square_sum += weight_squared;
    }

    let total = f64::from(total_weight as u32);
    let prior_concentration = prior_square_sum as f64 / (total * total);
    let expected_posterior_concentration: f64 = group_square_sums
        .iter()
        .map(|(outcome, square_sum)| {
            let group_weight = *group_weights.get(*outcome).expect("group exists") as u32;
            *square_sum as f64 / (total * f64::from(group_weight))
        })
        .sum();

    (expected_posterior_concentration - prior_concentration).max(0.0)
}

fn choose_action(actions: &[CandidateAction]) -> Option<CandidateAction> {
    choose_action_with_prior(actions, HYPOTHESIS_PRIOR_WEIGHTS)
}

fn choose_action_with_prior(
    actions: &[CandidateAction],
    prior_weights: [u64; 3],
) -> Option<CandidateAction> {
    let mut best: Option<(CandidateAction, f64)> = None;

    for action in actions.iter().copied() {
        if action.cost_units == 0 {
            continue;
        }
        let gain_per_cost = expected_brier_reduction_with_prior(action, prior_weights)
            / f64::from(action.cost_units);
        if gain_per_cost <= 1.0e-12 {
            continue;
        }

        let replace = match best {
            None => true,
            Some((current, current_gain)) => {
                gain_per_cost > current_gain + 1.0e-12
                    || ((gain_per_cost - current_gain).abs() <= 1.0e-12 && action.id < current.id)
            }
        };
        if replace {
            best = Some((action, gain_per_cost));
        }
    }

    best.map(|(action, _)| action)
}

fn actions() -> [CandidateAction; 5] {
    [
        CandidateAction {
            id: "repeat-current-geometry",
            outcomes: ["table-shaped", "table-shaped", "table-shaped"],
            cost_units: 10,
            provenance_validated: true,
        },
        CandidateAction {
            id: "validated-inventory-family",
            outcomes: ["chair", "other", "other"],
            cost_units: 1_000,
            provenance_validated: true,
        },
        CandidateAction {
            id: "validated-design-record",
            outcomes: ["chair", "table", "stool"],
            cost_units: 2_000,
            provenance_validated: true,
        },
        CandidateAction {
            id: "unverified-target-label",
            outcomes: ["chair", "table", "stool"],
            cost_units: 1,
            provenance_validated: false,
        },
        CandidateAction {
            id: "validated-source-with-zero-cost",
            outcomes: ["chair", "table", "stool"],
            cost_units: 0,
            provenance_validated: true,
        },
    ]
}

#[test]
fn nonuniform_prior_and_three_hypotheses_change_expected_decision_value() {
    let actions = actions();
    let repeated_scan = actions
        .iter()
        .find(|action| action.id == "repeat-current-geometry")
        .expect("repeat scan exists");
    let inventory = actions
        .iter()
        .find(|action| action.id == "validated-inventory-family")
        .expect("inventory action exists");
    let design_record = actions
        .iter()
        .find(|action| action.id == "validated-design-record")
        .expect("design record exists");

    assert!((expected_brier_reduction(*repeated_scan) - 0.0).abs() <= 1.0e-12);
    assert!((expected_brier_reduction(*inventory) - 0.39).abs() <= 1.0e-12);
    assert!((expected_brier_reduction(*design_record) - 0.54).abs() <= 1.0e-12);
}

#[test]
fn cost_sensitive_selector_can_prefer_partial_but_cheaper_evidence() {
    let selected = choose_action(&actions()).expect("validated informative action exists");
    assert_eq!(selected.id, "validated-inventory-family");
}

#[test]
fn unverified_or_zero_cost_actions_are_not_ranked_as_authoritative_measurements() {
    let actions = actions();
    let unverified = actions
        .iter()
        .find(|action| action.id == "unverified-target-label")
        .expect("unverified action exists");
    let zero_cost = actions
        .iter()
        .find(|action| action.id == "validated-source-with-zero-cost")
        .expect("zero-cost action exists");

    assert_eq!(expected_brier_reduction(*unverified), 0.0);
    assert!((expected_brier_reduction(*zero_cost) - 0.54).abs() <= 1.0e-12);
    assert_eq!(choose_action(&[actions[0], *unverified, *zero_cost]), None);
}

#[test]
fn deterministic_ties_use_a_stable_id_order() {
    let tied = [
        CandidateAction {
            id: "source-b",
            outcomes: ["chair", "table", "stool"],
            cost_units: 2_000,
            provenance_validated: true,
        },
        CandidateAction {
            id: "source-a",
            outcomes: ["chair", "table", "stool"],
            cost_units: 2_000,
            provenance_validated: true,
        },
    ];
    assert_eq!(
        choose_action(&tied).map(|action| action.id),
        Some("source-a")
    );
}

fn assert_all_action_orderings(
    actions: &mut [CandidateAction; 5],
    start: usize,
    checked: &mut usize,
) {
    if start == actions.len() {
        assert_eq!(
            choose_action(actions).map(|action| action.id),
            Some("validated-inventory-family")
        );
        *checked += 1;
        return;
    }

    for index in start..actions.len() {
        actions.swap(start, index);
        assert_all_action_orderings(actions, start + 1, checked);
        actions.swap(start, index);
    }
}

#[test]
fn action_ranking_is_invariant_to_all_120_input_orders() {
    let mut candidates = actions();
    let mut checked = 0;
    assert_all_action_orderings(&mut candidates, 0, &mut checked);
    assert_eq!(checked, 120);
}

#[test]
fn brier_reduction_and_action_choice_are_invariant_to_hypothesis_permutation() {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let original_actions = actions();
    let expected_choice = choose_action(&original_actions).map(|action| action.id);

    for permutation in PERMUTATIONS {
        let permuted_prior = permutation.map(|index| HYPOTHESIS_PRIOR_WEIGHTS[index]);
        let permuted_actions: Vec<_> = original_actions
            .iter()
            .copied()
            .map(|mut action| {
                action.outcomes = permutation.map(|index| action.outcomes[index]);
                action
            })
            .collect();

        for (original, permuted) in original_actions.iter().zip(&permuted_actions) {
            let expected = expected_brier_reduction(*original);
            let observed = expected_brier_reduction_with_prior(*permuted, permuted_prior);
            assert!(
                (expected - observed).abs() <= 1.0e-12,
                "{} changed Brier reduction under a hypothesis permutation",
                original.id
            );
        }

        assert_eq!(
            choose_action_with_prior(&permuted_actions, permuted_prior).map(|action| action.id),
            expected_choice
        );
    }
}

fn renamed_outcome(outcome: &str) -> &'static str {
    match outcome {
        "chair" => "label-a",
        "other" => "label-b",
        "table" => "label-c",
        "stool" => "label-d",
        "table-shaped" => "label-e",
        _ => panic!("unexpected fixture outcome: {outcome}"),
    }
}

#[test]
fn brier_reduction_depends_on_outcome_partition_not_label_spelling() {
    let original_actions = actions();
    let renamed_actions: Vec<_> = original_actions
        .iter()
        .copied()
        .map(|mut action| {
            action.outcomes = action.outcomes.map(renamed_outcome);
            action
        })
        .collect();

    for (original, renamed) in original_actions.iter().zip(&renamed_actions) {
        assert!(
            (expected_brier_reduction(*original) - expected_brier_reduction(*renamed)).abs()
                <= 1.0e-12,
            "{} changed Brier reduction under a bijective outcome rename",
            original.id
        );
    }

    assert_eq!(
        choose_action(&original_actions).map(|action| action.id),
        choose_action(&renamed_actions).map(|action| action.id)
    );
}
