//! Cost-sensitive evidence selection under non-uniform priors.
//!
//! This is a finite deterministic benchmark for action-ranking semantics. The score
//! is expected reduction in multiclass Brier risk under a declared hypothesis prior;
//! it is not expected information gain over stochastic observations.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
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
    if !action.provenance_validated || action.cost_units == 0 {
        return 0.0;
    }

    let total_weight: u64 = HYPOTHESIS_PRIOR_WEIGHTS.iter().sum();
    let mut group_weights: BTreeMap<&str, u64> = BTreeMap::new();
    let mut group_square_sums: BTreeMap<&str, u128> = BTreeMap::new();
    let mut prior_square_sum = 0_u128;

    for (weight, outcome) in HYPOTHESIS_PRIOR_WEIGHTS
        .iter()
        .copied()
        .zip(action.outcomes)
    {
        *group_weights.entry(outcome).or_default() += weight;
        *group_square_sums.entry(outcome).or_default() += u128::from(weight) * u128::from(weight);
        prior_square_sum += u128::from(weight) * u128::from(weight);
    }

    let total = f64::from(total_weight as u32);
    let prior_concentration = prior_square_sum as f64 / (total * total);
    let expected_posterior_concentration: f64 = group_square_sums
        .iter()
        .map(|(outcome, square_sum)| {
            *square_sum as f64 / (total * f64::from(*group_weights.get(outcome).expect("group exists") as u32))
        })
        .sum();

    (expected_posterior_concentration - prior_concentration).max(0.0)
}

fn choose_action(actions: &[CandidateAction]) -> Option<CandidateAction> {
    let mut best: Option<(CandidateAction, f64)> = None;

    for action in actions.iter().copied() {
        if action.cost_units == 0 {
            continue;
        }
        let gain_per_cost = expected_brier_reduction(action) / f64::from(action.cost_units);
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
    assert_eq!(expected_brier_reduction(*zero_cost), 0.0);
    assert_eq!(
        choose_action(&[
            actions[0],
            *unverified,
            *zero_cost,
        ]),
        None
    );
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
    assert_eq!(choose_action(&tied).map(|action| action.id), Some("source-a"));
}
