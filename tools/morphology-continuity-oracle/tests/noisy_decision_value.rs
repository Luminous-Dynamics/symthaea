//! Decision-sensitive measurement selection with declared noisy-observation likelihoods.
//!
//! Each measurement provides a discrete likelihood table P(outcome | hypothesis).
//! The reference oracle marginalizes the possible outcomes, selects the lowest
//! expected-loss declared action for each outcome, and computes expected value of
//! sample information (EVSI). Likelihoods, losses, and provenance status are fixture
//! inputs; this does not implement live source authentication or a production policy.

const PRIOR_WEIGHTS: [u64; 3] = [6, 3, 1];
const PROBABILITY_SCALE: u64 = 1_000;
const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Measurement {
    id: &'static str,
    /// Shared label for each possible output column.
    outcomes: [&'static str; 3],
    /// Rows follow [chair, table, stool]; columns follow outcomes above.
    likelihood_per_mille: [[u16; 3]; 3],
    cost_milliunits: u32,
    /// Fixture input only; no signature or provenance is verified here.
    provenance_validated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Decision {
    /// Losses ordered by [chair, table, stool].
    losses: [u32; 3],
}

fn decisions() -> [Decision; 2] {
    [
        Decision {
            losses: [0, 100, 100],
        },
        Decision { losses: [4, 4, 4] },
    ]
}

fn measurements() -> [Measurement; 6] {
    [
        Measurement {
            id: "repeat-geometry",
            outcomes: ["table-shaped", "not-observed", "unused"],
            likelihood_per_mille: [[1_000, 0, 0], [1_000, 0, 0], [1_000, 0, 0]],
            cost_milliunits: 1_000,
            provenance_validated: true,
        },
        Measurement {
            id: "noisy-inventory-family",
            outcomes: ["chair-like", "not-chair", "unused"],
            likelihood_per_mille: [[950, 50, 0], [50, 950, 0], [50, 950, 0]],
            cost_milliunits: 1_000,
            provenance_validated: true,
        },
        Measurement {
            id: "noisy-design-record",
            outcomes: ["chair-record", "table-record", "stool-record"],
            likelihood_per_mille: [[900, 50, 50], [50, 900, 50], [50, 100, 850]],
            cost_milliunits: 1_250,
            provenance_validated: true,
        },
        Measurement {
            id: "weak-noisy-inventory-family",
            outcomes: ["chair-like", "not-chair", "unused"],
            likelihood_per_mille: [[800, 200, 0], [100, 900, 0], [200, 800, 0]],
            cost_milliunits: 1_000,
            provenance_validated: true,
        },
        Measurement {
            id: "unverified-record",
            outcomes: ["chair-record", "table-record", "stool-record"],
            likelihood_per_mille: [[900, 50, 50], [50, 900, 50], [50, 100, 850]],
            cost_milliunits: 1,
            provenance_validated: false,
        },
        Measurement {
            id: "zero-cost-record",
            outcomes: ["chair-record", "table-record", "stool-record"],
            likelihood_per_mille: [[900, 50, 50], [50, 900, 50], [50, 100, 850]],
            cost_milliunits: 0,
            provenance_validated: true,
        },
    ]
}

fn valid_likelihoods(measurement: Measurement) -> bool {
    // The finite request requires distinct declared output labels and rows summing
    // exactly to the integer probability scale. Malformed models fail closed.
    for (index, outcome) in measurement.outcomes.iter().enumerate() {
        if outcome.is_empty() || measurement.outcomes[index + 1..].contains(outcome) {
            return false;
        }
    }
    measurement.likelihood_per_mille.iter().all(|row| {
        row.iter()
            .map(|probability| u32::from(*probability))
            .sum::<u32>()
            == PROBABILITY_SCALE as u32
    })
}

fn baseline_expected_loss(weights: [u64; 3], decisions: &[Decision]) -> f64 {
    let total_weight: f64 = weights.iter().map(|weight| *weight as f64).sum();
    decisions
        .iter()
        .map(|decision| {
            weights
                .iter()
                .zip(decision.losses)
                .map(|(weight, loss)| *weight as f64 * f64::from(loss))
                .sum::<f64>()
                / total_weight
        })
        .fold(f64::INFINITY, f64::min)
}

/// Marginal expected loss after a noisy observation, with optimal declared action
/// selection per possible output. Returns baseline loss for untrusted/invalid inputs.
fn expected_loss_after_measurement(
    measurement: Measurement,
    weights: [u64; 3],
    decisions: &[Decision],
) -> f64 {
    let baseline = baseline_expected_loss(weights, decisions);
    if !measurement.provenance_validated || !valid_likelihoods(measurement) {
        return baseline;
    }

    let total_weight: f64 = weights.iter().map(|weight| *weight as f64).sum();
    let mut expected_loss = 0.0;

    for outcome_index in 0..measurement.outcomes.len() {
        // Each term is the unnormalized joint mass P(h) * P(o | h), represented
        // with integer prior weights and per-mille likelihoods.
        let outcome_mass: u64 = weights
            .iter()
            .zip(measurement.likelihood_per_mille.iter())
            .map(|(weight, row)| *weight * u64::from(row[outcome_index]))
            .sum();
        if outcome_mass == 0 {
            continue;
        }

        let best_joint_loss = decisions
            .iter()
            .map(|decision| {
                weights
                    .iter()
                    .zip(decision.losses.iter())
                    .zip(measurement.likelihood_per_mille.iter())
                    .map(|((weight, loss), row)| {
                        *weight as f64 * f64::from(*loss) * f64::from(row[outcome_index])
                    })
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);

        expected_loss += best_joint_loss / (total_weight * PROBABILITY_SCALE as f64);
    }

    expected_loss
}

fn evsi(measurement: Measurement, weights: [u64; 3], decisions: &[Decision]) -> f64 {
    if !measurement.provenance_validated || !valid_likelihoods(measurement) {
        return 0.0;
    }
    (baseline_expected_loss(weights, decisions)
        - expected_loss_after_measurement(measurement, weights, decisions))
    .max(0.0)
}

fn choose_by_evsi_per_cost(
    candidates: &[Measurement],
    weights: [u64; 3],
    decisions: &[Decision],
) -> Option<Measurement> {
    let mut best: Option<(Measurement, f64)> = None;

    for candidate in candidates.iter().copied() {
        if candidate.cost_milliunits == 0 || !candidate.provenance_validated {
            continue;
        }
        let value = evsi(candidate, weights, decisions);
        if value <= EPSILON {
            continue;
        }
        let ratio = value / f64::from(candidate.cost_milliunits);

        let replace = match best {
            None => true,
            Some((current, current_ratio)) => {
                ratio > current_ratio + EPSILON
                    || ((ratio - current_ratio).abs() <= EPSILON && candidate.id < current.id)
            }
        };
        if replace {
            best = Some((candidate, ratio));
        }
    }

    best.map(|(candidate, _)| candidate)
}

#[test]
fn noisy_likelihoods_change_downstream_risk_and_choose_best_value_per_cost() {
    let candidates = measurements();
    let decisions = decisions();
    let baseline = baseline_expected_loss(PRIOR_WEIGHTS, &decisions);
    let inventory = candidates[1];
    let design = candidates[2];
    let weak = candidates[3];

    assert!((baseline - 4.0).abs() <= EPSILON);
    assert!(
        (expected_loss_after_measurement(inventory, PRIOR_WEIGHTS, &decisions) - 3.64).abs()
            <= EPSILON
    );
    assert!(
        (expected_loss_after_measurement(design, PRIOR_WEIGHTS, &decisions) - 3.76).abs()
            <= EPSILON
    );
    assert!((evsi(inventory, PRIOR_WEIGHTS, &decisions) - 0.36).abs() <= EPSILON);
    assert!((evsi(design, PRIOR_WEIGHTS, &decisions) - 0.24).abs() <= EPSILON);
    assert!((evsi(weak, PRIOR_WEIGHTS, &decisions) - 0.0).abs() <= EPSILON);

    assert_eq!(
        choose_by_evsi_per_cost(&candidates, PRIOR_WEIGHTS, &decisions).map(|item| item.id),
        Some("noisy-inventory-family")
    );
}

#[test]
fn weaker_but_valid_measurement_can_have_zero_decision_value() {
    let candidates = measurements();
    let weak = candidates[3];
    let decisions = decisions();

    assert!(valid_likelihoods(weak));
    assert!((evsi(weak, PRIOR_WEIGHTS, &decisions)).abs() <= EPSILON);
    assert_eq!(
        choose_by_evsi_per_cost(&[weak, candidates[0]], PRIOR_WEIGHTS, &decisions),
        None
    );
}

#[test]
fn malformed_likelihood_table_and_unverified_source_fail_closed() {
    let candidates = measurements();
    let decisions = decisions();
    let mut malformed = candidates[1];
    malformed.likelihood_per_mille[0][0] = 949;

    assert!(!valid_likelihoods(malformed));
    assert!((evsi(malformed, PRIOR_WEIGHTS, &decisions)).abs() <= EPSILON);
    assert!((evsi(candidates[4], PRIOR_WEIGHTS, &decisions)).abs() <= EPSILON);
    assert_eq!(
        choose_by_evsi_per_cost(
            &[malformed, candidates[4], candidates[5]],
            PRIOR_WEIGHTS,
            &decisions,
        ),
        None
    );
}

#[test]
fn joint_hypothesis_permutations_preserve_risk_and_action_choice() {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let original_measurements = measurements();
    let original_decisions = decisions();
    let expected_choice =
        choose_by_evsi_per_cost(&original_measurements, PRIOR_WEIGHTS, &original_decisions)
            .map(|item| item.id);
    let baseline = baseline_expected_loss(PRIOR_WEIGHTS, &original_decisions);

    for permutation in PERMUTATIONS {
        let weights = permutation.map(|index| PRIOR_WEIGHTS[index]);
        let permuted_measurements = original_measurements.map(|mut measurement| {
            measurement.likelihood_per_mille =
                permutation.map(|index| measurement.likelihood_per_mille[index]);
            measurement
        });
        let permuted_decisions = original_decisions.map(|mut decision| {
            decision.losses = permutation.map(|index| decision.losses[index]);
            decision
        });

        assert!((baseline_expected_loss(weights, &permuted_decisions) - baseline).abs() <= EPSILON);
        for (before, after) in original_measurements.iter().zip(&permuted_measurements) {
            assert!(
                (evsi(*before, PRIOR_WEIGHTS, &original_decisions)
                    - evsi(*after, weights, &permuted_decisions))
                .abs()
                    <= EPSILON,
                "{} changed EVSI under joint hypothesis permutation",
                before.id
            );
        }
        assert_eq!(
            choose_by_evsi_per_cost(&permuted_measurements, weights, &permuted_decisions,)
                .map(|item| item.id),
            expected_choice
        );
    }
}

fn symmetric_binary_sensor(reliability: u16) -> Measurement {
    assert!(u64::from(reliability) <= PROBABILITY_SCALE);
    let error_rate = (PROBABILITY_SCALE - u64::from(reliability)) as u16;
    Measurement {
        id: "symmetric-reliability-sweep",
        outcomes: ["chair-like", "not-chair", "unused"],
        likelihood_per_mille: [
            [reliability, error_rate, 0],
            [error_rate, reliability, 0],
            [error_rate, reliability, 0],
        ],
        cost_milliunits: 1_000,
        provenance_validated: true,
    }
}

#[test]
fn decision_value_has_a_measurable_reliability_threshold() {
    let candidates = [
        (900, 0.0),
        (920, 0.0),
        (930, 0.0),
        (940, 0.0),
        (941, 0.0),
        (942, 0.0336),
        (950, 0.36),
        (975, 1.38),
        (990, 1.992),
        (995, 2.196),
        (1_000, 2.4),
    ];
    let decisions = decisions();
    let mut previous_value = -EPSILON;

    for (reliability, expected_value) in candidates {
        let sensor = symmetric_binary_sensor(reliability);
        assert!(valid_likelihoods(sensor));
        let value = evsi(sensor, PRIOR_WEIGHTS, &decisions);
        assert!(
            (value - expected_value).abs() <= EPSILON,
            "unexpected EVSI at reliability {reliability}: expected {expected_value}, got {value}"
        );
        assert!(
            value + EPSILON >= previous_value,
            "EVSI decreased as symmetric sensor reliability rose to {reliability}"
        );
        previous_value = value;
    }
}

#[test]
fn prior_shift_can_eliminate_the_value_of_an_otherwise_useful_sensor() {
    let decisions = decisions();
    let sensor = symmetric_binary_sensor(950);

    assert!((evsi(sensor, PRIOR_WEIGHTS, &decisions) - 0.36).abs() <= EPSILON);
    assert!((evsi(sensor, [4, 4, 2], &decisions) - 0.0).abs() <= EPSILON);
    assert!((evsi(sensor, [8, 1, 1], &decisions) - 2.08).abs() <= EPSILON);

    assert_eq!(
        choose_by_evsi_per_cost(&[sensor], [4, 4, 2], &decisions),
        None
    );
    assert_eq!(
        choose_by_evsi_per_cost(&[sensor], [8, 1, 1], &decisions).map(|item| item.id),
        Some("symmetric-reliability-sweep")
    );
}

fn robust_evsi_lower_bound(
    minimum_reliability: u16,
    maximum_reliability: u16,
    weights: [u64; 3],
    decisions: &[Decision],
) -> Option<f64> {
    if minimum_reliability > maximum_reliability
        || u64::from(maximum_reliability) > PROBABILITY_SCALE
    {
        return None;
    }

    let mut lower_bound = f64::INFINITY;
    for reliability in minimum_reliability..=maximum_reliability {
        lower_bound = lower_bound.min(evsi(
            symmetric_binary_sensor(reliability),
            weights,
            decisions,
        ));
    }
    Some(lower_bound)
}

#[test]
fn robust_selector_abstains_when_reliability_bounds_cross_the_decision_threshold() {
    let decisions = decisions();

    let nominal_wide =
        robust_evsi_lower_bound(900, 950, PRIOR_WEIGHTS, &decisions).expect("valid interval");
    let nominal_narrow =
        robust_evsi_lower_bound(950, 1_000, PRIOR_WEIGHTS, &decisions).expect("valid interval");
    let shifted_wide =
        robust_evsi_lower_bound(950, 1_000, [4, 4, 2], &decisions).expect("valid interval");
    let shifted_narrow =
        robust_evsi_lower_bound(975, 1_000, [4, 4, 2], &decisions).expect("valid interval");
    let high_chair =
        robust_evsi_lower_bound(975, 1_000, [8, 1, 1], &decisions).expect("valid interval");

    assert!(nominal_wide.abs() <= EPSILON);
    assert!((nominal_narrow - 0.36).abs() <= EPSILON);
    assert!(shifted_wide.abs() <= EPSILON);
    assert!((shifted_narrow - 0.12).abs() <= EPSILON);
    assert!((high_chair - 2.64).abs() <= EPSILON);

    assert!(
        nominal_wide <= EPSILON,
        "wide interval must cause abstention"
    );
    assert!(
        shifted_wide <= EPSILON,
        "prior-shifted interval must cause abstention"
    );
    assert!(
        nominal_narrow > EPSILON,
        "qualified narrow interval has positive worst-case value"
    );
    assert!(
        shifted_narrow > EPSILON,
        "narrower shifted interval has positive worst-case value"
    );
}

#[test]
fn invalid_reliability_bounds_are_rejected_not_clamped() {
    let decisions = decisions();
    assert_eq!(
        robust_evsi_lower_bound(950, 900, PRIOR_WEIGHTS, &decisions),
        None
    );
    assert_eq!(
        robust_evsi_lower_bound(950, 1_001, PRIOR_WEIGHTS, &decisions),
        None
    );
}

const COST_PENALTY_PER_MILLIUNIT: f64 = 0.0001;

/// Evaluate decisions made using the assumed observation model against outcomes
/// generated by a different, explicitly supplied true likelihood table.
fn expected_loss_under_assumed_policy(
    assumed: Measurement,
    actual_likelihoods: [[u16; 3]; 3],
    weights: [u64; 3],
    decisions: &[Decision],
) -> Option<f64> {
    let mut actual = assumed;
    actual.likelihood_per_mille = actual_likelihoods;

    if !assumed.provenance_validated
        || !valid_likelihoods(assumed)
        || !valid_likelihoods(actual)
    {
        return None;
    }

    let total_weight: f64 = weights.iter().map(|weight| *weight as f64).sum();
    let mut expected_loss = 0.0;

    for outcome_index in 0..assumed.outcomes.len() {
        // The deployed policy selects the action using the assumed model.
        let (selected_index, _) = decisions
            .iter()
            .enumerate()
            .map(|(index, decision)| {
                let joint_loss = weights
                    .iter()
                    .zip(decision.losses.iter())
                    .zip(assumed.likelihood_per_mille.iter())
                    .map(|((weight, loss), row)| {
                        *weight as f64 * f64::from(*loss) * f64::from(row[outcome_index])
                    })
                    .sum::<f64>();
                (index, joint_loss)
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))?;

        // The environment emits outcomes under the true model, but action choice
        // remains the one produced by the assumed model.
        let realized_joint_loss = weights
            .iter()
            .zip(decisions[selected_index].losses.iter())
            .zip(actual.likelihood_per_mille.iter())
            .map(|((weight, loss), row)| {
                *weight as f64 * f64::from(*loss) * f64::from(row[outcome_index])
            })
            .sum::<f64>();

        expected_loss += realized_joint_loss / (total_weight * PROBABILITY_SCALE as f64);
    }

    Some(expected_loss)
}

fn total_loss_with_acquisition_cost(decision_loss: f64, cost_milliunits: u32) -> f64 {
    decision_loss + f64::from(cost_milliunits) * COST_PENALTY_PER_MILLIUNIT
}

#[test]
fn model_misspecification_can_make_a_nominally_best_sensor_worse_than_baseline() {
    let candidates = measurements();
    let decisions = decisions();
    let inventory = candidates[1];
    let design = candidates[2];
    let true_inventory_likelihood = [
        [900, 100, 0],
        [100, 900, 0],
        [100, 900, 0],
    ];

    // The selector sees the assumed model and prefers the inventory sensor.
    assert_eq!(
        choose_by_evsi_per_cost(&[inventory, design], PRIOR_WEIGHTS, &decisions)
            .map(|item| item.id),
        Some("noisy-inventory-family")
    );

    let selected_actual_risk = expected_loss_under_assumed_policy(
        inventory,
        true_inventory_likelihood,
        PRIOR_WEIGHTS,
        &decisions,
    )
    .expect("valid assumed and true likelihoods");
    assert!((selected_actual_risk - 5.68).abs() <= EPSILON);

    let selected_total =
        total_loss_with_acquisition_cost(selected_actual_risk, inventory.cost_milliunits);
    let baseline_total = baseline_expected_loss(PRIOR_WEIGHTS, &decisions);
    let true_inventory = Measurement {
        likelihood_per_mille: true_inventory_likelihood,
        ..inventory
    };
    let oracle_inventory_total = total_loss_with_acquisition_cost(
        expected_loss_after_measurement(true_inventory, PRIOR_WEIGHTS, &decisions),
        inventory.cost_milliunits,
    );
    let oracle_design_total = total_loss_with_acquisition_cost(
        expected_loss_after_measurement(design, PRIOR_WEIGHTS, &decisions),
        design.cost_milliunits,
    );
    let oracle_total = baseline_total.min(oracle_inventory_total).min(oracle_design_total);
    let decision_regret = selected_total - oracle_total;
    let model_regret = selected_actual_risk
        - expected_loss_after_measurement(true_inventory, PRIOR_WEIGHTS, &decisions);

    assert!((selected_total - 5.78).abs() <= EPSILON);
    assert!((oracle_design_total - 3.885).abs() <= EPSILON);
    assert!((oracle_total - 3.885).abs() <= EPSILON);
    assert!((model_regret - 1.68).abs() <= EPSILON);
    assert!((decision_regret - 1.895).abs() <= EPSILON);
}

#[test]
fn decision_regret_shrinks_as_true_reliability_approaches_the_assumed_model() {
    let candidates = measurements();
    let decisions = decisions();
    let inventory = candidates[1];
    let design = candidates[2];
    let sweep = [
        (900, 1.895),
        (920, 1.079),
        (930, 0.671),
        (940, 0.263),
        (941, 0.2222),
        (942, 0.1814),
        (945, 0.059),
        (946, 0.0182),
        (947, 0.0),
        (950, 0.0),
    ];
    let design_total = total_loss_with_acquisition_cost(
        expected_loss_after_measurement(design, PRIOR_WEIGHTS, &decisions),
        design.cost_milliunits,
    );
    let baseline_total = baseline_expected_loss(PRIOR_WEIGHTS, &decisions);
    let mut previous_regret = f64::INFINITY;

    for (reliability, expected_regret) in sweep {
        let error = (PROBABILITY_SCALE - u64::from(reliability)) as u16;
        let truth = [
            [reliability, error, 0],
            [error, reliability, 0],
            [error, reliability, 0],
        ];
        let selected_risk = expected_loss_under_assumed_policy(
            inventory,
            truth,
            PRIOR_WEIGHTS,
            &decisions,
        )
        .expect("valid likelihoods");
        let selected_total =
            total_loss_with_acquisition_cost(selected_risk, inventory.cost_milliunits);
        let true_inventory = Measurement {
            likelihood_per_mille: truth,
            ..inventory
        };
        let true_inventory_total = total_loss_with_acquisition_cost(
            expected_loss_after_measurement(true_inventory, PRIOR_WEIGHTS, &decisions),
            inventory.cost_milliunits,
        );
        let oracle_total = baseline_total.min(true_inventory_total).min(design_total);
        let regret = selected_total - oracle_total;

        assert!(
            (regret - expected_regret).abs() <= EPSILON,
            "unexpected decision regret at true reliability {reliability}: \
             expected {expected_regret}, got {regret}"
        );
        assert!(
            regret <= previous_regret + EPSILON,
            "decision regret increased as true reliability approached the assumed reliability"
        );
        previous_regret = regret;
    }
}

#[test]
fn misspecified_policy_rejects_malformed_true_likelihoods() {
    let inventory = measurements()[1];
    let mut malformed_truth = inventory.likelihood_per_mille;
    malformed_truth[0][0] -= 1;

    assert_eq!(
        expected_loss_under_assumed_policy(inventory, malformed_truth, PRIOR_WEIGHTS, &decisions()),
        None
    );
}
