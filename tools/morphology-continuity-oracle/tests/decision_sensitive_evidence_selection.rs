//! Decision-sensitive active evidence selection under asymmetric consequences.
//!
//! The example deliberately has only two downstream choices: restore a chair, or
//! hold for safety/manual review. A design record can distinguish table from stool,
//! but that distinction does not change this decision set. Thus more classification
//! information need not be more decision value. All losses and provenance flags are
//! synthetic fixture inputs, not an operational safety policy or verified provenance.

use std::collections::BTreeMap;

const PRIOR_WEIGHTS: [u64; 3] = [6, 3, 1];
const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Measurement {
    id: &'static str,
    /// Deterministic observation label for [chair, table, stool].
    outcomes: [&'static str; 3],
    cost_milliunits: u32,
    /// Fixture input only; it does not perform cryptographic verification.
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

fn measurements() -> [Measurement; 5] {
    [
        Measurement {
            id: "repeat-geometry",
            outcomes: ["table-shaped", "table-shaped", "table-shaped"],
            cost_milliunits: 1_000,
            provenance_validated: true,
        },
        Measurement {
            id: "inventory-family",
            outcomes: ["chair", "other", "other"],
            cost_milliunits: 1_000,
            provenance_validated: true,
        },
        Measurement {
            id: "design-record",
            outcomes: ["chair", "table", "stool"],
            cost_milliunits: 1_250,
            provenance_validated: true,
        },
        Measurement {
            id: "unverified-record",
            outcomes: ["chair", "table", "stool"],
            cost_milliunits: 1,
            provenance_validated: false,
        },
        Measurement {
            id: "zero-cost-record",
            outcomes: ["chair", "table", "stool"],
            cost_milliunits: 0,
            provenance_validated: true,
        },
    ]
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

/// Expected Bayes risk after observing this measurement, allowing the best declared
/// downstream decision per observation group. The outcome is not treated as trusted
/// if its fixture provenance flag is false.
fn expected_loss_after_measurement(
    measurement: Measurement,
    weights: [u64; 3],
    decisions: &[Decision],
) -> f64 {
    if !measurement.provenance_validated {
        return baseline_expected_loss(weights, decisions);
    }

    let total_weight: f64 = weights.iter().map(|weight| *weight as f64).sum();
    let mut labels = measurement.outcomes.to_vec();
    labels.sort_unstable();
    labels.dedup();

    let weighted_group_loss: f64 = labels
        .iter()
        .map(|label| {
            decisions
                .iter()
                .map(|decision| {
                    (0..weights.len())
                        .filter(|index| measurement.outcomes[*index] == *label)
                        .map(|index| weights[index] as f64 * f64::from(decision.losses[index]))
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .sum();

    weighted_group_loss / total_weight
}

/// Gross expected value of sample information (EVSI), before acquisition cost.
fn evsi(measurement: Measurement, weights: [u64; 3], decisions: &[Decision]) -> f64 {
    if !measurement.provenance_validated {
        return 0.0;
    }

    (baseline_expected_loss(weights, decisions)
        - expected_loss_after_measurement(measurement, weights, decisions))
    .max(0.0)
}

fn expected_brier_reduction(measurement: Measurement, weights: [u64; 3]) -> f64 {
    if !measurement.provenance_validated {
        return 0.0;
    }

    let total_weight: f64 = weights.iter().map(|weight| *weight as f64).sum();
    let prior_square_sum: u128 = weights
        .iter()
        .map(|weight| u128::from(*weight) * u128::from(*weight))
        .sum();
    let prior_concentration = prior_square_sum as f64 / total_weight.powi(2);

    let mut groups: BTreeMap<&str, (u64, u128)> = BTreeMap::new();
    for (weight, outcome) in weights.iter().copied().zip(measurement.outcomes) {
        let entry = groups.entry(outcome).or_default();
        entry.0 += weight;
        entry.1 += u128::from(weight) * u128::from(weight);
    }

    let posterior_concentration: f64 = groups
        .values()
        .map(|(group_weight, square_sum)| {
            *square_sum as f64 / (total_weight * *group_weight as f64)
        })
        .sum();

    (posterior_concentration - prior_concentration).max(0.0)
}

fn choose_by_value_per_cost(
    candidates: &[Measurement],
    weights: [u64; 3],
    decisions: &[Decision],
) -> Option<Measurement> {
    let mut best: Option<(Measurement, f64)> = None;

    for candidate in candidates.iter().copied() {
        if candidate.cost_milliunits == 0 {
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

fn choose_by_brier_per_cost(candidates: &[Measurement], weights: [u64; 3]) -> Option<Measurement> {
    candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.cost_milliunits > 0)
        .filter_map(|candidate| {
            let value = expected_brier_reduction(candidate, weights);
            (value > EPSILON).then_some((candidate, value / f64::from(candidate.cost_milliunits)))
        })
        .max_by(|(left, left_ratio), (right, right_ratio)| {
            left_ratio
                .total_cmp(right_ratio)
                .then_with(|| right.id.cmp(left.id))
        })
        .map(|(candidate, _)| candidate)
}

#[test]
fn decision_value_can_reverse_the_brier_per_cost_ranking() {
    let candidates = measurements();
    let decision_set = decisions();
    let inventory = candidates
        .iter()
        .copied()
        .find(|candidate| candidate.id == "inventory-family")
        .expect("inventory measurement exists");
    let design = candidates
        .iter()
        .copied()
        .find(|candidate| candidate.id == "design-record")
        .expect("design record exists");

    assert!((baseline_expected_loss(PRIOR_WEIGHTS, &decision_set) - 4.0).abs() <= EPSILON);
    assert!(
        (expected_loss_after_measurement(inventory, PRIOR_WEIGHTS, &decision_set) - 1.6).abs()
            <= EPSILON
    );
    assert!(
        (expected_loss_after_measurement(design, PRIOR_WEIGHTS, &decision_set) - 1.6).abs()
            <= EPSILON
    );
    assert!((evsi(inventory, PRIOR_WEIGHTS, &decision_set) - 2.4).abs() <= EPSILON);
    assert!((evsi(design, PRIOR_WEIGHTS, &decision_set) - 2.4).abs() <= EPSILON);

    assert!((expected_brier_reduction(inventory, PRIOR_WEIGHTS) - 0.39).abs() <= EPSILON);
    assert!((expected_brier_reduction(design, PRIOR_WEIGHTS) - 0.54).abs() <= EPSILON);

    // Brier reduction per cost chooses the more informative design record.
    assert_eq!(
        choose_by_brier_per_cost(&candidates, PRIOR_WEIGHTS).map(|candidate| candidate.id),
        Some("design-record")
    );

    // But distinguishing table from stool does not change any available decision.
    // The cheaper inventory action has equal EVSI and strictly higher EVSI/cost.
    assert_eq!(
        choose_by_value_per_cost(&candidates, PRIOR_WEIGHTS, &decision_set)
            .map(|candidate| candidate.id),
        Some("inventory-family")
    );
}

#[test]
fn repeated_or_unverified_measurements_have_no_qualified_decision_value() {
    let candidates = measurements();
    let decision_set = decisions();
    let repeat = candidates[0];
    let unverified = candidates[3];
    let zero_cost = candidates[4];

    assert!((evsi(repeat, PRIOR_WEIGHTS, &decision_set) - 0.0).abs() <= EPSILON);
    assert!((evsi(unverified, PRIOR_WEIGHTS, &decision_set) - 0.0).abs() <= EPSILON);
    assert_eq!(
        choose_by_value_per_cost(
            &[repeat, unverified, zero_cost],
            PRIOR_WEIGHTS,
            &decision_set
        ),
        None
    );
}

#[test]
fn permuting_hypotheses_with_losses_and_priors_preserves_decision_value() {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let original = measurements();
    let original_decisions = decisions();
    let expected_choice = choose_by_value_per_cost(&original, PRIOR_WEIGHTS, &original_decisions)
        .map(|candidate| candidate.id);

    for permutation in PERMUTATIONS {
        let weights = permutation.map(|index| PRIOR_WEIGHTS[index]);
        let candidates: Vec<_> = original
            .iter()
            .copied()
            .map(|mut candidate| {
                candidate.outcomes = permutation.map(|index| candidate.outcomes[index]);
                candidate
            })
            .collect();
        let permuted_decisions: Vec<_> = original_decisions
            .iter()
            .copied()
            .map(|mut decision| {
                decision.losses = permutation.map(|index| decision.losses[index]);
                decision
            })
            .collect();

        for (before, after) in original.iter().zip(&candidates) {
            let expected = evsi(*before, PRIOR_WEIGHTS, &original_decisions);
            let observed = evsi(*after, weights, &permuted_decisions);
            assert!(
                (expected - observed).abs() <= EPSILON,
                "{} changed EVSI under a joint hypothesis permutation",
                before.id
            );
        }
        assert_eq!(
            choose_by_value_per_cost(&candidates, weights, &permuted_decisions)
                .map(|candidate| candidate.id),
            expected_choice
        );
    }
}
