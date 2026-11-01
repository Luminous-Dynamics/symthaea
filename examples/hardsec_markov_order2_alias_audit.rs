use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const FIXTURE_BYTES: &[u8] =
    include_bytes!("../tests/fixtures/hardsec/markov-order2-alias-v1/sequences.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fraction {
    numerator: u64,
    denominator: u64,
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

impl Fraction {
    fn new(numerator: u64, denominator: u64) -> Self {
        assert!(denominator > 0);
        let divisor = gcd(numerator, denominator);
        Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        }
    }

    fn text(self) -> String {
        format!("{}/{}", self.numerator, self.denominator)
    }

    fn json(self) -> Value {
        json!({
            "numerator": self.numerator,
            "denominator": self.denominator,
            "exact": self.text()
        })
    }

    fn absolute_difference(self, other: Self) -> Self {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .expect("small fixture fraction cross-product overflow");
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .expect("small fixture fraction cross-product overflow");
        let numerator = left.abs_diff(right);
        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .expect("small fixture fraction denominator overflow");
        Self::new(numerator, denominator)
    }
}

fn parse_sequence(fixture: &Value, field: &str) -> Vec<usize> {
    let values = fixture[field]
        .as_array()
        .unwrap_or_else(|| panic!("fixture field {field} must be an array"));
    let sequence: Vec<usize> = values
        .iter()
        .map(|value| {
            let state = value
                .as_u64()
                .unwrap_or_else(|| panic!("fixture field {field} has non-integer state"));
            usize::try_from(state).expect("state index fits usize")
        })
        .collect();
    assert!(
        sequence.len() >= 3,
        "sequence must have at least three states"
    );
    assert!(
        sequence.iter().all(|state| *state < 2),
        "this known-answer fixture has binary states only"
    );
    sequence
}

fn transition_counts(sequence: &[usize]) -> [[u64; 2]; 2] {
    let mut counts = [[0_u64; 2]; 2];
    for pair in sequence.windows(2) {
        counts[pair[0]][pair[1]] += 1;
    }
    counts
}

// Counts [previous_previous][current][next] without cyclic wraparound.
fn lag2_counts(sequence: &[usize]) -> [[[u64; 2]; 2]; 2] {
    let mut counts = [[[0_u64; 2]; 2]; 2];
    for triple in sequence.windows(3) {
        counts[triple[0]][triple[1]][triple[2]] += 1;
    }
    counts
}

fn conditional_next_one_rate(
    counts: &[[[u64; 2]; 2]; 2],
    previous_previous: usize,
    current: usize,
) -> Option<Fraction> {
    let next_zero = counts[previous_previous][current][0];
    let next_one = counts[previous_previous][current][1];
    let total = next_zero + next_one;
    (total > 0).then(|| Fraction::new(next_one, total))
}

fn row_json(rates: [Option<Fraction>; 2]) -> Value {
    json!({
        "previous_previous_0": rates[0].map(Fraction::json),
        "previous_previous_1": rates[1].map(Fraction::json)
    })
}

fn fixture_sha256() -> String {
    let digest = Sha256::digest(FIXTURE_BYTES);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn report() -> Value {
    let fixture: Value =
        serde_json::from_slice(FIXTURE_BYTES).expect("parse embedded alias fixture JSON");
    assert_eq!(
        fixture["schema_version"],
        "hardsec-markov-order2-alias-fixture-v1"
    );

    let control = parse_sequence(&fixture, "first_order_control");
    let alias = parse_sequence(&fixture, "order2_alias");
    let control_edges = transition_counts(&control);
    let alias_edges = transition_counts(&alias);
    let control_lag2 = lag2_counts(&control);
    let alias_lag2 = lag2_counts(&alias);

    let control_rates = [
        conditional_next_one_rate(&control_lag2, 0, 0),
        conditional_next_one_rate(&control_lag2, 1, 0),
    ];
    let alias_rates = [
        conditional_next_one_rate(&alias_lag2, 0, 0),
        conditional_next_one_rate(&alias_lag2, 1, 0),
    ];

    let control_gap = control_rates[0]
        .expect("control context 0,0 has support")
        .absolute_difference(control_rates[1].expect("control context 1,0 has support"));
    let alias_gap = alias_rates[0]
        .expect("alias context 0,0 has support")
        .absolute_difference(alias_rates[1].expect("alias context 1,0 has support"));
    let one_step_equal = control_edges == alias_edges;
    let order2_difference = alias_gap.numerator > 0;

    json!({
        "schema_version": "hardsec-markov-order2-alias-audit-v1",
        "result": "exact_order2_alias_counterexample_verified",
        "claim_scope": "deterministic_finite_sequence_counterexample_only",
        "fixture": {
            "fixture_id": fixture["fixture_id"],
            "sha256": fixture_sha256(),
            "first_order_control": control,
            "order2_alias": alias
        },
        "first_order": {
            "control_transition_counts": control_edges,
            "order2_alias_transition_counts": alias_edges,
            "transition_counts_equal": one_step_equal,
            "control_row_0_next_one_rate": Fraction::new(
                control_edges[0][1],
                control_edges[0][0] + control_edges[0][1]
            ).json(),
            "alias_row_0_next_one_rate": Fraction::new(
                alias_edges[0][1],
                alias_edges[0][0] + alias_edges[0][1]
            ).json()
        },
        "lag_two_conditional": {
            "conditioned_current_state": 0,
            "control_next_one_rate_by_previous_previous_state": row_json(control_rates),
            "order2_alias_next_one_rate_by_previous_previous_state": row_json(alias_rates),
            "control_absolute_rate_gap": control_gap.json(),
            "order2_alias_absolute_rate_gap": alias_gap.json()
        },
        "eligibility": {
            "first_order_abstraction_falsified_for_alias_fixture": one_step_equal && order2_difference,
            "model_validity_proved": false,
            "statistical_p_value": null,
            "statistical_rejection_claim": false,
            "finite_test_non_rejection_proves_markov": false,
            "recommended_gate_outcome": "model_rejected_for_declared_fixture"
        },
        "limitations": [
            "The two sequences are a deterministic counterexample, not random samples from a calibrated process.",
            "The fixture demonstrates non-identifiability from one-step transition counts alone; it is not a general-purpose hypothesis test.",
            "No p-value, Type-I error rate, power guarantee, hardware measurement, or attack-detection result is produced.",
            "Passing other finite diagnostics cannot prove a first-order Markov assumption."
        ]
    })
}

fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&report()).expect("serialize exact alias receipt")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_has_identical_one_step_transition_counts() {
        let fixture: Value = serde_json::from_slice(FIXTURE_BYTES).unwrap();
        let control = parse_sequence(&fixture, "first_order_control");
        let alias = parse_sequence(&fixture, "order2_alias");
        assert_eq!(transition_counts(&control), [[3, 2], [2, 0]]);
        assert_eq!(transition_counts(&alias), [[3, 2], [2, 0]]);
    }

    #[test]
    fn control_has_equal_lag_two_rates_for_the_supported_current_zero_contexts() {
        let fixture: Value = serde_json::from_slice(FIXTURE_BYTES).unwrap();
        let control = parse_sequence(&fixture, "first_order_control");
        let counts = lag2_counts(&control);
        assert_eq!(
            conditional_next_one_rate(&counts, 0, 0),
            Some(Fraction::new(1, 2))
        );
        assert_eq!(
            conditional_next_one_rate(&counts, 1, 0),
            Some(Fraction::new(1, 2))
        );
    }

    #[test]
    fn order_two_alias_has_opposite_lag_two_conditionals_with_same_one_step_counts() {
        let fixture: Value = serde_json::from_slice(FIXTURE_BYTES).unwrap();
        let alias = parse_sequence(&fixture, "order2_alias");
        let counts = lag2_counts(&alias);
        assert_eq!(
            conditional_next_one_rate(&counts, 0, 0),
            Some(Fraction::new(1, 1))
        );
        assert_eq!(
            conditional_next_one_rate(&counts, 1, 0),
            Some(Fraction::new(0, 1))
        );
        assert_eq!(transition_counts(&alias), [[3, 2], [2, 0]]);
    }

    #[test]
    fn exact_report_rejects_only_the_declared_counterexample_not_markov_truth_in_general() {
        let value = report();
        assert_eq!(
            value["result"],
            "exact_order2_alias_counterexample_verified"
        );
        assert_eq!(
            value["eligibility"]["first_order_abstraction_falsified_for_alias_fixture"],
            true
        );
        assert_eq!(value["eligibility"]["model_validity_proved"], false);
        assert_eq!(value["eligibility"]["statistical_p_value"], Value::Null);
        assert_eq!(value["eligibility"]["statistical_rejection_claim"], false);
        assert_eq!(
            value["eligibility"]["finite_test_non_rejection_proves_markov"],
            false
        );
    }
}
