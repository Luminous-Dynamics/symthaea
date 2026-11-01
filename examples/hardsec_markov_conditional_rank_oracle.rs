use serde_json::{Value, json};

const TRANSITION_NUMERATORS: [[u128; 2]; 2] = [[9, 1], [2, 8]];
const TRANSITION_DENOMINATOR: u128 = 10;
const BET_NUMERATOR: u128 = 1;
const BET_DENOMINATOR: u128 = 2;
const HORIZON: u32 = 12;
const ALPHA_NUMERATOR: u128 = 1;
const ALPHA_DENOMINATOR: u128 = 10;
#[cfg(test)]
const EXPECTED_CROSSING_NUMERATORS: [u128; 2] = [52_840_309_168, 55_866_239_952];

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactEpisode {
    initial_state: usize,
    path_count: u64,
    total_probability_numerator: u128,
    total_probability_denominator: u128,
    crossing_probability_numerator: u128,
    crossing_probability_denominator: u128,
    reduced_crossing_numerator: u128,
    reduced_crossing_denominator: u128,
    pfa_at_or_below_alpha: bool,
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn pow_u128(base: u128, exponent: u32) -> u128 {
    let mut result = 1_u128;
    for _ in 0..exponent {
        result *= base;
    }
    result
}

fn exact_episode(initial_state: usize) -> ExactEpisode {
    assert!(initial_state < 2);

    let path_count = 1_u64 << HORIZON;
    let total_probability_denominator = pow_u128(TRANSITION_DENOMINATOR, HORIZON);
    let mut total_probability_numerator = 0_u128;
    let mut crossing_probability_numerator = 0_u128;

    for encoded_path in 0..path_count {
        let mut state = initial_state;
        let mut path_probability_numerator = 1_u128;
        let mut wealth_numerator = 1_u128;
        let mut wealth_denominator = 1_u128;
        let mut crossed = false;

        for time_index in 0..HORIZON {
            let next_state = ((encoded_path >> time_index) & 1) as usize;
            let transition_numerator = TRANSITION_NUMERATORS[state][next_state];

            path_probability_numerator *= transition_numerator;
            wealth_denominator *= transition_numerator;

            // q(next_state) / P(next_state | state)
            // = (1/2) / (transition_numerator/10)
            // = 5 / transition_numerator.
            wealth_numerator *= 5;

            let crossed_now =
                wealth_numerator * ALPHA_NUMERATOR >= wealth_denominator * ALPHA_DENOMINATOR;
            crossed |= crossed_now;
            state = next_state;
        }

        total_probability_numerator += path_probability_numerator;
        if crossed {
            crossing_probability_numerator += path_probability_numerator;
        }
    }

    let divisor = gcd(
        crossing_probability_numerator,
        total_probability_denominator,
    );
    let pfa_at_or_below_alpha = crossing_probability_numerator * ALPHA_DENOMINATOR
        <= total_probability_denominator * ALPHA_NUMERATOR;

    ExactEpisode {
        initial_state,
        path_count,
        total_probability_numerator,
        total_probability_denominator,
        crossing_probability_numerator,
        crossing_probability_denominator: total_probability_denominator,
        reduced_crossing_numerator: crossing_probability_numerator / divisor,
        reduced_crossing_denominator: total_probability_denominator / divisor,
        pfa_at_or_below_alpha,
    }
}

fn conditional_expected_factor_is_one(current_state: usize) -> bool {
    let mut numerator_sum = 0_u128;
    for next_state in 0..2 {
        let transition_numerator = TRANSITION_NUMERATORS[current_state][next_state];
        if transition_numerator == 0 {
            return false;
        }
        // (a/10) * (1/2) / (a/10) = 1/2 = 5/10.
        numerator_sum += BET_NUMERATOR * (TRANSITION_DENOMINATOR / BET_DENOMINATOR);
    }
    // Both next-state branches contribute 5/10, so their sum is exactly 10/10.
    numerator_sum == TRANSITION_DENOMINATOR
}

fn report() -> Value {
    let episodes = [exact_episode(0), exact_episode(1)];
    json!({
        "schema_version": "hardsec-markov-conditional-rank-oracle-v1",
        "result": "exact_enumeration_passed",
        "claim_scope": "known_fully_observed_finite_state_markov_null_only",
        "transition_model": {
            "transition_numerators": TRANSITION_NUMERATORS,
            "common_denominator": TRANSITION_DENOMINATOR,
            "rows": [[0.9, 0.1], [0.2, 0.8]]
        },
        "rank_alphabet": [0, 1],
        "betting_distribution": {
            "numerators": [BET_NUMERATOR, BET_NUMERATOR],
            "common_denominator": BET_DENOMINATOR
        },
        "target_alpha": {
            "numerator": ALPHA_NUMERATOR,
            "denominator": ALPHA_DENOMINATOR
        },
        "horizon": HORIZON,
        "conditional_expectation_identity": {
            "initial_state_0_equals_one": conditional_expected_factor_is_one(0),
            "initial_state_1_equals_one": conditional_expected_factor_is_one(1)
        },
        "episodes": episodes.iter().map(|episode| json!({
            "initial_state": episode.initial_state,
            "enumerated_path_count": episode.path_count,
            "path_probability_mass_numerator": episode.total_probability_numerator,
            "path_probability_mass_denominator": episode.total_probability_denominator,
            "path_probability_mass_is_one":
                episode.total_probability_numerator == episode.total_probability_denominator,
            "crossing_probability_numerator": episode.crossing_probability_numerator,
            "crossing_probability_denominator": episode.crossing_probability_denominator,
            "crossing_probability_reduced_numerator": episode.reduced_crossing_numerator,
            "crossing_probability_reduced_denominator": episode.reduced_crossing_denominator,
            "crossing_probability": episode.crossing_probability_numerator as f64
                / episode.crossing_probability_denominator as f64,
            "pfa_at_or_below_alpha": episode.pfa_at_or_below_alpha
        })).collect::<Vec<_>>(),
        "guarantee": {
            "known_transition_matrix_assumed": true,
            "current_state_observed": true,
            "bet_is_predictable": true,
            "tie_randomization_required": false,
            "time_uniform_pfa_claim_for_this_markov_model": true,
            "arbitrary_serial_pmu_pfa_claim": false,
            "unknown_transition_matrix_supported": false,
            "hidden_state_supported": false,
            "physical_hardware_evidence": false,
            "effective_sample_size_estimate": null
        },
        "limitations": [
            "The transition matrix is known exactly and is part of the null model.",
            "The rank alphabet is the two-state next-state partition in this exact oracle.",
            "The finite horizon enumeration is a code/proof cross-check, not a detector benchmark.",
            "This result does not establish a valid null model for real PMU telemetry."
        ]
    })
}

fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&report()).expect("JSON report serialization")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_betting_factor_has_exact_unit_expectation_in_each_state() {
        assert!(conditional_expected_factor_is_one(0));
        assert!(conditional_expected_factor_is_one(1));
    }

    #[test]
    fn exact_path_probabilities_sum_to_one_for_both_initial_states() {
        for initial_state in 0..2 {
            let episode = exact_episode(initial_state);
            assert_eq!(episode.path_count, 4_096);
            assert_eq!(
                episode.total_probability_numerator,
                episode.total_probability_denominator
            );
            assert_eq!(episode.total_probability_denominator, 1_000_000_000_000);
        }
    }

    #[test]
    fn exact_anytime_false_alarm_mass_matches_rational_known_answers() {
        let state_zero = exact_episode(0);
        let state_one = exact_episode(1);

        assert_eq!(
            state_zero.crossing_probability_numerator,
            EXPECTED_CROSSING_NUMERATORS[0]
        );
        assert_eq!(
            state_one.crossing_probability_numerator,
            EXPECTED_CROSSING_NUMERATORS[1]
        );
        assert_eq!(
            state_zero.crossing_probability_denominator,
            1_000_000_000_000
        );
        assert_eq!(
            state_one.crossing_probability_denominator,
            1_000_000_000_000
        );
        assert!(state_zero.pfa_at_or_below_alpha);
        assert!(state_one.pfa_at_or_below_alpha);
        assert_eq!(
            (
                state_zero.reduced_crossing_numerator,
                state_zero.reduced_crossing_denominator
            ),
            (3_302_519_323, 62_500_000_000)
        );
        assert_eq!(
            (
                state_one.reduced_crossing_numerator,
                state_one.reduced_crossing_denominator
            ),
            (3_491_639_997, 62_500_000_000)
        );
    }

    #[test]
    fn receipt_refuses_to_claim_general_serial_pmu_validity() {
        let value = report();
        assert_eq!(
            value["claim_scope"],
            "known_fully_observed_finite_state_markov_null_only"
        );
        assert_eq!(
            value["guarantee"]["time_uniform_pfa_claim_for_this_markov_model"],
            true
        );
        assert_eq!(value["guarantee"]["arbitrary_serial_pmu_pfa_claim"], false);
        assert_eq!(value["guarantee"]["unknown_transition_matrix_supported"], false);
        assert_eq!(value["guarantee"]["hidden_state_supported"], false);
        assert_eq!(
            value["guarantee"]["effective_sample_size_estimate"],
            Value::Null
        );
    }
}
