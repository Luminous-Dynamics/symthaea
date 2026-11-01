use serde_json::{Value, json};

const HORIZON: usize = 6;
const PROBABILITY_DENOMINATOR: u128 = 4;
const ALPHA: Rational = Rational {
    numerator: 1,
    denominator: 2,
};
const THRESHOLD: Rational = Rational {
    numerator: 2,
    denominator: 1,
};

// P = [[3/4, 1/4], [1/2, 1/2]].
// Each row is indexed by the current state; each column is the next state.
const TRANSITION_NUMERATORS: [[u128; 2]; 2] = [[3, 1], [2, 2]];

// Synthetic fixed-count calibration tally: 1,000 visits from each source state.
// These are deterministic known-answer fixture counts, not hardware observations.
const CALIBRATION_VISITS_PER_ROW: u128 = 1_000;
const CALIBRATION_SUCCESS_COUNTS: [[u128; 2]; 2] = [[750, 250], [500, 500]];
const CALIBRATION_DELTA_NUMERATOR: u128 = 1;
const CALIBRATION_DELTA_DENOMINATOR: u128 = 40;
const CALIBRATION_COORDINATE_COUNT: u128 = 4;
const FIXED_POINT_SCALE: u128 = 1_000_000_000_000_000_000;
const ENVELOPE_QUANTIZATION_DENOMINATOR: u128 = 10;
const LOG_SERIES_TERMS: u32 = 36;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rational {
    numerator: u128,
    denominator: u128,
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

impl Rational {
    fn new(numerator: u128, denominator: u128) -> Self {
        assert!(denominator > 0, "rational denominator must be positive");
        let divisor = gcd(numerator, denominator);
        Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        }
    }

    fn zero() -> Self {
        Self::new(0, 1)
    }

    fn one() -> Self {
        Self::new(1, 1)
    }

    fn add(self, other: Self) -> Self {
        let common = gcd(self.denominator, other.denominator);
        let left_scale = other.denominator / common;
        let right_scale = self.denominator / common;
        Self::new(
            self.numerator
                .checked_mul(left_scale)
                .and_then(|value| {
                    other
                        .numerator
                        .checked_mul(right_scale)
                        .and_then(|other_value| value.checked_add(other_value))
                })
                .expect("bounded exact fixture addition overflow"),
            self.denominator
                .checked_mul(left_scale)
                .expect("bounded exact fixture denominator overflow"),
        )
    }

    fn mul(self, other: Self) -> Self {
        let left_common = gcd(self.numerator, other.denominator);
        let right_common = gcd(other.numerator, self.denominator);
        Self::new(
            (self.numerator / left_common)
                .checked_mul(other.numerator / right_common)
                .expect("bounded exact fixture multiplication overflow"),
            (self.denominator / right_common)
                .checked_mul(other.denominator / left_common)
                .expect("bounded exact fixture denominator overflow"),
        )
    }

    fn div(self, other: Self) -> Self {
        assert!(other.numerator > 0, "division by zero rational");
        self.mul(Self::new(other.denominator, other.numerator))
    }

    fn ge(self, other: Self) -> bool {
        self.numerator
            .checked_mul(other.denominator)
            .expect("bounded exact fixture comparison overflow")
            >= other
                .numerator
                .checked_mul(self.denominator)
                .expect("bounded exact fixture comparison overflow")
    }

    fn le(self, other: Self) -> bool {
        other.ge(self)
    }

    fn as_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    fn text(self) -> String {
        format!("{}/{}", self.numerator, self.denominator)
    }
}

#[derive(Clone, Debug)]
struct RestartComponent {
    start_index: usize,
    portfolio_wealth: Vec<Rational>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EpisodeResult {
    initial_state: usize,
    enumerated_path_count: u64,
    total_probability_numerator: u128,
    probability_denominator: u128,
    crossing_probability_numerator: u128,
}

fn portfolios() -> Vec<[Rational; 3]> {
    let denominator = 2;
    let mut output = Vec::new();
    for cash in 0..=denominator {
        for state_zero in 0..=(denominator - cash) {
            let state_one = denominator - cash - state_zero;
            output.push([
                Rational::new(cash, denominator),
                Rational::new(state_zero, denominator),
                Rational::new(state_one, denominator),
            ]);
        }
    }
    output
}

fn calibration_ceil_div(numerator: u128, denominator: u128) -> u128 {
    assert!(denominator > 0, "calibration division denominator must be positive");
    numerator / denominator + u128::from(numerator % denominator != 0)
}

fn calibration_fixed_mul_ceil(left: u128, right: u128) -> u128 {
    calibration_ceil_div(
        left.checked_mul(right)
            .expect("fixed-point calibration multiplication overflow"),
        FIXED_POINT_SCALE,
    )
}

fn calibration_log_unit_interval_upper(z_numerator: u128, z_denominator: u128) -> u128 {
    assert!(z_denominator > 0, "log-series denominator must be positive");
    assert!(
        z_numerator.checked_mul(3).is_some_and(|value| value <= z_denominator),
        "log-series argument must be in [0, 1/3]"
    );

    // ln((1+z)/(1-z)) = 2 * sum(z^(2k+1)/(2k+1)).
    // All fixed-point powers and terms are rounded upward. Since z <= 1/3,
    // the omitted positive tail is bounded by 9/(4*(2K+1)*3^(2K+1)).
    let z_scaled = calibration_ceil_div(
        z_numerator
            .checked_mul(FIXED_POINT_SCALE)
            .expect("fixed-point log argument overflow"),
        z_denominator,
    );
    let z_squared_scaled = calibration_fixed_mul_ceil(z_scaled, z_scaled);
    let mut power_scaled = z_scaled;
    let mut series_sum_scaled = 0_u128;

    for index in 0..LOG_SERIES_TERMS {
        let odd_denominator = (2 * index + 1) as u128;
        series_sum_scaled = series_sum_scaled
            .checked_add(calibration_ceil_div(power_scaled, odd_denominator))
            .expect("fixed-point log series sum overflow");
        power_scaled = calibration_fixed_mul_ceil(power_scaled, z_squared_scaled);
    }

    let first_omitted_odd = 2 * LOG_SERIES_TERMS as u128 + 1;
    let tail_denominator = 4_u128
        .checked_mul(first_omitted_odd)
        .and_then(|value| value.checked_mul(3_u128.pow(first_omitted_odd as u32)))
        .expect("fixed-point log tail denominator overflow");
    let tail_upper_scaled =
        calibration_ceil_div(9 * FIXED_POINT_SCALE, tail_denominator);

    2 * series_sum_scaled + tail_upper_scaled
}

fn calibration_log_ratio_upper(numerator: u128, denominator: u128) -> u128 {
    assert!(denominator > 0 && numerator >= denominator);
    let mut scaled_denominator = denominator;
    let mut powers_of_two = 0_u128;

    while numerator >= 2 * scaled_denominator {
        scaled_denominator = scaled_denominator
            .checked_mul(2)
            .expect("log normalization overflow");
        powers_of_two += 1;
    }

    let log_two_upper = calibration_log_unit_interval_upper(1, 3);
    let y_log_upper = calibration_log_unit_interval_upper(
        numerator - scaled_denominator,
        numerator + scaled_denominator,
    );
    powers_of_two
        .checked_mul(log_two_upper)
        .and_then(|value| value.checked_add(y_log_upper))
        .expect("fixed-point log result overflow")
}

fn calibration_sqrt_ceil(value: u128) -> u128 {
    if value == 0 {
        return 0;
    }
    let mut low = 0_u128;
    let mut high = 1_u128 << 64;
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if middle <= value / middle {
            low = middle;
        } else {
            high = middle;
        }
    }
    if low * low == value { low } else { high }
}

fn fixed_n_hoeffding_upper_scaled(
    successes: u128,
    samples: u128,
    delta_numerator: u128,
    delta_denominator: u128,
) -> u128 {
    assert!(successes <= samples);
    assert!(delta_numerator > 0 && delta_numerator < delta_denominator);
    if samples == 0 {
        return FIXED_POINT_SCALE;
    }
    assert!(samples <= u64::MAX as u128);

    let log_inverse_delta_upper =
        calibration_log_ratio_upper(delta_denominator, delta_numerator);
    let radicand_upper = calibration_ceil_div(
        log_inverse_delta_upper
            .checked_mul(FIXED_POINT_SCALE)
            .expect("Hoeffding radicand overflow"),
        2 * samples,
    );
    let epsilon_upper_scaled = calibration_sqrt_ceil(radicand_upper);
    let empirical_rate_upper_scaled = calibration_ceil_div(
        successes
            .checked_mul(FIXED_POINT_SCALE)
            .expect("empirical rate scaling overflow"),
        samples,
    );
    empirical_rate_upper_scaled
        .saturating_add(epsilon_upper_scaled)
        .min(FIXED_POINT_SCALE)
}

fn upper_probability(current_state: usize, next_state: usize) -> Rational {
    let upper_scaled = fixed_n_hoeffding_upper_scaled(
        CALIBRATION_SUCCESS_COUNTS[current_state][next_state],
        CALIBRATION_VISITS_PER_ROW,
        CALIBRATION_DELTA_NUMERATOR,
        CALIBRATION_DELTA_DENOMINATOR,
    );
    let quantized_numerator = calibration_ceil_div(
        upper_scaled
            .checked_mul(ENVELOPE_QUANTIZATION_DENOMINATOR)
            .expect("envelope quantization overflow"),
        FIXED_POINT_SCALE,
    );
    Rational::new(
        quantized_numerator,
        ENVELOPE_QUANTIZATION_DENOMINATOR,
    )
}

fn transition_probability(current_state: usize, next_state: usize) -> Rational {
    Rational::new(
        TRANSITION_NUMERATORS[current_state][next_state],
        PROBABILITY_DENOMINATOR,
    )
}

// Return from a unit-capital portfolio holding [cash, state-0 asset, state-1 asset].
fn outcome_factor(portfolio: [Rational; 3], current_state: usize, next_state: usize) -> Rational {
    portfolio[0].add(portfolio[next_state + 1].div(upper_probability(current_state, next_state)))
}

// A start at s receives only pi_s of the original unit capital.
// The remaining tail is held as cash; restarting never creates new capital.
fn restart_mixture_for_path(path: &[usize], initial_state: usize) -> Vec<Rational> {
    let grid = portfolios();
    let mut components: Vec<RestartComponent> = Vec::new();
    let mut current_state = initial_state;
    let mut global_wealth_by_time = Vec::with_capacity(path.len());

    for (zero_based_index, &next_state) in path.iter().enumerate() {
        let time = zero_based_index + 1;

        for component in &mut components {
            for (portfolio_index, portfolio) in grid.iter().enumerate() {
                let factor = outcome_factor(*portfolio, current_state, next_state);
                component.portfolio_wealth[portfolio_index] =
                    component.portfolio_wealth[portfolio_index].mul(factor);
            }
        }

        // The new component starts immediately before this observation.
        let new_component_wealth = grid
            .iter()
            .map(|portfolio| outcome_factor(*portfolio, current_state, next_state))
            .collect();
        components.push(RestartComponent {
            start_index: time,
            portfolio_wealth: new_component_wealth,
        });

        // pi_s = 1 / (s(s+1)); sum_{s=1}^t pi_s = t/(t+1).
        // The unstarted residual 1/(t+1) completes the unit initial capital.
        let reserve = Rational::new(1, (time + 1) as u128);
        let mut mixture_wealth = reserve;
        let mut allocated_restart_mass = Rational::zero();

        for component in &components {
            let start = component.start_index as u128;
            let start_weight = Rational::new(1, start * (start + 1));
            allocated_restart_mass = allocated_restart_mass.add(start_weight);
            let mean_portfolio_wealth = component
                .portfolio_wealth
                .iter()
                .copied()
                .fold(Rational::zero(), Rational::add)
                .div(Rational::new(grid.len() as u128, 1));
            mixture_wealth = mixture_wealth.add(start_weight.mul(mean_portfolio_wealth));
        }

        assert_eq!(
            allocated_restart_mass.add(reserve),
            Rational::one(),
            "restart prior plus unstarted reserve must equal one"
        );
        global_wealth_by_time.push(mixture_wealth);
        current_state = next_state;
    }

    global_wealth_by_time
}

fn enumerate_episode(initial_state: usize) -> EpisodeResult {
    assert!(initial_state < 2);
    let path_count = 1_u64 << HORIZON;
    let probability_denominator = PROBABILITY_DENOMINATOR.pow(HORIZON as u32);
    let mut total_probability_numerator = 0_u128;
    let mut crossing_probability_numerator = 0_u128;

    for encoded_path in 0..path_count {
        let mut path = Vec::with_capacity(HORIZON);
        let mut current_state = initial_state;
        let mut path_probability_numerator = 1_u128;

        for time_index in 0..HORIZON {
            let next_state = ((encoded_path >> time_index) & 1) as usize;
            path_probability_numerator *= TRANSITION_NUMERATORS[current_state][next_state];
            path.push(next_state);
            current_state = next_state;
        }

        total_probability_numerator += path_probability_numerator;
        let wealth = restart_mixture_for_path(&path, initial_state);
        if wealth.iter().any(|value| value.ge(THRESHOLD)) {
            crossing_probability_numerator += path_probability_numerator;
        }
    }

    EpisodeResult {
        initial_state,
        enumerated_path_count: path_count,
        total_probability_numerator,
        probability_denominator,
        crossing_probability_numerator,
    }
}

fn conditional_expected_factor(current_state: usize, portfolio: [Rational; 3]) -> Rational {
    portfolio[0]
        .add(
            portfolio[1]
                .mul(transition_probability(current_state, 0))
                .div(upper_probability(current_state, 0)),
        )
        .add(
            portfolio[2]
                .mul(transition_probability(current_state, 1))
                .div(upper_probability(current_state, 1)),
        )
}

fn report() -> Value {
    let episodes = [enumerate_episode(0), enumerate_episode(1)];
    json!({
        "schema_version": "hardsec-restarted-markov-portfolio-oracle-v1",
        "result": "exact_rational_restart_mixture_enumeration_passed",
        "claim_scope": "known_fully_observed_two_state_markov_null_with_fixture_supplied_upper_envelope",
        "transition_model": {
            "numerators": TRANSITION_NUMERATORS,
            "common_denominator": PROBABILITY_DENOMINATOR
        },
        "frozen_upper_envelope": {
            "numerators": [
                [upper_probability(0, 0).numerator, upper_probability(0, 1).numerator],
                [upper_probability(1, 0).numerator, upper_probability(1, 1).numerator]
            ],
            "denominators": [
                [upper_probability(0, 0).denominator, upper_probability(0, 1).denominator],
                [upper_probability(1, 0).denominator, upper_probability(1, 1).denominator]
            ],
            "calibration": {
                "method": "fixed_n_hoeffding_upper_bound_with_integer_outward_rounding",
                "visits_per_source_state": CALIBRATION_VISITS_PER_ROW,
                "success_counts": CALIBRATION_SUCCESS_COUNTS,
                "coordinate_count": CALIBRATION_COORDINATE_COUNT,
                "delta_total": "1/10",
                "delta_per_coordinate": "1/40",
                "fixed_point_scale": FIXED_POINT_SCALE.to_string(),
                "envelope_quantization": "round_up_to_tenths",
                "calibration_tally_is_synthetic": true,
                "physical_calibration_data": false,
                "familywise_coverage_empirically_validated": false,
                "statistical_contract": "familywise coverage at least 0.9 only when each source-state row contributes the preregistered fixed number of independent Markov transitions"
            }
        },
        "portfolio": {
            "grid": "all rational simplex points with denominator 2 over cash and two outcome assets",
            "portfolio_count": portfolios().len(),
            "start_prior": "pi_s=1/(s(s+1))",
            "unstarted_residual": "1/(t+1)",
            "capital_conservation": "sum_{s=1..t} pi_s + residual_t = 1 exactly",
            "threshold": THRESHOLD.text(),
            "alpha": ALPHA.text()
        },
        "enumeration": {
            "horizon": HORIZON,
            "paths_per_initial_state": 1_u64 << HORIZON,
            "probability_denominator": PROBABILITY_DENOMINATOR.pow(HORIZON as u32).to_string(),
            "episodes": episodes.iter().map(|episode| json!({
                "initial_state": episode.initial_state,
                "path_count": episode.enumerated_path_count,
                "path_mass_numerator": episode.total_probability_numerator.to_string(),
                "path_mass_is_one": episode.total_probability_numerator == episode.probability_denominator,
                "anytime_crossing_mass_numerator": episode.crossing_probability_numerator.to_string(),
                "anytime_crossing_probability": episode.crossing_probability_numerator as f64
                    / episode.probability_denominator as f64,
                "anytime_crossing_probability_at_or_below_alpha":
                    episode.crossing_probability_numerator * ALPHA.denominator
                    <= episode.probability_denominator * ALPHA.numerator
            })).collect::<Vec<_>>()
        },
        "guarantee": {
            "conditional_factor_le_one_for_every_fixture_portfolio": portfolios().iter().all(|portfolio|
                (0..2).all(|state| conditional_expected_factor(state, *portfolio).le(Rational::one()))
            ),
            "restart_capital_created": false,
            "calibration_coverage_established": false,
            "runtime_calibration_guarantee": "not_established",
            "arbitrary_serial_pmu_pfa_claim": false,
            "hidden_state_claim": false,
            "physical_hardware_evidence": false,
            "effective_sample_size_estimate": null
        },
        "limitations": [
            "The transition matrix and upper envelope are exact fixture constants.",
            "The finite-horizon enumeration verifies this small rational implementation only.",
            "No independent-calibration confidence envelope is implemented in this oracle.",
            "No claim is made for hidden-state, higher-order, switching, adversarial, or arbitrary PMU processes.",
            "The 0.5 alpha is chosen for a compact exact oracle, not as an operational target."
        ]
    })
}

fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&report()).expect("serialize exact oracle receipt")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portfolio_grid_is_exact_simplex() {
        let grid = portfolios();
        assert_eq!(grid.len(), 6);
        for portfolio in grid {
            assert_eq!(
                portfolio
                    .iter()
                    .copied()
                    .fold(Rational::zero(), Rational::add),
                Rational::one()
            );
        }
    }

    #[test]
    fn fixed_point_log_and_hoeffding_bounds_are_outward_rounded() {
        let log_upper = calibration_log_ratio_upper(40, 1);
        assert!(log_upper >= 3_688_879_454_113_936_000);
        assert!(log_upper <= 3_688_879_454_113_938_000);

        let radicand = calibration_ceil_div(
            log_upper * FIXED_POINT_SCALE,
            2 * CALIBRATION_VISITS_PER_ROW,
        );
        let epsilon = calibration_sqrt_ceil(radicand);
        assert!(epsilon * epsilon >= radicand);
        assert_eq!(epsilon, 42_946_940_834_673_759);
    }

    #[test]
    fn calibration_envelope_matches_the_fixed_count_known_answer() {
        assert_eq!(upper_probability(0, 0), Rational::new(4, 5));
        assert_eq!(upper_probability(0, 1), Rational::new(3, 10));
        assert_eq!(upper_probability(1, 0), Rational::new(3, 5));
        assert_eq!(upper_probability(1, 1), Rational::new(3, 5));
        assert_eq!(
            CALIBRATION_COORDINATE_COUNT
                * CALIBRATION_DELTA_NUMERATOR
                * 10,
            CALIBRATION_DELTA_DENOMINATOR
        );
    }

    #[test]
    fn fixed_count_calibration_formula_covers_each_known_fixture_transition() {
        for state in 0..2 {
            for next_state in 0..2 {
                assert!(
                    transition_probability(state, next_state)
                        .le(upper_probability(state, next_state))
                );
            }
        }
    }

    #[test]
    fn every_fixture_upper_probability_dominates_the_true_transition() {
        for state in 0..2 {
            for next_state in 0..2 {
                assert!(
                    transition_probability(state, next_state)
                        .le(upper_probability(state, next_state))
                );
            }
        }
    }

    #[test]
    fn every_predictable_simplex_portfolio_has_conditional_expected_factor_at_most_one() {
        for portfolio in portfolios() {
            for current_state in 0..2 {
                assert!(conditional_expected_factor(current_state, portfolio).le(Rational::one()));
            }
        }
    }

    #[test]
    fn restart_prior_and_unstarted_reserve_conserve_unit_capital_exactly() {
        for time in 1..=HORIZON {
            let started = (1..=time).fold(Rational::zero(), |sum, start| {
                sum.add(Rational::new(1, (start * (start + 1)) as u128))
            });
            let reserve = Rational::new(1, (time + 1) as u128);
            assert_eq!(started.add(reserve), Rational::one());
        }
    }

    #[test]
    fn exact_path_probability_mass_is_one_for_each_initial_state() {
        for initial_state in 0..2 {
            let episode = enumerate_episode(initial_state);
            assert_eq!(episode.enumerated_path_count, 64);
            assert_eq!(
                episode.total_probability_numerator,
                episode.probability_denominator
            );
        }
    }

    #[test]
    fn exact_anytime_crossing_masses_match_the_known_answer() {
        let state_zero = enumerate_episode(0);
        let state_one = enumerate_episode(1);
        assert_eq!(state_zero.crossing_probability_numerator, 128);
        assert_eq!(state_one.crossing_probability_numerator, 128);
        assert_eq!(state_zero.probability_denominator, 4096);
        assert_eq!(state_one.probability_denominator, 4096);
    }

    #[test]
    fn exact_anytime_crossing_probability_is_below_the_declared_alpha() {
        for initial_state in 0..2 {
            let episode = enumerate_episode(initial_state);
            assert!(
                episode.crossing_probability_numerator * ALPHA.denominator
                    <= episode.probability_denominator * ALPHA.numerator
            );
        }
    }

    #[test]
    fn report_keeps_calibration_and_hardware_claims_unestablished() {
        let value = report();
        assert_eq!(
            value["guarantee"]["calibration_coverage_established"],
            false
        );
        assert_eq!(value["guarantee"]["arbitrary_serial_pmu_pfa_claim"], false);
        assert_eq!(value["guarantee"]["physical_hardware_evidence"], false);
        assert_eq!(
            value["guarantee"]["effective_sample_size_estimate"],
            Value::Null
        );
    }
}
