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

// Frozen, deliberately loose upper envelope. Every coordinate dominates P.
// These are fixture inputs, not confidence bounds estimated from hardware.
const UPPER_NUMERATORS: [[u128; 2]; 2] = [[4, 1], [3, 3]];
const UPPER_DENOMINATORS: [[u128; 2]; 2] = [[5, 2], [5, 5]];

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

fn upper_probability(current_state: usize, next_state: usize) -> Rational {
    Rational::new(
        UPPER_NUMERATORS[current_state][next_state],
        UPPER_DENOMINATORS[current_state][next_state],
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
            "numerators": UPPER_NUMERATORS,
            "denominators": UPPER_DENOMINATORS,
            "calibrated_from_independent_data": false,
            "interpretation": "fixture constants that dominate the stated transition rows"
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
        assert_eq!(state_zero.crossing_probability_numerator, 64);
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
