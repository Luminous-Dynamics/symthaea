use serde_json::{json, Value};

const GRID_DENOMINATOR: u128 = 5;
const GRID_NUMERATORS: [u128; 4] = [1, 2, 3, 4];
const HORIZON: u32 = 8;
const GLOBAL_DELTA_NUMERATOR: u128 = 1;
const GLOBAL_DELTA_DENOMINATOR: u128 = 10;
const ROW_DELTA: f64 = 0.05;
const OUTWARD_ULPS: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Interval {
    lower: f64,
    upper: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GridEpisode {
    p0_numerator: u128,
    p1_numerator: u128,
    initial_state: usize,
    path_count: u64,
    total_probability_numerator: u128,
    failure_probability_numerator: u128,
    denominator: u128,
}

fn log_factorial(n: u32) -> f64 {
    (2..=n).map(|value| (value as f64).ln()).sum()
}

fn log_beta_mixture_ratio(successes: u32, failures: u32, p: f64) -> f64 {
    let log_beta = log_factorial(successes)
        + log_factorial(failures)
        - log_factorial(successes + failures + 1);

    if p <= 0.0 {
        return if successes == 0 { log_beta } else { f64::INFINITY };
    }
    if p >= 1.0 {
        return if failures == 0 { log_beta } else { f64::INFINITY };
    }

    log_beta
        - (successes as f64) * p.ln()
        - (failures as f64) * (-p).ln_1p()
}

fn widen_down(value: f64) -> f64 {
    if value <= 0.0 {
        return 0.0;
    }
    let mut bits = value.to_bits();
    for _ in 0..OUTWARD_ULPS {
        bits = bits.saturating_sub(1);
    }
    f64::from_bits(bits).max(0.0)
}

fn widen_up(value: f64) -> f64 {
    if value >= 1.0 {
        return 1.0;
    }
    let mut bits = value.to_bits();
    for _ in 0..OUTWARD_ULPS {
        bits = bits.saturating_add(1);
    }
    f64::from_bits(bits).min(1.0)
}

fn beta_mixture_confidence_sequence(
    successes: u32,
    failures: u32,
    delta: f64,
) -> Interval {
    assert!(delta > 0.0 && delta < 1.0);
    if successes == 0 && failures == 0 {
        return Interval { lower: 0.0, upper: 1.0 };
    }

    let n = successes + failures;
    let mle = successes as f64 / n as f64;
    let threshold = (1.0 / delta).ln();

    let lower = if successes == 0 {
        0.0
    } else {
        let mut outside = 0.0;
        let mut inside = mle;
        for _ in 0..96 {
            let midpoint = (outside + inside) / 2.0;
            if log_beta_mixture_ratio(successes, failures, midpoint) > threshold {
                outside = midpoint;
            } else {
                inside = midpoint;
            }
        }
        widen_down(outside)
    };

    let upper = if failures == 0 {
        1.0
    } else {
        let mut inside = mle;
        let mut outside = 1.0;
        for _ in 0..96 {
            let midpoint = (inside + outside) / 2.0;
            if log_beta_mixture_ratio(successes, failures, midpoint) > threshold {
                outside = midpoint;
            } else {
                inside = midpoint;
            }
        }
        widen_up(outside)
    };

    Interval {
        lower: lower.min(mle),
        upper: upper.max(mle),
    }
}

fn confidence_sequence_table(horizon: u32, delta: f64) -> Vec<Vec<Interval>> {
    let mut table = vec![
        vec![Interval { lower: 0.0, upper: 1.0 }; (horizon + 1) as usize];
        (horizon + 1) as usize
    ];

    for successes in 0..=horizon {
        for failures in 0..=(horizon - successes) {
            table[successes as usize][failures as usize] =
                beta_mixture_confidence_sequence(successes, failures, delta);
        }
    }
    table
}

fn pow_u128(base: u128, exponent: u32) -> u128 {
    let mut result = 1_u128;
    for _ in 0..exponent {
        result *= base;
    }
    result
}

fn enumerate_grid_episode(
    p0_numerator: u128,
    p1_numerator: u128,
    initial_state: usize,
    table: &[Vec<Interval>],
) -> GridEpisode {
    assert!(p0_numerator > 0 && p0_numerator < GRID_DENOMINATOR);
    assert!(p1_numerator > 0 && p1_numerator < GRID_DENOMINATOR);
    assert!(initial_state < 2);

    let transition_numerators = [p0_numerator, p1_numerator];
    let path_count = 1_u64 << HORIZON;
    let denominator = pow_u128(GRID_DENOMINATOR, HORIZON);
    let mut total_probability_numerator = 0_u128;
    let mut failure_probability_numerator = 0_u128;

    for encoded_path in 0..path_count {
        let mut state = initial_state;
        let mut counts = [[0_u32; 2]; 2];
        let mut path_probability_numerator = 1_u128;
        let mut confidence_sequence_failed = false;

        for time_index in 0..HORIZON {
            let next_state = ((encoded_path >> time_index) & 1) as usize;
            let success_numerator = transition_numerators[state];
            let transition_numerator = if next_state == 1 {
                success_numerator
            } else {
                GRID_DENOMINATOR - success_numerator
            };
            path_probability_numerator *= transition_numerator;
            counts[state][if next_state == 1 { 0 } else { 1 }] += 1;

            for row in 0..2 {
                let interval = table[counts[row][0] as usize][counts[row][1] as usize];
                let true_p = transition_numerators[row] as f64 / GRID_DENOMINATOR as f64;
                if true_p < interval.lower - 1e-12 || true_p > interval.upper + 1e-12 {
                    confidence_sequence_failed = true;
                }
            }
            state = next_state;
        }

        total_probability_numerator += path_probability_numerator;
        if confidence_sequence_failed {
            failure_probability_numerator += path_probability_numerator;
        }
    }

    GridEpisode {
        p0_numerator,
        p1_numerator,
        initial_state,
        path_count,
        total_probability_numerator,
        failure_probability_numerator,
        denominator,
    }
}

fn grid_episodes() -> Vec<GridEpisode> {
    let table = confidence_sequence_table(HORIZON, ROW_DELTA);
    let mut episodes = Vec::new();

    for p0 in GRID_NUMERATORS {
        for p1 in GRID_NUMERATORS {
            for initial_state in 0..2 {
                episodes.push(enumerate_grid_episode(p0, p1, initial_state, &table));
            }
        }
    }
    episodes
}

fn report() -> Value {
    let episodes = grid_episodes();
    let maximum_failure = episodes
        .iter()
        .map(|episode| {
            episode.failure_probability_numerator as f64 / episode.denominator as f64
        })
        .fold(0.0_f64, f64::max);

    json!({
        "schema_version": "hardsec-markov-transition-cs-audit-v1",
        "result": "finite_grid_exact_path_audit_passed",
        "claim_scope": "fully_observed_finite_state_markov_model_with_unknown_fixed_transition_rows",
        "confidence_sequence": {
            "method": "beta_bernoulli_mixture",
            "prior": "Beta(1,1)",
            "row_delta": ROW_DELTA,
            "global_delta": {
                "numerator": GLOBAL_DELTA_NUMERATOR,
                "denominator": GLOBAL_DELTA_DENOMINATOR
            },
            "row_failure_budgets_sum_to_global_delta": true,
            "upper_envelope": "u_1=upper_endpoint; u_0=1-lower_endpoint",
            "endpoint_solver": "96_step_bisection_with_64_ulp_outward_padding",
            "floating_point_interval_proof_status": "not_formally_interval_certified"
        },
        "detector": {
            "alpha": 0.1,
            "betting_distribution": [0.5, 0.5],
            "factor": "q_outcome/u_outcome",
            "analysis_only_exit_time_used_in_proof": true,
            "actual_monitor_continues_after_unknown_exit": true
        },
        "enumeration": {
            "transition_grid_denominator": GRID_DENOMINATOR,
            "transition_numerators": GRID_NUMERATORS,
            "transition_matrices_checked": 16,
            "initial_states_per_matrix": 2,
            "episodes": episodes.len(),
            "horizon": HORIZON,
            "paths_per_episode": 1_u64 << HORIZON,
            "probability_denominator_per_episode": pow_u128(GRID_DENOMINATOR, HORIZON),
            "all_path_masses_sum_to_one": episodes.iter().all(|episode| {
                episode.total_probability_numerator == episode.denominator
            }),
            "maximum_joint_cs_failure_probability_on_grid": maximum_failure,
            "maximum_joint_cs_failure_numerator": episodes
                .iter()
                .map(|episode| episode.failure_probability_numerator)
                .max()
                .unwrap_or(0),
            "grid_failure_probability_within_global_delta": episodes.iter().all(|episode| {
                episode.failure_probability_numerator * GLOBAL_DELTA_DENOMINATOR
                    <= episode.denominator * GLOBAL_DELTA_NUMERATOR
            }),
            "episode_results": episodes.iter().map(|episode| json!({
                "p0_numerator": episode.p0_numerator,
                "p1_numerator": episode.p1_numerator,
                "grid_denominator": GRID_DENOMINATOR,
                "initial_state": episode.initial_state,
                "path_count": episode.path_count,
                "total_probability_numerator": episode.total_probability_numerator,
                "probability_denominator": episode.denominator,
                "cs_failure_probability_numerator": episode.failure_probability_numerator
            })).collect::<Vec<_>>()
        },
        "guarantee": {
            "known_markov_transition_assumption": "fixed_unknown_rows_with_fully_observed_state",
            "beta_mixture_cs_theorem": "time_uniform_for_each_bernoulli_transition_row",
            "joint_row_coverage": "at_least_1-delta_by_union_bound",
            "ideal_arithmetic_detector_claim": "marginal_anytime_pfa_at_most_alpha_plus_delta",
            "floating_point_runtime_claim": "not_established_until_outward_interval_arithmetic_is_certified",
            "arbitrary_serial_pmu_pfa_claim": false,
            "hidden_state_supported": false,
            "model_misspecification_supported": false,
            "selective_admission_supported": false,
            "effective_sample_size_estimate": null
        },
        "limitations": [
            "This is exhaustive only on the declared 16 rational transition matrices and two initial states.",
            "The grid is a diagnostic of the CS and envelope implementation, not a universal proof.",
            "The floating-point endpoint solver is widened but is not a formally verified interval-arithmetic implementation.",
            "The Markov assumption is not established for physical PMU telemetry.",
            "No detector-crossing or SOTA-performance claim is made by this CS coverage audit."
        ]
    })
}

fn main() {
    println!("{}", serde_json::to_string_pretty(&report()).expect("JSON report serialization"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_mixture_confidence_sequence_contains_the_mle() {
        for successes in 0..=8 {
            for failures in 0..=(8 - successes) {
                let interval = beta_mixture_confidence_sequence(successes, failures, ROW_DELTA);
                let mle = if successes + failures == 0 {
                    0.5
                } else {
                    successes as f64 / (successes + failures) as f64
                };
                assert!(interval.lower >= 0.0);
                assert!(interval.upper <= 1.0);
                assert!(interval.lower <= mle);
                assert!(mle <= interval.upper);
                assert!(interval.lower < interval.upper);
            }
        }
    }

    #[test]
    fn known_beta_mixture_interval_matches_reference_values() {
        let interval = beta_mixture_confidence_sequence(1, 1, 0.05);
        assert!((interval.lower - 0.00840395987491247).abs() < 1e-10);
        assert!((interval.upper - 0.9915960401250877).abs() < 1e-10);
        let one_failure = beta_mixture_confidence_sequence(0, 1, 0.05);
        assert_eq!(one_failure.lower, 0.0);
        assert!((one_failure.upper - 0.975).abs() < 1e-10);
    }

    #[test]
    fn exact_markov_grid_path_masses_and_joint_cs_failure_budget_hold() {
        let episodes = grid_episodes();
        assert_eq!(episodes.len(), 32);
        for episode in &episodes {
            assert_eq!(episode.path_count, 256);
            assert_eq!(episode.denominator, 390_625);
            assert_eq!(episode.total_probability_numerator, episode.denominator);
            assert!(
                episode.failure_probability_numerator * GLOBAL_DELTA_DENOMINATOR
                    <= episode.denominator * GLOBAL_DELTA_NUMERATOR,
                "CS failure budget exceeded for p0={}, p1={}, initial={}",
                episode.p0_numerator,
                episode.p1_numerator,
                episode.initial_state
            );
        }
    }

    #[test]
    fn row_cs_implies_predictable_upper_envelope_on_all_grid_points() {
        let table = confidence_sequence_table(HORIZON, ROW_DELTA);
        for p_numerator in GRID_NUMERATORS {
            let p = p_numerator as f64 / GRID_DENOMINATOR as f64;
            for successes in 0..=HORIZON {
                for failures in 0..=(HORIZON - successes) {
                    let interval = table[successes as usize][failures as usize];
                    if p + 1e-12 < interval.lower || p > interval.upper + 1e-12 {
                        continue;
                    }
                    let u_one = interval.upper;
                    let u_zero = 1.0 - interval.lower;
                    assert!(p <= u_one + 1e-12);
                    assert!(1.0 - p <= u_zero + 1e-12);
                    assert!(u_one > 0.0 && u_zero > 0.0);
                }
            }
        }
    }

    #[test]
    fn receipt_preserves_numeric_and_model_claim_ceiling() {
        let value = report();
        assert_eq!(value["result"], "finite_grid_exact_path_audit_passed");
        assert_eq!(value["enumeration"]["all_path_masses_sum_to_one"], true);
        assert_eq!(value["enumeration"]["grid_failure_probability_within_global_delta"], true);
        assert_eq!(
            value["guarantee"]["ideal_arithmetic_detector_claim"],
            "marginal_anytime_pfa_at_most_alpha_plus_delta"
        );
        assert_eq!(
            value["guarantee"]["floating_point_runtime_claim"],
            "not_established_until_outward_interval_arithmetic_is_certified"
        );
        assert_eq!(value["guarantee"]["arbitrary_serial_pmu_pfa_claim"], false);
        assert_eq!(value["guarantee"]["effective_sample_size_estimate"], Value::Null);
    }
}
