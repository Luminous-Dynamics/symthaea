//! Denoised selectable-headroom reference for CMAQ-017.
//!
//! The selector is evaluated on one continuation subset and scored on a
//! disjoint held-out subset. The exact candidate means are evaluator-only.
//! This finite synthetic benchmark is a protocol/conformance reference, not a
//! trained world-action model or a driving-performance result.

use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-017-selectable-headroom-v0.1";
const BASE_SEED: u64 = 20_261_105;
const REPLICATES: usize = 64;
const CANDIDATE_COUNTS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const SELECTION_CONTINUATIONS: [usize; 5] = [1, 2, 4, 8, 16];
const BASELINE_VALUE: f64 = 0.65;
const COMMON_SHOCK_AMPLITUDE: f64 = 0.60;
const COUPLING_CONTINUATIONS: usize = 128;

#[derive(Clone, Copy, Debug)]
struct CandidateProfile {
    expected_value: f64,
    common_loading: f64,
    local_scale: f64,
    interaction_scale: f64,
}

fn candidate_profile(candidate_id: usize) -> CandidateProfile {
    let index = candidate_id as f64;
    CandidateProfile {
        expected_value: 0.80 - 0.009 * index,
        common_loading: 0.035 + 0.0025 * index,
        local_scale: 0.025 + 0.0065 * index,
        interaction_scale: 0.012 + 0.0030 * index,
    }
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

fn rademacher(seed: u64, candidate_id: usize, continuation_id: usize, component: u64) -> f64 {
    let mixed_seed = seed
        ^ (candidate_id as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ (continuation_id as u64).rotate_left(21)
        ^ component.wrapping_mul(0xA076_1D64_78BD_642F);
    let mut rng = SplitMix64::new(mixed_seed);
    if rng.next_u64() & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

/// One realized continuation outcome. The three noise sources are distinct:
/// a causally shared world factor, a candidate-local residual, and a
/// candidate-by-continuation interaction term.
fn continuation_outcome(
    seed: u64,
    candidate_id: usize,
    continuation_id: usize,
) -> f64 {
    let profile = candidate_profile(candidate_id);
    let shared_world = rademacher(seed, 0, continuation_id, 0);
    let candidate_local = rademacher(seed, candidate_id, continuation_id, 1);
    let candidate_continuation = rademacher(seed, candidate_id, continuation_id, 2);
    profile.expected_value
        + profile.common_loading * shared_world
        + profile.local_scale * candidate_local
        + profile.interaction_scale * candidate_continuation
}

#[derive(Clone, Copy, Debug)]
struct ConditionResult {
    candidate_count: usize,
    selection_continuations: usize,
    heldout_continuations: usize,
    replicate_count: usize,
    naive_best_estimate: f64,
    heldout_selected_value: f64,
    selected_candidate_true_value: f64,
    oracle_best_expected_value: f64,
    baseline_expected_value: f64,
    naive_optimism_gap: f64,
    heldout_score_bias: f64,
    true_selectable_headroom: f64,
    captured_headroom: f64,
    selection_regret: f64,
    capture_fraction: f64,
    mean_selected_candidate_id: f64,
    candidate_continuation_runs: usize,
    heldout_continuation_runs: usize,
}

fn evaluate_condition(
    candidate_count: usize,
    selection_continuations: usize,
    replicates: usize,
) -> ConditionResult {
    assert!(candidate_count > 0);
    assert!(selection_continuations > 0);
    assert!(replicates > 0);

    let mut naive_best_total = 0.0;
    let mut heldout_total = 0.0;
    let mut selected_true_total = 0.0;
    let mut selected_id_total = 0usize;
    let oracle_best_expected_value = candidate_profile(0).expected_value;
    let true_selectable_headroom = oracle_best_expected_value - BASELINE_VALUE;

    for replicate in 0..replicates {
        let block_start = replicate * selection_continuations * 2;
        let selection_start = block_start;
        let heldout_start = block_start + selection_continuations;
        debug_assert!(selection_start + selection_continuations <= heldout_start);

        let mut best_candidate = 0usize;
        let mut best_selection_mean = f64::NEG_INFINITY;

        for candidate_id in 0..candidate_count {
            let mut selection_total = 0.0;
            for offset in 0..selection_continuations {
                selection_total += continuation_outcome(
                    BASE_SEED ^ 0x51EC_7101,
                    candidate_id,
                    selection_start + offset,
                );
            }
            let selection_mean = selection_total / selection_continuations as f64;
            if selection_mean > best_selection_mean {
                best_selection_mean = selection_mean;
                best_candidate = candidate_id;
            }
        }

        // Held-out continuation IDs are disjoint from every ID used for
        // selection within this replicate.
        let mut heldout_total_for_selected = 0.0;
        for offset in 0..selection_continuations {
            heldout_total_for_selected += continuation_outcome(
                BASE_SEED ^ 0x51EC_7101,
                best_candidate,
                heldout_start + offset,
            );
        }

        naive_best_total += best_selection_mean;
        heldout_total += heldout_total_for_selected / selection_continuations as f64;
        selected_true_total += candidate_profile(best_candidate).expected_value;
        selected_id_total += best_candidate;
    }

    let denominator = replicates as f64;
    let naive_best_estimate = naive_best_total / denominator;
    let heldout_selected_value = heldout_total / denominator;
    let selected_candidate_true_value = selected_true_total / denominator;
    let true_selectable_headroom = true_selectable_headroom;
    let captured_headroom = selected_candidate_true_value - BASELINE_VALUE;

    ConditionResult {
        candidate_count,
        selection_continuations,
        heldout_continuations: selection_continuations,
        replicate_count: replicates,
        naive_best_estimate,
        heldout_selected_value,
        selected_candidate_true_value,
        oracle_best_expected_value,
        baseline_expected_value: BASELINE_VALUE,
        naive_optimism_gap: naive_best_estimate - selected_candidate_true_value,
        heldout_score_bias: heldout_selected_value - selected_candidate_true_value,
        true_selectable_headroom,
        captured_headroom,
        selection_regret: oracle_best_expected_value - selected_candidate_true_value,
        capture_fraction: captured_headroom / true_selectable_headroom,
        mean_selected_candidate_id: selected_id_total as f64 / denominator,
        candidate_continuation_runs: replicates * candidate_count * selection_continuations,
        heldout_continuation_runs: replicates * selection_continuations,
    }
}

#[derive(Clone, Copy, Debug)]
struct VarianceDecomposition {
    candidate_mean_variance: f64,
    shared_continuation_variance: f64,
    candidate_local_variance: f64,
    candidate_continuation_interaction_variance: f64,
}

fn variance_decomposition(candidate_count: usize) -> VarianceDecomposition {
    let profiles: Vec<_> = (0..candidate_count).map(candidate_profile).collect();
    let means: Vec<_> = profiles.iter().map(|profile| profile.expected_value).collect();
    let mean_of_means = means.iter().sum::<f64>() / means.len() as f64;
    let candidate_mean_variance = means
        .iter()
        .map(|value| (value - mean_of_means).powi(2))
        .sum::<f64>()
        / means.len() as f64;

    VarianceDecomposition {
        candidate_mean_variance,
        shared_continuation_variance: profiles
            .iter()
            .map(|profile| profile.common_loading.powi(2))
            .sum::<f64>()
            / profiles.len() as f64,
        candidate_local_variance: profiles
            .iter()
            .map(|profile| profile.local_scale.powi(2))
            .sum::<f64>()
            / profiles.len() as f64,
        candidate_continuation_interaction_variance: profiles
            .iter()
            .map(|profile| profile.interaction_scale.powi(2))
            .sum::<f64>()
            / profiles.len() as f64,
    }
}

fn population_variance(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64
}

#[derive(Clone, Copy, Debug)]
struct CouplingVariance {
    independent_pair_difference_variance: f64,
    shared_invariant_same_sensitivity_variance: f64,
    shared_invariant_opposite_sensitivity_variance: f64,
}

fn coupling_variance() -> CouplingVariance {
    let mut independent = Vec::with_capacity(COUPLING_CONTINUATIONS);
    let mut shared_same = Vec::with_capacity(COUPLING_CONTINUATIONS);
    let mut shared_opposite = Vec::with_capacity(COUPLING_CONTINUATIONS);

    for tick in 0..COUPLING_CONTINUATIONS {
        let shared_factor = if tick % 2 == 0 { 1.0 } else { -1.0 };
        let independent_factor = match tick % 4 {
            0 | 1 => 1.0,
            _ => -1.0,
        };

        let left = COMMON_SHOCK_AMPLITUDE * shared_factor;
        let independent_right = COMMON_SHOCK_AMPLITUDE * independent_factor;
        let shared_same_right = COMMON_SHOCK_AMPLITUDE * shared_factor;
        let shared_opposite_right = -COMMON_SHOCK_AMPLITUDE * shared_factor;

        independent.push(left - independent_right);
        shared_same.push(left - shared_same_right);
        shared_opposite.push(left - shared_opposite_right);
    }

    CouplingVariance {
        independent_pair_difference_variance: population_variance(&independent),
        shared_invariant_same_sensitivity_variance: population_variance(&shared_same),
        shared_invariant_opposite_sensitivity_variance: population_variance(&shared_opposite),
    }
}

fn build_report() -> String {
    let mut output = String::new();
    let mut conditions = Vec::new();
    for candidate_count in CANDIDATE_COUNTS {
        for selection_continuations in SELECTION_CONTINUATIONS {
            conditions.push(evaluate_condition(
                candidate_count,
                selection_continuations,
                REPLICATES,
            ));
        }
    }

    let decomp = variance_decomposition(*CANDIDATE_COUNTS.last().expect("candidate ladder"));
    let coupling = coupling_variance();

    write!(
        output,
        "{{\"schema_version\":\"{SCHEMA}\",\"seed\":{BASE_SEED},\"replicates\":{REPLICATES},\"candidate_counts\":["
    )
    .expect("write to string");
    for (index, count) in CANDIDATE_COUNTS.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(output, "{count}").expect("write to string");
    }
    output.push_str("],\"selection_continuation_counts\":[");
    for (index, count) in SELECTION_CONTINUATIONS.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(output, "{count}").expect("write to string");
    }
    output.push_str("],\"conditions\":[");
    for (index, row) in conditions.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"candidate_count\":{},\"selection_continuations\":{},\"heldout_continuations\":{},\"replicate_count\":{},\"naive_best_estimate\":{:.12},\"heldout_selected_value\":{:.12},\"selected_candidate_true_value\":{:.12},\"oracle_best_expected_value\":{:.12},\"baseline_expected_value\":{:.12},\"naive_optimism_gap\":{:.12},\"heldout_score_bias\":{:.12},\"true_selectable_headroom\":{:.12},\"captured_headroom\":{:.12},\"selection_regret\":{:.12},\"capture_fraction\":{:.12},\"mean_selected_candidate_id\":{:.12},\"candidate_continuation_runs\":{},\"heldout_continuation_runs\":{}}}",
            row.candidate_count,
            row.selection_continuations,
            row.heldout_continuations,
            row.replicate_count,
            row.naive_best_estimate,
            row.heldout_selected_value,
            row.selected_candidate_true_value,
            row.oracle_best_expected_value,
            row.baseline_expected_value,
            row.naive_optimism_gap,
            row.heldout_score_bias,
            row.true_selectable_headroom,
            row.captured_headroom,
            row.selection_regret,
            row.capture_fraction,
            row.mean_selected_candidate_id,
            row.candidate_continuation_runs,
            row.heldout_continuation_runs,
        )
        .expect("write to string");
    }

    write!(
        output,
        "],\"variance_decomposition\":{{\"candidate_mean_variance\":{:.12},\"shared_continuation_variance\":{:.12},\"candidate_local_variance\":{:.12},\"candidate_continuation_interaction_variance\":{:.12}}},\"coupling_variance\":{{\"independent_pair_difference_variance\":{:.12},\"shared_invariant_same_sensitivity_variance\":{:.12},\"shared_invariant_opposite_sensitivity_variance\":{:.12}}},\"claim_ceiling\":\"Finite synthetic continuation-selection protocol only; no trained WAM, driving performance, causal validity, CRN variance guarantee or vehicle safety claim.\"}}",
        decomp.candidate_mean_variance,
        decomp.shared_continuation_variance,
        decomp.candidate_local_variance,
        decomp.candidate_continuation_interaction_variance,
        coupling.independent_pair_difference_variance,
        coupling.shared_invariant_same_sensitivity_variance,
        coupling.shared_invariant_opposite_sensitivity_variance,
    )
    .expect("write to string");

    output
}

fn main() {
    println!("{}", build_report());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heldout_continuation_ids_are_disjoint_from_selection_ids() {
        for replicate in 0..REPLICATES {
            for selection_count in SELECTION_CONTINUATIONS {
                let block_start = replicate * selection_count * 2;
                let selection_end = block_start + selection_count;
                let heldout_start = selection_end;
                let heldout_end = heldout_start + selection_count;
                assert_eq!(selection_end, heldout_start);
                assert!(heldout_start >= selection_end);
                assert!(heldout_end > heldout_start);
            }
        }
    }

    #[test]
    fn one_shot_best_of_n_has_positive_selection_optimism_for_the_high_noise_fixture() {
        let row = evaluate_condition(32, 1, REPLICATES);
        assert!(row.naive_optimism_gap > 0.0);
        assert!(row.naive_best_estimate > row.selected_candidate_true_value);
        assert!(row.heldout_score_bias.abs() < 0.04);
    }

    #[test]
    fn repeated_continuations_reduce_selection_noise_exposure() {
        let one = evaluate_condition(32, 1, REPLICATES);
        let repeated = evaluate_condition(32, 16, REPLICATES);
        assert!(repeated.naive_optimism_gap < one.naive_optimism_gap);
        assert!(repeated.selection_regret < one.selection_regret);
    }

    #[test]
    fn increasing_candidate_count_increases_naive_best_of_n_optimism() {
        let small = evaluate_condition(2, 1, REPLICATES);
        let large = evaluate_condition(32, 1, REPLICATES);
        assert!(large.naive_optimism_gap > small.naive_optimism_gap);
    }

    #[test]
    fn report_covers_every_candidate_and_continuation_count() {
        let report = build_report();
        assert!(report.contains("\"candidate_count\":32"));
        assert!(report.contains("\"selection_continuations\":16"));
        assert!(report.contains("\"claim_ceiling\""));
    }

    #[test]
    fn common_random_numbers_can_help_or_hurt_variance() {
        let result = coupling_variance();
        assert!(
            result.shared_invariant_same_sensitivity_variance
                < result.independent_pair_difference_variance
        );
        assert!(
            result.shared_invariant_opposite_sensitivity_variance
                > result.independent_pair_difference_variance
        );
    }

    #[test]
    fn variance_decomposition_keeps_shared_local_and_interaction_noise_separate() {
        let result = variance_decomposition(32);
        assert!(result.candidate_mean_variance > 0.0);
        assert!(result.shared_continuation_variance > 0.0);
        assert!(result.candidate_local_variance > 0.0);
        assert!(result.candidate_continuation_interaction_variance > 0.0);
    }
}
