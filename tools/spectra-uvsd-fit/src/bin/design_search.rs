// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Robust, pre-study category-boundary design search using expected information.
//!
//! This is an exploratory simulation/design tool. It does not authorize or
//! perform human-subject data collection, stimulus adaptation, or actuation.

use spectra_uvsd_fit::expected_fisher_information_for_design;

const CANDIDATE_THRESHOLDS: [f64; 13] = [
    -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
];
const THRESHOLD_COUNT: usize = 5;
const SIGNAL_TRIALS: u64 = 250;
const NOISE_TRIALS: u64 = 250;

const MU_GRID: [f64; 4] = [0.5, 1.0, 1.5, 2.0];
const SIGMA_GRID: [f64; 4] = [0.70, 1.0, 1.30, 1.60];

// Standard-normal equal-probability cut-points for six ordinal categories.
const EQUAL_PROBABILITY_BASELINE: [f64; 5] = [
    -0.967_421_566_101_701,
    -0.430_727_299_295_457,
    0.0,
    0.430_727_299_295_457,
    0.967_421_566_101_701,
];

// Existing illustrative fixed-bin design used as an additional baseline.
const FIXED_SPACING_BASELINE: [f64; 5] = [-1.5, -0.75, 0.0, 0.75, 1.5];

#[derive(Debug, Clone)]
struct DesignScore {
    thresholds: Vec<f64>,
    weak_cells: usize,
    worst_da_se: f64,
    mean_da_se: f64,
    worst_condition: f64,
}

fn enumerate_designs(
    start: usize,
    current: &mut Vec<f64>,
    designs: &mut Vec<Vec<f64>>,
) {
    if current.len() == THRESHOLD_COUNT {
        designs.push(current.clone());
        return;
    }
    let remaining = THRESHOLD_COUNT - current.len();
    if CANDIDATE_THRESHOLDS.len().saturating_sub(start) < remaining {
        return;
    }
    for index in start..CANDIDATE_THRESHOLDS.len() {
        current.push(CANDIDATE_THRESHOLDS[index]);
        enumerate_designs(index + 1, current, designs);
        current.pop();
    }
}

fn evaluate_design(thresholds: &[f64]) -> Option<DesignScore> {
    if thresholds.len() != THRESHOLD_COUNT
        || thresholds.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }

    let mut weak_cells = 0usize;
    let mut valid_cells = 0usize;
    let mut worst_da_se = 0.0_f64;
    let mut sum_da_se = 0.0_f64;
    let mut worst_condition = 0.0_f64;

    for &mu in &MU_GRID {
        for &sigma in &SIGMA_GRID {
            let Some(information) = expected_fisher_information_for_design(
                mu,
                sigma,
                thresholds,
                SIGNAL_TRIALS,
                NOISE_TRIALS,
            ) else {
                weak_cells += 1;
                continue;
            };

            if information.weakly_identified {
                weak_cells += 1;
                continue;
            }
            let Some(da_se) = information.d_a_standard_error else {
                weak_cells += 1;
                continue;
            };
            let Some(condition) = information.condition_number else {
                weak_cells += 1;
                continue;
            };
            if !da_se.is_finite() || !condition.is_finite() {
                weak_cells += 1;
                continue;
            }

            valid_cells += 1;
            worst_da_se = worst_da_se.max(da_se);
            sum_da_se += da_se;
            worst_condition = worst_condition.max(condition);
        }
    }

    if valid_cells == 0 {
        return None;
    }
    Some(DesignScore {
        thresholds: thresholds.to_vec(),
        weak_cells,
        worst_da_se,
        mean_da_se: sum_da_se / valid_cells as f64,
        worst_condition,
    })
}

fn print_score(label: &str, score: &DesignScore) {
    println!(
        "{label},{},{:.8},{:.8},{:.8},{}",
        score.weak_cells,
        score.worst_da_se,
        score.mean_da_se,
        score.worst_condition,
        quoted_thresholds(&score.thresholds)
    );
}

fn quoted_thresholds(values: &[f64]) -> String {
    let joined = values
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(";");
    format!(r#""{joined}""#)
}

fn main() {
    let equal_probability = evaluate_design(&EQUAL_PROBABILITY_BASELINE)
        .expect("the equal-probability baseline is a valid six-category design");
    let fixed_spacing = evaluate_design(&FIXED_SPACING_BASELINE)
        .expect("the fixed-spacing baseline is a valid six-category design");

    let mut designs = Vec::new();
    enumerate_designs(0, &mut Vec::new(), &mut designs);

    let mut scores: Vec<DesignScore> = designs
        .iter()
        .filter_map(|thresholds| evaluate_design(thresholds))
        .collect();
    scores.sort_by(|left, right| {
        left.weak_cells
            .cmp(&right.weak_cells)
            .then_with(|| left.worst_da_se.total_cmp(&right.worst_da_se))
            .then_with(|| left.mean_da_se.total_cmp(&right.mean_da_se))
            .then_with(|| left.worst_condition.total_cmp(&right.worst_condition))
    });

    println!(
        "design,weak_cells_of_16,worst_case_da_se,mean_da_se,worst_condition,thresholds"
    );
    print_score("equal_probability_baseline", &equal_probability);
    print_score("fixed_spacing_baseline", &fixed_spacing);
    for (index, score) in scores.iter().take(20).enumerate() {
        print_score(&format!("robust_candidate_{}", index + 1), score);
    }

    if let Some(best) = scores.first() {
        println!(
            "# candidate_count={}, best_worst_case_improvement_vs_equal_probability={:.6}, best_worst_case_improvement_vs_fixed_spacing={:.6}",
            scores.len(),
            1.0 - best.worst_da_se / equal_probability.worst_da_se,
            1.0 - best.worst_da_se / fixed_spacing.worst_da_se
        );
    }
}
