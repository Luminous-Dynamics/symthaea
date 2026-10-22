// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic synthetic harness for CASTING-002B.
//!
//! This test-only harness compares fixed-budget sampling designs on synthetic
//! objectives with known optima. It is not a foundry process model or optimizer
//! recommendation. Truth is used only by the evaluator to calculate regret after
//! a query; sampling designs do not receive unqueried candidate outcomes.
//!
//! Focused invocation:
//! `cargo test -p symthaea-core casting_optimization_benchmark`

use std::fmt;

const LOW_FIDELITY_COST_UNITS: u64 = 1;
const HIGH_FIDELITY_COST_UNITS: u64 = 10;
const SYNTHETIC_TARGET_OBJECTIVE: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum SyntheticCase {
    SmoothBowl = 0,
    CoupledFactors = 1,
    FeasibleIsland = 2,
    HeteroscedasticNoise = 3,
    FidelityDiscrepancy = 4,
    FailedCorner = 5,
}

impl SyntheticCase {
    const ALL: [Self; 6] = [
        Self::SmoothBowl,
        Self::CoupledFactors,
        Self::FeasibleIsland,
        Self::HeteroscedasticNoise,
        Self::FidelityDiscrepancy,
        Self::FailedCorner,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::SmoothBowl => "smooth-bowl",
            Self::CoupledFactors => "coupled-factors",
            Self::FeasibleIsland => "feasible-island",
            Self::HeteroscedasticNoise => "heteroscedastic-noise",
            Self::FidelityDiscrepancy => "fidelity-discrepancy",
            Self::FailedCorner => "failed-corner",
        }
    }

    fn optimum(self) -> Point {
        match self {
            Self::SmoothBowl
            | Self::HeteroscedasticNoise
            | Self::FidelityDiscrepancy
            | Self::FailedCorner => Point { x: 0.37, y: 0.62 },
            Self::CoupledFactors => Point { x: 0.20, y: 0.29 },
            Self::FeasibleIsland => Point { x: 0.72, y: 0.31 },
        }
    }

    /// Exact, noiseless synthetic objective; every current case has optimum zero.
    fn objective(self, point: Point) -> f64 {
        match self {
            Self::CoupledFactors => {
                (point.x - 0.20).powi(2)
                    + (point.y - (0.15 + 0.70 * point.x)).powi(2)
            }
            Self::FeasibleIsland => {
                (point.x - 0.72).powi(2) + (point.y - 0.31).powi(2)
            }
            _ => (point.x - 0.37).powi(2) + (point.y - 0.62).powi(2),
        }
    }

    fn is_feasible(self, point: Point) -> bool {
        match self {
            // The minimum lies inside a bounded feasible island.
            Self::FeasibleIsland => {
                let dx = point.x - 0.72;
                let dy = point.y - 0.31;
                dx * dx + dy * dy <= 0.22_f64.powi(2)
            }
            _ => true,
        }
    }

    fn known_optimum_value(self) -> f64 {
        self.objective(self.optimum())
    }

    fn observation(self, point: Point, fidelity: Fidelity, run_seed: u64) -> f64 {
        let truth = self.objective(point);
        match self {
            Self::HeteroscedasticNoise => {
                // Common-random-number noise: same seed + point replays exactly,
                // while variance grows with x.
                let noise_seed = run_seed
                    ^ point.x.to_bits().rotate_left(7)
                    ^ point.y.to_bits().rotate_left(23);
                let mut rng = SplitMix64::new(noise_seed);
                let signed_noise = 2.0 * rng.unit_f64() - 1.0;
                truth + (0.01 + 0.12 * point.x) * signed_noise
            }
            Self::FidelityDiscrepancy if fidelity == Fidelity::Low => {
                // Deliberately biased cheap oracle. This is a synthetic stress case,
                // not a claim about any physical casting solver.
                truth + 0.15 * (point.x - point.y) + 0.05
            }
            _ => truth,
        }
    }

    fn can_fail(self, point: Point) -> bool {
        // Synthetic equipment-failure analogue. A failed query still costs resources.
        self == Self::FailedCorner && point.x > 0.90 && point.y > 0.90
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum SamplingDesign {
    Random = 0,
    UniformGrid = 1,
    LatinHypercube = 2,
    AdaptiveNearestNeighbor = 3,
}

impl SamplingDesign {
    const ALL: [Self; 4] = [
        Self::Random,
        Self::UniformGrid,
        Self::LatinHypercube,
        Self::AdaptiveNearestNeighbor,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Fidelity {
    Low = 0,
    High = 1,
}

impl Fidelity {
    const fn cost_units(self) -> u64 {
        match self {
            Self::Low => LOW_FIDELITY_COST_UNITS,
            Self::High => HIGH_FIDELITY_COST_UNITS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum TrialStatus {
    Feasible = 0,
    Infeasible = 1,
    Failed = 2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct TrialRecord {
    point: Point,
    status: TrialStatus,
    fidelity: Fidelity,
    true_objective_for_evaluation_only: f64,
    observed_objective: Option<f64>,
    cost_units: u64,
    cumulative_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct BenchmarkRun {
    case: SyntheticCase,
    design: SamplingDesign,
    fidelity: Fidelity,
    seed: u64,
    budget: usize,
    total_cost_units: u64,
    feasible_count: usize,
    infeasible_count: usize,
    failed_count: usize,
    cost_to_first_feasible: Option<u64>,
    cost_to_target: Option<u64>,
    best_observed_objective: Option<f64>,
    simple_regret: Option<f64>,
    transcript_digest_blake3: [u8; 32],
    transcript: Vec<TrialRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkError {
    ZeroBudget,
    EmptySeedSet,
    AdaptiveRequiresHistory,
    CostOverflow,
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBudget => f.write_str("benchmark budget must be greater than zero"),
            Self::EmptySeedSet => f.write_str("paired benchmark requires at least one seed"),
            Self::AdaptiveRequiresHistory => {
                f.write_str("adaptive sampling requires a sequential observation history")
            }
            Self::CostOverflow => f.write_str("benchmark cumulative cost overflowed"),
        }
    }
}

impl std::error::Error for BenchmarkError {}

/// Small deterministic PRNG frozen with the benchmark version.
#[derive(Debug, Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        // Exactly 53 random mantissa bits, yielding values in [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

fn sample_points(
    design: SamplingDesign,
    budget: usize,
    seed: u64,
) -> Result<Vec<Point>, BenchmarkError> {
    if budget == 0 {
        return Err(BenchmarkError::ZeroBudget);
    }

    let points = match design {
        SamplingDesign::Random => {
            let mut rng = SplitMix64::new(seed);
            (0..budget)
                .map(|_| Point {
                    x: rng.unit_f64(),
                    y: rng.unit_f64(),
                })
                .collect()
        }
        SamplingDesign::UniformGrid => {
            let side = (budget as f64).sqrt().ceil() as usize;
            let cells = side * side;
            (0..budget)
                .map(|index| {
                    // Spread samples across the whole grid if budget is not a square.
                    let cell = index * cells / budget;
                    Point {
                        x: ((cell % side) as f64 + 0.5) / side as f64,
                        y: ((cell / side) as f64 + 0.5) / side as f64,
                    }
                })
                .collect()
        }
        SamplingDesign::LatinHypercube => {
            let mut rng = SplitMix64::new(seed);
            let mut y_strata: Vec<usize> = (0..budget).collect();
            for index in (1..budget).rev() {
                let other = rng.index(index + 1);
                y_strata.swap(index, other);
            }
            (0..budget)
                .map(|index| Point {
                    x: (index as f64 + rng.unit_f64()) / budget as f64,
                    y: (y_strata[index] as f64 + rng.unit_f64()) / budget as f64,
                })
                .collect()
        }
        SamplingDesign::AdaptiveNearestNeighbor => {
            return Err(BenchmarkError::AdaptiveRequiresHistory);
        }
    };

    debug_assert_eq!(points.len(), budget);
    debug_assert!(points
        .iter()
        .all(|point| (0.0..1.0).contains(&point.x) && (0.0..1.0).contains(&point.y)));
    Ok(points)
}

fn evaluate_trial(
    case: SyntheticCase,
    point: Point,
    fidelity: Fidelity,
    run_seed: u64,
    cumulative_cost_units: u64,
) -> TrialRecord {
    let status = if case.can_fail(point) {
        TrialStatus::Failed
    } else if !case.is_feasible(point) {
        TrialStatus::Infeasible
    } else {
        TrialStatus::Feasible
    };
    let observed_objective = if status == TrialStatus::Feasible {
        Some(case.observation(point, fidelity, run_seed))
    } else {
        None
    };

    TrialRecord {
        point,
        status,
        fidelity,
        true_objective_for_evaluation_only: case.objective(point),
        observed_objective,
        cost_units: fidelity.cost_units(),
        cumulative_cost_units,
    }
}

fn run_benchmark(
    case: SyntheticCase,
    design: SamplingDesign,
    fidelity: Fidelity,
    seed: u64,
    budget: usize,
) -> Result<BenchmarkRun, BenchmarkError> {
    if budget == 0 {
        return Err(BenchmarkError::ZeroBudget);
    }
    let fixed_points = if design == SamplingDesign::AdaptiveNearestNeighbor {
        Vec::new()
    } else {
        sample_points(design, budget, seed)?
    };
    let adaptive_warm_start = if design == SamplingDesign::AdaptiveNearestNeighbor {
        adaptive_points(seed, budget)?
    } else {
        Vec::new()
    };
    let mut records = Vec::with_capacity(budget);
    let mut total_cost_units = 0_u64;
    let mut feasible_count = 0_usize;
    let mut infeasible_count = 0_usize;
    let mut failed_count = 0_usize;
    let mut cost_to_first_feasible = None;
    let mut cost_to_target = None;
    let mut best_observed_objective: Option<f64> = None;
    let mut best_true_objective_at_selected_point: Option<f64> = None;

    for query_index in 0..budget {
        let point = if design == SamplingDesign::AdaptiveNearestNeighbor {
            if query_index < adaptive_warm_start.len() {
                adaptive_warm_start[query_index]
            } else {
                choose_adaptive_point(&records).ok_or(BenchmarkError::AdaptiveRequiresHistory)?
            }
        } else {
            fixed_points[query_index]
        };

        total_cost_units = total_cost_units
            .checked_add(fidelity.cost_units())
            .ok_or(BenchmarkError::CostOverflow)?;
        let record = evaluate_trial(case, point, fidelity, seed, total_cost_units);

        match record.status {
            TrialStatus::Feasible => {
                feasible_count += 1;
                if cost_to_first_feasible.is_none() {
                    cost_to_first_feasible = Some(record.cumulative_cost_units);
                }
                let observed = record
                    .observed_objective
                    .expect("feasible trial has an observation");
                if cost_to_target.is_none() && observed <= SYNTHETIC_TARGET_OBJECTIVE {
                    cost_to_target = Some(record.cumulative_cost_units);
                }
                if best_observed_objective.map_or(true, |best| observed < best) {
                    best_observed_objective = Some(observed);
                    best_true_objective_at_selected_point =
                        Some(record.true_objective_for_evaluation_only);
                }
            }
            TrialStatus::Infeasible => infeasible_count += 1,
            TrialStatus::Failed => failed_count += 1,
        }
        records.push(record);
    }

    let simple_regret = best_true_objective_at_selected_point
        .map(|value| (value - case.known_optimum_value()).max(0.0));
    let transcript_digest_blake3 =
        digest_transcript(case, design, fidelity, seed, budget, &records);

    Ok(BenchmarkRun {
        case,
        design,
        fidelity,
        seed,
        budget,
        total_cost_units,
        feasible_count,
        infeasible_count,
        failed_count,
        cost_to_first_feasible,
        cost_to_target,
        best_observed_objective,
        simple_regret,
        transcript_digest_blake3,
        transcript: records,
    })
}

fn digest_transcript(
    case: SyntheticCase,
    design: SamplingDesign,
    fidelity: Fidelity,
    seed: u64,
    budget: usize,
    records: &[TrialRecord],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-casting-002b-synthetic-benchmark-v1");
    hasher.update(&[case as u8, design as u8, fidelity as u8]);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&(budget as u64).to_le_bytes());
    for record in records {
        hasher.update(&record.point.x.to_bits().to_le_bytes());
        hasher.update(&record.point.y.to_bits().to_le_bytes());
        hasher.update(&[record.status as u8, record.fidelity as u8]);
        hasher.update(
            &record
                .true_objective_for_evaluation_only
                .to_bits()
                .to_le_bytes(),
        );
        match record.observed_objective {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_bits().to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(&record.cost_units.to_le_bytes());
        hasher.update(&record.cumulative_cost_units.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[derive(Debug, Clone, PartialEq)]
struct StrategySummary {
    design: SamplingDesign,
    run_count: usize,
    median_simple_regret: Option<f64>,
    target_hit_rate: f64,
    median_cost_to_target_when_reached: Option<f64>,
    mean_cost_units_per_run: f64,
}

fn median_f64(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

/// Run all fixed-budget designs against the same ordered seed set for one case.
/// This pairs stochastic designs by seed; it does not imply that different designs
/// visit identical query points.
fn compare_designs(
    case: SyntheticCase,
    fidelity: Fidelity,
    budget: usize,
    seeds: &[u64],
) -> Result<Vec<StrategySummary>, BenchmarkError> {
    if seeds.is_empty() {
        return Err(BenchmarkError::EmptySeedSet);
    }

    let mut summaries = Vec::with_capacity(SamplingDesign::ALL.len());
    for design in SamplingDesign::ALL {
        let mut runs = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            runs.push(run_benchmark(case, design, fidelity, seed, budget)?);
        }

        let regrets = runs.iter().filter_map(|run| run.simple_regret).collect();
        let reached_costs = runs
            .iter()
            .filter_map(|run| run.cost_to_target.map(|cost| cost as f64))
            .collect();
        let target_hits = runs.iter().filter(|run| run.cost_to_target.is_some()).count();
        let total_cost: u64 = runs.iter().map(|run| run.total_cost_units).sum();
        let mean_cost_units_per_run = total_cost as f64 / runs.len() as f64;

        summaries.push(StrategySummary {
            design,
            run_count: runs.len(),
            median_simple_regret: median_f64(regrets),
            target_hit_rate: target_hits as f64 / runs.len() as f64,
            median_cost_to_target_when_reached: median_f64(reached_costs),
            mean_cost_units_per_run,
        });
    }

    Ok(summaries)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGET: usize = 16;
    const PAIRED_SEEDS: u64 = 32;

    #[test]
    fn all_synthetic_fixture_optima_are_known_answers() {
        for case in SyntheticCase::ALL {
            let optimum = case.optimum();
            assert_abs_close(case.objective(optimum), 0.0, 1e-15);
            assert_abs_close(case.known_optimum_value(), 0.0, 1e-15);
            assert!((0.0..1.0).contains(&optimum.x));
            assert!((0.0..1.0).contains(&optimum.y));
            assert!(case.is_feasible(optimum));
            assert!(!case.can_fail(optimum));
            assert!(!case.name().is_empty());
        }
    }

    #[test]
    fn paired_designs_use_the_same_query_budget_and_charge_every_attempt() {
        for case in SyntheticCase::ALL {
            for seed in 0..PAIRED_SEEDS {
                for design in SamplingDesign::ALL {
                    let run =
                        run_benchmark(case, design, Fidelity::High, seed, BUDGET).unwrap();
                    assert_eq!(run.transcript.len(), BUDGET);
                    assert_eq!(
                        run.total_cost_units,
                        BUDGET as u64 * HIGH_FIDELITY_COST_UNITS
                    );
                    assert_eq!(
                        run.feasible_count + run.infeasible_count + run.failed_count,
                        BUDGET
                    );
                    assert!(run.transcript.windows(2).all(|pair| {
                        pair[0].cumulative_cost_units < pair[1].cumulative_cost_units
                    }));
                    if let Some(regret) = run.simple_regret {
                        assert!(regret.is_finite() && regret >= 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_seed_replays_identical_records_digest_and_metrics() {
        for case in SyntheticCase::ALL {
            for design in SamplingDesign::ALL {
                let left =
                    run_benchmark(case, design, Fidelity::High, 0xC457_1A6, BUDGET).unwrap();
                let right =
                    run_benchmark(case, design, Fidelity::High, 0xC457_1A6, BUDGET).unwrap();
                assert_eq!(left, right);
            }
        }
    }

    #[test]
    fn new_seed_changes_stochastic_design_transcript() {
        let left = run_benchmark(
            SyntheticCase::SmoothBowl,
            SamplingDesign::Random,
            Fidelity::High,
            17,
            BUDGET,
        )
        .unwrap();
        let right = run_benchmark(
            SyntheticCase::SmoothBowl,
            SamplingDesign::Random,
            Fidelity::High,
            18,
            BUDGET,
        )
        .unwrap();
        assert_ne!(left.transcript_digest_blake3, right.transcript_digest_blake3);
    }

    #[test]
    fn adaptive_selector_uses_observed_values_and_never_repeats_candidates() {
        let seed = run_benchmark(
            SyntheticCase::SmoothBowl,
            SamplingDesign::LatinHypercube,
            Fidelity::High,
            0xDEAD_BEEF,
            4,
        )
        .unwrap();
        let next_a = choose_adaptive_point(&seed.transcript).unwrap();

        let mut changed_evaluation_truth = seed.transcript.clone();
        for record in &mut changed_evaluation_truth {
            record.true_objective_for_evaluation_only += 1000.0;
        }
        let next_b = choose_adaptive_point(&changed_evaluation_truth).unwrap();
        assert_eq!(next_a, next_b);
        assert!(!seed.transcript.iter().any(|record| record.point == next_a));

        let adaptive = run_benchmark(
            SyntheticCase::SmoothBowl,
            SamplingDesign::AdaptiveNearestNeighbor,
            Fidelity::High,
            0xDEAD_BEEF,
            BUDGET,
        )
        .unwrap();
        assert_eq!(adaptive.transcript.len(), BUDGET);
        let distinct_points = adaptive
            .transcript
            .iter()
            .map(|record| (record.point.x.to_bits(), record.point.y.to_bits()))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(distinct_points.len(), BUDGET);
    }

    #[test]
    fn adaptive_strategy_replays_and_is_included_in_paired_comparison() {
        let left = run_benchmark(
            SyntheticCase::CoupledFactors,
            SamplingDesign::AdaptiveNearestNeighbor,
            Fidelity::High,
            0xBEEF,
            BUDGET,
        )
        .unwrap();
        let right = run_benchmark(
            SyntheticCase::CoupledFactors,
            SamplingDesign::AdaptiveNearestNeighbor,
            Fidelity::High,
            0xBEEF,
            BUDGET,
        )
        .unwrap();
        assert_eq!(left, right);

        let summary = compare_designs(
            SyntheticCase::CoupledFactors,
            Fidelity::High,
            BUDGET,
            &[1, 2, 3, 4],
        )
        .unwrap();
        assert_eq!(summary.len(), SamplingDesign::ALL.len());
        assert!(summary
            .iter()
            .any(|row| row.design == SamplingDesign::AdaptiveNearestNeighbor));
    }

    #[test]
    fn latin_hypercube_covers_each_stratum_in_each_dimension_once() {
        let points = sample_points(SamplingDesign::LatinHypercube, BUDGET, 0x1A2B_3C4D).unwrap();
        let mut x_strata: Vec<usize> = points
            .iter()
            .map(|point| (point.x * BUDGET as f64).floor() as usize)
            .collect();
        let mut y_strata: Vec<usize> = points
            .iter()
            .map(|point| (point.y * BUDGET as f64).floor() as usize)
            .collect();
        x_strata.sort_unstable();
        y_strata.sort_unstable();

        assert_eq!(x_strata, (0..BUDGET).collect::<Vec<_>>());
        assert_eq!(y_strata, (0..BUDGET).collect::<Vec<_>>());
    }

    #[test]
    fn infeasible_and_failed_queries_are_visible_and_still_cost_resources() {
        let island_outside = Point { x: 0.1, y: 0.9 };
        let island_record = evaluate_trial(
            SyntheticCase::FeasibleIsland,
            island_outside,
            Fidelity::High,
            9,
            HIGH_FIDELITY_COST_UNITS,
        );
        assert_eq!(island_record.status, TrialStatus::Infeasible);
        assert_eq!(island_record.observed_objective, None);
        assert_eq!(island_record.cost_units, HIGH_FIDELITY_COST_UNITS);

        let fail_point = Point { x: 0.95, y: 0.95 };
        let failure_record = evaluate_trial(
            SyntheticCase::FailedCorner,
            fail_point,
            Fidelity::High,
            9,
            HIGH_FIDELITY_COST_UNITS,
        );
        assert_eq!(failure_record.status, TrialStatus::Failed);
        assert_eq!(failure_record.observed_objective, None);
        assert_eq!(failure_record.cost_units, HIGH_FIDELITY_COST_UNITS);
    }

    #[test]
    fn low_and_high_fidelity_are_separately_charged_and_disagree_by_design() {
        let point = Point { x: 0.8, y: 0.1 };
        let low = evaluate_trial(
            SyntheticCase::FidelityDiscrepancy,
            point,
            Fidelity::Low,
            1,
            LOW_FIDELITY_COST_UNITS,
        );
        let high = evaluate_trial(
            SyntheticCase::FidelityDiscrepancy,
            point,
            Fidelity::High,
            1,
            HIGH_FIDELITY_COST_UNITS,
        );

        assert_eq!(low.cost_units, 1);
        assert_eq!(high.cost_units, 10);
        assert_abs_close(
            low.observed_objective.unwrap() - high.observed_objective.unwrap(),
            0.155,
            1e-12,
        );
        assert_abs_close(
            high.observed_objective.unwrap(),
            SyntheticCase::FidelityDiscrepancy.objective(point),
            1e-15,
        );
    }

    #[test]
    fn heteroscedastic_noise_replays_for_same_seed_and_varies_across_seeds() {
        let point = Point { x: 0.85, y: 0.14 };
        let a = SyntheticCase::HeteroscedasticNoise.observation(point, Fidelity::High, 1234);
        let b = SyntheticCase::HeteroscedasticNoise.observation(point, Fidelity::High, 1234);
        let c = SyntheticCase::HeteroscedasticNoise.observation(point, Fidelity::High, 1235);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.is_finite() && c.is_finite());
    }

    #[test]
    fn paired_seed_summary_retains_failures_and_exposes_target_hit_rate() {
        let seeds: Vec<u64> = (0..12).collect();
        let summaries = compare_designs(
            SyntheticCase::FeasibleIsland,
            Fidelity::High,
            BUDGET,
            &seeds,
        )
        .unwrap();

        assert_eq!(summaries.len(), SamplingDesign::ALL.len());
        for summary in summaries {
            assert_eq!(summary.run_count, seeds.len());
            assert!((0.0..=1.0).contains(&summary.target_hit_rate));
            assert_abs_close(
                summary.mean_cost_units_per_run,
                (BUDGET as u64 * HIGH_FIDELITY_COST_UNITS) as f64,
                0.0,
            );
            if summary.target_hit_rate > 0.0 {
                assert!(summary.median_cost_to_target_when_reached.is_some());
            }
        }
    }

    #[test]
    fn paired_summary_rejects_empty_seed_set() {
        assert_eq!(
            compare_designs(
                SyntheticCase::SmoothBowl,
                Fidelity::High,
                BUDGET,
                &[],
            ),
            Err(BenchmarkError::EmptySeedSet)
        );
    }

    #[test]
    fn zero_budget_fails_closed() {
        assert_eq!(
            sample_points(SamplingDesign::Random, 0, 1),
            Err(BenchmarkError::ZeroBudget)
        );
    }

    fn assert_abs_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.15e} differs from expected {expected:.15e} by more than {tolerance:.3e}"
        );
    }
}
