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
    RastriginMultimodal = 6,
}

impl SyntheticCase {
    const ALL: [Self; 7] = [
        Self::SmoothBowl,
        Self::CoupledFactors,
        Self::FeasibleIsland,
        Self::HeteroscedasticNoise,
        Self::FidelityDiscrepancy,
        Self::FailedCorner,
        Self::RastriginMultimodal,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::SmoothBowl => "smooth-bowl",
            Self::CoupledFactors => "coupled-factors",
            Self::FeasibleIsland => "feasible-island",
            Self::HeteroscedasticNoise => "heteroscedastic-noise",
            Self::FidelityDiscrepancy => "fidelity-discrepancy",
            Self::FailedCorner => "failed-corner",
            Self::RastriginMultimodal => "rastrigin-multimodal",
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
            Self::RastriginMultimodal => Point { x: 0.5, y: 0.5 },
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
            Self::RastriginMultimodal => {
                // Map normalized coordinates [0,1]^2 to the standard [-5.12, 5.12]^2
                // Rastrigin domain. Its known global minimum is exactly 0 at (0,0),
                // which maps to normalized point (0.5, 0.5); numerous local minima
                // make this a useful non-convex acquisition stress fixture.
                let x = 10.24 * (point.x - 0.5);
                let y = 10.24 * (point.y - 0.5);
                20.0
                    + x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos()
                    + y * y - 10.0 * (2.0 * std::f64::consts::PI * y).cos()
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
    GaussianProcessLcb = 4,
}

impl SamplingDesign {
    const ALL: [Self; 5] = [
        Self::Random,
        Self::UniformGrid,
        Self::LatinHypercube,
        Self::AdaptiveNearestNeighbor,
        Self::GaussianProcessLcb,
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

/// Restricted view that optimization policies are allowed to inspect.
/// This deliberately excludes evaluator-only ground truth and hidden known optima.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ProposalObservation {
    point: Point,
    status: TrialStatus,
    observed_objective: Option<f64>,
}

fn proposal_history(records: &[TrialRecord]) -> Vec<ProposalObservation> {
    records
        .iter()
        .map(|record| ProposalObservation {
            point: record.point,
            status: record.status,
            observed_objective: record.observed_objective,
        })
        .collect()
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
enum GaussianProcessError {
    EmptyHistory,
    NonFiniteInput,
    InvalidMatrix,
    NotPositiveDefinite,
    NonFinitePosterior,
    NegativePosteriorVariance,
}

impl fmt::Display for GaussianProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHistory => f.write_str("Gaussian-process history has no usable observations"),
            Self::NonFiniteInput => f.write_str("Gaussian-process input contains a non-finite value"),
            Self::InvalidMatrix => f.write_str("Gaussian-process covariance matrix is malformed"),
            Self::NotPositiveDefinite => f.write_str("Gaussian-process covariance is not positive definite"),
            Self::NonFinitePosterior => f.write_str("Gaussian-process posterior is non-finite"),
            Self::NegativePosteriorVariance => f.write_str("Gaussian-process posterior variance is materially negative"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkError {
    ZeroBudget,
    EmptySeedSet,
    AdaptiveRequiresHistory,
    AdaptiveCandidatePoolExhausted,
    GaussianProcess(GaussianProcessError),
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
            Self::AdaptiveCandidatePoolExhausted => {
                f.write_str("adaptive candidate lattice has no unqueried candidate left")
            }
            Self::GaussianProcess(error) => write!(f, "Gaussian-process baseline failed: {error}"),
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
        SamplingDesign::AdaptiveNearestNeighbor | SamplingDesign::GaussianProcessLcb => {
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

/// Select the next point using only the previously observed values.
///
/// This intentionally simple adaptive baseline is not Bayesian optimization: it uses
/// a nearest-observation surrogate plus a fixed distance-based exploration bonus over
/// a finite candidate lattice. It does not fit a Gaussian process or calibrated
/// posterior. Evaluation-only objective truth is not read by this selection function.
fn choose_adaptive_point(history: &[ProposalObservation]) -> Option<Point> {
    const SIDE: usize = 16;
    const EXPLORATION_WEIGHT: f64 = 0.20;

    let measured: Vec<&ProposalObservation> = history
        .iter()
        .filter(|record| record.status == TrialStatus::Feasible)
        .filter(|record| record.observed_objective.is_some())
        .collect();

    let mut best: Option<(f64, Point)> = None;
    for ix in 0..SIDE {
        for iy in 0..SIDE {
            let candidate = Point {
                x: (ix as f64 + 0.5) / SIDE as f64,
                y: (iy as f64 + 0.5) / SIDE as f64,
            };
            if history.iter().any(|record| {
                let dx = record.point.x - candidate.x;
                let dy = record.point.y - candidate.y;
                dx * dx + dy * dy < 1e-20
            }) {
                continue;
            }

            let mut nearest_any_distance_sq = f64::INFINITY;
            for record in history {
                let dx = record.point.x - candidate.x;
                let dy = record.point.y - candidate.y;
                nearest_any_distance_sq =
                    nearest_any_distance_sq.min(dx * dx + dy * dy);
            }

            let mut nearest_measured_distance_sq = f64::INFINITY;
            let mut nearest_observed: Option<f64> = None;
            for record in &measured {
                let dx = record.point.x - candidate.x;
                let dy = record.point.y - candidate.y;
                let distance_sq = dx * dx + dy * dy;
                if distance_sq < nearest_measured_distance_sq {
                    nearest_measured_distance_sq = distance_sq;
                    nearest_observed = record.observed_objective;
                }
            }

            let novelty = nearest_any_distance_sq.sqrt();
            // If every initial trial failed or was infeasible, continue space-filling
            // exploration rather than aborting. Otherwise balance local value and novelty.
            let acquisition = match nearest_observed {
                Some(predicted) => predicted - EXPLORATION_WEIGHT * novelty,
                None => -novelty,
            };
            if best.map_or(true, |(best_score, _)| acquisition < best_score) {
                best = Some((acquisition, candidate));
            }
        }
    }

    best.map(|(_, point)| point)
}

const GP_LENGTH_SCALE: f64 = 0.25;
const GP_SIGNAL_VARIANCE: f64 = 1.0;
const GP_OBSERVATION_NOISE_VARIANCE: f64 = 1e-6;
const GP_DIAGONAL_JITTER: f64 = 1e-10;
const GP_LCB_EXPLORATION_BETA: f64 = 1.5;

fn gp_rbf_kernel(left: Point, right: Point) -> f64 {
    let dx = left.x - right.x;
    let dy = left.y - right.y;
    let squared_distance = dx * dx + dy * dy;
    GP_SIGNAL_VARIANCE * (-0.5 * squared_distance / GP_LENGTH_SCALE.powi(2)).exp()
}

/// Cholesky factorization of a symmetric positive-definite matrix.
/// The input must already include observation noise and fixed diagonal jitter.
fn gp_cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GaussianProcessError> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err(GaussianProcessError::InvalidMatrix);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(GaussianProcessError::NonFiniteInput);
    }
    for row in 0..n {
        for column in 0..row {
            if (matrix[row][column] - matrix[column][row]).abs() > 1e-12 {
                return Err(GaussianProcessError::InvalidMatrix);
            }
        }
    }

    let mut lower = vec![vec![0.0; n]; n];
    for row in 0..n {
        for column in 0..=row {
            let mut value = matrix[row][column];
            for k in 0..column {
                value -= lower[row][k] * lower[column][k];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(GaussianProcessError::NotPositiveDefinite);
                }
                lower[row][column] = value.sqrt();
            } else {
                let pivot = lower[column][column];
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(GaussianProcessError::NotPositiveDefinite);
                }
                lower[row][column] = value / pivot;
                if !lower[row][column].is_finite() {
                    return Err(GaussianProcessError::NonFinitePosterior);
                }
            }
        }
    }
    Ok(lower)
}

fn gp_solve_lower(
    lower: &[Vec<f64>],
    right_hand_side: &[f64],
) -> Result<Vec<f64>, GaussianProcessError> {
    let n = lower.len();
    if n == 0 || right_hand_side.len() != n || lower.iter().any(|row| row.len() != n) {
        return Err(GaussianProcessError::InvalidMatrix);
    }
    let mut result = vec![0.0; n];
    for row in 0..n {
        let mut value = right_hand_side[row];
        for column in 0..row {
            value -= lower[row][column] * result[column];
        }
        let pivot = lower[row][row];
        if !pivot.is_finite() || pivot <= 0.0 {
            return Err(GaussianProcessError::NotPositiveDefinite);
        }
        result[row] = value / pivot;
        if !result[row].is_finite() {
            return Err(GaussianProcessError::NonFinitePosterior);
        }
    }
    Ok(result)
}

fn gp_solve_upper_from_lower_transpose(
    lower: &[Vec<f64>],
    right_hand_side: &[f64],
) -> Result<Vec<f64>, GaussianProcessError> {
    let n = lower.len();
    if n == 0 || right_hand_side.len() != n || lower.iter().any(|row| row.len() != n) {
        return Err(GaussianProcessError::InvalidMatrix);
    }
    let mut result = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = right_hand_side[row];
        for column in row + 1..n {
            value -= lower[column][row] * result[column];
        }
        let pivot = lower[row][row];
        if !pivot.is_finite() || pivot <= 0.0 {
            return Err(GaussianProcessError::NotPositiveDefinite);
        }
        result[row] = value / pivot;
        if !result[row].is_finite() {
            return Err(GaussianProcessError::NonFinitePosterior);
        }
    }
    Ok(result)
}

/// Fixed-hyperparameter Gaussian-process posterior for the currently observed data.
/// The GP is fit once per proposal; candidate posterior queries reuse the Cholesky
/// factor and alpha vector rather than refactoring per candidate.
#[derive(Debug, Clone, PartialEq)]
struct GaussianProcessPosterior {
    observed_points: Vec<Point>,
    mean_y: f64,
    alpha: Vec<f64>,
    cholesky_lower: Vec<Vec<f64>>,
}

impl GaussianProcessPosterior {
    fn fit(history: &[ProposalObservation]) -> Result<Self, GaussianProcessError> {
        let measured: Vec<(Point, f64)> = history
            .iter()
            .filter(|entry| entry.status == TrialStatus::Feasible)
            .filter_map(|entry| entry.observed_objective.map(|value| (entry.point, value)))
            .collect();
        if measured.is_empty() {
            return Err(GaussianProcessError::EmptyHistory);
        }
        if measured.iter().any(|(point, value)| {
            !point.x.is_finite()
                || !point.y.is_finite()
                || !value.is_finite()
                || !(0.0..=1.0).contains(&point.x)
                || !(0.0..=1.0).contains(&point.y)
        }) {
            return Err(GaussianProcessError::NonFiniteInput);
        }

        let n = measured.len();
        let mean_y = measured.iter().map(|(_, value)| *value).sum::<f64>() / n as f64;
        if !mean_y.is_finite() {
            return Err(GaussianProcessError::NonFiniteInput);
        }
        let observed_points: Vec<Point> = measured.iter().map(|(point, _)| *point).collect();
        let centered_y: Vec<f64> = measured.iter().map(|(_, value)| *value - mean_y).collect();

        let mut covariance = vec![vec![0.0; n]; n];
        for row in 0..n {
            for column in 0..=row {
                let value = gp_rbf_kernel(observed_points[row], observed_points[column]);
                covariance[row][column] = value;
                covariance[column][row] = value;
            }
            covariance[row][row] +=
                GP_OBSERVATION_NOISE_VARIANCE + GP_DIAGONAL_JITTER;
        }

        let cholesky_lower = gp_cholesky(&covariance)?;
        let projected = gp_solve_lower(&cholesky_lower, &centered_y)?;
        let alpha = gp_solve_upper_from_lower_transpose(&cholesky_lower, &projected)?;

        Ok(Self {
            observed_points,
            mean_y,
            alpha,
            cholesky_lower,
        })
    }

    fn predict(&self, candidate: Point) -> Result<(f64, f64), GaussianProcessError> {
        if !candidate.x.is_finite()
            || !candidate.y.is_finite()
            || !(0.0..=1.0).contains(&candidate.x)
            || !(0.0..=1.0).contains(&candidate.y)
        {
            return Err(GaussianProcessError::NonFiniteInput);
        }

        let cross_covariance: Vec<f64> = self
            .observed_points
            .iter()
            .map(|point| gp_rbf_kernel(*point, candidate))
            .collect();
        let mean = self.mean_y
            + cross_covariance
                .iter()
                .zip(&self.alpha)
                .map(|(kernel, weight)| kernel * weight)
                .sum::<f64>();
        let projected = gp_solve_lower(&self.cholesky_lower, &cross_covariance)?;
        let variance_reduction = projected.iter().map(|value| value * value).sum::<f64>();
        let raw_variance = GP_SIGNAL_VARIANCE - variance_reduction;
        let variance = if raw_variance < 0.0 && raw_variance >= -1e-8 {
            0.0
        } else if raw_variance < -1e-8 {
            return Err(GaussianProcessError::NegativePosteriorVariance);
        } else {
            raw_variance
        };

        if !mean.is_finite() || !variance.is_finite() || variance < 0.0 {
            return Err(GaussianProcessError::NonFinitePosterior);
        }
        Ok((mean, variance))
    }
}

/// Select the next candidate by a fixed lower-confidence-bound acquisition:
/// LCB(x) = posterior_mean(x) - beta * posterior_stddev(x). Lower is preferred.
/// Kernel, noise, jitter and beta are frozen constants; no hidden truth or unqueried
/// objective values are inputs. This is a minimal GP-BO baseline, not a tuned SOTA claim.
fn choose_gaussian_process_lcb(
    history: &[ProposalObservation],
) -> Result<Option<Point>, GaussianProcessError> {
    let posterior = match GaussianProcessPosterior::fit(history) {
        Ok(posterior) => posterior,
        Err(GaussianProcessError::EmptyHistory) => return Ok(None),
        Err(error) => return Err(error),
    };

    const SIDE: usize = 16;
    let mut best: Option<(f64, Point)> = None;
    for ix in 0..SIDE {
        for iy in 0..SIDE {
            let candidate = Point {
                x: (ix as f64 + 0.5) / SIDE as f64,
                y: (iy as f64 + 0.5) / SIDE as f64,
            };
            if history.iter().any(|entry| {
                let dx = entry.point.x - candidate.x;
                let dy = entry.point.y - candidate.y;
                dx * dx + dy * dy < 1e-20
            }) {
                continue;
            }

            let (mean, variance) = posterior.predict(candidate)?;
            let acquisition = mean - GP_LCB_EXPLORATION_BETA * variance.sqrt();
            if !acquisition.is_finite() {
                return Err(GaussianProcessError::NonFinitePosterior);
            }
            if best.map_or(true, |(best_score, _)| acquisition < best_score) {
                best = Some((acquisition, candidate));
            }
        }
    }
    Ok(best.map(|(_, point)| point))
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
    let is_sequential = matches!(
        design,
        SamplingDesign::AdaptiveNearestNeighbor | SamplingDesign::GaussianProcessLcb
    );
    let fixed_points = if is_sequential {
        Vec::new()
    } else {
        sample_points(design, budget, seed)?
    };
    let adaptive_warm_start = if is_sequential {
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
    let mut best_true_objective_among_feasible_queries: Option<f64> = None;

    for query_index in 0..budget {
        let point = if is_sequential {
            if query_index < adaptive_warm_start.len() {
                adaptive_warm_start[query_index]
            } else {
                let history = proposal_history(&records);
                if design == SamplingDesign::AdaptiveNearestNeighbor {
                    choose_adaptive_point(&history)
                        .ok_or(BenchmarkError::AdaptiveCandidatePoolExhausted)?
                } else {
                    match choose_gaussian_process_lcb(&history)
                        .map_err(BenchmarkError::GaussianProcess)?
                    {
                        Some(point) => point,
                        None => choose_adaptive_point(&history)
                            .ok_or(BenchmarkError::AdaptiveCandidatePoolExhausted)?,
                    }
                }
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
                // The synthetic evaluator confirms target achievement against
                // noise-free truth only after this query; the strategy sees observations.
                if cost_to_target.is_none()
                    && record.true_objective_for_evaluation_only <= SYNTHETIC_TARGET_OBJECTIVE
                {
                    cost_to_target = Some(record.cumulative_cost_units);
                }
                if best_observed_objective.map_or(true, |best| observed < best) {
                    best_observed_objective = Some(observed);
                }
                if best_true_objective_among_feasible_queries
                    .map_or(true, |best| record.true_objective_for_evaluation_only < best)
                {
                    best_true_objective_among_feasible_queries =
                        Some(record.true_objective_for_evaluation_only);
                }
            }
            TrialStatus::Infeasible => infeasible_count += 1,
            TrialStatus::Failed => failed_count += 1,
        }
        records.push(record);
    }

    // Simple regret uses the best evaluator-known truth among every feasible point
    // actually queried, not merely the candidate with the best noisy observation.
    let simple_regret = best_true_objective_among_feasible_queries
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
        let next_a = choose_adaptive_point(&proposal_history(&seed.transcript)).unwrap();

        let mut changed_evaluation_truth = seed.transcript.clone();
        for record in &mut changed_evaluation_truth {
            record.true_objective_for_evaluation_only += 1000.0;
        }
        let next_b = choose_adaptive_point(&proposal_history(&changed_evaluation_truth)).unwrap();
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
    fn adaptive_candidate_lattice_exhaustion_is_explicit() {
        let mut records = Vec::new();
        let mut cost = 0_u64;
        for ix in 0..16 {
            for iy in 0..16 {
                cost += HIGH_FIDELITY_COST_UNITS;
                let point = Point {
                    x: (ix as f64 + 0.5) / 16.0,
                    y: (iy as f64 + 0.5) / 16.0,
                };
                records.push(TrialRecord {
                    point,
                    status: TrialStatus::Feasible,
                    fidelity: Fidelity::High,
                    true_objective_for_evaluation_only: 1.0,
                    observed_objective: Some(1.0),
                    cost_units: HIGH_FIDELITY_COST_UNITS,
                    cumulative_cost_units: cost,
                });
            }
        }
        assert_eq!(choose_adaptive_point(&proposal_history(&records)), None);
    }

    #[test]
    fn adaptive_strategy_keeps_exploring_when_warm_start_has_no_usable_observation() {
        let points = sample_points(SamplingDesign::LatinHypercube, 4, 99).unwrap();
        let records: Vec<TrialRecord> = points
            .iter()
            .enumerate()
            .map(|(index, point)| TrialRecord {
                point: *point,
                status: TrialStatus::Infeasible,
                fidelity: Fidelity::High,
                true_objective_for_evaluation_only: 1.0,
                observed_objective: None,
                cost_units: HIGH_FIDELITY_COST_UNITS,
                cumulative_cost_units: (index as u64 + 1) * HIGH_FIDELITY_COST_UNITS,
            })
            .collect();

        let next = choose_adaptive_point(&proposal_history(&records)).unwrap();
        assert!(!records.iter().any(|record| record.point == next));
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
    fn rastrigin_multimodal_fixture_has_known_global_minimum() {
        let case = SyntheticCase::RastriginMultimodal;
        assert_abs_close(case.objective(Point { x: 0.5, y: 0.5 }), 0.0, 1e-12);
        assert!(case.objective(Point { x: 0.6, y: 0.5 }) > 0.0);
        assert_abs_close(case.known_optimum_value(), 0.0, 1e-12);
    }

    #[test]
    fn gp_kernel_and_cholesky_match_independent_known_answers() {
        let origin = Point { x: 0.0, y: 0.0 };
        let one_length_scale_away = Point {
            x: GP_LENGTH_SCALE,
            y: 0.0,
        };
        assert_abs_close(gp_rbf_kernel(origin, origin), 1.0, 1e-15);
        assert_abs_close(
            gp_rbf_kernel(origin, one_length_scale_away),
            (-0.5_f64).exp(),
            1e-15,
        );

        let lower = gp_cholesky(&[vec![4.0, 2.0], vec![2.0, 3.0]]).unwrap();
        assert_abs_close(lower[0][0], 2.0, 1e-15);
        assert_abs_close(lower[0][1], 0.0, 1e-15);
        assert_abs_close(lower[1][0], 1.0, 1e-15);
        assert_abs_close(lower[1][1], 2.0_f64.sqrt(), 1e-15);

        let solved = gp_solve_lower(&lower, &[2.0, 1.0]).unwrap();
        assert_abs_close(solved[0], 1.0, 1e-15);
        assert_abs_close(solved[1], 0.0, 1e-15);
        let back_solved = gp_solve_upper_from_lower_transpose(&lower, &solved).unwrap();
        assert_abs_close(back_solved[0], 0.5, 1e-15);
        assert_abs_close(back_solved[1], 0.0, 1e-15);
    }

    #[test]
    fn gp_posterior_matches_one_observation_known_answer() {
        let point = Point { x: 0.25, y: 0.4 };
        let posterior = GaussianProcessPosterior::fit(&[ProposalObservation {
            point,
            status: TrialStatus::Feasible,
            observed_objective: Some(2.0),
        }])
        .unwrap();

        let (mean, variance) = posterior.predict(point).unwrap();
        assert_abs_close(mean, 2.0, 1e-12);
        assert_abs_close(
            variance,
            GP_OBSERVATION_NOISE_VARIANCE + GP_DIAGONAL_JITTER,
            2e-10,
        );
    }

    #[test]
    fn gp_acquisition_is_deterministic_and_cannot_read_evaluation_truth() {
        let initial = run_benchmark(
            SyntheticCase::SmoothBowl,
            SamplingDesign::LatinHypercube,
            Fidelity::High,
            0xB0_71_A7,
            4,
        )
        .unwrap();
        let history = proposal_history(&initial.transcript);
        let next_a = choose_gaussian_process_lcb(&history).unwrap().unwrap();

        let mut changed_truth = initial.transcript.clone();
        for record in &mut changed_truth {
            record.true_objective_for_evaluation_only -= 123.0;
        }
        let next_b = choose_gaussian_process_lcb(&proposal_history(&changed_truth))
            .unwrap()
            .unwrap();
        assert_eq!(next_a, next_b);
        assert!(!initial.transcript.iter().any(|record| record.point == next_a));

        let run_a = run_benchmark(
            SyntheticCase::SmoothBowl,
            SamplingDesign::GaussianProcessLcb,
            Fidelity::High,
            0xB0_71_A7,
            BUDGET,
        )
        .unwrap();
        let run_b = run_benchmark(
            SyntheticCase::SmoothBowl,
            SamplingDesign::GaussianProcessLcb,
            Fidelity::High,
            0xB0_71_A7,
            BUDGET,
        )
        .unwrap();
        assert_eq!(run_a, run_b);
        let points = run_a
            .transcript
            .iter()
            .map(|record| (record.point.x.to_bits(), record.point.y.to_bits()))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(points.len(), BUDGET);
    }

    #[test]
    fn gp_covariance_jitter_handles_duplicate_observation_locations() {
        let point = Point { x: 0.5, y: 0.5 };
        let posterior = GaussianProcessPosterior::fit(&[
            ProposalObservation {
                point,
                status: TrialStatus::Feasible,
                observed_objective: Some(0.1),
            },
            ProposalObservation {
                point,
                status: TrialStatus::Feasible,
                observed_objective: Some(0.11),
            },
        ])
        .unwrap();
        let (mean, variance) = posterior.predict(point).unwrap();
        assert!(mean.is_finite());
        assert!(variance.is_finite() && variance >= 0.0);
    }

    #[test]
    fn gp_rejects_non_finite_observations_and_non_positive_covariance() {
        let result = GaussianProcessPosterior::fit(&[ProposalObservation {
            point: Point { x: 0.1, y: 0.2 },
            status: TrialStatus::Feasible,
            observed_objective: Some(f64::NAN),
        }]);
        assert_eq!(result, Err(GaussianProcessError::NonFiniteInput));

        assert_eq!(
            gp_cholesky(&[vec![1.0, 2.0], vec![2.0, 1.0]]),
            Err(GaussianProcessError::NotPositiveDefinite)
        );
        assert_eq!(
            gp_cholesky(&[vec![1.0, 0.3], vec![0.4, 1.0]]),
            Err(GaussianProcessError::InvalidMatrix)
        );
        assert_eq!(
            gp_cholesky(&[vec![f64::INFINITY]]),
            Err(GaussianProcessError::NonFiniteInput)
        );
        assert_eq!(
            gp_cholesky(&[]),
            Err(GaussianProcessError::InvalidMatrix)
        );
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
    fn sequential_methods_cannot_be_invoked_as_fixed_designs() {
        assert_eq!(
            sample_points(SamplingDesign::AdaptiveNearestNeighbor, BUDGET, 1),
            Err(BenchmarkError::AdaptiveRequiresHistory)
        );
        assert_eq!(
            sample_points(SamplingDesign::GaussianProcessLcb, BUDGET, 1),
            Err(BenchmarkError::AdaptiveRequiresHistory)
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
