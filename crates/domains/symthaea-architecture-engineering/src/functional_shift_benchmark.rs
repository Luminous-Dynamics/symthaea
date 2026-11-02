//! Frozen categorical shift fixtures for the AR-5B24 decision-functional benchmark.
//!
//! The harness compares one-sided simultaneous Hoeffding bounds for four
//! predeclared decision functionals with the Weissman finite-alphabet TV radius.
//! It includes an omitted fifth functional to expose the deliberate scope limit
//! of DFCE: coverage of the declared vector is not full-distribution coverage.
//!
//! The PRNG is a deterministic benchmark generator, not a cryptographic RNG.
//! Simulation results are formula-level evidence, not a theorem or Rust-SOTA claim.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt};

const REQUEST_SCHEMA: &str = "symthaea.architecture-engineering.functional-shift-benchmark-request";
const REQUEST_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.functional-shift-benchmark-report";
const REPORT_VERSION: &str = "0.1.0";
const SUPPORT_SIZE: usize = 8;
const FUNCTIONAL_COUNT: usize = 4;
const Z_975: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionalShiftBenchmarkError {
    pub path: String,
    pub message: String,
}

impl FunctionalShiftBenchmarkError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for FunctionalShiftBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for FunctionalShiftBenchmarkError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionalShiftScenario {
    OrthogonalToDeclaredFunctionals,
    DecisionAligned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionalDirection {
    Upper,
    Lower,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionalShiftFixture {
    pub scenario: FunctionalShiftScenario,
    pub nominal_probabilities: [f64; SUPPORT_SIZE],
    pub target_probabilities: [f64; SUPPORT_SIZE],
    pub functional_masks: [[bool; SUPPORT_SIZE]; FUNCTIONAL_COUNT],
    pub functional_directions: [FunctionalDirection; FUNCTIONAL_COUNT],
    pub omitted_function_mask: [bool; SUPPORT_SIZE],
}

impl FunctionalShiftFixture {
    pub fn new(scenario: FunctionalShiftScenario) -> Self {
        let nominal_probabilities = [0.125; SUPPORT_SIZE];
        let functional_masks = [
            [false, false, true, true, false, false, true, true],
            [false, true, false, true, false, true, false, true],
            [true, true, false, false, true, true, false, false],
            [false, false, false, false, true, true, true, true],
        ];
        let functional_directions = [
            FunctionalDirection::Upper,
            FunctionalDirection::Lower,
            FunctionalDirection::Upper,
            FunctionalDirection::Lower,
        ];
        let omitted_function_mask = [true, false, false, false, false, false, false, false];
        let target_probabilities = match scenario {
            FunctionalShiftScenario::OrthogonalToDeclaredFunctionals => [
                0.20, 0.05, 0.15, 0.10, 0.10, 0.15, 0.05, 0.20,
            ],
            FunctionalShiftScenario::DecisionAligned => [
                0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.15, 0.15,
            ],
        };
        Self {
            scenario,
            nominal_probabilities,
            target_probabilities,
            functional_masks,
            functional_directions,
            omitted_function_mask,
        }
    }

    pub fn nominal_functional_expectations(&self) -> [f64; FUNCTIONAL_COUNT] {
        std::array::from_fn(|index| {
            expectation(&self.nominal_probabilities, &self.functional_masks[index])
        })
    }

    pub fn target_functional_expectations(&self) -> [f64; FUNCTIONAL_COUNT] {
        std::array::from_fn(|index| {
            expectation(&self.target_probabilities, &self.functional_masks[index])
        })
    }

    pub fn nominal_omitted_expectation(&self) -> f64 {
        expectation(&self.nominal_probabilities, &self.omitted_function_mask)
    }

    pub fn target_omitted_expectation(&self) -> f64 {
        expectation(&self.target_probabilities, &self.omitted_function_mask)
    }

    pub fn true_total_variation_distance(&self) -> f64 {
        total_variation(&self.nominal_probabilities, &self.target_probabilities)
    }

    fn validate(&self) -> Result<(), FunctionalShiftBenchmarkError> {
        validate_probability_vector(
            &self.nominal_probabilities,
            "fixture.nominal_probabilities",
        )?;
        validate_probability_vector(
            &self.target_probabilities,
            "fixture.target_probabilities",
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionalShiftBenchmarkRequest {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub scenario: FunctionalShiftScenario,
    pub seed: u64,
    pub replications: usize,
    pub samples_per_replication: usize,
    /// Familywise failure probability for the four one-sided functional bounds.
    pub functional_family_failure_probability: f64,
    /// Failure probability for the single complete-path TV radius.
    pub tv_failure_probability: f64,
}

impl FunctionalShiftBenchmarkRequest {
    pub fn new(
        benchmark_id: impl Into<String>,
        scenario: FunctionalShiftScenario,
        seed: u64,
        replications: usize,
        samples_per_replication: usize,
        functional_family_failure_probability: f64,
        tv_failure_probability: f64,
    ) -> Result<Self, FunctionalShiftBenchmarkError> {
        let request = Self {
            schema: REQUEST_SCHEMA.into(),
            version: REQUEST_VERSION.into(),
            benchmark_id: benchmark_id.into(),
            scenario,
            seed,
            replications,
            samples_per_replication,
            functional_family_failure_probability,
            tv_failure_probability,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), FunctionalShiftBenchmarkError> {
        if self.schema != REQUEST_SCHEMA || self.version != REQUEST_VERSION {
            return Err(FunctionalShiftBenchmarkError::new(
                "request.schema",
                "unsupported functional-shift benchmark schema/version",
            ));
        }
        if self.benchmark_id.trim().is_empty() {
            return Err(FunctionalShiftBenchmarkError::new(
                "request.benchmark_id",
                "benchmark ID must be non-empty",
            ));
        }
        if self.replications == 0 || self.samples_per_replication == 0 {
            return Err(FunctionalShiftBenchmarkError::new(
                "request.sample_sizes",
                "replications and samples per replication must be positive",
            ));
        }
        validate_open_probability(
            self.functional_family_failure_probability,
            "request.functional_family_failure_probability",
        )?;
        validate_open_probability(
            self.tv_failure_probability,
            "request.tv_failure_probability",
        )?;
        Ok(())
    }

    pub fn request_sha256(&self) -> Result<String, FunctionalShiftBenchmarkError> {
        digest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionalShiftBenchmarkReport {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub scenario: FunctionalShiftScenario,
    pub support_size: usize,
    pub nominal_path_probabilities: [f64; SUPPORT_SIZE],
    pub target_path_probabilities: [f64; SUPPORT_SIZE],
    pub functional_masks: [[bool; SUPPORT_SIZE]; FUNCTIONAL_COUNT],
    pub functional_directions: [FunctionalDirection; FUNCTIONAL_COUNT],
    pub omitted_function_mask: [bool; SUPPORT_SIZE],
    pub functional_confidence_scope: String,
    pub tv_confidence_scope: String,
    pub request_sha256: String,
    pub seed: u64,
    pub replications: usize,
    pub samples_per_replication: usize,
    pub functional_family_failure_probability: f64,
    pub tv_failure_probability: f64,
    pub nominal_functional_expectations: [f64; FUNCTIONAL_COUNT],
    pub target_functional_expectations: [f64; FUNCTIONAL_COUNT],
    pub nominal_omitted_functional_expectation: f64,
    pub target_omitted_functional_expectation: f64,
    pub omitted_functional_expectation_shift: f64,
    pub true_total_variation_distance: f64,
    pub scalar_hoeffding_radius: f64,
    pub weissman_tv_concentration_slack: f64,
    pub simultaneous_functional_coverage_count: usize,
    pub simultaneous_functional_coverage_rate: f64,
    pub simultaneous_functional_coverage_wilson_95: [f64; 2],
    pub mean_empirical_tv_distance: f64,
    pub mean_sample_based_tv_radius: f64,
    pub median_sample_based_tv_radius: f64,
    pub p05_sample_based_tv_radius: f64,
    pub p95_sample_based_tv_radius: f64,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    benchmark_id: &'a str,
    scenario: FunctionalShiftScenario,
    support_size: usize,
    nominal_path_probabilities: [f64; SUPPORT_SIZE],
    target_path_probabilities: [f64; SUPPORT_SIZE],
    functional_masks: [[bool; SUPPORT_SIZE]; FUNCTIONAL_COUNT],
    functional_directions: [FunctionalDirection; FUNCTIONAL_COUNT],
    omitted_function_mask: [bool; SUPPORT_SIZE],
    functional_confidence_scope: &'a str,
    tv_confidence_scope: &'a str,
    request_sha256: &'a str,
    seed: u64,
    replications: usize,
    samples_per_replication: usize,
    functional_family_failure_probability: f64,
    tv_failure_probability: f64,
    nominal_functional_expectations: [f64; FUNCTIONAL_COUNT],
    target_functional_expectations: [f64; FUNCTIONAL_COUNT],
    nominal_omitted_functional_expectation: f64,
    target_omitted_functional_expectation: f64,
    omitted_functional_expectation_shift: f64,
    true_total_variation_distance: f64,
    scalar_hoeffding_radius: f64,
    weissman_tv_concentration_slack: f64,
    simultaneous_functional_coverage_count: usize,
    simultaneous_functional_coverage_rate: f64,
    simultaneous_functional_coverage_wilson_95: [f64; 2],
    mean_empirical_tv_distance: f64,
    mean_sample_based_tv_radius: f64,
    median_sample_based_tv_radius: f64,
    p05_sample_based_tv_radius: f64,
    p95_sample_based_tv_radius: f64,
}

fn validate_open_probability(value: f64, path: &str) -> Result<(), FunctionalShiftBenchmarkError> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(FunctionalShiftBenchmarkError::new(
            path,
            "failure probability must be finite and strictly between zero and one",
        ));
    }
    Ok(())
}

fn validate_probability_vector(
    probabilities: &[f64; SUPPORT_SIZE],
    path: &str,
) -> Result<(), FunctionalShiftBenchmarkError> {
    let mut sum = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        if !probability.is_finite() || *probability < 0.0 || *probability > 1.0 {
            return Err(FunctionalShiftBenchmarkError::new(
                format!("{path}[{index}]"),
                "probability must be finite and lie in [0, 1]",
            ));
        }
        sum += probability;
    }
    if (sum - 1.0).abs() > 1e-12 {
        return Err(FunctionalShiftBenchmarkError::new(
            path,
            "probability vector must sum to one",
        ));
    }
    Ok(())
}

fn expectation(probabilities: &[f64; SUPPORT_SIZE], mask: &[bool; SUPPORT_SIZE]) -> f64 {
    probabilities
        .iter()
        .zip(mask)
        .filter_map(|(probability, included)| included.then_some(*probability))
        .sum()
}

fn total_variation(
    left: &[f64; SUPPORT_SIZE],
    right: &[f64; SUPPORT_SIZE],
) -> f64 {
    0.5 * left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f64>()
}

fn hoeffding_radius(
    sample_size: usize,
    family_failure_probability: f64,
) -> Result<f64, FunctionalShiftBenchmarkError> {
    validate_open_probability(family_failure_probability, "confidence.failure_probability")?;
    if sample_size == 0 {
        return Err(FunctionalShiftBenchmarkError::new(
            "confidence.sample_size",
            "sample size must be positive",
        ));
    }
    let per_functional_delta = family_failure_probability / FUNCTIONAL_COUNT as f64;
    let radius =
        ((-per_functional_delta.ln()) / (2.0 * sample_size as f64)).sqrt();
    if radius.is_finite() {
        Ok(radius.min(1.0))
    } else {
        Err(FunctionalShiftBenchmarkError::new(
            "confidence.radius",
            "Hoeffding radius is non-finite",
        ))
    }
}

/// Weissman finite-alphabet confidence slack for TV distance under the conservative 2^K factor.
fn weissman_tv_slack(
    sample_size: usize,
    failure_probability: f64,
) -> Result<f64, FunctionalShiftBenchmarkError> {
    validate_open_probability(failure_probability, "tv.failure_probability")?;
    if sample_size == 0 {
        return Err(FunctionalShiftBenchmarkError::new(
            "tv.sample_size",
            "sample size must be positive",
        ));
    }
    let numerator =
        SUPPORT_SIZE as f64 * std::f64::consts::LN_2 + (1.0 / failure_probability).ln();
    let radius = (numerator / (2.0 * sample_size as f64)).sqrt();
    if radius.is_finite() {
        Ok(radius.min(1.0))
    } else {
        Err(FunctionalShiftBenchmarkError::new(
            "tv.radius",
            "Weissman TV slack is non-finite",
        ))
    }
}

#[derive(Debug, Clone)]
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

    fn next_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn draw_counts(
    rng: &mut SplitMix64,
    sample_size: usize,
    probabilities: &[f64; SUPPORT_SIZE],
) -> [u64; SUPPORT_SIZE] {
    let mut counts = [0_u64; SUPPORT_SIZE];
    let mut cumulative = [0.0_f64; SUPPORT_SIZE];
    let mut running = 0.0;
    for index in 0..SUPPORT_SIZE {
        running += probabilities[index];
        cumulative[index] = running;
    }
    cumulative[SUPPORT_SIZE - 1] = 1.0;

    for _ in 0..sample_size {
        let draw = rng.next_unit_f64();
        let mut category = SUPPORT_SIZE - 1;
        for (index, boundary) in cumulative.iter().enumerate() {
            if draw < *boundary {
                category = index;
                break;
            }
        }
        counts[category] += 1;
    }
    counts
}

fn empirical_expectation(
    counts: &[u64; SUPPORT_SIZE],
    sample_size: usize,
    mask: &[bool; SUPPORT_SIZE],
) -> f64 {
    let successes = counts
        .iter()
        .zip(mask)
        .filter_map(|(count, included)| included.then_some(*count))
        .sum::<u64>();
    successes as f64 / sample_size as f64
}

fn empirical_tv_distance(
    counts: &[u64; SUPPORT_SIZE],
    sample_size: usize,
    nominal: &[f64; SUPPORT_SIZE],
) -> f64 {
    let mut distance = 0.0;
    for index in 0..SUPPORT_SIZE {
        let empirical = counts[index] as f64 / sample_size as f64;
        distance += (empirical - nominal[index]).abs();
    }
    (0.5 * distance).min(1.0)
}

fn wilson_interval(successes: usize, trials: usize) -> [f64; 2] {
    if trials == 0 {
        return [0.0, 1.0];
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z_975 * Z_975;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = Z_975
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;
    [(center - half_width).max(0.0), (center + half_width).min(1.0)]
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

pub fn run_functional_shift_benchmark(
    request: &FunctionalShiftBenchmarkRequest,
) -> Result<FunctionalShiftBenchmarkReport, FunctionalShiftBenchmarkError> {
    request.validate()?;
    let fixture = FunctionalShiftFixture::new(request.scenario);
    fixture.validate()?;

    let nominal_functionals = fixture.nominal_functional_expectations();
    let target_functionals = fixture.target_functional_expectations();
    let nominal_omitted = fixture.nominal_omitted_expectation();
    let target_omitted = fixture.target_omitted_expectation();
    let true_tv = fixture.true_total_variation_distance();

    let dfce_radius = hoeffding_radius(
        request.samples_per_replication,
        request.functional_family_failure_probability,
    )?;
    let tv_slack = weissman_tv_slack(
        request.samples_per_replication,
        request.tv_failure_probability,
    )?;

    let request_sha256 = request.request_sha256()?;
    let mut rng = SplitMix64::new(request.seed);
    let mut covered_replications = 0usize;
    let mut empirical_tv_sum = 0.0;
    let mut tv_radii = Vec::with_capacity(request.replications);

    for _replication in 0..request.replications {
        let counts = draw_counts(
            &mut rng,
            request.samples_per_replication,
            &fixture.target_probabilities,
        );
        let mut simultaneously_covered = true;
        for functional_index in 0..FUNCTIONAL_COUNT {
            let empirical = empirical_expectation(
                &counts,
                request.samples_per_replication,
                &fixture.functional_masks[functional_index],
            );
            let true_mean = target_functionals[functional_index];
            match fixture.functional_directions[functional_index] {
                FunctionalDirection::Upper => {
                    if empirical + dfce_radius + 1e-12 < true_mean {
                        simultaneously_covered = false;
                    }
                }
                FunctionalDirection::Lower => {
                    if empirical - dfce_radius - 1e-12 > true_mean {
                        simultaneously_covered = false;
                    }
                }
            }
        }
        if simultaneously_covered {
            covered_replications += 1;
        }

        let empirical_tv = empirical_tv_distance(
            &counts,
            request.samples_per_replication,
            &fixture.nominal_probabilities,
        );
        empirical_tv_sum += empirical_tv;
        tv_radii.push((empirical_tv + tv_slack).min(1.0));
    }

    tv_radii.sort_by(f64::total_cmp);
    let mean_tv_radius = tv_radii.iter().sum::<f64>() / tv_radii.len() as f64;
    let coverage_rate = covered_replications as f64 / request.replications as f64;
    let wilson = wilson_interval(covered_replications, request.replications);

    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        benchmark_id: &request.benchmark_id,
        scenario: request.scenario,
        support_size: SUPPORT_SIZE,
        nominal_path_probabilities: fixture.nominal_probabilities,
        target_path_probabilities: fixture.target_probabilities,
        functional_masks: fixture.functional_masks,
        functional_directions: fixture.functional_directions,
        omitted_function_mask: fixture.omitted_function_mask,
        functional_confidence_scope: "SIMULTANEOUS_DECLARED_FUNCTIONALS_ONLY",
        tv_confidence_scope: "FINITE_CATEGORICAL_FULL_PATH_DISTRIBUTION",
        request_sha256: &request_sha256,
        seed: request.seed,
        replications: request.replications,
        samples_per_replication: request.samples_per_replication,
        functional_family_failure_probability: request.functional_family_failure_probability,
        tv_failure_probability: request.tv_failure_probability,
        nominal_functional_expectations: nominal_functionals,
        target_functional_expectations: target_functionals,
        nominal_omitted_functional_expectation: nominal_omitted,
        target_omitted_functional_expectation: target_omitted,
        omitted_functional_expectation_shift: target_omitted - nominal_omitted,
        true_total_variation_distance: true_tv,
        scalar_hoeffding_radius: dfce_radius,
        weissman_tv_concentration_slack: tv_slack,
        simultaneous_functional_coverage_count: covered_replications,
        simultaneous_functional_coverage_rate: coverage_rate,
        simultaneous_functional_coverage_wilson_95: wilson,
        mean_empirical_tv_distance: empirical_tv_sum / request.replications as f64,
        mean_sample_based_tv_radius: mean_tv_radius,
        median_sample_based_tv_radius: quantile(&tv_radii, 0.50),
        p05_sample_based_tv_radius: quantile(&tv_radii, 0.05),
        p95_sample_based_tv_radius: quantile(&tv_radii, 0.95),
    };
    let report_sha256 = digest(&payload)?;

    Ok(FunctionalShiftBenchmarkReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        benchmark_id: request.benchmark_id.clone(),
        scenario: request.scenario,
        support_size: SUPPORT_SIZE,
        nominal_path_probabilities: fixture.nominal_probabilities,
        target_path_probabilities: fixture.target_probabilities,
        functional_masks: fixture.functional_masks,
        functional_directions: fixture.functional_directions,
        omitted_function_mask: fixture.omitted_function_mask,
        functional_confidence_scope: "SIMULTANEOUS_DECLARED_FUNCTIONALS_ONLY".into(),
        tv_confidence_scope: "FINITE_CATEGORICAL_FULL_PATH_DISTRIBUTION".into(),
        request_sha256,
        seed: request.seed,
        replications: request.replications,
        samples_per_replication: request.samples_per_replication,
        functional_family_failure_probability: request.functional_family_failure_probability,
        tv_failure_probability: request.tv_failure_probability,
        nominal_functional_expectations: nominal_functionals,
        target_functional_expectations: target_functionals,
        nominal_omitted_functional_expectation: nominal_omitted,
        target_omitted_functional_expectation: target_omitted,
        omitted_functional_expectation_shift: target_omitted - nominal_omitted,
        true_total_variation_distance: true_tv,
        scalar_hoeffding_radius: dfce_radius,
        weissman_tv_concentration_slack: tv_slack,
        simultaneous_functional_coverage_count: covered_replications,
        simultaneous_functional_coverage_rate: coverage_rate,
        simultaneous_functional_coverage_wilson_95: wilson,
        mean_empirical_tv_distance: empirical_tv_sum / request.replications as f64,
        mean_sample_based_tv_radius: mean_tv_radius,
        median_sample_based_tv_radius: quantile(&tv_radii, 0.50),
        p05_sample_based_tv_radius: quantile(&tv_radii, 0.05),
        p95_sample_based_tv_radius: quantile(&tv_radii, 0.95),
        report_sha256,
    })
}

pub fn verify_functional_shift_benchmark_report(
    request: &FunctionalShiftBenchmarkRequest,
    report: &FunctionalShiftBenchmarkReport,
) -> Result<(), FunctionalShiftBenchmarkError> {
    if run_functional_shift_benchmark(request)? != *report {
        return Err(FunctionalShiftBenchmarkError::new(
            "report",
            "stored functional-shift report differs from deterministic reconstruction",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, FunctionalShiftBenchmarkError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| FunctionalShiftBenchmarkError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn orthogonal_fixture_changes_distribution_not_declared_functionals() {
        let fixture = FunctionalShiftFixture::new(
            FunctionalShiftScenario::OrthogonalToDeclaredFunctionals,
        );
        assert!((fixture.true_total_variation_distance() - 0.20).abs() < TOLERANCE);
        assert!(fixture
            .nominal_functional_expectations()
            .iter()
            .all(|value| (*value - 0.50).abs() < TOLERANCE));
        assert!(fixture
            .target_functional_expectations()
            .iter()
            .all(|value| (*value - 0.50).abs() < TOLERANCE));
        assert!((fixture.target_omitted_expectation() - 0.20).abs() < TOLERANCE);
        assert!((fixture.nominal_omitted_expectation() - 0.125).abs() < TOLERANCE);
        assert!((fixture.target_omitted_expectation() - fixture.nominal_omitted_expectation() - 0.075).abs() < TOLERANCE);
    }

    #[test]
    fn aligned_fixture_changes_rejection_and_cost_functionals() {
        let fixture = FunctionalShiftFixture::new(FunctionalShiftScenario::DecisionAligned);
        assert!((fixture.true_total_variation_distance() - 0.10).abs() < TOLERANCE);
        let expected = fixture.target_functional_expectations();
        let wanted = [0.60, 0.50, 0.40, 0.50];
        for (actual, expected) in expected.iter().zip(wanted) {
            assert!((*actual - expected).abs() < TOLERANCE);
        }
        assert!((fixture.target_omitted_expectation() - 0.10).abs() < TOLERANCE);
    }

    #[test]
    fn hoeffding_and_weissman_formula_match_the_frozen_settings() {
        let hoeffding = hoeffding_radius(5_000, 0.01).unwrap();
        let tv = weissman_tv_slack(5_000, 0.01).unwrap();
        assert!((hoeffding - 0.02447747).abs() < 1e-8);
        assert!((tv - 0.03185961).abs() < 1e-8);
    }

    #[test]
    fn identical_request_and_seed_reconstruct_identical_report() {
        let request = FunctionalShiftBenchmarkRequest::new(
            "deterministic-fixture",
            FunctionalShiftScenario::OrthogonalToDeclaredFunctionals,
            20261102,
            200,
            500,
            0.01,
            0.01,
        )
        .unwrap();
        let first = run_functional_shift_benchmark(&request).unwrap();
        let second = run_functional_shift_benchmark(&request).unwrap();
        assert_eq!(first, second);
        verify_functional_shift_benchmark_report(&request, &first).unwrap();
    }

    #[test]
    fn rejects_empty_replications_and_invalid_failure_budget() {
        assert!(FunctionalShiftBenchmarkRequest::new(
            "invalid",
            FunctionalShiftScenario::DecisionAligned,
            1,
            0,
            500,
            0.01,
            0.01,
        )
        .is_err());
        assert!(FunctionalShiftBenchmarkRequest::new(
            "invalid",
            FunctionalShiftScenario::DecisionAligned,
            1,
            10,
            500,
            1.0,
            0.01,
        )
        .is_err());
    }

    #[test]
    fn support_scaling_penalizes_tv_not_fixed_functional_dimension() {
        let dfce_d4 = hoeffding_radius(5_000, 0.01).unwrap();
        let tv_k8 = (8.0 * std::f64::consts::LN_2 + 100.0_f64.ln()).sqrt()
            / (2.0 * 5_000.0_f64).sqrt();
        let tv_k8192 = (8_192.0 * std::f64::consts::LN_2 + 100.0_f64.ln()).sqrt()
            / (2.0 * 5_000.0_f64).sqrt();
        assert!((tv_k8 - 0.03185961).abs() < 1e-8);
        assert!(tv_k8192 > 0.75);
        assert!((dfce_d4 - hoeffding_radius(5_000, 0.01).unwrap()).abs() < TOLERANCE);
    }
}
