//! Deterministic comparison of resonator cleanup nonlinearities.
//!
//! The rule-specific settings follow the 2026 cleanup-rule paper's bipolar
//! comparison: sign and softmax use bipolar sign projection; ReLU and
//! polynomial retain real-valued reconstructed states. Softmax uses temperature
//! 0.05 (inverse temperature beta ~= 20); polynomial uses degree 2.
//!
//! This is a mechanism comparison, not a leaderboard or universal rule claim.

use super::associative_cleanup_harness::{fixture_seed, noisy_query};
use super::resonator::{CleanupRule, Factor, MultiConstraint, ResonatorConfig, ResonatorNetwork};
use super::resonator_separation_harness::{correlated_codebook, exhaustive_pair, geometry, nearest};
use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u32 = 1;
pub const TASK_NAME: &str = "hdc_two_factor_resonator_cleanup_rule_comparison";
pub const SCENARIO_SET: &str = "two-factor-resonator-cleanup-rules-v1";
pub const SCENARIO_REVISION: &str = "sha256:two-factor-cleanup-rules-fixture-v1";
pub const DEFAULT_SEED: u64 = 0x4352_4c4e_52554c45;
pub const DEFAULT_DIMENSIONS: &[usize] = &[2_048, 8_192];
pub const DEFAULT_CODEBOOK_SIZES: &[usize] = &[4, 8];
pub const DEFAULT_SHARED_COMPONENT_PROBABILITIES: &[f32] = &[0.0, 0.50, 0.75];
pub const DEFAULT_QUERY_NOISE_WEIGHTS: &[f32] = &[0.0, 0.20];
pub const DEFAULT_SOLVER_SEEDS: &[u64] = &[0x51, 0xA7, 0xD3];
pub const QUERIES_PER_SIZE: usize = 1;
pub const MAX_SOLVER_ITERATIONS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    Sign,
    Softmax,
    Relu,
    Polynomial,
}

impl Rule {
    pub const ALL: &'static [Self] = &[Self::Sign, Self::Softmax, Self::Relu, Self::Polynomial];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Sign => "sign",
            Self::Softmax => "softmax",
            Self::Relu => "relu",
            Self::Polynomial => "polynomial",
        }
    }

    pub const fn config(self) -> (CleanupRule, f32, u32, bool) {
        match self {
            Self::Sign => (CleanupRule::Sign, 0.05, 2, true),
            Self::Softmax => (CleanupRule::Softmax, 0.05, 2, true),
            Self::Relu => (CleanupRule::Relu, 0.05, 2, false),
            Self::Polynomial => (CleanupRule::Polynomial, 0.05, 2, false),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub resolution: usize,
    pub codebook_size: usize,
    pub shared_component_probability: f32,
    pub query_noise_weight: f32,
    pub cleanup_rule: String,
    pub cleanup_temperature: f32,
    pub polynomial_degree: u32,
    pub sign_projection: bool,
    pub mean_abs_pairwise_similarity: f64,
    pub max_abs_pairwise_similarity: f64,
    pub exhaustive_accuracy: f64,
    pub exhaustive_mean_margin: f64,
    pub correct_convergence: u64,
    pub spurious_convergence: u64,
    pub non_convergence: u64,
    pub joint_accuracy: f64,
    pub mean_iterations: f64,
    pub converged_mean_iterations: f64,
    pub mean_factor_margin: f64,
    pub mean_terminal_energy: f64,
    pub resonator_trials: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub seed: u64,
    pub solver_seeds: Vec<u64>,
    pub cleanup_rules: Vec<String>,
    pub queries_per_size: usize,
    pub max_solver_iterations: usize,
    pub exhaustive_search_space_formula: String,
    pub cells: Vec<Cell>,
    pub spec_identity: String,
    pub artifact_digest: String,
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    let mut out = String::from("sha256:");
    for b in Sha256::digest(bytes) {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

pub fn run() -> Evidence {
    let mut cells = Vec::new();
    for &resolution in DEFAULT_DIMENSIONS {
        for &codebook_size in DEFAULT_CODEBOOK_SIZES {
            for &shared in DEFAULT_SHARED_COMPONENT_PROBABILITIES {
                let values = correlated_codebook(resolution, codebook_size, DEFAULT_SEED, shared);
                let (mean_abs_pairwise_similarity, max_abs_pairwise_similarity) = geometry(&values);

                for &noise in DEFAULT_QUERY_NOISE_WEIGHTS {
                    for &rule in Rule::ALL {
                        let (cleanup_rule, temperature, polynomial_degree, sign_projection) = rule.config();
                        let mut exhaustive_correct = 0u64;
                        let mut exhaustive_margin_sum = 0.0f64;
                        let mut correct = 0u64;
                        let mut spurious = 0u64;
                        let mut non_convergence = 0u64;
                        let mut joint_correct = 0u64;
                        let mut iteration_sum = 0u64;
                        let mut converged_iteration_sum = 0u64;
                        let mut factor_margin_sum = 0.0f64;
                        let mut terminal_energy_sum = 0.0f64;

                        for x_target in 0..codebook_size {
                            for y_target in 0..codebook_size {
                                let clean_query = values[x_target].bind(&values[y_target]);
                                let query = noisy_query(
                                    &clean_query,
                                    noise,
                                    fixture_seed(
                                        DEFAULT_SEED,
                                        x_target * codebook_size + y_target,
                                        0x4352_4c4e,
                                    ),
                                );
                                let (pair, best, second) = exhaustive_pair(&values, &query);
                                exhaustive_correct += u64::from(
                                    pair == (x_target, y_target) || pair == (y_target, x_target),
                                );
                                exhaustive_margin_sum += f64::from(best - second);

                                for &solver_seed in DEFAULT_SOLVER_SEEDS {
                                    let config = ResonatorConfig {
                                        cleanup_rule,
                                        temperature,
                                        polynomial_degree,
                                        cleanup_sign_projection: sign_projection,
                                        max_iterations: MAX_SOLVER_ITERATIONS,
                                        ..Default::default()
                                    };
                                    let mut network =
                                        ResonatorNetwork::with_config(resolution, config).unwrap();
                                    for (index, value) in values.iter().enumerate() {
                                        network
                                            .add_symbol(&format!("value-{index}"), value.values.clone())
                                            .unwrap();
                                    }

                                    let constraints = [
                                        MultiConstraint::new(
                                            Factor::Unknown("y".to_string()),
                                            Factor::Unknown("x".to_string()),
                                            Factor::Known(query.values.clone()),
                                        ),
                                        MultiConstraint::new(
                                            Factor::Unknown("x".to_string()),
                                            Factor::Unknown("y".to_string()),
                                            Factor::Known(query.values.clone()),
                                        ),
                                    ];
                                    let solutions = network
                                        .solve_system_seeded(
                                            &["x", "y"],
                                            &constraints,
                                            Some(MAX_SOLVER_ITERATIONS),
                                            solver_seed,
                                        )
                                        .unwrap();
                                    let x = solutions.get("x").unwrap();
                                    let y = solutions.get("y").unwrap();
                                    let (xi, xb, xs) =
                                        nearest(&values, &ContinuousHV::from_vec(x.vector.clone()));
                                    let (yi, yb, ys) =
                                        nearest(&values, &ContinuousHV::from_vec(y.vector.clone()));
                                    let joint = (xi == x_target && yi == y_target)
                                        || (xi == y_target && yi == x_target);
                                    joint_correct += u64::from(joint);
                                    factor_margin_sum += f64::from((xb - xs).min(yb - ys));
                                    let iterations = x.iterations.max(y.iterations) as u64;
                                    iteration_sum += iterations;
                                    terminal_energy_sum += f64::from(network.current_energy());

                                    if x.converged && y.converged {
                                        converged_iteration_sum += iterations;
                                        if joint { correct += 1; } else { spurious += 1; }
                                    } else {
                                        non_convergence += 1;
                                    }
                                }
                            }
                        }

                        let exhaustive_trials = (codebook_size * codebook_size) as u64;
                        let resonator_trials = exhaustive_trials * DEFAULT_SOLVER_SEEDS.len() as u64;
                        cells.push(Cell {
                            resolution,
                            codebook_size,
                            shared_component_probability: shared,
                            query_noise_weight: noise,
                            cleanup_rule: rule.name().to_owned(),
                            cleanup_temperature: temperature,
                            polynomial_degree,
                            sign_projection,
                            mean_abs_pairwise_similarity,
                            max_abs_pairwise_similarity,
                            exhaustive_accuracy: exhaustive_correct as f64 / exhaustive_trials as f64,
                            exhaustive_mean_margin: exhaustive_margin_sum / exhaustive_trials as f64,
                            correct_convergence: correct,
                            spurious_convergence: spurious,
                            non_convergence,
                            joint_accuracy: joint_correct as f64 / resonator_trials as f64,
                            mean_iterations: iteration_sum as f64 / resonator_trials as f64,
                            converged_mean_iterations: if correct + spurious > 0 {
                                converged_iteration_sum as f64 / (correct + spurious) as f64
                            } else { 0.0 },
                            mean_factor_margin: factor_margin_sum / resonator_trials as f64,
                            mean_terminal_energy: terminal_energy_sum / resonator_trials as f64,
                            resonator_trials,
                        });
                    }
                }
            }
        }
    }

    let canonical = format!(
        "{SCHEMA_VERSION}|{TASK_NAME}|{SCENARIO_SET}|{SCENARIO_REVISION}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}",
        DEFAULT_DIMENSIONS,
        DEFAULT_CODEBOOK_SIZES,
        DEFAULT_SHARED_COMPONENT_PROBABILITIES,
        DEFAULT_QUERY_NOISE_WEIGHTS,
        DEFAULT_SOLVER_SEEDS,
        QUERIES_PER_SIZE,
        MAX_SOLVER_ITERATIONS
    );
    let mut evidence = Evidence {
        schema_version: SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        seed: DEFAULT_SEED,
        solver_seeds: DEFAULT_SOLVER_SEEDS.to_vec(),
        cleanup_rules: Rule::ALL.iter().map(|r| r.name().to_owned()).collect(),
        queries_per_size: QUERIES_PER_SIZE,
        max_solver_iterations: MAX_SOLVER_ITERATIONS,
        exhaustive_search_space_formula: "n^2".to_owned(),
        cells,
        spec_identity: digest(canonical),
        artifact_digest: String::new(),
    };
    evidence.artifact_digest =
        digest(serde_json::to_vec(&evidence).expect("cleanup-rule evidence serializes"));
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_are_explicit_and_distinct() {
        assert_eq!(Rule::ALL.len(), 4);
        assert_ne!(Rule::Sign.config(), Rule::Softmax.config());
        assert!(!Rule::Relu.config().3);
        assert!(!Rule::Polynomial.config().3);
    }

    #[test]
    fn evidence_has_complete_rule_matrix() {
        let evidence = run();
        assert_eq!(evidence.cleanup_rules.len(), 4);
        assert_eq!(evidence.cells.len(), 128);
        for cell in &evidence.cells {
            assert_eq!(
                cell.correct_convergence + cell.spurious_convergence + cell.non_convergence,
                cell.resonator_trials
            );
            assert!(cell.mean_terminal_energy.is_finite());
        }
    }
}
