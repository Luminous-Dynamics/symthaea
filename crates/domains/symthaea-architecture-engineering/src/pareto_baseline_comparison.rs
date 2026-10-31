//! Matched-budget comparison harness for the in-repo Pareto evolutionary baselines.
//!
//! NSGA-II, MOEA/D-DE and SMS-EMOA are replay-verified, then scored with identical objective
//! specs, utility reference point and reference Pareto front. A comparison is rejected unless
//! problem ID, seed, objective dimensions and exact objective-evaluation count match.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::pareto_moead::{
    analyze_moead_report, verify_moead_report, MoeadConfig, MoeadRunReport,
};
use super::pareto_nsga2::{
    analyze_nsga2_report, verify_nsga2_report, Nsga2Config, Nsga2RunReport,
};
use super::pareto_objectives::{
    analyze_reference_front_indicators, ObjectiveDirection, ParetoError,
    ParetoObjectiveSpec,
};
use super::pareto_smsemoa::{
    analyze_smsemoa_report, verify_smsemoa_report, SmsEmoaConfig, SmsEmoaRunReport,
};

const SCHEMA: &str = "symthaea.architecture-engineering.pareto-baseline-comparison";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineComparisonError {
    pub path: String,
    pub message: String,
}

impl BaselineComparisonError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for BaselineComparisonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for BaselineComparisonError {}

/// A report/config pair from one of the pinned in-repository algorithms.
#[derive(Debug, Clone)]
pub enum ParetoBaselineRun {
    Nsga2 {
        config: Nsga2Config,
        report: Nsga2RunReport,
    },
    Moead {
        config: MoeadConfig,
        report: MoeadRunReport,
    },
    SmsEmoa {
        config: SmsEmoaConfig,
        report: SmsEmoaRunReport,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoBaselineResult {
    pub algorithm_id: String,
    pub problem_id: String,
    pub seed: u64,
    pub objective_evaluations: usize,
    pub final_population_size: usize,
    pub archive_size: usize,
    pub run_report_sha256: String,
    pub analysis_plan_sha256: String,
    pub analysis_receipt_sha256: String,
    pub hypervolume: f64,
    pub igd_plus: f64,
    pub additive_epsilon: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoBaselineComparison {
    pub schema: String,
    pub version: String,
    pub problem_id: String,
    pub seed: u64,
    /// All algorithms must spend exactly this many objective evaluations.
    pub matched_objective_evaluations: usize,
    pub objective_specs_sha256: String,
    pub reference_point_sha256: String,
    pub reference_front_sha256: String,
    pub results: Vec<ParetoBaselineResult>,
    pub comparison_sha256: String,
}

#[derive(Serialize)]
struct ComparisonReceiptPayload<'a> {
    schema: &'a str,
    version: &'a str,
    problem_id: &'a str,
    seed: u64,
    matched_objective_evaluations: usize,
    objective_specs_sha256: &'a str,
    reference_point_sha256: &'a str,
    reference_front_sha256: &'a str,
    results: &'a [ParetoBaselineResult],
}

pub fn compare_pareto_baselines(
    runs: &[ParetoBaselineRun],
    objective_specs: &[ParetoObjectiveSpec],
    reference_point_utility: &[f64],
    reference_front_utility: &[Vec<f64>],
) -> Result<ParetoBaselineComparison, BaselineComparisonError> {
    if runs.len() < 2 {
        return Err(BaselineComparisonError::new(
            "runs",
            "at least two evolutionary baselines are required",
        ));
    }
    if objective_specs.len() < 2 || objective_specs.len() > 4 {
        return Err(BaselineComparisonError::new(
            "objective_specs",
            "common comparison harness requires two through four objectives",
        ));
    }
    if objective_specs.iter().any(|spec| spec.direction != ObjectiveDirection::Minimize) {
        return Err(BaselineComparisonError::new(
            "objective_specs.direction",
            "ZDT1 and DTLZ2 baseline functions are native minimization problems",
        ));
    }
    if reference_point_utility.len() != objective_specs.len()
        || reference_point_utility.iter().any(|value| !value.is_finite())
    {
        return Err(BaselineComparisonError::new(
            "reference_point_utility",
            "reference point dimensions must match and every coordinate must be finite",
        ));
    }

    let objective_specs_sha256 = digest_serializable(objective_specs)?;
    let reference_point_sha256 = digest_serializable(reference_point_utility)?;
    let reference_front_sha256 = digest_serializable(reference_front_utility)?;

    let mut results = Vec::with_capacity(runs.len());
    let mut method_ids = BTreeSet::new();
    let mut common_problem: Option<String> = None;
    let mut common_seed: Option<u64> = None;
    let mut common_budget: Option<usize> = None;

    for (index, run) in runs.iter().enumerate() {
        let (algorithm_id, problem_id, seed, objective_evaluations, final_population_size, archive_size,
            run_report_sha256, analysis_plan, analysis_summary, indicator_receipt) =
            match run {
                ParetoBaselineRun::Nsga2 { config, report } => {
                    verify_nsga2_report(report, config)
                        .map_err(|error| BaselineComparisonError::new(
                            format!("runs[{index}].replay"),
                            error.to_string(),
                        ))?;
                    let analysis = analyze_nsga2_report(
                        report,
                        config,
                        objective_specs.to_vec(),
                        reference_point_utility.to_vec(),
                    ).map_err(map_nsga2_error)?;
                    let indicators = analyze_reference_front_indicators(
                        &analysis.plan,
                        reference_front_utility,
                    ).map_err(map_pareto_error)?;
                    (
                        report.algorithm_id.clone(),
                        report.problem_id.clone(),
                        report.seed,
                        report.objective_evaluations,
                        report.final_population.len(),
                        report.nondominated_front.len(),
                        report.report_sha256.clone(),
                        analysis.plan,
                        analysis.analysis,
                        indicators,
                    )
                }
                ParetoBaselineRun::Moead { config, report } => {
                    verify_moead_report(report, config)
                        .map_err(|error| BaselineComparisonError::new(
                            format!("runs[{index}].replay"),
                            error.to_string(),
                        ))?;
                    let analysis = analyze_moead_report(
                        report,
                        config,
                        objective_specs.to_vec(),
                        reference_point_utility.to_vec(),
                    ).map_err(map_moead_error)?;
                    let indicators = analyze_reference_front_indicators(
                        &analysis.plan,
                        reference_front_utility,
                    ).map_err(map_pareto_error)?;
                    (
                        report.algorithm_id.clone(),
                        report.problem_id.clone(),
                        report.seed,
                        report.objective_evaluations,
                        report.final_population.len(),
                        report.nondominated_front.len(),
                        report.report_sha256.clone(),
                        analysis.plan,
                        analysis.analysis,
                        indicators,
                    )
                }
                ParetoBaselineRun::SmsEmoa { config, report } => {
                    verify_smsemoa_report(report, config)
                        .map_err(|error| BaselineComparisonError::new(
                            format!("runs[{index}].replay"),
                            error.to_string(),
                        ))?;
                    let analysis = analyze_smsemoa_report(report, config)
                        .map_err(map_smsemoa_error)?;
                    // The config is the frozen source of objective anchors and reference point.
                    if config.objective_specs != objective_specs
                        || config.reference_point_utility != reference_point_utility
                    {
                        return Err(BaselineComparisonError::new(
                            format!("runs[{index}].config"),
                            "SMS-EMOA objective specs/reference point differ from the common comparison contract",
                        ));
                    }
                    let indicators = analyze_reference_front_indicators(
                        &analysis.plan,
                        reference_front_utility,
                    ).map_err(map_pareto_error)?;
                    (
                        report.algorithm_id.clone(),
                        report.problem_id.clone(),
                        report.seed,
                        report.objective_evaluations,
                        report.final_population.len(),
                        report.nondominated_front.len(),
                        report.report_sha256.clone(),
                        analysis.plan,
                        analysis.analysis,
                        indicators,
                    )
                }
            };

        if !method_ids.insert(algorithm_id.clone()) {
            return Err(BaselineComparisonError::new(
                format!("runs[{index}].algorithm_id"),
                "comparison contains a duplicate algorithm ID",
            ));
        }
        if analysis_plan.objective_specs != objective_specs
            || analysis_plan.reference_point_utility != reference_point_utility
        {
            return Err(BaselineComparisonError::new(
                format!("runs[{index}].analysis_plan"),
                "reconstructed plan differs from the common objective/reference contract",
            ));
        }

        match (&common_problem, &common_seed, &common_budget) {
            (None, None, None) => {
                common_problem = Some(problem_id.clone());
                common_seed = Some(seed);
                common_budget = Some(objective_evaluations);
            }
            (Some(expected_problem), Some(expected_seed), Some(expected_budget)) => {
                if expected_problem != &problem_id {
                    return Err(BaselineComparisonError::new(
                        format!("runs[{index}].problem_id"),
                        "all baselines must run the same benchmark problem",
                    ));
                }
                if *expected_seed != seed {
                    return Err(BaselineComparisonError::new(
                        format!("runs[{index}].seed"),
                        "all baselines must use the same seed block",
                    ));
                }
                if *expected_budget != objective_evaluations {
                    return Err(BaselineComparisonError::new(
                        format!("runs[{index}].objective_evaluations"),
                        "all baselines must use exactly the same objective-evaluation budget",
                    ));
                }
            }
            _ => {
                return Err(BaselineComparisonError::new(
                    "comparison",
                    "internal common-contract state is inconsistent",
                ));
            }
        }

        results.push(ParetoBaselineResult {
            algorithm_id,
            problem_id,
            seed,
            objective_evaluations,
            final_population_size,
            archive_size,
            run_report_sha256,
            analysis_plan_sha256: analysis_summary.plan_sha256,
            analysis_receipt_sha256: analysis_summary.receipt_sha256,
            hypervolume: analysis_summary.hypervolume,
            igd_plus: indicator_receipt.igd_plus,
            additive_epsilon: indicator_receipt.additive_epsilon,
        });
    }

    results.sort_by(|left, right| left.algorithm_id.cmp(&right.algorithm_id));
    let problem_id = common_problem.ok_or_else(|| {
        BaselineComparisonError::new("problem_id", "no baseline problem was reconstructed")
    })?;
    let seed = common_seed.ok_or_else(|| {
        BaselineComparisonError::new("seed", "no baseline seed was reconstructed")
    })?;
    let matched_objective_evaluations = common_budget.ok_or_else(|| {
        BaselineComparisonError::new("objective_evaluations", "no baseline budget was reconstructed")
    })?;
    let comparison_sha256 = digest_serializable(&ComparisonReceiptPayload {
        schema: SCHEMA,
        version: VERSION,
        problem_id: &problem_id,
        seed,
        matched_objective_evaluations,
        objective_specs_sha256: &objective_specs_sha256,
        reference_point_sha256: &reference_point_sha256,
        reference_front_sha256: &reference_front_sha256,
        results: &results,
    })?;

    Ok(ParetoBaselineComparison {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        problem_id,
        seed,
        matched_objective_evaluations,
        objective_specs_sha256,
        reference_point_sha256,
        reference_front_sha256,
        results,
        comparison_sha256,
    })
}

fn map_nsga2_error(error: super::pareto_nsga2::Nsga2Error) -> BaselineComparisonError {
    BaselineComparisonError::new(error.path, error.message)
}

fn map_moead_error(error: super::pareto_moead::MoeadError) -> BaselineComparisonError {
    BaselineComparisonError::new(error.path, error.message)
}

fn map_smsemoa_error(error: super::pareto_smsemoa::SmsEmoaError) -> BaselineComparisonError {
    BaselineComparisonError::new(error.path, error.message)
}

fn map_pareto_error(error: ParetoError) -> BaselineComparisonError {
    BaselineComparisonError::new(error.path, error.message)
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, BaselineComparisonError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        BaselineComparisonError::new("serialization", format!("cannot serialize comparison receipt: {error}"))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto_benchmarks::zdt1_reference_front;
    use crate::pareto_moead::{run_moead_zdt1, MoeadWeightMode};
    use crate::pareto_nsga2::run_nsga2_zdt1;
    use crate::pareto_smsemoa::run_smsemoa_zdt1;
    use serde_json::json;

    fn objective_specs() -> Vec<ParetoObjectiveSpec> {
        vec![
            ParetoObjectiveSpec {
                objective_id: "f1".to_owned(),
                direction: ObjectiveDirection::Minimize,
                scale_lower: 0.0,
                scale_upper: 1.0,
            },
            ParetoObjectiveSpec {
                objective_id: "f2".to_owned(),
                direction: ObjectiveDirection::Minimize,
                scale_lower: 0.0,
                scale_upper: 10.0,
            },
        ]
    }

    fn reference_front() -> Vec<Vec<f64>> {
        zdt1_reference_front(21).unwrap().into_iter()
            .map(|point| vec![1.0 - point[0], (10.0 - point[1]) / 10.0])
            .collect()
    }

    fn baseline_runs(seed: u64) -> Vec<ParetoBaselineRun> {
        let nsga_config = Nsga2Config {
            population_size: 20,
            generations: 2,
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            crossover_distribution_index: 15.0,
            mutation_distribution_index: 20.0,
            seed,
        };
        let moead_config = MoeadConfig {
            weight_mode: MoeadWeightMode::Uniform2d { population_size: 20 },
            generations: 2,
            neighbourhood_size: 8,
            neighbourhood_mating_probability: 0.9,
            differential_weight: 0.5,
            crossover_rate: 1.0,
            max_replacements: 2,
            seed,
        };
        let sms_config = SmsEmoaConfig {
            population_size: 20,
            offspring_evaluations: 40,
            objective_specs: objective_specs(),
            reference_point_utility: vec![-0.1, -0.1],
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            crossover_distribution_index: 15.0,
            mutation_distribution_index: 20.0,
            seed,
        };
        vec![
            ParetoBaselineRun::Nsga2 {
                report: run_nsga2_zdt1(&nsga_config, 10).unwrap(),
                config: nsga_config,
            },
            ParetoBaselineRun::Moead {
                report: run_moead_zdt1(&moead_config, 10).unwrap(),
                config: moead_config,
            },
            ParetoBaselineRun::SmsEmoa {
                report: run_smsemoa_zdt1(&sms_config, 10).unwrap(),
                config: sms_config,
            },
        ]
    }

    #[test]
    fn all_three_baselines_compare_with_identical_problem_seed_and_evaluation_budget() {
        let comparison = compare_pareto_baselines(
            &baseline_runs(42),
            &objective_specs(),
            &[-0.1, -0.1],
            &reference_front(),
        ).unwrap();
        assert_eq!(comparison.results.len(), 3);
        assert_eq!(comparison.matched_objective_evaluations, 60);
        assert_eq!(comparison.results[0].objective_evaluations, 60);
        assert_eq!(comparison.comparison_sha256.len(), 64);
        assert!(comparison.results.iter().all(|result| {
            result.hypervolume.is_finite()
                && result.igd_plus.is_finite()
                && result.additive_epsilon.is_finite()
        }));
    }

    #[test]
    fn unequal_budgets_different_seeds_and_duplicate_methods_are_rejected() {
        let runs = baseline_runs(42);
        let specs = objective_specs();
        let ref_front = reference_front();

        let mut unequal = runs.clone();
        if let ParetoBaselineRun::SmsEmoa { config, report } = &mut unequal[2] {
            config.offspring_evaluations = 41;
            report.objective_evaluations += 1;
        }
        assert!(compare_pareto_baselines(&unequal, &specs, &[-0.1, -0.1], &ref_front).is_err());

        let mut duplicate = runs.clone();
        duplicate.push(runs[0].clone());
        assert!(compare_pareto_baselines(&duplicate, &specs, &[-0.1, -0.1], &ref_front).is_err());

        assert!(compare_pareto_baselines(&runs[..1], &specs, &[-0.1, -0.1], &ref_front).is_err());
    }

    #[test]
    fn comparison_rejects_incompatible_reference_point_or_objective_order() {
        let runs = baseline_runs(7);
        let specs = objective_specs();
        assert!(compare_pareto_baselines(&runs, &specs, &[0.0], &reference_front()).is_err());

        let mut reordered = specs;
        reordered.swap(0, 1);
        assert!(compare_pareto_baselines(&runs, &reordered, &[-0.1, -0.1], &reference_front()).is_err());
    }
}
