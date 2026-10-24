//! Batch-level gate for preregistered five-arm cognitive-adaptation panels.
//!
//! This consumes individually scored v2 records, validates their provenance and metric digests,
//! checks the full dataset × seed × scenario grid, and only then computes the confirmatory
//! dataset-clustered paired comparison. An incomplete panel returns diagnostics with no effect
//! estimate; coverage never becomes a performance claim.
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet}, fs, path::Path};

use crate::cognitive_adaptation_utility::{ArmUtilityMetrics, CounterfactualUtilityScore};
use crate::cognitive_drift_compare::{
    compare, Arm as ComparisonArm, BootstrapSpec, ComparisonInput, MetricSpec, PairedObservation,
};

const REQUIRED_DATASET_COUNT: usize = 7;
const REQUIRED_SEED_COUNT: usize = 50;
const REQUIRED_SCENARIO_IDS: [&str; 8] = [
    "class-prior-abrupt",
    "class-prior-gradual",
    "label-swap-abrupt",
    "label-swap-gradual",
    "feature-permutation-abrupt",
    "feature-permutation-gradual",
    "feature-filtering-abrupt",
    "feature-filtering-gradual",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptationPanelInput {
    pub schema_version: u32,
    /// The information-access contract for every run in this panel.
    pub comparison_track: String,
    /// Exact pinned dataset identifiers expected in the confirmatory panel.
    pub expected_dataset_ids: Vec<String>,
    /// Exact seed identifiers; the same set is required for each dataset.
    pub expected_seed_ids: Vec<u64>,
    /// Must be the eight registered abrupt/gradual drift cells.
    pub expected_scenario_ids: Vec<String>,
    pub bootstrap: BootstrapSpec,
    /// One v2 utility score per dataset × seed × scenario cell. Repeated cells are rejected.
    pub runs: Vec<CounterfactualUtilityScore>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MissingPanelCell {
    pub dataset_id: String,
    pub seed: u64,
    pub scenario_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdaptationPanelResult {
    pub result_schema_version: u32,
    pub metric_version: String,
    pub canonical_input_sha256: String,
    /// SHA-256 of this report with result_sha256 set to null.
    pub result_sha256: Option<String>,
    pub comparison_track: String,
    pub metric_id: String,
    pub metric_unit: String,
    pub expected_dataset_count: usize,
    pub observed_dataset_count: usize,
    pub expected_seed_count_per_dataset: usize,
    pub observed_run_count: usize,
    pub expected_run_count: usize,
    pub missing_cell_count: usize,
    pub missing_cells: Vec<MissingPanelCell>,
    /// True only when every registered dataset × seed × scenario cell is present exactly once.
    pub complete_coverage_gate_pass: bool,
    /// No comparison is emitted unless the entire preregistered grid is complete.
    pub primary_comparison: Option<crate::cognitive_drift_compare::ComparisonResult>,
    pub interpretation_guard: String,
}

#[derive(Debug, Clone)]
struct ArmRegistry {
    version: String,
    repository_head_sha: String,
    config_by_dataset: BTreeMap<String, String>,
}

pub fn evaluate_file(path: &Path) -> anyhow::Result<AdaptationPanelResult> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading adaptation panel {}", path.display()))?;
    let input: AdaptationPanelInput = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing adaptation panel {}", path.display()))?;
    evaluate(input)
}

pub fn evaluate(mut input: AdaptationPanelInput) -> anyhow::Result<AdaptationPanelResult> {
    ensure!(input.schema_version == 1, "adaptation panel schema_version must be 1");
    ensure!(
        ["valid-pe-only", "feature-only-unsupervised", "auxiliary-loss-unsupervised", "observed-outcome"]
            .contains(&input.comparison_track.as_str()),
        "comparison_track must be a declared evidence-access track"
    );
    validate_unique_nonempty(&input.expected_dataset_ids, REQUIRED_DATASET_COUNT, "expected_dataset_ids")?;
    validate_unique_nonempty(&input.expected_scenario_ids, REQUIRED_SCENARIO_IDS.len(), "expected_scenario_ids")?;
    ensure!(
        input.expected_scenario_ids.iter().map(String::as_str).collect::<BTreeSet<_>>()
            == REQUIRED_SCENARIO_IDS.into_iter().collect::<BTreeSet<_>>(),
        "expected_scenario_ids must match the eight registered abrupt/gradual drift cells"
    );
    ensure!(
        input.expected_seed_ids.len() == REQUIRED_SEED_COUNT,
        "expected_seed_ids must contain exactly 50 preregistered seed identifiers"
    );
    ensure!(
        input.expected_seed_ids.iter().copied().collect::<BTreeSet<_>>().len() == REQUIRED_SEED_COUNT,
        "expected_seed_ids must be unique"
    );
    ensure!(
        (1_000..=100_000).contains(&input.bootstrap.replicates),
        "bootstrap.replicates must be between 1000 and 100000"
    );

    input.runs.sort_by(|a, b| {
        (&a.dataset_id, a.seed, &a.scenario_id).cmp(&(&b.dataset_id, b.seed, &b.scenario_id))
    });
    let canonical_input_sha256 = sha256_hex(&serde_json::to_vec(&input)?);

    let expected_datasets: BTreeSet<&str> = input.expected_dataset_ids.iter().map(String::as_str).collect();
    let expected_seeds: BTreeSet<u64> = input.expected_seed_ids.iter().copied().collect();
    let expected_scenarios: BTreeSet<&str> = input.expected_scenario_ids.iter().map(String::as_str).collect();
    let mut observed_keys = BTreeSet::new();
    let mut arm_registry: BTreeMap<String, ArmRegistry> = BTreeMap::new();
    let mut task_loss_metric_id: Option<String> = None;
    let mut task_loss_unit: Option<String> = None;

    for (index, score) in input.runs.iter().enumerate() {
        ensure!(
            score.metric_version == "counterfactual-adaptation-utility-v2",
            "runs[{index}] metric_version must be counterfactual-adaptation-utility-v2"
        );
        ensure!(
            score.comparison_track == input.comparison_track,
            "runs[{index}] comparison_track differs from the panel access track"
        );
        ensure!(
            expected_datasets.contains(score.dataset_id.as_str()),
            "runs[{index}] dataset_id is not in expected_dataset_ids"
        );
        ensure!(expected_seeds.contains(&score.seed), "runs[{index}] seed is not preregistered");
        ensure!(
            expected_scenarios.contains(score.scenario_id.as_str()),
            "runs[{index}] scenario_id is not in the registered scenario grid"
        );
        ensure!(!score.stream_id.trim().is_empty(), "runs[{index}] stream_id must not be empty");
        ensure!(!score.task_loss_metric_id.trim().is_empty(), "runs[{index}] task_loss_metric_id must not be empty");
        ensure!(!score.task_loss_unit.trim().is_empty(), "runs[{index}] task_loss_unit must not be empty");
        validate_digest(&score.shared_stream_sha256, "shared_stream_sha256")?;
        validate_score_digest(score, index)?;
        validate_score_values(score, index)?;

        if let Some(metric) = &task_loss_metric_id {
            ensure!(metric == &score.task_loss_metric_id, "task_loss_metric_id differs across panel records");
        } else {
            task_loss_metric_id = Some(score.task_loss_metric_id.clone());
        }
        if let Some(unit) = &task_loss_unit {
            ensure!(unit == &score.task_loss_unit, "task_loss_unit differs across panel records");
        } else {
            task_loss_unit = Some(score.task_loss_unit.clone());
        }

        let key = (score.dataset_id.clone(), score.seed, score.scenario_id.clone());
        ensure!(
            observed_keys.insert(key.clone()),
            "duplicate panel record at dataset={}, seed={}, scenario={}",
            key.0, key.1, key.2
        );
        for (arm, expected_name) in arm_views(score) {
            ensure!(arm.name == expected_name, "runs[{index}] has misnamed policy arm {expected_name}");
            validate_arm_metrics(arm, expected_name)?;
            ensure!(
                arm.exogenous_stream_sha256 == score.shared_stream_sha256,
                "runs[{index}] arm {expected_name} stream digest differs from shared stream digest"
            );
            match arm_registry.get_mut(expected_name) {
                Some(registry) => {
                    ensure!(
                        registry.version == arm.version,
                        "policy arm {expected_name} version changes across panel records"
                    );
                    ensure!(
                        registry.repository_head_sha == arm.repository_head_sha,
                        "policy arm {expected_name} repository head changes across panel records"
                    );
                    match registry.config_by_dataset.get(&score.dataset_id) {
                        Some(digest) => ensure!(
                            digest == &arm.configuration_digest,
                            "policy arm {expected_name} configuration changes within dataset {}",
                            score.dataset_id
                        ),
                        None => {
                            registry.config_by_dataset.insert(score.dataset_id.clone(), arm.configuration_digest.clone());
                        }
                    }
                }
                None => {
                    arm_registry.insert(expected_name.to_owned(), ArmRegistry {
                        version: arm.version.clone(),
                        repository_head_sha: arm.repository_head_sha.clone(),
                        config_by_dataset: BTreeMap::from([
                            (score.dataset_id.clone(), arm.configuration_digest.clone())
                        ]),
                    });
                }
            }
        }
    }

    let mut missing_cells = Vec::new();
    for dataset_id in &input.expected_dataset_ids {
        for seed in &input.expected_seed_ids {
            for scenario_id in &input.expected_scenario_ids {
                if !observed_keys.contains(&(dataset_id.clone(), *seed, scenario_id.clone())) {
                    missing_cells.push(MissingPanelCell {
                        dataset_id: dataset_id.clone(),
                        seed: *seed,
                        scenario_id: scenario_id.clone(),
                    });
                }
            }
        }
    }

    let expected_run_count = REQUIRED_DATASET_COUNT * REQUIRED_SEED_COUNT * REQUIRED_SCENARIO_IDS.len();
    let complete_coverage_gate_pass = missing_cells.is_empty() && input.runs.len() == expected_run_count;
    let metric_id = "post_shift_mean_task_loss".to_owned();
    let metric_unit = task_loss_unit.unwrap_or_else(|| "unavailable-no-run-records".to_owned());

    let primary_comparison = if complete_coverage_gate_pass {
        let staged_registry = arm_registry.get("staged-verify-rollback")
            .context("complete panel is missing staged-verify-rollback arm metadata")?;
        let detector_registry = arm_registry.get("detector-triggered")
            .context("complete panel is missing detector-triggered arm metadata")?;
        ensure!(
            staged_registry.config_by_dataset.len() == REQUIRED_DATASET_COUNT
                && detector_registry.config_by_dataset.len() == REQUIRED_DATASET_COUNT,
            "candidate and baseline configuration provenance must cover every dataset"
        );
        let candidate_configuration_digest = sha256_hex(&serde_json::to_vec(&staged_registry.config_by_dataset)?);
        let baseline_configuration_digest = sha256_hex(&serde_json::to_vec(&detector_registry.config_by_dataset)?);
        let pairs = input.runs.iter().map(|score| PairedObservation {
            dataset_id: score.dataset_id.clone(),
            seed: score.seed,
            scenario_id: score.scenario_id.clone(),
            candidate_value: score.staged_verify_rollback.post_shift_mean_task_loss,
            baseline_value: score.detector_triggered.post_shift_mean_task_loss,
        }).collect();

        Some(compare(ComparisonInput {
            schema_version: 1,
            metric: MetricSpec {
                id: metric_id.clone(),
                unit: metric_unit.clone(),
                higher_is_better: false,
            },
            candidate: ComparisonArm {
                name: "staged-verify-rollback".to_owned(),
                version: staged_registry.version.clone(),
                repository_head_sha: staged_registry.repository_head_sha.clone(),
                configuration_digest: candidate_configuration_digest,
                comparison_track: input.comparison_track.clone(),
            },
            baseline: ComparisonArm {
                name: "detector-triggered".to_owned(),
                version: detector_registry.version.clone(),
                repository_head_sha: detector_registry.repository_head_sha.clone(),
                configuration_digest: baseline_configuration_digest,
                comparison_track: input.comparison_track.clone(),
            },
            bootstrap: input.bootstrap.clone(),
            pairs,
        })?)
    } else {
        None
    };

    let mut result = AdaptationPanelResult {
        result_schema_version: 1,
        metric_version: "cognitive-adaptation-panel-v1".to_owned(),
        canonical_input_sha256,
        result_sha256: None,
        comparison_track: input.comparison_track,
        metric_id,
        metric_unit,
        expected_dataset_count: REQUIRED_DATASET_COUNT,
        observed_dataset_count: observed_keys.iter().map(|key| key.0.as_str()).collect::<BTreeSet<_>>().len(),
        expected_seed_count_per_dataset: REQUIRED_SEED_COUNT,
        observed_run_count: input.runs.len(),
        expected_run_count,
        missing_cell_count: missing_cells.len(),
        missing_cells,
        complete_coverage_gate_pass,
        primary_comparison,
        interpretation_guard: "The primary comparison is emitted only for the complete registered dataset × seed × scenario grid. Incomplete panels report missing cells and no confirmatory effect estimate. A complete-coverage pass is necessary, not sufficient, for SOTA claims. Scores are summary artifacts; this aggregator does not prove causal replay execution or scientific validity.".to_owned(),
    };
    result.result_sha256 = Some(panel_result_digest(&result)?);
    Ok(result)
}

fn arm_views(score: &CounterfactualUtilityScore) -> [(&ArmUtilityMetrics, &'static str); 5] {
    [
        (&score.no_adapt, "no-adapt"),
        (&score.detector_triggered, "detector-triggered"),
        (&score.detector_benefit_gated, "detector-benefit-gated"),
        (&score.staged_verify_rollback, "staged-verify-rollback"),
        (&score.oracle_timed, "oracle-timed"),
    ]
}

fn validate_score_digest(score: &CounterfactualUtilityScore, index: usize) -> anyhow::Result<()> {
    let supplied = score.result_sha256.as_deref()
        .context(format!("runs[{index}] result_sha256 is required"))?;
    validate_digest(supplied, "result_sha256")?;
    let mut copy = score.clone();
    copy.result_sha256 = None;
    let actual = sha256_hex(&serde_json::to_vec(&copy)?);
    ensure!(actual == supplied, "runs[{index}] result_sha256 does not match the scored record");
    Ok(())
}

fn validate_score_values(score: &CounterfactualUtilityScore, index: usize) -> anyhow::Result<()> {
    for value in [
        score.no_adapt.post_shift_mean_task_loss,
        score.detector_triggered.post_shift_mean_task_loss,
        score.detector_benefit_gated.post_shift_mean_task_loss,
        score.staged_verify_rollback.post_shift_mean_task_loss,
        score.oracle_timed.post_shift_mean_task_loss,
        score.staged_post_shift_task_loss_improvement_vs_detector_triggered,
    ] {
        ensure!(value.is_finite(), "runs[{index}] contains a non-finite primary metric");
    }
    let recomputed = score.detector_triggered.post_shift_mean_task_loss
        - score.staged_verify_rollback.post_shift_mean_task_loss;
    ensure!(
        (recomputed - score.staged_post_shift_task_loss_improvement_vs_detector_triggered).abs()
            <= 1e-10 * recomputed.abs().max(1.0),
        "runs[{index}] primary effect field does not match its arm outcomes"
    );
    Ok(())
}

fn validate_arm_metrics(arm: &ArmUtilityMetrics, expected_name: &str) -> anyhow::Result<()> {
    ensure!(!arm.version.trim().is_empty(), "{expected_name}.version must not be empty");
    ensure!(
        (arm.repository_head_sha.len() == 40 || arm.repository_head_sha.len() == 64)
            && arm.repository_head_sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "{expected_name}.repository_head_sha must be a full hexadecimal commit hash"
    );
    validate_digest(&arm.configuration_digest, "configuration_digest")?;
    validate_digest(&arm.exogenous_stream_sha256, "exogenous_stream_sha256")?;
    ensure!(arm.cycle_count > 0, "{expected_name}.cycle_count must be positive");
    for value in [
        arm.mean_task_loss,
        arm.post_shift_mean_task_loss,
        arm.total_adaptation_cost_units,
        arm.post_shift_loss_area_above_threshold,
    ] {
        ensure!(value.is_finite() && value >= 0.0, "{expected_name} has an invalid metric value");
    }
    ensure!(
        arm.recovery_delay_cycles.is_none() || arm.recovery_cycle.is_some(),
        "{expected_name}.recovery_delay_cycles cannot exist without recovery_cycle"
    );
    Ok(())
}

fn validate_unique_nonempty(values: &[String], expected_len: usize, label: &str) -> anyhow::Result<()> {
    ensure!(values.len() == expected_len, "{label} must contain exactly {expected_len} values");
    let mut unique = BTreeSet::new();
    for value in values {
        ensure!(!value.trim().is_empty(), "{label} entries must not be empty");
        ensure!(unique.insert(value.as_str()), "{label} entries must be unique");
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "{label} must be a lowercase 64-character SHA-256 digest"
    );
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn panel_result_digest(result: &AdaptationPanelResult) -> anyhow::Result<String> {
    let mut copy = serde_json::to_value(result)?;
    copy["result_sha256"] = serde_json::Value::Null;
    Ok(sha256_hex(&serde_json::to_vec(&copy)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_grid() -> (Vec<String>, Vec<u64>, Vec<String>) {
        (
            (0..7).map(|index| format!("dataset-{}", (b'a' + index as u8) as char)).collect(),
            (0..50).collect(),
            REQUIRED_SCENARIO_IDS.iter().map(|value| (*value).to_owned()).collect(),
        )
    }

    fn empty_panel() -> AdaptationPanelInput {
        let (expected_dataset_ids, expected_seed_ids, expected_scenario_ids) = expected_grid();
        AdaptationPanelInput {
            schema_version: 1,
            comparison_track: "valid-pe-only".to_owned(),
            expected_dataset_ids,
            expected_seed_ids,
            expected_scenario_ids,
            bootstrap: BootstrapSpec { replicates: 1_000, seed: 20261024 },
            runs: Vec::new(),
        }
    }

    fn hash(text: &str) -> String { sha256_hex(text.as_bytes()) }
    fn head(c: char) -> String { c.to_string().repeat(40) }

    fn arm_metrics(name: &str, dataset_id: &str, stream_sha: &str, post_mean: f64, cost: f64) -> ArmUtilityMetrics {
        let version = "fixture-v2".to_owned();
        let repository_head_sha = match name {
            "no-adapt" => head('a'),
            "detector-triggered" => head('b'),
            "detector-benefit-gated" => head('c'),
            "staged-verify-rollback" => head('d'),
            _ => head('e'),
        };
        let configuration_digest = hash(&format!("{dataset_id}:{name}:config"));
        ArmUtilityMetrics {
            name: name.to_owned(),
            version,
            exogenous_stream_sha256: stream_sha.to_owned(),
            repository_head_sha,
            configuration_digest,
            cycle_count: 4,
            cumulative_task_loss: 0.1 + 3.0 * post_mean,
            mean_task_loss: (0.1 + 3.0 * post_mean) / 4.0,
            pre_shift_mean_task_loss: 0.1,
            post_shift_mean_task_loss: post_mean,
            post_shift_cumulative_task_loss: 3.0 * post_mean,
            post_shift_loss_area_above_threshold: 3.0 * (post_mean - 0.35).max(0.0),
            recovery_cycle: if post_mean <= 0.35 { Some(1) } else { None },
            recovery_delay_cycles: if post_mean <= 0.35 { Some(0) } else { None },
            total_adaptation_cost_units: cost,
        }
    }

    fn score(dataset_id: &str, seed: u64, scenario_id: &str) -> CounterfactualUtilityScore {
        let stream = hash(&format!("{dataset_id}:{seed}:{scenario_id}:stream"));
        let no_adapt = arm_metrics("no-adapt", dataset_id, &stream, 0.5, 0.0);
        let detector_triggered = arm_metrics("detector-triggered", dataset_id, &stream, 0.4, 0.7);
        let detector_benefit_gated = arm_metrics("detector-benefit-gated", dataset_id, &stream, 0.32, 0.35);
        let staged_verify_rollback = arm_metrics("staged-verify-rollback", dataset_id, &stream, 0.3, 0.25);
        let oracle_timed = arm_metrics("oracle-timed", dataset_id, &stream, 0.2, 0.9);
        let mut result = CounterfactualUtilityScore {
            metric_version: "counterfactual-adaptation-utility-v2".to_owned(),
            canonical_input_sha256: hash(&format!("{dataset_id}:{seed}:{scenario_id}:input")),
            result_sha256: None,
            dataset_id: dataset_id.to_owned(),
            stream_id: format!("{dataset_id}-{seed}-{scenario_id}"),
            scenario_id: scenario_id.to_owned(),
            comparison_track: "valid-pe-only".to_owned(),
            shared_stream_sha256: stream,
            seed,
            task_loss_metric_id: "next-step-loss".to_owned(),
            task_loss_unit: "normalized-loss".to_owned(),
            no_adapt,
            detector_triggered,
            detector_benefit_gated,
            staged_verify_rollback,
            oracle_timed,
            detector_task_loss_improvement_vs_no_adapt: 0.075,
            detector_task_loss_regret_vs_oracle_timed: 0.15,
            detector_post_shift_task_loss_improvement_vs_no_adapt: 0.1,
            detector_post_shift_task_loss_regret_vs_oracle_timed: 0.2,
            detector_benefit_gated_post_shift_task_loss_improvement_vs_no_adapt: 0.18,
            staged_post_shift_task_loss_improvement_vs_no_adapt: 0.2,
            staged_post_shift_task_loss_improvement_vs_detector_triggered: 0.1,
            staged_post_shift_task_loss_regret_vs_oracle_timed: 0.1,
            detector_adaptation_cost_delta_vs_no_adapt: 0.7,
            detector_benefit_gated_adaptation_cost_delta_vs_no_adapt: 0.35,
            staged_adaptation_cost_delta_vs_no_adapt: 0.25,
            staged_adaptation_cost_delta_vs_detector_triggered: -0.45,
            staged_dominated_by_detector_triggered: false,
            detector_benefit_gated_dominated_by_no_adapt: false,
            staged_dominated_by_no_adapt: false,
            detector_dominated_by_no_adapt: false,
            adaptation_cost_units_per_task_loss_unit_reduced: Some(7.0 / 0.3),
            interpretation_guard: "synthetic test record".to_owned(),
        };
        let mut copy = result.clone();
        copy.result_sha256 = None;
        result.result_sha256 = Some(sha256_hex(&serde_json::to_vec(&copy).unwrap()));
        result
    }

    #[test]
    fn incomplete_panel_fails_closed_and_lists_all_missing_cells() {
        let result = evaluate(empty_panel()).expect("incomplete panel returns a coverage report");
        assert!(!result.complete_coverage_gate_pass);
        assert_eq!(result.expected_run_count, 2_800);
        assert_eq!(result.observed_run_count, 0);
        assert_eq!(result.missing_cell_count, 2_800);
        assert_eq!(result.missing_cells.len(), 2_800);
        assert!(result.primary_comparison.is_none());
        assert_eq!(result.result_sha256.as_ref().map(String::len), Some(64));
    }

    #[test]
    fn complete_panel_aggregates_paired_post_shift_effect_and_uses_dataset_clustered_bootstrap() {
        let mut input = empty_panel();
        for dataset_id in &input.expected_dataset_ids.clone() {
            for seed in &input.expected_seed_ids.clone() {
                for scenario_id in &input.expected_scenario_ids.clone() {
                    input.runs.push(score(dataset_id, *seed, scenario_id));
                }
            }
        }
        let result = evaluate(input).expect("complete panel passes structural checks");
        assert!(result.complete_coverage_gate_pass);
        assert_eq!(result.observed_run_count, 2_800);
        assert_eq!(result.missing_cell_count, 0);
        let comparison = result.primary_comparison.expect("complete panel emits primary comparison");
        assert_eq!(comparison.dataset_count, 7);
        assert_eq!(comparison.paired_record_count, 2_800);
        assert!(comparison.sota_coverage_gate_pass);
        assert!((comparison.dataset_macro_paired_effect - 0.1).abs() < 1e-12);
        assert!(comparison.interval_supports_positive_effect);
    }

    #[test]
    fn one_missing_seed_scenario_cell_suppresses_primary_comparison() {
        let mut input = empty_panel();
        for dataset_id in input.expected_dataset_ids.clone() {
            for seed in input.expected_seed_ids.clone() {
                for scenario_id in input.expected_scenario_ids.clone() {
                    if dataset_id == "dataset-g" && seed == 49 && scenario_id == REQUIRED_SCENARIO_IDS[7] {
                        continue;
                    }
                    input.runs.push(score(&dataset_id, seed, &scenario_id));
                }
            }
        }
        let result = evaluate(input).expect("incomplete grid returns coverage report");
        assert!(!result.complete_coverage_gate_pass);
        assert_eq!(result.observed_run_count, 2_799);
        assert_eq!(result.missing_cell_count, 1);
        assert_eq!(result.missing_cells[0].dataset_id, "dataset-g");
        assert_eq!(result.missing_cells[0].seed, 49);
        assert_eq!(result.missing_cells[0].scenario_id, REQUIRED_SCENARIO_IDS[7]);
        assert!(result.primary_comparison.is_none());
    }

    #[test]
    fn rejects_scores_from_another_access_track() {
        let mut input = empty_panel();
        let mut record = score("dataset-a", 0, REQUIRED_SCENARIO_IDS[0]);
        record.comparison_track = "observed-outcome".to_owned();
        let mut copy = record.clone();
        copy.result_sha256 = None;
        record.result_sha256 = Some(sha256_hex(&serde_json::to_vec(&copy).unwrap()));
        input.runs.push(record);
        assert!(evaluate(input).unwrap_err().to_string().contains("comparison_track differs"));
    }

    #[test]
    fn rejects_duplicate_panel_cells() {
        let mut input = empty_panel();
        input.runs = vec![
            score("dataset-a", 0, REQUIRED_SCENARIO_IDS[0]),
            score("dataset-a", 0, REQUIRED_SCENARIO_IDS[0]),
        ];
        assert!(evaluate(input).unwrap_err().to_string().contains("duplicate panel record"));
    }

    #[test]
    fn rejects_tampered_scored_record_digest() {
        let mut input = empty_panel();
        let mut record = score("dataset-a", 0, REQUIRED_SCENARIO_IDS[0]);
        record.staged_verify_rollback.post_shift_mean_task_loss = 0.99;
        input.runs.push(record);
        assert!(evaluate(input).unwrap_err().to_string().contains("result_sha256 does not match"));
    }

    #[test]
    fn rejects_primary_effect_that_disagrees_with_arm_metrics_even_if_record_is_rehashed() {
        let mut input = empty_panel();
        let mut record = score("dataset-a", 0, REQUIRED_SCENARIO_IDS[0]);
        record.staged_post_shift_task_loss_improvement_vs_detector_triggered = -99.0;
        let mut copy = record.clone();
        copy.result_sha256 = None;
        record.result_sha256 = Some(sha256_hex(&serde_json::to_vec(&copy).unwrap()));
        input.runs.push(record);
        assert!(evaluate(input).unwrap_err().to_string().contains("primary effect field"));
    }

    #[test]
    fn rejects_policy_revision_drift_across_runs() {
        let mut input = empty_panel();
        let first = score("dataset-a", 0, REQUIRED_SCENARIO_IDS[0]);
        let mut second = score("dataset-a", 1, REQUIRED_SCENARIO_IDS[0]);
        second.staged_verify_rollback.repository_head_sha = head('f');
        let mut copy = second.clone();
        copy.result_sha256 = None;
        second.result_sha256 = Some(sha256_hex(&serde_json::to_vec(&copy).unwrap()));
        input.runs = vec![first, second];
        assert!(evaluate(input).unwrap_err().to_string().contains("repository head changes"));
    }
}
