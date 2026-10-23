//! Counterfactual utility scoring for adaptation policies.
//!
//! Drift-event F1 measures alarms, not whether acting on those alarms helps the task.
//! This evaluator compares three replay arms on the same pinned exogenous stream:
//! no adaptation, detector-triggered adaptation, and ground-truth-timed adaptation.
//! It reports task loss and intervention cost separately; it never hides the trade-off
//! inside an unregistered weighted aggregate.

use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualUtilityInput {
    pub schema_version: u32,
    pub dataset_id: String,
    pub stream_id: String,
    pub scenario_id: String,
    pub shared_stream_sha256: String,
    pub seed: u64,
    pub task_loss_metric_id: String,
    pub task_loss_unit: String,
    pub no_adapt: CounterfactualArm,
    pub detector_triggered: CounterfactualArm,
    pub oracle_timed: CounterfactualArm,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualArm {
    pub name: String,
    pub version: String,
    pub repository_head_sha: String,
    pub configuration_digest: String,
    pub observations: Vec<ArmObservation>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArmObservation {
    pub cycle_index: u64,
    pub task_loss: f64,
    /// Observed adaptation resource cost, in the manifest-declared cost unit.
    pub adaptation_cost_units: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ArmUtilityMetrics {
    pub name: String,
    pub version: String,
    pub repository_head_sha: String,
    pub configuration_digest: String,
    pub cycle_count: usize,
    pub cumulative_task_loss: f64,
    pub mean_task_loss: f64,
    pub total_adaptation_cost_units: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CounterfactualUtilityScore {
    pub metric_version: String,
    pub canonical_input_sha256: String,
    /// SHA-256 of serialized result with this field set to null.
    pub result_sha256: Option<String>,
    pub dataset_id: String,
    pub stream_id: String,
    pub scenario_id: String,
    pub shared_stream_sha256: String,
    pub seed: u64,
    pub task_loss_metric_id: String,
    pub task_loss_unit: String,
    pub no_adapt: ArmUtilityMetrics,
    pub detector_triggered: ArmUtilityMetrics,
    pub oracle_timed: ArmUtilityMetrics,
    /// Positive means detector-triggered adaptation lowers task loss vs no adaptation.
    pub detector_task_loss_improvement_vs_no_adapt: f64,
    /// Positive means detector-triggered adaptation has more task loss than oracle timing.
    pub detector_task_loss_regret_vs_oracle_timed: f64,
    /// Positive means the detector policy costs more resources than no adaptation.
    pub detector_adaptation_cost_delta_vs_no_adapt: f64,
    /// True when no adaptation is no worse on task loss and uses strictly fewer resources.
    pub detector_dominated_by_no_adapt: bool,
    /// Resource units spent per total task-loss unit reduced; absent when there is no gain.
    pub adaptation_cost_units_per_task_loss_unit_reduced: Option<f64>,
    pub interpretation_guard: String,
}

pub fn evaluate_file(path: &Path) -> anyhow::Result<CounterfactualUtilityScore> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading counterfactual utility input {}", path.display()))?;
    let input: CounterfactualUtilityInput = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing counterfactual utility input {}", path.display()))?;
    score_counterfactual_utility(input)
}

pub fn score_counterfactual_utility(
    mut input: CounterfactualUtilityInput,
) -> anyhow::Result<CounterfactualUtilityScore> {
    ensure!(input.schema_version == 1, "counterfactual utility schema_version must be 1");
    nonempty(&input.dataset_id, "dataset_id")?;
    nonempty(&input.stream_id, "stream_id")?;
    nonempty(&input.scenario_id, "scenario_id")?;
    nonempty(&input.task_loss_metric_id, "task_loss_metric_id")?;
    nonempty(&input.task_loss_unit, "task_loss_unit")?;
    validate_digest(&input.shared_stream_sha256, 64, "shared_stream_sha256")?;
    ensure!(
        input.no_adapt.name == "no-adapt",
        "no_adapt arm name must be 'no-adapt'"
    );
    ensure!(
        input.detector_triggered.name == "detector-triggered",
        "detector_triggered arm name must be 'detector-triggered'"
    );
    ensure!(
        input.oracle_timed.name == "oracle-timed",
        "oracle_timed arm name must be 'oracle-timed'"
    );

    validate_arm(&mut input.no_adapt, "no_adapt")?;
    validate_arm(&mut input.detector_triggered, "detector_triggered")?;
    validate_arm(&mut input.oracle_timed, "oracle_timed")?;

    let cycle_count = input.no_adapt.observations.len();
    ensure!(cycle_count > 0, "counterfactual arms must contain at least one observation");
    ensure!(
        input.detector_triggered.observations.len() == cycle_count
            && input.oracle_timed.observations.len() == cycle_count,
        "all counterfactual arms must cover the same number of cycles"
    );
    for index in 0..cycle_count {
        let expected = index as u64;
        ensure!(
            input.no_adapt.observations[index].cycle_index == expected
                && input.detector_triggered.observations[index].cycle_index == expected
                && input.oracle_timed.observations[index].cycle_index == expected,
            "counterfactual arms must share the same contiguous cycle panel starting at zero"
        );
        ensure!(
            input.no_adapt.observations[index].adaptation_cost_units == 0.0,
            "no-adapt arm must have zero adaptation cost at every cycle"
        );
    }

    let canonical_input_sha256 = hex_sha256(&serde_json::to_vec(&input)?);
    let no_adapt = arm_metrics(&input.no_adapt)?;
    let detector_triggered = arm_metrics(&input.detector_triggered)?;
    let oracle_timed = arm_metrics(&input.oracle_timed)?;

    let improvement =
        no_adapt.mean_task_loss - detector_triggered.mean_task_loss;
    let regret =
        detector_triggered.mean_task_loss - oracle_timed.mean_task_loss;
    let cost_delta =
        detector_triggered.total_adaptation_cost_units - no_adapt.total_adaptation_cost_units;
    ensure!(
        improvement.is_finite() && regret.is_finite() && cost_delta.is_finite(),
        "counterfactual metric calculation overflowed"
    );

    let total_loss_reduction = improvement * cycle_count as f64;
    ensure!(
        total_loss_reduction.is_finite(),
        "counterfactual total loss reduction overflowed"
    );
    let adaptation_cost_units_per_task_loss_unit_reduced =
        if total_loss_reduction > 0.0 {
            let ratio = detector_triggered.total_adaptation_cost_units / total_loss_reduction;
            ensure!(ratio.is_finite(), "adaptation-cost efficiency ratio overflowed");
            Some(ratio)
        } else {
            None
        };

    let mut result = CounterfactualUtilityScore {
        metric_version: "counterfactual-adaptation-utility-v1".to_owned(),
        canonical_input_sha256,
        result_sha256: None,
        dataset_id: input.dataset_id,
        stream_id: input.stream_id,
        scenario_id: input.scenario_id,
        shared_stream_sha256: input.shared_stream_sha256,
        seed: input.seed,
        task_loss_metric_id: input.task_loss_metric_id,
        task_loss_unit: input.task_loss_unit,
        no_adapt,
        detector_triggered,
        oracle_timed,
        detector_task_loss_improvement_vs_no_adapt: improvement,
        detector_task_loss_regret_vs_oracle_timed: regret,
        detector_adaptation_cost_delta_vs_no_adapt: cost_delta,
        detector_dominated_by_no_adapt: detector_triggered.mean_task_loss
            >= no_adapt.mean_task_loss
            && detector_triggered.total_adaptation_cost_units
                > no_adapt.total_adaptation_cost_units,
        adaptation_cost_units_per_task_loss_unit_reduced,
        interpretation_guard: "Counterfactual task-loss and resource-cost results only; oracle-timed is a ground-truth-timing reference, not a deployable detector. Shared-stream digest and paired arms do not by themselves prove identical hidden state or implementation correctness. Do not collapse metrics into a composite score without a preregistered cost conversion.".to_owned(),
    };
    result.result_sha256 = Some(result_digest(&result)?);
    Ok(result)
}

fn validate_arm(arm: &mut CounterfactualArm, context: &str) -> anyhow::Result<()> {
    nonempty(&arm.name, &format!("{context}.name"))?;
    nonempty(&arm.version, &format!("{context}.version"))?;
    validate_commit_sha(&arm.repository_head_sha, &format!("{context}.repository_head_sha"))?;
    validate_digest(&arm.configuration_digest, 64, &format!("{context}.configuration_digest"))?;
    ensure!(!arm.observations.is_empty(), "{context}.observations must not be empty");
    for observation in &arm.observations {
        ensure!(
            observation.task_loss.is_finite() && observation.task_loss >= 0.0,
            "{context} task_loss must be finite and non-negative"
        );
        ensure!(
            observation.adaptation_cost_units.is_finite()
                && observation.adaptation_cost_units >= 0.0,
            "{context} adaptation cost must be finite and non-negative"
        );
    }
    arm.observations.sort_by_key(|o| o.cycle_index);
    for pair in arm.observations.windows(2) {
        ensure!(
            pair[0].cycle_index != pair[1].cycle_index,
            "{context} has duplicate cycle_index {}",
            pair[0].cycle_index
        );
    }
    Ok(())
}

fn arm_metrics(arm: &CounterfactualArm) -> anyhow::Result<ArmUtilityMetrics> {
    let cumulative_task_loss = checked_sum(
        arm.observations.iter().map(|o| o.task_loss),
        &format!("{} cumulative task loss", arm.name),
    )?;
    let total_adaptation_cost_units = checked_sum(
        arm.observations.iter().map(|o| o.adaptation_cost_units),
        &format!("{} adaptation cost", arm.name),
    )?;
    let mean_task_loss = cumulative_task_loss / arm.observations.len() as f64;
    ensure!(
        mean_task_loss.is_finite(),
        "{} mean task loss is not finite",
        arm.name
    );
    Ok(ArmUtilityMetrics {
        name: arm.name.clone(),
        version: arm.version.clone(),
        repository_head_sha: arm.repository_head_sha.clone(),
        configuration_digest: arm.configuration_digest.clone(),
        cycle_count: arm.observations.len(),
        cumulative_task_loss,
        mean_task_loss,
        total_adaptation_cost_units,
    })
}

fn checked_sum(values: impl Iterator<Item = f64>, label: &str) -> anyhow::Result<f64> {
    let mut sum = 0.0;
    for value in values {
        sum += value;
        ensure!(sum.is_finite(), "{label} overflowed");
    }
    Ok(sum)
}

fn result_digest(result: &CounterfactualUtilityScore) -> anyhow::Result<String> {
    let mut copy = result.clone();
    copy.result_sha256 = None;
    Ok(hex_sha256(&serde_json::to_vec(&copy)?))
}

fn validate_commit_sha(value: &str, context: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == 40 || value.len() == 64,
        "{context} must be a full 40- or 64-character commit hash"
    );
    ensure!(
        value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{context} must be hexadecimal"
    );
    Ok(())
}

fn validate_digest(value: &str, length: usize, context: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{context} must be a {length}-character hexadecimal SHA-256 digest"
    );
    Ok(())
}

fn nonempty(value: &str, context: &str) -> anyhow::Result<()> {
    ensure!(!value.trim().is_empty(), "{context} must not be empty");
    Ok(())
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(name: &str, losses: &[f64], costs: &[f64]) -> CounterfactualArm {
        CounterfactualArm {
            name: name.to_owned(),
            version: format!("{name}-v1"),
            repository_head_sha: "a".repeat(40),
            configuration_digest: "b".repeat(64),
            observations: losses
                .iter()
                .zip(costs.iter())
                .enumerate()
                .map(|(index, (&task_loss, &adaptation_cost_units))| ArmObservation {
                    cycle_index: index as u64,
                    task_loss,
                    adaptation_cost_units,
                })
                .collect(),
        }
    }

    fn fixture() -> CounterfactualUtilityInput {
        CounterfactualUtilityInput {
            schema_version: 1,
            dataset_id: "dataset-a".to_owned(),
            stream_id: "stream-17".to_owned(),
            scenario_id: "gradual-drift-recovery".to_owned(),
            shared_stream_sha256: "c".repeat(64),
            seed: 17,
            task_loss_metric_id: "one-step-prediction-loss".to_owned(),
            task_loss_unit: "normalized-loss".to_owned(),
            no_adapt: arm("no-adapt", &[0.1, 0.5, 0.9, 0.7], &[0.0; 4]),
            detector_triggered: arm(
                "detector-triggered",
                &[0.1, 0.4, 0.3, 0.3],
                &[0.0, 0.2, 0.4, 0.1],
            ),
            oracle_timed: arm(
                "oracle-timed",
                &[0.1, 0.2, 0.2, 0.2],
                &[0.0, 0.4, 0.4, 0.1],
            ),
        }
    }

    #[test]
    fn reports_task_loss_and_resource_cost_as_separate_axes() {
        let result = score_counterfactual_utility(fixture()).expect("valid counterfactual input");
        assert!((result.no_adapt.mean_task_loss - 0.55).abs() < 1e-12);
        assert!((result.detector_triggered.mean_task_loss - 0.275).abs() < 1e-12);
        assert!((result.oracle_timed.mean_task_loss - 0.175).abs() < 1e-12);
        assert!((result.detector_task_loss_improvement_vs_no_adapt - 0.275).abs() < 1e-12);
        assert!((result.detector_task_loss_regret_vs_oracle_timed - 0.1).abs() < 1e-12);
        assert!((result.detector_triggered.total_adaptation_cost_units - 0.7).abs() < 1e-12);
        assert!(!result.detector_dominated_by_no_adapt);
        assert_eq!(result.result_sha256.as_ref().map(String::len), Some(64));
    }

    #[test]
    fn detects_detector_policy_that_costs_resources_without_improving_loss() {
        let mut input = fixture();
        input.detector_triggered.observations = arm(
            "detector-triggered",
            &[0.2, 0.6, 0.9, 0.7],
            &[0.1, 0.1, 0.1, 0.1],
        )
        .observations;
        let result = score_counterfactual_utility(input).expect("valid input");
        assert!(result.detector_dominated_by_no_adapt);
        assert_eq!(result.adaptation_cost_units_per_task_loss_unit_reduced, None);
    }

    #[test]
    fn canonical_digest_is_invariant_to_observation_order() {
        let first = fixture();
        let mut second = first.clone();
        second.no_adapt.observations.reverse();
        second.detector_triggered.observations.reverse();
        second.oracle_timed.observations.reverse();
        let a = score_counterfactual_utility(first).expect("first scores");
        let b = score_counterfactual_utility(second).expect("second scores");
        assert_eq!(a.canonical_input_sha256, b.canonical_input_sha256);
        assert_eq!(a.result_sha256, b.result_sha256);
    }

    #[test]
    fn rejects_mismatched_cycle_panels_and_nonzero_no_adapt_cost() {
        let mut mismatched = fixture();
        mismatched.oracle_timed.observations[2].cycle_index = 9;
        assert!(score_counterfactual_utility(mismatched).is_err());

        let mut bad_baseline = fixture();
        bad_baseline.no_adapt.observations[1].adaptation_cost_units = 0.01;
        assert!(score_counterfactual_utility(bad_baseline)
            .unwrap_err()
            .to_string()
            .contains("zero adaptation cost"));
    }

    #[test]
    fn rejects_invalid_provenance_nonfinite_losses_and_duplicate_cycles() {
        let mut bad_provenance = fixture();
        bad_provenance.shared_stream_sha256 = "short".to_owned();
        assert!(score_counterfactual_utility(bad_provenance).is_err());

        let mut bad_loss = fixture();
        bad_loss.detector_triggered.observations[0].task_loss = f64::NAN;
        assert!(score_counterfactual_utility(bad_loss).is_err());

        let mut duplicate = fixture();
        duplicate.no_adapt.observations[1].cycle_index = 0;
        assert!(score_counterfactual_utility(duplicate).is_err());
    }
}
