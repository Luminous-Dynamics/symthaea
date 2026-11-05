//! Deterministic synthetic replay harness for the cognitive-adaptation evidence path.
//!
//! This is a smoke harness, not the production Symthaea replay engine. It owns one generated
//! exogenous stream and executes five scalar-regression policy controls against it. Every output
//! is tagged synthetic-not-measured and is ineligible for confirmatory panels. The harness tests
//! causal timing, transaction validity, score/receipt binding, and reproducible artifact plumbing.
use anyhow::{ensure, Context};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs, path::{Path, PathBuf}};

use crate::{
    cognitive_adaptation_utility::{
        self, ArmObservation, CounterfactualArm, CounterfactualUtilityInput, EvaluationWindows,
        CounterfactualUtilityScore,
    },
    cognitive_replay_receipt::{
        self, DeliveryRecord, ExogenousEvent, ReplayArm, ReplayDecision, ReplayReceipt,
        ReplayReceiptValidation,
    },
};

const REQUIRED_SCENARIOS: [&str; 4] = ["abrupt", "gradual", "no-impact", "recurring"];
const DATASET_ID: &str = "synthetic-scalar-regression-v1";
const POLICY_VERSION: &str = "synthetic-policy-v1";
const DELAYED_OUTCOME_TICKS: u64 = 1;

#[derive(Debug, Serialize)]
pub struct SyntheticReplaySummary {
    pub schema_version: u32,
    pub run_kind: String,
    pub scenario_id: String,
    pub comparison_track: String,
    pub seed: u64,
    pub ticks: usize,
    pub receipt_path: String,
    pub utility_input_path: String,
    pub utility_score_path: String,
    pub receipt_validation_path: String,
    pub receipt_sha256: String,
    pub score_sha256: String,
    pub no_adapt_post_shift_loss: f64,
    pub detector_post_shift_loss: f64,
    pub benefit_gated_post_shift_loss: f64,
    pub staged_post_shift_loss: f64,
    pub oracle_diagnostic_post_shift_loss: f64,
    pub staged_adaptation_cost_units: f64,
    pub interpretation_guard: String,
}

#[derive(Debug)]
struct GeneratedRun {
    receipt: ReplayReceipt,
    utility_input: CounterfactualUtilityInput,
}

#[derive(Debug, Clone, Copy)]
enum PolicyMode {
    NoAdapt,
    DetectorTriggered,
    BenefitGated,
    StagedVerifyRollback,
    OracleTimed,
}

impl PolicyMode {
    fn id(self) -> &'static str {
        match self {
            Self::NoAdapt => "no-adapt",
            Self::DetectorTriggered => "detector-triggered",
            Self::BenefitGated => "detector-benefit-gated",
            Self::StagedVerifyRollback => "staged-verify-rollback",
            Self::OracleTimed => "oracle-timed",
        }
    }

    fn uses_oracle(self) -> bool {
        matches!(self, Self::OracleTimed)
    }

    fn requires_out_of_sample_verification(self) -> bool {
        matches!(self, Self::StagedVerifyRollback)
    }
}

#[derive(Debug)]
struct PendingTransaction {
    snapshot: f64,
    candidate: f64,
    stage_tick: usize,
    next_evaluation_tick: usize,
    shadow_evaluations: Vec<(usize, f64, f64)>,
    /// Set by verify; the next decision commits or rolls back.
    verification_passed: Option<bool>,
}

fn repository_head_sha() -> String {
    let supplied = std::env::var("GITHUB_SHA").unwrap_or_default();
    if supplied.len() == 40 && supplied.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        supplied.to_ascii_lowercase()
    } else {
        // A standalone synthetic run has no trusted source commit; this zero sentinel is deliberate.
        "0".repeat(40)
    }
}

pub fn run_synthetic(
    scenario: &str,
    ticks: usize,
    seed: u64,
    out_dir: &Path,
) -> anyhow::Result<SyntheticReplaySummary> {
    let generated = build_synthetic_run(scenario, ticks, seed)?;
    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output directory {}", out_dir.display()))?;
    let receipt_path = out_dir.join("receipt.json");
    let utility_input_path = out_dir.join("utility-input.json");
    let score_path = out_dir.join("utility-score.json");
    let validation_path = out_dir.join("receipt-validation.json");

    write_json(&receipt_path, &generated.receipt)?;
    write_json(&utility_input_path, &generated.utility_input)?;
    let validation = cognitive_replay_receipt::evaluate_file(&receipt_path)
        .context("validating generated synthetic replay receipt")?;
    write_json(&validation_path, &validation)?;
    let score = cognitive_adaptation_utility::evaluate_file_with_receipt(
        &utility_input_path,
        &receipt_path,
    ).context("scoring generated synthetic replay with its raw receipt")?;
    write_json(&score_path, &score)?;

    ensure!(
        score.source_replay_run_kind.as_deref() == Some("synthetic-not-measured"),
        "synthetic replay harness must not emit measured run-kind provenance"
    );
    let summary = summary(
        scenario, seed, ticks, &receipt_path, &utility_input_path, &score_path,
        &validation_path, &validation, &score,
    );
    write_json(&out_dir.join("summary.json"), &summary)?;
    Ok(summary)
}

fn build_synthetic_run(scenario: &str, ticks: usize, seed: u64) -> anyhow::Result<GeneratedRun> {
    ensure!(
        REQUIRED_SCENARIOS.contains(&scenario),
        "unknown synthetic scenario {scenario}; expected one of {}",
        REQUIRED_SCENARIOS.join(", ")
    );
    ensure!((32..=4096).contains(&ticks), "ticks must be between 32 and 4096");
    let shift_tick = ticks / 2;
    let targets: Vec<f64> = (0..ticks)
        .map(|tick| target_at(scenario, tick, ticks, shift_tick, seed))
        .collect();
    let events: Vec<ExogenousEvent> = (0..ticks)
        .map(|tick| {
            let payload = serde_json::json!({
                "tick": tick,
                "feature_value": feature_at(scenario, tick, ticks, shift_tick),
                "feature_available_at_tick": tick,
            });
            ExogenousEvent {
                sequence_index: tick as u64,
                event_id: format!("event-{tick:06}"),
                observed_at_tick: tick as u64,
                available_at_tick: tick as u64,
                payload_sha256: sha256_hex(&serde_json::to_vec(&payload).expect("JSON payload serializes")),
            }
        })
        .collect();
    let event_ledger_sha256 = sha256_hex(&serde_json::to_vec(&events)?);
    let configuration = serde_json::json!({
        "runner": "synthetic-scalar-regression-v1",
        "scenario": scenario,
        "ticks": ticks,
        "seed": seed,
        "shift_tick": shift_tick,
        "outcome_delay_ticks": DELAYED_OUTCOME_TICKS,
        "threshold": 0.08,
        "no_impact_is_feature_only": true,
    });
    let transformation_config_sha256 = sha256_hex(&serde_json::to_vec(&configuration)?);
    let dataset_artifact_sha256 = sha256_hex(&serde_json::to_vec(&targets)?);
    let head_sha = repository_head_sha();
    let stream_id = format!("synthetic-{scenario}-seed-{seed}-ticks-{ticks}");
    let run_id = format!("synthetic-not-measured-{scenario}-seed-{seed}-ticks-{ticks}");
    let modes = [
        PolicyMode::NoAdapt,
        PolicyMode::DetectorTriggered,
        PolicyMode::BenefitGated,
        PolicyMode::StagedVerifyRollback,
        PolicyMode::OracleTimed,
    ];
    let mut replay_arms = Vec::with_capacity(modes.len());
    let mut scored_arms = Vec::with_capacity(modes.len());
    for mode in modes {
        let config_digest = sha256_hex(&serde_json::to_vec(&serde_json::json!({
            "policy": mode.id(),
            "policy_version": POLICY_VERSION,
            "threshold": 0.08,
            "verification": if mode.requires_out_of_sample_verification() { "two-shadow-outcomes" } else { "historical-evidence" },
            "oracle_channel": mode.uses_oracle(),
        }))?);
        let execution = simulate_policy(
            mode, scenario, seed, &targets, shift_tick, &events, &event_ledger_sha256,
            &head_sha, &config_digest,
        )?;
        scored_arms.push(execution.scored_arm);
        replay_arms.push(execution.replay_arm);
    }

    let receipt = ReplayReceipt {
        schema_version: 2,
        run_id,
        run_kind: "synthetic-not-measured".to_owned(),
        dataset_id: DATASET_ID.to_owned(),
        dataset_artifact_sha256,
        transformation_config_sha256: transformation_config_sha256.clone(),
        stream_id: stream_id.clone(),
        stream_length_ticks: ticks as u64,
        replay_driver_repository_head_sha: head_sha,
        replay_driver_configuration_digest: transformation_config_sha256,
        environment_seed: seed,
        event_ledger_sha256: event_ledger_sha256.clone(),
        events,
        arms: replay_arms,
    };
    cognitive_replay_receipt::validate_receipt(receipt.clone())
        .context("generated synthetic receipt failed the receipt contract")?;

    let post_shift_start = (shift_tick + if scenario == "gradual" { 9 } else { 2 })
        .min(ticks.saturating_sub(2));
    let utility_input = CounterfactualUtilityInput {
        schema_version: 3,
        dataset_id: DATASET_ID.to_owned(),
        stream_id,
        scenario_id: scenario.to_owned(),
        comparison_track: "observed-outcome".to_owned(),
        shared_stream_sha256: event_ledger_sha256,
        seed,
        task_loss_metric_id: "synthetic-squared-error".to_owned(),
        task_loss_unit: "squared-target-units".to_owned(),
        evaluation_windows: EvaluationWindows {
            pre_shift_start_cycle: shift_tick.saturating_sub(10) as u64,
            pre_shift_end_cycle: shift_tick as u64,
            post_shift_start_cycle: post_shift_start as u64,
            post_shift_end_cycle: ticks as u64,
            recovery_loss_threshold: 0.05,
            recovery_consecutive_cycles: 3,
        },
        no_adapt: scored_arms.remove(0),
        detector_triggered: scored_arms.remove(0),
        detector_benefit_gated: scored_arms.remove(0),
        staged_verify_rollback: scored_arms.remove(0),
        oracle_timed: scored_arms.remove(0),
    };
    Ok(GeneratedRun { receipt, utility_input })
}

struct PolicyExecution {
    replay_arm: ReplayArm,
    scored_arm: CounterfactualArm,
}

fn simulate_policy(
    mode: PolicyMode,
    scenario: &str,
    seed: u64,
    targets: &[f64],
    shift_tick: usize,
    events: &[ExogenousEvent],
    event_ledger_sha256: &str,
    repository_head_sha: &str,
    config_digest: &str,
) -> anyhow::Result<PolicyExecution> {
    let ticks = targets.len();
    let mut estimate = 0.0_f64;
    let mut attempted = false;
    let mut pending: Option<PendingTransaction> = None;
    let mut losses: Vec<f64> = Vec::with_capacity(ticks);
    let mut observations = Vec::with_capacity(ticks);
    let mut decisions = Vec::with_capacity(ticks);
    let deliveries = events.iter().map(|event| DeliveryRecord {
        event_id: event.event_id.clone(),
        delivered_at_tick: event.available_at_tick,
    }).collect::<Vec<_>>();

    for tick in 0..ticks {
        let state_before_sha256 = state_digest(estimate)?;
        let mut action = "no-op".to_owned();
        let mut snapshot_sha256: Option<String> = None;
        let mut candidate_state_sha256: Option<String> = None;
        let mut verification_evidence_sha256: Option<String> = None;
        let mut adaptation_cost_units = 0.0_f64;

        if let Some(mut transaction) = pending.take() {
            if let Some(passed) = transaction.verification_passed {
                snapshot_sha256 = Some(state_digest(transaction.snapshot)?);
                candidate_state_sha256 = Some(state_digest(transaction.candidate)?);
                if passed {
                    action = "commit".to_owned();
                    estimate = transaction.candidate;
                    adaptation_cost_units = 0.8;
                } else {
                    action = "rollback".to_owned();
                    estimate = transaction.snapshot;
                    adaptation_cost_units = 0.2;
                }
            } else {
                while transaction.next_evaluation_tick < tick {
                    let index = transaction.next_evaluation_tick;
                    let active_loss = *losses.get(index)
                        .context("missing active-model loss for a shadow verification tick")?;
                    let candidate_loss = (transaction.candidate - targets[index]).powi(2);
                    transaction.shadow_evaluations.push((index, active_loss, candidate_loss));
                    transaction.next_evaluation_tick += 1;
                }

                let should_verify = if mode.requires_out_of_sample_verification() {
                    transaction.shadow_evaluations.len() >= 2
                } else {
                    tick > transaction.stage_tick
                };
                if should_verify {
                    let (passed, evidence) = if mode.requires_out_of_sample_verification() {
                        let active_mean = mean(
                            &transaction.shadow_evaluations.iter().map(|row| row.1).collect::<Vec<_>>()
                        );
                        let candidate_mean = mean(
                            &transaction.shadow_evaluations.iter().map(|row| row.2).collect::<Vec<_>>()
                        );
                        let evidence = serde_json::json!({
                            "method": "out-of-sample-shadow-loss",
                            "observations": &transaction.shadow_evaluations,
                            "active_mean_loss": active_mean,
                            "candidate_mean_loss": candidate_mean,
                            "minimum_improvement": 0.005,
                        });
                        (candidate_mean + 0.005 < active_mean, evidence)
                    } else {
                        let end = tick;
                        let start = end.saturating_sub(3);
                        let known = &targets[start..end];
                        let active_mean = known.iter().map(|target| (estimate - target).powi(2)).sum::<f64>()
                            / known.len().max(1) as f64;
                        let candidate_mean = known.iter().map(|target| (transaction.candidate - target).powi(2)).sum::<f64>()
                            / known.len().max(1) as f64;
                        let passed = mode.uses_oracle() || candidate_mean < active_mean;
                        let evidence = serde_json::json!({
                            "method": if mode.uses_oracle() { "oracle-diagnostic" } else { "historical-observed-outcomes" },
                            "observed_target_ticks": (start..end).collect::<Vec<_>>(),
                            "active_mean_loss": active_mean,
                            "candidate_mean_loss": candidate_mean,
                            "oracle_channel_access": mode.uses_oracle(),
                        });
                        (passed, evidence)
                    };
                    action = "verify".to_owned();
                    verification_evidence_sha256 = Some(sha256_hex(&serde_json::to_vec(&evidence)?));
                    snapshot_sha256 = Some(state_digest(transaction.snapshot)?);
                    candidate_state_sha256 = Some(state_digest(transaction.candidate)?);
                    transaction.verification_passed = Some(passed);
                    adaptation_cost_units = 0.1;
                }
                pending = Some(transaction);
            }
        } else if !attempted && should_stage(mode, tick, scenario, shift_tick, &losses, targets, estimate, ticks) {
            let candidate = if mode.uses_oracle() {
                // Privileged target access is diagnostic-only and is explicitly declared in receipt.
                targets[tick]
            } else {
                mean_recent(&targets[..tick], 3).unwrap_or(estimate)
            };
            snapshot_sha256 = Some(state_before_sha256.clone());
            candidate_state_sha256 = Some(state_digest(candidate)?);
            action = "stage".to_owned();
            adaptation_cost_units = 0.1;
            attempted = true;
            pending = Some(PendingTransaction {
                snapshot: estimate,
                candidate,
                stage_tick: tick,
                next_evaluation_tick: tick,
                shadow_evaluations: Vec::new(),
                verification_passed: None,
            });
        }

        let state_after_sha256 = state_digest(estimate)?;
        let observed_start = tick.saturating_sub(7);
        let observed_event_ids = events[observed_start..=tick]
            .iter().map(|event| event.event_id.clone()).collect::<Vec<_>>();
        let task_loss = (estimate - targets[tick]).powi(2);
        observations.push(ArmObservation {
            cycle_index: tick as u64,
            task_loss,
            adaptation_cost_units,
            task_loss_available_at_tick: Some(tick as u64 + DELAYED_OUTCOME_TICKS),
        });
        losses.push(task_loss);
        decisions.push(ReplayDecision {
            decision_tick: tick as u64,
            observed_event_ids,
            action,
            state_before_sha256,
            state_after_sha256,
            snapshot_sha256,
            candidate_state_sha256,
            verification_evidence_sha256,
            task_loss: Some(task_loss),
            adaptation_cost_units: Some(adaptation_cost_units),
            task_loss_available_at_tick: Some(tick as u64 + DELAYED_OUTCOME_TICKS),
        });
    }

    ensure!(pending.is_none(), "policy {} ended with an unresolved synthetic transaction", mode.id());
    ensure!(observations.len() == ticks && decisions.len() == ticks,
        "policy {} did not cover the full stream", mode.id());

    let scored_arm = CounterfactualArm {
        name: mode.id().to_owned(),
        version: POLICY_VERSION.to_owned(),
        exogenous_stream_sha256: event_ledger_sha256.to_owned(),
        repository_head_sha: repository_head_sha.to_owned(),
        configuration_digest: config_digest.to_owned(),
        observations,
    };
    let replay_arm = ReplayArm {
        arm_id: mode.id().to_owned(),
        repository_head_sha: repository_head_sha.to_owned(),
        configuration_digest: config_digest.to_owned(),
        policy_seed: seed.wrapping_add(mode.id().bytes().map(u64::from).sum::<u64>()),
        oracle_channel_access: mode.uses_oracle(),
        deliveries,
        decisions,
    };
    Ok(PolicyExecution { replay_arm, scored_arm })
}

fn should_stage(
    mode: PolicyMode,
    tick: usize,
    scenario: &str,
    shift_tick: usize,
    losses: &[f64],
    targets: &[f64],
    estimate: f64,
    ticks: usize,
) -> bool {
    if matches!(mode, PolicyMode::NoAdapt) || tick == 0 || tick + 6 >= ticks {
        return false;
    }
    if matches!(mode, PolicyMode::OracleTimed) {
        return tick == shift_tick;
    }
    let recent_loss = mean_recent(losses, 3).unwrap_or(0.0);
    if recent_loss <= 0.08 {
        return false;
    }
    if matches!(mode, PolicyMode::BenefitGated) {
        let candidate = mean_recent(&targets[..tick], 3).unwrap_or(estimate);
        let start = tick.saturating_sub(3);
        let known = &targets[start..tick];
        let current_loss = known.iter().map(|target| (estimate - target).powi(2)).sum::<f64>()
            / known.len().max(1) as f64;
        let candidate_loss = known.iter().map(|target| (candidate - target).powi(2)).sum::<f64>()
            / known.len().max(1) as f64;
        return current_loss - candidate_loss > 0.05;
    }
    let _ = (scenario, shift_tick); // deployable decisions do not consult the hidden schedule
    true
}

fn target_at(scenario: &str, tick: usize, ticks: usize, shift_tick: usize, seed: u64) -> f64 {
    let noise = deterministic_noise(seed, tick) * 0.015;
    let shifted = match scenario {
        "abrupt" => if tick >= shift_tick { 1.0 } else { 0.0 },
        "gradual" => ((tick as f64 - shift_tick as f64) / 8.0).clamp(0.0, 1.0),
        "no-impact" => 0.0,
        "recurring" => if tick >= shift_tick && tick < (ticks * 3 / 4) { 1.0 } else { 0.0 },
        _ => 0.0,
    };
    shifted + noise
}

fn feature_at(scenario: &str, tick: usize, ticks: usize, shift_tick: usize) -> f64 {
    match scenario {
        "no-impact" => if tick >= shift_tick { 1.0 } else { 0.0 },
        "gradual" => ((tick as f64 - shift_tick as f64) / 8.0).clamp(0.0, 1.0),
        "recurring" => if tick >= shift_tick && tick < (ticks * 3 / 4) { 1.0 } else { 0.0 },
        _ => if tick >= shift_tick { 1.0 } else { 0.0 },
    }
}

fn deterministic_noise(seed: u64, tick: usize) -> f64 {
    let mut value = seed
        .wrapping_add((tick as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(0xD1B5_4A32_D192_ED03);
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    let mantissa = value.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
    (mantissa as f64 / ((1_u64 << 53) as f64)) * 2.0 - 1.0
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn mean_recent(values: &[f64], limit: usize) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(mean(&values[values.len().saturating_sub(limit)..]))
    }
}

fn state_digest(estimate: f64) -> anyhow::Result<String> {
    Ok(sha256_hex(&serde_json::to_vec(&("synthetic-scalar-model-v1", estimate))?))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn summary(
    scenario: &str,
    seed: u64,
    ticks: usize,
    receipt_path: &PathBuf,
    utility_input_path: &PathBuf,
    score_path: &PathBuf,
    validation_path: &PathBuf,
    validation: &ReplayReceiptValidation,
    score: &CounterfactualUtilityScore,
) -> SyntheticReplaySummary {
    SyntheticReplaySummary {
        schema_version: 1,
        run_kind: "synthetic-not-measured".to_owned(),
        scenario_id: scenario.to_owned(),
        comparison_track: score.comparison_track.clone(),
        seed,
        ticks,
        receipt_path: receipt_path.display().to_string(),
        utility_input_path: utility_input_path.display().to_string(),
        utility_score_path: score_path.display().to_string(),
        receipt_validation_path: validation_path.display().to_string(),
        receipt_sha256: validation.canonical_receipt_sha256.clone(),
        score_sha256: score.result_sha256.clone().unwrap_or_default(),
        no_adapt_post_shift_loss: score.no_adapt.post_shift_mean_task_loss,
        detector_post_shift_loss: score.detector_triggered.post_shift_mean_task_loss,
        benefit_gated_post_shift_loss: score.detector_benefit_gated.post_shift_mean_task_loss,
        staged_post_shift_loss: score.staged_verify_rollback.post_shift_mean_task_loss,
        oracle_diagnostic_post_shift_loss: score.oracle_timed.post_shift_mean_task_loss,
        staged_adaptation_cost_units: score.staged_verify_rollback.total_adaptation_cost_units,
        interpretation_guard: "Synthetic contract smoke only. These values exercise the receipt/score pipeline; they are not empirical results, do not qualify as benchmark evidence, and cannot enter a confirmatory panel.".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scenario_generates_a_valid_receipt_and_score_input() {
        for scenario in REQUIRED_SCENARIOS {
            let generated = build_synthetic_run(scenario, 64, 17).expect("fixture generation");
            let validation = cognitive_replay_receipt::validate_receipt(generated.receipt)
                .expect("receipt contract");
            assert_eq!(validation.arm_count, 5);
            assert_eq!(validation.total_decision_count, 64 * 5);
            let score = cognitive_adaptation_utility::score_counterfactual_utility(generated.utility_input)
                .expect("utility input contract");
            assert_eq!(score.no_adapt.cycle_count, 64);
            assert_eq!(score.source_replay_run_kind, None);
        }
    }

    #[test]
    fn no_impact_feature_shift_does_not_trigger_deployable_outcome_policies() {
        let generated = build_synthetic_run("no-impact", 64, 17).expect("fixture generation");
        for arm_id in ["detector-triggered", "detector-benefit-gated", "staged-verify-rollback"] {
            let arm = generated.receipt.arms.iter().find(|arm| arm.arm_id == arm_id).unwrap();
            assert!(arm.decisions.iter().all(|decision| decision.action != "stage"),
                "{arm_id} should not adapt to a feature-only shift with no task impact");
        }
    }

    #[test]
    fn deployable_policy_prefix_decisions_are_invariant_to_hidden_future_targets() {
        // This is a metamorphic regression test for the current simulator, not proof of sandboxing.
        // Keep the observed feature stream fixed, alter only hidden outcomes at and after a cutoff,
        // and assert that deployable policies cannot retroactively change earlier decisions.
        const TICKS: usize = 64;
        const CUTS: [usize; 3] = [16, 32, 48];
        let modes = [
            PolicyMode::NoAdapt,
            PolicyMode::DetectorTriggered,
            PolicyMode::BenefitGated,
            PolicyMode::StagedVerifyRollback,
        ];

        for scenario in REQUIRED_SCENARIOS {
            for seed in 0..8_u64 {
                let generated = build_synthetic_run(scenario, TICKS, seed)
                    .expect("synthetic fixture generation");
                let shift_tick = TICKS / 2;
                let targets: Vec<f64> = (0..TICKS)
                    .map(|tick| target_at(scenario, tick, TICKS, shift_tick, seed))
                    .collect();

                for cutoff in CUTS {
                    let mut altered_targets = targets.clone();
                    for tick in cutoff..TICKS {
                        altered_targets[tick] += 0.25 + deterministic_noise(seed.wrapping_add(91), tick) * 0.1;
                    }
                    assert_ne!(targets[cutoff], altered_targets[cutoff],
                        "the hidden suffix mutation must be non-vacuous");

                    for mode in modes {
                        let arm = generated.receipt.arms.iter()
                            .find(|arm| arm.arm_id == mode.id())
                            .expect("generated policy arm exists");
                        let baseline = simulate_policy(
                            mode, scenario, seed, &targets, shift_tick,
                            &generated.receipt.events, &generated.receipt.event_ledger_sha256,
                            &arm.repository_head_sha, &arm.configuration_digest,
                        ).expect("baseline policy execution");
                        let perturbed = simulate_policy(
                            mode, scenario, seed, &altered_targets, shift_tick,
                            &generated.receipt.events, &generated.receipt.event_ledger_sha256,
                            &arm.repository_head_sha, &arm.configuration_digest,
                        ).expect("future-perturbed policy execution");

                        let baseline_prefix = serde_json::to_value(&baseline.replay_arm.decisions[..cutoff])
                            .expect("baseline prefix serializes");
                        let perturbed_prefix = serde_json::to_value(&perturbed.replay_arm.decisions[..cutoff])
                            .expect("perturbed prefix serializes");
                        assert_eq!(
                            baseline_prefix, perturbed_prefix,
                            "policy {} used hidden outcomes at/after tick {} to change an earlier decision (scenario={scenario}, seed={seed})",
                            mode.id(), cutoff
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn synthetic_receipt_cannot_be_mislabeled_as_measured() {
        let mut generated = build_synthetic_run("abrupt", 64, 17).expect("fixture generation");
        generated.receipt.run_kind = "replay".to_owned();
        let error = cognitive_replay_receipt::validate_receipt(generated.receipt)
            .expect_err("executed replay requires stronger evidence");
        assert!(error.to_string().contains("synthetic run_id cannot be relabeled"));
    }

    #[test]
    fn fixed_seed_reproduces_identical_receipt_bytes() {
        let first = build_synthetic_run("abrupt", 64, 17).expect("fixture generation");
        let second = build_synthetic_run("abrupt", 64, 17).expect("fixture generation");
        assert_eq!(serde_json::to_vec(&first.receipt).unwrap(), serde_json::to_vec(&second.receipt).unwrap());
        assert_eq!(serde_json::to_vec(&first.utility_input).unwrap(), serde_json::to_vec(&second.utility_input).unwrap());
    }
}
