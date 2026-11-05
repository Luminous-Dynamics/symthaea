//! Deterministic, evaluator-separated generation of delayed and informatively missing outcomes.
//!
//! The controller receives a single-tick input, never the full future delivery queue. The
//! queue remains inside the trusted driver; hidden targets, selection probabilities, future
//! reveal times, and the missingness mechanism remain in the separate evaluator artifact.

use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleInput {
    pub schema_version: u32,
    pub run_id: String,
    pub dataset_id: String,
    pub scenario_id: String,
    pub seed: u64,
    pub evaluation_horizon_tick: u64,
    pub config: MissingnessScheduleConfig,
    pub events: Vec<HiddenOutcomeEvent>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HiddenOutcomeEvent {
    pub event_id: String,
    pub decision_tick: u64,
    /// Evaluator-only normalized task loss. Never copied into a pending notice.
    pub hidden_loss: f64,
    /// Registered covariate available to both the evaluator and the policy at decision time.
    pub observed_covariate: f64,
    /// Evaluator-only signal used for the missingness stress mechanism, not a default policy input.
    pub prediction_error_signal: f64,
    /// The action was already selected at decision time; this is not a future action.
    pub action_taken: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MissingnessScheduleConfig {
    pub mechanism: MissingnessMechanism,
    pub base_missing_probability: f64,
    pub missing_probability_slope: f64,
    pub declaration_mode: MissingnessDeclarationMode,
    pub declaration_delay_ticks: u64,
    pub delay: DelayPattern,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissingnessMechanism {
    Mcar,
    MarByObservedCovariate,
    MnarHighLoss,
    MnarLowLoss,
    PredictionErrorDependent,
    ActionDependent,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissingnessDeclarationMode {
    Explicit,
    Silent,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DelayPattern {
    Fixed { ticks: u64 },
    HeterogeneousBounded { max_ticks: u64 },
    HeavyTailedCapped { scale_ticks: u64, cap_ticks: u64 },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStateAtHorizon {
    Observed,
    DelayedPending,
    RightCensoredAtHorizon,
    DeclaredMissing,
    UnknownUnnotified,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyTickInput {
    pub schema_version: u32,
    pub generator_version: String,
    /// Only messages deliverable at this logical tick; never includes later-tick messages.
    pub tick: u64,
    pub messages: Vec<PolicyVisibleMessage>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PolicyVisibleMessage {
    OutcomePending {
        tick: u64,
        event_id: String,
        observed_covariate: f64,
        action_taken: bool,
    },
    OutcomeRevealed {
        tick: u64,
        event_id: String,
        outcome_loss: f64,
    },
    MissingDeclared {
        tick: u64,
        event_id: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorSchedule {
    pub schema_version: u32,
    pub generator_version: String,
    pub run_id: String,
    pub dataset_id: String,
    pub scenario_id: String,
    pub seed: u64,
    pub evaluation_horizon_tick: u64,
    pub source_input_sha256: String,
    pub config_sha256: String,
    /// This artifact is evaluator-only and MUST NOT be mounted into the policy process.
    pub events: Vec<EvaluatorOutcomeRecord>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorOutcomeRecord {
    /// Original source ID, evaluator-only.
    pub event_id: String,
    /// Opaque stable pseudonym used in all policy-visible messages.
    pub policy_event_id: String,
    pub decision_tick: u64,
    pub hidden_loss: f64,
    pub observed_covariate: f64,
    pub prediction_error_signal: f64,
    pub action_taken: bool,
    pub missingness_probability: f64,
    pub was_selected_missing: bool,
    pub delay_ticks: Option<u64>,
    pub expected_reveal_tick: Option<u64>,
    pub declaration_tick: Option<u64>,
    pub state_at_horizon: OutcomeStateAtHorizon,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationSummary {
    pub generator_version: String,
    pub run_id: String,
    pub eligible_event_count: usize,
    pub policy_tick: u64,
    pub policy_tick_message_count: usize,
    pub observed_count: usize,
    pub delayed_pending_count: usize,
    pub right_censored_count: usize,
    pub declared_missing_count: usize,
    pub unknown_unnotified_count: usize,
    pub source_input_sha256: String,
    pub policy_tick_sha256: String,
    pub evaluator_schedule_sha256: String,
    pub interpretation_guard: String,
}

pub struct GeneratedSchedules {
    // Orchestrator-only queue. It is not serialized as policy input because it contains future feedback.
    delivery_messages: Vec<PolicyVisibleMessage>,
    pub evaluator_schedule: EvaluatorSchedule,
}

impl GeneratedSchedules {
    /// Return one tick's messages only. The controller must receive this snapshot, never the queue.
    pub fn policy_tick_input(&self, tick: u64) -> PolicyTickInput {
        let messages = self
            .delivery_messages
            .iter()
            .filter(|message| policy_message_tick(message) == tick)
            .cloned()
            .collect();
        PolicyTickInput {
            schema_version: 1,
            generator_version: "cognitive-missingness-schedule-v1".to_string(),
            tick,
            messages,
        }
    }
}

/// Generate one policy-visible tick and a separate evaluator-only schedule sidecar.
pub fn generate_file(
    input_path: &Path,
    policy_output_path: &Path,
    evaluator_output_path: &Path,
    policy_tick: u64,
) -> anyhow::Result<GenerationSummary> {
    let policy_target = normalized_output_target(policy_output_path)?;
    let evaluator_target = normalized_output_target(evaluator_output_path)?;
    ensure!(
        policy_target != evaluator_target,
        "policy tick input and evaluator schedule must resolve to different files"
    );
    let input_bytes = fs::read(input_path)
        .with_context(|| format!("reading missingness schedule input {}", input_path.display()))?;
    let input: ScheduleInput = serde_json::from_slice(&input_bytes)
        .with_context(|| format!("parsing missingness schedule input {}", input_path.display()))?;
    let generated = generate(&input, &input_bytes)?;
    ensure!(
        policy_tick <= generated.evaluator_schedule.evaluation_horizon_tick,
        "requested policy tick {} is beyond evaluation horizon {}",
        policy_tick,
        generated.evaluator_schedule.evaluation_horizon_tick
    );
    let policy_tick_input = generated.policy_tick_input(policy_tick);
    let mut policy_bytes = serde_json::to_vec_pretty(&policy_tick_input)?;
    policy_bytes.push(b'\n');
    let mut evaluator_bytes = serde_json::to_vec_pretty(&generated.evaluator_schedule)?;
    evaluator_bytes.push(b'\n');
    fs::write(policy_output_path, &policy_bytes)
        .with_context(|| format!("writing policy tick input {}", policy_output_path.display()))?;
    fs::write(evaluator_output_path, &evaluator_bytes)
        .with_context(|| format!("writing evaluator schedule {}", evaluator_output_path.display()))?;

    let count_state = |state| {
        generated
            .evaluator_schedule
            .events
            .iter()
            .filter(|record| record.state_at_horizon == state)
            .count()
    };
    let summary = GenerationSummary {
        generator_version: "cognitive-missingness-schedule-v1".to_string(),
        run_id: generated.evaluator_schedule.run_id.clone(),
        eligible_event_count: generated.evaluator_schedule.events.len(),
        policy_tick,
        policy_tick_message_count: policy_tick_input.messages.len(),
        observed_count: count_state(OutcomeStateAtHorizon::Observed),
        delayed_pending_count: count_state(OutcomeStateAtHorizon::DelayedPending),
        right_censored_count: count_state(OutcomeStateAtHorizon::RightCensoredAtHorizon),
        declared_missing_count: count_state(OutcomeStateAtHorizon::DeclaredMissing),
        unknown_unnotified_count: count_state(OutcomeStateAtHorizon::UnknownUnnotified),
        source_input_sha256: generated.evaluator_schedule.source_input_sha256.clone(),
        policy_tick_sha256: sha256_hex(&policy_bytes),
        evaluator_schedule_sha256: sha256_hex(&evaluator_bytes),
        interpretation_guard: "The policy artifact contains only one logical tick. The complete delivery queue and the evaluator schedule must remain in the trusted driver, outside the policy process. Synthetic plumbing only; not measured task performance, causal identification, or SOTA evidence.".to_string(),
    };
    Ok(summary)
}

/// Generate deterministic, separated artifacts from an already parsed input.
pub fn generate(input: &ScheduleInput, source_bytes: &[u8]) -> anyhow::Result<GeneratedSchedules> {
    validate_input(input)?;
    let source_digest = sha256_hex(source_bytes);
    let config_bytes = serde_json::to_vec(&input.config)?;
    let config_digest = sha256_hex(&config_bytes);

    let mut events = input.events.iter().collect::<Vec<_>>();
    events.sort_by(|left, right| left.event_id.cmp(&right.event_id));

    let mut evaluator_records = Vec::with_capacity(events.len());
    let mut policy_messages = Vec::with_capacity(events.len() * 2);
    for event in events {
        let policy_event_id = opaque_policy_event_id(input.seed, &event.event_id);
        policy_messages.push(PolicyVisibleMessage::OutcomePending {
            tick: event.decision_tick,
            event_id: policy_event_id.clone(),
            observed_covariate: event.observed_covariate,
            action_taken: event.action_taken,
        });

        let p_missing = missingness_probability(&input.config, event);
        let was_selected_missing =
            uniform01(input.seed, &event.event_id, "missingness-selection") < p_missing;
        let mut delay_ticks = None;
        let mut expected_reveal_tick = None;
        let mut declaration_tick = None;

        let state_at_horizon = if was_selected_missing {
            match input.config.declaration_mode {
                MissingnessDeclarationMode::Explicit => {
                    let tick = event
                        .decision_tick
                        .checked_add(input.config.declaration_delay_ticks)
                        .context("missingness declaration tick overflow")?;
                    declaration_tick = Some(tick);
                    if tick <= input.evaluation_horizon_tick {
                        policy_messages.push(PolicyVisibleMessage::MissingDeclared {
                            tick,
                            event_id: policy_event_id.clone(),
                        });
                        OutcomeStateAtHorizon::DeclaredMissing
                    } else {
                        OutcomeStateAtHorizon::DelayedPending
                    }
                }
                MissingnessDeclarationMode::Silent => OutcomeStateAtHorizon::UnknownUnnotified,
            }
        } else {
            let delay = sample_delay(input.seed, &event.event_id, &input.config.delay)?;
            let reveal_tick = event
                .decision_tick
                .checked_add(delay)
                .context("outcome reveal tick overflow")?;
            delay_ticks = Some(delay);
            expected_reveal_tick = Some(reveal_tick);
            if reveal_tick <= input.evaluation_horizon_tick {
                policy_messages.push(PolicyVisibleMessage::OutcomeRevealed {
                    tick: reveal_tick,
                    event_id: policy_event_id.clone(),
                    outcome_loss: event.hidden_loss,
                });
                OutcomeStateAtHorizon::Observed
            } else {
                OutcomeStateAtHorizon::RightCensoredAtHorizon
            }
        };

        evaluator_records.push(EvaluatorOutcomeRecord {
            event_id: event.event_id.clone(),
            policy_event_id,
            decision_tick: event.decision_tick,
            hidden_loss: event.hidden_loss,
            observed_covariate: event.observed_covariate,
            prediction_error_signal: event.prediction_error_signal,
            action_taken: event.action_taken,
            missingness_probability: p_missing,
            was_selected_missing,
            delay_ticks,
            expected_reveal_tick,
            declaration_tick,
            state_at_horizon,
        });
    }

    policy_messages.sort_by(|left, right| {
        let left_key = policy_message_key(left);
        let right_key = policy_message_key(right);
        left_key.cmp(&right_key)
    });

    let count_state = |state| {
        evaluator_records.iter().filter(|record| record.state_at_horizon == state).count()
    };
    let observed_count = count_state(OutcomeStateAtHorizon::Observed);
    let delayed_pending_count = count_state(OutcomeStateAtHorizon::DelayedPending);
    let right_censored_count = count_state(OutcomeStateAtHorizon::RightCensoredAtHorizon);
    let declared_missing_count = count_state(OutcomeStateAtHorizon::DeclaredMissing);
    let unknown_unnotified_count = count_state(OutcomeStateAtHorizon::UnknownUnnotified);
    ensure!(
        observed_count
            + delayed_pending_count
            + right_censored_count
            + declared_missing_count
            + unknown_unnotified_count
            == input.events.len(),
        "generated outcome-state accounting did not reconcile to eligible event count"
    );

    let evaluator_schedule = EvaluatorSchedule {
        schema_version: 1,
        generator_version: "cognitive-missingness-schedule-v1".to_string(),
        run_id: input.run_id.clone(),
        dataset_id: input.dataset_id.clone(),
        scenario_id: input.scenario_id.clone(),
        seed: input.seed,
        evaluation_horizon_tick: input.evaluation_horizon_tick,
        source_input_sha256: source_digest,
        config_sha256: config_digest,
        events: evaluator_records,
    };
    Ok(GeneratedSchedules {
        delivery_messages: policy_messages,
        evaluator_schedule,
    })
}

fn validate_input(input: &ScheduleInput) -> anyhow::Result<()> {
    ensure!(input.schema_version == 1, "unsupported schedule schema_version {}", input.schema_version);
    ensure!(!input.run_id.trim().is_empty(), "run_id must not be empty");
    ensure!(!input.dataset_id.trim().is_empty(), "dataset_id must not be empty");
    ensure!(!input.scenario_id.trim().is_empty(), "scenario_id must not be empty");
    ensure!(!input.events.is_empty(), "at least one event is required");
    ensure!(
        input.config.base_missing_probability.is_finite()
            && (0.0..=1.0).contains(&input.config.base_missing_probability),
        "base_missing_probability must be finite and in [0, 1]"
    );
    ensure!(
        input.config.missing_probability_slope.is_finite()
            && (0.0..=1.0).contains(&input.config.missing_probability_slope),
        "missing_probability_slope must be finite and in [0, 1]"
    );
    if input.config.mechanism == MissingnessMechanism::Mcar {
        ensure!(
            input.config.missing_probability_slope == 0.0,
            "MCAR requires missing_probability_slope=0 so no covariate or outcome dependence is smuggled in"
        );
    }
    if let DelayPattern::HeterogeneousBounded { max_ticks } = input.config.delay {
        ensure!(max_ticks < u64::MAX, "max_ticks must be below u64::MAX");
    }
    if let DelayPattern::HeavyTailedCapped { scale_ticks, cap_ticks } = input.config.delay {
        ensure!(scale_ticks > 0, "heavy-tailed scale_ticks must be positive");
        ensure!(cap_ticks > 0, "heavy-tailed cap_ticks must be positive");
    }
    let mut seen = BTreeSet::new();
    for event in &input.events {
        ensure!(!event.event_id.trim().is_empty(), "event_id must not be empty");
        ensure!(seen.insert(event.event_id.as_str()), "duplicate event_id {}", event.event_id);
        ensure!(
            event.decision_tick <= input.evaluation_horizon_tick,
            "event {} decision tick is beyond the evaluation horizon",
            event.event_id
        );
        ensure!(
            event.hidden_loss.is_finite() && (0.0..=1.0).contains(&event.hidden_loss),
            "event {} hidden_loss must be finite and in [0, 1]",
            event.event_id
        );
        ensure!(
            event.observed_covariate.is_finite() && (0.0..=1.0).contains(&event.observed_covariate),
            "event {} observed_covariate must be finite and in [0, 1]",
            event.event_id
        );
        ensure!(
            event.prediction_error_signal.is_finite()
                && (0.0..=1.0).contains(&event.prediction_error_signal),
            "event {} prediction_error_signal must be finite and in [0, 1]",
            event.event_id
        );
    }
    Ok(())
}

fn missingness_probability(config: &MissingnessScheduleConfig, event: &HiddenOutcomeEvent) -> f64 {
    let risk = match config.mechanism {
        MissingnessMechanism::Mcar => 0.0,
        MissingnessMechanism::MarByObservedCovariate => event.observed_covariate,
        MissingnessMechanism::MnarHighLoss => event.hidden_loss,
        MissingnessMechanism::MnarLowLoss => 1.0 - event.hidden_loss,
        MissingnessMechanism::PredictionErrorDependent => event.prediction_error_signal,
        MissingnessMechanism::ActionDependent => if event.action_taken { 1.0 } else { 0.0 },
    };
    (config.base_missing_probability + config.missing_probability_slope * risk).clamp(0.0, 1.0)
}

fn sample_delay(seed: u64, event_id: &str, pattern: &DelayPattern) -> anyhow::Result<u64> {
    match *pattern {
        DelayPattern::Fixed { ticks } => Ok(ticks),
        DelayPattern::HeterogeneousBounded { max_ticks } => {
            let modulus = max_ticks.checked_add(1).context("max_ticks overflow")?;
            Ok(deterministic_u64(seed, event_id, "outcome-delay") % modulus)
        }
        DelayPattern::HeavyTailedCapped { scale_ticks, cap_ticks } => {
            ensure!(scale_ticks > 0 && cap_ticks > 0, "heavy-tailed scale and cap must be positive");
            let u = uniform01(seed, event_id, "outcome-delay");
            let ratio = u / (1.0 - u);
            let raw = (scale_ticks as f64) * ratio;
            if !raw.is_finite() || raw >= cap_ticks as f64 {
                Ok(cap_ticks)
            } else {
                Ok((raw.floor() as u64).min(cap_ticks))
            }
        }
    }
}

fn normalized_output_target(path: &Path) -> anyhow::Result<PathBuf> {
    if fs::symlink_metadata(path).is_ok() {
        return fs::canonicalize(path)
            .with_context(|| format!("canonicalizing existing output path {}", path.display()));
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(parent)
        .with_context(|| format!("canonicalizing output directory {}", parent.display()))?;
    let file_name = path
        .file_name()
        .context("output path must include a file name")?;
    Ok(canonical_parent.join(file_name))
}

fn policy_message_tick(message: &PolicyVisibleMessage) -> u64 {
    match message {
        PolicyVisibleMessage::OutcomePending { tick, .. }
        | PolicyVisibleMessage::OutcomeRevealed { tick, .. }
        | PolicyVisibleMessage::MissingDeclared { tick, .. } => *tick,
    }
}

fn policy_message_key(message: &PolicyVisibleMessage) -> (u64, u8, &str) {
    match message {
        // At a shared logical tick, all decisions/pending notices precede every feedback event.
        // Event ID is only a stable tie-breaker within one causality phase.
        PolicyVisibleMessage::OutcomePending { tick, event_id, .. } => (*tick, 0, event_id),
        PolicyVisibleMessage::OutcomeRevealed { tick, event_id, .. } => (*tick, 1, event_id),
        PolicyVisibleMessage::MissingDeclared { tick, event_id } => (*tick, 1, event_id),
    }
}

fn opaque_policy_event_id(seed: u64, source_event_id: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"symthaea-policy-event-id-v1\0");
    digest.update(seed.to_le_bytes());
    digest.update([0]);
    digest.update(source_event_id.as_bytes());
    let bytes = digest.finalize();
    format!("evt_{}", bytes[..16].iter().map(|byte| format!("{byte:02x}")).collect::<String>())
}

fn deterministic_u64(seed: u64, event_id: &str, domain: &str) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"symthaea-cognitive-missingness-schedule-v1\0");
    digest.update(seed.to_le_bytes());
    digest.update(domain.as_bytes());
    digest.update([0]);
    digest.update(event_id.as_bytes());
    let bytes = digest.finalize();
    u64::from_le_bytes(bytes[..8].try_into().expect("SHA-256 output always has at least eight bytes"))
}

fn uniform01(seed: u64, event_id: &str, domain: &str) -> f64 {
    ((deterministic_u64(seed, event_id, domain) >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(mechanism: MissingnessMechanism, declaration_mode: MissingnessDeclarationMode) -> ScheduleInput {
        let events = (0..64)
            .map(|index| HiddenOutcomeEvent {
                event_id: format!("evt-{index:03}"),
                decision_tick: index,
                hidden_loss: (index % 10) as f64 / 9.0,
                observed_covariate: ((index * 7) % 10) as f64 / 9.0,
                prediction_error_signal: ((index * 3) % 10) as f64 / 9.0,
                action_taken: index % 3 == 0,
            })
            .collect();
        ScheduleInput {
            schema_version: 1,
            run_id: "synthetic-run-not-measured".to_string(),
            dataset_id: "synthetic-data-not-measured".to_string(),
            scenario_id: "unit-test".to_string(),
            seed: 47,
            evaluation_horizon_tick: 96,
            config: MissingnessScheduleConfig {
                mechanism,
                base_missing_probability: 0.15,
                missing_probability_slope: if mechanism == MissingnessMechanism::Mcar { 0.0 } else { 0.7 },
                declaration_mode,
                declaration_delay_ticks: 1,
                delay: DelayPattern::HeterogeneousBounded { max_ticks: 24 },
            },
            events,
        }
    }

    #[test]
    fn generation_is_deterministic_and_event_order_independent() {
        let input = fixture(MissingnessMechanism::MnarHighLoss, MissingnessDeclarationMode::Explicit);
        let first = generate(&input, b"same-pinned-source").unwrap();
        let second = generate(&input, b"same-pinned-source").unwrap();
        assert_eq!(
            serde_json::to_vec(&first.policy_tick_input(7)).unwrap(),
            serde_json::to_vec(&second.policy_tick_input(7)).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&first.evaluator_schedule).unwrap(),
            serde_json::to_vec(&second.evaluator_schedule).unwrap()
        );

        let mut permuted = input.clone();
        permuted.events.reverse();
        let permuted_out = generate(&permuted, b"same-pinned-source").unwrap();
        assert_eq!(
            serde_json::to_vec(&first.policy_tick_input(7)).unwrap(),
            serde_json::to_vec(&permuted_out.policy_tick_input(7)).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&first.evaluator_schedule.events).unwrap(),
            serde_json::to_vec(&permuted_out.evaluator_schedule.events).unwrap()
        );
    }

    #[test]
    fn policy_tick_input_has_no_hidden_outcome_or_future_schedule_fields() {
        let generated = generate(
            &fixture(MissingnessMechanism::MnarHighLoss, MissingnessDeclarationMode::Explicit),
            b"policy-boundary",
        )
        .unwrap();
        let policy_tick = generated.policy_tick_input(7);
        assert!(policy_tick.messages.iter().all(|message| policy_message_tick(message) == 7));
        let policy_json = serde_json::to_string(&policy_tick).unwrap();
        assert_eq!(policy_tick.tick, 7);
        assert!(!policy_json.contains("hidden_loss"));
        assert!(!policy_json.contains("prediction_error_signal"));
        assert!(!policy_json.contains("missingness_probability"));
        assert!(!policy_json.contains("was_selected_missing"));
        assert!(!policy_json.contains("expected_reveal_tick"));
        assert!(!policy_json.contains("declaration_tick"));
        assert!(!policy_json.contains("mnar_high_loss"));
        assert!(!policy_json.contains("evt-007"));
        assert!(policy_tick.messages.iter().all(|message| match message {
            PolicyVisibleMessage::OutcomePending { event_id, .. }
            | PolicyVisibleMessage::OutcomeRevealed { event_id, .. }
            | PolicyVisibleMessage::MissingDeclared { event_id, .. } => event_id.starts_with("evt_") && event_id.len() == 36,
        }));
        assert!(!policy_json.contains("scenario_id"));
        assert!(!policy_json.contains("dataset_id"));
        assert!(!policy_json.contains("run_id"));
        assert!(!policy_json.contains("\"seed\""));
        assert!(policy_json.contains("observed_covariate"));
        // With immediate outcomes, tick 19 includes the event-19 pending notice and its
        // same-tick reveal, but never any messages from a different logical tick.
        assert!(!policy_json.contains("outcome_loss"));
        let mut immediate = fixture(MissingnessMechanism::Mcar, MissingnessDeclarationMode::Explicit);
        immediate.config.base_missing_probability = 0.0;
        immediate.config.delay = DelayPattern::Fixed { ticks: 0 };
        let immediate_generated = generate(&immediate, b"immediate-reveal").unwrap();
        let reveal_tick = immediate_generated.policy_tick_input(19);
        assert!(reveal_tick.messages.iter().all(|message| policy_message_tick(message) == 19));
        assert!(serde_json::to_string(&reveal_tick).unwrap().contains("outcome_loss"));
    }

    #[test]
    fn explicit_silent_pending_and_right_censored_states_remain_distinct() {
        let mut explicit = fixture(MissingnessMechanism::Mcar, MissingnessDeclarationMode::Explicit);
        explicit.config.base_missing_probability = 1.0;
        explicit.config.declaration_delay_ticks = 0;
        let explicit_out = generate(&explicit, b"explicit").unwrap();
        assert!(explicit_out.evaluator_schedule.events.iter().all(|e| e.state_at_horizon == OutcomeStateAtHorizon::DeclaredMissing));
        assert!(explicit_out.policy_tick_input(0).messages.iter().any(|m| matches!(m, PolicyVisibleMessage::MissingDeclared { .. })));

        let mut silent = explicit.clone();
        silent.config.declaration_mode = MissingnessDeclarationMode::Silent;
        let silent_out = generate(&silent, b"silent").unwrap();
        assert!(silent_out.evaluator_schedule.events.iter().all(|e| e.state_at_horizon == OutcomeStateAtHorizon::UnknownUnnotified));
        assert!(!silent_out.policy_tick_input(0).messages.iter().any(|m| matches!(m, PolicyVisibleMessage::MissingDeclared { .. })));

        let mut pending = explicit.clone();
        pending.config.declaration_mode = MissingnessDeclarationMode::Explicit;
        pending.config.declaration_delay_ticks = 100;
        let pending_out = generate(&pending, b"pending").unwrap();
        assert!(pending_out.evaluator_schedule.events.iter().any(|e| e.state_at_horizon == OutcomeStateAtHorizon::DelayedPending));
        assert!(!pending_out.policy_tick_input(0).messages.iter().any(|m| matches!(m, PolicyVisibleMessage::MissingDeclared { .. })));
    }

    #[test]
    fn fixed_long_delay_is_right_censored_at_horizon_without_revealing_future_tick() {
        let mut input = fixture(MissingnessMechanism::Mcar, MissingnessDeclarationMode::Explicit);
        input.config.base_missing_probability = 0.0;
        input.config.delay = DelayPattern::Fixed { ticks: 100 };
        let generated = generate(&input, b"right-censored").unwrap();
        assert!(generated.evaluator_schedule.events.iter().all(|e| e.state_at_horizon == OutcomeStateAtHorizon::RightCensoredAtHorizon));
        let policy_json = serde_json::to_string(&generated.policy_tick_input(0)).unwrap();
        assert!(!policy_json.contains("expected_reveal_tick"));
        assert!(!policy_json.contains("right_censored_at_horizon"));
    }

    #[test]
    fn all_same_tick_decisions_precede_feedback_available_at_that_tick() {
        let mut input = fixture(MissingnessMechanism::Mcar, MissingnessDeclarationMode::Explicit);
        input.config.base_missing_probability = 0.0;
        input.config.delay = DelayPattern::Fixed { ticks: 0 };
        for event in &mut input.events {
            event.decision_tick = 0;
        }
        let generated = generate(&input, b"same-tick-phase-order").unwrap();
        let messages = &generated.delivery_messages;
        assert_eq!(messages.len(), input.events.len() * 2);
        assert!(messages[..input.events.len()]
            .iter()
            .all(|message| matches!(message, PolicyVisibleMessage::OutcomePending { tick: 0, .. })));
        assert!(messages[input.events.len()..]
            .iter()
            .all(|message| matches!(message, PolicyVisibleMessage::OutcomeRevealed { tick: 0, .. })));
    }

    #[test]
    fn every_mechanism_produces_finite_bounded_selection_probabilities() {
        for mechanism in [
            MissingnessMechanism::Mcar,
            MissingnessMechanism::MarByObservedCovariate,
            MissingnessMechanism::MnarHighLoss,
            MissingnessMechanism::MnarLowLoss,
            MissingnessMechanism::PredictionErrorDependent,
            MissingnessMechanism::ActionDependent,
        ] {
            let slope = if mechanism == MissingnessMechanism::Mcar { 0.0 } else { 0.7 };
            let config = MissingnessScheduleConfig {
                mechanism,
                base_missing_probability: 0.2,
                missing_probability_slope: slope,
                declaration_mode: MissingnessDeclarationMode::Explicit,
                declaration_delay_ticks: 1,
                delay: DelayPattern::Fixed { ticks: 1 },
            };
            for event in &fixture(mechanism, MissingnessDeclarationMode::Explicit).events {
                let p = missingness_probability(&config, event);
                assert!(p.is_finite() && (0.0..=1.0).contains(&p));
            }
        }
    }

    #[test]
    fn informative_mechanisms_follow_their_registered_risk_direction() {
        let mut low = HiddenOutcomeEvent {
            event_id: "low".to_string(),
            decision_tick: 0,
            hidden_loss: 0.0,
            observed_covariate: 0.0,
            prediction_error_signal: 0.0,
            action_taken: false,
        };
        let mut high = low.clone();
        high.event_id = "high".to_string();
        high.hidden_loss = 1.0;
        high.observed_covariate = 1.0;
        high.prediction_error_signal = 1.0;
        high.action_taken = true;

        let config = |mechanism| MissingnessScheduleConfig {
            mechanism,
            base_missing_probability: 0.2,
            missing_probability_slope: if mechanism == MissingnessMechanism::Mcar { 0.0 } else { 0.7 },
            declaration_mode: MissingnessDeclarationMode::Explicit,
            declaration_delay_ticks: 1,
            delay: DelayPattern::Fixed { ticks: 1 },
        };

        for mechanism in [
            MissingnessMechanism::MarByObservedCovariate,
            MissingnessMechanism::MnarHighLoss,
            MissingnessMechanism::PredictionErrorDependent,
            MissingnessMechanism::ActionDependent,
        ] {
            let cfg = config(mechanism);
            assert!(missingness_probability(&cfg, &low) < missingness_probability(&cfg, &high));
        }
        let reverse = config(MissingnessMechanism::MnarLowLoss);
        assert!(missingness_probability(&reverse, &low) > missingness_probability(&reverse, &high));

        let mcar = config(MissingnessMechanism::Mcar);
        low.hidden_loss = 0.9;
        low.observed_covariate = 0.7;
        low.prediction_error_signal = 0.8;
        low.action_taken = true;
        assert_eq!(missingness_probability(&mcar, &low), 0.2);
        assert_eq!(missingness_probability(&mcar, &high), 0.2);
    }

    #[test]
    fn rejects_duplicate_ids_invalid_losses_and_mcar_dependence() {
        let mut duplicate = fixture(MissingnessMechanism::Mcar, MissingnessDeclarationMode::Explicit);
        duplicate.events[1].event_id = duplicate.events[0].event_id.clone();
        assert!(validate_input(&duplicate).unwrap_err().to_string().contains("duplicate event_id"));

        let mut invalid_loss = fixture(MissingnessMechanism::Mcar, MissingnessDeclarationMode::Explicit);
        invalid_loss.events[0].hidden_loss = f64::NAN;
        assert!(validate_input(&invalid_loss).is_err());

        let mut invalid_mcar = fixture(MissingnessMechanism::Mcar, MissingnessDeclarationMode::Explicit);
        invalid_mcar.config.missing_probability_slope = 0.1;
        assert!(validate_input(&invalid_mcar).unwrap_err().to_string().contains("MCAR requires"));
    }

    #[test]
    fn policy_messages_are_time_sorted_and_never_reveal_outcomes_before_arrival() {
        let mut input = fixture(MissingnessMechanism::Mcar, MissingnessDeclarationMode::Explicit);
        input.config.base_missing_probability = 0.0;
        input.config.delay = DelayPattern::HeterogeneousBounded { max_ticks: 24 };
        let generated = generate(&input, b"arrival-order").unwrap();
        let messages = &generated.delivery_messages;
        for pair in messages.windows(2) {
            let (left_tick, left_priority, left_id) = policy_message_key(&pair[0]);
            let (right_tick, right_priority, right_id) = policy_message_key(&pair[1]);
            assert!((left_tick, left_priority, left_id) <= (right_tick, right_priority, right_id));
        }
        let reveal_order = messages
            .iter()
            .filter_map(|message| match message {
                PolicyVisibleMessage::OutcomeRevealed { event_id, .. } => Some(event_id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut reveal_order_by_event = reveal_order.clone();
        reveal_order_by_event.sort();
        assert_ne!(
            reveal_order,
            reveal_order_by_event,
            "heterogeneous delays should produce at least one out-of-order arrival"
        );

        for message in messages {
            if let PolicyVisibleMessage::OutcomeRevealed { tick, event_id, .. } = message {
                let record = generated.evaluator_schedule.events.iter().find(|e| &e.policy_event_id == event_id).unwrap();
                assert!(record.expected_reveal_tick.is_some_and(|reveal| reveal == *tick));
                assert!(*tick >= record.decision_tick);
            }
        }
    }
}
