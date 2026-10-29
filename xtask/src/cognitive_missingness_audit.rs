//! Independent structural and deterministic replay audit for missingness sidecars.
//!
//! This module intentionally re-derives selection, delay, state-at-horizon, policy pseudonyms,
//! and the requested tick messages from the pinned source input. It does not call the generator.
//! A successful audit establishes internal reproducibility of these artifacts, not that a runner
//! executed honestly, that a causal estimand is identified, or that any policy performs better.

use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use crate::cognitive_missingness_schedule::{
    DelayPattern, EvaluatorOutcomeRecord, EvaluatorSchedule, HiddenOutcomeEvent,
    MissingnessDeclarationMode, MissingnessMechanism, MissingnessScheduleConfig,
    OutcomeStateAtHorizon, PolicyTickInput, PolicyVisibleMessage, ScheduleInput,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MissingnessAuditReport {
    pub audit_version: String,
    pub result_sha256: Option<String>,
    pub run_id: String,
    pub dataset_id: String,
    pub scenario_id: String,
    pub seed: u64,
    pub policy_tick: u64,
    pub eligible_event_count: usize,
    pub policy_message_count: usize,
    pub observed_count: usize,
    pub delayed_pending_count: usize,
    pub right_censored_count: usize,
    pub declared_missing_count: usize,
    pub unknown_unnotified_count: usize,
    pub source_input_sha256: String,
    pub policy_tick_sha256: String,
    pub evaluator_schedule_sha256: String,
    pub state_count_conserved: bool,
    pub evaluator_records_rederived: bool,
    pub policy_tick_rederived: bool,
    pub opaque_policy_ids_verified: bool,
    pub no_future_tick_messages: bool,
    pub interpretation_guard: String,
}

pub fn audit_files(
    input_path: &Path,
    policy_tick_path: &Path,
    evaluator_path: &Path,
) -> anyhow::Result<MissingnessAuditReport> {
    let input_bytes = fs::read(input_path)
        .with_context(|| format!("reading audit input {}", input_path.display()))?;
    let policy_bytes = fs::read(policy_tick_path)
        .with_context(|| format!("reading policy tick artifact {}", policy_tick_path.display()))?;
    let evaluator_bytes = fs::read(evaluator_path)
        .with_context(|| format!("reading evaluator artifact {}", evaluator_path.display()))?;
    audit_bytes(&input_bytes, &policy_bytes, &evaluator_bytes)
}

/// Validate the exact bytes of the input, policy-visible one-tick artifact, and evaluator sidecar.
pub fn audit_bytes(
    input_bytes: &[u8],
    policy_bytes: &[u8],
    evaluator_bytes: &[u8],
) -> anyhow::Result<MissingnessAuditReport> {
    let input: ScheduleInput = serde_json::from_slice(input_bytes)
        .context("parsing pinned missingness source input")?;
    let policy: PolicyTickInput = serde_json::from_slice(policy_bytes)
        .context("parsing one-tick policy artifact")?;
    let evaluator: EvaluatorSchedule = serde_json::from_slice(evaluator_bytes)
        .context("parsing evaluator-only schedule artifact")?;

    validate_source(&input)?;
    ensure!(policy.schema_version == 1, "unsupported policy tick schema");
    ensure!(
        policy.generator_version == "cognitive-missingness-schedule-v1",
        "unexpected policy tick generator version"
    );
    ensure!(evaluator.schema_version == 1, "unsupported evaluator schedule schema");
    ensure!(
        evaluator.generator_version == "cognitive-missingness-schedule-v1",
        "unexpected evaluator generator version"
    );
    ensure!(
        policy.tick <= input.evaluation_horizon_tick,
        "policy tick {} exceeds evaluation horizon {}",
        policy.tick,
        input.evaluation_horizon_tick
    );
    ensure!(
        evaluator.run_id == input.run_id
            && evaluator.dataset_id == input.dataset_id
            && evaluator.scenario_id == input.scenario_id
            && evaluator.seed == input.seed
            && evaluator.evaluation_horizon_tick == input.evaluation_horizon_tick,
        "evaluator identity/configuration does not match pinned source input"
    );

    let input_digest = sha256_hex(input_bytes);
    let config_digest = sha256_hex(&serde_json::to_vec(&input.config)?);
    ensure!(
        evaluator.source_input_sha256 == input_digest,
        "evaluator source-input digest does not match exact source bytes"
    );
    ensure!(
        evaluator.config_sha256 == config_digest,
        "evaluator configuration digest does not match pinned input config"
    );

    let expected_records = independently_rederive_records(&input)?;
    ensure!(
        evaluator.events == expected_records,
        "evaluator records differ from independently re-derived event schedule"
    );

    let expected_policy_messages = independently_rederive_policy_messages(&input)?;
    let expected_tick_messages = expected_policy_messages
        .into_iter()
        .filter(|message| message_tick(message) == policy.tick)
        .collect::<Vec<_>>();
    ensure!(
        policy.messages.iter().all(|message| message_tick(message) == policy.tick),
        "policy tick artifact contains a message from a different logical tick"
    );
    ensure!(
        policy.messages == expected_tick_messages,
        "policy-visible tick messages differ from independently re-derived messages"
    );

    let event_ids = expected_records.iter().map(|record| record.event_id.as_str()).collect::<BTreeSet<_>>();
    ensure!(event_ids.len() == input.events.len(), "duplicate source event IDs in independently re-derived records");
    let policy_ids = expected_records
        .iter()
        .map(|record| record.policy_event_id.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        policy_ids.len() == expected_records.len(),
        "opaque policy event IDs are not unique"
    );
    ensure!(
        expected_records.iter().all(|record| {
            record.policy_event_id == opaque_policy_event_id(input.seed, &record.event_id)
                && record.policy_event_id != record.event_id
                && record.policy_event_id.starts_with("evt_")
                && record.policy_event_id.len() == 36
        }),
        "a policy-visible event identifier is not the expected opaque pseudonym"
    );

    let count_state = |state| expected_records.iter().filter(|record| record.state_at_horizon == state).count();
    let observed_count = count_state(OutcomeStateAtHorizon::Observed);
    let delayed_pending_count = count_state(OutcomeStateAtHorizon::DelayedPending);
    let right_censored_count = count_state(OutcomeStateAtHorizon::RightCensoredAtHorizon);
    let declared_missing_count = count_state(OutcomeStateAtHorizon::DeclaredMissing);
    let unknown_unnotified_count = count_state(OutcomeStateAtHorizon::UnknownUnnotified);
    let total = observed_count + delayed_pending_count + right_censored_count
        + declared_missing_count + unknown_unnotified_count;
    ensure!(total == input.events.len(), "outcome-state counts do not conserve eligible events");

    let mut report = MissingnessAuditReport {
        audit_version: "cognitive-missingness-independent-audit-v1".to_string(),
        result_sha256: None,
        run_id: input.run_id,
        dataset_id: input.dataset_id,
        scenario_id: input.scenario_id,
        seed: input.seed,
        policy_tick: policy.tick,
        eligible_event_count: expected_records.len(),
        policy_message_count: policy.messages.len(),
        observed_count,
        delayed_pending_count,
        right_censored_count,
        declared_missing_count,
        unknown_unnotified_count,
        source_input_sha256: input_digest,
        policy_tick_sha256: sha256_hex(policy_bytes),
        evaluator_schedule_sha256: sha256_hex(evaluator_bytes),
        state_count_conserved: true,
        evaluator_records_rederived: true,
        policy_tick_rederived: true,
        opaque_policy_ids_verified: true,
        no_future_tick_messages: true,
        interpretation_guard: "Internal artifact consistency and deterministic re-derivation only; not evidence of honest execution, causal identification, measured task improvement, or SOTA superiority.".to_string(),
    };
    let canonical = serde_json::to_vec(&report)?;
    report.result_sha256 = Some(sha256_hex(&canonical));
    Ok(report)
}

fn validate_source(input: &ScheduleInput) -> anyhow::Result<()> {
    ensure!(input.schema_version == 1, "unsupported source schema");
    ensure!(!input.run_id.trim().is_empty(), "run_id must not be empty");
    ensure!(!input.dataset_id.trim().is_empty(), "dataset_id must not be empty");
    ensure!(!input.scenario_id.trim().is_empty(), "scenario_id must not be empty");
    ensure!(!input.events.is_empty(), "source input has no eligible events");
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
            "MCAR cannot depend on risk signals"
        );
    }
    if let DelayPattern::HeterogeneousBounded { max_ticks } = input.config.delay {
        ensure!(max_ticks < u64::MAX, "heterogeneous delay cap overflows");
    }
    if let DelayPattern::HeavyTailedCapped { scale_ticks, cap_ticks } = input.config.delay {
        ensure!(scale_ticks > 0 && cap_ticks > 0, "heavy-tailed delay parameters must be positive");
    }
    let mut seen = BTreeSet::new();
    for event in &input.events {
        ensure!(!event.event_id.trim().is_empty(), "event_id must not be empty");
        ensure!(seen.insert(event.event_id.as_str()), "duplicate source event ID {}", event.event_id);
        ensure!(event.decision_tick <= input.evaluation_horizon_tick, "decision occurs after horizon");
        ensure!(event.hidden_loss.is_finite() && (0.0..=1.0).contains(&event.hidden_loss), "hidden loss outside [0, 1]");
        ensure!(event.observed_covariate.is_finite() && (0.0..=1.0).contains(&event.observed_covariate), "covariate outside [0, 1]");
        ensure!(event.prediction_error_signal.is_finite() && (0.0..=1.0).contains(&event.prediction_error_signal), "error signal outside [0, 1]");
    }
    Ok(())
}

fn independently_rederive_records(input: &ScheduleInput) -> anyhow::Result<Vec<EvaluatorOutcomeRecord>> {
    let mut events = input.events.iter().collect::<Vec<_>>();
    events.sort_by(|left, right| left.event_id.cmp(&right.event_id));
    let mut records = Vec::with_capacity(events.len());
    for event in events {
        let p_missing = probability(&input.config, event);
        let was_selected_missing = uniform01(input.seed, &event.event_id, "missingness-selection") < p_missing;
        let mut delay_ticks = None;
        let mut expected_reveal_tick = None;
        let mut declaration_tick = None;

        let state_at_horizon = if was_selected_missing {
            match input.config.declaration_mode {
                MissingnessDeclarationMode::Explicit => {
                    let tick = event.decision_tick.checked_add(input.config.declaration_delay_ticks)
                        .context("declaration tick overflow during audit")?;
                    declaration_tick = Some(tick);
                    if tick <= input.evaluation_horizon_tick {
                        OutcomeStateAtHorizon::DeclaredMissing
                    } else {
                        OutcomeStateAtHorizon::DelayedPending
                    }
                }
                MissingnessDeclarationMode::Silent => OutcomeStateAtHorizon::UnknownUnnotified,
            }
        } else {
            let delay = delay_ticks_for(input.seed, &event.event_id, &input.config.delay)?;
            let reveal_tick = event.decision_tick.checked_add(delay)
                .context("reveal tick overflow during audit")?;
            delay_ticks = Some(delay);
            expected_reveal_tick = Some(reveal_tick);
            if reveal_tick <= input.evaluation_horizon_tick {
                OutcomeStateAtHorizon::Observed
            } else {
                OutcomeStateAtHorizon::RightCensoredAtHorizon
            }
        };

        records.push(EvaluatorOutcomeRecord {
            event_id: event.event_id.clone(),
            policy_event_id: opaque_policy_event_id(input.seed, &event.event_id),
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
    Ok(records)
}

fn independently_rederive_policy_messages(input: &ScheduleInput) -> anyhow::Result<Vec<PolicyVisibleMessage>> {
    let records = independently_rederive_records(input)?;
    let source_events = input.events.iter().map(|event| (event.event_id.as_str(), event)).collect::<BTreeMap<_, _>>();
    let mut messages = Vec::with_capacity(input.events.len() * 2);
    for record in records {
        let event = source_events.get(record.event_id.as_str()).context("missing source event for evaluator record")?;
        let policy_id = record.policy_event_id;
        messages.push(PolicyVisibleMessage::OutcomePending {
            tick: event.decision_tick,
            event_id: policy_id.clone(),
            observed_covariate: event.observed_covariate,
            action_taken: event.action_taken,
        });
        if record.state_at_horizon == OutcomeStateAtHorizon::Observed {
            messages.push(PolicyVisibleMessage::OutcomeRevealed {
                tick: record.expected_reveal_tick.context("observed event has no reveal tick")?,
                event_id: policy_id,
                outcome_loss: record.hidden_loss,
            });
        } else if record.state_at_horizon == OutcomeStateAtHorizon::DeclaredMissing {
            messages.push(PolicyVisibleMessage::MissingDeclared {
                tick: record.declaration_tick.context("declared-missing event has no declaration tick")?,
                event_id: policy_id,
            });
        }
    }
    messages.sort_by_key(message_key);
    Ok(messages)
}

fn probability(config: &MissingnessScheduleConfig, event: &HiddenOutcomeEvent) -> f64 {
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

fn delay_ticks_for(seed: u64, event_id: &str, pattern: &DelayPattern) -> anyhow::Result<u64> {
    match *pattern {
        DelayPattern::Fixed { ticks } => Ok(ticks),
        DelayPattern::HeterogeneousBounded { max_ticks } => {
            let modulus = max_ticks.checked_add(1).context("max delay cap overflow")?;
            Ok(deterministic_u64(seed, event_id, "outcome-delay") % modulus)
        }
        DelayPattern::HeavyTailedCapped { scale_ticks, cap_ticks } => {
            ensure!(scale_ticks > 0 && cap_ticks > 0, "heavy-tailed delay parameters must be positive");
            let u = uniform01(seed, event_id, "outcome-delay");
            let raw = (scale_ticks as f64) * (u / (1.0 - u));
            if !raw.is_finite() || raw >= cap_ticks as f64 {
                Ok(cap_ticks)
            } else {
                Ok((raw.floor() as u64).min(cap_ticks))
            }
        }
    }
}

fn message_tick(message: &PolicyVisibleMessage) -> u64 {
    match message {
        PolicyVisibleMessage::OutcomePending { tick, .. }
        | PolicyVisibleMessage::OutcomeRevealed { tick, .. }
        | PolicyVisibleMessage::MissingDeclared { tick, .. } => *tick,
    }
}

fn message_key(message: &PolicyVisibleMessage) -> (u64, u8, String) {
    match message {
        PolicyVisibleMessage::OutcomePending { tick, event_id, .. } => (*tick, 0, event_id.clone()),
        PolicyVisibleMessage::OutcomeRevealed { tick, event_id, .. } => (*tick, 1, event_id.clone()),
        PolicyVisibleMessage::MissingDeclared { tick, event_id } => (*tick, 1, event_id.clone()),
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
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_missingness_schedule::{self, DelayPattern, HiddenOutcomeEvent, MissingnessScheduleConfig, ScheduleInput};

    fn fixture() -> (ScheduleInput, Vec<u8>) {
        let events = (0..64)
            .map(|index| HiddenOutcomeEvent {
                event_id: format!("source-id-{index:03}"),
                decision_tick: index,
                hidden_loss: (index % 10) as f64 / 9.0,
                observed_covariate: ((index * 7) % 10) as f64 / 9.0,
                prediction_error_signal: ((index * 3) % 10) as f64 / 9.0,
                action_taken: index % 3 == 0,
            })
            .collect();
        let input = ScheduleInput {
            schema_version: 1,
            run_id: "audit-unit-run".into(),
            dataset_id: "audit-unit-dataset".into(),
            scenario_id: "audit-unit-scenario".into(),
            seed: 47,
            evaluation_horizon_tick: 96,
            config: MissingnessScheduleConfig {
                mechanism: MissingnessMechanism::MnarHighLoss,
                base_missing_probability: 0.15,
                missing_probability_slope: 0.7,
                declaration_mode: MissingnessDeclarationMode::Explicit,
                declaration_delay_ticks: 1,
                delay: DelayPattern::HeterogeneousBounded { max_ticks: 24 },
            },
            events,
        };
        let mut input_bytes = serde_json::to_vec_pretty(&input).unwrap();
        input_bytes.push(b'\n');
        (input, input_bytes)
    }

    fn artifacts(tick: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (input, input_bytes) = fixture();
        let generated = cognitive_missingness_schedule::generate(&input, &input_bytes).unwrap();
        let mut policy = serde_json::to_vec_pretty(&generated.policy_tick_input(tick)).unwrap();
        policy.push(b'\n');
        let mut evaluator = serde_json::to_vec_pretty(&generated.evaluator_schedule).unwrap();
        evaluator.push(b'\n');
        (input_bytes, policy, evaluator)
    }

    #[test]
    fn accepts_consistent_deterministically_rederived_artifacts() {
        let (input, policy, evaluator) = artifacts(19);
        let report = audit_bytes(&input, &policy, &evaluator).unwrap();
        assert!(report.state_count_conserved);
        assert!(report.evaluator_records_rederived);
        assert!(report.policy_tick_rederived);
        assert!(report.opaque_policy_ids_verified);
        assert!(report.no_future_tick_messages);
        assert_eq!(report.eligible_event_count, 64);
        assert_eq!(report.result_sha256.as_ref().unwrap().len(), 64);
    }

    #[test]
    fn rejects_policy_message_from_a_future_tick() {
        let (input, policy, evaluator) = artifacts(19);
        let mut policy_json: PolicyTickInput = serde_json::from_slice(&policy).unwrap();
        policy_json.messages.push(PolicyVisibleMessage::OutcomePending {
            tick: 20,
            event_id: "future".into(),
            observed_covariate: 0.0,
            action_taken: false,
        });
        let bad_policy = serde_json::to_vec(&policy_json).unwrap();
        let err = audit_bytes(&input, &bad_policy, &evaluator).unwrap_err();
        assert!(err.to_string().contains("different logical tick"));
    }

    #[test]
    fn rejects_tampered_hidden_truth_or_selection_receipts() {
        let (input, policy, evaluator) = artifacts(19);
        let mut schedule: EvaluatorSchedule = serde_json::from_slice(&evaluator).unwrap();
        schedule.events[0].hidden_loss = 0.99;
        let bad = serde_json::to_vec(&schedule).unwrap();
        assert!(audit_bytes(&input, &policy, &bad).unwrap_err().to_string().contains("independently re-derived"));
    }

    #[test]
    fn rejects_wrong_source_digest_even_when_json_parses() {
        let (input, policy, evaluator) = artifacts(19);
        let mut changed_input = input.clone();
        changed_input.push(b' ');
        let err = audit_bytes(&changed_input, &policy, &evaluator).unwrap_err();
        assert!(err.to_string().contains("source-input digest"));
    }

    #[test]
    fn rejects_missing_eligible_event_from_evaluator_census() {
        let (input, policy, evaluator) = artifacts(19);
        let mut schedule: EvaluatorSchedule = serde_json::from_slice(&evaluator).unwrap();
        schedule.events.pop();
        let bad = serde_json::to_vec(&schedule).unwrap();
        assert!(audit_bytes(&input, &policy, &bad).unwrap_err().to_string().contains("independently re-derived"));
    }

    #[test]
    fn rejects_non_opaque_policy_identifier() {
        let (input, policy, evaluator) = artifacts(19);
        let mut policy_json: PolicyTickInput = serde_json::from_slice(&policy).unwrap();
        if let Some(message) = policy_json.messages.first_mut() {
            match message {
                PolicyVisibleMessage::OutcomePending { event_id, .. }
                | PolicyVisibleMessage::OutcomeRevealed { event_id, .. }
                | PolicyVisibleMessage::MissingDeclared { event_id, .. } => *event_id = "source-id-001".into(),
            }
        }
        let bad = serde_json::to_vec(&policy_json).unwrap();
        assert!(audit_bytes(&input, &bad, &evaluator).unwrap_err().to_string().contains("policy-visible tick messages"));
    }

    #[test]
    fn rejects_policy_tick_beyond_horizon() {
        let (input, _policy, evaluator) = artifacts(19);
        let policy = PolicyTickInput {
            schema_version: 1,
            generator_version: "cognitive-missingness-schedule-v1".into(),
            tick: 97,
            messages: vec![],
        };
        let bad = serde_json::to_vec(&policy).unwrap();
        assert!(audit_bytes(&input, &bad, &evaluator).unwrap_err().to_string().contains("exceeds evaluation horizon"));
    }
}
