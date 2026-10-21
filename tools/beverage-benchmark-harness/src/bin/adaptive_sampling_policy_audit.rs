// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed audit for one adaptive-sampling decision record.
//!
//! The audit checks internal consistency and temporal availability. It does not
//! establish that measurements, digests, calibrations, or required-test lists are
//! true. The declared input-prefix digest is shape-checked, not recomputed here.

use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const SCHEMA_VERSION: &str = "beverage-adaptive-sampling-decision-v1";
const CLAIM_CEILING: &str = "This audit checks declared channel feasibility, decision-time availability, digest syntax, required-test preservation, calibration-aware deferral, and action fields. It does not authenticate measurement truth, recompute the input-prefix digest, prove that the required-test inventory is complete, qualify a plant process, or authorize actuation.";

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum EvidenceClass {
    SyntheticContractFixture,
    RetrospectiveReplay,
    ProspectiveShadow,
    ProspectivePilot,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum CalibrationState {
    Calibrated,
    Unavailable,
    Failed,
    NotApplicable,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum PolicyAction {
    Sample,
    Defer,
    Escalate,
    Abstain,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ObservationRef {
    observation_id: String,
    channel: String,
    observed_at_ms: u64,
    available_at_ms: u64,
    content_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DecisionRecord {
    schema_version: String,
    evidence_class: EvidenceClass,
    decision_id: String,
    batch_id: String,
    policy_version: String,
    policy_config_sha256: String,
    input_prefix_sha256: String,
    decision_time_ms: u64,
    decision_deadline_ms: u64,
    available_channels: Vec<String>,
    observations: Vec<ObservationRef>,
    required_tests_due: Vec<String>,
    required_tests_scheduled: Vec<String>,
    calibration_state: CalibrationState,
    action: PolicyAction,
    requested_channel: Option<String>,
    requested_at_ms: Option<u64>,
    expected_result_available_at_ms: Option<u64>,
    next_review_at_ms: Option<u64>,
    expected_loss_without_sample_units: f64,
    expected_loss_with_sample_units: Option<f64>,
    estimated_sampling_cost_units: Option<f64>,
    estimated_delay_cost_units: Option<f64>,
    reason_code: String,
}

#[derive(Debug, Serialize)]
struct AuditReceipt<'a> {
    schema_version: &'static str,
    decision_id: &'a str,
    batch_id: &'a str,
    evidence_class: &'a EvidenceClass,
    disposition: &'static str,
    valid: bool,
    violations: Vec<String>,
    claim_ceiling: &'static str,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn has_duplicate_or_blank(values: &[String]) -> bool {
    values.iter().any(|value| value.trim().is_empty())
        || values.iter().collect::<BTreeSet<_>>().len() != values.len()
}

fn audit_record(record: &DecisionRecord) -> Vec<String> {
    let mut violations = Vec::new();

    if record.schema_version != SCHEMA_VERSION {
        violations.push("schema_version_mismatch".to_string());
    }

    for (field, value) in [
        ("decision_id", record.decision_id.as_str()),
        ("batch_id", record.batch_id.as_str()),
        ("policy_version", record.policy_version.as_str()),
        ("reason_code", record.reason_code.as_str()),
    ] {
        if value.trim().is_empty() {
            violations.push(format!("{field}_blank"));
        }
    }

    for (field, digest) in [
        ("policy_config_sha256", record.policy_config_sha256.as_str()),
        ("input_prefix_sha256", record.input_prefix_sha256.as_str()),
    ] {
        if !is_sha256_hex(digest) {
            violations.push(format!("{field}_invalid"));
        }
    }

    if record.available_channels.is_empty() {
        violations.push("available_channels_empty".to_string());
    }
    if has_duplicate_or_blank(&record.available_channels) {
        violations.push("available_channels_duplicate_or_blank".to_string());
    }
    if has_duplicate_or_blank(&record.required_tests_due) {
        violations.push("required_tests_due_duplicate_or_blank".to_string());
    }
    if has_duplicate_or_blank(&record.required_tests_scheduled) {
        violations.push("required_tests_scheduled_duplicate_or_blank".to_string());
    }

    for required_test in &record.required_tests_due {
        if !record.required_tests_scheduled.contains(required_test) {
            violations.push(format!("required_test_not_preserved:{required_test}"));
        }
    }

    let mut observation_ids = BTreeSet::new();
    for observation in &record.observations {
        if observation.observation_id.trim().is_empty()
            || !observation_ids.insert(observation.observation_id.as_str())
        {
            violations.push("observation_id_blank_or_duplicate".to_string());
        }
        if observation.channel.trim().is_empty()
            || !record.available_channels.contains(&observation.channel)
        {
            violations.push(format!(
                "observation_channel_unavailable:{}",
                observation.channel
            ));
        }
        if observation.observed_at_ms > observation.available_at_ms {
            violations.push(format!(
                "observation_available_before_observed:{}",
                observation.observation_id
            ));
        }
        if observation.available_at_ms > record.decision_time_ms {
            violations.push(format!(
                "future_information_used:{}",
                observation.observation_id
            ));
        }
        if !is_sha256_hex(&observation.content_sha256) {
            violations.push(format!(
                "observation_digest_invalid:{}",
                observation.observation_id
            ));
        }
    }

    if record.decision_deadline_ms < record.decision_time_ms {
        violations.push("decision_deadline_precedes_decision".to_string());
    }
    if !record.expected_loss_without_sample_units.is_finite()
        || record.expected_loss_without_sample_units < 0.0
    {
        violations.push("expected_loss_without_sample_invalid".to_string());
    }

    match record.action {
        PolicyAction::Sample => {
            match record.requested_channel.as_deref() {
                Some(channel) if record.available_channels.iter().any(|item| item == channel) => {}
                Some(channel) => {
                    violations.push(format!("requested_channel_unavailable:{channel}"));
                }
                None => violations.push("sample_missing_requested_channel".to_string()),
            }

            match record.requested_at_ms {
                Some(timestamp) if timestamp >= record.decision_time_ms => {}
                Some(_) => violations.push("sample_request_precedes_decision".to_string()),
                None => violations.push("sample_missing_requested_at".to_string()),
            }

            match record.expected_result_available_at_ms {
                Some(timestamp) => {
                    if let Some(requested_at) = record.requested_at_ms {
                        if timestamp < requested_at {
                            violations.push("sample_result_precedes_request".to_string());
                        }
                    }
                    if timestamp > record.decision_deadline_ms {
                        violations.push("sample_result_after_decision_deadline".to_string());
                    }
                }
                None => violations.push("sample_missing_result_availability_estimate".to_string()),
            }

            let loss_with_sample = match record.expected_loss_with_sample_units {
                Some(value) if value.is_finite() && value >= 0.0 => Some(value),
                Some(_) => {
                    violations.push("expected_loss_with_sample_invalid".to_string());
                    None
                }
                None => {
                    violations.push("sample_missing_expected_loss_with_sample".to_string());
                    None
                }
            };
            if record.estimated_sampling_cost_units
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            {
                violations.push("estimated_sampling_cost_units_invalid".to_string());
            }
            if record.estimated_delay_cost_units
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            {
                violations.push("estimated_delay_cost_units_invalid".to_string());
            }

            match (loss_with_sample, record.estimated_sampling_cost_units, record.estimated_delay_cost_units) {
                (Some(loss_after), Some(sample_cost), Some(delay_cost))
                    if sample_cost.is_finite() && delay_cost.is_finite()
                        && sample_cost >= 0.0 && delay_cost >= 0.0 =>
                {
                    let net_value = record.expected_loss_without_sample_units - loss_after - sample_cost - delay_cost;
                    if net_value <= 0.0 {
                        violations.push("sample_expected_net_value_not_positive".to_string());
                    }
                }
                _ => {
                    if record.estimated_sampling_cost_units.is_none() {
                        violations.push("sample_missing_sampling_cost_estimate".to_string());
                    }
                    if record.estimated_delay_cost_units.is_none() {
                        violations.push("sample_missing_delay_cost_estimate".to_string());
                    }
                }
            }

            if record.next_review_at_ms.is_some() {
                violations.push("sample_must_not_set_defer_review_time".to_string());
            }
        }
        PolicyAction::Defer => {
            if record.requested_channel.is_some() || record.requested_at_ms.is_some() {
                violations.push("defer_must_not_contain_sample_request".to_string());
            }
            if record.expected_result_available_at_ms.is_some()
                || record.expected_loss_with_sample_units.is_some()
                || record.estimated_sampling_cost_units.is_some()
                || record.estimated_delay_cost_units.is_some()
            {
                violations.push("non_sample_contains_value_of_information_fields".to_string());
            }
            match record.next_review_at_ms {
                Some(timestamp) if timestamp > record.decision_time_ms
                    && timestamp <= record.decision_deadline_ms => {}
                Some(timestamp) if timestamp <= record.decision_time_ms => {
                    violations.push("defer_review_not_in_future".to_string());
                }
                Some(_) => violations.push("defer_review_after_decision_deadline".to_string()),
                None => violations.push("defer_missing_next_review".to_string()),
            }
            if record.calibration_state != CalibrationState::Calibrated {
                violations.push("defer_requires_calibrated_uncertainty".to_string());
            }
        }
        PolicyAction::Escalate | PolicyAction::Abstain => {
            if record.requested_channel.is_some() || record.requested_at_ms.is_some() {
                violations.push("escalate_or_abstain_must_not_issue_sample_request".to_string());
            }
            if record.expected_result_available_at_ms.is_some()
                || record.expected_loss_with_sample_units.is_some()
                || record.estimated_sampling_cost_units.is_some()
                || record.estimated_delay_cost_units.is_some()
            {
                violations.push("non_sample_contains_value_of_information_fields".to_string());
            }
            if record.next_review_at_ms.is_some() {
                violations.push("escalate_or_abstain_must_not_set_defer_review_time".to_string());
            }
        }
    }

    violations.sort();
    violations.dedup();
    violations
}

fn parse_args() -> Result<(PathBuf, Option<PathBuf>), String> {
    let mut args = env::args().skip(1);
    let mut input = None;
    let mut output = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--input" => {
                input = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--input requires a path".to_string())?,
                ));
            }
            "--out" => {
                output = Some(PathBuf::from(
                    args.next().ok_or_else(|| "--out requires a path".to_string())?,
                ));
            }
            "--help" | "-h" => {
                return Err(
                    "usage: adaptive_sampling_policy_audit --input <decision.json> [--out <receipt.json>]"
                        .to_string(),
                );
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    Ok((
        input.ok_or_else(|| {
            "missing --input; use --input <decision.json> [--out <receipt.json>]".to_string()
        })?,
        output,
    ))
}

fn run() -> Result<ExitCode, String> {
    let (input_path, output_path) = parse_args()?;
    let raw = fs::read_to_string(&input_path)
        .map_err(|error| format!("cannot read {}: {error}", input_path.display()))?;
    let record: DecisionRecord = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid decision JSON: {error}"))?;
    let violations = audit_record(&record);
    let valid = violations.is_empty();
    let receipt = AuditReceipt {
        schema_version: SCHEMA_VERSION,
        decision_id: &record.decision_id,
        batch_id: &record.batch_id,
        evidence_class: &record.evidence_class,
        disposition: if valid { "ACCEPT_CONTRACT" } else { "REJECT" },
        valid,
        violations,
        claim_ceiling: CLAIM_CEILING,
    };
    let serialized = serde_json::to_string_pretty(&receipt)
        .map_err(|error| format!("cannot serialize audit receipt: {error}"))?;

    if let Some(path) = output_path {
        fs::write(&path, format!("{serialized}\n"))
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{serialized}");
    }

    Ok(if valid {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_record() -> DecisionRecord {
        DecisionRecord {
            schema_version: SCHEMA_VERSION.to_string(),
            evidence_class: EvidenceClass::SyntheticContractFixture,
            decision_id: "sense-decision-001".to_string(),
            batch_id: "batch-001".to_string(),
            policy_version: "risk-voi-v1".to_string(),
            policy_config_sha256: "a".repeat(64),
            input_prefix_sha256: "b".repeat(64),
            decision_time_ms: 1_000,
            decision_deadline_ms: 1_500,
            available_channels: vec!["temperature".to_string(), "off_gas".to_string()],
            observations: vec![ObservationRef {
                observation_id: "obs-001".to_string(),
                channel: "temperature".to_string(),
                observed_at_ms: 800,
                available_at_ms: 900,
                content_sha256: "c".repeat(64),
            }],
            required_tests_due: vec!["required-sugar-assay".to_string()],
            required_tests_scheduled: vec![
                "required-sugar-assay".to_string(),
                "discretionary-off-gas".to_string(),
            ],
            calibration_state: CalibrationState::Calibrated,
            action: PolicyAction::Sample,
            requested_channel: Some("off_gas".to_string()),
            requested_at_ms: Some(1_000),
            expected_result_available_at_ms: Some(1_200),
            next_review_at_ms: None,
            expected_loss_without_sample_units: 10.0,
            expected_loss_with_sample_units: Some(2.0),
            estimated_sampling_cost_units: Some(3.0),
            estimated_delay_cost_units: Some(1.0),
            reason_code: "decision_value_exceeds_cost".to_string(),
        }
    }

    fn make_defer(record: &mut DecisionRecord, next_review_at_ms: u64) {
        record.action = PolicyAction::Defer;
        record.requested_channel = None;
        record.requested_at_ms = None;
        record.expected_result_available_at_ms = None;
        record.next_review_at_ms = Some(next_review_at_ms);
        record.expected_loss_with_sample_units = None;
        record.estimated_sampling_cost_units = None;
        record.estimated_delay_cost_units = None;
    }

    #[test]
    fn accepts_consistent_sample_decision() {
        assert!(audit_record(&valid_record()).is_empty());
    }

    #[test]
    fn rejects_future_observation_availability() {
        let mut record = valid_record();
        record.observations[0].available_at_ms = record.decision_time_ms + 1;
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "future_information_used:obs-001")
        );
    }

    #[test]
    fn rejects_required_test_omission() {
        let mut record = valid_record();
        record.required_tests_scheduled.clear();
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "required_test_not_preserved:required-sugar-assay")
        );
    }

    #[test]
    fn rejects_unavailable_requested_channel() {
        let mut record = valid_record();
        record.requested_channel = Some("spectrometer_not_installed".to_string());
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "requested_channel_unavailable:spectrometer_not_installed")
        );
    }

    #[test]
    fn rejects_malformed_digests() {
        let mut record = valid_record();
        record.input_prefix_sha256 = "not-a-digest".to_string();
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "input_prefix_sha256_invalid")
        );
    }

    #[test]
    fn rejects_deferral_when_calibration_is_unavailable() {
        let mut record = valid_record();
        make_defer(&mut record, 1_200);
        record.calibration_state = CalibrationState::Unavailable;
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "defer_requires_calibrated_uncertainty")
        );
    }

    #[test]
    fn rejects_non_future_deferral_review() {
        let mut record = valid_record();
        make_defer(&mut record, record.decision_time_ms);
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "defer_review_not_in_future")
        );
    }

    #[test]
    fn rejects_sample_request_before_decision() {
        let mut record = valid_record();
        record.requested_at_ms = Some(record.decision_time_ms - 1);
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "sample_request_precedes_decision")
        );
    }

    #[test]
    fn rejects_negative_sampling_cost() {
        let mut record = valid_record();
        record.estimated_sampling_cost_units = Some(-0.1);
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "estimated_cost_units_invalid")
        );
    }

    #[test]
    fn rejects_duplicate_available_channels() {
        let mut record = valid_record();
        record.available_channels.push("temperature".to_string());
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "available_channels_duplicate_or_blank")
        );
    }

    #[test]
    fn rejects_duplicate_observation_ids() {
        let mut record = valid_record();
        record.observations.push(ObservationRef {
            observation_id: "obs-001".to_string(),
            channel: "off_gas".to_string(),
            observed_at_ms: 900,
            available_at_ms: 950,
            content_sha256: "d".repeat(64),
        });
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "observation_id_blank_or_duplicate")
        );
    }

    #[test]
    fn rejects_unknown_json_fields() {
        let value = serde_json::to_value(valid_record()).expect("record serializes");
        let mut object = value.as_object().expect("object").clone();
        object.insert("override_safety_floor".to_string(), serde_json::json!(true));
        assert!(serde_json::from_value::<DecisionRecord>(serde_json::Value::Object(object)).is_err());
    }

    #[test]
    fn allows_discretionary_deferral_when_required_test_remains_scheduled() {
        let mut record = valid_record();
        make_defer(&mut record, 1_200);
        assert!(audit_record(&record).is_empty());
    }

    #[test]
    fn rejects_observation_with_invalid_content_digest() {
        let mut record = valid_record();
        record.observations[0].content_sha256 = "xyz".to_string();
        assert!(
            audit_record(&record)
                .iter()
                .any(|issue| issue == "observation_digest_invalid:obs-001")
        );
    }
    #[test]
    fn rejects_sample_with_nonpositive_expected_net_value() {
        let mut record = valid_record();
        record.expected_loss_with_sample_units = Some(6.0);
        record.estimated_sampling_cost_units = Some(3.0);
        record.estimated_delay_cost_units = Some(1.0);
        assert!(audit_record(&record).iter().any(|issue| issue == "sample_expected_net_value_not_positive"));
    }

    #[test]
    fn rejects_sample_result_after_decision_deadline() {
        let mut record = valid_record();
        record.expected_result_available_at_ms = Some(record.decision_deadline_ms + 1);
        assert!(audit_record(&record).iter().any(|issue| issue == "sample_result_after_decision_deadline"));
    }

    #[test]
    fn rejects_deferral_past_decision_deadline() {
        let mut record = valid_record();
        make_defer(&mut record, record.decision_deadline_ms + 1);
        assert!(audit_record(&record).iter().any(|issue| issue == "defer_review_after_decision_deadline"));
    }

    #[test]
    fn rejects_decision_deadline_before_decision_time() {
        let mut record = valid_record();
        record.decision_deadline_ms = record.decision_time_ms - 1;
        assert!(audit_record(&record).iter().any(|issue| issue == "decision_deadline_precedes_decision"));
    }

}
