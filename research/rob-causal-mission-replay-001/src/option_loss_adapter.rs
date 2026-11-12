//! Retrospective join for Symtropy future-option loss/restoration receipts.
//!
//! Consumes the upstream finite option-set ledger and joins its identities to
//! existing MissionTrace records. Does not recompute viability, physical
//! branches, thermal state, topology, actuator authority or restoration costs.

use super::{FindingSeverity, MissionTrace, ReplayDisposition, replay_mission};

/// Upstream option-ledger evidence state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionLedgerEvidenceDisposition {
    /// The upstream ledger qualified this exact transition.
    Qualified,
    /// Required evidence is unavailable or incomplete.
    Unknown,
    /// The declared evidence boundary cannot resolve the claim.
    Inconclusive,
    /// The upstream ledger rejected the transition.
    Rejected,
    /// Future-option evaluation was not performed.
    NotEvaluated,
}

/// Final disposition after joining the receipt to the mission trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionLedgerReplayDisposition {
    /// Receipt and exact causal/physical bindings are qualified.
    Qualified,
    /// Required receipt or mission evidence is unavailable.
    Unknown,
    /// Upstream evidence is inconclusive.
    Inconclusive,
    /// A deterministic identity, sequence or authority invariant is contradicted.
    Rejected,
    /// Upstream future-option evaluation was not performed.
    NotEvaluated,
}

/// Versioned option-loss/restoration receipt from Symtropy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionLossLedgerReceipt {
    /// Protocol identity.
    pub protocol_id: String,
    /// Wire schema version.
    pub schema_version: String,
    /// Producer crate identity.
    pub producer_package: String,
    /// Exact producer commit SHA.
    pub producer_commit_sha: String,
    /// Canonical payload SHA-256.
    pub payload_sha256: String,
    /// Mission identity.
    pub mission_id: String,
    /// Robot identity.
    pub robot_id: String,
    /// Scenario manifest digest.
    pub scenario_manifest_digest: String,
    /// Exact decision identity.
    pub decision_id: String,
    /// Ordered source observation identities.
    pub source_observation_ids: Vec<String>,
    /// Exact continuation-action domain digest.
    pub action_domain_digest: String,
    /// Delivered-observation partition digest.
    pub observation_partition_digest: String,
    /// Physically possible successor-branch manifest digest.
    pub branch_manifest_digest: String,
    /// Whether the branch manifest is complete.
    pub branch_manifest_complete: bool,
    /// Whether the action domain is complete.
    pub action_domain_complete: bool,
    /// Physical transition identity.
    pub physical_transition_id: String,
    /// Exact sequence of the physical transition.
    pub physical_transition_sequence: u64,
    /// Ledger event sequence; must follow the physical transition.
    pub ledger_event_sequence: u64,
    /// Reserve digest before the transition.
    pub predecessor_reserve_digest: String,
    /// Reserve digest after the transition.
    pub result_reserve_digest: String,
    /// Prior common-continuation option-set digest.
    pub predecessor_option_set_digest: Option<String>,
    /// Result common-continuation option-set digest.
    pub result_option_set_digest: Option<String>,
    /// This option-ledger transition digest.
    pub ledger_transition_digest: String,
    /// Number of option-loss entries.
    pub loss_entry_count: u64,
    /// Number of option-restoration entries.
    pub restoration_entry_count: u64,
    /// Upstream evidence disposition.
    pub disposition: OptionLedgerEvidenceDisposition,
    /// Must be false: retrospective accounting grants no actuation authority.
    pub actuation_authorized: bool,
}

/// Reason-coded adapter finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionLedgerFinding {
    /// Invalid or Unknown severity.
    pub severity: FindingSeverity,
    /// Stable reason code.
    pub code: &'static str,
    /// Primary record identity.
    pub record_id: String,
}

/// Result of the causal receipt join.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionLedgerReplayReport {
    /// Final conservative disposition.
    pub disposition: OptionLedgerReplayDisposition,
    /// Original upstream disposition, retained unchanged.
    pub producer_disposition: OptionLedgerEvidenceDisposition,
    /// Result of the existing mission replay.
    pub causal_replay_disposition: ReplayDisposition,
    /// Joined decision identity.
    pub decision_id: String,
    /// Joined physical transition identity.
    pub physical_transition_id: String,
    /// Upstream loss count.
    pub loss_entry_count: u64,
    /// Upstream restoration count.
    pub restoration_entry_count: u64,
    /// Join findings.
    pub findings: Vec<OptionLedgerFinding>,
    /// Always false.
    pub actuation_authorized: bool,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn lower_hex(value: &str, size: usize) -> bool {
    value.len() == size
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn push(
    findings: &mut Vec<OptionLedgerFinding>,
    severity: FindingSeverity,
    code: &'static str,
    record_id: impl Into<String>,
) {
    findings.push(OptionLedgerFinding {
        severity,
        code,
        record_id: record_id.into(),
    });
}

fn replay_state(value: ReplayDisposition) -> OptionLedgerReplayDisposition {
    match value {
        ReplayDisposition::Closed => OptionLedgerReplayDisposition::Qualified,
        ReplayDisposition::Unknown => OptionLedgerReplayDisposition::Unknown,
        ReplayDisposition::Rejected => OptionLedgerReplayDisposition::Rejected,
    }
}

fn producer_state(value: OptionLedgerEvidenceDisposition) -> OptionLedgerReplayDisposition {
    match value {
        OptionLedgerEvidenceDisposition::Qualified => OptionLedgerReplayDisposition::Qualified,
        OptionLedgerEvidenceDisposition::Unknown => OptionLedgerReplayDisposition::Unknown,
        OptionLedgerEvidenceDisposition::Inconclusive => {
            OptionLedgerReplayDisposition::Inconclusive
        }
        OptionLedgerEvidenceDisposition::Rejected => OptionLedgerReplayDisposition::Rejected,
        OptionLedgerEvidenceDisposition::NotEvaluated => {
            OptionLedgerReplayDisposition::NotEvaluated
        }
    }
}

fn conservative(
    a: OptionLedgerReplayDisposition,
    b: OptionLedgerReplayDisposition,
) -> OptionLedgerReplayDisposition {
    use OptionLedgerReplayDisposition::{Inconclusive, NotEvaluated, Qualified, Rejected, Unknown};
    if a == Rejected || b == Rejected {
        Rejected
    } else if a == Unknown || b == Unknown {
        Unknown
    } else if a == Inconclusive || b == Inconclusive {
        Inconclusive
    } else if a == NotEvaluated || b == NotEvaluated {
        NotEvaluated
    } else if a == Qualified && b == Qualified {
        Qualified
    } else {
        Unknown
    }
}

/// Join an upstream option-ledger receipt to its exact decision and physical transition.
///
/// The base mission is replayed first. Receipt scope, decision/source lineage,
/// action domain, physical-transition sequence and reserve-state continuity are
/// checked exactly. A producer disposition can only be preserved or downgraded.
pub fn replay_option_loss_receipt(
    trace: &MissionTrace,
    receipt: &OptionLossLedgerReceipt,
) -> OptionLedgerReplayReport {
    let mut findings = Vec::new();
    let causal = replay_mission(trace);
    let mut disposition = conservative(
        replay_state(causal.disposition),
        producer_state(receipt.disposition),
    );
    let receipt_id = if nonempty(&receipt.physical_transition_id) {
        receipt.physical_transition_id.clone()
    } else {
        "<missing-physical-transition>".to_owned()
    };

    if receipt.actuation_authorized {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "OPTION_LEDGER_ACTUATION_AUTHORITY_FORBIDDEN",
            &receipt_id,
        );
        disposition = OptionLedgerReplayDisposition::Rejected;
    }

    for (field, value) in [
        ("protocol_id", receipt.protocol_id.as_str()),
        ("schema_version", receipt.schema_version.as_str()),
        ("producer_package", receipt.producer_package.as_str()),
        ("producer_commit_sha", receipt.producer_commit_sha.as_str()),
        ("payload_sha256", receipt.payload_sha256.as_str()),
        ("mission_id", receipt.mission_id.as_str()),
        ("robot_id", receipt.robot_id.as_str()),
        (
            "scenario_manifest_digest",
            receipt.scenario_manifest_digest.as_str(),
        ),
        ("decision_id", receipt.decision_id.as_str()),
        (
            "action_domain_digest",
            receipt.action_domain_digest.as_str(),
        ),
        (
            "observation_partition_digest",
            receipt.observation_partition_digest.as_str(),
        ),
        (
            "branch_manifest_digest",
            receipt.branch_manifest_digest.as_str(),
        ),
        (
            "physical_transition_id",
            receipt.physical_transition_id.as_str(),
        ),
        (
            "predecessor_reserve_digest",
            receipt.predecessor_reserve_digest.as_str(),
        ),
        (
            "result_reserve_digest",
            receipt.result_reserve_digest.as_str(),
        ),
        (
            "ledger_transition_digest",
            receipt.ledger_transition_digest.as_str(),
        ),
    ] {
        if !nonempty(value) {
            push(
                &mut findings,
                FindingSeverity::Unknown,
                "OPTION_LEDGER_REQUIRED_FIELD_MISSING",
                format!("{receipt_id}:{field}"),
            );
            disposition = conservative(disposition, OptionLedgerReplayDisposition::Unknown);
        }
    }

    let valid_protocol = receipt.protocol_id == "LOCO-OPTION-LOSS-LEDGER-001"
        && receipt.schema_version == "0.1.0"
        && receipt.producer_package == "loco-option-loss-ledger"
        && lower_hex(&receipt.producer_commit_sha, 40)
        && lower_hex(&receipt.payload_sha256, 64)
        && lower_hex(&receipt.ledger_transition_digest, 64)
        && lower_hex(&receipt.scenario_manifest_digest, 64)
        && lower_hex(&receipt.action_domain_digest, 64)
        && lower_hex(&receipt.observation_partition_digest, 64)
        && lower_hex(&receipt.branch_manifest_digest, 64)
        && lower_hex(&receipt.predecessor_reserve_digest, 64)
        && lower_hex(&receipt.result_reserve_digest, 64);
    if !valid_protocol {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "OPTION_LEDGER_PROTOCOL_OR_INTEGRITY_IDENTITY_INVALID",
            &receipt_id,
        );
        disposition = OptionLedgerReplayDisposition::Rejected;
    }

    if receipt.mission_id != trace.scope.mission_id
        || receipt.robot_id != trace.scope.robot_id
        || receipt.scenario_manifest_digest != trace.scope.scenario_manifest_digest
    {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "OPTION_LEDGER_MISSION_SCOPE_MISMATCH",
            &receipt_id,
        );
        disposition = OptionLedgerReplayDisposition::Rejected;
    }

    let matching_decisions = trace
        .decisions
        .iter()
        .filter(|v| v.decision_id == receipt.decision_id)
        .collect::<Vec<_>>();
    match matching_decisions.as_slice() {
        [decision] => {
            if decision.action_domain_digest != receipt.action_domain_digest
                || decision.source_observation_ids != receipt.source_observation_ids
            {
                push(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "OPTION_LEDGER_DECISION_OR_OBSERVATION_BINDING_MISMATCH",
                    &receipt.decision_id,
                );
                disposition = OptionLedgerReplayDisposition::Rejected;
            }
        }
        [] => {
            push(
                &mut findings,
                FindingSeverity::Unknown,
                "OPTION_LEDGER_DECISION_NOT_FOUND",
                &receipt.decision_id,
            );
            disposition = conservative(disposition, OptionLedgerReplayDisposition::Unknown);
        }
        _ => {
            push(
                &mut findings,
                FindingSeverity::Invalid,
                "OPTION_LEDGER_DECISION_ID_AMBIGUOUS",
                &receipt.decision_id,
            );
            disposition = OptionLedgerReplayDisposition::Rejected;
        }
    }

    let matching_transitions = trace
        .physical_transitions
        .iter()
        .filter(|v| v.transition_id == receipt.physical_transition_id)
        .collect::<Vec<_>>();
    match matching_transitions.as_slice() {
        [transition] => {
            if transition.stamp.sequence != receipt.physical_transition_sequence
                || transition.predecessor_state_digest != receipt.predecessor_reserve_digest
                || transition.resulting_state_digest != receipt.result_reserve_digest
            {
                push(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "OPTION_LEDGER_PHYSICAL_RESERVE_BINDING_MISMATCH",
                    &receipt_id,
                );
                disposition = OptionLedgerReplayDisposition::Rejected;
            }
            if receipt.ledger_event_sequence <= transition.stamp.sequence {
                push(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "OPTION_LEDGER_EVENT_NOT_AFTER_PHYSICAL_TRANSITION",
                    &receipt_id,
                );
                disposition = OptionLedgerReplayDisposition::Rejected;
            }
            let applications = trace
                .applications
                .iter()
                .filter(|v| v.command_id == transition.command_id)
                .collect::<Vec<_>>();
            match applications.as_slice() {
                [application] => {
                    if application.decision_id != receipt.decision_id
                        || application.source_observation_ids != receipt.source_observation_ids
                    {
                        push(
                            &mut findings,
                            FindingSeverity::Invalid,
                            "OPTION_LEDGER_APPLICATION_LINEAGE_MISMATCH",
                            &transition.command_id,
                        );
                        disposition = OptionLedgerReplayDisposition::Rejected;
                    }
                }
                [] => {
                    push(
                        &mut findings,
                        FindingSeverity::Unknown,
                        "OPTION_LEDGER_COMMAND_APPLICATION_NOT_FOUND",
                        &transition.command_id,
                    );
                    disposition = conservative(disposition, OptionLedgerReplayDisposition::Unknown);
                }
                _ => {
                    push(
                        &mut findings,
                        FindingSeverity::Invalid,
                        "OPTION_LEDGER_COMMAND_APPLICATION_AMBIGUOUS",
                        &transition.command_id,
                    );
                    disposition = OptionLedgerReplayDisposition::Rejected;
                }
            }
        }
        [] => {
            push(
                &mut findings,
                FindingSeverity::Unknown,
                "OPTION_LEDGER_PHYSICAL_TRANSITION_NOT_FOUND",
                &receipt_id,
            );
            disposition = conservative(disposition, OptionLedgerReplayDisposition::Unknown);
        }
        _ => {
            push(
                &mut findings,
                FindingSeverity::Invalid,
                "OPTION_LEDGER_PHYSICAL_TRANSITION_ID_AMBIGUOUS",
                &receipt_id,
            );
            disposition = OptionLedgerReplayDisposition::Rejected;
        }
    }

    if !receipt.branch_manifest_complete || !receipt.action_domain_complete {
        push(
            &mut findings,
            FindingSeverity::Unknown,
            "OPTION_LEDGER_BRANCH_OR_ACTION_DOMAIN_INCOMPLETE",
            &receipt_id,
        );
        disposition = conservative(disposition, OptionLedgerReplayDisposition::Unknown);
    }

    match receipt.disposition {
        OptionLedgerEvidenceDisposition::Qualified => {
            let has_both_sets = receipt
                .predecessor_option_set_digest
                .as_ref()
                .is_some_and(|v| lower_hex(v, 64))
                && receipt
                    .result_option_set_digest
                    .as_ref()
                    .is_some_and(|v| lower_hex(v, 64));
            if !has_both_sets {
                push(
                    &mut findings,
                    FindingSeverity::Unknown,
                    "OPTION_LEDGER_QUALIFIED_SET_DIGEST_MISSING",
                    &receipt_id,
                );
                disposition = conservative(disposition, OptionLedgerReplayDisposition::Unknown);
            }
        }
        OptionLedgerEvidenceDisposition::NotEvaluated => {
            if receipt.predecessor_option_set_digest.is_some()
                || receipt.result_option_set_digest.is_some()
                || receipt.loss_entry_count != 0
                || receipt.restoration_entry_count != 0
            {
                push(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "OPTION_LEDGER_NOT_EVALUATED_CONTAINS_RESULTS",
                    &receipt_id,
                );
                disposition = OptionLedgerReplayDisposition::Rejected;
            } else {
                disposition =
                    conservative(disposition, OptionLedgerReplayDisposition::NotEvaluated);
            }
        }
        OptionLedgerEvidenceDisposition::Unknown
        | OptionLedgerEvidenceDisposition::Inconclusive
        | OptionLedgerEvidenceDisposition::Rejected => {}
    }

    OptionLedgerReplayReport {
        disposition,
        producer_disposition: receipt.disposition,
        causal_replay_disposition: causal.disposition,
        decision_id: receipt.decision_id.clone(),
        physical_transition_id: receipt.physical_transition_id.clone(),
        loss_entry_count: receipt.loss_entry_count,
        restoration_entry_count: receipt.restoration_entry_count,
        findings,
        actuation_authorized: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActionChunkRecord, ActuatorApplication, AssuranceDisposition, AssuranceRecord,
        CalibrationDisposition, DecisionRecord, EventStamp, MissionScope, PhysicalTransition,
        SourceObservation, TerminalDisposition, TerminalRecord, TimestampInterval,
        TopologyDisposition,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn stamp(id: &str, sequence: u64, tick: u64, time: u64) -> EventStamp {
        EventStamp {
            event_id: id.into(),
            sequence,
            tick,
            robot_id: "robot-01".into(),
            clock_id: "robot-clock".into(),
            boot_epoch: 7,
            timestamp: TimestampInterval {
                lower_ns: time,
                upper_ns: time,
            },
        }
    }

    fn trace() -> MissionTrace {
        MissionTrace {
            scope: MissionScope {
                mission_id: "mission-1".into(),
                robot_id: "robot-01".into(),
                clock_id: "robot-clock".into(),
                boot_epoch: 7,
                scenario_manifest_digest: digest('a'),
                initial_reserve_state_digest: digest('f'),
                required_thermal_node_ids: vec!["motor".into(), "driver".into()],
            },
            observations: vec![SourceObservation {
                observation_id: "obs-1".into(),
                stamp: stamp("event-1", 1, 1, 100),
                modality: "camera".into(),
                source_sequence: 1,
                sensor_digest: "sensor-sha256".into(),
            }],
            decisions: vec![DecisionRecord {
                decision_id: "decision-1".into(),
                stamp: stamp("event-2", 2, 1, 110),
                selected_action_id: "walk-forward".into(),
                source_observation_ids: vec!["obs-1".into()],
                loss_model_digest: "loss-model".into(),
                policy_digest: "policy".into(),
                loss_scale_id: "task-loss".into(),
                action_domain_digest: digest('c'),
                calibration_id: "cal-1".into(),
                calibration_digest: "cal-digest".into(),
                calibration_disposition: CalibrationDisposition::Qualified,
                regret_upper_units: 2,
                epsilon_task_loss_units: 5,
                valid_through_tick: 10,
            }],
            remote_requests: Vec::new(),
            chunks: vec![ActionChunkRecord {
                proposal_id: "proposal-1".into(),
                stamp: stamp("event-3", 3, 2, 120),
                decision_id: "decision-1".into(),
                request_id: None,
                action_id: "walk-forward".into(),
                model_digest: "loss-model".into(),
                policy_digest: "policy".into(),
                action_domain_digest: digest('c'),
                expected_executed_prefix_digest: "prefix-0".into(),
                observed_executed_prefix_digest: "prefix-0".into(),
            }],
            assurance: vec![AssuranceRecord {
                proposal_id: "proposal-1".into(),
                stamp: stamp("event-4", 4, 2, 130),
                disposition: AssuranceDisposition::Admitted,
            }],
            applications: vec![ActuatorApplication {
                command_id: "command-1".into(),
                stamp: stamp("event-5", 5, 3, 200),
                proposal_id: "proposal-1".into(),
                decision_id: "decision-1".into(),
                action_id: "walk-forward".into(),
                source_observation_ids: vec!["obs-1".into()],
                request_id: None,
                actuator_id: "left-knee".into(),
                actuator_configuration_digest: "actuator-config".into(),
                controller_configuration_digest: "controller-config".into(),
                application_receipt_digest: "apply-digest".into(),
            }],
            physical_transitions: vec![PhysicalTransition {
                transition_id: "transition-1".into(),
                stamp: stamp("event-6", 6, 3, 220),
                command_id: "command-1".into(),
                predecessor_state_digest: digest('f'),
                resulting_state_digest: digest('d'),
                thermal_model_digest: "thermal-model".into(),
                thermal_snapshot_digest: "thermal-snapshot".into(),
                thermal_sensor_calibration_digest: "thermal-cal".into(),
                thermal_node_ids: vec!["motor".into(), "driver".into()],
                topology: TopologyDisposition::Known,
            }],
            terminal: Some(TerminalRecord {
                stamp: stamp("event-7", 7, 4, 300),
                disposition: TerminalDisposition::SafeStopQualified,
                reserve_state_digest: digest('d'),
            }),
        }
    }

    fn receipt() -> OptionLossLedgerReceipt {
        OptionLossLedgerReceipt {
            protocol_id: "LOCO-OPTION-LOSS-LEDGER-001".into(),
            schema_version: "0.1.0".into(),
            producer_package: "loco-option-loss-ledger".into(),
            producer_commit_sha: "a".repeat(40),
            payload_sha256: digest('b'),
            mission_id: "mission-1".into(),
            robot_id: "robot-01".into(),
            scenario_manifest_digest: digest('a'),
            decision_id: "decision-1".into(),
            source_observation_ids: vec!["obs-1".into()],
            action_domain_digest: digest('c'),
            observation_partition_digest: digest('9'),
            branch_manifest_digest: digest('8'),
            branch_manifest_complete: true,
            action_domain_complete: true,
            physical_transition_id: "transition-1".into(),
            physical_transition_sequence: 6,
            ledger_event_sequence: 8,
            predecessor_reserve_digest: digest('f'),
            result_reserve_digest: digest('d'),
            predecessor_option_set_digest: Some(digest('7')),
            result_option_set_digest: Some(digest('6')),
            ledger_transition_digest: digest('e'),
            loss_entry_count: 1,
            restoration_entry_count: 0,
            disposition: OptionLedgerEvidenceDisposition::Qualified,
            actuation_authorized: false,
        }
    }

    #[test]
    fn qualified_receipt_joins_exact_decision_and_physical_transition() {
        let result = replay_option_loss_receipt(&trace(), &receipt());
        assert_eq!(result.disposition, OptionLedgerReplayDisposition::Qualified);
        assert_eq!(result.causal_replay_disposition, ReplayDisposition::Closed);
        assert_eq!(result.loss_entry_count, 1);
        assert!(!result.actuation_authorized);
    }

    #[test]
    fn wrong_mission_scope_is_rejected() {
        let mut value = receipt();
        value.mission_id = "other".into();
        assert_eq!(
            replay_option_loss_receipt(&trace(), &value).disposition,
            OptionLedgerReplayDisposition::Rejected
        );
    }

    #[test]
    fn wrong_action_domain_or_observation_lineage_is_rejected() {
        let mut domain = receipt();
        domain.action_domain_digest = digest('1');
        assert_eq!(
            replay_option_loss_receipt(&trace(), &domain).disposition,
            OptionLedgerReplayDisposition::Rejected
        );

        let mut observations = receipt();
        observations.source_observation_ids = vec!["other".into()];
        assert_eq!(
            replay_option_loss_receipt(&trace(), &observations).disposition,
            OptionLedgerReplayDisposition::Rejected
        );
    }

    #[test]
    fn physical_reserve_digests_and_sequence_must_match() {
        let mut value = receipt();
        value.predecessor_reserve_digest = digest('0');
        assert_eq!(
            replay_option_loss_receipt(&trace(), &value).disposition,
            OptionLedgerReplayDisposition::Rejected
        );

        let mut value = receipt();
        value.physical_transition_sequence = 5;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &value).disposition,
            OptionLedgerReplayDisposition::Rejected
        );
    }

    #[test]
    fn ledger_sequence_must_follow_physical_transition() {
        let mut value = receipt();
        value.ledger_event_sequence = 6;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &value).disposition,
            OptionLedgerReplayDisposition::Rejected
        );
    }

    #[test]
    fn missing_physical_transition_remains_unknown() {
        let mut mission = trace();
        mission.physical_transitions.clear();
        assert_eq!(
            replay_option_loss_receipt(&mission, &receipt()).disposition,
            OptionLedgerReplayDisposition::Unknown
        );
    }

    #[test]
    fn producer_unknown_and_inconclusive_are_never_upgraded() {
        let mut unknown = receipt();
        unknown.disposition = OptionLedgerEvidenceDisposition::Unknown;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &unknown).disposition,
            OptionLedgerReplayDisposition::Unknown
        );

        let mut inconclusive = receipt();
        inconclusive.disposition = OptionLedgerEvidenceDisposition::Inconclusive;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &inconclusive).disposition,
            OptionLedgerReplayDisposition::Inconclusive
        );
    }

    #[test]
    fn not_evaluated_remains_not_evaluated_when_empty() {
        let mut value = receipt();
        value.disposition = OptionLedgerEvidenceDisposition::NotEvaluated;
        value.predecessor_option_set_digest = None;
        value.result_option_set_digest = None;
        value.loss_entry_count = 0;
        value.restoration_entry_count = 0;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &value).disposition,
            OptionLedgerReplayDisposition::NotEvaluated
        );
    }

    #[test]
    fn incomplete_branch_or_action_domain_is_unknown() {
        let mut branch = receipt();
        branch.branch_manifest_complete = false;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &branch).disposition,
            OptionLedgerReplayDisposition::Unknown
        );

        let mut domain = receipt();
        domain.action_domain_complete = false;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &domain).disposition,
            OptionLedgerReplayDisposition::Unknown
        );
    }

    #[test]
    fn qualified_receipt_requires_both_option_set_digests() {
        let mut value = receipt();
        value.result_option_set_digest = None;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &value).disposition,
            OptionLedgerReplayDisposition::Unknown
        );
    }

    #[test]
    fn actuation_authority_is_always_forbidden() {
        let mut value = receipt();
        value.actuation_authorized = true;
        assert_eq!(
            replay_option_loss_receipt(&trace(), &value).disposition,
            OptionLedgerReplayDisposition::Rejected
        );
    }

    #[test]
    fn base_replay_rejection_downgrades_receipt() {
        let mut mission = trace();
        mission.physical_transitions[0].predecessor_state_digest = digest('0');
        let result = replay_option_loss_receipt(&mission, &receipt());
        assert_eq!(
            result.causal_replay_disposition,
            ReplayDisposition::Rejected
        );
        assert_eq!(result.disposition, OptionLedgerReplayDisposition::Rejected);
    }

    #[test]
    fn unsupported_schema_or_invalid_payload_digest_is_rejected() {
        let mut version = receipt();
        version.schema_version = "0.2.0".into();
        assert_eq!(
            replay_option_loss_receipt(&trace(), &version).disposition,
            OptionLedgerReplayDisposition::Rejected
        );

        let mut payload = receipt();
        payload.payload_sha256 = "not-a-digest".into();
        assert_eq!(
            replay_option_loss_receipt(&trace(), &payload).disposition,
            OptionLedgerReplayDisposition::Rejected
        );
    }
}
