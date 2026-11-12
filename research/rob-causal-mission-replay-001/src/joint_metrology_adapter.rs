//! Adapter from instrumented single-joint metrology receipts to causal mission replay.
//!
//! This module does not estimate actuator, thermal, sensor, clock or energy
//! state. It checks externally produced receipts and composes them with the
//! authoritative MissionTrace. It is retrospective and never authorizes actuation.

use super::{
    ActuatorApplication, EventStamp, FindingSeverity, MissionTrace, ReplayDisposition,
    ReplayFinding, ReplayReport, replay_mission,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationDisposition {
    Qualified,
    Unknown,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyClosureDisposition {
    Balanced,
    Unknown,
    OutsideUncertainty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandResponseDisposition {
    WithinReference,
    Unknown,
    ExceedsReference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointInstrumentReceipt {
    pub actuator_id: String,
    pub actuator_configuration_digest: String,
    pub controller_configuration_digest: String,
    pub policy_digest: String,
    pub model_digest: String,
    pub action_domain_digest: String,
    pub sensor_calibration_digest: String,
    pub torque_transducer_calibration_digest: String,
    pub power_analyzer_calibration_digest: String,
    pub thermal_sensor_calibration_digest: String,
    pub sampling_spec_digest: String,
    pub energy_boundary_manifest_digest: String,
    pub decision_calibration_receipt_digest: String,
    pub actuator_application_receipt_digest: String,
    pub executed_prefix_expected_digest: String,
    pub executed_prefix_observed_digest: String,
    pub thermal_snapshot_digest: String,
    pub measured_sample_rate_hz: u64,
    pub required_sample_rate_hz: u64,
    pub sampling_spec_valid: bool,
}

/// The caller integrates raw samples using the declared sign conventions and
/// uncertainty propagation. This adapter never infers omitted flows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnergyBoundaryReceipt {
    pub boundary_manifest_digest: String,
    pub boundary_complete: bool,
    pub electrical_input_uj: u64,
    pub shaft_work_output_uj: u64,
    pub heat_out_uj: u64,
    pub delta_stored_energy_uj: i64,
    pub combined_uncertainty_uj: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionOnsetReceipt {
    pub command_id: String,
    pub stamp: EventStamp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointMetrologyEvidence {
    pub mission: MissionTrace,
    pub instruments: JointInstrumentReceipt,
    pub energy: EnergyBoundaryReceipt,
    pub expected_predecessor_reserve_digest: String,
    pub physical_state_reset: bool,
    pub motion_onset: Option<MotionOnsetReceipt>,
    pub command_to_motion_limit_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnergyClosureResult {
    pub boundary_complete: bool,
    pub residual_uj: i128,
    pub combined_uncertainty_uj: u64,
    pub disposition: EnergyClosureDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandResponseResult {
    pub minimum_latency_ns: Option<u64>,
    pub maximum_latency_ns: Option<u64>,
    pub reference_limit_ns: Option<u64>,
    pub disposition: CommandResponseDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointMetrologyReport {
    pub disposition: QualificationDisposition,
    pub causal_replay: ReplayReport,
    pub energy: EnergyClosureResult,
    pub command_response: CommandResponseResult,
    pub findings: Vec<ReplayFinding>,
    pub actuation_authorized: bool,
}

fn push(
    findings: &mut Vec<ReplayFinding>,
    severity: FindingSeverity,
    code: &'static str,
    record_id: impl Into<String>,
) {
    findings.push(ReplayFinding {
        severity,
        code,
        record_id: record_id.into(),
    });
}

fn required_digest(
    findings: &mut Vec<ReplayFinding>,
    value: &str,
    field: &'static str,
    record_id: &str,
) {
    if value.trim().is_empty() {
        push(
            findings,
            FindingSeverity::Unknown,
            "JOINT_INSTRUMENT_IDENTITY_MISSING",
            format!("{record_id}:{field}"),
        );
    }
}

/// Compute electrical minus shaft work, heat out and stored-energy change.
pub fn energy_closure(receipt: EnergyBoundaryReceipt) -> EnergyClosureResult {
    let residual = i128::from(receipt.electrical_input_uj)
        - i128::from(receipt.shaft_work_output_uj)
        - i128::from(receipt.heat_out_uj)
        - i128::from(receipt.delta_stored_energy_uj);
    let disposition = if !receipt.boundary_complete {
        EnergyClosureDisposition::Unknown
    } else if residual.abs() > i128::from(receipt.combined_uncertainty_uj) {
        EnergyClosureDisposition::OutsideUncertainty
    } else {
        EnergyClosureDisposition::Balanced
    };
    EnergyClosureResult {
        boundary_complete: receipt.boundary_complete,
        residual_uj: residual,
        combined_uncertainty_uj: receipt.combined_uncertainty_uj,
        disposition,
    }
}

/// Check causal lineage and instrument receipts for one physical joint trace.
pub fn qualify_joint_metrology(evidence: &JointMetrologyEvidence) -> JointMetrologyReport {
    let causal_replay = replay_mission(&evidence.mission);
    let mut findings = Vec::new();
    let mission_id = evidence.mission.scope.mission_id.as_str();
    let instruments = &evidence.instruments;

    if evidence.mission.applications.len() != 1
        || evidence.mission.physical_transitions.len() != 1
    {
        push(
            &mut findings,
            FindingSeverity::Unknown,
            "JOINT_ADAPTER_REQUIRES_ONE_APPLICATION_AND_TRANSITION",
            mission_id,
        );
    }

    for (field, value) in [
        ("actuator_id", instruments.actuator_id.as_str()),
        ("actuator_configuration_digest", instruments.actuator_configuration_digest.as_str()),
        ("controller_configuration_digest", instruments.controller_configuration_digest.as_str()),
        ("policy_digest", instruments.policy_digest.as_str()),
        ("model_digest", instruments.model_digest.as_str()),
        ("action_domain_digest", instruments.action_domain_digest.as_str()),
        ("sensor_calibration_digest", instruments.sensor_calibration_digest.as_str()),
        ("torque_transducer_calibration_digest", instruments.torque_transducer_calibration_digest.as_str()),
        ("power_analyzer_calibration_digest", instruments.power_analyzer_calibration_digest.as_str()),
        ("thermal_sensor_calibration_digest", instruments.thermal_sensor_calibration_digest.as_str()),
        ("sampling_spec_digest", instruments.sampling_spec_digest.as_str()),
        ("energy_boundary_manifest_digest", instruments.energy_boundary_manifest_digest.as_str()),
        ("decision_calibration_receipt_digest", instruments.decision_calibration_receipt_digest.as_str()),
        ("actuator_application_receipt_digest", instruments.actuator_application_receipt_digest.as_str()),
        ("executed_prefix_expected_digest", instruments.executed_prefix_expected_digest.as_str()),
        ("executed_prefix_observed_digest", instruments.executed_prefix_observed_digest.as_str()),
        ("thermal_snapshot_digest", instruments.thermal_snapshot_digest.as_str()),
        ("expected_predecessor_reserve_digest", evidence.expected_predecessor_reserve_digest.as_str()),
    ] {
        required_digest(&mut findings, value, field, mission_id);
    }

    if instruments.measured_sample_rate_hz == 0
        || instruments.required_sample_rate_hz == 0
        || !instruments.sampling_spec_valid
        || instruments.measured_sample_rate_hz < instruments.required_sample_rate_hz
    {
        push(
            &mut findings,
            FindingSeverity::Unknown,
            "JOINT_SAMPLING_SPEC_NOT_SATISFIED",
            mission_id,
        );
    }

    if evidence.physical_state_reset {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "JOINT_PHYSICAL_STATE_RESET",
            mission_id,
        );
    }

    if let Some(application) = evidence.mission.applications.first()
        && (application.actuator_id != instruments.actuator_id
            || application.application_receipt_digest
                != instruments.actuator_application_receipt_digest)
    {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "JOINT_ACTUATOR_RECEIPT_BINDING_MISMATCH",
            &application.command_id,
        );
    }

    if let Some(decision) = evidence.mission.decisions.first()
        && (decision.calibration_digest != instruments.decision_calibration_receipt_digest
            || decision.action_domain_digest != instruments.action_domain_digest
            || decision.loss_model_digest != instruments.model_digest)
    {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "JOINT_DECISION_CALIBRATION_BINDING_MISMATCH",
            &decision.decision_id,
        );
    }

    if let Some(chunk) = evidence.mission.chunks.first()
        && (chunk.model_digest != instruments.model_digest
            || chunk.action_domain_digest != instruments.action_domain_digest
            || chunk.expected_executed_prefix_digest
                != instruments.executed_prefix_expected_digest
            || chunk.observed_executed_prefix_digest
                != instruments.executed_prefix_observed_digest)
    {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "JOINT_PROPOSAL_CONFIGURATION_OR_PREFIX_MISMATCH",
            &chunk.proposal_id,
        );
    }

    if let Some(transition) = evidence.mission.physical_transitions.first() {
        if transition.predecessor_state_digest != evidence.expected_predecessor_reserve_digest
            || transition.predecessor_state_digest
                != evidence.mission.scope.initial_reserve_state_digest
        {
            push(
                &mut findings,
                FindingSeverity::Invalid,
                "JOINT_THERMAL_PREDECESSOR_MISMATCH",
                &transition.transition_id,
            );
        }
        if transition.thermal_snapshot_digest != instruments.thermal_snapshot_digest {
            push(
                &mut findings,
                FindingSeverity::Invalid,
                "JOINT_THERMAL_SNAPSHOT_BINDING_MISMATCH",
                &transition.transition_id,
            );
        }
    }

    if let Some(application) = evidence.mission.applications.first()
        && let Some(transition) = evidence.mission.physical_transitions.first()
        && transition.command_id != application.command_id
    {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "JOINT_PHYSICAL_TRANSITION_COMMAND_MISMATCH",
            &transition.transition_id,
        );
    }

    if evidence.energy.boundary_manifest_digest.trim().is_empty() {
        push(
            &mut findings,
            FindingSeverity::Unknown,
            "JOINT_ENERGY_BOUNDARY_IDENTITY_MISSING",
            mission_id,
        );
    } else if evidence.energy.boundary_manifest_digest
        != instruments.energy_boundary_manifest_digest
    {
        push(
            &mut findings,
            FindingSeverity::Invalid,
            "JOINT_ENERGY_BOUNDARY_MANIFEST_MISMATCH",
            mission_id,
        );
    }

    let energy = energy_closure(evidence.energy);
    match energy.disposition {
        EnergyClosureDisposition::Balanced => {}
        EnergyClosureDisposition::Unknown => push(
            &mut findings,
            FindingSeverity::Unknown,
            "JOINT_ENERGY_BOUNDARY_INCOMPLETE",
            mission_id,
        ),
        EnergyClosureDisposition::OutsideUncertainty => push(
            &mut findings,
            FindingSeverity::Unknown,
            "JOINT_ENERGY_RESIDUAL_OUTSIDE_UNCERTAINTY",
            mission_id,
        ),
    }

    let command_response = match (
        evidence.mission.applications.first(),
        evidence.motion_onset.as_ref(),
        evidence.command_to_motion_limit_ns,
    ) {
        (Some(application), Some(onset), Some(limit)) => {
            if onset.command_id != application.command_id
                || onset.stamp.sequence <= application.stamp.sequence
                || onset.stamp.tick < application.stamp.tick
            {
                push(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "JOINT_MOTION_ONSET_CAUSAL_BINDING_INVALID",
                    &application.command_id,
                );
                CommandResponseResult {
                    minimum_latency_ns: None,
                    maximum_latency_ns: None,
                    reference_limit_ns: Some(limit),
                    disposition: CommandResponseDisposition::ExceedsReference,
                }
            } else if onset.stamp.event_id.trim().is_empty()
                || onset.stamp.timestamp.lower_ns > onset.stamp.timestamp.upper_ns
            {
                push(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "JOINT_MOTION_ONSET_STAMP_INVALID",
                    &application.command_id,
                );
                CommandResponseResult {
                    minimum_latency_ns: None,
                    maximum_latency_ns: None,
                    reference_limit_ns: Some(limit),
                    disposition: CommandResponseDisposition::ExceedsReference,
                }
            } else if onset.stamp.robot_id != application.stamp.robot_id
                || onset.stamp.clock_id != application.stamp.clock_id
                || onset.stamp.boot_epoch != application.stamp.boot_epoch
            {
                push(
                    &mut findings,
                    FindingSeverity::Unknown,
                    "JOINT_MOTION_ONSET_CLOCK_OR_ROBOT_MISMATCH",
                    &application.command_id,
                );
                CommandResponseResult {
                    minimum_latency_ns: None,
                    maximum_latency_ns: None,
                    reference_limit_ns: Some(limit),
                    disposition: CommandResponseDisposition::Unknown,
                }
            } else if onset.stamp.timestamp.lower_ns < application.stamp.timestamp.upper_ns {
                push(
                    &mut findings,
                    FindingSeverity::Unknown,
                    "JOINT_MOTION_ONSET_TIME_ORDER_UNCERTAIN",
                    &application.command_id,
                );
                CommandResponseResult {
                    minimum_latency_ns: None,
                    maximum_latency_ns: None,
                    reference_limit_ns: Some(limit),
                    disposition: CommandResponseDisposition::Unknown,
                }
            } else {
                let minimum = onset.stamp.timestamp.lower_ns
                    - application.stamp.timestamp.upper_ns;
                let maximum = onset.stamp.timestamp.upper_ns
                    - application.stamp.timestamp.lower_ns;
                let disposition = if maximum <= limit {
                    CommandResponseDisposition::WithinReference
                } else if minimum > limit {
                    push(
                        &mut findings,
                        FindingSeverity::Invalid,
                        "JOINT_COMMAND_TO_MOTION_LIMIT_EXCEEDED",
                        &application.command_id,
                    );
                    CommandResponseDisposition::ExceedsReference
                } else {
                    push(
                        &mut findings,
                        FindingSeverity::Unknown,
                        "JOINT_COMMAND_TO_MOTION_LIMIT_OVERLAPS_UNCERTAINTY",
                        &application.command_id,
                    );
                    CommandResponseDisposition::Unknown
                };
                CommandResponseResult {
                    minimum_latency_ns: Some(minimum),
                    maximum_latency_ns: Some(maximum),
                    reference_limit_ns: Some(limit),
                    disposition,
                }
            }
        }
        _ => {
            push(
                &mut findings,
                FindingSeverity::Unknown,
                "JOINT_COMMAND_TO_MOTION_EVIDENCE_MISSING",
                mission_id,
            );
            CommandResponseResult {
                minimum_latency_ns: None,
                maximum_latency_ns: None,
                reference_limit_ns: evidence.command_to_motion_limit_ns,
                disposition: CommandResponseDisposition::Unknown,
            }
        }
    };

    let replay_invalid = causal_replay.disposition == ReplayDisposition::Rejected;
    let replay_unknown = causal_replay.disposition == ReplayDisposition::Unknown;
    let has_invalid = findings
        .iter()
        .any(|finding| finding.severity == FindingSeverity::Invalid);
    let has_unknown = findings
        .iter()
        .any(|finding| finding.severity == FindingSeverity::Unknown);
    let disposition = if replay_invalid || has_invalid {
        QualificationDisposition::Rejected
    } else if replay_unknown || has_unknown {
        QualificationDisposition::Unknown
    } else {
        QualificationDisposition::Qualified
    };

    JointMetrologyReport {
        disposition,
        causal_replay,
        energy,
        command_response,
        findings,
        actuation_authorized: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(
        id: &str,
        sequence: u64,
        tick: u64,
        time_ns: u64,
    ) -> super::super::EventStamp {
        super::super::EventStamp {
            event_id: id.to_owned(),
            sequence,
            tick,
            robot_id: "robot-01".to_owned(),
            clock_id: "joint-clock".to_owned(),
            boot_epoch: 7,
            timestamp: super::super::TimestampInterval {
                lower_ns: time_ns,
                upper_ns: time_ns,
            },
        }
    }

    fn valid_evidence() -> JointMetrologyEvidence {
        use super::super::{
            ActionChunkRecord, ActuatorApplication, AssuranceDisposition, AssuranceRecord,
            CalibrationDisposition, DecisionRecord, MissionScope, PhysicalTransition,
            SourceObservation, TerminalDisposition, TerminalRecord, TopologyDisposition,
        };

        let mission = MissionTrace {
            scope: MissionScope {
                mission_id: "joint-mission-01".to_owned(),
                robot_id: "robot-01".to_owned(),
                clock_id: "joint-clock".to_owned(),
                boot_epoch: 7,
                scenario_manifest_digest: "scenario-digest".to_owned(),
                initial_reserve_state_digest: "reserve-0".to_owned(),
                required_thermal_node_ids: vec![
                    "motor-winding".to_owned(),
                    "motor-housing".to_owned(),
                    "drive".to_owned(),
                ],
            },
            observations: vec![SourceObservation {
                observation_id: "observation-01".to_owned(),
                stamp: stamp("event-01", 1, 1, 100_000_000),
                modality: "joint-encoder".to_owned(),
                source_sequence: 1,
                sensor_digest: "sensor-calibration".to_owned(),
            }],
            decisions: vec![DecisionRecord {
                decision_id: "decision-01".to_owned(),
                stamp: stamp("event-02", 2, 1, 101_000_000),
                selected_action_id: "action-01".to_owned(),
                source_observation_ids: vec!["observation-01".to_owned()],
                loss_model_digest: "model-digest".to_owned(),
                loss_scale_id: "joint-task-loss-v1".to_owned(),
                action_domain_digest: "action-domain-digest".to_owned(),
                calibration_id: "calibration-01".to_owned(),
                calibration_digest: "decision-calibration-digest".to_owned(),
                calibration_disposition: CalibrationDisposition::Qualified,
                regret_upper_units: 1,
                epsilon_task_loss_units: 2,
                valid_through_tick: 10,
            }],
            remote_requests: Vec::new(),
            chunks: vec![ActionChunkRecord {
                proposal_id: "proposal-01".to_owned(),
                stamp: stamp("event-03", 3, 2, 105_000_000),
                decision_id: "decision-01".to_owned(),
                request_id: None,
                action_id: "action-01".to_owned(),
                model_digest: "model-digest".to_owned(),
                action_domain_digest: "action-domain-digest".to_owned(),
                expected_executed_prefix_digest: "prefix-expected".to_owned(),
                observed_executed_prefix_digest: "prefix-expected".to_owned(),
            }],
            assurance: vec![AssuranceRecord {
                proposal_id: "proposal-01".to_owned(),
                stamp: stamp("event-04", 4, 2, 106_000_000),
                disposition: AssuranceDisposition::Admitted,
            }],
            applications: vec![ActuatorApplication {
                command_id: "command-01".to_owned(),
                stamp: stamp("event-06", 6, 3, 110_000_000),
                proposal_id: "proposal-01".to_owned(),
                decision_id: "decision-01".to_owned(),
                action_id: "action-01".to_owned(),
                source_observation_ids: vec!["observation-01".to_owned()],
                request_id: None,
                actuator_id: "joint-actuator-01".to_owned(),
                application_receipt_digest: "application-receipt".to_owned(),
            }],
            physical_transitions: vec![PhysicalTransition {
                transition_id: "physical-01".to_owned(),
                stamp: stamp("event-08", 8, 3, 128_000_000),
                command_id: "command-01".to_owned(),
                predecessor_state_digest: "reserve-0".to_owned(),
                resulting_state_digest: "reserve-1".to_owned(),
                thermal_model_digest: "thermal-model".to_owned(),
                thermal_snapshot_digest: "thermal-snapshot".to_owned(),
                thermal_node_ids: vec![
                    "motor-winding".to_owned(),
                    "motor-housing".to_owned(),
                    "drive".to_owned(),
                ],
                topology: TopologyDisposition::Known,
            }],
            terminal: Some(TerminalRecord {
                stamp: stamp("event-09", 9, 3, 135_000_000),
                disposition: TerminalDisposition::SafeStopQualified,
                reserve_state_digest: "reserve-1".to_owned(),
            }),
        };

        let instruments = JointInstrumentReceipt {
            actuator_id: "joint-actuator-01".to_owned(),
            actuator_configuration_digest: "actuator-config".to_owned(),
            controller_configuration_digest: "controller-config".to_owned(),
            policy_digest: "policy-digest".to_owned(),
            model_digest: "model-digest".to_owned(),
            action_domain_digest: "action-domain-digest".to_owned(),
            sensor_calibration_digest: "sensor-calibration".to_owned(),
            torque_transducer_calibration_digest: "torque-calibration".to_owned(),
            power_analyzer_calibration_digest: "power-calibration".to_owned(),
            thermal_sensor_calibration_digest: "thermal-calibration".to_owned(),
            sampling_spec_digest: "sampling-spec".to_owned(),
            energy_boundary_manifest_digest: "energy-boundary".to_owned(),
            decision_calibration_receipt_digest: "decision-calibration-digest".to_owned(),
            actuator_application_receipt_digest: "application-receipt".to_owned(),
            executed_prefix_expected_digest: "prefix-expected".to_owned(),
            executed_prefix_observed_digest: "prefix-expected".to_owned(),
            thermal_snapshot_digest: "thermal-snapshot".to_owned(),
            measured_sample_rate_hz: 20_000,
            required_sample_rate_hz: 20_000,
            sampling_spec_valid: true,
        };

        JointMetrologyEvidence {
            mission,
            instruments,
            energy: EnergyBoundaryReceipt {
                boundary_manifest_digest: "energy-boundary".to_owned(),
                boundary_complete: true,
                electrical_input_uj: 100_000_000,
                shaft_work_output_uj: 70_000_000,
                heat_out_uj: 25_000_000,
                delta_stored_energy_uj: 5_000_000,
                combined_uncertainty_uj: 100_000,
            },
            expected_predecessor_reserve_digest: "reserve-0".to_owned(),
            physical_state_reset: false,
            motion_onset: Some(MotionOnsetReceipt {
                command_id: "command-01".to_owned(),
                stamp: stamp("event-07", 7, 3, 125_000_000),
            }),
            command_to_motion_limit_ns: Some(20_000_000),
        }
    }

    #[test]
    fn complete_joint_trace_closes_without_actuation_authority() {
        let report = qualify_joint_metrology(&valid_evidence());
        assert_eq!(report.disposition, QualificationDisposition::Qualified);
        assert_eq!(report.causal_replay.disposition, ReplayDisposition::Closed);
        assert_eq!(report.energy.disposition, EnergyClosureDisposition::Balanced);
        assert_eq!(
            report.command_response.disposition,
            CommandResponseDisposition::WithinReference
        );
        assert_eq!(report.causal_replay.aoi_bounds[0].minimum_age_ns, 10_000_000);
        assert!(!report.actuation_authorized);
        assert!(!report.causal_replay.actuation_authorized);
    }

    #[test]
    fn incomplete_energy_boundary_remains_unknown_even_if_arithmetic_balances() {
        let mut evidence = valid_evidence();
        evidence.energy.boundary_complete = false;
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
        assert_eq!(report.energy.residual_uj, 0);
        assert_eq!(report.energy.disposition, EnergyClosureDisposition::Unknown);
    }

    #[test]
    fn energy_residual_outside_declared_uncertainty_is_not_a_pass() {
        let mut evidence = valid_evidence();
        evidence.energy.shaft_work_output_uj += 1_000_000;
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
        assert_eq!(
            report.energy.disposition,
            EnergyClosureDisposition::OutsideUncertainty
        );
    }

    #[test]
    fn missing_thermal_node_prevents_joint_qualification() {
        let mut evidence = valid_evidence();
        evidence.mission.physical_transitions[0].thermal_node_ids.pop();
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
        assert_eq!(report.causal_replay.disposition, ReplayDisposition::Unknown);
    }

    #[test]
    fn physical_state_reset_is_rejected() {
        let mut evidence = valid_evidence();
        evidence.physical_state_reset = true;
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Rejected);
    }

    #[test]
    fn predecessor_state_mismatch_is_rejected() {
        let mut evidence = valid_evidence();
        evidence.mission.physical_transitions[0].predecessor_state_digest =
            "different-predecessor".to_owned();
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Rejected);
    }

    #[test]
    fn executed_prefix_mismatch_is_rejected() {
        let mut evidence = valid_evidence();
        evidence.instruments.executed_prefix_observed_digest = "observed-other".to_owned();
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Rejected);
    }

    #[test]
    fn regret_expiry_is_rejected_by_causal_replay() {
        let mut evidence = valid_evidence();
        evidence.mission.decisions[0].regret_upper_units = 3;
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Rejected);
        assert_eq!(report.causal_replay.disposition, ReplayDisposition::Rejected);
    }

    #[test]
    fn command_response_and_aoi_remain_independent_metrics() {
        let mut evidence = valid_evidence();
        evidence.motion_onset.as_mut().unwrap().stamp.timestamp = super::super::TimestampInterval {
            lower_ns: 155_000_000,
            upper_ns: 155_000_000,
        };
        evidence.mission.physical_transitions[0].stamp.timestamp =
            super::super::TimestampInterval {
                lower_ns: 160_000_000,
                upper_ns: 160_000_000,
            };
        evidence.mission.terminal.as_mut().unwrap().stamp.timestamp =
            super::super::TimestampInterval {
                lower_ns: 165_000_000,
                upper_ns: 165_000_000,
            };
        evidence.command_to_motion_limit_ns = Some(20_000_000);
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Rejected);
        assert_eq!(
            report.command_response.disposition,
            CommandResponseDisposition::ExceedsReference
        );
        assert_eq!(report.causal_replay.aoi_bounds[0].minimum_age_ns, 10_000_000);
    }

    #[test]
    fn undersampled_or_invalid_instrument_receipt_is_unknown() {
        let mut evidence = valid_evidence();
        evidence.instruments.measured_sample_rate_hz = 10_000;
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
    }

    #[test]
    fn decision_and_instrument_calibration_digests_must_match() {
        let mut evidence = valid_evidence();
        evidence.instruments.decision_calibration_receipt_digest =
            "different-calibration".to_owned();
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Rejected);
    }

    #[test]
    fn missing_command_response_evidence_is_unknown() {
        let mut evidence = valid_evidence();
        evidence.motion_onset = None;
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
        assert_eq!(
            report.command_response.disposition,
            CommandResponseDisposition::Unknown
        );
    }

    #[test]
    fn timestamp_uncertainty_overlapping_response_limit_remains_unknown() {
        let mut evidence = valid_evidence();
        evidence.motion_onset.as_mut().unwrap().stamp.timestamp =
            super::super::TimestampInterval {
                lower_ns: 129_000_000,
                upper_ns: 132_000_000,
            };
        evidence.mission.physical_transitions[0].stamp.timestamp =
            super::super::TimestampInterval {
                lower_ns: 140_000_000,
                upper_ns: 140_000_000,
            };
        evidence.mission.terminal.as_mut().unwrap().stamp.timestamp =
            super::super::TimestampInterval {
                lower_ns: 150_000_000,
                upper_ns: 150_000_000,
            };
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(
            report.command_response.disposition,
            CommandResponseDisposition::Unknown
        );
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
    }

    #[test]
    fn energy_manifest_must_match_the_measured_energy_receipt() {
        let mut evidence = valid_evidence();
        evidence.energy.boundary_manifest_digest = "different-boundary".to_owned();
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.disposition, QualificationDisposition::Rejected);
    }

    #[test]
    fn signed_storage_change_is_accounted_without_unsigned_underflow() {
        let mut evidence = valid_evidence();
        evidence.energy.delta_stored_energy_uj = -2_000_000;
        evidence.energy.heat_out_uj = 32_000_000;
        let report = qualify_joint_metrology(&evidence);
        assert_eq!(report.energy.residual_uj, 0);
        assert_eq!(report.energy.disposition, EnergyClosureDisposition::Balanced);
    }
}
