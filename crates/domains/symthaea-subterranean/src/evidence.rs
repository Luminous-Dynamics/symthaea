// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded, serializable operational evidence for subterranean embodiment.
//!
//! Safety claims should be reconstructable from the actual command and state,
//! not inferred from a final enum. The ledger intentionally keeps a bounded
//! ring so long missions do not create an unbounded memory sink.

use crate::types::{BATTERY_RATIO, NUM_STATE_CHANNELS, SubterraneanCommand};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SensorQualityEvidenceSnapshot {
    pub aggregate_precision: f64,
    pub minimum_reliability: f64,
    pub maximum_residual: f64,
    pub degraded_channels: usize,
    pub critical_degraded_channels: usize,
}

impl Default for SensorQualityEvidenceSnapshot {
    fn default() -> Self {
        Self {
            aggregate_precision: 1.0,
            minimum_reliability: 1.0,
            maximum_residual: 0.0,
            degraded_channels: 0,
            critical_degraded_channels: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeologyEvidenceSnapshot {
    pub material: String,
    pub lookahead_risk: f64,
    pub minimum_survey_confidence: f64,
    pub transition_count: usize,
    pub probe_required: bool,
}

impl Default for GeologyEvidenceSnapshot {
    fn default() -> Self {
        Self {
            material: "unknown".to_string(),
            lookahead_risk: 0.0,
            minimum_survey_confidence: 1.0,
            transition_count: 0,
            probe_required: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ReturnPathEvidenceSnapshot {
    pub distance_home_m: f64,
    pub path_confidence: f64,
    pub obstruction_risk: f64,
    pub estimated_battery_required: f64,
    pub battery_margin: f64,
    pub feasible: bool,
}

impl Default for ReturnPathEvidenceSnapshot {
    fn default() -> Self {
        Self {
            distance_home_m: 0.0,
            path_confidence: 1.0,
            obstruction_risk: 0.0,
            estimated_battery_required: 0.0,
            battery_margin: 1.0,
            feasible: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TeamEvidenceSnapshot {
    pub known_peers: usize,
    pub fresh_peers: usize,
    pub stale_peers: usize,
    pub distressed_peers: usize,
    pub directive: String,
    pub conflicting_agent: Option<u64>,
    pub conflict_severity: f32,
    pub must_yield: bool,
    pub surface_reachable: bool,
    pub mesh_bottleneck_quality: f64,
    pub mesh_hops: usize,
    pub shared_known_bins: usize,
    pub shared_contributing_peers: usize,
    pub shared_obstruction_risk: f64,
    pub rescue_state: String,
}

impl Default for TeamEvidenceSnapshot {
    fn default() -> Self {
        Self {
            known_peers: 0,
            fresh_peers: 0,
            stale_peers: 0,
            distressed_peers: 0,
            directive: "none".to_string(),
            conflicting_agent: None,
            conflict_severity: 0.0,
            must_yield: false,
            surface_reachable: false,
            mesh_bottleneck_quality: 0.0,
            mesh_hops: 0,
            shared_known_bins: 0,
            shared_contributing_peers: 0,
            shared_obstruction_risk: 0.0,
            rescue_state: "idle".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutiveEvidenceSnapshot {
    pub directive: String,
    pub active_work_order: Option<u64>,
    pub queued_work_orders: usize,
    pub completed_work_orders: usize,
    pub failed_work_orders: usize,
    pub work_admitted: bool,
    pub admission_refusal: Option<String>,
    pub outbound_distance_m: f64,
    pub return_distance_m: f64,
    pub route_maximum_risk: f64,
    pub route_minimum_confidence: f64,
    pub battery_required: f64,
    pub battery_after_return: f64,
    pub minimum_component_health: f64,
    pub critical_component: Option<String>,
    pub maintenance_due: bool,
    pub mission_abort_required: bool,
    pub sample_fill: f64,
    pub spoil_fill: f64,
    pub coolant_health: f64,
}

impl Default for ExecutiveEvidenceSnapshot {
    fn default() -> Self {
        Self {
            directive: "idle".to_string(),
            active_work_order: None,
            queued_work_orders: 0,
            completed_work_orders: 0,
            failed_work_orders: 0,
            work_admitted: false,
            admission_refusal: None,
            outbound_distance_m: 0.0,
            return_distance_m: 0.0,
            route_maximum_risk: 0.0,
            route_minimum_confidence: 1.0,
            battery_required: 0.0,
            battery_after_return: 1.0,
            minimum_component_health: 1.0,
            critical_component: None,
            maintenance_due: false,
            mission_abort_required: false,
            sample_fill: 0.0,
            spoil_fill: 0.0,
            coolant_health: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuthorityEvidenceSnapshot {
    pub operator_constraint: String,
    pub operator_accepted_commands: u64,
    pub operator_rejected_commands: u64,
    pub operator_last_proposal: Option<u64>,
    pub degraded_mode: String,
    pub degraded_transitions: u64,
    pub operator_link_loss_steps: u32,
    pub update_state: Option<String>,
    pub successful_update_activations: u64,
    pub update_rollbacks: u64,
}

impl Default for AuthorityEvidenceSnapshot {
    fn default() -> Self {
        Self {
            operator_constraint: "none".to_string(),
            operator_accepted_commands: 0,
            operator_rejected_commands: 0,
            operator_last_proposal: None,
            degraded_mode: "normal".to_string(),
            degraded_transitions: 0,
            operator_link_loss_steps: 0,
            update_state: None,
            successful_update_activations: 0,
            update_rollbacks: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivabilityEvidenceSnapshot {
    pub declared_sensor_sources: usize,
    pub accepted_sensor_sources: usize,
    pub critical_channels_without_quorum: usize,
    pub maximum_sensor_disagreement: f64,
    pub minimum_source_reliability: f64,
    pub isolated_actuators: usize,
    pub total_actuator_isolations: u64,
    pub mobility_degraded: bool,
    pub cooling_degraded: bool,
    pub recovery_degraded: bool,
    pub envelope_mode: String,
    pub power_margin: f64,
    pub thermal_margin: f64,
    pub capability_disposition: String,
    pub mission_work_allowed: bool,
    pub partition_mode: String,
    pub partition_steps: u32,
    pub reconciliation_steps: u32,
    pub map_revision_gap: u64,
    pub team_state_authoritative: bool,
}

impl Default for SurvivabilityEvidenceSnapshot {
    fn default() -> Self {
        Self {
            declared_sensor_sources: 1,
            accepted_sensor_sources: 1,
            critical_channels_without_quorum: 0,
            maximum_sensor_disagreement: 0.0,
            minimum_source_reliability: 1.0,
            isolated_actuators: 0,
            total_actuator_isolations: 0,
            mobility_degraded: false,
            cooling_degraded: false,
            recovery_degraded: false,
            envelope_mode: "nominal".to_string(),
            power_margin: 1.0,
            thermal_margin: 1.0,
            capability_disposition: "full_mission".to_string(),
            mission_work_allowed: true,
            partition_mode: "connected".to_string(),
            partition_steps: 0,
            reconciliation_steps: 0,
            map_revision_gap: 0,
            team_state_authoritative: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RecoveryResourceSnapshot {
    pub sealant_ratio: f64,
    pub relay_units: u8,
    pub roof_support_units: u8,
    pub dewatering_health: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CertificationEvidenceSnapshot {
    pub invariant_violations: Vec<String>,
    pub invariant_command_modified: bool,
    pub total_invariant_breaches: u64,
    pub consecutive_invariant_breach_frames: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafetyEvidenceRecord {
    pub step: u64,
    pub state_channels: [f64; NUM_STATE_CHANNELS],
    pub command: SubterraneanCommand,
    pub raw_hazard: String,
    pub latched_hazard: String,
    pub raw_hazard_severity: f32,
    pub latched_hazard_severity: f32,
    pub safety_level: String,
    pub requested_mission: String,
    pub effective_mission: String,
    pub fallback_stage: String,
    pub control_effort: f32,
    pub free_energy: f64,
    pub prediction_error: f32,
    pub observation_confidence: f32,
    #[serde(default)]
    pub recovery_resource_limited: bool,
    #[serde(default)]
    pub addressed_hazards: Vec<String>,
    #[serde(default)]
    pub return_path: ReturnPathEvidenceSnapshot,
    #[serde(default)]
    pub geology: GeologyEvidenceSnapshot,
    #[serde(default)]
    pub sensor_quality: SensorQualityEvidenceSnapshot,
    #[serde(default)]
    pub team: TeamEvidenceSnapshot,
    #[serde(default)]
    pub executive: ExecutiveEvidenceSnapshot,
    #[serde(default)]
    pub authority: AuthorityEvidenceSnapshot,
    #[serde(default)]
    pub survivability: SurvivabilityEvidenceSnapshot,
    #[serde(default)]
    pub certification: CertificationEvidenceSnapshot,
    pub recovery_resources: RecoveryResourceSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafetyEvidenceSummary {
    pub retained_records: usize,
    pub total_records: u64,
    pub dropped_records: u64,
    pub safety_intervention_records: u64,
    pub fallback_transitions: u64,
    pub max_hazard_severity: f32,
    pub max_prediction_error: f32,
    pub min_battery_ratio: f64,
    pub min_return_battery_margin: f64,
    pub infeasible_return_records: u64,
    pub geological_probe_records: u64,
    pub minimum_observation_precision: f64,
    pub critical_sensor_degradation_records: u64,
    pub tunnel_conflict_records: u64,
    pub mesh_partition_records: u64,
    pub peer_distress_records: u64,
    pub rescue_active_records: u64,
    pub minimum_mesh_bottleneck_quality: f64,
    pub maximum_shared_obstruction_risk: f64,
    pub active_work_records: u64,
    pub work_admission_refusal_records: u64,
    pub maintenance_due_records: u64,
    pub mission_abort_records: u64,
    pub maximum_completed_work_orders: usize,
    pub minimum_component_health: f64,
    pub operator_constrained_records: u64,
    pub degraded_operation_records: u64,
    pub recovery_required_records: u64,
    pub update_transition_records: u64,
    pub maximum_operator_rejections: u64,
    pub sensor_quorum_failure_records: u64,
    pub actuator_isolation_records: u64,
    pub field_derated_records: u64,
    pub survival_hold_records: u64,
    pub maximum_isolated_actuators: usize,
    pub minimum_sensor_source_reliability: f64,
    pub partition_operation_records: u64,
    pub reconciliation_records: u64,
    pub maximum_partition_steps: u32,
    pub invariant_breach_records: u64,
    pub maximum_consecutive_invariant_breach_frames: u32,
    pub hazard_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Clone)]
pub struct SafetyEvidenceLedger {
    capacity: usize,
    records: VecDeque<SafetyEvidenceRecord>,
    total_records: u64,
    dropped_records: u64,
    fallback_transitions: u64,
    last_fallback_stage: Option<String>,
}

impl SafetyEvidenceLedger {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            records: VecDeque::with_capacity(capacity.max(1)),
            total_records: 0,
            dropped_records: 0,
            fallback_transitions: 0,
            last_fallback_stage: None,
        }
    }

    pub fn push(&mut self, record: SafetyEvidenceRecord) {
        self.total_records = self.total_records.saturating_add(1);
        if self
            .last_fallback_stage
            .as_deref()
            .is_some_and(|previous| previous != record.fallback_stage.as_str())
        {
            self.fallback_transitions = self.fallback_transitions.saturating_add(1);
        }
        self.last_fallback_stage = Some(record.fallback_stage.clone());
        if self.records.len() == self.capacity {
            self.records.pop_front();
            self.dropped_records = self.dropped_records.saturating_add(1);
        }
        self.records.push_back(record);
    }

    pub fn records(&self) -> Vec<SafetyEvidenceRecord> {
        self.records.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn summary(&self) -> SafetyEvidenceSummary {
        let mut safety_intervention_records = 0u64;
        let mut max_hazard_severity = 0.0f32;
        let mut max_prediction_error = 0.0f32;
        let mut min_battery_ratio = 1.0f64;
        let mut min_return_battery_margin = 1.0f64;
        let mut infeasible_return_records = 0u64;
        let mut geological_probe_records = 0u64;
        let mut minimum_observation_precision = 1.0f64;
        let mut critical_sensor_degradation_records = 0u64;
        let mut tunnel_conflict_records = 0u64;
        let mut mesh_partition_records = 0u64;
        let mut peer_distress_records = 0u64;
        let mut rescue_active_records = 0u64;
        let mut minimum_mesh_bottleneck_quality = 1.0f64;
        let mut maximum_shared_obstruction_risk = 0.0f64;
        let mut active_work_records = 0u64;
        let mut work_admission_refusal_records = 0u64;
        let mut maintenance_due_records = 0u64;
        let mut mission_abort_records = 0u64;
        let mut maximum_completed_work_orders = 0usize;
        let mut minimum_component_health = 1.0f64;
        let mut operator_constrained_records = 0u64;
        let mut degraded_operation_records = 0u64;
        let mut recovery_required_records = 0u64;
        let mut update_transition_records = 0u64;
        let mut maximum_operator_rejections = 0u64;
        let mut sensor_quorum_failure_records = 0u64;
        let mut actuator_isolation_records = 0u64;
        let mut field_derated_records = 0u64;
        let mut survival_hold_records = 0u64;
        let mut maximum_isolated_actuators = 0usize;
        let mut minimum_sensor_source_reliability = 1.0f64;
        let mut partition_operation_records = 0u64;
        let mut reconciliation_records = 0u64;
        let mut maximum_partition_steps = 0u32;
        let mut invariant_breach_records = 0u64;
        let mut maximum_consecutive_invariant_breach_frames = 0u32;
        let mut previous_update_state: Option<&str> = None;
        let mut hazard_counts = BTreeMap::new();

        for record in &self.records {
            if record.latched_hazard != "none" {
                safety_intervention_records = safety_intervention_records.saturating_add(1);
            }
            max_hazard_severity = max_hazard_severity.max(record.latched_hazard_severity);
            max_prediction_error = max_prediction_error.max(record.prediction_error);
            min_battery_ratio = min_battery_ratio.min(record.state_channels[BATTERY_RATIO]);
            min_return_battery_margin =
                min_return_battery_margin.min(record.return_path.battery_margin);
            if !record.return_path.feasible {
                infeasible_return_records = infeasible_return_records.saturating_add(1);
            }
            if record.geology.probe_required {
                geological_probe_records = geological_probe_records.saturating_add(1);
            }
            minimum_observation_precision =
                minimum_observation_precision.min(record.sensor_quality.aggregate_precision);
            if record.sensor_quality.critical_degraded_channels > 0 {
                critical_sensor_degradation_records =
                    critical_sensor_degradation_records.saturating_add(1);
            }
            if record.team.conflicting_agent.is_some() {
                tunnel_conflict_records = tunnel_conflict_records.saturating_add(1);
            }
            if record.team.fresh_peers > 0 && !record.team.surface_reachable {
                mesh_partition_records = mesh_partition_records.saturating_add(1);
            }
            if record.team.distressed_peers > 0 {
                peer_distress_records = peer_distress_records.saturating_add(1);
            }
            if matches!(record.team.rescue_state.as_str(), "accepted" | "active") {
                rescue_active_records = rescue_active_records.saturating_add(1);
            }
            if record.team.surface_reachable {
                minimum_mesh_bottleneck_quality =
                    minimum_mesh_bottleneck_quality.min(record.team.mesh_bottleneck_quality);
            }
            maximum_shared_obstruction_risk =
                maximum_shared_obstruction_risk.max(record.team.shared_obstruction_risk);
            if record.executive.active_work_order.is_some() {
                active_work_records = active_work_records.saturating_add(1);
            }
            if record.executive.admission_refusal.is_some() {
                work_admission_refusal_records = work_admission_refusal_records.saturating_add(1);
            }
            if record.executive.maintenance_due {
                maintenance_due_records = maintenance_due_records.saturating_add(1);
            }
            if record.executive.mission_abort_required {
                mission_abort_records = mission_abort_records.saturating_add(1);
            }
            maximum_completed_work_orders =
                maximum_completed_work_orders.max(record.executive.completed_work_orders);
            minimum_component_health =
                minimum_component_health.min(record.executive.minimum_component_health);
            if record.authority.operator_constraint != "none" {
                operator_constrained_records = operator_constrained_records.saturating_add(1);
            }
            if record.authority.degraded_mode != "normal" {
                degraded_operation_records = degraded_operation_records.saturating_add(1);
            }
            if record.authority.degraded_mode == "recovery_required" {
                recovery_required_records = recovery_required_records.saturating_add(1);
            }
            let current_update_state = record.authority.update_state.as_deref();
            if current_update_state != previous_update_state && current_update_state.is_some() {
                update_transition_records = update_transition_records.saturating_add(1);
            }
            previous_update_state = current_update_state;
            maximum_operator_rejections =
                maximum_operator_rejections.max(record.authority.operator_rejected_commands);
            if record.survivability.critical_channels_without_quorum > 0 {
                sensor_quorum_failure_records = sensor_quorum_failure_records.saturating_add(1);
            }
            if record.survivability.isolated_actuators > 0 {
                actuator_isolation_records = actuator_isolation_records.saturating_add(1);
            }
            if record.survivability.envelope_mode != "nominal" {
                field_derated_records = field_derated_records.saturating_add(1);
            }
            if record.survivability.capability_disposition == "hold_for_recovery" {
                survival_hold_records = survival_hold_records.saturating_add(1);
            }
            maximum_isolated_actuators =
                maximum_isolated_actuators.max(record.survivability.isolated_actuators);
            minimum_sensor_source_reliability = minimum_sensor_source_reliability
                .min(record.survivability.minimum_source_reliability);
            if !matches!(record.survivability.partition_mode.as_str(), "connected") {
                partition_operation_records = partition_operation_records.saturating_add(1);
            }
            if record.survivability.partition_mode == "reconciling" {
                reconciliation_records = reconciliation_records.saturating_add(1);
            }
            maximum_partition_steps =
                maximum_partition_steps.max(record.survivability.partition_steps);
            if !record.certification.invariant_violations.is_empty() {
                invariant_breach_records = invariant_breach_records.saturating_add(1);
            }
            maximum_consecutive_invariant_breach_frames =
                maximum_consecutive_invariant_breach_frames
                    .max(record.certification.consecutive_invariant_breach_frames);
            *hazard_counts
                .entry(record.latched_hazard.clone())
                .or_insert(0) += 1;
        }

        SafetyEvidenceSummary {
            retained_records: self.records.len(),
            total_records: self.total_records,
            dropped_records: self.dropped_records,
            safety_intervention_records,
            fallback_transitions: self.fallback_transitions,
            max_hazard_severity,
            max_prediction_error,
            min_battery_ratio,
            min_return_battery_margin,
            infeasible_return_records,
            geological_probe_records,
            minimum_observation_precision,
            critical_sensor_degradation_records,
            tunnel_conflict_records,
            mesh_partition_records,
            peer_distress_records,
            rescue_active_records,
            minimum_mesh_bottleneck_quality,
            maximum_shared_obstruction_risk,
            active_work_records,
            work_admission_refusal_records,
            maintenance_due_records,
            mission_abort_records,
            maximum_completed_work_orders,
            minimum_component_health,
            operator_constrained_records,
            degraded_operation_records,
            recovery_required_records,
            update_transition_records,
            maximum_operator_rejections,
            sensor_quorum_failure_records,
            actuator_isolation_records,
            field_derated_records,
            survival_hold_records,
            maximum_isolated_actuators,
            minimum_sensor_source_reliability,
            partition_operation_records,
            reconciliation_records,
            maximum_partition_steps,
            invariant_breach_records,
            maximum_consecutive_invariant_breach_frames,
            hazard_counts,
        }
    }

    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.records())
    }

    pub fn clear(&mut self) {
        self.records.clear();
        self.total_records = 0;
        self.dropped_records = 0;
        self.fallback_transitions = 0;
        self.last_fallback_stage = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(step: u64, fallback: &str) -> SafetyEvidenceRecord {
        SafetyEvidenceRecord {
            step,
            state_channels: [0.0; NUM_STATE_CHANNELS],
            command: SubterraneanCommand::zero(),
            raw_hazard: "none".to_string(),
            latched_hazard: "none".to_string(),
            raw_hazard_severity: 0.0,
            latched_hazard_severity: 0.0,
            safety_level: "green".to_string(),
            requested_mission: "explore".to_string(),
            effective_mission: "explore".to_string(),
            fallback_stage: fallback.to_string(),
            control_effort: 0.0,
            free_energy: 0.0,
            prediction_error: 0.0,
            observation_confidence: 1.0,
            recovery_resource_limited: false,
            addressed_hazards: Vec::new(),
            return_path: ReturnPathEvidenceSnapshot::default(),
            geology: GeologyEvidenceSnapshot::default(),
            sensor_quality: SensorQualityEvidenceSnapshot::default(),
            team: TeamEvidenceSnapshot::default(),
            executive: ExecutiveEvidenceSnapshot::default(),
            authority: AuthorityEvidenceSnapshot::default(),
            survivability: SurvivabilityEvidenceSnapshot::default(),
            certification: CertificationEvidenceSnapshot::default(),
            recovery_resources: RecoveryResourceSnapshot {
                sealant_ratio: 1.0,
                relay_units: 3,
                roof_support_units: 3,
                dewatering_health: 1.0,
            },
        }
    }

    #[test]
    fn bounded_ledger_drops_oldest_records() {
        let mut ledger = SafetyEvidenceLedger::new(2);
        ledger.push(record(1, "nominal"));
        ledger.push(record(2, "nominal"));
        ledger.push(record(3, "thermal_arrest"));
        let records = ledger.records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 2);
        assert_eq!(ledger.summary().dropped_records, 1);
        assert_eq!(ledger.summary().fallback_transitions, 1);
    }

    #[test]
    fn evidence_export_is_valid_json() {
        let mut ledger = SafetyEvidenceLedger::new(2);
        ledger.push(record(1, "nominal"));
        let json = ledger.to_pretty_json().expect("record is serializable");
        assert!(json.contains("requested_mission"));
    }
}
