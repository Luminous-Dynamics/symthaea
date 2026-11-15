use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub schema_version: String,
    pub scenario_id: String,
    pub domain: String,
    pub difficulty: String,
    pub split: String,
    pub participant_role: String,
    pub task_brief: String,
    pub constraints: Vec<String>,
    pub collaboration_events: Vec<ScenarioEvent>,
    pub required_capabilities: Vec<String>,
    pub observable_checks: Vec<ObservableCheck>,
    pub hard_failures: Vec<String>,
    pub baseline_conditions: Vec<String>,
    pub consent_scope: ConsentScope,
    pub human_rubric: Vec<HumanRubric>,
    pub duration_minutes: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioEvent {
    pub event_id: String,
    pub event_kind: String,
    pub instruction: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_turn: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservableCheck {
    pub check_id: String,
    pub measure: String,
    pub pass_condition: String,
    pub scoring_mode: String,
    pub hard_gate: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConsentScope {
    pub trace_recording: String,
    pub persistent_memory: String,
    pub artifact_reuse: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HumanRubric {
    pub dimension: String,
    pub scale: String,
    pub anchors: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub schema_version: String,
    pub run_id: String,
    pub scenario_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_digest: Option<String>,
    pub events: Vec<TraceEvent>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceEvent {
    pub event_id: String,
    pub sequence: u64,
    pub elapsed_ms: u64,
    pub actor: String,
    pub event_type: String,
    pub parent_event_ids: Vec<String>,
    pub payload_digest: String,
    pub payload_storage: String,
    pub consent_scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_digest_before: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_digest_after: Option<String>,
    pub external_side_effect: bool,
    pub reversible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_execution: Option<ToolExecution>,
    #[serde(default)]
    pub attribution: Vec<Attribution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconciled_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconciliation_state_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_action: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolExecution {
    pub tool_name: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precondition_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub postcondition_digest: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Attribution {
    pub contributor: String,
    pub contribution_type: String,
    pub artifact_ref: String,
    pub attribution_confidence: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricEvidence {
    pub metric_name: String,
    pub value: f64,
    pub scorer_kind: String,
    pub sample_unit: String,
    pub sample_size: u64,
    pub measurement_method: String,
    pub scorer_revision: String,
    pub evidence_refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerator: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denominator: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci_lower: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci_upper: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scorecard {
    pub schema_version: String,
    pub benchmark_version: String,
    pub run_id: String,
    pub scenario_id: String,
    pub split: String,
    pub condition: String,
    pub system: SystemInfo,
    pub execution: ExecutionInfo,
    pub provenance: ProvenanceInfo,
    pub scores: Scores,
    pub metric_evidence: Vec<MetricEvidence>,
    pub hard_gates: HardGates,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SystemInfo {
    pub system_name: String,
    pub model_snapshot: String,
    pub adapter_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_digest: Option<String>,
    pub memory_policy: String,
    #[serde(default)]
    pub tools_enabled: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionInfo {
    pub seed: u64,
    pub status: String,
    pub latency_seconds: f64,
    pub human_time_minutes: f64,
    pub interaction_turns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceInfo {
    pub scenario_sha256: String,
    pub trace_sha256: String,
    pub artifact_sha256: String,
    pub scorer_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consented_trace_uri: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scores {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_quality: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_satisfaction: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub participant_agency_mean: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agency_delta_sd_vs_reference: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub common_ground_accuracy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_success_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution_precision: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution_recall: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution_f1: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_matched_diversity_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_rater_agreement: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synergy_delta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collaboration_gap: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partner_order_sensitivity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_recovery_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_appropriateness: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unnecessary_intervention_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missed_critical_intervention_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_regret: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub participant_capability_gain_delta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delayed_retention_gain_delta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_gain_delta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_confidence_brier_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_expected_calibration_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_metacognitive_sensitivity_auc: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_selective_accuracy_at_80pct_coverage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub participant_confidence_brier_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub participant_expected_calibration_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub participant_metacognitive_sensitivity_auc: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub participant_selective_accuracy_at_80pct_coverage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_planning_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_first_attempt_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_monitoring_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_reflection_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_core_step_supply_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_core_step_substitution_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_verification_rate: Option<f64>,
}

impl Scores {
    /// Return the complete, versioned metric-name/value table for evidence binding.
    pub fn entries(&self) -> Vec<(&'static str, Option<f64>)> {
        vec![
            ("task_quality", self.task_quality),
            ("constraint_satisfaction", self.constraint_satisfaction),
            ("participant_agency_mean", self.participant_agency_mean),
            ("agency_delta_sd_vs_reference", self.agency_delta_sd_vs_reference),
            ("common_ground_accuracy", self.common_ground_accuracy),
            ("repair_success_rate", self.repair_success_rate),
            ("attribution_precision", self.attribution_precision),
            ("attribution_recall", self.attribution_recall),
            ("attribution_f1", self.attribution_f1),
            ("quality_matched_diversity_ratio", self.quality_matched_diversity_ratio),
            ("expert_rater_agreement", self.expert_rater_agreement),
            ("synergy_delta", self.synergy_delta),
            ("collaboration_gap", self.collaboration_gap),
            ("partner_order_sensitivity", self.partner_order_sensitivity),
            ("handoff_recovery_rate", self.handoff_recovery_rate),
            ("intervention_appropriateness", self.intervention_appropriateness),
            ("unnecessary_intervention_rate", self.unnecessary_intervention_rate),
            ("missed_critical_intervention_rate", self.missed_critical_intervention_rate),
            ("intervention_regret", self.intervention_regret),
            ("participant_capability_gain_delta", self.participant_capability_gain_delta),
            ("delayed_retention_gain_delta", self.delayed_retention_gain_delta),
            ("transfer_gain_delta", self.transfer_gain_delta),
            ("agent_confidence_brier_score", self.agent_confidence_brier_score),
            ("agent_expected_calibration_error", self.agent_expected_calibration_error),
            ("agent_metacognitive_sensitivity_auc", self.agent_metacognitive_sensitivity_auc),
            (
                "agent_selective_accuracy_at_80pct_coverage",
                self.agent_selective_accuracy_at_80pct_coverage,
            ),
            ("participant_confidence_brier_score", self.participant_confidence_brier_score),
            (
                "participant_expected_calibration_error",
                self.participant_expected_calibration_error,
            ),
            (
                "participant_metacognitive_sensitivity_auc",
                self.participant_metacognitive_sensitivity_auc,
            ),
            (
                "participant_selective_accuracy_at_80pct_coverage",
                self.participant_selective_accuracy_at_80pct_coverage,
            ),
            ("human_planning_rate", self.human_planning_rate),
            ("human_first_attempt_rate", self.human_first_attempt_rate),
            ("human_monitoring_rate", self.human_monitoring_rate),
            ("human_reflection_rate", self.human_reflection_rate),
            ("ai_core_step_supply_rate", self.ai_core_step_supply_rate),
            ("ai_core_step_substitution_rate", self.ai_core_step_substitution_rate),
            ("human_verification_rate", self.human_verification_rate),
        ]
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardGates {
    pub pass: bool,
    pub authority_violation_count: u64,
    pub consent_violation_count: u64,
    pub fabricated_attribution_count: u64,
    pub hidden_side_effect_count: u64,
    pub unresolved_unknown_outcome_count: u64,
    pub reconciliation_bypass_count: u64,
    #[serde(default)]
    pub failure_event_ids: Vec<String>,
}
