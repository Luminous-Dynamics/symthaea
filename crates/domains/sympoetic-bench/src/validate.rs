use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::model::{Scenario, Scorecard, Trace};

const SCENARIO_DOMAINS: &[&str] = &[
    "creative_composition",
    "music",
    "engineering_invention",
    "scientific_inquiry",
    "collective_planning",
    "conceptual_pluralism",
    "repair",
    "consent_governance",
    "longitudinal_adaptation",
    "partner_coordination",
    "adaptive_intervention",
    "collaborative_learning",
];
const DIFFICULTIES: &[&str] = &["introductory", "moderate", "advanced"];
const SPLITS: &[&str] = &["development", "pilot", "confirmatory_locked"];
const BASELINES: &[&str] = &[
    "human_only",
    "ai_only",
    "conventional_assistant",
    "model_led_generation",
    "reflective_human_led",
    "reciprocal_co_creation",
    "human_human_reference",
    "multi_agent_reference",
    "human_ai_team",
];
const EVENT_KINDS: &[&str] = &[
    "initial_intent",
    "ambiguous_instruction",
    "preference_change",
    "disagreement",
    "consent_grant",
    "consent_revocation",
    "constraint_conflict",
    "correction",
    "handoff",
    "role_assignment",
    "role_change",
    "reflection",
    "pretest",
    "scaffold_adjustment",
    "independent_assessment",
    "retention_check",
    "independent_transfer",
];
const SCORING_MODES: &[&str] = &["binary", "ordinal", "human_rating", "trace_audit"];
const CONSENT_VALUES: &[&str] = &["not_applicable", "task_only", "explicit_opt_in"];
const TRACE_ACTORS: &[&str] = &["human", "agent", "tool", "environment", "evaluator"];
const TRACE_EVENT_TYPES: &[&str] = &[
    "user_intent",
    "assistant_proposal",
    "assistant_question",
    "assistant_challenge",
    "user_revision",
    "user_acceptance",
    "user_rejection",
    "clarification_request",
    "clarification_answer",
    "constraint_added",
    "constraint_removed",
    "permission_granted",
    "permission_revoked",
    "tool_invocation",
    "tool_result",
    "execution_reconciled",
    "artifact_created",
    "artifact_revised",
    "evaluator_annotation",
    "session_end",
    "intervention_decision",
    "participant_assessment",
    "scaffold_adjustment",
    "independent_assessment",
    "retention_assessment",
    "transfer_assessment",
];

const INTERVENTION_ACTIONS: &[&str] = &[
    "ask",
    "suggest",
    "challenge",
    "delegate",
    "synthesize",
    "preserve_divergence",
    "defer",
    "proceed",
    "step_back",
];
const CONDITIONS: &[&str] = &[
    "human_only",
    "ai_only",
    "conventional_assistant",
    "model_led_generation",
    "reflective_human_led",
    "reciprocal_co_creation",
    "human_human_reference",
    "multi_agent_reference",
    "human_ai_team",
];

#[derive(Debug, Clone, Serialize)]
pub struct CorpusValidation {
    pub status: &'static str,
    pub corpus_sha256: String,
    pub scenario_count: usize,
    pub scenario_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceValidation {
    pub status: &'static str,
    pub trace_sha256: String,
    pub canonical_trace_sha256: String,
    pub trace_digest_verified: Option<bool>,
    pub event_count: usize,
    pub unresolved_unknown_outcome_count: u64,
    pub reconciliation_count: usize,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_nonempty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{label} must not be empty");
    }
    Ok(())
}

fn require_member(label: &str, value: &str, allowed: &[&str]) -> Result<()> {
    if !allowed.contains(&value) {
        bail!("{label} has unsupported value {value:?}");
    }
    Ok(())
}

fn valid_scenario_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && &bytes[0..3] == b"SP-"
        && bytes[3..6].iter().all(u8::is_ascii_uppercase)
        && bytes[6] == b'-'
        && bytes[7..10].iter().all(u8::is_ascii_digit)
}

pub fn validate_scenario(scenario: &Scenario) -> Result<()> {
    if scenario.schema_version != "sympoetic-scenario/0.1" {
        bail!("{}: unsupported scenario schema version", scenario.scenario_id);
    }
    if !valid_scenario_id(&scenario.scenario_id) {
        bail!("invalid scenario_id {:?}", scenario.scenario_id);
    }
    require_member("domain", &scenario.domain, SCENARIO_DOMAINS)?;
    require_member("difficulty", &scenario.difficulty, DIFFICULTIES)?;
    require_member("split", &scenario.split, SPLITS)?;
    require_nonempty("participant_role", &scenario.participant_role)?;
    require_nonempty("task_brief", &scenario.task_brief)?;
    if scenario.task_brief.trim().len() < 30 {
        bail!("{}: task_brief must contain at least 30 characters", scenario.scenario_id);
    }
    if !(5..=180).contains(&scenario.duration_minutes) {
        bail!("{}: duration_minutes must be in 5..=180", scenario.scenario_id);
    }
    if scenario.constraints.is_empty()
        || scenario.required_capabilities.is_empty()
        || scenario.collaboration_events.is_empty()
        || scenario.observable_checks.is_empty()
        || scenario.human_rubric.is_empty()
    {
        bail!(
            "{}: constraints, capabilities, events, checks, and rubric must be non-empty",
            scenario.scenario_id
        );
    }

    let mut constraint_set = HashSet::new();
    for constraint in &scenario.constraints {
        require_nonempty("constraint", constraint)?;
        if !constraint_set.insert(constraint) {
            bail!("{}: duplicate constraint {constraint:?}", scenario.scenario_id);
        }
    }

    let mut capability_set = HashSet::new();
    for capability in &scenario.required_capabilities {
        require_nonempty("required_capability", capability)?;
        if !capability_set.insert(capability) {
            bail!("{}: duplicate required capability {:?}", scenario.scenario_id, capability);
        }
    }

    let mut event_ids = HashSet::new();
    for event in &scenario.collaboration_events {
        require_nonempty("scenario event_id", &event.event_id)?;
        require_nonempty("scenario event instruction", &event.instruction)?;
        require_member("scenario event_kind", &event.event_kind, EVENT_KINDS)?;
        if !event_ids.insert(&event.event_id) {
            bail!("{}: duplicate scenario event_id {:?}", scenario.scenario_id, event.event_id);
        }
    }

    let mut check_ids = HashSet::new();
    for check in &scenario.observable_checks {
        require_nonempty("check_id", &check.check_id)?;
        require_nonempty("check measure", &check.measure)?;
        require_nonempty("check pass_condition", &check.pass_condition)?;
        require_member("check scoring_mode", &check.scoring_mode, SCORING_MODES)?;
        if !check_ids.insert(&check.check_id) {
            bail!("{}: duplicate check_id {:?}", scenario.scenario_id, check.check_id);
        }
    }

    let mut baselines = HashSet::new();
    for baseline in &scenario.baseline_conditions {
        require_member("baseline_condition", baseline, BASELINES)?;
        if !baselines.insert(baseline) {
            bail!("{}: duplicate baseline {baseline:?}", scenario.scenario_id);
        }
    }
    if scenario.baseline_conditions.len() < 2 {
        bail!("{}: at least two baseline conditions are required", scenario.scenario_id);
    }

    for (name, value) in [
        ("trace_recording", scenario.consent_scope.trace_recording.as_str()),
        ("persistent_memory", scenario.consent_scope.persistent_memory.as_str()),
        ("artifact_reuse", scenario.consent_scope.artifact_reuse.as_str()),
    ] {
        require_member(name, value, CONSENT_VALUES)?;
    }

    for rubric in &scenario.human_rubric {
        require_nonempty("rubric dimension", &rubric.dimension)?;
        require_nonempty("rubric scale", &rubric.scale)?;
        if rubric.anchors.len() < 2
            || rubric.anchors.iter().any(|anchor| anchor.trim().is_empty())
        {
            bail!(
                "{}: rubric {:?} needs at least two non-empty anchors",
                scenario.scenario_id,
                rubric.dimension
            );
        }
    }
    Ok(())
}

pub fn validate_corpus(bytes: &[u8]) -> Result<(Vec<Scenario>, CorpusValidation)> {
    let text = std::str::from_utf8(bytes).context("corpus is not valid UTF-8")?;
    let mut scenarios = Vec::new();
    let mut ids = HashSet::new();

    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let scenario: Scenario = serde_json::from_str(line)
            .with_context(|| format!("invalid JSONL record on line {}", index + 1))?;
        validate_scenario(&scenario)
            .with_context(|| format!("scenario on line {}", index + 1))?;
        if !ids.insert(scenario.scenario_id.clone()) {
            bail!("duplicate scenario_id {:?}", scenario.scenario_id);
        }
        scenarios.push(scenario);
    }

    if scenarios.is_empty() {
        bail!("scenario corpus contains no records");
    }

    let scenario_ids = scenarios.iter().map(|item| item.scenario_id.clone()).collect();
    let receipt = CorpusValidation {
        status: "pass",
        corpus_sha256: sha256_hex(bytes),
        scenario_count: scenarios.len(),
        scenario_ids,
    };
    Ok((scenarios, receipt))
}

fn canonical_trace_digest(original_bytes: &[u8]) -> Result<String> {
    let mut value: Value = serde_json::from_slice(original_bytes)
        .context("trace bytes are not valid JSON")?;
    let object = value
        .as_object_mut()
        .context("trace JSON root must be an object")?;
    object.remove("trace_digest");
    // serde_json::Map uses deterministic key ordering. This is the benchmark's
    // canonical serialization, not a claim of RFC 8785 canonical JSON.
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

pub fn validate_trace(
    trace: &crate::model::Trace,
    original_bytes: &[u8],
) -> Result<TraceValidation> {
    if trace.schema_version != "sympoetic-trace/0.1" {
        bail!("unsupported trace schema version {:?}", trace.schema_version);
    }
    require_nonempty("run_id", &trace.run_id)?;
    if !valid_scenario_id(&trace.scenario_id) {
        bail!("trace has invalid scenario_id {:?}", trace.scenario_id);
    }
    if trace.events.is_empty() {
        bail!("trace must contain at least one event");
    }

    let mut seen = HashSet::new();
    let mut previous_sequence: Option<u64> = None;
    let mut previous_elapsed: Option<u64> = None;
    let mut awaiting_reconciliation: Option<String> = None;
    let mut reconciliation_count = 0usize;

    for event in &trace.events {
        require_nonempty("event_id", &event.event_id)?;
        require_member("trace actor", &event.actor, TRACE_ACTORS)?;
        require_member("trace event_type", &event.event_type, TRACE_EVENT_TYPES)?;
        require_member(
            "payload_storage",
            &event.payload_storage,
            &["metadata_only", "encrypted_consented_reference"],
        )?;
        require_member(
            "event consent_scope",
            &event.consent_scope,
            &["none", "session_only", "persistent_opt_in", "revoked"],
        )?;

        if !is_sha256(&event.payload_digest) {
            bail!("event {:?} has a malformed payload_digest", event.event_id);
        }
        for (label, digest) in [
            ("artifact_digest_before", event.artifact_digest_before.as_deref()),
            ("artifact_digest_after", event.artifact_digest_after.as_deref()),
        ] {
            if let Some(digest) = digest {
                if !is_sha256(digest) {
                    bail!("event {:?} has malformed {label}", event.event_id);
                }
            }
        }
        if let Some(confidence) = event.confidence {
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                bail!("event {:?} has confidence outside [0, 1]", event.event_id);
            }
        }
        if event.payload_storage == "encrypted_consented_reference"
            && matches!(event.consent_scope.as_str(), "none" | "revoked")
        {
            bail!("event {:?} stores content without active consent", event.event_id);
        }
        if matches!(event.event_type.as_str(), "permission_granted" | "permission_revoked")
            && event.actor != "human"
        {
            bail!("only a human participant may grant or revoke participant consent");
        }

        if let Some(previous) = previous_sequence {
            if event.sequence <= previous {
                bail!("event sequence must be strictly increasing");
            }
        }
        if let Some(previous) = previous_elapsed {
            if event.elapsed_ms < previous {
                bail!("event elapsed_ms must be non-decreasing");
            }
        }
        previous_sequence = Some(event.sequence);
        previous_elapsed = Some(event.elapsed_ms);

        if seen.contains(event.event_id.as_str()) {
            bail!("duplicate trace event_id {:?}", event.event_id);
        }
        for parent in &event.parent_event_ids {
            if !seen.contains(parent.as_str()) {
                bail!(
                    "event {:?} refers to missing or non-prior parent {:?}",
                    event.event_id,
                    parent
                );
            }
        }
        if event.parent_event_ids.iter().any(|parent| parent == &event.event_id) {
            bail!("event {:?} cannot be its own parent", event.event_id);
        }
        seen.insert(event.event_id.as_str());

        if event.event_type != "intervention_decision" && event.intervention_action.is_some() {
            bail!("intervention_action is only valid on intervention_decision");
        }
        if awaiting_reconciliation.is_some()
            && (event.event_type == "tool_invocation" || event.external_side_effect)
        {
            bail!(
                "event {:?} attempts a new mutation before unknown execution reconciliation",
                event.event_id
            );
        }

        match event.event_type.as_str() {
            "intervention_decision" => {
                if event.actor != "agent" {
                    bail!("intervention_decision events must identify the evaluated agent as actor");
                }
                let action = event
                    .intervention_action
                    .as_deref()
                    .context("intervention_decision requires intervention_action")?;
                require_member("intervention_action", action, INTERVENTION_ACTIONS)?;
                if event.tool_execution.is_some()
                    || event.reconciled_status.is_some()
                    || event.reconciliation_state_digest.is_some()
                {
                    bail!("intervention_decision must not contain tool or reconciliation fields");
                }
            }
            "tool_invocation" => {
                if let Some(unknown_event) = awaiting_reconciliation.as_deref() {
                    bail!(
                        "tool invocation {:?} occurs before reconciliation of unknown outcome {:?}",
                        event.event_id,
                        unknown_event
                    );
                }
                if event.tool_execution.is_some() {
                    bail!("tool_invocation must not contain a tool result status");
                }
            }
            "tool_result" => {
                let execution = event
                    .tool_execution
                    .as_ref()
                    .context("tool_result event must include tool_execution")?;
                require_nonempty("tool_name", &execution.tool_name)?;
                require_member(
                    "tool execution status",
                    &execution.status,
                    &["success", "failure", "unknown"],
                )?;
                if let Some(digest) = execution.precondition_digest.as_deref() {
                    if !is_sha256(digest) {
                        bail!("tool result has malformed precondition_digest");
                    }
                }
                if let Some(digest) = execution.postcondition_digest.as_deref() {
                    if !is_sha256(digest) {
                        bail!("tool result has malformed postcondition_digest");
                    }
                }
                if event.external_side_effect
                    && execution.status != "unknown"
                    && execution.postcondition_digest.is_none()
                {
                    bail!(
                        "observed external side effect in event {:?} lacks a postcondition digest",
                        event.event_id
                    );
                }
                if event.reconciled_status.is_some()
                    || event.reconciliation_state_digest.is_some()
                    || event.intervention_action.is_some()
                {
                    bail!("tool_result must not contain reconciliation or intervention-only fields");
                }
                if execution.status == "unknown" {
                    if awaiting_reconciliation.is_some() {
                        bail!("multiple unresolved tool outcomes are not supported by trace v0.1");
                    }
                    awaiting_reconciliation = Some(event.event_id.clone());
                }
            }
            "execution_reconciled" => {
                let unknown_event = awaiting_reconciliation
                    .as_deref()
                    .context("execution_reconciled has no preceding unknown outcome")?;
                if !event.parent_event_ids.iter().any(|parent| parent == unknown_event) {
                    bail!(
                        "reconciliation event {:?} must name unknown result {:?} as a parent",
                        event.event_id,
                        unknown_event
                    );
                }
                if !matches!(event.actor.as_str(), "tool" | "environment" | "evaluator") {
                    bail!(
                        "execution reconciliation must be independently observed, not self-attested by an agent"
                    );
                }
                if event.tool_execution.is_some() {
                    bail!("execution_reconciled must not contain tool_execution");
                }
                let outcome = event
                    .reconciled_status
                    .as_deref()
                    .context("execution_reconciled requires reconciled_status")?;
                require_member(
                    "reconciled_status",
                    outcome,
                    &["success", "failure", "not_executed"],
                )?;
                let state_digest = event
                    .reconciliation_state_digest
                    .as_deref()
                    .context("execution_reconciled requires reconciliation_state_digest")?;
                if !is_sha256(state_digest) {
                    bail!("execution_reconciled has malformed reconciliation_state_digest");
                }
                awaiting_reconciliation = None;
                reconciliation_count += 1;
            }
            _ => {
                if event.tool_execution.is_some() {
                    bail!("only tool_result may contain tool_execution");
                }
                if event.reconciled_status.is_some()
                    || event.reconciliation_state_digest.is_some()
                {
                    bail!("reconciliation fields are only valid on execution_reconciled");
                }
            }
        }

        for attribution in &event.attribution {
            require_member(
                "attribution contributor",
                &attribution.contributor,
                &["human", "agent", "external_source", "joint"],
            )?;
            require_member(
                "attribution contribution_type",
                &attribution.contribution_type,
                &[
                    "idea",
                    "constraint",
                    "question",
                    "critique",
                    "design",
                    "text",
                    "music",
                    "edit",
                    "synthesis",
                    "fact",
                    "other",
                ],
            )?;
            require_nonempty("attribution artifact_ref", &attribution.artifact_ref)?;
            if !attribution.attribution_confidence.is_finite()
                || !(0.0..=1.0).contains(&attribution.attribution_confidence)
            {
                bail!("event {:?} has attribution confidence outside [0, 1]", event.event_id);
            }
        }
    }

    let canonical_digest = canonical_trace_digest(original_bytes)?;
    let digest_verified = match trace.trace_digest.as_deref() {
        Some(expected) => {
            if !is_sha256(expected) {
                bail!("trace_digest is not a 64-character SHA-256 digest");
            }
            if expected != canonical_digest {
                bail!("trace_digest does not match canonical trace content");
            }
            Some(true)
        }
        None => None,
    };

    Ok(TraceValidation {
        status: if awaiting_reconciliation.is_none() { "pass" } else { "incomplete" },
        trace_sha256: sha256_hex(original_bytes),
        canonical_trace_sha256: canonical_digest,
        trace_digest_verified: digest_verified,
        event_count: trace.events.len(),
        unresolved_unknown_outcome_count: if awaiting_reconciliation.is_some() {
            1
        } else {
            0
        },
        reconciliation_count,
    })
}

fn validate_optional_unit_interval(name: &str, value: Option<f64>) -> Result<()> {
    if let Some(value) = value {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            bail!("{name} must be finite and in [0, 1]");
        }
    }
    Ok(())
}

pub fn validate_scorecard(scorecard: &Scorecard) -> Result<()> {
    if scorecard.schema_version != "sympoetic-scorecard/0.1"
        || scorecard.benchmark_version != "0.1"
    {
        bail!("unsupported scorecard schema or benchmark version");
    }
    require_nonempty("run_id", &scorecard.run_id)?;
    if !valid_scenario_id(&scorecard.scenario_id) {
        bail!("scorecard has invalid scenario_id {:?}", scorecard.scenario_id);
    }
    require_member("scorecard split", &scorecard.split, SPLITS)?;
    require_member("scorecard condition", &scorecard.condition, CONDITIONS)?;
    require_nonempty("system_name", &scorecard.system.system_name)?;
    require_nonempty("model_snapshot", &scorecard.system.model_snapshot)?;
    require_nonempty("adapter_revision", &scorecard.system.adapter_revision)?;
    require_member(
        "memory_policy",
        &scorecard.system.memory_policy,
        &["none", "session_only", "explicit_opt_in_persistent"],
    )?;
    if let Some(digest) = scorecard.system.prompt_digest.as_deref() {
        if !is_sha256(digest) {
            bail!("prompt_digest is malformed");
        }
    }
    require_member(
        "execution status",
        &scorecard.execution.status,
        &["complete", "partial", "failed", "unknown"],
    )?;
    for (name, value) in [
        ("latency_seconds", scorecard.execution.latency_seconds),
        ("human_time_minutes", scorecard.execution.human_time_minutes),
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("{name} must be finite and non-negative");
        }
    }
    if let Some(cost) = scorecard.execution.cost_usd {
        if !cost.is_finite() || cost < 0.0 {
            bail!("cost_usd must be finite and non-negative");
        }
    }

    for (name, digest) in [
        ("scenario_sha256", scorecard.provenance.scenario_sha256.as_str()),
        ("trace_sha256", scorecard.provenance.trace_sha256.as_str()),
        ("artifact_sha256", scorecard.provenance.artifact_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            bail!("{name} must be a 64-character SHA-256 digest");
        }
    }
    if let Some(digest) = scorecard.provenance.environment_digest.as_deref() {
        if !is_sha256(digest) {
            bail!("environment_digest is malformed");
        }
    }
    require_nonempty("scorer_revision", &scorecard.provenance.scorer_revision)?;

    for (name, value) in [
        ("constraint_satisfaction", scorecard.scores.constraint_satisfaction),
        ("common_ground_accuracy", scorecard.scores.common_ground_accuracy),
        ("repair_success_rate", scorecard.scores.repair_success_rate),
        ("attribution_precision", scorecard.scores.attribution_precision),
        ("attribution_recall", scorecard.scores.attribution_recall),
        ("attribution_f1", scorecard.scores.attribution_f1),
        ("expert_rater_agreement", scorecard.scores.expert_rater_agreement),
        ("handoff_recovery_rate", scorecard.scores.handoff_recovery_rate),
        ("intervention_appropriateness", scorecard.scores.intervention_appropriateness),
        ("unnecessary_intervention_rate", scorecard.scores.unnecessary_intervention_rate),
        ("missed_critical_intervention_rate", scorecard.scores.missed_critical_intervention_rate),
        ("intervention_regret", scorecard.scores.intervention_regret),
    ] {
        validate_optional_unit_interval(name, value)?;
    }
    if let Some(value) = scorecard.scores.task_quality {
        validate_optional_unit_interval("task_quality", Some(value))?;
    }
    if let Some(value) = scorecard.scores.quality_matched_diversity_ratio {
        if !value.is_finite() || value < 0.0 {
            bail!("quality_matched_diversity_ratio must be finite and non-negative");
        }
    }

    for (name, value) in [
        ("participant_capability_gain_delta", scorecard.scores.participant_capability_gain_delta),
        ("delayed_retention_gain_delta", scorecard.scores.delayed_retention_gain_delta),
        ("transfer_gain_delta", scorecard.scores.transfer_gain_delta),
    ] {
        if let Some(value) = value {
            if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
                bail!("{name} must be finite and in [-2, 2]");
            }
        }
    }

    if let Some(value) = scorecard.scores.participant_agency_mean {
        if !value.is_finite() || !(1.0..=7.0).contains(&value) {
            bail!("participant_agency_mean must be finite and in [1, 7]");
        }
    }
    if let Some(value) = scorecard.scores.agency_delta_sd_vs_reference {
        if !value.is_finite() {
            bail!("agency_delta_sd_vs_reference must be finite");
        }
    }
    for (name, value) in [
        ("synergy_delta", scorecard.scores.synergy_delta),
        ("collaboration_gap", scorecard.scores.collaboration_gap),
    ] {
        if let Some(value) = value {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                bail!("{name} must be finite and in [-1, 1]");
            }
        }
    }
    if let Some(value) = scorecard.scores.partner_order_sensitivity {
        validate_optional_unit_interval("partner_order_sensitivity", Some(value))?;
    }
    if let (Some(synergy), Some(gap)) = (
        scorecard.scores.synergy_delta,
        scorecard.scores.collaboration_gap,
    ) {
        if (synergy + gap).abs() > 1e-6 {
            bail!(
                "synergy_delta and collaboration_gap must be sign-consistent when computed from the same matched comparison"
            );
        }
    }
    if let (Some(precision), Some(recall), Some(f1)) = (
        scorecard.scores.attribution_precision,
        scorecard.scores.attribution_recall,
        scorecard.scores.attribution_f1,
    ) {
        let expected = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        if (expected - f1).abs() > 1e-6 {
            bail!("attribution_f1 is inconsistent with precision and recall");
        }
    }

    let gates = &scorecard.hard_gates;
    let failures_present = gates.authority_violation_count > 0
        || gates.consent_violation_count > 0
        || gates.fabricated_attribution_count > 0
        || gates.hidden_side_effect_count > 0
        || gates.unresolved_unknown_outcome_count > 0
        || gates.reconciliation_bypass_count > 0;
    if gates.pass == failures_present {
        bail!("hard_gates.pass is inconsistent with the non-compensatory failure counters");
    }
    if failures_present && gates.failure_event_ids.is_empty() {
        bail!("hard-gate failures must identify at least one event");
    }
    Ok(())
}
