use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: &str = "mel-musical-agency-panel-v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgencyPanelV1 {
    schema_version: String,
    session_id: String,
    collaborator_id: String,
    clock_domain_id: String,
    hard_constraints: Vec<HardConstraintV1>,
    soft_preferences: Vec<SoftPreferenceV1>,
    proposals: Vec<MusicalProposalV1>,
    interventions: Vec<CollaboratorInterventionV1>,
    decision: AgencyDecisionV1,
    persistent_updates: Vec<PersistentUpdateV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HardConstraintV1 {
    constraint_id: String,
    kind: String,
    scope: String,
    protected_event_ids: Vec<String>,
    target_proposal_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SoftPreferenceV1 {
    preference_id: String,
    owner_collaborator_id: String,
    dimension: String,
    context_id: String,
    source_event_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MusicalProposalV1 {
    proposal_id: String,
    author: String,
    artifact_sha256: String,
    changed_event_ids: Vec<String>,
    description: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CollaboratorInterventionV1 {
    intervention_id: String,
    actor_collaborator_id: String,
    action: String,
    proposal_id: String,
    acted_at_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgencyDecisionV1 {
    action: String,
    selected_proposal_ids: Vec<String>,
    committed: bool,
    committed_at_ns: Option<u64>,
    conflict_preserved: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistentUpdateV1 {
    update_id: String,
    target_scope: String,
    purpose: String,
    owner_collaborator_id: String,
    grant_id: Option<String>,
    authorization_status: String,
    authorized_scope: Option<String>,
    attempted: bool,
    committed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgencyAuditReport {
    pub schema_version: String,
    pub session_id: String,
    pub collaborator_id: String,
    pub decision_action: String,
    pub selected_proposal_ids: Vec<String>,
    pub proposal_count: usize,
    pub hard_constraint_count: usize,
    pub soft_preference_count: usize,
    pub intervention_count: usize,
    pub persistent_update_count: usize,
    pub attempted_persistent_update_count: usize,
    pub committed_persistent_update_count: usize,
    pub cross_collaborator_preference_count: usize,
    pub vetoed_proposal_count: usize,
    pub ignored_proposal_count: usize,
    pub hard_constraint_compliance: Option<f64>,
    pub unauthorized_persistent_update_count: usize,
    pub structurally_valid_agency_trace: bool,
    pub qualification_findings: Vec<String>,
    pub claim_ceiling: Vec<String>,
}

pub fn parse_and_audit(input: &str) -> Result<AgencyAuditReport, String> {
    let panel: AgencyPanelV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid musical agency panel JSON: {error}"))?;
    audit_panel(&panel)
}

fn audit_panel(panel: &AgencyPanelV1) -> Result<AgencyAuditReport, String> {
    validate_header(panel)?;
    let mut findings = Vec::new();
    let mut proposals = BTreeMap::<&str, &MusicalProposalV1>::new();
    let mut proposal_ids = BTreeSet::new();

    for proposal in &panel.proposals {
        require_text(&proposal.proposal_id, "proposal_id")?;
        require_text(&proposal.author, "proposal author")?;
        require_text(&proposal.description, "proposal description")?;
        require_digest(&proposal.artifact_sha256, "proposal artifact digest")?;
        if !matches!(proposal.author.as_str(), "agent" | "collaborator" | "joint") {
            return Err(format!(
                "proposal {} has unsupported author",
                proposal.proposal_id
            ));
        }
        if !proposal_ids.insert(proposal.proposal_id.as_str()) {
            return Err(format!("duplicate proposal_id: {}", proposal.proposal_id));
        }
        let changed: BTreeSet<_> = proposal.changed_event_ids.iter().collect();
        if changed.len() != proposal.changed_event_ids.len() {
            return Err(format!(
                "proposal {} has duplicate changed event IDs",
                proposal.proposal_id
            ));
        }
        proposals.insert(proposal.proposal_id.as_str(), proposal);
    }

    let mut preference_ids = BTreeSet::new();
    let mut cross_collaborator_preference_count = 0usize;
    for preference in &panel.soft_preferences {
        require_text(&preference.preference_id, "preference_id")?;
        require_text(&preference.owner_collaborator_id, "preference owner")?;
        require_text(&preference.dimension, "preference dimension")?;
        require_text(&preference.context_id, "preference context")?;
        require_digest(&preference.source_event_sha256, "preference source digest")?;
        if !preference_ids.insert(preference.preference_id.as_str()) {
            return Err(format!(
                "duplicate preference_id: {}",
                preference.preference_id
            ));
        }
        if preference.owner_collaborator_id != panel.collaborator_id {
            cross_collaborator_preference_count += 1;
            findings.push(format!(
                "soft preference {} belongs to another collaborator",
                preference.preference_id
            ));
        }
    }

    let mut constraint_ids = BTreeSet::new();
    let mut constraint_checks = 0usize;
    let mut constraint_passes = 0usize;
    for constraint in &panel.hard_constraints {
        require_text(&constraint.constraint_id, "constraint_id")?;
        require_text(&constraint.scope, "constraint scope")?;
        if !constraint_ids.insert(constraint.constraint_id.as_str()) {
            return Err(format!(
                "duplicate constraint_id: {}",
                constraint.constraint_id
            ));
        }
        if constraint.scope != "current_operation" {
            return Err(format!(
                "constraint {} is not scoped to the current operation",
                constraint.constraint_id
            ));
        }
        match constraint.kind.as_str() {
            "preserve_events" => {
                if constraint.protected_event_ids.is_empty()
                    || constraint.target_proposal_id.is_some()
                {
                    return Err(format!(
                        "preserve_events constraint {} has invalid shape",
                        constraint.constraint_id
                    ));
                }
            }
            "veto_proposal" => {
                if !constraint.protected_event_ids.is_empty()
                    || constraint
                        .target_proposal_id
                        .as_deref()
                        .is_none_or(str::is_empty)
                {
                    return Err(format!(
                        "veto_proposal constraint {} has invalid shape",
                        constraint.constraint_id
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "constraint {} has unsupported kind",
                    constraint.constraint_id
                ));
            }
        }
    }

    let mut intervention_ids = BTreeSet::new();
    let mut vetoed_proposals = BTreeSet::new();
    let mut ignored_proposals = BTreeSet::new();
    let mut accept_or_steer = BTreeSet::new();
    let mut veto_before_commit = BTreeSet::new();
    for intervention in &panel.interventions {
        require_text(&intervention.intervention_id, "intervention_id")?;
        if !intervention_ids.insert(intervention.intervention_id.as_str()) {
            return Err(format!(
                "duplicate intervention_id: {}",
                intervention.intervention_id
            ));
        }
        if intervention.actor_collaborator_id != panel.collaborator_id {
            findings.push(format!(
                "intervention {} actor does not match the active collaborator",
                intervention.intervention_id
            ));
        }
        if !proposals.contains_key(intervention.proposal_id.as_str()) {
            return Err(format!(
                "intervention {} references unknown proposal {}",
                intervention.intervention_id, intervention.proposal_id
            ));
        }
        if panel
            .decision
            .committed_at_ns
            .is_some_and(|committed_at_ns| intervention.acted_at_ns > committed_at_ns)
            && matches!(intervention.action.as_str(), "accept" | "steer" | "branch")
        {
            findings.push(format!(
                "intervention {} arrived after commit and cannot authorize the committed result",
                intervention.intervention_id
            ));
        }
        match intervention.action.as_str() {
            "accept" | "steer" => {
                accept_or_steer.insert(intervention.proposal_id.as_str());
            }
            "veto" => {
                vetoed_proposals.insert(intervention.proposal_id.as_str());
                if panel
                    .decision
                    .committed_at_ns
                    .is_some_and(|committed_at_ns| intervention.acted_at_ns <= committed_at_ns)
                {
                    veto_before_commit.insert(intervention.proposal_id.as_str());
                }
            }
            "ignore" => {
                ignored_proposals.insert(intervention.proposal_id.as_str());
            }
            "branch" | "clarify" => {}
            _ => {
                return Err(format!(
                    "intervention {} has unsupported action",
                    intervention.intervention_id
                ));
            }
        }
    }

    let mut selected_ids = BTreeSet::new();
    for proposal_id in &panel.decision.selected_proposal_ids {
        if !proposals.contains_key(proposal_id.as_str()) {
            return Err(format!("decision selects unknown proposal {proposal_id}"));
        }
        if !selected_ids.insert(proposal_id.as_str()) {
            return Err(format!("decision selects duplicate proposal {proposal_id}"));
        }
    }
    if panel.decision.committed != panel.decision.committed_at_ns.is_some() {
        return Err("decision committed flag and commit timestamp disagree".to_owned());
    }
    match panel.decision.action.as_str() {
        "apply" => {
            if !panel.decision.committed || panel.decision.selected_proposal_ids.len() != 1 {
                findings.push("apply must commit exactly one selected proposal".to_owned());
            }
        }
        "branch" => {
            if panel.decision.committed || panel.decision.selected_proposal_ids.len() < 2 {
                findings.push("branch must retain at least two alternatives without choosing a canonical commit".to_owned());
            }
            let authors: BTreeSet<_> = panel
                .decision
                .selected_proposal_ids
                .iter()
                .filter_map(|proposal_id| proposals.get(proposal_id.as_str()))
                .map(|proposal| proposal.author.as_str())
                .collect();
            if authors.len() < 2 {
                findings.push(
                    "branch must preserve at least two distinct sources of proposal".to_owned(),
                );
            }
            if !panel.decision.conflict_preserved {
                findings.push(
                    "branch action did not preserve the alternatives as an explicit conflict"
                        .to_owned(),
                );
            }
        }
        "negotiate" | "ask" | "defer" | "abstain" | "veto" => {
            if panel.decision.committed || !panel.decision.selected_proposal_ids.is_empty() {
                findings.push(format!(
                    "{} must not commit or select a canonical proposal",
                    panel.decision.action
                ));
            }
            if matches!(panel.decision.action.as_str(), "negotiate" | "ask")
                && !panel.decision.conflict_preserved
            {
                findings.push(format!(
                    "{} must keep the unresolved conflict visible",
                    panel.decision.action
                ));
            }
        }
        _ => return Err("unsupported agency decision action".to_owned()),
    }

    if panel.decision.committed {
        for proposal_id in &panel.decision.selected_proposal_ids {
            if veto_before_commit.contains(proposal_id.as_str()) {
                findings.push(format!(
                    "proposal {proposal_id} was committed after collaborator veto and before authority was restored"
                ));
            }
            if ignored_proposals.contains(proposal_id.as_str())
                && !accept_or_steer.contains(proposal_id.as_str())
            {
                findings.push(format!(
                    "ignored proposal {proposal_id} was treated as accepted without a positive intervention"
                ));
            }
        }
    }

    for constraint in &panel.hard_constraints {
        if panel.decision.selected_proposal_ids.is_empty() {
            continue;
        }
        match constraint.kind.as_str() {
            "preserve_events" => {
                let protected: BTreeSet<_> = constraint
                    .protected_event_ids
                    .iter()
                    .map(String::as_str)
                    .collect();
                let mut passed = true;
                for proposal_id in &panel.decision.selected_proposal_ids {
                    let proposal = proposals[proposal_id.as_str()];
                    if proposal
                        .changed_event_ids
                        .iter()
                        .any(|event_id| protected.contains(event_id.as_str()))
                    {
                        passed = false;
                        findings.push(format!(
                            "proposal {proposal_id} violates protected events in constraint {}",
                            constraint.constraint_id
                        ));
                    }
                }
                constraint_checks += 1;
                if passed {
                    constraint_passes += 1;
                }
            }
            "veto_proposal" => {
                let target = constraint
                    .target_proposal_id
                    .as_deref()
                    .expect("validated target");
                let passed = !panel
                    .decision
                    .selected_proposal_ids
                    .iter()
                    .any(|id| id == target);
                constraint_checks += 1;
                if !passed {
                    findings.push(format!(
                        "proposal {target} violates hard veto constraint {}",
                        constraint.constraint_id
                    ));
                } else {
                    constraint_passes += 1;
                }
            }
            _ => unreachable!("constraint kinds are validated"),
        }
    }

    let mut update_ids = BTreeSet::new();
    let mut attempted_updates = 0usize;
    let mut committed_updates = 0usize;
    let mut unauthorized_updates = 0usize;
    for update in &panel.persistent_updates {
        require_text(&update.update_id, "persistent update_id")?;
        require_text(&update.target_scope, "persistent update target_scope")?;
        require_text(&update.purpose, "persistent update purpose")?;
        require_text(&update.owner_collaborator_id, "persistent update owner")?;
        if !update_ids.insert(update.update_id.as_str()) {
            return Err(format!(
                "duplicate persistent update_id: {}",
                update.update_id
            ));
        }
        if !matches!(
            update.target_scope.as_str(),
            "session" | "project" | "collaborator" | "global"
        ) {
            return Err(format!(
                "persistent update {} has unsupported scope",
                update.update_id
            ));
        }
        if update.committed && !update.attempted {
            findings.push(format!(
                "persistent update {} was committed without an attempt",
                update.update_id
            ));
        }
        if update.attempted {
            attempted_updates += 1;
        }
        if update.committed {
            committed_updates += 1;
        }
        let required_purpose = match update.target_scope.as_str() {
            "session" => "session_steering",
            "project" => "project_memory",
            "collaborator" => "personal_adaptation",
            "global" => "global_identity",
            _ => unreachable!("target scope is validated"),
        };
        let owner_matches = update.target_scope != "collaborator"
            || update.owner_collaborator_id == panel.collaborator_id;
        let grant_matches = update
            .grant_id
            .as_deref()
            .is_some_and(|grant| !grant.trim().is_empty())
            && update.authorized_scope.as_deref() == Some(required_purpose);
        let authorized = update.authorization_status == "authorized"
            && owner_matches
            && grant_matches
            && update.purpose == required_purpose;
        if update.committed && (!update.attempted || !authorized) {
            unauthorized_updates += 1;
            let reason = if !update.attempted {
                "was committed without a recorded attempt"
            } else if update.authorization_status == "revoked" {
                "was committed after its authorization was revoked"
            } else if update.authorization_status == "unavailable" {
                "was committed while its authorization verifier was unavailable"
            } else {
                "was committed without matching purpose, owner and authorization scope"
            };
            findings.push(format!("persistent update {} {reason}", update.update_id));
        }
    }

    let structurally_valid = findings.is_empty();
    Ok(AgencyAuditReport {
        schema_version: "mel-musical-agency-report-v1".to_owned(),
        session_id: panel.session_id.clone(),
        collaborator_id: panel.collaborator_id.clone(),
        decision_action: panel.decision.action.clone(),
        selected_proposal_ids: panel.decision.selected_proposal_ids.clone(),
        proposal_count: panel.proposals.len(),
        hard_constraint_count: panel.hard_constraints.len(),
        soft_preference_count: panel.soft_preferences.len(),
        intervention_count: panel.interventions.len(),
        persistent_update_count: panel.persistent_updates.len(),
        attempted_persistent_update_count: attempted_updates,
        committed_persistent_update_count: committed_updates,
        cross_collaborator_preference_count,
        vetoed_proposal_count: vetoed_proposals.len(),
        ignored_proposal_count: ignored_proposals.len(),
        hard_constraint_compliance: if constraint_checks == 0 {
            None
        } else {
            Some(constraint_passes as f64 / constraint_checks as f64)
        },
        unauthorized_persistent_update_count: unauthorized_updates,
        structurally_valid_agency_trace: structurally_valid,
        qualification_findings: findings,
        claim_ceiling: panel.claim_ceiling.clone(),
    })
}

fn validate_header(panel: &AgencyPanelV1) -> Result<(), String> {
    if panel.schema_version != SCHEMA_VERSION {
        return Err("unsupported musical agency schema_version".to_owned());
    }
    require_text(&panel.session_id, "session_id")?;
    require_text(&panel.collaborator_id, "collaborator_id")?;
    require_text(&panel.clock_domain_id, "clock_domain_id")?;
    if panel.proposals.is_empty() {
        return Err("proposals must be non-empty".to_owned());
    }
    if panel.hard_constraints.is_empty() {
        return Err("hard_constraints must be non-empty".to_owned());
    }
    if panel.claim_ceiling.is_empty()
        || panel
            .claim_ceiling
            .iter()
            .any(|claim| claim.trim().is_empty())
    {
        return Err("claim_ceiling must contain explicit limitations".to_owned());
    }
    Ok(())
}

fn require_text(value: &str, name: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must be non-empty"))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, name: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!(
            "{name} must be a lowercase 64-character SHA-256 digest"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../fixtures/valid-musical-agency-v1.json");

    fn panel() -> Value {
        serde_json::from_str(FIXTURE).expect("known-answer agency fixture")
    }

    fn audit(value: &Value) -> Result<AgencyAuditReport, String> {
        parse_and_audit(&value.to_string())
    }

    #[test]
    fn valid_negotiated_branch_preserves_two_distinct_proposals() {
        let report = parse_and_audit(FIXTURE).expect("known-answer agency fixture");
        assert!(report.structurally_valid_agency_trace);
        assert_eq!(report.decision_action, "branch");
        assert_eq!(
            report.selected_proposal_ids,
            vec!["agent-proposal", "collaborator-proposal"]
        );
        assert_eq!(report.hard_constraint_compliance, Some(1.0));
    }

    #[test]
    fn stale_collaborator_preference_is_not_silently_admitted() {
        let mut value = panel();
        value["soft_preferences"][0]["owner_collaborator_id"] = serde_json::json!("collaborator-B");
        let report = audit(&value).expect("cross-collaborator memory must be reportable");
        assert!(!report.structurally_valid_agency_trace);
        assert_eq!(report.cross_collaborator_preference_count, 1);
    }

    #[test]
    fn soft_preference_cannot_smuggle_in_hard_authority() {
        let mut value = panel();
        value["soft_preferences"][0]["authority"] = serde_json::json!("hard");
        let error = audit(&value).expect_err("soft preference schema rejects authority escalation");
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn hard_protected_event_cannot_be_changed_by_selected_proposal() {
        let mut value = panel();
        value["proposals"][0]["changed_event_ids"] = serde_json::json!(["protected-note-1"]);
        value["decision"]["action"] = serde_json::json!("apply");
        value["decision"]["selected_proposal_ids"] = serde_json::json!(["agent-proposal"]);
        value["decision"]["committed"] = true;
        value["decision"]["committed_at_ns"] = serde_json::json!(200);
        let report = audit(&value).expect("constraint violation should be a report finding");
        assert!(!report.structurally_valid_agency_trace);
        assert_eq!(report.hard_constraint_compliance, Some(0.0));
    }

    #[test]
    fn vetoed_proposal_cannot_be_committed_after_veto() {
        let mut value = panel();
        value["interventions"][0]["action"] = serde_json::json!("veto");
        value["decision"]["action"] = serde_json::json!("apply");
        value["decision"]["selected_proposal_ids"] = serde_json::json!(["agent-proposal"]);
        value["decision"]["committed"] = true;
        value["decision"]["committed_at_ns"] = serde_json::json!(300);
        let report = audit(&value).expect("veto race should be reported");
        assert!(!report.structurally_valid_agency_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|f| f.contains("veto"))
        );
    }

    #[test]
    fn ignored_preview_is_not_implicitly_accepted() {
        let mut value = panel();
        value["interventions"][0]["action"] = serde_json::json!("ignore");
        value["decision"]["action"] = serde_json::json!("apply");
        value["decision"]["selected_proposal_ids"] = serde_json::json!(["agent-proposal"]);
        value["decision"]["committed"] = true;
        value["decision"]["committed_at_ns"] = serde_json::json!(300);
        let report = audit(&value).expect("ignored proposal must not be treated as accepted");
        assert!(!report.structurally_valid_agency_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|f| f.contains("ignored proposal"))
        );
    }

    #[test]
    fn negotiation_must_preserve_the_unresolved_conflict() {
        let mut value = panel();
        value["decision"]["action"] = serde_json::json!("negotiate");
        value["decision"]["selected_proposal_ids"] = serde_json::json!([]);
        value["decision"]["committed"] = false;
        value["decision"]["committed_at_ns"] = serde_json::Value::Null;
        value["decision"]["conflict_preserved"] = false;
        let report = audit(&value).expect("hidden conflict scalarization must be visible");
        assert!(!report.structurally_valid_agency_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|f| f.contains("conflict"))
        );
    }

    #[test]
    fn a_branch_keeps_alternatives_without_canonical_commit() {
        let mut value = panel();
        value["decision"]["action"] = serde_json::json!("branch");
        value["decision"]["selected_proposal_ids"] = serde_json::json!(["agent-proposal"]);
        value["decision"]["committed"] = false;
        value["decision"]["committed_at_ns"] = serde_json::Value::Null;
        let report = audit(&value).expect("branch with one alternative is invalid");
        assert!(!report.structurally_valid_agency_trace);
    }

    #[test]
    fn revoked_memory_update_cannot_be_committed() {
        let mut value = panel();
        value["persistent_updates"][0]["authorization_status"] = serde_json::json!("revoked");
        value["persistent_updates"][0]["committed"] = true;
        let report = audit(&value).expect("revoked write is reported");
        assert!(!report.structurally_valid_agency_trace);
        assert_eq!(report.unauthorized_persistent_update_count, 1);
    }

    #[test]
    fn collaborator_memory_cannot_be_written_for_a_different_owner() {
        let mut value = panel();
        value["persistent_updates"][0]["owner_collaborator_id"] =
            serde_json::json!("collaborator-B");
        value["persistent_updates"][0]["committed"] = true;
        let report = audit(&value).expect("cross-collaborator write is reported");
        assert!(!report.structurally_valid_agency_trace);
        assert_eq!(report.unauthorized_persistent_update_count, 1);
    }

    #[test]
    fn authorization_unavailable_fails_closed_for_persistent_commit() {
        let mut value = panel();
        value["persistent_updates"][0]["authorization_status"] = serde_json::json!("unavailable");
        value["persistent_updates"][0]["committed"] = true;
        let report = audit(&value).expect("unavailable verifier must not authorize a write");
        assert!(!report.structurally_valid_agency_trace);
        assert_eq!(report.unauthorized_persistent_update_count, 1);
    }

    #[test]
    fn collaborator_can_ignore_preview_and_agent_can_retain_a_separate_proposal() {
        let mut value = panel();
        value["interventions"][0]["action"] = serde_json::json!("ignore");
        value["decision"]["action"] = serde_json::json!("branch");
        value["decision"]["selected_proposal_ids"] =
            serde_json::json!(["agent-proposal", "collaborator-proposal"]);
        value["decision"]["committed"] = false;
        value["decision"]["committed_at_ns"] = serde_json::Value::Null;
        let report =
            audit(&value).expect("branching keeps proposals distinct without overriding ignore");
        assert!(report.structurally_valid_agency_trace);
        assert_eq!(report.ignored_proposal_count, 1);
    }

    #[test]
    fn persistent_global_update_requires_an_explicit_global_identity_purpose() {
        let mut value = panel();
        value["persistent_updates"][0]["target_scope"] = serde_json::json!("global");
        value["persistent_updates"][0]["purpose"] = serde_json::json!("personal_adaptation");
        value["persistent_updates"][0]["authorized_scope"] =
            serde_json::json!("personal_adaptation");
        value["persistent_updates"][0]["committed"] = true;
        let report = audit(&value).expect("personal grant cannot authorize global identity update");
        assert!(!report.structurally_valid_agency_trace);
        assert_eq!(report.unauthorized_persistent_update_count, 1);
    }

    #[test]
    fn branch_requires_distinct_agent_and_collaborator_sources() {
        let mut value = panel();
        value["proposals"][1]["author"] = serde_json::json!("agent");
        let report = audit(&value).expect("single-source branching is reported");
        assert!(!report.structurally_valid_agency_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|f| f.contains("distinct sources"))
        );
    }

    #[test]
    fn positive_accept_after_commit_cannot_authorize_the_result() {
        let mut value = panel();
        value["decision"]["action"] = serde_json::json!("apply");
        value["decision"]["selected_proposal_ids"] = serde_json::json!(["agent-proposal"]);
        value["decision"]["committed"] = true;
        value["decision"]["committed_at_ns"] = serde_json::json!(200);
        value["interventions"][0]["action"] = serde_json::json!("accept");
        value["interventions"][0]["acted_at_ns"] = serde_json::json!(250);
        let report = audit(&value).expect("post-commit accept remains reportable");
        assert!(!report.structurally_valid_agency_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|f| f.contains("after commit"))
        );
    }

    #[test]
    fn abstention_does_not_claim_hard_constraint_compliance() {
        let mut value = panel();
        value["decision"]["action"] = serde_json::json!("abstain");
        value["decision"]["selected_proposal_ids"] = serde_json::json!([]);
        value["decision"]["committed"] = false;
        value["decision"]["committed_at_ns"] = serde_json::Value::Null;
        let report = audit(&value).expect("abstention is a valid no-commit result");
        assert_eq!(report.hard_constraint_compliance, None);
    }

    #[test]
    fn non_committing_abstention_is_a_valid_outcome() {
        let mut value = panel();
        value["decision"]["action"] = serde_json::json!("abstain");
        value["decision"]["selected_proposal_ids"] = serde_json::json!([]);
        value["decision"]["committed"] = false;
        value["decision"]["committed_at_ns"] = serde_json::Value::Null;
        value["decision"]["conflict_preserved"] = true;
        let report = audit(&value).expect("abstention is a valid non-commit outcome");
        assert!(report.structurally_valid_agency_trace);
    }
}
