//! Support Tickets Coordinator Zome
//! Business logic for ticket lifecycle, comments, autonomous actions,
//! undo operations, and preemptive alerts.

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use support_tickets_integrity::*;
use support_types::{sharded_anchor, TicketStatus};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}
#[cfg(test)]
use support_tickets_integrity::{EscalationLevel, SatisfactionSurvey};
#[cfg(test)]
use support_types::{AutonomyLevel, SupportCategory, TicketPriority};

// ============================================================================
// BRIDGE SIGNAL (for cross-domain UI notification)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub event_type: String,
    pub source_zome: String,
    pub payload: String,
}

// ============================================================================
// HELPERS
// ============================================================================

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

fn extract_ticket(record: &Record) -> ExternResult<SupportTicket> {
    record
        .entry()
        .to_app_option::<SupportTicket>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Not a SupportTicket".into()
        )))
}

fn extract_autonomous_action(record: &Record) -> ExternResult<AutonomousAction> {
    record
        .entry()
        .to_app_option::<AutonomousAction>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Not an AutonomousAction".into()
        )))
}

// ============================================================================
// INPUT TYPES
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateTicketInput {
    pub original_hash: ActionHash,
    pub updated: SupportTicket,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PromoteAlertInput {
    pub alert_hash: ActionHash,
    pub ticket: SupportTicket,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EscalateInput {
    pub ticket_hash: ActionHash,
    pub from_level: EscalationLevel,
    pub to_level: EscalationLevel,
    pub reason: String,
}

// ============================================================================
// TICKET LIFECYCLE
// ============================================================================

#[hdk_extern]
pub fn create_ticket(ticket: SupportTicket) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_ticket")?;
    let action_hash = create_entry(&EntryTypes::SupportTicket(ticket.clone()))?;

    // ShardedTickets link (time-sharded anchor)
    let shard_anchor = sharded_anchor("support", "tickets", &ticket.created_at);
    create_entry(&EntryTypes::Anchor(Anchor(shard_anchor.clone())))?;
    create_link(
        anchor_hash(&shard_anchor)?,
        action_hash.clone(),
        LinkTypes::ShardedTickets,
        (),
    )?;

    // AgentToTicket link
    create_link(
        ticket.requester.clone(),
        action_hash.clone(),
        LinkTypes::AgentToTicket,
        (),
    )?;

    // StatusToTicket link
    let status_anchor = format!("support:status:{:?}", ticket.status);
    create_entry(&EntryTypes::Anchor(Anchor(status_anchor.clone())))?;
    create_link(
        anchor_hash(&status_anchor)?,
        action_hash.clone(),
        LinkTypes::StatusToTicket,
        (),
    )?;

    // CategoryToTicket link
    let category_anchor = format!("support:category:{:?}", ticket.category);
    create_entry(&EntryTypes::Anchor(Anchor(category_anchor.clone())))?;
    create_link(
        anchor_hash(&category_anchor)?,
        action_hash.clone(),
        LinkTypes::CategoryToTicket,
        (),
    )?;

    let _ = emit_signal(&BridgeEventSignal {
        event_type: "ticket_created".to_string(),
        source_zome: "support_tickets".to_string(),
        payload: format!(r#"{{"ticket_hash":"{}"}}"#, action_hash),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created ticket".into()
    )))
}

#[hdk_extern]
pub fn update_ticket(input: UpdateTicketInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "update_ticket")?;
    let action_hash = update_entry(
        input.original_hash,
        &EntryTypes::SupportTicket(input.updated),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated ticket".into()
    )))
}

#[hdk_extern]
pub fn get_ticket(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

#[hdk_extern]
pub fn list_tickets_by_status(status: TicketStatus) -> ExternResult<Vec<Record>> {
    let status_anchor = format!("support:status:{:?}", status);
    create_entry(&EntryTypes::Anchor(Anchor(status_anchor.clone())))?;
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&status_anchor)?, LinkTypes::StatusToTicket)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn list_my_tickets(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToTicket)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn close_ticket(action_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "close_ticket")?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Ticket not found".into())
    ))?;
    let mut ticket = extract_ticket(&record)?;
    ticket.status = TicketStatus::Closed;
    let new_hash = update_entry(action_hash, &EntryTypes::SupportTicket(ticket))?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find closed ticket".into()
    )))
}

// ============================================================================
// COMMENTS
// ============================================================================

#[hdk_extern]
pub fn add_comment(comment: TicketComment) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "add_comment")?;
    let action_hash = create_entry(&EntryTypes::TicketComment(comment.clone()))?;
    create_link(
        comment.ticket_hash.clone(),
        action_hash.clone(),
        LinkTypes::TicketToComment,
        (),
    )?;

    let _ = emit_signal(&BridgeEventSignal {
        event_type: "comment_added".to_string(),
        source_zome: "support_tickets".to_string(),
        payload: format!(
            r#"{{"comment_hash":"{}","ticket_hash":"{}"}}"#,
            action_hash, comment.ticket_hash
        ),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created comment".into()
    )))
}

#[hdk_extern]
pub fn get_comments(ticket_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(ticket_hash, LinkTypes::TicketToComment)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// AUTONOMOUS ACTIONS
// ============================================================================

#[hdk_extern]
pub fn propose_action(action: AutonomousAction) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "propose_action")?;
    let action_hash = create_entry(&EntryTypes::AutonomousAction(action.clone()))?;
    create_link(
        action.ticket_hash.clone(),
        action_hash.clone(),
        LinkTypes::TicketToAction,
        (),
    )?;

    let _ = emit_signal(&BridgeEventSignal {
        event_type: "action_proposed".to_string(),
        source_zome: "support_tickets".to_string(),
        payload: format!(
            r#"{{"action_hash":"{}","ticket_hash":"{}"}}"#,
            action_hash, action.ticket_hash
        ),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created action".into()
    )))
}

#[hdk_extern]
pub fn approve_action(action_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "approve_action")?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Action not found".into())
    ))?;
    let mut action = extract_autonomous_action(&record)?;
    action.approved = true;
    let new_hash = update_entry(action_hash, &EntryTypes::AutonomousAction(action))?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find approved action".into()
    )))
}

#[hdk_extern]
pub fn execute_action(action_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "execute_action")?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Action not found".into())
    ))?;
    let mut action = extract_autonomous_action(&record)?;
    action.executed = true;
    let new_hash = update_entry(action_hash, &EntryTypes::AutonomousAction(action))?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find executed action".into()
    )))
}

#[hdk_extern]
pub fn rollback_action(action_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "rollback_action")?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Action not found".into())
    ))?;
    let mut action = extract_autonomous_action(&record)?;
    action.rolled_back = true;
    let new_hash = update_entry(action_hash, &EntryTypes::AutonomousAction(action))?;
    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find rolled-back action".into()
    )))
}

// ============================================================================
// UNDO
// ============================================================================

#[hdk_extern]
pub fn create_undo(undo: UndoAction) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_undo")?;
    let action_hash = create_entry(&EntryTypes::UndoAction(undo.clone()))?;
    create_link(
        undo.original_action_hash.clone(),
        action_hash.clone(),
        LinkTypes::ActionToUndo,
        (),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created undo action".into()
    )))
}

// ============================================================================
// PREEMPTIVE ALERTS
// ============================================================================

#[hdk_extern]
pub fn create_preemptive_alert(alert: PreemptiveAlert) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_preemptive_alert")?;
    let agent = agent_info()?.agent_initial_pubkey;
    let action_hash = create_entry(&EntryTypes::PreemptiveAlert(alert.clone()))?;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToAlert, ())?;

    let _ = emit_signal(&BridgeEventSignal {
        event_type: "preemptive_alert".to_string(),
        source_zome: "support_tickets".to_string(),
        payload: format!(
            r#"{{"alert_hash":"{}","predicted_failure":"{}"}}"#,
            action_hash, alert.predicted_failure
        ),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created alert".into()
    )))
}

#[hdk_extern]
pub fn list_preemptive_alerts(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToAlert)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn promote_alert_to_ticket(input: PromoteAlertInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "promote_alert_to_ticket")?;
    // Verify alert exists
    let _alert = get(input.alert_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Alert not found".into())))?;

    // Create the ticket (reuses create_ticket logic inline for linking)
    let mut ticket = input.ticket;
    ticket.is_preemptive = true;

    let action_hash = create_entry(&EntryTypes::SupportTicket(ticket.clone()))?;

    // ShardedTickets link
    let shard_anchor = sharded_anchor("support", "tickets", &ticket.created_at);
    create_entry(&EntryTypes::Anchor(Anchor(shard_anchor.clone())))?;
    create_link(
        anchor_hash(&shard_anchor)?,
        action_hash.clone(),
        LinkTypes::ShardedTickets,
        (),
    )?;

    // AgentToTicket link
    create_link(
        ticket.requester.clone(),
        action_hash.clone(),
        LinkTypes::AgentToTicket,
        (),
    )?;

    // StatusToTicket link
    let status_anchor = format!("support:status:{:?}", ticket.status);
    create_entry(&EntryTypes::Anchor(Anchor(status_anchor.clone())))?;
    create_link(
        anchor_hash(&status_anchor)?,
        action_hash.clone(),
        LinkTypes::StatusToTicket,
        (),
    )?;

    // CategoryToTicket link
    let category_anchor = format!("support:category:{:?}", ticket.category);
    create_entry(&EntryTypes::Anchor(Anchor(category_anchor.clone())))?;
    create_link(
        anchor_hash(&category_anchor)?,
        action_hash.clone(),
        LinkTypes::CategoryToTicket,
        (),
    )?;

    let _ = emit_signal(&BridgeEventSignal {
        event_type: "alert_promoted_to_ticket".to_string(),
        source_zome: "support_tickets".to_string(),
        payload: format!(
            r#"{{"alert_hash":"{}","ticket_hash":"{}"}}"#,
            input.alert_hash, action_hash
        ),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find promoted ticket".into()
    )))
}

// ============================================================================
// ESCALATION
// ============================================================================

#[hdk_extern]
pub fn escalate_ticket(input: EscalateInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "escalate_ticket")?;
    let _ticket = get(input.ticket_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Ticket not found".into())
    ))?;
    let agent = agent_info()?.agent_initial_pubkey;
    let escalation = Escalation {
        ticket_hash: input.ticket_hash.clone(),
        escalated_by: agent,
        from_level: input.from_level,
        to_level: input.to_level,
        reason: input.reason,
        escalated_at: sys_time()?,
    };
    let action_hash = create_entry(&EntryTypes::Escalation(escalation))?;
    create_link(
        input.ticket_hash,
        action_hash.clone(),
        LinkTypes::TicketToEscalations,
        (),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created escalation".into()
    )))
}

#[hdk_extern]
pub fn get_escalation_history(ticket_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(ticket_hash, LinkTypes::TicketToEscalations)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// SATISFACTION SURVEY
// ============================================================================

#[hdk_extern]
pub fn submit_satisfaction(survey: SatisfactionSurvey) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "submit_satisfaction")?;
    let action_hash = create_entry(&EntryTypes::SatisfactionSurvey(survey.clone()))?;
    create_link(
        survey.ticket_hash,
        action_hash.clone(),
        LinkTypes::TicketToSurvey,
        (),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created satisfaction survey".into()
    )))
}

#[hdk_extern]
pub fn get_ticket_satisfaction(ticket_hash: ActionHash) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(ticket_hash, LinkTypes::TicketToSurvey)?,
        GetStrategy::default(),
    )?;
    if links.is_empty() {
        return Ok(None);
    }
    let latest = links.last().unwrap();
    let action_hash = ActionHash::try_from(latest.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    get(action_hash, GetOptions::default())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xab; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xdb; 36])
    }

    fn fake_timestamp() -> Timestamp {
        Timestamp::from_micros(1_700_000_000_000_000)
    }

    // -- Serde roundtrip: UpdateTicketInput -----------------------------------

    #[test]
    fn serde_roundtrip_update_ticket_input() {
        let input = UpdateTicketInput {
            original_hash: fake_action_hash(),
            updated: SupportTicket {
                title: "Updated title".into(),
                description: "Updated description".into(),
                category: SupportCategory::Software,
                priority: TicketPriority::Medium,
                status: TicketStatus::InProgress,
                requester: fake_agent(),
                assignee: Some(fake_agent()),
                autonomy_level: AutonomyLevel::SemiAutonomous,
                system_info: None,
                is_preemptive: false,
                prediction_confidence: None,
                created_at: fake_timestamp(),
                updated_at: fake_timestamp(),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: UpdateTicketInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.updated.title, "Updated title");
        assert_eq!(back.updated.status, TicketStatus::InProgress);
    }

    #[test]
    fn serde_roundtrip_update_ticket_input_with_all_optionals() {
        let input = UpdateTicketInput {
            original_hash: fake_action_hash(),
            updated: SupportTicket {
                title: "Full optionals".into(),
                description: "All fields set".into(),
                category: SupportCategory::Security,
                priority: TicketPriority::Critical,
                status: TicketStatus::Resolved,
                requester: fake_agent(),
                assignee: Some(fake_agent()),
                autonomy_level: AutonomyLevel::FullAutonomous,
                system_info: Some(r#"{"kernel":"6.18.9"}"#.into()),
                is_preemptive: true,
                prediction_confidence: Some(0.95),
                created_at: fake_timestamp(),
                updated_at: fake_timestamp(),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: UpdateTicketInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.updated.prediction_confidence, Some(0.95));
        assert!(back.updated.is_preemptive);
        assert!(back.updated.system_info.is_some());
    }

    // -- Serde roundtrip: PromoteAlertInput -----------------------------------

    #[test]
    fn serde_roundtrip_promote_alert_input() {
        let input = PromoteAlertInput {
            alert_hash: fake_action_hash(),
            ticket: SupportTicket {
                title: "Alert-promoted ticket".into(),
                description: "Disk space running low".into(),
                category: SupportCategory::Hardware,
                priority: TicketPriority::High,
                status: TicketStatus::Open,
                requester: fake_agent(),
                assignee: None,
                autonomy_level: AutonomyLevel::Advisory,
                system_info: None,
                is_preemptive: true,
                prediction_confidence: Some(0.87),
                created_at: fake_timestamp(),
                updated_at: fake_timestamp(),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: PromoteAlertInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ticket.title, "Alert-promoted ticket");
        assert_eq!(back.ticket.prediction_confidence, Some(0.87));
    }

    #[test]
    fn serde_roundtrip_promote_alert_input_minimal() {
        let input = PromoteAlertInput {
            alert_hash: fake_action_hash(),
            ticket: SupportTicket {
                title: "Minimal".into(),
                description: "Bare minimum".into(),
                category: SupportCategory::General,
                priority: TicketPriority::Low,
                status: TicketStatus::Open,
                requester: fake_agent(),
                assignee: None,
                autonomy_level: AutonomyLevel::Advisory,
                system_info: None,
                is_preemptive: false,
                prediction_confidence: None,
                created_at: fake_timestamp(),
                updated_at: fake_timestamp(),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: PromoteAlertInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ticket.title, "Minimal");
        assert_eq!(back.ticket.assignee, None);
    }

    // -- Serde roundtrip: BridgeEventSignal -----------------------------------

    #[test]
    fn serde_roundtrip_bridge_event_signal() {
        let signal = BridgeEventSignal {
            event_type: "ticket_created".to_string(),
            source_zome: "support_tickets".to_string(),
            payload: r#"{"ticket_hash":"uhCkk..."}"#.to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let back: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type, "ticket_created");
        assert_eq!(back.source_zome, "support_tickets");
        assert!(back.payload.contains("ticket_hash"));
    }

    #[test]
    fn serde_roundtrip_bridge_event_signal_empty_payload() {
        let signal = BridgeEventSignal {
            event_type: "test".to_string(),
            source_zome: "support_tickets".to_string(),
            payload: String::new(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let back: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.payload, "");
    }

    #[test]
    fn serde_roundtrip_bridge_event_signal_json_fields() {
        let signal = BridgeEventSignal {
            event_type: "action_proposed".to_string(),
            source_zome: "support_tickets".to_string(),
            payload: "{}".to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("\"event_type\""));
        assert!(json.contains("\"source_zome\""));
        assert!(json.contains("\"payload\""));
    }

    #[test]
    fn bridge_event_signal_clone_is_equal() {
        let signal = BridgeEventSignal {
            event_type: "preemptive_alert".to_string(),
            source_zome: "support_tickets".to_string(),
            payload: r#"{"free_energy":42.5}"#.to_string(),
        };
        let cloned = signal.clone();
        assert_eq!(cloned.event_type, signal.event_type);
        assert_eq!(cloned.source_zome, signal.source_zome);
        assert_eq!(cloned.payload, signal.payload);
    }

    // -- Serde roundtrip: EscalateInput ---------------------------------------

    #[test]
    fn serde_roundtrip_escalate_input() {
        let input = EscalateInput {
            ticket_hash: fake_action_hash(),
            from_level: EscalationLevel::Tier1,
            to_level: EscalationLevel::Tier2,
            reason: "Customer requires specialist attention".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: EscalateInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.reason, "Customer requires specialist attention");
    }

    #[test]
    fn serde_roundtrip_escalate_input_all_levels() {
        for (from, to) in [
            (EscalationLevel::Tier1, EscalationLevel::Tier2),
            (EscalationLevel::Tier2, EscalationLevel::Management),
            (EscalationLevel::Management, EscalationLevel::Emergency),
            (EscalationLevel::Tier1, EscalationLevel::Emergency),
        ] {
            let input = EscalateInput {
                ticket_hash: fake_action_hash(),
                from_level: from.clone(),
                to_level: to.clone(),
                reason: "Escalation test".into(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let back: EscalateInput = serde_json::from_str(&json).unwrap();
            assert_eq!(back.from_level, from);
            assert_eq!(back.to_level, to);
        }
    }
}
