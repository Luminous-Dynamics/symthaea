//! Security Coordinator Zome - Business logic for security logging and auditing
use hdk::prelude::*;
use security_integrity::*;

// ===== Input/Output Types =====

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateSecurityLogInput {
    pub event_type: SecurityEventType,
    pub severity: SecuritySeverity,
    pub zome_name: String,
    pub function_name: String,
    pub description: String,
    pub trigger_input: Option<String>,
    pub metadata: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SecurityLogOutput {
    pub log_hash: ActionHash,
    pub log: SecurityLog,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SecurityLogsResponse {
    pub logs: Vec<SecurityLogOutput>,
    pub total_count: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SecurityStatsOutput {
    pub total_events: u32,
    pub critical_events: u32,
    pub high_events: u32,
    pub events_by_type: Vec<(String, u32)>,
    pub top_offenders: Vec<(AgentPubKey, u32)>,
}

// ===== Helper Functions =====

fn get_all_security_logs_links() -> ExternResult<Vec<Link>> {
    let path = Path::from("all_security_logs");
    let filter = LinkTypeFilter::try_from(LinkTypes::AllSecurityLogs)?;
    get_links(LinkQuery::new(path.path_entry_hash()?, filter), GetStrategy::default())
}

fn get_agent_security_logs_links(agent: AgentPubKey) -> ExternResult<Vec<Link>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToSecurityLogs)?;
    get_links(LinkQuery::new(agent, filter), GetStrategy::default())
}

// ===== Public API =====

/// Log a security event
///
/// This creates a security log entry on the DHT for auditing purposes.
#[hdk_extern]
pub fn log_security_event(input: CreateSecurityLogInput) -> ExternResult<ActionHash> {
    // Input validation
    if input.description.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Security log description cannot be empty".into()
        )));
    }

    if input.description.len() > 1000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Security log description too long (max 1000 characters)".into()
        )));
    }

    if input.zome_name.is_empty() || input.function_name.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Zome and function names required".into()
        )));
    }

    let agent_info = agent_info()?;

    let log = SecurityLog {
        event_type: input.event_type.clone(),
        severity: input.severity,
        agent: agent_info.agent_initial_pubkey.clone(),
        zome_name: input.zome_name,
        function_name: input.function_name,
        description: input.description,
        trigger_input: input.trigger_input,
        timestamp: sys_time()?,
        metadata: input.metadata,
    };

    // Create entry
    let action_hash = create_entry(&EntryTypes::SecurityLog(log.clone()))?;
    let entry_hash = hash_entry(&log)?;

    // Create discovery links

    // 1. Agent -> SecurityLog
    create_link(
        agent_info.agent_initial_pubkey,
        entry_hash.clone(),
        LinkTypes::AgentToSecurityLogs,
        (),
    )?;

    // 2. All security logs anchor
    let all_path = Path::from("all_security_logs");
    create_link(
        all_path.path_entry_hash()?,
        entry_hash.clone(),
        LinkTypes::AllSecurityLogs,
        (),
    )?;

    // 3. EventType -> SecurityLog (for filtering by type)
    let event_type_path = Path::from(format!("security_logs.event_type.{:?}", log.event_type));
    create_link(
        event_type_path.path_entry_hash()?,
        entry_hash,
        LinkTypes::EventTypeToLogs,
        (),
    )?;

    Ok(action_hash)
}

/// Get a specific security log by hash
#[hdk_extern]
pub fn get_security_log(log_hash: ActionHash) -> ExternResult<Option<SecurityLogOutput>> {
    let record = get(log_hash.clone(), GetOptions::default())?;

    match record {
        Some(record) => {
            let log: SecurityLog = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(e))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Failed to deserialize security log".into()
                )))?;

            Ok(Some(SecurityLogOutput {
                log_hash,
                log,
            }))
        }
        None => Ok(None),
    }
}

/// Get all security logs (admin function)
#[hdk_extern]
pub fn get_all_security_logs(_: ()) -> ExternResult<SecurityLogsResponse> {
    let links = get_all_security_logs_links()?;

    let mut logs = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(log_output) = get_security_log(action_hash)? {
                logs.push(log_output);
            }
        }
    }

    let total_count = logs.len() as u32;

    Ok(SecurityLogsResponse { logs, total_count })
}

/// Get security logs for a specific agent
#[hdk_extern]
pub fn get_agent_security_logs(agent: AgentPubKey) -> ExternResult<SecurityLogsResponse> {
    let links = get_agent_security_logs_links(agent)?;

    let mut logs = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(log_output) = get_security_log(action_hash)? {
                logs.push(log_output);
            }
        }
    }

    let total_count = logs.len() as u32;

    Ok(SecurityLogsResponse { logs, total_count })
}

/// Get my security logs (current agent)
#[hdk_extern]
pub fn get_my_security_logs(_: ()) -> ExternResult<SecurityLogsResponse> {
    let agent_info = agent_info()?;
    get_agent_security_logs(agent_info.agent_initial_pubkey)
}

/// Get security logs by event type
#[hdk_extern]
pub fn get_logs_by_event_type(event_type: SecurityEventType) -> ExternResult<SecurityLogsResponse> {
    let path = Path::from(format!("security_logs.event_type.{:?}", event_type));
    let filter = LinkTypeFilter::try_from(LinkTypes::EventTypeToLogs)?;
    let links = get_links(LinkQuery::new(path.path_entry_hash()?, filter), GetStrategy::default())?;

    let mut logs = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(log_output) = get_security_log(action_hash)? {
                logs.push(log_output);
            }
        }
    }

    let total_count = logs.len() as u32;

    Ok(SecurityLogsResponse { logs, total_count })
}

/// Get security statistics
#[hdk_extern]
pub fn get_security_stats(_: ()) -> ExternResult<SecurityStatsOutput> {
    let all_logs = get_all_security_logs(())?;

    let total_events = all_logs.total_count;
    let mut critical_events = 0u32;
    let mut high_events = 0u32;
    let mut event_type_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut agent_counts: std::collections::HashMap<AgentPubKey, u32> = std::collections::HashMap::new();

    for log_output in all_logs.logs {
        // Count by severity
        match log_output.log.severity {
            SecuritySeverity::Critical => critical_events += 1,
            SecuritySeverity::High => high_events += 1,
            _ => {}
        }

        // Count by event type
        let event_type_str = format!("{:?}", log_output.log.event_type);
        *event_type_counts.entry(event_type_str).or_insert(0) += 1;

        // Count by agent
        *agent_counts.entry(log_output.log.agent).or_insert(0) += 1;
    }

    // Convert to sorted vectors
    let mut events_by_type: Vec<(String, u32)> = event_type_counts.into_iter().collect();
    events_by_type.sort_by(|a, b| b.1.cmp(&a.1));

    let mut top_offenders: Vec<(AgentPubKey, u32)> = agent_counts.into_iter().collect();
    top_offenders.sort_by(|a, b| b.1.cmp(&a.1));
    top_offenders.truncate(10); // Top 10 only

    Ok(SecurityStatsOutput {
        total_events,
        critical_events,
        high_events,
        events_by_type,
        top_offenders,
    })
}

#[cfg(test)]
mod tests;
