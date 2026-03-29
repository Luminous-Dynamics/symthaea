// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audit Coordinator Zome
//!
//! Append-only audit log operations.

use hdk::prelude::*;
use audit_integrity::*;

const ALL_ENTRIES_ANCHOR: &str = "all_audit_entries";
const SUMMARIES_ANCHOR: &str = "audit_summaries";

// ==================== ENTRY CREATION ====================

/// Create an audit entry (append-only)
#[hdk_extern]
pub fn create_audit_entry(entry: AuditEntry) -> ExternResult<ActionHash> {
    let action_hash = hdk::entry::create_entry(EntryTypes::AuditEntry(entry.clone()))?;

    // Link to all entries
    let all_anchor = anchor_hash(ALL_ENTRIES_ANCHOR)?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllAuditEntries,
        entry.timestamp.to_be_bytes().to_vec(),
    )?;

    // Link by category
    let category_anchor = category_anchor(&entry.category)?;
    create_link(
        category_anchor,
        action_hash.clone(),
        LinkTypes::EntriesByCategory,
        entry.timestamp.to_be_bytes().to_vec(),
    )?;

    // Link by severity
    let severity_anchor = severity_anchor(&entry.severity)?;
    create_link(
        severity_anchor,
        action_hash.clone(),
        LinkTypes::EntriesBySeverity,
        entry.timestamp.to_be_bytes().to_vec(),
    )?;

    // Link by actor if available
    if let Some(agent) = &entry.actor.agent_pub_key {
        create_link(
            agent.clone(),
            action_hash.clone(),
            LinkTypes::EntriesByActor,
            entry.timestamp.to_be_bytes().to_vec(),
        )?;
    }

    // Link by correlation ID if available
    if let Some(correlation_id) = &entry.metadata.correlation_id {
        let correlation_anchor = correlation_anchor(correlation_id)?;
        create_link(
            correlation_anchor,
            action_hash.clone(),
            LinkTypes::EntriesByCorrelation,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Create multiple entries in batch
#[hdk_extern]
pub fn create_entries(entries: Vec<AuditEntry>) -> ExternResult<Vec<ActionHash>> {
    let mut hashes = Vec::new();
    for entry in entries {
        let hash = hdk::entry::create_entry(EntryTypes::AuditEntry(entry))?;
        hashes.push(hash);
    }
    Ok(hashes)
}

// ==================== QUERYING ====================

/// Query input for filtering entries
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditQuery {
    pub start_date: Option<u64>,
    pub end_date: Option<u64>,
    pub categories: Option<Vec<AuditCategory>>,
    pub severities: Option<Vec<AuditSeverity>>,
    pub actor_agent: Option<AgentPubKey>,
    pub correlation_id: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Get entries with optional filters
#[hdk_extern]
pub fn get_entries(query: Option<AuditQuery>) -> ExternResult<Vec<AuditEntry>> {
    let query = query.unwrap_or(AuditQuery {
        start_date: None,
        end_date: None,
        categories: None,
        severities: None,
        actor_agent: None,
        correlation_id: None,
        limit: Some(100),
        offset: None,
    });

    // Determine which anchor to query from
    let links = if let Some(correlation_id) = &query.correlation_id {
        // Query by correlation
        let anchor = correlation_anchor(correlation_id)?;
        get_links(LinkQuery::try_new(anchor, LinkTypes::EntriesByCorrelation)?, GetStrategy::default())?
    } else if let Some(agent) = &query.actor_agent {
        // Query by actor
        get_links(LinkQuery::try_new(agent.clone(), LinkTypes::EntriesByActor)?, GetStrategy::default())?
    } else if let Some(categories) = &query.categories {
        // Query by first category
        if let Some(category) = categories.first() {
            let anchor = category_anchor(category)?;
            get_links(LinkQuery::try_new(anchor, LinkTypes::EntriesByCategory)?, GetStrategy::default())?
        } else {
            get_all_entry_links()?
        }
    } else if let Some(severities) = &query.severities {
        // Query by first severity
        if let Some(severity) = severities.first() {
            let anchor = severity_anchor(severity)?;
            get_links(LinkQuery::try_new(anchor, LinkTypes::EntriesBySeverity)?, GetStrategy::default())?
        } else {
            get_all_entry_links()?
        }
    } else {
        get_all_entry_links()?
    };

    let mut entries = Vec::new();
    let _now = sys_time()?.as_micros() as u64;

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(entry) = record
                    .entry()
                    .to_app_option::<AuditEntry>()
                    .map_err(|e| wasm_error!(e))?
                {
                    // Apply filters
                    if let Some(start) = query.start_date {
                        if entry.timestamp < start {
                            continue;
                        }
                    }

                    if let Some(end) = query.end_date {
                        if entry.timestamp > end {
                            continue;
                        }
                    }

                    if let Some(ref categories) = query.categories {
                        if !categories.contains(&entry.category) {
                            continue;
                        }
                    }

                    if let Some(ref severities) = query.severities {
                        if !severities.contains(&entry.severity) {
                            continue;
                        }
                    }

                    entries.push(entry);
                }
            }
        }
    }

    // Sort by timestamp descending
    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    // Apply offset and limit
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(100);

    Ok(entries.into_iter().skip(offset).take(limit).collect())
}

/// Get entry by ID
#[hdk_extern]
pub fn get_entry_by_id(id: String) -> ExternResult<Option<AuditEntry>> {
    let links = get_all_entry_links()?;

    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(entry) = record
                    .entry()
                    .to_app_option::<AuditEntry>()
                    .map_err(|e| wasm_error!(e))?
                {
                    if entry.id == id {
                        return Ok(Some(entry));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get entries by correlation ID
#[hdk_extern]
pub fn get_correlated_entries(correlation_id: String) -> ExternResult<Vec<AuditEntry>> {
    let anchor = correlation_anchor(&correlation_id)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::EntriesByCorrelation)?, GetStrategy::default())?;

    let mut entries = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(entry) = record
                    .entry()
                    .to_app_option::<AuditEntry>()
                    .map_err(|e| wasm_error!(e))?
                {
                    entries.push(entry);
                }
            }
        }
    }

    entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    Ok(entries)
}

// ==================== REPORTING ====================

/// Generate audit summary for a period
#[hdk_extern]
pub fn generate_summary(input: SummaryInput) -> ExternResult<AuditSummary> {
    let entries = get_entries(Some(AuditQuery {
        start_date: Some(input.period_start),
        end_date: Some(input.period_end),
        categories: None,
        severities: None,
        actor_agent: None,
        correlation_id: None,
        limit: None,
        offset: None,
    }))?;

    let mut by_category: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut by_severity: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut actors: std::collections::HashSet<String> = std::collections::HashSet::new();

    for entry in &entries {
        *by_category.entry(format!("{:?}", entry.category)).or_insert(0) += 1;
        *by_severity.entry(format!("{:?}", entry.severity)).or_insert(0) += 1;

        if let Some(ref email) = entry.actor.email {
            actors.insert(email.clone());
        }
        if let Some(ref agent) = entry.actor.agent_pub_key {
            actors.insert(format!("{:?}", agent));
        }
    }

    let summary = AuditSummary {
        period_start: input.period_start,
        period_end: input.period_end,
        total_entries: entries.len() as u32,
        by_category: by_category.into_iter().collect(),
        by_severity: by_severity.into_iter().collect(),
        unique_actors: actors.len() as u32,
        generated_at: sys_time()?.as_micros() as u64,
        hash: format!("summary_{}", entries.len()),
    };

    // Store the summary
    let action_hash = hdk::entry::create_entry(EntryTypes::AuditSummary(summary.clone()))?;

    let anchor = anchor_hash(SUMMARIES_ANCHOR)?;
    create_link(
        anchor,
        action_hash,
        LinkTypes::AuditSummaries,
        summary.generated_at.to_be_bytes().to_vec(),
    )?;

    Ok(summary)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SummaryInput {
    pub period_start: u64,
    pub period_end: u64,
}

/// Get all summaries
#[hdk_extern]
pub fn get_summaries(_: ()) -> ExternResult<Vec<AuditSummary>> {
    let anchor = anchor_hash(SUMMARIES_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AuditSummaries)?, GetStrategy::default())?;

    let mut summaries = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(summary) = record
                    .entry()
                    .to_app_option::<AuditSummary>()
                    .map_err(|e| wasm_error!(e))?
                {
                    summaries.push(summary);
                }
            }
        }
    }

    summaries.sort_by(|a, b| b.generated_at.cmp(&a.generated_at));

    Ok(summaries)
}

/// Get entry count
#[hdk_extern]
pub fn get_entry_count(_: ()) -> ExternResult<u32> {
    let links = get_all_entry_links()?;
    Ok(links.len() as u32)
}

// ==================== HELPERS ====================

fn anchor_hash(name: &str) -> ExternResult<EntryHash> {
    let path = Path::from(name);
    path.path_entry_hash()
}

fn category_anchor(category: &AuditCategory) -> ExternResult<EntryHash> {
    let path = Path::from(format!("audit_category:{:?}", category));
    path.path_entry_hash()
}

fn severity_anchor(severity: &AuditSeverity) -> ExternResult<EntryHash> {
    let path = Path::from(format!("audit_severity:{:?}", severity));
    path.path_entry_hash()
}

fn correlation_anchor(id: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("audit_correlation:{}", id));
    path.path_entry_hash()
}

fn get_all_entry_links() -> ExternResult<Vec<Link>> {
    let anchor = anchor_hash(ALL_ENTRIES_ANCHOR)?;
    get_links(LinkQuery::try_new(anchor, LinkTypes::AllAuditEntries)?, GetStrategy::default())
}
