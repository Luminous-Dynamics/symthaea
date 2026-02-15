//! LUCID Temporal Coordinator Zome
//!
//! Belief versioning and time-travel queries.

use hdk::prelude::*;
use temporal_integrity::*;

fn anchor_hash(s: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(s.to_string())))
}

fn create_anchor(s: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(s.to_string())))?;
    anchor_hash(s)
}

/// Record a new version of a thought
#[hdk_extern]
pub fn record_version(input: RecordVersionInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let previous = input.previous_version.clone();

    let version = BeliefVersion {
        thought_id: input.thought_id.clone(),
        version: input.version,
        content: input.content,
        confidence: input.confidence,
        epistemic_code: input.epistemic_code,
        change_reason: input.change_reason,
        previous_version: input.previous_version,
        timestamp: now,
    };

    let action_hash = create_entry(&EntryTypes::BeliefVersion(version))?;

    // Link thought to this version
    let thought_anchor = format!("thought:{}", input.thought_id);
    create_anchor(&thought_anchor)?;
    create_link(
        anchor_hash(&thought_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtToVersions,
        (),
    )?;

    // Chain to previous version if exists
    if let Some(prev) = previous {
        create_link(action_hash.clone(), prev, LinkTypes::VersionChain, ())?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Version not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordVersionInput {
    pub thought_id: String,
    pub version: u32,
    pub content: String,
    pub confidence: f64,
    pub epistemic_code: String,
    pub change_reason: Option<String>,
    pub previous_version: Option<ActionHash>,
}

/// Get all versions of a thought
#[hdk_extern]
pub fn get_thought_history(thought_id: String) -> ExternResult<Vec<Record>> {
    let thought_anchor = format!("thought:{}", thought_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToVersions)?,
        GetStrategy::default(),
    )?;

    let mut versions = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            versions.push(record);
        }
    }

    // Sort by version number
    versions.sort_by(|a, b| {
        let va = a.entry().to_app_option::<BeliefVersion>().ok().flatten().map(|v| v.version).unwrap_or(0);
        let vb = b.entry().to_app_option::<BeliefVersion>().ok().flatten().map(|v| v.version).unwrap_or(0);
        va.cmp(&vb)
    });

    Ok(versions)
}

/// Get belief at a specific timestamp
#[hdk_extern]
pub fn get_belief_at_time(input: BeliefAtTimeInput) -> ExternResult<Option<Record>> {
    let history = get_thought_history(input.thought_id)?;

    // Find the version that was current at the given timestamp
    let mut best: Option<Record> = None;
    for record in history {
        if let Some(version) = record.entry().to_app_option::<BeliefVersion>().ok().flatten() {
            if version.timestamp <= input.timestamp {
                best = Some(record);
            } else {
                break;
            }
        }
    }

    Ok(best)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BeliefAtTimeInput {
    pub thought_id: String,
    pub timestamp: Timestamp,
}

/// Create a snapshot of the current knowledge graph
#[hdk_extern]
pub fn create_snapshot(input: CreateSnapshotInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let snapshot = GraphSnapshot {
        id: format!("snapshot_{}", now.as_micros()),
        name: input.name,
        description: input.description,
        timestamp: now,
        thought_count: input.thought_count,
        avg_confidence: input.avg_confidence,
        tags: input.tags.unwrap_or_default(),
    };

    let action_hash = create_entry(&EntryTypes::GraphSnapshot(snapshot))?;

    let agent_anchor = format!("agent_snapshots:{}", agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToSnapshots,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Snapshot not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateSnapshotInput {
    pub name: String,
    pub description: Option<String>,
    pub thought_count: u32,
    pub avg_confidence: f64,
    pub tags: Option<Vec<String>>,
}

/// Get all snapshots
#[hdk_extern]
pub fn get_my_snapshots(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = format!("agent_snapshots:{}", agent);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&agent_anchor)?, LinkTypes::AgentToSnapshots)?,
        GetStrategy::default(),
    )?;

    let mut snapshots = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            snapshots.push(record);
        }
    }

    Ok(snapshots)
}
