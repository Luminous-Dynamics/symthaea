use hdk::prelude::*;
use resonance_feed_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic,
    GovernanceEligibility, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("civic_bridge", requirement, action_name)
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))
}

/// Helper to ensure an anchor entry exists and return its hash
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))?;
    anchor_hash(anchor_str)
}

/// Publish content to the resonance feed (Participant+).
#[hdk_extern]
pub fn publish_content(entry: ContentEntry) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "publish_content")?;

    let action_hash = create_entry(&EntryTypes::ContentEntry(entry.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Link from all content
    let all_anchor = ensure_anchor("all_content")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllContent, ())?;

    // Link from domain
    let domain_anchor = ensure_anchor(&format!("domain/{}", entry.domain))?;
    create_link(domain_anchor, action_hash.clone(), LinkTypes::DomainContent, ())?;

    // Link from agent
    create_link(agent, action_hash.clone(), LinkTypes::AgentToContent, ())?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

/// Vote resonance on content (Observer+, lowest bar).
#[hdk_extern]
pub fn vote_resonance(vote: ResonanceVote) -> ExternResult<Record> {
    // Observer+ can vote — no consciousness gate needed (lowest bar)
    let action_hash = create_entry(&EntryTypes::ResonanceVote(vote.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;

    create_link(vote.content_hash, action_hash.clone(), LinkTypes::ContentToVotes, ())?;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToVotes, ())?;

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;
    Ok(record)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FeedInput {
    pub limit: usize,
    pub domain_filter: Option<String>,
}

/// Get resonance-ranked feed.
#[hdk_extern]
pub fn get_feed(input: FeedInput) -> ExternResult<Vec<ContentEntry>> {
    let base = if let Some(ref domain) = input.domain_filter {
        anchor_hash(&format!("domain/{}", domain))?
    } else {
        anchor_hash("all_content")?
    };

    let link_type = if input.domain_filter.is_some() {
        LinkTypes::DomainContent
    } else {
        LinkTypes::AllContent
    };

    let links = get_links(
        LinkQuery::try_new(base, link_type)?,
        GetStrategy::default(),
    )?;

    let mut entries = Vec::new();
    for link in links.into_iter().rev().take(input.limit.min(100)) {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(entry) = record.entry().to_app_option::<ContentEntry>().ok().flatten() {
                    entries.push(entry);
                }
            }
        }
    }
    Ok(entries)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrendingInput {
    pub domain: String,
    pub limit: usize,
}

/// Get trending content by resonance votes.
#[hdk_extern]
pub fn get_trending(input: TrendingInput) -> ExternResult<Vec<(ContentEntry, f64)>> {
    let feed = get_feed(FeedInput {
        limit: input.limit * 3,
        domain_filter: Some(input.domain),
    })?;

    // For each content, sum resonance votes
    // (simplified — in production would aggregate vote records)
    let mut scored: Vec<(ContentEntry, f64)> = feed.into_iter()
        .map(|entry| (entry, 0.5)) // Default score
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(input.limit);
    Ok(scored)
}
