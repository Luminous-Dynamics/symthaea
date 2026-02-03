//! Agent Reputation Coordinator Zome
//!
//! Manages AI agent profiles, reputation events, and MATL trust score computation.
//! This provides the Byzantine-resistant trust layer for Symthaea civic AI agents.
//!
//! ## Key Functions
//!
//! - `register_agent`: Create an agent profile
//! - `record_event`: Record a reputation event (feedback)
//! - `compute_trust_score`: Calculate MATL trust score
//! - `get_trustworthy_agents`: Find agents meeting trust threshold

use hdk::prelude::*;
use agent_reputation_integrity::*;

/// Helper to create or get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Ensure an anchor exists and return its hash
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor.clone()))?;
    anchor_hash(anchor_str)
}

/// Input for registering an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterAgentInput {
    pub agent_type: AgentType,
    pub name: String,
    pub specializations: Vec<AgentSpecialization>,
    pub description: String,
    pub config_metadata: Option<String>,
}

/// Register a new agent profile
#[hdk_extern]
pub fn register_agent(input: RegisterAgentInput) -> ExternResult<ActionHash> {
    let now = sys_time()?.as_micros() as u64;
    let my_pubkey = agent_info()?.agent_initial_pubkey;

    let profile = AgentProfile {
        agent_type: input.agent_type,
        name: input.name,
        specializations: input.specializations.clone(),
        description: input.description,
        is_active: true,
        created_at: now,
        last_active: now,
        config_metadata: input.config_metadata,
    };

    let action_hash = create_entry(&EntryTypes::AgentProfile(profile))?;

    // Link from agent pubkey to profile
    create_link(
        my_pubkey.clone(),
        action_hash.clone(),
        AgentReputationLinkTypes::AgentToProfile,
        (),
    )?;

    // Link to active agents anchor
    let active_anchor = ensure_anchor("active_agents")?;
    create_link(
        active_anchor,
        my_pubkey.clone(),
        AgentReputationLinkTypes::ActiveAgents,
        (),
    )?;

    // Link to specializations
    for spec in &input.specializations {
        let spec_anchor = format!("spec:{:?}", spec).to_lowercase();
        let spec_hash = ensure_anchor(&spec_anchor)?;
        create_link(
            spec_hash,
            my_pubkey.clone(),
            AgentReputationLinkTypes::SpecializationToAgent,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Get an agent's profile
#[hdk_extern]
pub fn get_agent_profile(agent_pubkey: AgentPubKey) -> ExternResult<Option<AgentProfile>> {
    let links = get_links(
        LinkQuery::try_new(
            agent_pubkey,
            AgentReputationLinkTypes::AgentToProfile,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            let profile: AgentProfile = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Not an AgentProfile".into())))?;
            return Ok(Some(profile));
        }
    }

    Ok(None)
}

/// Input for recording a reputation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordEventInput {
    pub agent_pubkey: AgentPubKey,
    pub event_type: ReputationEventType,
    pub context: Option<String>,
    pub conversation_id: Option<String>,
    pub weight: Option<f32>,
}

/// Record a reputation event for an agent
#[hdk_extern]
pub fn record_event(input: RecordEventInput) -> ExternResult<ActionHash> {
    let event = ReputationEvent {
        agent_pubkey: input.agent_pubkey.clone(),
        event_type: input.event_type,
        context: input.context,
        conversation_id: input.conversation_id,
        timestamp: sys_time()?.as_micros() as u64,
        weight: input.weight.unwrap_or(1.0),
    };

    let action_hash = create_entry(&EntryTypes::ReputationEvent(event))?;

    // Link from agent to event
    create_link(
        input.agent_pubkey,
        action_hash.clone(),
        AgentReputationLinkTypes::AgentToReputationEvent,
        (),
    )?;

    Ok(action_hash)
}

/// Get all reputation events for an agent
#[hdk_extern]
pub fn get_agent_events(agent_pubkey: AgentPubKey) -> ExternResult<Vec<ReputationEvent>> {
    let links = get_links(
        LinkQuery::try_new(
            agent_pubkey,
            AgentReputationLinkTypes::AgentToReputationEvent,
        )?,
        GetStrategy::default(),
    )?;

    let mut events = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(event) = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                events.push(event);
            }
        }
    }

    // Sort by timestamp (newest first)
    events.sort_by(|a: &ReputationEvent, b: &ReputationEvent| b.timestamp.cmp(&a.timestamp));

    Ok(events)
}

/// Compute and store a trust score for an agent
#[hdk_extern]
pub fn compute_trust_score(agent_pubkey: AgentPubKey) -> ExternResult<TrustScore> {
    let events = get_agent_events(agent_pubkey.clone())?;

    let mut positive_count: u32 = 0;
    let mut negative_count: u32 = 0;
    let mut quality_sum: f32 = 0.0;
    let mut consistency_sum: f32 = 0.0;
    let mut reputation_sum: f32 = 0.0;

    // Process events
    for event in &events {
        let impact = event.event_type.impact_value() * event.weight;

        if event.event_type.is_positive() {
            positive_count += 1;
        } else {
            negative_count += 1;
        }

        // Map event types to score components
        match event.event_type {
            ReputationEventType::Accurate | ReputationEventType::Inaccurate => {
                quality_sum += impact;
            }
            ReputationEventType::FastResponse
            | ReputationEventType::SlowResponse
            | ReputationEventType::ProperEscalation
            | ReputationEventType::FailedEscalation => {
                consistency_sum += impact;
            }
            ReputationEventType::Helpful
            | ReputationEventType::NotHelpful
            | ReputationEventType::AuthorityApproved
            | ReputationEventType::AuthorityRejected => {
                reputation_sum += impact;
            }
        }
    }

    let total_events = positive_count + negative_count;

    // Compute normalized scores (start at 0.5 baseline)
    let quality = if total_events > 0 {
        (0.5 + quality_sum / total_events as f32).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let consistency = if total_events > 0 {
        (0.5 + consistency_sum / total_events as f32).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let reputation = if total_events > 0 {
        (0.5 + reputation_sum / total_events as f32).clamp(0.0, 1.0)
    } else {
        0.5
    };

    // Compute composite MATL score
    let composite = TrustScore::compute_composite(quality, consistency, reputation);

    // Compute confidence based on event count
    let confidence = (total_events as f32 / 100.0).min(1.0);

    let score = TrustScore {
        agent_pubkey: agent_pubkey.clone(),
        quality,
        consistency,
        reputation,
        composite,
        positive_count,
        negative_count,
        computed_at: sys_time()?.as_micros() as u64,
        confidence,
    };

    // Store the score
    let action_hash = create_entry(&EntryTypes::TrustScore(score.clone()))?;

    // Link from agent to score
    create_link(
        agent_pubkey,
        action_hash,
        AgentReputationLinkTypes::AgentToTrustScore,
        (),
    )?;

    Ok(score)
}

/// Get the latest trust score for an agent
#[hdk_extern]
pub fn get_trust_score(agent_pubkey: AgentPubKey) -> ExternResult<Option<TrustScore>> {
    let links = get_links(
        LinkQuery::try_new(
            agent_pubkey,
            AgentReputationLinkTypes::AgentToTrustScore,
        )?,
        GetStrategy::default(),
    )?;

    // Get the most recent score
    let mut latest: Option<TrustScore> = None;
    let mut latest_time: u64 = 0;

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(score) = record
                .entry()
                .to_app_option::<TrustScore>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if score.computed_at > latest_time {
                    latest_time = score.computed_at;
                    latest = Some(score);
                }
            }
        }
    }

    Ok(latest)
}

/// Agent with their trust score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentWithScore {
    pub pubkey: AgentPubKey,
    pub profile: AgentProfile,
    pub trust_score: Option<TrustScore>,
}

/// Get all active agents with their trust scores
#[hdk_extern]
pub fn get_active_agents(_: ()) -> ExternResult<Vec<AgentWithScore>> {
    let active_anchor = anchor_hash("active_agents")?;
    let links = get_links(
        LinkQuery::try_new(
            active_anchor,
            AgentReputationLinkTypes::ActiveAgents,
        )?,
        GetStrategy::default(),
    )?;

    let mut agents = Vec::new();
    for link in links {
        let agent_pubkey = AgentPubKey::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(profile) = get_agent_profile(agent_pubkey.clone())? {
            if profile.is_active {
                let trust_score = get_trust_score(agent_pubkey.clone())?;
                agents.push(AgentWithScore {
                    pubkey: agent_pubkey,
                    profile,
                    trust_score,
                });
            }
        }
    }

    Ok(agents)
}

/// Get agents that meet the MATL trust threshold
#[hdk_extern]
pub fn get_trustworthy_agents(min_composite: Option<f32>) -> ExternResult<Vec<AgentWithScore>> {
    let threshold = min_composite.unwrap_or(0.55); // MATL default threshold

    let all_agents = get_active_agents(())?;

    let trustworthy = all_agents
        .into_iter()
        .filter(|a| {
            a.trust_score
                .as_ref()
                .map(|s| s.composite >= threshold && s.confidence >= 0.3)
                .unwrap_or(false)
        })
        .collect();

    Ok(trustworthy)
}

/// Get agents by specialization
#[hdk_extern]
pub fn get_agents_by_specialization(spec: AgentSpecialization) -> ExternResult<Vec<AgentWithScore>> {
    let spec_anchor = format!("spec:{:?}", spec).to_lowercase();
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&spec_anchor)?,
            AgentReputationLinkTypes::SpecializationToAgent,
        )?,
        GetStrategy::default(),
    )?;

    let mut agents = Vec::new();
    for link in links {
        let agent_pubkey = AgentPubKey::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(profile) = get_agent_profile(agent_pubkey.clone())? {
            if profile.is_active {
                let trust_score = get_trust_score(agent_pubkey.clone())?;
                agents.push(AgentWithScore {
                    pubkey: agent_pubkey,
                    profile,
                    trust_score,
                });
            }
        }
    }

    Ok(agents)
}

/// Get my own profile
#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<AgentProfile>> {
    let my_pubkey = agent_info()?.agent_initial_pubkey;
    get_agent_profile(my_pubkey)
}

/// Get my own trust score (compute if needed)
#[hdk_extern]
pub fn get_my_trust_score(_: ()) -> ExternResult<TrustScore> {
    let my_pubkey = agent_info()?.agent_initial_pubkey;
    compute_trust_score(my_pubkey)
}
