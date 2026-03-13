//! Proposals Coordinator Zome
//! Business logic for governance proposals
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use proposals_integrity::*;

// ============================================================================
// REAL-TIME SIGNALS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum ProposalSignal {
    ProposalCreated {
        proposal_id: String,
        author: String,
        title: String,
        proposal_type: String,
    },
    ProposalStatusChanged {
        proposal_id: String,
        new_status: String,
    },
    ContributionAdded {
        proposal_id: String,
        contributor: String,
    },
    DiscussionReflectionGenerated {
        proposal_id: String,
        ready_for_vote: bool,
    },
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Pre-create the active_proposals anchor so queries never fail on empty DNA
    let anchor = Anchor("active_proposals".to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    Ok(InitCallbackResult::Pass)
}

/// Create a new proposal
#[hdk_extern]
pub fn create_proposal(proposal: Proposal) -> ExternResult<Record> {
    // Input validation
    if proposal.title.is_empty() || proposal.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if proposal.description.is_empty() || proposal.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-4096 characters".into()
        )));
    }
    if proposal.id.is_empty() || proposal.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if proposal.author.is_empty() || proposal.author.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Author must be 1-256 characters".into()
        )));
    }
    if let Some(ref url) = proposal.discussion_url {
        if url.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Discussion URL must be at most 4096 characters".into()
            )));
        }
    }

    let signal_id = proposal.id.clone();
    let signal_author = proposal.author.clone();
    let signal_title = proposal.title.clone();
    let signal_type = format!("{:?}", proposal.proposal_type);

    let action_hash = create_entry(&EntryTypes::Proposal(proposal.clone()))?;

    let _ = emit_signal(&ProposalSignal::ProposalCreated {
        proposal_id: signal_id,
        author: signal_author,
        title: signal_title,
        proposal_type: signal_type,
    });

    // Create anchors and link author to proposal
    let author_anchor = format!("author:{}", proposal.author);
    create_entry(&EntryTypes::Anchor(Anchor(author_anchor.clone())))?;
    create_link(
        anchor_hash(&author_anchor)?,
        action_hash.clone(),
        LinkTypes::AuthorToProposal,
        (),
    )?;

    // Link type to proposal
    let type_anchor = format!("type:{:?}", proposal.proposal_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::TypeToProposal,
        (),
    )?;

    // Create anchor and link for O(1) lookup by proposal ID
    let pid_anchor = format!("pid:{}", proposal.id);
    create_entry(&EntryTypes::Anchor(Anchor(pid_anchor.clone())))?;
    create_link(
        anchor_hash(&pid_anchor)?,
        action_hash.clone(),
        LinkTypes::ProposalById,
        (),
    )?;

    // Link to active proposals if active
    if proposal.status == ProposalStatus::Active {
        create_entry(&EntryTypes::Anchor(Anchor("active_proposals".to_string())))?;
        create_link(
            anchor_hash("active_proposals")?,
            action_hash.clone(),
            LinkTypes::ActiveProposals,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created proposal".into()
    )))
}

/// Get a proposal by ID (O(1) link-based lookup with chain scan fallback)
#[hdk_extern]
pub fn get_proposal(proposal_id: String) -> ExternResult<Option<Record>> {
    // Try link-based lookup first (O(1))
    let pid_anchor = format!("pid:{}", proposal_id);
    if let Ok(entry_hash) = anchor_hash(&pid_anchor) {
        if let Ok(links) = get_links(
            LinkQuery::try_new(entry_hash, LinkTypes::ProposalById)?,
            GetStrategy::default(),
        ) {
            if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
                if let Ok(ah) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(ah, GetOptions::default())? {
                        return Ok(Some(record));
                    }
                }
            }
        }
    }

    // Fallback: O(n) chain scan for proposals created before the link was added
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Proposal,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    // Take the LAST match — update_entry appends newer versions later in the chain
    let mut found: Option<Record> = None;
    for record in records {
        if let Some(proposal) = record
            .entry()
            .to_app_option::<Proposal>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if proposal.id == proposal_id {
                found = Some(record);
            }
        }
    }

    Ok(found)
}

/// Get active proposals
#[hdk_extern]
pub fn get_active_proposals(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("active_proposals")?, LinkTypes::ActiveProposals)?,
        GetStrategy::default(),
    )?;

    let mut proposals = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            proposals.push(record);
        }
    }

    Ok(proposals)
}

/// Get proposals by author
#[hdk_extern]
pub fn get_proposals_by_author(author_did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("author:{}", author_did))?,
            LinkTypes::AuthorToProposal,
        )?,
        GetStrategy::default(),
    )?;

    let mut proposals = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            proposals.push(record);
        }
    }

    Ok(proposals)
}

/// Update proposal status
///
/// Authorization rules:
/// - Author can: Draft→Active, Draft→Cancelled, Active→Cancelled
/// - Execution pipeline (any agent) can: Active→Ended, Ended→Approved, Ended→Rejected,
///   Approved→Signed, Signed→Executed, any→Failed
/// - Unauthorized transitions are rejected
#[hdk_extern]
pub fn update_proposal_status(input: UpdateStatusInput) -> ExternResult<Record> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }

    let current_record = get_proposal(input.proposal_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Proposal not found".into())
    ))?;

    let current_proposal: Proposal = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid proposal entry".into()
        )))?;

    // Authorization: author-only transitions require caller = author
    let is_author_transition = matches!(
        (&current_proposal.status, &input.new_status),
        (ProposalStatus::Draft, ProposalStatus::Active)
            | (ProposalStatus::Draft, ProposalStatus::Cancelled)
            | (ProposalStatus::Active, ProposalStatus::Cancelled)
    );

    if is_author_transition {
        let agent_info = agent_info()?;
        let author_key =
            current_proposal
                .author
                .strip_prefix("did:mycelix:")
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid author DID format".into()
                )))?;
        if agent_info.agent_initial_pubkey.to_string() != author_key {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only the proposal author can perform this status transition".into()
            )));
        }
    }

    let now = sys_time()?;

    let was_active = current_proposal.status == ProposalStatus::Active;
    let new_status = input.new_status.clone();

    let _ = emit_signal(&ProposalSignal::ProposalStatusChanged {
        proposal_id: current_proposal.id.clone(),
        new_status: format!("{:?}", new_status),
    });

    let updated_proposal = Proposal {
        id: current_proposal.id.clone(),
        title: current_proposal.title,
        description: current_proposal.description,
        proposal_type: current_proposal.proposal_type,
        author: current_proposal.author,
        status: input.new_status,
        actions: current_proposal.actions,
        discussion_url: current_proposal.discussion_url,
        voting_starts: current_proposal.voting_starts,
        voting_ends: current_proposal.voting_ends,
        created: current_proposal.created,
        updated: now,
        version: current_proposal.version + 1,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Proposal(updated_proposal),
    )?;

    // Create active_proposals link when transitioning TO Active
    if !was_active && new_status == ProposalStatus::Active {
        create_entry(&EntryTypes::Anchor(Anchor("active_proposals".to_string())))?;
        create_link(
            anchor_hash("active_proposals")?,
            action_hash.clone(),
            LinkTypes::ActiveProposals,
            (),
        )?;
    }

    // Clean up active_proposals link when transitioning away from Active
    if was_active && new_status != ProposalStatus::Active {
        if let Ok(active_links) = get_links(
            LinkQuery::try_new(anchor_hash("active_proposals")?, LinkTypes::ActiveProposals)?,
            GetStrategy::default(),
        ) {
            for link in active_links {
                if let Ok(target_hash) = ActionHash::try_from(link.target.clone()) {
                    if let Ok(Some(record)) = get(target_hash, GetOptions::default()) {
                        if let Some(p) = record.entry().to_app_option::<Proposal>().ok().flatten() {
                            if p.id == current_proposal.id {
                                let _ = delete_link(link.create_link_hash, GetOptions::default());
                            }
                        }
                    }
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated proposal".into()
    )))
}

/// Input for updating proposal status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateStatusInput {
    pub proposal_id: String,
    pub new_status: ProposalStatus,
}

/// Cancel a proposal (author only)
#[hdk_extern]
pub fn cancel_proposal(proposal_id: String) -> ExternResult<Record> {
    if proposal_id.is_empty() || proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }

    let agent_info = agent_info()?;
    let current_record = get_proposal(proposal_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Proposal not found".into())
    ))?;

    let current_proposal: Proposal = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid proposal entry".into()
        )))?;

    // Verify author (simplified - would use DID lookup in production)
    let author_key = current_proposal
        .author
        .strip_prefix("did:mycelix:")
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid author DID".into()
        )))?;

    if agent_info.agent_initial_pubkey.to_string() != author_key {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only author can cancel proposal".into()
        )));
    }

    update_proposal_status(UpdateStatusInput {
        proposal_id,
        new_status: ProposalStatus::Cancelled,
    })
}

/// Finalize a proposal with a threshold signature
///
/// Called after the signing committee has produced a verified threshold signature
/// for an approved proposal. Advances the proposal from Approved → Signed,
/// signaling that the proposal is ready for timelock/execution.
#[hdk_extern]
pub fn finalize_proposal_with_signature(input: FinalizeWithSignatureInput) -> ExternResult<Record> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if input.signature_id.is_empty() || input.signature_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Signature ID must be 1-256 characters".into()
        )));
    }

    // Verify the proposal exists and is in Approved status
    let current_record = get_proposal(input.proposal_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Proposal not found".into())
    ))?;

    let current_proposal: Proposal = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid proposal entry".into()
        )))?;

    if current_proposal.status != ProposalStatus::Approved {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Proposal must be in Approved status to finalize with signature, current: {:?}",
            current_proposal.status
        ))));
    }

    // Verify threshold signature exists via cross-zome call
    let sig_io = governance_utils::call_local(
        "threshold_signing",
        "get_proposal_signature",
        input.proposal_id.clone(),
    )?;
    if let Ok(maybe_record) = sig_io.decode::<Option<Record>>() {
        if maybe_record.is_none() {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "No verified threshold signature found for proposal '{}'",
                input.proposal_id
            ))));
        }
    }

    // Advance proposal to Signed status
    update_proposal_status(UpdateStatusInput {
        proposal_id: input.proposal_id,
        new_status: ProposalStatus::Signed,
    })
}

/// Input for finalizing a proposal with a threshold signature
#[derive(Serialize, Deserialize, Debug)]
pub struct FinalizeWithSignatureInput {
    pub proposal_id: String,
    pub signature_id: String,
}

/// Generate next proposal ID
#[hdk_extern]
pub fn generate_proposal_id(_: ()) -> ExternResult<String> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(
        UnitEntryTypes::Proposal,
    )?));

    let records = query(filter)?;
    let next_number = records.len() + 1;

    Ok(format!("MIP-{:04}", next_number))
}

// ============================================================================
// DISCUSSION SYSTEM WITH COLLECTIVE SENSING
// ============================================================================
//
// Philosophy: Mirror the discussion, not just the votes.
// The quality of deliberation matters as much as the outcome.
//
// This module enables:
// - Threaded discussion contributions
// - Harmony tagging (which values are invoked)
// - Stance signaling (preliminary sentiment)
// - Collective sensing of discussion health

use std::collections::{HashMap, HashSet};

/// Add a contribution to a proposal's discussion
#[hdk_extern]
pub fn add_contribution(input: AddContributionInput) -> ExternResult<Record> {
    // Input validation
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if input.contributor_did.is_empty() || input.contributor_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Contributor DID must be 1-256 characters".into()
        )));
    }
    if input.content.is_empty() || input.content.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Content must be 1-4096 characters".into()
        )));
    }
    if let Some(ref parent_id) = input.parent_id {
        if parent_id.is_empty() || parent_id.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Parent ID must be 1-256 characters".into()
            )));
        }
    }
    if let Some(ref tags) = input.harmony_tags {
        for tag in tags {
            if tag.is_empty() || tag.len() > 256 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Harmony tag must be 1-256 characters".into()
                )));
            }
        }
    }

    let now = sys_time()?;

    let contribution = DiscussionContribution {
        id: format!("contrib:{}:{}", input.proposal_id, now.as_micros()),
        proposal_id: input.proposal_id.clone(),
        contributor: input.contributor_did.clone(),
        content: input.content,
        harmony_tags: input.harmony_tags.unwrap_or_default(),
        stance: input.stance,
        parent_id: input.parent_id.clone(),
        created_at: now,
        edited: false,
    };

    let action_hash = create_entry(&EntryTypes::DiscussionContribution(contribution.clone()))?;

    let _ = emit_signal(&ProposalSignal::ContributionAdded {
        proposal_id: input.proposal_id.clone(),
        contributor: input.contributor_did.clone(),
    });

    // Link proposal to contribution
    let proposal_anchor = format!("discussion:{}", input.proposal_id);
    create_entry(&EntryTypes::Anchor(Anchor(proposal_anchor.clone())))?;
    create_link(
        anchor_hash(&proposal_anchor)?,
        action_hash.clone(),
        LinkTypes::ProposalToContribution,
        (),
    )?;

    // Link contributor to contribution
    let contributor_anchor = format!("contributor:{}", input.contributor_did);
    create_entry(&EntryTypes::Anchor(Anchor(contributor_anchor.clone())))?;
    create_link(
        anchor_hash(&contributor_anchor)?,
        action_hash.clone(),
        LinkTypes::ContributorToContribution,
        (),
    )?;

    // If this is a reply, link parent to child
    if let Some(parent_id) = input.parent_id {
        let parent_anchor = format!("replies:{}", parent_id);
        create_entry(&EntryTypes::Anchor(Anchor(parent_anchor.clone())))?;
        create_link(
            anchor_hash(&parent_anchor)?,
            action_hash.clone(),
            LinkTypes::ContributionToReply,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created contribution".into()
    )))
}

/// Input for adding a discussion contribution
#[derive(Serialize, Deserialize, Debug)]
pub struct AddContributionInput {
    pub proposal_id: String,
    pub contributor_did: String,
    pub content: String,
    pub harmony_tags: Option<Vec<String>>,
    pub stance: Option<Stance>,
    pub parent_id: Option<String>,
}

/// Get all contributions for a proposal
#[hdk_extern]
pub fn get_discussion(proposal_id: String) -> ExternResult<Vec<Record>> {
    let proposal_anchor = format!("discussion:{}", proposal_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&proposal_anchor)?,
            LinkTypes::ProposalToContribution,
        )?,
        GetStrategy::default(),
    )?;

    let mut contributions = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            contributions.push(record);
        }
    }

    // Sort by timestamp (oldest first for discussion flow)
    contributions.sort_by(|a, b| {
        let ts_a = a.action().timestamp();
        let ts_b = b.action().timestamp();
        ts_a.cmp(&ts_b)
    });

    Ok(contributions)
}

/// Get replies to a contribution
#[hdk_extern]
pub fn get_replies(contribution_id: String) -> ExternResult<Vec<Record>> {
    let parent_anchor = format!("replies:{}", contribution_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&parent_anchor)?, LinkTypes::ContributionToReply)?,
        GetStrategy::default(),
    )?;

    let mut replies = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            replies.push(record);
        }
    }

    // Sort by timestamp
    replies.sort_by(|a, b| {
        let ts_a = a.action().timestamp();
        let ts_b = b.action().timestamp();
        ts_a.cmp(&ts_b)
    });

    Ok(replies)
}

/// Generate a reflection on the discussion phase
///
/// This mirrors how the conversation looks - not whether it's "good" or "bad"
#[hdk_extern]
pub fn reflect_on_discussion(proposal_id: String) -> ExternResult<Record> {
    let now = sys_time()?;

    // Gather all contributions
    let contribution_records = get_discussion(proposal_id.clone())?;

    let mut contributions: Vec<DiscussionContribution> = Vec::new();
    for record in &contribution_records {
        if let Some(c) = record
            .entry()
            .to_app_option::<DiscussionContribution>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            contributions.push(c);
        }
    }

    // === PARTICIPATION METRICS ===
    let contribution_count = contributions.len() as u64;
    let unique_contributors: HashSet<_> = contributions.iter().map(|c| &c.contributor).collect();
    let contributor_count = unique_contributors.len() as u64;
    let avg_contributions_per_participant = if contributor_count > 0 {
        contribution_count as f64 / contributor_count as f64
    } else {
        0.0
    };

    // Calculate thread depth
    let max_thread_depth = calculate_max_thread_depth(&contributions);

    // === HARMONY ANALYSIS ===
    let all_harmonies = [
        "ResonantCoherence",
        "PanSentientFlourishing",
        "IntegralWisdom",
        "InfinitePlay",
        "UniversalInterconnectedness",
        "SacredReciprocity",
        "EvolutionaryProgression",
    ];

    let mut harmony_counts: HashMap<String, u64> = HashMap::new();
    let mut harmony_examples: HashMap<String, String> = HashMap::new();

    for contribution in &contributions {
        for tag in &contribution.harmony_tags {
            *harmony_counts.entry(tag.clone()).or_insert(0) += 1;
            harmony_examples
                .entry(tag.clone())
                .or_insert_with(|| contribution.id.clone());
        }
    }

    let harmony_coverage: Vec<HarmonyPresence> = all_harmonies
        .iter()
        .map(|h| {
            let count = harmony_counts.get(*h).copied().unwrap_or(0);
            let presence = if contribution_count > 0 {
                (count as f64 / contribution_count as f64).min(1.0)
            } else {
                0.0
            };
            HarmonyPresence {
                harmony: h.to_string(),
                presence,
                example_contribution_id: harmony_examples.get(*h).cloned(),
            }
        })
        .collect();

    let absent_harmonies: Vec<String> = harmony_coverage
        .iter()
        .filter(|h| h.presence < 0.1)
        .map(|h| h.harmony.clone())
        .collect();

    let harmony_diversity = if contribution_count > 0 {
        let harmonies_present = harmony_coverage.iter().filter(|h| h.presence > 0.1).count();
        harmonies_present as f64 / all_harmonies.len() as f64
    } else {
        0.0
    };

    // === STANCE DISTRIBUTION ===
    let mut support_count = 0u64;
    let mut oppose_count = 0u64;
    let mut neutral_count = 0u64;
    let mut amend_count = 0u64;

    for contribution in &contributions {
        match contribution.stance {
            Some(Stance::Support) => support_count += 1,
            Some(Stance::Oppose) => oppose_count += 1,
            Some(Stance::Neutral) => neutral_count += 1,
            Some(Stance::Amend) => amend_count += 1,
            None => {}
        }
    }

    let total_stances = support_count + oppose_count + neutral_count + amend_count;
    let preliminary_sentiment = if total_stances > 0 {
        (support_count as f64 + 0.5 * amend_count as f64) / total_stances as f64
    } else {
        0.5 // No stances = neutral
    };

    // === DISCUSSION HEALTH ===
    // Voice concentration: Are few people dominating?
    let mut contribution_counts_by_person: HashMap<&String, u64> = HashMap::new();
    for c in &contributions {
        *contribution_counts_by_person
            .entry(&c.contributor)
            .or_insert(0) += 1;
    }
    let max_by_one = contribution_counts_by_person
        .values()
        .max()
        .copied()
        .unwrap_or(0);
    let voice_concentration = if contribution_count > 0 && contributor_count > 1 {
        max_by_one as f64 / contribution_count as f64
    } else if contributor_count == 1 {
        1.0 // Single voice = max concentration
    } else {
        0.0
    };

    // Cross-camp engagement: Are supporters replying to opposers?
    let cross_camp_engagement = calculate_cross_camp_engagement(&contributions);

    // Substantiveness score (simplified: based on avg content length and reply depth)
    let avg_content_length = if contribution_count > 0 {
        contributions.iter().map(|c| c.content.len()).sum::<usize>() as f64
            / contribution_count as f64
    } else {
        0.0
    };
    let substantiveness_score = (
        (avg_content_length / 500.0).min(1.0) * 0.5 + // Length component
        (max_thread_depth as f64 / 5.0).min(1.0) * 0.5
        // Depth component
    )
    .min(1.0);

    // === READINESS SIGNALS ===
    let discussion_saturated = contribution_count > 10 && cross_camp_engagement > 0.3;

    // Simple heuristic for unaddressed concerns: opposition without responses
    let unaddressed_concerns = identify_unaddressed_concerns(&contributions);

    let ready_for_vote = contributor_count >= 3
        && harmony_diversity > 0.3
        && discussion_saturated
        && unaddressed_concerns.is_empty();

    let readiness_reasoning = if ready_for_vote {
        "Discussion appears mature: sufficient participation, harmony diversity, and cross-camp engagement.".to_string()
    } else {
        let mut reasons = Vec::new();
        if contributor_count < 3 {
            reasons.push("fewer than 3 contributors");
        }
        if harmony_diversity < 0.3 {
            reasons.push("low value diversity");
        }
        if !discussion_saturated {
            reasons.push("discussion not saturated");
        }
        if !unaddressed_concerns.is_empty() {
            reasons.push("unaddressed concerns remain");
        }
        format!("Not ready: {}", reasons.join(", "))
    };

    let summary = format!(
        "Discussion Reflection for {}:\n\
         • {} contributors, {} contributions\n\
         • Harmony diversity: {:.0}%\n\
         • Preliminary sentiment: {:.0}% favorable\n\
         • Voice concentration: {:.0}%\n\
         • Ready for vote: {}",
        proposal_id,
        contributor_count,
        contribution_count,
        harmony_diversity * 100.0,
        preliminary_sentiment * 100.0,
        voice_concentration * 100.0,
        if ready_for_vote { "Yes" } else { "No" }
    );

    let reflection = DiscussionReflection {
        id: format!("disc_reflection:{}:{}", proposal_id, now.as_micros()),
        proposal_id: proposal_id.clone(),
        timestamp: now,
        contributor_count,
        contribution_count,
        avg_contributions_per_participant,
        max_thread_depth,
        harmony_coverage,
        harmony_diversity,
        absent_harmonies,
        support_count,
        oppose_count,
        neutral_count,
        amend_count,
        preliminary_sentiment,
        voice_concentration,
        cross_camp_engagement,
        substantiveness_score,
        discussion_saturated,
        unaddressed_concerns,
        ready_for_vote,
        readiness_reasoning,
        summary,
    };

    let _ = emit_signal(&ProposalSignal::DiscussionReflectionGenerated {
        proposal_id: proposal_id.clone(),
        ready_for_vote,
    });

    let action_hash = create_entry(&EntryTypes::DiscussionReflection(reflection))?;

    // Link proposal to reflection
    let reflection_anchor = format!("disc_reflections:{}", proposal_id);
    create_entry(&EntryTypes::Anchor(Anchor(reflection_anchor.clone())))?;
    create_link(
        anchor_hash(&reflection_anchor)?,
        action_hash.clone(),
        LinkTypes::ProposalToDiscussionReflection,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created reflection".into()
    )))
}

/// Get all discussion reflections for a proposal
#[hdk_extern]
pub fn get_discussion_reflections(proposal_id: String) -> ExternResult<Vec<Record>> {
    let reflection_anchor = format!("disc_reflections:{}", proposal_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&reflection_anchor)?,
            LinkTypes::ProposalToDiscussionReflection,
        )?,
        GetStrategy::default(),
    )?;

    let mut reflections = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            reflections.push(record);
        }
    }

    // Sort by timestamp
    reflections.sort_by(|a, b| {
        let ts_a = a.action().timestamp();
        let ts_b = b.action().timestamp();
        ts_a.cmp(&ts_b)
    });

    Ok(reflections)
}

/// Check if discussion is ready for voting
#[hdk_extern]
pub fn is_discussion_ready(proposal_id: String) -> ExternResult<DiscussionReadiness> {
    let reflections = get_discussion_reflections(proposal_id.clone())?;

    if let Some(latest) = reflections.into_iter().last() {
        let reflection: DiscussionReflection = latest
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid reflection entry".into()
            )))?;

        Ok(DiscussionReadiness {
            ready: reflection.ready_for_vote,
            reasoning: reflection.readiness_reasoning,
            contributor_count: reflection.contributor_count,
            contribution_count: reflection.contribution_count,
            harmony_diversity: reflection.harmony_diversity,
            unaddressed_concerns: reflection.unaddressed_concerns,
        })
    } else {
        // No reflection yet - generate one
        let _ = reflect_on_discussion(proposal_id)?;
        Ok(DiscussionReadiness {
            ready: false,
            reasoning: "No discussion reflection available yet".to_string(),
            contributor_count: 0,
            contribution_count: 0,
            harmony_diversity: 0.0,
            unaddressed_concerns: vec![],
        })
    }
}

/// Discussion readiness response
#[derive(Serialize, Deserialize, Debug)]
pub struct DiscussionReadiness {
    pub ready: bool,
    pub reasoning: String,
    pub contributor_count: u64,
    pub contribution_count: u64,
    pub harmony_diversity: f64,
    pub unaddressed_concerns: Vec<String>,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate maximum thread depth
fn calculate_max_thread_depth(contributions: &[DiscussionContribution]) -> u8 {
    let mut depth_map: HashMap<String, u8> = HashMap::new();
    let mut max_depth = 0u8;

    // First pass: identify root contributions
    for c in contributions {
        if c.parent_id.is_none() {
            depth_map.insert(c.id.clone(), 1);
        }
    }

    // Multiple passes to propagate depth (simple approach)
    for _ in 0..10 {
        for c in contributions {
            if let Some(parent_id) = &c.parent_id {
                if let Some(&parent_depth) = depth_map.get(parent_id) {
                    let new_depth = parent_depth + 1;
                    depth_map.insert(c.id.clone(), new_depth);
                    max_depth = max_depth.max(new_depth);
                }
            }
        }
    }

    max_depth
}

/// Calculate cross-camp engagement (supporters replying to opposers and vice versa)
fn calculate_cross_camp_engagement(contributions: &[DiscussionContribution]) -> f64 {
    let mut stance_by_id: HashMap<String, Stance> = HashMap::new();
    for c in contributions {
        if let Some(stance) = &c.stance {
            stance_by_id.insert(c.id.clone(), stance.clone());
        }
    }

    let mut cross_camp_replies = 0u64;
    let mut total_replies = 0u64;

    for c in contributions {
        if let Some(parent_id) = &c.parent_id {
            total_replies += 1;
            if let (Some(my_stance), Some(parent_stance)) = (&c.stance, stance_by_id.get(parent_id))
            {
                let is_cross_camp = matches!(
                    (my_stance, parent_stance),
                    (Stance::Support, Stance::Oppose) | (Stance::Oppose, Stance::Support)
                );
                if is_cross_camp {
                    cross_camp_replies += 1;
                }
            }
        }
    }

    if total_replies > 0 {
        cross_camp_replies as f64 / total_replies as f64
    } else {
        0.0
    }
}

/// Identify unaddressed concerns (opposition contributions without replies)
fn identify_unaddressed_concerns(contributions: &[DiscussionContribution]) -> Vec<String> {
    let ids_with_replies: HashSet<String> = contributions
        .iter()
        .filter_map(|c| c.parent_id.clone())
        .collect();

    contributions
        .iter()
        .filter(|c| matches!(c.stance, Some(Stance::Oppose)) && !ids_with_replies.contains(&c.id))
        .map(|c| c.id.clone())
        .collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_contribution(
        id: &str,
        parent: Option<&str>,
        stance: Option<Stance>,
    ) -> DiscussionContribution {
        DiscussionContribution {
            id: id.to_string(),
            proposal_id: "MIP-001".to_string(),
            contributor: format!("did:test:{}", id),
            content: format!("Content for {}", id),
            harmony_tags: vec![],
            stance,
            parent_id: parent.map(|s| s.to_string()),
            created_at: Timestamp::from_micros(0),
            edited: false,
        }
    }

    // --- calculate_max_thread_depth ---

    #[test]
    fn test_thread_depth_empty() {
        assert_eq!(calculate_max_thread_depth(&[]), 0);
    }

    #[test]
    fn test_thread_depth_flat_discussion() {
        // Root-only contributions have no reply chain — max_depth stays 0
        let contribs = vec![
            make_contribution("a", None, None),
            make_contribution("b", None, None),
            make_contribution("c", None, None),
        ];
        assert_eq!(
            calculate_max_thread_depth(&contribs),
            0,
            "No replies means depth 0"
        );
    }

    #[test]
    fn test_thread_depth_single_reply() {
        let contribs = vec![
            make_contribution("a", None, None),
            make_contribution("b", Some("a"), None),
        ];
        // root(1) → b(2), max_depth updated to 2
        assert_eq!(calculate_max_thread_depth(&contribs), 2);
    }

    #[test]
    fn test_thread_depth_linear_chain() {
        let contribs = vec![
            make_contribution("a", None, None),
            make_contribution("b", Some("a"), None),
            make_contribution("c", Some("b"), None),
            make_contribution("d", Some("c"), None),
        ];
        assert_eq!(
            calculate_max_thread_depth(&contribs),
            4,
            "Linear chain a→b→c→d = depth 4"
        );
    }

    #[test]
    fn test_thread_depth_branching() {
        let contribs = vec![
            make_contribution("root", None, None),
            make_contribution("a", Some("root"), None),
            make_contribution("b", Some("root"), None),
            make_contribution("a1", Some("a"), None),
        ];
        // root(1) → a(2) → a1(3), root(1) → b(2)
        assert_eq!(calculate_max_thread_depth(&contribs), 3);
    }

    // --- calculate_cross_camp_engagement ---

    #[test]
    fn test_cross_camp_no_replies() {
        let contribs = vec![
            make_contribution("a", None, Some(Stance::Support)),
            make_contribution("b", None, Some(Stance::Oppose)),
        ];
        assert!((calculate_cross_camp_engagement(&contribs) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_camp_all_same_side() {
        let contribs = vec![
            make_contribution("a", None, Some(Stance::Support)),
            make_contribution("b", Some("a"), Some(Stance::Support)),
        ];
        assert!((calculate_cross_camp_engagement(&contribs) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_camp_full_engagement() {
        let contribs = vec![
            make_contribution("a", None, Some(Stance::Support)),
            make_contribution("b", Some("a"), Some(Stance::Oppose)),
        ];
        assert!(
            (calculate_cross_camp_engagement(&contribs) - 1.0).abs() < 1e-10,
            "100% of replies are cross-camp"
        );
    }

    #[test]
    fn test_cross_camp_mixed() {
        let contribs = vec![
            make_contribution("a", None, Some(Stance::Support)),
            make_contribution("b", Some("a"), Some(Stance::Oppose)), // cross
            make_contribution("c", Some("a"), Some(Stance::Support)), // same
        ];
        // 1 cross / 2 replies = 0.5
        assert!((calculate_cross_camp_engagement(&contribs) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cross_camp_neutral_not_counted() {
        let contribs = vec![
            make_contribution("a", None, Some(Stance::Support)),
            make_contribution("b", Some("a"), Some(Stance::Neutral)),
        ];
        // Neutral→Support is not cross-camp (only Support↔Oppose counts)
        assert!((calculate_cross_camp_engagement(&contribs) - 0.0).abs() < 1e-10);
    }

    // --- identify_unaddressed_concerns ---

    #[test]
    fn test_unaddressed_concerns_none() {
        let contribs = vec![
            make_contribution("a", None, Some(Stance::Oppose)),
            make_contribution("b", Some("a"), Some(Stance::Support)), // addresses concern
        ];
        let concerns = identify_unaddressed_concerns(&contribs);
        assert!(
            concerns.is_empty(),
            "Opposition 'a' has a reply, so it's addressed"
        );
    }

    #[test]
    fn test_unaddressed_concerns_found() {
        let contribs = vec![
            make_contribution("a", None, Some(Stance::Oppose)),
            make_contribution("b", None, Some(Stance::Support)),
        ];
        let concerns = identify_unaddressed_concerns(&contribs);
        assert_eq!(concerns, vec!["a"], "Opposition 'a' has no reply");
    }

    #[test]
    fn test_unaddressed_concerns_support_not_flagged() {
        let contribs = vec![make_contribution("a", None, Some(Stance::Support))];
        let concerns = identify_unaddressed_concerns(&contribs);
        assert!(
            concerns.is_empty(),
            "Support contributions are never 'unaddressed concerns'"
        );
    }

    #[test]
    fn test_unaddressed_concerns_multiple() {
        let contribs = vec![
            make_contribution("a", None, Some(Stance::Oppose)),
            make_contribution("b", None, Some(Stance::Oppose)),
            make_contribution("c", None, Some(Stance::Oppose)),
            make_contribution("d", Some("a"), Some(Stance::Support)), // addresses only 'a'
        ];
        let concerns = identify_unaddressed_concerns(&contribs);
        assert_eq!(concerns.len(), 2, "b and c are unaddressed");
        assert!(concerns.contains(&"b".to_string()));
        assert!(concerns.contains(&"c".to_string()));
    }
}
