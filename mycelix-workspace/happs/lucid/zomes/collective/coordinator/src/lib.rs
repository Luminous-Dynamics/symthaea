//! Collective Sensemaking Coordinator Zome
//!
//! Implements distributed belief sharing, validation, consensus, and emergent truth discovery.

use hdk::prelude::*;
use collective_integrity::*;

/// Share a belief to the collective
#[hdk_extern]
pub fn share_belief(input: ShareBeliefInput) -> ExternResult<Record> {
    let belief_share = BeliefShare {
        content_hash: input.content_hash,
        content: input.content,
        belief_type: input.belief_type,
        epistemic_code: input.epistemic_code,
        confidence: input.confidence,
        tags: input.tags.clone(),
        shared_at: sys_time()?,
        evidence_hashes: input.evidence_hashes,
        embedding: input.embedding,
        stance: input.stance,
    };

    let action_hash = create_entry(EntryTypes::BeliefShare(belief_share.clone()))?;

    // Link to anchor for discovery
    let anchor = anchor_hash("all_belief_shares")?;
    create_link(anchor, action_hash.clone(), LinkTypes::AllBeliefShares, ())?;

    // Link to each tag
    for tag in &input.tags {
        let tag_anchor = anchor_hash(&format!("tag:{}", tag))?;
        create_link(tag_anchor, action_hash.clone(), LinkTypes::TagToBeliefShare, ())?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created belief share".into())))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShareBeliefInput {
    pub content_hash: String,
    pub content: String,
    pub belief_type: String,
    pub epistemic_code: String,
    pub confidence: f64,
    pub tags: Vec<String>,
    pub evidence_hashes: Vec<String>,
    /// HDC embedding for semantic similarity (optional, computed by Tauri if missing)
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// Stance on this belief
    #[serde(default)]
    pub stance: Option<BeliefStance>,
}

/// Vote on a shared belief
#[hdk_extern]
pub fn cast_vote(input: CastVoteInput) -> ExternResult<Record> {
    // Get voter's reputation weight
    let my_agent = agent_info()?.agent_initial_pubkey;
    let weight = get_agent_weight(my_agent)?;

    let vote = ValidationVote {
        belief_share_hash: input.belief_share_hash.clone(),
        vote_type: input.vote_type,
        evidence: input.evidence,
        voter_weight: weight,
        voted_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::ValidationVote(vote))?;

    // Link vote to belief share
    create_link(
        input.belief_share_hash.clone(),
        action_hash.clone(),
        LinkTypes::BeliefShareToVotes,
        (),
    )?;

    // Check if we should compute consensus
    check_and_create_consensus(input.belief_share_hash)?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created vote".into())))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CastVoteInput {
    pub belief_share_hash: ActionHash,
    pub vote_type: ValidationVoteType,
    pub evidence: Option<String>,
}

/// Get an agent's epistemic weight (based on reputation)
fn get_agent_weight(agent: AgentPubKey) -> ExternResult<f64> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToReputation)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(record) = get(ActionHash::try_from(link.target.clone()).unwrap(), GetOptions::default())? {
            if let Some(rep) = record.entry().to_app_option::<EpistemicReputation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                // Weight based on accuracy and calibration
                return Ok((rep.accuracy_score + rep.calibration_score) / 2.0);
            }
        }
    }

    // Default weight for new agents
    Ok(0.5)
}

/// Check if consensus can be computed and create record
fn check_and_create_consensus(belief_hash: ActionHash) -> ExternResult<Option<ActionHash>> {
    let links = get_links(
        LinkQuery::try_new(belief_hash.clone(), LinkTypes::BeliefShareToVotes)?,
        GetStrategy::default(),
    )?;

    if links.len() < 3 {
        return Ok(None); // Not enough votes
    }

    // Collect votes
    let mut corroborate_weight = 0.0;
    let mut contradict_weight = 0.0;
    let mut plausible_weight = 0.0;
    let mut implausible_weight = 0.0;
    let mut total_weight = 0.0;

    for link in &links {
        if let Some(record) = get(ActionHash::try_from(link.target.clone()).unwrap(), GetOptions::default())? {
            if let Some(vote) = record.entry().to_app_option::<ValidationVote>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                let w = vote.voter_weight;
                total_weight += w;

                match vote.vote_type {
                    ValidationVoteType::Corroborate => corroborate_weight += w,
                    ValidationVoteType::Contradict => contradict_weight += w,
                    ValidationVoteType::Plausible => plausible_weight += w,
                    ValidationVoteType::Implausible => implausible_weight += w,
                    ValidationVoteType::Abstain => {}
                }
            }
        }
    }

    if total_weight < 1.0 {
        return Ok(None); // Not enough weight
    }

    // Calculate agreement (corroborate + plausible vs contradict + implausible)
    let support = corroborate_weight + plausible_weight * 0.5;
    let oppose = contradict_weight + implausible_weight * 0.5;
    let agreement_score = support / (support + oppose + 0.0001);

    // Determine consensus type
    let consensus_type = if agreement_score > 0.8 {
        ConsensusType::StrongConsensus
    } else if agreement_score > 0.6 {
        ConsensusType::ModerateConsensus
    } else if agreement_score > 0.4 {
        ConsensusType::WeakConsensus
    } else if (agreement_score - 0.5).abs() < 0.1 {
        ConsensusType::Contested
    } else {
        ConsensusType::Insufficient
    };

    let summary = format!(
        "Consensus from {} validators: {:.0}% agreement ({:.1} support, {:.1} oppose)",
        links.len(),
        agreement_score * 100.0,
        support,
        oppose
    );

    let consensus = ConsensusRecord {
        belief_share_hash: belief_hash.clone(),
        consensus_type,
        validator_count: links.len() as u32,
        agreement_score,
        summary,
        reached_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::ConsensusRecord(consensus))?;
    create_link(belief_hash, action_hash.clone(), LinkTypes::BeliefShareToConsensus, ())?;

    Ok(Some(action_hash))
}

/// Get belief shares by tag
#[hdk_extern]
pub fn get_beliefs_by_tag(tag: String) -> ExternResult<Vec<Record>> {
    let tag_anchor = anchor_hash(&format!("tag:{}", tag))?;
    let links = get_links(
        LinkQuery::try_new(tag_anchor, LinkTypes::TagToBeliefShare)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(record) = get(ActionHash::try_from(link.target).unwrap(), GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get all belief shares
#[hdk_extern]
pub fn get_all_belief_shares(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = anchor_hash("all_belief_shares")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllBeliefShares)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links.into_iter().take(100) {
        if let Some(record) = get(ActionHash::try_from(link.target).unwrap(), GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get votes for a belief share
#[hdk_extern]
pub fn get_belief_votes(belief_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(belief_hash, LinkTypes::BeliefShareToVotes)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(record) = get(ActionHash::try_from(link.target).unwrap(), GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get consensus for a belief share
#[hdk_extern]
pub fn get_belief_consensus(belief_hash: ActionHash) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(belief_hash, LinkTypes::BeliefShareToConsensus)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        return get(ActionHash::try_from(link.target.clone()).unwrap(), GetOptions::default());
    }

    Ok(None)
}

/// Detect emergent patterns across belief shares
///
/// Implements real pattern detection using HDC embeddings:
/// 1. Groups beliefs by semantic similarity (cosine similarity > threshold)
/// 2. Identifies convergence patterns (many people, same belief)
/// 3. Identifies divergence patterns (opposing stances on similar topics)
/// 4. Identifies trends (beliefs appearing over time)
#[hdk_extern]
pub fn detect_patterns(input: DetectPatternsInput) -> ExternResult<Vec<PatternCluster>> {
    let threshold = input.similarity_threshold.unwrap_or(0.7);
    let min_cluster_size = input.min_cluster_size.unwrap_or(2);

    // Get all beliefs with embeddings
    let beliefs = get_all_beliefs_with_embeddings()?;

    if beliefs.len() < min_cluster_size {
        return Ok(Vec::new());
    }

    // Cluster by embedding similarity
    let clusters = cluster_beliefs_by_similarity(&beliefs, threshold, min_cluster_size);

    // Classify each cluster
    let mut pattern_clusters = Vec::new();
    for cluster in clusters {
        let pattern_type = classify_cluster_pattern(&cluster.beliefs);
        let representative = find_cluster_representative(&cluster.beliefs);
        let coherence = calculate_cluster_coherence(&cluster.beliefs);

        pattern_clusters.push(PatternCluster {
            pattern_id: format!("cluster_{}", cluster.beliefs[0].action_hash),
            representative_content: representative.content.clone(),
            representative_hash: representative.action_hash.clone(),
            member_hashes: cluster.beliefs.iter().map(|b| b.action_hash.clone()).collect(),
            member_count: cluster.beliefs.len() as u32,
            pattern_type,
            coherence,
            tags: aggregate_tags(&cluster.beliefs),
        });
    }

    Ok(pattern_clusters)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectPatternsInput {
    /// Minimum cosine similarity for clustering (default: 0.7)
    pub similarity_threshold: Option<f64>,
    /// Minimum beliefs per cluster (default: 2)
    pub min_cluster_size: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatternCluster {
    pub pattern_id: String,
    pub representative_content: String,
    pub representative_hash: ActionHash,
    pub member_hashes: Vec<ActionHash>,
    pub member_count: u32,
    pub pattern_type: PatternType,
    pub coherence: f64,
    pub tags: Vec<String>,
}

/// Internal struct for clustering
#[derive(Clone)]
struct BeliefWithEmbedding {
    action_hash: ActionHash,
    content: String,
    embedding: Vec<f32>,
    _confidence: f64,
    stance: Option<BeliefStance>,
    tags: Vec<String>,
}

struct Cluster {
    beliefs: Vec<BeliefWithEmbedding>,
}

/// Get all beliefs that have embeddings
fn get_all_beliefs_with_embeddings() -> ExternResult<Vec<BeliefWithEmbedding>> {
    let anchor = anchor_hash("all_belief_shares")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllBeliefShares)?,
        GetStrategy::default(),
    )?;

    let mut beliefs = Vec::new();
    for link in links.into_iter().take(500) { // Limit for performance
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(share) = record.entry().to_app_option::<BeliefShare>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                // Only include beliefs with embeddings
                if !share.embedding.is_empty() {
                    beliefs.push(BeliefWithEmbedding {
                        action_hash: hash,
                        content: share.content,
                        embedding: share.embedding,
                        _confidence: share.confidence,
                        stance: share.stance,
                        tags: share.tags,
                    });
                }
            }
        }
    }

    Ok(beliefs)
}

/// Compute cosine similarity between two embeddings
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        norm_a += (*x as f64) * (*x as f64);
        norm_b += (*y as f64) * (*y as f64);
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Cluster beliefs by embedding similarity (simple greedy clustering)
fn cluster_beliefs_by_similarity(
    beliefs: &[BeliefWithEmbedding],
    threshold: f64,
    min_size: usize,
) -> Vec<Cluster> {
    let mut assigned = vec![false; beliefs.len()];
    let mut clusters = Vec::new();

    for i in 0..beliefs.len() {
        if assigned[i] {
            continue;
        }

        // Start a new cluster with this belief
        let mut cluster_beliefs = vec![];
        cluster_beliefs.push(beliefs[i].clone());
        assigned[i] = true;

        // Find all similar beliefs
        for j in (i + 1)..beliefs.len() {
            if assigned[j] {
                continue;
            }

            let sim = cosine_similarity(&beliefs[i].embedding, &beliefs[j].embedding);
            if sim >= threshold {
                cluster_beliefs.push(beliefs[j].clone());
                assigned[j] = true;
            }
        }

        // Only keep clusters that meet minimum size
        if cluster_beliefs.len() >= min_size {
            clusters.push(Cluster { beliefs: cluster_beliefs });
        }
    }

    clusters
}

/// Classify the type of pattern in a cluster
fn classify_cluster_pattern(beliefs: &[BeliefWithEmbedding]) -> PatternType {
    // Check for divergence (opposing stances)
    let mut agree_count = 0;
    let mut disagree_count = 0;
    for belief in beliefs {
        match belief.stance {
            Some(BeliefStance::StronglyAgree) | Some(BeliefStance::Agree) => agree_count += 1,
            Some(BeliefStance::StronglyDisagree) | Some(BeliefStance::Disagree) => disagree_count += 1,
            _ => {}
        }
    }

    if agree_count > 0 && disagree_count > 0 {
        // Mix of agreeing and disagreeing stances = divergence or contradiction
        if (agree_count as f64 / beliefs.len() as f64) > 0.7
            || (disagree_count as f64 / beliefs.len() as f64) > 0.7 {
            PatternType::Divergence
        } else {
            PatternType::ContradictionCluster
        }
    } else if beliefs.len() >= 3 {
        // Multiple people converging on similar content
        PatternType::Convergence
    } else {
        PatternType::Cluster
    }
}

/// Find the belief closest to the cluster centroid
fn find_cluster_representative(beliefs: &[BeliefWithEmbedding]) -> &BeliefWithEmbedding {
    if beliefs.len() == 1 {
        return &beliefs[0];
    }

    // Compute centroid (average of all embeddings)
    let dim = beliefs[0].embedding.len();
    let mut centroid = vec![0.0f32; dim];

    for belief in beliefs {
        for (i, val) in belief.embedding.iter().enumerate() {
            centroid[i] += val / beliefs.len() as f32;
        }
    }

    // Find belief closest to centroid
    let mut best_idx = 0;
    let mut best_sim = -1.0;

    for (i, belief) in beliefs.iter().enumerate() {
        let sim = cosine_similarity(&belief.embedding, &centroid);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }

    &beliefs[best_idx]
}

/// Calculate cluster coherence (average pairwise similarity)
fn calculate_cluster_coherence(beliefs: &[BeliefWithEmbedding]) -> f64 {
    if beliefs.len() < 2 {
        return 1.0;
    }

    let mut total_sim = 0.0;
    let mut count = 0;

    for i in 0..beliefs.len() {
        for j in (i + 1)..beliefs.len() {
            total_sim += cosine_similarity(&beliefs[i].embedding, &beliefs[j].embedding);
            count += 1;
        }
    }

    if count > 0 {
        total_sim / count as f64
    } else {
        1.0
    }
}

/// Aggregate tags from all beliefs in cluster
fn aggregate_tags(beliefs: &[BeliefWithEmbedding]) -> Vec<String> {
    use std::collections::HashMap;
    let mut tag_counts: HashMap<String, usize> = HashMap::new();

    for belief in beliefs {
        for tag in &belief.tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    // Return tags that appear in at least 50% of beliefs
    let threshold = beliefs.len() / 2;
    tag_counts
        .into_iter()
        .filter(|(_, count)| *count > threshold)
        .map(|(tag, _)| tag)
        .collect()
}

/// Get existing detected patterns (legacy compatibility)
#[hdk_extern]
pub fn get_recorded_patterns(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = anchor_hash("all_patterns")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllPatterns)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(record) = get(ActionHash::try_from(link.target).unwrap(), GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Record an emergent pattern
#[hdk_extern]
pub fn record_pattern(input: RecordPatternInput) -> ExternResult<Record> {
    let pattern = EmergentPattern {
        pattern_id: input.pattern_id,
        description: input.description,
        belief_hashes: input.belief_hashes.clone(),
        pattern_type: input.pattern_type,
        confidence: input.confidence,
        detected_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::EmergentPattern(pattern))?;

    // Link to anchor
    let anchor = anchor_hash("all_patterns")?;
    create_link(anchor, action_hash.clone(), LinkTypes::AllPatterns, ())?;

    // Link to each belief
    for belief_hash in input.belief_hashes {
        create_link(action_hash.clone(), belief_hash, LinkTypes::PatternToBeliefs, ())?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created pattern".into())))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordPatternInput {
    pub pattern_id: String,
    pub description: String,
    pub belief_hashes: Vec<ActionHash>,
    pub pattern_type: PatternType,
    pub confidence: f64,
}

/// Update agent's epistemic reputation
#[hdk_extern]
pub fn update_reputation(input: UpdateReputationInput) -> ExternResult<Record> {
    let reputation = EpistemicReputation {
        agent: input.agent.clone(),
        domains: input.domains,
        accuracy_score: input.accuracy_score,
        calibration_score: input.calibration_score,
        validation_count: input.validation_count,
        updated_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::EpistemicReputation(reputation))?;

    // Link to agent
    create_link(input.agent, action_hash.clone(), LinkTypes::AgentToReputation, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created reputation".into())))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateReputationInput {
    pub agent: AgentPubKey,
    pub domains: Vec<String>,
    pub accuracy_score: f64,
    pub calibration_score: f64,
    pub validation_count: u32,
}

// ============================================================================
// DOMAIN-SPECIFIC EXPERTISE (Phase 2.2)
// ============================================================================

/// Get domain-specific expertise score for an agent
///
/// Calculates expertise based on:
/// - Historical accuracy in the domain
/// - Number of validations in the domain
/// - Recency of domain activity (trust decay)
#[hdk_extern]
pub fn get_domain_expertise(input: DomainExpertiseInput) -> ExternResult<DomainExpertiseResult> {
    let now = sys_time()?;
    let decay_period_micros = 30 * 24 * 3600 * 1_000_000i64; // 30 days

    // Get agent's reputation
    let reputation = get_agent_reputation(input.agent.clone())?;

    // Check if agent has expertise in this domain
    let has_domain = reputation
        .as_ref()
        .map(|r| r.domains.contains(&input.domain))
        .unwrap_or(false);

    // Get domain-specific votes
    let domain_votes = get_agent_domain_votes(input.agent.clone(), &input.domain)?;

    // Calculate accuracy within domain
    let domain_accuracy = calculate_domain_accuracy(&domain_votes);

    // Calculate trust decay based on last activity
    let last_activity = domain_votes
        .iter()
        .map(|v| v.voted_at)
        .max()
        .unwrap_or(Timestamp::from_micros(0));

    let time_since_last = now.as_micros() - last_activity.as_micros();
    let decay_factor = if time_since_last < decay_period_micros {
        1.0 - (time_since_last as f64 / decay_period_micros as f64) * 0.3 // Max 30% decay
    } else {
        0.7 // Floor at 70% after decay period
    };

    // Calculate expertise score
    let base_expertise = if has_domain {
        (domain_accuracy + reputation.as_ref().map(|r| r.accuracy_score).unwrap_or(0.5)) / 2.0
    } else {
        0.3 // Low base for non-experts
    };

    // Apply decay
    let expertise_score = base_expertise * decay_factor;

    // Confidence in the expertise (based on sample size)
    let confidence = (domain_votes.len() as f64 / 20.0).min(1.0);

    Ok(DomainExpertiseResult {
        agent: input.agent,
        domain: input.domain,
        expertise_score,
        confidence,
        validation_count: domain_votes.len() as u32,
        last_activity: if domain_votes.is_empty() { None } else { Some(last_activity) },
        decay_factor,
        is_recognized_expert: has_domain && expertise_score > 0.6,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainExpertiseInput {
    pub agent: AgentPubKey,
    pub domain: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainExpertiseResult {
    pub agent: AgentPubKey,
    pub domain: String,
    pub expertise_score: f64,
    pub confidence: f64,
    pub validation_count: u32,
    pub last_activity: Option<Timestamp>,
    pub decay_factor: f64,
    pub is_recognized_expert: bool,
}

fn get_agent_reputation(agent: AgentPubKey) -> ExternResult<Option<EpistemicReputation>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToReputation)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(record) = get(ActionHash::try_from(link.target.clone()).unwrap(), GetOptions::default())? {
            return Ok(record.entry().to_app_option::<EpistemicReputation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?);
        }
    }

    Ok(None)
}

/// Get votes by an agent that are related to a specific domain
fn get_agent_domain_votes(agent: AgentPubKey, domain: &str) -> ExternResult<Vec<ValidationVote>> {
    // This would ideally be indexed, but for now we scan all beliefs
    let anchor = anchor_hash("all_belief_shares")?;
    let belief_links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllBeliefShares)?,
        GetStrategy::default(),
    )?;

    let mut domain_votes = Vec::new();

    for belief_link in belief_links.into_iter().take(200) {
        let belief_hash = ActionHash::try_from(belief_link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;

        // Check if belief is in this domain
        if let Some(record) = get(belief_hash.clone(), GetOptions::default())? {
            if let Some(share) = record.entry().to_app_option::<BeliefShare>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                if !share.tags.contains(&domain.to_string()) {
                    continue;
                }

                // Get votes for this belief
                let vote_links = get_links(
                    LinkQuery::try_new(belief_hash, LinkTypes::BeliefShareToVotes)?,
                    GetStrategy::default(),
                )?;

                for vote_link in vote_links {
                    let vote_hash = ActionHash::try_from(vote_link.target)
                        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
                    if let Some(vote_record) = get(vote_hash, GetOptions::default())? {
                        // Check if this vote is by our agent
                        if vote_record.action().author() == &agent {
                            if let Some(vote) = vote_record.entry().to_app_option::<ValidationVote>()
                                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                                domain_votes.push(vote);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(domain_votes)
}

fn calculate_domain_accuracy(votes: &[ValidationVote]) -> f64 {
    if votes.is_empty() {
        return 0.5; // Neutral for no history
    }

    // For now, use average voter weight as proxy for accuracy
    // In a full implementation, we'd compare predictions to outcomes
    let total: f64 = votes.iter().map(|v| v.voter_weight).sum();
    total / votes.len() as f64
}

/// Apply trust decay to all relationships
#[hdk_extern]
pub fn apply_trust_decay(_: ()) -> ExternResult<u32> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let decay_threshold_micros = 14 * 24 * 3600 * 1_000_000i64; // 14 days
    let decay_amount = 0.05; // 5% decay per application

    let links = get_links(
        LinkQuery::try_new(my_agent.clone(), LinkTypes::AgentToRelationship)?,
        GetStrategy::default(),
    )?;

    let mut decayed_count = 0u32;

    for link in links {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(rel) = record.entry().to_app_option::<AgentRelationship>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {

                let time_since = now.as_micros() - rel.last_interaction.as_micros();

                if time_since > decay_threshold_micros && rel.trust_score > 0.1 {
                    // Apply decay by creating updated relationship
                    let new_trust = (rel.trust_score - decay_amount).max(0.1);

                    let updated_rel = AgentRelationship {
                        trust_score: new_trust,
                        ..rel.clone()
                    };

                    // Create new entry (immutable DHT)
                    let action_hash = create_entry(EntryTypes::AgentRelationship(updated_rel))?;
                    create_link(
                        my_agent.clone(),
                        action_hash,
                        LinkTypes::AgentToRelationship,
                        rel.other_agent.to_string().as_bytes().to_vec(),
                    )?;

                    decayed_count += 1;
                }
            }
        }
    }

    Ok(decayed_count)
}

/// Explain why a consensus was reached
#[hdk_extern]
pub fn explain_consensus(belief_hash: ActionHash) -> ExternResult<ConsensusExplanation> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Get all votes
    let vote_links = get_links(
        LinkQuery::try_new(belief_hash.clone(), LinkTypes::BeliefShareToVotes)?,
        GetStrategy::default(),
    )?;

    if vote_links.is_empty() {
        return Ok(ConsensusExplanation {
            belief_hash,
            summary: "No votes have been cast yet.".to_string(),
            vote_breakdown: VoteBreakdown::default(),
            key_voters: Vec::new(),
            trusted_network_opinion: None,
            confidence_factors: Vec::new(),
        });
    }

    let mut vote_breakdown = VoteBreakdown::default();
    let mut key_voters: Vec<VoterContribution> = Vec::new();

    for link in &vote_links {
        let vote_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;

        if let Some(record) = get(vote_hash, GetOptions::default())? {
            if let Some(vote) = record.entry().to_app_option::<ValidationVote>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {

                let voter = record.action().author().clone();
                let relationship_weight = get_relationship_weight(&my_agent, &voter)?;
                let combined_weight = (relationship_weight + vote.voter_weight) / 2.0;

                // Update breakdown
                match vote.vote_type {
                    ValidationVoteType::Corroborate => {
                        vote_breakdown.corroborate_count += 1;
                        vote_breakdown.corroborate_weight += combined_weight;
                    }
                    ValidationVoteType::Plausible => {
                        vote_breakdown.plausible_count += 1;
                        vote_breakdown.plausible_weight += combined_weight;
                    }
                    ValidationVoteType::Abstain => {
                        vote_breakdown.abstain_count += 1;
                    }
                    ValidationVoteType::Implausible => {
                        vote_breakdown.implausible_count += 1;
                        vote_breakdown.implausible_weight += combined_weight;
                    }
                    ValidationVoteType::Contradict => {
                        vote_breakdown.contradict_count += 1;
                        vote_breakdown.contradict_weight += combined_weight;
                    }
                }

                // Track high-influence voters
                if combined_weight > 0.6 {
                    key_voters.push(VoterContribution {
                        is_trusted: relationship_weight > 0.5,
                        vote_type: format!("{:?}", vote.vote_type),
                        weight: combined_weight,
                        evidence_provided: vote.evidence.is_some(),
                    });
                }
            }
        }
    }

    // Calculate trusted network opinion
    let support_weight = vote_breakdown.corroborate_weight + vote_breakdown.plausible_weight * 0.5;
    let oppose_weight = vote_breakdown.contradict_weight + vote_breakdown.implausible_weight * 0.5;
    let total_weight = support_weight + oppose_weight + 0.001;

    let trusted_network_opinion = if vote_links.len() >= 3 {
        Some(support_weight / total_weight)
    } else {
        None
    };

    // Generate confidence factors
    let mut confidence_factors = Vec::new();

    if key_voters.iter().any(|v| v.is_trusted) {
        confidence_factors.push("Trusted peers have contributed".to_string());
    }
    if key_voters.iter().any(|v| v.evidence_provided) {
        confidence_factors.push("Evidence has been provided".to_string());
    }
    if vote_links.len() >= 5 {
        confidence_factors.push("Multiple validators participated".to_string());
    }

    // Generate summary
    let summary = generate_consensus_summary(&vote_breakdown, trusted_network_opinion);

    Ok(ConsensusExplanation {
        belief_hash,
        summary,
        vote_breakdown,
        key_voters,
        trusted_network_opinion,
        confidence_factors,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusExplanation {
    pub belief_hash: ActionHash,
    pub summary: String,
    pub vote_breakdown: VoteBreakdown,
    pub key_voters: Vec<VoterContribution>,
    pub trusted_network_opinion: Option<f64>,
    pub confidence_factors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct VoteBreakdown {
    pub corroborate_count: u32,
    pub corroborate_weight: f64,
    pub plausible_count: u32,
    pub plausible_weight: f64,
    pub abstain_count: u32,
    pub implausible_count: u32,
    pub implausible_weight: f64,
    pub contradict_count: u32,
    pub contradict_weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoterContribution {
    pub is_trusted: bool,
    pub vote_type: String,
    pub weight: f64,
    pub evidence_provided: bool,
}

fn generate_consensus_summary(breakdown: &VoteBreakdown, trusted_opinion: Option<f64>) -> String {
    let total_votes = breakdown.corroborate_count
        + breakdown.plausible_count
        + breakdown.abstain_count
        + breakdown.implausible_count
        + breakdown.contradict_count;

    if total_votes == 0 {
        return "No votes have been cast.".to_string();
    }

    let support = breakdown.corroborate_count + breakdown.plausible_count;
    let oppose = breakdown.implausible_count + breakdown.contradict_count;

    let opinion_str = if let Some(o) = trusted_opinion {
        if o > 0.7 {
            "Your trusted network supports this belief."
        } else if o < 0.3 {
            "Your trusted network has concerns about this belief."
        } else {
            "Your trusted network is divided on this belief."
        }
    } else {
        "Not enough votes to determine network opinion."
    };

    format!(
        "{} of {} validators support this belief, {} oppose. {}",
        support, total_votes, oppose, opinion_str
    )
}

/// Get collective statistics
#[hdk_extern]
pub fn get_collective_stats(_: ()) -> ExternResult<CollectiveStats> {
    let beliefs_anchor = anchor_hash("all_belief_shares")?;
    let belief_links = get_links(
        LinkQuery::try_new(beliefs_anchor, LinkTypes::AllBeliefShares)?,
        GetStrategy::default(),
    )?;

    let patterns_anchor = anchor_hash("all_patterns")?;
    let pattern_links = get_links(
        LinkQuery::try_new(patterns_anchor, LinkTypes::AllPatterns)?,
        GetStrategy::default(),
    )?;

    Ok(CollectiveStats {
        total_belief_shares: belief_links.len() as u32,
        total_patterns: pattern_links.len() as u32,
        active_validators: 0, // Would need to track this
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollectiveStats {
    pub total_belief_shares: u32,
    pub total_patterns: u32,
    pub active_validators: u32,
}

/// Helper to create anchor hash
fn anchor_hash(anchor: &str) -> ExternResult<EntryHash> {
    let anchor_entry = Anchor(anchor.to_string());
    hash_entry(&EntryTypes::Anchor(anchor_entry))
}

/// Create anchor entry and return hash
#[allow(dead_code)]
fn create_anchor(anchor: &str) -> ExternResult<EntryHash> {
    let anchor_entry = Anchor(anchor.to_string());
    create_entry(&EntryTypes::Anchor(anchor_entry))?;
    anchor_hash(anchor)
}

// ============================================================================
// PARTNERSHIP / RELATIONSHIP TRACKING
// ============================================================================

/// Update or create a relationship with another agent
#[hdk_extern]
pub fn update_relationship(input: UpdateRelationshipInput) -> ExternResult<Record> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Check for existing relationship
    let existing = get_relationship_internal(my_agent.clone(), input.other_agent.clone())?;

    let relationship = if let Some(existing_rel) = existing {
        // Update existing relationship
        AgentRelationship {
            other_agent: input.other_agent.clone(),
            trust_score: input.trust_delta.map_or(existing_rel.trust_score, |d| {
                (existing_rel.trust_score + d).clamp(0.0, 1.0)
            }),
            interaction_count: existing_rel.interaction_count + 1,
            last_interaction: now,
            relationship_stage: input.relationship_stage.unwrap_or(existing_rel.relationship_stage),
            shared_domains: merge_domains(existing_rel.shared_domains, input.new_domains.unwrap_or_default()),
            agreement_ratio: input.agreement_ratio.unwrap_or(existing_rel.agreement_ratio),
        }
    } else {
        // Create new relationship
        AgentRelationship {
            other_agent: input.other_agent.clone(),
            trust_score: input.trust_delta.unwrap_or(0.5), // Start at neutral
            interaction_count: 1,
            last_interaction: now,
            relationship_stage: input.relationship_stage.unwrap_or(RelationshipStage::NoRelation),
            shared_domains: input.new_domains.unwrap_or_default(),
            agreement_ratio: input.agreement_ratio.unwrap_or(0.5),
        }
    };

    let action_hash = create_entry(EntryTypes::AgentRelationship(relationship.clone()))?;

    // Link from my agent to this relationship
    create_link(
        my_agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToRelationship,
        input.other_agent.to_string().as_bytes().to_vec(),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created relationship".into())))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateRelationshipInput {
    pub other_agent: AgentPubKey,
    /// Delta to apply to trust score (-1.0 to 1.0)
    pub trust_delta: Option<f64>,
    /// New relationship stage (if changed)
    pub relationship_stage: Option<RelationshipStage>,
    /// New domains to add
    pub new_domains: Option<Vec<String>>,
    /// Updated agreement ratio
    pub agreement_ratio: Option<f64>,
}

/// Get relationship with a specific agent
#[hdk_extern]
pub fn get_relationship(other_agent: AgentPubKey) -> ExternResult<Option<AgentRelationship>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    get_relationship_internal(my_agent, other_agent)
}

fn get_relationship_internal(my_agent: AgentPubKey, other_agent: AgentPubKey) -> ExternResult<Option<AgentRelationship>> {
    let links = get_links(
        LinkQuery::try_new(my_agent, LinkTypes::AgentToRelationship)?,
        GetStrategy::default(),
    )?;

    let other_str = other_agent.to_string();
    for link in links {
        // Check if this link is for the other_agent (tag contains their pubkey)
        if let Ok(tag_str) = String::from_utf8(link.tag.into_inner()) {
            if tag_str == other_str {
                if let Some(record) = get(ActionHash::try_from(link.target).unwrap(), GetOptions::default())? {
                    return Ok(record.entry().to_app_option::<AgentRelationship>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?);
                }
            }
        }
    }

    Ok(None)
}

/// Get all my relationships
#[hdk_extern]
pub fn get_my_relationships(_: ()) -> ExternResult<Vec<AgentRelationship>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(my_agent, LinkTypes::AgentToRelationship)?,
        GetStrategy::default(),
    )?;

    let mut relationships = Vec::new();
    for link in links {
        if let Some(record) = get(ActionHash::try_from(link.target).unwrap(), GetOptions::default())? {
            if let Some(rel) = record.entry().to_app_option::<AgentRelationship>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                relationships.push(rel);
            }
        }
    }

    Ok(relationships)
}

fn merge_domains(existing: Vec<String>, new: Vec<String>) -> Vec<String> {
    let mut merged = existing;
    for d in new {
        if !merged.contains(&d) {
            merged.push(d);
        }
    }
    merged
}

// ============================================================================
// WEIGHTED CONSENSUS (Using Relationships)
// ============================================================================

/// Calculate weighted consensus for a belief, factoring in relationships
#[hdk_extern]
pub fn calculate_weighted_consensus(belief_hash: ActionHash) -> ExternResult<WeightedConsensusResult> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Get all votes
    let vote_links = get_links(
        LinkQuery::try_new(belief_hash.clone(), LinkTypes::BeliefShareToVotes)?,
        GetStrategy::default(),
    )?;

    if vote_links.is_empty() {
        return Ok(WeightedConsensusResult {
            weighted_value: 0.5,
            confidence: 0.0,
            total_weight: 0.0,
            voter_count: 0,
            breakdown: ConsensusBreakdown::default(),
        });
    }

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    let mut breakdown = ConsensusBreakdown::default();

    for link in &vote_links {
        let vote_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        let record = get(vote_hash, GetOptions::default())?;

        if let Some(record) = record {
            if let Some(vote) = record.entry().to_app_option::<ValidationVote>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                // Get the voter's agent from the action
                let voter = record.action().author().clone();

                // Calculate weight from relationship + domain expertise
                let relationship_weight = get_relationship_weight(&my_agent, &voter)?;
                let expertise_weight = vote.voter_weight; // From reputation

                // Combined weight: 60% relationship, 40% expertise
                let combined_weight = relationship_weight * 0.6 + expertise_weight * 0.4;

                // Convert vote to numeric value
                let vote_value = match vote.vote_type {
                    ValidationVoteType::Corroborate => 1.0,
                    ValidationVoteType::Plausible => 0.75,
                    ValidationVoteType::Abstain => 0.5,
                    ValidationVoteType::Implausible => 0.25,
                    ValidationVoteType::Contradict => 0.0,
                };

                weighted_sum += vote_value * combined_weight;
                total_weight += combined_weight;

                // Update breakdown
                match vote.vote_type {
                    ValidationVoteType::Corroborate => breakdown.corroborate += combined_weight,
                    ValidationVoteType::Plausible => breakdown.plausible += combined_weight,
                    ValidationVoteType::Abstain => breakdown.abstain += combined_weight,
                    ValidationVoteType::Implausible => breakdown.implausible += combined_weight,
                    ValidationVoteType::Contradict => breakdown.contradict += combined_weight,
                }
            }
        }
    }

    let weighted_value = if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        0.5
    };

    // Confidence based on total weight and voter count
    let confidence = (total_weight / vote_links.len() as f64).min(1.0);

    Ok(WeightedConsensusResult {
        weighted_value,
        confidence,
        total_weight,
        voter_count: vote_links.len() as u32,
        breakdown,
    })
}

fn get_relationship_weight(my_agent: &AgentPubKey, voter: &AgentPubKey) -> ExternResult<f64> {
    // Check if this is myself
    if my_agent == voter {
        return Ok(1.0); // Full trust in own votes
    }

    // Get relationship
    if let Some(rel) = get_relationship_internal(my_agent.clone(), voter.clone())? {
        // Weight based on trust and relationship stage
        let stage_multiplier = match rel.relationship_stage {
            RelationshipStage::NoRelation => 0.5,
            RelationshipStage::Acquaintance => 0.7,
            RelationshipStage::Collaborator => 0.85,
            RelationshipStage::TrustedPeer => 0.95,
            RelationshipStage::PartnerInTruth => 1.0,
        };

        Ok(rel.trust_score * stage_multiplier)
    } else {
        // Unknown agent - default moderate weight
        Ok(0.5)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightedConsensusResult {
    /// Weighted consensus value (0.0 = full disagree, 1.0 = full agree)
    pub weighted_value: f64,
    /// Confidence in the consensus (based on weight coverage)
    pub confidence: f64,
    /// Total weight accumulated
    pub total_weight: f64,
    /// Number of voters
    pub voter_count: u32,
    /// Breakdown by vote type (weighted)
    pub breakdown: ConsensusBreakdown,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ConsensusBreakdown {
    pub corroborate: f64,
    pub plausible: f64,
    pub abstain: f64,
    pub implausible: f64,
    pub contradict: f64,
}

// ============================================================================
// TEMPORAL PATTERN ANALYSIS (Phase 2.1)
// ============================================================================

/// Analyze belief trends over time
///
/// Returns temporal patterns including:
/// - Frequency-over-time analysis
/// - Belief velocity (rapid adoption)
/// - Emerging consensus detection
/// - Herding effect warnings
#[hdk_extern]
pub fn analyze_belief_trends(input: TrendAnalysisInput) -> ExternResult<TrendAnalysisResult> {
    let now = sys_time()?;
    let window_micros = (input.time_window_hours.unwrap_or(24) as i64) * 3600 * 1_000_000;
    let cutoff = Timestamp::from_micros(now.as_micros() - window_micros);

    // Get all belief shares
    let all_beliefs = get_all_beliefs_with_timestamps()?;

    // Filter to time window
    let beliefs_in_window: Vec<_> = all_beliefs
        .iter()
        .filter(|b| b.shared_at >= cutoff)
        .collect();

    // Analyze by tag frequency
    let tag_trends = analyze_tag_frequency(&beliefs_in_window, input.min_occurrences.unwrap_or(2));

    // Detect belief velocity (rapid adoption patterns)
    let velocity_alerts = detect_belief_velocity(&beliefs_in_window, input.velocity_threshold.unwrap_or(3));

    // Detect emerging consensus
    let emerging_consensus = detect_emerging_consensus(&beliefs_in_window)?;

    // Detect herding effects
    let herding_warnings = detect_herding_effect(&beliefs_in_window);

    // Calculate overall trend direction
    let trend_direction = calculate_overall_trend(&beliefs_in_window, &all_beliefs);

    Ok(TrendAnalysisResult {
        time_window_hours: input.time_window_hours.unwrap_or(24),
        beliefs_in_window: beliefs_in_window.len() as u32,
        total_beliefs: all_beliefs.len() as u32,
        tag_trends,
        velocity_alerts,
        emerging_consensus,
        herding_warnings,
        trend_direction,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrendAnalysisInput {
    /// Time window in hours (default: 24)
    pub time_window_hours: Option<u32>,
    /// Minimum occurrences for a tag to be considered trending (default: 2)
    pub min_occurrences: Option<u32>,
    /// Minimum beliefs per hour to trigger velocity alert (default: 3)
    pub velocity_threshold: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrendAnalysisResult {
    pub time_window_hours: u32,
    pub beliefs_in_window: u32,
    pub total_beliefs: u32,
    pub tag_trends: Vec<TagTrend>,
    pub velocity_alerts: Vec<VelocityAlert>,
    pub emerging_consensus: Vec<EmergingConsensusAlert>,
    pub herding_warnings: Vec<HerdingWarning>,
    pub trend_direction: TrendDirection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TagTrend {
    pub tag: String,
    pub count_in_window: u32,
    pub count_total: u32,
    pub growth_rate: f64, // Percentage increase vs baseline
    pub is_trending: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VelocityAlert {
    pub tag: String,
    pub beliefs_per_hour: f64,
    pub peak_hour_beliefs: u32,
    pub description: String,
    pub severity: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmergingConsensusAlert {
    pub topic: String,
    pub belief_hashes: Vec<ActionHash>,
    pub agreement_level: f64,
    pub participant_count: u32,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HerdingWarning {
    pub topic: String,
    pub warning_type: HerdingType,
    pub severity: f64,
    pub description: String,
    pub suggestion: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HerdingType {
    /// Many people adopting same belief rapidly
    RapidAdoption,
    /// Uniformity with low diversity of reasoning
    UniformityBias,
    /// Early adopter influence (few sources, many followers)
    InfluenceCascade,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Collective is converging on shared truths
    Converging,
    /// Collective is exploring diverse perspectives
    Diverging,
    /// Collective activity is stable
    Stable,
    /// Collective activity is declining
    Declining,
}

/// Internal struct with timestamp
struct BeliefWithTimestamp {
    action_hash: ActionHash,
    _content: String,
    tags: Vec<String>,
    shared_at: Timestamp,
    confidence: f64,
    embedding: Vec<f32>,
}

fn get_all_beliefs_with_timestamps() -> ExternResult<Vec<BeliefWithTimestamp>> {
    let anchor = anchor_hash("all_belief_shares")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllBeliefShares)?,
        GetStrategy::default(),
    )?;

    let mut beliefs = Vec::new();
    for link in links.into_iter().take(1000) {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(share) = record.entry().to_app_option::<BeliefShare>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                beliefs.push(BeliefWithTimestamp {
                    action_hash: hash,
                    _content: share.content,
                    tags: share.tags,
                    shared_at: share.shared_at,
                    confidence: share.confidence,
                    embedding: share.embedding,
                });
            }
        }
    }

    // Sort by timestamp
    beliefs.sort_by(|a, b| a.shared_at.cmp(&b.shared_at));
    Ok(beliefs)
}

fn analyze_tag_frequency(beliefs: &[&BeliefWithTimestamp], min_occurrences: u32) -> Vec<TagTrend> {
    use std::collections::HashMap;

    let mut tag_counts: HashMap<String, u32> = HashMap::new();
    for belief in beliefs {
        for tag in &belief.tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    let total_beliefs = beliefs.len() as f64;
    let mut trends: Vec<TagTrend> = tag_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_occurrences)
        .map(|(tag, count)| {
            let proportion = count as f64 / total_beliefs;
            TagTrend {
                tag,
                count_in_window: count,
                count_total: count, // Would need historical data for accurate total
                growth_rate: proportion * 100.0,
                is_trending: proportion > 0.1, // >10% of beliefs
            }
        })
        .collect();

    trends.sort_by(|a, b| b.count_in_window.cmp(&a.count_in_window));
    trends.truncate(10); // Top 10 trending tags
    trends
}

fn detect_belief_velocity(beliefs: &[&BeliefWithTimestamp], threshold: u32) -> Vec<VelocityAlert> {
    use std::collections::HashMap;

    if beliefs.is_empty() {
        return Vec::new();
    }

    // Group by hour buckets
    let first_time = beliefs[0].shared_at.as_micros();
    let mut hourly_counts: HashMap<String, HashMap<i64, u32>> = HashMap::new();

    for belief in beliefs {
        let hour_bucket = (belief.shared_at.as_micros() - first_time) / (3600 * 1_000_000);
        for tag in &belief.tags {
            *hourly_counts
                .entry(tag.clone())
                .or_default()
                .entry(hour_bucket)
                .or_insert(0) += 1;
        }
    }

    let _total_hours = beliefs.len().max(1) as f64 / threshold as f64; // Rough estimate
    let mut alerts = Vec::new();

    for (tag, hours) in hourly_counts {
        let peak_hour = hours.values().max().copied().unwrap_or(0);
        let avg_per_hour = hours.values().sum::<u32>() as f64 / hours.len().max(1) as f64;

        if peak_hour >= threshold {
            let severity = (peak_hour as f64 / threshold as f64).min(1.0);
            alerts.push(VelocityAlert {
                tag: tag.clone(),
                beliefs_per_hour: avg_per_hour,
                peak_hour_beliefs: peak_hour,
                description: format!(
                    "'{}' is being rapidly adopted ({} beliefs in peak hour, {:.1} avg/hour)",
                    tag, peak_hour, avg_per_hour
                ),
                severity,
            });
        }
    }

    alerts.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    alerts.truncate(5); // Top 5 velocity alerts
    alerts
}

fn detect_emerging_consensus(beliefs: &[&BeliefWithTimestamp]) -> ExternResult<Vec<EmergingConsensusAlert>> {
    if beliefs.len() < 3 {
        return Ok(Vec::new());
    }

    // Group beliefs by semantic similarity (using embedding clustering)
    let beliefs_with_embeddings: Vec<_> = beliefs
        .iter()
        .filter(|b| !b.embedding.is_empty())
        .collect();

    if beliefs_with_embeddings.len() < 3 {
        return Ok(Vec::new());
    }

    // Simple clustering for consensus detection
    let threshold = 0.75;
    let mut alerts = Vec::new();
    let mut processed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for i in 0..beliefs_with_embeddings.len() {
        if processed.contains(&i) {
            continue;
        }

        let mut cluster_hashes = vec![beliefs_with_embeddings[i].action_hash.clone()];
        let mut total_confidence = beliefs_with_embeddings[i].confidence;

        for j in (i + 1)..beliefs_with_embeddings.len() {
            if processed.contains(&j) {
                continue;
            }

            let sim = cosine_similarity(
                &beliefs_with_embeddings[i].embedding,
                &beliefs_with_embeddings[j].embedding,
            );

            if sim >= threshold {
                cluster_hashes.push(beliefs_with_embeddings[j].action_hash.clone());
                total_confidence += beliefs_with_embeddings[j].confidence;
                processed.insert(j);
            }
        }

        if cluster_hashes.len() >= 3 {
            let avg_confidence = total_confidence / cluster_hashes.len() as f64;
            let topic = beliefs_with_embeddings[i]
                .tags
                .first()
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string());

            alerts.push(EmergingConsensusAlert {
                topic,
                belief_hashes: cluster_hashes.clone(),
                agreement_level: avg_confidence,
                participant_count: cluster_hashes.len() as u32,
                description: format!(
                    "Emerging consensus detected: {} participants with {:.0}% average confidence",
                    cluster_hashes.len(),
                    avg_confidence * 100.0
                ),
            });
        }

        processed.insert(i);
    }

    Ok(alerts)
}

fn detect_herding_effect(beliefs: &[&BeliefWithTimestamp]) -> Vec<HerdingWarning> {
    use std::collections::HashMap;

    if beliefs.len() < 5 {
        return Vec::new();
    }

    let mut warnings = Vec::new();
    let mut tag_confidences: HashMap<String, Vec<f64>> = HashMap::new();

    // Collect confidence levels by tag
    for belief in beliefs {
        for tag in &belief.tags {
            tag_confidences
                .entry(tag.clone())
                .or_default()
                .push(belief.confidence);
        }
    }

    for (tag, confidences) in tag_confidences {
        if confidences.len() < 3 {
            continue;
        }

        // Check for uniformity bias (all confidences very similar)
        let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let variance = confidences
            .iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f64>()
            / confidences.len() as f64;

        // Low variance with high mean = potential herding
        if variance < 0.02 && mean > 0.7 {
            warnings.push(HerdingWarning {
                topic: tag.clone(),
                warning_type: HerdingType::UniformityBias,
                severity: (1.0 - variance * 50.0).max(0.0) * mean,
                description: format!(
                    "'{}': High uniformity detected ({:.0}% avg confidence, {:.3} variance)",
                    tag,
                    mean * 100.0,
                    variance
                ),
                suggestion: "Consider seeking diverse perspectives or questioning shared assumptions.".to_string(),
            });
        }

        // Check for rapid adoption (many beliefs in short time)
        if confidences.len() >= 5 && mean > 0.6 {
            warnings.push(HerdingWarning {
                topic: tag.clone(),
                warning_type: HerdingType::RapidAdoption,
                severity: 0.5,
                description: format!(
                    "'{}': Rapid adoption pattern ({} beliefs with {:.0}% avg confidence)",
                    tag,
                    confidences.len(),
                    mean * 100.0
                ),
                suggestion: "Ensure independent verification rather than following the crowd.".to_string(),
            });
        }
    }

    warnings.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    warnings.truncate(5);
    warnings
}

fn calculate_overall_trend(window: &[&BeliefWithTimestamp], all: &[BeliefWithTimestamp]) -> TrendDirection {
    if all.is_empty() {
        return TrendDirection::Stable;
    }

    let window_ratio = window.len() as f64 / all.len().max(1) as f64;

    // Calculate embedding diversity in window
    let diversity = if window.len() >= 2 {
        let embeddings: Vec<_> = window.iter().filter(|b| !b.embedding.is_empty()).collect();
        if embeddings.len() >= 2 {
            let mut total_sim = 0.0;
            let mut count = 0;
            for i in 0..embeddings.len().min(10) {
                for j in (i + 1)..embeddings.len().min(10) {
                    total_sim += cosine_similarity(&embeddings[i].embedding, &embeddings[j].embedding);
                    count += 1;
                }
            }
            if count > 0 { 1.0 - (total_sim / count as f64) } else { 0.5 }
        } else {
            0.5
        }
    } else {
        0.5
    };

    if window_ratio < 0.1 {
        TrendDirection::Declining
    } else if diversity < 0.3 {
        TrendDirection::Converging
    } else if diversity > 0.6 {
        TrendDirection::Diverging
    } else {
        TrendDirection::Stable
    }
}

/// Get belief lifecycle analysis for a specific topic
#[hdk_extern]
pub fn get_belief_lifecycle(tag: String) -> ExternResult<BeliefLifecycle> {
    let all_beliefs = get_all_beliefs_with_timestamps()?;

    let tagged_beliefs: Vec<_> = all_beliefs
        .iter()
        .filter(|b| b.tags.contains(&tag))
        .collect();

    if tagged_beliefs.is_empty() {
        return Ok(BeliefLifecycle {
            tag: tag.clone(),
            total_beliefs: 0,
            lifecycle_stage: LifecycleStage::Nascent,
            first_appearance: None,
            last_activity: None,
            growth_phases: Vec::new(),
            current_activity_level: 0.0,
        });
    }

    let first = tagged_beliefs.first().unwrap().shared_at;
    let last = tagged_beliefs.last().unwrap().shared_at;

    // Determine lifecycle stage based on activity pattern
    let now = sys_time()?;
    let days_since_last = (now.as_micros() - last.as_micros()) / (86400 * 1_000_000);
    let total_days = (last.as_micros() - first.as_micros()).max(1) / (86400 * 1_000_000);
    let beliefs_per_day = tagged_beliefs.len() as f64 / total_days.max(1) as f64;

    let lifecycle_stage = if days_since_last > 30 {
        LifecycleStage::Dormant
    } else if tagged_beliefs.len() < 3 {
        LifecycleStage::Nascent
    } else if beliefs_per_day > 1.0 {
        LifecycleStage::Rapid
    } else if beliefs_per_day > 0.1 {
        LifecycleStage::Mature
    } else {
        LifecycleStage::Stable
    };

    Ok(BeliefLifecycle {
        tag,
        total_beliefs: tagged_beliefs.len() as u32,
        lifecycle_stage,
        first_appearance: Some(first),
        last_activity: Some(last),
        growth_phases: Vec::new(), // Would need more sophisticated analysis
        current_activity_level: beliefs_per_day,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BeliefLifecycle {
    pub tag: String,
    pub total_beliefs: u32,
    pub lifecycle_stage: LifecycleStage,
    pub first_appearance: Option<Timestamp>,
    pub last_activity: Option<Timestamp>,
    pub growth_phases: Vec<GrowthPhase>,
    pub current_activity_level: f64, // Beliefs per day
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LifecycleStage {
    /// Just emerging (< 3 beliefs)
    Nascent,
    /// Rapid growth phase
    Rapid,
    /// Steady state with regular activity
    Mature,
    /// Stable with occasional updates
    Stable,
    /// No recent activity
    Dormant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrowthPhase {
    pub start: Timestamp,
    pub end: Timestamp,
    pub phase_type: String,
    pub belief_count: u32,
}
