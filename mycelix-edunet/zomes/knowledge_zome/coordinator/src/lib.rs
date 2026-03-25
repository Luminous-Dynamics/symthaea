// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Knowledge Roots Coordinator Zome
//!
//! Implements the decentralized curriculum graph - community-built learning pathways.
//!
//! ## Core Functionality
//!
//! - **Node Management**: Create, update, deprecate knowledge nodes
//! - **Edge Governance**: Propose, vote on, and manage curriculum connections
//! - **Path Discovery**: Find optimal learning paths through the graph
//! - **Progress Tracking**: Track learner progress on nodes (private)
//! - **AI Recommendations**: Generate personalized learning paths

use hdk::prelude::*;
use hdk::prelude::HdkPathExt;
use knowledge_integrity::{
    EntryTypes, LinkTypes, KnowledgeNode, LearningEdge, LearningPath,
    SkillTree, NodeProgress, EdgeVote, PathRecommendation,
    DifficultyLevel, EdgeType, EdgeStatus, NodeStatus,
    ProgressStatus, VoteDirection,
};

// Helper function to ensure a path exists and return its entry hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed_path = path.typed(link_type)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

// Helper function to convert timestamp to i64 (microseconds)
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

// ============== Knowledge Node Functions ==============

/// Create a new knowledge node
#[hdk_extern]
pub fn create_node(node: KnowledgeNode) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::KnowledgeNode(node.clone()))?;

    // Create path anchor for all nodes
    let path = Path::from("all_nodes");
    let path_hash = ensure_path(path, LinkTypes::AllNodes)?;
    create_link(path_hash, action_hash.clone(), LinkTypes::AllNodes, ())?;

    // Link by domain
    let domain_path = Path::from(format!("domain/{}", node.domain));
    let domain_hash = ensure_path(domain_path, LinkTypes::DomainToNodes)?;
    create_link(domain_hash, action_hash.clone(), LinkTypes::DomainToNodes, ())?;

    // Link to related courses
    for course_hash in &node.related_courses {
        create_link(
            action_hash.clone(),
            course_hash.clone(),
            LinkTypes::NodeToCourses,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Get a knowledge node by its action hash
#[hdk_extern]
pub fn get_node(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// List all knowledge nodes
#[hdk_extern]
pub fn list_nodes(_: ()) -> ExternResult<Vec<Record>> {
    let path = Path::from("all_nodes");
    let path_hash = ensure_path(path, LinkTypes::AllNodes)?;

    let links = get_links(
        LinkQuery::try_new(path_hash, LinkTypes::AllNodes)?,
        GetStrategy::Local
    )?;

    let mut nodes = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            nodes.push(record);
        }
    }

    Ok(nodes)
}

/// Get nodes by domain
#[hdk_extern]
pub fn get_nodes_by_domain(domain: String) -> ExternResult<Vec<Record>> {
    let domain_path = Path::from(format!("domain/{}", domain));
    let domain_hash = ensure_path(domain_path, LinkTypes::DomainToNodes)?;

    let links = get_links(
        LinkQuery::try_new(domain_hash, LinkTypes::DomainToNodes)?,
        GetStrategy::Local
    )?;

    let mut nodes = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            nodes.push(record);
        }
    }

    Ok(nodes)
}

/// Search nodes by tags
#[hdk_extern]
pub fn search_nodes(query: SearchNodesInput) -> ExternResult<Vec<Record>> {
    // Get all nodes and filter
    // In production, would use a more efficient search index
    let all_nodes = list_nodes(())?;

    let filtered: Vec<Record> = all_nodes
        .into_iter()
        .filter(|record| {
            if let Some(node) = record.entry().to_app_option::<KnowledgeNode>().ok().flatten() {
                // Match by tag
                if let Some(ref tag) = query.tag {
                    if !node.tags.iter().any(|t| t.to_lowercase().contains(&tag.to_lowercase())) {
                        return false;
                    }
                }
                // Match by difficulty
                if let Some(ref diff) = query.difficulty {
                    if &node.difficulty != diff {
                        return false;
                    }
                }
                // Match by domain
                if let Some(ref domain) = query.domain {
                    if !node.domain.to_lowercase().contains(&domain.to_lowercase()) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        })
        .collect();

    Ok(filtered)
}

/// Update node status (for governance)
#[hdk_extern]
pub fn update_node_status(input: UpdateNodeStatusInput) -> ExternResult<ActionHash> {
    let record = get(input.node_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Node not found"))?;

    let mut node: KnowledgeNode = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Failed to deserialize node"))?;

    node.status = input.new_status;
    node.modified_at = timestamp_to_i64(sys_time()?);
    node.version += 1;

    update_entry(input.node_hash, EntryTypes::KnowledgeNode(node))
}

// ============== Learning Edge Functions ==============

/// Propose a new edge between nodes
#[hdk_extern]
pub fn propose_edge(edge: LearningEdge) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::LearningEdge(edge.clone()))?;

    // Link from source node to edge
    create_link(
        edge.source_node.clone(),
        action_hash.clone(),
        LinkTypes::NodeToEdges,
        (),
    )?;

    // Link from target node to edge (as prerequisite)
    create_link(
        edge.target_node.clone(),
        action_hash.clone(),
        LinkTypes::NodeToPrerequisites,
        (),
    )?;

    Ok(action_hash)
}

/// Get edges from a node
#[hdk_extern]
pub fn get_node_edges(node_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(node_hash, LinkTypes::NodeToEdges)?,
        GetStrategy::Local
    )?;

    let mut edges = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            edges.push(record);
        }
    }

    Ok(edges)
}

/// Get prerequisites for a node
#[hdk_extern]
pub fn get_node_prerequisites(node_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(node_hash, LinkTypes::NodeToPrerequisites)?,
        GetStrategy::Local
    )?;

    let mut prereqs = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!("Failed to convert link target"))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            // Get the edge to find the source node
            if let Ok(Some(edge)) = record.entry().to_app_option::<LearningEdge>() {
                if let Some(node_record) = get(edge.source_node, GetOptions::default())? {
                    prereqs.push(node_record);
                }
            }
        }
    }

    Ok(prereqs)
}

/// Vote on an edge proposal
#[hdk_extern]
pub fn vote_on_edge(vote: EdgeVote) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Ensure the voter is the caller
    if vote.voter != caller {
        return Err(wasm_error!("Can only vote as yourself"));
    }

    let action_hash = create_entry(EntryTypes::EdgeVote(vote.clone()))?;

    // Update edge vote counts
    let edge_record = get(vote.edge_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!("Edge not found"))?;

    let mut edge: LearningEdge = edge_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!("Failed to deserialize edge"))?;

    match vote.vote {
        VoteDirection::Up => edge.upvotes += 1,
        VoteDirection::Down => edge.downvotes += 1,
        VoteDirection::Abstain => {}
    }

    // Determine new edge status using pure function
    edge.status = determine_edge_status(edge.upvotes, edge.downvotes, edge.status.clone());

    update_entry(vote.edge_hash, EntryTypes::LearningEdge(edge))?;

    Ok(action_hash)
}

// ============== Pure Business Logic (HDK-free, unit-testable) ==============

use std::collections::{HashMap, VecDeque, HashSet};

/// Find shortest path via BFS on a pre-built adjacency list.
///
/// Returns `Some(path)` including both start and end, or `None` if unreachable.
/// If `start == end`, returns `Some(vec![start])`.
pub fn find_path_bfs(
    adjacency: &HashMap<ActionHash, Vec<ActionHash>>,
    start: &ActionHash,
    end: &ActionHash,
) -> Option<Vec<ActionHash>> {
    if start == end {
        return Some(vec![start.clone()]);
    }

    let mut visited = HashSet::new();
    let mut queue: VecDeque<(ActionHash, Vec<ActionHash>)> = VecDeque::new();
    queue.push_back((start.clone(), vec![start.clone()]));
    visited.insert(start.clone());

    while let Some((current, path)) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&current) {
            for neighbor in neighbors {
                if neighbor == end {
                    let mut result = path.clone();
                    result.push(neighbor.clone());
                    return Some(result);
                }
                if visited.insert(neighbor.clone()) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    queue.push_back((neighbor.clone(), new_path));
                }
            }
        }
    }

    None
}

/// Determine edge status after a vote is applied.
///
/// Rules:
/// - Needs >= 5 total votes (upvotes + downvotes) to change status
/// - Approval: ratio >= 0.6 and total >= 5
/// - Rejection: ratio < 0.4 and total >= 10
/// - Otherwise: remains in current status
pub fn determine_edge_status(
    upvotes: u32,
    downvotes: u32,
    current_status: EdgeStatus,
) -> EdgeStatus {
    let total = upvotes + downvotes;
    if total >= 5 {
        let approval_ratio = upvotes as f64 / total as f64;
        if approval_ratio >= 0.6 {
            return EdgeStatus::Approved;
        } else if approval_ratio < 0.4 && total >= 10 {
            return EdgeStatus::Rejected;
        }
    }
    current_status
}

// ============== Learning Path Functions ==============

/// Create a learning path
#[hdk_extern]
pub fn create_path(path: LearningPath) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::LearningPath(path.clone()))?;

    // Create path anchor
    let anchor = Path::from("all_paths");
    let anchor_hash = ensure_path(anchor, LinkTypes::AllPaths)?;
    create_link(anchor_hash, action_hash.clone(), LinkTypes::AllPaths, ())?;

    // Link path to nodes
    for node_hash in &path.nodes {
        create_link(
            action_hash.clone(),
            node_hash.clone(),
            LinkTypes::PathToNodes,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Get a learning path
#[hdk_extern]
pub fn get_path(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// List all learning paths
#[hdk_extern]
pub fn list_paths(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = Path::from("all_paths");
    let anchor_hash = anchor.path_entry_hash()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::AllPaths)?,
        GetStrategy::Local
    )?;

    let mut paths = Vec::new();
    for link in links {
        if let Ok(action_hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                paths.push(record);
            }
        }
    }

    Ok(paths)
}

/// Find optimal path between two nodes
#[hdk_extern]
pub fn find_path(input: FindPathInput) -> ExternResult<Vec<ActionHash>> {
    // Simple BFS path finding
    // In production, would use more sophisticated algorithms
    // considering edge strength, learner progress, etc.

    let mut visited: Vec<ActionHash> = vec![];
    let mut queue: Vec<(ActionHash, Vec<ActionHash>)> = vec![(input.start_node.clone(), vec![input.start_node.clone()])];

    while let Some((current, path)) = queue.pop() {
        if current == input.end_node {
            return Ok(path);
        }

        if visited.contains(&current) {
            continue;
        }
        visited.push(current.clone());

        // Get outgoing edges
        let edges = get_node_edges(current.clone())?;
        for edge_record in edges {
            if let Ok(Some(edge)) = edge_record.entry().to_app_option::<LearningEdge>() {
                // Only follow approved edges
                if edge.status == EdgeStatus::Approved {
                    let mut new_path = path.clone();
                    new_path.push(edge.target_node.clone());
                    queue.push((edge.target_node, new_path));
                }
            }
        }
    }

    // No path found
    Ok(vec![])
}

// ============== Progress Tracking Functions ==============

/// Update progress on a node (private entry)
#[hdk_extern]
pub fn update_node_progress(progress: NodeProgress) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Verify caller is the learner
    if progress.learner != caller {
        return Err(wasm_error!("Can only update your own progress"));
    }

    let action_hash = create_entry(EntryTypes::NodeProgress(progress.clone()))?;

    // Link node -> progress
    create_link(
        progress.node_hash.clone(),
        action_hash.clone(),
        LinkTypes::NodeToProgress,
        (),
    )?;

    // Link learner -> progress
    create_link(
        progress.learner,
        action_hash.clone(),
        LinkTypes::LearnerToProgress,
        (),
    )?;

    Ok(action_hash)
}

/// Get my progress across all nodes
#[hdk_extern]
pub fn get_my_progress(_: ()) -> ExternResult<Vec<Record>> {
    let caller = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        LinkQuery::try_new(caller, LinkTypes::LearnerToProgress)?,
        GetStrategy::Local
    )?;

    let mut progress = Vec::new();
    for link in links {
        if let Ok(action_hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                progress.push(record);
            }
        }
    }

    Ok(progress)
}

/// Check if prerequisites are met for a node
#[hdk_extern]
pub fn check_prerequisites(node_hash: ActionHash) -> ExternResult<PrerequisiteCheck> {
    let _caller = agent_info()?.agent_initial_pubkey;

    // Get prerequisites for the node
    let prereq_edges = get_links(
        LinkQuery::try_new(node_hash.clone(), LinkTypes::NodeToPrerequisites)?,
        GetStrategy::Local
    )?;

    let mut required: Vec<ActionHash> = vec![];
    let met: Vec<ActionHash> = vec![];
    let mut unmet: Vec<ActionHash> = vec![];

    for link in prereq_edges {
        if let Ok(edge_hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(edge_hash, GetOptions::default())? {
                if let Ok(Some(edge)) = record.entry().to_app_option::<LearningEdge>() {
                    // Only consider "Requires" edges
                    if edge.edge_type == EdgeType::Requires && edge.status == EdgeStatus::Approved {
                        required.push(edge.source_node.clone());

                        // Check if learner has completed this prerequisite
                        // (Simplified - would check actual progress)
                        unmet.push(edge.source_node);
                    }
                }
            }
        }
    }

    Ok(PrerequisiteCheck {
        node_hash,
        all_met: unmet.is_empty(),
        required,
        met,
        unmet,
    })
}

// ============== Recommendation Functions ==============

/// Generate a learning path recommendation
#[hdk_extern]
pub fn generate_recommendation(input: RecommendationInput) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_initial_pubkey;

    // Get learner's current progress
    let progress_records = get_my_progress(())?;
    let completed_nodes: Vec<ActionHash> = progress_records
        .iter()
        .filter_map(|r| {
            if let Some(p) = r.entry().to_app_option::<NodeProgress>().ok().flatten() {
                if p.progress_status == ProgressStatus::Completed ||
                   p.progress_status == ProgressStatus::Mastered {
                    Some(p.node_hash)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // Find nodes related to target skill
    // (Simplified - would use ML model in production)
    let all_nodes = list_nodes(())?;
    let relevant_nodes: Vec<ActionHash> = all_nodes
        .iter()
        .filter_map(|r| {
            if let Some(node) = r.entry().to_app_option::<KnowledgeNode>().ok().flatten() {
                if node.domain.to_lowercase().contains(&input.target_skill.to_lowercase()) ||
                   node.tags.iter().any(|t| t.to_lowercase().contains(&input.target_skill.to_lowercase())) {
                    // Get the action hash from the record
                    r.action_hashed().hash.clone().into()
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // Create recommendation
    let recommendation = PathRecommendation {
        learner: caller,
        target: input.target_skill,
        recommended_nodes: relevant_nodes.clone(),
        confidence_permille: 700, // Would be computed by ML model (700 = 70%)
        reasoning: "Based on your current progress and target skill".to_string(),
        current_progress: completed_nodes,
        estimated_hours: relevant_nodes.len() as u32 * 5, // Rough estimate
        generated_at: timestamp_to_i64(sys_time()?),
        model_version: "simple-v1".to_string(),
    };

    create_entry(EntryTypes::PathRecommendation(recommendation))
}

// ============== Skill Tree Functions ==============

/// Create a skill tree
#[hdk_extern]
pub fn create_skill_tree(tree: SkillTree) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::SkillTree(tree.clone()))?;

    // Create anchor
    let anchor = Path::from("all_skill_trees");
    let anchor_hash = ensure_path(anchor, LinkTypes::AllSkillTrees)?;
    create_link(anchor_hash, action_hash.clone(), LinkTypes::AllSkillTrees, ())?;

    // Link to all nodes in the tree
    for tier in &tree.structure.tiers {
        for node_hash in &tier.nodes {
            create_link(
                action_hash.clone(),
                node_hash.clone(),
                LinkTypes::SkillTreeToNodes,
                (),
            )?;
        }
    }

    Ok(action_hash)
}

/// Get a skill tree
#[hdk_extern]
pub fn get_skill_tree(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// List all skill trees
#[hdk_extern]
pub fn list_skill_trees(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = Path::from("all_skill_trees");
    let anchor_hash = anchor.path_entry_hash()?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash, LinkTypes::AllSkillTrees)?,
        GetStrategy::Local
    )?;

    let mut trees = Vec::new();
    for link in links {
        if let Ok(action_hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                trees.push(record);
            }
        }
    }

    Ok(trees)
}

// ============== Input/Output Types ==============

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchNodesInput {
    pub tag: Option<String>,
    pub domain: Option<String>,
    pub difficulty: Option<DifficultyLevel>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateNodeStatusInput {
    pub node_hash: ActionHash,
    pub new_status: NodeStatus,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FindPathInput {
    pub start_node: ActionHash,
    pub end_node: ActionHash,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PrerequisiteCheck {
    pub node_hash: ActionHash,
    pub all_met: bool,
    pub required: Vec<ActionHash>,
    pub met: Vec<ActionHash>,
    pub unmet: Vec<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecommendationInput {
    pub target_skill: String,
    pub max_hours: Option<u32>,
    pub preferred_difficulty: Option<DifficultyLevel>,
}

// ============================================================================
// Tests -- pure business logic only (no HDK required)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a fake ActionHash from a byte for test convenience
    fn fake_hash(id: u8) -> ActionHash {
        ActionHash::from_raw_36(vec![id; 36])
    }

    // ---- find_path_bfs ----

    #[test]
    fn test_bfs_simple_path() {
        // A -> B -> C
        let a = fake_hash(1);
        let b = fake_hash(2);
        let c = fake_hash(3);

        let mut adj = HashMap::new();
        adj.insert(a.clone(), vec![b.clone()]);
        adj.insert(b.clone(), vec![c.clone()]);

        let path = find_path_bfs(&adj, &a, &c);
        assert_eq!(path, Some(vec![a.clone(), b.clone(), c.clone()]));
    }

    #[test]
    fn test_bfs_no_path() {
        // A -> B, C is disconnected
        let a = fake_hash(1);
        let b = fake_hash(2);
        let c = fake_hash(3);

        let mut adj = HashMap::new();
        adj.insert(a.clone(), vec![b.clone()]);

        let path = find_path_bfs(&adj, &a, &c);
        assert_eq!(path, None);
    }

    #[test]
    fn test_bfs_start_equals_end() {
        let a = fake_hash(1);
        let adj = HashMap::new();

        let path = find_path_bfs(&adj, &a, &a);
        assert_eq!(path, Some(vec![a.clone()]));
    }

    #[test]
    fn test_bfs_cycle_handling() {
        // A -> B -> C -> A (cycle), target D not reachable
        let a = fake_hash(1);
        let b = fake_hash(2);
        let c = fake_hash(3);
        let d = fake_hash(4);

        let mut adj = HashMap::new();
        adj.insert(a.clone(), vec![b.clone()]);
        adj.insert(b.clone(), vec![c.clone()]);
        adj.insert(c.clone(), vec![a.clone()]); // cycle back

        let path = find_path_bfs(&adj, &a, &d);
        assert_eq!(path, None);
    }

    #[test]
    fn test_bfs_shortest_path() {
        // A -> B -> D (length 2)
        // A -> C -> D (length 2)
        // A -> D     (length 1, shortest)
        let a = fake_hash(1);
        let b = fake_hash(2);
        let c = fake_hash(3);
        let d = fake_hash(4);

        let mut adj = HashMap::new();
        adj.insert(a.clone(), vec![b.clone(), c.clone(), d.clone()]);
        adj.insert(b.clone(), vec![d.clone()]);
        adj.insert(c.clone(), vec![d.clone()]);

        let path = find_path_bfs(&adj, &a, &d);
        assert_eq!(path, Some(vec![a.clone(), d.clone()]));
    }

    #[test]
    fn test_bfs_cycle_with_reachable_target() {
        // A -> B -> C -> A (cycle), but B -> D exists
        let a = fake_hash(1);
        let b = fake_hash(2);
        let c = fake_hash(3);
        let d = fake_hash(4);

        let mut adj = HashMap::new();
        adj.insert(a.clone(), vec![b.clone()]);
        adj.insert(b.clone(), vec![c.clone(), d.clone()]);
        adj.insert(c.clone(), vec![a.clone()]);

        let path = find_path_bfs(&adj, &a, &d);
        assert_eq!(path, Some(vec![a.clone(), b.clone(), d.clone()]));
    }

    #[test]
    fn test_bfs_empty_graph() {
        let a = fake_hash(1);
        let b = fake_hash(2);
        let adj = HashMap::new();

        let path = find_path_bfs(&adj, &a, &b);
        assert_eq!(path, None);
    }

    // ---- determine_edge_status ----

    #[test]
    fn test_edge_approved_at_threshold() {
        // 3 up, 2 down = 5 total, ratio 0.6 => Approved
        let status = determine_edge_status(3, 2, EdgeStatus::Proposed);
        assert_eq!(status, EdgeStatus::Approved);
    }

    #[test]
    fn test_edge_stays_proposed_below_quorum() {
        // 2 up, 1 down = 3 total < 5 => stays Proposed
        let status = determine_edge_status(2, 1, EdgeStatus::Proposed);
        assert_eq!(status, EdgeStatus::Proposed);
    }

    #[test]
    fn test_edge_rejected_at_threshold() {
        // 3 up, 7 down = 10 total, ratio 0.3 < 0.4 and total >= 10 => Rejected
        let status = determine_edge_status(3, 7, EdgeStatus::Proposed);
        assert_eq!(status, EdgeStatus::Rejected);
    }

    #[test]
    fn test_edge_not_rejected_below_10_total() {
        // 1 up, 4 down = 5 total, ratio 0.2 < 0.4 but total < 10 => stays Proposed
        let status = determine_edge_status(1, 4, EdgeStatus::Proposed);
        assert_eq!(status, EdgeStatus::Proposed);
    }

    #[test]
    fn test_edge_indeterminate_middle_ratio() {
        // 4 up, 6 down = 10 total, ratio 0.4 (not < 0.4, not >= 0.6) => stays current
        let status = determine_edge_status(4, 6, EdgeStatus::Proposed);
        assert_eq!(status, EdgeStatus::Proposed);
    }

    #[test]
    fn test_edge_zero_votes() {
        let status = determine_edge_status(0, 0, EdgeStatus::Proposed);
        assert_eq!(status, EdgeStatus::Proposed);
    }
}
