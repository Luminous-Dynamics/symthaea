// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Graph Coordinator Zome
//! Business logic for knowledge graph relationships and traversal
//!
//! Updated to use HDK 0.6 patterns (LinkQuery, GetStrategy, Anchor pattern)

use hdk::prelude::*;
use graph_integrity::*;

/// Helper to get an anchor entry hash for link bases
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Create a relationship between two claims
#[hdk_extern]
pub fn create_relationship(relationship: Relationship) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Relationship(relationship.clone()))?;

    // Create anchors and links for source claim
    let source_anchor = format!("claim:{}", relationship.source);
    create_entry(&EntryTypes::Anchor(Anchor(source_anchor.clone())))?;
    create_link(
        anchor_hash(&source_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimToOutgoingRelation,
        (),
    )?;

    // Create anchors and links for target claim
    let target_anchor = format!("claim:{}", relationship.target);
    create_entry(&EntryTypes::Anchor(Anchor(target_anchor.clone())))?;
    create_link(
        anchor_hash(&target_anchor)?,
        action_hash.clone(),
        LinkTypes::ClaimToIncomingRelation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find relationship".into()
        )))
}

/// Get outgoing relationships from a claim
#[hdk_extern]
pub fn get_outgoing_relationships(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&claim_anchor)?,
            LinkTypes::ClaimToOutgoingRelation,
        )?,
        GetStrategy::default(),
    )?;

    let mut relationships = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            relationships.push(record);
        }
    }

    Ok(relationships)
}

/// Get incoming relationships to a claim
#[hdk_extern]
pub fn get_incoming_relationships(claim_id: String) -> ExternResult<Vec<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&claim_anchor)?,
            LinkTypes::ClaimToIncomingRelation,
        )?,
        GetStrategy::default(),
    )?;

    let mut relationships = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            relationships.push(record);
        }
    }

    Ok(relationships)
}

/// Find path between two claims
#[hdk_extern]
pub fn find_path(input: FindPathInput) -> ExternResult<Vec<String>> {
    // Simple BFS for path finding
    let mut visited: Vec<String> = Vec::new();
    let mut queue: Vec<(String, Vec<String>)> = vec![(input.source.clone(), vec![input.source.clone()])];

    while let Some((current, path)) = queue.pop() {
        if current == input.target {
            return Ok(path);
        }

        if visited.contains(&current) || path.len() > input.max_depth as usize {
            continue;
        }

        visited.push(current.clone());

        // Get outgoing relationships
        let claim_anchor = format!("claim:{}", current);
        let links = get_links(
            LinkQuery::try_new(
                anchor_hash(&claim_anchor)?,
                LinkTypes::ClaimToOutgoingRelation,
            )?,
            GetStrategy::default(),
        )?;

        for link in links {
            let action_hash = ActionHash::try_from(link.target).map_err(|_| {
                wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
            })?;

            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(rel) = record
                    .entry()
                    .to_app_option::<Relationship>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                {
                    if !visited.contains(&rel.target) {
                        let mut new_path = path.clone();
                        new_path.push(rel.target.clone());
                        queue.push((rel.target, new_path));
                    }
                }
            }
        }
    }

    // No path found
    Ok(Vec::new())
}

/// Input for path finding
#[derive(Serialize, Deserialize, Debug)]
pub struct FindPathInput {
    pub source: String,
    pub target: String,
    pub max_depth: u32,
}

/// Create an ontology
#[hdk_extern]
pub fn create_ontology(ontology: Ontology) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Ontology(ontology))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find ontology".into()
        )))
}

/// Create a concept in an ontology
#[hdk_extern]
pub fn create_concept(concept: Concept) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Concept(concept.clone()))?;

    // Create anchor and link ontology to concept
    let ontology_anchor = format!("ontology:{}", concept.ontology_id);
    create_entry(&EntryTypes::Anchor(Anchor(ontology_anchor.clone())))?;
    create_link(
        anchor_hash(&ontology_anchor)?,
        action_hash.clone(),
        LinkTypes::OntologyToConcept,
        (),
    )?;

    // Link parent to child if parent exists
    if let Some(ref parent) = concept.parent {
        let parent_anchor = format!("concept:{}", parent);
        create_entry(&EntryTypes::Anchor(Anchor(parent_anchor.clone())))?;
        create_link(
            anchor_hash(&parent_anchor)?,
            action_hash.clone(),
            LinkTypes::ConceptToChild,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find concept".into()
        )))
}

/// Get concepts in an ontology
#[hdk_extern]
pub fn get_ontology_concepts(ontology_id: String) -> ExternResult<Vec<Record>> {
    let ontology_anchor = format!("ontology:{}", ontology_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&ontology_anchor)?, LinkTypes::OntologyToConcept)?,
        GetStrategy::default(),
    )?;

    let mut concepts = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            concepts.push(record);
        }
    }

    Ok(concepts)
}

/// Get child concepts
#[hdk_extern]
pub fn get_child_concepts(concept_id: String) -> ExternResult<Vec<Record>> {
    let concept_anchor = format!("concept:{}", concept_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&concept_anchor)?, LinkTypes::ConceptToChild)?,
        GetStrategy::default(),
    )?;

    let mut children = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            children.push(record);
        }
    }

    Ok(children)
}

/// Get graph statistics
#[hdk_extern]
pub fn get_graph_stats(_: ()) -> ExternResult<GraphStats> {
    let claim_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Relationship,
        )?));

    let relationships = query(claim_filter)?;

    Ok(GraphStats {
        relationship_count: relationships.len() as u64,
    })
}

/// Graph statistics
#[derive(Serialize, Deserialize, Debug)]
pub struct GraphStats {
    pub relationship_count: u64,
}

// ============================================================================
// BELIEF PROPAGATION FUNCTIONS
// ============================================================================

/// Propagate belief updates through the graph
///
/// Uses iterative relaxation to propagate belief changes through connected claims.
#[hdk_extern]
pub fn propagate_belief(claim_id: String) -> ExternResult<PropagationResult> {
    let start_time = sys_time()?;
    let max_iterations = 100;
    let convergence_threshold = 0.001;

    let mut nodes_affected: Vec<String> = vec![];
    let mut max_delta = 1.0;
    let mut iterations = 0;

    // Get initial belief from source claim
    let source_belief = get_or_create_belief_node(&claim_id)?;
    nodes_affected.push(claim_id.clone());

    // BFS through connected claims
    let mut to_process: Vec<String> = vec![claim_id.clone()];
    let mut processed: Vec<String> = vec![];

    while !to_process.is_empty() && iterations < max_iterations && max_delta > convergence_threshold {
        iterations += 1;
        max_delta = 0.0;

        let current_batch = to_process.clone();
        to_process.clear();

        for current_id in current_batch {
            if processed.contains(&current_id) {
                continue;
            }
            processed.push(current_id.clone());

            // Get outgoing relationships
            let outgoing = get_outgoing_relationships(current_id.clone())?;

            for rel_record in outgoing {
                if let Some(rel) = rel_record
                    .entry()
                    .to_app_option::<Relationship>()
                    .ok()
                    .flatten()
                {
                    // Get target belief node
                    let mut target_belief = get_or_create_belief_node(&rel.target)?;
                    let old_belief = target_belief.belief_strength;

                    // Calculate new belief based on relationship type and weight
                    let current_belief = get_belief_strength(&current_id)?;
                    let influence = calculate_influence(&rel.relationship_type, rel.weight, current_belief);

                    // Update target belief
                    let new_belief = update_belief(target_belief.belief_strength, influence);
                    let delta = (new_belief - old_belief).abs();

                    if delta > 0.0001 {
                        target_belief.belief_strength = new_belief;
                        target_belief.propagation_iterations = iterations;
                        target_belief.last_updated = sys_time()?;
                        target_belief.converged = delta < convergence_threshold;

                        // Add influence record
                        target_belief.influences.push(BeliefInfluence {
                            source_claim_id: current_id.clone(),
                            influence_type: relationship_to_influence(&rel.relationship_type),
                            weight: rel.weight,
                            source_belief: current_belief,
                        });

                        // Update or create belief node
                        update_belief_node(&target_belief)?;

                        if !nodes_affected.contains(&rel.target) {
                            nodes_affected.push(rel.target.clone());
                        }

                        max_delta = max_delta.max(delta);
                        to_process.push(rel.target.clone());
                    }
                }
            }
        }
    }

    let end_time = sys_time()?;
    let processing_time_ms = (end_time.as_micros() - start_time.as_micros()) / 1000;

    Ok(PropagationResult {
        source_claim_id: claim_id,
        nodes_affected: nodes_affected.len() as u32,
        iterations,
        converged: max_delta <= convergence_threshold,
        max_delta,
        processing_time_ms: processing_time_ms as u64,
        updated_nodes: nodes_affected,
    })
}

/// Get or create a belief node for a claim
fn get_or_create_belief_node(claim_id: &str) -> ExternResult<BeliefNode> {
    let claim_anchor = format!("claim:{}", claim_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToBeliefNode)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(target) = link.target.clone().into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                if let Some(node) = record
                    .entry()
                    .to_app_option::<BeliefNode>()
                    .ok()
                    .flatten()
                {
                    return Ok(node);
                }
            }
        }
    }

    // Create new belief node with default prior
    let now = sys_time()?;
    let node = BeliefNode {
        id: format!("bn_{}", claim_id),
        claim_id: claim_id.to_string(),
        belief_strength: 0.5, // Default prior
        prior_belief: 0.5,
        confidence: 0.5,
        support_count: 0,
        contradiction_count: 0,
        last_updated: now,
        propagation_iterations: 0,
        converged: false,
        influences: vec![],
    };

    let action_hash = create_entry(EntryTypes::BeliefNode(node.clone()))?;

    // Create anchor and link
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(anchor_hash(&claim_anchor)?, action_hash, LinkTypes::ClaimToBeliefNode, ())?;

    Ok(node)
}

/// Get belief strength for a claim
fn get_belief_strength(claim_id: &str) -> ExternResult<f64> {
    let node = get_or_create_belief_node(claim_id)?;
    Ok(node.belief_strength)
}

/// Calculate influence based on relationship type
fn calculate_influence(rel_type: &RelationshipType, weight: f64, source_belief: f64) -> f64 {
    match rel_type {
        RelationshipType::Supports => source_belief * weight,
        RelationshipType::Contradicts => (1.0 - source_belief) * weight,
        RelationshipType::DerivedFrom => source_belief * weight * 0.8,
        RelationshipType::Causes => source_belief * weight * 0.9,
        RelationshipType::Equivalent => source_belief * weight,
        _ => source_belief * weight * 0.5,
    }
}

/// Update belief using weighted average
fn update_belief(current: f64, influence: f64) -> f64 {
    let learning_rate = 0.3;
    let new_belief = current * (1.0 - learning_rate) + influence * learning_rate;
    new_belief.clamp(0.0, 1.0)
}

/// Convert relationship type to influence type
fn relationship_to_influence(rel_type: &RelationshipType) -> InfluenceType {
    match rel_type {
        RelationshipType::Supports => InfluenceType::Support,
        RelationshipType::Contradicts => InfluenceType::Contradiction,
        RelationshipType::DerivedFrom | RelationshipType::Causes => InfluenceType::Entailment,
        _ => InfluenceType::Evidential,
    }
}

/// Update a belief node
fn update_belief_node(node: &BeliefNode) -> ExternResult<ActionHash> {
    let claim_anchor = format!("claim:{}", node.claim_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&claim_anchor)?, LinkTypes::ClaimToBeliefNode)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(target) = link.target.clone().into_action_hash() {
            return update_entry(target, &EntryTypes::BeliefNode(node.clone()));
        }
    }

    // Create if doesn't exist
    let action_hash = create_entry(EntryTypes::BeliefNode(node.clone()))?;
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(anchor_hash(&claim_anchor)?, action_hash.clone(), LinkTypes::ClaimToBeliefNode, ())?;
    Ok(action_hash)
}

/// Get dependency tree for a claim
#[hdk_extern]
pub fn get_dependency_tree(input: DependencyTreeInput) -> ExternResult<DependencyTree> {
    let mut nodes: Vec<DependencyTreeNode> = vec![];
    let mut total_weight = 0.0;
    let mut max_depth = 0u32;

    // BFS to build tree
    let mut queue: Vec<(String, u32)> = vec![(input.claim_id.clone(), 0)];
    let mut visited: Vec<String> = vec![];

    while let Some((current_id, depth)) = queue.pop() {
        if visited.contains(&current_id) || depth > input.max_depth {
            continue;
        }
        visited.push(current_id.clone());

        if depth > max_depth {
            max_depth = depth;
        }

        // Get outgoing relationships (dependencies)
        let outgoing = get_outgoing_relationships(current_id.clone())?;

        let mut children: Vec<String> = vec![];
        for rel_record in &outgoing {
            if let Some(rel) = rel_record
                .entry()
                .to_app_option::<Relationship>()
                .ok()
                .flatten()
            {
                children.push(rel.target.clone());
                total_weight += rel.weight;
                queue.push((rel.target.clone(), depth + 1));
            }
        }

        nodes.push(DependencyTreeNode {
            claim_id: current_id,
            depth,
            weight: 1.0, // Weight from parent
            children: children.clone(),
            is_leaf: children.is_empty(),
        });
    }

    Ok(DependencyTree {
        root_claim_id: input.claim_id,
        nodes,
        depth: max_depth,
        total_dependencies: visited.len() as u32 - 1, // Exclude root
        aggregate_weight: total_weight,
    })
}

/// Input for dependency tree query
#[derive(Serialize, Deserialize, Debug)]
pub struct DependencyTreeInput {
    pub claim_id: String,
    pub max_depth: u32,
}

/// Rank claims by information value (priority for verification)
#[hdk_extern]
pub fn rank_by_information_value(limit: u32) -> ExternResult<Vec<Record>> {
    // Query all claims and calculate information value
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::InformationValue,
        )?))
        .include_entries(true);

    let mut records = query(filter)?;

    // Sort by expected value (descending)
    records.sort_by(|a, b| {
        let a_val = a
            .entry()
            .to_app_option::<InformationValue>()
            .ok()
            .flatten()
            .map(|v| v.expected_value)
            .unwrap_or(0.0);
        let b_val = b
            .entry()
            .to_app_option::<InformationValue>()
            .ok()
            .flatten()
            .map(|v| v.expected_value)
            .unwrap_or(0.0);
        b_val.partial_cmp(&a_val).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(records.into_iter().take(limit as usize).collect())
}

/// Calculate and store information value for a claim
#[hdk_extern]
pub fn calculate_information_value(claim_id: String) -> ExternResult<Record> {
    let now = sys_time()?;

    // Get dependency tree to count dependents
    let tree = get_dependency_tree(DependencyTreeInput {
        claim_id: claim_id.clone(),
        max_depth: 5,
    })?;

    // Get incoming relationships (claims that depend on this one)
    let incoming = get_incoming_relationships(claim_id.clone())?;

    // Calculate uncertainty (Shannon entropy of binary belief)
    let belief = get_belief_strength(&claim_id)?;
    let uncertainty = if belief <= 0.0 || belief >= 1.0 {
        // At the extremes (certain true/false), entropy is 0
        0.0
    } else {
        -belief * belief.ln() - (1.0 - belief) * (1.0 - belief).ln()
    };

    // Calculate impact score
    let dependent_count = incoming.len() as u32;
    let avg_weight = if dependent_count > 0 {
        tree.aggregate_weight / dependent_count as f64
    } else {
        0.0
    };

    let impact_score = (dependent_count as f64 * avg_weight * uncertainty).min(1.0);
    let expected_value = impact_score * uncertainty;

    let info_value = InformationValue {
        id: format!("iv_{}_{}", claim_id, now.as_micros()),
        claim_id: claim_id.clone(),
        expected_value,
        dependent_count,
        average_dependency_weight: avg_weight,
        uncertainty,
        impact_score,
        recommended_for_verification: expected_value > 0.5,
        assessed_at: now,
        reasoning: format!(
            "{} dependents, {:.2} uncertainty, {:.2} impact",
            dependent_count, uncertainty, impact_score
        ),
    };

    let action_hash = create_entry(EntryTypes::InformationValue(info_value))?;

    // Create anchor and link to claim
    let claim_anchor = format!("claim:{}", claim_id);
    create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
    create_link(anchor_hash(&claim_anchor)?, action_hash.clone(), LinkTypes::ClaimToInformationValue, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find information value".into()
        )))
}

/// Detect circular dependencies in the graph
#[hdk_extern]
pub fn detect_circular_dependencies(claim_id: String) -> ExternResult<Vec<Vec<String>>> {
    let mut cycles: Vec<Vec<String>> = vec![];

    // DFS with path tracking
    let mut stack: Vec<(String, Vec<String>)> = vec![(claim_id.clone(), vec![claim_id.clone()])];
    let mut visited: Vec<String> = vec![];

    while let Some((current, path)) = stack.pop() {
        // Get outgoing relationships
        let outgoing = get_outgoing_relationships(current.clone())?;

        for rel_record in outgoing {
            if let Some(rel) = rel_record
                .entry()
                .to_app_option::<Relationship>()
                .ok()
                .flatten()
            {
                if let Some(cycle_start) = path.iter().position(|x| x == &rel.target) {
                    // Cycle detected
                    let mut cycle = path[cycle_start..].to_vec();
                    cycle.push(rel.target.clone());
                    cycles.push(cycle);
                } else if !visited.contains(&rel.target) {
                    let mut new_path = path.clone();
                    new_path.push(rel.target.clone());
                    stack.push((rel.target.clone(), new_path));
                }
            }
        }

        visited.push(current);
    }

    Ok(cycles)
}

/// Calculate cascade impact for a claim
#[hdk_extern]
pub fn calculate_cascade_impact(claim_id: String) -> ExternResult<CascadeImpact> {
    let now = sys_time()?;

    let mut affected_by_depth: Vec<u32> = vec![];
    let mut high_impact_claims: Vec<String> = vec![];
    let mut total_affected = 0u32;
    let mut max_depth = 0u32;
    let mut impact_score = 0.0;

    // BFS through dependents
    let mut queue: Vec<(String, u32)> = vec![(claim_id.clone(), 0)];
    let mut visited: Vec<String> = vec![];

    while let Some((current, depth)) = queue.pop() {
        if visited.contains(&current) || depth > 10 {
            continue;
        }
        visited.push(current.clone());

        if depth > 0 {
            total_affected += 1;
            if depth > max_depth {
                max_depth = depth;
            }

            // Ensure vector is long enough
            while affected_by_depth.len() <= depth as usize {
                affected_by_depth.push(0);
            }
            affected_by_depth[depth as usize] += 1;
        }

        // Get incoming relationships (claims that depend on this one)
        let incoming = get_incoming_relationships(current.clone())?;

        for rel_record in incoming {
            if let Some(rel) = rel_record
                .entry()
                .to_app_option::<Relationship>()
                .ok()
                .flatten()
            {
                // Track high-impact connections
                if rel.weight > 0.8 {
                    high_impact_claims.push(rel.source.clone());
                }

                impact_score += rel.weight / (depth as f64 + 1.0);
                queue.push((rel.source.clone(), depth + 1));
            }
        }
    }

    Ok(CascadeImpact {
        claim_id,
        total_affected,
        affected_by_depth,
        max_depth,
        impact_score: impact_score.min(1.0),
        high_impact_claims,
        assessed_at: now,
    })
}
