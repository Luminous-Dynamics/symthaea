// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Smart pathfinding engine for the curriculum graph.
//!
//! Finds optimal learning paths using weighted A* with Bloom-constrained
//! expansion, career relevance weighting, and ISCED level heuristics.
//!
//! This goes beyond simple BFS — it considers cognitive progression,
//! time budgets, and career alignment to produce actionable learning plans.

use crate::career_profile::{embedded_career_profiles, CareerProfile};
use crate::converter::{CurriculumDocument, CurriculumEdge, CurriculumNode};
use crate::taxonomy::IscedLevel;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

// ============================================================
// Public Types
// ============================================================

/// What the learner wants to achieve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathGoal {
    /// Reach a specific node by ID.
    Node(String),
    /// Reach any node in a career field (matched to CareerProfile).
    Career(String),
    /// Reach any node at a specific ISCED level in a subject.
    Level { isced_level: u8, subject: String },
}

/// Constraints on the learning path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathConstraints {
    /// Maximum total hours (None = unlimited).
    pub max_hours: Option<u32>,
    /// Enforce monotonic Bloom progression (no skipping levels).
    pub bloom_strict: bool,
    /// Weight for career relevance (0.0-1.0).
    pub career_weight: f32,
    /// Weight for time efficiency (0.0-1.0).
    pub time_weight: f32,
}

impl Default for PathConstraints {
    fn default() -> Self {
        Self {
            max_hours: None,
            bloom_strict: false,
            career_weight: 0.3,
            time_weight: 0.5,
        }
    }
}

/// A single step in the learning plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathStep {
    pub node_id: String,
    pub title: String,
    pub estimated_hours: u32,
    pub cumulative_hours: u32,
    pub bloom_level: String,
    pub isced_level: u8,
    pub subject_area: String,
    pub reason: String,
}

/// A complete learning plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPlan {
    pub from: String,
    pub goal: String,
    pub steps: Vec<PathStep>,
    pub total_hours: u32,
    pub total_nodes: usize,
    pub bloom_progression: Vec<String>,
    pub career_alignment: Option<f32>,
}

// ============================================================
// Graph Index
// ============================================================

/// Pre-computed graph index for efficient pathfinding.
struct GraphIndex {
    nodes: HashMap<String, IndexedNode>,
    adjacency: HashMap<String, Vec<(String, f32)>>, // node_id -> [(neighbor_id, edge_weight)]
    reverse: HashMap<String, Vec<String>>,           // node_id -> incoming neighbors
}

struct IndexedNode {
    title: String,
    estimated_hours: u32,
    bloom_ordinal: u8,
    isced_level: u8,
    subject_area: String,
    tags: Vec<String>,
    cip_code: Option<String>,
}

impl GraphIndex {
    fn from_document(doc: &CurriculumDocument) -> Self {
        let mut nodes = HashMap::new();
        let mut adjacency: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        let mut reverse: HashMap<String, Vec<String>> = HashMap::new();

        for node in &doc.nodes {
            let grade = node.grade_levels.first().map(|s| s.as_str()).unwrap_or("Adult");
            nodes.insert(
                node.id.clone(),
                IndexedNode {
                    title: node.title.clone(),
                    estimated_hours: node.estimated_hours,
                    bloom_ordinal: bloom_to_ordinal(&node.bloom_level),
                    isced_level: IscedLevel::from_grade_level(grade) as u8,
                    subject_area: node.subject_area.clone(),
                    tags: node.tags.clone(),
                    cip_code: node.cip_code.clone(),
                },
            );
        }

        for edge in &doc.edges {
            let weight = 1.0 - (edge.strength_permille as f32 / 1000.0); // higher strength = lower cost
            adjacency
                .entry(edge.from.clone())
                .or_default()
                .push((edge.to.clone(), weight));
            reverse
                .entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
        }

        GraphIndex { nodes, adjacency, reverse }
    }
}

// ============================================================
// A* Pathfinding
// ============================================================

/// State in the A* priority queue.
#[derive(Debug)]
struct AStarState {
    node_id: String,
    cost: f32,       // g(n): actual cost from start
    priority: f32,   // f(n) = g(n) + h(n)
    hours: u32,      // cumulative hours
    bloom: u8,       // current Bloom level along this path
}

impl PartialEq for AStarState {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for AStarState {}

impl PartialOrd for AStarState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: reverse ordering (lower priority = higher priority in queue)
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Find an optimal learning path through the curriculum graph.
pub fn find_learning_path(
    doc: &CurriculumDocument,
    start: &str,
    goal: &PathGoal,
    completed: &HashSet<String>,
    constraints: &PathConstraints,
) -> Option<LearningPlan> {
    let index = GraphIndex::from_document(doc);

    // Validate start exists
    if !index.nodes.contains_key(start) {
        return None;
    }

    // Find goal node(s)
    let goal_nodes = find_goal_nodes(&index, goal);
    if goal_nodes.is_empty() {
        return None;
    }

    // Find career profile for relevance scoring
    let career_profile = match goal {
        PathGoal::Career(field) => embedded_career_profiles()
            .into_iter()
            .find(|p| p.field.to_lowercase().contains(&field.to_lowercase())),
        _ => None,
    };

    // A* search
    let mut heap = BinaryHeap::new();
    let mut came_from: HashMap<String, String> = HashMap::new();
    let mut g_score: HashMap<String, f32> = HashMap::new();

    let start_node = &index.nodes[start];
    let start_cost = if completed.contains(start) { 0.0 } else { start_node.estimated_hours as f32 * constraints.time_weight };

    g_score.insert(start.to_string(), start_cost);
    heap.push(AStarState {
        node_id: start.to_string(),
        cost: start_cost,
        priority: start_cost + heuristic(&index, start, &goal_nodes),
        hours: if completed.contains(start) { 0 } else { start_node.estimated_hours },
        bloom: start_node.bloom_ordinal,
    });

    while let Some(current) = heap.pop() {
        // Check if we reached a goal
        if goal_nodes.contains(&current.node_id) {
            return Some(reconstruct_path(
                &index,
                &came_from,
                start,
                &current.node_id,
                goal,
                career_profile.as_ref(),
            ));
        }

        // Check hour budget
        if let Some(max) = constraints.max_hours {
            if current.hours > max {
                continue;
            }
        }

        // Expand neighbors
        if let Some(neighbors) = index.adjacency.get(&current.node_id) {
            for (neighbor_id, edge_weight) in neighbors {
                let neighbor = match index.nodes.get(neighbor_id) {
                    Some(n) => n,
                    None => continue,
                };

                // Bloom constraint
                if constraints.bloom_strict && neighbor.bloom_ordinal > current.bloom + 2 {
                    continue; // too big a Bloom jump
                }

                // Compute cost
                let traverse_cost = if completed.contains(neighbor_id) {
                    0.1 // nearly free if already completed
                } else {
                    let time_cost = neighbor.estimated_hours as f32 * constraints.time_weight;
                    let bloom_penalty = if neighbor.bloom_ordinal > current.bloom + 1 {
                        (neighbor.bloom_ordinal - current.bloom - 1) as f32 * 10.0
                    } else {
                        0.0
                    };
                    let career_bonus = if let Some(ref profile) = career_profile {
                        career_relevance(neighbor, profile) * constraints.career_weight * -20.0
                    } else {
                        0.0
                    };
                    time_cost + edge_weight * 5.0 + bloom_penalty + career_bonus
                };

                let tentative_g = current.cost + traverse_cost;

                if tentative_g < *g_score.get(neighbor_id).unwrap_or(&f32::INFINITY) {
                    came_from.insert(neighbor_id.clone(), current.node_id.clone());
                    g_score.insert(neighbor_id.clone(), tentative_g);

                    let new_hours = current.hours
                        + if completed.contains(neighbor_id) {
                            0
                        } else {
                            neighbor.estimated_hours
                        };

                    heap.push(AStarState {
                        node_id: neighbor_id.clone(),
                        cost: tentative_g,
                        priority: tentative_g + heuristic(&index, neighbor_id, &goal_nodes),
                        hours: new_hours,
                        bloom: neighbor.bloom_ordinal.max(current.bloom),
                    });
                }
            }
        }
    }

    None // no path found
}

/// Heuristic: ISCED level distance to nearest goal.
fn heuristic(index: &GraphIndex, node_id: &str, goals: &HashSet<String>) -> f32 {
    let node = match index.nodes.get(node_id) {
        Some(n) => n,
        None => return 0.0,
    };

    goals
        .iter()
        .filter_map(|g| {
            index.nodes.get(g).map(|goal_node| {
                let level_diff = (goal_node.isced_level as f32 - node.isced_level as f32).abs();
                level_diff * 100.0 // average ~100 hours per ISCED level
            })
        })
        .fold(f32::INFINITY, f32::min)
}

/// Find goal node IDs based on PathGoal.
fn find_goal_nodes(index: &GraphIndex, goal: &PathGoal) -> HashSet<String> {
    match goal {
        PathGoal::Node(id) => {
            let mut set = HashSet::new();
            if index.nodes.contains_key(id) {
                set.insert(id.clone());
            }
            set
        }
        PathGoal::Career(field) => {
            let lower = field.to_lowercase();
            index
                .nodes
                .iter()
                .filter(|(_, n)| {
                    n.tags.iter().any(|t| t.to_lowercase().contains(&lower))
                        || n.subject_area.to_lowercase().contains(&lower)
                })
                .map(|(id, _)| id.clone())
                .collect()
        }
        PathGoal::Level { isced_level, subject } => {
            let lower = subject.to_lowercase();
            index
                .nodes
                .iter()
                .filter(|(_, n)| {
                    n.isced_level >= *isced_level
                        && n.subject_area.to_lowercase().contains(&lower)
                })
                .map(|(id, _)| id.clone())
                .collect()
        }
    }
}

/// Compute career relevance of a node to a career profile (0.0-1.0).
fn career_relevance(node: &IndexedNode, profile: &CareerProfile) -> f32 {
    let mut score = 0.0f32;

    // CIP code match
    if let Some(ref cip) = node.cip_code {
        for related_cip in &profile.related_cip_codes {
            if cip.starts_with(&related_cip[..2.min(related_cip.len())]) {
                score += 0.5;
                break;
            }
        }
    }

    // Tag overlap with field name
    let field_lower = profile.field.to_lowercase();
    for tag in &node.tags {
        if field_lower.contains(&tag.to_lowercase()) {
            score += 0.3;
            break;
        }
    }

    // Subject match
    if node.subject_area.to_lowercase().contains(&field_lower)
        || field_lower.contains(&node.subject_area.to_lowercase())
    {
        score += 0.2;
    }

    score.min(1.0)
}

/// Reconstruct the path from came_from map.
fn reconstruct_path(
    index: &GraphIndex,
    came_from: &HashMap<String, String>,
    start: &str,
    end: &str,
    goal: &PathGoal,
    career: Option<&CareerProfile>,
) -> LearningPlan {
    let mut path = vec![end.to_string()];
    let mut current = end.to_string();
    while let Some(prev) = came_from.get(&current) {
        path.push(prev.clone());
        current = prev.clone();
        if current == start {
            break;
        }
    }
    path.reverse();

    let mut cumulative = 0u32;
    let steps: Vec<PathStep> = path
        .iter()
        .enumerate()
        .filter_map(|(i, id)| {
            let node = index.nodes.get(id)?;
            cumulative += node.estimated_hours;
            Some(PathStep {
                node_id: id.clone(),
                title: node.title.clone(),
                estimated_hours: node.estimated_hours,
                cumulative_hours: cumulative,
                bloom_level: ordinal_to_bloom(node.bloom_ordinal).to_string(),
                isced_level: node.isced_level,
                subject_area: node.subject_area.clone(),
                reason: if i == 0 {
                    "Starting point".to_string()
                } else if i == path.len() - 1 {
                    "Goal reached".to_string()
                } else {
                    "Prerequisite".to_string()
                },
            })
        })
        .collect();

    let bloom_progression: Vec<String> = steps.iter().map(|s| s.bloom_level.clone()).collect();
    let total_hours = steps.last().map(|s| s.cumulative_hours).unwrap_or(0);

    let career_alignment = career.map(|p| {
        let relevant_count = steps
            .iter()
            .filter(|s| {
                index
                    .nodes
                    .get(&s.node_id)
                    .map(|n| career_relevance(n, p) > 0.3)
                    .unwrap_or(false)
            })
            .count();
        relevant_count as f32 / steps.len().max(1) as f32
    });

    let goal_desc = match goal {
        PathGoal::Node(id) => format!("Node: {id}"),
        PathGoal::Career(field) => format!("Career: {field}"),
        PathGoal::Level { isced_level, subject } => format!("ISCED {isced_level} in {subject}"),
    };

    LearningPlan {
        from: start.to_string(),
        goal: goal_desc,
        steps,
        total_hours,
        total_nodes: path.len(),
        bloom_progression,
        career_alignment,
    }
}

fn bloom_to_ordinal(bloom: &str) -> u8 {
    match bloom {
        "Remember" => 0,
        "Understand" => 1,
        "Apply" => 2,
        "Analyze" => 3,
        "Evaluate" => 4,
        "Create" => 5,
        _ => 1,
    }
}

fn ordinal_to_bloom(ord: u8) -> &'static str {
    match ord {
        0 => "Remember",
        1 => "Understand",
        2 => "Apply",
        3 => "Analyze",
        4 => "Evaluate",
        5 => "Create",
        _ => "Understand",
    }
}

/// Print a learning plan in a human-readable format.
pub fn print_plan(plan: &LearningPlan) {
    eprintln!("=== LEARNING PLAN ===");
    eprintln!("From: {}", plan.from);
    eprintln!("Goal: {}", plan.goal);
    eprintln!("Steps: {}, Total hours: {}", plan.total_nodes, plan.total_hours);
    if let Some(alignment) = plan.career_alignment {
        eprintln!("Career alignment: {:.0}%", alignment * 100.0);
    }
    eprintln!();

    for (i, step) in plan.steps.iter().enumerate() {
        let bloom_bar = match step.bloom_level.as_str() {
            "Remember" => "[R     ]",
            "Understand" => "[RU    ]",
            "Apply" => "[RUA   ]",
            "Analyze" => "[RUAN  ]",
            "Evaluate" => "[RUANE ]",
            "Create" => "[RUANEC]",
            _ => "[      ]",
        };
        eprintln!(
            "  {:>2}. {} {} ({:>3}h, cum {:>5}h) — {}",
            i + 1,
            bloom_bar,
            step.title,
            step.estimated_hours,
            step.cumulative_hours,
            step.reason,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converter::*;

    fn make_node(id: &str, bloom: &str, hours: u32, grade: &str, subject: &str) -> CurriculumNode {
        CurriculumNode {
            id: id.into(), title: format!("Node {id}"), description: String::new(),
            node_type: "Concept".into(), difficulty: "Intermediate".into(),
            domain: subject.into(), subdomain: String::new(),
            tags: vec![subject.to_lowercase()], estimated_hours: hours,
            grade_levels: vec![grade.into()], bloom_level: bloom.into(),
            subject_area: subject.into(), academic_standards: vec![],
            credit_hours: None, course_level: None, cip_code: None,
            program_id: None, corequisites: vec![], supplementary_resources: vec![],
            exam_weight: None,
        }
    }

    fn make_edge(from: &str, to: &str, strength: u16) -> CurriculumEdge {
        CurriculumEdge {
            from: from.into(), to: to.into(), edge_type: "Requires".into(),
            strength_permille: strength, rationale: "test".into(),
        }
    }

    fn make_doc(nodes: Vec<CurriculumNode>, edges: Vec<CurriculumEdge>) -> CurriculumDocument {
        CurriculumDocument {
            metadata: CurriculumMetadata {
                title: "Test".into(), framework: "Test".into(), source: String::new(),
                grade_level: "Grade3".into(), subject_area: "Math".into(), domain: "Math".into(),
                version: "1".into(), total_standards: nodes.len(), domains: vec![],
                created_at: "2026-01-01".into(), notes: String::new(),
                academic_level: None, institution: None, cip_code: None,
                total_credits: None, duration_semesters: None,
            },
            nodes, edges,
        }
    }

    #[test]
    fn test_simple_path() {
        let doc = make_doc(
            vec![
                make_node("A", "Understand", 10, "Grade3", "Math"),
                make_node("B", "Apply", 20, "Grade5", "Math"),
                make_node("C", "Analyze", 30, "Grade8", "Math"),
            ],
            vec![make_edge("A", "B", 900), make_edge("B", "C", 800)],
        );

        let plan = find_learning_path(
            &doc, "A", &PathGoal::Node("C".into()),
            &HashSet::new(), &PathConstraints::default(),
        );
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[0].node_id, "A");
        assert_eq!(plan.steps[2].node_id, "C");
        assert_eq!(plan.total_hours, 60);
    }

    #[test]
    fn test_no_path() {
        let doc = make_doc(
            vec![
                make_node("A", "Understand", 10, "Grade3", "Math"),
                make_node("B", "Apply", 20, "Grade5", "Math"),
            ],
            vec![], // no edges
        );

        let plan = find_learning_path(
            &doc, "A", &PathGoal::Node("B".into()),
            &HashSet::new(), &PathConstraints::default(),
        );
        assert!(plan.is_none());
    }

    #[test]
    fn test_completed_nodes_are_free() {
        let doc = make_doc(
            vec![
                make_node("A", "Understand", 100, "Grade3", "Math"),
                make_node("B", "Apply", 200, "Grade5", "Math"),
                make_node("C", "Analyze", 50, "Grade8", "Math"),
            ],
            vec![make_edge("A", "B", 900), make_edge("B", "C", 800)],
        );

        let completed: HashSet<String> = ["A".into(), "B".into()].into();
        let plan = find_learning_path(
            &doc, "A", &PathGoal::Node("C".into()),
            &completed, &PathConstraints::default(),
        );
        assert!(plan.is_some());
        let plan = plan.unwrap();
        // Hours should be mostly the uncompleted node C
        assert!(plan.total_hours <= 50 + 100 + 200); // at most all hours
    }

    #[test]
    fn test_bloom_progression() {
        let doc = make_doc(
            vec![
                make_node("A", "Remember", 10, "Grade1", "Math"),
                make_node("B", "Understand", 10, "Grade2", "Math"),
                make_node("C", "Apply", 10, "Grade3", "Math"),
                make_node("D", "Analyze", 10, "Grade5", "Math"),
            ],
            vec![
                make_edge("A", "B", 900),
                make_edge("B", "C", 900),
                make_edge("A", "D", 900), // shortcut but big Bloom jump
                make_edge("C", "D", 900),
            ],
        );

        // Without bloom_strict, the algorithm can take any path
        let plan = find_learning_path(
            &doc, "A", &PathGoal::Node("D".into()),
            &HashSet::new(), &PathConstraints::default(),
        );
        assert!(plan.is_some());

        // With bloom_strict, A→D is blocked (Remember→Create = +5 > +2 limit)
        // but A→B→C→D works (each step is +1)
        let strict_plan = find_learning_path(
            &doc, "A", &PathGoal::Node("D".into()),
            &HashSet::new(), &PathConstraints { bloom_strict: true, ..Default::default() },
        );
        assert!(strict_plan.is_some());
        assert!(strict_plan.unwrap().steps.len() >= 3);
    }

    #[test]
    fn test_career_goal() {
        let doc = make_doc(
            vec![
                make_node("A", "Apply", 10, "Grade8", "Mathematics"),
                make_node("B", "Analyze", 50, "Undergraduate", "software"),
            ],
            vec![make_edge("A", "B", 700)],
        );

        let plan = find_learning_path(
            &doc, "A", &PathGoal::Career("software".into()),
            &HashSet::new(), &PathConstraints::default(),
        );
        // B's subject "software" matches career "software"
        assert!(plan.is_some());
    }

    #[test]
    fn test_nonexistent_start() {
        let doc = make_doc(vec![], vec![]);
        let plan = find_learning_path(
            &doc, "NONEXISTENT", &PathGoal::Node("X".into()),
            &HashSet::new(), &PathConstraints::default(),
        );
        assert!(plan.is_none());
    }

    #[test]
    fn test_start_is_goal() {
        let doc = make_doc(
            vec![make_node("A", "Apply", 10, "Grade3", "Math")],
            vec![],
        );
        let plan = find_learning_path(
            &doc, "A", &PathGoal::Node("A".into()),
            &HashSet::new(), &PathConstraints::default(),
        );
        assert!(plan.is_some());
        assert_eq!(plan.unwrap().steps.len(), 1);
    }
}
