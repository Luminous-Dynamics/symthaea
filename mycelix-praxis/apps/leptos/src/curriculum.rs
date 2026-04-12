// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Curriculum data loader and reactive state provider.
//!
//! Embeds the unified curriculum graph JSON at compile time via `include_str!`.
//! Parses lazily on first access and provides reactive Leptos signals for
//! subject/grade filtering and progress tracking.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::persistence;

// ============================================================
// Embedded curriculum data
// ============================================================

const CAPS_JSON: &str = include_str!("../../../examples/curriculum/caps/caps-unified-matric.json");

static CAPS_GRAPH: OnceLock<CurriculumGraph> = OnceLock::new();

/// Get the parsed CAPS graph (lazily initialized).
pub fn curriculum_graph() -> &'static CurriculumGraph {
    CAPS_GRAPH.get_or_init(|| {
        let raw: RawCurriculumDocument = serde_json::from_str(CAPS_JSON)
            .expect("embedded CAPS JSON must be valid");
        CurriculumGraph::from_raw(raw)
    })
}

// ============================================================
// Data types
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawCurriculumDocument {
    metadata: RawMetadata,
    nodes: Vec<CurriculumNode>,
    edges: Vec<CurriculumEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawMetadata {
    title: String,
    total_standards: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurriculumNode {
    pub id: String,
    pub title: String,
    pub description: String,
    pub node_type: String,
    pub difficulty: String,
    pub domain: String,
    pub subdomain: String,
    pub tags: Vec<String>,
    pub estimated_hours: u32,
    pub grade_levels: Vec<String>,
    pub bloom_level: String,
    pub subject_area: String,
    #[serde(default)]
    pub supplementary_resources: Vec<SupplementaryResource>,
    #[serde(default)]
    pub exam_weight: Option<ExamWeight>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExamWeight {
    pub paper: u8,
    pub marks: u16,
    pub total_paper_marks: u16,
    pub percentage: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupplementaryResource {
    pub title: String,
    pub url: String,
    pub source: serde_json::Value, // ResourceSource enum varies
    pub content_type: serde_json::Value,
    pub relevance_score: u8,
    #[serde(default)]
    pub aligned_standard: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurriculumEdge {
    #[serde(alias = "source")]
    pub from: String,
    #[serde(alias = "target")]
    pub to: String,
    #[serde(alias = "relationship")]
    pub edge_type: String,
    #[serde(default)]
    pub strength_permille: u16,
    #[serde(default)]
    pub rationale: String,
}

/// Parsed and indexed curriculum graph.
pub struct CurriculumGraph {
    pub nodes: Vec<CurriculumNode>,
    pub edges: Vec<CurriculumEdge>,
    pub by_id: HashMap<String, usize>,         // node id -> index
    pub by_subject_grade: HashMap<(String, String), Vec<usize>>, // (subject, grade) -> indices
    pub prerequisites: HashMap<String, Vec<String>>, // node id -> prerequisite node ids
    pub dependents: HashMap<String, Vec<String>>,    // node id -> nodes that depend on it
    /// Cross-subject edges (AppliedIn/RelatedTo) indexed by node id
    pub cross_subject: HashMap<String, Vec<usize>>,  // node id -> edge indices
}

impl CurriculumGraph {
    fn from_raw(raw: RawCurriculumDocument) -> Self {
        let mut by_id = HashMap::new();
        let mut by_subject_grade: HashMap<(String, String), Vec<usize>> = HashMap::new();
        let mut prerequisites: HashMap<String, Vec<String>> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

        // Normalize subject names (collapse spaces for consistency —
        // CAPS JSON has both "EnglishLanguageArts" and "English Language Arts")
        let mut nodes = raw.nodes;
        for node in &mut nodes {
            node.subject_area = node.subject_area.replace(' ', "");
        }

        for (i, node) in nodes.iter().enumerate() {
            by_id.insert(node.id.clone(), i);
            let grade = node.grade_levels.first().cloned().unwrap_or_default();
            by_subject_grade
                .entry((node.subject_area.clone(), grade))
                .or_default()
                .push(i);
        }

        let mut cross_subject: HashMap<String, Vec<usize>> = HashMap::new();

        for (edge_idx, edge) in raw.edges.iter().enumerate() {
            prerequisites
                .entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
            dependents
                .entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());

            // Index cross-subject edges (AppliedIn/RelatedTo)
            if edge.edge_type == "AppliedIn" || edge.edge_type == "RelatedTo" {
                cross_subject.entry(edge.from.clone()).or_default().push(edge_idx);
                cross_subject.entry(edge.to.clone()).or_default().push(edge_idx);
            }
        }

        CurriculumGraph {
            nodes,
            edges: raw.edges,
            by_id,
            by_subject_grade,
            prerequisites,
            dependents,
            cross_subject,
        }
    }

    /// Get nodes for a subject + grade combination.
    pub fn nodes_for(&self, subject: &str, grade: &str) -> Vec<&CurriculumNode> {
        self.by_subject_grade
            .get(&(subject.to_string(), grade.to_string()))
            .map(|indices| indices.iter().map(|&i| &self.nodes[i]).collect())
            .unwrap_or_default()
    }

    /// Get prerequisite node IDs for a given node.
    pub fn prereqs_for(&self, node_id: &str) -> Vec<&str> {
        self.prerequisites
            .get(node_id)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get a node by ID.
    pub fn node(&self, id: &str) -> Option<&CurriculumNode> {
        self.by_id.get(id).map(|&i| &self.nodes[i])
    }

    /// Get cross-subject connections for a node (AppliedIn/RelatedTo edges).
    /// Returns (neighbor_node, edge, is_outgoing) tuples.
    pub fn cross_subject_neighbors(&self, node_id: &str) -> Vec<(&CurriculumNode, &CurriculumEdge)> {
        self.cross_subject
            .get(node_id)
            .map(|edge_indices| {
                edge_indices.iter().filter_map(|&ei| {
                    let edge = &self.edges[ei];
                    let neighbor_id = if edge.from == node_id { &edge.to } else { &edge.from };
                    let neighbor = self.node(neighbor_id)?;
                    Some((neighbor, edge))
                }).collect()
            })
            .unwrap_or_default()
    }

    /// Find the highest-priority unmastered node whose prerequisites are all met.
    ///
    /// Sorts by `exam_weight.percentage` descending when available, then falls
    /// back to node order.  Returns the first node that is (a) not in
    /// `mastered_ids` and (b) has every prerequisite satisfied (i.e. all prereqs
    /// are in `mastered_ids`).  If no such node exists, returns the first
    /// unmastered node with no prerequisites at all.
    pub fn recommended_next(&self, mastered_ids: &std::collections::HashSet<String>) -> Option<&str> {
        // Build a list of (index, exam_weight_pct) sorted by weight desc
        let mut candidates: Vec<(usize, f32)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !mastered_ids.contains(&n.id))
            .map(|(i, n)| {
                let weight = n.exam_weight.as_ref().map(|w| w.percentage).unwrap_or(0.0);
                (i, weight)
            })
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // First pass: highest-weight node with all prereqs mastered
        for &(idx, _) in &candidates {
            let node = &self.nodes[idx];
            let prereqs = self.prereqs_for(&node.id);
            if prereqs.iter().all(|p| mastered_ids.contains(*p)) {
                return Some(&node.id);
            }
        }

        // Fallback: first unmastered node with no prerequisites
        for &(idx, _) in &candidates {
            let node = &self.nodes[idx];
            if self.prereqs_for(&node.id).is_empty() {
                return Some(&node.id);
            }
        }

        None
    }

    /// All unique subjects.
    pub fn subjects(&self) -> Vec<&str> {
        let mut subjects: Vec<&str> = self.nodes.iter().map(|n| n.subject_area.as_str()).collect();
        subjects.sort();
        subjects.dedup();
        subjects
    }

    /// All unique grades.
    pub fn grades(&self) -> Vec<&str> {
        let mut grades: Vec<&str> = self.nodes.iter().flat_map(|n| n.grade_levels.iter().map(|g| g.as_str())).collect();
        grades.sort();
        grades.dedup();
        grades
    }
}

// ============================================================
// Progress tracking
// ============================================================

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeProgress {
    pub mastery_permille: u16,
    pub attempts: u32,
    pub correct: u32,
    pub status: ProgressStatus,
    pub last_reviewed: Option<f64>, // JS timestamp
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressStatus {
    #[default]
    NotStarted,
    Studying,
    Mastered,
}

impl ProgressStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ProgressStatus::NotStarted => "Not Started",
            ProgressStatus::Studying => "Studying",
            ProgressStatus::Mastered => "Mastered",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            ProgressStatus::NotStarted => "status-not-started",
            ProgressStatus::Studying => "status-studying",
            ProgressStatus::Mastered => "status-mastered",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProgressStore {
    pub nodes: HashMap<String, NodeProgress>,
    pub exam_date: Option<String>, // ISO date
    /// SRS flashcard state — keyed by card_id (e.g., "math:3")
    #[serde(default)]
    pub srs_cards: HashMap<String, SrsCardState>,
    /// BKT adaptive difficulty state — keyed by node_id
    #[serde(default)]
    pub bkt_states: HashMap<String, BktState>,
}

/// SM-2 spaced repetition state for a single flashcard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SrsCardState {
    pub ease_factor: f32,      // starts at 2.5
    pub interval_days: u32,    // current interval
    pub repetitions: u32,      // consecutive correct
    pub next_review_ms: f64,   // JS timestamp of next review
}

impl Default for SrsCardState {
    fn default() -> Self {
        Self {
            ease_factor: 2.5,
            interval_days: 0,
            repetitions: 0,
            next_review_ms: 0.0,
        }
    }
}

impl SrsCardState {
    /// SM-2 update: quality 0-5 (0-1 = blackout, 2 = hard, 3-4 = good, 5 = easy)
    pub fn update(&mut self, quality: u8) {
        let q = quality as f32;
        if quality >= 3 {
            // Correct response
            self.interval_days = match self.repetitions {
                0 => 1,
                1 => 6,
                _ => (self.interval_days as f32 * self.ease_factor) as u32,
            };
            self.repetitions += 1;
        } else {
            // Incorrect — reset
            self.repetitions = 0;
            self.interval_days = 1;
        }
        // Update ease factor
        self.ease_factor += 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02);
        if self.ease_factor < 1.3 { self.ease_factor = 1.3; }

        // Set next review time
        let now = js_sys::Date::now();
        self.next_review_ms = now + (self.interval_days as f64 * 24.0 * 60.0 * 60.0 * 1000.0);
    }

    /// Is this card due for review?
    pub fn is_due(&self) -> bool {
        js_sys::Date::now() >= self.next_review_ms
    }
}

impl ProgressStore {
    pub fn get(&self, node_id: &str) -> &NodeProgress {
        static DEFAULT: NodeProgress = NodeProgress {
            mastery_permille: 0,
            attempts: 0,
            correct: 0,
            status: ProgressStatus::NotStarted,
            last_reviewed: None,
        };
        self.nodes.get(node_id).unwrap_or(&DEFAULT)
    }

    pub fn set_status(&mut self, node_id: &str, status: ProgressStatus) {
        let entry = self.nodes.entry(node_id.to_string()).or_default();
        entry.status = status;
        if status == ProgressStatus::Mastered {
            entry.mastery_permille = entry.mastery_permille.max(900);
        }
    }

    pub fn mastered_count(&self) -> usize {
        self.nodes.values().filter(|p| p.status == ProgressStatus::Mastered).count()
    }

    pub fn studying_count(&self) -> usize {
        self.nodes.values().filter(|p| p.status == ProgressStatus::Studying).count()
    }

    /// Mastery permille for a subject (0-1000).
    pub fn subject_mastery(&self, graph: &CurriculumGraph, subject: &str) -> u16 {
        let nodes: Vec<&CurriculumNode> = graph.nodes.iter().filter(|n| n.subject_area == subject).collect();
        if nodes.is_empty() { return 0; }
        let total: u32 = nodes.iter().map(|n| self.get(&n.id).mastery_permille as u32).sum();
        (total / nodes.len() as u32) as u16
    }
}

// ============================================================
// Bayesian Knowledge Tracing (BKT) — adaptive difficulty engine
// ============================================================

/// BKT parameters for a topic — calibrated for educational assessment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BktState {
    /// P(L) — probability the student has learned/mastered the skill
    pub p_mastery: f32,
    /// Total practice attempts on this topic
    pub attempts: u32,
    /// Correct responses
    pub correct: u32,
}

impl Default for BktState {
    fn default() -> Self {
        Self {
            p_mastery: 0.1, // P(L₀) — low prior, student hasn't demonstrated knowledge yet
            attempts: 0,
            correct: 0,
        }
    }
}

impl BktState {
    // BKT parameters (Corbett & Anderson, 1995)
    const P_TRANSIT: f32 = 0.1;   // P(T) — probability of learning per opportunity
    const P_SLIP: f32 = 0.1;      // P(S) — probability of slip (knows but answers wrong)
    const P_GUESS: f32 = 0.25;    // P(G) — probability of guessing correctly

    /// Update P(mastery) after observing a response (correct or incorrect).
    pub fn update(&mut self, correct: bool) {
        self.attempts += 1;
        if correct {
            self.correct += 1;
        }

        // BKT posterior update
        let p_l = self.p_mastery;
        if correct {
            // P(L|correct) = P(L) * (1 - P(S)) / P(correct)
            let p_correct = p_l * (1.0 - Self::P_SLIP) + (1.0 - p_l) * Self::P_GUESS;
            let p_l_given_correct = p_l * (1.0 - Self::P_SLIP) / p_correct;
            // Apply transition: P(L_new) = P(L|obs) + (1 - P(L|obs)) * P(T)
            self.p_mastery = p_l_given_correct + (1.0 - p_l_given_correct) * Self::P_TRANSIT;
        } else {
            // P(L|incorrect) = P(L) * P(S) / P(incorrect)
            let p_incorrect = p_l * Self::P_SLIP + (1.0 - p_l) * (1.0 - Self::P_GUESS);
            let p_l_given_incorrect = p_l * Self::P_SLIP / p_incorrect;
            // Apply transition
            self.p_mastery = p_l_given_incorrect + (1.0 - p_l_given_incorrect) * Self::P_TRANSIT;
        }

        // Clamp to [0.01, 0.99]
        self.p_mastery = self.p_mastery.clamp(0.01, 0.99);
    }

    /// Recommended difficulty tier (0-3) based on current mastery estimate.
    /// 0 = gentle soil, 1 = steady ground, 2 = rocky terrain, 3 = mountain path
    pub fn recommended_difficulty(&self) -> u8 {
        match self.p_mastery {
            p if p < 0.3 => 0,  // Still learning — gentle problems
            p if p < 0.5 => 1,  // Getting it — steady challenge
            p if p < 0.75 => 2, // Strong — push with harder problems
            _ => 3,             // Near mastery — mountain paths only
        }
    }

    /// Has the student likely mastered this topic? (P(L) > 0.95)
    pub fn is_likely_mastered(&self) -> bool {
        self.p_mastery > 0.95
    }

    /// Terrain label for current difficulty recommendation.
    pub fn terrain_label(&self) -> &'static str {
        match self.recommended_difficulty() {
            0 => "Gentle soil",
            1 => "Steady ground",
            2 => "Rocky terrain",
            _ => "Mountain path",
        }
    }
}

impl ProgressStore {
    /// Get or create BKT state for a topic.
    pub fn bkt(&self, node_id: &str) -> BktState {
        self.bkt_states.get(node_id).cloned().unwrap_or_default()
    }

    /// Record a practice response and update BKT.
    pub fn record_response(&mut self, node_id: &str, correct: bool) {
        let bkt = self.bkt_states.entry(node_id.to_string()).or_default();
        bkt.update(correct);

        // Also update basic progress
        let progress = self.nodes.entry(node_id.to_string()).or_default();
        progress.attempts += 1;
        if correct {
            progress.correct += 1;
        }
        progress.mastery_permille = (bkt.p_mastery * 1000.0) as u16;

        // Auto-promote status based on BKT
        if bkt.is_likely_mastered() && progress.status != ProgressStatus::Mastered {
            progress.status = ProgressStatus::Mastered;
        } else if bkt.attempts > 0 && progress.status == ProgressStatus::NotStarted {
            progress.status = ProgressStatus::Studying;
        }
    }

    /// Topics sorted by weakness (lowest P(mastery) first), filtered to those with attempts.
    pub fn weakest_topics(&self, limit: usize) -> Vec<(String, f32)> {
        let mut topics: Vec<(String, f32)> = self.bkt_states.iter()
            .filter(|(_, bkt)| bkt.attempts > 0)
            .map(|(id, bkt)| (id.clone(), bkt.p_mastery))
            .collect();
        topics.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        topics.truncate(limit);
        topics
    }
}

const PROGRESS_KEY: &str = "praxis_progress";

// ============================================================
// Leptos context provider
// ============================================================

/// Insert spaces before uppercase letters in camelCase subject names.
/// e.g. "EnglishLanguageArts" -> "English Language Arts"
pub fn display_subject(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && c.is_uppercase() {
            result.push(' ');
        }
        result.push(c);
    }
    result
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subject(pub String);

impl Subject {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Get all unique subjects from the graph.
    pub fn all_from_graph(graph: &CurriculumGraph) -> Vec<Subject> {
        let mut subjects: Vec<String> = graph.subjects().iter().map(|s| s.to_string()).collect();
        subjects.sort();
        subjects.into_iter().map(Subject).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade {
    Kindergarten,
    Gr1, Gr2, Gr3, Gr4, Gr5, Gr6, Gr7, Gr8, Gr9, Gr10, Gr11, Gr12,
    Undergraduate, Graduate, Doctoral, Adult,
}

impl Grade {
    pub fn as_str(&self) -> &'static str {
        match self {
            Grade::Kindergarten => "Kindergarten",
            Grade::Gr1 => "Grade1", Grade::Gr2 => "Grade2", Grade::Gr3 => "Grade3",
            Grade::Gr4 => "Grade4", Grade::Gr5 => "Grade5", Grade::Gr6 => "Grade6",
            Grade::Gr7 => "Grade7", Grade::Gr8 => "Grade8", Grade::Gr9 => "Grade9",
            Grade::Gr10 => "Grade10", Grade::Gr11 => "Grade11", Grade::Gr12 => "Grade12",
            Grade::Undergraduate => "Undergraduate", Grade::Graduate => "Graduate",
            Grade::Doctoral => "Doctoral", Grade::Adult => "Adult",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Grade::Kindergarten => "K",
            Grade::Gr1 => "1", Grade::Gr2 => "2", Grade::Gr3 => "3",
            Grade::Gr4 => "4", Grade::Gr5 => "5", Grade::Gr6 => "6",
            Grade::Gr7 => "7", Grade::Gr8 => "8", Grade::Gr9 => "9",
            Grade::Gr10 => "10", Grade::Gr11 => "11", Grade::Gr12 => "12",
            Grade::Undergraduate => "Uni", Grade::Graduate => "Postgrad",
            Grade::Doctoral => "PhD", Grade::Adult => "Life",
        }
    }

    /// All grades including post-secondary.
    pub fn all() -> &'static [Grade] {
        &[Grade::Kindergarten,
          Grade::Gr1, Grade::Gr2, Grade::Gr3, Grade::Gr4, Grade::Gr5, Grade::Gr6,
          Grade::Gr7, Grade::Gr8, Grade::Gr9, Grade::Gr10, Grade::Gr11, Grade::Gr12,
          Grade::Undergraduate, Grade::Graduate, Grade::Doctoral, Grade::Adult]
    }

    /// K-12 grades only (including Kindergarten).
    pub fn k12() -> &'static [Grade] {
        &[Grade::Kindergarten,
          Grade::Gr1, Grade::Gr2, Grade::Gr3, Grade::Gr4, Grade::Gr5, Grade::Gr6,
          Grade::Gr7, Grade::Gr8, Grade::Gr9, Grade::Gr10, Grade::Gr11, Grade::Gr12]
    }

    /// Post-secondary grades.
    pub fn post_secondary() -> &'static [Grade] {
        &[Grade::Undergraduate, Grade::Graduate, Grade::Doctoral, Grade::Adult]
    }

    /// Convert from profile grade number to Grade enum.
    pub fn from_profile_grade(g: u8) -> Grade {
        match g {
            0 => Grade::Kindergarten,
            1 => Grade::Gr1, 2 => Grade::Gr2, 3 => Grade::Gr3, 4 => Grade::Gr4,
            5 => Grade::Gr5, 6 => Grade::Gr6, 7 => Grade::Gr7, 8 => Grade::Gr8,
            9 => Grade::Gr9, 10 => Grade::Gr10, 11 => Grade::Gr11, 12 => Grade::Gr12,
            13 => Grade::Undergraduate, 14 => Grade::Graduate, 15 => Grade::Doctoral,
            16 => Grade::Adult, _ => Grade::Gr12,
        }
    }

    pub fn is_post_secondary(&self) -> bool {
        matches!(self, Grade::Undergraduate | Grade::Graduate | Grade::Doctoral | Grade::Adult)
    }
}

/// Provide curriculum context. Call once at app root.
pub fn provide_curriculum_context() {
    // Force parse on startup
    let _ = curriculum_graph();

    let initial_progress = persistence::load::<ProgressStore>(PROGRESS_KEY)
        .unwrap_or_default();

    // Default grade from student profile (so constellation shows relevant content)
    let profile_grade = persistence::load::<crate::student_profile::StudentProfile>("praxis_profile")
        .map(|p| Grade::from_profile_grade(p.grade))
        .unwrap_or(Grade::Gr12);
    let default_subject = match profile_grade {
        Grade::Adult => "Financial Literacy",
        Grade::Undergraduate => "Computer Science",
        _ => "Mathematics",
    };
    let (subject, set_subject) = signal(Subject(default_subject.to_string()));
    let (grade, set_grade) = signal(profile_grade);
    let (progress, set_progress) = signal(initial_progress);

    // Persist progress on change
    Effect::new(move |_| {
        let p = progress.get();
        persistence::save(PROGRESS_KEY, &p);
    });

    // Celebration signal — briefly true when a topic is mastered
    let (celebrating, set_celebrating) = signal(false);
    let (celebration_topic, set_celebration_topic) = signal(String::new());

    provide_context(subject);
    provide_context(set_subject);
    provide_context(grade);
    provide_context(set_grade);
    provide_context(progress);
    provide_context(set_progress);
    provide_context(celebrating);
    provide_context(set_celebrating);
    provide_context(celebration_topic);
    provide_context(set_celebration_topic);
}

pub fn use_subject() -> ReadSignal<Subject> {
    expect_context::<ReadSignal<Subject>>()
}

pub fn use_set_subject() -> WriteSignal<Subject> {
    expect_context::<WriteSignal<Subject>>()
}

pub fn use_grade() -> ReadSignal<Grade> {
    expect_context::<ReadSignal<Grade>>()
}

pub fn use_set_grade() -> WriteSignal<Grade> {
    expect_context::<WriteSignal<Grade>>()
}

pub fn use_progress() -> ReadSignal<ProgressStore> {
    expect_context::<ReadSignal<ProgressStore>>()
}

pub fn use_set_progress() -> WriteSignal<ProgressStore> {
    expect_context::<WriteSignal<ProgressStore>>()
}

/// Set a node's status with celebration trigger for Mastered.
pub fn set_node_status(node_id: &str, status: ProgressStatus, topic_title: &str) {
    let set_progress = use_set_progress();
    set_progress.update(|p| p.set_status(node_id, status));

    if status == ProgressStatus::Mastered {
        let set_celebrating = expect_context::<WriteSignal<bool>>();
        let set_celebration_topic = expect_context::<WriteSignal<String>>();
        set_celebrating.set(true);
        set_celebration_topic.set(topic_title.to_string());

        // Auto-dismiss after 2 seconds
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_millis(2000)).await;
            set_celebrating.set(false);
        });
    }
}
