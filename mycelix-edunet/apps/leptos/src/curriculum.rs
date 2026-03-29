// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CAPS curriculum data loader and reactive state provider.
//!
//! Embeds the unified Matric graph JSON at compile time via `include_str!`.
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

static CAPS_GRAPH: OnceLock<CapsGraph> = OnceLock::new();

/// Get the parsed CAPS graph (lazily initialized).
pub fn caps_graph() -> &'static CapsGraph {
    CAPS_GRAPH.get_or_init(|| {
        let raw: RawCapsDocument = serde_json::from_str(CAPS_JSON)
            .expect("embedded CAPS JSON must be valid");
        CapsGraph::from_raw(raw)
    })
}

// ============================================================
// Data types
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawCapsDocument {
    metadata: RawMetadata,
    nodes: Vec<CapsNode>,
    edges: Vec<CapsEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawMetadata {
    title: String,
    total_standards: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapsNode {
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
pub struct CapsEdge {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub strength_permille: u16,
    pub rationale: String,
}

/// Parsed and indexed curriculum graph.
pub struct CapsGraph {
    pub nodes: Vec<CapsNode>,
    pub edges: Vec<CapsEdge>,
    pub by_id: HashMap<String, usize>,         // node id -> index
    pub by_subject_grade: HashMap<(String, String), Vec<usize>>, // (subject, grade) -> indices
    pub prerequisites: HashMap<String, Vec<String>>, // node id -> prerequisite node ids
    pub dependents: HashMap<String, Vec<String>>,    // node id -> nodes that depend on it
}

impl CapsGraph {
    fn from_raw(raw: RawCapsDocument) -> Self {
        let mut by_id = HashMap::new();
        let mut by_subject_grade: HashMap<(String, String), Vec<usize>> = HashMap::new();
        let mut prerequisites: HashMap<String, Vec<String>> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

        for (i, node) in raw.nodes.iter().enumerate() {
            by_id.insert(node.id.clone(), i);
            let grade = node.grade_levels.first().cloned().unwrap_or_default();
            by_subject_grade
                .entry((node.subject_area.clone(), grade))
                .or_default()
                .push(i);
        }

        for edge in &raw.edges {
            prerequisites
                .entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
            dependents
                .entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
        }

        CapsGraph {
            nodes: raw.nodes,
            edges: raw.edges,
            by_id,
            by_subject_grade,
            prerequisites,
            dependents,
        }
    }

    /// Get nodes for a subject + grade combination.
    pub fn nodes_for(&self, subject: &str, grade: &str) -> Vec<&CapsNode> {
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
    pub fn node(&self, id: &str) -> Option<&CapsNode> {
        self.by_id.get(id).map(|&i| &self.nodes[i])
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
    pub fn subject_mastery(&self, graph: &CapsGraph, subject: &str) -> u16 {
        let nodes: Vec<&CapsNode> = graph.nodes.iter().filter(|n| n.subject_area == subject).collect();
        if nodes.is_empty() { return 0; }
        let total: u32 = nodes.iter().map(|n| self.get(&n.id).mastery_permille as u32).sum();
        (total / nodes.len() as u32) as u16
    }
}

const PROGRESS_KEY: &str = "edunet_progress";

// ============================================================
// Leptos context provider
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subject {
    Mathematics,
    PhysicalSciences,
    NaturalSciences,
}

impl Subject {
    pub fn as_str(&self) -> &'static str {
        match self {
            Subject::Mathematics => "Mathematics",
            Subject::PhysicalSciences => "Physical Sciences",
            Subject::NaturalSciences => "Natural Sciences",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade {
    Gr9,
    Gr10,
    Gr11,
    Gr12,
}

impl Grade {
    pub fn as_str(&self) -> &'static str {
        match self {
            Grade::Gr9 => "Grade9",
            Grade::Gr10 => "Grade10",
            Grade::Gr11 => "Grade11",
            Grade::Gr12 => "Grade12",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Grade::Gr9 => "Grade 9",
            Grade::Gr10 => "Grade 10",
            Grade::Gr11 => "Grade 11",
            Grade::Gr12 => "Grade 12",
        }
    }
}

/// Provide curriculum context. Call once at app root.
pub fn provide_curriculum_context() {
    // Force parse on startup
    let _ = caps_graph();

    let initial_progress = persistence::load::<ProgressStore>(PROGRESS_KEY)
        .unwrap_or_default();

    let (subject, set_subject) = signal(Subject::Mathematics);
    let (grade, set_grade) = signal(Grade::Gr12);
    let (progress, set_progress) = signal(initial_progress);

    // Persist progress on change
    Effect::new(move |_| {
        let p = progress.get();
        persistence::save(PROGRESS_KEY, &p);
    });

    provide_context(subject);
    provide_context(set_subject);
    provide_context(grade);
    provide_context(set_grade);
    provide_context(progress);
    provide_context(set_progress);
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
