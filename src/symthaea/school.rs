// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! School/curriculum learning subsystem types and methods.
//!
//! All items in this module are gated behind `#[cfg(feature = "school_learning")]`.

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use symthaea_core::hdc::ContinuousHV;

use crate::school::curriculum::Curriculum;
use crate::school::curriculum_loader::{CurriculumLoader, CurriculumMeta, LoadError};

use super::Symthaea;

// ── Public types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumObjectiveSummary {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub difficulty: String,
    pub estimated_minutes: u32,
    pub tags: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumReport {
    pub curriculum_id: String,
    pub curriculum_name: String,
    pub total_objectives: usize,
    pub dimension: usize,
    pub last_research_topic: Option<String>,
    pub last_research_at: Option<String>,
    pub last_saved_at: Option<String>,
    pub last_objectives_added: Option<usize>,
    pub recent_objectives: Vec<CurriculumObjectiveSummary>,
}

// ── Internal config types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub(super) struct CurriculumRecallConfig {
    pub(super) threshold: f32,
    pub(super) max_recall: usize,
    pub(super) log_top_k: usize,
    pub(super) budget: f32,
}

impl CurriculumRecallConfig {
    pub(super) fn from_env() -> Self {
        let threshold = std::env::var("SYMTHAEA_CURRICULUM_RECALL_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.65)
            .clamp(0.0, 1.0);
        let max_recall = std::env::var("SYMTHAEA_CURRICULUM_RECALL_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6)
            .max(1);
        let log_top_k = std::env::var("SYMTHAEA_CURRICULUM_RECALL_LOG_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);
        let budget = std::env::var("SYMTHAEA_CURRICULUM_RECALL_BUDGET")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(max_recall as f32)
            .max(0.0);

        Self {
            threshold,
            max_recall,
            log_top_k,
            budget,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct CurriculumPersistenceConfig {
    pub(super) path: PathBuf,
    pub(super) auto_save: bool,
}

impl CurriculumPersistenceConfig {
    pub(super) fn from_env() -> Self {
        let path = std::env::var("SYMTHAEA_CURRICULUM_PATH")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(default_curriculum_path);

        let auto_save = std::env::var("SYMTHAEA_CURRICULUM_AUTO_SAVE")
            .ok()
            .and_then(|v| parse_env_bool(&v))
            .unwrap_or(true);

        Self { path, auto_save }
    }
}

fn default_curriculum_path() -> PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::state_dir)
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("symthaea")
        .join("curriculum.json")
}

pub(super) fn parse_env_bool(value: &str) -> Option<bool> {
    match value.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

pub(super) struct CurriculumRecallScores {
    pub(super) scores: Vec<(f32, usize)>,
    pub(super) candidates: Vec<(f32, usize, ContinuousHV)>,
}

pub(super) fn load_curriculum_from_store(
    hdc_dim: usize,
    persistence: &CurriculumPersistenceConfig,
) -> (Curriculum, CurriculumMeta) {
    match CurriculumLoader::load_store_from_file_with_dimension(&persistence.path, hdc_dim) {
        Ok((curriculum, meta)) => (curriculum, meta),
        Err(LoadError::FileNotFound(_)) => (
            Curriculum::new("symthaea", "Main Curriculum").build(),
            CurriculumMeta::new(hdc_dim),
        ),
        Err(err) => {
            tracing::warn!(
                target: "symthaea::curriculum",
                error = %err,
                path = %persistence.path.display(),
                "Failed to load persisted curriculum, falling back to default"
            );
            (
                Curriculum::new("symthaea", "Main Curriculum").build(),
                CurriculumMeta::new(hdc_dim),
            )
        }
    }
}

// ── impl Symthaea methods ─────────────────────────────────────────────────

impl Symthaea {
    /// Record a curriculum research event and optionally auto-save.
    pub fn record_research(&mut self, topic: &str, objectives_added: usize) -> Result<()> {
        self.curriculum_meta.last_research_topic = Some(topic.to_string());
        self.curriculum_meta.last_research_at = Some(Utc::now().to_rfc3339());
        self.curriculum_meta.last_objectives_added = Some(objectives_added);
        self.curriculum_meta.total_objectives = self.curriculum.objectives.len();
        self.curriculum_meta.dimension = self.hdc_dim;

        if self.curriculum_persistence.auto_save {
            self.save_curriculum()?;
        }

        Ok(())
    }

    /// Persist the curriculum and metadata to disk.
    pub fn save_curriculum(&mut self) -> Result<()> {
        let path = &self.curriculum_persistence.path;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create curriculum directory: {}",
                    parent.display()
                )
            })?;
        }

        self.curriculum_meta.last_saved_at = Some(Utc::now().to_rfc3339());
        self.curriculum_meta.total_objectives = self.curriculum.objectives.len();
        self.curriculum_meta.dimension = self.hdc_dim;

        CurriculumLoader::save_store_to_json(&self.curriculum, &self.curriculum_meta, path)
            .with_context(|| format!("Failed to save curriculum to {}", path.display()))?;

        Ok(())
    }

    // NOTE: `curriculum_report()` removed — zero callers, dead code (Mar 2026).

    pub(super) fn curriculum_recall_scores(
        &self,
        input_embedding: &ContinuousHV,
        threshold: f32,
    ) -> CurriculumRecallScores {
        use std::cmp::Ordering;

        let target_dim = input_embedding.values.len();
        let mut scores = Vec::with_capacity(self.curriculum.objectives.len());
        let mut candidates = Vec::new();

        for (idx, obj) in self.curriculum.objectives.iter().enumerate() {
            let obj_hv = if obj.encoding.values.len() == target_dim {
                obj.encoding.clone()
            } else {
                let mut folded = vec![0.0f32; target_dim];
                for (i, &val) in obj.encoding.values.iter().enumerate() {
                    folded[i % target_dim] += val;
                }
                ContinuousHV::from_values(folded)
            };

            let similarity = input_embedding.similarity(&obj_hv);
            scores.push((similarity, idx));
            if similarity > threshold {
                candidates.push((similarity, idx, obj_hv));
            }
        }

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        CurriculumRecallScores { scores, candidates }
    }
}
