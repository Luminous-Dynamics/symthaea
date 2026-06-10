// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active Code Explorer — Free Energy Principle for codebase navigation
//!
//! Uses Active Inference to guide codebase exploration. Files with highest
//! "surprise" (most different from our current model) offer the most
//! information gain — we should explore those first to reduce uncertainty.
//!
//! # Key Concepts
//!
//! - **Surprise**: How much a file differs from our expected codebase model
//! - **Expected Information Gain**: Exploring surprising files reduces free energy
//! - **Model Update**: After exploring, update our expectations (Bayesian update)

use std::path::{Path, PathBuf};

use symthaea_core::hdc::ContinuousHV;

use crate::hdc::code_memory::CodebaseMemory;

/// A file that should be explored for maximum information gain
#[derive(Debug, Clone)]
pub struct ExplorationTarget {
    /// Path to the file
    pub path: PathBuf,
    /// How surprising this file is (0.0 = expected, 1.0 = completely unexpected)
    pub surprise: f32,
    /// Estimated information gain from exploring this file
    pub expected_info_gain: f32,
}

/// Active code explorer using Free Energy Principle
pub struct ActiveCodeExplorer {
    /// Current world model: what we expect the codebase to look like
    expected_model: Option<ContinuousHV>,
    /// Dimension of the HDC space
    dim: usize,
    /// History of explored files for tracking model evolution
    exploration_history: Vec<PathBuf>,
    /// Total surprise accumulated
    total_surprise: f32,
    /// Number of explorations performed
    exploration_count: u32,
}

impl ActiveCodeExplorer {
    /// Create a new explorer with the given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            expected_model: None,
            dim,
            exploration_history: Vec::new(),
            total_surprise: 0.0,
            exploration_count: 0,
        }
    }

    /// Initialize the expected model from a codebase memory
    pub fn initialize_from_memory(&mut self, memory: &CodebaseMemory) {
        self.expected_model = memory.codebase_hv().cloned();
    }

    /// Set the expected model directly
    pub fn set_expected_model(&mut self, model: ContinuousHV) {
        self.expected_model = Some(model);
    }

    /// Compute surprise: how much does this file differ from expectations?
    pub fn compute_surprise(&self, file_hv: &ContinuousHV) -> f32 {
        match &self.expected_model {
            Some(model) => {
                let similarity = model.similarity(file_hv);
                // Surprise = 1 - similarity (high similarity = low surprise)
                (1.0 - similarity).max(0.0)
            }
            None => 0.5, // neutral surprise if no model
        }
    }

    /// Suggest which files to explore next for maximum information gain.
    /// Returns files sorted by expected information gain (highest first).
    pub fn suggest_exploration(
        &self,
        memory: &CodebaseMemory,
        top_k: usize,
    ) -> Vec<ExplorationTarget> {
        let mut targets = Vec::new();

        for path in memory.module_paths() {
            if let Some(hv) = memory.module_hv(path) {
                let surprise = self.compute_surprise(hv);

                // Expected info gain is proportional to surprise
                // but diminished for files we've already explored
                let already_explored = self.exploration_history.contains(&path.to_path_buf());
                let info_gain = if already_explored {
                    surprise * 0.3 // Reduced gain for re-exploration
                } else {
                    surprise
                };

                targets.push(ExplorationTarget {
                    path: path.to_path_buf(),
                    surprise,
                    expected_info_gain: info_gain,
                });
            }
        }

        // Sort by expected information gain descending
        targets.sort_by(|a, b| {
            b.expected_info_gain
                .partial_cmp(&a.expected_info_gain)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        targets.truncate(top_k);
        targets
    }

    /// Update the world model after exploring a file (Bayesian update).
    /// Blends the new observation with our prior expectations.
    pub fn update_model(&mut self, file_hv: &ContinuousHV, path: &Path) {
        let surprise = self.compute_surprise(file_hv);
        self.total_surprise += surprise;
        self.exploration_count += 1;
        self.exploration_history.push(path.to_path_buf());

        match &self.expected_model {
            Some(model) => {
                // Bayesian-like update: blend new observation with prior
                // Weight new evidence proportionally to surprise (more surprising = more weight)
                let blend_weight = 0.1 + surprise * 0.2; // 0.1 to 0.3
                let prior_weight = 1.0 - blend_weight;

                let mut new_values = vec![0.0f32; self.dim];
                let model_len = model.values.len().min(self.dim);
                let file_len = file_hv.values.len().min(self.dim);

                for i in 0..model_len.min(file_len) {
                    new_values[i] =
                        model.values[i] * prior_weight + file_hv.values[i] * blend_weight;
                }

                // Normalize
                let magnitude: f32 = new_values.iter().map(|v| v * v).sum::<f32>().sqrt();
                if magnitude > 0.0 {
                    for v in &mut new_values {
                        *v /= magnitude;
                    }
                }

                self.expected_model = Some(ContinuousHV::from_values(new_values));
            }
            None => {
                self.expected_model = Some(file_hv.clone());
            }
        }
    }

    /// Get the average surprise across all explorations
    pub fn average_surprise(&self) -> f32 {
        if self.exploration_count > 0 {
            self.total_surprise / self.exploration_count as f32
        } else {
            0.0
        }
    }

    /// Get exploration statistics
    pub fn stats(&self) -> ExplorerStats {
        ExplorerStats {
            explorations: self.exploration_count,
            total_surprise: self.total_surprise,
            average_surprise: self.average_surprise(),
            has_model: self.expected_model.is_some(),
            unique_files_explored: self.exploration_history.len(),
        }
    }
}

/// Statistics from the code explorer
#[derive(Debug, Clone)]
pub struct ExplorerStats {
    pub explorations: u32,
    pub total_surprise: f32,
    pub average_surprise: f32,
    pub has_model: bool,
    pub unique_files_explored: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::code_encoder::CodeHDEncoder;
    use crate::hdc::code_memory::CodebaseMemory;
    use crate::language::code_parser::{Entity, EntityKind, ParsedCode, Span};

    fn test_span() -> Span {
        Span {
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 10,
        }
    }

    #[test]
    fn test_explorer_no_model() {
        let explorer = ActiveCodeExplorer::new(512);
        let hv = ContinuousHV::random(512, 42);
        let surprise = explorer.compute_surprise(&hv);
        assert!((surprise - 0.5).abs() < 1e-6); // neutral without model
    }

    #[test]
    fn test_explorer_with_model() {
        let mut explorer = ActiveCodeExplorer::new(512);
        let model = ContinuousHV::random(512, 42);
        explorer.set_expected_model(model.clone());

        // Same vector should have low surprise
        let surprise_same = explorer.compute_surprise(&model);
        assert!(
            surprise_same < 0.1,
            "Same vector should have low surprise: {}",
            surprise_same
        );

        // Random vector should have higher surprise
        let random = ContinuousHV::random(512, 99);
        let surprise_random = explorer.compute_surprise(&random);
        assert!(surprise_random > surprise_same);
    }

    #[test]
    fn test_model_update() {
        let mut explorer = ActiveCodeExplorer::new(512);
        let initial = ContinuousHV::random(512, 42);
        explorer.set_expected_model(initial);

        let new_file = ContinuousHV::random(512, 99);
        explorer.update_model(&new_file, Path::new("test.rs"));

        assert_eq!(explorer.exploration_count, 1);
        assert!(explorer.expected_model.is_some());
        assert!(explorer.total_surprise > 0.0);
    }

    #[test]
    fn test_suggest_exploration() {
        let encoder = CodeHDEncoder::new(512);
        let mut memory = CodebaseMemory::new(CodeHDEncoder::new(512));

        // Index diverse files
        let mut p1 = ParsedCode::new("", "rust");
        p1.entities
            .push(Entity::new(EntityKind::Function, "sort", test_span()));
        memory.index_file(Path::new("sort.rs"), &p1);

        let mut p2 = ParsedCode::new("", "rust");
        p2.entities
            .push(Entity::new(EntityKind::Function, "connect", test_span()));
        memory.index_file(Path::new("network.rs"), &p2);

        let mut explorer = ActiveCodeExplorer::new(512);
        explorer.initialize_from_memory(&memory);

        let suggestions = explorer.suggest_exploration(&memory, 5);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_exploration_stats() {
        let mut explorer = ActiveCodeExplorer::new(512);
        explorer.set_expected_model(ContinuousHV::random(512, 42));

        let stats = explorer.stats();
        assert_eq!(stats.explorations, 0);
        assert!(stats.has_model);

        explorer.update_model(&ContinuousHV::random(512, 99), Path::new("a.rs"));
        explorer.update_model(&ContinuousHV::random(512, 100), Path::new("b.rs"));

        let stats = explorer.stats();
        assert_eq!(stats.explorations, 2);
        assert_eq!(stats.unique_files_explored, 2);
        assert!(stats.average_surprise > 0.0);
    }
}
