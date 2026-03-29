// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Code Understanding via CfC
//!
//! Feeds git commit history through a CfC network to understand code
//! evolution patterns. The CfC state encodes the "trajectory" of the
//! codebase, enabling predictions about what changes are likely needed next.
//!
//! # Architecture
//!
//! ```text
//! Git History (commits)
//!     ↓ encode each diff as HV
//! Diff HV sequence
//!     ↓ feed through CfC
//! CfC State (trajectory encoding)
//!     ↓ analyze
//! EvolutionPattern (stability, predicted next change)
//! ```

use symthaea_core::hdc::ContinuousHV;

use crate::hdc::code_encoder::CodeHDEncoder;

/// A simplified git commit representation
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Commit hash (short)
    pub hash: String,
    /// Commit message
    pub message: String,
    /// Files changed as (path, change_type) pairs
    pub files_changed: Vec<(String, ChangeType)>,
    /// Timestamp (unix epoch)
    pub timestamp: u64,
}

/// Type of change in a commit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    Added,
    Modified,
    Deleted,
    Renamed,
}

/// Result of analyzing code evolution
#[derive(Debug, Clone)]
pub struct EvolutionPattern {
    /// The CfC trajectory state after processing all history
    pub trajectory: Vec<f32>,
    /// How stable the codebase is (0.0 = volatile, 1.0 = stable)
    pub stability: f32,
    /// Predicted category of next change
    pub predicted_next: PredictedChange,
    /// Number of commits analyzed
    pub commits_analyzed: usize,
}

/// Prediction about the next likely change
#[derive(Debug, Clone)]
pub struct PredictedChange {
    /// Category of predicted change
    pub category: ChangeCategory,
    /// Confidence in prediction (0.0 - 1.0)
    pub confidence: f32,
}

/// Categories of code changes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeCategory {
    /// Bug fix
    BugFix,
    /// New feature
    NewFeature,
    /// Refactoring
    Refactor,
    /// Documentation
    Documentation,
    /// Testing
    Testing,
    /// Maintenance (deps, CI, etc.)
    Maintenance,
    /// Unknown
    Unknown,
}

/// CfC-based temporal code analyzer
pub struct TemporalCodeAnalyzer {
    encoder: CodeHDEncoder,
    /// Hidden dimension for the simplified CfC state
    hidden_dim: usize,
    /// Time constants
    tau: Vec<f32>,
    /// Recurrent weights
    w_h: Vec<f32>,
    /// Projection from HDC to hidden
    projection: Vec<f32>,
}

impl TemporalCodeAnalyzer {
    /// Create a new temporal analyzer
    pub fn new(encoder: CodeHDEncoder, hidden_dim: usize) -> Self {
        let hdc_dim = encoder.dim();

        // Projection matrix
        let mut projection = vec![0.0f32; hdc_dim * hidden_dim];
        for i in 0..projection.len() {
            let seed = (i as u64).wrapping_mul(7_919).wrapping_add(13);
            let frac = (seed % 10000) as f32 / 10000.0;
            projection[i] = (frac * 2.0 - 1.0) / (hdc_dim as f32).sqrt();
        }

        // Time constants
        let tau: Vec<f32> = (0..hidden_dim)
            .map(|i| 1.0 + (i as f32 / hidden_dim as f32) * 19.0)
            .collect();

        // Recurrent weights
        let mut w_h = vec![0.0f32; hidden_dim * hidden_dim];
        for i in 0..hidden_dim {
            w_h[i * hidden_dim + i] = 0.95;
        }

        Self {
            encoder,
            hidden_dim,
            tau,
            w_h,
            projection,
        }
    }

    /// Create with default settings
    pub fn with_defaults() -> Self {
        Self::new(CodeHDEncoder::new(512), 32)
    }

    /// Analyze code evolution from a sequence of commits
    pub fn analyze_evolution(&self, commits: &[CommitInfo]) -> EvolutionPattern {
        if commits.is_empty() {
            return EvolutionPattern {
                trajectory: vec![0.0; self.hidden_dim],
                stability: 1.0,
                predicted_next: PredictedChange {
                    category: ChangeCategory::Unknown,
                    confidence: 0.0,
                },
                commits_analyzed: 0,
            };
        }

        let mut state = vec![0.0f32; self.hidden_dim];
        let mut state_norms = Vec::new();

        for commit in commits {
            // Encode the commit as an HV
            let commit_hv = self.encode_commit(commit);

            // Project to hidden space
            let input = self.project_to_hidden(&commit_hv);

            // CfC step with time delta
            let dt = 1.0; // Simplified: uniform time steps
            state = self.cfc_step(&state, &input, dt);

            // Track state norm for stability
            let norm: f32 = state.iter().map(|v| v * v).sum::<f32>().sqrt();
            state_norms.push(norm);
        }

        // Compute stability from variance of state norms
        let stability = self.compute_stability(&state_norms);

        // Predict next change from final state
        let predicted_next = self.predict_next_change(&state);

        EvolutionPattern {
            trajectory: state,
            stability,
            predicted_next,
            commits_analyzed: commits.len(),
        }
    }

    /// Encode a commit into an HDC vector
    fn encode_commit(&self, commit: &CommitInfo) -> ContinuousHV {
        // Encode message
        let msg_hv = self.encoder.encode_name(&commit.message);

        // Encode changed files
        let file_hvs: Vec<ContinuousHV> = commit
            .files_changed
            .iter()
            .map(|(path, _)| self.encoder.encode_name(path))
            .collect();

        if file_hvs.is_empty() {
            return msg_hv;
        }

        let files_bundled = ContinuousHV::bundle_owned(&file_hvs);

        ContinuousHV::bundle_owned(&[msg_hv, files_bundled])
    }

    /// Project HV to hidden space
    fn project_to_hidden(&self, hv: &ContinuousHV) -> Vec<f32> {
        let hdc_dim = hv.values.len().min(self.encoder.dim());
        let mut hidden = vec![0.0f32; self.hidden_dim];

        for h in 0..self.hidden_dim {
            let mut sum = 0.0f32;
            for i in 0..hdc_dim {
                sum += hv.values[i] * self.projection[i * self.hidden_dim + h];
            }
            hidden[h] = sum.tanh();
        }

        hidden
    }

    /// CfC step with input
    fn cfc_step(&self, state: &[f32], input: &[f32], dt: f32) -> Vec<f32> {
        let mut new_state = vec![0.0f32; self.hidden_dim];

        for i in 0..self.hidden_dim {
            let mut recurrent = 0.0f32;
            for j in 0..self.hidden_dim {
                recurrent += self.w_h[i * self.hidden_dim + j] * state[j];
            }

            // Add input
            let total_input = recurrent + input[i];

            // CfC closed-form update
            let decay = (-dt / self.tau[i]).exp();
            let activation = total_input.tanh();
            new_state[i] = state[i] * decay + activation * (1.0 - decay);
        }

        new_state
    }

    /// Compute stability from state norm history
    fn compute_stability(&self, norms: &[f32]) -> f32 {
        if norms.len() < 2 {
            return 1.0;
        }

        let mean: f32 = norms.iter().sum::<f32>() / norms.len() as f32;
        let variance: f32 =
            norms.iter().map(|n| (n - mean).powi(2)).sum::<f32>() / norms.len() as f32;

        // Convert variance to stability (0-1)
        // Low variance = high stability
        (1.0 / (1.0 + variance.sqrt())).clamp(0.0, 1.0)
    }

    /// Predict the next change category from final CfC state
    fn predict_next_change(&self, state: &[f32]) -> PredictedChange {
        // Use different regions of the state to map to categories
        if self.hidden_dim < 6 {
            return PredictedChange {
                category: ChangeCategory::Unknown,
                confidence: 0.0,
            };
        }

        let categories = [
            (
                ChangeCategory::BugFix,
                state[0..self.hidden_dim / 6]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>(),
            ),
            (
                ChangeCategory::NewFeature,
                state[self.hidden_dim / 6..2 * self.hidden_dim / 6]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>(),
            ),
            (
                ChangeCategory::Refactor,
                state[2 * self.hidden_dim / 6..3 * self.hidden_dim / 6]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>(),
            ),
            (
                ChangeCategory::Documentation,
                state[3 * self.hidden_dim / 6..4 * self.hidden_dim / 6]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>(),
            ),
            (
                ChangeCategory::Testing,
                state[4 * self.hidden_dim / 6..5 * self.hidden_dim / 6]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>(),
            ),
            (
                ChangeCategory::Maintenance,
                state[5 * self.hidden_dim / 6..]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>(),
            ),
        ];

        let total: f32 = categories.iter().map(|(_, v)| v).sum();
        if total == 0.0 {
            return PredictedChange {
                category: ChangeCategory::Unknown,
                confidence: 0.0,
            };
        }

        let (best_cat, best_score) = categories
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("categories is non-empty (checked via total > 0)");

        PredictedChange {
            category: *best_cat,
            confidence: (best_score / total).clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_commit(message: &str, files: &[&str]) -> CommitInfo {
        CommitInfo {
            hash: "abc123".to_string(),
            message: message.to_string(),
            files_changed: files
                .iter()
                .map(|f| (f.to_string(), ChangeType::Modified))
                .collect(),
            timestamp: 0,
        }
    }

    #[test]
    fn test_empty_history() {
        let analyzer = TemporalCodeAnalyzer::with_defaults();
        let result = analyzer.analyze_evolution(&[]);
        assert_eq!(result.commits_analyzed, 0);
        assert_eq!(result.stability, 1.0);
    }

    #[test]
    fn test_single_commit() {
        let analyzer = TemporalCodeAnalyzer::with_defaults();
        let commits = vec![make_commit("fix: sort bug", &["src/sort.rs"])];
        let result = analyzer.analyze_evolution(&commits);
        assert_eq!(result.commits_analyzed, 1);
        assert!(result.stability > 0.0);
    }

    #[test]
    fn test_multiple_commits() {
        let analyzer = TemporalCodeAnalyzer::with_defaults();
        let commits = vec![
            make_commit("feat: add sorting", &["src/sort.rs"]),
            make_commit("fix: edge case in sort", &["src/sort.rs"]),
            make_commit("docs: update readme", &["README.md"]),
            make_commit("test: add sort tests", &["tests/sort_test.rs"]),
        ];

        let result = analyzer.analyze_evolution(&commits);
        assert_eq!(result.commits_analyzed, 4);
        assert!(result.stability > 0.0);
        assert!(result.predicted_next.confidence > 0.0);
    }

    #[test]
    fn test_stability_calculation() {
        let analyzer = TemporalCodeAnalyzer::with_defaults();

        // Uniform state norms = high stability
        let uniform = vec![1.0, 1.0, 1.0, 1.0];
        let stability = analyzer.compute_stability(&uniform);
        assert!(
            stability > 0.9,
            "Uniform norms should be stable: {}",
            stability
        );

        // Volatile state norms = lower stability
        let volatile = vec![0.1, 5.0, 0.2, 4.0];
        let stability2 = analyzer.compute_stability(&volatile);
        assert!(stability2 < stability, "Volatile should be less stable");
    }

    #[test]
    fn test_trajectory_bounded() {
        let analyzer = TemporalCodeAnalyzer::with_defaults();
        let commits: Vec<CommitInfo> = (0..50)
            .map(|i| make_commit(&format!("commit {}", i), &["src/main.rs"]))
            .collect();

        let result = analyzer.analyze_evolution(&commits);

        // Trajectory should remain bounded
        for v in &result.trajectory {
            assert!(v.is_finite());
            assert!(v.abs() < 10.0, "Trajectory should be bounded: {}", v);
        }
    }
}
