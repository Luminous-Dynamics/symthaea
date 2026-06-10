// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Global Workspace Theory — attention bottleneck for the Spore WASM kernel.
//!
//! Port of the desktop UnifiedGlobalWorkspace / GwtManager into a lightweight,
//! WASM-compatible module. Implements Baars' Global Workspace Theory: multiple
//! content streams compete for broadcast to a shared "conscious" workspace.
//! Only the winning content becomes globally available.
//!
//! Fully WASM-compatible: no std::time, no filesystem, no tokio.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default workspace capacity (how many items can be "conscious" at once).
/// Typically 1-3, matching human attentional bottleneck (Cowan 2001).
const DEFAULT_CAPACITY: usize = 3;

/// Default minimum salience to enter the workspace.
const DEFAULT_BROADCAST_THRESHOLD: f32 = 0.2;

/// Activation decay per cycle for non-broadcast items.
const ACTIVATION_DECAY: f32 = 0.9;

/// Broadcast history ring buffer size.
const HISTORY_SIZE: usize = 20;

/// Ignition threshold: minimum total workspace activation for "conscious" experience.
/// Based on Dehaene & Changeux (2011) ignition model.
const IGNITION_THRESHOLD: f32 = 0.4;

/// Phi contribution scaling factor from workspace activation.
const PHI_CONTRIBUTION_SCALE: f32 = 0.15;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A content item competing for workspace access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceContent {
    /// Source module: "perception", "memory", "reasoning", "emotion", etc.
    pub source: String,
    /// Human-readable description of the content.
    pub description: String,
    /// How strongly it competes (0-1).
    pub salience: f32,
    /// Current activation level (0-1). Decays each cycle.
    pub activation: f32,
    /// How many times this content has been broadcast.
    pub broadcast_count: u32,
}

/// State of the workspace for WASM/JSON export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceState {
    /// Currently broadcast content (if any).
    pub current_broadcast: Option<WorkspaceContent>,
    /// All contents currently competing.
    pub contents: Vec<WorkspaceContent>,
    /// Recent broadcast history (descriptions).
    pub recent_broadcasts: Vec<String>,
    /// Whether the workspace has reached ignition.
    pub ignited: bool,
    /// Total activation across all contents.
    pub total_activation: f32,
    /// Phi contribution from the workspace.
    pub phi_contribution: f32,
}

/// Result of a competition round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitionResult {
    /// Index of the winning content (if any).
    pub winner_index: Option<usize>,
    /// Description of the winning content.
    pub winner_description: Option<String>,
    /// Number of candidates that competed.
    pub candidates: usize,
    /// Whether ignition occurred.
    pub ignited: bool,
}

// ---------------------------------------------------------------------------
// GlobalWorkspace
// ---------------------------------------------------------------------------

/// Global Workspace — attention bottleneck where contents compete for broadcast.
///
/// Implements Baars' Global Workspace Theory (1988, 2005): multiple specialized
/// processors submit content to a shared workspace. Through competition,
/// the highest-salience content wins broadcast access and becomes "conscious"
/// (globally available to all modules).
pub struct GlobalWorkspace {
    contents: Vec<WorkspaceContent>,
    broadcast_history: Vec<String>,
    capacity: usize,
    broadcast_threshold: f32,
    current_broadcast: Option<usize>,
    total_broadcasts: u64,
}

impl GlobalWorkspace {
    /// Create a new GlobalWorkspace with default settings.
    pub fn new() -> Self {
        Self {
            contents: Vec::with_capacity(16),
            broadcast_history: Vec::with_capacity(HISTORY_SIZE),
            capacity: DEFAULT_CAPACITY,
            broadcast_threshold: DEFAULT_BROADCAST_THRESHOLD,
            current_broadcast: None,
            total_broadcasts: 0,
        }
    }

    /// Create with custom capacity and threshold.
    pub fn with_config(capacity: usize, broadcast_threshold: f32) -> Self {
        Self {
            contents: Vec::with_capacity(16),
            broadcast_history: Vec::with_capacity(HISTORY_SIZE),
            capacity: capacity.max(1),
            broadcast_threshold: broadcast_threshold.clamp(0.0, 1.0),
            current_broadcast: None,
            total_broadcasts: 0,
        }
    }

    /// Submit content for workspace competition.
    ///
    /// Content enters the workspace with its salience as initial activation.
    /// If we already have content from the same source, update it rather
    /// than creating a duplicate.
    pub fn submit(&mut self, source: &str, description: &str, salience: f32) {
        let salience = salience.clamp(0.0, 1.0);

        // Update existing content from same source, or add new
        if let Some(existing) = self.contents.iter_mut().find(|c| c.source == source) {
            existing.description = description.to_string();
            existing.salience = salience;
            existing.activation = salience; // refresh activation
        } else {
            self.contents.push(WorkspaceContent {
                source: source.to_string(),
                description: description.to_string(),
                salience,
                activation: salience,
                broadcast_count: 0,
            });
        }
    }

    /// Run competition: select the highest-salience content for broadcast.
    ///
    /// 1. Apply activation decay to all non-broadcast items
    /// 2. Remove items below the broadcast threshold
    /// 3. Select the winner (highest activation)
    /// 4. Record in broadcast history
    pub fn compete(&mut self) -> CompetitionResult {
        // Apply activation decay
        for content in &mut self.contents {
            content.activation *= ACTIVATION_DECAY;
        }

        // Remove sub-threshold items
        self.contents
            .retain(|c| c.activation >= self.broadcast_threshold * 0.5);

        let candidates = self.contents.len();

        // Find winner: highest activation above threshold
        let winner_index = self
            .contents
            .iter()
            .enumerate()
            .filter(|(_, c)| c.activation >= self.broadcast_threshold)
            .max_by(|(_, a), (_, b)| {
                a.activation
                    .partial_cmp(&b.activation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        // Broadcast the winner
        if let Some(idx) = winner_index {
            self.contents[idx].broadcast_count += 1;
            self.current_broadcast = Some(idx);
            self.total_broadcasts += 1;

            // Record in history
            let desc = self.contents[idx].description.clone();
            if self.broadcast_history.len() >= HISTORY_SIZE {
                self.broadcast_history.remove(0);
            }
            self.broadcast_history.push(desc.clone());

            let ignited = self.ignition_check();

            CompetitionResult {
                winner_index: Some(idx),
                winner_description: Some(desc),
                candidates,
                ignited,
            }
        } else {
            self.current_broadcast = None;
            CompetitionResult {
                winner_index: None,
                winner_description: None,
                candidates,
                ignited: false,
            }
        }
    }

    /// The currently broadcast content (globally "conscious").
    pub fn current_broadcast(&self) -> Option<&WorkspaceContent> {
        self.current_broadcast
            .and_then(|idx| self.contents.get(idx))
    }

    /// Last N broadcast item descriptions.
    pub fn broadcast_history(&self, n: usize) -> Vec<String> {
        let start = if self.broadcast_history.len() > n {
            self.broadcast_history.len() - n
        } else {
            0
        };
        self.broadcast_history[start..].to_vec()
    }

    /// Check if the workspace has enough activation for conscious experience.
    ///
    /// Ignition occurs when the total activation exceeds the threshold,
    /// following Dehaene & Changeux's (2011) neural ignition model.
    pub fn ignition_check(&self) -> bool {
        let total_activation: f32 = self.contents.iter().map(|c| c.activation).sum();
        total_activation >= IGNITION_THRESHOLD
    }

    /// How much does the workspace contribute to overall Phi?
    ///
    /// Workspace contribution is proportional to:
    /// 1. Number of active streams (integration)
    /// 2. Activation spread (information)
    /// 3. Whether ignition has occurred
    pub fn workspace_phi_contribution(&self) -> f32 {
        if self.contents.is_empty() {
            return 0.0;
        }

        let n = self.contents.len() as f32;
        let total_activation: f32 = self.contents.iter().map(|c| c.activation).sum();
        let mean_activation = total_activation / n;

        // Integration: more active streams = more integration
        let integration = (n / self.capacity as f32).clamp(0.0, 1.0);

        // Information: variance in activation = differentiated content
        let variance: f32 = self
            .contents
            .iter()
            .map(|c| (c.activation - mean_activation).powi(2))
            .sum::<f32>()
            / n;

        let ignition_bonus = if self.ignition_check() { 0.2 } else { 0.0 };

        (PHI_CONTRIBUTION_SCALE * integration * (mean_activation + variance.sqrt())
            + ignition_bonus)
            .clamp(0.0, 1.0)
    }

    /// Number of contents currently in the workspace.
    pub fn content_count(&self) -> usize {
        self.contents.len()
    }

    /// Total broadcasts since creation.
    pub fn total_broadcasts(&self) -> u64 {
        self.total_broadcasts
    }

    /// Export full state for WASM/JSON serialization.
    pub fn state(&self) -> WorkspaceState {
        WorkspaceState {
            current_broadcast: self.current_broadcast().cloned(),
            contents: self.contents.clone(),
            recent_broadcasts: self.broadcast_history(5),
            ignited: self.ignition_check(),
            total_activation: self.contents.iter().map(|c| c.activation).sum(),
            phi_contribution: self.workspace_phi_contribution(),
        }
    }
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_and_compete() {
        let mut ws = GlobalWorkspace::new();

        ws.submit("perception", "saw a cat", 0.8);
        ws.submit("memory", "remembered a dog", 0.4);
        ws.submit("reasoning", "inferred relationship", 0.6);

        assert_eq!(ws.content_count(), 3);

        let result = ws.compete();
        assert!(result.winner_index.is_some());
        // Perception has highest salience, should win
        assert_eq!(result.winner_description.unwrap(), "saw a cat");
        assert_eq!(result.candidates, 3);
    }

    #[test]
    fn test_activation_decay() {
        let mut ws = GlobalWorkspace::new();
        ws.submit("test", "decaying content", 0.5);

        // After several competition rounds without refreshing, activation should decay
        for _ in 0..10 {
            ws.compete();
        }

        let content = &ws.contents[0];
        assert!(
            content.activation < 0.5,
            "activation should decay: {}",
            content.activation
        );
    }

    #[test]
    fn test_broadcast_history() {
        let mut ws = GlobalWorkspace::new();

        for i in 0..5 {
            ws.submit("source", &format!("event {}", i), 0.8);
            ws.compete();
        }

        let history = ws.broadcast_history(3);
        assert!(history.len() <= 3);
        assert_eq!(ws.total_broadcasts(), 5);
    }

    #[test]
    fn test_ignition_check() {
        let mut ws = GlobalWorkspace::new();

        // Single low-salience item: no ignition
        ws.submit("a", "weak", 0.1);
        assert!(!ws.ignition_check());

        // Multiple high-salience items: ignition
        ws.submit("b", "strong1", 0.8);
        ws.submit("c", "strong2", 0.7);
        assert!(ws.ignition_check());
    }

    #[test]
    fn test_phi_contribution() {
        let mut ws = GlobalWorkspace::new();

        // Empty workspace contributes nothing
        assert_eq!(ws.workspace_phi_contribution(), 0.0);

        // Active workspace contributes positively
        ws.submit("perception", "visual", 0.8);
        ws.submit("memory", "recall", 0.6);
        ws.submit("emotion", "joy", 0.7);

        let phi = ws.workspace_phi_contribution();
        assert!(
            phi > 0.0,
            "active workspace should contribute to Phi: {}",
            phi
        );
    }

    #[test]
    fn test_source_dedup() {
        let mut ws = GlobalWorkspace::new();

        ws.submit("perception", "first observation", 0.5);
        ws.submit("perception", "updated observation", 0.9);

        assert_eq!(ws.content_count(), 1);
        assert_eq!(ws.contents[0].description, "updated observation");
        assert!((ws.contents[0].salience - 0.9).abs() < 1e-6);
    }
}
