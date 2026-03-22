// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Attention Visualization
//!
//! Tools for capturing, analyzing, and visualizing Phi-gated attention patterns.
//!
//! This module provides mechanisms to understand how consciousness (Phi values)
//! influence information flow through attention mechanisms.

use crate::attention::PhiAttentionResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// ATTENTION SNAPSHOT
// ============================================================================

/// A snapshot of attention state at a single point in time.
///
/// Captures all relevant information needed to understand and debug
/// a Phi-gated attention decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSnapshot {
    /// Human-readable names for inputs (if provided)
    pub input_names: Vec<String>,

    /// Indices of inputs (0-indexed)
    pub input_indices: Vec<usize>,

    /// Raw Phi values at each position (consciousness measure)
    pub phi_values: Vec<f64>,

    /// Computed attention weights (sum to 1.0)
    pub attention_weights: Vec<f32>,

    /// Transformed Phi values (after learnable mapping if enabled)
    pub transformed_phi: Vec<f32>,

    /// Temperature used for softmax
    pub temperature: f32,

    /// Entropy of the attention distribution (higher = more spread)
    pub entropy: f32,

    /// Timestamp when snapshot was captured
    pub timestamp: DateTime<Utc>,

    /// Optional metadata for additional context
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl AttentionSnapshot {
    /// Create a snapshot from a PhiAttentionResult
    ///
    /// # Arguments
    /// * `result` - The attention computation result
    /// * `phi_values` - Original Phi values used
    /// * `input_names` - Human-readable names for inputs
    /// * `temperature` - Temperature used in computation
    pub fn from_result(
        result: &PhiAttentionResult,
        phi_values: &[f64],
        input_names: Vec<&str>,
        temperature: f32,
    ) -> Self {
        let n = result.weights.len();
        Self {
            input_names: input_names.iter().map(|s| s.to_string()).collect(),
            input_indices: (0..n).collect(),
            phi_values: phi_values.to_vec(),
            attention_weights: result.weights.clone(),
            transformed_phi: result.transformed_phi.clone(),
            temperature,
            entropy: result.entropy,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a snapshot with indexed inputs (no names)
    pub fn from_result_indexed(
        result: &PhiAttentionResult,
        phi_values: &[f64],
        temperature: f32,
    ) -> Self {
        let n = result.weights.len();
        Self {
            input_names: (0..n).map(|i| format!("input_{i}")).collect(),
            input_indices: (0..n).collect(),
            phi_values: phi_values.to_vec(),
            attention_weights: result.weights.clone(),
            transformed_phi: result.transformed_phi.clone(),
            temperature,
            entropy: result.entropy,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a snapshot manually (for testing or custom scenarios)
    pub fn new(
        input_names: Vec<String>,
        phi_values: Vec<f64>,
        attention_weights: Vec<f32>,
        temperature: f32,
    ) -> Self {
        let n = phi_values.len();
        let entropy = compute_entropy(&attention_weights);
        Self {
            input_names,
            input_indices: (0..n).collect(),
            phi_values,
            attention_weights: attention_weights.clone(),
            transformed_phi: attention_weights, // Default: no transformation
            temperature,
            entropy,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the snapshot
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get the index of the input with highest attention
    pub fn argmax(&self) -> Option<usize> {
        self.attention_weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Get top-k attended inputs by weight
    ///
    /// Returns Vec of (index, name, weight) tuples
    pub fn top_k_attended(&self, k: usize) -> Vec<(usize, String, f32)> {
        let mut indexed: Vec<(usize, f32)> =
            self.attention_weights.iter().copied().enumerate().collect();

        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        indexed
            .into_iter()
            .map(|(i, w)| {
                let name = self
                    .input_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("input_{i}"));
                (i, name, w)
            })
            .collect()
    }

    /// Compute entropy of attention distribution
    ///
    /// Higher entropy means more uniform attention (less focused).
    /// Lower entropy means concentrated attention (more focused).
    ///
    /// Returns value in range [0, ln(n)] where n is number of inputs.
    pub fn attention_entropy(&self) -> f32 {
        self.entropy
    }

    /// Check if attention is focused (low entropy)
    pub fn is_focused(&self) -> bool {
        let max_entropy = (self.attention_weights.len() as f32).ln();
        self.entropy < max_entropy * 0.5
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Generate a single-row ASCII visualization
    ///
    /// Shows attention weights as a bar chart using Unicode block characters.
    pub fn to_ascii_bar(&self) -> String {
        let blocks = [
            ' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}',
            '\u{2587}', '\u{2588}',
        ];

        let max_weight = self
            .attention_weights
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        let scale = if max_weight > 0.0 {
            1.0 / max_weight
        } else {
            1.0
        };

        let bar: String = self
            .attention_weights
            .iter()
            .map(|&w| {
                let normalized = (w * scale * 8.0).clamp(0.0, 8.0) as usize;
                blocks[normalized]
            })
            .collect();

        bar
    }

    /// Generate detailed ASCII representation
    pub fn to_ascii_detail(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "Attention Snapshot @ {} (T={:.2}, H={:.4})",
            self.timestamp.format("%H:%M:%S%.3f"),
            self.temperature,
            self.entropy
        ));
        lines.push("-".repeat(60));

        let max_name_len = self.input_names.iter().map(|n| n.len()).max().unwrap_or(10);

        for i in 0..self.attention_weights.len() {
            let name = self
                .input_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("input_{i}"));
            let phi = self.phi_values.get(i).copied().unwrap_or(0.0);
            let weight = self.attention_weights[i];

            // Create bar visualization
            let bar_len = (weight * 40.0).round() as usize;
            let bar = "#".repeat(bar_len);

            lines.push(format!(
                "{name:>max_name_len$} | Phi={phi:.3} | W={weight:.4} |{bar}"
            ));
        }

        lines.join("\n")
    }
}

// ============================================================================
// ATTENTION HISTORY
// ============================================================================

/// Tracks attention patterns over multiple inference steps.
///
/// Useful for understanding how attention evolves over time,
/// identifying patterns, and debugging attention drift.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AttentionHistory {
    /// Ordered list of snapshots
    snapshots: Vec<AttentionSnapshot>,

    /// Maximum number of snapshots to retain (0 = unlimited)
    max_snapshots: usize,
}

impl AttentionHistory {
    /// Create a new empty history
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots: 0,
        }
    }

    /// Create a history with a maximum size (rolling window)
    pub fn with_max_size(max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots,
        }
    }

    /// Record a new snapshot
    pub fn record(&mut self, snapshot: AttentionSnapshot) {
        self.snapshots.push(snapshot);

        // Trim if over limit
        if self.max_snapshots > 0 && self.snapshots.len() > self.max_snapshots {
            let excess = self.snapshots.len() - self.max_snapshots;
            self.snapshots.drain(0..excess);
        }
    }

    /// Get all snapshots
    pub fn snapshots(&self) -> &[AttentionSnapshot] {
        &self.snapshots
    }

    /// Get number of recorded snapshots
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Clear all snapshots
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Get the most recent snapshot
    pub fn latest(&self) -> Option<&AttentionSnapshot> {
        self.snapshots.last()
    }

    /// Get top-k most frequently attended inputs across all snapshots
    ///
    /// Returns Vec of (name, average_weight) sorted by frequency
    pub fn top_k_overall(&self, k: usize) -> Vec<(String, f32)> {
        if self.snapshots.is_empty() {
            return Vec::new();
        }

        // Aggregate weights by input name
        let mut weight_sums: HashMap<String, (f32, usize)> = HashMap::new();

        for snapshot in &self.snapshots {
            for (i, &weight) in snapshot.attention_weights.iter().enumerate() {
                let name = snapshot
                    .input_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("input_{i}"));
                let entry = weight_sums.entry(name).or_insert((0.0, 0));
                entry.0 += weight;
                entry.1 += 1;
            }
        }

        // Compute averages and sort
        let mut results: Vec<(String, f32)> = weight_sums
            .into_iter()
            .map(|(name, (sum, count))| (name, sum / count as f32))
            .collect();

        results.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Compute average entropy across all snapshots
    pub fn average_entropy(&self) -> f32 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.snapshots.iter().map(|s| s.entropy).sum();
        sum / self.snapshots.len() as f32
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Generate ASCII heatmap of attention over time
    ///
    /// Rows are inputs, columns are time steps.
    /// Intensity represents attention weight.
    pub fn to_ascii_heatmap(&self) -> String {
        if self.snapshots.is_empty() {
            return "No attention data recorded".to_string();
        }

        // Collect all unique input names
        let mut all_names: Vec<String> = Vec::new();
        for snapshot in &self.snapshots {
            for name in &snapshot.input_names {
                if !all_names.contains(name) {
                    all_names.push(name.clone());
                }
            }
        }

        if all_names.is_empty() {
            return "No inputs found".to_string();
        }

        let blocks = [' ', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];
        let max_name_len = all_names.iter().map(|n| n.len()).max().unwrap_or(10);

        let mut lines = Vec::new();

        // Header
        lines.push(format!(
            "{:>width$} | Attention Heatmap (time ->)",
            "Input",
            width = max_name_len
        ));
        lines.push(format!(
            "{:>width$} | {}",
            "",
            "-".repeat(self.snapshots.len().min(80)),
            width = max_name_len
        ));

        // One row per input
        for name in &all_names {
            let mut row = String::new();

            for snapshot in &self.snapshots {
                // Find this input in the snapshot
                let weight = snapshot
                    .input_names
                    .iter()
                    .position(|n| n == name)
                    .and_then(|i| snapshot.attention_weights.get(i).copied())
                    .unwrap_or(0.0);

                // Map weight to block character
                let block_idx = (weight * 4.0).round() as usize;
                let block_idx = block_idx.min(4);
                row.push(blocks[block_idx]);
            }

            lines.push(format!("{name:>max_name_len$} | {row}"));
        }

        // Footer with statistics
        lines.push(format!(
            "{:>width$} | {}",
            "",
            "-".repeat(self.snapshots.len().min(80)),
            width = max_name_len
        ));
        lines.push(format!(
            "Snapshots: {} | Avg entropy: {:.4}",
            self.snapshots.len(),
            self.average_entropy()
        ));

        lines.join("\n")
    }

    /// Generate time series data for plotting
    ///
    /// Returns a map from input name to Vec of (timestamp, weight) pairs
    pub fn to_time_series(&self) -> HashMap<String, Vec<(DateTime<Utc>, f32)>> {
        let mut series: HashMap<String, Vec<(DateTime<Utc>, f32)>> = HashMap::new();

        for snapshot in &self.snapshots {
            for (i, &weight) in snapshot.attention_weights.iter().enumerate() {
                let name = snapshot
                    .input_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("input_{i}"));
                series
                    .entry(name)
                    .or_default()
                    .push((snapshot.timestamp, weight));
            }
        }

        series
    }
}

// ============================================================================
// ATTENTION FLOW GRAPH
// ============================================================================

/// Represents an edge in the attention flow graph.
///
/// Shows how attention flows from inputs to outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlowEdge {
    /// Source input index
    pub source_idx: usize,

    /// Source input name
    pub source_name: String,

    /// Target (output) identifier
    pub target: String,

    /// Attention weight on this edge
    pub weight: f32,

    /// Phi value of the source
    pub source_phi: f64,
}

/// Graph representation of attention flow for visualization.
///
/// Can be exported to various formats for external visualization tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlowGraph {
    /// Input nodes
    pub inputs: Vec<AttentionFlowNode>,

    /// Output nodes
    pub outputs: Vec<AttentionFlowNode>,

    /// Edges (attention flow)
    pub edges: Vec<AttentionFlowEdge>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// A node in the attention flow graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlowNode {
    /// Node identifier
    pub id: String,

    /// Node label (human-readable)
    pub label: String,

    /// Node type (input/output)
    pub node_type: String,

    /// Associated value (Phi for inputs, aggregate weight for outputs)
    pub value: f64,
}

impl AttentionFlowGraph {
    /// Create a flow graph from a snapshot
    pub fn from_snapshot(snapshot: &AttentionSnapshot, output_name: &str) -> Self {
        let inputs: Vec<AttentionFlowNode> = snapshot
            .input_names
            .iter()
            .enumerate()
            .map(|(i, name)| AttentionFlowNode {
                id: format!("input_{i}"),
                label: name.clone(),
                node_type: "input".to_string(),
                value: snapshot.phi_values.get(i).copied().unwrap_or(0.0),
            })
            .collect();

        let outputs = vec![AttentionFlowNode {
            id: "output_0".to_string(),
            label: output_name.to_string(),
            node_type: "output".to_string(),
            value: 1.0, // Single output receives all attention
        }];

        let edges: Vec<AttentionFlowEdge> = snapshot
            .attention_weights
            .iter()
            .enumerate()
            .map(|(i, &weight)| AttentionFlowEdge {
                source_idx: i,
                source_name: snapshot
                    .input_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("input_{i}")),
                target: output_name.to_string(),
                weight,
                source_phi: snapshot.phi_values.get(i).copied().unwrap_or(0.0),
            })
            .collect();

        Self {
            inputs,
            outputs,
            edges,
            metadata: snapshot.metadata.clone(),
        }
    }

    /// Export to JSON for D3.js or other visualization libraries
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to DOT format for Graphviz
    pub fn to_dot(&self) -> String {
        let mut lines = vec!["digraph AttentionFlow {".to_string()];
        lines.push("  rankdir=LR;".to_string());
        lines.push("  node [shape=box];".to_string());
        lines.push("".to_string());

        // Subgraph for inputs
        lines.push("  subgraph cluster_inputs {".to_string());
        lines.push("    label=\"Inputs (with Phi)\";".to_string());
        lines.push("    style=dashed;".to_string());
        for node in &self.inputs {
            lines.push(format!(
                "    {} [label=\"{}\\nPhi={:.3}\"];",
                node.id, node.label, node.value
            ));
        }
        lines.push("  }".to_string());
        lines.push("".to_string());

        // Subgraph for outputs
        lines.push("  subgraph cluster_outputs {".to_string());
        lines.push("    label=\"Output\";".to_string());
        lines.push("    style=dashed;".to_string());
        for node in &self.outputs {
            lines.push(format!("    {} [label=\"{}\"];", node.id, node.label));
        }
        lines.push("  }".to_string());
        lines.push("".to_string());

        // Edges with weights
        for edge in &self.edges {
            let penwidth = 1.0 + edge.weight * 4.0; // Scale line width
            let label = format!("{:.3}", edge.weight);
            lines.push(format!(
                "  input_{} -> output_0 [label=\"{}\", penwidth={:.1}];",
                edge.source_idx, label, penwidth
            ));
        }

        lines.push("}".to_string());

        lines.join("\n")
    }
}

// ============================================================================
// ATTENTION VISUALIZER
// ============================================================================

/// Main visualizer struct for capturing and rendering attention data.
///
/// Provides a convenient interface for debugging attention in cognitive loops.
#[derive(Debug, Clone)]
pub struct AttentionVisualizer {
    /// History of attention snapshots
    history: AttentionHistory,

    /// Default input names (reused across captures)
    default_names: Option<Vec<String>>,

    /// Whether to auto-capture
    auto_capture: bool,
}

impl Default for AttentionVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionVisualizer {
    /// Create a new visualizer
    pub fn new() -> Self {
        Self {
            history: AttentionHistory::new(),
            default_names: None,
            auto_capture: false,
        }
    }

    /// Create a visualizer with rolling window history
    pub fn with_max_history(max_snapshots: usize) -> Self {
        Self {
            history: AttentionHistory::with_max_size(max_snapshots),
            default_names: None,
            auto_capture: false,
        }
    }

    /// Set default input names
    pub fn with_input_names(mut self, names: Vec<&str>) -> Self {
        self.default_names = Some(names.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Enable auto-capture mode
    pub fn with_auto_capture(mut self, enabled: bool) -> Self {
        self.auto_capture = enabled;
        self
    }

    /// Get a reference to the history
    pub fn history(&self) -> &AttentionHistory {
        &self.history
    }

    /// Get a mutable reference to the history
    pub fn history_mut(&mut self) -> &mut AttentionHistory {
        &mut self.history
    }

    /// Record a snapshot
    pub fn record(&mut self, snapshot: AttentionSnapshot) {
        self.history.record(snapshot);
    }

    /// Capture from a PhiAttentionResult using default names
    pub fn capture(&mut self, result: &PhiAttentionResult, phi_values: &[f64], temperature: f32) {
        let snapshot = if let Some(names) = &self.default_names {
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            AttentionSnapshot::from_result(result, phi_values, name_refs, temperature)
        } else {
            AttentionSnapshot::from_result_indexed(result, phi_values, temperature)
        };
        self.history.record(snapshot);
    }

    /// Capture with custom names
    pub fn capture_with_names(
        &mut self,
        result: &PhiAttentionResult,
        phi_values: &[f64],
        names: Vec<&str>,
        temperature: f32,
    ) {
        let snapshot = AttentionSnapshot::from_result(result, phi_values, names, temperature);
        self.history.record(snapshot);
    }

    /// Clear all recorded data
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Export full history to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        self.history.to_json()
    }

    /// Generate ASCII heatmap of attention history
    pub fn render_heatmap(&self) -> String {
        self.history.to_ascii_heatmap()
    }

    /// Generate flow graph for the latest snapshot
    pub fn latest_flow_graph(&self, output_name: &str) -> Option<AttentionFlowGraph> {
        self.history
            .latest()
            .map(|snapshot| AttentionFlowGraph::from_snapshot(snapshot, output_name))
    }

    /// Get summary statistics
    pub fn summary(&self) -> AttentionSummary {
        AttentionSummary {
            num_snapshots: self.history.len(),
            average_entropy: self.history.average_entropy(),
            top_attended: self.history.top_k_overall(5),
            latest_timestamp: self.history.latest().map(|s| s.timestamp),
        }
    }
}

/// Summary statistics for attention visualization
#[derive(Debug, Clone)]
pub struct AttentionSummary {
    /// Number of snapshots recorded
    pub num_snapshots: usize,

    /// Average entropy across all snapshots
    pub average_entropy: f32,

    /// Top-5 most attended inputs (name, avg weight)
    pub top_attended: Vec<(String, f32)>,

    /// Timestamp of most recent snapshot
    pub latest_timestamp: Option<DateTime<Utc>>,
}

impl std::fmt::Display for AttentionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Attention Summary")?;
        writeln!(f, "================")?;
        writeln!(f, "Snapshots: {}", self.num_snapshots)?;
        writeln!(f, "Average entropy: {:.4}", self.average_entropy)?;
        writeln!(f, "Top attended inputs:")?;
        for (i, (name, weight)) in self.top_attended.iter().enumerate() {
            writeln!(f, "  {}. {} ({:.4})", i + 1, name, weight)?;
        }
        if let Some(ts) = self.latest_timestamp {
            writeln!(f, "Latest capture: {}", ts.format("%Y-%m-%d %H:%M:%S UTC"))?;
        }
        Ok(())
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Compute entropy of a probability distribution
fn compute_entropy(weights: &[f32]) -> f32 {
    -weights
        .iter()
        .filter(|&&w| w > 1e-10)
        .map(|&w| w * w.ln())
        .sum::<f32>()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_creation() {
        let snapshot = AttentionSnapshot::new(
            vec![
                "visual".to_string(),
                "auditory".to_string(),
                "semantic".to_string(),
            ],
            vec![0.8, 0.3, 0.5],
            vec![0.6, 0.1, 0.3],
            1.0,
        );

        assert_eq!(snapshot.input_names.len(), 3);
        assert_eq!(snapshot.phi_values.len(), 3);
        assert_eq!(snapshot.attention_weights.len(), 3);
        assert!((snapshot.attention_weights.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_top_k() {
        let snapshot = AttentionSnapshot::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![0.8, 0.3, 0.5],
            vec![0.6, 0.1, 0.3],
            1.0,
        );

        let top2 = snapshot.top_k_attended(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].1, "a"); // Highest weight
        assert_eq!(top2[1].1, "c"); // Second highest
    }

    #[test]
    fn test_snapshot_argmax() {
        let snapshot = AttentionSnapshot::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![0.8, 0.3, 0.5],
            vec![0.2, 0.5, 0.3],
            1.0,
        );

        assert_eq!(snapshot.argmax(), Some(1)); // "b" has highest weight
    }

    #[test]
    fn test_snapshot_json_roundtrip() {
        let snapshot = AttentionSnapshot::new(
            vec!["visual".to_string(), "auditory".to_string()],
            vec![0.7, 0.4],
            vec![0.6, 0.4],
            0.5,
        );

        let json = snapshot.to_json().expect("JSON serialization failed");
        let restored = AttentionSnapshot::from_json(&json).expect("JSON deserialization failed");

        assert_eq!(snapshot.input_names, restored.input_names);
        assert_eq!(snapshot.phi_values.len(), restored.phi_values.len());
        for (a, b) in snapshot.phi_values.iter().zip(restored.phi_values.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_history_recording() {
        let mut history = AttentionHistory::new();

        for i in 0..5 {
            let snapshot = AttentionSnapshot::new(
                vec!["a".to_string(), "b".to_string()],
                vec![0.5, 0.5],
                vec![0.5 + i as f32 * 0.05, 0.5 - i as f32 * 0.05],
                1.0,
            );
            history.record(snapshot);
        }

        assert_eq!(history.len(), 5);
        assert!(history.latest().is_some());
    }

    #[test]
    fn test_history_max_size() {
        let mut history = AttentionHistory::with_max_size(3);

        for i in 0..10 {
            let snapshot =
                AttentionSnapshot::new(vec![format!("input_{}", i)], vec![0.5], vec![1.0], 1.0);
            history.record(snapshot);
        }

        assert_eq!(history.len(), 3);
        // Should have only the last 3 snapshots
        assert_eq!(history.snapshots()[0].input_names[0], "input_7");
    }

    #[test]
    fn test_history_top_k_overall() {
        let mut history = AttentionHistory::new();

        // Record multiple snapshots where "a" consistently gets higher weight
        for _ in 0..5 {
            let snapshot = AttentionSnapshot::new(
                vec!["a".to_string(), "b".to_string()],
                vec![0.8, 0.3],
                vec![0.7, 0.3],
                1.0,
            );
            history.record(snapshot);
        }

        let top = history.top_k_overall(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "a");
        assert!(top[0].1 > top[1].1);
    }

    #[test]
    fn test_entropy_calculation() {
        // Uniform distribution should have high entropy
        let uniform = vec![0.25f32, 0.25, 0.25, 0.25];
        let uniform_entropy = compute_entropy(&uniform);

        // Peaked distribution should have low entropy
        let peaked = vec![0.97f32, 0.01, 0.01, 0.01];
        let peaked_entropy = compute_entropy(&peaked);

        assert!(uniform_entropy > peaked_entropy);
    }

    #[test]
    fn test_ascii_heatmap_generation() {
        let mut history = AttentionHistory::new();

        for i in 0..5 {
            let snapshot = AttentionSnapshot::new(
                vec!["visual".to_string(), "auditory".to_string()],
                vec![0.5, 0.5],
                vec![0.3 + i as f32 * 0.1, 0.7 - i as f32 * 0.1],
                1.0,
            );
            history.record(snapshot);
        }

        let heatmap = history.to_ascii_heatmap();
        assert!(!heatmap.is_empty());
        assert!(heatmap.contains("visual"));
        assert!(heatmap.contains("auditory"));
    }

    #[test]
    fn test_flow_graph_creation() {
        let snapshot = AttentionSnapshot::new(
            vec!["visual".to_string(), "auditory".to_string()],
            vec![0.7, 0.3],
            vec![0.6, 0.4],
            1.0,
        );

        let graph = AttentionFlowGraph::from_snapshot(&snapshot, "combined");

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn test_flow_graph_dot_export() {
        let snapshot = AttentionSnapshot::new(
            vec!["a".to_string(), "b".to_string()],
            vec![0.8, 0.2],
            vec![0.7, 0.3],
            1.0,
        );

        let graph = AttentionFlowGraph::from_snapshot(&snapshot, "output");
        let dot = graph.to_dot();

        assert!(dot.contains("digraph"));
        assert!(dot.contains("input_0"));
        assert!(dot.contains("output_0"));
    }

    #[test]
    fn test_visualizer_capture() {
        let mut viz = AttentionVisualizer::new().with_input_names(vec!["x", "y", "z"]);

        // Simulate capturing attention results
        let snapshot = AttentionSnapshot::new(
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            vec![0.5, 0.3, 0.7],
            vec![0.3, 0.2, 0.5],
            1.0,
        );

        viz.record(snapshot);

        assert_eq!(viz.history().len(), 1);
    }

    #[test]
    fn test_visualizer_summary() {
        let mut viz = AttentionVisualizer::new();

        for _ in 0..10 {
            let snapshot = AttentionSnapshot::new(
                vec!["a".to_string(), "b".to_string()],
                vec![0.6, 0.4],
                vec![0.55, 0.45],
                1.0,
            );
            viz.record(snapshot);
        }

        let summary = viz.summary();
        assert_eq!(summary.num_snapshots, 10);
        assert!(summary.average_entropy > 0.0);
        assert!(!summary.top_attended.is_empty());
    }

    #[test]
    fn test_snapshot_ascii_bar() {
        let snapshot = AttentionSnapshot::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![0.8, 0.3, 0.5],
            vec![0.6, 0.1, 0.3],
            1.0,
        );

        let bar = snapshot.to_ascii_bar();
        assert_eq!(bar.chars().count(), 3);
    }

    #[test]
    fn test_history_json_roundtrip() {
        let mut history = AttentionHistory::new();

        for i in 0..3 {
            let snapshot = AttentionSnapshot::new(
                vec!["input".to_string()],
                vec![0.5 + i as f64 * 0.1],
                vec![1.0],
                1.0,
            );
            history.record(snapshot);
        }

        let json = history.to_json().expect("JSON serialization failed");
        let restored = AttentionHistory::from_json(&json).expect("JSON deserialization failed");

        assert_eq!(history.len(), restored.len());
    }

    #[test]
    fn test_attention_entropy_method() {
        let snapshot = AttentionSnapshot::new(
            vec!["a".to_string(), "b".to_string()],
            vec![0.5, 0.5],
            vec![0.5, 0.5], // Uniform
            1.0,
        );

        let entropy = snapshot.attention_entropy();
        let expected_max = 2.0f32.ln(); // ln(2) for 2 elements

        // Uniform should be close to max entropy
        assert!((entropy - expected_max).abs() < 0.01);
    }

    #[test]
    fn test_is_focused() {
        let focused = AttentionSnapshot::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.9, 0.05, 0.03, 0.02], // Very focused
            1.0,
        );

        let unfocused = AttentionSnapshot::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25], // Uniform
            1.0,
        );

        assert!(focused.is_focused());
        assert!(!unfocused.is_focused());
    }

    #[test]
    fn test_metadata() {
        let snapshot = AttentionSnapshot::new(vec!["a".to_string()], vec![0.5], vec![1.0], 1.0)
            .with_metadata("context", "test_run")
            .with_metadata("iteration", "42");

        assert_eq!(
            snapshot.metadata.get("context"),
            Some(&"test_run".to_string())
        );
        assert_eq!(snapshot.metadata.get("iteration"), Some(&"42".to_string()));
    }

    #[test]
    fn test_time_series_extraction() {
        let mut history = AttentionHistory::new();

        for _ in 0..5 {
            let snapshot = AttentionSnapshot::new(
                vec!["x".to_string(), "y".to_string()],
                vec![0.5, 0.5],
                vec![0.6, 0.4],
                1.0,
            );
            history.record(snapshot);
        }

        let series = history.to_time_series();
        assert!(series.contains_key("x"));
        assert!(series.contains_key("y"));
        assert_eq!(series["x"].len(), 5);
    }
}
