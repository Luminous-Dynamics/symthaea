// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Common Types for Recursive Improvement
//!
//! Shared types and utilities used across the recursive improvement modules.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Get current instant (for timing)
pub fn instant_now() -> Instant {
    Instant::now()
}

/// Calculate trend from a series of values
/// Returns a value between -1.0 (decreasing) and 1.0 (increasing)
pub fn calculate_trend(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f64;
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..values.len()).map(|i| (i * i) as f64).sum();

    let denominator = n * sum_x2 - sum_x * sum_x;
    if denominator.abs() < f64::EPSILON {
        return 0.0;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;

    // Normalize to [-1, 1] based on relative change
    let mean = sum_y / n;
    if mean.abs() < f64::EPSILON {
        return slope.signum();
    }

    (slope / mean.abs()).clamp(-1.0, 1.0)
}

/// Semantic input for consciousness processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticInput {
    /// Raw text input
    pub text: String,

    /// Embedding vector (optional, may be computed lazily)
    pub embedding: Option<Vec<f32>>,

    /// Input type/modality
    pub modality: InputModality,

    /// Timestamp of input
    pub timestamp: u64,

    /// Context/source identifier
    pub context: Option<String>,
}

impl SemanticInput {
    /// Create a new text input
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            text: content.into(),
            embedding: None,
            modality: InputModality::Text,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            context: None,
        }
    }

    /// Create with embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Create with context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Input modality types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputModality {
    /// Text input
    Text,
    /// Voice/audio input
    Audio,
    /// Visual/image input
    Visual,
    /// Proprioceptive/internal state
    Internal,
    /// Multi-modal combined input
    MultiModal,
}

/// Action context for consciousness decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionContext {
    /// Current goal or objective
    pub goal: Option<String>,

    /// Available actions
    pub available_actions: Vec<String>,

    /// Constraints on actions
    pub constraints: Vec<String>,

    /// Urgency level (0.0 = relaxed, 1.0 = urgent)
    pub urgency: f32,

    /// Risk tolerance (0.0 = risk-averse, 1.0 = risk-seeking)
    pub risk_tolerance: f32,
}

impl Default for ActionContext {
    fn default() -> Self {
        Self {
            goal: None,
            available_actions: Vec::new(),
            constraints: Vec::new(),
            urgency: 0.5,
            risk_tolerance: 0.5,
        }
    }
}

/// Time window for temporal analysis
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start of window
    pub start: Duration,

    /// End of window
    pub end: Duration,
}

impl TimeWindow {
    /// Create a new time window
    pub fn new(start: Duration, end: Duration) -> Self {
        Self { start, end }
    }

    /// Duration of the window
    pub fn duration(&self) -> Duration {
        self.end.saturating_sub(self.start)
    }

    /// Check if a timestamp falls within this window
    pub fn contains(&self, time: Duration) -> bool {
        time >= self.start && time <= self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_trend_increasing() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = calculate_trend(&values);
        assert!(
            trend > 0.0,
            "Trend should be positive for increasing values"
        );
    }

    #[test]
    fn test_calculate_trend_decreasing() {
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = calculate_trend(&values);
        assert!(
            trend < 0.0,
            "Trend should be negative for decreasing values"
        );
    }

    #[test]
    fn test_calculate_trend_flat() {
        let values = vec![3.0, 3.0, 3.0, 3.0];
        let trend = calculate_trend(&values);
        assert!(
            trend.abs() < 0.01,
            "Trend should be near zero for flat values"
        );
    }

    #[test]
    fn test_semantic_input_creation() {
        let input = SemanticInput::text("Hello, world!");
        assert_eq!(input.text, "Hello, world!");
        assert_eq!(input.modality, InputModality::Text);
    }
}
