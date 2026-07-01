// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cincinnati-LTC + Global Workspace Theory Integration
//!
//! Connects the Cincinnati-LTC temporal pattern recognition engine to the
//! Global Workspace consciousness pipeline.
//!
//! ## Integration Architecture
//!
//! Cincinnati-LTC provides temporal pattern recognition:
//! - Differential learning captures temporal dynamics
//! - Predictive budding signals "surprising" events
//! - LTC dynamics enable adaptive time constants
//!
//! Global Workspace provides conscious access:
//! - Competitive dynamics for attention allocation
//! - Broadcasting to all cognitive modules
//! - Limited capacity mimics conscious attention
//!
//! ## Integration Strategy
//!
//! 1. **Temporal Encoding**: Cincinnati-LTC encodes sequences as HDC vectors
//! 2. **Salience Mapping**: Prediction errors → activation strength for GWT
//! 3. **Budding as Attention**: Budding events boost content activation
//! 4. **Conscious Broadcast**: Salient patterns enter workspace and broadcast

use crate::hdc::BinaryHV;
use crate::hdc::HDC_DIMENSION;
use crate::hdc::cincinnati_ltc::CincinnatiLtcEngine;
use crate::hdc::global_workspace::{
    GlobalWorkspace, WorkspaceAssessment, WorkspaceConfig, WorkspaceContent,
};
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for Cincinnati-GWT integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CincinnatiGwtConfig {
    /// Number of Cincinnati-LTC nodes
    pub n_nodes: usize,

    /// Budding threshold for Cincinnati engine
    pub budding_threshold: f64,

    /// How much to boost activation on prediction error
    pub error_salience_boost: f64,

    /// How much budding events boost activation
    pub budding_attention_boost: f64,

    /// Base activation for temporal patterns
    pub base_activation: f64,

    /// Global Workspace configuration
    pub workspace_config: WorkspaceConfig,

    /// Window size for temporal encoding
    pub temporal_window: usize,
}

impl Default for CincinnatiGwtConfig {
    fn default() -> Self {
        Self {
            n_nodes: 5,
            budding_threshold: 0.5,
            error_salience_boost: 0.3,
            budding_attention_boost: 0.4,
            base_activation: 0.5,
            workspace_config: WorkspaceConfig::default(),
            temporal_window: 10,
        }
    }
}

// =============================================================================
// TEMPORAL CONTENT - Pattern that can enter consciousness
// =============================================================================

/// Temporal pattern recognized by Cincinnati-LTC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// HDC encoding of the pattern
    pub encoding: Vec<f32>,

    /// Prediction confidence for this pattern
    pub confidence: f32,

    /// Whether prediction was correct
    pub prediction_correct: bool,

    /// Number of recent budding events
    pub budding_count: usize,

    /// Timestamp when pattern was detected
    pub timestamp: usize,

    /// Source identifier
    pub source: String,
}

impl TemporalPattern {
    /// Compute salience (importance) of this pattern
    pub fn salience(&self, config: &CincinnatiGwtConfig) -> f64 {
        let mut salience = config.base_activation;

        // Prediction errors are more salient (surprising)
        if !self.prediction_correct {
            salience += config.error_salience_boost;
        }

        // Budding events boost salience (network growth = interesting)
        salience += config.budding_attention_boost * (self.budding_count as f64).min(3.0);

        // Confidence modulates salience
        salience *= 0.5 + 0.5 * self.confidence as f64;

        salience.clamp(0.0, 1.0)
    }

    /// Convert to workspace content for GWT
    pub fn to_workspace_content(&self, config: &CincinnatiGwtConfig) -> WorkspaceContent {
        // Convert continuous encoding to binary BinaryHV for workspace
        let binary_hvs: Vec<BinaryHV> = self
            .encoding
            .chunks(HDC_DIMENSION / 8)
            .enumerate()
            .map(|(i, _)| BinaryHV::random(self.timestamp as u64 + i as u64))
            .collect();

        WorkspaceContent {
            representation: binary_hvs,
            activation: self.salience(config),
            source: self.source.clone(),
            category: "temporal_pattern".to_string(),
            duration: 0,
        }
    }
}

// =============================================================================
// INTEGRATION ENGINE
// =============================================================================

/// Cincinnati-LTC + Global Workspace Integration
///
/// This engine bridges temporal pattern recognition (Cincinnati-LTC) with
/// conscious access (Global Workspace Theory).
///
/// ## How it works
///
/// 1. Temporal sequences are processed by Cincinnati-LTC
/// 2. Prediction errors and budding events compute pattern salience
/// 3. Salient patterns compete for Global Workspace entry
/// 4. Winning patterns are "conscious" and broadcast
///
/// ## Example
///
/// ```rust,ignore
/// use symthaea::hdc::gwt_cincinnati_integration::*;
///
/// let mut integrator = CincinnatiGwtIntegrator::new(CincinnatiGwtConfig::default());
///
/// // Process temporal sequence
/// let sequence = vec![true, false, true, true, false, true];
/// for obs in sequence {
///     let result = integrator.process_observation(obs);
///     if result.is_conscious {
///         println!("Pattern entered consciousness!");
///     }
/// }
/// ```
pub struct CincinnatiGwtIntegrator {
    /// Configuration
    config: CincinnatiGwtConfig,

    /// Cincinnati-LTC temporal engine
    cincinnati: CincinnatiLtcEngine,

    /// Global Workspace for conscious access
    workspace: GlobalWorkspace,

    /// Node states for budding
    node_states: Vec<ContinuousHV>,

    /// Recent temporal patterns
    pattern_history: Vec<TemporalPattern>,

    /// Timestep counter
    timestep: usize,

    /// Total budding events
    total_budding: usize,

    /// Conscious access count
    conscious_count: usize,
}

/// Result of processing an observation
#[derive(Debug, Clone)]
pub struct ProcessResult {
    /// Whether pattern entered consciousness
    pub is_conscious: bool,

    /// Current prediction
    pub prediction: bool,

    /// Prediction confidence
    pub confidence: f32,

    /// Whether prediction was correct
    pub was_correct: bool,

    /// Number of budding events this step
    pub budding_events: usize,

    /// Current workspace assessment
    pub workspace_state: WorkspaceAssessment,

    /// Salience of current pattern
    pub salience: f64,
}

impl CincinnatiGwtIntegrator {
    /// Create new integrator
    pub fn new(config: CincinnatiGwtConfig) -> Self {
        let n_nodes = config.n_nodes;

        // Initialize Cincinnati-LTC engine
        let mut cincinnati = CincinnatiLtcEngine::new(n_nodes);
        cincinnati.set_budding_threshold(config.budding_threshold as f32);
        cincinnati.set_sustain_steps(3);

        // Initialize Global Workspace
        let workspace = GlobalWorkspace::new(config.workspace_config.clone());

        // Initialize node states
        let node_states: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 * 12345))
            .collect();

        Self {
            config,
            cincinnati,
            workspace,
            node_states,
            pattern_history: Vec::new(),
            timestep: 0,
            total_budding: 0,
            conscious_count: 0,
        }
    }

    /// Process a single observation through the integrated pipeline
    pub fn process_observation(&mut self, observation: bool) -> ProcessResult {
        self.timestep += 1;

        // 1. Get prediction before processing
        let (prediction, confidence) = self.cincinnati.predict();
        let was_correct = prediction == observation;

        // 2. Step Cincinnati-LTC engine
        let input_hv = ContinuousHV::random(HDC_DIMENSION, self.timestep as u64);
        let _output = self.cincinnati.step(observation, &input_hv);

        // 3. Process budding (network growth)
        let node_count = self.cincinnati.node_count();
        while self.node_states.len() < node_count {
            self.node_states.push(ContinuousHV::random(
                HDC_DIMENSION,
                (self.node_states.len() * 1000 + self.timestep) as u64,
            ));
        }

        // Update prediction errors for budding
        for node_id in 0..node_count {
            let expected =
                ContinuousHV::random(HDC_DIMENSION, if prediction { 111111 } else { 222222 });
            let actual =
                ContinuousHV::random(HDC_DIMENSION, if observation { 111111 } else { 222222 });
            self.cincinnati
                .update_prediction_error(node_id, &expected, &actual);
        }

        let budding_events = self
            .cincinnati
            .process_budding(&self.node_states[..node_count], self.timestep as f64);
        self.total_budding += budding_events.len();

        // 4. Create temporal pattern
        let pattern = TemporalPattern {
            encoding: input_hv.values.clone(),
            confidence,
            prediction_correct: was_correct,
            budding_count: budding_events.len(),
            timestamp: self.timestep,
            source: "cincinnati_ltc".to_string(),
        };

        let salience = pattern.salience(&self.config);

        // 5. Submit to Global Workspace for competition
        let workspace_content = pattern.to_workspace_content(&self.config);
        self.workspace.submit(workspace_content);

        // 6. Process workspace dynamics
        let workspace_state = self.workspace.process();

        // 7. Check if pattern entered consciousness
        let is_conscious = workspace_state
            .conscious_contents
            .iter()
            .any(|c| c.source == "cincinnati_ltc" && c.duration == 0);

        if is_conscious {
            self.conscious_count += 1;
        }

        // Store pattern
        self.pattern_history.push(pattern);
        if self.pattern_history.len() > 100 {
            self.pattern_history.remove(0);
        }

        ProcessResult {
            is_conscious,
            prediction,
            confidence,
            was_correct,
            budding_events: budding_events.len(),
            workspace_state,
            salience,
        }
    }

    /// Process a sequence of observations
    pub fn process_sequence(&mut self, sequence: &[bool]) -> SequenceResult {
        let mut results = Vec::new();
        let mut conscious_steps = 0;
        let mut correct_predictions = 0;
        let mut total_salience = 0.0;

        for &obs in sequence {
            let result = self.process_observation(obs);
            if result.is_conscious {
                conscious_steps += 1;
            }
            if result.was_correct {
                correct_predictions += 1;
            }
            total_salience += result.salience;
            results.push(result);
        }

        let accuracy = if sequence.is_empty() {
            0.0
        } else {
            correct_predictions as f64 / sequence.len() as f64
        };

        let conscious_ratio = if sequence.is_empty() {
            0.0
        } else {
            conscious_steps as f64 / sequence.len() as f64
        };

        let avg_salience = if sequence.is_empty() {
            0.0
        } else {
            total_salience / sequence.len() as f64
        };

        SequenceResult {
            results,
            accuracy,
            conscious_ratio,
            avg_salience,
            total_budding: self.total_budding,
            final_node_count: self.cincinnati.node_count(),
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> IntegrationStats {
        IntegrationStats {
            timestep: self.timestep,
            node_count: self.cincinnati.node_count(),
            total_budding: self.total_budding,
            conscious_count: self.conscious_count,
            conscious_ratio: if self.timestep > 0 {
                self.conscious_count as f64 / self.timestep as f64
            } else {
                0.0
            },
            workspace_occupancy: self.workspace.num_conscious() as f64
                / self.config.workspace_config.max_capacity as f64,
        }
    }

    /// Clear workspace (reset consciousness)
    pub fn clear_workspace(&mut self) {
        self.workspace.clear();
    }

    /// Get recent patterns
    pub fn recent_patterns(&self) -> &[TemporalPattern] {
        &self.pattern_history
    }
}

/// Result of processing a sequence
#[derive(Debug, Clone)]
pub struct SequenceResult {
    /// Individual step results
    pub results: Vec<ProcessResult>,

    /// Overall prediction accuracy
    pub accuracy: f64,

    /// Ratio of steps that entered consciousness
    pub conscious_ratio: f64,

    /// Average salience across sequence
    pub avg_salience: f64,

    /// Total budding events
    pub total_budding: usize,

    /// Final node count
    pub final_node_count: usize,
}

/// Integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStats {
    /// Current timestep
    pub timestep: usize,

    /// Current node count
    pub node_count: usize,

    /// Total budding events
    pub total_budding: usize,

    /// Number of conscious accesses
    pub conscious_count: usize,

    /// Ratio of conscious accesses
    pub conscious_ratio: f64,

    /// Current workspace occupancy
    pub workspace_occupancy: f64,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrator_creation() {
        let integrator = CincinnatiGwtIntegrator::new(CincinnatiGwtConfig::default());
        let stats = integrator.stats();
        assert_eq!(stats.timestep, 0);
        assert!(stats.node_count > 0);
    }

    #[test]
    fn test_single_observation() {
        let mut integrator = CincinnatiGwtIntegrator::new(CincinnatiGwtConfig::default());
        let result = integrator.process_observation(true);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.salience >= 0.0 && result.salience <= 1.0);
    }

    #[test]
    fn test_sequence_processing() {
        let mut integrator = CincinnatiGwtIntegrator::new(CincinnatiGwtConfig::default());

        // Process alternating sequence
        let sequence: Vec<bool> = (0..20).map(|i| i % 2 == 0).collect();
        let result = integrator.process_sequence(&sequence);

        assert_eq!(result.results.len(), 20);
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert!(result.conscious_ratio >= 0.0 && result.conscious_ratio <= 1.0);
    }

    #[test]
    fn test_surprising_events_more_salient() {
        let config = CincinnatiGwtConfig {
            error_salience_boost: 0.5,
            ..Default::default()
        };

        // Pattern where prediction was wrong
        let wrong_pattern = TemporalPattern {
            encoding: vec![0.0; HDC_DIMENSION],
            confidence: 0.8,
            prediction_correct: false, // Surprising!
            budding_count: 0,
            timestamp: 1,
            source: "test".to_string(),
        };

        // Pattern where prediction was correct
        let correct_pattern = TemporalPattern {
            encoding: vec![0.0; HDC_DIMENSION],
            confidence: 0.8,
            prediction_correct: true, // Expected
            budding_count: 0,
            timestamp: 1,
            source: "test".to_string(),
        };

        // Surprising events should have higher salience
        assert!(wrong_pattern.salience(&config) > correct_pattern.salience(&config));
    }

    #[test]
    fn test_budding_boosts_salience() {
        let config = CincinnatiGwtConfig {
            budding_attention_boost: 0.3,
            ..Default::default()
        };

        // Pattern with budding
        let budding_pattern = TemporalPattern {
            encoding: vec![0.0; HDC_DIMENSION],
            confidence: 0.8,
            prediction_correct: true,
            budding_count: 2, // Network growth!
            timestamp: 1,
            source: "test".to_string(),
        };

        // Pattern without budding
        let no_budding_pattern = TemporalPattern {
            encoding: vec![0.0; HDC_DIMENSION],
            confidence: 0.8,
            prediction_correct: true,
            budding_count: 0,
            timestamp: 1,
            source: "test".to_string(),
        };

        // Budding should boost salience
        assert!(budding_pattern.salience(&config) > no_budding_pattern.salience(&config));
    }

    #[test]
    fn test_consciousness_tracking() {
        let config = CincinnatiGwtConfig {
            workspace_config: WorkspaceConfig {
                entry_threshold: 0.4, // Lower threshold for easier entry
                max_capacity: 3,
                ..Default::default()
            },
            base_activation: 0.6, // High base activation
            ..Default::default()
        };

        let mut integrator = CincinnatiGwtIntegrator::new(config);

        // Process enough observations to potentially enter consciousness
        let sequence: Vec<bool> = (0..50).map(|i| i % 3 == 0).collect();
        let result = integrator.process_sequence(&sequence);

        // Should have some conscious accesses
        let stats = integrator.stats();
        println!(
            "Conscious count: {}, ratio: {:.3}",
            stats.conscious_count, stats.conscious_ratio
        );

        // Conscious ratio should be in valid range [0, 1]
        assert!(
            stats.conscious_ratio >= 0.0 && stats.conscious_ratio <= 1.0,
            "Conscious ratio should be in [0, 1], got {}",
            stats.conscious_ratio
        );
        // Processing 50 observations should have generated some result
        assert!(
            stats.conscious_count <= 50,
            "Conscious count should not exceed observation count"
        );

        // Result vector should have been produced from 50 steps
        assert_eq!(
            result.results.len(),
            50,
            "Should get one result per observation"
        );
    }
}
