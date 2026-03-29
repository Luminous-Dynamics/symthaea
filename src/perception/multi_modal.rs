// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Modal Perception: Integrated Sensory Processing
//!
//! Provides multi-modal perception capabilities for integrating
//! information from different sensory modalities.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;

/// Types of sensory modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    /// Visual/image input
    Visual,
    /// Auditory/sound input
    Auditory,
    /// Textual input
    Textual,
    /// Tactile/touch input
    Tactile,
    /// Proprioceptive (body position)
    Proprioceptive,
    /// Temporal/time-based
    Temporal,
    /// Spatial/location-based
    Spatial,
}

/// Configuration for multi-modal perception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Attention mechanism
    pub use_attention: bool,
    /// Temporal window size
    pub temporal_window: usize,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            fusion_strategy: FusionStrategy::Concatenate,
            use_attention: true,
            temporal_window: 10,
        }
    }
}

/// Strategies for fusing multi-modal information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Concatenate embeddings
    Concatenate,
    /// Element-wise sum
    Sum,
    /// Element-wise product
    Product,
    /// Attention-weighted fusion
    Attention,
    /// Hierarchical fusion
    Hierarchical,
}

/// Input to the perception system
#[derive(Debug, Clone)]
pub struct PerceptionInput {
    /// Input identifier
    pub id: String,
    /// Modality of this input
    pub modality: ModalityType,
    /// Raw data (as bytes)
    pub data: Vec<u8>,
    /// Pre-computed embedding (if available)
    pub embedding: Option<ContinuousHV>,
    /// Timestamp
    pub timestamp: u64,
    /// Confidence/quality score
    pub confidence: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl PerceptionInput {
    /// Create a new perception input
    pub fn new(id: impl Into<String>, modality: ModalityType, data: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            modality,
            data,
            embedding: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Create with pre-computed embedding
    pub fn with_embedding(mut self, embedding: ContinuousHV) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Output from the perception system
#[derive(Debug, Clone)]
pub struct PerceptionOutput {
    /// Output identifier
    pub id: String,
    /// Integrated embedding
    pub embedding: ContinuousHV,
    /// Contributing modalities
    pub modalities: Vec<ModalityType>,
    /// Confidence scores per modality
    pub modality_confidences: HashMap<ModalityType, f32>,
    /// Overall confidence
    pub overall_confidence: f32,
    /// Attention weights (if applicable)
    pub attention_weights: Option<HashMap<ModalityType, f32>>,
    /// Timestamp
    pub timestamp: u64,
}

/// Result of multi-modal integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Whether integration succeeded
    pub success: bool,
    /// Number of modalities integrated
    pub modality_count: usize,
    /// Coherence of the integration
    pub coherence: f32,
    /// Integration latency (ms)
    pub latency_ms: f64,
}

/// The multi-modal perception system
#[derive(Debug)]
pub struct MultiModalPerception {
    /// Configuration
    config: MultiModalConfig,
    /// Per-modality encoders (embeddings)
    modality_encoders: HashMap<ModalityType, ContinuousHV>,
    /// Current inputs buffer
    input_buffer: Vec<PerceptionInput>,
    /// Statistics
    stats: PerceptionStats,
}

/// Statistics for the perception system
#[derive(Debug, Clone, Default)]
pub struct PerceptionStats {
    /// Total inputs processed
    pub inputs_processed: u64,
    /// Total integrations performed
    pub integrations: u64,
    /// Average confidence
    pub avg_confidence: f32,
    /// Per-modality counts
    pub modality_counts: HashMap<ModalityType, u64>,
}

impl MultiModalPerception {
    /// Create a new multi-modal perception system
    pub fn new(config: MultiModalConfig) -> Self {
        let dim = config.dimension;

        // Initialize modality-specific encoders
        let mut modality_encoders = HashMap::new();
        for modality in [
            ModalityType::Visual,
            ModalityType::Auditory,
            ModalityType::Textual,
            ModalityType::Tactile,
            ModalityType::Proprioceptive,
            ModalityType::Temporal,
            ModalityType::Spatial,
        ] {
            modality_encoders.insert(
                modality,
                ContinuousHV::random(dim, (modality as u64 + 1) * 42),
            );
        }

        Self {
            config,
            modality_encoders,
            input_buffer: Vec::new(),
            stats: PerceptionStats::default(),
        }
    }

    /// Process a single input
    pub fn process_input(&mut self, input: PerceptionInput) -> ContinuousHV {
        self.stats.inputs_processed += 1;
        *self
            .stats
            .modality_counts
            .entry(input.modality)
            .or_insert(0) += 1;

        // Use pre-computed embedding or encode
        let embedding = if let Some(emb) = &input.embedding {
            emb.clone()
        } else {
            self.encode(&input)
        };

        self.input_buffer.push(input);
        embedding
    }

    /// Encode raw input to embedding
    fn encode(&self, input: &PerceptionInput) -> ContinuousHV {
        let modality_basis = self
            .modality_encoders
            .get(&input.modality)
            .cloned()
            .unwrap_or_else(|| ContinuousHV::random(self.config.dimension, 999));

        // Simple encoding: hash data and combine with modality basis
        let data_hv = self.data_to_hv(&input.data);
        modality_basis.bind(&data_hv)
    }

    /// Convert raw data to hypervector
    fn data_to_hv(&self, data: &[u8]) -> ContinuousHV {
        let mut values = vec![0.0f32; self.config.dimension];

        for (i, &byte) in data.iter().enumerate() {
            let idx = (byte as usize + i) % self.config.dimension;
            values[idx] += 1.0;
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_slice(&values)
    }

    /// Integrate all buffered inputs
    pub fn integrate(&mut self) -> Option<PerceptionOutput> {
        if self.input_buffer.is_empty() {
            return None;
        }

        self.stats.integrations += 1;

        let modalities: Vec<_> = self.input_buffer.iter().map(|i| i.modality).collect();

        let mut modality_confidences = HashMap::new();
        let embeddings: Vec<ContinuousHV> = self
            .input_buffer
            .iter()
            .map(|input| {
                modality_confidences.insert(input.modality, input.confidence);
                input
                    .embedding
                    .clone()
                    .unwrap_or_else(|| self.encode(input))
            })
            .collect();

        // Fuse embeddings
        let fused = self.fuse_embeddings(&embeddings);

        // Calculate overall confidence
        let overall_confidence =
            modality_confidences.values().sum::<f32>() / modality_confidences.len().max(1) as f32;

        // Update stats
        let n = self.stats.integrations as f32;
        self.stats.avg_confidence =
            (self.stats.avg_confidence * (n - 1.0) + overall_confidence) / n;

        let output = PerceptionOutput {
            id: format!("integration_{}", self.stats.integrations),
            embedding: fused,
            modalities,
            modality_confidences,
            overall_confidence,
            attention_weights: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        // Clear buffer
        self.input_buffer.clear();

        Some(output)
    }

    /// Fuse multiple embeddings
    fn fuse_embeddings(&self, embeddings: &[ContinuousHV]) -> ContinuousHV {
        if embeddings.is_empty() {
            return ContinuousHV::random(self.config.dimension, 0);
        }
        if embeddings.len() == 1 {
            return embeddings[0].clone();
        }

        match self.config.fusion_strategy {
            FusionStrategy::Sum | FusionStrategy::Concatenate => {
                ContinuousHV::bundle_owned(embeddings)
            }
            FusionStrategy::Product => {
                let mut result = embeddings[0].clone();
                for emb in &embeddings[1..] {
                    result = result.bind(emb);
                }
                result
            }
            FusionStrategy::Attention | FusionStrategy::Hierarchical => {
                // Simplified attention-based fusion
                ContinuousHV::bundle_owned(embeddings)
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &PerceptionStats {
        &self.stats
    }

    /// Clear the input buffer
    pub fn clear_buffer(&mut self) {
        self.input_buffer.clear();
    }
}

impl Default for MultiModalPerception {
    fn default() -> Self {
        Self::new(MultiModalConfig::default())
    }
}

/// Multi-modal integrator for combining perception streams
#[derive(Debug)]
pub struct MultiModalIntegrator {
    /// Configuration
    config: MultiModalConfig,
    /// Integration history
    history: VecDeque<PerceptionOutput>,
    /// Max history size
    max_history: usize,
}

impl MultiModalIntegrator {
    /// Create a new integrator
    pub fn new(config: MultiModalConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Integrate perception outputs
    pub fn integrate(&mut self, outputs: &[PerceptionOutput]) -> IntegrationResult {
        if outputs.is_empty() {
            return IntegrationResult {
                success: false,
                modality_count: 0,
                coherence: 0.0,
                latency_ms: 0.0,
            };
        }

        let start = std::time::Instant::now();

        // Count unique modalities
        let mut unique_modalities = std::collections::HashSet::new();
        for output in outputs {
            for modality in &output.modalities {
                unique_modalities.insert(*modality);
            }
        }

        // Calculate coherence (average pairwise similarity)
        let coherence = if outputs.len() > 1 {
            let mut total_sim = 0.0;
            let mut count = 0;
            for i in 0..outputs.len() {
                for j in (i + 1)..outputs.len() {
                    let sim = outputs[i].embedding.similarity(&outputs[j].embedding);
                    total_sim += sim;
                    count += 1;
                }
            }
            if count > 0 {
                total_sim / count as f32
            } else {
                1.0
            }
        } else {
            1.0
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Store in history
        for output in outputs {
            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history.push_back(output.clone());
        }

        IntegrationResult {
            success: true,
            modality_count: unique_modalities.len(),
            coherence,
            latency_ms,
        }
    }

    /// Get history
    pub fn history(&self) -> &VecDeque<PerceptionOutput> {
        &self.history
    }
}

impl Default for MultiModalIntegrator {
    fn default() -> Self {
        Self::new(MultiModalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perception_creation() {
        let perception = MultiModalPerception::default();
        assert_eq!(perception.stats.inputs_processed, 0);
    }

    #[test]
    fn test_input_processing() {
        let mut perception = MultiModalPerception::default();

        let input = PerceptionInput::new("test", ModalityType::Visual, vec![1, 2, 3]);
        let embedding = perception.process_input(input);

        assert_eq!(embedding.dim(), 512);
        assert_eq!(perception.stats.inputs_processed, 1);
    }

    #[test]
    fn test_integration() {
        let mut perception = MultiModalPerception::default();

        perception.process_input(PerceptionInput::new(
            "v1",
            ModalityType::Visual,
            vec![1, 2, 3],
        ));
        perception.process_input(PerceptionInput::new(
            "a1",
            ModalityType::Auditory,
            vec![4, 5, 6],
        ));

        let output = perception.integrate();
        assert!(output.is_some());
        assert_eq!(output.unwrap().modalities.len(), 2);
    }
}
