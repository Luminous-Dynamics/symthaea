// Surgical compatibility stubs for workspace tracking
pub mod brain {
    pub mod prefrontal {
        #[derive(Debug, Clone, Default)]
        pub struct AttentionBid;
        impl AttentionBid {
            pub fn new<A, B>(_name: A, _payload: B) -> Self {
                Self
            }
            pub fn with_salience(self, _s: f32) -> Self {
                self
            }
            pub fn with_urgency(self, _u: f32) -> Self {
                self
            }
            pub fn with_tags(self, _t: Vec<String>) -> Self {
                self
            }
            pub fn with_hdc_semantic<T>(self, _v: Option<std::sync::Arc<T>>) -> Self {
                self
            }
        }
    }
}
pub mod embeddings {
    #[derive(Debug, Clone, Default)]
    pub struct Qwen3Config;
    #[derive(Debug, Clone, Default)]
    pub struct Qwen3EmbedderResult {
        pub embedding: Vec<f32>,
    }
    #[derive(Debug, Clone, Default)]
    pub struct Qwen3Embedder;
    impl Qwen3Embedder {
        pub fn new(_c: Qwen3Config) -> anyhow::Result<Self> {
            Ok(Self)
        }
        pub fn embed(&mut self, _t: &str) -> anyhow::Result<Qwen3EmbedderResult> {
            Ok(Qwen3EmbedderResult::default())
        }
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// # Perception-Consciousness Bridge
//
// Connects the perception system to the consciousness attention system.
// Converts MultiModalPerception into AttentionBids for the Global Workspace.
//
// ## Architecture
//
// ```text
// Perception (SigLIP/Qwen3)
//         │
//         ▼
// ┌─────────────────┐
// │ Perception      │
// │ Bridge          │ ← Converts to AttentionBid
// └────────┬────────┘
//          │
//          ▼
// ┌─────────────────┐
// │ Prefrontal      │
// │ Cortex          │ ← Global Workspace
// └────────┬────────┘
//          │
//          ▼
// ┌─────────────────┐
// │ Consciousness   │
// │ Integration     │ ← Φ measurement
// └─────────────────┘
// ```

use anyhow::Result;
use std::sync::Arc;

use crate::brain::prefrontal::AttentionBid;
use crate::embeddings::{Qwen3Config, Qwen3Embedder};
use crate::perception::{ModalityType, MultiModalIntegrator, MultiModalPerception, SemanticVision};

/// Configuration for the perception-consciousness bridge
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Minimum confidence to create attention bid
    pub min_confidence: f32,

    /// Base salience for perceptual inputs
    pub base_salience: f32,

    /// Base urgency for perceptual inputs
    pub base_urgency: f32,

    /// Whether to include HDC vectors in bids
    pub include_hdc: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            base_salience: 0.6,
            base_urgency: 0.4,
            include_hdc: true,
        }
    }
}

/// Bridge between perception and consciousness systems
pub struct PerceptionBridge {
    /// Configuration
    config: BridgeConfig,

    /// Multi-modal integrator for HDC projection
    integrator: MultiModalIntegrator,

    /// Text embedder (Qwen3)
    text_embedder: Option<Qwen3Embedder>,

    /// Vision system (SigLIP)
    vision: Option<SemanticVision>,

    /// Statistics
    stats: BridgeStats,
}

/// Statistics for the bridge
#[derive(Debug, Default, Clone)]
pub struct BridgeStats {
    /// Total perceptions processed
    pub perceptions_processed: u64,

    /// Total attention bids created
    pub bids_created: u64,

    /// Perceptions filtered (below confidence threshold)
    pub perceptions_filtered: u64,
}

impl PerceptionBridge {
    /// Create a new perception bridge
    pub fn new(config: BridgeConfig) -> Self {
        Self {
            config,
            integrator: MultiModalIntegrator::new(),
            text_embedder: None,
            vision: None,
            stats: BridgeStats::default(),
        }
    }

    /// Create with text embedder
    pub fn with_text_embedder(mut self, embedder: Qwen3Embedder) -> Self {
        self.text_embedder = Some(embedder);
        self
    }

    /// Create with vision system
    pub fn with_vision(mut self, vision: SemanticVision) -> Self {
        self.vision = Some(vision);
        self
    }

    /// Initialize the bridge with default models
    pub fn initialize(&mut self) -> Result<()> {
        // Initialize text embedder if not set
        if self.text_embedder.is_none() {
            let embedder = Qwen3Embedder::new(Qwen3Config::default())?;
            self.text_embedder = Some(embedder);
        }

        // Initialize vision if not set
        if self.vision.is_none() {
            let mut vision = SemanticVision::new(1000);
            vision.initialize()?;
            self.vision = Some(vision);
        }

        Ok(())
    }

    /// Convert a multi-modal perception into attention bids
    pub fn perception_to_bids(&mut self, perception: &MultiModalPerception) -> Vec<AttentionBid> {
        self.stats.perceptions_processed += 1;

        // Filter low-confidence perceptions
        if perception.confidence < self.config.min_confidence {
            self.stats.perceptions_filtered += 1;
            return Vec::new();
        }

        let mut bids = Vec::new();

        // Create a bid for the unified perception
        let unified_bid = self.create_unified_bid(perception);
        bids.push(unified_bid);

        // Optionally create bids for individual modalities
        for contribution in &perception.modalities {
            if contribution.confidence >= self.config.min_confidence {
                if let Some(bid) = self.create_modality_bid(contribution) {
                    bids.push(bid);
                }
            }
        }

        self.stats.bids_created += bids.len() as u64;
        bids
    }

    /// Create a unified attention bid from multi-modal perception
    fn create_unified_bid(&self, perception: &MultiModalPerception) -> AttentionBid {
        let modality_names: Vec<&str> = perception
            .modalities
            .iter()
            .map(|m| m.modality.name())
            .collect();

        let content = format!(
            "Unified perception: {} modalities with {:.1}% confidence",
            modality_names.join("+"),
            perception.confidence * 100.0
        );

        let mut bid = AttentionBid::new("Perception", content)
            .with_salience(self.config.base_salience * perception.confidence)
            .with_urgency(self.config.base_urgency)
            .with_tags(vec!["perception".to_string(), "multi-modal".to_string()]);

        // Add HDC vector if configured
        if self.config.include_hdc {
            let hdc_vector: Vec<i8> = self.hdc_from_perception(perception);
            bid = bid.with_hdc_semantic(Some(Arc::new(hdc_vector)));
        }

        bid
    }

    /// Create an attention bid for a single modality
    fn create_modality_bid(
        &self,
        contribution: &crate::multi_modal::ModalityContribution,
    ) -> Option<AttentionBid> {
        let content = format!(
            "{} perception: {:.1}% confidence",
            contribution.modality.name(),
            contribution.confidence * 100.0
        );

        let urgency = match contribution.modality {
            ModalityType::Vision => 0.5, // Visual input is usually important
            ModalityType::Voice => 0.7,  // Voice input needs quick response
            ModalityType::Code => 0.3,   // Code analysis can wait
            ModalityType::Ocr => 0.4,
            _ => 0.5, // OCR is informational
        };

        let bid = AttentionBid::new("Perception", content)
            .with_salience(contribution.confidence * 0.8)
            .with_urgency(urgency)
            .with_tags(vec![
                "perception".to_string(),
                contribution.modality.name().to_lowercase(),
            ]);

        Some(bid)
    }

    /// Convert perception to HDC vector for semantic matching
    /// Returns Vec<i8> compatible with SharedHdcVector (Arc<Vec<i8>>)
    fn hdc_from_perception(&self, perception: &MultiModalPerception) -> Vec<i8> {
        // Convert the boolean HdcVector to Vec<i8> for consciousness integration
        let dim = perception.unified_concept.dim();
        let mut values = Vec::with_capacity(dim);

        for bit in &perception.unified_concept.bits {
            values.push(if *bit { 1i8 } else { -1i8 });
        }

        values
    }

    /// Process text input and create attention bid
    pub fn process_text(&mut self, text: &str) -> Result<AttentionBid> {
        self.stats.perceptions_processed += 1;

        let embedding = if let Some(ref mut embedder) = self.text_embedder {
            embedder.embed(text)?.embedding
        } else {
            // Use stub embedding
            let mut embedder = Qwen3Embedder::new(Qwen3Config::default())?;
            embedder.embed(text)?.embedding
        };

        // Project to HDC
        let hdc_vector = self.integrator.project_text_embedding(&embedding)?;

        let bid = AttentionBid::new(
            "TextPerception",
            format!("Text: {}", &text[..text.len().min(50)]),
        )
        .with_salience(0.7)
        .with_urgency(0.5)
        .with_tags(vec!["perception".to_string(), "text".to_string()]);

        if self.config.include_hdc {
            let hdc_i8 = self.boolean_to_i8_vec(&hdc_vector);
            Ok(bid.with_hdc_semantic(Some(Arc::new(hdc_i8))))
        } else {
            Ok(bid)
        }
    }

    /// Process image input and create attention bid
    pub fn process_image(&mut self, image: &image::DynamicImage) -> Result<AttentionBid> {
        self.stats.perceptions_processed += 1;

        let embedding = if let Some(ref mut vision) = self.vision {
            vision.embed_image(image)?
        } else {
            // Create temporary vision system
            let mut vision = SemanticVision::new(100);
            vision.initialize()?;
            vision.embed_image(image)?
        };

        // Project to HDC
        let hdc_vector = self.integrator.project_image_embedding(&embedding)?;

        let (width, height) = image.dimensions();
        let bid = AttentionBid::new("VisionPerception", format!("Image: {}x{}", width, height))
            .with_salience(0.6)
            .with_urgency(0.4)
            .with_tags(vec!["perception".to_string(), "vision".to_string()]);

        if self.config.include_hdc {
            let hdc_i8 = self.boolean_to_i8_vec(&hdc_vector);
            Ok(bid.with_hdc_semantic(Some(Arc::new(hdc_i8))))
        } else {
            Ok(bid)
        }
    }

    /// Convert boolean HdcVector to Vec<i8> for SharedHdcVector compatibility
    fn boolean_to_i8_vec(&self, hdc: &crate::multi_modal::HdcVector) -> Vec<i8> {
        hdc.bits
            .iter()
            .map(|&bit| if bit { 1i8 } else { -1i8 })
            .collect()
    }

    /// Get bridge statistics
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }
}

impl Default for PerceptionBridge {
    fn default() -> Self {
        Self::new(BridgeConfig::default())
    }
}

use image::GenericImageView;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = PerceptionBridge::new(BridgeConfig::default());
        assert_eq!(bridge.stats.perceptions_processed, 0);
    }

    #[test]
    fn test_bridge_config() {
        let config = BridgeConfig {
            min_confidence: 0.5,
            base_salience: 0.8,
            base_urgency: 0.6,
            include_hdc: false,
        };
        let bridge = PerceptionBridge::new(config.clone());
        assert_eq!(bridge.config.min_confidence, 0.5);
    }

    #[test]
    fn test_text_processing() {
        let mut bridge = PerceptionBridge::default();
        let bid = bridge.process_text("Hello world").unwrap();
        assert_eq!(bid.source, "TextPerception");
        assert!(bid.content.contains("Hello"));
    }
}
