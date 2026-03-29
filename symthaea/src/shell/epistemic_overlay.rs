// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Overlays - Contextual Knowledge Markup for Shell TUI
//!
//! Floating overlays that show:
//! - Knowledge source and confidence for commands
//! - Epistemic uncertainty indicators
//! - Safety warnings with uncertainty quantification
//! - Theory-grounded recommendations

use serde::{Deserialize, Serialize};

/// Position hint for where to render the overlay
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayPosition {
    /// Near the input cursor
    InputCursor,
    /// Above current completion
    AboveCompletion,
    /// Bottom right metrics area
    MetricsArea,
    /// Center of screen (modal)
    Centered,
}

/// Type of epistemic overlay
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayType {
    /// Knowledge source indicator
    KnowledgeSource,
    /// Uncertainty warning
    UncertaintyWarning,
    /// Safety recommendation
    SafetyHint,
    /// Confidence indicator
    ConfidenceLevel,
    /// Theory-grounded insight
    TheoryInsight,
}

/// Visual style based on epistemic status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicStyle {
    /// High confidence (green)
    HighConfidence,
    /// Medium confidence (yellow)
    MediumConfidence,
    /// Low confidence (red)
    LowConfidence,
    /// Unknown/uncertain (gray)
    Unknown,
    /// Warning (orange)
    Warning,
}

impl EpistemicStyle {
    /// Get the terminal color for this style
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            Self::HighConfidence => (100, 200, 100),
            Self::MediumConfidence => (200, 200, 100),
            Self::LowConfidence => (200, 100, 100),
            Self::Unknown => (150, 150, 150),
            Self::Warning => (255, 165, 0),
        }
    }

    /// Get icon for this style
    pub fn icon(&self) -> &'static str {
        match self {
            Self::HighConfidence => "✓",
            Self::MediumConfidence => "~",
            Self::LowConfidence => "!",
            Self::Unknown => "?",
            Self::Warning => "⚠",
        }
    }
}

/// An epistemic overlay to display in the TUI
#[derive(Debug, Clone)]
pub struct EpistemicOverlay {
    /// Type of overlay
    pub overlay_type: OverlayType,
    /// Preferred position
    pub position: OverlayPosition,
    /// Visual style
    pub style: EpistemicStyle,
    /// Header/title text
    pub header: String,
    /// Body lines
    pub body: Vec<String>,
    /// Footer hint (e.g., "Press ? for details")
    pub footer: Option<String>,
    /// Time-to-live in milliseconds (0 = persistent)
    pub ttl_ms: u64,
    /// Epistemic metrics
    pub metrics: EpistemicMetrics,
}

/// Epistemic metrics for knowledge grounding
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpistemicMetrics {
    /// K-index (consciousness-weighted knowledge confidence)
    pub k_index: f64,
    /// Epistemic uncertainty (what we don't know)
    pub epistemic_uncertainty: f64,
    /// Aleatoric uncertainty (inherent randomness)
    pub aleatoric_uncertainty: f64,
    /// Knowledge source type
    pub source: KnowledgeSource,
    /// Theory grounding (which theories support this)
    pub theory_weights: Vec<(String, f64)>,
}

/// Source of knowledge
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeSource {
    /// Built-in knowledge (high confidence)
    Builtin,
    /// From NixOS documentation
    NixosDocs,
    /// From local configuration
    LocalConfig,
    /// From user history
    UserHistory,
    /// From semantic inference (medium confidence)
    SemanticInference,
    /// From external research
    ExternalResearch,
    /// Unknown source
    #[default]
    Unknown,
}

impl KnowledgeSource {
    pub fn confidence_modifier(&self) -> f64 {
        match self {
            Self::Builtin => 1.0,
            Self::NixosDocs => 0.95,
            Self::LocalConfig => 0.9,
            Self::UserHistory => 0.8,
            Self::SemanticInference => 0.6,
            Self::ExternalResearch => 0.7,
            Self::Unknown => 0.3,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Builtin => "Built-in",
            Self::NixosDocs => "NixOS Docs",
            Self::LocalConfig => "Local Config",
            Self::UserHistory => "History",
            Self::SemanticInference => "Inferred",
            Self::ExternalResearch => "Researched",
            Self::Unknown => "Unknown",
        }
    }
}

/// Engine for generating epistemic overlays
pub struct EpistemicOverlayEngine {
    /// Currently active overlays
    active_overlays: Vec<EpistemicOverlay>,
    /// Overlay generation enabled
    enabled: bool,
    /// Minimum confidence threshold to show uncertainty warnings
    uncertainty_threshold: f64,
}

impl Default for EpistemicOverlayEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicOverlayEngine {
    pub fn new() -> Self {
        Self {
            active_overlays: Vec::new(),
            enabled: true,
            uncertainty_threshold: 0.3,
        }
    }

    /// Generate overlays for a command input
    pub fn generate_for_command(
        &mut self,
        command: &str,
        context: &CommandContext,
    ) -> Vec<EpistemicOverlay> {
        if !self.enabled || command.is_empty() {
            return Vec::new();
        }

        let mut overlays = Vec::new();

        // Generate knowledge source overlay
        if let Some(source_overlay) = self.create_knowledge_source_overlay(command, context) {
            overlays.push(source_overlay);
        }

        // Generate uncertainty warning if needed
        if context.confidence < self.uncertainty_threshold {
            overlays.push(self.create_uncertainty_warning(command, context));
        }

        // Generate safety hints
        if let Some(safety_overlay) = self.create_safety_hint(command, context) {
            overlays.push(safety_overlay);
        }

        self.active_overlays = overlays.clone();
        overlays
    }

    fn create_knowledge_source_overlay(
        &self,
        _command: &str,
        context: &CommandContext,
    ) -> Option<EpistemicOverlay> {
        let style = match context.confidence {
            c if c >= 0.8 => EpistemicStyle::HighConfidence,
            c if c >= 0.5 => EpistemicStyle::MediumConfidence,
            c if c >= 0.3 => EpistemicStyle::LowConfidence,
            _ => EpistemicStyle::Unknown,
        };

        Some(EpistemicOverlay {
            overlay_type: OverlayType::KnowledgeSource,
            position: OverlayPosition::MetricsArea,
            style,
            header: format!("{} Knowledge", style.icon()),
            body: vec![
                format!("Source: {}", context.source.label()),
                format!("Confidence: {:.0}%", context.confidence * 100.0),
            ],
            footer: None,
            ttl_ms: 0,
            metrics: EpistemicMetrics {
                k_index: context.k_index,
                epistemic_uncertainty: context.epistemic_uncertainty,
                aleatoric_uncertainty: context.aleatoric_uncertainty,
                source: context.source.clone(),
                theory_weights: context.theory_weights.clone(),
            },
        })
    }

    fn create_uncertainty_warning(
        &self,
        command: &str,
        context: &CommandContext,
    ) -> EpistemicOverlay {
        let cmd_short = if command.len() > 20 {
            format!("{}...", &command[..20])
        } else {
            command.to_string()
        };

        EpistemicOverlay {
            overlay_type: OverlayType::UncertaintyWarning,
            position: OverlayPosition::InputCursor,
            style: EpistemicStyle::Warning,
            header: "Low Confidence".to_string(),
            body: vec![
                format!("Command: {}", cmd_short),
                format!("K-index: {:.2}", context.k_index),
                "Consider: /research to verify".to_string(),
            ],
            footer: Some("Epistemic uncertainty high".to_string()),
            ttl_ms: 5000,
            metrics: EpistemicMetrics {
                k_index: context.k_index,
                epistemic_uncertainty: context.epistemic_uncertainty,
                aleatoric_uncertainty: context.aleatoric_uncertainty,
                source: context.source.clone(),
                theory_weights: Vec::new(),
            },
        }
    }

    fn create_safety_hint(
        &self,
        command: &str,
        context: &CommandContext,
    ) -> Option<EpistemicOverlay> {
        // Check for commands that might benefit from safety hints
        let cmd_lower = command.to_lowercase();

        if cmd_lower.contains("rm") || cmd_lower.contains("delete") {
            return Some(EpistemicOverlay {
                overlay_type: OverlayType::SafetyHint,
                position: OverlayPosition::AboveCompletion,
                style: EpistemicStyle::Warning,
                header: "Safety Check".to_string(),
                body: vec![
                    "Destructive operation detected".to_string(),
                    format!("Phi threshold: {:.2}", context.phi_required),
                ],
                footer: Some("Use --dry-run first".to_string()),
                ttl_ms: 0,
                metrics: EpistemicMetrics::default(),
            });
        }

        if cmd_lower.contains("nixos-rebuild") {
            return Some(EpistemicOverlay {
                overlay_type: OverlayType::SafetyHint,
                position: OverlayPosition::AboveCompletion,
                style: EpistemicStyle::MediumConfidence,
                header: "System Change".to_string(),
                body: vec![
                    "System rebuild detected".to_string(),
                    "Rollback: nixos-rebuild --rollback".to_string(),
                ],
                footer: None,
                ttl_ms: 0,
                metrics: EpistemicMetrics::default(),
            });
        }

        None
    }

    /// Get currently active overlays
    pub fn active_overlays(&self) -> &[EpistemicOverlay] {
        &self.active_overlays
    }

    /// Clear all overlays
    pub fn clear(&mut self) {
        self.active_overlays.clear();
    }

    /// Enable/disable overlay generation
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear();
        }
    }
}

/// Context for generating epistemic overlays
#[derive(Debug, Clone)]
pub struct CommandContext {
    /// Knowledge source
    pub source: KnowledgeSource,
    /// Overall confidence (0.0-1.0)
    pub confidence: f64,
    /// K-index from epistemic consciousness
    pub k_index: f64,
    /// Epistemic uncertainty
    pub epistemic_uncertainty: f64,
    /// Aleatoric uncertainty
    pub aleatoric_uncertainty: f64,
    /// Theory weights (theory_name, weight)
    pub theory_weights: Vec<(String, f64)>,
    /// Required Phi for execution
    pub phi_required: f64,
    /// Current Phi level
    pub current_phi: f64,
}

impl Default for CommandContext {
    fn default() -> Self {
        Self {
            source: KnowledgeSource::Unknown,
            confidence: 0.5,
            k_index: 0.5,
            epistemic_uncertainty: 0.2,
            aleatoric_uncertainty: 0.1,
            theory_weights: Vec::new(),
            phi_required: 0.5,
            current_phi: 0.7,
        }
    }
}

impl CommandContext {
    /// Create context from completion data
    pub fn from_completion(confidence: f64, source: KnowledgeSource) -> Self {
        Self {
            source,
            confidence,
            k_index: confidence * 0.9, // K-index is slightly lower than raw confidence
            epistemic_uncertainty: 1.0 - confidence,
            ..Default::default()
        }
    }

    /// Create context with full epistemic data
    pub fn with_epistemic(
        source: KnowledgeSource,
        k_index: f64,
        epistemic_uncertainty: f64,
        aleatoric_uncertainty: f64,
    ) -> Self {
        let confidence = k_index * (1.0 - epistemic_uncertainty.min(1.0));
        Self {
            source,
            confidence,
            k_index,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_generation() {
        let mut engine = EpistemicOverlayEngine::new();
        let context = CommandContext::default();

        let overlays = engine.generate_for_command("nix-env -i firefox", &context);
        assert!(!overlays.is_empty());
    }

    #[test]
    fn test_knowledge_source_confidence() {
        assert_eq!(KnowledgeSource::Builtin.confidence_modifier(), 1.0);
        assert!(KnowledgeSource::Unknown.confidence_modifier() < 0.5);
    }

    #[test]
    fn test_uncertainty_warning() {
        let mut engine = EpistemicOverlayEngine::new();
        let context = CommandContext {
            confidence: 0.2, // Low confidence
            ..Default::default()
        };

        let overlays = engine.generate_for_command("some-unknown-command", &context);

        // Should include an uncertainty warning
        let has_warning = overlays
            .iter()
            .any(|o| o.overlay_type == OverlayType::UncertaintyWarning);
        assert!(has_warning);
    }

    #[test]
    fn test_safety_hint() {
        let mut engine = EpistemicOverlayEngine::new();
        let context = CommandContext::default();

        let overlays = engine.generate_for_command("rm -rf /some/path", &context);

        // Should include a safety hint
        let has_safety = overlays
            .iter()
            .any(|o| o.overlay_type == OverlayType::SafetyHint);
        assert!(has_safety);
    }
}
