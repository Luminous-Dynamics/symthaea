// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-model-tier consciousness subsystem manager.
//!
//! Groups the four self-model subsystems (narrative self, predictive self,
//! attention schema, meta-cognition) into a single struct. All are `Option<T>`
//! gated by independent `enable_*` config flags.

use super::SelfReflection;
use crate::consciousness::attention_schema::AttentionSchema;
use crate::consciousness::narrative_self::{NarrativeSelfConfig, NarrativeSelfModel};
use crate::consciousness::predictive_self::{PredictiveSelfConfig, PredictiveSelfModel};
use crate::wisdom::meta_cognition::MetaCognitiveLayer;

use super::CognitiveLoopConfig;

/// Groups all self-model-gated subsystems.
///
/// `self_reflection` is always present (required for meta-learning).
/// Other fields are `Option<T>` gated by independent `enable_*` config flags.
pub(crate) struct SelfModelTierManager {
    /// Self-reflection for meta-learning: threshold adaptation, state assessment.
    /// Always present — required for flow state, curiosity drive, and
    /// recommendation generation.
    pub self_reflection: SelfReflection,

    /// Meta-cognitive self-model layer.
    /// Tracks prediction error tendencies and uses self-model accuracy
    /// to modulate learning rate.
    pub meta_cognition: Option<MetaCognitiveLayer>,

    /// Narrative self-model for autobiographical identity.
    /// Maintains a three-level self-model (proto/core/autobio)
    /// and tracks self-Φ (integrated information of the self-model).
    pub narrative_self: Option<NarrativeSelfModel>,

    /// Predictive self-model for action safety evaluation.
    /// Predicts future self-states and evaluates action safety.
    pub predictive_self: Option<PredictiveSelfModel>,

    /// Attention schema (AST) for self-modeling attention state.
    /// Tracks attention focus, shifts, and generates control signals.
    pub attention_schema: Option<AttentionSchema>,
}

impl SelfModelTierManager {
    /// Construct from config. Each subsystem is independently gated.
    pub(crate) fn new(config: &CognitiveLoopConfig) -> Self {
        let meta_cognition = if config.enable_meta_cognition {
            Some(MetaCognitiveLayer::new())
        } else {
            None
        };

        let narrative_self = if config.enable_narrative_self {
            Some(NarrativeSelfModel::new(NarrativeSelfConfig::default()))
        } else {
            None
        };

        let predictive_self = if config.enable_predictive_self {
            Some(PredictiveSelfModel::new(PredictiveSelfConfig::default()))
        } else {
            None
        };

        let attention_schema = if config.enable_attention_schema {
            Some(AttentionSchema::new())
        } else {
            None
        };

        Self {
            self_reflection: SelfReflection::default(),
            meta_cognition,
            narrative_self,
            predictive_self,
            attention_schema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_self_model_tier() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_meta_cognition = false;
        config.enable_narrative_self = false;
        config.enable_predictive_self = false;
        config.enable_attention_schema = false;
        let tier = SelfModelTierManager::new(&config);
        assert!(tier.meta_cognition.is_none());
        assert!(tier.narrative_self.is_none());
        assert!(tier.predictive_self.is_none());
        assert!(tier.attention_schema.is_none());
    }

    #[test]
    fn test_enabled_self_model_tier() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_meta_cognition = true;
        config.enable_narrative_self = true;
        config.enable_predictive_self = true;
        config.enable_attention_schema = true;
        let tier = SelfModelTierManager::new(&config);
        assert!(tier.meta_cognition.is_some());
        assert!(tier.narrative_self.is_some());
        assert!(tier.predictive_self.is_some());
        assert!(tier.attention_schema.is_some());
    }

    #[test]
    fn test_partial_enable_meta_cognition_only() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_meta_cognition = true;
        config.enable_narrative_self = false;
        config.enable_predictive_self = false;
        config.enable_attention_schema = false;
        let tier = SelfModelTierManager::new(&config);
        assert!(tier.meta_cognition.is_some());
        assert!(tier.narrative_self.is_none());
        assert!(tier.predictive_self.is_none());
        assert!(tier.attention_schema.is_none());
    }

    #[test]
    fn test_partial_enable_predictive_self_only() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_meta_cognition = false;
        config.enable_narrative_self = false;
        config.enable_predictive_self = true;
        config.enable_attention_schema = false;
        let tier = SelfModelTierManager::new(&config);
        assert!(tier.meta_cognition.is_none());
        assert!(tier.narrative_self.is_none());
        assert!(tier.predictive_self.is_some());
        assert!(tier.attention_schema.is_none());
    }

    #[test]
    fn test_all_subsystems_enabled() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_meta_cognition = true;
        config.enable_narrative_self = true;
        config.enable_predictive_self = true;
        config.enable_attention_schema = true;
        let tier = SelfModelTierManager::new(&config);
        // All four optional subsystems should be present
        assert!(tier.meta_cognition.is_some());
        assert!(tier.narrative_self.is_some());
        assert!(tier.predictive_self.is_some());
        assert!(tier.attention_schema.is_some());
        // self_reflection is always present
        let _ = &tier.self_reflection;
    }

    #[test]
    fn test_self_reflection_always_present() {
        // Even with all optional subsystems disabled, self_reflection exists
        let config = CognitiveLoopConfig::default();
        let tier = SelfModelTierManager::new(&config);
        // self_reflection is not Option — always constructed
        // Verify learning_effectiveness is finite at initialization
        assert!(
            tier.self_reflection.learning_effectiveness().is_finite(),
            "self_reflection learning_effectiveness should be finite"
        );
    }

    #[test]
    fn test_empty_config_defaults() {
        let config = CognitiveLoopConfig::default();
        // Verify default config has all optional subsystems enabled
        assert!(config.enable_meta_cognition);
        assert!(config.enable_narrative_self);
        assert!(config.enable_predictive_self);
        assert!(config.enable_attention_schema);
        let tier = SelfModelTierManager::new(&config);
        assert!(tier.meta_cognition.is_some());
        assert!(tier.narrative_self.is_some());
        assert!(tier.predictive_self.is_some());
        assert!(tier.attention_schema.is_some());
    }

    #[test]
    fn test_repeated_construction_deterministic() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_meta_cognition = true;
        config.enable_narrative_self = true;
        let tier1 = SelfModelTierManager::new(&config);
        let tier2 = SelfModelTierManager::new(&config);
        // Both should produce the same pattern of Some/None
        assert_eq!(
            tier1.meta_cognition.is_some(),
            tier2.meta_cognition.is_some()
        );
        assert_eq!(
            tier1.narrative_self.is_some(),
            tier2.narrative_self.is_some()
        );
        assert_eq!(
            tier1.predictive_self.is_some(),
            tier2.predictive_self.is_some()
        );
        assert_eq!(
            tier1.attention_schema.is_some(),
            tier2.attention_schema.is_some()
        );
    }

    #[test]
    fn test_self_model_tier_debug_format() {
        let config = CognitiveLoopConfig::default();
        let tier = SelfModelTierManager::new(&config);
        // Verify Debug formatting doesn't panic
        let debug_str = format!("{:?}", tier.self_reflection);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_mixed_enable_narrative_and_attention() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_meta_cognition = false;
        config.enable_narrative_self = true;
        config.enable_predictive_self = false;
        config.enable_attention_schema = true;
        let tier = SelfModelTierManager::new(&config);
        assert!(tier.meta_cognition.is_none());
        assert!(tier.narrative_self.is_some());
        assert!(tier.predictive_self.is_none());
        assert!(tier.attention_schema.is_some());
    }
}
