//! Self-model-tier consciousness subsystem manager.
//!
//! Groups the four self-model subsystems (narrative self, predictive self,
//! attention schema, meta-cognition) into a single struct. All are `Option<T>`
//! gated by independent `enable_*` config flags.

use crate::consciousness::attention_schema::AttentionSchema;
use crate::consciousness::narrative_self::{NarrativeSelfConfig, NarrativeSelfModel};
use crate::consciousness::predictive_self::{PredictiveSelfConfig, PredictiveSelfModel};
use crate::wisdom::meta_cognition::MetaCognitiveLayer;

use super::CognitiveLoopConfig;

/// Groups all self-model-gated subsystems.
///
/// Every field is `Option<T>` and is `None` when the corresponding
/// `enable_*` flag is false.
pub(crate) struct SelfModelTierManager {
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
        let config = CognitiveLoopConfig::default();
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
}
