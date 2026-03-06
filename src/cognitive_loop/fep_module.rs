//! # FEP Module — Consolidated Free Energy Principle / Active Inference State
//!
//! Consolidates 10 previously scattered FEP fields from CognitiveLoopService
//! into a single coherent module.

use crate::consciousness::fep_active_inference::{ActiveInferenceAgent, EnhancedFEPBridge};
use crate::exploration::SurpriseExplorationBridge;

use super::learning::ClosedLearningLoop;
use super::memory_bridge::EpisodicMemoryBridge;
use super::goal_world::{GoalSystemBridge, WorldModelBridge};
use super::routing::ActiveInferenceBridge;

/// Consolidated Free Energy Principle module.
///
/// Groups all FEP/active-inference state that was previously scattered across
/// 10 separate fields in CognitiveLoopService.
pub struct FepModule {
    /// Active Inference Bridge for precision-weighted prediction.
    pub active_inference_bridge: ActiveInferenceBridge,

    /// Closed Learning Loop for strategy-based behavioral adaptation.
    pub closed_learning_loop: ClosedLearningLoop,

    /// Episodic Memory Bridge for memory encoding and recall during cycles.
    pub episodic_memory: EpisodicMemoryBridge,

    /// Goal System Bridge for goal-directed attention modulation.
    pub goal_system: GoalSystemBridge,

    /// World Model Bridge for hierarchical grounded prediction.
    pub world_model: WorldModelBridge,

    /// FEP Active Inference Agent for full perception-action loop.
    pub agent: ActiveInferenceAgent,

    /// Enhanced FEP Bridge with motor system integration.
    pub enhanced_bridge: EnhancedFEPBridge,

    /// Current learning signal from FEP (for downstream systems).
    pub learning_signal: f32,

    /// FEP-driven learning rate boost (applied during CfC training step).
    /// Range: [1.0, 3.0].
    pub lr_boost: f64,

    /// Surprise-driven exploration bridge for FEP-based exploration.
    pub surprise_bridge: Option<SurpriseExplorationBridge>,
}
