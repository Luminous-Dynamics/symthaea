//! FEP manager: groups active inference bridge, closed learning loop, episodic
//! memory bridge, goal system, world model, FEP agent, enhanced FEP bridge,
//! and the two FEP-derived scalar signals.

use crate::consciousness::fep_active_inference::{ActiveInferenceAgent, EnhancedFEPBridge};
use crate::cognitive_loop::goal_world::{GoalSystemBridge, WorldModelBridge};
use crate::cognitive_loop::routing::ActiveInferenceBridge;
use crate::cognitive_loop::learning::ClosedLearningLoop;
use crate::cognitive_loop::memory_bridge::EpisodicMemoryBridge;

/// Consolidates the nine FEP-related CLS fields.
pub(crate) struct FepManager {
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
    pub fep_agent: ActiveInferenceAgent,

    /// Enhanced FEP Bridge with motor system integration.
    pub enhanced_fep_bridge: EnhancedFEPBridge,

    /// Current learning signal from FEP (for downstream systems).
    pub learning_signal: f32,

    /// FEP-driven learning rate boost (applied during CfC training step).
    pub lr_boost: f64,
}
