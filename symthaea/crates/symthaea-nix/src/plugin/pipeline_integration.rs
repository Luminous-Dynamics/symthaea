//! Pipeline Integration — Register NixOS Mind into the Conscious Pipeline
//!
//! Provides the types and traits needed to wire symthaea-nix's active inference
//! engine into the main Symthaea `ConsciousPipeline`. Since symthaea-nix cannot
//! import the main crate, this module defines:
//!
//! - `NixPipelineStage`: stages of NixOS processing for the pipeline
//! - `NixPipelineResult`: domain-specific result that augments `PipelineResult`
//! - `NixPipelineHook`: trait the main crate implements to inject NixOS cognition

use symthaea_core::hdc::ContinuousHV;

use crate::action::executor::{NixOSCommand, SafetyLevel};
use crate::encoding::NixCodebook;
use crate::mind::{ActionCategory, ActionPlan, NixActiveInference};

// =============================================================================
// PIPELINE STAGES
// =============================================================================

/// Stages of NixOS processing within the consciousness pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixPipelineStage {
    /// Observe current system state (sensory input).
    Observe,
    /// Encode observations into HDC space (perception).
    Encode,
    /// Infer user's goal from input + context (cognition).
    InferGoal,
    /// Generate action plan via active inference (decision).
    PlanActions,
    /// Gate action execution based on Φ (consciousness).
    PhiGate,
    /// Execute the selected action (motor output).
    Execute,
    /// Record outcome and learn (consolidation).
    Learn,
}

impl NixPipelineStage {
    /// Human-readable name for display.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Observe => "Observe",
            Self::Encode => "Encode",
            Self::InferGoal => "Infer Goal",
            Self::PlanActions => "Plan Actions",
            Self::PhiGate => "Φ Gate",
            Self::Execute => "Execute",
            Self::Learn => "Learn",
        }
    }

    /// Ordered sequence of all stages.
    pub fn all() -> &'static [NixPipelineStage] {
        &[
            Self::Observe,
            Self::Encode,
            Self::InferGoal,
            Self::PlanActions,
            Self::PhiGate,
            Self::Execute,
            Self::Learn,
        ]
    }
}

// =============================================================================
// PIPELINE RESULT
// =============================================================================

/// NixOS-specific result that augments the main pipeline's `PipelineResult`.
#[derive(Debug, Clone)]
pub struct NixPipelineResult {
    /// The action plan from active inference.
    pub plan: ActionPlan,
    /// The consciousness quadrant at decision time.
    pub quadrant: NixConsciousnessQuadrant,
    /// Current free energy (how far from the goal).
    pub free_energy: f64,
    /// Prediction errors at each hierarchy level.
    pub hierarchy_errors: [f64; 4],
    /// Whether the system was surprised by recent observations.
    pub is_surprised: bool,
    /// Stage at which processing stopped (if early exit).
    pub completed_stage: NixPipelineStage,
    /// The generated NixOS command (if any).
    pub command: Option<NixOSCommand>,
    /// Safety level of the command.
    pub safety_level: SafetyLevel,
    /// Whether Φ gating allowed execution.
    pub phi_allowed: bool,
    /// Working memory items active during this decision.
    pub active_memory: Vec<(String, f64)>,
}

/// Consciousness quadrant for NixOS decision-making.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixConsciousnessQuadrant {
    /// High Φ + High Confidence: deep understanding, execute with trust.
    Confident,
    /// High Φ + Low Confidence: deep but exploring, dry-run first.
    Curious,
    /// Low Φ + High Confidence: pattern-matched routine.
    Habitual,
    /// Low Φ + Low Confidence: genuinely confused, ask for help.
    Confused,
}

impl NixConsciousnessQuadrant {
    /// Determine quadrant from Φ and confidence values.
    pub fn from_metrics(phi: f64, confidence: f64) -> Self {
        match (phi >= 0.5, confidence >= 0.5) {
            (true, true) => Self::Confident,
            (true, false) => Self::Curious,
            (false, true) => Self::Habitual,
            (false, false) => Self::Confused,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Confident => "Confident",
            Self::Curious => "Curious",
            Self::Habitual => "Habitual",
            Self::Confused => "Confused",
        }
    }

    /// Whether this quadrant allows action execution.
    pub fn allows_execution(&self) -> bool {
        matches!(self, Self::Confident | Self::Habitual)
    }

    /// Whether this quadrant suggests asking a clarifying question.
    pub fn suggests_clarification(&self) -> bool {
        matches!(self, Self::Curious | Self::Confused)
    }
}

// =============================================================================
// PIPELINE HOOK TRAIT
// =============================================================================

/// Trait that the main Symthaea crate implements to wire NixOS cognition
/// into the `ConsciousPipeline`.
pub trait NixPipelineHook: Send + Sync {
    /// Called before NixOS processing begins. Returns current system Φ.
    fn pre_process(&self) -> f64;

    /// Run the full NixOS cognition pipeline on user input.
    fn process_nix_input(&mut self, input: &str, phi: f64, confidence: f64) -> NixPipelineResult;

    /// Called after action execution to record the outcome.
    fn post_execute(&mut self, action: &str, success: bool, output: &str);

    /// Provide user feedback for adaptive threshold learning.
    fn feedback(&mut self, was_positive: bool);
}

// =============================================================================
// STANDALONE PIPELINE PROCESSOR
// =============================================================================

/// Standalone NixOS pipeline processor that can be used without the main
/// Symthaea pipeline. Used by the `nix-mind` CLI and TUI.
pub struct NixPipelineProcessor {
    engine: NixActiveInference,
    codebook: NixCodebook,
    phi_threshold: f64,
    skip_observe: bool,
}

impl NixPipelineProcessor {
    /// Create a new standalone processor.
    pub fn new() -> Self {
        Self {
            engine: NixActiveInference::new(),
            codebook: NixCodebook::new(),
            phi_threshold: 0.3,
            skip_observe: false,
        }
    }

    /// Set the minimum Φ for action execution.
    pub fn with_phi_threshold(mut self, threshold: f64) -> Self {
        self.phi_threshold = threshold;
        self
    }

    /// Skip live system observation (useful for testing or offline use).
    pub fn with_skip_observe(mut self, skip: bool) -> Self {
        self.skip_observe = skip;
        self
    }

    /// Process user input through the full NixOS cognition pipeline.
    #[tracing::instrument(skip(self), fields(phi = %phi, confidence = %confidence))]
    pub fn process(&mut self, input: &str, phi: f64, confidence: f64) -> NixPipelineResult {
        // Stage 1: Observe (skip if configured, e.g. in tests)
        if !self.skip_observe {
            if let Ok(snapshot) = crate::observe::SystemObserver::snapshot() {
                let mut encoder = crate::encoding::SystemStateEncoder::new(&mut self.codebook);
                let state_hv = encoder.encode_snapshot(&snapshot);
                self.engine.observe_state(state_hv);
            }
        }

        // Stage 2+3: Infer goal + plan actions
        let plan = self.engine.process_input(input);

        // Stage 4: Determine consciousness quadrant
        let quadrant = NixConsciousnessQuadrant::from_metrics(phi, confidence);

        // Get hierarchy state
        let wm = self.engine.world_model();
        let hierarchy = wm.prediction_hierarchy();

        // Determine safety level from best action
        let safety_level = if plan.needs_clarification {
            SafetyLevel::ReadOnly
        } else {
            match plan.actions.first().map(|a| &a.action) {
                Some(ActionCategory::Rebuild) | Some(ActionCategory::Update) => {
                    SafetyLevel::SystemModify
                }
                Some(ActionCategory::Rollback) => SafetyLevel::SystemCritical,
                Some(ActionCategory::GarbageCollect) => SafetyLevel::Destructive,
                Some(ActionCategory::Install) | Some(ActionCategory::Remove) => {
                    SafetyLevel::UserModify
                }
                Some(ActionCategory::Enable) | Some(ActionCategory::Disable) => {
                    SafetyLevel::SystemModify
                }
                Some(ActionCategory::Configure) => SafetyLevel::UserModify,
                _ => SafetyLevel::ReadOnly,
            }
        };

        // Stage 5: Φ gate
        let phi_allowed = phi >= self.phi_threshold && quadrant.allows_execution();

        // Build working memory snapshot
        let active_memory = self
            .engine
            .goal_inference()
            .working_memory()
            .items()
            .iter()
            .map(|item| (item.label.clone(), item.activation))
            .collect();

        NixPipelineResult {
            plan,
            quadrant,
            free_energy: wm.free_energy(),
            hierarchy_errors: hierarchy.errors(),
            is_surprised: hierarchy.is_surprised(),
            completed_stage: if phi_allowed {
                NixPipelineStage::PhiGate
            } else {
                NixPipelineStage::PlanActions
            },
            command: None,
            safety_level,
            phi_allowed,
            active_memory,
        }
    }

    /// Record outcome and learn from it.
    pub fn learn(&mut self, action: &str, success: bool, state_after: ContinuousHV) {
        let outcome = if success {
            crate::mind::EpisodeOutcome::Success
        } else {
            crate::mind::EpisodeOutcome::Failure("execution failed".into())
        };
        let state_before = self.engine.world_model().system_state().clone();
        let action_cat = ActionCategory::from_command(action);
        self.engine
            .learn_from_outcome(&state_before, action_cat, &state_after, outcome, 0.5);
    }

    /// Access the underlying engine.
    pub fn engine(&self) -> &NixActiveInference {
        &self.engine
    }

    /// Mutable access to the underlying engine.
    pub fn engine_mut(&mut self) -> &mut NixActiveInference {
        &mut self.engine
    }
}

impl Default for NixPipelineProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_stages() {
        assert_eq!(NixPipelineStage::all().len(), 7);
        assert_eq!(NixPipelineStage::Observe.name(), "Observe");
        assert_eq!(NixPipelineStage::PhiGate.name(), "Φ Gate");
    }

    #[test]
    fn test_consciousness_quadrant() {
        assert_eq!(
            NixConsciousnessQuadrant::from_metrics(0.8, 0.9),
            NixConsciousnessQuadrant::Confident,
        );
        assert_eq!(
            NixConsciousnessQuadrant::from_metrics(0.8, 0.2),
            NixConsciousnessQuadrant::Curious,
        );
        assert_eq!(
            NixConsciousnessQuadrant::from_metrics(0.2, 0.8),
            NixConsciousnessQuadrant::Habitual,
        );
        assert_eq!(
            NixConsciousnessQuadrant::from_metrics(0.2, 0.2),
            NixConsciousnessQuadrant::Confused,
        );
    }

    #[test]
    fn test_quadrant_execution_rules() {
        assert!(NixConsciousnessQuadrant::Confident.allows_execution());
        assert!(NixConsciousnessQuadrant::Habitual.allows_execution());
        assert!(!NixConsciousnessQuadrant::Curious.allows_execution());
        assert!(!NixConsciousnessQuadrant::Confused.allows_execution());
    }

    #[test]
    fn test_quadrant_clarification_rules() {
        assert!(!NixConsciousnessQuadrant::Confident.suggests_clarification());
        assert!(!NixConsciousnessQuadrant::Habitual.suggests_clarification());
        assert!(NixConsciousnessQuadrant::Curious.suggests_clarification());
        assert!(NixConsciousnessQuadrant::Confused.suggests_clarification());
    }

    #[test]
    fn test_standalone_processor() {
        let mut proc = NixPipelineProcessor::new().with_skip_observe(true);
        let result = proc.process("install firefox", 0.7, 0.8);

        assert_eq!(result.quadrant, NixConsciousnessQuadrant::Confident);
        assert!(result.phi_allowed);
        assert!(!result.plan.goal.description.is_empty());
    }

    #[test]
    fn test_low_phi_blocks_execution() {
        let mut proc = NixPipelineProcessor::new()
            .with_phi_threshold(0.5)
            .with_skip_observe(true);
        let result = proc.process("install firefox", 0.2, 0.8);

        assert_eq!(result.quadrant, NixConsciousnessQuadrant::Habitual);
        assert!(!result.phi_allowed);
    }

    #[test]
    fn test_curious_blocks_execution() {
        let mut proc = NixPipelineProcessor::new().with_skip_observe(true);
        let result = proc.process("make it faster", 0.7, 0.3);

        assert_eq!(result.quadrant, NixConsciousnessQuadrant::Curious);
        assert!(!result.phi_allowed);
        assert!(result.quadrant.suggests_clarification());
    }

    #[test]
    fn test_learn_from_outcome() {
        let mut proc = NixPipelineProcessor::new();
        let dim = proc.engine().world_model().system_state().dim();
        let state_after = ContinuousHV::random(dim, 1);
        proc.learn("install firefox", true, state_after);
        assert_eq!(proc.engine().episode_count(), 1);
    }
}
