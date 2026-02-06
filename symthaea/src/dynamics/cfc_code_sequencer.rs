//! CfC Code Sequencer — Use CfC temporal dynamics to plan code structure
//!
//! Given an intent HV and context, produces a sequence of code-structure
//! decisions using the Closed-form Continuous-time network. Each time step
//! represents one structural decision (e.g., "define struct" → "add field" →
//! "impl trait" → "add method").
//!
//! # Architecture
//!
//! ```text
//! Intent HV (16,384D)
//!     ↓ project to CfC state space
//! CfC State (hidden_dim)
//!     ↓ step() × N
//! Sequence of CodePlanSteps
//!     ↓ decode
//! Code generation plan
//! ```

use symthaea_core::hdc::RealHV;

use crate::hdc::code_encoder::CodeHDEncoder;
use crate::language::code_parser::EntityKind;

/// Maximum number of planning steps before forcing completion
const MAX_PLAN_STEPS: usize = 32;

/// A single step in a code generation plan
#[derive(Debug, Clone)]
pub struct CodePlanStep {
    /// What kind of code element to create
    pub action: PlanAction,
    /// Name for the element (if applicable)
    pub name: Option<String>,
    /// Additional context for this step
    pub context: Vec<String>,
    /// Confidence in this step (0.0 - 1.0)
    pub confidence: f32,
}

/// Actions the code planner can take
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanAction {
    /// Define a new function
    DefineFunction,
    /// Define a new struct/class
    DefineStruct,
    /// Define a new enum
    DefineEnum,
    /// Define a trait/interface
    DefineTrait,
    /// Implement a trait for a type
    ImplTrait,
    /// Add a field to a struct
    AddField,
    /// Add a method
    AddMethod,
    /// Add an import
    AddImport,
    /// Add a parameter
    AddParameter,
    /// Set return type
    SetReturnType,
    /// Add error handling
    AddErrorHandling,
    /// Add documentation
    AddDocumentation,
    /// Complete — plan is finished
    Complete,
}

/// Configuration for the CfC code sequencer
#[derive(Debug, Clone)]
pub struct CfCCodeSequencerConfig {
    /// Dimension of the HDC space
    pub hdc_dim: usize,
    /// Hidden dimension for the simplified CfC state
    pub hidden_dim: usize,
    /// Time step for CfC evolution
    pub dt: f32,
    /// Threshold below which we consider the plan complete
    pub completion_threshold: f32,
    /// Maximum planning steps
    pub max_steps: usize,
}

impl Default for CfCCodeSequencerConfig {
    fn default() -> Self {
        Self {
            hdc_dim: 512,
            hidden_dim: 64,
            dt: 1.0,
            completion_threshold: 0.1,
            max_steps: MAX_PLAN_STEPS,
        }
    }
}

/// CfC-based code structure planner
///
/// Uses simplified CfC dynamics to evolve a state vector that encodes
/// the current "plan progress". At each time step, the state is decoded
/// into a structural decision (PlanAction).
pub struct CfCCodeSequencer {
    config: CfCCodeSequencerConfig,
    /// Projection from HDC space to hidden state
    projection: Vec<f32>,
    /// Time constants for each hidden unit
    tau: Vec<f32>,
    /// Recurrent weights (hidden × hidden)
    w_h: Vec<f32>,
    /// Action prototypes: each PlanAction maps to a hidden-dim vector
    action_prototypes: Vec<(PlanAction, Vec<f32>)>,
}

impl CfCCodeSequencer {
    /// Create a new CfC code sequencer with the given config
    pub fn new(config: CfCCodeSequencerConfig) -> Self {
        let hdc_dim = config.hdc_dim;
        let hidden_dim = config.hidden_dim;

        // Initialize projection matrix (hdc_dim → hidden_dim) deterministically
        let mut projection = vec![0.0f32; hdc_dim * hidden_dim];
        for i in 0..projection.len() {
            // Deterministic pseudo-random initialization
            let seed = (i as u64).wrapping_mul(2_654_435_761).wrapping_add(1);
            let frac = (seed as f32) / (u64::MAX as f32);
            projection[i] = (frac * 2.0 - 1.0) / (hdc_dim as f32).sqrt();
        }

        // Time constants
        let tau: Vec<f32> = (0..hidden_dim)
            .map(|i| 0.5 + (i as f32 / hidden_dim as f32) * 9.5) // range [0.5, 10.0]
            .collect();

        // Recurrent weights (identity-like with small noise)
        let mut w_h = vec![0.0f32; hidden_dim * hidden_dim];
        for i in 0..hidden_dim {
            w_h[i * hidden_dim + i] = 0.9; // diagonal
            for j in 0..hidden_dim {
                if i != j {
                    let seed = ((i * hidden_dim + j) as u64).wrapping_mul(3_141_592_653);
                    let frac = (seed as f32) / (u64::MAX as f32);
                    w_h[i * hidden_dim + j] = (frac * 2.0 - 1.0) * 0.05;
                }
            }
        }

        // Action prototypes in hidden space
        let action_prototypes = Self::init_action_prototypes(hidden_dim);

        Self {
            config,
            projection,
            tau,
            w_h,
            action_prototypes,
        }
    }

    /// Initialize action prototype vectors
    fn init_action_prototypes(hidden_dim: usize) -> Vec<(PlanAction, Vec<f32>)> {
        let actions = [
            PlanAction::DefineFunction,
            PlanAction::DefineStruct,
            PlanAction::DefineEnum,
            PlanAction::DefineTrait,
            PlanAction::ImplTrait,
            PlanAction::AddField,
            PlanAction::AddMethod,
            PlanAction::AddImport,
            PlanAction::AddParameter,
            PlanAction::SetReturnType,
            PlanAction::AddErrorHandling,
            PlanAction::AddDocumentation,
            PlanAction::Complete,
        ];

        actions.iter().enumerate().map(|(i, action)| {
            let mut proto = vec![0.0f32; hidden_dim];
            // Deterministic initialization: each action activates different hidden units
            let base = (i * hidden_dim) / actions.len();
            let width = hidden_dim / actions.len();
            for j in base..(base + width).min(hidden_dim) {
                proto[j] = 1.0;
            }
            // Normalize
            let mag: f32 = proto.iter().map(|v| v * v).sum::<f32>().sqrt();
            if mag > 0.0 {
                for v in &mut proto {
                    *v /= mag;
                }
            }
            (action.clone(), proto)
        }).collect()
    }

    /// Project an HDC vector into the hidden state space
    fn project_to_hidden(&self, hv: &RealHV) -> Vec<f32> {
        let hdc_dim = self.config.hdc_dim;
        let hidden_dim = self.config.hidden_dim;
        let mut hidden = vec![0.0f32; hidden_dim];

        // Use only first hdc_dim values from the HV
        let hv_len = hv.values.len().min(hdc_dim);

        for h in 0..hidden_dim {
            let mut sum = 0.0f32;
            for i in 0..hv_len {
                sum += hv.values[i] * self.projection[i * hidden_dim + h];
            }
            hidden[h] = sum.tanh();
        }

        hidden
    }

    /// CfC step: evolve state by dt using closed-form dynamics
    fn cfc_step(&self, state: &[f32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let dt = self.config.dt;
        let mut new_state = vec![0.0f32; hidden_dim];

        for i in 0..hidden_dim {
            // Compute input from recurrent connections
            let mut recurrent_input = 0.0f32;
            for j in 0..hidden_dim {
                recurrent_input += self.w_h[i * hidden_dim + j] * state[j];
            }

            // CfC closed-form update: h(t+dt) = h(t) * exp(-dt/tau) + f(x) * (1 - exp(-dt/tau))
            let decay = (-dt / self.tau[i]).exp();
            let activation = recurrent_input.tanh();
            new_state[i] = state[i] * decay + activation * (1.0 - decay);
        }

        new_state
    }

    /// Decode hidden state into a plan action by finding nearest prototype
    fn decode_action(&self, state: &[f32]) -> (PlanAction, f32) {
        let mut best_action = PlanAction::Complete;
        let mut best_sim = f32::NEG_INFINITY;

        for (action, proto) in &self.action_prototypes {
            let sim = cosine_similarity(state, proto);
            if sim > best_sim {
                best_sim = sim;
                best_action = action.clone();
            }
        }

        (best_action, best_sim.max(0.0))
    }

    /// Plan code structure given an intent and context
    pub fn plan_structure(
        &self,
        intent_hv: &RealHV,
        context_hvs: &[&RealHV],
    ) -> Vec<CodePlanStep> {
        // Project intent into hidden space
        let mut state = self.project_to_hidden(intent_hv);

        // Blend in context
        for ctx_hv in context_hvs {
            let ctx_hidden = self.project_to_hidden(ctx_hv);
            for i in 0..state.len() {
                state[i] = state[i] * 0.7 + ctx_hidden[i] * 0.3;
            }
        }

        // Evolve CfC and collect plan steps
        let mut plan = Vec::new();
        let mut prev_action = None;

        for _step in 0..self.config.max_steps {
            state = self.cfc_step(&state);
            let (action, confidence) = self.decode_action(&state);

            // Stop if confidence is too low or we hit Complete
            if confidence < self.config.completion_threshold || action == PlanAction::Complete {
                break;
            }

            // Avoid repeating the same action consecutively
            if prev_action.as_ref() == Some(&action) {
                continue;
            }

            plan.push(CodePlanStep {
                action: action.clone(),
                name: None,
                context: Vec::new(),
                confidence,
            });

            prev_action = Some(action);
        }

        // Ensure we have at least one step
        if plan.is_empty() {
            plan.push(CodePlanStep {
                action: PlanAction::DefineFunction,
                name: None,
                context: Vec::new(),
                confidence: 0.5,
            });
        }

        plan
    }
}

impl Default for CfCCodeSequencer {
    fn default() -> Self {
        Self::new(CfCCodeSequencerConfig::default())
    }
}

/// Cosine similarity between two f32 slices
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;

    for i in 0..n {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }

    let mag = (mag_a * mag_b).sqrt();
    if mag > 0.0 { dot / mag } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sequencer() {
        let sequencer = CfCCodeSequencer::default();
        assert_eq!(sequencer.config.hidden_dim, 64);
    }

    #[test]
    fn test_plan_produces_steps() {
        let sequencer = CfCCodeSequencer::default();
        let intent = RealHV::random(512, 42);

        let plan = sequencer.plan_structure(&intent, &[]);
        assert!(!plan.is_empty());

        // All steps should have positive confidence
        for step in &plan {
            assert!(step.confidence > 0.0);
        }
    }

    #[test]
    fn test_plan_with_context() {
        let sequencer = CfCCodeSequencer::default();
        let intent = RealHV::random(512, 42);
        let ctx1 = RealHV::random(512, 43);
        let ctx2 = RealHV::random(512, 44);

        let plan = sequencer.plan_structure(&intent, &[&ctx1, &ctx2]);
        assert!(!plan.is_empty());
    }

    #[test]
    fn test_no_consecutive_duplicates() {
        let sequencer = CfCCodeSequencer::default();
        let intent = RealHV::random(512, 42);

        let plan = sequencer.plan_structure(&intent, &[]);
        for i in 1..plan.len() {
            assert_ne!(plan[i].action, plan[i - 1].action,
                "Plan should not have consecutive duplicate actions");
        }
    }

    #[test]
    fn test_cfc_step_stability() {
        let sequencer = CfCCodeSequencer::default();
        let mut state = vec![1.0f32; sequencer.config.hidden_dim];

        // Run many steps - should not blow up
        for _ in 0..100 {
            state = sequencer.cfc_step(&state);
        }

        // State should remain bounded
        for v in &state {
            assert!(v.is_finite());
            assert!(v.abs() < 10.0, "CfC state should remain bounded: {}", v);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }
}
