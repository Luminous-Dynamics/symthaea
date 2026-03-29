// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

use symthaea_core::hdc::ContinuousHV;

/// Maximum number of planning steps before forcing completion
const MAX_PLAN_STEPS: usize = 32;

/// HDC dimension used for algorithm pattern prototype encoding
#[allow(dead_code)] // RESERVED(code-generation): CfC code sequencer config
const ALGORITHM_PATTERN_DIM: usize = 512;

/// Recognized algorithm patterns with their template plan steps
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlgorithmPattern {
    /// Sorting: compare-swap-iterate (bubble/insertion) or divide-recurse-merge
    Sorting,
    /// Search: check-bounds-compare-recurse (binary) or init-visited-enqueue-process (BFS/DFS)
    Search,
    /// Dynamic Programming: init-table, define-recurrence, fill-table, extract-result
    DynamicProgramming,
    /// Graph: init-adjacency, init-visited, process-queue, update-distances
    Graph,
    /// Accumulation: init-accumulator, iterate, update, return (sum, count, max, filter)
    Accumulation,
    /// String processing: iterate-chars, transform, collect
    StringProcessing,
}

impl AlgorithmPattern {
    /// Convert this algorithm pattern into template plan steps with context strings
    pub fn to_plan_steps(&self) -> Vec<CodePlanStep> {
        match self {
            AlgorithmPattern::Sorting => vec![
                CodePlanStep {
                    action: PlanAction::DefineFunction,
                    name: None,
                    context: vec!["algorithm:sorting".into(), "step:compare-elements".into()],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddParameter,
                    name: None,
                    context: vec!["algorithm:sorting".into(), "param:input-collection".into()],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::SetReturnType,
                    name: None,
                    context: vec![
                        "algorithm:sorting".into(),
                        "return:sorted-collection".into(),
                    ],
                    confidence: 0.85,
                },
                CodePlanStep {
                    action: PlanAction::AddMethod,
                    name: None,
                    context: vec![
                        "algorithm:sorting".into(),
                        "step:swap-or-merge".into(),
                        "step:iterate-until-sorted".into(),
                    ],
                    confidence: 0.85,
                },
            ],
            AlgorithmPattern::Search => vec![
                CodePlanStep {
                    action: PlanAction::DefineFunction,
                    name: None,
                    context: vec!["algorithm:search".into(), "step:check-bounds".into()],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddParameter,
                    name: None,
                    context: vec![
                        "algorithm:search".into(),
                        "param:search-space".into(),
                        "param:target".into(),
                    ],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::SetReturnType,
                    name: None,
                    context: vec![
                        "algorithm:search".into(),
                        "return:found-index-or-none".into(),
                    ],
                    confidence: 0.85,
                },
                CodePlanStep {
                    action: PlanAction::AddMethod,
                    name: None,
                    context: vec![
                        "algorithm:search".into(),
                        "step:compare-midpoint-or-enqueue".into(),
                        "step:recurse-or-dequeue".into(),
                    ],
                    confidence: 0.85,
                },
            ],
            AlgorithmPattern::DynamicProgramming => vec![
                CodePlanStep {
                    action: PlanAction::DefineFunction,
                    name: None,
                    context: vec!["algorithm:dp".into(), "step:init-table".into()],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddParameter,
                    name: None,
                    context: vec!["algorithm:dp".into(), "param:problem-input".into()],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddMethod,
                    name: None,
                    context: vec![
                        "algorithm:dp".into(),
                        "step:define-recurrence".into(),
                        "step:fill-table-bottom-up".into(),
                    ],
                    confidence: 0.85,
                },
                CodePlanStep {
                    action: PlanAction::SetReturnType,
                    name: None,
                    context: vec!["algorithm:dp".into(), "step:extract-result".into()],
                    confidence: 0.85,
                },
            ],
            AlgorithmPattern::Graph => vec![
                CodePlanStep {
                    action: PlanAction::DefineStruct,
                    name: None,
                    context: vec!["algorithm:graph".into(), "step:init-adjacency".into()],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::DefineFunction,
                    name: None,
                    context: vec!["algorithm:graph".into(), "step:init-visited".into()],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddMethod,
                    name: None,
                    context: vec![
                        "algorithm:graph".into(),
                        "step:process-queue".into(),
                        "step:update-distances".into(),
                    ],
                    confidence: 0.85,
                },
                CodePlanStep {
                    action: PlanAction::SetReturnType,
                    name: None,
                    context: vec!["algorithm:graph".into(), "return:distances-or-path".into()],
                    confidence: 0.85,
                },
            ],
            AlgorithmPattern::Accumulation => vec![
                CodePlanStep {
                    action: PlanAction::DefineFunction,
                    name: None,
                    context: vec![
                        "algorithm:accumulation".into(),
                        "step:init-accumulator".into(),
                    ],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddParameter,
                    name: None,
                    context: vec![
                        "algorithm:accumulation".into(),
                        "param:input-collection".into(),
                    ],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddMethod,
                    name: None,
                    context: vec![
                        "algorithm:accumulation".into(),
                        "step:iterate-elements".into(),
                        "step:update-accumulator".into(),
                    ],
                    confidence: 0.85,
                },
                CodePlanStep {
                    action: PlanAction::SetReturnType,
                    name: None,
                    context: vec![
                        "algorithm:accumulation".into(),
                        "return:accumulated-result".into(),
                    ],
                    confidence: 0.85,
                },
            ],
            AlgorithmPattern::StringProcessing => vec![
                CodePlanStep {
                    action: PlanAction::DefineFunction,
                    name: None,
                    context: vec![
                        "algorithm:string-processing".into(),
                        "step:iterate-chars".into(),
                    ],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddParameter,
                    name: None,
                    context: vec![
                        "algorithm:string-processing".into(),
                        "param:input-string".into(),
                    ],
                    confidence: 0.9,
                },
                CodePlanStep {
                    action: PlanAction::AddMethod,
                    name: None,
                    context: vec![
                        "algorithm:string-processing".into(),
                        "step:transform-char".into(),
                        "step:collect-result".into(),
                    ],
                    confidence: 0.85,
                },
                CodePlanStep {
                    action: PlanAction::SetReturnType,
                    name: None,
                    context: vec![
                        "algorithm:string-processing".into(),
                        "return:transformed-string".into(),
                    ],
                    confidence: 0.85,
                },
            ],
        }
    }
}

/// Detects algorithm patterns from intent HVs using HDC similarity
#[allow(dead_code)] // RESERVED(code-generation): code action types
struct AlgorithmPatternDetector {
    dim: usize,
    sorting_prototype: ContinuousHV,
    search_prototype: ContinuousHV,
    dp_prototype: ContinuousHV,
    graph_prototype: ContinuousHV,
    accumulation_prototype: ContinuousHV,
    string_prototype: ContinuousHV,
}

#[allow(dead_code)] // RESERVED(code-generation): generated code result
impl AlgorithmPatternDetector {
    /// Minimum similarity to consider a pattern match
    const MIN_SIMILARITY: f32 = 0.15;

    fn new(dim: usize) -> Self {
        let dim = dim.max(1);
        Self {
            dim,
            sorting_prototype: Self::encode_prototype(
                dim,
                &[
                    "sort",
                    "order",
                    "compare",
                    "swap",
                    "bubble",
                    "merge",
                    "quick",
                    "insertion",
                    "ascending",
                    "descending",
                    "partition",
                    "pivot",
                ],
            ),
            search_prototype: Self::encode_prototype(
                dim,
                &[
                    "search", "find", "binary", "linear", "lookup", "index", "bfs", "dfs",
                    "breadth", "depth", "visited", "queue",
                ],
            ),
            dp_prototype: Self::encode_prototype(
                dim,
                &[
                    "dynamic",
                    "programming",
                    "memoize",
                    "tabulate",
                    "subproblem",
                    "optimal",
                    "recurrence",
                    "knapsack",
                    "fibonacci",
                    "subsequence",
                    "cache",
                    "memo",
                ],
            ),
            graph_prototype: Self::encode_prototype(
                dim,
                &[
                    "graph",
                    "node",
                    "edge",
                    "vertex",
                    "adjacent",
                    "dijkstra",
                    "shortest",
                    "path",
                    "traverse",
                    "neighbor",
                    "connected",
                    "weight",
                ],
            ),
            accumulation_prototype: Self::encode_prototype(
                dim,
                &[
                    "sum",
                    "count",
                    "total",
                    "accumulate",
                    "fold",
                    "reduce",
                    "aggregate",
                    "max",
                    "min",
                    "average",
                    "filter",
                    "collect",
                ],
            ),
            string_prototype: Self::encode_prototype(
                dim,
                &[
                    "string",
                    "char",
                    "parse",
                    "format",
                    "split",
                    "join",
                    "replace",
                    "trim",
                    "uppercase",
                    "lowercase",
                    "substring",
                    "regex",
                ],
            ),
        }
    }

    /// Encode a set of keywords into a prototype HV using the same modular hash
    /// approach as `CodeIntentClassifier`.
    fn encode_prototype(dim: usize, keywords: &[&str]) -> ContinuousHV {
        let mut values = vec![0.0f32; dim];

        for keyword in keywords {
            let keyword_lower = keyword.to_lowercase();
            for (i, byte) in keyword_lower.bytes().enumerate() {
                let idx = ((byte as usize)
                    .wrapping_mul(31)
                    .wrapping_add(i.wrapping_mul(7)))
                    % dim;
                values[idx] += 1.0;
            }
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Encode text into an HV for comparison against prototypes
    fn encode_text(&self, text: &str) -> ContinuousHV {
        let mut values = vec![0.0f32; self.dim];
        let text_lower = text.to_lowercase();

        for (i, byte) in text_lower.bytes().enumerate() {
            let idx = ((byte as usize)
                .wrapping_mul(31)
                .wrapping_add(i.wrapping_mul(7)))
                % self.dim;
            values[idx] += 1.0;
        }

        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Detect the best-matching algorithm pattern from an intent HV.
    ///
    /// Returns `None` if no pattern exceeds the minimum similarity threshold.
    fn detect(&self, intent_hv: &ContinuousHV) -> Option<AlgorithmPattern> {
        let scores = [
            (
                AlgorithmPattern::Sorting,
                intent_hv.similarity(&self.sorting_prototype),
            ),
            (
                AlgorithmPattern::Search,
                intent_hv.similarity(&self.search_prototype),
            ),
            (
                AlgorithmPattern::DynamicProgramming,
                intent_hv.similarity(&self.dp_prototype),
            ),
            (
                AlgorithmPattern::Graph,
                intent_hv.similarity(&self.graph_prototype),
            ),
            (
                AlgorithmPattern::Accumulation,
                intent_hv.similarity(&self.accumulation_prototype),
            ),
            (
                AlgorithmPattern::StringProcessing,
                intent_hv.similarity(&self.string_prototype),
            ),
        ];

        scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .filter(|(_, sim)| *sim >= Self::MIN_SIMILARITY)
            .map(|(pattern, _)| pattern.clone())
    }

    /// Detect with scores for all patterns (useful for debugging/testing)
    fn detect_with_scores(&self, intent_hv: &ContinuousHV) -> Vec<(AlgorithmPattern, f32)> {
        let mut scores = vec![
            (
                AlgorithmPattern::Sorting,
                intent_hv.similarity(&self.sorting_prototype),
            ),
            (
                AlgorithmPattern::Search,
                intent_hv.similarity(&self.search_prototype),
            ),
            (
                AlgorithmPattern::DynamicProgramming,
                intent_hv.similarity(&self.dp_prototype),
            ),
            (
                AlgorithmPattern::Graph,
                intent_hv.similarity(&self.graph_prototype),
            ),
            (
                AlgorithmPattern::Accumulation,
                intent_hv.similarity(&self.accumulation_prototype),
            ),
            (
                AlgorithmPattern::StringProcessing,
                intent_hv.similarity(&self.string_prototype),
            ),
        ];

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}

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
    /// Algorithm pattern detector for enriching plans
    pattern_detector: AlgorithmPatternDetector,
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

        let pattern_detector = AlgorithmPatternDetector::new(hdc_dim);

        Self {
            config,
            projection,
            tau,
            w_h,
            action_prototypes,
            pattern_detector,
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

        actions
            .iter()
            .enumerate()
            .map(|(i, action)| {
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
            })
            .collect()
    }

    /// Project an HDC vector into the hidden state space
    fn project_to_hidden(&self, hv: &ContinuousHV) -> Vec<f32> {
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

    /// Detect an algorithm pattern from the intent HV using HDC similarity
    /// against prototype vectors for each known algorithm family.
    pub fn detect_algorithm_pattern(intent_hv: &ContinuousHV) -> Option<AlgorithmPattern> {
        let detector = AlgorithmPatternDetector::new(intent_hv.values.len().max(1));
        detector.detect(intent_hv)
    }

    /// Plan code structure given an intent and context.
    ///
    /// If an algorithm pattern is detected from the intent HV, the plan is
    /// enriched with algorithm-specific template steps that guide code
    /// generation toward the recognized pattern.
    pub fn plan_structure(
        &self,
        intent_hv: &ContinuousHV,
        context_hvs: &[&ContinuousHV],
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

        // Detect algorithm pattern and prepend template steps
        let mut plan = Vec::new();
        if let Some(pattern) = self.pattern_detector.detect(intent_hv) {
            plan.extend(pattern.to_plan_steps());
        }

        // Evolve CfC and collect plan steps
        let mut prev_action = plan.last().map(|s| s.action.clone());

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

use symthaea_core::math::cosine_similarity_f32 as cosine_similarity;

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
        let intent = ContinuousHV::random(512, 42);

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
        let intent = ContinuousHV::random(512, 42);
        let ctx1 = ContinuousHV::random(512, 43);
        let ctx2 = ContinuousHV::random(512, 44);

        let plan = sequencer.plan_structure(&intent, &[&ctx1, &ctx2]);
        assert!(!plan.is_empty());
    }

    #[test]
    fn test_no_consecutive_duplicates() {
        let sequencer = CfCCodeSequencer::default();
        let intent = ContinuousHV::random(512, 42);

        let plan = sequencer.plan_structure(&intent, &[]);
        for i in 1..plan.len() {
            assert_ne!(
                plan[i].action,
                plan[i - 1].action,
                "Plan should not have consecutive duplicate actions"
            );
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

    // --- Algorithm Pattern Detection Tests ---

    fn make_keyword_hv(dim: usize, keywords: &[&str]) -> ContinuousHV {
        AlgorithmPatternDetector::encode_prototype(dim, keywords)
    }

    #[test]
    fn test_sorting_pattern_detected() {
        let dim = ALGORITHM_PATTERN_DIM;
        let detector = AlgorithmPatternDetector::new(dim);
        let intent = make_keyword_hv(dim, &["sort", "compare", "swap", "ascending", "order"]);
        let pattern = detector.detect(&intent);
        assert_eq!(pattern, Some(AlgorithmPattern::Sorting));

        // Verify template steps contain sorting context
        let steps = AlgorithmPattern::Sorting.to_plan_steps();
        assert!(steps.len() >= 3);
        assert!(steps[0].context.iter().any(|c| c.contains("sorting")));
    }

    #[test]
    fn test_search_pattern_detected() {
        let dim = ALGORITHM_PATTERN_DIM;
        let detector = AlgorithmPatternDetector::new(dim);
        let intent = make_keyword_hv(dim, &["search", "binary", "find", "index", "lookup"]);
        let pattern = detector.detect(&intent);
        assert_eq!(pattern, Some(AlgorithmPattern::Search));

        let steps = AlgorithmPattern::Search.to_plan_steps();
        assert!(steps.len() >= 3);
        assert!(steps[0].context.iter().any(|c| c.contains("search")));
    }

    #[test]
    fn test_dp_pattern_detected() {
        let dim = ALGORITHM_PATTERN_DIM;
        let detector = AlgorithmPatternDetector::new(dim);
        let intent = make_keyword_hv(
            dim,
            &["dynamic", "programming", "memoize", "subproblem", "optimal"],
        );
        let pattern = detector.detect(&intent);
        assert_eq!(pattern, Some(AlgorithmPattern::DynamicProgramming));

        let steps = AlgorithmPattern::DynamicProgramming.to_plan_steps();
        assert!(steps.len() >= 3);
        assert!(steps[0].context.iter().any(|c| c.contains("dp")));
        // Should include recurrence and table-filling steps
        let all_context: Vec<&str> = steps
            .iter()
            .flat_map(|s| s.context.iter().map(|c| c.as_str()))
            .collect();
        assert!(all_context.iter().any(|c| c.contains("recurrence")));
        assert!(all_context.iter().any(|c| c.contains("table")));
    }

    #[test]
    fn test_accumulation_pattern_detected() {
        let dim = ALGORITHM_PATTERN_DIM;
        let detector = AlgorithmPatternDetector::new(dim);
        let intent = make_keyword_hv(dim, &["sum", "count", "total", "accumulate", "fold"]);
        let pattern = detector.detect(&intent);
        assert_eq!(pattern, Some(AlgorithmPattern::Accumulation));

        let steps = AlgorithmPattern::Accumulation.to_plan_steps();
        assert!(steps.len() >= 3);
        assert!(steps[0].context.iter().any(|c| c.contains("accumulation")));
    }

    #[test]
    fn test_generic_fallback() {
        let dim = ALGORITHM_PATTERN_DIM;
        let sequencer = CfCCodeSequencer::new(CfCCodeSequencerConfig {
            hdc_dim: dim,
            ..Default::default()
        });

        // A random HV with no algorithm-specific keywords should still produce a plan
        let intent = ContinuousHV::random(dim, 9999);
        let plan = sequencer.plan_structure(&intent, &[]);
        assert!(!plan.is_empty(), "Generic fallback should produce a plan");

        // All steps should have positive confidence
        for step in &plan {
            assert!(step.confidence > 0.0);
        }
    }

    #[test]
    fn test_graph_pattern_detected() {
        let dim = ALGORITHM_PATTERN_DIM;
        let detector = AlgorithmPatternDetector::new(dim);
        let intent = make_keyword_hv(dim, &["graph", "node", "edge", "dijkstra", "shortest"]);
        let pattern = detector.detect(&intent);
        assert_eq!(pattern, Some(AlgorithmPattern::Graph));

        let steps = AlgorithmPattern::Graph.to_plan_steps();
        assert!(steps.len() >= 3);
        assert!(steps[0].context.iter().any(|c| c.contains("graph")));
    }

    #[test]
    fn test_string_processing_pattern_detected() {
        let dim = ALGORITHM_PATTERN_DIM;
        let detector = AlgorithmPatternDetector::new(dim);
        let intent = make_keyword_hv(dim, &["string", "char", "parse", "split", "uppercase"]);
        let pattern = detector.detect(&intent);
        assert_eq!(pattern, Some(AlgorithmPattern::StringProcessing));
    }

    #[test]
    fn test_pattern_enriches_plan() {
        let dim = ALGORITHM_PATTERN_DIM;
        let sequencer = CfCCodeSequencer::new(CfCCodeSequencerConfig {
            hdc_dim: dim,
            ..Default::default()
        });

        // Create a sorting-flavored intent HV
        let intent = make_keyword_hv(dim, &["sort", "compare", "swap", "order", "ascending"]);
        let plan = sequencer.plan_structure(&intent, &[]);

        // Plan should contain algorithm-specific context strings from the template
        let has_sorting_context = plan
            .iter()
            .any(|step| step.context.iter().any(|c| c.contains("sorting")));
        assert!(
            has_sorting_context,
            "Sorting pattern should inject algorithm-specific context into the plan"
        );
    }

    #[test]
    fn test_detect_with_scores() {
        let dim = ALGORITHM_PATTERN_DIM;
        let detector = AlgorithmPatternDetector::new(dim);
        let intent = make_keyword_hv(dim, &["sort", "compare", "order"]);
        let scores = detector.detect_with_scores(&intent);

        // Should return all 6 patterns
        assert_eq!(scores.len(), 6);
        // Should be sorted descending by score
        for i in 1..scores.len() {
            assert!(scores[i - 1].1 >= scores[i].1);
        }
        // Top match should be Sorting
        assert_eq!(scores[0].0, AlgorithmPattern::Sorting);
    }
}
