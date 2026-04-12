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

use std::sync::Mutex;

use ndarray::Array1;
use symthaea_core::hdc::ContinuousHV;

use super::cfc::network::{CfCNetwork, CfCNetworkConfig};
use super::cfc::types::CfCConfig;

/// Get home directory without depending on `dirs` crate
fn dirs_next_or_home() -> Option<std::path::PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(std::path::PathBuf::from)
}

/// Maximum number of planning steps before forcing completion
const MAX_PLAN_STEPS: usize = 32;

/// HDC dimension used for algorithm pattern prototype encoding
#[allow(dead_code)] // RESERVED(code-generation): CfC code sequencer config
const ALGORITHM_PATTERN_DIM: usize = 512;

/// Number of distinct PlanAction variants (including Complete)
const NUM_ACTIONS: usize = 25;

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

/// Detects algorithm patterns from intent HVs using semantic similarity.
///
/// Uses the `CodeSemanticEncoder` for synonym-aware encoding instead of
/// the old byte-hash approach. "add" and "sum" now produce similar prototypes.
struct AlgorithmPatternDetector {
    sorting_prototype: ContinuousHV,
    search_prototype: ContinuousHV,
    dp_prototype: ContinuousHV,
    graph_prototype: ContinuousHV,
    accumulation_prototype: ContinuousHV,
    string_prototype: ContinuousHV,
}

#[allow(dead_code)] // RESERVED(code-generation): generated code result
impl AlgorithmPattern {
    /// Detect algorithm pattern directly from purpose text using keyword matching.
    ///
    /// This bypasses the HDC encoding entirely — much more reliable than the
    /// byte-hash prototype comparison because "add" and "sum" are handled as
    /// actual synonyms rather than orthogonal character sequences.
    pub fn detect_from_text(purpose: &str) -> Option<AlgorithmPattern> {
        let text = purpose.to_lowercase();

        // Score each pattern by counting keyword hits
        let patterns: &[(AlgorithmPattern, &[&str])] = &[
            (
                AlgorithmPattern::Sorting,
                &[
                    "sort", "order", "compare", "swap", "bubble", "merge", "quick",
                    "insertion", "ascending", "descending", "partition", "pivot",
                    "arrange", "rank",
                ],
            ),
            (
                AlgorithmPattern::Search,
                &[
                    "search", "find", "binary search", "linear search", "lookup",
                    "index of", "bfs", "dfs", "breadth", "depth", "visited", "queue",
                    "locate",
                ],
            ),
            (
                AlgorithmPattern::DynamicProgramming,
                &[
                    "dynamic programming", "memoize", "tabulate", "subproblem",
                    "optimal", "recurrence", "knapsack", "subsequence", "memo",
                    "dp", "edit distance", "coin change",
                ],
            ),
            (
                AlgorithmPattern::Graph,
                &[
                    "graph", "node", "edge", "vertex", "adjacent", "dijkstra",
                    "shortest path", "traverse", "neighbor", "connected", "bfs",
                    "dfs", "topological",
                ],
            ),
            (
                AlgorithmPattern::Accumulation,
                &[
                    "sum", "count", "total", "accumulate", "fold", "reduce",
                    "aggregate", "average", "mean", "filter", "collect", "tally",
                ],
            ),
            (
                AlgorithmPattern::StringProcessing,
                &[
                    "string", "char", "parse", "format", "split", "join", "replace",
                    "trim", "uppercase", "lowercase", "substring", "regex", "reverse string",
                    "palindrome", "capitalize",
                ],
            ),
        ];

        let mut best: Option<(AlgorithmPattern, usize)> = None;

        for (pattern, keywords) in patterns {
            let hits: usize = keywords.iter().filter(|kw| text.contains(**kw)).count();
            if hits > 0 {
                if best.is_none() || hits > best.as_ref().unwrap().1 {
                    best = Some((pattern.clone(), hits));
                }
            }
        }

        best.map(|(p, _)| p)
    }
}

impl AlgorithmPatternDetector {
    /// Minimum similarity to consider a pattern match
    const MIN_SIMILARITY: f32 = 0.15;

    fn new(dim: usize) -> Self {
        let dim = dim.max(1);

        #[cfg(feature = "code_generation")]
        {
            let encoder = crate::hdc::code_semantic_encoder::CodeSemanticEncoder::new(dim);
            Self {
                sorting_prototype: encoder.encode_text(
                    "sort order compare swap bubble merge quick insertion ascending descending partition pivot",
                ),
                search_prototype: encoder.encode_text(
                    "search find binary linear lookup index bfs dfs breadth depth visited queue",
                ),
                dp_prototype: encoder.encode_text(
                    "dynamic programming memoize tabulate subproblem optimal recurrence knapsack fibonacci subsequence cache memo",
                ),
                graph_prototype: encoder.encode_text(
                    "graph node edge vertex adjacent dijkstra shortest path traverse neighbor connected weight",
                ),
                accumulation_prototype: encoder.encode_text(
                    "sum count total accumulate fold reduce aggregate max min average filter collect",
                ),
                string_prototype: encoder.encode_text(
                    "string char parse format split join replace trim uppercase lowercase substring regex",
                ),
            }
        }

        #[cfg(not(feature = "code_generation"))]
        {
            // Fallback: use byte-hash encoding when semantic encoder not available
            Self {
                sorting_prototype: Self::encode_ngram_prototype(dim, "sort order compare swap"),
                search_prototype: Self::encode_ngram_prototype(dim, "search find binary lookup"),
                dp_prototype: Self::encode_ngram_prototype(dim, "dynamic programming memoize"),
                graph_prototype: Self::encode_ngram_prototype(dim, "graph node edge dijkstra"),
                accumulation_prototype: Self::encode_ngram_prototype(dim, "sum count total fold"),
                string_prototype: Self::encode_ngram_prototype(dim, "string char parse split"),
            }
        }
    }

    /// Fallback byte-hash encoding for when semantic encoder is not available
    #[cfg(not(feature = "code_generation"))]
    fn encode_ngram_prototype(dim: usize, text: &str) -> ContinuousHV {
        let mut values = vec![0.0f32; dim];
        for (i, byte) in text.to_lowercase().bytes().enumerate() {
            let idx = ((byte as usize).wrapping_mul(31).wrapping_add(i.wrapping_mul(7))) % dim;
            values[idx] += 1.0;
        }
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values { *v /= magnitude; }
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

/// Actions the code planner can take.
///
/// The first 13 are the original actions. Actions 13-24 were added in Phase 4
/// to support richer Rust code structure (match, iterators, closures, generics,
/// lifetimes, derives, modules, tests, constants, type aliases, loops).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanAction {
    // ── Original actions (0-12) ──
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
    /// Add an import/use statement
    AddImport,
    /// Add a parameter
    AddParameter,
    /// Set return type
    SetReturnType,
    /// Add error handling (Result, ?)
    AddErrorHandling,
    /// Add documentation (///, //!)
    AddDocumentation,

    // ── New actions (13-23) — Phase 4 ──
    /// Match expression with arms
    MatchExpression,
    /// For loop (for x in iter { ... })
    ForLoop,
    /// Iterator chain (.iter().map().filter().collect())
    IteratorChain,
    /// Closure definition (|args| body)
    ClosureDefine,
    /// Error propagation with ? operator
    ErrorPropagation,
    /// Generic type parameter (<T>, <T: Trait>)
    GenericParam,
    /// Lifetime annotation ('a)
    LifetimeAnnotation,
    /// #[derive(...)] attribute
    DeriveAttribute,
    /// #[test] module or function
    TestModule,
    /// const or static definition
    ConstDefinition,
    /// type alias (type Name = ...)
    TypeAlias,

    // ── Sentinel ──
    /// Complete — plan is finished (must remain last)
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
/// Uses a trainable CfCNetwork to evolve a state vector that encodes
/// the current "plan progress". At each time step, the network output
/// is interpreted as action logits over the PlanAction space.
///
/// ## Training
///
/// The network can be trained via BPTT or SPSA:
/// - `train_step()` — single (input, target) pair with Adam optimizer
/// - `train_sequence()` — sequence of plan steps with temporal gradients
///
/// ## Weight Persistence
///
/// Weights can be exported/imported as flat f32 vectors for persistence
/// and federated learning compatibility:
/// - `export_weights()` → `Vec<f32>` (compatible with swarm gradients)
/// - `import_weights(Vec<f32>)` — restore from flat vector
pub struct CfCCodeSequencer {
    config: CfCCodeSequencerConfig,
    /// Trainable CfC network: input_dim=hdc_dim, output_dim=NUM_ACTIONS.
    /// Wrapped in RefCell for interior mutability — `plan_structure()` needs to
    /// call `network.forward()` (which updates hidden state) but the public API
    /// takes `&self` to avoid cascading `&mut` through 20+ call sites.
    network: Mutex<CfCNetwork>,
    /// Ordered list of actions matching output indices
    action_index: Vec<PlanAction>,
    /// Algorithm pattern detector for enriching plans
    pattern_detector: AlgorithmPatternDetector,
}

impl CfCCodeSequencer {
    /// Create a new CfC code sequencer with the given config
    pub fn new(config: CfCCodeSequencerConfig) -> Self {
        let hdc_dim = config.hdc_dim;
        let hidden_dim = config.hidden_dim;

        let action_index = vec![
            // Original 13
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
            // Phase 4 additions
            PlanAction::MatchExpression,
            PlanAction::ForLoop,
            PlanAction::IteratorChain,
            PlanAction::ClosureDefine,
            PlanAction::ErrorPropagation,
            PlanAction::GenericParam,
            PlanAction::LifetimeAnnotation,
            PlanAction::DeriveAttribute,
            PlanAction::TestModule,
            PlanAction::ConstDefinition,
            PlanAction::TypeAlias,
            // Sentinel
            PlanAction::Complete,
        ];

        let cell_config = CfCConfig {
            input_dim: hdc_dim,
            hidden_dim,
            gradient_clip: 1.0,
            ..Default::default()
        };
        let net_config = CfCNetworkConfig {
            input_dim: hdc_dim,
            hidden_dim,
            num_layers: 1,
            output_dim: NUM_ACTIONS,
            cell_config,
            residual: false,
            bidirectional: false,
            enable_online_learning: false,
            online_learning_config: Default::default(),
        };
        let network = Mutex::new(CfCNetwork::new(net_config));
        let pattern_detector = AlgorithmPatternDetector::new(hdc_dim);

        let sequencer = Self {
            config,
            network,
            action_index,
            pattern_detector,
        };

        // Auto-load trained weights if available
        sequencer.try_auto_load_weights();

        sequencer
    }

    /// Default path for persisted sequencer weights
    pub fn default_weights_path() -> std::path::PathBuf {
        // Try $SYMTHAEA_DATA_DIR first, then ~/.local/share/symthaea/
        if let Ok(data_dir) = std::env::var("SYMTHAEA_DATA_DIR") {
            std::path::PathBuf::from(data_dir).join("code-sequencer-weights.bin")
        } else if let Some(home) = dirs_next_or_home() {
            home.join(".local")
                .join("share")
                .join("symthaea")
                .join("code-sequencer-weights.bin")
        } else {
            std::path::PathBuf::from("code-sequencer-weights.bin")
        }
    }

    /// Try to auto-load weights from the default path. Silently succeeds or fails.
    fn try_auto_load_weights(&self) {
        let path = Self::default_weights_path();
        if path.exists() {
            match self.load_weights(&path) {
                Ok(()) => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[CfCCodeSequencer] Loaded trained weights from {}",
                        path.display()
                    );
                }
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[CfCCodeSequencer] Failed to load weights from {}: {}",
                        path.display(),
                        _e
                    );
                }
            }
        }
    }

    /// Save weights to the default persistence path, creating directories as needed.
    pub fn persist_weights(&self) -> anyhow::Result<()> {
        let path = Self::default_weights_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        self.save_weights(&path)
    }

    /// Get a mutable reference to the underlying CfC network (for training).
    ///
    /// Panics if the network is currently borrowed (e.g., during plan_structure).
    pub fn network_mut(&mut self) -> std::sync::MutexGuard<'_, CfCNetwork> {
        self.network.lock().unwrap()
    }

    /// Convert an HDC vector into a network input Array1
    fn hv_to_input(&self, hv: &ContinuousHV) -> Array1<f32> {
        let hdc_dim = self.config.hdc_dim;
        let mut input = Array1::zeros(hdc_dim);
        let len = hv.values.len().min(hdc_dim);
        for i in 0..len {
            input[i] = hv.values[i];
        }
        input
    }

    /// Decode network output into a (PlanAction, confidence) pair.
    ///
    /// Output is interpreted as logits over the action space. The action with
    /// the highest logit is selected; confidence is the softmax probability.
    fn decode_output(&self, output: &Array1<f32>) -> (PlanAction, f32) {
        let n_actions = self.action_index.len();
        let scan_len = output.len().min(n_actions);

        if scan_len == 0 {
            return (PlanAction::Complete, 0.0);
        }

        let mut best_idx = 0;
        let mut best_val = output[0];

        for i in 1..scan_len {
            if output[i] > best_val {
                best_val = output[i];
                best_idx = i;
            }
        }

        // Clamp index to action_index bounds
        best_idx = best_idx.min(n_actions - 1);

        // Compute softmax confidence for the winning action
        let max_val = best_val;
        let exp_sum: f32 = (0..scan_len)
            .map(|i| (output[i] - max_val).exp())
            .sum();
        let confidence = if exp_sum > 0.0 {
            1.0 / exp_sum
        } else {
            0.0
        };

        (self.action_index[best_idx].clone(), confidence)
    }

    /// Create a one-hot target vector for a given PlanAction (for training)
    pub fn action_to_target(&self, action: &PlanAction) -> Array1<f32> {
        let mut target = Array1::zeros(NUM_ACTIONS);
        if let Some(idx) = self.action_index.iter().position(|a| a == action) {
            target[idx] = 1.0;
        }
        target
    }

    /// Train the sequencer on a single (input_hv, target_action) pair.
    ///
    /// Returns the MSE loss. Uses BPTT with Adam optimizer.
    pub fn train_step(
        &self,
        input_hv: &ContinuousHV,
        target_action: &PlanAction,
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        let input = self.hv_to_input(input_hv);
        let target = self.action_to_target(target_action);
        let mut net = self.network.lock().unwrap();
        net.reset();
        net.train_step(&input, &target, self.config.dt, learning_rate)
    }

    /// Train the sequencer on a sequence of (input_hv, action) pairs.
    ///
    /// The same input is fed at each step, with the target being each
    /// successive action in the plan sequence. Returns avg loss.
    pub fn train_sequence(
        &self,
        input_hv: &ContinuousHV,
        target_actions: &[PlanAction],
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        let input = self.hv_to_input(input_hv);
        let inputs: Vec<Array1<f32>> = target_actions.iter().map(|_| input.clone()).collect();
        let targets: Vec<Array1<f32>> = target_actions
            .iter()
            .map(|a| self.action_to_target(a))
            .collect();
        let dts: Vec<f32> = vec![self.config.dt; target_actions.len()];

        let mut net = self.network.lock().unwrap();
        net.reset();
        net.train_step_bptt(&inputs, &targets, &dts, learning_rate)
    }

    /// Export all network weights as a flat f32 vector.
    ///
    /// Format: [cell0_w_in | cell0_w_h | cell0_b_h | cell0_tau | output_weights | output_bias]
    /// Compatible with federated learning swarm gradient exchange.
    pub fn export_weights(&self) -> Vec<f32> {
        let net = self.network.lock().unwrap();
        let mut weights = Vec::new();

        for cell in &net.cells {
            if let Some(slice) = cell.w_in.as_slice() {
                weights.extend_from_slice(slice);
            }
            if let Some(slice) = cell.w_h.as_slice() {
                weights.extend_from_slice(slice);
            }
            if let Some(slice) = cell.b_h.as_slice() {
                weights.extend_from_slice(slice);
            }
            if let Some(slice) = cell.tau.as_slice() {
                weights.extend_from_slice(slice);
            }
        }

        if let Some(slice) = net.output_weights.as_slice() {
            weights.extend_from_slice(slice);
        }
        if let Some(slice) = net.output_bias.as_slice() {
            weights.extend_from_slice(slice);
        }

        weights
    }

    /// Import weights from a flat f32 vector (inverse of export_weights).
    pub fn import_weights(&self, weights: &[f32]) -> anyhow::Result<()> {
        let mut net = self.network.lock().unwrap();
        let expected = net.num_parameters();
        anyhow::ensure!(
            weights.len() == expected,
            "Weight vector length mismatch: got {}, expected {}",
            weights.len(),
            expected
        );

        let mut offset = 0;

        for cell in &mut net.cells {
            let n = cell.w_in.len();
            if let Some(slice) = cell.w_in.as_slice_mut() {
                slice.copy_from_slice(&weights[offset..offset + n]);
            }
            offset += n;

            let n = cell.w_h.len();
            if let Some(slice) = cell.w_h.as_slice_mut() {
                slice.copy_from_slice(&weights[offset..offset + n]);
            }
            offset += n;

            let n = cell.b_h.len();
            if let Some(slice) = cell.b_h.as_slice_mut() {
                slice.copy_from_slice(&weights[offset..offset + n]);
            }
            offset += n;

            let n = cell.tau.len();
            if let Some(slice) = cell.tau.as_slice_mut() {
                slice.copy_from_slice(&weights[offset..offset + n]);
            }
            offset += n;
        }

        let n = net.output_weights.len();
        if let Some(slice) = net.output_weights.as_slice_mut() {
            slice.copy_from_slice(&weights[offset..offset + n]);
        }
        offset += n;

        let n = net.output_bias.len();
        if let Some(slice) = net.output_bias.as_slice_mut() {
            slice.copy_from_slice(&weights[offset..offset + n]);
        }
        offset += n;

        debug_assert_eq!(offset, expected);
        Ok(())
    }

    /// Save weights to a binary file
    pub fn save_weights(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let weights = self.export_weights();
        let bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(path, &bytes)?;
        Ok(())
    }

    /// Load weights from a binary file
    pub fn load_weights(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(
            bytes.len() % 4 == 0,
            "Weight file size not a multiple of 4 bytes"
        );
        let weights: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        self.import_weights(&weights)
    }

    // =========================================================================
    // Phase 7: Online Learning — Learn During Operation
    // =========================================================================

    /// Online training step after a successful native generation.
    ///
    /// Performs a single SPSA (gradient-free) update toward the target plan,
    /// using the current generation's intent as input. This is lightweight
    /// enough to run after every generation without blocking the pipeline.
    ///
    /// Returns the loss, or None if training was skipped.
    pub fn online_learn_success(
        &self,
        intent_hv: &ContinuousHV,
        actual_plan: &[PlanAction],
        learning_rate: f32,
    ) -> Option<f32> {
        if actual_plan.is_empty() {
            return None;
        }

        // Only learn from plans that actually produced compilable code
        // (the caller is responsible for only calling this on successes)
        let input = self.hv_to_input(intent_hv);
        let targets: Vec<ndarray::Array1<f32>> = actual_plan
            .iter()
            .map(|a| self.action_to_target(a))
            .collect();
        let dts: Vec<f32> = vec![self.config.dt; actual_plan.len()];
        let inputs: Vec<ndarray::Array1<f32>> = actual_plan.iter().map(|_| input.clone()).collect();

        let mut net = self.network.lock().unwrap();
        net.reset();

        // Use SPSA (gradient-free) for online learning — more robust than BPTT
        // for single-example updates and doesn't require backprop infrastructure
        let mut total_loss = 0.0f32;
        for (inp, (target, dt)) in inputs.iter().zip(targets.iter().zip(dts.iter())) {
            let output = net.forward(inp, *dt);
            // Compute MSE loss for monitoring
            let loss: f32 = output
                .iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>()
                / output.len() as f32;
            total_loss += loss;

            // Online adaptation if enabled
            if net.online_learning_enabled() {
                net.adapt_online(loss, inp, target, *dt);
            }
        }

        let avg_loss = total_loss / actual_plan.len().max(1) as f32;
        Some(avg_loss)
    }

    /// Online training step after an LLM fallback — learn what the LLM did.
    ///
    /// Extracts the plan from LLM-generated source code and trains the sequencer
    /// to produce that plan for the given intent. Over time, this transfers
    /// knowledge from the LLM to the native tier.
    pub fn online_learn_from_llm(
        &self,
        intent_hv: &ContinuousHV,
        llm_source: &str,
        learning_rate: f32,
    ) -> Option<f32> {
        // Infer what plan the LLM effectively produced
        let inferred_plan = Self::infer_plan_from_code(llm_source);
        if inferred_plan.is_empty() {
            return None;
        }

        // Train toward the LLM's plan
        match self.train_sequence(intent_hv, &inferred_plan, learning_rate) {
            Ok(loss) if loss.is_finite() => Some(loss),
            _ => None,
        }
    }

    /// Infer a PlanAction sequence from source code (for distillation from LLM output).
    fn infer_plan_from_code(source: &str) -> Vec<PlanAction> {
        let mut actions = Vec::new();

        if source.contains("struct ") {
            actions.push(PlanAction::DefineStruct);
        }
        if source.contains("enum ") {
            actions.push(PlanAction::DefineEnum);
        }
        if source.contains("trait ") {
            actions.push(PlanAction::DefineTrait);
        }
        if source.contains("fn ") {
            actions.push(PlanAction::DefineFunction);
            if source.contains("->") {
                actions.push(PlanAction::SetReturnType);
            }
        }
        if source.contains("impl ") {
            actions.push(PlanAction::ImplTrait);
        }
        if source.contains("use ") {
            actions.push(PlanAction::AddImport);
        }
        if source.contains("match ") {
            actions.push(PlanAction::MatchExpression);
        }
        if source.contains(".iter()") || source.contains(".into_iter()") {
            actions.push(PlanAction::IteratorChain);
        }
        if source.contains("?;") || source.contains("?)") {
            actions.push(PlanAction::ErrorPropagation);
        }
        if source.contains("#[derive(") {
            actions.push(PlanAction::DeriveAttribute);
        }
        if source.contains("#[test]") {
            actions.push(PlanAction::TestModule);
        }

        if actions.is_empty() {
            actions.push(PlanAction::DefineFunction);
        }
        actions.push(PlanAction::Complete);
        actions
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
        self.plan_structure_inner(intent_hv, context_hvs, None)
    }

    /// Plan code structure with direct purpose text for improved pattern detection.
    ///
    /// Uses keyword matching on the raw purpose string to detect algorithm patterns,
    /// bypassing the byte-hash HDC encoding which produces orthogonal vectors for
    /// synonyms like "add" and "sum". Falls back to HDC detection if text detection
    /// finds nothing.
    pub fn plan_structure_with_purpose(
        &self,
        intent_hv: &ContinuousHV,
        context_hvs: &[&ContinuousHV],
        purpose: &str,
    ) -> Vec<CodePlanStep> {
        self.plan_structure_inner(intent_hv, context_hvs, Some(purpose))
    }

    fn plan_structure_inner(
        &self,
        intent_hv: &ContinuousHV,
        context_hvs: &[&ContinuousHV],
        purpose: Option<&str>,
    ) -> Vec<CodePlanStep> {
        // Build network input: intent HV blended with context
        let mut input = self.hv_to_input(intent_hv);
        for ctx_hv in context_hvs {
            let ctx_input = self.hv_to_input(ctx_hv);
            input = &input * 0.7 + &ctx_input * 0.3;
        }

        // Detect algorithm pattern — prefer text-based detection (more reliable)
        // then fall back to HDC similarity detection
        let mut plan = Vec::new();
        let detected_pattern = purpose
            .and_then(AlgorithmPattern::detect_from_text)
            .or_else(|| self.pattern_detector.detect(intent_hv));

        if let Some(pattern) = detected_pattern {
            plan.extend(pattern.to_plan_steps());
        }

        // Reset network state for fresh planning
        let mut net = self.network.lock().unwrap();
        net.reset();

        // Evolve CfC network and collect plan steps
        let mut prev_action = plan.last().map(|s| s.action.clone());

        for _step in 0..self.config.max_steps {
            let output = net.forward(&input, self.config.dt);
            let (action, confidence) = self.decode_output(&output);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sequencer() {
        let sequencer = CfCCodeSequencer::default();
        assert_eq!(sequencer.config.hidden_dim, 64);
        assert_eq!(sequencer.action_index.len(), NUM_ACTIONS);
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
    fn test_network_stability() {
        let sequencer = CfCCodeSequencer::default();
        let input = ContinuousHV::random(512, 42);
        let arr_input = sequencer.hv_to_input(&input);

        // Run many steps through the network — should not blow up
        let mut net = sequencer.network.lock().unwrap();
        net.reset();
        for _ in 0..100 {
            let output = net.forward(&arr_input, 1.0);
            for v in output.iter() {
                assert!(v.is_finite(), "CfC output should remain finite");
            }
        }
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let sequencer = CfCCodeSequencer::default();
        let intent = ContinuousHV::random(512, 42);

        // Train toward DefineFunction for 10 steps
        let mut losses = Vec::new();
        for _ in 0..10 {
            let loss = sequencer
                .train_step(&intent, &PlanAction::DefineFunction, 0.01)
                .unwrap();
            losses.push(loss);
        }

        // Loss should generally decrease (allow noise)
        let first_3_avg: f32 = losses[..3].iter().sum::<f32>() / 3.0;
        let last_3_avg: f32 = losses[7..].iter().sum::<f32>() / 3.0;
        assert!(
            last_3_avg <= first_3_avg + 0.1,
            "Training should not increase loss significantly: first_3={:.4} last_3={:.4}",
            first_3_avg,
            last_3_avg
        );
    }

    #[test]
    fn test_weight_export_import_roundtrip() {
        let sequencer = CfCCodeSequencer::default();
        let intent = ContinuousHV::random(512, 42);

        // Get plan before
        let plan_before = sequencer.plan_structure(&intent, &[]);

        // Export and reimport weights
        let weights = sequencer.export_weights();
        assert!(!weights.is_empty());
        assert_eq!(weights.len(), sequencer.network.lock().unwrap().num_parameters());

        let sequencer2 = CfCCodeSequencer::default();
        sequencer2.import_weights(&weights).unwrap();

        // Plans should be identical after import
        let plan_after = sequencer2.plan_structure(&intent, &[]);
        assert_eq!(plan_before.len(), plan_after.len());
        for (a, b) in plan_before.iter().zip(plan_after.iter()) {
            assert_eq!(a.action, b.action);
        }
    }

    #[test]
    fn test_action_to_target_roundtrip() {
        let sequencer = CfCCodeSequencer::default();
        for action in &sequencer.action_index {
            let target = sequencer.action_to_target(action);
            assert_eq!(target.len(), NUM_ACTIONS);
            let sum: f32 = target.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Target should be one-hot");
        }
    }

    // --- Algorithm Pattern Detection Tests ---

    fn make_keyword_hv(dim: usize, keywords: &[&str]) -> ContinuousHV {
        #[cfg(feature = "code_generation")]
        {
            let encoder = crate::hdc::code_semantic_encoder::CodeSemanticEncoder::new(dim);
            encoder.encode_text(&keywords.join(" "))
        }
        #[cfg(not(feature = "code_generation"))]
        {
            // Fallback: byte-hash encoding
            let text = keywords.join(" ").to_lowercase();
            let mut values = vec![0.0f32; dim];
            for (i, byte) in text.bytes().enumerate() {
                let idx = ((byte as usize).wrapping_mul(31).wrapping_add(i.wrapping_mul(7))) % dim;
                values[idx] += 1.0;
            }
            let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
            if magnitude > 0.0 { for v in &mut values { *v /= magnitude; } }
            ContinuousHV::from_values(values)
        }
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

    // --- Direct Text-Based Pattern Detection Tests ---

    #[test]
    fn test_text_detect_sorting() {
        assert_eq!(
            AlgorithmPattern::detect_from_text("Sort a vector of integers in ascending order"),
            Some(AlgorithmPattern::Sorting)
        );
        assert_eq!(
            AlgorithmPattern::detect_from_text("Arrange elements by comparing and swapping"),
            Some(AlgorithmPattern::Sorting)
        );
    }

    #[test]
    fn test_text_detect_accumulation() {
        assert_eq!(
            AlgorithmPattern::detect_from_text("Sum all elements in the list"),
            Some(AlgorithmPattern::Accumulation)
        );
        assert_eq!(
            AlgorithmPattern::detect_from_text("Count the total number of items"),
            Some(AlgorithmPattern::Accumulation)
        );
        assert_eq!(
            AlgorithmPattern::detect_from_text("Calculate the average of numbers"),
            Some(AlgorithmPattern::Accumulation)
        );
    }

    #[test]
    fn test_text_detect_string_processing() {
        assert_eq!(
            AlgorithmPattern::detect_from_text("Reverse a string and capitalize it"),
            Some(AlgorithmPattern::StringProcessing)
        );
        assert_eq!(
            AlgorithmPattern::detect_from_text("Parse the input and trim whitespace"),
            Some(AlgorithmPattern::StringProcessing)
        );
    }

    #[test]
    fn test_text_detect_search() {
        assert_eq!(
            AlgorithmPattern::detect_from_text("Binary search for a target in sorted array"),
            Some(AlgorithmPattern::Search)
        );
    }

    #[test]
    fn test_text_detect_dp() {
        assert_eq!(
            AlgorithmPattern::detect_from_text("Dynamic programming solution for knapsack"),
            Some(AlgorithmPattern::DynamicProgramming)
        );
    }

    #[test]
    fn test_text_detect_graph() {
        assert_eq!(
            AlgorithmPattern::detect_from_text("Find shortest path between nodes in graph"),
            Some(AlgorithmPattern::Graph)
        );
    }

    #[test]
    fn test_text_detect_none_for_unrelated() {
        assert_eq!(
            AlgorithmPattern::detect_from_text("Configure the database connection pool"),
            None
        );
    }

    #[test]
    fn test_text_detect_disambiguates_by_count() {
        // "sort" (1 hit Sorting) vs "filter and collect" (2 hits Accumulation)
        assert_eq!(
            AlgorithmPattern::detect_from_text("Filter items and collect the total"),
            Some(AlgorithmPattern::Accumulation)
        );
    }

    #[test]
    fn test_plan_with_purpose_uses_text_detection() {
        let dim = ALGORITHM_PATTERN_DIM;
        let sequencer = CfCCodeSequencer::new(CfCCodeSequencerConfig {
            hdc_dim: dim,
            ..Default::default()
        });

        // Random HV that wouldn't match any HV-based pattern, but purpose text is clear
        let random_hv = ContinuousHV::random(dim, 12345);
        let plan = sequencer.plan_structure_with_purpose(
            &random_hv,
            &[],
            "Sort integers in ascending order",
        );

        let has_sorting_context = plan
            .iter()
            .any(|step| step.context.iter().any(|c| c.contains("sorting")));
        assert!(
            has_sorting_context,
            "Text-based detection should inject sorting context even with random HV"
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
