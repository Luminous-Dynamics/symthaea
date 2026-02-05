//! # Integration Layer: Unified Consciousness Routing Hub
//!
//! This integration layer combines all five revolutionary routing paradigms into
//! a single, coherent system that can dynamically select, blend, and orchestrate
//! multiple routing strategies based on the current consciousness state.
//!
//! ## Router Hierarchy
//!
//! 1. CausalValidatedRouter - Foundation: causal intervention validation
//! 2. InformationGeometricRouter - Riemannian manifold navigation
//! 3. TopologicalConsciousnessRouter - Persistent homology features
//! 4. QuantumCoherenceRouter - Quantum amplitude dynamics
//! 5. ActiveInferenceRouter - Free energy minimization
//!
//! ## Features
//!
//! - Unified interface for all routers
//! - Dynamic router selection based on context
//! - Ensemble routing with weighted combination
//! - Cross-router information sharing
//! - Performance tracking across all strategies

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use serde::{Serialize, Deserialize};

use crate::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};
use crate::consciousness::primitive_reasoning::{AdaptivePrimitiveSelector, TaskType};

use super::world_model::LatentConsciousnessState;
use super::routers::{
    RoutingStrategy,
    CausalValidatedRouter, CausalValidatedConfig,
    InformationGeometricRouter, GeometricRouterConfig,
    TopologicalConsciousnessRouter, TopologicalRouterConfig,
    QuantumCoherenceRouter, QuantumRouterConfig,
    ActiveInferenceRouter, ActiveInferenceConfig,
};

/// Mode for selecting which router(s) to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingMode {
    /// Use only one specific router
    Single(RouterType),
    /// Use the best router based on recent performance
    Adaptive,
    /// Combine outputs from all routers
    Ensemble,
    /// Hierarchical: use simpler routers first, escalate if needed
    Hierarchical,
    /// Use quantum coherence to superpose router outputs
    QuantumEnsemble,
}

/// Types of routers available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RouterType {
    Causal,
    Geometric,
    Topological,
    Quantum,
    ActiveInference,
}

impl RouterType {
    pub fn all() -> Vec<RouterType> {
        vec![
            RouterType::Causal,
            RouterType::Geometric,
            RouterType::Topological,
            RouterType::Quantum,
            RouterType::ActiveInference,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            RouterType::Causal => "Causal",
            RouterType::Geometric => "Geometric",
            RouterType::Topological => "Topological",
            RouterType::Quantum => "Quantum",
            RouterType::ActiveInference => "ActiveInference",
        }
    }

    pub fn complexity(&self) -> usize {
        match self {
            RouterType::Causal => 1,
            RouterType::Geometric => 2,
            RouterType::Topological => 3,
            RouterType::Quantum => 4,
            RouterType::ActiveInference => 5,
        }
    }
}

/// Unified routing decision from the hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedRoutingDecision {
    /// The selected strategy
    pub strategy: RoutingStrategy,
    /// Confidence in the decision
    pub confidence: f64,
    /// Which router(s) contributed
    pub contributors: Vec<RouterType>,
    /// Contribution weights
    pub weights: Vec<f64>,
    /// Reasoning for the decision
    pub reasoning: String,
    /// Alternative strategies considered
    pub alternatives: Vec<(RoutingStrategy, f64)>,
    /// Decision latency in microseconds
    pub latency_us: u64,
}

/// Performance tracking for each router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterPerformance {
    pub router_type: RouterType,
    pub decisions: usize,
    pub successful_outcomes: usize,
    pub average_latency_us: f64,
    pub average_confidence: f64,
    pub recent_scores: VecDeque<f64>,
}

impl RouterPerformance {
    pub fn new(router_type: RouterType) -> Self {
        Self {
            router_type,
            decisions: 0,
            successful_outcomes: 0,
            average_latency_us: 0.0,
            average_confidence: 0.0,
            recent_scores: VecDeque::with_capacity(100),
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.decisions == 0 {
            0.5 // Prior
        } else {
            self.successful_outcomes as f64 / self.decisions as f64
        }
    }

    pub fn recent_average(&self) -> f64 {
        if self.recent_scores.is_empty() {
            0.5
        } else {
            self.recent_scores.iter().sum::<f64>() / self.recent_scores.len() as f64
        }
    }

    pub fn record(&mut self, score: f64, latency_us: u64, confidence: f64) {
        self.decisions += 1;
        if score > 0.5 {
            self.successful_outcomes += 1;
        }

        // Running average for latency
        let n = self.decisions as f64;
        self.average_latency_us = (self.average_latency_us * (n - 1.0) + latency_us as f64) / n;
        self.average_confidence = (self.average_confidence * (n - 1.0) + confidence) / n;

        // Recent scores with window
        if self.recent_scores.len() >= 100 {
            self.recent_scores.pop_front();
        }
        self.recent_scores.push_back(score);
    }
}

/// Configuration for the routing hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingHubConfig {
    pub mode: RoutingMode,
    pub exploration_rate: f64,
    pub ensemble_temperature: f64,
    pub hierarchical_threshold: f64,
    pub enable_cross_router_learning: bool,
    pub track_performance: bool,
}

impl Default for RoutingHubConfig {
    fn default() -> Self {
        Self {
            mode: RoutingMode::Adaptive,
            exploration_rate: 0.1,
            ensemble_temperature: 1.0,
            hierarchical_threshold: 0.7,
            enable_cross_router_learning: true,
            track_performance: true,
        }
    }
}

/// Helper to convert routing cost to confidence (inverse relationship)
fn cost_to_confidence(cost: f64) -> f64 {
    1.0 / (1.0 + cost)
}

// =============================================================================
// PRIMITIVE ROUTING CONTEXT
// =============================================================================

/// Primitive context for routing decisions
///
/// Provides primitive-level insights to inform routing strategy selection.
/// This bridges the primitive consciousness system with routing decisions.
#[derive(Debug, Clone)]
pub struct PrimitiveRoutingContext {
    /// Dominant primitive tier based on recent usage
    pub dominant_tier: PrimitiveTier,
    /// Tier distribution (tier -> usage proportion)
    pub tier_distribution: HashMap<PrimitiveTier, f64>,
    /// Inferred task type based on primitive usage patterns
    pub inferred_task: TaskType,
    /// Primitive-based complexity estimate (0.0 = simple, 1.0 = complex)
    pub complexity_estimate: f64,
    /// Recommended router based on primitive patterns
    pub recommended_router: Option<RouterType>,
}

impl PrimitiveRoutingContext {
    /// Create primitive context from selector statistics
    pub fn from_selector(selector: &AdaptivePrimitiveSelector) -> Self {
        // Use cached global instance for O(1) access
        let prim_system = PrimitiveSystem::global();

        // Aggregate stats by tier
        let mut tier_counts: HashMap<PrimitiveTier, (usize, f64)> = HashMap::new();
        let mut total_uses = 0usize;

        for ((prim_name, _task), stats) in selector.all_stats() {
            if let Some(primitive) = prim_system.get(prim_name) {
                let entry = tier_counts.entry(primitive.tier).or_insert((0, 0.0));
                entry.0 += stats.use_count;
                entry.1 += stats.total_phi;
                total_uses += stats.use_count;
            }
        }

        // Compute tier distribution
        let tier_distribution: HashMap<PrimitiveTier, f64> = tier_counts.iter()
            .map(|(tier, (count, _))| {
                let proportion = if total_uses > 0 {
                    *count as f64 / total_uses as f64
                } else {
                    0.0
                };
                (*tier, proportion)
            })
            .collect();

        // Find dominant tier
        let dominant_tier = tier_distribution.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(t, _)| *t)
            .unwrap_or(PrimitiveTier::Physical);

        // Infer task type from dominant tier
        let inferred_task = match dominant_tier {
            PrimitiveTier::Mathematical => TaskType::Logical,
            PrimitiveTier::Physical => TaskType::Causal,
            PrimitiveTier::Geometric => TaskType::Spatial,
            PrimitiveTier::Strategic => TaskType::Social,
            PrimitiveTier::MetaCognitive => TaskType::MetaCognitive,
            PrimitiveTier::Temporal => TaskType::Memory,
            PrimitiveTier::Compositional => TaskType::Creative,
            PrimitiveTier::Consciousness => TaskType::MetaCognitive,
            PrimitiveTier::NSM => TaskType::General,
        };

        // Estimate complexity based on tier diversity
        let active_tiers = tier_distribution.values().filter(|&&v| v > 0.05).count();
        let complexity_estimate = (active_tiers as f64 / 9.0).min(1.0);

        // Recommend router based on patterns
        let recommended_router = match dominant_tier {
            PrimitiveTier::Mathematical | PrimitiveTier::Physical => Some(RouterType::Causal),
            PrimitiveTier::Geometric => Some(RouterType::Geometric),
            PrimitiveTier::Strategic | PrimitiveTier::MetaCognitive => Some(RouterType::ActiveInference),
            PrimitiveTier::Compositional => Some(RouterType::Topological),
            PrimitiveTier::Consciousness => Some(RouterType::Quantum),
            _ => None,
        };

        Self {
            dominant_tier,
            tier_distribution,
            inferred_task,
            complexity_estimate,
            recommended_router,
        }
    }

    /// Create empty context (no primitive data)
    pub fn empty() -> Self {
        Self {
            dominant_tier: PrimitiveTier::Physical,
            tier_distribution: HashMap::new(),
            inferred_task: TaskType::General,
            complexity_estimate: 0.5,
            recommended_router: None,
        }
    }
}

// =============================================================================
// CONSCIOUSNESS ROUTING HUB
// =============================================================================

/// Unified Consciousness Routing Hub
/// Orchestrates all five revolutionary routing paradigms
pub struct ConsciousnessRoutingHub {
    /// Base causal router
    causal_router: CausalValidatedRouter,
    /// Geometric router (wraps causal)
    geometric_router: InformationGeometricRouter,
    /// Topological router (wraps geometric)
    topological_router: TopologicalConsciousnessRouter,
    /// Quantum router (wraps topological)
    quantum_router: QuantumCoherenceRouter,
    /// Active inference router (wraps quantum)
    active_inference_router: ActiveInferenceRouter,
    /// Configuration
    config: RoutingHubConfig,
    /// Performance tracking
    performance: HashMap<RouterType, RouterPerformance>,
    /// Current state
    current_state: Option<LatentConsciousnessState>,
    /// Decision history
    history: VecDeque<UnifiedRoutingDecision>,
    /// Total decisions
    total_decisions: usize,
    /// Primitive routing context (provides task-aware router selection)
    primitive_context: Option<PrimitiveRoutingContext>,
}

impl ConsciousnessRoutingHub {
    pub fn new(config: RoutingHubConfig) -> Self {
        let mut performance = HashMap::new();
        for rt in RouterType::all() {
            performance.insert(rt, RouterPerformance::new(rt));
        }

        Self {
            causal_router: CausalValidatedRouter::new(CausalValidatedConfig::default()),
            geometric_router: InformationGeometricRouter::new(GeometricRouterConfig::default()),
            topological_router: TopologicalConsciousnessRouter::new(TopologicalRouterConfig::default()),
            quantum_router: QuantumCoherenceRouter::new(QuantumRouterConfig::default()),
            active_inference_router: ActiveInferenceRouter::new(ActiveInferenceConfig::default()),
            config,
            performance,
            current_state: None,
            history: VecDeque::with_capacity(1000),
            total_decisions: 0,
            primitive_context: None,
        }
    }

    /// Set primitive context from adaptive selector
    ///
    /// This method updates the routing hub's understanding of current
    /// primitive usage patterns, enabling task-aware router selection.
    pub fn set_primitive_context(&mut self, selector: &AdaptivePrimitiveSelector) {
        self.primitive_context = Some(PrimitiveRoutingContext::from_selector(selector));
    }

    /// Set primitive context directly
    pub fn set_primitive_context_direct(&mut self, context: PrimitiveRoutingContext) {
        self.primitive_context = Some(context);
    }

    /// Get current primitive context
    pub fn primitive_context(&self) -> Option<&PrimitiveRoutingContext> {
        self.primitive_context.as_ref()
    }

    /// Get primitive-recommended router
    ///
    /// Returns the router that primitive usage patterns suggest would
    /// perform best for the current task.
    pub fn primitive_recommended_router(&self) -> Option<RouterType> {
        self.primitive_context.as_ref().and_then(|ctx| ctx.recommended_router)
    }

    /// Observe a new consciousness state
    pub fn observe(&mut self, state: &LatentConsciousnessState) {
        self.current_state = Some(state.clone());

        // Propagate to all routers
        self.geometric_router.observe_state(state);
        self.topological_router.observe_state(state);
        self.quantum_router.observe_state(state);
        self.active_inference_router.observe_state(state);
    }

    /// Route using the configured mode
    pub fn route(&mut self, target: &LatentConsciousnessState) -> UnifiedRoutingDecision {
        let start = std::time::Instant::now();
        self.total_decisions += 1;

        // Ensure state is observed
        if self.current_state.is_none() {
            self.observe(target);
        }

        let decision = match self.config.mode {
            RoutingMode::Single(router_type) => self.route_single(router_type, target),
            RoutingMode::Adaptive => self.route_adaptive(target),
            RoutingMode::Ensemble => self.route_ensemble(target),
            RoutingMode::Hierarchical => self.route_hierarchical(target),
            RoutingMode::QuantumEnsemble => self.route_quantum_ensemble(target),
        };

        let mut decision = decision;
        decision.latency_us = start.elapsed().as_micros() as u64;

        // Track history
        if self.history.len() >= 1000 {
            self.history.pop_front();
        }
        self.history.push_back(decision.clone());

        decision
    }

    /// Route using a single specific router
    fn route_single(&mut self, router_type: RouterType, target: &LatentConsciousnessState) -> UnifiedRoutingDecision {
        let (strategy, confidence) = match router_type {
            RouterType::Causal => {
                let validated = self.causal_router.route_validated(target);
                (validated.strategy, validated.confidence)
            }
            RouterType::Geometric => {
                let decision = self.geometric_router.route(target);
                (decision.strategy, cost_to_confidence(decision.routing_cost))
            }
            RouterType::Topological => {
                let decision = self.topological_router.route(target);
                // Topological uses complexity score inversely
                let conf = if decision.transition_detected { 0.5 } else { 0.8 };
                (decision.strategy, conf)
            }
            RouterType::Quantum => {
                let decision = self.quantum_router.route(target);
                (decision.strategy, decision.probability)
            }
            RouterType::ActiveInference => {
                let decision = self.active_inference_router.route(target);
                (decision.strategy, decision.confidence)
            }
        };

        UnifiedRoutingDecision {
            strategy,
            confidence,
            contributors: vec![router_type],
            weights: vec![1.0],
            reasoning: format!("Single router: {}", router_type.name()),
            alternatives: Vec::new(),
            latency_us: 0,
        }
    }

    /// Adaptively select the best router based on performance
    fn route_adaptive(&mut self, target: &LatentConsciousnessState) -> UnifiedRoutingDecision {
        // Exploration vs exploitation
        if rand::random::<f64>() < self.config.exploration_rate {
            // Explore: random router
            let idx = (rand::random::<f64>() * 5.0) as usize;
            let router_type = RouterType::all()[idx.min(4)];
            return self.route_single(router_type, target);
        }

        // Exploit: use best performing router
        let best_router = RouterType::all()
            .into_iter()
            .max_by(|a, b| {
                let perf_a = self.performance.get(a).map(|p| p.recent_average()).unwrap_or(0.5);
                let perf_b = self.performance.get(b).map(|p| p.recent_average()).unwrap_or(0.5);
                perf_a.partial_cmp(&perf_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(RouterType::ActiveInference);

        let mut decision = self.route_single(best_router, target);
        decision.reasoning = format!("Adaptive selection: {} (best recent performance)", best_router.name());
        decision
    }

    /// Combine outputs from all routers with weighted voting
    fn route_ensemble(&mut self, target: &LatentConsciousnessState) -> UnifiedRoutingDecision {
        // Get decisions from all routers
        let geo_dec = self.geometric_router.route(target);
        let topo_dec = self.topological_router.route(target);
        let quantum_dec = self.quantum_router.route(target);
        let ai_dec = self.active_inference_router.route(target);

        let decisions: Vec<(RouterType, RoutingStrategy, f64)> = vec![
            (RouterType::Geometric, geo_dec.strategy, cost_to_confidence(geo_dec.routing_cost)),
            (RouterType::Topological, topo_dec.strategy, if topo_dec.transition_detected { 0.5 } else { 0.8 }),
            (RouterType::Quantum, quantum_dec.strategy, quantum_dec.probability),
            (RouterType::ActiveInference, ai_dec.strategy, ai_dec.confidence),
        ];

        // Weight by performance and confidence
        let mut strategy_votes: HashMap<RoutingStrategy, f64> = HashMap::new();
        let mut total_weight = 0.0;

        for (rt, strategy, confidence) in &decisions {
            let perf = self.performance.get(rt).map(|p| p.success_rate()).unwrap_or(0.5);
            let weight = confidence * perf;
            *strategy_votes.entry(*strategy).or_insert(0.0) += weight;
            total_weight += weight;
        }

        // Find winning strategy
        let (winning_strategy, winning_votes) = strategy_votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((RoutingStrategy::StandardProcessing, 0.5));

        // Calculate contributors and weights
        let contributors: Vec<RouterType> = decisions
            .iter()
            .filter(|(_, s, _)| *s == winning_strategy)
            .map(|(rt, _, _)| *rt)
            .collect();

        let weights: Vec<f64> = decisions
            .iter()
            .filter(|(_, s, _)| *s == winning_strategy)
            .map(|(rt, _, c)| {
                let perf = self.performance.get(rt).map(|p| p.success_rate()).unwrap_or(0.5);
                c * perf
            })
            .collect();

        let confidence = if total_weight > 0.0 {
            winning_votes / total_weight
        } else {
            0.5
        };

        UnifiedRoutingDecision {
            strategy: winning_strategy,
            confidence,
            contributors,
            weights,
            reasoning: format!("Ensemble voting: {} routers agreed", decisions.iter().filter(|(_, s, _)| *s == winning_strategy).count()),
            alternatives: decisions.iter().filter(|(_, s, _)| *s != winning_strategy).map(|(_, s, c)| (*s, *c)).collect(),
            latency_us: 0,
        }
    }

    /// Hierarchical routing: start simple, escalate if confidence is low
    fn route_hierarchical(&mut self, target: &LatentConsciousnessState) -> UnifiedRoutingDecision {
        let threshold = self.config.hierarchical_threshold;

        // Level 1: Causal
        let causal_decision = self.causal_router.route_validated(target);
        let causal_confidence = causal_decision.confidence;

        if causal_confidence >= threshold {
            return UnifiedRoutingDecision {
                strategy: causal_decision.strategy,
                confidence: causal_confidence,
                contributors: vec![RouterType::Causal],
                weights: vec![1.0],
                reasoning: "Hierarchical: Causal sufficient".to_string(),
                alternatives: Vec::new(),
                latency_us: 0,
            };
        }

        // Level 2: Geometric
        let geo_decision = self.geometric_router.route(target);
        let geo_confidence = cost_to_confidence(geo_decision.routing_cost);
        if geo_confidence >= threshold {
            return UnifiedRoutingDecision {
                strategy: geo_decision.strategy,
                confidence: geo_confidence,
                contributors: vec![RouterType::Causal, RouterType::Geometric],
                weights: vec![0.3, 0.7],
                reasoning: "Hierarchical: Geometric sufficient".to_string(),
                alternatives: Vec::new(),
                latency_us: 0,
            };
        }

        // Level 3: Topological
        let topo_decision = self.topological_router.route(target);
        let topo_confidence = if topo_decision.transition_detected { 0.5 } else { 0.8 };
        if topo_confidence >= threshold {
            return UnifiedRoutingDecision {
                strategy: topo_decision.strategy,
                confidence: topo_confidence,
                contributors: vec![RouterType::Causal, RouterType::Geometric, RouterType::Topological],
                weights: vec![0.2, 0.3, 0.5],
                reasoning: "Hierarchical: Topological sufficient".to_string(),
                alternatives: Vec::new(),
                latency_us: 0,
            };
        }

        // Level 4: Quantum
        let quantum_decision = self.quantum_router.route(target);
        let quantum_confidence = quantum_decision.probability;
        if quantum_confidence >= threshold {
            return UnifiedRoutingDecision {
                strategy: quantum_decision.strategy,
                confidence: quantum_confidence,
                contributors: vec![RouterType::Causal, RouterType::Geometric, RouterType::Topological, RouterType::Quantum],
                weights: vec![0.1, 0.2, 0.3, 0.4],
                reasoning: "Hierarchical: Quantum sufficient".to_string(),
                alternatives: Vec::new(),
                latency_us: 0,
            };
        }

        // Level 5: Active Inference (final authority)
        let ai_decision = self.active_inference_router.route(target);
        UnifiedRoutingDecision {
            strategy: ai_decision.strategy,
            confidence: ai_decision.confidence,
            contributors: RouterType::all(),
            weights: vec![0.1, 0.15, 0.2, 0.25, 0.3],
            reasoning: "Hierarchical: Full escalation to Active Inference".to_string(),
            alternatives: vec![
                (geo_decision.strategy, geo_confidence),
                (topo_decision.strategy, topo_confidence),
                (quantum_decision.strategy, quantum_confidence),
            ],
            latency_us: 0,
        }
    }

    /// Quantum ensemble: superpose router outputs using amplitude-like weighting
    fn route_quantum_ensemble(&mut self, target: &LatentConsciousnessState) -> UnifiedRoutingDecision {
        // Get decisions from each router
        let geo_decision = self.geometric_router.route(target);
        let topo_decision = self.topological_router.route(target);
        let quantum_decision = self.quantum_router.route(target);
        let ai_decision = self.active_inference_router.route(target);

        // Extract confidences
        let geo_conf = cost_to_confidence(geo_decision.routing_cost);
        let topo_conf = if topo_decision.transition_detected { 0.5 } else { 0.8 };
        let quantum_conf = quantum_decision.probability;
        let ai_conf = ai_decision.confidence;

        // Compute phase-weighted amplitudes (simulate interference)
        let phases = [0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0];
        let confs = [geo_conf, topo_conf, quantum_conf, ai_conf];
        let strategies = [geo_decision.strategy, topo_decision.strategy, quantum_decision.strategy, ai_decision.strategy];

        // Group by strategy and compute interference
        let mut strategy_amplitudes: HashMap<RoutingStrategy, (f64, f64)> = HashMap::new();
        for i in 0..4 {
            let amplitude = confs[i].sqrt();
            let real = amplitude * phases[i].cos();
            let imag = amplitude * phases[i].sin();
            let entry = strategy_amplitudes.entry(strategies[i]).or_insert((0.0, 0.0));
            entry.0 += real;
            entry.1 += imag;
        }

        // Calculate probabilities
        let total_prob: f64 = strategy_amplitudes.values()
            .map(|(r, i)| r * r + i * i)
            .sum();

        let mut strategy_probs: Vec<(RoutingStrategy, f64)> = strategy_amplitudes
            .into_iter()
            .map(|(s, (r, i))| (s, (r * r + i * i) / total_prob.max(0.001)))
            .collect();
        strategy_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (winning_strategy, winning_prob) = strategy_probs.first()
            .copied()
            .unwrap_or((RoutingStrategy::StandardProcessing, 0.5));

        let alternatives = if strategy_probs.len() > 1 {
            strategy_probs[1..].to_vec()
        } else {
            Vec::new()
        };

        // Determine contributors
        let contributors: Vec<RouterType> = vec![
            if geo_decision.strategy == winning_strategy { Some(RouterType::Geometric) } else { None },
            if topo_decision.strategy == winning_strategy { Some(RouterType::Topological) } else { None },
            if quantum_decision.strategy == winning_strategy { Some(RouterType::Quantum) } else { None },
            if ai_decision.strategy == winning_strategy { Some(RouterType::ActiveInference) } else { None },
        ].into_iter().flatten().collect();

        UnifiedRoutingDecision {
            strategy: winning_strategy,
            confidence: winning_prob,
            contributors,
            weights: vec![0.25; 4],
            reasoning: format!("Quantum ensemble: probability {:.2}%", winning_prob * 100.0),
            alternatives,
            latency_us: 0,
        }
    }

    /// Record outcome for learning
    pub fn record_outcome(&mut self, decision: &UnifiedRoutingDecision, score: f64) {
        for (i, router_type) in decision.contributors.iter().enumerate() {
            let weight = decision.weights.get(i).copied().unwrap_or(1.0);
            if let Some(perf) = self.performance.get_mut(router_type) {
                perf.record(score * weight, decision.latency_us, decision.confidence);
            }
        }
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> HashMap<RouterType, (f64, f64)> {
        self.performance
            .iter()
            .map(|(rt, perf)| (*rt, (perf.success_rate(), perf.average_latency_us)))
            .collect()
    }

    /// Reset all routers
    pub fn reset(&mut self) {
        // Note: individual routers don't have a general reset,
        // but we reset our tracking state
        self.history.clear();
        self.total_decisions = 0;
        for (_, perf) in self.performance.iter_mut() {
            perf.decisions = 0;
            perf.successful_outcomes = 0;
            perf.recent_scores.clear();
        }
    }

    /// Get current mode
    pub fn mode(&self) -> RoutingMode {
        self.config.mode
    }

    /// Set routing mode
    pub fn set_mode(&mut self, mode: RoutingMode) {
        self.config.mode = mode;
    }

    /// Get total decisions made
    pub fn total_decisions(&self) -> usize {
        self.total_decisions
    }
}
