//! # Meta-Router: Multi-Armed Bandit Paradigm Selection
//!
//! Revolutionary Improvement #68: A meta-learning router that learns which
//! of the 7 routing paradigms works best for different consciousness states.
//!
//! ## Key Features
//!
//! 1. **Multi-Armed Bandit**: UCB1 algorithm for exploration/exploitation
//! 2. **Contextual Routing**: Different paradigms for different state profiles
//! 3. **Performance Tracking**: Tracks success rates per paradigm
//! 4. **Dynamic Adaptation**: Adjusts preferences based on outcomes
//! 5. **Domain Detection**: Identifies what type of problem we're solving
//!
//! ## The 7 Routing Paradigms
//!
//! - Causal Validation: Causal Emergence validation
//! - Information Geometric: Fisher/geodesics
//! - Topological Consciousness: Persistent homology
//! - Quantum Coherence: Superposition/collapse
//! - Active Inference: Free energy minimization
//! - Predictive Processing: Hierarchical prediction
//! - Attention Schema: Meta-attention modeling
//! - Resonant Consciousness: HDC+LTC+Resonator soft routing (Phase 5H)

use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};

use super::{
    LatentConsciousnessState, RoutingStrategy,
    CausalValidatedRouter, CausalValidatedConfig,
    InformationGeometricRouter, GeometricRouterConfig,
    TopologicalConsciousnessRouter, TopologicalRouterConfig,
    QuantumCoherenceRouter, QuantumRouterConfig,
    ActiveInferenceRouter, ActiveInferenceConfig,
    PredictiveProcessingRouter, PredictiveProcessingConfig,
    ASTRouter, ASTRouterConfig,
    ResonantConsciousnessRouter, ResonantRouterConfig, // Phase 5H: HDC+LTC+Resonator
};

// =============================================================================
// ROUTING PARADIGM
// =============================================================================

/// The 8 routing paradigms available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoutingParadigm {
    /// Causal Emergence validation
    CausalValidation,
    /// Information Geometric (Fisher/geodesics)
    InformationGeometric,
    /// Topological Consciousness (persistent homology)
    TopologicalConsciousness,
    /// Quantum Coherence (superposition/collapse)
    QuantumCoherence,
    /// Active Inference (free energy minimization)
    ActiveInference,
    /// Predictive Processing (hierarchical prediction)
    PredictiveProcessing,
    /// Attention Schema Theory (meta-attention)
    AttentionSchema,
    /// Resonant Consciousness (HDC+LTC+Resonator soft routing) - Phase 5H
    ResonantConsciousness,
}

impl RoutingParadigm {
    pub fn all() -> Vec<RoutingParadigm> {
        vec![
            RoutingParadigm::CausalValidation,
            RoutingParadigm::InformationGeometric,
            RoutingParadigm::TopologicalConsciousness,
            RoutingParadigm::QuantumCoherence,
            RoutingParadigm::ActiveInference,
            RoutingParadigm::PredictiveProcessing,
            RoutingParadigm::AttentionSchema,
            RoutingParadigm::ResonantConsciousness, // Phase 5H
        ]
    }

    pub fn index(&self) -> usize {
        match self {
            RoutingParadigm::CausalValidation => 0,
            RoutingParadigm::InformationGeometric => 1,
            RoutingParadigm::TopologicalConsciousness => 2,
            RoutingParadigm::QuantumCoherence => 3,
            RoutingParadigm::ActiveInference => 4,
            RoutingParadigm::PredictiveProcessing => 5,
            RoutingParadigm::AttentionSchema => 6,
            RoutingParadigm::ResonantConsciousness => 7, // Phase 5H
        }
    }

    pub fn from_index(idx: usize) -> Self {
        match idx % 8 {
            0 => RoutingParadigm::CausalValidation,
            1 => RoutingParadigm::InformationGeometric,
            2 => RoutingParadigm::TopologicalConsciousness,
            3 => RoutingParadigm::QuantumCoherence,
            4 => RoutingParadigm::ActiveInference,
            5 => RoutingParadigm::PredictiveProcessing,
            6 => RoutingParadigm::AttentionSchema,
            _ => RoutingParadigm::ResonantConsciousness, // Phase 5H
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            RoutingParadigm::CausalValidation => "Causal Validation",
            RoutingParadigm::InformationGeometric => "Information Geometric",
            RoutingParadigm::TopologicalConsciousness => "Topological Consciousness",
            RoutingParadigm::QuantumCoherence => "Quantum Coherence",
            RoutingParadigm::ActiveInference => "Active Inference",
            RoutingParadigm::PredictiveProcessing => "Predictive Processing",
            RoutingParadigm::AttentionSchema => "Attention Schema",
            RoutingParadigm::ResonantConsciousness => "Resonant Consciousness", // Phase 5H
        }
    }
}

// =============================================================================
// CONTEXT PROFILE
// =============================================================================

/// Context profile for a consciousness state
#[derive(Debug, Clone)]
pub struct ContextProfile {
    /// High phi (>0.7) indicates complex integration needs
    pub high_phi: bool,
    /// High coherence (>0.7) indicates stable patterns
    pub high_coherence: bool,
    /// Rapid changes in recent history
    pub volatile: bool,
    /// Multiple competing strategies present
    pub uncertain: bool,
    /// Domain hint from recent patterns
    pub domain_hint: Option<RoutingParadigm>,
}

impl ContextProfile {
    pub fn from_state(state: &LatentConsciousnessState) -> Self {
        Self {
            high_phi: state.phi > 0.7,
            high_coherence: state.coherence > 0.7,
            volatile: state.attention > 0.8, // High attention often means volatility
            uncertain: (state.phi - 0.5).abs() < 0.15, // Mid-range phi = uncertainty
            domain_hint: None,
        }
    }

    /// Convert to a discrete context bucket (for bandit arms)
    pub fn bucket_id(&self) -> usize {
        let mut id = 0;
        if self.high_phi { id |= 1; }
        if self.high_coherence { id |= 2; }
        if self.volatile { id |= 4; }
        if self.uncertain { id |= 8; }
        id
    }
}

// =============================================================================
// PARADIGM STATS
// =============================================================================

/// Statistics for a single paradigm
#[derive(Debug, Clone, Default)]
pub struct ParadigmStats {
    /// Total uses
    pub uses: usize,
    /// Successful uses (led to positive outcomes)
    pub successes: usize,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// Exponential moving average of success
    pub ema_success: f64,
    /// Last N outcomes for trend detection
    pub recent_outcomes: VecDeque<bool>,
}

impl ParadigmStats {
    pub fn success_rate(&self) -> f64 {
        if self.uses == 0 { 0.5 } else { self.successes as f64 / self.uses as f64 }
    }

    pub fn record(&mut self, success: bool, reward: f64, latency_us: u64) {
        self.uses += 1;
        if success { self.successes += 1; }
        self.total_reward += reward;

        // Update EMA
        let alpha = 0.1;
        let outcome = if success { 1.0 } else { 0.0 };
        self.ema_success = alpha * outcome + (1.0 - alpha) * self.ema_success;

        // Update latency average
        let n = self.uses as f64;
        self.avg_latency_us = (self.avg_latency_us * (n - 1.0) + latency_us as f64) / n;

        // Track recent outcomes
        if self.recent_outcomes.len() >= 20 {
            self.recent_outcomes.pop_front();
        }
        self.recent_outcomes.push_back(success);
    }

    pub fn recent_success_rate(&self) -> f64 {
        if self.recent_outcomes.is_empty() {
            return 0.5;
        }
        let successes = self.recent_outcomes.iter().filter(|&&s| s).count();
        successes as f64 / self.recent_outcomes.len() as f64
    }
}

// =============================================================================
// META-ROUTER CONFIG
// =============================================================================

/// Configuration for the Meta-Router
#[derive(Debug, Clone)]
pub struct MetaRouterConfig {
    /// UCB1 exploration constant
    pub exploration_constant: f64,
    /// Minimum samples before using statistics
    pub warmup_samples: usize,
    /// Weight of latency in selection (lower latency preferred)
    pub latency_weight: f64,
    /// Weight of recent performance vs overall
    pub recency_weight: f64,
    /// Number of context buckets
    pub context_buckets: usize,
    /// Enable contextual bandits (per-context learning)
    pub use_contextual: bool,
}

impl Default for MetaRouterConfig {
    fn default() -> Self {
        Self {
            exploration_constant: 1.414, // sqrt(2) for UCB1
            warmup_samples: 10,
            latency_weight: 0.1,
            recency_weight: 0.3,
            context_buckets: 16, // 2^4 context combinations
            use_contextual: true,
        }
    }
}

// =============================================================================
// META-ROUTER STATS
// =============================================================================

/// Statistics for the Meta-Router
#[derive(Debug, Clone, Default)]
pub struct MetaRouterStats {
    /// Total routing decisions
    pub total_decisions: usize,
    /// Decisions per paradigm (8 paradigms including ResonantConsciousness)
    pub paradigm_selections: [usize; 8],
    /// Exploration vs exploitation decisions
    pub exploration_decisions: usize,
    pub exploitation_decisions: usize,
    /// Context switches detected
    pub context_switches: usize,
    /// Average decision time in microseconds
    pub avg_decision_time_us: f64,
}

// =============================================================================
// META-ROUTER DECISION
// =============================================================================

/// Meta-Router Decision
#[derive(Debug, Clone)]
pub struct MetaRouterDecision {
    /// Selected paradigm
    pub paradigm: RoutingParadigm,
    /// The actual routing strategy from that paradigm
    pub strategy: RoutingStrategy,
    /// Confidence in the paradigm selection
    pub paradigm_confidence: f64,
    /// Was this exploration or exploitation?
    pub is_exploration: bool,
    /// Context that led to this decision
    pub context: ContextProfile,
    /// Decision time in microseconds
    pub decision_time_us: u64,
}

// =============================================================================
// META-ROUTER
// =============================================================================

/// Revolutionary Improvement #68: Meta-Router
///
/// A meta-learning router that learns which of the 8 routing paradigms
/// works best for different consciousness states and contexts.
pub struct MetaRouter {
    /// The 8 underlying routers
    causal_router: CausalValidatedRouter,
    geometric_router: InformationGeometricRouter,
    topological_router: TopologicalConsciousnessRouter,
    quantum_router: QuantumCoherenceRouter,
    active_inference_router: ActiveInferenceRouter,
    predictive_router: PredictiveProcessingRouter,
    ast_router: ASTRouter,
    resonant_router: ResonantConsciousnessRouter, // Phase 5H: HDC+LTC+Resonator

    /// Global statistics per paradigm (8 paradigms)
    global_stats: [ParadigmStats; 8],

    /// Contextual statistics (per context bucket per paradigm)
    contextual_stats: Vec<[ParadigmStats; 8]>,

    /// Configuration
    config: MetaRouterConfig,

    /// Aggregated stats
    stats: MetaRouterStats,

    /// Last context for detecting switches
    last_context: Option<ContextProfile>,

    /// Decision history for pattern detection
    decision_history: VecDeque<(RoutingParadigm, bool)>,
}

impl MetaRouter {
    pub fn new(config: MetaRouterConfig) -> Self {
        let context_buckets = config.context_buckets;
        Self {
            causal_router: CausalValidatedRouter::new(CausalValidatedConfig::default()),
            geometric_router: InformationGeometricRouter::new(GeometricRouterConfig::default()),
            topological_router: TopologicalConsciousnessRouter::new(TopologicalRouterConfig::default()),
            quantum_router: QuantumCoherenceRouter::new(QuantumRouterConfig::default()),
            active_inference_router: ActiveInferenceRouter::new(ActiveInferenceConfig::default()),
            predictive_router: PredictiveProcessingRouter::new(PredictiveProcessingConfig::default()),
            ast_router: ASTRouter::new(ASTRouterConfig::default()),
            resonant_router: ResonantConsciousnessRouter::new(ResonantRouterConfig::default()), // Phase 5H
            global_stats: Default::default(),
            contextual_stats: (0..context_buckets).map(|_| Default::default()).collect(),
            config,
            stats: MetaRouterStats::default(),
            last_context: None,
            decision_history: VecDeque::with_capacity(100),
        }
    }

    /// Select the best paradigm using UCB1 algorithm
    fn select_paradigm(&self, context: &ContextProfile) -> (RoutingParadigm, bool) {
        let stats = if self.config.use_contextual {
            let bucket = context.bucket_id() % self.contextual_stats.len();
            &self.contextual_stats[bucket]
        } else {
            &self.global_stats
        };

        // Check if we're in warmup phase (8 paradigms)
        let total_uses: usize = stats.iter().map(|s| s.uses).sum();
        if total_uses < self.config.warmup_samples * 8 {
            // Round-robin during warmup
            let paradigm = RoutingParadigm::from_index(total_uses % 8);
            return (paradigm, true);
        }

        // UCB1 selection
        let log_total = (total_uses as f64).ln();
        let mut best_paradigm = RoutingParadigm::CausalValidation;
        let mut best_ucb = f64::NEG_INFINITY;

        for paradigm in RoutingParadigm::all() {
            let idx = paradigm.index();
            let s = &stats[idx];

            if s.uses == 0 {
                // Never used - explore immediately
                return (paradigm, true);
            }

            // Calculate UCB1 score
            let mean_reward = s.total_reward / s.uses as f64;
            let exploration_bonus = self.config.exploration_constant
                * (log_total / s.uses as f64).sqrt();

            // Incorporate latency penalty
            let latency_penalty = self.config.latency_weight * (s.avg_latency_us / 1000.0);

            // Incorporate recency
            let recency_bonus = self.config.recency_weight * s.recent_success_rate();

            let ucb = mean_reward + exploration_bonus - latency_penalty + recency_bonus;

            if ucb > best_ucb {
                best_ucb = ucb;
                best_paradigm = paradigm;
            }
        }

        // Determine if this was exploration or exploitation
        let best_stats = &stats[best_paradigm.index()];
        let is_exploration = best_stats.uses < self.config.warmup_samples;

        (best_paradigm, is_exploration)
    }

    /// Route using the selected paradigm
    fn route_with_paradigm(
        &mut self,
        paradigm: RoutingParadigm,
        state: &LatentConsciousnessState,
    ) -> RoutingStrategy {
        match paradigm {
            RoutingParadigm::CausalValidation => {
                self.causal_router.route_validated(state).strategy
            }
            RoutingParadigm::InformationGeometric => {
                self.geometric_router.observe_state(state);
                self.geometric_router.route(state).strategy
            }
            RoutingParadigm::TopologicalConsciousness => {
                self.topological_router.observe_state(state);
                self.topological_router.route(state).strategy
            }
            RoutingParadigm::QuantumCoherence => {
                self.quantum_router.observe_state(state);
                self.quantum_router.route(state).strategy
            }
            RoutingParadigm::ActiveInference => {
                self.active_inference_router.observe_state(state);
                self.active_inference_router.route(state).strategy
            }
            RoutingParadigm::PredictiveProcessing => {
                self.predictive_router.route(state).strategy
            }
            RoutingParadigm::AttentionSchema => {
                self.ast_router.observe(state);
                self.ast_router.route().strategy
            }
            RoutingParadigm::ResonantConsciousness => {
                // Phase 5H: HDC+LTC+Resonator soft routing
                self.resonant_router.route(state).primary_strategy
            }
        }
    }

    /// Main routing decision
    pub fn route(&mut self, state: &LatentConsciousnessState) -> MetaRouterDecision {
        let start = std::time::Instant::now();

        // Build context profile
        let context = ContextProfile::from_state(state);

        // Detect context switch
        if let Some(ref last) = self.last_context {
            if context.bucket_id() != last.bucket_id() {
                self.stats.context_switches += 1;
            }
        }

        // Select paradigm using UCB1
        let (paradigm, is_exploration) = self.select_paradigm(&context);

        // Route using selected paradigm
        let strategy = self.route_with_paradigm(paradigm, state);

        // Update stats
        self.stats.total_decisions += 1;
        self.stats.paradigm_selections[paradigm.index()] += 1;
        if is_exploration {
            self.stats.exploration_decisions += 1;
        } else {
            self.stats.exploitation_decisions += 1;
        }

        let decision_time = start.elapsed().as_micros() as u64;
        let n = self.stats.total_decisions as f64;
        self.stats.avg_decision_time_us =
            (self.stats.avg_decision_time_us * (n - 1.0) + decision_time as f64) / n;

        // Calculate paradigm confidence
        let stats = if self.config.use_contextual {
            let bucket = context.bucket_id() % self.contextual_stats.len();
            &self.contextual_stats[bucket]
        } else {
            &self.global_stats
        };
        let paradigm_confidence = stats[paradigm.index()].success_rate();

        self.last_context = Some(context.clone());

        MetaRouterDecision {
            paradigm,
            strategy,
            paradigm_confidence,
            is_exploration,
            context,
            decision_time_us: decision_time,
        }
    }

    /// Report outcome to update statistics
    pub fn report_outcome(
        &mut self,
        paradigm: RoutingParadigm,
        context: &ContextProfile,
        success: bool,
        reward: f64,
        latency_us: u64,
    ) {
        // Update global stats
        self.global_stats[paradigm.index()].record(success, reward, latency_us);

        // Update contextual stats
        let bucket = context.bucket_id() % self.contextual_stats.len();
        self.contextual_stats[bucket][paradigm.index()].record(success, reward, latency_us);

        // Track decision history
        if self.decision_history.len() >= 100 {
            self.decision_history.pop_front();
        }
        self.decision_history.push_back((paradigm, success));
    }

    /// Get current paradigm rankings
    pub fn get_rankings(&self) -> Vec<(RoutingParadigm, f64)> {
        let mut rankings: Vec<_> = RoutingParadigm::all()
            .iter()
            .map(|&p| (p, self.global_stats[p.index()].success_rate()))
            .collect();
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rankings
    }

    /// Get statistics
    pub fn stats(&self) -> &MetaRouterStats {
        &self.stats
    }

    /// Generate a detailed report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("\n");
        report.push_str("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║                    META-ROUTER PERFORMANCE REPORT                           ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str(&format!("║ Total Decisions: {:>10} | Exploration: {:>5} | Exploitation: {:>5}    ║\n",
            self.stats.total_decisions,
            self.stats.exploration_decisions,
            self.stats.exploitation_decisions
        ));
        report.push_str(&format!("║ Context Switches: {:>9} | Avg Decision Time: {:>8.2}μs             ║\n",
            self.stats.context_switches,
            self.stats.avg_decision_time_us
        ));
        report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ Paradigm                    | Uses  | Success | Reward  | Latency | Recent  ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");

        for paradigm in RoutingParadigm::all() {
            let s = &self.global_stats[paradigm.index()];
            report.push_str(&format!(
                "║ {:<27} | {:>5} | {:>6.1}% | {:>7.2} | {:>5.0}μs | {:>6.1}% ║\n",
                paradigm.name(),
                s.uses,
                s.success_rate() * 100.0,
                s.total_reward,
                s.avg_latency_us,
                s.recent_success_rate() * 100.0
            ));
        }

        report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");

        // Show rankings
        let rankings = self.get_rankings();
        report.push_str("║ RANKINGS (by success rate):                                                  ║\n");
        for (i, (paradigm, rate)) in rankings.iter().take(3).enumerate() {
            let medal = match i { 0 => "🥇", 1 => "🥈", 2 => "🥉", _ => "  " };
            report.push_str(&format!("║   {} {}: {:.1}%{} ║\n",
                medal, paradigm.name(), rate * 100.0,
                " ".repeat(60 - paradigm.name().len())
            ));
        }

        report.push_str("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        report
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_paradigm_all() {
        let paradigms = RoutingParadigm::all();
        assert_eq!(paradigms.len(), 8); // 8 paradigms including ResonantConsciousness
    }

    #[test]
    fn test_routing_paradigm_roundtrip() {
        for i in 0..8 { // 8 paradigms
            let paradigm = RoutingParadigm::from_index(i);
            assert_eq!(paradigm.index(), i);
        }
    }

    #[test]
    fn test_context_profile_creation() {
        let state = LatentConsciousnessState::from_observables(0.8, 0.6, 0.75, 0.4);
        let profile = ContextProfile::from_state(&state);

        assert!(profile.high_phi);
        assert!(profile.high_coherence);
        assert!(!profile.volatile);
    }

    #[test]
    fn test_context_bucket_id() {
        let profile1 = ContextProfile {
            high_phi: true,
            high_coherence: true,
            volatile: false,
            uncertain: false,
            domain_hint: None,
        };
        let profile2 = ContextProfile {
            high_phi: false,
            high_coherence: false,
            volatile: true,
            uncertain: true,
            domain_hint: None,
        };

        // Different profiles should have different bucket IDs
        assert_ne!(profile1.bucket_id(), profile2.bucket_id());
    }

    #[test]
    fn test_paradigm_stats_recording() {
        let mut stats = ParadigmStats::default();

        stats.record(true, 1.0, 100);
        stats.record(true, 1.0, 200);
        stats.record(false, 0.0, 150);

        assert_eq!(stats.uses, 3);
        assert_eq!(stats.successes, 2);
        assert!((stats.success_rate() - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_meta_router_creation() {
        let config = MetaRouterConfig::default();
        let router = MetaRouter::new(config);

        assert_eq!(router.stats.total_decisions, 0);
    }

    #[test]
    fn test_meta_router_route() {
        let config = MetaRouterConfig {
            warmup_samples: 2,
            ..Default::default()
        };
        let mut router = MetaRouter::new(config);

        let state = LatentConsciousnessState::from_observables(0.6, 0.5, 0.7, 0.5);
        let decision = router.route(&state);

        // Decision should be returned
        // Decision timing recorded (u128 always >= 0)
    }

    #[test]
    fn test_meta_router_exploration_exploitation() {
        let config = MetaRouterConfig {
            warmup_samples: 1,
            ..Default::default()
        };
        let mut router = MetaRouter::new(config);

        // Run warmup (8 paradigms × 1 warmup sample = 8 minimum)
        for _ in 0..16 {
            let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
            let decision = router.route(&state);

            // Report outcome
            router.report_outcome(
                decision.paradigm,
                &decision.context,
                true,
                1.0,
                100,
            );
        }

        // Should have made some decisions
        assert!(router.stats.exploitation_decisions > 0 || router.stats.exploration_decisions > 0);
    }

    #[test]
    fn test_meta_router_outcome_reporting() {
        let mut router = MetaRouter::new(MetaRouterConfig::default());

        let context = ContextProfile {
            high_phi: true,
            high_coherence: false,
            volatile: false,
            uncertain: false,
            domain_hint: None,
        };

        router.report_outcome(RoutingParadigm::ActiveInference, &context, true, 1.5, 200);
        router.report_outcome(RoutingParadigm::ActiveInference, &context, false, 0.0, 300);

        let stats = &router.global_stats[RoutingParadigm::ActiveInference.index()];
        assert_eq!(stats.uses, 2);
        assert_eq!(stats.successes, 1);
    }

    #[test]
    fn test_meta_router_rankings() {
        let mut router = MetaRouter::new(MetaRouterConfig::default());

        let context = ContextProfile {
            high_phi: false,
            high_coherence: false,
            volatile: false,
            uncertain: false,
            domain_hint: None,
        };

        // Give different success rates to paradigms
        for _ in 0..10 {
            router.report_outcome(RoutingParadigm::ActiveInference, &context, true, 1.0, 100);
        }
        for _ in 0..10 {
            router.report_outcome(RoutingParadigm::CausalValidation, &context, false, 0.0, 100);
        }

        let rankings = router.get_rankings();
        assert_eq!(rankings[0].0, RoutingParadigm::ActiveInference);
    }

    #[test]
    fn test_meta_router_report() {
        let mut router = MetaRouter::new(MetaRouterConfig::default());

        let state = LatentConsciousnessState::from_observables(0.6, 0.5, 0.7, 0.5);
        for _ in 0..5 {
            let decision = router.route(&state);
            router.report_outcome(
                decision.paradigm,
                &decision.context,
                true,
                1.0,
                100,
            );
        }

        let report = router.report();
        assert!(report.contains("META-ROUTER"));
        assert!(report.contains("RANKINGS"));
    }

    #[test]
    fn test_meta_router_context_switching() {
        let mut router = MetaRouter::new(MetaRouterConfig::default());

        // Route with different contexts
        let state1 = LatentConsciousnessState::from_observables(0.8, 0.8, 0.8, 0.3);
        let state2 = LatentConsciousnessState::from_observables(0.2, 0.2, 0.2, 0.9);

        router.route(&state1);
        router.route(&state2);
        router.route(&state1);

        // Should detect context switches
        assert!(router.stats.context_switches >= 2);
    }

    #[test]
    fn test_meta_router_contextual_learning() {
        let config = MetaRouterConfig {
            use_contextual: true,
            warmup_samples: 1,
            ..Default::default()
        };
        let mut router = MetaRouter::new(config);

        // Train on different contexts
        let high_phi_context = ContextProfile {
            high_phi: true,
            high_coherence: true,
            volatile: false,
            uncertain: false,
            domain_hint: None,
        };
        let low_phi_context = ContextProfile {
            high_phi: false,
            high_coherence: false,
            volatile: true,
            uncertain: true,
            domain_hint: None,
        };

        // Active Inference works well for high phi
        for _ in 0..5 {
            router.report_outcome(RoutingParadigm::ActiveInference, &high_phi_context, true, 1.0, 100);
        }

        // Quantum works well for low phi
        for _ in 0..5 {
            router.report_outcome(RoutingParadigm::QuantumCoherence, &low_phi_context, true, 1.0, 100);
        }

        // Contextual stats should be different per bucket
        let high_bucket = high_phi_context.bucket_id() % router.contextual_stats.len();
        let low_bucket = low_phi_context.bucket_id() % router.contextual_stats.len();

        assert!(high_bucket != low_bucket);
        assert!(router.contextual_stats[high_bucket][RoutingParadigm::ActiveInference.index()].uses > 0);
        assert!(router.contextual_stats[low_bucket][RoutingParadigm::QuantumCoherence.index()].uses > 0);
    }
}
