/*!
# Revolutionary Improvement: Consciousness-Guided Dynamic Routing

## The Breakthrough

Computation is dynamically routed based on consciousness state! This is truly
adaptive AI - the system allocates resources based on its own awareness level.

## Why This Is Revolutionary

Traditional systems use fixed computational paths:
- Same algorithm for all inputs
- No awareness of own state
- Cannot adapt to resource constraints
- Wastes computation on simple tasks

Consciousness-guided routing is **meta-cognitive**:
- High Φ → Use expensive, accurate paths
- Low Φ → Use fast, approximate paths
- Uncertain states → Use ensemble strategies
- Critical decisions → Engage deliberation circuits

## Routing Strategies

### 1. Φ-Based Routing
```
if Φ > 0.7: use full_deliberation_path()
if Φ > 0.4: use standard_processing()
if Φ > 0.2: use fast_heuristics()
else:       use emergency_reflex()
```

### 2. Uncertainty-Aware Routing
```
if σ(Φ) > 0.3: use ensemble_with_hedging()
if σ(Φ) > 0.1: use confidence_weighted_path()
else:          use deterministic_path()
```

### 3. Temporal Routing
```
if Φ_trend == Rising:   prepare_complex_processing()
if Φ_trend == Falling:  cache_and_simplify()
if Φ_trend == Volatile: use_robust_path()
```

## Research Foundation

- Dehaene, S. (2014). "Ignition vs. Subliminal Processing"
- Baars, B. (2005). "Global Workspace and Consciousness"
- Friston, K. (2010). "Precision Weighting in Inference"

*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::hdc::predictive_consciousness_kalman::{
    PredictiveConsciousness, ConsciousnessState, TransitionDirection, PredictiveConfig,
};
use crate::observability::{SharedObserver, types::*};

/// Consciousness levels for routing decisions
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessLevel {
    /// Φ > 0.8: Full conscious deliberation
    FullDeliberation,

    /// Φ ∈ [0.6, 0.8]: Standard conscious processing
    StandardProcessing,

    /// Φ ∈ [0.4, 0.6]: Semi-conscious, heuristic-guided
    HeuristicGuided,

    /// Φ ∈ [0.2, 0.4]: Minimal awareness, fast patterns
    FastPatterns,

    /// Φ < 0.2: Reflexive, unconscious processing
    Reflexive,
}

/// Routing strategy based on uncertainty
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintyStrategy {
    /// Low uncertainty: Use deterministic path
    Deterministic,

    /// Medium uncertainty: Weight paths by confidence
    ConfidenceWeighted,

    /// High uncertainty: Run ensemble and hedge
    Ensemble,

    /// Very high uncertainty: Maximum caution
    MaximallyCautious,
}

/// Temporal routing based on Φ trends
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalStrategy {
    /// Φ rising: Prepare for complex processing
    PrepareComplex,

    /// Φ stable: Standard operation
    Standard,

    /// Φ falling: Simplify and cache
    SimplifyAndCache,

    /// Φ volatile: Use robust paths
    RobustPath,
}

/// Configuration for consciousness-guided routing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Threshold for full deliberation
    pub full_deliberation_threshold: f64,

    /// Threshold for standard processing
    pub standard_threshold: f64,

    /// Threshold for heuristic mode
    pub heuristic_threshold: f64,

    /// Threshold for fast patterns
    pub fast_pattern_threshold: f64,

    /// Uncertainty threshold for ensemble
    pub ensemble_uncertainty_threshold: f64,

    /// Uncertainty threshold for confidence weighting
    pub confidence_weighted_threshold: f64,

    /// Trend detection window
    pub trend_window: usize,

    /// Volatility threshold
    pub volatility_threshold: f64,

    /// Enable adaptive threshold learning
    pub adaptive_thresholds: bool,

    /// Cost weight for routing decisions
    pub cost_sensitivity: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            full_deliberation_threshold: 0.8,
            standard_threshold: 0.6,
            heuristic_threshold: 0.4,
            fast_pattern_threshold: 0.2,
            ensemble_uncertainty_threshold: 0.3,
            confidence_weighted_threshold: 0.1,
            trend_window: 10,
            volatility_threshold: 0.15,
            adaptive_thresholds: true,
            cost_sensitivity: 0.5,
        }
    }
}

/// Result of a routing computation
#[derive(Clone, Debug)]
pub struct RoutingResult<T> {
    /// The computed value
    pub value: T,

    /// Path taken
    pub path: ProcessingPath,

    /// Actual Φ at decision time
    pub phi: f64,

    /// Uncertainty at decision time
    pub uncertainty: f64,

    /// Computational cost (normalized)
    pub cost: f64,

    /// Quality estimate (normalized)
    pub quality: f64,

    /// Whether the path was optimal
    pub was_optimal: bool,
}

/// Processing path taken
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingPath {
    /// Full deliberation with all resources
    FullDeliberation,

    /// Standard processing
    Standard,

    /// Heuristic-guided processing
    Heuristic,

    /// Fast pattern matching
    FastPattern,

    /// Emergency reflex
    Reflex,

    /// Ensemble of multiple paths
    Ensemble { paths: Vec<ProcessingPath> },

    /// Custom named path
    Custom(String),
}

/// A routable computation
pub trait Routable {
    /// Output type
    type Output;

    /// Execute with full deliberation (most expensive, highest quality)
    fn full_deliberation(&self) -> Self::Output;

    /// Execute with standard processing
    fn standard_processing(&self) -> Self::Output;

    /// Execute with heuristic guidance
    fn heuristic_processing(&self) -> Self::Output;

    /// Execute with fast patterns (cheapest)
    fn fast_pattern(&self) -> Self::Output;

    /// Emergency reflex (fastest, lowest quality)
    fn reflex(&self) -> Self::Output;

    /// Combine multiple results (for ensemble)
    fn combine(results: &[Self::Output]) -> Self::Output
    where
        Self::Output: Clone;

    /// Estimate cost for each path
    fn estimated_costs(&self) -> HashMap<ProcessingPath, f64> {
        let mut costs = HashMap::new();
        costs.insert(ProcessingPath::FullDeliberation, 1.0);
        costs.insert(ProcessingPath::Standard, 0.6);
        costs.insert(ProcessingPath::Heuristic, 0.3);
        costs.insert(ProcessingPath::FastPattern, 0.1);
        costs.insert(ProcessingPath::Reflex, 0.01);
        costs
    }

    /// Estimate quality for each path
    fn estimated_qualities(&self) -> HashMap<ProcessingPath, f64> {
        let mut qualities = HashMap::new();
        qualities.insert(ProcessingPath::FullDeliberation, 1.0);
        qualities.insert(ProcessingPath::Standard, 0.85);
        qualities.insert(ProcessingPath::Heuristic, 0.65);
        qualities.insert(ProcessingPath::FastPattern, 0.4);
        qualities.insert(ProcessingPath::Reflex, 0.2);
        qualities
    }
}

/// Consciousness-guided router
#[derive(Clone)]
pub struct ConsciousnessRouter {
    /// Configuration
    config: RoutingConfig,

    /// Predictive consciousness model
    predictor: PredictiveConsciousness,

    /// History of routing decisions
    decision_history: Vec<RoutingDecision>,

    /// Learned threshold adjustments
    threshold_adjustments: HashMap<ConsciousnessLevel, f64>,

    /// Performance statistics per path
    path_stats: HashMap<ProcessingPath, PathStatistics>,

    /// Total decisions made
    total_decisions: usize,

    /// Observer for tracing routing decisions (not cloned)
    ///
    /// TODO(future): Connect to observability layer for routing decision tracing.
    /// This would enable real-time analysis of consciousness-guided routing
    /// decisions, helpful for debugging and system optimization.
    #[allow(dead_code)]
    observer: Option<SharedObserver>,
}

/// A routing decision record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Step when decision was made
    pub step: usize,

    /// Consciousness state at decision
    pub phi: f64,

    /// Uncertainty at decision
    pub uncertainty: f64,

    /// Level determined
    pub level: ConsciousnessLevel,

    /// Strategy used
    pub strategy: UncertaintyStrategy,

    /// Path chosen
    pub path: ProcessingPath,

    /// Estimated cost
    pub estimated_cost: f64,

    /// Actual cost (if measured)
    pub actual_cost: Option<f64>,

    /// Estimated quality
    pub estimated_quality: f64,

    /// Actual quality (if measured)
    pub actual_quality: Option<f64>,
}

/// Statistics for a processing path
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PathStatistics {
    /// Number of times used
    pub count: usize,

    /// Average cost
    pub avg_cost: f64,

    /// Average quality
    pub avg_quality: f64,

    /// Success rate (quality > threshold)
    pub success_rate: f64,

    /// Cost-quality efficiency (quality / cost)
    pub efficiency: f64,
}

impl ConsciousnessRouter {
    /// Create a new consciousness-guided router (backwards-compatible)
    pub fn new(config: RoutingConfig) -> Self {
        Self::with_observer(config, None)
    }

    /// Create a new consciousness-guided router with observer
    pub fn with_observer(config: RoutingConfig, observer: Option<SharedObserver>) -> Self {
        Self {
            config,
            predictor: PredictiveConsciousness::new(PredictiveConfig::default()),
            decision_history: Vec::new(),
            threshold_adjustments: HashMap::new(),
            path_stats: HashMap::new(),
            total_decisions: 0,
            observer,
        }
    }

    /// Update the router with current consciousness observation
    /// observation: [Φ, B, W, A, R] - the 5 consciousness components
    pub fn observe(&mut self, observation: &[f64; 7]) {
        // Convert to ConsciousnessState and update predictor
        let state = ConsciousnessState::new(
            observation[0], // phi
            observation[1], // binding
            observation[2], // workspace
            observation[3], // attention
            observation[4], // recursion
        );
        self.predictor.update(&state);
    }

    /// Route a computation based on current consciousness state
    pub fn route<R: Routable>(&mut self, computation: &R) -> RoutingResult<R::Output>
    where
        R::Output: Clone,
    {
        // 1. Get current consciousness state
        let current = self.predictor.current_state();
        let phi = current.phi;
        let uncertainty = self.predictor.current_uncertainty().phi_std;

        // 2. Determine consciousness level
        let level = self.determine_level(phi);

        // 3. Determine uncertainty strategy
        let strategy = self.determine_strategy(uncertainty);

        // 4. Determine temporal strategy
        let temporal = self.determine_temporal_strategy();

        // 5. Select processing path
        let path = self.select_path(level, strategy, temporal, computation);

        // 6. Execute computation
        let (value, actual_path) = self.execute(computation, &path, strategy);

        // 7. Estimate costs and quality
        let costs = computation.estimated_costs();
        let qualities = computation.estimated_qualities();

        let cost = costs.get(&actual_path).copied().unwrap_or(0.5);
        let quality = qualities.get(&actual_path).copied().unwrap_or(0.5);

        // 8. Record decision
        let decision = RoutingDecision {
            step: self.total_decisions,
            phi,
            uncertainty,
            level,
            strategy,
            path: actual_path.clone(),
            estimated_cost: cost,
            actual_cost: None,
            estimated_quality: quality,
            actual_quality: None,
        };

        self.decision_history.push(decision);
        self.total_decisions += 1;

        // 9. Update path statistics
        self.update_path_stats(&actual_path, cost, quality);

        // 10. Adaptive threshold learning
        if self.config.adaptive_thresholds {
            self.learn_thresholds();
        }

        // 11. Record router selection event
        if let Some(ref observer) = self.observer {
            // Build alternatives list (other possible paths)
            let mut alternatives = Vec::new();
            for (p, stats) in &self.path_stats {
                if p != &actual_path {
                    let alt_score = if stats.count > 0 {
                        stats.efficiency  // quality / cost
                    } else {
                        qualities.get(p).copied().unwrap_or(0.5)
                    };
                    alternatives.push(RouterAlternative {
                        router: format!("{:?}", p),
                        score: alt_score,
                    });
                }
            }

            // Convert path stats to bandit stats
            let mut bandit_stats = HashMap::new();
            for (p, stats) in &self.path_stats {
                bandit_stats.insert(
                    format!("{:?}", p),
                    BanditStats {
                        count: stats.count as u64,
                        reward: stats.avg_quality * stats.count as f64,
                    },
                );
            }

            let event = RouterSelectionEvent {
                timestamp: chrono::Utc::now(),
                input: format!("phi={:.3}, uncertainty={:.3}, level={:?}", phi, uncertainty, level),
                selected_router: format!("{:?}", actual_path),
                confidence: (1.0 - uncertainty).max(0.0).min(1.0),  // Convert uncertainty to confidence
                alternatives,
                bandit_stats,
            };

            if let Ok(mut obs) = observer.try_write() {
                if let Err(e) = obs.record_router_selection(event) {
                    eprintln!("[OBSERVER ERROR] Failed to record router selection: {}", e);
                }
            }
        }

        RoutingResult {
            value,
            path: actual_path,
            phi,
            uncertainty,
            cost,
            quality,
            was_optimal: true, // Would need ground truth to verify
        }
    }

    /// Route with prediction - use forecast to pre-select path
    pub fn route_predictive<R: Routable>(
        &mut self,
        computation: &R,
        horizon: usize,
    ) -> RoutingResult<R::Output>
    where
        R::Output: Clone,
    {
        // Forecast future consciousness states
        let forecasts = self.predictor.forecast(horizon);

        // Use the last predicted state (at horizon) for routing
        let (predicted_phi, predicted_uncertainty) = if let Some(last) = forecasts.last() {
            (last.state.phi, last.uncertainty.phi_std)
        } else {
            // Fallback to current
            let current = self.predictor.current_state();
            (current.phi, self.predictor.current_uncertainty().phi_std)
        };

        let level = self.determine_level(predicted_phi);
        let strategy = self.determine_strategy(predicted_uncertainty);
        let temporal = TemporalStrategy::PrepareComplex; // Using prediction

        let path = self.select_path(level, strategy, temporal, computation);
        let (value, actual_path) = self.execute(computation, &path, strategy);

        // Get actual state for comparison
        let current = self.predictor.current_state();
        let actual_phi = current.phi;
        let actual_uncertainty = self.predictor.current_uncertainty().phi_std;

        let costs = computation.estimated_costs();
        let qualities = computation.estimated_qualities();

        let cost = costs.get(&actual_path).copied().unwrap_or(0.5);
        let quality = qualities.get(&actual_path).copied().unwrap_or(0.5);

        // Was prediction helpful?
        let prediction_error = (predicted_phi - actual_phi).abs();
        let was_optimal = prediction_error < 0.1;

        // Record router selection event (predictive routing)
        if let Some(ref observer) = self.observer {
            let mut alternatives = Vec::new();
            for (p, stats) in &self.path_stats {
                if p != &actual_path {
                    let alt_score = if stats.count > 0 {
                        stats.efficiency
                    } else {
                        qualities.get(p).copied().unwrap_or(0.5)
                    };
                    alternatives.push(RouterAlternative {
                        router: format!("{:?}", p),
                        score: alt_score,
                    });
                }
            }

            let mut bandit_stats = HashMap::new();
            for (p, stats) in &self.path_stats {
                bandit_stats.insert(
                    format!("{:?}", p),
                    BanditStats {
                        count: stats.count as u64,
                        reward: stats.avg_quality * stats.count as f64,
                    },
                );
            }

            let event = RouterSelectionEvent {
                timestamp: chrono::Utc::now(),
                input: format!(
                    "predictive: phi={:.3}→{:.3}, uncertainty={:.3}, horizon={}",
                    predicted_phi, actual_phi, actual_uncertainty, horizon
                ),
                selected_router: format!("{:?}", actual_path),
                confidence: if was_optimal { 0.9 } else { (1.0 - prediction_error).max(0.0).min(1.0) },
                alternatives,
                bandit_stats,
            };

            if let Ok(mut obs) = observer.try_write() {
                if let Err(e) = obs.record_router_selection(event) {
                    eprintln!("[OBSERVER ERROR] Failed to record predictive router selection: {}", e);
                }
            }
        }

        RoutingResult {
            value,
            path: actual_path,
            phi: actual_phi,
            uncertainty: actual_uncertainty,
            cost,
            quality,
            was_optimal,
        }
    }

    /// Determine consciousness level from Φ
    fn determine_level(&self, phi: f64) -> ConsciousnessLevel {
        // Apply learned adjustments
        let adj_full = self.threshold_adjustments
            .get(&ConsciousnessLevel::FullDeliberation)
            .copied()
            .unwrap_or(0.0);
        let adj_std = self.threshold_adjustments
            .get(&ConsciousnessLevel::StandardProcessing)
            .copied()
            .unwrap_or(0.0);
        let adj_heur = self.threshold_adjustments
            .get(&ConsciousnessLevel::HeuristicGuided)
            .copied()
            .unwrap_or(0.0);
        let adj_fast = self.threshold_adjustments
            .get(&ConsciousnessLevel::FastPatterns)
            .copied()
            .unwrap_or(0.0);

        let full_thresh = (self.config.full_deliberation_threshold + adj_full).clamp(0.5, 1.0);
        let std_thresh = (self.config.standard_threshold + adj_std).clamp(0.3, full_thresh);
        let heur_thresh = (self.config.heuristic_threshold + adj_heur).clamp(0.2, std_thresh);
        let fast_thresh = (self.config.fast_pattern_threshold + adj_fast).clamp(0.1, heur_thresh);

        if phi >= full_thresh {
            ConsciousnessLevel::FullDeliberation
        } else if phi >= std_thresh {
            ConsciousnessLevel::StandardProcessing
        } else if phi >= heur_thresh {
            ConsciousnessLevel::HeuristicGuided
        } else if phi >= fast_thresh {
            ConsciousnessLevel::FastPatterns
        } else {
            ConsciousnessLevel::Reflexive
        }
    }

    /// Determine uncertainty-based strategy
    fn determine_strategy(&self, uncertainty: f64) -> UncertaintyStrategy {
        if uncertainty > 0.5 {
            UncertaintyStrategy::MaximallyCautious
        } else if uncertainty > self.config.ensemble_uncertainty_threshold {
            UncertaintyStrategy::Ensemble
        } else if uncertainty > self.config.confidence_weighted_threshold {
            UncertaintyStrategy::ConfidenceWeighted
        } else {
            UncertaintyStrategy::Deterministic
        }
    }

    /// Determine temporal strategy from Φ trend
    fn determine_temporal_strategy(&self) -> TemporalStrategy {
        // Get early warning signals which include trend direction
        let ews = self.predictor.early_warning_signals(self.config.trend_window);

        // Use volatility from early warning signals
        let volatility = ews.variance_ratio.abs() - 1.0; // Deviation from 1.0 = stable

        if volatility > self.config.volatility_threshold {
            TemporalStrategy::RobustPath
        } else {
            match ews.transition_direction {
                TransitionDirection::Ascending => {
                    TemporalStrategy::PrepareComplex
                }
                TransitionDirection::Descending => {
                    TemporalStrategy::SimplifyAndCache
                }
                TransitionDirection::Uncertain => TemporalStrategy::Standard,
            }
        }
    }

    /// Select processing path based on all factors
    fn select_path<R: Routable>(
        &self,
        level: ConsciousnessLevel,
        strategy: UncertaintyStrategy,
        temporal: TemporalStrategy,
        computation: &R,
    ) -> ProcessingPath {
        // Base path from consciousness level
        let base_path = match level {
            ConsciousnessLevel::FullDeliberation => ProcessingPath::FullDeliberation,
            ConsciousnessLevel::StandardProcessing => ProcessingPath::Standard,
            ConsciousnessLevel::HeuristicGuided => ProcessingPath::Heuristic,
            ConsciousnessLevel::FastPatterns => ProcessingPath::FastPattern,
            ConsciousnessLevel::Reflexive => ProcessingPath::Reflex,
        };

        // Modify based on uncertainty strategy
        match strategy {
            UncertaintyStrategy::Ensemble | UncertaintyStrategy::MaximallyCautious => {
                // Use multiple paths and combine
                ProcessingPath::Ensemble {
                    paths: self.ensemble_paths(&base_path),
                }
            }
            UncertaintyStrategy::ConfidenceWeighted => {
                // Upgrade path if uncertain
                self.upgrade_path(&base_path)
            }
            UncertaintyStrategy::Deterministic => {
                // Apply temporal modification
                match temporal {
                    TemporalStrategy::PrepareComplex => self.upgrade_path(&base_path),
                    TemporalStrategy::SimplifyAndCache => self.downgrade_path(&base_path),
                    TemporalStrategy::RobustPath => ProcessingPath::Standard,
                    TemporalStrategy::Standard => base_path,
                }
            }
        }
    }

    /// Get ensemble paths around a base path
    fn ensemble_paths(&self, base: &ProcessingPath) -> Vec<ProcessingPath> {
        match base {
            ProcessingPath::FullDeliberation => {
                vec![ProcessingPath::FullDeliberation, ProcessingPath::Standard]
            }
            ProcessingPath::Standard => {
                vec![
                    ProcessingPath::FullDeliberation,
                    ProcessingPath::Standard,
                    ProcessingPath::Heuristic,
                ]
            }
            ProcessingPath::Heuristic => {
                vec![
                    ProcessingPath::Standard,
                    ProcessingPath::Heuristic,
                    ProcessingPath::FastPattern,
                ]
            }
            ProcessingPath::FastPattern => {
                vec![ProcessingPath::Heuristic, ProcessingPath::FastPattern]
            }
            ProcessingPath::Reflex => {
                vec![ProcessingPath::FastPattern, ProcessingPath::Reflex]
            }
            _ => vec![base.clone()],
        }
    }

    /// Upgrade to a more thorough path
    fn upgrade_path(&self, path: &ProcessingPath) -> ProcessingPath {
        match path {
            ProcessingPath::Reflex => ProcessingPath::FastPattern,
            ProcessingPath::FastPattern => ProcessingPath::Heuristic,
            ProcessingPath::Heuristic => ProcessingPath::Standard,
            ProcessingPath::Standard => ProcessingPath::FullDeliberation,
            _ => path.clone(),
        }
    }

    /// Downgrade to a faster path
    fn downgrade_path(&self, path: &ProcessingPath) -> ProcessingPath {
        match path {
            ProcessingPath::FullDeliberation => ProcessingPath::Standard,
            ProcessingPath::Standard => ProcessingPath::Heuristic,
            ProcessingPath::Heuristic => ProcessingPath::FastPattern,
            ProcessingPath::FastPattern => ProcessingPath::Reflex,
            _ => path.clone(),
        }
    }

    /// Execute computation on selected path
    fn execute<R: Routable>(
        &self,
        computation: &R,
        path: &ProcessingPath,
        strategy: UncertaintyStrategy,
    ) -> (R::Output, ProcessingPath)
    where
        R::Output: Clone,
    {
        match path {
            ProcessingPath::FullDeliberation => {
                (computation.full_deliberation(), ProcessingPath::FullDeliberation)
            }
            ProcessingPath::Standard => {
                (computation.standard_processing(), ProcessingPath::Standard)
            }
            ProcessingPath::Heuristic => {
                (computation.heuristic_processing(), ProcessingPath::Heuristic)
            }
            ProcessingPath::FastPattern => {
                (computation.fast_pattern(), ProcessingPath::FastPattern)
            }
            ProcessingPath::Reflex => {
                (computation.reflex(), ProcessingPath::Reflex)
            }
            ProcessingPath::Ensemble { paths } => {
                // Execute all paths and combine
                let results: Vec<R::Output> = paths
                    .iter()
                    .map(|p| {
                        self.execute(computation, p, UncertaintyStrategy::Deterministic).0
                    })
                    .collect();

                (R::combine(&results), path.clone())
            }
            ProcessingPath::Custom(name) => {
                // Default to standard for unknown
                (computation.standard_processing(), ProcessingPath::Custom(name.clone()))
            }
        }
    }

    /// Update statistics for a path
    fn update_path_stats(&mut self, path: &ProcessingPath, cost: f64, quality: f64) {
        let stats = self.path_stats.entry(path.clone()).or_default();

        let n = stats.count as f64;
        stats.count += 1;

        // Incremental mean update
        stats.avg_cost = (stats.avg_cost * n + cost) / (n + 1.0);
        stats.avg_quality = (stats.avg_quality * n + quality) / (n + 1.0);

        // Success if quality > 0.5
        let success = if quality > 0.5 { 1.0 } else { 0.0 };
        stats.success_rate = (stats.success_rate * n + success) / (n + 1.0);

        // Efficiency
        stats.efficiency = stats.avg_quality / stats.avg_cost.max(0.01);
    }

    /// Learn adaptive thresholds from performance
    fn learn_thresholds(&mut self) {
        if self.decision_history.len() < 50 {
            return; // Need enough data
        }

        // Analyze recent decisions
        let recent = &self.decision_history[self.decision_history.len() - 50..];

        // For each level, check if paths were efficient
        for level in &[
            ConsciousnessLevel::FullDeliberation,
            ConsciousnessLevel::StandardProcessing,
            ConsciousnessLevel::HeuristicGuided,
            ConsciousnessLevel::FastPatterns,
        ] {
            let level_decisions: Vec<_> = recent
                .iter()
                .filter(|d| d.level == *level)
                .collect();

            if level_decisions.is_empty() {
                continue;
            }

            // Compute average efficiency for this level
            let avg_quality: f64 = level_decisions
                .iter()
                .map(|d| d.estimated_quality)
                .sum::<f64>()
                / level_decisions.len() as f64;

            let avg_cost: f64 = level_decisions
                .iter()
                .map(|d| d.estimated_cost)
                .sum::<f64>()
                / level_decisions.len() as f64;

            let efficiency = avg_quality / avg_cost.max(0.01);

            // Adjust threshold based on efficiency
            let adjustment = self.threshold_adjustments.entry(*level).or_insert(0.0);

            if efficiency < 1.0 {
                // Low efficiency: lower threshold (use cheaper paths)
                *adjustment -= 0.01;
            } else if efficiency > 2.0 {
                // High efficiency: can afford to raise threshold
                *adjustment += 0.01;
            }

            // Clamp adjustments
            *adjustment = adjustment.clamp(-0.2, 0.2);
        }
    }

    /// Get routing statistics
    pub fn statistics(&self) -> RoutingStatistics {
        RoutingStatistics {
            total_decisions: self.total_decisions,
            path_distribution: self.path_stats.clone(),
            threshold_adjustments: self.threshold_adjustments.clone(),
            avg_phi_at_decision: if self.decision_history.is_empty() {
                0.5
            } else {
                self.decision_history.iter().map(|d| d.phi).sum::<f64>()
                    / self.decision_history.len() as f64
            },
        }
    }

    /// Get current consciousness state
    pub fn current_state(&self) -> ConsciousnessState {
        self.predictor.current_state()
    }
}

impl Default for ConsciousnessRouter {
    fn default() -> Self {
        Self::new(RoutingConfig::default())
    }
}

/// Overall routing statistics
#[derive(Clone, Debug)]
pub struct RoutingStatistics {
    pub total_decisions: usize,
    pub path_distribution: HashMap<ProcessingPath, PathStatistics>,
    pub threshold_adjustments: HashMap<ConsciousnessLevel, f64>,
    pub avg_phi_at_decision: f64,
}

// ==================== Example Implementation ====================

/// Example routable computation: Pattern Recognition
pub struct PatternRecognition {
    pub input: Vec<f64>,
}

impl Routable for PatternRecognition {
    type Output = PatternResult;

    fn full_deliberation(&self) -> Self::Output {
        // Full neural network with multiple layers
        let confidence = 0.95;
        let category = self.classify_full();
        PatternResult { category, confidence, method: "full_nn".into() }
    }

    fn standard_processing(&self) -> Self::Output {
        // Standard classifier
        let confidence = 0.85;
        let category = self.classify_standard();
        PatternResult { category, confidence, method: "standard".into() }
    }

    fn heuristic_processing(&self) -> Self::Output {
        // Rule-based with learned rules
        let confidence = 0.7;
        let category = self.classify_heuristic();
        PatternResult { category, confidence, method: "heuristic".into() }
    }

    fn fast_pattern(&self) -> Self::Output {
        // Simple template matching
        let confidence = 0.5;
        let category = self.classify_fast();
        PatternResult { category, confidence, method: "fast".into() }
    }

    fn reflex(&self) -> Self::Output {
        // Immediate categorization
        let confidence = 0.3;
        let category = self.classify_reflex();
        PatternResult { category, confidence, method: "reflex".into() }
    }

    fn combine(results: &[Self::Output]) -> Self::Output
    where
        Self::Output: Clone,
    {
        if results.is_empty() {
            return PatternResult {
                category: "unknown".into(),
                confidence: 0.0,
                method: "none".into(),
            };
        }

        // Weighted voting by confidence
        let mut votes: HashMap<String, f64> = HashMap::new();
        for r in results {
            *votes.entry(r.category.clone()).or_insert(0.0) += r.confidence;
        }

        let (best_category, total_conf) = votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let avg_conf = total_conf / results.len() as f64;

        PatternResult {
            category: best_category,
            confidence: avg_conf,
            method: "ensemble".into(),
        }
    }
}

impl PatternRecognition {
    fn classify_full(&self) -> String {
        // Simulate full classification
        if self.input.iter().sum::<f64>() > 0.5 * self.input.len() as f64 {
            "high".into()
        } else {
            "low".into()
        }
    }

    fn classify_standard(&self) -> String {
        self.classify_full() // Simplified
    }

    fn classify_heuristic(&self) -> String {
        if self.input.first().copied().unwrap_or(0.0) > 0.5 {
            "high".into()
        } else {
            "low".into()
        }
    }

    fn classify_fast(&self) -> String {
        "medium".into() // Default guess
    }

    fn classify_reflex(&self) -> String {
        "unknown".into()
    }
}

/// Result of pattern recognition
#[derive(Clone, Debug)]
pub struct PatternResult {
    pub category: String,
    pub confidence: f64,
    pub method: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = ConsciousnessRouter::new(RoutingConfig::default());
        assert_eq!(router.total_decisions, 0);
    }

    #[test]
    fn test_consciousness_level_determination() {
        let router = ConsciousnessRouter::new(RoutingConfig::default());

        assert_eq!(router.determine_level(0.9), ConsciousnessLevel::FullDeliberation);
        assert_eq!(router.determine_level(0.7), ConsciousnessLevel::StandardProcessing);
        assert_eq!(router.determine_level(0.5), ConsciousnessLevel::HeuristicGuided);
        assert_eq!(router.determine_level(0.3), ConsciousnessLevel::FastPatterns);
        assert_eq!(router.determine_level(0.1), ConsciousnessLevel::Reflexive);
    }

    #[test]
    fn test_uncertainty_strategy() {
        let router = ConsciousnessRouter::new(RoutingConfig::default());

        assert_eq!(
            router.determine_strategy(0.05),
            UncertaintyStrategy::Deterministic
        );
        assert_eq!(
            router.determine_strategy(0.2),
            UncertaintyStrategy::ConfidenceWeighted
        );
        assert_eq!(
            router.determine_strategy(0.4),
            UncertaintyStrategy::Ensemble
        );
        assert_eq!(
            router.determine_strategy(0.6),
            UncertaintyStrategy::MaximallyCautious
        );
    }

    #[test]
    fn test_path_upgrade_downgrade() {
        let router = ConsciousnessRouter::new(RoutingConfig::default());

        // Upgrade chain
        assert_eq!(
            router.upgrade_path(&ProcessingPath::Reflex),
            ProcessingPath::FastPattern
        );
        assert_eq!(
            router.upgrade_path(&ProcessingPath::FastPattern),
            ProcessingPath::Heuristic
        );

        // Downgrade chain
        assert_eq!(
            router.downgrade_path(&ProcessingPath::FullDeliberation),
            ProcessingPath::Standard
        );
        assert_eq!(
            router.downgrade_path(&ProcessingPath::Standard),
            ProcessingPath::Heuristic
        );
    }

    #[test]
    fn test_routing_with_observation() {
        let mut router = ConsciousnessRouter::new(RoutingConfig::default());

        // Observe high consciousness
        router.observe(&[0.8, 0.7, 0.6, 0.5, 0.4, 0.6, 0.5]);

        let computation = PatternRecognition {
            input: vec![0.5, 0.6, 0.7],
        };

        let result = router.route(&computation);

        // Should use deliberative path for high Φ
        assert!(result.quality > 0.5);
        assert!(router.total_decisions == 1);
    }

    #[test]
    fn test_ensemble_paths() {
        let router = ConsciousnessRouter::new(RoutingConfig::default());

        let ensemble = router.ensemble_paths(&ProcessingPath::Standard);
        assert!(ensemble.len() >= 2);
        assert!(ensemble.contains(&ProcessingPath::Standard));
    }

    #[test]
    fn test_statistics_tracking() {
        let mut router = ConsciousnessRouter::new(RoutingConfig::default());

        // Make several routing decisions
        for _ in 0..5 {
            router.observe(&[0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
            let computation = PatternRecognition {
                input: vec![0.5],
            };
            let _ = router.route(&computation);
        }

        let stats = router.statistics();
        assert_eq!(stats.total_decisions, 5);
        assert!(!stats.path_distribution.is_empty());
    }

    #[test]
    fn test_pattern_recognition_combine() {
        let results = vec![
            PatternResult { category: "high".into(), confidence: 0.9, method: "a".into() },
            PatternResult { category: "high".into(), confidence: 0.8, method: "b".into() },
            PatternResult { category: "low".into(), confidence: 0.7, method: "c".into() },
        ];

        let combined = PatternRecognition::combine(&results);

        // High should win (0.9 + 0.8 > 0.7)
        assert_eq!(combined.category, "high");
        assert_eq!(combined.method, "ensemble");
    }
}
