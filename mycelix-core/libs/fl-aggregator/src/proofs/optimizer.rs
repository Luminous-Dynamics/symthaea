//! Circuit Auto-Optimization
//!
//! Automatically optimizes proof circuits by:
//! - Analyzing constraint complexity
//! - Reducing trace width where possible
//! - Optimizing constraint degree
//! - Batching similar proofs
//! - Selecting optimal parameters

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::proofs::{ProofConfig, ProofType, SecurityLevel};

// ============================================================================
// Optimization Hints
// ============================================================================

/// Hints for circuit optimization
#[derive(Clone, Debug, Default)]
pub struct OptimizationHints {
    /// Expected number of proofs (for batching decisions)
    pub expected_proof_count: Option<usize>,
    /// Maximum acceptable generation time
    pub max_generation_time: Option<Duration>,
    /// Maximum acceptable proof size
    pub max_proof_size: Option<usize>,
    /// Prefer smaller proofs over faster generation
    pub prefer_compact: bool,
    /// Allow parallel computation
    pub allow_parallel: bool,
    /// Target verification time
    pub target_verification_time: Option<Duration>,
}

impl OptimizationHints {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn for_batch(count: usize) -> Self {
        Self {
            expected_proof_count: Some(count),
            allow_parallel: true,
            ..Default::default()
        }
    }

    pub fn for_realtime() -> Self {
        Self {
            max_generation_time: Some(Duration::from_millis(100)),
            allow_parallel: true,
            ..Default::default()
        }
    }

    pub fn for_storage() -> Self {
        Self {
            prefer_compact: true,
            max_proof_size: Some(20_000), // 20KB
            ..Default::default()
        }
    }
}

// ============================================================================
// Circuit Analysis
// ============================================================================

/// Analysis of circuit characteristics
#[derive(Clone, Debug)]
pub struct CircuitAnalysis {
    /// Proof type
    pub proof_type: ProofType,
    /// Number of columns in trace
    pub trace_width: usize,
    /// Number of rows in trace
    pub trace_length: usize,
    /// Maximum constraint degree
    pub max_constraint_degree: usize,
    /// Number of constraints
    pub constraint_count: usize,
    /// Estimated proof size in bytes
    pub estimated_proof_size: usize,
    /// Estimated generation time
    pub estimated_generation_time: Duration,
    /// Estimated verification time
    pub estimated_verification_time: Duration,
    /// Optimization opportunities
    pub opportunities: Vec<OptimizationOpportunity>,
}

/// A specific optimization opportunity
#[derive(Clone, Debug)]
pub struct OptimizationOpportunity {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Description of the optimization
    pub description: String,
    /// Estimated speedup factor (1.0 = no change, 2.0 = 2x faster)
    pub speedup_factor: f64,
    /// Estimated size reduction factor
    pub size_reduction_factor: f64,
    /// Implementation difficulty (0-10)
    pub difficulty: u8,
}

/// Type of optimization
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationType {
    /// Reduce trace width by combining columns
    ReduceTraceWidth,
    /// Lower constraint degree
    LowerConstraintDegree,
    /// Use more efficient hash function
    OptimizeHash,
    /// Batch similar proofs together
    BatchProofs,
    /// Use lower security level
    ReduceSecurity,
    /// Enable parallel processing
    EnableParallel,
    /// Cache intermediate computations
    CacheIntermediates,
    /// Use recursive aggregation
    RecursiveAggregation,
}

/// Analyze a circuit for optimization opportunities
pub fn analyze_circuit(proof_type: ProofType, config: &ProofConfig) -> CircuitAnalysis {
    let (trace_width, trace_length, max_degree) = match proof_type {
        ProofType::Range => (3, 64, 2),
        ProofType::Membership => (8, 20, 2),
        ProofType::GradientIntegrity => (16, 256, 2),
        ProofType::IdentityAssurance => (12, 9, 2),
        ProofType::VoteEligibility => (10, 7, 2),
    };

    let constraint_count = trace_width * 2; // Rough estimate

    // Estimate sizes and times based on security level
    let security_factor = match config.security_level {
        SecurityLevel::Standard96 => 0.7,
        SecurityLevel::Standard128 => 1.0,
        SecurityLevel::High256 => 2.0,
    };

    let base_size = trace_width * trace_length * 32; // 32 bytes per field element
    let estimated_proof_size = (base_size as f64 * security_factor * 0.5) as usize; // Compression

    let base_time = Duration::from_millis((trace_width * trace_length) as u64 / 10);
    let estimated_generation_time = Duration::from_secs_f64(
        base_time.as_secs_f64() * security_factor,
    );
    let estimated_verification_time = Duration::from_millis(
        (estimated_generation_time.as_millis() / 10) as u64,
    );

    // Identify optimization opportunities
    let mut opportunities = Vec::new();

    if trace_width > 8 {
        opportunities.push(OptimizationOpportunity {
            optimization_type: OptimizationType::ReduceTraceWidth,
            description: "Trace width could be reduced by packing columns".to_string(),
            speedup_factor: 1.2,
            size_reduction_factor: 1.15,
            difficulty: 5,
        });
    }

    if !config.parallel {
        opportunities.push(OptimizationOpportunity {
            optimization_type: OptimizationType::EnableParallel,
            description: "Enable parallel FFT operations".to_string(),
            speedup_factor: 2.0,
            size_reduction_factor: 1.0,
            difficulty: 1,
        });
    }

    if config.security_level == SecurityLevel::High256 {
        opportunities.push(OptimizationOpportunity {
            optimization_type: OptimizationType::ReduceSecurity,
            description: "Consider Standard128 for non-critical proofs".to_string(),
            speedup_factor: 1.5,
            size_reduction_factor: 1.3,
            difficulty: 0,
        });
    }

    CircuitAnalysis {
        proof_type,
        trace_width,
        trace_length,
        max_constraint_degree: max_degree,
        constraint_count,
        estimated_proof_size,
        estimated_generation_time,
        estimated_verification_time,
        opportunities,
    }
}

// ============================================================================
// Optimizer
// ============================================================================

/// Configuration for the optimizer
#[derive(Clone, Debug)]
pub struct OptimizerConfig {
    /// Enable automatic security level selection
    pub auto_security: bool,
    /// Enable proof batching
    pub enable_batching: bool,
    /// Minimum batch size before batching kicks in
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Enable caching of optimization decisions
    pub cache_decisions: bool,
    /// Learning rate for adaptive optimization
    pub learning_rate: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            auto_security: true,
            enable_batching: true,
            min_batch_size: 5,
            max_batch_size: 100,
            cache_decisions: true,
            learning_rate: 0.1,
        }
    }
}

/// Circuit optimizer
pub struct CircuitOptimizer {
    config: OptimizerConfig,
    /// Performance history for learning
    history: Arc<RwLock<OptimizationHistory>>,
    /// Cached optimization decisions
    decisions: Arc<RwLock<HashMap<String, OptimizedConfig>>>,
}

/// History of optimization decisions and outcomes
#[derive(Debug, Default)]
struct OptimizationHistory {
    /// Performance samples by proof type
    samples: HashMap<ProofType, Vec<PerformanceSample>>,
    /// Average generation times by config
    avg_times: HashMap<String, Duration>,
}

/// A single performance sample
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct PerformanceSample {
    config_hash: String,
    generation_time: Duration,
    proof_size: usize,
    timestamp: Instant,
}

/// Optimized configuration result
#[derive(Clone, Debug)]
pub struct OptimizedConfig {
    /// Original proof config
    pub original: ProofConfig,
    /// Optimized proof config
    pub optimized: ProofConfig,
    /// Applied optimizations
    pub applied: Vec<OptimizationType>,
    /// Expected speedup
    pub expected_speedup: f64,
    /// Expected size reduction
    pub expected_size_reduction: f64,
    /// Confidence in optimization (0.0-1.0)
    pub confidence: f64,
}

impl CircuitOptimizer {
    /// Create a new optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            history: Arc::new(RwLock::new(OptimizationHistory::default())),
            decisions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimize configuration for a proof type
    pub async fn optimize(
        &self,
        proof_type: ProofType,
        config: ProofConfig,
        hints: &OptimizationHints,
    ) -> OptimizedConfig {
        // Check cache
        let cache_key = format!("{:?}-{:?}-{:?}", proof_type, config.security_level, hints.prefer_compact);
        if self.config.cache_decisions {
            if let Some(cached) = self.decisions.read().await.get(&cache_key) {
                return cached.clone();
            }
        }

        let analysis = analyze_circuit(proof_type, &config);
        let mut optimized = config.clone();
        let mut applied = Vec::new();
        let mut speedup = 1.0;
        let mut size_reduction = 1.0;

        // Apply optimizations based on hints and analysis
        for opp in &analysis.opportunities {
            if self.should_apply(&opp, hints) {
                match opp.optimization_type {
                    OptimizationType::EnableParallel if hints.allow_parallel => {
                        optimized.parallel = true;
                        speedup *= opp.speedup_factor;
                        applied.push(opp.optimization_type);
                    }
                    OptimizationType::ReduceSecurity => {
                        if let Some(max_time) = hints.max_generation_time {
                            if analysis.estimated_generation_time > max_time {
                                optimized.security_level = SecurityLevel::Standard128;
                                speedup *= opp.speedup_factor;
                                size_reduction *= opp.size_reduction_factor;
                                applied.push(opp.optimization_type);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Consider size constraints
        if let Some(max_size) = hints.max_proof_size {
            if analysis.estimated_proof_size > max_size {
                // Try lower security level
                if optimized.security_level == SecurityLevel::High256 {
                    optimized.security_level = SecurityLevel::Standard128;
                }
            }
        }

        let result = OptimizedConfig {
            original: config,
            optimized,
            applied,
            expected_speedup: speedup,
            expected_size_reduction: size_reduction,
            confidence: 0.8, // Base confidence
        };

        // Cache decision
        if self.config.cache_decisions {
            self.decisions.write().await.insert(cache_key, result.clone());
        }

        result
    }

    /// Check if an optimization should be applied
    fn should_apply(&self, opp: &OptimizationOpportunity, hints: &OptimizationHints) -> bool {
        // Don't apply difficult optimizations automatically
        if opp.difficulty > 3 {
            return false;
        }

        // Check time constraints
        if hints.max_generation_time.is_some() && opp.speedup_factor > 1.0 {
            return true;
        }

        // Check size constraints
        if hints.prefer_compact && opp.size_reduction_factor > 1.0 {
            return true;
        }

        false
    }

    /// Record performance sample for learning
    pub async fn record_sample(
        &self,
        proof_type: ProofType,
        config: &ProofConfig,
        generation_time: Duration,
        proof_size: usize,
    ) {
        let config_hash = format!("{:?}-{:?}", config.security_level, config.parallel);

        let sample = PerformanceSample {
            config_hash: config_hash.clone(),
            generation_time,
            proof_size,
            timestamp: Instant::now(),
        };

        let mut history = self.history.write().await;
        history.samples.entry(proof_type).or_default().push(sample);

        // Update average
        let samples = history.samples.get(&proof_type).unwrap();
        let count = samples.iter().filter(|s| s.config_hash == config_hash).count();
        let total: Duration = samples
            .iter()
            .filter(|s| s.config_hash == config_hash)
            .map(|s| s.generation_time)
            .sum();

        if count > 0 {
            history.avg_times.insert(config_hash, total / count as u32);
        }
    }

    /// Get optimization statistics
    pub async fn stats(&self) -> OptimizerStats {
        let history = self.history.read().await;
        let decisions = self.decisions.read().await;

        let total_samples: usize = history.samples.values().map(|v| v.len()).sum();

        OptimizerStats {
            total_samples,
            cached_decisions: decisions.len(),
            proof_types_tracked: history.samples.len(),
        }
    }

    /// Clear optimization cache
    pub async fn clear_cache(&self) {
        self.decisions.write().await.clear();
    }

    /// Get recommendations for a workload
    pub async fn recommend(&self, workload: &Workload) -> WorkloadRecommendation {
        let mut recommendations = Vec::new();

        // Analyze workload patterns
        if workload.proofs_per_second > 10.0 {
            recommendations.push(Recommendation {
                title: "Enable batching".to_string(),
                description: "High throughput workload would benefit from batch processing".to_string(),
                impact: Impact::High,
            });
        }

        if workload.avg_proof_size > 30_000 {
            recommendations.push(Recommendation {
                title: "Consider lower security".to_string(),
                description: "Large proofs may indicate overly high security level".to_string(),
                impact: Impact::Medium,
            });
        }

        if !workload.parallel_enabled && workload.proofs_per_second > 5.0 {
            recommendations.push(Recommendation {
                title: "Enable parallel processing".to_string(),
                description: "Parallel FFT would significantly improve throughput".to_string(),
                impact: Impact::High,
            });
        }

        let optimal_config = ProofConfig {
            security_level: if workload.avg_proof_size > 40_000 {
                SecurityLevel::Standard128
            } else {
                workload.current_security
            },
            parallel: true,
            max_proof_size: 0,
        };

        let expected_improvement = if recommendations.is_empty() {
            1.0
        } else {
            1.5
        };

        WorkloadRecommendation {
            recommendations,
            optimal_config,
            expected_improvement,
        }
    }
}

/// Optimizer statistics
#[derive(Clone, Debug)]
pub struct OptimizerStats {
    pub total_samples: usize,
    pub cached_decisions: usize,
    pub proof_types_tracked: usize,
}

/// Description of a workload for optimization
#[derive(Clone, Debug)]
pub struct Workload {
    /// Average proofs generated per second
    pub proofs_per_second: f64,
    /// Average proof size in bytes
    pub avg_proof_size: usize,
    /// Most common proof type
    pub primary_proof_type: ProofType,
    /// Current security level
    pub current_security: SecurityLevel,
    /// Whether parallel is enabled
    pub parallel_enabled: bool,
}

/// Recommendation for workload optimization
#[derive(Clone, Debug)]
pub struct WorkloadRecommendation {
    pub recommendations: Vec<Recommendation>,
    pub optimal_config: ProofConfig,
    pub expected_improvement: f64,
}

/// A single recommendation
#[derive(Clone, Debug)]
pub struct Recommendation {
    pub title: String,
    pub description: String,
    pub impact: Impact,
}

/// Impact level of a recommendation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Impact {
    Low,
    Medium,
    High,
}

// ============================================================================
// Auto-Tuner
// ============================================================================

/// Auto-tuner that continuously optimizes based on observed performance
pub struct AutoTuner {
    optimizer: CircuitOptimizer,
    /// Current best configurations per proof type
    best_configs: Arc<RwLock<HashMap<ProofType, ProofConfig>>>,
    /// Target latency for auto-tuning
    target_latency: Duration,
}

impl AutoTuner {
    /// Create a new auto-tuner
    pub fn new(target_latency: Duration) -> Self {
        Self {
            optimizer: CircuitOptimizer::new(OptimizerConfig::default()),
            best_configs: Arc::new(RwLock::new(HashMap::new())),
            target_latency,
        }
    }

    /// Get the current best config for a proof type
    pub async fn get_config(&self, proof_type: ProofType) -> ProofConfig {
        self.best_configs
            .read()
            .await
            .get(&proof_type)
            .cloned()
            .unwrap_or_default()
    }

    /// Record a sample and potentially update the best config
    pub async fn record_and_tune(
        &self,
        proof_type: ProofType,
        config: ProofConfig,
        generation_time: Duration,
        proof_size: usize,
    ) {
        self.optimizer
            .record_sample(proof_type, &config, generation_time, proof_size)
            .await;

        // Check if we need to tune
        if generation_time > self.target_latency {
            // Current config is too slow, try optimization
            let hints = OptimizationHints {
                max_generation_time: Some(self.target_latency),
                allow_parallel: true,
                ..Default::default()
            };

            let optimized = self.optimizer.optimize(proof_type, config, &hints).await;

            if optimized.expected_speedup > 1.1 {
                self.best_configs
                    .write()
                    .await
                    .insert(proof_type, optimized.optimized);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_circuit() {
        let config = ProofConfig::default();
        let analysis = analyze_circuit(ProofType::Range, &config);

        assert_eq!(analysis.proof_type, ProofType::Range);
        assert_eq!(analysis.trace_width, 3);
        assert!(analysis.estimated_proof_size > 0);
    }

    #[test]
    fn test_optimization_hints_default() {
        let hints = OptimizationHints::default();
        assert!(!hints.prefer_compact);
        assert!(!hints.allow_parallel);
    }

    #[test]
    fn test_optimization_hints_for_batch() {
        let hints = OptimizationHints::for_batch(100);
        assert_eq!(hints.expected_proof_count, Some(100));
        assert!(hints.allow_parallel);
    }

    #[test]
    fn test_optimization_hints_for_realtime() {
        let hints = OptimizationHints::for_realtime();
        assert!(hints.max_generation_time.is_some());
    }

    #[tokio::test]
    async fn test_optimizer_basic() {
        let optimizer = CircuitOptimizer::new(OptimizerConfig::default());
        let config = ProofConfig {
            security_level: SecurityLevel::High256,
            parallel: false,
            max_proof_size: 0,
        };

        let hints = OptimizationHints {
            max_generation_time: Some(Duration::from_millis(50)),
            allow_parallel: true,
            ..Default::default()
        };

        let result = optimizer.optimize(ProofType::Range, config, &hints).await;

        assert!(result.optimized.parallel);
        assert!(result.expected_speedup >= 1.0);
    }

    #[tokio::test]
    async fn test_optimizer_record_sample() {
        let optimizer = CircuitOptimizer::new(OptimizerConfig::default());

        optimizer
            .record_sample(
                ProofType::Range,
                &ProofConfig::default(),
                Duration::from_millis(50),
                15000,
            )
            .await;

        let stats = optimizer.stats().await;
        assert_eq!(stats.total_samples, 1);
    }

    #[tokio::test]
    async fn test_optimizer_cache() {
        let optimizer = CircuitOptimizer::new(OptimizerConfig {
            cache_decisions: true,
            ..Default::default()
        });

        let hints = OptimizationHints::default();
        let config = ProofConfig::default();

        // First call - not cached
        let _ = optimizer
            .optimize(ProofType::Range, config.clone(), &hints)
            .await;

        let stats = optimizer.stats().await;
        assert_eq!(stats.cached_decisions, 1);

        // Second call - should use cache
        let _ = optimizer.optimize(ProofType::Range, config, &hints).await;
    }

    #[tokio::test]
    async fn test_optimizer_recommend() {
        let optimizer = CircuitOptimizer::new(OptimizerConfig::default());

        let workload = Workload {
            proofs_per_second: 15.0,
            avg_proof_size: 20000,
            primary_proof_type: ProofType::Range,
            current_security: SecurityLevel::Standard128,
            parallel_enabled: false,
        };

        let recommendation = optimizer.recommend(&workload).await;
        assert!(!recommendation.recommendations.is_empty());
    }

    #[test]
    fn test_opportunity_type() {
        let opp = OptimizationOpportunity {
            optimization_type: OptimizationType::EnableParallel,
            description: "Test".to_string(),
            speedup_factor: 2.0,
            size_reduction_factor: 1.0,
            difficulty: 1,
        };

        assert_eq!(opp.optimization_type, OptimizationType::EnableParallel);
        assert_eq!(opp.difficulty, 1);
    }

    #[tokio::test]
    async fn test_auto_tuner() {
        let tuner = AutoTuner::new(Duration::from_millis(100));

        // Get default config
        let config = tuner.get_config(ProofType::Range).await;
        assert_eq!(config.security_level, SecurityLevel::Standard128);

        // Record a slow sample
        tuner
            .record_and_tune(
                ProofType::Range,
                ProofConfig::default(),
                Duration::from_millis(200),
                15000,
            )
            .await;
    }

    #[test]
    fn test_impact_levels() {
        assert_ne!(Impact::Low, Impact::High);
        assert_ne!(Impact::Medium, Impact::Low);
    }

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert!(config.auto_security);
        assert!(config.enable_batching);
        assert_eq!(config.min_batch_size, 5);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let optimizer = CircuitOptimizer::new(OptimizerConfig::default());

        let _ = optimizer
            .optimize(ProofType::Range, ProofConfig::default(), &OptimizationHints::default())
            .await;

        optimizer.clear_cache().await;

        let stats = optimizer.stats().await;
        assert_eq!(stats.cached_decisions, 0);
    }
}
