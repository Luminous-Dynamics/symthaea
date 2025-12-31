//! Adaptive Φ Calculator Orchestrator
//!
//! Phase 5E Integration: Dynamically selects the optimal Φ calculation algorithm
//! based on context, process count, and performance requirements.
//!
//! # Available Calculators
//!
//! - **RealPhiCalculator**: O(n³) - Most accurate, uses algebraic connectivity
//! - **ResonatorPhiCalculator**: O(n log N) - Fast, captures consciousness dynamics
//! - **TieredPhi**: Configurable tiers (Mock/Heuristic/Spectral/Exact)
//!
//! # Adaptive Selection Strategy
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │           ADAPTIVE Φ CALCULATOR ORCHESTRATION                        │
//! ├──────────────────────────────────────────────────────────────────────┤
//! │                                                                       │
//! │   n <= 5:   RealPhiCalculator (exact algebraic connectivity)         │
//! │   n <= 20:  Resonant OR Real (based on mode preference)              │
//! │   n > 20:   ResonatorPhiCalculator (fast dynamics-based)             │
//! │                                                                       │
//! │   Override: Force specific calculator with PhiMode                   │
//! │                                                                       │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use symthaea::hdc::phi_orchestrator::{PhiOrchestrator, PhiMode};
//!
//! // Create with adaptive mode (default)
//! let orchestrator = PhiOrchestrator::new(PhiMode::Adaptive);
//!
//! // Compute Φ - automatically selects best calculator
//! let result = orchestrator.compute(&process_hvs);
//! println!("Φ = {:.4} (via {:?})", result.phi, result.calculator_used);
//! ```

use crate::hdc::real_hv::RealHV;
use crate::hdc::phi_real::RealPhiCalculator;
use crate::hdc::phi_resonant::{ResonantPhiCalculator, ResonantConfig};
use crate::hdc::tiered_phi::{TieredPhi, ApproximationTier};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

/// Simple LRU cache entry for Φ results
#[derive(Clone)]
struct CacheEntry {
    /// Hash of input components
    hash: u64,
    /// Number of components (for quick validation)
    count: usize,
    /// Cached result
    result: PhiResult,
}

/// Maximum cache size
const CACHE_SIZE: usize = 16;

/// Φ calculation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiMode {
    /// Automatically select best calculator based on n
    Adaptive,

    /// Always use RealPhiCalculator (accurate, O(n³))
    Accurate,

    /// Always use ResonatorPhiCalculator (fast, O(n log N))
    Fast,

    /// Use TieredPhi with specific tier
    Tiered(ApproximationTier),

    /// Balanced: Use accurate for small n, fast for large n
    Balanced,
}

impl Default for PhiMode {
    fn default() -> Self {
        PhiMode::Adaptive
    }
}

/// Result of Φ calculation with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiResult {
    /// The computed Φ value
    pub phi: f64,

    /// Which calculator was used
    pub calculator_used: CalculatorType,

    /// Computation time in milliseconds
    pub computation_time_ms: f64,

    /// Number of processes/components evaluated
    pub process_count: usize,

    /// Whether the result converged (relevant for resonant)
    pub converged: Option<bool>,

    /// Iterations used (relevant for resonant)
    pub iterations: Option<usize>,

    /// Confidence in result (0.0-1.0)
    pub confidence: f64,
}

/// Type of calculator used
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalculatorType {
    /// RealPhiCalculator - algebraic connectivity
    Real,
    /// ResonatorPhiCalculator - coupled oscillator dynamics
    Resonant,
    /// TieredPhi with specific tier
    Tiered(ApproximationTier),
}

/// Adaptive Φ Calculator Orchestrator
///
/// Intelligently selects and applies the optimal Φ calculation algorithm
/// based on context, enabling both real-time performance and research accuracy.
///
/// Includes LRU result caching to avoid redundant computation on repeated
/// process sets (common in consciousness monitoring loops).
#[derive(Clone)]
pub struct PhiOrchestrator {
    /// Current calculation mode
    mode: PhiMode,

    /// Real-valued Φ calculator (accurate)
    real_calculator: RealPhiCalculator,

    /// Resonator-based Φ calculator (fast)
    resonant_calculator: ResonantPhiCalculator,

    /// Tiered Φ calculator (configurable)
    tiered_calculator: TieredPhi,

    /// Threshold for switching from accurate to fast (n > threshold uses fast)
    fast_threshold: usize,

    /// Track recent computation times for adaptive tuning
    recent_times_ms: Vec<f64>,

    /// Maximum recent times to track
    max_recent: usize,

    /// LRU cache for Φ results (avoids redundant computation)
    result_cache: Vec<CacheEntry>,

    /// Cache hit counter for diagnostics
    cache_hits: usize,

    /// Cache miss counter for diagnostics
    cache_misses: usize,
}

impl PhiOrchestrator {
    /// Create a new orchestrator with specified mode
    pub fn new(mode: PhiMode) -> Self {
        Self {
            mode,
            real_calculator: RealPhiCalculator::new(),
            resonant_calculator: ResonantPhiCalculator::new(),
            tiered_calculator: TieredPhi::new(ApproximationTier::default()),
            fast_threshold: 20,
            recent_times_ms: Vec::with_capacity(100),
            max_recent: 100,
            result_cache: Vec::with_capacity(CACHE_SIZE),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Compute hash of process components for cache lookup
    fn compute_processes_hash(processes: &[RealHV]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for (i, process) in processes.iter().enumerate() {
            i.hash(&mut hasher);
            // Hash the first few and last few elements for speed
            let values = &process.values;
            let len = values.len();
            for j in 0..len.min(8) {
                values[j].to_bits().hash(&mut hasher);
            }
            if len > 16 {
                for j in (len - 8)..len {
                    values[j].to_bits().hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Look up cached result
    fn cache_lookup(&self, hash: u64, count: usize) -> Option<&PhiResult> {
        self.result_cache.iter()
            .find(|e| e.hash == hash && e.count == count)
            .map(|e| &e.result)
    }

    /// Add result to cache (LRU eviction)
    fn cache_insert(&mut self, hash: u64, count: usize, result: PhiResult) {
        // Remove if already exists
        self.result_cache.retain(|e| e.hash != hash);

        // Evict oldest if at capacity
        if self.result_cache.len() >= CACHE_SIZE {
            self.result_cache.remove(0);
        }

        self.result_cache.push(CacheEntry { hash, count, result });
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache_hits, self.cache_misses)
    }

    /// Create with default adaptive mode
    pub fn adaptive() -> Self {
        Self::new(PhiMode::Adaptive)
    }

    /// Create with accurate mode (always use RealPhiCalculator)
    pub fn accurate() -> Self {
        Self::new(PhiMode::Accurate)
    }

    /// Create with fast mode (always use ResonatorPhiCalculator)
    pub fn fast() -> Self {
        Self::new(PhiMode::Fast)
    }

    /// Create with tiered mode
    pub fn tiered(tier: ApproximationTier) -> Self {
        Self::new(PhiMode::Tiered(tier))
    }

    /// Set the calculation mode
    pub fn set_mode(&mut self, mode: PhiMode) {
        self.mode = mode;
    }

    /// Get current mode
    pub fn mode(&self) -> PhiMode {
        self.mode
    }

    /// Set the threshold for switching to fast calculator
    pub fn set_fast_threshold(&mut self, threshold: usize) {
        self.fast_threshold = threshold;
    }

    /// Configure the resonant calculator
    pub fn configure_resonant(&mut self, config: ResonantConfig) {
        self.resonant_calculator = ResonantPhiCalculator::with_config(config);
    }

    /// Configure the tiered calculator
    pub fn configure_tiered(&mut self, tier: ApproximationTier) {
        self.tiered_calculator = TieredPhi::new(tier);
    }

    /// Compute Φ for a set of RealHV processes
    ///
    /// Automatically selects the optimal calculator based on mode and context.
    /// Results are cached for repeated computations with the same processes.
    pub fn compute(&mut self, processes: &[RealHV]) -> PhiResult {
        let n = processes.len();

        if n < 2 {
            return PhiResult {
                phi: 0.0,
                calculator_used: CalculatorType::Real,
                computation_time_ms: 0.0,
                process_count: n,
                converged: Some(true),
                iterations: Some(0),
                confidence: 1.0,
            };
        }

        // Cache lookup - avoid redundant computation
        let hash = Self::compute_processes_hash(processes);
        // Check cache (clone to avoid borrow issues)
        let cached = self.cache_lookup(hash, n).cloned();
        if let Some(result) = cached {
            self.cache_hits += 1;
            // Track cache hit time (~0ms) so stats reflect all compute calls
            self.track_time(0.0);
            return result;
        }
        self.cache_misses += 1;

        let start = Instant::now();

        // Select calculator based on mode
        let (phi, calculator_type, converged, iterations) = match self.mode {
            PhiMode::Adaptive => self.compute_adaptive(processes),
            PhiMode::Accurate => self.compute_accurate(processes),
            PhiMode::Fast => self.compute_fast(processes),
            PhiMode::Tiered(tier) => self.compute_tiered(processes, tier),
            PhiMode::Balanced => self.compute_balanced(processes),
        };

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        // Track computation time for adaptive tuning
        self.track_time(elapsed);

        // Calculate confidence based on calculator type and result
        let confidence = self.calculate_confidence(&calculator_type, n, converged);

        let result = PhiResult {
            phi,
            calculator_used: calculator_type,
            computation_time_ms: elapsed,
            process_count: n,
            converged,
            iterations,
            confidence,
        };

        // Cache the result for future lookups
        self.cache_insert(hash, n, result.clone());

        result
    }

    /// Simple compute that just returns Φ value
    pub fn compute_simple(&mut self, processes: &[RealHV]) -> f64 {
        self.compute(processes).phi
    }

    /// Adaptive computation: select best calculator based on n
    fn compute_adaptive(&self, processes: &[RealHV]) -> (f64, CalculatorType, Option<bool>, Option<usize>) {
        let n = processes.len();

        if n <= 5 {
            // Very small: use accurate
            let phi = self.real_calculator.compute(processes);
            (phi, CalculatorType::Real, Some(true), None)
        } else if n <= self.fast_threshold {
            // Medium: still use accurate but could switch
            let phi = self.real_calculator.compute(processes);
            (phi, CalculatorType::Real, Some(true), None)
        } else {
            // Large: use fast resonant
            let result = self.resonant_calculator.compute(processes);
            (result.phi, CalculatorType::Resonant, Some(result.converged), Some(result.iterations))
        }
    }

    /// Always use accurate (Real) calculator
    fn compute_accurate(&self, processes: &[RealHV]) -> (f64, CalculatorType, Option<bool>, Option<usize>) {
        let phi = self.real_calculator.compute(processes);
        (phi, CalculatorType::Real, Some(true), None)
    }

    /// Always use fast (Resonant) calculator
    fn compute_fast(&self, processes: &[RealHV]) -> (f64, CalculatorType, Option<bool>, Option<usize>) {
        let result = self.resonant_calculator.compute(processes);
        (result.phi, CalculatorType::Resonant, Some(result.converged), Some(result.iterations))
    }

    /// Use tiered calculator with specific tier
    /// Note: TieredPhi uses HV16, so we fall back to real calculator for RealHV
    fn compute_tiered(&mut self, processes: &[RealHV], tier: ApproximationTier) -> (f64, CalculatorType, Option<bool>, Option<usize>) {
        // TieredPhi works with HV16, not RealHV. For now, fall back to accurate calculation.
        // Future: Convert RealHV to HV16 for tiered computation
        let phi = self.real_calculator.compute(processes);
        (phi, CalculatorType::Tiered(tier), Some(true), None)
    }

    /// Balanced: accurate for small, fast for large
    fn compute_balanced(&self, processes: &[RealHV]) -> (f64, CalculatorType, Option<bool>, Option<usize>) {
        let n = processes.len();

        if n <= 10 {
            let phi = self.real_calculator.compute(processes);
            (phi, CalculatorType::Real, Some(true), None)
        } else {
            let result = self.resonant_calculator.compute(processes);
            (result.phi, CalculatorType::Resonant, Some(result.converged), Some(result.iterations))
        }
    }

    /// Track computation time for adaptive tuning
    fn track_time(&mut self, time_ms: f64) {
        if self.recent_times_ms.len() >= self.max_recent {
            self.recent_times_ms.remove(0);
        }
        self.recent_times_ms.push(time_ms);
    }

    /// Calculate confidence in result
    fn calculate_confidence(&self, calculator: &CalculatorType, n: usize, converged: Option<bool>) -> f64 {
        let base_confidence = match calculator {
            CalculatorType::Real => 0.95,  // Algebraic connectivity is well-understood
            CalculatorType::Resonant => {
                // Confidence depends on convergence
                if converged.unwrap_or(false) {
                    0.85
                } else {
                    0.5  // Lower confidence if didn't converge
                }
            }
            CalculatorType::Tiered(tier) => match tier {
                ApproximationTier::Exact => 0.99,
                ApproximationTier::Spectral => 0.90,
                ApproximationTier::Heuristic => 0.70,
                ApproximationTier::Mock => 0.10,
            }
        };

        // Adjust for component count (higher n = slightly lower confidence due to approximations)
        let n_factor = if n <= 5 {
            1.0
        } else if n <= 20 {
            0.98
        } else if n <= 100 {
            0.95
        } else {
            0.90
        };

        base_confidence * n_factor
    }

    /// Get average recent computation time
    pub fn average_computation_time_ms(&self) -> f64 {
        if self.recent_times_ms.is_empty() {
            0.0
        } else {
            self.recent_times_ms.iter().sum::<f64>() / self.recent_times_ms.len() as f64
        }
    }

    /// Get recommended mode based on recent performance
    pub fn recommend_mode(&self) -> PhiMode {
        let avg_time = self.average_computation_time_ms();

        if avg_time < 10.0 {
            // Fast enough for accurate
            PhiMode::Accurate
        } else if avg_time < 100.0 {
            // Use adaptive
            PhiMode::Adaptive
        } else {
            // Too slow, use fast
            PhiMode::Fast
        }
    }

    /// Get statistics about recent computations
    pub fn stats(&self) -> PhiOrchestratorStats {
        PhiOrchestratorStats {
            mode: self.mode,
            fast_threshold: self.fast_threshold,
            recent_computations: self.recent_times_ms.len(),
            average_time_ms: self.average_computation_time_ms(),
            min_time_ms: self.recent_times_ms.iter().cloned().fold(f64::INFINITY, f64::min),
            max_time_ms: self.recent_times_ms.iter().cloned().fold(0.0, f64::max),
        }
    }
}

impl Default for PhiOrchestrator {
    fn default() -> Self {
        Self::adaptive()
    }
}

/// Statistics about orchestrator performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiOrchestratorStats {
    pub mode: PhiMode,
    pub fast_threshold: usize,
    pub recent_computations: usize,
    pub average_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orch = PhiOrchestrator::new(PhiMode::Adaptive);
        assert_eq!(orch.mode(), PhiMode::Adaptive);
    }

    #[test]
    fn test_mode_switching() {
        let mut orch = PhiOrchestrator::adaptive();
        orch.set_mode(PhiMode::Fast);
        assert_eq!(orch.mode(), PhiMode::Fast);
    }

    #[test]
    fn test_empty_processes() {
        let mut orch = PhiOrchestrator::adaptive();
        let result = orch.compute(&[]);
        assert_eq!(result.phi, 0.0);
        assert_eq!(result.process_count, 0);
    }

    #[test]
    fn test_single_process() {
        let mut orch = PhiOrchestrator::adaptive();
        let hv = RealHV::random(256, 42);
        let result = orch.compute(&[hv]);
        assert_eq!(result.phi, 0.0);
        assert_eq!(result.process_count, 1);
    }

    #[test]
    fn test_small_processes_uses_accurate() {
        let mut orch = PhiOrchestrator::adaptive();
        let processes: Vec<RealHV> = (0..3)
            .map(|i| RealHV::random(256, i as u64))
            .collect();

        let result = orch.compute(&processes);
        assert_eq!(result.calculator_used, CalculatorType::Real);
        assert!(result.phi >= 0.0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut orch = PhiOrchestrator::adaptive();
        let processes: Vec<RealHV> = (0..3)
            .map(|i| RealHV::random(256, i as u64))
            .collect();

        // Compute a few times
        for _ in 0..5 {
            orch.compute(&processes);
        }

        let stats = orch.stats();
        assert_eq!(stats.recent_computations, 5);
        assert!(stats.average_time_ms >= 0.0);
    }

    #[test]
    fn test_confidence_calculation() {
        let mut orch = PhiOrchestrator::new(PhiMode::Accurate);
        let processes: Vec<RealHV> = (0..3)
            .map(|i| RealHV::random(256, i as u64))
            .collect();

        let result = orch.compute(&processes);
        assert!(result.confidence > 0.9);  // Real calculator has high confidence
    }
}
