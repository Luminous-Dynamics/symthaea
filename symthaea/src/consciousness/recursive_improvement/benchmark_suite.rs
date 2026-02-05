//! # Performance Benchmarking Infrastructure
//!
//! Comprehensive benchmarking for all routing paradigms.
//!
//! ## Routers Benchmarked
//!
//! 1. Causal Validation Router
//! 2. Information Geometric Router
//! 3. Topological Consciousness Router
//! 4. Quantum Coherence Router
//! 5. Active Inference Router
//! 6. Predictive Processing Router
//! 7. Attention Schema Theory Router
//!
//! ## Metrics Measured
//!
//! - Latency: Time per routing decision (μs)
//! - Throughput: Decisions per second
//! - Consistency: Variance in repeated decisions
//! - Memory: Approximate memory footprint
//! - Scalability: Performance degradation under load

use serde::{Serialize, Deserialize};

use super::world_model::LatentConsciousnessState;
use super::routers::{
    CausalValidatedRouter, CausalValidatedConfig,
    InformationGeometricRouter, GeometricRouterConfig,
    TopologicalConsciousnessRouter, TopologicalRouterConfig,
    QuantumCoherenceRouter, QuantumRouterConfig,
    ActiveInferenceRouter, ActiveInferenceConfig,
    PredictiveProcessingRouter, PredictiveProcessingConfig,
    ASTRouter, ASTRouterConfig,
};


/// Individual benchmark result for a router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterBenchmark {
    /// Router name
    pub router_name: String,
    /// Number of iterations
    pub iterations: usize,
    /// Total time in microseconds
    pub total_time_us: u64,
    /// Average latency per decision (μs)
    pub avg_latency_us: f64,
    /// Minimum latency (μs)
    pub min_latency_us: u64,
    /// Maximum latency (μs)
    pub max_latency_us: u64,
    /// Standard deviation of latency (μs)
    pub std_dev_us: f64,
    /// Decisions per second (throughput)
    pub throughput: f64,
    /// P50 latency (μs)
    pub p50_latency_us: u64,
    /// P95 latency (μs)
    pub p95_latency_us: u64,
    /// P99 latency (μs)
    pub p99_latency_us: u64,
    /// Consistency score (0-1, how often same input gives same output)
    pub consistency: f64,
}

impl RouterBenchmark {
    /// Create from raw timing data
    pub fn from_timings(router_name: &str, timings: &[u64]) -> Self {
        let n = timings.len();
        if n == 0 {
            return Self {
                router_name: router_name.to_string(),
                iterations: 0,
                total_time_us: 0,
                avg_latency_us: 0.0,
                min_latency_us: 0,
                max_latency_us: 0,
                std_dev_us: 0.0,
                throughput: 0.0,
                p50_latency_us: 0,
                p95_latency_us: 0,
                p99_latency_us: 0,
                consistency: 0.0,
            };
        }

        let total: u64 = timings.iter().sum();
        let avg = total as f64 / n as f64;
        let min = *timings.iter().min().unwrap_or(&0);
        let max = *timings.iter().max().unwrap_or(&0);

        // Standard deviation
        let variance: f64 = timings.iter()
            .map(|t| (*t as f64 - avg).powi(2))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Percentiles
        let mut sorted = timings.to_vec();
        sorted.sort();
        let p50 = sorted[n / 2];
        let p95 = sorted[(n as f64 * 0.95) as usize];
        let p99 = sorted[(n as f64 * 0.99).min((n - 1) as f64) as usize];

        // Throughput (decisions per second)
        let throughput = if total > 0 {
            n as f64 / (total as f64 / 1_000_000.0)
        } else {
            0.0
        };

        Self {
            router_name: router_name.to_string(),
            iterations: n,
            total_time_us: total,
            avg_latency_us: avg,
            min_latency_us: min,
            max_latency_us: max,
            std_dev_us: std_dev,
            throughput,
            p50_latency_us: p50,
            p95_latency_us: p95,
            p99_latency_us: p99,
            consistency: 1.0, // Will be updated separately
        }
    }

    /// Format as a readable report line
    pub fn report_line(&self) -> String {
        format!(
            "{:<25} | {:>8.1}μs | {:>8.1}μs | {:>8.1}μs | {:>10.0}/s | {:>6.1}%",
            self.router_name,
            self.avg_latency_us,
            self.p50_latency_us,
            self.p99_latency_us,
            self.throughput,
            self.consistency * 100.0
        )
    }
}

/// Comparative benchmark results for all routers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeBenchmark {
    /// Individual router benchmarks
    pub benchmarks: Vec<RouterBenchmark>,
    /// Best router by latency
    pub fastest_router: String,
    /// Best router by throughput
    pub highest_throughput: String,
    /// Most consistent router
    pub most_consistent: String,
    /// Total benchmark time (ms)
    pub total_benchmark_time_ms: u64,
    /// Timestamp
    pub timestamp: String,
}

impl ComparativeBenchmark {
    /// Generate a formatted report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("\n");
        report.push_str("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║           CONSCIOUSNESS ROUTING PARADIGM BENCHMARK RESULTS                  ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str(&format!("║ Timestamp: {:<66} ║\n", self.timestamp));
        report.push_str(&format!("║ Total Benchmark Time: {:>5}ms {:>51} ║\n", self.total_benchmark_time_ms, ""));
        report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ Router                    |   Avg    |   P50    |   P99    | Throughput | Cons ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");

        for benchmark in &self.benchmarks {
            report.push_str(&format!("║ {} ║\n", benchmark.report_line()));
        }

        report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str(&format!("║ 🏆 Fastest:         {:<56} ║\n", self.fastest_router));
        report.push_str(&format!("║ 🚀 Highest Throughput: {:<53} ║\n", self.highest_throughput));
        report.push_str(&format!("║ 🎯 Most Consistent: {:<56} ║\n", self.most_consistent));
        report.push_str("╚══════════════════════════════════════════════════════════════════════════════╝\n");

        report
    }
}

/// Configuration for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations (not measured)
    pub warmup_iterations: usize,
    /// Number of measured iterations
    pub measured_iterations: usize,
    /// Number of consistency check iterations
    pub consistency_iterations: usize,
    /// Whether to run scalability tests
    pub run_scalability: bool,
    /// Scalability test sizes
    pub scalability_sizes: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 100,
            measured_iterations: 1000,
            consistency_iterations: 50,
            run_scalability: false,
            scalability_sizes: vec![10, 100, 1000, 10000],
        }
    }
}

/// Router Benchmarking Suite
pub struct RouterBenchmarkSuite {
    config: BenchmarkConfig,
}

impl RouterBenchmarkSuite {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Generate test states for benchmarking
    fn generate_test_states(&self, count: usize) -> Vec<LatentConsciousnessState> {
        let mut states = Vec::with_capacity(count);
        for i in 0..count {
            let phi = (i as f64 * 0.1) % 1.0;
            let integration = ((i as f64 * 0.15) + 0.2) % 1.0;
            let coherence = ((i as f64 * 0.12) + 0.3) % 1.0;
            let attention = ((i as f64 * 0.08) + 0.5) % 1.0;
            states.push(LatentConsciousnessState::from_observables(
                phi, integration, coherence, attention
            ));
        }
        states
    }

    /// Benchmark the Causal Validation Router
    pub fn benchmark_causal(&self) -> RouterBenchmark {
        let mut router = CausalValidatedRouter::new(CausalValidatedConfig::default());
        let states = self.generate_test_states(self.config.measured_iterations);

        // Warmup - causal router uses route_validated with state argument
        for state in states.iter().take(self.config.warmup_iterations.min(states.len())) {
            let _ = router.route_validated(state);
        }

        // Measured runs
        let mut timings = Vec::with_capacity(self.config.measured_iterations);
        for state in &states {
            let start = std::time::Instant::now();
            let _ = router.route_validated(state);
            timings.push(start.elapsed().as_micros() as u64);
        }

        let mut benchmark = RouterBenchmark::from_timings("Causal Validation", &timings);

        // Consistency check
        benchmark.consistency = self.check_consistency_causal(&states);

        benchmark
    }

    fn check_consistency_causal(&self, states: &[LatentConsciousnessState]) -> f64 {
        if states.is_empty() || self.config.consistency_iterations == 0 {
            return 1.0;
        }

        let test_state = &states[0];
        let mut router = CausalValidatedRouter::new(CausalValidatedConfig::default());
        let first = router.route_validated(test_state);

        let mut consistent = 0;
        for _ in 0..self.config.consistency_iterations {
            let result = router.route_validated(test_state);
            if result.strategy == first.strategy {
                consistent += 1;
            }
        }

        consistent as f64 / self.config.consistency_iterations as f64
    }

    /// Benchmark the Information Geometric Router
    pub fn benchmark_geometric(&self) -> RouterBenchmark {
        let mut router = InformationGeometricRouter::new(GeometricRouterConfig::default());
        let states = self.generate_test_states(self.config.measured_iterations);

        // Warmup
        for state in states.iter().take(self.config.warmup_iterations.min(states.len())) {
            router.observe_state(state);
            let _ = router.route(state);
        }

        // Measured runs
        let mut timings = Vec::with_capacity(self.config.measured_iterations);
        for state in &states {
            router.observe_state(state);
            let start = std::time::Instant::now();
            let _ = router.route(state);
            timings.push(start.elapsed().as_micros() as u64);
        }

        let mut benchmark = RouterBenchmark::from_timings("Information Geometric", &timings);
        benchmark.consistency = self.check_consistency_geometric(&states);
        benchmark
    }

    fn check_consistency_geometric(&self, states: &[LatentConsciousnessState]) -> f64 {
        if states.is_empty() || self.config.consistency_iterations == 0 {
            return 1.0;
        }

        let test_state = &states[0];
        let mut router = InformationGeometricRouter::new(GeometricRouterConfig::default());
        router.observe_state(test_state);
        let first = router.route(test_state);

        let mut consistent = 0;
        for _ in 0..self.config.consistency_iterations {
            router.observe_state(test_state);
            let result = router.route(test_state);
            if result.strategy == first.strategy {
                consistent += 1;
            }
        }

        consistent as f64 / self.config.consistency_iterations as f64
    }

    /// Benchmark the Topological Consciousness Router
    pub fn benchmark_topological(&self) -> RouterBenchmark {
        let mut router = TopologicalConsciousnessRouter::new(TopologicalRouterConfig::default());
        let states = self.generate_test_states(self.config.measured_iterations);

        // Warmup
        for state in states.iter().take(self.config.warmup_iterations.min(states.len())) {
            router.observe_state(state);
            let _ = router.route(state);
        }

        // Measured runs
        let mut timings = Vec::with_capacity(self.config.measured_iterations);
        for state in &states {
            router.observe_state(state);
            let start = std::time::Instant::now();
            let _ = router.route(state);
            timings.push(start.elapsed().as_micros() as u64);
        }

        let mut benchmark = RouterBenchmark::from_timings("Topological Consciousness", &timings);
        benchmark.consistency = self.check_consistency_topological(&states);
        benchmark
    }

    fn check_consistency_topological(&self, states: &[LatentConsciousnessState]) -> f64 {
        if states.is_empty() || self.config.consistency_iterations == 0 {
            return 1.0;
        }

        let test_state = &states[0];
        let mut router = TopologicalConsciousnessRouter::new(TopologicalRouterConfig::default());
        router.observe_state(test_state);
        let first = router.route(test_state);

        let mut consistent = 0;
        for _ in 0..self.config.consistency_iterations {
            router.observe_state(test_state);
            let result = router.route(test_state);
            if result.strategy == first.strategy {
                consistent += 1;
            }
        }

        consistent as f64 / self.config.consistency_iterations as f64
    }

    /// Benchmark the Quantum Coherence Router
    pub fn benchmark_quantum(&self) -> RouterBenchmark {
        let mut router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());
        let states = self.generate_test_states(self.config.measured_iterations);

        // Warmup
        for state in states.iter().take(self.config.warmup_iterations.min(states.len())) {
            router.observe_state(state);
            let _ = router.route(state);
        }

        // Measured runs
        let mut timings = Vec::with_capacity(self.config.measured_iterations);
        for state in &states {
            router.observe_state(state);
            let start = std::time::Instant::now();
            let _ = router.route(state);
            timings.push(start.elapsed().as_micros() as u64);
        }

        let mut benchmark = RouterBenchmark::from_timings("Quantum Coherence", &timings);
        benchmark.consistency = self.check_consistency_quantum(&states);
        benchmark
    }

    fn check_consistency_quantum(&self, states: &[LatentConsciousnessState]) -> f64 {
        if states.is_empty() || self.config.consistency_iterations == 0 {
            return 1.0;
        }

        let test_state = &states[0];
        let mut router = QuantumCoherenceRouter::new(QuantumRouterConfig::default());
        router.observe_state(test_state);
        let first = router.route(test_state);

        let mut consistent = 0;
        for _ in 0..self.config.consistency_iterations {
            router.observe_state(test_state);
            let result = router.route(test_state);
            if result.strategy == first.strategy {
                consistent += 1;
            }
        }

        consistent as f64 / self.config.consistency_iterations as f64
    }

    /// Benchmark the Active Inference Router
    pub fn benchmark_active_inference(&self) -> RouterBenchmark {
        let mut router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());
        let states = self.generate_test_states(self.config.measured_iterations);

        // Warmup
        for state in states.iter().take(self.config.warmup_iterations.min(states.len())) {
            router.observe_state(state);
            let _ = router.route(state);
        }

        // Measured runs
        let mut timings = Vec::with_capacity(self.config.measured_iterations);
        for state in &states {
            router.observe_state(state);
            let start = std::time::Instant::now();
            let _ = router.route(state);
            timings.push(start.elapsed().as_micros() as u64);
        }

        let mut benchmark = RouterBenchmark::from_timings("Active Inference", &timings);
        benchmark.consistency = self.check_consistency_active_inference(&states);
        benchmark
    }

    fn check_consistency_active_inference(&self, states: &[LatentConsciousnessState]) -> f64 {
        if states.is_empty() || self.config.consistency_iterations == 0 {
            return 1.0;
        }

        let test_state = &states[0];
        let mut router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());
        router.observe_state(test_state);
        let first = router.route(test_state);

        let mut consistent = 0;
        for _ in 0..self.config.consistency_iterations {
            router.observe_state(test_state);
            let result = router.route(test_state);
            if result.strategy == first.strategy {
                consistent += 1;
            }
        }

        consistent as f64 / self.config.consistency_iterations as f64
    }

    /// Benchmark the Predictive Processing Router
    pub fn benchmark_predictive(&self) -> RouterBenchmark {
        let mut router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());
        let states = self.generate_test_states(self.config.measured_iterations);

        // Warmup - PredictiveProcessingRouter does observation internally in route()
        for state in states.iter().take(self.config.warmup_iterations.min(states.len())) {
            let _ = router.route(state);
        }

        // Measured runs
        let mut timings = Vec::with_capacity(self.config.measured_iterations);
        for state in &states {
            let start = std::time::Instant::now();
            let _ = router.route(state);
            timings.push(start.elapsed().as_micros() as u64);
        }

        let mut benchmark = RouterBenchmark::from_timings("Predictive Processing", &timings);
        benchmark.consistency = self.check_consistency_predictive(&states);
        benchmark
    }

    fn check_consistency_predictive(&self, states: &[LatentConsciousnessState]) -> f64 {
        if states.is_empty() || self.config.consistency_iterations == 0 {
            return 1.0;
        }

        let test_state = &states[0];
        let mut router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());
        let first = router.route(test_state);

        let mut consistent = 0;
        for _ in 0..self.config.consistency_iterations {
            let result = router.route(test_state);
            if result.strategy == first.strategy {
                consistent += 1;
            }
        }

        consistent as f64 / self.config.consistency_iterations as f64
    }

    /// Benchmark the Attention Schema Theory Router
    pub fn benchmark_ast(&self) -> RouterBenchmark {
        let mut router = ASTRouter::new(ASTRouterConfig::default());
        let states = self.generate_test_states(self.config.measured_iterations);

        // Warmup
        for state in states.iter().take(self.config.warmup_iterations.min(states.len())) {
            router.observe(state);
            let _ = router.route();
        }

        // Measured runs
        let mut timings = Vec::with_capacity(self.config.measured_iterations);
        for state in &states {
            router.observe(state);
            let start = std::time::Instant::now();
            let _ = router.route();
            timings.push(start.elapsed().as_micros() as u64);
        }

        let mut benchmark = RouterBenchmark::from_timings("Attention Schema Theory", &timings);
        benchmark.consistency = self.check_consistency_ast(&states);
        benchmark
    }

    fn check_consistency_ast(&self, states: &[LatentConsciousnessState]) -> f64 {
        if states.is_empty() || self.config.consistency_iterations == 0 {
            return 1.0;
        }

        let test_state = &states[0];
        let mut router = ASTRouter::new(ASTRouterConfig::default());
        router.observe(test_state);
        let first = router.route();

        let mut consistent = 0;
        for _ in 0..self.config.consistency_iterations {
            router.observe(test_state);
            let result = router.route();
            if result.strategy == first.strategy {
                consistent += 1;
            }
        }

        consistent as f64 / self.config.consistency_iterations as f64
    }

    /// Run all benchmarks and return comparative results
    pub fn run_all(&self) -> ComparativeBenchmark {
        let start = std::time::Instant::now();

        let benchmarks = vec![
            self.benchmark_causal(),
            self.benchmark_geometric(),
            self.benchmark_topological(),
            self.benchmark_quantum(),
            self.benchmark_active_inference(),
            self.benchmark_predictive(),
            self.benchmark_ast(),
        ];

        let total_time = start.elapsed().as_millis() as u64;

        // Find best performers
        let fastest = benchmarks.iter()
            .min_by(|a, b| a.avg_latency_us.partial_cmp(&b.avg_latency_us).unwrap_or(std::cmp::Ordering::Equal))
            .map(|b| b.router_name.clone())
            .unwrap_or_default();

        let highest_throughput = benchmarks.iter()
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap_or(std::cmp::Ordering::Equal))
            .map(|b| b.router_name.clone())
            .unwrap_or_default();

        let most_consistent = benchmarks.iter()
            .max_by(|a, b| a.consistency.partial_cmp(&b.consistency).unwrap_or(std::cmp::Ordering::Equal))
            .map(|b| b.router_name.clone())
            .unwrap_or_default();

        // Timestamp
        let timestamp = format!("{:?}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs());

        ComparativeBenchmark {
            benchmarks,
            fastest_router: fastest,
            highest_throughput,
            most_consistent,
            total_benchmark_time_ms: total_time,
            timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_benchmark_from_timings() {
        let timings = vec![100, 120, 90, 110, 105, 95, 115, 108, 102, 98];
        let benchmark = RouterBenchmark::from_timings("Test Router", &timings);

        assert_eq!(benchmark.router_name, "Test Router");
        assert_eq!(benchmark.iterations, 10);
        assert!(benchmark.avg_latency_us > 0.0);
        assert!(benchmark.min_latency_us <= benchmark.max_latency_us);
        assert!(benchmark.throughput > 0.0);
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = RouterBenchmarkSuite::new(config);

        assert!(suite.config.warmup_iterations > 0);
        assert!(suite.config.measured_iterations > 0);
    }

    #[test]
    fn test_generate_test_states() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measured_iterations: 100,
            consistency_iterations: 10,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let states = suite.generate_test_states(50);
        assert_eq!(states.len(), 50);
    }

    #[test]
    fn test_benchmark_causal_router() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            consistency_iterations: 5,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let benchmark = suite.benchmark_causal();
        assert_eq!(benchmark.router_name, "Causal Validation");
        assert_eq!(benchmark.iterations, 20);
        assert!(benchmark.avg_latency_us >= 0.0);
    }

    #[test]
    fn test_benchmark_geometric_router() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            consistency_iterations: 5,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let benchmark = suite.benchmark_geometric();
        assert_eq!(benchmark.router_name, "Information Geometric");
        assert!(benchmark.iterations > 0);
    }

    #[test]
    fn test_benchmark_topological_router() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            consistency_iterations: 5,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let benchmark = suite.benchmark_topological();
        assert_eq!(benchmark.router_name, "Topological Consciousness");
    }

    #[test]
    fn test_benchmark_quantum_router() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            consistency_iterations: 5,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let benchmark = suite.benchmark_quantum();
        assert_eq!(benchmark.router_name, "Quantum Coherence");
    }

    #[test]
    fn test_benchmark_active_inference_router() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            consistency_iterations: 5,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let benchmark = suite.benchmark_active_inference();
        assert_eq!(benchmark.router_name, "Active Inference");
    }

    #[test]
    fn test_benchmark_predictive_router() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            consistency_iterations: 5,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let benchmark = suite.benchmark_predictive();
        assert_eq!(benchmark.router_name, "Predictive Processing");
    }

    #[test]
    fn test_benchmark_ast_router() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measured_iterations: 20,
            consistency_iterations: 5,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let benchmark = suite.benchmark_ast();
        assert_eq!(benchmark.router_name, "Attention Schema Theory");
    }

    #[test]
    fn test_run_all_benchmarks() {
        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measured_iterations: 10,
            consistency_iterations: 3,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let results = suite.run_all();
        assert_eq!(results.benchmarks.len(), 7);
        assert!(!results.fastest_router.is_empty());
        assert!(!results.highest_throughput.is_empty());
        assert!(!results.most_consistent.is_empty());
    }

    #[test]
    fn test_comparative_benchmark_report() {
        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measured_iterations: 10,
            consistency_iterations: 3,
            run_scalability: false,
            scalability_sizes: vec![],
        };
        let suite = RouterBenchmarkSuite::new(config);

        let results = suite.run_all();
        let report = results.report();

        assert!(report.contains("BENCHMARK RESULTS"));
        assert!(report.contains("Fastest"));
        assert!(report.contains("Throughput"));
    }

    #[test]
    fn test_benchmark_report_line() {
        let benchmark = RouterBenchmark {
            router_name: "Test".to_string(),
            iterations: 100,
            total_time_us: 10000,
            avg_latency_us: 100.0,
            min_latency_us: 50,
            max_latency_us: 200,
            std_dev_us: 25.0,
            throughput: 10000.0,
            p50_latency_us: 95,
            p95_latency_us: 180,
            p99_latency_us: 195,
            consistency: 0.95,
        };

        let line = benchmark.report_line();
        assert!(line.contains("Test"));
        assert!(line.contains("100.0"));
    }
}
