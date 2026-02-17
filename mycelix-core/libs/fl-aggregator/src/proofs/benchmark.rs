//! Comprehensive Benchmark Suite for Proof System
//!
//! Provides detailed performance profiling including:
//! - Generation and verification timings
//! - Memory usage tracking
//! - Throughput measurements
//! - Comparative analysis across security levels
//! - Statistical analysis with percentiles

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::proofs::{
    GradientIntegrityProof, IdentityAssuranceProof, ProofAssuranceLevel, ProofConfig,
    ProofIdentityFactor, ProofProposalType, ProofType, ProofVoterProfile, RangeProof,
    SecurityLevel, VoteEligibilityProof,
};

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Configuration for benchmark runs
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    /// Number of iterations per test
    pub iterations: usize,
    /// Warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Security levels to test
    pub security_levels: Vec<SecurityLevel>,
    /// Proof types to benchmark
    pub proof_types: Vec<ProofType>,
    /// Whether to track memory usage
    pub track_memory: bool,
    /// Gradient sizes to test
    pub gradient_sizes: Vec<usize>,
    /// Whether to run in parallel
    pub parallel: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            warmup_iterations: 10,
            security_levels: vec![
                SecurityLevel::Standard96,
                SecurityLevel::Standard128,
                SecurityLevel::High256,
            ],
            proof_types: vec![
                ProofType::Range,
                ProofType::GradientIntegrity,
                ProofType::IdentityAssurance,
                ProofType::VoteEligibility,
            ],
            track_memory: true,
            gradient_sizes: vec![100, 1000, 10000],
            parallel: true,
        }
    }
}

impl BenchmarkConfig {
    /// Create a quick config for fast testing
    pub fn quick() -> Self {
        Self {
            iterations: 10,
            warmup_iterations: 2,
            security_levels: vec![SecurityLevel::Standard128],
            gradient_sizes: vec![100],
            ..Default::default()
        }
    }

    /// Create a thorough config for comprehensive testing
    pub fn thorough() -> Self {
        Self {
            iterations: 500,
            warmup_iterations: 50,
            gradient_sizes: vec![100, 1000, 10000, 100000],
            ..Default::default()
        }
    }
}

// ============================================================================
// Timing Results
// ============================================================================

/// Statistics for a set of measurements
#[derive(Clone, Debug)]
pub struct TimingStats {
    /// Number of samples
    pub count: usize,
    /// Minimum time in microseconds
    pub min_us: u64,
    /// Maximum time in microseconds
    pub max_us: u64,
    /// Mean time in microseconds
    pub mean_us: f64,
    /// Median time in microseconds
    pub median_us: u64,
    /// Standard deviation in microseconds
    pub std_dev_us: f64,
    /// 95th percentile in microseconds
    pub p95_us: u64,
    /// 99th percentile in microseconds
    pub p99_us: u64,
    /// Total time in microseconds
    pub total_us: u64,
    /// Operations per second
    pub ops_per_sec: f64,
}

impl TimingStats {
    /// Calculate statistics from a list of durations
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self {
                count: 0,
                min_us: 0,
                max_us: 0,
                mean_us: 0.0,
                median_us: 0,
                std_dev_us: 0.0,
                p95_us: 0,
                p99_us: 0,
                total_us: 0,
                ops_per_sec: 0.0,
            };
        }

        let mut micros: Vec<u64> = durations.iter().map(|d| d.as_micros() as u64).collect();
        micros.sort_unstable();

        let count = micros.len();
        let total_us: u64 = micros.iter().sum();
        let mean_us = total_us as f64 / count as f64;

        // Calculate standard deviation
        let variance: f64 = micros
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean_us;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev_us = variance.sqrt();

        // Percentiles
        let p95_idx = (count as f64 * 0.95) as usize;
        let p99_idx = (count as f64 * 0.99) as usize;

        Self {
            count,
            min_us: micros[0],
            max_us: micros[count - 1],
            mean_us,
            median_us: micros[count / 2],
            std_dev_us,
            p95_us: micros[p95_idx.min(count - 1)],
            p99_us: micros[p99_idx.min(count - 1)],
            total_us,
            ops_per_sec: 1_000_000.0 / mean_us,
        }
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "mean: {:.2}ms, median: {:.2}ms, p95: {:.2}ms, p99: {:.2}ms ({:.1} ops/s)",
            self.mean_us as f64 / 1000.0,
            self.median_us as f64 / 1000.0,
            self.p95_us as f64 / 1000.0,
            self.p99_us as f64 / 1000.0,
            self.ops_per_sec
        )
    }
}

// ============================================================================
// Memory Tracking
// ============================================================================

/// Memory usage statistics
#[derive(Clone, Debug, Default)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Average memory usage in bytes
    pub avg_bytes: usize,
    /// Proof size in bytes
    pub proof_size_bytes: usize,
    /// Public inputs size in bytes
    pub public_inputs_bytes: usize,
}

impl MemoryStats {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "proof: {} bytes, peak mem: {} KB",
            self.proof_size_bytes,
            self.peak_bytes / 1024
        )
    }
}

// ============================================================================
// Benchmark Results
// ============================================================================

/// Results for a single proof type benchmark
#[derive(Clone, Debug)]
pub struct ProofBenchmarkResult {
    /// Proof type
    pub proof_type: ProofType,
    /// Security level
    pub security_level: SecurityLevel,
    /// Generation timing statistics
    pub generation: TimingStats,
    /// Verification timing statistics
    pub verification: TimingStats,
    /// Memory statistics
    pub memory: MemoryStats,
    /// Additional parameters (e.g., gradient size)
    pub parameters: HashMap<String, String>,
}

impl ProofBenchmarkResult {
    /// Format as summary string
    pub fn summary(&self) -> String {
        format!(
            "{:?} @ {:?}\n  gen: {}\n  ver: {}\n  mem: {}",
            self.proof_type,
            self.security_level,
            self.generation.format(),
            self.verification.format(),
            self.memory.format()
        )
    }
}

/// Complete benchmark suite results
#[derive(Clone, Debug)]
pub struct BenchmarkResults {
    /// Configuration used
    pub config: BenchmarkConfig,
    /// Individual proof results
    pub results: Vec<ProofBenchmarkResult>,
    /// Total benchmark duration
    pub total_duration: Duration,
    /// System information
    pub system_info: SystemInfo,
}

impl BenchmarkResults {
    /// Generate a full report
    pub fn report(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Proof System Benchmark Report ===\n\n");
        output.push_str(&format!("System: {}\n", self.system_info.format()));
        output.push_str(&format!(
            "Total Duration: {:.2}s\n",
            self.total_duration.as_secs_f64()
        ));
        output.push_str(&format!("Iterations: {}\n\n", self.config.iterations));

        // Group by proof type
        let mut by_type: HashMap<ProofType, Vec<&ProofBenchmarkResult>> = HashMap::new();
        for result in &self.results {
            by_type.entry(result.proof_type).or_default().push(result);
        }

        for (proof_type, results) in by_type {
            output.push_str(&format!("\n=== {:?} ===\n", proof_type));
            for result in results {
                output.push_str(&format!("\n{}\n", result.summary()));
            }
        }

        output
    }

    /// Export as JSON
    pub fn to_json(&self) -> serde_json::Value {
        let results: Vec<serde_json::Value> = self
            .results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "proof_type": format!("{:?}", r.proof_type),
                    "security_level": format!("{:?}", r.security_level),
                    "generation": {
                        "mean_ms": r.generation.mean_us / 1000.0,
                        "median_ms": r.generation.median_us as f64 / 1000.0,
                        "p95_ms": r.generation.p95_us as f64 / 1000.0,
                        "p99_ms": r.generation.p99_us as f64 / 1000.0,
                        "ops_per_sec": r.generation.ops_per_sec,
                    },
                    "verification": {
                        "mean_ms": r.verification.mean_us / 1000.0,
                        "median_ms": r.verification.median_us as f64 / 1000.0,
                        "p95_ms": r.verification.p95_us as f64 / 1000.0,
                        "p99_ms": r.verification.p99_us as f64 / 1000.0,
                        "ops_per_sec": r.verification.ops_per_sec,
                    },
                    "memory": {
                        "proof_size_bytes": r.memory.proof_size_bytes,
                        "peak_bytes": r.memory.peak_bytes,
                    },
                    "parameters": r.parameters,
                })
            })
            .collect();

        serde_json::json!({
            "config": {
                "iterations": self.config.iterations,
                "warmup_iterations": self.config.warmup_iterations,
            },
            "system": {
                "os": self.system_info.os,
                "cpu": self.system_info.cpu,
                "cores": self.system_info.cores,
            },
            "total_duration_secs": self.total_duration.as_secs_f64(),
            "results": results,
        })
    }

    /// Get comparative analysis between security levels
    pub fn compare_security_levels(&self, proof_type: ProofType) -> String {
        let mut output = String::new();
        output.push_str(&format!("\n=== {:?} Security Level Comparison ===\n", proof_type));

        let results: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.proof_type == proof_type)
            .collect();

        if let Some(baseline) = results.iter().find(|r| r.security_level == SecurityLevel::Standard128) {
            for result in &results {
                let gen_ratio = result.generation.mean_us as f64 / baseline.generation.mean_us;
                let size_ratio = result.memory.proof_size_bytes as f64
                    / baseline.memory.proof_size_bytes.max(1) as f64;

                output.push_str(&format!(
                    "{:?}: {:.1}x gen time, {:.1}x proof size vs Standard128\n",
                    result.security_level, gen_ratio, size_ratio
                ));
            }
        }

        output
    }
}

/// System information for benchmark context
#[derive(Clone, Debug)]
pub struct SystemInfo {
    pub os: String,
    pub cpu: String,
    pub cores: usize,
    pub memory_gb: f64,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            cpu: std::env::consts::ARCH.to_string(),
            cores: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            memory_gb: 0.0, // Would need platform-specific code to get actual RAM
        }
    }
}

impl SystemInfo {
    pub fn format(&self) -> String {
        format!("{} {} ({} cores)", self.os, self.cpu, self.cores)
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Benchmark runner for executing benchmark suites
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    results: Arc<RwLock<Vec<ProofBenchmarkResult>>>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Run all benchmarks
    pub async fn run_all(&self) -> BenchmarkResults {
        let start = Instant::now();

        for &security_level in &self.config.security_levels {
            for &proof_type in &self.config.proof_types {
                match proof_type {
                    ProofType::Range => {
                        self.benchmark_range(security_level).await;
                    }
                    ProofType::GradientIntegrity => {
                        for &size in &self.config.gradient_sizes {
                            self.benchmark_gradient(security_level, size).await;
                        }
                    }
                    ProofType::IdentityAssurance => {
                        self.benchmark_identity(security_level).await;
                    }
                    ProofType::VoteEligibility => {
                        self.benchmark_vote(security_level).await;
                    }
                    ProofType::Membership => {
                        // Skip membership for now - requires Merkle tree setup
                    }
                }
            }
        }

        BenchmarkResults {
            config: self.config.clone(),
            results: self.results.read().await.clone(),
            total_duration: start.elapsed(),
            system_info: SystemInfo::default(),
        }
    }

    /// Benchmark range proof
    async fn benchmark_range(&self, security_level: SecurityLevel) {
        let config = ProofConfig {
            security_level,
            parallel: self.config.parallel,
            max_proof_size: 0,
        };

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = RangeProof::generate(50, 0, 100, config.clone());
        }

        // Measure generation
        let mut gen_times = Vec::with_capacity(self.config.iterations);
        let mut proof_size = 0;

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let proof = RangeProof::generate(50, 0, 100, config.clone()).unwrap();
            gen_times.push(start.elapsed());

            if proof_size == 0 {
                proof_size = proof.to_bytes().len();
            }
        }

        // Measure verification
        let proof = RangeProof::generate(50, 0, 100, config.clone()).unwrap();
        let mut ver_times = Vec::with_capacity(self.config.iterations);

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = proof.verify();
            ver_times.push(start.elapsed());
        }

        let result = ProofBenchmarkResult {
            proof_type: ProofType::Range,
            security_level,
            generation: TimingStats::from_durations(&gen_times),
            verification: TimingStats::from_durations(&ver_times),
            memory: MemoryStats {
                proof_size_bytes: proof_size,
                ..Default::default()
            },
            parameters: HashMap::new(),
        };

        self.results.write().await.push(result);
    }

    /// Benchmark gradient integrity proof
    async fn benchmark_gradient(&self, security_level: SecurityLevel, gradient_size: usize) {
        let config = ProofConfig {
            security_level,
            parallel: self.config.parallel,
            max_proof_size: 0,
        };

        // Generate test gradients
        let gradients: Vec<f32> = (0..gradient_size)
            .map(|i| ((i as f32) / 1000.0).sin() * 0.1)
            .collect();
        let max_norm = 5.0;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = GradientIntegrityProof::generate(&gradients, max_norm, config.clone());
        }

        // Measure generation
        let mut gen_times = Vec::with_capacity(self.config.iterations);
        let mut proof_size = 0;

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let proof = GradientIntegrityProof::generate(&gradients, max_norm, config.clone()).unwrap();
            gen_times.push(start.elapsed());

            if proof_size == 0 {
                proof_size = proof.to_bytes().len();
            }
        }

        // Measure verification
        let proof = GradientIntegrityProof::generate(&gradients, max_norm, config.clone()).unwrap();
        let mut ver_times = Vec::with_capacity(self.config.iterations);

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = proof.verify();
            ver_times.push(start.elapsed());
        }

        let mut parameters = HashMap::new();
        parameters.insert("gradient_size".to_string(), gradient_size.to_string());

        let result = ProofBenchmarkResult {
            proof_type: ProofType::GradientIntegrity,
            security_level,
            generation: TimingStats::from_durations(&gen_times),
            verification: TimingStats::from_durations(&ver_times),
            memory: MemoryStats {
                proof_size_bytes: proof_size,
                ..Default::default()
            },
            parameters,
        };

        self.results.write().await.push(result);
    }

    /// Benchmark identity assurance proof
    async fn benchmark_identity(&self, security_level: SecurityLevel) {
        let config = ProofConfig {
            security_level,
            parallel: self.config.parallel,
            max_proof_size: 0,
        };

        let factors = vec![
            ProofIdentityFactor::new(0.5, 0, true),
            ProofIdentityFactor::new(0.3, 1, true),
            ProofIdentityFactor::new(0.2, 2, true),
        ];
        let did = "did:mycelix:benchmark";

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = IdentityAssuranceProof::generate(
                did,
                &factors,
                ProofAssuranceLevel::E2,
                config.clone(),
            );
        }

        // Measure generation
        let mut gen_times = Vec::with_capacity(self.config.iterations);
        let mut proof_size = 0;

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let proof = IdentityAssuranceProof::generate(
                did,
                &factors,
                ProofAssuranceLevel::E2,
                config.clone(),
            )
            .unwrap();
            gen_times.push(start.elapsed());

            if proof_size == 0 {
                proof_size = proof.to_bytes().len();
            }
        }

        // Measure verification
        let proof = IdentityAssuranceProof::generate(
            did,
            &factors,
            ProofAssuranceLevel::E2,
            config.clone(),
        )
        .unwrap();
        let mut ver_times = Vec::with_capacity(self.config.iterations);

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = proof.verify();
            ver_times.push(start.elapsed());
        }

        let result = ProofBenchmarkResult {
            proof_type: ProofType::IdentityAssurance,
            security_level,
            generation: TimingStats::from_durations(&gen_times),
            verification: TimingStats::from_durations(&ver_times),
            memory: MemoryStats {
                proof_size_bytes: proof_size,
                ..Default::default()
            },
            parameters: HashMap::new(),
        };

        self.results.write().await.push(result);
    }

    /// Benchmark vote eligibility proof
    async fn benchmark_vote(&self, security_level: SecurityLevel) {
        let config = ProofConfig {
            security_level,
            parallel: self.config.parallel,
            max_proof_size: 0,
        };

        let voter = ProofVoterProfile {
            did: "did:mycelix:voter".to_string(),
            assurance_level: 3,
            matl_score: 0.8,
            stake: 1000.0,
            account_age_days: 365,
            participation_rate: 0.7,
            has_humanity_proof: true,
            fl_contributions: 50,
        };

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = VoteEligibilityProof::generate(
                &voter,
                ProofProposalType::Constitutional,
                config.clone(),
            );
        }

        // Measure generation
        let mut gen_times = Vec::with_capacity(self.config.iterations);
        let mut proof_size = 0;

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let proof = VoteEligibilityProof::generate(
                &voter,
                ProofProposalType::Constitutional,
                config.clone(),
            )
            .unwrap();
            gen_times.push(start.elapsed());

            if proof_size == 0 {
                proof_size = proof.to_bytes().len();
            }
        }

        // Measure verification
        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Constitutional,
            config.clone(),
        )
        .unwrap();
        let mut ver_times = Vec::with_capacity(self.config.iterations);

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            let _ = proof.verify();
            ver_times.push(start.elapsed());
        }

        let result = ProofBenchmarkResult {
            proof_type: ProofType::VoteEligibility,
            security_level,
            generation: TimingStats::from_durations(&gen_times),
            verification: TimingStats::from_durations(&ver_times),
            memory: MemoryStats {
                proof_size_bytes: proof_size,
                ..Default::default()
            },
            parameters: HashMap::new(),
        };

        self.results.write().await.push(result);
    }
}

// ============================================================================
// Throughput Testing
// ============================================================================

/// Throughput test configuration
#[derive(Clone, Debug)]
pub struct ThroughputConfig {
    /// Test duration
    pub duration: Duration,
    /// Number of concurrent tasks
    pub concurrency: usize,
    /// Proof type to test
    pub proof_type: ProofType,
    /// Security level
    pub security_level: SecurityLevel,
}

impl Default for ThroughputConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(10),
            concurrency: 4,
            proof_type: ProofType::Range,
            security_level: SecurityLevel::Standard128,
        }
    }
}

/// Throughput test results
#[derive(Clone, Debug)]
pub struct ThroughputResult {
    /// Configuration used
    pub config: ThroughputConfig,
    /// Total operations completed
    pub total_ops: u64,
    /// Operations per second
    pub ops_per_sec: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// Errors encountered
    pub errors: u64,
}

/// Run throughput test
pub async fn run_throughput_test(config: ThroughputConfig) -> ThroughputResult {
    let ops = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let errors = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let latencies = Arc::new(RwLock::new(Vec::new()));

    let proof_config = ProofConfig {
        security_level: config.security_level,
        parallel: true,
        max_proof_size: 0,
    };

    let deadline = Instant::now() + config.duration;

    let mut handles = Vec::new();

    for _ in 0..config.concurrency {
        let ops = Arc::clone(&ops);
        let errors = Arc::clone(&errors);
        let latencies = Arc::clone(&latencies);
        let proof_type = config.proof_type;
        let proof_config = proof_config.clone();

        let handle = tokio::spawn(async move {
            while Instant::now() < deadline {
                let start = Instant::now();

                let result = match proof_type {
                    ProofType::Range => RangeProof::generate(50, 0, 100, proof_config.clone())
                        .map(|_| ()),
                    ProofType::GradientIntegrity => {
                        let gradients: Vec<f32> = (0..100).map(|i| (i as f32) * 0.001).collect();
                        GradientIntegrityProof::generate(&gradients, 5.0, proof_config.clone())
                            .map(|_| ())
                    }
                    ProofType::IdentityAssurance => {
                        let factors = vec![ProofIdentityFactor::new(0.5, 0, true)];
                        IdentityAssuranceProof::generate(
                            "did:test",
                            &factors,
                            ProofAssuranceLevel::E1,
                            proof_config.clone(),
                        )
                        .map(|_| ())
                    }
                    ProofType::VoteEligibility => {
                        let voter = ProofVoterProfile {
                            did: "did:test".to_string(),
                            assurance_level: 2,
                            matl_score: 0.5,
                            stake: 100.0,
                            account_age_days: 30,
                            participation_rate: 0.5,
                            has_humanity_proof: true,
                            fl_contributions: 5,
                        };
                        VoteEligibilityProof::generate(
                            &voter,
                            ProofProposalType::Standard,
                            proof_config.clone(),
                        )
                        .map(|_| ())
                    }
                    ProofType::Membership => Ok(()), // Skip
                };

                let elapsed = start.elapsed();

                match result {
                    Ok(()) => {
                        ops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        latencies.write().await.push(elapsed.as_micros() as u64);
                    }
                    Err(_) => {
                        errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        let _ = handle.await;
    }

    let total_ops = ops.load(std::sync::atomic::Ordering::Relaxed);
    let total_errors = errors.load(std::sync::atomic::Ordering::Relaxed);
    let latency_samples = latencies.read().await;
    let avg_latency_us = if !latency_samples.is_empty() {
        latency_samples.iter().sum::<u64>() as f64 / latency_samples.len() as f64
    } else {
        0.0
    };

    ThroughputResult {
        config: config.clone(),
        total_ops,
        ops_per_sec: total_ops as f64 / config.duration.as_secs_f64(),
        avg_latency_us,
        errors: total_errors,
    }
}

// ============================================================================
// Comparative Benchmarks
// ============================================================================

/// Compare proof types at same security level
pub async fn compare_proof_types(security_level: SecurityLevel, iterations: usize) -> String {
    let runner = BenchmarkRunner::new(BenchmarkConfig {
        iterations,
        warmup_iterations: iterations / 10,
        security_levels: vec![security_level],
        gradient_sizes: vec![1000],
        ..Default::default()
    });

    let results = runner.run_all().await;

    let mut output = String::new();
    output.push_str(&format!(
        "=== Proof Type Comparison at {:?} ===\n\n",
        security_level
    ));
    output.push_str("| Proof Type | Gen (ms) | Ver (ms) | Size (KB) | Ops/s |\n");
    output.push_str("|------------|----------|----------|-----------|-------|\n");

    for result in &results.results {
        output.push_str(&format!(
            "| {:?} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
            result.proof_type,
            result.generation.mean_us as f64 / 1000.0,
            result.verification.mean_us as f64 / 1000.0,
            result.memory.proof_size_bytes as f64 / 1024.0,
            result.generation.ops_per_sec
        ));
    }

    output
}

/// Compare security levels for same proof type
pub async fn compare_security_levels(proof_type: ProofType, iterations: usize) -> String {
    let runner = BenchmarkRunner::new(BenchmarkConfig {
        iterations,
        warmup_iterations: iterations / 10,
        proof_types: vec![proof_type],
        gradient_sizes: vec![1000],
        ..Default::default()
    });

    let results = runner.run_all().await;
    results.compare_security_levels(proof_type)
}

// ============================================================================
// Proof of Gradient Quality (PoGQ) Benchmarks
// ============================================================================
// NOTE: This section requires the `http-api` feature because ProofOfGradientQuality
// and GradientStatistics are defined in the api module.

#[cfg(feature = "http-api")]
use crate::proofs::api::{ProofOfGradientQuality, GradientStatistics};

/// Configuration for PoGQ benchmarks
#[cfg(feature = "http-api")]
#[derive(Clone, Debug)]
pub struct PoGQBenchmarkConfig {
    /// Number of iterations
    pub iterations: usize,
    /// Warmup iterations
    pub warmup: usize,
    /// Gradient sizes to test
    pub gradient_sizes: Vec<usize>,
    /// Security levels to test
    pub security_levels: Vec<SecurityLevel>,
    /// Maximum gradient norm
    pub max_norm: f32,
}

#[cfg(feature = "http-api")]
impl Default for PoGQBenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            warmup: 5,
            gradient_sizes: vec![100, 1000, 10000],
            security_levels: vec![
                SecurityLevel::Standard96,
                SecurityLevel::Standard128,
            ],
            max_norm: 100.0,
        }
    }
}

/// Results from PoGQ benchmarking
#[cfg(feature = "http-api")]
#[derive(Clone, Debug)]
pub struct PoGQBenchmarkResult {
    /// Gradient size
    pub gradient_size: usize,
    /// Security level
    pub security_level: SecurityLevel,
    /// Generation timing stats
    pub generation: TimingStats,
    /// Verification timing stats
    pub verification: TimingStats,
    /// Serialization timing stats
    pub serialization: TimingStats,
    /// Deserialization timing stats
    pub deserialization: TimingStats,
    /// Proof size in bytes
    pub proof_size_bytes: usize,
    /// Statistics computation time
    pub stats_time_us: u64,
}

/// Run comprehensive PoGQ benchmarks
#[cfg(feature = "http-api")]
pub fn run_pogq_benchmarks(config: PoGQBenchmarkConfig) -> Vec<PoGQBenchmarkResult> {
    let mut results = Vec::new();

    for &size in &config.gradient_sizes {
        for &security_level in &config.security_levels {
            let result = benchmark_pogq_single(
                size,
                security_level,
                config.max_norm,
                config.iterations,
                config.warmup,
            );
            results.push(result);
        }
    }

    results
}

/// Benchmark a single PoGQ configuration
#[cfg(feature = "http-api")]
fn benchmark_pogq_single(
    gradient_size: usize,
    security_level: SecurityLevel,
    max_norm: f32,
    iterations: usize,
    warmup: usize,
) -> PoGQBenchmarkResult {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    // Generate random gradient
    let gradients: Vec<f32> = (0..gradient_size)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let proof_config = ProofConfig {
        security_level,
        parallel: false,
        max_proof_size: 0,
    };

    // Warmup
    for _ in 0..warmup {
        let _ = ProofOfGradientQuality::generate(
            &gradients,
            max_norm,
            [0u8; 32],
            "bench_node",
            0,
            proof_config.clone(),
        );
    }

    // Benchmark statistics computation
    let stats_start = std::time::Instant::now();
    let _stats = GradientStatistics::from_gradient(&gradients);
    let stats_time_us = stats_start.elapsed().as_micros() as u64;

    // Benchmark generation
    let mut gen_times = Vec::with_capacity(iterations);
    let mut proofs = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = std::time::Instant::now();
        let proof = ProofOfGradientQuality::generate(
            &gradients,
            max_norm,
            [i as u8; 32],
            "bench_node",
            i as u32,
            proof_config.clone(),
        ).expect("Proof generation failed");
        gen_times.push(start.elapsed());
        proofs.push(proof);
    }

    // Get proof size from first proof
    let proof_size_bytes = proofs[0].size();

    // Benchmark serialization
    let mut ser_times = Vec::with_capacity(iterations);
    let mut serialized = Vec::with_capacity(iterations);

    for proof in &proofs {
        let start = std::time::Instant::now();
        let bytes = proof.serialize().expect("Serialization failed");
        ser_times.push(start.elapsed());
        serialized.push(bytes);
    }

    // Benchmark deserialization
    let mut deser_times = Vec::with_capacity(iterations);

    for bytes in &serialized {
        let start = std::time::Instant::now();
        let _ = ProofOfGradientQuality::deserialize(bytes).expect("Deserialization failed");
        deser_times.push(start.elapsed());
    }

    // Benchmark verification
    let mut ver_times = Vec::with_capacity(iterations);

    for proof in &proofs {
        let start = std::time::Instant::now();
        let _ = proof.verify().expect("Verification failed");
        ver_times.push(start.elapsed());
    }

    PoGQBenchmarkResult {
        gradient_size,
        security_level,
        generation: TimingStats::from_durations(&gen_times),
        verification: TimingStats::from_durations(&ver_times),
        serialization: TimingStats::from_durations(&ser_times),
        deserialization: TimingStats::from_durations(&deser_times),
        proof_size_bytes,
        stats_time_us,
    }
}

/// Format PoGQ benchmark results as a table
#[cfg(feature = "http-api")]
pub fn format_pogq_results(results: &[PoGQBenchmarkResult]) -> String {
    let mut output = String::new();
    output.push_str("=== Proof of Gradient Quality Benchmark Results ===\n\n");
    output.push_str("| Size | Security | Gen (ms) | Ver (ms) | Ser (us) | Deser (us) | Size (KB) |\n");
    output.push_str("|------|----------|----------|----------|----------|------------|----------|\n");

    for r in results {
        output.push_str(&format!(
            "| {:>5} | {:?} | {:>8.2} | {:>8.2} | {:>8.0} | {:>10.0} | {:>8.1} |\n",
            r.gradient_size,
            r.security_level,
            r.generation.mean_us as f64 / 1000.0,
            r.verification.mean_us as f64 / 1000.0,
            r.serialization.mean_us as f64,
            r.deserialization.mean_us as f64,
            r.proof_size_bytes as f64 / 1024.0,
        ));
    }

    output.push_str("\n### Performance Insights\n\n");

    // Find best and worst
    if let Some(fastest) = results.iter().min_by_key(|r| r.generation.mean_us as u64) {
        output.push_str(&format!(
            "- **Fastest generation**: {} elements @ {:?} = {:.2}ms\n",
            fastest.gradient_size,
            fastest.security_level,
            fastest.generation.mean_us / 1000.0
        ));
    }

    if let Some(smallest) = results.iter().min_by_key(|r| r.proof_size_bytes) {
        output.push_str(&format!(
            "- **Smallest proof**: {} elements @ {:?} = {:.1}KB\n",
            smallest.gradient_size,
            smallest.security_level,
            smallest.proof_size_bytes as f64 / 1024.0
        ));
    }

    // Throughput estimates
    if let Some(r) = results.first() {
        let gen_per_sec = 1_000_000.0 / r.generation.mean_us;
        let ver_per_sec = 1_000_000.0 / r.verification.mean_us;
        output.push_str(&format!(
            "\n### Throughput Estimates (single-threaded)\n\n"));
        output.push_str(&format!(
            "- Proof generation: {:.1} proofs/sec\n",
            gen_per_sec
        ));
        output.push_str(&format!(
            "- Proof verification: {:.1} proofs/sec\n",
            ver_per_sec
        ));
    }

    output
}

/// Quick PoGQ benchmark for testing
#[cfg(feature = "http-api")]
pub fn quick_pogq_benchmark() -> String {
    let config = PoGQBenchmarkConfig {
        iterations: 10,
        warmup: 2,
        gradient_sizes: vec![100, 1000],
        security_levels: vec![SecurityLevel::Standard96],
        ..Default::default()
    };

    let results = run_pogq_benchmarks(config);
    format_pogq_results(&results)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_stats_empty() {
        let stats = TimingStats::from_durations(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean_us, 0.0);
    }

    #[test]
    fn test_timing_stats_single() {
        let durations = vec![Duration::from_micros(1000)];
        let stats = TimingStats::from_durations(&durations);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.min_us, 1000);
        assert_eq!(stats.max_us, 1000);
        assert!((stats.mean_us - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_timing_stats_multiple() {
        let durations: Vec<Duration> = (1..=100)
            .map(|i| Duration::from_micros(i * 10))
            .collect();
        let stats = TimingStats::from_durations(&durations);

        assert_eq!(stats.count, 100);
        assert_eq!(stats.min_us, 10);
        assert_eq!(stats.max_us, 1000);
        assert!(stats.p95_us >= stats.median_us);
        assert!(stats.p99_us >= stats.p95_us);
    }

    #[test]
    fn test_benchmark_config_defaults() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.iterations, 100);
        assert!(config.security_levels.contains(&SecurityLevel::Standard128));
    }

    #[test]
    fn test_benchmark_config_quick() {
        let config = BenchmarkConfig::quick();
        assert_eq!(config.iterations, 10);
        assert_eq!(config.security_levels.len(), 1);
    }

    #[test]
    fn test_system_info() {
        let info = SystemInfo::default();
        assert!(!info.os.is_empty());
        assert!(info.cores > 0);
    }

    #[tokio::test]
    async fn test_benchmark_runner_range() {
        let runner = BenchmarkRunner::new(BenchmarkConfig {
            iterations: 2,
            warmup_iterations: 1,
            security_levels: vec![SecurityLevel::Standard96],
            proof_types: vec![ProofType::Range],
            ..Default::default()
        });

        let results = runner.run_all().await;
        assert!(!results.results.is_empty());

        let range_result = results
            .results
            .iter()
            .find(|r| r.proof_type == ProofType::Range)
            .unwrap();
        assert!(range_result.generation.count > 0);
        assert!(range_result.memory.proof_size_bytes > 0);
    }

    #[test]
    fn test_throughput_config_default() {
        let config = ThroughputConfig::default();
        assert_eq!(config.duration, Duration::from_secs(10));
        assert_eq!(config.concurrency, 4);
    }

    #[test]
    fn test_results_to_json() {
        let results = BenchmarkResults {
            config: BenchmarkConfig::default(),
            results: vec![ProofBenchmarkResult {
                proof_type: ProofType::Range,
                security_level: SecurityLevel::Standard128,
                generation: TimingStats::from_durations(&[Duration::from_millis(50)]),
                verification: TimingStats::from_durations(&[Duration::from_millis(10)]),
                memory: MemoryStats {
                    proof_size_bytes: 15000,
                    ..Default::default()
                },
                parameters: HashMap::new(),
            }],
            total_duration: Duration::from_secs(60),
            system_info: SystemInfo::default(),
        };

        let json = results.to_json();
        assert!(json.get("results").is_some());
        assert!(json.get("system").is_some());
    }

    #[test]
    fn test_timing_stats_format() {
        let durations = vec![Duration::from_millis(50), Duration::from_millis(60)];
        let stats = TimingStats::from_durations(&durations);
        let formatted = stats.format();

        assert!(formatted.contains("ms"));
        assert!(formatted.contains("ops/s"));
    }

    #[test]
    fn test_memory_stats_format() {
        let stats = MemoryStats {
            proof_size_bytes: 15000,
            peak_bytes: 1024 * 1024,
            ..Default::default()
        };
        let formatted = stats.format();

        assert!(formatted.contains("15000"));
        assert!(formatted.contains("KB"));
    }
}
