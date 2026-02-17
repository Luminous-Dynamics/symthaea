//! Parallel Proof Generation
//!
//! Concurrent proof generation for improved throughput using async tasks.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::{ParallelProofGenerator, ProofTask};
//!
//! let mut generator = ParallelProofGenerator::new(ProofConfig::default());
//!
//! // Queue multiple proofs
//! generator.add_range_proof(50, 0, 100);
//! generator.add_gradient_proof(&gradient_data, 5.0);
//!
//! // Generate all in parallel
//! let results = generator.generate_all().await?;
//! ```

use crate::proofs::{
    ProofConfig, ProofResult, ProofType,
    RangeProof, GradientIntegrityProof, IdentityAssuranceProof, VoteEligibilityProof,
    ProofIdentityFactor, ProofAssuranceLevel, ProofVoterProfile, ProofProposalType,
};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

/// Task specification for proof generation
#[derive(Clone)]
pub enum ProofTask {
    /// Range proof: value, min, max
    Range {
        value: u64,
        min: u64,
        max: u64,
        label: Option<String>,
    },
    /// Gradient integrity proof: values, max_norm
    Gradient {
        values: Vec<f32>,
        max_norm: f32,
        label: Option<String>,
    },
    /// Identity assurance proof
    Identity {
        did: String,
        factors: Vec<ProofIdentityFactor>,
        level: ProofAssuranceLevel,
        label: Option<String>,
    },
    /// Vote eligibility proof
    Vote {
        voter: ProofVoterProfile,
        proposal_type: ProofProposalType,
        label: Option<String>,
    },
}

impl ProofTask {
    fn label(&self) -> Option<&str> {
        match self {
            ProofTask::Range { label, .. } => label.as_deref(),
            ProofTask::Gradient { label, .. } => label.as_deref(),
            ProofTask::Identity { label, .. } => label.as_deref(),
            ProofTask::Vote { label, .. } => label.as_deref(),
        }
    }

    fn proof_type(&self) -> ProofType {
        match self {
            ProofTask::Range { .. } => ProofType::Range,
            ProofTask::Gradient { .. } => ProofType::GradientIntegrity,
            ProofTask::Identity { .. } => ProofType::IdentityAssurance,
            ProofTask::Vote { .. } => ProofType::VoteEligibility,
        }
    }
}

/// Result from parallel proof generation
#[derive(Clone)]
pub struct ParallelProofResult {
    /// Proof type
    pub proof_type: ProofType,

    /// Optional label
    pub label: Option<String>,

    /// Generation time
    pub generation_time: Duration,

    /// Success status
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,

    /// Proof size in bytes (if successful)
    pub proof_size: Option<usize>,
}

/// Results summary from parallel generation
#[derive(Clone)]
pub struct ParallelGenerationResult {
    /// Total number of tasks
    pub total_tasks: usize,

    /// Number of successful generations
    pub successful: usize,

    /// Number of failed generations
    pub failed: usize,

    /// Total wall-clock time
    pub total_time: Duration,

    /// Total CPU time (sum of individual generation times)
    pub total_cpu_time: Duration,

    /// Speedup factor (cpu_time / wall_time)
    pub speedup: f64,

    /// Individual results
    pub results: Vec<ParallelProofResult>,
}

impl ParallelGenerationResult {
    /// Get successful results
    pub fn successful_results(&self) -> impl Iterator<Item = &ParallelProofResult> {
        self.results.iter().filter(|r| r.success)
    }

    /// Get failed results
    pub fn failed_results(&self) -> impl Iterator<Item = &ParallelProofResult> {
        self.results.iter().filter(|r| !r.success)
    }

    /// Get result by label
    pub fn by_label(&self, label: &str) -> Option<&ParallelProofResult> {
        self.results.iter().find(|r| {
            r.label.as_ref().map(|l| l == label).unwrap_or(false)
        })
    }
}

/// Generator for parallel proof creation
pub struct ParallelProofGenerator {
    /// Tasks to execute
    tasks: Vec<ProofTask>,

    /// Configuration
    config: ProofConfig,

    /// Maximum concurrent tasks (0 = unlimited)
    max_concurrent: usize,
}

impl ParallelProofGenerator {
    /// Create a new parallel generator
    pub fn new(config: ProofConfig) -> Self {
        Self {
            tasks: Vec::new(),
            config,
            max_concurrent: 0,
        }
    }

    /// Set maximum concurrent tasks
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Add a range proof task
    pub fn add_range_proof(&mut self, value: u64, min: u64, max: u64) -> &mut Self {
        self.tasks.push(ProofTask::Range {
            value,
            min,
            max,
            label: None,
        });
        self
    }

    /// Add a labeled range proof task
    pub fn add_range_proof_labeled(
        &mut self,
        value: u64,
        min: u64,
        max: u64,
        label: impl Into<String>,
    ) -> &mut Self {
        self.tasks.push(ProofTask::Range {
            value,
            min,
            max,
            label: Some(label.into()),
        });
        self
    }

    /// Add a gradient proof task
    pub fn add_gradient_proof(&mut self, values: Vec<f32>, max_norm: f32) -> &mut Self {
        self.tasks.push(ProofTask::Gradient {
            values,
            max_norm,
            label: None,
        });
        self
    }

    /// Add a labeled gradient proof task
    pub fn add_gradient_proof_labeled(
        &mut self,
        values: Vec<f32>,
        max_norm: f32,
        label: impl Into<String>,
    ) -> &mut Self {
        self.tasks.push(ProofTask::Gradient {
            values,
            max_norm,
            label: Some(label.into()),
        });
        self
    }

    /// Add an identity proof task
    pub fn add_identity_proof(
        &mut self,
        did: impl Into<String>,
        factors: Vec<ProofIdentityFactor>,
        level: ProofAssuranceLevel,
    ) -> &mut Self {
        self.tasks.push(ProofTask::Identity {
            did: did.into(),
            factors,
            level,
            label: None,
        });
        self
    }

    /// Add a labeled identity proof task
    pub fn add_identity_proof_labeled(
        &mut self,
        did: impl Into<String>,
        factors: Vec<ProofIdentityFactor>,
        level: ProofAssuranceLevel,
        label: impl Into<String>,
    ) -> &mut Self {
        self.tasks.push(ProofTask::Identity {
            did: did.into(),
            factors,
            level,
            label: Some(label.into()),
        });
        self
    }

    /// Add a vote eligibility proof task
    pub fn add_vote_proof(
        &mut self,
        voter: ProofVoterProfile,
        proposal_type: ProofProposalType,
    ) -> &mut Self {
        self.tasks.push(ProofTask::Vote {
            voter,
            proposal_type,
            label: None,
        });
        self
    }

    /// Add a labeled vote eligibility proof task
    pub fn add_vote_proof_labeled(
        &mut self,
        voter: ProofVoterProfile,
        proposal_type: ProofProposalType,
        label: impl Into<String>,
    ) -> &mut Self {
        self.tasks.push(ProofTask::Vote {
            voter,
            proposal_type,
            label: Some(label.into()),
        });
        self
    }

    /// Add a custom task
    pub fn add_task(&mut self, task: ProofTask) -> &mut Self {
        self.tasks.push(task);
        self
    }

    /// Get the number of pending tasks
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if no tasks are pending
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Clear all tasks
    pub fn clear(&mut self) {
        self.tasks.clear();
    }

    /// Generate all proofs in parallel
    pub async fn generate_all(self) -> ProofResult<ParallelGenerationResult> {
        if self.tasks.is_empty() {
            return Ok(ParallelGenerationResult {
                total_tasks: 0,
                successful: 0,
                failed: 0,
                total_time: Duration::ZERO,
                total_cpu_time: Duration::ZERO,
                speedup: 1.0,
                results: Vec::new(),
            });
        }

        let start = Instant::now();
        let total_tasks = self.tasks.len();
        let config = self.config.clone();

        // Use tokio spawn_blocking for CPU-intensive proof generation
        let mut join_set = JoinSet::new();

        for task in self.tasks {
            let cfg = config.clone();
            join_set.spawn_blocking(move || {
                generate_single_proof(task, cfg)
            });
        }

        // Collect results
        let mut results = Vec::with_capacity(total_tasks);
        let mut successful = 0;
        let mut failed = 0;
        let mut total_cpu_time = Duration::ZERO;

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(proof_result) => {
                    total_cpu_time += proof_result.generation_time;
                    if proof_result.success {
                        successful += 1;
                    } else {
                        failed += 1;
                    }
                    results.push(proof_result);
                }
                Err(e) => {
                    failed += 1;
                    results.push(ParallelProofResult {
                        proof_type: ProofType::Range, // Unknown
                        label: None,
                        generation_time: Duration::ZERO,
                        success: false,
                        error: Some(format!("Task panic: {}", e)),
                        proof_size: None,
                    });
                }
            }
        }

        let total_time = start.elapsed();
        let speedup = if total_time.as_secs_f64() > 0.0 {
            total_cpu_time.as_secs_f64() / total_time.as_secs_f64()
        } else {
            1.0
        };

        Ok(ParallelGenerationResult {
            total_tasks,
            successful,
            failed,
            total_time,
            total_cpu_time,
            speedup,
            results,
        })
    }

    /// Generate proofs sequentially (for comparison)
    pub fn generate_sequential(self) -> ProofResult<ParallelGenerationResult> {
        if self.tasks.is_empty() {
            return Ok(ParallelGenerationResult {
                total_tasks: 0,
                successful: 0,
                failed: 0,
                total_time: Duration::ZERO,
                total_cpu_time: Duration::ZERO,
                speedup: 1.0,
                results: Vec::new(),
            });
        }

        let start = Instant::now();
        let total_tasks = self.tasks.len();
        let config = self.config.clone();

        let mut results = Vec::with_capacity(total_tasks);
        let mut successful = 0;
        let mut failed = 0;
        let mut total_cpu_time = Duration::ZERO;

        for task in self.tasks {
            let result = generate_single_proof(task, config.clone());
            total_cpu_time += result.generation_time;
            if result.success {
                successful += 1;
            } else {
                failed += 1;
            }
            results.push(result);
        }

        let total_time = start.elapsed();

        Ok(ParallelGenerationResult {
            total_tasks,
            successful,
            failed,
            total_time,
            total_cpu_time,
            speedup: 1.0,
            results,
        })
    }

    /// Generate proofs in parallel using rayon (CPU-bound parallelism)
    ///
    /// This is more efficient than async parallelism for CPU-bound proof generation
    /// as it uses true multi-threading without async overhead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use fl_aggregator::proofs::integration::ParallelProofGenerator;
    ///
    /// let mut generator = ParallelProofGenerator::default();
    /// for i in 0..10 {
    ///     generator.add_range_proof(i * 10, 0, 100);
    /// }
    ///
    /// // Uses all CPU cores for parallel generation
    /// let results = generator.generate_parallel_rayon()?;
    /// println!("Speedup: {:.2}x", results.speedup);
    /// ```
    #[cfg(feature = "proofs-parallel")]
    pub fn generate_parallel_rayon(self) -> ProofResult<ParallelGenerationResult> {
        use rayon::prelude::*;

        if self.tasks.is_empty() {
            return Ok(ParallelGenerationResult {
                total_tasks: 0,
                successful: 0,
                failed: 0,
                total_time: Duration::ZERO,
                total_cpu_time: Duration::ZERO,
                speedup: 1.0,
                results: Vec::new(),
            });
        }

        let start = Instant::now();
        let total_tasks = self.tasks.len();
        let config = self.config.clone();

        // Generate proofs in parallel using rayon's thread pool
        let results: Vec<ParallelProofResult> = self.tasks
            .into_par_iter()
            .map(|task| generate_single_proof(task, config.clone()))
            .collect();

        let total_time = start.elapsed();

        let successful = results.iter().filter(|r| r.success).count();
        let failed = results.iter().filter(|r| !r.success).count();
        let total_cpu_time: Duration = results.iter()
            .map(|r| r.generation_time)
            .sum();

        let speedup = if total_time.as_secs_f64() > 0.0 {
            total_cpu_time.as_secs_f64() / total_time.as_secs_f64()
        } else {
            1.0
        };

        Ok(ParallelGenerationResult {
            total_tasks,
            successful,
            failed,
            total_time,
            total_cpu_time,
            speedup,
            results,
        })
    }

    /// Generate proofs in parallel with a custom thread pool size
    #[cfg(feature = "proofs-parallel")]
    pub fn generate_parallel_rayon_with_threads(
        self,
        num_threads: usize,
    ) -> ProofResult<ParallelGenerationResult> {
        use rayon::ThreadPoolBuilder;

        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| crate::proofs::ProofError::GenerationFailed(
                format!("Failed to create thread pool: {}", e)
            ))?;

        pool.install(|| self.generate_parallel_rayon())
    }
}

impl Default for ParallelProofGenerator {
    fn default() -> Self {
        Self::new(ProofConfig::default())
    }
}

/// Generate a single proof (blocking)
fn generate_single_proof(task: ProofTask, config: ProofConfig) -> ParallelProofResult {
    let proof_type = task.proof_type();
    let label = task.label().map(|s| s.to_string());
    let start = Instant::now();

    let result = match task {
        ProofTask::Range { value, min, max, .. } => {
            RangeProof::generate(value, min, max, config)
                .map(|p| p.size())
        }
        ProofTask::Gradient { values, max_norm, .. } => {
            GradientIntegrityProof::generate(&values, max_norm, config)
                .map(|p| p.size())
        }
        ProofTask::Identity { did, factors, level, .. } => {
            IdentityAssuranceProof::generate(&did, &factors, level, config)
                .map(|p| p.size())
        }
        ProofTask::Vote { voter, proposal_type, .. } => {
            VoteEligibilityProof::generate(&voter, proposal_type, config)
                .map(|p| p.size())
        }
    };

    let generation_time = start.elapsed();

    match result {
        Ok(size) => ParallelProofResult {
            proof_type,
            label,
            generation_time,
            success: true,
            error: None,
            proof_size: Some(size),
        },
        Err(e) => ParallelProofResult {
            proof_type,
            label,
            generation_time,
            success: false,
            error: Some(format!("{:?}", e)),
            proof_size: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::SecurityLevel;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[tokio::test]
    async fn test_parallel_generator_empty() {
        let generator = ParallelProofGenerator::new(test_config());
        let result = generator.generate_all().await.unwrap();

        assert_eq!(result.total_tasks, 0);
        assert_eq!(result.successful, 0);
        assert_eq!(result.failed, 0);
    }

    #[tokio::test]
    async fn test_parallel_generator_single_range() {
        let mut generator = ParallelProofGenerator::new(test_config());
        generator.add_range_proof(50, 0, 100);

        let result = generator.generate_all().await.unwrap();

        assert_eq!(result.total_tasks, 1);
        assert_eq!(result.successful, 1);
        assert_eq!(result.failed, 0);
        assert!(result.results[0].success);
        assert!(result.results[0].proof_size.is_some());
    }

    #[tokio::test]
    async fn test_parallel_generator_multiple() {
        let mut generator = ParallelProofGenerator::new(test_config());

        // Add multiple tasks
        generator.add_range_proof_labeled(50, 0, 100, "range_1");
        generator.add_range_proof_labeled(25, 0, 50, "range_2");
        generator.add_range_proof_labeled(75, 50, 100, "range_3");

        let result = generator.generate_all().await.unwrap();

        assert_eq!(result.total_tasks, 3);
        assert_eq!(result.successful, 3);
        assert_eq!(result.failed, 0);

        // Check speedup (should be > 1 with parallel execution)
        println!("Speedup: {:.2}x", result.speedup);
        println!("Wall time: {:?}", result.total_time);
        println!("CPU time: {:?}", result.total_cpu_time);
    }

    #[tokio::test]
    async fn test_parallel_generator_mixed_types() {
        let mut generator = ParallelProofGenerator::new(test_config());

        // Add different proof types
        generator.add_range_proof_labeled(50, 0, 100, "range");

        generator.add_gradient_proof_labeled(
            vec![0.1, -0.2, 0.3, -0.4, 0.5],
            5.0,
            "gradient",
        );

        let factors = vec![
            ProofIdentityFactor::new(0.5, 0, true),
            ProofIdentityFactor::new(0.3, 1, true),
        ];
        generator.add_identity_proof_labeled(
            "did:test",
            factors,
            ProofAssuranceLevel::E2,
            "identity",
        );

        let result = generator.generate_all().await.unwrap();

        assert_eq!(result.total_tasks, 3);
        assert_eq!(result.successful, 3);

        // Check we can find by label
        assert!(result.by_label("range").is_some());
        assert!(result.by_label("gradient").is_some());
        assert!(result.by_label("identity").is_some());
    }

    #[test]
    fn test_sequential_generation() {
        let mut generator = ParallelProofGenerator::new(test_config());

        generator.add_range_proof(50, 0, 100);
        generator.add_range_proof(25, 0, 50);

        let result = generator.generate_sequential().unwrap();

        assert_eq!(result.total_tasks, 2);
        assert_eq!(result.successful, 2);
        assert_eq!(result.speedup, 1.0); // Sequential has no speedup
    }

    #[tokio::test]
    async fn test_parallel_with_failure() {
        let mut generator = ParallelProofGenerator::new(test_config());

        // Valid proof
        generator.add_range_proof_labeled(50, 0, 100, "valid");

        // Invalid gradient (norm too small)
        generator.add_gradient_proof_labeled(
            vec![10.0, 10.0, 10.0, 10.0, 10.0], // L2 norm ~22.4
            1.0, // max_norm too small
            "invalid",
        );

        let result = generator.generate_all().await.unwrap();

        assert_eq!(result.total_tasks, 2);
        assert_eq!(result.successful, 1);
        assert_eq!(result.failed, 1);

        let valid_result = result.by_label("valid").unwrap();
        assert!(valid_result.success);

        let invalid_result = result.by_label("invalid").unwrap();
        assert!(!invalid_result.success);
        assert!(invalid_result.error.is_some());
    }

    #[cfg(feature = "proofs-parallel")]
    #[test]
    fn test_rayon_parallel_generation() {
        let mut generator = ParallelProofGenerator::new(test_config());

        // Add multiple range proofs
        for i in 0..4 {
            generator.add_range_proof_labeled(i * 20, 0, 100, format!("range_{}", i));
        }

        let result = generator.generate_parallel_rayon().unwrap();

        assert_eq!(result.total_tasks, 4);
        assert_eq!(result.successful, 4);
        assert_eq!(result.failed, 0);

        // With multiple cores, we should see speedup > 1
        println!("Rayon speedup: {:.2}x", result.speedup);
        println!("Wall time: {:?}", result.total_time);
        println!("CPU time: {:?}", result.total_cpu_time);
    }

    #[cfg(feature = "proofs-parallel")]
    #[test]
    fn test_rayon_parallel_with_custom_threads() {
        let mut generator = ParallelProofGenerator::new(test_config());

        for i in 0..4 {
            generator.add_range_proof(i * 20, 0, 100);
        }

        // Use 2 threads explicitly
        let result = generator.generate_parallel_rayon_with_threads(2).unwrap();

        assert_eq!(result.total_tasks, 4);
        assert_eq!(result.successful, 4);
        println!("2-thread speedup: {:.2}x", result.speedup);
    }
}
