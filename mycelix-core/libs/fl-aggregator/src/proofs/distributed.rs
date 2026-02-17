//! Distributed Proof Generation
//!
//! Split large proof computations across multiple worker nodes for parallel processing.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │ Coordinator │────▶│  Worker 1   │────▶│   Proof     │
//! │             │────▶│  Worker 2   │────▶│  Fragment 1 │
//! │             │────▶│  Worker N   │────▶│  Fragment N │
//! └─────────────┘     └─────────────┘     └──────┬──────┘
//!                                                │
//!                                         ┌──────▼──────┐
//!                                         │  Aggregate  │
//!                                         │    Proof    │
//!                                         └─────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::distributed::{
//!     ProofCoordinator, WorkerConfig, DistributedProofRequest,
//! };
//!
//! // Create coordinator
//! let coordinator = ProofCoordinator::new(CoordinatorConfig::default());
//!
//! // Register workers
//! coordinator.register_worker("worker-1", "http://worker1:8080").await?;
//! coordinator.register_worker("worker-2", "http://worker2:8080").await?;
//!
//! // Submit distributed proof request
//! let request = DistributedProofRequest::gradient(gradients, max_norm);
//! let proof = coordinator.generate_distributed(request).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Semaphore};

use crate::proofs::{
    ProofConfig, ProofResult, ProofType,
    GradientIntegrityProof,
};
#[cfg(feature = "prometheus")]
use crate::proofs::METRICS;

// ============================================================================
// Configuration
// ============================================================================

/// Coordinator configuration
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Maximum concurrent workers
    pub max_workers: usize,
    /// Worker timeout in seconds
    pub worker_timeout_secs: u64,
    /// Retry attempts for failed workers
    pub retry_attempts: u32,
    /// Minimum chunk size for distribution
    pub min_chunk_size: usize,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Enable load balancing
    pub load_balancing: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            max_workers: 16,
            worker_timeout_secs: 300,
            retry_attempts: 3,
            min_chunk_size: 1000,
            health_check_interval_secs: 30,
            load_balancing: true,
        }
    }
}

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Worker ID
    pub id: String,
    /// Worker endpoint URL
    pub endpoint: String,
    /// Maximum concurrent tasks
    pub max_tasks: usize,
    /// Worker capabilities
    pub capabilities: WorkerCapabilities,
}

/// Worker capabilities
#[derive(Debug, Clone, Default)]
pub struct WorkerCapabilities {
    /// GPU acceleration available
    pub gpu_available: bool,
    /// Maximum memory in bytes
    pub max_memory_bytes: u64,
    /// Supported proof types
    pub supported_types: Vec<ProofType>,
    /// Processing speed factor (1.0 = baseline)
    pub speed_factor: f64,
}

// ============================================================================
// Worker State
// ============================================================================

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Worker is available
    Available,
    /// Worker is busy
    Busy,
    /// Worker is unhealthy
    Unhealthy,
    /// Worker is offline
    Offline,
}

/// Worker state tracking
#[derive(Debug, Clone)]
pub struct WorkerState {
    pub config: WorkerConfig,
    pub status: WorkerStatus,
    pub current_tasks: usize,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub average_task_time_ms: f64,
    pub last_health_check: Option<Instant>,
    pub last_error: Option<String>,
}

impl WorkerState {
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            status: WorkerStatus::Available,
            current_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            average_task_time_ms: 0.0,
            last_health_check: None,
            last_error: None,
        }
    }

    pub fn is_available(&self) -> bool {
        self.status == WorkerStatus::Available
            && self.current_tasks < self.config.max_tasks
    }

    pub fn load_factor(&self) -> f64 {
        if self.config.max_tasks == 0 {
            return 1.0;
        }
        self.current_tasks as f64 / self.config.max_tasks as f64
    }
}

// ============================================================================
// Distributed Request/Response
// ============================================================================

/// Distributed proof request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedProofRequest {
    /// Request ID
    pub id: String,
    /// Proof type
    pub proof_type: ProofType,
    /// Proof configuration
    pub config: ProofConfig,
    /// Request-specific data
    pub data: DistributedProofData,
    /// Priority (higher = more important)
    pub priority: u8,
}

/// Request-specific data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedProofData {
    /// Gradient proof data
    Gradient {
        /// Gradient values (can be chunked)
        gradients: Vec<f32>,
        /// Maximum L2 norm
        max_norm: f32,
    },
    /// Range proof batch
    RangeBatch {
        /// Values to prove
        values: Vec<(u64, u64, u64)>, // (value, min, max)
    },
    /// Generic chunked data
    Chunked {
        /// Data chunks
        chunks: Vec<Vec<u8>>,
        /// Chunk metadata
        metadata: HashMap<String, String>,
    },
}

impl DistributedProofRequest {
    /// Create a gradient proof request
    pub fn gradient(gradients: Vec<f32>, max_norm: f32) -> Self {
        Self {
            id: uuid_v4(),
            proof_type: ProofType::GradientIntegrity,
            config: ProofConfig::default(),
            data: DistributedProofData::Gradient { gradients, max_norm },
            priority: 5,
        }
    }

    /// Create a range batch request
    pub fn range_batch(values: Vec<(u64, u64, u64)>) -> Self {
        Self {
            id: uuid_v4(),
            proof_type: ProofType::Range,
            config: ProofConfig::default(),
            data: DistributedProofData::RangeBatch { values },
            priority: 5,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set config
    pub fn with_config(mut self, config: ProofConfig) -> Self {
        self.config = config;
        self
    }
}

/// Work chunk assigned to a worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkChunk {
    /// Chunk ID
    pub id: String,
    /// Parent request ID
    pub request_id: String,
    /// Chunk index
    pub index: usize,
    /// Total chunks
    pub total: usize,
    /// Chunk data
    pub data: ChunkData,
    /// Configuration
    pub config: ProofConfig,
}

/// Chunk-specific data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkData {
    /// Gradient chunk
    GradientChunk {
        gradients: Vec<f32>,
        start_index: usize,
        max_norm: f32,
    },
    /// Range values chunk
    RangeChunk {
        values: Vec<(u64, u64, u64)>,
    },
    /// Raw bytes chunk
    RawChunk {
        data: Vec<u8>,
        metadata: HashMap<String, String>,
    },
}

/// Chunk result from worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResult {
    /// Chunk ID
    pub chunk_id: String,
    /// Success status
    pub success: bool,
    /// Partial proof bytes
    pub proof_bytes: Option<Vec<u8>>,
    /// Chunk commitment (for verification)
    pub commitment: [u8; 32],
    /// Processing time in ms
    pub processing_time_ms: f64,
    /// Error message if failed
    pub error: Option<String>,
}

/// Distributed proof result
#[derive(Debug, Clone)]
pub struct DistributedProofResult {
    /// Request ID
    pub request_id: String,
    /// Success status
    pub success: bool,
    /// Aggregated proof bytes
    pub proof_bytes: Option<Vec<u8>>,
    /// Total processing time
    pub total_time_ms: f64,
    /// Worker times
    pub worker_times: HashMap<String, f64>,
    /// Chunks processed
    pub chunks_processed: usize,
    /// Failed chunks
    pub chunks_failed: usize,
    /// Error if failed
    pub error: Option<String>,
}

// ============================================================================
// Coordinator
// ============================================================================

/// Proof generation coordinator
///
/// **Internal**: This struct is not production-ready. Worker dispatch is
/// simulated with local processing only. Do not use in production.
#[doc(hidden)]
#[allow(dead_code)]
pub struct ProofCoordinator {
    config: CoordinatorConfig,
    workers: Arc<RwLock<HashMap<String, WorkerState>>>,
    task_semaphore: Arc<Semaphore>,
    pending_requests: Arc<RwLock<HashMap<String, PendingRequest>>>,
}

#[allow(dead_code)]
struct PendingRequest {
    request: DistributedProofRequest,
    chunks: Vec<WorkChunk>,
    results: Vec<Option<ChunkResult>>,
    start_time: Instant,
}

impl ProofCoordinator {
    /// Create a new coordinator
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            task_semaphore: Arc::new(Semaphore::new(config.max_workers)),
            config,
            workers: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a worker
    pub async fn register_worker(&self, id: &str, endpoint: &str) -> ProofResult<()> {
        let config = WorkerConfig {
            id: id.to_string(),
            endpoint: endpoint.to_string(),
            max_tasks: 4,
            capabilities: WorkerCapabilities::default(),
        };

        let state = WorkerState::new(config);
        self.workers.write().await.insert(id.to_string(), state);

        Ok(())
    }

    /// Register worker with full config
    pub async fn register_worker_with_config(&self, config: WorkerConfig) -> ProofResult<()> {
        let id = config.id.clone();
        let state = WorkerState::new(config);
        self.workers.write().await.insert(id, state);
        Ok(())
    }

    /// Unregister a worker
    pub async fn unregister_worker(&self, id: &str) -> ProofResult<()> {
        self.workers.write().await.remove(id);
        Ok(())
    }

    /// Get worker status
    pub async fn worker_status(&self, id: &str) -> Option<WorkerStatus> {
        self.workers.read().await.get(id).map(|w| w.status)
    }

    /// Get all workers
    pub async fn list_workers(&self) -> Vec<WorkerState> {
        self.workers.read().await.values().cloned().collect()
    }

    /// Get available worker count
    pub async fn available_workers(&self) -> usize {
        self.workers
            .read()
            .await
            .values()
            .filter(|w| w.is_available())
            .count()
    }

    /// Generate proof using distributed workers
    pub async fn generate_distributed(
        &self,
        request: DistributedProofRequest,
    ) -> ProofResult<DistributedProofResult> {
        let start = Instant::now();
        let request_id = request.id.clone();

        // Check available workers
        let available = self.available_workers().await;
        if available == 0 {
            return Ok(DistributedProofResult {
                request_id,
                success: false,
                proof_bytes: None,
                total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                worker_times: HashMap::new(),
                chunks_processed: 0,
                chunks_failed: 0,
                error: Some("No available workers".to_string()),
            });
        }

        // Split request into chunks
        let chunks = self.split_into_chunks(&request, available)?;
        let total_chunks = chunks.len();

        if total_chunks == 0 {
            return Ok(DistributedProofResult {
                request_id,
                success: false,
                proof_bytes: None,
                total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                worker_times: HashMap::new(),
                chunks_processed: 0,
                chunks_failed: 0,
                error: Some("No chunks to process".to_string()),
            });
        }

        // Store pending request
        {
            let pending = PendingRequest {
                request: request.clone(),
                chunks: chunks.clone(),
                results: vec![None; total_chunks],
                start_time: start,
            };
            self.pending_requests
                .write()
                .await
                .insert(request_id.clone(), pending);
        }

        // Process chunks (simulated - in real impl would dispatch to workers)
        let mut results = Vec::with_capacity(total_chunks);
        let mut worker_times = HashMap::new();

        for chunk in chunks {
            let chunk_start = Instant::now();
            let result = self.process_chunk_local(&chunk).await;
            let elapsed = chunk_start.elapsed().as_secs_f64() * 1000.0;

            worker_times.insert(format!("chunk_{}", chunk.index), elapsed);
            results.push(result);
        }

        // Remove pending request
        self.pending_requests.write().await.remove(&request_id);

        // Aggregate results
        let successful: Vec<_> = results.iter().filter(|r| r.success).collect();
        let failed_count = results.len() - successful.len();

        if failed_count > 0 {
            let errors: Vec<_> = results
                .iter()
                .filter_map(|r| r.error.as_ref())
                .collect();

            return Ok(DistributedProofResult {
                request_id,
                success: false,
                proof_bytes: None,
                total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                worker_times,
                chunks_processed: successful.len(),
                chunks_failed: failed_count,
                error: Some(format!("Chunk failures: {:?}", errors)),
            });
        }

        // Aggregate proof bytes
        let aggregated = self.aggregate_chunk_results(&results, &request)?;

        #[cfg(feature = "prometheus")]
        METRICS.record_generation(
            request.proof_type,
            start.elapsed().as_secs_f64(),
            aggregated.len(),
        );

        Ok(DistributedProofResult {
            request_id,
            success: true,
            proof_bytes: Some(aggregated),
            total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            worker_times,
            chunks_processed: total_chunks,
            chunks_failed: 0,
            error: None,
        })
    }

    /// Split request into chunks for distribution
    fn split_into_chunks(
        &self,
        request: &DistributedProofRequest,
        worker_count: usize,
    ) -> ProofResult<Vec<WorkChunk>> {
        let mut chunks = Vec::new();

        match &request.data {
            DistributedProofData::Gradient { gradients, max_norm } => {
                let chunk_size = (gradients.len() / worker_count).max(self.config.min_chunk_size);
                let mut start = 0;
                let mut index = 0;

                while start < gradients.len() {
                    let end = (start + chunk_size).min(gradients.len());
                    let chunk_gradients = gradients[start..end].to_vec();

                    chunks.push(WorkChunk {
                        id: format!("{}-{}", request.id, index),
                        request_id: request.id.clone(),
                        index,
                        total: 0, // Will be set after
                        data: ChunkData::GradientChunk {
                            gradients: chunk_gradients,
                            start_index: start,
                            max_norm: *max_norm,
                        },
                        config: request.config.clone(),
                    });

                    start = end;
                    index += 1;
                }
            }
            DistributedProofData::RangeBatch { values } => {
                let chunk_size = (values.len() / worker_count).max(1);
                let mut start = 0;
                let mut index = 0;

                while start < values.len() {
                    let end = (start + chunk_size).min(values.len());
                    let chunk_values = values[start..end].to_vec();

                    chunks.push(WorkChunk {
                        id: format!("{}-{}", request.id, index),
                        request_id: request.id.clone(),
                        index,
                        total: 0,
                        data: ChunkData::RangeChunk {
                            values: chunk_values,
                        },
                        config: request.config.clone(),
                    });

                    start = end;
                    index += 1;
                }
            }
            DistributedProofData::Chunked { chunks: data_chunks, metadata } => {
                for (index, data) in data_chunks.iter().enumerate() {
                    chunks.push(WorkChunk {
                        id: format!("{}-{}", request.id, index),
                        request_id: request.id.clone(),
                        index,
                        total: 0,
                        data: ChunkData::RawChunk {
                            data: data.clone(),
                            metadata: metadata.clone(),
                        },
                        config: request.config.clone(),
                    });
                }
            }
        }

        // Update total count
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total = total;
        }

        Ok(chunks)
    }

    /// Process a chunk locally (for testing/fallback)
    async fn process_chunk_local(&self, chunk: &WorkChunk) -> ChunkResult {
        let start = Instant::now();

        match &chunk.data {
            ChunkData::GradientChunk { gradients, max_norm, .. } => {
                // Generate partial proof for gradient chunk
                match GradientIntegrityProof::generate(gradients, *max_norm, chunk.config.clone()) {
                    Ok(proof) => {
                        let bytes = proof.to_bytes();
                        let commitment = blake3::hash(&bytes);

                        ChunkResult {
                            chunk_id: chunk.id.clone(),
                            success: true,
                            proof_bytes: Some(bytes),
                            commitment: *commitment.as_bytes(),
                            processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                            error: None,
                        }
                    }
                    Err(e) => ChunkResult {
                        chunk_id: chunk.id.clone(),
                        success: false,
                        proof_bytes: None,
                        commitment: [0u8; 32],
                        processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                        error: Some(e.to_string()),
                    },
                }
            }
            ChunkData::RangeChunk { values } => {
                // Generate range proofs for batch
                use crate::proofs::RangeProof;

                let mut all_bytes = Vec::new();
                for (value, min, max) in values {
                    match RangeProof::generate(*value, *min, *max, chunk.config.clone()) {
                        Ok(proof) => {
                            all_bytes.extend_from_slice(&proof.to_bytes());
                        }
                        Err(e) => {
                            return ChunkResult {
                                chunk_id: chunk.id.clone(),
                                success: false,
                                proof_bytes: None,
                                commitment: [0u8; 32],
                                processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                                error: Some(e.to_string()),
                            };
                        }
                    }
                }

                let commitment = blake3::hash(&all_bytes);
                ChunkResult {
                    chunk_id: chunk.id.clone(),
                    success: true,
                    proof_bytes: Some(all_bytes),
                    commitment: *commitment.as_bytes(),
                    processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    error: None,
                }
            }
            ChunkData::RawChunk { data, .. } => {
                // Just hash raw data as placeholder
                let commitment = blake3::hash(data);
                ChunkResult {
                    chunk_id: chunk.id.clone(),
                    success: true,
                    proof_bytes: Some(data.clone()),
                    commitment: *commitment.as_bytes(),
                    processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    error: None,
                }
            }
        }
    }

    /// Aggregate chunk results into final proof
    fn aggregate_chunk_results(
        &self,
        results: &[ChunkResult],
        _request: &DistributedProofRequest,
    ) -> ProofResult<Vec<u8>> {
        // Build aggregated proof structure
        let mut aggregated = Vec::new();

        // Header: magic + version + chunk count
        aggregated.extend_from_slice(b"DIST");
        aggregated.push(1); // version
        aggregated.extend_from_slice(&(results.len() as u32).to_le_bytes());

        // Merkle root of commitments
        let commitments: Vec<[u8; 32]> = results.iter().map(|r| r.commitment).collect();
        let merkle_root = compute_merkle_root(&commitments);
        aggregated.extend_from_slice(&merkle_root);

        // Chunk proofs
        for result in results {
            if let Some(bytes) = &result.proof_bytes {
                aggregated.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                aggregated.extend_from_slice(bytes);
            }
        }

        Ok(aggregated)
    }

    /// Get coordinator statistics
    pub async fn stats(&self) -> CoordinatorStats {
        let workers = self.workers.read().await;
        let pending = self.pending_requests.read().await;

        let total_workers = workers.len();
        let available_workers = workers.values().filter(|w| w.is_available()).count();
        let busy_workers = workers
            .values()
            .filter(|w| w.status == WorkerStatus::Busy)
            .count();
        let unhealthy_workers = workers
            .values()
            .filter(|w| w.status == WorkerStatus::Unhealthy)
            .count();

        let total_completed: u64 = workers.values().map(|w| w.completed_tasks).sum();
        let total_failed: u64 = workers.values().map(|w| w.failed_tasks).sum();

        CoordinatorStats {
            total_workers,
            available_workers,
            busy_workers,
            unhealthy_workers,
            pending_requests: pending.len(),
            total_completed,
            total_failed,
        }
    }
}

/// Coordinator statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorStats {
    pub total_workers: usize,
    pub available_workers: usize,
    pub busy_workers: usize,
    pub unhealthy_workers: usize,
    pub pending_requests: usize,
    pub total_completed: u64,
    pub total_failed: u64,
}

// ============================================================================
// Worker (for running on worker nodes)
// ============================================================================

/// Worker instance for processing proof chunks
#[allow(dead_code)]
pub struct ProofWorker {
    id: String,
    config: WorkerConfig,
    task_semaphore: Semaphore,
    stats: Arc<RwLock<WorkerStats>>,
}

/// Worker statistics
#[derive(Debug, Clone, Default)]
pub struct WorkerStats {
    pub tasks_processed: u64,
    pub tasks_failed: u64,
    pub total_processing_time_ms: f64,
    pub current_tasks: usize,
}

impl ProofWorker {
    /// Create a new worker
    pub fn new(id: &str, max_tasks: usize) -> Self {
        Self {
            id: id.to_string(),
            config: WorkerConfig {
                id: id.to_string(),
                endpoint: String::new(),
                max_tasks,
                capabilities: WorkerCapabilities::default(),
            },
            task_semaphore: Semaphore::new(max_tasks),
            stats: Arc::new(RwLock::new(WorkerStats::default())),
        }
    }

    /// Process a work chunk
    pub async fn process_chunk(&self, chunk: WorkChunk) -> ChunkResult {
        let _permit = match self.task_semaphore.acquire().await {
            Ok(permit) => permit,
            Err(_) => {
                // Semaphore was closed - return failed result
                return ChunkResult {
                    chunk_id: chunk.id.clone(),
                    success: false,
                    proof_bytes: None,
                    commitment: [0u8; 32],
                    processing_time_ms: 0.0,
                    error: Some("Worker semaphore closed".to_string()),
                };
            }
        };

        {
            let mut stats = self.stats.write().await;
            stats.current_tasks += 1;
        }

        let start = Instant::now();
        let result = self.do_process(&chunk).await;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        {
            let mut stats = self.stats.write().await;
            stats.current_tasks -= 1;
            stats.total_processing_time_ms += elapsed;
            if result.success {
                stats.tasks_processed += 1;
            } else {
                stats.tasks_failed += 1;
            }
        }

        result
    }

    async fn do_process(&self, chunk: &WorkChunk) -> ChunkResult {
        let start = Instant::now();

        match &chunk.data {
            ChunkData::GradientChunk { gradients, max_norm, .. } => {
                match GradientIntegrityProof::generate(gradients, *max_norm, chunk.config.clone()) {
                    Ok(proof) => {
                        let bytes = proof.to_bytes();
                        let commitment = blake3::hash(&bytes);

                        ChunkResult {
                            chunk_id: chunk.id.clone(),
                            success: true,
                            proof_bytes: Some(bytes),
                            commitment: *commitment.as_bytes(),
                            processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                            error: None,
                        }
                    }
                    Err(e) => ChunkResult {
                        chunk_id: chunk.id.clone(),
                        success: false,
                        proof_bytes: None,
                        commitment: [0u8; 32],
                        processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                        error: Some(e.to_string()),
                    },
                }
            }
            _ => ChunkResult {
                chunk_id: chunk.id.clone(),
                success: false,
                proof_bytes: None,
                commitment: [0u8; 32],
                processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                error: Some("Unsupported chunk type".to_string()),
            },
        }
    }

    /// Get worker statistics
    pub async fn stats(&self) -> WorkerStats {
        self.stats.read().await.clone()
    }

    /// Get worker ID
    pub fn id(&self) -> &str {
        &self.id
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Generate a UUID v4
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    format!("{:032x}", timestamp)
}

/// Compute Merkle root from commitments
fn compute_merkle_root(commitments: &[[u8; 32]]) -> [u8; 32] {
    if commitments.is_empty() {
        return [0u8; 32];
    }
    if commitments.len() == 1 {
        return commitments[0];
    }

    let mut current_level: Vec<[u8; 32]> = commitments.to_vec();

    while current_level.len() > 1 {
        let mut next_level = Vec::new();

        for i in (0..current_level.len()).step_by(2) {
            let left = current_level[i];
            let right = if i + 1 < current_level.len() {
                current_level[i + 1]
            } else {
                current_level[i] // Duplicate last if odd
            };

            let mut hasher = blake3::Hasher::new();
            hasher.update(&left);
            hasher.update(&right);
            let hash = hasher.finalize();
            next_level.push(*hash.as_bytes());
        }

        current_level = next_level;
    }

    current_level[0]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_config_default() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.max_workers, 16);
        assert_eq!(config.retry_attempts, 3);
    }

    #[test]
    fn test_worker_state_available() {
        let config = WorkerConfig {
            id: "test".to_string(),
            endpoint: "http://localhost".to_string(),
            max_tasks: 4,
            capabilities: WorkerCapabilities::default(),
        };
        let state = WorkerState::new(config);

        assert!(state.is_available());
        assert_eq!(state.load_factor(), 0.0);
    }

    #[test]
    fn test_distributed_request_gradient() {
        let gradients = vec![0.1, 0.2, 0.3, 0.4];
        let request = DistributedProofRequest::gradient(gradients, 5.0);

        assert_eq!(request.proof_type, ProofType::GradientIntegrity);
        assert_eq!(request.priority, 5);
    }

    #[test]
    fn test_distributed_request_range_batch() {
        let values = vec![(10, 0, 100), (50, 0, 100)];
        let request = DistributedProofRequest::range_batch(values);

        assert_eq!(request.proof_type, ProofType::Range);
    }

    #[tokio::test]
    async fn test_coordinator_register_worker() {
        let coordinator = ProofCoordinator::new(CoordinatorConfig::default());

        coordinator
            .register_worker("worker-1", "http://localhost:8080")
            .await
            .unwrap();

        let workers = coordinator.list_workers().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].config.id, "worker-1");
    }

    #[tokio::test]
    async fn test_coordinator_unregister_worker() {
        let coordinator = ProofCoordinator::new(CoordinatorConfig::default());

        coordinator
            .register_worker("worker-1", "http://localhost:8080")
            .await
            .unwrap();
        coordinator.unregister_worker("worker-1").await.unwrap();

        let workers = coordinator.list_workers().await;
        assert!(workers.is_empty());
    }

    #[tokio::test]
    async fn test_coordinator_stats() {
        let coordinator = ProofCoordinator::new(CoordinatorConfig::default());

        coordinator
            .register_worker("worker-1", "http://localhost:8080")
            .await
            .unwrap();

        let stats = coordinator.stats().await;
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.available_workers, 1);
    }

    #[test]
    fn test_compute_merkle_root_single() {
        let commitments = vec![[1u8; 32]];
        let root = compute_merkle_root(&commitments);
        assert_eq!(root, [1u8; 32]);
    }

    #[test]
    fn test_compute_merkle_root_multiple() {
        let commitments = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let root = compute_merkle_root(&commitments);
        assert_ne!(root, [0u8; 32]);
    }

    #[tokio::test]
    async fn test_worker_process() {
        let worker = ProofWorker::new("test-worker", 4);

        let chunk = WorkChunk {
            id: "chunk-1".to_string(),
            request_id: "req-1".to_string(),
            index: 0,
            total: 1,
            data: ChunkData::GradientChunk {
                gradients: vec![0.1, 0.2, 0.3],
                start_index: 0,
                max_norm: 5.0,
            },
            config: ProofConfig::default(),
        };

        let result = worker.process_chunk(chunk).await;
        assert!(result.success);
        assert!(result.proof_bytes.is_some());
    }

    #[tokio::test]
    async fn test_worker_stats() {
        let worker = ProofWorker::new("test-worker", 4);

        let stats = worker.stats().await;
        assert_eq!(stats.tasks_processed, 0);
        assert_eq!(stats.current_tasks, 0);
    }

    #[test]
    fn test_uuid_generation() {
        let id1 = uuid_v4();
        let id2 = uuid_v4();
        assert_ne!(id1, id2);
        assert_eq!(id1.len(), 32);
    }
}
