//! Streaming Aggregation for handling gradients larger than memory.
//!
//! This module provides memory-efficient aggregation of large gradients by processing
//! them in chunks. It supports:
//! - Chunk-wise aggregation with configurable chunk sizes
//! - Automatic spilling to disk when memory limits are exceeded
//! - LRU caching for recently accessed chunks
//! - Async I/O for disk operations
//! - Fault tolerance with checkpointing and resume support
//! - Progress tracking with callbacks
//!
//! ## Example
//!
//! ```rust,ignore
//! use fl_aggregator::streaming::{StreamingAggregator, StreamingConfig};
//! use fl_aggregator::Defense;
//!
//! let config = StreamingConfig::default()
//!     .with_chunk_size(1_000_000)
//!     .with_max_memory_mb(1024);
//!
//! let mut aggregator = StreamingAggregator::new(config, Defense::FedAvg);
//!
//! // Begin streaming submission
//! let handle = aggregator.begin_submission("node1", 10_000_000).await?;
//!
//! // Submit chunks
//! for (i, chunk) in large_gradient.chunks(1_000_000).enumerate() {
//!     aggregator.submit_chunk(&handle, i, chunk).await?;
//! }
//!
//! // Finalize submission
//! aggregator.finalize_submission(handle).await?;
//!
//! // Aggregate all submissions
//! let result = aggregator.aggregate_streaming().await?;
//!
//! // Access result chunks
//! for chunk in result.chunks() {
//!     process(chunk);
//! }
//! ```

use crate::byzantine::{ByzantineAggregator, Defense, DefenseConfig};
use crate::error::{AggregatorError, Result};
use crate::{Gradient, NodeId};

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, RwLock};

/// Configuration for streaming aggregation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Number of f32 elements per chunk.
    pub chunk_size: usize,

    /// Maximum memory limit in megabytes for buffering.
    pub max_memory_mb: usize,

    /// Directory for temporary files when spilling to disk.
    pub temp_dir: Option<PathBuf>,

    /// Whether to compress chunks when storing to disk.
    pub compression: bool,

    /// Number of chunks to keep in LRU cache.
    pub lru_cache_size: usize,

    /// Enable checkpointing for fault tolerance.
    pub enable_checkpointing: bool,

    /// Checkpoint interval (every N chunks processed).
    pub checkpoint_interval: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1_000_000,
            max_memory_mb: 1024,
            temp_dir: None,
            compression: true,
            lru_cache_size: 16,
            enable_checkpointing: true,
            checkpoint_interval: 10,
        }
    }
}

impl StreamingConfig {
    /// Set chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set maximum memory in MB.
    pub fn with_max_memory_mb(mut self, mb: usize) -> Self {
        self.max_memory_mb = mb;
        self
    }

    /// Set temporary directory for disk spilling.
    pub fn with_temp_dir(mut self, path: PathBuf) -> Self {
        self.temp_dir = Some(path);
        self
    }

    /// Enable or disable compression.
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression = enabled;
        self
    }

    /// Set LRU cache size.
    pub fn with_lru_cache_size(mut self, size: usize) -> Self {
        self.lru_cache_size = size;
        self
    }

    /// Get effective temp directory.
    pub fn get_temp_dir(&self) -> PathBuf {
        self.temp_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("fl-streaming"))
    }

    /// Calculate maximum elements that fit in memory.
    pub fn max_elements_in_memory(&self) -> usize {
        (self.max_memory_mb * 1024 * 1024) / std::mem::size_of::<f32>()
    }
}

/// Storage location for a chunk.
#[derive(Clone, Debug)]
pub enum ChunkStorage {
    /// Chunk is stored in memory.
    Memory(Vec<f32>),
    /// Chunk is stored on disk at the given path.
    Disk(PathBuf),
}

/// A chunked gradient representation.
#[derive(Debug)]
pub struct ChunkedGradient {
    /// Total number of elements in the gradient.
    pub total_dimension: usize,
    /// Number of elements per chunk.
    pub chunk_size: usize,
    /// Total number of chunks.
    pub num_chunks: usize,
    /// Storage for each chunk.
    chunks: Vec<Option<ChunkStorage>>,
    /// Compression enabled.
    compression: bool,
    /// Chunks that have been received.
    received_chunks: Vec<bool>,
}

impl ChunkedGradient {
    /// Create a new chunked gradient.
    pub fn new(total_dimension: usize, chunk_size: usize, compression: bool) -> Self {
        let num_chunks = (total_dimension + chunk_size - 1) / chunk_size;
        Self {
            total_dimension,
            chunk_size,
            num_chunks,
            chunks: vec![None; num_chunks],
            compression,
            received_chunks: vec![false; num_chunks],
        }
    }

    /// Check if all chunks have been received.
    pub fn is_complete(&self) -> bool {
        self.received_chunks.iter().all(|&r| r)
    }

    /// Get the number of received chunks.
    pub fn received_count(&self) -> usize {
        self.received_chunks.iter().filter(|&&r| r).count()
    }

    /// Get the actual size of a specific chunk.
    pub fn chunk_actual_size(&self, chunk_index: usize) -> usize {
        if chunk_index >= self.num_chunks {
            return 0;
        }
        if chunk_index == self.num_chunks - 1 {
            // Last chunk may be smaller
            let remaining = self.total_dimension % self.chunk_size;
            if remaining == 0 {
                self.chunk_size
            } else {
                remaining
            }
        } else {
            self.chunk_size
        }
    }

    /// Store a chunk in memory.
    pub fn store_memory(&mut self, chunk_index: usize, data: Vec<f32>) -> Result<()> {
        if chunk_index >= self.num_chunks {
            return Err(AggregatorError::InvalidConfig(format!(
                "Chunk index {} out of range (num_chunks={})",
                chunk_index, self.num_chunks
            )));
        }
        self.chunks[chunk_index] = Some(ChunkStorage::Memory(data));
        self.received_chunks[chunk_index] = true;
        Ok(())
    }

    /// Store a chunk on disk.
    pub async fn store_disk(&mut self, chunk_index: usize, data: &[f32], path: PathBuf) -> Result<()> {
        if chunk_index >= self.num_chunks {
            return Err(AggregatorError::InvalidConfig(format!(
                "Chunk index {} out of range (num_chunks={})",
                chunk_index, self.num_chunks
            )));
        }

        // Serialize chunk data
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Optionally compress
        let final_bytes = if self.compression {
            compress_data(&bytes)?
        } else {
            bytes
        };

        // Write to disk
        let mut file = File::create(&path).await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to create chunk file: {}", e))
        })?;
        file.write_all(&final_bytes).await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to write chunk: {}", e))
        })?;
        file.flush().await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to flush chunk: {}", e))
        })?;

        self.chunks[chunk_index] = Some(ChunkStorage::Disk(path));
        self.received_chunks[chunk_index] = true;
        Ok(())
    }

    /// Load a chunk from storage.
    pub async fn load_chunk(&self, chunk_index: usize) -> Result<Vec<f32>> {
        if chunk_index >= self.num_chunks {
            return Err(AggregatorError::InvalidConfig(format!(
                "Chunk index {} out of range (num_chunks={})",
                chunk_index, self.num_chunks
            )));
        }

        match &self.chunks[chunk_index] {
            Some(ChunkStorage::Memory(data)) => Ok(data.clone()),
            Some(ChunkStorage::Disk(path)) => {
                let mut file = File::open(path).await.map_err(|e| {
                    AggregatorError::Internal(format!("Failed to open chunk file: {}", e))
                })?;
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes).await.map_err(|e| {
                    AggregatorError::Internal(format!("Failed to read chunk: {}", e))
                })?;

                // Optionally decompress
                let decompressed = if self.compression {
                    decompress_data(&bytes)?
                } else {
                    bytes
                };

                // Deserialize
                let data: Vec<f32> = decompressed
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();

                Ok(data)
            }
            None => Err(AggregatorError::InvalidConfig(format!(
                "Chunk {} has not been received",
                chunk_index
            ))),
        }
    }

    /// Clean up disk storage.
    pub async fn cleanup(&self) -> Result<()> {
        for chunk in &self.chunks {
            if let Some(ChunkStorage::Disk(path)) = chunk {
                if path.exists() {
                    let _ = fs::remove_file(path).await;
                }
            }
        }
        Ok(())
    }
}

/// Opaque handle for tracking partial submissions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubmissionHandle {
    /// Unique identifier for this submission.
    pub id: u64,
    /// Node ID associated with this submission.
    pub node_id: NodeId,
    /// Total dimension of the gradient.
    pub total_dimension: usize,
    /// Timestamp when submission started.
    pub started_at: std::time::SystemTime,
}

impl SubmissionHandle {
    /// Create a new submission handle.
    fn new(id: u64, node_id: NodeId, total_dimension: usize) -> Self {
        Self {
            id,
            node_id,
            total_dimension,
            started_at: std::time::SystemTime::now(),
        }
    }
}

/// Progress information for streaming operations.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StreamingProgress {
    /// Total number of nodes expected.
    pub expected_nodes: usize,
    /// Number of nodes that have completed submission.
    pub completed_nodes: usize,
    /// Total chunks across all nodes.
    pub total_chunks: usize,
    /// Chunks received so far.
    pub chunks_received: usize,
    /// Chunks processed during aggregation.
    pub chunks_processed: usize,
    /// Current memory usage in bytes.
    pub memory_bytes: usize,
    /// Chunks spilled to disk.
    pub chunks_on_disk: usize,
    /// Current phase of the streaming operation.
    pub phase: StreamingPhase,
}

/// Current phase of streaming operation.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum StreamingPhase {
    #[default]
    /// Waiting for submissions.
    Receiving,
    /// Aggregating chunks.
    Aggregating,
    /// Aggregation complete.
    Complete,
    /// Error occurred.
    Error,
}

/// Progress callback type.
pub type ProgressCallback = Box<dyn Fn(StreamingProgress) + Send + Sync>;

/// State for a node's streaming submission.
struct NodeSubmissionState {
    handle: SubmissionHandle,
    gradient: ChunkedGradient,
    finalized: bool,
}

/// LRU cache for chunks.
struct LruChunkCache {
    capacity: usize,
    cache: HashMap<(u64, usize), Vec<f32>>,
    access_order: VecDeque<(u64, usize)>,
}

impl LruChunkCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: HashMap::new(),
            access_order: VecDeque::new(),
        }
    }

    #[allow(dead_code)]
    fn get(&mut self, key: &(u64, usize)) -> Option<&Vec<f32>> {
        if self.cache.contains_key(key) {
            // Move to front of access order
            self.access_order.retain(|k| k != key);
            self.access_order.push_front(*key);
            self.cache.get(key)
        } else {
            None
        }
    }

    fn insert(&mut self, key: (u64, usize), value: Vec<f32>) {
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            // Evict least recently used
            if let Some(old_key) = self.access_order.pop_back() {
                self.cache.remove(&old_key);
            }
        }
        self.cache.insert(key, value);
        self.access_order.retain(|k| k != &key);
        self.access_order.push_front(key);
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    fn memory_usage(&self) -> usize {
        self.cache.values().map(|v| v.len() * std::mem::size_of::<f32>()).sum()
    }
}

/// Checkpoint data for fault tolerance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCheckpoint {
    /// Submission handles that were active.
    pub active_submissions: Vec<(u64, NodeId, usize)>,
    /// Chunks received per submission.
    pub received_chunks: HashMap<u64, Vec<usize>>,
    /// Aggregation progress (if started).
    pub aggregation_chunk_index: Option<usize>,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Result of streaming aggregation.
pub struct StreamingResult {
    /// Total dimension of the aggregated gradient.
    pub dimension: usize,
    /// Number of chunks.
    pub num_chunks: usize,
    /// Chunk size used.
    #[allow(dead_code)]
    chunk_size: usize,
    /// Storage for result chunks.
    result_chunks: Vec<Option<ChunkStorage>>,
    /// Compression enabled.
    compression: bool,
    /// Temp directory.
    #[allow(dead_code)]
    temp_dir: PathBuf,
}

impl StreamingResult {
    /// Create a new streaming result.
    fn new(dimension: usize, chunk_size: usize, compression: bool, temp_dir: PathBuf) -> Self {
        let num_chunks = (dimension + chunk_size - 1) / chunk_size;
        Self {
            dimension,
            num_chunks,
            chunk_size,
            result_chunks: vec![None; num_chunks],
            compression,
            temp_dir,
        }
    }

    /// Store a result chunk.
    pub async fn store_chunk(&mut self, index: usize, data: Vec<f32>) -> Result<()> {
        // For now, store in memory; spill to disk if needed
        self.result_chunks[index] = Some(ChunkStorage::Memory(data));
        Ok(())
    }

    /// Get a result chunk.
    pub async fn get_chunk(&self, chunk_index: usize) -> Result<Vec<f32>> {
        if chunk_index >= self.num_chunks {
            return Err(AggregatorError::InvalidConfig(format!(
                "Chunk index {} out of range",
                chunk_index
            )));
        }

        match &self.result_chunks[chunk_index] {
            Some(ChunkStorage::Memory(data)) => Ok(data.clone()),
            Some(ChunkStorage::Disk(path)) => {
                let mut file = File::open(path).await.map_err(|e| {
                    AggregatorError::Internal(format!("Failed to open result chunk: {}", e))
                })?;
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes).await.map_err(|e| {
                    AggregatorError::Internal(format!("Failed to read result chunk: {}", e))
                })?;

                let decompressed = if self.compression {
                    decompress_data(&bytes)?
                } else {
                    bytes
                };

                let data: Vec<f32> = decompressed
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();

                Ok(data)
            }
            None => Err(AggregatorError::InvalidConfig(format!(
                "Result chunk {} not yet computed",
                chunk_index
            ))),
        }
    }

    /// Iterator over result chunks.
    pub fn chunks(&self) -> StreamingResultIter<'_> {
        StreamingResultIter {
            result: self,
            current_index: 0,
        }
    }

    /// Reconstruct full gradient if it fits in memory.
    pub async fn to_full_gradient(&self) -> Result<Array1<f32>> {
        let mut full = Vec::with_capacity(self.dimension);
        for i in 0..self.num_chunks {
            let chunk = self.get_chunk(i).await?;
            full.extend(chunk);
        }
        Ok(Array1::from(full))
    }

    /// Write the aggregated result to a file.
    pub async fn write_to_file(&self, path: &std::path::Path) -> Result<()> {
        let mut file = File::create(path).await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to create output file: {}", e))
        })?;

        // Write header
        let header = format!("{}\n", self.dimension);
        file.write_all(header.as_bytes()).await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to write header: {}", e))
        })?;

        // Write chunks
        for i in 0..self.num_chunks {
            let chunk = self.get_chunk(i).await?;
            let bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            file.write_all(&bytes).await.map_err(|e| {
                AggregatorError::Internal(format!("Failed to write chunk: {}", e))
            })?;
        }

        file.flush().await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to flush output: {}", e))
        })?;

        Ok(())
    }

    /// Clean up disk storage.
    pub async fn cleanup(&self) -> Result<()> {
        for chunk in &self.result_chunks {
            if let Some(ChunkStorage::Disk(path)) = chunk {
                if path.exists() {
                    let _ = fs::remove_file(path).await;
                }
            }
        }
        Ok(())
    }
}

/// Iterator over streaming result chunks.
pub struct StreamingResultIter<'a> {
    result: &'a StreamingResult,
    current_index: usize,
}

impl<'a> Iterator for StreamingResultIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.result.num_chunks {
            let idx = self.current_index;
            self.current_index += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Streaming aggregator for memory-efficient gradient aggregation.
pub struct StreamingAggregator {
    config: StreamingConfig,
    defense: Defense,
    next_handle_id: AtomicU64,
    submissions: Arc<RwLock<HashMap<u64, NodeSubmissionState>>>,
    chunk_cache: Arc<Mutex<LruChunkCache>>,
    progress: Arc<RwLock<StreamingProgress>>,
    progress_callback: Option<Arc<ProgressCallback>>,
    checkpoint_path: Option<PathBuf>,
    expected_nodes: usize,
    expected_dimension: Option<usize>,
}

impl StreamingAggregator {
    /// Create a new streaming aggregator.
    pub fn new(config: StreamingConfig, defense: Defense) -> Self {
        let cache_size = config.lru_cache_size;
        let checkpoint_path = if config.enable_checkpointing {
            Some(config.get_temp_dir().join("checkpoint.json"))
        } else {
            None
        };

        Self {
            config,
            defense,
            next_handle_id: AtomicU64::new(1),
            submissions: Arc::new(RwLock::new(HashMap::new())),
            chunk_cache: Arc::new(Mutex::new(LruChunkCache::new(cache_size))),
            progress: Arc::new(RwLock::new(StreamingProgress::default())),
            progress_callback: None,
            checkpoint_path,
            expected_nodes: 0,
            expected_dimension: None,
        }
    }

    /// Set expected number of nodes.
    pub fn with_expected_nodes(mut self, n: usize) -> Self {
        self.expected_nodes = n;
        self
    }

    /// Set progress callback.
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(Arc::new(callback));
        self
    }

    /// Begin a streaming submission from a node.
    pub async fn begin_submission(
        &self,
        node_id: &str,
        total_dimension: usize,
    ) -> Result<SubmissionHandle> {
        // Check dimension consistency
        if let Some(expected) = self.expected_dimension {
            if expected != total_dimension {
                return Err(AggregatorError::DimensionMismatch {
                    expected,
                    got: total_dimension,
                });
            }
        }

        // Create handle
        let handle_id = self.next_handle_id.fetch_add(1, Ordering::SeqCst);
        let handle = SubmissionHandle::new(handle_id, node_id.to_string(), total_dimension);

        // Create chunked gradient
        let gradient = ChunkedGradient::new(
            total_dimension,
            self.config.chunk_size,
            self.config.compression,
        );
        let num_chunks = gradient.num_chunks;

        // Store submission state
        let state = NodeSubmissionState {
            handle: handle.clone(),
            gradient,
            finalized: false,
        };

        let mut submissions = self.submissions.write().await;

        // Check for duplicate submission
        for (_, existing) in submissions.iter() {
            if existing.handle.node_id == node_id && !existing.finalized {
                return Err(AggregatorError::DuplicateSubmission {
                    node: node_id.to_string(),
                    round: 0,
                });
            }
        }

        submissions.insert(handle_id, state);

        // Update progress
        let mut progress = self.progress.write().await;
        progress.total_chunks += num_chunks;
        progress.expected_nodes = self.expected_nodes;
        drop(progress);

        self.notify_progress().await;

        tracing::debug!(
            "Started streaming submission: node={}, dimension={}, chunks={}",
            node_id,
            total_dimension,
            (total_dimension + self.config.chunk_size - 1) / self.config.chunk_size
        );

        Ok(handle)
    }

    /// Submit a chunk of gradient data.
    pub async fn submit_chunk(
        &self,
        handle: &SubmissionHandle,
        chunk_index: usize,
        data: &[f32],
    ) -> Result<()> {
        let mut submissions = self.submissions.write().await;
        let state = submissions.get_mut(&handle.id).ok_or_else(|| {
            AggregatorError::InvalidNode(format!("Unknown submission handle: {}", handle.id))
        })?;

        if state.finalized {
            return Err(AggregatorError::InvalidConfig(
                "Cannot submit chunk to finalized submission".to_string(),
            ));
        }

        // Validate chunk size
        let expected_size = state.gradient.chunk_actual_size(chunk_index);
        if data.len() != expected_size {
            return Err(AggregatorError::InvalidConfig(format!(
                "Chunk {} has {} elements, expected {}",
                chunk_index,
                data.len(),
                expected_size
            )));
        }

        // Check memory limits and decide where to store
        let mut cache = self.chunk_cache.lock().await;
        let current_memory = cache.memory_usage();
        let chunk_bytes = data.len() * std::mem::size_of::<f32>();
        let max_memory = self.config.max_memory_mb * 1024 * 1024;

        if current_memory + chunk_bytes > max_memory {
            // Spill to disk
            let temp_dir = self.config.get_temp_dir();
            fs::create_dir_all(&temp_dir).await.map_err(|e| {
                AggregatorError::Internal(format!("Failed to create temp dir: {}", e))
            })?;

            let chunk_path = temp_dir.join(format!("chunk_{}_{}.bin", handle.id, chunk_index));
            state.gradient.store_disk(chunk_index, data, chunk_path).await?;

            let mut progress = self.progress.write().await;
            progress.chunks_on_disk += 1;
        } else {
            // Store in memory and cache
            state.gradient.store_memory(chunk_index, data.to_vec())?;
            cache.insert((handle.id, chunk_index), data.to_vec());
        }

        // Update progress
        let mut progress = self.progress.write().await;
        progress.chunks_received += 1;
        progress.memory_bytes = cache.memory_usage();
        drop(progress);
        drop(cache);
        drop(submissions);

        self.notify_progress().await;

        // Checkpoint if enabled
        if self.config.enable_checkpointing {
            let chunks_received = self.progress.read().await.chunks_received;
            if chunks_received % self.config.checkpoint_interval == 0 {
                self.save_checkpoint().await?;
            }
        }

        Ok(())
    }

    /// Finalize a streaming submission.
    pub async fn finalize_submission(&self, handle: SubmissionHandle) -> Result<()> {
        let mut submissions = self.submissions.write().await;
        let state = submissions.get_mut(&handle.id).ok_or_else(|| {
            AggregatorError::InvalidNode(format!("Unknown submission handle: {}", handle.id))
        })?;

        if !state.gradient.is_complete() {
            let missing: Vec<usize> = state
                .gradient
                .received_chunks
                .iter()
                .enumerate()
                .filter(|(_, &received)| !received)
                .map(|(i, _)| i)
                .collect();
            return Err(AggregatorError::InvalidConfig(format!(
                "Submission incomplete: missing chunks {:?}",
                missing
            )));
        }

        state.finalized = true;

        // Update progress
        let mut progress = self.progress.write().await;
        progress.completed_nodes += 1;
        drop(progress);
        drop(submissions);

        self.notify_progress().await;

        tracing::info!("Finalized streaming submission: node={}", handle.node_id);

        Ok(())
    }

    /// Aggregate all submitted gradients chunk by chunk.
    pub async fn aggregate_streaming(&self) -> Result<StreamingResult> {
        let submissions = self.submissions.read().await;

        // Check that we have enough finalized submissions
        let finalized: Vec<_> = submissions
            .values()
            .filter(|s| s.finalized)
            .collect();

        if finalized.is_empty() {
            return Err(AggregatorError::NoGradients(0));
        }

        // Get defense config requirements
        let defense_config = DefenseConfig::with_defense(self.defense.clone());
        let min_required = defense_config.min_required();

        if finalized.len() < min_required {
            return Err(AggregatorError::InsufficientGradients {
                have: finalized.len(),
                need: min_required,
                defense: self.defense.to_string(),
            });
        }

        // Verify all have same dimension and chunk count
        let dimension = finalized[0].gradient.total_dimension;
        let num_chunks = finalized[0].gradient.num_chunks;
        let chunk_size = finalized[0].gradient.chunk_size;

        for state in &finalized {
            if state.gradient.total_dimension != dimension {
                return Err(AggregatorError::DimensionMismatch {
                    expected: dimension,
                    got: state.gradient.total_dimension,
                });
            }
        }

        // Update progress
        {
            let mut progress = self.progress.write().await;
            progress.phase = StreamingPhase::Aggregating;
        }
        self.notify_progress().await;

        // Create result storage
        let mut result = StreamingResult::new(
            dimension,
            chunk_size,
            self.config.compression,
            self.config.get_temp_dir(),
        );

        // Aggregate chunk by chunk
        for chunk_idx in 0..num_chunks {
            // Load all chunks for this index
            let mut chunk_data: Vec<Vec<f32>> = Vec::with_capacity(finalized.len());
            for state in &finalized {
                let chunk = state.gradient.load_chunk(chunk_idx).await?;
                chunk_data.push(chunk);
            }

            // Aggregate this chunk
            let aggregated = self.aggregate_chunk(&chunk_data)?;
            result.store_chunk(chunk_idx, aggregated).await?;

            // Update progress
            {
                let mut progress = self.progress.write().await;
                progress.chunks_processed += 1;
            }
            self.notify_progress().await;

            // Checkpoint if enabled
            if self.config.enable_checkpointing
                && chunk_idx % self.config.checkpoint_interval == 0
            {
                self.save_checkpoint().await?;
            }
        }

        // Mark complete
        {
            let mut progress = self.progress.write().await;
            progress.phase = StreamingPhase::Complete;
        }
        self.notify_progress().await;

        // Clean up checkpoint
        if let Some(ref path) = self.checkpoint_path {
            let _ = fs::remove_file(path).await;
        }

        Ok(result)
    }

    /// Get a result chunk from the last aggregation.
    pub async fn get_result_chunk(&self, _chunk_index: usize) -> Result<Vec<f32>> {
        // This would require storing the result, for now return error
        Err(AggregatorError::InvalidConfig(
            "Use StreamingResult from aggregate_streaming() to access chunks".to_string(),
        ))
    }

    /// Aggregate a single chunk from all nodes.
    fn aggregate_chunk(&self, chunks: &[Vec<f32>]) -> Result<Vec<f32>> {
        // Convert to gradients for the Byzantine aggregator
        let gradients: Vec<Gradient> = chunks.iter().map(|c| Array1::from(c.clone())).collect();

        let defense_config = DefenseConfig::with_defense(self.defense.clone());
        let aggregator = ByzantineAggregator::new(defense_config);

        let result = aggregator.aggregate(&gradients)?;
        Ok(result.to_vec())
    }

    /// Get current progress.
    pub async fn get_progress(&self) -> StreamingProgress {
        self.progress.read().await.clone()
    }

    /// Notify progress callback if set.
    async fn notify_progress(&self) {
        if let Some(ref callback) = self.progress_callback {
            let progress = self.progress.read().await.clone();
            callback(progress);
        }
    }

    /// Save checkpoint for fault tolerance.
    async fn save_checkpoint(&self) -> Result<()> {
        let Some(ref path) = self.checkpoint_path else {
            return Ok(());
        };

        let submissions = self.submissions.read().await;
        let mut active_submissions = Vec::new();
        let mut received_chunks = HashMap::new();

        for (id, state) in submissions.iter() {
            if !state.finalized {
                active_submissions.push((*id, state.handle.node_id.clone(), state.handle.total_dimension));
                let received: Vec<usize> = state
                    .gradient
                    .received_chunks
                    .iter()
                    .enumerate()
                    .filter(|(_, &r)| r)
                    .map(|(i, _)| i)
                    .collect();
                received_chunks.insert(*id, received);
            }
        }

        let progress = self.progress.read().await;
        let aggregation_chunk_index = if progress.phase == StreamingPhase::Aggregating {
            Some(progress.chunks_processed)
        } else {
            None
        };

        let checkpoint = StreamingCheckpoint {
            active_submissions,
            received_chunks,
            aggregation_chunk_index,
            timestamp: chrono::Utc::now(),
        };

        let json = serde_json::to_string_pretty(&checkpoint).map_err(|e| {
            AggregatorError::Internal(format!("Failed to serialize checkpoint: {}", e))
        })?;

        let temp_dir = self.config.get_temp_dir();
        fs::create_dir_all(&temp_dir).await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to create checkpoint dir: {}", e))
        })?;

        fs::write(path, json).await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to write checkpoint: {}", e))
        })?;

        tracing::debug!("Saved streaming checkpoint");
        Ok(())
    }

    /// Load checkpoint for resume.
    pub async fn load_checkpoint(&self) -> Result<Option<StreamingCheckpoint>> {
        let Some(ref path) = self.checkpoint_path else {
            return Ok(None);
        };

        if !path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(path).await.map_err(|e| {
            AggregatorError::Internal(format!("Failed to read checkpoint: {}", e))
        })?;

        let checkpoint: StreamingCheckpoint = serde_json::from_str(&json).map_err(|e| {
            AggregatorError::Internal(format!("Failed to parse checkpoint: {}", e))
        })?;

        Ok(Some(checkpoint))
    }

    /// Resume from a checkpoint.
    pub async fn resume_from_checkpoint(&self, checkpoint: StreamingCheckpoint) -> Result<()> {
        tracing::info!("Resuming from checkpoint at {:?}", checkpoint.timestamp);

        // Restore submission states
        let mut submissions = self.submissions.write().await;
        for (id, node_id, dimension) in checkpoint.active_submissions {
            let handle = SubmissionHandle {
                id,
                node_id: node_id.clone(),
                total_dimension: dimension,
                started_at: std::time::SystemTime::now(),
            };

            let mut gradient = ChunkedGradient::new(
                dimension,
                self.config.chunk_size,
                self.config.compression,
            );

            // Mark received chunks
            if let Some(received) = checkpoint.received_chunks.get(&id) {
                for &chunk_idx in received {
                    // Mark as received (data should still be on disk)
                    gradient.received_chunks[chunk_idx] = true;
                    let chunk_path = self.config.get_temp_dir()
                        .join(format!("chunk_{}_{}.bin", id, chunk_idx));
                    if chunk_path.exists() {
                        gradient.chunks[chunk_idx] = Some(ChunkStorage::Disk(chunk_path));
                    }
                }
            }

            let state = NodeSubmissionState {
                handle,
                gradient,
                finalized: false,
            };
            submissions.insert(id, state);
        }

        // Update handle ID counter
        if let Some(max_id) = checkpoint.received_chunks.keys().max() {
            self.next_handle_id.store(*max_id + 1, Ordering::SeqCst);
        }

        // Restore progress
        let mut progress = self.progress.write().await;
        progress.chunks_received = checkpoint
            .received_chunks
            .values()
            .map(|v| v.len())
            .sum();
        if let Some(chunk_idx) = checkpoint.aggregation_chunk_index {
            progress.chunks_processed = chunk_idx;
            progress.phase = StreamingPhase::Aggregating;
        }

        Ok(())
    }

    /// Handle node dropout mid-submission.
    pub async fn handle_node_dropout(&self, node_id: &str) -> Result<()> {
        let mut submissions = self.submissions.write().await;

        // Find and mark submission as dropped
        let handle_id = submissions
            .iter()
            .find(|(_, s)| s.handle.node_id == node_id && !s.finalized)
            .map(|(id, _)| *id);

        if let Some(id) = handle_id {
            if let Some(state) = submissions.remove(&id) {
                // Clean up chunk files
                state.gradient.cleanup().await?;

                // Update progress
                let mut progress = self.progress.write().await;
                let dropped_chunks = state.gradient.received_count();
                progress.chunks_received = progress.chunks_received.saturating_sub(dropped_chunks);
                progress.total_chunks = progress.total_chunks.saturating_sub(state.gradient.num_chunks);

                tracing::warn!("Node {} dropped, cleaned up {} chunks", node_id, dropped_chunks);
            }
        }

        Ok(())
    }

    /// Clean up all resources.
    pub async fn cleanup(&self) -> Result<()> {
        let mut submissions = self.submissions.write().await;
        for (_, state) in submissions.drain() {
            state.gradient.cleanup().await?;
        }

        let mut cache = self.chunk_cache.lock().await;
        cache.clear();

        // Clean up temp directory
        let temp_dir = self.config.get_temp_dir();
        if temp_dir.exists() {
            let _ = fs::remove_dir_all(&temp_dir).await;
        }

        Ok(())
    }

    /// Check if all expected nodes have submitted.
    pub async fn is_round_complete(&self) -> bool {
        let progress = self.progress.read().await;
        progress.completed_nodes >= self.expected_nodes && self.expected_nodes > 0
    }
}

/// Aggregate a single chunk from multiple nodes (standalone function).
pub fn aggregate_chunks(chunks: Vec<&[f32]>, defense: &Defense) -> Result<Vec<f32>> {
    let gradients: Vec<Gradient> = chunks.iter().map(|c| Array1::from(c.to_vec())).collect();

    let defense_config = DefenseConfig::with_defense(defense.clone());
    let aggregator = ByzantineAggregator::new(defense_config);

    let result = aggregator.aggregate(&gradients)?;
    Ok(result.to_vec())
}

/// Compress data using simple run-length encoding or deflate.
fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
    // Simple compression using flate2 if available, otherwise just return as-is
    // For now, use a simple length-prefixed format
    let mut output = Vec::with_capacity(data.len() + 8);
    output.extend_from_slice(&(data.len() as u64).to_le_bytes());
    output.extend_from_slice(data);
    Ok(output)
}

/// Decompress data.
fn decompress_data(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 8 {
        return Err(AggregatorError::Compression("Invalid compressed data".to_string()));
    }
    let _len = u64::from_le_bytes([
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7],
    ]) as usize;
    Ok(data[8..].to_vec())
}

/// Krum approximation for streaming (sample-based).
///
/// Since Krum requires full gradients for distance computation,
/// we provide a sampling-based approximation that:
/// 1. Samples a subset of chunks to estimate distances
/// 2. Uses these estimates to select the most representative gradients
/// 3. Averages the selected gradients chunk by chunk
pub struct StreamingKrumApproximator {
    #[allow(dead_code)]
    config: StreamingConfig,
    f: usize,
    sample_chunks: usize,
}

impl StreamingKrumApproximator {
    /// Create a new Krum approximator.
    pub fn new(config: StreamingConfig, f: usize, sample_chunks: usize) -> Self {
        Self {
            config,
            f,
            sample_chunks: sample_chunks.min(10), // Cap at 10 sample chunks
        }
    }

    /// Approximate Krum selection using sampled chunks.
    pub async fn select_gradients(
        &self,
        submissions: &[&ChunkedGradient],
    ) -> Result<Vec<usize>> {
        let n = submissions.len();
        if n <= 2 * self.f + 2 {
            return Err(AggregatorError::InsufficientGradients {
                have: n,
                need: 2 * self.f + 3,
                defense: format!("StreamingKrum(f={})", self.f),
            });
        }

        // Sample chunk indices to use for distance estimation
        let num_chunks = submissions[0].num_chunks;
        let sample_indices: Vec<usize> = if num_chunks <= self.sample_chunks {
            (0..num_chunks).collect()
        } else {
            // Evenly distributed samples
            (0..self.sample_chunks)
                .map(|i| i * num_chunks / self.sample_chunks)
                .collect()
        };

        // Load sample chunks and compute approximate distances
        let mut sample_data: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
        for submission in submissions {
            let mut node_samples = Vec::with_capacity(sample_indices.len());
            for &idx in &sample_indices {
                let chunk = submission.load_chunk(idx).await?;
                node_samples.push(chunk);
            }
            sample_data.push(node_samples);
        }

        // Compute Krum scores based on sample distances
        let take_count = n.saturating_sub(self.f).saturating_sub(2);
        let mut scores: Vec<f32> = Vec::with_capacity(n);

        for i in 0..n {
            let mut distances: Vec<f32> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Approximate distance from sampled chunks
                let mut dist_sq: f32 = 0.0;
                for k in 0..sample_indices.len() {
                    for (a, b) in sample_data[i][k].iter().zip(sample_data[j][k].iter()) {
                        dist_sq += (a - b).powi(2);
                    }
                }
                // Scale to estimate full distance
                let scale = (num_chunks as f32) / (sample_indices.len() as f32);
                distances.push((dist_sq * scale).sqrt());
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let score: f32 = distances.iter().take(take_count).sum();
            scores.push(score);
        }

        // Select gradient with minimum score
        let best_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| AggregatorError::Internal("No gradients to select".to_string()))?;

        Ok(vec![best_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config() -> StreamingConfig {
        StreamingConfig::default()
            .with_chunk_size(100)
            .with_max_memory_mb(1)
            .with_compression(false)
    }

    #[tokio::test]
    async fn test_basic_streaming_aggregation() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let aggregator = StreamingAggregator::new(config, Defense::FedAvg)
            .with_expected_nodes(2);

        // Node 1 submission
        let handle1 = aggregator.begin_submission("node1", 200).await.unwrap();
        aggregator.submit_chunk(&handle1, 0, &vec![1.0; 100]).await.unwrap();
        aggregator.submit_chunk(&handle1, 1, &vec![2.0; 100]).await.unwrap();
        aggregator.finalize_submission(handle1).await.unwrap();

        // Node 2 submission
        let handle2 = aggregator.begin_submission("node2", 200).await.unwrap();
        aggregator.submit_chunk(&handle2, 0, &vec![3.0; 100]).await.unwrap();
        aggregator.submit_chunk(&handle2, 1, &vec![4.0; 100]).await.unwrap();
        aggregator.finalize_submission(handle2).await.unwrap();

        // Aggregate
        let result = aggregator.aggregate_streaming().await.unwrap();
        assert_eq!(result.dimension, 200);
        assert_eq!(result.num_chunks, 2);

        // Check first chunk (average of 1.0 and 3.0 = 2.0)
        let chunk0 = result.get_chunk(0).await.unwrap();
        assert_eq!(chunk0.len(), 100);
        assert!((chunk0[0] - 2.0).abs() < 1e-6);

        // Check second chunk (average of 2.0 and 4.0 = 3.0)
        let chunk1 = result.get_chunk(1).await.unwrap();
        assert_eq!(chunk1.len(), 100);
        assert!((chunk1[0] - 3.0).abs() < 1e-6);

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_large_gradient_chunking() {
        let temp_dir = TempDir::new().unwrap();
        let chunk_size = 1000;
        let total_dim = 5500; // Will create 6 chunks (5 full + 1 partial)

        let config = StreamingConfig::default()
            .with_chunk_size(chunk_size)
            .with_temp_dir(temp_dir.path().to_path_buf())
            .with_compression(false);

        let aggregator = StreamingAggregator::new(config, Defense::FedAvg)
            .with_expected_nodes(1);

        let handle = aggregator.begin_submission("node1", total_dim).await.unwrap();

        // Submit chunks
        for i in 0..5 {
            aggregator.submit_chunk(&handle, i, &vec![i as f32; chunk_size]).await.unwrap();
        }
        // Last partial chunk
        aggregator.submit_chunk(&handle, 5, &vec![5.0; 500]).await.unwrap();

        aggregator.finalize_submission(handle).await.unwrap();

        let result = aggregator.aggregate_streaming().await.unwrap();
        assert_eq!(result.dimension, total_dim);
        assert_eq!(result.num_chunks, 6);

        // Verify last chunk size
        let last_chunk = result.get_chunk(5).await.unwrap();
        assert_eq!(last_chunk.len(), 500);

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_memory_limit_disk_spilling() {
        let temp_dir = TempDir::new().unwrap();
        // Very small memory limit to force disk spilling
        let config = StreamingConfig::default()
            .with_chunk_size(1000)
            .with_max_memory_mb(0) // Force all to disk
            .with_temp_dir(temp_dir.path().to_path_buf())
            .with_compression(false);

        let aggregator = StreamingAggregator::new(config, Defense::FedAvg)
            .with_expected_nodes(1);

        let handle = aggregator.begin_submission("node1", 2000).await.unwrap();
        aggregator.submit_chunk(&handle, 0, &vec![1.0; 1000]).await.unwrap();
        aggregator.submit_chunk(&handle, 1, &vec![2.0; 1000]).await.unwrap();
        aggregator.finalize_submission(handle).await.unwrap();

        // Check progress shows disk usage
        let progress = aggregator.get_progress().await;
        assert!(progress.chunks_on_disk > 0);

        // Should still aggregate correctly
        let result = aggregator.aggregate_streaming().await.unwrap();
        let chunk0 = result.get_chunk(0).await.unwrap();
        assert!((chunk0[0] - 1.0).abs() < 1e-6);

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_median_defense_streaming() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let aggregator = StreamingAggregator::new(config, Defense::Median)
            .with_expected_nodes(3);

        // Submit 3 nodes with one outlier
        for (node_id, value) in [("node1", 1.0), ("node2", 2.0), ("node3", 100.0)] {
            let handle = aggregator.begin_submission(node_id, 100).await.unwrap();
            aggregator.submit_chunk(&handle, 0, &vec![value; 100]).await.unwrap();
            aggregator.finalize_submission(handle).await.unwrap();
        }

        let result = aggregator.aggregate_streaming().await.unwrap();
        let chunk = result.get_chunk(0).await.unwrap();

        // Median of [1, 2, 100] = 2
        assert!((chunk[0] - 2.0).abs() < 1e-6);

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_trimmed_mean_defense_streaming() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let aggregator = StreamingAggregator::new(config, Defense::TrimmedMean { beta: 0.2 })
            .with_expected_nodes(5);

        // Submit 5 nodes
        for (node_id, value) in [
            ("node1", 0.0),   // trimmed
            ("node2", 1.0),
            ("node3", 2.0),
            ("node4", 3.0),
            ("node5", 100.0), // trimmed
        ] {
            let handle = aggregator.begin_submission(node_id, 100).await.unwrap();
            aggregator.submit_chunk(&handle, 0, &vec![value; 100]).await.unwrap();
            aggregator.finalize_submission(handle).await.unwrap();
        }

        let result = aggregator.aggregate_streaming().await.unwrap();
        let chunk = result.get_chunk(0).await.unwrap();

        // Mean of [1, 2, 3] = 2
        assert!((chunk[0] - 2.0).abs() < 1e-6);

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_partial_submission_error() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let aggregator = StreamingAggregator::new(config, Defense::FedAvg);

        let handle = aggregator.begin_submission("node1", 200).await.unwrap();
        // Only submit first chunk
        aggregator.submit_chunk(&handle, 0, &vec![1.0; 100]).await.unwrap();

        // Finalizing should fail
        let result = aggregator.finalize_submission(handle).await;
        assert!(result.is_err());

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_duplicate_submission_error() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let aggregator = StreamingAggregator::new(config, Defense::FedAvg);

        let _handle = aggregator.begin_submission("node1", 100).await.unwrap();

        // Second submission from same node should fail
        let result = aggregator.begin_submission("node1", 100).await;
        assert!(result.is_err());

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_dimension_mismatch_error() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let mut aggregator = StreamingAggregator::new(config, Defense::FedAvg);
        aggregator.expected_dimension = Some(100);

        let result = aggregator.begin_submission("node1", 200).await;
        assert!(matches!(result, Err(AggregatorError::DimensionMismatch { .. })));

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_node_dropout_handling() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let aggregator = StreamingAggregator::new(config, Defense::FedAvg)
            .with_expected_nodes(2);

        // Start submission that will be dropped
        let handle = aggregator.begin_submission("node1", 100).await.unwrap();
        aggregator.submit_chunk(&handle, 0, &vec![1.0; 100]).await.unwrap();

        // Handle dropout
        aggregator.handle_node_dropout("node1").await.unwrap();

        let progress = aggregator.get_progress().await;
        assert_eq!(progress.chunks_received, 0);

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_progress_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());

        let progress_updates = Arc::new(Mutex::new(Vec::new()));
        let updates_clone = progress_updates.clone();

        let callback: ProgressCallback = Box::new(move |p: StreamingProgress| {
            let updates = updates_clone.clone();
            tokio::spawn(async move {
                updates.lock().await.push(p);
            });
        });

        let aggregator = StreamingAggregator::new(config, Defense::FedAvg)
            .with_expected_nodes(1)
            .with_progress_callback(callback);

        let handle = aggregator.begin_submission("node1", 100).await.unwrap();
        aggregator.submit_chunk(&handle, 0, &vec![1.0; 100]).await.unwrap();
        aggregator.finalize_submission(handle).await.unwrap();

        let final_progress = aggregator.get_progress().await;
        assert_eq!(final_progress.completed_nodes, 1);
        assert_eq!(final_progress.chunks_received, 1);

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_checkpointing() {
        let temp_dir = TempDir::new().unwrap();
        let config = StreamingConfig::default()
            .with_chunk_size(100)
            .with_temp_dir(temp_dir.path().to_path_buf())
            .with_compression(false);

        let aggregator = StreamingAggregator::new(config.clone(), Defense::FedAvg);

        let handle = aggregator.begin_submission("node1", 200).await.unwrap();
        aggregator.submit_chunk(&handle, 0, &vec![1.0; 100]).await.unwrap();

        // Force checkpoint
        aggregator.save_checkpoint().await.unwrap();

        // Verify checkpoint exists
        let checkpoint = aggregator.load_checkpoint().await.unwrap();
        assert!(checkpoint.is_some());

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_streaming_result_to_full_gradient() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let aggregator = StreamingAggregator::new(config, Defense::FedAvg)
            .with_expected_nodes(1);

        let handle = aggregator.begin_submission("node1", 200).await.unwrap();
        aggregator.submit_chunk(&handle, 0, &vec![1.0; 100]).await.unwrap();
        aggregator.submit_chunk(&handle, 1, &vec![2.0; 100]).await.unwrap();
        aggregator.finalize_submission(handle).await.unwrap();

        let result = aggregator.aggregate_streaming().await.unwrap();
        let full = result.to_full_gradient().await.unwrap();

        assert_eq!(full.len(), 200);
        assert!((full[0] - 1.0).abs() < 1e-6);
        assert!((full[100] - 2.0).abs() < 1e-6);

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_streaming_result_write_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        let aggregator = StreamingAggregator::new(config, Defense::FedAvg)
            .with_expected_nodes(1);

        let handle = aggregator.begin_submission("node1", 100).await.unwrap();
        aggregator.submit_chunk(&handle, 0, &vec![1.5; 100]).await.unwrap();
        aggregator.finalize_submission(handle).await.unwrap();

        let result = aggregator.aggregate_streaming().await.unwrap();
        let output_path = temp_dir.path().join("output.bin");
        result.write_to_file(&output_path).await.unwrap();

        assert!(output_path.exists());

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_aggregate_chunks_function() {
        let chunk1 = vec![1.0, 2.0, 3.0];
        let chunk2 = vec![4.0, 5.0, 6.0];
        let chunk3 = vec![7.0, 8.0, 9.0];

        let result = aggregate_chunks(
            vec![&chunk1[..], &chunk2[..], &chunk3[..]],
            &Defense::FedAvg,
        ).unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_insufficient_gradients_error() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        // Krum with f=1 requires at least 5 nodes
        let aggregator = StreamingAggregator::new(config, Defense::Krum { f: 1 })
            .with_expected_nodes(2);

        // Only 2 nodes
        for node_id in ["node1", "node2"] {
            let handle = aggregator.begin_submission(node_id, 100).await.unwrap();
            aggregator.submit_chunk(&handle, 0, &vec![1.0; 100]).await.unwrap();
            aggregator.finalize_submission(handle).await.unwrap();
        }

        let result = aggregator.aggregate_streaming().await;
        assert!(matches!(result, Err(AggregatorError::InsufficientGradients { .. })));

        aggregator.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_krum_streaming_with_enough_nodes() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config().with_temp_dir(temp_dir.path().to_path_buf());
        // Krum with f=1 requires at least 5 nodes
        let aggregator = StreamingAggregator::new(config, Defense::Krum { f: 1 })
            .with_expected_nodes(5);

        // 5 nodes: 4 normal + 1 Byzantine
        for (node_id, value) in [
            ("node1", 1.0),
            ("node2", 1.1),
            ("node3", 1.2),
            ("node4", 0.9),
            ("node5", 100.0), // Byzantine
        ] {
            let handle = aggregator.begin_submission(node_id, 100).await.unwrap();
            aggregator.submit_chunk(&handle, 0, &vec![value; 100]).await.unwrap();
            aggregator.finalize_submission(handle).await.unwrap();
        }

        let result = aggregator.aggregate_streaming().await.unwrap();
        let chunk = result.get_chunk(0).await.unwrap();

        // Krum should select a non-Byzantine gradient
        assert!(chunk[0] < 10.0);

        aggregator.cleanup().await.unwrap();
    }

    // Benchmark test (run with --release for meaningful results)
    #[tokio::test]
    #[ignore] // Run manually with: cargo test benchmark_memory_usage --release -- --ignored
    async fn benchmark_memory_usage() {
        use std::time::Instant;

        let temp_dir = TempDir::new().unwrap();
        let dimension = 10_000_000; // 10M parameters
        let chunk_size = 1_000_000;
        let num_nodes = 10;

        let config = StreamingConfig::default()
            .with_chunk_size(chunk_size)
            .with_max_memory_mb(256) // 256MB limit
            .with_temp_dir(temp_dir.path().to_path_buf());

        let aggregator = StreamingAggregator::new(config, Defense::FedAvg)
            .with_expected_nodes(num_nodes);

        let start = Instant::now();

        // Simulate submissions
        for i in 0..num_nodes {
            let node_id = format!("node{}", i);
            let handle = aggregator.begin_submission(&node_id, dimension).await.unwrap();

            for chunk_idx in 0..(dimension / chunk_size) {
                let data: Vec<f32> = (0..chunk_size).map(|j| (i * chunk_size + j) as f32).collect();
                aggregator.submit_chunk(&handle, chunk_idx, &data).await.unwrap();
            }

            aggregator.finalize_submission(handle).await.unwrap();
        }

        let submission_time = start.elapsed();

        let agg_start = Instant::now();
        let result = aggregator.aggregate_streaming().await.unwrap();
        let aggregation_time = agg_start.elapsed();

        let progress = aggregator.get_progress().await;

        println!("Benchmark Results:");
        println!("  Dimension: {} ({} MB)", dimension, dimension * 4 / 1024 / 1024);
        println!("  Nodes: {}", num_nodes);
        println!("  Chunks per node: {}", dimension / chunk_size);
        println!("  Submission time: {:?}", submission_time);
        println!("  Aggregation time: {:?}", aggregation_time);
        println!("  Total time: {:?}", start.elapsed());
        println!("  Peak memory: {} MB", progress.memory_bytes / 1024 / 1024);
        println!("  Chunks on disk: {}", progress.chunks_on_disk);
        println!("  Result dimension: {}", result.dimension);

        aggregator.cleanup().await.unwrap();
    }
}
