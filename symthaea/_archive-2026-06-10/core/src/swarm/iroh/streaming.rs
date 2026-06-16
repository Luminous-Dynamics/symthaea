// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tensor Streaming over Iroh
//!
//! High-performance bi-directional streaming for neural state synchronization.
//! Designed for <50ms latency tensor exchange between consciousness nodes.

use crate::swarm::{ConsciousnessVector, SwarmError, SwarmResult, TensorPayload};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

/// Configuration for tensor streaming
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum message size in bytes
    pub max_message_size: usize,

    /// Buffer size for send queue
    pub send_buffer_size: usize,

    /// Buffer size for receive queue
    pub recv_buffer_size: usize,

    /// Timeout for send operations
    pub send_timeout: Duration,

    /// Timeout for receive operations
    pub recv_timeout: Duration,

    /// Enable compression for large tensors
    pub compression_enabled: bool,

    /// Compression threshold (bytes)
    pub compression_threshold: usize,

    /// Target frames per second for continuous streaming
    pub target_fps: u32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_message_size: 1024 * 1024, // 1 MB
            send_buffer_size: 16,
            recv_buffer_size: 16,
            send_timeout: Duration::from_millis(100),
            recv_timeout: Duration::from_millis(100),
            compression_enabled: true,
            compression_threshold: 4096,
            target_fps: 30,
        }
    }
}

impl StreamConfig {
    /// Config optimized for consciousness sync (smaller, faster)
    pub fn consciousness_sync() -> Self {
        Self {
            max_message_size: 16 * 1024, // 16 KB
            send_timeout: Duration::from_millis(50),
            recv_timeout: Duration::from_millis(50),
            compression_enabled: false, // Skip compression for speed
            target_fps: 60,
            ..Default::default()
        }
    }

    /// Config optimized for weight streaming (larger, can be slower)
    pub fn weight_streaming() -> Self {
        Self {
            max_message_size: 16 * 1024 * 1024, // 16 MB
            send_timeout: Duration::from_millis(500),
            recv_timeout: Duration::from_millis(500),
            compression_enabled: true,
            compression_threshold: 1024,
            target_fps: 10,
            ..Default::default()
        }
    }
}

/// A bi-directional tensor stream between two nodes
pub struct TensorStream {
    /// Stream configuration
    config: StreamConfig,

    /// Sequence number for ordering
    sequence: Arc<Mutex<u64>>,

    /// Last send timestamp
    last_send: Arc<Mutex<std::time::Instant>>,

    /// Statistics
    stats: Arc<Mutex<StreamStats>>,
}

/// Statistics for a tensor stream
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total messages sent
    pub messages_sent: u64,

    /// Total messages received
    pub messages_received: u64,

    /// Total bytes sent
    pub bytes_sent: u64,

    /// Total bytes received
    pub bytes_received: u64,

    /// Average send latency (milliseconds)
    pub avg_send_latency_ms: f64,

    /// Average receive latency (milliseconds)
    pub avg_recv_latency_ms: f64,

    /// Dropped messages (buffer full)
    pub dropped_messages: u64,

    /// Compression ratio (if enabled)
    pub compression_ratio: f64,
}

impl TensorStream {
    /// Create a new tensor stream
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            sequence: Arc::new(Mutex::new(0)),
            last_send: Arc::new(Mutex::new(std::time::Instant::now())),
            stats: Arc::new(Mutex::new(StreamStats::default())),
        }
    }

    /// Prepare a consciousness vector for streaming
    pub fn prepare_consciousness(&self, state: &ConsciousnessVector) -> SwarmResult<Vec<u8>> {
        // Update sequence number
        let mut seq = self.sequence.lock();
        *seq += 1;
        let mut state = state.clone();
        state.sequence = *seq;
        drop(seq);

        // Serialize
        let bytes = bincode::serialize(&state)
            .map_err(|e| SwarmError::SerializationError(e.to_string()))?;

        // Check size
        if bytes.len() > self.config.max_message_size {
            return Err(SwarmError::TensorStreamError {
                reason: format!(
                    "Message too large: {} bytes (max: {})",
                    bytes.len(),
                    self.config.max_message_size
                ),
            });
        }

        // Update stats
        let mut stats = self.stats.lock();
        stats.bytes_sent += bytes.len() as u64;
        stats.messages_sent += 1;

        Ok(bytes)
    }

    /// Prepare a tensor payload for streaming
    pub fn prepare_tensor(&self, tensor: &TensorPayload) -> SwarmResult<Vec<u8>> {
        // Serialize
        let bytes = bincode::serialize(tensor)
            .map_err(|e| SwarmError::SerializationError(e.to_string()))?;

        // Check size
        if bytes.len() > self.config.max_message_size {
            return Err(SwarmError::TensorStreamError {
                reason: format!(
                    "Tensor too large: {} bytes (max: {})",
                    bytes.len(),
                    self.config.max_message_size
                ),
            });
        }

        // Optionally compress large tensors
        let final_bytes =
            if self.config.compression_enabled && bytes.len() > self.config.compression_threshold {
                self.compress(&bytes)?
            } else {
                bytes
            };

        // Update stats
        let mut stats = self.stats.lock();
        stats.bytes_sent += final_bytes.len() as u64;
        stats.messages_sent += 1;

        Ok(final_bytes)
    }

    /// Decompress and deserialize received consciousness data
    pub fn receive_consciousness(&self, data: &[u8]) -> SwarmResult<ConsciousnessVector> {
        // Decompress if needed
        let bytes = if self.is_compressed(data) {
            self.decompress(data)?
        } else {
            data.to_vec()
        };

        // Deserialize
        let state: ConsciousnessVector = bincode::deserialize(&bytes)
            .map_err(|e| SwarmError::SerializationError(e.to_string()))?;

        // Update stats
        let mut stats = self.stats.lock();
        stats.bytes_received += data.len() as u64;
        stats.messages_received += 1;

        Ok(state)
    }

    /// Decompress and deserialize received tensor data
    pub fn receive_tensor(&self, data: &[u8]) -> SwarmResult<TensorPayload> {
        // Decompress if needed
        let bytes = if self.is_compressed(data) {
            self.decompress(data)?
        } else {
            data.to_vec()
        };

        // Deserialize
        let tensor: TensorPayload = bincode::deserialize(&bytes)
            .map_err(|e| SwarmError::SerializationError(e.to_string()))?;

        // Update stats
        let mut stats = self.stats.lock();
        stats.bytes_received += data.len() as u64;
        stats.messages_received += 1;

        Ok(tensor)
    }

    /// Get stream statistics
    pub fn stats(&self) -> StreamStats {
        self.stats.lock().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.lock() = StreamStats::default();
    }

    /// Check if we should throttle sending (rate limiting)
    pub fn should_throttle(&self) -> bool {
        let last = self.last_send.lock();
        let frame_duration = Duration::from_secs_f64(1.0 / self.config.target_fps as f64);
        last.elapsed() < frame_duration
    }

    /// Mark a send as complete (for rate limiting)
    pub fn mark_sent(&self) {
        *self.last_send.lock() = std::time::Instant::now();
    }

    /// Compress data (simple LZ4-style prefix)
    fn compress(&self, data: &[u8]) -> SwarmResult<Vec<u8>> {
        // Simple compression marker + original data
        // In production, use LZ4 or zstd
        let mut compressed = vec![0xC0]; // Compression marker
        compressed.extend_from_slice(data);
        Ok(compressed)
    }

    /// Decompress data
    fn decompress(&self, data: &[u8]) -> SwarmResult<Vec<u8>> {
        if data.is_empty() {
            return Err(SwarmError::TensorStreamError {
                reason: "Empty compressed data".to_string(),
            });
        }
        // Skip compression marker
        Ok(data[1..].to_vec())
    }

    /// Check if data is compressed
    fn is_compressed(&self, data: &[u8]) -> bool {
        !data.is_empty() && data[0] == 0xC0
    }
}

impl Default for TensorStream {
    fn default() -> Self {
        Self::new(StreamConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();
        assert_eq!(config.max_message_size, 1024 * 1024);
        assert!(config.compression_enabled);
    }

    #[test]
    fn test_consciousness_sync_config() {
        let config = StreamConfig::consciousness_sync();
        assert_eq!(config.target_fps, 60);
        assert!(!config.compression_enabled);
    }

    #[test]
    fn test_prepare_consciousness() {
        let stream = TensorStream::new(StreamConfig::default());
        let state = ConsciousnessVector::new(vec![0.0; 64], 0.5);

        let bytes = stream.prepare_consciousness(&state).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_receive_consciousness() {
        let stream = TensorStream::new(StreamConfig::default());
        let state = ConsciousnessVector::new(vec![0.0; 64], 0.5);

        let bytes = stream.prepare_consciousness(&state).unwrap();
        let received = stream.receive_consciousness(&bytes).unwrap();

        assert_eq!(received.phi, 0.5);
        assert_eq!(received.attention.len(), 64);
    }

    #[test]
    fn test_stats_tracking() {
        let stream = TensorStream::new(StreamConfig::default());
        let state = ConsciousnessVector::new(vec![0.0; 64], 0.5);

        let _ = stream.prepare_consciousness(&state);
        let stats = stream.stats();

        assert_eq!(stats.messages_sent, 1);
        assert!(stats.bytes_sent > 0);
    }
}
