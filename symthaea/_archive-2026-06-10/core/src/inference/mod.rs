// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Inference Module: Production Deployment for Symthaea Neural Networks
//!
//! This module provides production-ready inference capabilities including:
//! - **Streaming inference** for real-time applications
//! - **Batch inference** for high-throughput processing
//! - **Async support** with tokio integration
//!
//! ## Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        INFERENCE ARCHITECTURE                               │
//! │                                                                             │
//! │   Input Stream                 Processing                 Output Stream    │
//! │   ───────────────             ─────────────              ───────────────   │
//! │                                                                             │
//! │   ┌───────────┐              ┌─────────────┐             ┌───────────┐     │
//! │   │   Push    │─────────────▶│   Ring      │             │   Poll    │     │
//! │   │  (single) │              │   Buffer    │             │  (single) │     │
//! │   └───────────┘              └──────┬──────┘             └─────▲─────┘     │
//! │                                     │                          │           │
//! │   ┌───────────┐              ┌──────▼──────┐             ┌─────┴─────┐     │
//! │   │   Push    │─────────────▶│  Pending    │─────────────▶  Output   │     │
//! │   │  (batch)  │              │   Batch     │             │   Queue   │     │
//! │   └───────────┘              └──────┬──────┘             └───────────┘     │
//! │                                     │                                       │
//! │                              ┌──────▼──────┐                                │
//! │                              │    CfC      │                                │
//! │                              │  Network    │                                │
//! │                              │ (stateful)  │                                │
//! │                              └─────────────┘                                │
//! │                                                                             │
//! │   Triggers:                                                                 │
//! │   • Batch size reached (batch_accumulation)                                │
//! │   • Latency timeout (max_latency_ms)                                       │
//! │   • Manual flush()                                                          │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ### Synchronous Streaming
//!
//! ```rust,ignore
//! use symthaea::inference::{StreamingInference, StreamingConfig};
//! use symthaea::dynamics::cfc::{CfCNetwork, CfCNetworkConfig};
//!
//! // Create network and streaming inference
//! let network = CfCNetwork::new(CfCNetworkConfig::default());
//! let streamer = StreamingInference::new(network, StreamingConfig::low_latency());
//!
//! // Push inputs as they arrive
//! for input in sensor_stream {
//!     streamer.push(input);
//!
//!     // Check for outputs
//!     while let Some(output) = streamer.poll() {
//!         handle_output(output);
//!     }
//! }
//! ```
//!
//! ### Asynchronous Streaming (with tokio feature)
//!
//! ```rust,ignore
//! use symthaea::inference::{AsyncStreamingInference, StreamingConfig};
//! use symthaea::dynamics::cfc::{CfCNetwork, CfCNetworkConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let network = CfCNetwork::new(CfCNetworkConfig::default());
//!     let mut async_streamer = AsyncStreamingInference::new(
//!         network,
//!         StreamingConfig::balanced(),
//!     );
//!
//!     // Spawn output handler
//!     let handle = tokio::spawn(async move {
//!         while let Some(output) = async_streamer.recv().await {
//!             handle_output(output).await;
//!         }
//!     });
//!
//!     // Push inputs
//!     for input in sensor_stream {
//!         async_streamer.push(input).await.unwrap();
//!     }
//! }
//! ```
//!
//! ## Configuration Presets
//!
//! | Preset | Batch Size | Max Latency | Use Case |
//! |--------|------------|-------------|----------|
//! | `low_latency()` | 1 | 10ms | Real-time control |
//! | `balanced()` | 8 | 25ms | Interactive apps |
//! | `high_throughput()` | 16 | 100ms | Batch processing |
//!
//! ## Backpressure Handling
//!
//! When the output queue fills up, behavior depends on `drop_on_backpressure`:
//! - `true`: Old outputs are dropped to make room (real-time mode)
//! - `false`: Queue grows temporarily (batch mode)
//!
//! ## State Management
//!
//! The streaming inference maintains network state across calls, enabling:
//! - Continuous temporal learning
//! - State checkpointing for recovery
//! - State synchronization across replicas

pub mod streaming;

// Re-export main types
pub use streaming::{StreamingConfig, StreamingInference, StreamingOutput, StreamingStats};

pub use streaming::AsyncStreamingInference;

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;

/// Create a streaming inference with default configuration
pub fn default_streaming(input_dim: usize, output_dim: usize) -> StreamingInference {
    use crate::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};

    let config = CfCNetworkConfig {
        input_dim,
        hidden_dim: 128,
        num_layers: 2,
        output_dim,
        cell_config: CfCConfig {
            input_dim,
            hidden_dim: 128,
            ..Default::default()
        },
        ..Default::default()
    };

    StreamingInference::new(CfCNetwork::new(config), StreamingConfig::default())
}

/// Create a low-latency streaming inference
pub fn low_latency_streaming(input_dim: usize, output_dim: usize) -> StreamingInference {
    use crate::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};

    let config = CfCNetworkConfig {
        input_dim,
        hidden_dim: 64, // Smaller for speed
        num_layers: 1,  // Single layer for speed
        output_dim,
        cell_config: CfCConfig {
            input_dim,
            hidden_dim: 64,
            use_backbone: false, // No backbone for speed
            ..Default::default()
        },
        ..Default::default()
    };

    StreamingInference::new(CfCNetwork::new(config), StreamingConfig::low_latency())
}

/// Create a high-capacity streaming inference
pub fn high_capacity_streaming(input_dim: usize, output_dim: usize) -> StreamingInference {
    use crate::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};

    let config = CfCNetworkConfig {
        input_dim,
        hidden_dim: 256,
        num_layers: 4,
        output_dim,
        cell_config: CfCConfig {
            input_dim,
            hidden_dim: 256,
            use_backbone: true,
            backbone_layers: 2,
            backbone_dim: 256,
            ..Default::default()
        },
        ..Default::default()
    };

    StreamingInference::new(CfCNetwork::new(config), StreamingConfig::high_throughput())
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Benchmark results for streaming inference
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Throughput in samples per second
    pub throughput_sps: f64,
    /// Latency percentiles (p50, p95, p99) in microseconds
    pub latency_p50_us: u64,
    pub latency_p95_us: u64,
    pub latency_p99_us: u64,
    /// Memory usage in bytes (estimated)
    pub memory_bytes: usize,
    /// Total samples processed
    pub total_samples: u64,
    /// Duration of benchmark in seconds
    pub duration_secs: f64,
}

/// Run a throughput benchmark
pub fn benchmark_throughput(
    streamer: &StreamingInference,
    input_dim: usize,
    num_samples: usize,
) -> BenchmarkResults {
    use std::time::Instant;

    let mut latencies = Vec::with_capacity(num_samples);
    let start = Instant::now();

    for i in 0..num_samples {
        let input = Array1::from_shape_fn(input_dim, |j| ((i + j) as f32).sin());
        streamer.push(input);

        // Drain outputs and record latency
        while let Some(output) = streamer.poll() {
            latencies.push(output.latency_us);
        }
    }

    // Flush any remaining
    while let Some(output) = streamer.flush() {
        latencies.push(output.latency_us);
    }

    let duration = start.elapsed();
    let duration_secs = duration.as_secs_f64();

    // Sort latencies for percentiles
    latencies.sort_unstable();

    let p50_idx = latencies.len() / 2;
    let p95_idx = (latencies.len() as f64 * 0.95) as usize;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;

    BenchmarkResults {
        throughput_sps: num_samples as f64 / duration_secs,
        latency_p50_us: *latencies.get(p50_idx).unwrap_or(&0),
        latency_p95_us: *latencies.get(p95_idx).unwrap_or(&0),
        latency_p99_us: *latencies.get(p99_idx).unwrap_or(&0),
        memory_bytes: estimate_memory(streamer),
        total_samples: num_samples as u64,
        duration_secs,
    }
}

/// Estimate memory usage of streaming inference
fn estimate_memory(streamer: &StreamingInference) -> usize {
    let config = streamer.config();

    // Ring buffer: capacity * (input_dim * 4 bytes + dt + timestamp)
    let ring_buffer_size = config.buffer_size * (64 * 4 + 8 + 16); // Approximate

    // Output queue: max_queue * output_size
    let output_queue_size = config.max_output_queue * 16 * 4; // Approximate output dim

    // Network state (estimated)
    let network_state_size = 128 * 128 * 4 * 2; // hidden_dim^2 * f32 * num_layers

    ring_buffer_size + output_queue_size + network_state_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_streaming() {
        let streamer = default_streaming(64, 16);
        assert!(streamer.is_active());
        assert!(!streamer.is_warmed_up());
    }

    #[test]
    fn test_low_latency_streaming() {
        let streamer = low_latency_streaming(32, 8);
        assert_eq!(streamer.config().batch_accumulation, 1);
        assert_eq!(streamer.config().max_latency_ms, 10);
    }

    #[test]
    fn test_benchmark() {
        let streamer = default_streaming(64, 16);
        let results = benchmark_throughput(&streamer, 64, 100);

        assert!(results.throughput_sps > 0.0);
        assert!(results.total_samples == 100);
    }
}
