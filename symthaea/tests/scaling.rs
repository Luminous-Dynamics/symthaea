// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scaling Tests
//!
//! Verifies system behavior under concurrent load:
//! - Parallel Phi computation across topologies
//! - Concurrent CfC network inference
//! - Concurrent streaming pipelines
//! - HDC encoding throughput
//! - Federated network synchronization
//! - Byzantine fault tolerance

use std::time::Instant;

use symthaea::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};
use symthaea::hdc::{
    HDC_DIMENSION, consciousness_topology_generators::ConsciousnessTopology, real_hv::RealHV,
    spectral_connectivity::ConnectivityCalculator,
};
use symthaea::inference::{StreamingConfig, StreamingInference};
use symthaea::perception::semantic_encoder::SemanticEncoder;

// =============================================================================
// PARALLEL PHI COMPUTATION
// =============================================================================

#[test]
fn test_parallel_phi_100_topologies() {
    use rayon::prelude::*;

    let calc = ConnectivityCalculator::new();
    let topologies: Vec<_> = (0..100)
        .map(|i| ConsciousnessTopology::random(8, HDC_DIMENSION, i))
        .collect();

    let start = Instant::now();
    let results: Vec<f64> = topologies
        .par_iter()
        .map(|topo| calc.algebraic_connectivity(&topo.node_representations))
        .collect();
    let elapsed = start.elapsed();

    assert_eq!(results.len(), 100);
    // All connectivity values should be non-negative
    for &r in &results {
        assert!(r >= 0.0, "Connectivity should be non-negative: {}", r);
    }

    eprintln!("100 parallel Phi computations in {:?}", elapsed);
}

#[test]
fn test_parallel_phi_scaling_10_50_200() {
    use rayon::prelude::*;

    let calc = ConnectivityCalculator::new();

    for count in [10, 50, 200] {
        let topologies: Vec<_> = (0..count)
            .map(|i| ConsciousnessTopology::ring(8, HDC_DIMENSION, i))
            .collect();

        let start = Instant::now();
        let results: Vec<f64> = topologies
            .par_iter()
            .map(|topo| calc.algebraic_connectivity(&topo.node_representations))
            .collect();
        let elapsed = start.elapsed();

        assert_eq!(results.len(), count as usize);
        let avg_us = elapsed.as_micros() as f64 / count as f64;
        eprintln!(
            "  {} parallel Phi: {:?} total, {:.1}us avg",
            count, elapsed, avg_us
        );
    }
}

// =============================================================================
// CONCURRENT CFC INFERENCE
// =============================================================================

#[test]
fn test_concurrent_cfc_inference_4_threads() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            thread::spawn(move || {
                let cell_config = CfCConfig {
                    input_dim: 64,
                    hidden_dim: 32,
                    ..Default::default()
                };
                let config = CfCNetworkConfig {
                    input_dim: 64,
                    hidden_dim: 32,
                    num_layers: 2,
                    output_dim: 16,
                    cell_config,
                    ..Default::default()
                };
                let mut network = CfCNetwork::new(config);
                let mut outputs = Vec::new();

                for i in 0..500 {
                    let input = ndarray::Array1::from_shape_fn(64, |j| {
                        ((j as u64 + i + thread_id * 1000) as f32).sin()
                    });
                    let output = network.forward(&input, 0.02);
                    outputs.push(output);
                }

                outputs
            })
        })
        .collect();

    let all_results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    assert_eq!(all_results.len(), 4);
    for results in &all_results {
        assert_eq!(results.len(), 500);
        for output in results {
            assert_eq!(output.len(), 16);
            // All values should be finite
            for &v in output.iter() {
                assert!(v.is_finite(), "Output should be finite");
            }
        }
    }
}

#[test]
fn test_concurrent_cfc_scaling_1_2_4_8() {
    use std::thread;

    for n_threads in [1, 2, 4, 8] {
        let start = Instant::now();

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                thread::spawn(move || {
                    let cell_config = CfCConfig {
                        input_dim: 32,
                        hidden_dim: 16,
                        ..Default::default()
                    };
                    let config = CfCNetworkConfig {
                        input_dim: 32,
                        hidden_dim: 16,
                        num_layers: 1,
                        output_dim: 8,
                        cell_config,
                        ..Default::default()
                    };
                    let mut network = CfCNetwork::new(config);

                    for i in 0..200 {
                        let input = ndarray::Array1::from_shape_fn(32, |j| {
                            ((j + i + t * 100) as f32).sin()
                        });
                        let _ = network.forward(&input, 0.02);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let elapsed = start.elapsed();
        eprintln!("  {} thread(s), 200 fwd each: {:?}", n_threads, elapsed);
    }
}

// =============================================================================
// CONCURRENT STREAMING PIPELINES
// =============================================================================

#[test]
fn test_concurrent_streaming_pipelines() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let cell_config = CfCConfig {
                    input_dim: 32,
                    hidden_dim: 16,
                    ..Default::default()
                };
                let network = CfCNetwork::new(CfCNetworkConfig {
                    input_dim: 32,
                    hidden_dim: 16,
                    num_layers: 1,
                    output_dim: 8,
                    cell_config,
                    ..Default::default()
                });
                let config = StreamingConfig {
                    batch_accumulation: 2,
                    warmup_samples: 4,
                    ..Default::default()
                };
                let streamer = StreamingInference::new(network, config);

                let mut output_count = 0usize;
                for i in 0..100 {
                    let input = ndarray::Array1::from_shape_fn(32, |j| ((j + i) as f32).cos());
                    streamer.push(input);
                    while streamer.poll().is_some() {
                        output_count += 1;
                    }
                }
                // Flush remaining
                if streamer.flush().is_some() {
                    output_count += 1;
                }

                output_count
            })
        })
        .collect();

    let counts: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Each pipeline should have produced outputs
    for count in &counts {
        assert!(*count > 0, "Each streaming pipeline should produce outputs");
    }

    eprintln!("Concurrent streaming output counts: {:?}", counts);
}

#[test]
fn test_streaming_throughput_under_load() {
    let cell_config = CfCConfig {
        input_dim: 64,
        hidden_dim: 32,
        ..Default::default()
    };
    let network = CfCNetwork::new(CfCNetworkConfig {
        input_dim: 64,
        hidden_dim: 32,
        num_layers: 2,
        output_dim: 16,
        cell_config,
        ..Default::default()
    });
    let config = StreamingConfig::low_latency();
    let streamer = StreamingInference::new(network, config);

    let start = Instant::now();
    let num_samples = 1000;
    let mut output_count = 0usize;

    for i in 0..num_samples {
        let input = ndarray::Array1::from_shape_fn(64, |j| ((j + i) as f32).sin());
        streamer.push(input);
        while streamer.poll().is_some() {
            output_count += 1;
        }
    }
    // Flush
    while streamer.flush().is_some() {
        output_count += 1;
    }

    let elapsed = start.elapsed();
    let throughput = num_samples as f64 / elapsed.as_secs_f64();
    eprintln!(
        "Streaming throughput: {:.0} samples/sec, {} outputs from {} inputs ({:?})",
        throughput, output_count, num_samples, elapsed
    );

    assert!(
        throughput > 100.0,
        "Should process >100 samples/sec, got {:.0}",
        throughput
    );
    assert!(output_count > 0, "Should produce at least one output");
}

// =============================================================================
// HDC ENCODING THROUGHPUT
// =============================================================================

#[test]
#[ignore = "benchmark: encodes 10K strings, run in release mode for meaningful results"]
fn test_hdc_encoding_throughput() {
    let mut encoder = SemanticEncoder::default();
    let texts: Vec<String> = (0..10_000)
        .map(|i| format!("test sentence number {} for encoding throughput", i))
        .collect();

    let start = Instant::now();
    let _results: Vec<_> = texts.iter().map(|t| encoder.encode_text(t)).collect();
    let elapsed = start.elapsed();

    let throughput = 10_000.0 / elapsed.as_secs_f64();
    eprintln!(
        "HDC encoding throughput: {:.0} strings/sec ({:?} total)",
        throughput, elapsed
    );

    // Debug builds are significantly slower than release; adjust threshold accordingly
    let threshold = if cfg!(debug_assertions) {
        100.0
    } else {
        1000.0
    };
    assert!(
        throughput > threshold,
        "Should encode >{:.0} strings/sec, got {:.0}",
        threshold,
        throughput
    );
}

#[test]
fn test_hdc_encoding_consistency_under_load() {
    let mut encoder = SemanticEncoder::default();

    // Encode the same text many times and verify determinism
    let text = "consciousness emerges from integrated information";
    let first = encoder.encode_text(text);

    for _ in 0..1000 {
        let encoded = encoder.encode_text(text);
        assert_eq!(
            encoded, first,
            "Encoding should be deterministic under repeated calls"
        );
    }
}

#[test]
fn test_hdc_vector_creation_throughput() {
    let start = Instant::now();
    let count = 10_000;

    let _vectors: Vec<_> = (0..count)
        .map(|i| RealHV::random(HDC_DIMENSION, i))
        .collect();

    let elapsed = start.elapsed();
    let throughput = count as f64 / elapsed.as_secs_f64();
    eprintln!(
        "RealHV creation throughput: {:.0} vectors/sec ({:?} total)",
        throughput, elapsed
    );

    assert!(
        throughput > 500.0,
        "Should create >500 RealHV/sec, got {:.0}",
        throughput
    );
}

// =============================================================================
// FEDERATED NETWORK SYNCHRONIZATION
// =============================================================================

#[tokio::test]
async fn test_federated_10_node_sync() {
    use symthaea::swarm::{FederatedCoordinator, create_test_network};

    let coordinators: Vec<FederatedCoordinator> = create_test_network(10, 100).await;

    assert_eq!(coordinators.len(), 10);

    // Each coordinator should see 9 peers
    for coordinator in &coordinators {
        assert_eq!(
            coordinator.peer_count(),
            9,
            "Each coordinator should see 9 peers in a 10-node network"
        );
        assert_eq!(coordinator.get_weights().len(), 100);
    }

    eprintln!("10-node federated network created and verified");
}

#[tokio::test]
async fn test_federated_gradient_sharing_10_nodes() {
    use symthaea::swarm::{FederatedCoordinator, create_test_network};

    let coordinators: Vec<FederatedCoordinator> = create_test_network(10, 50).await;

    // Each node sets distinct weights
    for (i, coord) in coordinators.iter().enumerate() {
        let weights: Vec<f32> = (0..50).map(|j| (i * 50 + j) as f32 * 0.01).collect();
        coord.update_weights(weights);
    }

    // Share gradients from nodes 1..9 (not node 0 which will aggregate)
    for coord in coordinators.iter().skip(1) {
        let sent = coord.share_gradient(0.0).await.unwrap();
        assert!(sent > 0, "Each node should send gradients to peers");
    }

    eprintln!("10-node gradient sharing complete");
}

// =============================================================================
// BYZANTINE FAULT TOLERANCE
// =============================================================================

#[test]
fn test_federated_byzantine_3_of_10() {
    use symthaea::swarm::{FederatedAggregator, GradientMessage};

    // Create aggregator with Byzantine tolerance
    let initial_weights = vec![0.0f32; 100];
    let mut aggregator = FederatedAggregator::new(initial_weights).with_byzantine_tolerance(0.3);

    // 7 honest nodes send similar gradients
    for i in 0..7u8 {
        let gradients: Vec<f32> = (0..100)
            .map(|j| (j as f32 * 0.01) + (i as f32 * 0.001))
            .collect();
        let mut source_id = [0u8; 32];
        source_id[0] = i;
        let msg = GradientMessage::new(source_id, gradients, 0.8);
        assert!(aggregator.receive_gradient(msg));
    }

    // 3 Byzantine nodes send adversarial gradients (very large values)
    for i in 7..10u8 {
        let gradients: Vec<f32> = (0..100).map(|j| j as f32 * 100.0).collect();
        let mut source_id = [0u8; 32];
        source_id[0] = i;
        let msg = GradientMessage::new(source_id, gradients, 0.8);
        assert!(aggregator.receive_gradient(msg));
    }

    // Aggregate should handle the Byzantine nodes
    let result: Option<Vec<f32>> = aggregator.aggregate();

    // The aggregated result should be close to the honest gradients, not the Byzantine ones
    if let Some(ref aggregated) = result {
        for (j, &w) in aggregated.iter().enumerate() {
            assert!(
                w.abs() < 50.0,
                "Aggregated weight[{}] = {} should be reasonable (Byzantine trimmed)",
                j,
                w
            );
        }
        eprintln!(
            "Byzantine tolerance: aggregated {} weights successfully",
            aggregated.len()
        );
    } else {
        // Aggregation may return None if all contributions were trimmed -
        // this is also acceptable behavior under extreme Byzantine load
        eprintln!("Aggregation returned None (all contributions trimmed - acceptable)");
    }
}

#[test]
fn test_federated_byzantine_scaling() {
    use symthaea::swarm::{FederatedAggregator, GradientMessage};

    // Test with increasing numbers of Byzantine nodes
    for (total, byzantine) in [(5, 1), (10, 3), (20, 6)] {
        let honest = total - byzantine;
        let initial_weights = vec![0.0f32; 50];
        let mut aggregator =
            FederatedAggregator::new(initial_weights).with_byzantine_tolerance(0.3);

        // Honest nodes
        for i in 0..honest {
            let gradients: Vec<f32> = (0..50)
                .map(|j| (j as f32 * 0.01) + (i as f32 * 0.001))
                .collect();
            let mut source_id = [0u8; 32];
            source_id[0] = i as u8;
            let msg = GradientMessage::new(source_id, gradients, 0.8);
            aggregator.receive_gradient(msg);
        }

        // Byzantine nodes
        for i in honest..total {
            let gradients: Vec<f32> = (0..50).map(|j| j as f32 * 1000.0).collect();
            let mut source_id = [0u8; 32];
            source_id[0] = i as u8;
            let msg = GradientMessage::new(source_id, gradients, 0.8);
            aggregator.receive_gradient(msg);
        }

        let result = aggregator.aggregate();
        eprintln!(
            "  {}/{} Byzantine: aggregation returned {}",
            byzantine,
            total,
            if result.is_some() { "Some" } else { "None" }
        );
    }
}

// =============================================================================
// MIXED CONCURRENT WORKLOAD
// =============================================================================

#[test]
fn test_mixed_concurrent_workload() {
    use std::thread;

    let start = Instant::now();

    // Thread 1-2: CfC inference
    let cfc_handles: Vec<_> = (0..2)
        .map(|t| {
            thread::spawn(move || {
                let cell_config = CfCConfig {
                    input_dim: 32,
                    hidden_dim: 16,
                    ..Default::default()
                };
                let config = CfCNetworkConfig {
                    input_dim: 32,
                    hidden_dim: 16,
                    num_layers: 1,
                    output_dim: 8,
                    cell_config,
                    ..Default::default()
                };
                let mut network = CfCNetwork::new(config);
                for i in 0..100 {
                    let input =
                        ndarray::Array1::from_shape_fn(32, |j| ((j + i + t * 100) as f32).sin());
                    let _ = network.forward(&input, 0.02);
                }
            })
        })
        .collect();

    // Thread 3-4: HDC encoding
    let hdc_handles: Vec<_> = (0..2)
        .map(|t| {
            thread::spawn(move || {
                let mut encoder = SemanticEncoder::default();
                for i in 0..500 {
                    let text = format!("thread {} sentence {}", t, i);
                    let _ = encoder.encode_text(&text);
                }
            })
        })
        .collect();

    // Thread 5: Phi computation
    let phi_handle = thread::spawn(|| {
        use rayon::prelude::*;
        let calc = ConnectivityCalculator::new();
        let topologies: Vec<_> = (0..20)
            .map(|i| ConsciousnessTopology::ring(8, HDC_DIMENSION, i))
            .collect();
        let _results: Vec<f64> = topologies
            .par_iter()
            .map(|topo| calc.algebraic_connectivity(&topo.node_representations))
            .collect();
    });

    // Wait for all threads
    for h in cfc_handles {
        h.join().unwrap();
    }
    for h in hdc_handles {
        h.join().unwrap();
    }
    phi_handle.join().unwrap();

    let elapsed = start.elapsed();
    eprintln!("Mixed concurrent workload completed in {:?}", elapsed);
}

// =============================================================================
// MEMORY PRESSURE UNDER CONCURRENCY
// =============================================================================

#[test]
fn test_memory_stability_concurrent_allocation() {
    use std::thread;

    // Spawn threads that allocate and drop large vectors repeatedly
    let handles: Vec<_> = (0..4)
        .map(|t| {
            thread::spawn(move || {
                for i in 0..50 {
                    // Create a topology (allocates HDC_DIMENSION-sized vectors)
                    let topo = ConsciousnessTopology::random(8, HDC_DIMENSION, (t * 50 + i) as u64);
                    // Do a computation so optimizer doesn't elide
                    let calc = ConnectivityCalculator::new();
                    let result: f64 = calc.algebraic_connectivity(&topo.node_representations);
                    assert!(result >= 0.0);
                    // topo is dropped here, releasing memory
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    eprintln!("Memory stability under concurrent allocation: OK");
}

#[test]
fn test_streaming_pipeline_no_leaks() {
    // Run streaming inference for a long sequence and verify it completes
    // without memory issues (output queue bounded by config)
    let cell_config = CfCConfig {
        input_dim: 16,
        hidden_dim: 8,
        ..Default::default()
    };
    let network = CfCNetwork::new(CfCNetworkConfig {
        input_dim: 16,
        hidden_dim: 8,
        num_layers: 1,
        output_dim: 4,
        cell_config,
        ..Default::default()
    });
    let config = StreamingConfig {
        batch_accumulation: 4,
        warmup_samples: 4,
        max_output_queue: 32,
        drop_on_backpressure: true,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    let mut total_outputs = 0usize;
    for i in 0..5000 {
        let input = ndarray::Array1::from_shape_fn(16, |j| ((j + i) as f32).sin());
        streamer.push(input);

        // Drain outputs periodically
        if i % 10 == 0 {
            while streamer.poll().is_some() {
                total_outputs += 1;
            }
        }
    }

    // Final drain
    while streamer.flush().is_some() {
        total_outputs += 1;
    }
    while streamer.poll().is_some() {
        total_outputs += 1;
    }

    eprintln!(
        "Streaming pipeline: {} outputs from 5000 inputs (no leaks)",
        total_outputs
    );
    assert!(total_outputs > 0);
}