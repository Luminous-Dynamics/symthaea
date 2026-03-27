// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mycelix FL Coordinator -- standalone binary
//!
//! Usage: mycelix-fl-coordinator [OPTIONS]
//!
//! With the `grpc` feature enabled, this binary runs a gRPC server on
//! `[::1]:50051` (override with `MYCELIX_FL_ADDR`). Without `grpc`,
//! it runs a local demo of the coordinator lifecycle.

use mycelix_fl::coordinator::config::CoordinatorConfig;
#[cfg(not(feature = "grpc"))]
use mycelix_fl::coordinator::node_manager::NodeCredential;
use mycelix_fl::coordinator::orchestrator::FLCoordinator;
use mycelix_fl::defenses::fedavg::FedAvg;
use mycelix_fl::pogq::config::PoGQv41Config;
#[cfg(not(feature = "grpc"))]
use mycelix_fl::types::Gradient;

// ---------------------------------------------------------------------------
// gRPC mode (feature = "grpc")
// ---------------------------------------------------------------------------

#[cfg(feature = "grpc")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr_str =
        std::env::var("MYCELIX_FL_ADDR").unwrap_or_else(|_| "[::1]:50051".to_string());
    let addr: std::net::SocketAddr = addr_str.parse()?;

    println!("=====================================================");
    println!("  Mycelix Federated Learning Coordinator v0.1.0");
    println!("  Byzantine Fault Tolerance: 45% (PoGQ v4.1)");
    println!("  Transport: gRPC on {}", addr);
    println!("=====================================================");
    println!();

    let config = CoordinatorConfig::default();
    let pogq_config = PoGQv41Config::default();
    let coordinator = FLCoordinator::new(config.clone(), Box::new(FedAvg), Some(pogq_config));

    println!("Coordinator config:");
    println!("  Min nodes: {}", config.min_nodes);
    println!("  Max nodes: {}", config.max_nodes);
    println!("  Round timeout: {}s", config.round_timeout_secs);
    println!("  PoGQ: enabled");
    println!();

    mycelix_fl::grpc::server::serve(coordinator, addr).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Demo mode (no grpc feature)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "grpc"))]
fn main() {
    println!("=====================================================");
    println!("  Mycelix Federated Learning Coordinator v0.1.0");
    println!("  Byzantine Fault Tolerance: 45% (PoGQ v4.1)");
    println!("=====================================================");
    println!();

    // Parse config from CLI args or defaults
    let config = CoordinatorConfig::default();
    let pogq_config = PoGQv41Config::default();

    let mut coordinator =
        FLCoordinator::new(config.clone(), Box::new(FedAvg), Some(pogq_config));

    println!("Coordinator started:");
    println!("  Min nodes: {}", config.min_nodes);
    println!("  Max nodes: {}", config.max_nodes);
    println!("  Round timeout: {}s", config.round_timeout_secs);
    println!("  PoGQ: enabled");
    println!();
    println!("Ready to accept connections.");
    println!("(Build with --features grpc for network transport.)");
    println!();

    // Demo: register some nodes and run a round
    for i in 0..5 {
        let cred = NodeCredential {
            node_id: format!("node-{}", i),
            public_key: None,
            did: Some(format!("did:mycelix:node{}", i)),
            registered_at: 0,
        };
        coordinator.register_node(cred).expect("registration failed");
    }
    println!("Registered {} nodes", coordinator.node_count());

    // Start a round
    let round_num = coordinator.start_round().expect("start failed");
    println!("Started round {}", round_num);

    // Submit gradients (demo: 4 honest + 1 Byzantine)
    let dim = 100;
    for i in 0..4 {
        let values: Vec<f32> = (0..dim)
            .map(|d| (d as f32 * 0.01) + (i as f32 * 0.001))
            .collect();
        let grad = Gradient::new(format!("node-{}", i), values, round_num);
        coordinator.submit_gradient(grad).expect("submit failed");
    }
    // Byzantine node
    let byz_values: Vec<f32> = (0..dim).map(|d| -(d as f32 * 0.05)).collect();
    let byz_grad = Gradient::new("node-4".to_string(), byz_values, round_num);
    coordinator
        .submit_gradient(byz_grad)
        .expect("submit failed");

    // Complete round
    match coordinator.complete_round() {
        Ok(result) => {
            println!();
            println!("Round {} complete:", round_num);
            println!("  Included: {:?}", result.included_nodes);
            println!("  Excluded: {:?}", result.excluded_nodes);
            println!("  Result dimension: {}", result.gradient.len());
        }
        Err(e) => println!("Round failed: {}", e),
    }

    println!();
    println!("Round history:");
    for summary in coordinator.round_history() {
        println!(
            "  Round {}: {} ({} nodes)",
            summary.round_number, summary.status, summary.node_count
        );
    }
}
