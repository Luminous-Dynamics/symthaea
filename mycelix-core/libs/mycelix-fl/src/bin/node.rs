// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mycelix FL Node -- connects to coordinator and participates in federated learning
//!
//! Usage:
//!   mycelix-fl-node
//!
//! Environment variables:
//!   COORDINATOR_URL  -- coordinator gRPC address (default: http://[::1]:50051)
//!   NODE_ID          -- unique node identifier (default: node-default)
//!   NODE_DID         -- optional DID for this node

use mycelix_fl::grpc::client::FlClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coordinator_addr = std::env::var("COORDINATOR_URL")
        .unwrap_or_else(|_| "http://[::1]:50051".to_string());
    let node_id =
        std::env::var("NODE_ID").unwrap_or_else(|_| "node-default".to_string());
    let did = std::env::var("NODE_DID").ok();

    println!("=====================================================");
    println!("  Mycelix FL Node v0.1.0");
    println!("=====================================================");
    println!("Node ID:      {}", node_id);
    println!("Coordinator:  {}", coordinator_addr);
    if let Some(ref d) = did {
        println!("DID:          {}", d);
    }
    println!();

    // Connect
    println!("Connecting to coordinator...");
    let client = FlClient::connect(&coordinator_addr, node_id.clone()).await?;

    // Health check
    match client.health_check().await {
        Ok(health) => println!(
            "Coordinator healthy: {} nodes registered, round {}",
            health.registered_nodes, health.current_round
        ),
        Err(e) => {
            eprintln!("Coordinator unreachable: {}", e);
            return Err(e.into());
        }
    }

    // Register
    println!("Registering node...");
    let reg = client.register(did).await?;
    if reg.success {
        println!(
            "Registered successfully. Current round: {}",
            reg.current_round
        );
    } else {
        eprintln!("Registration failed: {}", reg.message);
        return Ok(());
    }

    // Participate in rounds
    println!();
    println!("Participating in rounds...");
    let gradient_dim = 100;

    for round in 0..5u64 {
        // Simulate local training (in production: actual model training)
        println!("\nRound {}: Computing local gradient...", round);
        let gradient: Vec<f32> = (0..gradient_dim)
            .map(|i| (i as f32) * 0.01 + (round as f32) * 0.001)
            .collect();

        // Submit
        match client.submit_gradient(gradient, round).await {
            Ok(resp) => {
                if resp.accepted {
                    println!("  Gradient accepted");
                } else {
                    println!("  Gradient rejected: {}", resp.message);
                }
            }
            Err(e) => println!("  Submit error: {}", e),
        }

        // Check result
        match client.round_result(round).await {
            Ok(result) => {
                if result.success {
                    println!(
                        "  Round result: {} nodes included, {} excluded",
                        result.included_nodes.len(),
                        result.excluded_nodes.len()
                    );
                } else {
                    println!("  Round not yet aggregated");
                }
            }
            Err(_) => println!("  Round not yet complete"),
        }
    }

    println!("\nNode shutting down.");
    Ok(())
}
