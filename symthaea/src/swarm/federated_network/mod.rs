//! Federated Network Communication Layer
//!
//! This module provides network communication primitives for federated CfC learning
//! across the Symthaea swarm. It supports both local channel-based simulation for
//! testing and a real TCP backend for distributed network deployment.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      FEDERATED NETWORK LAYER                                 │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │  ┌───────────────────────┐         ┌───────────────────────┐                │
//! │  │   FederatedNode       │         │   FederatedCoordinator│                │
//! │  │                       │         │                       │                │
//! │  │ • node_id (32 bytes)  │         │ • Node registry       │                │
//! │  │ • address (socket)    │    ────▶│ • Message routing     │                │
//! │  │ • trust_score         │         │ • Sync coordination   │                │
//! │  │ • local aggregator    │         │ • Heartbeat manager   │                │
//! │  └───────────────────────┘         └───────────────────────┘                │
//! │                                                                              │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │                      NetworkBackend (Trait)                            │  │
//! │  ├───────────────────────────────────────────────────────────────────────┤  │
//! │  │                                                                        │  │
//! │  │  ┌─────────────────────────┐    ┌─────────────────────────┐           │  │
//! │  │  │   LocalChannelBackend   │    │      TcpBackend        │           │  │
//! │  │  │   (Testing/Simulation)  │    │   (Real TCP Network)   │           │  │
//! │  │  │                         │    │                         │           │  │
//! │  │  │ • Tokio MPSC channels   │    │ • Length-prefixed wire  │           │  │
//! │  │  │ • Zero latency          │    │ • Async TCP with tokio  │           │  │
//! │  │  │ • Perfect reliability   │    │ • Connection pooling    │           │  │
//! │  │  └─────────────────────────┘    └─────────────────────────┘           │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use symthaea::swarm::federated_network::{
//!     FederatedCoordinator, FederatedNetworkConfig, LocalChannelBackend,
//! };
//!
//! // Create coordinator with 3 nodes
//! let config = FederatedNetworkConfig::default()
//!     .with_num_nodes(3)
//!     .with_sync_interval_ms(1000);
//!
//! let coordinator = FederatedCoordinator::new_with_backend(
//!     config,
//!     LocalChannelBackend::new(),
//! ).await?;
//!
//! // Run federated learning round
//! coordinator.run_sync_round().await?;
//! ```

mod types;
mod local_backend;
mod tcp_backend;
mod coordinator;

#[cfg(test)]
mod tests;

pub use types::{
    FederatedNetworkConfig, FederatedMessage, FederatedNode, NodeAddress,
    NetworkBackend, NetworkResult, NetworkError,
};

pub use local_backend::LocalChannelBackend;

pub use tcp_backend::TcpBackend;

pub use coordinator::{
    FederatedCoordinator, CoordinatorStats, CoordinatorEvent,
    create_test_network,
};
