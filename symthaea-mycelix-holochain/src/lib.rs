// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Holochain conductor bridge for Symthaea.
//!
//! This crate provides async connectivity to a running Holochain conductor
//! via WebSocket, enabling Symthaea's cognitive loop to dispatch real zome
//! calls to the Mycelix governance and finance clusters.
//!
//! # Isolation
//!
//! This crate is intentionally **not** part of the main symthaea workspace
//! to avoid serde version conflicts between Holochain's pinned serde and
//! symthaea's broader dependency graph. It maintains its own `Cargo.lock`.
//!
//! # Usage
//!
//! ```rust,no_run
//! use symthaea_mycelix_holochain::{HolochainConductor, GovernanceDispatcher, FinanceDispatcher};
//! use std::sync::Arc;
//!
//! # async fn example() -> symthaea_mycelix_holochain::Result<()> {
//! let conductor = Arc::new(HolochainConductor::new(
//!     "ws://localhost:8888",
//!     "mycelix-unified",
//! ));
//! conductor.connect().await?;
//!
//! let gov = GovernanceDispatcher::new(conductor.clone());
//! let proposals = gov.get_active_proposals().await?;
//!
//! let fin = FinanceDispatcher::new(conductor.clone());
//! let balance = fin.query_tend_balance("did:key:z6Mk...").await?;
//! # Ok(())
//! # }
//! ```

pub mod conductor;
pub mod dispatch_loop;
pub mod dispatcher;
pub mod error;
pub mod types;

// Re-exports for ergonomic usage.
pub use conductor::{HolochainConductor, MultiConductor};
pub use dispatch_loop::{run_dispatch_loop, GovernanceDispatchCommand};
pub use dispatcher::{FinanceDispatcher, GovernanceDispatcher};
pub use error::{BridgeError, Result};
pub use types::{BalanceResponse, FeeTierResponse, TendBalanceResponse};

// Re-export holochain_client types used in integration tests.
pub use holochain_client::{AdminWebsocket, IssueAppAuthenticationTokenPayload};
