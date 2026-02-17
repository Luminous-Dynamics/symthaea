//! Ethereum Integration Module
//!
//! Provides Ethereum client for:
//! - Payment distribution via PaymentRouter contract
//! - Proof anchoring on-chain
//! - Contract interactions
//! - Bridge between Holochain signals and Ethereum transactions
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    FL Aggregator                             │
//! │                          │                                   │
//! │    ┌─────────────────────▼─────────────────────┐            │
//! │    │         EthereumBridgeHandler             │            │
//! │    │  (Holochain signals → Ethereum txns)      │            │
//! │    └─────────────────────┬─────────────────────┘            │
//! │                          │                                   │
//! │              ┌───────────▼───────────┐                      │
//! │              │    EthereumClient     │                      │
//! │              │                       │                      │
//! │    ┌─────────┴─────────┬─────────────┴─────────┐           │
//! │    │                   │                       │           │
//! │    ▼                   ▼                       ▼           │
//! │ PaymentRouter    ReputationAnchor    ContributionRegistry  │
//! │   Contract          Contract              Contract          │
//! └─────────────────────────────────────────────────────────────┘
//!                          │
//!                          ▼
//!                  Ethereum Network
//! ```
//!
//! ## Bridge Integration
//!
//! The bridge handler connects Holochain zome signals to Ethereum:
//!
//! 1. Holochain Bridge zome emits `EthereumBridgeSignal`
//! 2. Native host listener receives signal
//! 3. `EthereumBridgeHandler` processes signal
//! 4. `EthereumClient` executes transaction
//! 5. Handler calls back into Holochain to update intent status

pub mod bridge;
pub mod client;
pub mod contracts;
pub mod mock;
pub mod types;

pub use bridge::*;
pub use client::*;
pub use contracts::*;
pub use mock::{MockEthereumClient, MockError, FailureConfig, RecordedPayment, RecordedAnchor};
pub use types::*;
