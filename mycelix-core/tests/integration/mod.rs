//! Mycelix Core Integration Tests
//!
//! End-to-end integration tests covering:
//! - MATL Trust System (PoGQ, TCDM, Entropy)
//! - K-Vector ZK Proofs
//! - Feldman DKG Ceremony
//! - RB-BFT Consensus
//! - FL Aggregation Pipeline
//! - Symthaea ↔ Mycelix Bridge (AI → Chain)
//!
//! Run with: cargo test --test integration

mod matl_bridge;
mod kvector_zkp;
mod feldman_dkg;
mod rb_bft_consensus;
mod fl_aggregation;
mod end_to_end;
mod symthaea_bridge;
