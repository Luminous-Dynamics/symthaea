//! Aggregation logic for reputation-weighted voting
//!
//! This module contains:
//! - DynamicQuorumCalculator - adaptive participation requirements ✅
//!
//! Future implementations:
//! - CompositeReputationCalculator - combines PoGQ, TCDM, Entropy, Stake
//! - VoteTallyAggregator - weighted vote counting

pub mod dynamic_quorum;

pub use dynamic_quorum::{DynamicQuorumCalculator, QuorumStatus};
