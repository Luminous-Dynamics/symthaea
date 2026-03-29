// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
