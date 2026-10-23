// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-ledger
//!
//! Replayable evidence records for the Symthaea Futures Laboratory.
//!
//! [`EvidenceRecord`] is the original v1 schema used by existing seeded
//! simulation backtests. It remains wire/API compatible.
//!
//! [`v2`] adds a time/provenance-neutral two-phase lifecycle:
//! [`v2::ForecastCommitment`] contains only information available before an
//! outcome, while [`v2::ForecastResolution`] records the later realization and
//! score. v2 supports both seeded simulations and hashed external observation
//! snapshots without sentinel seeds or fake tick semantics.
//!
//! The v2 *schema* does not by itself prove wall-clock precedence. A durable
//! prospective registry must additionally enforce unique immutable commitment IDs
//! (and, in a later hardening, content-addressed/append-only commitment evidence).
//! Never treat possession of a `ForecastCommitmentId` alone as cryptographic proof
//! that a forecast existed before its outcome.

use serde::{Deserialize, Serialize};
use symthaea_futures_core::{ForecastDistribution, OutcomeRegion};

pub mod v2;

/// Original seeded-simulation evidence schema. Retained unchanged for existing
/// artifacts and backtests; new prospective/external integrations should use v2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub scenario_family: String,
    pub world_seed: u64,
    pub observation_policy_version: String,
    pub observation_cutoff_tick: u64,
    pub belief_state_snapshot_hash: String,
    pub model_versions: Vec<String>,
    pub trajectory_generator_ids: Vec<String>,
    pub branch_clustering_method: String,
    pub predicted_distribution: ForecastDistribution,
    pub scoring_rule: String,
    pub actual_continuation: OutcomeRegion,
    pub score: f64,
    pub calibration_bucket: String,
    pub wall_clock_cost_ms: u64,
    pub notes: String,
}
