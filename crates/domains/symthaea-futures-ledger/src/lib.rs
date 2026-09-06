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
//! [`v2`] adds time/provenance-neutral commitment and resolution primitives.
//! [`prospective`] builds the stricter prospective-evaluation protocol above
//! those primitives: evaluation policy is committed before outcome reveal, a
//! forecast attempt can explicitly abstain, and later resolutions can be
//! cross-validated against the exact attempt they claim to resolve.
//!
//! Neither schema by itself proves wall-clock precedence. A durable prospective
//! registry must additionally enforce unique immutable attempt IDs and, in a
//! later hardening, content-addressed/append-only commitment evidence. Never
//! treat possession of an ID alone as cryptographic proof that a forecast existed
//! before its outcome.

use serde::{Deserialize, Serialize};
use symthaea_futures_core::{ForecastDistribution, OutcomeRegion};

pub mod prospective;
pub mod v2;

/// Original seeded-simulation evidence schema. Retained unchanged for existing
/// artifacts and backtests; new prospective/external integrations should use the
/// `prospective` protocol above v2 rather than this post-hoc record.
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
