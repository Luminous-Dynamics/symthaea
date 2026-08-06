// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-ledger
//!
//! Replayable evidence records for the Symthaea Futures Laboratory
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`). Every scored forecast writes one
//! [`EvidenceRecord`] — the mechanism that makes a later "was this calibrated?" question
//! answerable without rerunning the experiment, and that lets a later loosening of an
//! observation policy be detected against old scores instead of silently invalidating them.
//!
//! Field set mirrors the plan's "Evidence ledger schema" section exactly — keep them in sync.

use serde::{Deserialize, Serialize};
use symthaea_futures_core::{ForecastDistribution, OutcomeRegion};

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
