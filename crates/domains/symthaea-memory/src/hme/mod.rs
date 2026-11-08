// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holographic Memory Engine query algorithms and representation-aware policy.

pub mod group_envelope;
pub mod interval_select;

pub use group_envelope::{
    ContinuousHVGroupEnvelope, GroupEnvelopeError, member_id_set_digest,
};
pub use interval_select::{
    ExactScoreRecord, IntervalCoverageCounts, IntervalSelectionError, IntervalSelectionPlan,
    ScoreIntervalBuildError, ScoreIntervalRecord, select_refinement_set, stable_exact_top_k,
};
