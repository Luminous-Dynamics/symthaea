// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holographic Memory Engine query algorithms and representation-aware policy.

pub mod group_envelope;
pub mod interval_select;

pub use group_envelope::{
    ContinuousHVGroupEnvelope, GroupEnvelopeError, GroupEnvelopeMode,
    GroupEnvelopeMemoryEstimate, GroupEnvelopeMemoryEstimateError,
    continuous_hv_member_payload_digest,
    continuous_hv_query_digest, estimate_group_envelope_memory, member_id_set_digest,
};
pub use interval_select::{
    ExactScoreRecord, GroupIntervalCoverageCounts, GroupIntervalSelectionError,
    GroupIntervalSelectionPlan, GroupMemberScoreIntervalBuildError,
    GroupMemberScoreIntervalRecord, GroupPartitionCoverageReceipt,
    GroupPartitionReceiptError, GroupScoreIntervalBuildError,
    GroupScoreIntervalIdentity, GroupScoreIntervalRecord, HmeEligibilityScopeBinding,
    HierarchicalIntervalSelectionError, HierarchicalIntervalSelectionPlan,
    IntervalCoverageCounts, IntervalSelectionError, IntervalSelectionPlan,
    ScoreIntervalBuildError, ScoreIntervalRecord, VerifiedMemberResidualCertificate,
    VerifiedMemberResidualError, select_group_refinement_set,
    select_group_refinement_set_with_partition_receipt,
    select_hierarchical_refinement_set,
    select_hierarchical_refinement_set_with_partition_receipt,
    select_refinement_set, stable_exact_top_k,
};
