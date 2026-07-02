// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integrity zome for the cross-cluster reputation aggregator.

use hdi::prelude::*;

/// Aggregated reputation entry combining scores from multiple clusters.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AggregatedReputation {
    /// Agent whose reputation is aggregated
    pub agent_pubkey_b64: String,
    /// Score from identity/web-of-trust (0.0-1.0)
    pub web_of_trust_score: f64,
    /// MYCEL score from finance/recognition (0.0-1.0)
    pub mycel_score: f64,
    /// Per-cluster domain scores: Vec of (cluster_name, score)
    pub domain_scores: Vec<(String, f64)>,
    /// Weighted composite score (0.0-1.0)
    pub composite: f64,
    /// When this aggregation was computed
    pub computed_at: Timestamp,
    /// True if any source score is >24h stale
    pub staleness_warning: bool,
}

/// Domain score report from another cluster.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DomainScoreReport {
    /// Agent whose score is being reported
    pub agent_pubkey_b64: String,
    /// Cluster that is reporting (e.g., "commons", "civic")
    pub cluster: String,
    /// The domain-specific score (0.0-1.0)
    pub score: f64,
    /// When this score was computed at the source
    pub source_timestamp: Timestamp,
    /// Who reported this score
    pub reporter_pubkey_b64: String,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    AggregatedReputation(AggregatedReputation),
    DomainScoreReport(DomainScoreReport),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToReputation,
    AgentToDomainScore,
    AllReputations,
    ClusterToDomainScore,
}

#[hdk_extern]
pub fn validate(_op: Op) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
