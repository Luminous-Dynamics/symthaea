// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holochain knowledge graph bridge for DeSci.
//!
//! Prepares for P2P claim validation via mycelix-knowledge zomes.
//! Currently operates in mock mode — when the Holochain conductor
//! is running, it will connect via WebSocket and dispatch zome calls
//! for claim submission, fact-checking, and truth assessment.
//!
//! Zomes available in mycelix-knowledge:
//! - claims: submit_claim, get_claim, verify_claim
//! - factcheck: fact_check (returns True/MostlyTrue/Mixed/MostlyFalse/False)
//! - dkg: submit_claim (S-P-O triples), attest_claim, get_truth
//! - inference: create_inference, assess_credibility
//! - invention: register_invention, prior_art_search

use serde::{Deserialize, Serialize};

/// Connection status to the Holochain conductor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HolochainStatus {
    /// No conductor available — using REST API fallback
    Offline,
    /// Connected to conductor via WebSocket
    Connected,
    /// Attempting connection
    Connecting,
}

impl HolochainStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Offline => "REST API (centralized)",
            Self::Connected => "Holochain DHT (P2P)",
            Self::Connecting => "Connecting to conductor...",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Offline => "status-offline",
            Self::Connected => "status-connected",
            Self::Connecting => "status-connecting",
        }
    }
}

/// Result from a fact-check against the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCheckResult {
    pub verdict: FactCheckVerdict,
    pub confidence: f64,
    pub supporting_claims: usize,
    pub contradicting_claims: usize,
}

/// Fact-check verdict levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactCheckVerdict {
    True,
    MostlyTrue,
    Mixed,
    MostlyFalse,
    False,
    Insufficient,
}

impl FactCheckVerdict {
    pub fn label(&self) -> &'static str {
        match self {
            Self::True => "True",
            Self::MostlyTrue => "Mostly True",
            Self::Mixed => "Mixed",
            Self::MostlyFalse => "Mostly False",
            Self::False => "False",
            Self::Insufficient => "Insufficient Evidence",
        }
    }

    pub fn css_color(&self) -> &'static str {
        match self {
            Self::True => "var(--accent-emerald)",
            Self::MostlyTrue => "var(--tier-e3)",
            Self::Mixed => "var(--tier-e1)",
            Self::MostlyFalse => "var(--tier-e0)",
            Self::False => "#ef4444",
            Self::Insufficient => "var(--text-secondary)",
        }
    }
}

/// Mock fact-check (used when conductor is offline).
///
/// In production, this would call:
/// ```text
/// zome_call("mycelix-knowledge", "factcheck", "fact_check", {
///     statement: description,
///     min_e: 2,
///     min_n: 1,
/// })
/// ```
pub fn mock_fact_check(description: &str) -> FactCheckResult {
    // Simple heuristic based on known keywords
    let desc_lower = description.to_lowercase();

    if desc_lower.contains("gravity-a") || desc_lower.contains("lazar") || desc_lower.contains("element 115") {
        FactCheckResult {
            verdict: FactCheckVerdict::Insufficient,
            confidence: 0.1,
            supporting_claims: 0,
            contradicting_claims: 3,
        }
    } else if desc_lower.contains("waveguide") && desc_lower.contains("bismuth") {
        FactCheckResult {
            verdict: FactCheckVerdict::MostlyFalse,
            confidence: 0.8,
            supporting_claims: 1,
            contradicting_claims: 2,
        }
    } else if desc_lower.contains("island") && desc_lower.contains("stability") {
        FactCheckResult {
            verdict: FactCheckVerdict::MostlyTrue,
            confidence: 0.6,
            supporting_claims: 5,
            contradicting_claims: 0,
        }
    } else {
        FactCheckResult {
            verdict: FactCheckVerdict::Insufficient,
            confidence: 0.0,
            supporting_claims: 0,
            contradicting_claims: 0,
        }
    }
}

/// Check if the Holochain conductor is reachable.
///
/// In production, this pings ws://localhost:8888 or the configured conductor URL.
pub fn check_conductor_status() -> HolochainStatus {
    // TODO: When mycelix-leptos-client is added as dependency,
    // attempt BrowserWsTransport::connect() and return status.
    HolochainStatus::Offline
}
