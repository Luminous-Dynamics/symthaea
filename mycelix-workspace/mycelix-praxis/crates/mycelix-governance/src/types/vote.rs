// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reputation-weighted voting types

use serde::{Deserialize, Serialize};

/// Reputation-weighted vote with composite trust scoring
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReputationVote {
    pub proposal_id: String,
    pub voter: String, // AgentPubKey as string
    pub choice: VoteChoice,
    pub justification: Option<String>,

    // Reputation weighting
    pub reputation_weight: f64, // Composite PoGQ + TCDM + Entropy + Stake

    pub timestamp: i64,
}

/// Vote choice enumeration
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}
