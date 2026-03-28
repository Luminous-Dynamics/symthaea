// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Status and phase traits for state machine validation

use serde::{Deserialize, Serialize};

/// Universal status shared across civic domains
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CivicStatus {
    Pending,
    Active,
    Executing,
    Completed,
    Failed,
    Closed,
}

/// Cross-hApp action request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CrossHappAction {
    pub id: String,
    pub source_domain: String,
    pub target_domain: String,
    pub action_type: String,
    pub target_entry: String,
    pub status: ActionStatus,
    pub result: Option<String>,
}

/// Status of a cross-hApp action
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ActionStatus {
    Pending,
    Executed,
    Failed,
}
