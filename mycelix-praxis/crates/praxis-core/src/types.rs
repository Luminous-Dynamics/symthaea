// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Common types used across the EduNet platform

use serde::{Deserialize, Serialize};

/// Unique identifier for a federated learning round
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoundId(pub String);

/// Unique identifier for a machine learning model
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

/// Unique identifier for a course
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CourseId(pub String);

/// Hash of model parameters or gradients
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHash(pub String);

/// Federated learning round state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundState {
    /// Discovering participants
    Discover,
    /// Accepting join requests
    Join,
    /// Assigning tasks to participants
    Assign,
    /// Collecting updates from participants
    Update,
    /// Aggregating updates
    Aggregate,
    /// Releasing new model
    Release,
    /// Round completed
    Completed,
    /// Round failed or cancelled
    Failed,
}

/// Privacy parameters for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyParams {
    /// Epsilon parameter for differential privacy
    pub epsilon: Option<f64>,
    /// Delta parameter for differential privacy
    pub delta: Option<f64>,
    /// L2 norm clipping threshold
    pub clip_norm: f32,
}

impl Default for PrivacyParams {
    fn default() -> Self {
        Self {
            epsilon: None,
            delta: None,
            clip_norm: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_id_creation() {
        let id = RoundId("round-123".to_string());
        assert_eq!(id.0, "round-123");
    }

    #[test]
    fn test_default_privacy_params() {
        let params = PrivacyParams::default();
        assert_eq!(params.clip_norm, 1.0);
        assert!(params.epsilon.is_none());
    }
}
