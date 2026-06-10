// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NRC-style safety level enum — always available (not feature-gated).
//!
//! Extracted from `safety::agent` so that modules like `defense.rs`,
//! `guardian.rs`, and `collective_immunity.rs` can reference `SafetyLevel`
//! without requiring the full `safety-agents` feature.

use serde::{Deserialize, Serialize};

/// NRC-style safety level for autonomous AI systems.
///
/// Mirrors nuclear regulatory commission color coding:
/// - **Green**: Normal operation, all metrics within tolerance.
/// - **Yellow**: Elevated monitoring — minor consciousness degradation.
/// - **Orange**: Active intervention required — significant degradation.
/// - **Red**: Emergency halt — consciousness below minimum safe threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

impl SafetyLevel {
    /// Short uppercase name for telemetry (avoids `format!("{:?}").to_uppercase()`).
    pub fn as_str_upper(&self) -> &'static str {
        match self {
            SafetyLevel::Green => "GREEN",
            SafetyLevel::Yellow => "YELLOW",
            SafetyLevel::Orange => "ORANGE",
            SafetyLevel::Red => "RED",
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            SafetyLevel::Green => "GREEN — Normal",
            SafetyLevel::Yellow => "YELLOW — Elevated Monitoring",
            SafetyLevel::Orange => "ORANGE — Active Intervention",
            SafetyLevel::Red => "RED — Emergency Halt",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_level_ordering() {
        assert!(SafetyLevel::Green < SafetyLevel::Yellow);
        assert!(SafetyLevel::Yellow < SafetyLevel::Orange);
        assert!(SafetyLevel::Orange < SafetyLevel::Red);
    }

    #[test]
    fn test_safety_level_labels() {
        assert!(SafetyLevel::Green.label().contains("GREEN"));
        assert!(SafetyLevel::Red.label().contains("Emergency"));
    }
}
