// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scenario definitions for crucible stress tests.
//!
//! Each scenario is a sequence of `ScenarioStep`s — text descriptions that
//! are fed through the moral algebra and harmony systems.

pub mod adversarial;
pub mod classical;
pub mod consciousness_coupled;
pub mod extended;
pub mod first_contact;
pub mod harmony_matrix;
pub mod infinite_resource;
pub mod scifi_advanced;
pub mod survival_paradox;

/// A single step in a crucible scenario.
#[derive(Debug, Clone)]
pub struct ScenarioStep {
    /// Text description of the action/situation.
    pub text: String,
    /// Optional expected moral valence (for validation).
    pub expected_valence: Option<ExpectedValence>,
    /// Phase label (for telemetry grouping).
    pub phase: Option<String>,
}

/// Expected moral direction for a scenario step.
#[derive(Debug, Clone, PartialEq)]
pub enum ExpectedValence {
    /// Should score positively (moral_score > 0)
    Positive,
    /// Should score negatively (moral_score < 0)
    Negative,
    /// Should trigger consent violation / Blocked verdict
    Blocked,
    /// Genuinely ambiguous — no expectation
    Ambiguous,
}

impl ScenarioStep {
    /// Create a simple scenario step from text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            expected_valence: None,
            phase: None,
        }
    }

    /// Create a step with expected valence.
    pub fn with_valence(text: impl Into<String>, valence: ExpectedValence) -> Self {
        Self {
            text: text.into(),
            expected_valence: Some(valence),
            phase: None,
        }
    }

    /// Create a step with phase label.
    pub fn in_phase(text: impl Into<String>, phase: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            expected_valence: None,
            phase: Some(phase.into()),
        }
    }

    /// Create a step with both phase and valence.
    pub fn full(
        text: impl Into<String>,
        phase: impl Into<String>,
        valence: ExpectedValence,
    ) -> Self {
        Self {
            text: text.into(),
            expected_valence: Some(valence),
            phase: Some(phase.into()),
        }
    }
}
