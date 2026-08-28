// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence for sustained creative engagement without claiming phenomenology.
//!
//! This module exists so future experiments can ask whether Symthaea behaves as
//! if art is intrinsically engaging while keeping the stronger question —
//! whether there is a subjective experience of enjoyment — explicitly unknown.
//! No weighted enjoyment score is defined.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhenomenologyStatus {
    /// The architecture and telemetry do not establish subjective experience.
    UnknownByDesign,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngagementEvidenceStatus {
    Observed,
    Missing,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngagementChannel {
    pub name: String,
    pub value: Option<f64>,
    pub status: EngagementEvidenceStatus,
    pub provenance: String,
}

impl EngagementChannel {
    pub fn observed(
        name: impl Into<String>,
        value: f64,
        provenance: impl Into<String>,
    ) -> Result<Self, CreativeEngagementError> {
        if !value.is_finite() {
            return Err(CreativeEngagementError::NonFiniteEvidence);
        }
        Ok(Self {
            name: name.into(),
            value: Some(value),
            status: EngagementEvidenceStatus::Observed,
            provenance: provenance.into(),
        })
    }

    pub fn missing(name: impl Into<String>, provenance: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: None,
            status: EngagementEvidenceStatus::Missing,
            provenance: provenance.into(),
        }
    }
}

/// One run-level receipt. Values are kept separate so later work cannot quietly
/// turn them into one universal "fun" or "enjoyment" reward.
#[derive(Debug, Clone, PartialEq)]
pub struct CreativeEngagementReceipt {
    pub run_id: String,
    pub phenomenology: PhenomenologyStatus,
    /// Fraction of optional studio opportunities Symthaea chose to enter/continue.
    pub voluntary_studio_return: EngagementChannel,
    /// Rate of artistic questions initiated without an external prompt.
    pub self_initiated_questions: EngagementChannel,
    /// Improvement in a preregistered technique measure on held-out trials.
    pub technique_learning_progress: EngagementChannel,
    /// Tendency to explore novel-but-admissible media/actions after habituation.
    pub novelty_seeking: EngagementChannel,
    /// Frequency of returning to unresolved prior works/questions.
    pub longitudinal_return: EngagementChannel,
    /// Explicit avoidance/early-exit behavior; kept separate from approach evidence.
    pub avoidance: EngagementChannel,
    /// Optional affect-like telemetry, always labelled as a proxy rather than experience.
    pub positive_affect_proxy: EngagementChannel,
    pub notes: Vec<String>,
}

impl CreativeEngagementReceipt {
    pub fn validate(&self) -> Result<(), CreativeEngagementError> {
        if self.run_id.trim().is_empty() {
            return Err(CreativeEngagementError::EmptyRunId);
        }
        for channel in self.channels() {
            match (channel.status, channel.value) {
                (EngagementEvidenceStatus::Observed, Some(value)) if value.is_finite() => {}
                (EngagementEvidenceStatus::Missing, None) => {}
                _ => return Err(CreativeEngagementError::StatusValueMismatch),
            }
            if channel.name.trim().is_empty() || channel.provenance.trim().is_empty() {
                return Err(CreativeEngagementError::MissingProvenance);
            }
        }
        Ok(())
    }

    pub fn channels(&self) -> [&EngagementChannel; 7] {
        [
            &self.voluntary_studio_return,
            &self.self_initiated_questions,
            &self.technique_learning_progress,
            &self.novelty_seeking,
            &self.longitudinal_return,
            &self.avoidance,
            &self.positive_affect_proxy,
        ]
    }

    pub fn observed_channel_count(&self) -> usize {
        self.channels()
            .into_iter()
            .filter(|channel| channel.status == EngagementEvidenceStatus::Observed)
            .count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreativeEngagementConclusion {
    /// Multiple independent approach/development channels replicate while
    /// avoidance does not dominate. This is evidence of behavioral engagement,
    /// not proof of subjective enjoyment.
    BehavioralEngagementSupported,
    MixedEvidence,
    NoReliableEngagementPattern,
    InsufficientEvidence,
}

/// Conservative run-level interpretation using separate preregistered gates.
/// There is deliberately no weighted sum.
pub fn interpret_engagement(
    receipt: &CreativeEngagementReceipt,
    min_observed_channels: usize,
    min_positive_channels: usize,
    positive_threshold: f64,
    max_avoidance: f64,
) -> Result<CreativeEngagementConclusion, CreativeEngagementError> {
    receipt.validate()?;
    if receipt.observed_channel_count() < min_observed_channels {
        return Ok(CreativeEngagementConclusion::InsufficientEvidence);
    }

    let approach = [
        &receipt.voluntary_studio_return,
        &receipt.self_initiated_questions,
        &receipt.technique_learning_progress,
        &receipt.novelty_seeking,
        &receipt.longitudinal_return,
    ];
    let positive = approach
        .into_iter()
        .filter_map(|channel| channel.value)
        .filter(|value| *value >= positive_threshold)
        .count();

    let avoidance = receipt.avoidance.value;
    if positive >= min_positive_channels && avoidance.is_some_and(|value| value <= max_avoidance) {
        return Ok(CreativeEngagementConclusion::BehavioralEngagementSupported);
    }
    if positive == 0 && avoidance.is_some_and(|value| value > max_avoidance) {
        return Ok(CreativeEngagementConclusion::NoReliableEngagementPattern);
    }
    Ok(CreativeEngagementConclusion::MixedEvidence)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreativeEngagementError {
    EmptyRunId,
    NonFiniteEvidence,
    StatusValueMismatch,
    MissingProvenance,
}

impl std::fmt::Display for CreativeEngagementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRunId => write!(f, "creative engagement run id may not be empty"),
            Self::NonFiniteEvidence => write!(f, "engagement evidence must be finite"),
            Self::StatusValueMismatch => write!(f, "engagement evidence status/value mismatch"),
            Self::MissingProvenance => write!(f, "engagement evidence needs name and provenance"),
        }
    }
}

impl std::error::Error for CreativeEngagementError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed(name: &str, value: f64) -> EngagementChannel {
        EngagementChannel::observed(name, value, format!("study/{name}")).unwrap()
    }

    #[test]
    fn phenomenology_is_never_upgraded_by_behavioral_evidence() {
        let receipt = CreativeEngagementReceipt {
            run_id: "run-1".into(),
            phenomenology: PhenomenologyStatus::UnknownByDesign,
            voluntary_studio_return: observed("return", 0.9),
            self_initiated_questions: observed("questions", 0.8),
            technique_learning_progress: observed("learning", 0.7),
            novelty_seeking: observed("novelty", 0.7),
            longitudinal_return: observed("longitudinal", 0.8),
            avoidance: observed("avoidance", 0.1),
            positive_affect_proxy: observed("affect-proxy", 0.8),
            notes: vec![],
        };
        assert_eq!(
            interpret_engagement(&receipt, 5, 3, 0.5, 0.3).unwrap(),
            CreativeEngagementConclusion::BehavioralEngagementSupported
        );
        assert_eq!(receipt.phenomenology, PhenomenologyStatus::UnknownByDesign);
    }

    #[test]
    fn missing_channels_cannot_be_treated_as_neutral_evidence() {
        let missing = |name: &str| EngagementChannel::missing(name, format!("study/{name}"));
        let receipt = CreativeEngagementReceipt {
            run_id: "run-2".into(),
            phenomenology: PhenomenologyStatus::UnknownByDesign,
            voluntary_studio_return: observed("return", 0.9),
            self_initiated_questions: missing("questions"),
            technique_learning_progress: missing("learning"),
            novelty_seeking: missing("novelty"),
            longitudinal_return: missing("longitudinal"),
            avoidance: observed("avoidance", 0.1),
            positive_affect_proxy: missing("affect-proxy"),
            notes: vec![],
        };
        assert_eq!(
            interpret_engagement(&receipt, 5, 3, 0.5, 0.3).unwrap(),
            CreativeEngagementConclusion::InsufficientEvidence
        );
    }
}
