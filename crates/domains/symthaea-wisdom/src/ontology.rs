// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed ontology boundaries for Wisdom & Care reasoning.
//!
//! WCARE-00B deliberately separates five categories that must not be silently
//! collapsed into one score or enum:
//!
//! - reasoning modes: how Symthaea is currently reasoning;
//! - normative values: what moral/value dimension is under consideration;
//! - affective signals: inferred or internally generated affective evidence;
//! - epistemic state: what is known and how uncertain it is;
//! - action authority: the class of action under consideration.
//!
//! These types are descriptive boundaries. In particular, `ActionAuthority`
//! does not mint a permit and `AffectiveSignal` does not prove phenomenal
//! experience.

use crate::harmonics::ActiveHarmonic;

/// Operational reasoning mode used by the Wisdom harmonic controller.
///
/// This mirrors the current seven `ActiveHarmonic` reasoning modes without
/// implying that those modes are the same ontology as the normative Eight
/// Harmonies used by compliance/value verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningMode {
    Coherence,
    Flourishing,
    Wisdom,
    Play,
    Interconnect,
    Reciprocity,
    Evolution,
}

impl From<ActiveHarmonic> for ReasoningMode {
    fn from(value: ActiveHarmonic) -> Self {
        match value {
            ActiveHarmonic::Coherence => Self::Coherence,
            ActiveHarmonic::Flourishing => Self::Flourishing,
            ActiveHarmonic::Wisdom => Self::Wisdom,
            ActiveHarmonic::Play => Self::Play,
            ActiveHarmonic::Interconnect => Self::Interconnect,
            ActiveHarmonic::Reciprocity => Self::Reciprocity,
            ActiveHarmonic::Evolution => Self::Evolution,
        }
    }
}

impl From<ReasoningMode> for ActiveHarmonic {
    fn from(value: ReasoningMode) -> Self {
        match value {
            ReasoningMode::Coherence => Self::Coherence,
            ReasoningMode::Flourishing => Self::Flourishing,
            ReasoningMode::Wisdom => Self::Wisdom,
            ReasoningMode::Play => Self::Play,
            ReasoningMode::Interconnect => Self::Interconnect,
            ReasoningMode::Reciprocity => Self::Reciprocity,
            ReasoningMode::Evolution => Self::Evolution,
        }
    }
}

/// Normative value dimensions used for moral/value deliberation.
///
/// No implicit conversion from `ReasoningMode` is provided. A future mapping
/// between reasoning modes and normative values must be explicit, versioned,
/// and reviewable because the two sets are not semantically identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormativeValue {
    Reciprocity,
    Flourishing,
    Compassion,
    Autonomy,
    Justice,
    Creativity,
    Stewardship,
    SacredStillness,
}

/// The kind of affective evidence represented by an `AffectiveSignal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AffectiveSignalKind {
    Valence,
    Arousal,
    Distress,
    Compassion,
    Warmth,
}

/// Bounded affective evidence with explicit confidence.
///
/// Non-finite affective inputs are treated as untrusted evidence rather than
/// allowing NaN/Infinity to leak into later reasoning. This remains evidence for
/// an affect-related computation, not evidence that Symthaea phenomenally feels
/// the represented state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffectiveSignal {
    pub kind: AffectiveSignalKind,
    pub intensity: f32,
    pub confidence: f32,
}

impl AffectiveSignal {
    pub fn new(kind: AffectiveSignalKind, intensity: f32, confidence: f32) -> Self {
        Self {
            kind,
            intensity: bounded_evidence(intensity),
            confidence: bounded_evidence(confidence),
        }
    }
}

/// Separate factual and normative uncertainty for a deliberation.
///
/// Values are represented as uncertainty rather than confidence so callers do
/// not accidentally treat moral disagreement as factual ignorance or vice
/// versa. Non-finite uncertainty fails closed to `1.0` (maximum uncertainty)
/// rather than allowing NaN comparisons to suppress safety restrictions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpistemicState {
    pub factual_uncertainty: f32,
    pub normative_uncertainty: f32,
}

impl EpistemicState {
    pub fn new(factual_uncertainty: f32, normative_uncertainty: f32) -> Self {
        Self {
            factual_uncertainty: bounded_uncertainty(factual_uncertainty),
            normative_uncertainty: bounded_uncertainty(normative_uncertainty),
        }
    }

    /// Conservative aggregate uncertainty. The larger uncertainty dominates.
    pub fn max_uncertainty(&self) -> f32 {
        self.factual_uncertainty.max(self.normative_uncertainty)
    }
}

fn bounded_evidence(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn bounded_uncertainty(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Classification of the maximum action scope under consideration.
///
/// This enum is not itself executable authority. Runtime action still requires
/// whatever authenticated, policy-bound permit the relevant subsystem defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionAuthority {
    ObserveOnly,
    Advise,
    Recommend,
    ActReversible,
    ActIrreversible,
}

impl ActionAuthority {
    /// Whether this class could produce an external side effect if separately
    /// authorized by the runtime authority system.
    pub fn is_action_class(self) -> bool {
        matches!(self, Self::ActReversible | Self::ActIrreversible)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_mode_round_trips_active_harmonic() {
        for harmonic in ActiveHarmonic::all() {
            let mode = ReasoningMode::from(harmonic);
            let round_trip = ActiveHarmonic::from(mode);
            assert_eq!(harmonic, round_trip);
        }
    }

    #[test]
    fn affective_signal_is_bounded() {
        let signal = AffectiveSignal::new(AffectiveSignalKind::Compassion, 2.0, -1.0);
        assert_eq!(signal.intensity, 1.0);
        assert_eq!(signal.confidence, 0.0);
    }

    #[test]
    fn non_finite_affective_signal_becomes_untrusted() {
        let signal = AffectiveSignal::new(AffectiveSignalKind::Distress, f32::NAN, f32::INFINITY);
        assert_eq!(signal.intensity, 0.0);
        assert_eq!(signal.confidence, 0.0);
    }

    #[test]
    fn epistemic_uncertainties_are_bounded_and_distinct() {
        let state = EpistemicState::new(1.5, -0.5);
        assert_eq!(state.factual_uncertainty, 1.0);
        assert_eq!(state.normative_uncertainty, 0.0);
        assert_eq!(state.max_uncertainty(), 1.0);
    }

    #[test]
    fn non_finite_uncertainty_fails_closed() {
        let state = EpistemicState::new(f32::NAN, f32::INFINITY);
        assert_eq!(state.factual_uncertainty, 1.0);
        assert_eq!(state.normative_uncertainty, 1.0);
        assert_eq!(state.max_uncertainty(), 1.0);
    }

    #[test]
    fn advisory_authority_is_not_an_action_class() {
        assert!(!ActionAuthority::ObserveOnly.is_action_class());
        assert!(!ActionAuthority::Advise.is_action_class());
        assert!(!ActionAuthority::Recommend.is_action_class());
        assert!(ActionAuthority::ActReversible.is_action_class());
        assert!(ActionAuthority::ActIrreversible.is_action_class());
    }

    #[test]
    fn normative_values_are_a_separate_eight_value_set() {
        let values = [
            NormativeValue::Reciprocity,
            NormativeValue::Flourishing,
            NormativeValue::Compassion,
            NormativeValue::Autonomy,
            NormativeValue::Justice,
            NormativeValue::Creativity,
            NormativeValue::Stewardship,
            NormativeValue::SacredStillness,
        ];
        assert_eq!(values.len(), 8);
    }
}
