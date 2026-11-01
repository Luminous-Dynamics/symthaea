// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, additive performance-gesture sidecar for instrument-aware
//! realization.
//!
//! A gesture event is not a renderer capability claim. This module validates
//! event identity, exact score-relative time, parameter/value units, domains,
//! ordering and scope. Renderer adapters and signed exact-cell capability
//! decisions remain separate authorities.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

/// Current wire-schema version for performance gesture tracks.
pub const PERFORMANCE_GESTURE_SCHEMA_VERSION: u16 = 1;

/// Exact, non-negative score-relative position measured in quarter-note beats.
/// The denominator must be positive; constructors reduce the fraction. The
/// fields remain private so ordinary callers cannot construct a non-canonical
/// value, while validation still protects deserialized data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BeatPosition {
    numerator: u64,
    denominator: u32,
}

impl BeatPosition {
    pub fn new(numerator: u64, denominator: u32) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        let divisor = gcd_u64(numerator, denominator as u64);
        Some(Self {
            numerator: numerator / divisor,
            denominator: (denominator as u64 / divisor) as u32,
        })
    }

    pub fn numerator(self) -> u64 {
        self.numerator
    }

    pub fn denominator(self) -> u32 {
        self.denominator
    }

    fn is_canonical(self) -> bool {
        self.denominator > 0
            && gcd_u64(self.numerator, self.denominator as u64) == 1
    }

    fn cmp_exact(self, other: Self) -> Ordering {
        ((self.numerator as u128) * (other.denominator as u128))
            .cmp(&((other.numerator as u128) * (self.denominator as u128)))
    }
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

/// The target is a stable score part or a stable note within a part.
/// Part identity is not inferred from pitch, register, role, channel or overlap.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum GestureTarget {
    Part { part_id: u16 },
    Note { part_id: u16, note_id: String },
}

/// Instrument-specific controls. These are semantic parameter identities,
/// not claims that any current renderer implements them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GestureParameter {
    BowSpeed,
    BowForce,
    BowContactPoint,
    BreathPressure,
    TonguingMode,
    PluckForce,
    PluckPosition,
    DampingAmount,
    MalletProfile,
    StrikePosition,
    DamperPedal,
    VibraphoneMotorEnabled,
    TromboneSlidePosition,
    BellowsPressure,
    BellowsDirection,
    RegisterStop,
    VowelIdentity,
    FormantShift,
    OscillatorFrequency,
    FilterCutoff,
    EnvelopeLevel,
    PitchBend,
    VibratoRate,
    VibratoDepth,
}

/// Units are explicit. Normalized values are only meaningful alongside the
/// parameter's declared reference frame in the adapter/profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GestureUnit {
    MetersPerSecond,
    Newtons,
    Pascals,
    Normalized,
    Millimeters,
    Hertz,
    Cents,
}

/// Values are typed so categorical/binary controls cannot accidentally be
/// linearly interpolated or supplied as unlabelled floating-point numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GestureValue {
    Scalar { amount: f64, unit: GestureUnit },
    Boolean { value: bool },
    Choice { value: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GestureInterpolation {
    Step,
    Linear,
}

/// One immutable event in a performance-control stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceGestureEvent {
    /// Stable event identity, unique within one track.
    pub event_id: String,
    /// Stable ontology identity, not a GM program or renderer patch name.
    pub instrument_id: String,
    /// Optional source model/setup identity. None means the model is unresolved.
    pub source_model_id: Option<String>,
    pub target: GestureTarget,
    pub at: BeatPosition,
    pub parameter: GestureParameter,
    pub value: GestureValue,
    pub interpolation: GestureInterpolation,
}

/// Additive sidecar. Existing Score and ScoreNote serialization are unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceGestureTrackV1 {
    pub schema_version: u16,
    pub events: Vec<PerformanceGestureEvent>,
}

impl Default for PerformanceGestureTrackV1 {
    fn default() -> Self {
        Self {
            schema_version: PERFORMANCE_GESTURE_SCHEMA_VERSION,
            events: Vec::new(),
        }
    }
}

impl PerformanceGestureTrackV1 {
    pub fn new(events: Vec<PerformanceGestureEvent>) -> Self {
        Self {
            schema_version: PERFORMANCE_GESTURE_SCHEMA_VERSION,
            events,
        }
    }

    /// Validate the wire contract without asserting that a technique or
    /// renderer capability is supported.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut issues = Vec::new();
        if self.schema_version != PERFORMANCE_GESTURE_SCHEMA_VERSION {
            issues.push(format!(
                "unsupported gesture schema version: {}",
                self.schema_version
            ));
        }

        let mut event_ids = HashSet::new();
        let mut cell_keys = HashSet::new();
        let mut previous: Option<&PerformanceGestureEvent> = None;

        for (index, event) in self.events.iter().enumerate() {
            let prefix = format!("event[{index}]");

            if event.event_id.trim().is_empty() {
                issues.push(format!("{prefix}: event_id is empty"));
            } else if !event_ids.insert(event.event_id.as_str()) {
                issues.push(format!("{prefix}: duplicate event_id {}", event.event_id));
            }
            if event.instrument_id.trim().is_empty() {
                issues.push(format!("{prefix}: instrument_id is empty"));
            }
            if event
                .source_model_id
                .as_ref()
                .is_some_and(|model| model.trim().is_empty())
            {
                issues.push(format!("{prefix}: source_model_id is empty when present"));
            }

            match &event.target {
                GestureTarget::Part { part_id } if *part_id == u16::MAX => {
                    issues.push(format!("{prefix}: unassigned part identity is not valid"));
                }
                GestureTarget::Note { part_id, note_id } => {
                    if *part_id == u16::MAX {
                        issues.push(format!("{prefix}: note target has unassigned part identity"));
                    }
                    if note_id.trim().is_empty() {
                        issues.push(format!("{prefix}: note target has an empty note_id"));
                    }
                }
                GestureTarget::Part { .. } => {}
            }

            if !event.at.is_canonical() {
                issues.push(format!(
                    "{prefix}: beat position denominator must be positive and fraction reduced"
                ));
            }

            if !value_matches_parameter(event.parameter, &event.value) {
                issues.push(format!(
                    "{prefix}: value kind/unit does not match parameter {:?}",
                    event.parameter
                ));
            } else if let GestureValue::Scalar { amount, unit } = &event.value {
                if !amount.is_finite() {
                    issues.push(format!("{prefix}: scalar value is not finite"));
                } else if !scalar_domain_is_valid(event.parameter, *amount, *unit) {
                    issues.push(format!(
                        "{prefix}: scalar value is outside the declared domain for {:?}",
                        event.parameter
                    ));
                }
            }

            if event.interpolation == GestureInterpolation::Linear
                && !matches!(&event.value, GestureValue::Scalar { .. })
            {
                issues.push(format!(
                    "{prefix}: categorical and boolean gestures require step interpolation"
                ));
            }

            let key = (
                event.instrument_id.as_str(),
                &event.target,
                event.at,
                event.parameter,
            );
            if !cell_keys.insert(key) {
                issues.push(format!(
                    "{prefix}: duplicate parameter event for the same target and beat position"
                ));
            }

            if let Some(prior) = previous {
                match prior.at.cmp_exact(event.at) {
                    Ordering::Greater => {
                        issues.push(format!("{prefix}: events are not in chronological order"));
                    }
                    Ordering::Equal if prior.event_id.as_str() > event.event_id.as_str() => {
                        issues.push(format!(
                            "{prefix}: simultaneous events must be ordered by event_id"
                        ));
                    }
                    _ => {}
                }
            }
            previous = Some(event);
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }
}

fn value_matches_parameter(parameter: GestureParameter, value: &GestureValue) -> bool {
    match (parameter, value) {
        (
            GestureParameter::BowSpeed
            | GestureParameter::TromboneSlidePosition
            | GestureParameter::OscillatorFrequency
            | GestureParameter::FilterCutoff
            | GestureParameter::VibratoRate,
            GestureValue::Scalar { unit, .. },
        ) => {
            *unit
                == match parameter {
                    GestureParameter::BowSpeed => GestureUnit::MetersPerSecond,
                    GestureParameter::TromboneSlidePosition => GestureUnit::Millimeters,
                    GestureParameter::OscillatorFrequency
                    | GestureParameter::FilterCutoff
                    | GestureParameter::VibratoRate => GestureUnit::Hertz,
                    _ => unreachable!(),
                }
        }
        (
            GestureParameter::BowForce
            | GestureParameter::PluckForce,
            GestureValue::Scalar { unit, .. },
        ) => *unit == GestureUnit::Newtons,
        (
            GestureParameter::BreathPressure
            | GestureParameter::BellowsPressure,
            GestureValue::Scalar { unit, .. },
        ) => *unit == GestureUnit::Pascals,
        (
            GestureParameter::BowContactPoint
            | GestureParameter::PluckPosition
            | GestureParameter::DampingAmount
            | GestureParameter::StrikePosition
            | GestureParameter::EnvelopeLevel,
            GestureValue::Scalar { unit, .. },
        ) => *unit == GestureUnit::Normalized,
        (
            GestureParameter::FormantShift
            | GestureParameter::PitchBend
            | GestureParameter::VibratoDepth,
            GestureValue::Scalar { unit, .. },
        ) => *unit == GestureUnit::Cents,
        (
            GestureParameter::TonguingMode
            | GestureParameter::MalletProfile
            | GestureParameter::BellowsDirection
            | GestureParameter::RegisterStop
            | GestureParameter::VowelIdentity,
            GestureValue::Choice { .. },
        ) => true,
        (
            GestureParameter::DamperPedal | GestureParameter::VibraphoneMotorEnabled,
            GestureValue::Boolean { .. },
        ) => true,
        _ => false,
    }
}

fn scalar_domain_is_valid(
    parameter: GestureParameter,
    amount: f64,
    unit: GestureUnit,
) -> bool {
    if !amount.is_finite() {
        return false;
    }
    match (parameter, unit) {
        (
            GestureParameter::BowSpeed
            | GestureParameter::BowForce
            | GestureParameter::BreathPressure
            | GestureParameter::PluckForce
            | GestureParameter::TromboneSlidePosition
            | GestureParameter::BellowsPressure,
            _,
        ) => amount >= 0.0,
        (
            GestureParameter::BowContactPoint
            | GestureParameter::PluckPosition
            | GestureParameter::DampingAmount
            | GestureParameter::StrikePosition
            | GestureParameter::EnvelopeLevel,
            GestureUnit::Normalized,
        ) => (0.0..=1.0).contains(&amount),
        (
            GestureParameter::OscillatorFrequency
            | GestureParameter::FilterCutoff
            | GestureParameter::VibratoRate,
            GestureUnit::Hertz,
        ) => amount > 0.0,
        (GestureParameter::FormantShift | GestureParameter::PitchBend, GestureUnit::Cents) => true,
        (GestureParameter::VibratoDepth, GestureUnit::Cents) => amount >= 0.0,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(
        event_id: &str,
        at: BeatPosition,
        parameter: GestureParameter,
        value: GestureValue,
        interpolation: GestureInterpolation,
    ) -> PerformanceGestureEvent {
        PerformanceGestureEvent {
            event_id: event_id.to_string(),
            instrument_id: "violin".to_string(),
            source_model_id: Some("test.violin.model-a".to_string()),
            target: GestureTarget::Note {
                part_id: 2,
                note_id: "note-17".to_string(),
            },
            at,
            parameter,
            value,
            interpolation,
        }
    }

    #[test]
    fn beat_positions_are_exact_and_reduced() {
        let half = BeatPosition::new(2, 4).unwrap();
        assert_eq!(half.numerator(), 1);
        assert_eq!(half.denominator(), 2);
        assert!(BeatPosition::new(1, 0).is_none());
        assert_eq!(
            BeatPosition::new(1, 2).unwrap().cmp_exact(BeatPosition::new(2, 4).unwrap()),
            Ordering::Equal
        );
    }

    #[test]
    fn validates_instrument_specific_typed_gestures() {
        let track = PerformanceGestureTrackV1::new(vec![
            event(
                "bow-force",
                BeatPosition::new(0, 1).unwrap(),
                GestureParameter::BowForce,
                GestureValue::Scalar {
                    amount: 2.5,
                    unit: GestureUnit::Newtons,
                },
                GestureInterpolation::Linear,
            ),
            event(
                "bow-speed",
                BeatPosition::new(1, 4).unwrap(),
                GestureParameter::BowSpeed,
                GestureValue::Scalar {
                    amount: 0.8,
                    unit: GestureUnit::MetersPerSecond,
                },
                GestureInterpolation::Linear,
            ),
        ]);
        assert!(track.validate().is_ok());
    }

    #[test]
    fn rejects_unit_mismatch_non_finite_and_out_of_domain_values() {
        let bad_unit = event(
            "bad-unit",
            BeatPosition::new(0, 1).unwrap(),
            GestureParameter::BowSpeed,
            GestureValue::Scalar {
                amount: 0.5,
                unit: GestureUnit::Pascals,
            },
            GestureInterpolation::Linear,
        );
        let bad_domain = event(
            "bad-domain",
            BeatPosition::new(1, 1).unwrap(),
            GestureParameter::BowContactPoint,
            GestureValue::Scalar {
                amount: 1.2,
                unit: GestureUnit::Normalized,
            },
            GestureInterpolation::Linear,
        );
        let bad_non_finite = event(
            "bad-nan",
            BeatPosition::new(2, 1).unwrap(),
            GestureParameter::PitchBend,
            GestureValue::Scalar {
                amount: f64::NAN,
                unit: GestureUnit::Cents,
            },
            GestureInterpolation::Linear,
        );
        let track = PerformanceGestureTrackV1::new(vec![bad_unit, bad_domain, bad_non_finite]);
        let issues = track.validate().unwrap_err();
        assert!(issues.iter().any(|issue| issue.contains("value kind/unit")));
        assert!(issues.iter().any(|issue| issue.contains("outside the declared domain")));
        assert!(issues.iter().any(|issue| issue.contains("not finite")));
    }

    #[test]
    fn categorical_and_boolean_controls_are_step_only() {
        let track = PerformanceGestureTrackV1::new(vec![event(
            "vowel",
            BeatPosition::new(0, 1).unwrap(),
            GestureParameter::VowelIdentity,
            GestureValue::Choice {
                value: "aah".to_string(),
            },
            GestureInterpolation::Linear,
        )]);
        assert!(track
            .validate()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("require step interpolation")));
    }

    #[test]
    fn note_scoped_gestures_require_stable_part_and_note_identity() {
        let mut bad = event(
            "missing-note",
            BeatPosition::new(0, 1).unwrap(),
            GestureParameter::DamperPedal,
            GestureValue::Boolean { value: true },
            GestureInterpolation::Step,
        );
        bad.target = GestureTarget::Note {
            part_id: u16::MAX,
            note_id: String::new(),
        };
        let issues = PerformanceGestureTrackV1::new(vec![bad])
            .validate()
            .unwrap_err();
        assert!(issues.iter().any(|issue| issue.contains("unassigned part")));
        assert!(issues.iter().any(|issue| issue.contains("empty note_id")));
    }

    #[test]
    fn events_are_chronological_and_duplicate_cells_are_rejected() {
        let first = event(
            "b",
            BeatPosition::new(1, 1).unwrap(),
            GestureParameter::BowForce,
            GestureValue::Scalar {
                amount: 1.0,
                unit: GestureUnit::Newtons,
            },
            GestureInterpolation::Step,
        );
        let duplicate = event(
            "c",
            BeatPosition::new(1, 1).unwrap(),
            GestureParameter::BowForce,
            GestureValue::Scalar {
                amount: 1.5,
                unit: GestureUnit::Newtons,
            },
            GestureInterpolation::Step,
        );
        let earlier = event(
            "a",
            BeatPosition::new(1, 2).unwrap(),
            GestureParameter::BowSpeed,
            GestureValue::Scalar {
                amount: 0.4,
                unit: GestureUnit::MetersPerSecond,
            },
            GestureInterpolation::Step,
        );
        let issues = PerformanceGestureTrackV1::new(vec![first, duplicate, earlier])
            .validate()
            .unwrap_err();
        assert!(issues.iter().any(|issue| issue.contains("duplicate parameter event")));
        assert!(issues.iter().any(|issue| issue.contains("not in chronological order")));
    }

    #[test]
    fn unknown_source_model_is_preserved_without_being_promoted() {
        let mut unresolved = event(
            "unknown-model",
            BeatPosition::new(0, 1).unwrap(),
            GestureParameter::BellowsDirection,
            GestureValue::Choice {
                value: "push".to_string(),
            },
            GestureInterpolation::Step,
        );
        unresolved.instrument_id = "accordion".to_string();
        unresolved.source_model_id = None;
        assert!(PerformanceGestureTrackV1::new(vec![unresolved])
            .validate()
            .is_ok());
    }

    #[test]
    fn schema_version_mismatch_is_rejected() {
        let track = PerformanceGestureTrackV1 {
            schema_version: PERFORMANCE_GESTURE_SCHEMA_VERSION + 1,
            events: Vec::new(),
        };
        assert!(track
            .validate()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("unsupported gesture schema version")));
    }
}
