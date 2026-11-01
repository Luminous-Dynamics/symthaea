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
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::HashSet;

/// Current wire-schema version for performance gesture tracks.
pub const PERFORMANCE_GESTURE_SCHEMA_VERSION: u16 = 1;

/// Exact, non-negative score-relative position measured in quarter-note beats.
/// The denominator must be positive; constructors reduce the fraction. The
/// fields remain private so ordinary callers cannot construct a non-canonical
/// value, while validation still protects deserialized data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

    fn checked_add(self, other: Self) -> Option<Self> {
        let numerator = (self.numerator as u128)
            .checked_mul(other.denominator as u128)?
            .checked_add(
                (other.numerator as u128).checked_mul(self.denominator as u128)?,
            )?;
        let denominator = (self.denominator as u128)
            .checked_mul(other.denominator as u128)?;
        let divisor = gcd_u128(numerator, denominator);
        let reduced_numerator = numerator / divisor;
        let reduced_denominator = denominator / divisor;
        Some(Self {
            numerator: u64::try_from(reduced_numerator).ok()?,
            denominator: u32::try_from(reduced_denominator).ok()?,
        })
    }
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
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
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

impl PerformanceGestureEvent {
    /// Stable content digest over the versioned Serde representation. This is
    /// a content commitment, not an author signature; the canonical JSON
    /// serializer and schema version are part of the digest contract.
    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:muse:performance-gesture-event:v1\n");
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

/// Producer-assigned identity and instrument binding for one legacy score note.
///
/// This sidecar avoids breaking ScoreNote literals/serialization while still
/// requiring explicit note IDs. note_index is meaningful only under the exact
/// score digest carried by this map; it is never used to invent note identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreNoteIdentityV1 {
    pub note_index: usize,
    pub note_id: String,
    pub part_id: u16,
    pub instrument_id: String,
    pub source_model_id: Option<String>,
    pub onset: BeatPosition,
    pub duration: BeatPosition,
}

/// Complete score-bound identity map for one immutable score serialization.
///
/// The map digest detects accidental or adversarial mutation of the sidecar's
/// contents, but is not an author signature. The expected score digest must
/// come from the score owner's canonical serializer; capability promotion
/// remains subject to the independent signed verifier used by #9586.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreNoteIdentityMapV1 {
    pub schema_version: u16,
    pub score_sha256: String,
    pub note_count: usize,
    pub notes: Vec<ScoreNoteIdentityV1>,
    pub identity_map_sha256: String,
}

impl ScoreNoteIdentityMapV1 {
    pub fn new(
        score_sha256: impl Into<String>,
        note_count: usize,
        notes: Vec<ScoreNoteIdentityV1>,
    ) -> Self {
        let mut map = Self {
            schema_version: 1,
            score_sha256: score_sha256.into(),
            note_count,
            notes,
            identity_map_sha256: String::new(),
        };
        map.identity_map_sha256 = map.compute_digest();
        map
    }

    /// Validate schema, exact score binding, complete index coverage and
    /// every producer-assigned identity. No IDs are inferred here.
    pub fn validate_for_score(
        &self,
        expected_score_sha256: &str,
        expected_note_count: usize,
    ) -> Result<(), Vec<String>> {
        let mut issues = Vec::new();
        if self.schema_version != 1 {
            issues.push(format!(
                "unsupported note identity map schema version: {}",
                self.schema_version
            ));
        }
        if !is_sha256_hex(&self.score_sha256) {
            issues.push("identity map score_sha256 is malformed".to_string());
        } else if self.score_sha256 != expected_score_sha256 {
            issues.push("identity map is bound to a different score digest".to_string());
        }
        if self.note_count != expected_note_count {
            issues.push("identity map note_count does not match score".to_string());
        }
        if self.notes.len() != self.note_count || self.notes.len() != expected_note_count {
            issues.push("identity map must contain exactly one entry per score note".to_string());
        }
        if !is_sha256_hex(&self.identity_map_sha256)
            || self.identity_map_sha256 != self.compute_digest()
        {
            issues.push("identity map digest is invalid or stale".to_string());
        }

        let mut note_ids = HashSet::new();
        for (index, note) in self.notes.iter().enumerate() {
            if note.note_index != index {
                issues.push(format!(
                    "identity map entry {index} must bind exact score note index {index}"
                ));
            }
            if note.note_id.trim().is_empty() || !note_ids.insert(note.note_id.as_str()) {
                issues.push(format!("identity map entry {index} has an empty or duplicate note_id"));
            }
            if note.part_id == u16::MAX {
                issues.push(format!("identity map entry {index} has unassigned part identity"));
            }
            if note.instrument_id.trim().is_empty() {
                issues.push(format!("identity map entry {index} has empty instrument_id"));
            }
            if note.source_model_id.as_ref().is_some_and(|id| id.trim().is_empty()) {
                issues.push(format!("identity map entry {index} has empty source_model_id"));
            }
            if !note.onset.is_canonical() || !note.duration.is_canonical() {
                issues.push(format!("identity map entry {index} has non-canonical timing"));
            }
            if note.duration.numerator == 0 {
                issues.push(format!("identity map entry {index} has zero duration"));
            }
            if note.onset.checked_add(note.duration).is_none() {
                issues.push(format!("identity map entry {index} timing overflows exact beat range"));
            }
        }

        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }

    pub fn note(&self, part_id: u16, note_id: &str) -> Option<&ScoreNoteIdentityV1> {
        self.notes.iter().find(|note| {
            note.part_id == part_id && note.note_id == note_id
        })
    }

    fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea:muse:score-note-identity-map:v1\n");
        hasher.update(self.schema_version.to_be_bytes());
        hash_string(&mut hasher, &self.score_sha256);
        hasher.update((self.note_count as u64).to_be_bytes());
        hasher.update((self.notes.len() as u64).to_be_bytes());
        for note in &self.notes {
            hasher.update((note.note_index as u64).to_be_bytes());
            hash_string(&mut hasher, &note.note_id);
            hasher.update(note.part_id.to_be_bytes());
            hash_string(&mut hasher, &note.instrument_id);
            match note.source_model_id.as_deref() {
                Some(model) => {
                    hasher.update([1u8]);
                    hash_string(&mut hasher, model);
                }
                None => hasher.update([0u8]),
            }
            hasher.update(note.onset.numerator.to_be_bytes());
            hasher.update(note.onset.denominator.to_be_bytes());
            hasher.update(note.duration.numerator.to_be_bytes());
            hasher.update(note.duration.denominator.to_be_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

/// Additive sidecar. Existing Score and ScoreNote serialization are unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceGestureTrackV1 {
    pub schema_version: u16,
    /// Optional digest binding to the exact canonical score serialization.
    /// An unbound track may be authored and validated, but must not be rendered
    /// against a score until validate_for_score() succeeds.
    pub score_sha256: Option<String>,
    pub events: Vec<PerformanceGestureEvent>,
}

impl Default for PerformanceGestureTrackV1 {
    fn default() -> Self {
        Self {
            schema_version: PERFORMANCE_GESTURE_SCHEMA_VERSION,
            score_sha256: None,
            events: Vec::new(),
        }
    }
}

impl PerformanceGestureTrackV1 {
    pub fn new(events: Vec<PerformanceGestureEvent>) -> Self {
        Self {
            schema_version: PERFORMANCE_GESTURE_SCHEMA_VERSION,
            score_sha256: None,
            events,
        }
    }

    /// Attach the digest produced by the score owner's canonical serializer.
    /// This method validates digest shape; the caller remains responsible for
    /// using the exact score serialization contract.
    pub fn bind_to_score_sha256(&mut self, score_sha256: impl Into<String>) {
        self.score_sha256 = Some(score_sha256.into());
    }

    /// Require this track to match the exact score digest supplied by the
    /// score owner before renderer execution.
    pub fn validate_for_score(&self, expected_score_sha256: &str) -> Result<(), Vec<String>> {
        let mut issues = self.validate().err().unwrap_or_default();
        match self.score_sha256.as_deref() {
            None => issues.push("gesture track is not bound to a score digest".to_string()),
            Some(actual) if actual != expected_score_sha256 => {
                issues.push("gesture track score digest does not match requested score".to_string());
            }
            Some(_) => {}
        }
        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }

    /// Validate this stream and resolve every target against producer-assigned
    /// note identities for the exact score. Renderers must call this before
    /// consuming the gesture events.
    pub fn validate_against_identity_map(
        &self,
        expected_score_sha256: &str,
        expected_note_count: usize,
        identity_map: &ScoreNoteIdentityMapV1,
    ) -> Result<(), Vec<String>> {
        let mut issues = self.validate().err().unwrap_or_default();
        if let Err(map_issues) = identity_map.validate_for_score(
            expected_score_sha256,
            expected_note_count,
        ) {
            issues.extend(map_issues);
        }
        if self.score_sha256.as_deref() != Some(expected_score_sha256) {
            issues.push("gesture track score digest does not match requested score".to_string());
        }
        if identity_map.score_sha256 != expected_score_sha256 {
            issues.push("gesture track and note identity map do not share the requested score".to_string());
        }

        for (index, event) in self.events.iter().enumerate() {
            match &event.target {
                GestureTarget::Note { part_id, note_id } => match identity_map.note(*part_id, note_id) {
                    None => issues.push(format!(
                        "event[{index}] references an unknown part/note identity"
                    )),
                    Some(note) => {
                        if note.instrument_id != event.instrument_id {
                            issues.push(format!(
                                "event[{index}] instrument identity does not match target note"
                            ));
                        }
                        if note.source_model_id != event.source_model_id {
                            issues.push(format!(
                                "event[{index}] source model does not exactly match target note"
                            ));
                        }
                        if event.at.cmp_exact(note.onset) == Ordering::Less {
                            issues.push(format!("event[{index}] occurs before target note onset"));
                        }
                        if let Some(note_end) = note.onset.checked_add(note.duration) {
                            if event.at.cmp_exact(note_end) == Ordering::Greater {
                                issues.push(format!("event[{index}] occurs after target note end"));
                            }
                        } else {
                            issues.push(format!("event[{index}] target note timing is not representable"));
                        }
                    }
                },
                GestureTarget::Part { part_id } => {
                    let part_notes: Vec<_> = identity_map.notes.iter()
                        .filter(|note| note.part_id == *part_id).collect();
                    let part_model = part_notes.first().and_then(|note| note.source_model_id.as_deref());
                    let all_share_one_assignment = !part_notes.is_empty()
                        && part_notes.iter().all(|note| {
                            note.instrument_id == event.instrument_id
                                && note.source_model_id.as_deref() == part_model
                        });
                    let event_model_matches = event.source_model_id.as_deref()
                        .is_none_or(|model| Some(model) == part_model);
                    let homogeneous_assignment = all_share_one_assignment && event_model_matches;
                    if !homogeneous_assignment {
                        issues.push(format!(
                            "event[{index}] part scope is absent or spans multiple instrument/model assignments; use note-scoped gestures"
                        ));
                    }
                }
            }
        }

        if issues.is_empty() { Ok(()) } else { Err(issues) }
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

        if let Some(digest) = self.score_sha256.as_deref() {
            if !is_sha256_hex(digest) {
                issues.push("score_sha256 must be 64 lowercase hexadecimal characters".to_string());
            }
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

            if let GestureValue::Choice { value } = &event.value {
                if value.trim().is_empty() {
                    issues.push(format!("{prefix}: categorical gesture value is empty"));
                }
            }

            if event.interpolation == GestureInterpolation::Linear
                && !matches!(&event.value, GestureValue::Scalar { .. })
            {
                issues.push(format!(
                    "{prefix}: categorical and boolean gestures require step interpolation"
                ));
            }

            let key = (&event.target, event.at, event.parameter);
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
    fn duplicate_target_parameter_cell_rejected_even_with_conflicting_instrument_label() {
        let mut first = event(
            "a",
            BeatPosition::new(1, 1).unwrap(),
            GestureParameter::BowForce,
            GestureValue::Scalar { amount: 1.0, unit: GestureUnit::Newtons },
            GestureInterpolation::Step,
        );
        let mut second = first.clone();
        second.event_id = "b".to_string();
        second.instrument_id = "flute".to_string();
        let issues = PerformanceGestureTrackV1::new(vec![first, second])
            .validate().unwrap_err();
        assert!(issues.iter().any(|issue| issue.contains("duplicate parameter event")));
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
    fn rendering_requires_an_exact_score_digest_binding() {
        let mut track = PerformanceGestureTrackV1::default();
        assert!(track.validate_for_score(&"a".repeat(64)).is_err());

        track.bind_to_score_sha256("a".repeat(64));
        assert!(track.validate_for_score(&"a".repeat(64)).is_ok());
        assert!(track.validate_for_score(&"b".repeat(64)).is_err());

        track.score_sha256 = Some("not-a-digest".to_string());
        assert!(track.validate().is_err());
    }

    #[test]
    fn unknown_wire_fields_are_rejected() {
        let json = r#"{
            "schema_version": 1,
            "score_sha256": null,
            "events": [],
            "silently_ignored_capability": true
        }"#;
        assert!(serde_json::from_str::<PerformanceGestureTrackV1>(json).is_err());
    }

    #[test]
    fn empty_categorical_values_are_rejected() {
        let track = PerformanceGestureTrackV1::new(vec![event(
            "empty-choice",
            BeatPosition::new(0, 1).unwrap(),
            GestureParameter::VowelIdentity,
            GestureValue::Choice { value: "  ".to_string() },
            GestureInterpolation::Step,
        )]);
        assert!(track
            .validate()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("categorical gesture value is empty")));
    }

    fn identity_map_fixture() -> ScoreNoteIdentityMapV1 {
        ScoreNoteIdentityMapV1::new(
            "a".repeat(64),
            2,
            vec![
                ScoreNoteIdentityV1 {
                    note_index: 0,
                    note_id: "lead-note-1".to_string(),
                    part_id: 1,
                    instrument_id: "violin".to_string(),
                    source_model_id: Some("violin.model-a".to_string()),
                    onset: BeatPosition::new(0, 1).unwrap(),
                    duration: BeatPosition::new(2, 1).unwrap(),
                },
                ScoreNoteIdentityV1 {
                    note_index: 1,
                    note_id: "lead-note-2".to_string(),
                    part_id: 1,
                    instrument_id: "violin".to_string(),
                    source_model_id: Some("violin.model-a".to_string()),
                    onset: BeatPosition::new(2, 1).unwrap(),
                    duration: BeatPosition::new(2, 1).unwrap(),
                },
            ],
        )
    }

    #[test]
    fn note_identity_map_is_complete_digest_bound_and_resolves_stable_ids() {
        let map = identity_map_fixture();
        assert!(map.validate_for_score(&"a".repeat(64), 2).is_ok());
        assert_eq!(
            map.note(1, "lead-note-2").unwrap().note_index,
            1
        );
        assert!(map.validate_for_score(&"b".repeat(64), 2).is_err());

        let mut tampered = map.clone();
        tampered.notes[0].instrument_id = "flute".to_string();
        assert!(tampered.validate_for_score(&"a".repeat(64), 2).is_err());
    }

    #[test]
    fn identity_map_rejects_missing_or_reordered_score_note_entries() {
        let mut missing = identity_map_fixture();
        missing.notes.pop();
        missing.identity_map_sha256 = missing.compute_digest();
        assert!(missing.validate_for_score(&"a".repeat(64), 2).is_err());

        let mut reordered = identity_map_fixture();
        reordered.notes.swap(0, 1);
        reordered.identity_map_sha256 = reordered.compute_digest();
        assert!(reordered.validate_for_score(&"a".repeat(64), 2).is_err());
    }

    #[test]
    fn note_scoped_gesture_must_match_score_note_part_instrument_model_and_time() {
        let map = identity_map_fixture();
        let mut gesture = event(
            "bow-pressure",
            BeatPosition::new(1, 1).unwrap(),
            GestureParameter::BowForce,
            GestureValue::Scalar {
                amount: 1.2,
                unit: GestureUnit::Newtons,
            },
            GestureInterpolation::Linear,
        );
        gesture.instrument_id = "violin".to_string();
        gesture.source_model_id = None;
        gesture.target = GestureTarget::Note {
            part_id: 1,
            note_id: "lead-note-1".to_string(),
        };
        let mut unresolved_model = PerformanceGestureTrackV1::new(vec![gesture.clone()]);
        unresolved_model.bind_to_score_sha256("a".repeat(64));
        assert!(unresolved_model
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("source model does not exactly match")));

        gesture.source_model_id = Some("violin.model-a".to_string());

        let mut track = PerformanceGestureTrackV1::new(vec![gesture.clone()]);
        track.bind_to_score_sha256("a".repeat(64));
        assert!(track
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .is_ok());

        gesture.target = GestureTarget::Note {
            part_id: 1,
            note_id: "lead-note-2".to_string(),
        };
        gesture.at = BeatPosition::new(3, 1).unwrap();
        let mut second_note_target = PerformanceGestureTrackV1::new(vec![gesture.clone()]);
        second_note_target.bind_to_score_sha256("a".repeat(64));
        // A time inside the second note is valid; the ID is never reconstructed
        // from pitch, timing or its position in the event stream.
        assert!(second_note_target
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .is_ok());

        gesture.target = GestureTarget::Note {
            part_id: 99,
            note_id: "lead-note-1".to_string(),
        };
        let mut wrong_part = PerformanceGestureTrackV1::new(vec![gesture]);
        wrong_part.bind_to_score_sha256("a".repeat(64));
        assert!(wrong_part
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("unknown part/note")));
    }

    #[test]
    fn note_identity_map_rejects_wrong_instrument_wrong_model_and_out_of_note_time() {
        let map = identity_map_fixture();
        let mut base = event(
            "gesture",
            BeatPosition::new(1, 1).unwrap(),
            GestureParameter::BowForce,
            GestureValue::Scalar {
                amount: 0.8,
                unit: GestureUnit::Newtons,
            },
            GestureInterpolation::Linear,
        );
        base.target = GestureTarget::Note {
            part_id: 1,
            note_id: "lead-note-1".to_string(),
        };
        base.instrument_id = "flute".to_string();
        let mut wrong_instrument = PerformanceGestureTrackV1::new(vec![base.clone()]);
        wrong_instrument.bind_to_score_sha256("a".repeat(64));
        assert!(wrong_instrument
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("instrument identity")));

        base.instrument_id = "violin".to_string();
        base.source_model_id = Some("another.model".to_string());
        let mut wrong_model = PerformanceGestureTrackV1::new(vec![base.clone()]);
        wrong_model.bind_to_score_sha256("a".repeat(64));
        assert!(wrong_model
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("source model")));

        base.source_model_id = Some("violin.model-a".to_string());
        base.at = BeatPosition::new(3, 1).unwrap();
        let mut after_note = PerformanceGestureTrackV1::new(vec![base]);
        after_note.bind_to_score_sha256("a".repeat(64));
        assert!(after_note
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("after target note end")));
    }

    #[test]
    fn part_scoped_gesture_requires_explicit_part_instrument_mapping() {
        let map = identity_map_fixture();
        let mut gesture = event(
            "part-bow",
            BeatPosition::new(1, 1).unwrap(),
            GestureParameter::BowSpeed,
            GestureValue::Scalar {
                amount: 0.4,
                unit: GestureUnit::MetersPerSecond,
            },
            GestureInterpolation::Linear,
        );
        gesture.target = GestureTarget::Part { part_id: 1 };
        gesture.instrument_id = "violin".to_string();
        gesture.source_model_id = Some("violin.model-a".to_string());

        let mut track = PerformanceGestureTrackV1::new(vec![gesture.clone()]);
        track.bind_to_score_sha256("a".repeat(64));
        assert!(track
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .is_ok());

        gesture.instrument_id = "flute".to_string();
        let mut wrong_part_instrument = PerformanceGestureTrackV1::new(vec![gesture]);
        wrong_part_instrument.bind_to_score_sha256("a".repeat(64));
        assert!(wrong_part_instrument
            .validate_against_identity_map(&"a".repeat(64), 2, &map)
            .is_err());
    }

    #[test]
    fn part_scoped_gestures_abstain_when_instrument_changes_within_part() {
        let mut map = identity_map_fixture();
        map.notes[1].instrument_id = "flute".to_string();
        map.identity_map_sha256 = map.compute_digest();
        let mut gesture = event(
            "part-wide-control",
            BeatPosition::new(1, 1).unwrap(),
            GestureParameter::BowSpeed,
            GestureValue::Scalar { amount: 0.5, unit: GestureUnit::MetersPerSecond },
            GestureInterpolation::Linear,
        );
        gesture.target = GestureTarget::Part { part_id: 1 };
        gesture.instrument_id = "violin".to_string();
        gesture.source_model_id = Some("violin.model-a".to_string());
        let mut track = PerformanceGestureTrackV1::new(vec![gesture]);
        track.bind_to_score_sha256("a".repeat(64));
        assert!(track.validate_against_identity_map(&"a".repeat(64), 2, &map)
            .unwrap_err().iter().any(|issue| issue.contains("spans multiple instrument/model assignments")));
    }
    #[test]
    fn schema_version_mismatch_is_rejected() {
        let track = PerformanceGestureTrackV1 {
            schema_version: PERFORMANCE_GESTURE_SCHEMA_VERSION + 1,
            score_sha256: None,
            events: Vec::new(),
        };
        assert!(track
            .validate()
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("unsupported gesture schema version")));
    }
}
