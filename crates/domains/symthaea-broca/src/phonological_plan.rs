// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed phonological planning between linguistic formulation and speech realization.
//!
//! This module is intentionally content-safe: it can carry phoneme/syllable structure
//! only when an upstream linguistic formatter actually supplies it. It never fabricates
//! lexical material from a role-only semantic readout.

use serde::{Deserialize, Serialize};

use crate::lexical_binding::LexicalMorphosyntacticBinding;
use crate::linguistic_frame::LinguisticFrame;
use crate::speech_plan::{IntonationIntent, SpeechPlan};

/// Stable identity for the phonological-plan contract.
pub const PHONOLOGICAL_PLAN_VERSION: &str = "broca-phonological-plan-v2";

/// How much linguistic content is actually bound to the production plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentBindingStatus {
    /// Only roles/prosodic intent are known; no lexical or phonological sequence exists.
    RoleStructureOnly,
    /// A phonological segment sequence has been supplied, but lexical source text is not retained.
    PhonologicallyBound,
    /// Lexical material and its phonological sequence are both explicitly supplied.
    LexicallyBound,
}

/// Stress assigned to a syllabic nucleus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyllableStress {
    None,
    Secondary,
    Primary,
}

impl SyllableStress {
    /// Stable wire encoding shared with the realization layer: 0=none, 1=primary, 2=secondary.
    pub fn ordinal(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Primary => 1,
            Self::Secondary => 2,
        }
    }

    /// Semantic precedence for choosing one stress when multiple slot annotations agree.
    pub fn strength(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Secondary => 1,
            Self::Primary => 2,
        }
    }

    pub fn from_ordinal(value: u8) -> Self {
        match value {
            1 => Self::Primary,
            2 => Self::Secondary,
            _ => Self::None,
        }
    }
}

/// A single phoneme slot with explicit syllable/prosodic attachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhonemeSlot {
    /// IPA/ARPABET/other caller-defined phoneme identity. The planner does not interpret it.
    pub symbol: String,
    /// Zero-based syllable index.
    pub syllable_index: usize,
    /// Stress attached to the syllable containing this phoneme.
    pub stress: SyllableStress,
    /// Whether this slot begins the syllable onset.
    pub is_syllable_onset: bool,
    /// Whether the containing syllable is the information-structural focus.
    pub is_focus: bool,
    /// Whether a phrase boundary follows this phoneme.
    pub phrase_boundary_after: bool,
}

impl PhonemeSlot {
    pub fn new(
        symbol: impl Into<String>,
        syllable_index: usize,
        stress: SyllableStress,
        is_syllable_onset: bool,
        is_focus: bool,
        phrase_boundary_after: bool,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            syllable_index,
            stress,
            is_syllable_onset,
            is_focus,
            phrase_boundary_after,
        }
    }
}

/// A compact syllable-level view derived from bound phoneme slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyllableSlot {
    pub index: usize,
    pub stress: SyllableStress,
    pub is_focus: bool,
    pub phrase_boundary_after: bool,
}

/// Deterministic bridge from a linguistic formulation frame into phonological planning.
///
/// When lexical/phonological content is unavailable, segments remains empty and the
/// binding status stays RoleStructureOnly. This is deliberate: absence of content is
/// represented as missing data rather than guessed words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonologicalPlan {
    pub version: String,
    pub content_binding: ContentBindingStatus,
    /// Optional provenance token supplied by the lexical formatter when lexical content is bound.
    pub lexical_provenance: Option<String>,
    pub source_intent: String,
    /// Whether the upstream linguistic stage explicitly authorizes realization.
    pub realization_authorized: bool,
    pub focus_role: Option<String>,
    pub intonation: IntonationIntent,
    /// Relative pitch-range multiplier inherited from the speech-production plan.
    /// 1.0 is neutral; the realization layer must not reinterpret it as speaker base F0.
    pub pitch_range: f32,
    /// Desired prominence of the focus constituent, inherited without re-inference.
    pub prominence: f32,
    pub rate: f32,
    pub pause_weight: f32,
    pub segments: Vec<PhonemeSlot>,
    pub syllables: Vec<SyllableSlot>,
}

impl PhonologicalPlan {
    /// Build phonological state from the explicit linguistic formulation boundary.
    pub fn from_linguistic_frame(frame: &LinguisticFrame) -> Self {
        Self {

            version: PHONOLOGICAL_PLAN_VERSION.to_string(),
            content_binding: ContentBindingStatus::RoleStructureOnly,
            lexical_provenance: None,
            source_intent: frame.source_intent.clone(),
            realization_authorized: frame.ready_for_phonology(),
            focus_role: frame.focus_role.clone(),
            intonation: frame.prosody.intonation,
            pitch_range: frame.prosody.pitch_range,
            prominence: frame.prosody.prominence,
            rate: sanitize_rate(frame.prosody.rate),
            pause_weight: sanitize_unit(frame.prosody.pause_weight),
            segments: Vec::new(),
            syllables: Vec::new(),
        }
    }

    /// Validate this persisted phonological plan against its exact upstream linguistic frame.
    ///
    /// Bound segment/lexical state may be added downstream; source-derived fields may not drift.
    pub fn validate_against_frame(
        &self,
        frame: &LinguisticFrame,
    ) -> Result<(), PhonologicalPlanError> {
        if frame.validate().is_err() {
            return Err(PhonologicalPlanError::UpstreamMismatch);
        }
        self.validate()?;
        if !self.source_intent.is_empty()
            && (self.source_intent != frame.source_intent
                || self.focus_role != frame.focus_role
                || self.intonation != frame.prosody.intonation
                || self.pitch_range != frame.prosody.pitch_range
                || self.prominence != frame.prosody.prominence
                || self.rate != sanitize_rate(frame.prosody.rate)
                || self.pause_weight != sanitize_unit(frame.prosody.pause_weight)
            || self.realization_authorized != frame.ready_for_phonology())
        {
            return Err(PhonologicalPlanError::UpstreamMismatch);
        }
        Ok(())
    }

    /// Compatibility wrapper for callers that have only a SpeechPlan.
    pub fn from_speech_plan(plan: &SpeechPlan) -> Self {
        let frame = LinguisticFrame::from_speech_plan(plan);
        Self::from_linguistic_frame(&frame)
    }

    /// Bind phonology using the exact provenance identity of an existing lexical binding.
    ///
    /// The lexical binding remains the authority for lexical identity; this method only carries
    /// its deterministic token into the downstream phonological plan.
    pub fn bind_lexical_segments_from_binding(
        &mut self,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        segments: Vec<PhonemeSlot>,
    ) -> Result<(), PhonologicalPlanError> {
        binding
            .validate_against_frame(frame)
            .map_err(|_| PhonologicalPlanError::LexicalBindingMismatch)?;
        self.validate_against_frame(frame)?;
        self.bind_lexical_segments(segments, binding.provenance_token())
    }

    /// Validate that this lexicalized phonological plan carries the exact supplied binding.
    pub fn validate_against_lexical_binding(
        &self,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
    ) -> Result<(), PhonologicalPlanError> {
        binding
            .validate_against_frame(frame)
            .map_err(|_| PhonologicalPlanError::LexicalBindingMismatch)?;
        self.validate_against_frame(frame)?;

        if self.content_binding != ContentBindingStatus::LexicallyBound
            || self.lexical_provenance.as_deref() != Some(binding.provenance_token().as_str())
        {
            return Err(PhonologicalPlanError::LexicalBindingMismatch);
        }

        Ok(())
    }

    /// Bind an explicit phoneme sequence without changing the upstream speech plan.
    ///
    /// LexicallyBound is permitted only when the caller has separately retained lexical
    /// provenance. The phonological plan itself deliberately stores no invented word forms.
    pub fn bind_segments(
        &mut self,
        segments: Vec<PhonemeSlot>,
        status: ContentBindingStatus,
    ) -> Result<(), PhonologicalPlanError> {
        if !self.realization_authorized && !segments.is_empty() {
            return Err(PhonologicalPlanError::RealizationNotAuthorized);
        }
        if matches!(status, ContentBindingStatus::LexicallyBound) {
            return Err(PhonologicalPlanError::LexicalBindingWithoutProvenance);
        }
        validate_binding_status(&segments, status)?;
        validate_segment_sequence(&segments)?;

        self.content_binding = status;
        self.lexical_provenance = None;
        self.syllables = derive_syllables(&segments);
        self.segments = segments;
        Ok(())
    }

    /// Bind explicitly lexicalized phonology only when provenance is supplied.
    pub fn bind_lexical_segments(
        &mut self,
        segments: Vec<PhonemeSlot>,
        provenance: impl Into<String>,
    ) -> Result<(), PhonologicalPlanError> {
        let provenance = provenance.into();
        if !self.realization_authorized {
            return Err(PhonologicalPlanError::RealizationNotAuthorized);
        }
        if provenance.trim().is_empty() {
            return Err(PhonologicalPlanError::EmptyLexicalProvenance);
        }
        if !is_blake3_token(&provenance) {
            return Err(PhonologicalPlanError::InvalidLexicalProvenanceFormat);
        }
        validate_segment_sequence(&segments)?;
        if segments.is_empty() {
            return Err(PhonologicalPlanError::LexicalBindingWithoutSegments);
        }

        self.content_binding = ContentBindingStatus::LexicallyBound;
        self.lexical_provenance = Some(provenance);
        self.syllables = derive_syllables(&segments);
        self.segments = segments;
        Ok(())
    }

    /// Validate the whole persisted/deserialized object, including cross-field invariants.
    pub fn validate(&self) -> Result<(), PhonologicalPlanError> {
        if self.version != PHONOLOGICAL_PLAN_VERSION {
            return Err(PhonologicalPlanError::InvalidVersion);
        }
        if self.source_intent.trim().is_empty() {
            return Err(PhonologicalPlanError::EmptySourceIntent);
        }
        if !self.pitch_range.is_finite() || !(0.65..=1.45).contains(&self.pitch_range) {
            return Err(PhonologicalPlanError::InvalidPitchRange);
        }
        if !self.prominence.is_finite() || !(0.0..=1.0).contains(&self.prominence) {
            return Err(PhonologicalPlanError::InvalidProminence);
        }
        if !self.rate.is_finite() || !(0.55..=1.35).contains(&self.rate) {
            return Err(PhonologicalPlanError::InvalidRate);
        }
        if !self.pause_weight.is_finite() || !(0.0..=1.0).contains(&self.pause_weight) {
            return Err(PhonologicalPlanError::InvalidPauseWeight);
        }
        let focused_segments = self.segments.iter().filter(|segment| segment.is_focus).count();
        match self.focus_role.as_deref() {
            Some(_) if focused_segments == 0 => {
                return Err(PhonologicalPlanError::FocusRoleWithoutSegments)
            }
            None if focused_segments != 0 => {
                return Err(PhonologicalPlanError::FocusSegmentsWithoutRole)
            }
            _ => {}
        }

        if !self.realization_authorized && !self.segments.is_empty() {
            return Err(PhonologicalPlanError::RealizationNotAuthorized);
        }

        validate_binding_status(&self.segments, self.content_binding)?;

        match self.content_binding {
            ContentBindingStatus::LexicallyBound => {
                let provenance = self
                    .lexical_provenance
                    .as_deref()
                    .ok_or(PhonologicalPlanError::LexicalBindingWithoutProvenance)?;
                if provenance.trim().is_empty() {
                    return Err(PhonologicalPlanError::EmptyLexicalProvenance);
                }
                if !is_blake3_token(provenance) {
                    return Err(PhonologicalPlanError::InvalidLexicalProvenanceFormat);
                }
            }
            ContentBindingStatus::RoleStructureOnly | ContentBindingStatus::PhonologicallyBound => {
                if self.lexical_provenance.is_some() {
                    return Err(PhonologicalPlanError::NonLexicalProvenance);
                }
            }
        }

        validate_segment_sequence(&self.segments)?;

        let expected_syllables = derive_syllables(&self.segments);
        if expected_syllables != self.syllables {
            return Err(PhonologicalPlanError::SyllableSummaryMismatch);
        }

        Ok(())
    }

    /// Whether enough valid phonological content exists for segment-level realization.
    pub fn ready_for_realization(&self) -> bool {
        !self.segments.is_empty() && self.validate().is_ok()
    }

    /// Stable trace representation for evidence capture.
    pub fn grounding_surface(&self) -> String {
        let segment_surface = self
            .segments
            .iter()
            .map(|segment| {
                format!(
                    "{}@{}:{:?}:onset={}:focus={}:boundary={}",
                    segment.symbol,
                    segment.syllable_index,
                    segment.stress,
                    segment.is_syllable_onset,
                    segment.is_focus,
                    segment.phrase_boundary_after,
                )
            })
            .collect::<Vec<_>>()
            .join("|");

        format!(
            "{};binding={:?};intent={};focus={};intonation={:?};pitch={:.4};prominence={:.4};rate={:.4};pause={:.4};syllables={};lexical_provenance={};segments={}",
            self.version,
            self.content_binding,
            self.source_intent,
            self.focus_role.as_deref().unwrap_or("NONE"),
            self.intonation,
            self.pitch_range,
            self.prominence,
            self.rate,
            self.pause_weight,
            self.syllables.len(),
            self.lexical_provenance.is_some(),
            segment_surface,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhonologicalPlanError {
    InvalidVersion,
    EmptySourceIntent,
    InvalidRate,
    InvalidPitchRange,
    InvalidProminence,
    InvalidPauseWeight,
    FocusRoleWithoutSegments,
    FocusSegmentsWithoutRole,
    RealizationNotAuthorized,
    UpstreamMismatch,
    ConflictingSyllableStress { syllable_index: usize },
    MixedSyllableFocus { syllable_index: usize },
    MultipleSyllableOnsets { syllable_index: usize },
    MidSyllablePhraseBoundary { syllable_index: usize },
    EmptySegmentSymbol { index: usize },
    NonContiguousSyllableIndex { expected: usize, found: usize },
    LexicalBindingWithoutSegments,
    LexicalBindingWithoutProvenance,
    LexicalBindingMismatch,
    EmptyLexicalProvenance,
    InvalidLexicalProvenanceFormat,
    NonLexicalProvenance,
    SyllableSummaryMismatch,
    PhonologicalBindingWithoutSegments,
    RoleOnlyWithSegments,
}

impl std::fmt::Display for PhonologicalPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "phonological plan version is unsupported"),
            Self::EmptySourceIntent => write!(f, "phonological plan source intent must be non-empty"),
            Self::InvalidRate => write!(f, "phonological plan rate is outside the supported range"),
            Self::InvalidPitchRange => write!(f, "phonological plan pitch range is outside the supported range"),
            Self::InvalidProminence => write!(f, "phonological plan prominence is outside the supported range"),
            Self::InvalidPauseWeight => write!(f, "phonological plan pause weight is outside [0, 1]"),
            Self::FocusRoleWithoutSegments => write!(f, "a focus role requires focused phonemes"),
            Self::FocusSegmentsWithoutRole => write!(f, "focused phonemes require a focus role"),
            Self::RealizationNotAuthorized => write!(f, "upstream linguistic state does not authorize speech realization"),
            Self::UpstreamMismatch => write!(f, "phonological plan no longer matches its upstream linguistic frame"),
            Self::ConflictingSyllableStress { syllable_index } => write!(f, "syllable {syllable_index} contains conflicting stress annotations"),
            Self::MixedSyllableFocus { syllable_index } => write!(f, "syllable {syllable_index} contains mixed focus annotations"),
            Self::MultipleSyllableOnsets { syllable_index } => write!(f, "syllable {syllable_index} contains multiple onset markers"),
            Self::MidSyllablePhraseBoundary { syllable_index } => write!(f, "syllable {syllable_index} has an internal phrase boundary"),
            Self::EmptySegmentSymbol { index } => {
                write!(f, "phoneme slot {index} has an empty symbol")
            }
            Self::NonContiguousSyllableIndex { expected, found } => {
                write!(
                    f,
                    "syllable indices must be contiguous from zero: expected {expected}, found {found}"
                )
            }
            Self::LexicalBindingWithoutSegments => {
                write!(f, "lexically bound plans require explicit phonological segments")
            }
            Self::LexicalBindingWithoutProvenance => {
                write!(f, "lexically bound plans require explicit lexical provenance")
            }
            Self::LexicalBindingMismatch => {
                write!(f, "phonological plan does not match the supplied lexical binding")
            }
            Self::EmptyLexicalProvenance => {
                write!(f, "lexically bound plans require non-empty lexical provenance")
            }
            Self::InvalidLexicalProvenanceFormat => {
                write!(f, "lexical provenance must be a 64-character BLAKE3 hexadecimal token")
            }
            Self::NonLexicalProvenance => {
                write!(f, "non-lexical plans must not carry lexical provenance")
            }
            Self::SyllableSummaryMismatch => {
                write!(f, "cached syllable summary does not match phoneme slots")
            }
            Self::PhonologicalBindingWithoutSegments => {
                write!(f, "phonologically bound plans require explicit phonological segments")
            }
            Self::RoleOnlyWithSegments => {
                write!(f, "role-only plans cannot carry phonological segments")
            }
        }
    }
}

impl std::error::Error for PhonologicalPlanError {}

fn validate_binding_status(
    segments: &[PhonemeSlot],
    status: ContentBindingStatus,
) -> Result<(), PhonologicalPlanError> {
    if matches!(status, ContentBindingStatus::RoleStructureOnly) && !segments.is_empty() {
        return Err(PhonologicalPlanError::RoleOnlyWithSegments);
    }

    if segments.is_empty() {
        match status {
            ContentBindingStatus::LexicallyBound => {
                return Err(PhonologicalPlanError::LexicalBindingWithoutSegments)
            }
            ContentBindingStatus::PhonologicallyBound => {
                return Err(PhonologicalPlanError::PhonologicalBindingWithoutSegments)
            }
            ContentBindingStatus::RoleStructureOnly => {}
        }
    }
    Ok(())
}

fn validate_segment_sequence(segments: &[PhonemeSlot]) -> Result<(), PhonologicalPlanError> {
    for (index, segment) in segments.iter().enumerate() {
        if segment.symbol.trim().is_empty() {
            return Err(PhonologicalPlanError::EmptySegmentSymbol { index });
        }
        if segment.phrase_boundary_after
            && segments
                .get(index + 1)
                .is_some_and(|next| next.syllable_index == segment.syllable_index)
        {
            return Err(PhonologicalPlanError::MidSyllablePhraseBoundary {
                syllable_index: segment.syllable_index,
            });
        }
    }

    let mut expected_syllable = 0usize;
    let mut last_syllable = None;
    let mut syllable_stress = SyllableStress::None;
    let mut syllable_focus = false;
    let mut onset_count = 0u8;

    for segment in segments {
        if last_syllable != Some(segment.syllable_index) {
            if segment.syllable_index != expected_syllable {
                return Err(PhonologicalPlanError::NonContiguousSyllableIndex {
                    expected: expected_syllable,
                    found: segment.syllable_index,
                });
            }
            expected_syllable += 1;
            last_syllable = Some(segment.syllable_index);
            syllable_stress = segment.stress;
            syllable_focus = segment.is_focus;
            onset_count = 0;
        } else {
            if segment.stress != syllable_stress {
                return Err(PhonologicalPlanError::ConflictingSyllableStress {
                    syllable_index: segment.syllable_index,
                });
            }
            if segment.is_focus != syllable_focus {
                return Err(PhonologicalPlanError::MixedSyllableFocus {
                    syllable_index: segment.syllable_index,
                });
            }
        }

        if segment.is_syllable_onset {
            onset_count += 1;
            if onset_count > 1 {
                return Err(PhonologicalPlanError::MultipleSyllableOnsets {
                    syllable_index: segment.syllable_index,
                });
            }
        }
    }

    Ok(())
}

fn is_blake3_token(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn derive_syllables(segments: &[PhonemeSlot]) -> Vec<SyllableSlot> {
    let mut syllables = Vec::new();

    for segment in segments {
        if segment.syllable_index >= syllables.len() {
            syllables.resize_with(segment.syllable_index + 1, || SyllableSlot {
                index: 0,
                stress: SyllableStress::None,
                is_focus: false,
                phrase_boundary_after: false,
            });
        }

        let syllable = &mut syllables[segment.syllable_index];
        syllable.index = segment.syllable_index;
        if segment.stress.strength() > syllable.stress.strength() {
            syllable.stress = segment.stress;
        }
        syllable.is_focus |= segment.is_focus;
        syllable.phrase_boundary_after |= segment.phrase_boundary_after;
    }

    syllables
}

fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_rate(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.55, 1.35)
    } else {
        0.92
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SpeechPlan, StructuredDecoder, ThoughtChannels};
    use symthaea_core::genesis::GenesisSeed;

    fn plan() -> SpeechPlan {
        let genesis = GenesisSeed::from_phrase("phonological-plan-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(4);
        let readout = decoder.decode(&channels);
        SpeechPlan::from_readout(&channels, &readout)
    }

    #[test]
    fn legacy_schema_version_fails_closed() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.version = "broca-phonological-plan-v1".to_string();

        assert_eq!(
            plan.validate()
                .expect_err("legacy phonological schema must be rejected"),
            PhonologicalPlanError::InvalidVersion
        );
    }

    #[test]
    fn unbound_plan_is_explicitly_not_ready() {
        let plan = PhonologicalPlan::from_speech_plan(&plan());
        assert_eq!(plan.content_binding, ContentBindingStatus::RoleStructureOnly);
        assert!(!plan.ready_for_realization());
        assert_eq!(plan.syllables.len(), 0);
    }

    
    #[test]
    fn pitch_range_is_carried_from_speech_plan_and_validated() {
        let mut speech_plan = plan();
        speech_plan.prosody.pitch_range = 1.25;

        let mut phonological = PhonologicalPlan::from_speech_plan(&speech_plan);
        assert!((phonological.pitch_range - 1.25).abs() < f32::EPSILON);

        phonological.pitch_range = 2.0;
        assert_eq!(
            phonological
                .validate()
                .expect_err("out-of-range pitch must fail closed"),
            PhonologicalPlanError::InvalidPitchRange
        );
    }

    #[test]
    fn prominence_is_carried_from_speech_plan_and_validated() {
        let mut speech_plan = plan();
        speech_plan.prosody.prominence = 0.73;

        let mut phonological = PhonologicalPlan::from_speech_plan(&speech_plan);
        assert!((phonological.prominence - 0.73).abs() < f32::EPSILON);

        phonological.prominence = 1.5;
        assert_eq!(
            phonological
                .validate()
                .expect_err("out-of-range prominence must fail closed"),
            PhonologicalPlanError::InvalidProminence
        );
    }

    #[test]
    fn pitch_range_mismatch_with_upstream_frame_is_rejected() {
        let speech_plan = plan();
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);
        let mut phonological = PhonologicalPlan::from_linguistic_frame(&frame);
        phonological.pitch_range = if phonological.pitch_range < 1.0 {
            1.0
        } else {
            0.65
        };

        assert_eq!(
            phonological
                .validate_against_frame(&frame)
                .expect_err("tampered pitch range must fail lineage validation"),
            PhonologicalPlanError::UpstreamMismatch
        );
    }

    #[test]
    fn explicit_segments_bind_without_fabricating_lexical_content() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        let segments = vec![
            PhonemeSlot::new("B", 0, SyllableStress::Primary, true, true, false),
            PhonemeSlot::new("AE", 0, SyllableStress::Primary, false, true, false),
            PhonemeSlot::new("T", 0, SyllableStress::Primary, false, true, true),
        ];

        plan.bind_segments(segments, ContentBindingStatus::PhonologicallyBound)
            .expect("valid phonological sequence");

        assert!(plan.ready_for_realization());
        assert_eq!(plan.content_binding, ContentBindingStatus::PhonologicallyBound);
        assert_eq!(plan.syllables.len(), 1);
        assert_eq!(plan.syllables[0].stress, SyllableStress::Primary);
        assert!(plan.syllables[0].is_focus);
        assert!(plan.syllables[0].phrase_boundary_after);
    }

    #[test]
    fn noncontiguous_syllables_are_rejected() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        let segments = vec![
            PhonemeSlot::new("AH", 0, SyllableStress::None, true, false, false),
            PhonemeSlot::new("T", 2, SyllableStress::Primary, true, false, false),
        ];

        let error = plan
            .bind_segments(segments, ContentBindingStatus::PhonologicallyBound)
            .expect_err("gap must fail closed");

        assert!(matches!(
            error,
            PhonologicalPlanError::NonContiguousSyllableIndex {
                expected: 1,
                found: 2
            }
        ));
    }

    #[test]
    fn stress_wire_encoding_matches_realization_convention() {
        assert_eq!(SyllableStress::None.ordinal(), 0);
        assert_eq!(SyllableStress::Primary.ordinal(), 1);
        assert_eq!(SyllableStress::Secondary.ordinal(), 2);
        assert_eq!(SyllableStress::from_ordinal(1), SyllableStress::Primary);
        assert_eq!(SyllableStress::from_ordinal(2), SyllableStress::Secondary);
    }

    #[test]
    fn primary_stress_dominates_secondary_when_derived() {
        let segments = vec![
            PhonemeSlot::new("AE", 0, SyllableStress::Secondary, true, false, false),
            PhonemeSlot::new("IY", 0, SyllableStress::Primary, false, false, true),
        ];

        let syllables = derive_syllables(&segments);
        assert_eq!(syllables[0].stress, SyllableStress::Primary);
    }

    #[test]
    fn role_only_status_rejects_bound_segments() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        let segments = vec![PhonemeSlot::new(
            "AH",
            0,
            SyllableStress::None,
            true,
            false,
            false,
        )];

        let error = plan
            .bind_segments(segments, ContentBindingStatus::RoleStructureOnly)
            .expect_err("binding status must match payload");

        assert_eq!(error, PhonologicalPlanError::RoleOnlyWithSegments);
    }

    #[test]
    fn lexical_binding_requires_provenance() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        let segments = vec![PhonemeSlot::new(
            "AE",
            0,
            SyllableStress::Primary,
            true,
            false,
            true,
        )];

        let error = plan
            .bind_segments(segments, ContentBindingStatus::LexicallyBound)
            .expect_err("lexical status must require provenance");

        assert_eq!(
            error,
            PhonologicalPlanError::LexicalBindingWithoutProvenance
        );
    }

    #[test]
    fn lexical_binding_records_non_empty_provenance() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        let provenance = blake3::hash(b"lexeme:answer:v1").to_hex().to_string();
        plan.bind_lexical_segments(
            vec![PhonemeSlot::new(
                "AE",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            provenance.clone(),
        )
        .expect("explicit BLAKE3 provenance should permit lexical binding");

        assert_eq!(plan.content_binding, ContentBindingStatus::LexicallyBound);
        assert_eq!(plan.lexical_provenance.as_deref(), Some(provenance.as_str()));
        assert!(plan.grounding_surface().contains("lexical_provenance=true"));
    }

    #[test]
    fn lexical_binding_rejects_non_blake3_provenance() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());

        let error = plan
            .bind_lexical_segments(
                vec![PhonemeSlot::new(
                    "AE",
                    0,
                    SyllableStress::Primary,
                    true,
                    false,
                    true,
                )],
                "arbitrary-provenance",
            )
            .expect_err("arbitrary provenance must not elevate lexical binding");

        assert_eq!(
            error,
            PhonologicalPlanError::InvalidLexicalProvenanceFormat
        );
    }

    #[test]
    fn persisted_lexical_binding_rejects_malformed_provenance() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.content_binding = ContentBindingStatus::LexicallyBound;
        plan.segments = vec![PhonemeSlot::new(
            "AE",
            0,
            SyllableStress::Primary,
            true,
            false,
            true,
        )];
        plan.syllables = derive_syllables(&plan.segments);
        plan.lexical_provenance = Some("arbitrary-persisted-provenance".to_string());

        assert_eq!(
            plan.validate()
                .expect_err("persisted lexical provenance must be canonical"),
            PhonologicalPlanError::InvalidLexicalProvenanceFormat
        );
    }

    #[test]
    fn lexical_binding_requires_actual_segments() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());

        let error = plan
            .bind_segments(Vec::new(), ContentBindingStatus::LexicallyBound)
            .expect_err("lexical binding cannot be asserted without phonology");

        assert_eq!(error, PhonologicalPlanError::LexicalBindingWithoutProvenance);
    }

    #[test]
    fn unauthorized_deserialized_segments_fail_closed() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.realization_authorized = false;
        plan.content_binding = ContentBindingStatus::PhonologicallyBound;
        plan.segments = vec![PhonemeSlot::new(
            "AH",
            0,
            SyllableStress::Primary,
            true,
            false,
            true,
        )];
        plan.syllables = derive_syllables(&plan.segments);

        let error = plan
            .validate()
            .expect_err("unauthorized segments must fail closed");

        assert_eq!(error, PhonologicalPlanError::RealizationNotAuthorized);
        assert!(!plan.ready_for_realization());
    }

    #[test]
    fn deserialized_style_cross_field_invariants_are_validated() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.segments = vec![PhonemeSlot::new(
            "AH",
            0,
            SyllableStress::None,
            true,
            false,
            false,
        )];

        let error = plan
            .validate()
            .expect_err("role-only plans cannot carry phonological data");

        assert_eq!(error, PhonologicalPlanError::RoleOnlyWithSegments);
        assert!(!plan.ready_for_realization());
    }

    #[test]
    fn focus_state_must_match_the_plan() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.bind_segments(
            vec![PhonemeSlot::new("AE", 0, SyllableStress::Primary, true, true, true)],
            ContentBindingStatus::PhonologicallyBound,
        ).unwrap();
        assert_eq!(
            plan.validate().expect_err("focus without target"),
            PhonologicalPlanError::FocusSegmentsWithoutRole
        );
    }

    #[test]
    fn malformed_sequence_annotations_fail_closed() {
        let mixed_stress = vec![
            PhonemeSlot::new("A", 0, SyllableStress::Primary, true, false, false),
            PhonemeSlot::new("B", 0, SyllableStress::None, false, false, true),
        ];
        assert_eq!(
            validate_segment_sequence(&mixed_stress).expect_err("stress mismatch"),
            PhonologicalPlanError::ConflictingSyllableStress { syllable_index: 0 }
        );

        let mixed_focus = vec![
            PhonemeSlot::new("A", 0, SyllableStress::Primary, true, true, false),
            PhonemeSlot::new("B", 0, SyllableStress::Primary, false, false, true),
        ];
        assert_eq!(
            validate_segment_sequence(&mixed_focus).expect_err("focus mismatch"),
            PhonologicalPlanError::MixedSyllableFocus { syllable_index: 0 }
        );

        let double_onset = vec![
            PhonemeSlot::new("A", 0, SyllableStress::Primary, true, false, false),
            PhonemeSlot::new("B", 0, SyllableStress::Primary, true, false, true),
        ];
        assert_eq!(
            validate_segment_sequence(&double_onset).expect_err("multiple onsets"),
            PhonologicalPlanError::MultipleSyllableOnsets { syllable_index: 0 }
        );

        let internal_boundary = vec![
            PhonemeSlot::new("A", 0, SyllableStress::Primary, true, false, true),
            PhonemeSlot::new("B", 0, SyllableStress::Primary, false, false, false),
        ];
        assert_eq!(
            validate_segment_sequence(&internal_boundary).expect_err("internal boundary"),
            PhonologicalPlanError::MidSyllablePhraseBoundary { syllable_index: 0 }
        );
    }

        #[test]
    fn lexical_provenance_is_required_on_the_persisted_object() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.content_binding = ContentBindingStatus::LexicallyBound;
        plan.segments = sample_for_validation();

        let error = plan
            .validate()
            .expect_err("lexical binding without provenance must fail closed");

        assert_eq!(error, PhonologicalPlanError::LexicalBindingWithoutProvenance);
    }

    #[test]
    fn persisted_syllable_summary_must_match_segments() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "AE",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .unwrap();
        plan.syllables[0].stress = SyllableStress::None;

        let error = plan
            .validate()
            .expect_err("stale derived syllable data must fail validation");

        assert_eq!(error, PhonologicalPlanError::SyllableSummaryMismatch);
    }

    #[test]
    fn non_lexical_plan_cannot_retain_lexical_provenance() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.lexical_provenance = Some("stale".to_string());

        let error = plan
            .validate()
            .expect_err("stale provenance must fail validation");

        assert_eq!(error, PhonologicalPlanError::NonLexicalProvenance);
    }

    fn sample_for_validation() -> Vec<PhonemeSlot> {
        vec![PhonemeSlot::new(
            "AE",
            0,
            SyllableStress::Primary,
            true,
            false,
            true,
        )]
    }

    #[test]
    fn grounding_surface_is_stable_and_marks_binding() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "IY",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .unwrap();

        let first = plan.grounding_surface();
        let second = plan.grounding_surface();

        assert_eq!(first, second);
        assert!(first.contains("binding=PhonologicallyBound"));
        assert!(first.contains("IY@0"));
    }
}

