// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed phonological planning between semantic/linguistic intent and speech realization.
//!
//! This module is intentionally content-safe: it can carry phoneme/syllable structure
//! only when an upstream linguistic formatter actually supplies it. It never fabricates
//! lexical material from a role-only semantic readout.

use serde::{Deserialize, Serialize};

use crate::speech_plan::{IntonationIntent, SpeechPlan};

/// Stable identity for the phonological-plan contract.
pub const PHONOLOGICAL_PLAN_VERSION: &str = "broca-phonological-plan-v1";

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
    pub fn ordinal(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Primary => 2,
            Self::Secondary => 1,
        }
    }

    pub fn from_ordinal(value: u8) -> Self {
        match value {
            1 => Self::Secondary,
            2 => Self::Primary,
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

/// Deterministic bridge from a SpeechPlan into phonological planning.
///
/// When lexical/phonological content is unavailable, segments remains empty and the
/// binding status stays RoleStructureOnly. This is deliberate: absence of content is
/// represented as missing data rather than guessed words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonologicalPlan {
    pub version: String,
    pub content_binding: ContentBindingStatus,
    pub source_intent: String,
    pub focus_role: Option<String>,
    pub intonation: IntonationIntent,
    pub rate: f32,
    pub pause_weight: f32,
    pub segments: Vec<PhonemeSlot>,
    pub syllables: Vec<SyllableSlot>,
}

impl PhonologicalPlan {
    pub fn from_speech_plan(plan: &SpeechPlan) -> Self {
        Self {
            version: PHONOLOGICAL_PLAN_VERSION.to_string(),
            content_binding: ContentBindingStatus::RoleStructureOnly,
            source_intent: plan.intent.clone(),
            focus_role: plan.focus_role.clone(),
            intonation: plan.prosody.intonation,
            rate: sanitize_rate(plan.prosody.rate),
            pause_weight: sanitize_unit(plan.prosody.pause_weight),
            segments: Vec::new(),
            syllables: Vec::new(),
        }
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
        validate_binding_status(&segments, status)?;
        validate_segment_sequence(&segments)?;

        self.content_binding = status;
        self.syllables = derive_syllables(&segments);
        self.segments = segments;
        Ok(())
    }

    /// Whether enough phonological content exists for segment-level realization.
    pub fn ready_for_realization(&self) -> bool {
        !self.segments.is_empty()
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
            "{};binding={:?};intent={};focus={};intonation={:?};rate={:.4};pause={:.4};syllables={};segments={}",
            self.version,
            self.content_binding,
            self.source_intent,
            self.focus_role.as_deref().unwrap_or("NONE"),
            self.intonation,
            self.rate,
            self.pause_weight,
            self.syllables.len(),
            segment_surface,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhonologicalPlanError {
    EmptySegmentSymbol { index: usize },
    NonContiguousSyllableIndex { expected: usize, found: usize },
    LexicalBindingWithoutSegments,
    PhonologicalBindingWithoutSegments,
    RoleOnlyWithSegments,
}

impl std::fmt::Display for PhonologicalPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
    }

    let mut expected_syllable = 0usize;
    let mut last_syllable = None;
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
        }
    }

    Ok(())
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
        if segment.stress.ordinal() > syllable.stress.ordinal() {
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
    fn unbound_plan_is_explicitly_not_ready() {
        let plan = PhonologicalPlan::from_speech_plan(&plan());
        assert_eq!(plan.content_binding, ContentBindingStatus::RoleStructureOnly);
        assert!(!plan.ready_for_realization());
        assert_eq!(plan.syllables.len(), 0);
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
    fn lexical_binding_requires_actual_segments() {
        let mut plan = PhonologicalPlan::from_speech_plan(&plan());

        let error = plan
            .bind_segments(Vec::new(), ContentBindingStatus::LexicallyBound)
            .expect_err("lexical binding cannot be asserted without phonology");

        assert_eq!(error, PhonologicalPlanError::LexicalBindingWithoutSegments);
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
