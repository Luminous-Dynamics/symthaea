// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Blinded presentation state for deliberate A/B comparison.
//!
//! `ComparisonSession` owns the real subjects and transport semantics. This
//! module maps the underlying sides to visible A/B labels and exposes a
//! metadata-free view until an explicit, irreversible reveal. It does not
//! generate randomness: callers must supply the assignment bit so research
//! workflows can record and reproduce their randomization procedure.

use crate::comparison::{ComparisonSession, ComparisonSide};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlindTrialId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlindLabel {
    A,
    B,
}

impl BlindLabel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlindRevealState {
    Blinded,
    Revealed,
}

/// Stable assignment between visible blind labels and the comparison session's
/// underlying sides. `swapped` is supplied by the caller; this type does not
/// make a claim about the randomness source or its suitability for a study.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindAssignment {
    swapped: bool,
}

impl BlindAssignment {
    pub const fn new(swapped: bool) -> Self {
        Self { swapped }
    }

    pub const fn side_for_label(self, label: BlindLabel) -> ComparisonSide {
        match (self.swapped, label) {
            (false, BlindLabel::A) | (true, BlindLabel::B) => ComparisonSide::A,
            (false, BlindLabel::B) | (true, BlindLabel::A) => ComparisonSide::B,
        }
    }

    pub const fn label_for_side(self, side: ComparisonSide) -> BlindLabel {
        match (self.swapped, side) {
            (false, ComparisonSide::A) | (true, ComparisonSide::B) => BlindLabel::A,
            (false, ComparisonSide::B) | (true, ComparisonSide::A) => BlindLabel::B,
        }
    }
}

/// The only subject data a blinded presentation needs. There is intentionally
/// no title, style, renderer, URL, identity hash, provenance, or lineage field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindSubjectView {
    pub label: BlindLabel,
    pub active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RevealedBlindAssignment {
    pub visible_a_side: ComparisonSide,
    pub visible_b_side: ComparisonSide,
}

/// One irreversible blind/reveal lifecycle.
///
/// This state is deliberately neither `Copy` nor `Clone`: UI code cannot retain
/// a copied pre-reveal value, reveal another copy, then continue presenting the
/// stale copy as if the same trial were still blinded. A new blind state can be
/// constructed for a new trial, but this instance only transitions forward.
#[derive(Debug, PartialEq, Eq)]
pub struct BlindComparisonState {
    trial_id: BlindTrialId,
    assignment: BlindAssignment,
    reveal_state: BlindRevealState,
    judgment_recorded: bool,
}

impl BlindComparisonState {
    pub const fn new(trial_id: BlindTrialId, swapped: bool) -> Self {
        Self {
            trial_id,
            assignment: BlindAssignment::new(swapped),
            reveal_state: BlindRevealState::Blinded,
            judgment_recorded: false,
        }
    }

    pub const fn trial_id(&self) -> BlindTrialId {
        self.trial_id
    }

    pub const fn reveal_state(&self) -> BlindRevealState {
        self.reveal_state
    }

    pub const fn is_blinded(&self) -> bool {
        matches!(self.reveal_state, BlindRevealState::Blinded)
    }

    pub const fn has_recorded_judgment(&self) -> bool {
        self.judgment_recorded
    }

    pub fn visible_subjects(&self, session: &ComparisonSession) -> [BlindSubjectView; 2] {
        [
            BlindSubjectView {
                label: BlindLabel::A,
                active: session.active_side == self.assignment.side_for_label(BlindLabel::A),
            },
            BlindSubjectView {
                label: BlindLabel::B,
                active: session.active_side == self.assignment.side_for_label(BlindLabel::B),
            },
        ]
    }

    /// Resolve a blind label for transport without exposing presentation or
    /// identity metadata through the blind-view type itself.
    pub const fn side_for_transport(&self, label: BlindLabel) -> ComparisonSide {
        self.assignment.side_for_label(label)
    }

    /// Used by the judgment boundary only after all judgment validations pass.
    /// Returns false if this trial already has a recorded judgment.
    pub(crate) fn mark_judgment_recorded(&mut self) -> bool {
        if self.judgment_recorded {
            return false;
        }
        self.judgment_recorded = true;
        true
    }

    /// Reveal the mapping exactly once. Repeated calls are idempotent but there
    /// is deliberately no inverse operation.
    pub fn reveal(&mut self) -> RevealedBlindAssignment {
        self.reveal_state = BlindRevealState::Revealed;
        RevealedBlindAssignment {
            visible_a_side: self.assignment.side_for_label(BlindLabel::A),
            visible_b_side: self.assignment.side_for_label(BlindLabel::B),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::ComparisonSubject;
    use crate::playback::{PlaybackPresentation, PlaybackSource};

    const TRIAL: BlindTrialId = BlindTrialId(17);

    fn source(title: &str) -> PlaybackSource {
        PlaybackSource {
            rendition_id: None,
            audio_url: format!("/audio/{title}"),
            duration_hint_seconds: Some(30.0),
            advance_on_end: false,
            presentation: PlaybackPresentation::review(title.to_string()),
        }
    }

    fn session() -> ComparisonSession {
        let a = ComparisonSubject::new(source("descriptive-title-a"), None).unwrap();
        let b = ComparisonSubject::new(source("descriptive-title-b"), None).unwrap();
        ComparisonSession::new(a, b).unwrap()
    }

    #[test]
    fn blind_state_preserves_explicit_trial_identity() {
        let blind = BlindComparisonState::new(TRIAL, false);
        assert_eq!(blind.trial_id(), TRIAL);
        assert!(!blind.has_recorded_judgment());
    }

    #[test]
    fn unswapped_assignment_maps_visible_labels_directly() {
        let assignment = BlindAssignment::new(false);
        assert_eq!(assignment.side_for_label(BlindLabel::A), ComparisonSide::A);
        assert_eq!(assignment.side_for_label(BlindLabel::B), ComparisonSide::B);
        assert_eq!(assignment.label_for_side(ComparisonSide::A), BlindLabel::A);
        assert_eq!(assignment.label_for_side(ComparisonSide::B), BlindLabel::B);
    }

    #[test]
    fn swapped_assignment_hides_underlying_side_names() {
        let assignment = BlindAssignment::new(true);
        assert_eq!(assignment.side_for_label(BlindLabel::A), ComparisonSide::B);
        assert_eq!(assignment.side_for_label(BlindLabel::B), ComparisonSide::A);
        assert_eq!(assignment.label_for_side(ComparisonSide::A), BlindLabel::B);
        assert_eq!(assignment.label_for_side(ComparisonSide::B), BlindLabel::A);
    }

    #[test]
    fn blind_view_contains_no_subject_metadata() {
        let session = session();
        let blind = BlindComparisonState::new(TRIAL, true);
        assert_eq!(
            blind.visible_subjects(&session),
            [
                BlindSubjectView {
                    label: BlindLabel::A,
                    active: false,
                },
                BlindSubjectView {
                    label: BlindLabel::B,
                    active: true,
                },
            ]
        );
    }

    #[test]
    fn active_indicator_follows_underlying_transport_through_assignment() {
        let mut session = session();
        let blind = BlindComparisonState::new(TRIAL, true);
        assert!(blind.visible_subjects(&session)[1].active);

        session.switch_to(ComparisonSide::B);
        let visible = blind.visible_subjects(&session);
        assert!(visible[0].active);
        assert!(!visible[1].active);
    }

    #[test]
    fn reveal_is_explicit_and_irreversible_on_the_state_instance() {
        let mut blind = BlindComparisonState::new(TRIAL, true);
        assert!(blind.is_blinded());

        let revealed = blind.reveal();
        assert_eq!(blind.reveal_state(), BlindRevealState::Revealed);
        assert_eq!(revealed.visible_a_side, ComparisonSide::B);
        assert_eq!(revealed.visible_b_side, ComparisonSide::A);

        // A second reveal returns the same mapping; there is intentionally no
        // `hide`/`reblind` transition after metadata may have been observed.
        assert_eq!(blind.reveal(), revealed);
        assert_eq!(blind.reveal_state(), BlindRevealState::Revealed);
    }

    #[test]
    fn assignment_policy_is_external_and_therefore_reproducible() {
        let first = BlindComparisonState::new(TRIAL, true);
        let second = BlindComparisonState::new(TRIAL, true);
        for label in [BlindLabel::A, BlindLabel::B] {
            assert_eq!(
                first.side_for_transport(label),
                second.side_for_transport(label)
            );
        }
    }
}
