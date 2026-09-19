// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Human judgment records for blinded A/B comparison.
//!
//! A judgment is a human report, not musical ground truth. It is recorded in
//! visible blind-label space before reveal and can only be resolved to the
//! underlying comparison side after the exact same trial has been revealed.

use std::fmt;

use crate::comparison::{ComparisonSide, MusicalComparisonAnchor};
use crate::comparison_blind::{
    BlindComparisonState, BlindLabel, BlindRevealState, BlindTrialId,
};

/// Bookkeeping supplied by the comparison controller about successful visible
/// A/B auditions. This pure type does not observe browser playback itself; the
/// future browser adapter must increment it only after the corresponding
/// transport transaction has actually committed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlindListeningExposure {
    pub visible_a_auditions: u32,
    pub visible_b_auditions: u32,
}

impl BlindListeningExposure {
    pub fn record(&mut self, label: BlindLabel) {
        match label {
            BlindLabel::A => {
                self.visible_a_auditions = self.visible_a_auditions.saturating_add(1);
            }
            BlindLabel::B => {
                self.visible_b_auditions = self.visible_b_auditions.saturating_add(1);
            }
        }
    }

    pub const fn both_sides_auditioned(self) -> bool {
        self.visible_a_auditions > 0 && self.visible_b_auditions > 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlindComparisonChoice {
    Prefer(BlindLabel),
    NoPreference,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlindComparisonJudgment {
    pub trial_id: BlindTrialId,
    pub choice: BlindComparisonChoice,
    pub anchor: Option<MusicalComparisonAnchor>,
    pub exposure: BlindListeningExposure,
    /// Optional listener self-report in [0, 1]. It is not a probability and is
    /// never promoted into evidence authority by this type.
    pub self_reported_confidence: Option<f32>,
    pub note: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedComparisonChoice {
    Prefer(ComparisonSide),
    NoPreference,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedBlindComparisonJudgment {
    pub trial_id: BlindTrialId,
    pub choice: ResolvedComparisonChoice,
    pub anchor: Option<MusicalComparisonAnchor>,
    pub exposure: BlindListeningExposure,
    pub self_reported_confidence: Option<f32>,
    pub note: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlindJudgmentError {
    TrialAlreadyRevealed,
    JudgmentAlreadyRecorded,
    BothSidesMustBeAuditioned,
    InvalidConfidence,
    TrialNotRevealed,
    TrialIdMismatch,
}

impl fmt::Display for BlindJudgmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TrialAlreadyRevealed => {
                write!(f, "blind judgment must be recorded before reveal")
            }
            Self::JudgmentAlreadyRecorded => {
                write!(f, "this blind trial already has a recorded judgment")
            }
            Self::BothSidesMustBeAuditioned => {
                write!(f, "both visible sides must be auditioned before judgment")
            }
            Self::InvalidConfidence => {
                write!(f, "self-reported confidence must be finite and within [0, 1]")
            }
            Self::TrialNotRevealed => {
                write!(f, "blind judgment cannot resolve before trial reveal")
            }
            Self::TrialIdMismatch => {
                write!(f, "judgment belongs to a different blind comparison trial")
            }
        }
    }
}

impl std::error::Error for BlindJudgmentError {}

/// Record one final human judgment while the trial is still blind.
///
/// The supplied exposure record must show at least one committed audition of
/// each visible side. This pure boundary validates the record but cannot prove
/// that browser playback occurred; that stronger claim belongs to the future
/// transport/controller integration. Validation happens before the state is
/// marked, so malformed input does not consume the trial's one judgment slot.
pub fn record_blind_judgment(
    blind: &mut BlindComparisonState,
    exposure: BlindListeningExposure,
    choice: BlindComparisonChoice,
    anchor: Option<MusicalComparisonAnchor>,
    self_reported_confidence: Option<f32>,
    note: String,
) -> Result<BlindComparisonJudgment, BlindJudgmentError> {
    if blind.reveal_state() != BlindRevealState::Blinded {
        return Err(BlindJudgmentError::TrialAlreadyRevealed);
    }
    if blind.has_recorded_judgment() {
        return Err(BlindJudgmentError::JudgmentAlreadyRecorded);
    }
    if !exposure.both_sides_auditioned() {
        return Err(BlindJudgmentError::BothSidesMustBeAuditioned);
    }
    if let Some(confidence) = self_reported_confidence
        && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
    {
        return Err(BlindJudgmentError::InvalidConfidence);
    }
    if !blind.mark_judgment_recorded() {
        return Err(BlindJudgmentError::JudgmentAlreadyRecorded);
    }

    Ok(BlindComparisonJudgment {
        trial_id: blind.trial_id(),
        choice,
        anchor,
        exposure,
        self_reported_confidence,
        note,
    })
}

/// Resolve a previously recorded visible-label judgment only after reveal and
/// only against the exact trial that produced it.
pub fn resolve_blind_judgment(
    blind: &BlindComparisonState,
    judgment: &BlindComparisonJudgment,
) -> Result<ResolvedBlindComparisonJudgment, BlindJudgmentError> {
    if blind.reveal_state() != BlindRevealState::Revealed {
        return Err(BlindJudgmentError::TrialNotRevealed);
    }
    if blind.trial_id() != judgment.trial_id {
        return Err(BlindJudgmentError::TrialIdMismatch);
    }

    let choice = match judgment.choice {
        BlindComparisonChoice::Prefer(label) => {
            ResolvedComparisonChoice::Prefer(blind.side_for_transport(label))
        }
        BlindComparisonChoice::NoPreference => ResolvedComparisonChoice::NoPreference,
    };

    Ok(ResolvedBlindComparisonJudgment {
        trial_id: judgment.trial_id,
        choice,
        anchor: judgment.anchor,
        exposure: judgment.exposure,
        self_reported_confidence: judgment.self_reported_confidence,
        note: judgment.note.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIAL: BlindTrialId = BlindTrialId(44);

    fn heard_both() -> BlindListeningExposure {
        BlindListeningExposure {
            visible_a_auditions: 2,
            visible_b_auditions: 1,
        }
    }

    #[test]
    fn exposure_counts_saturate_and_require_both_visible_sides() {
        let mut exposure = BlindListeningExposure::default();
        exposure.record(BlindLabel::A);
        assert!(!exposure.both_sides_auditioned());
        exposure.record(BlindLabel::B);
        assert!(exposure.both_sides_auditioned());

        exposure.visible_a_auditions = u32::MAX;
        exposure.record(BlindLabel::A);
        assert_eq!(exposure.visible_a_auditions, u32::MAX);
    }

    #[test]
    fn judgment_requires_both_sides_before_reveal() {
        let mut blind = BlindComparisonState::new(TRIAL, false);
        let result = record_blind_judgment(
            &mut blind,
            BlindListeningExposure {
                visible_a_auditions: 1,
                visible_b_auditions: 0,
            },
            BlindComparisonChoice::Prefer(BlindLabel::A),
            None,
            None,
            String::new(),
        );
        assert_eq!(result, Err(BlindJudgmentError::BothSidesMustBeAuditioned));
        assert!(!blind.has_recorded_judgment());
    }

    #[test]
    fn malformed_confidence_does_not_consume_judgment_slot() {
        let mut blind = BlindComparisonState::new(TRIAL, false);
        for confidence in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert_eq!(
                record_blind_judgment(
                    &mut blind,
                    heard_both(),
                    BlindComparisonChoice::NoPreference,
                    None,
                    Some(confidence),
                    String::new(),
                ),
                Err(BlindJudgmentError::InvalidConfidence)
            );
            assert!(!blind.has_recorded_judgment());
        }
    }

    #[test]
    fn only_one_blind_judgment_can_be_recorded_per_state_instance() {
        let mut blind = BlindComparisonState::new(TRIAL, false);
        let first = record_blind_judgment(
            &mut blind,
            heard_both(),
            BlindComparisonChoice::Prefer(BlindLabel::A),
            MusicalComparisonAnchor::new(2, 1.0).ok(),
            Some(0.8),
            "A felt clearer here".into(),
        )
        .unwrap();
        assert_eq!(first.trial_id, TRIAL);
        assert!(blind.has_recorded_judgment());

        assert_eq!(
            record_blind_judgment(
                &mut blind,
                heard_both(),
                BlindComparisonChoice::NoPreference,
                None,
                None,
                String::new(),
            ),
            Err(BlindJudgmentError::JudgmentAlreadyRecorded)
        );
    }

    #[test]
    fn revealing_before_judgment_blocks_blind_recording() {
        let mut blind = BlindComparisonState::new(TRIAL, false);
        blind.reveal();
        assert_eq!(
            record_blind_judgment(
                &mut blind,
                heard_both(),
                BlindComparisonChoice::Prefer(BlindLabel::B),
                None,
                None,
                String::new(),
            ),
            Err(BlindJudgmentError::TrialAlreadyRevealed)
        );
    }

    #[test]
    fn judgment_resolves_through_the_exact_revealed_assignment() {
        let mut blind = BlindComparisonState::new(TRIAL, true);
        let judgment = record_blind_judgment(
            &mut blind,
            heard_both(),
            BlindComparisonChoice::Prefer(BlindLabel::A),
            None,
            Some(0.75),
            "blind note".into(),
        )
        .unwrap();

        assert_eq!(
            resolve_blind_judgment(&blind, &judgment),
            Err(BlindJudgmentError::TrialNotRevealed)
        );

        blind.reveal();
        let resolved = resolve_blind_judgment(&blind, &judgment).unwrap();
        assert_eq!(
            resolved.choice,
            ResolvedComparisonChoice::Prefer(ComparisonSide::B)
        );
        assert_eq!(resolved.note, "blind note");
    }

    #[test]
    fn judgment_cannot_be_resolved_against_another_trial() {
        let mut first = BlindComparisonState::new(TRIAL, false);
        let judgment = record_blind_judgment(
            &mut first,
            heard_both(),
            BlindComparisonChoice::NoPreference,
            None,
            None,
            String::new(),
        )
        .unwrap();
        first.reveal();

        let mut other = BlindComparisonState::new(BlindTrialId(45), false);
        other.reveal();
        assert_eq!(
            resolve_blind_judgment(&other, &judgment),
            Err(BlindJudgmentError::TrialIdMismatch)
        );
    }
}
