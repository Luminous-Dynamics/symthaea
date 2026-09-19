// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scoped runtime for one evidence-qualified blind A/B comparison workspace.
//!
//! Comparison-only state lives here; the application's single `PlaybackState`
//! remains external and authoritative for media transport. The runtime exposes
//! only blind-label presentation before reveal, owns a fixed musical anchor for
//! the whole trial, and freezes deliberate comparison actions once the final
//! human judgment is recorded.

use std::fmt;

use crate::comparison::{
    ComparisonSession, ComparisonSide, ComparisonTransportIntent, MusicalComparisonAnchor,
};
use crate::comparison_blind::{
    BlindAssignment, BlindLabel, BlindSubjectView, BlindTrialId, RevealedBlindAssignment,
};
use crate::comparison_browser_events::{
    ComparisonBrowserError, ComparisonBrowserEvent, ComparisonBrowserRouter,
    ComparisonBrowserUpdate,
};
use crate::comparison_duration_exposure::{
    BlindDurationEvidence, BlindExposureError, BlindExposurePolicy,
    DurationQualifiedBlindController,
};
use crate::comparison_identity::{ComparisonIdentityDiff, compare_subject_identity};
use crate::comparison_judgment::{
    BlindComparisonChoice, BlindComparisonJudgment, ResolvedBlindComparisonJudgment,
    ResolvedComparisonChoice,
};
use crate::comparison_switch::{ComparisonSwitchPlanError, plan_comparison_switch};
use crate::comparison_timeline::ResolvedComparisonAnchor;
use crate::playback::{PlaybackEffect, PlaybackState};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlindComparisonWorkspaceStatus {
    pub trial_id: BlindTrialId,
    pub anchor: MusicalComparisonAnchor,
    pub active_label: BlindLabel,
    pub transport_intent: ComparisonTransportIntent,
    pub switch_in_flight: bool,
    pub exposure_policy: BlindExposurePolicy,
    /// Frozen at judgment once a final response exists; before that this is the
    /// live duration-qualified evidence.
    pub duration_evidence: BlindDurationEvidence,
    pub qualifies_for_judgment: bool,
    pub judgment_recorded: bool,
    pub revealed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlindComparisonRevealReport {
    pub assignment: RevealedBlindAssignment,
    pub identity_diff: ComparisonIdentityDiff,
    pub judgment: ResolvedBlindComparisonJudgment,
    /// Duration-qualified progress exactly as it stood when the judgment was
    /// accepted. Later browser callbacks cannot rewrite this evidence snapshot.
    pub duration_evidence_at_judgment: BlindDurationEvidence,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonWorkspaceError {
    MissingMusicalAnchor,
    SwitchInFlight,
    TrialFinalized,
    TrialAlreadyRevealed,
    JudgmentAlreadyRecorded,
    JudgmentRequiredBeforeReveal,
    JudgmentTrialMismatch,
    SwitchPlan(ComparisonSwitchPlanError),
    Browser(ComparisonBrowserError),
    Exposure(BlindExposureError),
}

impl fmt::Display for ComparisonWorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMusicalAnchor => {
                write!(f, "blind comparison workspace requires a fixed musical anchor")
            }
            Self::SwitchInFlight => write!(
                f,
                "comparison workspace cannot perform this action while a switch is in flight"
            ),
            Self::TrialFinalized => write!(
                f,
                "the final blind judgment is already recorded; only reveal remains as a comparison action"
            ),
            Self::TrialAlreadyRevealed => {
                write!(f, "this blind comparison trial has already been revealed")
            }
            Self::JudgmentAlreadyRecorded => {
                write!(f, "this blind comparison workspace already owns its final judgment")
            }
            Self::JudgmentRequiredBeforeReveal => write!(
                f,
                "blind comparison reveal requires a recorded judgment; use an explicit no-preference judgment when appropriate"
            ),
            Self::JudgmentTrialMismatch => {
                write!(f, "stored judgment does not belong to this blind comparison trial")
            }
            Self::SwitchPlan(error) => error.fmt(f),
            Self::Browser(error) => error.fmt(f),
            Self::Exposure(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ComparisonWorkspaceError {}

impl From<ComparisonSwitchPlanError> for ComparisonWorkspaceError {
    fn from(value: ComparisonSwitchPlanError) -> Self {
        Self::SwitchPlan(value)
    }
}

impl From<ComparisonBrowserError> for ComparisonWorkspaceError {
    fn from(value: ComparisonBrowserError) -> Self {
        Self::Browser(value)
    }
}

impl From<BlindExposureError> for ComparisonWorkspaceError {
    fn from(value: BlindExposureError) -> Self {
        Self::Exposure(value)
    }
}

/// One blind comparison trial. Subjects and the musical anchor are private and
/// immutable for the lifetime of the trial; a new anchor means a new runtime.
#[derive(Debug, PartialEq)]
pub struct BlindComparisonWorkspaceRuntime {
    session: ComparisonSession,
    assignment: BlindAssignment,
    blind: DurationQualifiedBlindController,
    browser: ComparisonBrowserRouter,
    trial_anchor: MusicalComparisonAnchor,
    judgment: Option<BlindComparisonJudgment>,
    duration_evidence_at_judgment: Option<BlindDurationEvidence>,
    revealed: bool,
}

impl BlindComparisonWorkspaceRuntime {
    pub fn new(
        session: ComparisonSession,
        trial_id: BlindTrialId,
        swapped: bool,
        exposure_policy: BlindExposurePolicy,
    ) -> Result<Self, ComparisonWorkspaceError> {
        let trial_anchor = session
            .musical_anchor
            .ok_or(ComparisonWorkspaceError::MissingMusicalAnchor)?;
        let blind = DurationQualifiedBlindController::new(
            trial_id,
            &session,
            swapped,
            exposure_policy,
        )?;
        Ok(Self {
            session,
            assignment: BlindAssignment::new(swapped),
            blind,
            browser: ComparisonBrowserRouter::default(),
            trial_anchor,
            judgment: None,
            duration_evidence_at_judgment: None,
            revealed: false,
        })
    }

    pub fn status(&self) -> BlindComparisonWorkspaceStatus {
        let duration_evidence = self
            .duration_evidence_at_judgment
            .unwrap_or_else(|| self.blind.duration_evidence());
        BlindComparisonWorkspaceStatus {
            trial_id: self.blind.trial_id(),
            anchor: self.trial_anchor,
            active_label: self.assignment.label_for_side(self.session.active_side),
            transport_intent: self.session.transport_intent,
            switch_in_flight: self.browser.has_active_switch(),
            exposure_policy: self.blind.policy(),
            duration_evidence,
            qualifies_for_judgment: self.blind.qualifies_for_judgment(),
            judgment_recorded: self.judgment.is_some(),
            revealed: self.revealed,
        }
    }

    /// Metadata-free blind presentation for the two subjects.
    pub fn visible_subjects(&self) -> [BlindSubjectView; 2] {
        [BlindLabel::A, BlindLabel::B].map(|label| BlindSubjectView {
            label,
            active: self.assignment.side_for_label(label) == self.session.active_side,
        })
    }

    /// Change play/pause intent only before the final judgment and between
    /// semantic switches. Subjects and anchor remain immutable.
    pub fn set_transport_intent(
        &mut self,
        intent: ComparisonTransportIntent,
    ) -> Result<(), ComparisonWorkspaceError> {
        self.require_open_trial()?;
        if self.browser.has_active_switch() {
            return Err(ComparisonWorkspaceError::SwitchInFlight);
        }
        self.session.set_transport_intent(intent);
        Ok(())
    }

    /// Begin a switch to a visible blind label using a target already resolved
    /// against that subject's authoritative timeline.
    pub fn begin_visible_switch(
        &mut self,
        visible_label: BlindLabel,
        resolved_target: ResolvedComparisonAnchor,
        playback: &mut PlaybackState,
    ) -> Result<Vec<PlaybackEffect>, ComparisonWorkspaceError> {
        self.require_open_trial()?;
        let target_side = self.assignment.side_for_label(visible_label);
        let plan = plan_comparison_switch(&self.session, target_side, resolved_target)?;
        Ok(self
            .browser
            .begin_switch(&self.session, playback, plan)?
            .effects)
    }

    /// Route browser callbacks through the comparison-local transaction adapter.
    ///
    /// Browser callbacks may still arrive after the final judgment while the
    /// audio element settles. They remain admissible so global playback state
    /// stays truthful, but the judgment and its duration evidence are already
    /// frozen and cannot be rewritten.
    pub fn handle_browser_event(
        &mut self,
        playback: &mut PlaybackState,
        event: ComparisonBrowserEvent,
    ) -> Result<ComparisonBrowserUpdate, ComparisonWorkspaceError> {
        if self.revealed {
            return Err(ComparisonWorkspaceError::TrialAlreadyRevealed);
        }
        Ok(self
            .browser
            .handle(&mut self.session, &mut self.blind, playback, event)?)
    }

    /// Store the trial's single final human report. Qualification is checked by
    /// the duration controller first. Successful recording freezes deliberate
    /// comparison actions and snapshots the qualified duration evidence.
    pub fn record_judgment(
        &mut self,
        choice: BlindComparisonChoice,
        self_reported_confidence: Option<f32>,
        note: String,
    ) -> Result<&BlindComparisonJudgment, ComparisonWorkspaceError> {
        if self.revealed {
            return Err(ComparisonWorkspaceError::TrialAlreadyRevealed);
        }
        if self.browser.has_active_switch() {
            return Err(ComparisonWorkspaceError::SwitchInFlight);
        }
        if self.judgment.is_some() {
            return Err(ComparisonWorkspaceError::JudgmentAlreadyRecorded);
        }
        let judgment = self.blind.record_judgment(
            &self.session,
            choice,
            Some(self.trial_anchor),
            self_reported_confidence,
            note,
        )?;
        self.duration_evidence_at_judgment = Some(self.blind.duration_evidence());
        self.judgment = Some(judgment);
        Ok(self.judgment.as_ref().expect("judgment was just stored"))
    }

    /// Reveal exactly once after an explicit final response. Resolution is
    /// prepared from the private assignment before the irreversible reveal so
    /// all fallible workspace validation occurs first.
    pub fn reveal(&mut self) -> Result<BlindComparisonRevealReport, ComparisonWorkspaceError> {
        if self.revealed {
            return Err(ComparisonWorkspaceError::TrialAlreadyRevealed);
        }
        if self.browser.has_active_switch() {
            return Err(ComparisonWorkspaceError::SwitchInFlight);
        }
        let judgment = self
            .judgment
            .as_ref()
            .ok_or(ComparisonWorkspaceError::JudgmentRequiredBeforeReveal)?;
        if judgment.trial_id != self.blind.trial_id() {
            return Err(ComparisonWorkspaceError::JudgmentTrialMismatch);
        }
        let duration_evidence_at_judgment = self
            .duration_evidence_at_judgment
            .ok_or(ComparisonWorkspaceError::JudgmentRequiredBeforeReveal)?;
        let resolved = self.resolve_judgment(judgment);
        let identity_diff = compare_subject_identity(&self.session.a, &self.session.b);

        // Last fallible step before the wrapper marks itself revealed. The
        // session's subjects are private and immutable, so binding verification
        // cannot be invalidated by callers between judgment and reveal.
        let assignment = self.blind.reveal(&self.session)?;
        self.revealed = true;
        Ok(BlindComparisonRevealReport {
            assignment,
            identity_diff,
            judgment: resolved,
            duration_evidence_at_judgment,
        })
    }

    /// Underlying subject details remain unavailable through this runtime until
    /// reveal. Returned subjects are read-only.
    pub fn revealed_subject(
        &self,
        side: ComparisonSide,
    ) -> Option<&crate::comparison::ComparisonSubject> {
        self.revealed.then(|| self.session.subject(side))
    }

    fn require_open_trial(&self) -> Result<(), ComparisonWorkspaceError> {
        if self.revealed {
            return Err(ComparisonWorkspaceError::TrialAlreadyRevealed);
        }
        if self.judgment.is_some() {
            return Err(ComparisonWorkspaceError::TrialFinalized);
        }
        Ok(())
    }

    fn resolve_judgment(&self, judgment: &BlindComparisonJudgment) -> ResolvedBlindComparisonJudgment {
        let choice = match judgment.choice {
            BlindComparisonChoice::Prefer(label) => {
                ResolvedComparisonChoice::Prefer(self.assignment.side_for_label(label))
            }
            BlindComparisonChoice::NoPreference => ResolvedComparisonChoice::NoPreference,
        };
        ResolvedBlindComparisonJudgment {
            trial_id: judgment.trial_id,
            choice,
            anchor: judgment.anchor,
            exposure: judgment.exposure,
            self_reported_confidence: judgment.self_reported_confidence,
            note: judgment.note.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::ComparisonSubject;
    use crate::playback::{
        PlaybackEvent, PlaybackPresentation, PlaybackSource, PlaybackSubjectKind,
    };
    use symthaea_muse_protocol::{
        ArtifactIdentity, CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

    fn identity(score: char, composition: char, rendition: char) -> ArtifactIdentity {
        ArtifactIdentity {
            score_content: ScoreContentArtifactId(score.to_string().repeat(64)),
            composition: CompositionArtifactId(composition.to_string().repeat(64)),
            rendition: RenditionArtifactId(rendition.to_string().repeat(64)),
        }
    }

    fn source(name: &str, rendition: char) -> PlaybackSource {
        PlaybackSource {
            rendition_id: Some(RenditionArtifactId(rendition.to_string().repeat(64))),
            audio_url: format!("/audio/{name}"),
            duration_hint_seconds: Some(30.0),
            advance_on_end: false,
            presentation: PlaybackPresentation {
                kind: PlaybackSubjectKind::Review,
                title: name.to_string(),
                subtitle: None,
                style_hint: None,
            },
        }
    }

    fn session() -> ComparisonSession {
        let a = ComparisonSubject::new(source("a", 'c'), Some(identity('a', 'b', 'c'))).unwrap();
        let b = ComparisonSubject::new(source("b", 'f'), Some(identity('d', 'e', 'f'))).unwrap();
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(2, 0.5).unwrap()));
        session.set_transport_intent(ComparisonTransportIntent::Playing);
        session
    }

    fn runtime(swapped: bool) -> BlindComparisonWorkspaceRuntime {
        BlindComparisonWorkspaceRuntime::new(
            session(),
            BlindTrialId(202),
            swapped,
            BlindExposurePolicy::new(0.5, 0.6).unwrap(),
        )
        .unwrap()
    }

    fn initial_playback(runtime: &BlindComparisonWorkspaceRuntime) -> PlaybackState {
        let mut playback = PlaybackState::default();
        let _ = playback.reduce(PlaybackEvent::LoadRequested {
            source: runtime.session.active_subject().source.clone(),
            autoplay: false,
        });
        playback
    }

    fn resolved(seconds: f64) -> ResolvedComparisonAnchor {
        ResolvedComparisonAnchor {
            bar_index: 2,
            beat_offset: 0.5,
            absolute_beats: 8.5,
            seconds,
            meter_numerator: 4,
            meter_denominator: 4,
            tempo_bpm: 120.0,
        }
    }

    fn start_active_side(
        runtime: &mut BlindComparisonWorkspaceRuntime,
        playback: &mut PlaybackState,
    ) {
        let epoch = playback.load_epoch;
        runtime
            .handle_browser_event(
                playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: epoch,
                    duration_seconds: 30.0,
                },
            )
            .unwrap();
        runtime
            .handle_browser_event(
                playback,
                ComparisonBrowserEvent::PlaybackStarted { load_epoch: epoch },
            )
            .unwrap();
    }

    fn progress(
        runtime: &mut BlindComparisonWorkspaceRuntime,
        playback: &mut PlaybackState,
        seconds: f64,
    ) {
        let epoch = playback.load_epoch;
        runtime
            .handle_browser_event(
                playback,
                ComparisonBrowserEvent::TimeAdvanced {
                    load_epoch: epoch,
                    seconds,
                },
            )
            .unwrap();
    }

    #[test]
    fn runtime_requires_fixed_anchor_and_exposes_only_blind_labels() {
        let mut missing = session();
        missing.set_musical_anchor(None);
        assert_eq!(
            BlindComparisonWorkspaceRuntime::new(
                missing,
                BlindTrialId(1),
                false,
                BlindExposurePolicy::new(0.5, 0.6).unwrap(),
            ),
            Err(ComparisonWorkspaceError::MissingMusicalAnchor)
        );

        let runtime = runtime(true);
        assert_eq!(runtime.status().anchor, MusicalComparisonAnchor::new(2, 0.5).unwrap());
        assert_eq!(runtime.status().active_label, BlindLabel::B);
        assert_eq!(
            runtime.visible_subjects(),
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
        assert!(runtime.revealed_subject(ComparisonSide::A).is_none());
    }

    #[test]
    fn transport_intent_cannot_change_mid_switch() {
        let mut runtime = runtime(false);
        let mut playback = initial_playback(&runtime);
        runtime
            .begin_visible_switch(BlindLabel::B, resolved(5.0), &mut playback)
            .unwrap();
        assert_eq!(
            runtime.set_transport_intent(ComparisonTransportIntent::Paused),
            Err(ComparisonWorkspaceError::SwitchInFlight)
        );
    }

    #[test]
    fn reveal_requires_explicit_final_response() {
        let mut runtime = runtime(false);
        assert_eq!(
            runtime.reveal(),
            Err(ComparisonWorkspaceError::JudgmentRequiredBeforeReveal)
        );
    }

    #[test]
    fn qualified_judgment_freezes_deliberate_trial_actions_then_reveals() {
        let mut runtime = runtime(false);
        let mut playback = initial_playback(&runtime);
        start_active_side(&mut runtime, &mut playback);
        progress(&mut runtime, &mut playback, 0.25);
        progress(&mut runtime, &mut playback, 0.5);

        let effects = runtime
            .begin_visible_switch(BlindLabel::B, resolved(5.0), &mut playback)
            .unwrap();
        assert_eq!(effects.len(), 1);
        let epoch = playback.load_epoch;
        let metadata = runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: epoch,
                    duration_seconds: 30.0,
                },
            )
            .unwrap();
        assert_eq!(metadata.effects.len(), 1);
        let seeked = runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::SeekCompleted {
                    load_epoch: epoch,
                    seconds: 5.0,
                },
            )
            .unwrap();
        assert_eq!(seeked.effects.len(), 1);
        runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::PlaybackStarted { load_epoch: epoch },
            )
            .unwrap();
        progress(&mut runtime, &mut playback, 5.25);
        progress(&mut runtime, &mut playback, 5.5);

        assert!(runtime.status().qualifies_for_judgment);
        runtime
            .record_judgment(
                BlindComparisonChoice::NoPreference,
                Some(0.5),
                "no preference under the fixed anchor".into(),
            )
            .unwrap();
        let frozen = runtime.status().duration_evidence;

        assert_eq!(
            runtime.set_transport_intent(ComparisonTransportIntent::Paused),
            Err(ComparisonWorkspaceError::TrialFinalized)
        );
        assert_eq!(
            runtime.begin_visible_switch(BlindLabel::A, resolved(5.0), &mut playback),
            Err(ComparisonWorkspaceError::TrialFinalized)
        );
        assert!(runtime.revealed_subject(ComparisonSide::A).is_none());

        // Late browser progress can keep global transport state truthful, but
        // it cannot rewrite the duration snapshot attached to the final report.
        progress(&mut runtime, &mut playback, 5.6);
        assert_eq!(runtime.status().duration_evidence, frozen);

        let report = runtime.reveal().unwrap();
        assert_eq!(
            report.identity_diff.relation,
            crate::comparison_identity::ComparisonIdentityRelation::DifferentComposition
        );
        assert_eq!(
            report.judgment.choice,
            ResolvedComparisonChoice::NoPreference
        );
        assert_eq!(report.duration_evidence_at_judgment, frozen);
        assert!(runtime.revealed_subject(ComparisonSide::A).is_some());
        assert!(runtime.status().revealed);
    }
}