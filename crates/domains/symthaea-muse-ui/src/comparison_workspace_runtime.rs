// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scoped runtime for one evidence-qualified blind A/B comparison workspace.
//!
//! The runtime owns comparison-only state (session, blind assignment, duration
//! qualification, browser-event router, and optional human judgment) while the
//! application's single `PlaybackState` remains external. It exposes no mutable
//! `ComparisonSession`, so the subjects and musical anchor bound to one blind
//! trial cannot be silently replaced halfway through evidence collection.

use std::fmt;

use crate::comparison::{
    ComparisonSession, ComparisonSide, ComparisonTransportIntent, MusicalComparisonAnchor,
};
use crate::comparison_blind::{BlindAssignment, BlindLabel, BlindTrialId, RevealedBlindAssignment};
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
};
use crate::comparison_switch::{
    ComparisonSwitchPlanError, plan_comparison_switch,
};
use crate::comparison_timeline::ResolvedComparisonAnchor;
use crate::playback::{PlaybackEffect, PlaybackState};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlindComparisonWorkspaceStatus {
    pub trial_id: BlindTrialId,
    pub anchor: MusicalComparisonAnchor,
    pub transport_intent: ComparisonTransportIntent,
    pub switch_in_flight: bool,
    pub duration_evidence: BlindDurationEvidence,
    pub qualifies_for_judgment: bool,
    pub judgment_recorded: bool,
    pub revealed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlindComparisonRevealReport {
    pub assignment: RevealedBlindAssignment,
    pub identity_diff: ComparisonIdentityDiff,
    pub judgment: Option<ResolvedBlindComparisonJudgment>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonWorkspaceError {
    MissingMusicalAnchor,
    SwitchInFlight,
    TrialAlreadyRevealed,
    JudgmentAlreadyRecorded,
    JudgmentRequiredBeforeReveal,
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
                "comparison workspace cannot change transport intent or start another action while a switch is in flight"
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

/// One blind comparison trial. The anchor present at construction is immutable
/// for the life of this runtime. A new anchor means a new trial.
#[derive(Debug, PartialEq)]
pub struct BlindComparisonWorkspaceRuntime {
    session: ComparisonSession,
    assignment: BlindAssignment,
    blind: DurationQualifiedBlindController,
    browser: ComparisonBrowserRouter,
    trial_anchor: MusicalComparisonAnchor,
    judgment: Option<BlindComparisonJudgment>,
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
            revealed: false,
        })
    }

    pub fn status(&self) -> BlindComparisonWorkspaceStatus {
        BlindComparisonWorkspaceStatus {
            trial_id: self.blind.trial_id(),
            anchor: self.trial_anchor,
            transport_intent: self.session.transport_intent,
            switch_in_flight: self.browser.has_active_switch(),
            duration_evidence: self.blind.duration_evidence(),
            qualifies_for_judgment: self.blind.qualifies_for_judgment(),
            judgment_recorded: self.judgment.is_some(),
            revealed: self.revealed,
        }
    }

    /// Change play/pause intent only between semantic switches. The trial anchor
    /// and subjects remain immutable.
    pub fn set_transport_intent(
        &mut self,
        intent: ComparisonTransportIntent,
    ) -> Result<(), ComparisonWorkspaceError> {
        if self.browser.has_active_switch() {
            return Err(ComparisonWorkspaceError::SwitchInFlight);
        }
        self.session.set_transport_intent(intent);
        Ok(())
    }

    /// Begin a switch using a target already resolved against the visible
    /// label's underlying subject timeline. This runtime verifies anchor
    /// freshness through `plan_comparison_switch`; provenance of the resolved
    /// timeline itself remains the caller's responsibility until durable keeper
    /// semantic bundles are available for both sides.
    pub fn begin_visible_switch(
        &mut self,
        visible_label: BlindLabel,
        resolved_target: ResolvedComparisonAnchor,
        playback: &mut PlaybackState,
    ) -> Result<Vec<PlaybackEffect>, ComparisonWorkspaceError> {
        if self.revealed {
            return Err(ComparisonWorkspaceError::TrialAlreadyRevealed);
        }
        let target_side = self.assignment.side_for_label(visible_label);
        let plan = plan_comparison_switch(&self.session, target_side, resolved_target)?;
        Ok(self
            .browser
            .begin_switch(&self.session, playback, plan)?
            .effects)
    }

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

    /// Store the trial's single final human judgment. The lower-level duration
    /// controller enforces the configured per-side progress threshold first.
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
        self.judgment = Some(judgment);
        Ok(self.judgment.as_ref().expect("judgment was just stored"))
    }

    /// Reveal only after a final human response exists. `NoPreference` is the
    /// explicit neutral response; reveal without any response is not silently
    /// treated as no preference because those are different research events.
    pub fn reveal(
        &mut self,
    ) -> Result<BlindComparisonRevealReport, ComparisonWorkspaceError> {
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
        let assignment = self.blind.reveal(&self.session)?;
        let resolved = self.blind.resolve_judgment(&self.session, judgment)?;
        let identity_diff = compare_subject_identity(&self.session.a, &self.session.b);
        self.revealed = true;
        Ok(BlindComparisonRevealReport {
            assignment,
            identity_diff,
            judgment: Some(resolved),
        })
    }

    /// Underlying subject details remain unavailable through this runtime until
    /// reveal. The returned sides are deliberately read-only.
    pub fn revealed_subject(
        &self,
        side: ComparisonSide,
    ) -> Option<&crate::comparison::ComparisonSubject> {
        self.revealed.then(|| self.session.subject(side))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::{ComparisonSubject, ComparisonTransportIntent};
    use crate::comparison_browser_events::ComparisonBrowserEvent;
    use crate::comparison_duration_exposure::BlindExposurePolicy;
    use crate::comparison_timeline::ResolvedComparisonAnchor;
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
        let source = runtime.session.active_subject().source.clone();
        let _ = playback.reduce(PlaybackEvent::LoadRequested {
            source,
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

    fn start_and_progress(
        runtime: &mut BlindComparisonWorkspaceRuntime,
        playback: &mut PlaybackState,
        start: f64,
        end: f64,
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
        runtime
            .handle_browser_event(
                playback,
                ComparisonBrowserEvent::TimeAdvanced {
                    load_epoch: epoch,
                    seconds: start,
                },
            )
            .unwrap();
        runtime
            .handle_browser_event(
                playback,
                ComparisonBrowserEvent::TimeAdvanced {
                    load_epoch: epoch,
                    seconds: end,
                },
            )
            .unwrap();
    }

    #[test]
    fn runtime_requires_and_freezes_the_trial_anchor() {
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

        let runtime = runtime(false);
        assert_eq!(runtime.status().anchor, MusicalComparisonAnchor::new(2, 0.5).unwrap());
        // There is intentionally no public mutable session/anchor API on the
        // runtime; a different anchor requires a new runtime/trial.
    }

    #[test]
    fn visible_label_mapping_selects_the_hidden_underlying_target() {
        let mut runtime = runtime(true);
        let mut playback = initial_playback(&runtime);
        let effects = runtime
            .begin_visible_switch(BlindLabel::A, resolved(5.0), &mut playback)
            .unwrap();
        assert_eq!(effects.len(), 1);
        // Swapped assignment maps visible A to underlying B. The underlying
        // side is not exposed by the public switch API.
        assert!(runtime.status().switch_in_flight);
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
    fn reveal_requires_explicit_final_response_not_implicit_no_preference() {
        let mut runtime = runtime(false);
        assert_eq!(
            runtime.reveal(),
            Err(ComparisonWorkspaceError::JudgmentRequiredBeforeReveal)
        );
    }

    #[test]
    fn subject_metadata_is_withheld_until_reveal() {
        let runtime = runtime(false);
        assert!(runtime.revealed_subject(ComparisonSide::A).is_none());
        assert!(runtime.revealed_subject(ComparisonSide::B).is_none());
    }

    #[test]
    fn qualified_trial_records_judgment_and_reveals_identity_only_afterward() {
        let mut runtime = runtime(false);
        let mut playback = initial_playback(&runtime);
        start_and_progress(&mut runtime, &mut playback, 0.25, 0.5);
        // Need another contiguous 0.25s sample to reach the 0.5s policy.
        let epoch = playback.load_epoch;
        runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::TimeAdvanced {
                    load_epoch: epoch,
                    seconds: 0.75,
                },
            )
            .unwrap();

        let effects = runtime
            .begin_visible_switch(BlindLabel::B, resolved(5.0), &mut playback)
            .unwrap();
        assert_eq!(effects.len(), 1);
        let switch_epoch = playback.load_epoch;
        runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: switch_epoch,
                    duration_seconds: 30.0,
                },
            )
            .unwrap();
        runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::SeekCompleted {
                    load_epoch: switch_epoch,
                    seconds: 5.0,
                },
            )
            .unwrap();
        runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::PlaybackStarted {
                    load_epoch: switch_epoch,
                },
            )
            .unwrap();
        runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::TimeAdvanced {
                    load_epoch: switch_epoch,
                    seconds: 5.25,
                },
            )
            .unwrap();
        runtime
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::TimeAdvanced {
                    load_epoch: switch_epoch,
                    seconds: 5.5,
                },
            )
            .unwrap();

        assert!(runtime.status().qualifies_for_judgment);
        runtime
            .record_judgment(
                BlindComparisonChoice::NoPreference,
                Some(0.5),
                "no preference under the fixed anchor".into(),
            )
            .unwrap();
        assert!(runtime.revealed_subject(ComparisonSide::A).is_none());

        let report = runtime.reveal().unwrap();
        assert_eq!(
            report.identity_diff.relation,
            crate::comparison_identity::ComparisonIdentityRelation::DifferentComposition
        );
        assert!(matches!(
            report.judgment.as_ref().map(|value| value.choice),
            Some(crate::comparison_judgment::ResolvedComparisonChoice::NoPreference)
        ));
        assert!(runtime.revealed_subject(ComparisonSide::A).is_some());
        assert!(runtime.status().revealed);
    }
}
