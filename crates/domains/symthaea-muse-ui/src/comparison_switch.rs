// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure planning boundary between semantic comparison state and future browser
//! transport effects.
//!
//! CMP-001 owns A/B intent and CMP-002 resolves a musical anchor against one
//! subject's authoritative timeline. This module combines those facts into a
//! switch plan but deliberately performs no media load, seek, play, state
//! mutation, or canonical-piece change. The browser adapter comes later.

use std::fmt;

use crate::comparison::{
    ComparisonSession, ComparisonSide, ComparisonTransportIntent, MusicalComparisonAnchor,
};
use crate::comparison_timeline::ResolvedComparisonAnchor;
use crate::playback::PlaybackSource;

const ANCHOR_EPS: f64 = 1.0e-7;

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonSwitchPlan {
    pub from_side: ComparisonSide,
    pub to_side: ComparisonSide,
    pub source: PlaybackSource,
    pub target_seconds: f64,
    pub play_after_seek: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonSwitchPlanError {
    AlreadyActive,
    MissingMusicalAnchor,
    StaleResolvedAnchor,
    InvalidResolvedSeconds,
}

impl fmt::Display for ComparisonSwitchPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyActive => write!(f, "comparison target is already active"),
            Self::MissingMusicalAnchor => {
                write!(f, "comparison session has no musical anchor to preserve")
            }
            Self::StaleResolvedAnchor => write!(
                f,
                "resolved target does not match the comparison session's current musical anchor"
            ),
            Self::InvalidResolvedSeconds => {
                write!(f, "resolved target seconds must be finite and non-negative")
            }
        }
    }
}

impl std::error::Error for ComparisonSwitchPlanError {}

/// Plan a switch to `to_side` using a target already resolved against that
/// subject's authoritative timeline.
///
/// The returned plan is immutable evidence for the later browser adapter. This
/// function does not update `session.active_side`: state should change only once
/// the transport adapter has accepted the load/seek operation under its own
/// epoch discipline.
pub fn plan_comparison_switch(
    session: &ComparisonSession,
    to_side: ComparisonSide,
    resolved_target: ResolvedComparisonAnchor,
) -> Result<ComparisonSwitchPlan, ComparisonSwitchPlanError> {
    if to_side == session.active_side {
        return Err(ComparisonSwitchPlanError::AlreadyActive);
    }

    let session_anchor = session
        .musical_anchor
        .ok_or(ComparisonSwitchPlanError::MissingMusicalAnchor)?;
    if !same_anchor(session_anchor, resolved_target) {
        return Err(ComparisonSwitchPlanError::StaleResolvedAnchor);
    }
    if !resolved_target.seconds.is_finite() || resolved_target.seconds < 0.0 {
        return Err(ComparisonSwitchPlanError::InvalidResolvedSeconds);
    }

    Ok(ComparisonSwitchPlan {
        from_side: session.active_side,
        to_side,
        source: session.subject(to_side).source.clone(),
        target_seconds: resolved_target.seconds,
        play_after_seek: session.transport_intent == ComparisonTransportIntent::Playing,
    })
}

fn same_anchor(anchor: MusicalComparisonAnchor, resolved: ResolvedComparisonAnchor) -> bool {
    anchor.bar_index == resolved.bar_index
        && (anchor.beat_offset - resolved.beat_offset).abs() <= ANCHOR_EPS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::{ComparisonSubject, ComparisonTransportIntent};
    use crate::playback::{PlaybackPresentation, PlaybackSubjectKind};

    fn source(url: &str) -> PlaybackSource {
        PlaybackSource {
            rendition_id: None,
            audio_url: url.to_string(),
            duration_hint_seconds: Some(30.0),
            advance_on_end: false,
            presentation: PlaybackPresentation {
                kind: PlaybackSubjectKind::Review,
                title: url.to_string(),
                subtitle: None,
                style_hint: None,
            },
        }
    }

    fn session() -> ComparisonSession {
        let a = ComparisonSubject::new(source("/audio/a"), None).unwrap();
        let b = ComparisonSubject::new(source("/audio/b"), None).unwrap();
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(3, 1.5).unwrap()));
        session
    }

    fn resolved(bar_index: u32, beat_offset: f64, seconds: f64) -> ResolvedComparisonAnchor {
        ResolvedComparisonAnchor {
            bar_index,
            beat_offset,
            absolute_beats: 13.5,
            seconds,
            meter_numerator: 4,
            meter_denominator: 4,
            tempo_bpm: 120.0,
        }
    }

    #[test]
    fn paused_session_plans_exact_target_without_mutating_active_side() {
        let session = session();
        let plan = plan_comparison_switch(
            &session,
            ComparisonSide::B,
            resolved(3, 1.5, 6.75),
        )
        .unwrap();

        assert_eq!(plan.from_side, ComparisonSide::A);
        assert_eq!(plan.to_side, ComparisonSide::B);
        assert_eq!(plan.source.audio_url, "/audio/b");
        assert_eq!(plan.target_seconds, 6.75);
        assert!(!plan.play_after_seek);
        assert_eq!(session.active_side, ComparisonSide::A);
    }

    #[test]
    fn playing_intent_is_carried_as_resume_after_seek_not_an_immediate_effect() {
        let mut session = session();
        session.set_transport_intent(ComparisonTransportIntent::Playing);
        let plan = plan_comparison_switch(
            &session,
            ComparisonSide::B,
            resolved(3, 1.5, 9.25),
        )
        .unwrap();

        assert!(plan.play_after_seek);
        assert_eq!(session.active_side, ComparisonSide::A);
    }

    #[test]
    fn resolved_anchor_from_an_old_session_position_is_rejected() {
        let session = session();
        assert_eq!(
            plan_comparison_switch(
                &session,
                ComparisonSide::B,
                resolved(2, 1.5, 4.0),
            ),
            Err(ComparisonSwitchPlanError::StaleResolvedAnchor)
        );
        assert_eq!(
            plan_comparison_switch(
                &session,
                ComparisonSide::B,
                resolved(3, 1.0, 4.0),
            ),
            Err(ComparisonSwitchPlanError::StaleResolvedAnchor)
        );
    }

    #[test]
    fn missing_anchor_refuses_to_guess_a_normalized_or_seconds_position() {
        let mut session = session();
        session.set_musical_anchor(None);
        assert_eq!(
            plan_comparison_switch(
                &session,
                ComparisonSide::B,
                resolved(0, 0.0, 0.0),
            ),
            Err(ComparisonSwitchPlanError::MissingMusicalAnchor)
        );
    }

    #[test]
    fn already_active_side_is_a_noop_error_not_a_reload() {
        let session = session();
        assert_eq!(
            plan_comparison_switch(
                &session,
                ComparisonSide::A,
                resolved(3, 1.5, 6.75),
            ),
            Err(ComparisonSwitchPlanError::AlreadyActive)
        );
    }

    #[test]
    fn invalid_seconds_are_rejected_even_if_a_resolver_object_is_forged() {
        let session = session();
        assert_eq!(
            plan_comparison_switch(
                &session,
                ComparisonSide::B,
                resolved(3, 1.5, f64::NAN),
            ),
            Err(ComparisonSwitchPlanError::InvalidResolvedSeconds)
        );
        assert_eq!(
            plan_comparison_switch(
                &session,
                ComparisonSide::B,
                resolved(3, 1.5, -1.0),
            ),
            Err(ComparisonSwitchPlanError::InvalidResolvedSeconds)
        );
    }
}
