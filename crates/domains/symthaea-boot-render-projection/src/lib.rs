// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Truth-preserving projection from renderer-neutral Spore boot frames into the
//! absolute timeline expected by the exact Boot Ecology renderer.
//!
//! This crate owns no DRM, systemd, journal, input, health, or boot authority.
//! It intentionally separates two projection paths:
//!
//! - live semantic frames may enter the terminal Handoff segment only after the
//!   frame carries the protocol-derived `handoff_ready` fact;
//! - an authoritative host display-transfer request may project a bounded
//!   terminal animation without carrying or manufacturing any boot-health fact.

#![forbid(unsafe_code)]

use symthaea_boot_presentation::{EcologyFrameInput, FrameError};

/// Normalized semantic visual phase scale shared by the live presentation path.
pub const PHASE_SCALE: u32 = 1_000_000;
/// The live ecology contract reserves the final 2.5% exclusively for terminal
/// Handoff rendering after explicit Ready.
pub const PRE_HANDOFF_PHASE_MAX: u32 = 975_000;
pub const LIVE_HANDOFF_PHASE_SPAN: u32 = PHASE_SCALE - PRE_HANDOFF_PHASE_MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTimelineLayout {
    /// Exact renderer duration before its terminal Handoff stage begins.
    pub pre_handoff_ms: u32,
    /// Exact duration of the terminal Handoff stage.
    pub handoff_ms: u32,
}

impl RenderTimelineLayout {
    pub fn validate(self) -> Result<(), ProjectionError> {
        if self.pre_handoff_ms == 0 || self.handoff_ms == 0 {
            return Err(ProjectionError::ZeroDuration);
        }
        self.pre_handoff_ms
            .checked_add(self.handoff_ms)
            .ok_or(ProjectionError::TimelineOverflow)?;
        Ok(())
    }

    pub fn total_ms(self) -> Result<u32, ProjectionError> {
        self.validate()?;
        self.pre_handoff_ms
            .checked_add(self.handoff_ms)
            .ok_or(ProjectionError::TimelineOverflow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderSegment {
    PreHandoff,
    Handoff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionSource {
    LiveSemanticFrame,
    HostHandoffRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderProjection {
    /// Absolute time supplied to the exact ecology renderer.
    pub elapsed_ms: u32,
    /// Segment that the exact integer-millisecond renderer will actually see.
    /// At the exact pre-Handoff boundary the current #238 renderer still returns
    /// the preceding stage, so this is derived from projected time rather than
    /// merely from requested semantic tail progress.
    pub segment: RenderSegment,
    pub source: ProjectionSource,
}

/// Project one validated live semantic frame onto the exact renderer timeline.
///
/// The ordinary live path can consume the full pre-Handoff ecology but cannot
/// cross into the terminal segment until `handoff_ready` is explicitly true.
pub fn project_live_frame(
    frame: &EcologyFrameInput,
    layout: RenderTimelineLayout,
) -> Result<RenderProjection, ProjectionError> {
    frame.validate().map_err(ProjectionError::InvalidFrame)?;
    layout.validate()?;

    if frame.visual_phase <= PRE_HANDOFF_PHASE_MAX {
        let elapsed_ms = scale_floor(
            frame.visual_phase,
            PRE_HANDOFF_PHASE_MAX,
            layout.pre_handoff_ms,
        );
        return Ok(RenderProjection {
            elapsed_ms,
            segment: segment_for_elapsed(elapsed_ms, layout),
            source: ProjectionSource::LiveSemanticFrame,
        });
    }

    if !frame.handoff_ready {
        return Err(ProjectionError::TerminalStageRequiresReady {
            visual_phase: frame.visual_phase,
        });
    }

    let terminal_phase = frame.visual_phase - PRE_HANDOFF_PHASE_MAX;
    let handoff_elapsed = scale_floor(
        terminal_phase,
        LIVE_HANDOFF_PHASE_SPAN,
        layout.handoff_ms,
    );
    let elapsed_ms = layout
        .pre_handoff_ms
        .checked_add(handoff_elapsed)
        .ok_or(ProjectionError::TimelineOverflow)?;

    Ok(RenderProjection {
        elapsed_ms,
        segment: segment_for_elapsed(elapsed_ms, layout),
        source: ProjectionSource::LiveSemanticFrame,
    })
}

/// Project the host-owned display-transfer animation.
///
/// `handoff_phase` is a normalized local progress coordinate for the bounded
/// terminal visual only. This function deliberately accepts no health/snapshot
/// input, so host lifecycle enforcement cannot accidentally bless boot health.
pub fn project_host_handoff(
    handoff_phase: u32,
    layout: RenderTimelineLayout,
) -> Result<RenderProjection, ProjectionError> {
    layout.validate()?;
    if handoff_phase > PHASE_SCALE {
        return Err(ProjectionError::HostHandoffPhaseOutOfRange(handoff_phase));
    }

    let terminal_elapsed = scale_floor(handoff_phase, PHASE_SCALE, layout.handoff_ms);
    let elapsed_ms = layout
        .pre_handoff_ms
        .checked_add(terminal_elapsed)
        .ok_or(ProjectionError::TimelineOverflow)?;

    Ok(RenderProjection {
        elapsed_ms,
        segment: segment_for_elapsed(elapsed_ms, layout),
        source: ProjectionSource::HostHandoffRequest,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionError {
    InvalidFrame(FrameError),
    ZeroDuration,
    TimelineOverflow,
    TerminalStageRequiresReady { visual_phase: u32 },
    HostHandoffPhaseOutOfRange(u32),
}

impl std::fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFrame(error) => write!(f, "invalid ecology frame: {error}"),
            Self::ZeroDuration => {
                write!(f, "renderer timeline segments must have non-zero duration")
            }
            Self::TimelineOverflow => write!(f, "renderer timeline duration overflow"),
            Self::TerminalStageRequiresReady { visual_phase } => write!(
                f,
                "live visual phase {visual_phase} attempted terminal Handoff without Ready"
            ),
            Self::HostHandoffPhaseOutOfRange(phase) => write!(
                f,
                "host Handoff phase exceeds normalized scale: {phase} > {PHASE_SCALE}"
            ),
        }
    }
}

impl std::error::Error for ProjectionError {}

const fn segment_for_elapsed(elapsed_ms: u32, layout: RenderTimelineLayout) -> RenderSegment {
    if elapsed_ms <= layout.pre_handoff_ms {
        RenderSegment::PreHandoff
    } else {
        RenderSegment::Handoff
    }
}

const fn scale_floor(value: u32, input_max: u32, output_max: u32) -> u32 {
    (((value as u64) * (output_max as u64)) / (input_max as u64)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_boot_ecology_live::{
        DiagnosticFloor, DomainMask, LiveEcologyModulation, SemanticBootAnchor, VisualAccent,
    };
    use symthaea_boot_presentation::PresentationDriver;
    use symthaea_boot_protocol::BootHealth;
    use symthaea_boot_visual_clock::{ClockMode, VisualClockPolicy};

    const LAYOUT: RenderTimelineLayout = RenderTimelineLayout {
        pre_handoff_ms: 4_500,
        handoff_ms: 500,
    };

    fn modulation(
        anchor: SemanticBootAnchor,
        health: BootHealth,
        sequence: u64,
    ) -> LiveEcologyModulation {
        LiveEcologyModulation {
            observation_sequence: sequence,
            anchor,
            health,
            reveal_floor: anchor.reveal_floor(),
            delayed_domains: DomainMask::empty(),
            degraded_domains: DomainMask::empty(),
            failed_domains: DomainMask::empty(),
            diagnostic_floor: match health {
                BootHealth::Normal | BootHealth::Unknown => DiagnosticFloor::Ambient,
                BootHealth::Delayed => DiagnosticFloor::Status,
                BootHealth::Degraded | BootHealth::Failed => DiagnosticFloor::Diagnostics,
            },
            accent_token: 0,
            accent: VisualAccent::None,
            handoff_ready: anchor == SemanticBootAnchor::SessionReady,
        }
    }

    #[test]
    fn last_non_ready_phase_ends_exactly_at_pre_handoff_boundary() {
        let target = modulation(SemanticBootAnchor::SessionPhase, BootHealth::Normal, 9);
        let mut driver = PresentationDriver::restore(
            SemanticBootAnchor::SessionPhase,
            PRE_HANDOFF_PHASE_MAX,
            VisualClockPolicy::default(),
        )
        .unwrap();
        let frame = driver.advance_ms(0, &target).unwrap();
        let projection = project_live_frame(&frame, LAYOUT).unwrap();
        assert_eq!(projection.elapsed_ms, LAYOUT.pre_handoff_ms);
        assert_eq!(projection.segment, RenderSegment::PreHandoff);
    }

    #[test]
    fn explicit_ready_can_enter_terminal_segment_during_catch_up() {
        let target = modulation(SemanticBootAnchor::SessionReady, BootHealth::Unknown, 10);
        let mut driver = PresentationDriver::restore(
            SemanticBootAnchor::SessionPhase,
            PRE_HANDOFF_PHASE_MAX,
            VisualClockPolicy::default(),
        )
        .unwrap();
        let frame = driver.advance_ms(16, &target).unwrap();
        assert!(frame.visual_phase > PRE_HANDOFF_PHASE_MAX);
        assert!(frame.handoff_ready);
        let projection = project_live_frame(&frame, LAYOUT).unwrap();
        assert_eq!(projection.segment, RenderSegment::Handoff);
        assert_eq!(projection.source, ProjectionSource::LiveSemanticFrame);
        // Unknown may factually reach Ready/Handoff but is not upgraded to a
        // known-normal health/celebration by this projection layer.
        assert_eq!(frame.health, BootHealth::Unknown);
        assert_ne!(frame.accent, VisualAccent::Ready);
    }

    #[test]
    fn live_projection_rejects_fabricated_non_ready_terminal_phase() {
        let target = modulation(SemanticBootAnchor::SessionPhase, BootHealth::Normal, 9);
        let mut driver = PresentationDriver::restore(
            SemanticBootAnchor::SessionPhase,
            PRE_HANDOFF_PHASE_MAX,
            VisualClockPolicy::default(),
        )
        .unwrap();
        let mut frame = driver.advance_ms(0, &target).unwrap();
        frame.visual_phase = PRE_HANDOFF_PHASE_MAX + 1;
        frame.truth_ceiling = PHASE_SCALE;
        // The frame contract rejects the fabrication before the terminal gate is
        // even consulted. Defense exists at both boundaries.
        assert!(matches!(
            project_live_frame(&frame, LAYOUT),
            Err(ProjectionError::InvalidFrame(_))
        ));
    }

    #[test]
    fn host_handoff_is_independent_of_health_and_live_ready() {
        let start = project_host_handoff(0, LAYOUT).unwrap();
        let middle = project_host_handoff(PHASE_SCALE / 2, LAYOUT).unwrap();
        let end = project_host_handoff(PHASE_SCALE, LAYOUT).unwrap();

        assert_eq!(start.elapsed_ms, LAYOUT.pre_handoff_ms);
        assert_eq!(start.segment, RenderSegment::PreHandoff);
        assert_eq!(middle.elapsed_ms, 4_750);
        assert_eq!(middle.segment, RenderSegment::Handoff);
        assert_eq!(end.elapsed_ms, 5_000);
        assert_eq!(end.segment, RenderSegment::Handoff);
        assert_eq!(end.source, ProjectionSource::HostHandoffRequest);
    }

    #[test]
    fn quantized_terminal_progress_is_labeled_as_pixels_will_render() {
        // One normalized terminal unit is far below one renderer millisecond at
        // this layout. The exact renderer still sees the final pre-Handoff frame,
        // so projection metadata must say the same thing.
        let tiny = project_host_handoff(1, LAYOUT).unwrap();
        assert_eq!(tiny.elapsed_ms, LAYOUT.pre_handoff_ms);
        assert_eq!(tiny.segment, RenderSegment::PreHandoff);

        // Once projection advances by at least one actual renderer millisecond,
        // the terminal segment is genuinely visible.
        let first_visible = project_host_handoff(2_000, LAYOUT).unwrap();
        assert!(first_visible.elapsed_ms > LAYOUT.pre_handoff_ms);
        assert_eq!(first_visible.segment, RenderSegment::Handoff);
    }

    #[test]
    fn projection_is_monotonic_and_bounded_across_entire_live_scale() {
        let mut previous = 0;
        for phase in (0..=PRE_HANDOFF_PHASE_MAX).step_by(975) {
            let elapsed = scale_floor(phase, PRE_HANDOFF_PHASE_MAX, LAYOUT.pre_handoff_ms);
            assert!(elapsed >= previous);
            assert!(elapsed <= LAYOUT.pre_handoff_ms);
            previous = elapsed;
        }
        assert_eq!(previous, LAYOUT.pre_handoff_ms);
    }

    #[test]
    fn invalid_layouts_and_host_phase_fail_closed_as_presentation() {
        assert_eq!(
            RenderTimelineLayout {
                pre_handoff_ms: 0,
                handoff_ms: 500,
            }
            .validate(),
            Err(ProjectionError::ZeroDuration)
        );
        assert_eq!(
            project_host_handoff(PHASE_SCALE + 1, LAYOUT),
            Err(ProjectionError::HostHandoffPhaseOutOfRange(PHASE_SCALE + 1))
        );
    }

    #[test]
    fn complete_ready_projects_to_exact_timeline_end() {
        let target = modulation(SemanticBootAnchor::SessionReady, BootHealth::Normal, 11);
        let mut driver = PresentationDriver::restore(
            SemanticBootAnchor::SessionReady,
            PHASE_SCALE,
            VisualClockPolicy::default(),
        )
        .unwrap();
        let frame = driver.advance_ms(0, &target).unwrap();
        assert_eq!(frame.clock_mode, ClockMode::Complete);
        let projection = project_live_frame(&frame, LAYOUT).unwrap();
        assert_eq!(projection.elapsed_ms, LAYOUT.total_ms().unwrap());
        assert_eq!(projection.segment, RenderSegment::Handoff);
    }
}
