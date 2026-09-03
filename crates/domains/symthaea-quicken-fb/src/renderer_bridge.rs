// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Low-trust semantic bridge from validated boot truth to the fallible ecology renderer.
//!
//! This module deliberately owns no DRM, login/session authority, telemetry socket,
//! journal access, or host lifecycle decision. A caller supplies an already-validated
//! `BootSnapshot`; any reducer, clock, projection, or renderer failure is converted
//! into a visual fallback request rather than propagated into the boot control path.

use std::panic::{AssertUnwindSafe, catch_unwind};

use symthaea_boot_ecology::{
    BootEcologyComposer, BootGenome, BootStageKind, BootStateReceipt, GenerationHealth,
    GenerationTransition, MorphologyLineage, PreviousTermination, StorageState,
};
use symthaea_boot_ecology_live::LiveEcologyReducer;
use symthaea_boot_presentation::{FrameDigest, PresentationDriver};
use symthaea_boot_protocol::{BootHealth, BootSnapshot};
use symthaea_boot_render_projection::{
    RenderProjection, RenderTimelineLayout, project_live_frame,
};
use symthaea_boot_visual_clock::VisualClockPolicy;

use crate::ecology_renderer_identity::{EcologyFrameState, EcologyRenderer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcologyBridgeInitError {
    ZeroExtent,
    MissingTerminalHandoff,
    MultipleTerminalHandoffs,
    StageAfterTerminalHandoff,
    TimelineOverflow,
    InvalidTimeline,
    PresentationDriver,
    RendererPanicked,
}

impl std::fmt::Display for EcologyBridgeInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroExtent => write!(f, "ecology renderer requires non-zero dimensions"),
            Self::MissingTerminalHandoff => write!(f, "ecology genome has no terminal Handoff stage"),
            Self::MultipleTerminalHandoffs => write!(f, "ecology genome has multiple Handoff stages"),
            Self::StageAfterTerminalHandoff => write!(f, "ecology genome contains a stage after Handoff"),
            Self::TimelineOverflow => write!(f, "ecology renderer timeline overflow"),
            Self::InvalidTimeline => write!(f, "ecology renderer timeline is invalid"),
            Self::PresentationDriver => write!(f, "ecology presentation driver initialization failed"),
            Self::RendererPanicked => write!(f, "ecology renderer initialization panicked"),
        }
    }
}

impl std::error::Error for EcologyBridgeInitError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcologyFallbackReason {
    SnapshotRejected,
    PresentationRejected,
    ProjectionRejected,
    ProjectionRegressed,
    BufferTooSmall,
    RendererPanicked,
}

impl EcologyFallbackReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SnapshotRejected => "snapshot-rejected",
            Self::PresentationRejected => "presentation-rejected",
            Self::ProjectionRejected => "projection-rejected",
            Self::ProjectionRegressed => "projection-regressed",
            Self::BufferTooSmall => "buffer-too-small",
            Self::RendererPanicked => "renderer-panicked",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderedEcologyFrame {
    pub observation_sequence: u64,
    pub health: BootHealth,
    pub semantic_digest: FrameDigest,
    pub visual_phase: u32,
    pub handoff_ready: bool,
    pub projection: RenderProjection,
    pub renderer_state: EcologyFrameState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EcologyRenderOutcome {
    Rendered(RenderedEcologyFrame),
    Fallback(EcologyFallbackReason),
}

/// Stateful composition of the existing semantic boot crates and the exact final
/// CPU ecology renderer chain. One instance belongs to one observation lineage.
pub struct EcologyRendererBridge {
    width: u32,
    height: u32,
    reducer: LiveEcologyReducer,
    presentation: PresentationDriver,
    renderer: EcologyRenderer,
    layout: RenderTimelineLayout,
    last_projected_ms: u32,
}

impl EcologyRendererBridge {
    pub fn new(
        width: u32,
        height: u32,
        genesis_phrase: &str,
    ) -> Result<Self, EcologyBridgeInitError> {
        if width == 0 || height == 0 {
            return Err(EcologyBridgeInitError::ZeroExtent);
        }

        let genome = compose_neutral_genome(genesis_phrase);
        let layout = timeline_layout(&genome)?;
        layout
            .validate()
            .map_err(|_| EcologyBridgeInitError::InvalidTimeline)?;
        let presentation = PresentationDriver::new(VisualClockPolicy::default())
            .map_err(|_| EcologyBridgeInitError::PresentationDriver)?;
        let renderer = catch_unwind(AssertUnwindSafe(|| {
            EcologyRenderer::new(width, height, genome)
        }))
        .map_err(|_| EcologyBridgeInitError::RendererPanicked)?;

        Ok(Self {
            width,
            height,
            reducer: LiveEcologyReducer::new(),
            presentation,
            renderer,
            layout,
            last_projected_ms: 0,
        })
    }

    /// Reset only semantic lineage state. The deterministic visual genome remains
    /// stable so a telemetry lineage change cannot itself mutate visual identity.
    pub fn reset_semantics(&mut self) {
        self.reducer.reset();
        self.presentation.reset();
        self.last_projected_ms = 0;
    }

    pub const fn layout(&self) -> RenderTimelineLayout {
        self.layout
    }

    /// Consume one authoritative snapshot and attempt one ecology frame.
    ///
    /// No error escapes this function. The caller must render `MycelialNetwork`
    /// whenever `Fallback` is returned.
    pub fn render_snapshot(
        &mut self,
        snapshot: &BootSnapshot,
        elapsed_ms: u32,
        buffer: &mut [u32],
    ) -> EcologyRenderOutcome {
        let modulation = match self.reducer.reduce(snapshot) {
            Ok(modulation) => modulation,
            Err(_) => return EcologyRenderOutcome::Fallback(EcologyFallbackReason::SnapshotRejected),
        };
        let frame = match self.presentation.advance_ms(elapsed_ms, &modulation) {
            Ok(frame) => frame,
            Err(_) => {
                return EcologyRenderOutcome::Fallback(EcologyFallbackReason::PresentationRejected);
            }
        };
        let projection = match project_live_frame(&frame, self.layout) {
            Ok(projection) => projection,
            Err(_) => return EcologyRenderOutcome::Fallback(EcologyFallbackReason::ProjectionRejected),
        };

        if projection.elapsed_ms < self.last_projected_ms {
            return EcologyRenderOutcome::Fallback(EcologyFallbackReason::ProjectionRegressed);
        }
        let required = (self.width as usize).saturating_mul(self.height as usize);
        if buffer.len() < required {
            return EcologyRenderOutcome::Fallback(EcologyFallbackReason::BufferTooSmall);
        }

        let renderer_state = match catch_unwind(AssertUnwindSafe(|| {
            self.renderer.render_at(projection.elapsed_ms, buffer)
        })) {
            Ok(state) => state,
            Err(_) => return EcologyRenderOutcome::Fallback(EcologyFallbackReason::RendererPanicked),
        };
        self.last_projected_ms = projection.elapsed_ms;

        EcologyRenderOutcome::Rendered(RenderedEcologyFrame {
            observation_sequence: frame.observation_sequence,
            health: frame.health,
            semantic_digest: frame.semantic_digest(),
            visual_phase: frame.visual_phase,
            handoff_ready: frame.handoff_ready,
            projection,
            renderer_state,
        })
    }
}

fn compose_neutral_genome(genesis_phrase: &str) -> BootGenome {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"quicken-fb-ecology-visual-seed-v1\0");
    hasher.update(genesis_phrase.as_bytes());
    let seed = *hasher.finalize().as_bytes();

    // The quicken executable does not yet receive the persisted previous-boot
    // ecology receipt. Use explicit Unknown facts rather than falsely claiming
    // that every invocation is a first boot. The stable visual seed remains the
    // only input derived from the genesis phrase.
    let mut receipt = BootStateReceipt::first_boot(seed);
    receipt.previous_termination = PreviousTermination::Unknown;
    receipt.generation_transition = GenerationTransition::Unknown;
    receipt.generation_health = GenerationHealth::Unknown;
    receipt.storage_state = StorageState::Unknown;
    BootEcologyComposer::compose(&receipt, &MorphologyLineage::default())
}

fn timeline_layout(genome: &BootGenome) -> Result<RenderTimelineLayout, EcologyBridgeInitError> {
    let mut pre_handoff_ms = 0u32;
    let mut handoff_ms = None;

    for stage in &genome.stages {
        if stage.kind == BootStageKind::Handoff {
            if handoff_ms.is_some() {
                return Err(EcologyBridgeInitError::MultipleTerminalHandoffs);
            }
            handoff_ms = Some(stage.duration_ms);
            continue;
        }
        if handoff_ms.is_some() {
            return Err(EcologyBridgeInitError::StageAfterTerminalHandoff);
        }
        pre_handoff_ms = pre_handoff_ms
            .checked_add(stage.duration_ms)
            .ok_or(EcologyBridgeInitError::TimelineOverflow)?;
    }

    let handoff_ms = handoff_ms.ok_or(EcologyBridgeInitError::MissingTerminalHandoff)?;
    let layout = RenderTimelineLayout {
        pre_handoff_ms,
        handoff_ms,
    };
    layout
        .total_ms()
        .map_err(|_| EcologyBridgeInitError::InvalidTimeline)?;
    Ok(layout)
}
