// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Display-oriented projection of the facade's most recent experience-bridge cycle.
//!
//! The service daemon historically reached back into `Symthaea::last_bridge_cycle()`
//! after each successful query. That becomes impossible once the facade is handed
//! to the sole runtime owner. This module projects only the data the live UI already
//! consumes while the owner still has immutable access to the facade.
//!
//! Large diagnostic-only fields such as the cycle output vectors, thought vector,
//! wisdom hypervector, and mental-movie HDC trajectory are intentionally excluded.
//! Base64 encoding is also left to the transport edge so cognition does not spend
//! owner time performing wire-format expansion.

use symthaea::Symthaea;
use symthaea::cognitive_loop::{CycleMetadata, CycleResult};

/// Raw display frames for one geodesic mental simulation.
///
/// Frame bytes remain unencoded. HTTP/WebSocket transports may base64-encode or
/// otherwise frame them after this snapshot has left the cognition owner.
#[derive(Debug, Clone, PartialEq)]
pub struct MentalMovieTelemetry {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub semantic_coherence: f32,
    pub frames: Vec<Vec<u8>>,
}

/// Compatibility telemetry copied from one completed bridge cycle.
#[derive(Debug, Clone)]
pub struct BridgeTelemetrySnapshot {
    pub metadata: CycleMetadata,
    pub canvas_svg: Option<String>,
    pub mental_movie: Option<MentalMovieTelemetry>,
}

impl BridgeTelemetrySnapshot {
    pub fn from_cycle(cycle: &CycleResult) -> Self {
        #[cfg(feature = "canvas")]
        let canvas_svg = cycle.canvas_svg.clone();
        #[cfg(not(feature = "canvas"))]
        let canvas_svg = None;

        #[cfg(feature = "vision-manifold")]
        let mental_movie = cycle.mental_movie.as_ref().map(|movie| MentalMovieTelemetry {
            width: movie.width,
            height: movie.height,
            channels: movie.channels,
            semantic_coherence: movie.semantic_coherence,
            frames: movie.frames.clone(),
        });
        #[cfg(not(feature = "vision-manifold"))]
        let mental_movie = None;

        Self {
            metadata: cycle.metadata.clone(),
            canvas_svg,
            mental_movie,
        }
    }
}

/// Capture the most recent bridge-cycle projection, if the experience bridge has
/// actually produced one. This is called only while the sole owner holds immutable
/// access to the facade after command execution.
pub(crate) fn capture_bridge_telemetry(engine: &Symthaea) -> Option<BridgeTelemetrySnapshot> {
    engine
        .last_bridge_cycle()
        .map(BridgeTelemetrySnapshot::from_cycle)
}
