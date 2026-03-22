// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vision & Sensory Manager — groups vision bridge, foveation, virtual body,
//! and coherence field into a single CLS field.

/// Consolidated manager for vision and sensory subsystems.
///
/// Replaces 6 top-level CLS fields:
/// - `coherence_field` — physiology coherence via hormone modulation
/// - `virtual_body` — embodied cognition adapter
/// - `vision_bridge` — frame → attention-boosted HDC encoding (vision-manifold)
/// - `vision_frame_buffer` — latest frame buffer (vision-manifold)
/// - `cross_manifold_predictor` — vision→cognitive Hebbian mapping (vision-manifold)
/// - `foveation_manager` — dorsal surprise → ventral recognition (foveation)
pub struct VisionAndSensoryManager {
    /// Physiology coherence field — consciousness integration via hormone modulation.
    pub coherence_field: Option<crate::physiology::CoherenceField>,

    /// Virtual body adapter for embodied cognition.
    pub(crate) virtual_body: Option<super::virtual_body::VirtualBody>,

    /// Vision bridge: frame → attention-boosted HDC encoding.
    #[cfg(feature = "vision-manifold")]
    pub vision_bridge: Option<symthaea_vision_manifold::VisionBridge>,

    /// Latest frame buffer for vision processing.
    #[cfg(feature = "vision-manifold")]
    pub vision_frame_buffer: Option<Vec<u8>>,

    /// Cross-manifold predictor: vision→cognitive Hebbian mapping.
    #[cfg(feature = "vision-manifold")]
    pub cross_manifold_predictor: Option<symthaea_vision_manifold::CrossManifoldPredictor>,

    /// Foveation bridge: dorsal surprise → ventral recognition dispatch.
    #[cfg(feature = "foveation")]
    pub foveation_manager: Option<std::sync::Mutex<symthaea_foveation::FoveationManager>>,
}

impl std::fmt::Debug for VisionAndSensoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionAndSensoryManager")
            .field("coherence_field", &self.coherence_field.is_some())
            .field("virtual_body", &self.virtual_body.is_some())
            .finish_non_exhaustive()
    }
}

impl VisionAndSensoryManager {
    /// Reset sensory state for a new episode.
    /// Vision/foveation retain learned models; coherence field and virtual body reset.
    pub fn reset(&mut self) {
        if let Some(ref mut cf) = self.coherence_field {
            *cf = crate::physiology::CoherenceField::new();
        }
        // Virtual body has no reset — interoceptive state is ephemeral per-cycle.
        // Vision bridge, foveation, and cross-manifold predictor retain learned state.
    }
}
