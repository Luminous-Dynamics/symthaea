// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sparse factual identity layer for the exact Spore renderer.
//!
//! The organic/holographic ecology should stand on its own. This final wrapper
//! therefore renders only a restrained `SPORE` mark plus one factual lifecycle
//! cue and one tiny current-stage label. No external font stack is required.

use crate::color::{LEAF_GREEN, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};
use crate::ecology_renderer_fidelity_v2::EcologyRenderer as FidelityEcologyRenderer;
pub use crate::ecology_renderer_fidelity_v2::EcologyFrameState;
use crate::microtype;
use symthaea_boot_ecology::{BootCue, BootGenome, BootStageKind};

const HOLO_CYAN: Rgba = Rgba(0x58, 0xd8, 0xd2, 0xff);

pub struct EcologyRenderer {
    width: usize,
    height: usize,
    cue: BootCue,
    inner: FidelityEcologyRenderer,
}

impl EcologyRenderer {
    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {
        let cue = genome.cue;
        Self {
            width: width as usize,
            height: height as usize,
            cue,
            inner: FidelityEcologyRenderer::new(width, height, genome),
        }
    }

    pub fn genome(&self) -> &BootGenome {
        self.inner.genome()
    }

    pub fn total_duration_ms(&self) -> u32 {
        self.inner.total_duration_ms()
    }

    pub fn frame_state(&self, elapsed_ms: u32) -> EcologyFrameState {
        self.inner.frame_state(elapsed_ms)
    }

    pub fn render_at(&self, elapsed_ms: u32, buffer: &mut [u32]) -> EcologyFrameState {
        let state = self.inner.render_at(elapsed_ms, buffer);
        if state.stage != BootStageKind::Blackout {
            self.draw_identity(buffer, state);
        }
        state
    }

    fn draw_identity(&self, buffer: &mut [u32], state: EcologyFrameState) {
        if self.width < 120 || self.height < 70 || buffer.len() < self.width * self.height {
            return;
        }

        let arrival = smoothstep(((state.sequence_progress - 0.06) * 5.0).clamp(0.0, 1.0));
        let departure = if state.stage == BootStageKind::Handoff {
            1.0 - smoothstep(state.stage_progress)
        } else {
            1.0
        };
        let opacity = arrival * departure;
        if opacity <= 0.01 {
            return;
        }

        let scale = (self.height / 270).clamp(1, 4);
        let small_scale = scale.max(1);
        let margin_x = (self.width / 24).max(8);
        let margin_y = (self.height / 18).max(8);
        let title_metrics = microtype::measure("SPORE", scale, scale);
        let title_y = self
            .height
            .saturating_sub(margin_y + title_metrics.height + small_scale * 12);

        let title = MYCELIAL_WHITE.with_opacity((0.82 * opacity).clamp(0.0, 0.90));
        let shadow = HOLO_CYAN.with_opacity((0.10 * opacity).clamp(0.0, 0.13));
        microtype::draw_text(
            buffer,
            self.width,
            self.height,
            margin_x + 1,
            title_y + 1,
            "SPORE",
            scale,
            scale,
            shadow,
        );
        microtype::draw_text(
            buffer,
            self.width,
            self.height,
            margin_x,
            title_y,
            "SPORE",
            scale,
            scale,
            title,
        );

        let cue = cue_label(self.cue);
        let cue_color = cue_color(self.cue).with_opacity((0.62 * opacity).clamp(0.0, 0.70));
        microtype::draw_text(
            buffer,
            self.width,
            self.height,
            margin_x,
            title_y + title_metrics.height + small_scale * 4,
            cue,
            small_scale,
            small_scale,
            cue_color,
        );

        let stage = stage_label(state.stage);
        let metrics = microtype::measure(stage, small_scale, small_scale);
        if metrics.width + margin_x < self.width {
            let stage_x = self.width.saturating_sub(margin_x + metrics.width);
            let stage_y = self.height.saturating_sub(margin_y + metrics.height);
            microtype::draw_text(
                buffer,
                self.width,
                self.height,
                stage_x,
                stage_y,
                stage,
                small_scale,
                small_scale,
                Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, 0.45)
                    .with_opacity((0.34 * opacity).clamp(0.0, 0.40)),
            );
        }
    }
}

fn cue_label(cue: BootCue) -> &'static str {
    match cue {
        BootCue::FirstBoot => "GERMINATION",
        BootCue::Starting => "SYSTEM INITIALIZING",
        BootCue::Resuming => "RELIGHTING",
        BootCue::ApplyingGeneration => "APPLYING GENERATION",
        BootCue::RestoringKnownGood => "RESTORING KNOWN GOOD",
        BootCue::RecoveringState => "RECOVERY",
    }
}

fn cue_color(cue: BootCue) -> Rgba {
    match cue {
        BootCue::FirstBoot | BootCue::Starting => LEAF_GREEN,
        BootCue::Resuming => HOLO_CYAN,
        BootCue::ApplyingGeneration | BootCue::RestoringKnownGood => SOLAR_GOLD,
        BootCue::RecoveringState => Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, 0.40),
    }
}

fn stage_label(stage: BootStageKind) -> &'static str {
    match stage {
        BootStageKind::Blackout => "",
        BootStageKind::DormantCore => "DORMANT CORE",
        BootStageKind::Relight => "RELIGHTING",
        BootStageKind::Germinate => "GERMINATING",
        BootStageKind::Grow => "WEAVING",
        BootStageKind::Anastomose => "LINKING",
        BootStageKind::Repair => "REPAIRING",
        BootStageKind::GrowthRing => "GENERATION RING",
        BootStageKind::HardwareBud => "HARDWARE CHANGE",
        BootStageKind::RetractFailedGrowth => "RESTORING",
        BootStageKind::MeshLink => "MESH RETURN",
        BootStageKind::Settle => "SETTLING",
        BootStageKind::Handoff => "HANDOFF",
    }
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_boot_ecology::{BootEcologyComposer, BootStateReceipt, MorphologyLineage};

    fn genome() -> BootGenome {
        BootEcologyComposer::compose(
            &BootStateReceipt::first_boot([0x39; 32]),
            &MorphologyLineage::default(),
        )
    }

    #[test]
    fn identity_layer_is_deterministic() {
        let a = EcologyRenderer::new(320, 180, genome());
        let b = EcologyRenderer::new(320, 180, genome());
        let mut fa = vec![0u32; 320 * 180];
        let mut fb = vec![0u32; 320 * 180];
        a.render_at(2_000, &mut fa);
        b.render_at(2_000, &mut fb);
        assert_eq!(fa, fb);
    }

    #[test]
    fn factual_labels_avoid_consciousness_claims() {
        for cue in [
            BootCue::FirstBoot,
            BootCue::Starting,
            BootCue::Resuming,
            BootCue::ApplyingGeneration,
            BootCue::RestoringKnownGood,
            BootCue::RecoveringState,
        ] {
            let label = cue_label(cue).to_ascii_lowercase();
            assert!(!label.contains("conscious"));
            assert!(!label.contains("sentient"));
            assert!(!label.contains("aware"));
        }
    }
}
