#!/usr/bin/env python3
"""Second guarded pass for Spore v0.3.3 exact renderer semantics.

This runs only after ``spore_v033_apply_v2.py``. Every substitution targets the
post-v2 source exactly once, so drift fails before any source file is accepted.
The pass remains presentation-only: it wires semantic-progress choreography and
localized hardware-bud geometry into existing renderer layers.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one guarded match, found {count}")
    target.write_text(text.replace(old, new, 1))


BASE = "crates/domains/symthaea-quicken-fb/src/ecology_renderer.rs"
HOLO = "crates/domains/symthaea-quicken-fb/src/ecology_renderer_holo.rs"
FIDELITY = "crates/domains/symthaea-quicken-fb/src/ecology_renderer_fidelity_v2.rs"

# ---------------------------------------------------------------------------
# Base organism: derive motion from semantic stage progress, not global elapsed
# sequence time, and realize HardwareBud as one localized structural addition.
# ---------------------------------------------------------------------------
replace_once(
    BASE,
    "use crate::color::{BLACK, LEAF_GREEN, MOSS_DEEP, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};\nuse crate::visual_sampling::node_visible;",
    "use crate::color::{BLACK, LEAF_GREEN, MOSS_DEEP, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};\nuse crate::hardware_bud::HardwareBudPlan;\nuse crate::temporal_choreography::TemporalChoreography;\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};\nuse crate::visual_sampling::node_visible;",
)

replace_once(
    BASE,
    "        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n\n        if VisualCompositionBudget::should_render(budget.topology) {\n            self.draw_spores(buffer, state, budget.topology);\n            self.draw_curves(buffer, state, budget.topology);\n        }\n\n        if matches!(state.stage, BootStageKind::GrowthRing)\n            && VisualCompositionBudget::should_render(budget.accent)\n        {\n            self.draw_growth_ring(buffer, state, budget.accent);\n        }\n        if matches!(\n            state.stage,\n            BootStageKind::MeshLink | BootStageKind::Settle | BootStageKind::Handoff\n        ) && VisualCompositionBudget::should_render(budget.mesh)\n        {\n            self.draw_mesh_links(buffer, state, budget.mesh);\n        }",
    "        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n        let choreography =\n            TemporalChoreography::derive(state.stage, state.stage_progress, self.profile);\n\n        if VisualCompositionBudget::should_render(budget.topology) {\n            self.draw_spores(buffer, state, budget.topology, choreography);\n            self.draw_curves(buffer, state, budget.topology, choreography);\n        }\n\n        if matches!(state.stage, BootStageKind::GrowthRing)\n            && VisualCompositionBudget::should_render(budget.accent)\n        {\n            self.draw_growth_ring(buffer, budget.accent, choreography);\n        }\n        if matches!(state.stage, BootStageKind::HardwareBud)\n            && VisualCompositionBudget::should_render(budget.accent)\n        {\n            self.draw_hardware_bud(buffer, state, budget.accent, choreography);\n        }\n        if matches!(\n            state.stage,\n            BootStageKind::MeshLink | BootStageKind::Settle | BootStageKind::Handoff\n        ) && VisualCompositionBudget::should_render(budget.mesh)\n        {\n            self.draw_mesh_links(buffer, state, budget.mesh, choreography);\n        }",
)

replace_once(
    BASE,
    "    fn draw_spores(&self, buffer: &mut [u32], state: EcologyFrameState, opacity: f32) {\n        let pulse = 0.5 + 0.5 * (state.sequence_progress * std::f32::consts::TAU * 2.0).sin();\n        for spore in &self.spores {\n            let local = 0.5 + 0.5 * (state.sequence_progress * 8.0 + spore.phase).sin();",
    "    fn draw_spores(\n        &self,\n        buffer: &mut [u32],\n        _state: EcologyFrameState,\n        opacity: f32,\n        choreography: TemporalChoreography,\n    ) {\n        let phase = choreography.ambient_phase * std::f32::consts::TAU;\n        let pulse = 0.5 + 0.5 * choreography.ambient_gain * (phase * 2.0).sin();\n        for spore in &self.spores {\n            let local =\n                0.5 + 0.5 * choreography.ambient_gain * (phase * 1.25 + spore.phase).sin();",
)

replace_once(
    BASE,
    "    fn draw_curves(&self, buffer: &mut [u32], state: EcologyFrameState, opacity: f32) {",
    "    fn draw_curves(\n        &self,\n        buffer: &mut [u32],\n        state: EcologyFrameState,\n        opacity: f32,\n        choreography: TemporalChoreography,\n    ) {",
)

replace_once(
    BASE,
    "            let pulse = 0.5\n                + 0.5\n                    * (state.sequence_progress * self.genome.parameters.pulse_velocity * 18.0\n                        + curve.node_phase)\n                        .sin();",
    "            let pulse_phase = choreography.ambient_phase\n                * std::f32::consts::TAU\n                * self.genome.parameters.pulse_velocity.max(0.05);\n            let pulse = 0.5\n                + 0.5\n                    * choreography.ambient_gain\n                    * (pulse_phase + curve.node_phase).sin();",
)

replace_once(
    BASE,
    "    fn draw_growth_ring(&self, buffer: &mut [u32], state: EcologyFrameState, opacity: f32) {\n        if self.spores.is_empty() {\n            return;\n        }\n        let center = self.spores[0].center;\n        let min_dim = self.width.min(self.height) as f32;\n        let radius = min_dim * (0.08 + 0.34 * smoothstep(state.stage_progress));\n        let color = SOLAR_GOLD.with_opacity((0.60 * (1.0 - state.stage_progress) * opacity).clamp(0.0, 1.0));",
    "    fn draw_growth_ring(\n        &self,\n        buffer: &mut [u32],\n        opacity: f32,\n        choreography: TemporalChoreography,\n    ) {\n        if self.spores.is_empty() {\n            return;\n        }\n        let center = self.spores[0].center;\n        let min_dim = self.width.min(self.height) as f32;\n        let radius = min_dim * (0.08 + 0.34 * choreography.narrative_phase);\n        let color = SOLAR_GOLD.with_opacity(\n            (0.60 * choreography.hero_envelope * opacity).clamp(0.0, 1.0),\n        );",
)

replace_once(
    BASE,
    "    fn draw_mesh_links(&self, buffer: &mut [u32], state: EcologyFrameState, opacity: f32) {",
    "    fn draw_mesh_links(\n        &self,\n        buffer: &mut [u32],\n        state: EcologyFrameState,\n        opacity: f32,\n        choreography: TemporalChoreography,\n    ) {",
)

replace_once(
    BASE,
    "        for (a, b, phase) in &self.mesh_links {\n            let pulse = 0.55 + 0.45 * (state.sequence_progress * 16.0 + phase).sin();",
    "        for (a, b, phase) in &self.mesh_links {\n            let pulse = 0.55\n                + 0.45\n                    * choreography.ambient_gain\n                    * (choreography.ambient_phase * std::f32::consts::TAU * 2.0 + phase).sin();",
)

replace_once(
    BASE,
    "    fn draw_growth_ring(\n",
    "    fn draw_hardware_bud(\n        &self,\n        buffer: &mut [u32],\n        state: EcologyFrameState,\n        opacity: f32,\n        choreography: TemporalChoreography,\n    ) {\n        let plan = HardwareBudPlan::derive(&self.genome.seed, state.stage_progress);\n        if !plan.should_render() {\n            return;\n        }\n\n        let min_dim = self.width.min(self.height) as f32;\n        let center = self.spores.first().map_or(\n            Point {\n                x: self.width as f32 * 0.5,\n                y: self.height as f32 * 0.48,\n            },\n            |spore| spore.center,\n        );\n        let target = Point {\n            x: center.x + plan.anchor_angle.cos() * plan.anchor_radius * min_dim,\n            y: center.y + plan.anchor_angle.sin() * plan.anchor_radius * min_dim,\n        };\n\n        // Attach to the nearest existing endpoint so the event reads as new\n        // anatomy of the established organism, not an unrelated floating icon.\n        let mut anchor = center;\n        let mut nearest = f32::INFINITY;\n        for curve in &self.curves {\n            let distance = curve.end.distance(target);\n            if distance < nearest {\n                nearest = distance;\n                anchor = curve.end;\n            }\n        }\n\n        let stem_length = plan.stem_length * min_dim * plan.growth;\n        let stem_end = Point {\n            x: anchor.x + plan.anchor_angle.cos() * stem_length,\n            y: anchor.y + plan.anchor_angle.sin() * stem_length,\n        };\n        let structural = Rgba::lerp(LEAF_GREEN, MYCELIAL_WHITE, 0.46)\n            .with_opacity((0.82 * opacity).clamp(0.0, 1.0));\n        draw_soft_line(\n            buffer,\n            self.width as usize,\n            self.height as usize,\n            anchor,\n            stem_end,\n            1.35,\n            structural,\n            1,\n        );\n\n        // The stem establishes the persistent change first; small child branches\n        // follow locally. Gold is reserved for the event junction/tips rather\n        // than washing the entire organism in an alert color.\n        let branch_growth = ((plan.growth - 0.38) / 0.62).clamp(0.0, 1.0);\n        let branch_length =\n            plan.stem_length * plan.branch_length_scale * min_dim * branch_growth;\n        for index in 0..plan.branch_count {\n            let angle = plan.branch_angle(index);\n            let end = Point {\n                x: stem_end.x + angle.cos() * branch_length,\n                y: stem_end.y + angle.sin() * branch_length,\n            };\n            draw_soft_line(\n                buffer,\n                self.width as usize,\n                self.height as usize,\n                stem_end,\n                end,\n                1.05,\n                structural.with_opacity((0.72 * opacity).clamp(0.0, 1.0)),\n                1,\n            );\n            if branch_growth > 0.82 {\n                let tip = Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, 0.28).with_opacity(\n                    (0.48 * opacity * choreography.hero_envelope).clamp(0.0, 1.0),\n                );\n                draw_glow_circle(\n                    buffer,\n                    self.width as usize,\n                    self.height as usize,\n                    end.x,\n                    end.y,\n                    1.2,\n                    tip,\n                    self.genome.parameters.glow_radius * 0.55,\n                );\n            }\n        }\n\n        let junction = Rgba::lerp(SOLAR_GOLD, MYCELIAL_WHITE, 0.36).with_opacity(\n            (0.42 * opacity * choreography.hero_envelope).clamp(0.0, 1.0),\n        );\n        draw_glow_circle(\n            buffer,\n            self.width as usize,\n            self.height as usize,\n            stem_end.x,\n            stem_end.y,\n            1.4,\n            junction,\n            self.genome.parameters.glow_radius * 0.65,\n        );\n    }\n\n    fn draw_growth_ring(\n",
)

# ---------------------------------------------------------------------------
# Holographic field: its breathing, anchors and sweep follow the same semantic
# choreography. Secondary field motion can therefore never outrun boot truth.
# ---------------------------------------------------------------------------
replace_once(
    HOLO,
    "pub use crate::ecology_renderer_base::EcologyFrameState;\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};",
    "pub use crate::ecology_renderer_base::EcologyFrameState;\nuse crate::temporal_choreography::TemporalChoreography;\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};",
)

replace_once(
    HOLO,
    "        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n        if state.stage != BootStageKind::Blackout\n            && VisualCompositionBudget::should_render(budget.holography)\n        {\n            self.field\n                .render(buffer, state, budget.holography, budget.accent);\n        }",
    "        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n        let choreography =\n            TemporalChoreography::derive(state.stage, state.stage_progress, self.profile);\n        if state.stage != BootStageKind::Blackout\n            && VisualCompositionBudget::should_render(budget.holography)\n        {\n            self.field.render(\n                buffer,\n                state,\n                budget.holography,\n                budget.accent,\n                choreography,\n            );\n        }",
)

replace_once(
    HOLO,
    "        accent_gain: f32,\n    ) {",
    "        accent_gain: f32,\n        choreography: TemporalChoreography,\n    ) {",
)

replace_once(
    HOLO,
    "        let arrival = smoothstep((state.sequence_progress * 4.5).clamp(0.0, 1.0));\n        let gain = self.field_strength * arrival * holography_gain;\n        if gain <= 0.001 {\n            return;\n        }\n\n        let phase = state.sequence_progress * std::f32::consts::TAU * 1.8;",
    "        let arrival = smoothstep((state.sequence_progress * 4.5).clamp(0.0, 1.0));\n        let gain = self.field_strength * arrival * holography_gain;\n        if gain <= 0.001 {\n            return;\n        }\n\n        let phase = choreography.ambient_phase\n            * std::f32::consts::TAU\n            * (0.35 + choreography.ambient_gain * 1.45);",
)

replace_once(
    HOLO,
    "        self.draw_anchor_field(buffer, state, gain, primary);\n        self.draw_energy_sweep(buffer, state, gain * accent_gain, primary);\n        self.draw_scanline_sheen(buffer, phase, gain);",
    "        self.draw_anchor_field(buffer, gain, primary, choreography);\n        self.draw_energy_sweep(\n            buffer,\n            state,\n            gain * accent_gain,\n            primary,\n            choreography,\n        );\n        self.draw_scanline_sheen(buffer, phase, gain);",
)

replace_once(
    HOLO,
    "        state: EcologyFrameState,\n        gain: f32,\n        primary: Rgba,\n    ) {\n        let pulse_phase = state.sequence_progress * 18.0;",
    "        gain: f32,\n        primary: Rgba,\n        choreography: TemporalChoreography,\n    ) {\n        let pulse_phase = choreography.ambient_phase * std::f32::consts::TAU * 2.8;",
)

replace_once(
    HOLO,
    "        primary: Rgba,\n    ) {\n        let min_dim = self.width.min(self.height) as f32;\n        let cycle = (state.sequence_progress * 2.25).fract();\n        let radius = min_dim * (0.10 + smoothstep(cycle) * 0.38);\n        let fade = (1.0 - cycle).powf(1.7);",
    "        primary: Rgba,\n        choreography: TemporalChoreography,\n    ) {\n        let min_dim = self.width.min(self.height) as f32;\n        let cycle = choreography.narrative_phase;\n        let radius = min_dim * (0.10 + cycle * 0.38);\n        let fade = choreography.hero_envelope;",
)

# ---------------------------------------------------------------------------
# Membrane/caustics: keep their existing spatial grammar but drive drift from
# semantic choreography rather than whole-sequence time.
# ---------------------------------------------------------------------------
replace_once(
    FIDELITY,
    "pub use crate::ecology_renderer_holo::EcologyFrameState;\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};",
    "pub use crate::ecology_renderer_holo::EcologyFrameState;\nuse crate::temporal_choreography::TemporalChoreography;\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};",
)

replace_once(
    FIDELITY,
    "        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n        self.fidelity\n            .render(buffer, state, budget.membrane, budget.caustics);",
    "        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n        let choreography =\n            TemporalChoreography::derive(state.stage, state.stage_progress, self.profile);\n        self.fidelity.render(\n            buffer,\n            state,\n            budget.membrane,\n            budget.caustics,\n            choreography,\n        );",
)

replace_once(
    FIDELITY,
    "        caustic_gain: f32,\n    ) {",
    "        caustic_gain: f32,\n        choreography: TemporalChoreography,\n    ) {",
)

replace_once(
    FIDELITY,
    "            self.draw_membrane(buffer, state, membrane_gain);\n        }\n        if VisualCompositionBudget::should_render(caustic_gain) {\n            self.draw_caustics(buffer, state, caustic_gain);",
    "            self.draw_membrane(buffer, state, membrane_gain, choreography);\n        }\n        if VisualCompositionBudget::should_render(caustic_gain) {\n            self.draw_caustics(buffer, state, caustic_gain, choreography);",
)

replace_once(
    FIDELITY,
    "    fn draw_membrane(&self, buffer: &mut [u32], state: EcologyFrameState, gain: f32) {\n        let time = state.sequence_progress * std::f32::consts::TAU * 1.35 + self.shell_phase;\n        let radius = self.shell_radius * (1.0 + 0.045 * time.sin());",
    "    fn draw_membrane(\n        &self,\n        buffer: &mut [u32],\n        state: EcologyFrameState,\n        gain: f32,\n        choreography: TemporalChoreography,\n    ) {\n        let time = choreography.ambient_phase * std::f32::consts::TAU * 1.35\n            + self.shell_phase;\n        let radius = self.shell_radius\n            * (1.0 + 0.045 * choreography.ambient_gain * time.sin());",
)

replace_once(
    FIDELITY,
    "    fn draw_caustics(&self, buffer: &mut [u32], state: EcologyFrameState, gain: f32) {\n        let min_dim = self.width.min(self.height) as f32;\n        let phase = state.sequence_progress * std::f32::consts::TAU + self.shell_phase;",
    "    fn draw_caustics(\n        &self,\n        buffer: &mut [u32],\n        state: EcologyFrameState,\n        gain: f32,\n        choreography: TemporalChoreography,\n    ) {\n        let min_dim = self.width.min(self.height) as f32;\n        let phase = choreography.ambient_phase * std::f32::consts::TAU + self.shell_phase;",
)

print("Spore v0.3.3 semantic choreography + HardwareBud integration applied")
