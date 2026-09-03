#!/usr/bin/env python3
"""One-shot guarded integration for Spore visual composition v0.3.3.

This script is intentionally mechanical: every substitution must match exactly
once or the run fails before writing the affected file.
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
IDENTITY = "crates/domains/symthaea-quicken-fb/src/ecology_renderer_identity.rs"

# ---------------------------------------------------------------------------
# Base ecology: expose semantic stage intensity, own the selected profile,
# apply topology/accent/mesh budgets, and realize deterministic node density.
# ---------------------------------------------------------------------------
replace_once(
    BASE,
    "use crate::color::{BLACK, LEAF_GREEN, MOSS_DEEP, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};\nuse rand::rngs::StdRng;",
    "use crate::color::{BLACK, LEAF_GREEN, MOSS_DEEP, MYCELIAL_WHITE, Rgba, SOLAR_GOLD};\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};\nuse crate::visual_sampling::node_visible;\nuse rand::rngs::StdRng;",
)

replace_once(
    BASE,
    "pub struct EcologyFrameState {\n    pub stage: BootStageKind,\n    pub stage_progress: f32,\n    pub sequence_progress: f32,\n    pub visible_fraction: f32,\n}",
    "pub struct EcologyFrameState {\n    pub stage: BootStageKind,\n    pub stage_progress: f32,\n    pub stage_intensity: f32,\n    pub sequence_progress: f32,\n    pub visible_fraction: f32,\n}",
)

replace_once(
    BASE,
    "    genome: BootGenome,\n    curves: Vec<Curve>,",
    "    genome: BootGenome,\n    profile: VisualProfile,\n    curves: Vec<Curve>,",
)

replace_once(
    BASE,
    "impl EcologyRenderer {\n    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {\n        let total_duration_ms = genome.visual_budget_ms().max(1);\n        let mut rng = StdRng::from_seed(genome.seed);\n        let spores = build_spores(width, height, &genome, &mut rng);\n        let curves = build_topology(width, height, &genome, &spores, &mut rng);\n        let mesh_links = build_mesh_links(&genome, &spores, &curves, &mut rng);\n        Self {\n            width,\n            height,\n            genome,\n            curves,\n            spores,\n            mesh_links,\n            total_duration_ms,\n        }\n    }",
    "impl EcologyRenderer {\n    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {\n        Self::new_with_profile(width, height, genome, VisualProfile::Standard)\n    }\n\n    pub fn new_with_profile(\n        width: u32,\n        height: u32,\n        genome: BootGenome,\n        profile: VisualProfile,\n    ) -> Self {\n        let total_duration_ms = genome.visual_budget_ms().max(1);\n        let mut rng = StdRng::from_seed(genome.seed);\n        let spores = build_spores(width, height, &genome, &mut rng);\n        let curves = build_topology(width, height, &genome, &spores, &mut rng);\n        let mesh_links = build_mesh_links(&genome, &spores, &curves, &mut rng);\n        Self {\n            width,\n            height,\n            genome,\n            profile,\n            curves,\n            spores,\n            mesh_links,\n            total_duration_ms,\n        }\n    }",
)

replace_once(
    BASE,
    "                return EcologyFrameState {\n                    stage: stage.kind,\n                    stage_progress,\n                    sequence_progress: elapsed_ms as f32 / self.total_duration_ms as f32,\n                    visible_fraction: self.visible_fraction(elapsed_ms),\n                };",
    "                return EcologyFrameState {\n                    stage: stage.kind,\n                    stage_progress,\n                    stage_intensity: stage.intensity.clamp(0.0, 1.0),\n                    sequence_progress: elapsed_ms as f32 / self.total_duration_ms as f32,\n                    visible_fraction: self.visible_fraction(elapsed_ms),\n                };",
)

replace_once(
    BASE,
    "        EcologyFrameState {\n            stage: BootStageKind::Handoff,\n            stage_progress: 1.0,\n            sequence_progress: 1.0,\n            visible_fraction: 1.0,\n        }",
    "        EcologyFrameState {\n            stage: BootStageKind::Handoff,\n            stage_progress: 1.0,\n            stage_intensity: 0.0,\n            sequence_progress: 1.0,\n            visible_fraction: 1.0,\n        }",
)

replace_once(
    BASE,
    "        let handoff_opacity = if state.stage == BootStageKind::Handoff {\n            1.0 - smoothstep(state.stage_progress)\n        } else {\n            1.0\n        };\n\n        self.draw_spores(buffer, state, handoff_opacity);\n        self.draw_curves(buffer, state, handoff_opacity);\n\n        if matches!(state.stage, BootStageKind::GrowthRing) {\n            self.draw_growth_ring(buffer, state, handoff_opacity);\n        }\n        if matches!(state.stage, BootStageKind::MeshLink | BootStageKind::Settle | BootStageKind::Handoff)\n        {\n            self.draw_mesh_links(buffer, state, handoff_opacity);\n        }",
    "        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n\n        if VisualCompositionBudget::should_render(budget.topology) {\n            self.draw_spores(buffer, state, budget.topology);\n            self.draw_curves(buffer, state, budget.topology);\n        }\n\n        if matches!(state.stage, BootStageKind::GrowthRing)\n            && VisualCompositionBudget::should_render(budget.accent)\n        {\n            self.draw_growth_ring(buffer, state, budget.accent);\n        }\n        if matches!(\n            state.stage,\n            BootStageKind::MeshLink | BootStageKind::Settle | BootStageKind::Handoff\n        ) && VisualCompositionBudget::should_render(budget.mesh)\n        {\n            self.draw_mesh_links(buffer, state, budget.mesh);\n        }",
)

replace_once(
    BASE,
    "        for curve in &self.curves {",
    "        for (curve_index, curve) in self.curves.iter().enumerate() {",
)

replace_once(
    BASE,
    "            if local > 0.96 && curve.depth > 0 {",
    "            if local > 0.96\n                && curve.depth > 0\n                && node_visible(\n                    &self.genome.seed,\n                    curve_index,\n                    self.genome.parameters.node_density,\n                    curve.repair_mark,\n                )\n            {",
)

# ---------------------------------------------------------------------------
# Holographic wrapper: share the same profile and let the central budget govern
# the field. Energy sweep is additionally coupled to semantic accent so repair
# and rollback cannot become generic holographic spectacle.
# ---------------------------------------------------------------------------
replace_once(
    HOLO,
    "use crate::ecology_renderer_base::EcologyRenderer as BaseEcologyRenderer;\npub use crate::ecology_renderer_base::EcologyFrameState;",
    "use crate::ecology_renderer_base::EcologyRenderer as BaseEcologyRenderer;\npub use crate::ecology_renderer_base::EcologyFrameState;\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};",
)

replace_once(
    HOLO,
    "pub struct EcologyRenderer {\n    base: BaseEcologyRenderer,\n    field: HolographicField,\n}",
    "pub struct EcologyRenderer {\n    base: BaseEcologyRenderer,\n    field: HolographicField,\n    profile: VisualProfile,\n}",
)

replace_once(
    HOLO,
    "impl EcologyRenderer {\n    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {\n        let field = HolographicField::new(width, height, &genome);\n        let base = BaseEcologyRenderer::new(width, height, genome);\n        Self { base, field }\n    }",
    "impl EcologyRenderer {\n    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {\n        Self::new_with_profile(width, height, genome, VisualProfile::Standard)\n    }\n\n    pub fn new_with_profile(\n        width: u32,\n        height: u32,\n        genome: BootGenome,\n        profile: VisualProfile,\n    ) -> Self {\n        let field = HolographicField::new(width, height, &genome);\n        let base = BaseEcologyRenderer::new_with_profile(width, height, genome, profile);\n        Self {\n            base,\n            field,\n            profile,\n        }\n    }",
)

replace_once(
    HOLO,
    "        let state = self.base.render_at(elapsed_ms, buffer);\n        if state.stage != BootStageKind::Blackout {\n            self.field.render(buffer, state);\n        }\n        state",
    "        let state = self.base.render_at(elapsed_ms, buffer);\n        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n        if state.stage != BootStageKind::Blackout\n            && VisualCompositionBudget::should_render(budget.holography)\n        {\n            self.field\n                .render(buffer, state, budget.holography, budget.accent);\n        }\n        state",
)

replace_once(
    HOLO,
    "    fn render(&self, buffer: &mut [u32], state: EcologyFrameState) {",
    "    fn render(\n        &self,\n        buffer: &mut [u32],\n        state: EcologyFrameState,\n        holography_gain: f32,\n        accent_gain: f32,\n    ) {",
)

replace_once(
    HOLO,
    "        let handoff = if state.stage == BootStageKind::Handoff {\n            1.0 - smoothstep(state.stage_progress)\n        } else {\n            1.0\n        };\n        let arrival = smoothstep((state.sequence_progress * 4.5).clamp(0.0, 1.0));\n        let gain = self.field_strength * handoff * arrival;",
    "        let arrival = smoothstep((state.sequence_progress * 4.5).clamp(0.0, 1.0));\n        let gain = self.field_strength * arrival * holography_gain;",
)

replace_once(
    HOLO,
    "        self.draw_anchor_field(buffer, state, gain, primary);\n        self.draw_energy_sweep(buffer, state, gain, primary);\n        self.draw_scanline_sheen(buffer, phase, gain);",
    "        self.draw_anchor_field(buffer, state, gain, primary);\n        self.draw_energy_sweep(buffer, state, gain * accent_gain, primary);\n        self.draw_scanline_sheen(buffer, phase, gain);",
)

# ---------------------------------------------------------------------------
# Fidelity wrapper: budget membrane, caustics and bloom independently. Remove
# duplicated stage-level bloom/handoff policy; the central composition budget is
# now the only semantic attention governor.
# ---------------------------------------------------------------------------
replace_once(
    FIDELITY,
    "use crate::ecology_renderer_holo::EcologyRenderer as HolographicEcologyRenderer;\npub use crate::ecology_renderer_holo::EcologyFrameState;",
    "use crate::ecology_renderer_holo::EcologyRenderer as HolographicEcologyRenderer;\npub use crate::ecology_renderer_holo::EcologyFrameState;\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};",
)

replace_once(
    FIDELITY,
    "pub struct EcologyRenderer {\n    inner: HolographicEcologyRenderer,\n    fidelity: FidelityField,\n    bloom: RefCell<BloomWorkspace>,\n}",
    "pub struct EcologyRenderer {\n    inner: HolographicEcologyRenderer,\n    fidelity: FidelityField,\n    bloom: RefCell<BloomWorkspace>,\n    profile: VisualProfile,\n}",
)

replace_once(
    FIDELITY,
    "impl EcologyRenderer {\n    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {\n        let fidelity = FidelityField::new(width, height, &genome);\n        let bloom = RefCell::new(BloomWorkspace::new(width as usize, height as usize));\n        let inner = HolographicEcologyRenderer::new(width, height, genome);\n        Self {\n            inner,\n            fidelity,\n            bloom,\n        }\n    }",
    "impl EcologyRenderer {\n    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {\n        Self::new_with_profile(width, height, genome, VisualProfile::Standard)\n    }\n\n    pub fn new_with_profile(\n        width: u32,\n        height: u32,\n        genome: BootGenome,\n        profile: VisualProfile,\n    ) -> Self {\n        let fidelity = FidelityField::new(width, height, &genome);\n        let bloom = RefCell::new(BloomWorkspace::new(width as usize, height as usize));\n        let inner = HolographicEcologyRenderer::new_with_profile(width, height, genome, profile);\n        Self {\n            inner,\n            fidelity,\n            bloom,\n            profile,\n        }\n    }",
)

replace_once(
    FIDELITY,
    "        self.fidelity.render(buffer, state);\n        self.bloom.borrow_mut().apply(\n            buffer,\n            self.fidelity.width,\n            self.fidelity.height,\n            self.fidelity.bloom_strength(state),\n        );\n        state",
    "        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n        self.fidelity\n            .render(buffer, state, budget.membrane, budget.caustics);\n        if VisualCompositionBudget::should_render(budget.bloom) {\n            self.bloom.borrow_mut().apply(\n                buffer,\n                self.fidelity.width,\n                self.fidelity.height,\n                self.fidelity.bloom_strength(budget.bloom),\n            );\n        }\n        state",
)

replace_once(
    FIDELITY,
    "    fn bloom_strength(&self, state: EcologyFrameState) -> f32 {\n        let stage_gain = match state.stage {\n            BootStageKind::Repair | BootStageKind::GrowthRing => 0.38,\n            BootStageKind::Relight | BootStageKind::MeshLink => 0.34,\n            BootStageKind::Germinate | BootStageKind::Grow => 0.31,\n            BootStageKind::Settle => 0.28,\n            BootStageKind::Handoff => 0.22 * (1.0 - smoothstep(state.stage_progress)),\n            _ => 0.25,\n        };\n        (stage_gain * self.bloom_gain).clamp(0.0, 0.42)\n    }",
    "    fn bloom_strength(&self, budget_gain: f32) -> f32 {\n        (0.36 * self.bloom_gain * budget_gain).clamp(0.0, 0.42)\n    }",
)

replace_once(
    FIDELITY,
    "    fn render(&self, buffer: &mut [u32], state: EcologyFrameState) {\n        if self.width == 0 || self.height == 0 || buffer.len() < self.width * self.height {\n            return;\n        }\n        let arrival = smoothstep((state.sequence_progress * 4.0).clamp(0.0, 1.0));\n        let handoff = if state.stage == BootStageKind::Handoff {\n            1.0 - smoothstep(state.stage_progress)\n        } else {\n            1.0\n        };\n        let gain = self.shell_gain * arrival * handoff;\n        if gain <= 0.002 {\n            return;\n        }\n        self.draw_membrane(buffer, state, gain);\n        self.draw_caustics(buffer, state, gain);\n    }",
    "    fn render(\n        &self,\n        buffer: &mut [u32],\n        state: EcologyFrameState,\n        membrane_gain: f32,\n        caustic_gain: f32,\n    ) {\n        if self.width == 0 || self.height == 0 || buffer.len() < self.width * self.height {\n            return;\n        }\n        let arrival = smoothstep((state.sequence_progress * 4.0).clamp(0.0, 1.0));\n        let membrane_gain = self.shell_gain * arrival * membrane_gain;\n        let caustic_gain = self.shell_gain * arrival * caustic_gain;\n        if VisualCompositionBudget::should_render(membrane_gain) {\n            self.draw_membrane(buffer, state, membrane_gain);\n        }\n        if VisualCompositionBudget::should_render(caustic_gain) {\n            self.draw_caustics(buffer, state, caustic_gain);\n        }\n    }",
)

# ---------------------------------------------------------------------------
# Identity wrapper: profile-aware factual microtype, with handoff fading governed
# by the same central budget rather than another independent semantic curve.
# ---------------------------------------------------------------------------
replace_once(
    IDENTITY,
    "use crate::microtype;\nuse symthaea_boot_ecology::{BootCue, BootGenome, BootStageKind};",
    "use crate::microtype;\nuse crate::visual_composition::{VisualCompositionBudget, VisualProfile};\nuse symthaea_boot_ecology::{BootCue, BootGenome, BootStageKind};",
)

replace_once(
    IDENTITY,
    "    cue: BootCue,\n    inner: FidelityEcologyRenderer,",
    "    cue: BootCue,\n    inner: FidelityEcologyRenderer,\n    profile: VisualProfile,",
)

replace_once(
    IDENTITY,
    "impl EcologyRenderer {\n    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {\n        let cue = genome.cue;\n        Self {\n            width: width as usize,\n            height: height as usize,\n            cue,\n            inner: FidelityEcologyRenderer::new(width, height, genome),\n        }\n    }",
    "impl EcologyRenderer {\n    pub fn new(width: u32, height: u32, genome: BootGenome) -> Self {\n        Self::new_with_profile(width, height, genome, VisualProfile::Standard)\n    }\n\n    pub fn new_with_profile(\n        width: u32,\n        height: u32,\n        genome: BootGenome,\n        profile: VisualProfile,\n    ) -> Self {\n        let cue = genome.cue;\n        Self {\n            width: width as usize,\n            height: height as usize,\n            cue,\n            inner: FidelityEcologyRenderer::new_with_profile(width, height, genome, profile),\n            profile,\n        }\n    }",
)

replace_once(
    IDENTITY,
    "        let state = self.inner.render_at(elapsed_ms, buffer);\n        if state.stage != BootStageKind::Blackout {\n            self.draw_identity(buffer, state);\n        }\n        state",
    "        let state = self.inner.render_at(elapsed_ms, buffer);\n        let budget = VisualCompositionBudget::derive(\n            state.stage,\n            state.stage_progress,\n            state.stage_intensity,\n            self.profile,\n        );\n        if state.stage != BootStageKind::Blackout\n            && VisualCompositionBudget::should_render(budget.identity)\n        {\n            self.draw_identity(buffer, state, budget.identity);\n        }\n        state",
)

replace_once(
    IDENTITY,
    "    fn draw_identity(&self, buffer: &mut [u32], state: EcologyFrameState) {",
    "    fn draw_identity(\n        &self,\n        buffer: &mut [u32],\n        state: EcologyFrameState,\n        identity_gain: f32,\n    ) {",
)

replace_once(
    IDENTITY,
    "        let arrival = smoothstep(((state.sequence_progress - 0.06) * 5.0).clamp(0.0, 1.0));\n        let departure = if state.stage == BootStageKind::Handoff {\n            1.0 - smoothstep(state.stage_progress)\n        } else {\n            1.0\n        };\n        let opacity = arrival * departure;",
    "        let arrival = smoothstep(((state.sequence_progress - 0.06) * 5.0).clamp(0.0, 1.0));\n        let opacity = arrival * identity_gain;",
)

replace_once(
    IDENTITY,
    "    #[test]\n    fn factual_labels_avoid_consciousness_claims() {",
    "    #[test]\n    fn profiles_change_pixels_without_changing_semantic_frame() {\n        let calm = EcologyRenderer::new_with_profile(\n            320,\n            180,\n            genome(),\n            VisualProfile::Calm,\n        );\n        let rich = EcologyRenderer::new_with_profile(\n            320,\n            180,\n            genome(),\n            VisualProfile::Rich,\n        );\n        let mut calm_pixels = vec![0u32; 320 * 180];\n        let mut rich_pixels = vec![0u32; 320 * 180];\n        let calm_state = calm.render_at(2_000, &mut calm_pixels);\n        let rich_state = rich.render_at(2_000, &mut rich_pixels);\n        assert_eq!(calm_state.stage, rich_state.stage);\n        assert_eq!(calm_state.stage_progress, rich_state.stage_progress);\n        assert_ne!(calm_pixels, rich_pixels);\n    }\n\n    #[test]\n    fn factual_labels_avoid_consciousness_claims() {",
)

print("Spore v0.3.3 guarded renderer integration applied")
