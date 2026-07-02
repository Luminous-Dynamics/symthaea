// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-Modal Bridge between Broca (Language) and Vision (Manifold).
//!
//! Establishes a shared semantic space where linguistic concepts
//! are grounded in visual perception.

#[cfg(feature = "mamba-cpu")]
use symthaea_broca::liquid_mamba::LiquidMambaGenerator;
use symthaea_vision_manifold::bridge::VisionBridge;

/// Orchestrates cross-modal alignment between vision and language.
pub struct CrossModalBridge {
    pub vision: VisionBridge,
    #[cfg(feature = "mamba-cpu")]
    pub language: LiquidMambaGenerator,
    /// Coupling strength between modalities [0, 1].
    pub coupling: f32,
}

impl CrossModalBridge {
    #[cfg(feature = "mamba-cpu")]
    pub fn new(vision: VisionBridge, language: LiquidMambaGenerator, coupling: f32) -> Self {
        Self {
            vision,
            language,
            coupling: coupling.clamp(0.0, 1.0),
        }
    }

    /// Sync modalities: top-down language goals drive vision,
    /// bottom-up vision surprise modulates language.
    #[cfg(feature = "mamba-cpu")]
    pub fn sync(&mut self) -> anyhow::Result<()> {
        // 1. Top-down: Language state (thought) becomes a visual goal
        // (Simplified: using the last generated chunk's thought HV)
        if let Some(last_chunk) = self.language.chunk_history.back() {
            let goal = symthaea_vision_manifold::bridge::CognitiveGoalSignal {
                task_hv: Some(last_chunk.thought_hv.clone()),
                task_gain: 0.4 * self.coupling,
                learning_rate: 0.05,
            };
            self.vision.set_goal_signal(goal);
        }

        // 2. Bottom-up: Vision surprise modulates language FEP
        // (High visual surprise increases language reactivity)
        let vision_surprise = self.vision.manifold().last_fep().free_energy;
        let fep_modulation = 1.0 + (vision_surprise * self.coupling);
        self.language.set_fep_modulation(fep_modulation);

        // 3. Embodied Curiosity: Inject curiosity based on visual surprise
        if vision_surprise > 0.5 {
            // Map visual coordinates/salience to a semantic intent sector (Simplified)
            let salient_sector = (vision_surprise * 1000.0) as usize % 1000;
            self.language
                .inject_curiosity(salient_sector, vision_surprise * self.coupling);
        }

        Ok(())
    }
}
