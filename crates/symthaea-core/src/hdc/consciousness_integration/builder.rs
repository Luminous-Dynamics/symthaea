// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Builder for constructing a ConsciousnessPipeline with desired configuration.

use super::super::consciousness_subsystem::ConsciousnessSubsystem;
use super::pipeline::ConsciousnessPipeline;
use super::types::IntegrationConfig;

/// Builder for constructing a ConsciousnessPipeline with desired configuration
pub struct ConsciousnessPipelineBuilder {
    config: IntegrationConfig,
    embodiment_level: f64,
    integrated_systems: bool,
    phi_feedback: bool,
    creativity: bool,
    fractal: bool,
    meta_consciousness: bool,
    temporal_consciousness: bool,
    phase_transitions: bool,
    epistemic: bool,
    collective: bool,
    subsystems: Vec<Box<dyn ConsciousnessSubsystem>>,
    verification_interval: Option<usize>,
}

impl ConsciousnessPipelineBuilder {
    pub fn new() -> Self {
        Self {
            config: IntegrationConfig::default(),
            embodiment_level: 0.5,
            integrated_systems: false,
            phi_feedback: false,
            creativity: false,
            fractal: false,
            meta_consciousness: false,
            temporal_consciousness: false,
            phase_transitions: false,
            epistemic: false,
            collective: false,
            subsystems: Vec::new(),
            verification_interval: None,
        }
    }

    pub fn config(mut self, config: IntegrationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn embodiment(mut self, level: f64) -> Self {
        self.embodiment_level = level;
        self
    }

    pub fn integrated_systems(mut self) -> Self {
        self.integrated_systems = true;
        self
    }

    pub fn phi_feedback(mut self) -> Self {
        self.phi_feedback = true;
        self
    }

    pub fn creativity(mut self) -> Self {
        self.creativity = true;
        self
    }

    pub fn fractal(mut self) -> Self {
        self.fractal = true;
        self
    }

    pub fn meta_consciousness(mut self) -> Self {
        self.meta_consciousness = true;
        self
    }

    pub fn temporal_consciousness(mut self) -> Self {
        self.temporal_consciousness = true;
        self
    }

    pub fn phase_transitions(mut self) -> Self {
        self.phase_transitions = true;
        self
    }

    pub fn epistemic(mut self) -> Self {
        self.epistemic = true;
        self
    }

    pub fn collective(mut self) -> Self {
        self.collective = true;
        self
    }

    pub fn subsystem(mut self, sub: Box<dyn ConsciousnessSubsystem>) -> Self {
        self.subsystems.push(sub);
        self
    }

    pub fn verification(mut self, interval: usize) -> Self {
        self.verification_interval = Some(interval);
        self
    }

    /// Enable all consciousness systems (delegates to enable_full_consciousness)
    pub fn full_consciousness(self) -> Self {
        // Mark as full; build() will use enable_full_consciousness()
        Self {
            integrated_systems: true,
            phi_feedback: true,
            creativity: true,
            fractal: true,
            meta_consciousness: true,
            temporal_consciousness: true,
            phase_transitions: true,
            epistemic: true,
            collective: true,
            ..self
        }
    }

    pub fn build(self) -> ConsciousnessPipeline {
        let mut pipeline = ConsciousnessPipeline::new(self.config);
        pipeline.embodiment_level = self.embodiment_level;

        // If all systems requested, use the pipeline's full setup (includes phi_optimization etc.)
        let is_full = self.integrated_systems
            && self.phi_feedback
            && self.creativity
            && self.fractal
            && self.meta_consciousness
            && self.temporal_consciousness
            && self.phase_transitions
            && self.epistemic
            && self.collective;

        if is_full {
            pipeline.enable_full_consciousness();
        } else {
            if self.integrated_systems {
                pipeline.enable_integrated_systems();
            }
            if self.phi_feedback {
                pipeline.enable_phi_feedback();
            }
            if self.creativity {
                pipeline.enable_creativity();
            }
            if self.fractal {
                pipeline.enable_fractal(3);
            }
            if self.meta_consciousness {
                pipeline.enable_meta_consciousness(8);
            }
            if self.temporal_consciousness {
                pipeline.enable_temporal_consciousness(8);
            }
            if self.phase_transitions {
                pipeline.enable_phase_transitions();
            }
            if self.epistemic {
                pipeline.enable_epistemic(8);
            }
            if self.collective {
                pipeline.enable_collective("primary");
            }
        }
        for sub in self.subsystems {
            pipeline.register_subsystem(sub);
        }
        if let Some(interval) = self.verification_interval {
            pipeline.enable_verification(interval);
        }
        pipeline
    }
}

impl Default for ConsciousnessPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
