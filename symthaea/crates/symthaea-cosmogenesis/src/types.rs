// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveCosmologyParams {
    pub matter_density: f32,
    pub dark_matter_density: f32,
    pub lambda: f32,
    pub attraction_strength: f32,
    pub perturbation_scale: f32,
    pub cooling_rate: f32,
    pub steps: usize,
}

impl Default for CognitiveCosmologyParams {
    fn default() -> Self {
        Self {
            matter_density: 1.0,
            dark_matter_density: 0.5,
            lambda: 0.02,
            attraction_strength: 0.02,
            perturbation_scale: 0.2,
            cooling_rate: 0.8,
            steps: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveCosmogenesisMetrics {
    pub current_step: usize,
    pub separation_proxy: f32,
    pub mean_intra_class_distance: f32,
    pub mean_inter_class_distance: f32,
    pub davies_bouldin_index: f32,
    pub retrieval_precision_at_k: f32,
    pub entropy: f32,
    pub cluster_stability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticParticle {
    pub id: String,
    pub class_id: usize,
    pub position: Vec<f32>,
    pub velocity: Vec<f32>,
    pub mass: f32,
    pub latent_mass: f32,
}
