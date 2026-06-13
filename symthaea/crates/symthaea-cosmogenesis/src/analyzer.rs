// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::SemanticParticle;

/// Analyzes the semantic manifold to extract high-connectivity pathways (the "Cosmic Web").
pub struct CosmicWebAnalyzer {
    pub connectivity_threshold: f32,
}

impl CosmicWebAnalyzer {
    pub fn new(connectivity_threshold: f32) -> Self {
        Self {
            connectivity_threshold,
        }
    }

    /// Generates an adjacency matrix representing the semantic 'circuitry'
    /// where particles are linked if their proximity and mass product exceeds the threshold.
    pub fn extract_circuitry(&self, particles: &[SemanticParticle]) -> Vec<Vec<f32>> {
        let n = particles.len();
        let mut adj = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq: f32 = particles[i]
                    .position
                    .iter()
                    .zip(&particles[j].position)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();

                let mass_product = particles[i].latent_mass * particles[j].latent_mass;

                // Weight is a function of proximity and historical priority (mass)
                let weight = mass_product / (dist_sq + 1e-6);

                if weight > self.connectivity_threshold {
                    adj[i][j] = weight;
                    adj[j][i] = weight;
                }
            }
        }
        adj
    }

    /// Identifies 'hubs' (high-degree nodes) in the cosmic web.
    pub fn identify_hubs(&self, adjacency: &[Vec<f32>]) -> Vec<(usize, f32)> {
        let mut degree = vec![0.0; adjacency.len()];
        for (i, row) in adjacency.iter().enumerate() {
            for &weight in row {
                degree[i] += weight;
            }
        }
        let mut indexed: Vec<(usize, f32)> = degree.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed
    }
}
