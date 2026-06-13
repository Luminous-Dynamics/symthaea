// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::{CognitiveCosmogenesisMetrics, CognitiveCosmologyParams, SemanticParticle};

pub struct CosmogenesisSimulator {
    params: CognitiveCosmologyParams,
    particles: Vec<SemanticParticle>,
}

impl CosmogenesisSimulator {
    pub fn new(params: CognitiveCosmologyParams, initial_particles: Vec<SemanticParticle>) -> Self {
        if let Some(first) = initial_particles.first() {
            let dim = first.position.len();
            for p in &initial_particles {
                assert_eq!(p.position.len(), dim, "Dimension mismatch in particles");
            }
        }
        Self {
            params,
            particles: initial_particles,
        }
    }

    pub fn particles(&self) -> &[SemanticParticle] {
        &self.particles
    }

    pub fn calculate_metrics(&self) -> CognitiveCosmogenesisMetrics {
        let mut intra_dist = 0.0;
        let mut inter_dist = 0.0;
        let mut intra_count = 0;
        let mut inter_count = 0;

        for i in 0..self.particles.len() {
            for j in (i + 1)..self.particles.len() {
                let dist: f32 = self.particles[i]
                    .position
                    .iter()
                    .zip(&self.particles[j].position)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();

                if self.particles[i].class_id == self.particles[j].class_id {
                    intra_dist += dist;
                    intra_count += 1;
                } else {
                    inter_dist += dist;
                    inter_count += 1;
                }
            }
        }

        let mean_intra = if intra_count > 0 {
            intra_dist / intra_count as f32
        } else {
            0.0
        };
        let mean_inter = if inter_count > 0 {
            inter_dist / inter_count as f32
        } else {
            0.0
        };

        CognitiveCosmogenesisMetrics {
            separation_proxy: if mean_intra + mean_inter > 0.0 {
                (mean_inter - mean_intra) / mean_inter.max(mean_intra)
            } else {
                0.0
            },
            davies_bouldin_index: 0.0,
            retrieval_precision_at_k: 0.0,
            entropy: 0.0,
            cluster_stability: 0.0,
        }
    }

    pub fn run_simulation(&mut self) -> CognitiveCosmogenesisMetrics {
        for _step in 0..self.params.steps {
            let mut next_particles = self.particles.clone();

            for i in 0..self.particles.len() {
                let mut force = vec![0.0; self.particles[i].position.len()];

                for j in 0..self.particles.len() {
                    if i == j {
                        continue;
                    }
                    let dist_vec: Vec<f32> = self.particles[i]
                        .position
                        .iter()
                        .zip(&self.particles[j].position)
                        .map(|(a, b)| a - b)
                        .collect();
                    let dist_sq: f32 = dist_vec.iter().map(|d| d * d).sum::<f32>();
                    let softened = dist_sq + self.params.perturbation_scale.max(1e-6);

                    if self.particles[i].class_id == self.particles[j].class_id {
                        for k in 0..force.len() {
                            force[k] -=
                                self.params.attraction_strength * dist_vec[k] / softened.powf(1.5);
                        }
                    } else {
                        for k in 0..force.len() {
                            force[k] += self.params.lambda * dist_vec[k] / softened.powf(1.5);
                        }
                    }
                }

                for k in 0..force.len() {
                    next_particles[i].velocity[k] =
                        (next_particles[i].velocity[k] + force[k] * 0.1) * self.params.cooling_rate;
                    next_particles[i].position[k] += next_particles[i].velocity[k];
                }
            }
            self.particles = next_particles;
        }

        self.calculate_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_simulation_stability() {
        let particles = vec![
            SemanticParticle {
                id: "1".into(),
                class_id: 0,
                position: vec![0.0, 0.0],
                velocity: vec![0.0, 0.0],
                mass: 1.0,
                latent_mass: 1.0,
            },
            SemanticParticle {
                id: "2".into(),
                class_id: 0,
                position: vec![0.1, 0.1],
                velocity: vec![0.0, 0.0],
                mass: 1.0,
                latent_mass: 1.0,
            },
            SemanticParticle {
                id: "3".into(),
                class_id: 1,
                position: vec![1.0, 1.0],
                velocity: vec![0.0, 0.0],
                mass: 1.0,
                latent_mass: 1.0,
            },
            SemanticParticle {
                id: "4".into(),
                class_id: 1,
                position: vec![1.1, 1.1],
                velocity: vec![0.0, 0.0],
                mass: 1.0,
                latent_mass: 1.0,
            },
        ];

        let mut sim = CosmogenesisSimulator::new(CognitiveCosmologyParams::default(), particles);
        let metrics = sim.run_simulation();

        // Stability verification
        assert!(metrics.separation_proxy.is_finite());
        for p in sim.particles() {
            for pos in &p.position {
                assert!(pos.is_finite());
            }
        }
    }

    #[ignore]
    #[test]
    fn test_tuned_params_can_improve_separation() {
        // Experimental tuning target
    }
}
