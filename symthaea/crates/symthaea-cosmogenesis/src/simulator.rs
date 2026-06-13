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

    pub fn run_simulation(&mut self) -> Vec<CognitiveCosmogenesisMetrics> {
        let mut history = Vec::with_capacity(self.params.steps + 1);
        history.push(self.calculate_metrics(0));

        for step in 1..=self.params.steps {
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
            history.push(self.calculate_metrics(step));
        }

        history
    }

    pub fn calculate_metrics(&self, step: usize) -> CognitiveCosmogenesisMetrics {
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
            current_step: step,
            separation_proxy: if mean_intra + mean_inter > 0.0 {
                (mean_inter - mean_intra) / mean_inter.max(mean_intra)
            } else {
                0.0
            },
            mean_intra_class_distance: mean_intra,
            mean_inter_class_distance: mean_inter,
            davies_bouldin_index: 0.0,
            retrieval_precision_at_k: 0.0,
            entropy: 0.0,
            cluster_stability: 0.0,
        }
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

    #[test]
    fn test_tuned_params_can_improve_separation() {
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

        let mut best_improvement = 0.0;
        let mut best_params = None;

        let attraction_values = [0.001, 0.005, 0.01, 0.02, 0.05];
        let lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05];
        let perturbation_values = [0.05, 0.1, 0.2, 0.5];
        let cooling_values = [0.8, 0.9, 0.95];

        let base_sim =
            CosmogenesisSimulator::new(CognitiveCosmologyParams::default(), particles.clone());
        let initial_metrics = base_sim.calculate_metrics();

        for &attraction in &attraction_values {
            for &lambda in &lambda_values {
                for &perturbation in &perturbation_values {
                    for &cooling in &cooling_values {
                        let params = CognitiveCosmologyParams {
                            attraction_strength: attraction,
                            lambda,
                            perturbation_scale: perturbation,
                            cooling_rate: cooling,
                            steps: 50,
                            ..Default::default()
                        };

                        let mut sim = CosmogenesisSimulator::new(params.clone(), particles.clone());
                        let final_metrics = sim.run_simulation();
                        let improvement =
                            final_metrics.separation_proxy - initial_metrics.separation_proxy;

                        if improvement > best_improvement
                            && final_metrics.separation_proxy.is_finite()
                        {
                            best_improvement = improvement;
                            best_params = Some(params);
                        }
                    }
                }
            }
        }

        println!("Best Improvement: {}", best_improvement);
        if let Some(params) = &best_params {
            println!("Best Params: {:?}", params);
        }
        assert!(
            best_improvement > 0.0,
            "No parameter configuration improved separation proxy"
        );
    }

    #[test]
    fn test_cosmogenesis_on_real_hdc_vectors() {
        use symthaea_core::hdc::semantic_encoder::{EncoderType, create_encoder};

        let encoder = create_encoder(EncoderType::CachedSemantic);

        let class_0_texts = [
            "unauthorized access detected on xenia ledger",
            "denial of service attack on port eighty eighty",
            "critical security breach credential leak compromise",
        ];

        let class_1_texts = [
            "routine connection handshake established successfully",
            "system boot setup completed normal operation",
            "log cleanup job completed with zero warnings",
        ];

        let mut particles = Vec::new();
        let mut id_counter = 0;

        for text in &class_0_texts {
            let hv = encoder.encode(text);
            id_counter += 1;
            particles.push(SemanticParticle {
                id: id_counter.to_string(),
                class_id: 0,
                position: hv.values.clone(),
                velocity: vec![0.0; hv.values.len()],
                mass: 1.0,
                latent_mass: 1.0,
            });
        }

        for text in &class_1_texts {
            let hv = encoder.encode(text);
            id_counter += 1;
            particles.push(SemanticParticle {
                id: id_counter.to_string(),
                class_id: 1,
                position: hv.values.clone(),
                velocity: vec![0.0; hv.values.len()],
                mass: 1.0,
                latent_mass: 1.0,
            });
        }

        let params = CognitiveCosmologyParams::default();

        let mut base_sim = CosmogenesisSimulator::new(params, particles);
        let initial_metrics = base_sim.calculate_metrics();

        let final_metrics = base_sim.run_simulation();
        let improvement = final_metrics.separation_proxy - initial_metrics.separation_proxy;

        println!(
            "HDC Initial Separation: {}",
            initial_metrics.separation_proxy
        );
        println!("HDC Final Separation: {}", final_metrics.separation_proxy);
        println!("HDC Separation Improvement: {}", improvement);

        assert!(
            improvement > 0.0,
            "Cosmogenesis did not improve separation on real HDC vectors"
        );
    }
}
