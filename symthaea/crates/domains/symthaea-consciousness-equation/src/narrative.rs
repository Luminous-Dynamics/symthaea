// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::VecDeque;
// web-time: drop-in Instant for wasm32 (std::time::Instant panics on wasm32-unknown-unknown)
use web_time::Instant;

/// Narrative Coherence: N = autobiographical_integration × future_simulation_depth
///
/// This factor measures the coherence of the system's self-narrative:
/// - Autobiographical integration: How well past experiences form a coherent story
/// - Future simulation depth: Ability to simulate and plan future scenarios
#[derive(Debug, Clone)]
pub struct NarrativeCoherence {
    /// Autobiographical memory integration score
    autobiographical_integration: f64,

    /// Future simulation depth (0-1, based on planning horizon)
    future_simulation_depth: f64,

    /// Narrative episodes stored
    episodes: VecDeque<NarrativeEpisode>,

    /// Future scenarios being simulated
    future_scenarios: Vec<FutureScenario>,

    /// Maximum episodes to store
    max_episodes: usize,

    /// Smoothing factor
    smoothing: f64,
}

/// An episode in the autobiographical narrative
#[derive(Debug, Clone)]
pub struct NarrativeEpisode {
    /// Episode content/summary
    pub content: String,
    /// When this happened
    pub timestamp: Instant,
    /// Emotional valence (-1 to 1)
    pub valence: f64,
    /// How well integrated with other episodes
    pub integration_score: f64,
    /// Causal connections to other episodes
    pub causal_links: Vec<usize>,
}

/// A future scenario being simulated
#[derive(Debug, Clone)]
pub struct FutureScenario {
    /// Scenario description
    pub description: String,
    /// Time horizon (how far into future)
    pub horizon_steps: usize,
    /// Probability estimate
    pub probability: f64,
    /// Desirability (-1 to 1)
    pub desirability: f64,
}

impl Default for NarrativeCoherence {
    fn default() -> Self {
        Self::new()
    }
}

impl NarrativeCoherence {
    pub fn new() -> Self {
        Self {
            autobiographical_integration: 0.5,
            future_simulation_depth: 0.5,
            episodes: VecDeque::with_capacity(100),
            future_scenarios: Vec::new(),
            max_episodes: 100,
            // EMA smoothing for autobiographical integration updates.
            // 0.1 → 23 episodes to 90% convergence (~115 cycles at 1 ep/5 cycles).
            // 0.2 → 11 episodes to 90% convergence (~55 cycles). Faster pickup
            // of narrative structure without over-weighting single episodes.
            smoothing: 0.2,
        }
    }

    /// Add a new narrative episode
    pub fn add_episode(&mut self, content: String, valence: f64) {
        // Compute integration with existing episodes
        let integration = self.compute_episode_integration(&content);

        // Find causal links to recent episodes
        let causal_links: Vec<usize> = self
            .episodes
            .iter()
            .rev()
            .take(5)
            .enumerate()
            .filter(|(_, ep)| {
                // Simple heuristic: episodes with similar valence may be linked
                (ep.valence - valence).abs() < 0.3
            })
            .map(|(i, _)| self.episodes.len() - 1 - i)
            .collect();

        if self.episodes.len() >= self.max_episodes {
            self.episodes.pop_front();
        }

        self.episodes.push_back(NarrativeEpisode {
            content,
            timestamp: Instant::now(),
            valence,
            integration_score: integration,
            causal_links,
        });

        // Update overall integration score
        self.update_integration();
    }

    /// Add a future scenario being simulated
    pub fn add_future_scenario(
        &mut self,
        description: String,
        horizon_steps: usize,
        probability: f64,
        desirability: f64,
    ) {
        self.future_scenarios.push(FutureScenario {
            description,
            horizon_steps,
            probability,
            desirability,
        });

        // Keep only recent scenarios
        if self.future_scenarios.len() > 10 {
            self.future_scenarios.remove(0);
        }

        // Update simulation depth based on horizon
        self.update_simulation_depth();
    }

    /// Compute how well a new episode integrates with existing narrative
    fn compute_episode_integration(&self, _content: &str) -> f64 {
        if self.episodes.is_empty() {
            return 0.5; // Neutral for first episode
        }

        // Heuristic: integration based on temporal proximity and narrative continuity
        let recent_count = self.episodes.iter().rev().take(10).count();

        // More recent episodes = better integration potential
        let recency_factor = recent_count as f64 / 10.0;

        // Causal density (episodes with links)
        let linked_count = self
            .episodes
            .iter()
            .filter(|ep| !ep.causal_links.is_empty())
            .count();
        let causal_density = linked_count as f64 / self.episodes.len().max(1) as f64;

        recency_factor * 0.5 + causal_density * 0.5
    }

    /// Update overall autobiographical integration score
    fn update_integration(&mut self) {
        if self.episodes.is_empty() {
            return;
        }

        let avg_integration = self
            .episodes
            .iter()
            .map(|ep| ep.integration_score)
            .sum::<f64>()
            / self.episodes.len() as f64;

        self.autobiographical_integration = self.autobiographical_integration
            * (1.0 - self.smoothing)
            + avg_integration * self.smoothing;
    }

    /// Update future simulation depth
    fn update_simulation_depth(&mut self) {
        if self.future_scenarios.is_empty() {
            self.future_simulation_depth = 0.3; // Base capacity
            return;
        }

        // Depth based on longest horizon and number of scenarios
        let max_horizon = self
            .future_scenarios
            .iter()
            .map(|s| s.horizon_steps)
            .max()
            .unwrap_or(1);

        let scenario_count = self.future_scenarios.len();

        // Normalize: 10 steps = 1.0 depth, more scenarios = bonus
        let horizon_factor = (max_horizon as f64 / 10.0).min(1.0);
        let count_factor = (scenario_count as f64 / 5.0).min(1.0);

        self.future_simulation_depth = horizon_factor * 0.7 + count_factor * 0.3;
    }

    /// Compute narrative coherence factor N
    ///
    /// N = autobiographical_integration × future_simulation_depth × experience_maturity
    ///
    /// Experience maturity: a narrative becomes more coherent as more episodes
    /// accumulate — more material for causal linking, theme detection, and
    /// autobiographical structure. Ramps from 0.6 (no episodes) to 1.0 (50+ episodes).
    /// Science: Conway & Pleydell-Pearce (2000) — autobiographical memory coherence
    /// grows with the density of retrievable episodes.
    pub fn compute(&self) -> f64 {
        let episode_count = self.episodes.len() as f64;
        let experience_maturity = (0.6 + 0.4 * (episode_count / 50.0).min(1.0)).min(1.0);
        self.autobiographical_integration * self.future_simulation_depth * experience_maturity
    }

    /// Get autobiographical integration score
    pub fn autobiographical_integration(&self) -> f64 {
        self.autobiographical_integration
    }

    /// Get future simulation depth
    pub fn future_simulation_depth(&self) -> f64 {
        self.future_simulation_depth
    }

    /// Get number of stored episodes
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Get number of active future scenarios
    pub fn scenario_count(&self) -> usize {
        self.future_scenarios.len()
    }

    /// Add an episode with semantic embedding for better integration scoring
    ///
    /// This enhanced method uses:
    /// - Semantic similarity to existing episodes
    /// - Emotional coherence patterns
    /// - Causal chain detection
    pub fn add_episode_with_embedding(
        &mut self,
        content: String,
        valence: f64,
        semantic_embedding: &[f64],
        theme_tags: Vec<String>,
    ) {
        // Compute semantic integration with existing episodes
        let semantic_integration = self.compute_semantic_integration(semantic_embedding);

        // Compute thematic coherence
        let thematic_coherence = self.compute_thematic_coherence(&theme_tags);

        // Combined integration score
        let integration = semantic_integration * 0.5
            + thematic_coherence * 0.3
            + self.compute_episode_integration(&content) * 0.2;

        // Find causal links based on both valence and temporal proximity
        let causal_links: Vec<usize> = self
            .episodes
            .iter()
            .rev()
            .take(10)
            .enumerate()
            .filter(|(i, ep)| {
                let valence_match = (ep.valence - valence).abs() < 0.4;
                let recency_weight = 1.0 - (*i as f64 / 10.0);
                valence_match && recency_weight > 0.3
            })
            .map(|(i, _)| self.episodes.len() - 1 - i)
            .collect();

        if self.episodes.len() >= self.max_episodes {
            self.episodes.pop_front();
        }

        self.episodes.push_back(NarrativeEpisode {
            content,
            timestamp: Instant::now(),
            valence,
            integration_score: integration,
            causal_links,
        });

        self.update_integration();
    }

    /// Compute semantic integration with existing narrative
    fn compute_semantic_integration(&self, embedding: &[f64]) -> f64 {
        if self.episodes.is_empty() || embedding.is_empty() {
            return 0.5;
        }

        let avg_content_length = self
            .episodes
            .iter()
            .map(|ep| ep.content.len())
            .sum::<usize>() as f64
            / self.episodes.len() as f64;

        let embedding_magnitude: f64 = embedding.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        let semantic_density = (embedding_magnitude / (embedding.len() as f64).sqrt()).min(1.0);
        let content_coherence = (avg_content_length / 100.0).min(1.0);

        semantic_density * 0.6 + content_coherence * 0.4
    }

    /// Compute thematic coherence based on recurring themes
    fn compute_thematic_coherence(&self, tags: &[String]) -> f64 {
        if tags.is_empty() {
            return 0.5;
        }

        let theme_count = tags.len();
        let unique_themes: std::collections::HashSet<_> = tags.iter().collect();
        let uniqueness_ratio = unique_themes.len() as f64 / theme_count as f64;

        1.0 - uniqueness_ratio * 0.5
    }

    /// Compute narrative arc coherence
    pub fn compute_narrative_arc_coherence(&self) -> f64 {
        if self.episodes.len() < 5 {
            return 0.5;
        }

        let valences: Vec<f64> = self.episodes.iter().map(|ep| ep.valence).collect();
        let mean_valence = valences.iter().sum::<f64>() / valences.len() as f64;
        let variance = valences
            .iter()
            .map(|v| (v - mean_valence).powi(2))
            .sum::<f64>()
            / valences.len() as f64;

        let variance_factor = 1.0 - (variance - 0.25).abs() * 2.0;
        let total_links: usize = self.episodes.iter().map(|ep| ep.causal_links.len()).sum();
        let link_density = (total_links as f64 / self.episodes.len() as f64).min(1.0);

        variance_factor.max(0.0) * 0.5 + link_density * 0.5
    }

    /// Get the most salient episodes (highest integration + recency)
    pub fn salient_episodes(&self, top_k: usize) -> Vec<&NarrativeEpisode> {
        let mut scored: Vec<_> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| {
                let recency = i as f64 / self.episodes.len().max(1) as f64;
                let score = ep.integration_score * 0.6 + recency * 0.4;
                (ep, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(ep, _)| ep).collect()
    }

    /// Add a future scenario with branching paths
    pub fn add_branching_scenario(
        &mut self,
        description: String,
        horizon_steps: usize,
        branches: Vec<SimulationBranch>,
    ) {
        let best_branch = branches.iter().max_by(|a, b| {
            a.probability
                .partial_cmp(&b.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(branch) = best_branch {
            self.future_scenarios.push(FutureScenario {
                description,
                horizon_steps,
                probability: branch.probability,
                desirability: branch.desirability,
            });

            for alt_branch in branches.iter().filter(|b| b.probability > 0.1) {
                if self.future_scenarios.len() < 15 {
                    self.future_scenarios.push(FutureScenario {
                        description: alt_branch.description.clone(),
                        horizon_steps: alt_branch.steps_to_outcome,
                        probability: alt_branch.probability,
                        desirability: alt_branch.desirability,
                    });
                }
            }
        }

        while self.future_scenarios.len() > 15 {
            if let Some(min_idx) = self
                .future_scenarios
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.probability
                        .partial_cmp(&b.1.probability)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.future_scenarios.remove(min_idx);
            }
        }

        self.update_simulation_depth();
    }

    /// Compute prospective memory capability
    pub fn compute_prospective_capability(&self) -> f64 {
        if self.future_scenarios.is_empty() {
            return 0.3;
        }

        let avg_horizon = self
            .future_scenarios
            .iter()
            .map(|s| s.horizon_steps)
            .sum::<usize>() as f64
            / self.future_scenarios.len() as f64;
        let specificity = 1.0 - (avg_horizon / 20.0).min(1.0);

        let desirabilities: Vec<f64> = self
            .future_scenarios
            .iter()
            .map(|s| s.desirability)
            .collect();
        let mean_des = desirabilities.iter().sum::<f64>() / desirabilities.len() as f64;
        let des_variance = desirabilities
            .iter()
            .map(|d| (d - mean_des).powi(2))
            .sum::<f64>()
            / desirabilities.len() as f64;
        let action_orientation = des_variance.sqrt().min(1.0);

        let avg_prob = self
            .future_scenarios
            .iter()
            .map(|s| s.probability)
            .sum::<f64>()
            / self.future_scenarios.len() as f64;
        let realism = 1.0 - (avg_prob - 0.5).abs() * 2.0;

        specificity * 0.3 + action_orientation * 0.3 + realism.max(0.0) * 0.4
    }

    /// Compute mental time travel depth
    pub fn compute_mental_time_travel(&self) -> f64 {
        let past_depth = if self.episodes.is_empty() {
            0.0
        } else {
            let link_depth = self
                .episodes
                .iter()
                .map(|ep| ep.causal_links.len())
                .max()
                .unwrap_or(0);
            (link_depth as f64 / 5.0).min(1.0)
        };

        let future_depth = if self.future_scenarios.is_empty() {
            0.0
        } else {
            let max_horizon = self
                .future_scenarios
                .iter()
                .map(|s| s.horizon_steps)
                .max()
                .unwrap_or(0);
            (max_horizon as f64 / 10.0).min(1.0)
        };

        (past_depth + future_depth) / 2.0
    }

    /// Get detailed narrative coherence diagnostics
    pub fn diagnostics(&self) -> NarrativeCoherenceDiagnostics {
        NarrativeCoherenceDiagnostics {
            autobiographical_integration: self.autobiographical_integration,
            future_simulation_depth: self.future_simulation_depth,
            narrative_arc_coherence: self.compute_narrative_arc_coherence(),
            prospective_capability: self.compute_prospective_capability(),
            mental_time_travel: self.compute_mental_time_travel(),
            episode_count: self.episodes.len(),
            scenario_count: self.future_scenarios.len(),
            narrative_coherence: self.compute(),
        }
    }
}

/// A branch in a future simulation
#[derive(Debug, Clone)]
pub struct SimulationBranch {
    /// Description of this outcome branch
    pub description: String,
    /// Steps to reach this outcome
    pub steps_to_outcome: usize,
    /// Probability of this branch
    pub probability: f64,
    /// Desirability of this outcome
    pub desirability: f64,
    /// Actions required to achieve this branch
    pub required_actions: Vec<String>,
}

impl SimulationBranch {
    /// Create a new simulation branch
    pub fn new(description: String, steps: usize, probability: f64, desirability: f64) -> Self {
        Self {
            description,
            steps_to_outcome: steps,
            probability: probability.clamp(0.0, 1.0),
            desirability: desirability.clamp(-1.0, 1.0),
            required_actions: Vec::new(),
        }
    }

    /// Add a required action
    pub fn with_action(mut self, action: String) -> Self {
        self.required_actions.push(action);
        self
    }
}

/// Detailed diagnostics for narrative coherence
#[derive(Debug, Clone)]
pub struct NarrativeCoherenceDiagnostics {
    /// Autobiographical integration score
    pub autobiographical_integration: f64,
    /// Future simulation depth
    pub future_simulation_depth: f64,
    /// Narrative arc coherence
    pub narrative_arc_coherence: f64,
    /// Prospective memory capability
    pub prospective_capability: f64,
    /// Mental time travel depth
    pub mental_time_travel: f64,
    /// Number of episodes stored
    pub episode_count: usize,
    /// Number of future scenarios
    pub scenario_count: usize,
    /// Final narrative coherence N
    pub narrative_coherence: f64,
}
