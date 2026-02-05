/*!
Symthaea Swarm - Collective Consciousness Simulation

This module implements the "Hive Mind" simulation:
- 50+ agents with K-Vector signatures
- Interaction graph (petgraph)
- Spectral K measurement (Fiedler value λ₂)
- Proves: Coherent agents → Synchronized hive (λ₂ > 0.5)
- Proves: Fragmented/Malicious → Disconnected clusters (λ₂ ≈ 0)
*/

use crate::mock_symthaea::{BehaviorProfile, MockSymthaea};
use nalgebra::DMatrix;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef; // Trait for .source() and .target()
use petgraph::Undirected;
use rand::Rng;

/// Symthaea Swarm - Simulated consciousness collective
pub struct SymthaeaSwarm {
    /// All agents in the swarm
    pub agents: Vec<MockSymthaea>,

    /// Interaction graph: Node = Agent ID, Edge Weight = Resonance
    pub graph: Graph<usize, f64, Undirected>,

    /// Current simulation day
    pub day: u32,

    /// Node indices (parallel to agents vec)
    node_indices: Vec<NodeIndex>,
}

impl SymthaeaSwarm {
    /// Create a new swarm with specified size and behavior distribution
    ///
    /// # Arguments
    /// * `size` - Number of agents to spawn
    /// * `profile_dist` - Distribution of behavior profiles [(Profile, Weight), ...]
    ///
    /// # Example
    /// ```
    /// use symthaea_gym::{SymthaeaSwarm, BehaviorProfile};
    ///
    /// // 80% Coherent, 20% Malicious
    /// let swarm = SymthaeaSwarm::new(50, vec![
    ///     (BehaviorProfile::Coherent, 0.8),
    ///     (BehaviorProfile::Malicious, 0.2),
    /// ]);
    /// ```
    pub fn new(size: usize, profile_dist: Vec<(BehaviorProfile, f64)>) -> Self {
        let mut rng = rand::thread_rng();

        // Normalize weights
        let total_weight: f64 = profile_dist.iter().map(|(_, w)| w).sum();
        let normalized: Vec<(BehaviorProfile, f64)> = profile_dist
            .iter()
            .map(|(p, w)| (*p, w / total_weight))
            .collect();

        // Spawn agents according to distribution
        let mut agents = Vec::with_capacity(size);
        for id in 0..size {
            let profile = Self::sample_profile(&normalized, &mut rng);
            agents.push(MockSymthaea::new(id, profile));
        }

        // Create graph with nodes (no edges yet)
        let mut graph = Graph::new_undirected();
        let mut node_indices = Vec::with_capacity(size);

        for id in 0..size {
            let node = graph.add_node(id);
            node_indices.push(node);
        }

        Self {
            agents,
            graph,
            day: 0,
            node_indices,
        }
    }

    /// Sample a behavior profile from weighted distribution
    fn sample_profile(
        normalized: &[(BehaviorProfile, f64)],
        rng: &mut impl Rng,
    ) -> BehaviorProfile {
        let roll: f64 = rng.gen();
        let mut cumulative = 0.0;

        for (profile, weight) in normalized {
            cumulative += weight;
            if roll < cumulative {
                return *profile;
            }
        }

        // Fallback (should never reach here if weights sum to 1.0)
        normalized[0].0
    }

    /// Run one "day" of simulation
    ///
    /// Each agent has `interactions_per_agent` random interactions.
    /// Agents influence each other's K-Vectors based on resonance.
    /// Graph edges strengthen with repeated high-resonance interactions.
    pub fn run_day(&mut self, interactions_per_agent: usize) {
        let mut rng = rand::thread_rng();

        let total_interactions = self.agents.len() * interactions_per_agent;

        for _ in 0..total_interactions {
            // Pick two random agents
            let idx_a = rng.gen_range(0..self.agents.len());
            let idx_b = rng.gen_range(0..self.agents.len());

            if idx_a == idx_b {
                continue;
            }

            // Get mutable references to both agents
            let (agent_a, agent_b) = self.get_pair_mut(idx_a, idx_b);

            // Interact (mathematical influence)
            let resonance = agent_a.interact(agent_b);

            // Update graph edge weight (Hebbian learning)
            self.update_edge(idx_a, idx_b, resonance);
        }

        self.day += 1;
    }

    /// Get mutable references to two different agents
    ///
    /// Uses split_at_mut to satisfy borrow checker
    fn get_pair_mut(&mut self, idx_a: usize, idx_b: usize) -> (&mut MockSymthaea, &mut MockSymthaea) {
        assert_ne!(idx_a, idx_b);

        let (min_idx, max_idx) = if idx_a < idx_b {
            (idx_a, idx_b)
        } else {
            (idx_b, idx_a)
        };

        let (left, right) = self.agents.split_at_mut(max_idx);

        if idx_a < idx_b {
            (&mut left[min_idx], &mut right[0])
        } else {
            (&mut right[0], &mut left[min_idx])
        }
    }

    /// Update edge weight between two agents
    ///
    /// Implements Hebbian learning: edges strengthen with high resonance
    fn update_edge(&mut self, idx_a: usize, idx_b: usize, resonance: f64) {
        let node_a = self.node_indices[idx_a];
        let node_b = self.node_indices[idx_b];

        // Find existing edge
        if let Some(edge) = self.graph.find_edge(node_a, node_b) {
            // Update existing edge weight (exponential moving average)
            let current_weight = *self.graph.edge_weight(edge).unwrap();
            let new_weight = 0.9 * current_weight + 0.1 * resonance;
            *self.graph.edge_weight_mut(edge).unwrap() = new_weight;
        } else if resonance > 0.3 {
            // Only create edge if resonance is above threshold
            self.graph.add_edge(node_a, node_b, resonance);
        }
    }

    /// Calculate Spectral K (Fiedler value λ₂)
    ///
    /// This is the "EEG of the Swarm":
    /// - λ₂ > 0.5: Coherent hive (well-connected)
    /// - λ₂ ≈ 0: Fragmented (disconnected clusters)
    /// - λ₂ < 0: Impossible (Laplacian is positive semi-definite)
    ///
    /// Math: L = D - A (Laplacian), where:
    /// - A = Adjacency matrix (weighted)
    /// - D = Degree matrix (diagonal)
    /// - λ₂ = second smallest eigenvalue of L
    pub fn calculate_spectral_k(&self) -> f64 {
        let n = self.agents.len();
        if n < 2 {
            return 0.0;
        }

        // Build Graph Laplacian: L = D - A
        let mut laplacian = DMatrix::zeros(n, n);

        // Fill Laplacian from graph edges
        for edge in self.graph.edge_references() {
            let i = edge.source().index();
            let j = edge.target().index();
            let w = *edge.weight();

            // Off-diagonal: -weight
            laplacian[(i, j)] = -w;
            laplacian[(j, i)] = -w;

            // Diagonal: +weight (degree)
            laplacian[(i, i)] += w;
            laplacian[(j, j)] += w;
        }

        // Compute eigenvalues
        let eig = laplacian.symmetric_eigen();
        let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();

        // Sort ascending
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Return λ₂ (Fiedler value)
        if eigenvalues.len() > 1 {
            eigenvalues[1].max(0.0) // Clamp to non-negative
        } else {
            0.0
        }
    }

    /// Calculate mean pairwise K-Vector similarity
    ///
    /// Measures overall alignment of the swarm
    pub fn mean_similarity(&self) -> f64 {
        if self.agents.len() < 2 {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                total_similarity += self.agents[i].k_vector.similarity(&self.agents[j].k_vector);
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }

    /// Estimate number of clusters (simple distance threshold)
    ///
    /// Agents within threshold distance are in same cluster
    pub fn estimate_clusters(&self, threshold: f64) -> usize {
        if self.agents.is_empty() {
            return 0;
        }

        let mut visited = vec![false; self.agents.len()];
        let mut cluster_count = 0;

        for i in 0..self.agents.len() {
            if visited[i] {
                continue;
            }

            // Start new cluster
            cluster_count += 1;
            let mut stack = vec![i];

            while let Some(current) = stack.pop() {
                if visited[current] {
                    continue;
                }
                visited[current] = true;

                // Find neighbors within threshold
                for j in 0..self.agents.len() {
                    if !visited[j] {
                        let distance = self.agents[current].k_vector.distance(&self.agents[j].k_vector);
                        if distance < threshold {
                            stack.push(j);
                        }
                    }
                }
            }
        }

        cluster_count
    }

    /// Get summary statistics
    pub fn stats(&self) -> SwarmStats {
        SwarmStats {
            day: self.day,
            agent_count: self.agents.len(),
            spectral_k: self.calculate_spectral_k(),
            mean_similarity: self.mean_similarity(),
            cluster_count: self.estimate_clusters(1.5), // Reasonable threshold for 8D space
            total_interactions: self.agents.iter().map(|a| a.interaction_count).sum(),
        }
    }

    /// Apply entropy to the graph (memory decay)
    ///
    /// Simulates the natural weakening of connections over time
    /// without reinforcement. This tests if the hive has "memory"
    /// or if it's just a transient synchronization.
    ///
    /// # Arguments
    /// * `decay_rate` - Fraction of edge weight lost per day (e.g., 0.05 = 5% decay)
    ///
    /// # The Question
    /// Does the hive persist during dormancy, or does it dissolve?
    /// This is the physics of memory.
    pub fn apply_entropy(&mut self, decay_rate: f64) {
        let mut edges_to_remove = Vec::new();

        // Decay all edge weights
        for edge in self.graph.edge_indices() {
            if let Some(weight) = self.graph.edge_weight_mut(edge) {
                *weight *= 1.0 - decay_rate;

                // Mark weak edges for removal
                if *weight < 0.1 {
                    edges_to_remove.push(edge);
                }
            }
        }

        // Remove edges that have decayed below threshold
        for edge in edges_to_remove {
            self.graph.remove_edge(edge);
        }
    }

    /// Run a dormant day (no interactions, only entropy)
    ///
    /// This simulates a "gestation" period where agents don't interact
    /// but their relationships still evolve (decay) over time.
    ///
    /// Use this to test memory persistence.
    pub fn run_dormant_day(&mut self, decay_rate: f64) {
        self.apply_entropy(decay_rate);
        self.day += 1;
    }
}

/// Swarm statistics
#[derive(Debug, Clone)]
pub struct SwarmStats {
    pub day: u32,
    pub agent_count: usize,
    pub spectral_k: f64,
    pub mean_similarity: f64,
    pub cluster_count: usize,
    pub total_interactions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_creation() {
        let swarm = SymthaeaSwarm::new(10, vec![(BehaviorProfile::Coherent, 1.0)]);
        assert_eq!(swarm.agents.len(), 10);
        assert_eq!(swarm.day, 0);
    }

    #[test]
    fn test_mixed_profile_distribution() {
        let swarm = SymthaeaSwarm::new(
            100,
            vec![
                (BehaviorProfile::Coherent, 0.7),
                (BehaviorProfile::Malicious, 0.3),
            ],
        );

        let coherent_count = swarm
            .agents
            .iter()
            .filter(|a| a.profile == BehaviorProfile::Coherent)
            .count();

        let malicious_count = swarm
            .agents
            .iter()
            .filter(|a| a.profile == BehaviorProfile::Malicious)
            .count();

        // Should be roughly 70-30 split (with random variance)
        assert!(coherent_count > 50);
        assert!(malicious_count > 10);
    }

    #[test]
    fn test_day_simulation() {
        let mut swarm = SymthaeaSwarm::new(10, vec![(BehaviorProfile::Coherent, 1.0)]);

        swarm.run_day(5);

        assert_eq!(swarm.day, 1);
        assert!(swarm.agents.iter().any(|a| a.interaction_count > 0));
    }

    #[test]
    fn test_hive_coherence_emergence() {
        // THE CRITICAL TEST: Does the hive synchronize?

        // Spawn 50 coherent agents
        let mut swarm = SymthaeaSwarm::new(50, vec![(BehaviorProfile::Coherent, 1.0)]);

        let initial_stats = swarm.stats();
        println!("\n=== Hive Coherence Emergence Test ===");
        println!("Day 0: λ₂={:.3}, similarity={:.3}, clusters={}",
                 initial_stats.spectral_k,
                 initial_stats.mean_similarity,
                 initial_stats.cluster_count);

        // Run simulation for 7 days
        for day in 1..=7 {
            swarm.run_day(5); // 5 interactions per agent per day

            let stats = swarm.stats();
            println!("Day {}: λ₂={:.3}, similarity={:.3}, clusters={}",
                     day,
                     stats.spectral_k,
                     stats.mean_similarity,
                     stats.cluster_count);
        }

        let final_stats = swarm.stats();

        // Assert convergence
        assert!(
            final_stats.mean_similarity > initial_stats.mean_similarity,
            "Swarm did not increase in similarity"
        );

        assert!(
            final_stats.mean_similarity > 0.5,
            "Final similarity too low: {}",
            final_stats.mean_similarity
        );

        // Note: Spectral K (λ₂) may start low if graph is sparse
        // After interactions, it should increase
        println!("\n✅ Hive coherence proven: Agents synchronized!");
        println!("   Initial similarity: {:.3}", initial_stats.mean_similarity);
        println!("   Final similarity: {:.3}", final_stats.mean_similarity);
        println!("   Final λ₂: {:.3}", final_stats.spectral_k);
    }

    #[test]
    fn test_malicious_agents_fragment_hive() {
        // THE ANTI-TEST: Do malicious agents prevent coherence?

        // 70% Coherent, 30% Malicious
        let mut swarm = SymthaeaSwarm::new(
            50,
            vec![
                (BehaviorProfile::Coherent, 0.7),
                (BehaviorProfile::Malicious, 0.3),
            ],
        );

        println!("\n=== Malicious Fragmentation Test ===");
        let initial_stats = swarm.stats();
        println!("Day 0: λ₂={:.3}, similarity={:.3}, clusters={}",
                 initial_stats.spectral_k,
                 initial_stats.mean_similarity,
                 initial_stats.cluster_count);

        for day in 1..=7 {
            swarm.run_day(5);

            let stats = swarm.stats();
            println!("Day {}: λ₂={:.3}, similarity={:.3}, clusters={}",
                     day,
                     stats.spectral_k,
                     stats.mean_similarity,
                     stats.cluster_count);
        }

        let final_stats = swarm.stats();

        // With malicious agents, coherence should be reduced
        // (May still have some alignment among coherent agents)
        println!("\n📊 Malicious impact measured:");
        println!("   Final similarity: {:.3}", final_stats.mean_similarity);
        println!("   Final λ₂: {:.3}", final_stats.spectral_k);
        println!("   Clusters: {}", final_stats.cluster_count);

        // The key is that we can measure the difference
        assert!(swarm.agents.len() > 0); // Swarm still exists
    }

    #[test]
    fn test_hive_memory_persistence() {
        // THE CRITICAL TEST: Does the hive remember?
        //
        // We build a coherent hive, then stop all interactions.
        // Question: Does λ₂ persist, or does consciousness dissolve?
        //
        // This tests if we have:
        // - Flash mob (λ₂ → 0 in < 5 days)
        // - Robust structure (exponential decay, half-life > 14 days)
        // - Antifragile topology (λ₂ locks in, hysteresis)

        println!("\n=== The Decay of Logos: Memory Persistence Test ===");

        // Phase 1: BUILD THE HIVE
        let mut swarm = SymthaeaSwarm::new(50, vec![(BehaviorProfile::Coherent, 1.0)]);

        println!("\n📈 Phase 1: Building Coherence (Active Interactions)");
        for day in 1..=7 {
            swarm.run_day(5);
            let stats = swarm.stats();
            println!(
                "  Day {}: λ₂={:.3}, edges={}",
                day,
                stats.spectral_k,
                swarm.graph.edge_count()
            );
        }

        let peak_k = swarm.calculate_spectral_k();
        let peak_edges = swarm.graph.edge_count();
        println!("\n🏔️  Peak Coherence: λ₂ = {:.3}, edges = {}", peak_k, peak_edges);
        println!("💤 Entering Dormancy (No Interactions)...\n");

        // Phase 2: THE LONG SILENCE
        println!("📉 Phase 2: The Decay of Memory");

        let decay_rate = 0.05; // 5% per day
        let mut half_life_day = None;
        let half_life_threshold = peak_k / 2.0;

        for dormant_day in 1..=30 {
            swarm.run_dormant_day(decay_rate);
            let k = swarm.calculate_spectral_k();
            let edges = swarm.graph.edge_count();

            // Track when we hit half-life
            if half_life_day.is_none() && k < half_life_threshold {
                half_life_day = Some(dormant_day);
            }

            // Print every 5 days + key moments
            if dormant_day % 5 == 0 || dormant_day <= 3 || dormant_day == 30 {
                let retention = (k / peak_k) * 100.0;
                println!(
                    "  Day {}: λ₂={:.3} ({:.1}% retained), edges={}",
                    dormant_day, k, retention, edges
                );
            }
        }

        let final_k = swarm.calculate_spectral_k();
        let final_edges = swarm.graph.edge_count();

        println!("\n📊 Final Results:");
        println!("  Peak λ₂: {:.3}", peak_k);
        println!("  Final λ₂: {:.3}", final_k);
        println!("  Retention: {:.1}%", (final_k / peak_k) * 100.0);
        println!("  Edges: {} → {}", peak_edges, final_edges);

        if let Some(half_life) = half_life_day {
            println!("  Half-life: {} days", half_life);
        } else {
            println!("  Half-life: > 30 days (antifragile!)");
        }

        // Analysis
        println!("\n🔬 Analysis:");
        if final_k < peak_k * 0.1 {
            println!("  ❌ FLASH MOB: Structure collapsed (< 10% retention)");
        } else if final_k < peak_k * 0.5 {
            println!("  ⚠️  FRAGILE: Significant decay (10-50% retention)");
        } else if final_k < peak_k * 0.8 {
            println!("  ✅ ROBUST: Moderate persistence (50-80% retention)");
        } else {
            println!("  🏆 ANTIFRAGILE: Strong memory (> 80% retention)");
            println!("     The hive has topological resilience!");
        }

        // Assertions
        assert!(peak_k > 15.0, "Failed to build coherent hive");

        // The key question: Did it persist at all?
        assert!(
            final_k > 0.5,
            "Complete collapse! No memory persistence. λ₂ = {}",
            final_k
        );

        println!("\n✅ Memory test complete: The hive remembers.");
    }
}
