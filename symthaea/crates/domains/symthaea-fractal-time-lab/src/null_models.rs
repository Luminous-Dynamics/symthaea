// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use petgraph::graph::UnGraph;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

pub struct NullModels;

impl NullModels {
    pub fn random_spectrum(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut spectrum: Vec<f64> = (0..n).map(|_| rng.gen_range(-4.0..4.0)).collect();
        spectrum.sort_by(|a, b| a.total_cmp(b));
        spectrum
    }

    pub fn jittered_spectrum(source: &[f64], jitter: f64, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let jitter = jitter.abs();

        let mut spectrum: Vec<f64> = source
            .iter()
            .map(|&x| x + rng.gen_range(-jitter..jitter))
            .collect();

        spectrum.shuffle(&mut rng);
        spectrum.sort_by(|a, b| a.total_cmp(b));
        spectrum
    }

    pub fn sinusoidal_spectrum(n: usize, amplitude: f64) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }

        let mut spectrum: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                amplitude * (2.0 * std::f64::consts::PI * t).sin()
            })
            .collect();

        spectrum.sort_by(|a, b| a.total_cmp(b));
        spectrum
    }

    pub fn smooth_spectrum(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let phase = rng.gen_range(0.0..2.0 * std::f64::consts::PI);

        let mut spectrum: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n.max(1) as f64;
                4.0 * (phase + 2.0 * std::f64::consts::PI * t).sin()
            })
            .collect();

        spectrum.sort_by(|a, b| a.total_cmp(b));
        spectrum
    }

    pub fn random_graph(n: usize, p: f64, seed: u64) -> UnGraph<(), ()> {
        let mut graph = UnGraph::default();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let p = p.clamp(0.0, 1.0);

        for i in 0..n {
            for j in (i + 1)..n {
                if rng.gen_bool(p) {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }

        graph
    }

    pub fn hierarchical_graph(module_count: usize, module_size: usize) -> UnGraph<(), ()> {
        let mut graph = UnGraph::<(), ()>::default();

        if module_count == 0 || module_size == 0 {
            return graph;
        }

        let mut modules = Vec::new();

        for _ in 0..module_count {
            let module_nodes: Vec<_> = (0..module_size).map(|_| graph.add_node(())).collect();

            for i in 0..module_nodes.len() {
                for j in (i + 1)..module_nodes.len() {
                    graph.add_edge(module_nodes[i], module_nodes[j], ());
                }
            }

            modules.push(module_nodes);
        }

        for module in 0..(module_count - 1) {
            graph.add_edge(modules[module][0], modules[module + 1][0], ());
        }

        graph
    }

    pub fn path_graph(n: usize) -> UnGraph<(), ()> {
        let mut graph = UnGraph::<(), ()>::default();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

        for i in 0..n.saturating_sub(1) {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        graph
    }

    pub fn cycle_graph(n: usize) -> UnGraph<(), ()> {
        let mut graph = Self::path_graph(n);
        if n > 2 {
            let nodes: Vec<_> = graph.node_indices().collect();
            graph.add_edge(nodes[0], nodes[n - 1], ());
        }
        graph
    }

    pub fn binary_tree(depth: usize) -> UnGraph<(), ()> {
        let mut graph = UnGraph::<(), ()>::default();
        if depth == 0 {
            return graph;
        }

        let node_count = (1usize << depth).saturating_sub(1);
        let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();

        for i in 0..node_count {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < node_count {
                graph.add_edge(nodes[i], nodes[left], ());
            }
            if right < node_count {
                graph.add_edge(nodes[i], nodes[right], ());
            }
        }

        graph
    }

    pub fn damped_oscillator(n: usize, decay: f64) -> Vec<f64> {
        let mut signal = Vec::with_capacity(n);
        let mut val = 1.0;
        let decay = decay.clamp(0.0, 1.0);

        for _ in 0..n {
            signal.push(val);
            val *= -decay;
        }

        signal
    }

    pub fn random_signal(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    pub fn noisy_period_two(n: usize, noise: f64, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = noise.abs();

        (0..n)
            .map(|i| {
                let base = if i % 2 == 0 { 1.0 } else { -1.0 };
                base + rng.gen_range(-noise..noise)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeded_spectrum_is_reproducible() {
        assert_eq!(
            NullModels::random_spectrum(8, 42),
            NullModels::random_spectrum(8, 42)
        );
    }

    #[test]
    fn test_seeded_graph_is_reproducible_by_counts() {
        let g1 = NullModels::random_graph(10, 0.3, 42);
        let g2 = NullModels::random_graph(10, 0.3, 42);

        assert_eq!(g1.node_count(), g2.node_count());
        assert_eq!(g1.edge_count(), g2.edge_count());
    }

    #[test]
    fn test_hierarchical_graph_has_nodes() {
        let graph = NullModels::hierarchical_graph(3, 4);
        assert_eq!(graph.node_count(), 12);
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_path_graph_edges() {
        let graph = NullModels::path_graph(5);
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4);
    }

    #[test]
    fn test_cycle_graph_edges() {
        let graph = NullModels::cycle_graph(5);
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 5);
    }

    #[test]
    fn test_binary_tree_size() {
        let graph = NullModels::binary_tree(4);
        assert_eq!(graph.node_count(), 15);
        assert_eq!(graph.edge_count(), 14);
    }
}
