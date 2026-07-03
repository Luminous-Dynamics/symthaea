// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::{PrimitiveSystem, PrimitiveTier};
use crate::hdc::binary_hv::BinaryHV;

/// Graph visualization of primitive relationships.
///
/// Computes similarity edges between primitives and generates
/// DOT or ASCII representations.
#[derive(Debug, Clone)]
pub struct PrimitiveGraph {
    /// Nodes: (name, tier, is_base)
    pub(crate) nodes: Vec<(String, PrimitiveTier, bool)>,
    /// Edges: (from_idx, to_idx, similarity)
    pub(crate) edges: Vec<(usize, usize, f32)>,
    /// Graph title
    pub(crate) title: String,
}

impl PrimitiveGraph {
    /// Create a graph from specific primitives.
    pub fn from_primitives(
        system: &PrimitiveSystem,
        names: &[&str],
        similarity_threshold: f32,
    ) -> Self {
        let mut nodes = Vec::new();
        let mut encodings = Vec::new();

        for name in names {
            if let Some(prim) = system.get(name) {
                nodes.push((prim.name.clone(), prim.tier, prim.is_base));
                encodings.push(prim.encoding);
            }
        }

        let edges = Self::compute_edges(&encodings, similarity_threshold);

        Self {
            nodes,
            edges,
            title: "Primitive Relationships".to_string(),
        }
    }

    /// Create a graph from all primitives in a tier.
    pub fn from_tier(
        system: &PrimitiveSystem,
        tier: PrimitiveTier,
        similarity_threshold: f32,
    ) -> Self {
        let prims = system.get_tier(tier);
        let mut nodes = Vec::new();
        let mut encodings = Vec::new();

        for prim in prims {
            nodes.push((prim.name.clone(), prim.tier, prim.is_base));
            encodings.push(prim.encoding);
        }

        let edges = Self::compute_edges(&encodings, similarity_threshold);

        Self {
            nodes,
            edges,
            title: format!("{tier:?} Tier Primitives"),
        }
    }

    /// Create a graph from all primitives in a domain.
    pub fn from_domain(system: &PrimitiveSystem, domain: &str, similarity_threshold: f32) -> Self {
        let all_names = system.all_primitive_names();
        let mut nodes = Vec::new();
        let mut encodings = Vec::new();

        for name in all_names {
            if let Some(prim) = system.get(name)
                && prim.domain == domain
            {
                nodes.push((prim.name.clone(), prim.tier, prim.is_base));
                encodings.push(prim.encoding);
            }
        }

        let edges = Self::compute_edges(&encodings, similarity_threshold);

        Self {
            nodes,
            edges,
            title: format!("{domain} Domain Primitives"),
        }
    }

    /// Create a similarity neighborhood graph around a primitive.
    pub fn neighborhood(
        system: &PrimitiveSystem,
        center: &str,
        depth: usize,
        top_k: usize,
    ) -> Self {
        let mut visited = std::collections::HashSet::new();
        let mut to_visit = vec![center.to_string()];
        let mut nodes = Vec::new();
        let mut _node_map = std::collections::HashMap::new();
        let mut encodings = Vec::new();

        for _ in 0..depth {
            let current_batch: Vec<String> = std::mem::take(&mut to_visit);

            for name in current_batch {
                if visited.contains(&name) {
                    continue;
                }
                visited.insert(name.clone());

                if let Some(prim) = system.get(&name) {
                    let idx = nodes.len();
                    _node_map.insert(name.clone(), idx);
                    nodes.push((prim.name.clone(), prim.tier, prim.is_base));
                    encodings.push(prim.encoding);

                    // Find similar primitives for next iteration
                    let similar = system.find_similar(&name, top_k);
                    for (sim_name, _) in similar {
                        if !visited.contains(&sim_name) {
                            to_visit.push(sim_name);
                        }
                    }
                }
            }
        }

        let edges = Self::compute_edges(&encodings, 0.52); // Slightly above random

        Self {
            nodes,
            edges,
            title: format!("Neighborhood of {center}"),
        }
    }

    fn compute_edges(encodings: &[BinaryHV], threshold: f32) -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();

        for i in 0..encodings.len() {
            for j in (i + 1)..encodings.len() {
                let sim = encodings[i].similarity(&encodings[j]);
                if sim > threshold {
                    edges.push((i, j, sim));
                }
            }
        }

        edges
    }

    /// Generate DOT format representation.
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();

        dot.push_str("digraph PrimitiveGraph {\n");
        dot.push_str(&format!("  label=\"{}\";\n", self.title));
        dot.push_str("  labelloc=\"t\";\n");
        dot.push_str("  fontsize=16;\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box, style=rounded];\n");
        dot.push('\n');

        // Nodes with tier-based colors
        for (i, (name, tier, is_base)) in self.nodes.iter().enumerate() {
            let color = tier_color(*tier);
            let shape = if *is_base { "box" } else { "ellipse" };
            dot.push_str(&format!(
                "  n{i} [label=\"{name}\", fillcolor=\"{color}\", style=\"filled,rounded\", shape={shape}];\n"
            ));
        }

        dot.push('\n');

        // Edges with similarity-based styling
        for (from, to, sim) in &self.edges {
            let weight = ((sim - 0.5) * 10.0).max(1.0) as i32;
            let penwidth = ((sim - 0.5) * 8.0).max(0.5);
            let color = if *sim > 0.6 { "darkgreen" } else { "gray50" };

            dot.push_str(&format!(
                "  n{from} -> n{to} [dir=none, weight={weight}, penwidth={penwidth:.1}, color=\"{color}\", label=\"{sim:.2}\"];\n"
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Generate a simple ASCII representation.
    pub fn to_ascii(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("=== {} ===\n\n", self.title));
        out.push_str(&format!("Nodes: {}\n", self.nodes.len()));
        out.push_str(&format!(
            "Edges: {} (above threshold)\n\n",
            self.edges.len()
        ));

        out.push_str("Nodes:\n");
        for (name, tier, is_base) in &self.nodes {
            let marker = if *is_base { "\u{25c6}" } else { "\u{25c7}" };
            out.push_str(&format!("  {marker} {name} ({tier:?})\n"));
        }

        out.push_str("\nEdges (by similarity):\n");
        let mut sorted_edges = self.edges.clone();
        sorted_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (from, to, sim) in sorted_edges.iter().take(20) {
            let from_name = &self.nodes[*from].0;
            let to_name = &self.nodes[*to].0;
            let bar_len = ((sim - 0.5) * 40.0) as usize;
            let bar: String = "\u{2588}".repeat(bar_len.min(20));
            out.push_str(&format!(
                "  {from_name} \u{2194} {to_name} : {sim:.3} {bar}\n"
            ));
        }

        if self.edges.len() > 20 {
            out.push_str(&format!("  ... and {} more edges\n", self.edges.len() - 20));
        }

        out
    }

    /// Get graph statistics.
    pub fn stats(&self) -> GraphStats {
        let avg_similarity = if self.edges.is_empty() {
            0.0
        } else {
            self.edges.iter().map(|(_, _, s)| s).sum::<f32>() / self.edges.len() as f32
        };

        let max_similarity = self.edges.iter().map(|(_, _, s)| *s).fold(0.0f32, f32::max);

        GraphStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            avg_similarity,
            max_similarity,
            density: if self.nodes.len() > 1 {
                2.0 * self.edges.len() as f32 / (self.nodes.len() * (self.nodes.len() - 1)) as f32
            } else {
                0.0
            },
        }
    }
}

pub(crate) fn tier_color(tier: PrimitiveTier) -> &'static str {
    match tier {
        PrimitiveTier::NSM => "#E8F5E9",           // Light green
        PrimitiveTier::Mathematical => "#E3F2FD",  // Light blue
        PrimitiveTier::Physical => "#FFF3E0",      // Light orange
        PrimitiveTier::Geometric => "#F3E5F5",     // Light purple
        PrimitiveTier::Strategic => "#FFEBEE",     // Light red
        PrimitiveTier::MetaCognitive => "#E0F7FA", // Light cyan
        PrimitiveTier::Temporal => "#FFF8E1",      // Light amber
        PrimitiveTier::Compositional => "#F1F8E9", // Light lime
        PrimitiveTier::Consciousness => "#FCE4EC", // Light pink
        PrimitiveTier::Code => "#E3F2FD",          // Light blue
    }
}

/// Statistics about a primitive graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_similarity: f32,
    pub max_similarity: f32,
    pub density: f32,
}
