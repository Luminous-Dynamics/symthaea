// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flake Dependency Graph HDC Encoding
//!
//! Encodes flake dependency graphs (inputs, follows, overrides) as hypervectors
//! so the causal mind can reason about flake-level dependencies and cascading
//! updates. Each input becomes a bound vector (input_name ⊗ input_type), and
//! the full graph is a bundled representation of all edges.

use super::codebook::NixCodebook;
use symthaea_core::hdc::ContinuousHV;

/// A parsed flake input for encoding.
#[derive(Debug, Clone)]
pub struct FlakeInput {
    /// Input name (e.g., "nixpkgs", "home-manager").
    pub name: String,
    /// Input type (e.g., "github", "path", "git").
    pub input_type: String,
    /// Optional: which other input this follows.
    pub follows: Option<String>,
}

/// A flake dependency edge (input → output relationship).
#[derive(Debug, Clone)]
pub struct FlakeDependencyEdge {
    /// Source input name.
    pub from: String,
    /// Target (either another input or an output).
    pub to: String,
    /// Relationship type ("follows", "overrides", "depends").
    pub relation: String,
}

/// Encoded flake dependency graph.
#[derive(Debug, Clone)]
pub struct EncodedFlakeGraph {
    /// Overall graph vector (bundled from all edges).
    pub graph_vector: ContinuousHV,
    /// Per-input vectors.
    pub input_vectors: Vec<(String, ContinuousHV)>,
    /// Per-edge vectors.
    pub edge_vectors: Vec<(String, String, ContinuousHV)>,
}

/// Encodes flake dependency structures into HDC space.
pub struct FlakeGraphEncoder<'a> {
    codebook: &'a mut NixCodebook,
}

impl<'a> FlakeGraphEncoder<'a> {
    /// Create a new flake graph encoder.
    pub fn new(codebook: &'a mut NixCodebook) -> Self {
        Self { codebook }
    }

    /// Encode a single flake input as an HDC vector.
    ///
    /// Result = input_name ⊗ input_type_role
    pub fn encode_input(&mut self, input: &FlakeInput) -> ContinuousHV {
        let name_hv = self.codebook.get_or_create(&input.name).clone();
        let type_hv = self.codebook.get_or_create(&input.input_type).clone();
        let role = self.codebook.package_role.clone(); // Reuse package role for input names

        // input = name ⊗ role + type
        let bound = name_hv.bind(&role);
        let refs = [&bound, &type_hv];
        let weights = [0.7, 0.3];
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Encode a dependency edge between two flake components.
    ///
    /// Result = from ⊗ relation_role ⊗ to
    pub fn encode_edge(&mut self, edge: &FlakeDependencyEdge) -> ContinuousHV {
        let from_hv = self.codebook.get_or_create(&edge.from).clone();
        let to_hv = self.codebook.get_or_create(&edge.to).clone();
        let relation_hv = self.codebook.get_or_create(&edge.relation).clone();

        // edge = from ⊗ relation ⊗ to
        let intermediate = from_hv.bind(&relation_hv);
        intermediate.bind(&to_hv)
    }

    /// Encode a complete flake dependency graph.
    ///
    /// The graph vector is a weighted bundle of all input and edge vectors.
    pub fn encode_graph(
        &mut self,
        inputs: &[FlakeInput],
        edges: &[FlakeDependencyEdge],
    ) -> EncodedFlakeGraph {
        // Encode individual inputs
        let input_vectors: Vec<(String, ContinuousHV)> = inputs
            .iter()
            .map(|input| (input.name.clone(), self.encode_input(input)))
            .collect();

        // Encode edges
        let edge_vectors: Vec<(String, String, ContinuousHV)> = edges
            .iter()
            .map(|edge| (edge.from.clone(), edge.to.clone(), self.encode_edge(edge)))
            .collect();

        // Bundle everything into a graph vector
        let all_hvs: Vec<&ContinuousHV> = input_vectors
            .iter()
            .map(|(_, hv)| hv)
            .chain(edge_vectors.iter().map(|(_, _, hv)| hv))
            .collect();

        let graph_vector = if all_hvs.is_empty() {
            ContinuousHV::zero(self.codebook.dim())
        } else {
            let weight = 1.0 / all_hvs.len() as f32;
            let weights: Vec<f32> = vec![weight; all_hvs.len()];
            ContinuousHV::weighted_bundle(&all_hvs, &weights)
        };

        EncodedFlakeGraph {
            graph_vector,
            input_vectors,
            edge_vectors,
        }
    }

    /// Predict which inputs would be affected by updating a specific input.
    ///
    /// Uses the encoded graph to find inputs whose edge vectors reference
    /// the updated input, indicating dependency relationships. Inputs
    /// connected via "follows" or "depends" edges are considered affected.
    pub fn predict_cascade(
        &mut self,
        updated_input: &str,
        graph: &EncodedFlakeGraph,
        threshold: f32,
    ) -> Vec<(String, f32)> {
        // Use the encoded graph vector for the updated input (if available)
        // rather than the raw basis vector, since graph vectors live in the
        // encoded space that includes binding with roles.
        let updated_hv = graph
            .input_vectors
            .iter()
            .find(|(name, _)| name == updated_input)
            .map(|(_, hv)| hv.clone())
            .unwrap_or_else(|| self.codebook.get_or_create(updated_input).clone());

        let mut affected: Vec<(String, f32)> = graph
            .input_vectors
            .iter()
            .filter(|(name, _)| name != updated_input)
            .map(|(name, hv)| (name.clone(), updated_hv.similarity(hv)))
            .filter(|(_, sim)| *sim > threshold)
            .collect();

        // Check edges that reference the updated input (either direction).
        // An edge from X to updated_input means X depends on it (cascade target).
        // An edge from updated_input to Y means Y is a dependency (also affected).
        for (from, to, edge_hv) in &graph.edge_vectors {
            let target = if to == updated_input {
                Some(from.as_str())
            } else if from == updated_input {
                Some(to.as_str())
            } else {
                None
            };

            if let Some(target_name) = target {
                if target_name != updated_input
                    && !affected.iter().any(|(name, _)| name == target_name)
                {
                    let sim = updated_hv.similarity(edge_hv).abs();
                    let effective_sim = sim.max(threshold + 0.01); // Ensure structurally-connected inputs appear
                    affected.push((target_name.to_string(), effective_sim));
                }
            }
        }

        affected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        affected
    }
}

/// Parse a flake.lock JSON into FlakeInput and FlakeDependencyEdge structures.
pub fn parse_flake_lock(
    lock_json: &str,
) -> Result<(Vec<FlakeInput>, Vec<FlakeDependencyEdge>), String> {
    let value: serde_json::Value =
        serde_json::from_str(lock_json).map_err(|e| format!("Failed to parse flake.lock: {e}"))?;

    let nodes = value["nodes"]
        .as_object()
        .ok_or_else(|| "Missing 'nodes' in flake.lock".to_string())?;

    let mut inputs = Vec::new();
    let mut edges = Vec::new();

    // Get root inputs to know which nodes are direct dependencies
    let root_inputs: std::collections::HashSet<String> = value["nodes"]["root"]["inputs"]
        .as_object()
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();

    for (name, node) in nodes {
        if name == "root" {
            continue;
        }

        let input_type = node["locked"]["type"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        let follows = node["inputs"].as_object().and_then(|inputs_map| {
            inputs_map.iter().find_map(|(_, v)| {
                // If the value is a string (not an array), it's a "follows" reference
                v.as_str().map(|s| s.to_string())
            })
        });

        inputs.push(FlakeInput {
            name: name.clone(),
            input_type,
            follows: follows.clone(),
        });

        // Create dependency edges
        if root_inputs.contains(name) {
            edges.push(FlakeDependencyEdge {
                from: "root".to_string(),
                to: name.clone(),
                relation: "depends".to_string(),
            });
        }

        if let Some(ref followed) = follows {
            edges.push(FlakeDependencyEdge {
                from: name.clone(),
                to: followed.clone(),
                relation: "follows".to_string(),
            });
        }
    }

    Ok((inputs, edges))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lock_json() -> &'static str {
        r#"{
  "nodes": {
    "nixpkgs": {
      "locked": {
        "type": "github",
        "owner": "NixOS",
        "repo": "nixpkgs"
      }
    },
    "home-manager": {
      "locked": {
        "type": "github",
        "owner": "nix-community",
        "repo": "home-manager"
      },
      "inputs": {
        "nixpkgs": "nixpkgs"
      }
    },
    "flake-utils": {
      "locked": {
        "type": "github",
        "owner": "numtide",
        "repo": "flake-utils"
      }
    },
    "root": {
      "inputs": {
        "nixpkgs": "nixpkgs",
        "home-manager": "home-manager",
        "flake-utils": "flake-utils"
      }
    }
  },
  "version": 7
}"#
    }

    #[test]
    fn test_parse_flake_lock() {
        let (inputs, edges) = parse_flake_lock(sample_lock_json()).unwrap();
        assert_eq!(inputs.len(), 3);
        assert!(inputs.iter().any(|i| i.name == "nixpkgs"));
        assert!(inputs.iter().any(|i| i.name == "home-manager"));

        // home-manager follows nixpkgs
        let hm = inputs.iter().find(|i| i.name == "home-manager").unwrap();
        assert_eq!(hm.follows, Some("nixpkgs".to_string()));

        // Should have dependency edges from root
        assert!(edges.iter().any(|e| e.from == "root" && e.to == "nixpkgs"));
        // Should have follows edge
        assert!(
            edges
                .iter()
                .any(|e| e.from == "home-manager" && e.relation == "follows")
        );
    }

    #[test]
    fn test_encode_flake_input() {
        let mut codebook = NixCodebook::new();
        let mut encoder = FlakeGraphEncoder::new(&mut codebook);

        let input = FlakeInput {
            name: "nixpkgs".to_string(),
            input_type: "github".to_string(),
            follows: None,
        };
        let hv = encoder.encode_input(&input);
        // Encoding should produce a non-zero vector
        let norm: f32 = hv.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "Encoded input should be non-zero");
    }

    #[test]
    fn test_encode_graph() {
        let mut codebook = NixCodebook::new();
        let (inputs, edges) = parse_flake_lock(sample_lock_json()).unwrap();

        let mut encoder = FlakeGraphEncoder::new(&mut codebook);
        let graph = encoder.encode_graph(&inputs, &edges);

        assert_eq!(graph.input_vectors.len(), 3);
        assert!(!graph.edge_vectors.is_empty());
        // Graph vector should be non-zero
        let norm: f32 = graph
            .graph_vector
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_different_inputs_distinguishable() {
        let mut codebook = NixCodebook::new();
        let mut encoder = FlakeGraphEncoder::new(&mut codebook);

        let nixpkgs = FlakeInput {
            name: "nixpkgs".to_string(),
            input_type: "github".to_string(),
            follows: None,
        };
        let hm = FlakeInput {
            name: "home-manager".to_string(),
            input_type: "github".to_string(),
            follows: Some("nixpkgs".to_string()),
        };

        let hv1 = encoder.encode_input(&nixpkgs);
        let hv2 = encoder.encode_input(&hm);

        let sim = hv1.similarity(&hv2).abs();
        assert!(
            sim < 0.5,
            "Different inputs should be distinguishable: sim={}",
            sim
        );
    }

    #[test]
    fn test_predict_cascade() {
        let mut codebook = NixCodebook::new();
        let (inputs, edges) = parse_flake_lock(sample_lock_json()).unwrap();

        let mut encoder = FlakeGraphEncoder::new(&mut codebook);
        let graph = encoder.encode_graph(&inputs, &edges);

        // Updating nixpkgs should potentially affect home-manager (which follows nixpkgs)
        let cascade = encoder.predict_cascade("nixpkgs", &graph, 0.0);
        assert!(
            !cascade.is_empty(),
            "Updating nixpkgs should cascade to dependents"
        );
    }
}
