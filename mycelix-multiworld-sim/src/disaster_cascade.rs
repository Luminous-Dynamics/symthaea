// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Causal disaster cascade graph (Fix 6).
//!
//! Models how one disaster triggers secondary events — a Carrington event
//! knocks out power, which cascades into thermal failure, which cascades
//! into O2 failure. Uses BFS propagation with per-edge probability and
//! optional tick delays.

use serde::{Deserialize, Serialize};

/// A single edge in the cascade graph: "if trigger fires, consequence
/// fires with probability `probability` after `delay_ticks` ticks."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeEdge {
    pub trigger: String,
    pub consequence: String,
    pub probability: f64,
    pub delay_ticks: u32,
}

/// Build the default cascade graph (~20 edges covering the major
/// failure chains in the sim's disaster taxonomy.
pub fn build_default_cascade_graph() -> Vec<CascadeEdge> {
    vec![
        // Solar → infrastructure
        CascadeEdge {
            trigger: "carrington".into(),
            consequence: "power_failure".into(),
            probability: 0.8,
            delay_ticks: 0,
        },
        CascadeEdge {
            trigger: "carrington".into(),
            consequence: "comms_failure".into(),
            probability: 0.6,
            delay_ticks: 0,
        },
        CascadeEdge {
            trigger: "x_class_flare".into(),
            consequence: "power_failure".into(),
            probability: 0.3,
            delay_ticks: 0,
        },
        // Power → life support
        CascadeEdge {
            trigger: "power_failure".into(),
            consequence: "thermal_failure".into(),
            probability: 0.6,
            delay_ticks: 0,
        },
        CascadeEdge {
            trigger: "power_failure".into(),
            consequence: "water_failure".into(),
            probability: 0.4,
            delay_ticks: 1,
        },
        CascadeEdge {
            trigger: "power_failure".into(),
            consequence: "hydroponic_failure".into(),
            probability: 0.5,
            delay_ticks: 1,
        },
        // Thermal → atmosphere
        CascadeEdge {
            trigger: "thermal_failure".into(),
            consequence: "o2_failure".into(),
            probability: 0.4,
            delay_ticks: 1,
        },
        CascadeEdge {
            trigger: "thermal_failure".into(),
            consequence: "co2_failure".into(),
            probability: 0.3,
            delay_ticks: 1,
        },
        // Seismic → structural
        CascadeEdge {
            trigger: "moonquake".into(),
            consequence: "seal_failure".into(),
            probability: 0.3,
            delay_ticks: 0,
        },
        CascadeEdge {
            trigger: "seal_failure".into(),
            consequence: "o2_loss".into(),
            probability: 0.5,
            delay_ticks: 0,
        },
        CascadeEdge {
            trigger: "seal_failure".into(),
            consequence: "pressure_loss".into(),
            probability: 0.4,
            delay_ticks: 0,
        },
        // Europa-specific
        CascadeEdge {
            trigger: "tidal_quake".into(),
            consequence: "seal_failure".into(),
            probability: 0.25,
            delay_ticks: 0,
        },
        CascadeEdge {
            trigger: "ice_instability".into(),
            consequence: "seal_failure".into(),
            probability: 0.6,
            delay_ticks: 0,
        },
        CascadeEdge {
            trigger: "radiation_surge".into(),
            consequence: "comms_failure".into(),
            probability: 0.4,
            delay_ticks: 0,
        },
        // Titan-specific
        CascadeEdge {
            trigger: "heating_failure".into(),
            consequence: "embrittlement_spike".into(),
            probability: 0.5,
            delay_ticks: 1,
        },
        CascadeEdge {
            trigger: "methane_storm".into(),
            consequence: "seal_failure".into(),
            probability: 0.2,
            delay_ticks: 0,
        },
        // Hydroponic → food
        CascadeEdge {
            trigger: "hydroponic_failure".into(),
            consequence: "food_shortage".into(),
            probability: 0.7,
            delay_ticks: 2,
        },
        // Comms → governance
        CascadeEdge {
            trigger: "comms_failure".into(),
            consequence: "governance_disruption".into(),
            probability: 0.3,
            delay_ticks: 1,
        },
        // Water → health
        CascadeEdge {
            trigger: "water_failure".into(),
            consequence: "health_crisis".into(),
            probability: 0.5,
            delay_ticks: 2,
        },
        // Compound
        CascadeEdge {
            trigger: "food_shortage".into(),
            consequence: "social_unrest".into(),
            probability: 0.4,
            delay_ticks: 3,
        },
    ]
}

/// BFS through the cascade graph starting from `triggered`.
///
/// Each edge fires independently with its own probability, tested
/// against `rng_val` (a single f64 from the caller). For deterministic
/// testing, pass 0.0 to fire all edges; pass 1.0 to fire none.
///
/// Only immediate (delay_ticks == 0) consequences are returned.
/// Delayed consequences would need a queue in the caller (future work).
pub fn propagate_cascade(triggered: &str, graph: &[CascadeEdge], rng_val: f64) -> Vec<String> {
    let mut fired: Vec<String> = Vec::new();
    let mut queue: Vec<String> = vec![triggered.to_string()];
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    visited.insert(triggered.to_string());

    while let Some(current) = queue.pop() {
        for edge in graph
            .iter()
            .filter(|e| e.trigger == current && e.delay_ticks == 0)
        {
            if !visited.contains(&edge.consequence) && rng_val < edge.probability {
                visited.insert(edge.consequence.clone());
                fired.push(edge.consequence.clone());
                queue.push(edge.consequence.clone());
            }
        }
    }

    fired
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_carrington_triggers_power_cascade() {
        let graph = build_default_cascade_graph();
        // rng_val = 0.0 means all edges fire (0.0 < probability is always true)
        let consequences = propagate_cascade("carrington", &graph, 0.0);
        assert!(
            consequences.contains(&"power_failure".to_string()),
            "Carrington should trigger power failure, got: {:?}",
            consequences
        );
        assert!(
            consequences.contains(&"thermal_failure".to_string()),
            "Power failure should cascade to thermal failure, got: {:?}",
            consequences
        );
    }

    #[test]
    fn test_no_cascade_with_high_rng() {
        let graph = build_default_cascade_graph();
        // rng_val = 1.0 means no edges fire (1.0 < probability is never true)
        let consequences = propagate_cascade("carrington", &graph, 1.0);
        assert!(
            consequences.is_empty(),
            "High rng_val should prevent all cascades, got: {:?}",
            consequences
        );
    }

    #[test]
    fn test_moonquake_seal_cascade() {
        let graph = build_default_cascade_graph();
        let consequences = propagate_cascade("moonquake", &graph, 0.0);
        assert!(
            consequences.contains(&"seal_failure".to_string()),
            "Moonquake should trigger seal failure"
        );
        assert!(
            consequences.contains(&"o2_loss".to_string()),
            "Seal failure should cascade to O2 loss"
        );
    }

    #[test]
    fn test_cascade_graph_has_expected_edges() {
        let graph = build_default_cascade_graph();
        assert!(
            graph.len() >= 15,
            "Should have at least 15 cascade edges, got {}",
            graph.len()
        );
        // All probabilities in valid range
        for edge in &graph {
            assert!(
                edge.probability > 0.0 && edge.probability <= 1.0,
                "Edge {} -> {} has invalid probability {}",
                edge.trigger,
                edge.consequence,
                edge.probability
            );
        }
    }

    #[test]
    fn test_no_infinite_loop() {
        // Even with a cycle in the graph, visited set prevents infinite loops
        let graph = vec![
            CascadeEdge {
                trigger: "a".into(),
                consequence: "b".into(),
                probability: 1.0,
                delay_ticks: 0,
            },
            CascadeEdge {
                trigger: "b".into(),
                consequence: "a".into(),
                probability: 1.0,
                delay_ticks: 0,
            },
        ];
        let consequences = propagate_cascade("a", &graph, 0.0);
        assert_eq!(consequences, vec!["b".to_string()]);
    }
}
