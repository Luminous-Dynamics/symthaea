//! NixOS Causal Graph
//!
//! Uses causal discovery to analyze NixOS configurations:
//! - Detect root causes of build failures
//! - Understand option dependencies
//! - Predict side effects of changes
//! - Recommend fixes based on causal structure

use crate::traits::{CausalDiscoveryEngine, CausalDirection};
use std::collections::{HashMap, HashSet};

/// A NixOS configuration variable with observed values
#[derive(Debug, Clone)]
pub struct ConfigVariable {
    pub name: String,
    pub path: String,
    pub values: Vec<f64>,
}

/// A causal edge between configuration variables
#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub from: String,
    pub to: String,
    pub direction: CausalDirection,
    pub confidence: f64,
}

/// Root cause analysis result
#[derive(Debug, Clone)]
pub struct RootCauseAnalysis {
    pub symptom: String,
    pub root_causes: Vec<RootCause>,
    pub causal_chain: Vec<CausalEdge>,
}

/// A potential root cause
#[derive(Debug, Clone)]
pub struct RootCause {
    pub variable: String,
    pub confidence: f64,
    pub explanation: String,
}

/// Side effect prediction
#[derive(Debug, Clone)]
pub struct SideEffectPrediction {
    pub affected_variable: String,
    pub direction: String,
    pub confidence: f64,
}

/// The NixOS Causal Analyzer
pub struct NixCausalGraph {
    engine: CausalDiscoveryEngine,
    causal_graph: HashMap<(String, String), CausalEdge>,
    observations: HashMap<String, Vec<f64>>,
}

impl NixCausalGraph {
    /// Create a new analyzer
    pub fn new(seed: u64) -> Self {
        Self {
            engine: CausalDiscoveryEngine::new(seed),
            causal_graph: HashMap::new(),
            observations: HashMap::new(),
        }
    }

    /// Record an observation of a configuration variable
    pub fn observe(&mut self, variable: &str, value: f64) {
        self.observations
            .entry(variable.to_string())
            .or_default()
            .push(value);
    }

    /// Record a batch of observations
    pub fn observe_batch(&mut self, variable: &str, values: &[f64]) {
        self.observations
            .entry(variable.to_string())
            .or_default()
            .extend_from_slice(values);
    }

    /// Discover causal structure between observed variables
    pub fn discover_structure(&mut self) -> Vec<CausalEdge> {
        let variables: Vec<String> = self.observations.keys().cloned().collect();
        let mut edges = Vec::new();

        for i in 0..variables.len() {
            for j in (i + 1)..variables.len() {
                let var_a = &variables[i];
                let var_b = &variables[j];

                if let (Some(obs_a), Some(obs_b)) = (
                    self.observations.get(var_a),
                    self.observations.get(var_b),
                ) {
                    let min_len = obs_a.len().min(obs_b.len());
                    if min_len < 20 {
                        continue;
                    }

                    let x: Vec<f64> = obs_a.iter().take(min_len).cloned().collect();
                    let y: Vec<f64> = obs_b.iter().take(min_len).cloned().collect();

                    let (direction, confidence) = self.engine.predict_with_confidence(&x, &y);

                    let edge = CausalEdge {
                        from: if direction == CausalDirection::Forward {
                            var_a.clone()
                        } else {
                            var_b.clone()
                        },
                        to: if direction == CausalDirection::Forward {
                            var_b.clone()
                        } else {
                            var_a.clone()
                        },
                        direction,
                        confidence,
                    };

                    self.causal_graph.insert((edge.from.clone(), edge.to.clone()), edge.clone());
                    edges.push(edge);
                }
            }
        }

        edges
    }

    /// Analyze root causes of a symptom
    pub fn analyze_root_causes(&self, symptom: &str) -> RootCauseAnalysis {
        let mut root_causes = Vec::new();
        let mut causal_chain = Vec::new();

        let mut visited = HashSet::new();
        let mut to_visit = vec![symptom.to_string()];

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            for ((_from, to), edge) in &self.causal_graph {
                if to == &current {
                    causal_chain.push(edge.clone());
                    to_visit.push(edge.from.clone());
                }
            }
        }

        let has_incoming: HashSet<String> = causal_chain.iter().map(|e| e.to.clone()).collect();

        for edge in &causal_chain {
            if !has_incoming.contains(&edge.from) {
                root_causes.push(RootCause {
                    variable: edge.from.clone(),
                    confidence: edge.confidence,
                    explanation: format!(
                        "{} causally influences {} with confidence {:.1}%",
                        edge.from, symptom, edge.confidence * 100.0
                    ),
                });
            }
        }

        root_causes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        RootCauseAnalysis {
            symptom: symptom.to_string(),
            root_causes,
            causal_chain,
        }
    }

    /// Predict side effects of changing a variable
    pub fn predict_side_effects(&self, variable: &str) -> Vec<SideEffectPrediction> {
        let mut effects = Vec::new();

        let mut visited = HashSet::new();
        let mut to_visit = vec![variable.to_string()];

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            for ((from, to), edge) in &self.causal_graph {
                if from == &current && to != variable {
                    effects.push(SideEffectPrediction {
                        affected_variable: to.clone(),
                        direction: "change".to_string(),
                        confidence: edge.confidence,
                    });
                    to_visit.push(to.clone());
                }
            }
        }

        effects.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        effects
    }

    /// Generate fix recommendations based on causal analysis
    pub fn recommend_fixes(&self, symptom: &str) -> Vec<String> {
        let analysis = self.analyze_root_causes(symptom);
        let mut recommendations = Vec::new();

        for cause in &analysis.root_causes {
            if cause.confidence > 0.6 {
                recommendations.push(format!(
                    "Consider adjusting '{}' - it has a {:.0}% likelihood of being the root cause",
                    cause.variable,
                    cause.confidence * 100.0
                ));
            }
        }

        if recommendations.is_empty() {
            recommendations.push(
                "Insufficient causal evidence to make strong recommendations. Try collecting more observations.".to_string()
            );
        }

        recommendations
    }

    /// Encode a NixOS option path as a numeric value for causal analysis
    pub fn encode_option_value(value: &str) -> f64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        let hash = hasher.finish();

        (hash as f64) / (u64::MAX as f64)
    }

    /// Encode a boolean as numeric
    pub fn encode_bool(value: bool) -> f64 {
        if value { 1.0 } else { 0.0 }
    }
}

/// Common NixOS causal patterns
pub struct NixOSCausalPatterns;

impl NixOSCausalPatterns {
    /// Known causal relationships in NixOS
    pub fn known_patterns() -> Vec<(&'static str, &'static str, &'static str)> {
        vec![
            ("hardware.opengl.enable", "services.xserver.enable", "enables"),
            ("services.xserver.enable", "services.displayManager", "requires"),
            ("networking.firewall.enable", "services.*", "blocks"),
            ("boot.kernelPackages", "hardware.nvidia.package", "determines"),
            ("nixpkgs.config.allowUnfree", "packages.*", "enables"),
            ("hardware.nvidia.modesetting.enable", "services.xserver.videoDrivers", "affects"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nixos_analyzer() {
        let mut analyzer = NixCausalGraph::new(42);

        for i in 0..50 {
            let x = i as f64;
            analyzer.observe("boot.kernelPackages", x);
            analyzer.observe("hardware.nvidia.package", 2.0 * x + 0.5);
            analyzer.observe("services.xserver.enable", if x > 25.0 { 1.0 } else { 0.0 });
        }

        let edges = analyzer.discover_structure();
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_side_effect_prediction() {
        let mut analyzer = NixCausalGraph::new(42);

        for i in 0..100 {
            let x = i as f64 / 10.0;
            analyzer.observe("config.A", x);
            analyzer.observe("config.B", 2.0 * x + (i % 3) as f64);
            analyzer.observe("config.C", 1.5 * x + (i % 5) as f64);
        }

        analyzer.discover_structure();

        let effects = analyzer.predict_side_effects("config.A");
        println!("Side effects of changing config.A: {:?}", effects);
    }
}
