// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Causal Graph
//!
//! Uses causal discovery to analyze NixOS configurations:
//! - Detect root causes of build failures
//! - Understand option dependencies
//! - Predict side effects of changes
//! - Recommend fixes based on causal structure

use crate::traits::{CausalDirection, CausalDiscoveryEngine};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// Serializable snapshot of the causal graph for persistence.
#[derive(Debug, Serialize, Deserialize)]
struct CausalGraphSnapshot {
    version: u32,
    edges: Vec<CausalEdge>,
    observations: HashMap<String, Vec<f64>>,
}

/// A NixOS configuration variable with observed values
#[derive(Debug, Clone)]
pub struct ConfigVariable {
    pub name: String,
    pub path: String,
    pub values: Vec<f64>,
}

/// A causal edge between configuration variables
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Secondary index: from_node → set of to_nodes for O(1) outgoing edge lookup.
    from_index: HashMap<String, HashSet<String>>,
    observations: HashMap<String, Vec<f64>>,
    /// Hebbian learning rate for edge strengthening/weakening.
    learning_rate: f64,
    /// Minimum confidence before an edge is pruned.
    min_confidence: f64,
}

impl NixCausalGraph {
    /// Create a new analyzer
    pub fn new(seed: u64) -> Self {
        Self {
            engine: CausalDiscoveryEngine::new(seed),
            causal_graph: HashMap::new(),
            from_index: HashMap::new(),
            observations: HashMap::new(),
            learning_rate: 0.1,
            min_confidence: 0.05,
        }
    }

    /// Insert an edge and update the from_index.
    fn insert_edge(&mut self, from: String, to: String, edge: CausalEdge) {
        self.from_index
            .entry(from.clone())
            .or_default()
            .insert(to.clone());
        self.causal_graph.insert((from, to), edge);
    }

    /// Merge causal edges from another graph.
    pub fn merge(&mut self, other: &Self) {
        for (k, edge) in &other.causal_graph {
            if let Some(existing) = self.causal_graph.get_mut(k) {
                existing.confidence = (existing.confidence + edge.confidence) / 2.0;
            } else {
                self.causal_graph.insert(k.clone(), edge.clone());
                self.from_index
                    .entry(k.0.clone())
                    .or_default()
                    .insert(k.1.clone());
            }
        }
    }

    /// Set the Hebbian learning rate (default 0.1).
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate.clamp(0.001, 1.0);
        self
    }

    /// Dynamically update the learning rate of the causal graph.
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.001, 1.0);
    }

    /// Access the current causal learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Observe an execution outcome and update causal edges via Hebbian learning.
    ///
    /// When action A is followed by outcome effects \[B, C, ...\]:
    /// - Strengthen A→B and A→C edges (they co-occurred)
    /// - Weaken A→X edges where X was predicted but didn't occur
    ///
    /// This makes the causal graph self-improving over time.
    pub fn observe_outcome(
        &mut self,
        action: &str,
        observed_effects: &[&str],
        predicted_effects: &[&str],
    ) {
        let observed_set: HashSet<&str> = observed_effects.iter().copied().collect();
        let predicted_set: HashSet<&str> = predicted_effects.iter().copied().collect();

        // Strengthen edges for effects that actually occurred
        for &effect in &observed_set {
            let key = (action.to_string(), effect.to_string());
            let entry = self.causal_graph.entry(key).or_insert_with(|| {
                self.from_index
                    .entry(action.to_string())
                    .or_default()
                    .insert(effect.to_string());
                CausalEdge {
                    from: action.to_string(),
                    to: effect.to_string(),
                    direction: CausalDirection::Forward,
                    confidence: 0.3, // Start with moderate confidence for newly discovered edges
                }
            });
            // Hebbian strengthening: Δw = η * (1 - w) — bounded growth
            entry.confidence += self.learning_rate * (1.0 - entry.confidence);
            entry.confidence = entry.confidence.clamp(0.0, 1.0);
        }

        // Weaken edges for predicted effects that didn't materialize
        for &predicted in &predicted_set {
            if !observed_set.contains(predicted) {
                let key = (action.to_string(), predicted.to_string());
                if let Some(edge) = self.causal_graph.get_mut(&key) {
                    // Anti-Hebbian: Δw = -η * w — bounded decay
                    edge.confidence -= self.learning_rate * edge.confidence;
                    edge.confidence = edge.confidence.max(0.0);
                }
            }
        }

        // Prune edges below minimum confidence threshold
        let min_conf = self.min_confidence;
        self.causal_graph.retain(|(from, to), edge| {
            if edge.confidence >= min_conf {
                true
            } else {
                if let Some(targets) = self.from_index.get_mut(from) {
                    targets.remove(to);
                }
                false
            }
        });
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

                if let (Some(obs_a), Some(obs_b)) =
                    (self.observations.get(var_a), self.observations.get(var_b))
                {
                    let min_len = obs_a.len().min(obs_b.len());
                    if min_len < 20 {
                        continue;
                    }

                    let x: Vec<f64> = obs_a.iter().take(min_len).cloned().collect();
                    let y: Vec<f64> = obs_b.iter().take(min_len).cloned().collect();

                    let (direction, confidence) = self.engine.predict_with_confidence(&x, &y);

                    let (from, to) = if direction == CausalDirection::Forward {
                        (var_a.clone(), var_b.clone())
                    } else {
                        (var_b.clone(), var_a.clone())
                    };

                    let edge = CausalEdge {
                        from: from.clone(),
                        to: to.clone(),
                        direction,
                        confidence,
                    };

                    self.insert_edge(from, to, edge.clone());
                    edges.push(edge);
                }
            }
        }

        edges
    }

    /// Analyze root causes of a symptom
    #[tracing::instrument(skip(self), fields(edges = self.causal_graph.len()))]
    pub fn analyze_root_causes(&self, symptom: &str) -> RootCauseAnalysis {
        let mut root_causes = Vec::new();
        let mut causal_chain = Vec::new();

        let mut visited = HashSet::new();
        let mut to_visit = vec![symptom.to_string()];

        // Path-multiplied transitive confidence (hierarchical causal mapping)
        let mut path_confidence = std::collections::HashMap::new();
        path_confidence.insert(symptom.to_string(), 1.0f64);

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            let current_conf = *path_confidence.get(&current).unwrap_or(&0.0);

            for ((from, to), edge) in &self.causal_graph {
                if to == &current {
                    causal_chain.push(edge.clone());

                    let new_conf = current_conf * (edge.confidence as f64);
                    let entry = path_confidence.entry(from.clone()).or_insert(0.0);
                    if new_conf > *entry {
                        *entry = new_conf;
                    }
                    to_visit.push(edge.from.clone());
                }
            }
        }

        let has_incoming: HashSet<String> = causal_chain.iter().map(|e| e.to.clone()).collect();

        for edge in &causal_chain {
            if !has_incoming.contains(&edge.from) {
                let trans_conf = *path_confidence
                    .get(&edge.from)
                    .unwrap_or(&(edge.confidence as f64));
                root_causes.push(RootCause {
                    variable: edge.from.clone(),
                    confidence: trans_conf,
                    explanation: format!(
                        "{} causally influences {} (transitive path confidence: {:.1}%)",
                        edge.from,
                        symptom,
                        trans_conf * 100.0
                    ),
                });
            }
        }

        root_causes.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        RootCauseAnalysis {
            symptom: symptom.to_string(),
            root_causes,
            causal_chain,
        }
    }

    /// Predict side effects of changing a variable.
    ///
    /// Uses the `from_index` for O(1) outgoing-edge lookup per node
    /// instead of iterating all edges in the graph.
    pub fn predict_side_effects(&self, variable: &str) -> Vec<SideEffectPrediction> {
        let mut effects = Vec::new();

        let mut visited = HashSet::new();
        let mut to_visit = vec![variable.to_string()];

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(targets) = self.from_index.get(&current) {
                for to in targets {
                    if to != variable
                        && let Some(edge) = self.causal_graph.get(&(current.clone(), to.clone()))
                    {
                        effects.push(SideEffectPrediction {
                            affected_variable: to.clone(),
                            direction: "change".to_string(),
                            confidence: edge.confidence,
                        });
                        to_visit.push(to.clone());
                    }
                }
            }
        }

        effects.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        let hash = hasher.finish();

        (hash as f64) / (u64::MAX as f64)
    }

    /// Encode a boolean as numeric
    pub fn encode_bool(value: bool) -> f64 {
        if value { 1.0 } else { 0.0 }
    }

    /// Add a structural causal edge from known NixOS knowledge.
    ///
    /// Unlike `discover_structure()` which requires observation data, this
    /// injects edges with a specified confidence from domain expertise.
    pub fn add_structural_edge(&mut self, from: &str, to: &str, confidence: f64) {
        let edge = CausalEdge {
            from: from.to_string(),
            to: to.to_string(),
            direction: CausalDirection::Forward,
            confidence,
        };
        self.insert_edge(from.to_string(), to.to_string(), edge);
    }

    /// Save the causal graph to a JSON file for persistence across sessions.
    #[cfg(not(target_arch = "wasm32"))]
    ///
    /// Saves edges and observations so the graph can be restored without
    /// re-discovering structure from scratch.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let snapshot = CausalGraphSnapshot {
            version: 1,
            edges: self.causal_graph.values().cloned().collect(),
            observations: self.observations.clone(),
        };
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Failed to serialize causal graph: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write causal graph to {}: {}", path.display(), e))
    }

    /// Load a previously saved causal graph, merging edges into this instance.
    #[cfg(not(target_arch = "wasm32"))]
    ///
    /// Existing edges are preserved; loaded edges are added with their saved
    /// confidence values. Observations are merged (appended).
    pub fn load(&mut self, path: &Path) -> Result<usize, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read causal graph from {}: {}", path.display(), e))?;
        let snapshot: CausalGraphSnapshot = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize causal graph: {e}"))?;

        let mut loaded = 0;
        for edge in snapshot.edges {
            let key = (edge.from.clone(), edge.to.clone());
            if let std::collections::hash_map::Entry::Vacant(e) = self.causal_graph.entry(key) {
                self.from_index
                    .entry(edge.from.clone())
                    .or_default()
                    .insert(edge.to.clone());
                e.insert(edge);
                loaded += 1;
            }
        }

        for (var, values) in snapshot.observations {
            self.observations.entry(var).or_default().extend(values);
        }

        Ok(loaded)
    }

    /// Number of edges in the causal graph.
    pub fn edge_count(&self) -> usize {
        self.causal_graph.len()
    }

    /// Apply a global decay to all edges in the causal graph to simulate forgetting/pruning.
    pub fn decay_all(&mut self, factor: f64) {
        for edge in self.causal_graph.values_mut() {
            edge.confidence *= factor;
        }

        let min_conf = self.min_confidence;
        self.causal_graph.retain(|(from, to), edge| {
            if edge.confidence >= min_conf {
                true
            } else {
                if let Some(targets) = self.from_index.get_mut(from) {
                    targets.remove(to);
                }
                false
            }
        });
    }

    /// Return the top-N strongest causal edges, sorted by descending confidence.
    pub fn top_edges(&self, n: usize) -> Vec<CausalEdge> {
        let mut edges: Vec<CausalEdge> = self.causal_graph.values().cloned().collect();
        edges.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        edges.truncate(n);
        edges
    }

    /// Get the confidence of a specific edge, or None if it doesn't exist.
    pub fn edge_confidence(&self, from: &str, to: &str) -> Option<f64> {
        self.causal_graph
            .get(&(from.to_string(), to.to_string()))
            .map(|e| e.confidence)
    }

    /// Bootstrap the causal graph from parsed NixOS option schemas.
    ///
    /// Discovers structural edges by analyzing option paths:
    /// - `services.X.enable` → `services.X.*` (enable gates sub-options)
    /// - `services.X.package` ← `nixpkgs.config.allowUnfree` (if unfree)
    /// - `networking.firewall.enable` → `networking.firewall.allowed*`
    /// - Sibling `.enable` options under the same parent are potential conflicts
    #[cfg(feature = "native")]
    pub fn bootstrap_from_schemas(
        &mut self,
        schemas: &[crate::parser::option_schema::NixOptionSchema],
    ) {
        // Index enable options by parent path
        let mut enable_opts: HashMap<String, Vec<&str>> = HashMap::new();

        for schema in schemas {
            let parts = schema.path_components();

            // enable gates sub-options in the same service
            if schema.is_enable_option() && parts.len() >= 3 {
                let parent = parts[..parts.len() - 1].join(".");
                for other in schemas {
                    if other.name != schema.name && other.name.starts_with(&parent) {
                        self.add_structural_edge(&schema.name, &other.name, 0.85);
                    }
                }
            }

            // Track enable options per module prefix for conflict detection
            if schema.is_enable_option() && parts.len() >= 2 {
                let module = parts[0].to_string();
                enable_opts.entry(module).or_default().push(&schema.name);
            }

            // Package lists depend on allowUnfree
            if schema.is_package_list() {
                self.add_structural_edge("nixpkgs.config.allowUnfree", &schema.name, 0.6);
            }

            // Service options that reference firewall need firewall ports
            if schema.name.contains("openFirewall") {
                let parent = parts[..2.min(parts.len())].join(".");
                self.add_structural_edge(&format!("{parent}.enable"), &schema.name, 0.9);
                self.add_structural_edge(&schema.name, "networking.firewall.allowedTCPPorts", 0.7);
            }
        }

        // Detect potential conflicts between boot loaders
        for opts in enable_opts.values() {
            if opts.len() >= 2 {
                for i in 0..opts.len() {
                    for j in (i + 1)..opts.len() {
                        // Only flag as conflicts if they share the same immediate parent
                        let a_parts: Vec<&str> = opts[i].split('.').collect();
                        let b_parts: Vec<&str> = opts[j].split('.').collect();
                        if a_parts.len() >= 3
                            && b_parts.len() >= 3
                            && a_parts[..a_parts.len() - 1] == b_parts[..b_parts.len() - 1]
                        {
                            self.add_structural_edge(opts[i], opts[j], 0.5);
                        }
                    }
                }
            }
        }
    }
}

/// Common NixOS causal patterns
pub struct NixOSCausalPatterns;

impl NixOSCausalPatterns {
    /// Known causal relationships in NixOS (~200 patterns).
    ///
    /// Format: (cause, effect, relationship_type)
    pub fn known_patterns() -> Vec<(&'static str, &'static str, &'static str)> {
        let mut patterns = Vec::with_capacity(210);

        // ── Boot & Init (~20) ──────────────────────────────────────────
        patterns.extend_from_slice(&[
            (
                "boot.loader.grub.enable",
                "boot.loader.systemd-boot.enable",
                "conflicts",
            ),
            (
                "boot.loader.systemd-boot.enable",
                "boot.loader.grub.enable",
                "conflicts",
            ),
            (
                "boot.loader.grub.device",
                "boot.loader.grub.enable",
                "requires",
            ),
            (
                "boot.loader.grub.efiSupport",
                "boot.loader.efi.canTouchEfiVariables",
                "enables",
            ),
            (
                "boot.loader.efi.efiSysMountPoint",
                "boot.loader.systemd-boot.enable",
                "configures",
            ),
            (
                "boot.initrd.availableKernelModules",
                "boot.initrd.kernelModules",
                "provides",
            ),
            (
                "boot.initrd.kernelModules",
                "hardware.enableAllFirmware",
                "requires",
            ),
            (
                "boot.initrd.luks.devices",
                "boot.initrd.kernelModules",
                "requires",
            ),
            (
                "boot.initrd.luks.devices",
                "boot.initrd.availableKernelModules",
                "requires",
            ),
            (
                "boot.kernelPackages",
                "boot.extraModulePackages",
                "determines",
            ),
            ("boot.kernelPackages", "boot.kernelModules", "provides"),
            ("boot.kernelParams", "boot.kernelPackages", "configures"),
            (
                "boot.kernelModules",
                "hardware.enableAllFirmware",
                "supplements",
            ),
            (
                "boot.supportedFilesystems",
                "boot.initrd.supportedFilesystems",
                "enables",
            ),
            ("boot.tmp.useTmpfs", "boot.tmp.tmpfsSize", "enables"),
            (
                "boot.binfmt.emulatedSystems",
                "boot.binfmt.registrations",
                "configures",
            ),
            (
                "boot.plymouth.enable",
                "boot.initrd.systemd.enable",
                "requires",
            ),
            (
                "boot.loader.timeout",
                "boot.loader.systemd-boot.enable",
                "configures",
            ),
            (
                "boot.initrd.systemd.enable",
                "boot.initrd.services",
                "enables",
            ),
            ("boot.isContainer", "boot.loader.grub.enable", "disables"),
        ]);

        // ── Display & GPU (~30) ────────────────────────────────────────
        patterns.extend_from_slice(&[
            (
                "hardware.opengl.enable",
                "services.xserver.enable",
                "enables",
            ),
            (
                "services.xserver.enable",
                "services.xserver.displayManager",
                "requires",
            ),
            (
                "services.xserver.enable",
                "services.xserver.desktopManager",
                "enables",
            ),
            (
                "services.xserver.enable",
                "services.xserver.windowManager",
                "enables",
            ),
            (
                "services.xserver.displayManager.gdm.enable",
                "services.xserver.enable",
                "requires",
            ),
            (
                "services.xserver.displayManager.sddm.enable",
                "services.xserver.enable",
                "requires",
            ),
            (
                "services.xserver.displayManager.lightdm.enable",
                "services.xserver.enable",
                "requires",
            ),
            (
                "services.xserver.displayManager.gdm.enable",
                "services.xserver.displayManager.sddm.enable",
                "conflicts",
            ),
            (
                "services.xserver.displayManager.gdm.enable",
                "services.xserver.displayManager.lightdm.enable",
                "conflicts",
            ),
            (
                "services.xserver.displayManager.sddm.enable",
                "services.xserver.displayManager.lightdm.enable",
                "conflicts",
            ),
            (
                "services.xserver.desktopManager.gnome.enable",
                "services.xserver.displayManager.gdm.enable",
                "recommends",
            ),
            (
                "services.xserver.desktopManager.plasma5.enable",
                "services.xserver.displayManager.sddm.enable",
                "recommends",
            ),
            (
                "hardware.nvidia.modesetting.enable",
                "services.xserver.videoDrivers",
                "affects",
            ),
            (
                "hardware.nvidia.package",
                "hardware.nvidia.modesetting.enable",
                "requires",
            ),
            (
                "boot.kernelPackages",
                "hardware.nvidia.package",
                "determines",
            ),
            (
                "hardware.nvidia.open",
                "hardware.nvidia.package",
                "configures",
            ),
            (
                "hardware.nvidia.powerManagement.enable",
                "hardware.nvidia.modesetting.enable",
                "requires",
            ),
            (
                "hardware.nvidia.prime.offload.enable",
                "hardware.nvidia.prime.intelBusId",
                "requires",
            ),
            (
                "hardware.nvidia.prime.offload.enable",
                "hardware.nvidia.prime.nvidiaBusId",
                "requires",
            ),
            (
                "hardware.nvidia.prime.sync.enable",
                "hardware.nvidia.prime.offload.enable",
                "conflicts",
            ),
            (
                "services.xserver.videoDrivers",
                "hardware.opengl.driSupport",
                "enables",
            ),
            (
                "hardware.opengl.driSupport32Bit",
                "hardware.opengl.enable",
                "requires",
            ),
            (
                "hardware.opengl.extraPackages",
                "hardware.opengl.enable",
                "requires",
            ),
            (
                "services.xserver.desktopManager.gnome.enable",
                "services.xserver.enable",
                "requires",
            ),
            (
                "services.xserver.desktopManager.plasma5.enable",
                "services.xserver.enable",
                "requires",
            ),
            (
                "services.xserver.desktopManager.xfce.enable",
                "services.xserver.enable",
                "requires",
            ),
            (
                "programs.sway.enable",
                "services.xserver.enable",
                "conflicts",
            ),
            ("programs.sway.enable", "security.polkit.enable", "requires"),
            (
                "services.xserver.displayManager.gdm.wayland",
                "services.xserver.displayManager.gdm.enable",
                "requires",
            ),
            (
                "services.xserver.xkb.layout",
                "services.xserver.enable",
                "requires",
            ),
        ]);

        // ── Networking (~25) ───────────────────────────────────────────
        patterns.extend_from_slice(&[
            (
                "networking.networkmanager.enable",
                "networking.wireless.enable",
                "conflicts",
            ),
            (
                "networking.wireless.enable",
                "networking.wireless.userControlled.enable",
                "enables",
            ),
            (
                "networking.networkmanager.enable",
                "networking.useDHCP",
                "overrides",
            ),
            (
                "networking.firewall.enable",
                "networking.firewall.allowedTCPPorts",
                "gates",
            ),
            (
                "networking.firewall.enable",
                "networking.firewall.allowedUDPPorts",
                "gates",
            ),
            (
                "networking.firewall.enable",
                "networking.firewall.allowedTCPPortRanges",
                "gates",
            ),
            (
                "networking.firewall.enable",
                "services.openssh.enable",
                "blocks",
            ),
            (
                "networking.firewall.allowedTCPPorts",
                "services.openssh.enable",
                "enables",
            ),
            ("networking.hostName", "networking.domain", "configures"),
            (
                "networking.nameservers",
                "networking.resolvconf.enable",
                "configures",
            ),
            ("networking.useDHCP", "networking.interfaces", "overrides"),
            (
                "networking.nat.enable",
                "networking.nat.internalInterfaces",
                "enables",
            ),
            (
                "networking.nat.enable",
                "networking.nat.externalInterface",
                "requires",
            ),
            (
                "networking.wireguard.enable",
                "networking.wireguard.interfaces",
                "enables",
            ),
            (
                "networking.wireguard.interfaces",
                "networking.firewall.allowedUDPPorts",
                "requires",
            ),
            (
                "networking.wg-quick.interfaces",
                "networking.wireguard.interfaces",
                "alternative",
            ),
            (
                "networking.proxy.default",
                "networking.proxy.httpProxy",
                "provides",
            ),
            ("networking.bridges", "networking.interfaces", "requires"),
            ("networking.vlans", "networking.interfaces", "requires"),
            ("networking.bonds", "networking.interfaces", "requires"),
            (
                "networking.nftables.enable",
                "networking.firewall.enable",
                "replaces",
            ),
            (
                "services.resolved.enable",
                "networking.resolvconf.enable",
                "conflicts",
            ),
            (
                "services.dnsmasq.enable",
                "networking.resolvconf.enable",
                "configures",
            ),
            (
                "networking.enableIPv6",
                "networking.firewall.allowedTCPPorts",
                "affects",
            ),
            (
                "networking.stealth",
                "networking.firewall.enable",
                "requires",
            ),
        ]);

        // ── Services (~40) ─────────────────────────────────────────────
        patterns.extend_from_slice(&[
            // Web servers
            (
                "services.nginx.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            (
                "services.nginx.enable",
                "services.nginx.virtualHosts",
                "enables",
            ),
            (
                "services.nginx.virtualHosts",
                "security.acme.certs",
                "requires",
            ),
            (
                "services.caddy.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            (
                "services.apache.enable",
                "services.nginx.enable",
                "conflicts",
            ),
            // Databases
            (
                "services.postgresql.enable",
                "services.postgresql.package",
                "requires",
            ),
            (
                "services.postgresql.enable",
                "services.postgresql.dataDir",
                "configures",
            ),
            (
                "services.postgresql.enable",
                "services.postgresql.authentication",
                "enables",
            ),
            (
                "services.mysql.enable",
                "services.mysql.package",
                "requires",
            ),
            (
                "services.redis.servers",
                "services.redis.package",
                "requires",
            ),
            (
                "services.mongodb.enable",
                "services.mongodb.package",
                "requires",
            ),
            // Docker & containers
            (
                "virtualisation.docker.enable",
                "users.extraGroups.docker",
                "requires",
            ),
            (
                "virtualisation.docker.enable",
                "networking.firewall.trustedInterfaces",
                "recommends",
            ),
            (
                "virtualisation.docker.rootless.enable",
                "virtualisation.docker.enable",
                "alternative",
            ),
            (
                "virtualisation.podman.enable",
                "virtualisation.docker.enable",
                "alternative",
            ),
            (
                "virtualisation.oci-containers.containers",
                "virtualisation.docker.enable",
                "requires",
            ),
            // SSH & security
            (
                "services.openssh.enable",
                "services.openssh.settings.PermitRootLogin",
                "enables",
            ),
            (
                "services.openssh.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            (
                "services.fail2ban.enable",
                "services.openssh.enable",
                "protects",
            ),
            // System services
            ("services.ntp.enable", "services.chrony.enable", "conflicts"),
            (
                "services.chrony.enable",
                "services.timesyncd.enable",
                "conflicts",
            ),
            (
                "services.timesyncd.enable",
                "services.ntp.enable",
                "conflicts",
            ),
            (
                "services.cron.enable",
                "services.cron.systemCronJobs",
                "enables",
            ),
            (
                "services.logrotate.enable",
                "services.journald.extraConfig",
                "supplements",
            ),
            (
                "services.printing.enable",
                "services.printing.drivers",
                "enables",
            ),
            (
                "services.avahi.enable",
                "services.avahi.nssmdns4",
                "enables",
            ),
            (
                "services.avahi.enable",
                "networking.firewall.allowedUDPPorts",
                "requires",
            ),
            // Mail
            (
                "services.postfix.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            (
                "services.dovecot2.enable",
                "services.postfix.enable",
                "requires",
            ),
            // Monitoring
            (
                "services.prometheus.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            (
                "services.grafana.enable",
                "services.prometheus.enable",
                "recommends",
            ),
            (
                "services.grafana.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            // Misc
            (
                "services.xrdp.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            (
                "services.syncthing.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            ("services.tor.enable", "services.tor.settings", "enables"),
            (
                "services.tailscale.enable",
                "networking.firewall.trustedInterfaces",
                "recommends",
            ),
            (
                "services.k3s.enable",
                "networking.firewall.allowedTCPPorts",
                "requires",
            ),
            (
                "services.nextcloud.enable",
                "services.nginx.enable",
                "requires",
            ),
            (
                "services.nextcloud.enable",
                "services.postgresql.enable",
                "recommends",
            ),
            (
                "services.gitea.enable",
                "services.postgresql.enable",
                "recommends",
            ),
        ]);

        // ── Sound (~15) ────────────────────────────────────────────────
        patterns.extend_from_slice(&[
            (
                "services.pipewire.enable",
                "hardware.pulseaudio.enable",
                "conflicts",
            ),
            (
                "hardware.pulseaudio.enable",
                "services.pipewire.enable",
                "conflicts",
            ),
            (
                "services.pipewire.enable",
                "services.pipewire.alsa.enable",
                "enables",
            ),
            (
                "services.pipewire.enable",
                "services.pipewire.pulse.enable",
                "enables",
            ),
            (
                "services.pipewire.enable",
                "services.pipewire.jack.enable",
                "enables",
            ),
            (
                "services.pipewire.alsa.enable",
                "services.pipewire.alsa.support32Bit",
                "enables",
            ),
            (
                "hardware.pulseaudio.enable",
                "hardware.pulseaudio.package",
                "requires",
            ),
            (
                "hardware.pulseaudio.support32Bit",
                "hardware.pulseaudio.enable",
                "requires",
            ),
            ("sound.enable", "hardware.pulseaudio.enable", "enables"),
            ("sound.enable", "services.pipewire.enable", "enables"),
            (
                "hardware.bluetooth.enable",
                "services.pipewire.enable",
                "recommends",
            ),
            (
                "hardware.bluetooth.enable",
                "hardware.bluetooth.powerOnBoot",
                "configures",
            ),
            (
                "services.blueman.enable",
                "hardware.bluetooth.enable",
                "requires",
            ),
            (
                "hardware.pulseaudio.extraModules",
                "hardware.pulseaudio.enable",
                "requires",
            ),
            (
                "services.pipewire.wireplumber.enable",
                "services.pipewire.enable",
                "requires",
            ),
        ]);

        // ── Users & Security (~20) ─────────────────────────────────────
        patterns.extend_from_slice(&[
            ("users.users", "users.groups", "references"),
            ("users.users.*.extraGroups", "users.groups", "requires"),
            (
                "security.sudo.enable",
                "security.sudo.extraRules",
                "enables",
            ),
            ("security.sudo.enable", "security.doas.enable", "conflicts"),
            ("security.doas.enable", "security.sudo.enable", "conflicts"),
            (
                "security.polkit.enable",
                "security.polkit.extraConfig",
                "enables",
            ),
            (
                "security.apparmor.enable",
                "security.apparmor.policies",
                "enables",
            ),
            (
                "security.pam.services",
                "security.pam.loginLimits",
                "configures",
            ),
            (
                "security.rtkit.enable",
                "services.pipewire.enable",
                "supports",
            ),
            (
                "security.acme.defaults.email",
                "security.acme.certs",
                "configures",
            ),
            (
                "security.acme.acceptTerms",
                "security.acme.certs",
                "requires",
            ),
            (
                "services.openssh.settings.PasswordAuthentication",
                "services.openssh.enable",
                "requires",
            ),
            ("users.mutableUsers", "users.users", "configures"),
            ("users.defaultUserShell", "environment.shells", "requires"),
            (
                "security.pam.u2f.enable",
                "security.pam.services",
                "configures",
            ),
            ("security.auditd.enable", "security.audit.rules", "enables"),
            (
                "security.tpm2.enable",
                "security.tpm2.pkcs11.enable",
                "enables",
            ),
            (
                "nix.settings.trusted-users",
                "nix.settings.allowed-users",
                "extends",
            ),
            ("security.wrappers", "security.wrappers.*.source", "enables"),
            (
                "services.fprintd.enable",
                "security.pam.services",
                "configures",
            ),
        ]);

        // ── Packages & Nix Settings (~25) ──────────────────────────────
        patterns.extend_from_slice(&[
            (
                "nixpkgs.config.allowUnfree",
                "environment.systemPackages",
                "enables",
            ),
            (
                "nixpkgs.config.allowUnfree",
                "programs.steam.enable",
                "enables",
            ),
            ("nixpkgs.overlays", "environment.systemPackages", "modifies"),
            (
                "nix.settings.experimental-features",
                "nix.settings.flakes",
                "enables",
            ),
            ("nix.gc.automatic", "nix.gc.dates", "enables"),
            ("nix.gc.automatic", "nix.gc.options", "enables"),
            (
                "nix.settings.auto-optimise-store",
                "nix.gc.automatic",
                "supplements",
            ),
            (
                "nix.settings.substituters",
                "nix.settings.trusted-public-keys",
                "requires",
            ),
            ("nix.settings.max-jobs", "nix.settings.cores", "configures"),
            (
                "programs.steam.enable",
                "hardware.opengl.driSupport32Bit",
                "requires",
            ),
            (
                "programs.steam.enable",
                "hardware.pulseaudio.support32Bit",
                "recommends",
            ),
            ("programs.zsh.enable", "users.defaultUserShell", "provides"),
            ("programs.fish.enable", "users.defaultUserShell", "provides"),
            (
                "programs.bash.enableCompletion",
                "environment.systemPackages",
                "supplements",
            ),
            (
                "programs.gnupg.agent.enable",
                "programs.gnupg.agent.pinentryPackage",
                "configures",
            ),
            ("programs.java.enable", "programs.java.package", "requires"),
            (
                "programs.nix-ld.enable",
                "programs.nix-ld.libraries",
                "enables",
            ),
            ("programs.git.enable", "programs.git.config", "enables"),
            ("environment.shells", "programs.zsh.enable", "requires"),
            (
                "environment.variables",
                "environment.sessionVariables",
                "supplements",
            ),
            (
                "environment.etc",
                "environment.systemPackages",
                "supplements",
            ),
            (
                "programs.dconf.enable",
                "services.xserver.desktopManager.gnome.enable",
                "supports",
            ),
            (
                "programs.xwayland.enable",
                "services.xserver.enable",
                "supplements",
            ),
            ("programs.mtr.enable", "security.wrappers", "requires"),
            (
                "nixpkgs.hostPlatform",
                "nixpkgs.buildPlatform",
                "configures",
            ),
        ]);

        // ── Desktop Environment (~10) ──────────────────────────────────
        patterns.extend_from_slice(&[
            ("services.flatpak.enable", "xdg.portal.enable", "requires"),
            ("xdg.portal.enable", "xdg.portal.extraPortals", "enables"),
            (
                "services.dbus.enable",
                "services.xserver.enable",
                "supports",
            ),
            (
                "services.udisks2.enable",
                "services.dbus.enable",
                "requires",
            ),
            ("services.gvfs.enable", "services.dbus.enable", "requires"),
            (
                "services.gnome.gnome-keyring.enable",
                "security.pam.services",
                "configures",
            ),
            (
                "services.xserver.desktopManager.gnome.enable",
                "services.dbus.enable",
                "requires",
            ),
            (
                "services.xserver.desktopManager.gnome.enable",
                "services.udev.packages",
                "configures",
            ),
            (
                "programs.thunar.enable",
                "services.gvfs.enable",
                "recommends",
            ),
            (
                "services.xserver.desktopManager.plasma5.enable",
                "services.dbus.enable",
                "requires",
            ),
        ]);

        // ── Virtualization (~15) ───────────────────────────────────────
        patterns.extend_from_slice(&[
            (
                "virtualisation.docker.enable",
                "virtualisation.docker.storageDriver",
                "configures",
            ),
            (
                "virtualisation.docker.enable",
                "virtualisation.docker.daemon.settings",
                "enables",
            ),
            (
                "virtualisation.libvirtd.enable",
                "virtualisation.libvirtd.qemu.package",
                "requires",
            ),
            (
                "virtualisation.libvirtd.enable",
                "users.extraGroups.libvirtd",
                "requires",
            ),
            (
                "virtualisation.libvirtd.enable",
                "boot.kernelModules",
                "requires",
            ),
            (
                "virtualisation.podman.enable",
                "virtualisation.podman.dockerCompat",
                "enables",
            ),
            (
                "virtualisation.podman.enable",
                "virtualisation.podman.defaultNetwork.settings",
                "enables",
            ),
            (
                "virtualisation.virtualbox.host.enable",
                "users.extraGroups.vboxusers",
                "requires",
            ),
            (
                "virtualisation.virtualbox.host.enable",
                "boot.kernelModules",
                "requires",
            ),
            (
                "virtualisation.virtualbox.host.enableExtensionPack",
                "virtualisation.virtualbox.host.enable",
                "requires",
            ),
            (
                "virtualisation.kvmgt.enable",
                "boot.kernelModules",
                "requires",
            ),
            (
                "virtualisation.lxd.enable",
                "virtualisation.lxd.package",
                "requires",
            ),
            (
                "virtualisation.waydroid.enable",
                "virtualisation.lxc.enable",
                "requires",
            ),
            (
                "virtualisation.vmware.host.enable",
                "boot.kernelModules",
                "requires",
            ),
            (
                "boot.kernelModules",
                "virtualisation.libvirtd.enable",
                "supports",
            ),
        ]);

        patterns
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

        let edges = analyzer.discover_structure();
        assert!(
            !edges.is_empty(),
            "Should discover causal edges from correlated observations"
        );

        // HashMap iteration order is non-deterministic, so which variable
        // becomes `from` vs `to` depends on key ordering. Check that at least
        // one of the three variables has outgoing side effects.
        let mut any_has_effects = false;
        for var in &["config.A", "config.B", "config.C"] {
            let effects = analyzer.predict_side_effects(var);
            if !effects.is_empty() {
                any_has_effects = true;
                // Validate all predicted effects
                for effect in &effects {
                    assert!(
                        effect.confidence >= 0.0 && effect.confidence <= 1.0,
                        "Side effect confidence should be in [0, 1], got {} for {}",
                        effect.confidence,
                        effect.affected_variable
                    );
                    assert!(
                        !effect.affected_variable.is_empty(),
                        "Affected variable name should not be empty"
                    );
                    assert!(
                        !effect.direction.is_empty(),
                        "Direction should not be empty"
                    );
                }
            }
        }
        assert!(
            any_has_effects,
            "At least one variable should have predicted side effects"
        );
    }

    #[test]
    fn test_known_patterns_count() {
        let patterns = NixOSCausalPatterns::known_patterns();
        // Should have at least 200 curated patterns
        assert!(
            patterns.len() >= 200,
            "Expected >=200 patterns, got {}",
            patterns.len()
        );
        // Verify all entries are non-empty
        for (cause, effect, rel) in &patterns {
            assert!(!cause.is_empty());
            assert!(!effect.is_empty());
            assert!(!rel.is_empty());
        }
    }

    #[test]
    fn test_add_structural_edge() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge(
            "services.nginx.enable",
            "networking.firewall.allowedTCPPorts",
            0.9,
        );
        let effects = graph.predict_side_effects("services.nginx.enable");
        assert_eq!(effects.len(), 1);
        assert_eq!(
            effects[0].affected_variable,
            "networking.firewall.allowedTCPPorts"
        );
    }

    #[test]
    #[cfg(feature = "native")]
    fn test_bootstrap_from_schemas() {
        use crate::parser::option_schema::parse_nixos_options;

        let json = r#"{
            "services.nginx.enable": { "type": "bool", "default": false, "description": "Enable nginx" },
            "services.nginx.package": { "type": "package", "default": "pkgs.nginx", "description": "Nginx package" },
            "services.nginx.virtualHosts": { "type": "attrsOf submodule", "description": "Virtual hosts" }
        }"#;

        let schemas = parse_nixos_options(json).unwrap();
        let mut graph = NixCausalGraph::new(42);
        graph.bootstrap_from_schemas(&schemas);

        // The enable option should gate the sub-options
        let effects = graph.predict_side_effects("services.nginx.enable");
        assert!(!effects.is_empty(), "enable should gate sub-options");
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("causal_graph.json");

        // Build a graph with known patterns + some structural edges
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("boot.loader.grub.enable", "boot.loader.grub.device", 0.9);
        graph.add_structural_edge("services.nginx.enable", "services.nginx.package", 0.85);
        graph.observe("services.nginx.enable", 1.0);
        graph.observe("services.nginx.enable", 0.0);

        let edge_count = graph.edge_count();
        assert!(edge_count >= 2);

        // Save
        graph.save(&path).unwrap();
        assert!(path.exists());

        // Load into a fresh graph
        let mut graph2 = NixCausalGraph::new(42);
        let loaded = graph2.load(&path).unwrap();
        assert_eq!(loaded, edge_count);
        assert_eq!(graph2.edge_count(), edge_count);

        // Observations should be merged
        let obs = graph2.observations.get("services.nginx.enable").unwrap();
        assert_eq!(obs.len(), 2);
    }

    #[test]
    fn test_hebbian_strengthening() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.5);

        // Observe A→B occurring — should strengthen
        graph.observe_outcome("A", &["B"], &["B"]);
        let edge = &graph.causal_graph[&("A".to_string(), "B".to_string())];
        assert!(
            edge.confidence > 0.5,
            "Edge should strengthen: {}",
            edge.confidence
        );
    }

    #[test]
    fn test_hebbian_weakening() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.5);

        // Predict A→B but B doesn't occur — should weaken
        graph.observe_outcome("A", &[], &["B"]);
        let edge = &graph.causal_graph[&("A".to_string(), "B".to_string())];
        assert!(
            edge.confidence < 0.5,
            "Edge should weaken: {}",
            edge.confidence
        );
    }

    #[test]
    fn test_hebbian_discovers_new_edges() {
        let mut graph = NixCausalGraph::new(42);

        // Observe A→C for the first time (no prior edge)
        graph.observe_outcome("A", &["C"], &[]);
        assert!(
            graph
                .causal_graph
                .contains_key(&("A".to_string(), "C".to_string()))
        );
        let edge = &graph.causal_graph[&("A".to_string(), "C".to_string())];
        assert!(
            edge.confidence > 0.3,
            "New edge should have boosted confidence: {}",
            edge.confidence
        );
    }

    #[test]
    fn test_hebbian_prunes_weak_edges() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.06);

        // Weaken it below threshold
        graph.observe_outcome("A", &[], &["B"]);
        // 0.06 - 0.1 * 0.06 = 0.054 — still above 0.05
        // Weaken again
        graph.observe_outcome("A", &[], &["B"]);
        // Should be pruned below min_confidence
        assert!(
            !graph
                .causal_graph
                .contains_key(&("A".to_string(), "B".to_string()))
                || graph.causal_graph[&("A".to_string(), "B".to_string())].confidence
                    >= graph.min_confidence,
            "Weak edges should be pruned"
        );
    }

    #[test]
    fn test_load_merges_without_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("causal_graph.json");

        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("a", "b", 0.9);
        graph.save(&path).unwrap();

        // Load into a graph that already has the same edge
        let mut graph2 = NixCausalGraph::new(42);
        graph2.add_structural_edge("a", "b", 0.5);
        graph2.add_structural_edge("c", "d", 0.7);

        let loaded = graph2.load(&path).unwrap();
        assert_eq!(loaded, 0); // a→b already exists, so 0 new edges
        assert_eq!(graph2.edge_count(), 2); // a→b + c→d
    }

    #[test]
    fn test_root_cause_analysis_chain() {
        let mut graph = NixCausalGraph::new(42);
        // Build chain: A → B → C → D (symptom)
        graph.add_structural_edge("A", "B", 0.8);
        graph.add_structural_edge("B", "C", 0.7);
        graph.add_structural_edge("C", "D", 0.9);

        let analysis = graph.analyze_root_causes("D");
        assert_eq!(analysis.symptom, "D");
        assert!(
            !analysis.causal_chain.is_empty(),
            "Should find causal chain to D"
        );
        assert!(
            !analysis.root_causes.is_empty(),
            "Should identify root causes"
        );

        // A should be a root cause (no incoming edges)
        assert!(
            analysis.root_causes.iter().any(|rc| rc.variable == "A"),
            "A should be root cause. Got: {:?}",
            analysis
                .root_causes
                .iter()
                .map(|rc| &rc.variable)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_root_cause_no_causes() {
        let graph = NixCausalGraph::new(42);
        let analysis = graph.analyze_root_causes("unknown");
        assert!(analysis.root_causes.is_empty());
        assert!(analysis.causal_chain.is_empty());
    }

    #[test]
    fn test_side_effect_transitive() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.8);
        graph.add_structural_edge("B", "C", 0.6);

        let effects = graph.predict_side_effects("A");
        let affected: Vec<&str> = effects
            .iter()
            .map(|e| e.affected_variable.as_str())
            .collect();
        assert!(affected.contains(&"B"), "Should predict B as affected");
        assert!(affected.contains(&"C"), "Should predict C transitively");
    }

    #[test]
    fn test_recommend_fixes_high_confidence() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("misconfigured-firewall", "service-unreachable", 0.8);

        let fixes = graph.recommend_fixes("service-unreachable");
        assert!(
            fixes.iter().any(|f| f.contains("misconfigured-firewall")),
            "Should recommend adjusting the firewall. Got: {:?}",
            fixes
        );
    }

    #[test]
    fn test_recommend_fixes_insufficient_evidence() {
        let mut graph = NixCausalGraph::new(42);
        // Low confidence edge shouldn't generate a recommendation
        graph.add_structural_edge("maybe-cause", "symptom", 0.3);

        let fixes = graph.recommend_fixes("symptom");
        assert!(
            fixes.iter().any(|f| f.contains("Insufficient")),
            "Low-confidence edges shouldn't produce recommendations. Got: {:?}",
            fixes
        );
    }

    #[test]
    fn test_observe_and_batch() {
        let mut graph = NixCausalGraph::new(42);
        graph.observe("cpu_load", 0.5);
        graph.observe("cpu_load", 0.7);
        graph.observe_batch("memory", &[0.3, 0.4, 0.5]);

        assert_eq!(graph.observations["cpu_load"].len(), 2);
        assert_eq!(graph.observations["memory"].len(), 3);
    }

    #[test]
    fn test_learning_rate_setter() {
        let graph = NixCausalGraph::new(42).with_learning_rate(0.5);
        assert!((graph.learning_rate - 0.5).abs() < 1e-6);

        // Should clamp to valid range
        let graph = NixCausalGraph::new(42).with_learning_rate(5.0);
        assert!((graph.learning_rate - 1.0).abs() < 1e-6);

        let graph = NixCausalGraph::new(42).with_learning_rate(0.0);
        assert!((graph.learning_rate - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_hebbian_repeated_strengthening_bounded() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.5);

        // Strengthen many times — confidence should approach but never exceed 1.0
        for _ in 0..100 {
            graph.observe_outcome("A", &["B"], &["B"]);
        }
        let edge = &graph.causal_graph[&("A".to_string(), "B".to_string())];
        assert!(
            edge.confidence <= 1.0,
            "Confidence should be bounded: {}",
            edge.confidence
        );
        assert!(
            edge.confidence > 0.95,
            "Should be very high after many observations: {}",
            edge.confidence
        );
    }

    #[test]
    fn test_edge_confidence_accessor() {
        let mut graph = NixCausalGraph::new(42);

        // No edge yet
        assert!(graph.edge_confidence("A", "B").is_none());

        graph.add_structural_edge("A", "B", 0.75);
        let conf = graph.edge_confidence("A", "B");
        assert!(conf.is_some());
        assert!((conf.unwrap() - 0.75).abs() < 1e-6);

        // Reverse direction should not exist
        assert!(graph.edge_confidence("B", "A").is_none());
    }

    #[test]
    fn test_edge_count_tracks_additions_and_pruning() {
        let mut graph = NixCausalGraph::new(42);
        assert_eq!(graph.edge_count(), 0);

        graph.add_structural_edge("A", "B", 0.8);
        assert_eq!(graph.edge_count(), 1);

        graph.add_structural_edge("C", "D", 0.6);
        assert_eq!(graph.edge_count(), 2);

        // Observe new edge via Hebbian
        graph.observe_outcome("E", &["F"], &[]);
        assert_eq!(graph.edge_count(), 3);

        // Weaken C→D below pruning threshold
        graph.add_structural_edge("X", "Y", 0.06);
        assert_eq!(graph.edge_count(), 4);
        // Repeatedly weaken X→Y to get it pruned
        for _ in 0..10 {
            graph.observe_outcome("X", &[], &["Y"]);
        }
        // X→Y should be pruned, count should decrease
        assert!(graph.edge_count() < 4, "Pruned edges should reduce count");
    }

    #[test]
    fn test_top_edges_sorted_by_confidence() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.3);
        graph.add_structural_edge("C", "D", 0.9);
        graph.add_structural_edge("E", "F", 0.6);

        let top = graph.top_edges(10);
        assert_eq!(top.len(), 3);
        assert!((top[0].confidence - 0.9).abs() < 1e-6);
        assert!((top[1].confidence - 0.6).abs() < 1e-6);
        assert!((top[2].confidence - 0.3).abs() < 1e-6);

        // top_edges with limit
        let top1 = graph.top_edges(1);
        assert_eq!(top1.len(), 1);
        assert!((top1[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_bounds_after_random_operations() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.5);
        graph.add_structural_edge("A", "C", 0.5);
        graph.add_structural_edge("B", "D", 0.5);

        // Mix of strengthening and weakening
        let operations: Vec<(&[&str], &[&str])> = vec![
            (&["B", "C"], &["B", "C", "D"]),
            (&["B"], &["B", "C"]),
            (&[], &["B", "C"]),
            (&["C", "D"], &["C"]),
            (&["B", "C"], &["B"]),
        ];
        for (observed, predicted) in &operations {
            graph.observe_outcome("A", observed, predicted);
        }

        // All remaining edges should have confidence in [0.0, 1.0]
        for edge in graph.causal_graph.values() {
            assert!(
                edge.confidence >= 0.0 && edge.confidence <= 1.0,
                "Edge {}->{} confidence out of bounds: {}",
                edge.from,
                edge.to,
                edge.confidence
            );
        }
    }

    #[test]
    fn test_from_index_maintained_on_add() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.9);
        graph.add_structural_edge("A", "C", 0.8);
        graph.add_structural_edge("B", "C", 0.7);

        let a_targets = graph.from_index.get("A").unwrap();
        assert!(a_targets.contains("B"));
        assert!(a_targets.contains("C"));
        assert_eq!(a_targets.len(), 2);

        let b_targets = graph.from_index.get("B").unwrap();
        assert!(b_targets.contains("C"));
        assert_eq!(b_targets.len(), 1);
    }

    #[test]
    fn test_from_index_maintained_on_prune() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.06); // just above min_confidence (0.05)

        // Weaken until pruned
        graph.observe_outcome("A", &[], &["B"]);
        graph.observe_outcome("A", &[], &["B"]);

        if !graph
            .causal_graph
            .contains_key(&("A".to_string(), "B".to_string()))
        {
            // Edge was pruned — index should not contain B as target of A
            let a_targets = graph.from_index.get("A");
            assert!(
                a_targets.is_none() || !a_targets.unwrap().contains("B"),
                "from_index should be cleaned up on prune"
            );
        }
    }

    #[test]
    fn test_from_index_on_discover() {
        let mut graph = NixCausalGraph::new(42);
        for i in 0..50 {
            graph.observe("x", i as f64);
            graph.observe("y", 2.0 * i as f64);
        }
        let edges = graph.discover_structure();
        assert!(!edges.is_empty());

        // Verify index matches the graph
        for ((from, to), _) in &graph.causal_graph {
            let targets = graph.from_index.get(from).expect("from_index missing key");
            assert!(
                targets.contains(to),
                "from_index missing target {to} for {from}"
            );
        }
    }

    #[test]
    fn test_from_index_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.json");

        let mut g1 = NixCausalGraph::new(42);
        g1.add_structural_edge("X", "Y", 0.9);
        g1.add_structural_edge("X", "Z", 0.8);
        g1.save(&path).unwrap();

        let mut g2 = NixCausalGraph::new(42);
        g2.load(&path).unwrap();

        let targets = g2.from_index.get("X").unwrap();
        assert!(targets.contains("Y"));
        assert!(targets.contains("Z"));
    }

    // ── Phase 2.8: Causal Graph Structural Hardening ────────────────────────
    // These tests exercise failure modes NOT covered by the Hebbian / discovery
    // tests above: graph topology edge-cases and degraded-input safety.

    /// Self-edge: a node that causes itself.
    /// Must not panic, must be stored, and must not cause infinite traversal
    /// in root-cause or side-effect walks.
    #[test]
    fn test_self_edge_no_panic_no_infinite_loop() {
        let mut graph = NixCausalGraph::new(1);
        graph.add_structural_edge("A", "A", 0.7);

        assert_eq!(graph.edge_count(), 1, "self-edge must be stored");

        // Side-effect walk must terminate (visited set prevents infinite loop)
        let effects = graph.predict_side_effects("A");
        // Self-edges targeting the originating variable are filtered out
        // by the `if to != variable` guard — result is empty, not a hang.
        assert!(
            effects.is_empty(),
            "self-edge should produce no self-referential side-effect"
        );

        // Root-cause walk must also terminate
        let rca = graph.analyze_root_causes("A");
        assert!(
            rca.root_causes.len() <= 1,
            "self-edge root-cause analysis must terminate"
        );
    }

    /// Duplicate edge insertion: adding the same edge twice must not corrupt
    /// the edge count or the index.
    #[test]
    fn test_duplicate_edge_no_double_count() {
        let mut graph = NixCausalGraph::new(1);
        graph.add_structural_edge("X", "Y", 0.5);
        graph.add_structural_edge("X", "Y", 0.9); // overwrite, not a second entry

        // The map key (from, to) is unique — only one edge survives.
        assert_eq!(
            graph.edge_count(),
            1,
            "duplicate edge must not inflate count"
        );

        // The from_index must have exactly one entry for X→Y.
        let targets = graph.from_index.get("X").expect("index missing X");
        assert_eq!(
            targets.len(),
            1,
            "from_index must not have duplicate target"
        );
    }

    /// Cycle: A→B→C→A. Side-effect and root-cause walks must terminate.
    #[test]
    fn test_cycle_traversal_terminates() {
        let mut graph = NixCausalGraph::new(1);
        graph.add_structural_edge("A", "B", 0.8);
        graph.add_structural_edge("B", "C", 0.8);
        graph.add_structural_edge("C", "A", 0.8); // closes cycle

        assert_eq!(graph.edge_count(), 3);

        // Both traversals must return without hanging.
        let effects = graph.predict_side_effects("A");
        assert!(
            effects.len() <= 3,
            "cycle walk must not produce unbounded results: got {}",
            effects.len()
        );

        let rca = graph.analyze_root_causes("A");
        assert!(
            rca.causal_chain.len() <= 3,
            "cycle root-cause chain must be bounded"
        );
    }

    /// Disconnected sub-graph: two completely separate components.
    /// Querying one must not expose nodes from the other.
    #[test]
    fn test_disconnected_subgraph_isolation() {
        let mut graph = NixCausalGraph::new(1);
        // Component 1
        graph.add_structural_edge("A", "B", 0.9);
        // Component 2 (disconnected)
        graph.add_structural_edge("X", "Y", 0.9);

        // Side effects of A must not include X or Y
        let effects_a = graph.predict_side_effects("A");
        let names: Vec<&str> = effects_a
            .iter()
            .map(|e| e.affected_variable.as_str())
            .collect();
        assert!(
            !names.contains(&"X") && !names.contains(&"Y"),
            "disconnected component must not appear in side-effect walk: {names:?}"
        );

        // Root causes of X must not include A or B
        let rca_x = graph.analyze_root_causes("X");
        let cause_names: Vec<&str> = rca_x
            .root_causes
            .iter()
            .map(|c| c.variable.as_str())
            .collect();
        assert!(
            !cause_names.contains(&"A") && !cause_names.contains(&"B"),
            "disconnected component must not appear in root-cause walk: {cause_names:?}"
        );
    }

    /// NaN confidence via observe_outcome must not propagate into the graph.
    /// The Hebbian update formula uses `entry.confidence.clamp(0.0, 1.0)`, so
    /// even if an edge somehow acquired a NaN confidence, subsequent queries
    /// must remain finite.
    #[test]
    fn test_confidence_never_nan_after_repeated_learning() {
        let mut graph = NixCausalGraph::new(42);

        // Add an edge near the boundary
        graph.add_structural_edge("cause", "effect", 0.999);

        // Repeatedly strengthen — Hebbian rule: w += η*(1-w) — must converge, not NaN
        for _ in 0..100 {
            graph.observe_outcome("cause", &["effect"], &["effect"]);
        }

        let conf = graph.edge_confidence("cause", "effect").unwrap_or(f64::NAN);
        assert!(
            conf.is_finite(),
            "confidence must be finite after 100 strengthening steps, got {conf}"
        );
        assert!(
            (0.0..=1.0).contains(&conf),
            "confidence must be in [0, 1], got {conf}"
        );
    }

    /// Querying side-effects for a node not in the graph must return empty,
    /// not a panic.
    #[test]
    fn test_side_effects_unknown_node_returns_empty() {
        let graph = NixCausalGraph::new(1);
        let effects = graph.predict_side_effects("totally_unknown_node");
        assert!(
            effects.is_empty(),
            "unknown node must return empty side effects"
        );
    }

    /// Root-cause analysis for a node with no incoming edges must return empty
    /// root_causes (not a crash).
    #[test]
    fn test_root_cause_leaf_node_returns_empty() {
        let mut graph = NixCausalGraph::new(1);
        graph.add_structural_edge("X", "Y", 0.8);
        // X is a source — it has no incoming edges.
        let rca = graph.analyze_root_causes("X");
        assert!(
            rca.root_causes.is_empty(),
            "leaf source node must have no root causes"
        );
    }

    /// Stable drift replay: simulates the full causal failure path
    ///     sensor shift → drift detected → causal edge queried
    /// and verifies that degraded/missing input produces safe uncertainty
    /// (not false certainty and not a panic).
    ///
    /// This is the Phase 2.8 evidence replay example.
    #[test]
    fn test_stable_drift_replay_example() {
        let mut graph = NixCausalGraph::new(7);

        // Baseline structural knowledge
        graph.add_structural_edge("sensor.temp", "services.thermal.enable", 0.75);
        graph.add_structural_edge("services.thermal.enable", "boot.kernelParams", 0.60);

        // --- Step 1: Normal operation — sensor observed, no drift --------
        // (World model drift is checked separately; here we verify the causal
        // side: sensor has a known downstream chain.)
        let normal_effects = graph.predict_side_effects("sensor.temp");
        assert!(
            !normal_effects.is_empty(),
            "normal sensor should have known downstream effects"
        );
        for effect in &normal_effects {
            assert!(
                effect.confidence.is_finite() && effect.confidence >= 0.0,
                "normal effects must have finite non-negative confidence: {:?}",
                effect.confidence
            );
        }

        // --- Step 2: Sensor anomaly — large shift observed ---------------
        // Simulate: the sensor reading changes wildly (drift detected upstream),
        // so we call observe_outcome with an empty observed_effects (the
        // expected effect didn't materialise — the system adapted differently).
        graph.observe_outcome(
            "sensor.temp",
            &[],                          // nothing materialised as expected
            &["services.thermal.enable"], // this was predicted
        );

        // The edge must weaken, not disappear immediately (min_confidence=0.05).
        let conf_after_miss = graph.edge_confidence("sensor.temp", "services.thermal.enable");
        match conf_after_miss {
            Some(c) => {
                assert!(
                    c.is_finite() && c >= 0.0 && c <= 1.0,
                    "confidence after sensor miss must be in [0,1], got {c}"
                );
                // Confidence should have decreased
                assert!(c < 0.75, "missed prediction must weaken edge: conf={c}");
            }
            None => {
                // Edge was pruned — that is safe uncertainty, not false certainty.
                // The system correctly reports "insufficient evidence".
                let recs = graph.recommend_fixes("services.thermal.enable");
                assert!(
                    recs.iter().any(|r| r.contains("Insufficient")),
                    "pruned edge should trigger insufficient-evidence recommendation"
                );
            }
        }

        // --- Step 3: Missing sensor data — unknown node queried ----------
        // Simulates a sensor that dropped off entirely. Must not panic.
        let missing_effects = graph.predict_side_effects("sensor.missing_entirely");
        assert!(
            missing_effects.is_empty(),
            "missing sensor must produce empty effects, not false certainty"
        );

        // All remaining edge confidences must be finite and in [0,1].
        for ((from, to), edge) in &graph.causal_graph {
            assert!(
                edge.confidence.is_finite() && (0.0..=1.0).contains(&edge.confidence),
                "edge {from}→{to} has out-of-bounds confidence: {}",
                edge.confidence
            );
        }
    }
}
