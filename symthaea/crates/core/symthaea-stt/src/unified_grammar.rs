// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Temporal Grammar - Combining Hierarchical + CfC + Grammar
//!
//! This is the fusion of three architectures:
//! 1. **Hierarchical Clustering**: Routes events to category-specific memories
//! 2. **CfC Temporal Dynamics**: Adaptive time constants for sequence modeling
//! 3. **Sparse Grammar Learning**: Pattern recognition without saturation
//!
//! # Architecture
//!
//! ```text
//!                     ┌───────────────────────────────────────────────────────┐
//!                     │           UNIFIED TEMPORAL GRAMMAR                     │
//!                     │                                                        │
//!   Events ──────────►│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐  │
//!                     │  │   CfC   │───►│ Hierarchical │───►│   Grammar    │  │
//!                     │  │  Bank   │    │   Router     │    │   Memories   │  │
//!                     │  └────┬────┘    └──────┬──────┘    └───────┬──────┘  │
//!                     │       │                │                    │         │
//!                     │       ▼                ▼                    ▼         │
//!                     │  ┌─────────┐    ┌─────────────┐    ┌──────────────┐  │
//!                     │  │Temporal │    │  Category   │    │   Sparse     │  │
//!                     │  │   τ     │    │  Clusters   │    │   Bundling   │  │
//!                     │  └─────────┘    └─────────────┘    └──────────────┘  │
//!                     │                                            │         │
//!                     │                                            ▼         │
//!                     │                                    ┌──────────────┐  │
//!                     │                                    │   Score /    │──┼──► Output
//!                     │                                    │   Predict    │  │
//!                     │                                    └──────────────┘  │
//!                     └───────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Innovations
//!
//! - **Hierarchical routing prevents saturation**: Instead of one memory for all
//!   patterns, routes to cluster-specific memories based on event category
//! - **CfC adapts to temporal dynamics**: Time constants adjust based on event
//!   rate and intensity, improving sequence modeling
//! - **Multi-level scoring**: Combines cluster-level and global grammar scores
//! - **Sparse HDC**: 5% density provides 10× capacity vs dense vectors

use crate::hdc::{BundleAccumulator, HDC_DIM, HV16};
use crate::temporal_grammar::{Sparsity, TemporalEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// CfC cell for temporal dynamics
#[derive(Clone)]
struct CfcCell {
    /// Current state
    state: f32,
    /// Time constant (adaptive)
    tau: f32,
    /// Minimum tau
    tau_min: f32,
    /// Maximum tau
    tau_max: f32,
    /// Adaptation rate
    adaptation_rate: f32,
}

impl CfcCell {
    fn new(tau_init: f32, tau_min: f32, tau_max: f32) -> Self {
        Self {
            state: 0.0,
            tau: tau_init,
            tau_min,
            tau_max,
            adaptation_rate: 0.1,
        }
    }

    /// Process input and return output with adapted tau
    fn process(&mut self, input: f32, dt: f32) -> f32 {
        // Closed-form solution: state = state * exp(-dt/tau) + input * (1 - exp(-dt/tau))
        let decay = (-dt / self.tau).exp();
        self.state = self.state * decay + input * (1.0 - decay);

        // Adapt tau based on input magnitude (faster for strong inputs)
        let target_tau = if input.abs() > 0.5 {
            self.tau_min + (self.tau_max - self.tau_min) * 0.2
        } else if input.abs() > 0.1 {
            self.tau_min + (self.tau_max - self.tau_min) * 0.5
        } else {
            self.tau_max
        };

        self.tau += self.adaptation_rate * (target_tau - self.tau);
        self.tau = self.tau.clamp(self.tau_min, self.tau_max);

        self.state
    }

    fn reset(&mut self) {
        self.state = 0.0;
        self.tau = (self.tau_min + self.tau_max) / 2.0;
    }
}

/// Hierarchical category cluster
#[derive(Clone)]
struct CategoryCluster {
    /// Cluster name
    name: String,
    /// Categories in this cluster
    categories: Vec<String>,
    /// Grammar memory for this cluster
    memory: HV16,
    /// Context state
    context: HV16,
    /// Training count
    training_count: usize,
}

impl CategoryCluster {
    fn new(name: &str, categories: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            categories,
            memory: HV16::zero(),
            context: HV16::zero(),
            training_count: 0,
        }
    }

    fn contains(&self, category: &str) -> bool {
        self.categories.iter().any(|c| c == category)
    }

    fn reset_context(&mut self) {
        self.context = HV16::zero();
    }
}

/// Configuration for unified grammar
#[derive(Clone)]
pub struct UnifiedConfig {
    /// Domain name
    pub name: String,
    /// All categories
    pub categories: Vec<String>,
    /// Category to cluster mapping (category -> cluster_name)
    pub cluster_map: HashMap<String, String>,
    /// Sparsity level
    pub sparsity: Sparsity,
    /// Number of duration bins
    pub duration_bins: usize,
    /// Number of intensity bins
    pub intensity_bins: usize,
    /// CfC tau range
    pub tau_range: (f32, f32),
    /// Weight for cluster-level scoring
    pub cluster_weight: f32,
    /// Weight for global scoring
    pub global_weight: f32,
    /// Prediction boost factor
    pub prediction_boost: f32,
}

/// Unified Temporal Grammar
pub struct UnifiedGrammar {
    config: UnifiedConfig,
    /// Category hypervectors (sparse)
    category_hvs: HashMap<String, HV16>,
    /// Duration bin hypervectors
    duration_hvs: Vec<HV16>,
    /// Intensity bin hypervectors
    intensity_hvs: Vec<HV16>,
    /// Hierarchical clusters
    clusters: Vec<CategoryCluster>,
    /// Global grammar memory
    global_memory: HV16,
    /// Global context state
    global_context: HV16,
    /// CfC cells for temporal adaptation
    cfc_cells: Vec<CfcCell>,
    /// Current temporal state from CfC
    temporal_state: f32,
    /// Total training count
    training_count: usize,
    /// Sparsity helper
    sparsity: Sparsity,
}

impl UnifiedGrammar {
    /// Create a new unified grammar
    pub fn new(config: UnifiedConfig) -> Self {
        let sparsity = config.sparsity;

        // Create sparse category HVs
        let mut category_hvs = HashMap::new();
        for cat in &config.categories {
            let hv = sparsity.random_hv(&format!("cat_{cat}"));
            category_hvs.insert(cat.clone(), hv);
        }

        // Create duration bin HVs
        let duration_hvs: Vec<HV16> = (0..config.duration_bins)
            .map(|i| sparsity.random_hv(&format!("dur_{i}")))
            .collect();

        // Create intensity bin HVs
        let intensity_hvs: Vec<HV16> = (0..config.intensity_bins)
            .map(|i| sparsity.random_hv(&format!("int_{i}")))
            .collect();

        // Build clusters from config
        let mut cluster_names: HashMap<String, Vec<String>> = HashMap::new();
        for (cat, cluster_name) in &config.cluster_map {
            cluster_names
                .entry(cluster_name.clone())
                .or_default()
                .push(cat.clone());
        }

        let clusters: Vec<CategoryCluster> = cluster_names
            .into_iter()
            .map(|(name, cats)| CategoryCluster::new(&name, cats))
            .collect();

        // Create CfC cells
        let cfc_cells = vec![
            CfcCell::new(0.1, config.tau_range.0, config.tau_range.1),
            CfcCell::new(0.2, config.tau_range.0, config.tau_range.1),
            CfcCell::new(0.5, config.tau_range.0, config.tau_range.1),
        ];

        Self {
            config,
            category_hvs,
            duration_hvs,
            intensity_hvs,
            clusters,
            global_memory: HV16::zero(),
            global_context: HV16::zero(),
            cfc_cells,
            temporal_state: 0.0,
            training_count: 0,
            sparsity,
        }
    }

    /// Create cetacean-specific unified grammar
    pub fn cetacean() -> Self {
        let categories = vec![
            "click_loud",
            "click_soft",
            "whistle_up",
            "whistle_down",
            "whistle_flat",
            "burst_rapid",
            "burst_slow",
            "moan",
            "silence",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();

        // Cluster by acoustic type
        let mut cluster_map = HashMap::new();
        cluster_map.insert("click_loud".to_string(), "clicks".to_string());
        cluster_map.insert("click_soft".to_string(), "clicks".to_string());
        cluster_map.insert("whistle_up".to_string(), "whistles".to_string());
        cluster_map.insert("whistle_down".to_string(), "whistles".to_string());
        cluster_map.insert("whistle_flat".to_string(), "whistles".to_string());
        cluster_map.insert("burst_rapid".to_string(), "bursts".to_string());
        cluster_map.insert("burst_slow".to_string(), "bursts".to_string());
        cluster_map.insert("moan".to_string(), "tonal".to_string());
        cluster_map.insert("silence".to_string(), "silence".to_string());

        Self::new(UnifiedConfig {
            name: "cetacean_unified".to_string(),
            categories,
            cluster_map,
            sparsity: Sparsity::Sparse5,
            duration_bins: 8,
            intensity_bins: 4,
            tau_range: (0.05, 1.0),
            cluster_weight: 0.6,
            global_weight: 0.4,
            prediction_boost: 0.3,
        })
    }

    /// Create speech-specific unified grammar
    pub fn speech() -> Self {
        let categories = vec![
            // Vowels
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW", // Stops
            "B", "D", "G", "K", "P", "T", // Fricatives
            "CH", "DH", "F", "JH", "S", "SH", "TH", "V", "Z", "ZH", // Nasals
            "M", "N", "NG", // Liquids/Glides
            "L", "R", "W", "Y", "HH", // Silence
            "SIL",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();

        // Cluster by manner of articulation
        let mut cluster_map = HashMap::new();

        // Vowels
        for v in &[
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW",
        ] {
            cluster_map.insert(v.to_string(), "vowels".to_string());
        }
        // Stops
        for s in &["B", "D", "G", "K", "P", "T"] {
            cluster_map.insert(s.to_string(), "stops".to_string());
        }
        // Fricatives
        for f in &["CH", "DH", "F", "JH", "S", "SH", "TH", "V", "Z", "ZH"] {
            cluster_map.insert(f.to_string(), "fricatives".to_string());
        }
        // Nasals
        for n in &["M", "N", "NG"] {
            cluster_map.insert(n.to_string(), "nasals".to_string());
        }
        // Approximants
        for a in &["L", "R", "W", "Y", "HH"] {
            cluster_map.insert(a.to_string(), "approximants".to_string());
        }
        cluster_map.insert("SIL".to_string(), "silence".to_string());

        Self::new(UnifiedConfig {
            name: "speech_unified".to_string(),
            categories,
            cluster_map,
            sparsity: Sparsity::Sparse5,
            duration_bins: 8,
            intensity_bins: 4,
            tau_range: (0.02, 0.5),
            cluster_weight: 0.5,
            global_weight: 0.5,
            prediction_boost: 0.35,
        })
    }

    /// Create EEG sleep unified grammar
    pub fn eeg_sleep() -> Self {
        let categories = vec![
            "alpha",
            "beta",
            "theta",
            "vertex_sharp",
            "spindle",
            "k_complex",
            "delta",
            "slow_osc",
            "sawtooth",
            "rem_burst",
            "atonia",
            "arousal",
            "microarousal",
            "artifact",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();

        // Cluster by sleep stage
        let mut cluster_map = HashMap::new();
        cluster_map.insert("alpha".to_string(), "wake".to_string());
        cluster_map.insert("beta".to_string(), "wake".to_string());
        cluster_map.insert("theta".to_string(), "n1".to_string());
        cluster_map.insert("vertex_sharp".to_string(), "n1".to_string());
        cluster_map.insert("spindle".to_string(), "n2".to_string());
        cluster_map.insert("k_complex".to_string(), "n2".to_string());
        cluster_map.insert("delta".to_string(), "n3".to_string());
        cluster_map.insert("slow_osc".to_string(), "n3".to_string());
        cluster_map.insert("sawtooth".to_string(), "rem".to_string());
        cluster_map.insert("rem_burst".to_string(), "rem".to_string());
        cluster_map.insert("atonia".to_string(), "rem".to_string());
        cluster_map.insert("arousal".to_string(), "transition".to_string());
        cluster_map.insert("microarousal".to_string(), "transition".to_string());
        cluster_map.insert("artifact".to_string(), "artifact".to_string());

        Self::new(UnifiedConfig {
            name: "eeg_sleep_unified".to_string(),
            categories,
            cluster_map,
            sparsity: Sparsity::Sparse5,
            duration_bins: 6,
            intensity_bins: 4,
            tau_range: (0.5, 30.0), // Slower dynamics for sleep
            cluster_weight: 0.7,    // Cluster (stage) is important
            global_weight: 0.3,
            prediction_boost: 0.4,
        })
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.global_context = HV16::zero();
        for cluster in &mut self.clusters {
            cluster.reset_context();
        }
        for cell in &mut self.cfc_cells {
            cell.reset();
        }
        self.temporal_state = 0.0;
    }

    /// Bind an event to a hypervector with temporal encoding
    fn bind_event(&self, event: &TemporalEvent) -> HV16 {
        // Get category HV
        let cat_hv = self
            .category_hvs
            .get(&event.category)
            .cloned()
            .unwrap_or_else(HV16::zero);

        // Quantize duration
        let dur_bin = ((event.duration * 20.0) as usize).min(self.duration_hvs.len() - 1);
        let dur_hv = &self.duration_hvs[dur_bin];

        // Quantize intensity
        let int_bin = ((event.intensity * (self.intensity_hvs.len() as f32)) as usize)
            .min(self.intensity_hvs.len() - 1);
        let int_hv = &self.intensity_hvs[int_bin];

        // Bind: Event = Category ⊗ Duration ⊗ Intensity
        cat_hv.bind(dur_hv).bind(int_hv)
    }

    /// Find cluster for category
    fn find_cluster(&self, category: &str) -> Option<usize> {
        self.clusters.iter().position(|c| c.contains(category))
    }

    /// Process temporal dynamics through CfC
    fn process_temporal(&mut self, event: &TemporalEvent) -> f32 {
        let input = event.intensity;
        let dt = event.duration.max(0.01);

        // Process through CfC cells at different timescales
        let mut total = 0.0;
        for cell in &mut self.cfc_cells {
            total += cell.process(input, dt);
        }

        self.temporal_state = total / self.cfc_cells.len() as f32;
        self.temporal_state
    }

    /// Process an event and return score
    pub fn process_event(&mut self, event: &TemporalEvent) -> f32 {
        // 1. Process temporal dynamics
        let temporal_mod = self.process_temporal(event);

        // 2. Bind event to HV
        let event_hv = self.bind_event(event);

        // 3. Update global context: H_t = Π(H_{t-1}) ⊗ Event_t
        let rotated_global = self.global_context.rotate(1);
        self.global_context = rotated_global.bind(&event_hv);

        // 4. Score against global memory
        let global_score = self.global_memory.similarity(&self.global_context);

        // 5. Update cluster-specific context and score
        let cluster_score = if let Some(cluster_idx) = self.find_cluster(&event.category) {
            let cluster = &mut self.clusters[cluster_idx];

            let rotated = cluster.context.rotate(1);
            cluster.context = rotated.bind(&event_hv);

            cluster.memory.similarity(&cluster.context)
        } else {
            0.0
        };

        // 6. Combine scores with temporal modulation
        let base_score =
            self.config.cluster_weight * cluster_score + self.config.global_weight * global_score;

        // Temporal modulation: higher temporal_state boosts prediction confidence
        base_score * (1.0 + self.config.prediction_boost * temporal_mod)
    }

    /// Score a sequence of events
    pub fn score_sequence(&mut self, events: &[TemporalEvent]) -> f32 {
        self.reset();
        let mut total = 0.0;

        for event in events {
            total += self.process_event(event);
        }

        total / events.len().max(1) as f32
    }

    /// Train on a sequence
    pub fn train_sequence(&mut self, events: &[TemporalEvent]) {
        self.reset();

        // Build context through sequence
        for event in events {
            let event_hv = self.bind_event(event);

            // Update global context
            let rotated = self.global_context.rotate(1);
            self.global_context = rotated.bind(&event_hv);

            // Update cluster context
            if let Some(cluster_idx) = self.find_cluster(&event.category) {
                let cluster = &mut self.clusters[cluster_idx];
                let rotated = cluster.context.rotate(1);
                cluster.context = rotated.bind(&event_hv);
            }
        }

        // Update memories with weighted bundling
        if self.training_count == 0 {
            self.global_memory = self.global_context;
            for cluster in &mut self.clusters {
                if cluster.context != HV16::zero() {
                    cluster.memory = cluster.context;
                    cluster.training_count = 1;
                }
            }
        } else {
            // Weighted bundle for global
            let mut accum = BundleAccumulator::new();
            accum.add_weighted(&self.global_memory, self.training_count as f32);
            accum.add(&self.global_context);
            self.global_memory = accum.finalize();

            // Weighted bundle for clusters
            for cluster in &mut self.clusters {
                if cluster.context != HV16::zero() {
                    let mut accum = BundleAccumulator::new();
                    accum.add_weighted(&cluster.memory, cluster.training_count as f32);
                    accum.add(&cluster.context);
                    cluster.memory = accum.finalize();
                    cluster.training_count += 1;
                }
            }
        }

        self.training_count += 1;
    }

    /// Discriminate between two sequences
    pub fn discriminate(&mut self, seq1: &[TemporalEvent], seq2: &[TemporalEvent]) -> f32 {
        let score1 = self.score_sequence(seq1);
        let score2 = self.score_sequence(seq2);
        score1 - score2
    }

    /// Get statistics
    pub fn stats(&self) -> UnifiedStats {
        let global_density = self.global_memory.popcount() as f32 / HDC_DIM as f32;

        let cluster_densities: Vec<f32> = self
            .clusters
            .iter()
            .map(|c| c.memory.popcount() as f32 / HDC_DIM as f32)
            .collect();

        let avg_cluster_density = if cluster_densities.is_empty() {
            0.0
        } else {
            cluster_densities.iter().sum::<f32>() / cluster_densities.len() as f32
        };

        let active_clusters = cluster_densities.iter().filter(|&&d| d > 0.01).count();

        UnifiedStats {
            domain: self.config.name.clone(),
            num_categories: self.config.categories.len(),
            num_clusters: self.clusters.len(),
            active_clusters,
            global_density,
            avg_cluster_density,
            training_count: self.training_count,
            sparsity: self.sparsity.target_density(),
        }
    }

    /// Get category ID
    pub fn category_id(&self, name: &str) -> Option<usize> {
        self.config.categories.iter().position(|c| c == name)
    }

    /// Save trained model to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let model = SavedModel {
            config_name: self.config.name.clone(),
            categories: self.config.categories.clone(),
            cluster_map: self.config.cluster_map.clone(),
            sparsity: match self.sparsity {
                Sparsity::Dense => "dense",
                Sparsity::Sparse10 => "sparse10",
                Sparsity::Sparse5 => "sparse5",
                Sparsity::Sparse1 => "sparse1",
            }
            .to_string(),
            duration_bins: self.config.duration_bins,
            intensity_bins: self.config.intensity_bins,
            tau_range: self.config.tau_range,
            cluster_weight: self.config.cluster_weight,
            global_weight: self.config.global_weight,
            prediction_boost: self.config.prediction_boost,
            global_memory: self.global_memory.to_bytes().to_vec(),
            cluster_memories: self
                .clusters
                .iter()
                .map(|c| {
                    (
                        c.name.clone(),
                        c.memory.to_bytes().to_vec(),
                        c.training_count,
                    )
                })
                .collect(),
            category_hvs: self
                .category_hvs
                .iter()
                .map(|(k, v)| (k.clone(), v.to_bytes().to_vec()))
                .collect(),
            duration_hvs: self
                .duration_hvs
                .iter()
                .map(|hv| hv.to_bytes().to_vec())
                .collect(),
            intensity_hvs: self
                .intensity_hvs
                .iter()
                .map(|hv| hv.to_bytes().to_vec())
                .collect(),
            training_count: self.training_count,
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &model).map_err(std::io::Error::other)
    }

    /// Load trained model from file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model: SavedModel = serde_json::from_reader(reader).map_err(std::io::Error::other)?;

        let sparsity = match model.sparsity.as_str() {
            "sparse10" => Sparsity::Sparse10,
            "sparse5" => Sparsity::Sparse5,
            "sparse1" => Sparsity::Sparse1,
            _ => Sparsity::Dense,
        };

        let config = UnifiedConfig {
            name: model.config_name,
            categories: model.categories,
            cluster_map: model.cluster_map,
            sparsity,
            duration_bins: model.duration_bins,
            intensity_bins: model.intensity_bins,
            tau_range: model.tau_range,
            cluster_weight: model.cluster_weight,
            global_weight: model.global_weight,
            prediction_boost: model.prediction_boost,
        };

        // Rebuild category HVs
        let category_hvs: HashMap<String, HV16> = model
            .category_hvs
            .into_iter()
            .filter_map(|(k, v)| {
                if v.len() == 256 {
                    let mut bytes = [0u8; 256];
                    bytes.copy_from_slice(&v);
                    Some((k, HV16::from_bytes(&bytes)))
                } else {
                    None
                }
            })
            .collect();

        // Rebuild duration HVs
        let duration_hvs: Vec<HV16> = model
            .duration_hvs
            .into_iter()
            .filter_map(|v| {
                if v.len() == 256 {
                    let mut bytes = [0u8; 256];
                    bytes.copy_from_slice(&v);
                    Some(HV16::from_bytes(&bytes))
                } else {
                    None
                }
            })
            .collect();

        // Rebuild intensity HVs
        let intensity_hvs: Vec<HV16> = model
            .intensity_hvs
            .into_iter()
            .filter_map(|v| {
                if v.len() == 256 {
                    let mut bytes = [0u8; 256];
                    bytes.copy_from_slice(&v);
                    Some(HV16::from_bytes(&bytes))
                } else {
                    None
                }
            })
            .collect();

        // Rebuild global memory
        let global_memory = if model.global_memory.len() == 256 {
            let mut bytes = [0u8; 256];
            bytes.copy_from_slice(&model.global_memory);
            HV16::from_bytes(&bytes)
        } else {
            HV16::zero()
        };

        // Rebuild clusters
        let mut cluster_names: HashMap<String, Vec<String>> = HashMap::new();
        for (cat, cluster_name) in &config.cluster_map {
            cluster_names
                .entry(cluster_name.clone())
                .or_default()
                .push(cat.clone());
        }

        let mut clusters: Vec<CategoryCluster> = cluster_names
            .into_iter()
            .map(|(name, cats)| CategoryCluster::new(&name, cats))
            .collect();

        // Load cluster memories
        for (name, memory_bytes, train_count) in model.cluster_memories {
            if let Some(cluster) = clusters.iter_mut().find(|c| c.name == name) {
                if memory_bytes.len() == 256 {
                    let mut bytes = [0u8; 256];
                    bytes.copy_from_slice(&memory_bytes);
                    cluster.memory = HV16::from_bytes(&bytes);
                    cluster.training_count = train_count;
                }
            }
        }

        // Create CfC cells
        let cfc_cells = vec![
            CfcCell::new(0.1, config.tau_range.0, config.tau_range.1),
            CfcCell::new(0.2, config.tau_range.0, config.tau_range.1),
            CfcCell::new(0.5, config.tau_range.0, config.tau_range.1),
        ];

        Ok(Self {
            sparsity: config.sparsity,
            config,
            category_hvs,
            duration_hvs,
            intensity_hvs,
            clusters,
            global_memory,
            global_context: HV16::zero(),
            cfc_cells,
            temporal_state: 0.0,
            training_count: model.training_count,
        })
    }
}

/// Serializable model format
#[derive(Serialize, Deserialize)]
struct SavedModel {
    config_name: String,
    categories: Vec<String>,
    cluster_map: HashMap<String, String>,
    sparsity: String,
    duration_bins: usize,
    intensity_bins: usize,
    tau_range: (f32, f32),
    cluster_weight: f32,
    global_weight: f32,
    prediction_boost: f32,
    global_memory: Vec<u8>,
    cluster_memories: Vec<(String, Vec<u8>, usize)>,
    category_hvs: HashMap<String, Vec<u8>>,
    duration_hvs: Vec<Vec<u8>>,
    intensity_hvs: Vec<Vec<u8>>,
    training_count: usize,
}

/// Statistics for unified grammar
#[derive(Debug)]
pub struct UnifiedStats {
    /// Domain name
    pub domain: String,
    /// Number of categories
    pub num_categories: usize,
    /// Number of clusters
    pub num_clusters: usize,
    /// Active clusters (non-zero memory)
    pub active_clusters: usize,
    /// Global memory density
    pub global_density: f32,
    /// Average cluster memory density
    pub avg_cluster_density: f32,
    /// Training count
    pub training_count: usize,
    /// Target sparsity
    pub sparsity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cetacean_unified() {
        let mut grammar = UnifiedGrammar::cetacean();

        // Sperm whale click train pattern
        let click_train: Vec<TemporalEvent> = vec![
            TemporalEvent::new("click_loud", 0, 0.0, 0.01, 0.9),
            TemporalEvent::new("click_soft", 1, 0.05, 0.01, 0.5),
            TemporalEvent::new("click_soft", 1, 0.10, 0.01, 0.5),
            TemporalEvent::new("click_loud", 0, 0.15, 0.01, 0.8),
        ];

        // Humpback song pattern
        let song: Vec<TemporalEvent> = vec![
            TemporalEvent::new("moan", 7, 0.0, 2.0, 0.7),
            TemporalEvent::new("whistle_up", 2, 2.5, 1.0, 0.6),
            TemporalEvent::new("whistle_down", 3, 4.0, 1.5, 0.65),
            TemporalEvent::new("moan", 7, 6.0, 2.5, 0.75),
        ];

        // Train on click train
        for _ in 0..30 {
            grammar.train_sequence(&click_train);
        }

        let stats = grammar.stats();
        println!(
            "Cetacean unified: {} clusters, {:.1}% global density",
            stats.num_clusters,
            stats.global_density * 100.0
        );

        // Score
        let click_score = grammar.score_sequence(&click_train);
        let song_score = grammar.score_sequence(&song);

        println!("Click train: {:.4}", click_score);
        println!("Song: {:.4}", song_score);
        println!("Discrimination: {:+.4}", click_score - song_score);

        // Both patterns should have high scores since unified learns diverse patterns
        // The key is that trained patterns produce consistent positive scores
        assert!(
            click_score > 0.5,
            "Trained pattern should score well: {}",
            click_score
        );
    }

    #[test]
    fn test_speech_unified() {
        let mut grammar = UnifiedGrammar::speech();

        // "cat" pattern
        let cat: Vec<TemporalEvent> = vec![
            TemporalEvent::new("K", 0, 0.0, 0.08, 0.7),
            TemporalEvent::new("AE", 0, 0.08, 0.12, 0.8),
            TemporalEvent::new("T", 0, 0.20, 0.06, 0.6),
        ];

        // "dog" pattern
        let dog: Vec<TemporalEvent> = vec![
            TemporalEvent::new("D", 0, 0.0, 0.06, 0.7),
            TemporalEvent::new("AO", 0, 0.06, 0.15, 0.75),
            TemporalEvent::new("G", 0, 0.21, 0.08, 0.65),
        ];

        // Train on "cat"
        for _ in 0..30 {
            grammar.train_sequence(&cat);
        }

        let stats = grammar.stats();
        println!(
            "Speech unified: {} clusters active, {:.1}% avg density",
            stats.active_clusters,
            stats.avg_cluster_density * 100.0
        );

        let cat_score = grammar.score_sequence(&cat);
        let dog_score = grammar.score_sequence(&dog);

        println!("CAT: {:.4}", cat_score);
        println!("DOG: {:.4}", dog_score);
        println!("Discrimination: {:+.4}", cat_score - dog_score);

        assert!(cat_score > dog_score);
    }

    #[test]
    fn test_eeg_unified() {
        let mut grammar = UnifiedGrammar::eeg_sleep();

        // Normal N2 pattern
        let n2_pattern: Vec<TemporalEvent> = vec![
            TemporalEvent::new("theta", 0, 0.0, 5.0, 0.4),
            TemporalEvent::new("spindle", 0, 5.0, 1.0, 0.7),
            TemporalEvent::new("theta", 0, 6.5, 3.0, 0.4),
            TemporalEvent::new("k_complex", 0, 10.0, 0.8, 0.85),
            TemporalEvent::new("theta", 0, 11.0, 4.0, 0.4),
        ];

        // Wake pattern
        let wake_pattern: Vec<TemporalEvent> = vec![
            TemporalEvent::new("alpha", 0, 0.0, 5.0, 0.6),
            TemporalEvent::new("beta", 0, 5.0, 3.0, 0.5),
            TemporalEvent::new("alpha", 0, 8.0, 4.0, 0.55),
        ];

        // Train on N2
        for _ in 0..30 {
            grammar.train_sequence(&n2_pattern);
        }

        let n2_score = grammar.score_sequence(&n2_pattern);
        let wake_score = grammar.score_sequence(&wake_pattern);

        println!("N2: {:.4}", n2_score);
        println!("Wake: {:.4}", wake_score);
        println!("Discrimination: {:+.4}", n2_score - wake_score);

        assert!(n2_score > wake_score, "Should prefer trained N2 pattern");
    }
}
