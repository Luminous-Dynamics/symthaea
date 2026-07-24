// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unsupervised Acoustic Discovery Module
//!
//! Discovers natural "units" in any acoustic signal without prior
//! knowledge of the species, language, or communication system.
//!
//! # Acoustic Unit Discovery Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    UNSUPERVISED DISCOVERY PIPELINE                       │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Unknown Audio ──► LTC Projector ──► Salience Segmentation              │
//! │       │                │                      │                          │
//! │       │          (Adaptive τ             (Natural                       │
//! │       │           handles ANY            boundaries)                    │
//! │       │           timescale)                  │                          │
//! │       │                                       ▼                          │
//! │       │                              ┌──────────────┐                    │
//! │       │                              │   Segment    │                    │
//! │       │                              │   Vectors    │                    │
//! │       │                              │    (HV16)    │                    │
//! │       │                              └──────┬───────┘                    │
//! │       │                                     │                            │
//! │       │                                     ▼                            │
//! │       │                        ┌────────────────────────┐                │
//! │       │                        │  HDC Online Clustering │                │
//! │       │                        │  (Resonator Discovery) │                │
//! │       │                        └───────────┬────────────┘                │
//! │       │                                    │                             │
//! │       │              ┌─────────────────────┼─────────────────────┐       │
//! │       │              ▼                     ▼                     ▼       │
//! │       │        ┌─────────┐           ┌─────────┐           ┌─────────┐   │
//! │       │        │Cluster A│           │Cluster B│           │Cluster C│   │
//! │       │        │ "Unit 1"│           │ "Unit 2"│           │ "Unit 3"│   │
//! │       │        └─────────┘           └─────────┘           └─────────┘   │
//! │       │                                                                  │
//! │       │                              OPTIONAL                            │
//! │       │                                 │                                │
//! │       ▼                                 ▼                                │
//! │  ┌─────────────┐                 ┌─────────────┐                        │
//! │  │   Visual    │ ──── BIND ────► │  Grounded   │                        │
//! │  │   Context   │                 │   Meaning   │                        │
//! │  └─────────────┘                 └─────────────┘                        │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Insight
//!
//! By removing the dependency on "text" and focusing on "temporal dynamics,"
//! This module detects candidate boundaries, clusters recurring acoustic units,
//! and records transitions. It does not infer reference, intent, or translation.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

// Use shared HDC types from the hdc module
use crate::hdc::{HV16, bundle};

// ============================================================================
// DISCOVERED UNIT
// ============================================================================

/// A discovered acoustic unit. It is not presumed to be a phoneme or word.
#[derive(Clone, Debug)]
pub struct DiscoveredUnit {
    /// Unique identifier (e.g., "UNIT_001")
    pub id: String,
    /// Centroid hypervector (prototype)
    pub centroid: HV16,
    /// Number of instances observed
    pub instance_count: usize,
    /// All instance vectors (for refinement)
    instances: Vec<HV16>,
    /// Temporal statistics
    pub mean_duration_ms: f32,
    pub min_duration_ms: f32,
    pub max_duration_ms: f32,
    /// Spectral characteristics (optional descriptors)
    pub spectral_centroid_hz: Option<f32>,
    pub dominant_frequency_hz: Option<f32>,
    /// Co-occurrence patterns (which units follow this one)
    pub following_units: HashMap<String, usize>,
    pub preceding_units: HashMap<String, usize>,
}

impl DiscoveredUnit {
    pub fn new(id: &str, first_instance: HV16, duration_ms: f32) -> Self {
        Self {
            id: id.to_string(),
            centroid: first_instance,
            instance_count: 1,
            instances: vec![first_instance],
            mean_duration_ms: duration_ms,
            min_duration_ms: duration_ms,
            max_duration_ms: duration_ms,
            spectral_centroid_hz: None,
            dominant_frequency_hz: None,
            following_units: HashMap::new(),
            preceding_units: HashMap::new(),
        }
    }

    /// Add a new instance to this unit
    pub fn add_instance(&mut self, instance: HV16, duration_ms: f32) {
        self.instances.push(instance);
        self.instance_count += 1;

        // Update centroid (online bundling)
        self.centroid = bundle(&self.instances);

        // Update duration stats
        let n = self.instance_count as f32;
        self.mean_duration_ms = ((n - 1.0) * self.mean_duration_ms + duration_ms) / n;
        self.min_duration_ms = self.min_duration_ms.min(duration_ms);
        self.max_duration_ms = self.max_duration_ms.max(duration_ms);
    }

    /// Record that another unit followed this one
    pub fn record_following(&mut self, unit_id: &str) {
        *self.following_units.entry(unit_id.to_string()).or_insert(0) += 1;
    }

    /// Record that another unit preceded this one
    pub fn record_preceding(&mut self, unit_id: &str) {
        *self.preceding_units.entry(unit_id.to_string()).or_insert(0) += 1;
    }

    /// Get similarity to a candidate vector
    pub fn similarity(&self, candidate: &HV16) -> f32 {
        self.centroid.similarity(candidate)
    }

    /// Trim instance history to save memory
    pub fn trim_instances(&mut self, max_instances: usize) {
        if self.instances.len() > max_instances {
            // Keep a representative sample
            let step = self.instances.len() / max_instances;
            self.instances = self
                .instances
                .iter()
                .enumerate()
                .filter(|(i, _)| i % step == 0)
                .map(|(_, hv)| *hv)
                .take(max_instances)
                .collect();
        }
    }
}

// ============================================================================
// DISCOVERY CONFIGURATION
// ============================================================================

/// Configuration for unsupervised discovery
#[derive(Clone, Debug)]
pub struct DiscoveryConfig {
    /// Similarity threshold for assigning to existing cluster
    pub similarity_threshold: f32,
    /// Minimum instances to consider a unit "stable"
    pub min_instances: usize,
    /// Maximum number of units to discover
    pub max_units: usize,
    /// Salience threshold for boundary detection
    pub salience_threshold: f32,
    /// Minimum segment duration (ms)
    pub min_segment_ms: f32,
    /// Maximum segment duration (ms)
    pub max_segment_ms: f32,
    /// Enable transition tracking
    pub track_transitions: bool,
    /// Maximum instances to store per unit (memory limit)
    pub max_instances_per_unit: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.65,
            min_instances: 3,
            max_units: 100,
            salience_threshold: 0.5,
            min_segment_ms: 10.0,
            max_segment_ms: 5000.0, // 5 seconds (for whale songs)
            track_transitions: true,
            max_instances_per_unit: 1000,
        }
    }
}

impl DiscoveryConfig {
    /// Configuration for dolphin/whale communication
    pub fn cetacean() -> Self {
        Self {
            similarity_threshold: 0.60,
            min_instances: 5,
            max_units: 200,
            salience_threshold: 0.4,
            min_segment_ms: 5.0,     // Very short clicks
            max_segment_ms: 10000.0, // Long whale songs
            track_transitions: true,
            max_instances_per_unit: 500,
        }
    }

    /// Configuration for bird songs
    pub fn avian() -> Self {
        Self {
            similarity_threshold: 0.62,
            min_instances: 3,
            max_units: 150,
            salience_threshold: 0.5,
            min_segment_ms: 20.0,
            max_segment_ms: 2000.0,
            track_transitions: true,
            max_instances_per_unit: 500,
        }
    }

    /// Configuration for primate calls
    pub fn primate() -> Self {
        Self {
            similarity_threshold: 0.65,
            min_instances: 5,
            max_units: 80,
            salience_threshold: 0.5,
            min_segment_ms: 50.0,
            max_segment_ms: 3000.0,
            track_transitions: true,
            max_instances_per_unit: 500,
        }
    }

    /// Permissive configuration for an unknown acoustic signal.
    pub fn unknown_signal() -> Self {
        Self {
            similarity_threshold: 0.55,
            min_instances: 2,
            max_units: 500,
            salience_threshold: 0.3,
            min_segment_ms: 1.0,
            max_segment_ms: 60000.0,
            track_transitions: true,
            max_instances_per_unit: 200,
        }
    }

    /// Legacy name retained for API compatibility. This configuration performs
    /// anomaly segmentation and clustering, not xenolinguistic understanding.
    #[deprecated(note = "use unknown_signal(); clustering does not establish language")]
    pub fn xenolinguistic() -> Self {
        Self::unknown_signal()
    }
}

// ============================================================================
// ACOUSTIC SEGMENT
// ============================================================================

/// A segment of audio (potential unit instance)
#[derive(Clone, Debug)]
pub struct AcousticSegment {
    /// Start time (seconds)
    pub start_time: f32,
    /// End time (seconds)
    pub end_time: f32,
    /// Start frame index
    pub start_frame: usize,
    /// End frame index
    pub end_frame: usize,
    /// Acoustic hypervector
    pub hv: HV16,
    /// Assigned unit ID (after clustering)
    pub unit_id: Option<String>,
    /// Salience score at boundaries
    pub boundary_salience: f32,
    /// Optional: spectral features
    pub spectral_centroid: Option<f32>,
}

impl AcousticSegment {
    pub fn duration_ms(&self) -> f32 {
        (self.end_time - self.start_time) * 1000.0
    }
}

// ============================================================================
// ONLINE CLUSTERER (The Discovery Engine)
// ============================================================================

/// Online HDC-based clustering for acoustic discovery
///
/// Uses the natural similarity structure of hypervectors to
/// discover acoustic units without pre-defined categories.
pub struct OnlineClusterer {
    /// Discovered units
    units: HashMap<String, DiscoveredUnit>,
    /// Unit ID counter
    next_unit_id: usize,
    /// Configuration
    config: DiscoveryConfig,
    /// Previous unit (for transition tracking)
    previous_unit: Option<String>,
    /// Statistics
    pub stats: ClusteringStats,
}

#[derive(Clone, Debug, Default)]
pub struct ClusteringStats {
    pub segments_processed: usize,
    pub units_created: usize,
    pub instances_assigned: usize,
    pub transitions_recorded: usize,
}

impl OnlineClusterer {
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            units: HashMap::new(),
            next_unit_id: 1,
            config,
            previous_unit: None,
            stats: ClusteringStats::default(),
        }
    }

    /// Process a segment and assign/create a unit
    pub fn process_segment(&mut self, segment: &AcousticSegment) -> String {
        self.stats.segments_processed += 1;

        // Check duration constraints
        let duration = segment.duration_ms();
        if duration < self.config.min_segment_ms || duration > self.config.max_segment_ms {
            return "NOISE".to_string();
        }

        // Find best matching existing unit
        let mut best_match: Option<(String, f32)> = None;

        for (id, unit) in &self.units {
            let sim = unit.similarity(&segment.hv);
            if sim > self.config.similarity_threshold
                && best_match.as_ref().map(|(_, s)| sim > *s).unwrap_or(true)
            {
                best_match = Some((id.clone(), sim));
            }
        }

        let unit_id = if let Some((id, _)) = best_match {
            // Assign to existing unit
            if let Some(unit) = self.units.get_mut(&id) {
                unit.add_instance(segment.hv, duration);
                self.stats.instances_assigned += 1;
            }
            id
        } else {
            // Create new unit
            if self.units.len() >= self.config.max_units {
                // At capacity - assign to closest (even if below threshold)
                let closest = self
                    .units
                    .iter()
                    .map(|(id, u)| (id.clone(), u.similarity(&segment.hv)))
                    .max_by(|a, b| a.1.total_cmp(&b.1));

                if let Some((id, _)) = closest {
                    if let Some(unit) = self.units.get_mut(&id) {
                        unit.add_instance(segment.hv, duration);
                    }
                    id
                } else {
                    "OVERFLOW".to_string()
                }
            } else {
                let id = format!("UNIT_{:03}", self.next_unit_id);
                self.next_unit_id += 1;

                let unit = DiscoveredUnit::new(&id, segment.hv, duration);
                self.units.insert(id.clone(), unit);
                self.stats.units_created += 1;

                id
            }
        };

        // Track transitions
        if self.config.track_transitions {
            if let Some(ref prev_id) = self.previous_unit
                && prev_id != &unit_id
            {
                // Record A → B transition
                if let Some(prev_unit) = self.units.get_mut(prev_id) {
                    prev_unit.record_following(&unit_id);
                }
                if let Some(curr_unit) = self.units.get_mut(&unit_id) {
                    curr_unit.record_preceding(prev_id);
                }
                self.stats.transitions_recorded += 1;
            }
            self.previous_unit = Some(unit_id.clone());
        }

        unit_id
    }

    /// Get all discovered units
    pub fn units(&self) -> &HashMap<String, DiscoveredUnit> {
        &self.units
    }

    /// Get stable units (above minimum instance threshold)
    pub fn stable_units(&self) -> Vec<&DiscoveredUnit> {
        self.units
            .values()
            .filter(|u| u.instance_count >= self.config.min_instances)
            .collect()
    }

    /// Get unit by ID
    pub fn get_unit(&self, id: &str) -> Option<&DiscoveredUnit> {
        self.units.get(id)
    }

    /// Merge similar units
    pub fn merge_similar(&mut self, threshold: f32) {
        let unit_ids: Vec<String> = self.units.keys().cloned().collect();
        let mut to_merge: Vec<(String, String)> = Vec::new();

        // Find pairs to merge
        for i in 0..unit_ids.len() {
            for j in (i + 1)..unit_ids.len() {
                let id1 = &unit_ids[i];
                let id2 = &unit_ids[j];

                let sim = match (self.units.get(id1), self.units.get(id2)) {
                    (Some(u1), Some(u2)) => u1.centroid.similarity(&u2.centroid),
                    _ => continue,
                };

                if sim > threshold {
                    to_merge.push((id1.clone(), id2.clone()));
                }
            }
        }

        // Merge (keep the one with more instances)
        for (id1, id2) in to_merge {
            let count1 = self.units.get(&id1).map(|u| u.instance_count).unwrap_or(0);
            let count2 = self.units.get(&id2).map(|u| u.instance_count).unwrap_or(0);

            let (keep, remove) = if count1 >= count2 {
                (id1, id2)
            } else {
                (id2, id1)
            };

            if let Some(removed) = self.units.remove(&remove)
                && let Some(keeper) = self.units.get_mut(&keep)
            {
                // Add instances from removed unit
                for inst in removed.instances {
                    keeper.add_instance(inst, removed.mean_duration_ms);
                }
                // Merge transition stats
                for (k, v) in removed.following_units {
                    *keeper.following_units.entry(k).or_insert(0) += v;
                }
                for (k, v) in removed.preceding_units {
                    *keeper.preceding_units.entry(k).or_insert(0) += v;
                }
            }
        }
    }

    /// Trim memory usage
    pub fn trim_memory(&mut self) {
        for unit in self.units.values_mut() {
            unit.trim_instances(self.config.max_instances_per_unit);
        }
    }

    /// Reset state (but keep discovered units)
    pub fn reset_session(&mut self) {
        self.previous_unit = None;
    }
}

// ============================================================================
// GROUNDING (Multimodal Binding)
// ============================================================================

/// A grounded meaning (acoustic + context binding)
#[derive(Clone, Debug)]
pub struct GroundedMeaning {
    /// The acoustic unit
    pub acoustic_unit_id: String,
    /// The visual/context vector
    pub context_hv: HV16,
    /// The bound representation
    pub grounded_hv: HV16,
    /// Descriptive label (if provided)
    pub label: Option<String>,
    /// Confidence (based on co-occurrence count)
    pub confidence: f32,
    /// Number of observations
    pub observation_count: usize,
}

/// Grounding engine for multimodal binding
pub struct GroundingEngine {
    /// Learned groundings
    groundings: Vec<GroundedMeaning>,
    /// Acoustic unit to grounding index
    acoustic_index: HashMap<String, Vec<usize>>,
    /// Similarity threshold for matching
    threshold: f32,
}

impl GroundingEngine {
    pub fn new(threshold: f32) -> Self {
        Self {
            groundings: Vec::new(),
            acoustic_index: HashMap::new(),
            threshold,
        }
    }

    /// Learn a grounding: "When I hear X, I see Y"
    pub fn learn_grounding(
        &mut self,
        acoustic_unit_id: &str,
        acoustic_hv: &HV16,
        context_hv: &HV16,
        label: Option<&str>,
    ) {
        // Check if we already have this grounding
        if let Some(indices) = self.acoustic_index.get(acoustic_unit_id) {
            for &idx in indices {
                let grounding = &mut self.groundings[idx];
                let sim = grounding.context_hv.similarity(context_hv);
                if sim > self.threshold {
                    // Reinforce existing grounding
                    grounding.observation_count += 1;
                    grounding.confidence = 1.0 - (0.5f32).powi(grounding.observation_count as i32);
                    return;
                }
            }
        }

        // Create new grounding
        let grounded_hv = acoustic_hv.bind(context_hv);

        let idx = self.groundings.len();
        self.groundings.push(GroundedMeaning {
            acoustic_unit_id: acoustic_unit_id.to_string(),
            context_hv: *context_hv,
            grounded_hv,
            label: label.map(|s| s.to_string()),
            confidence: 0.5,
            observation_count: 1,
        });

        self.acoustic_index
            .entry(acoustic_unit_id.to_string())
            .or_default()
            .push(idx);
    }

    /// Query: "What does this sound mean?"
    pub fn query_acoustic(&self, acoustic_unit_id: &str) -> Vec<&GroundedMeaning> {
        self.acoustic_index
            .get(acoustic_unit_id)
            .map(|indices| indices.iter().map(|&i| &self.groundings[i]).collect())
            .unwrap_or_default()
    }

    /// Query: "What sound goes with this context?"
    pub fn query_context(&self, context_hv: &HV16) -> Vec<(&GroundedMeaning, f32)> {
        self.groundings
            .iter()
            .map(|g| (g, context_hv.similarity(&g.context_hv)))
            .filter(|(_, sim)| *sim > self.threshold)
            .collect()
    }

    /// Get all groundings
    pub fn all_groundings(&self) -> &[GroundedMeaning] {
        &self.groundings
    }
}

// ============================================================================
// DISCOVERY RESULT
// ============================================================================

/// Complete discovery result
#[derive(Clone)]
pub struct DiscoveryResult {
    /// All discovered units
    pub units: Vec<DiscoveredUnit>,
    /// Segment assignments
    pub segment_labels: Vec<(f32, f32, String)>, // (start, end, unit_id)
    /// Transition matrix (unit_i → unit_j count)
    pub transition_matrix: HashMap<(String, String), usize>,
    /// Statistics
    pub stats: DiscoveryStats,
}

#[derive(Clone, Debug, Default)]
pub struct DiscoveryStats {
    pub total_duration_sec: f32,
    pub num_segments: usize,
    pub num_units: usize,
    pub num_stable_units: usize,
    pub mean_unit_duration_ms: f32,
    pub unit_entropy: f32, // Diversity measure
}

impl DiscoveryResult {
    /// Export to JSON
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");

        json.push_str("  \"stats\": {\n");
        json.push_str(&format!(
            "    \"total_duration_sec\": {:.2},\n",
            self.stats.total_duration_sec
        ));
        json.push_str(&format!(
            "    \"num_segments\": {},\n",
            self.stats.num_segments
        ));
        json.push_str(&format!("    \"num_units\": {},\n", self.stats.num_units));
        json.push_str(&format!(
            "    \"num_stable_units\": {},\n",
            self.stats.num_stable_units
        ));
        json.push_str(&format!(
            "    \"unit_entropy\": {:.3}\n",
            self.stats.unit_entropy
        ));
        json.push_str("  },\n");

        json.push_str("  \"units\": [\n");
        for (i, unit) in self.units.iter().enumerate() {
            let comma = if i < self.units.len() - 1 { "," } else { "" };
            json.push_str(&format!(
                "    {{\n      \"id\": \"{}\",\n      \"instances\": {},\n      \"mean_duration_ms\": {:.1},\n      \"duration_range_ms\": [{:.1}, {:.1}]\n    }}{}\n",
                unit.id, unit.instance_count, unit.mean_duration_ms,
                unit.min_duration_ms, unit.max_duration_ms, comma
            ));
        }
        json.push_str("  ],\n");

        json.push_str("  \"segments\": [\n");
        for (i, (start, end, label)) in self.segment_labels.iter().enumerate() {
            let comma = if i < self.segment_labels.len() - 1 {
                ","
            } else {
                ""
            };
            json.push_str(&format!(
                "    {{\n      \"start\": {start:.3},\n      \"end\": {end:.3},\n      \"unit\": \"{label}\"\n    }}{comma}\n"
            ));
        }
        json.push_str("  ],\n");

        // Transition summary
        json.push_str("  \"frequent_transitions\": [\n");
        let mut transitions: Vec<_> = self.transition_matrix.iter().collect();
        transitions.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

        for (i, ((from, to), count)) in transitions.iter().take(20).enumerate() {
            let comma = if i < 19 && i < transitions.len() - 1 {
                ","
            } else {
                ""
            };
            json.push_str(&format!(
                "    {{\n      \"from\": \"{from}\",\n      \"to\": \"{to}\",\n      \"count\": {count}\n    }}{comma}\n"
            ));
        }
        json.push_str("  ]\n");

        json.push_str("}\n");
        json
    }

    /// Export segment labels to TSV (for Praat/Audacity)
    pub fn to_tsv(&self) -> String {
        let mut tsv = String::from("start\tend\tlabel\n");
        for (start, end, label) in &self.segment_labels {
            tsv.push_str(&format!("{start:.6}\t{end:.6}\t{label}\n"));
        }
        tsv
    }

    /// Save to files
    pub fn save(&self, base_path: &Path) -> std::io::Result<()> {
        // JSON report
        let json_path = base_path.with_extension("json");
        let mut json_file = File::create(&json_path)?;
        json_file.write_all(self.to_json().as_bytes())?;

        // TSV labels (for annotation tools)
        let tsv_path = base_path.with_extension("tsv");
        let mut tsv_file = File::create(&tsv_path)?;
        tsv_file.write_all(self.to_tsv().as_bytes())?;

        Ok(())
    }
}

// ============================================================================
// DISCOVERY PIPELINE
// ============================================================================

/// Complete unsupervised discovery pipeline
pub struct DiscoveryPipeline {
    /// Online clusterer
    clusterer: OnlineClusterer,
    /// Configuration
    config: DiscoveryConfig,
    /// Accumulated segments
    segments: Vec<AcousticSegment>,
}

impl DiscoveryPipeline {
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            clusterer: OnlineClusterer::new(config.clone()),
            config,
            segments: Vec::new(),
        }
    }

    /// Process acoustic frames with salience
    pub fn process(&mut self, frames: &[HV16], salience: &[f32], timestamps: &[f32]) {
        // Detect boundaries
        let boundaries = self.detect_boundaries(salience, timestamps);

        // Create segments
        for window in boundaries.windows(2) {
            let (start_time, start_frame) = window[0];
            let (end_time, end_frame) = window[1];

            if end_frame <= start_frame || end_frame > frames.len() {
                continue;
            }

            // Bundle segment frames
            let segment_frames = &frames[start_frame..end_frame];
            let hv = bundle(segment_frames);

            let segment = AcousticSegment {
                start_time,
                end_time,
                start_frame,
                end_frame,
                hv,
                unit_id: None,
                boundary_salience: salience.get(start_frame).cloned().unwrap_or(0.0),
                spectral_centroid: None,
            };

            // Check duration
            let duration = segment.duration_ms();
            if duration >= self.config.min_segment_ms && duration <= self.config.max_segment_ms {
                let unit_id = self.clusterer.process_segment(&segment);

                let mut assigned_segment = segment;
                assigned_segment.unit_id = Some(unit_id);
                self.segments.push(assigned_segment);
            }
        }
    }

    /// Detect boundaries from salience
    fn detect_boundaries(&self, salience: &[f32], timestamps: &[f32]) -> Vec<(f32, usize)> {
        if salience.is_empty() {
            return Vec::new();
        }

        let mean: f32 = salience.iter().sum::<f32>() / salience.len() as f32;
        let variance: f32 =
            salience.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / salience.len() as f32;
        let std = variance.sqrt();
        let threshold = mean + std * self.config.salience_threshold;

        let mut boundaries = vec![(0.0, 0)];
        let mut last_time = -0.010f32;
        let min_gap = self.config.min_segment_ms / 1000.0;

        // Method 1: Peak detection
        for i in 1..salience.len() - 1 {
            let is_peak = salience[i] > salience[i - 1] && salience[i] > salience[i + 1];
            let above_threshold = salience[i] > threshold;
            let far_enough = timestamps[i] - last_time > min_gap;

            if is_peak && above_threshold && far_enough {
                boundaries.push((timestamps[i], i));
                last_time = timestamps[i];
            }
        }

        // Method 2: If no peaks found, try gradient-based detection
        if boundaries.len() <= 1 {
            let window = 5;
            for i in window..salience.len() - window {
                let prev_mean: f32 = salience[i - window..i].iter().sum::<f32>() / window as f32;
                let next_mean: f32 = salience[i..i + window].iter().sum::<f32>() / window as f32;
                let gradient = (next_mean - prev_mean).abs();

                let far_enough = timestamps[i] - last_time > min_gap;
                if gradient > mean * 0.5 && far_enough {
                    boundaries.push((timestamps[i], i));
                    last_time = timestamps[i];
                }
            }
        }

        // Method 3: Fallback to regular intervals if still no boundaries
        if boundaries.len() <= 1 {
            let interval = 0.5; // 500ms segments
            let mut t = interval;
            while t < *timestamps.last().unwrap_or(&0.0) {
                let idx = timestamps
                    .iter()
                    .position(|&ts| ts >= t)
                    .unwrap_or(salience.len() - 1);
                if idx > 0 && idx < salience.len() - 1 {
                    boundaries.push((t, idx));
                }
                t += interval;
            }
        }

        // Add end boundary
        let last_t = *timestamps.last().unwrap_or(&0.0);
        let last_f = salience.len().saturating_sub(1);
        if boundaries
            .last()
            .map(|(t, _)| last_t - t > min_gap)
            .unwrap_or(true)
        {
            boundaries.push((last_t, last_f));
        }

        // Sort by time
        boundaries.sort_by(|a, b| a.0.total_cmp(&b.0));
        boundaries
    }

    /// Finalize and get results
    pub fn finalize(&mut self) -> DiscoveryResult {
        // Optional: merge similar units
        self.clusterer.merge_similar(0.75);

        // Collect stable units
        let mut units: Vec<DiscoveredUnit> =
            self.clusterer.stable_units().into_iter().cloned().collect();
        units.sort_by_key(|u| std::cmp::Reverse(u.instance_count));

        // Collect segment labels
        let segment_labels: Vec<(f32, f32, String)> = self
            .segments
            .iter()
            .filter_map(|s| {
                s.unit_id
                    .as_ref()
                    .map(|id| (s.start_time, s.end_time, id.clone()))
            })
            .collect();

        // Build transition matrix
        let mut transition_matrix: HashMap<(String, String), usize> = HashMap::new();
        for unit in &units {
            for (following, count) in &unit.following_units {
                transition_matrix.insert((unit.id.clone(), following.clone()), *count);
            }
        }

        // Compute statistics
        let total_duration_sec = self.segments.last().map(|s| s.end_time).unwrap_or(0.0);

        let mean_duration: f32 =
            units.iter().map(|u| u.mean_duration_ms).sum::<f32>() / units.len().max(1) as f32;

        // Shannon entropy of unit distribution
        let total_instances: usize = units.iter().map(|u| u.instance_count).sum();
        let entropy: f32 = if total_instances > 0 {
            -units
                .iter()
                .map(|u| {
                    let p = u.instance_count as f32 / total_instances as f32;
                    if p > 0.0 { p * p.ln() } else { 0.0 }
                })
                .sum::<f32>()
        } else {
            0.0
        };

        DiscoveryResult {
            units: units.clone(),
            segment_labels,
            transition_matrix,
            stats: DiscoveryStats {
                total_duration_sec,
                num_segments: self.segments.len(),
                num_units: self.clusterer.units().len(),
                num_stable_units: units.len(),
                mean_unit_duration_ms: mean_duration,
                unit_entropy: entropy,
            },
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &ClusteringStats {
        &self.clusterer.stats
    }

    /// Reset for new recording
    pub fn reset(&mut self) {
        self.segments.clear();
        self.clusterer.reset_session();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovered_unit() {
        let hv1 = HV16::random("sound1");
        let hv2 = HV16::random("sound1_variant");

        let mut unit = DiscoveredUnit::new("UNIT_001", hv1, 100.0);
        assert_eq!(unit.instance_count, 1);

        unit.add_instance(hv2, 90.0);
        assert_eq!(unit.instance_count, 2);
        assert!((unit.mean_duration_ms - 95.0).abs() < 0.1);
    }

    #[test]
    fn test_online_clusterer() {
        let config = DiscoveryConfig {
            similarity_threshold: 0.6,
            min_instances: 2,
            max_units: 10,
            ..Default::default()
        };

        let mut clusterer = OnlineClusterer::new(config);

        // Create similar segments
        let hv1 = HV16::random("click_pattern_1");
        let hv2 = HV16::random("click_pattern_1_var");
        let hv3 = HV16::random("whistle_pattern");

        let seg1 = AcousticSegment {
            start_time: 0.0,
            end_time: 0.1,
            start_frame: 0,
            end_frame: 10,
            hv: hv1,
            unit_id: None,
            boundary_salience: 1.0,
            spectral_centroid: None,
        };

        let seg2 = AcousticSegment {
            start_time: 0.2,
            end_time: 0.3,
            start_frame: 20,
            end_frame: 30,
            hv: hv2,
            unit_id: None,
            boundary_salience: 1.0,
            spectral_centroid: None,
        };

        let seg3 = AcousticSegment {
            start_time: 0.4,
            end_time: 0.5,
            start_frame: 40,
            end_frame: 50,
            hv: hv3,
            unit_id: None,
            boundary_salience: 1.0,
            spectral_centroid: None,
        };

        let id1 = clusterer.process_segment(&seg1);
        let id2 = clusterer.process_segment(&seg2);
        let id3 = clusterer.process_segment(&seg3);

        // hv1 and hv2 might cluster together (depending on similarity)
        // hv3 should be different
        assert!(clusterer.units().len() >= 1);
        assert!(clusterer.units().len() <= 3);
    }

    #[test]
    fn test_grounding() {
        let mut grounding = GroundingEngine::new(0.6);

        let acoustic = HV16::random("alarm_call");
        let context = HV16::random("predator_visual");

        grounding.learn_grounding("UNIT_001", &acoustic, &context, Some("predator"));

        let results = grounding.query_acoustic("UNIT_001");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].label, Some("predator".to_string()));
    }

    #[test]
    fn test_discovery_result_json() {
        let result = DiscoveryResult {
            units: vec![DiscoveredUnit::new("UNIT_001", HV16::zero(), 100.0)],
            segment_labels: vec![(0.0, 0.1, "UNIT_001".to_string())],
            transition_matrix: HashMap::new(),
            stats: DiscoveryStats::default(),
        };

        let json = result.to_json();
        assert!(json.contains("UNIT_001"));
        assert!(json.contains("\"units\""));
    }
}
