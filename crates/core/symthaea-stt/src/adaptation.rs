// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Speaker Adaptation Module
//!
//! Online adaptation to new speakers using:
//! - Mean adaptation (shift prototypes toward speaker's acoustic space)
//! - Variance normalization (compensate for speaker vocal tract differences)
//! - Prototype refinement (online learning of speaker-specific patterns)
//!
//! # Key Insight
//!
//! HDC's algebraic properties make adaptation elegant:
//! - Binding creates speaker-specific transforms
//! - Bundling accumulates adaptation evidence
//! - Similarity-preserving means adapted prototypes still work

use crate::hdc::{BundleAccumulator, HV16, bundle};
use std::collections::HashMap;

/// Speaker profile containing adaptation parameters
#[derive(Clone, Debug)]
pub struct SpeakerProfile {
    /// Speaker identifier
    pub id: String,
    /// Speaker transform hypervector (bind with prototypes)
    pub transform: HV16,
    /// Accumulated acoustic evidence per phoneme
    accumulators: HashMap<String, BundleAccumulator>,
    /// Number of observations per phoneme
    counts: HashMap<String, usize>,
    /// Running mean of acoustic features (for normalization)
    feature_mean: Vec<f32>,
    /// Running variance of acoustic features
    feature_var: Vec<f32>,
    /// Number of frames observed
    num_frames: usize,
    /// Adaptation weight (0 = no adaptation, 1 = full adaptation)
    pub adaptation_weight: f32,
}

impl SpeakerProfile {
    /// Create a new speaker profile
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            transform: HV16::zero(),
            accumulators: HashMap::new(),
            counts: HashMap::new(),
            feature_mean: Vec::new(),
            feature_var: Vec::new(),
            num_frames: 0,
            adaptation_weight: 0.0,
        }
    }

    /// Create from an existing speaker's acoustic samples
    pub fn from_samples(id: &str, samples: &[(String, HV16)]) -> Self {
        let mut profile = Self::new(id);

        for (phoneme, hv) in samples {
            profile.observe_phoneme(phoneme, hv);
        }

        profile.finalize_transform();
        profile
    }

    /// Observe a phoneme instance from this speaker
    pub fn observe_phoneme(&mut self, phoneme: &str, acoustic_hv: &HV16) {
        self.accumulators
            .entry(phoneme.to_string())
            .or_default()
            .add(acoustic_hv);

        *self.counts.entry(phoneme.to_string()).or_insert(0) += 1;
    }

    /// Observe raw acoustic features (for normalization)
    pub fn observe_features(&mut self, features: &[f32]) {
        if self.feature_mean.is_empty() {
            self.feature_mean = features.to_vec();
            self.feature_var = vec![1.0; features.len()];
            self.num_frames = 1;
            return;
        }

        // Online mean and variance update (Welford's algorithm)
        self.num_frames += 1;
        let n = self.num_frames as f32;

        for (i, &x) in features.iter().enumerate() {
            if i >= self.feature_mean.len() {
                break;
            }

            let delta = x - self.feature_mean[i];
            self.feature_mean[i] += delta / n;
            let delta2 = x - self.feature_mean[i];
            self.feature_var[i] += delta * delta2;
        }
    }

    /// Finalize the speaker transform after observations
    pub fn finalize_transform(&mut self) {
        // Create transform by bundling all speaker-specific patterns
        let speaker_patterns: Vec<HV16> = self
            .accumulators
            .values()
            .map(|acc| acc.finalize())
            .collect();

        if !speaker_patterns.is_empty() {
            self.transform = bundle(&speaker_patterns);

            // Set adaptation weight based on evidence
            let total_obs: usize = self.counts.values().sum();
            self.adaptation_weight = (total_obs as f32 / 100.0).min(1.0);
        }
    }

    /// Get normalized features using speaker statistics
    pub fn normalize_features(&self, features: &[f32]) -> Vec<f32> {
        if self.num_frames < 10 || self.feature_mean.is_empty() {
            return features.to_vec();
        }

        features
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i < self.feature_mean.len() {
                    let var = self.feature_var[i] / self.num_frames as f32;
                    let std = var.sqrt().max(1e-6);
                    (x - self.feature_mean[i]) / std
                } else {
                    x
                }
            })
            .collect()
    }

    /// Get speaker-adapted version of a prototype
    pub fn adapt_prototype(&self, prototype: &HV16) -> HV16 {
        if self.adaptation_weight < 0.01 {
            return *prototype;
        }

        // Bind prototype with speaker transform to create adapted version
        let adapted = prototype.bind(&self.transform);

        // Interpolate between original and adapted based on weight
        // Since we can't directly interpolate binary vectors, we use
        // a probabilistic approach: flip bits with probability (1-weight)
        if self.adaptation_weight >= 0.99 {
            adapted
        } else {
            // For partial adaptation, bundle original and adapted
            let orig_weight = 1.0 - self.adaptation_weight;
            let mut acc = BundleAccumulator::new();
            acc.add_weighted(prototype, orig_weight);
            acc.add_weighted(&adapted, self.adaptation_weight);
            acc.finalize()
        }
    }

    /// Get phoneme-specific adapted prototype if available
    pub fn get_adapted_phoneme(&self, phoneme: &str, base_prototype: &HV16) -> HV16 {
        if let Some(acc) = self.accumulators.get(phoneme)
            && self.counts.get(phoneme).copied().unwrap_or(0) >= 3
        {
            // Have enough speaker-specific examples
            let speaker_proto = acc.finalize();

            // Blend base prototype with speaker-specific
            let mut blend_acc = BundleAccumulator::new();
            blend_acc.add_weighted(base_prototype, 1.0 - self.adaptation_weight);
            blend_acc.add_weighted(&speaker_proto, self.adaptation_weight);
            return blend_acc.finalize();
        }

        // Fall back to general adaptation
        self.adapt_prototype(base_prototype)
    }

    /// Number of phoneme observations
    pub fn total_observations(&self) -> usize {
        self.counts.values().sum()
    }

    /// Get observation count for a specific phoneme
    pub fn phoneme_count(&self, phoneme: &str) -> usize {
        self.counts.get(phoneme).copied().unwrap_or(0)
    }
}

/// Speaker adaptation engine managing multiple speaker profiles
pub struct AdaptationEngine {
    /// Known speaker profiles
    profiles: HashMap<String, SpeakerProfile>,
    /// Current active speaker
    active_speaker: Option<String>,
    /// Base prototypes (unadapted)
    base_prototypes: HashMap<String, HV16>,
    /// Minimum observations before using adaptation
    min_observations: usize,
}

impl AdaptationEngine {
    /// Create a new adaptation engine
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            active_speaker: None,
            base_prototypes: HashMap::new(),
            min_observations: 10,
        }
    }

    /// Load base prototypes
    pub fn load_prototypes(&mut self, prototypes: &[(String, HV16)]) {
        self.base_prototypes.clear();
        for (label, hv) in prototypes {
            self.base_prototypes.insert(label.clone(), *hv);
        }
    }

    /// Create or get a speaker profile
    pub fn get_or_create_speaker(&mut self, speaker_id: &str) -> &mut SpeakerProfile {
        if !self.profiles.contains_key(speaker_id) {
            self.profiles
                .insert(speaker_id.to_string(), SpeakerProfile::new(speaker_id));
        }
        self.profiles.get_mut(speaker_id).unwrap()
    }

    /// Set active speaker
    pub fn set_active_speaker(&mut self, speaker_id: &str) {
        if !self.profiles.contains_key(speaker_id) {
            self.profiles
                .insert(speaker_id.to_string(), SpeakerProfile::new(speaker_id));
        }
        self.active_speaker = Some(speaker_id.to_string());
    }

    /// Get adapted prototype for current speaker
    pub fn get_adapted_prototype(&self, phoneme: &str) -> Option<HV16> {
        let base = self.base_prototypes.get(phoneme)?;

        if let Some(ref speaker_id) = self.active_speaker
            && let Some(profile) = self.profiles.get(speaker_id)
            && profile.total_observations() >= self.min_observations
        {
            return Some(profile.get_adapted_phoneme(phoneme, base));
        }

        Some(*base)
    }

    /// Observe phoneme from current speaker (online learning)
    pub fn observe(&mut self, phoneme: &str, acoustic_hv: &HV16) {
        if let Some(ref speaker_id) = self.active_speaker
            && let Some(profile) = self.profiles.get_mut(speaker_id)
        {
            profile.observe_phoneme(phoneme, acoustic_hv);

            // Periodically update transform
            if profile.total_observations() % 50 == 0 {
                profile.finalize_transform();
            }
        }
    }

    /// Observe features for normalization
    pub fn observe_features(&mut self, features: &[f32]) {
        if let Some(ref speaker_id) = self.active_speaker
            && let Some(profile) = self.profiles.get_mut(speaker_id)
        {
            profile.observe_features(features);
        }
    }

    /// Normalize features for current speaker
    pub fn normalize_features(&self, features: &[f32]) -> Vec<f32> {
        if let Some(ref speaker_id) = self.active_speaker
            && let Some(profile) = self.profiles.get(speaker_id)
        {
            return profile.normalize_features(features);
        }
        features.to_vec()
    }

    /// Get all adapted prototypes for current speaker
    pub fn get_all_adapted(&self) -> Vec<(String, HV16)> {
        self.base_prototypes
            .keys()
            .filter_map(|phoneme| {
                self.get_adapted_prototype(phoneme)
                    .map(|adapted| (phoneme.clone(), adapted))
            })
            .collect()
    }

    /// Number of known speakers
    pub fn num_speakers(&self) -> usize {
        self.profiles.len()
    }

    /// Get speaker profile
    pub fn get_speaker(&self, speaker_id: &str) -> Option<&SpeakerProfile> {
        self.profiles.get(speaker_id)
    }

    /// Identify likely speaker from acoustic sample
    /// Returns (speaker_id, confidence) sorted by confidence
    pub fn identify_speaker(&self, acoustic_hv: &HV16) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = self
            .profiles
            .iter()
            .filter(|(_, profile)| profile.total_observations() >= self.min_observations)
            .map(|(id, profile)| {
                let sim = acoustic_hv.similarity(&profile.transform);
                (id.clone(), sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}

impl Default for AdaptationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Unsupervised speaker diarization
/// Clusters acoustic segments by speaker without labels
pub struct SpeakerDiarizer {
    /// Discovered speaker clusters
    clusters: Vec<SpeakerCluster>,
    /// Similarity threshold for same speaker
    threshold: f32,
    /// Maximum speakers to track
    max_speakers: usize,
}

#[derive(Clone)]
struct SpeakerCluster {
    /// Cluster centroid
    centroid: HV16,
    /// Accumulator for updating centroid
    accumulator: BundleAccumulator,
    /// Number of segments assigned
    count: usize,
    /// Recent segment timestamps
    recent_times: Vec<f32>,
}

impl SpeakerDiarizer {
    /// Create a new diarizer
    pub fn new(threshold: f32, max_speakers: usize) -> Self {
        Self {
            clusters: Vec::new(),
            threshold,
            max_speakers,
        }
    }

    /// Process an acoustic segment
    /// Returns assigned speaker index (0-based)
    pub fn process(&mut self, segment_hv: &HV16, timestamp: f32) -> usize {
        // Find best matching cluster
        let mut best_match: Option<(usize, f32)> = None;

        for (i, cluster) in self.clusters.iter().enumerate() {
            let sim = segment_hv.similarity(&cluster.centroid);
            if sim > self.threshold && best_match.as_ref().map(|(_, s)| sim > *s).unwrap_or(true) {
                best_match = Some((i, sim));
            }
        }

        if let Some((idx, _)) = best_match {
            // Assign to existing cluster
            self.clusters[idx].accumulator.add(segment_hv);
            self.clusters[idx].count += 1;
            self.clusters[idx].centroid = self.clusters[idx].accumulator.finalize();
            self.clusters[idx].recent_times.push(timestamp);
            if self.clusters[idx].recent_times.len() > 100 {
                self.clusters[idx].recent_times.remove(0);
            }
            idx
        } else if self.clusters.len() < self.max_speakers {
            // Create new cluster
            let mut accumulator = BundleAccumulator::new();
            accumulator.add(segment_hv);

            self.clusters.push(SpeakerCluster {
                centroid: *segment_hv,
                accumulator,
                count: 1,
                recent_times: vec![timestamp],
            });
            self.clusters.len() - 1
        } else {
            // At capacity - assign to closest
            let (closest_idx, _) = self
                .clusters
                .iter()
                .enumerate()
                .map(|(i, c)| (i, segment_hv.similarity(&c.centroid)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, 0.0));

            self.clusters[closest_idx].accumulator.add(segment_hv);
            self.clusters[closest_idx].count += 1;
            self.clusters[closest_idx].centroid = self.clusters[closest_idx].accumulator.finalize();
            closest_idx
        }
    }

    /// Get number of discovered speakers
    pub fn num_speakers(&self) -> usize {
        self.clusters.len()
    }

    /// Get segment count per speaker
    pub fn speaker_counts(&self) -> Vec<usize> {
        self.clusters.iter().map(|c| c.count).collect()
    }

    /// Merge similar clusters
    pub fn merge_similar(&mut self, merge_threshold: f32) {
        let mut i = 0;
        while i < self.clusters.len() {
            let mut j = i + 1;
            while j < self.clusters.len() {
                let sim = self.clusters[i]
                    .centroid
                    .similarity(&self.clusters[j].centroid);
                if sim > merge_threshold {
                    // Copy data from j before merging
                    let centroid_j = self.clusters[j].centroid;
                    let count_j = self.clusters[j].count;

                    // Merge j into i
                    for _ in 0..count_j.min(10) {
                        self.clusters[i].accumulator.add(&centroid_j);
                    }
                    self.clusters[i].count += count_j;
                    self.clusters[i].centroid = self.clusters[i].accumulator.finalize();
                    self.clusters.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    /// Reset diarizer
    pub fn reset(&mut self) {
        self.clusters.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speaker_profile() {
        let mut profile = SpeakerProfile::new("speaker_1");

        // Observe some phonemes
        let hv1 = HV16::random("phoneme_p_speaker1_1");
        let hv2 = HV16::random("phoneme_p_speaker1_2");

        profile.observe_phoneme("p", &hv1);
        profile.observe_phoneme("p", &hv2);

        assert_eq!(profile.phoneme_count("p"), 2);
        assert_eq!(profile.total_observations(), 2);
    }

    #[test]
    fn test_adaptation_engine() {
        let mut engine = AdaptationEngine::new();

        // Load base prototypes
        let protos = vec![
            ("p".to_string(), HV16::random("base_p")),
            ("b".to_string(), HV16::random("base_b")),
        ];
        engine.load_prototypes(&protos);

        // Set active speaker
        engine.set_active_speaker("alice");

        // Get unadapted prototype (not enough observations)
        let adapted = engine.get_adapted_prototype("p").unwrap();
        assert_eq!(adapted.similarity(&protos[0].1), 1.0);
    }

    #[test]
    fn test_speaker_diarization() {
        let mut diarizer = SpeakerDiarizer::new(0.6, 10);

        // Simulate two different speakers
        let speaker1_hv = HV16::random("speaker1_pattern");
        let speaker2_hv = HV16::random("speaker2_pattern");

        // Process segments
        let s1 = diarizer.process(&speaker1_hv, 0.0);
        let s2 = diarizer.process(&speaker2_hv, 1.0);
        let s3 = diarizer.process(&speaker1_hv, 2.0); // Should match speaker 1

        assert_eq!(s1, s3, "Same speaker pattern should cluster together");
        assert_ne!(s1, s2, "Different speakers should be in different clusters");
        assert_eq!(diarizer.num_speakers(), 2);
    }
}
