// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Jungian Shadow Work — Computational Model
//!
//! Maps Jung's shadow dynamics to Fristonian predictive coding:
//!
//! - **Shadow content** = high-emotional-charge fragments that are NOT integrated
//!   into the conscious narrative, evidenced by persistent prediction errors
//! - **Shadow pressure** = accumulated prediction error from unintegrated content;
//!   rises the longer material is "repressed" (not integrated)
//! - **Shadow projection** = when shadow-encoded qualities appear in descriptions
//!   of others (high HDC similarity between shadow fragments and other-descriptions)
//! - **Shadow surfacing** = when shadow pressure exceeds a threshold, indicating
//!   the unconscious material is "demanding" integration
//!
//! ## Observability Mode
//!
//! This module operates in **observability mode only**. It tracks and measures
//! shadow dynamics but does NOT automatically trigger interventions or alter
//! Broca outputs. All signals are exposed via `ShadowTelemetry` for human
//! observation and validation before any active intervention is wired.
//!
//! ## Theoretical Grounding
//!
//! - Jung (1951) — *Aion: Researches into the Phenomenology of the Self*
//! - Friston (2010) — *The free-energy principle: a unified brain theory?*
//! - Carhart-Harris & Friston (2019) — *REBUS and the anarchic brain*
//! - Dehaene & Changeux (2011) — *Experimental and theoretical approaches to
//!   conscious processing* (GWT workspace exclusion = repression)

use crate::semantic_encoding::encode_therapeutic_text;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use symthaea_core::hdc::BinaryHV;

// ── Shadow Fragment ──────────────────────────────────────────────────────

/// A fragment of psychic content that has been excluded from the conscious
/// narrative — the computational analogue of Jungian shadow material.
///
/// Shadow fragments are identified by the conjunction of:
/// 1. High emotional charge (|valence| > threshold)
/// 2. Low narrative integration (integration_level < threshold)
/// 3. Persistent prediction error (the system keeps being "surprised" by
///    content related to this fragment)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowFragment {
    /// Stable identifier used by persistence and dream-queue references.
    #[serde(default)]
    pub fragment_id: u64,

    /// Text/content that was excluded from the narrative.
    pub content: String,

    /// Cycle when this shadow fragment was first detected.
    pub first_detected_cycle: u64,

    /// Cycle of most recent prediction error associated with this fragment.
    pub last_active_cycle: u64,

    /// Emotional valence of the shadow content (-1 to +1).
    /// Shadow material typically has strong negative valence.
    pub emotional_valence: f32,

    /// Emotional arousal associated with this content (0 to 1).
    pub emotional_arousal: f32,

    /// Number of cycles this fragment has generated prediction errors
    /// without being integrated into the narrative.
    pub recurrence_count: u32,

    /// Accumulated prediction error from this fragment.
    /// Grows each time related content generates surprise.
    /// This IS shadow pressure at the fragment level.
    pub cumulative_prediction_error: f32,

    /// Mean prediction error per recurrence (cumulative / recurrence_count).
    pub mean_prediction_error: f32,

    /// HDC encoding of the shadow content (bag-of-words compositional).
    #[serde(skip)]
    pub encoding: Option<BinaryHV>,

    /// Whether this fragment has been marked for dream processing.
    pub queued_for_dream: bool,

    /// Number of times this fragment appeared in dream counterfactuals.
    pub dream_processing_count: u32,

    /// Best Phi improvement found in dream simulations involving this fragment.
    /// Positive = integration would improve consciousness quality.
    pub dream_phi_improvement: f32,
}

impl ShadowFragment {
    /// Create a new shadow fragment from excluded narrative content.
    pub fn new(
        content: &str,
        cycle: u64,
        emotional_valence: f32,
        emotional_arousal: f32,
        prediction_error: f32,
    ) -> Self {
        let encoding = encode_therapeutic_text(content);
        let fragment_id = stable_fragment_id(content, cycle, 0);

        Self {
            fragment_id,
            content: content.to_string(),
            first_detected_cycle: cycle,
            last_active_cycle: cycle,
            emotional_valence: emotional_valence.clamp(-1.0, 1.0),
            emotional_arousal: emotional_arousal.clamp(0.0, 1.0),
            recurrence_count: 1,
            cumulative_prediction_error: prediction_error,
            mean_prediction_error: prediction_error,
            encoding,
            queued_for_dream: false,
            dream_processing_count: 0,
            dream_phi_improvement: 0.0,
        }
    }

    /// Record another recurrence of this shadow content.
    pub fn record_recurrence(&mut self, cycle: u64, prediction_error: f32) {
        self.last_active_cycle = cycle;
        self.recurrence_count += 1;
        self.cumulative_prediction_error += prediction_error;
        self.mean_prediction_error =
            self.cumulative_prediction_error / self.recurrence_count as f32;
    }

    /// Shadow pressure: how much this fragment is "demanding" integration.
    ///
    /// Pressure = cumulative_PE × recurrence × (1 + |valence|)
    ///
    /// High emotional charge amplifies pressure (Jung: the more something
    /// is repressed, the more psychic energy it accumulates).
    pub fn pressure(&self) -> f32 {
        self.cumulative_prediction_error
            * (self.recurrence_count as f32).sqrt()
            * (1.0 + self.emotional_valence.abs())
    }

    /// How long this fragment has been in the shadow (cycles since first detection).
    pub fn shadow_duration(&self, current_cycle: u64) -> u64 {
        current_cycle.saturating_sub(self.first_detected_cycle)
    }

    /// HDC similarity to another encoding.
    pub fn similarity_to(&self, other: &BinaryHV) -> f32 {
        self.encoding
            .as_ref()
            .map(|e| e.similarity(other))
            .unwrap_or(0.0)
    }

    /// Record dream processing results.
    pub fn record_dream_result(&mut self, phi_improvement: f32) {
        self.dream_processing_count += 1;
        if phi_improvement > self.dream_phi_improvement {
            self.dream_phi_improvement = phi_improvement;
        }
    }
}

// ── Shadow Telemetry ─────────────────────────────────────────────────────

/// Telemetry output for shadow dynamics — observability mode.
///
/// All fields are diagnostic: they describe what the shadow detector
/// has observed, NOT what actions it has taken (it takes none).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShadowTelemetry {
    /// Total shadow pressure across all fragments (sum of individual pressures).
    pub total_shadow_pressure: f32,

    /// Number of active shadow fragments.
    pub shadow_fragment_count: u32,

    /// Highest individual fragment pressure (the "loudest" shadow).
    pub peak_fragment_pressure: f32,

    /// Mean prediction error for shadow-related content.
    pub shadow_mean_prediction_error: f32,

    /// Number of possible projection events detected this cycle.
    pub projection_detections: u32,

    /// Whether shadow pressure exceeds the surfacing threshold.
    /// In observability mode, this is diagnostic only — no action is taken.
    pub surfacing_indicated: bool,

    /// Number of shadow fragments queued for dream processing.
    pub dream_queue_depth: u32,

    /// Best Phi improvement from dream-processed shadow content.
    pub best_dream_phi_improvement: f32,

    /// Shadow pressure trend: positive = rising (material accumulating),
    /// negative = falling (material integrating or decaying).
    pub pressure_trend: f32,

    /// Ratio of shadow content to total narrative content.
    /// High ratio suggests significant unintegrated material.
    pub shadow_to_narrative_ratio: f32,
}

// ── Shadow Snapshot (Persistence) ────────────────────────────────────────

/// Serializable snapshot of shadow detector state for session persistence.
///
/// Captures active fragments, pressure history, and dream queue.
/// HDC encodings within fragments are skipped (serde(skip) on BinaryHV).
/// Config thresholds are NOT included — they come from defaults on restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowSnapshot {
    /// Snapshot schema version. Version 2 introduced stable fragment IDs.
    #[serde(default)]
    pub version: u16,
    /// Active shadow fragments (HDC encodings are rebuilt on restore).
    pub fragments: Vec<ShadowFragment>,
    /// Rolling pressure history for trend computation.
    pub pressure_history: Vec<f32>,
    /// Stable fragment IDs queued for dream processing.
    #[serde(default)]
    pub dream_queue_fragment_ids: Vec<u64>,
    /// Legacy version-1 queue indices, accepted only for migration.
    #[serde(default, rename = "dream_queue", skip_serializing_if = "Vec::is_empty")]
    pub legacy_dream_queue_indices: Vec<usize>,
}

// ── Shadow Detector ──────────────────────────────────────────────────────

/// Detects and tracks Jungian shadow dynamics using predictive coding signals.
///
/// **Observability mode**: tracks metrics only, does NOT intervene.
///
/// The detector identifies shadow content by monitoring prediction errors
/// that correlate with high-emotional-charge, low-integration narrative
/// fragments. When the same content repeatedly generates surprise without
/// being integrated into the conscious narrative, it accumulates as shadow.
pub struct ShadowDetector {
    /// Active shadow fragments.
    fragments: Vec<ShadowFragment>,

    /// Maximum number of shadow fragments to track (prevents unbounded growth).
    max_fragments: usize,

    /// HDC similarity threshold for matching input to existing shadow fragments.
    /// Above this = same shadow content recurring.
    similarity_threshold: f32,

    /// Minimum |valence| for content to be shadow-eligible.
    /// Neutral content doesn't become shadow — only emotionally charged material.
    valence_threshold: f32,

    /// Minimum prediction error for a cycle to contribute to shadow accumulation.
    /// Below this, the error is likely system noise, not meaningful signal.
    prediction_error_floor: f32,

    /// Shadow pressure at which surfacing is indicated.
    surfacing_threshold: f32,

    /// Rolling window of total shadow pressure for trend computation.
    pressure_history: VecDeque<f32>,

    /// Maximum history length.
    pressure_history_max: usize,

    /// Projection detection: similarity threshold for matching shadow content
    /// against descriptions of others.
    projection_similarity_threshold: f32,

    /// Dream queue: stable fragment IDs to feed to the dream engine.
    dream_queue: VecDeque<u64>,

    /// Maximum dream queue depth.
    dream_queue_max: usize,
}

impl ShadowDetector {
    /// Create a new shadow detector with default parameters.
    pub fn new() -> Self {
        Self {
            fragments: Vec::new(),
            max_fragments: 64,
            similarity_threshold: 0.60,
            valence_threshold: 0.3,
            prediction_error_floor: 0.15,
            surfacing_threshold: 5.0,
            pressure_history: VecDeque::new(),
            pressure_history_max: 50,
            projection_similarity_threshold: 0.55,
            dream_queue: VecDeque::new(),
            dream_queue_max: 8,
        }
    }

    /// Process a cycle's signals to detect and track shadow dynamics.
    ///
    /// This is called each time the TherapeuticManager fires (interval 11).
    /// It examines prediction error, affect, and narrative state to identify
    /// shadow content and track its accumulation.
    ///
    /// Returns updated telemetry (observability only — no side effects on
    /// the cognitive loop).
    pub fn process(
        &mut self,
        cycle: u64,
        prediction_error: f32,
        valence: f32,
        arousal: f32,
        input_text: &str,
        narrative_integration: f32,
    ) -> ShadowTelemetry {
        // ── 1. Filter: is this cycle's content shadow-eligible? ──────────
        let emotionally_charged = valence.abs() > self.valence_threshold;
        let surprising = prediction_error > self.prediction_error_floor;
        let poorly_integrated = narrative_integration < 0.4;

        if emotionally_charged && surprising && poorly_integrated && !input_text.is_empty() {
            self.accumulate_or_create(cycle, prediction_error, valence, arousal, input_text);
        }

        // ── 2. Decay dormant fragments ───────────────────────────────────
        // Shadow that hasn't recurred for 200+ cycles loses pressure slowly
        // (spontaneous integration / natural forgetting)
        for frag in &mut self.fragments {
            if cycle.saturating_sub(frag.last_active_cycle) > 200 {
                frag.cumulative_prediction_error *= 0.98; // 2% decay per check
            }
        }

        // ── 3. Evict depleted fragments ──────────────────────────────────
        self.fragments.retain(|f| f.pressure() > 0.01);

        // ── 4. Queue high-pressure fragments for dream processing ────────
        self.update_dream_queue();

        // ── 5. Compute telemetry ─────────────────────────────────────────
        self.compute_telemetry(cycle)
    }

    /// Check for shadow projection: does the input text resemble any
    /// shadow fragment? If so, the client may be projecting shadow content
    /// onto others.
    ///
    /// Returns the number of projection matches and their details.
    pub fn detect_projections(&self, input_text: &str) -> Vec<ProjectionEvent> {
        if input_text.is_empty() || self.fragments.is_empty() {
            return vec![];
        }

        let Some(input_hv) = encode_therapeutic_text(input_text) else {
            return vec![];
        };

        let mut projections = Vec::new();
        for (i, frag) in self.fragments.iter().enumerate() {
            let sim = frag.similarity_to(&input_hv);
            if sim > self.projection_similarity_threshold {
                projections.push(ProjectionEvent {
                    shadow_fragment_id: frag.fragment_id,
                    shadow_fragment_index: i,
                    shadow_content: frag.content.clone(),
                    similarity: sim,
                    shadow_pressure: frag.pressure(),
                });
            }
        }
        projections
    }

    /// Get the next shadow fragment to feed to the dream engine.
    ///
    /// Returns the fragment's content and encoding as a dream action vector.
    /// The dream engine will generate counterfactuals: "what if this shadow
    /// content were integrated rather than repressed?"
    pub fn next_dream_fragment(&mut self) -> Option<DreamShadowInput> {
        while let Some(fragment_id) = self.dream_queue.pop_front() {
            let Some(idx) = self
                .fragments
                .iter()
                .position(|fragment| fragment.fragment_id == fragment_id)
            else {
                continue;
            };
            let frag = &mut self.fragments[idx];
            frag.queued_for_dream = false;

            return Some(DreamShadowInput {
                fragment_id,
                fragment_index: idx,
                content: frag.content.clone(),
                emotional_valence: frag.emotional_valence,
                emotional_arousal: frag.emotional_arousal,
                shadow_pressure: frag.pressure(),
                // Dream state vector: [valence, arousal, pressure, duration_normalized]
                state_vector: vec![
                    frag.emotional_valence,
                    frag.emotional_arousal,
                    frag.pressure().min(10.0) / 10.0, // normalize to [0,1]
                    (frag.recurrence_count as f32).min(50.0) / 50.0,
                ],
                // Dream action vector: [integration_attempt, exploration, containment]
                action_vector: vec![1.0, 0.5, 0.3],
            });
        }
        None
    }

    /// Record dream results using the fragment's stable identifier.
    pub fn record_dream_result_by_id(&mut self, fragment_id: u64, phi_improvement: f32) -> bool {
        if let Some(frag) = self
            .fragments
            .iter_mut()
            .find(|fragment| fragment.fragment_id == fragment_id)
        {
            frag.record_dream_result(phi_improvement);
            true
        } else {
            false
        }
    }

    /// Record dream results by transient vector index.
    ///
    /// Prefer [`Self::record_dream_result_by_id`]; indices can change when
    /// fragments are evicted or compacted.
    #[deprecated(note = "use record_dream_result_by_id with DreamShadowInput::fragment_id")]
    pub fn record_dream_result(&mut self, fragment_index: usize, phi_improvement: f32) {
        if let Some(fragment_id) = self
            .fragments
            .get(fragment_index)
            .map(|fragment| fragment.fragment_id)
        {
            let _ = self.record_dream_result_by_id(fragment_id, phi_improvement);
        }
    }

    /// Number of active shadow fragments.
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// Total shadow pressure.
    pub fn total_pressure(&self) -> f32 {
        self.fragments.iter().map(|f| f.pressure()).sum()
    }

    /// Whether any fragment is surfacing-indicated.
    pub fn surfacing_indicated(&self) -> bool {
        self.total_pressure() > self.surfacing_threshold
    }

    /// Read-only access to shadow fragments (for telemetry/debugging).
    pub fn fragments(&self) -> &[ShadowFragment] {
        &self.fragments
    }

    /// Dream queue depth.
    pub fn dream_queue_depth(&self) -> usize {
        self.dream_queue.len()
    }

    /// Create a serializable snapshot of shadow state for session persistence.
    ///
    /// Captures fragments (with HDC vectors skipped via serde(skip)),
    /// pressure history, and dream queue state. Config thresholds are NOT
    /// persisted — they come from the detector's defaults on restore.
    pub fn snapshot(&self) -> ShadowSnapshot {
        ShadowSnapshot {
            version: 2,
            fragments: self.fragments.clone(),
            pressure_history: self.pressure_history.iter().copied().collect(),
            dream_queue_fragment_ids: self.dream_queue.iter().copied().collect(),
            legacy_dream_queue_indices: Vec::new(),
        }
    }

    /// Restore shadow state from a persisted snapshot.
    ///
    /// Rebuilds skipped HDC encodings, migrates legacy index queues to stable
    /// fragment IDs, removes stale queue entries, and synchronizes each
    /// fragment's `queued_for_dream` flag.
    pub fn restore(&mut self, snapshot: ShadowSnapshot) {
        let ShadowSnapshot {
            version: _,
            mut fragments,
            pressure_history,
            dream_queue_fragment_ids,
            legacy_dream_queue_indices,
        } = snapshot;

        let mut assigned_ids = std::collections::HashSet::new();
        for (ordinal, fragment) in fragments.iter_mut().enumerate() {
            fragment.encoding = encode_therapeutic_text(&fragment.content);
            if fragment.fragment_id == 0 || !assigned_ids.insert(fragment.fragment_id) {
                let mut salt = ordinal as u64;
                loop {
                    let candidate =
                        stable_fragment_id(&fragment.content, fragment.first_detected_cycle, salt);
                    if assigned_ids.insert(candidate) {
                        fragment.fragment_id = candidate;
                        break;
                    }
                    salt = salt.saturating_add(1);
                }
            }
            fragment.queued_for_dream = false;
        }

        let requested_ids: Vec<u64> = if dream_queue_fragment_ids.is_empty() {
            legacy_dream_queue_indices
                .into_iter()
                .filter_map(|index| fragments.get(index).map(|fragment| fragment.fragment_id))
                .collect()
        } else {
            dream_queue_fragment_ids
        };

        let mut seen_queue_ids = std::collections::HashSet::new();
        let queue_ids: Vec<u64> = requested_ids
            .into_iter()
            .filter(|fragment_id| assigned_ids.contains(fragment_id))
            .filter(|fragment_id| seen_queue_ids.insert(*fragment_id))
            .take(self.dream_queue_max)
            .collect();

        for fragment in &mut fragments {
            fragment.queued_for_dream = queue_ids.contains(&fragment.fragment_id);
        }

        self.fragments = fragments;
        self.pressure_history = VecDeque::from(pressure_history);
        let max = self.pressure_history_max;
        while self.pressure_history.len() > max {
            self.pressure_history.pop_front();
        }
        self.dream_queue = VecDeque::from(queue_ids);
    }

    // ── Private ──────────────────────────────────────────────────────────

    /// Find an existing fragment that matches this content, or create a new one.
    fn accumulate_or_create(
        &mut self,
        cycle: u64,
        prediction_error: f32,
        valence: f32,
        arousal: f32,
        input_text: &str,
    ) {
        let Some(input_hv) = encode_therapeutic_text(input_text) else {
            return;
        };

        // Try to find a matching existing fragment
        let mut best_match: Option<(usize, f32)> = None;
        for (i, frag) in self.fragments.iter().enumerate() {
            let sim = frag.similarity_to(&input_hv);
            if sim > self.similarity_threshold {
                if best_match.map_or(true, |(_, best_sim)| sim > best_sim) {
                    best_match = Some((i, sim));
                }
            }
        }

        if let Some((idx, _)) = best_match {
            // Existing shadow fragment — accumulate
            self.fragments[idx].record_recurrence(cycle, prediction_error);
        } else {
            // New shadow fragment
            if self.fragments.len() >= self.max_fragments {
                // Evict lowest-pressure fragment
                if let Some(min_idx) = self
                    .fragments
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.pressure()
                            .partial_cmp(&b.pressure())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    self.fragments.swap_remove(min_idx);
                }
            }
            self.fragments.push(ShadowFragment::new(
                input_text,
                cycle,
                valence,
                arousal,
                prediction_error,
            ));
        }
    }

    /// Update the dream queue with highest-pressure, unprocessed fragments.
    fn update_dream_queue(&mut self) {
        // Clear stale entries by stable identity, not transient vector index.
        let valid_fragment_ids: std::collections::HashSet<u64> = self
            .fragments
            .iter()
            .map(|fragment| fragment.fragment_id)
            .collect();
        self.dream_queue
            .retain(|fragment_id| valid_fragment_ids.contains(fragment_id));

        // Synchronize flags before selecting new candidates.
        let queued_fragment_ids: std::collections::HashSet<u64> =
            self.dream_queue.iter().copied().collect();
        for fragment in &mut self.fragments {
            fragment.queued_for_dream = queued_fragment_ids.contains(&fragment.fragment_id);
        }

        // Find high-pressure fragments not yet queued.
        let mut candidates: Vec<(u64, f32)> = self
            .fragments
            .iter()
            .filter(|fragment| !fragment.queued_for_dream && fragment.pressure() > 1.0)
            .map(|fragment| (fragment.fragment_id, fragment.pressure()))
            .collect();

        // Sort by pressure descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (fragment_id, _) in candidates {
            if self.dream_queue.len() >= self.dream_queue_max {
                break;
            }
            if !self.dream_queue.contains(&fragment_id) {
                self.dream_queue.push_back(fragment_id);
                if let Some(fragment) = self
                    .fragments
                    .iter_mut()
                    .find(|fragment| fragment.fragment_id == fragment_id)
                {
                    fragment.queued_for_dream = true;
                }
            }
        }
    }

    /// Compute telemetry snapshot.
    fn compute_telemetry(&mut self, _cycle: u64) -> ShadowTelemetry {
        let total_pressure = self.total_pressure();

        // Update pressure history for trend
        self.pressure_history.push_back(total_pressure);
        if self.pressure_history.len() > self.pressure_history_max {
            self.pressure_history.pop_front();
        }

        // Compute trend: slope of pressure over recent history
        let pressure_trend = if self.pressure_history.len() >= 3 {
            let n = self.pressure_history.len();
            let recent =
                self.pressure_history.iter().rev().take(5).sum::<f32>() / 5.0_f32.min(n as f32);
            let older = self.pressure_history.iter().take(5).sum::<f32>() / 5.0_f32.min(n as f32);
            recent - older
        } else {
            0.0
        };

        let peak = self
            .fragments
            .iter()
            .map(|f| f.pressure())
            .fold(0.0_f32, f32::max);

        let mean_pe = if self.fragments.is_empty() {
            0.0
        } else {
            self.fragments
                .iter()
                .map(|f| f.mean_prediction_error)
                .sum::<f32>()
                / self.fragments.len() as f32
        };

        let dream_queue_depth = self.dream_queue.len() as u32;
        let best_dream = self
            .fragments
            .iter()
            .map(|f| f.dream_phi_improvement)
            .fold(0.0_f32, f32::max);

        ShadowTelemetry {
            total_shadow_pressure: total_pressure,
            shadow_fragment_count: self.fragments.len() as u32,
            peak_fragment_pressure: peak,
            shadow_mean_prediction_error: mean_pe,
            projection_detections: 0, // Updated externally when detect_projections() is called
            surfacing_indicated: total_pressure > self.surfacing_threshold,
            dream_queue_depth,
            best_dream_phi_improvement: best_dream,
            pressure_trend,
            shadow_to_narrative_ratio: 0.0, // Set by caller who has narrative access
        }
    }
}

impl Default for ShadowDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Supporting Types ─────────────────────────────────────────────────────

/// A detected projection event: the client's current input resembles
/// their own shadow content, suggesting possible projection onto others.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionEvent {
    /// Stable identifier of the shadow fragment being projected.
    pub shadow_fragment_id: u64,
    /// Current vector index, retained for local inspection only.
    pub shadow_fragment_index: usize,
    /// Content of the shadow fragment.
    pub shadow_content: String,
    /// HDC similarity between current input and shadow content.
    pub similarity: f32,
    /// Current pressure of the shadow fragment.
    pub shadow_pressure: f32,
}

/// Input for the dream engine to process shadow content.
#[derive(Debug, Clone)]
pub struct DreamShadowInput {
    /// Stable identifier used when recording dream results.
    pub fragment_id: u64,
    /// Current index into the detector's fragment list (diagnostic only).
    pub fragment_index: usize,
    /// Text content of the shadow fragment.
    pub content: String,
    /// Emotional valence of the shadow content.
    pub emotional_valence: f32,
    /// Emotional arousal.
    pub emotional_arousal: f32,
    /// Current shadow pressure.
    pub shadow_pressure: f32,
    /// State vector for dream engine [valence, arousal, norm_pressure, norm_recurrence].
    pub state_vector: Vec<f32>,
    /// Action vector for dream engine [integration, exploration, containment].
    pub action_vector: Vec<f32>,
}

fn stable_fragment_id(content: &str, cycle: u64, salt: u64) -> u64 {
    let hash = blake3::hash(
        format!(
            "shadow_fragment:{cycle}:{salt}:{}",
            content.trim().to_lowercase()
        )
        .as_bytes(),
    );
    let mut id = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
    // Zero is reserved for snapshots created before stable IDs existed.
    if id == 0 {
        id = 1;
    }
    id
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_fragment_creation() {
        let frag = ShadowFragment::new("I'm terrified of abandonment", 10, -0.8, 0.9, 0.3);
        assert_eq!(frag.recurrence_count, 1);
        assert!((frag.cumulative_prediction_error - 0.3).abs() < 0.001);
        assert!(frag.encoding.is_some());
        assert!(!frag.queued_for_dream);
    }

    #[test]
    fn test_shadow_pressure_grows_with_recurrence() {
        let mut frag = ShadowFragment::new("rage at authority", 10, -0.9, 0.8, 0.25);
        let p1 = frag.pressure();

        frag.record_recurrence(20, 0.3);
        let p2 = frag.pressure();

        frag.record_recurrence(30, 0.35);
        let p3 = frag.pressure();

        assert!(p2 > p1, "pressure should grow: {} > {}", p2, p1);
        assert!(p3 > p2, "pressure should keep growing: {} > {}", p3, p2);
    }

    #[test]
    fn test_shadow_pressure_amplified_by_valence() {
        let mild = ShadowFragment::new("slight annoyance", 10, -0.2, 0.3, 0.3);
        let intense = ShadowFragment::new("overwhelming shame", 10, -0.9, 0.8, 0.3);

        // Same PE and recurrence, but intense valence amplifies pressure
        assert!(
            intense.pressure() > mild.pressure(),
            "intense valence should amplify pressure: {} > {}",
            intense.pressure(),
            mild.pressure(),
        );
    }

    #[test]
    fn test_detector_creates_shadow_from_charged_surprising_content() {
        let mut detector = ShadowDetector::new();
        assert_eq!(detector.fragment_count(), 0);

        // High charge (-0.8), surprising (0.4 > 0.15 floor), low integration (0.2)
        let _telemetry = detector.process(
            1,
            0.4,  // prediction_error
            -0.8, // valence (high charge)
            0.7,  // arousal
            "I'm furious at my father",
            0.2, // narrative_integration (low)
        );

        assert_eq!(detector.fragment_count(), 1);
    }

    #[test]
    fn test_detector_ignores_neutral_content() {
        let mut detector = ShadowDetector::new();

        // Low charge (0.1), not surprising enough
        let _telemetry = detector.process(
            1,
            0.05, // low prediction error
            0.1,  // neutral valence
            0.3,  // low arousal
            "the weather is fine",
            0.8, // high integration
        );

        assert_eq!(
            detector.fragment_count(),
            0,
            "neutral content should not become shadow"
        );
    }

    #[test]
    fn test_detector_accumulates_recurring_shadow() {
        let mut detector = ShadowDetector::new();

        // First occurrence
        detector.process(1, 0.35, -0.7, 0.6, "fear of rejection", 0.2);
        assert_eq!(detector.fragment_count(), 1);

        // Same content recurs (identical text → same HDC encoding → high similarity)
        detector.process(15, 0.4, -0.75, 0.65, "fear of rejection", 0.2);
        assert_eq!(
            detector.fragment_count(),
            1,
            "same content should accumulate, not create new fragment"
        );
        assert_eq!(detector.fragments()[0].recurrence_count, 2);
    }

    #[test]
    fn test_shadow_pressure_in_telemetry() {
        let mut detector = ShadowDetector::new();

        // Create shadow content
        detector.process(1, 0.5, -0.9, 0.8, "deep shame about failure", 0.1);
        let telemetry = detector.process(11, 0.45, -0.85, 0.75, "deep shame about failure", 0.15);

        assert!(
            telemetry.total_shadow_pressure > 0.0,
            "should have shadow pressure"
        );
        assert_eq!(telemetry.shadow_fragment_count, 1);
        assert!(telemetry.shadow_mean_prediction_error > 0.0);
    }

    #[test]
    fn test_surfacing_threshold() {
        let mut detector = ShadowDetector::new();

        // Build up significant shadow pressure
        for i in 0..20 {
            detector.process(i * 11, 0.5, -0.9, 0.8, "I hate myself for what I did", 0.1);
        }

        let telemetry = detector.process(221, 0.5, -0.9, 0.8, "I hate myself for what I did", 0.1);
        assert!(
            telemetry.surfacing_indicated,
            "20+ recurrences of high-charge content should indicate surfacing, pressure={}",
            telemetry.total_shadow_pressure,
        );
    }

    #[test]
    fn test_dream_queue_populated() {
        let mut detector = ShadowDetector::new();

        // Create high-pressure shadow
        for i in 0..10 {
            detector.process(i * 11, 0.5, -0.9, 0.8, "hidden rage", 0.1);
        }

        assert!(
            detector.dream_queue_depth() > 0,
            "high-pressure shadow should be queued for dreaming"
        );
    }

    #[test]
    fn test_next_dream_fragment() {
        let mut detector = ShadowDetector::new();

        for i in 0..10 {
            detector.process(i * 11, 0.5, -0.9, 0.8, "hidden rage", 0.1);
        }

        let dream_input = detector.next_dream_fragment();
        assert!(dream_input.is_some(), "should produce a dream input");

        let input = dream_input.unwrap();
        assert_eq!(input.state_vector.len(), 4);
        assert_eq!(input.action_vector.len(), 3);
        assert_ne!(input.fragment_id, 0);
        assert!(input.shadow_pressure > 0.0);
    }

    #[test]
    fn test_dream_result_recording() {
        let mut detector = ShadowDetector::new();

        for i in 0..10 {
            detector.process(i * 11, 0.5, -0.9, 0.8, "hidden rage", 0.1);
        }

        let fragment_id = detector.fragments()[0].fragment_id;
        assert!(detector.record_dream_result_by_id(fragment_id, 0.15));
        assert!((detector.fragments()[0].dream_phi_improvement - 0.15).abs() < 0.001,);
        assert_eq!(detector.fragments()[0].dream_processing_count, 1);
    }

    #[test]
    fn test_projection_detection() {
        let mut detector = ShadowDetector::new();

        // Create shadow about aggression
        for i in 0..5 {
            detector.process(i * 11, 0.4, -0.8, 0.7, "my violent anger", 0.1);
        }

        // Input describing someone else with the same content → projection
        let projections = detector.detect_projections("my violent anger toward him");
        // Note: exact text match gives high similarity
        // Whether this fires depends on blake3 hash similarity
        // At minimum, the function should not panic
        for p in &projections {
            assert!(p.similarity > detector.projection_similarity_threshold);
        }
    }

    #[test]
    fn test_decay_of_dormant_shadow() {
        let mut detector = ShadowDetector::new();

        // Create shadow
        detector.process(1, 0.5, -0.9, 0.8, "old wound", 0.1);
        let initial_pressure = detector.fragments()[0].pressure();

        // Simulate 300 cycles of dormancy (no recurrence)
        for c in 0..30 {
            // Process with benign content (won't match shadow)
            detector.process(
                200 + (c * 11),
                0.05, // low PE
                0.0,  // neutral
                0.3,
                "just a regular day",
                0.8,
            );
        }

        let final_pressure = detector
            .fragments()
            .first()
            .map(|f| f.pressure())
            .unwrap_or(0.0);
        assert!(
            final_pressure < initial_pressure,
            "dormant shadow should decay: {} < {}",
            final_pressure,
            initial_pressure,
        );
    }

    #[test]
    fn test_pressure_trend_tracking() {
        let mut detector = ShadowDetector::new();

        // Build rising pressure
        let mut last_telemetry = ShadowTelemetry::default();
        for i in 0..15 {
            last_telemetry = detector.process(
                i * 11,
                0.4 + (i as f32 * 0.01), // rising PE
                -0.8,
                0.7,
                "recurring nightmare content",
                0.15,
            );
        }

        // Trend should be non-negative (pressure is accumulating)
        assert!(
            last_telemetry.pressure_trend >= 0.0,
            "accumulating shadow should have non-negative trend, got {}",
            last_telemetry.pressure_trend,
        );
    }

    #[test]
    fn test_max_fragments_eviction() {
        let mut detector = ShadowDetector::new();
        detector.max_fragments = 5;

        // Create 6 distinct shadow fragments
        for i in 0..6 {
            detector.process(
                i * 11,
                0.5,
                -0.8,
                0.7,
                &format!("unique shadow content {}", i),
                0.1,
            );
        }

        assert!(
            detector.fragment_count() <= 5,
            "should evict to stay within max_fragments, got {}",
            detector.fragment_count(),
        );
    }

    #[test]
    fn test_empty_input_no_shadow() {
        let mut detector = ShadowDetector::new();
        detector.process(1, 0.5, -0.9, 0.8, "", 0.1);
        assert_eq!(
            detector.fragment_count(),
            0,
            "empty input should not create shadow"
        );
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        let mut det = ShadowDetector::new();
        for i in 0..15 {
            det.process(i * 10, 0.5, -0.8, 0.7, "recurring painful content", 0.1);
        }
        let pre_count = det.fragment_count();
        let pre_pressure = det.total_pressure();
        assert!(pre_count > 0);

        let snap = det.snapshot();
        let json = serde_json::to_string(&snap).expect("snapshot should serialize");

        let restored_snap: ShadowSnapshot =
            serde_json::from_str(&json).expect("snapshot should deserialize");

        let mut det2 = ShadowDetector::new();
        det2.restore(restored_snap);

        assert_eq!(det2.fragment_count(), pre_count);
        assert!((det2.total_pressure() - pre_pressure).abs() < 0.01);
        assert!(
            det2.fragments()
                .iter()
                .all(|fragment| fragment.fragment_id != 0)
        );
        assert!(
            det2.fragments()
                .iter()
                .all(|fragment| fragment.encoding.is_some())
        );

        let before_recurrence = det2.fragments()[0].recurrence_count;
        det2.process(200, 0.5, -0.8, 0.7, "recurring painful content again", 0.1);
        assert_eq!(det2.fragment_count(), pre_count);
        assert!(det2.fragments()[0].recurrence_count > before_recurrence);
    }

    #[test]
    fn test_legacy_snapshot_queue_indices_migrate_to_fragment_ids() {
        let fragment = ShadowFragment::new("legacy queued content", 10, -0.8, 0.7, 0.5);
        let snapshot = ShadowSnapshot {
            version: 0,
            fragments: vec![fragment],
            pressure_history: vec![1.0],
            dream_queue_fragment_ids: Vec::new(),
            legacy_dream_queue_indices: vec![0],
        };

        let mut detector = ShadowDetector::new();
        detector.restore(snapshot);
        let dream_input = detector
            .next_dream_fragment()
            .expect("legacy queue should migrate");
        assert_eq!(dream_input.content, "legacy queued content");
        assert_eq!(dream_input.fragment_id, detector.fragments()[0].fragment_id);
    }

    #[test]
    fn stable_dream_queue_survives_vector_reordering() {
        let mut detector = ShadowDetector::new();
        let first = ShadowFragment::new("first shadow", 1, -0.8, 0.7, 0.5);
        let second = ShadowFragment::new("second shadow", 2, -0.8, 0.7, 0.5);
        let second_id = second.fragment_id;
        detector.fragments = vec![first, second];
        detector.dream_queue.push_back(second_id);
        detector.fragments.swap_remove(0);

        let input = detector
            .next_dream_fragment()
            .expect("stable ID should survive swap_remove");
        assert_eq!(input.fragment_id, second_id);
        assert_eq!(input.content, "second shadow");
    }

    #[test]
    fn test_telemetry_all_finite() {
        let mut detector = ShadowDetector::new();
        for i in 0..10 {
            detector.process(i * 11, 0.4, -0.7, 0.6, "shadow content", 0.2);
        }
        let t = detector.process(110, 0.4, -0.7, 0.6, "shadow content", 0.2);

        assert!(t.total_shadow_pressure.is_finite());
        assert!(t.peak_fragment_pressure.is_finite());
        assert!(t.shadow_mean_prediction_error.is_finite());
        assert!(t.pressure_trend.is_finite());
        assert!(t.best_dream_phi_improvement.is_finite());
        assert!(t.shadow_to_narrative_ratio.is_finite());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Shadow pressure must never go negative regardless of inputs.
        #[test]
        fn pressure_never_negative(
            cycles in 1u64..200,
            pe in 0.0f32..2.0,
            valence in -1.0f32..1.0,
            arousal in 0.0f32..1.0,
            narrative_coh in 0.0f32..1.0,
        ) {
            let mut det = ShadowDetector::new();
            for c in 0..cycles {
                let t = det.process(c, pe, valence, arousal, "distress pain", narrative_coh);
                prop_assert!(
                    t.total_shadow_pressure >= 0.0,
                    "pressure was {} at cycle {}", t.total_shadow_pressure, c,
                );
            }
        }

        /// Fragment count must stay within the configured maximum (64).
        #[test]
        fn fragments_bounded(
            cycles in 1u64..500,
            pe in 0.1f32..1.5,
            valence in -1.0f32..0.0,
        ) {
            let mut det = ShadowDetector::new();
            for c in 0..cycles {
                det.process(c, pe, valence, 0.7, &format!("unique content {}", c), 0.2);
            }
            prop_assert!(
                det.fragment_count() <= 64,
                "fragments exceeded max: {}", det.fragment_count(),
            );
        }

        /// Dormant decay: after N inactive cycles, pressure must monotonically decrease.
        #[test]
        fn decay_monotonic(
            seed_cycles in 5u64..20,
            idle_cycles in 250u64..600,
        ) {
            let mut det = ShadowDetector::new();
            // Seed some shadow fragments
            for c in 0..seed_cycles {
                det.process(c, 0.8, -0.9, 0.7, "painful memory", 0.1);
            }
            let pressure_before_idle = det.process(seed_cycles, 0.8, -0.9, 0.7, "painful memory", 0.1)
                .total_shadow_pressure;

            // Run idle cycles with low PE (below threshold) — shadow should decay
            let mut last_pressure = pressure_before_idle;
            for c in (seed_cycles + 1)..(seed_cycles + 1 + idle_cycles) {
                let t = det.process(c, 0.01, 0.0, 0.1, "neutral calm ok", 0.9);
                // After enough idle cycles (200+ dormancy), pressure should stop growing
                if c > seed_cycles + 250 {
                    prop_assert!(
                        t.total_shadow_pressure <= last_pressure + 0.01,
                        "pressure grew after dormancy: {} -> {} at cycle {}",
                        last_pressure, t.total_shadow_pressure, c,
                    );
                }
                last_pressure = t.total_shadow_pressure;
            }
        }

        /// Dream queue depth is always ≤ 8 (MAX_DREAM_QUEUE).
        #[test]
        fn dream_queue_bounded(
            cycles in 1u64..300,
            pe in 0.2f32..1.5,
        ) {
            let mut det = ShadowDetector::new();
            for c in 0..cycles {
                let t = det.process(c, pe, -0.8, 0.7, &format!("shadow item {}", c), 0.2);
                prop_assert!(
                    t.dream_queue_depth <= 8,
                    "dream queue exceeded 8: {}", t.dream_queue_depth,
                );
            }
        }

        /// All telemetry fields are finite for any valid inputs.
        #[test]
        fn telemetry_always_finite(
            pe in 0.0f32..5.0,
            valence in -1.0f32..1.0,
            arousal in 0.0f32..1.0,
            narrative in 0.0f32..1.0,
        ) {
            let mut det = ShadowDetector::new();
            let t = det.process(0, pe, valence, arousal, "test input", narrative);
            prop_assert!(t.total_shadow_pressure.is_finite());
            prop_assert!(t.peak_fragment_pressure.is_finite());
            prop_assert!(t.shadow_mean_prediction_error.is_finite());
            prop_assert!(t.pressure_trend.is_finite());
            prop_assert!(t.best_dream_phi_improvement.is_finite());
            prop_assert!(t.shadow_to_narrative_ratio.is_finite());
        }
    }
}
