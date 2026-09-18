// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FoveationManager: coordinates dorsal→ventral foveation dispatch.
//!
//! Manages a priority queue of salient regions, dispatches the most
//! surprising ones to the background ventral pipeline, and collects
//! completed results for GWT injection.
//!
//! VIS-001R adds a strict frame-binding invariant: pending saliency is valid only for the exact
//! framebuffer that produced it. Replacing the frame clears undispatched saliency, and dispatch
//! independently checks frame/time/provenance equality before cropping.

use std::collections::{BinaryHeap, VecDeque};
use std::sync::mpsc;

use symthaea_vision_manifold::VisualObservationRef;

use crate::channel::FoveationChannel;
use crate::crop::extract_crop;
use crate::types::{
    FoveationConfig, FoveationRequest, FoveationResult, FoveationTelemetry, FrameBuffer,
    FrameObservationError,
};

/// Prioritized request wrapper for the BinaryHeap (highest surprise first).
#[derive(Debug)]
struct PrioritizedRequest {
    grid_row: usize,
    grid_col: usize,
    surprise: f32,
    frame_id: u64,
    timestamp_us: u64,
    source_observation: Option<VisualObservationRef>,
    velocity: [f32; 2],
}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.surprise == other.surprise
    }
}

impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.surprise
            .partial_cmp(&other.surprise)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Manages the foveation bridge between dorsal surprise and ventral recognition.
///
/// # Lifecycle
///
/// 1. `on_frame()` / `on_observed_frame()` — store the latest full-res frame for cropping
/// 2. `on_saliency()` — enqueue salient patches from the SurpriseMap
/// 3. `tick()` — dispatch highest-priority request + collect completed results
/// 4. `drain_results()` — pop all ready results for GWT injection
pub struct FoveationManager {
    config: FoveationConfig,
    channel: FoveationChannel,
    pending: BinaryHeap<PrioritizedRequest>,
    in_flight: Vec<mpsc::Receiver<FoveationResult>>,
    results: VecDeque<FoveationResult>,
    last_dispatch_us: u64,
    next_id: u64,
    frame_buffer: Option<FrameBuffer>,
    /// Exact provenance for `frame_buffer`, when supplied by the capture owner.
    frame_observation: Option<VisualObservationRef>,
    patch_size: usize,
    // Neuromodulated effective parameters (updated via modulate())
    effective_surprise_threshold: f32,
    effective_max_concurrent: usize,
    // Telemetry counters
    total_dispatched: u64,
    total_completed: u64,
    avg_processing_time_us: f32,
    last_confidence: f32,
}

impl FoveationManager {
    /// Create a new FoveationManager with the given config and patch size.
    pub fn new(config: FoveationConfig, patch_size: usize) -> Self {
        let channel = FoveationChannel::spawn(&config);
        let effective_surprise_threshold = config.min_surprise_threshold;
        let effective_max_concurrent = config.max_concurrent;
        Self {
            config,
            channel,
            pending: BinaryHeap::new(),
            in_flight: Vec::new(),
            results: VecDeque::new(),
            last_dispatch_us: 0,
            next_id: 0,
            frame_buffer: None,
            frame_observation: None,
            patch_size,
            effective_surprise_threshold,
            effective_max_concurrent,
            total_dispatched: 0,
            total_completed: 0,
            avg_processing_time_us: 0.0,
            last_confidence: 0.0,
        }
    }

    /// Store the latest full-resolution frame for future cropping through the compatibility path.
    ///
    /// Typed provenance is explicitly cleared. Pending saliency is also cleared because it was
    /// computed over the prior framebuffer and must never be cropped from replacement pixels.
    pub fn on_frame(&mut self, frame: FrameBuffer) {
        self.pending.clear();
        self.frame_buffer = Some(frame);
        self.frame_observation = None;
    }

    /// Store a frame together with capture-owner provenance.
    ///
    /// The legacy frame fields must exactly agree with the typed observation. A mismatch fails
    /// before framebuffer, provenance, or pending-queue state is modified.
    pub fn on_observed_frame(
        &mut self,
        frame: FrameBuffer,
        observation: VisualObservationRef,
    ) -> Result<(), FrameObservationError> {
        if frame.frame_id != observation.frame_id() {
            return Err(FrameObservationError::FrameIdMismatch {
                frame_id: frame.frame_id,
                observation_frame_id: observation.frame_id(),
            });
        }
        if frame.timestamp_us != observation.captured_at_us() {
            return Err(FrameObservationError::TimestampMismatch {
                timestamp_us: frame.timestamp_us,
                observation_timestamp_us: observation.captured_at_us(),
            });
        }

        self.pending.clear();
        self.frame_buffer = Some(frame);
        self.frame_observation = Some(observation);
        Ok(())
    }

    /// Enqueue salient patches from the dorsal stream.
    ///
    /// `patches` is a slice of `(row, col, surprise_value, velocity)` tuples,
    /// typically from `SurpriseMap::salient_patches()` combined with motion vectors.
    /// The velocity `[dx, dy]` in pixels/frame is used for predictive binding:
    /// when the ventral result arrives later, the cognitive loop compensates for object motion.
    pub fn on_saliency(&mut self, patches: &[(usize, usize, f32, [f32; 2])]) {
        let (frame_id, timestamp_us) = match &self.frame_buffer {
            Some(fb) => (fb.frame_id, fb.timestamp_us),
            None => return, // No frame stored yet
        };
        let source_observation = self.frame_observation;

        for &(row, col, surprise, velocity) in patches {
            if surprise >= self.effective_surprise_threshold {
                self.pending.push(PrioritizedRequest {
                    grid_row: row,
                    grid_col: col,
                    surprise,
                    frame_id,
                    timestamp_us,
                    source_observation,
                    velocity,
                });
            }
        }
    }

    /// Main tick: dispatch one request (if budget allows) and collect results.
    ///
    /// Call this once per cognitive cycle. It:
    /// 1. Collects any completed in-flight results
    /// 2. Dispatches the highest-priority pending request (if budget allows)
    pub fn tick(&mut self, now_us: u64) {
        // Collect completed results
        self.collect_results();

        // Check dispatch budget (uses neuromodulated effective value)
        if self.in_flight.len() >= self.effective_max_concurrent {
            return;
        }

        let cooldown_us = self.config.cooldown_ms * 1000;
        if now_us.saturating_sub(self.last_dispatch_us) < cooldown_us {
            return;
        }

        // Dispatch highest-priority pending request
        if let Some(prioritized) = self.pending.pop()
            && let Some(ref frame) = self.frame_buffer
        {
            // Defense in depth. Frame replacement normally clears pending, but an inconsistent
            // internal state must still fail closed rather than crop new pixels under old identity.
            if frame.frame_id != prioritized.frame_id
                || frame.timestamp_us != prioritized.timestamp_us
                || self.frame_observation != prioritized.source_observation
            {
                return;
            }

            let (crop_pixels, crop_width, crop_height) = extract_crop(
                frame,
                prioritized.grid_row,
                prioritized.grid_col,
                self.patch_size,
                1, // 1 patch padding for context
                self.config.max_crop_pixels,
            );

            if crop_width > 0 && crop_height > 0 {
                let id = self.next_id;
                self.next_id += 1;

                let request = FoveationRequest {
                    id,
                    crop_pixels,
                    crop_width,
                    crop_height,
                    channels: frame.channels,
                    grid_row: prioritized.grid_row,
                    grid_col: prioritized.grid_col,
                    surprise_value: prioritized.surprise,
                    frame_id: prioritized.frame_id,
                    timestamp_us: prioritized.timestamp_us,
                    source_observation: prioritized.source_observation,
                    velocity: prioritized.velocity,
                };

                match self.channel.request(request) {
                    Ok(rx) => {
                        self.in_flight.push(rx);
                        self.last_dispatch_us = now_us;
                        self.total_dispatched += 1;
                    }
                    Err(_) => {
                        // Channel full — re-enqueue the exact same frame-bound request.
                        self.pending.push(prioritized);
                    }
                }
            }
        }
    }

    /// Collect completed results from in-flight receivers.
    fn collect_results(&mut self) {
        let mut still_in_flight = Vec::new();

        for rx in self.in_flight.drain(..) {
            match rx.try_recv() {
                Ok(result) => {
                    self.total_completed += 1;
                    // Exponential moving average of processing time
                    let alpha = 0.2;
                    self.avg_processing_time_us = self.avg_processing_time_us * (1.0 - alpha)
                        + result.processing_time_us as f32 * alpha;
                    self.last_confidence = result.confidence;
                    self.results.push_back(result);
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Still processing
                    still_in_flight.push(rx);
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Thread died — discard
                }
            }
        }

        self.in_flight = still_in_flight;
    }

    /// Drain all ready results for GWT injection.
    pub fn drain_results(&mut self) -> Vec<FoveationResult> {
        self.results.drain(..).collect()
    }

    /// Get current telemetry for CycleMetadata.
    pub fn telemetry(&self) -> FoveationTelemetry {
        FoveationTelemetry {
            pending_count: self.pending.len(),
            in_flight_count: self.in_flight.len(),
            ready_count: self.results.len(),
            total_dispatched: self.total_dispatched,
            total_completed: self.total_completed,
            avg_processing_time_us: self.avg_processing_time_us,
            last_confidence: self.last_confidence,
        }
    }

    /// Number of pending requests in the queue.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of in-flight requests.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Number of ready results.
    pub fn ready_count(&self) -> usize {
        self.results.len()
    }

    /// Clear the pending queue.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Access the current config.
    pub fn config(&self) -> &FoveationConfig {
        &self.config
    }

    /// Apply neuromodulated budgeting from the cognitive loop.
    ///
    /// Called once per tick with the current neuromodulator levels (0.0–2.0).
    ///
    /// * **Noradrenaline (NE)** modulates the surprise threshold (attention aperture):
    ///   - Low NE (< 1.0): broadens attention → lowers threshold (more foveations)
    ///   - High NE (> 1.0): narrows attention → raises threshold (fewer, more urgent)
    ///
    /// * **Dopamine (DA)** modulates compute budget (concurrent capacity):
    ///   - Low DA (< 1.0): reduces budget → fewer concurrent foveations
    ///   - High DA (> 1.0): increases budget → more concurrent foveations
    ///
    /// Both scale around 1.0 as baseline (no modulation).
    pub fn modulate(&mut self, ne: f32, da: f32) {
        let ne = ne.clamp(0.0, 2.0);
        let da = da.clamp(0.0, 2.0);

        // NE scales surprise threshold: base * ne (higher NE → higher threshold)
        self.effective_surprise_threshold = self.config.min_surprise_threshold * ne.max(0.1);

        // DA scales max concurrent: base * da, rounded, min 1
        let base = self.config.max_concurrent as f32;
        self.effective_max_concurrent = (base * da).round().max(1.0) as usize;
    }

    /// Current effective surprise threshold (after neuromodulation).
    pub fn effective_surprise_threshold(&self) -> f32 {
        self.effective_surprise_threshold
    }

    /// Current effective max concurrent (after neuromodulation).
    pub fn effective_max_concurrent(&self) -> usize {
        self.effective_max_concurrent
    }

    /// Predict current grid position given velocity and processing latency.
    ///
    /// Because ventral processing takes time, the recognized object may have moved since the
    /// crop was captured. This compensates by projecting the original grid position forward.
    pub fn predict_current_position(result: &crate::types::FoveationResult) -> (usize, usize) {
        let dt_frames = result.processing_time_us as f32 / 20_000.0; // ~50fps
        let pred_row = result.grid_row as f32 + result.velocity[1] * dt_frames;
        let pred_col = result.grid_col as f32 + result.velocity[0] * dt_frames;
        (
            pred_row.round().max(0.0) as usize,
            pred_col.round().max(0.0) as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_vision_manifold::{VisualCaptureClock, VisualStreamRef};

    fn make_frame(w: u32, h: u32, val: u8) -> FrameBuffer {
        FrameBuffer {
            pixels: vec![val; (w * h) as usize],
            width: w,
            height: h,
            channels: 1,
            frame_id: 1,
            timestamp_us: 1_000_000,
        }
    }

    fn make_observation(frame_id: u64, timestamp_us: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(12, 34).unwrap(),
            frame_id,
            timestamp_us,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    fn default_manager() -> FoveationManager {
        FoveationManager::new(FoveationConfig::default(), 8)
    }

    #[test]
    fn test_manager_construction() {
        let mgr = default_manager();
        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.in_flight_count(), 0);
        assert_eq!(mgr.ready_count(), 0);
    }

    #[test]
    fn test_on_frame_stores_buffer() {
        let mut mgr = default_manager();
        assert!(mgr.frame_buffer.is_none());

        mgr.on_frame(make_frame(64, 64, 128));
        assert!(mgr.frame_buffer.is_some());
        assert!(mgr.frame_observation.is_none());
    }

    #[test]
    fn test_on_observed_frame_binds_provenance() {
        let mut mgr = default_manager();
        let frame = make_frame(64, 64, 128);
        let observation = make_observation(frame.frame_id, frame.timestamp_us);
        mgr.on_observed_frame(frame, observation).unwrap();
        assert_eq!(mgr.frame_observation, Some(observation));
    }

    #[test]
    fn test_on_observed_frame_rejects_frame_mismatch_without_mutation() {
        let mut mgr = default_manager();
        let frame = make_frame(64, 64, 128);
        let observation = make_observation(frame.frame_id + 1, frame.timestamp_us);
        assert!(matches!(
            mgr.on_observed_frame(frame, observation),
            Err(FrameObservationError::FrameIdMismatch { .. })
        ));
        assert!(mgr.frame_buffer.is_none());
        assert!(mgr.frame_observation.is_none());
    }

    #[test]
    fn test_on_observed_frame_rejects_timestamp_mismatch_without_mutation() {
        let mut mgr = default_manager();
        let frame = make_frame(64, 64, 128);
        let observation = make_observation(frame.frame_id, frame.timestamp_us + 1);
        assert!(matches!(
            mgr.on_observed_frame(frame, observation),
            Err(FrameObservationError::TimestampMismatch { .. })
        ));
        assert!(mgr.frame_buffer.is_none());
        assert!(mgr.frame_observation.is_none());
    }

    #[test]
    fn test_new_frame_clears_pending_old_saliency() {
        let mut mgr = default_manager();
        let frame = make_frame(64, 64, 128);
        let observation = make_observation(frame.frame_id, frame.timestamp_us);
        mgr.on_observed_frame(frame, observation).unwrap();
        mgr.on_saliency(&[(2, 2, 0.9, [0.0, 0.0])]);
        assert_eq!(mgr.pending_count(), 1);

        let mut next = make_frame(64, 64, 129);
        next.frame_id = 2;
        next.timestamp_us = 1_020_000;
        mgr.on_frame(next);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_on_saliency_filters_by_threshold() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));

        // Default threshold is 0.5
        let patches = vec![
            (0, 0, 0.3, [0.0, 0.0]), // Below threshold
            (1, 1, 0.7, [0.0, 0.0]), // Above threshold
            (2, 2, 0.9, [0.0, 0.0]), // Above threshold
            (3, 3, 0.1, [0.0, 0.0]), // Below threshold
        ];
        mgr.on_saliency(&patches);

        assert_eq!(
            mgr.pending_count(),
            2,
            "Only patches above 0.5 should be enqueued"
        );
    }

    #[test]
    fn test_on_saliency_without_frame_is_noop() {
        let mut mgr = default_manager();
        mgr.on_saliency(&[(0, 0, 0.9, [0.0, 0.0])]);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_priority_ordering() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));

        let patches = vec![
            (0, 0, 0.6, [0.0, 0.0]),
            (1, 1, 0.9, [1.5, -0.3]), // highest — also has velocity
            (2, 2, 0.7, [0.0, 0.0]),
        ];
        mgr.on_saliency(&patches);

        let top = mgr.pending.pop().unwrap();
        assert_eq!(top.grid_row, 1);
        assert_eq!(top.grid_col, 1);
        assert!((top.surprise - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_tick_dispatches_and_collects() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(2, 3, 0.8, [0.5, -0.2])]);

        mgr.tick(1_000_000);
        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.total_dispatched, 1);

        std::thread::sleep(std::time::Duration::from_millis(100));
        mgr.tick(1_100_000);
        assert!(mgr.ready_count() > 0 || mgr.in_flight_count() > 0);

        for i in 0..10 {
            if mgr.ready_count() > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            mgr.tick(1_200_000 + i * 100_000);
        }

        let results = mgr.drain_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].grid_row, 2);
        assert_eq!(results[0].grid_col, 3);
        assert_eq!(results[0].semantic_hv.dim(), 16_384);
        assert_eq!(results[0].source_observation, None);
        assert!((results[0].velocity[0] - 0.5).abs() < 1e-6);
        assert!((results[0].velocity[1] - (-0.2)).abs() < 1e-6);
    }

    #[test]
    fn test_observed_provenance_reaches_completed_result() {
        let config = FoveationConfig {
            cooldown_ms: 0,
            ..FoveationConfig::default()
        };
        let mut mgr = FoveationManager::new(config, 8);
        let frame = make_frame(64, 64, 128);
        let observation = make_observation(frame.frame_id, frame.timestamp_us);
        mgr.on_observed_frame(frame, observation).unwrap();
        mgr.on_saliency(&[(2, 3, 0.8, [0.0, 0.0])]);
        mgr.tick(1_000_000);

        for i in 0..10 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            mgr.tick(1_100_000 + i * 50_000);
            if mgr.ready_count() > 0 {
                break;
            }
        }

        let results = mgr.drain_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_observation, Some(observation));
    }

    #[test]
    fn test_tick_respects_max_concurrent() {
        let config = FoveationConfig {
            max_concurrent: 1,
            cooldown_ms: 0,
            ..FoveationConfig::default()
        };
        let mut mgr = FoveationManager::new(config, 8);
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(0, 0, 0.8, [0.0, 0.0]), (1, 1, 0.9, [0.0, 0.0])]);

        mgr.tick(1_000_000);
        assert_eq!(mgr.total_dispatched, 1);

        mgr.tick(1_000_001);
        assert!(mgr.total_dispatched >= 1);
        assert!(mgr.total_dispatched <= 2);
        assert!(mgr.effective_max_concurrent() == 1);
    }

    #[test]
    fn test_tick_respects_cooldown() {
        let config = FoveationConfig {
            max_concurrent: 10,
            cooldown_ms: 100,
            ..FoveationConfig::default()
        };
        let mut mgr = FoveationManager::new(config, 8);
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(0, 0, 0.8, [0.0, 0.0]), (1, 1, 0.9, [0.0, 0.0])]);

        mgr.tick(1_000_000);
        assert_eq!(mgr.total_dispatched, 1);

        mgr.tick(1_050_000);
        assert_eq!(
            mgr.total_dispatched, 1,
            "Should not dispatch within cooldown"
        );

        mgr.tick(1_200_000);
        assert_eq!(mgr.total_dispatched, 2, "Should dispatch after cooldown");
    }

    #[test]
    fn test_drain_results_empties() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(2, 2, 0.9, [0.0, 0.0])]);

        mgr.tick(1_000_000);
        std::thread::sleep(std::time::Duration::from_millis(200));
        mgr.tick(2_000_000);

        let _ = mgr.drain_results();
        let results2 = mgr.drain_results();
        assert!(results2.is_empty());
        assert_eq!(mgr.ready_count(), 0);
    }

    #[test]
    fn test_telemetry() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(1, 1, 0.7, [0.0, 0.0]), (2, 2, 0.8, [0.0, 0.0])]);

        let tel = mgr.telemetry();
        assert_eq!(tel.pending_count, 2);
        assert_eq!(tel.in_flight_count, 0);
        assert_eq!(tel.ready_count, 0);
        assert_eq!(tel.total_dispatched, 0);
    }

    #[test]
    fn test_clear_pending() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[
            (0, 0, 0.6, [0.0, 0.0]),
            (1, 1, 0.8, [0.0, 0.0]),
            (2, 2, 0.9, [0.0, 0.0]),
        ]);

        assert_eq!(mgr.pending_count(), 3);
        mgr.clear_pending();
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_multiple_frames_updates_buffer_and_clears_old_pending() {
        let mut mgr = default_manager();

        mgr.on_frame(make_frame(64, 64, 100));
        assert_eq!(mgr.frame_buffer.as_ref().unwrap().frame_id, 1);
        mgr.on_saliency(&[(0, 0, 0.9, [0.0, 0.0])]);
        assert_eq!(mgr.pending_count(), 1);

        let mut frame2 = make_frame(64, 64, 200);
        frame2.frame_id = 2;
        frame2.timestamp_us = 1_020_000;
        mgr.on_frame(frame2);
        assert_eq!(mgr.frame_buffer.as_ref().unwrap().frame_id, 2);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_full_pipeline_multiple_saliency_cycles() {
        let config = FoveationConfig {
            max_concurrent: 2,
            cooldown_ms: 0,
            min_surprise_threshold: 0.3,
            ..FoveationConfig::default()
        };
        let mut mgr = FoveationManager::new(config, 8);

        // Each new frame intentionally supersedes undispatched saliency from the prior frame.
        for cycle in 0..3 {
            let mut frame = make_frame(64, 64, 128);
            frame.frame_id = cycle as u64 + 1;
            frame.timestamp_us = 1_000_000 + cycle as u64 * 20_000;
            mgr.on_frame(frame);
            mgr.on_saliency(&[
                (cycle, 0, 0.5 + cycle as f32 * 0.1, [0.0, 0.0]),
                (cycle, 1, 0.4, [0.0, 0.0]),
            ]);
            mgr.tick(1_000_000 + cycle as u64 * 200_000);
        }

        std::thread::sleep(std::time::Duration::from_millis(300));
        for i in 0..5 {
            mgr.tick(2_000_000 + i * 200_000);
        }

        let results = mgr.drain_results();
        assert!(!results.is_empty(), "Should have completed results");
        for r in &results {
            assert_eq!(r.semantic_hv.dim(), 16_384);
            assert!(r.confidence >= 0.0);
        }
    }

    #[test]
    fn test_modulate_default_is_baseline() {
        let mgr = default_manager();
        assert!((mgr.effective_surprise_threshold() - 0.5).abs() < 1e-6);
        assert_eq!(mgr.effective_max_concurrent(), 2);
    }

    #[test]
    fn test_modulate_high_ne_raises_threshold() {
        let mut mgr = default_manager();
        mgr.modulate(1.5, 1.0);
        assert!((mgr.effective_surprise_threshold() - 0.75).abs() < 1e-4);
        assert_eq!(mgr.effective_max_concurrent(), 2);
    }

    #[test]
    fn test_modulate_low_ne_lowers_threshold() {
        let mut mgr = default_manager();
        mgr.modulate(0.5, 1.0);
        assert!((mgr.effective_surprise_threshold() - 0.25).abs() < 1e-4);
    }

    #[test]
    fn test_modulate_high_da_increases_budget() {
        let mut mgr = default_manager();
        mgr.modulate(1.0, 2.0);
        assert_eq!(mgr.effective_max_concurrent(), 4);
    }

    #[test]
    fn test_modulate_low_da_decreases_budget() {
        let mut mgr = default_manager();
        mgr.modulate(1.0, 0.3);
        assert_eq!(mgr.effective_max_concurrent(), 1);
    }

    #[test]
    fn test_modulate_clamps_input() {
        let mut mgr = default_manager();
        mgr.modulate(5.0, -1.0);
        assert!((mgr.effective_surprise_threshold() - 1.0).abs() < 1e-4);
        assert_eq!(mgr.effective_max_concurrent(), 1);
    }

    #[test]
    fn test_modulate_affects_saliency_filtering() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));

        mgr.modulate(2.0, 1.0);
        mgr.on_saliency(&[(0, 0, 0.9, [0.0, 0.0])]);
        assert_eq!(mgr.pending_count(), 0);

        mgr.modulate(0.2, 1.0);
        mgr.on_saliency(&[(0, 0, 0.3, [0.0, 0.0])]);
        assert_eq!(mgr.pending_count(), 1);
    }
}
