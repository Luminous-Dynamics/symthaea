// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FoveationManager: coordinates dorsal→ventral foveation dispatch.
//!
//! A pending saliency item is meaningful only for the exact framebuffer that produced it.
//! Replacing the current frame therefore clears pending (not yet dispatched) saliency so old
//! frame identity can never be attached to crops taken from newer pixels.

use std::collections::{BinaryHeap, VecDeque};
use std::sync::mpsc;

use symthaea_vision_manifold::VisualObservationRef;

use crate::channel::FoveationChannel;
use crate::crop::extract_crop;
use crate::types::{
    FoveationConfig, FoveationRequest, FoveationResult, FoveationTelemetry, FrameBuffer,
    FrameObservationError,
};

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

pub struct FoveationManager {
    config: FoveationConfig,
    channel: FoveationChannel,
    pending: BinaryHeap<PrioritizedRequest>,
    in_flight: Vec<mpsc::Receiver<FoveationResult>>,
    results: VecDeque<FoveationResult>,
    last_dispatch_us: u64,
    next_id: u64,
    frame_buffer: Option<FrameBuffer>,
    frame_observation: Option<VisualObservationRef>,
    patch_size: usize,
    effective_surprise_threshold: f32,
    effective_max_concurrent: usize,
    total_dispatched: u64,
    total_completed: u64,
    avg_processing_time_us: f32,
    last_confidence: f32,
}

impl FoveationManager {
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

    /// Store an unproven frame through the compatibility path.
    ///
    /// Pending saliency and prior provenance are cleared because both belong to the previous
    /// framebuffer. Already-dispatched requests retain their frozen source identity.
    pub fn on_frame(&mut self, frame: FrameBuffer) {
        self.pending.clear();
        self.frame_buffer = Some(frame);
        self.frame_observation = None;
    }

    /// Store a frame together with capture-owner provenance.
    ///
    /// Legacy metadata must agree exactly with the typed observation. Failure occurs before
    /// frame, pending-queue, or provenance state is mutated.
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

    pub fn on_saliency(&mut self, patches: &[(usize, usize, f32, [f32; 2])]) {
        let (frame_id, timestamp_us) = match &self.frame_buffer {
            Some(fb) => (fb.frame_id, fb.timestamp_us),
            None => return,
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

    pub fn tick(&mut self, now_us: u64) {
        self.collect_results();

        if self.in_flight.len() >= self.effective_max_concurrent {
            return;
        }

        let cooldown_us = self.config.cooldown_ms * 1000;
        if now_us.saturating_sub(self.last_dispatch_us) < cooldown_us {
            return;
        }

        if let Some(prioritized) = self.pending.pop()
            && let Some(ref frame) = self.frame_buffer
        {
            // Defense in depth: frame replacement normally clears pending, but never crop if
            // internal state is inconsistent for any reason.
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
                1,
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
                    Err(_) => self.pending.push(prioritized),
                }
            }
        }
    }

    fn collect_results(&mut self) {
        let mut still_in_flight = Vec::new();
        for rx in self.in_flight.drain(..) {
            match rx.try_recv() {
                Ok(result) => {
                    self.total_completed += 1;
                    let alpha = 0.2;
                    self.avg_processing_time_us = self.avg_processing_time_us * (1.0 - alpha)
                        + result.processing_time_us as f32 * alpha;
                    self.last_confidence = result.confidence;
                    self.results.push_back(result);
                }
                Err(mpsc::TryRecvError::Empty) => still_in_flight.push(rx),
                Err(mpsc::TryRecvError::Disconnected) => {}
            }
        }
        self.in_flight = still_in_flight;
    }

    pub fn drain_results(&mut self) -> Vec<FoveationResult> {
        self.results.drain(..).collect()
    }

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

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }
    pub fn ready_count(&self) -> usize {
        self.results.len()
    }
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }
    pub fn config(&self) -> &FoveationConfig {
        &self.config
    }

    pub fn modulate(&mut self, ne: f32, da: f32) {
        let ne = ne.clamp(0.0, 2.0);
        let da = da.clamp(0.0, 2.0);
        self.effective_surprise_threshold = self.config.min_surprise_threshold * ne.max(0.1);
        self.effective_max_concurrent =
            (self.config.max_concurrent as f32 * da).round().max(1.0) as usize;
    }

    pub fn effective_surprise_threshold(&self) -> f32 {
        self.effective_surprise_threshold
    }
    pub fn effective_max_concurrent(&self) -> usize {
        self.effective_max_concurrent
    }

    pub fn predict_current_position(result: &FoveationResult) -> (usize, usize) {
        let dt_frames = result.processing_time_us as f32 / 20_000.0;
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
    fn manager_construction() {
        let mgr = default_manager();
        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.in_flight_count(), 0);
        assert_eq!(mgr.ready_count(), 0);
    }

    #[test]
    fn legacy_frame_clears_provenance() {
        let mut mgr = default_manager();
        let frame = make_frame(64, 64, 128);
        let observation = make_observation(frame.frame_id, frame.timestamp_us);
        mgr.on_observed_frame(frame, observation).unwrap();
        assert_eq!(mgr.frame_observation, Some(observation));
        mgr.on_frame(make_frame(64, 64, 129));
        assert_eq!(mgr.frame_observation, None);
    }

    #[test]
    fn observed_frame_mismatch_is_non_mutating() {
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
    fn observed_timestamp_mismatch_is_non_mutating() {
        let mut mgr = default_manager();
        let frame = make_frame(64, 64, 128);
        let observation = make_observation(frame.frame_id, frame.timestamp_us + 1);
        assert!(matches!(
            mgr.on_observed_frame(frame, observation),
            Err(FrameObservationError::TimestampMismatch { .. })
        ));
        assert!(mgr.frame_buffer.is_none());
    }

    #[test]
    fn frame_replacement_discards_undispatched_old_saliency() {
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
        mgr.tick(2_000_000);
        assert_eq!(mgr.total_dispatched, 0);
    }

    #[test]
    fn provenance_is_frozen_into_pending_request() {
        let mut mgr = default_manager();
        let frame = make_frame(64, 64, 128);
        let observation = make_observation(frame.frame_id, frame.timestamp_us);
        mgr.on_observed_frame(frame, observation).unwrap();
        mgr.on_saliency(&[(1, 1, 0.8, [0.0, 0.0])]);
        assert_eq!(mgr.pending.pop().unwrap().source_observation, Some(observation));
    }

    #[test]
    fn saliency_filters_by_threshold() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[
            (0, 0, 0.3, [0.0, 0.0]),
            (1, 1, 0.7, [0.0, 0.0]),
            (2, 2, 0.9, [0.0, 0.0]),
            (3, 3, 0.1, [0.0, 0.0]),
        ]);
        assert_eq!(mgr.pending_count(), 2);
    }

    #[test]
    fn saliency_without_frame_is_noop() {
        let mut mgr = default_manager();
        mgr.on_saliency(&[(0, 0, 0.9, [0.0, 0.0])]);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn priority_ordering_is_highest_surprise_first() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[
            (0, 0, 0.6, [0.0, 0.0]),
            (1, 1, 0.9, [1.5, -0.3]),
            (2, 2, 0.7, [0.0, 0.0]),
        ]);
        let top = mgr.pending.pop().unwrap();
        assert_eq!((top.grid_row, top.grid_col), (1, 1));
    }

    #[test]
    fn legacy_pipeline_completes_without_fabricating_observation() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(2, 3, 0.8, [0.5, -0.2])]);
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
        assert_eq!(results[0].source_observation, None);
    }

    #[test]
    fn observed_provenance_reaches_completed_result() {
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
    fn max_concurrent_is_respected() {
        let config = FoveationConfig {
            max_concurrent: 1,
            cooldown_ms: 0,
            ..FoveationConfig::default()
        };
        let mut mgr = FoveationManager::new(config, 8);
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(0, 0, 0.8, [0.0, 0.0]), (1, 1, 0.9, [0.0, 0.0])]);
        mgr.tick(1_000_000);
        mgr.tick(1_000_001);
        assert_eq!(mgr.total_dispatched, 1);
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn cooldown_is_respected() {
        let config = FoveationConfig {
            max_concurrent: 10,
            cooldown_ms: 100,
            ..FoveationConfig::default()
        };
        let mut mgr = FoveationManager::new(config, 8);
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(0, 0, 0.8, [0.0, 0.0]), (1, 1, 0.9, [0.0, 0.0])]);
        mgr.tick(1_000_000);
        mgr.tick(1_050_000);
        assert_eq!(mgr.total_dispatched, 1);
        mgr.tick(1_200_000);
        assert_eq!(mgr.total_dispatched, 2);
    }

    #[test]
    fn clear_pending_works() {
        let mut mgr = default_manager();
        mgr.on_frame(make_frame(64, 64, 128));
        mgr.on_saliency(&[(0, 0, 0.6, [0.0, 0.0]), (1, 1, 0.8, [0.0, 0.0])]);
        assert_eq!(mgr.pending_count(), 2);
        mgr.clear_pending();
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn neuromodulation_changes_threshold_and_budget() {
        let mut mgr = default_manager();
        mgr.modulate(1.5, 2.0);
        assert!((mgr.effective_surprise_threshold() - 0.75).abs() < 1e-4);
        assert_eq!(mgr.effective_max_concurrent(), 4);

        mgr.modulate(5.0, -1.0);
        assert!((mgr.effective_surprise_threshold() - 1.0).abs() < 1e-4);
        assert_eq!(mgr.effective_max_concurrent(), 1);
    }
}
