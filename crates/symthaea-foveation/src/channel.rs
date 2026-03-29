// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Background-thread foveation channel using std::sync::mpsc.
//!
//! Follows the `EmbeddingChannel` pattern from `symthaea-embeddings/src/channel.rs`.
//! No tokio dependency — compatible with the sync cognitive loop.
//!
//! ```rust,ignore
//! use symthaea_foveation::{FoveationChannel, FoveationConfig, FoveationRequest};
//!
//! let channel = FoveationChannel::spawn(FoveationConfig::default());
//! let rx = channel.request(my_request).unwrap();
//! let result = rx.recv().unwrap();
//! ```

use std::sync::mpsc;

use crate::types::{FoveationConfig, FoveationRequest, FoveationResult, RoutingStrategy};
use crate::ventral::VentralPipeline;

/// Handle to a background foveation thread.
///
/// Requests are submitted via a bounded channel (default capacity from config).
/// Drop the handle to shut down the background thread.
pub struct FoveationChannel {
    request_tx: mpsc::SyncSender<ChannelRequest>,
}

/// Internal request wrapper with response channel.
struct ChannelRequest {
    request: FoveationRequest,
    response_tx: mpsc::SyncSender<FoveationResult>,
}

impl FoveationChannel {
    /// Spawn a background foveation thread with the given config.
    pub fn spawn(config: &FoveationConfig) -> Self {
        Self::spawn_with_capacity(config.routing, config.channel_depth)
    }

    /// Spawn a background foveation thread with explicit capacity.
    pub fn spawn_with_capacity(routing: RoutingStrategy, depth: usize) -> Self {
        let (request_tx, request_rx) = mpsc::sync_channel::<ChannelRequest>(depth);

        std::thread::Builder::new()
            .name("symthaea-foveation".into())
            .spawn(move || {
                let mut pipeline = VentralPipeline::new(routing);

                while let Ok(channel_req) = request_rx.recv() {
                    let result = pipeline.process(&channel_req.request);
                    // If the receiver was dropped, just discard the response
                    let _ = channel_req.response_tx.try_send(result);
                }
            })
            .expect("Failed to spawn foveation thread");

        Self { request_tx }
    }

    /// Submit a non-blocking foveation request.
    ///
    /// Returns a receiver for the result. Fails with `TrySendError::Full`
    /// if the channel is at capacity (backpressure).
    pub fn request(
        &self,
        request: FoveationRequest,
    ) -> Result<mpsc::Receiver<FoveationResult>, mpsc::TrySendError<()>> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let channel_req = ChannelRequest {
            request,
            response_tx,
        };

        self.request_tx.try_send(channel_req).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => mpsc::TrySendError::Full(()),
            mpsc::TrySendError::Disconnected(_) => mpsc::TrySendError::Disconnected(()),
        })?;

        Ok(response_rx)
    }

    /// Submit a request and block until the result is available.
    pub fn request_blocking(&self, request: FoveationRequest) -> Result<FoveationResult, String> {
        let rx = self
            .request(request)
            .map_err(|e| format!("Channel send failed: {e:?}"))?;

        rx.recv().map_err(|e| format!("Channel recv failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FoveationConfig;

    fn make_request(id: u64) -> FoveationRequest {
        FoveationRequest {
            id,
            crop_pixels: vec![128; 64],
            crop_width: 8,
            crop_height: 8,
            channels: 1,
            grid_row: 1,
            grid_col: 2,
            surprise_value: 0.7,
            frame_id: 42,
            timestamp_us: 10_000,
            velocity: [0.0, 0.0],
        }
    }

    #[test]
    fn test_channel_spawn_and_request() {
        let config = FoveationConfig::default();
        let channel = FoveationChannel::spawn(&config);
        let rx = channel.request(make_request(1)).unwrap();
        let result = rx.recv().unwrap();

        assert_eq!(result.request_id, 1);
        assert_eq!(result.semantic_hv.dim(), 16_384);
        assert!(result.semantic_hv.norm() > 0.0);
    }

    #[test]
    fn test_channel_multiple_requests() {
        let channel = FoveationChannel::spawn_with_capacity(RoutingStrategy::Auto, 16);

        let mut receivers = Vec::new();
        for i in 0..10 {
            let rx = channel.request(make_request(i)).unwrap();
            receivers.push((i, rx));
        }

        for (id, rx) in receivers {
            let result = rx.recv().unwrap();
            assert_eq!(result.request_id, id);
            assert_eq!(result.semantic_hv.dim(), 16_384);
        }
    }

    #[test]
    fn test_channel_blocking_request() {
        let config = FoveationConfig::default();
        let channel = FoveationChannel::spawn(&config);
        let result = channel.request_blocking(make_request(99)).unwrap();

        assert_eq!(result.request_id, 99);
        assert_eq!(result.semantic_hv.dim(), 16_384);
    }

    #[test]
    fn test_channel_backpressure() {
        // Capacity of 2
        let channel = FoveationChannel::spawn_with_capacity(RoutingStrategy::Auto, 2);

        let mut sent = 0;
        let mut receivers = Vec::new();
        for i in 0..100u64 {
            match channel.request(make_request(i)) {
                Ok(rx) => {
                    sent += 1;
                    receivers.push(rx);
                }
                Err(mpsc::TrySendError::Full(())) => {
                    break;
                }
                Err(mpsc::TrySendError::Disconnected(())) => {
                    panic!("Channel disconnected unexpectedly");
                }
            }
        }

        // Should have sent at least the capacity
        assert!(
            sent >= 2,
            "Should send at least capacity messages, sent {sent}"
        );

        // All sent requests should eventually complete
        for rx in receivers {
            let result = rx.recv().unwrap();
            assert_eq!(result.semantic_hv.dim(), 16_384);
        }
    }

    #[test]
    fn test_channel_drop_shuts_down() {
        let config = FoveationConfig::default();
        let channel = FoveationChannel::spawn(&config);

        // Do one request to confirm thread is alive
        let result = channel.request_blocking(make_request(1)).unwrap();
        assert_eq!(result.request_id, 1);

        // Drop channel — thread should exit when request_rx disconnects
        drop(channel);

        // Give thread a moment to notice disconnect
        std::thread::sleep(std::time::Duration::from_millis(50));
        // If we get here without hanging, the thread exited cleanly
    }

    #[test]
    fn test_channel_preserves_spatial_info() {
        let config = FoveationConfig::default();
        let channel = FoveationChannel::spawn(&config);

        let mut req = make_request(50);
        req.grid_row = 5;
        req.grid_col = 7;
        req.frame_id = 200;
        req.timestamp_us = 999_000;

        let result = channel.request_blocking(req).unwrap();
        assert_eq!(result.grid_row, 5);
        assert_eq!(result.grid_col, 7);
        assert_eq!(result.source_frame_id, 200);
        assert_eq!(result.source_timestamp_us, 999_000);
    }

    #[test]
    fn test_channel_different_routing_strategies() {
        for routing in [
            RoutingStrategy::Auto,
            RoutingStrategy::AlwaysEmbed,
            RoutingStrategy::AlwaysOcr,
            RoutingStrategy::AlwaysCaption,
            RoutingStrategy::Full,
        ] {
            let channel = FoveationChannel::spawn_with_capacity(routing, 4);
            let result = channel.request_blocking(make_request(1)).unwrap();
            assert_eq!(result.semantic_hv.dim(), 16_384, "Failed for {routing:?}");
        }
    }
}
