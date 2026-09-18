// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Background-thread foveation channel using std::sync::mpsc.

use std::sync::mpsc;

use crate::types::{FoveationConfig, FoveationRequest, FoveationResult, RoutingStrategy};
use crate::ventral::VentralPipeline;

pub struct FoveationChannel {
    request_tx: mpsc::SyncSender<ChannelRequest>,
}

struct ChannelRequest {
    request: FoveationRequest,
    response_tx: mpsc::SyncSender<FoveationResult>,
}

impl FoveationChannel {
    pub fn spawn(config: &FoveationConfig) -> Self {
        Self::spawn_with_capacity(config.routing, config.channel_depth)
    }

    pub fn spawn_with_capacity(routing: RoutingStrategy, depth: usize) -> Self {
        let (request_tx, request_rx) = mpsc::sync_channel::<ChannelRequest>(depth);

        std::thread::Builder::new()
            .name("symthaea-foveation".into())
            .spawn(move || {
                let mut pipeline = VentralPipeline::new(routing);
                while let Ok(channel_req) = request_rx.recv() {
                    let result = pipeline.process(&channel_req.request);
                    let _ = channel_req.response_tx.try_send(result);
                }
            })
            .expect("Failed to spawn foveation thread");

        Self { request_tx }
    }

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
    use crate::types::{FoveationConfig, VentralExecutionKind};
    use symthaea_vision_manifold::{VisualCaptureClock, VisualObservationRef, VisualStreamRef};

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
            source_observation: None,
            velocity: [0.0, 0.0],
        }
    }

    #[test]
    fn spawn_and_request() {
        let channel = FoveationChannel::spawn(&FoveationConfig::default());
        let result = channel.request(make_request(1)).unwrap().recv().unwrap();
        assert_eq!(result.request_id, 1);
        assert_eq!(result.semantic_hv.dim(), 16_384);
        assert_eq!(result.execution.kind, VentralExecutionKind::HashStubV1);
    }

    #[test]
    fn multiple_requests_preserve_identity() {
        let channel = FoveationChannel::spawn_with_capacity(RoutingStrategy::Auto, 16);
        let mut receivers = Vec::new();
        for i in 0..10 {
            receivers.push((i, channel.request(make_request(i)).unwrap()));
        }
        for (id, rx) in receivers {
            assert_eq!(rx.recv().unwrap().request_id, id);
        }
    }

    #[test]
    fn blocking_request_works() {
        let channel = FoveationChannel::spawn(&FoveationConfig::default());
        assert_eq!(channel.request_blocking(make_request(99)).unwrap().request_id, 99);
    }

    #[test]
    fn provenance_survives_background_channel() {
        let channel = FoveationChannel::spawn(&FoveationConfig::default());
        let mut request = make_request(7);
        let observation = VisualObservationRef::new(
            VisualStreamRef::new(5, 6).unwrap(),
            request.frame_id,
            request.timestamp_us,
            VisualCaptureClock::StreamMonotonic,
        );
        request.source_observation = Some(observation);
        let result = channel.request_blocking(request).unwrap();
        assert_eq!(result.source_observation, Some(observation));
    }

    #[test]
    fn backpressure_is_bounded() {
        let channel = FoveationChannel::spawn_with_capacity(RoutingStrategy::Auto, 2);
        let mut sent = 0;
        let mut receivers = Vec::new();
        for i in 0..100u64 {
            match channel.request(make_request(i)) {
                Ok(rx) => {
                    sent += 1;
                    receivers.push(rx);
                }
                Err(mpsc::TrySendError::Full(())) => break,
                Err(mpsc::TrySendError::Disconnected(())) => {
                    panic!("Channel disconnected unexpectedly")
                }
            }
        }
        assert!(sent >= 2);
        for rx in receivers {
            assert_eq!(rx.recv().unwrap().semantic_hv.dim(), 16_384);
        }
    }

    #[test]
    fn requested_routing_is_retained_in_execution_receipt() {
        for routing in [
            RoutingStrategy::Auto,
            RoutingStrategy::AlwaysEmbed,
            RoutingStrategy::AlwaysOcr,
            RoutingStrategy::AlwaysCaption,
            RoutingStrategy::Full,
        ] {
            let channel = FoveationChannel::spawn_with_capacity(routing, 4);
            let result = channel.request_blocking(make_request(1)).unwrap();
            assert_eq!(result.execution.requested_routing, routing);
        }
    }
}
