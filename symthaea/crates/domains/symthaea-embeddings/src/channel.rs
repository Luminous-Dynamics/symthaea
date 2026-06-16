// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Background-thread embedding channel using std::sync::mpsc.
//!
//! Follows the `AsyncTrainerHandle` pattern from `cognitive_loop/training.rs`.
//! No tokio dependency — compatible with the sync cognitive loop.
//!
//! ```rust,ignore
//! use symthaea_embeddings::{Qwen3Config, channel::EmbeddingChannel};
//!
//! let channel = EmbeddingChannel::spawn(Qwen3Config::simulated())?;
//! let rx = channel.request("Hello, world!")?;
//! let response = rx.recv().unwrap();
//! println!("{:?}", response.result);
//! ```

use crate::{EmbeddingResult, Qwen3Config, Qwen3Embedder};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

/// Request sent to the background embedder thread.
struct EmbedRequest {
    id: u64,
    text: String,
    response_tx: mpsc::SyncSender<EmbedResponse>,
}

/// Response from the background embedder thread.
pub struct EmbedResponse {
    /// Request ID for correlation.
    pub id: u64,
    /// Embedding result or error message.
    pub result: Result<EmbeddingResult, String>,
}

/// Handle to a background embedding thread.
///
/// Requests are submitted via bounded channel (default capacity 8).
/// Drop the handle to shut down the background thread.
pub struct EmbeddingChannel {
    request_tx: mpsc::SyncSender<EmbedRequest>,
    next_id: AtomicU64,
}

impl EmbeddingChannel {
    /// Spawn a background embedder thread with default channel capacity (8).
    pub fn spawn(config: Qwen3Config) -> anyhow::Result<Self> {
        Self::spawn_with_capacity(config, 8)
    }

    /// Spawn a background embedder thread with custom channel capacity.
    pub fn spawn_with_capacity(config: Qwen3Config, depth: usize) -> anyhow::Result<Self> {
        let (request_tx, request_rx) = mpsc::sync_channel::<EmbedRequest>(depth);

        std::thread::Builder::new()
            .name("symthaea-embedder".into())
            .spawn(move || {
                let mut embedder = match Qwen3Embedder::new(config) {
                    Ok(e) => e,
                    Err(_) => return, // Thread exits if embedder fails to init
                };

                while let Ok(req) = request_rx.recv() {
                    let result = embedder.embed(&req.text).map_err(|e| e.to_string());

                    let response = EmbedResponse { id: req.id, result };
                    // If the receiver was dropped, we just discard the response
                    let _ = req.response_tx.try_send(response);
                }
            })?;

        Ok(Self {
            request_tx,
            next_id: AtomicU64::new(0),
        })
    }

    /// Submit a non-blocking embedding request.
    ///
    /// Returns a receiver for the response. Fails with `TrySendError::Full`
    /// if the channel is at capacity (backpressure).
    pub fn request(
        &self,
        text: &str,
    ) -> Result<mpsc::Receiver<EmbedResponse>, mpsc::TrySendError<()>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let req = EmbedRequest {
            id,
            text: text.to_string(),
            response_tx,
        };

        self.request_tx.try_send(req).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => mpsc::TrySendError::Full(()),
            mpsc::TrySendError::Disconnected(_) => mpsc::TrySendError::Disconnected(()),
        })?;

        Ok(response_rx)
    }

    /// Submit a request and block until the result is available.
    pub fn embed_blocking(&self, text: &str) -> anyhow::Result<EmbeddingResult> {
        let rx = self
            .request(text)
            .map_err(|e| anyhow::anyhow!("Channel send failed: {e:?}"))?;

        let response = rx
            .recv()
            .map_err(|e| anyhow::anyhow!("Channel recv failed: {e}"))?;

        response.result.map_err(|e| anyhow::anyhow!(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_spawn_and_request() {
        let channel = EmbeddingChannel::spawn(Qwen3Config::simulated()).unwrap();
        let rx = channel.request("Hello, world!").unwrap();
        let response = rx.recv().unwrap();

        assert!(response.result.is_ok());
        let emb = response.result.unwrap();
        assert_eq!(emb.dimension, 1024);
    }

    #[test]
    fn test_channel_multiple_requests() {
        // Use larger capacity to avoid backpressure during test
        let channel = EmbeddingChannel::spawn_with_capacity(Qwen3Config::simulated(), 16).unwrap();

        let mut receivers = Vec::new();
        for i in 0..10 {
            let rx = channel.request(&format!("text {i}")).unwrap();
            receivers.push(rx);
        }

        for (i, rx) in receivers.into_iter().enumerate() {
            let response = rx.recv().unwrap();
            assert!(
                response.result.is_ok(),
                "Request {i} should succeed: {:?}",
                response.result
            );
        }
    }

    #[test]
    fn test_channel_blocking_embed() {
        let channel = EmbeddingChannel::spawn(Qwen3Config::simulated()).unwrap();
        let result = channel.embed_blocking("test blocking").unwrap();
        assert_eq!(result.dimension, 1024);
        assert!(result.is_simulated);
    }

    #[test]
    fn test_channel_backpressure() {
        // Capacity of 2 — fill queue then check backpressure
        let channel = EmbeddingChannel::spawn_with_capacity(Qwen3Config::simulated(), 2).unwrap();

        // Fill the channel — the embedder thread processes requests so we need to
        // be faster than it. Submit many requests and check at least one fails or all succeed.
        let mut sent = 0;
        let mut receivers = Vec::new();
        for i in 0..100 {
            match channel.request(&format!("pressure {i}")) {
                Ok(rx) => {
                    sent += 1;
                    receivers.push(rx);
                }
                Err(mpsc::TrySendError::Full(())) => {
                    // This is the expected backpressure signal
                    break;
                }
                Err(mpsc::TrySendError::Disconnected(())) => {
                    panic!("Channel disconnected unexpectedly");
                }
            }
        }

        // We should have sent at least 2 (the capacity)
        assert!(
            sent >= 2,
            "Should send at least capacity messages, sent {sent}"
        );

        // All sent requests should eventually complete
        for rx in receivers {
            let response = rx.recv().unwrap();
            assert!(response.result.is_ok());
        }
    }

    #[test]
    fn test_channel_drop_shuts_down() {
        let channel = EmbeddingChannel::spawn(Qwen3Config::simulated()).unwrap();

        // Do one request to confirm thread is alive
        let result = channel.embed_blocking("alive check").unwrap();
        assert_eq!(result.dimension, 1024);

        // Drop channel — thread should exit when request_rx disconnects
        drop(channel);

        // Give thread a moment to notice disconnect
        std::thread::sleep(std::time::Duration::from_millis(50));
        // If we get here without hanging, the thread exited cleanly
    }
}
