// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lock-free audio output via cpal + ring buffer.
//!
//! Producer/consumer architecture:
//!   Synthesis thread → HeapRb<f32> → cpal output callback (audio thread)
//!
//! The ring buffer decouples synthesis timing from audio device timing while keeping
//! only a bounded near-term acoustic horizon. On underrun, the callback writes silence
//! (no click/pop). Barge-in may request consumer-owned queued-audio invalidation.
//!
//! Feature-gated under `live-voice`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    HeapRb,
    traits::{Consumer, Observer, Producer, Split},
};

/// Default maximum amount of synthesized speech allowed to sit ahead of device
/// playback in the live ring buffer.
///
/// This is an engineering default, not a measured optimum. It replaces the historical
/// ~2 second queue with a conversationally bounded horizon while remaining configurable
/// for devices that need more underrun tolerance.
pub const DEFAULT_BUFFER_AHEAD_MS: u32 = 250;

fn buffer_capacity_samples(sample_rate: u32, buffer_ahead_ms: u32) -> usize {
    let samples = (u64::from(sample_rate) * u64::from(buffer_ahead_ms)) / 1000;
    samples.max(1) as usize
}

/// Cloneable, capability-narrow request to discard queued live audio.
///
/// Calling [`AudioFlushHandle::request_flush`] never touches the ring-buffer
/// consumer directly. It only raises an atomic flag that the real-time CPAL
/// callback consumes at its next callback boundary, keeping consumer ownership
/// single-threaded and avoiding locks on the audio thread.
#[derive(Debug, Clone)]
pub struct AudioFlushHandle {
    requested: Arc<AtomicBool>,
}

impl AudioFlushHandle {
    fn new(requested: Arc<AtomicBool>) -> Self {
        Self { requested }
    }

    /// Request that all audio currently queued in the ring buffer be discarded.
    pub fn request_flush(&self) {
        self.requested.store(true, Ordering::Release);
    }

    /// Whether a flush request has not yet been consumed by the audio callback.
    pub fn is_pending(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }
}

/// Apply one pending flush request on the consumer-owning thread.
///
/// `Consumer::clear()` removes the complete occupied region while advancing the
/// consumer read index once. The atomic is consumed only after the callback has
/// exclusive access to the consumer, so control threads never touch ring state.
fn flush_consumer_if_requested<C>(consumer: &mut C, requested: &AtomicBool) -> usize
where
    C: Consumer<Item = f32>,
{
    if requested.swap(false, Ordering::AcqRel) {
        consumer.clear()
    } else {
        0
    }
}

/// Real-time audio output via cpal + ring buffer.
///
/// Can be in either `Live` mode (real audio device) or `Dummy` mode
/// (for headless/CI use with `speak_to_file()`).
pub struct AudioOutput {
    _stream: Option<cpal::Stream>,
    producer: Option<ringbuf::HeapProd<f32>>,
    sample_rate: u32,
    channels: u16,
    buffer_capacity: usize,
    buffer_ahead_ms: u32,
    flush_requested: Arc<AtomicBool>,
}

impl AudioOutput {
    /// Open the default audio output device with the default conversational
    /// ahead-of-playback budget.
    pub fn new() -> Result<Self> {
        Self::with_buffer_ahead_ms(DEFAULT_BUFFER_AHEAD_MS)
    }

    /// Open the default audio device with an explicit ring-buffer horizon.
    pub fn with_buffer_ahead_ms(buffer_ahead_ms: u32) -> Result<Self> {
        if buffer_ahead_ms == 0 {
            anyhow::bail!("audio buffer ahead budget must be greater than zero");
        }
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No audio output device found")?;
        Self::from_device(device, buffer_ahead_ms)
    }

    /// Open a specific audio output device by name using the default buffer horizon.
    pub fn with_device(device_name: &str) -> Result<Self> {
        Self::with_device_buffer_ahead_ms(device_name, DEFAULT_BUFFER_AHEAD_MS)
    }

    /// Open a specific output device with an explicit ahead-of-playback budget.
    pub fn with_device_buffer_ahead_ms(device_name: &str, buffer_ahead_ms: u32) -> Result<Self> {
        if buffer_ahead_ms == 0 {
            anyhow::bail!("audio buffer ahead budget must be greater than zero");
        }
        let host = cpal::default_host();
        let device = host
            .output_devices()
            .context("Failed to enumerate output devices")?
            .find(|d| {
                d.description()
                    .map(|desc| desc.name().contains(device_name))
                    .unwrap_or(false)
            })
            .with_context(|| format!("No output device matching '{device_name}'"))?;
        Self::from_device(device, buffer_ahead_ms)
    }

    /// Create a dummy output (no device). `push_samples()` discards all data.
    ///
    /// Used by [`super::live_voice::LiveVoice::new_headless()`] for `speak_to_file()`.
    pub fn new_dummy(sample_rate: u32) -> Self {
        Self {
            _stream: None,
            producer: None,
            sample_rate,
            channels: 1,
            buffer_capacity: 0,
            buffer_ahead_ms: 0,
            flush_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    fn from_device(device: cpal::Device, buffer_ahead_ms: u32) -> Result<Self> {
        let supported = device
            .default_output_config()
            .context("No supported output config")?;
        let sample_rate = supported.sample_rate();
        let channels = supported.channels();

        let buffer_capacity = buffer_capacity_samples(sample_rate, buffer_ahead_ms);
        let rb = HeapRb::<f32>::new(buffer_capacity);
        let (producer, mut consumer) = rb.split();
        let flush_requested = Arc::new(AtomicBool::new(false));
        let callback_flush = Arc::clone(&flush_requested);

        let ch = channels;
        let stream_config: cpal::StreamConfig = supported.into();

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Barge-in invalidation runs on the consumer-owning callback.
                    // No mutex is acquired and control threads never touch `consumer`.
                    let _ = flush_consumer_if_requested(&mut consumer, &callback_flush);

                    for sample in data.chunks_mut(ch as usize) {
                        if let Some(s) = consumer.try_pop() {
                            for out in sample.iter_mut() {
                                *out = s; // Mono → all channels
                            }
                        } else {
                            for out in sample.iter_mut() {
                                *out = 0.0; // Underrun → silence
                            }
                        }
                    }
                },
                |err| tracing::error!("Audio output stream error: {}", err),
                None,
            )
            .context("Failed to build output stream")?;

        stream.play().context("Failed to start output stream")?;

        Ok(Self {
            _stream: Some(stream),
            producer: Some(producer),
            sample_rate,
            channels,
            buffer_capacity,
            buffer_ahead_ms,
            flush_requested,
        })
    }

    /// Push audio samples into the ring buffer (non-blocking).
    ///
    /// Returns the number of samples actually written. If the buffer is full,
    /// remaining samples are dropped. Returns 0 on dummy output.
    pub fn push_samples(&mut self, samples: &[f32]) -> usize {
        let producer = match &mut self.producer {
            Some(p) => p,
            None => return 0,
        };
        let mut written = 0;
        for &s in samples {
            if producer.try_push(s).is_ok() {
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Take the ring buffer producer for use on a background thread.
    ///
    /// After calling this, `push_samples()` becomes a no-op until a new
    /// AudioOutput is created. Returns `None` if already taken or dummy.
    pub fn take_producer(&mut self) -> Option<ringbuf::HeapProd<f32>> {
        self.producer.take()
    }

    /// Return a cloneable request-only handle for queued-audio invalidation.
    pub fn flush_handle(&self) -> AudioFlushHandle {
        AudioFlushHandle::new(Arc::clone(&self.flush_requested))
    }

    /// Request queued-audio invalidation at the next audio callback boundary.
    pub fn request_flush(&self) {
        self.flush_requested.store(true, Ordering::Release);
    }

    /// Audio sample rate negotiated with the device (or headless default).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of output channels.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Ring buffer total capacity in mono samples.
    pub fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
    }

    /// Configured ahead-of-playback horizon in milliseconds. Dummy outputs report 0.
    pub fn buffer_ahead_ms(&self) -> u32 {
        self.buffer_ahead_ms
    }

    /// Approximate space remaining in the ring buffer. Returns 0 on dummy output.
    pub fn available_space(&self) -> usize {
        match &self.producer {
            Some(p) => p.vacant_len(),
            None => 0,
        }
    }

    /// Whether this is a live audio device (not dummy).
    pub fn is_live(&self) -> bool {
        self._stream.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_buffer_budget_is_explicit_and_bounded() {
        assert_eq!(DEFAULT_BUFFER_AHEAD_MS, 250);
        assert_eq!(buffer_capacity_samples(48_000, DEFAULT_BUFFER_AHEAD_MS), 12_000);
        assert_eq!(buffer_capacity_samples(24_000, DEFAULT_BUFFER_AHEAD_MS), 6_000);
    }

    #[test]
    fn custom_buffer_budget_scales_with_sample_rate() {
        assert_eq!(buffer_capacity_samples(48_000, 100), 4_800);
        assert_eq!(buffer_capacity_samples(44_100, 500), 22_050);
        assert_eq!(buffer_capacity_samples(24_000, 1), 24);
    }

    #[test]
    fn test_dummy_output() {
        let mut dummy = AudioOutput::new_dummy(24000);
        assert_eq!(dummy.sample_rate(), 24000);
        assert_eq!(dummy.channels(), 1);
        assert_eq!(dummy.buffer_ahead_ms(), 0);
        assert_eq!(dummy.available_space(), 0);
        assert!(!dummy.is_live());

        // push_samples on dummy returns 0
        let samples = vec![0.5f32; 100];
        assert_eq!(dummy.push_samples(&samples), 0);

        let flush = dummy.flush_handle();
        assert!(!flush.is_pending());
        flush.request_flush();
        assert!(flush.is_pending());
    }

    #[test]
    fn pending_flush_bulk_clears_consumer_and_is_consumed_once() {
        let rb = HeapRb::<f32>::new(8);
        let (mut producer, mut consumer) = rb.split();
        for sample in [0.1, 0.2, 0.3, 0.4] {
            producer.try_push(sample).unwrap();
        }

        let requested = AtomicBool::new(true);
        assert_eq!(flush_consumer_if_requested(&mut consumer, &requested), 4);
        assert!(!requested.load(Ordering::Acquire));
        assert!(consumer.try_pop().is_none());
        assert_eq!(flush_consumer_if_requested(&mut consumer, &requested), 0);
    }

    #[test]
    #[ignore] // Requires audio device
    fn test_audio_output_creation() {
        let output = AudioOutput::new();
        assert!(
            output.is_ok(),
            "AudioOutput should create on a system with audio"
        );
        let output = output.unwrap();
        assert!(output.sample_rate() > 0);
        assert!(output.channels() > 0);
        assert_eq!(output.buffer_ahead_ms(), DEFAULT_BUFFER_AHEAD_MS);
        assert!(output.available_space() > 0);
        assert!(output.is_live());
    }
}
