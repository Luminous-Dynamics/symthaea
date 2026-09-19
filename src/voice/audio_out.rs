// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lock-free audio output via cpal + ring buffer.
//!
//! Producer/consumer architecture:
//!   Synthesis thread(s) → serialized producer handle → HeapRb<f32>
//!   HeapRb<f32> → cpal output callback (audio thread)
//!
//! The CPAL consumer remains lock-free. Producer-side serialization is allowed because
//! synthesis/push workers are not the real-time callback. The ring keeps only a bounded
//! near-term acoustic horizon; on underrun the callback writes silence.
//!
//! Feature-gated under `live-voice`.

use std::sync::{Arc, Mutex};
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

/// Cloneable producer-side capability for live audio.
///
/// Multiple synthesis/push workers may retain this handle, but actual producer access
/// is serialized through a small producer-side mutex. The CPAL consumer callback does
/// not touch this mutex and remains lock-free.
#[derive(Clone)]
pub struct AudioProducerHandle {
    producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
}

impl AudioProducerHandle {
    fn new(producer: ringbuf::HeapProd<f32>) -> Self {
        Self {
            producer: Arc::new(Mutex::new(producer)),
        }
    }

    /// Push as many samples as fit while `keep_going` remains true.
    ///
    /// The producer lock is acquired once for the slice, avoiding per-sample mutex
    /// traffic while still letting callers observe interruption between samples.
    pub fn push_samples_while<F>(&self, samples: &[f32], mut keep_going: F) -> usize
    where
        F: FnMut() -> bool,
    {
        let Ok(mut producer) = self.producer.lock() else {
            return 0;
        };
        let mut written = 0;
        for &sample in samples {
            if !keep_going() {
                break;
            }
            if producer.try_push(sample).is_ok() {
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Push as many samples as currently fit.
    pub fn push_samples(&self, samples: &[f32]) -> usize {
        self.push_samples_while(samples, || true)
    }

    /// Approximate remaining ring space. Returns 0 if the producer mutex is poisoned.
    pub fn available_space(&self) -> usize {
        self.producer
            .lock()
            .map(|producer| producer.vacant_len())
            .unwrap_or(0)
    }

    fn try_into_raw(self) -> Result<ringbuf::HeapProd<f32>, Self> {
        match Arc::try_unwrap(self.producer) {
            Ok(mutex) => Ok(match mutex.into_inner() {
                Ok(producer) => producer,
                Err(poisoned) => poisoned.into_inner(),
            }),
            Err(producer) => Err(Self { producer }),
        }
    }
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
pub struct AudioOutput {
    _stream: Option<cpal::Stream>,
    producer: Option<AudioProducerHandle>,
    sample_rate: u32,
    channels: u16,
    buffer_capacity: usize,
    buffer_ahead_ms: u32,
    flush_requested: Arc<AtomicBool>,
}

impl AudioOutput {
    /// Open the default audio output device with the default conversational budget.
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
        let producer = AudioProducerHandle::new(producer);
        let flush_requested = Arc::new(AtomicBool::new(false));
        let callback_flush = Arc::clone(&flush_requested);

        let ch = channels;
        let stream_config: cpal::StreamConfig = supported.into();

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let _ = flush_consumer_if_requested(&mut consumer, &callback_flush);

                    for sample in data.chunks_mut(ch as usize) {
                        if let Some(s) = consumer.try_pop() {
                            for out in sample.iter_mut() {
                                *out = s;
                            }
                        } else {
                            for out in sample.iter_mut() {
                                *out = 0.0;
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
    pub fn push_samples(&mut self, samples: &[f32]) -> usize {
        self.producer
            .as_ref()
            .map(|producer| producer.push_samples(samples))
            .unwrap_or(0)
    }

    /// Return a cloneable producer-side capability without consuming `AudioOutput`.
    pub fn producer_handle(&self) -> Option<AudioProducerHandle> {
        self.producer.clone()
    }

    /// Take the raw producer when no cloneable producer capability has been shared.
    ///
    /// New code should prefer [`Self::producer_handle`]. This compatibility method
    /// returns `None` rather than invalidating existing producer handles.
    #[deprecated(note = "prefer producer_handle(); raw producer ownership is one-shot")]
    pub fn take_producer(&mut self) -> Option<ringbuf::HeapProd<f32>> {
        let handle = self.producer.take()?;
        match handle.try_into_raw() {
            Ok(producer) => Some(producer),
            Err(handle) => {
                self.producer = Some(handle);
                None
            }
        }
    }

    /// Return a cloneable request-only handle for queued-audio invalidation.
    pub fn flush_handle(&self) -> AudioFlushHandle {
        AudioFlushHandle::new(Arc::clone(&self.flush_requested))
    }

    /// Request queued-audio invalidation at the next audio callback boundary.
    pub fn request_flush(&self) {
        self.flush_requested.store(true, Ordering::Release);
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
    }

    pub fn buffer_ahead_ms(&self) -> u32 {
        self.buffer_ahead_ms
    }

    /// Approximate space remaining in the ring buffer. Returns 0 on dummy output.
    pub fn available_space(&self) -> usize {
        self.producer
            .as_ref()
            .map(AudioProducerHandle::available_space)
            .unwrap_or(0)
    }

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
    fn cloneable_producer_handle_reuses_one_ring() {
        let rb = HeapRb::<f32>::new(8);
        let (producer, mut consumer) = rb.split();
        let first = AudioProducerHandle::new(producer);
        let second = first.clone();

        assert_eq!(first.push_samples(&[0.1, 0.2]), 2);
        assert_eq!(consumer.try_pop(), Some(0.1));
        assert_eq!(consumer.try_pop(), Some(0.2));

        assert_eq!(second.push_samples(&[0.3, 0.4]), 2);
        assert_eq!(consumer.try_pop(), Some(0.3));
        assert_eq!(consumer.try_pop(), Some(0.4));
    }

    #[test]
    fn producer_handle_can_stop_mid_slice_without_consumer_locking() {
        let rb = HeapRb::<f32>::new(8);
        let (producer, mut consumer) = rb.split();
        let handle = AudioProducerHandle::new(producer);
        let mut allowed = 2usize;
        let written = handle.push_samples_while(&[1.0, 2.0, 3.0, 4.0], || {
            if allowed == 0 {
                false
            } else {
                allowed -= 1;
                true
            }
        });

        assert_eq!(written, 2);
        assert_eq!(consumer.try_pop(), Some(1.0));
        assert_eq!(consumer.try_pop(), Some(2.0));
        assert!(consumer.try_pop().is_none());
    }

    #[test]
    fn test_dummy_output() {
        let mut dummy = AudioOutput::new_dummy(24000);
        assert_eq!(dummy.sample_rate(), 24000);
        assert_eq!(dummy.channels(), 1);
        assert_eq!(dummy.buffer_ahead_ms(), 0);
        assert_eq!(dummy.available_space(), 0);
        assert!(!dummy.is_live());
        assert!(dummy.producer_handle().is_none());

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
    #[ignore]
    fn test_audio_output_creation() {
        let output = AudioOutput::new();
        assert!(output.is_ok());
        let output = output.unwrap();
        assert!(output.sample_rate() > 0);
        assert!(output.channels() > 0);
        assert_eq!(output.buffer_ahead_ms(), DEFAULT_BUFFER_AHEAD_MS);
        assert!(output.available_space() > 0);
        assert!(output.is_live());
    }
}
