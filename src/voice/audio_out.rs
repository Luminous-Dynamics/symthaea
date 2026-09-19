// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lock-free audio output via cpal + ring buffer.
//!
//! Producer/consumer architecture:
//!   Synthesis thread → HeapRb<f32> → cpal output callback (audio thread)
//!
//! The ring buffer decouples synthesis timing from audio device timing.
//! On underrun, the callback writes silence (no click/pop).
//!
//! Feature-gated under `live-voice`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    HeapRb,
    traits::{Observer, Producer, Split},
};

fn try_push_accounted(
    producer: &mut ringbuf::HeapProd<f32>,
    queued_samples: &AtomicUsize,
    sample: f32,
) -> bool {
    // Reserve accounting before publishing the sample into the lock-free ring so
    // the consumer can never pop a sample whose queue count has not been recorded.
    queued_samples.fetch_add(1, Ordering::AcqRel);
    if producer.try_push(sample).is_ok() {
        true
    } else {
        queued_samples.fetch_sub(1, Ordering::AcqRel);
        false
    }
}

/// Producer moved to a background thread while preserving exact queue accounting.
pub struct TrackedAudioProducer {
    inner: ringbuf::HeapProd<f32>,
    queued_samples: Arc<AtomicUsize>,
}

impl TrackedAudioProducer {
    /// Push as many samples as the ring currently accepts and return the count.
    pub fn push_samples(&mut self, samples: &[f32]) -> usize {
        let mut written = 0;
        for &sample in samples {
            if try_push_accounted(&mut self.inner, &self.queued_samples, sample) {
                written += 1;
            } else {
                break;
            }
        }
        written
    }
}

/// Cloneable, lock-free request and observation capability for live PCM playback.
///
/// The audio callback consumes purge requests with `swap(false)` and drains the
/// ring before producing the next device buffer. Queue occupancy is accounted across
/// both the foreground and detached-producer paths, so higher layers can distinguish
/// "synthesis finished" from "software-buffered audio is still pending".
#[derive(Debug, Clone)]
pub struct AudioOutputPurgeHandle {
    requested: Arc<AtomicBool>,
    queued_samples: Arc<AtomicUsize>,
}

impl AudioOutputPurgeHandle {
    fn new(requested: Arc<AtomicBool>, queued_samples: Arc<AtomicUsize>) -> Self {
        Self {
            requested,
            queued_samples,
        }
    }

    /// Request that the audio callback discard all PCM currently queued in the ring.
    pub fn request_purge(&self) {
        self.requested.store(true, Ordering::Release);
    }

    /// Whether a purge request is still waiting for an audio callback to consume it.
    pub fn is_pending(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    /// Number of mono samples currently accounted as queued in the software ring.
    pub fn queued_samples(&self) -> usize {
        self.queued_samples.load(Ordering::Acquire)
    }

    /// Whether software-buffered PCM remains pending for the device callback.
    pub fn has_queued_audio(&self) -> bool {
        self.queued_samples() != 0
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
    purge_requested: Arc<AtomicBool>,
    queued_samples: Arc<AtomicUsize>,
}

impl AudioOutput {
    /// Open the default audio output device and start streaming.
    ///
    /// Creates a ring buffer of `sample_rate * 2` capacity (~2 seconds).
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No audio output device found")?;
        Self::from_device(device)
    }

    /// Open a specific audio output device by name.
    pub fn with_device(device_name: &str) -> Result<Self> {
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
        Self::from_device(device)
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
            purge_requested: Arc::new(AtomicBool::new(false)),
            queued_samples: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn from_device(device: cpal::Device) -> Result<Self> {
        let supported = device
            .default_output_config()
            .context("No supported output config")?;
        let sample_rate = supported.sample_rate();
        let channels = supported.channels();

        let buffer_capacity = sample_rate as usize * 2;
        let rb = HeapRb::<f32>::new(buffer_capacity);
        let (producer, mut consumer) = rb.split();
        let purge_requested = Arc::new(AtomicBool::new(false));
        let callback_purge = Arc::clone(&purge_requested);
        let queued_samples = Arc::new(AtomicUsize::new(0));
        let callback_queued = Arc::clone(&queued_samples);

        let ch = channels;
        let stream_config: cpal::StreamConfig = supported.into();

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    use ringbuf::traits::Consumer;

                    if callback_purge.swap(false, Ordering::AcqRel) {
                        while consumer.try_pop().is_some() {
                            callback_queued.fetch_sub(1, Ordering::AcqRel);
                        }
                    }

                    for sample in data.chunks_mut(ch as usize) {
                        if let Some(s) = consumer.try_pop() {
                            callback_queued.fetch_sub(1, Ordering::AcqRel);
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
            purge_requested,
            queued_samples,
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
        for &sample in samples {
            if try_push_accounted(producer, &self.queued_samples, sample) {
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Take a tracked ring-buffer producer for use on a background thread.
    ///
    /// After calling this, `push_samples()` becomes a no-op until a new
    /// AudioOutput is created. Returns `None` if already taken or dummy.
    pub fn take_producer(&mut self) -> Option<TrackedAudioProducer> {
        self.producer.take().map(|inner| TrackedAudioProducer {
            inner,
            queued_samples: Arc::clone(&self.queued_samples),
        })
    }

    /// Return a cloneable capability for purge requests and queue observation.
    pub fn purge_handle(&self) -> AudioOutputPurgeHandle {
        AudioOutputPurgeHandle::new(
            Arc::clone(&self.purge_requested),
            Arc::clone(&self.queued_samples),
        )
    }

    /// Audio sample rate negotiated with the device (or headless default).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of output channels.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Ring buffer total capacity in samples.
    pub fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
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
    fn test_buffer_capacity_calculation() {
        let sample_rate: u32 = 48000;
        let expected_capacity = sample_rate as usize * 2;
        assert_eq!(expected_capacity, 96000);
    }

    #[test]
    fn test_dummy_output() {
        let mut dummy = AudioOutput::new_dummy(24000);
        assert_eq!(dummy.sample_rate(), 24000);
        assert_eq!(dummy.channels(), 1);
        assert_eq!(dummy.available_space(), 0);
        assert!(!dummy.is_live());

        // push_samples on dummy returns 0
        let samples = vec![0.5f32; 100];
        assert_eq!(dummy.push_samples(&samples), 0);
        assert_eq!(dummy.purge_handle().queued_samples(), 0);
    }

    #[test]
    fn purge_requests_are_cloneable_and_not_lost_before_consumption() {
        let output = AudioOutput::new_dummy(24000);
        let first = output.purge_handle();
        let second = first.clone();

        assert!(!first.is_pending());
        assert!(!first.has_queued_audio());
        second.request_purge();
        assert!(first.is_pending());
        first.request_purge();
        assert!(second.is_pending());
    }

    #[test]
    fn tracked_producer_accounting_reserves_before_publish() {
        let rb = HeapRb::<f32>::new(2);
        let (inner, _consumer) = rb.split();
        let queued_samples = Arc::new(AtomicUsize::new(0));
        let mut producer = TrackedAudioProducer {
            inner,
            queued_samples: Arc::clone(&queued_samples),
        };

        assert_eq!(producer.push_samples(&[0.1, 0.2, 0.3]), 2);
        assert_eq!(queued_samples.load(Ordering::Acquire), 2);
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
        assert!(output.available_space() > 0);
        assert!(output.is_live());
    }
}
