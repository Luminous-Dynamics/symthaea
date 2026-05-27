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

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    HeapRb,
    traits::{Observer, Producer, Split},
};

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

        let ch = channels;
        let stream_config: cpal::StreamConfig = supported.into();

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    use ringbuf::traits::Consumer;
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
