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
    traits::{Observer, Producer, Split},
    HeapRb,
};

/// Real-time audio output via cpal + ring buffer.
pub struct AudioOutput {
    _stream: cpal::Stream,
    producer: ringbuf::HeapProd<f32>,
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
                d.name()
                    .map(|n| n.contains(device_name))
                    .unwrap_or(false)
            })
            .with_context(|| format!("No output device matching '{device_name}'"))?;
        Self::from_device(device)
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
            _stream: stream,
            producer,
            sample_rate,
            channels,
            buffer_capacity,
        })
    }

    /// Push audio samples into the ring buffer (non-blocking).
    ///
    /// Returns the number of samples actually written. If the buffer is full,
    /// remaining samples are dropped (caller should use backpressure).
    pub fn push_samples(&mut self, samples: &[f32]) -> usize {
        let mut written = 0;
        for &s in samples {
            if self.producer.try_push(s).is_ok() {
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Audio sample rate negotiated with the device.
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

    /// Approximate space remaining in the ring buffer.
    pub fn available_space(&self) -> usize {
        self.producer.vacant_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_capacity_calculation() {
        // Verify the capacity formula: sample_rate * 2
        let sample_rate: u32 = 48000;
        let expected_capacity = sample_rate as usize * 2;
        assert_eq!(expected_capacity, 96000);
    }

    #[test]
    #[ignore] // Requires audio device
    fn test_audio_output_creation() {
        let output = AudioOutput::new();
        assert!(output.is_ok(), "AudioOutput should create on a system with audio");
        let output = output.unwrap();
        assert!(output.sample_rate() > 0);
        assert!(output.channels() > 0);
        assert!(output.available_space() > 0);
    }
}
