// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-time audio output via cpal.
//!
//! Bridges [`StreamingSynth`] to the system audio device using a lock-free
//! ring buffer. The synthesis thread renders PCM chunks and pushes them into
//! a [`ringbuf`] producer; the cpal audio callback pulls from the consumer.
//!
//! # Architecture
//!
//! ```text
//! Game/Cognitive thread          Synthesis thread              cpal callback
//!       │                              │                            │
//!       │ update_state(MusicalState)   │                            │
//!       ├─────────────────────────────→│                            │
//!       │   (mpsc channel)             │                            │
//!       │                              │ StreamingSynth::render_chunk()
//!       │                              │ → Vec<[f32; 2]>           │
//!       │                              │                            │
//!       │                              │ ringbuf Producer::push()   │
//!       │                              ├───────────────────────────→│
//!       │                              │                            │ Consumer::pop()
//!       │                              │                            │ → cpal output
//! ```

use crate::streaming::StreamingSynth;
use crate::{MuseConfig, MusicalState};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Errors from the live audio system.
#[derive(Debug)]
pub enum LiveError {
    NoOutputDevice,
    NoSupportedConfig,
    DeviceError(String),
    StreamError(String),
}

impl std::fmt::Display for LiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoOutputDevice => write!(f, "no audio output device found"),
            Self::NoSupportedConfig => write!(f, "no supported audio output config"),
            Self::DeviceError(e) => write!(f, "audio device error: {e}"),
            Self::StreamError(e) => write!(f, "audio stream error: {e}"),
        }
    }
}

impl std::error::Error for LiveError {}

/// Real-time audio output engine.
///
/// Plays consciousness-driven music through the system audio device.
/// Call [`update_state()`](Self::update_state) from any thread to change
/// the cognitive state driving synthesis.
pub struct LiveMuseOutput {
    /// Sender for state updates (game thread → synthesis thread).
    state_tx: std::sync::mpsc::Sender<MusicalState>,
    /// Handle to the cpal stream (held to keep audio alive).
    _stream: cpal::Stream,
    /// Handle to the synthesis thread.
    _synth_thread: Option<std::thread::JoinHandle<()>>,
    /// Flag to signal shutdown.
    running: Arc<AtomicBool>,
    /// Negotiated sample rate.
    sample_rate: u32,
}

impl LiveMuseOutput {
    /// Open the default audio device and start streaming synthesis.
    pub fn new(config: MuseConfig) -> Result<Self, LiveError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(LiveError::NoOutputDevice)?;

        // Negotiate output config — prefer f32 stereo
        let supported = device
            .supported_output_configs()
            .map_err(|e| LiveError::DeviceError(e.to_string()))?
            .find(|c| c.channels() == 2 && c.sample_format() == cpal::SampleFormat::F32);

        let stream_config = match supported {
            Some(range) => {
                let target_rate = config.sample_rate;
                let min = range.min_sample_rate();
                let max = range.max_sample_rate();
                let rate = if target_rate >= min && target_rate <= max {
                    target_rate
                } else {
                    max
                };
                range.with_sample_rate(rate).config()
            }
            None => {
                // Fallback: use default config
                device
                    .default_output_config()
                    .map_err(|e| LiveError::DeviceError(e.to_string()))?
                    .config()
            }
        };

        let sample_rate = stream_config.sample_rate;
        let channels = stream_config.channels as usize;

        // Ring buffer: 4096 stereo frames (~93ms at 44100Hz)
        let ring_capacity = 4096 * channels;
        let ring = ringbuf::HeapRb::<f32>::new(ring_capacity);
        let (mut prod, mut cons) = ring.split();

        // State update channel
        let (state_tx, state_rx) = std::sync::mpsc::channel::<MusicalState>();

        let running = Arc::new(AtomicBool::new(true));
        let running_synth = running.clone();

        // Synthesis thread: renders chunks and pushes to ring buffer
        let synth_thread = std::thread::Builder::new()
            .name("muse-synth".into())
            .spawn(move || {
                let mut synth = StreamingSynth::new(config, sample_rate);

                while running_synth.load(Ordering::Relaxed) {
                    // Drain state updates (non-blocking, take latest)
                    let mut latest_state = None;
                    while let Ok(state) = state_rx.try_recv() {
                        latest_state = Some(state);
                    }
                    if let Some(state) = latest_state {
                        synth.update_state(&state);
                    }

                    // Render a chunk
                    let chunk = synth.render_chunk();

                    // Interleave stereo pairs into ring buffer
                    for pair in &chunk {
                        // Back-pressure: if ring is full, spin-wait briefly
                        let mut attempts = 0;
                        while prod.vacant_len() < channels {
                            std::thread::sleep(std::time::Duration::from_micros(100));
                            attempts += 1;
                            if attempts > 1000 || !running_synth.load(Ordering::Relaxed) {
                                return;
                            }
                        }
                        let _ = prod.try_push(pair[0]);
                        if channels > 1 {
                            let _ = prod.try_push(pair[1]);
                        }
                    }
                }
            })
            .map_err(|e| LiveError::StreamError(e.to_string()))?;

        // cpal callback: pulls from ring buffer
        let running_audio = running.clone();
        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if !running_audio.load(Ordering::Relaxed) {
                        data.fill(0.0);
                        return;
                    }
                    for sample in data.iter_mut() {
                        *sample = cons.try_pop().unwrap_or(0.0);
                    }
                },
                move |err| {
                    eprintln!("cpal stream error: {err}");
                },
                None,
            )
            .map_err(|e| LiveError::StreamError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| LiveError::StreamError(e.to_string()))?;

        Ok(Self {
            state_tx,
            _stream: stream,
            _synth_thread: Some(synth_thread),
            running,
            sample_rate,
        })
    }

    /// Update the cognitive state driving audio synthesis.
    ///
    /// Safe to call from any thread. If multiple updates arrive between
    /// synthesis ticks, only the latest is used.
    pub fn update_state(&self, state: &MusicalState) {
        let _ = self.state_tx.send(state.clone());
    }

    /// Negotiated output sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Stop audio playback and release resources.
    pub fn stop(self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Drop for LiveMuseOutput {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}
