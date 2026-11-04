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

use crate::output_clock::{
    OutputClockAnchorSchemaV1, OutputClockAnchorSourceV1, OutputClockAnchorV1,
    OutputClockStreamDescriptorV1,
};
use crate::streaming::StreamingSynth;
use crate::{MuseConfig, MusicalState};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use std::sync::{Arc, Mutex, PoisonError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static NEXT_STREAM_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Errors from the live audio system.
#[derive(Debug)]
pub enum LiveError {
    NoOutputDevice,
    NoSupportedConfig,
    DeviceError(String),
    StreamError(String),
    StreamInstanceIdExhausted,
    InvalidClockStreamDescriptor,
}

impl std::fmt::Display for LiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoOutputDevice => write!(f, "no audio output device found"),
            Self::NoSupportedConfig => write!(f, "no supported audio output config"),
            Self::DeviceError(e) => write!(f, "audio device error: {e}"),
            Self::StreamError(e) => write!(f, "audio stream error: {e}"),
            Self::StreamInstanceIdExhausted => write!(f, "CPAL stream instance ID space exhausted"),
            Self::InvalidClockStreamDescriptor => write!(f, "invalid CPAL output clock stream descriptor"),
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
    /// Non-real-time consumer for callback clock anchors.
    clock_anchor_consumer: Mutex<ringbuf::HeapCons<OutputClockAnchorV1>>,
    /// Number of callbacks whose anchor could not be retained (timestamp
    /// invalid, frame counter invalid, or anchor queue full).
    clock_anchor_dropped_callbacks: Arc<AtomicU64>,
    /// Stable identity for this CPAL stream instance's clock domain.
    stream_instance_id: u64,
    /// Stream-level clock and device configuration identity shared by anchors.
    clock_stream_descriptor: OutputClockStreamDescriptorV1,
}

impl LiveMuseOutput {
    /// Open the default audio device and start streaming synthesis.
    pub fn new(config: MuseConfig) -> Result<Self, LiveError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(LiveError::NoOutputDevice)?;
        let device_id = device
            .id()
            .map_err(|error| LiveError::DeviceError(error.to_string()))?
            .to_string();

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
        let channel_count = stream_config.channels;
        let channels = (channel_count as usize).max(1);

        let stream_instance_id = NEXT_STREAM_INSTANCE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| LiveError::StreamInstanceIdExhausted)?;
        let clock_stream_descriptor =
            OutputClockStreamDescriptorV1::from_cpal_device_id(
                stream_instance_id,
                &device_id,
                sample_rate,
                channel_count,
            )
            .ok_or(LiveError::InvalidClockStreamDescriptor)?;

        // Ring buffer: 4096 stereo frames (~93ms at 44100Hz)
        let ring_capacity = 4096 * channels;
        let ring = ringbuf::HeapRb::<f32>::new(ring_capacity);
        let (mut prod, mut cons) = ring.split();

        // Separate SPSC channel for timestamped output-clock evidence. The
        // CPAL callback owns the producer; callers drain the consumer outside
        // the real-time thread.
        let anchor_ring = ringbuf::HeapRb::<OutputClockAnchorV1>::new(1024);
        let (mut anchor_prod, anchor_cons) = anchor_ring.split();
        let clock_anchor_dropped_callbacks = Arc::new(AtomicU64::new(0));

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
        let mut stream_origin: Option<cpal::StreamInstant> = None;
        let mut output_frame_cursor = 0_u64;
        let mut anchor_index = 0_u64;
        let running_audio = running.clone();
        let anchor_drop_counter = clock_anchor_dropped_callbacks.clone();
        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    if !running_audio.load(Ordering::Relaxed) {
                        data.fill(0.0);
                        return;
                    }

                    for sample in data.iter_mut() {
                        *sample = cons.try_pop().unwrap_or(0.0);
                    }

                    // Capture the CPAL stream clock without cross-stream
                    // assumptions. The first callback is this stream's local
                    // origin. All callback/playback times are stored relative
                    // to that origin and bound to the first sample frame in
                    // this callback's output buffer.
                    let timestamp = info.timestamp();
                    let frame_count = data.len() / channels;
                    let frame_shape_valid = data.len() % channels == 0;
                    let frame_count_u32 = u32::try_from(frame_count).ok();
                    let mut anchor_dropped = !frame_shape_valid || frame_count == 0;

                    if let (Some(frame_count), true) = (frame_count_u32, frame_shape_valid) {
                        let callback_instant = timestamp.callback;
                        let playback_instant = timestamp.playback;
                        let origin = stream_origin.get_or_insert(callback_instant);

                        let callback_ns = callback_instant
                            .duration_since(origin)
                            .and_then(|duration| u64::try_from(duration.as_nanos()).ok());
                        let playback_ns = playback_instant
                            .duration_since(origin)
                            .and_then(|duration| u64::try_from(duration.as_nanos()).ok());
                        let callback_to_playback_ns = playback_instant
                            .duration_since(&callback_instant)
                            .and_then(|duration| u64::try_from(duration.as_nanos()).ok());

                        let next_frame = output_frame_cursor.checked_add(frame_count as u64);
                        let next_anchor = anchor_index.checked_add(1);

                        match (
                            callback_ns,
                            playback_ns,
                            callback_to_playback_ns,
                            next_frame,
                            next_anchor,
                        ) {
                            (Some(callback_ns), Some(playback_ns), Some(delay_ns), Some(frame_end), Some(anchor_end)) => {
                                let anchor = OutputClockAnchorV1 {
                                    schema_version: OutputClockAnchorSchemaV1::V1,
                                    source: OutputClockAnchorSourceV1::CpalOutputStreamTimestampV1,
                                    stream_instance_id,
                                    anchor_index,
                                    first_output_frame_index: output_frame_cursor,
                                    frame_count,
                                    sample_rate_hz: sample_rate,
                                    channel_count,
                                    callback_stream_time_ns: callback_ns,
                                    playback_stream_time_ns: playback_ns,
                                    callback_to_playback_ns: delay_ns,
                                    timestamp_uncertainty_ns: None,
                                };
                                if anchor_prod.try_push(anchor).is_err() {
                                    anchor_dropped = true;
                                }
                                output_frame_cursor = frame_end;
                                anchor_index = anchor_end;
                            }
                            _ => {
                                anchor_dropped = true;
                                if let Some(frame_end) = next_frame {
                                    output_frame_cursor = frame_end;
                                }
                                if let Some(anchor_end) = next_anchor {
                                    anchor_index = anchor_end;
                                }
                            }
                        }
                    }

                    // One relaxed counter update per callback, never one per
                    // sample. Anchor loss is explicit and does not block audio.
                    if anchor_dropped {
                        anchor_drop_counter.fetch_add(1, Ordering::Relaxed);
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
            clock_anchor_consumer: Mutex::new(anchor_cons),
            clock_anchor_dropped_callbacks,
            stream_instance_id,
            clock_stream_descriptor,
        })
    }

    /// Update the cognitive state driving audio synthesis.
    ///
    /// Safe to call from any thread. If multiple updates arrive between
    /// synthesis ticks, only the latest is used.
    pub fn update_state(&self, state: &MusicalState) {
        let _ = self.state_tx.send(state.clone());
    }


    /// Drain up to max_items native CPAL output-clock anchors.
    ///
    /// This method allocates and takes a mutex, so it must be called from a
    /// non-real-time thread. The audio callback only performs a bounded,
    /// lock-free SPSC push.
    pub fn drain_output_clock_anchors(&self, max_items: usize) -> Vec<OutputClockAnchorV1> {
        let mut consumer = self
            .clock_anchor_consumer
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut anchors = Vec::with_capacity(max_items.min(1024));
        while anchors.len() < max_items {
            match consumer.try_pop() {
                Some(anchor) => anchors.push(anchor),
                None => break,
            }
        }
        anchors
    }

    /// Stream-level device/config identity shared by the anchors returned
    /// from drain_output_clock_anchors. The hardware device ID is represented
    /// by a SHA-256 fingerprint rather than a raw host identifier.
    pub fn clock_stream_descriptor(&self) -> &OutputClockStreamDescriptorV1 {
        &self.clock_stream_descriptor
    }

    /// Number of callbacks whose timestamp/frame anchor could not be retained.
    ///
    /// This includes missing/inconsistent timestamp values and a full
    /// anchor ring. It is not an audio underrun count.
    pub fn clock_anchor_dropped_callbacks(&self) -> u64 {
        self.clock_anchor_dropped_callbacks.load(Ordering::Relaxed)
    }

    /// Stable identity for this CPAL stream instance's clock domain.
    pub fn stream_instance_id(&self) -> u64 {
        self.stream_instance_id
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
