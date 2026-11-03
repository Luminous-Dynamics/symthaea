// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-time audio output via cpal.
//!
//! Bridges StreamingSynth to the system audio device using a lock-free ring
//! buffer. Rendered sample frames carry a generation epoch so an urgent
//! update can invalidate stale application-queued output before the callback
//! copies it into a device buffer.
//!
//! An epoch invalidation fences the application ring buffer only. It cannot
//! recall frames already copied into an in-flight callback or submitted to
//! the operating system / device queue. Acoustic silence needs a separate
//! loopback or physical-output observation.

use crate::output_epoch::{EpochTaggedStereoFrame, OutputEpochClock, OutputEpochError};
use crate::streaming::StreamingSynth;
use crate::{MuseConfig, MusicalState};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Errors from the live audio system.
#[derive(Debug)]
pub enum LiveError {
    NoOutputDevice,
    NoSupportedConfig,
    DeviceError(String),
    StreamError(String),
    ControlChannelClosed,
    OutputEpochExhausted,
}

impl std::fmt::Display for LiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoOutputDevice => write!(f, "no audio output device found"),
            Self::NoSupportedConfig => write!(f, "no supported audio output config"),
            Self::DeviceError(e) => write!(f, "audio device error: {e}"),
            Self::StreamError(e) => write!(f, "audio stream error: {e}"),
            Self::ControlChannelClosed => write!(f, "audio synthesis control channel closed"),
            Self::OutputEpochExhausted => write!(f, "output generation epoch exhausted"),
        }
    }
}

impl std::error::Error for LiveError {}

enum LiveCommand {
    UpdateState(MusicalState),
    InvalidateOutput { epoch: u64 },
    UpdateStateImmediately { state: MusicalState, epoch: u64 },
}

/// Real-time audio output engine.
///
/// Plays consciousness-driven music through the system audio device.
/// Ordinary state updates preserve already-rendered application-queued
/// frames. Use update_state_immediately for an urgent update that also
/// invalidates output from older generations.
pub struct LiveMuseOutput {
    /// Control sender (game thread → synthesis thread).
    state_tx: std::sync::mpsc::Sender<LiveCommand>,
    /// Handle to the cpal stream (held to keep audio alive).
    _stream: cpal::Stream,
    /// Handle to the synthesis thread.
    _synth_thread: Option<std::thread::JoinHandle<()>>,
    /// Flag to signal shutdown.
    running: Arc<AtomicBool>,
    /// Negotiated sample rate.
    sample_rate: u32,
    /// Shared output generation fence.
    output_epoch: OutputEpochClock,
    /// Frames discarded by the callback because their epoch was obsolete.
    stale_frames_dropped: Arc<AtomicU64>,
    /// Output frames for which the callback had no current-epoch data.
    underrun_frames: Arc<AtomicU64>,
    /// Frames rejected by the producer because the ring buffer was full.
    producer_enqueue_failures: Arc<AtomicU64>,
}

impl LiveMuseOutput {
    /// Open the default audio device and start streaming synthesis.
    pub fn new(config: MuseConfig) -> Result<Self, LiveError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(LiveError::NoOutputDevice)?;

        // Negotiate output config — prefer f32 stereo.
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
                // Fallback: use default config.
                device
                    .default_output_config()
                    .map_err(|e| LiveError::DeviceError(e.to_string()))?
                    .config()
            }
        };

        let sample_rate = stream_config.sample_rate;
        let channels = (stream_config.channels as usize).max(1);

        // Capacity is 4,096 stereo sample frames (~92.9ms at 44.1kHz).
        let ring = ringbuf::HeapRb::<EpochTaggedStereoFrame>::new(4096);
        let (mut prod, mut cons) = ring.split();

        // Synthesis control messages are drained before each rendered chunk.
        let (state_tx, state_rx) = std::sync::mpsc::channel::<LiveCommand>();

        let output_epoch = OutputEpochClock::new();
        let stale_frames_dropped = Arc::new(AtomicU64::new(0));
        let underrun_frames = Arc::new(AtomicU64::new(0));
        let producer_enqueue_failures = Arc::new(AtomicU64::new(0));

        let running = Arc::new(AtomicBool::new(true));
        let running_synth = running.clone();
        let synth_enqueue_failures = producer_enqueue_failures.clone();

        // Synthesis thread: render chunks and tag every sample frame with the
        // epoch that was current before the chunk began rendering.
        let synth_thread = std::thread::Builder::new()
            .name("muse-synth".into())
            .spawn(move || {
                let mut synth = StreamingSynth::new(config, sample_rate);
                let mut render_epoch = 0_u64;

                while running_synth.load(Ordering::Relaxed) {
                    // Drain control messages in order. An immediate state
                    // update applies the new state before changing render_epoch,
                    // so new-state output cannot be mislabeled with an old tag.
                    while let Ok(command) = state_rx.try_recv() {
                        match command {
                            LiveCommand::UpdateState(state) => synth.update_state(&state),
                            LiveCommand::InvalidateOutput { epoch } => {
                                render_epoch = epoch;
                            }
                            LiveCommand::UpdateStateImmediately { state, epoch } => {
                                synth.update_state(&state);
                                render_epoch = epoch;
                            }
                        }
                    }

                    let chunk_epoch = render_epoch;
                    let chunk = synth.render_chunk();

                    // A chunk interrupted by invalidation retains its old tag
                    // and is rejected even if this loop enqueues it afterward.
                    for pair in &chunk {
                        let mut attempts = 0;
                        while prod.vacant_len() < 1 {
                            std::thread::sleep(std::time::Duration::from_micros(100));
                            attempts += 1;
                            if attempts > 1000 || !running_synth.load(Ordering::Relaxed) {
                                return;
                            }
                        }

                        let frame = EpochTaggedStereoFrame::new(chunk_epoch, *pair);
                        if prod.try_push(frame).is_err() {
                            synth_enqueue_failures.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
            .map_err(|e| LiveError::StreamError(e.to_string()))?;

        // The real-time callback removes obsolete frames without emitting
        // them. It performs no allocation, lock, blocking call or channel access.
        let running_audio = running.clone();
        let audio_epoch = output_epoch.clone();
        let audio_stale_frames = stale_frames_dropped.clone();
        let audio_underrun_frames = underrun_frames.clone();
        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if !running_audio.load(Ordering::Relaxed) {
                        data.fill(0.0);
                        return;
                    }

                    for output_frame in data.chunks_mut(channels) {
                        // Keep popping obsolete frames until a current frame
                        // is found or the application queue is empty.
                        let current = loop {
                            match cons.try_pop() {
                                Some(frame) if audio_epoch.accepts(frame.epoch) => {
                                    break Some(frame);
                                }
                                Some(_) => {
                                    audio_stale_frames.fetch_add(1, Ordering::Relaxed);
                                }
                                None => break None,
                            }
                        };

                        match current {
                            Some(frame) if channels == 1 => {
                                output_frame[0] = (frame.samples[0] + frame.samples[1]) * 0.5;
                            }
                            Some(frame) => {
                                output_frame[0] = frame.samples[0];
                                if channels > 1 {
                                    output_frame[1] = frame.samples[1];
                                }
                                for extra_channel in output_frame.iter_mut().skip(2) {
                                    *extra_channel = 0.0;
                                }
                            }
                            None => {
                                output_frame.fill(0.0);
                                audio_underrun_frames.fetch_add(1, Ordering::Relaxed);
                            }
                        }
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
            output_epoch,
            stale_frames_dropped,
            underrun_frames,
            producer_enqueue_failures,
        })
    }

    /// Update the cognitive state driving future synthesis.
    ///
    /// Already-rendered application-queued frames are preserved.
    pub fn update_state(&self, state: &MusicalState) {
        let _ = self.state_tx.send(LiveCommand::UpdateState(state.clone()));
    }

    /// Advance the output epoch and invalidate older application-queued
    /// frames. This does not recall frames already copied into an in-flight
    /// callback or submitted to the operating system / physical device.
    pub fn invalidate_pending_output(&self) -> Result<u64, LiveError> {
        let epoch = self
            .output_epoch
            .invalidate()
            .map_err(|OutputEpochError::Exhausted| LiveError::OutputEpochExhausted)?;
        self.state_tx
            .send(LiveCommand::InvalidateOutput { epoch })
            .map_err(|_| LiveError::ControlChannelClosed)?;
        Ok(epoch)
    }

    /// Update cognitive state and invalidate older application-queued output
    /// as one ordered control operation. The callback fence advances
    /// immediately; the synthesis thread applies the new state before tagging
    /// subsequent chunks with the new epoch.
    pub fn update_state_immediately(
        &self,
        state: &MusicalState,
    ) -> Result<u64, LiveError> {
        let epoch = self
            .output_epoch
            .invalidate()
            .map_err(|OutputEpochError::Exhausted| LiveError::OutputEpochExhausted)?;
        self.state_tx
            .send(LiveCommand::UpdateStateImmediately {
                state: state.clone(),
                epoch,
            })
            .map_err(|_| LiveError::ControlChannelClosed)?;
        Ok(epoch)
    }

    /// Current generation epoch authorized by the callback.
    pub fn output_epoch(&self) -> u64 {
        self.output_epoch.current()
    }

    /// Number of obsolete frames removed by the callback.
    ///
    /// This is callback-observed application-queue discard, not a count of
    /// physically canceled or acoustically silent frames.
    pub fn stale_output_frames_dropped(&self) -> u64 {
        self.stale_frames_dropped.load(Ordering::Relaxed)
    }

    /// Number of output frames for which the callback had no current-epoch
    /// data. This is separate from stale-frame discard.
    pub fn underrun_output_frames(&self) -> u64 {
        self.underrun_frames.load(Ordering::Relaxed)
    }

    /// Number of ring-buffer enqueue failures observed by the producer.
    pub fn producer_enqueue_failures(&self) -> u64 {
        self.producer_enqueue_failures.load(Ordering::Relaxed)
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
