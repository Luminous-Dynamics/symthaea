// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Real-time Microphone Capture for STT → Perception
//!
//! Wires a cpal input stream + ringbuf on the audio thread, a background
//! worker that runs `symthaea_stt::StreamProcessor`, and an mpsc channel
//! that delivers `ContinuousHV` snapshots to the cognitive loop's perception
//! phase.
//!
//! ```text
//! cpal callback (audio RT) ─► ringbuf (lockfree) ─► STT worker thread
//!                                                      │
//!                                                      ▼
//!                          mpsc::sync_channel(cap=N) ─► perception_phase
//!                          (bounded, drops on full)     (try_recv latest)
//! ```
//!
//! Each layer has a clear job:
//! - The cpal callback must not allocate or block. It only pushes raw f32
//!   samples into a lockfree ring buffer.
//! - The worker drains the ring buffer, runs LTC + simple mel features, and
//!   periodically bundles N HV16 frames into one 16,384-dim `ContinuousHV`.
//! - Perception polls the mpsc via `drain_latest()` — non-blocking, latest-wins.

#![cfg(feature = "voice-stt-live")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::{HeapCons, HeapRb};
use symthaea_core::hdc::ContinuousHV;
use symthaea_stt::{bundle, StreamConfig, StreamProcessor, HV16};

/// Configuration for live microphone capture.
#[derive(Debug, Clone)]
pub struct MicCaptureConfig {
    /// Target sample rate (Hz). 16,000 matches the STT AudioProjector default.
    pub sample_rate: u32,
    /// Bounded HV channel capacity. Older values drop when full; perception
    /// only consumes the newest via `drain_latest()`.
    pub hv_channel_cap: usize,
    /// Ring buffer capacity in samples (1 second of 16 kHz ≈ 16,000).
    pub ringbuf_samples: usize,
    /// Stream processor config. `low_latency()` is a reasonable default.
    pub stream_config: StreamConfig,
    /// Worker idle wait when the ring buffer is empty, in microseconds.
    /// Small enough not to delay frames; large enough not to busy-loop.
    pub worker_idle_us: u64,
    /// Minimum number of HV16 frames bundled before an HV is emitted.
    /// A single 10 ms frame is noisy; 5 frames ≈ 50 ms of auditory context.
    pub min_frames_per_emit: usize,
}

impl Default for MicCaptureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            hv_channel_cap: 8,
            ringbuf_samples: 16_000,
            stream_config: StreamConfig::low_latency(),
            worker_idle_us: 1_000,
            min_frames_per_emit: 5,
        }
    }
}

/// Owns a live mic stream + background STT worker.
///
/// `drain_latest()` is the non-blocking API the perception phase polls each
/// cycle. The stream and worker are torn down when the handle is dropped.
pub struct MicCaptureHandle {
    _stream: cpal::Stream,
    // Wrapped in Mutex because `mpsc::Receiver` is !Sync and the containing
    // CognitiveLoopService must stay Sync for its MetricsProvider bound.
    // Drain is single-threaded (perception phase) so the mutex is uncontended.
    hv_rx: Mutex<Receiver<ContinuousHV>>,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    sample_rate: u32,
}

impl MicCaptureHandle {
    /// Build the capture graph against the default input device.
    pub fn start(config: MicCaptureConfig) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?;
        let supported = device
            .default_input_config()
            .context("query default input config")?;

        let sample_format = supported.sample_format();
        let mut cpal_config: cpal::StreamConfig = supported.into();
        cpal_config.sample_rate = config.sample_rate;

        let ring = HeapRb::<f32>::new(config.ringbuf_samples);
        let (producer, consumer) = ring.split();

        let shutdown = Arc::new(AtomicBool::new(false));
        let (hv_tx, hv_rx) = mpsc::sync_channel(config.hv_channel_cap);

        let stream = build_input_stream(&device, &cpal_config, sample_format, producer)?;
        stream.play().context("start input stream")?;

        let worker_shutdown = shutdown.clone();
        let worker_config = config.clone();
        let worker = thread::Builder::new()
            .name("symthaea-stt-worker".into())
            .spawn(move || {
                run_worker(consumer, worker_shutdown, hv_tx, worker_config);
            })
            .context("spawn stt worker thread")?;

        Ok(Self {
            _stream: stream,
            hv_rx: Mutex::new(hv_rx),
            shutdown,
            worker: Some(worker),
            sample_rate: config.sample_rate,
        })
    }

    /// Drain the HV channel and return the most recent `ContinuousHV`, or `None`.
    ///
    /// Non-blocking. Older HVs in the queue are discarded so the perception
    /// phase always sees the latest auditory snapshot without accumulating lag.
    pub fn drain_latest(&self) -> Option<ContinuousHV> {
        let rx = match self.hv_rx.lock() {
            Ok(rx) => rx,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut latest = None;
        while let Ok(hv) = rx.try_recv() {
            latest = Some(hv);
        }
        latest
    }

    /// Configured sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Drop for MicCaptureHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Build the cpal input stream dispatched on the device's sample format.
///
/// The `producer` is moved into exactly one callback (the runtime-selected
/// match arm). Sample conversion is done manually to avoid pulling additional
/// traits.
fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: SampleFormat,
    producer: ringbuf::HeapProd<f32>,
) -> Result<cpal::Stream> {
    let err_fn = |err| eprintln!("stt input stream error: {err}");
    let stream = match sample_format {
        SampleFormat::F32 => {
            let mut producer = producer;
            device
                .build_input_stream(
                    config,
                    move |data: &[f32], _| {
                        for &s in data {
                            if producer.try_push(s).is_err() {
                                break;
                            }
                        }
                    },
                    err_fn,
                    None,
                )
                .context("build f32 input stream")?
        }
        SampleFormat::I16 => {
            let mut producer = producer;
            device
                .build_input_stream(
                    config,
                    move |data: &[i16], _| {
                        for &s in data {
                            let f = s as f32 / i16::MAX as f32;
                            if producer.try_push(f).is_err() {
                                break;
                            }
                        }
                    },
                    err_fn,
                    None,
                )
                .context("build i16 input stream")?
        }
        SampleFormat::U16 => {
            let mut producer = producer;
            device
                .build_input_stream(
                    config,
                    move |data: &[u16], _| {
                        for &s in data {
                            let f = (s as f32 - 32_768.0) / 32_768.0;
                            if producer.try_push(f).is_err() {
                                break;
                            }
                        }
                    },
                    err_fn,
                    None,
                )
                .context("build u16 input stream")?
        }
        other => return Err(anyhow!("unsupported sample format: {other:?}")),
    };
    Ok(stream)
}

/// Drain the ring buffer, run STT, emit bundled `ContinuousHV`s.
///
/// Exposed `pub(super)` so the synthetic-sample tests can drive the worker
/// directly without needing a real audio device.
pub(super) fn run_worker(
    mut consumer: HeapCons<f32>,
    shutdown: Arc<AtomicBool>,
    hv_tx: SyncSender<ContinuousHV>,
    config: MicCaptureConfig,
) {
    let mut processor = StreamProcessor::new(config.stream_config.clone());
    let mut scratch = vec![0.0_f32; 4096];
    let mut pending: Vec<HV16> = Vec::new();

    while !shutdown.load(Ordering::Relaxed) {
        let n = consumer.pop_slice(&mut scratch);
        if n > 0 {
            processor.push_audio(&scratch[..n]);
            for frame in processor.process() {
                pending.push(frame.hv);
            }

            if pending.len() >= config.min_frames_per_emit {
                let bundled = bundle(&pending);
                pending.clear();
                let hv = ContinuousHV::from_vec(bundled.to_core_continuous());
                match hv_tx.try_send(hv) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        // Receiver is behind — drop silently; next emit replaces.
                    }
                    Err(TrySendError::Disconnected(_)) => break,
                }
            }
        } else {
            thread::sleep(Duration::from_micros(config.worker_idle_us));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ringbuf::traits::Producer as _;

    fn make_config() -> MicCaptureConfig {
        MicCaptureConfig {
            // Tight config so tests emit quickly.
            hv_channel_cap: 4,
            ringbuf_samples: 32_000,
            worker_idle_us: 100,
            min_frames_per_emit: 2,
            ..Default::default()
        }
    }

    fn sine(n: usize, freq_hz: f32, sample_rate: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn synthetic_samples_produce_hv() {
        // Drive the worker directly with synthetic samples — no cpal, no mic.
        let config = make_config();
        let ring = HeapRb::<f32>::new(config.ringbuf_samples);
        let (mut producer, consumer) = ring.split();

        let shutdown = Arc::new(AtomicBool::new(false));
        let (hv_tx, hv_rx) = mpsc::sync_channel(config.hv_channel_cap);

        let worker_shutdown = shutdown.clone();
        let worker_config = config.clone();
        let worker = thread::spawn(move || {
            run_worker(consumer, worker_shutdown, hv_tx, worker_config);
        });

        // Push 500ms of 440Hz sine at 16kHz = 8000 samples.
        let samples = sine(8000, 440.0, config.sample_rate);
        let pushed = producer.push_slice(&samples);
        assert_eq!(pushed, samples.len(), "ringbuf must fit 500ms of samples");

        // Wait up to 1s for at least one HV emission.
        let hv = wait_for_hv(&hv_rx, Duration::from_millis(1000));

        shutdown.store(true, Ordering::SeqCst);
        let _ = worker.join();

        let hv = hv.expect("worker should emit at least one HV within 1s");
        assert_eq!(
            hv.values.len(),
            16_384,
            "bundled HV must match symthaea-core CORE_HDC_DIM"
        );
        // Bipolar encoding guarantees values in [-1, 1].
        assert!(hv.values.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn drain_latest_keeps_newest_only() {
        let (tx, rx) = mpsc::sync_channel::<ContinuousHV>(8);

        let a = ContinuousHV::from_vec(vec![0.1; 16_384]);
        let b = ContinuousHV::from_vec(vec![0.5; 16_384]);
        let c = ContinuousHV::from_vec(vec![0.9; 16_384]);

        tx.send(a).unwrap();
        tx.send(b).unwrap();
        tx.send(c).unwrap();

        // Simulate the drain_latest algorithm against a bare receiver.
        let mut latest = None;
        while let Ok(hv) = rx.try_recv() {
            latest = Some(hv);
        }

        let hv = latest.expect("at least one HV");
        assert!((hv.values[0] - 0.9).abs() < 1e-6, "must keep newest");
    }

    #[test]
    fn worker_exits_on_shutdown_even_with_no_samples() {
        let config = make_config();
        let ring = HeapRb::<f32>::new(config.ringbuf_samples);
        let (_producer, consumer) = ring.split();

        let shutdown = Arc::new(AtomicBool::new(false));
        let (hv_tx, _hv_rx) = mpsc::sync_channel(config.hv_channel_cap);

        let worker_shutdown = shutdown.clone();
        let worker = thread::spawn(move || {
            run_worker(consumer, worker_shutdown, hv_tx, config);
        });

        // No samples pushed. Signal shutdown immediately.
        thread::sleep(Duration::from_millis(20));
        shutdown.store(true, Ordering::SeqCst);

        // Worker must exit within a reasonable timeout.
        let joined = join_with_timeout(worker, Duration::from_millis(500));
        assert!(joined, "worker must honor shutdown flag when idle");
    }

    fn wait_for_hv(rx: &Receiver<ContinuousHV>, timeout: Duration) -> Option<ContinuousHV> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if let Ok(hv) = rx.try_recv() {
                return Some(hv);
            }
            thread::sleep(Duration::from_millis(10));
        }
        None
    }

    fn join_with_timeout(handle: JoinHandle<()>, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if handle.is_finished() {
                let _ = handle.join();
                return true;
            }
            thread::sleep(Duration::from_millis(10));
        }
        false
    }
}
