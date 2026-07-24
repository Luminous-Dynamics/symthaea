// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live phone-screen capture → cognitive-loop vision frames.
//!
//! Background thread grabs ADB screenshots from a connected Android device,
//! resizes them to the loop's configured vision frame dimensions, and queues
//! them for the perception phase. Mirrors the microphone capture pattern
//! (`audio_stream::MicCaptureHandle`): opt-in via
//! `CognitiveLoopService::start_phone_capture()`, torn down on Drop.
//!
//! Until 2026-07-15 the Pixel's real pixels never reached the cognitive loop
//! at all — `symthaea-phone-embodiment` captured into its own private
//! `VisionManifold` that only standalone examples ever drove (see
//! VISION_PROJECTION_REVIEW_2026-07-15.md, P1.1). This pump feeds the loop's
//! own VisionBridge instead; no second manifold.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use symthaea_phone_embodiment::adb::AdbDevice;

/// Configuration for live phone-screen capture.
#[derive(Debug, Clone)]
pub struct PhoneCaptureConfig {
    /// ADB device serial (e.g. `41201FDJG000UM`).
    pub serial: String,
    /// Output frame width — must match the loop's `vision_frame_width`.
    pub frame_width: u32,
    /// Output frame height — must match the loop's `vision_frame_height`.
    pub frame_height: u32,
    /// Output channels: 3 (RGB) when the loop's VisionBridge has
    /// `enable_color` (the default), 1 (luma) otherwise.
    pub channels: u8,
    /// Capture cadence. ADB screenshots take ~100-300ms each; the default
    /// 500ms keeps the ADB link comfortably under saturation while still
    /// refreshing several times per second of wall time.
    pub interval: Duration,
}

impl PhoneCaptureConfig {
    pub fn new(serial: impl Into<String>, frame_width: u32, frame_height: u32) -> Self {
        Self {
            serial: serial.into(),
            frame_width,
            frame_height,
            channels: 3,
            interval: Duration::from_millis(500),
        }
    }
}

/// Handle to a running phone-capture worker. Dropping stops the worker.
pub struct PhoneCaptureHandle {
    // Mutex because `mpsc::Receiver` is !Sync and CognitiveLoopService must
    // stay Sync (same reasoning as MicCaptureHandle). Drain is
    // single-threaded (perception phase) so the mutex is uncontended.
    frame_rx: Mutex<Receiver<Vec<u8>>>,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl PhoneCaptureHandle {
    /// Start the capture worker. Fails fast if the device is not connected
    /// and authorized, rather than spawning a thread that can never capture.
    pub fn start(config: PhoneCaptureConfig) -> Result<Self> {
        let adb = AdbDevice::new(config.serial.clone());
        if !adb.is_connected() {
            return Err(anyhow!(
                "phone {} not connected/authorized (check `adb devices`)",
                config.serial
            ));
        }
        // Prove the capture path works once, synchronously, so callers get a
        // real error instead of a silently idle stream.
        adb.screenshot()
            .map_err(|e| anyhow!("initial screenshot failed: {e}"))
            .context("phone capture startup probe")?;

        let shutdown = Arc::new(AtomicBool::new(false));
        // Capacity 1: only the freshest frame matters; the worker drops
        // frames when the loop hasn't drained yet.
        let (frame_tx, frame_rx) = sync_channel::<Vec<u8>>(1);

        let worker_shutdown = Arc::clone(&shutdown);
        let worker = std::thread::Builder::new()
            .name("phone-capture".into())
            .spawn(move || {
                let mut consecutive_failures: u32 = 0;
                while !worker_shutdown.load(Ordering::Relaxed) {
                    match adb.screenshot() {
                        Ok(png_bytes) => match image::load_from_memory(&png_bytes) {
                            Ok(img) => {
                                consecutive_failures = 0;
                                let resized = img.resize_exact(
                                    config.frame_width,
                                    config.frame_height,
                                    image::imageops::FilterType::Triangle,
                                );
                                let frame: Vec<u8> = if config.channels == 3 {
                                    resized.to_rgb8().into_raw()
                                } else {
                                    resized.to_luma8().into_raw()
                                };
                                match frame_tx.try_send(frame) {
                                    Ok(()) | Err(TrySendError::Full(_)) => {}
                                    Err(TrySendError::Disconnected(_)) => break,
                                }
                            }
                            Err(e) => {
                                consecutive_failures += 1;
                                tracing::warn!(error = %e, "phone frame decode failed");
                            }
                        },
                        Err(e) => {
                            consecutive_failures += 1;
                            // Log at warn only occasionally — a disconnected
                            // phone would otherwise spam every interval.
                            if consecutive_failures == 1 || consecutive_failures % 60 == 0 {
                                tracing::warn!(
                                    error = %e,
                                    consecutive_failures,
                                    "phone screenshot failed (device disconnected?)"
                                );
                            }
                        }
                    }
                    // Back off while failing so a dead ADB link doesn't spin.
                    let sleep = if consecutive_failures > 3 {
                        config.interval * 4
                    } else {
                        config.interval
                    };
                    std::thread::sleep(sleep);
                }
            })
            .context("spawn phone-capture worker")?;

        Ok(Self {
            frame_rx: Mutex::new(frame_rx),
            shutdown,
            worker: Some(worker),
        })
    }

    /// Take the freshest captured frame, if any arrived since the last drain.
    pub fn drain_latest(&self) -> Option<Vec<u8>> {
        let rx = self.frame_rx.lock().ok()?;
        let mut latest = None;
        while let Ok(frame) = rx.try_recv() {
            latest = Some(frame);
        }
        latest
    }
}

impl Drop for PhoneCaptureHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
