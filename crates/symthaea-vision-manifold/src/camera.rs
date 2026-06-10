// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Camera input source for live video processing.
//!
//! When the `camera` feature is enabled, provides `CameraSource` backed by
//! the `v4l` crate for real webcam capture via Video4Linux2. Without the
//! feature, provides a `MockCameraSource` for testing.
//!
//! `CameraManifold` combines a camera source with a `VisionManifold` for
//! convenient tick-based processing.
//!
//! ## NixOS requirements for `camera` feature
//!
//! Add to your `flake.nix` `buildInputs`:
//! ```nix
//! libclang.lib    # bindgen needs libclang
//! linuxHeaders    # v4l2-sys-mit needs linux/videodev2.h
//! ```
//! And to `shellHook`:
//! ```bash
//! export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"
//! export BINDGEN_EXTRA_CLANG_ARGS="-I${pkgs.linuxHeaders}/include"
//! ```

use crate::manifold::VisionManifold;
use crate::types::{VisionConfig, VisionTelemetry};

/// A frame captured from a camera source.
pub struct CapturedFrame {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub timestamp_us: u64,
}

// ── Real camera (v4l) ──────────────────────────────────────────────

#[cfg(feature = "camera")]
mod real_camera {
    use super::CapturedFrame;
    use v4l::Device;
    use v4l::FourCC;
    use v4l::buffer::Type;
    use v4l::io::mmap::Stream;
    use v4l::io::traits::CaptureStream;
    use v4l::video::Capture;

    /// Live camera source backed by V4L2 (Video4Linux2).
    ///
    /// Stores the `Device` to keep the file descriptor alive alongside the
    /// mmap `Stream`. The stream's internal buffers are memory-mapped and do
    /// not actually borrow from `Device`, but Rust's type system ties the
    /// `Stream` lifetime to the `Device` reference used during construction.
    /// We keep `_device` alive for soundness and use a `'static` lifetime on
    /// the stream (the mmap'd buffers are valid until explicitly unmapped in
    /// `Stream::drop`).
    pub struct CameraSource {
        _device: Device,
        stream: Stream<'static>,
        width: u32,
        height: u32,
        frame_count: u64,
    }

    impl CameraSource {
        /// Open camera at the given device path (e.g., `/dev/video0`).
        pub fn new(device_index: u32, width: u32, height: u32) -> Result<Self, String> {
            let path = format!("/dev/video{device_index}");
            let dev =
                Device::with_path(&path).map_err(|e| format!("Failed to open {path}: {e}"))?;

            // Request YUYV format (widely supported), we'll convert to grayscale
            let mut fmt = dev
                .format()
                .map_err(|e| format!("Failed to get format: {e}"))?;
            fmt.width = width;
            fmt.height = height;
            fmt.fourcc = FourCC::new(b"YUYV");
            dev.set_format(&fmt)
                .map_err(|e| format!("Failed to set format: {e}"))?;

            let actual = dev
                .format()
                .map_err(|e| format!("Failed to read format: {e}"))?;

            let stream = Stream::with_buffers(&dev, Type::VideoCapture, 4)
                .map_err(|e| format!("Failed to create stream: {e}"))?;

            // SAFETY: The mmap'd buffers inside Stream are backed by kernel memory-mapped
            // regions (via mmap syscall) that remain valid until Stream::drop calls munmap.
            // The Stream internally holds an Arc<Handle> (cloned from Device), so it does
            // not actually borrow from Device. We keep _device alive as an extra guarantee.
            // The 'a lifetime on Stream<'a> comes from Arena<'a> which holds the mmap'd
            // slices — these are effectively 'static since they're kernel-managed.
            #[allow(unsafe_code)]
            let stream: Stream<'static> = unsafe { std::mem::transmute(stream) };

            Ok(Self {
                _device: dev,
                stream,
                width: actual.width,
                height: actual.height,
                frame_count: 0,
            })
        }

        /// Capture the next frame as grayscale pixels (Y channel from YUYV).
        pub fn next_frame(&mut self) -> Result<CapturedFrame, String> {
            let (buf, _meta) = self
                .stream
                .next()
                .map_err(|e| format!("Failed to capture frame: {e}"))?;

            // YUYV: every 2 bytes = [Y, U/V], extract Y channel for grayscale
            let num_pixels = (self.width * self.height) as usize;
            let mut pixels = Vec::with_capacity(num_pixels);
            for i in 0..num_pixels {
                let byte_idx = i * 2; // Y is at even indices in YUYV
                if byte_idx < buf.len() {
                    pixels.push(buf[byte_idx]);
                } else {
                    pixels.push(0);
                }
            }

            self.frame_count += 1;
            let timestamp_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;

            Ok(CapturedFrame {
                pixels,
                width: self.width,
                height: self.height,
                channels: 1,
                timestamp_us,
            })
        }

        /// Capture the next frame as RGB pixels (converted from YUYV).
        ///
        /// Uses ITU-R BT.601 YCbCr→RGB conversion:
        /// ```text
        /// R = Y + 1.402 * (Cr - 128)
        /// G = Y - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
        /// B = Y + 1.772 * (Cb - 128)
        /// ```
        pub fn next_frame_rgb(&mut self) -> Result<CapturedFrame, String> {
            let (buf, _meta) = self
                .stream
                .next()
                .map_err(|e| format!("Failed to capture frame: {e}"))?;

            let num_pixels = (self.width * self.height) as usize;
            let mut pixels = Vec::with_capacity(num_pixels * 3);

            // YUYV packs 2 pixels per 4 bytes: [Y0, U, Y1, V]
            for pair in 0..(num_pixels / 2) {
                let base = pair * 4;
                let y0 = buf.get(base).copied().unwrap_or(0) as f32;
                let u = buf.get(base + 1).copied().unwrap_or(128) as f32 - 128.0;
                let y1 = buf.get(base + 2).copied().unwrap_or(0) as f32;
                let v = buf.get(base + 3).copied().unwrap_or(128) as f32 - 128.0;

                // Pixel 0
                let r0 = (y0 + 1.402 * v).clamp(0.0, 255.0) as u8;
                let g0 = (y0 - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                let b0 = (y0 + 1.772 * u).clamp(0.0, 255.0) as u8;
                pixels.extend_from_slice(&[r0, g0, b0]);

                // Pixel 1
                let r1 = (y1 + 1.402 * v).clamp(0.0, 255.0) as u8;
                let g1 = (y1 - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                let b1 = (y1 + 1.772 * u).clamp(0.0, 255.0) as u8;
                pixels.extend_from_slice(&[r1, g1, b1]);
            }

            // Handle odd-width case (last pixel)
            if num_pixels % 2 == 1 {
                let base = (num_pixels / 2) * 4;
                let y = buf.get(base).copied().unwrap_or(0) as f32;
                let u = buf.get(base + 1).copied().unwrap_or(128) as f32 - 128.0;
                let v = buf.get(base + 3).copied().unwrap_or(128) as f32 - 128.0;
                let r = (y + 1.402 * v).clamp(0.0, 255.0) as u8;
                let g = (y - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                let b = (y + 1.772 * u).clamp(0.0, 255.0) as u8;
                pixels.extend_from_slice(&[r, g, b]);
            }

            self.frame_count += 1;
            let timestamp_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;

            Ok(CapturedFrame {
                pixels,
                width: self.width,
                height: self.height,
                channels: 3,
                timestamp_us,
            })
        }

        pub fn frame_count(&self) -> u64 {
            self.frame_count
        }
    }
}

#[cfg(feature = "camera")]
pub use real_camera::CameraSource;

// ── Mock camera (always available) ────────────────────────────────────

/// Mock camera source for testing without hardware.
///
/// Generates synthetic frames (solid gray cycling through brightness levels).
pub struct MockCameraSource {
    width: u32,
    height: u32,
    frame_count: u64,
}

impl MockCameraSource {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frame_count: 0,
        }
    }

    /// Generate the next synthetic frame.
    pub fn next_frame(&mut self) -> Result<CapturedFrame, String> {
        let brightness = ((self.frame_count * 7) % 256) as u8;
        let num_pixels = (self.width * self.height) as usize;
        let pixels = vec![brightness; num_pixels];

        self.frame_count += 1;

        Ok(CapturedFrame {
            pixels,
            width: self.width,
            height: self.height,
            channels: 1,
            timestamp_us: self.frame_count * 33_333, // ~30fps timestamps
        })
    }

    /// Generate the next synthetic frame as RGB.
    ///
    /// Produces colored frames (varying R/G/B channels independently).
    pub fn next_frame_rgb(&mut self) -> Result<CapturedFrame, String> {
        let brightness = ((self.frame_count * 7) % 256) as u8;
        let num_pixels = (self.width * self.height) as usize;
        let mut pixels = Vec::with_capacity(num_pixels * 3);
        for _ in 0..num_pixels {
            pixels.push(brightness);
            pixels.push(brightness / 2);
            pixels.push(255u8.wrapping_sub(brightness));
        }

        self.frame_count += 1;

        Ok(CapturedFrame {
            pixels,
            width: self.width,
            height: self.height,
            channels: 3,
            timestamp_us: self.frame_count * 33_333,
        })
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

/// Convenience wrapper combining a camera source with a `VisionManifold`.
///
/// Provides a simple `tick()` interface for frame-by-frame processing.
/// Always uses `MockCameraSource` — use `CameraSource` directly with
/// `VisionManifold` for real camera input.
pub struct CameraManifold {
    source: MockCameraSource,
    manifold: VisionManifold,
    last_timestamp_us: u64,
}

impl CameraManifold {
    /// Create with a mock camera source (always available).
    pub fn with_mock(config: VisionConfig, width: u32, height: u32) -> Self {
        let manifold = VisionManifold::new(config, width, height);
        Self {
            source: MockCameraSource::new(width, height),
            manifold,
            last_timestamp_us: 0,
        }
    }

    /// Grab one frame, feed it to the manifold, return telemetry.
    pub fn tick(&mut self) -> Result<VisionTelemetry, String> {
        let frame = self.source.next_frame()?;

        // Compute dt from timestamps
        let dt = if self.last_timestamp_us > 0 && frame.timestamp_us > self.last_timestamp_us {
            (frame.timestamp_us - self.last_timestamp_us) as f32 / 1_000_000.0
        } else {
            0.033 // Default to ~30fps on first frame
        };
        self.last_timestamp_us = frame.timestamp_us;

        let telemetry = self.manifold.observe_frame(
            &frame.pixels,
            frame.width,
            frame.height,
            frame.channels,
            dt,
        );

        Ok(telemetry)
    }

    /// Access the underlying manifold.
    pub fn manifold(&self) -> &VisionManifold {
        &self.manifold
    }

    /// Mutable access to the underlying manifold.
    pub fn manifold_mut(&mut self) -> &mut VisionManifold {
        &mut self.manifold
    }

    /// Reset the manifold state.
    pub fn reset(&mut self) {
        self.manifold.reset();
        self.last_timestamp_us = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_camera_source() {
        let mut mock = MockCameraSource::new(64, 64);
        let frame = mock.next_frame().unwrap();
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.channels, 1);
        assert_eq!(frame.pixels.len(), 64 * 64);
        assert_eq!(mock.frame_count(), 1);
    }

    #[test]
    fn test_mock_camera_varying_brightness() {
        let mut mock = MockCameraSource::new(32, 32);
        let f1 = mock.next_frame().unwrap();
        let f2 = mock.next_frame().unwrap();
        // Different frames should have different brightness
        assert_ne!(f1.pixels[0], f2.pixels[0]);
    }

    #[test]
    fn test_camera_manifold_with_mock() {
        let cfg = VisionConfig::default();
        let mut cm = CameraManifold::with_mock(cfg, 64, 64);

        let tel = cm.tick().unwrap();
        assert_eq!(tel.frame_sequence, 1);

        let tel2 = cm.tick().unwrap();
        assert_eq!(tel2.frame_sequence, 2);
    }

    #[test]
    fn test_camera_manifold_reset() {
        let cfg = VisionConfig::default();
        let mut cm = CameraManifold::with_mock(cfg, 64, 64);

        cm.tick().unwrap();
        cm.tick().unwrap();
        assert_eq!(cm.manifold().frame_count(), 2);

        cm.reset();
        assert_eq!(cm.manifold().frame_count(), 0);
    }

    #[test]
    fn test_camera_manifold_dt_computation() {
        let cfg = VisionConfig::default();
        let mut cm = CameraManifold::with_mock(cfg, 64, 64);

        // First tick uses default dt
        cm.tick().unwrap();
        // Subsequent ticks compute dt from mock timestamps
        let tel = cm.tick().unwrap();
        assert!(tel.frame_sequence == 2);
    }

    #[test]
    fn test_mock_camera_rgb() {
        let mut mock = MockCameraSource::new(64, 64);
        let frame = mock.next_frame_rgb().unwrap();
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.channels, 3);
        assert_eq!(frame.pixels.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_mock_camera_rgb_varying_channels() {
        let mut mock = MockCameraSource::new(32, 32);
        let f1 = mock.next_frame_rgb().unwrap();
        let f2 = mock.next_frame_rgb().unwrap();
        // Different frames should have different color values
        assert_ne!(f1.pixels[0], f2.pixels[0]);
        // Second frame has brightness=7, so R=7, G=3, B=248 — channels differ
        assert_ne!(f2.pixels[0], f2.pixels[1]); // R != G
        assert_ne!(f2.pixels[1], f2.pixels[2]); // G != B
    }

    #[test]
    fn test_mock_camera_rgb_with_manifold() {
        let cfg = VisionConfig::default();
        let mut manifold = VisionManifold::new(cfg, 64, 64);
        let mut mock = MockCameraSource::new(64, 64);

        for _ in 0..5 {
            let frame = mock.next_frame_rgb().unwrap();
            let tel = manifold.observe_frame(
                &frame.pixels,
                frame.width,
                frame.height,
                frame.channels,
                0.033,
            );
            assert!(tel.prediction_error.is_finite());
        }
        assert_eq!(manifold.frame_count(), 5);
    }
}
