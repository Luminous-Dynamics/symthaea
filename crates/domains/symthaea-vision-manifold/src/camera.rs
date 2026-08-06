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

use serde::{Deserialize, Serialize};

use crate::manifold::VisionManifold;
use crate::types::{ManifoldState, VisionConfig, VisionTelemetry};

/// Current serialized `CameraManifold` schema.
pub const CAMERA_MANIFOLD_STATE_SCHEMA_VERSION: u32 = 2;

/// Default allocation ceiling for one synthetic RGB frame.
pub const DEFAULT_MOCK_CAMERA_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
/// Default maximum copied YUYV bytes accepted from one real-camera buffer.
pub const DEFAULT_REAL_CAMERA_MAX_RAW_FRAME_BYTES: usize = 64 * 1024 * 1024;
/// Default maximum decoded grayscale or RGB frame allocation.
pub const DEFAULT_REAL_CAMERA_MAX_DECODED_FRAME_BYTES: usize = 64 * 1024 * 1024;
/// Default number of V4L mmap buffers requested per capture stream.
pub const DEFAULT_REAL_CAMERA_MMAP_BUFFER_COUNT: u32 = 4;
/// Default aggregate ceiling for the requested V4L mmap buffer set.
pub const DEFAULT_REAL_CAMERA_MAX_MMAP_BYTES: usize = 256 * 1024 * 1024;

fn default_mock_camera_max_frame_bytes() -> usize {
    DEFAULT_MOCK_CAMERA_MAX_FRAME_BYTES
}

/// Serializable mock-camera clock and geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MockCameraSourceState {
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    /// Maximum bytes permitted for one generated frame.
    #[serde(default = "default_mock_camera_max_frame_bytes")]
    pub max_frame_bytes: usize,
}

/// Serializable camera/manifold recovery point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraManifoldState {
    pub schema_version: u32,
    pub source: MockCameraSourceState,
    pub manifold: ManifoldState,
    pub last_timestamp_us: Option<u64>,
}

/// Explicit allocation and buffering policy for real V4L capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CameraCapturePolicy {
    /// Number of mmap buffers requested from V4L.
    pub mmap_buffer_count: u32,
    /// Maximum owned bytes copied from one raw YUYV capture buffer.
    pub max_raw_frame_bytes: usize,
    /// Maximum bytes allocated for one decoded frame.
    pub max_decoded_frame_bytes: usize,
    /// Maximum aggregate bytes represented by the requested mmap buffer set.
    pub max_mmap_bytes: usize,
}

impl Default for CameraCapturePolicy {
    fn default() -> Self {
        Self {
            mmap_buffer_count: DEFAULT_REAL_CAMERA_MMAP_BUFFER_COUNT,
            max_raw_frame_bytes: DEFAULT_REAL_CAMERA_MAX_RAW_FRAME_BYTES,
            max_decoded_frame_bytes: DEFAULT_REAL_CAMERA_MAX_DECODED_FRAME_BYTES,
            max_mmap_bytes: DEFAULT_REAL_CAMERA_MAX_MMAP_BYTES,
        }
    }
}

impl CameraCapturePolicy {
    /// Validate policy ranges independent of a particular negotiated format.
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=32).contains(&self.mmap_buffer_count) {
            return Err(format!(
                "camera mmap buffer count must be in 1..=32, got {}",
                self.mmap_buffer_count
            ));
        }
        if self.max_raw_frame_bytes == 0 {
            return Err("camera raw-frame budget must be non-zero".to_string());
        }
        if self.max_decoded_frame_bytes == 0 {
            return Err("camera decoded-frame budget must be non-zero".to_string());
        }
        if self.max_mmap_bytes == 0 {
            return Err("camera aggregate mmap budget must be non-zero".to_string());
        }
        Ok(())
    }

    /// Validate requested or negotiated geometry against raw and decoded bounds.
    pub fn validate_geometry(&self, width: u32, height: u32) -> Result<(), String> {
        self.validate()?;
        if width == 0 || height == 0 {
            return Err(format!(
                "camera geometry must be non-zero, got {width}x{height}"
            ));
        }
        let raw_bytes = yuyv_min_stride(width)?
            .checked_mul(height as usize)
            .ok_or_else(|| "camera raw-frame geometry overflow".to_string())?;
        let decoded_bytes = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| "camera decoded-frame geometry overflow".to_string())?;
        if raw_bytes > isize::MAX as usize || decoded_bytes > isize::MAX as usize {
            return Err("camera geometry exceeds Vec allocation limits".to_string());
        }
        if raw_bytes > self.max_raw_frame_bytes {
            return Err(format!(
                "camera raw YUYV frame exceeds allocation budget: {raw_bytes} > {} bytes",
                self.max_raw_frame_bytes
            ));
        }
        let aggregate_raw = raw_bytes
            .checked_mul(self.mmap_buffer_count as usize)
            .ok_or_else(|| "camera aggregate mmap geometry overflow".to_string())?;
        if aggregate_raw > self.max_mmap_bytes {
            return Err(format!(
                "camera mmap buffer set exceeds aggregate budget: {aggregate_raw} > {} bytes",
                self.max_mmap_bytes
            ));
        }
        if decoded_bytes > self.max_decoded_frame_bytes {
            return Err(format!(
                "camera decoded RGB frame exceeds allocation budget: {decoded_bytes} > {} bytes",
                self.max_decoded_frame_bytes
            ));
        }
        Ok(())
    }
}

/// A frame captured from a camera source.
pub struct CapturedFrame {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub timestamp_us: u64,
}

fn yuyv_min_stride(width: u32) -> Result<usize, String> {
    if width == 0 {
        return Err("YUYV width must be non-zero".to_string());
    }
    let pairs = (width as usize)
        .checked_add(1)
        .ok_or_else(|| "YUYV width overflow".to_string())?
        / 2;
    pairs
        .checked_mul(4)
        .ok_or_else(|| "YUYV row-stride overflow".to_string())
}

// The YUYV decode/validate helpers below (through decode_yuyv_rgb) have no caller under
// clippy's default lib-only build: their real caller is the `camera` hardware-capture path
// (gated `#[cfg(feature = "camera")]` below), and their test coverage lives in the general
// `#[cfg(test)]` module further down -- both invisible to a plain `cargo clippy` lib check.
// Genuinely used, not dead; suppressing rather than deleting real, tested logic.
#[allow(dead_code)]
fn validate_yuyv_stride(
    width: u32,
    height: u32,
    stride: usize,
    buffer_len: usize,
) -> Result<(), String> {
    if height == 0 {
        return Err("YUYV height must be non-zero".to_string());
    }
    let minimum = yuyv_min_stride(width)?;
    if stride < minimum {
        return Err(format!(
            "YUYV stride is too short: {stride}, minimum {minimum}"
        ));
    }
    let required = stride
        .checked_mul(height as usize)
        .ok_or_else(|| "YUYV buffer geometry overflow".to_string())?;
    if buffer_len < required {
        return Err(format!(
            "YUYV buffer is too short: {buffer_len}, requires {required} bytes"
        ));
    }
    Ok(())
}

/// Infer a uniform scanline stride from an owned buffer.
#[allow(dead_code)]
fn infer_yuyv_stride(buffer_len: usize, width: u32, height: u32) -> Result<usize, String> {
    if height == 0 {
        return Err("YUYV height must be non-zero".to_string());
    }
    if !buffer_len.is_multiple_of(height as usize) {
        return Err(format!(
            "YUYV buffer length {buffer_len} is not divisible by height {height}"
        ));
    }
    let stride = buffer_len / height as usize;
    validate_yuyv_stride(width, height, stride, buffer_len)?;
    Ok(stride)
}

#[allow(dead_code)]
fn decode_yuyv_gray_with_stride(
    buffer: &[u8],
    width: u32,
    height: u32,
    stride: usize,
) -> Result<Vec<u8>, String> {
    validate_yuyv_stride(width, height, stride, buffer.len())?;
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "YUYV image geometry overflow".to_string())?;
    let mut pixels = Vec::with_capacity(pixel_count);
    for row in 0..height as usize {
        let row_start = row
            .checked_mul(stride)
            .ok_or_else(|| "YUYV row offset overflow".to_string())?;
        for x in 0..width as usize {
            let offset = row_start + (x / 2) * 4 + (x % 2) * 2;
            let y = buffer
                .get(offset)
                .copied()
                .ok_or_else(|| format!("YUYV row {row} is truncated at pixel {x}"))?;
            pixels.push(y);
        }
    }
    Ok(pixels)
}

#[allow(dead_code)]
fn decode_yuyv_gray(buffer: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let stride = infer_yuyv_stride(buffer.len(), width, height)?;
    decode_yuyv_gray_with_stride(buffer, width, height, stride)
}

#[allow(dead_code)]
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let y = y as f32;
    let u = u as f32 - 128.0;
    let v = v as f32 - 128.0;
    [
        (y + 1.402 * v).clamp(0.0, 255.0) as u8,
        (y - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8,
        (y + 1.772 * u).clamp(0.0, 255.0) as u8,
    ]
}

#[allow(dead_code)]
fn decode_yuyv_rgb_with_stride(
    buffer: &[u8],
    width: u32,
    height: u32,
    stride: usize,
) -> Result<Vec<u8>, String> {
    validate_yuyv_stride(width, height, stride, buffer.len())?;
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "YUYV image geometry overflow".to_string())?;
    let capacity = pixel_count
        .checked_mul(3)
        .ok_or_else(|| "RGB output geometry overflow".to_string())?;
    let mut pixels = Vec::with_capacity(capacity);
    for row in 0..height as usize {
        let row_start = row
            .checked_mul(stride)
            .ok_or_else(|| "YUYV row offset overflow".to_string())?;
        for x in 0..width as usize {
            let pair = row_start + (x / 2) * 4;
            let y_offset = pair + (x % 2) * 2;
            let y = *buffer
                .get(y_offset)
                .ok_or_else(|| format!("YUYV row {row} is truncated at pixel {x}"))?;
            let u = *buffer
                .get(pair + 1)
                .ok_or_else(|| format!("YUYV row {row} is missing chroma U"))?;
            let v = *buffer
                .get(pair + 3)
                .ok_or_else(|| format!("YUYV row {row} is missing chroma V"))?;
            pixels.extend_from_slice(&yuv_to_rgb(y, u, v));
        }
    }
    Ok(pixels)
}

#[allow(dead_code)]
fn decode_yuyv_rgb(buffer: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let stride = infer_yuyv_stride(buffer.len(), width, height)?;
    decode_yuyv_rgb_with_stride(buffer, width, height, stride)
}

// ── Real camera (v4l) ──────────────────────────────────────────────

#[cfg(feature = "camera")]
mod real_camera {
    use super::{
        CameraCapturePolicy, CapturedFrame, decode_yuyv_gray_with_stride,
        decode_yuyv_rgb_with_stride, validate_yuyv_stride, yuyv_min_stride,
    };
    use v4l::Device;
    use v4l::FourCC;
    use v4l::buffer::Type;
    use v4l::io::mmap::Stream;
    use v4l::io::traits::CaptureStream;
    use v4l::video::Capture;

    /// Live camera source backed by V4L2 (Video4Linux2).
    ///
    /// The source stores only the device path and negotiated geometry. A fresh
    /// mmap stream is opened inside each capture call, keeping the stream's
    /// borrow tied to a local `Device` and avoiding self-referential lifetime
    /// tricks. This is deliberately a correctness-first implementation; callers
    /// that need persistent zero-copy streaming can wrap capture in a dedicated
    /// worker thread that owns both values in one stack frame.
    pub struct CameraSource {
        device_path: String,
        width: u32,
        height: u32,
        frame_count: u64,
        clock_origin: std::time::Instant,
        policy: CameraCapturePolicy,
    }

    impl CameraSource {
        fn validate_negotiated_format(
            actual: &v4l::Format,
            policy: CameraCapturePolicy,
        ) -> Result<usize, String> {
            if actual.fourcc != FourCC::new(b"YUYV") {
                return Err(format!(
                    "camera did not negotiate YUYV; actual format is {:?}",
                    actual.fourcc
                ));
            }
            policy.validate_geometry(actual.width, actual.height)?;
            if actual.size as usize > policy.max_raw_frame_bytes {
                return Err(format!(
                    "negotiated V4L frame buffer exceeds allocation budget: {} > {} bytes",
                    actual.size, policy.max_raw_frame_bytes
                ));
            }
            let minimum = yuyv_min_stride(actual.width)?;
            let stride = if actual.stride == 0 {
                minimum
            } else {
                actual.stride as usize
            };
            if stride < minimum {
                return Err(format!(
                    "negotiated YUYV stride is too short: {stride}, minimum {minimum}"
                ));
            }
            let required = stride
                .checked_mul(actual.height as usize)
                .ok_or_else(|| "negotiated YUYV buffer geometry overflow".to_string())?;
            if required > policy.max_raw_frame_bytes {
                return Err(format!(
                    "negotiated YUYV stride exceeds allocation budget: {required} > {} bytes",
                    policy.max_raw_frame_bytes
                ));
            }
            let per_buffer = required.max(actual.size as usize);
            let aggregate = per_buffer
                .checked_mul(policy.mmap_buffer_count as usize)
                .ok_or_else(|| "negotiated aggregate mmap allocation overflow".to_string())?;
            if aggregate > policy.max_mmap_bytes {
                return Err(format!(
                    "negotiated mmap buffer set exceeds aggregate budget: {aggregate} > {} bytes",
                    policy.max_mmap_bytes
                ));
            }
            Ok(stride)
        }

        /// Open camera at the given device path (e.g., `/dev/video0`).
        pub fn new(device_index: u32, width: u32, height: u32) -> Result<Self, String> {
            Self::with_policy(device_index, width, height, CameraCapturePolicy::default())
        }

        /// Open a camera under explicit mmap and frame-allocation limits.
        pub fn with_policy(
            device_index: u32,
            width: u32,
            height: u32,
            policy: CameraCapturePolicy,
        ) -> Result<Self, String> {
            policy.validate_geometry(width, height)?;
            let path = format!("/dev/video{device_index}");
            let dev =
                Device::with_path(&path).map_err(|e| format!("Failed to open {path}: {e}"))?;

            // Request YUYV format (widely supported), we'll convert to grayscale.
            let mut fmt = dev
                .format()
                .map_err(|e| format!("Failed to get format: {e}"))?;
            fmt.width = width;
            fmt.height = height;
            fmt.fourcc = FourCC::new(b"YUYV");
            let actual = dev
                .set_format(&fmt)
                .map_err(|e| format!("Failed to set format: {e}"))?;
            Self::validate_negotiated_format(&actual, policy)?;

            // Verify that the negotiated device can create the requested mmap
            // buffer set without retaining a self-referential stream.
            Stream::with_buffers(&dev, Type::VideoCapture, policy.mmap_buffer_count)
                .map_err(|e| format!("Failed to create stream: {e}"))?;

            Ok(Self {
                device_path: path,
                width: actual.width,
                height: actual.height,
                frame_count: 0,
                clock_origin: std::time::Instant::now(),
                policy,
            })
        }

        /// Capture one owned YUYV buffer while the `Device` and mmap stream are
        /// both alive in this stack frame.
        fn capture_yuyv(&self) -> Result<(Vec<u8>, usize), String> {
            let dev = Device::with_path(&self.device_path)
                .map_err(|e| format!("Failed to open {}: {e}", self.device_path))?;
            let mut format = dev
                .format()
                .map_err(|e| format!("Failed to get format: {e}"))?;
            format.width = self.width;
            format.height = self.height;
            format.fourcc = FourCC::new(b"YUYV");
            let actual = dev
                .set_format(&format)
                .map_err(|e| format!("Failed to set format: {e}"))?;
            if actual.width != self.width || actual.height != self.height {
                return Err(format!(
                    "camera geometry changed during capture: negotiated={}x{}, expected={}x{}",
                    actual.width, actual.height, self.width, self.height
                ));
            }
            let stride = Self::validate_negotiated_format(&actual, self.policy)?;
            let mut stream =
                Stream::with_buffers(&dev, Type::VideoCapture, self.policy.mmap_buffer_count)
                    .map_err(|e| format!("Failed to create stream: {e}"))?;
            let (buf, _meta) = stream
                .next()
                .map_err(|e| format!("Failed to capture frame: {e}"))?;
            if buf.len() > self.policy.max_raw_frame_bytes {
                return Err(format!(
                    "captured YUYV buffer exceeds allocation budget: {} > {} bytes",
                    buf.len(),
                    self.policy.max_raw_frame_bytes
                ));
            }
            validate_yuyv_stride(self.width, self.height, stride, buf.len())?;
            Ok((buf.to_vec(), stride))
        }

        /// Capture the next frame as grayscale pixels (Y channel from YUYV).
        pub fn next_frame(&mut self) -> Result<CapturedFrame, String> {
            let (buf, stride) = self.capture_yuyv()?;

            // Decode with the negotiated bytes-per-line value so trailing V4L
            // storage and scanline padding never become image luminance.
            let pixels = decode_yuyv_gray_with_stride(&buf, self.width, self.height, stride)?;

            self.frame_count = self
                .frame_count
                .checked_add(1)
                .ok_or_else(|| "camera frame counter overflow".to_string())?;
            let timestamp_us = u64::try_from(self.clock_origin.elapsed().as_micros())
                .map_err(|_| "camera monotonic timestamp overflow".to_string())?;

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
            let (buf, stride) = self.capture_yuyv()?;

            // Decode with the negotiated bytes-per-line value so padding cannot
            // shift chroma or contaminate the following scanline.
            let pixels = decode_yuyv_rgb_with_stride(&buf, self.width, self.height, stride)?;

            self.frame_count = self
                .frame_count
                .checked_add(1)
                .ok_or_else(|| "camera frame counter overflow".to_string())?;
            let timestamp_us = u64::try_from(self.clock_origin.elapsed().as_micros())
                .map_err(|_| "camera monotonic timestamp overflow".to_string())?;

            Ok(CapturedFrame {
                pixels,
                width: self.width,
                height: self.height,
                channels: 3,
                timestamp_us,
            })
        }

        pub fn policy(&self) -> CameraCapturePolicy {
            self.policy
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
    max_frame_bytes: usize,
}

impl MockCameraSource {
    pub fn new(width: u32, height: u32) -> Self {
        Self::try_new(width, height)
            .unwrap_or_else(|error| panic!("Invalid MockCameraSource construction: {error}"))
    }

    /// Construct a source under the default one-frame allocation budget.
    pub fn try_new(width: u32, height: u32) -> Result<Self, String> {
        Self::try_with_budget(width, height, DEFAULT_MOCK_CAMERA_MAX_FRAME_BYTES)
    }

    /// Construct a source under an explicit worst-case RGB frame budget.
    pub fn try_with_budget(
        width: u32,
        height: u32,
        max_frame_bytes: usize,
    ) -> Result<Self, String> {
        Self::validate_geometry(width, height, max_frame_bytes)?;
        Ok(Self {
            width,
            height,
            frame_count: 0,
            max_frame_bytes,
        })
    }

    fn validate_geometry(width: u32, height: u32, max_frame_bytes: usize) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err(format!(
                "mock camera geometry must be non-zero, got {width}x{height}"
            ));
        }
        if max_frame_bytes == 0 {
            return Err("mock camera frame budget must be non-zero".to_string());
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "mock camera geometry overflows addressable memory".to_string())?;
        let rgb_bytes = pixels
            .checked_mul(3)
            .ok_or_else(|| "mock camera RGB geometry overflows addressable memory".to_string())?;
        if rgb_bytes > isize::MAX as usize {
            return Err("mock camera geometry exceeds Vec allocation limits".to_string());
        }
        if rgb_bytes > max_frame_bytes {
            return Err(format!(
                "mock camera RGB frame exceeds allocation budget: {rgb_bytes} > {max_frame_bytes} bytes"
            ));
        }
        Ok(())
    }

    fn frame_bytes(&self, channels: usize) -> Result<usize, String> {
        let bytes = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|pixels| pixels.checked_mul(channels))
            .ok_or_else(|| "mock camera frame geometry overflow".to_string())?;
        if bytes > self.max_frame_bytes {
            return Err(format!(
                "mock camera frame exceeds allocation budget: {bytes} > {} bytes",
                self.max_frame_bytes
            ));
        }
        Ok(bytes)
    }

    /// Generate the next synthetic frame.
    pub fn next_frame(&mut self) -> Result<CapturedFrame, String> {
        let brightness = (self.frame_count.wrapping_mul(7) % 256) as u8;
        let num_pixels = self.frame_bytes(1)?;
        let pixels = vec![brightness; num_pixels];

        self.frame_count = self
            .frame_count
            .checked_add(1)
            .ok_or_else(|| "mock camera frame counter overflow".to_string())?;
        let timestamp_us = self
            .frame_count
            .checked_mul(33_333)
            .ok_or_else(|| "mock camera timestamp overflow".to_string())?;

        Ok(CapturedFrame {
            pixels,
            width: self.width,
            height: self.height,
            channels: 1,
            timestamp_us, // ~30fps timestamps
        })
    }

    /// Generate the next synthetic frame as RGB.
    ///
    /// Produces colored frames (varying R/G/B channels independently).
    pub fn next_frame_rgb(&mut self) -> Result<CapturedFrame, String> {
        let brightness = (self.frame_count.wrapping_mul(7) % 256) as u8;
        let capacity = self.frame_bytes(3)?;
        let num_pixels = capacity / 3;
        let mut pixels = Vec::with_capacity(capacity);
        for _ in 0..num_pixels {
            pixels.push(brightness);
            pixels.push(brightness / 2);
            pixels.push(255u8.wrapping_sub(brightness));
        }

        self.frame_count = self
            .frame_count
            .checked_add(1)
            .ok_or_else(|| "mock camera frame counter overflow".to_string())?;
        let timestamp_us = self
            .frame_count
            .checked_mul(33_333)
            .ok_or_else(|| "mock camera timestamp overflow".to_string())?;

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

    /// Snapshot deterministic source geometry and sequence state.
    pub fn save_state(&self) -> MockCameraSourceState {
        MockCameraSourceState {
            width: self.width,
            height: self.height,
            frame_count: self.frame_count,
            max_frame_bytes: self.max_frame_bytes,
        }
    }

    fn validate_state(state: &MockCameraSourceState) -> Result<(), String> {
        Self::validate_geometry(state.width, state.height, state.max_frame_bytes)?;
        if state.frame_count >= u64::MAX / 33_333 {
            return Err("mock camera frame count cannot produce another timestamp".to_string());
        }
        Ok(())
    }

    /// Restore deterministic source state after validation.
    pub fn load_state(&mut self, state: &MockCameraSourceState) -> Result<(), String> {
        Self::validate_state(state)?;
        self.width = state.width;
        self.height = state.height;
        self.frame_count = state.frame_count;
        self.max_frame_bytes = state.max_frame_bytes;
        Ok(())
    }

    /// Restart the synthetic sequence from its first frame.
    pub fn reset(&mut self) {
        self.frame_count = 0;
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
    last_timestamp_us: Option<u64>,
}

impl CameraManifold {
    /// Create with a mock camera source (always available).
    pub fn with_mock(config: VisionConfig, width: u32, height: u32) -> Self {
        Self::try_with_mock(config, width, height)
            .unwrap_or_else(|error| panic!("Invalid CameraManifold construction: {error}"))
    }

    /// Create a mock camera/manifold pair without panicking on invalid inputs.
    pub fn try_with_mock(config: VisionConfig, width: u32, height: u32) -> Result<Self, String> {
        let manifold = VisionManifold::try_new(config, width, height)?;
        Ok(Self {
            source: MockCameraSource::try_new(width, height)?,
            manifold,
            last_timestamp_us: None,
        })
    }

    /// Grab one frame, feed it to the manifold, return telemetry.
    pub fn tick(&mut self) -> Result<VisionTelemetry, String> {
        let frame = self.source.next_frame()?;
        self.process_captured_frame(frame)
    }

    /// Validate and process an externally captured frame under a monotonic clock
    /// contract. The timestamp is committed only after the manifold accepts the
    /// frame, so malformed input cannot consume time or change the next `dt`.
    pub fn process_captured_frame(
        &mut self,
        frame: CapturedFrame,
    ) -> Result<VisionTelemetry, String> {
        let dt = match self.last_timestamp_us {
            Some(previous) => {
                if frame.timestamp_us <= previous {
                    return Err(format!(
                        "camera timestamp must increase strictly: previous={previous}, current={}",
                        frame.timestamp_us
                    ));
                }
                (frame.timestamp_us - previous) as f32 / 1_000_000.0
            }
            None => 0.033, // Default to ~30fps on the first accepted frame.
        };

        let telemetry = self.manifold.observe_frame_checked(
            &frame.pixels,
            frame.width,
            frame.height,
            frame.channels,
            dt,
        )?;
        self.last_timestamp_us = Some(frame.timestamp_us);
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

    /// Snapshot the complete mock-camera and manifold recovery point.
    pub fn save_state(&self) -> CameraManifoldState {
        CameraManifoldState {
            schema_version: CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            source: self.source.save_state(),
            manifold: self.manifold.save_state(),
            last_timestamp_us: self.last_timestamp_us,
        }
    }

    /// Serialize camera sequence and manifold state into an integrity envelope.
    pub fn save_checkpoint_bytes(&self) -> Result<Vec<u8>, String> {
        self.save_checkpoint_bytes_with_limit(
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    pub fn save_checkpoint_bytes_with_limit(
        &self,
        max_payload_bytes: usize,
    ) -> Result<Vec<u8>, String> {
        crate::checkpoint::encode_checkpoint(
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            max_payload_bytes,
        )
    }

    /// Serialize the complete camera stack into a caller-authenticated envelope.
    pub fn save_authenticated_checkpoint_bytes<S>(
        &self,
        max_tag_bytes: usize,
        sign: S,
    ) -> Result<Vec<u8>, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    {
        crate::checkpoint::encode_authenticated_checkpoint(
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            sign,
        )
    }

    /// Atomically persist a caller-authenticated camera checkpoint.
    pub fn save_authenticated_checkpoint_file<S>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        sign: S,
    ) -> Result<crate::checkpoint::CheckpointWriteReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file(
            path,
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            sign,
        )
    }

    /// Atomically persist camera sequence and manifold state to disk.
    pub fn save_checkpoint_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        crate::checkpoint::save_checkpoint_file(
            path,
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    /// Persist a camera checkpoint while retaining the previous verified generation.
    pub fn save_checkpoint_file_recoverable(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        crate::checkpoint::save_checkpoint_file_recoverable(
            path,
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    /// Persist a bounded retained generation set and return complete write evidence.
    pub fn save_checkpoint_file_with_retention_report(
        &self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String> {
        crate::checkpoint::save_checkpoint_file_with_retention_report(
            path,
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            policy,
        )
    }

    /// Persist retained generations under one cross-process writer lease.
    pub fn save_checkpoint_file_with_retention_locked_report(
        &self,
        path: impl AsRef<std::path::Path>,
        retention: crate::checkpoint::CheckpointRetentionPolicy,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String> {
        crate::checkpoint::save_checkpoint_file_with_retention_locked_report(
            path,
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            retention,
            lock_policy,
        )
    }

    /// Persist authenticated retained generations and return complete write evidence.
    pub fn save_authenticated_checkpoint_file_with_retention_report<S, V>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
        sign: S,
        verify: V,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file_with_retention_report(
            path,
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            policy,
            sign,
            verify,
        )
    }

    /// Persist authenticated retained generations under one writer lease.
    pub fn save_authenticated_checkpoint_file_with_retention_locked_report<S, V>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        retention: crate::checkpoint::CheckpointRetentionPolicy,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
        sign: S,
        verify: V,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file_with_retention_locked_report(
            path,
            "symthaea-vision-camera",
            CAMERA_MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            retention,
            lock_policy,
            sign,
            verify,
        )
    }

    /// Inspect the primary and retained checkpoint generations for this stack.
    pub fn inspect_checkpoint_generations(
        &self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<Vec<crate::checkpoint::CheckpointGenerationInspection>, String> {
        crate::checkpoint::inspect_checkpoint_generations(
            path,
            "symthaea-vision-camera",
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            policy,
        )
    }

    /// Remove generations beyond a reduced retention bound.
    pub fn prune_checkpoint_generations(
        &self,
        path: impl AsRef<std::path::Path>,
        keep_previous_generations: usize,
    ) -> Result<crate::checkpoint::CheckpointPruneReport, String> {
        crate::checkpoint::prune_checkpoint_generations(path, keep_previous_generations)
    }

    /// Remove old generations while coordinating with retained checkpoint writers.
    pub fn prune_checkpoint_generations_locked(
        &self,
        path: impl AsRef<std::path::Path>,
        keep_previous_generations: usize,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
    ) -> Result<crate::checkpoint::CheckpointPruneReport, String> {
        crate::checkpoint::prune_checkpoint_generations_locked(
            path,
            keep_previous_generations,
            lock_policy,
        )
    }

    /// Authenticate, deserialize, validate, and atomically restore camera bytes.
    pub fn load_authenticated_checkpoint_bytes<V>(
        &mut self,
        encoded: &[u8],
        max_tag_bytes: usize,
        verify: V,
    ) -> Result<(), String>
    where
        V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state): (u32, CameraManifoldState) =
            crate::checkpoint::decode_authenticated_checkpoint(
                encoded,
                "symthaea-vision-camera",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                verify,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "authenticated camera checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Read, authenticate, validate, and atomically restore a camera checkpoint.
    pub fn load_authenticated_checkpoint_file<V>(
        &mut self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        verify: V,
    ) -> Result<(), String>
    where
        V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state): (u32, CameraManifoldState) =
            crate::checkpoint::load_authenticated_checkpoint_file(
                path,
                "symthaea-vision-camera",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                verify,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "authenticated camera checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Read, verify, and atomically restore a camera checkpoint file.
    pub fn load_checkpoint_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        let (payload_schema, state): (u32, CameraManifoldState) =
            crate::checkpoint::load_checkpoint_file(
                path,
                "symthaea-vision-camera",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "camera checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Restore the primary checkpoint or its previous verified generation.
    pub fn load_checkpoint_file_recoverable(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<crate::checkpoint::CheckpointRecoverySource, String> {
        let (payload_schema, state, source): (
            u32,
            CameraManifoldState,
            crate::checkpoint::CheckpointRecoverySource,
        ) = crate::checkpoint::load_checkpoint_file_recoverable(
            path,
            "symthaea-vision-camera",
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "camera checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)?;
        Ok(source)
    }

    /// Restore the newest semantically compatible retained camera generation.
    pub fn load_checkpoint_file_with_retention_audited(
        &mut self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<
        crate::checkpoint::CheckpointSemanticRecoveryReport,
        crate::checkpoint::CheckpointSemanticRecoveryFailure,
    > {
        let (payload_schema, state, report) =
            crate::checkpoint::load_checkpoint_file_with_retention_audited_detailed(
                path,
                "symthaea-vision-camera",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                policy,
                |schema, candidate: &CameraManifoldState| {
                    if schema != candidate.schema_version {
                        return Err(format!(
                            "camera checkpoint envelope/payload schema mismatch: envelope={schema}, payload={}",
                            candidate.schema_version
                        ));
                    }
                    self.validate_checkpoint_state(candidate)
                },
            )?;
        if payload_schema != state.schema_version {
            return Err(crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected camera checkpoint schema mismatch: envelope={payload_schema}, payload={}",
                    state.schema_version
                )),
            });
        }
        self.load_state(&state).map_err(|error| {
            crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected camera checkpoint failed final atomic restore: {error}"
                )),
            }
        })?;
        Ok(report)
    }

    /// Restore the newest authenticated and semantically compatible retained generation.
    pub fn load_authenticated_checkpoint_file_with_retention_audited<V>(
        &mut self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
        verify: V,
    ) -> Result<
        crate::checkpoint::CheckpointSemanticRecoveryReport,
        crate::checkpoint::CheckpointSemanticRecoveryFailure,
    >
    where
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state, report) =
            crate::checkpoint::load_authenticated_checkpoint_file_with_retention_audited_detailed(
                path,
                "symthaea-vision-camera",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                policy,
                verify,
                |schema, candidate: &CameraManifoldState| {
                    if schema != candidate.schema_version {
                        return Err(format!(
                            "camera checkpoint envelope/payload schema mismatch: envelope={schema}, payload={}",
                            candidate.schema_version
                        ));
                    }
                    self.validate_checkpoint_state(candidate)
                },
            )?;
        if payload_schema != state.schema_version {
            return Err(crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected authenticated camera checkpoint schema mismatch: envelope={payload_schema}, payload={}",
                    state.schema_version
                )),
            });
        }
        self.load_state(&state).map_err(|error| {
            crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected authenticated camera checkpoint failed final atomic restore: {error}"
                )),
            }
        })?;
        Ok(report)
    }

    /// Validate and atomically restore a camera integrity envelope.
    pub fn load_checkpoint_bytes(&mut self, encoded: &[u8]) -> Result<(), String> {
        self.load_checkpoint_bytes_with_limit(
            encoded,
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    pub fn load_checkpoint_bytes_with_limit(
        &mut self,
        encoded: &[u8],
        max_payload_bytes: usize,
    ) -> Result<(), String> {
        let (payload_schema, state): (u32, CameraManifoldState) =
            crate::checkpoint::decode_checkpoint(
                encoded,
                "symthaea-vision-camera",
                max_payload_bytes,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "camera checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    fn checkpoint_validation_probe(&self) -> Result<Self, String> {
        let mut probe = Self::try_with_mock(
            self.manifold.config().clone(),
            self.source.width,
            self.source.height,
        )?;
        probe.load_state(&self.save_state())?;
        Ok(probe)
    }

    /// Validate a camera checkpoint against the current source and manifold
    /// topology without mutating the live camera sequence.
    pub fn validate_checkpoint_state(&self, state: &CameraManifoldState) -> Result<(), String> {
        let mut probe = self.checkpoint_validation_probe()?;
        probe.load_state(state)
    }

    /// Atomically restore camera sequence, timing, and manifold state.
    pub fn load_state(&mut self, state: &CameraManifoldState) -> Result<(), String> {
        if state.schema_version == 0 || state.schema_version > CAMERA_MANIFOLD_STATE_SCHEMA_VERSION
        {
            return Err(format!(
                "unsupported camera checkpoint schema: saved={}, supported<= {}",
                state.schema_version, CAMERA_MANIFOLD_STATE_SCHEMA_VERSION
            ));
        }
        MockCameraSource::validate_state(&state.source)?;
        if state.source.width != self.source.width || state.source.height != self.source.height {
            return Err(format!(
                "camera source geometry mismatch: saved={}x{}, current={}x{}",
                state.source.width, state.source.height, self.source.width, self.source.height
            ));
        }

        // `VisionManifold::load_state` validates the complete payload before
        // mutating live state. Source assignment cannot fail after this preflight.
        self.manifold.load_state(&state.manifold)?;
        self.source.load_state(&state.source)?;
        self.last_timestamp_us = state.last_timestamp_us;
        Ok(())
    }

    /// Reset perception, accepted-camera time, and the synthetic source sequence.
    pub fn reset(&mut self) {
        self.manifold.reset();
        self.source.reset();
        self.last_timestamp_us = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_camera_policy_bounds_buffers_and_negotiated_geometry() {
        let policy = CameraCapturePolicy::default();
        assert!(policy.validate_geometry(640, 480).is_ok());

        let mut invalid = policy;
        invalid.mmap_buffer_count = 0;
        assert!(invalid.validate().is_err());
        invalid.mmap_buffer_count = 33;
        assert!(invalid.validate().is_err());

        let mut raw_bounded = policy;
        raw_bounded.max_raw_frame_bytes = yuyv_min_stride(17).unwrap() * 9 - 1;
        assert!(raw_bounded.validate_geometry(17, 9).is_err());

        let mut decoded_bounded = policy;
        decoded_bounded.max_decoded_frame_bytes = 17 * 9 * 3 - 1;
        assert!(decoded_bounded.validate_geometry(17, 9).is_err());

        let mut aggregate_bounded = policy;
        aggregate_bounded.max_mmap_bytes = yuyv_min_stride(17).unwrap() * 9 * 4 - 1;
        assert!(aggregate_bounded.validate_geometry(17, 9).is_err());
    }

    #[test]
    fn explicit_yuyv_stride_accepts_trailing_storage_without_reinterpreting_it() {
        let buffer = vec![
            10, 128, 20, 128, 30, 128, 99, 128, 40, 128, 50, 128, 60, 128, 88, 128, 250, 251, 252,
        ];
        assert!(decode_yuyv_gray(&buffer, 3, 2).is_err());
        assert_eq!(
            decode_yuyv_gray_with_stride(&buffer, 3, 2, 8).unwrap(),
            vec![10, 20, 30, 40, 50, 60]
        );
    }

    #[test]
    fn test_yuyv_gray_decoder_ignores_scanline_padding() {
        // width=3 requires 8 active bytes per row; stride=12 adds four pad bytes.
        let buffer = vec![
            10, 128, 20, 128, 30, 128, 99, 128, 250, 251, 252, 253, 40, 128, 50, 128, 60, 128, 88,
            128, 240, 241, 242, 243,
        ];
        assert_eq!(
            decode_yuyv_gray(&buffer, 3, 2).unwrap(),
            vec![10, 20, 30, 40, 50, 60]
        );
    }

    #[test]
    fn test_yuyv_rgb_decoder_handles_odd_width_and_padding() {
        let buffer = vec![
            10, 128, 20, 128, 30, 128, 99, 128, 250, 251, 252, 253, 40, 128, 50, 128, 60, 128, 88,
            128, 240, 241, 242, 243,
        ];
        let rgb = decode_yuyv_rgb(&buffer, 3, 2).unwrap();
        assert_eq!(rgb.len(), 18);
        assert_eq!(&rgb[0..3], &[10, 10, 10]);
        assert_eq!(&rgb[6..9], &[30, 30, 30]);
        assert_eq!(&rgb[9..12], &[40, 40, 40]);
    }

    #[test]
    fn test_yuyv_decoder_rejects_truncated_rows() {
        assert!(decode_yuyv_gray(&[0; 7], 3, 1).is_err());
        assert!(decode_yuyv_rgb(&[0; 8], 3, 2).is_err());
    }

    #[test]
    fn test_try_with_mock_rejects_invalid_capacity() {
        assert!(CameraManifold::try_with_mock(VisionConfig::default(), 0, 16).is_err());
    }

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

    fn captured_gray(timestamp_us: u64, pixels: Vec<u8>) -> CapturedFrame {
        CapturedFrame {
            pixels,
            width: 8,
            height: 8,
            channels: 1,
            timestamp_us,
        }
    }

    #[test]
    fn test_mock_camera_budget_rejects_oversized_geometry_before_allocation() {
        assert!(MockCameraSource::try_with_budget(64, 64, 64 * 64 * 3).is_ok());
        let error = MockCameraSource::try_with_budget(64, 64, 64 * 64 * 3 - 1)
            .err()
            .expect("one byte below the RGB requirement must fail");
        assert!(error.contains("allocation budget"));
        assert!(MockCameraSource::try_new(0, 64).is_err());
    }

    #[test]
    fn test_mock_camera_budget_checkpoint_rejection_is_atomic() {
        let mut source = MockCameraSource::try_with_budget(8, 8, 1024).unwrap();
        source.next_frame().unwrap();
        let before = source.save_state();
        let mut malformed = before.clone();
        malformed.max_frame_bytes = 8;
        assert!(source.load_state(&malformed).is_err());
        assert_eq!(source.save_state(), before);
    }

    #[test]
    fn test_mock_camera_legacy_state_uses_default_budget() {
        let json = r#"{"width":8,"height":8,"frame_count":3}"#;
        let state: MockCameraSourceState = serde_json::from_str(json).unwrap();
        assert_eq!(state.max_frame_bytes, DEFAULT_MOCK_CAMERA_MAX_FRAME_BYTES);
        let mut source = MockCameraSource::try_new(8, 8).unwrap();
        source.load_state(&state).unwrap();
        assert_eq!(source.frame_count(), 3);
    }

    #[test]
    fn test_camera_rejects_non_monotonic_timestamps_without_mutation() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut camera = CameraManifold::with_mock(config, 8, 8);
        camera
            .process_captured_frame(captured_gray(10_000, vec![1; 64]))
            .unwrap();
        let frame_count = camera.manifold().frame_count();

        let error = camera
            .process_captured_frame(captured_gray(10_000, vec![2; 64]))
            .unwrap_err();

        assert!(error.contains("increase strictly"));
        assert_eq!(camera.manifold().frame_count(), frame_count);
        assert_eq!(camera.last_timestamp_us, Some(10_000));
    }

    #[test]
    fn test_rejected_camera_frame_does_not_advance_clock() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut camera = CameraManifold::with_mock(config, 8, 8);
        camera
            .process_captured_frame(captured_gray(20_000, vec![1; 64]))
            .unwrap();

        let malformed = captured_gray(30_000, vec![9; 63]);
        assert!(camera.process_captured_frame(malformed).is_err());
        assert_eq!(camera.last_timestamp_us, Some(20_000));
        assert_eq!(camera.manifold().frame_count(), 1);

        camera
            .process_captured_frame(captured_gray(25_000, vec![2; 64]))
            .expect("timestamp after the last accepted frame remains valid");
        assert_eq!(camera.last_timestamp_us, Some(25_000));
        assert_eq!(camera.manifold().frame_count(), 2);
    }

    #[test]
    fn test_camera_accepts_zero_as_first_timestamp_without_sentinel_collision() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut camera = CameraManifold::with_mock(config, 8, 8);

        camera
            .process_captured_frame(captured_gray(0, vec![1; 64]))
            .unwrap();
        camera
            .process_captured_frame(captured_gray(33_333, vec![2; 64]))
            .unwrap();

        assert_eq!(camera.last_timestamp_us, Some(33_333));
        assert_eq!(camera.manifold().frame_count(), 2);
    }

    #[test]
    fn test_camera_checkpoint_roundtrip_resumes_source_sequence() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = CameraManifold::with_mock(config.clone(), 8, 8);
        source.tick().unwrap();
        source.tick().unwrap();
        let saved = source.save_state();
        let expected_next = source.source.next_frame().unwrap();

        let mut restored = CameraManifold::with_mock(config, 8, 8);
        restored.load_state(&saved).unwrap();
        let actual_next = restored.source.next_frame().unwrap();

        assert_eq!(actual_next.pixels, expected_next.pixels);
        assert_eq!(actual_next.timestamp_us, expected_next.timestamp_us);
        assert_eq!(restored.manifold.frame_count(), saved.manifold.frame_count);
        assert_eq!(restored.last_timestamp_us, saved.last_timestamp_us);
    }

    #[test]
    fn test_camera_checkpoint_rejection_is_atomic() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut camera = CameraManifold::with_mock(config, 8, 8);
        camera.tick().unwrap();
        let before = camera.save_state();
        let before_hv = camera.manifold.state().clone();
        let mut malformed = before.clone();
        malformed.source.width = 16;

        let error = camera.load_state(&malformed).unwrap_err();
        assert!(error.contains("geometry mismatch"));
        assert_eq!(camera.source.save_state(), before.source);
        assert_eq!(camera.manifold.frame_count(), before.manifold.frame_count);
        assert_eq!(camera.last_timestamp_us, before.last_timestamp_us);
        assert!(camera.manifold.state().similarity(&before_hv) > 0.999_999);
    }

    #[test]
    fn test_camera_reset_restarts_mock_clock_and_pixels() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut camera = CameraManifold::with_mock(config, 8, 8);
        camera.tick().unwrap();
        camera.tick().unwrap();
        camera.reset();

        assert_eq!(camera.source.frame_count(), 0);
        assert_eq!(camera.manifold.frame_count(), 0);
        assert_eq!(camera.last_timestamp_us, None);
        let first = camera.source.next_frame().unwrap();
        assert_eq!(first.timestamp_us, 33_333);
        assert!(first.pixels.iter().all(|pixel| *pixel == 0));
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
    #[test]
    fn audited_camera_loader_preserves_state_on_failure() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-camera-audited-load-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("camera.chk");
        std::fs::write(&path, b"broken").unwrap();
        let mut camera = CameraManifold::with_mock(VisionConfig::default(), 16, 16);
        let before = serde_json::to_vec(&camera.save_state()).unwrap();
        assert!(
            camera
                .load_checkpoint_file_with_retention_audited(
                    &path,
                    crate::checkpoint::CheckpointRetentionPolicy {
                        previous_generations: 0
                    },
                )
                .is_err()
        );
        assert_eq!(serde_json::to_vec(&camera.save_state()).unwrap(), before);
        let _ = std::fs::remove_dir_all(directory);
    }
}
