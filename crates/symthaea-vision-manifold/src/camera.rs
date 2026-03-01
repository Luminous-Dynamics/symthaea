//! Camera input source for live video processing.
//!
//! When the `camera` feature is enabled, provides `CameraSource` backed by
//! the `nokhwa` crate for real webcam capture. Without the feature, provides
//! a `MockCameraSource` for testing.
//!
//! `CameraManifold` combines a camera source with a `VisionManifold` for
//! convenient tick-based processing.

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

// ── Real camera (nokhwa) ──────────────────────────────────────────────

#[cfg(feature = "camera")]
mod real_camera {
    use super::CapturedFrame;
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::Camera;

    /// Live camera source backed by nokhwa.
    pub struct CameraSource {
        camera: Camera,
        width: u32,
        height: u32,
        frame_count: u64,
    }

    impl CameraSource {
        /// Open camera at the given index with requested resolution.
        pub fn new(camera_index: u32, width: u32, height: u32) -> Result<Self, String> {
            let index = CameraIndex::Index(camera_index);
            let requested = RequestedFormat::new::<RgbFormat>(
                RequestedFormatType::AbsoluteHighestResolution,
            );

            let camera = Camera::new(index, requested)
                .map_err(|e| format!("Failed to open camera: {e}"))?;

            Ok(Self {
                camera,
                width,
                height,
                frame_count: 0,
            })
        }

        /// Capture the next frame as RGB pixels.
        pub fn next_frame(&mut self) -> Result<CapturedFrame, String> {
            let frame = self
                .camera
                .frame()
                .map_err(|e| format!("Failed to capture frame: {e}"))?;

            let decoded = frame
                .decode_image::<RgbFormat>()
                .map_err(|e| format!("Failed to decode frame: {e}"))?;

            let (w, h) = (decoded.width(), decoded.height());
            let pixels = decoded.into_raw();

            self.frame_count += 1;
            let timestamp_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;

            Ok(CapturedFrame {
                pixels,
                width: w,
                height: h,
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

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

/// Convenience wrapper combining a camera source with a `VisionManifold`.
///
/// Provides a simple `tick()` interface for frame-by-frame processing.
pub struct CameraManifold {
    #[cfg(feature = "camera")]
    source: CameraSource,
    #[cfg(not(feature = "camera"))]
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

    /// Create with a real camera (requires `camera` feature).
    #[cfg(feature = "camera")]
    pub fn with_camera(
        config: VisionConfig,
        camera_index: u32,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let source = CameraSource::new(camera_index, width, height)?;
        let manifold = VisionManifold::new(config, width, height);
        Ok(Self {
            source,
            manifold,
            last_timestamp_us: 0,
        })
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
}
