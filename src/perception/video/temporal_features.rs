// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Feature Extraction - Converting video to LTC-ready features
//!
//! This module extracts time-varying features from video frames that can be
//! fed into the HierarchicalLTC for temporal pattern recognition.
//!
//! ## Feature Vector
//!
//! Each frame produces a 32D feature vector:
//! - [0-7]: Brightness histogram (8 bins)
//! - [8-15]: Edge density histogram (8 bins)
//! - [16-23]: Motion magnitude histogram (8 bins)
//! - [24-27]: Quadrant motion (4 values)
//! - [28-31]: Global features (brightness, edges, motion, variance)

use super::frame::Frame;
use super::optical_flow::OpticalFlow;

/// Temporal features extracted from a video frame
#[derive(Debug, Clone)]
pub struct TemporalFeatures {
    /// Raw feature vector (32D)
    pub features: [f32; 32],
    /// Frame sequence number
    pub sequence: u64,
    /// Global brightness (0.0 to 1.0)
    pub brightness: f32,
    /// Global edge density (0.0 to 1.0)
    pub edge_density: f32,
    /// Global motion magnitude (0.0 to 1.0)
    pub motion: f32,
    /// Brightness variance (measure of visual complexity)
    pub variance: f32,
}

impl TemporalFeatures {
    /// Create zero features (for initialization)
    pub fn zero() -> Self {
        Self {
            features: [0.0; 32],
            sequence: 0,
            brightness: 0.0,
            edge_density: 0.0,
            motion: 0.0,
            variance: 0.0,
        }
    }

    /// Get feature vector as slice
    pub fn as_slice(&self) -> &[f32] {
        &self.features
    }

    /// Compute cosine similarity with another feature vector
    pub fn similarity(&self, other: &TemporalFeatures) -> f32 {
        let dot: f32 = self
            .features
            .iter()
            .zip(other.features.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        let mag_a: f32 = self.features.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.features.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if mag_a > 0.0 && mag_b > 0.0 {
            dot / (mag_a * mag_b)
        } else {
            0.0
        }
    }
}

/// Extracts temporal features from video frames
pub struct TemporalFeatureExtractor {
    /// Optical flow calculator
    optical_flow: OpticalFlow,
    /// Previous brightness for change detection
    prev_brightness: f32,
    /// Running average for normalization
    avg_brightness: f32,
    avg_motion: f32,
    /// Exponential moving average factor
    ema_alpha: f32,
    /// Frame counter
    frame_count: u64,
}

impl TemporalFeatureExtractor {
    /// Create a new temporal feature extractor
    pub fn new() -> Self {
        Self {
            optical_flow: OpticalFlow::new(8, 5),
            prev_brightness: 0.5,
            avg_brightness: 0.5,
            avg_motion: 0.0,
            ema_alpha: 0.1,
            frame_count: 0,
        }
    }

    /// Extract temporal features from a frame
    pub fn extract(&mut self, frame: &Frame) -> TemporalFeatures {
        let mut features = [0.0f32; 32];

        // Convert to grayscale for processing
        let gray = frame.to_grayscale();
        let pixels = &gray.data;

        // === BRIGHTNESS HISTOGRAM (features 0-7) ===
        let brightness = self.compute_brightness_histogram(pixels, &mut features[0..8]);

        // === EDGE DENSITY HISTOGRAM (features 8-15) ===
        let edge_density = self.compute_edge_histogram(&gray, &mut features[8..16]);

        // === MOTION HISTOGRAM (features 16-23) ===
        let (motion, motion_vectors) = self.optical_flow.process(frame);
        self.compute_motion_histogram(&motion_vectors, &mut features[16..24]);

        // === QUADRANT MOTION (features 24-27) ===
        self.compute_quadrant_motion(&motion_vectors, &mut features[24..28]);

        // === GLOBAL FEATURES (features 28-31) ===
        features[28] = brightness;
        features[29] = edge_density;
        features[30] = motion;

        // Compute variance
        let variance = self.compute_variance(pixels, brightness);
        features[31] = variance;

        // Update running averages
        self.avg_brightness =
            symthaea_core::math::ema_update(self.avg_brightness, brightness, self.ema_alpha);
        self.avg_motion = symthaea_core::math::ema_update(self.avg_motion, motion, self.ema_alpha);
        self.prev_brightness = brightness;
        self.frame_count += 1;

        TemporalFeatures {
            features,
            sequence: frame.sequence,
            brightness,
            edge_density,
            motion,
            variance,
        }
    }

    /// Compute brightness histogram and return average brightness
    fn compute_brightness_histogram(&self, pixels: &[u8], hist: &mut [f32]) -> f32 {
        if pixels.is_empty() {
            return 0.0;
        }

        // 8 bins for brightness levels
        let bin_size = 256.0 / 8.0;
        let mut counts = [0u32; 8];
        let mut sum: u64 = 0;

        for &p in pixels {
            let bin = ((p as f32) / bin_size).min(7.0) as usize;
            counts[bin] += 1;
            sum += p as u64;
        }

        // Normalize histogram
        let total = pixels.len() as f32;
        for (i, &count) in counts.iter().enumerate() {
            hist[i] = count as f32 / total;
        }

        // Return average brightness
        (sum as f32) / (pixels.len() as f32 * 255.0)
    }

    /// Compute edge density histogram
    fn compute_edge_histogram(&self, gray: &Frame, hist: &mut [f32]) -> f32 {
        let (width, height) = (gray.width as usize, gray.height as usize);
        let pixels = &gray.data;

        if width < 2 || height < 2 {
            return 0.0;
        }

        // Simple Sobel-like edge detection
        let mut edge_values = Vec::with_capacity(width * height);
        let mut total_edge: f32 = 0.0;

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = y * width + x;

                // Horizontal gradient
                let gx = pixels[idx + 1] as i16 - pixels[idx - 1] as i16;
                // Vertical gradient
                let gy = pixels[idx + width] as i16 - pixels[idx - width] as i16;

                let magnitude = ((gx * gx + gy * gy) as f32).sqrt() / 360.0; // Normalize
                let magnitude = magnitude.min(1.0);
                edge_values.push(magnitude);
                total_edge += magnitude;
            }
        }

        // Build histogram
        let bin_size = 1.0 / 8.0;
        let mut counts = [0u32; 8];

        for &e in &edge_values {
            let bin = ((e / bin_size) as usize).min(7);
            counts[bin] += 1;
        }

        let total = edge_values.len() as f32;
        if total > 0.0 {
            for (i, &count) in counts.iter().enumerate() {
                hist[i] = count as f32 / total;
            }
            total_edge / total
        } else {
            0.0
        }
    }

    /// Compute motion histogram from motion vectors
    fn compute_motion_histogram(
        &self,
        vectors: &[super::optical_flow::MotionVector],
        hist: &mut [f32],
    ) {
        if vectors.is_empty() {
            return;
        }

        let bin_size = 1.0 / 8.0;
        let mut counts = [0u32; 8];

        for v in vectors {
            let bin = ((v.magnitude / bin_size) as usize).min(7);
            counts[bin] += 1;
        }

        let total = vectors.len() as f32;
        for (i, &count) in counts.iter().enumerate() {
            hist[i] = count as f32 / total;
        }
    }

    /// Compute motion per quadrant
    fn compute_quadrant_motion(
        &self,
        vectors: &[super::optical_flow::MotionVector],
        quadrants: &mut [f32],
    ) {
        let mut quad_sum = [0.0f32; 4];
        let mut quad_count = [0u32; 4];

        for v in vectors {
            let qx = if v.x < 0.5 { 0 } else { 1 };
            let qy = if v.y < 0.5 { 0 } else { 1 };
            let quad = qy * 2 + qx;
            quad_sum[quad] += v.magnitude;
            quad_count[quad] += 1;
        }

        for i in 0..4 {
            quadrants[i] = if quad_count[i] > 0 {
                quad_sum[i] / quad_count[i] as f32
            } else {
                0.0
            };
        }
    }

    /// Compute brightness variance
    fn compute_variance(&self, pixels: &[u8], mean: f32) -> f32 {
        if pixels.is_empty() {
            return 0.0;
        }

        let mean_byte = (mean * 255.0) as i32;
        let sum_sq: i64 = pixels
            .iter()
            .map(|&p| {
                let diff = p as i32 - mean_byte;
                (diff * diff) as i64
            })
            .sum();

        let variance = (sum_sq as f64) / (pixels.len() as f64 * 255.0 * 255.0);
        variance.sqrt() as f32
    }

    /// Reset the extractor state
    pub fn reset(&mut self) {
        self.optical_flow.reset();
        self.prev_brightness = 0.5;
        self.avg_brightness = 0.5;
        self.avg_motion = 0.0;
        self.frame_count = 0;
    }

    /// Get the current running averages
    pub fn averages(&self) -> (f32, f32) {
        (self.avg_brightness, self.avg_motion)
    }
}

impl Default for TemporalFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::frame::FrameFormat;
    use super::*;

    fn create_test_frame(brightness: u8, seq: u64) -> Frame {
        let data = vec![brightness; 64 * 64 * 3];
        Frame::new(data, 64, 64, FrameFormat::Rgb24, seq)
    }

    #[test]
    fn test_feature_extraction() {
        let mut extractor = TemporalFeatureExtractor::new();

        let frame = create_test_frame(128, 0);
        let features = extractor.extract(&frame);

        assert_eq!(features.features.len(), 32);
        assert!(features.brightness > 0.4 && features.brightness < 0.6);
    }

    #[test]
    fn test_brightness_change_detection() {
        let mut extractor = TemporalFeatureExtractor::new();

        let frame1 = create_test_frame(50, 0);
        let frame2 = create_test_frame(200, 1);

        let f1 = extractor.extract(&frame1);
        let f2 = extractor.extract(&frame2);

        // Motion should be detected due to brightness change
        assert!(f2.motion > f1.motion);
    }

    #[test]
    fn test_similarity() {
        let mut extractor = TemporalFeatureExtractor::new();

        // Same brightness should give high similarity
        let frame1 = create_test_frame(128, 0);
        let frame2 = create_test_frame(128, 1);

        let f1 = extractor.extract(&frame1);
        let f2 = extractor.extract(&frame2);

        let sim = f1.similarity(&f2);
        assert!(sim > 0.9);
    }

    #[test]
    fn test_zero_features() {
        let zero = TemporalFeatures::zero();
        assert_eq!(zero.brightness, 0.0);
        assert_eq!(zero.motion, 0.0);
    }
}
