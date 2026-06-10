// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Optical Flow - Motion detection via frame differencing
//!
//! Provides simple but effective motion detection for temporal perception.
//! Uses frame differencing rather than full optical flow for speed (5W target).

use super::frame::Frame;

/// Motion vector for a region of the image
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    /// Region center X
    pub x: f32,
    /// Region center Y
    pub y: f32,
    /// Motion magnitude (0.0 to 1.0)
    pub magnitude: f32,
    /// Motion direction in radians (0 = right, PI/2 = down)
    pub direction: f32,
}

/// Optical flow calculator using frame differencing
pub struct OpticalFlow {
    /// Previous frame (grayscale, downsampled)
    prev_frame: Option<Vec<u8>>,
    /// Frame dimensions
    width: u32,
    height: u32,
    /// Grid size for motion regions
    grid_size: u32,
    /// Threshold for motion detection (0-255)
    threshold: u8,
}

impl OpticalFlow {
    /// Create a new optical flow calculator
    ///
    /// # Arguments
    /// * `grid_size` - Size of grid cells for motion detection (e.g., 8 for 8x8 cells)
    /// * `threshold` - Minimum pixel difference to count as motion (0-255)
    pub fn new(grid_size: u32, threshold: u8) -> Self {
        Self {
            prev_frame: None,
            width: 0,
            height: 0,
            grid_size,
            threshold,
        }
    }

    /// Process a new frame and compute motion
    ///
    /// Returns overall motion magnitude (0.0 to 1.0) and motion vectors
    pub fn process(&mut self, frame: &Frame) -> (f32, Vec<MotionVector>) {
        let gray = frame.to_grayscale();
        let (width, height) = (gray.width, gray.height);

        // Handle dimension changes
        if width != self.width || height != self.height {
            self.prev_frame = None;
            self.width = width;
            self.height = height;
        }

        let prev = match &self.prev_frame {
            Some(p) => p,
            None => {
                // First frame - store it and return zero motion
                self.prev_frame = Some(gray.data.clone());
                return (0.0, vec![]);
            }
        };

        // Compute frame difference
        let diff: Vec<u8> = gray
            .data
            .iter()
            .zip(prev.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .collect();

        // Store current frame for next iteration
        self.prev_frame = Some(gray.data);

        // Compute overall motion magnitude
        let total_diff: u64 = diff.iter().map(|&d| d as u64).sum();
        let max_possible = (diff.len() * 255) as f64;
        let overall_motion = (total_diff as f64 / max_possible) as f32;

        // Compute motion vectors for grid cells
        let grid_w = width / self.grid_size;
        let grid_h = height / self.grid_size;
        let mut vectors = Vec::with_capacity((grid_w * grid_h) as usize);

        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let (magnitude, dx, dy) = self.compute_cell_motion(
                    &diff,
                    width,
                    height,
                    gx * self.grid_size,
                    gy * self.grid_size,
                    self.grid_size,
                );

                if magnitude > (self.threshold as f32 / 255.0) {
                    let direction = dy.atan2(dx);
                    vectors.push(MotionVector {
                        x: (gx as f32 + 0.5) * self.grid_size as f32 / width as f32,
                        y: (gy as f32 + 0.5) * self.grid_size as f32 / height as f32,
                        magnitude,
                        direction,
                    });
                }
            }
        }

        (overall_motion, vectors)
    }

    /// Compute motion for a single grid cell
    fn compute_cell_motion(
        &self,
        diff: &[u8],
        width: u32,
        _height: u32,
        x: u32,
        y: u32,
        size: u32,
    ) -> (f32, f32, f32) {
        let mut total: u64 = 0;
        let mut count: u32 = 0;

        // Simple magnitude (could be extended with gradient for direction)
        for cy in y..(y + size) {
            for cx in x..(x + size) {
                let idx = (cy * width + cx) as usize;
                if idx < diff.len() {
                    total += diff[idx] as u64;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return (0.0, 0.0, 0.0);
        }

        let magnitude = (total as f32) / (count as f32 * 255.0);

        // For now, return zero direction (would need more complex algorithm)
        // Full optical flow would compute actual motion direction
        (magnitude, 0.0, 0.0)
    }

    /// Reset the motion detector (clears previous frame)
    pub fn reset(&mut self) {
        self.prev_frame = None;
    }

    /// Get motion magnitude history statistics
    pub fn motion_in_region(&self, frame: &Frame, x: u32, y: u32, w: u32, h: u32) -> f32 {
        if let Some(ref prev) = self.prev_frame {
            let gray = frame.to_grayscale();
            if gray.data.len() != prev.len() {
                return 0.0;
            }

            let width = frame.width;
            let mut total: u64 = 0;
            let mut count: u32 = 0;

            for cy in y..(y + h).min(frame.height) {
                for cx in x..(x + w).min(frame.width) {
                    let idx = (cy * width + cx) as usize;
                    if idx < gray.data.len() && idx < prev.len() {
                        let diff = (gray.data[idx] as i16 - prev[idx] as i16).unsigned_abs();
                        total += diff as u64;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                (total as f32) / (count as f32 * 255.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::frame::FrameFormat;
    use super::*;

    fn create_frame(brightness: u8, width: u32, height: u32, seq: u64) -> Frame {
        let data = vec![brightness; (width * height * 3) as usize];
        Frame::new(data, width, height, FrameFormat::Rgb24, seq)
    }

    #[test]
    fn test_optical_flow_creation() {
        let flow = OpticalFlow::new(8, 10);
        assert_eq!(flow.grid_size, 8);
        assert_eq!(flow.threshold, 10);
    }

    #[test]
    fn test_no_motion() {
        let mut flow = OpticalFlow::new(8, 10);

        // Two identical frames should show no motion
        let frame1 = create_frame(128, 64, 64, 0);
        let frame2 = create_frame(128, 64, 64, 1);

        let (mag1, _) = flow.process(&frame1);
        assert_eq!(mag1, 0.0); // First frame, no previous

        let (mag2, _) = flow.process(&frame2);
        assert!(mag2 < 0.01); // Same brightness, no motion
    }

    #[test]
    fn test_full_motion() {
        let mut flow = OpticalFlow::new(8, 10);

        // Black to white should show maximum motion
        let frame1 = create_frame(0, 64, 64, 0);
        let frame2 = create_frame(255, 64, 64, 1);

        flow.process(&frame1);
        let (mag2, _) = flow.process(&frame2);

        assert!(mag2 > 0.9); // Should be near 1.0
    }

    #[test]
    fn test_partial_motion() {
        let mut flow = OpticalFlow::new(8, 10);

        let frame1 = create_frame(100, 64, 64, 0);
        let frame2 = create_frame(150, 64, 64, 1);

        flow.process(&frame1);
        let (mag2, _) = flow.process(&frame2);

        // Difference of 50/255 ≈ 0.2
        assert!(mag2 > 0.1 && mag2 < 0.3);
    }
}
