// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Head-Related Transfer Function (HRTF) processing

use crate::{Position, Result, SpatialError};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// HRTF measurement point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrtfMeasurement {
    /// Azimuth angle in radians
    pub azimuth: f32,
    /// Elevation angle in radians
    pub elevation: f32,
    /// Left ear impulse response
    pub left_ir: Vec<f32>,
    /// Right ear impulse response
    pub right_ir: Vec<f32>,
    /// Interaural time difference in samples
    pub itd: i32,
}

/// HRTF database for binaural rendering
#[derive(Debug, Clone)]
pub struct HrtfDatabase {
    measurements: Vec<HrtfMeasurement>,
    sample_rate: u32,
    ir_length: usize,
}

impl HrtfDatabase {
    pub fn new(sample_rate: u32, ir_length: usize) -> Self {
        Self {
            measurements: Vec::new(),
            sample_rate,
            ir_length,
        }
    }

    /// Create default MIT KEMAR-like HRTF database
    pub fn default_mit() -> Self {
        let sample_rate = 44100;
        let ir_length = 128;
        let mut db = Self::new(sample_rate, ir_length);

        // Generate synthetic HRTF based on simplified head model
        // In production, this would load actual measured HRTF data
        for elev_idx in -4..=9 {
            let elevation = (elev_idx as f32 * 10.0).to_radians();

            let num_azimuths = match elev_idx {
                -4 | 9 => 1,
                -3 | 8 => 6,
                -2 | 7 => 12,
                _ => 24,
            };

            for az_idx in 0..num_azimuths {
                let azimuth = (az_idx as f32 * 360.0 / num_azimuths as f32).to_radians();
                let measurement = generate_synthetic_hrtf(azimuth, elevation, ir_length, sample_rate);
                db.measurements.push(measurement);
            }
        }

        db
    }

    /// Load HRTF from SOFA format (placeholder)
    pub fn load_sofa(_path: &str) -> Result<Self> {
        // In production, would parse SOFA file format
        Ok(Self::default_mit())
    }

    /// Get interpolated HRIR for given direction
    pub fn get_hrir(&self, azimuth: f32, elevation: f32) -> (Vec<f32>, Vec<f32>) {
        // Find nearest measurements for interpolation
        let nearest = self.find_nearest_measurements(azimuth, elevation, 4);

        if nearest.is_empty() {
            // Return impulse if no measurements
            let mut ir = vec![0.0f32; self.ir_length];
            ir[0] = 1.0;
            return (ir.clone(), ir);
        }

        if nearest.len() == 1 {
            return (
                nearest[0].0.left_ir.clone(),
                nearest[0].0.right_ir.clone(),
            );
        }

        // Interpolate between measurements
        self.interpolate_hrir(&nearest, azimuth, elevation)
    }

    fn find_nearest_measurements(
        &self,
        azimuth: f32,
        elevation: f32,
        count: usize,
    ) -> Vec<(&HrtfMeasurement, f32)> {
        let mut distances: Vec<_> = self
            .measurements
            .iter()
            .map(|m| {
                let dist = angular_distance(azimuth, elevation, m.azimuth, m.elevation);
                (m, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(count);
        distances
    }

    fn interpolate_hrir(
        &self,
        nearest: &[(&HrtfMeasurement, f32)],
        _azimuth: f32,
        _elevation: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut left = vec![0.0f32; self.ir_length];
        let mut right = vec![0.0f32; self.ir_length];

        // Calculate weights (inverse distance weighting)
        let total_inv_dist: f32 = nearest.iter().map(|(_, d)| 1.0 / (d + 0.001)).sum();

        for (measurement, dist) in nearest {
            let weight = (1.0 / (dist + 0.001)) / total_inv_dist;

            for i in 0..self.ir_length.min(measurement.left_ir.len()) {
                left[i] += measurement.left_ir[i] * weight;
                right[i] += measurement.right_ir[i] * weight;
            }
        }

        (left, right)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn ir_length(&self) -> usize {
        self.ir_length
    }

    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }
}

impl Default for HrtfDatabase {
    fn default() -> Self {
        Self::default_mit()
    }
}

/// Calculate angular distance between two directions
fn angular_distance(az1: f32, el1: f32, az2: f32, el2: f32) -> f32 {
    // Great circle distance on unit sphere
    let sin_el1 = el1.sin();
    let sin_el2 = el2.sin();
    let cos_el1 = el1.cos();
    let cos_el2 = el2.cos();
    let cos_daz = (az1 - az2).cos();

    let cos_dist = sin_el1 * sin_el2 + cos_el1 * cos_el2 * cos_daz;
    cos_dist.clamp(-1.0, 1.0).acos()
}

/// Generate synthetic HRTF based on simplified head model
fn generate_synthetic_hrtf(
    azimuth: f32,
    elevation: f32,
    ir_length: usize,
    sample_rate: u32,
) -> HrtfMeasurement {
    let mut left_ir = vec![0.0f32; ir_length];
    let mut right_ir = vec![0.0f32; ir_length];

    // Head radius in meters (average adult)
    let head_radius = 0.0875;
    let speed_of_sound = 343.0;

    // Calculate interaural time difference (ITD)
    // Woodworth formula
    let theta = azimuth.abs();
    let itd_seconds = (head_radius / speed_of_sound) * (theta + theta.sin());
    let itd_samples = (itd_seconds * sample_rate as f32).round() as i32;

    // Calculate interaural level difference (ILD)
    // Simple head shadow model
    let ild_db = 10.0 * (1.0 + azimuth.cos()) / 2.0 - 5.0;
    let ild_linear = 10.0_f32.powf(ild_db / 20.0);

    // Generate simple impulse responses
    // In reality, these would come from measurements or more complex models
    let decay_rate = 0.95;
    let frequency = 1000.0; // Dominant frequency for shape

    for i in 0..ir_length {
        let t = i as f32 / sample_rate as f32;
        let envelope = (-t * 50.0).exp(); // Quick decay

        // Simple damped sinusoid
        let sample = envelope * (2.0 * PI * frequency * t).sin();

        // Apply ITD and ILD
        if azimuth >= 0.0 {
            // Source on right
            let left_delay = itd_samples.max(0) as usize;
            let right_delay = (-itd_samples).max(0) as usize;

            if i >= left_delay && i - left_delay < ir_length {
                left_ir[i] = sample / ild_linear;
            }
            if i >= right_delay && i - right_delay < ir_length {
                right_ir[i] = sample * ild_linear;
            }
        } else {
            // Source on left
            let left_delay = (-itd_samples).max(0) as usize;
            let right_delay = itd_samples.max(0) as usize;

            if i >= left_delay && i - left_delay < ir_length {
                left_ir[i] = sample * ild_linear;
            }
            if i >= right_delay && i - right_delay < ir_length {
                right_ir[i] = sample / ild_linear;
            }
        }
    }

    // Add elevation cues (simple spectral coloring)
    apply_elevation_cues(&mut left_ir, &mut right_ir, elevation, sample_rate);

    // Normalize
    let max_val = left_ir
        .iter()
        .chain(right_ir.iter())
        .map(|x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));

    if max_val > 0.0 {
        for sample in &mut left_ir {
            *sample /= max_val;
        }
        for sample in &mut right_ir {
            *sample /= max_val;
        }
    }

    HrtfMeasurement {
        azimuth,
        elevation,
        left_ir,
        right_ir,
        itd: itd_samples,
    }
}

/// Apply simple elevation cues (high-frequency emphasis for above, reduction for below)
fn apply_elevation_cues(left: &mut [f32], right: &mut [f32], elevation: f32, sample_rate: u32) {
    // Simple one-pole filter to add elevation coloring
    let cutoff = if elevation > 0.0 {
        // Above: boost high frequencies
        8000.0 + elevation.sin() * 4000.0
    } else {
        // Below: reduce high frequencies
        8000.0 - elevation.sin().abs() * 4000.0
    };

    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate as f32;
    let alpha = dt / (rc + dt);

    // Apply to both channels
    let mut prev_left = 0.0;
    let mut prev_right = 0.0;

    for i in 0..left.len() {
        let new_left = prev_left + alpha * (left[i] - prev_left);
        let new_right = prev_right + alpha * (right[i] - prev_right);

        if elevation > 0.0 {
            // High-pass for above
            left[i] = left[i] - new_left;
            right[i] = right[i] - new_right;
        } else {
            // Low-pass for below
            left[i] = new_left;
            right[i] = new_right;
        }

        prev_left = new_left;
        prev_right = new_right;
    }
}

/// HRTF convolver for efficient binaural rendering
pub struct HrtfConvolver {
    hrtf: HrtfDatabase,
    block_size: usize,
    /// FFT-domain HRTF filters
    left_filter: Vec<num_complex::Complex<f32>>,
    right_filter: Vec<num_complex::Complex<f32>>,
    /// Overlap-add buffers
    overlap_left: Vec<f32>,
    overlap_right: Vec<f32>,
    /// Current azimuth/elevation
    current_azimuth: f32,
    current_elevation: f32,
}

impl HrtfConvolver {
    pub fn new(hrtf: HrtfDatabase, block_size: usize) -> Self {
        let fft_size = (block_size + hrtf.ir_length()).next_power_of_two();

        Self {
            hrtf,
            block_size,
            left_filter: vec![num_complex::Complex::new(0.0, 0.0); fft_size / 2 + 1],
            right_filter: vec![num_complex::Complex::new(0.0, 0.0); fft_size / 2 + 1],
            overlap_left: vec![0.0; fft_size],
            overlap_right: vec![0.0; fft_size],
            current_azimuth: 0.0,
            current_elevation: 0.0,
        }
    }

    /// Update the HRTF filters for new direction
    pub fn set_direction(&mut self, azimuth: f32, elevation: f32) {
        // Only update if direction changed significantly
        if (azimuth - self.current_azimuth).abs() < 0.01
            && (elevation - self.current_elevation).abs() < 0.01
        {
            return;
        }

        self.current_azimuth = azimuth;
        self.current_elevation = elevation;

        // Get new HRIR and convert to frequency domain
        let (_left_ir, _right_ir) = self.hrtf.get_hrir(azimuth, elevation);

        // In production, would compute FFT of IRs and store in filter buffers
    }

    /// Process a block of mono input to stereo binaural output
    pub fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // In production, this would use overlap-add FFT convolution
        // For now, use simple time-domain convolution
        let (left_ir, right_ir) = self.hrtf.get_hrir(self.current_azimuth, self.current_elevation);

        let mut left_output = vec![0.0f32; input.len()];
        let mut right_output = vec![0.0f32; input.len()];

        // Simple convolution (would be FFT-based in production)
        for (i, &x) in input.iter().enumerate() {
            for (j, (&hl, &hr)) in left_ir.iter().zip(right_ir.iter()).enumerate() {
                if i + j < left_output.len() {
                    left_output[i + j] += x * hl;
                    right_output[i + j] += x * hr;
                }
            }
        }

        (left_output, right_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hrtf_database() {
        let db = HrtfDatabase::default_mit();
        assert!(db.measurement_count() > 0);
    }

    #[test]
    fn test_hrir_interpolation() {
        let db = HrtfDatabase::default_mit();
        let (left, right) = db.get_hrir(0.5, 0.0);
        assert_eq!(left.len(), db.ir_length());
        assert_eq!(right.len(), db.ir_length());
    }

    #[test]
    fn test_angular_distance() {
        // Same point should have zero distance
        let dist = angular_distance(0.0, 0.0, 0.0, 0.0);
        assert!(dist < 0.001);

        // Opposite points should have PI distance
        let dist = angular_distance(0.0, 0.0, PI, 0.0);
        assert!((dist - PI).abs() < 0.001);
    }
}
