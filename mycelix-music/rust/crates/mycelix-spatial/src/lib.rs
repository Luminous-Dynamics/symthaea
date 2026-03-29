// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spatial Audio Engine
//!
//! Binaural rendering, ambisonics, HRTF processing for immersive audio.

use nalgebra::{Point3, Rotation3, Unit, UnitQuaternion, Vector3};
use ndarray::{Array1, Array2};
use parking_lot::RwLock;
use realfft::{RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::Arc;
use thiserror::Error;

pub mod ambisonics;
pub mod binaural;
pub mod hrtf;
pub mod room;

#[derive(Error, Debug)]
pub enum SpatialError {
    #[error("Invalid position: {0}")]
    InvalidPosition(String),

    #[error("Invalid HRTF data: {0}")]
    InvalidHrtf(String),

    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    #[error("Invalid ambisonics order: {0}")]
    InvalidAmbisonicsOrder(u8),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, SpatialError>;

/// 3D position in space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn to_point3(&self) -> Point3<f32> {
        Point3::new(self.x, self.y, self.z)
    }

    pub fn to_vector3(&self) -> Vector3<f32> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Convert to spherical coordinates (azimuth, elevation, distance)
    pub fn to_spherical(&self) -> (f32, f32, f32) {
        let distance = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if distance < 1e-6 {
            return (0.0, 0.0, 0.0);
        }

        let azimuth = self.x.atan2(self.z);
        let elevation = (self.y / distance).asin();

        (azimuth, elevation, distance)
    }

    /// Create from spherical coordinates
    pub fn from_spherical(azimuth: f32, elevation: f32, distance: f32) -> Self {
        let cos_elev = elevation.cos();
        Self {
            x: distance * cos_elev * azimuth.sin(),
            y: distance * elevation.sin(),
            z: distance * cos_elev * azimuth.cos(),
        }
    }

    pub fn distance_to(&self, other: &Position) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl From<Point3<f32>> for Position {
    fn from(p: Point3<f32>) -> Self {
        Self::new(p.x, p.y, p.z)
    }
}

impl From<Vector3<f32>> for Position {
    fn from(v: Vector3<f32>) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

/// Listener orientation (head tracking)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Orientation {
    /// Rotation quaternion
    pub rotation: [f32; 4],
}

impl Orientation {
    pub fn new(yaw: f32, pitch: f32, roll: f32) -> Self {
        let quat = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        Self {
            rotation: [quat.i, quat.j, quat.k, quat.w],
        }
    }

    pub fn identity() -> Self {
        Self {
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn to_quaternion(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            self.rotation[3],
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
        ))
    }

    /// Get forward direction vector
    pub fn forward(&self) -> Vector3<f32> {
        let quat = self.to_quaternion();
        quat * Vector3::new(0.0, 0.0, -1.0)
    }

    /// Get up direction vector
    pub fn up(&self) -> Vector3<f32> {
        let quat = self.to_quaternion();
        quat * Vector3::new(0.0, 1.0, 0.0)
    }

    /// Get right direction vector
    pub fn right(&self) -> Vector3<f32> {
        let quat = self.to_quaternion();
        quat * Vector3::new(1.0, 0.0, 0.0)
    }
}

impl Default for Orientation {
    fn default() -> Self {
        Self::identity()
    }
}

/// Audio source in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSource {
    pub id: String,
    pub position: Position,
    pub gain: f32,
    /// Inner cone angle in radians (full volume)
    pub inner_cone_angle: f32,
    /// Outer cone angle in radians (attenuated)
    pub outer_cone_angle: f32,
    /// Attenuation outside outer cone
    pub cone_outer_gain: f32,
    /// Source direction for directional sources
    pub direction: Option<Vector3<f32>>,
    /// Distance model parameters
    pub ref_distance: f32,
    pub max_distance: f32,
    pub rolloff_factor: f32,
}

impl AudioSource {
    pub fn new(id: impl Into<String>, position: Position) -> Self {
        Self {
            id: id.into(),
            position,
            gain: 1.0,
            inner_cone_angle: 2.0 * PI,
            outer_cone_angle: 2.0 * PI,
            cone_outer_gain: 0.0,
            direction: None,
            ref_distance: 1.0,
            max_distance: 100.0,
            rolloff_factor: 1.0,
        }
    }

    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }

    pub fn with_direction(mut self, direction: Vector3<f32>) -> Self {
        self.direction = Some(direction.normalize());
        self
    }

    /// Calculate distance attenuation
    pub fn distance_attenuation(&self, listener_pos: &Position) -> f32 {
        let distance = self.position.distance_to(listener_pos);

        if distance <= self.ref_distance {
            return 1.0;
        }

        if distance >= self.max_distance {
            return 0.0;
        }

        // Inverse distance attenuation
        let attenuation = self.ref_distance
            / (self.ref_distance
                + self.rolloff_factor * (distance - self.ref_distance));

        attenuation.clamp(0.0, 1.0)
    }

    /// Calculate cone attenuation for directional sources
    pub fn cone_attenuation(&self, listener_pos: &Position) -> f32 {
        let direction = match &self.direction {
            Some(d) => d,
            None => return 1.0, // Omnidirectional
        };

        let to_listener = (listener_pos.to_vector3() - self.position.to_vector3()).normalize();
        let angle = direction.dot(&to_listener).acos();

        if angle <= self.inner_cone_angle / 2.0 {
            1.0
        } else if angle >= self.outer_cone_angle / 2.0 {
            self.cone_outer_gain
        } else {
            // Linear interpolation between inner and outer cone
            let t = (angle - self.inner_cone_angle / 2.0)
                / (self.outer_cone_angle / 2.0 - self.inner_cone_angle / 2.0);
            1.0 - t * (1.0 - self.cone_outer_gain)
        }
    }
}

/// Listener in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Listener {
    pub position: Position,
    pub orientation: Orientation,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: Position::origin(),
            orientation: Orientation::identity(),
        }
    }
}

impl Listener {
    pub fn new(position: Position, orientation: Orientation) -> Self {
        Self {
            position,
            orientation,
        }
    }

    /// Transform a world position to listener-relative coordinates
    pub fn world_to_listener(&self, world_pos: &Position) -> Position {
        let relative = world_pos.to_vector3() - self.position.to_vector3();
        let rotated = self.orientation.to_quaternion().inverse() * relative;
        rotated.into()
    }
}

/// Spatial audio renderer
pub struct SpatialRenderer {
    sample_rate: u32,
    block_size: usize,
    hrtf: Arc<hrtf::HrtfDatabase>,
    sources: RwLock<HashMap<String, AudioSource>>,
    listener: RwLock<Listener>,
    /// FFT planner for convolution
    fft_planner: RwLock<RealFftPlanner<f32>>,
    /// Per-source processing state
    source_states: RwLock<HashMap<String, SourceState>>,
}

struct SourceState {
    /// Previous output for crossfade
    prev_left: Vec<f32>,
    prev_right: Vec<f32>,
    /// Overlap-add buffer
    overlap_left: Vec<f32>,
    overlap_right: Vec<f32>,
}

impl SpatialRenderer {
    pub fn new(sample_rate: u32, block_size: usize) -> Self {
        Self {
            sample_rate,
            block_size,
            hrtf: Arc::new(hrtf::HrtfDatabase::default_mit()),
            sources: RwLock::new(HashMap::new()),
            listener: RwLock::new(Listener::default()),
            fft_planner: RwLock::new(RealFftPlanner::new()),
            source_states: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_hrtf(mut self, hrtf: hrtf::HrtfDatabase) -> Self {
        self.hrtf = Arc::new(hrtf);
        self
    }

    /// Add or update an audio source
    pub fn set_source(&self, source: AudioSource) {
        let id = source.id.clone();
        self.sources.write().insert(id.clone(), source);

        // Initialize source state if needed
        let mut states = self.source_states.write();
        states.entry(id).or_insert_with(|| SourceState {
            prev_left: vec![0.0; self.block_size],
            prev_right: vec![0.0; self.block_size],
            overlap_left: vec![0.0; self.block_size],
            overlap_right: vec![0.0; self.block_size],
        });
    }

    /// Remove an audio source
    pub fn remove_source(&self, id: &str) {
        self.sources.write().remove(id);
        self.source_states.write().remove(id);
    }

    /// Update listener position and orientation
    pub fn set_listener(&self, listener: Listener) {
        *self.listener.write() = listener;
    }

    /// Process a mono source and render to binaural stereo
    pub fn process_source(&self, source_id: &str, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let sources = self.sources.read();
        let source = sources
            .get(source_id)
            .ok_or_else(|| SpatialError::InvalidPosition(format!("Source {} not found", source_id)))?;

        let listener = self.listener.read();

        // Transform source position to listener space
        let relative_pos = listener.world_to_listener(&source.position);
        let (azimuth, elevation, distance) = relative_pos.to_spherical();

        // Calculate attenuation
        let distance_atten = source.distance_attenuation(&listener.position);
        let cone_atten = source.cone_attenuation(&listener.position);
        let total_gain = source.gain * distance_atten * cone_atten;

        // Get HRTF filters for this direction
        let (hrir_left, hrir_right) = self.hrtf.get_hrir(azimuth, elevation);

        // Apply gain to input
        let gained_input: Vec<f32> = input.iter().map(|s| s * total_gain).collect();

        // Convolve with HRTF
        let left = self.convolve(&gained_input, &hrir_left);
        let right = self.convolve(&gained_input, &hrir_right);

        Ok((left, right))
    }

    /// Process multiple sources and mix to binaural output
    pub fn process_all(
        &self,
        source_inputs: &HashMap<String, Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let block_size = source_inputs
            .values()
            .next()
            .map(|v| v.len())
            .unwrap_or(self.block_size);

        let mut left_mix = vec![0.0f32; block_size];
        let mut right_mix = vec![0.0f32; block_size];

        for (source_id, input) in source_inputs {
            let (left, right) = self.process_source(source_id, input)?;

            for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
                if i < left_mix.len() {
                    left_mix[i] += l;
                    right_mix[i] += r;
                }
            }
        }

        Ok((left_mix, right_mix))
    }

    /// Simple time-domain convolution (for short filters)
    fn convolve(&self, input: &[f32], kernel: &[f32]) -> Vec<f32> {
        let output_len = input.len() + kernel.len() - 1;
        let mut output = vec![0.0f32; output_len];

        for (i, &x) in input.iter().enumerate() {
            for (j, &h) in kernel.iter().enumerate() {
                output[i + j] += x * h;
            }
        }

        // Truncate to input length
        output.truncate(input.len());
        output
    }

    /// Get all source positions for visualization
    pub fn get_source_positions(&self) -> Vec<(String, Position)> {
        self.sources
            .read()
            .iter()
            .map(|(id, source)| (id.clone(), source.position))
            .collect()
    }
}

/// Distance attenuation models
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceModel {
    /// No distance attenuation
    None,
    /// Linear attenuation
    Linear,
    /// Inverse distance
    Inverse,
    /// Exponential falloff
    Exponential,
}

impl DistanceModel {
    pub fn calculate(
        &self,
        distance: f32,
        ref_distance: f32,
        max_distance: f32,
        rolloff: f32,
    ) -> f32 {
        match self {
            DistanceModel::None => 1.0,
            DistanceModel::Linear => {
                let d = distance.clamp(ref_distance, max_distance);
                1.0 - rolloff * (d - ref_distance) / (max_distance - ref_distance)
            }
            DistanceModel::Inverse => {
                ref_distance / (ref_distance + rolloff * (distance.max(ref_distance) - ref_distance))
            }
            DistanceModel::Exponential => {
                (distance.max(ref_distance) / ref_distance).powf(-rolloff)
            }
        }
    }
}

/// Panning laws for stereo sources
#[derive(Debug, Clone, Copy)]
pub enum PanLaw {
    /// Linear panning (-6dB center)
    Linear,
    /// Constant power (-3dB center)
    ConstantPower,
    /// -4.5dB center
    Balanced,
}

impl PanLaw {
    /// Calculate left/right gains for pan value (-1 to 1)
    pub fn calculate(&self, pan: f32) -> (f32, f32) {
        let pan = pan.clamp(-1.0, 1.0);

        match self {
            PanLaw::Linear => {
                let left = (1.0 - pan) / 2.0;
                let right = (1.0 + pan) / 2.0;
                (left, right)
            }
            PanLaw::ConstantPower => {
                let angle = (pan + 1.0) * PI / 4.0;
                (angle.cos(), angle.sin())
            }
            PanLaw::Balanced => {
                // Compromise between linear and constant power
                let linear_left = (1.0 - pan) / 2.0;
                let linear_right = (1.0 + pan) / 2.0;
                let angle = (pan + 1.0) * PI / 4.0;
                let cp_left = angle.cos();
                let cp_right = angle.sin();

                // Blend 50/50
                ((linear_left + cp_left) / 2.0, (linear_right + cp_right) / 2.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_spherical() {
        let pos = Position::new(1.0, 0.0, 0.0);
        let (az, el, dist) = pos.to_spherical();
        assert!((dist - 1.0).abs() < 0.001);
        assert!((az - PI / 2.0).abs() < 0.001);
        assert!(el.abs() < 0.001);
    }

    #[test]
    fn test_distance_attenuation() {
        let source = AudioSource::new("test", Position::new(10.0, 0.0, 0.0));
        let listener = Position::origin();
        let atten = source.distance_attenuation(&listener);
        assert!(atten < 1.0);
        assert!(atten > 0.0);
    }

    #[test]
    fn test_pan_law() {
        let (l, r) = PanLaw::ConstantPower.calculate(0.0);
        // At center, both should be approximately equal
        assert!((l - r).abs() < 0.01);
        // And sum of squares should be approximately 1
        assert!((l * l + r * r - 1.0).abs() < 0.01);
    }
}
