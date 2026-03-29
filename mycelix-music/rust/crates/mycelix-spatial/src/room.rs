// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Room acoustics simulation

use crate::{Position, Result, SpatialError};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Room material absorption coefficients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    /// Absorption coefficients at different frequencies [125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz]
    pub absorption: [f32; 6],
    /// Scattering coefficient (0-1)
    pub scattering: f32,
}

impl Material {
    pub fn concrete() -> Self {
        Self {
            name: "Concrete".to_string(),
            absorption: [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
            scattering: 0.1,
        }
    }

    pub fn wood() -> Self {
        Self {
            name: "Wood".to_string(),
            absorption: [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
            scattering: 0.2,
        }
    }

    pub fn carpet() -> Self {
        Self {
            name: "Carpet".to_string(),
            absorption: [0.02, 0.06, 0.14, 0.37, 0.60, 0.65],
            scattering: 0.3,
        }
    }

    pub fn glass() -> Self {
        Self {
            name: "Glass".to_string(),
            absorption: [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],
            scattering: 0.05,
        }
    }

    pub fn acoustic_tile() -> Self {
        Self {
            name: "Acoustic Tile".to_string(),
            absorption: [0.50, 0.70, 0.60, 0.70, 0.70, 0.50],
            scattering: 0.4,
        }
    }

    pub fn curtain() -> Self {
        Self {
            name: "Heavy Curtain".to_string(),
            absorption: [0.07, 0.31, 0.49, 0.75, 0.70, 0.60],
            scattering: 0.5,
        }
    }

    /// Average absorption across frequencies
    pub fn average_absorption(&self) -> f32 {
        self.absorption.iter().sum::<f32>() / self.absorption.len() as f32
    }
}

/// Shoebox room definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    /// Width (X axis) in meters
    pub width: f32,
    /// Height (Y axis) in meters
    pub height: f32,
    /// Depth (Z axis) in meters
    pub depth: f32,
    /// Wall materials [left, right, floor, ceiling, front, back]
    pub materials: [Material; 6],
}

impl Room {
    pub fn new(width: f32, height: f32, depth: f32) -> Self {
        Self {
            width,
            height,
            depth,
            materials: [
                Material::concrete(), // left
                Material::concrete(), // right
                Material::wood(),     // floor
                Material::concrete(), // ceiling
                Material::concrete(), // front
                Material::concrete(), // back
            ],
        }
    }

    pub fn with_uniform_material(mut self, material: Material) -> Self {
        for m in &mut self.materials {
            *m = material.clone();
        }
        self
    }

    /// Room volume in cubic meters
    pub fn volume(&self) -> f32 {
        self.width * self.height * self.depth
    }

    /// Total surface area
    pub fn surface_area(&self) -> f32 {
        2.0 * (self.width * self.height + self.height * self.depth + self.width * self.depth)
    }

    /// Calculate RT60 (reverberation time) using Sabine equation
    pub fn rt60(&self) -> f32 {
        let volume = self.volume();
        let total_absorption = self.total_absorption();

        // Sabine equation: RT60 = 0.161 * V / A
        0.161 * volume / total_absorption
    }

    /// Total absorption area (sum of surface * absorption)
    fn total_absorption(&self) -> f32 {
        let surfaces = [
            self.height * self.depth, // left
            self.height * self.depth, // right
            self.width * self.depth,  // floor
            self.width * self.depth,  // ceiling
            self.width * self.height, // front
            self.width * self.height, // back
        ];

        surfaces
            .iter()
            .zip(self.materials.iter())
            .map(|(&s, m)| s * m.average_absorption())
            .sum()
    }

    /// Check if position is inside room
    pub fn contains(&self, pos: &Position) -> bool {
        pos.x >= 0.0
            && pos.x <= self.width
            && pos.y >= 0.0
            && pos.y <= self.height
            && pos.z >= 0.0
            && pos.z <= self.depth
    }
}

impl Default for Room {
    fn default() -> Self {
        // Small studio room
        Self::new(5.0, 3.0, 4.0)
    }
}

/// Early reflection
#[derive(Debug, Clone)]
pub struct EarlyReflection {
    /// Delay in seconds
    pub delay: f32,
    /// Gain (including distance attenuation and absorption)
    pub gain: f32,
    /// Direction of arrival
    pub direction: Position,
    /// Which surface was reflected
    pub surface_index: usize,
}

/// Image source method for early reflections
pub struct ImageSourceMethod {
    room: Room,
    max_order: u8,
    speed_of_sound: f32,
    sample_rate: u32,
}

impl ImageSourceMethod {
    pub fn new(room: Room, max_order: u8, sample_rate: u32) -> Self {
        Self {
            room,
            max_order,
            speed_of_sound: 343.0,
            sample_rate,
        }
    }

    /// Calculate early reflections from source to listener
    pub fn calculate_reflections(
        &self,
        source: &Position,
        listener: &Position,
    ) -> Vec<EarlyReflection> {
        let mut reflections = Vec::new();

        // Direct sound
        let direct_distance = source.distance_to(listener);
        let direct_delay = direct_distance / self.speed_of_sound;

        // First-order reflections (image sources)
        for order in 1..=self.max_order as i32 {
            self.add_reflections_for_order(
                source,
                listener,
                order,
                &mut reflections,
                direct_delay,
            );
        }

        // Sort by delay
        reflections.sort_by(|a, b| a.delay.partial_cmp(&b.delay).unwrap());

        reflections
    }

    fn add_reflections_for_order(
        &self,
        source: &Position,
        listener: &Position,
        order: i32,
        reflections: &mut Vec<EarlyReflection>,
        direct_delay: f32,
    ) {
        // Generate image sources for this order
        let images = self.generate_image_sources(source, order);

        for (image, surface_indices) in images {
            // Calculate path from image source to listener
            let distance = image.distance_to(listener);
            let delay = distance / self.speed_of_sound;

            // Skip if delay is too short (before direct sound)
            if delay <= direct_delay {
                continue;
            }

            // Calculate gain (distance attenuation + absorption)
            let distance_gain = 1.0 / distance.max(0.01);
            let absorption_gain = self.calculate_absorption_gain(&surface_indices);
            let gain = distance_gain * absorption_gain;

            // Direction from listener to image source
            let direction = Position::new(
                image.x - listener.x,
                image.y - listener.y,
                image.z - listener.z,
            );

            reflections.push(EarlyReflection {
                delay: delay - direct_delay, // Relative to direct sound
                gain,
                direction,
                surface_index: *surface_indices.last().unwrap_or(&0),
            });
        }
    }

    fn generate_image_sources(
        &self,
        source: &Position,
        order: i32,
    ) -> Vec<(Position, Vec<usize>)> {
        let mut images = Vec::new();

        // For simplicity, only handling first order here
        // Full implementation would recursively generate higher orders
        if order == 1 {
            // Left wall (x = 0)
            images.push((
                Position::new(-source.x, source.y, source.z),
                vec![0],
            ));

            // Right wall (x = width)
            images.push((
                Position::new(2.0 * self.room.width - source.x, source.y, source.z),
                vec![1],
            ));

            // Floor (y = 0)
            images.push((
                Position::new(source.x, -source.y, source.z),
                vec![2],
            ));

            // Ceiling (y = height)
            images.push((
                Position::new(source.x, 2.0 * self.room.height - source.y, source.z),
                vec![3],
            ));

            // Front wall (z = 0)
            images.push((
                Position::new(source.x, source.y, -source.z),
                vec![4],
            ));

            // Back wall (z = depth)
            images.push((
                Position::new(source.x, source.y, 2.0 * self.room.depth - source.z),
                vec![5],
            ));
        }

        images
    }

    fn calculate_absorption_gain(&self, surface_indices: &[usize]) -> f32 {
        let mut gain = 1.0;
        for &idx in surface_indices {
            if idx < self.room.materials.len() {
                gain *= 1.0 - self.room.materials[idx].average_absorption();
            }
        }
        gain.sqrt() // Convert energy to amplitude
    }
}

/// Feedback delay network reverb
pub struct FDNReverb {
    sample_rate: u32,
    /// Delay lines
    delay_lines: Vec<Vec<f32>>,
    delay_indices: Vec<usize>,
    delay_lengths: Vec<usize>,
    /// Feedback matrix coefficients
    feedback_matrix: Vec<Vec<f32>>,
    /// Input gains
    input_gains: Vec<f32>,
    /// Output gains
    output_gains: Vec<f32>,
    /// Lowpass filter states
    lowpass_states: Vec<f32>,
    /// Decay rate
    decay: f32,
    /// Damping (high-frequency absorption)
    damping: f32,
}

impl FDNReverb {
    pub fn new(sample_rate: u32, rt60: f32) -> Self {
        let num_delays = 8;

        // Prime-based delay lengths for density
        let delay_lengths: Vec<usize> = vec![
            (sample_rate as f32 * 0.0293).round() as usize,  // 29.3ms
            (sample_rate as f32 * 0.0371).round() as usize,  // 37.1ms
            (sample_rate as f32 * 0.0411).round() as usize,  // 41.1ms
            (sample_rate as f32 * 0.0437).round() as usize,  // 43.7ms
            (sample_rate as f32 * 0.0531).round() as usize,  // 53.1ms
            (sample_rate as f32 * 0.0587).round() as usize,  // 58.7ms
            (sample_rate as f32 * 0.0647).round() as usize,  // 64.7ms
            (sample_rate as f32 * 0.0709).round() as usize,  // 70.9ms
        ];

        let max_length = *delay_lengths.iter().max().unwrap_or(&1024);

        let mut reverb = Self {
            sample_rate,
            delay_lines: vec![vec![0.0; max_length]; num_delays],
            delay_indices: vec![0; num_delays],
            delay_lengths,
            feedback_matrix: Self::hadamard_matrix(num_delays),
            input_gains: vec![1.0 / (num_delays as f32).sqrt(); num_delays],
            output_gains: vec![1.0 / (num_delays as f32).sqrt(); num_delays],
            lowpass_states: vec![0.0; num_delays],
            decay: 0.0,
            damping: 0.5,
        };

        reverb.set_rt60(rt60);
        reverb
    }

    /// Set reverb time
    pub fn set_rt60(&mut self, rt60: f32) {
        // Calculate decay per sample
        // RT60 is time for -60dB decay
        let avg_delay = self.delay_lengths.iter().sum::<usize>() as f32
            / self.delay_lengths.len() as f32;
        let loops_per_rt60 = rt60 * self.sample_rate as f32 / avg_delay;
        self.decay = 10.0_f32.powf(-3.0 / loops_per_rt60);
    }

    /// Set high-frequency damping (0-1)
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.clamp(0.0, 0.99);
    }

    /// Process mono input to stereo output
    pub fn process(&mut self, input: f32) -> (f32, f32) {
        // Read from delay lines
        let mut outputs: Vec<f32> = self
            .delay_lines
            .iter()
            .zip(self.delay_indices.iter())
            .zip(self.delay_lengths.iter())
            .map(|((line, &idx), &len)| {
                let read_idx = (idx + line.len() - len) % line.len();
                line[read_idx]
            })
            .collect();

        // Apply damping (one-pole lowpass)
        for (i, output) in outputs.iter_mut().enumerate() {
            self.lowpass_states[i] =
                self.lowpass_states[i] * self.damping + *output * (1.0 - self.damping);
            *output = self.lowpass_states[i];
        }

        // Apply feedback matrix
        let mut feedback: Vec<f32> = vec![0.0; outputs.len()];
        for (i, fb) in feedback.iter_mut().enumerate() {
            for (j, &out) in outputs.iter().enumerate() {
                *fb += self.feedback_matrix[i][j] * out * self.decay;
            }
        }

        // Write to delay lines (input + feedback)
        for (i, line) in self.delay_lines.iter_mut().enumerate() {
            let write_sample = input * self.input_gains[i] + feedback[i];
            line[self.delay_indices[i]] = write_sample;
            self.delay_indices[i] = (self.delay_indices[i] + 1) % line.len();
        }

        // Mix outputs to stereo
        let mut left = 0.0;
        let mut right = 0.0;
        for (i, &out) in outputs.iter().enumerate() {
            if i % 2 == 0 {
                left += out * self.output_gains[i];
            } else {
                right += out * self.output_gains[i];
            }
        }

        (left, right)
    }

    /// Generate Hadamard matrix for feedback
    fn hadamard_matrix(size: usize) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0; size]; size];
        let norm = 1.0 / (size as f32).sqrt();

        // Simple Hadamard construction
        for i in 0..size {
            for j in 0..size {
                let sign = (i & j).count_ones() % 2;
                matrix[i][j] = if sign == 0 { norm } else { -norm };
            }
        }

        matrix
    }
}

/// Room impulse response generator
pub struct RoomIRGenerator {
    room: Room,
    sample_rate: u32,
    image_source: ImageSourceMethod,
    fdn_reverb: FDNReverb,
}

impl RoomIRGenerator {
    pub fn new(room: Room, sample_rate: u32) -> Self {
        let rt60 = room.rt60();
        let image_source = ImageSourceMethod::new(room.clone(), 2, sample_rate);
        let fdn_reverb = FDNReverb::new(sample_rate, rt60);

        Self {
            room,
            sample_rate,
            image_source,
            fdn_reverb,
        }
    }

    /// Generate room impulse response
    pub fn generate_ir(&mut self, source: &Position, listener: &Position, length_seconds: f32) -> (Vec<f32>, Vec<f32>) {
        let length_samples = (length_seconds * self.sample_rate as f32) as usize;
        let mut left = vec![0.0f32; length_samples];
        let mut right = vec![0.0f32; length_samples];

        // Direct sound
        let direct_distance = source.distance_to(listener);
        let direct_delay_samples =
            (direct_distance / 343.0 * self.sample_rate as f32) as usize;
        let direct_gain = 1.0 / direct_distance.max(0.1);

        if direct_delay_samples < length_samples {
            left[direct_delay_samples] = direct_gain;
            right[direct_delay_samples] = direct_gain;
        }

        // Early reflections
        let reflections = self.image_source.calculate_reflections(source, listener);
        for reflection in reflections {
            let delay_samples = ((reflection.delay + direct_distance / 343.0)
                * self.sample_rate as f32) as usize;
            if delay_samples < length_samples {
                // Simple panning based on direction
                let pan = reflection.direction.x / (reflection.direction.x.abs() + 0.001);
                let left_gain = reflection.gain * (1.0 - pan * 0.3);
                let right_gain = reflection.gain * (1.0 + pan * 0.3);

                left[delay_samples] += left_gain;
                right[delay_samples] += right_gain;
            }
        }

        // Late reverb (FDN)
        let late_start = (0.08 * self.sample_rate as f32) as usize; // 80ms
        for i in late_start..length_samples {
            // Sparse noise excitation
            let excite = if i % 50 == 0 { 0.01 } else { 0.0 };
            let (l, r) = self.fdn_reverb.process(excite);
            left[i] += l;
            right[i] += r;
        }

        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_rt60() {
        // Use a room with absorptive materials for realistic RT60
        let room = Room::new(10.0, 3.0, 8.0)
            .with_uniform_material(Material::acoustic_tile());
        let rt60 = room.rt60();
        // Typical treated room RT60 should be between 0.1 and 2 seconds
        assert!(rt60 > 0.1 && rt60 < 2.0, "RT60 was {}", rt60);
    }

    #[test]
    fn test_material_absorption() {
        let carpet = Material::carpet();
        let concrete = Material::concrete();
        // Carpet should absorb more than concrete
        assert!(carpet.average_absorption() > concrete.average_absorption());
    }

    #[test]
    fn test_image_source() {
        let room = Room::default();
        let ism = ImageSourceMethod::new(room, 1, 44100);

        let source = Position::new(2.0, 1.5, 2.0);
        let listener = Position::new(3.0, 1.5, 3.0);

        let reflections = ism.calculate_reflections(&source, &listener);
        assert!(!reflections.is_empty());
    }

    #[test]
    fn test_fdn_reverb() {
        let mut reverb = FDNReverb::new(44100, 1.0);

        // Process an impulse and wait for reverb to build up
        reverb.process(1.0);
        // Process enough samples for delay lines to propagate
        for _ in 0..2000 {
            reverb.process(0.0);
        }
        let (l, r) = reverb.process(0.0);
        // After enough samples, reverb tail should be non-zero
        assert!(l != 0.0 || r != 0.0, "Expected non-zero reverb tail");
    }
}
