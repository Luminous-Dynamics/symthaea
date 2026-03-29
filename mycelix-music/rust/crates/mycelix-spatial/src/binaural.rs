// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binaural audio processing

use crate::hrtf::{HrtfConvolver, HrtfDatabase};
use crate::{Listener, Position, Result, SpatialError};
use std::collections::HashMap;

/// Binaural source with smooth position interpolation
pub struct BinauralSource {
    id: String,
    position: Position,
    target_position: Position,
    interpolation_speed: f32,
    convolver: HrtfConvolver,
    gain: f32,
}

impl BinauralSource {
    pub fn new(id: impl Into<String>, position: Position, hrtf: HrtfDatabase, block_size: usize) -> Self {
        let (azimuth, elevation, _) = position.to_spherical();
        let mut convolver = HrtfConvolver::new(hrtf, block_size);
        convolver.set_direction(azimuth, elevation);

        Self {
            id: id.into(),
            position,
            target_position: position,
            interpolation_speed: 0.1,
            convolver,
            gain: 1.0,
        }
    }

    pub fn set_position(&mut self, position: Position) {
        self.target_position = position;
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.max(0.0);
    }

    /// Process mono input to binaural stereo
    pub fn process(&mut self, input: &[f32], listener: &Listener) -> (Vec<f32>, Vec<f32>) {
        // Interpolate position
        self.position.x += (self.target_position.x - self.position.x) * self.interpolation_speed;
        self.position.y += (self.target_position.y - self.position.y) * self.interpolation_speed;
        self.position.z += (self.target_position.z - self.position.z) * self.interpolation_speed;

        // Transform to listener space
        let relative = listener.world_to_listener(&self.position);
        let (azimuth, elevation, distance) = relative.to_spherical();

        // Update convolver direction
        self.convolver.set_direction(azimuth, elevation);

        // Apply distance attenuation
        let attenuation = if distance > 1.0 {
            1.0 / distance
        } else {
            1.0
        };

        let gained_input: Vec<f32> = input.iter().map(|s| s * self.gain * attenuation).collect();

        self.convolver.process(&gained_input)
    }
}

/// Binaural renderer managing multiple sources
pub struct BinauralRenderer {
    sources: HashMap<String, BinauralSource>,
    listener: Listener,
    hrtf: HrtfDatabase,
    block_size: usize,
    sample_rate: u32,
}

impl BinauralRenderer {
    pub fn new(sample_rate: u32, block_size: usize) -> Self {
        Self {
            sources: HashMap::new(),
            listener: Listener::default(),
            hrtf: HrtfDatabase::default_mit(),
            block_size,
            sample_rate,
        }
    }

    pub fn with_hrtf(mut self, hrtf: HrtfDatabase) -> Self {
        self.hrtf = hrtf;
        self
    }

    pub fn add_source(&mut self, id: impl Into<String>, position: Position) -> &mut BinauralSource {
        let id = id.into();
        let source = BinauralSource::new(&id, position, self.hrtf.clone(), self.block_size);
        self.sources.insert(id.clone(), source);
        self.sources.get_mut(&id).unwrap()
    }

    pub fn remove_source(&mut self, id: &str) {
        self.sources.remove(id);
    }

    pub fn get_source_mut(&mut self, id: &str) -> Option<&mut BinauralSource> {
        self.sources.get_mut(id)
    }

    pub fn set_listener(&mut self, listener: Listener) {
        self.listener = listener;
    }

    /// Process all sources and mix to stereo
    pub fn process(&mut self, inputs: &HashMap<String, Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
        let block_size = inputs
            .values()
            .next()
            .map(|v| v.len())
            .unwrap_or(self.block_size);

        let mut left_mix = vec![0.0f32; block_size];
        let mut right_mix = vec![0.0f32; block_size];

        for (id, input) in inputs {
            if let Some(source) = self.sources.get_mut(id) {
                let (left, right) = source.process(input, &self.listener);

                for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
                    if i < block_size {
                        left_mix[i] += l;
                        right_mix[i] += r;
                    }
                }
            }
        }

        (left_mix, right_mix)
    }
}

/// Binaural panner for simple left/right positioning
pub struct BinauralPanner {
    sample_rate: u32,
    /// Crossfeed amount (0-1)
    crossfeed: f32,
    /// Delay for crossfeed in samples
    crossfeed_delay: usize,
    /// Delay buffers
    delay_left: Vec<f32>,
    delay_right: Vec<f32>,
    delay_index: usize,
}

impl BinauralPanner {
    pub fn new(sample_rate: u32) -> Self {
        // ITD is approximately 0.6ms maximum
        let max_delay = (sample_rate as f32 * 0.001) as usize;

        Self {
            sample_rate,
            crossfeed: 0.3,
            crossfeed_delay: max_delay / 2,
            delay_left: vec![0.0; max_delay],
            delay_right: vec![0.0; max_delay],
            delay_index: 0,
        }
    }

    pub fn set_crossfeed(&mut self, amount: f32) {
        self.crossfeed = amount.clamp(0.0, 1.0);
    }

    /// Pan mono source to stereo with binaural cues
    pub fn process(&mut self, input: &[f32], pan: f32) -> (Vec<f32>, Vec<f32>) {
        let pan = pan.clamp(-1.0, 1.0);

        let mut left = Vec::with_capacity(input.len());
        let mut right = Vec::with_capacity(input.len());

        // Calculate gains and delays based on pan
        let left_gain = ((1.0 - pan) / 2.0).sqrt();
        let right_gain = ((1.0 + pan) / 2.0).sqrt();

        // ITD: delay the further ear
        let itd_samples = ((pan.abs() * self.crossfeed_delay as f32) as usize).min(self.delay_left.len() - 1);

        for &sample in input {
            // Direct signal
            let direct_left = sample * left_gain;
            let direct_right = sample * right_gain;

            // Crossfeed (opposite channel, delayed and attenuated)
            let crossfeed_left = self.delay_right[self.delay_index] * self.crossfeed * right_gain;
            let crossfeed_right = self.delay_left[self.delay_index] * self.crossfeed * left_gain;

            // Apply ITD
            if pan > 0.0 {
                // Source on right, delay left
                self.delay_left[self.delay_index] = direct_left;
                let delayed_idx = (self.delay_index + self.delay_left.len() - itd_samples) % self.delay_left.len();
                left.push(self.delay_left[delayed_idx] + crossfeed_left);
                right.push(direct_right + crossfeed_right);
            } else {
                // Source on left, delay right
                self.delay_right[self.delay_index] = direct_right;
                let delayed_idx = (self.delay_index + self.delay_right.len() - itd_samples) % self.delay_right.len();
                left.push(direct_left + crossfeed_left);
                right.push(self.delay_right[delayed_idx] + crossfeed_right);
            }

            self.delay_left[self.delay_index] = direct_left;
            self.delay_right[self.delay_index] = direct_right;
            self.delay_index = (self.delay_index + 1) % self.delay_left.len();
        }

        (left, right)
    }
}

/// Stereo widener using mid-side processing and binaural cues
pub struct StereoWidener {
    width: f32,
    crossfeed_panner: BinauralPanner,
}

impl StereoWidener {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            width: 1.0,
            crossfeed_panner: BinauralPanner::new(sample_rate),
        }
    }

    pub fn set_width(&mut self, width: f32) {
        self.width = width.clamp(0.0, 2.0);
    }

    /// Process stereo input with width adjustment
    pub fn process(&mut self, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut out_left = Vec::with_capacity(left.len());
        let mut out_right = Vec::with_capacity(right.len());

        for (&l, &r) in left.iter().zip(right.iter()) {
            // Mid-side encoding
            let mid = (l + r) * 0.5;
            let side = (l - r) * 0.5;

            // Adjust width
            let adjusted_side = side * self.width;

            // Mid-side decoding
            out_left.push(mid + adjusted_side);
            out_right.push(mid - adjusted_side);
        }

        (out_left, out_right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binaural_panner() {
        let mut panner = BinauralPanner::new(44100);
        let input = vec![1.0; 100];

        // Center pan should give equal output
        let (left, right) = panner.process(&input, 0.0);
        assert!((left[50] - right[50]).abs() < 0.1);

        // Full right pan
        let (left, right) = panner.process(&input, 1.0);
        assert!(right[50] > left[50]);
    }

    #[test]
    fn test_stereo_widener() {
        let mut widener = StereoWidener::new(44100);

        let left = vec![1.0; 100];
        let right = vec![0.5; 100];

        // Width 0 should give mono
        widener.set_width(0.0);
        let (out_l, out_r) = widener.process(&left, &right);
        assert!((out_l[50] - out_r[50]).abs() < 0.01);

        // Width 1 should preserve original
        widener.set_width(1.0);
        let (out_l, out_r) = widener.process(&left, &right);
        assert!((out_l[50] - left[50]).abs() < 0.01);
    }
}
