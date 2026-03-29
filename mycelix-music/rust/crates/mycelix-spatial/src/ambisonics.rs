// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ambisonics encoding and decoding

use crate::{Listener, Orientation, Position, Result, SpatialError};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Ambisonics channel ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelOrdering {
    /// Furse-Malham (FuMa) - legacy format
    FuMa,
    /// Ambisonics Channel Number (ACN) - modern standard
    ACN,
}

/// Ambisonics normalization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Normalization {
    /// MaxN (FuMa default)
    MaxN,
    /// SN3D (ACN default)
    SN3D,
    /// N3D
    N3D,
}

/// Ambisonics order (0 = mono, 1 = first-order, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmbisonicsOrder(pub u8);

impl AmbisonicsOrder {
    pub fn new(order: u8) -> Result<Self> {
        if order > 7 {
            return Err(SpatialError::InvalidAmbisonicsOrder(order));
        }
        Ok(Self(order))
    }

    /// Number of channels for this order
    pub fn channel_count(&self) -> usize {
        let n = self.0 as usize + 1;
        n * n
    }

    pub fn value(&self) -> u8 {
        self.0
    }
}

/// First-order ambisonics encoder (B-format)
pub struct FOAEncoder {
    ordering: ChannelOrdering,
    normalization: Normalization,
}

impl FOAEncoder {
    pub fn new() -> Self {
        Self {
            ordering: ChannelOrdering::ACN,
            normalization: Normalization::SN3D,
        }
    }

    pub fn with_format(ordering: ChannelOrdering, normalization: Normalization) -> Self {
        Self {
            ordering,
            normalization,
        }
    }

    /// Encode mono source at given position to first-order ambisonics
    /// Returns [W, Y, Z, X] for ACN or [W, X, Y, Z] for FuMa
    pub fn encode(&self, input: &[f32], position: &Position) -> Vec<Vec<f32>> {
        let (azimuth, elevation, _distance) = position.to_spherical();

        // Calculate encoding coefficients
        let (w, x, y, z) = self.encoding_coefficients(azimuth, elevation);

        // Encode to 4 channels
        let mut output = vec![Vec::with_capacity(input.len()); 4];

        for &sample in input {
            match self.ordering {
                ChannelOrdering::ACN => {
                    output[0].push(sample * w); // W
                    output[1].push(sample * y); // Y
                    output[2].push(sample * z); // Z
                    output[3].push(sample * x); // X
                }
                ChannelOrdering::FuMa => {
                    output[0].push(sample * w); // W
                    output[1].push(sample * x); // X
                    output[2].push(sample * y); // Y
                    output[3].push(sample * z); // Z
                }
            }
        }

        output
    }

    fn encoding_coefficients(&self, azimuth: f32, elevation: f32) -> (f32, f32, f32, f32) {
        let cos_elev = elevation.cos();
        let sin_elev = elevation.sin();
        let cos_az = azimuth.cos();
        let sin_az = azimuth.sin();

        // Basic spherical harmonics
        let w = 1.0;
        let x = cos_elev * cos_az;
        let y = cos_elev * sin_az;
        let z = sin_elev;

        // Apply normalization
        match self.normalization {
            Normalization::SN3D => {
                // W is already 1, XYZ are already correct for SN3D
                (w, x, y, z)
            }
            Normalization::N3D => {
                // N3D = SN3D * sqrt(2l+1)
                let sqrt3 = 3.0_f32.sqrt();
                (w, x * sqrt3, y * sqrt3, z * sqrt3)
            }
            Normalization::MaxN => {
                // FuMa convention
                let sqrt2 = 2.0_f32.sqrt();
                (w / sqrt2, x, y, z)
            }
        }
    }
}

impl Default for FOAEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// First-order ambisonics decoder
pub struct FOADecoder {
    ordering: ChannelOrdering,
    normalization: Normalization,
    speaker_positions: Vec<Position>,
    decoder_matrix: Vec<Vec<f32>>,
}

impl FOADecoder {
    /// Create decoder for stereo output (binaural-like)
    pub fn stereo() -> Self {
        let speakers = vec![
            Position::from_spherical(-PI / 6.0, 0.0, 1.0), // Left 30°
            Position::from_spherical(PI / 6.0, 0.0, 1.0),  // Right 30°
        ];
        Self::new(speakers, ChannelOrdering::ACN, Normalization::SN3D)
    }

    /// Create decoder for quadraphonic output
    pub fn quad() -> Self {
        let speakers = vec![
            Position::from_spherical(-PI / 4.0, 0.0, 1.0),  // Front Left 45°
            Position::from_spherical(PI / 4.0, 0.0, 1.0),   // Front Right 45°
            Position::from_spherical(-3.0 * PI / 4.0, 0.0, 1.0), // Rear Left 135°
            Position::from_spherical(3.0 * PI / 4.0, 0.0, 1.0),  // Rear Right 135°
        ];
        Self::new(speakers, ChannelOrdering::ACN, Normalization::SN3D)
    }

    /// Create decoder for 5.1 surround
    pub fn surround_5_1() -> Self {
        let speakers = vec![
            Position::from_spherical(0.0, 0.0, 1.0),        // Center
            Position::from_spherical(-PI / 6.0, 0.0, 1.0),  // Left 30°
            Position::from_spherical(PI / 6.0, 0.0, 1.0),   // Right 30°
            Position::from_spherical(-PI * 0.6, 0.0, 1.0),  // Surround Left 110°
            Position::from_spherical(PI * 0.6, 0.0, 1.0),   // Surround Right 110°
        ];
        Self::new(speakers, ChannelOrdering::ACN, Normalization::SN3D)
    }

    pub fn new(
        speaker_positions: Vec<Position>,
        ordering: ChannelOrdering,
        normalization: Normalization,
    ) -> Self {
        let decoder_matrix = Self::calculate_decoder_matrix(&speaker_positions, ordering, normalization);

        Self {
            ordering,
            normalization,
            speaker_positions,
            decoder_matrix,
        }
    }

    fn calculate_decoder_matrix(
        speakers: &[Position],
        ordering: ChannelOrdering,
        normalization: Normalization,
    ) -> Vec<Vec<f32>> {
        let num_speakers = speakers.len();
        let mut matrix = vec![vec![0.0f32; 4]; num_speakers];

        for (i, speaker) in speakers.iter().enumerate() {
            let (azimuth, elevation, _) = speaker.to_spherical();

            let cos_elev = elevation.cos();
            let sin_elev = elevation.sin();
            let cos_az = azimuth.cos();
            let sin_az = azimuth.sin();

            // Decoding coefficients (transposed encoding)
            let w = 1.0;
            let x = cos_elev * cos_az;
            let y = cos_elev * sin_az;
            let z = sin_elev;

            // Apply normalization inverse
            let (w, x, y, z) = match normalization {
                Normalization::SN3D => (w, x, y, z),
                Normalization::N3D => {
                    let sqrt3 = 3.0_f32.sqrt();
                    (w, x / sqrt3, y / sqrt3, z / sqrt3)
                }
                Normalization::MaxN => {
                    let sqrt2 = 2.0_f32.sqrt();
                    (w * sqrt2, x, y, z)
                }
            };

            // Normalize by number of speakers
            let norm = 1.0 / num_speakers as f32;

            match ordering {
                ChannelOrdering::ACN => {
                    matrix[i][0] = w * norm;
                    matrix[i][1] = y * norm;
                    matrix[i][2] = z * norm;
                    matrix[i][3] = x * norm;
                }
                ChannelOrdering::FuMa => {
                    matrix[i][0] = w * norm;
                    matrix[i][1] = x * norm;
                    matrix[i][2] = y * norm;
                    matrix[i][3] = z * norm;
                }
            }
        }

        matrix
    }

    /// Decode FOA to speaker feeds
    pub fn decode(&self, ambi_channels: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if ambi_channels.len() != 4 {
            return vec![];
        }

        let block_size = ambi_channels[0].len();
        let num_speakers = self.speaker_positions.len();
        let mut output = vec![vec![0.0f32; block_size]; num_speakers];

        for sample_idx in 0..block_size {
            for (speaker_idx, speaker_coeffs) in self.decoder_matrix.iter().enumerate() {
                let mut sum = 0.0;
                for (ch, &coeff) in speaker_coeffs.iter().enumerate() {
                    sum += ambi_channels[ch][sample_idx] * coeff;
                }
                output[speaker_idx][sample_idx] = sum;
            }
        }

        output
    }

    pub fn speaker_count(&self) -> usize {
        self.speaker_positions.len()
    }
}

/// Higher-order ambisonics encoder
pub struct HOAEncoder {
    order: AmbisonicsOrder,
    ordering: ChannelOrdering,
    normalization: Normalization,
}

impl HOAEncoder {
    pub fn new(order: AmbisonicsOrder) -> Self {
        Self {
            order,
            ordering: ChannelOrdering::ACN,
            normalization: Normalization::SN3D,
        }
    }

    /// Encode mono source to HOA
    pub fn encode(&self, input: &[f32], position: &Position) -> Vec<Vec<f32>> {
        let num_channels = self.order.channel_count();
        let mut output = vec![Vec::with_capacity(input.len()); num_channels];

        let (azimuth, elevation, _) = position.to_spherical();
        let coefficients = self.calculate_coefficients(azimuth, elevation);

        for &sample in input {
            for (ch, &coeff) in coefficients.iter().enumerate() {
                output[ch].push(sample * coeff);
            }
        }

        output
    }

    fn calculate_coefficients(&self, azimuth: f32, elevation: f32) -> Vec<f32> {
        let mut coeffs = Vec::with_capacity(self.order.channel_count());

        for l in 0..=self.order.value() as i32 {
            for m in -l..=l {
                let y = spherical_harmonic(l, m, azimuth, elevation);

                // Apply normalization
                let normalized = match self.normalization {
                    Normalization::SN3D => y,
                    Normalization::N3D => y * ((2 * l + 1) as f32).sqrt(),
                    Normalization::MaxN => {
                        // MaxN normalization
                        if l == 0 {
                            y / 2.0_f32.sqrt()
                        } else {
                            y
                        }
                    }
                };

                coeffs.push(normalized);
            }
        }

        coeffs
    }
}

/// Calculate real spherical harmonic Y_l^m
fn spherical_harmonic(l: i32, m: i32, azimuth: f32, elevation: f32) -> f32 {
    let theta = PI / 2.0 - elevation; // Convert elevation to colatitude
    let phi = azimuth;

    // Associated Legendre polynomial P_l^|m|(cos(theta))
    let plm = associated_legendre(l, m.abs(), theta.cos());

    // Normalization factor for SN3D
    let norm = if m == 0 {
        1.0
    } else {
        let num: f32 = (1..=(l - m.abs()) as usize + 1).map(|i| i as f32).product();
        let den: f32 = (1..=(l + m.abs()) as usize + 1).map(|i| i as f32).product();
        (2.0 * num / den).sqrt()
    };

    // Real spherical harmonic
    let y = if m > 0 {
        norm * plm * (m as f32 * phi).cos() * 2.0_f32.sqrt()
    } else if m < 0 {
        norm * plm * ((-m) as f32 * phi).sin() * 2.0_f32.sqrt()
    } else {
        norm * plm
    };

    y
}

/// Associated Legendre polynomial P_l^m(x)
fn associated_legendre(l: i32, m: i32, x: f32) -> f32 {
    if m > l {
        return 0.0;
    }

    // Start with P_m^m
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = (1.0 - x * x).sqrt();
        let mut fact = 1.0;
        for _ in 1..=m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if l == m {
        return pmm;
    }

    // Compute P_{m+1}^m
    let mut pmmp1 = x * (2 * m + 1) as f32 * pmm;
    if l == m + 1 {
        return pmmp1;
    }

    // Compute P_l^m using recurrence
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        pll = (x * (2 * ll - 1) as f32 * pmmp1 - (ll + m - 1) as f32 * pmm) / (ll - m) as f32;
        pmm = pmmp1;
        pmmp1 = pll;
    }

    pll
}

/// Ambisonics rotator for head tracking
pub struct AmbisonicsRotator {
    order: AmbisonicsOrder,
    rotation_matrix: Vec<Vec<f32>>,
}

impl AmbisonicsRotator {
    pub fn new(order: AmbisonicsOrder) -> Self {
        let num_channels = order.channel_count();
        let rotation_matrix = vec![vec![0.0; num_channels]; num_channels];

        // Initialize with identity
        let mut rotator = Self {
            order,
            rotation_matrix,
        };
        rotator.set_rotation(&Orientation::identity());
        rotator
    }

    /// Update rotation matrix for new orientation
    pub fn set_rotation(&mut self, orientation: &Orientation) {
        let num_channels = self.order.channel_count();

        // For first-order, rotation is straightforward
        // For higher orders, use Wigner-D matrices (simplified here)

        // Identity for now - would implement proper rotation
        for i in 0..num_channels {
            for j in 0..num_channels {
                self.rotation_matrix[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // First-order rotation using Euler angles
        if self.order.value() >= 1 {
            let quat = orientation.to_quaternion();
            let rot = quat.to_rotation_matrix();

            // Rotate XYZ components (channels 1-3 for ACN)
            for i in 0..3 {
                for j in 0..3 {
                    self.rotation_matrix[i + 1][j + 1] = rot[(i, j)];
                }
            }
        }
    }

    /// Apply rotation to ambisonics signal
    pub fn rotate(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let num_channels = input.len().min(self.order.channel_count());
        let block_size = input.get(0).map(|v| v.len()).unwrap_or(0);

        let mut output = vec![vec![0.0f32; block_size]; num_channels];

        for sample_idx in 0..block_size {
            for out_ch in 0..num_channels {
                let mut sum = 0.0;
                for in_ch in 0..num_channels {
                    sum += input[in_ch][sample_idx] * self.rotation_matrix[out_ch][in_ch];
                }
                output[out_ch][sample_idx] = sum;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foa_encoder() {
        let encoder = FOAEncoder::new();
        let input = vec![1.0; 100];
        let pos = Position::new(1.0, 0.0, 0.0);

        let output = encoder.encode(&input, &pos);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0].len(), 100);
    }

    #[test]
    fn test_foa_decoder_stereo() {
        let decoder = FOADecoder::stereo();
        assert_eq!(decoder.speaker_count(), 2);
    }

    #[test]
    fn test_ambisonics_order() {
        let order = AmbisonicsOrder::new(1).unwrap();
        assert_eq!(order.channel_count(), 4);

        let order = AmbisonicsOrder::new(2).unwrap();
        assert_eq!(order.channel_count(), 9);
    }
}
