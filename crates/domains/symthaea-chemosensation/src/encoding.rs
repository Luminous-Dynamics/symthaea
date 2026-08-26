// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Locality-preserving HDC encoding for continuous chemical measurements.
//!
//! Unlike categorical bucket encoders, neighboring scalar values should remain
//! neighboring representations. This encoder creates deterministic anchor HVs
//! across a numeric range and linearly interpolates between adjacent anchors.

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

#[derive(Debug, Clone)]
pub struct ScalarHdcEncoder {
    min: f32,
    max: f32,
    anchors: Vec<ContinuousHV>,
}

impl ScalarHdcEncoder {
    /// Create a deterministic scalar encoder.
    ///
    /// `anchor_count >= 2` and `max > min` are required. The seed defines the
    /// representation family, so distinct semantic channels should use distinct
    /// seeds or role-binding at a higher layer.
    pub fn new(min: f32, max: f32, anchor_count: usize, seed: u64) -> Self {
        assert!(min.is_finite() && max.is_finite(), "range must be finite");
        assert!(max > min, "max must be greater than min");
        assert!(anchor_count >= 2, "at least two anchors are required");

        let anchors = (0..anchor_count)
            .map(|i| {
                ContinuousHV::random(
                    HDC_DIMENSION,
                    seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                )
            })
            .collect();

        Self { min, max, anchors }
    }

    pub fn min(&self) -> f32 {
        self.min
    }

    pub fn max(&self) -> f32 {
        self.max
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// Encode a scalar, saturating values outside the configured range.
    pub fn encode(&self, value: f32) -> ContinuousHV {
        let value = if value.is_finite() {
            value.clamp(self.min, self.max)
        } else {
            self.min
        };

        let normalized = (value - self.min) / (self.max - self.min);
        let position = normalized * (self.anchors.len() - 1) as f32;
        let lower = position.floor() as usize;
        let upper = (lower + 1).min(self.anchors.len() - 1);
        let t = position - lower as f32;

        if lower == upper || t <= f32::EPSILON {
            return self.anchors[lower].clone();
        }

        let a = &self.anchors[lower].values;
        let b = &self.anchors[upper].values;
        let values = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| x * (1.0 - t) + y * t)
            .collect();

        let mut hv = ContinuousHV::from_vec(values);
        hv.l2_normalize();
        hv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_is_deterministic() {
        let a = ScalarHdcEncoder::new(0.0, 10.0, 8, 42);
        let b = ScalarHdcEncoder::new(0.0, 10.0, 8, 42);
        assert_eq!(a.encode(3.25), b.encode(3.25));
    }

    #[test]
    fn neighboring_values_are_more_similar_than_distant_values() {
        let encoder = ScalarHdcEncoder::new(0.0, 100.0, 16, 7);
        let center = encoder.encode(50.0);
        let near = encoder.encode(51.0);
        let far = encoder.encode(90.0);

        assert!(
            center.similarity(&near) > center.similarity(&far),
            "continuous chemistry values must preserve neighborhood structure"
        );
    }

    #[test]
    fn out_of_range_values_saturate() {
        let encoder = ScalarHdcEncoder::new(0.0, 1.0, 4, 99);
        assert_eq!(encoder.encode(-100.0), encoder.encode(0.0));
        assert_eq!(encoder.encode(100.0), encoder.encode(1.0));
    }

    #[test]
    fn non_finite_input_does_not_create_non_finite_hv() {
        let encoder = ScalarHdcEncoder::new(0.0, 1.0, 4, 99);
        let hv = encoder.encode(f32::NAN);
        assert_eq!(hv.values.len(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }
}
