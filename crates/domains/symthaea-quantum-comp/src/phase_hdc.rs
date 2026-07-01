//! Quantum-inspired phase hypervector operations.
//!
//! These are classical simulations of phase-like representations. They are not
//! hardware quantum states.

use crate::classical_hdc::BinaryHypervector;
use crate::errors::{QuantumCompError, Result};
use crate::rng::XorShift64;

const TAU: f32 = core::f32::consts::PI * 2.0;

/// Hypervector represented as angles on the unit circle.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseHypervector {
    phases: Vec<f32>,
}

impl PhaseHypervector {
    /// Creates deterministic random phases in `[0, 2π)`.
    pub fn random(dimension: usize, seed: u64) -> Result<Self> {
        if dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        let mut rng = XorShift64::new(seed);
        let phases = (0..dimension).map(|_| rng.next_f32() * TAU).collect();
        Ok(Self { phases })
    }

    /// Creates phases from a vector, wrapping all values into `[0, 2π)`.
    pub fn from_phases(phases: Vec<f32>) -> Result<Self> {
        if phases.is_empty() {
            return Err(QuantumCompError::InvalidDimension);
        }
        Ok(Self {
            phases: phases.into_iter().map(wrap_phase).collect(),
        })
    }

    /// Converts a binary hypervector into phases: `0 -> 0`, `1 -> π`.
    pub fn from_binary(binary: &BinaryHypervector) -> Self {
        let mut phases = Vec::with_capacity(binary.dimension());
        for i in 0..binary.dimension() {
            phases.push(if binary.bit(i).unwrap_or(false) {
                core::f32::consts::PI
            } else {
                0.0
            });
        }
        Self { phases }
    }

    /// Quantizes the phase vector back into a binary hypervector by phase half-plane.
    pub fn to_binary_halfplane(&self) -> Result<BinaryHypervector> {
        let mut out = BinaryHypervector::zeros(self.dimension())?;
        for (i, phase) in self.phases.iter().enumerate() {
            let bit = *phase >= core::f32::consts::PI;
            out.set_bit(i, bit)?;
        }
        Ok(out)
    }

    /// Returns the dimension.
    pub fn dimension(&self) -> usize {
        self.phases.len()
    }

    /// Returns phase slice.
    pub fn phases(&self) -> &[f32] {
        &self.phases
    }

    /// Phase binding by angle addition modulo `2π`.
    pub fn bind_phase(&self, other: &Self) -> Result<Self> {
        self.check_same_dimension(other)?;
        let phases = self
            .phases
            .iter()
            .zip(&other.phases)
            .map(|(a, b)| wrap_phase(a + b))
            .collect();
        Ok(Self { phases })
    }

    /// Phase unbinding by angle subtraction modulo `2π`.
    pub fn unbind_phase(&self, key: &Self) -> Result<Self> {
        self.check_same_dimension(key)?;
        let phases = self
            .phases
            .iter()
            .zip(&key.phases)
            .map(|(a, b)| wrap_phase(a - b))
            .collect();
        Ok(Self { phases })
    }

    /// Circular similarity in `[0, 1]` based on mean cosine agreement.
    pub fn circular_similarity(&self, other: &Self) -> Result<f32> {
        self.check_same_dimension(other)?;
        let mean = self
            .phases
            .iter()
            .zip(&other.phases)
            .map(|(a, b)| (a - b).cos())
            .sum::<f32>()
            / self.dimension() as f32;
        Ok((mean + 1.0) * 0.5)
    }

    /// Mean resultant length of the phase distribution in `[0, 1]`.
    ///
    /// Values near zero indicate broadly distributed phase angles. Values near one
    /// indicate phase concentration. This is useful as a simple coherence proxy.
    pub fn mean_resultant_length(&self) -> f32 {
        let n = self.dimension() as f32;
        let sin_sum = self.phases.iter().map(|p| p.sin()).sum::<f32>();
        let cos_sum = self.phases.iter().map(|p| p.cos()).sum::<f32>();
        (sin_sum.mul_add(sin_sum, cos_sum * cos_sum)).sqrt() / n
    }

    /// Adds Gaussian-like phase jitter using a cheap sum-of-uniforms approximation.
    pub fn with_phase_noise(&self, sigma: f32, seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);
        let phases = self
            .phases
            .iter()
            .map(|p| {
                let approx_normal = (rng.next_f32() + rng.next_f32() + rng.next_f32()) - 1.5;
                wrap_phase(*p + approx_normal * sigma)
            })
            .collect();
        Self { phases }
    }

    fn check_same_dimension(&self, other: &Self) -> Result<()> {
        if self.dimension() != other.dimension() {
            return Err(QuantumCompError::DimensionMismatch {
                expected: self.dimension(),
                actual: other.dimension(),
            });
        }
        Ok(())
    }
}

fn wrap_phase(x: f32) -> f32 {
    let mut y = x % TAU;
    if y < 0.0 {
        y += TAU;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_binding_round_trips() {
        let a = PhaseHypervector::random(256, 10).unwrap();
        let key = PhaseHypervector::random(256, 11).unwrap();
        let bound = a.bind_phase(&key).unwrap();
        let recovered = bound.unbind_phase(&key).unwrap();
        assert!(a.circular_similarity(&recovered).unwrap() > 0.999);
    }
}
