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

    /// Quantizes the phase vector back into a binary hypervector by nearest-symbol decision.
    ///
    /// `from_binary` encodes bit `0` at phase `0` and bit `1` at phase `π`.
    /// The decode boundary must sit at the *midpoints* between those symbols
    /// (`π/2` and `3π/2`), not at the symbols themselves — deciding by
    /// `phase >= π` (as an earlier version of this function did) put the
    /// boundary exactly on top of both symbol points, so any infinitesimal
    /// noise in the "wrong" direction flipped the bit regardless of noise
    /// magnitude, and BER jumped to ~0.5 as soon as any noise was applied at
    /// all. `cos(phase) < 0` is the correct nearest-symbol rule for
    /// antipodal symbols at `0`/`π`: it is the sign of the projection onto
    /// the symbol axis, positive near `0`, negative near `π`. Found via
    /// `symthaea-quantum-comp`'s calibrated cross-representation comparison
    /// (`calibrated_comparison.rs`), which was the first code in this crate
    /// to actually exercise this function.
    pub fn to_binary_halfplane(&self) -> Result<BinaryHypervector> {
        let mut out = BinaryHypervector::zeros(self.dimension())?;
        for (i, phase) in self.phases.iter().enumerate() {
            let bit = phase.cos() < 0.0;
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

    /// Circular mean angle across this hypervector's dimensions, in `[0, 2π)`.
    ///
    /// Collapses this vector's `dimension` per-dimension angle readings into
    /// a single scalar estimate — the phase-representation analog of
    /// collapsing `dimension` popcounts into a fraction in
    /// `BinaryHypervector::thermometer_decode`. Meant for vectors where every
    /// dimension carries a (possibly noisy) reading of the *same* underlying
    /// quantity, e.g. a constant-angle vector built with `from_phases` and
    /// then perturbed by `with_phase_noise`.
    pub fn circular_mean(&self) -> f32 {
        let sin_sum: f32 = self.phases.iter().map(|p| p.sin()).sum();
        let cos_sum: f32 = self.phases.iter().map(|p| p.cos()).sum();
        wrap_phase(sin_sum.atan2(cos_sum))
    }

    /// Bundles a nonempty slice of phase hypervectors via per-dimension circular mean.
    ///
    /// This is the phase-representation analog of
    /// `BinaryHypervector::majority_bundle`: each output dimension's angle is
    /// the circular mean (mean resultant direction) of that dimension's angle
    /// across all input vectors — the natural continuous generalization of a
    /// per-dimension majority vote for angles, and the standard bundling rule
    /// for phase/holographic representations.
    pub fn circular_bundle(vectors: &[Self]) -> Result<Self> {
        if vectors.is_empty() {
            return Err(QuantumCompError::InvalidConfig(
                "bundle requires at least one vector",
            ));
        }
        let dimension = vectors[0].dimension();
        for v in vectors {
            if v.dimension() != dimension {
                return Err(QuantumCompError::DimensionMismatch {
                    expected: dimension,
                    actual: v.dimension(),
                });
            }
        }
        let mut phases = Vec::with_capacity(dimension);
        for d in 0..dimension {
            let mut sin_sum = 0.0f32;
            let mut cos_sum = 0.0f32;
            for v in vectors {
                sin_sum += v.phases[d].sin();
                cos_sum += v.phases[d].cos();
            }
            phases.push(wrap_phase(sin_sum.atan2(cos_sum)));
        }
        Ok(Self { phases })
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

    #[test]
    fn circular_mean_of_a_constant_vector_recovers_the_constant() {
        for theta in [0.0f32, 1.0, core::f32::consts::PI, 5.5] {
            let v = PhaseHypervector::from_phases(vec![theta; 512]).unwrap();
            let mean = v.circular_mean();
            let diff = (mean - wrap_phase(theta)).abs();
            assert!(
                diff < 1e-4 || (TAU - diff) < 1e-4,
                "theta={theta} mean={mean}"
            );
        }
    }

    #[test]
    fn circular_mean_averages_out_small_symmetric_noise() {
        let base = PhaseHypervector::from_phases(vec![core::f32::consts::PI; 4096]).unwrap();
        let noisy = base.with_phase_noise(0.3, 42);
        let mean = noisy.circular_mean();
        assert!(
            (mean - core::f32::consts::PI).abs() < 0.05,
            "mean={mean} should be close to PI after averaging 4096 noisy readings"
        );
    }

    #[test]
    fn circular_bundle_of_identical_copies_reproduces_the_vector() {
        let a = PhaseHypervector::random(128, 3).unwrap();
        let bundled =
            PhaseHypervector::circular_bundle(&[a.clone(), a.clone(), a.clone()]).unwrap();
        assert!(a.circular_similarity(&bundled).unwrap() > 0.999);
    }

    #[test]
    fn circular_bundle_rejects_empty_and_mismatched_dimensions() {
        assert!(PhaseHypervector::circular_bundle(&[]).is_err());
        let a = PhaseHypervector::random(64, 1).unwrap();
        let b = PhaseHypervector::random(32, 2).unwrap();
        assert!(PhaseHypervector::circular_bundle(&[a, b]).is_err());
    }

    #[test]
    fn circular_bundle_member_is_more_similar_than_a_random_foil() {
        let members: Vec<_> = (0..8)
            .map(|i| PhaseHypervector::random(1024, 100 + i))
            .collect::<Result<_>>()
            .unwrap();
        let bundle = PhaseHypervector::circular_bundle(&members).unwrap();
        let foil = PhaseHypervector::random(1024, 999).unwrap();
        let member_sim = members[0].circular_similarity(&bundle).unwrap();
        let foil_sim = foil.circular_similarity(&bundle).unwrap();
        assert!(
            member_sim > foil_sim,
            "member_sim={member_sim} foil_sim={foil_sim}"
        );
    }

    #[test]
    fn to_binary_halfplane_decodes_clean_symbols_exactly() {
        let original = BinaryHypervector::random(1024, 42).unwrap();
        let phases = PhaseHypervector::from_binary(&original);
        let decoded = phases.to_binary_halfplane().unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn to_binary_halfplane_tolerates_small_noise_near_either_symbol() {
        // Regression test: an earlier version of `to_binary_halfplane` decided
        // `bit = phase >= PI`, putting the decode boundary exactly on top of
        // both encoded symbols (0 and PI), so *any* nonzero noise in the
        // "wrong" direction flipped the bit regardless of magnitude. The
        // correct boundary sits at the symbol midpoints (PI/2, 3*PI/2).
        let small_noise = 0.01_f32; // well within either symbol's basin
        for &phase in &[0.0_f32, core::f32::consts::PI] {
            for &delta in &[-small_noise, small_noise] {
                let noisy = PhaseHypervector::from_phases(vec![phase + delta]).unwrap();
                let decoded = noisy.to_binary_halfplane().unwrap();
                let expected = (core::f32::consts::FRAC_PI_2..3.0 * core::f32::consts::FRAC_PI_2)
                    .contains(&phase);
                assert_eq!(
                    decoded.bit(0).unwrap(),
                    expected,
                    "phase={phase} delta={delta} should not flip on tiny noise"
                );
            }
        }
    }
}
