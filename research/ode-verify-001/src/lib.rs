//! Isolated test harness for Symthaea's actual ODE implementation.
//!
//! The HDC API below is a deterministic test stub, not the production HDC implementation.
//! This harness validates the included numerical module and its unit tests only.

pub mod hdc {
    pub mod primitive_system {
        /// Stable FNV-1a hash for deterministic test-basis naming.
        pub fn seed_from_name(name: &str) -> u64 {
            let mut hash = 0xcbf29ce484222325_u64;
            for byte in name.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash
        }
    }

    pub mod unified_hv {
        #[derive(Debug, Clone)]
        pub struct ContinuousHV {
            pub values: Vec<f32>,
        }

        impl ContinuousHV {
            pub const DEFAULT_DIM: usize = 16_384;

            pub fn zero(dim: usize) -> Self {
                Self {
                    values: vec![0.0; dim],
                }
            }

            /// Produce a deterministic pseudo-random ±1 test basis.
            pub fn random(dim: usize, seed: u64) -> Self {
                let mut state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
                let mut values = Vec::with_capacity(dim);
                for _ in 0..dim {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    values.push(if state & 1 == 0 { -1.0 } else { 1.0 });
                }
                Self { values }
            }

            pub fn from_vec(values: Vec<f32>) -> Self {
                Self { values }
            }

            pub fn dim(&self) -> usize {
                self.values.len()
            }

            pub fn similarity(&self, other: &Self) -> f32 {
                if self.values.len() != other.values.len() || self.values.is_empty() {
                    return 0.0;
                }
                let dot = self
                    .values
                    .iter()
                    .zip(&other.values)
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                let lhs_norm = self.values.iter().map(|v| v * v).sum::<f32>().sqrt();
                let rhs_norm = other.values.iter().map(|v| v * v).sum::<f32>().sqrt();
                let denominator = lhs_norm * rhs_norm;
                if denominator > 0.0 && denominator.is_finite() {
                    dot / denominator
                } else {
                    0.0
                }
            }
        }
    }
}

#[path = "../../../crates/core/symthaea-core/src/hdc/dynamical_system.rs"]
pub mod dynamical_system;
