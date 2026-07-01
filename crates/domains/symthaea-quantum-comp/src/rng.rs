//! Tiny deterministic RNG for reproducible examples and tests.
//!
//! This is not cryptographic randomness.

/// Deterministic xorshift64* RNG.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Creates a new deterministic generator.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    /// Returns the next `u64` sample.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Returns a float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let v = self.next_u64() >> 40;
        (v as f32) / ((1u32 << 24) as f32)
    }

    /// Returns a centered float in `[-1, 1)`.
    pub fn next_centered_f32(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Returns true with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p.clamp(0.0, 1.0)
    }

    /// Returns a deterministic integer in `0..upper`.
    pub fn next_usize(&mut self, upper: usize) -> Option<usize> {
        if upper == 0 {
            return None;
        }
        Some((self.next_u64() as usize) % upper)
    }
}
