//! Tiny deterministic PRNG (SplitMix64) so every statistic in this crate is
//! exactly reproducible from a seed — the null-model draws in `flavor_network`
//! must be falsifiable and stable across runs, not `rand`-nondeterministic.

/// SplitMix64 (Steele et al. 2014). Fast, seedable, good enough for Monte-Carlo
/// null models; not for cryptography.
#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, n). Returns 0 if `n == 0`.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    pub fn unit(&mut self) -> f64 {
        // 53-bit mantissa
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_from_seed() {
        let a: Vec<u64> = (0..5)
            .scan(SplitMix64::new(1), |r, _| Some(r.next_u64()))
            .collect();
        let b: Vec<u64> = (0..5)
            .scan(SplitMix64::new(1), |r, _| Some(r.next_u64()))
            .collect();
        assert_eq!(a, b);
        let c: Vec<u64> = (0..5)
            .scan(SplitMix64::new(2), |r, _| Some(r.next_u64()))
            .collect();
        assert_ne!(a, c);
    }

    #[test]
    fn unit_in_range() {
        let mut r = SplitMix64::new(99);
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
