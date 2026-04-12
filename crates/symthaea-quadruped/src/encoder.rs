use crate::types::{QuadrupedState, NUM_STATE_CHANNELS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
const R: [[f32; 2]; NUM_STATE_CHANNELS] = [
    [-10.0, 10.0],
    [-10.0, 10.0],
    [0.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-5.0, 5.0],
    [-5.0, 5.0],
    [-5.0, 5.0],
    [-5.0, 5.0],
    [-5.0, 5.0],
    [-5.0, 5.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
];
pub struct QuadrupedHdcEncoder {
    base: Vec<ContinuousHV>,
    levels: Vec<ContinuousHV>,
    nl: usize,
}
impl QuadrupedHdcEncoder {
    pub fn new(g: &GenesisSeed, nl: usize) -> Self {
        Self {
            base: (0..NUM_STATE_CHANNELS)
                .map(|i| ContinuousHV::from_genesis(g, &format!("quad::ch::{i}"), DIM))
                .collect(),
            levels: (0..nl)
                .map(|l| ContinuousHV::from_genesis(g, &format!("quad::lv::{l}"), DIM))
                .collect(),
            nl,
        }
    }
    pub fn encode(&mut self, s: &QuadrupedState) -> ContinuousHV {
        let ch = s.to_channels();
        let mut res = ContinuousHV::zero(DIM);
        for i in 0..NUM_STATE_CHANNELS {
            let [lo, hi] = R[i];
            let r = hi - lo;
            let n = if r.abs() < 1e-10 {
                0.5
            } else {
                ((ch[i] - lo) / r).clamp(0.0, 1.0)
            };
            let k = (n * (self.nl - 1) as f32).round() as usize;
            let k = k.min(self.nl - 1);
            let mut lhv = ContinuousHV::zero(DIM);
            for l in 0..=k {
                lhv.add_in_place(&self.levels[l]);
            }
            lhv = lhv.normalize();
            res.add_in_place(&lhv.bind(&self.base[i]));
        }
        res.normalize()
    }
    pub fn reset(&mut self) {}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dim() {
        let mut e = QuadrupedHdcEncoder::new(&GenesisSeed::from_phrase("t"), 32);
        assert_eq!(e.encode(&QuadrupedState::standing()).dim(), DIM);
    }
}
