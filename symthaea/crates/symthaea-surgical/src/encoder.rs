// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use symthaea_core::genesis::GenesisSeed; use symthaea_core::hdc::ContinuousHV;
use crate::types::{SurgicalState, NUM_STATE_CHANNELS};
const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
const R: [[f32;2]; NUM_STATE_CHANNELS] = [[-1.5,1.5],[-1.2,1.2],[-0.8,0.8],[-1.5,1.5],[-1.2,1.2],[-0.8,0.8],[-5.0,5.0],[-5.0,5.0],[-5.0,5.0],[-5.0,5.0],[-5.0,5.0],[-5.0,5.0],[-200.0,200.0],[-200.0,200.0],[-200.0,200.0],[-10.0,10.0],[-10.0,10.0],[-10.0,10.0],[0.0,1.0],[0.0,1.0],[0.0,50.0],[0.0,1.0],[0.0,10.0],[0.0,20.0]];
pub struct SurgicalHdcEncoder { base: Vec<ContinuousHV>, levels: Vec<ContinuousHV>, prev: Option<[f32; NUM_STATE_CHANNELS]>, nl: usize }
impl SurgicalHdcEncoder {
    pub fn new(g: &GenesisSeed, nl: usize) -> Self { Self { base: (0..NUM_STATE_CHANNELS).map(|i| ContinuousHV::from_genesis(g, &format!("surg::ch::{i}"), DIM)).collect(), levels: (0..nl).map(|l| ContinuousHV::from_genesis(g, &format!("surg::lv::{l}"), DIM)).collect(), prev: None, nl } }
    pub fn encode(&mut self, state: &SurgicalState) -> ContinuousHV {
        let ch = state.to_channels(); let mut res = ContinuousHV::zero(DIM);
        for i in 0..NUM_STATE_CHANNELS { let [lo,hi] = R[i]; let r = hi-lo; let n = if r.abs() < 1e-10 { 0.5 } else { ((ch[i]-lo)/r).clamp(0.0, 1.0) }; let k = (n * (self.nl-1) as f32).round() as usize; let k = k.min(self.nl-1); let mut lhv = ContinuousHV::zero(DIM); for l in 0..=k { lhv.add_in_place(&self.levels[l]); } lhv = lhv.normalize(); res.add_in_place(&lhv.bind(&self.base[i])); }
        self.prev = Some(ch); res.normalize()
    }
    pub fn reset(&mut self) { self.prev = None; }
}
#[cfg(test)] mod tests { use super::*; #[test] fn test_dim() { let mut e = SurgicalHdcEncoder::new(&GenesisSeed::from_phrase("t"), 32); assert_eq!(e.encode(&SurgicalState::home()).dim(), DIM); } }
