// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// One-compartment pharmacokinetic injection.
///
/// Science: Standard PK model — C(t) = dose × e^(-t/τ).
#[derive(Debug, Clone)]
pub struct ActiveInjection {
    /// Target transmitter index (0=DA, 1=NE, 2=5-HT, 3=ACh, 4=GABA, 5=Oxy, 6=Glut, 7=Aden, 8=ECB)
    pub transmitter_idx: usize,
    /// Initial dose (positive = agonist, negative = antagonist)
    pub remaining_dose: f32,
    /// Half-life in cycles
    pub half_life_cycles: u32,
    /// Elapsed cycles since injection
    pub elapsed: u32,
}

impl ActiveInjection {
    /// Current effective dose (exponential decay).
    pub fn current_dose(&self) -> f32 {
        if self.half_life_cycles == 0 {
            return 0.0;
        }
        let decay = (-0.693 * self.elapsed as f32 / self.half_life_cycles as f32).exp();
        self.remaining_dose * decay
    }

    /// Whether the injection has effectively expired (< 0.001 absolute dose).
    pub fn is_expired(&self) -> bool {
        self.current_dose().abs() < 0.001
    }
}
