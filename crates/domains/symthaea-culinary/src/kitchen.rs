//! A minimal, honest kitchen simulator — the observation source for Phase 3's
//! active-inference palate. There is no real pH/temperature/viscosity hardware
//! here, so per `CULINARY_PLAN_2026-07-09.md` Phase 3, the physical simulator
//! itself *is* the sensor: every action genuinely mutates real conserved
//! quantities, and every observation is read back from that mutated state.
//!
//! Reuses Phase 2's own physics rather than re-deriving it: core-temperature
//! relaxation is [`NewtonCooling`] (Newton's law is symmetric — the same closed
//! form heats toward a hot burner exactly as it cools toward room temperature),
//! and the dispersed-phase fraction φ is tracked via mass-conserving water/oil
//! volumes so [`crate::dynamics::emulsion_relative_viscosity`] and the Phase-1
//! random-close-packing bound apply unchanged.

use crate::dynamics::NewtonCooling;

/// Five real actions a cook (or the palate) can take on the pot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KitchenAction {
    HeatUp,
    HeatDown,
    AddWater,
    AddAcid,
    AddSalt,
}

impl KitchenAction {
    pub const ALL: [KitchenAction; 5] = [
        KitchenAction::HeatUp,
        KitchenAction::HeatDown,
        KitchenAction::AddWater,
        KitchenAction::AddAcid,
        KitchenAction::AddSalt,
    ];

    pub fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|a| *a == self)
            .expect("action in ALL")
    }

    pub fn from_index(i: usize) -> Option<Self> {
        Self::ALL.get(i).copied()
    }
}

/// Bounds and step sizes for the simulated pot. Not a precision culinary model —
/// a deliberately simple, physically-consistent toy: real conservation
/// (water volume evaporates, dispersed volume does not) driving a real
/// derived quantity (φ) that Phase 1/2's actual validators and viscosity model
/// apply to unchanged.
pub const TEMP_MIN_C: f64 = 20.0;
pub const TEMP_MAX_C: f64 = 100.0;
pub const PH_MIN: f64 = 2.0;
pub const PH_MAX: f64 = 9.0;
pub const SALINITY_MAX_PCT: f64 = 8.0;

const HEAT_STEP_C: f64 = 20.0;
const ADD_WATER_VOL: f64 = 0.15;
const ACID_STEP: f64 = 0.6;
const SALT_STEP_PCT: f64 = 0.3;
/// Evaporation rate (1/s) at full boil (≥100 °C); ramps linearly from 0 at 60 °C.
const EVAP_RATE_MAX_PER_S: f64 = 0.0020;

/// The pot's physical state.
#[derive(Clone, Copy, Debug)]
pub struct KitchenState {
    pub core_temp_c: f64,
    /// What the burner is currently set to; core temp relaxes toward this.
    pub heat_setpoint_c: f64,
    /// Volume of the continuous (water) phase — this is what evaporates.
    pub v_water: f64,
    /// Volume of the dispersed (oil/fat) phase — conserved; does not evaporate.
    pub v_dispersed: f64,
    pub ph: f64,
    pub salinity_pct: f64,
}

impl KitchenState {
    /// Dispersed-phase fraction φ = V_dispersed / (V_dispersed + V_water).
    /// Rises toward 1.0 as the water phase boils away — the same physical
    /// event Phase 1's `validate_emulsion` calls "breaking."
    pub fn phi(&self) -> f64 {
        let total = self.v_dispersed + self.v_water;
        if total <= 0.0 {
            1.0
        } else {
            self.v_dispersed / total
        }
    }
}

/// The simulated pot plus its physics-timestep.
#[derive(Clone, Copy, Debug)]
pub struct Kitchen {
    pub state: KitchenState,
    pub dt_s: f64,
}

impl Kitchen {
    /// A pot starting at `initial_phi` dispersed fraction and `initial_temp_c`,
    /// burner off (ambient), stepping physics in `dt_s`-second increments.
    pub fn new(initial_phi: f64, initial_temp_c: f64, dt_s: f64) -> Self {
        let v_dispersed = initial_phi.clamp(0.0, 0.99);
        let v_water = 1.0 - v_dispersed;
        Self {
            state: KitchenState {
                core_temp_c: initial_temp_c,
                heat_setpoint_c: TEMP_MIN_C,
                v_water,
                v_dispersed,
                ph: 7.0,
                salinity_pct: 0.0,
            },
            dt_s,
        }
    }

    /// Apply a discrete action — a real mutation of conserved state, not a
    /// label. Called once per step before [`Kitchen::step_physics`].
    pub fn apply(&mut self, action: KitchenAction) {
        let s = &mut self.state;
        match action {
            KitchenAction::HeatUp => {
                s.heat_setpoint_c = (s.heat_setpoint_c + HEAT_STEP_C).min(TEMP_MAX_C);
            }
            KitchenAction::HeatDown => {
                s.heat_setpoint_c = (s.heat_setpoint_c - HEAT_STEP_C).max(TEMP_MIN_C);
            }
            KitchenAction::AddWater => {
                s.v_water += ADD_WATER_VOL;
            }
            KitchenAction::AddAcid => {
                s.ph = (s.ph - ACID_STEP).max(PH_MIN);
            }
            KitchenAction::AddSalt => {
                s.salinity_pct = (s.salinity_pct + SALT_STEP_PCT).min(SALINITY_MAX_PCT);
            }
        }
    }

    /// Advance the physics by one `dt_s` step: core temperature relaxes toward
    /// the heat setpoint (Newton's law, reused from Phase 2), and the water
    /// phase evaporates at a rate that ramps in above 60 °C.
    pub fn step_physics(&mut self) {
        // Tuned for tau ≈ 330 s (≈ 11 steps at dt=30s) — a small saucepan, not
        // a swimming pool. The original guessed constants gave tau ≈ 97 min,
        // caught by heating_raises_temp_and_boils_off_water actually failing.
        let nc = NewtonCooling {
            h: 80.0,
            area: 0.08,
            mass: 0.5,
            specific_heat: 4186.0,
            t_env: self.state.heat_setpoint_c,
        };
        self.state.core_temp_c = nc.temperature_at(self.state.core_temp_c, self.dt_s);

        let ramp = ((self.state.core_temp_c - 60.0) / 40.0).clamp(0.0, 1.0);
        let rate = EVAP_RATE_MAX_PER_S * ramp;
        self.state.v_water *= (-rate * self.dt_s).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_index_roundtrips() {
        for (i, a) in KitchenAction::ALL.iter().enumerate() {
            assert_eq!(a.index(), i);
            assert_eq!(KitchenAction::from_index(i), Some(*a));
        }
        assert_eq!(KitchenAction::from_index(99), None);
    }

    #[test]
    fn heating_raises_temp_and_boils_off_water() {
        let mut k = Kitchen::new(0.30, 20.0, 30.0);
        let phi0 = k.state.phi();
        k.apply(KitchenAction::HeatUp);
        k.apply(KitchenAction::HeatUp);
        k.apply(KitchenAction::HeatUp);
        k.apply(KitchenAction::HeatUp);
        for _ in 0..40 {
            k.step_physics();
        }
        assert!(k.state.core_temp_c > 90.0, "temp={}", k.state.core_temp_c);
        assert!(k.state.phi() > phi0, "phi should rise under sustained heat");
    }

    #[test]
    fn adding_water_dilutes_phi() {
        let mut k = Kitchen::new(0.50, 80.0, 30.0);
        let phi0 = k.state.phi();
        k.apply(KitchenAction::AddWater);
        assert!(k.state.phi() < phi0);
    }

    #[test]
    fn acid_and_salt_actions_are_bounded() {
        let mut k = Kitchen::new(0.3, 20.0, 30.0);
        for _ in 0..50 {
            k.apply(KitchenAction::AddAcid);
        }
        assert_eq!(k.state.ph, PH_MIN);
        for _ in 0..50 {
            k.apply(KitchenAction::AddSalt);
        }
        assert_eq!(k.state.salinity_pct, SALINITY_MAX_PCT);
    }

    #[test]
    fn off_burner_relaxes_toward_ambient() {
        let mut k = Kitchen::new(0.3, 95.0, 30.0);
        // heat_setpoint_c starts at TEMP_MIN_C (burner off) by construction.
        for _ in 0..60 {
            k.step_physics();
        }
        assert!(k.state.core_temp_c < 30.0, "temp={}", k.state.core_temp_c);
    }
}
