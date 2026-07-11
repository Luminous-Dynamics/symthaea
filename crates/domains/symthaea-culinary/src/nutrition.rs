//! Item 5 of `CULINARY_PLAN_2026-07-09.md`'s post-plan improvements — real,
//! ground-truthable nutrition science in the same discipline as the rest of
//! this crate, and with the same scope discipline that trimmed the original
//! "chef" pitch down to what's buildable: this validates a recipe's declared
//! nutrient totals against **published population-level thresholds** (an FDA
//! Nutrition Facts Daily Value is exactly that kind of number), the same
//! category as McGee's smoke points or Berryman's random-close-packing limit.
//! It is explicitly **not** personalized diet/health advice for an
//! individual's medical condition — the same category error the original
//! pitch had to be trimmed of once already.
//!
//! **Deliberately out of scope**: a full per-ingredient nutrient database
//! (e.g. USDA FoodData Central) keyed to the ~1,530 Ahn ingredient names. That
//! is a genuine data-acquisition undertaking on the scale of Phase 0's
//! flavor-network dataset, not a natural extension of the validator/kinetics
//! machinery below — which is complete and independently useful without it,
//! the same way the candy/dairy/frying validators operate on a recipe's
//! *declared* quantities rather than an ingredient lookup table. A caller
//! supplies a [`NutrientProfile`] (its own per-serving totals, from any
//! source); this module does not try to compute one from an ingredient list.
//!
//! Three real, falsifiable pieces:
//! 1. **Atwater energy** ([`NutrientProfile::energy_kcal`]) — the actual
//!    formula printed on every nutrition label (4/4/9/7 kcal per gram of
//!    protein/carb/fat/alcohol).
//! 2. **FDA Daily Value limits** ([`crate::validate::validate_nutrition`]) —
//!    sodium and added-sugar Daily Values from the FDA's 2016 Nutrition Facts
//!    label rule, the same "published population-level threshold" category as
//!    this crate's other validators.
//! 3. **Vitamin-C degradation kinetics** (tests below) — reuses
//!    [`crate::reaction::reaction_extent`] and [`crate::reaction::q10`] (the
//!    *same* Arrhenius engine as the Maillard/caramelization model, just a
//!    different activation energy) rather than duplicating it. Following
//!    `reaction.rs`'s own honesty precedent: the falsifiable claim is
//!    *relative* sensitivity (a boiling trajectory destroys far more vitamin C
//!    than a sous-vide one at the same duration), not an absolute percentage
//!    destroyed — that needs an independently fitted pre-exponential this
//!    module does not claim to have.

use serde::{Deserialize, Serialize};

/// A recipe's (or a single serving's) total nutrient content — supplied by the
/// caller from whatever source it has (a recipe's own accounting, a food
/// database, a lab analysis). This module does not look ingredients up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NutrientProfile {
    pub protein_g: f64,
    pub carb_g: f64,
    pub fat_g: f64,
    pub alcohol_g: f64,
    pub sodium_mg: f64,
    pub added_sugar_g: f64,
}

impl NutrientProfile {
    /// Atwater general-factor energy, kcal — the same formula every nutrition
    /// label uses.
    pub fn energy_kcal(&self) -> f64 {
        self.protein_g * crate::thresholds::PROTEIN_KCAL_PER_G
            + self.carb_g * crate::thresholds::CARB_KCAL_PER_G
            + self.fat_g * crate::thresholds::FAT_KCAL_PER_G
            + self.alcohol_g * crate::thresholds::ALCOHOL_KCAL_PER_G
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reaction::{q10, reaction_extent};
    use crate::thermal::ThermalTrajectory;
    use crate::thresholds::VITAMIN_C_DEGRADATION_EA_J_PER_MOL;

    #[test]
    fn atwater_energy_matches_hand_computed_kcal() {
        let p = NutrientProfile {
            protein_g: 10.0,
            carb_g: 20.0,
            fat_g: 5.0,
            ..Default::default()
        };
        // 10*4 + 20*4 + 5*9 = 40 + 80 + 45 = 165.
        assert!(
            (p.energy_kcal() - 165.0).abs() < 1e-9,
            "{}",
            p.energy_kcal()
        );
    }

    #[test]
    fn alcohol_contributes_its_own_real_factor() {
        let wine = NutrientProfile {
            alcohol_g: 12.0,
            ..Default::default()
        };
        // 12 g * 7 kcal/g = 84 kcal — a factor distinct from carb/protein/fat.
        assert!((wine.energy_kcal() - 84.0).abs() < 1e-9);
    }

    #[test]
    fn boiling_destroys_far_more_vitamin_c_than_sous_vide_at_equal_time() {
        let sous_vide = reaction_extent(
            &ThermalTrajectory::hold(60.0, 10.0),
            VITAMIN_C_DEGRADATION_EA_J_PER_MOL,
            1e10,
        );
        let boiling = reaction_extent(
            &ThermalTrajectory::hold(100.0, 10.0),
            VITAMIN_C_DEGRADATION_EA_J_PER_MOL,
            1e10,
        );
        assert!(
            boiling > sous_vide * 10.0,
            "boiling={boiling} sous_vide={sous_vide}"
        );
    }

    #[test]
    fn vitamin_c_q10_is_within_the_common_shelf_life_heuristic_range() {
        // The "shelf life roughly halves/doubles per 10 °C" rule of thumb used
        // broadly in food science corresponds to Q10 in roughly 1.5-4.
        let q = q10(70.0, VITAMIN_C_DEGRADATION_EA_J_PER_MOL);
        assert!((1.5..=4.0).contains(&q), "Q10={q}");
    }
}
