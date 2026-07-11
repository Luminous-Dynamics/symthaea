// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nutrition & metabolism: BMI, BMR, TDEE, macronutrient energy.

/// Body mass index `BMI = kg/m²`.
pub fn bmi(mass_kg: f64, height_m: f64) -> f64 {
    mass_kg / (height_m * height_m)
}

/// Basal metabolic rate (kcal/day), Mifflin–St Jeor equation.
/// `10·kg + 6.25·cm − 5·age + s`, with `s = +5` (male) or `−161` (female).
pub fn bmr_mifflin(mass_kg: f64, height_cm: f64, age_years: f64, is_male: bool) -> f64 {
    let s = if is_male { 5.0 } else { -161.0 };
    10.0 * mass_kg + 6.25 * height_cm - 5.0 * age_years + s
}

/// Total daily energy expenditure `TDEE = BMR · activity_factor`
/// (~1.2 sedentary … ~1.9 very active).
pub fn tdee(bmr: f64, activity_factor: f64) -> f64 {
    bmr * activity_factor
}

/// Energy (kcal) from macronutrient grams: carbs & protein 4 kcal/g, fat 9 kcal/g.
pub fn macronutrient_calories(carbs_g: f64, protein_g: f64, fat_g: f64) -> f64 {
    4.0 * carbs_g + 4.0 * protein_g + 9.0 * fat_g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmi_known() {
        // 70 kg, 1.75 m → 22.857.
        assert!((bmi(70.0, 1.75) - 22.857).abs() < 1e-3);
    }

    #[test]
    fn bmr_mifflin_known() {
        // Male 70 kg, 175 cm, 30 y → 1648.75; female → 1482.75.
        assert!((bmr_mifflin(70.0, 175.0, 30.0, true) - 1648.75).abs() < 1e-6);
        assert!((bmr_mifflin(70.0, 175.0, 30.0, false) - 1482.75).abs() < 1e-6);
    }

    #[test]
    fn macros_known() {
        // 50 g carb + 30 g protein + 20 g fat = 200+120+180 = 500 kcal.
        assert!((macronutrient_calories(50.0, 30.0, 20.0) - 500.0).abs() < 1e-9);
    }

    #[test]
    fn tdee_scales_bmr() {
        let b = bmr_mifflin(70.0, 175.0, 30.0, true);
        assert!((tdee(b, 1.55) - b * 1.55).abs() < 1e-9);
    }
}
